#pragma once

// The template definitions behind the accuracy-multiplier surface of
// boys/boys.hpp: the kernel every templated entry of that header is compiled
// from, shipped so that every value of kAccuracyMultiplier those entries
// document is instantiable at the caller's call site. boys/boys.hpp includes
// this header at its end, after the entries it defines and the tags they name
// are declared, which is the only order in which the definitions compile: it
// is a continuation of that header, not a header a translation unit includes
// on its own.
//
// The library's own instantiations are the default m = 1 set and the sampled
// rungs (boys.cpp / boys_simd.cpp). The extern-template declarations in
// boys/boys.hpp route the default call sites to them; a call site that names
// any other multiplier, or any other evaluation policy, compiles the rung or
// the route it asks for from here.
//
// Every entry is compiled twice under if constexpr: the m = 1 branch is
// today's certified body, which lives in the shared template bodies above and
// is instantiated at the shipped fit (the bit-identity pin - the discarded
// relaxed branch adds no instruction, branch, or load to the m = 1 path), and
// the m > 1 branch evaluates the seed fits at the compile-time effective
// degrees of boys_effective_degrees.hpp.
// The relaxed region-C paths are m-invariant (the asymptotic form has no
// coefficients to truncate); only their scalar tails carry the multiplier.
//
// Everything the caller selects about *how* a value is produced travels as one
// parameter, the evaluation policy (EvalPolicy in backend.hpp): the fit route,
// the scheme its coefficients are summed in and a single-precision engine's
// budget. The engines and the bodies take the policy and read its fields, so
// the surface between a call site and a fit does not grow with the number of
// axes, and the two axes meet in one place (RouteFit, backend.hpp) rather than
// in every signature they would otherwise be threaded through.

/// \cond
// Not API: the kernel the entries are compiled from. The header ships because
// the entries are header-defined; the API reference documents the entries.

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"

#if BoysFp16
#include "boys/f16.hpp"
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace boys {
namespace detail {

constexpr double kBoysHalfSqrtPi = 0.886226925452758014; // sqrt(pi)/2
constexpr float kBoysHalfSqrtPiF32 = 0.88622693f; // sqrt(pi)/2, single lane

// ---------------------------------------------------------------------------
// Scalar double lane helpers (today's anonymous-namespace helpers, verbatim)
// ---------------------------------------------------------------------------
// Piece containing x for this order; the pieces partition [0, kX0).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::OrderPiece& FindPiece(int order, double x) noexcept {
    const int first = detail::kPieceStart[order];
    const int last = detail::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::kPieces[i].b)
        {
            return detail::kPieces[i];
        }
    }

    return detail::kPieces[last - 1];
}

// The even/odd split Clenshaw evaluation of a Chebyshev sum, in the arithmetic
// of backend B and at its width. One body, one arithmetic per instantiation:
// the lanes differ in the width of their values and in nothing else here, and
// the mapped argument t is the caller's, because the lanes map it differently
// on purpose.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
template <backend::ArithmeticBackend B>
typename B::Packed
ClenshawSplit(const typename B::Value* c, int deg, typename B::Packed t) noexcept {
    using V = typename B::Value;
    using P = typename B::Packed;

    if (deg == 0)
    {
        return B::Broadcast(c[0]);
    }

    if (deg == 1)
    {
        return B::MulAdd(t, B::Broadcast(c[1]), B::Broadcast(c[0]));
    }

    const P v = B::MulAdd(B::Broadcast(V{2}), B::Mul(t, t), B::Broadcast(-V{1}));
    const P twoV = B::Add(v, v);

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4.
        return B::MulAdd(
            t, B::Broadcast(c[1]), B::MulAdd(v, B::Broadcast(c[2]), B::Broadcast(c[0])));
    }

    // The odd part below is the Clenshaw finalization for the D family with
    // the top odd coefficient c[2m-1]; it is valid only for even deg >= 4
    // (deg == 2 is handled above). The generator
    // (tools/gen_boys_coefficients.py) only emits such degrees.
    assert(deg >= 4 && deg % 2 == 0);

    // Even part: coefficients c[0], c[2], ..., c[2m].
    const int m = deg / 2;
    P b1 = B::Broadcast(c[std::ptrdiff_t{2} * m]);
    P b2 = B::Broadcast(V{0});

    for (int k = m - 1; k >= 1; --k)
    {
        const P b0 = B::MulAdd(twoV, b1, B::Sub(B::Broadcast(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const P even = B::MulAdd(v, b1, B::Sub(B::Broadcast(c[0]), b2));

    // Odd part: coefficients c[1], c[3], ..., c[2m-1] with the D recurrence.
    P o1 = B::Broadcast(c[2 * m - 1]);
    P o2 = B::Broadcast(V{0});

    for (int k = m - 2; k >= 1; --k)
    {
        const P o0 = B::MulAdd(twoV, o1, B::Sub(B::Broadcast(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const P odd = B::MulAdd(B::Sub(twoV, B::Broadcast(V{1})), o1, B::Sub(B::Broadcast(c[1]), o2));
    return B::MulAdd(t, odd, even);
}

// Horner's rule over the monomial form of a fit, in the arithmetic of backend
// B and at its width: ascending coefficients, one multiply-add per stored
// coefficient, evaluated at the caller's mapped argument t.
//
// The conversion the coefficients came from is benign because t never leaves
// [-1, 1] - the affine map is the fit's own interval - so the monomial
// coefficients of a fit on that interval stay within a small factor of the
// Chebyshev ones however high the degree, and the summation's rounding is the
// same shape as the Chebyshev recurrence's rather than a cancellation problem.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
template <backend::ArithmeticBackend B>
typename B::Packed HornerMono(const typename B::Value* c, int deg, typename B::Packed t) noexcept {
    if (deg == 0)
    {
        return B::Broadcast(c[0]);
    }

    typename B::Packed acc = B::Broadcast(c[deg]);

    for (int k = deg - 1; k >= 0; --k)
    {
        acc = B::MulAdd(acc, t, B::Broadcast(c[k]));
    }

    return acc;
}

// One stored fit, summed by the named scheme. The two coefficient tables are
// parallel - same pieces, same intervals, same degrees, same offsets - so a
// scheme picks a table and a summation and nothing else about the fit changes.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (cheb, mono) reads naturally.
template <EvalScheme kScheme, backend::ArithmeticBackend B>
typename B::Packed FitSum(const typename B::Value* cheb,
                          const typename B::Value* mono,
                          int deg,
                          typename B::Packed t) noexcept {
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return HornerMono<B>(mono, deg, t);
    } else
    {
        return ClenshawSplit<B>(cheb, deg, t);
    }
}

// ---------------------------------------------------------------------------
// The fit families and the one evaluation body over them
// ---------------------------------------------------------------------------
// A region-A route is a coefficient set and the scheme it is read in, named as
// a type (the FitPolicy contract in backend.hpp). A family is written once per
// scheme it holds a stored form for, and the body below is written once and
// instantiated per (route, scheme) pair: the zero argument's closed form, the
// region split, the piece lookup, the mapped argument, the recurrences, the
// per-order rule and the domains are the body's, and a family supplies only
// what is genuinely its own. The coefficients, the degrees and each piece's
// interval are compile-time facts of the generated tables; which piece an
// (order, argument) pair falls in is the one run-time choice there is, so a
// family reads its degree at the index the body hands it rather than carrying
// the degree in its type.
//
// The two axes meet in RouteFit, below: the pair is the fit type a body
// evaluates, and a pair the library does not carry fails there with the reason
// rather than inside a recurrence. The bodies themselves are parameterised on
// the policy (EvalPolicy) and read the axes as fields, so a further axis is a
// field on the policy and touches no body's signature.

// The extended band's per-(n, x) dispatch (the pure-function rule): an
// order n takes the extended seed exactly when x >= kTierThresholds[n],
// regardless of the entry point or the other orders of a batch call. The
// threshold table is n-indexed (structurally like the per-order degree
// tables): each order's entry is the certified boundary of the smallest
// kmax row that covers it (the rows 4/8/16/32), so
// BoysAllOrders(n,x)[n] == BoysSingle(n,x) == BoysAllOrders(m>=n,x)[n] exactly.
// The band is the m = 1 lane's: the m > 1 branch keeps today's
// region-A dispatch in the band (the per-order amplification of the
// band's upward recursion needs its own derivation - named future work,
// not ported here), and the float lane's dispatch is untouched (its
// double-seeded region-A branch already serves the band at the float
// budget).

// The extended-band seed (the per-range seed design): F_0(x) on
// [kExtendedBX0, kX0) via the same split Clenshaw evaluation as the
// region-B seed. It serves the upward recursion below kX0 in the m = 1
// double lanes only, dispatched per (n, x) at the certified per-order
// thresholds kTierThresholds; the m > 1 branch and the float lanes keep
// their existing dispatch untouched.
// The extended-band seed F_0(x) on [kExtendedBX0, kX0), summed by the named
// scheme; see ChebyshevValue for the name.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBExtendedSeed(double x) noexcept {
    const double t = 2.0 * (x - kExtendedBX0) / (kX0 - kExtendedBX0) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(detail::kExtendedBcoeffs.data(),
                                                detail::kMonoExtendedBcoeffs.data(),
                                                detail::kExtendedBDeg,
                                                t);
}

// One step of the upward recursion: F_l(x) from F_{l-1}(x). The band's orders,
// the single-order band entry and region B all ride this step.
inline double UpwardStep(int l, double f, double x, double expx) noexcept {
    return ((l - 0.5) * f - expx) / x;
}

// Region-A F_order(x) in a fit's own scheme, over the piece the shared table
// puts the argument in and at that piece's own mapped argument.
template <FitPolicy Fit>
inline double RegionAValue(int order, double x) noexcept {
    const std::size_t index =
        static_cast<std::size_t>(&FindPiece(order, x) - detail::kPieces.data());
    const detail::OrderPiece& piece = detail::kPieces[index];
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return Fit::EvalPiece(index, t);
}

// The Chebyshev route, at the scheme its coefficients are summed in. The shipped
// family, and the one that holds both stored forms: the Chebyshev table the
// split Clenshaw recurrence reads and the monomial table Horner reads, over the
// same pieces at the same degrees.
template <EvalScheme kScheme>
struct ChebyshevFit {
    // The band's orders are read from the band's seed, so this route's own
    // answer starts at the band's left edge.
    static constexpr double kRegionAFitsFrom = detail::kTierThresholds[0];

    static double EvalPiece(std::size_t index, double t) noexcept {
        const detail::OrderPiece& piece = detail::kPieces[index];
        return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                    detail::kMonoCoeffs.data() + piece.offset,
                                                    piece.deg,
                                                    t);
    }

    static double RegionBSeed(double x) noexcept {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        return FitSum<kScheme, backend::ScalarFp64>(
            detail::kBcoeffs.data(), detail::kMonoBcoeffs.data(), detail::kBDeg, t);
    }

    // The band's orders: one seed at the band's left edge, then one upward step
    // per order. The state is the step's, so a batch pays one division per
    // order rather than one run of the recurrence per order.
    struct BandSource {
        double f;
        double expx;

        explicit BandSource(double x) noexcept
            : f(RegionBExtendedSeed<kScheme>(x)), expx(0.5 * std::exp(-x)) {}

        double Next(int l, double x) noexcept {
            return l == 0 ? f : (f = UpwardStep(l, f, x, expx));
        }
    };
};

// The rational minimax route: a numerator/denominator pair per piece, read by
// Horner in the same mapped argument, the denominator stored as q_1..q_k with
// its constant term held at 1 - one fused multiply-add per coefficient and a
// single division. An alternative to the shipped route rather than a
// replacement: the same pieces, the same intervals, the same mapping, the
// other scheme.
struct RationalFit {
    // Below this argument the order's own piece is what the lane documents, at
    // the per-order 1e-15; at and above it the lane reads the order from the
    // band and documents the band's bound, which is the bar these fits hold.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    static double EvalPiece(std::size_t index, double t) noexcept {
        const double* c = detail::kRatACoeffs.data() + detail::kRatAOffset[index];
        const int m = detail::kRatANumDeg[index];
        const int k = detail::kRatADenDeg[index];
        double num = c[m];

        for (int j = m - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, c[j]);
        }

        if (k == 0)
        {
            return num;
        }

        double den = c[m + k];

        for (int j = k - 1; j >= 1; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, c[m + j]);
        }

        return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
    }

    // The region-B seed from the region's own minimax pair.
    static double RegionBSeed(double x) noexcept {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        double num = detail::kRatBnum[detail::kRatBnumDeg];

        for (int j = detail::kRatBnumDeg - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, detail::kRatBnum[j]);
        }

        double den = detail::kRatBden[detail::kRatBdenDeg - 1];

        for (int j = detail::kRatBdenDeg - 2; j >= 0; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, detail::kRatBden[j]);
        }

        den = backend::ScalarFp64::MulAdd(den, t, 1.0);
        return num / den;
    }

    // The band's orders: this route has no seed to carry up - each of those
    // orders is its own fit - so the source carries no state.
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFit>(l, x); }
    };
};

// Region-A seed F_order(x) of the Chebyshev route at a scheme, for the lanes
// and the reports that read one fit rather than a whole route.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double ChebyshevValue(int order, double x) noexcept {
    return RegionAValue<ChebyshevFit<kScheme>>(order, x);
}

// Region-B seed F_0(x) of the Chebyshev route at a scheme, valid on [kX0, kX1).
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeed(double x) noexcept {
    return ChebyshevFit<kScheme>::RegionBSeed(x);
}

// ---------------------------------------------------------------------------
// Scalar float lane helpers (today's anonymous-namespace helpers, verbatim)
// ---------------------------------------------------------------------------
// Float-lane piece lookup; see FindPiece.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::f32::OrderPiece& FindPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kPieceStart[order];
    const int last = detail::f32::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kPieces[i].b)
        {
            return detail::f32::kPieces[i];
        }
    }

    return detail::f32::kPieces[last - 1];
}

// Float-lane region-A seed; see ChebyshevValue.
// Forced inline: the m = 1 F32-single engine must keep the full-accuracy
// code shape (piece scan inlined); with two call sites (the kFloat and the
// kFp16 budget instantiations) MSVC's size heuristic keeps this helper
// out-of-line otherwise. Semantics are unaffected — inline never changes the
// bit-identity pin.
// __forceinline is MSVC-only; GCC/Clang spell the same intent with
// always_inline (plain inline is a hint there, not a requirement).
#if defined(_MSC_VER)
#define BoysForceInline __forceinline
#else
#define BoysForceInline inline __attribute__((always_inline))
#endif
BoysForceInline float ChebyshevValueF32(int order, float x) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const float* c = detail::f32::kCoeffs.data() + piece.offset;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return ClenshawSplit<backend::ScalarFp32>(c, piece.deg, t);
}

#undef BoysForceInline

// Float-lane region-B seed; see RegionBSeed.
inline float RegionBSeedF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return ClenshawSplit<backend::ScalarFp32>(detail::f32::kBcoeffs.data(), detail::f32::kBDeg, t);
}

// ---------------------------------------------------------------------------
// Degree-parameterized variants (the relaxation path)
// ---------------------------------------------------------------------------
// The coefficient table a scheme's summation reads, which is the table a
// rung's truncation has to be judged against: ClenshawSplit reads the
// Chebyshev table, HornerMono the monomial table, and the two hold the same
// polynomial as different numbers. A degree table is certified against one of
// them (boys_effective_degrees.hpp), so a call site picks the table by the
// scheme its policy names rather than sharing one.
template <EvalScheme kScheme> constexpr detail::TailBasis SchemeTailBasis() noexcept {
    return kScheme == EvalScheme::kHorner ? detail::TailBasis::kMonomial
                                          : detail::TailBasis::kChebyshev;
}

// The region-A seed at the piece's effective degree; the piece index is the
// flat kPieces index (the degrees tables are indexed the same way).
template <EvalScheme kScheme = kDefaultEvalScheme, typename DegreesArray>
double ChebyshevValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    const detail::OrderPiece& piece = FindPiece(order, x);
    const std::ptrdiff_t index = &piece - detail::kPieces.data();
    const int deg = degrees[static_cast<std::size_t>(index)];
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                detail::kMonoCoeffs.data() + piece.offset,
                                                deg,
                                                t);
}

template <typename DegreesArray>
float ChebyshevValueF32WithDegrees(int order, float x, const DegreesArray& degrees) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const std::ptrdiff_t index = &piece - detail::f32::kPieces.data();
    const float* c = detail::f32::kCoeffs.data() + piece.offset;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return ClenshawSplit<backend::ScalarFp32>(c, degrees[static_cast<std::size_t>(index)], t);
}

template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeedWithDegrees(double x, int degree) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(
        detail::kBcoeffs.data(), detail::kMonoBcoeffs.data(), degree, t);
}

inline float RegionBSeedF32WithDegrees(float x, int degree) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return ClenshawSplit<backend::ScalarFp32>(detail::f32::kBcoeffs.data(), degree, t);
}

// ---------------------------------------------------------------------------
// The bodies: one per entry shape, over the policy a call site selected
// ---------------------------------------------------------------------------
// A body takes the policy as its ONE selection parameter and reads the axes as
// fields, so the fit family, the scheme and anything added later reach the
// recurrences without a parameter per axis: the family is Policy::Fit, the
// scheme is Policy::kScheme, and the tail orders an order's own fit does not
// answer are the shipped family's at that scheme.
template <EvalPolicyLike Policy>
void AllOrdersBody(int nmax, double x, double* out) noexcept {
    using Fit = typename Policy::Fit;
    using Shipped = ChebyshevFit<Policy::kScheme>;

    static_assert(FitPolicy<Fit>,
                  "the fit a policy names must satisfy the contract the bodies are written "
                  "against (backend.hpp, FitPolicy)");
    if (x == 0.0)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            out[l] = 1.0 / (2.0 * l + 1.0);
        }

        return;
    }

    if (x < kX0)
    {
        // The pure per-(n, x) dispatch, driven by the n-indexed threshold
        // table: the orders k with x >= kTierThresholds[k] are a prefix (the
        // thresholds are non-decreasing in n) and are read from this route's
        // own band answer; the tail orders keep the shipped piece's value,
        // which is the value the lane documents for them - below its own end an
        // order is documented at the per-order 1e-15, and a route whose fits
        // hold the wider bar does not answer there. On the shipped route out[k]
        // is bit-identical to the single-order entry's value for every k: that
        // entry applies the same per-(n, x) rule to the same fits.
        //
        // The fallback below, taken when x is under the band's left edge and no
        // order takes the band's answer at all, is the one case that is not: it
        // seeds the downward recursion once, at nmax, and pays one fit for the
        // batch where the per-order reading would pay one fit per order. So
        // out[nmax] is still the single-order entry's value for nmax bit for
        // bit, while out[k] for k < nmax carries the recurrence's value rather
        // than the fit's. The two readings of one cell differ by a few units in
        // the last place of the result - the swept worst is 4 ULP and 3.34e-16
        // absolute over region A - and both are inside the bound this entry
        // documents.
        int served = 0;

        while (served < nmax &&
               x >= detail::kTierThresholds[static_cast<std::size_t>(served + 1)])
        {
            ++served;
        }

        if (x >= Fit::kRegionAFitsFrom)
        {
            typename Fit::BandSource source(x);

            for (int l = 0; l <= served; ++l)
            {
                out[l] = source.Next(l, x);
            }

            for (int l = served + 1; l <= nmax; ++l)
            {
                out[l] = RegionAValue<Shipped>(l, x);
            }

            return;
        }

        double f = RegionAValue<Shipped>(nmax, x);
        out[nmax] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            out[l] = f;
        }

        return;
    }

    if (x < kX1)
    {
        double f = Fit::RegionBSeed(x);
        out[0] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = UpwardStep(l, f, x, expx);
            out[l] = f;
        }

        return;
    }

    double f = kBoysHalfSqrtPi / std::sqrt(x);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = (l - 0.5) * f / x;
        out[l] = f;
    }
}

// One order at one argument, in the policy's family and scheme.
template <EvalPolicyLike Policy>
double SingleOrder(int n, double x) noexcept {
    using Fit = typename Policy::Fit;
    using Shipped = ChebyshevFit<Policy::kScheme>;

    static_assert(FitPolicy<Fit>,
                  "the fit a policy names must satisfy the contract the bodies are written "
                  "against (backend.hpp, FitPolicy)");

    if (x == 0.0)
    {
        return 1.0 / (2.0 * n + 1.0);
    }

    if (x < kX0)
    {
        if (x >= detail::kTierThresholds[static_cast<std::size_t>(n)])
        {
            typename Fit::BandSource source(x);
            double f = 0.0;

            for (int l = 0; l <= n; ++l)
            {
                f = source.Next(l, x);
            }

            return f;
        }

        return RegionAValue<Shipped>(n, x);
    }

    if (x < kX1)
    {
        double f = Fit::RegionBSeed(x);
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= n; ++l)
        {
            f = UpwardStep(l, f, x, expx);
        }

        return f;
    }

    double f = kBoysHalfSqrtPi / std::sqrt(x);

    for (int l = 0; l < n; ++l)
    {
        f = (l + 0.5) * f / x;
    }

    return f;
}

// ---------------------------------------------------------------------------
// The lane engines (compiled twice: the m = 1 branch is today's body
// verbatim; the relaxed branch evaluates at the effective degrees)
// ---------------------------------------------------------------------------
// Each engine takes the policy and the multiplier, and nothing else: the fit
// family, the scheme and the single-precision budget arrive as the policy's
// fields, so a further axis does not reopen these signatures.

// The relaxation path's route, and the batch entries' route. A relaxed rung
// truncates the shipped fits to their certified effective degrees and no other
// family carries such a degree table, so every m > 1 branch asks for the
// shipped route here: the combination is rejected where the rung is
// instantiated, with the reason, rather than silently evaluating the shipped
// fits for a caller who named another route.
//
// The many-argument and fixed-order entries ask for it at every rung, and that
// is their own scope rather than the rung's: their region bodies read the
// shipped family's seed and per-order fits directly (BoysAllNBodyRegionB,
// BoysFixedNImpl), so a policy naming the rational route would be answered with
// the shipped fits while reporting the route it asked for. The per-argument
// entries carry the route because their bodies take it from the policy's fit;
// these do not, and a refusal is what says so.
template <EvalPolicyLike Policy>
constexpr void RequireShippedRoute() noexcept
{
    static_assert(Policy::kRoute == kDefaultFitRoute,
                  "a relaxed rung truncates the shipped fits to their certified effective "
                  "degrees, and the rational minimax fits carry no such degree table: the "
                  "multiplier selects a rung of the shipped route only");
}

// The many-argument batch entries' route: the shipped one, at every rung.
// Their region bodies evaluate the shipped seed and per-order fits as their
// own, so another route is refused where the call is named rather than
// answered with the shipped fits under the other route's name.
template <EvalPolicyLike Policy>
constexpr void RequireShippedRouteOnBatchEntry() noexcept
{
    static_assert(Policy::kRoute == kDefaultFitRoute,
                  "the many-argument batch entry evaluates the shipped seed and the shipped "
                  "per-order fits as its own bodies: the rational minimax route is carried on "
                  "the per-argument entries, which take their fit from the policy, and a "
                  "policy naming it here is rejected rather than answered with the shipped "
                  "fits");
}

// The fixed-order vector entry's route: the shipped one, at every rung. Same
// reason and same refusal as the batch entry's.
template <EvalPolicyLike Policy>
constexpr void RequireShippedRouteOnFixedEntry() noexcept
{
    static_assert(Policy::kRoute == kDefaultFitRoute,
                  "the fixed-order entry evaluates the shipped Chebyshev fits: the rational "
                  "route is carried on the all-orders entries, not on this one, so a policy "
                  "naming it is rejected there rather than answered with the other route's "
                  "values");
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
double BoysSingleImpl(int n, double x) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order, so it has one order to put in a vector lane "
                  "and the orders axis is not an axis here: the axis this library carries on "
                  "this shape is the arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return SingleOrder<Policy>(n, x);
    } else
    {
        RequireShippedRoute<Policy>();
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleSingle,
                                                         SchemeTailBasis<Policy::kScheme>()>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleSingle,
                                                         SchemeTailBasis<Policy::kScheme>()>();

        if (x == 0.0)
        {
            return 1.0 / (2.0 * n + 1.0);
        }

        if (x < kX0)
        {
            return ChebyshevValueWithDegrees<Policy::kScheme>(n, x, kDegreesA);
        }

        double f =
            RegionBSeedWithDegrees<Policy::kScheme>(x, kDegreesB[static_cast<std::size_t>(n)]);

        if (x < kX1)
        {
            const double expx = 0.5 * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((l + 0.5) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPi / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (l + 0.5) * f / x;
        }

        return f;
    }
}

// Named by BoysAllOrdersImpl below, so it is declared before it. The call there
// passes a dependent template argument but arguments of fundamental type, so
// neither lookup at the point of definition nor ADL at instantiation finds the
// declaration near the end of this file. Its default argument is set there.
template <EvalScheme kScheme>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersImpl(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if constexpr (Policy::kPack == PackAxis::kOrders)
        {
            static_assert(Policy::kRoute == kDefaultFitRoute,
                          "the across-orders packed lane reads the shipped region-A piece table "
                          "and nothing else: the rational minimax route's fits are not carried "
                          "on that axis, so naming the two together is not a combination this "
                          "library serves");
            BoysAllOrdersPacked<Policy::kScheme>(nmax, x, out);
        } else
        {
            AllOrdersBody<Policy>(nmax, x, out);
        }
    } else
    {
        static_assert(Policy::kRoute == kDefaultFitRoute,
                      "a relaxed rung truncates the shipped fits to their certified effective "
                      "degrees, and the rational minimax fits carry no such degree table: the "
                      "multiplier selects a rung of the shipped route only");
        static_assert(Policy::kPack == PackAxis::kArguments,
                      "the across-orders packed lane evaluates every stored fit at its full "
                      "degree and reads no effective-degree table, so it carries no rung: a "
                      "relaxed multiplier on the orders axis is not a combination this library "
                      "serves");
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleBatch,
                                                         SchemeTailBasis<Policy::kScheme>()>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleBatch,
                                                         SchemeTailBasis<Policy::kScheme>()>();

        if (x == 0.0)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0 / (2.0 * l + 1.0);
            }

            return;
        }

        if (x < kX0)
        {
            double f = ChebyshevValueWithDegrees<Policy::kScheme>(nmax, x, kDegreesA);
            out[nmax] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (l + 0.5);
                out[l] = f;
            }

            return;
        }

        if (x < kX1)
        {
            // The F0 seed's truncation error reaches EVERY output order with
            // gain A_B(l) = prod_{j=0..l-1} (j+1/2)/x0^l, the same quantity as
            // prod_{j=1..l} (2j-1)/(2 x0): the upward step's factor is (2l-1)
            // over 2x, and kX0 sits at the value that makes A_B(32) exactly
            // one. Shipped, it is 1 + 1.846e-17 at l = 32 --- above one by a
            // last-digit excess, and the maximum over l <= 32 --- so the
            // seed's error is not amplified: the order-0 entry is the seed
            // degree, the one whose criterion is
            // Delta(d') * 1 <= (m-1) * B. Written with the product running
            // j = 0..l-1 as above; run from j = 1 it carries one extra factor,
            // (2l+1) = 65 at l = 32, which is not the gain. The per-order
            // entries kDegreesB[n] (gain far below one at small n) bound only
            // the order-n single-style path; seeding with kDegreesB[nmax]
            // leaves F0's error at Delta(d'(nmax)) — up to
            // (m-1)*B/A_B(nmax) — unbounded.
            double f = RegionBSeedWithDegrees<Policy::kScheme>(x, kDegreesB[0]);
            out[0] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((l - 0.5) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        double f = kBoysHalfSqrtPi / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (l - 0.5) * f / x;
            out[l] = f;
        }
    }
}

// The fixed-n vector engine: F_n at every argument of an array, one fixed
// order (the batch shape of angular-momentum-grouped inner loops; the
// strided output layout belongs to the public surface in boys.hpp). The
// region bodies below mirror BoysSingleImpl's verbatim - the m = 1 branch
// is the certified scalar single-lane code (the bit-identity pin), the
// relaxed branch the same bodies at the single-lane effective degrees
// (BoysRole::kDoubleSingle) - so every output element is bit-identical to
// the corresponding BoysSingle call by construction; keep the two engines'
// bodies in lockstep. The region dispatch is per element: mixed-region
// arguments need no pre-partitioning (the portable shape). The
// dispatch-once-per-batch region structure lives in the AVX2 region-sorted
// lanes (boys_simd.cpp), whose callers partition by region first.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysFixedNImpl(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    RequireShippedRouteOnFixedEntry<Policy>();
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "the orders axis cannot be formed on this entry: a packed lane keeps four "
                  "orders of one argument in a register, and this call produces exactly one "
                  "order at every argument of the array, so there are not four orders here to "
                  "fill a lane with - the wide dimension it does have is count, and that is the "
                  "arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x != nullptr);
    assert(out != nullptr);
    assert(stride >= 1);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const double xi = x[i];
            assert(xi >= 0.0);

            if (xi == 0.0)
            {
                out[i * stride] = 1.0 / (2.0 * n + 1.0);
                continue;
            }

            if (xi < kX0)
            {
                if (xi >= detail::kTierThresholds[static_cast<std::size_t>(n)])
                {
                    double f = RegionBExtendedSeed<Policy::kScheme>(xi);
                    const double expx = 0.5 * std::exp(-xi);

                    for (int l = 0; l < n; ++l)
                    {
                        f = ((l + 0.5) * f - expx) / xi;
                    }

                    out[i * stride] = f;
                    continue;
                }

                out[i * stride] = ChebyshevValue<Policy::kScheme>(n, xi);
                continue;
            }

            double f = RegionBSeed<Policy::kScheme>(xi);

            if (xi < kX1)
            {
                const double expx = 0.5 * std::exp(-xi);

                for (int l = 0; l < n; ++l)
                {
                    f = ((l + 0.5) * f - expx) / xi;
                }

                out[i * stride] = f;
                continue;
            }

            f = kBoysHalfSqrtPi / std::sqrt(xi);

            for (int l = 0; l < n; ++l)
            {
                f = (l + 0.5) * f / xi;
            }

            out[i * stride] = f;
        }
    } else
    {
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleSingle,
                                                         SchemeTailBasis<Policy::kScheme>()>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleSingle,
                                                         SchemeTailBasis<Policy::kScheme>()>();

        for (std::size_t i = 0; i < count; ++i)
        {
            const double xi = x[i];
            assert(xi >= 0.0);

            if (xi == 0.0)
            {
                out[i * stride] = 1.0 / (2.0 * n + 1.0);
                continue;
            }

            if (xi < kX0)
            {
                out[i * stride] = ChebyshevValueWithDegrees<Policy::kScheme>(n, xi, kDegreesA);
                continue;
            }

            // The order-n single-lane seed entry mirrors the relaxed single
            // path; the per-order amplification analysis of BoysSingleImpl
            // applies unchanged.
            double f =
                RegionBSeedWithDegrees<Policy::kScheme>(xi, kDegreesB[static_cast<std::size_t>(n)]);

            if (xi < kX1)
            {
                const double expx = 0.5 * std::exp(-xi);

                for (int l = 0; l < n; ++l)
                {
                    f = ((l + 0.5) * f - expx) / xi;
                }

                out[i * stride] = f;
                continue;
            }

            f = kBoysHalfSqrtPi / std::sqrt(xi);

            for (int l = 0; l < n; ++l)
            {
                f = (l + 0.5) * f / xi;
            }

            out[i * stride] = f;
        }
    }
}

// The float lanes' scope (the per-range seed design is fp64-only v1): the
// float dispatch is untouched, still keyed to kX0/kX1, so the carved band
// [kExtendedBX0, kX0) stays EXACTLY today's float path - the per-order
// region-A fits, double-seeded in the batch form (ChebyshevValue's double
// evaluation), serving the band at the float budget. The certified table
// is the double recursion's; the float band is measured, not certified.
//
// The lanes hold the shipped route's single-precision fits and its split
// Clenshaw recurrence, and carry no monomial or rational coefficient set, so
// the policy reaches this engine as its budget alone: another route or another
// scheme is rejected here rather than silently evaluated at this one.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
float BoysSingleF32Impl(int n, float x) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kRoute == kDefaultFitRoute && Policy::kScheme == kDefaultEvalScheme,
                  "the single-precision lanes evaluate the shipped Chebyshev fits by the split "
                  "Clenshaw recurrence: the monomial and rational coefficient sets are the "
                  "double lane's");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0f);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0f)
        {
            return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            return ChebyshevValueF32(n, x);
        }

        float f = RegionBSeedF32(x);

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((static_cast<float>(l) + 0.5f) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (static_cast<float>(l) + 0.5f) * f / x;
        }

        return f;
    } else
    {
        constexpr BoysRole kRole =
            (Policy::kBudget == BoysBudget::kFloat) ? BoysRole::kF32Single : BoysRole::kF32Fp16Single;
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier, kRole>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier, kRole>();

        if (x == 0.0f)
        {
            return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            return ChebyshevValueF32WithDegrees(n, x, kDegreesA);
        }

        float f = RegionBSeedF32WithDegrees(x, kDegreesB[static_cast<std::size_t>(n)]);

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((static_cast<float>(l) + 0.5f) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (static_cast<float>(l) + 0.5f) * f / x;
        }

        return f;
    }
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersF32Impl(int nmax, float x, float* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kRoute == kDefaultFitRoute && Policy::kScheme == kDefaultEvalScheme,
                  "the single-precision lanes evaluate the shipped Chebyshev fits by the split "
                  "Clenshaw recurrence: the monomial and rational coefficient sets are the "
                  "double lane's");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            // The seed must be double precision: the downward recursion
            // amplifies a float seed error by up to x0^n / prod(j+1/2)
            // (5e4 at nmax=8), far beyond the certified 1.5e-7 float budget;
            // one double evaluation per batch is negligible; the recursion
            // itself stays in float.
            const double seed = ChebyshevValue(nmax, static_cast<double>(x));
            out[nmax] = static_cast<float>(seed);
            float f = out[nmax];
            const float expx = 0.5f * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (static_cast<float>(l) + 0.5f);
                out[l] = f;
            }

            return;
        }

        float f = RegionBSeedF32(x);
        out[0] = f;

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((static_cast<float>(l) - 0.5f) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (static_cast<float>(l) - 0.5f) * f / x;
            out[l] = f;
        }
    } else
    {
        constexpr BoysRole kRole =
            (Policy::kBudget == BoysBudget::kFloat) ? BoysRole::kF32Batch : BoysRole::kF32Fp16Batch;
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier, kRole>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier, kRole>();

        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            // The seed must be double precision (see the m = 1 branch); the
            // recursion itself stays in float.
            const double seed = ChebyshevValueWithDegrees(nmax, static_cast<double>(x), kDegreesA);
            out[nmax] = static_cast<float>(seed);
            float f = out[nmax];
            const float expx = 0.5f * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (static_cast<float>(l) + 0.5f);
                out[l] = f;
            }

            return;
        }

        // See the double batch branch: the F0 seed's error reaches every
        // output with gain <= 1 + 1.846e-17 (the ladder's gain at the band
        // edge, one at the last order), so the order-0 entry is
        // the one that bounds the whole batch (kDegreesB[nmax] is loose:
        // the gain is far below 1 at small nmax, and the entry is 0 for many
        // orders at m >= 1e4 — a degree-0 seed error ~5e-2 on F0).
        float f = RegionBSeedF32WithDegrees(x, kDegreesB[0]);
        out[0] = f;

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((static_cast<float>(l) - 0.5f) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (static_cast<float>(l) - 0.5f) * f / x;
            out[l] = f;
        }
    }
}

// ---------------------------------------------------------------------------
// The region kernels (internal: defined in boys_simd.cpp)
// ---------------------------------------------------------------------------
// One region of the argument line per kernel, same order across an array of
// arguments, AVX2 with a scalar tail. They are the vector tier the entries
// dispatch to; they are not the public surface, because a caller reaching them
// directly has to partition its arguments by region itself, which is the work
// the batch entries exist to do.
//
// The certified per-value bounds: A |F̂ − F| ≤ m·1e-15 over [0, kX0), B and C
// ≤ m·5.5e-14. Measured against the committed reference grid (this tree's test
// suite): A holds that bound below the extended band and lands 1.4e-15 at
// n = 32 on the band itself, B lands 2.2e-13 at n = 32, x = kX0 — four times
// its own bound, which is why the batch entry serves region B with the scalar
// body instead — and C lands 5.0e-14, inside its bound with 10% slack.
//
// On a non-x86_64 target these are defined against the certified scalar lanes
// (see the guard in boys_simd.cpp) and BoysAvx2Available() reports false.
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for arguments in region B; out[order * count + i].
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept;

template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
// The fp16/bf16 lanes: the same kernels on half-precision I/O types, per order.
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

extern template void BoysRegionASimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionBSimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionCSimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
extern template void BoysRegionASimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionBSimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionCSimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionASimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
extern template void BoysRegionBSimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
extern template void BoysRegionCSimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

// ---------------------------------------------------------------------------
// The across-orders packed lane (internal: defined in boys_orders_simd.cpp)
// ---------------------------------------------------------------------------
// The companion of the region kernels above: the same region-A stored fits,
// evaluated four ORDERS to a vector instead of four arguments, which is the
// axis BoysAllOrders(nmax, x, out) actually has. It is the orders axis of
// PackAxis, and the entries that carry it dispatch here.
//
// It evaluates each order's own fit and reaches no order by a recursion, so its
// values are the per-order fits' values: at the certified split Clenshaw scheme
// they are the across-arguments lane's values bit for bit, and that identity is
// asserted by the suite. The coefficients are fetched composed - four loads and
// the shuffles that join them - rather than with one gather instruction, which
// is the same lane by construction and cheaper in retired slots on the machines
// measured; the gathered fetch is kept beside it so the pair stays measurable.
//
// Below kX0 only. Past it the entry runs the certified scalar single lane one
// order at a time, which is a defined answer inside the entry's own bound
// rather than the packed lane.
template <EvalScheme kScheme = kDefaultEvalScheme>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

extern template void BoysAllOrdersPacked<kDefaultEvalScheme>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner>(
    int nmax, double x, double* out) noexcept;

// ---------------------------------------------------------------------------
// The all-orders batch over an argument array (BoysAllN)
// ---------------------------------------------------------------------------
// Every dispatch path is an interval of the argument line and the intervals are
// ordered, so the classification is monotone in x: a non-decreasing array is
// already contiguous by path, which is what the BoysSortedArgs overload
// declares and why one comparison per argument is the honest statement of it.
//
// A path is served by one of two kernel shapes. The ungrouped shape walks one
// argument at a time and writes the caller's planes; its bodies are the
// per-argument entry's own (BoysAllOrdersImpl) called at the batch's nmax,
// restructured onto the caller's layout and kept in lockstep with it, so at
// m = 1 the batch values ARE that entry's values, bit for bit, on every path it
// serves. (Region A's body seeds at nmax and recurses down, so a batch's F_k for
// k < nmax is that recurrence's value, not the one a per-order call at nmax = k
// walks; both are inside the entry's bound.) The grouped shape runs the
// region-A lane over a homogeneous run, one order at a time (the lane's shape is
// per order), staged through a fixed stack frame whatever count is.
//
// Which shape serves region A is decided by measurement, because the lane pays
// for every order separately: it evaluates each order from that order's own
// fit, while the scalar body evaluates one seed and recurses down. On a
// 65536-argument run, entry against a plain per-argument loop over the same
// arguments, the lane is 2.6x faster serving F_0, at parity at n = 4, and 10 to
// 20% slower from n = 8 up. kBoysAllNLaneMaxOrder is where the two cross.
//
// The zero path and region C run scalar in both shapes: the asymptotic form is
// a seed and nmax recursion steps written into the caller's planes, which is
// less work than the per-order lane, and the scalar body is the per-argument
// path's own.
constexpr std::size_t kBoysAllNChunk = 128;
constexpr int kBoysAllNLaneMaxOrder = 4;

enum class BoysPath : std::uint8_t {
    kZero = 0,
    kTabulated = 1,
    kExtended = 2,
    kMiddle = 3,
    kAsymptotic = 4,
};

constexpr int kBoysPathCount = 5;

// The path of one argument, in the per-argument entry's dispatch order. The
// region-A band split arrives as the caller's tierSplit: at m = 1 the scalar
// dispatch splits the band at the first extended-seed tier and that split is a
// path of its own, at m > 1 the band does not exist and the whole region is one
// path.
inline BoysPath BoysClassifyPath(double x, double tierSplit) noexcept {
    if (x == 0.0)
    {
        return BoysPath::kZero;
    }

    if (x < tierSplit)
    {
        return BoysPath::kTabulated;
    }

    if (x < kX0)
    {
        return BoysPath::kExtended;
    }

    if (x < kX1)
    {
        return BoysPath::kMiddle;
    }

    return BoysPath::kAsymptotic;
}

// The band split of the multiplier in use: the classification has to reproduce
// the per-argument entry's dispatch, band and all, and to stay a split of the
// argument line whatever the multiplier is.
template <double kAccuracyMultiplier> constexpr double BoysAllNTierSplit() noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return kTierThresholds[0];
    } else
    {
        return kX0;
    }
}

// out[order * count + i] = values[t], for the run's t-th argument. The
// unsorted entry's runs carry their permutation; the sorted overload's runs are
// the caller's own order, so its plane writes are sequential.
inline void BoysAllNScatter(int order,
                            std::size_t count,
                            const std::size_t* index,
                            std::size_t begin,
                            std::size_t block,
                            const double* values,
                            double* out) noexcept {
    double* plane = out + static_cast<std::size_t>(order) * count;

    if (index == nullptr)
    {
        for (std::size_t t = 0; t < block; ++t)
        {
            plane[begin + t] = values[t];
        }

        return;
    }

    for (std::size_t t = 0; t < block; ++t)
    {
        plane[index[begin + t]] = values[t];
    }
}

// One argument's column of the caller's planes: plane[k * count] = F_k(x) for
// k = 0..nmax. These bodies are the per-argument entry's, restructured onto the
// caller's layout; keep the two in lockstep.
inline void BoysAllNBodyZero(int nmax, std::size_t count, double* plane) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * count] = 1.0 / (2.0 * l + 1.0);
    }
}

// Region A below the band: the downward recursion from the per-order fit.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
inline void BoysAllNBodyRegionADown(int nmax, double x, std::size_t count, double* plane) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        double f = ChebyshevValue<Policy::kScheme>(nmax, x);
        plane[static_cast<std::size_t>(nmax) * count] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    } else
    {
        RequireShippedRoute<Policy>();
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleBatch,
                                                         SchemeTailBasis<Policy::kScheme>()>();
        double f = ChebyshevValueWithDegrees<Policy::kScheme>(nmax, x, kDegreesA);
        plane[static_cast<std::size_t>(nmax) * count] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    }
}

// Region A's extended band: the per-range seed + upward recursion, with the
// per-order fits for the orders the tier does not reach. The classification
// admits only x >= kTierThresholds[0] here, which is the per-argument entry's
// guard on that condition - the band is this path.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
inline void BoysAllNBodyRegionAExtended(int nmax,
                                        double x,
                                        std::size_t count,
                                        double* plane) noexcept {
    static_assert(kAccuracyMultiplier == 1.0,
                  "the extended band is a path of the m = 1 dispatch only");

    int served = 0;

    while (served < nmax && x >= kTierThresholds[static_cast<std::size_t>(served + 1)])
    {
        ++served;
    }

    double f = RegionBExtendedSeed<Policy::kScheme>(x);
    plane[0] = f;
    const double expx = 0.5 * std::exp(-x);

    for (int l = 1; l <= served; ++l)
    {
        f = ((l - 0.5) * f - expx) / x;
        plane[static_cast<std::size_t>(l) * count] = f;
    }

    for (int l = served + 1; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * count] = ChebyshevValue<Policy::kScheme>(l, x);
    }
}

// Region B: the F0 seed + upward recursion.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
inline void BoysAllNBodyRegionB(int nmax, double x, std::size_t count, double* plane) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        double f = RegionBSeed<Policy::kScheme>(x);
        plane[0] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((l - 0.5) * f - expx) / x;
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    } else
    {
        // The seed degree must be the order-0 entry: the seed's truncation
        // error reaches every output order with amplification A_B(l), at most
        // 1 + 1.846e-17 over the supported orders (its maximum at l = 32).
        RequireShippedRoute<Policy>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier,
                                                         BoysRole::kDoubleBatch,
                                                         SchemeTailBasis<Policy::kScheme>()>();
        double f = RegionBSeedWithDegrees<Policy::kScheme>(x, kDegreesB[0]);
        plane[0] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((l - 0.5) * f - expx) / x;
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    }
}

// Region C: the asymptotic form (m-invariant).
inline void BoysAllNBodyRegionC(int nmax, double x, std::size_t count, double* plane) noexcept {
    double f = kBoysHalfSqrtPi / std::sqrt(x);
    plane[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = (l - 0.5) * f / x;
        plane[static_cast<std::size_t>(l) * count] = f;
    }
}

// The ungrouped kernel: one argument at a time over the run, the path's body
// per argument, the caller's planes.
template <double kAccuracyMultiplier, BoysPath kPath, EvalPolicyLike Policy>
void BoysAllNRunUngrouped(int nmax,
                          const double* x,
                          double* out,
                          std::size_t count,
                          const std::size_t* index,
                          std::size_t begin,
                          std::size_t end) noexcept {
    for (std::size_t j = begin; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;
        const double xi = x[i];
        double* plane = out + i;

        if constexpr (kPath == BoysPath::kZero)
        {
            BoysAllNBodyZero(nmax, count, plane);
        }
        else if constexpr (kPath == BoysPath::kTabulated)
        {
            BoysAllNBodyRegionADown<kAccuracyMultiplier, Policy>(nmax, xi, count, plane);
        }
        else if constexpr (kPath == BoysPath::kExtended)
        {
            BoysAllNBodyRegionAExtended<kAccuracyMultiplier, Policy>(nmax, xi, count, plane);
        }
        else if constexpr (kPath == BoysPath::kMiddle)
        {
            BoysAllNBodyRegionB<kAccuracyMultiplier, Policy>(nmax, xi, count, plane);
        }
        else
        {
            BoysAllNBodyRegionC(nmax, xi, count, plane);
        }
    }
}

// The grouped kernel: the region-A lane over a homogeneous run, chunked so the
// staging planes stay inside a fixed stack frame. Both region-A paths take this
// shape when they take it at all - the lane covers [0, kX0), so which of the
// two scalar bodies the sort's path split assigned an argument to does not
// choose the kernel; the lane serves the band as well, at its band accuracy.
//
// Region B keeps its scalar body. BoysRegionBSimd does not hold its own
// documented bound across the region - at n = 32, x = kX0 it lands 2.2e-13 from
// the reference against its 5.5e-14 - and the scalar body is the per-argument
// path's own, exact to the bit.
inline void BoysAllNRunGrouped(int nmax,
                               const double* x,
                               double* out,
                               std::size_t count,
                               const std::size_t* index,
                               std::size_t begin,
                               std::size_t end) noexcept {
    std::array<double, (kMaxBoysOrder + 1) * kBoysAllNChunk> stage;
    std::array<double, kBoysAllNChunk> args;

    for (std::size_t base = begin; base < end; base += kBoysAllNChunk)
    {
        const std::size_t block = std::min(kBoysAllNChunk, end - base);

        for (std::size_t t = 0; t < block; ++t)
        {
            args[t] = x[(index != nullptr) ? index[base + t] : base + t];
        }

        // The region-A lane is per order, so the entry asks it for one order at
        // a time, which is the shape the region-A fits have.
        for (int l = 0; l <= nmax; ++l)
        {
            BoysRegionASimd(l, args.data(), stage.data(), block);
            BoysAllNScatter(l, count, index, base, block, stage.data(), out);
        }
    }
}

// One run of one path, by the shape the tier can serve.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNRun(int nmax,
                 const double* x,
                 const std::size_t* index,
                 double* out,
                 std::size_t count,
                 bool vectorTier,
                 BoysPath path,
                 std::size_t begin,
                 std::size_t end) noexcept {
    // The packed region-A lane holds the shipped route's Chebyshev
    // coefficients and its split Clenshaw recurrence, so another route or
    // another scheme takes the scalar body here rather than the lane's values.
    //
    // BoysRegionASimd packs four ARGUMENTS of one order, so the grouped kernel
    // is the arguments axis's and only the arguments axis's: a call naming the
    // orders axis is served one argument at a time by the entry's per-argument
    // path, and its lane packs four orders of one argument instead. The
    // condition is stated rather than inherited from that path, because which
    // axis a packed lane fills is the property this kernel implements.
    if constexpr (kAccuracyMultiplier == 1.0 && Policy::kRoute == kDefaultFitRoute &&
                  Policy::kScheme == EvalScheme::kSplitClenshaw &&
                  Policy::kPack == PackAxis::kArguments)
    {
        // The lane serves region A below the crossover order; at and above it
        // the scalar body is the faster of the two.
        if (vectorTier && nmax <= kBoysAllNLaneMaxOrder &&
            (path == BoysPath::kTabulated || path == BoysPath::kExtended))
        {
            BoysAllNRunGrouped(nmax, x, out, count, index, begin, end);
            return;
        }
    }

    switch (path)
    {
    case BoysPath::kZero:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kZero, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kTabulated:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kTabulated, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kExtended:
        if constexpr (kAccuracyMultiplier == 1.0)
        {
            BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kExtended, Policy>(
                nmax, x, out, count, index, begin, end);
        }
        break;

    case BoysPath::kMiddle:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kMiddle, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kAsymptotic:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kAsymptotic, Policy>(
            nmax, x, out, count, index, begin, end);
        break;
    }
}

// The per-argument path, in the caller's planes - the shape the C entry point's
// batch loop already has, and the batch entry's total fallback.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNRunPerArgument(int nmax,
                            const double* x,
                            double* out,
                            std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<kAccuracyMultiplier, Policy>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
    }
}

// The plane entry's arguments-axis shape: the dispatch read off the argument
// line, the sort over it, and one kernel per homogeneous run. Declared before
// the two entry wrappers that dispatch to it.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNPartitionedImpl(int nmax,
                             const double* x,
                             double* out,
                             std::size_t count,
                             std::size_t* workspace) noexcept;

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNSortedPartitionedImpl(int nmax,
                                   const double* x,
                                   double* out,
                                   std::size_t count) noexcept;

// The plane entry. Two shapes serve it, and which one a call takes is the
// packing axis it names, because the axis is what says what a vector lane is
// filled with:
//
//   the arguments axis  an arguments-axis lane keeps four arguments of one
//                       order in a register, so the entry partitions its
//                       arguments by the interval their answer comes from and
//                       hands each homogeneous run to the shape that serves it.
//                       That is BoysAllNPartitionedImpl below, and it is the
//                       shipped shape.
//   the orders axis     an orders-axis lane keeps four orders of ONE argument
//                       in a register, which is this entry's call shape one
//                       argument at a time - out[k * count + i] is F_k(x[i]),
//                       so an argument's whole order vector is already what the
//                       entry writes. There is nothing for the region grouping
//                       to group, and the body is the all-orders entry's own,
//                       which is where the packed orders lane is dispatched.
//
// The orders axis is carried here at the reference multiplier only, and the
// refusal above it is the engine's own, stated once where the lane is
// dispatched (BoysAllOrdersImpl): the packed orders lane evaluates every
// stored fit at its full degree and reads no effective-degree table, so it
// carries no rung. Carrying the axis on this entry does not change that, and
// this entry does not restate it.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNImpl(int nmax,
                  const double* x,
                  double* out,
                  std::size_t count,
                  std::size_t* workspace) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    if constexpr (Policy::kPack == PackAxis::kOrders)
    {
        // The grouping scratch is the arguments axis's: this path partitions
        // nothing and takes none of it.
        static_cast<void>(workspace);
        BoysAllNRunPerArgument<kAccuracyMultiplier, Policy>(nmax, x, out, count);
    }
    else
    {
        BoysAllNPartitionedImpl<kAccuracyMultiplier, Policy>(nmax, x, out, count, workspace);
    }
}

// The plane entry's arguments-axis shape: the dispatch read off the argument
// line, the sort over it, and one kernel per homogeneous run.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNPartitionedImpl(int nmax,
                             const double* x,
                             double* out,
                             std::size_t count,
                             std::size_t* workspace) noexcept {
    RequireShippedRouteOnBatchEntry<Policy>();
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this is the plane entry's arguments-axis shape: it partitions its arguments "
                  "by region and dispatches each group to a lane that packs four arguments. A "
                  "call naming the orders axis is served by BoysAllNImpl's per-argument path, "
                  "whose lane packs four orders of one argument");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    const bool vectorTier = (kAccuracyMultiplier == 1.0) && BoysAvx2Available();
    const double tierSplit = BoysAllNTierSplit<kAccuracyMultiplier>();

    std::size_t* owned = nullptr;

    if (workspace == nullptr)
    {
        owned = new (std::nothrow) std::size_t[count];
        workspace = owned;
    }

    if (workspace == nullptr)
    {
        // Total by construction: the per-argument path, the same values at the
        // same bound, without the grouping.
        BoysAllNRunPerArgument<kAccuracyMultiplier, Policy>(nmax, x, out, count);
        return;
    }

    // The counting sort over the dispatch paths. The paths are ordered
    // intervals, so this both groups a shuffled array and leaves a non-decreasing
    // one in place - there is no separate already-grouped case to detect.
    std::size_t counts[kBoysPathCount] = {};

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0);
        ++counts[static_cast<std::size_t>(BoysClassifyPath(x[i], tierSplit))];
    }

    std::size_t starts[kBoysPathCount + 1] = {};
    std::size_t cursor[kBoysPathCount] = {};

    for (int p = 0; p < kBoysPathCount; ++p)
    {
        starts[static_cast<std::size_t>(p) + 1] =
            starts[static_cast<std::size_t>(p)] + counts[static_cast<std::size_t>(p)];
        cursor[static_cast<std::size_t>(p)] = starts[static_cast<std::size_t>(p)];
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        const BoysPath path = BoysClassifyPath(x[i], tierSplit);
        workspace[cursor[static_cast<std::size_t>(path)]++] = i;
    }

    for (int p = 0; p < kBoysPathCount; ++p)
    {
        const std::size_t from = starts[static_cast<std::size_t>(p)];
        const std::size_t to = starts[static_cast<std::size_t>(p) + 1];

        if (from != to)
        {
            BoysAllNRun<kAccuracyMultiplier, Policy>(
                nmax, x, workspace, out, count, vectorTier, static_cast<BoysPath>(p), from, to);
        }
    }

    delete[] owned;
}

// The sorted overload, split the same way and for the same reason: the
// ordering the caller declared is what makes the arguments-axis grouping free,
// so it is a fact about that shape's input and not about this entry. A call
// naming the orders axis takes the per-argument path and neither reads nor
// needs the ordering, which is why the overload accepts the axis without
// asking the caller for anything it does not already promise.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNSortedImpl(int nmax,
                        const double* x,
                        double* out,
                        std::size_t count) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    if constexpr (Policy::kPack == PackAxis::kOrders)
    {
        BoysAllNRunPerArgument<kAccuracyMultiplier, Policy>(nmax, x, out, count);
    }
    else
    {
        BoysAllNSortedPartitionedImpl<kAccuracyMultiplier, Policy>(nmax, x, out, count);
    }
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNSortedPartitionedImpl(int nmax,
                                   const double* x,
                                   double* out,
                                   std::size_t count) noexcept {
    RequireShippedRouteOnBatchEntry<Policy>();
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this is the sorted overload's arguments-axis shape: the ordering the caller "
                  "declared is what makes its grouping free. A call naming the orders axis is "
                  "served by BoysAllNSortedImpl's per-argument path, which needs no grouping");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    const bool vectorTier = (kAccuracyMultiplier == 1.0) && BoysAvx2Available();
    const double tierSplit = BoysAllNTierSplit<kAccuracyMultiplier>();

    assert(x[0] >= 0.0);

    // The runs are found by the classification itself rather than declared, so
    // every run is homogeneous whatever the array is: a caller that declared an
    // order it did not have pays the run boundaries, not a wrong value.
    std::size_t begin = 0;

    while (begin < count)
    {
        const BoysPath path = BoysClassifyPath(x[begin], tierSplit);
        std::size_t end = begin + 1;

        while (end < count && BoysClassifyPath(x[end], tierSplit) == path)
        {
            assert(x[end] >= x[end - 1]); // the BoysSortedArgs declaration
            ++end;
        }

        BoysAllNRun<kAccuracyMultiplier, Policy>(nmax, x, nullptr, out, count, vectorTier, path, begin, end);
        begin = end;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// The public template entries (declared in boys/boys.hpp)
// ---------------------------------------------------------------------------

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
double BoysSingle(int n, double x) noexcept {
    return detail::BoysSingleImpl<kAccuracyMultiplier, Policy>(n, x);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrders(int nmax, double x, double* out) noexcept {
    detail::BoysAllOrdersImpl<kAccuracyMultiplier, Policy>(nmax, x, out);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    detail::BoysFixedNImpl<kAccuracyMultiplier, Policy>(n, x, out, count, stride);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace) noexcept {
    detail::BoysAllNImpl<kAccuracyMultiplier, Policy>(nmax, x, out, count, workspace);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllN(int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept {
    detail::BoysAllNSortedImpl<kAccuracyMultiplier, Policy>(nmax, x, out, count);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
float BoysSingleF32(int n, float x) noexcept {
    return detail::BoysSingleF32Impl<kAccuracyMultiplier, Policy>(n, x);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept {
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier, Policy>(nmax, x, out);
}

#if BoysFp16
// The fp16/bf16 lanes forward the multiplier to the F32 engine with the
// fp16 computation budget (the m*1e-7 + 1/2-ULP formula); at m = 1 the
// engine branch is the certified F32 path verbatim, so the lanes are
// bit-unchanged. The half-ULP representation term is m-independent.

template <double kAccuracyMultiplier> F16 BoysSingleF16(int n, F16 x) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    return static_cast<F16>(
        detail::BoysSingleF32Impl<kAccuracyMultiplier,
                              EvalPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>>(
            n, static_cast<float>(x)));
}

template <double kAccuracyMultiplier> void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier,
                           EvalPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>>(
        nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<F16>(scratch[l]);
    }
}

template <double kAccuracyMultiplier> Bf16 BoysSingleBf16(int n, Bf16 x) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    return static_cast<Bf16>(
        detail::BoysSingleF32Impl<kAccuracyMultiplier,
                              EvalPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>>(
            n, static_cast<float>(x)));
}

template <double kAccuracyMultiplier> void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier,
                           EvalPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>>(
        nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<Bf16>(scratch[l]);
    }
}
#endif // BoysFp16

} // namespace boys

/// \endcond
