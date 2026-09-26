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

// ---------------------------------------------------------------------------
// The two stored partitions of region A
// ---------------------------------------------------------------------------
// A fit reads one of two stored partitions of the same region, and what it asks
// of one is the same on both: which piece the argument falls in, and what that
// piece's interval is. So the partitions differ in the tables they name and in
// nothing the body can see, and the body asks the fit for its partition rather
// than the fit carrying a lookup of its own.
//
// The shipped partition is the per-order piecewise table the lane has always
// read. The narrow one is the second partition of the same region, derived at a
// lower degree per piece and more pieces: the same values at fewer coefficients
// per evaluation, against more stored rows and a longer scan to the piece.
struct ShippedRegionAPartition {
    static std::size_t PieceIndex(int order, double x) noexcept {
        return static_cast<std::size_t>(&FindPiece(order, x) - detail::kPieces.data());
    }

    static const detail::OrderPiece& PieceAt(std::size_t index) noexcept {
        return detail::kPieces[index];
    }
};

// The narrow partition's piece containing x for this order; see FindPiece, whose
// scan this is over the second partition's rows for the same order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::OrderPiece& FindNarrowAPiece(int order, double x) noexcept {
    const int first = detail::kNarrowAPieceStart[order];
    const int last = detail::kNarrowAPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::kNarrowAPieces[i].b)
        {
            return detail::kNarrowAPieces[i];
        }
    }

    return detail::kNarrowAPieces[last - 1];
}

struct NarrowRegionAPartition {
    static std::size_t PieceIndex(int order, double x) noexcept {
        return static_cast<std::size_t>(&FindNarrowAPiece(order, x)
                                        - detail::kNarrowAPieces.data());
    }

    static const detail::OrderPiece& PieceAt(std::size_t index) noexcept {
        return detail::kNarrowAPieces[index];
    }
};

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

// Region-A F_order(x) in a fit's own scheme, over the piece the fit's own
// partition of the region puts the argument in and at that piece's own mapped
// argument.
template <FitPolicy Fit>
inline double RegionAValue(int order, double x) noexcept {
    using Partition = typename Fit::Partition;
    const std::size_t index = Partition::PieceIndex(order, x);
    const detail::OrderPiece& piece = Partition::PieceAt(index);
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return Fit::EvalPiece(index, t);
}

// Which piece of the narrow partition an argument falls in. The edges are the
// derived partition's own, non-decreasing, and the last is kX1, so the scan
// ends inside the table for any argument region B dispatches here.
inline int NarrowBPieceOf(double x) noexcept {
    int piece = 0;

    while (piece + 1 < detail::kNarrowBPieces && x >= detail::kNarrowBEdges[static_cast<std::size_t>(piece + 1)])
    {
        ++piece;
    }

    return piece;
}

// Region B's seed from the narrow partition: the piece the argument falls in,
// at that piece's own mapped argument. The pieces tile the interval the shipped
// seed serves, so this answers the same domain at fewer coefficients per
// evaluation and more table rows.
template <EvalScheme kScheme>
inline double NarrowRegionBSeed(double x) noexcept {
    const int piece = NarrowBPieceOf(x);
    const std::size_t index = static_cast<std::size_t>(piece);
    const double a = detail::kNarrowBEdges[index];
    const double b = detail::kNarrowBEdges[index + 1];
    const double t = 2.0 * (x - a) / (b - a) - 1.0;
    const std::size_t offset =
        index * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);

    return FitSum<kScheme, backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                                detail::kNarrowBMonoCoeffs.data() + offset,
                                                detail::kNarrowBDeg,
                                                t);
}

// The Chebyshev route, at the scheme its coefficients are summed in. The shipped
// family, and the one that holds both stored forms: the Chebyshev table the
// split Clenshaw recurrence reads and the monomial table Horner reads, over the
// same pieces at the same degrees.
//
// It holds both partitions of both regions as well, which is what the
// granularity axis selects: region A's pieces and region B's seed are read from
// the shipped tables or from the narrow ones by the granularity. The two
// partitions answer the same domain, so everything around them - the recurrences,
// the region split, the per-order rule - is the same code either way. See
// FitGranularity for what the axis is and is not.
template <EvalScheme kScheme, FitGranularity kGranularity>
struct ChebyshevFit {
    // The region-A partition this fit reads; the bodies ask for it rather than
    // for the table, so a route over the fit does not have to know which.
    using Partition = std::conditional_t<kGranularity == FitGranularity::kShipped,
                                         ShippedRegionAPartition,
                                         NarrowRegionAPartition>;

    // The band's orders are read from the band's seed, so this route's own
    // answer starts at the band's left edge.
    static constexpr double kRegionAFitsFrom = detail::kTierThresholds[0];

    static double EvalPiece(std::size_t index, double t) noexcept {
        if constexpr (kGranularity == FitGranularity::kShipped)
        {
            const detail::OrderPiece& piece = detail::kPieces[index];
            return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                        detail::kMonoCoeffs.data() + piece.offset,
                                                        piece.deg,
                                                        t);
        } else
        {
            const detail::OrderPiece& piece = detail::kNarrowAPieces[index];
            return FitSum<kScheme, backend::ScalarFp64>(
                detail::kNarrowACoeffs.data() + piece.offset,
                detail::kNarrowAMonoCoeffs.data() + piece.offset,
                piece.deg,
                t);
        }
    }

    static double RegionBSeed(double x) noexcept {
        if constexpr (kGranularity == FitGranularity::kShipped)
        {
            const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
            return FitSum<kScheme, backend::ScalarFp64>(
                detail::kBcoeffs.data(), detail::kMonoBcoeffs.data(), detail::kBDeg, t);
        } else
        {
            return NarrowRegionBSeed<kScheme>(x);
        }
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
    // The shipped partition. This route's pieces are fitted over the shipped
    // pieces' own intervals - one minimax pair per piece - so there is no
    // second partition for it to read, and a policy naming both is refused
    // where it is named (see RouteFit).
    using Partition = ShippedRegionAPartition;

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

// One rational piece at a cut pair, in the mapped argument the shipped
// evaluation reads it in. The cut keeps the low-order terms of both parts: the
// numerator's p_0..p_{numDeg} and the denominator's q_1..q_{denDeg}. The
// denominator's coefficients sit above the *stored* numerator, so the position
// of q_j is the piece's full numerator degree's, not the cut's.
inline double RationalPieceAtCut(std::size_t index, int numDeg, int denDeg, double t) noexcept {
    const double* c = detail::kRatACoeffs.data() + detail::kRatAOffset[index];
    const int storedNumDeg = detail::kRatANumDeg[index];
    double num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    double den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
}

// The region-B seed at a cut pair; same reading as RationalFit::RegionBSeed.
inline double RationalSeedAtCut(int numDeg, int denDeg, double t) noexcept {
    double num = detail::kRatBnum[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, t, detail::kRatBnum[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    double den = detail::kRatBden[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, t, detail::kRatBden[j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
}

// The rational route at a relaxed rung: the same pieces, the same seed and the
// same reading as RationalFit, at the pair each of them was certified for at
// this multiplier. A rung of the rational route is a rung of its own pairs
// rather than of the shipped fits, so it is carried by the route's own body:
// the polynomial rung's shape - seed at the top order, then the batch downward
// recursion - is the one this family cannot take, because its pieces are fitted
// for the values alone.
template <double kAccuracyMultiplier>
struct RationalFitAtRung {
    // The pair at each region-A piece and at the region-B seed, from the
    // criterion in boys_effective_degrees.hpp.
    static constexpr detail::RationalRegionAPairs kPairsA =
        detail::RationalRegionADegrees<kAccuracyMultiplier>();
    static constexpr detail::RationalRegionBPairs kPairsB =
        detail::RationalRegionBDegrees<kAccuracyMultiplier>();

    // The shipped partition, as the uncut rational route reads it; see
    // RationalFit.
    using Partition = ShippedRegionAPartition;

    // The route hands each order over at its own region-A boundary, as the
    // uncut rational family does: below it the shipped lane's value is the one
    // that order is documented at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    static double EvalPiece(std::size_t index, double t) noexcept {
        return RationalPieceAtCut(index, kPairsA.num[index], kPairsA.den[index], t);
    }

    static double RegionBSeed(double x) noexcept {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        return RationalSeedAtCut(kPairsB.num[0], kPairsB.den[0], t);
    }

    // Each of the band's orders is its own fit, as it is on the uncut route.
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFitAtRung>(l, x); }
    };
};

// Region-A seed F_order(x) of the Chebyshev route at a scheme, for the lanes
// and the reports that read one fit rather than a whole route.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double ChebyshevValue(int order, double x) noexcept {
    return RegionAValue<ChebyshevFit<kScheme, kDefaultFitGranularity>>(order, x);
}

// Region-B seed F_0(x) of the Chebyshev route at a scheme, valid on [kX0, kX1).
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeed(double x) noexcept {
    return ChebyshevFit<kScheme, kDefaultFitGranularity>::RegionBSeed(x);
}

// Region B's seed at the partition a policy names. The bodies that call this
// read the Chebyshev family's seed by construction - they are region B's own
// bodies, and the route's fit is read where the policy's fit is read - so this
// is the Chebyshev seed at the policy's scheme and granularity and it does not
// consult the route. It exists so that a region-B seed read from a policy is
// the partition that policy names rather than always the shipped one.
template <EvalPolicyLike Policy>
inline double PolicyRegionBSeed(double x) noexcept {
    return ChebyshevFit<Policy::kScheme, Policy::kGranularity>::RegionBSeed(x);
}

// Region A's per-order value at the partition a policy names, for the bodies
// that read the Chebyshev route's region-A pieces directly rather than through
// a policy's fit. It is the Chebyshev partition for the same reason
// PolicyRegionBSeed is - the piece tables are the Chebyshev family's, and the
// bodies that call this read them by construction - and it is the policy's
// granularity so that naming the narrow partition reaches every region-A read
// in a policy-driven body, not only the ones a route's own fit answers.
//
// The rational routes hold the shipped partition, so a policy naming one reads
// the shipped pieces here; that is the partition its own fits-only tail is
// documented at, and the narrow partition beside it is refused where it is
// named (see RouteFit).
template <EvalPolicyLike Policy>
inline double PolicyRegionAValue(int order, double x) noexcept {
    return RegionAValue<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(order, x);
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

// Float-lane region-A seed; see ChebyshevValue. The scheme picks which of the
// two parallel coefficient tables - the Chebyshev one ClenshawSplit reads or
// the monomial one HornerMono reads - the piece is summed from; the pieces,
// their intervals and their degrees are the same table under either.
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
template <EvalScheme kScheme = kDefaultEvalScheme>
BoysForceInline float ChebyshevValueF32(int order, float x) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const float* c = detail::f32::kCoeffs.data() + piece.offset;
    const float* m = detail::f32::kMonoCoeffs.data() + piece.offset;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return FitSum<kScheme, backend::ScalarFp32>(c, m, piece.deg, t);
}

#undef BoysForceInline

// Float-lane region-B seed; see RegionBSeed.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline float RegionBSeedF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return FitSum<kScheme, backend::ScalarFp32>(detail::f32::kBcoeffs.data(),
                                               detail::f32::kMonoBcoeffs.data(),
                                               detail::f32::kBDeg,
                                               t);
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
// flat index of the partition's own piece table, which is how that partition's
// degrees table is indexed.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
double ChebyshevValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    if constexpr (kGranularity == kDefaultFitGranularity)
    {
        const detail::OrderPiece& piece = FindPiece(order, x);
        const std::ptrdiff_t index = &piece - detail::kPieces.data();
        const int deg = degrees[static_cast<std::size_t>(index)];
        const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
        return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                    detail::kMonoCoeffs.data() + piece.offset,
                                                    deg,
                                                    t);
    } else
    {
        const detail::OrderPiece& piece = FindNarrowAPiece(order, x);
        const std::ptrdiff_t index = &piece - detail::kNarrowAPieces.data();
        const int deg = degrees[static_cast<std::size_t>(index)];
        const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
        return FitSum<kScheme, backend::ScalarFp64>(
            detail::kNarrowACoeffs.data() + piece.offset,
            detail::kNarrowAMonoCoeffs.data() + piece.offset,
            deg,
            t);
    }
}

// The region-B seed at the degree the rung certifies for the peak order the
// caller is about to reach, over the partition named. The shipped seed is one
// polynomial over the whole region, so its degrees table is indexed by that
// order; the narrow partition's seed is one polynomial per piece, and its
// table carries the order beside the piece because a piece's own tail is not
// the same as its neighbour's.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
inline double RegionBSeedWithDegrees(double x,
                                     const DegreesArray& degrees,
                                     int peakOrder) noexcept {
    if constexpr (kGranularity == kDefaultFitGranularity)
    {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        return FitSum<kScheme, backend::ScalarFp64>(detail::kBcoeffs.data(),
                                                    detail::kMonoBcoeffs.data(),
                                                    degrees[static_cast<std::size_t>(peakOrder)],
                                                    t);
    } else
    {
        const std::size_t piece = static_cast<std::size_t>(NarrowBPieceOf(x));
        const double a = detail::kNarrowBEdges[piece];
        const double b = detail::kNarrowBEdges[piece + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;
        const std::size_t offset = piece * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);
        const std::size_t degree =
            degrees[piece * (static_cast<std::size_t>(kMaxOrder) + 1)
                    + static_cast<std::size_t>(peakOrder)];
        return FitSum<kScheme, backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                                    detail::kNarrowBMonoCoeffs.data() + offset,
                                                    static_cast<int>(degree),
                                                    t);
    }
}

// The effective-degree tables a policy's rung reads, over the partition the
// policy names. The criterion is the same one either way; the table it is
// measured against is the partition's own, which is what makes a rung a
// reading of the partition the caller chose rather than of the shipped one.
template <double kAccuracyMultiplier, EvalPolicyLike Policy, BoysRole kRole>
constexpr auto RegionADegreeTableOf() noexcept {
    if constexpr (Policy::kGranularity == kDefaultFitGranularity)
    {
        return RegionADegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<Policy::kScheme>()>();
    } else
    {
        return NarrowRegionADegrees<kAccuracyMultiplier,
                                    kRole,
                                    SchemeTailBasis<Policy::kScheme>()>();
    }
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy, BoysRole kRole>
constexpr auto RegionBDegreeTableOf() noexcept {
    if constexpr (Policy::kGranularity == kDefaultFitGranularity)
    {
        return RegionBDegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<Policy::kScheme>()>();
    } else
    {
        return NarrowRegionBDegrees<kAccuracyMultiplier,
                                    kRole,
                                    SchemeTailBasis<Policy::kScheme>()>();
    }
}

// The shipped seed read at a degree handed in directly, which is how the gate's
// granularity book asks a stored row: the partition-naming form above is what the
// rung bodies call, and this one is the shipped partition's reading of the same fit.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeedWithDegrees(double x, int degree) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(
        detail::kBcoeffs.data(), detail::kMonoBcoeffs.data(), degree, t);
}


template <EvalScheme kScheme = kDefaultEvalScheme, typename DegreesArray>
float ChebyshevValueF32WithDegrees(int order, float x, const DegreesArray& degrees) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const std::ptrdiff_t index = &piece - detail::f32::kPieces.data();
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return FitSum<kScheme, backend::ScalarFp32>(detail::f32::kCoeffs.data() + piece.offset,
                                                detail::f32::kMonoCoeffs.data() + piece.offset,
                                                degrees[static_cast<std::size_t>(index)],
                                                t);
}

template <EvalScheme kScheme = kDefaultEvalScheme>
inline float RegionBSeedF32WithDegrees(float x, int degree) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return FitSum<kScheme, backend::ScalarFp32>(detail::f32::kBcoeffs.data(),
                                                detail::f32::kMonoBcoeffs.data(),
                                                degree,
                                                t);
}

// The narrow partition's counterparts of the two above: the same mapped
// argument and the same summation, over the pieces that partition holds and at
// the degree its own table certifies for them. A degree table is indexed the
// way the partition it cuts is read - one entry per narrow region-A piece, one
// per order for the narrow region-B seed - so the pair of tables and the pair
// of helpers move together and neither is usable with the other partition's.
template <EvalScheme kScheme = kDefaultEvalScheme, typename DegreesArray>
double NarrowRegionAValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    const detail::OrderPiece& piece = FindNarrowAPiece(order, x);
    const std::ptrdiff_t index = &piece - detail::kNarrowAPieces.data();
    const int deg = degrees[static_cast<std::size_t>(index)];
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(detail::kNarrowACoeffs.data() + piece.offset,
                                                detail::kNarrowAMonoCoeffs.data() + piece.offset,
                                                deg,
                                                t);
}

template <EvalScheme kScheme = kDefaultEvalScheme>
inline double NarrowRegionBSeedWithDegrees(double x, int degree) noexcept {
    const std::size_t index = static_cast<std::size_t>(NarrowBPieceOf(x));
    const double a = detail::kNarrowBEdges[index];
    const double b = detail::kNarrowBEdges[index + 1];
    const double t = 2.0 * (x - a) / (b - a) - 1.0;
    const std::size_t offset = index * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);
    return FitSum<kScheme, backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                                detail::kNarrowBMonoCoeffs.data() + offset,
                                                degree,
                                                t);
}

// A relaxed rung's two region reads, as one call: the degree table and the
// piece table are one partition's pair, so a body that reaches a stored row at a
// cut degree takes both from the policy it was named with rather than naming a
// partition of its own. Each body below goes through these - the single-order
// entry, the batch entry, the fixed-order entry and the all-N region bodies -
// because a body that reads a coefficient table directly is exactly how a
// narrow policy came to be handed the shipped partition's values under its own
// name, and a fork taken once here cannot be left out of one of them.
//
// The role is the body's own: the single-order shapes read the per-order
// amplification and the batch shapes the recursion's, and the criterion that
// turns a tail into a degree differs between them.
template <EvalPolicyLike Policy, double kAccuracyMultiplier, BoysRole kRole>
inline double PolicyRegionAValueAtRung(int order, double x) noexcept {
    static constexpr auto kDegrees = RegionADegreeTableOf<kAccuracyMultiplier, Policy, kRole>();
    return ChebyshevValueWithDegrees<Policy::kScheme, Policy::kGranularity>(order, x, kDegrees);
}

template <EvalPolicyLike Policy, double kAccuracyMultiplier, BoysRole kRole>
inline double PolicyRegionBSeedAtRung(double x, int order) noexcept {
    static constexpr auto kDegrees = RegionBDegreeTableOf<kAccuracyMultiplier, Policy, kRole>();
    return RegionBSeedWithDegrees<Policy::kScheme, Policy::kGranularity>(x, kDegrees, order);
}

// ---------------------------------------------------------------------------
// The float lane's fit routes
// ---------------------------------------------------------------------------
// A float-lane single-order call reads a coefficient in exactly two places:
// the region-A seed and the region-B seed. This names which pair of fits
// those two are. They are alternatives rather than rungs of one design - the
// Chebyshev tables are the default and the route the lane has always been
// certified with, and naming the rational one changes the coefficients those
// two intervals are evaluated from and nothing else.
//
// The rational route's pieces are the family's own cover of each order's
// interval rather than the Chebyshev table's breaks, because a cover places
// its breaks where its own fit needs them; the two routes therefore read
// different piece tables at the same mapped argument.

// Rational-route piece lookup; see FindPieceF32.
inline const detail::f32::RatPiece& FindRatPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kRatAPieceStart[order];
    const int last = detail::f32::kRatAPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kRatAPieces[i].b)
        {
            return detail::f32::kRatAPieces[i];
        }
    }

    return detail::f32::kRatAPieces[last - 1];
}

// Region-A seed of the rational route: the stored numerator over one plus t
// times the stored denominator, read by Horner in the lane's own arithmetic.
// The mapped argument is the expression ChebyshevValueF32 evaluates, so the
// two routes' fits are read at one t.
inline float RationalValueF32(int order, float x) noexcept {
    const detail::f32::RatPiece& piece = FindRatPieceF32(order, x);
    const float* c = detail::f32::kRatACoeffs.data() + piece.offset;
    const int m = piece.numdeg;
    const int k = piece.dendeg;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (k == 0)
    {
        return num;
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// Region-B seed of the rational route; see RegionBSeedF32.
inline float RegionBSeedRationalF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    float num = detail::f32::kRatBnum[detail::f32::kRatBnumDeg];

    for (int j = detail::f32::kRatBnumDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, detail::f32::kRatBnum[j]);
    }

    float den = detail::f32::kRatBden[detail::f32::kRatBdenDeg - 1];

    for (int j = detail::f32::kRatBdenDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, detail::f32::kRatBden[j]);
    }

    den = backend::ScalarFp32::MulAdd(den, t, 1.0f);
    return num / den;
}

// One rational piece of this lane at a cut pair, in the mapped argument the
// full pair is read in. The cut keeps the low-order terms of both parts: the
// numerator's p_0..p_numDeg and the denominator's q_1..q_denDeg. The
// denominator's coefficients sit above the *stored* numerator, so the position
// of q_j is the piece's full numerator degree's, not the cut's.
inline float RationalPieceF32AtCut(std::size_t index, int numDeg, int denDeg, float t) noexcept {
    const detail::f32::RatPiece& piece = detail::f32::kRatAPieces[index];
    const float* c = detail::f32::kRatACoeffs.data() + piece.offset;
    float num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = c[piece.numdeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[piece.numdeg + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// The region-B seed of this lane at a cut pair; same reading as
// RegionBSeedRationalF32. The stored coefficients are q_1..q_k with the
// constant term held at 1, and each descent below reads the next one down.
inline float RationalSeedF32AtCut(int numDeg, int denDeg, float t) noexcept {
    float num = detail::f32::kRatBnum[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, detail::f32::kRatBnum[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = detail::f32::kRatBden[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, detail::f32::kRatBden[j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// The two routes as one float-lane call reads them. Each policy forwards to
// the helper above, so the default policy's arithmetic is the shipped one
// operation for operation.
template <EvalScheme kScheme = kDefaultEvalScheme>
struct ChebyshevFit32 {
    static float EvalOrder(int n, float x) noexcept {
        return ChebyshevValueF32<kScheme>(n, x);
    }

    // This family's seed covers the whole region at one degree, so the order is
    // not read; the rung form of the fit reads it (ChebyshevFit32AtRung).
    static float RegionBSeed(float x, int /*order*/) noexcept {
        return RegionBSeedF32<kScheme>(x);
    }
};

// The rational family takes no scheme: its numerator and denominator are stored
// in monomial form and read by Horner, so there is no second table for a scheme
// to choose between. A policy that names this family therefore composes with
// either scheme and evaluates the same values under both - which is what the
// double lane's rational route does with its own scheme axis as well.
struct RationalFit32 {
    // Where a batch reading of this route hands an order over to the route's
    // own fit rather than to the band's seed: the lane's own constant, the one
    // the double lane's rational route hands over at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    static float EvalOrder(int n, float x) noexcept {
        return RationalValueF32(n, x);
    }

    // One pair for the whole region, so the order is not read.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        return RegionBSeedRationalF32(x);
    }
};

// The two families under one name: what a float engine that takes a route and a
// scheme reads, the way the double bodies read Policy::Fit.
template <FitRoute kRoute, EvalScheme kScheme>
using FloatRouteFit =
    std::conditional_t<kRoute == FitRoute::kChebyshev, ChebyshevFit32<kScheme>, RationalFit32>;

// ---------------------------------------------------------------------------
// The same two families at a relaxed rung
// ---------------------------------------------------------------------------
// A rung cuts the lane's stored fits by the effective-degree table the role and
// the table's basis certify (boys_effective_degrees.hpp), and each family has
// its own table to cut: the Chebyshev family's degree scan over the piece's
// stored coefficients, and the rational family's pair criterion over its stored
// numerator and denominator.
//
// A fit at a rung answers the same two reads as its m = 1 form - the region-A
// value of one order and the region-B seed - so the engine above it is one body
// per route at every multiplier, and the route and the scheme reach it the way
// they reach it at m = 1.
template <double kAccuracyMultiplier, BoysRole kRole, EvalScheme kScheme>
struct ChebyshevFit32AtRung {
    // The degrees the criterion certifies for this role and this basis. The
    // basis is the table the summation reads (SchemeTailBasis), which is the
    // table its tail is summed from: the two stored forms of one fit hold
    // different numbers, and a degree the Chebyshev tail admits can drop a
    // monomial tail several orders over budget.
    static constexpr auto kDegreesA =
        RegionADegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<kScheme>()>();
    static constexpr auto kDegreesB =
        RegionBDegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<kScheme>()>();

    // One piece of one order, at that piece's effective degree.
    static float EvalOrder(int n, float x) noexcept {
        return ChebyshevValueF32WithDegrees<kScheme>(n, x, kDegreesA);
    }

    // The seed's error reaches order n through the upward recursion, whose gain
    // is A_B(n) - the quantity RegionBDegrees certifies the degree at - so the
    // degree is the one the order it is read for was certified for.
    static float RegionBSeed(float x, int order) noexcept {
        return RegionBSeedF32WithDegrees<kScheme>(x, kDegreesB[static_cast<std::size_t>(order)]);
    }
};

// The rational route at a rung: this lane's own stored pairs, cut by the pair
// criterion at the role's budget. Region A's cut is the single-order reading's -
// the piece is the order's own value and nothing amplifies the cut - so the
// route pays A = 1 there, and region B's seed is one pair for the whole region
// read at order 0; both are what the criterion's region tables derive
// (RationalRegionAF32Degrees, RationalRegionBF32Degrees).
template <double kAccuracyMultiplier, BoysRole kRole>
struct RationalFit32AtRung {
    static constexpr detail::RationalRegionAF32Pairs kPairsA =
        detail::RationalRegionAF32Degrees<kAccuracyMultiplier, kRole>();
    static constexpr detail::RationalRegionBF32Pairs kPairsB =
        detail::RationalRegionBF32Degrees<kAccuracyMultiplier, kRole>();

    // This route's cover of an order's interval is its own, not the Chebyshev
    // table's breaks, so the piece is looked up in the rational table and the
    // pair is read at that piece's cut.
    static float EvalOrder(int n, float x) noexcept {
        const detail::f32::RatPiece& piece = FindRatPieceF32(n, x);
        const std::size_t index =
            static_cast<std::size_t>(&piece - detail::f32::kRatAPieces.data());
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        return RationalPieceF32AtCut(index, kPairsA.num[index], kPairsA.den[index], t);
    }

    // One pair for the region rather than one per order, so the order is not
    // read: every order's output carries the same cut.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
        return RationalSeedF32AtCut(kPairsB.num[0], kPairsB.den[0], t);
    }
};

// The two families at a rung under one name; see FloatRouteFit.
template <double kAccuracyMultiplier, FitRoute kRoute, EvalScheme kScheme, BoysRole kRole>
using FloatRouteFitAtRung =
    std::conditional_t<kRoute == FitRoute::kChebyshev,
                       ChebyshevFit32AtRung<kAccuracyMultiplier, kRole, kScheme>,
                       RationalFit32AtRung<kAccuracyMultiplier, kRole>>;

// Region-A seed of a float-lane batch, in the double precision the downward
// recursion needs (see the batch body below). The route's own fit answers where
// that route's selector takes over, and the shipped family answers below it:
// the same pair of choices, gated by the same constant, that the double lane's
// batch makes for its own seed. It is the double lane's fit and not this
// lane's for the reason the batch body gives - a 1.5e-7 seed is a 5e-3 result
// at nmax = 8, so the per-order floats this lane is certified at cannot seed a
// batch at any order worth the name.
template <FitRoute kRoute, EvalScheme kScheme>
double FloatBatchRegionASeed(int order, double x) noexcept {
    if constexpr (kRoute == FitRoute::kRationalMinimax)
    {
        if (x >= RationalFit32::kRegionAFitsFrom)
        {
            return RegionAValue<RationalFit>(order, x);
        }
    }

    return ChebyshevValue<kScheme>(order, x);
}

// The same seed at a relaxed rung: the route's own fit at the pair or the
// degree this rung certifies, over the double lane's tables - which is the pair
// of tables this lane's batch reads at every multiplier, so the only thing the
// rung changes is where the fit is cut. The role's region-A degrees are the
// double table's own for a batch role, because that is the table the seed is
// evaluated from (RoleUsesDoubleTables), and the cut pair is the double lane's
// rational pieces at the reading this seed gives them, w(b) at the piece's
// right end (RationalRegionASeedDegrees).
template <double kAccuracyMultiplier, FitRoute kRoute, EvalScheme kScheme, BoysRole kRole>
double FloatBatchRegionASeedAtRung(int order, double x) noexcept {
    if constexpr (kRoute == FitRoute::kRationalMinimax)
    {
        if (x >= RationalFit32::kRegionAFitsFrom)
        {
            static constexpr detail::RationalRegionAPairs kPairsA =
                detail::RationalRegionASeedDegrees<kAccuracyMultiplier, kRole>();
            const std::size_t index = ShippedRegionAPartition::PieceIndex(order, x);
            const detail::OrderPiece& piece = ShippedRegionAPartition::PieceAt(index);
            const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
            return RationalPieceAtCut(index, kPairsA.num[index], kPairsA.den[index], t);
        }
    }

    static constexpr auto kDegreesA =
        RegionADegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<kScheme>()>();
    return ChebyshevValueWithDegrees<kScheme>(order, x, kDegreesA);
}

// The float lane's single-order body over a fit policy: one body, so the
// route names the two fits and changes nothing else. Region C reads no
// coefficient at all and is the same three lines under either route.
template <typename Fit>
float SingleOrderF32Body(int n, float x) noexcept {
    if (x == 0.0f)
    {
        return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
    }

    const float x0 = static_cast<float>(kX0);
    const float x1 = static_cast<float>(kX1);

    if (x < x0)
    {
        return Fit::EvalOrder(n, x);
    }

    float f = Fit::RegionBSeed(x, n);

    if (x < x1)
    {
        const float expx = 0.5f * std::exp(-x);

        // The recurrence step is written as the backend's two-rounding
        // multiply-subtract rather than as a bare product and difference. A
        // bare one contracts where the compiler contracts and not where it
        // does not, which would make this value a property of the calling
        // translation unit's flags - and worse, of the optimizer's choice
        // between two inlined copies of this call in one unit, so that two
        // call sites of the same entry could return adjacent values. The
        // spelled form rounds the product and then the difference on every
        // build, so every unit that computes this recurrence computes the same
        // number.
        for (int l = 0; l < n; ++l)
        {
            f = backend::ScalarFp32::MulSub(static_cast<float>(l) + 0.5f, f, expx) / x;
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

// ---------------------------------------------------------------------------
// The bodies: one per entry shape, over the policy a call site selected
// ---------------------------------------------------------------------------
// A body takes the policy as its ONE selection parameter and reads the axes as
// fields, so the fit family, the scheme and anything added later reach the
// recurrences without a parameter per axis: the family is Policy::Fit, the
// scheme is Policy::kScheme, the partition is Policy::kGranularity, and the
// tail orders an order's own fit does not answer are the Chebyshev family's at
// that scheme and that partition.
// The fit is the policy's by default and is overridden only by the relaxed rung
// of a route whose rung is a fit of its own rather than a cut of the shipped
// one (see RationalFitAtRung); everything else about the body - the zero
// argument, the region split, the per-order rule, the domains - is the same
// under either, which is why the rung is a parameter here and not a second body.
template <EvalPolicyLike Policy, typename Fit = typename Policy::Fit>
void AllOrdersBody(int nmax, double x, double* out) noexcept {
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
        // own band answer; the tail orders keep the per-order piece value, at
        // the partition the policy names, which is the value the lane documents
        // for them - below its own end an order is documented at the per-order
        // 1e-15, and a route whose fits hold the wider bar does not answer
        // there. On the Chebyshev route out[k] is bit-identical to the
        // single-order entry's value for every k: that entry applies the same
        // per-(n, x) rule to the same fits.
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
                out[l] = PolicyRegionAValue<Policy>(l, x);
            }

            return;
        }

        double f = PolicyRegionAValue<Policy>(nmax, x);
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

// One order at one argument, in the policy's family and scheme. The fit is
// overridden by the relaxed rung as it is in AllOrdersBody, for the same reason.
template <EvalPolicyLike Policy, typename Fit = typename Policy::Fit>
double SingleOrder(int n, double x) noexcept {
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

        return PolicyRegionAValue<Policy>(n, x);
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

// The route the batch entries' region bodies carry, and the one the fixed-order
// entry carries. Each is the shipped one at every rung: those bodies arrive at
// their values by a path of their own - the batch's region A seeds its downward
// recursion from the top order's stored fit and recurses, region B seeds its
// upward recursion from the stored seed - and they read the shipped tables
// through it. A policy naming the rational route is rejected where the call is
// named, with the reason, rather than answered with the shipped fits under the
// other route's name.
//
// The per-argument entries do not ask for it, because their bodies take the fit
// from the policy: they carry either route at either rung, and the rational one
// through RationalFitAtRung.
// The route the region-partitioned batch shapes carry: the shipped one, at
// every rung. Their region bodies evaluate the shipped seed and the shipped
// per-order fits as their own, arriving at a value by a path of their own - the
// downward recursion from a top order's stored piece, the upward recursion from
// a stored seed - so a policy naming another route takes the entry's
// per-argument path instead, which is the body that reads its fit from the
// policy. The assertion states which shape this is, and the entry's dispatch is
// where the choice is made; nothing falls back silently, because the
// per-argument path is the entry's own second shape and not another route's
// values.
template <EvalPolicyLike Policy>
constexpr void RequireShippedRoute() noexcept
{
    static_assert(Policy::kRoute == kDefaultFitRoute,
                  "this body reaches its values through the shipped fits' own path - the "
                  "downward recursion from a stored piece, or the upward recursion from a "
                  "stored seed - and reads the shipped tables through it: a policy naming "
                  "another route is served by the shape that takes its fit from the policy "
                  "instead, and this shape is not instantiated for one");
}

// A relaxed rung truncates a stored row to a per-order effective degree, and
// each partition carries its own such table: the shipped row's degrees are cut
// from the shipped pieces' coefficients and the narrow row's from the narrow
// pieces', so a rung of the narrow partition is a rung of the narrow fit and not
// the shipped row's degrees applied to narrower intervals. The two tables are
// derived by one criterion over the two tables of coefficients (see
// boys_effective_degrees.hpp) and every m > 1 body below picks the pair that
// matches the partition its policy names, so the granularity and the rung are
// one choice rather than a crossing.
//
// What the choice costs is stated where the partitions are: a rung of the narrow
// partition reads fewer coefficients than a rung of the shipped one at the same
// budget - the narrow pieces are lower-degree to begin with - and pays the
// longer piece scan its table needs.

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

    // m = 1 is the full-accuracy body selected at compile time: no branch,
    // indirection or runtime dispatch sits on that path, and a relaxed rung is
    // a separate instantiation taken one branch below rather than a test the
    // full-accuracy call pays for.
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return SingleOrder<Policy>(n, x);
    } else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        // A rung of the rational route: the route's own body at the pair the
        // rung's criterion certifies. The body is the same one the uncut route
        // runs, so the rung cannot reach a value the route does not serve, and
        // the arguments the route's fits do not cover keep the shipped lane's
        // answer exactly as they do at m = 1.
        return SingleOrder<Policy, RationalFitAtRung<kAccuracyMultiplier>>(n, x);
    } else
    {
        RequireShippedRoute<Policy>();

        if (x == 0.0)
        {
            return 1.0 / (2.0 * n + 1.0);
        }

        if (x < kX0)
        {
            return PolicyRegionAValueAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleSingle>(
                n, x);
        }

        double f =
            PolicyRegionBSeedAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleSingle>(x, n);

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
// declaration near the end of this file. Its default arguments are set there.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

// The single-precision lane's entry on the same axis (defined in
// boys_orders_simd.cpp), declared here for the same reason and one more: this
// lane's degree table is certified against a region budget as well as against a
// table, so the computation budget is a choice it carries and the double lane's
// entry does not.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept;

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersImpl(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (Policy::kPack == PackAxis::kOrders)
    {
        static_assert(Policy::kRoute == FitRoute::kChebyshev ||
                          Policy::kRoute == FitRoute::kRationalMinimax,
                      "a policy naming a route outside the FitRoute enumeration is not one this "
                      "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
        // The across-orders packed lane carries every combination the axis
        // offers: either route's region-A fits, either partition of the shipped
        // route's, and any rung. The four choices reach the lane as its template
        // arguments, so what the entry answers inside the packed interval,
        // outside it, and on a host without the vector tier is one policy's
        // answer throughout. The narrow partition does not carry the stride the
        // shipped lane's fetch uses - its pieces are cut per order - so its lane
        // reads each order's own piece; the axis is the same one either way.
        BoysAllOrdersPacked<Policy::kScheme,
                            kAccuracyMultiplier,
                            Policy::kRoute,
                            Policy::kGranularity>(nmax, x, out);
    } else if constexpr (kAccuracyMultiplier == 1.0)
    {
        AllOrdersBody<Policy>(nmax, x, out);
    } else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        // A rung of the rational route: the route's own body at the pair the
        // rung's criterion certifies, exactly as the single-order entry runs it.
        // The rung is not the polynomial rung's shape - seed at the top order
        // and recurse down - because the cut pair is judged for the value it is
        // read for, and the batch recursion's gain is what the route's pieces
        // were never fitted through.
        AllOrdersBody<Policy, RationalFitAtRung<kAccuracyMultiplier>>(nmax, x, out);
    } else
    {
        static_assert(Policy::kRoute == kDefaultFitRoute,
                      "a policy naming a route outside the FitRoute enumeration is not one this "
                      "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");

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
            double f =
                PolicyRegionAValueAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleBatch>(
                    nmax, x);
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
            double f = PolicyRegionBSeedAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleBatch>(
                x, 0);
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
// is the certified scalar single-lane code, the relaxed branch the same
// bodies at the single-lane effective degrees (BoysRole::kDoubleSingle) -
// so every output element returns the corresponding BoysSingle call's
// value, and the same bits on a build whose bare product-plus-add is two
// roundings. On a build that contracts that form the compiler decides per
// call site whether to fuse it, and this call shape is not the single
// entry's: a value can move by a unit in the last place and no further.
// Keep the two engines' bodies in lockstep - identical source is what holds
// them inside one bound, and it is all that holds them together. The region
// dispatch is per element: mixed-region arguments need no pre-partitioning
// (the portable shape). The dispatch-once-per-batch region structure lives
// in the AVX2 region-sorted lanes (boys_simd.cpp), whose callers partition
// by region first.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysFixedNImpl(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
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

    if constexpr (Policy::kRoute != kDefaultFitRoute)
    {
        // The fixed-order entry carries the route too, and by the entry that
        // takes its fit from the policy: one order at every argument of an
        // array is the per-argument single entry called once per argument -
        // which is the body this entry's own m = 1 path already mirrors
        // verbatim, region for region, so naming a route makes the identity the
        // documentation already claims exact by construction rather than by
        // inspection. The shaped body below is the bit-identity pin's, and the
        // shipped route keeps it.
        for (std::size_t i = 0; i < count; ++i)
        {
            assert(x[i] >= 0.0);
            out[i * stride] = BoysSingleImpl<kAccuracyMultiplier, Policy>(n, x[i]);
        }

        return;
    }
    // The two selections are alternatives and not two independent tests: the
    // branch above returns whenever it is taken, so saying so here is what the
    // code already means - and it is the difference between a compiler reading
    // the block below as discarded and reading it as unreachable.
    else if constexpr (kAccuracyMultiplier == 1.0)
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

                out[i * stride] = PolicyRegionAValue<Policy>(n, xi);
                continue;
            }

            double f = PolicyRegionBSeed<Policy>(xi);

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
                out[i * stride] =
                    PolicyRegionAValueAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleSingle>(
                        n, xi);
                continue;
            }

            // The order-n single-lane seed entry mirrors the relaxed single
            // path; the per-order amplification analysis of BoysSingleImpl
            // applies unchanged.
            double f =
                PolicyRegionBSeedAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleSingle>(xi,
                                                                                              n);

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
// At the reference multiplier the policy reaches this engine as the pair of
// fits its two seeds are read from and the table those fits are summed out of:
// the route picks the family - the shipped Chebyshev fits or the rational set
// BoysSingleF32WithRoute serves - and the scheme picks which of the two
// parallel tables the Chebyshev family is read from, the same choice the double
// lane's bodies make. The engine is one body per route, so naming a policy
// cannot reach a fit the caller did not name.
//
// Past the reference multiplier the same two names pick the same two fits, cut
// where the rung's criterion certifies them: the degrees are the role's own,
// derived from the tables this lane stores and at the basis the scheme reads,
// so a rung is a rung of the family the caller named rather than a rung of one
// family served to every caller.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
float BoysSingleF32Impl(int n, float x) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kGranularity == kDefaultFitGranularity,
                  "the single-precision lane's fits are its own, one partition of region A and "
                  "one region-B seed, with no narrow counterpart: naming the narrow granularity "
                  "on this lane is unbuilt work rather than an unavailable option, since the "
                  "float lane's own narrow table is a table to generate - read the narrow "
                  "partition on the double lane, or the shipped one here");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order at one argument, so neither axis has a vector "
                  "lane to fill here: the value is a single float, and the axis this library "
                  "carries on this shape is the arguments axis - the field's default");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0f);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return SingleOrderF32Body<FloatRouteFit<Policy::kRoute, Policy::kScheme>>(n, x);
    } else
    {
        constexpr BoysRole kRole =
            (Policy::kBudget == BoysBudget::kFloat) ? BoysRole::kF32Single : BoysRole::kF32Fp16Single;
        return SingleOrderF32Body<
            FloatRouteFitAtRung<kAccuracyMultiplier, Policy::kRoute, Policy::kScheme, kRole>>(n, x);
    }
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersF32Impl(int nmax, float x, float* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kGranularity == kDefaultFitGranularity,
                  "the single-precision lane's fits are its own, one partition of region A and "
                  "one region-B seed, with no narrow counterpart: naming the narrow granularity "
                  "on this lane is unbuilt work rather than an unavailable option, since the "
                  "float lane's own narrow table is a table to generate - read the narrow "
                  "partition on the double lane, or the shipped one here");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this call produces nmax + 1 orders of one argument, so the orders axis names "
                  "real work on this shape and this engine has no body to answer it with: the "
                  "single-precision lane carries no packed kernel - the AVX2 tier is double and "
                  "half only - so an orders-axis policy here would be read and then ignored. "
                  "This assertion refuses it instead, and what it refuses is a body nobody has "
                  "written rather than a combination that cannot exist: read the orders axis on "
                  "the double lane, or the shipped arguments axis here");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if constexpr (Policy::kPack == PackAxis::kOrders)
    {
        // The across-orders packed lane: eight orders of one argument in one
        // vector register, which is the shape BoysAllOrdersF32(nmax, x, out)
        // has and the axis the caller named. It carries the same tables and the
        // same arithmetic as this engine - a lane's value is the per-order
        // value, and the lane's own suite asserts it - so the axis changes
        // which lane runs and not which fit is read.
        //
        // The rung restriction is the one every other float entry states: a
        // degree table is certified against one stored table of one fit family
        // and against one region budget, so past the reference multiplier this
        // lane serves the shipped route and scheme at the budget it was given.
        if constexpr (kAccuracyMultiplier != 1.0)
        {
            static_assert(Policy::kRoute == kDefaultFitRoute &&
                              Policy::kScheme == kDefaultEvalScheme,
                          "the relaxed rungs cut the float lane's fits by a table of effective "
                          "degrees, and a degree table is certified against one stored table of "
                          "one fit family: past the reference multiplier this engine serves the "
                          "shipped route and scheme alone, and every route and scheme is served "
                          "at it");
        }

        BoysAllOrdersF32Packed<Policy::kScheme,
                               kAccuracyMultiplier,
                               Policy::kRoute,
                               Policy::kBudget>(nmax, x, out);
    } else if constexpr (kAccuracyMultiplier == 1.0)
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
            const double seed =
                FloatBatchRegionASeed<Policy::kRoute, Policy::kScheme>(nmax,
                                                                       static_cast<double>(x));
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

        float f = FloatRouteFit<Policy::kRoute, Policy::kScheme>::RegionBSeed(x, 0);
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
        using Fit =
            FloatRouteFitAtRung<kAccuracyMultiplier, Policy::kRoute, Policy::kScheme, kRole>;

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
            const double seed =
                FloatBatchRegionASeedAtRung<kAccuracyMultiplier, Policy::kRoute, Policy::kScheme,
                                            kRole>(nmax, static_cast<double>(x));
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
        float f = Fit::RegionBSeed(x, 0);
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

// The float lane's all-N batch: the per-argument all-orders body at every
// argument, the results scattered into the caller's planes. The AVX2 tier is
// double and half only - there is no packed float region kernel - so this entry
// has no homogeneous run to hand to a lane, no scratch to ask the caller for,
// and no speed of its own to claim over the same loop written at the call site.
// What it carries is the shape: the one BoysAllN has in the double lane and
// BoysCuda::AllNF32 has on the device.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNF32Impl(int nmax, const float* x, float* out, std::size_t count) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry answers each argument through the all-orders entry, so it has no "
                  "run to hand to a packed lane of its own: the single-precision lane's one "
                  "all-orders body is the scalar one (the AVX2 tier is double and half only), "
                  "and that body refuses the orders axis for the reason its own assertion "
                  "states. Naming the axis here would be read and ignored; read the orders axis "
                  "on the double lane, or the shipped arguments axis here");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0f);

        float row[kMaxBoysOrder + 1];
        BoysAllOrdersF32Impl<kAccuracyMultiplier, Policy>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
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
// order at a time - at the policy the caller named, so a relaxed multiplier
// falls back to that rung of that route rather than to the reference one -
// which is a defined answer inside the entry's own bound rather than the packed
// lane.
//
// The four choices a policy makes reach the lane as template arguments: the
// scheme picks which polynomial table and which summation the shipped route's
// fits are read with, the route picks which region-A fits the lane carries, the
// accuracy multiplier picks the degree a fit is read at, and the partition
// picks the table those fits are cut into. The definition and its
// instantiations are in boys_orders_simd.cpp.
template <EvalScheme kScheme = kDefaultEvalScheme,
          double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          FitRoute kRoute = kDefaultFitRoute,
          FitGranularity kGranularity = kDefaultFitGranularity>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 1.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 64.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 256.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 1024.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 4096.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 16384.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 65536.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 1.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 64.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 256.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 1024.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 4096.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 16384.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme, 65536.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;

extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 1.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 64.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 256.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 1024.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 4096.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 16384.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 65536.0, FitRoute::kChebyshev>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 1.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 64.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 256.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 1024.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 4096.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 16384.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner, 65536.0, FitRoute::kRationalMinimax>(
    int nmax, double x, double* out) noexcept;

// The narrow partition, at both schemes and every rung, on the shipped route
// alone: the partition is a partition of that route's region-A fits.
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         1.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         64.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         256.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         1024.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         4096.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         16384.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<kDefaultEvalScheme,
                                         65536.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         1.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         64.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         256.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         1024.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         4096.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         16384.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
extern template void BoysAllOrdersPacked<EvalScheme::kHorner,
                                         65536.0,
                                         FitRoute::kChebyshev,
                                         FitGranularity::kNarrow>(int nmax,
                                                                  double x,
                                                                  double* out) noexcept;
// The single-precision lane's shapes on the same axis: two schemes and two
// computation budgets at the reference multiplier with either route, and the
// six relaxed rungs of the shipped route and scheme alone. Declared here for
// the reason above - so that a call site reaches the definition the library
// already holds rather than instantiating a second copy of the body.
#define BOYS_F32_ORDERS_PACKED_REFERENCE(kScheme, kBudget)                                         \
    extern template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget>(      \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget>(                     \
        int nmax, float x, float* out) noexcept;

#define BOYS_F32_ORDERS_PACKED_RUNGS(kBudget)                                                      \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 64.0, FitRoute::kChebyshev, kBudget>(               \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 256.0, FitRoute::kChebyshev, kBudget>(              \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 1024.0, FitRoute::kChebyshev, kBudget>(             \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 4096.0, FitRoute::kChebyshev, kBudget>(             \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 16384.0, FitRoute::kChebyshev, kBudget>(            \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void                                                                            \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 65536.0, FitRoute::kChebyshev, kBudget>(            \
        int nmax, float x, float* out) noexcept;

BOYS_F32_ORDERS_PACKED_REFERENCE(kDefaultEvalScheme, BoysBudget::kFloat)
BOYS_F32_ORDERS_PACKED_REFERENCE(kDefaultEvalScheme, BoysBudget::kFp16)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16)
BOYS_F32_ORDERS_PACKED_RUNGS(BoysBudget::kFloat)
BOYS_F32_ORDERS_PACKED_RUNGS(BoysBudget::kFp16)

#undef BOYS_F32_ORDERS_PACKED_REFERENCE
#undef BOYS_F32_ORDERS_PACKED_RUNGS

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
        double f = PolicyRegionAValue<Policy>(nmax, x);
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
        double f =
            PolicyRegionAValueAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleBatch>(nmax, x);
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
        plane[static_cast<std::size_t>(l) * count] = PolicyRegionAValue<Policy>(l, x);
    }
}

// Region B: the F0 seed + upward recursion.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
inline void BoysAllNBodyRegionB(int nmax, double x, std::size_t count, double* plane) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        double f = PolicyRegionBSeed<Policy>(x);
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
        double f =
            PolicyRegionBSeedAtRung<Policy, kAccuracyMultiplier, BoysRole::kDoubleBatch>(x, 0);
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
    //
    // The narrow partition is the scalar bodies' here as well: the grouped
    // kernel reads the shipped piece table by index, one order at a time, so
    // the granularity joins the condition above rather than the lane. What a
    // narrow policy trades away on this entry is the four-wide lane, which is a
    // cost of the option and not a value it returns - the scalar bodies below
    // are the per-argument path's own, exact to the bit.
    if constexpr (kAccuracyMultiplier == 1.0 && Policy::kRoute == kDefaultFitRoute &&
                  Policy::kScheme == EvalScheme::kSplitClenshaw &&
                  Policy::kPack == PackAxis::kArguments &&
                  Policy::kGranularity == kDefaultFitGranularity)
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
// selectors it names, because each says what the call can be evaluated by:
//
//   the region-partitioned shape  an arguments-axis lane keeps four arguments
//                       of one order in a register, so the entry partitions its
//                       arguments by the interval their answer comes from and
//                       hands each homogeneous run to the shape that serves it.
//                       That shape evaluates the shipped fits as its own body,
//                       so it is what the shipped route and the arguments axis
//                       are served by. It is BoysAllNPartitionedImpl below.
//   the per-argument shape  every other call. An orders-axis lane keeps four
//                       orders of ONE argument in a register, which is this
//                       entry's call shape one argument at a time -
//                       out[k * count + i] is F_k(x[i]), so an argument's whole
//                       order vector is already what the entry writes - and a
//                       route other than the shipped one is carried by the
//                       all-orders entry's body, which takes its fit from the
//                       policy. Both are the per-argument path, so both are
//                       served by it, and neither has anything for the region
//                       grouping to group.
//
// Every value either shape returns is inside the bound the entry documents:
// the partitioned shape's region-A lane answers at the per-order region-A bar,
// the per-argument body is the all-orders entry's own, and that entry's bound
// is this entry's.
//
// A relaxed multiplier on the orders axis reaches its rung the same way the
// shipped axis's does, in the engine itself (BoysAllOrdersImpl), and the route
// reaches its rung there too, so this entry states neither: it dispatches, and
// the engine is where both are answered.
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

    if constexpr (Policy::kPack == PackAxis::kOrders || Policy::kRoute != kDefaultFitRoute)
    {
        // The grouping scratch is the partitioned shape's: this path partitions
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
    RequireShippedRoute<Policy>();
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this is the plane entry's arguments-axis shape: it partitions its arguments "
                  "by region and dispatches each group to a lane that packs four arguments. A "
                  "call naming the orders axis, or a route other than the shipped one, is "
                  "served by BoysAllNImpl's per-argument path");
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
// ordering the caller declared is what makes the partitioned shape's grouping
// free, so it is a fact about that shape's input and not about this entry. A
// call naming the orders axis, or a route other than the shipped one, takes the
// per-argument path, which neither reads nor needs the ordering - which is why
// the overload accepts both without asking the caller for anything it does not
// already promise.
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

    if constexpr (Policy::kPack == PackAxis::kOrders || Policy::kRoute != kDefaultFitRoute)
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
    RequireShippedRoute<Policy>();
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this is the sorted overload's arguments-axis shape: the ordering the caller "
                  "declared is what makes its grouping free. A call naming the orders axis, or "
                  "a route other than the shipped one, is served by BoysAllNSortedImpl's "
                  "per-argument path, which needs no grouping");
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

// The per-element-top-order batch: the per-argument all-orders body at each
// argument's own top order, scattered into the caller's planes. The tops differ
// per element, so a run has no common nmax to be grouped at and this entry has
// nothing to group; its whole content is the layout, and the cells above each
// column's own top order, which it leaves exactly as the caller left them.
//
// The body is the per-argument entry's, so this entry takes the policies that
// entry takes - including a named fit route, which the plane entry (whose runs
// carry the shipped fits as their own region bodies) does not.
template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNAtOrdersImpl(const int* n, const double* x, double* out, std::size_t count) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(count == 0 || n != nullptr);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(n[i] >= 0 && n[i] <= kMaxBoysOrder);
        assert(x[i] >= 0.0);

        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<kAccuracyMultiplier, Policy>(n[i], x[i], row);

        for (int l = 0; l <= n[i]; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
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
void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept {
    detail::BoysAllNAtOrdersImpl<kAccuracyMultiplier, Policy>(n, x, out, count);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
float BoysSingleF32(int n, float x) noexcept {
    return detail::BoysSingleF32Impl<kAccuracyMultiplier, Policy>(n, x);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept {
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier, Policy>(nmax, x, out);
}

template <double kAccuracyMultiplier, EvalPolicyLike Policy>
void BoysAllNF32(int nmax, const float* x, float* out, std::size_t count) noexcept {
    detail::BoysAllNF32Impl<kAccuracyMultiplier, Policy>(nmax, x, out, count);
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
