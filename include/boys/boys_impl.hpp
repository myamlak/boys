#pragma once

// The template definitions behind the accuracy-multiplier surface of
// boys/boys.hpp: the kernel every templated entry of that header is compiled
// from. boys/boys.hpp includes it at its end, after the entries and the tags
// they name are declared - the only order in which the definitions compile - so
// it is a continuation of that header and never included on its own.
//
// The library's own instantiations are the default set (boys.cpp /
// boys_simd.cpp), which the extern-template declarations in boys/boys.hpp route
// the default call sites to. A call site naming another evaluation policy
// compiles its route from here.
//
// Every entry is compiled twice under if constexpr. The m = 1 branch is the
// certified body, instantiated at the shipped fit and pinned bit-for-bit: the
// discarded relaxed branch must add no instruction, branch or load to it. The
// m > 1 branch evaluates the seed fits at the compile-time effective degrees of
// boys_effective_degrees.hpp; the relaxed region-C paths are m-invariant (the
// asymptotic form has no coefficients to truncate) and only their scalar tails
// carry the multiplier.
//
// Everything the caller selects about *how* a value is produced travels as one
// parameter, the evaluation policy (EvalPolicy in backend.hpp): the fit route,
// the summation scheme and a single-precision engine's budget. The engines read
// its fields, so the surface between a call site and a fit does not grow with
// the number of axes, and the two axes meet in one place (RouteFit, backend.hpp).

/// \cond
// Not API: the kernel the entries are compiled from. The header ships only
// because the entries are header-defined.

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"

#if BoysFp16
#include "boys/f16.hpp"
#endif

#include <algorithm>
#include <array>
#include <bit>
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
// Which partition a fit answers
// ---------------------------------------------------------------------------
// Every fit family in this file is written against a set of partitions and reads
// every other granularity as one of them. The derived families - the Chebyshev
// family, the rational route's derived members, and the rung forms of both - carry
// the shipped and the narrow partition: their granularity parameter selects the
// shipped tables and reads everything else as the narrow ones. The grid's own two
// members carry the grid and nothing else.
//
// That two-case conditional is what the uniform-substitution defects this library
// has had have in common. A body that resolves a policy's partition through a
// derived family answers a policy naming the grid out of the narrow pieces, at a
// certified bound, under the grid's name, with nothing reporting it - the read
// succeeds and the numbers are another partition's. Each defect was found by a
// person reading code, and each was closed where it happened to be.
//
// The question is one question at every one of them: does the fit this path will
// read answer the partition it was named with? It is answered once, below, and
// every path that resolves a partition for an answer asks it rather than restating
// the check where it stands - so a further path fails to compile where it reads
// instead of where somebody remembered to look.

/// Whether the fit \c Fit carries the uniform grid's own table.
///
/// The derived families do not carry it: their granularity parameter is a two-case
/// conditional over the shipped and the narrow partition, and the grid is a third
/// value it has no answer for. What a path naming the grid through one of them gets
/// is the narrow member, read under the grid's name. The specializations below are
/// the two families that do carry it - the two members \c RouteFit resolves a policy
/// naming the grid to, one per route - and both are named here because a fact about
/// which fits carry the grid that named one of the two would be the same kind of
/// omission as a two-case conditional that forgot a third partition.
template <typename Fit>
inline constexpr bool kFitCarriesUniform = false;

template <EvalScheme kScheme>
inline constexpr bool kFitCarriesUniform<UniformFit<kScheme>> = true;

template <>
inline constexpr bool kFitCarriesUniform<RationalFitUniform> = true;

/// Whether the fit \c Fit answers an argument at the partition \c kGranularity from
/// that partition's own table, rather than from another partition's fit under its
/// name.
///
/// \param kGranularity the partition a policy or a body named
///
/// \returns true where \c Fit's stored tables are a cut of that partition
///
/// The switch carries no `default:` arm on purpose: gcc and clang warn for an
/// enumerator it does not name, and this tree builds with -Werror, so a partition
/// added to the enumeration is a failed build at every site that asks this question
/// rather than a site somebody has to remember. MSVC emits no -Wswitch, and what
/// carries the same fact there is the assertion each caller writes on the answer.
template <typename Fit>
constexpr bool FitAnswersPartition(FitGranularity kGranularity) noexcept
{
    switch (kGranularity)
    {
    case FitGranularity::kCoarsest:
    case FitGranularity::kNarrow:
        // The derived families' own two. The grid's members carry neither: they are
        // read over the grid's intervals and have no shipped or narrow fit at all.
        return !kFitCarriesUniform<Fit>;

    case FitGranularity::kUniform:
        return kFitCarriesUniform<Fit>;
    }

    // A value outside the enumeration, which no family here is written against and
    // no fit answers from a table of its own. The answer is no, so that a partition
    // named through such a value fails closed wherever a path asks this rather than
    // being resolved to the narrow tables as the bodies' own conditionals would.
    return false;
}

/// The granularity a body was named with, carried as a type so that the dependent
/// false below can be instantiated on it at the arm that refuses.
template <FitGranularity kGranularity>
struct GranularityTag {};

/// A false that depends on what it is instantiated with.
///
/// The arm a granularity switch keeps for the enumerators it does not name: an
/// assertion on this is evaluated where the arm it stands in is instantiated and
/// nowhere else, so a fourth partition added to FitGranularity fails the build at
/// every such switch rather than being answered out of the last arm's table under
/// its own name. The parameter is what makes the assertion dependent - a bare
/// `false` would be rejected where the arm is written.
template <typename>
inline constexpr bool kAlwaysFalse = false;

// ---------------------------------------------------------------------------
// Scalar double lane helpers
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
// A fit reads one of two stored partitions of the same region, asking each the
// same two things: which piece the argument falls in and what that piece's
// interval is. The body asks the fit for its partition rather than carrying a
// lookup of its own, so the partitions differ only in the tables they name.
//
// The shipped partition is the per-order piecewise table the lane reads. The
// narrow one is derived from the same region at a lower degree per piece and
// more pieces: the same values at fewer coefficients per evaluation, against
// more stored rows and a longer scan to the piece.
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
// of backend B and at its width. The mapped argument t is the caller's, because
// the lanes map it differently on purpose.
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

    // The odd part below is the Clenshaw finalization for the D family with the
    // top odd coefficient c[2m-1] and needs an even deg >= 4 (deg == 2 is handled
    // above), which is all the generator (tools/gen_boys_coefficients.py) emits.
    assert(deg >= 4 && deg % 2 == 0);

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

    // Odd part: c[1], c[3], ..., c[2m-1] with the D recurrence.
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

// Horner's rule over the monomial form of a fit, in the arithmetic of backend B
// and at its width, at the caller's mapped argument t.
//
// The monomial coefficients are safe to sum because t never leaves [-1, 1] (the
// affine map is the fit's own interval): they stay within a small factor of the
// Chebyshev ones however high the degree, so the rounding has the recurrence's
// shape rather than a cancellation problem.
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
// A region-A route is a coefficient set and the scheme it is read in, named as a
// type (the FitPolicy contract in backend.hpp). A family is written once per scheme
// it holds a stored form for; the body is written once and instantiated per
// (route, scheme) pair, owning the zero argument's closed form, the region split,
// the piece lookup, the mapped argument, the recurrences, the per-order rule and
// the domains. The coefficients, degrees and piece intervals are compile-time facts
// of the generated tables; which piece an (order, argument) pair falls in is the one
// run-time choice, so a family reads its degree at the index the body hands it
// rather than carrying the degree in its type. The two axes meet in RouteFit below,
// where a pair the library does not carry fails with the reason rather than inside
// a recurrence.

// The extended band's per-(n, x) dispatch (the pure-function rule): an order n takes
// the extended seed exactly when x >= kTierThresholds[n], whatever the entry point
// or the other orders of a batch. Each entry of the n-indexed threshold table is the
// certified boundary of the smallest kmax row covering it (the rows 4/8/16/32), so
// BoysAllOrders(n,x)[n] == BoysSingle(n,x) == BoysAllOrders(m>=n,x)[n] exactly. The
// band is the m = 1 lane's only; the m > 1 branch and the float lane keep their own
// dispatch.

// The extended-band seed: F_0(x) on [kExtendedBX0, kX0), summed by the named
// scheme (see ChebyshevValue for the name). It serves the upward recursion below
// kX0 in the m = 1 double lanes only, dispatched per (n, x) at kTierThresholds;
// the m > 1 branch and the float lanes keep their own dispatch.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBExtendedSeed(double x) noexcept {
    const double t = 2.0 * (x - kExtendedBX0) / (kX0 - kExtendedBX0) - 1.0;
    return FitSum<kScheme, backend::ScalarFp64>(detail::kExtendedBcoeffs.data(),
                                                detail::kMonoExtendedBcoeffs.data(),
                                                detail::kExtendedBDeg,
                                                t);
}

// The three division forms, one entry each, and the selectors the bodies read.
// Which is cheapest is a property of the host (DivisionForm in accuracy.hpp), so the
// bodies take the form from the policy rather than writing one of the three into a
// body, and each entry is templated on the value type because a form the caller
// names is that lane's arithmetic for every step of the ladder it governs.

// The exact form: the correctly rounded quotient, and what a published bound
// over a region is stated for.
template <typename T>
inline T DivideExact(T a, T x) noexcept {
    return a / x;
}

// The plain form: the quotient through the divisor's reciprocal. It rounds
// twice where the exact form rounds once, so a step may differ by an ulp and a
// ladder of them accumulates the difference.
template <typename T>
inline T DividePlain(T a, T invx) noexcept {
    return a * invx;
}

// The refined form: the plain product, then the classical refinement. The product's
// error is recovered exactly by the fused multiply-add and carried back through the
// reciprocal by the second, which is the correctly rounded quotient whenever the
// reciprocal is the correctly rounded 1/x.
//
// An infinite divisor is the one argument where that recovery cannot run: the
// residual is `a - quotient * x` and `quotient` is a signed zero there, so
// `quotient * x` is `0 * inf` and the first fused multiply-add hands back a NaN the
// second spreads. The branch costs nothing, since a finite numerator over an
// infinite divisor is exactly the signed zero the product already is.
template <typename T>
inline T DivideByReciprocal(T a, T x, T invx) noexcept {
    const T quotient = a * invx;

    if (std::isinf(x))
    {
        return quotient;
    }

    return std::fma(std::fma(-quotient, x, a), invx, quotient);
}

// One divide in the form named, taking whichever of the divisor and its
// reciprocal that form reads.
template <DivisionForm kForm, typename T>
inline T DivideStep(T a, T x, T invx) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(invx);
        return DivideExact(a, x);
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        static_cast<void>(x);
        return DividePlain(a, invx);
    } else
    {
        return DivideByReciprocal(a, x, invx);
    }
}

// The reciprocal a form needs, and the exact form's is not formed at all: the
// three are ranked on the work each actually does, so a form that divides must
// not also pay for a reciprocal it never reads.
template <DivisionForm kForm, typename T>
inline T StepReciprocal(T x) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(x);
        return T{0};
    } else
    {
        return T{1} / x;
    }
}

// The downward step's divisor, l + 1/2, as a table of its reciprocals.
//
// The downward ladder is the one chain of steps that does not divide by the
// argument: F_l is recovered from F_{l+1} by dividing by the step's constant, and
// that constant is exact in both value types for every order the recurrences run to,
// so its reciprocal is a compile-time constant and the trade the axis names - a
// product in place of a division - is available here as on the steps that divide by
// the argument. Forming the reciprocal at the step from `l` would be a division per
// order, which a form whose whole point is not to divide cannot pay and still
// undercut the exact form. Reading the constant leaves the plain form one product
// per order and the refined form the product and its two fused multiply-adds.
template <typename T>
inline constexpr std::array<T, static_cast<std::size_t>(kMaxBoysOrder) + 1>
    kDownwardReciprocals = [] {
        std::array<T, static_cast<std::size_t>(kMaxBoysOrder) + 1> table{};

        for (int l = 0; l <= kMaxBoysOrder; ++l)
        {
            table[static_cast<std::size_t>(l)] = T{1} / (static_cast<T>(l) + T{0.5});
        }

        return table;
    }();

// The downward step's divide, in the form named. The exact form divides by
// l + 1/2 as the recurrence is written; the two reciprocal forms read the
// constant table above, and the refined one carries its product back to the
// correctly rounded quotient because the table entry is the correctly rounded
// 1/(l + 1/2).
template <DivisionForm kForm, typename T>
inline T DivideDownwardStep(int l, T a) noexcept {
    const std::size_t index = static_cast<std::size_t>(l);

    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        return DivideExact(a, static_cast<T>(l) + T{0.5});
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        return DividePlain(a, kDownwardReciprocals<T>[index]);
    } else
    {
        return DivideByReciprocal(a, static_cast<T>(l) + T{0.5}, kDownwardReciprocals<T>[index]);
    }
}


// ---------------------------------------------------------------------------
// Region B's exponential, at the accuracy the ladder demands of it
// ---------------------------------------------------------------------------
// The region-B body seeds F_0 and steps to the caller's top order N with the same
// term at every step:
//
//     F_{l+1} = ((l + 1/2) F_l - t) / x,        t = e^{-x}/2.
//
// An error d in `t` reaches F_N multiplied by the products of the steps that follow
// it, summed over where it can be injected:
//
//     T(N, x) = (1/(2x)) * sum_{l=0}^{N-1} prod_{j=l+1}^{N-1} (2j+1)/(2x),
//
// so the ladder's answer moves by |delta| T(N, x) and the library's absolute promise
// admits the term as long as |delta| <= kRegionBExpBar / T(N, x) - a relative
// requirement of kRegionBExpBar / (t * T(N, x)) on it, near double precision at kX0
// where the ladder's own gain is 3.8e4 and slack at the top of the region where the
// gain is below 0.2. The requirement and not a preference decides which exponential
// an argument gets: above kRegionBExpCheapFrom the polynomial below stands inside it
// ten times over, below it the libm call is the only admissible one. It is read at
// the widest ladder (N = 32), so a batch's column and the per-argument entry
// documented to equal it both evaluate one function of x.
inline constexpr double kRegionBExpBar = 1.0e-14;

// The smallest x whose requirement reaches ten times the polynomial's own
// error, solved from T(32, 0, x). At a narrower ladder the requirement is
// slacker, so what this buys at N = 32 it buys at every N <= 32 as well.
inline constexpr double kRegionBExpCheapFrom = 16.173039304440838;

// e^{-r} on |r| <= ln 2/2, degree 7, Chebyshev fit, 8.336e-11 relative
// (re-measured at 40 digits against the exponential itself).
inline constexpr double kRegionBExpReduced[8] = {
    0.9999999999190966,
    -0.9999999999910163,
    0.5000000168381031,
    -0.16666666853653805,
    0.04166608365203089,
    -0.008333268585902836,
    0.0013956053200368968,
    -0.00019915868993476617,
};

inline constexpr double kRegionBExpLog2e = 1.4426950408889634073599246810018921;
inline constexpr double kRegionBExpLn2 = 0.6931471805599453094172321214581766;
inline constexpr double kRegionBExpRoundMagic = 6755399441055744.0; // 1.5 * 2^52

// 0.5 * e^{-x} for a region-B argument, in the member the policy named.
//
// The accurate member is the library routine, and it is not a second body: the
// single-precision lane's region B has always run it, and both lanes' band and
// downward seeds run it at every member of the axis, because those are region A
// and the axis is region B's.
//
// The fast member is the reduced-argument polynomial. The reduction is the
// textbook one - x = k ln2 + r with |r| <= ln2/2, so e^{-x} = 2^{-k} e^{-r} - with
// k out of a magic constant rather than a libm rounding call and 2^{-k} from the
// exponent field rather than ldexp. Over region B k is in [25, 42], far from the
// exponent field's ends, so the scale is exact. Below kRegionBExpCheapFrom the
// ladder's own requirement, which kRegionBExpBar states above, is tighter than the
// polynomial's error, and this member reads the library routine there: that arm is
// part of the member rather than a fallback, because the polynomial alone would
// fail a bound over a band interior to region B, and a member whose failing band
// is interior to the region is not offered at all.
template <RegionBExp kExp>
inline double RegionBHalfExp(double x) noexcept {
    if constexpr (kExp == RegionBExp::kAccurate)
    {
        return 0.5 * std::exp(-x);
    }
    else
    {
        if (x < kRegionBExpCheapFrom)
        {
            return 0.5 * std::exp(-x);
        }

        const double biased = x * kRegionBExpLog2e + kRegionBExpRoundMagic;
        const double kd = biased - kRegionBExpRoundMagic;
        const double r = x - kd * kRegionBExpLn2;

        double p = kRegionBExpReduced[7];

        for (int k = 6; k >= 0; --k)
        {
            p = p * r + kRegionBExpReduced[k];
        }

        const int k = static_cast<int>(kd);
        return 0.5 * std::bit_cast<double>(static_cast<std::uint64_t>(1023 - k) << 52) * p;
    }
}

// The same member at the single-precision lane's own type.
//
// The accurate member is the single-precision routine - 0.5f * expf, the
// arithmetic that lane's published figures were measured at - and not the double
// one narrowed: routing it through the shape above would move values the lane's
// documents speak for. The fast member is the double arithmetic rounded once, which
// is the same program the double lane runs at that member; the widening is exact
// and the narrowing costs half an ulp, which is the whole of the member's error
// above the cut and all of it but the routine's own below.
template <RegionBExp kExp>
inline float RegionBHalfExpF32(float x) noexcept {
    if constexpr (kExp == RegionBExp::kAccurate)
    {
        return 0.5f * std::exp(-x);
    }
    else
    {
        return static_cast<float>(RegionBHalfExp<RegionBExp::kFast>(static_cast<double>(x)));
    }
}

// One step of the upward recursion: F_l(x) from F_{l-1}(x). The band's orders,
// the single-order band entry and region B all ride this step, in the division
// form the policy names.
template <DivisionForm kForm = kDefaultDivisionForm>
inline double UpwardStep(int l, double f, double x, double invx, double expx) noexcept {
    return DivideStep<kForm>((l - 0.5) * f - expx, x, invx);
}

// Region-A F_order(x) in a fit's own scheme, over the piece the fit's own
// partition puts the argument in, at that piece's mapped argument.
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

// Region B's seed from the shipped partition: one polynomial over the whole of
// [kX0, kX1), at the degree the table carries. Region B's seed is this one fit at
// every granularity except the narrow one - the grid's member stores no seed of its
// own, and the extended band is the same fit because nothing amplifies it - so the
// shipped and the uniform partition read the table this returns, and only the narrow
// one has pieces to cut it into.
template <EvalScheme kScheme>
inline double ShippedRegionBSeed(double x) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;

    return FitSum<kScheme, backend::ScalarFp64>(
        detail::kBcoeffs.data(), detail::kMonoBcoeffs.data(), detail::kBDeg, t);
}

// The Chebyshev route, at the scheme its coefficients are summed in: the shipped
// family, and the one holding both stored forms - the Chebyshev table the split
// Clenshaw recurrence reads and the monomial table Horner reads, over the same
// pieces at the same degrees. It holds both partitions of both regions too, which is
// what the granularity axis selects; the two partitions answer the same domain, so
// the recurrences, the region split and the per-order rule are the same code either
// way. See FitGranularity for what the axis is and is not.
template <EvalScheme kScheme, FitGranularity kGranularity>
struct ChebyshevFit {
    // This family stores the shipped and the narrow partition and no uniform one: its
    // region-A read is piece-indexed, and the grid's cells are interval-major, read at
    // the degrees they were fitted at through UniformFit's own locator. A uniform
    // instantiation would therefore answer every read of it out of the narrow pieces
    // under the grid's name - which is why the third name is refused here, at the
    // alias that would otherwise resolve it, rather than by that alias.
    static_assert(kGranularity != FitGranularity::kUniform,
                  "ChebyshevFit carries the shipped and the narrow partition and no uniform "
                  "table: the grid's cells are not pieces of this family, and a uniform "
                  "instantiation of it answers every read out of the narrow pieces under the "
                  "grid's name. The grid's own read is UniformFit's");

    // The region-A partition this fit reads; the bodies ask for it rather than
    // for the table, so a route over the fit does not have to know which. The two
    // values the assertion above leaves are the two this alias names, in the
    // enumeration's order.
    using Partition = std::conditional_t<kGranularity == FitGranularity::kCoarsest,
                                         ShippedRegionAPartition,
                                         NarrowRegionAPartition>;

    // The band's orders are read from the band's seed, so this route's own
    // answer starts at the band's left edge.
    static constexpr double kRegionAFitsFrom = detail::kTierThresholds[0];

    static double EvalPiece(std::size_t index, double t) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            const detail::OrderPiece& piece = detail::kPieces[index];
            return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                        detail::kMonoCoeffs.data() + piece.offset,
                                                        piece.deg,
                                                        t);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            const detail::OrderPiece& piece = detail::kNarrowAPieces[index];
            return FitSum<kScheme, backend::ScalarFp64>(
                detail::kNarrowACoeffs.data() + piece.offset,
                detail::kNarrowAMonoCoeffs.data() + piece.offset,
                piece.deg,
                t);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // Region A is where the three partitions differ, and this one has no table
            // of this shape: it is piece-indexed, and the grid's cells are interval-major
            // with a degree of their own. The arm stands rather than falling through to
            // the narrow pieces, so a uniform instantiation that reaches it is a build
            // failure and not a value read out of another partition's rows.
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "ChebyshevFit's region-A read is piece-indexed and the uniform grid "
                          "is not: its cells are interval-major and are summed at the degree "
                          "each was fitted at, through UniformFit's own locator. Read the grid "
                          "there instead of routing it to this family");
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    static double RegionBSeed(double x) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return ShippedRegionBSeed<kScheme>(x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return NarrowRegionBSeed<kScheme>(x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // The same fit as the shipped arm, by construction and not by fallback:
            // region B's seed is one fit over [kX0, kX1) at every granularity except
            // the narrow one, and the extended band is that same fit because nothing
            // amplifies its seed. The grid has no seed of its own to store, so the two
            // arms above and here name one table of the table's own rows.
            return ShippedRegionBSeed<kScheme>(x);
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    // The band's orders: one seed at the band's left edge, then one upward step
    // per order. The state is the step's, so a batch pays one division per
    // order rather than one run of the recurrence per order.
    //
    // The source is constructed at the argument, so its reciprocal is the
    // construction's too and the step never divides; the stepped argument still
    // arrives with each step because the step's arithmetic is the quotient's.
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        double f;
        double expx;
        double invx;

        explicit BandSource(double x) noexcept
            : f(RegionBExtendedSeed<kScheme>(x)),
              expx(0.5 * std::exp(-x)),
              invx(StepReciprocal<kForm>(x)) {}

        double Next(int l, double x) noexcept {
            return l == 0 ? f : (f = UpwardStep<kForm>(l, f, x, invx, expx));
        }
    };
};

// The uniform route: one fixed grid over the fitted domain, every order fitted
// independently, no order built from another.
//
// The table is interval-major - [interval][order][coefficient] - so the ladder one
// argument needs is a contiguous block, and each order is summed from its own
// coefficients by the same FitSum the derived routes read their pieces with. Nothing
// here recurs, which is the route's point: the derived routes build a ladder upward
// from a seed, a serial dependency chain over the orders that no amount of
// instruction-level parallelism shortens.
//
// The grid is fixed rather than derived, so locating an argument is one multiply and
// a truncation and not a scan of piece edges; what that costs is stored coefficients
// - one grid for all orders rather than a walk that spends pieces where the function
// needs them - and a floor on the work each order does, since every order carries
// its own degree where a recursion's tail orders cost a step each. Above kFlatHi the
// call falls through to the asymptotic every other route ends in, and the join needs
// no interpolation: kFlatHi is above kX1, so an argument the table does not serve is
// one the asymptotic already served.
//
// The reciprocal is taken from the stored width, never pasted beside it: the width
// belongs to the derivation and moves when the derivation moves, so a constant here
// would read the table one interval off the cell it was fitted on, with nothing
// reporting it. Taken this way the two move together or the static_assert below
// fails the build.
inline constexpr double kFlatPerUnit = 1.0 / detail::kFlatWidth;

static_assert(detail::kFlatWidth * kFlatPerUnit == 1.0,
              "the uniform table's grid is addressed by multiplying x by "
              "kFlatPerUnit and truncating, so that product must be exactly the "
              "reciprocal of the stored interval width");

static_assert(kFlatPerUnit * detail::kFlatHi ==
                  static_cast<double>(detail::kFlatIntervals),
              "the grid's own identity: the top of the table's domain scaled by the locate's "
              "constant must be exactly its interval count, so that an argument inside the "
              "domain locates no further than the last interval");

// Every interval's degree is checked against the read cap rather than assumed: the
// split Clenshaw scheme seeds its odd part at c[2m-1] and so needs an even degree of
// at least 4, and a degree above the cap is beyond the coefficients the interval
// stores. A table violating either would be summed into a different polynomial.
constexpr bool FlatDegreesCarried() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatIntervals); ++iv)
    {
        const int deg = detail::kFlatDegs[iv];

        if (deg < 4 || deg % 2 != 0 || deg > detail::kFlatReadCap)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatDegreesCarried(),
              "every interval of the uniform table must be fitted at an even degree between 4 "
              "and the read cap: an odd degree would drop the leading odd coefficient the split "
              "Clenshaw summation seeds from, and a degree above the cap is beyond the "
              "coefficients the interval carries");

static_assert(std::size(detail::kFlatOffsets) ==
                      static_cast<std::size_t>(detail::kFlatIntervals) + 1 &&
                  detail::kFlatOffsets[detail::kFlatIntervals] ==
                      static_cast<int>(std::size(detail::kFlatCoeffs)),
              "the uniform table's offsets must run one per interval and end at the stored "
              "count: the read rule reaches an interval's last coefficient through them, so an "
              "offset past the end reads a neighbouring fit's numbers");

template <EvalScheme kScheme>
struct UniformFit {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    // The uniform route reads one fixed grid and nothing else: it has no
    // piece-indexed fit and no region-B seed. These members exist only because
    // FitPolicy asks every fit for the whole contract the bodies are written
    // against; a body that reaches one is asking this route for a value it does not
    // have, and any answer would come from somewhere else entirely.
    //
    // They cannot refuse at compile time, which is what they ought to do: FitPolicy
    // asks for them by name and the concept check instantiates these bodies, so a
    // static_assert here refuses the route's own path instead of the bodies that are
    // not served by it. The refusal lives at the entries that do not carry the
    // partition, and these answer with a value no route can produce, so a body that
    // slips past that guard fails a comparison rather than returning plausible
    // numbers from elsewhere.
    //
    // An earlier revision delegated them to the narrow Chebyshev fit under a comment
    // claiming no body reached them; the all-n entry's region-A body does, and was
    // answered from a route the caller never named, with nothing reporting it.
    static double EvalPiece(std::size_t index, double t) noexcept {
        assert(!"the uniform fit is not piece-indexed: it is one fixed grid read "
                        "interval-major. A body reaching here is a body this partition "
                        "does not serve, and it must refuse the partition where it is named");
        return std::nan("") + static_cast<double>(index) + t;
    }

    static double RegionBSeed(double x) noexcept {
        assert(!"the uniform fit stores no region-B seed: region B is one of the walks "
                        "this route replaces with a fixed grid, not a region it reads");
        return std::nan("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double x) noexcept
        {
            assert(!"the uniform fit has no band source: it steps no upward recursion "
                            "from a stored seed");
            (void)x;
        }

        double Next(int l, double x) noexcept {
            assert(!"the uniform fit has no band to step along: every order is read "
                            "from its own coefficients, so there is no next one to build");
            return std::nan("") + static_cast<double>(l) + x;
        }
    };
};

// The rational member over the uniform grid: one numerator/denominator pair per
// interval of the same grid, fitted over that interval's own cell and read at the
// mapped argument the grid's own locator builds for it, in the stored form
// RationalFit and RationalFitNarrow read their rows with - the numerator ascending,
// then the denominator's q_1..q_k with q_0 held at 1.
//
// A fit of its own rather than a mode of UniformFit, because the two store different
// things: UniformFit holds a Chebyshev table and its monomial twin, this one holds
// pairs, and the scheme axis selects between two stored forms this family does not
// have. The partition is the grid's - its cells, not a cut this family makes - so
// the member brings one pair per interval to it.
//
// The degree is the pair's own, so a reader reaches an interval's rows through the
// interval and RationalUniformOrderAt below answers a call. This struct exists to
// satisfy FitPolicy's contract, and its three unserved members refuse exactly as
// UniformFit's do, for the reason stated there.
struct RationalFitUniform {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    static double EvalPiece(std::size_t index, double t) noexcept {
        assert(!"the uniform grid's rational member is not piece-indexed: it is one "
                        "pair per interval, reached through the interval's own offset. A "
                        "body reaching here is a body this partition does not serve, and it "
                        "must refuse the partition where it is named");
        return std::nan("") + static_cast<double>(index) + t;
    }

    static double RegionBSeed(double x) noexcept {
        assert(!"the uniform grid's rational member stores no region-B seed: region B is "
                        "one of the walks this route replaces with a fixed grid, not a "
                        "region it reads");
        return std::nan("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double x) noexcept
        {
            assert(!"the uniform grid's rational member has no band source: every order "
                            "is read from its own pair, so there is no seed to step from");
            (void)x;
        }

        double Next(int l, double x) noexcept {
            assert(!"the uniform grid's rational member has no band to step along: every "
                            "order is read from its own pair, so there is no next one to build");
            return std::nan("") + static_cast<double>(l) + x;
        }
    };
};

// Every interval's pair is checked against the read rule rather than assumed
// from it. A row is reached at offsets[iv] + l * stored[iv], the numerator's
// m + 1 coefficients and the denominator's k after them, so a stored count that
// is not m + 1 + k, a pair with no denominator term, or a block that is not one
// row per order reads a neighbouring interval's coefficients as this one's, with
// nothing downstream reporting it.
constexpr bool FlatRatPairsCarried() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatRatIntervals); ++iv)
    {
        const int m = detail::kFlatRatNumDeg[iv];
        const int k = detail::kFlatRatDenDeg[iv];

        if (m < 1 || k < 1 || detail::kFlatRatStored[iv] != m + 1 + k ||
            detail::kFlatRatOffsets[iv + 1] - detail::kFlatRatOffsets[iv] !=
                (kMaxBoysOrder + 1) * detail::kFlatRatStored[iv])
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatRatPairsCarried(),
              "the rational member over the uniform grid must carry, at every interval, a "
              "numerator, a denominator with a non-constant term and one row per order at the "
              "stride its stored count states: the reader reaches a row at "
              "offset + order * stored, so anything else reads another interval's pair");

// Where an argument sits on the uniform grid: the interval it falls in, that
// interval's first coefficient, and the argument mapped into it. One copy,
// because the ladder and the single order must agree on it to the bit: two
// spellings of the same index arithmetic would be two chances to disagree about
// which interval an argument falls in, with nothing reporting it.
//
// The interval is carried and not only its offset, because the table's degree is
// the interval's own: an order's coefficients are reached at the stride
// kFlatDegs[iv] + 1, and the group reader takes the same stride for the four
// orders it gathers, well defined because one argument puts all four in one
// interval.
struct FlatPoint {
    std::size_t iv;    ///< the interval the argument falls in
    std::size_t block; ///< kFlatOffsets[iv], this interval's order-0 coefficient
    double t;          ///< the argument mapped into [-1, 1) on that interval
};

inline FlatPoint FlatLocate(double x) noexcept {
    const double u = x * kFlatPerUnit;
    int iv = static_cast<int>(u);

    if (iv > detail::kFlatIntervals - 1)
    {
        iv = detail::kFlatIntervals - 1;
    }

    const std::size_t index = static_cast<std::size_t>(iv);

    return FlatPoint{index,
                     static_cast<std::size_t>(detail::kFlatOffsets[index]),
                     2.0 * (u - static_cast<double>(iv)) - 1.0};
}

/// One order off the rational member over the uniform grid: this interval's row
/// at this order, read by the same steps RationalFit::EvalPiece reads a piece's
/// row with - the numerator by Horner, then the denominator's q_1..q_k with its
/// constant term held at 1, then one division.
inline double RationalUniformOrderAt(const FlatPoint& at, int l) noexcept {
    const std::size_t stored = static_cast<std::size_t>(detail::kFlatRatStored[at.iv]);
    const double* c = detail::kFlatRatCoeffs.data() +
                      static_cast<std::size_t>(detail::kFlatRatOffsets[at.iv]) +
                      static_cast<std::size_t>(l) * stored;
    const int m = detail::kFlatRatNumDeg[at.iv];
    const int k = detail::kFlatRatDenDeg[at.iv];
    double num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, at.t, c[j]);
    }

    double den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, at.t, c[m + j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, at.t, 1.0);
}

/// One order off the uniform grid, at the route the policy names: the Chebyshev
/// member's cell at that cell's own degree, or the rational member's pair at that
/// interval's own degrees. Both are reached through one `FlatPoint`, so the dispatch
/// is the whole of the difference.
template <typename Policy>
double UniformOrderAt(const FlatPoint& at, int l) noexcept {
    if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        return RationalUniformOrderAt(at, l);
    }
    else
    {
        const std::size_t base =
            at.block + static_cast<std::size_t>(l) *
                           static_cast<std::size_t>(detail::kFlatDegs[at.iv] + 1);

        return FitSum<Policy::kScheme, backend::ScalarFp64>(
            detail::kFlatCoeffs.data() + base, detail::kFlatMonoCoeffs.data() + base,
            detail::kFlatDegs[at.iv], at.t);
    }
}

// One argument's ladder off the uniform table: every order read from its own
// coefficients, none from another's.
template <typename Policy>
void UniformAllOrders(int nmax, double x, double* out) noexcept {
    const FlatPoint at = FlatLocate(x);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = UniformOrderAt<Policy>(at, l);
    }
}

// One order at one argument, off the same table and the same index: a call that
// needs one order pays for one order. The ladder form computes every order it is
// asked for, which is the cost the option probe measures against the recursing
// routes - they seed once and step, so their tail orders are nearly free while
// this one's are not.
template <typename Policy>
double UniformSingleOrder(int n, double x) noexcept {
    return UniformOrderAt<Policy>(FlatLocate(x), n);
}

// The rational minimax route: a numerator/denominator pair per piece, read by Horner
// in the same mapped argument, the denominator stored as q_1..q_k with q_0 held at
// 1. An alternative to the shipped route rather than a replacement: the same pieces,
// intervals and mapping, the other scheme.
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
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFit>(l, x); }
    };
};

// The rational route over the narrow partition: the family's own degree pair per
// narrow piece, read at that piece's own interval and mapped argument, in the same
// stored form as the shipped member - the numerator ascending, then the denominator's
// q_1..q_k with q_0 held at 1.
//
// The partition is not the family's: it is the narrow Chebyshev partition's cut of
// both regions (see FitGranularity), so the member brings only the pairs - region A's
// one per kNarrowAPieces row, region B's one per narrow piece at the mapped argument
// the Chebyshev seed on that piece uses, so the two routes over a piece read one t.
//
// Every pair was accepted at the criterion the shipped member's were - the bare
// delivered error of the stored pair in the kernel's own arithmetic, under the 3e-14
// bar in region A and the 5e-14 one in region B - read at BOTH multiply-add routes
// with the worse taken: a bound taken at one route is not a bound on the other's
// evaluation.
struct RationalFitNarrow {
    // The narrow cut of region A; the pieces, their intervals and the piece
    // lookup are the Chebyshev narrow partition's own.
    using Partition = NarrowRegionAPartition;

    // The route hands each order over at its own region-A boundary, as the
    // shipped rational family does: below it the shipped lane's value is the one
    // that order is documented at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    static double EvalPiece(std::size_t index, double t) noexcept {
        const double* c = detail::kNarrowRatACoeffs.data() + detail::kNarrowRatAOffset[index];
        const int m = detail::kNarrowRatANumDeg[index];
        const int k = detail::kNarrowRatADenDeg[index];
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

    // The region-B seed from the narrow piece the argument falls in, at that
    // piece's own pair and mapped argument.
    static double RegionBSeed(double x) noexcept {
        const std::size_t index = static_cast<std::size_t>(NarrowBPieceOf(x));
        const double a = detail::kNarrowBEdges[index];
        const double b = detail::kNarrowBEdges[index + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;
        const double* c = detail::kNarrowRatBCoeffs.data() + detail::kNarrowRatBOffset[index];
        const int m = detail::kNarrowRatBNumDeg[index];
        const int k = detail::kNarrowRatBDenDeg[index];
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

    // Each of the band's orders is its own fit, as it is on the shipped member.
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFitNarrow>(l, x); }
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

// The same piece read at a cut pair over the double lane's narrow table; see
// RationalFitNarrow::EvalPiece for the full pair this cuts and
// RationalPieceAtCut for the shipped partition's reading of the same shape.
inline double RationalPieceNarrowAtCut(std::size_t index,
                                       int numDeg,
                                       int denDeg,
                                       double t) noexcept {
    const double* c = detail::kNarrowRatACoeffs.data() + detail::kNarrowRatAOffset[index];
    const int storedNumDeg = detail::kNarrowRatANumDeg[index];
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

// Region B's seed at the partition a policy names: the Chebyshev seed at the
// policy's scheme and granularity, not the route's fit. The bodies that call it
// read the Chebyshev family's seed by construction - they are region B's own
// bodies - and it exists so that a region-B seed read from a policy is the
// partition that policy names rather than always the shipped one.
//
// It is a read that resolves a policy's partition through a fit family, so it asks
// whether that family answers the partition the policy names - the question every
// uniform-partition guard in this file is asking - and refuses where it does not.
// This is the barrier and not a restatement of one elsewhere: a body that reaches
// the derived family from a policy reaches it here, and a body written later that
// does the same fails to compile where it reads rather than where somebody
// remembered to look.
template <EvalPolicyLike Policy>
inline double PolicyRegionBSeed(double x) noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this read resolves the policy's partition through the Chebyshev family, which "
                  "carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one: a policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Give this body a "
                  "branch that reads the grid's own table, or route a policy naming the grid to "
                  "one that has it");
    return ChebyshevFit<Policy::kScheme, Policy::kGranularity>::RegionBSeed(x);
}

// Region A's per-order value at the partition a policy names, for the bodies that
// read the Chebyshev route's region-A pieces directly rather than through a policy's
// fit. It is the Chebyshev partition for the reason PolicyRegionBSeed is - the piece
// tables are the Chebyshev family's - and the policy's granularity, so naming the
// narrow partition reaches every region-A read in a policy-driven body, not only the
// ones a route's own fit answers. The rational routes are read here at the policy's
// partition as well: below each route's fits-first crossover the value a policy
// answers with is the Chebyshev lane's, which is the lane that region is documented
// at on both partitions.
//
// **This read carries the same assertion PolicyRegionBSeed carries.** It was the one
// read of the pair that could not: it is reached from the `x < kX0` block of SingleOrder
// and of AllOrdersBody, and those blocks are instantiated for a policy naming the grid
// even though the grid's own branch has returned for every argument they cover - below
// kFlatHi the grid's body answers, and kFlatHi is above kX1 and so above kX0. An
// assertion here refused those two served bodies for a read they cannot take, which is
// a refusal of something that compiles and is served. The blocks are now conditionally
// dead, on this same question of this same family, so the barrier stands here too: what
// keeps them dead is the ordering asserted above AllOrdersBody, and what keeps a read
// from being answered by another partition's tables is this assertion, at the read.
template <EvalPolicyLike Policy>
inline double PolicyRegionAValue(int order, double x) noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this read resolves the policy's partition through the Chebyshev family, "
                  "which carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one: a policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Give this body a "
                  "branch that reads the grid's own table, or route a policy naming the grid "
                  "to one that has it");
    return RegionAValue<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(order, x);
}

// ---------------------------------------------------------------------------
// Scalar float lane helpers
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

// Narrow-partition piece lookup; see FindPieceF32: the narrow pieces are cut per
// order like the shipped ones, so the walk is the same over the narrow
// piece-start table.
inline const detail::f32::OrderPiece& FindNarrowPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kNarrowAPieceStartF32[order];
    const int last = detail::f32::kNarrowAPieceStartF32[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kNarrowAPiecesF32[i].b)
        {
            return detail::f32::kNarrowAPiecesF32[i];
        }
    }

    return detail::f32::kNarrowAPiecesF32[last - 1];
}

// Which narrow region-B piece an argument falls in; see NarrowBPieceOf, whose
// walk this is over the float lane's own edges.
inline int NarrowBPieceF32(float x) noexcept {
    const float* edges = detail::f32::kNarrowBEdgesF32.data();
    const int pieces = detail::f32::kNarrowBPiecesF32;

    for (int i = 0; i + 1 < pieces; ++i)
    {
        if (x < edges[i + 1])
        {
            return i;
        }
    }

    return pieces - 1;
}

// Which rational narrow region-A piece an argument falls in; see FindPieceF32.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::f32::RatPiece& FindNarrowRatPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kNarrowRatAPieceStartF32[order];
    const int last = detail::f32::kNarrowRatAPieceStartF32[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kNarrowRatAPiecesF32[i].b)
        {
            return detail::f32::kNarrowRatAPiecesF32[i];
        }
    }

    return detail::f32::kNarrowRatAPiecesF32[last - 1];
}

// Which narrow rational region-B piece an argument falls in.
inline const detail::f32::RatPiece& FindNarrowRatBPieceF32(float x) noexcept {
    const detail::f32::RatPiece* pieces = detail::f32::kNarrowRatBPiecesF32.data();
    const int count = detail::f32::kNarrowRatBPiecesCountF32;

    for (int i = 0; i + 2 <= count; ++i)
    {
        if (x < pieces[i].b)
        {
            return pieces[i];
        }
    }

    return pieces[count - 1];
}

// Float-lane region-A seed; see ChebyshevValue. The scheme picks which of the
// two parallel coefficient tables - the Chebyshev one ClenshawSplit reads or the
// monomial one HornerMono reads - the piece is summed from; the pieces, their
// intervals and their degrees are the same under either.
// Forced inline: the m = 1 F32-single engine must keep the full-accuracy code
// shape (piece scan inlined), which MSVC's size heuristic drops with two call
// sites (the kFloat and the kFp16 budget instantiations). Semantics are
// unaffected - inline never changes the bit-identity pin.
// __forceinline is MSVC-only; GCC/Clang spell the same intent with
// always_inline (plain inline is a hint there, not a requirement).
#if defined(_MSC_VER)
#define BoysForceInline __forceinline
#else
#define BoysForceInline inline __attribute__((always_inline))
#endif
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
BoysForceInline float ChebyshevValueF32(int order, float x) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
        const float* c = detail::f32::kCoeffs.data() + piece.offset;
        const float* m = detail::f32::kMonoCoeffs.data() + piece.offset;
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        return FitSum<kScheme, backend::ScalarFp32>(c, m, piece.deg, t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::OrderPiece& piece = FindNarrowPieceF32(order, x);
        const float* c = detail::f32::kNarrowACoeffsF32.data() + piece.offset;
        const float* m = detail::f32::kNarrowAMonoCoeffsF32.data() + piece.offset;
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        return FitSum<kScheme, backend::ScalarFp32>(c, m, piece.deg, t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A of a piece-indexed family has no uniform table: the grid's cells
        // are interval-major and read at their own degrees. The arm refuses rather
        // than reading the narrow pieces under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this is the piece-indexed region-A read, and the uniform grid has no "
                      "piece-indexed table: its cells are summed interval-major, at the degree "
                      "each was fitted at, by the grid's own locator. Route the grid there "
                      "rather than to this family");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

#undef BoysForceInline

// ---------------------------------------------------------------------------
// The float lane's uniform table
// ---------------------------------------------------------------------------
// The uniform partition's grid, read the way the emitted table names it: one degree
// and one offset per interval, so the coefficient of order l and term k is
//
//   kFlatCoeffsF32[kFlatOffsetsF32[iv] + l * (kFlatDegsF32[iv] + 1) + k]
//
// and the interval an argument falls in is iv = floor(x * kFlatPerUnitF32).
//
// The degrees are the interval's own and not one stride for the whole table, each
// cell having been given the smallest admissible even degree its proved truncation
// bound holds it to, so a walk with a single stride would read a neighbouring cell's
// polynomial with nothing reporting it. kFlatOffsetsF32 is the same fact from the
// other end - what makes a per-interval degree indexable - and the two extra array
// reads it costs are the whole run-time price of the shape.
//
// This is the lane's own grid and not the double lane's: derived from this lane's
// bound, format and read cap (the emitted kFlatReadCapF32), so its width, interval
// count and degrees are the double lane's only by coincidence, and a caller reading
// the double lane's grid under this lane's name would be handed a table fitted to a
// bound this lane's arithmetic cannot reach.
//
// The width's reciprocal is exact in binary32, which is what the derivation takes a
// dyadic width for: the locate is one multiply and a truncation, so the interval an
// argument falls in has to be a property of the argument and not of a rounding. The
// device mirror asserts the same identity from the other side
// (boys_cuda_arithmetic.hpp).
inline constexpr float kFlatPerUnitF32 = 1.0f / detail::f32::kFlatWidthF32;

static_assert(detail::f32::kFlatWidthF32 * kFlatPerUnitF32 == 1.0f,
              "the locate is one multiply by this constant and a truncation, so the product must "
              "be exactly the width's reciprocal in binary32: a width whose reciprocal does not "
              "round back reads the table one interval off the cell it was fitted on, silently");

static_assert(kFlatPerUnitF32 * detail::f32::kFlatHiF32 ==
                  static_cast<float>(detail::f32::kFlatIntervalsF32),
              "the grid's own identity: the top of the table's domain scaled by the locate's "
              "constant must be exactly its interval count, so that an argument inside the "
              "domain locates no further than the last interval");

// Every interval's degree is checked against the read cap rather than assumed: the
// split Clenshaw scheme seeds its odd part at c[2m-1] and so needs an even degree of
// at least 4, and a degree above the cap is beyond the coefficients the interval
// stores. A table violating either would be summed into a different polynomial.
constexpr bool FlatDegreesCarriedF32() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::f32::kFlatIntervalsF32); ++iv)
    {
        const int deg = detail::f32::kFlatDegsF32[iv];

        if (deg < 4 || deg % 2 != 0 || deg > detail::f32::kFlatReadCapF32)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatDegreesCarriedF32(),
              "every interval of the float uniform table must be fitted at an even degree between "
              "4 and the read cap: an odd degree would drop the leading odd coefficient the split "
              "Clenshaw summation seeds from, and a degree above the cap is beyond the "
              "coefficients the interval carries");

/// Where an argument sits on the float uniform grid: the interval it falls in,
/// that interval's first coefficient, and the argument mapped into it.
///
/// One copy, as the double lane's FlatLocate is one copy: the ladder and the
/// single-order read must agree on the interval to the bit, and two spellings of
/// the same index arithmetic would be two chances for them to disagree about
/// which interval an argument is in, with nothing reporting it.
struct FlatPointF32 {
    std::size_t iv;    ///< the interval the argument falls in
    std::size_t block; ///< kFlatOffsetsF32[iv], this interval's order-0 coefficient
    float t;           ///< the argument mapped into [-1, 1) on that interval
};

inline FlatPointF32 FlatLocateF32(float x) noexcept {
    const float u = x * kFlatPerUnitF32;
    int iv = static_cast<int>(u);

    if (iv > detail::f32::kFlatIntervalsF32 - 1)
    {
        iv = detail::f32::kFlatIntervalsF32 - 1;
    }

    const std::size_t index = static_cast<std::size_t>(iv);

    return FlatPointF32{index,
                        static_cast<std::size_t>(detail::f32::kFlatOffsetsF32[index]),
                        2.0f * (u - static_cast<float>(iv)) - 1.0f};
}

// The rational member over the FLOAT lane's uniform grid: one numerator/denominator
// pair per interval of the same grid, fitted over that interval's own cell and read
// at the mapped argument FlatLocateF32 builds for it, in the stored form
// RationalFit32 and RationalFitNarrow read their rows with - the numerator
// ascending, then the denominator's q_1..q_k with q_0 held at 1.
//
// The double member's shape at this lane's tables, and a fit of its own rather than
// a reading of that one: the double lane's pairs are stored in binary64 over the
// double grid's cells, this lane's in binary32 over its own - a different width,
// interval count and cells. Like the shipped and narrow members it is one fit under
// either scheme: its coefficients are a monomial numerator and denominator with no
// Chebyshev form to sum.
//
// Its three unserved members refuse exactly as RationalFitUniform's do, for the
// reason stated there.
struct RationalFitUniformF32 {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    static float EvalPiece(std::size_t index, float t) noexcept {
        assert(!"the float uniform grid's rational member is not piece-indexed: it is "
                        "one pair per interval, reached through the interval's own "
                        "offset. A body reaching here is a body this partition does not "
                        "serve, and it must refuse the partition where it is named");
        return std::nanf("") + static_cast<float>(index) + t;
    }

    static float RegionBSeed(float x, int /*order*/) noexcept {
        assert(!"the float uniform grid's rational member stores no region-B seed: "
                        "region B is one of the walks this route replaces with a fixed "
                        "grid, not a region it reads");
        return std::nanf("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(float x) noexcept
        {
            assert(!"the float uniform grid's rational member has no band source: every "
                            "order is read from its own pair, so there is no seed to step "
                            "from");
            (void)x;
        }

        float Next(int l, float x) noexcept {
            assert(!"the float uniform grid's rational member has no band to step along: "
                            "every order is read from its own pair, so there is no next "
                            "one to build");
            return std::nanf("") + static_cast<float>(l) + x;
        }
    };
};

// Every interval's pair is checked against the read rule here rather than
// assumed from it, on the double member's reading: a row is reached at
// offsets[iv] + l * stored[iv], with the numerator's m + 1 coefficients and the
// denominator's k after them, so a stored count that is not m + 1 + k, a pair
// with no denominator term to divide by, or a block that is not one row per
// order reads a neighbouring interval's coefficients as though they were this
// one's - and nothing downstream would report it.
constexpr bool FlatRatPairsCarriedF32() noexcept
{
    for (std::size_t iv = 0;
         iv < static_cast<std::size_t>(detail::f32::kFlatRatIntervalsF32);
         ++iv)
    {
        const int m = detail::f32::kFlatRatNumDegF32[iv];
        const int k = detail::f32::kFlatRatDenDegF32[iv];

        if (m < 1 || k < 1 || detail::f32::kFlatRatStoredF32[iv] != m + 1 + k ||
            detail::f32::kFlatRatOffsetsF32[iv + 1] -
                    detail::f32::kFlatRatOffsetsF32[iv] !=
                (kMaxBoysOrder + 1) * detail::f32::kFlatRatStoredF32[iv])
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatRatPairsCarriedF32(),
              "the float rational member over the uniform grid must carry, at every "
              "interval, a numerator, a denominator with a non-constant term and one row "
              "per order at the stride its stored count states: the reader reaches a row "
              "at offset + order * stored, so anything else reads another interval's "
              "pair");

/// One order off the float grid's rational member: this interval's row at this
/// order, read by the same steps RationalFit32::EvalOrder reads a shipped
/// piece's row with - the numerator by Horner, then the denominator's q_1..q_k
/// with its constant term held at 1, then one division - in the lane's own
/// width, which is the arithmetic the pairs were fitted and certified in.
inline float RationalUniformOrderAtF32(const FlatPointF32& at, int l) noexcept {
    const std::size_t stored =
        static_cast<std::size_t>(detail::f32::kFlatRatStoredF32[at.iv]);
    const float* c = detail::f32::kFlatRatCoeffsF32.data() +
                     static_cast<std::size_t>(detail::f32::kFlatRatOffsetsF32[at.iv]) +
                     static_cast<std::size_t>(l) * stored;
    const int m = detail::f32::kFlatRatNumDegF32[at.iv];
    const int k = detail::f32::kFlatRatDenDegF32[at.iv];
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, at.t, c[j]);
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, at.t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, at.t, 1.0f);
}

/// One order off the float uniform grid, at the route the policy names: the
/// Chebyshev member's cell, summed at that cell's own degree, or the rational
/// member's pair, read at that interval's own degrees. The two are reached
/// through one `FlatPointF32` - the interval the argument fell in and the mapped
/// argument into it are the locator's, and each member finds its own rows
/// through them - so the dispatch is the whole of the difference.
template <typename Policy>
float UniformOrderAtF32(const FlatPointF32& at, int l) noexcept {
    if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        return RationalUniformOrderAtF32(at, l);
    }
    else
    {
        const int deg = detail::f32::kFlatDegsF32[at.iv];
        const std::size_t base =
            at.block + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        return FitSum<Policy::kScheme, backend::ScalarFp32>(
            detail::f32::kFlatCoeffsF32.data() + base,
            detail::f32::kFlatMonoCoeffsF32.data() + base,
            deg,
            at.t);
    }
}

/// One argument's ladder off the float uniform table: every order read from its
/// own coefficients, none from another's, so a lane of the across-orders packed
/// entry carries the per-order value the arguments axis carries - which is the
/// identity that entry's own suite asserts.
template <typename Policy>
void UniformAllOrdersF32(int nmax, float x, float* out) noexcept {
    const FlatPointF32 at = FlatLocateF32(x);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = UniformOrderAtF32<Policy>(at, l);
    }
}

/// One order at one argument, off the same table and the same index.
///
/// This is where the route is strongest rather than weakest: a call that needs
/// one order pays for one order, and the ladder form above cannot know that only
/// one is wanted.
template <typename Policy>
float UniformSingleOrderF32(int n, float x) noexcept {
    return UniformOrderAtF32<Policy>(FlatLocateF32(x), n);
}

// This lane's shipped region-B seed: one polynomial over [kX0, kX1), the fit the
// shipped and the uniform partition read between them; see ShippedRegionBSeed for why
// the two coincide.
template <EvalScheme kScheme>
inline float ShippedRegionBSeedF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;

    return FitSum<kScheme, backend::ScalarFp32>(detail::f32::kBcoeffs.data(),
                                                detail::f32::kMonoBcoeffs.data(),
                                                detail::f32::kBDeg,
                                                t);
}

// Float-lane region-B seed; see RegionBSeed.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
inline float RegionBSeedF32(float x) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedF32<kScheme>(x);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const int index = NarrowBPieceF32(x);
        const float a = detail::f32::kNarrowBEdgesF32[static_cast<std::size_t>(index)];
        const float b = detail::f32::kNarrowBEdgesF32[static_cast<std::size_t>(index) + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const float* c = detail::f32::kNarrowBcoeffsF32.data()
                         + static_cast<std::size_t>(index)
                               * static_cast<std::size_t>(detail::f32::kNarrowBDegF32 + 1);
        const float* m = detail::f32::kNarrowBMonoCoeffsF32.data()
                         + static_cast<std::size_t>(index)
                               * static_cast<std::size_t>(detail::f32::kNarrowBDegF32 + 1);
        return FitSum<kScheme, backend::ScalarFp32>(c, m, detail::f32::kNarrowBDegF32, t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region
        // B's seed is one fit over [kX0, kX1) at every granularity except the narrow
        // one, and the extended band is that same fit because nothing amplifies its
        // seed. This lane is no exception, and the grid has no seed of its own to store.
        return ShippedRegionBSeedF32<kScheme>(x);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
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
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::OrderPiece& piece = FindPiece(order, x);
        const std::ptrdiff_t index = &piece - detail::kPieces.data();
        const int deg = degrees[static_cast<std::size_t>(index)];
        const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
        return FitSum<kScheme, backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                    detail::kMonoCoeffs.data() + piece.offset,
                                                    deg,
                                                    t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
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
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A is where the three partitions differ, and this read is piece-indexed
        // off a per-order effective-degree table: the grid has neither. Its cells are
        // interval-major, summed at the degree each was fitted at and located by index
        // arithmetic rather than by a piece scan, so the arm refuses rather than reading
        // the narrow pieces under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this region-A read takes a piece index and a per-order degree, and the "
                      "uniform grid has neither: its cells are read interval-major at the "
                      "degree each was fitted at. Route the grid to its own read rather than "
                      "here");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The shipped region-B seed at a cut degree: one polynomial over [kX0, kX1), read at
// the degree the caller's table certifies for the peak order. The shipped and the
// uniform partition read this fit between them - see ShippedRegionBSeed - so the two
// arms of RegionBSeedWithDegrees below reach the table through here.
template <EvalScheme kScheme, typename DegreesArray>
inline double ShippedRegionBSeedWithDegrees(double x,
                                            const DegreesArray& degrees,
                                            int peakOrder) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;

    return FitSum<kScheme, backend::ScalarFp64>(
        detail::kBcoeffs.data(),
        detail::kMonoBcoeffs.data(),
        degrees[static_cast<std::size_t>(peakOrder)],
        t);
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
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedWithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
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
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region
        // B's seed is one fit over [kX0, kX1) at every granularity except the narrow
        // one, so the degree this read is cut at is the shipped table's own. The arm is
        // written rather than left to an `else`, so a reader sees that the two coincide
        // rather than inferring it from what one arm's `else` happens to mean.
        return ShippedRegionBSeedWithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The effective-degree tables a policy's rung reads, over the partition the
// policy names. The criterion is the same one either way; the table it is
// measured against is the partition's own, which is what makes a rung a
// reading of the partition the caller chose rather than of the shipped one.
//
// Both bodies ask the shared question first, and each then names its own two or three
// partitions rather than resolving the rest through an `else`: which table a partition
// reads is a fact about the partition, and a partition the arms do not name must fail
// the read rather than be answered out of the last arm's table. The callers today are
// the two rung reads below, and the assertions hold for a body written later that
// reaches a table directly.
template <EvalPolicyLike Policy, BoysRole kRole>
constexpr auto RegionADegreeTableOf() noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this table is what a rung cuts the Chebyshev family's region-A fit with, and "
                  "a policy naming the uniform grid has no rung of it: the grid's cells are "
                  "read interval-major at the degrees they were fitted at, so there is no "
                  "piece-indexed degree of this shape to cut them at");
    if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
    {
        return RegionADegrees<kRole, SchemeTailBasis<Policy::kScheme>()>();
    }
    else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
    {
        return NarrowRegionADegrees<
                                    kRole,
                                    SchemeTailBasis<Policy::kScheme>()>();
    }
    else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        // Refused by the assertion above, and refused again here so that the arm stands
        // where the partition is named: the grid's degree table is not this family's,
        // and answering with the narrow rows under the grid's name is the substitution
        // every guard in this file exists to stop.
        static_assert(kAlwaysFalse<GranularityTag<Policy::kGranularity>>,
                      "region A is where the three partitions differ, and the uniform grid has "
                      "no piece-indexed fit in this family to cut: its cells carry the degree "
                      "they were fitted at. Read the grid at its own table rather than through "
                      "this one");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<Policy::kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

template <EvalPolicyLike Policy, BoysRole kRole>
constexpr auto RegionBDegreeTableOf() noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this table is what a rung cuts the Chebyshev family's region-B fit with, and "
                  "a policy naming the uniform grid has no rung of it: the grid reads its own "
                  "cells at the degrees they were fitted at. The degrees below are the shipped "
                  "partition's, which the grid's region-B read shares by construction");
    if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
    {
        return RegionBDegrees<kRole, SchemeTailBasis<Policy::kScheme>()>();
    }
    else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
    {
        return NarrowRegionBDegrees<
                                    kRole,
                                    SchemeTailBasis<Policy::kScheme>()>();
    }
    else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region
        // B's seed is one fit over [kX0, kX1) at every granularity except the narrow
        // one, so the table a rung would cut it with is the shipped table's. The arm is
        // written rather than left to an `else`; the assertion above is what keeps a
        // policy naming the grid from reaching a rung at all.
        return RegionBDegrees<kRole, SchemeTailBasis<Policy::kScheme>()>();
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<Policy::kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
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

// The float lane's narrow partition at a cut degree: the same reads, over this
// lane's own narrow pieces and at the degree its own table certifies.

template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
float ChebyshevValueF32WithDegrees(int order, float x, const DegreesArray& degrees) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
        const std::ptrdiff_t index = &piece - detail::f32::kPieces.data();
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        return FitSum<kScheme, backend::ScalarFp32>(detail::f32::kCoeffs.data() + piece.offset,
                                                    detail::f32::kMonoCoeffs.data() + piece.offset,
                                                    degrees[static_cast<std::size_t>(index)],
                                                    t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::OrderPiece& piece = FindNarrowPieceF32(order, x);
        const std::ptrdiff_t index = &piece - detail::f32::kNarrowAPiecesF32.data();
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        return FitSum<kScheme, backend::ScalarFp32>(
            detail::f32::kNarrowACoeffsF32.data() + piece.offset,
            detail::f32::kNarrowAMonoCoeffsF32.data() + piece.offset,
            degrees[static_cast<std::size_t>(index)],
            t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A is where the three partitions differ, and this read is piece-indexed
        // off a per-order effective-degree table: the grid has neither. Its cells are
        // interval-major, summed at the degree each was fitted at, so the arm refuses
        // rather than reading the narrow pieces under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this region-A read takes a piece index and a per-order degree, and the "
                      "uniform grid has neither: its cells are read interval-major at the "
                      "degree each was fitted at. Route the grid to its own read rather than "
                      "here");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The region-B seed at the degree the rung certifies for the peak order the
// caller is about to reach, over the partition named - the float lane's reading
// of the double lane's pair above, with the same two shapes: one polynomial
// over the whole region for the shipped partition, one polynomial per piece for
// the narrow one, whose table carries the order beside the piece.
// This lane's shipped region-B seed at a cut degree; the counterpart of
// ShippedRegionBSeedWithDegrees, and the fit the shipped and the uniform partition read
// between them.
template <EvalScheme kScheme, typename DegreesArray>
inline float ShippedRegionBSeedF32WithDegrees(float x,
                                              const DegreesArray& degrees,
                                              int peakOrder) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;

    return FitSum<kScheme, backend::ScalarFp32>(
        detail::f32::kBcoeffs.data(),
        detail::f32::kMonoBcoeffs.data(),
        degrees[static_cast<std::size_t>(peakOrder)],
        t);
}

template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
inline float RegionBSeedF32WithDegrees(float x,
                                       const DegreesArray& degrees,
                                       int peakOrder) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedF32WithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const std::size_t piece = static_cast<std::size_t>(NarrowBPieceF32(x));
        const float a = detail::f32::kNarrowBEdgesF32[piece];
        const float b = detail::f32::kNarrowBEdgesF32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const std::size_t offset =
            piece * (static_cast<std::size_t>(detail::f32::kNarrowBDegF32) + 1);
        return FitSum<kScheme, backend::ScalarFp32>(
            detail::f32::kNarrowBcoeffsF32.data() + offset,
            detail::f32::kNarrowBMonoCoeffsF32.data() + offset,
            degrees[piece * (static_cast<std::size_t>(kMaxOrder) + 1)
                    + static_cast<std::size_t>(peakOrder)],
            t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region
        // B's seed is one fit over [kX0, kX1) at every granularity except the narrow
        // one, so the degree this read is cut at is the shipped table's own. See
        // ShippedRegionBSeed for why the two coincide.
        return ShippedRegionBSeedF32WithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}




// ---------------------------------------------------------------------------
// The float lane's fit routes
// ---------------------------------------------------------------------------
// A float-lane single-order call reads a coefficient in exactly two places:
// the region-A seed and the region-B seed. This names which pair of fits
// those two are. They are alternatives rather than rungs of one design:
// naming the rational one changes the coefficients those two intervals are
// evaluated from and nothing else.
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
//
// The partition names the table the index belongs to, and the caller has
// already looked the piece up in that partition and mapped t in it, so the two
// readings differ in the table alone.
// The summation itself, over the row the partition's own table named: the cut degree
// and the stored numerator degree are all the body reads besides the coefficients.
inline float RationalPieceF32AtCutBody(const float* c,
                                       int storedNumDeg,
                                       int numDeg,
                                       int denDeg,
                                       float t) noexcept {
    float num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

template <FitGranularity kGranularity = kDefaultFitGranularity>
inline float RationalPieceF32AtCut(std::size_t index, int numDeg, int denDeg, float t) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::f32::RatPiece& piece = detail::f32::kRatAPieces[index];
        return RationalPieceF32AtCutBody(detail::f32::kRatACoeffs.data() + piece.offset,
                                         piece.numdeg,
                                         numDeg,
                                         denDeg,
                                         t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::RatPiece& piece = detail::f32::kNarrowRatAPiecesF32[index];
        return RationalPieceF32AtCutBody(detail::f32::kNarrowRatACoeffsF32.data() + piece.offset,
                                         piece.numdeg,
                                         numDeg,
                                         denDeg,
                                         t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A of this family, where the three partitions differ and this one has no
        // row of this shape: the grid's numerator and denominator are one pair per
        // interval, reached by the interval's own offset, so the arm refuses rather than
        // summing the narrow pieces' rows under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "the rational family's piece-indexed region-A read has no uniform table: "
                      "the grid's numerator and denominator are one pair per interval, reached "
                      "by the interval's own offset. Read the grid through its own order entry "
                      "rather than here");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The region-B seed of this lane at a cut pair; same reading as
// RegionBSeedRationalF32. The stored coefficients are q_1..q_k with the
// constant term held at 1, and each descent below reads the next one down.
// The shipped region holds one pair over the whole interval, the narrow one a
// pair per piece, so the narrow branch takes the row's own stored degrees and
// the cut above them.
// This lane's shipped region-B pair at a cut; the fit the shipped and the uniform
// partition read between them (see ShippedRegionBSeed).
inline float RationalSeedF32ShippedAtCut(int numDeg, int denDeg, float t) noexcept {
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

template <FitGranularity kGranularity = kDefaultFitGranularity>
inline float RationalSeedF32AtCut(std::size_t index, int numDeg, int denDeg, float t) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return RationalSeedF32ShippedAtCut(numDeg, denDeg, t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::RatPiece& piece = detail::f32::kNarrowRatBPiecesF32[index];
        const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
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
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same pair as the shipped arm, by construction and not by fallback: region
        // B's seed is one fit over [kX0, kX1) at every granularity except the narrow
        // one. The arm is written rather than left to an `else`, so the sharing is
        // visible here rather than inferred.
        return RationalSeedF32ShippedAtCut(numDeg, denDeg, t);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// One narrow rational piece of this lane at a cut pair; the degree pair is the
// rung's and the piece, its interval and its stored degrees are the narrow
// partition's own. The denominator's coefficients sit above the *stored*
// numerator here as they do in the shipped table, so q_j is read at the
// piece's full numerator degree's offset.
inline float RationalPieceNarrowF32AtCut(std::size_t index,
                                         int numDeg,
                                         int denDeg,
                                         float t) noexcept {
    const detail::f32::RatPiece& piece = detail::f32::kNarrowRatAPiecesF32[index];
    const float* c = detail::f32::kNarrowRatACoeffsF32.data() + piece.offset;
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

// The narrow region-B seed of this lane at a cut pair, over the narrow piece
// the argument falls in; same reading as RegionBSeedRationalNarrowF32.
inline float RationalSeedNarrowF32AtCut(std::size_t index,
                                        int numDeg,
                                        int denDeg,
                                        float t) noexcept {
    const detail::f32::RatPiece& piece = detail::f32::kNarrowRatBPiecesF32[index];
    const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
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

// Region-A seed of the rational route over the narrow partition: the same
// numerator over one plus t times the stored denominator, at the narrow
// pieces' own intervals and mapped arguments, so the two routes over a piece
// read one t. The degree pair is per piece here rather than per order, since
// the narrow pieces are shorter than the shipped cover's and the family's own
// search is what placed them.
inline float RationalValueNarrowF32(int order, float x) noexcept {
    const detail::f32::RatPiece& piece = FindNarrowRatPieceF32(order, x);
    const float* c = detail::f32::kNarrowRatACoeffsF32.data() + piece.offset;
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

// Region-B seed of the rational route over the narrow partition; see
// RegionBSeedRationalF32 for the shipped seed this is the narrow counterpart of.
inline float RegionBSeedRationalNarrowF32(float x) noexcept {
    const detail::f32::RatPiece& piece = FindNarrowRatBPieceF32(x);
    const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
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

// The two routes as one float-lane call reads them. Each policy forwards to
// the helper above, so a policy's arithmetic is the helper's operation for
// operation at the axes the policy names. The granularity names which partition
// of region A and which region-B seed the route reads; both routes carry both
// members, so the partition is the route's reading of the same cut of the
// region rather than a second family.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
struct ChebyshevFit32 {
    /// The partition this fit was named with. A body that reads a fit and has to know
    /// whether the fit answers that partition asks the shared question here rather than
    /// re-deriving it from the type.
    static constexpr FitGranularity kPartition = kGranularity;

    static float EvalOrder(int n, float x) noexcept {
        return ChebyshevValueF32<kScheme, kGranularity>(n, x);
    }

    // This family's seed covers the whole region at one degree, so the order is
    // not read.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        return RegionBSeedF32<kScheme, kGranularity>(x);
    }
};

// The rational family takes no scheme: its numerator and denominator are stored
// in monomial form and read by Horner, so there is no second table for a scheme
// to choose between. A policy that names this family therefore composes with
// either scheme and evaluates the same values under both - which is what the
// double lane's rational route does with its own scheme axis as well.
template <FitGranularity kGranularity = kDefaultFitGranularity>
struct RationalFit32 {
    // Where a batch reading of this route hands an order over to the route's
    // own fit rather than to the band's seed: the lane's own constant, the one
    // the double lane's rational route hands over at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    /// The partition this fit was named with; see ChebyshevFit32::kPartition.
    static constexpr FitGranularity kPartition = kGranularity;

    static float EvalOrder(int n, float x) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return RationalValueF32(n, x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return RationalValueNarrowF32(n, x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // Region A of this family, and the grid has no order of this shape: its
            // pairs are one per interval of a fixed grid, reached by the interval's own
            // offset. The arm refuses rather than reading the narrow pairs under the
            // grid's name.
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this is the rational family's piece-indexed region-A read, and the "
                          "uniform grid has no piece-indexed pair: its numerator and "
                          "denominator are one per interval. Read the grid through its own "
                          "order entry rather than here");
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    // One pair for the whole region, so the order is not read.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return RegionBSeedRationalF32(x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return RegionBSeedRationalNarrowF32(x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // The same pair as the shipped arm, by construction and not by fallback:
            // region B's seed is one fit over [kX0, kX1) at every granularity except the
            // narrow one, and this route's shipped region B is one pair over the interval.
            return RegionBSeedRationalF32(x);
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }
};

// The two families under one name: what a float engine that takes a route and a
// scheme reads, the way the double bodies read Policy::Fit.
template <FitRoute kRoute, EvalScheme kScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
using FloatRouteFit = std::conditional_t<kRoute == FitRoute::kChebyshev,
                                         ChebyshevFit32<kScheme, kGranularity>,
                                         RationalFit32<kGranularity>>;

// Region-A seed of a float-lane batch, in the double precision the downward
// recursion needs (see the batch body below). The route's own fit answers where
// that route's selector takes over, the shipped family below it - the same pair
// of choices, gated by the same constant, that the double lane's batch makes.
// It is the double lane's fit and not this lane's because a 1.5e-7 seed is a
// 5e-3 result at nmax = 8, so the per-order floats this lane is certified at
// cannot seed a batch at any order worth the name.
template <FitRoute kRoute, EvalScheme kScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
double FloatBatchRegionASeed(int order, double x) noexcept {
    if constexpr (kRoute == FitRoute::kRationalMinimax)
    {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            if (x >= RationalFit::kRegionAFitsFrom)
            {
                return RegionAValue<RationalFit>(order, x);
            }
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            if (x >= RationalFitNarrow::kRegionAFitsFrom)
            {
                return RegionAValue<RationalFitNarrow>(order, x);
            }
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // Region A of the float batch's seed is read through the double lane's
            // piece-indexed families, and the grid has no piece-indexed fit: its cells are
            // read interval-major by the grid's own body, which answers this shape before
            // this body is reached. The arm refuses rather than reading the narrow pieces
            // under the grid's name.
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "region A of the float batch's seed is read through the piece-indexed "
                          "families, and the uniform grid has no piece-indexed fit: its cells "
                          "are read interval-major at the degrees they were fitted at");
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    // The Chebyshev family, below the route's own selector and over the whole of region A
    // on the Chebyshev route. It answers the shipped and the narrow partition; the grid
    // is refused by the family itself, at the read.
    return RegionAValue<ChebyshevFit<kScheme, kGranularity>>(order, x);
}

// The float lane's single-order body over a fit policy: one body, so the
// route names the two fits and changes nothing else. Region C reads no
// coefficient at all and is the same three lines under either route.
//
// The division form is a parameter here rather than a field the fit carries,
// because the fit is what this body's callers select and the form is what the
// policy selects beside it. The region-B exponential is the same kind of
// parameter, and it has no default: a caller that forgot it would run one
// member's seed under a policy naming the other, which is the substitution this
// body's call site exists to prevent.
template <typename Fit, DivisionForm kForm = kDefaultDivisionForm, RegionBExp kExp>
float SingleOrderF32Body(int n, float x) noexcept {
    if (x == 0.0f)
    {
        return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
    }

    const float x0 = static_cast<float>(kX0);
    const float x1 = static_cast<float>(kX1);

    // The region-A read goes through the fit's own stored pieces, which the uniform grid
    // does not have: its cells are interval-major and are read by UniformSingleOrderF32.
    // The read is dropped for a partition the fit does not answer rather than answered
    // from the narrow pieces under the grid's name - and it is dead where it is dropped,
    // because the entry that routes a uniform policy here has already answered every
    // argument below the grid's join and returned.
    if constexpr (FitAnswersPartition<Fit>(Fit::kPartition))
    {
        if (x < x0)
        {
            return Fit::EvalOrder(n, x);
        }
    }

    float f = Fit::RegionBSeed(x, n);

    if (x < x1)
    {
        const float expx = RegionBHalfExpF32<kExp>(x);

        // The recurrence step is written as the backend's two-rounding
        // multiply-subtract rather than as a bare product and difference: a bare
        // one contracts where the compiler contracts and not where it does not,
        // which would make this value a property of the calling translation unit's
        // flags - and of the optimizer's choice between two inlined copies of this
        // call in one unit, so that two call sites of the same entry could return
        // adjacent values. The spelled form rounds the product and then the
        // difference on every build.
        const float invx = StepReciprocal<kForm>(x);

        for (int l = 0; l < n; ++l)
        {
            f = DivideStep<kForm>(backend::ScalarFp32::MulSub(
                                      static_cast<float>(l) + 0.5f, f, expx),
                                  x, invx);
        }

        return f;
    }

    f = kBoysHalfSqrtPiF32 / std::sqrt(x);
    const float invx = StepReciprocal<kForm>(x);

    for (int l = 0; l < n; ++l)
    {
        f = DivideStep<kForm>((static_cast<float>(l) + 0.5f) * f, x, invx);
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
// The fit is the policy's by default and is overridden by a route whose fits are
// its own rather than the shipped family's; everything else about the body - the
// zero argument, the region split, the per-order rule, the domains - is the same
// under either, which is why the fit is a parameter here and not a second body.
//
// **What the two grid branches rest on, asserted once for both bodies.** A policy
// naming the grid is answered by the grid's own branch, which returns for every
// argument below kFlatHi; the region-A and region-B blocks beneath that branch
// resolve the policy's partition through the Chebyshev family, which does not carry
// the grid. The blocks are therefore dead for such a policy - and this is why: the
// grid's join is above the end of region B, so every argument the blocks cover has
// already been answered above them. The blocks are dropped for a partition that
// family does not answer (an `if constexpr` on the same question, in AllOrdersBody
// and in SingleOrder below), so the reads inside them are not instantiated for such
// a policy and carry the assertion of every other read of that family. A grid whose
// join fell inside the fitted domain would make both facts false at once - the drop
// would compile and the reads would be live - which is what this asserts against.
static_assert(kX0 < kX1 && kX1 < kFlatHi,
              "the double lane's grid must reach past the end of region B: below kFlatHi the "
              "grid's own branch answers and returns, so a policy naming it never reaches the "
              "region-A and region-B reads that resolve through the Chebyshev family. A grid "
              "stopping inside the fitted regions would leave those reads live for that policy "
              "and answer it from another partition's tables under the grid's name");
template <EvalPolicyLike Policy,
          typename Fit = typename Policy::Fit>
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

    // The uniform route answers the whole of its table's domain from the table
    // and hands everything above it to the asymptotic below, which reaches no
    // fit: the join is above kX1, so an argument past it satisfies neither of
    // the two region tests that follow and falls through to the asymptotic by
    // the tests themselves rather than by a third one written here.
    if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        if (x < detail::kFlatHi)
        {
            UniformAllOrders<Policy>(nmax, x, out);

            return;
        }
    }

    if constexpr (FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The block below resolves the policy's partition through the Chebyshev family,
        // which carries the shipped and the narrow partition and does not answer the grid,
        // so it is dropped for a partition that family does not answer: the reads inside it
        // are then not instantiated for such a policy, and the assertion they carry cannot
        // refuse the grid's own served bodies. Dropping it changes nothing a served call
        // runs - the grid's branch above returned for every argument this block covers, by
        // the ordering asserted above AllOrdersBody - and it is what makes the barrier in
        // PolicyRegionAValue the same one PolicyRegionBSeed carries.
        if (x < kX0)
        {
            // The pure per-(n, x) dispatch, driven by the n-indexed threshold table: the
            // orders k with x >= kTierThresholds[k] are a prefix (the thresholds are
            // non-decreasing in n) and are read from this route's own band answer; the
            // tail orders keep the per-order piece value at the partition the policy
            // names - below its own end an order is documented at the per-order 1e-15,
            // and a route whose fits hold the wider bar does not answer there. On the
            // Chebyshev route at or above the band's left edge out[k] is bit-identical to
            // the single-order entry's value for every k.
            //
            // The fallback below, taken when x is under the band's left edge and no
            // order takes the band's answer at all, is the one dispatch that is not: it
            // seeds the downward recursion once, at nmax, and pays one fit for the batch
            // where the per-order reading would pay one per order. So out[nmax] is still
            // the single-order entry's value for nmax bit for bit at the reference
            // multiplier, where the two roles read one stored table; at a rung the seed
            // is read at the batch role's cut degrees and the single entry answers at the
            // single role's, so the two part by up to 8.27e-09 relative at m = 1024, each
            // reading still inside its own bound. out[k] for k < nmax carries the
            // recurrence's value rather than the fit's, and the two readings of one cell
            // differ by a fixed absolute amount and not a fixed number of last places:
            // swept over the accuracy gate's committed grid at every nmax, the worst is
            // 3.33e-16 absolute (nmax = 1, k = 0, x = 0.91067553232796428), 3 ULP of
            // that result, while the worst in ULP is 140 (nmax = 29, k = 27,
            // x = 1.0418128089831911), where the result is 6.66e-3. Both readings are
            // inside the bound this entry documents.
            int served = 0;

            while (served < nmax &&
                   x >= detail::kTierThresholds[static_cast<std::size_t>(served + 1)])
            {
                ++served;
            }

            if (x >= Fit::kRegionAFitsFrom)
            {
                typename Fit::template BandSource<Policy::kDivision> source(x);

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

            // Seeds the downward recursion, so its error is the batch's: the degree
            // is the batch role's, as it is in the Chebyshev rung's own batch body.
            double f = PolicyRegionAValue<Policy>(nmax, x);
            out[nmax] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = DivideDownwardStep<Policy::kDivision>(l, x * f + expx);
                out[l] = f;
            }

            return;
        }
    }

    if (x < kX1)
    {
        double f = Fit::RegionBSeed(x);
        out[0] = f;
        const double expx = RegionBHalfExp<Policy::kRegionBExp>(x);
        const double invx = StepReciprocal<Policy::kDivision>(x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = UpwardStep<Policy::kDivision>(l, f, x, invx, expx);
            out[l] = f;
        }

        return;
    }

    double f = kBoysHalfSqrtPi / std::sqrt(x);
    out[0] = f;
    const double invx = StepReciprocal<Policy::kDivision>(x);

    for (int l = 1; l <= nmax; ++l)
    {
        f = DivideStep<Policy::kDivision>((l - 0.5) * f, x, invx);
        out[l] = f;
    }
}

// One order at one argument, in the policy's family and scheme. The fit is
// overridden by the caller as it is in AllOrdersBody, for the same reason.
template <EvalPolicyLike Policy,
          typename Fit = typename Policy::Fit>
double SingleOrder(int n, double x) noexcept {
    static_assert(FitPolicy<Fit>,
                  "the fit a policy names must satisfy the contract the bodies are written "
                  "against (backend.hpp, FitPolicy)");
    assert(n >= 0 && n <= kMaxBoysOrder);

    if (x == 0.0)
    {
        return 1.0 / (2.0 * n + 1.0);
    }

    // The uniform route's own domain, as in AllOrdersBody and for the same
    // reason: above the join the tests below reach the asymptotic by themselves,
    // so this is the whole of what the route has to answer for here.
    if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        if (x < detail::kFlatHi)
        {
            return UniformSingleOrder<Policy>(n, x);
        }
    }

    if constexpr (FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The block below resolves the policy's partition through the Chebyshev family,
        // which carries the shipped and the narrow partition and does not answer the grid,
        // so it is dropped for a partition that family does not answer: the reads inside it
        // are then not instantiated for such a policy, and the assertion they carry cannot
        // refuse the grid's own served bodies. Dropping it changes nothing a served call
        // runs - the grid's branch above returned for every argument this block covers, by
        // the ordering asserted above AllOrdersBody - and it is what makes the barrier in
        // PolicyRegionAValue the same one PolicyRegionBSeed carries.
        if (x < kX0)
        {
            if (x >= detail::kTierThresholds[static_cast<std::size_t>(n)])
            {
                typename Fit::template BandSource<Policy::kDivision> source(x);
                double f = 0.0;

                // The order is at most kMaxBoysOrder, and bounding the walk by it
                // as well as by n is what lets a compiler see the induction
                // terminate: on an order outside the contract the walk is defined
                // rather than an overflow waiting to happen, and for every order
                // inside it the two bounds agree.
                for (int l = 0; l <= n && l <= kMaxBoysOrder; ++l)
                {
                    f = source.Next(l, x);
                }

                return f;
            }

            return PolicyRegionAValue<Policy>(n, x);
        }
    }

    if (x < kX1)
    {
        double f = Fit::RegionBSeed(x);
        const double expx = RegionBHalfExp<Policy::kRegionBExp>(x);
        const double invx = StepReciprocal<Policy::kDivision>(x);

        for (int l = 1; l <= n; ++l)
        {
            f = UpwardStep<Policy::kDivision>(l, f, x, invx, expx);
        }

        return f;
    }

    double f = kBoysHalfSqrtPi / std::sqrt(x);
    const double invx = StepReciprocal<Policy::kDivision>(x);

    for (int l = 0; l < n; ++l)
    {
        f = DivideStep<Policy::kDivision>((l + 0.5) * f, x, invx);
    }

    return f;
}

// ---------------------------------------------------------------------------
// The lane engines (compiled twice: the m = 1 branch is the certified body
// verbatim; the relaxed branch evaluates at the effective degrees)
// ---------------------------------------------------------------------------
// Each engine takes the policy and the multiplier, and nothing else: the fit
// family, the scheme and the single-precision budget arrive as the policy's
// fields, so a further axis does not reopen these signatures.

// The route the batch entries' region bodies carry, and the one the fixed-order
// entry carries: the shipped one, at every rung. Those bodies reach their values by
// a path of their own - the batch's region A seeds its downward recursion from the
// top order's stored fit and recurses, region B seeds its upward recursion from the
// stored seed - and read the shipped tables through it. A policy naming another route
// takes the entry's per-argument path instead, which reads its fit from the policy;
// nothing falls back silently.
//
// The per-argument entries do not ask for it, because their bodies take the fit from
// the policy: they carry either route, the rational one through its own fit. The
// uniform partition is carried by the all-orders and
// single-order bodies, which read its table directly, and by no other body here:
// every other body reaches its values through a recursion over the orders - a seed
// stepped upward or a piece stepped downward - and the uniform table offers neither.
//
// A body without a uniform branch that is handed a uniform policy does not fail to
// compile on its own: it asks the policy's contract members for values, and those
// are answered by the narrow fits, so the call returns certified numbers from a
// partition the caller never named and reports nothing. The all-n entry did exactly
// that; this refuses the combination where it is named.
//
// Every batched entry now routes the partition to those two bodies instead of
// asking this guard's question: the plane entry and its sorted overload on the
// arguments axis, and the fixed-order entry, all hand a uniform policy to the
// per-argument path, which reads the grid. What is left under the guard is the
// partitioned path, which no policy naming the grid reaches - so the assertion
// below is a contract on the path rather than a cell an entry refuses, and a
// revision that routed the partition back into it would be refused here by name.
//
// The condition is the shared question and not a second statement of it: these
// bodies reach their values through PolicyRegionAValue, PolicyRegionBSeed and the
// two rung reads, which resolve the policy's partition through the Chebyshev
// family, so what this refuses is the family not answering the partition the
// policy named. A partition that family does not carry is refused by name here
// before the read is reached, which is what makes the refusal this body's own.
template <EvalPolicyLike Policy>
constexpr void RefuseUniformPartition() noexcept
{
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this body has no uniform branch: it reaches its values through a recursion "
                  "over the orders, and the uniform table is fitted per order with no "
                  "recurrence to enter. A policy naming that partition here would be answered "
                  "by another partition's fits under the uniform name. Either give this body "
                  "a branch that reads the uniform table, or refuse the partition here - do "
                  "not leave it to the policy's contract members");
}
// The rational member over the uniform grid is a fit of this lane's own and it is
// stored (see RationalFitUniformF32 above), so FloatRouteFit's fall-through no
// longer reaches a policy naming it: UniformOrderAtF32 dispatches on the route and
// reads the pairs this lane's grid carries. RationalFit32 would otherwise resolve
// the grid to the shipped rational member's cover - a different partition's fits
// under the uniform name - which is why the member is a fit of its own rather than
// a reading of the double lane's.

template <EvalPolicyLike Policy>
constexpr void RequireShippedRoute() noexcept
{
    static_assert(Policy::kRoute == FitRoute::kChebyshev,
                  "this body reaches its values through the shipped fits' own path - the "
                  "downward recursion from a stored piece, or the upward recursion from a "
                  "stored seed - and reads the shipped tables through it: a policy naming "
                  "another route is served by the shape that takes its fit from the policy "
                  "instead, and this shape is not instantiated for one");
}

// A relaxed rung truncates a stored row to a per-order effective degree, and each
// partition carries its own such table: the shipped row's degrees are cut from the
// shipped pieces' coefficients, the narrow row's from the narrow pieces', so a rung
// of the narrow partition is a rung of the narrow fit and not the shipped row's
// degrees over narrower intervals. The two tables are derived by one criterion over
// the two tables of coefficients (boys_effective_degrees.hpp), and every m > 1 body
// below picks the pair matching the partition its policy names, so the granularity
// and the rung are one choice rather than a crossing.
//
// What the choice costs is stated where the partitions are: a rung of the narrow
// partition reads fewer coefficients at the same budget - the narrow pieces are
// lower-degree to begin with - and pays the longer piece scan its table needs.

template <EvalPolicyLike Policy>
double BoysSingleImpl(int n, double x) noexcept {
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order, so it has one order to put in a vector lane "
                  "and the orders axis is not an axis here: the axis this library carries on "
                  "this shape is the arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0);

    // m = 1 is the full-accuracy body selected at compile time: no branch,
    // indirection or runtime dispatch sits on that path, and a relaxed rung is a
    // separate instantiation taken one branch below rather than a test the
    // full-accuracy call pays for.
    //
    // The uniform partition is selected here at every multiplier: its cells are
    // stored at the degrees the grid derived them at, so naming it is the
    // reference reading and not a thinner one. Both members of the partition are
    // admissible alike; the body the call lands on carries the partition's own
    // branch, so the grid answers below its join and the region tests above it.
    {
        return SingleOrder<Policy>(n, x);
    }
}

// Named by BoysAllOrdersImpl below, so it is declared before it. The call there
// passes a dependent template argument but arguments of fundamental type, so
// neither lookup at the point of definition nor ADL at instantiation finds the
// declaration near the end of this file. Its default arguments are set there.
template <EvalScheme kScheme,
          FitRoute kRoute,
          FitGranularity kGranularity,
          DivisionForm kForm>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

// The single-precision lane's entry on the same axis (defined in
// boys_orders_simd.cpp), declared here for the same reason and one more: this
// lane's degree table is certified against a region budget as well as against a
// table, so the computation budget is a choice it carries and the double lane's
// entry does not.
template <EvalScheme kScheme, FitRoute kRoute, BoysBudget kBudget,
          FitGranularity kGranularity = kDefaultFitGranularity, DivisionForm kForm>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept;

template <EvalPolicyLike Policy>
void BoysAllOrdersImpl(int nmax, double x, double* out) noexcept {
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
        // route's, any rung, and any of the three division forms. The five choices
        // reach the lane as its template arguments, so what the entry answers
        // inside the packed interval, outside it, and on a host without the vector
        // tier is one policy's answer throughout. The form is one of them because
        // the lane hands the orders it does not pack to the scalar single lane,
        // whose recurrence steps divide - dropping it here would answer such a
        // caller with another form's arithmetic.
        //
        // The narrow partition does not carry the stride the shipped lane's fetch
        // uses (its pieces are cut per order), so its lane reads each order's own
        // piece; the axis is the same one either way.
        BoysAllOrdersPacked<Policy::kScheme,
                            Policy::kRoute,
                            Policy::kGranularity,
                            Policy::kDivision>(nmax, x, out);
    } else {
        // The uniform partition is here at every multiplier, for the reason the
        // single-order entry states: its ladder is the reference reading, and the
        // body below carries the partition's own branch. On the orders axis the
        // branch above is the lane that reaches this same partition's ladder, at
        // this same multiplier, one argument at a time.
        AllOrdersBody<Policy>(nmax, x, out);
    }
}

// The fixed-n vector engine: F_n at every argument of an array, one fixed order (the
// batch shape of angular-momentum-grouped inner loops; the strided output layout
// belongs to the public surface in boys.hpp). The region bodies below mirror
// BoysSingleImpl's verbatim - the m = 1 branch is the certified scalar single-lane
// code, the relaxed branch the same bodies at the single-lane effective degrees
// (BoysRole::kDoubleSingle) - so every output element returns the corresponding
// BoysSingle call's value, and the same bits on a build whose bare product-plus-add
// is two roundings. On a build that contracts that form the compiler decides per call
// site whether to fuse it, and this call shape is not the single entry's: a value can
// move by a unit in the last place and no further. Keep the two engines' bodies in
// lockstep - identical source is what holds them inside one bound. The region
// dispatch is per element, so mixed-region arguments need no pre-partitioning; the
// dispatch-once-per-batch structure lives in the AVX2 region-sorted lanes
// (boys_simd.cpp), whose callers partition by region first.
template <EvalPolicyLike Policy>
void BoysFixedNImpl(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
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

    // The partition clause is the shared question rather than a statement that this
    // entry refuses the grid: the shaped body below reads PolicyRegionAValue and
    // PolicyRegionBSeed, which resolve the policy's partition through the Chebyshev
    // family, so what sends a policy past this branch is that family not answering
    // the partition it named - and a further partition the derived families do not
    // carry is routed the same way rather than being read as the narrow one.
    if constexpr (Policy::kRoute != FitRoute::kChebyshev ||
                  !FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The fixed-order entry carries the route too, and by the entry that
        // takes its fit from the policy: one order at every argument of an
        // array is the per-argument single entry called once per argument -
        // which is the body this entry's own m = 1 path already mirrors
        // verbatim, region for region, so naming a route makes the identity the
        // documentation already claims exact by construction rather than by
        // inspection. The shaped body below is the bit-identity pin's, and the
        // shipped route keeps it.
        //
        // The uniform partition takes this path for the reason the route does.
        // The two Chebyshev branches below reach their values through
        // PolicyRegionAValue and PolicyRegionBSeed, and those resolve the
        // partition to ChebyshevFit, whose else branch is the NARROW member - so
        // a uniform policy sent down either of them would be answered from
        // another partition's fits under the grid's name. BoysSingleImpl carries
        // the partition's own branch instead: the grid's table below its join,
        // read at every multiplier, and the region tests above it.
        //
        // What this costs the caller is the shaped body's own arithmetic: this
        // path is one order at one argument per call, so it does not run the
        // region dispatch the block below is. A caller naming the grid has
        // already chosen a partition whose every order is its own fit, so the
        // recurrence the shaped body is built around is not one of its costs.
        for (std::size_t i = 0; i < count; ++i)
        {
            assert(x[i] >= 0.0);
            out[i * stride] = BoysSingleImpl<Policy>(n, x[i]);
        }

        return;
    }
    // The two selections are alternatives and not two independent tests: the
    // branch above returns whenever it is taken, so saying so here is what the
    // code already means - and it is the difference between a compiler reading
    // the block below as discarded and reading it as unreachable.
    else {
        // The guard belongs in the two Chebyshev branches and not at the top of
        // the function. This one reaches its values through PolicyRegionAValue
        // and PolicyRegionBSeed, which resolve the partition to ChebyshevFit, and
        // that family's else branch reads the NARROW tables - so a uniform policy
        // here would be answered from another partition's fits under the uniform
        // name. The branch above needs no guard: it hands each argument to
        // BoysSingleImpl, which reads the grid.
        //
        // The assertion stands as this branch's own contract rather than as a
        // cell this entry refuses: the branch above takes the uniform partition
        // too, so no policy naming the grid reaches here. It stays because this
        // body genuinely cannot answer the grid - the two reads above are the
        // recursion's, and the grid has no next order to build from this one.
        RefuseUniformPartition<Policy>();

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
                    const double invxi = StepReciprocal<Policy::kDivision>(xi);

                    for (int l = 0; l < n; ++l)
                    {
                        f = DivideStep<Policy::kDivision>((l + 0.5) * f - expx, xi, invxi);
                    }

                    out[i * stride] = f;
                    continue;
                }

                out[i * stride] = PolicyRegionAValue<Policy>(n, xi);
                continue;
            }

            double f = PolicyRegionBSeed<Policy>(xi);
            const double invxi = StepReciprocal<Policy::kDivision>(xi);

            if (xi < kX1)
            {
                const double expx = RegionBHalfExp<Policy::kRegionBExp>(xi);

                for (int l = 0; l < n; ++l)
                {
                    f = DivideStep<Policy::kDivision>((l + 0.5) * f - expx, xi, invxi);
                }

                out[i * stride] = f;
                continue;
            }

            f = kBoysHalfSqrtPi / std::sqrt(xi);

            for (int l = 0; l < n; ++l)
            {
                f = DivideStep<Policy::kDivision>((l + 0.5) * f, xi, invxi);
            }

            out[i * stride] = f;
        }
    }
}

// The float lanes' scope (the extended-band seed is a double lane): the float
// dispatch is untouched, still keyed to kX0/kX1, so the carved band
// [kExtendedBX0, kX0) stays EXACTLY the float path - the per-order region-A fits,
// double-seeded in the batch form, serving the band at the float budget. The
// certified table is the double recursion's; the float band is measured.
//
// At the reference multiplier the policy reaches this engine as the pair of fits its
// two seeds are read from and the table those fits are summed out of: the route picks
// the family (the shipped Chebyshev fits or the rational set BoysSingleF32WithRoute
// serves) and the scheme which of the two parallel tables the Chebyshev family is
// read from. The engine is one body per route, so naming a policy cannot reach a fit
// the caller did not name. Past the reference multiplier the same two names pick the
// same two fits, cut where the rung's criterion certifies them: the degrees are the
// role's own, derived from the tables this lane stores, so a rung is a rung of the
// family the caller named.
template <EvalPolicyLike Policy>
float BoysSingleF32Impl(int n, float x) noexcept {
    // The orders axis is refused here on the same reading as the double entry's
    // refusal above: this shape evaluates one order at one argument, so there is no
    // second order to fill a lane with and no argument array to widen over, and the
    // axis is a property of the shape rather than of the precision. BoysSingleF32
    // documents the refusal; without the assertion the policy compiled through and
    // the arguments axis answered in its place.
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order, so it has one order to put in a vector lane "
                  "and the orders axis is not an axis here: the axis this library carries on "
                  "this shape is the arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0f);

    // The uniform partition's own domain, taken before the route and the rung below
    // rather than through them: the grid is a table no other body here reads, so this
    // branch keeps a policy naming it from being answered by the narrow fits the
    // engine below would reach. Above the join the table does not reach and the
    // region tests inside that engine answer, which is why this is a test on the
    // argument and no second domain.
    //
    // The multiplier is not read and the rung is not refused here: a rung cuts a
    // stored row to a per-order effective degree, and this table's cells are already
    // at the degrees the grid was fitted at, so a relaxed call is served the stored
    // cells uncut - a saving left on the table rather than a value missed, since the
    // caller named a bound and an uncut cell is inside it.
    //
    // Above the join the body that answers is the one the reference multiplier
    // answers with, which is why the selection below takes that body for this
    // partition. The route body is not what answers past the join: its fit resolves
    // every partition but the shipped one to the narrow pieces, so a uniform policy
    // would have been answered past the join by the narrow member's region-B seed
    // bit for bit - certified numbers, from a partition the caller never named, with
    // nothing reporting it. That is the substitution the guard refuses rather than
    // serves.
    if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        if (x < detail::f32::kFlatHiF32)
        {
            // The origin is answered in closed form, as every other route here
            // answers it: F_l(0) is 1 / (2l + 1) exactly, and the grid's first
            // cell is a fit of it, so the two would part at zero by its truncation.
            if (x == 0.0f)
            {
                return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
            }

            return UniformSingleOrderF32<Policy>(n, x);
        }
    }

    {
        return SingleOrderF32Body<FloatRouteFit<Policy::kRoute, Policy::kScheme,
                                                Policy::kGranularity>,
                                  Policy::kDivision, Policy::kRegionBExp>(n, x);
    }
}

template <EvalPolicyLike Policy>
void BoysAllOrdersF32Impl(int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    // The uniform partition's whole domain, both packing axes, before the routes
    // below: the grid is a table no other body here reads, and its ladder gives the
    // per-order value at every order, which is the value the across-orders packed lane
    // carries into its lanes. So one reading answers either axis, and the axis this
    // branch does not dispatch on changes which lane would have run rather than which
    // fit is read.
    //
    // The multiplier is not read, for the reason the single-order entry states: the
    // cells are stored at the grid's own degrees, so a relaxed call is served them
    // uncut and inside the bound it named. Past the join the branch below selects the
    // reference body's ladder for this partition at every multiplier, and on the
    // orders axis the lane the first branch names hands its arguments past the join to
    // the single-order entry at this same partition, which the same selection answers
    // from the same fit - so this partition's ladder is one reading at every rung on
    // either axis.
    if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        if (x < detail::f32::kFlatHiF32)
        {
            // The origin in closed form, for the reason the single-order entry
            // gives: the grid's first cell fits F_l(0) rather than being it.
            if (x == 0.0f)
            {
                for (int l = 0; l <= nmax; ++l)
                {
                    out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
                }

                return;
            }

            UniformAllOrdersF32<Policy>(nmax, x, out);

            return;
        }
    }

    if constexpr (Policy::kPack == PackAxis::kOrders)
    {
        // The across-orders packed lane: eight orders of one argument in one vector
        // register, which is the shape BoysAllOrdersF32(nmax, x, out) has and the axis
        // the caller named. It carries the same tables and the same arithmetic as this
        // engine - a lane's value is the per-order value, and the lane's own suite
        // asserts it - so the axis changes which lane runs and not which fit is read.
        //
        // At a relaxed multiplier the lane serves all four of its (scheme, route) pairs
        // on either partition, as at the reference one: a rung is a table of effective
        // degrees cut from the coefficients the named family stores, and each partition
        // holds its own family's coefficients, so a rung of either is a reading of the
        // family the caller named.
        //
        // The lane carries the partition the policy names: its narrow body reads each
        // order's own piece of the narrow table, and the fallback outside the lane's
        // interval is the scalar lane at the same partition.
        BoysAllOrdersF32Packed<Policy::kScheme,
                               Policy::kRoute,
                               Policy::kBudget,
                               Policy::kGranularity,
                               Policy::kDivision>(nmax, x, out);
    } else {
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

        // Region A of the batch, through the double lane's piece-indexed families, which
        // the uniform grid does not have: the entry above answers every argument below
        // the grid's join and returns, so this block is dead for a policy naming the
        // grid - and it is dropped for one, so the read inside it is not instantiated
        // and the family's own refusal cannot be raised by a call that never runs.
        if constexpr (FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                          Policy::kGranularity))
        {
            if (x < x0)
            {
                // The seed must be double precision: the downward recursion amplifies
                // a float seed error by up to x0^n / prod(j+1/2) (5e4 at nmax=8), far
                // beyond the certified 1.5e-7 float budget; one double evaluation per
                // batch is negligible; the recursion itself stays in float.
                const double seed =
                    FloatBatchRegionASeed<Policy::kRoute, Policy::kScheme,
                                          Policy::kGranularity>(nmax,
                                                                static_cast<double>(x));
                out[nmax] = static_cast<float>(seed);
                float f = out[nmax];
                const float expx = 0.5f * std::exp(-x);

                for (int l = nmax - 1; l >= 0; --l)
                {
                    f = DivideDownwardStep<Policy::kDivision>(l, x * f + expx);
                    out[l] = f;
                }

                return;
            }
        }

        float f = FloatRouteFit<Policy::kRoute, Policy::kScheme, Policy::kGranularity>::RegionBSeed(x, 0);
        out[0] = f;
        const float invx = StepReciprocal<Policy::kDivision>(x);

        if (x < x1)
        {
            const float expx = RegionBHalfExpF32<Policy::kRegionBExp>(x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = DivideStep<Policy::kDivision>((static_cast<float>(l) - 0.5f) * f - expx, x, invx);
                out[l] = f;
            }

            return;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = DivideStep<Policy::kDivision>((static_cast<float>(l) - 0.5f) * f, x, invx);
            out[l] = f;
        }
    }
}

// The float lane's all-N batch: the per-argument all-orders body at every
// argument, the results scattered into the caller's planes. The packed float
// lane this lane has packs the orders of one argument, and this entry's run is
// one argument after another, so neither axis gets a homogeneous run out of it:
// the run the axis would pack is the orders of a single argument, which the
// all-orders body below already fills, one call per argument. What it carries
// is the shape: the one BoysAllN has in the double lane and BoysCuda::AllNF32
// has on the device.
template <EvalPolicyLike Policy>
void BoysAllNF32Impl(int nmax, const float* x, float* out, std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    // Both axes are served, and the axis is read: the per-argument body this loop
    // calls is the all-orders entry, which packs eight orders of one argument when
    // the policy names the orders axis and fits one argument's region otherwise.
    // The arguments axis's region partitioning stays at the caller's loop, as the
    // doc comment above says. The uniform partition reaches the grid through that
    // same call, which carries the partition's own branch, so this loop owes it
    // nothing.
    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0f);

        float row[kMaxBoysOrder + 1];
        BoysAllOrdersF32Impl<Policy>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
    }
}

// The float lane's fixed-order batch: the per-argument single body at every
// argument, one fixed order, the strided output the public surface documents.
// It is the loop the double lane's fixed-order entry falls back to when its
// policy names a route or the grid, and it is a body of its own here because
// the float lane has no shaped region kernel to carry the other path: the
// packed lane this entry would need is an AVX2 kernel over doubles, so the
// per-argument body is the whole of what this entry serves at every policy,
// which is also why no speed is claimed for it over the caller's own loop.
template <EvalPolicyLike Policy>
void BoysFixedNF32Impl(
    int n, const float* x, float* out, std::size_t count, std::size_t stride) noexcept {
    // The packing axis is not asserted here: the per-argument body this loop
    // calls, BoysSingleF32Impl, refuses the orders axis itself, on the reading
    // its own comment gives - one order at one argument has no second order to
    // fill a lane with - and a guard here would state the same refusal twice.
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x != nullptr);
    assert(out != nullptr);
    assert(stride >= 1);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0f);
        out[i * stride] = BoysSingleF32Impl<Policy>(n, x[i]);
    }
}

// The float lane's per-element-top-order batch: the per-argument all-orders
// body at each argument's own top order, written into the caller's planes
// exactly as the double lane's entry of this shape writes them, and with the
// same property - the cells above each column's own top order are left as the
// caller left them, because each column stops at its own n[i].
template <EvalPolicyLike Policy>
void BoysAllNAtOrdersF32Impl(
    const int* n, const float* x, float* out, std::size_t count) noexcept {
    // No RefuseUniformPartition here, on the double lane's own reading: every
    // value below comes from BoysAllOrdersF32Impl, which reads the uniform grid
    // where the policy names it, so a guard at this entry would refuse a
    // combination this body serves.
    assert(count == 0 || n != nullptr);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(n[i] >= 0 && n[i] <= kMaxBoysOrder);
        assert(x[i] >= 0.0f);

        float row[kMaxBoysOrder + 1];
        BoysAllOrdersF32Impl<Policy>(n[i], x[i], row);

        for (int l = 0; l <= n[i]; ++l)
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
// The certified per-value bounds: A |F̂ − F| ≤ 1e-15 over [0, kX0), B and C
// ≤ 5.5e-14. Measured against the committed reference grid (this tree's test
// suite): A holds that bound below the extended band and lands 1.4e-15 at
// n = 32 on the band itself, B lands 2.2e-13 at n = 32, x = kX0 — four times
// its own bound, which is why the batch entry serves region B with the scalar
// body instead — and C lands 5.0e-14, inside its bound with 10% slack.
//
// On a non-x86_64 target these are defined against the certified scalar lanes
// (see the guard in boys_simd.cpp) and BoysAvx2Available() reports false.
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for arguments in region B; out[order * count + i].
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept;

void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
// The fp16/bf16 lanes: the same kernels on half-precision I/O types, per order.
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16


// ---------------------------------------------------------------------------
// The across-orders packed lane (internal: defined in boys_orders_simd.cpp)
// ---------------------------------------------------------------------------
// The companion of the region kernels above: the same region-A stored fits, evaluated
// four ORDERS to a vector instead of four arguments, which is the axis
// BoysAllOrders(nmax, x, out) has. It is the orders axis of PackAxis, and the entries
// that carry it dispatch here.
//
// It evaluates each order's own fit and reaches no order by a recursion, so its values
// are the per-order fits' values: at the certified split Clenshaw scheme they are the
// across-arguments lane's values bit for bit, and the suite asserts that identity.
// The coefficients are fetched composed - four loads and the shuffles that join them -
// rather than with one gather instruction, which is the same lane by construction and
// cheaper in retired slots on the machines measured; the gathered fetch is kept beside
// it so the pair stays measurable.
//
// Below kX0 only. Past it the entry runs the certified scalar single lane one order at
// a time, at the policy the caller named, so a relaxed multiplier falls back to that
// rung of that route rather than to the reference one - a defined answer inside the
// entry's own bound rather than the packed lane.
//
// The four choices a policy makes reach the lane as template arguments: the scheme
// picks which polynomial table and which summation the shipped route's fits are read
// with, the route which region-A fits the lane carries, the accuracy multiplier the
// degree a fit is read at, and the partition the table those fits are cut into.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitRoute kRoute = kDefaultFitRoute,
          FitGranularity kGranularity = kDefaultFitGranularity,
          DivisionForm kForm = kDefaultDivisionForm>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

// The shapes this entry is instantiated at, written once and expanded at each of the
// three division forms: the form is a field of the policy the entry is named at, so
// the axis triples the cells rather than being read at the default alone. Declared
// here so that a call site reaches the definition the library already holds rather
// than instantiating a second copy of the body - and expanded at every form so that a
// form dropped from a list is a cell a caller naming it reaches and the library does
// not hold. tools/check_orders_packed_cells.py holds this block and the definitions
// in boys_orders_simd.cpp to each other.
#define BOYS_ORD_EXTERN(kScheme, kRoute, kGranularity, kForm)              \
    extern template void BoysAllOrdersPacked<kScheme, kRoute, kGranularity, kForm>(\
        int nmax, double x, double* out) noexcept;

#define BOYS_ORD_SHIPPED(kScheme, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kChebyshev, FitGranularity::kCoarsest, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kRationalMinimax, FitGranularity::kCoarsest, kForm)

#define BOYS_ORD_NARROW(kScheme, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kChebyshev, FitGranularity::kNarrow, kForm)

#define BOYS_ORD_NARROW_RAT(kScheme, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kRationalMinimax, FitGranularity::kNarrow, kForm)

// The uniform partition on the Chebyshev route, at every rung this lane names:
// the grid's cells are stored at the degrees the derivation fitted them at and
// the criterion that would cut them reaches the full degree at every multiplier,
// so a rung of this partition is the reference reading rather than a second,
// shorter one. The cells are therefore the same seven the shipped and narrow
// partitions are declared at, and a caller naming one reaches the library's own
// body rather than a copy of it.
#define BOYS_ORD_UNIFORM(kScheme, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kChebyshev, FitGranularity::kUniform, kForm)

// The uniform partition on the rational route, for the same reading. The
// member stores one pair per interval, so it is read as the stored pairs are
// - a saving left on the table rather than a value missing. The body delegates
// this route's call to the scalar orders lane, and the declarations are here for
// the reason every one above them is: so a call site naming a cell reaches the
// definition the library holds rather than making a second copy of the
// delegation.
#define BOYS_ORD_UNIFORM_RAT(kScheme, kForm) \
    BOYS_ORD_EXTERN(kScheme, FitRoute::kRationalMinimax, FitGranularity::kUniform, kForm)

BOYS_ORD_SHIPPED(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORD_SHIPPED(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORD_SHIPPED(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORD_SHIPPED(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORD_SHIPPED(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORD_SHIPPED(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORD_NARROW(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORD_NARROW(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORD_NARROW(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORD_NARROW(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORD_NARROW(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORD_NARROW(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORD_NARROW_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORD_NARROW_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORD_NARROW_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORD_NARROW_RAT(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORD_NARROW_RAT(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORD_NARROW_RAT(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORD_UNIFORM(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORD_UNIFORM(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORD_UNIFORM(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORD_UNIFORM(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORD_UNIFORM(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORD_UNIFORM(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORD_UNIFORM_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORD_UNIFORM_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORD_UNIFORM_RAT(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORD_UNIFORM_RAT(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORD_UNIFORM_RAT(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORD_UNIFORM_RAT(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

#undef BOYS_ORD_EXTERN
#undef BOYS_ORD_SHIPPED
#undef BOYS_ORD_NARROW
#undef BOYS_ORD_NARROW_RAT
#undef BOYS_ORD_UNIFORM
#undef BOYS_ORD_UNIFORM_RAT

// The single-precision lane's shapes on the same axis: two schemes and two
// computation budgets, with either route. Declared here for the
// reason above - so that a call site reaches the definition the library
// already holds rather than instantiating a second copy of the body.
#define BOYS_F32_ORDERS_PACKED_REFERENCE(kScheme, kBudget, kForm)                                  \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kChebyshev, kBudget,       \
                                                FitGranularity::kCoarsest, kForm>(                  \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kRationalMinimax, kBudget, \
                                                FitGranularity::kCoarsest, kForm>(                  \
        int nmax, float x, float* out) noexcept;



// The narrow partition's shapes on this lane, which boys_orders_simd.cpp instantiates
// and this block declared none of until they were counted: the same four (scheme,
// route) pairs at the reference rung plus the six relaxed ones, cut against the narrow
// table. Declared here so a call site reaching one of them reaches the definition the
// library already holds; the absence was invisible because nothing compared the two
// lists, and tools/check_orders_packed_cells.py now does.
#define BOYS_F32_ORDERS_PACKED_NARROW(kScheme, kBudget, kForm)                                     \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kChebyshev, kBudget,       \
                                                FitGranularity::kNarrow, kForm>(                   \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kRationalMinimax, kBudget, \
                                                FitGranularity::kNarrow, kForm>(                   \
        int nmax, float x, float* out) noexcept;



// The uniform partition's shapes on this lane, which boys_orders_simd.cpp instantiates
// at every multiplier and this block declares at every one of them: the double lane's
// uniform block is a single multiplier because its arm refuses the rung where the
// policy is named, and this lane's arm does not - the grid's cells are stored at the
// degrees the derivation fitted them at, so a relaxed call is served them uncut - so
// its cells are the seven the axis and the probe both name.
//
// Both routes are carried: the rational member over this lane's grid is a fit of the
// lane's own arithmetic rather than the double lane's pairs under the uniform name,
// stored beside the Chebyshev member it shares the grid with.

// The same cells on the rational route, whose member over this lane's grid is stored
// beside the Chebyshev one and whose relaxed multipliers are the stored pairs uncut.
// The body delegates this route's call to the scalar orders lane, so the declarations
// are here to keep a call site off a second copy of it.

#define BOYS_F32_ORDERS_PACKED_UNIFORM(kScheme, kBudget, kForm)                                    \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kChebyshev, kBudget,            \
                                                FitGranularity::kUniform, kForm>(                  \
        int nmax, float x, float* out) noexcept;                                                   \
    extern template void BoysAllOrdersF32Packed<kScheme, FitRoute::kRationalMinimax, kBudget,      \
                                                FitGranularity::kUniform, kForm>(                  \
        int nmax, float x, float* out) noexcept;

BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kRefinedReciprocal)

BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kRefinedReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kPlainReciprocal)
BOYS_F32_ORDERS_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kRefinedReciprocal)

#undef BOYS_F32_ORDERS_PACKED_REFERENCE
#undef BOYS_F32_ORDERS_PACKED_RUNGS
#undef BOYS_F32_ORDERS_PACKED_RUNG
#undef BOYS_F32_ORDERS_PACKED_NARROW
#undef BOYS_F32_ORDERS_PACKED_NARROW_RUNG
#undef BOYS_F32_ORDERS_PACKED_NARROW_RUNGS
#undef BOYS_F32_ORDERS_PACKED_UNIFORM
#undef BOYS_F32_ORDERS_PACKED_UNIFORM_RUNG
#undef BOYS_F32_ORDERS_PACKED_UNIFORM_RATIONAL_RUNG

// ---------------------------------------------------------------------------
// The all-orders batch over an argument array (BoysAllN)
// ---------------------------------------------------------------------------
// Every dispatch path is an interval of the argument line and the intervals are
// ordered, so the classification is monotone in x: a non-decreasing array is already
// contiguous by path, which is what the BoysSortedArgs overload declares and why one
// comparison per argument is the honest statement of it.
//
// A path is served by one of two kernel shapes. The ungrouped shape walks one
// argument at a time and writes the caller's planes; its bodies are the per-argument
// entry's own (BoysAllOrdersImpl) called at the batch's nmax, restructured onto the
// caller's layout and kept in lockstep with it, so at m = 1 the batch values ARE that
// entry's values, bit for bit, on every path it serves. (Region A's body seeds at
// nmax and recurses down, so a batch's F_k for k < nmax is that recurrence's value,
// not the one a per-order call at nmax = k walks; both are inside the entry's bound.)
// The grouped shape runs the region-A lane over a homogeneous run, one order at a
// time - the lane's shape is per order - staged through a fixed stack frame whatever
// count is.
//
// Which shape serves region A is decided by measurement, because the lane pays for
// every order separately: it evaluates each order from that order's own fit, while
// the scalar body evaluates one seed and recurses down. On a 65536-argument run,
// entry against a plain per-argument loop over the same arguments, the lane is 2.6x
// faster serving F_0, at parity at n = 4, and 10 to 20% slower from n = 8 up;
// kBoysAllNLaneMaxOrder is where the two cross.
//
// The zero path and region C run scalar in both shapes: the asymptotic form is a seed
// and nmax recursion steps written into the caller's planes, less work than the
// per-order lane, and the scalar body is the per-argument path's own.
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
// dispatch splits the band at the first extended-seed tier, and that split is a
// path of its own; at m > 1 the band does not exist and the whole region is one
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

// The band split of the classification: the classification has to reproduce
// the per-argument entry's dispatch, band and all.
constexpr double BoysAllNTierSplit() noexcept {
    return kTierThresholds[0];
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

// One argument's column, at `stride` doubles per order: plane[k * stride] = F_k(x)
// for k = 0..nmax. These bodies are the per-argument entry's, restructured onto the
// caller's layout; keep the two in lockstep.
//
// The stride is the caller's count when the kernel writes a column straight into the
// caller's planes, and the tile's width when the kernel stages one. No body reads the
// stride's value - it is a store address, never an operand, and it appears nowhere but
// in the index of a store - so the values do not depend on it, and a batch staged
// through a tile returns what the same bodies returned written straight into the
// caller's array.
inline void BoysAllNBodyZero(int nmax, std::size_t stride, double* plane) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * stride] = 1.0 / (2.0 * l + 1.0);
    }
}

// Region A below the band: the downward recursion from the per-order fit.
template <EvalPolicyLike Policy>
inline void BoysAllNBodyRegionADown(int nmax, double x, std::size_t stride, double* plane) noexcept {
    {
        double f = PolicyRegionAValue<Policy>(nmax, x);
        plane[static_cast<std::size_t>(nmax) * stride] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = DivideDownwardStep<Policy::kDivision>(l, x * f + expx);
            plane[static_cast<std::size_t>(l) * stride] = f;
        }
    }
}

// Region A's extended band: the per-range seed + upward recursion, with the
// per-order fits for the orders the tier does not reach. The classification
// admits only x >= kTierThresholds[0] here, which is the per-argument entry's
// guard on that condition - the band is this path.
template <EvalPolicyLike Policy>
inline void BoysAllNBodyRegionAExtended(int nmax,
                                        double x,
                                        std::size_t stride,
                                        double* plane) noexcept {
    int served = 0;

    while (served < nmax && x >= kTierThresholds[static_cast<std::size_t>(served + 1)])
    {
        ++served;
    }

    double f = RegionBExtendedSeed<Policy::kScheme>(x);
    plane[0] = f;
    const double expx = 0.5 * std::exp(-x);
    const double invx = StepReciprocal<Policy::kDivision>(x);

    for (int l = 1; l <= served; ++l)
    {
        f = DivideStep<Policy::kDivision>((l - 0.5) * f - expx, x, invx);
        plane[static_cast<std::size_t>(l) * stride] = f;
    }

    for (int l = served + 1; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * stride] = PolicyRegionAValue<Policy>(l, x);
    }
}

// Region B: the F0 seed + upward recursion.
template <EvalPolicyLike Policy>
inline void BoysAllNBodyRegionB(int nmax, double x, std::size_t stride, double* plane) noexcept {
    {
        double f = PolicyRegionBSeed<Policy>(x);
        plane[0] = f;
        const double expx = RegionBHalfExp<Policy::kRegionBExp>(x);
        const double invx = StepReciprocal<Policy::kDivision>(x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = DivideStep<Policy::kDivision>((l - 0.5) * f - expx, x, invx);
            plane[static_cast<std::size_t>(l) * stride] = f;
        }
    }
}

// Region C: the asymptotic form (m-invariant).
template <DivisionForm kForm = kDefaultDivisionForm>
inline void BoysAllNBodyRegionC(int nmax, double x, std::size_t stride, double* plane) noexcept {
    double f = kBoysHalfSqrtPi / std::sqrt(x);
    plane[0] = f;
    const double invx = StepReciprocal<kForm>(x);

    for (int l = 1; l <= nmax; ++l)
    {
        f = DivideStep<kForm>((l - 0.5) * f, x, invx);
        plane[static_cast<std::size_t>(l) * stride] = f;
    }
}

// The tile every kernel of the partitioned entry writes through.
//
// A run of the argument line is walked in tiles of kBoysAllNChunk arguments. Each tile
// is filled in a fixed stack frame by `fill` - the kernel's own body, filling it with
// the values that kernel produces - and then written to the caller's planes one order
// at a time: BoysAllNScatter turns one order's tile into one contiguous run of `block`
// doubles inside that plane.
//
// That is the whole reason the tile exists: writing an argument's column straight into
// the caller's planes puts consecutive stores count * 8 bytes apart - 33.5 MB on a
// four-million-argument call - so every store lands on a cache line of its own and is
// followed by a miss. Inside a tile a store is followed by its neighbour, and the
// plane stride is paid once per tile per order instead of once per element.
//
// The frame is the entry's own, (kMaxBoysOrder + 1) * kBoysAllNChunk doubles, the same
// size whatever `count` is: a batch's stack use does not grow with its argument count.
//
// `fill(base, block, stage)` writes the tile at stage[l * block + t] for order l and
// the tile's t-th argument, t < block. The tile's width is `block` and not
// kBoysAllNChunk, which is why it is passed rather than assumed.
template <typename Fill>
void BoysAllNTile(int nmax,
                  std::size_t count,
                  const std::size_t* index,
                  double* out,
                  std::size_t begin,
                  std::size_t end,
                  Fill fill) noexcept {
    std::array<double, (kMaxBoysOrder + 1) * kBoysAllNChunk> stage;

    for (std::size_t base = begin; base < end; base += kBoysAllNChunk)
    {
        const std::size_t block = std::min(kBoysAllNChunk, end - base);
        fill(base, block, stage.data());

        for (int l = 0; l <= nmax; ++l)
        {
            BoysAllNScatter(l,
                            count,
                            index,
                            base,
                            block,
                            stage.data() + static_cast<std::size_t>(l) * block,
                            out);
        }
    }
}

// The path's body, at one argument and one stride. Both writers below run it,
// and the stride is the only thing that differs between them.
template <BoysPath kPath, EvalPolicyLike Policy>
inline void BoysAllNBodyForPath(int nmax, double xi, std::size_t stride, double* plane) noexcept {
    if constexpr (kPath == BoysPath::kZero)
    {
        BoysAllNBodyZero(nmax, stride, plane);
    }
    else if constexpr (kPath == BoysPath::kTabulated)
    {
        BoysAllNBodyRegionADown<Policy>(nmax, xi, stride, plane);
    }
    else if constexpr (kPath == BoysPath::kExtended)
    {
        BoysAllNBodyRegionAExtended<Policy>(nmax, xi, stride, plane);
    }
    else if constexpr (kPath == BoysPath::kMiddle)
    {
        BoysAllNBodyRegionB<Policy>(nmax, xi, stride, plane);
    }
    else
    {
        BoysAllNBodyRegionC<Policy::kDivision>(nmax, xi, stride, plane);
    }
}

// The ungrouped kernel: one argument at a time over the run, the path's body per
// argument.
//
// Which of the two writers it uses is decided by where its stores land, a property of
// the caller's array and not of the shape. Both write the same values - the bodies are
// the per-argument path's own, called with the same arguments at the same nmax either
// way - so this is a choice about the stores alone, made by measurement:
//
//   the run's arguments are consecutive in the output. That is the sorted overload,
//     whose run's index IS the argument's own position, so the tile turns one order's
//     run into one contiguous run of stores: on the uniform stream at nmax 32 the
//     entry is 19.9% faster through it over three paired rounds, with the
//     machine-speed canary agreeing to 0.1%.
//   the run's arguments are the caller's positions in the caller's own order. That is
//     the unsorted entry, which sorted for the caller and must return the values in
//     the order it was handed them, so a plane's stores land wherever the permutation
//     puts them. The tile cannot make those consecutive, only less far apart, and
//     staging them costs more than that buys: the same measurement puts the unsorted
//     entry at +0.2% on the uniform stream and +13.8% on the molecular one. This
//     kernel therefore writes the unsorted entry's columns straight into the caller's
//     planes.
//
// The two writers are the same function and differ in one argument: what the body is
// told its stride is.
template <BoysPath kPath, EvalPolicyLike Policy>
void BoysAllNRunUngroupedDirect(int nmax,
                                const double* x,
                                double* out,
                                std::size_t count,
                                const std::size_t* index,
                                std::size_t begin,
                                std::size_t end) noexcept {
    for (std::size_t j = begin; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;
        BoysAllNBodyForPath<kPath, Policy>(nmax, x[i], count, out + i);
    }
}

template <BoysPath kPath, EvalPolicyLike Policy>
void BoysAllNRunUngroupedTiled(int nmax,
                               const double* x,
                               double* out,
                               std::size_t count,
                               std::size_t begin,
                               std::size_t end) noexcept {
    BoysAllNTile(nmax, count, nullptr, out, begin, end,
                 [&](std::size_t base, std::size_t block, double* stage) {
                     for (std::size_t t = 0; t < block; ++t)
                     {
                         // The body's stride is the tile's width: it writes the
                         // column it owns in the frame, and the tile's writes to
                         // the caller's planes are the scatter's.
                         BoysAllNBodyForPath<kPath, Policy>(
                             nmax, x[base + t], block, stage + t);
                     }
                 });
}

template <BoysPath kPath, EvalPolicyLike Policy>
void BoysAllNRunUngrouped(int nmax,
                          const double* x,
                          double* out,
                          std::size_t count,
                          const std::size_t* index,
                          std::size_t begin,
                          std::size_t end) noexcept {
    if (index == nullptr)
    {
        BoysAllNRunUngroupedTiled<kPath, Policy>(
            nmax, x, out, count, begin, end);
        return;
    }

    BoysAllNRunUngroupedDirect<kPath, Policy>(
        nmax, x, out, count, index, begin, end);
}

// The grouped kernel: the region-A lane over a homogeneous run, filling the same tile
// the scalar bodies fill. Both region-A paths take this shape when they take it at all
// - the lane covers [0, kX0), so which of the two scalar bodies the sort's path split
// assigned an argument to does not choose the kernel; the lane serves the band as
// well, at its band accuracy.
//
// Region B keeps its scalar body: BoysRegionBSimd does not hold its own documented
// bound across the region - at n = 32, x = kX0 it lands 2.2e-13 from the reference
// against its 5.5e-14 - and the scalar body is the per-argument path's own, exact to
// the bit.
inline void BoysAllNRunGrouped(int nmax,
                               const double* x,
                               double* out,
                               std::size_t count,
                               const std::size_t* index,
                               std::size_t begin,
                               std::size_t end) noexcept {
    std::array<double, kBoysAllNChunk> args;

    BoysAllNTile(nmax, count, index, out, begin, end,
                 [&](std::size_t base, std::size_t block, double* stage) {
                     for (std::size_t t = 0; t < block; ++t)
                     {
                         args[t] = x[(index != nullptr) ? index[base + t] : base + t];
                     }

                     // The region-A lane is per order, so the entry asks it for
                     // one order at a time, which is the shape the region-A fits
                     // have.
                     for (int l = 0; l <= nmax; ++l)
                     {
                         BoysRegionASimd(
                             l, args.data(), stage + static_cast<std::size_t>(l) * block, block);
                     }
                 });
}

// One run of one path, by the shape the tier can serve.
template <EvalPolicyLike Policy>
void BoysAllNRun(int nmax,
                 const double* x,
                 const std::size_t* index,
                 double* out,
                 std::size_t count,
                 bool vectorTier,
                 BoysPath path,
                 std::size_t begin,
                 std::size_t end) noexcept {
    // The packed region-A lane holds the shipped route's Chebyshev coefficients and
    // its split Clenshaw recurrence, so another route or another scheme takes the
    // scalar body here rather than the lane's values.
    //
    // BoysRegionASimd packs four ARGUMENTS of one order, so the grouped kernel is the
    // arguments axis's and only the arguments axis's: a call naming the orders axis is
    // served one argument at a time by the entry's per-argument path. The condition is
    // stated rather than inherited from that path, because which axis a packed lane
    // fills is the property this kernel implements.
    //
    // The narrow partition is the scalar bodies' here as well: the grouped kernel
    // reads the shipped piece table by index, one order at a time, so the granularity
    // joins the condition above rather than the lane, and a narrow policy trades away
    // the four-wide lane rather than a value it returns.
    if constexpr (Policy::kRoute == FitRoute::kChebyshev &&
                  Policy::kScheme == EvalScheme::kSplitClenshaw &&
                  Policy::kPack == PackAxis::kArguments &&
                  Policy::kGranularity == FitGranularity::kCoarsest)
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
        BoysAllNRunUngrouped<BoysPath::kZero, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kTabulated:
        BoysAllNRunUngrouped<BoysPath::kTabulated, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kExtended:
        {
            BoysAllNRunUngrouped<BoysPath::kExtended, Policy>(
                nmax, x, out, count, index, begin, end);
        }
        break;

    case BoysPath::kMiddle:
        BoysAllNRunUngrouped<BoysPath::kMiddle, Policy>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kAsymptotic:
        BoysAllNRunUngrouped<BoysPath::kAsymptotic, Policy>(
            nmax, x, out, count, index, begin, end);
        break;
    }
}

// The per-argument path, in the caller's planes - the shape the C entry point's
// batch loop already has, and the batch entry's total fallback.
//
// It writes its columns straight into the caller's planes, like the at-orders
// entry above and for the same reason: a column here is short where the shape that
// needs this path is the orders axis's, and staging short columns through a tile
// costs more than the stores it saves.
template <EvalPolicyLike Policy>
void BoysAllNRunPerArgument(int nmax,
                            const double* x,
                            double* out,
                            std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<Policy>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
    }
}

// The plane entry's arguments-axis shape: the dispatch read off the argument
// line, the sort over it, and one kernel per homogeneous run. Declared before
// the two entry wrappers that dispatch to it.
template <EvalPolicyLike Policy>
void BoysAllNPartitionedImpl(int nmax,
                             const double* x,
                             double* out,
                             std::size_t count,
                             std::size_t* workspace) noexcept;

template <EvalPolicyLike Policy>
void BoysAllNSortedPartitionedImpl(int nmax,
                                   const double* x,
                                   double* out,
                                   std::size_t count) noexcept;

// The plane entry. Two shapes serve it, and which one a call takes is the selectors
// it names, because each says what the call can be evaluated by:
//
//   the region-partitioned shape  an arguments-axis lane keeps four arguments of one
//                       order in a register, so the entry partitions its arguments by
//                       the interval their answer comes from and hands each
//                       homogeneous run to the shape that serves it. That shape
//                       evaluates the shipped fits as its own body, so it is what the
//                       shipped route and the arguments axis are served by. It is
//                       BoysAllNPartitionedImpl below.
//   the per-argument shape  every other call. An orders-axis lane keeps four orders of
//                       ONE argument in a register - out[k * count + i] is F_k(x[i]),
//                       so an argument's whole order vector is already what the entry
//                       writes - and a route other than the shipped one is carried by
//                       the all-orders entry's body, which takes its fit from the
//                       policy. The uniform partition is here too: the grid is read one
//                       order at a time from its own interval's coefficients, so there
//                       is no region walk for the grouping to sort by, and the body it
//                       reaches carries the partition's own branch. All three are the
//                       per-argument path, and none of them has anything for the region
//                       grouping to group.
//
// Every value either shape returns is inside the bound the entry documents: the
// partitioned shape's region-A lane answers at the per-order region-A bar, and the
// per-argument body is the all-orders entry's own.
//
// A relaxed multiplier on the orders axis reaches its rung the same way the shipped
// axis's does, in the engine itself (BoysAllOrdersImpl), and the route reaches its
// rung there too, so this entry states neither.
template <EvalPolicyLike Policy>
void BoysAllNImpl(int nmax,
                  const double* x,
                  double* out,
                  std::size_t count,
                  std::size_t* workspace) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    if constexpr (Policy::kPack == PackAxis::kOrders || Policy::kRoute != FitRoute::kChebyshev)
    {
        // The grouping scratch is the partitioned shape's: this path partitions
        // nothing and takes none of it.
        static_cast<void>(workspace);
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
    }
    else if constexpr (!FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                           Policy::kGranularity))
    {
        // The partition clause is the shared question: the partitioned body this
        // branch keeps out reaches its values through PolicyRegionAValue and
        // PolicyRegionBSeed, so a partition the Chebyshev family does not answer is
        // one that path would read out of the narrow tables under the name it was
        // given. It is served by the path above's body, not by the partitioned one.
        // This entry groups arguments by the region walk each of them takes, and
        // the grid has no walk to group: an order is read from its own interval's
        // coefficients and nothing is built from another order, so there is no
        // seed to share across a run and no rung to cut. The body it reaches
        // carries the partition's own branch - the grid's table below its join,
        // the asymptotic form above it - which is what makes the cells this
        // branch serves the grid's numbers rather than another partition's.
        static_cast<void>(workspace);
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
    }
    else
    {
        // The guard belongs here and not at the top of the function. This path
        // reads the shipped and narrow tables directly and has no uniform branch,
        // so a uniform policy would be answered from another partition's fits -
        // the substitution this refuses. Neither path above needs a guard: both
        // hand every argument to BoysAllOrdersImpl, which reads the grid where the
        // policy names it.
        RefuseUniformPartition<Policy>();
        BoysAllNPartitionedImpl<Policy>(nmax, x, out, count, workspace);
    }
}

template <EvalPolicyLike Policy>
void BoysAllNPartitionedImpl(int nmax,
                             const double* x,
                             double* out,
                             std::size_t count,
                             std::size_t* workspace) noexcept {
    // The path's own contract, one level below the entry's guard: every read below
    // resolves the policy's partition through the Chebyshev family - the region
    // bodies reach PolicyRegionAValue, PolicyRegionBSeed and the two rung reads -
    // and that family does not answer the grid. The entry routes such a policy to
    // the per-argument path and asks RefuseUniformPartition before delegating here,
    // so this refuses nothing the entry serves; what it refuses is a call that
    // reaches the partitioned shapes without that routing, which is the shape every
    // one of the three substitution defects had.
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this is the partitioned path: it groups its arguments by the region walk "
                  "each of them takes and reads each group through the Chebyshev family, which "
                  "carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one. A policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Route it to the "
                  "per-argument path, which reads the grid's own table");
    RequireShippedRoute<Policy>();
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

    const bool vectorTier = BoysAvx2Available();
    const double tierSplit = kTierThresholds[0];

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
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
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
            BoysAllNRun<Policy>(
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
template <EvalPolicyLike Policy>
void BoysAllNSortedImpl(int nmax,
                        const double* x,
                        double* out,
                        std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    if constexpr (Policy::kPack == PackAxis::kOrders || Policy::kRoute != FitRoute::kChebyshev)
    {
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
    }
    else if constexpr (!FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                           Policy::kGranularity))
    {
        // As in BoysAllNImpl, and for the same reason: the partition clause is the
        // shared question, the grid has no walk for this entry's grouping to sort
        // by, and the body the path above reaches - BoysAllOrdersImpl - carries the
        // partition's own branch, so the grid is read where the policy names it.
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
    }
    else
    {
        // As in BoysAllNImpl: the guard covers the path that reads the shipped and
        // narrow tables with no uniform branch, and not the two that delegate.
        RefuseUniformPartition<Policy>();
        BoysAllNSortedPartitionedImpl<Policy>(nmax, x, out, count);
    }
}

template <EvalPolicyLike Policy>
void BoysAllNSortedPartitionedImpl(int nmax,
                                   const double* x,
                                   double* out,
                                   std::size_t count) noexcept {
    // As in BoysAllNPartitionedImpl, and for the same reason: the path's own
    // contract one level below the entry's guard, on the same read of the same
    // family.
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this is the partitioned path: it groups its arguments by the region walk "
                  "each of them takes and reads each group through the Chebyshev family, which "
                  "carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one. A policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Route it to the "
                  "per-argument path, which reads the grid's own table");
    RequireShippedRoute<Policy>();
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

    const bool vectorTier = BoysAvx2Available();
    const double tierSplit = kTierThresholds[0];

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

        BoysAllNRun<Policy>(nmax, x, nullptr, out, count, vectorTier, path, begin, end);
        begin = end;
    }
}

// The per-element-top-order batch: the per-argument all-orders body at each argument's
// own top order, written into the caller's planes. The tops differ per element, so a
// run has no common nmax to be grouped at and this entry has nothing to group; its
// whole content is the layout, and the cells above each column's own top order, which
// it leaves exactly as the caller left them.
//
// The body is the per-argument entry's, so this entry takes the policies that entry
// takes - including a named fit route, which the plane entry does not.
//
// The columns are written straight into the caller's planes and not staged through the
// tile the plane entry's kernels use, because here the tile costs more than it saves:
// a column here is short - the molecular stream's tops average about one order - so a
// tile of them has to be scattered one plane at a time over the whole tile's width to
// reach the few cells each column owns, a full pass per order for every column that
// reaches that order. With the tile this entry was 24.7% slower on the uniform stream
// and 54.6% slower on the molecular one, the latter over four repeats of an
// alternating before/after pair with the canary agreeing to 1%.
template <EvalPolicyLike Policy>
void BoysAllNAtOrdersImpl(const int* n, const double* x, double* out, std::size_t count) noexcept {
    // No RefuseUniformPartition here, and the delegation is what earns the
    // exception: every value below comes from BoysAllOrdersImpl, which reads the
    // uniform grid where the policy names it (AllOrdersBody, UniformAllOrders). A
    // guard at this entry would refuse a combination this body serves.
    assert(count == 0 || n != nullptr);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(n[i] >= 0 && n[i] <= kMaxBoysOrder);
        assert(x[i] >= 0.0);

        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<Policy>(n[i], x[i], row);

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

template <EvalPolicyLike Policy>
double BoysSingle(int n, double x) noexcept {
    return detail::BoysSingleImpl<Policy>(n, x);
}

template <EvalPolicyLike Policy>
void BoysAllOrders(int nmax, double x, double* out) noexcept {
    detail::BoysAllOrdersImpl<Policy>(nmax, x, out);
}

template <EvalPolicyLike Policy>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    detail::BoysFixedNImpl<Policy>(n, x, out, count, stride);
}

template <EvalPolicyLike Policy>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace) noexcept {
    detail::BoysAllNImpl<Policy>(nmax, x, out, count, workspace);
}

template <EvalPolicyLike Policy>
void BoysAllN(int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept {
    detail::BoysAllNSortedImpl<Policy>(nmax, x, out, count);
}

template <EvalPolicyLike Policy>
void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept {
    detail::BoysAllNAtOrdersImpl<Policy>(n, x, out, count);
}

// The combination named in the type and the rung named in the call: the rung
// ladder is the one place the two selections meet, and a caller pays for it in
// one branch, with the four structural axes resolved at the call site.

template <EvalPolicyLike Policy>
float BoysSingleF32(int n, float x) noexcept {
    return detail::BoysSingleF32Impl<Policy>(n, x);
}

template <EvalPolicyLike Policy>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept {
    detail::BoysAllOrdersF32Impl<Policy>(nmax, x, out);
}

template <EvalPolicyLike Policy>
void BoysAllNF32(int nmax, const float* x, float* out, std::size_t count) noexcept {
    detail::BoysAllNF32Impl<Policy>(nmax, x, out, count);
}

template <EvalPolicyLike Policy>
void BoysFixedNF32(
    int n, const float* x, float* out, std::size_t count, std::size_t stride) noexcept {
    detail::BoysFixedNF32Impl<Policy>(n, x, out, count, stride);
}

template <EvalPolicyLike Policy>
void BoysAllNAtOrdersF32(const int* n, const float* x, float* out, std::size_t count) noexcept {
    detail::BoysAllNAtOrdersF32Impl<Policy>(n, x, out, count);
}


#if BoysFp16
// The fp16/bf16 lanes forward the multiplier and the policy to the F32 engine with the
// fp16 computation budget (the m*1e-7 + 1/2-ULP formula); at m = 1 the engine branch
// is the certified F32 path verbatim, so the lanes are bit-unchanged, and the half-ULP
// representation term is m-independent.
//
// The policy is the caller's, defaulted to the lane's own: the five axes a policy
// carries are the option space's, and every combination this lane's book carries is a
// policy a consumer can name, so an entry that took no policy would be three quarters
// of the lane's cells with no way to ask for them. The budget is not one of those axes
// - it is what makes this lane the half lane - so it is DefaultPolicyFp16's rather
// than the caller's, and a policy named here is read for its route, scheme, partition,
// packing axis and division form.

// The budget a policy named on a half lane has to carry: it is the axis that makes
// this lane the half lane, so a policy built at the float lane's budget names the
// float lane's combination, and reaching this entry with one would answer that
// combination under this lane's name. It is refused where it is named rather than
// honoured: a half-precision lane quietly computing the float lane's combination is
// exactly the failure a name that does not mean what it says.
template <EvalPolicyLike Policy>
F16 BoysSingleF16(int n, F16 x) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    return static_cast<F16>(detail::BoysSingleF32Impl<Policy>(
        n, static_cast<float>(x)));
}

template <EvalPolicyLike Policy>
void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<Policy>(nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<F16>(scratch[l]);
    }
}

template <EvalPolicyLike Policy>
Bf16 BoysSingleBf16(int n, Bf16 x) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    return static_cast<Bf16>(detail::BoysSingleF32Impl<Policy>(
        n, static_cast<float>(x)));
}

template <EvalPolicyLike Policy>
void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<Policy>(nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<Bf16>(scratch[l]);
    }
}

// The half lane's array shapes: the float lane's own bodies at each argument,
// the results stored to half once, exactly as BoysAllOrdersF16 stores one
// argument's ladder. Nothing is computed in half here beyond the store - the
// arithmetic is the fp32 engine's, which is what makes these the fp16 lane's
// entries rather than a second engine - and each element is the fp16 lane's
// single or all-orders entry at that argument, so the bound is that entry's.
template <EvalPolicyLike Policy>
void BoysFixedNF16(
    int n, const F16* x, F16* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x != nullptr);
    assert(out != nullptr);
    assert(stride >= 1);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= static_cast<F16>(0.0f));
        out[i * stride] = static_cast<F16>(
            detail::BoysSingleF32Impl<Policy>(n, static_cast<float>(x[i])));
    }
}

template <EvalPolicyLike Policy>
void BoysAllNF16(int nmax, const F16* x, F16* out, std::size_t count) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= static_cast<F16>(0.0f));

        float row[kMaxBoysOrder + 1];
        detail::BoysAllOrdersF32Impl<Policy>(nmax, static_cast<float>(x[i]), row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = static_cast<F16>(row[l]);
        }
    }
}

template <EvalPolicyLike Policy>
void BoysAllNAtOrdersF16(const int* n, const F16* x, F16* out, std::size_t count) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(count == 0 || n != nullptr);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(n[i] >= 0 && n[i] <= kMaxBoysOrder);
        assert(x[i] >= static_cast<F16>(0.0f));

        float row[kMaxBoysOrder + 1];
        detail::BoysAllOrdersF32Impl<Policy>(n[i], static_cast<float>(x[i]), row);

        for (int l = 0; l <= n[i]; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = static_cast<F16>(row[l]);
        }
    }
}

// The half lanes' rung-argument entries: the fp16 engine's ladder through the
// store this lane makes, at the policy named, so the two selections meet here as
// they do on the double lane.

#endif // BoysFp16

} // namespace boys

/// \endcond
