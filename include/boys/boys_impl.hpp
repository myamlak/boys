#pragma once

// The template bodies behind boys/boys.hpp's evaluation surface (that header includes this file
// last, never alone). SeamPolicy (policy.hpp) carries every caller choice - route, scheme,
// budget - and each entry compiles one body at the one accuracy this library carries: the
// certified body, whose seeds read the degrees stored in boys_effective_degrees.hpp.

/// \cond
// Not API: the kernel the entries are compiled from. The header ships only
// because the entries are header-defined.

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/policy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_flat_grid.hpp"
#include "boys/boys_default_tables.hpp"
#include "boys/boys_fit_core.hpp"
#include "boys/boys_fit_chebyshev.hpp"
#include "boys/boys_fit_uniform.hpp"
#include "boys/boys_fit_rational.hpp"
#include "boys/boys_fit32.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/isa_tiers.hpp"
#include "boys/dependent_false.hpp"
#include "boys/status.hpp"

#if BoysFp16
#include "boys/f16.hpp"
#include "boys/half2.hpp"
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>

namespace boys {

namespace detail {

constexpr double kBoysHalfSqrtPi = 0.886226925452758014; // sqrt(pi)/2
constexpr float kBoysHalfSqrtPiF32 = 0.88622693f; // sqrt(pi)/2, single lane




// The region-B body seeds F_0 and steps to the top order with F_{l+1} = ((l + 1/2) F_l - t)/x,
// t = e^{-x}/2, so an error d in t reaches F_N multiplied by the products of the steps after it:
//   T(N, x) = (1/(2x)) * sum_{l=0}^{N-1} prod_{j=l+1}^{N-1} (2j+1)/(2x),  read at N = 32.
// The ladder's sensitivity is exactly twice that, dF_N/dt = -2 T(N, x), checked symbolically.

// The promise admits the term while |delta| <= kRegionBExpBar / (2 T(N, x)): near kX0 the ladder's
// gain, 2 T(N, x) t / F_N, is 1.598e6 and only the library routine is admissible, while above
// kRegionBExpCheapFrom the polynomial stands ten times inside.
inline constexpr double kRegionBExpBar = 1.0e-14;

// The smallest x whose requirement reaches ten times the fast member's own error, solved from
// bar / (2 T(32, x) t) at 60 digits against the member's measured 5.465e-16 over region B: no x
// below it satisfies the rule. A narrower ladder asks less, so what it buys at N = 32 it buys at
// every N <= 32, and sitting below kX0 is what lets the polynomial arm serve all of region B.
inline constexpr double kRegionBExpCheapFrom = 11.202305053129882;

// e^{-r} on |r| <= ln 2/2, degree 10, Chebyshev fit: 4.458e-16 relative with its coefficients
// rounded to double, the floor they carry; degree 9 delivers 2.819e-14 and misses the cut's
// target, degree 11 and up read the evaluation's own rounding, and the member delivers 5.465e-16
// over region B. The old degree-7 array carried 8.336e-11, 2.44e-07 at the gate's 1e-13 bar.
inline constexpr double kRegionBExpReduced[11] = {
    1.0,
    -1.0000000000000122,
    0.500000000000001,
    -0.16666666666493138,
    0.041666666666522106,
    -0.008333333404031027,
    0.0013888888947785523,
    -0.00019841149885492404,
    2.4801487366025675e-05,
    -2.764772381929414e-06,
    2.7632640675430235e-07,
};

inline constexpr double kRegionBExpLog2e = 1.4426950408889634073599246810018921;

// ln 2 in two pieces, the head being the double ln 2 with its lowest 25 significand bits cleared
// and the tail the rest: head + tail is ln 2 to 4.0e-26, and kd * head is exact for every kd a
// region-B or a region-C argument reaches. The reduction's first subtraction is exact with it,
// so only the second rounds: the member reads 2.927e-15 with one constant, 5.465e-16 with the split.
inline constexpr double kRegionBExpLn2Head = 0.6931471787393093;
inline constexpr double kRegionBExpLn2Tail = 1.8206359985041462e-09;
inline constexpr double kRegionBExpRoundMagic = 6755399441055744.0; // 1.5 * 2^52

/// 0.5 * e^{-x} for a region-B argument, in the member the policy named.
///
/// The accurate member is the library routine, and it is not a second body: both
/// lanes' band and downward seeds run it at every member of the axis too, because
/// those are region A and the axis is region B's.
///
/// The fast member is the reduced-argument polynomial, x = k ln2 + r with
/// |r| <= ln2/2, k out of a magic constant and 2^{-k} taken from the exponent
/// field. Over region B k is in [17, 42], far from the exponent field's ends, so
/// the scale is exact. Below kRegionBExpCheapFrom the ladder's own requirement
/// (kRegionBExpBar, above) is tighter than the member's error and this member
/// reads the library routine there: that arm is part of the member, because a
/// polynomial alone would fail a bound over a band interior to region B. It
/// delivers 5.465e-16 relative at worst over region B, the only domain a policy
/// reads it in, and 2.927e-15 at the same arithmetic with one ln 2 constant in
/// place of the split below - which is what the constant above is solved against,
/// the split being what keeps the member 71 times inside the requirement at kX0
/// rather than 13.
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
        const double r = (x - kd * kRegionBExpLn2Head) - kd * kRegionBExpLn2Tail;

        double p = kRegionBExpReduced[10];

        for (int k = 9; k >= 0; --k)
        {
            p = p * r + kRegionBExpReduced[k];
        }

        const int k = static_cast<int>(kd);
        return 0.5 * std::bit_cast<double>(static_cast<std::uint64_t>(1023 - k) << 52) * p;
    }
}

// The same member at the single-precision lane's own type. The accurate member is 0.5f * expf, the
// arithmetic that lane's published figures were measured at - not the double one narrowed, which
// would move values the lane's documents speak for. The fast member is the double arithmetic
// rounded once: the widening is exact, and the narrowing's half ulp is the member's whole error.
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





// The far region's stored representation: a fit of the scaled H_n = 2 F_n x^(n+1/2)/Gamma(n+1/2) -
// order one where F spans twenty orders - read back by one multiply per order (kScaledTable).
constexpr bool ScaledFarDegreesCarried() noexcept {
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kScaledFarIntervals); ++iv)
    {
        const int deg = detail::kScaledFarDegs[iv];

        if (deg < 4 || deg % 2 != 0 || deg > detail::kScaledFarReadCap)
        {
            return false;
        }
    }

    return true;
}

static_assert(ScaledFarDegreesCarried(),
              "every interval of the scaled far table must be fitted at an even degree "
              "between 4 and the read cap: an odd degree would drop the leading odd "
              "coefficient the split Clenshaw summation seeds from, and a degree above the "
              "cap is beyond the coefficients the interval carries");

/// Which leg of the library a far ladder is instantiated in.
///
/// Not an arithmetic choice: the two instantiations are one body, and the tag says
/// where that body is compiled. It is a tag rather than a second name because a
/// shared definition cannot serve both legs. The far ladders' arithmetic contains
/// `std::fma` - the refined form's exact product, and ClenshawSplit's multiply-add
/// at every form - which is the fused instruction in a unit licensed for it and a
/// call into the C runtime in one that is not. One definition of a body that is
/// reached from both a licensed and an unlicensed unit is one weak symbol, the
/// link resolves it to whichever unit it meets first, and the leg that asked for
/// the instruction runs the runtime call instead. That is not a hypothetical: it
/// is what a single shared definition measured, over the whole far ladder, at
/// 28.84 instructions per order against a fused step's 6.09.
///
/// So the body carries the leg, each leg's instantiations are named once and
/// defined in the unit that leg belongs to, and the linker has nothing to choose.
/// The two legs deliver the same bits - a fused step and the library's `fma` are
/// the same correctly rounded operation, which the bit comparison against the
/// pre-change build is what establishes - so what the tag decides is where the
/// instruction comes from, never what arithmetic runs.
enum class LadderLeg {
    /// The leg that asks: the body of the unit compiled without the fused instruction set, which
    /// reaches it as a call. The scalar tier's arm.
    kAskingLeg,
    /// The leg of boys_orders_simd.cpp, which carries the fused instruction set. The vector
    /// tier's arm, defined in that unit alone.
    kFusedLeg,
};

/// Where an argument sits on the far partition: the interval it falls in, that
/// interval's first coefficient, and the argument mapped into it.
///
/// The partition's intervals are not of one width - a derived one is graded, and
/// the placeholder is equal only because nobody has derived one - so the interval
/// is found by bisection over the edges and not by a product. The edges are the
/// partition's data; that they increase is the only fact about them a reader is
/// entitled to assume, and kScaledFarEdges' own static_assert is what makes it one.
struct FarPoint {
    std::size_t iv; ///< the interval the argument falls in
    std::size_t block; ///< kScaledFarOffsets[iv], this interval's order-0 coefficient
    double t; ///< the argument mapped into [-1, 1) on that interval
};

inline FarPoint FarLocate(double x) noexcept {
    constexpr std::size_t kIntervals = static_cast<std::size_t>(detail::kScaledFarIntervals);
    // The largest iv with kScaledFarEdges[iv] <= x, the caller having established
    // kScaledFarX0 <= x < kScaledFarX1 and the edges tiling exactly that span.
    std::size_t lo = 0;
    std::size_t hi = kIntervals;

    while (hi - lo > 1)
    {
        const std::size_t mid = lo + (hi - lo) / 2;

        if (x < detail::kScaledFarEdges[mid])
        {
            hi = mid;
        } else
        {
            lo = mid;
        }
    }

    const double a = detail::kScaledFarEdges[lo];
    const double b = detail::kScaledFarEdges[lo + 1];

    return FarPoint{
        lo, static_cast<std::size_t>(detail::kScaledFarOffsets[lo]), 2.0 * (x - a) / (b - a) - 1.0};
}

/// One order off the scaled far table, at the split Clenshaw its coefficients
/// were certified in - the same summation, operation for operation, as the
/// generator measures the table's delivered error with.
inline double ScaledFarOrderAt(const FarPoint& at, int l) noexcept {
    const int deg = detail::kScaledFarDegs[at.iv];
    const double* c = detail::kScaledFarCoeffs.data() + at.block +
                      static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

    return ClenshawSplit<backend::ScalarFp64>(c, deg, at.t);
}

/// One argument's whole ladder off the scaled far table, as the portable arm of
/// the entry below writes it, with the recovery the stored function needs: H_n is
/// what the table holds and F_n is what a caller receives, and the two differ by
/// m_n = Gamma(n+1/2) / (2 x^(n+1/2)).
///
/// The running form of m is the region's own (BoysAllNBodyRegionC's ladder step
/// for step), so the recovery costs the far call the arithmetic it already paid
/// rather than a second one: one multiply-add per order here, and the division
/// form is the policy's, as every other ladder's is.
///
/// The table is read by ClenshawSplit over the scalar arithmetic, whose
/// multiply-add is `std::fma` at every division form - so this ladder's cost is
/// the asking unit's for the same reason the corrected seed's is.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, kScaledFarX0 <= x < kScaledFarX1
/// \param out  receives nmax + 1 values, out[k] = F_k(x)
template <DivisionForm kForm, LadderLeg kLeg>
void ScaledFarLadder(int nmax, double x, double* out) noexcept {
    static_assert(
        kCoefficientGroupPresent<
            DependentGroup<std::integral_constant<DivisionForm, kForm>, ScaledFarGroup>> ||
            kAlwaysFalse<std::integral_constant<DivisionForm, kForm>>,
        "the scaled far table reads boys/boys_coefficients_d_far.hpp: include that "
        "header in the translation unit that names the kScaledTable far "
        "representation");
    const FarPoint at = FarLocate(x);
    double m = kBoysHalfSqrtPi / std::sqrt(x);
    const double invx = StepReciprocal<kForm>(x);
    out[0] = ScaledFarOrderAt(at, 0) * m;

    for (int l = 1; l <= nmax; ++l)
    {
        m = DivideStep<kForm>((l - 0.5) * m, x, invx);
        out[l] = ScaledFarOrderAt(at, l) * m;
    }
}

/// One argument's whole ladder off the scaled far table, with the recovery the
/// stored function needs: H_n is what the table holds and F_n is what a caller
/// receives, and the two differ by m_n = Gamma(n+1/2) / (2 x^(n+1/2)).
///
/// Two bodies of one arithmetic, chosen as CorrectedSeedAllOrders' are: the fused
/// leg where this unit carries the fused instruction set, the asking leg everywhere else.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, kScaledFarX0 <= x < kScaledFarX1
/// \param out  receives nmax + 1 values, out[k] = F_k(x)
template <DivisionForm kForm> void ScaledFarAllOrders(int nmax, double x, double* out) noexcept {
#if BOYS_ISA_AVX2_FMA
    ScaledFarLadder<kForm, LadderLeg::kFusedLeg>(nmax, x, out);
#else
    ScaledFarLadder<kForm, LadderLeg::kAskingLeg>(nmax, x, out);
#endif
}

// The far region's third representation: the default seed and ladder with their two errors
// repaired - the H_0 = 1 held in the seed, and the exp(-x)/2 term the asymptotic step drops.

// The correction's degree, held to the reader's contract rather than assumed, for
// the reason the far table's degrees are: the split Clenshaw's odd part is the D
// family's finalization and is derived for an even degree of at least 4, so a
// degree that did not meet it would be summed into a different polynomial.
constexpr bool SeedCorrectionDegreeCarried() noexcept {
    return detail::kSeedCorrectionDeg >= 4 && detail::kSeedCorrectionDeg % 2 == 0;
}

static_assert(SeedCorrectionDegreeCarried(),
              "the corrected seed's fit must be at an even degree of at least 4: an odd "
              "degree would drop the leading odd coefficient the split Clenshaw summation "
              "seeds from");

static_assert(detail::kSeedCorrectionX0 == detail::kX1,
              "the corrected seed's fit must open at kX1, where the region it corrects does");

/// The correction the closed form is missing, H_0 - 1, at an argument inside its
/// span. The caller reads it as 1.0 + this, which is where the 1 comes back.
inline double SeedCorrectionAt(double x) noexcept {
    const double t = 2.0 * (x - detail::kSeedCorrectionX0) /
                         (detail::kSeedCorrectionX1 - detail::kSeedCorrectionX0) -
                     1.0;

    return ClenshawSplit<backend::ScalarFp64>(detail::kSeedCorrectionCoeffs.data(),
                                              detail::kSeedCorrectionDeg, t);
}

/// The far region's seed at one argument: the closed form with the error the closed
/// form's own truncation carries, over the span where that error is above the roundoff.
///
/// Every ladder above \c kX1 seeds here, which is what keeps the region one arithmetic:
/// the ladder's step adds the \c exp(-x)/2 term the asymptotic step drops, so a ladder
/// seeded at the bare closed form would be stepping a different function rather than a
/// coarser one.
///
/// \param x argument, kX1 <= x
inline double FarSeedAt(double x) noexcept {
    double f = kBoysHalfSqrtPi / std::sqrt(x);

    if (x < detail::kSeedCorrectionX1)
    {
        f *= 1.0 + SeedCorrectionAt(x);
    }

    return f;
}

/// One argument's whole ladder from the corrected seed, in the leg named.
///
/// The exp is the accurate member of region B's axis, which is the library
/// routine; the fast member is region B's bound's business and is not measured
/// over this region. The division form is the policy's, as every ladder's is.
///
/// The leg is what the body's `std::fma` - the refined form's exact product, and
/// the seed correction's ClenshawSplit - is compiled as: see LadderLeg.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, kX1 <= x
/// \param out  receives nmax + 1 values, out[k] = F_k(x)
template <DivisionForm kForm, LadderLeg kLeg>
void CorrectedSeedLadder(int nmax, double x, double* out) noexcept {
    double f = FarSeedAt(x);

    const double invx = StepReciprocal<kForm>(x);
    const double expx = RegionBHalfExp<RegionBExp::kAccurate>(x);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = UpwardStep<kForm>(l, f, x, invx, expx);
        out[l] = f;
    }
}

/// One argument's whole ladder from the corrected seed.
///
/// Two legs of one arithmetic, chosen by the tier the compiling unit carries: the
/// fused leg where the fused instruction set is in force, which is where the
/// refined form's exact product costs one instruction, and the asking leg
/// everywhere else, which is where it costs a call. The predicate is the vector
/// tier's own - the fused leg's unit is compiled for AVX2, so entering it is its
/// licence - and a unit without it takes the asking leg, which is the certified
/// lane and is present on every target.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, kX1 <= x
/// \param out  receives nmax + 1 values, out[k] = F_k(x)
template <DivisionForm kForm> void CorrectedSeedAllOrders(int nmax, double x, double* out) noexcept {
#if BOYS_ISA_AVX2_FMA
    CorrectedSeedLadder<kForm, LadderLeg::kFusedLeg>(nmax, x, out);
#else
    CorrectedSeedLadder<kForm, LadderLeg::kAskingLeg>(nmax, x, out);
#endif
}

// The far ladder's coefficients: A[m] = (2m-1)!!, exact as doubles to m = 15 (29!! = 6.19e15,
// under 2^53) and rounded from m = 16. The ratio A[m] = (2m - 1) A[m - 1] is what makes a term
// one table read and two multiplies; every entry agrees bit for bit with gamma(m + 1/2) * 2^m /
// sqrt(pi) at 60 digits, all 33 of them. Aligned 32 for the four-orders lane's group loads.
alignas(32) inline constexpr double kDoubleFactorial[kMaxBoysOrder + 1] = {
    1.0,                    // A[0], the empty product
    1.0,                    // 1
    3.0,                    // 3
    15.0,                   // 15
    105.0,                  // 105
    945.0,                  // 945
    10395.0,                // 10395
    135135.0,               // 135135
    2027025.0,              // 2027025
    34459425.0,             // 34459425
    654729075.0,            // 654729075
    13749310575.0,          // 13749310575
    316234143225.0,         // 316234143225
    7905853580625.0,        // 7905853580625
    213458046676875.0,      // 213458046676875
    6190283353629375.0,     // 6190283353629375, the last integer the double holds exactly
    1.9189878396251062e+17, // 31!!
    6.33265987076285e+18,   // 33!!
    2.2164309547669976e+20, // 35!!
    8.200794532637892e+21,  // 37!!
    3.1983098677287775e+23, // 39!!
    1.3113070457687988e+25, // 41!!
    5.638620296805835e+26,  // 43!!
    2.5373791335626256e+28, // 45!!
    1.1925681927744342e+30, // 47!!
    5.843584144594727e+31,  // 49!!
    2.980227913743311e+33,  // 51!!
    1.5795207942839547e+35, // 53!!
    8.687364368561751e+36,  // 55!!
    4.951797690080198e+38,  // 57!!
    2.921560637147317e+40,  // 59!!
    1.7821519886598634e+42, // 61!!
    1.1227557528557138e+44, // 63!!
};

// The highest order the far region answers with the upward recurrence instead of the ladder's
// independent terms. The two bodies' cost was measured to cross here: on the argument stream the
// far region is timed over, the recurrence is the cheaper body at every order up to this one and
// the ladder's are cheaper from n = 16 on, so the four lowest orders take it, in every request.
inline constexpr int kFarRecurOrders = 4;

/// The far region's orders up to \c kFarRecurOrders, stepped as a recurrence: the same terms
/// \c F_m = F_0 A[m] h^m, taken \c F_{m+1} = F_m (2m + 1) h from the value the ladder seeds with.
///
/// The stepper \c (2m + 1) h rides an add chain of its own, so one term costs one multiply on the
/// dependent chain where the ladder pays a table read and two multiplies an order and forms a
/// power every four. The values are the ladder's to within a rounding of the seed: measured
/// against the ladder over the far region's stream, the largest divergence scaled by \c F_0 is
/// 7.9e-19, five orders of magnitude under the 1e-13 the region publishes.
///
/// \param nmax   highest order, 0..kFarRecurOrders
/// \param r      the seed quotient 1/sqrt(x), kX1 <= x, formed by the caller
/// \param stride distance between two orders in \c out
/// \param out    receives nmax + 1 values, out[l * stride] = F_l(x)
inline void BoysFarRecurLowOrdersFromR(int nmax, double r, std::size_t stride,
                                       double* out) noexcept {
    assert(nmax >= 0 && nmax <= kFarRecurOrders);

    double f = kBoysHalfSqrtPi * r;
    out[0] = f;

    if (nmax == 0)
    {
        return;
    }

    const double h = 0.5 * (r * r);
    const double step = 2.0 * h;
    double c = h;

#if BOYS_ISA_AVX2_FMA
    // The orders in line for a runtime nmax, each carrying the same two multiplies the loop
    // stepped - the store's address advanced by the stride, same terms, order and roundings.
    // The vector tier's own body: the scalar tier keeps the loop below, which the arm64 unit
    // MSVC builds compiles where this body crashes its back end (C1001, at /O2).
    double* p = out + stride;

    f = f * c;
    *p = f;

    if (nmax == 1)
    {
        return;
    }

    c = c + step;
    p += stride;
    f = f * c;
    *p = f;

    if (nmax == 2)
    {
        return;
    }

    c = c + step;
    p += stride;
    f = f * c;
    *p = f;

    if (nmax == 3)
    {
        return;
    }

    c = c + step;
    p += stride;
    f = f * c;
    *p = f;
#else
    for (int m = 1; m <= nmax; ++m)
    {
        f = f * c;
        out[static_cast<std::size_t>(m) * stride] = f;
        c = c + step;
    }
#endif
}

/// The recurrence at the caller's own argument: forms the seed quotient and defers to the core
/// above. A caller about to walk the region hands the core its own \c r instead, so the pair
/// pays one sqrt with the walk, not two.
///
/// \param nmax   highest order, 0..kFarRecurOrders
/// \param x      argument, kX1 <= x
/// \param stride distance between two orders in \c out
/// \param out    receives nmax + 1 values, out[l * stride] = F_l(x)
inline void BoysFarRecurLowOrders(int nmax, double x, std::size_t stride, double* out) noexcept {
    BoysFarRecurLowOrdersFromR(nmax, 1.0 / std::sqrt(x), stride, out);
}

#if BOYS_ISA_AVX2_FMA
// The far ladder's four-orders lane, defined by boys/boys_orders_lanes.hpp (included at the bottom
// of this header at the tier this unit's flags name). The run below hands it its contiguous call,
// so it is declared here the way the arguments lane's kernels are declared below their bodies.
void BoysFarLadderOrdersLane(int nmax, double x, std::size_t stride, double* out) noexcept;
#endif

/// The far region's ladder, written into a run: out[l * stride] = F_l(x), l = 0..nmax.
///
/// The region's terms are the asymptotic series' own, \c F_m = F_0 A[m] h^m with
/// \c h = 1/(2x) and \c A[m] the double factorial above. There is no upward step and
/// no \c exp(-x)/2 term in it: the step the region-B body takes carries that term,
/// and over this region the closed form's own truncation \c erfc(sqrt(x)) stands in
/// for it, one rounding of the seed rather than a term the ladder subtracts 32 times.
///
/// What the arithmetic delivers, at the far region's own figures: at \c kX1 it is
/// 4.31e-15 at order 0 and 5.0e-14 at n = 32, the series' tail
/// \c 0.5 x^-(m+1/2) Gamma(m + 1/2, x) reaching its largest value at the region's
/// left edge and falling away with x. The far region's fits are cut at an absolute
/// 5.5e-14, which covers every one of them, and the ladder-scaled figure the entries
/// publish - 1e-13 of \c F_0(x), the contract's wider arm - does not cover the tail
/// below \c detail::kFlatHi: at x = 29 the tail at n = 32 is 4.94e-14 against a bar
/// of 1.65e-14 there, three times it, while at the cut itself, x = 31.0, it reads
/// 0.288 of the bar, 3.47x to spare. That is why this body is entered only at or
/// above \c detail::kFlatHi, the grid's own hi and so the arm's own left edge. It
/// is also why \c FarRepresentation::kCorrectedSeed pays for the reading that holds
/// the figure across the region, keeping the recurrence above.
///
/// The orders above \c kFarRecurOrders are this walk's and the ones up to it the
/// recurrence's (\c BoysFarRecurLowOrders), whose own doc states the crossing - in every
/// request, not only the short ones. A request that reaches past the cut is answered by
/// both bodies: the recurrence's orders first, then the walk from order kFarRecurOrders + 1
/// with its registers seeded from that state, so the request's low orders are the per-order
/// lane's own values bit for bit and its high orders the values a walk from order zero
/// writes, bit for bit.
///
/// The order's power of h is carried in fours: \c w holds \c F_0 h^q for q = 0..3 and
/// \c p the running \c h^(4j), so a term is two multiplies and one table read.
///
/// No branch guards the powers: \c h^32 leaves the normal range above x = 2.06e9 and
/// flushes to zero above x = 6.48e9, and the largest order's value there is below
/// 7.7e-285 - under every figure this region states by 270 orders of magnitude.
///
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      argument, kX1 <= x
/// \param stride distance between two orders in \c out
/// \param out    receives nmax + 1 values, out[l * stride] = F_l(x)
inline void FarLadderRun(int nmax, double x, std::size_t stride, double* out) noexcept {
    // The lowest orders are the recurrence's, \c kFarRecurOrders of them, in every request: one
    // that stops at the cut is answered by the recurrence alone and a longer one by the
    // recurrence first and the walk above the cut, so both call shapes stay each other's values
    // at every order.
    if (nmax <= kFarRecurOrders)
    {
        BoysFarRecurLowOrders(nmax, x, stride, out);
        return;
    }

#if BOYS_ISA_AVX2_FMA
    // The contiguous call is the four-orders lane's: one vector holds four orders' terms, the
    // same two multiplies an order, and every per-argument entry calls with stride 1. The
    // strided caller keeps the walk below, whose values are the lane's own bit for bit.
    if (stride == 1)
    {
        BoysFarLadderOrdersLane(nmax, x, stride, out);
        return;
    }
#endif

    // The recurrence answers the orders up to the cut here too, this body's own call of it at
    // this request's own r, so its low orders are the per-order lane's own values bit for bit.
    const double r = 1.0 / std::sqrt(x);
    BoysFarRecurLowOrdersFromR(kFarRecurOrders, r, stride, out);

    // The walk from order kFarRecurOrders + 1 on, its registers seeded from the recurrence's
    // state: that order is lane 1 of the group kFarRecurOrders..+3, whose running power is h^4,
    // so above the cut every order is the value a walk from order zero writes - and the walk's
    // division is its own: sqrt(2 * h) and f / kBoysHalfSqrtPi are other roundings of r.
    const double h = 0.5 * (r * r);
    const double h2 = h * h;
    const double h4 = h2 * h2;
    const double w[4] = {kBoysHalfSqrtPi * r, kBoysHalfSqrtPi * (r * h), kBoysHalfSqrtPi * (r * h2),
                         kBoysHalfSqrtPi * (r * (h2 * h))};

    double p = h4;
    int q = (kFarRecurOrders + 1) & 3;

    for (int l = kFarRecurOrders + 1; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = (w[q] * kDoubleFactorial[l]) * p;

        if (++q == 4)
        {
            q = 0;
            p *= h4;
        }
    }
}

/// The far region's ladder, in the closed form the arm named
/// \c FarRepresentation::kAsymptoticRecurrence answers with.
///
/// It is the same arithmetic as \c FarLadderRun and not a second one, read at the
/// orders the caller names: the seed is the closed form with \c H_0 held at 1, whose
/// relative error is \c erfc(sqrt(x)) - 2.65e-14 at \c kX1 - and the series' tail is
/// the ladder's, which is what the absolute figure above covers and the relative
/// figure does not.
///
/// The division form is the caller's and does not reach this arithmetic: the ladder's
/// one division is the seed's closed form, so the axis is inert here.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, kX1 <= x
/// \param out  receives nmax + 1 values, out[k] = F_k(x)
template <DivisionForm kForm>
void AsymptoticRecurrenceAllOrders(int nmax, double x, double* out) noexcept {
    static_cast<void>(kForm);
    FarLadderRun(nmax, x, 1, out);
}






// The effective-degree tables a policy reads, over the partition the policy names: the table is
// the partition's own, so a policy reads its own partition's degrees rather than the shipped one's.
// Both bodies ask the shared question first and
// then name their own partitions, so an unnamed one fails the read, not takes the last arm's table.
template <EvalPolicyLike Policy, BoysRole kRole>
constexpr auto RegionADegreeTableOf() noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this table is what the Chebyshev family's region-A fit is read at, and "
                  "a policy naming the uniform grid has none: the grid's cells are "
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
                  "this table is what the Chebyshev family's region-B fit is read at, and "
                  "a policy naming the uniform grid has none: the grid reads its own "
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
        // The same fit as the shipped arm, by construction and not by fallback: region B's
        // seed is one fit over [kX0, kFlatHi) at every granularity except the narrow one, so the
        // table a cut would use is the shipped table's. The assertion above keeps a
        // policy naming the grid from reaching a cut at all.
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







// The two families under one name: what a float engine that takes a route and a
// scheme reads, the way the double bodies read Policy::Fit.
template <FitRoute kRoute, EvalScheme kScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
using FloatRouteFit = std::conditional_t<kRoute == FitRoute::kChebyshev,
                                         ChebyshevFit32<kScheme, kGranularity>,
                                         RationalFit32<kGranularity>>;

// Region-A seed of a float-lane batch, in the double precision the downward recursion needs.
// The route's own fit answers where that route's selector takes over, the shipped family below
// it - the same pair of choices, gated by the same constant, that the double lane's batch makes.
// It is the double lane's fit because a 1.5e-7 seed is a 5e-3 result at nmax = 8.
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
            // Region A of the float batch's seed is read through the double lane's piece-indexed
            // families, and the grid has no piece-indexed fit: its cells are read interval-major by
            // the grid's own body, which answers this shape before this body is reached. The arm
            // refuses rather than reading the narrow pieces under the grid's name.
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

/// The float lane's single-order body over a fit policy: one body, so the route names the two
/// fits and changes nothing else. Region C reads no coefficient at all and is the same three
/// lines under either route.
///
/// The division form is a parameter here rather than a field the fit carries, because the fit is
/// what this body's callers select and the form is what the policy selects beside it. The
/// region-B exponential is the same kind of parameter and has no default: a caller that forgot it
/// would run one member's seed under a policy naming the other.
template <typename Fit, DivisionForm kForm = kDefaultDivisionForm, RegionBExp kExp>
float SingleOrderF32Body(int n, float x) noexcept {
    if (x == 0.0f)
    {
        return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
    }

    const float x0 = static_cast<float>(kX0);
    const float x1 = static_cast<float>(kX1);

    // The region-A read goes through the fit's own stored pieces, which the uniform grid does not
    // have: its cells are interval-major and are read by UniformSingleOrderF32. The read is dropped
    // for a partition the fit does not answer, not answered from the narrow pieces under the grid's
    // name - dead where dropped, the entry routing a uniform policy here having answered below the join.
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

        // The recurrence step is written as the backend's two-rounding multiply-subtract rather
        // than a bare product and difference: a bare one contracts where the compiler contracts and
        // not where it does not, making this value a property of the calling unit's flags - and of
        // the optimizer's choice between two inlined copies - while the spelled form always rounds.
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

    // The far step carries the exp(-x)/2 term the bare asymptotic step drops.
    const float expx = RegionBHalfExpF32<RegionBExp::kAccurate>(x);

    for (int l = 0; l < n; ++l)
    {
        f = DivideStep<kForm>(backend::ScalarFp32::MulSub(
                                  static_cast<float>(l) + 0.5f, f, expx),
                              x, invx);
    }

    return f;
}

// A body takes the policy as its ONE selection parameter and reads the axes as fields, so anything
// added later reaches the recurrences without a parameter per axis; the fit is overridden by a
// route whose fits are its own, and everything else - the zero argument, the region split, the
// per-order rule, the domains - is the same under either.

// The assertion below is a contract on the reads beneath it: a grid whose join fell inside the
// fitted domain would leave the region-A and region-B blocks live for a grid-naming policy and
// answer it from another partition's tables under the grid's name.
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

    // The uniform route answers the whole of its table's domain from the table and hands
    // everything above it to the asymptotic below, which reaches no fit: the join is above kX1,
    // so an argument past it satisfies neither of the two region tests that follow and falls
    // through to the asymptotic by the tests themselves.
    if constexpr (Policy::kGranularity == FitGranularity::kUniform)
    {
        // kFlatHi belongs to the grid: the locator clamps the last interval's own
        // index and maps the argument to that interval's right end.
        if (x <= detail::kFlatHi)
        {
            UniformAllOrders<Policy>(nmax, x, out);

            return;
        }
    }

    if constexpr (FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The block below resolves the policy's partition through the Chebyshev family, which does not
        // answer the grid, so it is dropped for such a partition: its reads are then not instantiated
        // and the assertion they carry cannot refuse the grid's own served bodies. Nothing a served
        // call runs changes, and the barrier in PolicyRegionAValue stays the one PolicyRegionBSeed's is.
        if (x < kX0)
        {
            // The pure per-(n, x) dispatch, driven by the n-indexed threshold table: the orders k with
            // x >= kTierThresholds[k] are a prefix (the thresholds are non-decreasing in n) and are read from
            // this route's own band answer; the tail orders keep the per-order piece value at the partition
            // the policy names. The fallback below seeds the downward recursion once, at nmax.

            // out[k] for k < nmax carries the recurrence's value rather than the fit's: swept over the
            // accuracy gate's committed grid at every nmax, the worst is 3.33e-16 absolute (nmax = 1, k = 0,
            // x = 0.91067553232796428), 3 ULP of that result, and the worst in ULP is 140 (nmax = 29, k = 27,
            // x = 1.0418128089831911), where the result is 6.66e-3.
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
            // is the batch role's, as it is in the Chebyshev body's own batch path.
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

    if (x < kFlatHi)
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

    FarLadderRun(nmax, x, 1, out);
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
        // kFlatHi belongs to the grid, for the reason AllOrdersBody gives.
        if (x <= detail::kFlatHi)
        {
            return FlatSingleOrder<Policy>(n, x);
        }
    }

    if constexpr (FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The block below resolves the policy's partition through the Chebyshev family, which does not
        // answer the grid, so it is dropped for such a partition: its reads are then not instantiated
        // and the assertion they carry cannot refuse the grid's own served bodies. Nothing a served
        // call runs changes, and the barrier in PolicyRegionAValue stays the one PolicyRegionBSeed's is.
        if (x < kX0)
        {
            if (x >= detail::kTierThresholds[static_cast<std::size_t>(n)])
            {
                typename Fit::template BandSource<Policy::kDivision> source(x);
                double f = 0.0;

                // The order is at most kMaxBoysOrder, and bounding the walk by it as well as by n
                // is what lets a compiler see the induction terminate: on an order outside the
                // contract the walk is defined rather than an overflow waiting to happen, and for
                // every order inside it the two bounds agree.
                for (int l = 0; l <= n && l <= kMaxBoysOrder; ++l)
                {
                    f = source.Next(l, x);
                }

                return f;
            }

            return PolicyRegionAValue<Policy>(n, x);
        }
    }

    if (x < kFlatHi)
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

    // One order of the far ladder, read where the ladder writes it: the buffer is the run's own
    // shape at the widest order and the term asked for is one of its entries, so the order is
    // answered by the same arithmetic every other order of this argument is. The write below is
    // the reader's guarantee: the run defines every entry it is asked for.
    double ladder[kMaxBoysOrder + 1];
    ladder[n] = 0.0;
    FarLadderRun(n, x, 1, ladder);

    return ladder[n];
}

// Each engine takes the policy, and nothing else: the fit family, the scheme and the
// single-precision budget arrive as the policy's fields.

// The uniform partition is carried by the all-orders and single-order bodies, which read its table
// directly: every other body recurses over the orders - a seed stepped upward or a piece stepped
// downward - and the uniform table offers neither.

// A body without a uniform branch handed a uniform policy does not fail to compile on its own: it
// asks the policy's contract members and the narrow fits answer them, returning certified numbers
// from a partition the caller never named. This refuses the combination where it is named; the
// condition is the shared question, the reads below resolving the partition through the family.
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
// The rational member over the uniform grid is a fit of this lane's own and it is stored (see
// RationalFitUniformF32 above), so FloatRouteFit's fall-through does not reach a policy naming
// it: UniformOrderAtF32 dispatches on the route and reads the pairs this lane's grid carries.
// RationalFit32 would otherwise resolve the grid to the shipped member's cover, another partition's.

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

// A stored row carries its degrees per order, and each partition carries its own table: the
// shipped row's degrees are cut from the shipped pieces' coefficients, the narrow row's from the
// narrow pieces', both derived by one criterion over the two tables of coefficients
// (boys_effective_degrees.hpp).

template <EvalPolicyLike Policy>
double BoysSingleImpl(int n, double x) noexcept {
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order, so it has one order to put in a vector lane "
                  "and the orders axis is not an axis here: the axis this library carries on "
                  "this shape is the arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0);

    // The body is selected at compile time: no branch, indirection or runtime dispatch sits on
    // the path. The uniform partition is selected the same way: its cells are stored at the
    // grid's own degrees, so it is the reference reading, both members admissible.
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

// The single-precision lane's entry on the same axis (defined by boys/boys_orders_lanes.hpp),
// declared here for the same reason and one more: this lane's degree table is certified against
// a region budget as well as against a table, so the computation budget is a choice it carries
// and the double lane's entry does not.
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
        // The across-orders packed lane carries every combination the axis offers: either route's
        // region-A fits, either partition of the shipped route's, any granularity, any division
        // form, the four choices reaching the lane as template arguments. The form is one of them
        // because the orders it does not pack go to the scalar single lane, whose steps divide.
        if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            // Above kFlatHi the uniform partition has one answer whatever the scheme, route and
            // form: the packed arm's own predicate sends x > kFlatHi through the certified scalar
            // orders lane into this same ladder, so the ladder is read here and the packed body's
            // prologue and its classification are not paid on the far body's low orders.
            if (x > kFlatHi)
            {
                FarLadderRun(nmax, x, 1, out);
                return;
            }
        }
        BoysAllOrdersPacked<Policy::kScheme,
                            Policy::kRoute,
                            Policy::kGranularity,
                            Policy::kDivision>(nmax, x, out);
    } else {
        // The uniform partition is served here too: its ladder is the reference reading,
        // and the body below carries the partition's own branch. On the orders axis the branch above
        // reaches this same partition's ladder, one argument at a time.
        AllOrdersBody<Policy>(nmax, x, out);
    }
}

/// The fixed-n vector engine: F_n at every argument of an array, one fixed order (the batch
/// shape of angular-momentum-grouped inner loops; the strided output layout belongs to the
/// public surface in boys.hpp). The region bodies below mirror BoysSingleImpl's, the certified
/// scalar single-lane code at BoysRole::kDoubleSingle, so every output element
/// returns the corresponding BoysSingle call's value, and the same bits on a build whose bare
/// product-plus-add is two roundings. On a build that contracts that form the compiler decides
/// per call site whether to fuse it: a value can move by a unit in the last place and no
/// further. Keep the two engines' bodies in lockstep - identical source is what holds them
/// inside one bound. The region dispatch is per element, so mixed-region arguments need no
/// pre-partitioning; the dispatch-once-per-batch structure lives in the AVX2 region-sorted
/// lanes (boys_simd.cpp).
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
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);
    assert(stride >= 1);

    // The partition clause is the shared question rather than a statement that this entry refuses
    // the grid: the shaped body below reads PolicyRegionAValue and PolicyRegionBSeed, which resolve
    // the policy's partition through the Chebyshev family, so what sends a policy past this branch
    // is that family not answering the partition it named.
    if constexpr (Policy::kRoute != FitRoute::kChebyshev ||
                  !FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity))
    {
        // The fixed-order entry carries the route too, and by the entry that takes its fit from the policy:
        // one order at every argument of an array is the per-argument single entry called once per
        // argument, so naming a route makes the identity the documentation claims exact by construction.

        // The uniform partition takes this path for the same reason: the two Chebyshev branches below
        // reach PolicyRegionAValue and PolicyRegionBSeed, whose else branch is the NARROW member, so a
        // uniform policy sent down either would be answered from another partition's fits under the
        // grid's name.
        for (std::size_t i = 0; i < count; ++i)
        {
            assert(x[i] >= 0.0);
            out[i * stride] = BoysSingleImpl<Policy>(n, x[i]);
        }

        return;
    }
    // The two selections are alternatives and not two independent tests: the branch above returns
    // whenever it is taken, and saying so is the difference between a compiler reading the block
    // below as discarded and reading it as unreachable.
    else {
        // The guard belongs in the two Chebyshev branches and not at the top of the function: this one
        // reaches PolicyRegionAValue and PolicyRegionBSeed, whose NARROW else branch would answer a
        // uniform policy under the uniform name. The branch above needs no guard - it hands each argument
        // to BoysSingleImpl, which reads the grid.

        // The assertion stands as this branch's own contract rather than as a cell this entry refuses:
        // this body cannot answer the grid, the two reads being the recursion's and the grid having no
        // next order to build from this one.
        RefuseUniformPartition<Policy>();

        // The far ladder's run at the widest order, written once and read per argument: this entry
        // asks one order of every argument, and the entry is not the ladder's - the run is.
        double ladder[kMaxBoysOrder + 1];

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

            if (xi < kFlatHi)
            {
                double f = PolicyRegionBSeed<Policy>(xi);
                const double expx = RegionBHalfExp<Policy::kRegionBExp>(xi);
                const double invxi = StepReciprocal<Policy::kDivision>(xi);

                for (int l = 0; l < n; ++l)
                {
                    f = DivideStep<Policy::kDivision>((l + 0.5) * f - expx, xi, invxi);
                }

                out[i * stride] = f;
                continue;
            }

            FarLadderRun(n, xi, 1, ladder);
            out[i * stride] = ladder[n];
        }
    }
}

// The float lanes' scope: the float dispatch is untouched, still keyed to kX0/kX1, so the carved
// band [kExtendedBX0, kX0) stays exactly the float path; the certified table is the double
// recursion's and the float band is measured. The policy reaches this engine as the pair of fits
// its two seeds are read from and the table they are summed out of, one body per route.

// The two route names pick the two fits this lane carries, read at the degrees the stored rows
// carry.
template <EvalPolicyLike Policy>
float BoysSingleF32Impl(int n, float x) noexcept {
    // The orders axis is refused here on the same reading as the double entry's refusal: this shape
    // evaluates one order at one argument, so there is no second order to fill a lane with and no
    // argument array to widen over. BoysSingleF32 documents the refusal; without the assertion the
    // policy compiled through and the arguments axis answered in its place.
    static_assert(Policy::kPack == PackAxis::kArguments,
                  "this entry evaluates one order, so it has one order to put in a vector lane "
                  "and the orders axis is not an axis here: the axis this library carries on "
                  "this shape is the arguments axis");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0f);

    // The uniform partition's own domain, taken before the route and the degree table below rather
    // than through them: the grid is a table no other body here reads, so this branch keeps a policy
    // naming it from being answered by the narrow fits the engine below reaches; above the join the
    // region tests inside that engine answer.

    // The grid's cells are stored at the degrees it was fitted at, so the entry reads them
    // uncut.
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

    // The uniform partition's whole domain, both packing axes: the grid's ladder gives the per-order
    // value at every order, which is what the across-orders packed lane carries into its lanes, so one
    // reading answers either axis - the axis changing which lane runs, not which fit is read.
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
        // The across-orders packed lane: eight orders of one argument in one vector register, the shape
        // BoysAllOrdersF32(nmax, x, out) has and the axis the caller named. It carries the same tables and
        // arithmetic as this engine, so the axis changes which lane runs, not which fit is read. It
        // serves all four (scheme, route) pairs on either partition.
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

        // Region A of the batch, through the double lane's piece-indexed families, which the uniform
        // grid does not have: the entry above answers every argument below the grid's join and returns,
        // so this block is dropped for a policy naming the grid and the read inside it is not
        // instantiated.
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

        const float invx = StepReciprocal<Policy::kDivision>(x);

        if (x < x1)
        {
            const float expx = RegionBHalfExpF32<Policy::kRegionBExp>(x);

            // The region's own seed, and the only arm that reads it: the far arm's seed below is
            // the closed form, so a seed formed above the join would be formed for a value the far
            // arm overwrites - and paid for at every far argument.
            float f =
                FloatRouteFit<Policy::kRoute, Policy::kScheme, Policy::kGranularity>::RegionBSeed(x, 0);
            out[0] = f;

            for (int l = 1; l <= nmax; ++l)
            {
                // The step is the backend's two-rounding multiply-subtract, the single-order
                // body's own spelling. A bare product and difference is one arithmetic where the
                // calling unit contracts and another where it does not, so this body's value at
                // an (n, x) would move with the caller's flags, and part from the single body's.
                f = DivideStep<Policy::kDivision>(
                    backend::ScalarFp32::MulSub(static_cast<float>(l) - 0.5f, f, expx), x, invx);
                out[l] = f;
            }

            return;
        }

        float f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        // The far step carries the exp(-x)/2 term the bare asymptotic step drops, the arm where
        // that term matters most: the same spelled multiply-subtract as the region's arm above,
        // for the same reason.
        const float expx = RegionBHalfExpF32<RegionBExp::kAccurate>(x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = DivideStep<Policy::kDivision>(
                backend::ScalarFp32::MulSub(static_cast<float>(l) - 0.5f, f, expx), x, invx);
            out[l] = f;
        }
    }
}

/// The float lane's all-N batch: the per-argument all-orders body at every argument, the results
/// scattered into the caller's planes. The packed float lane packs the orders of one argument, and this
/// entry's run is one argument after another, so neither axis gets a homogeneous run out of it: the run
/// the axis would pack is the orders of a single argument, which the all-orders body below already
/// fills, one call per argument. What it carries is the shape: the one BoysAllN has in the double lane
/// and BoysCuda::AllNF32 has on the device.
template <EvalPolicyLike Policy>
void BoysAllNF32Impl(int nmax, const float* x, float* out, std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    // Both axes are served, and the axis is read: the per-argument body this loop calls is the
    // all-orders entry, which packs eight orders of one argument when the policy names the orders axis
    // and fits one argument's region otherwise; the arguments axis's region partitioning stays at the
    // caller's loop. The uniform partition reaches the grid through that same call and its own branch.
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

/// The float lane's fixed-order batch: the per-argument single body at every argument, one fixed order,
/// the strided output the public surface documents. It is the loop the double lane's fixed-order entry
/// falls back to when its policy names a route or the grid, and it is a body of its own here because the
/// float lane has no shaped region kernel: the packed lane this entry would need is an AVX2 kernel over
/// doubles, which is also why no speed is claimed for it over the caller's own loop.
template <EvalPolicyLike Policy>
void BoysFixedNF32Impl(
    int n, const float* x, float* out, std::size_t count, std::size_t stride) noexcept {
    // The packing axis is not asserted here: the per-argument body this loop calls, BoysSingleF32Impl,
    // refuses the orders axis itself - one order at one argument has no second order to fill a lane
    // with - and a guard here would state the same refusal twice.
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);
    assert(stride >= 1);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0f);
        out[i * stride] = BoysSingleF32Impl<Policy>(n, x[i]);
    }
}

// The float lane's per-element-top-order batch: the per-argument all-orders body at each argument's own
// top order, written into the caller's planes as the double lane's entry of this shape writes them, and
// with the same property - the cells above each column's own top order are left as the caller left
// them.
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

// One region of the argument line per kernel, same order across an array of arguments, AVX2 with a
// scalar tail. They are the vector tier the entries dispatch to, not the public surface: a caller
// reaching them directly partitions its arguments by region itself, which is the batch entries'
// work.

// The per-region bars the fits are cut against, absolute and none of them the lane's published
// bound (the larger of 1e-13 of F_n(x) and 1e-13 of F_0(x)): A at most 1e-15 over [0, kX0), B's and C's at most 5.5e-14. The
// committed reference grid lands 1.4e-15 on the band at n = 32, C 5.0e-14 inside its bar with 10%
// slack, and B 2.2e-13 at n = 32, x = kX0 - four times its own bar.

// Region B is therefore served by the scalar body in the batch entry. The region-A lane is the one
// the entries reach and is defined by boys_arguments_lanes.hpp at this unit's own tier; these two
// are the measurement-only lanes and stay out of line in boys_simd.cpp, at the tier it is compiled
// at - its scalar twin where the vector flags are absent.
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


/// The companion of the region kernels above: the same region-A stored fits, evaluated four ORDERS to a
/// vector instead of four arguments, which is the axis BoysAllOrders(nmax, x, out) has. It is the orders
/// axis of PackAxis, and the entries that carry it dispatch here.
///
/// It evaluates each order's own fit and reaches no order by a recursion, so its values are the per-order
/// fits' values: at the certified split Clenshaw scheme they are the across-arguments lane's values bit
/// for bit, and the suite asserts that identity. The shipped cover's coefficients are fetched composed -
/// four loads and the shuffles that join them - rather than with one gather instruction, which is the same
/// lane by construction and cheaper in retired slots on the machines measured; the gathered fetch is kept
/// beside it so the pair stays measurable. The uniform grid stores its coefficients with the order
/// innermost, so its lane reads one coefficient index at a time and has no fetch to choose.
///
/// Below kX0 only. Past it the entry runs the certified scalar single lane one order at a time,
/// at the policy the caller named.
///
/// The four choices a policy makes reach the lane as template arguments: the scheme picks which
/// polynomial table and which summation the shipped route's fits are read with, the route which
/// region-A fits the lane carries, the partition the table those fits are cut into, and the
/// division form the recurrence steps use.
///
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    the argument
/// \param out  receives \c nmax + 1 values, one per order
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitRoute kRoute = kDefaultFitRoute,
          FitGranularity kGranularity = kDefaultFitGranularity,
          DivisionForm kForm = kDefaultDivisionForm>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept;

// The shapes this entry is instantiated at are the call sites' own: the lane's bodies are
// defined by boys_orders_lanes.hpp below at the tier this unit's flags name, and by its
// scalar twin where they name none, so a call site instantiates the cell it named and the
// header reaches nothing outside this translation unit.

// The single-precision lane's shapes on the same axis - two schemes and two computation
// budgets, with either route - are instantiated by their call sites on the same terms.

// Every dispatch path is an interval of the argument line and the intervals are ordered, so the
// classification is monotone in x: a non-decreasing array is already contiguous by path, which is
// what the BoysSortedArgs overload declares and why one comparison per argument states it.

// A path is served by one of two kernel shapes. The ungrouped one walks one argument at a time
// with the per-argument entry's own bodies, kept in lockstep with the batch's nmax, so the batch
// values ARE that entry's values, bit for bit. The grouped one runs the region-A lane over a
// homogeneous run, one order at a time.

// Which shape serves region A is decided by measurement: over a 65536-argument run against a plain
// per-argument loop, the lane is 2.6x faster serving F_0, at parity at n = 4 and 10 to 20% slower
// from n = 8 up; kBoysAllNLaneMaxOrder is where the two cross.
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

// The path of one argument, in the per-argument entry's dispatch order. The region-A band split
// arrives as the caller's tierSplit: the scalar dispatch splits the band at the first
// extended-seed tier, and that split is a path of its own.
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

    if (x < kFlatHi)
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

// One argument's column, at `stride` doubles per order: plane[k * stride] = F_k(x) for k = 0..nmax.
// These bodies are the per-argument entry's, restructured onto the caller's layout; keep the two in
// lockstep. The stride is the caller's count when the kernel writes a column straight into the caller's
// planes, the tile's width when it stages one; it is never an operand, so the values ignore it.
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

// Region C: the far ladder's terms, written where the caller's planes are (m-invariant). The
// division form does not reach this arithmetic - the ladder has one division, the seed's closed
// form, and it is not a step of a chain - so the axis is inert here and the parameter is kept
// because the entry's callers name it.
template <DivisionForm kForm = kDefaultDivisionForm>
inline void BoysAllNBodyRegionC(int nmax, double x, std::size_t stride, double* plane) noexcept {
    static_cast<void>(kForm);
    FarLadderRun(nmax, x, stride, plane);
}

/// The tile every kernel of the partitioned entry writes through.
///
/// A run of the argument line is walked in tiles of kBoysAllNChunk arguments. Each tile is filled in a
/// fixed stack frame by `fill` - the kernel's own body - and then written to the caller's planes one
/// order at a time: BoysAllNScatter turns one order's tile into one contiguous run of `block` doubles
/// inside that plane.
///
/// That is the whole reason the tile exists: writing an argument's column straight into the caller's
/// planes puts consecutive stores count * 8 bytes apart - 33.5 MB on a four-million-argument call - so
/// every store lands on a cache line of its own. Inside a tile a store is followed by its neighbour, and
/// the plane stride is paid once per tile per order instead of once per element.
///
/// The frame is the entry's own, (kMaxBoysOrder + 1) * kBoysAllNChunk doubles, the same size whatever
/// `count` is: a batch's stack use does not grow with its argument count.
///
/// `fill(base, block, stage)` writes the tile at stage[l * block + t] for order l and the tile's t-th
/// argument, t < block. The tile's width is `block` and not kBoysAllNChunk, which is why it is passed
/// rather than assumed.
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

/// The far ladder's lane: one ladder per lane of an AVX2 double register, four of the run's
/// ARGUMENTS at once - the arguments axis, since the ladder's terms are consumed in the order.
/// Arithmetic for arithmetic with the scalar body, so its test is equality and not a bound.
/// boys_arguments_lanes.hpp owns both tiers' bodies.
///
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      the four arguments, each at or above kX1
/// \param stride distance between two orders in each column
/// \param cols   the four columns the run writes into
/// \param form   the division form the tier's bodies step in
void BoysFarLadderLanes(int nmax,
                        const double* x,
                        std::size_t stride,
                        double* const* cols,
                        DivisionForm form) noexcept;

// How many arguments the far ladder's lane holds at once: the tier's own double width, and the
// width it is compiled and measured at.
constexpr std::size_t kBoysFarLaneWidth = 4;

/// The far run's lane kernel: one run of far arguments through BoysFarLadderLanes, four
/// arguments at a time, with the per-argument body for what is left over.
///
/// The groups are the run cut at the lane's width, so what the lane does not take is a run shorter
/// than four and the tail of one that is not a multiple of four - and a run shorter than four does
/// not reach this kernel at all, which is the entry's own condition: three arguments or fewer keep
/// the ungrouped kernel, whose body is this kernel's own arithmetic one argument at a time.
///
/// The whole run is the kernel's, marshalling included, and it is out of line for a reason the
/// entry's price shows: the arguments a run does NOT put through the lane must cost what they cost
/// without it. A lane inlined into the entry's own code moves the paths that never take it - the
/// counter reading behind this kernel's first shape, where a one-argument far run came out 11% over
/// the same run without a lane - so what the entry carries is a call.
///
/// Which of the two writers it uses is the ungrouped kernel's choice and its reasons: the sorted
/// overload's run is consecutive in the output, so the tile turns an order's four-argument groups
/// into four consecutive doubles and the lane writes one register store an order; the unsorted
/// entry's run is the caller's positions, so its groups land wherever the permutation put them and
/// staging them costs more than it buys.
///
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      the caller's argument array
/// \param index  the run's argument positions in `x` (the unsorted entry), or nullptr where the
///               run's arguments are the positions themselves (the sorted overload)
/// \param out    the caller's planes, `count` doubles an order
/// \param count  the call's argument count, the planes' stride
/// \param begin  first argument of the run
/// \param end    one past its last; `end - begin` >= kBoysFarLaneWidth
/// \param form   the division form the run's steps run, the policy's own
///
/// Defined by the arguments lane's own header (boys_arguments_lanes.hpp at the vector tier, its
/// scalar twin where the flags are absent), which is what keeps the lane free of the library.
void BoysFarLadderRun(int nmax,
                      const double* x,
                      const std::size_t* index,
                      double* out,
                      std::size_t count,
                      std::size_t begin,
                      std::size_t end,
                      DivisionForm form) noexcept;

// The lanes the entries below reach, at the tier this unit's own flags name: the region-A packed
// lane and the far ladder's, whose vector bodies the header holds and whose no-flag bodies it holds
// beside them. Both are consumer-reachable - the entries call them - so both are defined here
// rather than in a shipped unit, and a unit without the vector flags parses the scalar twin alone.

// The namespaces close around the includes and reopen after them, which is not a style: a
// standard header read inside a namespace is a different standard library.
} // namespace detail

} // namespace boys

#include "boys/boys_arguments_lanes.hpp"

// The orders-axis lanes, on the same terms: the across-orders packed lane (four ORDERS to a
// register) and the flat grid's one-order fit it and the single-order entry share.
#include "boys/boys_orders_lanes.hpp"

namespace boys {

namespace detail {

/// The ungrouped kernel: one argument at a time over the run, the path's body per argument.
///
/// Which of the two writers it uses is decided by where its stores land, a property of the caller's array
/// and not of the shape. Both write the same values - the bodies are the per-argument path's own, called
/// with the same arguments at the same nmax either way - so this is a choice about the stores alone, made
/// by measurement:
///
///   the run's arguments are consecutive in the output. That is the sorted overload, whose run's index IS
///     the argument's own position, so the tile turns one order's run into one contiguous run of stores:
///     on the uniform stream at nmax 32 the entry is 19.9% faster through it over three paired rounds,
///     with the machine-speed canary agreeing to 0.1%.
///   the run's arguments are the caller's positions in the caller's own order. That is the unsorted
///     entry, which sorts for the caller and must return the values in the order it was handed them, so a
///     plane's stores land wherever the permutation puts them. Staging them costs more than it buys: the
///     same measurement puts the unsorted entry at +0.2% on the uniform stream and +13.8% on the
///     molecular one. This kernel therefore writes the unsorted entry's columns straight into the
///     caller's planes.
///
/// The two writers are the same function and differ in one argument: what the body is told its stride
/// is.
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

/// The grouped kernel: the region-A lane over a homogeneous run, filling the same tile the scalar bodies
/// fill. Both region-A paths take this shape when they take it at all - the lane covers [0, kX0), so
/// which of the two scalar bodies the sort's path split assigned an argument to does not choose the
/// kernel; the lane serves the band as well, at its band accuracy.
///
/// Region B keeps its scalar body: BoysRegionBSimd does not hold the region's own bar across the
/// region - at n = 32, x = kX0 it lands 2.2e-13 from the reference against the 5.5e-14 the region's
/// fits are cut against - and the scalar body is the per-argument path's own, exact to the bit.
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
    // The packed region-A lane holds the shipped route's Chebyshev coefficients and its split Clenshaw
    // recurrence, so another route or scheme takes the scalar body here. BoysRegionASimd packs four
    // ARGUMENTS of one order, so the grouped kernel is the arguments axis's and only that axis's,
    // stated rather than inherited because which axis a packed lane fills is what this kernel does.

    // The narrow partition is the scalar bodies' here as well: the grouped kernel reads the shipped
    // piece table by index, so a narrow policy trades away the four-wide lane rather than a value.
    if constexpr (Policy::kRoute == FitRoute::kChebyshev &&
                  Policy::kScheme == EvalScheme::kSplitClenshaw &&
                  Policy::kPack == PackAxis::kArguments &&
                  Policy::kGranularity == FitGranularity::kCoarsest)
    {
#if BOYS_ISA_AVX2_FMA
        // Only the vector tier's lane reads the shipped Chebyshev coefficients; the scalar
        // tier's BoysRegionASimd is the scalar body and reads none.
        static_assert(kCoefficientGroupPresent<DependentGroup<Policy, CoarseChebGroup>> ||
                          kAlwaysFalse<Policy>,
                      "this policy's packed region-A lane reads "
                      "boys/boys_coefficients_d_coarse_cheb.hpp: include that header in the "
                      "translation unit that names this policy");
#endif
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

/// Whether the far region's ladder is the body this path answers an argument with.
///
/// The region's terms are read order by order at one argument (a run of F_0..F_nmax, the arms of
/// boys_orders_lanes.hpp's packed body ending at BoysAllNBodyRegionC), so what a register holds of
/// them is four ARGUMENTS - that is the arguments axis's lane, and this path reaches it here or not
/// at all. The lane's own form has an orders axis in it now, the region's terms being independent
/// products rather than a chain (four orders of one argument take the same seed, the same power and
/// four table entries): the packed bodies' orders lane does not have a far arm of that shape today,
/// and this predicate selects the axis the lane that exists is written across.
///
/// Nothing about the arithmetic changes, because the body the lane reproduces
/// (boys/boys_arguments_lanes.hpp, BoysFarLadderLanes) is the body this predicate selects: at the
/// orders axis the packed body hands an argument past its fits to the certified scalar orders lane
/// (boys/boys_orders_lanes.hpp, BoysAllOrdersPacked), and that lane's arm above its fits is
/// BoysAllNBodyRegionC - one argument at a time, this ladder.
///
/// The join and the axis are that lane's own predicate, ScalarOrders' `aboveFits`: the grid ends at
/// kFlatHi and so do the derived partitions. The route, the scheme and the form do not enter it,
/// because the arm above the fits reads no fit. A policy this predicate does not name keeps the
/// per-argument body: the lane is BoysAllNBodyRegionC's operations and nothing else, and a body
/// that reached the ladder another way would be a second arithmetic taken on no comparison.
template <EvalPolicyLike Policy>
constexpr bool FarLadderIsThePerArgumentBody(double x, bool vectorTier) noexcept {
    if constexpr (Policy::kPack != PackAxis::kOrders)
    {
        return false;
    }
    else
    {
        return vectorTier && (Policy::kGranularity == FitGranularity::kUniform
                                  ? x > detail::kFlatHi
                                  : x >= detail::kFlatHi);
    }
}

// The per-argument path, in the caller's planes: the C entry point's batch loop shape and the
// batch entry's total fallback. The far region is the one exception - its ladder is a recurrence,
// so the orders axis has no lane there to trade away, and the arguments go four at a time to the
// arguments lane it does have, by FarLadderIsThePerArgumentBody rather than a second arithmetic.
template <EvalPolicyLike Policy>
void BoysAllNRunPerArgument(int nmax,
                            const double* x,
                            double* out,
                            std::size_t count) noexcept {
    // One register's worth of the far region, held until it fills or the path ends: the arguments
    // and where each one's column is. The arm is the vector tier's, and the tier is this unit's own
    // flags: a unit compiled without them answers the far region one argument at a time, which is
    // the body the lane reproduces.
#if BOYS_ISA_AVX2_FMA
    double farX[kBoysFarLaneWidth];
    double* farCol[kBoysFarLaneWidth];
    std::size_t held = 0;
#endif

    for (std::size_t i = 0; i < count; ++i)
    {
#if BOYS_ISA_AVX2_FMA
        if (FarLadderIsThePerArgumentBody<Policy>(x[i], true))
        {
            farX[held] = x[i];
            farCol[held] = out + i;

            if (++held == kBoysFarLaneWidth)
            {
                BoysFarLadderLanes(nmax, farX, count, farCol, Policy::kDivision);
                held = 0;
            }

            continue;
        }
#endif

        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<Policy>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
    }

#if BOYS_ISA_AVX2_FMA
    // What the register did not fill is this path's own, one argument at a time: three arguments
    // are not a lane, and the same body answers them.
    for (std::size_t t = 0; t < held; ++t)
    {
        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<Policy>(nmax, farX[t], row);

        for (int l = 0; l <= nmax; ++l)
        {
            farCol[t][static_cast<std::size_t>(l) * count] = row[static_cast<std::size_t>(l)];
        }
    }
#endif
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

/// The plane entry. Two shapes serve it, and which one a call takes is the selectors it names, because
/// each says what the call can be evaluated by:
///
///   the region-partitioned shape  an arguments-axis lane keeps four arguments of one order in a
///                       register, so the entry partitions its arguments by the interval their answer
///                       comes from and hands each homogeneous run to the shape that serves it. That
///                       shape evaluates the shipped fits as its own body, so it is what the shipped
///                       route and the arguments axis are served by. It is BoysAllNPartitionedImpl
///                       below.
///   the per-argument shape  every other call. An orders-axis lane keeps four orders of ONE argument in
///                       a register - out[k * count + i] is F_k(x[i]), so an argument's whole order
///                       vector is already what the entry writes - and a route other than the shipped
///                       one is carried by the all-orders entry's body, which takes its fit from the
///                       policy. The uniform partition is here too, and none of the three has anything
///                       for the region grouping to group.
///
/// Every value either shape returns is inside the bound the entry documents. The orders axis
/// reads its fits through the engine itself (BoysAllOrdersImpl), as the shipped axis does.
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
        // The partition clause is the shared question: the partitioned body this branch keeps out reaches
        // its values through PolicyRegionAValue and PolicyRegionBSeed, so a partition the Chebyshev family
        // does not answer would be read out of the narrow tables under the name it was given; the path
        // above serves it instead.

        // The grid has no region walk to group - an order is read from its own interval's coefficients,
        // nothing is built from another order, so there is no seed to share and no cut to make.
        static_cast<void>(workspace);
        BoysAllNRunPerArgument<Policy>(nmax, x, out, count);
    }
    else
    {
        // The guard belongs here and not at the top of the function. This path reads the shipped and
        // narrow tables directly and has no uniform branch, so a uniform policy would be answered from
        // another partition's fits - the substitution this refuses. Neither path above needs a guard:
        // both hand every argument to BoysAllOrdersImpl, which reads the grid where the policy names it.
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
    // The path's own contract, one level below the entry's guard: every read below resolves the
    // policy's partition through the Chebyshev family, which does not answer the grid. The entry asks
    // RefuseUniformPartition before delegating here, so this refuses nothing the entry serves - only a
    // call reaching the partitioned shapes unrouted.
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

    // The tier is this unit's own flags, not the host's processor: the region-A group here is
    // BoysAllNRunGrouped, which calls the packed lane BoysRegionASimd, and the lane the unit's flags
    // select is the one it runs. At the scalar tier the grouped kernel is the extra indirection over
    // the scalar body and the runs are better served one group at a time.
    constexpr bool vectorTier = BOYS_ISA_AVX2_FMA != 0;
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
            // The far region's ladder takes the arguments lane at every policy: one recurrence
            // whose every policy read the lane makes the same way, and no fit and no partition
            // to trade away. The test is at the caller, not in BoysAllNRun, which the non-taking
            // paths share: a branch inside cost a one-argument far run 5.7%, at the caller 1%.
            if (vectorTier && static_cast<BoysPath>(p) == BoysPath::kAsymptotic &&
                to - from >= kBoysFarLaneWidth)
            {
                BoysFarLadderRun(nmax, x, workspace, out, count, from, to, Policy::kDivision);
                continue;
            }

            BoysAllNRun<Policy>(
                nmax, x, workspace, out, count, vectorTier, static_cast<BoysPath>(p), from, to);
        }
    }

    delete[] owned;
}

// The sorted overload, split the same way and for the same reason: the ordering the caller declared is
// what makes the partitioned shape's grouping free, so it is a fact about that shape's input and not
// about this entry. A call naming the orders axis, or a route other than the shipped one, takes the
// per-argument path, which neither reads nor needs the ordering.
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

    // The tier is this unit's own flags, as in BoysAllNPartitionedImpl, whose reason is stated there.
    constexpr bool vectorTier = BOYS_ISA_AVX2_FMA != 0;
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

        // The far ladder's lane, as in BoysAllNPartitionedImpl and for the same reason, including
        // the one its own comment gives for testing here and not inside BoysAllNRun.
        if (vectorTier && path == BoysPath::kAsymptotic && end - begin >= kBoysFarLaneWidth)
        {
            BoysFarLadderRun(nmax, x, nullptr, out, count, begin, end, Policy::kDivision);
            begin = end;
            continue;
        }

        BoysAllNRun<Policy>(nmax, x, nullptr, out, count, vectorTier, path, begin, end);
        begin = end;
    }
}

/// The per-element-top-order batch: the per-argument all-orders body at each argument's own top order,
/// written into the caller's planes. The tops differ per element, so a run has no common nmax to be
/// grouped at and this entry has nothing to group; its whole content is the layout, and the cells above
/// each column's own top order, which it leaves exactly as the caller left them.
///
/// The body is the per-argument entry's, so this entry takes the policies that entry takes - including a
/// named fit route, which the plane entry does not.
///
/// The columns are written straight into the caller's planes and not staged through the tile the plane
/// entry's kernels use, because here the tile costs more than it saves: a column here is short - the
/// molecular stream's tops average about one order - so a tile of them has to be scattered one plane at a
/// time. With the tile this entry was 24.7% slower on the uniform stream and 54.6% slower on the
/// molecular one, the latter over four repeats of an alternating before/after pair with the canary
/// agreeing to 1%.
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
BoysStatus BoysSingleChecked(int n, double x, double* out) noexcept {
    // `!(x >= 0.0)` and not `x < 0.0`: a NaN fails every comparison, and a NaN is exactly the
    // argument an upstream 0/0 hands a caller who has no way to know it is coming.
    if (out == nullptr || n < 0 || n > kMaxBoysOrder || !(x >= 0.0)) {
        return BoysStatus::kInvalidArgument;
    }

    *out = detail::BoysSingleImpl<Policy>(n, x);
    return BoysStatus::kSuccess;
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

// The checked overloads of the five shapes. Each validates the whole call before evaluating
// anything, so a refused call writes nothing: a caller that tests the status cannot be looking at
// a half-written array. The scan is O(count) against an evaluation that is O(count * nmax), so it
// is not what the call costs.

/// Whether an argument the entries require to be non-negative and not NaN is one.
inline bool AcceptableArgument(double x) noexcept {
    // `!(x >= 0.0)` and not `x < 0.0`: a NaN fails every comparison, and a NaN from an upstream
    // 0/0 is exactly the argument a caller has no way to see coming.
    return x >= 0.0;
}

/// The order a batch entry was given, and the one it accepts.
inline bool AcceptableOrder(int n) noexcept {
    return n >= 0 && n <= kMaxBoysOrder;
}

/// Whether a batch's two buffers are usable by a call of `count` elements.
///
/// A batch of no elements reads and writes nothing, so a null buffer is in contract there - which
/// is the layout the unchecked entries state. What a checked entry refuses is a broken contract,
/// not a null pointer, and the two must agree: a caller that swaps one for the other may not get a
/// different answer to the same in-contract call.
inline bool AcceptableBuffers(const void* x, const void* out, std::size_t count) noexcept {
    return count == 0 || (x != nullptr && out != nullptr);
}

/// The same, for the shape that also carries an array of orders.
inline bool AcceptableBuffers(const void* n, const void* x, const void* out,
                              std::size_t count) noexcept {
    return count == 0 || (n != nullptr && x != nullptr && out != nullptr);
}

template <EvalPolicyLike Policy>
BoysStatus BoysAllOrdersChecked(int nmax, double x, double* out) noexcept {
    if (out == nullptr || !AcceptableOrder(nmax) || !AcceptableArgument(x)) {
        return BoysStatus::kInvalidArgument;
    }

    detail::BoysAllOrdersImpl<Policy>(nmax, x, out);
    return BoysStatus::kSuccess;
}

template <EvalPolicyLike Policy>
BoysStatus BoysFixedNChecked(int n, const double* x, double* out, std::size_t count,
                             std::size_t stride, std::size_t* badIndex) noexcept {
    // `stride == 0` is refused because the layout beside this entry requires stride >= 1: every
    // element would be written to out[0] and the call would report success.
    if (!AcceptableOrder(n) || stride == 0 || !AcceptableBuffers(x, out, count)) {
        return BoysStatus::kInvalidArgument;
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (!AcceptableArgument(x[i])) {
            if (badIndex != nullptr) {
                *badIndex = i;
            }

            return BoysStatus::kInvalidArgument;
        }
    }

    detail::BoysFixedNImpl<Policy>(n, x, out, count, stride);
    return BoysStatus::kSuccess;
}

template <EvalPolicyLike Policy>
BoysStatus BoysAllNChecked(int nmax, const double* x, double* out, std::size_t count,
                           std::size_t* workspace, std::size_t* badIndex) noexcept {
    if (!AcceptableOrder(nmax) || !AcceptableBuffers(x, out, count)) {
        return BoysStatus::kInvalidArgument;
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (!AcceptableArgument(x[i])) {
            if (badIndex != nullptr) {
                *badIndex = i;
            }

            return BoysStatus::kInvalidArgument;
        }
    }

    detail::BoysAllNImpl<Policy>(nmax, x, out, count, workspace);
    return BoysStatus::kSuccess;
}

template <EvalPolicyLike Policy>
BoysStatus BoysAllNAtOrdersChecked(const int* n, const double* x, double* out, std::size_t count,
                                   std::size_t* badIndex) noexcept {
    if (!AcceptableBuffers(n, x, out, count)) {
        return BoysStatus::kInvalidArgument;
    }

    for (std::size_t i = 0; i < count; ++i) {
        if (!AcceptableOrder(n[i]) || !AcceptableArgument(x[i])) {
            if (badIndex != nullptr) {
                *badIndex = i;
            }

            return BoysStatus::kInvalidArgument;
        }
    }

    detail::BoysAllNAtOrdersImpl<Policy>(n, x, out, count);
    return BoysStatus::kSuccess;
}

// The far-region entry: defined here rather than in a shipped unit, so a consumer that names a
// representation compiles that arm and not a dispatch over the others; it forwards to the three
// far bodies above. Its contract is the declaration's on boys/boys.hpp.
template <FarRepresentation kFar>
void BoysAllOrdersWithFarRepresentation(int nmax, double x, double* out) noexcept {
    // Every member of the axis is served below, so a caller naming one gets what the name
    // says: what the fallback is for is a name this build does not serve, which is the
    // default entry outright, the fallback BoysAllOrdersWithRoute takes for a route it
    // does not serve. The default is named where it is declared, not spelled here.
    constexpr bool kServed = kFar == FarRepresentation::kAsymptoticRecurrence ||
                             kFar == FarRepresentation::kScaledTable ||
                             kFar == FarRepresentation::kCorrectedSeed;
    constexpr FarRepresentation kSelected = kServed ? kFar : kDefaultFarRepresentation;

    // The bare ladder is the region's own arithmetic, and its span opens where the fitted
    // grid ends: below kFlatHi the default entry answers, whose fitted regions serve as
    // they do under every name on this axis.
    if constexpr (kSelected == FarRepresentation::kAsymptoticRecurrence)
    {
        if (x >= detail::kFlatHi)
        {
            detail::AsymptoticRecurrenceAllOrders<kDefaultDivisionForm>(nmax, x, out);
            return;
        }
    }

    // The table serves the span its rows report and nothing else: outside it the
    // default entry answers, at its own values and by its own code, so the join at
    // kScaledFarX0 and the edge at kScaledFarX1 are not a second implementation of
    // the arguments beyond them.
    if constexpr (kSelected == FarRepresentation::kScaledTable)
    {
        if (x >= detail::kScaledFarX0 && x < detail::kScaledFarX1)
        {
            // The division form is the default entry's: this entry names the far
            // representation and no other axis, so the recovery's ladder divides the
            // way an unnamed division form divides rather than substituting a form the
            // caller did not name.
            detail::ScaledFarAllOrders<kDefaultDivisionForm>(nmax, x, out);
            return;
        }
    }

    // The corrected seed serves region C, which is the whole of [kX1, inf): below
    // it the fitted regions answer. Its span is the region itself rather than a
    // table's, because the seed correction and the stepped exp term are properties
    // of the region and not of a table's rows.
    if constexpr (kSelected == FarRepresentation::kCorrectedSeed)
    {
        if (x >= detail::kX1)
        {
            detail::CorrectedSeedAllOrders<kDefaultDivisionForm>(nmax, x, out);
            return;
        }
    }

    boys::BoysAllOrders<DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>(nmax, x, out);
}

// The route entries: defined here rather than in a shipped unit, so a consumer that names a
// route compiles that route's arm and not a dispatch over the others; each forwards to the
// public entry at the policy its route builds. Their contract is the declarations' on
// boys/boys.hpp.
template <FitRoute kRoute>
void BoysAllOrdersWithRoute(int nmax, double x, double* out) noexcept {
    // The class's own scheme, not the reference one: the values a route does not serve are the
    // default entry's bit for bit, which requires the scheme that entry reads, so the scheme is
    // read off the row rather than off the seam's five.
    using SelectorClass = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>;

    BoysAllOrdersWithRoute<kRoute>(SelectorClass::kScheme, nmax, x, out);
}

template <FitRoute kRoute>
void BoysAllOrdersWithRoute(EvalScheme scheme, int nmax, double x, double* out) noexcept {
    // One body, instantiated once per (route, scheme) pair through the same entry a compile-time
    // caller uses: the policy names only the fits and the summation they are read in. Every
    // argument the named route's rows do not cover is answered by the body's own shipped-family
    // branches, and every route this build does not serve is the default entry outright.
    using SelectorClass = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>;

    constexpr bool kServed = kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax;
    constexpr FitRoute kSelected = kServed ? kRoute : SelectorClass::kRoute;

    // A scheme outside the enumeration is the split Clenshaw's body, which is the declared
    // fallback: the Horner test is an equality and the fallthrough is that body.
    using HornerPolicy = SeamPolicy<kSelected,
                                    EvalScheme::kHorner,
                                    SelectorClass::kBudget,
                                    SelectorClass::kPack,
                                    SelectorClass::kGranularity,
                                    SelectorClass::kDivision,
                                    SelectorClass::kRegionBExp>;
    using SplitClenshawPolicy = SeamPolicy<kSelected,
                                           EvalScheme::kSplitClenshaw,
                                           SelectorClass::kBudget,
                                           SelectorClass::kPack,
                                           SelectorClass::kGranularity,
                                           SelectorClass::kDivision,
                                           SelectorClass::kRegionBExp>;

    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrders<HornerPolicy>(nmax, x, out);
        return;
    }

    BoysAllOrders<SplitClenshawPolicy>(nmax, x, out);
}

// The float lane's route entry, the same template at the single-order class.
template <FitRoute kRoute>
float BoysSingleF32WithRoute(int n, float x) noexcept {
    // One body instantiated per route, as the double lane's entries above are: the closed form at
    // zero, the region split, the two recurrences and the domains are the body's, and the route
    // names only its two fits. Every cell but the route is read off the row this entry's default
    // arm runs: an arm naming its own fit family would answer from another partition's tables.
    using SingleOrderClass = DefaultPolicy<Precision::kFp32, Shape::kSingle>;

    // Every route this build does not serve is the default entry outright; the fallback is
    // read from the class row, never pinned to kChebyshev: an emitted seam may default this
    // class to the other route (measured 2026-10-12, the probe's vote names kRationalMinimax
    // for the float single class), and a pinned arm answers the wrong tables.
    constexpr bool kServed = kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax;
    constexpr FitRoute kSelected = kServed ? kRoute : SingleOrderClass::kRoute;

    using RoutePolicy = SeamPolicy<kSelected,
                                   SingleOrderClass::kScheme,
                                   SingleOrderClass::kBudget,
                                   SingleOrderClass::kPack,
                                   SingleOrderClass::kGranularity,
                                   SingleOrderClass::kDivision,
                                   SingleOrderClass::kRegionBExp>;

    return BoysSingleF32<RoutePolicy>(n, x);
}

// The combination named in the type and the policy resolved at the call: the
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
// The fp16/bf16 lanes forward the policy to the F32 engine with the fp16
// budget: the engine branch is the certified F32 path verbatim, so the lanes are
// bit-unchanged. The policy is the caller's, defaulted to the lane's own, because every
// combination this lane's book carries is a policy a consumer can name.

// The budget is not one of the policy's axes - it is what makes this lane the half lane - so it is
// DefaultPolicyFp16's, not the caller's.

// The budget a policy named on a half lane has to carry: it is the axis that makes this lane the half
// lane, so a policy built at the float lane's budget names the float lane's combination, and reaching
// this entry with one would answer that combination under this lane's name. It is refused where it is
// named rather than honoured.
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

// The half lane's array shapes: the float lane's own bodies at each argument, the results stored to
// half once, as BoysAllOrdersF16 stores one argument's ladder. Nothing is computed in half beyond the
// store - the arithmetic is the fp32 engine's - and each element is the fp16 lane's single or
// all-orders entry at that argument, so the bound is that entry's.
template <EvalPolicyLike Policy>
void BoysFixedNF16(
    int n, const F16* x, F16* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);
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

// The bf16 lane's array shapes: the same three bodies as the fp16 array entries above, op for op, with
// this format's conversions in place of fp16's. Nothing is computed in half here beyond the store
// either, so the arithmetic is the float lane's, and the one term differing between the two half formats
// is the half digit each store makes: 2^-8 for a bfloat16 return against 2^-11 for a binary16 one.
template <EvalPolicyLike Policy>
void BoysFixedNBf16(
    int n, const Bf16* x, Bf16* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);
    assert(stride >= 1);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= static_cast<Bf16>(0.0f));
        out[i * stride] = static_cast<Bf16>(
            detail::BoysSingleF32Impl<Policy>(n, static_cast<float>(x[i])));
    }
}

template <EvalPolicyLike Policy>
void BoysAllNBf16(int nmax, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    static_assert(Policy::kBudget == BoysBudget::kFp16,
                  "the half lanes run the fp16 engine budget, so a policy named here is one built "
                  "at it: name BoysBudget::kFp16, or call the float lane's entry to run the float "
                  "lane's arithmetic");

    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= static_cast<Bf16>(0.0f));

        float row[kMaxBoysOrder + 1];
        detail::BoysAllOrdersF32Impl<Policy>(nmax, static_cast<float>(x[i]), row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = static_cast<Bf16>(row[l]);
        }
    }
}

template <EvalPolicyLike Policy>
void BoysAllNAtOrdersBf16(const int* n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
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
        assert(x[i] >= static_cast<Bf16>(0.0f));

        float row[kMaxBoysOrder + 1];
        detail::BoysAllOrdersF32Impl<Policy>(n[i], static_cast<float>(x[i]), row);

        for (int l = 0; l <= n[i]; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = static_cast<Bf16>(row[l]);
        }
    }
}

// The half lanes' per-argument entries: the fp16 engine's ladder through the
// store this lane makes, at the policy named, so the two selections meet here as
// they do on the double lane.

// The packed half lane's own bodies, under the seam that declares the entries: the packed
// arithmetic is half2.hpp's, one correctly rounded operation at a time, and the entry the front
// door declares is defined here so that a consumer naming the lane compiles it.

namespace detail {

// Region C's asymptotic ladder in packed correctly rounded half arithmetic: a square root and a
// divide for the seed, then a packed multiply and divide per order. The roundings are relative,
// so the error is about one half ULP carried; the running value is scaled by 2^15 to keep the
// ladder's values normal down to F_n(x) = 2^-29, and the lane has a precondition (x >= kX1).

// The seed (1/2)sqrt(pi/x) is one divide by the square root, so the constant is
// sqrt(pi)/2 rounded to half once here and exact from then on.
constexpr F16 kHalfNativeSeedConstant = F16FromDouble(kBoysHalfSqrtPi);

// The per-order scale: the largest power of two binary16 holds.
constexpr F16 kHalfNativeScaleConstant = F16FromDouble(32768.0); // 2^15

/// F_0(x)..F_nmax(x) for one packed pair of arguments, scaled by 2^15.
///
/// The output is 2^15 * F_k(x) half by half; the scale is the same for every
/// order, so a caller unscales by one exact power-of-two division.
///
/// \param nmax Highest order, 0..kMaxBoysOrder.
/// \param x    Packed pair of arguments, both >= kX1.
/// \param out  Receives nmax + 1 packed pairs, out[k] = 2^15 * (F_k(x.low), F_k(x.high)).
inline void HalfNativeLadder(int nmax, Half2 x, Half2* out) noexcept {
    const Half2 seed = Half2::Broadcast(kHalfNativeSeedConstant);
    const Half2 scale = Half2::Broadcast(kHalfNativeScaleConstant);

    // Seed: F_0(x) = (1/2) sqrt(pi/x), then the exact per-order scale.
    Half2 f = Half2Mul(Half2Div(seed, Half2Sqrt(x)), scale);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        // The shipped region-C step, f = (l - 1/2) * f / x, without the exp(-x)/2 term the
        // scalar ladder carries: in this lane's arithmetic, scaled by 2^15, that term is
        // 4.21221e-09 at x = kX1 and under half's smallest subnormal (2^-24 = 5.96046e-08)
        // everywhere the lane serves, so it cannot change a delivered bit here.
        const F16 factor = F16FromDouble(static_cast<double>(l) - 0.5);
        f = Half2Div(Half2Mul(Half2::Broadcast(factor), f), x);
        out[l] = f;
    }
}

} // namespace detail

// Defined here rather than in a shipped unit, at the header that declares it: the entry is part
// of the front door's surface and takes the header's own packed type, so a consumer that names it
// compiles it. Its contract is the declaration's on boys/boys.hpp.
inline void BoysAllOrdersHalf2(int nmax, Half2 x, Half2* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x.Low() >= detail::F16FromDouble(detail::kX1) &&
           x.High() >= detail::F16FromDouble(detail::kX1));
    assert(out != nullptr);

    detail::HalfNativeLadder(nmax, x, out);
}

inline void BoysAllNF16Native(int nmax, const F16* x, F16* out, std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x != nullptr || count == 0);
    assert(out != nullptr || count == 0);

    for (std::size_t i = 0; i < count; i += 2)
    {
        // The pair is the unit of work; a lone trailing argument rides in both
        // halves of the register and only the low half is written back.
        const bool paired = i + 1 < count;
        const Half2 arguments = Half2(x[i], paired ? x[i + 1] : x[i]);

        assert(arguments.Low() >= detail::F16FromDouble(detail::kX1) &&
               arguments.High() >= detail::F16FromDouble(detail::kX1));

        Half2 values[kMaxBoysOrder + 1];
        detail::HalfNativeLadder(nmax, arguments, values);

        for (int k = 0; k <= nmax; ++k)
        {
            out[static_cast<std::size_t>(k) * count + i] = values[k].Low();

            if (paired)
            {
                out[static_cast<std::size_t>(k) * count + i + 1] = values[k].High();
            }
        }
    }
}

#endif // BoysFp16

} // namespace boys

/// \endcond
