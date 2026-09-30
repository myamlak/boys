#pragma once

/// \cond
// Not API: the seed fits and the region recurrences the CUDA lane runs. The
// header ships because the lane's device-callable entries are header-defined
// and have to run the arithmetic the batch kernels run rather than a second
// copy of it; the API reference documents the entries.
///
/// One body per (precision, shape) and every one of them takes the tables as
/// a lane object, so the same body serves a batch kernel reading the
/// __constant__ tables and a device entry reading the caller's own handle.
/// A lane object supplies, all of them inlineable and all of them cheap:
///
///   int          Count(int order)                  pieces the fit is cut into
///   T            A(int order, int piece)           piece lower edge
///   T            B(int order, int piece)           piece upper edge
///   const T*     Coeffs(int order, int piece)      the piece's coefficients
///   int          Deg(int order, int piece)         the piece's degree
///   T            BSeed(T x, int order)             region-B seed at an argument
///
/// A lane that stores the monomial form of its fits carries a `kMonomial`
/// member as well, and the piece summation reads it: the same pieces at the
/// same degrees, summed by Horner in the monomial basis rather than by the
/// split Clenshaw in the Chebyshev one. It is optional because a lane's stored
/// form is a property of the tables it was handed and not of the body reading
/// them, and a lane without the member reads the Chebyshev pool.
///
/// A lane whose pieces are rational pairs carries a `kRational` member and
/// answers the piece read with four members in place of one: `NumDeg` and
/// `DenDeg` for the pair's two degrees, `Coeffs` for the numerator's block and
/// `DenCoeffs` for the denominator's, and the summation is the two Horner sums
/// and the division above. The route is a family and not a basis: a pair has
/// one stored form and either scheme sums it.
///
/// T is double for the double lane's seeds and float for the float lane's.
/// BSeed is the whole of region B's seed, taken at the argument rather than as
/// one polynomial over the region: the shipped partition's seed is one fit
/// over [kX0, kX1] and a second partition's is a piecewise one, so a lane
/// supplies the seed and not the coefficients of a fixed shape. It takes the
/// order because the relaxed batches read their region-B degree from the
/// order-0 entry — the F_0 seed's error reaches every output with gain at most
/// 1 + 1.846e-17 — while the single lanes read it per order; a lane object is
/// what decides which, and the arithmetic below is the same either way.
///
/// The region structure is the library's: region A (x below kX0) is the
/// piecewise Chebyshev fit of F_n itself, region B (up to kX1) is the upward
/// recursion seeded by the region-B fit of F_0, and region C is the F_0
/// asymptotic form with the same recursion. The all-orders bodies carry a
/// second, downward recursion inside region A: the fit there is of F_n, and
/// the lower orders come back down from it.

#include "boys/boys_coefficients.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace boys::detail {

/// The F_0 asymptotic prefactor: the integral that gives F_0 at large x is
/// sqrt(pi/x)/2, so F_0 = sqrt(pi)/2 * rsqrt(x) is the value the region-C
/// recursion starts from.
inline constexpr double kHalfSqrtPi = 0.886226925452758014;

// ---------------------------------------------------------------------------
// piecewise Chebyshev evaluation
// ---------------------------------------------------------------------------

// Split Clenshaw: the even and odd parts are summed apart and combined once,
// which keeps every step a fused multiply-add on a value of one sign. The
// deg <= 2 cases are unrolled because the split below assumes the odd part
// has at least two terms.
__device__ __forceinline__ double DeviceClenshawSplit(const double* c, int deg, double t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return __fma_rn(t, c[1], c[0]);
    }

    const double v = __fma_rn(2.0, t * t, -1.0);
    const double twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4
        // (the odd part's m = 1 finalization would double the t*c[1] term).
        return __fma_rn(t, c[1], __fma_rn(v, c[2], c[0]));
    }

    const int m = deg / 2;
    double b1 = c[2 * m];
    double b2 = 0.0;

    for (int k = m - 1; k >= 1; --k)
    {
        const double b0 = __fma_rn(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const double even = __fma_rn(v, b1, c[0] - b2);

    double o1 = c[2 * m - 1];
    double o2 = 0.0;

    for (int k = m - 2; k >= 1; --k)
    {
        const double o0 = __fma_rn(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const double odd = __fma_rn(twoV - 1.0, o1, c[1] - o2);
    return __fma_rn(t, odd, even);
}

__device__ __forceinline__ float DeviceClenshawSplit32(const float* c, int deg, float t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return __fmaf_rn(t, c[1], c[0]);
    }

    const float v = __fmaf_rn(2.0f, t * t, -1.0f);
    const float twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4
        // (the odd part's m = 1 finalization would double the t*c[1] term).
        return __fmaf_rn(t, c[1], __fmaf_rn(v, c[2], c[0]));
    }

    const int m = deg / 2;
    float b1 = c[2 * m];
    float b2 = 0.0f;

    for (int k = m - 1; k >= 1; --k)
    {
        const float b0 = __fmaf_rn(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const float even = __fmaf_rn(v, b1, c[0] - b2);

    float o1 = c[2 * m - 1];
    float o2 = 0.0f;

    for (int k = m - 2; k >= 1; --k)
    {
        const float o0 = __fmaf_rn(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const float odd = __fmaf_rn(twoV - 1.0f, o1, c[1] - o2);
    return __fmaf_rn(t, odd, even);
}

// The monomial scheme's summation: the same fit in the other basis, stored at
// the same offsets and degrees as the Chebyshev one and summed by Horner in
// ascending order. One multiply-add per coefficient, against the split
// Clenshaw's two, so the scheme is a cost choice at equal degree — the two
// tables carry the same fit and the delivered accuracy of each is its own
// certified row.
__device__ __forceinline__ double DeviceHornerMono(const double* c, int deg, double t) {
    double acc = c[deg];

    for (int j = deg - 1; j >= 0; --j)
    {
        acc = __fma_rn(acc, t, c[j]);
    }

    return acc;
}

// The float lanes' form of the same summation, the way DeviceClenshawSplit32 is
// the float lanes' split Clenshaw: one walk, in the lane's own arithmetic and
// with the same fused step, so a float piece's two forms are summed the way the
// two certified rows of their table were measured (kNarrowARowsF32,
// kFlatRowsF32, boys_coefficients.hpp). The double helper above is not it: it
// would read a float table through a double pointer and sum a double polynomial,
// which is a different arithmetic and a different rounding.
__device__ __forceinline__ float DeviceHornerMono32(const float* c, int deg, float t) {
    float acc = c[deg];

    for (int j = deg - 1; j >= 0; --j)
    {
        acc = __fmaf_rn(acc, t, c[j]);
    }

    return acc;
}

// The rational route's summation: a piece is a numerator and a denominator, both
// stored ascending, both summed by Horner, divided once. The denominator's
// constant term is held at one, so its own sum ends in a multiply-add against
// that one rather than carrying a coefficient for it; a pair whose denominator
// degree is zero has no denominator at all and the numerator is the value. It is
// the kernel's reading of the stored form, coefficient for coefficient the same
// as the host lane's piece and seed evaluations (boys_impl.hpp
// RationalPieceAtCut, RationalSeedAtCut): the two lanes' figures are comparable
// because they sum the same numbers the same way.
__device__ __forceinline__ double DeviceRatSum(const double* num,
                                              int numDeg,
                                              const double* den,
                                              int denDeg,
                                              double t) {
    double numerator = num[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        numerator = __fma_rn(numerator, t, num[j]);
    }

    if (denDeg == 0)
    {
        return numerator;
    }

    double denominator = den[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        denominator = __fma_rn(denominator, t, den[j]);
    }

    return numerator / __fma_rn(denominator, t, 1.0);
}

// The float lanes' form of that summation, walked in the float lanes'
// arithmetic the way DeviceClenshawSplit32 is their form of the split Clenshaw:
// the same two Horner sums, the same held denominator constant, the same one
// division, with every step fused. It is the kernel's reading of the same
// stored form, so an entry that reached it and the host lane's own float
// rational reading sum the same numbers the same way and their figures are
// comparable.
//
// The two coefficient blocks are the caller's to place: the float lane's
// shipped region-B seed stores its numerator and denominator in two arrays and
// its narrow partition stores them in one pool with the denominator above the
// numerator, and both are named by a pointer rather than computed here.
__device__ __forceinline__ float DeviceRatSum32(const float* num,
                                                int numDeg,
                                                const float* den,
                                                int denDeg,
                                                float t) {
    float numerator = num[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        numerator = __fmaf_rn(numerator, t, num[j]);
    }

    if (denDeg == 0)
    {
        return numerator;
    }

    float denominator = den[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        denominator = __fmaf_rn(denominator, t, den[j]);
    }

    return numerator / __fmaf_rn(denominator, t, 1.0f);
}

// Which basis a lane's coefficients are stored in. A lane that reads the
// monomial pool states so with a kMonomial member, and the trait reads its
// value rather than its presence: a form that shares its geometry with the
// other form of the same table states false here, and the bodies below then
// take the Chebyshev arm, which is the pool that form reads. So the scheme axis
// is this trait and the summation below it, and every body above stays one body
// per (precision, shape).
template <typename Lane, typename = void>
struct LaneMonomial : std::false_type {};

template <typename Lane>
struct LaneMonomial<Lane, std::void_t<decltype(Lane::kMonomial)>>
    : std::bool_constant<Lane::kMonomial> {};

// A lane whose pieces are rational pairs rather than polynomials: it carries a
// kRational member and names its two coefficient blocks and their two degrees in
// place of Deg. The route is a family of its own and not a basis, so the trait is
// beside the scheme's rather than under it - a rational piece has no Chebyshev
// form and no monomial one, and neither scheme changes the numbers it reads.
template <typename Lane, typename = void>
struct LaneRational : std::false_type {};

template <typename Lane>
struct LaneRational<Lane, std::void_t<decltype(Lane::kRational)>>
    : std::bool_constant<Lane::kRational> {};

template <typename Lane>
__device__ __forceinline__ double DevicePieceSum(const Lane& lane,
                                                 const double* c,
                                                 int deg,
                                                 double t) {
    if constexpr (LaneMonomial<Lane>::value)
    {
        return DeviceHornerMono(c, deg, t);
    } else
    {
        return DeviceClenshawSplit(c, deg, t);
    }
}

// ---------------------------------------------------------------------------
// seeds
// ---------------------------------------------------------------------------

// The piece a value falls in: the pieces of an order are ascending and their
// upper edges partition the order's fitted range, so the first edge above x
// names the piece (and the last piece holds everything above the topmost
// edge). The scan is linear — a binary search would not pay for its branches
// until the rows are much longer than the twelve pieces a lane may have.
template <typename Lane>
__device__ __forceinline__ double DeviceSeed(const Lane& lane, int order, double x) {
    const int count = lane.Count(order);
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < lane.B(order, i))
        {
            p = i;
            break;
        }
    }

    const double t = 2.0 * (x - lane.A(order, p)) / (lane.B(order, p) - lane.A(order, p)) - 1.0;

    if constexpr (LaneRational<Lane>::value)
    {
        return DeviceRatSum(lane.Coeffs(order, p),
                            lane.NumDeg(order, p),
                            lane.DenCoeffs(order, p),
                            lane.DenDeg(order, p),
                            t);
    } else
    {
        return DevicePieceSum(lane, lane.Coeffs(order, p), lane.Deg(order, p), t);
    }
}

template <typename Lane>
__device__ __forceinline__ float DeviceSeed32(const Lane& lane, int order, float x) {
    const int count = lane.Count(order);
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < lane.B(order, i))
        {
            p = i;
            break;
        }
    }

    const float* c = lane.Coeffs(order, p);
    const float t =
        2.0f * (x - lane.A(order, p)) / (lane.B(order, p) - lane.A(order, p)) - 1.0f;
    return DeviceClenshawSplit32(c, lane.Deg(order, p), t);
}

// ---------------------------------------------------------------------------
// the region-B exponential of the float lane
// ---------------------------------------------------------------------------

// The two arithmetics the single float entry offers (RegionBExp,
// boys_cuda.hpp), and the one factor of the region-B path a caller can trade
// accuracy for speed on.
//
//  - kFastExp is the hardware approximation with its argument-scaling
//    residual removed: __expf(y) evaluates 2^fl(y log2 e), and that one
//    rounding is the term that grows with |y|. The residual
//    fma(y, log2 e, -t) of that product is exact, and 2^(t + d) = 2^t 2^d
//    ~= 2^t (1 + d ln 2), so two fused steps take the error back to the
//    approximation's own few ulp, flat in the argument.
//  - otherwise the library routine, which is what the batch bodies compute.
template <bool kFastExp>
__device__ __forceinline__ float DeviceRegionBExp(float xx) {
    if constexpr (kFastExp)
    {
        const float y = -xx;
        const float t = y * 1.4426950408889634f;
        const float d = __fmaf_rn(y, 1.4426950408889634f, -t);
        return 0.5f * __expf(y) * __fmaf_rn(d, 0.6931471805599453f, 1.0f);
    } else
    {
        return 0.5f * expf(-xx);
    }
}

// ---------------------------------------------------------------------------
// one order at one argument
// ---------------------------------------------------------------------------

// Region A asks the fit for the order directly; the higher regions seed F_0
// and recur upward once per order, since a fit of every order over the whole
// range would cost more table than the recursion costs work.
template <typename Lane>
__device__ __forceinline__ double DeviceSingleF64(const Lane& lane, int order, double xx) {
    if (xx < kX0)
    {
        return DeviceSeed(lane, order, xx);
    }

    double f = lane.BSeed(xx, order);

    if (xx < kX1)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5) * f - expx) / xx;
        }

        return f;
    }

    f = kHalfSqrtPi * rsqrt(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5) * f / xx;
    }

    return f;
}

// The float lane's single body, and the fp16 lane's: the fp16 entries round
// at the boundary and run this engine in between, so there is one body and
// not two.
template <bool kFastExp, typename Lane>
__device__ __forceinline__ float DeviceSingleF32(const Lane& lane, int order, float xx) {
    if (xx < static_cast<float>(kX0))
    {
        return DeviceSeed32(lane, order, xx);
    }

    if (xx < static_cast<float>(kX1))
    {
        float f = lane.BSeed(xx, order);
        const float expx = DeviceRegionBExp<kFastExp>(xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5f) * f - expx) / xx;
        }

        return f;
    }

    float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5f) * f / xx;
    }

    return f;
}

// ---------------------------------------------------------------------------
// every order at one argument
// ---------------------------------------------------------------------------

// `store(int order, double value)` is called once per order, in the order the
// recursion produces it, so a batch kernel can write each value straight to
// its plane and a caller can keep the ladder in registers. Region A descends,
// so its stores arrive high order first and the value in the register chain
// is the one the downward recursion carries.
template <typename Lane, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF64(
    const Lane& lane, int order, double xx, Store store) {
    if (xx < kX0)
    {
        double f = DeviceSeed(lane, order, xx);
        store(order, f);
        const double expx = 0.5 * exp(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5);
            store(l, f);
        }

        return;
    }

    double f = lane.BSeed(xx, order);
    store(0, f);

    if (xx < kX1)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5) * f - expx) / xx;
            store(l, f);
        }

        return;
    }

    f = kHalfSqrtPi * rsqrt(xx);
    store(0, f); // F_0 from the asymptotic form: the region-B seed computed
                 // above is outside its validity domain here

    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5) * f / xx;
        store(l, f);
    }
}

// The float lane's all-orders body and the fp16 lane's. The region-A seed is
// the DOUBLE piece table even on the float lanes: the downward recursion
// amplifies a float seed error past the float budget, so the seed is computed
// in double and rounded once on entry to the recursion. That is what the
// second lane argument is, and it is why this body takes two of them.
template <typename SeedLane, typename Lane, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, Store store) {
    if (xx < static_cast<float>(kX0))
    {
        const double seed = DeviceSeed(seedLane, order, static_cast<double>(xx));
        float f = static_cast<float>(seed);
        store(order, f);
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5f);
            store(l, f);
        }

        return;
    }

    float f = lane.BSeed(xx, order);
    store(0, f);

    if (xx < static_cast<float>(kX1))
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5f) * f - expx) / xx;
            store(l, f);
        }

        return;
    }

    f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    store(0, f); // F_0 from the asymptotic form: the region-B seed computed
                 // above is outside its validity domain here

    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5f) * f / xx;
        store(l, f);
    }
}

// ---------------------------------------------------------------------------
// every order at one argument, from one uniform grid
// ---------------------------------------------------------------------------

// The route whose fit is one table of equal intervals over [0, kFlatHi)
// rather than pieces cut where the function needs them: every order's block
// sits at a fixed offset inside its interval's, so the interval an argument
// falls in is one multiply and a truncation and the degree is a property of
// the stored table rather than of the argument. The geometry is compile-time,
// which is why these bodies take the two coefficient pools and nothing else —
// there are no per-interval edges or degrees to supply, so a lane object would
// be a pair of pointers with no behaviour behind it.
//
// The pools are the two stored forms of one fit, and kMonomial selects the
// reader and nothing else, exactly as a lane's own kMonomial member does: the
// index arithmetic, the join at kFlatHi and the asymptotic arm above it are
// one spelling for both forms, so the two cannot come to disagree about which
// interval an argument falls in or where the table stops.
//
// Above kFlatHi no fit reaches and the call falls to the one-term asymptotic
// and its own upward recurrence — the arm region C runs elsewhere, read from
// the same prefactor and stepping by the same (l + 1/2)/x. The join needs no
// interpolation between the two: kFlatHi is above kX1, so an argument the
// table does not serve is one the asymptotic already served.

// The uniform route's ladder, in double.
template <bool kMonomial, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF64Flat(const double* cheb,
                                                       const double* mono,
                                                       const int* degs,
                                                       const int* offsets,
                                                       int order,
                                                       double xx,
                                                       Store store) {
    if (xx >= kFlatHi)
    {
        double f = kHalfSqrtPi * rsqrt(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = (l + 0.5) * f / xx;
        }

        return;
    }

    // The interval index is the argument times the reciprocal of the stored
    // interval width, truncated, so the product is the interval's own boundary
    // and not a rounding of it: the reciprocal is exact here, and the two
    // assertions below hold the factor and the grid to the stored width rather
    // than restating either. The clamp is unreachable below the join and is
    // kept as the guard the host's own spelling of this map keeps.
    constexpr double kPerUnit = 1.0 / kFlatWidth;
    static_assert(kFlatWidth * kPerUnit == 1.0,
                  "the uniform grid's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * kFlatHi == static_cast<double>(kFlatIntervals),
                  "the uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const double u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > kFlatIntervals - 1)
    {
        iv = kFlatIntervals - 1;
    }

    const double t = 2.0 * (u - static_cast<double>(iv)) - 1.0;

    // The interval's own block, at the interval's own degree. Both are read
    // per interval because the grid's cells do not all carry the same count:
    // each was given the smallest admissible even degree its own truncation
    // bound holds it to, so an interval's block is (its degree + 1)
    // coefficients per order and the table has no stride a reader could
    // assume. A reader that assumed one would sum a neighbouring cell's
    // polynomial and no check of the coefficients alone would report it.
    const int deg = degs[iv];
    const double* interval = (kMonomial ? mono : cheb) + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const double* c =
            interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        if constexpr (kMonomial)
        {
            store(l, DeviceHornerMono(c, deg, t));
        }
        else
        {
            store(l, DeviceClenshawSplit(c, deg, t));
        }
    }
}

// The float lane's uniform ladder. The mapped argument is this lane's own
// spelling and not the double body's: the double lane maps x by the exact
// product x * kPerUnit truncated, which is the grid's own boundary and is exact
// in its arithmetic, while the float lane maps by 2 (x - a)/(b - a) - 1 with a
// and b the interval's own edges. That is the map the lane's fits were read at
// when they were measured (tools/gen_boys_coefficients.py, f32_map) and the map
// its narrow pieces are read with (boys_impl.hpp, ChebyshevValueF32): one map
// for the lane. The edges are the grid's own boundaries as the generator
// fitted on them (tools/gen_boys_coefficients.py, flat_order_f32: a = iv *
// width, b = a + width) and not a scan of stored edges, so the interval an
// argument is mapped inside is the one the fit was measured in.
template <bool kMonomial, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF32Flat(const float* cheb,
                                                       const float* mono,
                                                       const int* degs,
                                                       const int* offsets,
                                                       int order,
                                                       float xx,
                                                       Store store) {
    if (xx >= f32::kFlatHiF32)
    {
        float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = (l + 0.5f) * f / xx;
        }

        return;
    }

    // The lane's own factor, and the host lane's spelling of it: the width's
    // reciprocal, which the locate multiplies by, and which the grid's own
    // identity below is written against. The two are one requirement, and
    // boys_impl.hpp asserts the same product from the host side.
    constexpr float kPerUnit = 1.0f / f32::kFlatWidthF32;
    static_assert(f32::kFlatWidthF32 * kPerUnit == 1.0f,
                  "the float lane's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * f32::kFlatHiF32 == static_cast<float>(f32::kFlatIntervalsF32),
                  "the float lane's uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const float u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > f32::kFlatIntervalsF32 - 1)
    {
        iv = f32::kFlatIntervalsF32 - 1;
    }

    const float a = static_cast<float>(iv) / kPerUnit;
    const float b = static_cast<float>(iv + 1) / kPerUnit;
    const float t = 2.0f * (xx - a) / (b - a) - 1.0f;

    // The interval's own block at the interval's own degree, as the double
    // body reads it and for the same reason: this lane's cells carry their own
    // degrees too, so there is no stride to read them with.
    const int deg = degs[iv];
    const float* interval = (kMonomial ? mono : cheb) + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const float* c =
            interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        if constexpr (kMonomial)
        {
            store(l, DeviceHornerMono32(c, deg, t));
        }
        else
        {
            store(l, DeviceClenshawSplit32(c, deg, t));
        }
    }
}

// ---------------------------------------------------------------------------
// every order at one argument, each from its own fit
// ---------------------------------------------------------------------------

// The same ladder as the body above with region A read the other way: every
// order's own piece is located and its own fit summed, so no value here came
// down a recurrence from a higher order's fit. The two agree to the fit's own
// accuracy and differ in where the rounding happens, which is the whole of
// what the choice between them buys.
//
// The axis covers region A and nothing else — past kX0 the body is the
// certified one above, which is also where the region-A fit it replaces ends.
// Outside region A the two bodies are one body, so a row that carries this
// axis names its interval as region A rather than claiming the rest.
template <typename Lane, typename Store>
__device__ __forceinline__ void DeviceOrdersF64(
    const Lane& lane, int order, double xx, Store store) {
    if (xx < kX0)
    {
        for (int l = 0; l <= order; ++l)
        {
            store(l, DeviceSeed(lane, l, xx));
        }

        return;
    }

    DeviceAllOrdersF64(lane, order, xx, store);
}

// The float lane's and the fp16 lane's, over the double region-A seed lane the
// body above takes for the same reason.
template <typename SeedLane, typename Lane, typename Store>
__device__ __forceinline__ void DeviceOrdersF32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, Store store) {
    if (xx < static_cast<float>(kX0))
    {
        for (int l = 0; l <= order; ++l)
        {
            store(l, static_cast<float>(DeviceSeed(seedLane, l, static_cast<double>(xx))));
        }

        return;
    }

    DeviceAllOrdersF32(seedLane, lane, order, xx, store);
}

} // namespace boys::detail

/// \endcond
