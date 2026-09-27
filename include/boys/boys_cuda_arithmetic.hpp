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
/// and the division above. The route is a family and not a basis, so this is
/// beside the scheme's member rather than under it: a pair has one stored form
/// and either scheme sums it.
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

// Which basis a lane's coefficients are stored in. A lane that reads the
// monomial pool carries a kMonomial member and one that reads the Chebyshev
// pool does not, so the scheme axis is this trait and the summation below it,
// and every body above stays one body per (precision, shape).
template <typename Lane, typename = void>
struct LaneMonomial : std::false_type {};

template <typename Lane>
struct LaneMonomial<Lane, std::void_t<decltype(Lane::kMonomial)>> : std::true_type {};

// A lane whose pieces are rational pairs rather than polynomials: it carries a
// kRational member and names its two coefficient blocks and their two degrees in
// place of Deg. The route is a family of its own and not a basis, so the trait is
// beside the scheme's rather than under it - a rational piece has no Chebyshev
// form and no monomial one, and neither scheme changes the numbers it reads.
template <typename Lane, typename = void>
struct LaneRational : std::false_type {};

template <typename Lane>
struct LaneRational<Lane, std::void_t<decltype(Lane::kRational)>> : std::true_type {};

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
// The recurrence that consumes this value amplifies its error by the
// condition number of the forward recursion, which is up to 7.6e4 at the
// region-B boundary and falls as the argument grows. An approximation whose
// relative error grows with the argument therefore fails a bound that the
// same approximation holds everywhere else, so what matters is not the ulp
// count at one argument but whether that count stays flat.
//
//  - kFastExp is the hardware approximation with its argument-scaling
//    residual removed: __expf(y) evaluates 2^fl(y log2 e), and that one
//    rounding is the term that grows with |y|. The residual
//    fma(y, log2 e, -t) of that product is exact, and 2^(t + d) = 2^t 2^d
//    ~= 2^t (1 + d ln 2), so two fused steps take the error back to the
//    approximation's own few ulp, flat in the argument.
//  - otherwise the library routine, which is what the batch bodies compute:
//    at m = 1 a single and a batch evaluation of the same (n, x) return the
//    same bits outside region A.
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
