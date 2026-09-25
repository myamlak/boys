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
///   const T*     BSeedCoeffs()                     region-B seed coefficients
///   int          BSeedDeg(int order)               region-B seed degree
///
/// T is double for the double lane's seeds and float for the float lane's.
/// BSeedDeg takes the order because the relaxed batches read their region-B
/// degree from the order-0 entry — the F_0 seed's error reaches every output
/// with gain at most 1 + 1.846e-17 — while the single lanes read it per order;
/// a lane object is what decides which, and the arithmetic below is the same
/// either way.
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

    const double* c = lane.Coeffs(order, p);
    const double t = 2.0 * (x - lane.A(order, p)) / (lane.B(order, p) - lane.A(order, p)) - 1.0;
    return DeviceClenshawSplit(c, lane.Deg(order, p), t);
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

// The region-B seed: one fit of F_0 over the whole of region B, with the
// degree the caller's lane names for this order.
template <typename Lane>
__device__ __forceinline__ double DeviceSeedB(const Lane& lane, double x, int order) {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
    return DeviceClenshawSplit(lane.BSeedCoeffs(), lane.BSeedDeg(order), t);
}

template <typename Lane>
__device__ __forceinline__ float DeviceSeedB32(const Lane& lane, float x, int order) {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return DeviceClenshawSplit32(lane.BSeedCoeffs(), lane.BSeedDeg(order), t);
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

    double f = DeviceSeedB(lane, xx, order);

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
        float f = DeviceSeedB32(lane, xx, order);
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

    double f = DeviceSeedB(lane, xx, order);
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

    float f = DeviceSeedB32(lane, xx, order);
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

} // namespace boys::detail

/// \endcond
