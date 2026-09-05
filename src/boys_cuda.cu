// CUDA lane of the Boys kernel. Mirrors the scalar/simd design (boys.cpp,
// boys_simd.cpp): piecewise Chebyshev seeds + recurrences, region structure
// A/B/C with the weighted region-A fits (see tools/gen_boys_coefficients.py).
// C++20 with a CUDA-safe include list only: the library's C++23 headers
// would poison the nvcc translation unit.

#include "boys_coefficients.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace boysymmetriad {
namespace {

constexpr int kMaxPieces = 12;
constexpr int kMaxCoeffs = 2304;

constexpr double kX0d = detail::kX0;
constexpr double kX1d = detail::kX1;
constexpr double kHalfSqrtPi = 0.886226925452758014;

// Constant-memory tables (double lane: 1876 coeffs ~ 15KB; float lane ~ 6KB;
// piece metadata small - all within the 64KB constant budget).
__constant__ double cCoeffs[kMaxCoeffs];
__constant__ int cOffset[33][kMaxPieces];
__constant__ double cA[33][kMaxPieces];
__constant__ double cB[33][kMaxPieces];
__constant__ int cDeg[33][kMaxPieces];
__constant__ int cCount[33];
__constant__ double cBcoeffs[24];
__constant__ int cBDeg;

__constant__ float cCoeffs32[kMaxCoeffs];
__constant__ int cOffset32[33][kMaxPieces];
__constant__ float cA32[33][kMaxPieces];
__constant__ float cB32[33][kMaxPieces];
__constant__ int cDeg32[33][kMaxPieces];
__constant__ int cCount32[33];
__constant__ float cBcoeffs32[24];
__constant__ int cBDeg32;

// The accuracy-multiplier effective-degree tables (design record D-F6 of
// the accompanying paper): one lane per CUDA entry
// point, in the order the host fills them (boys_cuda.cpp FillEffLane):
//   0 = double single (BoysSingleF64KernelEff, region-B degree per order)
//   1 = double batch (BoysBatchF64KernelEff, region-B degree = order-0 entry)
//   2 = float single (BoysSingleF32KernelEff, region-B degree per order)
//   3 = float batch (BoysBatchF32KernelEff, region-B degree = order-0 entry;
//       the region-A seed is the DOUBLE piece table, budget 1.5e-7)
//   4 = fp16 single (BoysSingleF16KernelEff, region-B degree per order)
//   5 = fp16 batch (BoysBatchF16KernelEff, region-B degree = order-0 entry;
//       the region-A seed is the DOUBLE piece table, budget 1e-7)
// Filled host-side once per (device, m) — the single-m-per-process cache,
// same idempotence contract as BoysCudaUploadTables. The relaxed kernels
// read these exactly like the kernels above read cDeg/cBDeg (the degree
// shape is unchanged; only the values differ).
constexpr int kEffLaneCount = 6;
__constant__ int cDegEff[kEffLaneCount][33][kMaxPieces];
__constant__ int cBDegEff[kEffLaneCount][33];

// ---------------------------------------------------------------------------
// device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ double DevClenshawSplit(const double* c, int deg, double t) {
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

__device__ __forceinline__ float DevClenshawSplit32(const float* c, int deg, float t) {
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

__device__ __forceinline__ double DevSeed(int order, double x) {
    const int count = cCount[order];
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < cB[order][i])
        {
            p = i;
            break;
        }
    }

    const double* c = cCoeffs + cOffset[order][p];
    const double t = 2.0 * (x - cA[order][p]) / (cB[order][p] - cA[order][p]) - 1.0;
    return DevClenshawSplit(c, cDeg[order][p], t);
}

__device__ __forceinline__ float DevSeed32(int order, float x) {
    const int count = cCount32[order];
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < cB32[order][i])
        {
            p = i;
            break;
        }
    }

    const float* c = cCoeffs32 + cOffset32[order][p];
    const float t = 2.0f * (x - cA32[order][p]) / (cB32[order][p] - cA32[order][p]) - 1.0f;
    return DevClenshawSplit32(c, cDeg32[order][p], t);
}

__device__ __forceinline__ double DevSeedB(double x) {
    const double t = 2.0 * (x - kX0d) / (kX1d - kX0d) - 1.0;
    return DevClenshawSplit(cBcoeffs, cBDeg, t);
}

__device__ __forceinline__ float DevSeedB32(float x) {
    const float t = 2.0f * (x - static_cast<float>(kX0d)) / static_cast<float>(kX1d - kX0d) - 1.0f;
    return DevClenshawSplit32(cBcoeffs32, cBDeg32, t);
}

// ---------------------------------------------------------------------------
// effective-degree variants (the D-F3 relaxation path)
// ---------------------------------------------------------------------------
// Identical shapes to the full-accuracy helpers above; the degrees come from
// the per-lane cDegEff/cBDegEff tables instead of cDeg/cBDeg. kLane is the
// compile-time lane index of the calling kernel (cDegEff[0..5] above); the
// region-B entry is the per-order degree for the single lanes and the
// order-0 entry for the batch lanes (the F0 seed's error reaches every
// output with amplification <= A_B(0) = 1 — the CPU fix, boys_impl.hpp).
template <int kLane> __device__ __forceinline__ double DevSeedEff(int order, double x) {
    const int count = cCount[order];
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < cB[order][i])
        {
            p = i;
            break;
        }
    }

    const double* c = cCoeffs + cOffset[order][p];
    const double t = 2.0 * (x - cA[order][p]) / (cB[order][p] - cA[order][p]) - 1.0;
    return DevClenshawSplit(c, cDegEff[kLane][order][p], t);
}

template <int kLane> __device__ __forceinline__ float DevSeed32Eff(int order, float x) {
    const int count = cCount32[order];
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < cB32[order][i])
        {
            p = i;
            break;
        }
    }

    const float* c = cCoeffs32 + cOffset32[order][p];
    const float t = 2.0f * (x - cA32[order][p]) / (cB32[order][p] - cA32[order][p]) - 1.0f;
    return DevClenshawSplit32(c, cDegEff[kLane][order][p], t);
}

template <int kLane> __device__ __forceinline__ double DevSeedBEff(double x, int order) {
    const double t = 2.0 * (x - kX0d) / (kX1d - kX0d) - 1.0;
    return DevClenshawSplit(cBcoeffs, cBDegEff[kLane][order], t);
}

template <int kLane> __device__ __forceinline__ float DevSeedB32Eff(float x, int order) {
    const float t = 2.0f * (x - static_cast<float>(kX0d)) / static_cast<float>(kX1d - kX0d) - 1.0f;
    return DevClenshawSplit32(cBcoeffs32, cBDegEff[kLane][order], t);
}

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
// __restrict__ on x/out: the buffers are distinct DeviceBuffers by construction,
// and without it every out store would force nvcc to reload x (assumed aliasing).
__global__ void BoysSingleF32Kernel(const int* n,
                                    const double* __restrict__ x,
                                    float* __restrict__ out,
                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = static_cast<float>(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        out[i] = DevSeed32(order, xx);
        return;
    }

    if (xx < static_cast<float>(kX1d))
    {
        float f = DevSeedB32(xx);
        const float expx = 0.5f * __expf(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5f) * f - expx) / xx;
        }

        out[i] = f;
        return;
    }

    float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5f) * f / xx;
    }

    out[i] = f;
}

// See BoysSingleF32Kernel: x and out never alias.
__global__ void BoysBatchF32Kernel(const int* n,
                                   const double* __restrict__ x,
                                   float* __restrict__ out,
                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = static_cast<float>(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        // Double seed: the downward recursion amplifies a float seed error
        // beyond the certified 1.5e-7 float budget (see BoysBatchF32 in
        // boys.cpp).
        const double seed = DevSeed(order, static_cast<double>(x[i]));
        float f = static_cast<float>(seed);
        out[order * count + i] = f;
        // expf, not __expf: the downward recursion amplifies the e^{-x}
        // rounding error; 2 ulp of __expf would eat most of the 1.5e-7
        // budget.
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5f);
            out[l * count + i] = f;
        }

        return;
    }

    float f = DevSeedB32(xx);
    out[i] = f;

    if (xx < static_cast<float>(kX1d))
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5f) * f - expx) / xx;
            out[l * count + i] = f;
        }

        return;
    }

    f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    out[i] = f; // F_0 from the asymptotic form (the region-B seed stored
                // above is outside its validity domain here)
    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5f) * f / xx;
        out[l * count + i] = f;
    }
}

__global__ void BoysSingleF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double xx = x[i];

    if (xx < kX0d)
    {
        out[i] = DevSeed(order, xx);
        return;
    }

    double f = DevSeedB(xx);

    if (xx < kX1d)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5) * f - expx) / xx;
        }

        out[i] = f;
        return;
    }

    f = kHalfSqrtPi * rsqrt(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5) * f / xx;
    }

    out[i] = f;
}

__global__ void BoysBatchF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double xx = x[i];

    if (xx < kX0d)
    {
        double f = DevSeed(order, xx);
        out[order * count + i] = f;
        const double expx = 0.5 * exp(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5);
            out[l * count + i] = f;
        }

        return;
    }

    double f = DevSeedB(xx);
    out[i] = f;

    if (xx < kX1d)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5) * f - expx) / xx;
            out[l * count + i] = f;
        }

        return;
    }

    f = kHalfSqrtPi * rsqrt(xx);
    out[i] = f; // F_0 from the asymptotic form (the region-B seed stored
                // above is outside its validity domain here)
    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5) * f / xx;
        out[l * count + i] = f;
    }
}

// ---------------------------------------------------------------------------
// fp16 lane (the certified mixed-precision extension; BoysFp16 seam).
// __half I/O around the fp32 engine above -- the same __constant__ tables,
// the same region-partitioned dispatch, exactly the DevSeed*/expf paths of
// the F32 kernels.
// ---------------------------------------------------------------------------
#if BoysFp16
__global__ void BoysSingleF16Kernel(const int* n,
                                    const __half* __restrict__ x,
                                    __half* __restrict__ out,
                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = __half2float(x[i]);

    float f;

    if (xx < static_cast<float>(kX0d))
    {
        f = DevSeed32(order, xx);
    } else if (xx < static_cast<float>(kX1d))
    {
        f = DevSeedB32(xx);
        const float expx = 0.5f * expf(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5f) * f - expx) / xx;
        }
    } else
    {
        f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

        for (int l = 0; l < order; ++l)
        {
            f = (l + 0.5f) * f / xx;
        }
    }

    out[i] = __float2half(f);
}

// See BoysSingleF16Kernel; the batch layout matches BoysBatchF32Kernel
// (the region-A seed in double precision -- the downward recursion
// amplifies float seed errors beyond the 1e-7 budget).
__global__ void BoysBatchF16Kernel(const int* n,
                                   const __half* __restrict__ x,
                                   __half* __restrict__ out,
                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = __half2float(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        const double seed = DevSeed(order, static_cast<double>(xx));
        float f = static_cast<float>(seed);
        out[order * count + i] = __float2half(f);
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5f);
            out[l * count + i] = __float2half(f);
        }

        return;
    }

    float f = DevSeedB32(xx);
    out[i] = __float2half(f);

    if (xx < static_cast<float>(kX1d))
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5f) * f - expx) / xx;
            out[l * count + i] = __float2half(f);
        }

        return;
    }

    f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    out[i] = __float2half(f); // F_0 from the asymptotic form (see the F32 kernel)

    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5f) * f / xx;
        out[l * count + i] = __float2half(f);
    }
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// effective-degree kernels (the D-F3 relaxation path)
// ---------------------------------------------------------------------------
// One per CUDA lane, mirroring the full-accuracy kernels above exactly
// (same arithmetic, same region structure, same output layout) except that
// the seed degrees come from cDegEff/cBDegEff — the m = 1 kernels stay
// byte-identical (the full-accuracy pin). The lane index is the kernel's
// identity:
//   0 double single, 1 double batch, 2 float single, 3 float batch,
//   4 fp16 single, 5 fp16 batch.
// The batch lanes read the order-0 region-B entry (the F0 seed's error
// reaches every output with amplification <= A_B(0) = 1) and the per-order
// region-A entry at the batch's top order (the A_A(nmax) amplification
// covers the downward recursion) — exactly the CPU relaxed batches.
template <int kLane>
__global__ void BoysSingleF64KernelEff(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double xx = x[i];

    if (xx < kX0d)
    {
        out[i] = DevSeedEff<kLane>(order, xx);
        return;
    }

    double f = DevSeedBEff<kLane>(xx, order);

    if (xx < kX1d)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5) * f - expx) / xx;
        }

        out[i] = f;
        return;
    }

    f = kHalfSqrtPi * rsqrt(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5) * f / xx;
    }

    out[i] = f;
}

template <int kLane>
__global__ void BoysBatchF64KernelEff(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double xx = x[i];

    if (xx < kX0d)
    {
        double f = DevSeedEff<kLane>(order, xx);
        out[order * count + i] = f;
        const double expx = 0.5 * exp(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5);
            out[l * count + i] = f;
        }

        return;
    }

    double f = DevSeedBEff<kLane>(xx, 0);
    out[i] = f;

    if (xx < kX1d)
    {
        const double expx = 0.5 * exp(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5) * f - expx) / xx;
            out[l * count + i] = f;
        }

        return;
    }

    f = kHalfSqrtPi * rsqrt(xx);
    out[i] = f; // F_0 from the asymptotic form (the region-B seed stored
                // above is outside its validity domain here)
    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5) * f / xx;
        out[l * count + i] = f;
    }
}

template <int kLane>
__global__ void BoysSingleF32KernelEff(const int* n,
                                       const double* __restrict__ x,
                                       float* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = static_cast<float>(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        out[i] = DevSeed32Eff<kLane>(order, xx);
        return;
    }

    if (xx < static_cast<float>(kX1d))
    {
        float f = DevSeedB32Eff<kLane>(xx, order);
        const float expx = 0.5f * __expf(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5f) * f - expx) / xx;
        }

        out[i] = f;
        return;
    }

    float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

    for (int l = 0; l < order; ++l)
    {
        f = (l + 0.5f) * f / xx;
    }

    out[i] = f;
}

template <int kLane>
__global__ void BoysBatchF32KernelEff(const int* n,
                                      const double* __restrict__ x,
                                      float* __restrict__ out,
                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = static_cast<float>(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        // Double seed, as the full-accuracy kernel: the downward recursion
        // amplifies a float seed error beyond the lane budget (the float
        // lane's 1.5e-7; the fp16 roles' tighter 1e-7 base).
        const double seed = DevSeedEff<kLane>(order, static_cast<double>(x[i]));
        float f = static_cast<float>(seed);
        out[order * count + i] = f;
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5f);
            out[l * count + i] = f;
        }

        return;
    }

    float f = DevSeedB32Eff<kLane>(xx, 0);
    out[i] = f;

    if (xx < static_cast<float>(kX1d))
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5f) * f - expx) / xx;
            out[l * count + i] = f;
        }

        return;
    }

    f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    out[i] = f; // F_0 from the asymptotic form (the region-B seed stored
                // above is outside its validity domain here)
    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5f) * f / xx;
        out[l * count + i] = f;
    }
}

#if BoysFp16
template <int kLane>
__global__ void BoysSingleF16KernelEff(const int* n,
                                       const __half* __restrict__ x,
                                       __half* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = __half2float(x[i]);

    float f;

    if (xx < static_cast<float>(kX0d))
    {
        f = DevSeed32Eff<kLane>(order, xx);
    } else if (xx < static_cast<float>(kX1d))
    {
        f = DevSeedB32Eff<kLane>(xx, order);
        const float expx = 0.5f * expf(-xx);

        for (int l = 0; l < order; ++l)
        {
            f = ((l + 0.5f) * f - expx) / xx;
        }
    } else
    {
        f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);

        for (int l = 0; l < order; ++l)
        {
            f = (l + 0.5f) * f / xx;
        }
    }

    out[i] = __float2half(f);
}

template <int kLane>
__global__ void BoysBatchF16KernelEff(const int* n,
                                      const __half* __restrict__ x,
                                      __half* __restrict__ out,
                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float xx = __half2float(x[i]);

    if (xx < static_cast<float>(kX0d))
    {
        const double seed = DevSeedEff<kLane>(order, static_cast<double>(xx));
        float f = static_cast<float>(seed);
        out[order * count + i] = __float2half(f);
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = (xx * f + expx) / (l + 0.5f);
            out[l * count + i] = __float2half(f);
        }

        return;
    }

    float f = DevSeedB32Eff<kLane>(xx, 0);
    out[i] = __float2half(f);

    if (xx < static_cast<float>(kX1d))
    {
        const float expx = 0.5f * expf(-xx);

        for (int l = 1; l <= order; ++l)
        {
            f = ((l - 0.5f) * f - expx) / xx;
            out[l * count + i] = __float2half(f);
        }

        return;
    }

    f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    out[i] = __float2half(f);

    for (int l = 1; l <= order; ++l)
    {
        f = (l - 0.5f) * f / xx;
        out[l * count + i] = __float2half(f);
    }
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// host layer: plain exported functions (no library C++ types — this TU
// stays C++20). The BoysStatus layer lives in boys_cuda.cpp (C++23).
// ---------------------------------------------------------------------------
int gTablesDevice = -1; // device ordinal the tables were uploaded to

extern "C" int BoysCudaUploadTables() {
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return 2;
    }

    if (gTablesDevice == device)
    {
        return 0;
    }

    // Double lane.
    {
        double coeffs[kMaxCoeffs] = {};
        int offset[33][kMaxPieces] = {};
        double a[33][kMaxPieces] = {};
        double b[33][kMaxPieces] = {};
        int deg[33][kMaxPieces] = {};
        int count[33] = {};
        int runningOffset = 0;

        for (int o = 0; o <= detail::kMaxOrder; ++o)
        {
            const int first = detail::kPieceStart[o];
            const int last = detail::kPieceStart[o + 1];

            if (last - first > kMaxPieces)
            {
                return 1;
            }

            for (int p = first; p < last; ++p)
            {
                const detail::OrderPiece& piece = detail::kPieces[p];
                const int index = p - first;
                a[o][index] = piece.a;
                b[o][index] = piece.b;
                deg[o][index] = piece.deg;
                offset[o][index] = runningOffset;

                for (int k = 0; k <= piece.deg; ++k)
                {
                    if (runningOffset + k >= kMaxCoeffs)
                    {
                        return 1;
                    }

                    coeffs[runningOffset + k] = detail::kCoeffs[piece.offset + k];
                }

                runningOffset += piece.deg + 1;
            }

            count[o] = last - first;
        }

        if (cudaMemcpyToSymbol(cCoeffs, coeffs, sizeof(coeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cOffset, offset, sizeof(offset)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cA, a, sizeof(a)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cB, b, sizeof(b)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cDeg, deg, sizeof(deg)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cCount, count, sizeof(count)) != cudaSuccess)
        {
            return 2;
        }
        // std::array does not decay to a pointer: hand the runtime the data.
        if (cudaMemcpyToSymbol(cBcoeffs, detail::kBcoeffs.data(), sizeof(detail::kBcoeffs)) !=
            cudaSuccess)
        {
            return 2;
        }

        const int bDeg = detail::kBDeg;

        if (cudaMemcpyToSymbol(cBDeg, &bDeg, sizeof(int)) != cudaSuccess)
        {
            return 2;
        }
    }

    // Float lane.
    {
        float coeffs[kMaxCoeffs] = {};
        int offset[33][kMaxPieces] = {};
        float a[33][kMaxPieces] = {};
        float b[33][kMaxPieces] = {};
        int deg[33][kMaxPieces] = {};
        int count[33] = {};
        int runningOffset = 0;

        for (int o = 0; o <= detail::kMaxOrder; ++o)
        {
            const int first = detail::f32::kPieceStart[o];
            const int last = detail::f32::kPieceStart[o + 1];

            if (last - first > kMaxPieces)
            {
                return 1;
            }

            for (int p = first; p < last; ++p)
            {
                const detail::f32::OrderPiece& piece = detail::f32::kPieces[p];
                const int index = p - first;
                a[o][index] = piece.a;
                b[o][index] = piece.b;
                deg[o][index] = piece.deg;
                offset[o][index] = runningOffset;

                for (int k = 0; k <= piece.deg; ++k)
                {
                    if (runningOffset + k >= kMaxCoeffs)
                    {
                        return 1;
                    }

                    coeffs[runningOffset + k] = detail::f32::kCoeffs[piece.offset + k];
                }

                runningOffset += piece.deg + 1;
            }

            count[o] = last - first;
        }

        if (cudaMemcpyToSymbol(cCoeffs32, coeffs, sizeof(coeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cOffset32, offset, sizeof(offset)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cA32, a, sizeof(a)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cB32, b, sizeof(b)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cDeg32, deg, sizeof(deg)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(cCount32, count, sizeof(count)) != cudaSuccess)
        {
            return 2;
        }
        // std::array does not decay to a pointer: hand the runtime the data.
        if (cudaMemcpyToSymbol(cBcoeffs32,
                               detail::f32::kBcoeffs.data(),
                               sizeof(detail::f32::kBcoeffs)) != cudaSuccess)
        {
            return 2;
        }

        const int bDeg32 = detail::f32::kBDeg;

        if (cudaMemcpyToSymbol(cBDeg32, &bDeg32, sizeof(int)) != cudaSuccess)
        {
            return 2;
        }
    }

    // The cudaMemcpyToSymbol calls above are asynchronous (default stream);
    // sync before the guard flips so a concurrent launch on a non-default
    // stream can never read partially uploaded constant tables.
    const cudaError_t uploadSync = cudaStreamSynchronize(static_cast<cudaStream_t>(0));

    if (uploadSync != cudaSuccess)
    {
        return static_cast<int>(uploadSync);
    }

    gTablesDevice = device;
    return 0;
}

namespace {

int LaunchBlocks(std::size_t count) {
    const int threads = 256;
    return static_cast<int>((count + threads - 1) / threads);
}

} // namespace

extern "C" int BoysCudaLaunchSingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysSingleF32Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysBatchF32Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysSingleF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysBatchF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

#if BoysFp16
// Async launch contract (uniform with the F32/F64 launchers): the kernel
// is queued on the caller's stream and the call returns once the launch is
// accepted — callers synchronize the stream before reading the outputs.
extern "C" int BoysCudaLaunchSingleF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysSingleF16Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysBatchF16Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}
#endif // BoysFp16

// Accuracy-multiplier effective-degree upload (design record D-F6 of
// the accompanying paper). The host layer (boys_cuda.cpp, C++23) computes
// the per-lane degree tables from the
// constexpr machinery and hands them over as flat arrays in the lane order
// documented at cDegEff; this function caches the upload per (device, m) —
// the single-m-per-process contract, same idempotence as BoysCudaUploadTables.
// degA layout: [lane][order][pieceInOrder] (kEffLaneCount x 33 x kMaxPieces);
// degB layout: [lane][order] (kEffLaneCount x 33). Return codes as the
// launch functions.
int gEffDevice = -1;
double gEffM = -1.0;

extern "C" int BoysCudaUploadEffTables(double m, const int* degA, const int* degB) {
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return 2;
    }

    if (gEffDevice == device && gEffM == m)
    {
        return 0;
    }

    if (cudaMemcpyToSymbol(cDegEff, degA, kEffLaneCount * 33 * kMaxPieces * sizeof(int)) !=
        cudaSuccess)
    {
        return 2;
    }

    if (cudaMemcpyToSymbol(cBDegEff, degB, kEffLaneCount * 33 * sizeof(int)) != cudaSuccess)
    {
        return 2;
    }

    // Same ordering guarantee as BoysCudaUploadTables: the effective-degree
    // tables land on the default stream asynchronously, so sync before the
    // (device, m) guard flips — otherwise a concurrent eff kernel could read
    // the zero-initialized constants.
    const cudaError_t uploadSync = cudaStreamSynchronize(static_cast<cudaStream_t>(0));

    if (uploadSync != cudaSuccess)
    {
        return static_cast<int>(uploadSync);
    }

    gEffDevice = device;
    gEffM = m;
    return 0;
}

extern "C" int BoysCudaLaunchSingleF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysSingleF64KernelEff<0>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysBatchF64KernelEff<1>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysSingleF32KernelEff<2>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysBatchF32KernelEff<3>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

#if BoysFp16
extern "C" int BoysCudaLaunchSingleF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysSingleF16KernelEff<4><<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchBatchF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysBatchF16KernelEff<5><<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}
#endif // BoysFp16
} // namespace

} // namespace boysymmetriad
