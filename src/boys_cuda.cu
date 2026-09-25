// CUDA lane of the Boys kernel. Mirrors the scalar/simd design (boys.cpp,
// boys_simd.cpp): piecewise Chebyshev seeds + recurrences, region structure
// A/B/C with the weighted region-A fits (see tools/gen_boys_coefficients.py).
// C++20 with a CUDA-safe include list only: the library's C++23 headers
// would poison the nvcc translation unit.

#include "boys/boys_cuda_arithmetic.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace boys {
namespace {

constexpr int kMaxPieces = 12;
constexpr int kMaxCoeffs = 2304;

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

// The accuracy-multiplier effective-degree tables: one lane per CUDA role,
// in the order the host fills them (boys_cuda.cpp FillEffLane). Each all-orders
// lane serves both entries of its precision — the per-element orders and the
// uniform nmax — so the two read one degree table:
//   0 = double single (BoysSingleF64KernelEff, region-B degree per order)
//   1 = double batch (BoysAllOrdersF64KernelEff/BoysAllNF64KernelEff,
//       region-B degree = order-0 entry)
//   2 = float single (BoysSingleF32KernelEff, region-B degree per order)
//   3 = float batch (BoysAllOrdersF32KernelEff/BoysAllNF32KernelEff,
//       region-B degree = order-0 entry;
//       the region-A seed is the DOUBLE piece table, budget 1.5e-7)
//   4 = fp16 single (BoysSingleF16KernelEff, region-B degree per order)
//   5 = fp16 batch (BoysAllOrdersF16KernelEff/BoysAllNF16KernelEff,
//       region-B degree = order-0 entry;
//       the region-A seed is the DOUBLE piece table, budget 1e-7)
// Filled host-side once per (device, m) — the single-m-per-process cache,
// same idempotence contract as BoysCudaUploadTables. The relaxed kernels
// read these exactly like the kernels above read cDeg/cBDeg (the degree
// shape is unchanged; only the values differ).
constexpr int kEffLaneCount = 6;
__constant__ int cDegEff[kEffLaneCount][33][kMaxPieces];
__constant__ int cBDegEff[kEffLaneCount][33];

// ---------------------------------------------------------------------------
// the flat device image: the tables a caller's own kernel reads
// ---------------------------------------------------------------------------
// A consumer's kernel cannot name this translation unit's __constant__ arrays
// (naming them across translation units takes relocatable device code in the
// consumer's build), so the tables are copied a second time into __device__
// globals whose addresses the host hands out. The piece metadata is the flat
// form of the __constant__ tables — indexed by pieceStart[order] + piece,
// which is the piece-start table the coefficient header already carries, so no
// stride constant has to agree between the library and a consumer. The handle
// a caller passes to a device entry is a value, and kernel parameters live in
// the constant bank, so reading it costs no global memory traffic.
//
// The two lanes' piece tables are separate: the float lane cuts its orders
// differently and needs its own starts, degrees and pool.
constexpr int kPiecesTotal = detail::kPieceStart[detail::kMaxOrder + 1];
constexpr int kPiecesTotal32 = detail::f32::kPieceStart[detail::kMaxOrder + 1];

__device__ int dPieceStart[detail::kMaxOrder + 2];
__device__ int dOffset[kPiecesTotal];
__device__ double dA[kPiecesTotal];
__device__ double dB[kPiecesTotal];
__device__ int dDeg[kPiecesTotal];
__device__ double dCoeffs[kMaxCoeffs];
__device__ double dBSeed[24];

__device__ int dPieceStart32[detail::kMaxOrder + 2];
__device__ int dOffset32[kPiecesTotal32];
__device__ float dA32[kPiecesTotal32];
__device__ float dB32[kPiecesTotal32];
__device__ int dDeg32[kPiecesTotal32];
__device__ float dCoeffs32[kMaxCoeffs];
__device__ float dBSeed32[24];

// ---------------------------------------------------------------------------
// the lanes: what the kernels below hand the shared arithmetic
// ---------------------------------------------------------------------------
// boys_cuda_arithmetic.hpp holds one body per (precision, shape), and it takes
// the tables as a lane object rather than reading a symbol, so the kernels
// here and the device-callable entries a caller's own kernel calls run the
// same code. These six lane objects are the kernels' side of that: the
// __constant__ tables of this translation unit.
//
// The relaxed lanes read their degrees from cDegEff/cBDegEff. Which degree
// table a lane reads, and whether its region-B degree is the per-order entry
// or the order-0 one, is the identity of the lane a kernel names — 0 double
// single, 1 double batch, 2 float single, 3 float batch, 4 fp16 single,
// 5 fp16 batch — and it is the only thing the relaxed lane objects differ in.
// The batch lanes read the order-0 region-B entry because the F0 seed's error
// reaches every output with gain at most 1 + 1.846e-17, and the per-order
// region-A entry at the batch's top order, which is the order the batch
// bodies pass.

struct Lane64Full {
    __device__ __forceinline__ int Count(int order) const {
        return cCount[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return cA[order][piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return cB[order][piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return cCoeffs + cOffset[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDeg[order][piece];
    }

    __device__ __forceinline__ const double* BSeedCoeffs() const {
        return cBcoeffs;
    }

    __device__ __forceinline__ int BSeedDeg(int) const {
        return cBDeg;
    }
};

struct Lane32Full {
    __device__ __forceinline__ int Count(int order) const {
        return cCount32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return cA32[order][piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return cB32[order][piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return cCoeffs32 + cOffset32[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDeg32[order][piece];
    }

    __device__ __forceinline__ const float* BSeedCoeffs() const {
        return cBcoeffs32;
    }

    __device__ __forceinline__ int BSeedDeg(int) const {
        return cBDeg32;
    }
};

// kLane 0 and 1, the double single and the double batch.
template <int kLane> struct Lane64EffSingle {
    __device__ __forceinline__ int Count(int order) const {
        return cCount[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return cA[order][piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return cB[order][piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return cCoeffs + cOffset[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDegEff[kLane][order][piece];
    }

    __device__ __forceinline__ const double* BSeedCoeffs() const {
        return cBcoeffs;
    }

    __device__ __forceinline__ int BSeedDeg(int order) const {
        return cBDegEff[kLane][order];
    }
};

template <int kLane> struct Lane64EffBatch {
    __device__ __forceinline__ int Count(int order) const {
        return cCount[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return cA[order][piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return cB[order][piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return cCoeffs + cOffset[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDegEff[kLane][order][piece];
    }

    __device__ __forceinline__ const double* BSeedCoeffs() const {
        return cBcoeffs;
    }

    // The order-0 entry, whatever order the batch's body passes.
    __device__ __forceinline__ int BSeedDeg(int) const {
        return cBDegEff[kLane][0];
    }
};

// kLane 2 and 3, the float single and the float batch.
template <int kLane> struct Lane32EffSingle {
    __device__ __forceinline__ int Count(int order) const {
        return cCount32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return cA32[order][piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return cB32[order][piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return cCoeffs32 + cOffset32[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDegEff[kLane][order][piece];
    }

    __device__ __forceinline__ const float* BSeedCoeffs() const {
        return cBcoeffs32;
    }

    __device__ __forceinline__ int BSeedDeg(int order) const {
        return cBDegEff[kLane][order];
    }
};

template <int kLane> struct Lane32EffBatch {
    __device__ __forceinline__ int Count(int order) const {
        return cCount32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return cA32[order][piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return cB32[order][piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return cCoeffs32 + cOffset32[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDegEff[kLane][order][piece];
    }

    __device__ __forceinline__ const float* BSeedCoeffs() const {
        return cBcoeffs32;
    }

    // The order-0 entry, whatever order the batch's body passes.
    __device__ __forceinline__ int BSeedDeg(int) const {
        return cBDegEff[kLane][0];
    }
};

// ---------------------------------------------------------------------------
// kernels
// ---------------------------------------------------------------------------
// Each of these is one thread's worth of index arithmetic around one call into
// the shared bodies of boys_cuda_arithmetic.hpp — the same bodies the
// device-callable entries run. The batch kernels and a caller's own fused
// kernel are therefore one piece of arithmetic rather than two that can drift.
//
// __restrict__ on x/out: the buffers are distinct DeviceBuffers by
// construction, and without it every out store would force nvcc to reload x
// (assumed aliasing).
template <bool kFastExp>
__global__ void BoysSingleF32Kernel(const int* n,
                                    const double* __restrict__ x,
                                    float* __restrict__ out,
                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF32<kFastExp>(Lane32Full{}, n[i], static_cast<float>(x[i]));
}

__global__ void BoysAllOrdersF32Kernel(const int* n,
                                       const double* __restrict__ x,
                                       float* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Full{},
                               Lane32Full{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

// The uniform-order entry: one nmax for the whole batch, so the recursion
// bounds are warp-uniform and no order array is read.
__global__ void BoysAllNF32Kernel(int nmax,
                                  const double* __restrict__ x,
                                  float* __restrict__ out,
                                  size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Full{},
                               Lane32Full{},
                               nmax,
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysSingleF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF64(Lane64Full{}, n[i], x[i]);
}

__global__ void BoysAllOrdersF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64Full{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

// The uniform-order entry: one nmax for the whole batch, so the recursion
// bounds are warp-uniform and no order array is read.
__global__ void BoysAllNF64Kernel(int nmax, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64Full{}, nmax, x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

// ---------------------------------------------------------------------------
// fp16 lane (the certified mixed-precision extension; BoysFp16 seam).
// __half I/O around the fp32 engine above -- the same __constant__ tables,
// the same region-partitioned dispatch, the same bodies.
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

    out[i] = __float2half(
        detail::DeviceSingleF32<false>(Lane32Full{}, n[i], __half2float(x[i])));
}

__global__ void BoysAllOrdersF16Kernel(const int* n,
                                       const __half* __restrict__ x,
                                       __half* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Full{},
                               Lane32Full{},
                               n[i],
                               __half2float(x[i]),
                               [&](int l, float v) { out[l * count + i] = __float2half(v); });
}

// The uniform-order entry (see BoysAllNF64Kernel).
__global__ void BoysAllNF16Kernel(int nmax,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ out,
                                  size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Full{},
                               Lane32Full{},
                               nmax,
                               __half2float(x[i]),
                               [&](int l, float v) { out[l * count + i] = __float2half(v); });
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// effective-degree kernels (the relaxation path)
// ---------------------------------------------------------------------------
// One per CUDA lane, mirroring the full-accuracy kernels above exactly
// (same arithmetic, same region structure, same output layout) except that
// the seed degrees come from cDegEff/cBDegEff — the m = 1 kernels stay
// byte-identical (the full-accuracy pin). The lane index is the kernel's
// identity:
//   0 double single, 1 double batch, 2 float single, 3 float batch,
//   4 fp16 single, 5 fp16 batch.
// The batch lanes read the order-0 region-B entry (the F0 seed's error
// reaches every output with gain <= 1 + 1.846e-17) and the per-order
// region-A entry at the batch's top order (the A_A(nmax) amplification
// covers the downward recursion) — exactly the CPU relaxed batches. Which
// lane object a kernel passes is the whole of the difference from the kernels
// above: the relaxed single lanes read their region-B degree per order, and
// the relaxed batch lanes read the order-0 entry whatever order their body
// hands them.
template <int kLane>
__global__ void BoysSingleF64KernelEff(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF64(Lane64EffSingle<kLane>{}, n[i], x[i]);
}

template <int kLane>
__global__ void BoysAllOrdersF64KernelEff(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64EffBatch<kLane>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

// The uniform-order entry reads its lane's degree table exactly as its
// per-element twin does (kDoubleBatch: the order-0 region-B entry and the
// per-order region-A entry at the batch's top order).
template <int kLane>
__global__ void BoysAllNF64KernelEff(int nmax, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64EffBatch<kLane>{}, nmax, x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <int kLane, bool kFastExp>
__global__ void BoysSingleF32KernelEff(const int* n,
                                       const double* __restrict__ x,
                                       float* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF32<kFastExp>(
        Lane32EffSingle<kLane>{}, n[i], static_cast<float>(x[i]));
}

template <int kLane>
__global__ void BoysAllOrdersF32KernelEff(const int* n,
                                          const double* __restrict__ x,
                                          float* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64EffBatch<kLane>{},
                               Lane32EffBatch<kLane>{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

template <int kLane>
__global__ void BoysAllNF32KernelEff(int nmax,
                                     const double* __restrict__ x,
                                     float* __restrict__ out,
                                     size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64EffBatch<kLane>{},
                               Lane32EffBatch<kLane>{},
                               nmax,
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
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

    out[i] = __float2half(
        detail::DeviceSingleF32<false>(Lane32EffSingle<kLane>{}, n[i], __half2float(x[i])));
}

template <int kLane>
__global__ void BoysAllOrdersF16KernelEff(const int* n,
                                          const __half* __restrict__ x,
                                          __half* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64EffBatch<kLane>{},
                               Lane32EffBatch<kLane>{},
                               n[i],
                               __half2float(x[i]),
                               [&](int l, float v) { out[l * count + i] = __float2half(v); });
}

template <int kLane>
__global__ void BoysAllNF16KernelEff(int nmax,
                                     const __half* __restrict__ x,
                                     __half* __restrict__ out,
                                     size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64EffBatch<kLane>{},
                               Lane32EffBatch<kLane>{},
                               nmax,
                               __half2float(x[i]),
                               [&](int l, float v) { out[l * count + i] = __float2half(v); });
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
        // The flat image of the same tables, for the device-callable entries.
        int flatStart[detail::kMaxOrder + 2] = {};
        int flatOffset[kPiecesTotal] = {};
        double flatA[kPiecesTotal] = {};
        double flatB[kPiecesTotal] = {};
        int flatDeg[kPiecesTotal] = {};
        int runningOffset = 0;

        for (int o = 0; o <= detail::kMaxOrder + 1; ++o)
        {
            flatStart[o] = detail::kPieceStart[o];
        }

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
                flatA[p] = piece.a;
                flatB[p] = piece.b;
                flatDeg[p] = piece.deg;
                flatOffset[p] = runningOffset;

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

        // The flat image the device-callable entries read.
        if (cudaMemcpyToSymbol(dPieceStart, flatStart, sizeof(flatStart)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dOffset, flatOffset, sizeof(flatOffset)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dA, flatA, sizeof(flatA)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dB, flatB, sizeof(flatB)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dDeg, flatDeg, sizeof(flatDeg)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dCoeffs, coeffs, sizeof(coeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dBSeed, detail::kBcoeffs.data(), sizeof(detail::kBcoeffs)) !=
            cudaSuccess)
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
        int flatStart[detail::kMaxOrder + 2] = {};
        int flatOffset[kPiecesTotal32] = {};
        float flatA[kPiecesTotal32] = {};
        float flatB[kPiecesTotal32] = {};
        int flatDeg[kPiecesTotal32] = {};
        int runningOffset = 0;

        for (int o = 0; o <= detail::kMaxOrder + 1; ++o)
        {
            flatStart[o] = detail::f32::kPieceStart[o];
        }

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
                flatA[p] = piece.a;
                flatB[p] = piece.b;
                flatDeg[p] = piece.deg;
                flatOffset[p] = runningOffset;

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

        if (cudaMemcpyToSymbol(dPieceStart32, flatStart, sizeof(flatStart)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dOffset32, flatOffset, sizeof(flatOffset)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dA32, flatA, sizeof(flatA)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dB32, flatB, sizeof(flatB)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dDeg32, flatDeg, sizeof(flatDeg)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dCoeffs32, coeffs, sizeof(coeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dBSeed32,
                               detail::f32::kBcoeffs.data(),
                               sizeof(detail::f32::kBcoeffs)) != cudaSuccess)
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

// The addresses of the flat device image, in the order the status layer fills
// BoysDeviceTables (boys_cuda.hpp): the double lane's pieceStart, offset, a, b,
// deg, coeffs and region-B seed, then the float lane's seven. The status layer
// cannot name a CUDA symbol, so the order is the seam between this file and
// boys_cuda.cpp and both sides state it. Uploads first, so a caller that
// skipped InitializeTables still gets a table rather than a null pointer.
extern "C" int BoysCudaDeviceTableAddresses(void** out) {
    const int uploaded = BoysCudaUploadTables();

    if (uploaded != 0)
    {
        return uploaded;
    }

    // The element type is const void*, not void**: cudaGetSymbolAddress has a
    // template overload taking const T&, and a void** lvalue binds to it as
    // T = void**, which hands the runtime the address of the array slot instead
    // of the address of the variable. A const void* value matches the
    // non-template overload, which is the one that resolves a symbol.
    const void* const symbols[] = {&dPieceStart, &dOffset,     &dA,         &dB,
                                  &dDeg,        &dCoeffs,     &dBSeed,     &dPieceStart32,
                                  &dOffset32,   &dA32,        &dB32,       &dDeg32,
                                  &dCoeffs32,   &dBSeed32};

    for (int i = 0; i < 14; ++i)
    {
        const cudaError_t got = cudaGetSymbolAddress(&out[i], symbols[i]);

        if (got != cudaSuccess)
        {
            return 2;
        }
    }

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
    BoysSingleF32Kernel<false>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF32Fast(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysSingleF32Kernel<true>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF32(
    int nmax, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllNF32Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        nmax, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysSingleF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF64(
    int nmax, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllNF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        nmax, x, out, count);
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

extern "C" int BoysCudaLaunchAllOrdersF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysAllOrdersF16Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF16(
    int nmax, const void* x, void* out, std::size_t count, void* stream) {
    BoysAllNF16Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        nmax, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}
#endif // BoysFp16

// Accuracy-multiplier effective-degree upload. The host layer (boys_cuda.cpp,
// C++23) computes the per-lane degree tables from the
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

extern "C" int BoysCudaLaunchAllOrdersF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64KernelEff<1>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF64Eff(
    int nmax, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllNF64KernelEff<1>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysSingleF32KernelEff<2, false>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchSingleF32EffFast(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysSingleF32KernelEff<2, true>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32KernelEff<3>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF32Eff(
    int nmax, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllNF32KernelEff<3>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

#if BoysFp16
extern "C" int BoysCudaLaunchSingleF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysSingleF16KernelEff<4><<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream) {
    BoysAllOrdersF16KernelEff<5><<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF16Eff(
    int nmax, const void* x, void* out, std::size_t count, void* stream) {
    BoysAllNF16KernelEff<5><<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        nmax, static_cast<const __half*>(x), static_cast<__half*>(out), count);
    return static_cast<int>(cudaGetLastError());
}
#endif // BoysFp16
} // namespace

} // namespace boys
