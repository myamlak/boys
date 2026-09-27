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

// The relaxed degrees, in the same flat form and one set for the whole device:
// one rung is resident at a time, so a table per rung is not what a caller can
// have. The stride is the larger of the two lanes' piece counts, so one lane
// axis serves both piece tables; a lane's own entries are indexed with the
// piece-start table its region-A seed reads its coefficients with, exactly as
// the full-accuracy table above is.
//
// The lane order is cDegEff's: 0 double single, 1 double batch, 2 float single,
// 3 float batch, 4 fp16 single, 5 fp16 batch. Lanes 0, 1, 3 and 5 are indexed by
// the double piece table (the batch lanes' region-A seed is computed in double
// whatever precision they return — RoleUsesDoubleTables) and lanes 2 and 4 by
// the float one.
constexpr int kRelaxedPieces = kPiecesTotal > kPiecesTotal32 ? kPiecesTotal : kPiecesTotal32;

__device__ int dDegEff[kEffLaneCount * kRelaxedPieces];
__device__ int dBDegEff[kEffLaneCount * (detail::kMaxOrder + 1)];
// The rung the two arrays above are cut for, zero when none has been uploaded.
// It is read through the handle a consumer holds rather than copied into it, so
// a handle filled for one rung reports itself as retired when another rung
// replaces it instead of running tables that have since been overwritten.
__device__ double dRelaxedM;

// ---------------------------------------------------------------------------
// the narrow partition, on this lane
// ---------------------------------------------------------------------------
// The second cut of the double lane's fits: region A's pieces are cut per
// order, and region B's seed is 5 pieces at degree 10 rather than one
// polynomial over the interval. A lane reading it therefore supplies the seed
// at the argument rather than the coefficients of a fixed shape, which is what
// the lane interface says a lane supplies.
//
// It is the double lane's partition, so there is no float copy of it here: the
// effective degrees a rung cuts it to are derived for the roles that evaluate
// these fits (boys_effective_degrees.hpp), and the entries that carry it are
// the double ones.
//
// __device__ and not __constant__: the narrow region-A pool is 3421
// coefficients against the shipped lane's 1876, and the two do not fit the
// 64 KB constant bank beside the float lane's tables. One fetch per piece per
// thread is the pattern the flat image above already serves from global
// memory.
constexpr int kNarrowPiecesTotal = detail::kNarrowAPieceStart[detail::kMaxOrder + 1];
constexpr int kNarrowCoeffsTotal = kNarrowPiecesTotal * (detail::kNarrowADeg + 1);

__device__ int dNarrowAPieceStart[detail::kMaxOrder + 2];
__device__ int dNarrowAOffset[kNarrowPiecesTotal];
__device__ double dNarrowAA[kNarrowPiecesTotal];
__device__ double dNarrowAB[kNarrowPiecesTotal];
__device__ int dNarrowAStoredDeg[kNarrowPiecesTotal];
__device__ double dNarrowACoeffs[kNarrowCoeffsTotal];
__device__ double dNarrowBEdges[detail::kNarrowBPieces + 1];
__device__ double dNarrowBCoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)];

// The rung's cut of the same partition, laid out as the shipped lanes' relaxed
// tables are: region A flat over the partition's rows, region B flat over
// (piece, order), matching boys_effective_degrees.hpp's derivation. A rung's
// table is one role's — the double batch, the shape the entries carrying the
// partition have — so it is one table and not a lane axis.
//
// They are written only by a relaxed upload, so a full-accuracy call reads the
// stored degrees above instead and is not served whatever rung is resident.
__device__ int dNarrowDegEff[kNarrowPiecesTotal];
__device__ int dNarrowBDegEff[detail::kNarrowBPieces * (detail::kMaxOrder + 1)];

// ---------------------------------------------------------------------------
// the monomial scheme's stored form, on this lane
// ---------------------------------------------------------------------------
// The same fits in the other basis: every piece table above is carried in both
// forms (boys_coefficients.hpp), so this family stores no second fit and needs
// no second partition — it is the pools and the summation, and the pieces,
// edges, counts and stored degrees are the Chebyshev lanes' own.
//
// All of it is global rather than constant memory. The double lane's Chebyshev
// pool and the float lane's tables already fill most of the 64 KB constant
// bank, so a second double pool of the same size does not fit beside them; the
// narrow pool is out of the bank for the same reason the Chebyshev narrow one
// is. One fetch per piece per thread is the pattern the flat image above
// serves that way.
__device__ double dMonoCoeffs[kMaxCoeffs];
__device__ double dMonoBcoeffs[24];
__device__ double dNarrowAMonoCoeffs[kNarrowCoeffsTotal];
__device__ double dNarrowBMonoCoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)];

// The rung's cut, in the monomial basis: the same derivations as the Chebyshev
// tables above at the other form of the same table (TailBasis::kMonomial), so
// a rung's degrees are the ones that rung certifies for the pool and the
// summation this family reads. Region A is order-major as cDegEff is, since the
// cut is derived per order and piece; region B is one row per order, read at
// the order-0 entry by the batch shape.
//
// The narrow partition's cut is one table each for the same reason its
// Chebyshev counterpart is: the entries carrying the partition are the double
// batch, so there is no lane axis to carry.
__device__ int dMonoDegEff[detail::kMaxOrder + 1][kMaxPieces];
__device__ int dMonoBDegEff[detail::kMaxOrder + 1];
__device__ int dNarrowMonoDegEff[kNarrowPiecesTotal];
__device__ int dNarrowMonoBDegEff[detail::kNarrowBPieces * (detail::kMaxOrder + 1)];

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

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceClenshawSplit(cBcoeffs, cBDeg, t);
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

    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) / static_cast<float>(detail::kX1 - detail::kX0) - 1.0f;
        return detail::DeviceClenshawSplit32(cBcoeffs32, cBDeg32, t);
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

    __device__ __forceinline__ double BSeed(double x, int order) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceClenshawSplit(cBcoeffs, cBDegEff[kLane][order], t);
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

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceClenshawSplit(cBcoeffs, cBDegEff[kLane][0], t);
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

    __device__ __forceinline__ float BSeed(float x, int order) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) / static_cast<float>(detail::kX1 - detail::kX0) - 1.0f;
        return detail::DeviceClenshawSplit32(cBcoeffs32, cBDegEff[kLane][order], t);
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

    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) / static_cast<float>(detail::kX1 - detail::kX0) - 1.0f;
        return detail::DeviceClenshawSplit32(cBcoeffs32, cBDegEff[kLane][0], t);
    }
};

// The narrow partition's lane, in the two forms a rung has: kRelaxed false
// reads the degrees the partition was stored at, which no rung's upload
// touches, and true reads the rung's own cut. The distinction is the shipped
// lanes' cDeg/cDegEff one, and it is what keeps a full-accuracy call from being
// served a resident rung's table.
//
// Region B's seed is piecewise here and the batch shape reads it at order 0, so
// BSeed ignores the order it is passed for the same reason Lane64EffBatch does.
template <bool kRelaxed> struct Lane64Narrow {
    __device__ __forceinline__ int Count(int order) const {
        return dNarrowAPieceStart[order + 1] - dNarrowAPieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return dNarrowAA[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return dNarrowAB[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return dNarrowACoeffs + dNarrowAOffset[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        const int flat = dNarrowAPieceStart[order] + piece;

        if constexpr (kRelaxed)
        {
            return dNarrowDegEff[flat];
        } else
        {
            return dNarrowAStoredDeg[flat];
        }
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        int piece = 0;

        while (piece + 1 < detail::kNarrowBPieces && x >= dNarrowBEdges[piece + 1])
        {
            ++piece;
        }

        const double a = dNarrowBEdges[piece];
        const double b = dNarrowBEdges[piece + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;

        return detail::DeviceClenshawSplit(dNarrowBCoeffs + piece * (detail::kNarrowBDeg + 1),
                                   kRelaxed ? dNarrowBDegEff[piece * (detail::kMaxOrder + 1)]
                                            : detail::kNarrowBDeg,
                                   t);
    }
};

// The monomial scheme's lanes: the shipped lanes' pieces and degrees with the
// coefficients read from the monomial pool and the piece summed by Horner. The
// kMonomial member is the whole of the difference the shared bodies see — they
// choose the summation with it (boys_cuda_arithmetic.hpp) and read the degrees
// the lane hands them either way, so a rung of this family is a cut of this
// family's table and not the Chebyshev one's.
struct Lane64MonoFull {
    static constexpr bool kMonomial = true;

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
        return dMonoCoeffs + cOffset[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return cDeg[order][piece];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceHornerMono(dMonoBcoeffs, cBDeg, t);
    }
};

// The rung's form: the cut's degrees in place of the stored ones, at the
// order-0 region-B entry the batch shape reads.
struct Lane64MonoEff {
    static constexpr bool kMonomial = true;

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
        return dMonoCoeffs + cOffset[order][piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return dMonoDegEff[order][piece];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceHornerMono(dMonoBcoeffs, dMonoBDegEff[0], t);
    }
};

// The narrow partition in the monomial basis, in the two forms a rung has, as
// Lane64Narrow carries the Chebyshev one: kRelaxed false reads the degrees the
// partition was stored at and true the rung's own cut of this basis. Region B
// is piecewise here and read at order 0, for the reason Lane64Narrow gives.
template <bool kRelaxed> struct Lane64NarrowMono {
    static constexpr bool kMonomial = true;

    __device__ __forceinline__ int Count(int order) const {
        return dNarrowAPieceStart[order + 1] - dNarrowAPieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return dNarrowAA[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return dNarrowAB[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return dNarrowAMonoCoeffs + dNarrowAOffset[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        const int flat = dNarrowAPieceStart[order] + piece;

        if constexpr (kRelaxed)
        {
            return dNarrowMonoDegEff[flat];
        } else
        {
            return dNarrowAStoredDeg[flat];
        }
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        int piece = 0;

        while (piece + 1 < detail::kNarrowBPieces && x >= dNarrowBEdges[piece + 1])
        {
            ++piece;
        }

        const double a = dNarrowBEdges[piece];
        const double b = dNarrowBEdges[piece + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;

        return detail::DeviceHornerMono(
            dNarrowBMonoCoeffs + piece * (detail::kNarrowBDeg + 1),
            kRelaxed ? dNarrowMonoBDegEff[piece * (detail::kMaxOrder + 1)]
                     : detail::kNarrowBDeg,
            t);
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

// ---------------------------------------------------------------------------
// the two shapes the lane gained: the orders axis, and the narrow partition
// ---------------------------------------------------------------------------
// Six kernels and not two, because each shape has a full-accuracy form and a
// rung's form, exactly as the all-orders kernel above does. The orders axis is
// a body choice and the partition is a lane choice, so the two compose: the
// last pair is the narrow partition read with the orders axis, and it is the
// combination a caller asking for both gets rather than a third arithmetic.
//
// The orders axis is a choice inside region A only: past kX0 these kernels run
// the certified all-orders body, whose row is the lane's own bound over the
// whole range, so the axis adds a claim in region A and takes none away
// outside it.
template <typename Lane>
__device__ __forceinline__ void DeviceOrdersBody(
    const Lane& lane, int order, double xx, double* out, size_t count, size_t i) {
    detail::DeviceOrdersF64(lane, order, xx, [&](int l, double v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF64OrdersKernel(const int* n,
                                             const double* __restrict__ x,
                                             double* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64Full{}, n[i], x[i], out, count, i);
}

template <int kLane>
__global__ void BoysAllOrdersF64OrdersKernelEff(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64EffBatch<kLane>{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowKernel(const int* n,
                                             const double* __restrict__ x,
                                             double* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64Narrow<false>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowKernelEff(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64Narrow<true>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowOrdersKernel(const int* n,
                                                   const double* __restrict__ x,
                                                   double* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64Narrow<false>{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowOrdersKernelEff(const int* n,
                                                      const double* __restrict__ x,
                                                      double* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64Narrow<true>{}, n[i], x[i], out, count, i);
}

// ---------------------------------------------------------------------------
// the evaluation scheme: the same fits in the monomial basis
// ---------------------------------------------------------------------------
// The scheme axis is a lane choice and not a body choice — the pieces, the
// edges, the region structure and the shape are the shipped ones, and what
// changes is the pool a piece's coefficients come from and the summation they
// are read by (boys_cuda_arithmetic.hpp) — so these are the same four shapes
// with a monomial lane, each with the full-accuracy form and the rung's.
//
// The orders axis composes with it for the same reason it composes with the
// partition: DeviceOrdersBody takes the lane, and the axis is a choice inside
// region A whichever basis that lane sums.
__global__ void BoysAllOrdersF64MonoKernel(const int* n,
                                           const double* __restrict__ x,
                                           double* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64MonoFull{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64MonoKernelEff(const int* n,
                                              const double* __restrict__ x,
                                              double* __restrict__ out,
                                              size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64MonoEff{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64OrdersMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 double* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64MonoFull{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64OrdersMonoKernelEff(const int* n,
                                                    const double* __restrict__ x,
                                                    double* __restrict__ out,
                                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64MonoEff{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 double* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64NarrowMono<false>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowMonoKernelEff(const int* n,
                                                    const double* __restrict__ x,
                                                    double* __restrict__ out,
                                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64NarrowMono<true>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowOrdersMonoKernel(const int* n,
                                                       const double* __restrict__ x,
                                                       double* __restrict__ out,
                                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64NarrowMono<false>{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowOrdersMonoKernelEff(const int* n,
                                                          const double* __restrict__ x,
                                                          double* __restrict__ out,
                                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64NarrowMono<true>{}, n[i], x[i], out, count, i);
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
        // The monomial scheme's pool, packed at the same offsets as the
        // Chebyshev one — the two forms of one fit over the same pieces.
        double monoCoeffs[kMaxCoeffs] = {};
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
                    monoCoeffs[runningOffset + k] = detail::kMonoCoeffs[piece.offset + k];
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

        // The monomial scheme's pools. Only the pool is copied: the pieces,
        // their edges and their degrees are the ones above, which is what the
        // two forms of a fit share.
        if (cudaMemcpyToSymbol(dMonoCoeffs, monoCoeffs, sizeof(monoCoeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dMonoBcoeffs, detail::kMonoBcoeffs.data(),
                               sizeof(detail::kMonoBcoeffs)) != cudaSuccess)
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

    // The narrow partition of the double lane's fits. One pool of coefficients
    // and one row per piece of the partition, laid out as the header stores it:
    // a row's flat index is its position in kNarrowAPieces, which is also the
    // index its effective degree is derived at.
    {
        int start[detail::kMaxOrder + 2] = {};
        int offset[kNarrowPiecesTotal] = {};
        double a[kNarrowPiecesTotal] = {};
        double b[kNarrowPiecesTotal] = {};
        int stored[kNarrowPiecesTotal] = {};
        double coeffs[kNarrowCoeffsTotal] = {};
        // The partition's own fits in the monomial basis, at the same offsets:
        // the pieces, edges and stored degrees above are the two forms' shared
        // ones.
        double monoCoeffs[kNarrowCoeffsTotal] = {};
        double edges[detail::kNarrowBPieces + 1] = {};
        double bcoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)] = {};
        double monoBcoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)] = {};

        for (int o = 0; o <= detail::kMaxOrder + 1; ++o)
        {
            start[o] = detail::kNarrowAPieceStart[o];
        }

        for (int p = 0; p < kNarrowPiecesTotal; ++p)
        {
            const detail::OrderPiece& piece = detail::kNarrowAPieces[static_cast<std::size_t>(p)];

            if (piece.offset < 0 || piece.offset + piece.deg >= kNarrowCoeffsTotal)
            {
                return 1;
            }

            offset[p] = piece.offset;
            a[p] = piece.a;
            b[p] = piece.b;
            stored[p] = piece.deg;

            for (int k = 0; k <= piece.deg; ++k)
            {
                coeffs[piece.offset + k] = detail::kNarrowACoeffs[static_cast<std::size_t>(
                    piece.offset + k)];
                monoCoeffs[piece.offset + k] =
                    detail::kNarrowAMonoCoeffs[static_cast<std::size_t>(piece.offset + k)];
            }
        }

        for (int p = 0; p <= detail::kNarrowBPieces; ++p)
        {
            edges[p] = detail::kNarrowBEdges[static_cast<std::size_t>(p)];
        }

        for (int k = 0; k < detail::kNarrowBPieces * (detail::kNarrowBDeg + 1); ++k)
        {
            bcoeffs[k] = detail::kNarrowBcoeffs[static_cast<std::size_t>(k)];
            monoBcoeffs[k] = detail::kNarrowBMonoCoeffs[static_cast<std::size_t>(k)];
        }

        if (cudaMemcpyToSymbol(dNarrowAPieceStart, start, sizeof(start)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAOffset, offset, sizeof(offset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAA, a, sizeof(a)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAB, b, sizeof(b)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAStoredDeg, stored, sizeof(stored)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowACoeffs, coeffs, sizeof(coeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAMonoCoeffs, monoCoeffs, sizeof(monoCoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBEdges, edges, sizeof(edges)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBCoeffs, bcoeffs, sizeof(bcoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBMonoCoeffs, monoBcoeffs, sizeof(monoBcoeffs)) != cudaSuccess)
        {
            return 2;
        }
    }

    gTablesDevice = device;
    return 0;
}

// The addresses of the flat device image, in the order the status layer fills
// BoysDeviceTables (boys_cuda.hpp): the double lane's pieceStart, offset, a, b,
// deg, coeffs and region-B seed, then the float lane's seven, then the relaxed
// image's three. The status layer cannot name a CUDA symbol, so the order is
// the seam between this file and boys_cuda.cpp and both sides state it.
// Uploads first, so a caller that skipped InitializeTables still gets a table
// rather than a null pointer.
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
    const void* const symbols[] = {&dPieceStart, &dOffset,     &dA,       &dB,
                                  &dDeg,        &dCoeffs,     &dBSeed,   &dPieceStart32,
                                  &dOffset32,   &dA32,        &dB32,     &dDeg32,
                                  &dCoeffs32,   &dBSeed32,    &dRelaxedM, &dDegEff,
                                  &dBDegEff};

    for (int i = 0; i < 17; ++i)
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

// Whether this multiplier's tables are resident on this device, given the
// record of what was last uploaded and where. The record carries the device as
// well as the multiplier because cDegEff and cBDegEff are per-device copies: a
// comparison on the multiplier alone would answer for a device that has never
// held them, and a device no upload has reached reads zero-initialized
// constants. The device arrives as an argument rather than from a call so that
// every row of the comparison is exercisable on one card, the row that matters
// included — record naming device 0, caller asking about device 1, same
// multiplier, which must answer not resident.
extern "C" int BoysCudaEffTablesResidentOn(
    int device, double m, int recordedDevice, double recordedM) {
    return (device == recordedDevice && m == recordedM) ? 1 : 0;
}

// The same question for the device the calling thread is on. A query that
// cannot name a device answers not resident, which costs an upload rather than
// returning a wrong answer.
extern "C" int BoysCudaEffTablesResident(double m) {
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return 0;
    }

    return BoysCudaEffTablesResidentOn(device, m, gEffDevice, gEffM);
}

// The monomial scheme's rungs travel with the Chebyshev ones and not in an
// upload of their own: one rung is one cut of every table the lane holds, and a
// caller that changed the multiplier must not be able to reach a device holding
// the new shipped tables beside the old monomial ones. monoA layout:
// [order][pieceInOrder] ((kMaxOrder + 1) x kMaxPieces); monoB layout: [order]
// (kMaxOrder + 1); narrowMonoA: flat over the narrow partition's rows;
// narrowMonoB: [piece][order], as the Chebyshev narrow region-B table is.
extern "C" int BoysCudaUploadEffTables(double m,
                                       const int* degA,
                                       const int* degB,
                                       const int* narrowA,
                                       const int* narrowB,
                                       const int* monoA,
                                       const int* monoB,
                                       const int* narrowMonoA,
                                       const int* narrowMonoB) {
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        return 2;
    }

    if (BoysCudaEffTablesResidentOn(device, m, gEffDevice, gEffM) == 1)
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

    // The flat image the device entries read (dDegEff/dBDegEff), cut from the
    // same tables the constant arrays just took: a lane's piece p of order n is
    // the flat entry pieceStart[order] + p of that lane, which is the indexing
    // the handle's own piece-start pointer gives a consumer. The host hands the
    // degrees in the order-major layout the batch kernels read, so this is the
    // one place the two layouts meet.
    static int flatA[kEffLaneCount * kRelaxedPieces];
    static int flatB[kEffLaneCount * 33];
    const int* const lanePieceStart[kEffLaneCount] = {detail::kPieceStart.data(),
                                                      detail::kPieceStart.data(),
                                                      detail::f32::kPieceStart.data(),
                                                      detail::kPieceStart.data(),
                                                      detail::f32::kPieceStart.data(),
                                                      detail::kPieceStart.data()};

    for (int lane = 0; lane < kEffLaneCount; ++lane)
    {
        const int* const pieceStart = lanePieceStart[lane];

        for (int order = 0; order <= detail::kMaxOrder; ++order)
        {
            for (int p = pieceStart[order]; p < pieceStart[order + 1]; ++p)
            {
                flatA[lane * kRelaxedPieces + p] =
                    degA[(lane * 33 + order) * kMaxPieces + (p - pieceStart[order])];
            }

            flatB[lane * 33 + order] = degB[lane * 33 + order];
        }
    }

    if (cudaMemcpyToSymbol(dDegEff, flatA, sizeof(flatA)) != cudaSuccess)
    {
        return 2;
    }

    if (cudaMemcpyToSymbol(dBDegEff, flatB, sizeof(flatB)) != cudaSuccess)
    {
        return 2;
    }

    // The narrow partition's cut for the same rung. It is the double batch
    // role's table alone — the role the entries carrying the partition have —
    // and no lane axis, so it lands as one table per rung.
    if (cudaMemcpyToSymbol(dNarrowDegEff, narrowA, kNarrowPiecesTotal * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowBDegEff,
                           narrowB,
                           detail::kNarrowBPieces * (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess)
    {
        return 2;
    }

    // The monomial scheme's cut of the same rung, in the same four shapes.
    if (cudaMemcpyToSymbol(dMonoDegEff,
                           monoA,
                           (detail::kMaxOrder + 1) * kMaxPieces * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dMonoBDegEff, monoB, (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowMonoDegEff, narrowMonoA, kNarrowPiecesTotal * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowMonoBDegEff,
                           narrowMonoB,
                           detail::kNarrowBPieces * (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess)
    {
        return 2;
    }

    // The rung goes last, after a sync: an entry that reads a rung it can serve
    // must not be able to observe the tables of the rung before it.
    const cudaError_t tableSync = cudaStreamSynchronize(static_cast<cudaStream_t>(0));

    if (tableSync != cudaSuccess)
    {
        return static_cast<int>(tableSync);
    }

    if (cudaMemcpyToSymbol(dRelaxedM, &m, sizeof(m)) != cudaSuccess)
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

extern "C" int BoysCudaLaunchAllOrdersF64Orders(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersKernelEff<1>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64Narrow(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrders(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The monomial scheme's four shapes, each in the two forms a rung has.
extern "C" int BoysCudaLaunchAllOrdersF64Mono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64MonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64MonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64MonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersMonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersMonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowMonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowMonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersMonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersMonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
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
