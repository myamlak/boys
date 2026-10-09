#pragma once

/// \file
/// Boys evaluation inside the caller's own kernel: F_0..F_n for an argument
/// the caller computed in a register, with no round trip through global
/// memory.
///
/// **What it is for.** A production GPU integral kernel forms x per thread — one x
/// per shell-quartet per primitive pair — and needs F_0..F_n for that thread's own x,
/// usually inside the same kernel that formed it. The batch entries of this lane
/// evaluate an array of arguments instead, so a caller has to materialise every x to
/// global memory, launch a second kernel, synchronise and read the results back — two
/// extra memory passes per integral batch. The entries here take one (order, x) per
/// call and return the values to the calling thread, so the fused kernel needs
/// neither the pass out nor the pass back.
///
/// **How a caller obtains it, and what it costs the build.** The tables are handed
/// over at run time as a value, BoysDeviceTables, filled host-side by
/// BoysCuda::DeviceTables. The entries are header-defined device functions with no
/// device-side symbol to link: including this header is the whole of what the
/// caller's build pays. No relocatable device code (\c -rdc=true), no device link
/// step, no library on the link line, no additional compilation flag — a kernel that
/// calls an entry compiles with the same nvcc command line as one that does not — and
/// the arithmetic is inlined into the calling kernel, so there is no call overhead per
/// order. What it does cost is the compile of the coefficient tables this header pulls
/// in (boys_coefficients.hpp) and the instructions the compiler emits at each call
/// site, which is the same arithmetic the batch kernels run.
///
/// **Precision.** The double entries are the primary surface: a fused integral kernel
/// is a double-precision computation, and the fp64 bound is what decides whether it
/// can use the lane at all. The float entries follow — on consumer hardware fp32 runs
/// many times faster, and the fused kernel is where that matters most — and the
/// half-format entries, fp16 and bf16, each with its own store, mirror the half batch
/// lane for a caller packing many orders into registers. Every entry reads the same
/// handle: there is one handle type and one upload, not one per precision. The float
/// single entry carries one option the others do not: its region-B exponential is a
/// certified choice of arithmetic (boys::RegionBExp), named as a template argument.
///
/// **Order range, and what the entries do when they cannot serve a request.** Orders
/// run 0..kMaxBoysOrder (32). Two shapes are offered per precision: the
/// runtime-top-order entry takes the order per call, which is what a fused kernel
/// needs (the order follows the shell quartets) and takes a capacity — the number of
/// values the caller's array holds — so that an order the array cannot receive is
/// reported rather than written past the end; the compile-time-top-order entry takes
/// the order as a template argument, where it bounds the recursion loops and the
/// compiler can unroll them.
///
/// Every failure is a BoysDeviceStatus the caller branches on, and a call that fails
/// writes nothing: no truncated ladder, no partial fill. An order outside the range is
/// \c kOrderOutOfRange; an order whose values do not fit the caller's array is
/// \c kCapacityTooSmall; a handle that carries no tables is \c kTablesNotReady. The
/// checks are ordered tables, then order, then capacity: a request that is malformed is
/// reported as malformed, so a caller branching on these statuses always has one thing
/// to fix.
///
/// **Cost in registers.** A caller keeping the whole ladder live holds order + 1
/// doubles — 33 of them at the top order, which is the real cost of this interface and
/// the reason the entries write into the caller's array rather than into an internal
/// one: the caller's own liveness decides what stays in registers, and a caller that
/// consumes each order as it arrives can hold none. BoysDeviceEachOrderF64 and its
/// siblings are that shape: they hand one value at a time to a caller-supplied sink and
/// keep no ladder at all.
///
/// **The handle as a kernel argument.** Pass it by value into the kernel — it
/// is plain data, and kernel parameters live in the constant bank — and declare
/// that parameter \c __grid_constant__ so that taking its address inside the
/// kernel does not copy it to local memory per thread:
/// \code
/// __global__ void Fused(__grid_constant__ const boys::BoysDeviceTables tables,
///                       const double* rho, const double* d2, ...)
/// \endcode
/// A caller on a toolchain without \c __grid_constant__ passes the same value
/// without the qualifier and reads the handle through a generic pointer, which
/// is correct and slower; the arithmetic is identical either way.
///
/// **Accuracy.** Every entry answers with the arithmetic its tables were stored at,
/// which is the arithmetic the corresponding batch entry documents: the bound below is
/// the lane's own figure and not a multiple of it.
///
/// **The bound.** Every entry holds the lane's documented bound for its precision: the
/// same bound the corresponding batch entry documents, because it is the same
/// arithmetic. The device gate measures these entries against the committed
/// high-precision reference grid and reports the worst ratio in the same vocabulary as
/// every other lane.
///
/// \ingroup boys

#include "boys/boys_cuda_arithmetic.hpp"
#include "boys/boys_device_tables.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace boys {

/// Result status of the device-callable entries.
///
/// Device code cannot throw and cannot report an error out of band, so every refusal
/// is one of these values, returned to the calling thread. A call that does not return
/// \c kSuccess wrote nothing: the caller's array is untouched and no value is in
/// flight. \c kSuccess is 0.
enum class BoysDeviceStatus : int {
    kSuccess = 0, ///< the values were written
    /// The handle carries no tables: BoysCuda::DeviceTables was never called
    /// for this device, or it failed and the caller ignored the status.
    kTablesNotReady,
    /// The order is outside 0..kMaxBoysOrder. Nothing was written.
    kOrderOutOfRange,
    /// The caller's array holds fewer than order + 1 values. Nothing was
    /// written.
    kCapacityTooSmall,
};

/// \cond
namespace detail {

// A lane's degrees, resolved once per call: regionA/regionB are the handle's tables, and the lane
// split and stored degrees are BoysDeviceLane's (boys_device_tables.hpp). The single entries read
// the seed's own degree, not the order-0 entry the batch lanes read, so stride is 0: the degree is
// a scalar in the handle here rather than a table (boys_cuda_arithmetic.hpp).
struct Degrees {
    const int* regionA;
    const int* regionB;
    int stride;
};

// The handle as a lane object — the same seven members the batch kernels' lane objects carry
// (boys_cuda_arithmetic.hpp), reading the caller's handle instead of a __constant__ symbol. The
// piece index of order n, piece p is pieceStart[n] + p, so no stride constant has to agree between
// this header and the tables the library uploaded.
struct TableLane64 {
    const BoysDeviceTables* tables;
    Degrees deg;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart[order + 1] - tables->pieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return tables->pieceA[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return tables->pieceB[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return tables->coeffs + tables->pieceOffset[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return deg.regionA[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ double BSeed(double x, int order) const {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        return DeviceClenshawSplit(tables->bSeedCoeffs, deg.regionB[order * deg.stride], t);
    }
};

struct TableLane32 {
    const BoysDeviceTables* tables;
    Degrees deg;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart32[order + 1] - tables->pieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return tables->pieceA32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return tables->pieceB32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return tables->coeffs32 + tables->pieceOffset32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return deg.regionA[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ float BSeed(float x, int order) const {
        const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
        return DeviceClenshawSplit32(tables->bSeedCoeffs32, deg.regionB[order * deg.stride], t);
    }
};

// The double lane's monomial reading of the coarsest partition: the same pieces, edges,
// degrees and piece index base as the lane above, region A's pool from the monomial table, the
// piece and region B's seed summed by Horner, not by the split Clenshaw; kMonomial is all the
// shared bodies see (boys_cuda_arithmetic.hpp), two pools over one cut, no second piece table.
struct TableLane64Mono {
    const BoysDeviceTables* tables;
    Degrees deg;

    static constexpr bool kMonomial = true;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart[order + 1] - tables->pieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return tables->pieceA[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return tables->pieceB[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return tables->monoCoeffs + tables->pieceOffset[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return deg.regionA[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ double BSeed(double x, int order) const {
        const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
        return DeviceHornerMono(tables->monoBSeedCoeffs, deg.regionB[order * deg.stride], t);
    }
};

// The same form one lane down, in the float lane's own arithmetic, as TableLane32 is the
// lane above one lane down. The two pools are that lane's, and the summation is the float
// helper's: the double helper would read a float table through a double pointer and sum a
// double polynomial.
struct TableLane32Mono {
    const BoysDeviceTables* tables;
    Degrees deg;

    static constexpr bool kMonomial = true;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart32[order + 1] - tables->pieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return tables->pieceA32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return tables->pieceB32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return tables->monoCoeffs32 + tables->pieceOffset32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return deg.regionA[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ float BSeed(float x, int order) const {
        const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
        return DeviceHornerMono32(tables->monoBSeedCoeffs32, deg.regionB[order * deg.stride], t);
    }
};

// The two refusals every entry shares. The table test is one pointer, and the
// difference between a reported status and a null dereference; the order test is what
// makes an out-of-range request a value rather than a read outside the piece tables.
__device__ __forceinline__ BoysDeviceStatus DeviceReady(const BoysDeviceTables& tables) {
    return tables.pieceStart == nullptr ? BoysDeviceStatus::kTablesNotReady
                                        : BoysDeviceStatus::kSuccess;
}

// The same test for the uniform route, on the pointers its bodies actually read: the grid is a
// table of its own, and testing the piece table would report a readiness the read below does not
// rest on. The two per-interval tables are part of the test too - a body reads a degree and a block
// start from them first, so a test passing with either missing answers a status this route cannot.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatReady(const BoysDeviceTables& tables) {
    return tables.flatCoeffs == nullptr || tables.flatDegs == nullptr ||
                   tables.flatOffsets == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

// The float lane's own test, on its own lane's pointers: the two lanes' tables are
// derived separately and a handle may carry one without the other, so a float entry
// testing the double lane's pointers would report a readiness its own read does not
// rest on — and would dereference a null one it never tested.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatReady32(const BoysDeviceTables& tables) {
    return tables.flatCoeffs32 == nullptr || tables.flatDegs32 == nullptr ||
                   tables.flatOffsets32 == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

// The same test for the grid's rational route, which reads five pointers and not three: the pool
// plus four per-interval columns. A body reads both degrees and the block's start before a
// coefficient, so a pool alone is read at another interval's length. The count is a column, not a
// grid constant: one more named than the monomial twin's.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatRatReady(const BoysDeviceTables& tables) {
    return tables.flatRatCoeffs == nullptr || tables.flatRatNumDeg == nullptr ||
                   tables.flatRatDenDeg == nullptr || tables.flatRatStored == nullptr ||
                   tables.flatRatOffsets == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

/// The float lane's own test on the same route, on its own lane's pointers, for
/// the reason DeviceFlatReady32 gives against DeviceFlatReady.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatRatReady32(
    const BoysDeviceTables& tables) {
    return tables.flatRatCoeffs32 == nullptr || tables.flatRatNumDeg32 == nullptr ||
                   tables.flatRatDenDeg32 == nullptr || tables.flatRatStored32 == nullptr ||
                   tables.flatRatOffsets32 == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

// The coarsest partition's monomial reading, and the two tables it rests on that the Chebyshev
// reading does not touch: region A's pool and region B's seed. A handle may carry the piece tables
// without them - one filled before this form's slots existed is the case that happens on its own -
// and the piece table alone reports a readiness the read does not rest on, handing null to Horner.
__device__ __forceinline__ BoysDeviceStatus DeviceMonoReady(const BoysDeviceTables& tables) {
    return tables.pieceStart == nullptr || tables.monoCoeffs == nullptr ||
                   tables.monoBSeedCoeffs == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

// The same test for the entries that run the float lane, on the tables that reading reads:
// the float family's seed lane is the double lane's pieces, so region A's pool is the double
// one and region B's seed is the float one. The test above would refuse a handle it could
// serve and pass one whose float seed is missing.
__device__ __forceinline__ BoysDeviceStatus DeviceMonoReady32(const BoysDeviceTables& tables) {
    return tables.pieceStart == nullptr || tables.monoCoeffs == nullptr ||
                   tables.monoBSeedCoeffs32 == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

__device__ __forceinline__ bool DeviceOrderValid(int order) {
    return order >= 0 && order <= kMaxBoysOrder;
}

// Which lane's degrees a call reads, resolved once per call: the double entries and the two batch
// entries of the narrow precisions seed region A from the double piece table, the single entries
// from the float one - the split the batch kernels' lane objects make (boys_cuda.cu). It is what
// makes the piece index below the calling lane's own. Stride 0 is the read: one scalar, not a table.
template <BoysDeviceLane kLane>
__device__ __forceinline__ Degrees DeviceStoredDegrees(const BoysDeviceTables& tables) {
    constexpr bool kDoublePieces =
        kLane != BoysDeviceLane::kF32Single && kLane != BoysDeviceLane::kF16Single;

    Degrees out;
    out.regionA = kDoublePieces ? tables.pieceDeg : tables.pieceDeg32;
    out.regionB =
        kDoublePieces ? static_cast<const int*>(&tables.bSeedDeg) : &tables.bSeedDeg32;
    out.stride = 0;
    return out;
}

// The narrow partition as a lane object, one per stored form of region A: pieces cut per order,
// edges and degree through the handle's piece-start table; the form is the pool and its summation -
// same pieces, same degrees. Region B's seed is piecewise, so the lane finds its piece and reads
// its order-0 degree, as the batch kernels' narrow lanes do (boys_cuda.cu): that seed is F_0's fit.
template <bool kMono>
struct NarrowLane64 {
    const BoysDeviceTables* tables;

    static constexpr bool kMonomial = kMono;

    __device__ __forceinline__ int Count(int order) const {
        return tables->narrowPieceStart[order + 1] - tables->narrowPieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return tables->narrowPieceA[tables->narrowPieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return tables->narrowPieceB[tables->narrowPieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        const double* const pool =
            kMono ? tables->narrowMonoCoeffs : tables->narrowCoeffs;
        return pool + tables->narrowPieceOffset[tables->narrowPieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return tables->narrowStoredDeg[tables->narrowPieceStart[order] + piece];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        int piece = 0;

        while (piece + 1 < kNarrowBPieces && x >= tables->narrowBEdges[piece + 1])
        {
            ++piece;
        }

        const double a = tables->narrowBEdges[piece];
        const double b = tables->narrowBEdges[piece + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;
        const double* const c = kMono ? tables->narrowBMonoCoeffs : tables->narrowBCoeffs;
        const int degree = kNarrowBDeg;

        // The form of the lane selects the summation here as it does in region A:
        // the two forms are one fit stored twice, and summing either one with the
        // other's reader would read the half of the table that is not there.
        if constexpr (kMono)
        {
            return DeviceHornerMono(c + piece * (degree + 1), degree, t);
        } else
        {
            return DeviceClenshawSplit(c + piece * (degree + 1), degree, t);
        }
    }
};

// The same partition one lane down, over the float lane's own pieces and its own
// piecewise seed. Region A is the double lane's above, which is the one seed lane
// the float entries seed from.
template <bool kMono>
struct NarrowLane32 {
    const BoysDeviceTables* tables;

    static constexpr bool kMonomial = kMono;

    __device__ __forceinline__ int Count(int order) const {
        return tables->narrowPieceStart32[order + 1] - tables->narrowPieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return tables->narrowPieceA32[tables->narrowPieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return tables->narrowPieceB32[tables->narrowPieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        const float* const pool =
            kMono ? tables->narrowMonoCoeffs32 : tables->narrowCoeffs32;
        return pool + tables->narrowPieceOffset32[tables->narrowPieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return tables->narrowStoredDeg32[tables->narrowPieceStart32[order] + piece];
    }

    __device__ __forceinline__ float BSeed(float x, int) const {
        int piece = 0;

        while (piece + 1 < f32::kNarrowBPiecesF32 && x >= tables->narrowBEdges32[piece + 1])
        {
            ++piece;
        }

        const float a = tables->narrowBEdges32[piece];
        const float b = tables->narrowBEdges32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const float* const c =
            kMono ? tables->narrowBMonoCoeffs32 : tables->narrowBCoeffs32;
        const int degree = f32::kNarrowBDegF32;
        return kMono ? DeviceHornerMono32(c + piece * (degree + 1), degree, t)
                         : DeviceClenshawSplit32(c + piece * (degree + 1), degree, t);
    }
};

// The fit route as a lane object, one per partition, at the reading its entry makes. Its pieces
// are the partition's own - the coarsest piece tables for one, the narrow ones for the other - and
// what it adds is the pair: the numerator's block and the denominator's, the two degrees, and the
// seed the region-B recursion starts from.
template <bool kNarrow>
struct RatLane64 {
    const BoysDeviceTables* tables;

    static constexpr bool kRational = true;

    __device__ __forceinline__ int Count(int order) const {
        return kNarrow ? tables->narrowPieceStart[order + 1] - tables->narrowPieceStart[order]
                       : tables->pieceStart[order + 1] - tables->pieceStart[order];
    }

    __device__ __forceinline__ int Flat(int order, int piece) const {
        return (kNarrow ? tables->narrowPieceStart[order] : tables->pieceStart[order]) + piece;
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return kNarrow ? tables->narrowPieceA[Flat(order, piece)]
                       : tables->pieceA[Flat(order, piece)];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return kNarrow ? tables->narrowPieceB[Flat(order, piece)]
                       : tables->pieceB[Flat(order, piece)];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        const double* const pool =
            kNarrow ? tables->narrowRatCoeffs : tables->ratCoeffs;
        const int* const offset = kNarrow ? tables->narrowRatOffset : tables->ratOffset;

        return pool + offset[Flat(order, piece)];
    }

    __device__ __forceinline__ const double* DenCoeffs(int order, int piece) const {
        const double* const pool =
            kNarrow ? tables->narrowRatCoeffs : tables->ratCoeffs;
        const int* const offset =
            kNarrow ? tables->narrowRatDenOffset : tables->ratDenOffset;

        return pool + offset[Flat(order, piece)];
    }

    __device__ __forceinline__ int NumDeg(int order, int piece) const {
        return (kNarrow ? tables->narrowRatNumDeg : tables->ratNumDeg)[Flat(order, piece)];
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        return (kNarrow ? tables->narrowRatDenDeg : tables->ratDenDeg)[Flat(order, piece)];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        if constexpr (!kNarrow)
        {
            const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
            const int numDeg = kRatBnumDeg;
            const int denDeg = kRatBdenDeg;

            return DeviceRatSum(tables->ratBNum, numDeg, tables->ratBDen, denDeg, t);
        } else
        {
            int piece = 0;

            while (piece + 1 < kNarrowBPieces && x >= tables->narrowBEdges[piece + 1])
            {
                ++piece;
            }

            const double a = tables->narrowBEdges[piece];
            const double b = tables->narrowBEdges[piece + 1];
            const double t = 2.0 * (x - a) / (b - a) - 1.0;
            const int* const stored = tables->narrowRatBStoredNumDeg;
            const int* const denDeg = tables->narrowRatBDenDeg;
            const double* const c = tables->narrowRatBCoeffs +
                                    tables->narrowRatBOffset[piece];
            const int numDeg = stored[piece];
            const int den = denDeg[piece];

            return DeviceRatSum(c, numDeg, c + stored[piece] + 1, den, t);
        }
    }
};

// The float lane's region B on the same two partitions. The pair is the lane's
// own and region A is the double lane's above, so this lane supplies the seed
// and nothing else, exactly as the float batch kernels' region-B lanes do.
template <bool kNarrow>
struct RatLane32 {
    const BoysDeviceTables* tables;

    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(kX0)) /
                            static_cast<float>(kX1 - kX0) -
                        1.0f;

        if constexpr (!kNarrow)
        {
            const int numDeg = f32::kRatBnumDeg;
            const int denDeg = f32::kRatBdenDeg;

            return DeviceRatSum32(tables->ratBNum32, numDeg, tables->ratBDen32, denDeg, t);
        } else
        {
            int piece = 0;

            while (piece + 1 < f32::kNarrowRatBPiecesCountF32 &&
                   x >= tables->narrowBEdges32[piece + 1])
            {
                ++piece;
            }

            const float a = tables->narrowBEdges32[piece];
            const float b = tables->narrowBEdges32[piece + 1];
            const float u = 2.0f * (x - a) / (b - a) - 1.0f;
            const float* const c = tables->narrowRatBCoeffs32 +
                                   tables->narrowRatBOffset32[piece];
            const int storedNum = tables->narrowRatBStoredNumDeg32[piece];
            const int numDeg = storedNum;
            const int denDeg = tables->narrowRatBDenDeg32[piece];

            return DeviceRatSum32(c, numDeg, c + storedNum + 1, denDeg, u);
        }
    }
};

// The readiness test for a partition or a route whose geometry is not the coarsest
// piece table: the same one-pointer test as DeviceReady, on a pointer the entry's
// own body reads.
__device__ __forceinline__ BoysDeviceStatus DeviceGroupReady(const void* first) {
    return first == nullptr ? BoysDeviceStatus::kTablesNotReady
                            : BoysDeviceStatus::kSuccess;
}

// The same test for an entry that reads two lanes of one partition: the float entries of the
// partitions below seed region A from the double lane's tables and sum region B from their own, so
// a handle carrying one lane's tables without the other's has the call read what is not there; the
// test is on both pointers, not on whichever happens to be tested first.
__device__ __forceinline__ BoysDeviceStatus DeviceGroupReady2(const void* first,
                                                              const void* second) {
    return first != nullptr && second != nullptr ? BoysDeviceStatus::kSuccess
                                                 : BoysDeviceStatus::kTablesNotReady;
}

// The three checks every ladder-shaped entry makes before it touches a table, in
// the order the entries document: the tables it reads, the order, then the
// capacity of the caller's array.
__device__ __forceinline__ BoysDeviceStatus DeviceLadderRequest(
    const void* firstTable, int order, int capacity) {
    const BoysDeviceStatus ready = DeviceGroupReady(firstTable);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    return capacity < order + 1 ? BoysDeviceStatus::kCapacityTooSmall
                                : BoysDeviceStatus::kSuccess;
}

} // namespace detail
/// \endcond

/// F_n(x) in double precision, inside the caller's kernel.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so which member
///         a call names is part of what places it: the figure is the lane's, and
///         \c BoysLaneContracts states the term this axis adds to it, if it adds
///         one.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x); untouched unless kSuccess is returned
///
/// \pre \c x >= 0, as for the batch entries. A negative argument names a
///      different function than the one the tables fit, and is not checked: the
///      arithmetic's domain is the caller's to keep, exactly as the batch
///      entries' is.
/// \pre \c out is a device-writable location for one double.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady when \c tables
/// carries no tables; kOrderOutOfRange when \c order is outside
/// 0..kMaxBoysOrder. A refused call writes nothing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceSingleF64(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Single>(tables);

    *out = detail::DeviceSingleF64<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg}, order, x);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, inside the caller's kernel.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so which member
///         a call names is part of what places it: the figure is the lane's, and
///         \c BoysLaneContracts states the term this axis adds to it, if it adds
///         one.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive doubles
/// \param capacity   the number of doubles \c out holds
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   double* out,
                                                   int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg}, order, x,
                                      [&](int l, double v) {
                                          out[l] = v;
                                      });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision from the coarsest partition's monomial
/// form, inside the caller's kernel.
///
/// The same ladder BoysDeviceAllOrdersF64 runs, over the other stored form of the
/// same fit: the pieces, their edges and their degrees are the ones that entry
/// reads, and this entry reads the monomial pool beside them and sums a piece by
/// Horner rather than by the split Clenshaw. Contract, bound and refusals are
/// BoysDeviceAllOrdersF64's, the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Mono(const BoysDeviceTables& tables,
                                                       int order,
                                                       double x,
                                                       double* out,
                                                       int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64Mono{&tables, deg},
        order,
        x,
        [&](int l, double v) {
            out[l] = v;
        });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in double precision, with the top order fixed where the call
/// site names it.
///
/// The order is a template argument, so the recursion bounds are compile-time
/// constants: the loops can be unrolled and no order has to be validated. For a caller
/// whose fused kernel works at one order — or at a handful it instantiates separately —
/// this is the entry to use, and it is the shape a caller can wrap in its own dispatch
/// where the order is a run-time value.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so which member
///         a call names is part of what places it: the figure is the lane's, and
///         \c BoysLaneContracts states the term this axis adds to it, if it adds
///         one.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   doubles
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady when
/// \c tables carries no tables. There is no order or capacity check: the top
/// order is compiled in and \c out is the caller's declaration.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, int kTopOrder, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllNF64(const BoysDeviceTables& tables,
                                              double x,
                                              double* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg}, kTopOrder, x,
                                      [&](int l, double v) {
                                          out[l] = v;
                                      });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, one value at a time into a caller's sink.
///
/// The shape to use when the caller consumes each order as it arrives — an integral
/// kernel contracting F_l with a coefficient does exactly that — and therefore has no
/// reason to hold the ladder: nothing is stored inside this call. The sink is called
/// once per order, in the order the recursion produces it, which is the top order
/// downwards inside region A and 0 upwards in regions B and C.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam Sink a callable taking (int order, double value), callable from
///         device code. Passing it by value is deliberate: a lambda capturing
///         the caller's accumulators stays in registers.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so which member
///         a call names is part of what places it: the figure is the lane's, and
///         \c BoysLaneContracts states the term this axis adds to it, if it adds
///         one.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c sink is device-callable and its own work does not fault.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, in which case \c sink is
/// not called at all.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, typename Sink, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceEachOrderF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg}, order, x, sink);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the uniform grid, inside the
/// caller's kernel.
///
/// The partition the CPU lane names \c FitGranularity::kUniform: one table of equal
/// intervals over [0, kFlatHi) rather than pieces cut where the function needs them.
/// Below the join every order is summed from its own block and none is built from
/// another, which is why every block is stored at one degree. Above kFlatHi the call
/// falls to the asymptotic every other route ends in, so the figure a caller places
/// this entry by is the double batch lane's bound.
///
/// The arithmetic is the launched row's kernel's (boys_cuda.cu,
/// BoysAllOrdersF64FlatKernel): the two differ in where the coefficients come from
/// and not in what is done with them.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the number of values \c out holds
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Uniform(const BoysDeviceTables& tables,
                                                          int order,
                                                          double x,
                                                          double* out,
                                                          int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF64Flat<kForm, false>(tables.flatCoeffs, tables.flatMonoCoeffs,
                                                 tables.flatDegs, tables.flatOffsets, order, x,
                                                 [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) from the uniform grid's other stored form, inside the caller's
/// kernel.
///
/// The same table and the same degrees, read by Horner in the monomial basis rather
/// than by the split Clenshaw in the Chebyshev one. The two blocks are two rows of one
/// table, so this entry and the one above cannot disagree about which interval an
/// argument falls in, where the table stops or what its degree is. Contract, bound and
/// refusals are BoysDeviceAllOrdersF64Uniform's, its own scheme aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the number of values \c out holds
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64UniformHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF64Flat<kForm, true>(tables.flatCoeffs, tables.flatMonoCoeffs,
                                                tables.flatDegs, tables.flatOffsets, order, x,
                                                [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) from the uniform grid's RATIONAL member, inside the caller's
/// kernel.
///
/// The double lane's member over the same grid the two entries above read: one
/// numerator/denominator pair per interval, summed by DeviceRatSum. The route is
/// a family and not a basis, so there is one stored form and the Horner entry
/// below is a forwarder to this one rather than a second arithmetic.
///
/// The pair is one per interval and its stored count is the interval's own, so
/// the read takes four per-interval columns beside the pool, which
/// DeviceFlatRatReady tests before any coefficient is touched.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the caller's out capacity, which must be >= order + 1
/// \return kSuccess, or a refusal naming the argument that was not servable
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64UniformRat(const BoysDeviceTables& tables,
                                                             int order,
                                                             double x,
                                                             double* out,
                                                             int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatRatReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF64FlatRat<kForm>(tables.flatRatCoeffs, tables.flatRatNumDeg,
                                             tables.flatRatDenDeg, tables.flatRatStored,
                                             tables.flatRatOffsets, order, x,
                                             [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its float counterpart
/// gives: the rational member is stored in one form, so both scheme names reach
/// one arithmetic.
///
/// \copydoc boys::BoysDeviceAllOrdersF64UniformRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64UniformRatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    return BoysDeviceAllOrdersF64UniformRat<kForm>(tables, order, x, out, capacity);
}

/// F_n(x) in single precision, inside the caller's kernel.
///
/// The float lane's bound is the one the f32 batch entries document; this
/// entry is the same arithmetic, and it takes the lane's own choice of
/// region-B exponential.
///
/// \tparam kExp  the region-B exponential to evaluate; the default is
///         \c kDefaultRegionBExp, the lane's documented default
///         (\c RegionBExp::kAccurate), which is the tighter bound of the two and
///         the name the batch entry of the same precision defaults to. A caller
///         that selects RegionBExp::kFast takes that option's bound, which is
///         the lane's plus the corrected seed's own contribution. The choice is a
///         template argument and not a run-time one because it selects an
///         arithmetic inside the caller's own kernel rather than branching within
///         one: the exponential is evaluated once per call, outside the order
///         loop, so the option not selected is absent from the caller's kernel
///         instead of merely untaken. Both options of a given template argument
///         are one binary; a caller that needs both instantiates both.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x); untouched unless kSuccess is returned
///
/// \pre \c x >= 0; \c out is a device-writable location for one float.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady, kOrderOutOfRange or
/// otherwise, without writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceSingleF32(const BoysDeviceTables& tables,
                                                int order,
                                                float x,
                                                float* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Single>(tables);

    *out = detail::DeviceSingleF32<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane32{&tables, deg},
        order,
        x);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in single precision, inside the caller's kernel.
///
/// The region-A seed is computed in double precision, as the f32 batch entry
/// documents: the downward recursion amplifies a float seed error beyond the
/// lane's float budget.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive floats
/// \param capacity   the number of floats \c out holds
///
/// \pre \c x >= 0; \c order + 1 <= \c capacity, or the call is refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32(const BoysDeviceTables& tables,
                                                   int order,
                                                   float x,
                                                   float* out,
                                                   int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    // One resolution for both lane objects: the batch lane's region-A degree is
    // indexed by the double piece table and its region-B degree by the float
    // one, which is what the two lane objects read it with. Only the seed lane
    // reads a degree here; the returned degrees come from the other.
    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      x,
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in single precision from the coarsest partition's monomial
/// form, inside the caller's kernel.
///
/// The same ladder BoysDeviceAllOrdersF32 runs, over the other stored form of the
/// same fit: the pieces, their edges and their degrees are the ones that entry
/// reads, this entry reads the monomial pools beside them and sums a piece by
/// Horner rather than by the split Clenshaw, and the two lanes it hands the body
/// are the monomial reading of that entry's two. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32's, the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Mono(const BoysDeviceTables& tables,
                                                       int order,
                                                       float x,
                                                       float* out,
                                                       int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64Mono{&tables, deg},
        detail::TableLane32Mono{&tables, deg},
        order,
        x,
        [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in single precision, with the top order fixed where
/// the call site names it (see BoysDeviceAllNF64).
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   floats
///
/// \pre \c x >= 0.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady
/// otherwise.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, int kTopOrder, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllNF32(const BoysDeviceTables& tables,
                                              float x,
                                              float* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      kTopOrder,
                                      x,
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in single precision, one value at a time into a caller's
/// sink (see BoysDeviceEachOrderF64 for the arrival order and the reason the
/// shape exists).
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam Sink a callable taking (int order, float value), callable from
///         device code.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x >= 0; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, with \c sink not called.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, typename Sink, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceEachOrderF32(const BoysDeviceTables& tables,
                                                   int order,
                                                   float x,
                                                   Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      x,
                                      sink);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in single precision from the float lane's uniform grid,
/// inside the caller's kernel.
///
/// The float lane's own grid and not a re-cut of the double lane's: 245 intervals at
/// degree 4 against the double lane's 245 at degree 8, fitted over the same
/// [0, kFlatHi). It is the lane's table throughout — region A's seed is the grid's own
/// block rather than the double piece table the float lane's piecewise entries seed
/// from, and every sum below the join is a float one — so the bound is the float lane's
/// own.
///
/// The interval's edges are the correctly rounded float quotients iv/7 and (iv + 1)/7,
/// which is the grid the fits were measured on. Above kFlatHi the call falls to the
/// same asymptotic the other float entries end in.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the number of values \c out holds
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Uniform(const BoysDeviceTables& tables,
                                                          int order,
                                                          float x,
                                                          float* out,
                                                          int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, false>(tables.flatCoeffs32, tables.flatMonoCoeffs32,
                                                 tables.flatDegs32, tables.flatOffsets32, order, x,
                                                 [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) from the float lane's grid in its other stored form, inside
/// the caller's kernel.
///
/// The same table read by Horner rather than by the split Clenshaw, as
/// BoysDeviceAllOrdersF64UniformHorner is the double lane's. Contract, bound
/// and refusals are BoysDeviceAllOrdersF32Uniform's, its own scheme aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the number of values \c out holds
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32UniformHorner(
    const BoysDeviceTables& tables,
    int order,
    float x,
    float* out,
    int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, true>(tables.flatCoeffs32, tables.flatMonoCoeffs32,
                                                tables.flatDegs32, tables.flatOffsets32, order, x,
                                                [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) from the uniform grid's RATIONAL member, inside the caller's
/// kernel.
///
/// One numerator/denominator pair per interval of the same grid the two entries
/// above read, summed by the two Horner sums and the division the lane's coarsest
/// and narrow rational routes are summed by (DeviceRatSum32). The route is a
/// family and not a basis, so there is one stored form and the Horner entry
/// below is a forwarder to this one rather than a second arithmetic.
///
/// The pair is one per interval and its stored count is the interval's own, so
/// the read takes four per-interval columns beside the pool; a handle missing
/// any of them is refused before a coefficient is touched, which is what
/// DeviceFlatRatReady32 tests. Everything above the grid's join is the same
/// one-term asymptotic and the same upward recurrence every uniform entry runs,
/// so the bound this entry carries over the whole of x >= 0 is that row's.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the caller's out capacity, which must be >= order + 1
/// \return kSuccess, or a refusal naming the argument that was not servable
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32UniformRat(const BoysDeviceTables& tables,
                                                             int order,
                                                             float x,
                                                             float* out,
                                                             int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatRatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32FlatRat<kForm>(tables.flatRatCoeffs32, tables.flatRatNumDeg32,
                                             tables.flatRatDenDeg32, tables.flatRatStored32,
                                             tables.flatRatOffsets32, order, x,
                                             [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body: the rational member is stored in one form,
/// so both scheme names a caller may use reach one arithmetic - the same relation
/// the lane's coarsest and narrow rational pairs stand in. Contract, bound and
/// refusals are BoysDeviceAllOrdersF32UniformRat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF32UniformRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32UniformRatHorner(
    const BoysDeviceTables& tables,
    int order,
    float x,
    float* out,
    int capacity) {
    return BoysDeviceAllOrdersF32UniformRat<kForm>(tables, order, x, out, capacity);
}

#if BoysFp16
// The half lane's four shapes, in the caller's own kernel. The half lane is two stores over one
// engine, and each body below answers in one of them: a format's entry is written out beside the
// other's rather than folded into it, so the two are the same lane, the same tables and the same
// arithmetic - the store and its half digit the whole of the difference a reader looks for.

/// F_n(x) in fp16, inside the caller's kernel.
///
/// The fp16 lane rounds at the boundary and runs the fp32 engine in between, so
/// this entry takes and returns the raw IEEE-754 binary16 bit patterns (the
/// __half type), as the fp16 batch entries take and return this library's F16.
/// The bound is the fp16 lane's.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x) in fp16; untouched unless kSuccess is returned
///
/// \pre \c x is a binary16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __half.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady, kOrderOutOfRange or
/// otherwise, without writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceSingleF16(const BoysDeviceTables& tables,
                                                int order,
                                                __half x,
                                                __half* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Single>(tables);

    *out = __float2half(detail::DeviceSingleF32<kForm, false>(
        detail::TableLane32{&tables, deg}, order, __half2float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_n(x) in bfloat16, inside the caller's kernel.
///
/// The bfloat16 lane rounds at the boundary and runs the fp32 engine in between, so
/// this entry takes and returns the raw IEEE-754 bfloat16 bit patterns (the
/// __nv_bfloat16 type), as the bf16 batch entries take and return this library's Bf16.
/// The bound is the bf16 lane's.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x) in bfloat16; untouched unless kSuccess is returned
///
/// \pre \c x is a bfloat16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __nv_bfloat16.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady, kOrderOutOfRange or
/// otherwise, without writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceSingleBf16(const BoysDeviceTables& tables,
                                                int order,
                                                __nv_bfloat16 x,
                                                __nv_bfloat16* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Single>(tables);

    *out = __float2bfloat16(detail::DeviceSingleF32<kForm, false>(
        detail::TableLane32{&tables, deg}, order, __bfloat162float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, inside the caller's kernel.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive __half
/// \param capacity   the number of __half values \c out holds
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __half x,
                                                   __half* out,
                                                   int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16 from the coarsest partition's monomial form, inside the
/// caller's kernel.
///
/// BoysDeviceAllOrdersF16's body over the other stored form of the same fit: the
/// pieces, their edges and their degrees are the ones that entry reads, this entry
/// reads the monomial pools beside them and sums a piece by Horner rather than by the
/// split Clenshaw, and the value crosses the boundary as it does there. Contract,
/// bound and refusals are that entry's, the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16Mono(const BoysDeviceTables& tables,
                                                       int order,
                                                       __half x,
                                                       __half* out,
                                                       int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64Mono{&tables, deg},
        detail::TableLane32Mono{&tables, deg},
        order,
        __half2float(x),
        [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in bfloat16 from the coarsest partition's monomial form, inside the
/// caller's kernel.
///
/// BoysDeviceAllOrdersBf16's body over the other stored form of the same fit: the
/// pieces, their edges and their degrees are the ones that entry reads, this entry
/// reads the monomial pools beside them and sums a piece by Horner rather than by the
/// split Clenshaw, and the value crosses the boundary as it does there. Contract,
/// bound and refusals are that entry's, the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16Mono(const BoysDeviceTables& tables,
                                                       int order,
                                                       __nv_bfloat16 x,
                                                       __nv_bfloat16* out,
                                                       int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64Mono{&tables, deg},
        detail::TableLane32Mono{&tables, deg},
        order,
        __bfloat162float(x),
        [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in bfloat16, inside the caller's kernel.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive __nv_bfloat16
/// \param capacity   the number of __nv_bfloat16 values \c out holds
///
/// \pre \c x is a bfloat16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __nv_bfloat16 x,
                                                   __nv_bfloat16* out,
                                                   int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in fp16, with the top order fixed where the call site
/// names it (see BoysDeviceAllNF64).
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   __half values
///
/// \pre \c x is a binary16 value.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady
/// otherwise.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, int kTopOrder, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllNF16(const BoysDeviceTables& tables,
                                              __half x,
                                              __half* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      kTopOrder,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in bfloat16, with the top order fixed where the call site
/// names it (see BoysDeviceAllNF64).
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   __nv_bfloat16 values
///
/// \pre \c x is a bfloat16 value.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady
/// otherwise.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, int kTopOrder, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllNBf16(const BoysDeviceTables& tables,
                                              __nv_bfloat16 x,
                                              __nv_bfloat16* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      kTopOrder,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, one value at a time into a caller's sink (see
/// BoysDeviceEachOrderF64 for the arrival order and the reason the shape
/// exists).
///
/// \tparam Sink a callable taking (int order, __half value), callable from
///         device code.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x is a binary16 value; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, with \c sink not called.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, typename Sink, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceEachOrderF16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __half x,
                                                   Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { sink(l, __float2half(v)); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in bfloat16, one value at a time into a caller's sink (see
/// BoysDeviceEachOrderF64 for the arrival order and the reason the shape
/// exists).
///
/// \tparam Sink a callable taking (int order, __nv_bfloat16 value), callable from
///         device code.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \tparam kExp which region-B exponential the ladder's seed evaluates; the
///         default is \c kDefaultRegionBExp, the lane's documented default. The
///         member named is part of the combination the call runs, so the figure it
///         is owed is the lane's bound plus whatever that member adds to it
///         (BoysAccuracyGuaranteed, boys/boys.hpp).
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x is a bfloat16 value; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, with \c sink not called.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, typename Sink, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceEachOrderBf16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __nv_bfloat16 x,
                                                   Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                      detail::TableLane32{&tables, deg},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { sink(l, __float2bfloat16(v)); });
    return BoysDeviceStatus::kSuccess;
}
#endif // BoysFp16

// The narrow partition and the fit route, in the caller's own kernel: the axes the launched group
// carries beyond the coarsest partition and the Chebyshev scheme, in the shapes this header offers.
// Each entry is the same ladder as its coarsest-partition sibling, differing in the tables its lane
// reads (boys_cuda_arithmetic.hpp), and no entry resolves a second reading of those tables.

/// F_0(x)..F_n(x) in double precision from the narrow partition, inside the
/// caller's kernel.
///
/// The partition the CPU lane names \c FitGranularity::kNarrow: the pieces are cut per
/// order rather than shared, so region A's fit of order \c n is the one cut for that
/// order. Region B's seed is piecewise on this partition, over the edges the handle
/// carries.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Narrow(const BoysDeviceTables& tables,
                                                         int order,
                                                         double x,
                                                         double* out,
                                                         int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables}, order, x,
                                      [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the narrow partition, summed in the
/// monomial basis, inside the caller's kernel.
///
/// The same partition and the same pieces as BoysDeviceAllOrdersF64Narrow, with
/// each piece's fit read from the monomial form of the pool and summed by
/// Horner. The two forms are one fit stored twice at the same offsets and the
/// same degree, so the two entries differ in the summation and in nothing else —
/// and an entry that read the other form's coefficients would sum the half of
/// the table that fit was not stored in.
///
/// The bound is the one BoysDeviceAllOrdersF64Narrow states: the two forms are
/// two summations of one fit, and the lane's bound is over the route.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowMono(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables}, order, x,
                                      [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route, inside the caller's
/// kernel.
///
/// The route the CPU lane names \c FitRoute::kRational: the same pieces as the coarsest
/// partition, each stored as a numerator and a denominator and read as the quotient of
/// the two sums. The division is the arithmetic's own — one per piece — and it is the
/// whole of what separates this route from the Chebyshev one, whose fits are
/// polynomials.
///
/// The route's region-A fit is read at the degrees its table stores, and region
/// B's seed at the degrees its own table stores.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Rat(const BoysDeviceTables& tables,
                                                      int order,
                                                      double x,
                                                      double* out,
                                                      int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables}, order, x,
                                      [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route, named at the Horner
/// scheme.
///
/// One arithmetic under two names, which is what the launched group's two rows for this
/// route are: the route's pair is stored once, in one basis, and both scheme names
/// select it. This entry is BoysDeviceAllOrdersF64Rat and answers exactly as it does.
///
/// \copydoc boys::BoysDeviceAllOrdersF64Rat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64RatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    return BoysDeviceAllOrdersF64Rat<kForm, kExp>(tables, order, x, out, capacity);
}

/// F_0(x)..F_n(x) in double precision from the fit route on the narrow
/// partition, inside the caller's kernel.
///
/// The two axes together: the narrow partition's per-order pieces, stored as numerator
/// and denominator pairs, with region B's piecewise seed stored as a pair as well. Both
/// halves of the route are read at the degrees their tables store, as
/// BoysDeviceAllOrdersF64Rat does. The bound is the double batch lane's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowRat(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF64<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables}, order, x,
                                      [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route on the narrow
/// partition, named at the Horner scheme.
///
/// One arithmetic under two names, as BoysDeviceAllOrdersF64RatHorner is: this
/// entry is BoysDeviceAllOrdersF64NarrowRat and answers exactly as it does.
///
/// \copydoc boys::BoysDeviceAllOrdersF64NarrowRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowRatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity) {
    return BoysDeviceAllOrdersF64NarrowRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// F_0(x)..F_n(x) in float precision from the narrow partition, inside the
/// caller's kernel.
///
/// The float lane's region-B seed is the float lane's own piecewise fit, and
/// region A is the double lane's narrow pieces: the float lane's downward
/// recursion amplifies a float seed error past the float budget, so the seed is
/// computed in double and rounded once on entry to the recursion, exactly as the
/// launched row of this partition does.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Narrow(const BoysDeviceTables& tables,
                                                         int order,
                                                         double x,
                                                         float* out,
                                                         int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    // Both halves read the same seed of the same partition: region A's is the double
    // lane's narrow pieces and region B's is this lane's own piecewise fit over them.
    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                      detail::NarrowLane32<false>{&tables},
                                      order,
                                      static_cast<float>(x),
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the narrow partition, summed in the
/// monomial basis, inside the caller's kernel.
///
/// The same partition and the same pieces as BoysDeviceAllOrdersF32Narrow, with
/// both region A and the float lane's region-B seed read from the monomial form
/// of their pools and summed by Horner.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowMono(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    // The same pair of halves the Chebyshev entry above resolves, at this form
    // of the same seed: the double lane's pieces for region A and this lane's
    // own monomial seed for region B.
    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                      detail::NarrowLane32<true>{&tables},
                                      order,
                                      static_cast<float>(x),
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route, inside the caller's
/// kernel.
///
/// The route's pair on the coarsest partition, with region A seeded from the
/// double lane's pairs and region B from the float lane's own pair — the split
/// the launched row makes, and the reason a float seed is never amplified.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Rat(const BoysDeviceTables& tables,
                                                      int order,
                                                      double x,
                                                      float* out,
                                                      int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    // Both halves read the stored fit: region A's seed is the double lane's pair and
    // region B's is this lane's own pair over its own coefficients.
    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                      detail::RatLane32<false>{&tables},
                                      order,
                                      static_cast<float>(x),
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route, named at the Horner
/// scheme.
///
/// One arithmetic under two names, as the double lane's pair of rows is: this
/// entry is BoysDeviceAllOrdersF32Rat and answers exactly as it does.
///
/// \copydoc boys::BoysDeviceAllOrdersF32Rat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32RatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity) {
    return BoysDeviceAllOrdersF32Rat<kForm, kExp>(tables, order, x, out, capacity);
}

/// F_0(x)..F_n(x) in float precision from the fit route on the narrow partition,
/// inside the caller's kernel.
///
/// The two axes together, as the double lane's narrow rational entry carries
/// them: the narrow partition's per-order pieces as pairs, over the double
/// lane's narrow pairs in region A and the float lane's piecewise pair in
/// region B.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowRat(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    // The same pair of halves the coarsest entry resolves, on this partition: the
    // double lane's narrow pairs for region A, and this lane's own narrow pair for
    // region B.
    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                      detail::RatLane32<true>{&tables},
                                      order,
                                      static_cast<float>(x),
                                      [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route on the narrow partition,
/// named at the Horner scheme.
///
/// One arithmetic under two names, as the double lane's pair of rows is: this
/// entry is BoysDeviceAllOrdersF32NarrowRat and answers exactly as it does.
///
/// \copydoc boys::BoysDeviceAllOrdersF32NarrowRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowRatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity) {
    return BoysDeviceAllOrdersF32NarrowRat<kForm, kExp>(tables, order, x, out, capacity);
}

// The orders reading (the packing axis) of every partition, and the half lane's ladder rows:
// region A's per-order fits are summed where they lie, not seeded from the top order's and
// brought down the recurrence (boys_cuda_arithmetic.hpp, DeviceOrdersF64/F32). Region A alone:
// past kX0 the body is the certified ladder, so an entry answers at its bound, not a second one.

/// F_0(x)..F_n(x) in double precision from the coarsest partition, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The packing axis the launched group names \c DevicePacking::kPerOrder, over the
/// same pieces BoysDeviceAllOrdersF64 reads: every order's own piece is located and
/// its own fit summed, so no value here came down a recurrence from a higher order's
/// fit. Contract, bound and refusals are BoysDeviceAllOrdersF64's, the reading of
/// region A aside.
///
/// The arithmetic is the launched row's kernel's (boys_cuda.cu,
/// BoysAllOrdersF64OrdersKernel): the two differ in where the coefficients come from
/// and not in what is done with them.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Orders(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64{&tables, deg}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the coarsest partition's monomial form,
/// each order summed from its own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersF64Orders reads, over the other stored form
/// of the same fit: every order's own piece is located and its own fit summed, and
/// the piece comes from the monomial pool and is summed by Horner rather than by the
/// split Clenshaw. Contract, bound and refusals are BoysDeviceAllOrdersF64's, the
/// reading of region A and the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64OrdersMono(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF64Batch>(tables);

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::TableLane64Mono{&tables, deg}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the narrow partition, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The same partition and the same pieces as BoysDeviceAllOrdersF64Narrow, read on
/// the other packing axis: each order's own piece is located and its own fit summed
/// rather than seeded from the top order's and brought back down the recurrence.
/// Contract, bound and refusals are BoysDeviceAllOrdersF64Narrow's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowOrders(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::NarrowLane64<false>{&tables}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values from the narrow partition's monomial form, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The lane BoysDeviceAllOrdersF64NarrowMono reads, on the packing axis
/// BoysDeviceAllOrdersF64NarrowOrders reads: the pieces are the narrow partition's
/// and the coefficients the monomial form of them, summed by Horner. Contract,
/// bound and refusals are BoysDeviceAllOrdersF64Narrow's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowOrdersMono(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::NarrowLane64<true>{&tables}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route, each order summed from
/// its own piece, inside the caller's kernel.
///
/// The route and the pieces BoysDeviceAllOrdersF64Rat reads, on the packing axis
/// BoysDeviceAllOrdersF64Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF64Rat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64OrdersRat(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::RatLane64<false>{&tables}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route's narrow partition, each
/// order summed from its own piece, inside the caller's kernel.
///
/// The route and the pieces BoysDeviceAllOrdersF64NarrowRat reads, on the packing
/// axis BoysDeviceAllOrdersF64Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF64NarrowRat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowOrdersRat(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF64<kForm, kExp == RegionBExp::kFast>(
        detail::RatLane64<true>{&tables}, order, x, [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its ladder counterpart gives:
/// the rational member is stored in one form, so both scheme names reach one
/// arithmetic. Contract, bound and refusals are
/// BoysDeviceAllOrdersF64OrdersRat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF64OrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64OrdersRatHorner(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    return BoysDeviceAllOrdersF64OrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as BoysDeviceAllOrdersF64OrdersRatHorner is.
///
/// \copydoc boys::BoysDeviceAllOrdersF64NarrowOrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowOrdersRatHorner(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    return BoysDeviceAllOrdersF64NarrowOrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// F_0(x)..F_n(x) in float precision from the coarsest partition, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The float lane's counterpart of BoysDeviceAllOrdersF64Orders, over the same two
/// lanes that entry's siblings read: region A's seed is the double lane's piece
/// table and region B's is this lane's own order-0 entry. Contract, bound and
/// refusals are BoysDeviceAllOrdersF32's, the reading of region A aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Orders(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                   detail::TableLane32{&tables, deg},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in single precision from the coarsest partition's monomial form,
/// each order summed from its own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersF32Orders reads, over the other stored form
/// of the same fit: every order's own piece is located and its own fit summed, the
/// piece comes from the monomial pools and is summed by Horner rather than by the
/// split Clenshaw, and the two lanes are BoysDeviceAllOrdersF32Mono's. Contract,
/// bound and refusals are BoysDeviceAllOrdersF32's, the reading of region A and the
/// form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32OrdersMono(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF32Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64Mono{&tables, deg},
                                   detail::TableLane32Mono{&tables, deg},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the narrow partition, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The partition and the lanes BoysDeviceAllOrdersF32Narrow reads, on the packing
/// axis BoysDeviceAllOrdersF32Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32Narrow's.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowOrders(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                   detail::NarrowLane32<false>{&tables},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values from the narrow partition's monomial form, each order summed
/// from its own fit, inside the caller's kernel.
///
/// The lanes BoysDeviceAllOrdersF32NarrowMono reads, on the packing axis
/// BoysDeviceAllOrdersF32NarrowOrders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32Narrow's.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowOrdersMono(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                   detail::NarrowLane32<true>{&tables},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route, each order summed from its
/// own piece, inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersF32Rat reads, on the packing axis
/// BoysDeviceAllOrdersF32Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32Rat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32OrdersRat(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus lanes = detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                   detail::RatLane32<false>{&tables},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route's narrow partition, each
/// order summed from its own piece, inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersF32NarrowRat reads, on the packing
/// axis BoysDeviceAllOrdersF32Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32NarrowRat's.
///
/// \copydoc boys::BoysDeviceAllOrdersF32
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowOrdersRat(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                   detail::RatLane32<true>{&tables},
                                   order,
                                   static_cast<float>(x),
                                   [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its double counterpart is.
///
/// \copydoc boys::BoysDeviceAllOrdersF32OrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32OrdersRatHorner(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    return BoysDeviceAllOrdersF32OrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its double counterpart is.
///
/// \copydoc boys::BoysDeviceAllOrdersF32NarrowOrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowOrdersRatHorner(
    const BoysDeviceTables& tables, int order, double x, float* out, int capacity) {
    return BoysDeviceAllOrdersF32NarrowOrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

#if BoysFp16
// The half lane's rows of the ladder family, in the caller's own kernel. The lane runs the float
// lane's arithmetic and stores what it returns: same lanes, tables, partitions and route, with the
// half store around every entry, and the value crossing the boundary twice - through the format's
// widen and its narrow - which is the whole of the difference and of the bound's difference.

/// F_n(x) in fp16 from the lane's fast region-B exponential, inside the caller's
/// kernel.
///
/// The same body BoysDeviceSingleF16 runs, over the region-B exponential
/// BoysCuda::SingleF16Fast was launched at: the fast exponential is this entry's
/// identity rather than a template argument, because the launched row it answers
/// for is named at that option and at no other. A caller that selects
/// RegionBExp::kFast takes that option's bound, which is the lane's plus the
/// corrected seed's own contribution.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x) in fp16; untouched unless kSuccess is returned
///
/// \pre \c x is a binary16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __half.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady or kOrderOutOfRange
/// otherwise, without writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus
BoysDeviceSingleF16Fast(const BoysDeviceTables& tables, int order, __half x, __half* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Single>(tables);

    *out = __float2half(detail::DeviceSingleF32<kForm, true>(
        detail::TableLane32{&tables, deg}, order, __half2float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_n(x) in bfloat16 from the lane's fast region-B exponential, inside the caller's
/// kernel.
///
/// BoysDeviceSingleF16Fast's body in the bfloat16 store: the same entry over the same
/// region-B exponential. The fast exponential is this entry's identity rather than a
/// template argument, because the fp16 row it mirrors is named at that member and at
/// no other, and that member's bound is the lane's plus the corrected seed's own
/// contribution.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x) in bfloat16; untouched unless kSuccess is returned
///
/// \pre \c x is a bfloat16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __nv_bfloat16.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady or kOrderOutOfRange
/// otherwise, without writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus
BoysDeviceSingleBf16Fast(const BoysDeviceTables& tables,
                         int order,
                         __nv_bfloat16 x,
                         __nv_bfloat16* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Single>(tables);

    *out = __float2bfloat16(detail::DeviceSingleF32<kForm, true>(
        detail::TableLane32{&tables, deg}, order, __bfloat162float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the coarsest partition, each order summed from its
/// own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersF16's own pieces carry, in the half store:
/// each order's own piece is located and its own fit summed rather than seeded from
/// the top order's. Contract, bound and refusals are BoysDeviceAllOrdersF16's, the
/// reading of region A aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16Orders(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                   detail::TableLane32{&tables, deg},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the coarsest partition's monomial form, each order
/// summed from its own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersF16Orders reads, over the other stored form of
/// the same fit: every order's own piece is located and its own fit summed, the piece
/// comes from the monomial pools and is summed by Horner rather than by the split
/// Clenshaw, and the value crosses the boundary as it does there. Contract, bound and
/// refusals are BoysDeviceAllOrdersF16's, the reading of region A and the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16OrdersMono(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64Mono{&tables, deg},
                                   detail::TableLane32Mono{&tables, deg},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the coarsest partition's monomial form, each order
/// summed from its own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersBf16Orders reads, over the other stored form of
/// the same fit: every order's own piece is located and its own fit summed, the piece
/// comes from the monomial pools and is summed by Horner rather than by the split
/// Clenshaw, and the value crosses the boundary as it does there. Contract, bound and
/// refusals are BoysDeviceAllOrdersBf16's, the reading of region A and the form aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
///
/// A handle carrying the Chebyshev tables and not the monomial ones is \c kTablesNotReady:
/// this entry's read rests on the pools it names.
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16OrdersMono(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceMonoReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64Mono{&tables, deg},
                                   detail::TableLane32Mono{&tables, deg},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the coarsest partition, each order summed from its
/// own fit, inside the caller's kernel.
///
/// The packing axis BoysDeviceAllOrdersBf16's own pieces carry, in the half store:
/// each order's own piece is located and its own fit summed rather than seeded from
/// the top order's. Contract, bound and refusals are BoysDeviceAllOrdersBf16's, the
/// reading of region A aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16Orders(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    const detail::Degrees deg = detail::DeviceStoredDegrees<BoysDeviceLane::kF16Batch>(tables);

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::TableLane64{&tables, deg},
                                   detail::TableLane32{&tables, deg},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the narrow partition, inside the caller's kernel.
///
/// The float lane's narrow ladder in the half store: the pieces and the lanes
/// BoysDeviceAllOrdersF32Narrow reads, with the value rounded at the boundary.
/// Contract, bound and refusals are BoysDeviceAllOrdersF16's, the partition aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16Narrow(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                      detail::NarrowLane32<false>{&tables},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the narrow partition, inside the caller's kernel.
///
/// The float lane's narrow ladder in the half store: the pieces and the lanes
/// BoysDeviceAllOrdersF32Narrow reads, with the value rounded at the boundary.
/// Contract, bound and refusals are BoysDeviceAllOrdersBf16's, the partition aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16Narrow(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                      detail::NarrowLane32<false>{&tables},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the narrow partition, each order summed from its own
/// fit, inside the caller's kernel.
///
/// The partition and the lanes BoysDeviceAllOrdersF16Narrow reads, on the packing
/// axis BoysDeviceAllOrdersF16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the partition aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowOrders(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                   detail::NarrowLane32<false>{&tables},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the narrow partition, each order summed from its own
/// fit, inside the caller's kernel.
///
/// The partition and the lanes BoysDeviceAllOrdersBf16Narrow reads, on the packing
/// axis BoysDeviceAllOrdersBf16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the partition aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowOrders(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<false>{&tables},
                                   detail::NarrowLane32<false>{&tables},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the float lane's grid, inside the caller's kernel.
///
/// The grid is the float lane's and this lane's rows of it are that lane's rows
/// in the half store (src/boys_cuda.cu, BoysAllOrdersF16FlatKernel), so contract,
/// bound and refusals are BoysDeviceAllOrdersF32Uniform's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __half
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16Uniform(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, false>(tables.flatCoeffs32,
                                                 tables.flatMonoCoeffs32,
                                                 tables.flatDegs32,
                                                 tables.flatOffsets32,
                                                 order,
                                                 __half2float(x),
                                                 [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the float lane's grid, inside the caller's kernel.
///
/// BoysDeviceAllOrdersF16Uniform's body in the bfloat16 store: the grid is the float
/// lane's, and the rows this entry answers are that lane's rows in bfloat16, as the
/// fp16 sibling's are in binary16 (src/boys_cuda.cu, BoysAllOrdersF16FlatKernel).
/// Contract, bound and refusals are BoysDeviceAllOrdersF32Uniform's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __nv_bfloat16
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a bfloat16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16Uniform(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, false>(tables.flatCoeffs32,
                                                 tables.flatMonoCoeffs32,
                                                 tables.flatDegs32,
                                                 tables.flatOffsets32,
                                                 order,
                                                 __bfloat162float(x),
                                                 [&](int l, float v) {
                                                     out[l] = __float2bfloat16(v);
                                                 });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the narrow partition's monomial form, inside the
/// caller's kernel.
///
/// The lanes BoysDeviceAllOrdersF32NarrowMono reads, in the half store. Contract,
/// bound and refusals are BoysDeviceAllOrdersF16's, the partition and the basis
/// aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowMono(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                      detail::NarrowLane32<true>{&tables},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the narrow partition's monomial form, inside the
/// caller's kernel.
///
/// The lanes BoysDeviceAllOrdersF32NarrowMono reads, in the half store. Contract,
/// bound and refusals are BoysDeviceAllOrdersBf16's, the partition and the basis
/// aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowMono(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                      detail::NarrowLane32<true>{&tables},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the narrow partition's monomial form, each order
/// summed from its own fit, inside the caller's kernel.
///
/// The lanes BoysDeviceAllOrdersF16NarrowMono reads, on the packing axis
/// BoysDeviceAllOrdersF16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the partition and the basis aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowOrdersMono(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                   detail::NarrowLane32<true>{&tables},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the narrow partition's monomial form, each order
/// summed from its own fit, inside the caller's kernel.
///
/// The lanes BoysDeviceAllOrdersBf16NarrowMono reads, on the packing axis
/// BoysDeviceAllOrdersBf16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the partition and the basis aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowOrdersMono(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowPieceStart, tables.narrowStoredDeg);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::NarrowLane64<true>{&tables},
                                   detail::NarrowLane32<true>{&tables},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the float lane's grid in its other stored form,
/// inside the caller's kernel.
///
/// The same table read by Horner rather than by the split Clenshaw, as
/// BoysDeviceAllOrdersF16Uniform is the float lane's first form. Contract, bound
/// and refusals are BoysDeviceAllOrdersF32Uniform's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __half
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16UniformHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, true>(tables.flatCoeffs32,
                                                tables.flatMonoCoeffs32,
                                                tables.flatDegs32,
                                                tables.flatOffsets32,
                                                order,
                                                __half2float(x),
                                                [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the float lane's grid in its other stored form,
/// inside the caller's kernel.
///
/// The same table read by Horner rather than by the split Clenshaw, as
/// BoysDeviceAllOrdersBf16Uniform is the float lane's first form. Contract, bound
/// and refusals are BoysDeviceAllOrdersF32Uniform's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __nv_bfloat16
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a bfloat16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16UniformHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32Flat<kForm, true>(tables.flatCoeffs32,
                                                tables.flatMonoCoeffs32,
                                                tables.flatDegs32,
                                                tables.flatOffsets32,
                                                order,
                                                __bfloat162float(x),
                                                [&](int l, float v) {
                                                    out[l] = __float2bfloat16(v);
                                                });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the fit route, inside the caller's kernel.
///
/// The float lane's coarsest rational pair in the half store: the route and the
/// lanes BoysDeviceAllOrdersF32Rat reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16Rat(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes = detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                      detail::RatLane32<false>{&tables},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the fit route, inside the caller's kernel.
///
/// The float lane's coarsest rational pair in the half store: the route and the
/// lanes BoysDeviceAllOrdersF32Rat reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16Rat(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes = detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                      detail::RatLane32<false>{&tables},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the fit route, each order summed from its own piece,
/// inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersF16Rat reads, on the packing axis
/// BoysDeviceAllOrdersF16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16OrdersRat(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes = detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                   detail::RatLane32<false>{&tables},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the fit route, each order summed from its own piece,
/// inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersBf16Rat reads, on the packing axis
/// BoysDeviceAllOrdersBf16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16OrdersRat(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes = detail::DeviceGroupReady2(tables.ratCoeffs, tables.ratBNum32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request = detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<false>{&tables},
                                   detail::RatLane32<false>{&tables},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the fit route's narrow partition, inside the
/// caller's kernel.
///
/// The float lane's narrow rational pair in the half store: the route and the lanes
/// BoysDeviceAllOrdersF32NarrowRat reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowRat(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                      detail::RatLane32<true>{&tables},
                                      order,
                                      __half2float(x),
                                      [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the fit route's narrow partition, inside the
/// caller's kernel.
///
/// The float lane's narrow rational pair in the half store: the route and the lanes
/// BoysDeviceAllOrdersF32NarrowRat reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowRat(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceAllOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                      detail::RatLane32<true>{&tables},
                                      order,
                                      __bfloat162float(x),
                                      [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the fit route's narrow partition, each order summed
/// from its own piece, inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersF16NarrowRat reads, on the packing
/// axis BoysDeviceAllOrdersF16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersF16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersF16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowOrdersRat(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                   detail::RatLane32<true>{&tables},
                                   order,
                                   __half2float(x),
                                   [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the fit route's narrow partition, each order summed
/// from its own piece, inside the caller's kernel.
///
/// The route and the lanes BoysDeviceAllOrdersBf16NarrowRat reads, on the packing
/// axis BoysDeviceAllOrdersBf16Orders reads. Contract, bound and refusals are
/// BoysDeviceAllOrdersBf16's, the route aside.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowOrdersRat(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus lanes =
        detail::DeviceGroupReady2(tables.narrowRatCoeffs, tables.narrowRatBCoeffs32);

    if (lanes != BoysDeviceStatus::kSuccess)
    {
        return lanes;
    }

    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatBCoeffs32, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::DeviceOrdersF32<kForm, kExp == RegionBExp::kFast>(detail::RatLane64<true>{&tables},
                                   detail::RatLane32<true>{&tables},
                                   order,
                                   __bfloat162float(x),
                                   [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in fp16 from the float lane's grid on the rational route, inside
/// the caller's kernel.
///
/// The grid's rational member in the half store: one numerator/denominator pair per
/// interval at the interval's own stored count, which is the arithmetic
/// BoysDeviceAllOrdersF32UniformRat runs. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32UniformRat's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __half
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16UniformRat(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatRatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32FlatRat<kForm>(tables.flatRatCoeffs32,
                                             tables.flatRatNumDeg32,
                                             tables.flatRatDenDeg32,
                                             tables.flatRatStored32,
                                             tables.flatRatOffsets32,
                                             order,
                                             __half2float(x),
                                             [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in bfloat16 from the float lane's grid on the rational route, inside
/// the caller's kernel.
///
/// The grid's rational member in the half store: one numerator/denominator pair per
/// interval at the interval's own stored count, which is the arithmetic
/// BoysDeviceAllOrdersF32UniformRat runs. Contract, bound and refusals are
/// BoysDeviceAllOrdersF32UniformRat's, the store aside.
///
/// \tparam kForm the division form the entry's ladder steps divide in. The
///         three forms are certified and differ in the arithmetic of the
///         division, never in what is approximated, so a call is placed by the
///         lane's figure plus whatever the form adds to it — the axis states
///         that term where a form has one (boys::DivisionForm). The default is
///         the build's own, \c kDefaultDeviceDivisionForm, which is what a call site
///         that names no form is compiled as; the choice is a template argument
///         because it selects an arithmetic inside the caller's own kernel, so
///         the form not named is absent from it rather than merely untaken.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive __nv_bfloat16
/// \param capacity   the number of values \c out holds
///
/// \pre \c x is a bfloat16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16UniformRat(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceFlatRatReady32(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32FlatRat<kForm>(tables.flatRatCoeffs32,
                                             tables.flatRatNumDeg32,
                                             tables.flatRatDenDeg32,
                                             tables.flatRatStored32,
                                             tables.flatRatOffsets32,
                                             order,
                                             __bfloat162float(x),
                                             [&](int l, float v) { out[l] = __float2bfloat16(v); });
    return BoysDeviceStatus::kSuccess;
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its float counterpart gives.
///
/// \copydoc boys::BoysDeviceAllOrdersF16Rat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16RatHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    return BoysDeviceAllOrdersF16Rat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its float counterpart gives.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16Rat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16RatHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    return BoysDeviceAllOrdersBf16Rat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersF16OrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16OrdersRatHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    return BoysDeviceAllOrdersF16OrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16OrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16OrdersRatHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    return BoysDeviceAllOrdersBf16OrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersF16NarrowRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowRatHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    return BoysDeviceAllOrdersF16NarrowRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16NarrowRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowRatHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    return BoysDeviceAllOrdersBf16NarrowRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersF16NarrowOrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16NarrowOrdersRatHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    return BoysDeviceAllOrdersF16NarrowOrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, as its siblings are.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16NarrowOrdersRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm, RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16NarrowOrdersRatHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    return BoysDeviceAllOrdersBf16NarrowOrdersRat<kForm, kExp>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its float counterpart gives:
/// the grid's rational member is stored in one form, so both scheme names reach one
/// arithmetic.
///
/// \copydoc boys::BoysDeviceAllOrdersF16UniformRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16UniformRatHorner(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    return BoysDeviceAllOrdersF16UniformRat<kForm>(tables, order, x, out, capacity);
}

/// The same values under the Horner scheme name, inside the caller's kernel.
///
/// A forwarder and not a second body, for the reason its float counterpart gives:
/// the grid's rational member is stored in one form, so both scheme names reach one
/// arithmetic.
///
/// \copydoc boys::BoysDeviceAllOrdersBf16UniformRat
template <DivisionForm kForm = kDefaultDeviceDivisionForm>
__device__ BoysDeviceStatus BoysDeviceAllOrdersBf16UniformRatHorner(
    const BoysDeviceTables& tables, int order, __nv_bfloat16 x, __nv_bfloat16* out, int capacity) {
    return BoysDeviceAllOrdersBf16UniformRat<kForm>(tables, order, x, out, capacity);
}
#endif // BoysFp16

} // namespace boys
