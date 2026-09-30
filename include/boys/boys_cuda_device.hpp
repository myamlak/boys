#pragma once

/// \file
/// Boys evaluation inside the caller's own kernel: F_0..F_n for an argument
/// the caller computed in a register, with no round trip through global
/// memory.
///
/// **What it is for.** A production GPU integral kernel forms x per thread —
/// one x per shell-quartet per primitive pair — and needs F_0..F_n for that
/// thread's own x, usually inside the same kernel that formed it. The batch
/// entries of this lane cannot serve that directly: they evaluate an array of
/// arguments, so a caller has to materialise every x to global memory, launch
/// a second kernel, synchronise and read the results back — two extra memory
/// passes per integral batch. The entries here take one (order, x) per call and
/// return the values to the calling thread, so the fused kernel needs neither
/// the pass out nor the pass back.
///
/// **How a caller obtains it, and what it costs the build.** The tables are
/// handed over at run time as a value, BoysDeviceTables, filled host-side by
/// BoysCuda::DeviceTables. The entries are therefore header-defined device
/// functions with no device-side symbol to link: including this header is the
/// whole of what the caller's build pays. No relocatable device code
/// (\c -rdc=true), no device link step, no library on the link line, no
/// additional compilation flag — a kernel that calls an entry compiles with the
/// same nvcc command line as one that does not, and the arithmetic is inlined
/// into the calling kernel, so there is no call overhead per order. What it
/// does cost is the compile of the coefficient tables this header pulls in
/// (boys_coefficients.hpp) and the instructions the compiler emits for the
/// arithmetic at each call site, which is the same arithmetic the batch kernels
/// run.
///
/// **Precision.** The double entries are the primary surface: a fused integral
/// kernel is a double-precision computation, and the fp64 bound is what decides
/// whether it can use the lane at all. The float entries follow — on consumer
/// hardware fp32 runs many times faster, and the fused kernel is where that
/// matters most, since the arithmetic is done once per primitive pair — and the
/// fp16 entries mirror the fp16 batch lane for a caller packing many orders
/// into registers. All three read the same handle: there is one handle type and
/// one upload, not one per precision. The float single entry carries one option
/// the others do not: its region-B exponential is a certified choice of
/// arithmetic (boys::RegionBExp), which the caller names as a template
/// argument, exactly as the f32 batch single entry names it as an argument.
///
/// **Order range, and what the entries do when they cannot serve a request.**
/// Orders run 0..kMaxBoysOrder (32). Two shapes are offered per precision. The
/// runtime-top-order entry takes the order per call, which is what a fused
/// kernel needs (the order follows the shell quartets) and takes a capacity —
/// the number of values the caller's array holds — so that an order the array
/// cannot receive is reported rather than written past the end. The
/// compile-time-top-order entry takes the order as a template argument, where
/// it bounds the recursion loops and the compiler can unroll them.
///
/// Every failure is a BoysDeviceStatus the caller branches on, and a call that
/// fails writes nothing: no truncated ladder, no partial fill. An order outside
/// the range is \c kOrderOutOfRange; an order whose values do not fit the
/// caller's array is \c kCapacityTooSmall; a handle that carries no tables is
/// \c kTablesNotReady; a multiplier that is not the resident rung is
/// \c kMultiplierNotResident.
///
/// The checks are ordered tables, then order, then capacity, then rung: a
/// request that is malformed is reported as malformed whether or not the rung it
/// names is resident, so a caller that sees \c kMultiplierNotResident has a
/// well-formed request and one thing to fix.
///
/// **Cost in registers.** A caller keeping the whole ladder live holds order + 1
/// doubles — 33 of them at the top order, which is the real cost of this
/// interface and the reason the entries write into the caller's array rather
/// than into an internal one: the caller's own liveness decides what stays in
/// registers, and a caller that consumes each order as it arrives can hold
/// none. BoysDeviceEachOrderF64 and its siblings are that shape: they hand one
/// value at a time to a caller-supplied sink and keep no ladder at all.
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
/// **The accuracy multiplier.** Every entry takes the multiplier as its last
/// argument, defaulted to m = 1, so the default is the strictest rung and a call
/// can never be relaxed by omission. The rung is a run-time argument here rather
/// than the template argument the batch entries take, because it replaces a
/// degree table read inside the caller's own kernel rather than selecting a
/// kernel: one compiled entry serves every rung, and the caller names the rung
/// where its own work decides it.
///
/// A rung is served only while it is resident. The relaxed degree tables are one
/// set, cut for one multiplier at a time — the one the last
/// BoysCuda::DeviceTables call named, on the same per-(device, m) upload the
/// batch entries share — so an entry asked for any other rung returns
/// \c kMultiplierNotResident and writes nothing, rather than running the
/// full-accuracy arithmetic under a relaxed name or a relaxed one under a name
/// that promises more. m = 1 needs no such table, is resident from the first
/// upload, and is never refused this way.
///
/// The multiplier is matched exactly against the one DeviceTables was
/// instantiated with, and a value below 1.0 is refused like any other rung that
/// is not resident (the library's compile-time entries make it a compile-time
/// error; here it is a status). So a caller names a rung the way the library
/// does — one of the twelve of kDeviceRungs (boys_cuda_options.hpp): the option
/// space's 1, 64, 256, 1024, 4096, 16384 and 65536, beside this lane's own 1, 2,
/// 10, 100, 1e4 and 1e8 — and reads the status instead of assuming the rung is
/// still the resident one.
///
/// **The bound.** Every entry holds the lane's documented bound for its
/// precision at the rung it was asked for — the same bound the corresponding
/// batch entry documents, because it is the same arithmetic: m times the m = 1
/// bound. The device gate measures these entries against the committed
/// high-precision reference grid at every rung, and reports the worst ratio in
/// the same vocabulary as every other lane.
///
/// \ingroup boys

#include "boys/boys_cuda_arithmetic.hpp"
#include "boys/boys_device_tables.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace boys {

/// Result status of the device-callable entries.
///
/// Device code cannot throw and cannot report an error out of band, so every
/// refusal is one of these values, returned to the calling thread. A call that
/// does not return \c kSuccess wrote nothing: the caller's array is untouched
/// and no value is in flight. \c kSuccess is 0.
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
    /// The handle carries no relaxed degree tables for the multiplier the call
    /// named: a different rung is the resident one, or none has been uploaded.
    /// Nothing was written. The full-accuracy rung, m = 1, is never refused
    /// this way — it is the table the handle carries from the first upload.
    kMultiplierNotResident,
};

/// \cond
namespace detail {

// Where a lane's effective degrees come from, resolved once per call rather
// than once per order. The rungs differ here and nowhere else: the piece edges,
// the coefficient pool and the piece count are the same tables for every
// multiplier, and the degree a fit is cut to is the whole of what a rung buys.
//
// The region-A table is read in the calling lane's own piece indexing — the
// double lane's for kF64Single, kF64Batch, kF32Batch and kF16Batch, whose
// region-A seed is the double piece table whatever precision the entry returns,
// and the float lane's for kF32Single and kF16Single. The lane object below
// indexes it through the piece-start table it reads its coefficients with, so
// the two agree by construction.
//
// The region-B table is read per order by the single lanes and at the order-0
// entry by the batch lanes: stride carries that, and the batch shape is what
// the region-B relaxation argues — the F_0 seed's error reaches every output
// with gain at most 1 + 1.846e-17, so one degree relaxes a whole family.
struct Degrees {
    const int* regionA;
    const int* regionB;
    int stride;
};

// The handle as a lane object — the same seven members the batch kernels' lane
// objects carry (boys_cuda_arithmetic.hpp), reading the caller's handle
// instead of a __constant__ symbol. The piece index of order n, piece p is
// pieceStart[n] + p, so no stride constant has to agree between this header
// and the tables the library uploaded.
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

// The two refusals every entry shares. The table test is one pointer, and it
// is the difference between a reported status and a null dereference; the order
// test is what makes an out-of-range request a value rather than a read outside
// the piece tables.
__device__ __forceinline__ BoysDeviceStatus DeviceReady(const BoysDeviceTables& tables) {
    return tables.pieceStart == nullptr ? BoysDeviceStatus::kTablesNotReady
                                        : BoysDeviceStatus::kSuccess;
}

// The same test for the uniform route, on the pointers that route's bodies
// actually read: the grid is a table of its own and an entry taking it touches
// no piece table, so testing the piece table would report a readiness the read
// below does not rest on.
//
// The grid's two per-interval tables are part of that test and not a separate
// one. A body addresses a cell through both, so coefficients without them is a
// table that is not resident — the read would take a degree and a block start
// from nothing — and a test that passed while either was missing would answer a
// status this route cannot serve.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatReady(const BoysDeviceTables& tables) {
    return tables.flatCoeffs == nullptr || tables.flatDegs == nullptr ||
                   tables.flatOffsets == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

// The float lane's own test, on its own lane's pointers. The two lanes' tables
// are derived separately and a handle may carry one without the other, so a
// float entry testing the double lane's pointers would report a readiness its
// own read does not rest on — and would dereference a null one it never tested.
__device__ __forceinline__ BoysDeviceStatus DeviceFlatReady32(const BoysDeviceTables& tables) {
    return tables.flatCoeffs32 == nullptr || tables.flatDegs32 == nullptr ||
                   tables.flatOffsets32 == nullptr
               ? BoysDeviceStatus::kTablesNotReady
               : BoysDeviceStatus::kSuccess;
}

__device__ __forceinline__ bool DeviceOrderValid(int order) {
    return order >= 0 && order <= kMaxBoysOrder;
}

// Which lane's degrees a call reads, resolved once per call from the rung the
// caller named. m = 1 reads the handle's own full-accuracy tables, and every
// other rung reads the resident relaxed set, which the lane index selects.
//
// The lanes whose region-A seed is the double piece table are the double single
// entry and the three family entries; the two single entries of the narrow
// precisions seed from the float piece table. That is the same split the batch
// kernels' lane objects make (boys_cuda.cu), and it is what makes the piece
// index below the calling lane's own.
//
// The batch lanes read the order-0 region-B entry and the single lanes the entry
// for the order the recursion has reached; stride carries that, and stride 0 is
// also what the full-accuracy case reads, where the degree is one scalar in the
// handle rather than a table.
template <BoysDeviceLane kLane>
__device__ __forceinline__ BoysDeviceStatus DeviceDegrees(
    const BoysDeviceTables& tables, double multiplier, Degrees* out) {
    constexpr int kLaneIndex = static_cast<int>(kLane);
    constexpr bool kDoublePieces =
        kLane != BoysDeviceLane::kF32Single && kLane != BoysDeviceLane::kF16Single;
    constexpr bool kBatch = kLane == BoysDeviceLane::kF64Batch ||
                            kLane == BoysDeviceLane::kF32Batch ||
                            kLane == BoysDeviceLane::kF16Batch;

    if (multiplier == kBoysFullAccuracyMultiplier)
    {
        out->regionA = kDoublePieces ? tables.pieceDeg : tables.pieceDeg32;
        out->regionB =
            kDoublePieces ? static_cast<const int*>(&tables.bSeedDeg) : &tables.bSeedDeg32;
        out->stride = 0;
        return BoysDeviceStatus::kSuccess;
    }

    if (multiplier < kBoysFullAccuracyMultiplier || tables.relaxedRung == nullptr ||
        *tables.relaxedRung != multiplier || tables.relaxedDegA[kLaneIndex] == nullptr ||
        tables.relaxedDegB[kLaneIndex] == nullptr)
    {
        return BoysDeviceStatus::kMultiplierNotResident;
    }

    out->regionA = tables.relaxedDegA[kLaneIndex];
    out->regionB = tables.relaxedDegB[kLaneIndex];
    out->stride = kBatch ? 0 : 1;
    return BoysDeviceStatus::kSuccess;
}

// Where a narrow-partition lane's degrees come from, resolved once per call. The
// partition's piece indexing is its own — order n's pieces are entries n and n+1
// of the handle's narrowPieceStart — so both tables are read at that index and
// no stride constant has to agree with the library.
//
// regionB is the resident rung's cut of region B's seed, one degree per piece
// and per order of the recursion, and the piece's own order-0 entry is the one a
// batch-shaped lane reads because that seed is F_0's fit. It is null at the
// full-accuracy multiplier, where the seed is stored at one degree for every
// piece and that degree is a constant of the table rather than a value the
// handle carries.
struct NarrowDegrees {
    const int* regionA;
    const int* regionB;
};

template <bool kMonomial>
__device__ __forceinline__ BoysDeviceStatus DeviceNarrowDegrees(
    const BoysDeviceTables& tables, double multiplier, NarrowDegrees* out) {
    const int* const cutA =
        kMonomial ? tables.narrowMonoRelaxedDegA : tables.narrowRelaxedDegA;
    const int* const cutB =
        kMonomial ? tables.narrowMonoRelaxedDegB : tables.narrowRelaxedDegB;

    if (multiplier == kBoysFullAccuracyMultiplier)
    {
        out->regionA = tables.narrowStoredDeg;
        out->regionB = nullptr;
        return BoysDeviceStatus::kSuccess;
    }

    if (multiplier < kBoysFullAccuracyMultiplier || tables.relaxedRung == nullptr ||
        *tables.relaxedRung != multiplier || cutA == nullptr || cutB == nullptr)
    {
        return BoysDeviceStatus::kMultiplierNotResident;
    }

    out->regionA = cutA;
    out->regionB = cutB;
    return BoysDeviceStatus::kSuccess;
}

// Where a fit-route lane's region-A degrees come from, resolved once per call.
// The route's cut is a pair per piece and comes in two readings, and the reading
// a device-callable entry makes is the ladder's: region A descends from the top
// order's own piece, so the cut the handle carries is the one that reading takes.
//
// The two members are the pair's two degrees and the stride says how to reach
// the denominator from the numerator: the stored degrees are two tables of one
// degree per piece, and a cut leaves them interleaved with the denominator
// second, which is why the resident rung's reading is one pointer and a stride.
//
// relaxed says which of the route's two readings of a degree this rung is: the
// degrees the fit was stored at, or the resident rung's cut of them. Region B's
// seed is read whole rather than per piece on the shipped partition, so its
// degrees are matched to the same reading, and a lane that read the cut while
// the call was for the stored degrees — or the other way round — would sum a
// fit it was not asked for.
struct RatDegrees {
    const int* num;
    const int* den;
    int stride;
    bool relaxed;
};

template <bool kNarrow>
__device__ __forceinline__ BoysDeviceStatus DeviceRatDegrees(
    const BoysDeviceTables& tables, double multiplier, RatDegrees* out) {
    if (multiplier == kBoysFullAccuracyMultiplier)
    {
        out->num = kNarrow ? tables.narrowRatNumDeg : tables.ratNumDeg;
        out->den = kNarrow ? tables.narrowRatDenDeg : tables.ratDenDeg;
        out->stride = 1;
        out->relaxed = false;
        return BoysDeviceStatus::kSuccess;
    }

    const int* const cut = kNarrow ? tables.narrowRatSeedDeg : tables.ratSeedDeg;

    if (multiplier < kBoysFullAccuracyMultiplier || tables.relaxedRung == nullptr ||
        *tables.relaxedRung != multiplier || cut == nullptr)
    {
        return BoysDeviceStatus::kMultiplierNotResident;
    }

    out->num = cut;
    out->den = cut + 1;
    out->stride = 2;
    out->relaxed = true;
    return BoysDeviceStatus::kSuccess;
}

// The narrow partition as a lane object, one per stored form of region A. The
// pieces are cut per order, so the lane reads its edges and its stored degree
// through the handle's own piece-start table, and the form is the coefficient
// pool and the summation that reads it — the same pieces at the same degrees.
//
// Region B's seed is piecewise on this partition, so the lane locates the piece
// itself and reads that piece's degree at the order-0 entry, exactly as the
// batch kernels' narrow lanes do (boys_cuda.cu): that seed is F_0's fit, so the
// batch shape reads one degree whatever order the ladder reaches.
template <bool kMono>
struct NarrowLane64 {
    const BoysDeviceTables* tables;
    NarrowDegrees deg;

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
        return deg.regionA[tables->narrowPieceStart[order] + piece];
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
        const int stored = kNarrowBDeg;
        const int cut = deg.regionB == nullptr
                            ? stored
                            : deg.regionB[piece * (kMaxBoysOrder + 1)];

        // The form of the lane selects the summation here as it does in region A:
        // the two forms are one fit stored twice, and summing either one with the
        // other's reader would read the half of the table that is not there.
        if constexpr (kMono)
        {
            return DeviceHornerMono(c + piece * (stored + 1), cut, t);
        } else
        {
            return DeviceClenshawSplit(c + piece * (stored + 1), cut, t);
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

        return kMono ? DeviceHornerMono32(c + piece * (f32::kNarrowBDegF32 + 1),
                                              f32::kNarrowBDegF32,
                                              t)
                         : DeviceClenshawSplit32(c + piece * (f32::kNarrowBDegF32 + 1),
                                                 f32::kNarrowBDegF32,
                                                 t);
    }
};

// The fit route as a lane object, one per partition, at the reading its entry
// makes. Its pieces are the partition's own — the shipped piece tables for one
// and the narrow ones for the other — and what the lane adds is the pair: the
// numerator's block and the denominator's, the two degrees, and the seed the
// region-B recursion starts from.
template <bool kNarrow>
struct RatLane64 {
    const BoysDeviceTables* tables;
    RatDegrees deg;

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
        return deg.num[Flat(order, piece) * deg.stride];
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        return deg.den[Flat(order, piece) * deg.stride];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        if constexpr (!kNarrow)
        {
            const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
            const int numDeg = deg.relaxed ? tables->ratRelaxedDegB[0] : kRatBnumDeg;
            const int denDeg = deg.relaxed ? tables->ratRelaxedDegB[1] : kRatBdenDeg;

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
            const int numDeg = deg.relaxed ? tables->narrowRatRelaxedDegB[2 * piece]
                                           : stored[piece];
            const int den = deg.relaxed ? tables->narrowRatRelaxedDegB[2 * piece + 1]
                                        : denDeg[piece];

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
            return DeviceRatSum32(tables->ratBNum32, f32::kRatBnumDeg, tables->ratBDen32,
                                  f32::kRatBdenDeg, t);
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
            const int numDeg = tables->narrowRatBStoredNumDeg32[piece];

            return DeviceRatSum32(c, numDeg, c + numDeg + 1,
                                  tables->narrowRatBDenDeg32[piece], u);
        }
    }
};

// The readiness test for a partition or a route whose geometry is not the shipped
// piece table: the same one-pointer test as DeviceReady, on a pointer the entry's
// own body reads.
__device__ __forceinline__ BoysDeviceStatus DeviceGroupReady(const void* first) {
    return first == nullptr ? BoysDeviceStatus::kTablesNotReady
                            : BoysDeviceStatus::kSuccess;
}

// The same test for an entry that reads two lanes of one partition. The float
// entries of the partitions below seed region A from the double lane's tables
// and sum region B from their own, so a handle that carried one lane's tables
// without the other's would have the call read what is not there — and the test
// is on both pointers rather than on the one that happens to be tested first.
__device__ __forceinline__ BoysDeviceStatus DeviceGroupReady2(const void* first,
                                                              const void* second) {
    return first != nullptr && second != nullptr ? BoysDeviceStatus::kSuccess
                                                 : BoysDeviceStatus::kTablesNotReady;
}

// The rung test of an entry whose partition or route is stored at one rung: this
// build holds no cut of that table for any other multiplier, so the rung such a
// call names is not resident and is refused with the status every entry of this
// lane refuses a non-resident rung with. Answering from the stored table under a
// relaxed name is the substitution this test exists to prevent, and it is a
// refusal of work this build owes rather than a property of the table.
__device__ __forceinline__ BoysDeviceStatus DeviceFullAccuracyOnly(double multiplier) {
    return multiplier == kBoysFullAccuracyMultiplier
               ? BoysDeviceStatus::kSuccess
               : BoysDeviceStatus::kMultiplierNotResident;
}

// The rung test of an entry whose table has no cut to make: the full-accuracy
// multiplier always, and another multiplier while it is the one the handle was
// made resident at.
//
// A table stored at one degree for every order and every interval is admissible
// at every rung — no rung's criterion cuts a degree out of it — so the rung a
// call names is answered by the table's own coefficients and delivers that
// rung's arithmetic, which is the same statement the launched rows of such a
// route make. A call at a multiplier that is neither the full-accuracy one nor
// the handle's resident rung is refused: that is a handle whose tables were cut
// for another rung, and what such a call cannot be given is a table this build
// does not hold.
__device__ __forceinline__ BoysDeviceStatus DeviceResidentRung(const BoysDeviceTables& tables,
                                                               double multiplier) {
    if (multiplier == kBoysFullAccuracyMultiplier)
    {
        return BoysDeviceStatus::kSuccess;
    }

    if (multiplier < kBoysFullAccuracyMultiplier || tables.relaxedRung == nullptr ||
        *tables.relaxedRung != multiplier)
    {
        return BoysDeviceStatus::kMultiplierNotResident;
    }

    return BoysDeviceStatus::kSuccess;
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
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x); untouched unless kSuccess is returned
/// \param multiplier the accuracy multiplier, m >= 1.0, matched exactly against
///        the rung BoysCuda::DeviceTables made resident. The bound delivered is
///        m times the m = 1 bound: 1e-15 below the region-A edge, 3e-14 through
///        the extended band and region B, 5.5e-14 in region C.
///
/// \pre \c x >= 0, as for the batch entries. A negative argument names a
///      different function than the one the tables fit, and is not checked: the
///      arithmetic's domain is the caller's to keep, exactly as the batch
///      entries' is.
/// \pre \c out is a device-writable location for one double.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady when \c tables
/// carries no tables; kOrderOutOfRange when \c order is outside
/// 0..kMaxBoysOrder; kMultiplierNotResident when \c multiplier is not the
/// resident rung. A refused call writes nothing.
__device__ BoysDeviceStatus BoysDeviceSingleF64(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    double multiplier = kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF64Single>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    *out = detail::DeviceSingleF64(detail::TableLane64{&tables, deg}, order, x);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, inside the caller's kernel.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive doubles
/// \param capacity   the number of doubles \c out holds
/// \param multiplier the accuracy multiplier, as BoysDeviceSingleF64 states. The
///        family reads its region-B degree at the order-0 entry, so one rung
///        covers the whole ladder.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   double* out,
                                                   int capacity,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
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

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF64Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables, deg}, order, x, [&](int l, double v) {
        out[l] = v;
    });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in double precision, with the top order fixed where
/// the call site names it.
///
/// The order is a template argument, so the recursion bounds are compile-time
/// constants: the loops can be unrolled and no order has to be validated. For a
/// caller whose fused kernel works at one order — or at a handful it
/// instantiates separately — this is the entry to use, and it is also the
/// shape a caller can wrap in its own dispatch where the order is a run-time
/// value.
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   doubles
/// \param multiplier the accuracy multiplier, as BoysDeviceSingleF64 states
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady when
/// \c tables carries no tables, or kMultiplierNotResident when \c multiplier is
/// not the resident rung. There is no order or capacity check: the top order is
/// compiled in and \c out is the caller's declaration.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF64(const BoysDeviceTables& tables,
                                              double x,
                                              double* out,
                                              double multiplier = kBoysFullAccuracyMultiplier) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF64Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables, deg}, kTopOrder, x,
                               [&](int l, double v) {
                                   out[l] = v;
                               });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, one value at a time into a caller's
/// sink.
///
/// The shape to use when the caller consumes each order as it arrives — an
/// integral kernel contracting F_l with a coefficient does exactly that — and
/// therefore has no reason to hold the ladder: nothing is stored inside this
/// call and the register cost is the caller's sink's own. The sink is called
/// once per order, in the order the recursion produces it, which is the top
/// order downwards inside region A and 0 upwards in regions B and C: a sink
/// that cares about the sequence of calls can see region A's descending
/// arrival, and a sink that contracts into an accumulator cannot.
///
/// \tparam Sink a callable taking (int order, double value), callable from
///         device code. Passing it by value is deliberate: a lambda capturing
///         the caller's accumulators stays in registers.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
/// \param multiplier the accuracy multiplier, as BoysDeviceSingleF64 states
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c sink is device-callable and its own work does not fault.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady, kOrderOutOfRange or kMultiplierNotResident otherwise, in
/// which case \c sink is not called at all.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF64(const BoysDeviceTables& tables,
                                                   int order,
                                                   double x,
                                                   Sink sink,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF64Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables, deg}, order, x, sink);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the uniform grid, inside the
/// caller's kernel.
///
/// The partition the CPU lane names \c FitGranularity::kUniform: one table of
/// equal intervals over [0, kFlatHi) rather than pieces cut where the function
/// needs them. Below the join every order is summed from its own block and none
/// is built from another, so the top-order argument's ladder is that many
/// independent chains instead of one recurrence — the independence is the whole
/// of what the route buys, and it is why every block is stored at one degree.
/// Above kFlatHi the table does not reach and the call falls to the asymptotic
/// every other route ends in, so the figure a caller places this entry by is the
/// double batch lane's bound, which a rung of the call scales by m exactly as it
/// does on every route of this lane.
///
/// That is the bound the launched row of this route documents, and the
/// arithmetic is its kernel's (boys_cuda.cu, BoysAllOrdersF64FlatKernel): the
/// two differ in where the coefficients come from and not in what is done with
/// them.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier, m >= 1.0, matched exactly against
///        the rung BoysCuda::DeviceTables made resident, as every entry of this
///        header is. This route's table is stored at one degree for every order
///        and every interval, so no rung's criterion cuts it and no rung is
///        answered from a shorter fit: every rung the handle is resident for is
///        served by the table's own coefficients, and a rung the handle is not
///        resident for is refused with kMultiplierNotResident, exactly as the
///        launched entries of this route answer.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Uniform(const BoysDeviceTables& tables,
                                                          int order,
                                                          double x,
                                                          double* out,
                                                          int capacity,
                                                          double multiplier =
                                                              kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceResidentRung(tables, multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64Flat<false>(tables.flatCoeffs, tables.flatMonoCoeffs,
                                          tables.flatDegs, tables.flatOffsets, order, x,
                                          [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) from the uniform grid's other stored form, inside the
/// caller's kernel.
///
/// The same table and the same degrees, read by Horner in the monomial basis
/// rather than by the split Clenshaw in the Chebyshev one. The two blocks are
/// two rows of one table, so this entry and the one above cannot come to
/// disagree about which interval an argument falls in, where the table stops or
/// what its degree is; they differ only in the summation, which is the whole of
/// what a caller selects between them. Contract, bound and refusals are
/// BoysDeviceAllOrdersF64Uniform's, its own scheme aside.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64UniformHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceResidentRung(tables, multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64Flat<true>(tables.flatCoeffs, tables.flatMonoCoeffs,
                                         tables.flatDegs, tables.flatOffsets, order, x,
                                         [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_n(x) in single precision, inside the caller's kernel.
///
/// The float lane's bound is the one the f32 batch entries document; this
/// entry is the same arithmetic at m = 1, and it takes the lane's own choice of
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
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x); untouched unless kSuccess is returned
/// \param multiplier the accuracy multiplier, as BoysDeviceSingleF64 states. The
///        bound delivered is m times the one the option above names.
///
/// \pre \c x >= 0; \c out is a device-writable location for one float.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady, kOrderOutOfRange or
/// kMultiplierNotResident otherwise, without writing.
template <RegionBExp kExp = kDefaultRegionBExp>
__device__ BoysDeviceStatus BoysDeviceSingleF32(const BoysDeviceTables& tables,
                                                int order,
                                                float x,
                                                float* out,
                                                double multiplier =
                                                    kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF32Single>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    *out = detail::DeviceSingleF32<kExp == RegionBExp::kFast>(detail::TableLane32{&tables, deg},
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
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive floats
/// \param capacity   the number of floats \c out holds
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x >= 0; \c order + 1 <= \c capacity, or the call is refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32(const BoysDeviceTables& tables,
                                                   int order,
                                                   float x,
                                                   float* out,
                                                   int capacity,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
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

    // One rung for both lane objects: the batch lane's region-A degree is
    // indexed by the double piece table and its region-B degree by the float
    // one, which is what the two lane objects read it with. Only the seed lane
    // reads a degree here; the returned degrees come from the other.
    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF32Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
                               detail::TableLane32{&tables, deg},
                               order,
                               x,
                               [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in single precision, with the top order fixed where
/// the call site names it (see BoysDeviceAllNF64).
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   floats
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x >= 0.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady or
/// kMultiplierNotResident otherwise.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF32(const BoysDeviceTables& tables,
                                              float x,
                                              float* out,
                                              double multiplier = kBoysFullAccuracyMultiplier) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF32Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
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
/// \tparam Sink a callable taking (int order, float value), callable from
///         device code.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x >= 0; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady, kOrderOutOfRange or kMultiplierNotResident otherwise, with
/// \c sink not called.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF32(const BoysDeviceTables& tables,
                                                   int order,
                                                   float x,
                                                   Sink sink,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF32Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
                               detail::TableLane32{&tables, deg},
                               order,
                               x,
                               sink);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in single precision from the float lane's uniform grid,
/// inside the caller's kernel.
///
/// The float lane's own grid and not a re-cut of the double lane's: 245
/// intervals at degree 4 against the double lane's 245 at degree 8, fitted over
/// the same [0, kFlatHi). It is the lane's table throughout — region A's seed is
/// the grid's own block rather than the double piece table the float lane's
/// piecewise entries seed from, and every sum below the join is a float one —
/// so the bound is the float lane's bound, scaled by m at a relaxed rung exactly
/// as it is on the lane's other routes, and it is the bound the launched row of
/// this route documents.
///
/// The mapped argument is this lane's own spelling and not the double entry's:
/// the interval's edges are the correctly rounded float quotients iv/7 and
/// (iv + 1)/7, which is the grid the fits were measured on. Above kFlatHi the
/// call falls to the same asymptotic the other float entries end in.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier, m >= 1.0, matched against the rung
///        BoysCuda::DeviceTables made resident for the reason
///        BoysDeviceAllOrdersF64Uniform gives: the grid is stored at one degree
///        for every order and every interval, so every resident rung of it is
///        served by the table's own coefficients.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Uniform(const BoysDeviceTables& tables,
                                                          int order,
                                                          float x,
                                                          float* out,
                                                          int capacity,
                                                          double multiplier =
                                                              kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceResidentRung(tables, multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32Flat<false>(tables.flatCoeffs32, tables.flatMonoCoeffs32,
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
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32UniformHorner(
    const BoysDeviceTables& tables,
    int order,
    float x,
    float* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceResidentRung(tables, multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32Flat<true>(tables.flatCoeffs32, tables.flatMonoCoeffs32,
                                         tables.flatDegs32, tables.flatOffsets32, order, x,
                                         [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

#if BoysFp16
/// F_n(x) in fp16, inside the caller's kernel.
///
/// The fp16 lane rounds at the boundary and runs the fp32 engine in between, so
/// this entry takes and returns the raw IEEE-754 binary16 bit patterns (the
/// __half type), as the fp16 batch entries take and return this library's F16.
/// The bound is the fp16 lane's.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_n(x) in fp16; untouched unless kSuccess is returned
/// \param multiplier the accuracy multiplier, as BoysDeviceSingleF64 states. The
///        bound delivered is m * 1e-7 plus the returned value's half ULP.
///
/// \pre \c x is a binary16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __half.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady, kOrderOutOfRange or
/// kMultiplierNotResident otherwise, without writing.
__device__ BoysDeviceStatus BoysDeviceSingleF16(const BoysDeviceTables& tables,
                                                int order,
                                                __half x,
                                                __half* out,
                                                double multiplier =
                                                    kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF16Single>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    *out = __float2half(detail::DeviceSingleF32<false>(
        detail::TableLane32{&tables, deg}, order, __half2float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, inside the caller's kernel.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_order(x), order + 1 consecutive __half
/// \param capacity   the number of __half values \c out holds
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __half x,
                                                   __half* out,
                                                   int capacity,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
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

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF16Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
                               detail::TableLane32{&tables, deg},
                               order,
                               __half2float(x),
                               [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in fp16, with the top order fixed where the call site
/// names it (see BoysDeviceAllNF64).
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///                   __half values
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x is a binary16 value.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady or
/// kMultiplierNotResident otherwise.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF16(const BoysDeviceTables& tables,
                                              __half x,
                                              __half* out,
                                              double multiplier = kBoysFullAccuracyMultiplier) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF16Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
                               detail::TableLane32{&tables, deg},
                               kTopOrder,
                               __half2float(x),
                               [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, one value at a time into a caller's sink (see
/// BoysDeviceEachOrderF64 for the arrival order and the reason the shape
/// exists).
///
/// \tparam Sink a callable taking (int order, __half value), callable from
///         device code.
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the top order, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param sink       receives (l, F_l(x)) for every l in 0..order
/// \param multiplier the accuracy multiplier, as BoysDeviceAllOrdersF64 states
///
/// \pre \c x is a binary16 value; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady, kOrderOutOfRange or kMultiplierNotResident otherwise, with
/// \c sink not called.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF16(const BoysDeviceTables& tables,
                                                   int order,
                                                   __half x,
                                                   Sink sink,
                                                   double multiplier =
                                                       kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::Degrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceDegrees<BoysDeviceLane::kF16Batch>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables, deg},
                               detail::TableLane32{&tables, deg},
                               order,
                               __half2float(x),
                               [&](int l, float v) { sink(l, __float2half(v)); });
    return BoysDeviceStatus::kSuccess;
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// the narrow partition, and the fit route, in the caller's own kernel
// ---------------------------------------------------------------------------
// The axes the launched group carries beyond the shipped partition and the
// Chebyshev scheme, in the shapes this header offers. Each entry is the same
// ladder as its sibling on the shipped partition and differs in the tables its
// lane reads, which is the whole of what the partition, the scheme and the route
// axes are (boys_cuda_arithmetic.hpp): the regions, the shapes and the
// recourses are one arithmetic.
//
// The narrow partition's pieces are cut per order, so its entries read the
// handle's own piece-start table and a piece index is the partition's; the fit
// route stores each piece as a numerator and a denominator and its entries sum
// the pair and divide, which is the one operation the other routes do not carry.
//
// The rung a call names is matched against the one BoysCuda::DeviceTables made
// resident, as everywhere in this header. The double lane's narrow and rational
// tables are cut per rung and this build holds every cut of them, so those
// entries answer at every rung of the lane. The float lane's narrow and rational
// tables have a cut this build does not derive or hold — it derives and uploads
// the double role's cut alone — so an entry over them answers at the
// full-accuracy multiplier and names the rung it refuses; the launched rows of
// those partitions refuse the same rungs for the same reason, which is work this
// build owes and not a property of the tables.

/// F_0(x)..F_n(x) in double precision from the narrow partition, inside the
/// caller's kernel.
///
/// The partition the CPU lane names \c FitGranularity::kNarrow: the pieces are
/// cut per order rather than shared, so this entry reads the handle's own
/// piece-start table and region A's fit of order \c n is the one cut for that
/// order. Region B's seed is piecewise on this partition, over the edges the
/// handle carries.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier, m >= 1.0, matched exactly against
///        the rung BoysCuda::DeviceTables made resident. The bound delivered is
///        m times the double batch lane's, which is the bound the launched row
///        of this partition documents.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Narrow(const BoysDeviceTables& tables,
                                                         int order,
                                                         double x,
                                                         double* out,
                                                         int capacity,
                                                         double multiplier =
                                                             kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::NarrowDegrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceNarrowDegrees<false>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::NarrowLane64<false>{&tables, deg}, order, x,
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
/// The parameters, the preconditions and the statuses are those of
/// BoysDeviceAllOrdersF64Narrow, and the bound is the same one: the two forms
/// are two summations of one fit, and the lane's bound is over the route.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowMono(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowPieceStart, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::NarrowDegrees deg;
    const BoysDeviceStatus rung =
        detail::DeviceNarrowDegrees<true>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::NarrowLane64<true>{&tables, deg}, order, x,
                               [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route, inside the caller's
/// kernel.
///
/// The route the CPU lane names \c FitRoute::kRational: the same pieces as the
/// shipped partition, each stored as a numerator and a denominator and read as
/// the quotient of the two sums. The division is the arithmetic's own — one per
/// piece — and it is the whole of what separates this route from the Chebyshev
/// one, whose fits are polynomials.
///
/// The route's region-A fit is cut per reading, and the reading this entry makes
/// is its own: the ladder descends from the top order's piece, so the seed's cut
/// is the one the entry reads and the resident rung's table is that reading's.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive doubles
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier, m >= 1.0, matched exactly against
///        the rung BoysCuda::DeviceTables made resident. The bound delivered is
///        m times the double batch lane's, which is the bound the launched row
///        of this route documents.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64Rat(const BoysDeviceTables& tables,
                                                      int order,
                                                      double x,
                                                      double* out,
                                                      int capacity,
                                                      double multiplier =
                                                          kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.ratCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::RatDegrees deg;
    const BoysDeviceStatus rung = detail::DeviceRatDegrees<false>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::RatLane64<false>{&tables, deg}, order, x,
                               [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route, named at the Horner
/// scheme.
///
/// One arithmetic under two names, which is what the launched group's two rows
/// for this route are: the route's pair is stored once, in one basis, and both
/// scheme names a caller may use select it. This entry is BoysDeviceAllOrdersF64Rat
/// and answers exactly as it does, at the same rungs and with the same bound.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64RatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    return BoysDeviceAllOrdersF64Rat(tables, order, x, out, capacity, multiplier);
}

/// F_0(x)..F_n(x) in double precision from the fit route on the narrow
/// partition, inside the caller's kernel.
///
/// The two axes together: the narrow partition's per-order pieces, stored as
/// numerator and denominator pairs, with region B's piecewise seed stored as a
/// pair as well. Both of the route's readings are cut on this partition, and
/// this entry makes the ladder's, as BoysDeviceAllOrdersF64Rat does.
///
/// The parameters, the preconditions and the statuses are those of
/// BoysDeviceAllOrdersF64Rat, and the bound is the double batch lane's at the
/// rung named.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowRat(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    const BoysDeviceStatus request =
        detail::DeviceLadderRequest(tables.narrowRatCoeffs, order, capacity);

    if (request != BoysDeviceStatus::kSuccess)
    {
        return request;
    }

    detail::RatDegrees deg;
    const BoysDeviceStatus rung = detail::DeviceRatDegrees<true>(tables, multiplier, &deg);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::DeviceAllOrdersF64(detail::RatLane64<true>{&tables, deg}, order, x,
                               [&](int l, double v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in double precision from the fit route on the narrow
/// partition, named at the Horner scheme.
///
/// One arithmetic under two names, as BoysDeviceAllOrdersF64RatHorner is: this
/// entry is BoysDeviceAllOrdersF64NarrowRat and answers exactly as it does.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64NarrowRatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    double* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    return BoysDeviceAllOrdersF64NarrowRat(tables, order, x, out, capacity, multiplier);
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
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier. This partition's float tables have
///        a rung cut this build does not hold, so m = 1 is the only multiplier
///        they are served at, and any other is refused with
///        kMultiplierNotResident rather than answered from a table cut for
///        another rung or from the double lane's cut. The bound delivered is the
///        float batch lane's, which is the bound the launched row of this
///        partition documents.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Narrow(const BoysDeviceTables& tables,
                                                         int order,
                                                         double x,
                                                         float* out,
                                                         int capacity,
                                                         double multiplier =
                                                             kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceFullAccuracyOnly(multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::NarrowDegrees seed;
    detail::DeviceNarrowDegrees<false>(tables, kBoysFullAccuracyMultiplier, &seed);

    detail::DeviceAllOrdersF32(detail::NarrowLane64<false>{&tables, seed},
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
/// The parameters, the preconditions, the rung and the statuses are those of
/// BoysDeviceAllOrdersF32Narrow.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowMono(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceFullAccuracyOnly(multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::NarrowDegrees seed;
    detail::DeviceNarrowDegrees<true>(tables, kBoysFullAccuracyMultiplier, &seed);

    detail::DeviceAllOrdersF32(detail::NarrowLane64<true>{&tables, seed},
                               detail::NarrowLane32<true>{&tables},
                               order,
                               static_cast<float>(x),
                               [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_n(x) in float precision from the fit route, inside the caller's
/// kernel.
///
/// The route's pair on the shipped partition, with region A seeded from the
/// double lane's pairs and region B from the float lane's own pair — the split
/// the launched row makes, and the reason a float seed is never amplified.
///
/// \param tables     the handle BoysCuda::DeviceTables filled
/// \param order      the order n, 0..kMaxBoysOrder
/// \param x          the argument, >= 0, formed by the calling thread
/// \param out        receives F_0(x)..F_n(x), order + 1 consecutive floats
/// \param capacity   the number of values \c out holds
/// \param multiplier the accuracy multiplier. This route's float lane has a
///        region-B pair whose rung cut this build does not hold, so m = 1 is the
///        only multiplier it is served at, and any other is refused with
///        kMultiplierNotResident rather than answered from a cut of one side of
///        the body. The bound delivered is the float batch lane's, which is the
///        bound the launched row of this route documents.
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange, kCapacityTooSmall or kMultiplierNotResident otherwise, in
/// every case without writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32Rat(const BoysDeviceTables& tables,
                                                      int order,
                                                      double x,
                                                      float* out,
                                                      int capacity,
                                                      double multiplier =
                                                          kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceFullAccuracyOnly(multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::RatDegrees deg;
    detail::DeviceRatDegrees<false>(tables, kBoysFullAccuracyMultiplier, &deg);

    detail::DeviceAllOrdersF32(detail::RatLane64<false>{&tables, deg},
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
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32RatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    return BoysDeviceAllOrdersF32Rat(tables, order, x, out, capacity, multiplier);
}

/// F_0(x)..F_n(x) in float precision from the fit route on the narrow partition,
/// inside the caller's kernel.
///
/// The two axes together, as the double lane's narrow rational entry carries
/// them: the narrow partition's per-order pieces as pairs, over the double
/// lane's narrow pairs in region A and the float lane's piecewise pair in
/// region B.
///
/// The parameters, the preconditions, the rung and the statuses are those of
/// BoysDeviceAllOrdersF32Rat.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowRat(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
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

    const BoysDeviceStatus rung = detail::DeviceFullAccuracyOnly(multiplier);

    if (rung != BoysDeviceStatus::kSuccess)
    {
        return rung;
    }

    detail::RatDegrees deg;
    detail::DeviceRatDegrees<true>(tables, kBoysFullAccuracyMultiplier, &deg);

    detail::DeviceAllOrdersF32(detail::RatLane64<true>{&tables, deg},
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
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32NarrowRatHorner(
    const BoysDeviceTables& tables,
    int order,
    double x,
    float* out,
    int capacity,
    double multiplier = kBoysFullAccuracyMultiplier) {
    return BoysDeviceAllOrdersF32NarrowRat(tables, order, x, out, capacity, multiplier);
}

} // namespace boys
