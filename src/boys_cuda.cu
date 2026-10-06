// CUDA lane of the Boys kernel. Mirrors the scalar/simd design (boys.cpp, boys_simd.cpp):
// piecewise Chebyshev seeds + recurrences, region structure A/B/C with the weighted
// region-A fits (see tools/gen_boys_coefficients.py). C++20 with a CUDA-safe include
// list only: the library's C++23 headers would poison the nvcc translation unit.

#include "boys/boys_cuda_arithmetic.hpp"

// The handle's own declaration, for the count the address order's two halves are asserted
// against: the fields and the export order are one statement, and the assertion keeps them one.
#include "boys/boys_device_tables.hpp"

#include <cuda_bf16.h>
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

// ---------------------------------------------------------------------------
// the flat device image: the tables a caller's own kernel reads
// ---------------------------------------------------------------------------
// A consumer's kernel cannot name this translation unit's __constant__ arrays (that takes
// relocatable device code in the consumer's build), so the tables are copied a second time into
// __device__ globals whose addresses the host hands out. The piece metadata is the flat form of
// the __constant__ tables, indexed by pieceStart[order] + piece - the piece-start table the
// coefficient header already carries - so no stride constant has to agree between the library
// and a consumer. The handle is a value and kernel parameters live in the constant bank, so
// reading it costs no global memory traffic.
//
// The two lanes' piece tables are separate: the float lane cuts its orders differently and needs
// its own starts, degrees and pool.
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
// the narrow partition, on this lane
// ---------------------------------------------------------------------------
// The second cut of the double lane's fits: region A's pieces are cut per order, and region B's
// seed is 5 pieces at degree 10 rather than one polynomial over the interval. A lane reading it
// therefore supplies the seed at the argument rather than the coefficients of a fixed shape.
//
// It is the double lane's partition; the float lane has one of its own below. A partition is a cut
// of a region rather than a property of a lane, so the two lanes' partitions are different fits
// and not one table read twice.
//
// __device__ and not __constant__: the narrow region-A pool is 3421 coefficients against the
// coarsest lane's 1876, and the two do not fit the 64 KB constant bank beside the float lane's
// tables. One fetch per piece per thread is the pattern the flat image above already serves from
// global memory.
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

// The float lane's own narrow partition, in the lane's coefficients and its degree: 218 pieces at
// degree 6 against the double lane's 311 at degree 10, so the pool is 1526 coefficients against
// 3421. Both are in global memory for the reason the double lane's is: the reads are per-thread
// piece lookups and not a uniform broadcast.
//
// The float lane's own narrow tables, stored at the degrees the fits were made at: this lane's
// region A and its piecewise region-B seed, each read whole.
constexpr int kNarrowPiecesTotal32 =
    detail::f32::kNarrowAPieceStartF32[detail::kMaxOrder + 1];
constexpr int kNarrowCoeffsTotal32 =
    kNarrowPiecesTotal32 * (detail::f32::kNarrowADegF32 + 1);
constexpr int kNarrowBCoeffsTotal32 =
    detail::f32::kNarrowBPiecesF32 * (detail::f32::kNarrowBDegF32 + 1);

__device__ int dNarrowAPieceStart32[detail::kMaxOrder + 2];
__device__ int dNarrowAOffset32[kNarrowPiecesTotal32];
__device__ float dNarrowAA32[kNarrowPiecesTotal32];
__device__ float dNarrowAB32[kNarrowPiecesTotal32];
__device__ int dNarrowAStoredDeg32[kNarrowPiecesTotal32];
__device__ float dNarrowACoeffs32[kNarrowCoeffsTotal32];
__device__ float dNarrowAMonoCoeffs32[kNarrowCoeffsTotal32];
__device__ float dNarrowBEdges32[detail::f32::kNarrowBPiecesF32 + 1];
__device__ float dNarrowBCoeffs32[kNarrowBCoeffsTotal32];
__device__ float dNarrowBMonoCoeffs32[kNarrowBCoeffsTotal32];

// The float lane's rational region-B seed, in the lane's own coefficients and at the lane's own
// degree: one pair per partition the lane carries. Region A's pairs are not here - the float lanes
// seed region A from the double lane's table, which the seed lane argument below carries - so the
// float rational tables this lane reads are region B's alone.
//
// The coarsest partition holds one pair over the whole of region B, read at the interval's own
// mapped argument; the narrow partition holds one per piece, read at the piece's own, and its two
// blocks sit in one pool with the denominator above the STORED numerator's position - the reading
// DeviceRatSum32 and the host's RationalSeedNarrowF32AtCut share.
__device__ float dRatBnum32[detail::f32::kRatBnumDeg + 1];
__device__ float dRatBden32[detail::f32::kRatBdenDeg];

constexpr int kNarrowRatBCoeffsTotal32F32 =
    static_cast<int>(std::size(detail::f32::kNarrowRatBCoeffsF32));

__device__ float dNarrowRatBCoeffs32[kNarrowRatBCoeffsTotal32F32];
__device__ int dNarrowRatBOffset32[detail::f32::kNarrowRatBPiecesCountF32];
__device__ int dNarrowRatBStoredNumDeg32[detail::f32::kNarrowRatBPiecesCountF32];
__device__ int dNarrowRatBDenDeg32[detail::f32::kNarrowRatBPiecesCountF32];

// ---------------------------------------------------------------------------
// the monomial scheme's stored form, on this lane
// ---------------------------------------------------------------------------
// The same fits in the other basis: every piece table above is carried in both forms
// (boys_coefficients.hpp), so this family stores no second fit and needs no second partition - it
// is the pools and the summation, and the pieces, edges, counts and stored degrees are the
// Chebyshev lanes' own.
//
// All of it global rather than constant memory: the double lane's Chebyshev pool and the float
// lane's tables already fill most of the 64 KB constant bank, so a second double pool of the same
// size does not fit beside them.
__device__ double dMonoCoeffs[kMaxCoeffs];
__device__ double dMonoBcoeffs[24];
// The float lane's coarsest pool in the other basis, the reading Lane32MonoFull makes of what
// Lane64MonoFull reads here. The same offsets as the Chebyshev pool above, so the pieces, edges,
// degrees and counts stay the float lane's own.
__device__ float dMonoCoeffs32[kMaxCoeffs];
__device__ float dMonoBcoeffs32[24];
__device__ double dNarrowAMonoCoeffs[kNarrowCoeffsTotal];
__device__ double dNarrowBMonoCoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)];

// ---------------------------------------------------------------------------
// the rational route's stored form, on this lane
// ---------------------------------------------------------------------------
// The other route's tables: a numerator/denominator pair per piece rather than one polynomial, over
// the same pieces - the route reads the coarsest partition's own intervals and the narrow
// partition's, so no piece table, edge or count here is new. What is new is what a piece's
// coefficients are: the numerator ascending from p_0, then the denominator's q_1..q_k with q_0 held
// at one, so the denominator's base is the STORED numerator degree's position
// (boys_impl.hpp RationalPieceAtCut). The pair is evaluated as two Horner sums and a division.
//
// All of it global memory, for the reason the pools above are: the constant bank is full.
constexpr int kRatACoeffsTotal = static_cast<int>(std::size(detail::kRatACoeffs));
constexpr int kNarrowRatACoeffsTotal = static_cast<int>(std::size(detail::kNarrowRatACoeffs));
constexpr int kNarrowRatBCoeffsTotal = static_cast<int>(std::size(detail::kNarrowRatBCoeffs));
constexpr int kRatBPool = static_cast<int>(std::size(detail::kRatBnum));
constexpr int kRatBPoolDen = static_cast<int>(std::size(detail::kRatBden));

// The coarsest partition's pairs, indexed as the Chebyshev pieces are: a row per order, a column per
// piece within the order. The two coefficient blocks are addressed separately: the denominator's
// base is the STORED numerator degree's position, which is why it is carried as an offset rather
// than computed at the read.
__device__ double dRatACoeffs[kRatACoeffsTotal];
__device__ int dRatAOffset[detail::kMaxOrder + 1][kMaxPieces];
__device__ int dRatADenOffset[detail::kMaxOrder + 1][kMaxPieces];
__device__ int dRatANumDeg[detail::kMaxOrder + 1][kMaxPieces];
__device__ int dRatADenDeg[detail::kMaxOrder + 1][kMaxPieces];
__device__ double dRatBnum[kRatBPool];
__device__ double dRatBden[kRatBPoolDen];

// The uniform route's own table: one grid over [0, kFlatHi), every interval the same width, every
// order fitted on its own at that interval's own degree. Interval-major, so one argument's whole
// ladder is one contiguous run and a thread reads it in cache-line steps. Both forms of every fit
// are stored at the same offsets, chosen by the scheme a call names.
//
// The intervals do not all carry the same degree, so the table has no stride and its size is the
// header's own stored count and not a product of one degree written out again:
// kFlatOffsets[kFlatIntervals] is that count, and the generator asserts it against the coefficient
// array's length. The size is read from that array here because the device copy is uploaded from
// it, so the two cannot come to disagree.
//
// Global memory and not the constant bank: the stored count is well past what this card's 64 KiB
// bank holds, and a warp of arguments spans several intervals of the grid, so a constant-bank read
// would serialize. The pool is read-only after the upload and is the same size on every device.
constexpr int kFlatPool = static_cast<int>(std::size(detail::kFlatCoeffs));
__device__ double dFlatCoeffs[kFlatPool];
__device__ double dFlatMonoCoeffs[kFlatPool];

// The two tables the read rule names: interval iv's block is (kFlatDegs[iv] + 1) coefficients per
// order and starts at kFlatOffsets[iv]. They are uploaded from the header's arrays rather than
// recomputed here, so the device and the host cannot disagree about a cell's shape.
constexpr int kFlatCells = static_cast<int>(std::size(detail::kFlatDegs));
constexpr int kFlatCellOffsets = static_cast<int>(std::size(detail::kFlatOffsets));
__device__ int dFlatDegs[kFlatCells];
__device__ int dFlatOffsets[kFlatCellOffsets];

// The float lane's own uniform table, on the lane's own grid: derived from this lane's bound, this
// lane's format and this lane's read cap, so its width, interval count and degrees are the double
// lane's only by coincidence. A reader that took the double lane's grid under this lane's name
// would be handed a table fitted to a bound this lane's arithmetic cannot reach.
constexpr int kFlatPoolF32 = static_cast<int>(std::size(detail::f32::kFlatCoeffsF32));
__device__ float dFlatCoeffsF32[kFlatPoolF32];
__device__ float dFlatMonoCoeffsF32[kFlatPoolF32];

constexpr int kFlatCellsF32 = static_cast<int>(std::size(detail::f32::kFlatDegsF32));
constexpr int kFlatCellOffsetsF32 = static_cast<int>(std::size(detail::f32::kFlatOffsetsF32));
__device__ int dFlatDegsF32[kFlatCellsF32];
__device__ int dFlatOffsetsF32[kFlatCellOffsetsF32];

// The same two grids on their rational route. Pool and four per-interval columns are one image: a
// reader handed the pairs without the degrees, the stored counts or the block starts would address
// a row at a length the table does not have, which is the substitution the readiness test below
// refuses. The pool's rows are intervals here and not orders, and its stride is the interval's own
// stored count.
constexpr int kFlatRatPool = static_cast<int>(std::size(detail::kFlatRatCoeffs));
__device__ double dFlatRatCoeffs[kFlatRatPool];

constexpr int kFlatRatCells = static_cast<int>(std::size(detail::kFlatRatNumDeg));
constexpr int kFlatRatCellOffsets = static_cast<int>(std::size(detail::kFlatRatOffsets));
__device__ int dFlatRatNumDeg[kFlatRatCells];
__device__ int dFlatRatDenDeg[kFlatRatCells];
__device__ int dFlatRatStored[kFlatRatCells];
__device__ int dFlatRatOffsets[kFlatRatCellOffsets];

constexpr int kFlatRatPoolF32 = static_cast<int>(std::size(detail::f32::kFlatRatCoeffsF32));
__device__ float dFlatRatCoeffsF32[kFlatRatPoolF32];

constexpr int kFlatRatCellsF32 = static_cast<int>(std::size(detail::f32::kFlatRatNumDegF32));
constexpr int kFlatRatCellOffsetsF32 =
    static_cast<int>(std::size(detail::f32::kFlatRatOffsetsF32));
__device__ int dFlatRatNumDegF32[kFlatRatCellsF32];
__device__ int dFlatRatDenDegF32[kFlatRatCellsF32];
__device__ int dFlatRatStoredF32[kFlatRatCellsF32];
__device__ int dFlatRatOffsetsF32[kFlatRatCellOffsetsF32];

// The same route's metadata in the flat form a caller's kernel can index: the device image's one
// convention is pieceStart[order] + piece, so the row-major tables above are re-cut once at the
// upload rather than read with a stride the handle would have to state and keep in step with this
// file. The route's own pool and region-B pair are flat already; the per-piece tables need the
// second image.
__device__ int dRatOffsetFlat[kPiecesTotal];
__device__ int dRatDenOffsetFlat[kPiecesTotal];
__device__ int dRatNumDegFlat[kPiecesTotal];
__device__ int dRatDenDegFlat[kPiecesTotal];

// The narrow partition's, flat over its rows as its Chebyshev counterpart is.
__device__ double dNarrowRatACoeffs[kNarrowRatACoeffsTotal];
__device__ int dNarrowRatAOffset[kNarrowPiecesTotal];
__device__ int dNarrowRatADenOffset[kNarrowPiecesTotal];
__device__ int dNarrowRatANumDeg[kNarrowPiecesTotal];
__device__ int dNarrowRatADenDeg[kNarrowPiecesTotal];
__device__ double dNarrowRatBCoeffs[kNarrowRatBCoeffsTotal];
__device__ int dNarrowRatBOffset[detail::kNarrowBPieces];
// One stored numerator degree per piece: the degree the numerator's read sums to and the position
// the denominator's q_1 sits above.
__device__ int dNarrowRatBStoredNumDeg[detail::kNarrowBPieces];
__device__ int dNarrowRatBDenDeg[detail::kNarrowBPieces];

// ---------------------------------------------------------------------------
// the lanes: what the kernels below hand the shared arithmetic
// ---------------------------------------------------------------------------
// boys_cuda_arithmetic.hpp holds one body per (precision, shape) and takes the tables as a lane
// object rather than reading a symbol, so the kernels here and the device-callable entries a
// caller's own kernel calls run the same code. These six lane objects are the kernels' side of
// that: the __constant__ tables of this translation unit.
//
// The six lane objects below are the kernels' side of the shared bodies: the __constant__ and
// __device__ tables of this translation unit, read at the degrees they were stored with. The
// batch lanes read the order-0 region-B entry because the F0 seed's error reaches every output
// with gain at most 1 + 1.846e-17, and the per-order region-A entry at the batch's top order.

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

// The coarsest partition in the other basis, the float lane's reading of what Lane64MonoFull reads
// on the double lane: region B's seed is the same seed's monomial form, as it is there.
//
// The call sites place this lane in the region-B seat alone, so the five region-A members of the
// lane interface (boys_cuda_arithmetic.hpp) are not stated here. A body that reaches for one of
// them fails to compile and names it, which is what makes leaving the half out safe.
//
// kMonomial stays, and it is the one member that rule does not cover: the LaneMonomial trait reads
// it and tolerates its absence, so dropping it would be a silent change rather than a named one.
// The lane interface asks a lane whose coefficients are the monomial pool's to state it
// (boys_cuda_arithmetic.hpp), and this lane's are.
struct Lane32MonoFull {
    static constexpr bool kMonomial = true;

    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) / static_cast<float>(detail::kX1 - detail::kX0) - 1.0f;
        return detail::DeviceHornerMono32(dMonoBcoeffs32, cBDeg32, t);
    }
};

// The narrow partition's lane, reading the degrees the partition was stored at.
//
// Region B's seed is piecewise here and the batch shape reads it at order 0, so BSeed ignores the
// order it is passed.
struct Lane64Narrow {
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
        return dNarrowAStoredDeg[dNarrowAPieceStart[order] + piece];
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
                                           detail::kNarrowBDeg,
                                           t);
    }
};

// The monomial scheme's lanes: the coarsest lanes' pieces and degrees with the coefficients read
// from the monomial pool and the piece summed by Horner. The kMonomial member is the whole of the
// difference the shared bodies see: they choose the summation with it (boys_cuda_arithmetic.hpp)
// and read the degrees the lane hands them either way.
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

// The narrow partition in the monomial basis, as Lane64Narrow carries the Chebyshev one, reading
// the stored degrees. Region B is piecewise here and read at order 0.
struct Lane64NarrowMono {
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
        return dNarrowAStoredDeg[dNarrowAPieceStart[order] + piece];
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
            detail::kNarrowBDeg,
            t);
    }
};

// ---------------------------------------------------------------------------
// the float lane's narrow partition
// ---------------------------------------------------------------------------
// The same shape of lane one lane down, over the float lane's own pieces. The call sites place both
// in the region-B seat alone, so their five region-A members are not stated here; a body that
// reaches for one fails to compile and names it.
//
// The Chebyshev lane reads the float narrow pool and sums each piece by a split Clenshaw; the
// monomial one reads the same partition's other form by Horner. kMonomial stays on the monomial
// lane for the reason the coarsest monomial lane above gives: the trait that reads it tolerates its
// absence, so removing it is not a compile-checked change.
struct Lane32Narrow {
    __device__ __forceinline__ float BSeed(float x, int) const {
        int piece = 0;

        while (piece + 1 < detail::f32::kNarrowBPiecesF32 && x >= dNarrowBEdges32[piece + 1])
        {
            ++piece;
        }

        const float a = dNarrowBEdges32[piece];
        const float b = dNarrowBEdges32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;

        return detail::DeviceClenshawSplit32(
            dNarrowBCoeffs32 + piece * (detail::f32::kNarrowBDegF32 + 1),
            detail::f32::kNarrowBDegF32,
            t);
    }
};

struct Lane32NarrowMono {
    static constexpr bool kMonomial = true;

    __device__ __forceinline__ float BSeed(float x, int) const {
        int piece = 0;

        while (piece + 1 < detail::f32::kNarrowBPiecesF32 && x >= dNarrowBEdges32[piece + 1])
        {
            ++piece;
        }

        const float a = dNarrowBEdges32[piece];
        const float b = dNarrowBEdges32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;

        return detail::DeviceHornerMono32(
            dNarrowBMonoCoeffs32 + piece * (detail::f32::kNarrowBDegF32 + 1),
            detail::f32::kNarrowBDegF32,
            t);
    }
};

// ---------------------------------------------------------------------------
// the rational route's lanes
// ---------------------------------------------------------------------------
// The other family's pieces: the same partitions, intervals, region structure and bodies, with a
// piece summed as a pair (boys_cuda_arithmetic.hpp). The kRational member tells the shared bodies
// so, exactly as kMonomial does for the scheme.
//
// The route's arithmetic is per reading, and the reading is the body a shape calls rather than a
// table it reads: the arguments shape seeds at the top order's piece and recurses down, paying that
// piece's own w(b), and the orders shape reads each order's own piece, paying A = 1. Region B is
// one pair read at order 0 either way.

struct Lane64RatFull {
    static constexpr bool kRational = true;

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
        return dRatACoeffs + dRatAOffset[order][piece];
    }

    __device__ __forceinline__ const double* DenCoeffs(int order, int piece) const {
        return dRatACoeffs + dRatADenOffset[order][piece];
    }

    __device__ __forceinline__ int NumDeg(int order, int piece) const {
        return dRatANumDeg[order][piece];
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        return dRatADenDeg[order][piece];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceRatSum(dRatBnum, detail::kRatBnumDeg, dRatBden, detail::kRatBdenDeg, t);
    }
};

// The narrow partition's pairs, read at the degrees the table stores. Region B's seed is a pair
// per piece here.
struct Lane64NarrowRat {
    static constexpr bool kRational = true;

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
        return dNarrowRatACoeffs + dNarrowRatAOffset[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* DenCoeffs(int order, int piece) const {
        return dNarrowRatACoeffs + dNarrowRatADenOffset[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ int NumDeg(int order, int piece) const {
        return dNarrowRatANumDeg[dNarrowAPieceStart[order] + piece];
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        return dNarrowRatADenDeg[dNarrowAPieceStart[order] + piece];
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
        const double* num = dNarrowRatBCoeffs + dNarrowRatBOffset[piece];

        return detail::DeviceRatSum(num,
                                    dNarrowRatBStoredNumDeg[piece],
                                    num + dNarrowRatBStoredNumDeg[piece] + 1,
                                    dNarrowRatBDenDeg[piece],
                                    t);
    }
};

// The float lane's rational route. It carries region B alone: the float lanes seed region A from
// the double lane's table, so the pair a float lane is asked for is the region-B seed and there is
// no float region-A pair in this lane to read.
//
// The two members below are the coarsest and the narrow partition's seed, each reading the tables
// its partition stores (dRatBnum32/dRatBden32, or the narrow pool with its denominator above the
// numerator) through the float form of the route's summation, at the degrees those tables store.
struct Lane32Rat {
    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) /
                            static_cast<float>(detail::kX1 - detail::kX0) -
                        1.0f;

        return detail::DeviceRatSum32(dRatBnum32,
                                      detail::f32::kRatBnumDeg,
                                      dRatBden32,
                                      detail::f32::kRatBdenDeg,
                                      t);
    }
};

struct Lane32NarrowRat {
    __device__ __forceinline__ float BSeed(float x, int) const {
        int piece = 0;

        while (piece + 1 < detail::f32::kNarrowRatBPiecesCountF32 &&
               x >= dNarrowBEdges32[piece + 1])
        {
            ++piece;
        }

        const float a = dNarrowBEdges32[piece];
        const float b = dNarrowBEdges32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const float* c = dNarrowRatBCoeffs32 + dNarrowRatBOffset32[piece];
        const int m = dNarrowRatBStoredNumDeg32[piece];
        const int k = dNarrowRatBDenDeg32[piece];

        return detail::DeviceRatSum32(c, m, c + m + 1, k, t);
    }
};

// --------------------------------------------------------------------------- kernels
// --------------------------------------------------------------------------- Each of these is one
// thread's worth of index arithmetic around one call into the shared bodies of
// boys_cuda_arithmetic.hpp - the same bodies the device-callable entries run.
//
// __restrict__ on x/out: the buffers are distinct DeviceBuffers by construction, and without it
// every out store would force nvcc to reload x (assumed aliasing).
template <DivisionForm kForm, bool kFastExp>
__global__ void BoysSingleF32Kernel(const int* n,
                                    const double* __restrict__ x,
                                    float* __restrict__ out,
                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF32<kForm, kFastExp>(Lane32Full{}, n[i],
                                                      static_cast<float>(x[i]));
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32Kernel(const int* n,
                                       const double* __restrict__ x,
                                       float* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

// The uniform-order entry: one nmax for the whole batch, so the recursion
// bounds are warp-uniform and no order array is read.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllNF32Kernel(int nmax,
                                  const double* __restrict__ x,
                                  float* __restrict__ out,
                                  size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      nmax,
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

// The single-precision each-order shape (see BoysEachOrderF64Kernel).
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysEachOrderF32Kernel(const int* n,
                                       const double* __restrict__ x,
                                       const int* __restrict__ offset,
                                       float* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const size_t base = static_cast<size_t>(offset[i]);

    detail::DeviceAllOrdersF32<kForm, kFastExp>(
        Lane64Full{}, Lane32Full{}, n[i], static_cast<float>(x[i]),
        [&](int l, float v) { out[base + l] = v; });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysSingleF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = detail::DeviceSingleF64<kForm, kFastExp>(Lane64Full{}, n[i], x[i]);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64Full{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

// The uniform-order entry (see BoysAllNF32Kernel).
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllNF64Kernel(int nmax, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64Full{}, nmax, x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

// The each-order shape, launched: every argument's ladder to that argument's own order, written
// contiguously from the caller's own offset array rather than spread over the padded planes
// BoysAllOrdersF64Kernel writes. The body is the ladder body above - the same lane, the same
// recurrence, the same seed - handed a sink that indexes by the argument's offset; nothing here is
// a second arithmetic. Only offset[i] .. offset[i] + n[i] of `out` is written.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysEachOrderF64Kernel(const int* n,
                                       const double* __restrict__ x,
                                       const int* __restrict__ offset,
                                       double* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const size_t base = static_cast<size_t>(offset[i]);

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64Full{}, n[i], x[i],
                                                [&](int l, double v) { out[base + l] = v; });
}

// ---------------------------------------------------------------------------
// the uniform route's ladder
// ---------------------------------------------------------------------------
// One thread per argument, the whole ladder to that argument's own order, the output order-major as
// the entries above write it.
//
// Below the join every order is summed from its own coefficients and none is built from another, so
// the 33 sums a top-order argument walks are 33 independent chains rather than one recurrence: that
// independence is the whole of what this route buys, and it is why the pool is read at a fixed
// stride. At and above the join no fit reaches and the call falls to the one-term asymptotic and
// its own upward recurrence - the arm region C runs, read from the header's own prefactor and
// stepping by the same (l + 1/2)/x.
//
// Thread i owns argument i, so its stores to out[l * count + i] are coalesced across the warp while
// it walks its own interval's 2112 contiguous bytes. Consecutive lanes of a warp therefore usually
// sit in different intervals, so the pool's read is not uniform across a warp; that is a property
// of an argument stream and not of this kernel.
//
// The ladder itself is the header's (DeviceAllOrdersF64Flat): the pools are the whole of what this
// route reads, so a kernel holding them by their local symbols and a caller holding them through
// the handle run one body.
template <DivisionForm kForm, bool kMonomial>
__global__ void BoysAllOrdersF64FlatKernel(const int* n,
                                           const double* x,
                                           double* out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    double* o = out + i;

    detail::DeviceAllOrdersF64Flat<kForm, kMonomial>(dFlatCoeffs, dFlatMonoCoeffs, dFlatDegs,
                                                     dFlatOffsets, n[i], x[i], [&](int, double v) {
                                                         *o = v;
                                                         o += count;
                                                     });
}

// The same grid on its rational route. The structure is the kernel above - one thread per argument,
// the whole ladder to that argument's own order, the output order-major - and what differs is the
// table it steps: one numerator/denominator pair per interval, addressed at the interval's own
// stored count, so the kernel closes over the pool and the four per-interval columns.
//
// No basis parameter: the route is a family and not a basis, so both scheme names a caller may use
// reach this one kernel.
template <DivisionForm kForm>
__global__ void BoysAllOrdersF64FlatRatKernel(const int* n,
                                              const double* x,
                                              double* out,
                                              size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    double* o = out + i;

    detail::DeviceAllOrdersF64FlatRat<kForm>(dFlatRatCoeffs, dFlatRatNumDeg, dFlatRatDenDeg,
                                             dFlatRatStored, dFlatRatOffsets, n[i], x[i],
                                             [&](int, double v) {
                                                 *o = v;
                                                 o += count;
                                             });
}

// ---------------------------------------------------------------------------
// the float lane's uniform route
// ---------------------------------------------------------------------------
// One thread per argument on the float lane's own grid, the whole ladder to that argument's own
// order, the output order-major as every entry above writes it. The structure is the double
// route's - read each order from its own fit, fall to the asymptotic at the join - and the
// arithmetic is the lane's: every sum is a float one, and the mapped argument is the float lane's
// own spelling rather than a narrowing of the double lane's.
//
// The ladder is the header's (DeviceAllOrdersF32Flat), for the same cause the double kernel above
// gives: the two pools and the grid's two per-interval tables are the whole of what this route
// reads.
template <DivisionForm kForm, bool kMonomial>
__global__ void BoysAllOrdersF32FlatKernel(const int* n,
                                           const double* x,
                                           float* out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    // The entry takes doubles and the kernel narrows itself, as every float entry of this lane
    // does: the argument measured at is the float it evaluates at.
    float* o = out + i;

    detail::DeviceAllOrdersF32Flat<kForm, kMonomial>(dFlatCoeffsF32,
                                                     dFlatMonoCoeffsF32,
                                                     dFlatDegsF32,
                                                     dFlatOffsetsF32,
                                                     n[i],
                                                     static_cast<float>(x[i]),
                                                     [&](int, float v) {
                                                         *o = v;
                                                         o += count;
                                                     });
}

// The float lane's grid on its rational route: the double kernel above at this lane's grid, tables
// and arithmetic, and with no basis parameter for the same reason.
template <DivisionForm kForm>
__global__ void BoysAllOrdersF32FlatRatKernel(const int* n,
                                              const double* x,
                                              float* out,
                                              size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    float* o = out + i;

    detail::DeviceAllOrdersF32FlatRat<kForm>(dFlatRatCoeffsF32, dFlatRatNumDegF32,
                                             dFlatRatDenDegF32,
                                             dFlatRatStoredF32, dFlatRatOffsetsF32, n[i],
                                             static_cast<float>(x[i]),
                                             [&](int, float v) {
                                                 *o = v;
                                                 o += count;
                                             });
}

// ---------------------------------------------------------------------------
// the float lane's narrow partition
// ---------------------------------------------------------------------------
// The same body as the coarsest float batch entry with the partition named: the float lane's own
// narrow pieces in region A and its own narrow region-B seed, read one fit at a time.
//
// Region A's seed is the DOUBLE lane's piece table - the downward recursion amplifies a float seed
// error past the float budget, which is why DeviceAllOrdersF32 takes a seed lane at all - and the
// region-B seed is the float lane's own, the split the coarsest entry (Lane64Full beside Lane32Full)
// already makes. Here the double lane's parts are its narrow partition's, so the entry is the
// narrow route rather than a mixed one.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowKernel(const int* n,
                                             const double* __restrict__ x,
                                             float* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Narrow{},
                                      Lane32Narrow{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 float* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64NarrowMono{},
                                      Lane32NarrowMono{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

// The float lane's coarsest partition in the monomial basis, the reading the double lane's
// BoysAllOrdersF64MonoKernel makes of its own: the same body, the seed lane the double lane's
// coarsest monomial one, and the float lane's coarsest monomial pair beside it.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32MonoKernel(const int* n,
                                           const double* __restrict__ x,
                                           float* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64MonoFull{},
                                      Lane32MonoFull{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

// The float lane's rational route, one kernel per partition it is carried on. The seed lane is the
// double lane's rational pair at that partition, which is what makes the two lanes' region-A seeds
// one reading, and the float lane supplies the region-B seed. The route has no second scheme: its
// pair is stored in monomial form and read by Horner, so both scheme names reach one kernel.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32RatKernel(const int* n,
                                          const double* __restrict__ x,
                                          float* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64RatFull{},
                                      Lane32Rat{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64NarrowRat{},
                                      Lane32NarrowRat{},
                                      n[i],
                                      static_cast<float>(x[i]),
                                      [&](int l, float v) { out[l * count + i] = v; });
}

// ---------------------------------------------------------------------------
// fp16 lane (the certified mixed-precision extension; BoysFp16 seam).
// __half I/O around the fp32 engine above -- the same __constant__ tables, the same
// region-partitioned dispatch, the same bodies.
// ---------------------------------------------------------------------------
#if BoysFp16
template <DivisionForm kForm>
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
        detail::DeviceSingleF32<kForm, false>(Lane32Full{}, n[i], __half2float(x[i])));
}

// The same value read through the fast region-B exponential, which is the option the float single
// entry carries (DeviceSingleF32's second parameter). The half lane's form of it: the arithmetic is
// the float lane's fast reading, unaltered, and only the store is the half one.
template <DivisionForm kForm>
__global__ void BoysSingleF16FastKernel(const int* n,
                                        const __half* __restrict__ x,
                                        __half* __restrict__ out,
                                        size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = __float2half(
        detail::DeviceSingleF32<kForm, true>(Lane32Full{}, n[i], __half2float(x[i])));
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF16Kernel(const int* n,
                                       const __half* __restrict__ x,
                                       __half* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      n[i],
                                      __half2float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2half(v);
                                      });
}

// The uniform-order entry (see BoysAllNF64Kernel).
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllNF16Kernel(int nmax,
                                  const __half* __restrict__ x,
                                  __half* __restrict__ out,
                                  size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      nmax,
                                      __half2float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2half(v);
                                      });
}

// The half lane's each-order shape: the float kernel above with this lane's __half store around the
// same body (see BoysEachOrderF64Kernel for the layout and BoysAllOrdersF16Kernel for the store).
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysEachOrderF16Kernel(const int* n,
                                       const __half* __restrict__ x,
                                       const int* __restrict__ offset,
                                       __half* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const size_t base = static_cast<size_t>(offset[i]);

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{}, Lane32Full{}, n[i],
                                                __half2float(x[i]), [&](int l, float v) {
                                                    out[base + l] = __float2half(v);
                                                });
}

// The lane's other bodies, which are the float lane's bodies: the half lane's
// own definition is that it runs the float engine's arithmetic and stores what
// it returns, so an entry of it is the float kernel beside it with __half I/O
// around the same body, the same seed lane and the same lane. Nothing below is
// a second arithmetic - the seed lane and the lane of each launcher are the
// ones the float kernel of that name hands its body (src/boys_cuda.cu, the
// block above), and the float lane's own tables are what the grid kernels read.
//
// The seed lane's first argument is the double lane's piece table on the float
// bodies (DeviceAllOrdersF32's own comment), which is why a half kernel names a
// Lane64 seed: it is not a widening of this lane, it is the float lane's seed.
template <DivisionForm kForm, typename SeedLane, typename Lane, bool kFastExp = false>
__global__ void BoysAllOrdersF16LaneKernel(const int* n,
                                           const __half* __restrict__ x,
                                           __half* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(SeedLane{},
                                      Lane{},
                                      n[i],
                                      __half2float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2half(v);
                                      });
}

// The float lane's grid, both stored bases of it, in the half lane's store.
template <DivisionForm kForm, bool kMonomial>
__global__ void BoysAllOrdersF16FlatKernel(const int* n,
                                           const __half* __restrict__ x,
                                           __half* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32Flat<kForm, kMonomial>(dFlatCoeffsF32,
                                                     dFlatMonoCoeffsF32,
                                                     dFlatDegsF32,
                                                     dFlatOffsetsF32,
                                                     n[i],
                                                     __half2float(x[i]),
                                                     [&](int l, float v) {
                                                         out[l * count + i] = __float2half(v);
                                                     });
}

template <DivisionForm kForm>
__global__ void BoysAllOrdersF16FlatRatKernel(const int* n,
                                              const __half* __restrict__ x,
                                              __half* __restrict__ out,
                                              size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32FlatRat<kForm>(dFlatRatCoeffsF32,
                                             dFlatRatNumDegF32,
                                             dFlatRatDenDegF32,
                                             dFlatRatStoredF32,
                                             dFlatRatOffsetsF32,
                                             n[i],
                                             __half2float(x[i]),
                                             [&](int l, float v) {
                                                 out[l * count + i] = __float2half(v);
                                             });
}

// The orders reading, over the same seed/lane pairs the float lane's orders
// kernels name.
template <DivisionForm kForm, typename SeedLane, typename Lane, bool kFastExp = false>
__global__ void BoysOrdersF16LaneKernel(const int* n,
                                        const __half* __restrict__ x,
                                        __half* __restrict__ out,
                                        size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceOrdersF32<kForm, kFastExp>(SeedLane{},
                                   Lane{},
                                   n[i],
                                   __half2float(x[i]),
                                   [&](int l, float v) { out[l * count + i] = __float2half(v); });
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// the orders axis and the narrow partition
// ---------------------------------------------------------------------------
// One kernel per shape, as the all-orders block above has. The orders axis is a body choice and
// the partition is a lane choice, so the two compose: the last pair is the narrow partition read
// with the orders axis.
//
// The orders axis is a choice inside region A only: past kX0 these kernels run the certified
// all-orders body, whose row is the lane's own bound over the whole range.
template <DivisionForm kForm, bool kFastExp = false, typename Lane>
__device__ __forceinline__ void DeviceOrdersBody(
    const Lane& lane, int order, double xx, double* out, size_t count, size_t i) {
    detail::DeviceOrdersF64<kForm, kFastExp>(lane, order, xx,
                                   [&](int l, double v) { out[l * count + i] = v; });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64OrdersKernel(const int* n,
                                             const double* __restrict__ x,
                                             double* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64Full{}, n[i], x[i], out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowKernel(const int* n,
                                             const double* __restrict__ x,
                                             double* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64Narrow{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowOrdersKernel(const int* n,
                                                   const double* __restrict__ x,
                                                   double* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64Narrow{}, n[i], x[i], out, count, i);
}

// ---------------------------------------------------------------------------
// the evaluation scheme: the same fits in the monomial basis
// ---------------------------------------------------------------------------
// The scheme axis is a lane choice and not a body choice - the pieces, the edges, the region
// structure and the shape are the coarsest ones, and what changes is the pool a piece's
// coefficients come from and the summation they are read by (boys_cuda_arithmetic.hpp) - so these
// are the same four shapes with a monomial lane.
//
// The orders axis composes with it too: DeviceOrdersBody takes the lane, so the axis is a choice
// inside region A whichever basis that lane sums.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64MonoKernel(const int* n,
                                           const double* __restrict__ x,
                                           double* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64MonoFull{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64OrdersMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 double* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64MonoFull{}, n[i], x[i], out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 double* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64NarrowMono{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowOrdersMonoKernel(const int* n,
                                                       const double* __restrict__ x,
                                                       double* __restrict__ out,
                                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64NarrowMono{}, n[i], x[i], out, count, i);
}

// ---------------------------------------------------------------------------
// the fit route: the same fits as rational minimax pairs
// ---------------------------------------------------------------------------
// The route axis is a lane choice like the scheme axis: the pieces, edges, partitions, region
// structure and shapes are the coarsest ones, and what changes is what a piece's coefficients are -
// a numerator/denominator pair summed by two Horner passes and a division
// (boys_cuda_arithmetic.hpp) - so these are the same four shapes with a rational lane.
//
// Each shape appears twice because the route's arithmetic is per reading: the arguments shape seeds
// at its top order's piece and carries that piece's w(b), and the orders shape reads each order's
// own piece at A = 1, so the two read the same stored pairs two ways.
//
// The scheme axis is inert on this route: the pair is stored once, in one basis, so both scheme
// names launch this same kernel.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64RatKernel(const int* n,
                                          const double* __restrict__ x,
                                          double* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64RatFull{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64OrdersRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64RatFull{}, n[i], x[i], out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64<kForm, kFastExp>(Lane64NarrowRat{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF64NarrowOrdersRatKernel(const int* n,
                                                      const double* __restrict__ x,
                                                      double* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody<kForm, kFastExp>(Lane64NarrowRat{}, n[i], x[i], out, count, i);
}

// ---------------------------------------------------------------------------
// the orders axis on the float lane
// ---------------------------------------------------------------------------
// The float lane's mirror of the orders-axis block above, one kernel per shape and the same body:
// region A read as one fit per order rather than seeded at the top order and brought down a
// recurrence, and past kX0 the certified all-orders body of the shape the kernel names.
//
// The seed lane is the double lane's, the pairing every float lane kernel of this file already
// makes (Lane64Full beside Lane32Full): the region-A seed is computed in double and rounded once on
// entry to the recursion, because the downward recursion amplifies a float seed error past the
// float budget. So each kernel below hands DeviceOrdersF32 a lane object this file already runs and
// a piece table this lane already stores - the axis is a reading of the same stored fits and not a
// new fit.
template <DivisionForm kForm, typename SeedLane, typename Lane, bool kFastExp = false>
__device__ __forceinline__ void DeviceOrdersBody32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, float* out, size_t count,
    size_t i) {
    detail::DeviceOrdersF32<kForm, kFastExp>(seedLane, lane, order, xx,
                                   [&](int l, float v) { out[l * count + i] = v; });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32OrdersKernel(const int* n,
                                             const double* __restrict__ x,
                                             float* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64Full, Lane32Full, kFastExp>(
        Lane64Full{}, Lane32Full{}, n[i], static_cast<float>(x[i]), out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowOrdersKernel(const int* n,
                                                   const double* __restrict__ x,
                                                   float* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64Narrow, Lane32Narrow, kFastExp>(
        Lane64Narrow{}, Lane32Narrow{}, n[i], static_cast<float>(x[i]), out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowOrdersMonoKernel(const int* n,
                                                       const double* __restrict__ x,
                                                       float* __restrict__ out,
                                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64NarrowMono, Lane32NarrowMono, kFastExp>(
        Lane64NarrowMono{}, Lane32NarrowMono{}, n[i], static_cast<float>(x[i]), out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32OrdersMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 float* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64MonoFull, Lane32MonoFull, kFastExp>(
        Lane64MonoFull{}, Lane32MonoFull{}, n[i], static_cast<float>(x[i]), out, count, i);
}

// The fit route's orders shapes. The pair is stored once and read by the two readings: this one
// reads each order's own piece at A = 1, which is what an order's own value is, where the
// per-argument shape seeds at its top order's piece and carries that piece's w(b) down the
// recursion.
template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32OrdersRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64RatFull, Lane32Rat, kFastExp>(
        Lane64RatFull{}, Lane32Rat{}, n[i], static_cast<float>(x[i]), out, count, i);
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersF32NarrowOrdersRatKernel(const int* n,
                                                      const double* __restrict__ x,
                                                      float* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32<kForm, Lane64NarrowRat, Lane32NarrowRat, kFastExp>(
        Lane64NarrowRat{}, Lane32NarrowRat{}, n[i], static_cast<float>(x[i]), out, count, i);
}

// ---------------------------------------------------------------------------
// host layer: plain exported functions (no library C++ types - this TU
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
        // The fit route's own blocks, at the same offsets: a piece's numerator and denominator are
        // one run of the pool and the route's metadata says where each begins. The two blocks are
        // addressed separately: the denominator's start is where the stored numerator's degree put
        // it.
        int ratOffset[33][kMaxPieces] = {};
        int ratDenOffset[33][kMaxPieces] = {};
        int ratNumDeg[33][kMaxPieces] = {};
        int ratDenDeg[33][kMaxPieces] = {};
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
        // The fit route's metadata in the same flat form, for the reason the
        // symbols state.
        int flatRatOffset[kPiecesTotal] = {};
        int flatRatDenOffset[kPiecesTotal] = {};
        int flatRatNumDeg[kPiecesTotal] = {};
        int flatRatDenDeg[kPiecesTotal] = {};
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
                ratOffset[o][index] = detail::kRatAOffset[p];
                ratDenOffset[o][index] =
                    detail::kRatAOffset[p] + detail::kRatANumDeg[p] + 1;
                ratNumDeg[o][index] = detail::kRatANumDeg[p];
                ratDenDeg[o][index] = detail::kRatADenDeg[p];
                flatRatOffset[p] = ratOffset[o][index];
                flatRatDenOffset[p] = ratDenOffset[o][index];
                flatRatNumDeg[p] = ratNumDeg[o][index];
                flatRatDenDeg[p] = ratDenDeg[o][index];

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

        // The monomial scheme's pools. Only the pool is copied: the pieces, their edges and their
        // degrees are the ones above, which is what the two forms of a fit share.
        if (cudaMemcpyToSymbol(dMonoCoeffs, monoCoeffs, sizeof(monoCoeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dMonoBcoeffs, detail::kMonoBcoeffs.data(),
                               sizeof(detail::kMonoBcoeffs)) != cudaSuccess)
        {
            return 2;
        }

        // The fit route's pool, metadata and region-B pair. The pool is copied as the header stores
        // it, and region B is the two arrays the header
        // keeps, so it needs no per-piece table here.
        if (cudaMemcpyToSymbol(dRatACoeffs,
                               detail::kRatACoeffs.data(),
                               sizeof(detail::kRatACoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatAOffset, ratOffset, sizeof(ratOffset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatADenOffset, ratDenOffset, sizeof(ratDenOffset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatANumDeg, ratNumDeg, sizeof(ratNumDeg)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatADenDeg, ratDenDeg, sizeof(ratDenDeg)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatBnum,
                               detail::kRatBnum.data(),
                               sizeof(detail::kRatBnum)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatBden,
                               detail::kRatBden.data(),
                               sizeof(detail::kRatBden)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatOffsetFlat, flatRatOffset, sizeof(flatRatOffset)) !=
                cudaSuccess ||
            cudaMemcpyToSymbol(dRatDenOffsetFlat, flatRatDenOffset, sizeof(flatRatDenOffset)) !=
                cudaSuccess ||
            cudaMemcpyToSymbol(dRatNumDegFlat, flatRatNumDeg, sizeof(flatRatNumDeg)) !=
                cudaSuccess ||
            cudaMemcpyToSymbol(dRatDenDegFlat, flatRatDenDeg, sizeof(flatRatDenDeg)) !=
                cudaSuccess)
        {
            return 2;
        }

        // The float lane's rational region-B seed over the same partition. The header keeps it as
        // two arrays exactly as the double lane's, so it is copied the same way; the float lane's
        // region-A pair is the double lane's and is uploaded above with the rest of the route.
        if (cudaMemcpyToSymbol(dRatBnum32,
                               detail::f32::kRatBnum.data(),
                               sizeof(detail::f32::kRatBnum)) != cudaSuccess ||
            cudaMemcpyToSymbol(dRatBden32,
                               detail::f32::kRatBden.data(),
                               sizeof(detail::f32::kRatBden)) != cudaSuccess)
        {
            return 2;
        }

        // The uniform route's pool, both forms of every fit, copied as the header stores them, and
        // the grid's two per-interval tables beside them. The cells carry the degrees the grid's
        // own cell law placed them at, and this pool is the whole of what an entry of this route
        // reads.
        //
        // The coefficients and the two tables are one image: a reader given the coefficients and a
        // stale degree would sum a cell's block at another cell's length, so a copy that left
        // either table behind would be a table the kernel reads wrongly rather than one it does not
        // find.
        if (cudaMemcpyToSymbol(dFlatCoeffs,
                               detail::kFlatCoeffs.data(),
                               sizeof(detail::kFlatCoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dFlatMonoCoeffs,
                               detail::kFlatMonoCoeffs.data(),
                               sizeof(detail::kFlatMonoCoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dFlatDegs,
                               detail::kFlatDegs.data(),
                               sizeof(detail::kFlatDegs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dFlatOffsets,
                               detail::kFlatOffsets.data(),
                               sizeof(detail::kFlatOffsets)) != cudaSuccess)
        {
            return 2;
        }
    }

    // Float lane.
    {
        float coeffs[kMaxCoeffs] = {};
        // The coarsest pool in the monomial basis, packed at the offsets of the Chebyshev one, so
        // the pieces, edges, degrees and counts above are the two forms' common reading.
        float monoCoeffs[kMaxCoeffs] = {};
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
                    monoCoeffs[runningOffset + k] = detail::f32::kMonoCoeffs[piece.offset + k];
                }

                runningOffset += piece.deg + 1;
            }

            count[o] = last - first;
        }

        if (cudaMemcpyToSymbol(cCoeffs32, coeffs, sizeof(coeffs)) != cudaSuccess)
        {
            return 2;
        }

        // The monomial scheme's pools for the float lane, copied as the double lane's are: only
        // the pool, at the offsets above, and the seed's monomial form beside it.
        if (cudaMemcpyToSymbol(dMonoCoeffs32, monoCoeffs, sizeof(monoCoeffs)) != cudaSuccess)
        {
            return 2;
        }

        if (cudaMemcpyToSymbol(dMonoBcoeffs32,
                               detail::f32::kMonoBcoeffs.data(),
                               sizeof(detail::f32::kMonoBcoeffs)) != cudaSuccess)
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

    // The float lane's narrow partition, laid out as the double lane's is: one pool per form and
    // one row per piece, a row's flat index being its position in kNarrowAPiecesF32. The stored
    // degrees travel with the pieces, one per piece.
    {
        int start[detail::kMaxOrder + 2] = {};
        int offset[kNarrowPiecesTotal32] = {};
        float a[kNarrowPiecesTotal32] = {};
        float b[kNarrowPiecesTotal32] = {};
        int stored[kNarrowPiecesTotal32] = {};
        float coeffs[kNarrowCoeffsTotal32] = {};
        float monoCoeffs[kNarrowCoeffsTotal32] = {};
        float edges[detail::f32::kNarrowBPiecesF32 + 1] = {};
        float bcoeffs[kNarrowBCoeffsTotal32] = {};
        float monoBcoeffs[kNarrowBCoeffsTotal32] = {};

        for (int o = 0; o <= detail::kMaxOrder + 1; ++o)
        {
            start[o] = detail::f32::kNarrowAPieceStartF32[o];
        }

        for (int p = 0; p < kNarrowPiecesTotal32; ++p)
        {
            const detail::f32::OrderPiece& piece =
                detail::f32::kNarrowAPiecesF32[static_cast<std::size_t>(p)];

            if (piece.offset < 0 || piece.offset + piece.deg >= kNarrowCoeffsTotal32)
            {
                return 1;
            }

            offset[p] = piece.offset;
            a[p] = piece.a;
            b[p] = piece.b;
            stored[p] = piece.deg;

            for (int k = 0; k <= piece.deg; ++k)
            {
                coeffs[piece.offset + k] =
                    detail::f32::kNarrowACoeffsF32[static_cast<std::size_t>(piece.offset + k)];
                monoCoeffs[piece.offset + k] =
                    detail::f32::kNarrowAMonoCoeffsF32[static_cast<std::size_t>(piece.offset + k)];
            }
        }

        for (int p = 0; p <= detail::f32::kNarrowBPiecesF32; ++p)
        {
            edges[p] = detail::f32::kNarrowBEdgesF32[static_cast<std::size_t>(p)];
        }

        for (int k = 0; k < kNarrowBCoeffsTotal32; ++k)
        {
            bcoeffs[k] = detail::f32::kNarrowBcoeffsF32[static_cast<std::size_t>(k)];
            monoBcoeffs[k] = detail::f32::kNarrowBMonoCoeffsF32[static_cast<std::size_t>(k)];
        }

        if (cudaMemcpyToSymbol(dNarrowAPieceStart32, start, sizeof(start)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAOffset32, offset, sizeof(offset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAA32, a, sizeof(a)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAB32, b, sizeof(b)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAStoredDeg32, stored, sizeof(stored)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowACoeffs32, coeffs, sizeof(coeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowAMonoCoeffs32, monoCoeffs, sizeof(monoCoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBEdges32, edges, sizeof(edges)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBCoeffs32, bcoeffs, sizeof(bcoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowBMonoCoeffs32, monoBcoeffs, sizeof(monoBcoeffs)) !=
                cudaSuccess)
        {
            return 2;
        }
    }

    // The float lane's rational region-B seed over that same partition: one pair per piece, at the
    // piece's own two degrees. The pieces, their edges and their intervals are the narrow Chebyshev
    // partition's own, so what is uploaded here is the pairs and their degrees and nothing else.
    {
        int offset[detail::f32::kNarrowRatBPiecesCountF32] = {};
        int numDeg[detail::f32::kNarrowRatBPiecesCountF32] = {};
        int denDeg[detail::f32::kNarrowRatBPiecesCountF32] = {};

        for (int p = 0; p < detail::f32::kNarrowRatBPiecesCountF32; ++p)
        {
            const detail::f32::RatPiece& piece =
                detail::f32::kNarrowRatBPiecesF32[static_cast<std::size_t>(p)];

            // The pair occupies offset .. offset + numdeg, then the denominator's q_1..q_k above
            // it, so the last coefficient a read touches is offset + numdeg + 1 + (dendeg - 1).
            if (piece.offset < 0 ||
                piece.offset + piece.numdeg + 1 + piece.dendeg > kNarrowRatBCoeffsTotal32F32)
            {
                return 1;
            }

            offset[p] = piece.offset;
            numDeg[p] = piece.numdeg;
            denDeg[p] = piece.dendeg;
        }

        if (cudaMemcpyToSymbol(dNarrowRatBCoeffs32,
                               detail::f32::kNarrowRatBCoeffsF32.data(),
                               sizeof(detail::f32::kNarrowRatBCoeffsF32)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBOffset32, offset, sizeof(offset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBStoredNumDeg32, numDeg, sizeof(numDeg)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBDenDeg32, denDeg, sizeof(denDeg)) != cudaSuccess)
        {
            return 2;
        }
    }

    // The float lane's uniform table, both forms of every fit, copied as the header stores them and
    // for the reason the double lane's is.
    if (cudaMemcpyToSymbol(dFlatCoeffsF32,
                           detail::f32::kFlatCoeffsF32.data(),
                           sizeof(detail::f32::kFlatCoeffsF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatMonoCoeffsF32,
                           detail::f32::kFlatMonoCoeffsF32.data(),
                           sizeof(detail::f32::kFlatMonoCoeffsF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatDegsF32,
                           detail::f32::kFlatDegsF32.data(),
                           sizeof(detail::f32::kFlatDegsF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatOffsetsF32,
                           detail::f32::kFlatOffsetsF32.data(),
                           sizeof(detail::f32::kFlatOffsetsF32)) != cudaSuccess)
    {
        return 2;
    }

    // The two grids' rational route, uploaded as the same kind of one image and for the same
    // reason: the pool and its four per-interval columns are read by one body, and a reader given
    // the pool without them would address a row at another interval's length.
    if (cudaMemcpyToSymbol(dFlatRatCoeffs,
                           detail::kFlatRatCoeffs.data(),
                           sizeof(detail::kFlatRatCoeffs)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatNumDeg,
                           detail::kFlatRatNumDeg.data(),
                           sizeof(detail::kFlatRatNumDeg)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatDenDeg,
                           detail::kFlatRatDenDeg.data(),
                           sizeof(detail::kFlatRatDenDeg)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatStored,
                           detail::kFlatRatStored.data(),
                           sizeof(detail::kFlatRatStored)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatOffsets,
                           detail::kFlatRatOffsets.data(),
                           sizeof(detail::kFlatRatOffsets)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatCoeffsF32,
                           detail::f32::kFlatRatCoeffsF32.data(),
                           sizeof(detail::f32::kFlatRatCoeffsF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatNumDegF32,
                           detail::f32::kFlatRatNumDegF32.data(),
                           sizeof(detail::f32::kFlatRatNumDegF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatDenDegF32,
                           detail::f32::kFlatRatDenDegF32.data(),
                           sizeof(detail::f32::kFlatRatDenDegF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatStoredF32,
                           detail::f32::kFlatRatStoredF32.data(),
                           sizeof(detail::f32::kFlatRatStoredF32)) != cudaSuccess ||
        cudaMemcpyToSymbol(dFlatRatOffsetsF32,
                           detail::f32::kFlatRatOffsetsF32.data(),
                           sizeof(detail::f32::kFlatRatOffsetsF32)) != cudaSuccess)
    {
        return 2;
    }

    // The cudaMemcpyToSymbol calls above are asynchronous (default stream); sync before the guard
    // flips so a concurrent launch on a non-default stream can never read partially uploaded
    // constant tables.
    const cudaError_t uploadSync = cudaStreamSynchronize(static_cast<cudaStream_t>(0));

    if (uploadSync != cudaSuccess)
    {
        return static_cast<int>(uploadSync);
    }

    // The narrow partition of the double lane's fits: one pool of coefficients and one row per
    // piece of the partition, laid out as the header stores it. A row's flat index is its position
    // in kNarrowAPieces, and each row carries the degree its fit was stored at.
    {
        int start[detail::kMaxOrder + 2] = {};
        int offset[kNarrowPiecesTotal] = {};
        double a[kNarrowPiecesTotal] = {};
        double b[kNarrowPiecesTotal] = {};
        int stored[kNarrowPiecesTotal] = {};
        double coeffs[kNarrowCoeffsTotal] = {};
        // The partition's own fits in the monomial basis, at the same offsets: the pieces, edges
        // and stored degrees above are the two forms' shared ones.
        double monoCoeffs[kNarrowCoeffsTotal] = {};
        double edges[detail::kNarrowBPieces + 1] = {};
        double bcoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)] = {};
        double monoBcoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)] = {};
        // The partition's fits in the rational route's form: a pair per row, at the header's own
        // offsets, and region B's pairs per piece above the shared offsets the Chebyshev seed does
        // not use.
        int ratOffset[kNarrowPiecesTotal] = {};
        int ratDenOffset[kNarrowPiecesTotal] = {};
        int ratNumDeg[kNarrowPiecesTotal] = {};
        int ratDenDeg[kNarrowPiecesTotal] = {};
        int ratBOffset[detail::kNarrowBPieces] = {};
        int ratBStored[detail::kNarrowBPieces] = {};
        int ratBDenDeg[detail::kNarrowBPieces] = {};

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
            ratOffset[p] = detail::kNarrowRatAOffset[static_cast<std::size_t>(p)];
            ratDenOffset[p] = detail::kNarrowRatAOffset[static_cast<std::size_t>(p)] +
                              detail::kNarrowRatANumDeg[static_cast<std::size_t>(p)] + 1;
            ratNumDeg[p] = detail::kNarrowRatANumDeg[static_cast<std::size_t>(p)];
            ratDenDeg[p] = detail::kNarrowRatADenDeg[static_cast<std::size_t>(p)];

            for (int k = 0; k <= piece.deg; ++k)
            {
                coeffs[piece.offset + k] = detail::kNarrowACoeffs[static_cast<std::size_t>(
                    piece.offset + k)];
                monoCoeffs[piece.offset + k] =
                    detail::kNarrowAMonoCoeffs[static_cast<std::size_t>(piece.offset + k)];
            }
        }

        for (int p = 0; p < detail::kNarrowBPieces; ++p)
        {
            ratBOffset[p] = detail::kNarrowRatBOffset[static_cast<std::size_t>(p)];
            ratBStored[p] = detail::kNarrowRatBNumDeg[static_cast<std::size_t>(p)];
            ratBDenDeg[p] = detail::kNarrowRatBDenDeg[static_cast<std::size_t>(p)];
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
            cudaMemcpyToSymbol(dNarrowBMonoCoeffs, monoBcoeffs, sizeof(monoBcoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatACoeffs,
                               detail::kNarrowRatACoeffs.data(),
                               sizeof(detail::kNarrowRatACoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatAOffset, ratOffset, sizeof(ratOffset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatADenOffset,
                               ratDenOffset,
                               sizeof(ratDenOffset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatANumDeg, ratNumDeg, sizeof(ratNumDeg)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatADenDeg, ratDenDeg, sizeof(ratDenDeg)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBCoeffs,
                               detail::kNarrowRatBCoeffs.data(),
                               sizeof(detail::kNarrowRatBCoeffs)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBOffset, ratBOffset, sizeof(ratBOffset)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBStoredNumDeg,
                               ratBStored,
                               sizeof(ratBStored)) != cudaSuccess ||
            cudaMemcpyToSymbol(dNarrowRatBDenDeg, ratBDenDeg, sizeof(ratBDenDeg)) != cudaSuccess)
        {
            return 2;
        }
    }

    gTablesDevice = device;
    return 0;
}

// The addresses of the flat device image, in the order the status layer fills BoysDeviceTables
// (boys_cuda.hpp): the double lane's pieceStart, offset, a, b, deg, coeffs and region-B seed, then
// the float lane's seven, then three slots no table occupies, then the uniform grid's four pools.
// status layer cannot name a CUDA symbol, so the order is the seam between this file and
// boys_cuda.cpp and both sides state it.
// Uploads first, so a caller that skipped InitializeTables still gets a table rather than a null
// pointer.
namespace {

// The address order, one entry per symbol, in the slot order the handle's fields
// are declared and the status layer fills them. A slot this revision has no table
// for holds nullptr, and the export writes the null the status layer then stores.
const void* const kTableAddressSymbols[] = {&dPieceStart,      &dOffset,
                              &dA,               &dB,
                              &dDeg,             &dCoeffs,
                              &dBSeed,           &dPieceStart32,
                              &dOffset32,        &dA32,
                              &dB32,             &dDeg32,
                              &dCoeffs32,        &dBSeed32,
                              nullptr,           nullptr,
                              nullptr,           &dFlatCoeffs,
                              &dFlatMonoCoeffs,  &dFlatCoeffsF32,
                              &dFlatMonoCoeffsF32,
                              &dNarrowAPieceStart, &dNarrowAOffset,
                              &dNarrowAA,        &dNarrowAB,
                              &dNarrowAStoredDeg, &dNarrowACoeffs,
                              &dNarrowAMonoCoeffs, &dNarrowBEdges,
                              &dNarrowBCoeffs,   &dNarrowBMonoCoeffs,
                              nullptr,           nullptr,
                              nullptr,           nullptr,
                              &dNarrowAPieceStart32, &dNarrowAOffset32,
                              &dNarrowAA32,      &dNarrowAB32,
                              &dNarrowAStoredDeg32, &dNarrowACoeffs32,
                              &dNarrowAMonoCoeffs32, &dNarrowBEdges32,
                              &dNarrowBCoeffs32, &dNarrowBMonoCoeffs32,
                              &dRatACoeffs,      &dRatOffsetFlat,
                              &dRatDenOffsetFlat, &dRatNumDegFlat,
                              &dRatDenDegFlat,   &dRatBnum,
                              &dRatBden,         nullptr,
                              nullptr,           &dRatBnum32,
                              &dRatBden32,
                              &dNarrowRatACoeffs, &dNarrowRatAOffset,
                              &dNarrowRatADenOffset, &dNarrowRatANumDeg,
                              &dNarrowRatADenDeg, nullptr,
                              &dNarrowRatBCoeffs,
                              &dNarrowRatBOffset, &dNarrowRatBStoredNumDeg,
                              &dNarrowRatBDenDeg, nullptr,
                              &dNarrowRatBCoeffs32, &dNarrowRatBOffset32,
                              &dNarrowRatBStoredNumDeg32, &dNarrowRatBDenDeg32,
                              nullptr,           nullptr,
                              nullptr,           nullptr,
                              &dFlatDegs,        &dFlatOffsets,
                              &dFlatDegsF32,     &dFlatOffsetsF32,
                              &dFlatRatCoeffs,   &dFlatRatNumDeg,
                              &dFlatRatDenDeg,   &dFlatRatStored,
                              &dFlatRatOffsets,
                              &dFlatRatCoeffsF32, &dFlatRatNumDegF32,
                              &dFlatRatDenDegF32, &dFlatRatStoredF32,
                              &dFlatRatOffsetsF32,
                              // The coarsest partition's monomial pools, in the order the handle
                              // states them (the double lane's pool, its region-B seed, the float
                              // lane's pool, that lane's seed): the four slots no named group above
                              // claims, appended as the append-only rule requires.
                              &dMonoCoeffs,
                              &dMonoBcoeffs,
                              &dMonoCoeffs32,
                              &dMonoBcoeffs32};
constexpr int kTableAddressCount =
    static_cast<int>(sizeof(kTableAddressSymbols) / sizeof(kTableAddressSymbols[0]));

// The slots BoysCudaDeviceTableAddresses fills: the handle's first twenty-one
// fields, which is the array a caller of the earlier surface already holds.
constexpr int kTableAddressFirstCount = 21;

static_assert(kTableAddressCount >= kTableAddressFirstCount,
              "the two calls that fill the address order must cover it in one direction: "
              "the first cannot hold more slots than the order has");

// The order's second half is exactly the handle's count of the fields those slots belong to. The
// status layer sizes the array it offers this export by that count, so the two sides state the same
// width and a table appended here without a field to receive it is a compile error rather than an
// address written past the end of the caller's array.
static_assert(kTableAddressCount - kTableAddressFirstCount == kBoysDeviceTablesTailCount,
              "the tail export's width must be the handle's own count of the symbols after "
              "the first twenty-one: one slot per symbol, in the order both sides state");

} // namespace

extern "C" int BoysCudaDeviceTableAddresses(void** out) {
    const int uploaded = BoysCudaUploadTables();

    if (uploaded != 0)
    {
        return uploaded;
    }

    // The element type is const void*, not void**: cudaGetSymbolAddress has a template overload
    // taking const T&, and a void** lvalue binds to it as T = void**, which hands the runtime the
    // address of the array slot instead of the address of the variable. A const void* value matches
    // the non-template overload, which is the one that resolves a symbol.
    //
    // Every table added after the original seventeen is appended rather than slotted in beside the
    // tables it belongs to, so that a slot a caller's build already reads keeps its index. Slots 17
    // to 20 are the uniform grid's four pools; 21 to 34 the narrow partition's double lane, 35 to
    // 44 its float lane; 45 to 55 the fit route on the coarsest partition, 56 to 66 on the narrow
    // one, 67 to 70 its float lane; 71 to 74 the uniform grid's two per-interval tables in each
    // lane. A new group goes after those and never inside them.
    //
    // The order is read by two calls: the first twenty-one slots are the ones a caller's build
    // already holds an array for, written by BoysCudaDeviceTableAddresses, and the rest by
    // BoysCudaDeviceTableAddressesTail. Splitting there keeps a caller not rebuilt for the wider
    // handle from being handed more addresses than its array holds.

    // The bound is the array's own length, so a pool added here cannot be
    // exported without one of the two loops below reaching it.
    //
    // A slot this revision has no table for holds nullptr, and the runtime has no
    // symbol to resolve for it: cudaGetSymbolAddress answers cudaErrorInvalidSymbol
    // for a null symbol, which would fail the whole call and leave the handle
    // unfilled. The null is written through instead, which is what the array's own
    // comment promises the caller's handle receives. The slots holding null are the
    // ones the accuracy axis's single member left behind - the tables they named are
    // gone from this revision, the handle's fields for them stay null, and no entry
    // reads one.
    for (int i = 0; i < kTableAddressFirstCount; ++i)
    {
        if (kTableAddressSymbols[i] == nullptr)
        {
            out[i] = nullptr;
            continue;
        }

        const cudaError_t got = cudaGetSymbolAddress(&out[i], kTableAddressSymbols[i]);

        if (got != cudaSuccess)
        {
            return 2;
        }
    }

    return 0;
}

extern "C" int BoysCudaDeviceTableAddressesTail(void** out) {
    const int uploaded = BoysCudaUploadTables();

    if (uploaded != 0)
    {
        return uploaded;
    }

    // The same call for the rest of the order, at the same element type and for the same reason.
    // The caller's array holds one entry per symbol from the first of these onwards, so slot
    // kTableAddressFirstCount + i lands at its index i. A null slot is written through rather
    // than resolved, for the reason the first export's loop gives.
    for (int i = kTableAddressFirstCount; i < kTableAddressCount; ++i)
    {
        if (kTableAddressSymbols[i] == nullptr)
        {
            out[i - kTableAddressFirstCount] = nullptr;
            continue;
        }

        const cudaError_t got =
            cudaGetSymbolAddress(&out[i - kTableAddressFirstCount], kTableAddressSymbols[i]);

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

// The one place a launcher's run-time form becomes a kernel's template argument, so every kernel
// holds one division and not three. The host layer validates the form before it gets here, so the
// arm below the switch exists only because a switch needs one.
template <typename Body>
int UnderForm(int form, Body body) {
    switch (static_cast<DivisionForm>(form))
    {
    case DivisionForm::kExactDivision:
        return body(std::integral_constant<DivisionForm, DivisionForm::kExactDivision>{});
    case DivisionForm::kPlainReciprocal:
        return body(std::integral_constant<DivisionForm, DivisionForm::kPlainReciprocal>{});
    case DivisionForm::kRefinedReciprocal:
        return body(std::integral_constant<DivisionForm, DivisionForm::kRefinedReciprocal>{});
    }

    return static_cast<int>(cudaErrorInvalidValue);
}

} // namespace

extern "C" int BoysCudaLaunchSingleF32(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF32Kernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchSingleF32Fast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF32Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The float lane's partition and grid, one launcher per arithmetic: the route is an argument of the
// caller's and not of the launch, so the choice is the symbol it names.
extern "C" int BoysCudaLaunchAllOrdersF32Uniform(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32FlatKernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32UniformHorner(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32FlatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The float lane's grid on the same route, one launcher for the reason above.
extern "C" int BoysCudaLaunchAllOrdersF32UniformRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32Narrow(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32Mono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32MonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The float lane's rational route, one launcher per partition: the two scheme names a caller may
// use differ in the report's row and not here, because the route's pair is stored in one form and
// the two names reach one kernel.
extern "C" int BoysCudaLaunchAllOrdersF32Rat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32RatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The float lane's orders axis: the same launches the rows above make, one per kernel of that axis.
// The rational route's rows have the one launcher each, and the uniform grid's two orders rows need
// no launcher of their own: its packing axis has one member and those rows launch
// BoysCudaLaunchAllOrdersF32Uniform and its Horner twin.
extern "C" int BoysCudaLaunchAllOrdersF32Orders(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrders(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF32(
    int form, int nmax, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF32Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchSingleF64(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF64Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64Uniform(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64FlatKernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The same launch over the same table's other basis. Two launchers and not one taking the basis as
// an argument: the basis is the arithmetic, and an argument would be a choice made at run time by
// the caller of a function whose whole contract is which numbers it delivers.
extern "C" int BoysCudaLaunchAllOrdersF64UniformHorner(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64FlatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The grid's rational route, one launcher and not two: the pair is stored in one form, so both
// scheme names a caller may use reach it. The orders-axis rows of that route are this same launch,
// because the route's packing axis has no second member to run.
extern "C" int BoysCudaLaunchAllOrdersF64UniformRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF64(
    int form, int nmax, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF64Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The each-order shape's two region-B readings, one launcher each for the reason the single lane's
// pair has two: the member is a compile-time choice of arithmetic and not a run-time argument.
extern "C" int BoysCudaLaunchEachOrderF64(
    int form,
    const int* n,
    const double* x,
    const int* offset,
    double* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF64Kernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, offset, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderF64Fast(
    int form,
    const int* n,
    const double* x,
    const int* offset,
    double* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF64Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, offset, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderF32(
    int form,
    const int* n,
    const double* x,
    const int* offset,
    float* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF32Kernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, offset, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderF32Fast(
    int form,
    const int* n,
    const double* x,
    const int* offset,
    float* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF32Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, offset, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

#if BoysFp16
// Async launch contract (uniform with the F32/F64 launchers): the kernel is queued on the caller's
// stream and the call returns once the launch is accepted - callers synchronize the stream before
// reading the outputs.
extern "C" int BoysCudaLaunchSingleF16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The half lane's fast region-B reading: the symbol a caller names is the choice here as it is one
// launcher up, and the choice is the float lane's fast exponential.
extern "C" int BoysCudaLaunchSingleF16Fast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF16FastKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF16(
    int form, int nmax, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                nmax, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The half lane's each-order shape, over the float lane's ladder body and this lane's store (see
// BoysEachOrderF64Kernel for the layout).
extern "C" int BoysCudaLaunchEachOrderF16(
    int form,
    const int* n,
    const void* x,
    const int* offset,
    void* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF16Kernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), offset, static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderF16Fast(
    int form,
    const int* n,
    const void* x,
    const int* offset,
    void* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderF16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), offset, static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The lane's other bodies, one launcher per body the float lane above launches:
// the symbol a caller names is the choice, and the half lane's choice is the
// same one (the launchers' own comment above the float block).
extern "C" int BoysCudaLaunchAllOrdersF16Narrow(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16Mono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16Rat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16Uniform(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16FlatKernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16UniformHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16FlatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16UniformRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16Orders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64Full, Lane32Full>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}
#endif // BoysFp16

extern "C" int BoysCudaLaunchAllOrdersF64Orders(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64Narrow(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrders(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The monomial scheme's four shapes.
extern "C" int BoysCudaLaunchAllOrdersF64Mono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64MonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersMonoKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The fit route's four shapes.
extern "C" int BoysCudaLaunchAllOrdersF64Rat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64RatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The double lane's ladders at the other region-B exponential, one launcher per kernel above whose
// entry the option space carries at both members of the axis. Each is the launcher of its own name
// with the last template argument alone moved: the body, the lane, the seed lane and the tables are
// the ones the launcher above states, so what these add is one instantiation and not an arithmetic.
// The region-B exponential is a compile-time choice and not a run-time argument, which is why the
// member is a second symbol rather than a parameter - the same reason the rational route's pairs
// have two rows and one kernel (the launcher above).
extern "C" int BoysCudaLaunchAllOrdersF64Fast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64MonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64MonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The rational route's ladders at the other region-B exponential. One launcher per kernel, and the
// two scheme names of a pair reach the one of their own partition as their accurate rows reach one
// (the launcher above): the pair is stored in one form, so a second scheme name selects no second
// arithmetic at either exponential.
extern "C" int BoysCudaLaunchAllOrdersF64RatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64RatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64OrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF64NarrowOrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The float lane's ladders at the other region-B exponential, the double lane's block above read
// at this lane's kernels: the same body, the same seed lane, the same stored tables, and the
// second template argument alone moved. One launcher per kernel this lane has, and the two scheme
// names of a pair reach the one of their partition as their accurate rows reach one, because the
// pair is stored in monomial form and a second scheme name selects no second arithmetic.
extern "C" int BoysCudaLaunchAllOrdersF32Fast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32MonoFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32MonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersMonoFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowMonoFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersMonoFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersMonoKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32RatFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32RatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersRatFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowRatFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersRatFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The single and all-N shapes at the other region-B exponential, one launcher per shape for the
// reason the ladders have one per kernel: the member is a compile-time choice of arithmetic, so
// the choice is a second symbol and not a parameter.
extern "C" int BoysCudaLaunchSingleF64Fast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleF64Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF64Fast(
    int form, int nmax, const double* x, double* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF64Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF32Fast(
    int form, int nmax, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF32Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(nmax, x, out,
                                                                                 count);
        return static_cast<int>(cudaGetLastError());
    });
}

#if BoysFp16
// The half lane's ladders at the other region-B exponential, its own block above read the way the
// float lane's block above it is: the same kernels, the same seed/lane pairs, the second argument
// of the pair moved. The lane's definition is that it runs the float engine's arithmetic and
// stores what it returns, so the member here is the float lane's fast reading and not a second
// half arithmetic.
extern "C" int BoysCudaLaunchAllOrdersF16Fast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64Full, Lane32Full, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16MonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16RatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNF16Fast(
    int form, int nmax, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNF16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                nmax, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}
#endif // BoysFp16

// The float lane's rational pair at the other region-B exponential: one symbol per surface
// the table books, over the kernel the pair's split-Clenshaw sibling launches. The pair is
// stored in one form, so the two names select one arithmetic.
extern "C" int BoysCudaLaunchAllOrdersF32RatHornerFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32RatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}
extern "C" int BoysCudaLaunchAllOrdersF32OrdersRatHornerFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32OrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}
extern "C" int BoysCudaLaunchAllOrdersF32NarrowRatHornerFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}
extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersRatHornerFast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF32NarrowOrdersRatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
        return static_cast<int>(cudaGetLastError());
    });
}

// ---------------------------------------------------------------------------
// the bfloat16 lane
// ---------------------------------------------------------------------------
// The half lane's block with the other format around it: the same float engine, the same
// seed lanes, the same stored tables, and a store into bfloat16 in place of fp16. The
// lane's definition is that it runs the float lane's arithmetic and stores what it returns,
// so an entry of it is the fp16 kernel of its own name with this lane's I/O around the same
// body. Region B's exponential is a template argument of the bodies below, and the single
// shape spells it as the two symbols the library offers, as that lane's does.
#if BoysFp16
template <DivisionForm kForm>
__global__ void BoysSingleBf16Kernel(const int* n,
                                    const __nv_bfloat16* __restrict__ x,
                                    __nv_bfloat16* __restrict__ out,
                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = __float2bfloat16(
        detail::DeviceSingleF32<kForm, false>(Lane32Full{}, n[i], __bfloat162float(x[i])));
}

template <DivisionForm kForm>
__global__ void BoysSingleBf16FastKernel(const int* n,
                                        const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ out,
                                        size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    out[i] = __float2bfloat16(
        detail::DeviceSingleF32<kForm, true>(Lane32Full{}, n[i], __bfloat162float(x[i])));
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllOrdersBf16Kernel(const int* n,
                                       const __nv_bfloat16* __restrict__ x,
                                       __nv_bfloat16* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      n[i],
                                      __bfloat162float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2bfloat16(v);
                                      });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysAllNBf16Kernel(int nmax,
                                  const __nv_bfloat16* __restrict__ x,
                                  __nv_bfloat16* __restrict__ out,
                                  size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{},
                                      Lane32Full{},
                                      nmax,
                                      __bfloat162float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2bfloat16(v);
                                      });
}

template <DivisionForm kForm, bool kFastExp = false>
__global__ void BoysEachOrderBf16Kernel(const int* n,
                                       const __nv_bfloat16* __restrict__ x,
                                       const int* __restrict__ offset,
                                       __nv_bfloat16* __restrict__ out,
                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const size_t base = static_cast<size_t>(offset[i]);

    detail::DeviceAllOrdersF32<kForm, kFastExp>(Lane64Full{}, Lane32Full{}, n[i],
                                                __bfloat162float(x[i]), [&](int l, float v) {
                                                    out[base + l] = __float2bfloat16(v);
                                                });
}

template <DivisionForm kForm, typename SeedLane, typename Lane, bool kFastExp = false>
__global__ void BoysAllOrdersBf16LaneKernel(const int* n,
                                           const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32<kForm, kFastExp>(SeedLane{},
                                      Lane{},
                                      n[i],
                                      __bfloat162float(x[i]),
                                      [&](int l, float v) {
                                          out[l * count + i] = __float2bfloat16(v);
                                      });
}

template <DivisionForm kForm, typename SeedLane, typename Lane, bool kFastExp = false>
__global__ void BoysOrdersBf16LaneKernel(const int* n,
                                        const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ out,
                                        size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceOrdersF32<kForm, kFastExp>(SeedLane{},
                                   Lane{},
                                   n[i],
                                   __bfloat162float(x[i]),
                                   [&](int l, float v) { out[l * count + i] = __float2bfloat16(v); });
}

template <DivisionForm kForm, bool kMonomial>
__global__ void BoysAllOrdersBf16FlatKernel(const int* n,
                                           const __nv_bfloat16* __restrict__ x,
                                           __nv_bfloat16* __restrict__ out,
                                           size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32Flat<kForm, kMonomial>(dFlatCoeffsF32,
                                                     dFlatMonoCoeffsF32,
                                                     dFlatDegsF32,
                                                     dFlatOffsetsF32,
                                                     n[i],
                                                     __bfloat162float(x[i]),
                                                     [&](int l, float v) {
                                                         out[l * count + i] = __float2bfloat16(v);
                                                     });
}

template <DivisionForm kForm>
__global__ void BoysAllOrdersBf16FlatRatKernel(const int* n,
                                              const __nv_bfloat16* __restrict__ x,
                                              __nv_bfloat16* __restrict__ out,
                                              size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32FlatRat<kForm>(dFlatRatCoeffsF32,
                                             dFlatRatNumDegF32,
                                             dFlatRatDenDegF32,
                                             dFlatRatStoredF32,
                                             dFlatRatOffsetsF32,
                                             n[i],
                                             __bfloat162float(x[i]),
                                             [&](int l, float v) {
                                                 out[l * count + i] = __float2bfloat16(v);
                                             });
}

extern "C" int BoysCudaLaunchSingleBf16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleBf16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNBf16(
    int form, int nmax, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNBf16Kernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                nmax, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderBf16(
    int form,
    const int* n,
    const void* x,
    const int* offset,
    void* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderBf16Kernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), offset, static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Orders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64Full, Lane32Full>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Narrow(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Uniform(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatKernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16UniformHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Rat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16UniformRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16RatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersRatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowRatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16UniformRatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Mono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersUniform(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatKernel<kForm(), false>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersUniformHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatKernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersUniformRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersUniformRatHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16FlatRatKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchSingleBf16Fast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysSingleBf16FastKernel<kForm()>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchEachOrderBf16Fast(
    int form,
    const int* n,
    const void* x,
    const int* offset,
    void* out,
    std::size_t count,
    void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysEachOrderBf16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), offset, static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16Fast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64Full, Lane32Full, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16RatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16RatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64Narrow, Lane32Narrow, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16MonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16OrdersMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64MonoFull, Lane32MonoFull, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersMonoFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowMono, Lane32NarrowMono, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersBf16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllNBf16Fast(
    int form, int nmax, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllNBf16Kernel<kForm(), true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                nmax, static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

// The rational pair's other scheme name at the other region-B exponential: one symbol per
// surface the table books, over the kernel the pair's split-Clenshaw sibling launches. The
// pair is stored in one form, so the two names select one arithmetic - the same sharing the
// accurate pair above makes, spelled per surface because each is booked as a row of its own.




extern "C" int BoysCudaLaunchAllOrdersF16RatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16OrdersRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64RatFull, Lane32Rat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysAllOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

extern "C" int BoysCudaLaunchAllOrdersF16NarrowOrdersRatHornerFast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream) {
    return UnderForm(form, [&](auto kForm) {
        BoysOrdersF16LaneKernel<kForm(), Lane64NarrowRat, Lane32NarrowRat, true>
            <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
                n, static_cast<const __half*>(x), static_cast<__half*>(out), count);
        return static_cast<int>(cudaGetLastError());
    });
}

#endif // BoysFp16

} // namespace

} // namespace boys
