// CUDA lane of the Boys kernel. Mirrors the scalar/simd design (boys.cpp, boys_simd.cpp):
// piecewise Chebyshev seeds + recurrences, region structure A/B/C with the weighted
// region-A fits (see tools/gen_boys_coefficients.py). C++20 with a CUDA-safe include
// list only: the library's C++23 headers would poison the nvcc translation unit.

#include "boys/boys_cuda_arithmetic.hpp"

// The handle's own declaration, for the count the address order's two halves are asserted
// against: the fields and the export order are one statement, and the assertion keeps them one.
#include "boys/boys_device_tables.hpp"

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

// The accuracy-multiplier effective-degree tables: one lane per CUDA role, in the order the
// host fills them (boys_cuda.cpp FillEffLane). Each all-orders lane serves both entries of its
// precision - the per-element orders and the uniform nmax - so the two read one degree table:
//   0 = double single (region-B degree per order)
//   1 = double batch (region-B degree = order-0 entry)
//   2 = float single (region-B degree per order)
//   3 = float batch (region-B degree = order-0 entry; the region-A seed is the DOUBLE piece
//       table, budget 1.5e-7)
//   4 = fp16 single (region-B degree per order)
//   5 = fp16 batch (region-B degree = order-0 entry; the region-A seed is the DOUBLE piece
//       table, budget 1e-7)
// Filled host-side once per (device, m) - the single-m-per-process cache, same idempotence
// contract as BoysCudaUploadTables. The relaxed kernels read these exactly as the kernels above
// read cDeg/cBDeg: the degree shape is unchanged, only the values differ.
constexpr int kEffLaneCount = 6;
__constant__ int cDegEff[kEffLaneCount][33][kMaxPieces];
__constant__ int cBDegEff[kEffLaneCount][33];

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

// The relaxed degrees, in the same flat form and one set for the whole device: one rung is
// resident at a time, so a table per rung is not what a caller can have. The stride is the larger
// of the two lanes' piece counts, so one lane axis serves both piece tables; a lane's own entries
// are indexed with the piece-start table its region-A seed reads its coefficients with, exactly
// as the full-accuracy table above is.
//
// The lane order is cDegEff's. Lanes 0, 1, 3 and 5 are indexed by the double piece table (the
// batch lanes' region-A seed is computed in double whatever precision they return -
// RoleUsesDoubleTables) and lanes 2 and 4 by the float one.
constexpr int kRelaxedPieces = kPiecesTotal > kPiecesTotal32 ? kPiecesTotal : kPiecesTotal32;

__device__ int dDegEff[kEffLaneCount * kRelaxedPieces];
__device__ int dBDegEff[kEffLaneCount * (detail::kMaxOrder + 1)];
// The rung the two arrays above are cut for, zero when none has been uploaded. Read through the
// handle a consumer holds rather than copied into it, so a handle filled for one rung reports
// itself retired when another rung replaces it instead of running tables since overwritten.
__device__ double dRelaxedM;

// ---------------------------------------------------------------------------
// the narrow partition, on this lane
// ---------------------------------------------------------------------------
// The second cut of the double lane's fits: region A's pieces are cut per order, and region B's
// seed is 5 pieces at degree 10 rather than one polynomial over the interval. A lane reading it
// therefore supplies the seed at the argument rather than the coefficients of a fixed shape.
//
// It is the double lane's partition; the float lane has one of its own below. A partition is a cut
// of a region rather than a property of a lane, so the two lanes' partitions are different fits
// and not one table read twice. The effective degrees a rung cuts this one to are derived for the
// roles that evaluate these fits (boys_effective_degrees.hpp), and the entries carrying this
// partition's rungs are the double ones.
//
// __device__ and not __constant__: the narrow region-A pool is 3421 coefficients against the
// shipped lane's 1876, and the two do not fit the 64 KB constant bank beside the float lane's
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
// The rung's cut of this partition is one table per basis, carried beside the stored pool and
// derived over it: this lane's region-B degrees in the Chebyshev form and in the monomial one
// (FillNarrowF32Lane, FillNarrowMonoF32Lane, boys_cuda.cpp). Region A carries none of its own
// here because this lane's region-A seed is the double lane's, whose cut the same upload carries.
// So the entries reading this partition are served at every rung of the lane in both of the bases
// it sums in (DeviceEntryServedAtRung, boys_cuda_options.hpp).
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
// The shipped partition holds one pair over the whole of region B, read at the interval's own
// mapped argument; the narrow partition holds one per piece, read at the piece's own, and its two
// blocks sit in one pool with the denominator above the STORED numerator's position - the reading
// DeviceRatSum32 and the host's RationalSeedNarrowF32AtCut share.
//
// The rung's cut of that seed is carried beside it, one table per partition, for the reason the
// double role's is: the criterion cuts the pair together, so the degrees a rung leaves are the
// pair's own and a call at a rung of this lane reads the same fit short the launched row reads.
// The numerator's cut is the first entry of a pair and the denominator's the second, as the double
// lane's dRatBDeg and dNarrowRatBDeg are laid out.
__device__ float dRatBnum32[detail::f32::kRatBnumDeg + 1];
__device__ float dRatBden32[detail::f32::kRatBdenDeg];
__device__ int dRatBDeg32[2];

constexpr int kNarrowRatBCoeffsTotal32F32 =
    static_cast<int>(std::size(detail::f32::kNarrowRatBCoeffsF32));

__device__ float dNarrowRatBCoeffs32[kNarrowRatBCoeffsTotal32F32];
__device__ int dNarrowRatBOffset32[detail::f32::kNarrowRatBPiecesCountF32];
__device__ int dNarrowRatBStoredNumDeg32[detail::f32::kNarrowRatBPiecesCountF32];
__device__ int dNarrowRatBDenDeg32[detail::f32::kNarrowRatBPiecesCountF32];
__device__ int dNarrowRatBDeg32[2 * detail::f32::kNarrowRatBPiecesCountF32];

// The rung's cut of the same partition, laid out as the shipped lanes' relaxed tables are: region A
// flat over the partition's rows, region B flat over (piece, order), matching
// boys_effective_degrees.hpp's derivation. It is one role's table - the double batch, the shape the
// entries carrying the partition have - so one table and not a lane axis.
//
// Written only by a relaxed upload, so a full-accuracy call reads the stored degrees above instead
// and is not served whatever rung is resident.
__device__ int dNarrowDegEff[kNarrowPiecesTotal];
__device__ int dNarrowBDegEff[detail::kNarrowBPieces * (detail::kMaxOrder + 1)];

// The float lane's own cut of the same rung, region B. Region A needs no table beside it: the float
// lane's region-A seed is the double lane's, so the pieces a rung cuts on the float lane are this
// lane's region-B pieces and those alone.
//
// A table of its own and not a second reader of the double lane's: this lane's region B is 4 pieces
// at degree 6 against the double lane's 5 at degree 10, so a cut derived over the double lane's
// coefficients is a cut of a fit this lane does not read.
__device__ int dNarrowBDegEff32[detail::f32::kNarrowBPiecesF32 * (detail::kMaxOrder + 1)];

// The other basis's cut of the same rung, over the same float pieces and the same region-B edges. A
// table of its own rather than a second reader of the one above: the two bases are two stored forms
// of one fit, and a degree the Chebyshev form's coefficients certify is the degree of a polynomial
// the monomial form's coefficients do not sum - the tail a rung drops is read from the coefficients
// the kernel reads, which is what NarrowRegionBDegrees' TailBasis argument is.
__device__ int dNarrowMonoBDegEff32[detail::f32::kNarrowBPiecesF32 * (detail::kMaxOrder + 1)];

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
__device__ double dNarrowAMonoCoeffs[kNarrowCoeffsTotal];
__device__ double dNarrowBMonoCoeffs[detail::kNarrowBPieces * (detail::kNarrowBDeg + 1)];

// The rung's cut, in the monomial basis: the same derivations as the Chebyshev tables above at the
// other form of the same table (TailBasis::kMonomial), so a rung's degrees are the ones that rung
// certifies for the pool and the summation this family reads. Region A is order-major as cDegEff
// is; region B is one row per order, read at the order-0 entry by the batch shape.
//
// The narrow partition's cut is one table each, as its Chebyshev counterpart's is.
__device__ int dMonoDegEff[detail::kMaxOrder + 1][kMaxPieces];
__device__ int dMonoBDegEff[detail::kMaxOrder + 1];
__device__ int dNarrowMonoDegEff[kNarrowPiecesTotal];
__device__ int dNarrowMonoBDegEff[detail::kNarrowBPieces * (detail::kMaxOrder + 1)];

// ---------------------------------------------------------------------------
// the rational route's stored form, on this lane
// ---------------------------------------------------------------------------
// The other route's tables: a numerator/denominator pair per piece rather than one polynomial, over
// the same pieces - the route reads the shipped partition's own intervals and the narrow
// partition's, so no piece table, edge or count here is new. What is new is what a piece's
// coefficients are: the numerator ascending from p_0, then the denominator's q_1..q_k with q_0 held
// at one, so the denominator's base is the STORED numerator degree's position and not the cut's
// (boys_impl.hpp RationalPieceAtCut). The pair is evaluated as two Horner sums and a division.
//
// The degrees are carried twice per piece: the pair the table was stored at, and the cut a rung's
// criterion certifies. A rung's cut is per reading as well as per partition - region A read one
// piece per order pays A = 1, and the same piece read as a batch seed pays its own w(b) - so the
// two shapes carry their own cut table and the argument shape's cells read the seed's.
//
// All of it global memory, for the reason the pools above are: the constant bank is full.
constexpr int kRatACoeffsTotal = static_cast<int>(std::size(detail::kRatACoeffs));
constexpr int kNarrowRatACoeffsTotal = static_cast<int>(std::size(detail::kNarrowRatACoeffs));
constexpr int kNarrowRatBCoeffsTotal = static_cast<int>(std::size(detail::kNarrowRatBCoeffs));
constexpr int kRatBPool = static_cast<int>(std::size(detail::kRatBnum));
constexpr int kRatBPoolDen = static_cast<int>(std::size(detail::kRatBden));

// The shipped partition's pairs, indexed as the Chebyshev pieces are: a row per order, a column per
// piece within the order. The two coefficient blocks are addressed separately because the cut moves
// the numerator's base - the denominator's is the STORED numerator degree's position, which is why
// it is carried as an offset rather than computed at the read.
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

// The rungs' cuts, one table per region-A reading; region B's is one pair either way, so the two
// readings share it. A cell is the pair a cut leaves - the numerator's degree then the
// denominator's - which is why every table here has a trailing two: the criterion derives the two
// together and a rung using one without the other would be a cut nobody certified.
__device__ int dRatSeedDeg[detail::kMaxOrder + 1][kMaxPieces][2];
__device__ int dRatOrdDeg[detail::kMaxOrder + 1][kMaxPieces][2];
__device__ int dRatBDeg[2];

// The same route's metadata in the flat form a caller's kernel can index: the device image's one
// convention is pieceStart[order] + piece, so the row-major tables above are re-cut once at the
// upload rather than read with a stride the handle would have to state and keep in step with this
// file. The route's own pool and region-B pair are flat already; the four per-piece tables and the
// cut of the one reading a device entry carries need the second image.
//
// A cut's two degrees are the two consecutive entries of one cell here as they are above, the
// numerator first. Only the seed's reading has an image: it is the reading of the ladder shape,
// the one shape this lane's device-callable entries carry.
__device__ int dRatOffsetFlat[kPiecesTotal];
__device__ int dRatDenOffsetFlat[kPiecesTotal];
__device__ int dRatNumDegFlat[kPiecesTotal];
__device__ int dRatDenDegFlat[kPiecesTotal];
__device__ int dRatSeedDegFlat[2 * kPiecesTotal];

// The narrow partition's, flat over its rows as its Chebyshev counterpart is.
__device__ double dNarrowRatACoeffs[kNarrowRatACoeffsTotal];
__device__ int dNarrowRatAOffset[kNarrowPiecesTotal];
__device__ int dNarrowRatADenOffset[kNarrowPiecesTotal];
__device__ int dNarrowRatANumDeg[kNarrowPiecesTotal];
__device__ int dNarrowRatADenDeg[kNarrowPiecesTotal];
__device__ int dNarrowRatSeedDeg[kNarrowPiecesTotal][2];
__device__ int dNarrowRatOrdDeg[kNarrowPiecesTotal][2];
__device__ double dNarrowRatBCoeffs[kNarrowRatBCoeffsTotal];
__device__ int dNarrowRatBOffset[detail::kNarrowBPieces];
// One stored numerator degree per piece: both the degree a full-accuracy read sums to and the
// position the denominator's q_1 sits above, which is what a cut leaves where it is.
__device__ int dNarrowRatBStoredNumDeg[detail::kNarrowBPieces];
__device__ int dNarrowRatBDenDeg[detail::kNarrowBPieces];
__device__ int dNarrowRatBDeg[detail::kNarrowBPieces][2];

// ---------------------------------------------------------------------------
// the lanes: what the kernels below hand the shared arithmetic
// ---------------------------------------------------------------------------
// boys_cuda_arithmetic.hpp holds one body per (precision, shape) and takes the tables as a lane
// object rather than reading a symbol, so the kernels here and the device-callable entries a
// caller's own kernel calls run the same code. These six lane objects are the kernels' side of
// that: the __constant__ tables of this translation unit.
//
// The relaxed lanes read their degrees from cDegEff/cBDegEff, in the lane order documented there.
// Which degree table a lane reads, and whether its region-B degree is the per-order entry or the
// order-0 one, is the only thing the relaxed lane objects differ in. The batch lanes read the
// order-0 region-B entry because the F0 seed's error reaches every output with gain at most
// 1 + 1.846e-17, and the per-order region-A entry at the batch's top order.

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

// The narrow partition's lane, in the two forms a rung has: kRelaxed false reads the degrees the
// partition was stored at, which no rung's upload touches, and true reads the rung's own cut. That
// is what keeps a full-accuracy call from being served a resident rung's table.
//
// Region B's seed is piecewise here and the batch shape reads it at order 0, so BSeed ignores the
// order it is passed.
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

// The monomial scheme's lanes: the shipped lanes' pieces and degrees with the coefficients read
// from the monomial pool and the piece summed by Horner. The kMonomial member is the whole of the
// difference the shared bodies see: they choose the summation with it (boys_cuda_arithmetic.hpp)
// and read the degrees the lane hands them either way, so a rung of this family is a cut of this
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

// The narrow partition in the monomial basis, in the two forms a rung has, as Lane64Narrow carries
// the Chebyshev one: kRelaxed false reads the stored degrees, true the rung's own cut of this
// basis. Region B is piecewise here and read at order 0.
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
// the float lane's narrow partition
// ---------------------------------------------------------------------------
// The same shape of lane one lane down, over the float lane's own pieces, and in the same two
// forms a rung has: the stored form reads the degrees the partition was stored at, and the relaxed
// form the rung's own cut of region B (the symbol block above carries both bases').
//
// The Chebyshev lane reads the float narrow pool and sums each piece by a split Clenshaw; the
// monomial one reads the same partition's other form by Horner, and the kMonomial member is the
// whole of the difference the shared bodies see.
struct Lane32Narrow {
    __device__ __forceinline__ int Count(int order) const {
        return dNarrowAPieceStart32[order + 1] - dNarrowAPieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return dNarrowAA32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return dNarrowAB32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return dNarrowACoeffs32 + dNarrowAOffset32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return dNarrowAStoredDeg32[dNarrowAPieceStart32[order] + piece];
    }

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

// The same lane at a rung: the stored coefficients, read at the degree that rung's cut left. The
// float lane's counterpart of Lane64Narrow<true>, for the same reason - a rung's arithmetic is the
// stored fit read short, and a call asking for a rung must not be answered with the stored degree.
// Region A is not here: this lane's region-A seed is the double lane's (Lane64Narrow), so the cut
// this object reads is region B's alone.
struct Lane32NarrowRelaxed {
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
            dNarrowBDegEff32[piece * (detail::kMaxOrder + 1)],
            t);
    }
};

// The other form of that rung: the monomial pool's coefficients read at the cut derived over those
// coefficients. The kMonomial member tells the shared bodies to sum by Horner, and the table it
// reads is the monomial one - a rung of the monomial form is a cut of that form's tail and not of
// the other's.
struct Lane32NarrowMonoRelaxed {
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
            dNarrowMonoBDegEff32[piece * (detail::kMaxOrder + 1)],
            t);
    }
};

struct Lane32NarrowMono {
    static constexpr bool kMonomial = true;

    __device__ __forceinline__ int Count(int order) const {
        return dNarrowAPieceStart32[order + 1] - dNarrowAPieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return dNarrowAA32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return dNarrowAB32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return dNarrowAMonoCoeffs32 + dNarrowAOffset32[dNarrowAPieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return dNarrowAStoredDeg32[dNarrowAPieceStart32[order] + piece];
    }

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
// The route's degree table is per piece AND per reading, so a lane names which reading its cut is:
// the arguments shape seeds at the top order's piece and recurses down, paying that piece's own
// w(b), and the orders shape reads each order's own piece, paying A = 1. Region B is one pair read
// at order 0 either way, so the two readings share its table.
enum class RatCut : int {
    kPerOrder = 0, // the argument shape's reading: the piece is the order's value
    kBatchSeed = 1, // the recursion's reading: the seed is carried down from the piece
};

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

// A rung of the shipped partition, at one of the route's two readings.
template <RatCut kCut> struct Lane64RatEff {
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
        return kCut == RatCut::kBatchSeed ? dRatSeedDeg[order][piece][0]
                                          : dRatOrdDeg[order][piece][0];
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        return kCut == RatCut::kBatchSeed ? dRatSeedDeg[order][piece][1]
                                          : dRatOrdDeg[order][piece][1];
    }

    __device__ __forceinline__ double BSeed(double x, int) const {
        const double t = 2.0 * (x - detail::kX0) / (detail::kX1 - detail::kX0) - 1.0;
        return detail::DeviceRatSum(dRatBnum, dRatBDeg[0], dRatBden, dRatBDeg[1], t);
    }
};

// The narrow partition, in the two forms a rung has and at either reading: kRelaxed false reads the
// pairs the table was stored at and true the rung's own cut, and kCut chooses which of the two cuts
// that is. Region B's seed is a pair per piece here, so the two readings share its cut.
template <bool kRelaxed, RatCut kCut> struct Lane64NarrowRat {
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
        const int flat = dNarrowAPieceStart[order] + piece;

        if constexpr (kRelaxed)
        {
            return kCut == RatCut::kBatchSeed ? dNarrowRatSeedDeg[flat][0]
                                              : dNarrowRatOrdDeg[flat][0];
        } else
        {
            return dNarrowRatANumDeg[flat];
        }
    }

    __device__ __forceinline__ int DenDeg(int order, int piece) const {
        const int flat = dNarrowAPieceStart[order] + piece;

        if constexpr (kRelaxed)
        {
            return kCut == RatCut::kBatchSeed ? dNarrowRatSeedDeg[flat][1]
                                              : dNarrowRatOrdDeg[flat][1];
        } else
        {
            return dNarrowRatADenDeg[flat];
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
        const double* num = dNarrowRatBCoeffs + dNarrowRatBOffset[piece];

        return detail::DeviceRatSum(num,
                                    kRelaxed ? dNarrowRatBDeg[piece][0]
                                             : dNarrowRatBStoredNumDeg[piece],
                                    num + dNarrowRatBStoredNumDeg[piece] + 1,
                                    kRelaxed ? dNarrowRatBDeg[piece][1]
                                             : dNarrowRatBDenDeg[piece],
                                    t);
    }
};

// The float lane's rational route. It carries region B alone: the float lanes seed region A from
// the double lane's table, so the pair a float lane is asked for is the region-B seed and there is
// no float region-A pair in this lane to read.
//
// The two members below are the shipped and the narrow partition's seed, each reading the tables
// its partition stores (dRatBnum32/dRatBden32, or the narrow pool with its denominator above the
// numerator) through the float form of the route's summation. Nothing here is a cut: these lanes
// are read at the full-accuracy multiplier only.
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

// The same two lanes at a rung: the stored pairs read at the degrees that rung's criterion left.
// The float counterparts of Lane64RatEff and Lane64NarrowRat<true, ...>, cut one table per
// partition rather than one per reading: region B is one seed either way, so both readings share
// it.
//
// The denominator's position is the STORED numerator's, in the narrow pool as in the full-accuracy
// lane above: a cut shortens the two blocks and does not move the second of them.
struct Lane32RatRelaxed {
    static constexpr bool kRational = true;

    __device__ __forceinline__ float BSeed(float x, int) const {
        const float t = 2.0f * (x - static_cast<float>(detail::kX0)) /
                            static_cast<float>(detail::kX1 - detail::kX0) -
                        1.0f;

        return detail::DeviceRatSum32(dRatBnum32, dRatBDeg32[0], dRatBden32, dRatBDeg32[1], t);
    }
};

struct Lane32NarrowRatRelaxed {
    static constexpr bool kRational = true;

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
        const float* const c = dNarrowRatBCoeffs32 + dNarrowRatBOffset32[piece];
        const int storedNum = dNarrowRatBStoredNumDeg32[piece];

        return detail::DeviceRatSum32(c,
                                      dNarrowRatBDeg32[2 * piece],
                                      c + storedNum + 1,
                                      dNarrowRatBDeg32[2 * piece + 1],
                                      t);
    }
};

// --------------------------------------------------------------------------- kernels
// --------------------------------------------------------------------------- Each of these is one
// thread's worth of index arithmetic around one call into the shared bodies of
// boys_cuda_arithmetic.hpp - the same bodies the device-callable entries run.
//
// __restrict__ on x/out: the buffers are distinct DeviceBuffers by construction, and without it
// every out store would force nvcc to reload x (assumed aliasing).
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

// The uniform-order entry (see BoysAllNF32Kernel).
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
template <bool kMonomial>
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

    detail::DeviceAllOrdersF64Flat<kMonomial>(dFlatCoeffs, dFlatMonoCoeffs, dFlatDegs,
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
// No form parameter: the route is a family and not a basis, so both scheme names a caller may use
// reach this one kernel.
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

    detail::DeviceAllOrdersF64FlatRat(dFlatRatCoeffs, dFlatRatNumDeg, dFlatRatDenDeg,
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
template <bool kMonomial>
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

    detail::DeviceAllOrdersF32Flat<kMonomial>(dFlatCoeffsF32, dFlatMonoCoeffsF32, dFlatDegsF32,
                                              dFlatOffsetsF32, n[i], static_cast<float>(x[i]),
                                              [&](int, float v) {
                                                  *o = v;
                                                  o += count;
                                              });
}

// The float lane's grid on its rational route: the double kernel above at this lane's grid, tables
// and arithmetic, and with no form parameter for the same reason.
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

    detail::DeviceAllOrdersF32FlatRat(dFlatRatCoeffsF32, dFlatRatNumDegF32, dFlatRatDenDegF32,
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
// The same body as the shipped float batch entry with the partition named: the float lane's own
// narrow pieces in region A and its own narrow region-B seed, read one fit at a time.
//
// Region A's seed is the DOUBLE lane's piece table - the downward recursion amplifies a float seed
// error past the float budget, which is why DeviceAllOrdersF32 takes a seed lane at all - and the
// region-B seed is the float lane's own, the split the shipped entry (Lane64Full beside Lane32Full)
// already makes. Here the double lane's parts are its narrow partition's, so the entry is the
// narrow route rather than a mixed one.
__global__ void BoysAllOrdersF32NarrowKernel(const int* n,
                                             const double* __restrict__ x,
                                             float* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Narrow<false>{},
                               Lane32Narrow{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32NarrowMonoKernel(const int* n,
                                                 const double* __restrict__ x,
                                                 float* __restrict__ out,
                                                 size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64NarrowMono<false>{},
                               Lane32NarrowMono{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

// The float lane's rational route, one kernel per partition it is carried on. The seed lane is the
// double lane's rational pair at that partition, which is what makes the two lanes' region-A seeds
// one reading, and the float lane supplies the region-B seed. The route has no second scheme: its
// pair is stored in monomial form and read by Horner, so both scheme names reach one kernel.
__global__ void BoysAllOrdersF32RatKernel(const int* n,
                                          const double* __restrict__ x,
                                          float* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64RatFull{},
                               Lane32Rat{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32NarrowRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64NarrowRat<false, RatCut::kBatchSeed>{},
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
// One per CUDA lane, mirroring the full-accuracy kernels above exactly (same arithmetic, same
// region structure, same output layout) except that the seed degrees come from
// cDegEff/cBDegEff - the m = 1 kernels stay byte-identical (the full-accuracy pin). The lane
// index is the kernel's identity, in the order documented at cDegEff. The batch lanes read the
// order-0 region-B entry and the per-order region-A entry at the batch's top order (the A_A(nmax)
// amplification covers the downward recursion) - exactly the CPU relaxed batches. The relaxed
// single lanes read their region-B degree per order.
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
// per-element twin does.
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
// the orders axis and the narrow partition
// ---------------------------------------------------------------------------
// Six kernels and not two, because each shape has a full-accuracy form and a rung's form, exactly
// as the all-orders kernel above does. The orders axis is a body choice and the partition is a
// lane choice, so the two compose: the last pair is the narrow partition read with the orders
// axis.
//
// The orders axis is a choice inside region A only: past kX0 these kernels run the certified
// all-orders body, whose row is the lane's own bound over the whole range.
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
// The scheme axis is a lane choice and not a body choice - the pieces, the edges, the region
// structure and the shape are the shipped ones, and what changes is the pool a piece's
// coefficients come from and the summation they are read by (boys_cuda_arithmetic.hpp) - so these
// are the same four shapes with a monomial lane, each with the full-accuracy form and the rung's.
//
// The orders axis composes with it too: DeviceOrdersBody takes the lane, so the axis is a choice
// inside region A whichever basis that lane sums.
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

// ---------------------------------------------------------------------------
// the fit route: the same fits as rational minimax pairs
// ---------------------------------------------------------------------------
// The route axis is a lane choice like the scheme axis: the pieces, edges, partitions, region
// structure and shapes are the shipped ones, and what changes is what a piece's coefficients are -
// a numerator/denominator pair summed by two Horner passes and a division
// (boys_cuda_arithmetic.hpp) - so these are the same four shapes with a rational lane.
//
// Each shape appears twice because the route's cut is per reading: the arguments shape seeds at its
// top order's piece and carries that piece's w(b), and the orders shape reads each order's own
// piece at A = 1, so the two name different cuts of the same stored pairs.
//
// The scheme axis is inert on this route: the pair is stored once, in one basis, so both scheme
// names launch this same kernel.
__global__ void BoysAllOrdersF64RatKernel(const int* n,
                                          const double* __restrict__ x,
                                          double* __restrict__ out,
                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64RatFull{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64RatKernelEff(const int* n,
                                             const double* __restrict__ x,
                                             double* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64RatEff<RatCut::kBatchSeed>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64OrdersRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64RatFull{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64OrdersRatKernelEff(const int* n,
                                                   const double* __restrict__ x,
                                                   double* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64RatEff<RatCut::kPerOrder>{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                double* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64NarrowRat<false, RatCut::kPerOrder>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowRatKernelEff(const int* n,
                                                   const double* __restrict__ x,
                                                   double* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF64(Lane64NarrowRat<true, RatCut::kBatchSeed>{}, n[i], x[i], [&](int l, double v) {
        out[l * count + i] = v;
    });
}

__global__ void BoysAllOrdersF64NarrowOrdersRatKernel(const int* n,
                                                      const double* __restrict__ x,
                                                      double* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64NarrowRat<false, RatCut::kPerOrder>{}, n[i], x[i], out, count, i);
}

__global__ void BoysAllOrdersF64NarrowOrdersRatKernelEff(const int* n,
                                                         const double* __restrict__ x,
                                                         double* __restrict__ out,
                                                         size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody(Lane64NarrowRat<true, RatCut::kPerOrder>{}, n[i], x[i], out, count, i);
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
// the orders axis on the float lane
// ---------------------------------------------------------------------------
// The float lane's mirror of the orders-axis block above, one kernel per shape and the same body:
// region A read as one fit per order rather than seeded at the top order and brought down a
// recurrence, and past kX0 the certified all-orders body of the shape the kernel names.
//
// The seed lane is the double lane's, the pairing every float lane kernel of this file already
// makes (Lane64Full beside Lane32Full, and Lane64EffBatch<kLane> beside Lane32EffBatch<kLane> for a
// rung): the region-A seed is computed in double and rounded once on entry to the recursion,
// because the downward recursion amplifies a float seed error past the float budget. So each
// kernel below hands DeviceOrdersF32 a lane object this file already runs and a piece table this
// lane already stores - the axis is a reading of the same stored fits and not a new fit.
//
// The shipped partition's rung is the float batch's own cut, so its row has the two forms a rung
// has, and the narrow partition's pair has them too, one per basis. The rational route's rows are
// served at the full-accuracy multiplier alone: this lane holds no rung cut of the float lane's
// rational pairs, and a rung kernel here would have to read one. A rung those rows do not serve is
// refused at the entry and is never answered from the stored table.
template <typename SeedLane, typename Lane>
__device__ __forceinline__ void DeviceOrdersBody32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, float* out, size_t count,
    size_t i) {
    detail::DeviceOrdersF32(seedLane, lane, order, xx,
                            [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32OrdersKernel(const int* n,
                                             const double* __restrict__ x,
                                             float* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64Full{}, Lane32Full{}, n[i], static_cast<float>(x[i]), out, count, i);
}

template <int kLane>
__global__ void BoysAllOrdersF32OrdersKernelEff(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64EffBatch<kLane>{}, Lane32EffBatch<kLane>{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

__global__ void BoysAllOrdersF32NarrowOrdersKernel(const int* n,
                                                   const double* __restrict__ x,
                                                   float* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64Narrow<false>{}, Lane32Narrow{}, n[i], static_cast<float>(x[i]), out,
                       count, i);
}

__global__ void BoysAllOrdersF32NarrowOrdersMonoKernel(const int* n,
                                                       const double* __restrict__ x,
                                                       float* __restrict__ out,
                                                       size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64NarrowMono<false>{}, Lane32NarrowMono{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

// The fit route's orders shapes. The pair is stored once and read by the two readings the double
// lane's route names: this one reads each order's own piece at A = 1 (RatCut::kPerOrder), which is
// what an order's own value is, where the per-argument shape seeds at its top order's piece and
// carries that piece's w(b) down the recursion.
__global__ void BoysAllOrdersF32OrdersRatKernel(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64RatFull{}, Lane32Rat{}, n[i], static_cast<float>(x[i]), out, count, i);
}

// The narrow partition at a rung, both axes: region A is the double lane's narrow cut through the
// per-order seed lane and region B this lane's own cut (Lane32NarrowRelaxed).
__global__ void BoysAllOrdersF32NarrowKernelEff(const int* n,
                                                const double* __restrict__ x,
                                                float* __restrict__ out,
                                                size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64Narrow<true>{},
                               Lane32NarrowRelaxed{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32NarrowOrdersKernelEff(const int* n,
                                                      const double* __restrict__ x,
                                                      float* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64Narrow<true>{}, Lane32NarrowRelaxed{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

// The same partition's rung in the monomial basis, on both axes: the other form of the seed lane
// and the other form's cut of region B. The two are one pair - the basis a lane sums in and the
// table its degree was cut from - and nothing here reads the Chebyshev pair.
__global__ void BoysAllOrdersF32NarrowMonoKernelEff(const int* n,
                                                    const double* __restrict__ x,
                                                    float* __restrict__ out,
                                                    size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64NarrowMono<true>{},
                               Lane32NarrowMonoRelaxed{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32NarrowOrdersMonoKernelEff(const int* n,
                                                          const double* __restrict__ x,
                                                          float* __restrict__ out,
                                                          size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64NarrowMono<true>{}, Lane32NarrowMonoRelaxed{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

// The fit route at a rung, on both of this lane's partitions and both of the route's readings: the
// seed lane is the double lane's pair at the rung's own cut of the reading this shape makes, and
// the float lane supplies the region-B seed at the same rung's cut of its own pair - the one seed
// both readings share.
__global__ void BoysAllOrdersF32RatKernelEff(const int* n,
                                             const double* __restrict__ x,
                                             float* __restrict__ out,
                                             size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64RatEff<RatCut::kBatchSeed>{},
                               Lane32RatRelaxed{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32OrdersRatKernelEff(const int* n,
                                                   const double* __restrict__ x,
                                                   float* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64RatEff<RatCut::kPerOrder>{}, Lane32RatRelaxed{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

__global__ void BoysAllOrdersF32NarrowRatKernelEff(const int* n,
                                                   const double* __restrict__ x,
                                                   float* __restrict__ out,
                                                   size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    detail::DeviceAllOrdersF32(Lane64NarrowRat<true, RatCut::kBatchSeed>{},
                               Lane32NarrowRatRelaxed{},
                               n[i],
                               static_cast<float>(x[i]),
                               [&](int l, float v) { out[l * count + i] = v; });
}

__global__ void BoysAllOrdersF32NarrowOrdersRatKernelEff(const int* n,
                                                         const double* __restrict__ x,
                                                         float* __restrict__ out,
                                                         size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64NarrowRat<true, RatCut::kPerOrder>{}, Lane32NarrowRatRelaxed{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
}

__global__ void BoysAllOrdersF32NarrowOrdersRatKernel(const int* n,
                                                      const double* __restrict__ x,
                                                      float* __restrict__ out,
                                                      size_t count) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    DeviceOrdersBody32(Lane64NarrowRat<false, RatCut::kPerOrder>{}, Lane32NarrowRat{}, n[i],
                       static_cast<float>(x[i]), out, count, i);
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
        // addressed separately because a rung's cut moves the numerator's end and leaves the
        // denominator's start where the stored numerator's degree put it.
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
        // it - a cut reads a prefix of the same run - and region B is the two arrays the header
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
        // the grid's two per-interval tables beside them. It is not a cut and no rung moves it: the
        // cells carry the degrees the grid's own cell law placed them at, and an entry of this
        // route is served at every rung by this pool.
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

    // The float lane's narrow partition, laid out as the double lane's is: one pool per form and
    // one row per piece, a row's flat index being its position in kNarrowAPiecesF32. The stored
    // degrees travel with the pieces because no rung table is carried beside them here (see the
    // symbol block).
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
    // for the reason the double lane's is: it is not a cut and no rung moves it.
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
    // in kNarrowAPieces, which is also the index its effective degree is derived at.
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
// the float lane's seven, then the relaxed image's three, then the uniform grid's four pools. The
// status layer cannot name a CUDA symbol, so the order is the seam between this file and
// boys_cuda.cpp and both sides state it.
// Uploads first, so a caller that skipped InitializeTables still gets a table rather than a null
// pointer.
namespace {

// The address order, one entry per symbol, in the slot order the handle's fields
// are declared and the status layer fills them.
const void* const kTableAddressSymbols[] = {&dPieceStart,      &dOffset,
                              &dA,               &dB,
                              &dDeg,             &dCoeffs,
                              &dBSeed,           &dPieceStart32,
                              &dOffset32,        &dA32,
                              &dB32,             &dDeg32,
                              &dCoeffs32,        &dBSeed32,
                              &dRelaxedM,        &dDegEff,
                              &dBDegEff,         &dFlatCoeffs,
                              &dFlatMonoCoeffs,  &dFlatCoeffsF32,
                              &dFlatMonoCoeffsF32,
                              &dNarrowAPieceStart, &dNarrowAOffset,
                              &dNarrowAA,        &dNarrowAB,
                              &dNarrowAStoredDeg, &dNarrowACoeffs,
                              &dNarrowAMonoCoeffs, &dNarrowBEdges,
                              &dNarrowBCoeffs,   &dNarrowBMonoCoeffs,
                              &dNarrowDegEff,    &dNarrowBDegEff,
                              &dNarrowMonoDegEff, &dNarrowMonoBDegEff,
                              &dNarrowAPieceStart32, &dNarrowAOffset32,
                              &dNarrowAA32,      &dNarrowAB32,
                              &dNarrowAStoredDeg32, &dNarrowACoeffs32,
                              &dNarrowAMonoCoeffs32, &dNarrowBEdges32,
                              &dNarrowBCoeffs32, &dNarrowBMonoCoeffs32,
                              &dRatACoeffs,      &dRatOffsetFlat,
                              &dRatDenOffsetFlat, &dRatNumDegFlat,
                              &dRatDenDegFlat,   &dRatBnum,
                              &dRatBden,         &dRatBDeg,
                              &dRatSeedDegFlat,  &dRatBnum32,
                              &dRatBden32,
                              &dNarrowRatACoeffs, &dNarrowRatAOffset,
                              &dNarrowRatADenOffset, &dNarrowRatANumDeg,
                              &dNarrowRatADenDeg, &dNarrowRatSeedDeg,
                              &dNarrowRatBCoeffs,
                              &dNarrowRatBOffset, &dNarrowRatBStoredNumDeg,
                              &dNarrowRatBDenDeg, &dNarrowRatBDeg,
                              &dNarrowRatBCoeffs32, &dNarrowRatBOffset32,
                              &dNarrowRatBStoredNumDeg32, &dNarrowRatBDenDeg32,
                              &dNarrowBDegEff32, &dNarrowMonoBDegEff32,
                              &dRatBDeg32,       &dNarrowRatBDeg32,
                              &dFlatDegs,        &dFlatOffsets,
                              &dFlatDegsF32,     &dFlatOffsetsF32,
                              &dFlatRatCoeffs,   &dFlatRatNumDeg,
                              &dFlatRatDenDeg,   &dFlatRatStored,
                              &dFlatRatOffsets,
                              &dFlatRatCoeffsF32, &dFlatRatNumDegF32,
                              &dFlatRatDenDegF32, &dFlatRatStoredF32,
                              &dFlatRatOffsetsF32};
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
    // 44 its float lane; 45 to 55 the fit route on the shipped partition, 56 to 66 on the narrow
    // one, 67 to 70 its float lane; 71 to 74 the uniform grid's two per-interval tables in each
    // lane. A new group goes after those and never inside them.
    //
    // The order is read by two calls: the first twenty-one slots are the ones a caller's build
    // already holds an array for, written by BoysCudaDeviceTableAddresses, and the rest by
    // BoysCudaDeviceTableAddressesTail. Splitting there keeps a caller not rebuilt for the wider
    // handle from being handed more addresses than its array holds.

    // The bound is the array's own length, so a pool added here cannot be
    // exported without one of the two loops below reaching it.
    for (int i = 0; i < kTableAddressFirstCount; ++i)
    {
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
    // kTableAddressFirstCount + i lands at its index i.
    for (int i = kTableAddressFirstCount; i < kTableAddressCount; ++i)
    {
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

// The float lane's partition and grid, one launcher per arithmetic: the route is an argument of the
// caller's and not of the launch, so the choice is the symbol it names. The uniform pair's rung
// axis is empty (the table is stored at one degree per order and interval); the narrow pair's is
// refused at the entry instead, for the reason DeviceEntryServedAtRung gives.
extern "C" int BoysCudaLaunchAllOrdersF32Uniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32FlatKernel<false>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32UniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32FlatKernel<true>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The float lane's grid on the same route, one launcher for the reason above.
extern "C" int BoysCudaLaunchAllOrdersF32UniformRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32FlatRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32Narrow(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowKernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowMonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The float lane's rational route, one launcher per partition: the two scheme names a caller may
// use differ in the report's row and not here, because the route's pair is stored in one form and
// the two names reach one kernel.
extern "C" int BoysCudaLaunchAllOrdersF32Rat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32RatKernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The float lane's orders axis: the same launches the rows above make, one per kernel of that axis.
// The shipped partition's row and the narrow partition's two have the two forms a rung has, so
// their pairs are here; the rational route's rows serve the full-accuracy multiplier alone and have
// the one launcher each. The uniform grid's two orders rows need no launcher of their own: its
// packing axis has one member and those rows launch BoysCudaLaunchAllOrdersF32Uniform and its
// Horner twin.
extern "C" int BoysCudaLaunchAllOrdersF32Orders(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32OrdersKernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32OrdersKernelEff<3>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The narrow partition's rung in the Chebyshev basis, on both axes. One launcher each and no lane
// argument: what selects the rung is which degree table the host made resident. The other basis has
// a pair of its own below, for the reason it has a table of its own.
extern "C" int BoysCudaLaunchAllOrdersF32NarrowEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The same partition's rung in the monomial basis, on both axes. The rung is selected by the
// monomial degree table the host made resident rather than by a lane argument here, and the two
// forms are two launchers because the kernel each runs reads a different pool's degrees.
extern "C" int BoysCudaLaunchAllOrdersF32NarrowMonoEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowMonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersMonoEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersMonoKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrders(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersMonoKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32OrdersRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The same four rows at a rung. One launcher each and no lane argument, as every relaxed launcher
// of this file: what selects the rung is which pair tables the host made resident.
extern "C" int BoysCudaLaunchAllOrdersF32RatEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32RatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowRatEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowRatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32OrdersRatEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32OrdersRatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF32NarrowOrdersRatEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    BoysAllOrdersF32NarrowOrdersRatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
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

extern "C" int BoysCudaLaunchAllOrdersF64Uniform(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64FlatKernel<false>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The same launch over the same table's other form. Two launchers and not one taking the form as an
// argument: the form is the arithmetic, and an argument would be a choice made at run time by the
// caller of a function whose whole contract is which numbers it delivers.
extern "C" int BoysCudaLaunchAllOrdersF64UniformHorner(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64FlatKernel<true>
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

// The grid's rational route, one launcher and not two: the pair is stored in one form, so both
// scheme names a caller may use reach it. The orders-axis rows of that route are this same launch,
// because the route's packing axis has no second member to run.
extern "C" int BoysCudaLaunchAllOrdersF64UniformRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64FlatRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllNF64(
    int nmax, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllNF64Kernel<<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        nmax, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

#if BoysFp16
// Async launch contract (uniform with the F32/F64 launchers): the kernel is queued on the caller's
// stream and the call returns once the launch is accepted - callers synchronize the stream before
// reading the outputs.
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
// both rows of the comparison are exercisable on one card.
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
// narrowMonoB: [piece][order], as the Chebyshev narrow region-B table is;
// narrowB32 / narrowMonoB32: the same layouts again, over the float lane's own
// pieces and one per basis, which is the split the symbols above state.
//
// The fit route's cuts travel the same way and for the same reason, two ints
// per cell — the numerator's cut degree then the denominator's. ratSeedA:
// [order][pieceInOrder], the cut the arguments shape reads; ratOrdA: the same
// layout, the cut the orders shape reads; ratB: one pair, the region B both
// readings share; narrowRatSeedA / narrowRatOrdA: flat over the narrow
// partition's rows; narrowRatB: one pair per narrow region-B piece.
extern "C" int BoysCudaUploadEffTables(double m,
                                       const int* degA,
                                       const int* degB,
                                       const int* narrowA,
                                       const int* narrowB,
                                       const int* narrowB32,
                                       const int* monoA,
                                       const int* monoB,
                                       const int* narrowMonoA,
                                       const int* narrowMonoB,
                                       const int* narrowMonoB32,
                                       const int* ratSeedA,
                                       const int* ratOrdA,
                                       const int* ratB,
                                       const int* ratB32,
                                       const int* narrowRatSeedA,
                                       const int* narrowRatOrdA,
                                       const int* narrowRatB,
                                       const int* narrowRatB32) {
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

    // The route's seed reading of the same rung, flattened to the device image's
    // indexing: the host hands every lane a row-major (order, pieceInOrder) cell
    // holding the pair a cut leaves, and a caller's entry reads a piece by its
    // flat index. This is the shipped partition's own piece-start table, the one
    // the route's rows are cut on.
    static int flatRatSeedDeg[2 * kPiecesTotal];

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        const std::size_t first = detail::kPieceStart[static_cast<std::size_t>(order)];
        const std::size_t last = detail::kPieceStart[static_cast<std::size_t>(order) + 1];

        for (std::size_t p = first; p < last; ++p)
        {
            const std::size_t cell =
                static_cast<std::size_t>(order) * kMaxPieces + (p - first);
            flatRatSeedDeg[2 * p] = ratSeedA[cell * 2];
            flatRatSeedDeg[2 * p + 1] = ratSeedA[cell * 2 + 1];
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

    if (cudaMemcpyToSymbol(dRatSeedDegFlat, flatRatSeedDeg, sizeof(flatRatSeedDeg)) != cudaSuccess)
    {
        return 2;
    }

    // The narrow partition's cut for the same rung: one table per lane, no lane
    // axis. The float lane's region B is the second of them, and the two are
    // different tables rather than one read twice: the lanes' pieces, their
    // degrees and their coefficients are each their own (see the symbols above).
    if (cudaMemcpyToSymbol(dNarrowDegEff, narrowA, kNarrowPiecesTotal * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowBDegEff,
                           narrowB,
                           detail::kNarrowBPieces * (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowBDegEff32,
                           narrowB32,
                           detail::f32::kNarrowBPiecesF32 * (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess)
    {
        return 2;
    }

    // The monomial scheme's cut of the same rung, in the same shapes, plus the
    // float lane's narrow region B — the one table of this family whose region-A
    // counterpart is absent for the reason the symbol block states: the float
    // lanes seed region A from the double lane's pieces, whichever basis they
    // sum in.
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
            cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowMonoBDegEff32,
                           narrowMonoB32,
                           detail::f32::kNarrowBPiecesF32 * (detail::kMaxOrder + 1) * sizeof(int)) !=
            cudaSuccess)
    {
        return 2;
    }

    // The fit route's cut of the same rung, in the same shapes as its stored
    // tables: region A per piece and per reading, region B one pair either way.
    if (cudaMemcpyToSymbol(dRatSeedDeg,
                           ratSeedA,
                           (detail::kMaxOrder + 1) * kMaxPieces * 2 * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpyToSymbol(dRatOrdDeg,
                           ratOrdA,
                           (detail::kMaxOrder + 1) * kMaxPieces * 2 * sizeof(int)) !=
            cudaSuccess ||
        cudaMemcpyToSymbol(dRatBDeg, ratB, 2 * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dRatBDeg32, ratB32, 2 * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowRatSeedDeg,
                           narrowRatSeedA,
                           kNarrowPiecesTotal * 2 * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowRatOrdDeg,
                           narrowRatOrdA,
                           kNarrowPiecesTotal * 2 * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowRatBDeg,
                           narrowRatB,
                           detail::kNarrowBPieces * 2 * sizeof(int)) != cudaSuccess ||
        cudaMemcpyToSymbol(dNarrowRatBDeg32,
                           narrowRatB32,
                           detail::f32::kNarrowRatBPiecesCountF32 * 2 * sizeof(int)) !=
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

// The fit route's four shapes, each in the two forms a rung has.
extern "C" int BoysCudaLaunchAllOrdersF64Rat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64RatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64RatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64RatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64OrdersRatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64OrdersRatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowRatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowRatKernelEff
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersRatKernel
        <<<LaunchBlocks(count), 256, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    BoysAllOrdersF64NarrowOrdersRatKernelEff
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
