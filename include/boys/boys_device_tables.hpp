#pragma once

/// \file
/// What a caller hands the CUDA lane's entries: the handle the device-callable
/// entries read, the one arithmetic option the single entries of that lane take,
/// and the lanes the handle's relaxed degree tables are cut into.
///
/// Kept free of CUDA runtime headers, so the C++ side of the CUDA lane can name
/// these types without a CUDA build requirement, and free of this library's
/// implementation headers, so a device translation unit does not compile the CPU
/// lane's C++23 implementation.

#include "boys/accuracy.hpp"

namespace boys {

/// How the CUDA lane's f32 single entries evaluate the region-B exponential
/// e^{-x} that their upward recursion carries.
///
/// Two arithmetics with two measured bounds, neither a fallback for the other.
/// The recurrence that consumes the value is why the choice matters: its condition
/// number — the ratio of the dominant solution of the homogeneous recurrence to the
/// wanted one ([Gautschi1967]) — is 7.6e4 at the region-B boundary, n = 32, and falls
/// as the argument grows, so a seed error whose *relative* size grows with the
/// argument fails a bound over a band of region B at the highest order while holding
/// it everywhere else.
///
/// Both hold their bounds in every region; what separates them is the bound, not a
/// counted cost — the exponential is evaluated once per element, outside the order
/// loop.
///
///  - \c kAccurate is the library routine and the arithmetic the f32 batch entries
///    run: at m = 1 a single and a batch evaluation of the same (n, x) return the
///    same bits outside region A. Its relative error is flat at 2 ulp, so its bound
///    is the lane's, m * 1.5e-7, in every region.
///  - \c kFast is the hardware approximation with its argument-scaling residual
///    removed: 2^fl(y log2 e) is rounded once in that product, which is what grows
///    its error with |y|. The residual fma(y, log2 e, -t) is exact and
///    2^(t + d) = 2^t 2^d approximates 2^t (1 + d log 2), so two fused steps take
///    the error back to the approximation's own few ulp, flat in the argument. Its
///    bound is the lane's plus its own seed's contribution, certified at 8e-8 —
///    0.41 of the lane's budget — which the condition number derives and the device
///    gate's sweep confirms (5.0e-8 measured at m = 1).
///
/// The bare approximation is not offered at any multiplier: its failing band is
/// interior to region B, which a caller cannot name a sub-range of.
///
/// One name serves both lanes that take the option: the batch entry
/// BoysCuda::SingleF32 (a run-time argument) and the device entry
/// BoysDeviceSingleF32 (a template argument).
///
/// \ingroup boys
enum class RegionBExp : int {
    /// expf: the library routine, the batch bodies' arithmetic, 2 ulp.
    kAccurate = 0,
    /// The hardware approximation with its argument-scaling residual removed.
    /// Read it with the bound SingleF32 documents for it, not with the lane's.
    kFast,
};

/// The region-B exponential the device lane's f32 entries evaluate when the call
/// site names none: \c RegionBExp::kAccurate, the library routine the f32 batch
/// bodies run, whose bound is the lane's own.
///
/// The device lane's other default is the accuracy multiplier
/// \c kBoysFullAccuracyMultiplier, the same name the CPU entries default to.
///
/// \ingroup boys
inline constexpr RegionBExp kDefaultRegionBExp = RegionBExp::kAccurate;

/// The six degree tables an entry can read, in the order the handle's relaxed
/// degree arrays hold them.
///
/// Relaxation is a property of a lane and not of a precision: two lanes of one
/// precision carry different seed-error amplifications, so the same multiplier
/// buys them different effective degrees and each lane's region-A and region-B
/// tables are cut separately.
///
/// It names that layout; it is not a choice a caller makes. An entry reads the
/// table of the lane it serves, and no lane is a parameter of any call.
///
/// \ingroup boys
enum class BoysDeviceLane : int {
    /// The double single entry. Region A and region B are both the double piece
    /// table, and neither path amplifies the seed.
    kF64Single = 0,
    /// The double family entries: the runtime-top-order ladder, the fixed-top-
    /// order ladder and the sink. Region A is the double piece table under the
    /// downward recursion's amplification, region B the order-0 entry.
    kF64Batch,
    /// The float single entry. Region A and region B are both the float piece
    /// table, and neither path amplifies the seed.
    kF32Single,
    /// The three float family entries: as kF64Batch, except that region A is
    /// still the double piece table, and region B the float piece table's
    /// order-0 entry.
    kF32Batch,
    /// The fp16 single entry: as kF32Single, under the fp16 lane's budget.
    kF16Single,
    /// The three fp16 family entries: as kF32Batch, under the fp16 lane's
    /// budget.
    kF16Batch,
};

/// The tables a device-callable entry reads.
///
/// Plain data — device pointers and degrees — one handle for every precision. Fill
/// it with BoysCuda::DeviceTables, then pass it by value into a kernel and hand it
/// to an entry; a kernel parameter lives in the constant bank, so the handle itself
/// costs no global memory traffic. The tables it points at are the library's: they
/// live as long as the process does and are read-only. The handle owns nothing and
/// the caller copies it; its fields are the entries' to read, and their names are
/// here so that a reader of a kernel signature can see what the argument is.
///
/// A handle names the device that was current when it was filled, as every other
/// entry of the CUDA lane names the current device: a caller that runs on more than
/// one device fills one handle per device.
///
/// The full-accuracy degree tables are the handle's own fields, resident from the
/// first upload. The relaxed ones are a second set, resident for one multiplier at a
/// time — the one the last BoysCuda::DeviceTables call named. An entry reads that
/// rung through \c relaxedRung rather than a value copied into the handle, so filling
/// a handle for a rung retires the rung the previous one named, and an entry asked
/// for a retired rung says so instead of reading tables that have since been
/// overwritten. A call that names m = 1 retires nothing.
///
/// \ingroup boys
struct BoysDeviceTables {
    /// The double lane's piece index base per order: order \c n's pieces are
    /// entries \c n and \c n+1 of this array. kMaxOrder + 2 entries.
    const int* pieceStart = nullptr;
    const double* pieceA = nullptr; ///< [piece] the piece's lower edge
    const double* pieceB = nullptr; ///< [piece] the piece's upper edge
    const int* pieceDeg = nullptr;  ///< [piece] the piece's stored degree
    /// [piece] the index of the piece's first coefficient in \c coeffs
    const int* pieceOffset = nullptr;
    const double* coeffs = nullptr;      ///< the double lane's coefficient pool
    const double* bSeedCoeffs = nullptr; ///< the region-B seed's coefficients
    int bSeedDeg = 0;                    ///< the region-B seed's degree

    /// The float lane's tables, read by the f32 and fp16 entries; the same
    /// fields as above, cut into the float lane's own pieces.
    const int* pieceStart32 = nullptr;
    const float* pieceA32 = nullptr; ///< [piece] the piece's lower edge
    const float* pieceB32 = nullptr; ///< [piece] the piece's upper edge
    const int* pieceDeg32 = nullptr; ///< [piece] the piece's stored degree
    /// [piece] the index of the piece's first coefficient in \c coeffs32
    const int* pieceOffset32 = nullptr;
    const float* coeffs32 = nullptr;      ///< the float lane's coefficient pool
    const float* bSeedCoeffs32 = nullptr; ///< the region-B seed's coefficients
    int bSeedDeg32 = 0;                   ///< the region-B seed's degree

    /// The accuracy multiplier the relaxed degree tables are currently cut for,
    /// read per call. Zero when no relaxed rung has been made resident; the
    /// library owns the value and no caller writes it.
    const double* relaxedRung = nullptr;
    /// [lane] The relaxed region-A effective degrees, in the indexing the piece
    /// tables above use: piece \c p of order \c n is the entry
    /// \c pieceStart[n] + \c p, and the lane's own piece-start table is the one
    /// its region-A seed reads its coefficients with. Indexed by BoysDeviceLane.
    const int* relaxedDegA[6] = {};
    /// [lane] The relaxed region-B effective degrees, kMaxBoysOrder + 1 per lane,
    /// indexed by BoysDeviceLane. The single lanes read the entry for the order the
    /// recursion has reached and the batch lanes the order-0 entry.
    const int* relaxedDegB[6] = {};

    /// The uniform route's table: one grid of equal intervals over [0, kFlatHi),
    /// every order fitted on its own inside its interval's block. It is the route
    /// the CPU lane names \c FitGranularity::kUniform, and the whole of what an
    /// entry taking it reads: the interval an argument falls in and the join above
    /// which the table stops are compile-time facts of the grid.
    ///
    /// The two pointers of a lane are the two stored forms of that one fit: the
    /// Chebyshev blocks, which the split Clenshaw sums, and the monomial ones,
    /// which Horner sums. They are two rows of one table and not two arithmetics,
    /// so an entry that named the wrong one would sum the other half of the table.
    /// Each lane's rows are indexed by that lane's own degrees and offsets, which
    /// is why those tables are carried per lane.
    const double* flatCoeffs = nullptr;     ///< the double lane's Chebyshev blocks
    const double* flatMonoCoeffs = nullptr; ///< the double lane's monomial blocks
    const float* flatCoeffs32 = nullptr;    ///< the float lane's Chebyshev blocks
    const float* flatMonoCoeffs32 = nullptr; ///< the float lane's monomial blocks

    /// The narrow partition's tables, in the double lane: a second cut of the
    /// same domain, whose pieces are cut per order. Order \c n's pieces are
    /// entries \c n and \c n+1 of \c narrowPieceStart, so a piece index is the
    /// partition's own and the argument alone does not name one. Region A's
    /// pieces carry the fit of their order's F_n directly, as the shipped
    /// partition's do; region B's seed is piecewise, which is why it carries its
    /// own edge table rather than the two edges of one fit.
    ///
    /// The degree fields hold the partition's own cut of a rung: a stored degree
    /// per piece for region A, and one per piece and per order for region B,
    /// whose seed's degree is read at the order the calling lane has reached.
    const int* narrowPieceStart = nullptr;
    /// [piece] the index of the piece's first coefficient in \c narrowCoeffs
    const int* narrowPieceOffset = nullptr;
    const double* narrowPieceA = nullptr; ///< [piece] the piece's lower edge
    const double* narrowPieceB = nullptr; ///< [piece] the piece's upper edge
    const int* narrowStoredDeg = nullptr; ///< [piece] the piece's stored degree
    const double* narrowCoeffs = nullptr;     ///< region A, Chebyshev form
    const double* narrowMonoCoeffs = nullptr; ///< region A, monomial form
    /// [piece] region B's piece edges, one more than there are pieces
    const double* narrowBEdges = nullptr;
    const double* narrowBCoeffs = nullptr;     ///< region B's seed, Chebyshev form
    const double* narrowBMonoCoeffs = nullptr; ///< region B's seed, monomial form
    /// [piece] region A's relaxed degree for the resident rung
    const int* narrowRelaxedDegA = nullptr;
    /// [piece * (kMaxBoysOrder + 1) + order] region B's relaxed degrees, the
    /// degree of the piece's seed fit at that order
    const int* narrowRelaxedDegB = nullptr;
    /// [piece] the monomial form of region A's relaxed degree
    const int* narrowMonoRelaxedDegA = nullptr;
    /// [piece * (kMaxBoysOrder + 1) + order], the monomial form of region B's
    const int* narrowMonoRelaxedDegB = nullptr;

    /// The same partition one lane down. Its region B is the float lane's own
    /// piecewise seed; its region A is the double lane's narrow pieces above —
    /// the one seed lane the float entries seed from. So a rung of this partition
    /// reads the double lane's region-A cut above and this lane's own region-B cut
    /// below, one table per basis.
    const int* narrowPieceStart32 = nullptr;
    /// [piece] the index of the piece's first coefficient in \c narrowCoeffs32
    const int* narrowPieceOffset32 = nullptr;
    const float* narrowPieceA32 = nullptr; ///< [piece] the piece's lower edge
    const float* narrowPieceB32 = nullptr; ///< [piece] the piece's upper edge
    const int* narrowStoredDeg32 = nullptr; ///< [piece] the piece's stored degree
    const float* narrowCoeffs32 = nullptr;     ///< region A, Chebyshev form
    const float* narrowMonoCoeffs32 = nullptr; ///< region A, monomial form
    /// [piece] region B's piece edges, one more than there are pieces
    const float* narrowBEdges32 = nullptr;
    const float* narrowBCoeffs32 = nullptr;     ///< region B's seed, Chebyshev
    const float* narrowBMonoCoeffs32 = nullptr; ///< region B's seed, monomial

    /// The rational route's tables on the shipped partition, double lane: the
    /// same pieces as the piece tables above, with each piece stored as a
    /// numerator and a denominator summed apart and divided once.
    ///
    /// The route's degree tables come in pairs — the numerator's degree then the
    /// denominator's — and at two resolutions: \c ratNumDeg and \c ratDenDeg hold
    /// the degrees the table was stored at, and \c ratSeedDeg the resident rung's
    /// cut of the reading the consuming entry makes, which is the reading of the
    /// ladder shape every device-callable entry of this lane carries. Every one of
    /// them is read at the flat piece index \c pieceStart[order] + \c piece.
    ///
    /// \c ratBNum and \c ratBDen are region B's single seed pair, read whole at
    /// every order, and \c ratRelaxedDegB its degrees at the resident rung.
    const double* ratCoeffs = nullptr; ///< the numerator and denominator pool
    /// [piece] the numerator's first coefficient in \c ratCoeffs
    const int* ratOffset = nullptr;
    /// [piece] the denominator's first coefficient in \c ratCoeffs
    const int* ratDenOffset = nullptr;
    const int* ratNumDeg = nullptr; ///< [piece] the stored numerator degree
    const int* ratDenDeg = nullptr; ///< [piece] the stored denominator degree
    /// [2 * piece] the resident rung's cut of the seed's own reading
    const int* ratSeedDeg = nullptr;
    const double* ratBNum = nullptr; ///< region B's seed numerator
    const double* ratBDen = nullptr; ///< region B's seed denominator
    /// [2] region B's seed degrees at the resident rung: numerator, denominator
    const int* ratRelaxedDegB = nullptr;
    const float* ratBNum32 = nullptr; ///< region B's seed numerator, float lane
    const float* ratBDen32 = nullptr; ///< region B's seed denominator, float lane

    /// The rational route on the narrow partition, double lane: the narrow pieces
    /// above, each stored as a pair. Region A's metadata is one table per
    /// attribute here, as the partition's other tables are, and region B's seed is
    /// a piecewise pair whose denominator sits above its numerator in
    /// \c narrowRatBCoeffs.
    const double* narrowRatCoeffs = nullptr;
    const int* narrowRatOffset = nullptr;   ///< [piece] numerator's first index
    const int* narrowRatDenOffset = nullptr; ///< [piece] denominator's first index
    const int* narrowRatNumDeg = nullptr;   ///< [piece] the stored numerator degree
    const int* narrowRatDenDeg = nullptr;   ///< [piece] the stored denominator degree
    /// [2 * piece] the resident rung's cut of the seed's own reading
    const int* narrowRatSeedDeg = nullptr;
    const double* narrowRatBCoeffs = nullptr; ///< region B's numerator and denominator
    /// [piece] the piece's first coefficient in \c narrowRatBCoeffs
    const int* narrowRatBOffset = nullptr;
    const int* narrowRatBStoredNumDeg = nullptr; ///< [piece] the stored numerator degree
    const int* narrowRatBDenDeg = nullptr;       ///< [piece] the stored denominator degree
    /// [2 * piece] region B's degrees at the resident rung
    const int* narrowRatRelaxedDegB = nullptr;

    /// The rational route on the narrow partition one lane down: region B's
    /// piecewise pair in the float lane's own pieces, over the double lane's
    /// narrow pair above. A rung cuts it in \c narrowRatRelaxedDegB32 below.
    const float* narrowRatBCoeffs32 = nullptr;
    /// [piece] the piece's first coefficient in \c narrowRatBCoeffs32
    const int* narrowRatBOffset32 = nullptr;
    const int* narrowRatBStoredNumDeg32 = nullptr; ///< [piece] numerator degree
    const int* narrowRatBDenDeg32 = nullptr;       ///< [piece] denominator degree

    /// [piece * (kMaxBoysOrder + 1) + order] the float narrow partition's own cut
    /// of region B's seed at the resident rung, in the Chebyshev form of that
    /// seed: the degree the piece's fit is read to, the same cut the launched rows
    /// of that partition take. Region A carries no table beside it — the seed lane
    /// is the double lane's narrow pieces, whose cut is \c narrowRelaxedDegA above.
    const int* narrowRelaxedDegB32 = nullptr;
    /// The same table for the monomial form of the same seed, which is what a
    /// Horner call at that rung reads.
    const int* narrowMonoRelaxedDegB32 = nullptr;
    /// [2] the float lane's fit route on the shipped partition: its region-B
    /// pair's degrees at the resident rung, the numerator's first. The route's
    /// float pair is that lane's own fit, so the double lane's pair above
    /// (ratRelaxedDegB) is a cut of other coefficients and is not read here.
    const int* ratRelaxedDegB32 = nullptr;
    /// [2 * piece] the same pair's degrees on the narrow partition.
    const int* narrowRatRelaxedDegB32 = nullptr;

    /// The uniform grid's cells, one entry per interval, read by the route's
    /// own entries (BoysDeviceAllOrdersF64Uniform and its float counterpart).
    ///
    /// The grid's intervals do not all carry the same degree — each was given the
    /// smallest its own truncation bound holds it to, so a cell near the join needs
    /// less than one near the origin — and the table is stored interval-major, each
    /// interval's block holding (its degree + 1) coefficients per order. So the
    /// coefficient of order \c l and term \c k in interval \c iv is
    ///
    ///   flatCoeffs[flatOffsets[iv] + l * (flatDegs[iv] + 1) + k]
    ///
    /// and a reader needs both tables to address one cell: neither the degree nor
    /// the block's start is a constant of the grid, and a reader that held either
    /// fixed would sum a neighbouring cell's polynomial.
    const int* flatDegs = nullptr; ///< [interval] that interval's own degree
    /// [interval + 1] the interval's first coefficient; the last entry is the
    /// pool's stored count, so a block never runs past the end of the table.
    const int* flatOffsets = nullptr;
    /// The float lane's own grid, which is derived from that lane's bound,
    /// format and read cap: its degrees and its offsets are the double lane's
    /// only by coincidence and are never read in its place.
    const int* flatDegs32 = nullptr;
    /// [interval + 1] the float grid's first coefficient per interval, the same
    /// layout and the same closing count as \c flatOffsets above.
    const int* flatOffsets32 = nullptr;

    /// The same two grids on their RATIONAL route: one numerator/denominator
    /// pair per interval, read by the route's own entries
    /// (BoysDeviceAllOrdersF64UniformRat and its float counterpart).
    ///
    /// The stored form is the lane's own rational storage, the one the shipped and
    /// narrow rational routes already read (DeviceRatSum): the numerator's
    /// coefficients ascending, then the denominator's q_1..q_k with q_0 held at 1.
    /// The interval's block is interval-major at the interval's own stored count,
    /// which is its pair plus the held constant term, so the numerator of order
    /// \c l in interval \c iv begins at
    ///
    ///   flatRatCoeffs[flatRatOffsets[iv] + l * flatRatStored[iv]]
    ///
    /// and the denominator begins \c flatRatNumDeg[iv] + 1 coefficients later. Four
    /// per-interval columns address one row — the two degrees, the stored count and
    /// the block's start — and a reader that held any of them fixed would read a
    /// neighbouring interval's pair, which no check of the coefficients alone would
    /// report. This table has no single stride, which is why the stored count is one
    /// of the columns and not a constant of the grid.
    const double* flatRatCoeffs = nullptr; ///< the double lane's numerator/denominator blocks
    /// [interval] the two degrees of that interval's pair and the count one row
    /// stores, which is numDeg + 1 + denDeg.
    const int* flatRatNumDeg = nullptr;
    const int* flatRatDenDeg = nullptr; ///< [interval] the denominator's degree
    const int* flatRatStored = nullptr; ///< [interval] the count one row stores
    /// [interval + 1] the interval's first coefficient; the last entry is the
    /// pool's stored count, so a block never runs past the end of the table.
    const int* flatRatOffsets = nullptr;

    /// The float lane's own grid on the same route, which is a fit of that
    /// lane's arithmetic over that lane's intervals and is never read in the
    /// double lane's place: its pairs, its degrees and its stored counts are the
    /// double lane's only by coincidence.
    const float* flatRatCoeffs32 = nullptr;
    /// The four per-interval columns of the float grid, the same layout and the
    /// same meaning as the double grid's above.
    const int* flatRatNumDeg32 = nullptr;
    const int* flatRatDenDeg32 = nullptr; ///< [interval] the denominator's degree, as above
    const int* flatRatStored32 = nullptr; ///< [interval] the count one row stores, as above
    const int* flatRatOffsets32 = nullptr; ///< [interval + 1] the first coefficient, as above
};

/// The number of addresses the handle's tail export writes: one for every symbol
/// the order carries after the first twenty-one, in that order. It counts symbols
/// and not fields — one of these symbols fills an array of degree pointers. Read by
/// the status layer, which sizes its array with it, and asserted in the device image
/// against the order's own length, so a table added to either end without the other
/// is a compile error rather than an address written past an array.
inline constexpr int kBoysDeviceTablesTailCount = 68;

/// The tail export's four slots for the uniform grid's per-interval tables on its
/// Chebyshev route, in the order (double degrees, double offsets, float degrees,
/// float offsets). The rational route's two groups were appended after them, so
/// these are named from the end of those groups: a group appended without re-cutting
/// them reads the wrong slots, and the device image's own assert on the tail's length
/// makes that a compile error.
inline constexpr int kBoysDeviceTablesTailFlatGrid = kBoysDeviceTablesTailCount - 14;

/// The tail export's five slots for the double lane's uniform grid on its
/// RATIONAL route, in the order (the pool, numDeg, denDeg, stored, offsets).
/// They are appended after the Chebyshev grid's group, as the append-only rule
/// requires, so a slot a caller's build already reads keeps its index.
inline constexpr int kBoysDeviceTablesTailRatGrid = kBoysDeviceTablesTailCount - 10;

/// The tail export's five slots for the float lane's grid on the same route,
/// the same five in the same order, appended after the double lane's.
inline constexpr int kBoysDeviceTablesTailRatGrid32 = kBoysDeviceTablesTailCount - 5;

} // namespace boys
