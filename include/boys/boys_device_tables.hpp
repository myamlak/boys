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

/// The region-B exponential the device lane's f32 entries evaluate when the call
/// site names none: \c RegionBExp::kAccurate, the library routine the f32 batch
/// bodies run, whose bound is the lane's own.
///
/// **The axis is one shared type.** \c RegionBExp is defined in
/// `boys/accuracy.hpp`, which states the member set both targets name and each
/// target's own arithmetic for each member: the CPU lanes read it as a field of
/// their evaluation policy and the device entries take it here, so a member one
/// side grows is a member of the axis the other side already names. There is one
/// definition and this header adds none - the enumerators below are the shared
/// ones, and the f32 single entry BoysDeviceSingleF32 (a template argument) and
/// the batch entry BoysCuda::SingleF32 (a run-time argument) both name them.
///
/// This constant is the device lane's own, and it is the lane's own **name in
/// the build's seam** - \c BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP, in
/// `boys/boys_build_defaults.hpp` - because it is not the host's value: it is
/// \c kAccurate rather than the host's \c kFast because this lane's published
/// figures were measured at the library routine - which is also the arithmetic
/// the f32 batch bodies run, the same bits for the same (n, x) outside region A
/// - so naming nothing keeps the values the documents state. One name for both
/// targets could only be wrong on one of them, so the build that measured its
/// own card names this lane's member here, in the seam file, and not through the
/// host's. The seam name itself is read once, in `boys/accuracy.hpp`,
/// as \c boys::kDefaultDeviceRegionBExp: this name is the same constant, spelled
/// here because the tables it is read with are here, and the policy table's device
/// rows name the other spelling where this header is not carried.
///
/// The device lane's other defaults are the accuracy multiplier
/// \c kBoysFullAccuracyMultiplier, the same name the CPU entries default to, and
/// the division form \c boys::kDefaultDeviceDivisionForm, which is declared
/// beside the host's own in `boys/accuracy.hpp` because the policy table's
/// device rows name it in `boys/boys.hpp`.
///
/// \ingroup boys
inline constexpr RegionBExp kDefaultRegionBExp = kDefaultDeviceRegionBExp;

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

    /// Lanes this build defines; one past the last. Named by no arm of a switch
    /// over \c BoysDeviceLane, and owed no lane's name, so a report of the
    /// lanes has nothing to say about it.
    kCount,
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
/// The degree tables are the handle's own fields, resident from the first upload.
/// The accuracy axis has one member — m = 1 is the whole of it — so there is no
/// second, relaxed set beside them and no call that makes one resident.
///
/// The fields named for a relaxed reading — \c relaxedRung, \c relaxedDegA,
/// \c relaxedDegB, the partitions' \c *Relaxed* tables and the two seed cuts
/// (\c ratSeedDeg, \c narrowRatSeedDeg) — are placeholders: null in every handle,
/// and read by no entry of this revision. They are placeholders and not deletions
/// because each holds one slot of the address order the device image exports, and
/// that order is positional — the reserved slots sit between live ones, so a slot
/// dropped renumbers every slot after it and a caller's build reads another
/// table's addresses. Each one's comment below says which table vacated it.
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

    /// Reserved: the removed accuracy rung's residency scalar. Always null — the
    /// device image exports null into this slot and this handle leaves the field at
    /// the default its type gives it — and read by no entry of this revision.
    const double* relaxedRung = nullptr;
    /// [lane] Reserved: the removed rung's region-A effective degrees, in the
    /// indexing the piece tables above use — piece \c p of order \c n is the entry
    /// \c pieceStart[n] + \c p. Always null, read by no entry, indexed by
    /// BoysDeviceLane as every per-lane table here is.
    const int* relaxedDegA[6] = {};
    /// [lane] Reserved: the removed rung's region-B effective degrees, of the shape
    /// a per-lane table takes here — kMaxBoysOrder + 1 entries per lane, indexed by
    /// BoysDeviceLane. Always null and read by no entry of this revision.
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
    /// pieces carry the fit of their order's F_n directly, as the coarsest
    /// partition's do; region B's seed is piecewise, which is why it carries its
    /// own edge table rather than the two edges of one fit.
    ///
    /// \c narrowStoredDeg holds the fit's degree per piece, which is the degree
    /// region A's pieces are read at. The rung's own cuts that stood beside it are
    /// the reserved fields below: null in every handle, read by no entry.
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
    /// [piece] Reserved: region A's degree at the removed rung. Always null, read
    /// by no entry.
    const int* narrowRelaxedDegA = nullptr;
    /// [piece * (kMaxBoysOrder + 1) + order] Reserved: region B's degrees at the
    /// removed rung. Always null, read by no entry.
    const int* narrowRelaxedDegB = nullptr;
    /// [piece] The monomial form of the same reserved field. Always null, read by
    /// no entry.
    const int* narrowMonoRelaxedDegA = nullptr;
    /// [piece * (kMaxBoysOrder + 1) + order] The monomial form of the same reserved
    /// field. Always null, read by no entry.
    const int* narrowMonoRelaxedDegB = nullptr;

    /// The same partition one lane down. Its region B is the float lane's own
    /// piecewise seed; its region A is the double lane's narrow pieces above —
    /// the one seed lane the float entries seed from. So an entry of this partition
    /// reads the double lane's region-A pieces above and this lane's own region-B
    /// pieces below, one table per basis.
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

    /// The rational route's tables on the coarsest partition, double lane: the
    /// same pieces as the piece tables above, with each piece stored as a
    /// numerator and a denominator summed apart and divided once.
    ///
    /// The route's degree tables come in pairs — the numerator's degree then the
    /// denominator's — and \c ratNumDeg and \c ratDenDeg hold the degrees the table
    /// was stored at, which is the reading the consuming entry makes. Both are read
    /// at the flat piece index \c pieceStart[order] + \c piece. The rung's cuts that
    /// stood beside them (\c ratSeedDeg, \c ratRelaxedDegB and their siblings below)
    /// are reserved fields: null in every handle, read by no entry.
    ///
    /// \c ratBNum and \c ratBDen are region B's single seed pair, read whole at
    /// every order.
    const double* ratCoeffs = nullptr; ///< the numerator and denominator pool
    /// [piece] the numerator's first coefficient in \c ratCoeffs
    const int* ratOffset = nullptr;
    /// [piece] the denominator's first coefficient in \c ratCoeffs
    const int* ratDenOffset = nullptr;
    const int* ratNumDeg = nullptr; ///< [piece] the stored numerator degree
    const int* ratDenDeg = nullptr; ///< [piece] the stored denominator degree
    /// [2 * piece] Reserved: the seed's own reading at the removed rung. Always
    /// null, read by no entry.
    const int* ratSeedDeg = nullptr;
    const double* ratBNum = nullptr; ///< region B's seed numerator
    const double* ratBDen = nullptr; ///< region B's seed denominator
    /// [2] Reserved: region B's seed degrees at the removed rung, numerator then
    /// denominator. Always null, read by no entry.
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
    /// [2 * piece] Reserved: the same seed cut on the narrow partition. Always
    /// null, read by no entry.
    const int* narrowRatSeedDeg = nullptr;
    const double* narrowRatBCoeffs = nullptr; ///< region B's numerator and denominator
    /// [piece] the piece's first coefficient in \c narrowRatBCoeffs
    const int* narrowRatBOffset = nullptr;
    const int* narrowRatBStoredNumDeg = nullptr; ///< [piece] the stored numerator degree
    const int* narrowRatBDenDeg = nullptr;       ///< [piece] the stored denominator degree
    /// [2 * piece] Reserved: region B's degrees at the removed rung. Always null,
    /// read by no entry.
    const int* narrowRatRelaxedDegB = nullptr;

    /// The rational route on the narrow partition one lane down: region B's
    /// piecewise pair in the float lane's own pieces, over the double lane's
    /// narrow pair above.
    const float* narrowRatBCoeffs32 = nullptr;
    /// [piece] the piece's first coefficient in \c narrowRatBCoeffs32
    const int* narrowRatBOffset32 = nullptr;
    const int* narrowRatBStoredNumDeg32 = nullptr; ///< [piece] numerator degree
    const int* narrowRatBDenDeg32 = nullptr;       ///< [piece] denominator degree

    /// [piece * (kMaxBoysOrder + 1) + order] Reserved: the float narrow
    /// partition's cut of region B's seed at the removed rung, in the Chebyshev
    /// form of that seed. Always null, read by no entry. Region A carries no table
    /// beside it — the seed lane is the double lane's narrow pieces, whose reserved
    /// cut is \c narrowRelaxedDegA above.
    const int* narrowRelaxedDegB32 = nullptr;
    /// The same table for the monomial form of the same seed. Always null, read by
    /// no entry.
    const int* narrowMonoRelaxedDegB32 = nullptr;
    /// [2] Reserved: the float lane's fit route on the coarsest partition had its
    /// own region-B cut here, the numerator's degree first. Always null, read by no
    /// entry. The route's float pair is that lane's own fit, so the double lane's
    /// pair above is a cut of other coefficients and is not read here.
    const int* ratRelaxedDegB32 = nullptr;
    /// [2 * piece] Reserved: the same pair's degrees on the narrow partition.
    /// Always null, read by no entry.
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
    /// The stored form is the lane's own rational storage, the one the coarsest and
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
