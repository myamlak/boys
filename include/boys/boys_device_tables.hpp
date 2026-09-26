#pragma once

/// \file
/// What a caller hands the CUDA lane's entries: the handle the device-callable
/// Boys entries read (boys_cuda_device.hpp), the one arithmetic option the
/// single entries of that lane take, and the lanes the handle's relaxed degree
/// tables are cut into.
///
/// Kept free of CUDA runtime headers, so the C++ side of the CUDA lane
/// (boys_cuda.hpp) can name these types in a signature without dragging a CUDA
/// build requirement onto a plain translation unit, and kept free of this
/// library's implementation headers, so that a device translation unit that
/// includes boys_cuda_device.hpp compiles the coefficient tables it needs and
/// not the CPU lane's C++23 implementation.

#include "boys/accuracy.hpp"

namespace boys {

/// How the CUDA lane's f32 single entries evaluate the region-B exponential
/// e^{-x} that their upward recursion carries.
///
/// The two options are two arithmetics with two measured bounds, and neither is
/// a fallback for the other. What makes the choice worth stating is the
/// recurrence that consumes the value. Its condition number — the ratio of the
/// dominant solution of the homogeneous recurrence to the wanted one, which is
/// what Gautschi's treatment of three-term recurrences is about (see
/// CITATION.bib) — is 7.6e4 at the region-B boundary, n = 32, and falls as the
/// argument grows. A seed error whose *relative* size grows with the argument
/// therefore fails a bound over a band of region B at the highest order while
/// holding it everywhere else, and the instrument that predicts that is that
/// factor, not the ulp count at one argument.
///
///  - \c kAccurate is the library routine, and is the arithmetic the f32 batch
///    entries already run: at m = 1 a single and a batch evaluation of the same
///    (n, x) return the same bits outside region A. Its relative error is flat
///    at 2 ulp, so its bound is the lane's, m * 1.5e-7, in every region.
///  - \c kFast is the hardware approximation with its argument-scaling residual
///    removed. The approximation evaluates 2^fl(y log2 e), so the one rounding
///    of that product is what grows its error with |y|; the residual
///    fma(y, log2 e, -t) is exact and 2^(t + d) = 2^t 2^d approximates
///    2^t (1 + d log 2), so two fused steps take the error back to the
///    approximation's own few ulp, flat in the argument. Its bound is the
///    lane's plus its own seed's contribution, certified at 8e-8 — 0.41 of the
///    lane's budget — which the condition number derives and the device gate's
///    sweep confirms (5.0e-8 measured at m = 1).
///
/// **What the ulp count alone would have missed.** Without the correction the
/// approximation carries (2 + floor(1.16 |x|)) ulp: fifteen at the region-B
/// boundary, thirty-five at its far end. Multiplying by the amplification
/// factor gives 2.30e-7 at n = 32 at the boundary, 1.53 times the lane's whole
/// bound, and a failing band out to x = 12.07 at that order. Measured: 2.57e-7
/// at the same argument, and three cells over the bound at m = 1. A returned
/// value there also carries the opposite sign to the function — at
/// x = 11.899847984313965, n = 32, the approximation returns
/// -9.6178212061204249e-08 where F_32(x) = +1.6072968370095873e-07. That sign
/// consequence is an inference from the magnitude bound plus the eventual
/// domination of the unwanted solution, and it holds only where that solution
/// has the opposite sign to the wanted one; the corrected form returns the
/// value's sign, and the device gate's audit reports zero wrong-sign cells for
/// it at m = 1.
///
/// The bare approximation is not offered at any multiplier. Its failing band is
/// interior to region B — a caller cannot name a piece of a region — so there
/// is no certified sub-range to restrict it to, and a bound that a wrong sign
/// fits inside is not a bound for a consumer that reads the value rather than
/// its distance from F_n(x). That is the whole of the exclusion: with the
/// correction, both options hold their bounds in every region.
///
/// What separates the two is then the bound and not a counted cost: the
/// corrected form's kernel is the same size statically as the accurate one (944
/// instructions against 944 at m = 1, 976 against 976 on the relaxed lane),
/// where the bare approximation is eight fewer and certified nowhere, and the
/// exponential is evaluated once per element, outside the order loop. No speed
/// is claimed for either option, and the accurate one carries the tighter
/// bound of the two.
///
/// One name serves both lanes that take the option — the batch entry
/// BoysCuda::SingleF32, where it is a run-time argument of a multi-element
/// call, and the device entry BoysDeviceSingleF32, where it is a template
/// argument, because there the choice replaces an arithmetic inside the
/// caller's own kernel rather than branching within one.
///
/// \ingroup boys
enum class RegionBExp : int {
    /// expf: the library routine, the batch bodies' arithmetic, 2 ulp.
    kAccurate = 0,
    /// The hardware approximation with its argument-scaling residual removed.
    /// Pair it with the bound SingleF32 documents for it, and not with the
    /// lane's.
    kFast,
};

/// The region-B exponential the device lane's f32 entries evaluate when the
/// call site names none, which is the option whose bound is the lane's own:
/// \c RegionBExp::kAccurate, the library routine the f32 batch bodies have
/// always run.
///
/// It is named for the same reason the CPU surface's defaults are: so that the
/// selection in force is a thing a caller can point at, a document can cite and
/// a test can compare against, rather than the zero value of a template
/// parameter. A caller who names no exponential gets this name; naming it
/// explicitly is the same instantiation and no second arithmetic.
///
/// The other axis of the device lane's default is its accuracy multiplier,
/// which is \c kBoysFullAccuracyMultiplier, the same name the CPU entries
/// default to. The two are the whole of what "the device lane's default"
/// selects: the device surface takes no fit route, no scheme, no budget, no
/// packing axis and no partition, so a caller choosing a device precision and
/// naming neither of these is choosing everything the lane has to choose.
///
/// Neither default was chosen against a measurement — see the per-precision
/// defaults in docs/lane-contract.md for what that means and what changes when
/// one is set from a timing.
///
/// \ingroup boys
inline constexpr RegionBExp kDefaultRegionBExp = RegionBExp::kAccurate;

/// The six degree tables an entry can read.
///
/// Relaxation is a property of a lane and not of a precision: two lanes of one
/// precision carry different seed-error amplifications, so the same multiplier
/// buys them different effective degrees, and the region-A and region-B tables
/// of each are cut separately. This names the six, in the order the handle's
/// relaxed degree arrays hold them.
///
/// It is a naming of that layout and not a choice a caller makes. An entry
/// reads the table of the lane it serves, and the lane is a property of the
/// entry: no call takes one of these as an argument, and a caller reaches the
/// arithmetic of a different lane by calling a different entry.
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
    /// still the double piece table — the region-A seed is computed in double
    /// whatever precision the entry returns — and region B is the float piece
    /// table's order-0 entry.
    kF32Batch,
    /// The fp16 single entry: as kF32Single, under the fp16 lane's budget.
    kF16Single,
    /// The three fp16 family entries: as kF32Batch, under the fp16 lane's
    /// budget.
    kF16Batch,
};

/// The tables a device-callable entry reads (boys_cuda_device.hpp).
///
/// Plain data — device pointers and three degrees — and the same for every
/// precision: one handle serves the double, float and fp16 entries, and it is
/// the caller's to copy around, since it owns nothing. Fill it with
/// BoysCuda::DeviceTables, then pass it by value into a kernel and hand it to
/// an entry; a kernel parameter lives in the constant bank, so the handle
/// itself costs no global memory traffic. The tables it points at are the
/// library's: they live as long as the process does, and they are read-only.
///
/// A handle names the device that was current when it was filled, as every
/// other entry of the CUDA lane names the current device. A caller that runs on
/// more than one device fills one handle per device.
///
/// The fields are not a layout a caller writes to or reads from: the entries
/// read them, and their names are here so that a reader of a kernel signature
/// can see what the argument is. Only the library fills a handle.
///
/// **Two rungs, not one.** The full-accuracy degree tables are the handle's own
/// fields and are resident from the first upload. The relaxed degree tables are
/// a second set, resident for one multiplier at a time — the one the last
/// BoysCuda::DeviceTables call named, on the same per-(device, m) upload the
/// batch entries share — and the entries report a rung they have no tables for
/// rather than running the full-accuracy arithmetic under a relaxed name. What
/// the two arrays below hold is where those tables are; which rung they hold is
/// library state, and relaxedRung is the address of it, so an entry reads the
/// resident rung rather than a value copied into the handle when it was filled.
/// That is what makes a retired handle report itself: filling a handle for a
/// rung does not disturb any other handle, but it does retire the rung the
/// previous one named, and every entry asked for that rung then says so instead
/// of reading tables that have since been overwritten.
///
/// The full-accuracy tables are not in that image and are never retired: a call
/// that names m = 1 leaves whatever relaxed rung is resident in place, because
/// it has nothing to make resident.
///
/// The price of the second set is device memory and not constant-bank space:
/// the six region-A tables are 582 ints and the six region-B tables 198, so
/// 3120 bytes of the device's global memory stand behind the three fields
/// below, plus the eight bytes of the resident-rung scalar relaxedRung points
/// at, once for the whole process. The handle itself grows by thirteen
/// pointers — 104 bytes, from 128 to 232 — because the relaxed tables are
/// addressed once per call rather than held inline.
///
/// \ingroup boys
struct BoysDeviceTables {
    /// The double lane's piece index base per order: order \c n's pieces are
    /// the entries \c n and \c n+1 of this array, so order \c n has
    /// \c pieceStart[n+1] - \c pieceStart[n] pieces. kMaxOrder + 2 entries.
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
    /// [lane] The relaxed region-A effective degrees, in the indexing the table
    /// above uses: piece \c p of order \c n is the entry
    /// \c pieceStart[n] + \c p, and the lane's own piece-start table is the one
    /// its region-A seed reads its coefficients with. Indexed by
    /// BoysDeviceLane.
    const int* relaxedDegA[6] = {};
    /// [lane] The relaxed region-B effective degrees, kMaxBoysOrder + 1 per
    /// lane, indexed by BoysDeviceLane. The single lanes read the entry for the
    /// order the recursion has reached and the batch lanes the order-0 entry,
    /// for the reason their region-B bounds state.
    const int* relaxedDegB[6] = {};
};

} // namespace boys
