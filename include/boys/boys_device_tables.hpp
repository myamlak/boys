#pragma once

/// \file
/// What a caller hands the CUDA lane's entries: the handle the device-callable
/// Boys entries read (boys_cuda_device.hpp), and the one arithmetic option the
/// single entries of that lane take.
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

/// The tables a device-callable entry reads (boys_cuda_device.hpp).
///
/// Plain data — device pointers and two degrees — and the same for every
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
};

} // namespace boys
