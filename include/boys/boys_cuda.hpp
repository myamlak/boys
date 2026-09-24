#pragma once

#include "boys/boys.hpp"

#include <cstddef>

#if BoysFp16
#include "boys/f16.hpp"
#endif

/// \file
/// CUDA lane of the Boys kernel.
///
/// Kept free of CUDA runtime headers so plain C++ translation units can
/// include it; the implementation lives in boys_cuda.cu. Streams are
/// opaque `void*` here and `cudaStream_t` inside the .cu file. All entry
/// points take raw device pointers — the caller owns the device memory
/// (cudaMalloc/cudaMemcpy/cudaFree) and the stream.

namespace boys {

/// Result status of the CUDA lane entry points.
///
/// The CUDA lane is the one fallible surface of this library: table
/// uploads, parameter validation, and launches report through this status
/// (never exceptions). \c kSuccess is 0. The enum carries no payload —
/// when \c kDeviceError is returned the caller can use the CUDA runtime's
/// own error reporting (cudaGetLastError, stream capture) for the detail.
enum class BoysStatus {
    kSuccess = 0, ///< the call succeeded
    kInvalidArgument, ///< a parameter was invalid (see the entry's contract)
    kDeviceError, ///< a CUDA operation failed
};

/// How the f32 single entry evaluates the region-B exponential e^{-x} that its
/// upward recursion carries.
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
/// \ingroup boys
enum class RegionBExp : int {
    /// expf: the library routine, the batch bodies' arithmetic, 2 ulp.
    kAccurate = 0,
    /// The hardware approximation with its argument-scaling residual removed.
    /// Pair it with the bound SingleF32 documents for it, and not with the
    /// lane's.
    kFast,
};

/// Device-side Boys evaluation over arrays of (n, x) inputs.
///
/// The coefficient tables are uploaded on first use by any entry point
/// (idempotent per device; switching devices re-uploads automatically).
/// The relaxed-multiplier degree tables are uploaded once per (device, m)
/// on the first call at that m — InitializeTables and the entries are
/// safe to call from multiple host threads.
///
/// Accuracy is a compile-time property of the call: the entry's
/// kAccuracyMultiplier selects the certified degree table its instantiation
/// is built with, which is why the first call at a new m uploads that
/// instantiation's tables. The lane has no per-call accuracy parameter, and
/// the multiplier is monotonically relaxing exactly as the CPU lanes
/// document it.
///
/// That is the one place the device surface is narrower than the CPU's. The
/// CPU's double lane takes an accuracy tier as a per-call argument
/// (BoysAllOrdersAtTier — one argument, all orders) over the rungs m = 1, 64,
/// 256, 1024, 4096, 16384, 65536, and answers "what would this tier deliver
/// here" before the call (QueryTier, AccuracyMultiplier, TierCoverage, which
/// names the region component that would limit it). The device has no such
/// parameter: a call is fixed at the m its instantiation was built with
/// (1, 2, 10, 100, 1e4, 1e8 — a finer set than the CPU's at the low end,
/// coarser at the top), the entry cannot be handed a tier at run time, and
/// nothing here reports what an m delivers: that is m * B_region from the
/// contract table, which the caller computes. A caller carrying a CPU tier
/// gets the nearest device behaviour by naming, at the call site, the largest
/// instantiated m whose bound does not exceed the tier's own.
///
/// **The device lane does not honour the multiply-add route, and this is a
/// stated limitation rather than an untested property.** The host's routes are
/// selected at build time (`BOYS_MULADD_SEPARATE`) and reported by
/// `BoysBackends()`, whose `route` field is what the corresponding arithmetic
/// was *measured* to deliver. The device kernels are not on that report and do
/// not read the selection: they name the fused intrinsic (`__fma_rn`)
/// directly, so the device arithmetic is fused whatever the host build
/// selected, and a caller who asked for the separate route gets a second
/// rounding on the host and a single one on the device. Nothing in a returned
/// value shows it and no entry reports it — a caller has to read the kernel or
/// know this paragraph.
///
/// This is a work item and not an impossibility. The fused intrinsic is what
/// makes it unfixable today; a kernel written as a bare product-plus-add leaves
/// the choice to the device compiler's own contraction setting, which is the
/// same mechanism the host build uses to deliver the two routes, and the device
/// lane would then honour the route the same way the scalar backends do. Until
/// that lands, treat the device lane as fused unconditionally: it is the fused
/// route's arithmetic, and the fused route's bounds are the ones its values
/// hold, whatever the host build selected.
///
/// The three shapes per precision family: Single* (one order per argument),
/// AllOrders* (all orders per argument, top order per element), AllN* (all
/// orders at every argument, one common top order).
///
/// One entry carries a second axis. The f32 single entry's region-B
/// exponential is a certified choice of arithmetic — two options, two measured
/// bounds (RegionBExp) — where every other axis of this surface is a choice of
/// shape or of accuracy multiplier.
///
/// All entries are asynchronous: the kernel is queued on the caller's
/// stream and the call returns once the launch is accepted (errors are
/// reported by the return status). Synchronize the stream (or use
/// cudaStreamSynchronize on a per-call stream) before reading the outputs.
///
/// An empty batch (count == 0) is a no-op in every entry of every family: no
/// kernel is queued (a zero-block launch is a CUDA error), the output is
/// untouched, and the call returns kSuccess. The AllN* entries validate nmax
/// before they look at count, so an out-of-range nmax is kInvalidArgument
/// whether or not the batch is empty.
///
/// \ingroup boys
class BoysCuda {
public:
    /// Uploads the Chebyshev coefficient tables (double and float lanes) to
    /// the current CUDA device. Idempotent; synchronous with respect to the
    /// host.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus InitializeTables();

    /// F_n(x[i]) in single precision — the recommended GPU lane on
    /// consumer hardware (double precision runs there at a fraction of
    /// single-precision throughput).
    ///
    /// Two bounds, one per region-B exponential (RegionBExp), each derived from
    /// the condition number of that region's recurrence and confirmed by the
    /// device gate's sweep:
    ///
    ///  - \c RegionBExp::kAccurate, the default: |F̂ − F| ≤ m * 1.5e-7, the f32
    ///    lane's documented bound, in every region. Measured at m = 1: 1.29e-7
    ///    at its worst, 0.86 of the bound.
    ///  - \c RegionBExp::kFast: |F̂ − F| ≤ m * 1.5e-7 + 8e-8 — the lane's bound
    ///    plus the corrected seed's own contribution, which the recurrence's
    ///    amplification caps at 8e-8. Measured at m = 1: 1.44e-7 at its worst,
    ///    0.63 of that bound; the contribution itself is 5.0e-8, and the option
    ///    returns the function's sign at every cell the gate audits.
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 1.5e-7 via compile-time Chebyshev degree truncation. Monotone
    ///   in m.
    /// \tparam kExp which region-B exponential the call runs. Both are
    ///   certified at every multiplier this lane instantiates, each against its
    ///   own bound above; nothing here substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
              RegionBExp kExp = RegionBExp::kAccurate>
    static BoysStatus SingleF32(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in single precision per input (i) — all orders
    /// at every argument, the top order read per element.
    ///
    /// Output layout: out[order * count + i] = F_order(x[i]), order 0..nmax[i]:
    /// plane l holds F_l(x[i]) for the arguments whose order reaches l. For a
    /// batch whose arguments share one top order — every plane fully
    /// populated, and no order array to upload — use AllNF32.
    /// The region-A seed is computed in double precision on the device — the
    /// downward recursion amplifies float seed errors beyond the certified
    /// 1.5e-7 float budget (same reasoning as the CPU float batch).
    ///
    /// \tparam kAccuracyMultiplier as SingleF32; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry (the F0
    ///   seed's error reaches every output with amplification A_B(l), at most
    ///   1 + 1.846e-17 over the supported orders).
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax, single precision — every
    /// order at every argument of the batch, in one launch.
    ///
    /// Output layout: out[k * count + i] = F_k(x[i]), k = 0..nmax — the
    /// order-major planes the CPU lane's BoysAllN returns, so a batch moved
    /// between the CPU and the device lanes is not transposed.
    ///
    /// The CPU float lane stops at BoysSingleF32 and BoysAllOrdersF32: it has
    /// no uniform-order batch, so this entry (and AllNF16, which mirrors it)
    /// adds a shape the CPU float lane does not have, and takes its layout
    /// from the double lane's BoysAllN.
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count): the arguments are
    ///      non-decreasing. The kernel classifies each argument itself, so an
    ///      unsorted batch still returns correct values; what the ordering
    ///      buys is the classification landing on one path per warp. This
    ///      entry does not sort its arguments (an internal sort is a launch, a
    ///      permutation and device scratch, and the caller that builds x is
    ///      the one that knows its order): a batch in arbitrary order is
    ///      AllOrdersF32 with the order array set to nmax.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF32; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) floats
    /// \param count  number of arguments
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllNF32(
        int nmax, const double* x, float* out, std::size_t count, void* stream);

    /// F_n(x[i]) in double precision, |error| <= 5.5e-14 (the double single
    /// lane's loosest per-region bound; the others are tighter).
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 5.5e-14 via compile-time Chebyshev degree truncation.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus SingleF64(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in double precision per input (i) — shape and
    /// layout as AllOrdersF32 (a per-element top order, order-major planes).
    ///
    /// This is the entry the CPU's run-time tier entry is shaped like
    /// (BoysAllOrdersAtTier is one argument, all orders, double), and it takes
    /// no tier: the multiplier is the template argument below, fixed where the
    /// call site names it. The class contract states the whole of that
    /// asymmetry.
    ///
    /// \tparam kAccuracyMultiplier as SingleF64; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax, double precision — the
    /// uniform-order batch: many arguments, all nmax + 1 orders each, the
    /// layout AllNF32 documents and the bound the CPU lane's BoysAllN
    /// documents, |F_hat - F| <= m * 5.5e-14 (the double batch lane's
    /// per-region budget, the same in every region).
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count) — the ordering contract
    ///      AllNF32 states, for the reason it states there.
    ///
    /// \tparam kAccuracyMultiplier as SingleF64; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) doubles
    /// \param count  number of arguments
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllNF64(
        int nmax, const double* x, double* out, std::size_t count, void* stream);

#if BoysFp16
    /// F_n(x[i]) in fp16 — the fp16 lane of the certified mixed-precision
    /// boundary (behind the BoysFp16 seam). Device pointers and stream
    /// contract as the F32/F64 entries; the fp16 values are the raw
    /// IEEE-754 binary16 bit patterns of this library's F16 type (see
    /// f16.hpp), so host copies are plain byte copies of count * sizeof(F16).
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 1e-7 + 1/2 ULP via compile-time Chebyshev degree truncation.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in fp16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when a device operation fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus SingleF16(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in fp16 per input (i), layout as AllOrdersF32
    /// (out[order * count + i] = F_order(x[i])), device pointers and
    /// stream contract as SingleF16.
    ///
    /// \tparam kAccuracyMultiplier as SingleF16; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when a device operation fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF16(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax in fp16 — the uniform-order
    /// batch of the fp16 lane, layout as AllNF32, device pointers and stream
    /// contract as SingleF16. Like AllNF32 it has no CPU fp16 counterpart (the
    /// CPU fp16 lane stops at BoysSingleF16 and BoysAllOrdersF16).
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count) — the ordering contract
    ///      AllNF32 states, for the reason it states there.
    ///
    /// \tparam kAccuracyMultiplier as SingleF16; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) fp16 values
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when a device operation fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllNF16(int nmax, const F16* x, F16* out, std::size_t count, void* stream);
#endif // BoysFp16
};

} // namespace boys
