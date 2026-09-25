#pragma once

#include "boys/boys.hpp"
#include "boys/boys_device_tables.hpp"

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

/// Device-side Boys evaluation over arrays of (n, x) inputs.
///
/// The coefficient tables are uploaded on first use by any entry point
/// (idempotent per device; switching devices re-uploads automatically).
/// InitializeTables and the entries are safe to call from multiple host
/// threads.
///
/// Accuracy is a compile-time property of a batch call: the entry's
/// kAccuracyMultiplier selects the certified degree table its instantiation
/// is built with, which is why the first call at a new m uploads that
/// instantiation's tables. The multiplier is monotonically relaxing exactly as
/// the CPU lanes document it, and the rungs this lane instantiates are
/// 1, 2, 10, 100, 1e4 and 1e8 — a finer set than the CPU tier lane's at the low
/// end and coarser at the top.
///
/// The device-callable entries (boys_cuda_device.hpp) are at once the wider
/// surface and the narrower one. Wider, because each takes the multiplier as a
/// run-time argument and reads the rung the caller names, so one compiled entry
/// serves every rung. Narrower, because what those rungs are is the one handle's
/// tables, and one relaxed rung of them is resident at a time: filling a handle
/// at a relaxed rung replaces the resident one, and an entry asked for a rung
/// that is not resident returns BoysDeviceStatus::kMultiplierNotResident and
/// writes nothing — a value the caller branches on, not a silent choice of
/// whichever rung happens to be resident. A call at
/// kBoysFullAccuracyMultiplier reads tables that are always uploaded, so it is
/// served whatever rung is resident and does not depend on that choice.
///
/// What neither surface has is the CPU double lane's per-call tier machinery:
/// BoysAllOrdersAtTier (one tier, all orders), QueryTier, AccuracyMultiplier and
/// TierCoverage, the last of which names the region component that would limit a
/// tier. No device entry reports what an m delivers, because m is the input and
/// not a selection from a table: the answer is m * B_region from the contract,
/// which the caller computes. A caller carrying a CPU tier gets the nearest
/// device behaviour by choosing, at the call site, the largest instantiated m
/// whose bound does not exceed the tier's own.
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
/// bounds (boys::RegionBExp in boys_device_tables.hpp) — where every other axis
/// of this surface is a choice of shape or of accuracy multiplier. The
/// device-callable single entry of the same precision takes the same option, as
/// a template argument, so the two lanes' f32 single entries carry one choice
/// between them and not one each.
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

    /// Fills the handle the device-callable entries read
    /// (boys_cuda_device.hpp) for the current device, uploading the tables if
    /// they are not uploaded yet — idempotent, on the same terms as
    /// InitializeTables. An entry reads this handle at the rung this call
    /// named.
    ///
    /// The handle's full-accuracy tables are uploaded by the call whatever
    /// multiplier it names, and they are the whole of what a call at
    /// kBoysFullAccuracyMultiplier needs: it has no relaxed table to make
    /// resident and leaves whatever rung already is, so a caller may fill one
    /// handle at a relaxed rung and then run both rungs through it.
    ///
    /// One relaxed rung is resident at a time, process-wide per device. Filling
    /// a handle at a relaxed rung replaces the resident one, and every entry
    /// then asked for the rung it replaced returns
    /// BoysDeviceStatus::kMultiplierNotResident rather than reading tables that
    /// now hold another rung. A caller that wants a relaxed rung makes it
    /// resident here before the kernel that reads it is launched.
    ///
    /// \tparam kAccuracyMultiplier the rung to make resident, one of the
    ///   multipliers this lane instantiates (1, 2, 10, 100, 1e4, 1e8), matched
    ///   against the entries' own multiplier argument exactly.
    /// \param out the handle to fill; untouched when the call fails
    ///
    /// \pre \c out is a valid pointer to one BoysDeviceTables.
    ///
    /// \returns kSuccess after filling \c out with the current device's table
    /// addresses and, for a relaxed rung, after that rung's degree tables are
    /// resident; kDeviceError when the upload or an address query fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus DeviceTables(BoysDeviceTables* out);

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
    /// call site names it, and the entry cannot be handed one at run time. The
    /// device-callable entry of the same shape (BoysDeviceAllOrdersF64) can, and
    /// the class contract states what that costs it.
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
