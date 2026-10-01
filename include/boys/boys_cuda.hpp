#pragma once

#include "boys/boys.hpp"
#include "boys/boys_cuda_options.hpp"
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

// ---------------------------------------------------------------------------
// The device option space.
//
// One row per option of the surface below, with what a chooser needs to place
// it, in boys_cuda_options.hpp. The rows are declared there and not here
// because they are one table with two readers: this header's entries are one,
// and the device translation units that implement the entries of
// boys_cuda_device.hpp are the other — those are compiled by nvcc, which cannot
// take this header (it pulls in the whole library through boys.hpp). The
// enumerators, the row struct and BoysDeviceOptions() are therefore in the
// small header, and it is included here so that a caller who reads this surface
// has the space in the same include.
// ---------------------------------------------------------------------------

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
/// the CPU lanes document it, and the rungs this lane instantiates are the
/// twelve of kDeviceRungs (boys_cuda_options.hpp): the option space's seven —
/// 1, 64, 256, 1024, 4096, 16384 and 65536, the multipliers the CPU tier lane's
/// AccuracyTier names — beside the lane's own six, the finer set at the low end
/// and the coarser one at the top, 1, 2, 10, 100, 1e4 and 1e8. The two sets meet
/// at m = 1 alone. The rung a caller names is therefore a rung the lane serves,
/// and a caller carrying a tier names it here instead of approximating it by the
/// nearest member of another set.
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
/// **One rung is resident, and the size of the rung set does not change that.**
/// What a larger set changes is which multipliers can be made resident, not what
/// residency is or what a switch costs. One rung is resident because one rung is
/// one cut of every table the lane holds, and the batch kernels read their own
/// cut from the constant bank, where the six lanes' degree tables are 2574 ints:
/// twelve rungs of them would be 121 KB against the 64 KB it has. So a caller
/// moving along the ladder pays a fill on each switch, and what a filled handle
/// holds is unchanged: the addresses of the resident rung's degree tables, the
/// address of the scalar naming it, and the full-accuracy tables, which no fill
/// retires.
///
/// **A combination is a name and a rung is an argument, on this surface too.**
/// Every named entry of this class that queues a kernel carries its multiplier
/// as a template argument, fixed where the call site is written, and every one
/// of them has an \c AtRung sibling whose first argument is that multiplier as a
/// value: \c AllOrdersF64AtRung(4096.0, n, x, out, count, stream) is the same
/// call with the rung decided where the call is made. The two decisions a call
/// site makes then come apart the way the CPU surface's do — the combination is
/// written once as a name and resolved where it is written, and the rung is read
/// off what the caller knows at the call — and a caller moving along the ladder
/// writes no switch over the twelve.
///
/// An \c AtRung call makes the rung it named resident and runs that rung's own
/// launcher, so the rung it answers at is the rung it was handed and never
/// another. It follows that an \c AtRung call at a relaxed rung retires whichever
/// other relaxed rung was resident — there is one set of relaxed tables and one
/// rung of them, as the paragraph above states — and that a device-callable entry
/// asked for the retired rung then reports it rather than reading tables that
/// hold another rung. A call at m = 1 has no relaxed table to make resident,
/// leaves the resident rung where it is, and retires nothing.
///
/// A multiplier that is not one of \c kDeviceRungs is
/// \c BoysStatus::kInvalidArgument, nothing is launched and the caller's output
/// is untouched. That is the refusal this surface makes and the device-callable
/// entries make as \c BoysDeviceStatus::kMultiplierNotResident: the lane answers
/// at twelve rungs, a multiplier outside them is resident at none of them and can
/// never be, and a call that answered at whichever rung happened to be resident
/// instead would be exactly the outcome the rung argument exists to rule out.
///
/// What neither surface has is the CPU double lane's per-call tier machinery:
/// BoysAllOrdersAtTier (one tier, all orders), QueryTier, AccuracyMultiplier and
/// TierCoverage, the last of which names the region component that would limit a
/// tier. No device entry reports what an m delivers, because m is the input and
/// not a selection from a table: the answer is m * B_region from the contract,
/// which the caller computes. What a caller carrying a CPU tier does not have to
/// do is approximate it: every tier's multiplier is one of the rungs this lane
/// serves, so AccuracyMultiplier(tier) is what the entry takes — the template
/// argument of a batch call, and the run-time argument of a device-callable one,
/// which is served once BoysCuda::DeviceTables has made that rung resident.
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
/// This is unbuilt work rather than an impossibility: a kernel written as a bare
/// product-plus-add leaves the choice to the device compiler's own contraction
/// setting, the same mechanism the host build uses to deliver the two routes, and
/// the device lane would then honour the route the way the scalar backends do.
/// Until that lands, treat the device lane as fused unconditionally: the fused
/// route's bounds are the ones its values hold, whatever the host build selected.
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
    /// \tparam kAccuracyMultiplier the rung to make resident, one of
    ///   kDeviceRungs: the option space's rungs (1, 64, 256, 1024, 4096, 16384,
    ///   65536) beside this lane's own (1, 2, 10, 100, 1e4, 1e8), matched
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
    ///  - \c RegionBExp::kAccurate, \c kDefaultRegionBExp and the default:
    ///    |F̂ − F| ≤ m * 1.5e-7, the f32 lane's documented bound, in every
    ///    region. Measured at m = 1: 1.29e-7 at its worst, 0.86 of the bound.
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
    /// \tparam kExp which region-B exponential the call runs; the default is
    ///   \c kDefaultRegionBExp, the lane's documented default
    ///   (\c RegionBExp::kAccurate). Both are certified at every multiplier
    ///   this lane instantiates, each against its own bound above; nothing here
    ///   substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
              RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus SingleF32(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c SingleF32 at a rung named in the call, over the twelve rungs this lane
    /// serves.
    ///
    /// The same entry with the multiplier as the call's first argument instead of
    /// a template argument: the rung is made resident by this call and that
    /// rung's own launcher runs, so the values are the ones the template spelling
    /// at that multiplier returns. \c kExp stays the template argument it is on
    /// the entry, because it selects which arithmetic runs and not how much
    /// accuracy is bought. The class contract states what an \c AtRung call makes
    /// resident, what it retires, and what it refuses.
    ///
    /// \tparam kExp which region-B exponential the call runs; as \c SingleF32
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus SingleF32AtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

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

    /// \c AllOrdersF32 at a rung named in the call, over the twelve rungs this
    /// lane serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32AtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

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

    /// \c AllNF32 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) floats
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane or \c nmax is outside
    /// [0, kMaxBoysOrder], nothing launched and nothing written; kDeviceError
    /// when the table upload or the launch fails.
    static BoysStatus AllNF32AtRung(
        double multiplier, int nmax, const double* x, float* out, std::size_t count,
        void* stream);

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

    /// \c SingleF64 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus SingleF64AtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in double precision per input (i) — shape and
    /// layout as AllOrdersF32 (a per-element top order, order-major planes).
    ///
    /// This is the entry the CPU's run-time tier entry is shaped like
    /// (BoysAllOrdersAtTier is one argument, all orders, double). The tier is
    /// named here the way this surface names it — \c AllOrdersF64AtRung takes
    /// the multiplier as the call's first argument, and \c AllOrdersF64 takes
    /// it as the template argument below — and the two spellings of one rung
    /// are one instantiation and one launch rather than two arithmetics. The
    /// device-callable entry of the same shape (BoysDeviceAllOrdersF64) takes
    /// the rung as an argument too, and the class contract states what that
    /// costs it.
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

    /// \c AllOrdersF64 at a rung named in the call, over the twelve rungs this
    /// lane serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64AtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, with the library's second
    /// partition of the double lane's fits: region A's pieces are cut per
    /// order instead of two to an order, and region B's seed is 5 pieces at
    /// degree 10 instead of one polynomial over the interval. Shape, layout,
    /// arguments and the multiplier are AllOrdersF64's.
    ///
    /// The narrow partition is the double lane's (its effective degrees are
    /// derived for the roles that evaluate those fits, see
    /// boys_effective_degrees.hpp), so this is a double entry and there is no
    /// float or fp16 twin of it.
    ///
    /// A call at the full-accuracy multiplier reads the degrees the partition
    /// was stored at and uploads nothing, so it does not disturb a relaxed rung
    /// another call made resident.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64Narrow(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64Narrow at a rung named in the call, over the twelve rungs
    /// this lane serves. The rung is made resident by this call and that rung's
    /// own launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, with region A read as one fit per
    /// order: every order's own piece is located and its own fit summed, where
    /// the shipped entry seeds the top order's fit and brings the lower orders
    /// back down a recurrence. Shape, layout, arguments and the multiplier are
    /// AllOrdersF64's, and the values agree to the fit's own accuracy.
    ///
    /// The choice covers region A and nothing else: past kX0 this entry runs
    /// the certified all-orders body, which is the interval its claim names.
    /// The CPU lane's packing axis is the same choice (BoysPackAxes), and its
    /// orders member's interval is region A for the same reason.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64Orders(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64Orders at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) with both of the choices above in force: the
    /// narrow partition's pieces, read one fit per order inside region A and
    /// its piecewise region-B seed outside it, with the certified all-orders
    /// body past kX0.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64NarrowOrders(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64NarrowOrders at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowOrdersAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64 with the other evaluation scheme:
    /// every piece is summed in the monomial basis by Horner in ascending
    /// order, where the shipped entry sums the Chebyshev basis by a split
    /// Clenshaw. Shape, layout, arguments and the multiplier are AllOrdersF64's.
    ///
    /// The scheme is a choice of basis and not of fit: the two tables carry the
    /// same fit over the same pieces at the same degree, and each scheme's
    /// delivered accuracy on it is its own certified row (kSchemeRows,
    /// boys_coefficients.hpp). What the choice buys and costs is counted rather
    /// than asserted: one multiply-add per coefficient against the split
    /// Clenshaw's two, at the same stored table size and the same degree, so
    /// this entry is the cheaper summation of the two on hardware where the two
    /// issue alike, and the two are ranked by the option probe.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64Mono(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64Mono at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64MonoAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// The monomial scheme's orders-axis member: F_0(x[i])..F_n(x[i]) with
    /// region A read as one fit per order, each summed in the basis
    /// AllOrdersF64Mono names. The two choices compose — the axis is a body
    /// choice inside region A, and the scheme is the basis that body sums — so
    /// this is the same relation to AllOrdersF64Mono that AllOrdersF64Orders
    /// has to AllOrdersF64.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64OrdersMono(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64OrdersMono at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersMonoAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// The monomial scheme over the narrow partition: F_0(x[i])..F_n(x[i]) from
    /// the narrow pieces, their piecewise region-B seed and their per-rung
    /// effective degrees, summed in the basis AllOrdersF64Mono names.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64NarrowMono(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64NarrowMono at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowMonoAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// Both of the choices above in force at once: the narrow partition read
    /// one fit per order inside region A, summed in the monomial basis, with
    /// the certified all-orders body past kX0.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64NarrowOrdersMono(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64NarrowOrdersMono at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowOrdersMonoAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64 with the other fit route: every
    /// piece is a numerator/denominator pair, evaluated as two Horner sums and
    /// a division, over the same pieces and intervals and in the same region
    /// structure as the shipped entry. Shape, layout, arguments and the
    /// multiplier are AllOrdersF64's.
    ///
    /// The route is a choice of fit and not of scheme: the pair is stored once,
    /// so both scheme names select this arithmetic and the scheme axis is inert
    /// here (kEvalSchemeRows, boys_coefficients.hpp, carries each route's own
    /// certified figures). What the route buys and costs is the pair's degree
    /// against the polynomial's at the same budget — a numerator and a
    /// denominator summed, one division, and the route's own stored tables —
    /// and the two routes are ranked by the option probe.
    ///
    /// The rung's cut is the route's own, certified per reading: this shape
    /// seeds at its top order's piece and carries that piece's w(b), where
    /// AllOrdersF64OrdersRat reads each order's piece at A = 1.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64Rat(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64Rat at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64RatAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// The rational route's orders-axis member: F_0(x[i])..F_n(x[i]) with
    /// region A read as one fit per order, each summed as the pair
    /// AllOrdersF64Rat names. The two choices compose — the axis is a body
    /// choice inside region A, and the route is the family those bodies read —
    /// so this is the same relation to AllOrdersF64Rat that AllOrdersF64Orders
    /// has to AllOrdersF64. Its rung reads the per-order cut.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64OrdersRat(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64OrdersRat at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersRatAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// The rational route over the narrow partition: F_0(x[i])..F_n(x[i]) from
    /// the narrow pieces, their piecewise region-B seed and their per-rung
    /// effective degrees, each piece summed as the pair AllOrdersF64Rat names.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64NarrowRat(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64NarrowRat at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowRatAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// Both of the choices above in force at once: the narrow partition read
    /// one fit per order inside region A, each piece summed as the pair
    /// AllOrdersF64Rat names, with the certified all-orders body past kX0.
    ///
    /// \tparam kAccuracyMultiplier as AllOrdersF64: m = 1 reads the degrees the
    ///   tables were stored at, m > 1 this entry's certified table at that rung.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64NarrowOrdersRat(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// AllOrdersF64NarrowOrdersRat at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64NarrowOrdersRatAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, off the library's third cut of
    /// the double lane's domain: one *uniform* table — a single grid over
    /// [0, kFlatHi) at one interval width and one stored degree for every
    /// order, 245 intervals of width 1/7 at degree 7 (kFlatCoeffs,
    /// boys_coefficients.hpp). Shape, layout, arguments and the multiplier are
    /// AllOrdersF64's, and a call delivers inside the same documented double
    /// batch budget.
    ///
    /// The route is not a re-cut of the pieces the entries above read: every
    /// order is fitted on its own over each interval, so an argument's ladder
    /// is a set of independent polynomials rather than one recurrence seeded
    /// once and walked up. Nothing is built from anything else, which is what
    /// the table costs 505.3 KiB and a stored degree per order to buy, and it
    /// is why this entry's ladder is the shape the route is strongest at.
    ///
    /// Above kFlatHi the table reaches no further and the call falls to the
    /// one-term asymptotic and its own upward recurrence — the arm region C
    /// runs, from the lane's own prefactor and stepping by the same
    /// (l + 1/2)/x — so the join is not a third test written here but where
    /// this route's fit ends.
    ///
    /// **This entry serves every rung of \c kDeviceRungs, and a rung of it is
    /// the route's full-accuracy arithmetic.** The route's table is stored at one
    /// degree for every order and every interval, and that degree is admissible
    /// at every multiplier: the criterion spends a rung on the dropped-coefficient
    /// tail, and this table's stored degree drops none, so it meets every rung's
    /// budget. A call at a rung is therefore answered by the route's own
    /// coefficients and by nothing else, and the rung it names is the rung it
    /// makes resident rather than a second arithmetic. What a rung does *not* buy
    /// on this route is less work: a shorter degree would meet a looser budget and
    /// this lane derives none, so a call costs what the route costs at every
    /// multiplier, and the figure a rung of it is worth is this entry's bound
    /// times m.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs: the table's one degree is that rung's own arithmetic
    ///   at every one of them, so a value the lane does not hold is a
    ///   compile-time error rather than another arithmetic
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64Uniform(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64Uniform at a rung named in the call.
    ///
    /// Every rung of \c kDeviceRungs is served, for the reason the template
    /// above states; naming one launches the same kernel the template above
    /// launches, because the route's table is not cut per rung and there is no
    /// second arithmetic to run, and it makes the rung it named resident, so a
    /// device-callable entry asked at that rung afterwards finds it. A
    /// multiplier that is not a rung of this lane is refused, nothing launched
    /// and nothing written.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF64UniformAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) off the same uniform table as AllOrdersF64Uniform,
    /// summed in the other form the table stores: every order's coefficients
    /// are read from the monomial image of its own fit and summed by Horner in
    /// ascending order, where the entry above sums the Chebyshev image by a
    /// split Clenshaw. Shape, layout, arguments and the multiplier are
    /// AllOrdersF64Uniform's, and the grid arithmetic, the join at kFlatHi and
    /// the asymptotic arm above it are one body for both forms.
    ///
    /// The two images are one fit stored twice — the two rows of kFlatRows
    /// (boys_coefficients.hpp) — and each form's delivered accuracy on it is
    /// its own certified row. What the choice buys and costs is counted rather
    /// than asserted: one multiply-add per coefficient against the split
    /// Clenshaw's two, at the same stored table size and the same degree, so
    /// this entry is the cheaper summation of the two on hardware where the two
    /// issue alike, and the two are ranked by the option probe.
    ///
    /// **This entry serves every rung of \c kDeviceRungs**, for the reason
    /// AllOrdersF64Uniform states: the route's table is stored at one degree for
    /// every order and every interval, so no rung's criterion has a cut to make
    /// of it and every rung's arithmetic is that one degree.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason AllOrdersF64Uniform states
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64UniformHorner(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64UniformHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF64UniformAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF64UniformHornerAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64Uniform, with the route's region-A
    /// reading named at the packing axis: every order is read from its own fit
    /// over the argument's interval, and no order is built from another's.
    ///
    /// **This is the route's only reading of the grid, so this entry and
    /// AllOrdersF64Uniform run one kernel and deliver one arithmetic.** A grid
    /// that stores one fit per order and per interval has no seeded ladder to
    /// step, and the route's own fit refuses a band source and a region-B seed
    /// at compile time (UniformFit, boys_impl.hpp) rather than answering from
    /// another route's tables. The row is therefore the axis member named and
    /// not a second reading, which is what the sibling families give the same
    /// axis: a chooser reading this family across the packing axis is told that
    /// the uniform route holds one member of it, rather than finding the member
    /// absent and no reason given.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason AllOrdersF64Uniform states
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64OrdersUniform(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64OrdersUniform at a rung named in the call, with the
    /// contract of \c AllOrdersF64UniformAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersUniformAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// Both of the choices above at once: the uniform route's per-order reading
    /// summed in the other form its table stores, by Horner over the monomial
    /// image of every order's own fit.
    ///
    /// It is the pair \c AllOrdersF64OrdersUniform and \c AllOrdersF64UniformHorner
    /// name, composed, and it runs one kernel: the packing axis has one member
    /// on this route and the scheme axis two, so the four rows of the route are
    /// two readings and four names.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason AllOrdersF64Uniform states
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64OrdersUniformHorner(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64OrdersUniformHorner at a rung named in the call, with
    /// the contract of \c AllOrdersF64UniformAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersUniformHornerAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's own narrow partition:
    /// the lane's 218 pieces at degree 6 against the shipped route's pieces, the
    /// same shape, layout and arguments as AllOrdersF32.
    ///
    /// The two lane arguments the entry's body takes are the split the shipped
    /// float batch entry already makes, one partition down: region A's seed is
    /// the double lane's piece table — the downward recursion amplifies a float
    /// seed error past the float budget, which is why the float bodies take a
    /// seed lane at all — and the region-B seed is the float lane's own. So this
    /// entry is the float lane's narrow route and not a mixture of two lanes'.
    ///
    /// **This entry serves every rung of \c kDeviceRungs, and a rung of it is
    /// this lane's own cut of the partition.** The region-B degrees a rung
    /// leaves are derived over the float lane's pieces (NarrowRegionBDegrees at
    /// the float batch role, FillNarrowF32Lane in boys_cuda.cpp), uploaded with
    /// the rung's other tables and read by the rung's kernel
    /// (Lane32NarrowRelaxed). Region A needs no table beside them: this lane's
    /// region-A seed is the double lane's piece table, whose rung cut the same
    /// upload carries. So a call at a rung reads the stored fit short — less
    /// work than the reference call, at the figure that row states.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason above
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32Narrow(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32Narrow at a rung named in the call, with the contract of
    /// \c AllOrdersF32AtRung: a rung the entry serves is answered by that rung's
    /// own arithmetic, and every rung of \c kDeviceRungs is one of them.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// The same partition with the other summation: every piece of region A and
    /// region B is read from the monomial form of its own fit and summed by
    /// Horner, where \c AllOrdersF32Narrow sums the Chebyshev form by a split
    /// Clenshaw. Shape, layout, arguments and the rungs are that entry's.
    ///
    /// The two forms are one partition stored twice — the two rows of
    /// kNarrowARowsF32 and kNarrowBRowsF32 (boys_coefficients.hpp) — and each
    /// form's delivered accuracy on it is its own certified row, which is also
    /// why each has a rung cut of its own: the rung's degrees are read from the
    /// coefficients the row sums (NarrowRegionBDegrees at TailBasis::kMonomial,
    /// FillNarrowMonoF32Lane), so a degree cut from the Chebyshev form's tail is
    /// not this entry's to read.
    ///
    /// **This entry serves every rung of \c kDeviceRungs**, for the reason
    /// \c AllOrdersF32Narrow states: the cut is this lane's own, in this basis as
    /// in that one.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason above
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowMono(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowMono at a rung named in the call, with the contract
    /// of \c AllOrdersF32NarrowAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowMonoAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's own uniform table: the
    /// same grid the double route reads — one interval width, one stored degree
    /// for every order and every interval, the table stopping at kFlatHiF32
    /// where the closed form takes over — at the lane's degree of 4 rather than
    /// the double lane's 8. Shape, layout, arguments and the multiplier are
    /// AllOrdersF32's, and a call delivers inside the float lane's documented
    /// budget.
    ///
    /// **This entry serves every rung of \c kDeviceRungs**, for the reason
    /// \c AllOrdersF64Uniform states: the table is stored at one degree for every
    /// order and every interval, so no rung's criterion has a cut to make of it,
    /// every rung's arithmetic is that one degree, and a rung of it buys no less
    /// work.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs, for the reason \c AllOrdersF64Uniform states
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32Uniform(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32Uniform at a rung named in the call, with the contract of
    /// \c AllOrdersF64UniformAtRung: every rung of \c kDeviceRungs is served, a
    /// call at one makes that rung resident and queues the same kernel, and a
    /// multiplier that is not a rung of the lane is refused.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32UniformAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// The same grid summed in the other form the table stores: every order's
    /// coefficients are read from the monomial image of its own fit and summed
    /// by Horner, where \c AllOrdersF32Uniform sums the Chebyshev image by a
    /// split Clenshaw. Shape, layout, arguments and the multiplier are that
    /// entry's, and the two are ranked by the option probe.
    ///
    /// **This entry serves every rung of \c kDeviceRungs**, for the reason
    /// \c AllOrdersF32Uniform states.
    ///
    /// \tparam kAccuracyMultiplier the rung this instantiation is, one of
    ///   \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32UniformHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32UniformHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF32NarrowAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32UniformHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the uniform grid's RATIONAL member.
    ///
    /// The grid is the same one \c AllOrdersF32Uniform reads — the lane's own,
    /// one table of equal intervals over [0, kFlatHiF32) — and the route over it
    /// is the rational family: one numerator/denominator pair per interval, fitted
    /// in this lane's arithmetic and certified against this lane's bar, where the
    /// entry above sums the Chebyshev member's cell. The partition is not a second
    /// one: a caller naming this row and a caller naming that one are read at the
    /// same intervals, located by the same map.
    ///
    /// Every rung of the lane is served, for the reason the entry above states:
    /// the member stores one pair per interval and no per-order effective-degree
    /// column, so no rung's criterion has anything to cut, and a rung of it is the
    /// route's own arithmetic rather than a thinner one.
    ///
    /// The pair is stored in monomial form and read by Horner, so neither scheme
    /// name a caller may use reaches a second arithmetic and this row's Horner
    /// twin runs the same kernel. Both names are rows of the report, because a
    /// caller who named a scheme named a call.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32UniformRat(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32UniformRat at a rung named in the call, with the contract
    /// of \c AllOrdersF32UniformHornerAtRung: every rung of \c kDeviceRungs is
    /// served, a call at one makes that rung resident and queues the same kernel,
    /// and a multiplier that is not a rung of the lane is refused.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32UniformRatAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32UniformRat under the Horner scheme name. A forwarder and
    /// not a second row over a second table: the rational member is stored in one
    /// form, so both scheme names reach one arithmetic — the same relation the
    /// lane's shipped and narrow rational pairs stand in.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns what \c AllOrdersF32UniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32UniformRatHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32UniformRatHorner at a rung named in the call.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns what \c AllOrdersF32UniformRatAtRung returns, and its refusals
    /// with it: this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF32UniformRatHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// The same four on the double lane's grid, whose rational member is the
    /// double lane's own pair per interval over that lane's intervals. The
    /// contract, the rungs and the Horner relation are the float lane's above.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64UniformRat(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64UniformRat at a rung named in the call.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64UniformRatAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF64UniformRat under the Horner scheme name.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns what \c AllOrdersF64UniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF64UniformRatHorner(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// \c AllOrdersF64UniformRatHorner at a rung named in the call.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns what \c AllOrdersF64UniformRatAtRung returns, and its refusals
    /// with it: this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF64UniformRatHornerAtRung(
        double multiplier, const int* n, const double* x, double* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's rational route: region
    /// A's seed is read from the double lane's rational pair over the shipped
    /// partition — the same family and the same partition the double route's own
    /// rows read, and the same lane the float lane's other batch entries take
    /// their region-A seed from, because the downward recursion amplifies a
    /// float seed error past the float budget — and the region-B seed is the
    /// float lane's own rational pair over the same partition
    /// (kNarrowRatBCoeffsF32's shipped counterpart, kRatBnum and kRatBden).
    ///
    /// Shape, layout, arguments and the multiplier are \c AllOrdersF32's, and
    /// the delivered figure is the float lane's, because the seed the ladder
    /// amplifies and the ladder itself are the float lane's.
    ///
    /// **This entry and its narrow sibling are served at the full-accuracy
    /// multiplier only, and that is unbuilt work rather than a property of the
    /// route.** The region-A half of a relaxed call could be answered here —
    /// the double lane's rational cut for the batch reading is derived and
    /// uploaded, since the double route's own rows read it — but the region-B
    /// pair the float lane supplies is not, and a rung certified on one half of
    /// a body and not the other is a cut nobody derived. Lifting the refusal is
    /// deriving, uploading and reading the float lane's rational pair cut.
    ///
    /// \tparam kAccuracyMultiplier the multiplier this entry serves, which is
    ///   \c kBoysFullAccuracyMultiplier and no other; any other value is a
    ///   compile-time error rather than a second arithmetic
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32Rat(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32Rat at a rung named in the call, with the contract of
    /// \c AllOrdersF32NarrowAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32RatAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32Rat under the route's other scheme name. The route's pair
    /// is stored in monomial form and read by Horner, so neither name selects a
    /// second arithmetic: this entry runs the one kernel \c AllOrdersF32Rat
    /// runs, and the two rows exist because the scheme is an axis of this
    /// surface and a caller naming it must reach the combination it named. The
    /// double lane's route carries the same pair of names for the same reason.
    ///
    /// Shape, layout, arguments and the multiplier are \c AllOrdersF32Rat's.
    ///
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32Rat
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32RatHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32RatHorner at a rung named in the call, with the contract
    /// of \c AllOrdersF32RatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32RatHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32Rat over the float lane's narrow partition: region A's
    /// seed is the double lane's rational pair over the *narrow* pieces and
    /// region B's is the float lane's pair per narrow piece
    /// (kNarrowRatBCoeffsF32), which is the partition
    /// \c AllOrdersF32Narrow reads its Chebyshev fits from. The two lanes'
    /// readings of one piece therefore share its interval and its mapped
    /// argument, as the two routes over a piece do on the host lane.
    ///
    /// **This entry is served at the full-accuracy multiplier only**, for the
    /// reason \c AllOrdersF32Rat states.
    ///
    /// \tparam kAccuracyMultiplier the multiplier this entry serves, which is
    ///   \c kBoysFullAccuracyMultiplier and no other
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowRat(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowRat at a rung named in the call, with the contract
    /// of \c AllOrdersF32RatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowRatAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32NarrowRat under the route's other scheme name, with the
    /// contract of \c AllOrdersF32RatHorner.
    ///
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32Rat
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowRatHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowRatHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF32RatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowRatHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// F_0(x[i])..F_n(x[i]) as \c AllOrdersF32, with region A read as one fit per
    /// order: every order's own piece is located and its own fit summed, where
    /// the per-argument entry seeds the top order's fit and brings the lower
    /// orders back down a recurrence. Shape, layout, arguments and the multiplier
    /// are \c AllOrdersF32's, and the values agree to the fit's own accuracy.
    ///
    /// The choice covers region A and nothing else: past kX0 these entries run
    /// the certified all-orders body, which is the interval their claim names.
    /// The CPU lane's packing axis is the same choice (BoysPackAxes), and its
    /// orders member's interval is region A for the same reason.
    ///
    /// The rows of this axis mirror the double lane's, one per partition and
    /// route it carries: the shipped partition's entry and the narrow partition's
    /// four are instantiated at every rung of \c kDeviceRungs and read this
    /// lane's own cut of the table their row sums at a relaxed one, while the
    /// rational route's serve the full-accuracy multiplier alone — this lane
    /// derives, uploads and reads no rung cut of the float lane's rational pairs,
    /// so a rung of theirs would have to be answered by degrees no kernel here
    /// holds. Each entry states its own axis in \c DeviceEntryServedAtRung
    /// (boys_cuda_options.hpp); a call at a rung its entry does not serve is
    /// refused with nothing launched.
    ///
    /// The uniform grid's two orders entries are the route's one reading of the
    /// axis and launch the kernels their per-argument rows launch: the grid's
    /// cells each carry their own degree and block start, so an order's own value
    /// is the order's own block of its interval and there is no second reading of
    /// the axis to have.
    ///
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32: m = 1 reads the degrees
    ///   the tables were stored at, m > 1 this entry's certified table at that
    ///   rung, where the entry serves it.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32Orders(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32Orders at a rung named in the call, with the contract of
    /// \c AllOrdersF32RatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32Orders on the narrow partition, with the contract of
    /// \c AllOrdersF32Narrow: the partition's pieces one fit per order, and the
    /// same rungs — the cut a rung reads is the table's and not the reading's.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32Narrow
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowOrders(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowOrders at a rung named in the call, with the contract
    /// of \c AllOrdersF32NarrowAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32NarrowOrders in the monomial basis, with the contract of
    /// \c AllOrdersF32NarrowMono and the same rungs.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32NarrowMono
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowOrdersMono(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowOrdersMono at a rung named in the call, with the
    /// contract of \c AllOrdersF32NarrowMonoAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersMonoAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32Orders over the fit route's pairs, with the contract of
    /// \c AllOrdersF32Rat: each order's own piece read at A = 1, where the
    /// per-argument shape seeds at its top order's piece and carries that piece's
    /// w(b) down the recursion. The two scheme names select one arithmetic, as
    /// they do on the route's other shapes.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32Rat
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32OrdersRat(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32OrdersRat at a rung named in the call, with the contract of
    /// \c AllOrdersF32RatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersRatAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32OrdersRat under the route's other scheme name, over the same
    /// kernel and the same rung.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32RatHorner
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32OrdersRatHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32OrdersRatHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF32RatHornerAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersRatHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32OrdersRat on the narrow partition, with the contract of
    /// \c AllOrdersF32NarrowRat.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32NarrowRat
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowOrdersRat(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowOrdersRat at a rung named in the call, with the
    /// contract of \c AllOrdersF32NarrowRatAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersRatAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32NarrowOrdersRat under the route's other scheme name, over the
    /// same kernel and the same rung.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32NarrowRatHorner
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32NarrowOrdersRatHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32NarrowOrdersRatHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF32NarrowRatHornerAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane *or is a rung this entry does
    /// not serve*, nothing launched and nothing written; kDeviceError when the
    /// table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersRatHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32Uniform with the orders axis's own rows: the route's one
    /// reading of the grid, and therefore the kernel \c AllOrdersF32Uniform
    /// launches. The grid's cells each carry the degree and block start their own
    /// truncation bound gave them, so an order's value is read from its own block
    /// of the interval's table and there is no second reading of the axis here —
    /// the two rows are two *cells* of the space and one arithmetic, which is what
    /// the grid's own row states.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32Uniform
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32OrdersUniform(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32OrdersUniform at a rung named in the call, with the
    /// contract of \c AllOrdersF32UniformAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersUniformAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

    /// \c AllOrdersF32OrdersUniform over the grid's other stored form, with the
    /// contract of \c AllOrdersF32UniformHorner.
    /// \tparam kAccuracyMultiplier as \c AllOrdersF32UniformHorner
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus AllOrdersF32OrdersUniformHorner(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// \c AllOrdersF32OrdersUniformHorner at a rung named in the call, with the
    /// contract of \c AllOrdersF32UniformHornerAtRung.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersUniformHornerAtRung(
        double multiplier, const int* n, const double* x, float* out, std::size_t count,
        void* stream);

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

    /// AllNF64 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) doubles
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane or \c nmax is outside
    /// [0, kMaxBoysOrder], nothing launched and nothing written; kDeviceError
    /// when the table upload or the launch fails.
    static BoysStatus AllNF64AtRung(
        double multiplier, int nmax, const double* x, double* out, std::size_t count,
        void* stream);

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

    /// SingleF16 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in fp16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus SingleF16AtRung(
        double multiplier, const int* n, const F16* x, F16* out, std::size_t count,
        void* stream);

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

    /// AllOrdersF16 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane, nothing launched and nothing
    /// written; kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF16AtRung(
        double multiplier, const int* n, const F16* x, F16* out, std::size_t count,
        void* stream);

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

    /// AllNF16 at a rung named in the call, over the twelve rungs this lane
    /// serves. The rung is made resident by this call and that rung's own
    /// launcher runs; the class contract states what an \c AtRung call makes
    /// resident and what it refuses.
    ///
    /// \param multiplier the accuracy multiplier m, one of \c kDeviceRungs,
    ///   matched exactly against the rung this call makes resident
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) fp16 values
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kSuccess after the launch is queued; kInvalidArgument when
    /// \c multiplier is not a rung of this lane or \c nmax is outside
    /// [0, kMaxBoysOrder], nothing launched and nothing written; kDeviceError
    /// when the table upload or the launch fails.
    static BoysStatus AllNF16AtRung(
        double multiplier, int nmax, const F16* x, F16* out, std::size_t count, void* stream);
#endif // BoysFp16
};

} // namespace boys
