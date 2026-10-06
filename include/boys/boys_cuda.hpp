#pragma once

#include "boys/boys.hpp"
#include "boys/boys_cuda_muladd.hpp"
#include "boys/boys_cuda_options.hpp"
#include "boys/boys_device_tables.hpp"

#include <cstddef>
#include <string>

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
/// Accuracy is a property of the entry and not a choice of the call: an entry
/// evaluates the certified degree tables this library stores, and it documents
/// the bound those tables deliver. The device option space
/// (boys_cuda_options.hpp) carries that figure per row, and the device accuracy
/// gate certifies each entry against a committed high-precision reference grid.
/// A caller who needs a figure rather than a table of figures reads it off the
/// row of the entry it calls.
///
/// The device-callable entries (boys_cuda_device.hpp) are the same arithmetic
/// seen from inside a kernel: they read the one handle DeviceTables fills and
/// answer at the same entries' bounds. A caller that writes its own kernel and
/// calls one of them gets the entry's own arithmetic, not an approximation of
/// it.
///
/// A device entry documents one bound and answers at it: the library serves one
/// accuracy, and the entry a caller picks is the arithmetic and the figure it
/// gets. Nothing about a call selects a weaker arithmetic than the entry's own:
/// the division form a call may name is one of the three certified divisions of
/// that arithmetic's ladder steps.
///
/// **The device lane honours the multiply-add route, and reports it.** The
/// host's routes are selected at build time (`BOYS_MULADD_SEPARATE`) and
/// reported by `BoysBackends()`, whose `route` field is what the corresponding
/// arithmetic was *measured* to deliver. The device lane runs that selection
/// too: it is a public compile definition, so the kernels' translation unit and
/// a consumer's own inherit the value the host build reads, and every entry of
/// this class and every device-callable entry of boys_cuda_device.hpp runs the
/// arithmetic that selection names. `BoysCuda::MulAddRouteInForce()` is where a caller reads it —
/// the device's counterpart of `RouteInForce<T>()`, one selection covering both
/// precisions of the lane.
///
/// The device spells both routes out rather than leaving the choice to the
/// device compiler's contraction setting: the fused step as the fused intrinsic,
/// the separate step as a product rounded once and then summed, a form no
/// setting has anything left to fuse. The separate route is therefore two
/// roundings on the device as it is on the host, and the device delivers the
/// route it names — measured, not asserted: the CUDA route test holds each route
/// to the arithmetic it names, bit for bit, over a fixed value set.
///
/// The three shapes per precision family: Single* (one order per argument),
/// AllOrders* (all orders per argument, top order per element), AllN* (all
/// orders at every argument, one common top order).
///
/// Every entry that launches a kernel takes a division form (boys/accuracy.hpp)
/// — the entry's arithmetic with a different division of its ladder steps, all
/// three certified, kDefaultDeviceDivisionForm unless the caller names another; the
/// \c form parameter of each entry below names it. One entry carries a second
/// axis on top of it: the f32 single entry's region-B exponential
/// is a certified choice of arithmetic — two options, two measured bounds
/// (boys::RegionBExp in boys_device_tables.hpp) — where every other difference
/// between two entries of this surface is the shape (a different entry). The device-callable
/// single entry of the same precision takes the same option, as a template
/// argument, so the two lanes' f32 single entries carry one choice between them.
///
/// **Every class of this surface is also reachable by naming a policy.** Beside
/// the named entries above and below, each class carries a policy-templated member
/// of its own, `BoysCuda::AllOrdersF64WithPolicy<Policy>(...)`, that dispatches,
/// while the call site compiles, to the entry the policy names:
/// boys/boys_cuda_policy.hpp states the rule, the axes each class reads and the
/// defaults. \c Policy defaults to this build's own row for the class
/// (`DefaultPolicy<Precision::kFp64Device, Shape::kAllOrders, Device::kDevice>`),
/// so a call that names no policy reaches the kernel the build's defaults name,
/// and a policy whose combination has **no kernel** fails to compile, with the
/// combination named, rather than resolving to a nearby entry.
///
/// The member carries the entry's name with `WithPolicy` after it, and not the
/// entry's name alone, because a second function of a name makes that name an
/// **overload set**, and the address of an overload set cannot be taken where the
/// pointer type is deduced — a use this library's own tests make, handing
/// `&BoysCuda::AllOrdersF64` to a helper that deduces its launch type. With the
/// suffix each named entry stays the one function of its name: a call that names
/// one, an address taken by name, and a call naming the class with a division form
/// all reach exactly what they reached before.
///
/// All entries are asynchronous: the kernel is queued on the caller's
/// stream and the call returns once the launch is accepted (errors are
/// reported by the return status). Synchronize the stream (or use
/// cudaStreamSynchronize on a per-call stream) before reading the outputs.
///
/// An empty batch (count == 0) is a no-op in every entry of every family: no
/// kernel is queued (a zero-block launch is a CUDA error), the output is
/// untouched, and the call returns kSuccess. The AllN* entries validate nmax
/// before they look at count, and every entry that takes a division form
/// validates it before the launch, so neither a bad nmax nor a bad form is
/// answered by that no-op: an nmax outside [0, kMaxBoysOrder] is
/// kInvalidArgument on any call, and a form outside the three is
/// kInvalidArgument on the call that reaches the launch - an entry whose tables
/// cannot be made resident reports kDeviceError first.
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

    /// Fills the handle the device-callable entries read (boys_cuda_device.hpp)
    /// for the current device, uploading the tables if they are not uploaded yet —
    /// idempotent, on the same terms as InitializeTables. A handle is filled once
    /// and read by every device-callable entry; nothing about it is per-call.
    ///
    /// \param out the handle to fill; untouched when the call fails
    ///
    /// \pre \c out is a valid pointer to one BoysDeviceTables.
    ///
    /// \returns kSuccess after filling \c out with the current device's table
    /// addresses; kDeviceError when the upload or an address query fails.
    static BoysStatus DeviceTables(BoysDeviceTables* out);

    /// The multiply-add route the device arithmetic of this lane runs.
    ///
    /// The device's counterpart of the host's \c RouteInForce<T>() and of the
    /// \c route field of \c BoysBackends(): one computation serves both
    /// precisions of the lane, so one reader answers for both. It is the build's
    /// \c BOYS_MULADD_SEPARATE selection — a public compile definition, so the
    /// value a consumer's own translation unit reads is the value the kernels
    /// were compiled with.
    ///
    /// It is a report and not a restatement of the selection: the device spells
    /// both routes out rather than leaving the choice to the device compiler's
    /// contraction setting, so the arithmetic delivers the route it names. That
    /// the delivered arithmetic is the named one is measured — the CUDA route
    /// test holds each route to its own reference, bit for bit, over a fixed
    /// value set.
    ///
    /// \returns the route every entry of this class, and every device-callable
    ///          entry of boys_cuda_device.hpp, delivers on this device
    static constexpr backend::MulAddRoute MulAddRouteInForce() noexcept {
        return detail::kDeviceMulAddRoute;
    }

    /// F_n(x[i]) in single precision — the recommended GPU lane on consumer
    /// hardware (double precision runs there at a fraction of single-precision
    /// throughput).
    ///
    /// Two bounds, one per region-B exponential (RegionBExp), each derived from the
    /// condition number of that region's recurrence and confirmed by the device
    /// gate's sweep:
    ///
    ///  - \c RegionBExp::kAccurate, \c kDefaultRegionBExp and the default:
    ///    |F̂ − F| ≤ 1.5e-7, the f32 lane's documented bound, in every region.
    ///    Measured: 1.29e-7 at its worst, 0.86 of the bound.
    ///  - \c RegionBExp::kFast: |F̂ − F| ≤ 1.5e-7 + 8e-8 — the lane's bound plus
    ///    the corrected seed's own contribution, which the recurrence's amplification
    ///    caps at 8e-8. Measured: 1.44e-7 at its worst, 0.63 of that bound;
    ///    the contribution itself is 5.0e-8, and the option returns the function's
    ///    sign at every cell the gate audits.
    ///
    /// \tparam kExp which region-B exponential the call runs, default
    ///   \c kDefaultRegionBExp (\c RegionBExp::kAccurate). Both are certified,
    ///   each against its own bound above; nothing substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus SingleF32(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the kernel \c Policy names, chosen while the call site compiles, run at the
    /// policy's division form. This is the one class of the surface whose rows
    /// differ by the region-B exponential, so a policy naming either member
    /// reaches that member's entry.
    ///
    /// **This class's entry is itself a template** (`SingleF32<RegionBExp>`), and
    /// this member is a different name, so the two do not compete: the call naming
    /// the entry keeps reaching it, and a call of this member naming no policy
    /// reaches this build's row for the class, whose own member is that entry's
    /// default region-B exponential (\c kDefaultDeviceRegionBExp, boys/accuracy.hpp).
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for; a
    ///   combination it has none for is a compile error naming the combination.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp32Device, Shape::kSingle, Device::kDevice>>
    static BoysStatus SingleF32WithPolicy(
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
    /// The order-0 region-B entry's error reaches every output with
    /// amplification A_B(l), at most 1 + 1.846e-17 over the supported orders.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF32(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the route, scheme, partition and packing axis the policy names, dispatched
    /// to this class's entry for that combination while the call site compiles,
    /// run at the policy's division form. A combination this class has no kernel
    /// for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp32Device, Shape::kAllOrders, Device::kDevice>>
    static BoysStatus AllOrdersF32WithPolicy(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax, single precision — every order
    /// at every argument of the batch, in one launch.
    ///
    /// Output layout: out[k * count + i] = F_k(x[i]), k = 0..nmax — the order-major
    /// planes the CPU lane's BoysAllN returns, so a batch moved between the CPU and
    /// the device lanes is not transposed. The CPU float lane has no uniform-order
    /// batch, so this entry (and AllNF16, which mirrors it) adds a shape that lane
    /// does not have.
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count): the arguments are
    ///      non-decreasing. An unsorted batch still returns correct values — the
    ///      kernel classifies each argument itself — but the ordering lands that
    ///      classification on one path per warp. This entry does not sort its
    ///      arguments (an internal sort is a launch, a permutation and device
    ///      scratch, and the caller that builds x is the one that knows its order):
    ///      a batch in arbitrary order is AllOrdersF32 with the order array set to
    ///      nmax.
    ///
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) floats
    /// \param count  number of arguments
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when the launch fails.
    static BoysStatus AllNF32(
        int nmax,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the region-B exponential is the axis this class reads, and a policy naming
    /// the member the class has no kernel for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp32Device, Shape::kAllN, Device::kDevice>>
    static BoysStatus AllNF32WithPolicy(
        int nmax, const double* x, float* out, std::size_t count, void* stream);

    /// F_n(x[i]) in double precision, |error| <= 5.5e-14 (the double single
    /// lane's loosest per-region bound; the others are tighter).
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus SingleF64(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the kernel \c Policy names, chosen while the call site compiles, run at the
    /// policy's division form. The region-B exponential is the axis this class
    /// reads; a policy naming the member the class has no kernel for is a compile
    /// error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp64Device, Shape::kSingle, Device::kDevice>>
    static BoysStatus SingleF64WithPolicy(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in double precision per input (i) — shape and
    /// layout as AllOrdersF32 (a per-element top order, order-major planes).
    ///
    /// The device-callable entry of the same shape (BoysDeviceAllOrdersF64)
    /// answers at the same bound.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the route, scheme, partition and packing axis the policy names, dispatched
    /// to this class's entry for that combination while the call site compiles,
    /// run at the policy's division form. A combination this class has no kernel
    /// for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp64Device, Shape::kAllOrders, Device::kDevice>>
    static BoysStatus AllOrdersF64WithPolicy(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, with the library's second partition of
    /// the double lane's fits: region A's pieces are cut per order instead of two to
    /// an order, and region B's seed is 5 pieces at degree 10 instead of one
    /// polynomial over the interval. Shape, layout and arguments are
    /// AllOrdersF64's.
    ///
    /// The partition is the double lane's, so there is no float or fp16 twin of this
    /// entry (its effective degrees are derived for the roles that evaluate those
    /// fits, see boys_effective_degrees.hpp).
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64Narrow(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, with region A read as one fit per order:
    /// every order's own piece is located and its own fit summed, where the coarsest
    /// entry seeds the top order's fit and brings the lower orders back down a
    /// recurrence. Shape, layout and arguments are AllOrdersF64's, and
    /// the values agree to the fit's own accuracy.
    ///
    /// The choice covers region A and nothing else: past kX0 this entry runs the
    /// certified all-orders body, which is the interval its claim names. The CPU
    /// lane's packing axis is the same choice (BoysPackAxes).
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64Orders(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) with both of the choices above in force: the
    /// narrow partition's pieces, read one fit per order inside region A and
    /// its piecewise region-B seed outside it, with the certified all-orders
    /// body past kX0.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64NarrowOrders(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64 with the other evaluation scheme: every
    /// piece is summed in the monomial basis by Horner in ascending order, where the
    /// coarsest entry sums the Chebyshev basis by a split Clenshaw. Shape, layout,
    /// arguments are AllOrdersF64's.
    ///
    /// The scheme is a choice of basis and not of fit: the two tables carry the same
    /// fit over the same pieces at the same degree, and each scheme's delivered
    /// accuracy is its own certified row (kSchemeRows, boys_coefficients.hpp). What
    /// the choice buys is counted rather than asserted: one multiply-add per
    /// coefficient against the split Clenshaw's two, at the same stored table size and
    /// the same degree, so this entry is the cheaper summation where the two issue
    /// alike; the two are ranked by the option probe.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64Mono(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The monomial scheme's orders-axis member: F_0(x[i])..F_n(x[i]) with
    /// region A read as one fit per order, each summed in the basis
    /// AllOrdersF64Mono names. The two choices compose — the axis is a body
    /// choice inside region A, and the scheme is the basis that body sums — so
    /// this is the same relation to AllOrdersF64Mono that AllOrdersF64Orders
    /// has to AllOrdersF64.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64OrdersMono(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The monomial scheme over the narrow partition: F_0(x[i])..F_n(x[i]) from
    /// the narrow pieces, their piecewise region-B seed and their own
    /// effective degrees, summed in the basis AllOrdersF64Mono names.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64NarrowMono(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// Both of the choices above in force at once: the narrow partition read
    /// one fit per order inside region A, summed in the monomial basis, with
    /// the certified all-orders body past kX0.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64NarrowOrdersMono(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64 with the other fit route: every piece is
    /// a numerator/denominator pair, evaluated as two Horner sums and a division,
    /// over the same pieces and intervals and in the same region structure as the
    /// coarsest entry. Shape, layout and arguments are
    /// AllOrdersF64's.
    ///
    /// The route is a choice of fit and not of scheme: the pair is stored once, so
    /// both scheme names select this arithmetic and the scheme axis is inert here
    /// (kEvalSchemeRows, boys_coefficients.hpp, carries each route's own certified
    /// figures). What the route buys is the pair's degree against the polynomial's at
    /// the same budget — a numerator and a denominator summed, one division, and the
    /// route's own stored tables — and the two routes are ranked by the option probe.
    ///
    /// The cut is the route's own, certified per reading: this shape
    /// seeds at its top order's piece and carries that piece's w(b), where
    /// AllOrdersF64OrdersRat reads each order's piece at A = 1.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64Rat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The rational route's orders-axis member: F_0(x[i])..F_n(x[i]) with
    /// region A read as one fit per order, each summed as the pair
    /// AllOrdersF64Rat names. The two choices compose — the axis is a body
    /// choice inside region A, and the route is the family those bodies read —
    /// so this is the same relation to AllOrdersF64Rat that AllOrdersF64Orders
    /// has to AllOrdersF64. It reads the per-order cut.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64OrdersRat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The rational route over the narrow partition: F_0(x[i])..F_n(x[i]) from
    /// the narrow pieces, their piecewise region-B seed and their own
    /// effective degrees, each piece summed as the pair AllOrdersF64Rat names.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64NarrowRat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// Both of the choices above in force at once: the narrow partition read
    /// one fit per order inside region A, each piece summed as the pair
    /// AllOrdersF64Rat names, with the certified all-orders body past kX0.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    static BoysStatus AllOrdersF64NarrowOrdersRat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64, off the library's third cut of
    /// the double lane's domain: one *uniform* table — a single grid over
    /// [0, kFlatHi) at one interval width and one stored degree for every
    /// order, 245 intervals of width 1/7 at degree 7 (kFlatCoeffs,
    /// boys_coefficients.hpp). Shape, layout and arguments are
    /// AllOrdersF64's, and a call delivers inside the same documented double
    /// batch budget.
    ///
    /// The route is not a re-cut of the pieces the entries above read: every order
    /// is fitted on its own over each interval, so an argument's ladder is a set of
    /// independent polynomials rather than one recurrence seeded once and walked up.
    /// That independence is what the table costs 505.3 KiB and a stored degree per
    /// order to buy.
    ///
    /// Above kFlatHi the table reaches no further and the call falls to the one-term
    /// asymptotic and its own upward recurrence — the arm region C runs, from the
    /// lane's own prefactor and stepping by the same (l + 1/2)/x — so the join is
    /// where this route's fit ends.
    ///
    /// **This entry reads the route's full-accuracy arithmetic.** The table's
    /// one stored degree is the route's own: no coefficient of it is dropped,
    /// and a criterion that would cut one has nothing here to cut. What the
    /// entry costs is what the route costs.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64Uniform(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) off the same uniform table as AllOrdersF64Uniform,
    /// summed in the other form the table stores: every order's coefficients are
    /// read from the monomial image of its own fit and summed by Horner in
    /// ascending order, where the entry above sums the Chebyshev image by a split
    /// Clenshaw. Shape, layout and arguments are
    /// AllOrdersF64Uniform's, and the grid arithmetic, the join at kFlatHi and the
    /// asymptotic arm above it are one body for both forms.
    ///
    /// The two images are one fit stored twice — the two rows of kFlatRows
    /// (boys_coefficients.hpp) — and each form's delivered accuracy is its own
    /// certified row. What the choice buys is counted rather than asserted: one
    /// multiply-add per coefficient against the split Clenshaw's two, at the same
    /// stored table size and degree, so this entry is the cheaper summation where
    /// the two issue alike; the two are ranked by the option probe.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64UniformHorner(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as AllOrdersF64Uniform, with the route's region-A
    /// reading named at the packing axis: every order is read from its own fit
    /// over the argument's interval, and no order is built from another's.
    ///
    /// **This is the route's only reading of the grid, so this entry and
    /// AllOrdersF64Uniform run one kernel and deliver one arithmetic.** A grid that
    /// stores one fit per order and per interval has no seeded ladder to step, and
    /// the route's own fit refuses a band source and a region-B seed at compile time
    /// (UniformFit, boys_impl.hpp) rather than answering from another route's tables.
    /// The row names the axis member and not a second reading, so a chooser reading
    /// this family across the packing axis is told the uniform route holds one
    /// member of it.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersUniform(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// Both of the choices above at once: the uniform route's per-order reading
    /// summed in the other form its table stores, by Horner over the monomial
    /// image of every order's own fit.
    ///
    /// It is the pair \c AllOrdersF64OrdersUniform and \c AllOrdersF64UniformHorner
    /// name, composed, and it runs one kernel: the packing axis has one member
    /// on this route and the scheme axis two, so the four rows of the route are
    /// two readings and four names.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersUniformHorner(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's own narrow partition: the
    /// lane's 218 pieces at degree 6, the same shape, layout and arguments as
    /// AllOrdersF32.
    ///
    /// The body's two lane arguments are the split the coarsest float batch entry
    /// already makes, one partition down: region A's seed is the double lane's piece
    /// table — the downward recursion amplifies a float seed error past the float
    /// budget, which is why the float bodies take a seed lane at all — and the
    /// region-B seed is the float lane's own.
    ///
    /// The double lane's ladders at the other region-B exponential.
    ///
    /// Each member below is the entry of its own name at \c RegionBExp::kFast — the same lane, the
    /// same pieces, the same seed lane and the same recurrence, whose region-B seed is that
    /// member's. The axis is a coordinate of the option and not a second entry: what separates
    /// `AllOrdersF64` from `AllOrdersF64Fast` is the arithmetic the seed is evaluated in, so the
    /// two carry a figure each and nothing substitutes one for the other
    /// (`RegionBExp`, boys/accuracy.hpp).
    ///
    /// The figure is the lane's own, \c 5.5e-14: this lane's fast member is the host's reduced-
    /// argument polynomial ported rather than re-derived, and the host's own documentation of that
    /// member is that its relative error sits inside the ladder's requirement below the member's
    /// cut and that no published figure moves at it. The float lane's `+ 8e-8` is that lane's fast
    /// member's own contribution, and the double lane's member has no such term.
    ///
    /// **One member per kernel, not one per scheme name.** The rational route's pair below is one
    /// member for both scheme names at this exponential, exactly as the accurate pair above is: the
    /// pair is stored in one form, so neither name selects a second arithmetic.
    ///
    /// The parameters, the layout and the precondition of each member are the entry of its own name
    /// above; only the region-B exponential differs.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64Fast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64Orders at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF64OrdersFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64Narrow at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64NarrowOrders at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowOrdersFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64Mono at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF64MonoFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64OrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF64OrdersMonoFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64NarrowMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowMonoFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64NarrowOrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowOrdersMonoFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64Rat at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's, and both of that entry's scheme names reach this one
    /// member for the reason the block states.
    static BoysStatus AllOrdersF64RatFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64OrdersRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF64OrdersRatFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64NarrowRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowRatFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64NarrowOrdersRat at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF64NarrowOrdersRatFast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// **This entry reads this lane's own cut of the partition.** The region-B
    /// degrees are derived over the float lane's pieces (NarrowRegionBDegrees at
    /// the float batch role, boys_effective_degrees.hpp) and read by the kernel.
    /// Region A needs no table beside them: this lane's region-A seed is the
    /// double lane's piece table. So a call reads the stored fit, at the figure
    /// that row states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Narrow(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The same partition with the other summation: every piece of region A and
    /// region B is read from the monomial form of its own fit and summed by Horner,
    /// where \c AllOrdersF32Narrow sums the Chebyshev form by a split Clenshaw.
    /// Shape, layout and arguments are that entry's.
    ///
    /// The two forms are one partition stored twice: the two rows of
    /// kNarrowARowsF32 and kNarrowBRowsF32 (boys_coefficients.hpp). Each form's
    /// delivered accuracy is its own certified row, and each form's degrees are
    /// read from the coefficients the row sums (NarrowRegionBDegrees at
    /// TailBasis::kMonomial), so the Chebyshev form's table is not this entry's
    /// to read.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowMono(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's own uniform table: the same
    /// grid the double route reads — one interval width, one stored degree for every
    /// order and every interval, the table stopping at kFlatHiF32 where the closed
    /// form takes over — at the lane's degree of 4 rather than the double lane's 8.
    /// Shape, layout and arguments are AllOrdersF32's, and a call
    /// delivers inside the float lane's documented budget.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Uniform(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The same grid summed in the other form the table stores: every order's
    /// coefficients are read from the monomial image of its own fit and summed by
    /// Horner, where \c AllOrdersF32Uniform sums the Chebyshev image by a split
    /// Clenshaw. Shape, layout and arguments are that entry's; the
    /// two are ranked by the option probe.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32UniformHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the uniform grid's RATIONAL member.
    ///
    /// The grid is the same one \c AllOrdersF32Uniform reads — the lane's own, one
    /// table of equal intervals over [0, kFlatHiF32) — and the route over it is the
    /// rational family: one numerator/denominator pair per interval, fitted in this
    /// lane's arithmetic and certified against this lane's bar. It is not a second
    /// partition: a caller naming this row and one naming the row above are read at
    /// the same intervals, located by the same map.
    ///
    /// The pair is stored in monomial form and read by Horner, so neither scheme name
    /// a caller may use reaches a second arithmetic and this row's Horner twin runs
    /// the same kernel.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32UniformRat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32UniformRat under the Horner scheme name. A forwarder and not a
    /// second row over a second table: the rational member is stored in one form, so
    /// both scheme names reach one arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns what \c AllOrdersF32UniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF32UniformRatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The same four on the double lane's grid, whose rational member is the
    /// double lane's own pair per interval over that lane's intervals. The
    /// contract and the Horner relation are the float lane's above.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64UniformRat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64UniformRat under the Horner scheme name.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns what \c AllOrdersF64UniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF64UniformRatHorner(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as \c AllOrdersF32UniformRat over the grid's other packing
    /// axis, whose one member this route has: the grid's rational member stores one
    /// numerator/denominator pair per interval at the interval's own stored count, so
    /// there is no per-order reading of the interval to take and this row runs the
    /// per-argument entry's kernel and arithmetic.
    ///
    /// It is not a second fit and not a second partition: a caller naming this row and one
    /// naming \c AllOrdersF32UniformRat is read at the same intervals, located by the same
    /// map, and delivered the same figure.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersUniformRat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersUniformRat under the Horner scheme name. A forwarder and not a
    /// second row over a second table: the rational member is stored in one form, so both
    /// scheme names reach one arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns what \c AllOrdersF32OrdersUniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF32OrdersUniformRatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The same four on the double lane's grid, whose rational member is that lane's own
    /// pair per interval over its own intervals: the contract and the Horner
    /// relation are the float lane's above, and the bound is the double batch lane's.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF64OrdersUniformRat(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF64OrdersUniformRat under the Horner scheme name.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns what \c AllOrdersF64OrdersUniformRat returns, and its refusals with it:
    /// this name is that entry's and adds none of its own.
    static BoysStatus AllOrdersF64OrdersUniformRatHorner(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) in fp32 off the float lane's rational route: region A's
    /// seed comes from the double lane's rational pair over the coarsest partition —
    /// the same lane the float lane's other batch entries take their region-A seed
    /// from, because the downward recursion amplifies a float seed error past the
    /// float budget — and the region-B seed is the float lane's own rational pair
    /// over the same partition (kNarrowRatBCoeffsF32's coarsest counterpart,
    /// kRatBnum and kRatBden).
    ///
    /// Shape, layout and arguments are \c AllOrdersF32's, and the
    /// delivered figure is the float lane's.
    ///
    /// **This entry reads this lane's own rational fit.** The region-B pair is
    /// derived over this lane's own rational fit (RationalRegionBF32Degrees at
    /// the float batch role) and read by the kernel; region A's seed is the
    /// double lane's pair. So a call reads the stored pair, at the figure that
    /// row states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Rat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Rat under the route's other scheme name. The route's pair is
    /// stored in monomial form and read by Horner, so neither name selects a second
    /// arithmetic: this entry runs the one kernel \c AllOrdersF32Rat runs, and the two
    /// rows exist because a caller naming the scheme must reach the combination it
    /// named. Shape, layout and arguments are \c AllOrdersF32Rat's.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32RatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Rat over the float lane's narrow partition: region A's
    /// seed is the double lane's rational pair over the *narrow* pieces and
    /// region B's is the float lane's pair per narrow piece
    /// (kNarrowRatBCoeffsF32), which is the partition
    /// \c AllOrdersF32Narrow reads its Chebyshev fits from. The two lanes'
    /// readings of one piece therefore share its interval and its mapped
    /// argument, as the two routes over a piece do on the host lane.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowRat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowRat under the route's other scheme name, with the
    /// contract of \c AllOrdersF32RatHorner.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowRatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_n(x[i]) as \c AllOrdersF32, with region A read as one fit per
    /// order: every order's own piece is located and its own fit summed, where
    /// the per-argument entry seeds the top order's fit and brings the lower
    /// orders back down a recurrence. Shape, layout and arguments
    /// are \c AllOrdersF32's, and the values agree to the fit's own accuracy.
    ///
    /// The choice covers region A and nothing else: past kX0 these entries run the
    /// certified all-orders body, which is the interval their claim names. The CPU
    /// lane's packing axis is the same choice (BoysPackAxes).
    ///
    /// The rows of this axis mirror the double lane's, one per partition and route
    /// it carries, and each row reads the stored table it sums, the rational
    /// route's included.
    ///
    /// The uniform grid's two orders entries are the route's one reading of the axis:
    /// the grid's cells each carry their own degree and block start, so an order's own
    /// value is the order's own block of its interval and there is no second reading
    /// of the axis to have.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Orders(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Orders on the narrow partition, with the contract of
    /// \c AllOrdersF32Narrow: the partition's pieces one fit per order.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrders(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrders in the monomial basis, with the contract of
    /// \c AllOrdersF32NarrowMono.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersMono(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Orders over the fit route's pairs, with the contract of
    /// \c AllOrdersF32Rat: each order's own piece read at A = 1, where the
    /// per-argument shape seeds at its top order's piece and carries that piece's
    /// w(b) down the recursion. The two scheme names select one arithmetic, as
    /// they do on the route's other shapes.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersRat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRat under the route's other scheme name, over the same
    /// kernel.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersRatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRat on the narrow partition, with the contract of
    /// \c AllOrdersF32NarrowRat.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersRat(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersRat under the route's other scheme name, over the
    /// same kernel.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32NarrowOrdersRatHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Uniform with the orders axis's own rows: the route's one
    /// reading of the grid, and therefore the kernel \c AllOrdersF32Uniform
    /// launches. The grid's cells each carry the degree and block start their own
    /// truncation bound gave them, so an order's value is read from its own block
    /// of the interval's table and there is no second reading of the axis here —
    /// the two rows are two *cells* of the space and one arithmetic, which is what
    /// the grid's own row states.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersUniform(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersUniform over the grid's other stored form, with the
    /// contract of \c AllOrdersF32UniformHorner.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersUniformHorner(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax, double precision — the
    /// uniform-order batch: many arguments, all nmax + 1 orders each, the
    /// layout AllNF32 documents and the bound the CPU lane's BoysAllN
    /// documents, |F_hat - F| <= 5.5e-14 (the double batch lane's
    /// per-region budget, the same in every region).
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count) — the ordering contract
    ///      AllNF32 states, for the reason it states there.
    ///
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) doubles
    /// \param count  number of arguments
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when the launch fails.
    static BoysStatus AllNF64(
        int nmax,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the region-B exponential is the axis this class reads, and a policy naming
    /// the member the class has no kernel for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp64Device, Shape::kAllN, Device::kDevice>>
    static BoysStatus AllNF64WithPolicy(
        int nmax, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_n[i](x[i]) in double precision, each argument's ladder to that
    /// argument's own order — the shape \c BoysDeviceEachOrderF64 answers in the
    /// caller's kernel, launched.
    ///
    /// **Output layout.** `out[offset[i] + l] = F_l(x[i])` for `l = 0..n[i]`, and no
    /// other element of `out` is written: the ladder stops at the argument's own top
    /// order instead of running to a common one. `offset` is the caller's, a device
    /// array of `count` ints, so it is the caller's prefix sum that places the
    /// ladders — at the front of `out`, packed, or interleaved with anything else the
    /// same kernel produces, which is what the shape is for. The padded alternative is
    /// \c AllOrdersF64, whose planes carry every order for every argument: it spends
    /// `count * (kMaxBoysOrder + 1)` values where these spend `sum_i (n[i] + 1)`, and a
    /// batch whose orders differ pays the difference for values no argument asks for.
    /// A caller who wants ladders packed in argument order passes `offset[i] =` the
    /// running total of `n[j] + 1` for `j < i`.
    ///
    /// **Ladders must not overlap.** `offset` is trusted, as `n` and `x` are: two
    /// arguments whose ranges overlap are two arguments writing into one slot. A caller
    /// building `offset` by prefix sum cannot produce that, and one placing ladders by
    /// hand is told here that placing two in one place is undefined.
    ///
    /// The arithmetic is \c AllOrdersF64's, entry for entry: the same lane, the same
    /// pieces, the same seed and the same recurrence, handed one value at a time to a
    /// sink that indexes by `offset[i]` where the padded kernel indexes by a plane.
    /// Nothing here is a second arithmetic and the bound is the lane's.
    ///
    /// \tparam kExp which region-B exponential the call runs, default
    ///   \c kDefaultRegionBExp (\c RegionBExp::kAccurate). Both are certified, each
    ///   against its own bound; nothing substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param offset device array of `count` ints; `offset[i]` is where argument i's
    ///   ladder starts
    /// \param out    device array, at least the largest `offset[i] + n[i] + 1` the
    ///   call reaches
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \pre `offset[i] >= 0` and `offset[i] + n[i] + 1` is inside `out`, for every i
    ///
    /// \returns kDeviceError when the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus EachOrderF64(
        const int* n,
        const double* x,
        const int* offset,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c EachOrderF64 in single precision: the layout, the offsets and the
    /// precondition are that entry's, and the arithmetic is \c AllOrdersF32's.
    ///
    /// \tparam kExp which region-B exponential the call runs, default
    ///   \c kDefaultRegionBExp (\c RegionBExp::kAccurate). Both are certified, each
    ///   against its own bound; nothing substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param offset device array of `count` ints; `offset[i]` is where argument i's
    ///   ladder starts
    /// \param out    device array, at least the largest `offset[i] + n[i] + 1` the
    ///   call reaches
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus EachOrderF32(
        const int* n,
        const double* x,
        const int* offset,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

#if BoysFp16
    /// F_n(x[i]) in fp16 — the fp16 lane of the certified mixed-precision
    /// boundary (behind the BoysFp16 seam). Device pointers and stream
    /// contract as the F32/F64 entries; the fp16 values are the raw
    /// IEEE-754 binary16 bit patterns of this library's F16 type (see
    /// f16.hpp), so host copies are plain byte copies of count * sizeof(F16).
    ///
    /// The lane's bound: |F̂ − F| ≤ 1e-7 + 1/2 ULP of the returned value — the
    /// fp16 rows of the device option space (boys_cuda_options.hpp) carry the
    /// constant part and the half ULP is the format's, and the device accuracy
    /// gate measures these entries against the sum.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in fp16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus SingleF16(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the kernel \c Policy names, chosen while the call site compiles, run at the
    /// policy's division form. The region-B exponential is the axis this class
    /// reads; a policy naming the member the class has no kernel for is a compile
    /// error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp16Device, Shape::kSingle, Device::kDevice>>
    static BoysStatus SingleF16WithPolicy(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in fp16 per input (i), layout as AllOrdersF32
    /// (out[order * count + i] = F_order(x[i])), device pointers and
    /// stream contract as SingleF16.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the route, scheme, partition and packing axis the policy names, dispatched
    /// to this class's entry for that combination while the call site compiles,
    /// run at the policy's division form. A combination this class has no kernel
    /// for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp16Device, Shape::kAllOrders, Device::kDevice>>
    static BoysStatus AllOrdersF16WithPolicy(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax in fp16 — the uniform-order
    /// batch of the fp16 lane, layout as AllNF32, device pointers and stream
    /// contract as SingleF16. Like AllNF32 it has no CPU fp16 counterpart (the
    /// CPU fp16 lane stops at BoysSingleF16 and BoysAllOrdersF16).
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count) — the ordering contract
    ///      AllNF32 states, for the reason it states there.
    ///
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) fp16 values
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when a device operation fails.
    static BoysStatus AllNF16(
        int nmax,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The class above reached by naming a policy (boys/boys_cuda_policy.hpp):
    /// the region-B exponential is the axis this class reads, and a policy naming
    /// the member the class has no kernel for is a compile error naming it.
    ///
    /// \tparam Policy a policy naming a combination this class has a kernel for.
    ///   Defaults to this build's row for the class.
    template <EvalPolicyLike Policy =
                  DefaultPolicy<Precision::kFp16Device, Shape::kAllN, Device::kDevice>>
    static BoysStatus AllNF16WithPolicy(
        int nmax, const F16* x, F16* out, std::size_t count, void* stream);

    /// The fp16 counterparts of the float lane's other bodies: each entry
    /// below computes what the float entry of the same name computes
    /// (AllOrdersF32Narrow and the family beside it), at the same tables, the
    /// same partitions, the same seed lane and the same arithmetic, and stores
    /// what it returns as fp16. The half lane's own definition is that it runs
    /// the float engine's bodies, so an entry of it is not a second arithmetic:
    /// it is that body with `__half` I/O around it.
    ///
    /// Each is named for the float entry it mirrors, with `F16` in place of
    /// `F32`, so the two lanes' surfaces are one table read twice rather than
    /// two tables. The contract each carries is the one its float counterpart's
    /// declaration states, and every one of them takes the device pointers, the
    /// stream and the division form \c AllOrdersF16 above takes. None of them
    /// takes the ordering precondition: that belongs to the uniform-order shapes
    /// (\c AllNF16), and every entry here carries a per-element order array.
    ///
    /// The figure they answer at is \c Precision::kFp16Device's own row of
    /// \c BoysLaneContracts, which is `1e-7` plus half of the last representable
    /// digit of the returned value, which in this format is 2^-11. It is not the
    /// float lane's \c 1.5e-7, and it is not the bf16 class's 1e-7 + 2^-8: the
    /// two stores are two classes of this lane (boys/boys_build_defaults.hpp) and
    /// the store is this format's.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16
    ///   values, order-major as \c AllOrdersF16 writes it
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs, of the three certified
    ///   forms (DivisionForm, boys/accuracy.hpp)
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16Narrow(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowMono's arithmetic and contract, stored fp16: the
    /// narrow partition in its monomial basis, which is the form the Horner
    /// scheme name sums.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowMono(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Rat's arithmetic and contract, stored fp16: the rational
    /// route over the coarsest partition. The route's pair is stored in one form
    /// and read by Horner, so both scheme names reach this one entry.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16Rat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Rat under the route's other scheme name. A forwarder and
    /// not a second arithmetic: it runs \c AllOrdersF16Rat's kernel and returns
    /// its status.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16RatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowRat's arithmetic and contract, stored fp16: the
    /// rational route over the narrow partition, with the same two scheme names
    /// reaching one kernel that \c AllOrdersF16Rat states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowRat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowRatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Uniform's arithmetic and contract, stored fp16: the float
    /// lane's uniform grid in its Chebyshev blocks.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16Uniform(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32UniformHorner's arithmetic and contract, stored fp16. The
    /// grid's two entries are two stored forms of one fit and not two
    /// arithmetics, for the reason the float declaration states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16UniformHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32UniformRat's arithmetic and contract, stored fp16: the
    /// grid's rational route, with the same two scheme names reaching one
    /// kernel that \c AllOrdersF16Rat states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16UniformRat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16UniformRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16UniformRatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Orders' arithmetic and contract, stored fp16: the float
    /// lane's other packing axis over the coarsest partition. It reads region A
    /// one fit per order rather than seeding the top order's fit and bringing
    /// the lower orders down a recurrence, and it is one entry for both scheme
    /// names, as its float counterpart is.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16Orders(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrders' arithmetic and contract, stored fp16: the
    /// orders axis over the narrow partition.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowOrders(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersMono's arithmetic and contract, stored fp16:
    /// the orders axis over the narrow partition in its monomial basis.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowOrdersMono(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRat's arithmetic and contract, stored fp16: the
    /// orders axis on the rational route over the coarsest partition, with the
    /// route's two scheme names reaching one kernel.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersRat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16OrdersRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersRatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersRat's arithmetic and contract, stored fp16:
    /// the orders axis on the rational route over the narrow partition, with
    /// the route's two scheme names reaching one kernel.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowOrdersRat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowOrdersRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16NarrowOrdersRatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersUniform's arithmetic and contract, stored fp16. The
    /// grid's cells carry their own degree and block start, so the route's
    /// packing axis has one member here and this entry runs the kernel
    /// \c AllOrdersF16Uniform launches.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersUniform(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The grid's monomial basis over the orders axis: the kernel
    /// \c AllOrdersF16UniformHorner launches, for the reason
    /// \c AllOrdersF16OrdersUniform states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersUniformHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The grid's rational route over the orders axis: the kernel
    /// \c AllOrdersF16UniformRat launches, for the reason
    /// \c AllOrdersF16OrdersUniform states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersUniformRat(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16OrdersUniformRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16OrdersUniformRatHorner(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The half lane's fast region-B reading of the single shape, beside
    /// \c SingleF16 as \c SingleF32Fast stands beside \c SingleF32.
    ///
    /// The two differing exponentials are the two certified members of \c RegionBExp
    /// (boys_device_tables.hpp), and the choice is one of arithmetic rather than of
    /// kernel: this entry runs the same float engine, storing what it returns into
    /// fp16, with the lane's fast exponential in region B's seed in place of the
    /// library routine's. Its bound is \c SingleF32Fast's constant term plus the half
    /// format's own.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in fp16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus SingleF16Fast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

#endif // BoysFp16

    /// The coarsest partition's ladder in the monomial basis on the float lane, beside
    /// \c AllOrdersF64Mono as this lane's \c AllOrdersF32NarrowMono stands beside
    /// \c AllOrdersF64NarrowMono.
    ///
    /// The pieces are the coarsest partition's own, the fit is the one
    /// \c AllOrdersF32 carries, and what differs is the summation its stored
    /// coefficients are read by: Horner over the monomial form rather than the split
    /// Clenshaw recurrence over the Chebyshev one. The two forms are one fit stored
    /// twice, so the two entries differ in the summation and in nothing else.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Mono(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Mono on the coarsest partition's orders reading, with the
    /// contract of \c AllOrdersF32Orders: each order's own piece located and summed
    /// where it lies, in the monomial basis.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32OrdersMono(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The float lane's ladders at the other region-B exponential, the double lane's
    /// block above read at this lane's kernels: the same lane, the same pieces, the
    /// same seed lane and the same recurrence, whose region-B seed is that member's.
    ///
    /// The axis is a coordinate of the option and not a second entry, so what
    /// separates `AllOrdersF32` from `AllOrdersF32Fast` is the arithmetic the seed is
    /// evaluated in. This lane's fast member is the floated reading of the double
    /// lane's own, whose contribution to the returned value the recurrence caps at
    /// 8e-8: the figure is the lane's bound plus that term, \c 1.5e-7 \c + \c 8e-8,
    /// and nothing substitutes one exponential for the other (`RegionBExp`,
    /// boys/accuracy.hpp).
    ///
    /// **One member per kernel, not one per scheme name.** The rational route's pair
    /// is one member for both scheme names at this exponential, exactly as the
    /// accurate pair above is: the pair is stored in one form, so neither name
    /// selects a second arithmetic.
    ///
    /// The parameters, the layout and the precondition of each member are the entry
    /// of its own name above; only the region-B exponential differs.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF32Fast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Orders at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF32OrdersFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Narrow at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrders at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowOrdersFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Mono at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF32MonoFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32OrdersMonoFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowMonoFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowOrdersMonoFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Rat at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF32RatFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32RatHorner at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF32RatHornerFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF32OrdersRatFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32OrdersRatHornerFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowRatFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowRatHornerFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersRat at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowOrdersRatFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersRatHorner at the region-B exponential the block above states;
    /// its parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF32NarrowOrdersRatHornerFast(
        const int* n,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The single and all-N shapes at the other region-B exponential, one member per
    /// shape for the reason the ladders above have one per kernel: the member is a
    /// compile-time choice of arithmetic, so the choice is a second entry and not a
    /// parameter.
    ///
    /// The parameters, the layout and the precondition of each member are the entry
    /// of its own name; only the region-B exponential differs. The double lane's
    /// member carries the lane's own figure and no added term; the float lane's
    /// carries \c 1.5e-7 \c + \c 8e-8, as the block above states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder (SingleF64Fast); or the
    ///   highest order, 0..kMaxBoysOrder (the all-N members)
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) (SingleF64Fast) or
    ///   out[k * count + i] = F_k(x[i]), k = 0..nmax
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus SingleF64Fast(
        const int* n,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllNF64 at the region-B exponential the block above states; its parameters and its
    /// precondition are that entry's.
    static BoysStatus AllNF64Fast(
        int nmax,
        const double* x,
        double* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllNF32 at the region-B exponential the block above states; its parameters and its
    /// precondition are that entry's.
    static BoysStatus AllNF32Fast(
        int nmax,
        const double* x,
        float* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

#if BoysFp16
    /// The half lane's rows of the two entries above: the same float engine's bodies
    /// with this lane's fp16 store around them, as every half-lane ladder row is.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF16Mono(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Mono on the coarsest partition's orders reading.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersF16OrdersMono(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c EachOrderF64 in fp16 — the each-order shape (see that entry for the
    /// layout, the offsets and the precondition) over this lane's store, behind
    /// the BoysFp16 seam.
    ///
    /// The lane's bound: |F̂ − F| ≤ 1e-7 + 1/2 ULP of the returned value, the figure
    /// the fp16 rows of the device option space carry.
    ///
    /// \tparam kExp which region-B exponential the call runs, default
    ///   \c kDefaultRegionBExp (\c RegionBExp::kAccurate). Both are certified, each
    ///   against its own bound; nothing substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of F16 arguments, >= 0
    /// \param offset device array of `count` ints; `offset[i]` is where argument i's
    ///   ladder starts
    /// \param out    device array of F16, at least the largest `offset[i] + n[i] + 1`
    ///   the call reaches
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus EachOrderF16(
        const int* n,
        const F16* x,
        const int* offset,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The half lane's ladders at the other region-B exponential, its own block above
    /// read the way the float lane's block is: the same kernels, the same seed/lane
    /// pairs, and the second argument of the pair moved.
    ///
    /// The lane's definition is that it runs the float engine's arithmetic and stores
    /// what it returns, so the member here is the float lane's fast reading and not a
    /// second half arithmetic. Its figure is \c SingleF16Fast's: the float lane's
    /// fast term plus the half format's own, and nothing substitutes one exponential
    /// for the other (`RegionBExp`, boys/accuracy.hpp).
    ///
    /// **One member per kernel, not one per scheme name**, as the double lane's block
    /// states it: a rational pair is stored in one form, so the two scheme names
    /// reach the one member.
    ///
    /// The parameters, the layout and the precondition of each member are the entry
    /// of its own name above; only the region-B exponential differs.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder (AllNF16Fast: the
    ///   highest order, 0..kMaxBoysOrder)
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersF16Fast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Orders at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF16OrdersFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Narrow at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowOrders at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowOrdersFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Mono at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF16MonoFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16OrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16OrdersMonoFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowMonoFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowOrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowOrdersMonoFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16Rat at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersF16RatFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16RatHorner at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF16RatHornerFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16OrdersRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF16OrdersRatFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16OrdersRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16OrdersRatHornerFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowRatFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowRatHornerFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowOrdersRat at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowOrdersRatFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF16NarrowOrdersRatHorner at the region-B exponential the block above states;
    /// its parameters and its precondition are that entry's.
    static BoysStatus AllOrdersF16NarrowOrdersRatHornerFast(
        const int* n,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllNF16 at the region-B exponential the block above states; its parameters and its
    /// precondition are that entry's.
    static BoysStatus AllNF16Fast(
        int nmax,
        const F16* x,
        F16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);



    /// F_n(x[i]) in bfloat16 — the bfloat16 lane of the certified mixed-precision
    /// boundary (behind the BoysFp16 seam). Device pointers and stream
    /// contract as the F32/F64 entries; the bfloat16 values are the raw
    /// IEEE-754 binary16 bit patterns of this library's Bf16 type (see
    /// f16.hpp), so host copies are plain byte copies of count * sizeof(Bf16).
    ///
    /// The lane's bound: |F̂ − F| ≤ 1e-7 + 1/2 ULP of the returned value — the
    /// bfloat16 rows of the device option space (boys_cuda_options.hpp) carry the
    /// constant part and the half ULP is the format's, and the device accuracy
    /// gate measures these entries against the sum.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in bfloat16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus SingleBf16(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_nmax(x[i]) in bfloat16 per input (i), layout as AllOrdersF32
    /// (out[order * count + i] = F_order(x[i])), device pointers and
    /// stream contract as SingleBf16.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The bfloat16 counterparts of the float lane's other bodies: each entry
    /// below computes what the float entry of the same name computes
    /// (AllOrdersF32Narrow and the family beside it), at the same tables, the
    /// same partitions, the same seed lane and the same arithmetic, and stores
    /// what it returns as bfloat16. The bfloat16 lane's own definition is that it runs
    /// the float engine's bodies, so an entry of it is not a second arithmetic:
    /// it is that body with `__half` I/O around it.
    ///
    /// Each is named for the float entry it mirrors, with `Bf16` in place of
    /// `F32`, so the two lanes' surfaces are one table read twice rather than
    /// two tables. The contract each carries is the one its float counterpart's
    /// declaration states, and every one of them takes the device pointers, the
    /// stream and the division form \c AllOrdersBf16 above takes. None of them
    /// takes the ordering precondition: that belongs to the uniform-order shapes
    /// (\c AllNBf16), and every entry here carries a per-element order array.
    ///
    /// The figure they answer at is \c Precision::kBf16Device's own row of
    /// \c BoysLaneContracts, which is `1e-7` plus half of the last representable
    /// digit of the returned value, which in this format is 2^-8 = 3.90625e-03.
    /// It is not the float lane's \c 1.5e-7, and it is not the fp16 class's
    /// 1e-7 + 2^-11 either: the two stores are two classes of this lane
    /// (boys/boys_build_defaults.hpp) and the store is this format's.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16
    ///   values, order-major as \c AllOrdersBf16 writes it
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs, of the three certified
    ///   forms (DivisionForm, boys/accuracy.hpp)
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16Narrow(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowMono's arithmetic and contract, stored bfloat16: the
    /// narrow partition in its monomial basis, which is the form the Horner
    /// scheme name sums.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowMono(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Uniform's arithmetic and contract, stored bfloat16: the float
    /// lane's uniform grid in its Chebyshev blocks.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16Uniform(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32UniformHorner's arithmetic and contract, stored bfloat16. The
    /// grid's two entries are two stored forms of one fit and not two
    /// arithmetics, for the reason the float declaration states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16UniformHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Rat's arithmetic and contract, stored bfloat16: the rational
    /// route over the coarsest partition. The route's pair is stored in one form
    /// and read by Horner, so both scheme names reach this one entry.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16Rat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Rat under the route's other scheme name. A forwarder and
    /// not a second arithmetic: it runs \c AllOrdersBf16Rat's kernel and returns
    /// its status.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16RatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowRat's arithmetic and contract, stored bfloat16: the
    /// rational route over the narrow partition, with the same two scheme names
    /// reaching one kernel that \c AllOrdersBf16Rat states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowRat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowRatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32UniformRat's arithmetic and contract, stored bfloat16: the
    /// grid's rational route, with the same two scheme names reaching one
    /// kernel that \c AllOrdersBf16Rat states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16UniformRat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16UniformRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16UniformRatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32Orders' arithmetic and contract, stored bfloat16: the float
    /// lane's other packing axis over the coarsest partition. It reads region A
    /// one fit per order rather than seeding the top order's fit and bringing
    /// the lower orders down a recurrence, and it is one entry for both scheme
    /// names, as its float counterpart is.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16Orders(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrders' arithmetic and contract, stored bfloat16: the
    /// orders axis over the narrow partition.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowOrders(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersMono's arithmetic and contract, stored bfloat16:
    /// the orders axis over the narrow partition in its monomial basis.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowOrdersMono(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersRat's arithmetic and contract, stored bfloat16: the
    /// orders axis on the rational route over the coarsest partition, with the
    /// route's two scheme names reaching one kernel.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersRat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16OrdersRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersRatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32NarrowOrdersRat's arithmetic and contract, stored bfloat16:
    /// the orders axis on the rational route over the narrow partition, with
    /// the route's two scheme names reaching one kernel.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowOrdersRat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowOrdersRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16NarrowOrdersRatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersF32OrdersUniform's arithmetic and contract, stored bfloat16. The
    /// grid's cells carry their own degree and block start, so the route's
    /// packing axis has one member here and this entry runs the kernel
    /// \c AllOrdersBf16Uniform launches.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersUniform(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The grid's monomial basis over the orders axis: the kernel
    /// \c AllOrdersBf16UniformHorner launches, for the reason
    /// \c AllOrdersBf16OrdersUniform states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersUniformHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The grid's rational route over the orders axis: the kernel
    /// \c AllOrdersBf16UniformRat launches, for the reason
    /// \c AllOrdersBf16OrdersUniform states.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersUniformRat(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16OrdersUniformRat under the route's other scheme name. A
    /// forwarder and not a second arithmetic.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16OrdersUniformRatHorner(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// F_0(x[i])..F_nmax(x[i]) at one common nmax in bfloat16 — the uniform-order
    /// batch of the bfloat16 lane, layout as AllNF32, device pointers and stream
    /// contract as SingleBf16. Like AllNF32 it has no CPU bfloat16 counterpart (the
    /// CPU bfloat16 lane stops at BoysSingleBf16 and BoysAllOrdersBf16).
    ///
    /// \pre x[i - 1] <= x[i] for every i in [1, count) — the ordering contract
    ///      AllNF32 states, for the reason it states there.
    ///
    /// \param nmax   highest order, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, non-decreasing, each >= 0
    /// \param out    device array, at least count * (nmax + 1) bfloat16 values
    /// \param count  number of arguments; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kInvalidArgument when nmax is outside [0, kMaxBoysOrder],
    /// kDeviceError when a device operation fails.
    static BoysStatus AllNBf16(
        int nmax,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The bfloat16 lane's rows of the two entries above: the same float engine's bodies
    /// with this lane's bfloat16 store around them, as every half-lane ladder row is.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersBf16Mono(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Mono on the coarsest partition's orders reading.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the table upload or the launch fails.
    static BoysStatus AllOrdersBf16OrdersMono(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The bfloat16 lane's fast region-B reading of the single shape, beside
    /// \c SingleBf16 as \c SingleF32Fast stands beside \c SingleF32.
    ///
    /// The two differing exponentials are the two certified members of \c RegionBExp
    /// (boys_device_tables.hpp), and the choice is one of arithmetic rather than of
    /// kernel: this entry runs the same float engine, storing what it returns into
    /// bfloat16, with the lane's fast exponential in region B's seed in place of the
    /// library routine's. Its bound is \c SingleF32Fast's constant term plus the half
    /// format's own.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in bfloat16
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus SingleBf16Fast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c EachOrderF64 in bfloat16 — the each-order shape (see that entry for the
    /// layout, the offsets and the precondition) over this lane's store, behind
    /// the BoysFp16 seam.
    ///
    /// The lane's bound: |F̂ − F| ≤ 1e-7 + 1/2 ULP of the returned value, the figure
    /// the bfloat16 rows of the device option space carry.
    ///
    /// \tparam kExp which region-B exponential the call runs, default
    ///   \c kDefaultRegionBExp (\c RegionBExp::kAccurate). Both are certified, each
    ///   against its own bound; nothing substitutes one for the other.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of Bf16 arguments, >= 0
    /// \param offset device array of `count` ints; `offset[i]` is where argument i's
    ///   ladder starts
    /// \param out    device array of Bf16, at least the largest `offset[i] + n[i] + 1`
    ///   the call reaches
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when the launch fails.
    template <RegionBExp kExp = kDefaultRegionBExp>
    static BoysStatus EachOrderBf16(
        const int* n,
        const Bf16* x,
        const int* offset,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// The bfloat16 lane's ladders at the other region-B exponential, its own block above
    /// read the way the float lane's block is: the same kernels, the same seed/lane
    /// pairs, and the second argument of the pair moved.
    ///
    /// The lane's definition is that it runs the float engine's arithmetic and stores
    /// what it returns, so the member here is the float lane's fast reading and not a
    /// second half arithmetic. Its figure is \c SingleBf16Fast's: the float lane's
    /// fast term plus the half format's own, and nothing substitutes one exponential
    /// for the other (`RegionBExp`, boys/accuracy.hpp).
    ///
    /// **One member per kernel, not one per scheme name**, as the double lane's block
    /// states it: a rational pair is stored in one form, so the two scheme names
    /// reach the one member.
    ///
    /// The parameters, the layout and the precondition of each member are the entry
    /// of its own name above; only the region-B exponential differs.
    ///
    /// \param n      device array of orders, 0..kMaxBoysOrder (AllNBf16Fast: the
    ///   highest order, 0..kMaxBoysOrder)
    /// \param x      device array of bfloat16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) bfloat16 values
    /// \param count  number of elements; 0 is the no-op the class documents
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    /// \param form   which division form the call runs: the entry's arithmetic with a different
    ///   division of its ladder steps, one of the three certified forms (DivisionForm,
    ///   boys/accuracy.hpp); kDefaultDeviceDivisionForm unless another is named.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus AllOrdersBf16Fast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Orders at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersBf16OrdersFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Narrow at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowOrders at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowOrdersFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Mono at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersBf16MonoFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16OrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16OrdersMonoFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowMonoFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowOrdersMono at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowOrdersMonoFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16Rat at the region-B exponential the block above states; its parameters and
    /// its precondition are that entry's.
    static BoysStatus AllOrdersBf16RatFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16RatHorner at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersBf16RatHornerFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16OrdersRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersBf16OrdersRatFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16OrdersRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16OrdersRatHornerFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowRat at the region-B exponential the block above states; its parameters
    /// and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowRatFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowRatHorner at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowRatHornerFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowOrdersRat at the region-B exponential the block above states; its
    /// parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowOrdersRatFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllOrdersBf16NarrowOrdersRatHorner at the region-B exponential the block above states;
    /// its parameters and its precondition are that entry's.
    static BoysStatus AllOrdersBf16NarrowOrdersRatHornerFast(
        const int* n,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

    /// \c AllNBf16 at the region-B exponential the block above states; its parameters and its
    /// precondition are that entry's.
    static BoysStatus AllNBf16Fast(
        int nmax,
        const Bf16* x,
        Bf16* out,
        std::size_t count,
        void* stream,
        DivisionForm form = kDefaultDeviceDivisionForm);

#endif // BoysFp16
};

/// The option space's rows counted by the arithmetic each one selects, as a report states
/// it: how many names the space offers against how many distinct arithmetic they reach, and
/// the names that are a second one of a row beside them.
///
/// Declared here rather than in boys_cuda_options.hpp, which is the space's own header,
/// because it returns a string and that header is compiled by nvcc. The counts are the
/// space's and not this function's: the rows come from \c BoysDeviceOptions and the division
/// forms from \c BoysDivisionForms, so a row added to the space is counted here without an
/// edit.
///
/// \returns the statement, one line per alias row under the counts, ending with a newline
///
/// \ingroup boys
std::string FormatDeviceArithmeticStatement();

} // namespace boys

// The policy-templated members declared above - each class's entry under the name
// <Entry>WithPolicy - their definitions, and the rule
// they dispatch by. Last, because a definition of a class member needs the class
// complete, and here so that a translation unit that includes this header gets
// the layer with the surface it belongs to. The other order works too - the
// policy header includes this one at its top for a caller who names it first.
#include "boys/boys_cuda_policy.hpp"
