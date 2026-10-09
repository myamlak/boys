#pragma once

// The policy-templated entry layer of the device surface (boys/boys_cuda.hpp), included at that
// header's end because every definition here names a declaration of the class above, and includes
// it back at its top so a translation unit that includes this one alone gets the class - the same
// continuation boys_impl.hpp is to boys.hpp, and either order a caller writes compiles.

#include "boys/boys_cuda.hpp"

#include <cstddef>

/// \file
/// The device entries reached by naming an evaluation policy.
///
/// The host surface lets a caller choose a combination by *naming a policy*:
/// `template <EvalPolicyLike Policy = DefaultPolicy<Precision, Shape>>` is on every
/// host entry, so a caller writes `BoysAllOrders<Policy>(...)`, and a call site that
/// names no policy is answered by the build's own row for the class
/// (`boys/boys.hpp`, `DefaultPolicyFor`). The device surface carried no such layer:
/// its combinations are entry *names* - `BoysCuda::AllOrdersF64NarrowMono` - so a
/// caller reaches a combination only by knowing its name, and a class has no default
/// at all.
///
/// This header is that layer. Every class of the device surface carries a
/// policy-templated member beside the entries it already has, named after the class
/// with the two words `WithPolicy` after it:
///
///     BoysCuda::AllOrdersF64WithPolicy<>            (n, x, out, count, stream)   - the seam's row
///     BoysCuda::AllOrdersF64WithPolicy<Policy>(...) - the kernel the policy names
///     BoysCuda::AllOrdersF64(...)                   - the shipped entry, unchanged
///
/// and the same for every other class of the surface: the single and all-N classes of
/// the f64, f32, fp16 and bf16 lanes, and the all-orders class of each of the four.
/// **The suffix is load-bearing and not a style.** A second function of an
/// entry's own name would make that name an overload set, and the address of an
/// overload set cannot be taken where the pointer type is deduced - a use the shipped
/// tests make, handing `&BoysCuda::AllOrdersF64` to a helper that deduces its launch
/// type (`tests/boys_cuda_test.cpp`, `CrossEntryWithForms`). With the suffix every
/// entry stays the one function of its name, so a call and an address taken by name
/// both reach what they reached before. **The layer is an addition and not a
/// replacement**: the named entries keep their declarations, their parameter lists
/// and their arithmetic, and the calls that reach them today compile and deliver
/// exactly what they delivered before.
///
/// ## A policy reaches the kernel that implements it, and nothing else
///
/// The layer is a **dispatch from a policy to the kernel that exists**, written as
/// the host probe's own cascade is (`src/boys_probe.cpp`, `CellPolicy` ->
/// `CellExpPolicy` -> ... -> `CellEntry`): one arm per member of each axis the class
/// carries, each arm naming the entry that member reaches, and no default arm. A
/// combination the surface **has no kernel for is a compile error**, and the message
/// names the combination and states that no entry of the class answers it. Nothing
/// here falls back to a nearby kernel: a policy naming a reading the lane does not
/// carry is never answered out of another reading's tables, which is the substitution
/// the whole seam machinery of this library exists to prevent.
///
/// ## Which cells a class reads
///
/// A class reads the axes **its own entries vary over**, and the arm of each is the
/// member the entry's own statement of its arithmetic names
/// (`DeviceEntryAxesOf`/`DevicePartitionOf`, `boys/boys_cuda_options.hpp`). The
/// four all-orders classes read the route, the scheme, the partition and the
/// packing axis, because their entries are one entry per member of each axis, bar
/// the two members a single entry answers (the paragraph below); every
/// class reads the region-B exponential, and an arm answers a member with the entry
/// the class's own row at it carries. The one member no class carries an entry at is
/// the uniform grid's `kFast`, which the table books for no row of any lane, and the
/// refusal there states that reason. The cells of
/// the axes a class does not carry are not coordinates of it and are not read: the
/// single-order and all-N shapes hold one reading of region A per element and no
/// ladder to cut (`DevicePacking::kNotApplicable`, stated of their rows), the route
/// and the scheme of those shapes are the lane's own arithmetic and not a choice
/// (`DeviceEntryAxesOf`, "the route and the scheme are the lane's own"), and the
/// build's budget is not an axis of this surface at all - its kernels read stored
/// tables rather than a compiled multiplier. The seam row for such a class names
/// cells those classes do not read, and the row's own arithmetic is read where the
/// row is: the layer reads the row's cells for the axes the class carries, which is
/// what makes `BoysCuda::AllNF64WithPolicy<>` reach a real kernel.
///
/// **Where the table books one arithmetic under two rows, the arm answers with that
/// arithmetic and does not refuse the second row.** Two such cells are stated, and
/// neither is a substitution: one kernel runs, and it is the kernel both rows name -
/// `DeviceEntryArithmeticOf` is the library's own statement of which rows those are.
/// The rational route's pairs are one of them: `kAllOrdersF64Rat` and
/// `kAllOrdersF64RatHorner` are two rows naming one entry, because "the pair is stored
/// once, so both scheme names select this arithmetic and the scheme axis is inert
/// here" (`AllOrdersF64Rat`), and the float, fp16 and bf16 lanes book the same pair
/// with the route's other scheme name as a forwarder. The uniform grid's cells are the
/// other: the grid holds one member of the packing axis, so its per-order row and its
/// per-argument row are one kernel under two names, and the arm names the per-order
/// one. Refusal is kept for the members no row of the class answers, and there are
/// two of them on every all-orders class: the grid's ladder cell, which no row books,
/// and its `kFast`, which no row of any lane books.
///
/// **Where the table books two rows of one cell that are two arithmetics, the arm
/// answers on the row the member's own name carries.** The monomial pair of the
/// coarsest cut is the case on every all-orders class, and the float lane is the one
/// this layer states: `kAllOrdersF32Mono` and `kAllOrdersF32MonoFast` are rows of
/// that cut beside `kAllOrdersF32` and `kAllOrdersF32Fast`, each booked at
/// `EvalScheme::kHorner`, each launched, and each an arithmetic of its own beside the
/// split Clenshaw row of the same fit (`AllOrdersF32Mono`, boys_cuda.hpp) - so a
/// policy naming `kHorner` over that cut reaches the monomial row rather than the
/// reading beside it, which is the substitution this layer exists to prevent. The
/// double, half and bfloat16 lanes book the pair under their own names.
///
/// ## The packing axis is the host's axis, in this surface's spelling
///
/// `PackAxis::kArguments` is `DevicePacking::kLadder` - the top order's piece is
/// seeded and the lower orders are brought back down the recurrence - and
/// `PackAxis::kOrders` is `DevicePacking::kPerOrder` - every order's own piece is
/// located and its own fit summed. The layer's arms are the crossing of the two
/// names. The uniform partition holds **one member of the packing axis** ("The
/// row names the axis member and not a second reading, so a chooser reading this
/// family across the packing axis is told the uniform route holds one member of
/// it", `boys_cuda.hpp` on `AllOrdersF64OrdersUniform`), so a policy naming the
/// ladder member over the grid is refused by name rather than answered by the
/// grid's per-order kernel.
///
/// ## The division form is a coordinate of the call
///
/// Every entry of this surface runs every certified form: the form is a
/// compile-time argument of the kernel a launch builds, and a caller's form is which
/// instantiation the launch selects. It is therefore **never** the reason a
/// combination is missing, and the layer passes the policy's own form to the entry it
/// reaches, which validates it as it validates every form it is handed. A policy
/// naming a value outside `DivisionForm`'s enumerators is refused at run time by the
/// entry's own check - one statement of that rule, in the entry, and not a second
/// one here.
///
/// ## The default of a class is the seam's row
///
/// `Policy` defaults to `DefaultPolicy<Precision::kFp64Device|kFp32Device|kFp16Device|kBf16Device,
/// Shape::kSingle|kAllOrders|kAllN, Device::kDevice>`: the device half of the
/// default-policy table (`include/boys/boys_build_defaults.hpp`), whose cells are the
/// build's own names - of which `kDefaultDeviceDivisionForm` and
/// `kDefaultDeviceRegionBExp` are this lane's two (`boys/accuracy.hpp`). A device call
/// naming no policy therefore reaches the row the build wrote for that class, and a
/// class whose row names a combination the surface has no kernel for fails to compile
/// where the call is written rather than being answered by a nearby kernel.
///
/// Every class's policy is defaulted - the f32 single class's included, whose entry
/// is itself a template (`SingleF32<RegionBExp>`) and whose layer member is a
/// different name and therefore no ambiguity - so no class of this layer is an
/// exception to the rule above.
///
/// **A call naming no policy, and the same call naming the entry, are two
/// spellings that never compete.** `BoysCuda::AllOrdersF64WithPolicy<>(n, x, out,
/// count, stream)` reaches this layer, hence the build's row for the class, and
/// `BoysCuda::AllOrdersF64(n, x, out, count, stream)` reaches the shipped entry:
/// the entry is not a template, the layer's member is not an overload of it, and
/// neither call is a reading of the other. The two name different kernels where the
/// row's combination is not the entry's, and the row is the one that moves with the
/// build.
///
/// ## What this costs at run time
///
/// Nothing. Every choice in the dispatch is a value of the policy *type*, so the
/// cascade resolves while the call site is compiled: a call compiles to one call of
/// one entry, with no lookup, no registry and no branch on anything a caller chose.
///
/// \ingroup boys

namespace boys {
/// \cond
namespace detail {

/// The fit route an assertion is instantiated on, so that the refusal is a
/// dependent one: the arm a route switch keeps for the enumerators it does not
/// name fires only where a policy reaches it, and a fifth member of `FitRoute`
/// fails the build at every such arm rather than being answered out of the last
/// one. The same reading `GranularityTag` carries for the partitions
/// (`boys/boys_impl.hpp`).
template <FitRoute kRoute>
struct RouteTag {};

/// The scheme an assertion is instantiated on, on the reading `RouteTag` states.
template <EvalScheme kScheme>
struct SchemeTag {};

/// The partition an assertion is instantiated on, on the reading `RouteTag`
/// states. It is this header's own and not `boys_impl.hpp`'s `GranularityTag`:
/// that one is the host kernel's, and a device refusal that named it would put a
/// host tag in the device surface's diagnostics.
template <FitGranularity kPartition>
struct PartitionTag {};

/// The packing axis an assertion is instantiated on, on the reading `RouteTag`
/// states. The tag carries the *device* axis and not the policy's cell, because a
/// member of `DevicePacking` the surface does not read is not a member a caller
/// can spell.
template <DevicePacking kPacking>
struct PackingTag {};

/// The region-B exponential an assertion is instantiated on, on the reading
/// `RouteTag` states.
template <RegionBExp kExp>
struct ExpTag {};

/// The member of a pair of entries the policy's region-B exponential selects: the
/// member the policy names, where the class carries both, and the class's own away
/// from the pair.
///
/// A *pair* is two entries of one class that differ in nothing but the member of
/// \c RegionBExp their region-B seed is evaluated in — the same tables, the same
/// lane, the same pieces, the same recurrence — so a policy naming either member has
/// a kernel and the two are the certified readings of one arithmetic rather than one
/// reading and one fallback. The two callables are passed rather than a pair of
/// entry pointers so that each arm of a cascade writes the two entries it chooses
/// between where the arm is: a reader of an arm sees which pair it dispatches over,
/// and a member that gains a counterpart later is added to its own arm and not to a
/// table beside the cascade.
///
/// The choice is a value of the policy type, so nothing here survives compilation.
template <RegionBExp kExp, typename Accurate, typename Fast>
BoysStatus DevicePickExp(Accurate accurate, Fast fast) {
    if constexpr (kExp == RegionBExp::kFast)
    {
        return fast();
    }
    else
    {
        return accurate();
    }
}

/// The double all-orders class's cascade over the route, the partition, the packing axis and the
/// scheme, at whichever member of the region-B exponential the policy names. Declared here and
/// defined with that class's own layer below, because its arms name the class's members.
///
/// \param n the class's own first argument, as its entries document it
/// \param x the class's own second argument, as its entries document it
/// \param out the class's own third argument, as its entries document it
/// \param count the class's own fourth argument, as its entries document it
/// \param stream the class's own fifth argument, as its entries document it
///
/// \returns the status of the entry the policy's combination reaches
template <EvalPolicyLike Policy>
BoysStatus AllOrdersF64Cascade(
    const int* n, const double* x, double* out, std::size_t count, void* stream);

/// The float, half and bfloat16 all-orders classes' cascades, declared here with the double
/// one and defined with each class's own layer below, for the same reason.
template <EvalPolicyLike Policy>
BoysStatus AllOrdersF32Cascade(
    const int* n, const double* x, float* out, std::size_t count, void* stream);

#if BoysFp16
template <EvalPolicyLike Policy>
BoysStatus AllOrdersF16Cascade(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream);

template <EvalPolicyLike Policy>
BoysStatus AllOrdersBf16Cascade(
    const int* n, const Bf16* x, Bf16* out, std::size_t count, void* stream);
#endif

} // namespace detail
/// \endcond

// The single-order classes: F_n(x[i]) for one order per element. With no ladder to cut, their
// entries differ only in the region-B exponential a lane carries, and every lane here carries a
// kernel at both members of it - the tables' own seed and the lane's fast exponential - each with
// its own bound.

/// The f64 single class reached by naming a policy: the entry, its bound and its
/// parameter contract are `BoysCuda::SingleF64` as boys_cuda.hpp documents it, and
/// this overload is that entry reached from the policy's region-B exponential and
/// division form. The class carries a kernel at both members of the axis
/// (`kSingleF64` and `kSingleF64Fast`, boys_cuda_options.hpp), so a policy naming
/// either reaches that member's entry and names an arithmetic of its own.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::SingleF64WithPolicy(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        // BoysCuda:: names the class member set, so this reaches the shipped entry
        // declared above (six arguments) and not this template (five).
        return BoysCuda::SingleF64(n, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        return BoysCuda::SingleF64Fast(n, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

/// The f32 single class reached by naming a policy: the one class of this surface
/// whose entries differ by a region-B exponential, both members certified, each
/// against its own bound (boys_cuda.hpp, `SingleF32`). The entry a policy reaches is
/// that member's, run at the policy's division form.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::SingleF32WithPolicy(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        // The explicit template argument is a *value*, which removes this
        // template from the candidate set (its parameter is a type) and leaves the
        // shipped entry's own `<RegionBExp kExp>` declaration to answer.
        return BoysCuda::SingleF32<RegionBExp::kFast>(n, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        return BoysCuda::SingleF32<RegionBExp::kAccurate>(
            n, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials: a third value added "
                      "to RegionBExp must be given its own arm here rather than inheriting the "
                      "last one's kernel");
        return BoysStatus::kDeviceError;
    }
}

#if BoysFp16
/// The fp16 single class reached by naming a policy: the entry, its bound and its
/// parameter contract are `BoysCuda::SingleF16` as boys_cuda.hpp documents it, and
/// this overload is that entry reached from the policy's region-B exponential and
/// division form. The class carries a kernel at both members of the axis
/// (`kSingleF16` and `kSingleF16Fast`, boys_cuda_options.hpp), as the float lane's
/// single class does.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::SingleF16WithPolicy(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        return BoysCuda::SingleF16(n, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        return BoysCuda::SingleF16Fast(n, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}
#endif // BoysFp16

// The all-N classes: F_0(x[i])..F_nmax(x[i]) at one common top order. One entry per precision and
// per member of the region-B exponential, as the single-order classes have: the class reads that
// axis and reaches each member's own entry, and the other cells are not coordinates of the shape.

/// The f64 all-N class reached by naming a policy: the entry, its bound and its
/// parameter contract are `BoysCuda::AllNF64` as boys_cuda.hpp documents it, and this
/// overload is that entry reached from the policy's region-B exponential and division
/// form. The class carries a kernel at both members of the axis (`kAllNF64` and
/// `kAllNF64Fast`, boys_cuda_options.hpp), so a policy naming either reaches that
/// member's entry.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllNF64WithPolicy(
    int nmax, const double* x, double* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        return BoysCuda::AllNF64(nmax, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        return BoysCuda::AllNF64Fast(nmax, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

/// The f32 all-N class reached by naming a policy: the entry, its bound and its
/// parameter contract are `BoysCuda::AllNF32` as boys_cuda.hpp documents it, and this
/// overload is that entry reached from the policy's region-B exponential and division
/// form. The class carries a kernel at both members of the axis (`kAllNF32` and
/// `kAllNF32Fast`, boys_cuda_options.hpp), so a policy naming either reaches that
/// member's entry.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllNF32WithPolicy(
    int nmax, const double* x, float* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        return BoysCuda::AllNF32(nmax, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        return BoysCuda::AllNF32Fast(nmax, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

#if BoysFp16
/// The fp16 all-N class reached by naming a policy: the entry, its bound and its
/// parameter contract are `BoysCuda::AllNF16` as boys_cuda.hpp documents it, and this
/// overload is that entry reached from the policy's region-B exponential and division
/// form. The class carries a kernel at both members of the axis (`kAllNF16` and
/// `kAllNF16Fast`, boys_cuda_options.hpp), as the float lane's all-N class does.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllNF16WithPolicy(int nmax, const F16* x, F16* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate)
    {
        return BoysCuda::AllNF16(nmax, x, out, count, stream, Policy::kDivision);
    }
    else if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
    {
        return BoysCuda::AllNF16Fast(nmax, x, out, count, stream, Policy::kDivision);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}
#endif // BoysFp16

// The all-orders classes: one kernel per member of the route, scheme, partition and packing axes.
// Each dispatch below is one cascade in the host probe's shape, nested route, partition, packing
// axis, then scheme - the order the entries are grouped in (`boys_cuda_options.hpp`,
// `DevicePartitionOf`, `DeviceEntryAxesOf`), so an arm reads as the table it dispatches from does.

template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllOrdersF64WithPolicy(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate
                  || Policy::kRegionBExp == RegionBExp::kFast)
    {
        return detail::AllOrdersF64Cascade<Policy>(n, x, out, count, stream);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

/// \cond
namespace detail {

template <EvalPolicyLike Policy>
BoysStatus AllOrdersF64Cascade(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRoute == FitRoute::kChebyshev)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64Fast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64Orders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64OrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64Mono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64MonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64OrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64OrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64Narrow(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowOrders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowOrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowOrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowOrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                // The grid is the one partition of this class at which the axis has no member,
                // and the sentence is the library's own - the same one the option space cites
                // for the cells it counts as impossible.
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry; the grid's own exponential is not a choice of "
                              "arithmetic and neither member is read by it");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    // The grid's two names run one kernel (AllOrdersF64OrdersUniform
                    // states it of the route's own fit): the dispatch reaches the one
                    // that names the axis member this call named.
                    return BoysCuda::AllOrdersF64OrdersUniform(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF64OrdersUniformHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one fit per order and per interval, each read from its own block, "
                              "with no seeded ladder to step (BoysCuda::AllOrdersF64OrdersUniform) "
                              "- so a policy naming PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64Rat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64RatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64OrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64OrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // The double lane's rational pair is stored once, so both scheme names select this
                // arithmetic and the scheme axis is inert on this route - the route is a choice of
                // fit, not of scheme (boys_cuda.hpp, `AllOrdersF64Rat`). The class has no -Horner
                // name of this lane's rational entries: the row states the entry it is reached by.
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64Rat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64RatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64OrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64OrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowOrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowOrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // Inert on this route, as the coarsest arm states: the pair is stored
                // once and the two scheme names select it.
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF64NarrowOrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF64NarrowOrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersF64OrdersUniformRat(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF64OrdersUniformRatHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one numerator/denominator pair per interval, read per order, with "
                              "no ladder for a second reading to be (the library states it of the "
                              "grid's rows, boys_cuda_options.hpp) - so a policy naming "
                              "PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::RouteTag<Policy::kRoute>>,
                      "this switch enumerates the two fit routes, FitRoute::kChebyshev and "
                      "FitRoute::kRationalMinimax: a third member must be given its own arm here "
                      "rather than inheriting the last one's entries");
        return BoysStatus::kDeviceError;
    }
}

} // namespace detail
/// \endcond

template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllOrdersF32WithPolicy(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate
                  || Policy::kRegionBExp == RegionBExp::kFast)
    {
        return detail::AllOrdersF32Cascade<Policy>(n, x, out, count, stream);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

/// \cond
namespace detail {

template <EvalPolicyLike Policy>
BoysStatus AllOrdersF32Cascade(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRoute == FitRoute::kChebyshev)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32Fast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32Orders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32OrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // This lane books the cut's monomial rows itself (kAllOrdersF32Mono,
                // kAllOrdersF32OrdersMono, boys_cuda_options.hpp), so the Horner member reaches
                // them: the split Clenshaw row is a second arithmetic of the same fit and not a
                // name of this one (AllOrdersF32Mono, boys_cuda.hpp).
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32Mono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32MonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32OrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32OrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32Narrow(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowOrders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowOrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowOrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowOrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersF32OrdersUniform(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF32OrdersUniformHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one fit per order and per interval, each read from its own block, "
                              "with no seeded ladder to step (the library states it of the grid's "
                              "rows, boys_cuda_options.hpp) - so a policy naming "
                              "PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32Rat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32RatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32OrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32OrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32RatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32RatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32OrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32OrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowOrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowOrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF32NarrowOrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF32NarrowOrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersF32OrdersUniformRat(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF32OrdersUniformRatHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one numerator/denominator pair per interval, read per order, with "
                              "no ladder for a second reading to be (the library states it of the "
                              "grid's rows, boys_cuda_options.hpp) - so a policy naming "
                              "PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::RouteTag<Policy::kRoute>>,
                      "this switch enumerates the two fit routes, FitRoute::kChebyshev and "
                      "FitRoute::kRationalMinimax: a third member must be given its own arm here "
                      "rather than inheriting the last one's entries");
        return BoysStatus::kDeviceError;
    }
}

} // namespace detail
/// \endcond

#if BoysFp16
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllOrdersF16WithPolicy(const int* n, const F16* x, F16* out, std::size_t count,
                                  void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate
                  || Policy::kRegionBExp == RegionBExp::kFast)
    {
        return detail::AllOrdersF16Cascade<Policy>(n, x, out, count, stream);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}

/// \cond
namespace detail {

template <EvalPolicyLike Policy>
BoysStatus AllOrdersF16Cascade(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRoute == FitRoute::kChebyshev)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16Fast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16Orders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16OrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // This lane books the cut's monomial rows itself too (kAllOrdersF16Mono,
                // kAllOrdersF16OrdersMono, boys_cuda_options.hpp), so the Horner member reaches
                // them: the split Clenshaw row is a second arithmetic of the same fit and not a
                // name of this one (AllOrdersF16Mono, boys_cuda.hpp).
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16Mono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16MonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16OrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16OrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16Narrow(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowOrders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowOrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowOrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowOrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersF16OrdersUniform(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF16OrdersUniformHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "this lane reads the float lane's own grid, one fit per order and "
                              "per interval, each read from its own block, with no seeded ladder "
                              "to step (boys_cuda_options.hpp, the grid's rows) - so a policy "
                              "naming PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16Rat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16RatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16OrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16OrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16RatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16RatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16OrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16OrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowOrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowOrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersF16NarrowOrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersF16NarrowOrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(
                        detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                        "this switch enumerates the two packing axes, PackAxis::kArguments (the "
                        "ladder reading) and PackAxis::kOrders (the per-order reading): a third "
                        "member must be given its own arm here rather than inheriting the ladder's "
                        "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersF16OrdersUniformRat(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersF16OrdersUniformRatHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one numerator/denominator pair per interval, read per order, with "
                              "no ladder for a second reading to be (boys_cuda_options.hpp, the "
                              "grid's rows) - so a policy naming PackAxis::kArguments (the ladder "
                              "reading) over FitGranularity::kUniform reaches no entry of this "
                              "class. The grid is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::RouteTag<Policy::kRoute>>,
                      "this switch enumerates the two fit routes, FitRoute::kChebyshev and "
                      "FitRoute::kRationalMinimax: a third member must be given its own arm here "
                      "rather than inheriting the last one's entries");
        return BoysStatus::kDeviceError;
    }
}
} // namespace detail
/// \endcond

#endif // BoysFp16

#if BoysFp16
/// The bfloat16 all-orders class reached by naming a policy: the entry the option table's own
/// row books for the combination the policy names (`boys_cuda_options.hpp`, the `AllOrdersBf16`
/// rows), run at the policy's division form. **The arms are the table's rows one for one**, and
/// that includes the coarsest cut's monomial pair, which this lane books as rows of its own
/// beside the split Clenshaw ones: a policy naming `EvalScheme::kHorner` over that cut reaches
/// `AllOrdersBf16Mono`, and the split Clenshaw row is not run in its place. The two cells no row
/// of the class books - the grid's ladder reading and its `kFast` - are refused with that reason.
template <EvalPolicyLike Policy>
BoysStatus BoysCuda::AllOrdersBf16WithPolicy(
    const int* n, const Bf16* x, Bf16* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRegionBExp == RegionBExp::kAccurate
                  || Policy::kRegionBExp == RegionBExp::kFast)
    {
        return detail::AllOrdersBf16Cascade<Policy>(n, x, out, count, stream);
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                      "this switch enumerates the two region-B exponentials, "
                      "RegionBExp::kAccurate and RegionBExp::kFast: a third value added to "
                      "RegionBExp must be given its own arm here rather than inheriting the last "
                      "one's kernel");
        return BoysStatus::kDeviceError;
    }
}
#endif // BoysFp16

/// \cond
namespace detail {

#if BoysFp16
/// The bfloat16 all-orders class's cascade over the route, the partition, the packing axis and
/// the scheme, at whichever member of the region-B exponential the policy names. Declared with
/// the other three at the head of this file and defined here, because its arms name this class's
/// members.
///
/// \param n the class's own first argument, as its entries document it
/// \param x the class's own second argument, as its entries document it
/// \param out the class's own third argument, as its entries document it
/// \param count the class's own fourth argument, as its entries document it
/// \param stream the class's own fifth argument, as its entries document it
///
/// \returns the status of the entry the policy's combination reaches
template <EvalPolicyLike Policy>
BoysStatus AllOrdersBf16Cascade(
    const int* n, const Bf16* x, Bf16* out, std::size_t count, void* stream) {
    if constexpr (Policy::kRoute == FitRoute::kChebyshev)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16Fast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16Orders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16OrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // This lane books the cut's monomial rows itself (kAllOrdersBf16Mono,
                // kAllOrdersBf16OrdersMono, boys_cuda_options.hpp), so the Horner member reaches
                // them: the split Clenshaw row is a second arithmetic of the same fit and not a
                // name of this one (AllOrdersBf16Mono, boys_cuda.hpp).
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16Mono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16MonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16OrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16OrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16Narrow(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowOrders(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowOrdersFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // The narrow partition's monomial rows, on the reading the coarsest cut's arm
                // states.
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowOrdersMono(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowOrdersMonoFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                // The grid's two scheme rows are one kernel under two names
                // (kAllOrdersBf16OrdersUniform and kAllOrdersBf16Uniform,
                // DeviceEntryArithmeticOf), and the arm names the per-order row, which is the
                // reading the grid holds.
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersBf16OrdersUniform(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersBf16OrdersUniformHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one fit per order and per interval, each read from its own block, "
                              "with no seeded ladder to step (the library states it of the grid's "
                              "rows, boys_cuda_options.hpp) - so a policy naming "
                              "PackAxis::kArguments (the ladder reading) over "
                              "FitGranularity::kUniform reaches no entry of this class. The grid "
                              "is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        if constexpr (Policy::kGranularity == FitGranularity::kCoarsest)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16Rat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16RatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16OrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16OrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // The pair is stored once, so both scheme names select one kernel
                // (AllOrdersBf16Rat, boys_cuda.hpp); this lane books the route's second name as
                // its own row (kAllOrdersBf16RatHorner, boys_cuda_options.hpp), so the Horner
                // member reaches that row.
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16RatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16RatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16OrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16OrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kNarrow)
        {
            if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
            {
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowOrdersRat(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowOrdersRatFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kScheme == EvalScheme::kHorner)
            {
                // The narrow partition's pair, on the reading the coarsest cut's arm states.
                if constexpr (Policy::kPack == PackAxis::kArguments)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else if constexpr (Policy::kPack == PackAxis::kOrders)
                {
                    return detail::DevicePickExp<Policy::kRegionBExp>(
                        [&] { return BoysCuda::AllOrdersBf16NarrowOrdersRatHorner(n, x, out, count, stream, Policy::kDivision); },
                        [&] {
                            return BoysCuda::AllOrdersBf16NarrowOrdersRatHornerFast(n, x, out, count, stream,
                                                         Policy::kDivision);
                        });
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                                  "this switch enumerates the two packing axes, "
                                  "PackAxis::kArguments (the ladder reading) and "
                                  "PackAxis::kOrders (the per-order reading): a third member must "
                                  "be given its own arm here rather than inheriting the ladder's "
                                  "entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                              "this switch enumerates the two schemes, EvalScheme::kSplitClenshaw "
                              "and EvalScheme::kHorner: a third member must be given its own arm "
                              "here rather than inheriting the split Clenshaw's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else if constexpr (Policy::kGranularity == FitGranularity::kUniform)
        {
            if constexpr (Policy::kRegionBExp == RegionBExp::kFast)
            {
                static_assert(detail::kAlwaysFalse<detail::ExpTag<Policy::kRegionBExp>>,
                              "no kernel: the grid reads no exponential at any argument: below the "
                              "join every order is summed from its own stored block, and above it "
                              "the call falls to the asymptote, whose seed is the reciprocal "
                              "square root (boys/boys_cuda_options.hpp). RegionBExp has no member "
                              "at FitGranularity::kUniform, so a policy naming "
                              "RegionBExp::kFast over the grid names a combination this class "
                              "does not carry");
                return BoysStatus::kDeviceError;
            }
            else if constexpr (Policy::kPack == PackAxis::kOrders)
            {
                // The grid's rational rows: the pair is stored once and read by Horner, so the
                // scheme names select the row the table books for each.
                if constexpr (Policy::kScheme == EvalScheme::kSplitClenshaw)
                {
                    return BoysCuda::AllOrdersBf16OrdersUniformRat(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else if constexpr (Policy::kScheme == EvalScheme::kHorner)
                {
                    return BoysCuda::AllOrdersBf16OrdersUniformRatHorner(
                        n, x, out, count, stream, Policy::kDivision);
                }
                else
                {
                    static_assert(detail::kAlwaysFalse<detail::SchemeTag<Policy::kScheme>>,
                                  "this switch enumerates the two schemes, "
                                  "EvalScheme::kSplitClenshaw and EvalScheme::kHorner: a third "
                                  "member must be given its own arm here rather than inheriting "
                                  "the split Clenshaw's entry");
                    return BoysStatus::kDeviceError;
                }
            }
            else if constexpr (Policy::kPack == PackAxis::kArguments)
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "no kernel: the uniform grid holds one member of the packing axis - "
                              "one numerator/denominator pair per interval, read per order, with "
                              "no ladder for a second reading to be (boys_cuda_options.hpp, the "
                              "grid's rows) - so a policy naming PackAxis::kArguments (the ladder "
                              "reading) over FitGranularity::kUniform reaches no entry of this "
                              "class. The grid is read at PackAxis::kOrders");
                return BoysStatus::kDeviceError;
            }
            else
            {
                static_assert(detail::kAlwaysFalse<detail::PackingTag<DevicePacking::kLadder>>,
                              "this switch enumerates the two packing axes, PackAxis::kArguments "
                              "(the ladder reading) and PackAxis::kOrders (the per-order "
                              "reading): a third member must be given its own arm here rather than "
                              "inheriting the last one's entry");
                return BoysStatus::kDeviceError;
            }
        }
        else
        {
            static_assert(detail::kAlwaysFalse<detail::PartitionTag<Policy::kGranularity>>,
                          "this switch enumerates the three fit partitions, "
                          "FitGranularity::kCoarsest, FitGranularity::kNarrow and "
                          "FitGranularity::kUniform: a fourth member must be given its own arm "
                          "here rather than inheriting the last one's entry");
            return BoysStatus::kDeviceError;
        }
    }
    else
    {
        static_assert(detail::kAlwaysFalse<detail::RouteTag<Policy::kRoute>>,
                      "this switch enumerates the two fit routes, FitRoute::kChebyshev and "
                      "FitRoute::kRationalMinimax: a third member must be given its own arm here "
                      "rather than inheriting the last one's entries");
        return BoysStatus::kDeviceError;
    }
}
#endif // BoysFp16

} // namespace detail
/// \endcond

} // namespace boys
