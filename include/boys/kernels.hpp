#pragma once

/// \file kernels.hpp
/// The entries of `boys/boys.hpp` as templates over the evaluation policy a caller names.
///
/// A call that names no policy is resolved by the library: the plain declarations in the
/// front door are compiled once, in the library's own translation unit and under the build's
/// seam, so every consumer of a build links the same values for them. This header is the
/// other half - the spelling for a caller that wants a combination the build does not carry
/// as a default, and the one a translation unit that links no library at all has to use.
///
/// **Nothing here is declared in the front door.** Naming a policy without this header is a
/// compile error at the call site rather than a promise the linker has to keep, which is the
/// difference between one surface and two that agree until they do not.
///
/// What this header costs is the kernel: the coefficient tables, the effective degrees and
/// the bodies behind every entry. That is what a caller that names no policy does not pay
/// for.

#include "boys/boys.hpp"

namespace boys {

/// The front door's \c BoysSingle at a policy the caller names.
///
/// \tparam Policy the evaluation policy (\c EvalPolicy): the fit route, the
///         scheme its coefficients are summed in, the partition of the fitted
///         regions, and a single-precision engine's budget, selected together.
///         The default is this call's own class row - \c DefaultPolicy at the
///         precision and shape of the entry above - so a call site that names no
///         axis resolves through the table the build's seam carries, and naming
///         any axis is how a caller asks for another
///
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kSingle>>
double BoysSingle(int n, double x) noexcept;

/// The front door's \c BoysSingleChecked at a policy the caller names.
///
/// \tparam Policy as \c BoysSingle.
///
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 and not NaN
/// \param out   written with F_n(x) on success and left untouched otherwise
/// \returns     \c BoysStatus::kSuccess, or \c BoysStatus::kInvalidArgument when the contract is
///              broken. Every value is written or none is.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kSingle>>
BoysStatus BoysSingleChecked(int n, double x, double* out) noexcept;

/// The front door's \c BoysAllOrders at a policy the caller names.
///
/// \tparam Policy see BoysSingle
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>
void BoysAllOrders(int nmax, double x, double* out) noexcept;

/// The front door's \c BoysFixedN at a policy the caller names.
///
/// \tparam Policy see BoysSingle. A call naming a route other than the shipped
///         one, or naming the uniform grid, is answered by the per-argument
///         single entry once per argument, whose body this entry's own path
///         already mirrors region for region. Both keep this entry's documented
///         accuracy - the path costs the shaped body's region dispatch, not a
///         value.
///
/// \param n      order, 0..kMaxBoysOrder - the batch's single fixed order
/// \param x      array of count arguments, each >= 0
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in doubles, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride = 1) noexcept;

/// The front door's \c BoysAllN at a policy the caller names.
///
/// \tparam Policy see BoysSingle. The route and the partition are properties of
///         the fit a value is read from, so a call naming the rational route or
///         the uniform grid takes the per-argument path, whose body is the
///         all-orders entry's own; both shapes answer inside this entry's own
///         bound
///
/// \param nmax      highest order, 0..kMaxBoysOrder
/// \param x         array of count arguments, each >= 0
/// \param out       receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count     number of arguments; may be 0 (no writes)
/// \param workspace grouping scratch of BoysAllNWorkspaceSize(count) words, or
///                  nullptr to allocate internally; its contents on return are
///                  unspecified
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace = nullptr) noexcept;

/// The front door's \c BoysAllN at a policy the caller names.
///
/// \tparam Policy see BoysAllN: the same entry and the same paths, reached
///         without the sort
///
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      array of count arguments, non-decreasing, each >= 0
/// \param out    receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count  number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
void BoysAllN(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;

/// The front door's \c BoysAllNAtOrders at a policy the caller names.
///
/// \tparam Policy see BoysSingle. This entry runs the per-argument all-orders
///         body, so it takes the same policies that entry takes
///
/// \param n     array of count top orders, each 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives the planes: out[k * count + i] = F_k(x[i]) for
///              k = 0..n[i], every cell above n[i] untouched
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>
void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept;

/// The front door's \c BoysAllOrdersChecked at a policy the caller names.
///
/// \tparam Policy  as the unchecked entry of the same shape.
///
/// \param nmax     the highest order, 0..kMaxBoysOrder
/// \param x        the argument, >= 0 and not NaN
/// \param out      written on success and left untouched otherwise
/// \returns        \c BoysStatus::kSuccess, or \c BoysStatus::kInvalidArgument when the contract
///                 is broken.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>
BoysStatus BoysAllOrdersChecked(int nmax, double x, double* out) noexcept;

/// The front door's \c BoysFixedNChecked at a policy the caller names.
///
/// \tparam Policy  as \c BoysFixedN.
///
/// \param n        the order, 0..kMaxBoysOrder
/// \param x        the arguments, each >= 0 and not NaN
/// \param out      written on success and left untouched otherwise
/// \param count    the number of arguments; may be 0 (no writes, and then \p x and \p out may be
///                 null)
/// \param stride   the distance between the elements the values are written to, >= 1
/// \param badIndex written with the index of the first argument that was refused, or left alone
///                 when every one of them is in contract. May be null.
/// \returns        \c BoysStatus::kSuccess, or \c BoysStatus::kInvalidArgument.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
BoysStatus BoysFixedNChecked(int n, const double* x, double* out, std::size_t count,
                             std::size_t stride, std::size_t* badIndex = nullptr) noexcept;

/// The front door's \c BoysAllNChecked at a policy the caller names.
///
/// \tparam Policy   as \c BoysAllN.
///
/// \param nmax      the highest order, 0..kMaxBoysOrder
/// \param x         the arguments, each >= 0 and not NaN
/// \param out       written on success and left untouched otherwise
/// \param count     the number of arguments
/// \param workspace scratch the grouped path uses, or null
/// \param badIndex  written with the index of the first argument that was refused, or left alone
///                  when every one of them is in contract. May be null.
/// \returns         \c BoysStatus::kSuccess, or \c BoysStatus::kInvalidArgument.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
BoysStatus BoysAllNChecked(int nmax, const double* x, double* out, std::size_t count,
                           std::size_t* workspace = nullptr,
                           std::size_t* badIndex = nullptr) noexcept;

/// The front door's \c BoysAllNAtOrdersChecked at a policy the caller names.
///
/// \tparam Policy   as \c BoysAllNAtOrders.
///
/// \param n         the top order of each argument, each 0..kMaxBoysOrder
/// \param x         the arguments, each >= 0 and not NaN
/// \param out       written on success and left untouched otherwise
/// \param count     the number of arguments
/// \param badIndex  written with the index of the first argument that was refused, or left alone
///                  when every one of them is in contract. May be null.
/// \returns         \c BoysStatus::kSuccess, or \c BoysStatus::kInvalidArgument.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>
BoysStatus BoysAllNAtOrdersChecked(const int* n, const double* x, double* out, std::size_t count,
                                   std::size_t* badIndex = nullptr) noexcept;

/// The front door's \c BoysSingleF32 at a policy the caller names.
///
/// \tparam Policy see BoysSingle. The lane reads the policy's fit route and its
///         evaluation scheme: the route selects which family supplies the lane's
///         own fits - the shipped Chebyshev table or the rational minimax one -
///         and the scheme selects which of the Chebyshev family's two parallel
///         tables, the Chebyshev form or the monomial form of the same fits, is
///         summed. The packing axis is not an axis of this shape and is refused
///         where it is named: this entry answers one order at one argument, so
///         the value is a single float and there is no lane to fill
///
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kSingle>>
float BoysSingleF32(int n, float x) noexcept;

/// The front door's \c BoysAllOrdersF32 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits -
///         the route for this lane's region-B seed and for the double lane's fit
///         that seeds region A, the scheme for which of the route's tables each
///         of those reads. Region A's seed is the double lane's fit at the
///         policy's pair, so its degrees are the double lane's; region B's is
///         this lane's own. The packing axis is read here too: the orders axis
///         packs eight orders of the single argument and reads the tables the
///         policy names, each order's own piece of the partition named, and it
///         carries either route. Naming the arguments axis names no lane this
///         shape has to fill: the values are the per-argument body's either way,
///         and the batch entry is where an array of arguments is answered, one
///         argument at a time
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kAllOrders>>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept;

/// The front door's \c BoysAllNF32 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits.
///         The packing axis is read at every argument - the all-orders body this
///         entry calls packs eight orders of one argument under the orders axis -
///         and what stays at the caller's loop is the region partitioning of the
///         arguments axis
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives count * (nmax + 1) floats, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kAllN>>
void BoysAllNF32(int nmax, const float* x, float* out, std::size_t count) noexcept;

/// The front door's \c BoysFixedNF32 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits.
///         Naming PackAxis::kOrders is rejected at the call site by the assertion
///         the single body carries - one order at one argument has no second
///         order to fill a vector lane with - and the wide dimension this call
///         does have, count, is the arguments axis the caller's own loop is
///
/// \param n      order, 0..kMaxBoysOrder - the batch's single fixed order
/// \param x      array of count arguments, each >= 0
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in floats, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kFixedN>>
void BoysFixedNF32(
    int n, const float* x, float* out, std::size_t count, std::size_t stride = 1) noexcept;

/// The front door's \c BoysAllNAtOrdersF32 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits.
///         This entry runs the per-argument all-orders body, so it takes the
///         same policies that entry takes, the packing axis read at every
///         argument
///
/// \param n     array of count top orders, each 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives the planes: out[k * count + i] = F_k(x[i]) for
///              k = 0..n[i], every cell above n[i] untouched
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kAllNAtOrders>>
void BoysAllNAtOrdersF32(const int* n, const float* x, float* out, std::size_t count) noexcept;

#if BoysFp16
/// The front door's \c BoysSingleF16 at a policy the caller names.
///
/// \tparam Policy the evaluation policy (\c EvalPolicy): the fit route, the
///         scheme its coefficients are summed in, the partition of the fitted
///         regions and the packing axis, selected together;
///
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \returns     F_n(x) in fp16
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kSingle>>
F16 BoysSingleF16(int n, F16 x) noexcept;

/// The front door's \c BoysAllOrdersF16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF16
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kAllOrders>>
void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept;

/// The front door's \c BoysFixedNF16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF16: the budget is this lane's, and naming
///         another is rejected at the call site
///
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of count arguments, each >= 0 (fp16)
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in fp16 values, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kFixedN>>
void BoysFixedNF16(
    int n, const F16* x, F16* out, std::size_t count, std::size_t stride = 1) noexcept;

/// The front door's \c BoysAllNF16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF16
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0 (fp16)
/// \param out   receives count * (nmax + 1) values, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kAllN>>
void BoysAllNF16(int nmax, const F16* x, F16* out, std::size_t count) noexcept;

/// The front door's \c BoysAllNAtOrdersF16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF16
///
/// \param n     array of count top orders, each 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0 (fp16)
/// \param out   receives the planes: out[k * count + i] = F_k(x[i]) for
///              k = 0..n[i], every cell above n[i] untouched
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kAllNAtOrders>>
void BoysAllNAtOrdersF16(const int* n, const F16* x, F16* out, std::size_t count) noexcept;

/// The front door's \c BoysSingleBf16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleF16. The half lane's budget is one budget and its
///         two formats are two classes of it, so the default here is this class's
///         own row: \c DefaultPolicy<Precision::kBf16, Shape::kSingle>, which the
///         table carries beside the fp16 class's and not in place of it
///
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \returns     F_n(x) in bf16
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kBf16, Shape::kSingle>>
Bf16 BoysSingleBf16(int n, Bf16 x) noexcept;

/// The front door's \c BoysAllOrdersBf16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleBf16
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kBf16, Shape::kAllOrders>>
void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept;

/// The front door's \c BoysFixedNBf16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleBf16: the budget is this lane's, and naming
///         another is rejected at the call site
///
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of count arguments, each >= 0 (bf16)
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in bf16 values, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kBf16, Shape::kFixedN>>
void BoysFixedNBf16(
    int n, const Bf16* x, Bf16* out, std::size_t count, std::size_t stride = 1) noexcept;

/// The front door's \c BoysAllNBf16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleBf16
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0 (bf16)
/// \param out   receives count * (nmax + 1) values, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kBf16, Shape::kAllN>>
void BoysAllNBf16(int nmax, const Bf16* x, Bf16* out, std::size_t count) noexcept;

/// The front door's \c BoysAllNAtOrdersBf16 at a policy the caller names.
///
/// \tparam Policy see BoysSingleBf16
///
/// \param n     array of count top orders, each 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0 (bf16)
/// \param out   receives the planes: out[k * count + i] = F_k(x[i]) for
///              k = 0..n[i], every cell above n[i] untouched
/// \param count number of arguments; may be 0 (no writes)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kBf16, Shape::kAllNAtOrders>>
void BoysAllNAtOrdersBf16(const int* n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

/// \cond
/// Hidden from the API reference: each of these is an instantiation of the entries
/// declared above at the default policy, not an entry of its own. They are what
/// the default call sites link against instead of compiling the kernel again in
/// their own translation unit; a caller that names another policy compiles the
/// combination it asks for from the definition below.
///
/// Declared only for the table this build carries. A build that supplies its own
/// through BOYS_BUILD_DEFAULTS has its own policy types - the table is read by
/// value, so a different row is a different instantiation with a different
/// mangled name - and the library carries no definition for those. Declaring them
/// here would promise a definition that does not exist and turn a working consumer
/// build into an unresolved external.
#if !defined(BOYS_BUILD_DEFAULTS_REPLACED)
extern template double BoysSingle<>(int n, double x) noexcept;
extern template void BoysAllOrders<>(
    int nmax, double x, double* out) noexcept;
extern template void BoysFixedN<>(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept;
extern template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept;
extern template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;
extern template void BoysAllNAtOrders<>(
    const int* n, const double* x, double* out, std::size_t count) noexcept;
// The single-precision entries are not declared extern here: an extern promises a definition
// elsewhere, and the library's fixed list of them (src/boys_orders_simd.cpp) need not hold the
// one a call site selects from the table - declaring it anyway left four consumer targets
// unresolved on gcc. Their definitions are in this header: a consumer instantiates what it names.

#if BoysFp16
extern template F16 BoysSingleF16<>(int n, F16 x) noexcept;
extern template void BoysAllOrdersF16<>(
    int nmax, F16 x, F16* out) noexcept;
extern template Bf16 BoysSingleBf16<>(int n, Bf16 x) noexcept;
extern template void BoysAllOrdersBf16<>(
    int nmax, Bf16 x, Bf16* out) noexcept;
#endif // BoysFp16
#endif // !BOYS_BUILD_DEFAULTS_REPLACED
/// \endcond

} // namespace boys

// The kernel behind the entries above, shipped as a header so that every policy a caller names
// is instantiable at the call site. Included here rather than at the top of this file because
// its definitions name the declarations above, and a declaration that follows a definition may
// not carry a default argument.
#include "boys/boys_impl.hpp"
