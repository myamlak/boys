#pragma once

/// \file backend.hpp
/// Named arithmetic backends: the value type, the packed type, the load /
/// store / broadcast transport, and the multiply-add with its contraction
/// behaviour, in one place per precision and width.
///
/// A lane is an arithmetic over one value type at one width, and the kernels
/// that differ only in that arithmetic are written once against this concept
/// instead of once per width. Which arithmetic a lane runs in is a name a
/// report can print (see BoysBackends), so a number a caller holds can be
/// attributed to the arithmetic that produced it rather than to a type only
/// the compiler knows.
///
/// Two multiply-adds are named, and the difference between them is the whole
/// point of naming them:
///
///  - MulAdd is fused: the product is exact and the sum rounds once. It is the
///    operation a Chebyshev recurrence wants, and the one the kernels are
///    written to use. Which arithmetic a build delivers for it is the lane's
///    MulAddRoute: the fused step is an instruction where the target has one
///    and a call into the C runtime where it does not, and the alternative to
///    that call is the two-rounding route described below.
///  - MulSub rounds twice: the product rounds, then the difference rounds. The
///    two are not interchangeable, and neither is a compiler option: a source
///    that leaves `a * b - c` bare is a different arithmetic on a target that
///    contracts than on one that does not, so a kernel that needs the two
///    roundings says MulSub and gets them on every build.
///
/// The fused step is the expensive one on a target without the instruction: a
/// Clenshaw evaluation is deg + 1 of them per value, so a plain x86-64 build
/// that leaves MulAdd on the C runtime pays that many calls per value. The
/// separate route is the alternative: the product rounds and then the sum
/// rounds, two roundings rather than one, written bare. It is a different
/// arithmetic and carries a different error, which is why it is a route a
/// build selects and a report prints rather than a silent substitution.
///
/// The two routes coincide on a build that contracts a bare product-plus-add,
/// because the bare form then IS the fused step. That is measured rather than
/// assumed (Contracts()), and it is why RouteInForce() can report the fused
/// route for a build that was asked for the separate one.
///
/// Contracts() reports whether a bare `a * b + c` in a backend's arithmetic is
/// a single rounding. It is measured rather than declared, because contraction
/// follows from the target and the flags rather than from the source — and it
/// is therefore a property of one compiled translation unit, not of the
/// library as a whole. Each backend's Contracts() answers for the unit that
/// calls it; BackendInfo carries the answer measured where that backend's
/// kernels are compiled, so the table describes the arithmetic the lanes
/// actually run rather than some other unit's.
///
/// What a call site selects about *how* an evaluation runs — the fit route
/// (FitRoute), the scheme the fit's coefficients are summed in (EvalScheme),
/// a single-precision engine's budget (BoysBudget), the axis a packed
/// evaluation vectorises over (PackAxis) and how narrowly the fitted domain is
/// cut into pieces (FitGranularity) — is named here too, and is
/// carried as ONE parameter (EvalPolicy) rather than as one parameter per axis,
/// so a further axis is a field on the policy rather than a parameter on every
/// entry, engine and kernel between the call site and the fit. The contract a
/// fit route's coefficient set satisfies is FitPolicy, and EvalPolicyLike is
/// the constraint the entries carry, so a combination the library does not
/// carry fails where it is named rather than inside a recurrence.
///
/// \ingroup boys

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace boys {

/// Which certified fit serves a region, where the library carries more than
/// one for it.
///
/// Region A and region B each carry two tables of fitted coefficients, placed
/// against the same bar over the same interval, and this names which one a call
/// evaluates: region B's is one lowest-order seed per route that the higher
/// orders are reached from, and region A's is a fit of every order over each of
/// the region's two bands. They are alternatives rather than rungs of one
/// design: the Chebyshev tables are the default and the route every lane has
/// always been certified with, and naming the rational one changes the fits
/// that serve the intervals it covers and nothing else.
///
/// What each route promises, over what interval, and from which argument naming
/// it changes anything, is BoysFitRoutes' report rather than this enumeration's:
/// those are properties of the fit and of the selector, and a route whose fit
/// reaches further than its selector takes over says so there instead of being
/// stretched to claim a domain it does not serve. Region A's rational route is
/// the case that rule exists for.
///
/// A value outside the enumerators - cast in from outside the enum, or named by
/// a newer header - is not a route, and every entry on this surface treats it
/// as \c kChebyshev: the default is the route every build carries, so a caller
/// is never handed a fit they did not ask for.
///
/// \ingroup boys
enum class FitRoute : int {
    /// The Chebyshev fits: the default, and the route the certified lanes are
    /// defined by.
    kChebyshev = 0,
    /// The rational minimax fits of region A's pieces and of region B's seed.
    kRationalMinimax,
};

/// The route the entries evaluate when the caller names none.
inline constexpr FitRoute kDefaultFitRoute = FitRoute::kChebyshev;

/// Which summation a stored fit is evaluated by.
///
/// A fit is a polynomial whatever scheme sums it: the Chebyshev coefficients
/// and the monomial coefficients of one fit describe the same function, at the
/// same degree, to within the rounding of the two tables. The schemes differ
/// in the arithmetic they round in and in the work they do, not in what they
/// approximate, so both are offered and neither replaces the other.
///
/// The scheme reaches the double scalar lane (per-order region-A fits, the
/// extended-band seed and the region-B seed). Region C is evaluated by its
/// asymptotic form and has no stored fit to sum, so it is the same arithmetic
/// under either scheme; the half-precision lanes and the packed region-A lane
/// are the split Clenshaw route's and are not offered under the other scheme.
///
/// The scheme is one axis of the evaluation and the fit route (FitRoute) is
/// another, and the two are named together by an EvalPolicy. They select
/// different things: the route names the fits that serve the regions, and the
/// scheme names the summation a fit's coefficients are read in. A fit whose
/// coefficients have two stored forms is summed by either scheme; a fit whose
/// coefficients have one - the rational family's monomial numerator and
/// denominator - is read in that form whichever scheme is named, and the scheme
/// still reaches the parts of a call the named route's fits do not serve, which
/// are the shipped family's.
///
/// \ingroup boys
enum class EvalScheme : std::uint8_t {
    /// The even/odd split Clenshaw recurrence on the Chebyshev form: the
    /// certified route, and the default.
    kSplitClenshaw = 0,

    /// Horner's rule on the monomial form of the same fit.
    kHorner = 1,
};

/// The scheme the entries evaluate in when the caller names none.
///
/// The certified route: naming the other one is how a caller asks for it, so a
/// call site that names no scheme is compiled exactly as it was before one
/// existed.
inline constexpr EvalScheme kDefaultEvalScheme = EvalScheme::kSplitClenshaw;

/// Which axis a packed lane vectorises over.
///
/// A packed lane keeps four doubles in a register and a call has to supply four
/// of something. Which something is a property of the call shape rather than of
/// the kernel: \c BoysAllOrders is one argument and every order, so the orders
/// are the only axis with anything in it, while the batch shapes carry several
/// arguments at one order and can fill a register with those. The two are not
/// rungs of one design and neither replaces the other - they evaluate the same
/// stored fits by the same arithmetic, and differ in what the vector lanes
/// hold and therefore in what a caller pays.
///
/// The axis is one field of an \c EvalPolicy, like the fit route and the
/// scheme. Naming the orders axis on an entry that has one order is a
/// combination this library does not carry and is refused where it is named:
/// a packed lane keeps four orders of one argument and such a call produces
/// one. The plane entry carries the axis as well, trading the region grouping
/// that exists to feed the across-arguments lane for it.
///
/// \ingroup boys
enum class PackAxis : std::uint8_t {
    /// Four arguments of one order: the packed lanes this library has always
    /// shipped, reached through the fixed-order and plane entries.
    kArguments = 0,

    /// Four orders of one argument: the shape \c BoysAllOrders has, and the
    /// axis its own packed lane packs.
    kOrders = 1,
};

/// The axis the entries pack when the caller names none.
///
/// The shipped one, so a call site that names no axis compiles exactly as it
/// did before the other existed.
inline constexpr PackAxis kDefaultPackAxis = PackAxis::kArguments;

/// The name a report prints a packing axis under.
///
/// \param axis the axis
///
/// \returns a string literal naming it
const char* PackAxisName(PackAxis axis) noexcept;
/// How narrowly the fitted domain is cut into pieces.
///
/// A stored fit is a polynomial over one interval, and a narrower interval
/// needs a lower degree to hold the same bound: the truncation bound carries
/// the interval's half-width as `(h/(2d))^d`, so halving a piece buys roughly
/// `2^d` and raising the degree at a fixed width buys far less. Splitting is
/// therefore the lever, and how far to take it is a choice with a price on each
/// side — a narrower piece is fewer coefficients to evaluate per call and more
/// pieces to store and to select between.
///
/// Two partitions are offered rather than a spectrum. Each carries its own
/// stored counts and its own certified bound, and neither is a rung of the
/// other: naming one changes the fits that serve the intervals its own report
/// names, and the shipped partition's coefficients are byte-identical whether
/// or not the other exists.
///
/// **Narrowing is a trade and not a saving.** A narrower piece is fewer
/// coefficients to read per evaluation and more pieces to store and to look up
/// in, so the axis trades per-evaluation work against table size. A consumer
/// whose cost is per evaluation gains; one whose cost is the table gains
/// nothing and pays the lookup. It is offered for a consumer to choose between
/// and not as a winner.
///
/// **The axis cuts both fitted regions, and region A pays a second criterion.**
/// Region B's seed is read for itself, so a narrow piece there is held to the
/// same bound as any other fit of that interval. Region A's pieces are read two
/// ways: a single-order call reads one for its own value, and the batch entry's
/// relaxed path reads one as the seed of a downward recursion that carries its
/// error down to F_0 with a gain of `max(1, b^n / prod(j + 1/2))` at the piece's
/// right end `b`. That gain's envelope over region A reaches 1.04e5, at order 12
/// and the region's right edge, while the seeding fallback is taken only below
/// the band's left edge, where the gain a call reaches is the calling order's
/// own - 2.18 at order 1, 1.58 at order 2 and 1 at every order from 3 up - and
/// the walk holds the envelope rather than that, so that no piece's reading
/// depends on which order an entry seeds from. A narrow piece is therefore held
/// to whichever of the two readings is tighter at its own right end, so the gain
/// binds only where it exceeds the ratio of the two bars, and a piece far enough
/// left is cut to the single-order bar alone. Both readings are parts of one
/// criterion rather than a choice.
///
/// A member the build cannot serve is refused where it is named, with the
/// reason, rather than answered from the shipped tables: the two partitions'
/// coefficients are different fits of the same function over the same
/// interval, so a silent substitution would return the shipped partition's
/// values under the other's name. The refusals are the route that carries no
/// narrow table, the relaxed rungs that truncate the shipped fits to certified
/// effective degrees, the single-precision lanes, which hold one coefficient
/// set, and the across-orders packing axis, whose kernel reads one order's
/// coefficients at a fixed stride and so needs the pieces to share their shape
/// from order to order, which a per-order cut does not. Each is unbuilt work
/// rather than an impossible combination, and each is named where it is refused
/// so that it can be counted.
///
/// \ingroup boys
enum class FitGranularity : std::uint8_t {
    /// The partition this library has always shipped: region A's two bands per
    /// order and region B's single seed, at the degrees the committed tables
    /// carry. The default, and the partition every certified lane is defined
    /// by.
    kShipped = 0,

    /// A deliberately narrower partition of both fitted regions, at the degrees
    /// the proved truncation bound gives a piece of that width at the bar the
    /// piece is read under. Fewer coefficients per evaluation, more pieces in
    /// the table.
    kNarrow = 1,
};

/// The partition the entries evaluate when the caller names none.
///
/// The shipped one, so a call site that names no partition compiles the tables
/// it always did.
inline constexpr FitGranularity kDefaultFitGranularity = FitGranularity::kShipped;

/// The name a report prints a granularity under.
///
/// \param granularity the partition
///
/// \returns a string literal naming it
const char* GranularityName(FitGranularity granularity) noexcept;

/// The computation budget a single-precision engine evaluates at.
///
/// The lanes that keep half-precision arguments and compute in single (f16.hpp)
/// hold a tighter bar than the float lane's, so the engine both lanes call is
/// told which of the two budgets it runs under rather than assuming one. The
/// double lanes carry no such choice and read no budget.
///
/// \ingroup boys
enum class BoysBudget : std::uint8_t {
    /// The float lane's 1.5e-7 budget.
    kFloat = 0,

    /// The fp16/bf16 lanes' tighter 1e-7 budget.
    kFp16 = 1,
};

/// The region-A partition a fit reads: which piece of the region an argument
/// falls in and what that piece is.
///
/// A fit names its partition rather than carrying a lookup of its own, because
/// the lookup is the same on every partition - scan the order's pieces for the
/// first whose right edge is past the argument - and only the table differs.
/// The two members are that lookup and that table, so a fit over a second
/// partition of the same region is the same evaluation over other rows.
///
/// \ingroup boys
template <typename P>
concept RegionAPartition = requires(std::size_t index, int order, double x) {
    { P::PieceIndex(order, x) } -> std::convertible_to<std::size_t>;
    { P::PieceAt(index) };
};

/// The fit one region-A route evaluates: its coefficient set and the scheme
/// the coefficients are read in, named as a type so the evaluation body around
/// it is written once and instantiated per route. The body is in
/// boys_impl.hpp, and each model of this concept sits with the coefficient
/// tables it reads, because a model reads them.
///
/// The body owns everything that is not the fit - the zero argument's closed
/// form, the region split, the piece lookup, the mapped argument, the
/// recurrences, the per-order rule and the domains - so a route cannot drift
/// from the lane around it, and asks a policy for four things:
///
///  - `Partition` is the region-A partition the fit's pieces are stored in, as
///    a model of RegionAPartition. The pieces, their intervals and the mapping
///    are that table's, so two fits over one region differ in the scheme or in
///    the partition rather than in how either is read.
///  - `EvalPiece(index, t)` is one region-A piece's value at its own mapped
///    argument `t`, named by the piece's index in that partition.
///  - `RegionBSeed(x)` is region B's seed F_0(x).
///  - `BandSource` reads the orders the band's regime covers. A source is
///    constructed at an argument and stepped order by order, so a batch pays
///    one step per order rather than one whole run of the recurrence per
///    order. The requirement spells the type `typename`: a dependent
///    qualified name is otherwise read as a value, and a compiler that does
///    read it that way rejects the policy.
///
/// `kRegionAFitsFrom` is the argument at and above which the policy's own
/// answer reads those orders.
template <typename F>
concept FitPolicy = requires(std::size_t index, double t, double x, int l) {
    { F::kRegionAFitsFrom } -> std::convertible_to<double>;
    { F::EvalPiece(index, t) } -> std::same_as<double>;
    { F::RegionBSeed(x) } -> std::same_as<double>;
    { typename F::BandSource(x).Next(l, x) } -> std::same_as<double>;
} && RegionAPartition<typename F::Partition>;

/// The fit one (route, scheme) pair evaluates. The families themselves are in
/// boys_impl.hpp beside the coefficient tables they read, and this is the join
/// between the two axes.
///
/// The axes select different things, which is why the join is not a triangle:
/// the route names the fits, and the scheme names the summation those fits'
/// coefficients are read in where they have two stored forms. A route whose own
/// fit has one form is the same fit under either scheme, and the scheme still
/// reaches the parts of a call that route's fits do not serve - so every pair of
/// the two enumerations is carried, and the only value rejected here is one
/// outside the route enumeration.
namespace detail {

template <EvalScheme kScheme, FitGranularity kGranularity>
struct ChebyshevFit;

struct RationalFit;

/// A route outside the FitRoute enumeration: not a selection this library can
/// answer, and rejected where the caller names it rather than quietly evaluated
/// at the default. The run-time selector takes the other reading - a route value
/// outside the enumeration is the default there - because a value that arrives
/// at run time is not a caller's compile-time claim that the option exists.
template <FitRoute kRoute, EvalScheme kScheme, FitGranularity kGranularity>
struct RouteFit {
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a fit route outside the FitRoute enumeration is not a route the library "
                  "carries: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    /// Unreachable: the static assert above refuses a call site naming a route
    /// outside the enumeration, so only the two specializations below are asked.
    using Type = ChebyshevFit<kScheme, kGranularity>;
};

/// The Chebyshev family: it holds both coefficient sets, so either scheme is
/// defined for it, and both partitions, so either granularity is too.
template <EvalScheme kScheme, FitGranularity kGranularity>
struct RouteFit<FitRoute::kChebyshev, kScheme, kGranularity> {
    /// The Chebyshev coefficients at this scheme.
    using Type = ChebyshevFit<kScheme, kGranularity>;
};

/// The rational family: one fit under either scheme, because its coefficients
/// are a monomial numerator and denominator with no Chebyshev form to sum.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kShipped> {
    /// The single rational fit, shared by both schemes.
    using Type = RationalFit;
};

/// The rational route under the narrow partition: refused where it is named.
/// The route's region-B seed is one minimax pair over the whole interval and
/// has no partition of its own, so there is no narrow rational table to read -
/// and answering from the pair this route does ship would return the shipped
/// partition's values under the narrow partition's name.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kNarrow> {
    static_assert(kScheme == EvalScheme::kSplitClenshaw && kScheme == EvalScheme::kHorner,
                  "the rational minimax route carries one numerator/denominator pair over the "
                  "whole of region B and no narrow partition of it: name "
                  "FitGranularity::kShipped for this route, or FitRoute::kChebyshev for the "
                  "narrow partition");
    using Type = RationalFit;
};

} // namespace detail

/// Everything a call site selects about how an evaluation is performed, apart
/// from the accuracy multiplier: the fit route, the scheme the fit's
/// coefficients are summed in, the budget a single-precision engine runs at,
/// and the axis a packed evaluation vectorises over. One parameter rather than
/// one per axis, so an axis added later is a field here rather than an argument
/// on every entry, engine and kernel between the call site and the fit.
///
/// The axes are selected together because they are one evaluation rather than
/// because either implies the other: each names a different thing, and the pair
/// of the first two names the fit the bodies evaluate. \c Fit is where that
/// join happens.
///
/// The packing axis is not part of that join: it names which axis a call's
/// values are spread over, so it is read by the entries that have a wide axis
/// to fill and changes nothing about which fit answers. It composes with the
/// other axes rather than selecting among them.
/// coefficients are summed in, and the budget a single-precision engine runs
/// at. One parameter rather than one per axis, so an axis added later is a
/// field here rather than an argument on every entry, engine and kernel between
/// the call site and the fit.
///
/// The axes are selected together because they are one evaluation rather than
/// because either implies the other: each names a different thing, and the pair
/// names the fit the bodies evaluate. \c Fit is where that join happens.
///
/// The one error this type can carry is a route outside the FitRoute
/// enumeration, and it is reported where it is named: the entries constrain
/// their parameter with EvalPolicyLike, and RouteFit's own assertion is where a
/// value that is not an option fails.
///
/// The one error the axes carry beyond a route outside the enumeration is a
/// combination the build cannot serve - a partition for a route that has none,
/// a partition on a rung that truncates the shipped fits, a partition on a lane
/// that holds one coefficient set, a partition on the packing axis whose kernel
/// reads the shipped pieces' shape from order to order - and each is refused
/// where it is named rather than at a kernel, because the partitions are
/// different fits of the same function over the same interval and a fallback
/// would return the shipped values under the other partition's name. Each is
/// unbuilt work and is refused with the work it names, so a gap is countable.
///
/// \tparam kFitRoute      the fit route; \c FitRoute::kChebyshev by default
/// \tparam kEvalScheme    the scheme the fit's coefficients are summed in;
///                        \c EvalScheme::kSplitClenshaw by default
/// \tparam kEngineBudget  the computation budget of a single-precision engine,
///                        read by the float and half-precision lanes and by
///                        nothing else; \c BoysBudget::kFloat by default
/// \tparam kPackedAxis    the axis a packed evaluation vectorises over, read by
///                        the entries that have a wide axis to fill;
///                        \c PackAxis::kArguments by default
/// \tparam kGranularity   how narrowly the fitted domain is cut into pieces;
///                        \c FitGranularity::kShipped by default
///
/// \ingroup boys
template <FitRoute kFitRoute = kDefaultFitRoute,
          EvalScheme kEvalScheme = kDefaultEvalScheme,
          BoysBudget kEngineBudget = BoysBudget::kFloat,
          PackAxis kPackedAxis = kDefaultPackAxis,
          FitGranularity kFitGranularity = kDefaultFitGranularity>
struct EvalPolicy {
    /// The fit this policy evaluates, where the combination is one the library
    /// carries.
    using Fit = typename detail::RouteFit<kFitRoute, kEvalScheme, kFitGranularity>::Type;

    /// The fit route. This is the fit route, not the multiply-add route a
    /// BackendInfo carries: the two are different selections of different
    /// things.
    static constexpr FitRoute kRoute = kFitRoute;

    /// The scheme the fit's coefficients are summed in.
    static constexpr EvalScheme kScheme = kEvalScheme;

    /// The budget a single-precision engine runs at.
    static constexpr BoysBudget kBudget = kEngineBudget;

    /// The axis a packed evaluation vectorises over.
    static constexpr PackAxis kPack = kPackedAxis;
    /// How narrowly the fitted domain is cut into pieces.
    static constexpr FitGranularity kGranularity = kFitGranularity;
};

/// The constraint the entries put on a policy, so a combination the library
/// does not carry is rejected where the caller names it.
///
/// It requires the four axes and the fit the first two join to. The fit's own
/// contract (FitPolicy) is asserted where the fit is read rather than here,
/// because a fit the library carries is complete only where its tables are.
///
/// \ingroup boys
template <typename P>
concept EvalPolicyLike = requires {
    typename P::Fit;
    { P::kRoute } -> std::convertible_to<FitRoute>;
    { P::kScheme } -> std::convertible_to<EvalScheme>;
    { P::kBudget } -> std::convertible_to<BoysBudget>;
    { P::kPack } -> std::convertible_to<PackAxis>;
    { P::kGranularity } -> std::convertible_to<FitGranularity>;
};

/// The evaluation policy a caller gets by naming no axis, one name per
/// precision: the shortcut for a consumer who has chosen a precision and does
/// not want to choose anything else.
///
/// Each name is what the entries of that precision run when the call site
/// names no policy, so a caller may write \c BoysSingle<1.0, DefaultPolicyFp64>
/// and get the call a caller who named nothing gets — the same instantiation,
/// not a second one that happens to agree, and the entries' own template
/// defaults are these names. The four differ in one field and in one only:
///
///  - the **double** lanes read no budget, so \c DefaultPolicyFp64 selects the
///    shipped route, the shipped scheme, the shipped partition and the
///    arguments-packing axis, and the budget its policy carries is inert;
///  - the **float** lane reads the budget at a relaxed multiplier, and its own
///    is \c BoysBudget::kFloat;
///  - the **half** lanes (\c fp16 and \c bf16) are the same engine under the
///    tighter \c BoysBudget::kFp16 budget, which is the axis that makes their
///    bound 1e-7 rather than the float lane's 1.5e-7 — this is the one place
///    the four names differ in more than their spelling, and it is why a
///    single default for every precision would be the float lane's budget
///    imposed on the half lanes;
///  - \c DefaultPolicyFp16 and \c DefaultPolicyBf16 denote one and the same
///    policy type, because fp16 and bf16 are one lane at one budget; they are
///    named twice so that a document can cite the default for the format its
///    reader is using, and a reader comparing the two names is comparing a
///    lane, not a choice.
///
/// **These are the shipped settings and not a measurement.** No default here
/// was chosen against a timing: which option is fastest is a property of the
/// host, its flags and its card, and this library answers that question with
/// the option probe a consumer runs where they deploy rather than with a
/// recommendation. The names exist so that the shipping default is a thing a
/// caller can point at, cite and change in one line, and setting one of them
/// from a measurement is that line alone. What each name selects is stated
/// lane by lane in docs/lane-contract.md.
///
/// \ingroup boys
using DefaultPolicyFp64 = EvalPolicy<>;

/// The float lane's default policy. See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyFp32 = EvalPolicy<>;

/// The fp16 lane's default policy. See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyFp16 = EvalPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>;

/// The bf16 lane's default policy: the fp16 lane's, because the two formats are
/// one lane at one budget. See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyBf16 = DefaultPolicyFp16;

namespace backend {

/// Which multiply-add a lane's kernels are built from.
///
/// The two are different arithmetics, not two spellings of one: the fused step
/// rounds once and the separate step rounds twice, so the same kernel delivers
/// a different value under each and a bound certified for one does not carry
/// to the other.
enum class MulAddRoute : std::uint8_t {
    /// One rounding: the product is exact and the sum rounds once. An
    /// instruction where the target has one, a call into the C runtime
    /// otherwise.
    kFused,

    /// Two roundings: the product rounds, then the sum rounds. No call on any
    /// target, and a different value from the fused step.
    kSeparate,
};

/// The name a report prints a route under.
///
/// \param route the route
///
/// \returns a string literal naming it
const char* MulAddRouteName(MulAddRoute route) noexcept;

/// The arithmetic one lane runs in: a value type, a storage type, the packed
/// width the kernel operates at, the transport between them, the multiply-add
/// route, and the multiply-adds the kernels are built from.
///
/// The storage type is the one the lane's arguments and results are held in
/// and equals the value type except in the lanes that keep half-precision
/// arguments and compute in single (f16.hpp), where the same arithmetic is
/// reached through a narrower transport.
template <typename B>
concept ArithmeticBackend = requires(const typename B::Storage* storage,
                                     typename B::Storage* destination,
                                     typename B::Packed packed,
                                     typename B::Value scalar) {
    typename B::Value;
    typename B::Storage;
    typename B::Packed;
    { B::kWidth } -> std::convertible_to<std::size_t>;
    { B::kName } -> std::convertible_to<const char*>;
    { B::kRoute } -> std::convertible_to<MulAddRoute>;
    { B::Load(storage) } -> std::same_as<typename B::Packed>;
    { B::Store(destination, packed) } -> std::same_as<void>;
    { B::Broadcast(scalar) } -> std::same_as<typename B::Packed>;
    { B::Mul(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Add(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Sub(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::MulAdd(packed, packed, packed) } -> std::same_as<typename B::Packed>;
    { B::MulSub(packed, packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Contracts() } -> std::same_as<bool>;
};

inline const char* MulAddRouteName(MulAddRoute route) noexcept {
    switch (route)
    {
    case MulAddRoute::kFused:
        return "fused";
    case MulAddRoute::kSeparate:
        return "separate";
    }

    return "unknown";
}

namespace detail {

/// The fused multiply-add of one scalar type, spelled for that type: the
/// single-precision form is the single-precision operation, not the
/// double-precision one narrowed afterwards.
///
/// \param a the multiplicand
/// \param b the multiplier
/// \param c the addend
///
/// \returns `a * b + c` with the product exact and the sum rounded once
template <typename T>
T Fused(T a, T b, T c) noexcept {
    if constexpr (std::is_same_v<T, float>)
    {
        return std::fmaf(a, b, c);
    } else
    {
        return std::fma(a, b, c);
    }
}

/// The separate multiply-add: the product rounds, then the sum rounds.
///
/// Written bare, which costs no call on any target. On a target that contracts
/// a bare product-plus-add it is the fused step instead of two roundings, so a
/// route that means two roundings has to know its build; Contracts() is that
/// question's answer and RouteInForce() is where the two are put together.
///
/// \param a the multiplicand
/// \param b the multiplier
/// \param c the addend
///
/// \returns `a * b + c` with two roundings, where the build does not contract
template <typename T>
T Separate(T a, T b, T c) noexcept {
    return a * b + c;
}

/// The route this translation unit's scalar arithmetic was compiled with.
///
/// A build fact, set once for the library and inherited by a consumer through
/// the same public compile definition, so a call site that instantiates a
/// kernel in its own unit gets the arithmetic the library reports. The default
/// is the fused route: the certified bounds are the fused arithmetic's, and a
/// build that changes route changes them.
#if defined(BOYS_MULADD_SEPARATE)
inline constexpr MulAddRoute kSelectedRoute = MulAddRoute::kSeparate;
#else
inline constexpr MulAddRoute kSelectedRoute = MulAddRoute::kFused;
#endif

/// Whether the build contracts a bare product-plus-add in the scalar
/// arithmetic of `T`. Declared here, defined below, because the route in force
/// is asked of it.
///
/// \returns whether the bare step and the fused step agree here
template <typename T>
bool MeasureContraction() noexcept;

/// The route in force for one precision.
///
/// The selected route, unless the build contracts the bare form: a contracting
/// build compiles both routes to the fused arithmetic, so the route in force is
/// the fused one whatever the selection says, and reporting the selection would
/// describe an arithmetic the lane does not run.
///
/// \returns the route the arithmetic of `T` is actually delivered by
template <typename T>
MulAddRoute RouteInForce() noexcept {
    // The selection is a compile-time constant, so it is tested as one: MSVC
    // reports a constant conditional expression under /W4, and this tree makes
    // that an error. Only a build that selected the separate route asks the
    // contraction question at all.
    if constexpr (kSelectedRoute == MulAddRoute::kSeparate)
    {
        if (MeasureContraction<T>())
        {
            return MulAddRoute::kFused;
        }
    }

    return kSelectedRoute;
}

} // namespace detail

/// The scalar arithmetic of one precision: one value per operation, one
/// rounding per operation, and the fused multiply-add where one rounding is
/// what the kernel wants.
///
/// The scalar backend is also the arithmetic the split-precision lanes run
/// their recurrences in, and the reference the packed backends are held to.
template <typename T>
struct Scalar {
    static_assert(std::is_floating_point_v<T>, "a scalar backend is a floating-point type");

    /// The arithmetic's value type.
    using Value = T;

    /// The type the lane's arguments and results are stored in.
    using Storage = T;

    /// One value per operation: the scalar backend is its own packed form.
    using Packed = T;

    /// Values per packed operand.
    static constexpr std::size_t kWidth = 1;

    /// The name a report prints this arithmetic under.
    static constexpr const char* kName =
        std::is_same_v<T, float> ? "scalar-fp32" : "scalar-fp64";

    /// The multiply-add route this backend was compiled with. The selection,
    /// not the route in force: whether the separate route is really two
    /// roundings is a property of the build's contraction, which is measured
    /// per translation unit, and BoysBackends reports that answer.
    static constexpr MulAddRoute kRoute = detail::kSelectedRoute;

    /// \param p one stored value
    ///
    /// \returns it as a packed operand
    static Packed Load(const Storage* p) noexcept { return *p; }

    /// \param p where the value is written
    /// \param v the value to write
    static void Store(Storage* p, Packed v) noexcept { *p = v; }

    /// \param v the value to replicate
    ///
    /// \returns `v` in the one lane there is
    static Packed Broadcast(Value v) noexcept { return v; }

    /// \param a the multiplicand
    /// \param b the multiplier
    ///
    /// \returns `a * b`, one rounding
    static Packed Mul(Packed a, Packed b) noexcept { return a * b; }

    /// \param a the augend
    /// \param b the addend
    ///
    /// \returns `a + b`, one rounding
    static Packed Add(Packed a, Packed b) noexcept { return a + b; }

    /// \param a the minuend
    /// \param b the subtrahend
    ///
    /// \returns `a - b`, one rounding
    static Packed Sub(Packed a, Packed b) noexcept { return a - b; }

    /// `a * b + c` at this backend's route.
    ///
    /// At the fused route the product is exact and the sum rounds once, which
    /// on a target without the instruction is a call into the C runtime. At the
    /// separate route the product rounds and then the sum rounds: two
    /// roundings, no call, and a different value.
    ///
    /// \param a the multiplicand
    /// \param b the multiplier
    /// \param c the addend
    ///
    /// \returns `a * b + c` at the selected route
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kRoute == MulAddRoute::kFused)
        {
            return detail::Fused(a, b, c);
        } else
        {
            return detail::Separate(a, b, c);
        }
    }

    /// `a * b - c` with two roundings: the product rounds, then the difference
    /// rounds. Written through the fused form with a zero addend rather than
    /// as the bare product and difference, because the bare form is a single
    /// fused step wherever the target has FMA and the contraction setting
    /// allows it — the default on aarch64, and on any x86 build that passes
    /// -mfma — so the same source would be two arithmetics. The zero addend is
    /// the product rounded once and nothing more, leaving a contraction pass
    /// nothing to fuse.
    ///
    /// \param a the multiplicand
    /// \param b the multiplier
    /// \param c the subtrahend
    ///
    /// The route does not reach this operation. Its contract is two roundings
    /// on EVERY build, and the spelling above is the portable way to keep it: a
    /// bare `a * b - c` is two roundings only where the build does not
    /// contract, so spelling it that way would make the operation's own
    /// contract a property of the flags. The zero addend costs one call on a
    /// target without the instruction, and that is the residue the separate
    /// route leaves: a correct price for a contract that does not move.
    ///
    /// \returns `a * b - c` with two roundings
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return detail::Fused(a, b, Packed{0}) - c;
    }

    /// Whether a bare `a * b + c` written in this arithmetic is a single
    /// rounding, as the translation unit that calls this compiles it. The
    /// answer is the asking unit's because contraction follows from that
    /// unit's target and flags; see the file comment.
    ///
    /// \returns whether the bare form and the fused step agree here
    static bool Contracts() noexcept;
};

/// The double-precision scalar arithmetic.
using ScalarFp64 = Scalar<double>;

/// The single-precision scalar arithmetic.
using ScalarFp32 = Scalar<float>;

namespace detail {

/// Whether the build contracts a bare product-plus-add in the scalar
/// arithmetic of `T`.
///
/// Measured, not declared: contraction follows from the target and the flags,
/// so the only answer worth reporting is the one the build gives. The operands
/// make the product inexact and then cancel its leading term, so a bare step
/// that contracts reaches the fused value exactly while one that does not
/// lands 2^-54 away in double and 2^-26 in single. The operands are read
/// through volatile so the compiler evaluates the expression rather than the
/// constant.
///
/// \returns whether the bare step and the fused step agree here
template <typename T>
bool MeasureContraction() noexcept {
    constexpr int kShift = std::numeric_limits<T>::digits / 2 + 1;
    const T unit{1};
    const T small = static_cast<T>(std::ldexp(1.0, -kShift));
    const T half = static_cast<T>(std::ldexp(1.0, -(kShift - 1)));
    volatile const T a = unit + small;
    volatile const T b = unit + small;
    volatile const T c = -(unit + half);
    const T productThenSum = a * b + c;
    const T fused = Fused<T>(a, b, c);
    return productThenSum == fused;
}

} // namespace detail

template <typename T>
bool Scalar<T>::Contracts() noexcept {
    return detail::MeasureContraction<T>();
}

/// One arithmetic backend as a report states it.
struct BackendInfo {
    /// The name a report prints for this arithmetic.
    const char* name;

    /// Whether a bare product-plus-add in this arithmetic is a single rounding
    /// in this build.
    bool contracts;

    /// The multiply-add route in force for this arithmetic, which is the route
    /// its values were computed at and not merely the one the build selected.
    /// The two differ where a contracting build was asked for the separate
    /// route, because there the bare form IS the fused step and the arithmetic
    /// is the fused one.
    MulAddRoute route;
};

/// The arithmetic backends this build carries.
///
/// The scalar pair is always present. The packed pair is present exactly when
/// the build has the AVX2 + FMA tier, which is decided at run time on an
/// x86_64 build and absent entirely elsewhere.
///
/// \returns the backends, in a fixed order.
///
/// \ingroup boys
std::span<const BackendInfo> BoysBackends() noexcept;

} // namespace backend
} // namespace boys
