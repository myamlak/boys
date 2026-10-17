#pragma once

/// \file policy.hpp
/// The evaluation policy: the axes a call site selects together, the constraint
/// that rejects a combination this library does not carry, the point a build's
/// seam names, and each axis renamed over a base so one cell goes in and the rest
/// are the base's.
///
/// A policy is a type the compiler selects for a call, with no registry, string
/// key, function pointer or branch in it, so this header names only enumerations
/// and type aliases and includes no `<cmath>`: that is what keeps the front door
/// (`boys/boys.hpp`) cheap for a call site that names no policy at all.
///
/// One cell of an evaluation policy replaced, every other cell derived from a base
/// rather than written down.
///
/// \c SeamPolicy is seven positional template parameters and a parameter list has no
/// holes: a caller who wants the last of them replaced writes the six before it, and
/// those six are then the values that caller wrote and not the build's. The aliases
/// below take the cell the caller has an opinion about and a policy to derive the rest
/// from, so the cells a caller does not name are the base's own and not a copy of them:
///
///     // the build's own point, with one cell named
///     using P = boys::WithDivisionForm<boys::DivisionForm::kPlainReciprocal>;
///
///     // a class's own row of the default-policy table, with one cell named
///     using Q = boys::WithDivisionForm<boys::DivisionForm::kPlainReciprocal,
///                                      boys::DefaultPolicy<boys::Precision::kFp64,
///                                                          boys::Shape::kAllOrders>>;
///
/// **Where a caller names no base, the base is \c SeamPolicy<>, which is the build's
/// own point and not a class's row.** \c SeamPolicy<> is the seven the build's seam
/// header (`boys/boys_build_defaults.hpp`) names; what an entry compiles when its call
/// site names no policy is \c DefaultPolicy<Precision, Shape>, the class's own row, and
/// the cells of that row need not be the seam's. A caller who wants that row with one
/// cell replaced names it as the base, as \c Q does above; a caller who names none gets
/// the point \c SeamPolicy<> is. Neither form writes a cell its caller did not choose.
///
/// Each alias is the type its long spelling is, so a call through one is compiled as
/// the call through the other, to the last bit.
///
/// \ingroup boys

#include "boys/accuracy.hpp"

#include <concepts>
#include <cstdint>

namespace boys {

/// The region-A partition a fit reads: which piece of the region an argument falls
/// in and what that piece is.
///
/// A fit names its partition rather than carrying a lookup of its own, because the
/// lookup is the same on every partition - scan the order's pieces for the first
/// whose right edge is past the argument - and only the table differs.
///
/// \ingroup boys
template <typename P>
concept RegionAPartition = requires(std::size_t index, int order, double x) {
    { P::PieceIndex(order, x) } -> std::convertible_to<std::size_t>;
    { P::PieceAt(index) };
};

/// The fit one region-A route evaluates: its coefficient set and the scheme the
/// coefficients are read in, named as a type so the evaluation body around it is
/// written once and instantiated per route. The body is in boys_impl.hpp, and each
/// model of this concept sits with the coefficient tables it reads.
///
/// The body owns everything that is not the fit - the zero argument's closed form,
/// the region split, the piece lookup, the mapped argument, the recurrences, the
/// per-order rule and the domains - so a route cannot drift from the lane around
/// it, and asks a policy for four things:
///
///  - `Partition` is the region-A partition the fit's pieces are stored in, as a
///    model of RegionAPartition;
///  - `EvalPiece(index, t)` is one region-A piece's value at its own mapped
///    argument `t`, named by the piece's index in that partition;
///  - `RegionBSeed(x)` is region B's seed F_0(x);
///  - `BandSource` reads the orders the band's regime covers. A source is
///    constructed at an argument and stepped order by order, so a batch pays one
///    step per order rather than one whole run of the recurrence per order. The
///    requirement spells the type `typename`: a dependent qualified name is
///    otherwise read as a value, and a compiler that does read it that way rejects
///    the policy.
///
/// `kRegionAFitsFrom` is the argument at and above which the policy's own answer
/// reads those orders.
template <typename F>
concept FitPolicy = requires(std::size_t index, double t, double x, int l) {
    { F::kRegionAFitsFrom } -> std::convertible_to<double>;
    { F::EvalPiece(index, t) } -> std::same_as<double>;
    { F::RegionBSeed(x) } -> std::same_as<double>;
    { typename F::template BandSource<kDefaultDivisionForm>(x).Next(l, x) } -> std::same_as<double>;
} && RegionAPartition<typename F::Partition>;

/// The fit one (route, scheme) pair evaluates: the join between the two axes.
///
/// The route names the fits and the scheme names the summation those fits'
/// coefficients are read in where they have two stored forms. A route whose own fit
/// has one form is the same fit under either scheme, and the scheme still reaches
/// the parts of a call that route's fits do not serve - so every pair of the two
/// enumerations is carried, and the only value rejected here is one outside the
/// route enumeration.
namespace detail {

template <EvalScheme kScheme, FitGranularity kGranularity>
struct ChebyshevFit;

/// The uniform table's fit; see boys_impl.hpp. A family of its own rather than a
/// third branch of ChebyshevFit, because it is read at a partition that is not a
/// cut of either derived one: the grid is fixed and its table is laid out
/// interval-major, so every branch that asks a Chebyshev fit to choose between two
/// derived partitions has no third answer to give it.
template <EvalScheme kScheme>
struct UniformFit;

/// The rational member over the uniform grid; defined in boys_impl.hpp beside the
/// table it reads.
///
/// The rational family fits a numerator/denominator pair over an interval, and a
/// partition's pieces are what it cuts, so the fixed grid is a partition its fits
/// are cut over like any other. It is one fit under either scheme: its coefficients
/// are a monomial numerator and denominator with no Chebyshev form to sum.
///
/// **What it hands a caller.** One pair per interval of the grid the Chebyshev
/// member is read at, fitted over that interval's own cell and read at the mapped
/// argument the grid's own locator builds for it - \c FlatPoint's \c t,
/// `2 (x * kFlatPerUnit - iv) - 1`, the same double both members are read at, so one
/// lookup addresses them. The numerator's coefficients are stored first and the
/// denominator's `q_1..q_k` after them with its constant term held at 1, so a caller
/// sums the numerator by Horner, sums the denominator, and divides once.
///
/// **The degree is the pair's own, and the table has no single one.** An interval
/// stores `m + 1 + k` doubles for each of the `kMaxBoysOrder + 1` orders, with `m`
/// and `k` its own and read off the table's per-interval degree columns; its rows
/// start at its own offset and step by its own stored count. A caller therefore
/// reaches an interval's rows through the interval, never through one stride for the
/// whole member. The cover is the grid's - the intervals the locator spans, and
/// nothing above them.
///
/// The four arrays that carry the layout are the emitter's: a per-interval numerator
/// degree, denominator degree, stored count and offset beside the coefficients.
///
/// **It models FitPolicy, and the members the grid does not serve refuse.**
/// \c Partition names the same \c RegionAPartition model \c UniformFit names.
/// \c EvalPiece, \c RegionBSeed and \c BandSource answer with a value no route can
/// produce rather than with another partition's fits, exactly as \c UniformFit's do:
/// they complete the contract, and a body that reaches one refuses the partition
/// where it is named.
struct RationalFitUniform;

struct RationalFit;

/// The rational family over the narrow partition; see the specialization below.
struct RationalFitNarrow;

/// A route outside the FitRoute enumeration: not a selection this library can
/// answer, and rejected where the caller names it rather than quietly evaluated at
/// the default. The route entry takes the other reading - a route value outside
/// the enumeration is the default there - because the value can arrive from a
/// report's rows rather than from a claim the caller makes.
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

/// The uniform route: a Chebyshev family read at a fixed grid rather than a derived
/// partition, so both schemes are defined for it - it holds both stored forms of its
/// coefficients - and it is selected by the partition axis.
///
/// The rational route's member over the same grid is a family of its own and not a
/// mode of this one: the two store different things.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kChebyshev, kScheme, FitGranularity::kUniform> {
    /// The uniform table's fit.
    using Type = UniformFit<kScheme>;
};

/// The rational family at the uniform partition: the grid's cells are intervals like
/// any other partition's pieces, and this member is one numerator/denominator pair
/// fitted over each of them, in the same stored form the shipped and narrow members
/// use, and one fit under either scheme for the same reason.
///
/// This specialization is what makes that route reachable over the grid: without it
/// the primary template above would answer a caller naming this route over the grid
/// with the Chebyshev member, at a certified bound, under the rational route's name,
/// with nothing reporting it.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kUniform> {
    /// The single rational member over the grid, one pair per interval.
    using Type = RationalFitUniform;
};

/// The rational family: one fit under either scheme, because its coefficients
/// are a monomial numerator and denominator with no Chebyshev form to sum.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kCoarsest> {
    /// The single rational fit, shared by both schemes.
    using Type = RationalFit;
};

/// The rational route under the narrow partition: the same family fitted over the
/// narrow pieces' own intervals, one numerator/denominator pair per piece in both
/// regions, in the same stored form and read at the same mapped arguments as the
/// shipped member. A partition is a cut of the region and not a property of a
/// family, so what this member carries is the pairs.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kNarrow> {
    /// The single rational fit, one pair per narrow piece.
    using Type = RationalFitNarrow;
};

} // namespace detail

/// Everything a call site selects about how an evaluation is performed: the fit
/// route, the scheme the fit's coefficients are summed in, the budget a
/// single-precision engine runs at, the axis a packed evaluation vectorises over,
/// how narrowly the fitted domain is cut into pieces, how a ladder's steps divide,
/// and which exponential seeds a region-B ladder.
///
/// The axes are selected together because they are one evaluation, not because
/// either implies the other: each names a different thing, and the pair of the first
/// two names the fit the bodies evaluate. \c Fit is where that join happens. The
/// packing axis is not part of that join - it names which axis a call's values are
/// spread over, so it changes nothing about which fit answers.
///
/// A combination the build cannot serve - a member of a partition whose fits it does
/// not hold, or a degree table it has not derived - is refused where it is named
/// rather than at a kernel, because a fallback would return one partition's values
/// under another's name. The single-precision lanes are not an exception: their
/// narrow partition's tables are derived per partition at compile time like the
/// double lane's.
///
/// **Every axis is stated, and no parameter of this template carries a default.** A
/// spelling that omits a cell is therefore a compile error here, whatever wrote it:
/// this is the name the default-policy table's rows expand, so a row with a cell
/// missing fails at the class rather than at the macro that expanded it. The
/// defaulted spelling is \c SeamPolicy below.
///
/// \tparam kFitRoute      the fit route
/// \tparam kEvalScheme    the scheme the fit's coefficients are summed in
/// \tparam kEngineBudget  the computation budget of a single-precision engine,
///                        read by the float and half-precision lanes and by
///                        nothing else
/// \tparam kPackedAxis    the axis a packed evaluation vectorises over, read by
///                        the entries that have a wide axis to fill
/// \tparam kGranularity   how narrowly the fitted domain is cut into pieces
/// \tparam kDivision      how the recursion's per-order division is performed
/// \tparam kExp           which exponential seeds a region-B ladder; the host
///                        default is \c kDefaultHostRegionBExp, which is
///                        \c RegionBExp::kFast, the arithmetic the double
///                        lane's per-region figures were measured at
///
/// \ingroup boys
template <FitRoute kFitRoute,
          EvalScheme kEvalScheme,
          BoysBudget kEngineBudget,
          PackAxis kPackedAxis,
          FitGranularity kFitGranularity,
          DivisionForm kDivisionForm,
          RegionBExp kExp>
struct StatedEvalPolicy {
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
    /// How the recursion's per-order division is performed.
    static constexpr DivisionForm kDivision = kDivisionForm;
    /// Which exponential seeds a region-B ladder.
    static constexpr RegionBExp kRegionBExp = kExp;
};

/// The policy a caller spells out explicitly: every axis of \c StatedEvalPolicy,
/// each parameter defaulted to the name the build's seam header
/// (`boys/boys_build_defaults.hpp`) carries, so a caller may name the one axis it
/// has an opinion about and keep the rest.
///
/// **This is not "the default".** An entry's template default is its class's row,
/// \c DefaultPolicy<kPrecision, kShape>, and the cells of that row need not be the
/// seam's. This alias is the seam's own point: the seven names the header states,
/// which a class's row may hold or differ from. A caller who wants this build's
/// point writes \c SeamPolicy<>; a caller who wants the class's row names no
/// policy at the call site.
///
/// \ingroup boys
template <FitRoute kFitRoute = kDefaultFitRoute,
          EvalScheme kEvalScheme = kDefaultEvalScheme,
          BoysBudget kEngineBudget = BoysBudget::kFloat,
          PackAxis kPackedAxis = kDefaultPackAxis,
          FitGranularity kFitGranularity = kDefaultFitGranularity,
          DivisionForm kDivisionForm = kDefaultDivisionForm,
          RegionBExp kExp = kDefaultHostRegionBExp>
using SeamPolicy = StatedEvalPolicy<kFitRoute, kEvalScheme, kEngineBudget, kPackedAxis, kFitGranularity,
                                    kDivisionForm, kExp>;

/// The constraint the entries put on a policy, so a combination the library does not
/// carry is rejected where the caller names it.
///
/// It requires every axis and the fit the first two join to. The fit's own contract
/// (FitPolicy) is asserted where the fit is read rather than here, because a fit the
/// library carries is complete only where its tables are.
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
    { P::kDivision } -> std::convertible_to<DivisionForm>;
    { P::kRegionBExp } -> std::convertible_to<RegionBExp>;
};

/// The evaluation policy a caller gets by naming no axis, one name per precision:
/// the shortcut for a consumer who has chosen a precision and does not want to
/// choose anything else.
///
/// Each name is \c SeamPolicy<> at one lane's budget: the point the build's seam
/// header names, and never a class's row. A caller may write
/// \c BoysSingle<DefaultPolicyFp64>(n, x) to name that point explicitly, and it is
/// the call a call site that named no policy gets exactly where the table's row for
/// that class carries the combination this point composes - not the same
/// instantiation in general. The four differ in the budget alone:
///
///  - the **double** lanes read no budget, so \c DefaultPolicyFp64 selects whatever
///    \c kDefaultFitRoute, \c kDefaultEvalScheme, \c kDefaultFitGranularity,
///    \c kDefaultPackAxis and \c kDefaultDivisionForm name at the revision a
///    caller builds against, and its budget cell is inert;
///  - the **float** lane reads the budget, and its own is \c BoysBudget::kFloat;
///  - the **half** lanes (fp16 and bf16) are one engine under the tighter
///    \c BoysBudget::kFp16 budget, which cuts their fits for a 1e-7 region target
///    rather than the float lane's. That is not an accuracy they can publish: they
///    run the float lane's arithmetic and store what it returns, so their published
///    figure is the fp32 engine's arithmetic stated absolutely, 1.5e-7, plus the half-ULP term the store adds - 2^-11 for
///    the binary16 store, 2^-8 for the bfloat16 one;
///  - \c DefaultPolicyFp16 and \c DefaultPolicyBf16 denote one and the same policy
///    type - one engine, one budget, and no cell of a policy is a format - and
///    neither is a class's default. What an entry that names no policy resolves to is
///    \c DefaultPolicy<Precision, Shape>, one row per class, and the fp16 and bf16
///    classes are two of those rows. These two names are here so that a document can
///    cite the lane it is writing about.
///
/// **Which option is fastest is a property of the host, its flags and its card.** The
/// figures behind the two measured axes were read off the option probe this library
/// ships for a consumer to run where they deploy, and the axes not measured against
/// it carry the shipped settings. Which axis is which is stated in
/// docs/lane-contract.md; the numbers are the probe's own report, on the machine it
/// is run on, and not a figure this header can restate.
///
/// \ingroup boys
using DefaultPolicyFp64 = SeamPolicy<>;

/// The float lane's default policy. See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyFp32 = SeamPolicy<>;

/// The fp16 lane's default policy: the seam's five at the half lane's budget. A
/// class's own default is \c DefaultPolicy<Precision::kFp16, Shape>, which the
/// table may state with other cells. See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyFp16 = SeamPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>;

/// The bf16 lane's default policy: the seam's five at the same half budget, and the
/// same policy type as \c DefaultPolicyFp16, because the two half formats are one
/// engine and a policy carries no format cell. The classes are two: an unnamed bf16
/// call resolves through \c DefaultPolicy<Precision::kBf16, Shape>, which is this
/// lane's own row, so this name is the lane's five and not the fp16 class's answer.
/// See \c DefaultPolicyFp64.
///
/// \ingroup boys
using DefaultPolicyBf16 = SeamPolicy<kDefaultFitRoute, kDefaultEvalScheme, BoysBudget::kFp16>;


/// The fit route replaced: the base's policy, with \c kRoute taken from the argument
/// and every other cell left as the base has it.
///
/// The route names which fits are evaluated, and it is read together with the scheme
/// and the granularity: a combination the library does not carry is refused where the
/// caller names it, at the fit the three select.
///
/// \tparam kFitRoute the route the derived policy evaluates
/// \tparam Base      the policy to derive from; \c SeamPolicy<>, the build's own
///                   point, where the caller names none
///
/// \ingroup boys
template <FitRoute kFitRoute, EvalPolicyLike Base = SeamPolicy<>>
using WithFitRoute = StatedEvalPolicy<kFitRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                      Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The evaluation scheme replaced: the base's policy, with \c kScheme taken from the
/// argument and every other cell left as the base has it.
///
/// The scheme is how one stored fit is summed, so it changes the arithmetic of every
/// evaluation and not the fit that is evaluated.
///
/// \tparam kEvalScheme the scheme the derived policy sums its fits in
/// \tparam Base        the policy to derive from; \c SeamPolicy<>, the build's own
///                     point, where the caller names none
///
/// \ingroup boys
template <EvalScheme kEvalScheme, EvalPolicyLike Base = SeamPolicy<>>
using WithEvalScheme = StatedEvalPolicy<Base::kRoute, kEvalScheme, Base::kBudget, Base::kPack,
                                        Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The engine budget replaced: the base's policy, with \c kBudget taken from the
/// argument and every other cell left as the base has it.
///
/// The budget is read by the single-precision and half-precision engines and by no
/// other lane, so on a double-precision policy this cell is inert: two policies that
/// differ in it alone compile the same call.
///
/// \tparam kEngineBudget the budget a single-precision engine runs at
/// \tparam Base          the policy to derive from; \c SeamPolicy<>, the build's own
///                       point, where the caller names none
///
/// \ingroup boys
template <BoysBudget kEngineBudget, EvalPolicyLike Base = SeamPolicy<>>
using WithBudget = StatedEvalPolicy<Base::kRoute, Base::kScheme, kEngineBudget, Base::kPack,
                                    Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The packing axis replaced: the base's policy, with \c kPack taken from the argument
/// and every other cell left as the base has it.
///
/// The axis is a property of the shape a call has: an entry that produces one order
/// has no second order to pack, so the axis an entry reads is the one its own packed
/// lane fills.
///
/// \tparam kPackedAxis the axis a packed evaluation vectorises over
/// \tparam Base        the policy to derive from; \c SeamPolicy<>, the build's own
///                     point, where the caller names none
///
/// \ingroup boys
template <PackAxis kPackedAxis, EvalPolicyLike Base = SeamPolicy<>>
using WithPackAxis = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, kPackedAxis,
                                      Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The fit granularity replaced: the base's policy, with \c kGranularity taken from the
/// argument and every other cell left as the base has it.
///
/// The granularity says how narrowly the fitted domain is cut into pieces, and it
/// selects tables: a piece of another width is another fit at another degree, so this
/// cell moves the stored coefficients a call reads and not only its arithmetic.
///
/// \tparam kFitGranularity how narrowly the derived policy cuts the fitted domain
/// \tparam Base            the policy to derive from; \c SeamPolicy<>, the build's own
///                         point, where the caller names none
///
/// \ingroup boys
template <FitGranularity kFitGranularity, EvalPolicyLike Base = SeamPolicy<>>
using WithFitGranularity = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                            kFitGranularity, Base::kDivision, Base::kRegionBExp>;

/// The division form replaced: the base's policy, with \c kDivision taken from the
/// argument and every other cell left as the base has it.
///
/// The form is how the recursions' per-order division is performed. The three are three
/// arithmetics of a chain and not three spellings of one, and the two reciprocal forms
/// differ from each other in the last bits they deliver as well as in what they cost.
///
/// \tparam kDivisionForm how the derived policy performs the recursion's divisions
/// \tparam Base          the policy to derive from; \c SeamPolicy<>, the build's own
///                       point, where the caller names none
///
/// \ingroup boys
template <DivisionForm kDivisionForm, EvalPolicyLike Base = SeamPolicy<>>
using WithDivisionForm = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                          Base::kGranularity, kDivisionForm, Base::kRegionBExp>;

/// The region-B exponential replaced: the base's policy, with \c kRegionBExp taken from
/// the argument and every other cell left as the base has it.
///
/// The cell selects which exponential seeds a region-B ladder, so it changes the values
/// region B returns and nothing an earlier region returns.
///
/// \tparam kExp which exponential the derived policy's region-B ladder is seeded with
/// \tparam Base the policy to derive from; \c SeamPolicy<>, the build's own point, where
///              the caller names none
///
/// \ingroup boys
template <RegionBExp kExp, EvalPolicyLike Base = SeamPolicy<>>
using WithRegionBExp = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                        Base::kGranularity, Base::kDivision, kExp>;

}  // namespace boys
