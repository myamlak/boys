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
/// Two multiply-adds are named, because they are two arithmetics:
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

#include "boys/accuracy.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace boys {

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
    /// Four arguments of one order: the packed lanes the fixed-order and plane
    /// entries reach.
    kArguments = 0,

    /// Four orders of one argument: the shape \c BoysAllOrders has, and the
    /// axis its own packed lane packs.
    kOrders = 1,
};

/// The axis the entries pack when the caller names none.
///
/// The shipped one: a call site that names no axis packs the arguments axis.
inline constexpr PackAxis kDefaultPackAxis = PackAxis::kArguments;

/// The name a report prints a packing axis under, and never null.
///
/// \param axis the axis
///
/// \returns a string literal naming it: "arguments" or "orders", and "unknown"
///          for a value outside the enumerators
///
/// A value outside the enumerators - cast in from outside the enum, or named by
/// a newer header - is answered rather than refused: a caller that has to tell
/// "this build does not carry that axis" from "that is not an axis at all" tests
/// the value against the enumerators itself, because both arrive here as
/// "unknown".
const char* PackAxisName(PackAxis axis) noexcept;

/// How the recursion's per-order division is performed.
///
/// A ladder step divides in region B and in region C, and each of those ends in
/// one of these. They are different arithmetic and not three spellings of one: a
/// quotient is correctly rounded, and a product by a rounded reciprocal rounds
/// twice, so the plain form may differ from the exact one by an ulp per step and
/// a ladder of them accumulates that; the refined form is the plain one carried
/// back to the exact one's rounding by a fused multiply-add, at the price of two
/// dependent operations per step.
///
/// **Which divisions a form governs.** A step divides either by the argument or
/// by the step's constant `l + 1/2`, and a form governs both. The argument's
/// reciprocal is formed once per call and multiplied through the ladder, which is
/// the division the caller named the axis for. The constant's is the downward
/// ladder's, and its reciprocal is a compile-time table rather than a value
/// formed at the step: `l + 1/2` is exact, so a reciprocal computed from `l` would
/// be a division there, and a form whose whole point is not to divide would
/// divide once per order anyway. The divisions that are not a step of a chain are
/// outside the axis — the zero argument's `1/(2n+1)`, a piece's affine map — and
/// not because a form was declined for them: they are closed formulas rather than
/// a recurrence, no form shortens them, and the exact form is what they are.
///
/// **The single-precision lane's downward ladder is the one division outside it.**
/// That lane keeps exact division there whichever form is named. The reason is a
/// measurement and not a preference: the lane publishes one figure for every form
/// — the accuracy accessor has no form-keyed bound — and over the accuracy gate's
/// own reference grid at the reference multiplier the plain form's reciprocal at
/// that step would take the ladder outside that figure at two cells, both at
/// order 0: x = 9.74054909, where it delivers 1.7514e-07 against the lane's
/// 1.5e-7, and x = 7, where it delivers 1.5547e-07. Both cells are that form's
/// alone - at them the lane's exact and refined forms deliver 3.6736e-09 and
/// 3.626e-08, and their own worst over the grid is 1.0835e-07, at n = 0,
/// x = 11.1509647, inside the figure. So the form is not available on that
/// ladder rather than served outside the figure it publishes, and a caller
/// naming it there is answered by exact division; the figure has to gain a form
/// dimension before the form can be. Every other division on that lane takes the
/// form, and the double lane's downward ladder takes it with room to spare: the
/// plain form's worst cell over the grid's arguments below kX0 is 6.7292e-15
/// where exact division's is 3.2162e-15, and that cell lies in the band, which
/// publishes 3e-14.
///
/// **Which of the three is cheapest is a property of the host and not of this
/// library.** A processor whose division is a multi-instruction sequence pays
/// far more for the exact form than for either product; one whose fused
/// multiply-add is scarce relative to its multiplier pays more for the refined
/// form than the plain one. The three are therefore all carried, and ranked by
/// the option probe on the machine it is run on, rather than one being chosen
/// here on the evidence of a laptop. The measured figures for this host are in
/// the probe's own report.
///
/// \ingroup boys
enum class DivisionForm : std::uint8_t {
    /// One division per step: the form the recurrences are written in.
    kExactDivision = 0,

    /// One reciprocal per divisor and one product per step: the argument's
    /// reciprocal, formed once and multiplied through the ladder, and the step
    /// constant's, read from a compile-time table and multiplied the same way.
    /// Not the single-precision lane's downward ladder; see above.
    kPlainReciprocal = 1,

    /// The plain form with the correctly rounded quotient recovered from it, by
    /// the product's error and one fused multiply-add per step.
    kRefinedReciprocal = 2,
};

/// The division form the recurrences take when the caller names none.
///
/// **Measured.** At the per-call entry, over eight interleaved rounds whose
/// fixed-work canary held to 6%, both reciprocal forms came out about a quarter
/// cheaper than exact division on the molecular stream: 0.751 and 0.742 of its
/// cost, the two within 1% of each other against a spread wider than that.
///
/// **This one, because it is the reciprocal that moves nothing.** The refinement
/// recovers the correctly rounded quotient from the product with one fused
/// multiply-add per step, so this form is bit-identical to exact division: it
/// delivers the values every published per-region figure was measured at, at the
/// cost the plain form pays where the real workload lives. The plain form is the
/// one to reach for when ladders are long — on the host's own dependent chain it
/// reads 3.220 ns per step against 5.333 here, the difference being the
/// refinement's two fused multiply-adds paid once per order. All three forms are
/// served, and the option probe measures them.
inline constexpr DivisionForm kDefaultDivisionForm = DivisionForm::kRefinedReciprocal;

/// The name a report prints a division form under, and never null.
///
/// \param form the form
///
/// \returns a string literal naming it: "exact-division", "plain-reciprocal" or
///          "refined-reciprocal", and "unknown" for a value outside the
///          enumerators
///
/// A value outside the enumerators is answered rather than refused, on the same
/// reading as \c PackAxisName.
const char* DivisionFormName(DivisionForm form) noexcept;

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
/// Each partition is whole rather than a point on a spectrum. It carries its
/// own stored counts and its own certified bound, and none is a rung of
/// another: naming one changes the fits that serve the intervals its own report
/// names.
///
/// **Narrowing is a trade and not a saving.** A narrower piece is fewer
/// coefficients to read per evaluation and more pieces to store and to look up
/// in, so the axis trades per-evaluation work against table size. A consumer
/// whose cost is per evaluation gains; one whose cost is the table gains
/// nothing and pays the lookup.
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
/// Both routes and both precisions carry the member: the rational family is
/// fitted over the narrow pieces as its own per-piece pairs, and the
/// single-precision lanes carry their own narrow tables, each accepted in the
/// arithmetic its lane runs at both multiply-add routes.
///
/// A member the build cannot serve is refused where it is named, with the
/// reason, rather than answered from the shipped tables: the partitions'
/// coefficients are different fits of the same function over the same
/// interval, so a silent substitution would return one partition's values
/// under another's name, at a certified bound, with nothing reporting it.
///
/// Every refusal names the work it would need, and none of them is a
/// combination that cannot exist - each is a body, a table or a fit not yet
/// written. What a lane refuses is a member of a partition it has not stored or a
/// rung its tables do not admit, and each is refused where the call names it
/// rather than answered from another partition's fits.
///
/// The relaxed rungs and the across-orders packing axis are otherwise built: a
/// rung of the narrow partition is derived against its own pieces rather than
/// truncated from the shipped rows, and the packed lane reaches a per-order cut
/// with a gathered fetch, reading each of the four orders it holds its own
/// piece and coefficients instead of stepping one piece's coefficients at a
/// fixed stride. This includes the single-precision lanes, whose narrow rung
/// tables are derived like the double lane's rather than absent.
///
/// \ingroup boys
enum class FitGranularity : std::uint8_t {
    /// The partition the certified lanes are defined by: region A's two bands per
    /// order and region B's single seed, at the degrees the committed tables
    /// carry. A caller names it to read those tables.
    kShipped = 0,

    /// A narrower partition of both fitted regions, at the degrees the proved
    /// truncation bound gives a piece of that width at the bar the piece is read
    /// under. Fewer coefficients per evaluation, more pieces in the table.
    kNarrow = 1,

    /// A fixed grid over the whole fitted domain rather than a derived one: one
    /// uniform interval width, every order fitted independently at one degree,
    /// and no order built from another.
    ///
    /// The other two partitions are walks: each places a piece where the proved
    /// bound says the function needs one, so the pieces are of different widths
    /// and locating one is a scan of piece edges. This one trades that for a
    /// grid whose index is one multiply and a truncation, and it trades the
    /// per-order recursion for independent polynomials. Both are what the
    /// option probe measures; neither is free, and the table pays for them in
    /// stored coefficients and in a floor on the work each order does.
    kUniform = 2,
};

/// The partition the entries evaluate when the caller names none.
///
/// The narrow one: a call site that names no partition reads the narrow pieces'
/// coefficients, and naming another value asks for that partition's. The
/// derived partitions cut the same fits, so choosing between them is a choice
/// of arithmetic rather than of accuracy.
///
/// **It is not the cheaper of the two at either setting of the other axis.** On
/// the host these were last measured on, the narrow partition was 0.3% to 0.8%
/// behind the shipped one where the fit is summed by the split Clenshaw
/// recurrence — inside the 5.6 to 7.4 points one of these rows moves by from one
/// run to the next — while summed by Horner's rule, the scheme this default
/// reads, it was 15% to 17% cheaper. What the measurement establishes is a tie:
/// three rows of the double lane's full-accuracy class for the all-orders shape —
/// the shipped partition summed by the split Clenshaw recurrence, and the narrow
/// partition summed by either scheme — came out within 0.9% of each other, with
/// the class's next row 11.5% to 12% behind them. This default is one of the
/// three tied rows. A caller who wants this ranking on their own machine runs the
/// option probe (boys_probe.hpp), which measures it there and names the pairs it
/// could and could not separate.
inline constexpr FitGranularity kDefaultFitGranularity = FitGranularity::kNarrow;

/// The name a report prints a granularity under, and never null.
///
/// \param granularity the partition
///
/// \returns a string literal naming it: "shipped", "narrow" or "uniform", and
///          "unknown" for a value outside the enumerators
///
/// A value outside the enumerators - cast in from outside the enum, or named by
/// a newer header - is answered rather than refused. This function names a
/// partition; it does not report whether the build serves one, which is
/// FitGranularityHasRoute and FitGranularityHasAxis, so a caller that needs the
/// distinction asks those rather than reading this string.
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

    /// The fp16/bf16 engine's 1e-7 region target. The half lanes run the float
    /// lane's engine with the fits cut for the tighter target, and what they
    /// publish is that lane's own figure, 1.5e-7, plus the half-ULP term their
    /// store adds: a budget is what the cuts are placed against, not a bound a
    /// caller is given.
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
    { typename F::template BandSource<kDefaultDivisionForm>(x).Next(l, x) } -> std::same_as<double>;
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

/// The uniform table's fit; see boys_impl.hpp. A family of its own rather than
/// a third branch of ChebyshevFit, because it is read at a partition that is
/// not a cut of either derived one: the grid is fixed and the table is laid out
/// interval-major, so every branch that asks a Chebyshev fit to choose between
/// two derived partitions has no third answer to give it.
template <EvalScheme kScheme>
struct UniformFit;

/// The rational member over the uniform grid; defined in boys_impl.hpp beside
/// the table it reads.
///
/// It exists because the grid's cells are intervals. The rational family fits a
/// numerator/denominator pair over an interval and a partition's pieces are
/// what it cuts, so a fixed grid is a partition its fits can be cut over exactly
/// as the shipped and narrow pieces are, and the specialization of \c RouteFit
/// below is what names this member. It is one fit under either scheme for the
/// reason the other two members of the family are: its coefficients are a
/// monomial numerator and denominator with no Chebyshev form to sum.
///
/// **What it hands a caller.** One pair per interval of the grid the Chebyshev
/// member is read at, fitted over that interval's own cell and read at the
/// mapped argument the grid's own locator builds for it - \c FlatPoint's \c t,
/// `2 (x * kFlatPerUnit - iv) - 1`, the same double both members are read at, so
/// one lookup addresses them. The numerator's coefficients are stored first and
/// the denominator's `q_1..q_k` after them with its constant term held at 1 -
/// the stored form the shipped and narrow members of this family read their own
/// rows with - so a caller sums the numerator by Horner, sums the denominator,
/// and divides once.
///
/// **The degree is the pair's own, and the table has no single one.** An
/// interval stores `m + 1 + k` doubles for each of the `kMaxBoysOrder + 1`
/// orders, with `m` and `k` its own and read off the table's per-interval degree
/// columns; its rows start at its own offset and step by its own stored count.
/// A caller therefore reaches an interval's rows through the interval, never
/// through one stride for the whole member. The cover is the grid's - the
/// intervals the locator spans, and nothing above them.
///
/// The four arrays that carry the layout are the emitter's: a per-interval
/// numerator degree, denominator degree, stored count and offset beside the
/// coefficients, in the shape the Chebyshev member's own grid block has and
/// written by the same generation.
///
/// **It models FitPolicy, and the members the grid does not serve refuse.**
/// \c Partition names the same \c RegionAPartition model \c UniformFit names,
/// because the concept asks every fit for one and this member's values are read
/// from the grid rather than from any piece of it. \c EvalPiece, \c RegionBSeed
/// and \c BandSource answer with a value no route can produce rather than with
/// another partition's fits, exactly as \c UniformFit's do: they complete the
/// contract, and a body that reaches one is a body this partition does not
/// serve, which refuses the partition where it is named. The contract is checked
/// where the fit is read and not here - the bodies default their fit parameter
/// to the policy's own and assert \c FitPolicy on it at the top, outside every
/// branch - so a member that does not model the concept fails at the entry
/// rather than inside a recurrence.
struct RationalFitUniform;

struct RationalFit;

/// The rational family over the narrow partition; see the specialization below.
struct RationalFitNarrow;

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

/// The uniform route: a Chebyshev family read at a fixed grid rather than a
/// derived partition, so both schemes are defined for it - it holds both stored
/// forms of its coefficients - and it is selected by the partition axis, which
/// is where "how the fitted intervals are cut" is decided.
///
/// The rational route's member over the same grid is a family of its own and not
/// a mode of this one: the two store different things, and it is the
/// specialization below that names it.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kChebyshev, kScheme, FitGranularity::kUniform> {
    /// The uniform table's fit.
    using Type = UniformFit<kScheme>;
};

/// The rational family at the uniform partition: the grid's cells are intervals
/// like any other partition's pieces, and this member is one numerator/
/// denominator pair fitted over each of them, in the same stored form the
/// shipped and narrow members use. It is one fit under either scheme for the
/// reason those are: its coefficients are a monomial numerator and denominator
/// with no Chebyshev form to sum.
///
/// This specialization carries the combination rather than falling to the
/// primary template above, whose `Type` is the Chebyshev family at whatever
/// granularity was named: without it a caller naming this route over the grid
/// would be answered by the Chebyshev member over the grid, at a certified
/// bound, under the rational route's name, with nothing reporting it. What the
/// member must provide is stated on \c RationalFitUniform above.
///
/// The grid's rung is refused where a policy names one - the table stores one
/// pair per interval and no per-order effective degree, so a relaxed rung has
/// nothing to cut by. Giving this member one is a fit to derive rather than a
/// switch to flip: the rung bodies resolve their fit through
/// \c detail::RationalRouteFitAtRung, whose conditional reads every partition
/// but the shipped one as the narrow one, so a uniform rung reaching it would
/// be answered by the narrow member's pairs under the grid's name - the
/// substitution this declaration is here to make impossible.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kUniform> {
    /// The single rational member over the grid, one pair per interval.
    using Type = RationalFitUniform;
};

/// The rational family: one fit under either scheme, because its coefficients
/// are a monomial numerator and denominator with no Chebyshev form to sum.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kShipped> {
    /// The single rational fit, shared by both schemes.
    using Type = RationalFit;
};

/// The rational route under the narrow partition: the same family fitted over
/// the narrow pieces' own intervals, one numerator/denominator pair per piece in
/// both regions, in the same stored form and read at the same mapped arguments
/// as the shipped member. The intervals are the Chebyshev route's narrow cut of
/// the region - a partition is a cut of the region and not a property of a
/// family - so what this member carries is the pairs, and it is one fit under
/// either scheme for the reason the shipped member is.
template <EvalScheme kScheme>
struct RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kNarrow> {
    /// The single rational fit, one pair per narrow piece.
    using Type = RationalFitNarrow;
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
///
/// The one error this type can carry is a route outside the FitRoute
/// enumeration, and it is reported where it is named: the entries constrain
/// their parameter with EvalPolicyLike, and RouteFit's own assertion is where a
/// value that is not an option fails.
///
/// The one error the axes carry beyond a route outside the enumeration is a
/// combination the build cannot serve - a member of a partition whose fits it does not
/// hold, or a rung whose table it has not derived - and each is refused where it is
/// named rather than at a kernel, because the partitions are different fits of the same
/// function over the same interval and a fallback would return one partition's values
/// under another's name. The single-precision lanes are not an exception to that:
/// their narrow partition's rung tables are derived like the double lane's - over
/// the narrow pieces' own degrees,
/// and, on the rational route, over the route's own pairs of those pieces - so a relaxed
/// rung of either family is a call those lanes answer as the double lane answers it. A
/// relaxed rung cuts a stored fit's own coefficients, so the rung is derived per partition
/// at compile time rather than truncated from a shipped row.
///
/// \tparam kFitRoute      the fit route; \c FitRoute::kChebyshev by default
/// \tparam kEvalScheme    the scheme the fit's coefficients are summed in;
///                        \c kDefaultEvalScheme by default, which is
///                        \c EvalScheme::kHorner
/// \tparam kEngineBudget  the computation budget of a single-precision engine,
///                        read by the float and half-precision lanes and by
///                        nothing else; \c BoysBudget::kFloat by default
/// \tparam kPackedAxis    the axis a packed evaluation vectorises over, read by
///                        the entries that have a wide axis to fill;
///                        \c PackAxis::kArguments by default
/// \tparam kGranularity   how narrowly the fitted domain is cut into pieces;
///                        \c kDefaultFitGranularity by default, which is
///                        \c FitGranularity::kNarrow
/// \tparam kDivision      how the recursion's per-order division is performed;
///                        \c kDefaultDivisionForm by default, which is
///                        \c DivisionForm::kRefinedReciprocal
///
/// \ingroup boys
template <FitRoute kFitRoute = kDefaultFitRoute,
          EvalScheme kEvalScheme = kDefaultEvalScheme,
          BoysBudget kEngineBudget = BoysBudget::kFloat,
          PackAxis kPackedAxis = kDefaultPackAxis,
          FitGranularity kFitGranularity = kDefaultFitGranularity,
          DivisionForm kDivisionForm = kDefaultDivisionForm>
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
    /// How the recursion's per-order division is performed.
    static constexpr DivisionForm kDivision = kDivisionForm;
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
    { P::kDivision } -> std::convertible_to<DivisionForm>;
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
///    Chebyshev route, the Horner scheme, the narrow partition, the
///    arguments-packing axis and the exact division form - whatever
///    \c kDefaultFitRoute, \c kDefaultEvalScheme, \c kDefaultFitGranularity,
///    \c kDefaultPackAxis and \c kDefaultDivisionForm
///    name at the revision a caller builds against - and the budget its policy
///    carries is inert;
///  - the **float** lane reads the budget at a relaxed multiplier, and its own
///    is \c BoysBudget::kFloat;
///  - the **half** lanes (\c fp16 and \c bf16) are the same engine under the
///    tighter \c BoysBudget::kFp16 budget, which is the axis that cuts their
///    fits for a 1e-7 region target rather than the float lane's — this is the
///    one place the four names differ in more than their spelling, and it is
///    why a single default for every precision would be the float lane's
///    budget imposed on the half lanes. It is not an accuracy they can
///    publish: they run the float lane's arithmetic and store what it returns,
///    so their published figure is that lane's (1.5e-7) plus the half-ULP
///    term the store adds;
///  - \c DefaultPolicyFp16 and \c DefaultPolicyBf16 denote one and the same
///    policy type, because fp16 and bf16 are one lane at one budget; they are
///    named twice so that a document can cite the default for the format its
///    reader is using, and a reader comparing the two names is comparing a
///    lane, not a choice.
///
/// **These defaults are a measurement, and the option probe is how it was
/// taken.** Which option is fastest is a property of the host, its flags and
/// its card, so the figures behind the two axes that have been measured were
/// read off the option probe this library ships for a consumer to run where
/// they deploy, and the axes that have not been measured against it carry the
/// shipped settings. The line between the two is stated axis by axis in
/// docs/lane-contract.md; the numbers behind the measured ones are the probe's
/// own report, on the machine it is run on, and not a figure this header can
/// restate.
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

/// The name a report prints a route under, and never null.
///
/// \param route the route
///
/// \returns a string literal naming it: "fused" or "separate", and "unknown"
///          for a value outside the enumerators
///
/// A value outside the enumerators is answered rather than refused. The route
/// this names is the one a caller selected, which is not always the one the
/// build runs: whether the separate route is really two roundings is the
/// build's contraction, measured per translation unit, and \c RouteInForce<T>()
/// and BoysBackends() are what report that.
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
