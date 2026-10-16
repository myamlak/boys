#pragma once

/// \file
/// The compile-time facts the rest of the library is written against: the
/// highest order the kernel serves, the multiplier m = 1 that is every lane's
/// default accuracy, and the axes a call site selects an evaluation by - the
/// route its fits come from, the scheme their coefficients are summed in, the
/// exponential a region-B ladder is seeded with, the cut of the domain those
/// fits are read from, the form each of a ladder's divisions is performed in,
/// the axis a packed evaluation vectorises over, and the budget a
/// single-precision engine evaluates at.
///
/// They stand in their own header so that a header naming them need not carry
/// the kernel: the CUDA lane's option rows are read from a header that has to
/// stay free of the library's C++23 ones, and defining the axes here is what
/// lets one name describe an option on both sides of the device boundary.
///
/// **One axis, one enumeration, wherever the members are the same choice on both
/// targets.** The route, the scheme, the region-B exponential, the cut of the
/// domain and the division form are one type each, named by the CPU lanes and the
/// CUDA lane alike: the choice is the same question on either target, and only the
/// arithmetic that answers it differs.
/// Two enumerations for one axis is how the two targets come apart unnoticed - a
/// member one side carries and the other never grew reads as a member the library
/// does not have.
///
/// The packing axis is the one axis whose member sets genuinely differ, and it is
/// therefore the one axis the two sides name separately: \c PackAxis is the call
/// shapes' reading of it, and \c DevicePacking (`boys/boys_cuda_options.hpp`) adds
/// \c kNotApplicable - the member the device entry that produces one order states,
/// so that a row saying nothing there is not read as the ladder. That is a
/// difference in the member set and not in the question, it is the only one, and
/// it is written here so that it reads as the decision it is.

#include "boys/boys_build_defaults.hpp"

#include <cstdint>

namespace boys {

/// Highest Boys order supported by the kernel.
inline constexpr int kMaxBoysOrder = 32;

/// The accuracy multiplier of every lane, and the only one: m = 1 is full static
/// accuracy, bit-identical to the certified lanes (the documented contract).
/// Every bound this library documents is this multiplier's figure.
inline constexpr double kBoysFullAccuracyMultiplier = 1.0;

/// Which certified fit serves a region, where the library carries more than
/// one for it.
///
/// Region A and region B each carry two tables of fitted coefficients, placed
/// against the same bar over the same interval, and this names which one a call
/// evaluates: region B's is one lowest-order seed per route that the higher
/// orders are reached from, and region A's is a piecewise fit of every order
/// over each of the pieces its lane's tables cut the region into - two equal
/// bands an order on the double lane, each order's own pieces on the
/// single-precision one. They are alternatives rather than rungs of one design:
/// naming the rational one changes the fits that serve the intervals it covers
/// and nothing else.
///
/// What each route promises, over what interval, and from which argument naming
/// it changes anything, is BoysFitRoutes' report rather than this enumeration's:
/// a route whose fit reaches further than its selector takes over says so there
/// instead of being stretched to claim a domain it does not serve. Region A's
/// rational route is that case.
///
/// A value outside the enumerators - cast in from outside the enum, or named by
/// a newer header - is not a route, and every entry on this surface treats it
/// as \c kChebyshev: the default is the route every build carries.
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
///
/// The value is the build's rather than this header's: it is
/// \c BOYS_BUILD_DEFAULT_FIT_ROUTE, from `boys/boys_build_defaults.hpp`, whose
/// committed file carries \c FitRoute::kChebyshev and which a build that has
/// measured its own machine replaces with its own choice.
inline constexpr FitRoute kDefaultFitRoute = BOYS_BUILD_DEFAULT_FIT_ROUTE;

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
    /// summation the certified lanes were first written with, and the one a
    /// caller names to read a fit that way.
    kSplitClenshaw = 0,

    /// Horner's rule on the monomial form of the same fit.
    kHorner = 1,
};

/// The scheme the entries evaluate in when the caller names none.
///
/// The value is the build's rather than this header's: it is
/// \c BOYS_BUILD_DEFAULT_EVAL_SCHEME, from `boys/boys_build_defaults.hpp`, whose
/// committed file carries \c EvalScheme::kHorner and which a build that has
/// measured its own machine replaces with its own choice. The paragraphs below
/// are why that shipped choice is the one it is.
///
/// Horner's rule: a call site that names no scheme is compiled as the entry
/// naming \c kHorner, and naming \c kSplitClenshaw is how a caller asks for the
/// other summation. The two sum one fit, so this value is a choice between two
/// ways of computing the same answer rather than between two accuracies.
///
/// **It is not the cheaper of the two at either partition.** On the host these
/// were last measured on, the two schemes came out within 0.6% of each other
/// where the fit they sum is a piece of the narrow partition, while at the
/// shipped partition the split Clenshaw recurrence was 20% to 22% cheaper than
/// Horner's rule. What that measurement establishes is a tie: three rows of the
/// double lane's full-accuracy class for the all-orders shape — the shipped
/// partition summed by the split Clenshaw recurrence, and the narrow partition
/// summed by either scheme — came out within 0.9% of each other, against the 5.6
/// to 7.4 points one of those rows moves by from one run to the next, with the
/// class's next row 11.5% to 12% behind them. This default is one of the three
/// tied rows. A caller who wants this ranking on their own machine runs the
/// option probe (boys_probe.hpp), which measures it there and names the pairs it
/// could and could not separate.
inline constexpr EvalScheme kDefaultEvalScheme = BOYS_BUILD_DEFAULT_EVAL_SCHEME;

/// Which exponential seeds region B's ladder.
///
/// Region B evaluates F_0 from a stored fit and steps to the caller's top order
/// with one term at every step, t = e^{-x}/2. This axis names the arithmetic `t`
/// is evaluated in. It is one axis with one member set for the CPU lanes and the
/// CUDA lane alike, and each target states its own arithmetic for each member
/// below: what the two share is the choice, and what they do not share is the
/// code that answers it.
///
/// The recurrence that consumes the value is why the choice matters at all. Its
/// condition number - the ratio of the dominant solution of the homogeneous
/// recurrence to the wanted one ([Gautschi1967]) - is 7.6e4 at the region-B
/// boundary, n = 32, and falls as the argument grows, so a seed error whose
/// *relative* size grows with the argument fails a bound over a band of region B
/// at the highest order while holding it everywhere else.
///
///  - \c kAccurate is the library routine of the target: `std::exp` on the host,
///    `expf` on the device. Its relative error is flat - under a ulp on the host,
///    whose worst over a dense sweep of region B is 0.53 ulp, and 2 ulp on the
///    device - so it holds the target's own figure in every region
///    and needs no term of its own there. On the device it is the arithmetic the
///    f32 batch bodies run: at m = 1 a single and a batch evaluation of the same
///    (n, x) return the same bits outside region A, and that is why it is that
///    lane's default. On the host it is the arithmetic the single-precision
///    lane's region B has always run, and the one both lanes' region-A paths
///    (the extended band's seed and the downward recursion's term) run at every
///    member of this axis, since neither is region B.
///
///  - \c kFast is a fast approximation of the same function, and what the library
///    may spend on it is what separates the two targets here. On the device it is
///    the hardware approximation with its argument-scaling residual removed:
///    2^fl(y log2 e) is rounded once in that product, which is what grows its
///    error with |y|; the residual fma(y, log2 e, -t) is exact and
///    2^(t + d) = 2^t 2^d approximates 2^t (1 + d log 2), so two fused steps take
///    the error back to the approximation's own few ulp, flat in the argument. Its
///    bound is the lane's plus its own seed's contribution, certified at 8e-8 -
///    0.41 of the lane's budget - which the condition number derives and the
///    device gate's sweep confirms (5.0e-8 measured at m = 1). On the host it is
///    the reduced-argument polynomial this library fits for itself
///    (`kRegionBExpReduced` and the reduction around it, `boys/boys_impl.hpp`):
///    degree 10 on |r| <= ln 2/2, 4.458e-16 relative, reduced by the textbook
///    x = k ln 2 + r with k out of a magic constant. It is read where the ladder's
///    own requirement admits it - the polynomial stands inside that requirement
///    ten times over from kRegionBExpCheapFrom = 11.202305053129882 upward - and
///    below that argument the requirement is tighter than the polynomial's error
///    and the member reads the library routine instead. **That arm is part of the
///    member and not a fallback**, for the reason stated below: a member whose
///    failing band is interior to region B is not offered at all. The
///    single-precision lane reads this member through one widening and one
///    narrowing - the same evaluation, rounded once to single - so the cut applies
///    to it unchanged and its error there is the narrowing's half ulp.
///
/// The bare approximation is not offered at any multiplier: its failing band is
/// interior to region B, which a caller cannot name a sub-range of. On the device
/// that is what removing the residual buys, and on the host what the library
/// routine below the cut buys.
///
/// **Which member a call runs when it names none is each target's own**, because
/// each target's own arithmetic is what its published figures were measured at:
/// the device lane's default is \c kDefaultRegionBExp (`boys/boys_device_tables.hpp`)
/// and the host's is \c kDefaultHostRegionBExp below. Naming the other member is
/// naming another arithmetic and never a cheaper spelling of the same one.
///
/// One name serves both device lanes that take the option: the batch entry
/// BoysCuda::SingleF32 (a run-time argument) and the device entry
/// BoysDeviceSingleF32 (a template argument). On the host the member is a
/// template argument of the evaluation policy, so every entry whose policy names
/// one runs it.
///
/// \ingroup boys
enum class RegionBExp : int {
    /// The library routine: `std::exp` on the host, `expf` on the device, and the
    /// arithmetic the device's f32 batch bodies run. Flat relative error, so it
    /// reads the lane's own figure in every region.
    kAccurate = 0,
    /// The fast approximation, read with the bound its own target documents for
    /// it: the hardware approximation with its argument-scaling residual removed
    /// on the device, the reduced-argument polynomial of `boys/boys_impl.hpp` on
    /// the host. Read it with the member's own bound and not with the lane's.
    kFast,
};

/// The region-B exponential the host entries evaluate when the call site names
/// none: \c RegionBExp::kFast, the reduced-argument polynomial read with the
/// library routine below its cut.
///
/// **This one, because the double lane's own arithmetic is this member.** Every
/// per-region figure this repository publishes for the double lane was measured
/// at it, so the axis arrives additive there: naming \c kAccurate is what changes
/// the arithmetic a region-B seed is evaluated in, and a caller who names nothing
/// keeps the values the documents state. The single-precision lane's figures were
/// measured at the library routine, and it reads this member narrowed - the
/// double arithmetic rounded once to single at the lane's own type. Below the
/// member's cut those are one evaluation bit for bit, because the narrowed member
/// reads the same library routine there; above the cut the member adds the
/// polynomial's 4.458e-16 relative to the half ulp the narrowing contributes,
/// which is 4.458e-16 read beside 5.96e-8 - the narrowing's rounding and nothing
/// else. So no published figure moves on that lane either - what moved is a
/// fraction of a rounding whose share of the ladder the figure was measured with
/// room for - and the values a caller gets there move by at most that one
/// rounding, half an ulp of that lane's type, and not at all below the cut.
///
/// The device lane's default is the other member and is its own constant, because
/// that lane's published figures were measured at \c kAccurate. Both are stated
/// rather than inferred: the two targets' defaults differ because their
/// arithmetic does, not because one of them is a preference.
inline constexpr RegionBExp kDefaultHostRegionBExp = RegionBExp::kFast;

/// The region-B exponential the device lane evaluates when the call site names
/// none: \c BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP, from
/// `boys/boys_build_defaults.hpp`.
///
/// **Declared here beside the host's, and read by two.** It is the device lane's
/// own member and not the host's - the lane's published figures were measured at
/// the library routine, which the constant's own documentation in
/// `boys/boys_device_tables.hpp` states - and it is declared here because the
/// policy table's device rows name it where they are composed (`boys/boys.hpp`
/// does not carry the device tables header). `boys::kDefaultRegionBExp` in
/// `boys/boys_device_tables.hpp` is this lane's own spelling of the same
/// constant, declared where the tables it is read with are.
inline constexpr RegionBExp kDefaultDeviceRegionBExp = BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP;

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
/// combination that cannot exist. What a lane refuses is a member of a partition it
/// has not stored.
///
/// The across-orders packing axis is otherwise built: the packed lane reaches a
/// per-order cut
/// with a gathered fetch, reading each of the four orders it holds its own
/// piece and coefficients instead of stepping one piece's coefficients at a
/// fixed stride. This includes the single-precision lanes, whose narrow rung
/// tables are derived like the double lane's rather than absent.
///
/// **It stands here rather than in backend.hpp, where it was defined,** because it
/// is an axis of the device lane's option space as well as of the host's calls: the
/// rows of BoysDeviceOptions() state the cut their entry reads its fits from
/// (boys_cuda_options.hpp, DevicePartitionOf), and that header is compiled by nvcc in
/// a translation unit that has to stay free of the library's C++23 headers. One type
/// serves both sides — the device's coarsest rows read the same committed tables the
/// host's \c kCoarsest reads — so the axis is named once rather than mirrored.
///
/// \ingroup boys
enum class FitGranularity : std::uint8_t {
    /// The partition the certified lanes are defined by: region A's per-order
    /// pieces and region B's single seed, at the degrees the committed tables
    /// carry. Region A's pieces are two equal bands an order, at degree 20 and
    /// 18, on the double lane, and two to four pieces an order, all at degree 10,
    /// on the single-precision lane; region B is one seed on both. A caller names
    /// it to read those tables.
    kCoarsest = 0,

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
inline constexpr FitGranularity kDefaultFitGranularity = BOYS_BUILD_DEFAULT_FIT_GRANULARITY;

/// How the recursion's per-order division is performed.
///
/// A ladder step divides in region B and in region C, and each of those ends in
/// one of these. The double host lanes' region C is the exception: its terms are
/// the asymptotic series' own products and hold no step of a chain, so that
/// region is answered under all three forms by the same values, and the one
/// division it has is the seed's closed form - a closed formula, which the
/// paragraph below places outside the axis. The float, half and device lanes'
/// far ladders are chains and take the form the caller names. They are different
/// arithmetic and not three spellings of one: a
/// quotient is correctly rounded, and a product by a rounded reciprocal rounds
/// twice, so the plain form may differ from the exact one by an ulp per step and
/// a ladder of them accumulates that; the refined form is the plain one carried
/// back to the exact one's rounding by a fused multiply-add, at the price of two
/// dependent operations per step.
///
/// **It stands here rather than in backend.hpp, where it was defined,** because it
/// is an axis of the device lane's option space as well as of the host's calls: the
/// rows of BoysDeviceOptions() name the form their entry runs (boys_cuda_options.hpp),
/// and that header is compiled by nvcc in a translation unit that has to stay free of
/// the library's C++23 headers. Declaring it beside the other axes is what lets one
/// name describe the axis on both sides of the device boundary.
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
/// **The single-precision lane's plain reciprocal publishes its own figure.**
/// That lane's downward ladder divides by the step's constant, and the step reads
/// whichever form the caller named. The plain form rounds once more there, which
/// takes the ladder past the lane's base: over the accuracy gate's own
/// reference grid at the reference multiplier its worst cell is 1.7514e-07 at
/// n = 0, x = 9.74054909, where the lane's exact and refined forms deliver
/// 1.0835e-07 at worst — two absolute readings of one cell, 6.67e-8 apart. So the
/// lane publishes a term beside its base for that form — 2^-22 · F_n(x), or
/// 1.5e-7 where that is the larger — plus 1e-7, and the accuracy accessor answers it
/// when the caller names the form. The double lane's downward ladder takes the form with room to
/// spare: the plain form's worst cell over the grid's arguments below kX0 is
/// 6.7292e-15 where exact division's is 3.2162e-15, and that cell lies in the band,
/// which publishes 3e-14.
///
/// **Which of the three is cheapest is a property of the host and not of this
/// library.** A processor whose division is a multi-instruction sequence pays far
/// more for the exact form than for either product; one whose fused multiply-add is
/// scarce relative to its multiplier pays more for the refined form than the plain
/// one. The three are therefore all carried, and ranked by the option probe on the
/// machine it is run on. The measured figures for this host are in the probe's own
/// report.
///
/// \ingroup boys
enum class DivisionForm : std::uint8_t {
    /// One division per step: the form the recurrences are written in.
    kExactDivision = 0,

    /// One reciprocal per divisor and one product per step: the argument's
    /// reciprocal, formed once and multiplied through the ladder, and the step
    /// constant's, read from a compile-time table and multiplied the same way.
    /// It governs every division of a chain, the single-precision lane's
    /// downward ladder included: that lane publishes a figure of its own for
    /// this form rather than substituting another arithmetic for it.
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
///
/// **This is the host's name, and the device lane has its own.** The CUDA
/// entries take the form as an argument and every one of them runs every form,
/// so what a device call performs its divisions in when it names none is
/// \c kDefaultDeviceDivisionForm below and not this: a build whose card
/// measured another form names that lane's form in the seam
/// (\c BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM) without moving the host's
/// arithmetic, and one that names nothing keeps each lane at the form its own
/// figures were measured at.
inline constexpr DivisionForm kDefaultDivisionForm = BOYS_BUILD_DEFAULT_DIVISION_FORM;

/// The division form the device lane's divisions are performed in when the call
/// site names none: \c BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM, from
/// `boys/boys_build_defaults.hpp`.
///
/// **One name per target, because the two targets' figures are two sets.** It is
/// declared here beside the host's so that the policy table's device rows can
/// name it where they are composed (`boys/boys.hpp`), and it is declared at all
/// so that the device lane's form is a seam name of its own: the committed value is
/// the form every figure this repository publishes for that lane was measured at,
/// which is what a build that names nothing keeps.
inline constexpr DivisionForm kDefaultDeviceDivisionForm = BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM;

/// Which axis a packed lane vectorises over.
///
/// A packed lane keeps four doubles in a register and a call has to supply four of
/// something. Which something is a property of the call shape: \c BoysAllOrders is
/// one argument and every order, so the orders are the only axis with anything in
/// it, while the batch shapes carry several arguments at one order and can fill a
/// register with those. The two evaluate the same stored fits by the same
/// arithmetic and differ in what the vector lanes hold, so they are not rungs of
/// one design and neither replaces the other.
///
/// The axis is one field of an \c EvalPolicy. Naming the orders axis on an entry
/// that has one order is a combination this library does not carry and is refused
/// where it is named: a packed lane keeps four orders of one argument and such a
/// call produces one.
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
/// The shipped one: a call site that names no axis packs the arguments axis. A
/// build replaces it through `BOYS_BUILD_DEFAULTS` like the other four
/// (boys/boys_build_defaults.hpp).
inline constexpr PackAxis kDefaultPackAxis = BOYS_BUILD_DEFAULT_PACK_AXIS;

/// The computation budget a single-precision engine evaluates at.
///
/// The lanes that keep half-precision arguments and compute in single (f16.hpp)
/// hold a tighter bar than the float lane's, so the engine both lanes call is
/// told which of the two budgets it runs under rather than assuming one. The
/// double lanes carry no such choice and read no budget.
///
/// \ingroup boys
enum class BoysBudget : std::uint8_t {
    /// The float lane's 1.5e-7 budget: the absolute figure that lane's fits are
    /// cut against, in the region-A and region-B tails. It is not the lane's
    /// published bound, which is stated as a fraction of the value.
    kFloat = 0,

    /// The fp16/bf16 engine's 1e-7 region target. The half lanes run the float
    /// lane's engine with the fits cut for the tighter target, and what they
    /// publish is the fp32 engine's own arithmetic stated absolutely, 1.5e-7,
    /// plus the half-ULP term their store adds: a budget is what the cuts are
    /// placed against, not a bound a caller is given.
    kFp16 = 1,
};

// Each axis is named here, beside its own enumeration: a caller that prints an axis value
// inside a report reads the name from this header rather than carrying the kernel for it.

/// The name a report prints a packing axis under, and never null.
///
/// \param axis the axis
///
/// \returns a string literal naming it: "arguments" or "orders", and "unknown"
///          for a value outside the enumerators
///
/// A value outside the enumerators - cast in from outside the enum, or named by a
/// newer header - is answered rather than refused, so both arrive here as
/// "unknown": a caller that has to tell "this build does not carry that axis" from
/// "that is not an axis at all" tests the value against the enumerators itself.
const char* PackAxisName(PackAxis axis) noexcept;


/// The name a report prints a division form under, and never null.
///
/// \param form the form
///
/// \returns a string literal naming it: "exact-division", "plain-reciprocal" or
///          "refined-reciprocal", and "unknown" for a value outside the
///          enumerators
///
/// A value outside the enumerators is answered rather than refused.
const char* DivisionFormName(DivisionForm form) noexcept;


/// The name a report prints a region-B exponential under, and never null.
///
/// \param exp the member
///
/// \returns a string literal naming it: "accurate" or "fast", and "unknown" for a
///          value outside the enumerators
///
/// A value outside the enumerators is answered rather than refused.
const char* RegionBExpName(RegionBExp exp) noexcept;


/// The name a report prints a granularity under, and never null.
///
/// \param granularity the partition
///
/// \returns a string literal naming it: "shipped", "narrow" or "uniform", and
///          "unknown" for a value outside the enumerators
///
/// This names a partition; it does not report whether the build serves one, which
/// is FitGranularityHasRoute and FitGranularityHasAxis.
const char* GranularityName(FitGranularity granularity) noexcept;

namespace backend {

/// Which multiply-add a lane's kernels are built from.
///
/// The two are different arithmetics, not two spellings of one: the fused step
/// rounds once and the separate step rounds twice, so the same kernel delivers
/// a different value under each and a bound certified for one does not carry
/// to the other.
///
/// It stands in this header rather than beside the lanes that run it because the
/// front door reports each scheme row's route (`EvalSchemeInfo`), and a header
/// naming that row must not have to carry the kernel for it.
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
/// \returns a string literal naming it: "fused" or "separate", and "unknown" for a
///          value outside the enumerators
///
/// This is the route a caller selected, which is not always the one the build runs:
/// whether the separate route is really two roundings is the build's contraction,
/// measured per translation unit, and \c RouteInForce<T>() and BoysBackends() are
/// what report that.
///
/// The definition is inline and sits beside the lanes that run the route
/// (boys/backend.hpp); this declaration is here so that a caller of the name does not have to
/// carry the kernel for it.
inline const char* MulAddRouteName(MulAddRoute route) noexcept;
} // namespace backend
/// How the far region's values are reached: the representations of one region,
/// all carried, none substituted for another.
///
/// The far region is \c [kX1, inf), and its boundary is not a partition's to
/// move: whatever cuts the fitted regions carry, the arguments above the grid's
/// hi-bound are served by one seed and a dependent upward recurrence. That is
/// one representation of the region, and it is what this axis is beside.
///
/// **They are not a partition.** Every partition of the fitted regions is
/// served by the same seed-plus-recurrence above \c kFlatHi, so an arm of the
/// partition axis could not name any of them: the representation is orthogonal
/// to where the fitted regions are cut, and it is a choice of its own. \c kCorrectedSeed
/// is the one arm that is not a second definition of the function at all - it is the
/// recurrence's arithmetic with its two defects repaired - which is why the pairs
/// are ranked by measurement rather than by construction.
///
/// **A named representation serves the arguments it covers and changes nothing
/// else.** \c kAsymptoticRecurrence is the bare ladder above the fitted grid's
/// hi-bound, \c [kFlatHi, inf), at the default entry's values: the same seed the
/// other two arms read with the series' own terms, and neither of the two repairs,
/// so it is the one arm whose error is the region's own rather than a fit's or a
/// table's, and the fitted regions below its span answer as they do under every name
/// on this axis. \c kScaledTable serves the arguments in \c [kScaledFarX0,
/// kScaledFarX1) from the stored scaled table and is the default entry's elsewhere,
/// so every argument a caller may pass is answered under either name and the values
/// outside the table's span are the same under both.
///
/// \ingroup boys
enum class FarRepresentation : std::uint8_t {
    /// The closed form and the asymptotic series' own terms: \c F_m = F_0 A[m] h^m
    /// with \c h = 1/(2x), one table read and two multiplies an order. It is the
    /// region's own arithmetic with neither of the two repairs \c kCorrectedSeed
    /// carries: the seed is the closed form with \c H_0 held at 1, whose relative
    /// error is \c erfc(sqrt(x)), 2.65e-14 at \c kX1, and the series truncates at the
    /// order asked for, its tail \c 0.5 x^-(m+1/2) Gamma(m + 1/2, x) reaching 5.0e-14
    /// absolute at n = 32, x = \c kX1 - 2.55 times the value 1.95e-14 there, an
    /// argument the fitted grid serves. Over the span the arm is asked for, \c
    /// [kFlatHi, inf), the worst measured cell sits at the span's low end: n = 32,
    /// x = \c kFlatHi reads 0.288 of the bound, a 3.47x margin, and x = 32 reads
    /// 0.0916. The bound's relative arm is the one it does not hold, at any x: at
    /// n = 32, x = 31 its error is 1.4e13 times \c 1e-13 F_n(x), where \c
    /// kCorrectedSeed reads 0.0197 of \c 1e-13 F_n over its span.
    kAsymptoticRecurrence = 0,

    /// The stored scaled table: a Chebyshev fit per (interval, order) of
    /// \c H_n = 2 F_n x^(n+1/2) / Gamma(n+1/2), the regularized lower incomplete
    /// gamma, read at each order and multiplied by a running \c m_n. The far
    /// orders are then independent of one another where the recurrence makes them
    /// a chain, and the table's own delivered error is published in its rows.
    kScaledTable = 1,

    /// The same recurrence, seeded and stepped correctly. The seed above is the
    /// closed form \c sqrt(pi)/(2 sqrt(x)), which is \c F_0 * H_0 with \c H_0
    /// held at 1: its relative error is \c erfc(sqrt(x)), 2.65e-14 at \c kX1.
    /// This arm carries that error as one polynomial over the span where it is
    /// above the unit roundoff, \c [kX1, kSeedCorrectionX1), and the ladder adds
    /// the \c exp(-x)/2 term the asymptotic step drops, which at \c kX1 is
    /// larger than \c F_32 itself.
    kCorrectedSeed = 2,
};

/// The name a report prints a far representation under, and never null.
///
/// \param representation the member
///
/// \returns a string literal naming it: "asymptotic-recurrence", "scaled-table" or
///          "corrected-seed", and "unknown" for a value outside the enumerators
///
/// A value outside the enumerators is answered rather than refused.
const char* FarRepresentationName(FarRepresentation representation) noexcept;

/// The representation the far region's default entries run, and the fallback for a name this
/// build does not serve. Those are one statement here and not two: every member of the axis is
/// served, so what the fallback answers is a value that is no member of the enum at all.
///
/// **The bare ladder, chosen for what it costs and priced where it is written.** The
/// asymptotic series' terms are the region's cheapest arithmetic - one table read and
/// two multiplies an order, no exponential and no chain - and what they deliver is the
/// series' own tail. Under the published bound, the larger of `1e-13 F_n(x)` and
/// `1e-13 F_0(x)`, the tail holds it wherever the arm is asked for: over
/// `[kFlatHi, inf)` its worst measured cell is n = 32, x = `kFlatHi`, at 0.288 of the
/// bound and a 3.47x margin. The one place the bare ladder fell outside the bound
/// under the old 29.0 hi-bound was the sliver `(29.0, 29.9290065]`, and the fitted
/// grid serves that sliver once the hi-bound is `kFlatHi`. What the tail does not
/// hold is the bound's relative arm, and that is the price this choice pays: near the
/// series' knee the tail is the term (0.78 of the value at n = 28, x = 29) and at
/// n = 32 its error is 1.4e13 times `1e-13 F_n(x)`, so a caller reading relative to
/// F_n(x) needs `kCorrectedSeed` - the same arithmetic with the seed's error carried
/// and the term restored, reading 0.0197 of `1e-13 F_n(x)` over the span where it
/// serves. The scaled table is ranked against both by the option probe, which has not
/// run.
inline constexpr FarRepresentation kDefaultFarRepresentation = FarRepresentation::kAsymptoticRecurrence;

} // namespace boys
