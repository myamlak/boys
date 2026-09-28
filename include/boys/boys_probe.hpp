#pragma once

/// \file boys_probe.hpp
/// The option probe: which of this library's evaluation options is fastest
/// where the caller actually runs it, and whether the measurement is strong
/// enough to name one at all.
///
/// **Why this is a library entry and not a benchmark.** Which option wins is a
/// property of the machine and the build, not of the library: the vector tier is
/// present on x86_64 and absent elsewhere, a bare `a * b + c` is one rounding or
/// two depending on the target and the flags, so an arithmetic with more multiply
/// adds can be the cheap one on a host that fuses and the expensive one on a host
/// that calls out, and a lane that halves the precision of the argument trades
/// accuracy for cost by an amount that depends on how the consumer builds. A
/// ranking printed in documentation would be a claim about the machine it was
/// written on; this entry measures the ranking on the machine it is called on and
/// reports what it found, including how confident it is.
///
/// **What it measures.** Every option is asked the library's own question:
/// a set of arguments in which each argument carries its own highest order, and
/// every order from 0 up to it evaluated at that argument — one seed, then the
/// recursion up to that argument's order, which is the shape an integral engine
/// asks for. The arguments are log-uniform over a range that populates every
/// region of the kernel, and the per-argument orders are drawn as the sum of two
/// shell angular momenta, which is the distribution a shell-pair batch presents.
///
/// **What it reports per option**: the accuracy the option actually delivered
/// beside its cost, so a consumer can see whether a faster option was faster at
/// the same accuracy or merely at a lower one. Every column is documented on the
/// field that carries it.
///
/// **The whole option space, enumerated from the library.** The fit route, the
/// evaluation scheme, the partition of the fitted regions and the packing axis
/// are each a member of a set the library reports (BoysFitRoutes for the double
/// lane and its carriers, BoysFitRoutesF32 for the single-precision engine,
/// BoysEvalSchemes, BoysFitGranularities, BoysPackAxes), crossed with the
/// accuracy rungs the build can name. The probe walks that product once per
/// precision lane, from the lane's own fit table and carriage answer rather than
/// from a list written beside it, so a member the library gains is measured
/// without the probe being edited, and no lane is measured at fewer axes than
/// the library serves it at. A cell the library does not serve is stated with the
/// library's reason in a coverage section that accounts for the whole product: an
/// unstated omission would read as an option that does not exist.
///
/// **The classes are one precision, one rung and one question shape.** A class
/// is the set of options that are alternatives for one need a caller has:
/// everything the library picks on the caller's behalf is ranked inside that one
/// class. Three things decide membership, because each of them changes what the
/// caller gets back. The precision — fp64, fp32, fp16 and bf16 — is the choice
/// the caller has already made from the accuracy their calculation needs, and
/// halving the precision is not a faster answer to the same question. The rung —
/// the multiplier an option was built at, as the library's own tables report it
/// — is the accuracy an option was answering at. The shape is the question
/// itself: the all-orders entry returns one argument's own ladder per call and
/// the all-N entry a common ladder for a whole array, so a caller asking one of
/// them is not served by the other however fast it is. Membership in a class is
/// decided by those three and never by comparing one row's documented figure
/// against another's: a figure belongs to one lane, and reading one lane's figure
/// across lanes is how a class ends up empty by construction.
///
/// **Everything that only changes how one answer is computed is ranked inside
/// the class.** A class's rows differ by the fit route they read, the scheme
/// they sum it with, the partition of the fitted regions and the packing axis
/// they carry, and by whether a sorted array is declared to the all-N entry so
/// that it skips the sort — all of them interchangeable bodies for one answer,
/// so they compete inside the class and never divide it. A class's winner is
/// therefore the fastest option of that precision, built at that rung, answering
/// that question on this machine. A row of the class that documents another
/// figure — the same rung reached with a looser bound — is a member and is
/// reported by name rather than ranked as an equal silently.
///
/// **The default comes from the reference class of the workload's own shape.**
/// The default this report names is chosen inside one class: the certified
/// double lane's precision at the reference multiplier, for the all-orders
/// shape, whose every row documents the certified bound and answers the question
/// this probe's workload asks. Choosing there is choosing among equals — the
/// pool trades nothing but speed, which is the one thing a default is allowed to
/// trade. A relaxed rung is never the default however fast it measures, because
/// the speed it shows is bought with accuracy, and neither is the winner of
/// another shape's class.
///
/// **How the default is reached, and what happens when the pool ties.** The
/// default is the fastest row of that class by the figures this report prints, so
/// the name it ends with and the class table above it never disagree about which
/// option is cheapest. Where the class's own rounds place every rival behind its
/// leader, that row is the default and the report says it is a measured ordering.
/// Where they do not, the probe does not fall back to a figure counted from the
/// library's tables: it re-runs the tied options alone at a larger protocol,
/// several times, and reports what each run led with. A vote naming the same row
/// is reported as the longer protocol confirming the class's fastest; a vote
/// naming another is reported as a class the run could not separate — a tie
/// among equals, stated as one, rather than a licence to hand back a row the
/// table shows behind. Either way a run that measured the pool ends with exactly
/// one default, and says which of these ways it got there.
///
/// **The comparison is paired, and made inside a round.** Every option is called
/// once in every round, and the comparison between two options is the ratio of
/// their times *within the same round*, never a ratio of two figures taken from
/// different rounds. That is what makes the answer survive a machine whose clock
/// is not constant: a laptop's opportunistic boost decays and its thermal state
/// drifts, so the wall-clock time of fixed work grows through a run with no
/// other process involved, and two options timed in different rounds are then
/// being compared across two different clocks. Inside one round both options ran
/// under the same clock, so a drift that is common to the round cancels in the
/// ratio. The order the options are visited in is shuffled per round with the
/// workload's own seed, so a position in the round cannot become a systematic
/// advantage either, and every figure below is an aggregate of those paired
/// ratios rather than of absolute times taken across the run.
///
/// **It refuses to order noise, and it says how close it could look.** The
/// reported cost of an option is the reference lane's own lower-quartile cost
/// scaled by that option's ratio to the lane at the middle of the run's rounds,
/// and its spread is the ratio of the upper quartile to the lower quartile of
/// those same ratios. For each rival of a class's leader the probe forms the
/// pair's own within-round ratio, and the rival is ordered only when the middle
/// half of those rounds puts it behind the leader — when the pair's own band
/// clears one. A rival whose band straddles one cannot be placed, and the probe
/// prints the run's resolution and says which options it could not separate and
/// in how many of the rounds each was the slower of the two. The same happens
/// when too few rounds were run for a band to exist.
///
/// **The instrument is a diagnostic and gates nothing.** Every pass carries runs
/// of a fixed-work integer spin — the canary — bracketing the timed region and
/// taken between its rounds. The spin holds no floating-point state, so the
/// arithmetic question this probe is about cannot change the instrument's own
/// cost. Its readings are reported beside each pass, and a pass whose spin
/// disagreed with itself by more than ProbeOptions::canarySpreadAlarm is marked
/// as one that ran on a wandering machine — **and is still used**, because a
/// fixed-work spin measured by wall clock measures the clock as much as the load,
/// so a decaying clock widens the spin's own spread on a machine that is doing
/// nothing else, and a rule that discarded on that spread would discard the
/// measurement rather than the machine. What decides what can be ordered is the
/// spread of the paired ratios, measured in the units the comparison is made in.
///
/// The load readings printed beside each pass are context too: a background
/// average over the quiet window before the timed region, a point reading either
/// side of it, and the level the spin held while the timed rounds ran. A reading
/// is the percentage by which the spin took longer than the fastest run of that
/// same work ever seen on this machine, so it says how busy the machine looked
/// rather than how well it could be measured. It is not the machine's total
/// processor utilization — the standard library cannot reach another process's
/// processor time, which is what a system-wide counter would need — it is the
/// load a single-threaded, CPU-bound caller actually suffered.
///
/// **The accuracy column.** Each option's values are compared against the
/// certified per-order fp64 lane — `BoysSingle` at the reference multiplier —
/// evaluated at the order and the argument the option was actually asked about.
/// That last clause matters for the lanes that narrow the argument: a
/// single-precision lane rounds `x` before it evaluates, so holding it to the
/// double-precision argument would measure the representation of the argument
/// rather than the arithmetic of the lane. It matters again for the batch
/// entries, whose F_k below its own highest order is a recurrence's value rather
/// than a per-order walk's. The comparison has a floor: the certified lane is
/// itself documented at a bound, that bound is read from the library and
/// printed, and a difference at or below it means the two are indistinguishable
/// at this comparison's resolution — not that the option is proven to that
/// bound, which is what the committed reference grid and the test suite are for.
/// A row whose measured difference is above its own bound but at or below that
/// floor is reported in a third state rather than as a failure: the difference
/// between two partitions or two rungs carries the certified lane's own error
/// with it, and that error is what the floor is.
///
/// **What this probe does not measure**, and why. The native half lane is left
/// out: it covers region C only, its orders are useful up to 8, and its results
/// are scaled by a power of two, so its cost is not the cost of the same
/// question. The region-A matrix-product transform is left out: it computes
/// region A alone, so measuring it means composing the rest of an evaluation, and
/// that composition would be measured rather than the entry. The CUDA lanes are
/// left out: they are a separate optional build that needs a device, and the
/// options this probe ranks are the CPU ones the caller's own build carries.
///
/// What the option space itself refuses is not left out but named: a cell this
/// build cannot serve is listed with the library's reason in the report's
/// coverage section, counted rather than omitted. The list is read from the
/// library's own tables rather than kept here, so it shrinks by itself as cells
/// are built; where those tables serve every cell of the space, the list is empty
/// and the coverage section says so in those words.
///
/// **This result is about the machine it was measured on.** The report says so
/// in its own output, not only here, because a table of costs pasted into a
/// bug report months later would otherwise read as a general claim about the
/// library.
///
/// \ingroup boys

#include "boys/backend.hpp"
#include "boys/boys.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace boys {

/// The workload and the pass protocol the probe runs, with the defaults a
/// caller who wants a representative answer should leave alone.
///
/// The knobs exist because the right workload is the caller's, not this
/// library's: a consumer whose basis has a lower highest angular momentum, or
/// whose argument range is narrower, gets a more relevant ranking by saying so.
/// Every one of the values below is printed in the report, so a figure is never
/// read without the protocol that produced it.
///
/// \ingroup boys
struct ProbeOptions {
    /// Arguments per call. The library's own batch entries are built for a
    /// batch of this order, and the timed region has to be long enough to
    /// measure.
    std::size_t count = 1u << 14;

    /// Highest order any argument carries, 1..kMaxBoysOrder.
    int nmax = kMaxBoysOrder;

    /// Lower end of the log-uniform argument range.
    double xLo = 1e-3;

    /// Upper end of the log-uniform argument range.
    double xHi = 40.0;

    /// Seed of the workload's generator, so a run is reproducible from the
    /// report alone.
    std::uint64_t seed = 47;

    /// Timed passes. Every pass is used: no pass is discarded, so the paired
    /// rounds the figures rest on are this number times \c rounds.
    int passes = 5;

    /// Timed rounds per pass. Every option is called once per round, in an order
    /// shuffled per round, and the comparison between two options is made from
    /// their ratio inside one round. The pooled rounds are \c passes * \c rounds,
    /// and a quartile band needs at least four of them.
    int rounds = 5;

    /// Length of the quiet window the background load average is taken over,
    /// in seconds.
    double backgroundWindowSeconds = 1.0;

    /// Length of the calibration window the spin's quiet floor is taken over,
    /// in seconds. The floor is the denominator of the load readings, which are
    /// context; a run whose window never settled still measures, and says that
    /// its load columns have nothing to be relative to.
    double calibrationSeconds = 0.5;

    /// Spread percentage of the canary's own runs across a pass above which the
    /// report marks the pass as one that ran on a wandering machine. **A
    /// diagnostic, and the only thing it decides is a flag.** It admits nothing
    /// and excludes nothing: the canary is a fixed-work spin read by wall clock,
    /// so a machine whose clock decays widens this number with no other process
    /// involved, and a run that discarded on it would discard the measurement
    /// rather than the machine.
    double canarySpreadAlarm = 5.0;

    /// The entry every ratio of this run is formed against, by name, or empty
    /// for the probe's own choice of anchor — which is the library's own default
    /// double-precision all-orders entry.
    ///
    /// The anchor is a choice of the *reporting*, not of the measurement: every
    /// cost column is this row's own cost scaled by its ratio to the anchor, and
    /// the ordering is made of the ratios, so a run anchored on one option and a
    /// run anchored on another measure the same thing and differ only in the
    /// units the absolute columns are printed in and in the order the two
    /// anchors' own figures are divided by. A name this run did not measure falls
    /// back to the default choice and the report says so.
    std::string reference;

    /// Runs of the refinement protocol: how many times the probe measures a
    /// reference class whose own rounds did not separate, before it reads the
    /// vote. Each run is ordered in the figure the report prints, so the vote is
    /// the same question asked again over more rounds; where the runs are split
    /// with no plurality the report says so and names the row the run's own
    /// figures put first.
    int refinementRuns = 5;

    /// How much longer each refinement run is than the pass protocol above: the
    /// refinement re-measures only the tied options, so it can spend more per
    /// option than the run that left them tied. Its passes are \c passes times
    /// this, its rounds \c rounds times this, and the report prints the protocol
    /// every run was taken under.
    int refinementFactor = 5;

    /// The options to measure, named as the report prints them. Empty measures
    /// every option this build offers, which is what a caller who has not
    /// chosen yet wants.
    ///
    /// Naming a set narrows every figure and every conclusion below to that
    /// set: the fastest option reported is then the fastest of the ones asked
    /// for. A name is answered in one of three ways and the report keeps them
    /// apart: an option this build serves is measured, a cell the library refuses
    /// is reported with the library's reason, which is unbuilt work and not a
    /// misspelling, and a name that is neither is reported as no option of this
    /// library, so a typo is never read as a machine on which nothing is fast.
    std::vector<std::string> only;
};

/// One timed pass and the canary readings taken beside it.
///
/// \ingroup boys
struct OptionProbePass {
    /// Wall seconds of the timed rounds, the canary's own runs between them not
    /// counted.
    double seconds = 0.0;

    /// Mean load percentage over the quiet window before the timed region.
    double backgroundLoad = 0.0;

    /// Load percentage of the point reading before the timed region.
    double beforeLoad = 0.0;

    /// Load percentage of the point reading after the timed region.
    double afterLoad = 0.0;

    /// Load percentage of the median canary run taken inside the pass, which is
    /// the load the pass's rounds were taken under.
    double inPassLoad = 0.0;

    /// Spread percentage of the canary's runs across the pass: the slowest run
    /// over the fastest, less one, in percent. **Context, not a gate**: the spin
    /// is fixed work read by wall clock, so this number moves with the clock as
    /// well as with the load.
    double canarySpread = 0.0;

    /// Whether the canary's runs disagreed by more than
    /// ProbeOptions::canarySpreadAlarm. The pass is reported with the flag and
    /// excluded by nothing.
    bool canaryWide = false;

    /// Widest relative width of a within-round paired ratio this pass measured:
    /// the upper quartile of an option's ratio to the reference over the slower
    /// of the pass's rounds, divided by the lower quartile of the same ratio,
    /// less one, in percent, taken at its widest over the options. This is the
    /// spread the ordering is made in — a ratio inside one round, where a drift
    /// common to the round cancels — so it is the paired analogue of the canary
    /// column above, and it is reported beside it rather than gating anything
    /// itself. Zero when the pass held too few rounds for a band to exist.
    double pairedSpread = 0.0;
};

/// The precision an option computes in: the class this probe ranks it in.
///
/// Precision is the choice the caller has already made, from the accuracy their
/// calculation needs, and it is not available to be traded for speed: a lane
/// that halves the precision of the argument and the return is not a faster
/// answer to the double lane's question, it is an answer to a different one. So
/// an option is only ever ordered against an option of its own precision, and
/// the classes below are the whole of that partition.
///
/// The documented bound is a column of each class, not its key. Two options in
/// one class can be documented at different bounds, and the report shows each
/// row's own bound beside it, so a class's leader is the fastest option at *some*
/// accuracy in that precision, never "the fastest at your accuracy".
///
/// \ingroup boys
enum class OptionPrecision : int {
    /// The certified double lane's precision, and the lanes that carry it.
    kFp64 = 0,
    /// Single precision throughout: the argument, the arithmetic and the return.
    kFp32,
    /// Binary16 arguments and returns, computed in single precision.
    kFp16,
    /// Bfloat16 arguments and returns, computed in single precision.
    kBf16,
};

/// The name a report prints a precision class under.
///
/// \param precision the class
/// \returns         a string literal naming it
///
/// \ingroup boys
const char* PrecisionName(OptionPrecision precision) noexcept;

/// The question an option answers: what it hands back to the caller who asks
/// it, which is what makes two options alternatives for one need.
///
/// Two entries are answers to one question when they return the same values for
/// the same arguments. This library's entries are not all one question: the
/// all-orders entry is asked one argument at a time and returns that argument's
/// own ladder, while the all-N entry is asked one call over an array of
/// arguments at one common top order. An option of one shape is not a slower
/// answer to the other shape's question, and the cost this report states is per
/// argument rather than per value, so an ordering across the two would rank the
/// amount of output against the arithmetic.
///
/// The shape is not the entry an option is reached through, and it is not the
/// axes of an evaluation policy, which are interchangeable implementations of
/// the answer the shape asks for and belong inside a class rather than in its
/// key. A caller whose arguments are sorted states it and skips the sort the
/// all-N entry would otherwise pay for — the overload returns the same values,
/// re-deriving the runs from the classification rather than from the
/// declaration, and its documented bound is the other overload's — so that
/// declaration is a way of computing one answer and not a question of its own.
///
/// \ingroup boys
enum class OptionProbeShape : int {
    /// One argument per call, the ladder to that argument's own order: the shape
    /// an integral engine asks for, and the shape this probe's workload is.
    kAllOrders = 0,
    /// One call over many arguments at one common top order: the ladder to the
    /// same order for every argument of the array.
    kAllN,
};

/// The name a report prints a question shape under: "all-orders" or "all-n".
///
/// \param shape the shape to name
///
/// \returns the name, which is never empty
///
/// \ingroup boys
const char* OptionProbeShapeName(OptionProbeShape shape) noexcept;

/// The question a shape asks, said in full: what an option of it produces for a
/// caller, which is what makes two options alternatives or not.
///
/// \param shape the shape to state
///
/// \returns one sentence naming what the caller gets back
///
/// \ingroup boys
const char* OptionProbeShapeQuestion(OptionProbeShape shape) noexcept;

/// One option, as this machine measured it.
///
/// \ingroup boys
struct OptionProbeMeasurement {
    /// The option's name, as the report prints it.
    std::string name;

    /// The precision of the class this option is ranked in, and the first of the
    /// three parts of that class's key.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The axes the option instantiates, as the library reports them: the route
    /// whose fits it reads, the scheme it sums them with, the partition of the
    /// fitted regions it reads, and the packing axis its entry carries. A
    /// defaulted axis is still printed, so a row states the whole combination
    /// rather than the part that differs from a default.
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme; ///< the scheme it sums them with
    FitGranularity granularity = kDefaultFitGranularity; ///< the partition it reads
    PackAxis pack = PackAxis::kArguments; ///< the packing axis its entry carries

    /// The accuracy rung it evaluates at, the second part of its class's key:
    /// two options of one precision at different rungs document different bounds
    /// and are not alternatives to each other.
    AccuracyTier tier = AccuracyTier::kReference;

    /// The question shape it answers, the third part of its class's key: what it
    /// hands back for the caller who asks it. An option is only ever ordered
    /// against options of the same shape, because a faster answer to a different
    /// question is not a faster option.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    /// The arithmetic the option ran in, named by the library's own backend
    /// table (see backend::BoysBackends), so the number is attributable to the
    /// arithmetic that produced it.
    std::string arithmetic;

    /// Whether a bare `a * b + c` in that arithmetic is a single rounding in
    /// this build.
    bool contracts = false;

    /// Whether the option produced a figure on a pass that ran, so the cost and
    /// spread below rest on a measurement. False means the option produced no
    /// figure on this run, and the accuracy below is still the measured one.
    bool measured = false;

    /// Cost per argument in nanoseconds, the figure this option is ordered by.
    /// It is the reference lane's own lower-quartile cost scaled by
    /// \c ratioToReference, which is a central quantile and not this option's own
    /// band; see \c ratioToReference for why that is the fair choice.
    double nsPerArgument = 0.0;

    /// Cost per argument at the upper quartile of the same rounds.
    double nsPerArgumentMax = 0.0;

    /// The fastest single round this option was ever seen in, nanoseconds per
    /// argument. **A peak-clock figure**, kept as one column because it bounds
    /// what the option can do, and never the reported cost: under a decaying
    /// clock the fastest observation comes from the earliest rounds at the
    /// highest clock, which is not what a caller with a long workload meets.
    double nsPerArgumentPeak = 0.0;

    /// Upper quartile over lower quartile: 1.00 is a perfectly repeatable
    /// measurement and a large value is a result, not a nuisance. Both ends are
    /// within-round ratios' worth of cost, so this spread does not carry a clock
    /// drift common to a round.
    double spread = 0.0;

    /// This option's cost as a fraction of the reference lane's, at the **middle**
    /// of the per-round ratios between them: one exactly for the reference lane
    /// itself, and the figure \c nsPerArgument is scaled by.
    ///
    /// The middle, and not the lower quartile of those ratios, because the two
    /// are not the same kind of reading for the reference and for everyone else:
    /// the reference's ratios are one in every round, so any quantile of them is
    /// one, while a rival's lower quartile sits below its middle by a fraction of
    /// its own round-to-round spread — so scoring every row at a lower quartile
    /// would hand the reference a credit no measurement supports and take it from
    /// its rivals. The middle is the quantile that credits the two alike: it is
    /// reciprocal under a change of anchor, element for element over the same
    /// rounds, so this option's figure against the reference and the reference's
    /// figure against this option always agree on which of the two is faster.
    double ratioToReference = 1.0;

    /// The band the middle half of the run put that ratio in, at the lower and
    /// the upper quartile of the same rounds. **The band and not the figure**:
    /// these are where a pair that cannot be separated is read from, and they
    /// bound \c ratioToReference rather than being equal to its ends.
    double ratioLo = 0.0;
    double ratioHi = 0.0; ///< the band's other end, the ratio at its upper quartile

    /// How far this option's ratio to the reference moved between the run's
    /// first and second half of rounds: the second half's median ratio over the
    /// first half's, less one. Zero for the reference lane by construction, and
    /// non-zero for another option only when the option's cost relative to the
    /// lane changed as the run went on — which is what a clock whose decay falls
    /// differently on the two would look like. **A warning, not a correction**:
    /// a large value here says the ordering this option takes part in is not
    /// clock-independent on this run.
    double ratioDrift = 0.0;

    /// Rounds the figures above rest on. Every round of the run is pooled, so
    /// this is the same for every option that produced a figure.
    int rounds = 0;

    /// Largest absolute difference between this option's values and the
    /// certified per-order fp64 lane's value for the same order at the same
    /// argument, over the workload.
    double maxError = 0.0;

    /// The bound this option is judged against, read from the library: the
    /// accuracy rung's own figure for a rung option, the figure the library
    /// documents for the entry the option reaches for a cell, and the published
    /// per-lane bound for the narrower lanes. A measured error at or below it is
    /// the contract being met on this workload. It applies over every argument
    /// the row was measured on, whatever the row's measured range was, so a
    /// verdict does not move with \c --xrange.
    ///
    /// Naming a partition does not change it: a partition replaces the fitted
    /// tables an entry reads and leaves the entry's own figure where it was. The
    /// partition's own figure is over its stored fits' interval and nothing
    /// outside it, and it is not what the measured difference is judged by —
    /// past that interval the entry runs the certified lane's arithmetic, and the
    /// measured column is a difference from that lane rather than an error
    /// against the true function. It is carried in \c ownBound and the row prints
    /// it, with its interval, beside the verdict.
    double bound = 0.0;

    /// The figure the option's own fitted tables are certified at, and the
    /// interval that figure holds on. Zero, with the interval unread, for an
    /// option whose arithmetic is the shipped tables'.
    double ownBound = 0.0;
    double ownLo = 0.0; ///< the left edge of the interval \c ownBound holds on
    double ownHi = 0.0; ///< its right edge, exclusive

    /// Whether every value was bit-identical to the certified lane's, at the
    /// same order and the same argument.
    bool bitIdenticalToReference = false;

    /// Whether the measured error is at or below \c bound.
    bool meetsBound = false;

    /// Whether the measured error is above the bound above but at or below the
    /// certified lane's own documented figure, so the two partitions or rungs
    /// cannot be told apart by this comparison. A row can be honest at its own
    /// bound and still read this state: the column compares against the
    /// certified lane, whose own error the difference carries with it.
    bool withinReferenceFloor = false;

    /// Sum of every value the option returned, so a caller can see that the
    /// timed work ran and ran on the intended arguments.
    double checkedSum = 0.0;
};

/// How the entry a class or a report names was reached.
///
/// The distinction this exists for is the one a reader needs in order to weigh
/// the answer: a measured ordering is evidence, a vote over repeated runs is
/// weaker evidence, and a choice among options the measurement could not separate
/// is not evidence at all. Every one of them names one option, and the report
/// says which one it is rather than presenting them alike.
///
/// \ingroup boys
enum class OptionProbeDefaultHow : int {
    /// Nothing named: no option of the class was measured.
    kNone = 0,
    /// The class's own rounds placed every rival behind its leader, so the entry
    /// is a measured ordering of equals.
    kOrdered,
    /// The class holds one option. There is nothing in it to order the entry
    /// against, so it is named by there being no alternative — which is an
    /// answer, not a refusal, and not a ranking either.
    kOnlyEntry,
    /// The class's rounds left the leader tied with others, the tied options were
    /// re-run alone at a larger protocol, and every one of those runs was fastest
    /// with the entry the report names — the row the run's own figures put first,
    /// measured again, longer, and agreed with.
    kRefined,
    /// The same re-runs, with a majority rather than all of them leading with the
    /// entry the report names: the vote had a plurality and it agreed with the
    /// class's own rounds, which are printed with it.
    kVote,
    /// The class's top entries could not be separated: the vote named another of
    /// them, or was split across several, or no run placed a leader at all. The
    /// entry is the row the report's own figures put first among options it cannot
    /// tell apart — named as that, with the vote and both figures printed, and not
    /// as a ranking.
    kChosenAmongEquals,
};

/// The name of one of those, as one token a script or a report can print beside
/// a default: "ordered", "only-entry", "refined", "vote",
/// "chosen-among-equals" or "none".
///
/// \param how the state to name
///
/// \returns the name, which is never empty
///
/// \ingroup boys
std::string OptionProbeDefaultHowName(OptionProbeDefaultHow how);

/// One class's ranking: the options of one precision at one accuracy rung for
/// one question shape, fastest measured first.
///
/// A class is one precision, one rung of the accuracy axis — the multiplier the
/// option was built at, m = 1, 64, 256 and so on — and one question shape, and
/// it is the only set the probe orders inside: two options are only alternatives
/// if they answer the same question, and an option of another precision, another
/// rung or another shape answers a different one. The keys are the precision,
/// the rung and the shape, never a comparison of documented figures: a figure
/// belongs to one lane, and reading it against another lane's is how a class ends
/// up empty by construction. So the leader is the fastest option *at that
/// accuracy, for that question*, and a caller who needs a different accuracy or
/// is asking a different question reads a different class.
///
/// \ingroup boys
struct OptionProbeClass {
    /// The precision this class is.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The accuracy rung this class is.
    AccuracyTier tier = AccuracyTier::kReference;

    /// The question shape this class is: what every row of it hands back.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    /// The name a report prints it under, which is the class's whole key: the
    /// precision, the rung and the question shape, in that order — "fp64 m=1
    /// all-orders". A reader who sees two of these knows which question each
    /// answer belongs to, and that nothing was compared across them.
    std::string name;

    /// The bound every row of this class documents, read from the library.
    double bound = 0.0;

    /// This class's measured options, fastest first: the first of them is the row
    /// whose cost per argument, the figure the report prints for it, is the
    /// smallest of the class's, so this order is read off that column and not off
    /// another statistic. Empty when nothing of this precision and rung was
    /// measured.
    std::vector<std::string> ranked;

    /// The fastest of them, empty when the class is empty.
    std::string leader;

    /// The leader's cost per argument, nanoseconds.
    double leaderNsPerArgument = 0.0;

    /// The options this class's rounds could not place behind the leader: their
    /// within-round ratio to it did not clear one in the middle half of the
    /// rounds, so the class has no ordering among them and the leader is only
    /// the fastest by the run's own statistic. Empty when the class is ordered.
    std::vector<std::string> unplaced;

    /// Whether every other option of the class was the slower of the two against
    /// the leader in the middle half of the paired rounds, so an order exists
    /// inside this class. False for a class whose run was too short for a band,
    /// whose pairs straddled one, or which holds a single option: one entry is
    /// not a ranking.
    bool ordered = false;

    /// How the entry this class names — its leader — was reached: the class's own
    /// ordering, the vote over a refinement stage, a choice among options the
    /// measurement could not separate, or, for a class of one, the fact that
    /// there was no alternative to name. \c kNone for a class that produced no
    /// measured option.
    OptionProbeDefaultHow how = OptionProbeDefaultHow::kNone;

    /// Members documenting a figure other than the leader's, each named with its
    /// own: a class is keyed on the precision, the rung and the shape, not on a
    /// bound, so a row of the same key that documents a looser figure is in the
    /// class and is reported here rather than being ranked as an equal
    /// silently. Empty when every member of the class documents the leader's own
    /// figure, which is the ordinary case.
    std::vector<std::string> differingBounds;

    /// What the class's ordering rests on, or why it was not made.
    std::string note;
};

/// One cell of the option space this library defines: one combination of the
/// five axes an evaluation policy carries, and whether this build serves it.
///
/// The space is finite and small — two routes by two schemes by two partitions
/// by two packing axes by the accuracy rungs this build can name — so the probe
/// enumerates all of it and states a reason for every cell it does not measure.
/// A cell that is not served is refused by the library, not by the probe: the
/// reason is the library's own, and the work it describes is unbuilt rather
/// than impossible.
///
/// The space is enumerated once per precision lane, because the lanes do not
/// answer alike: the double lane's fits are its own table and the single and
/// half lanes' are another, and a cell the double lane refuses can be one the
/// single-precision engine serves. So a cell is a combination *of a lane's*
/// axes, and the library is asked the carriage question of the lane the cell
/// belongs to.
///
/// \ingroup boys
struct OptionProbeCell {
    /// The cell's name, in the report's own option grammar, so a caller can
    /// name it in ProbeOptions::only and be told what this build does with it.
    std::string name;

    /// The class this cell's option row is ranked in, and the classification
    /// whose book the cell was enumerated for. The two half formats are one lane
    /// in the library and two classes here, so this is what tells their books
    /// apart.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The precision lane whose fit table and whose carriage answer this cell
    /// was enumerated from. The lanes that carry the double lane's arithmetic
    /// report theirs as \c kFp64; the half lanes are one lane at one budget and
    /// are enumerated as \c kFp16.
    Precision lane = Precision::kFp64;

    /// The fit route this cell fixes.
    FitRoute route = kDefaultFitRoute;

    /// The evaluation scheme this cell fixes.
    EvalScheme scheme = kDefaultEvalScheme;

    /// The interval partition this cell fixes.
    FitGranularity granularity = kDefaultFitGranularity;

    /// The packing axis this cell fixes.
    PackAxis pack = PackAxis::kArguments;

    /// The accuracy rung this cell fixes.
    AccuracyTier tier = AccuracyTier::kReference;

    /// Whether this build serves the cell, so a served cell has an option row in
    /// this report unless the run was narrowed by ProbeOptions::only. The
    /// coverage is the library's book rather than the caller's selection, so a
    /// narrowed run still accounts for every cell.
    bool served = false;

    /// Why not, when it does not.
    std::string reason;
};

/// What the probe concluded, and the two ways it can end.
///
/// \ingroup boys
enum class OptionProbeVerdict : int {
    /// The probe named a default: one option of the certified double lane's
    /// precision at the reference rung, with every rival of its class placed
    /// behind it by the class's own rounds or, where they were not, with the
    /// default reached by the vote over the refinement runs. The report's
    /// \c defaultHow says which of those it was, and a reader who needs a
    /// measurement rather than a choice reads the confidence line beside it.
    kRecommend = 0,
    /// The probe declined, which it does for exactly one reason: no option of the
    /// certified double lane's precision at the reference rung was measured at
    /// all, so there is no pool to name a default from — nothing was measured, or
    /// the run was narrowed to a set that holds no such option. A refusal always
    /// says which. Every run that measured the pool ends with a default, whether
    /// or not it could order it.
    kCannotDetermine,
};

/// The refinement stage: the tied options re-measured at a larger protocol, and
/// the vote over those runs.
///
/// This is what the report does instead of naming an entry from a figure counted
/// off the library's tables. It runs only when a class's own rounds left its
/// leader tied with a rival and the class holds more than one option: the tied
/// options alone are measured, each run is a fresh protocol, and the option that
/// led most of the runs is the entry. Where the runs disagree the vote is printed
/// in full, so a reader can see how much of a majority the entry really had.
///
/// \ingroup boys
struct OptionProbeRefinement {
    /// The precision of the class that was refined.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The accuracy rung of the class that was refined, which is the reference
    /// rung: a stage re-runs the pool the default may be taken from, and the
    /// looser rungs are reported from their own measured rounds alone.
    AccuracyTier tier = AccuracyTier::kReference;

    /// The question shape of the class that was refined. A stage refines one
    /// class and no other, so its tied set is the class's own: the options it
    /// re-measures answer the question the class asks, and cannot be read as an
    /// ordering of options that answer a different one.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    /// Whether the stage ran at all: false when the class was ordered by the main
    /// run, when it holds a single option, and when there was no class to refine.
    bool ran = false;

    /// Runs taken, the passes each ran, and the rounds each ran. Zero when the
    /// stage did not run.
    int runs = 0;
    int passes = 0; ///< passes each run took
    int rounds = 0; ///< rounds each pass took

    /// The options re-measured: the class's leader and every option of it the
    /// main run could not place behind the leader.
    std::vector<std::string> pool;

    /// What each run led with, in the order the runs were taken, empty when a run
    /// placed no leader.
    std::vector<std::string> runLeaders;

    /// The tally over those runs: one line per candidate that led at least one
    /// run, in the order the runs first led with it, naming how many of the runs
    /// it won.
    std::vector<std::string> tally;

    /// The option the vote named — the row that led the most runs, ordered in
    /// each run by the same within-round figure the report orders its classes by.
    /// It is what the vote says about the class's fastest row, and the default is
    /// that row whether or not the vote named it: a vote for another entry is this
    /// run saying it could not separate the class's top entries, which is how the
    /// report states it. Never empty once the stage ran: a vote with no plurality
    /// still names one of the tied options, and \c note says that is what
    /// happened.
    std::string winner;

    /// Whether every run led with \c winner: a majority with no dissent.
    bool unanimous = false;

    /// Whether \c winner led strictly more runs than any other candidate. True
    /// whenever \c unanimous is true; false is a split vote whose winner was
    /// taken by choice.
    bool plurality = false;

    /// What the stage concluded, including whether the vote had a plurality and
    /// what was done when it did not.
    std::string note;
};

/// Everything the probe measured and concluded.
///
/// \ingroup boys
struct OptionProbeReport {
    /// The protocol the figures below were taken under.
    ProbeOptions options;

    /// The machine's logical processors, as the standard library reports them.
    int logicalProcessors = 0;

    /// Whether the AVX2 + FMA tier is live on this machine.
    bool avx2 = false;

    /// Wall milliseconds of the fastest spin of the fixed-work canary on this
    /// machine: the floor every load percentage is relative to.
    double canaryFloorMs = 0.0;

    /// Runs of the canary the calibration window fitted in.
    int canaryCalibrationRuns = 0;

    /// Median of those runs, milliseconds. How far this sits above the floor is
    /// what the calibration found about the machine.
    double canaryCalibrationMedianMs = 0.0;

    /// Spread percentage of those runs: the slowest over the fastest, less one.
    /// A wide spread here is a machine that cannot repeat fixed work, and the
    /// probe says so rather than measuring on it.
    double canaryCalibrationSpread = 0.0;

    /// Whether a floor was established at all: the denominator the load
    /// percentages above are relative to. A false here leaves the load columns
    /// with nothing to be relative to, and the report says so; the comparison is
    /// not made in them, so the run still measures and still concludes.
    bool calibrated = false;

    /// The arithmetic backends this build carries, read from the library. The
    /// names the measurements above are attributed to come from this table.
    std::span<const backend::BackendInfo> backends;

    /// The partitions of the fitted regions this build ships, read from the
    /// library. The report prints what each one's tables hold and which cells
    /// its kernels cover.
    std::span<const FitGranularityInfo> granularities;

    /// One entry per option measured, in the order the report prints them.
    std::vector<OptionProbeMeasurement> measurements;

    /// One entry per class measured, in the classes' own order: the reference
    /// rung of each precision first, then the looser rungs of it, each rung's
    /// question shapes beside each other. The ranking inside a class is that
    /// class's own and never crosses into another — not into another precision,
    /// not into another accuracy, and not into another question shape.
    std::vector<OptionProbeClass> classes;

    /// Every cell of the option space, served ones and refused ones alike. The
    /// coverage is the library's, not the caller's selection: it is the same
    /// book whichever set was asked for, so a cell this build does not serve is
    /// never absent from a narrowed run either.
    std::vector<OptionProbeCell> cells;

    /// Options the library offers on other builds but not on this one, because
    /// this build's backend table does not carry the arithmetic they run in.
    std::vector<std::string> unoffered;

    /// Options the library offers on other builds but not on this one, because
    /// the build-time seam that declares them is closed in it. Kept apart from
    /// the unoffered list above, which holds options this build has and cannot
    /// run: the two are different facts about the build, and a caller choosing
    /// an option is told which one it is looking at.
    std::vector<std::string> notCarried;

    /// Names the caller asked for that are no option of this library at all.
    /// Kept apart from the unoffered list above, which holds options the library
    /// has and this build cannot serve: a name here is a misspelling, and
    /// nothing was measured for it.
    std::vector<std::string> notAnOption;

    /// Names the caller asked for that are cells of this library's option space
    /// which the library refuses where they are named, with the reason. Kept
    /// apart from both lists above: a refused cell is neither a misspelling nor
    /// an option some other build carries, and reading it as either would hide
    /// the work it names.
    std::vector<std::string> refused;

    /// Order runs the workload's arguments fall into: the number of calls the
    /// grouped options make for one pass, and the shape the per-argument
    /// options do not see.
    std::size_t orderRuns = 0;

    /// Arguments in the largest run.
    std::size_t largestRun = 0;

    /// One entry per pass run, in order.
    std::vector<OptionProbePass> passes;

    /// Passes whose canary stayed within ProbeOptions::canarySpreadAlarm.
    int passesWithinAlarm = 0;

    /// Passes whose canary wandered further than that. **Reported, and used
    /// anyway**: the flag says the machine's fixed work was not repeating, which
    /// a decaying clock produces by itself. Both of these add up to the passes
    /// run, and every pass contributed to every figure above. Both are zero when
    /// \c calibrated is false: nothing was read, so no pass is placed on either
    /// side of an alarm that was never established.
    int passesAboveAlarm = 0;

    /// Paired rounds the figures rest on: the rounds of every pass, pooled. A
    /// quartile band needs four of them, and the probe refuses rather than
    /// reporting a band it could not form.
    int pairedRounds = 0;

    /// The option every ratio is formed against: the library's default
    /// double-precision entry when the run measured it, else the first measured
    /// option of that precision, else the first measured option.
    std::string referenceOption;

    /// Its own cost per argument, nanoseconds, at the lower quartile of its
    /// rounds. Every cost column above is this figure scaled by an option's
    /// ratio to it, so the reference is the anchor the absolute numbers hang
    /// from and the ratios are what the ordering is made of. The anchor carries
    /// the clock the reference itself ran under; the ratios do not carry a drift
    /// common to a round, which is the point of pairing them.
    ///
    /// **The anchor is a unit and not a participant.** A rank here is one
    /// option's ratio against another's and does not depend on where the anchor
    /// was put; the absolute columns do, and that is all it changes. Which row is
    /// the anchor is \c ProbeOptions::reference.
    double referenceNsPerArgument = 0.0;

    /// Spread percentage of the canary's runs across the widest pass this run
    /// took. **Context, not the gate**: it is what the machine's fixed work did,
    /// and it moves with the clock as well as with the load.
    double canarySpread = 0.0;

    /// Median of the canary's in-pass load readings over the passes, which is
    /// the load the run's rounds were mostly taken under. One number rather than
    /// a column per option, because every option was called in every round: the
    /// pooled rounds put every option under the same sequence of loads, and a
    /// load that is common to a round is what a paired ratio cancels.
    double loadMedian = 0.0;

    /// What this run can order, as a fraction of a cost, measured in the paired
    /// ratios: the widest relative width of a within-round ratio band anything in
    /// the certified double lane's precision showed on this run — an option's own
    /// band against the reference lane, or a rival's band against the class's
    /// leader. It is a statement of how coarse the class's own measurement got,
    /// printed beside the figures; the ordering itself is made pair by pair from
    /// each pair's own band, so this number bars nothing and is zero only when
    /// the class produced no measured option.
    double resolution = 0.0;

    /// The loudest bound the reference option is documented at, read from the
    /// library. This is the floor of the accuracy comparison: two options that
    /// agree to within it cannot be separated by this probe, whichever
    /// precision class they are in.
    double referenceBound = 0.0;

    /// Whether the probe named a winner. A false here means no option of the
    /// certified double lane's precision at the reference rung was measured, so
    /// there was no pool to choose a default from; every other run ends with
    /// \c kRecommend and one option named, whether or not the pool could be
    /// ordered.
    OptionProbeVerdict verdict = OptionProbeVerdict::kCannotDetermine;

    /// **The default combination, and there is exactly one of it whenever the
    /// pool was measured at all.** It is the certified double lane's precision,
    /// at the reference rung, for the all-orders shape — the class whose members
    /// were all built at the library's own full-accuracy multiplier and all
    /// answer the question this probe's workload asks, so choosing among them
    /// trades nothing but speed within one question. A caller asking the other
    /// shape reads that class's own block, which names its fastest option.
    ///
    /// It is **the row the report's own figures put first in that class** —
    /// \c fastestAtReferenceAccuracy — whether the class ordered or the run could
    /// not separate its members: a report whose default were any other row would
    /// be printing a name its own table contradicts. \c defaultHow says how that
    /// row was confirmed, or that the run could not separate it from its rivals.
    /// Empty exactly when \c verdict is \c kCannotDetermine.
    std::string recommended;

    /// How \c recommended was reached. \c kOrdered is a measured ordering;
    /// \c kRefined and \c kVote are the refinement runs naming this same row — a
    /// longer protocol measuring the class's fastest and agreeing with the run;
    /// \c kChosenAmongEquals is a tie this run could not break, including a
    /// refinement vote that named another row, for which the report prints both
    /// figures: the default is the row the run's own figures put first, not the
    /// row the vote preferred. \c kOnlyEntry appears when the caller narrowed the
    /// run to a class of one, which is named by there being no alternative — an
    /// answer, and not a ranking either.
    OptionProbeDefaultHow defaultHow = OptionProbeDefaultHow::kNone;

    /// The fastest option of the certified lane's reference class by the main
    /// run's own rounds alone, empty when no option of that class was measured.
    /// It is the class's winner by the figures this report prints, and so it is
    /// what \c recommended names whenever the class holds more than one measured
    /// option: the refinement vote may confirm that row or fail to, and both are
    /// reported, but it does not name the default in its place.
    std::string fastestAtReferenceAccuracy;

    /// The fastest option measured, whatever its precision, empty when no option
    /// was measured. It is a row to read beside the classes, never across them:
    /// a faster lane of a different precision is faster at a different accuracy,
    /// which is what the classes exist to keep apart.
    std::string fastestOverall;

    /// One sentence saying what the verdict rests on.
    std::string reason;

    /// The options the probe will not order against the recommendation, with
    /// their figures, the band their within-round ratio to the leader fell in,
    /// and in how many rounds each was the slower of the two. Empty when the
    /// recommendation is clear of the field.
    std::vector<std::string> inseparable;

    /// One line naming how far the recommendation can be trusted, built from
    /// the same numbers the verdict is: how far the ratio of the widest-moving
    /// pair of options travelled between the first and second half of the run,
    /// beside the resolution that figure is read against, with a warning when it
    /// went past it. It also says whether every option this comparison put
    /// against another ran the same arithmetic route, since options can draw the
    /// clock differently — a wider vector register is a lower frequency — and two
    /// register widths were then compared. A run that ordered nothing still
    /// carries it, so a reader can tell a clock that wandered from options that
    /// were too close to separate.
    std::string confidence = "not measured";

    /// The refinement stages this run took: one entry per precision whose
    /// reference class the main run left tied, its tied options re-measured at a
    /// larger protocol and voted on. Empty when every reference class was ordered
    /// by the main run or held a single option, which are the cases where nothing
    /// needed refining.
    std::vector<OptionProbeRefinement> refinements;

    /// Whether the report ends with a default for the caller to take: true
    /// exactly when \c recommended names an option.
    bool hasDefault = false;
};

/// Measures every evaluation option this build offers on this machine.
///
/// Enumerates the options from the library rather than from a list: the
/// arithmetic each option runs in is resolved against backend::BoysBackends(),
/// so an option whose arithmetic the build does not carry is reported through
/// OptionProbeReport::unoffered instead of being measured under a name that is
/// not this build's; the relaxed accuracy tiers are found by asking the library
/// what each tier it can name delivers, so a build serving fewer rungs offers
/// fewer options; and the four axes a policy carries are walked over the sets
/// the library reports — BoysFitRoutes and BoysFitRoutesF32, BoysEvalSchemes,
/// BoysFitGranularities and BoysPackAxes — crossed with those rungs and taken
/// once per precision lane, so the option space is the library's own, lane by
/// lane, and the coverage section can account for every cell of it, served or
/// refused.
///
/// Then it runs the pass protocol in ProbeOptions: every pass carries runs of the
/// fixed-work canary beside its rounds, each round calls every option once in an
/// order shuffled per round, and the report carries, per option, its cost and the
/// spread beside it, its ratio to the reference lane with the band it fell in,
/// the drift of that ratio between the run's halves, the resolution the paired
/// ratios support, and the canary and load readings taken along the way as
/// context. No pass is discarded on the canary's word: a fixed work read by wall
/// clock measures the clock as much as the load, so the canary is reported and
/// never gates.
///
/// The entry allocates (the workload, the samples and the report) and is not a
/// hot path; it is meant to be called once, from a program the consumer builds
/// and runs where they deploy. It is not noexcept, unlike the kernel entries:
/// an allocation failure is a real outcome here and there is no meaningful
/// result to return instead.
///
/// \param options the workload and the pass protocol; the defaults are the ones
///                the preamble describes
///
/// \returns the report, which names one default combination whenever the
///          certified lane's reference class was measured at all — by its own
///          ordering, by the vote over the refinement runs, or as a declared
///          choice among options that could not be separated
///
/// \ingroup boys
OptionProbeReport RunOptionProbe(const ProbeOptions& options = {});

/// The report as the text a consumer reads, machine statement and all.
///
/// The wording is the report's own: it is plain text, it names this machine,
/// and it says what was discarded and why. It is written for a reader who has
/// nothing but this output.
///
/// \param report a report, from RunOptionProbe
///
/// \returns the report as text, newline-terminated
///
/// \ingroup boys
std::string FormatOptionProbe(const OptionProbeReport& report);

} // namespace boys
