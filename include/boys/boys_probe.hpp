#pragma once

/// \file boys_probe.hpp
/// The option probe: which of this library's evaluation options is fastest on
/// the machine the caller actually runs on, and whether the measurement is
/// strong enough to name one at all.
///
/// Which option wins is a property of the machine and the build, not of the
/// library: a ranking printed in documentation would be a claim about the
/// machine it was written on. This entry measures the ranking where it is called
/// and reports what it found, including how confident it is.
///
/// Options are ranked only inside a class: one precision, one accuracy rung and
/// one question shape. Everything else an option varies — fit route, scheme,
/// region partition, packing axis, division form, and whether a sorted array is
/// declared — is an interchangeable way of computing one answer, so it competes
/// inside the class and never divides it. The default this report names is the
/// fastest measured row of the certified double lane's reference class for the
/// all-orders shape.
///
/// Timing is paired: every option is called once in every round, and two options
/// are compared through the ratio of their times within one round, so a clock
/// that drifts or a load that wanders cancels rather than biasing the ranking.
///
/// **This result is about the machine it was measured on**, which the report
/// says in its own output rather than only here.
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
/// library's. Every value below is printed in the report, so a figure is never
/// read without the protocol that produced it.
///
/// \ingroup boys
struct ProbeOptions {
    /// Arguments per call: the timed region has to be long enough to measure.
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
    /// shuffled per round; a quartile band needs at least four pooled rounds.
    int rounds = 5;

    /// Length of the quiet window the background load average is taken over,
    /// in seconds.
    double backgroundWindowSeconds = 1.0;

    /// Length of the calibration window the spin's quiet floor is taken over, in
    /// seconds. The floor is the denominator of the load readings, which are
    /// context.
    double calibrationSeconds = 0.5;

    /// Spread percentage of the canary's own runs across a pass above which the
    /// report marks the pass as one that ran on a wandering machine. **A
    /// diagnostic, and the only thing it decides is a flag**: the canary is a
    /// fixed-work spin read by wall clock, so a decaying clock widens this number
    /// with no other process involved.
    double canarySpreadAlarm = 5.0;

    /// The entry every ratio of this run is formed against, by name, or empty
    /// for the probe's own choice of anchor, which is the library's own default
    /// double-precision all-orders entry. A name this run did not measure falls
    /// back to that choice and the report says so.
    ///
    /// The anchor is a choice of the *reporting*, not of the measurement: the
    /// ordering is made of the ratios, so a run anchored one way and a run
    /// anchored another differ only in the units the absolute columns are
    /// printed in.
    std::string reference;

    /// Runs of the refinement protocol: how many times the probe measures a class
    /// whose own rounds did not separate, before it reads the vote.
    int refinementRuns = 5;

    /// How much longer each refinement run is than the pass protocol above: the
    /// refinement re-measures only the tied options, so it can spend more per
    /// option. Its passes and rounds are \c passes and \c rounds times this.
    int refinementFactor = 5;

    /// The options to measure, named as the report prints them. Empty measures
    /// every option this build offers.
    ///
    /// Naming a set narrows every figure and every conclusion to that set. A name
    /// is answered in one of three ways, and the report keeps them apart: an
    /// option this build serves is measured, a cell the library refuses is
    /// reported with the library's reason, and a name that is neither is reported
    /// as no option of this library — so a typo is never read as a machine on
    /// which nothing is fast.
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

    /// Spread percentage of the canary's runs across the pass. **Context, not a
    /// gate**: the spin is fixed work read by wall clock, so this number moves
    /// with the clock as well as with the load.
    double canarySpread = 0.0;

    /// Whether the canary's runs disagreed by more than
    /// ProbeOptions::canarySpreadAlarm. The pass is reported with the flag and
    /// excluded by nothing.
    bool canaryWide = false;

    /// Widest relative width of a within-round paired ratio this pass measured,
    /// at the widest over the options, in percent. This is the spread the
    /// ordering is made in, reported beside the canary column rather than gating
    /// anything. Zero when the pass held too few rounds for a band.
    double pairedSpread = 0.0;
};

/// The precision an option computes in: the class this probe ranks it in.
///
/// Precision is the choice the caller has already made, not one traded for
/// speed: an option is only ever ordered against an option of its own precision.
/// The documented bound is a column of each class rather than its key, so a
/// class's leader is the fastest option at *some* accuracy in that precision.
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
    /// The device lane's single precision: the arithmetic of a single-precision
    /// call run on a CUDA device, with the lane's own figure beside it. Its
    /// entries are the ones a host without such a device cannot run.
    kFp32Device,
};

/// The name a report prints a precision class under, and never null.
///
/// \param precision the class
/// \returns         a string literal naming it: "fp64", "fp32", "fp16", "bf16"
///                  or "fp32-device", and "unknown" for a value outside the
///                  enumerators
///
/// A value outside the enumerators is answered rather than refused, so
/// "unknown" is not a class the probe measured.
///
/// \ingroup boys
const char* PrecisionName(OptionPrecision precision) noexcept;

/// The question an option answers: what it hands back to the caller who asks
/// it, which is what makes two options alternatives for one need.
///
/// Two entries are answers to one question when they return the same values for
/// the same arguments. This library's entries are not all one question: the
/// all-orders entry is asked one argument at a time and returns that argument's
/// own ladder, while the all-N entry is asked one call over an array at one
/// common top order. An option of one shape is not a slower answer to the other
/// shape's question.
///
/// The shape is not the entry an option is reached through, and not the axes of
/// an evaluation policy — those are interchangeable implementations of the
/// answer the shape asks for, and belong inside a class rather than in its key.
/// A caller whose arguments are sorted may say so and skip the all-N entry's
/// sort, which is another way of computing the same answer, not another
/// question.
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
/// \returns the name, which is never empty: one of those two, and "unknown"
///          for a value outside the enumerators
///
/// "unknown" is not a shape the probe ranked.
///
/// \ingroup boys
const char* OptionProbeShapeName(OptionProbeShape shape) noexcept;

/// The question a shape asks, said in full: what an option of it produces for a
/// caller, which is what makes two options alternatives or not.
///
/// \param shape the shape to state
///
/// \returns one sentence naming what the caller gets back, and "an unknown
///          question" for a value outside the enumerators
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

    /// The axes the option instantiates, as the library reports them. A
    /// defaulted axis is still printed, so a row states the whole combination
    /// rather than the part that differs from a default.
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme; ///< the scheme it sums them with
    FitGranularity granularity = kDefaultFitGranularity; ///< the partition it reads
    PackAxis pack = PackAxis::kArguments; ///< the packing axis its entry carries
    DivisionForm division = kDefaultDivisionForm; ///< the form its steps divide in

    /// The accuracy rung it evaluates at, the second part of its class's key:
    /// two options of one precision at different rungs document different bounds
    /// and are not alternatives to each other.
    AccuracyTier tier = AccuracyTier::kReference;

    /// The question shape it answers, the third part of its class's key. An
    /// option is only ordered against options of the same shape.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    /// The arithmetic the option ran in, named by the library's own backend
    /// table (see backend::BoysBackends).
    std::string arithmetic;

    /// Whether a bare `a * b + c` in that arithmetic is a single rounding in
    /// this build.
    bool contracts = false;

    /// Whether the option produced a figure on a pass that ran. False means no
    /// figure was produced, and the accuracy below is still the measured one.
    bool measured = false;

    /// Cost per argument in nanoseconds, the figure this option is ordered by:
    /// the reference lane's own lower-quartile cost scaled by
    /// \c ratioToReference.
    double nsPerArgument = 0.0;

    /// Cost per argument at the upper quartile of the same rounds.
    double nsPerArgumentMax = 0.0;

    /// The fastest single round this option was ever seen in, nanoseconds per
    /// argument. **A peak-clock figure**, kept because it bounds what the option
    /// can do, and never the reported cost.
    double nsPerArgumentPeak = 0.0;

    /// Upper quartile over lower quartile: 1.00 is a perfectly repeatable
    /// measurement. Both ends are within-round ratios' worth of cost, so this
    /// spread does not carry a clock drift common to a round.
    double spread = 0.0;

    /// This option's cost as a fraction of the reference lane's, at the **middle**
    /// of the per-round ratios between them: one exactly for the reference lane
    /// itself, and the figure \c nsPerArgument is scaled by.
    ///
    /// The middle, and not a lower quartile, because the reference's ratios are
    /// one in every round while a rival's lower quartile sits below its middle by
    /// a fraction of its own spread; the middle is the quantile that credits the
    /// two alike, and it is reciprocal under a change of anchor.
    double ratioToReference = 1.0;

    /// The band the middle half of the run put that ratio in, at its lower and
    /// upper quartile. **The band and not the figure**: these are where a pair
    /// that cannot be separated is read from.
    double ratioLo = 0.0;
    double ratioHi = 0.0; ///< the band's other end, the ratio at its upper quartile

    /// How far this option's ratio to the reference moved between the run's first
    /// and second half of rounds: the second half's median ratio over the first
    /// half's, less one. Zero for the reference lane by construction. **A
    /// warning, not a correction**: a large value says the ordering this option
    /// takes part in is not clock-independent on this run.
    double ratioDrift = 0.0;

    /// Rounds the figures above rest on. Every round of the run is pooled, so
    /// this is the same for every option that produced a figure.
    int rounds = 0;

    /// Largest absolute difference between this option's values and the
    /// certified per-order fp64 lane's value for the same order at the same
    /// argument, over the workload.
    double maxError = 0.0;

    /// The bound this option is judged against, read from the library. A measured
    /// error at or below it is the contract being met on this workload. It applies
    /// over every argument the row was measured on, so a verdict does not move
    /// with \c --xrange.
    ///
    /// Naming a partition does not change it: a partition replaces the fitted
    /// tables an entry reads and leaves the entry's own figure where it was. The
    /// partition's own figure is over its stored fits' interval and nothing
    /// outside it, so it is carried in \c ownBound rather than being what the
    /// measured difference is judged by.
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

    /// Whether the measured error is above \c bound but at or below the certified
    /// lane's own documented figure, so the two cannot be told apart by this
    /// comparison. The column compares against the certified lane, whose own
    /// error the difference carries with it.
    bool withinReferenceFloor = false;

    /// Sum of every value the option returned, so a caller can see that the
    /// timed work ran and ran on the intended arguments.
    double checkedSum = 0.0;
};

/// How the entry a class or a report names was reached.
///
/// The distinction exists so a reader can weigh the answer: a measured ordering
/// is evidence, a vote over repeated runs is weaker evidence, and a choice among
/// options the measurement could not separate is not evidence at all.
///
/// \ingroup boys
enum class OptionProbeDefaultHow : int {
    /// Nothing named: no option of the class was measured.
    kNone = 0,
    /// The class's own rounds placed every rival behind its leader, so the entry
    /// is a measured ordering of equals.
    kOrdered,
    /// The class holds one option, named by there being no alternative to it —
    /// which is an answer and not a ranking either.
    kOnlyEntry,
    /// The tied options were re-run alone at a larger protocol, and every one of
    /// those runs was fastest with the entry the report names.
    kRefined,
    /// The same re-runs, with a majority rather than all of them leading with the
    /// entry the report names.
    kVote,
    /// The class's top entries could not be separated: the entry is the row the
    /// report's own figures put first among options it cannot tell apart, named
    /// as that and not as a ranking.
    kChosenAmongEquals,
};

/// The name of one of those, as one token a script or a report can print beside
/// a default: "ordered", "only-entry", "refined", "vote",
/// "chosen-among-equals" or "none".
///
/// \param how the state to name
///
/// \returns the name, which is never empty: one of those six tokens
///
/// "none" carries both \c kNone and a value outside the enumerators, so a caller
/// that has to tell "nothing was measured" from "this is not a state" tests the
/// value against the enumerators itself.
///
/// \ingroup boys
std::string OptionProbeDefaultHowName(OptionProbeDefaultHow how);

/// One class's ranking: the options of one precision at one accuracy rung for
/// one question shape, fastest measured first.
///
/// A class is the only set the probe orders inside, and its key is the precision,
/// the rung and the shape — never a comparison of documented figures, since a
/// figure belongs to one lane. So the leader is the fastest option *at that
/// accuracy, for that question*.
///
/// \ingroup boys
struct OptionProbeClass {
    /// The precision this class is.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The accuracy rung this class is.
    AccuracyTier tier = AccuracyTier::kReference;

    /// The question shape this class is: what every row of it hands back.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    /// The class's whole key, as a report prints it: the precision, the rung and
    /// the question shape, in that order — "fp64 m=1 all-orders".
    std::string name;

    /// The bound every row of this class documents, read from the library.
    double bound = 0.0;

    /// This class's measured options, fastest first by cost per argument. Empty
    /// when nothing of this precision and rung was measured.
    std::vector<std::string> ranked;

    /// The fastest of them, empty when the class is empty.
    std::string leader;

    /// The leader's cost per argument, nanoseconds.
    double leaderNsPerArgument = 0.0;

    /// The options this class's rounds could not place behind the leader. Empty
    /// when the class is ordered.
    std::vector<std::string> unplaced;

    /// Whether every other option of the class was the slower of the two against
    /// the leader in the middle half of the paired rounds. False for a class whose
    /// run was too short for a band, whose pairs straddled one, or which holds a
    /// single option: one entry is not a ranking.
    bool ordered = false;

    /// How the entry this class names — its leader — was reached.
    /// \c kNone for a class that produced no measured option.
    OptionProbeDefaultHow how = OptionProbeDefaultHow::kNone;

    /// Members documenting a figure other than the leader's, each named with its
    /// own: a class is keyed on the precision, the rung and the shape, not on a
    /// bound, so such a row is in the class and is reported here rather than
    /// ranked as an equal silently.
    std::vector<std::string> differingBounds;

    /// What the class's ordering rests on, or why it was not made.
    std::string note;
};

/// One cell of the option space this library defines: one combination of the
/// axes an evaluation policy carries, and whether this build serves it.
///
/// A cell that is not served is refused by the library, not by the probe: the
/// reason is the library's own, and the work it describes is unbuilt rather than
/// impossible.
///
/// The space is enumerated once per precision lane, because the lanes do not
/// answer alike: a cell the double lane refuses can be one the single-precision
/// engine serves, so the carriage question is asked of the lane the cell belongs
/// to.
///
/// \ingroup boys
struct OptionProbeCell {
    /// The cell's name, in the report's own option grammar, so a caller can
    /// name it in ProbeOptions::only and be told what this build does with it.
    std::string name;

    /// The class this cell's option row is ranked in, and the classification
    /// whose book the cell was enumerated for. The two half formats are one lane
    /// in the library and two classes here, so this tells their books apart.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The precision lane whose fit table and whose carriage answer this cell was
    /// enumerated from. The half lanes are one lane at one budget, enumerated as
    /// \c kFp16.
    Precision lane = Precision::kFp64;

    /// The fit route this cell fixes.
    FitRoute route = kDefaultFitRoute;

    /// The evaluation scheme this cell fixes.
    EvalScheme scheme = kDefaultEvalScheme;

    /// The interval partition this cell fixes.
    FitGranularity granularity = kDefaultFitGranularity;

    /// The packing axis this cell fixes.
    PackAxis pack = PackAxis::kArguments;

    /// The division form this cell fixes. Every member is served at every cell
    /// of the other axes, so this one never decides carriage; it is part of the
    /// cell because it is part of the combination an option runs.
    DivisionForm division = kDefaultDivisionForm;

    /// The accuracy rung this cell fixes.
    AccuracyTier tier = AccuracyTier::kReference;

    /// Whether this build serves the cell, so a served cell has an option row in
    /// this report unless the run was narrowed by ProbeOptions::only, or the cell
    /// belongs to the device lane: that lane's entries need a CUDA device, so its
    /// cells are counted here and measured nowhere. The coverage is the library's
    /// book, so a narrowed run still accounts for every cell.
    bool served = false;

    /// Why not, when it does not.
    std::string reason;
};

/// What the probe concluded, and the two ways it can end.
///
/// \ingroup boys
enum class OptionProbeVerdict : int {
    /// The probe named a default: one option of the certified double lane's
    /// precision at the reference rung. The report's \c defaultHow says whether
    /// the class's own rounds placed it or a refinement vote confirmed it.
    kRecommend = 0,
    /// The probe declined, for exactly one reason: no option of the certified
    /// double lane's precision at the reference rung was measured at all, so there
    /// was no pool to name a default from. A refusal always says which.
    kCannotDetermine,
};

/// The refinement stage: the tied options re-measured at a larger protocol, and
/// the vote over those runs.
///
/// It runs only when a class's own rounds left its leader tied with a rival and
/// the class holds more than one option. Where the runs disagree the vote is
/// printed in full.
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
    /// class and no other, so its tied set is the class's own.
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
    /// The default is that row whether or not the vote named it: a vote for
    /// another entry is this run saying it could not separate the class's top
    /// entries. Never empty once the stage ran.
    std::string winner;

    /// Whether every run led with \c winner: a majority with no dissent.
    bool unanimous = false;

    /// Whether \c winner led strictly more runs than any other candidate. True
    /// whenever \c unanimous is true; false is a split vote.
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
    /// A wide spread here is a machine that cannot repeat fixed work.
    double canaryCalibrationSpread = 0.0;

    /// Whether a floor was established at all: the denominator the load
    /// percentages above are relative to. False leaves them with nothing to be
    /// relative to, and the report says so; the comparison is not made in them, so
    /// the run still measures.
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

    /// One entry per class measured, in the classes' own order. The ranking inside
    /// a class never crosses into another — not another precision, not another
    /// accuracy, and not another question shape.
    std::vector<OptionProbeClass> classes;

    /// Every cell of the option space this machine measures, served ones and
    /// refused ones alike. The coverage is the library's, not the caller's
    /// selection, so a cell this build does not serve is never absent from a
    /// narrowed run either.
    std::vector<OptionProbeCell> cells;

    /// The device lane's book: every cell of that class, counted here and
    /// measured nowhere, because an entry of that lane is a call on a CUDA device
    /// and this is a host build. It is kept apart from \c cells so that a served
    /// cell of that vector is one this build ran, or one the caller's selection
    /// narrowed away, and never one no body here could run.
    std::vector<OptionProbeCell> deviceCells;

    /// Options the library offers on other builds but not on this one, because
    /// this build's backend table does not carry the arithmetic they run in.
    std::vector<std::string> unoffered;

    /// Options the library offers on other builds but not on this one, because
    /// the build-time seam that declares them is closed in it. Kept apart from
    /// \c unoffered, which holds options this build has and cannot run.
    std::vector<std::string> notCarried;

    /// Names the caller asked for that are no option of this library at all — a
    /// misspelling, for which nothing was measured. Kept apart from the lists
    /// above.
    std::vector<std::string> notAnOption;

    /// Names the caller asked for that are cells of this library's option space
    /// which the library refuses where they are named, with the reason. Kept apart
    /// from both lists above: reading it as either would hide the work it names.
    std::vector<std::string> refused;

    /// Order runs the workload's arguments fall into: the number of calls the
    /// grouped options make for one pass.
    std::size_t orderRuns = 0;

    /// Arguments in the largest run.
    std::size_t largestRun = 0;

    /// One entry per pass run, in order.
    std::vector<OptionProbePass> passes;

    /// Passes whose canary stayed within ProbeOptions::canarySpreadAlarm.
    int passesWithinAlarm = 0;

    /// Passes whose canary wandered further than that. **Reported, and used
    /// anyway**. Both counts add up to the passes run, and every pass contributed
    /// to every figure above. Both are zero when \c calibrated is false.
    int passesAboveAlarm = 0;

    /// Paired rounds the figures rest on: the rounds of every pass, pooled. A
    /// quartile band needs four of them, and the probe refuses rather than
    /// reporting a band it could not form.
    int pairedRounds = 0;

    /// The option every ratio is formed against: the library's default
    /// double-precision entry when the run measured it, else the first measured
    /// option of that precision, else the first measured option.
    std::string referenceOption;

    /// Its own cost per argument, nanoseconds, at the lower quartile of its rounds.
    /// Every cost column above is this figure scaled by an option's ratio to it.
    ///
    /// **The anchor is a unit and not a participant**: a rank is one option's
    /// ratio against another's and does not depend on where the anchor was put;
    /// the absolute columns do, and that is all it changes. Which row is the anchor
    /// is \c ProbeOptions::reference.
    double referenceNsPerArgument = 0.0;

    /// Spread percentage of the canary's runs across the widest pass this run took.
    /// **Context, not the gate**: it moves with the clock as well as with the
    /// load.
    double canarySpread = 0.0;

    /// Median of the canary's in-pass load readings over the passes. One number
    /// rather than a column per option, because every option was called in every
    /// round, so a load common to a round cancels in a paired ratio.
    double loadMedian = 0.0;

    /// What this run can order, as a fraction of a cost, measured in the paired
    /// ratios: the widest relative width of a within-round ratio band anything in
    /// the certified double lane's precision showed on this run. The ordering is
    /// made pair by pair from each pair's own band, so this number bars nothing.
    double resolution = 0.0;

    /// The loudest bound the reference option is documented at, read from the
    /// library: the floor of the accuracy comparison.
    double referenceBound = 0.0;

    /// Whether the probe named a winner. False means no option of the certified
    /// double lane's precision at the reference rung was measured, so there was no
    /// pool to choose a default from.
    OptionProbeVerdict verdict = OptionProbeVerdict::kCannotDetermine;

    /// **The default combination, and there is exactly one of it whenever the pool
    /// was measured at all.** It is the certified double lane's precision, at the
    /// reference rung, for the all-orders shape — a class whose members are all
    /// built at the library's full-accuracy multiplier, so choosing among them
    /// trades nothing but speed within one question.
    ///
    /// It is **the row the report's own figures put first in that class** —
    /// \c fastestAtReferenceAccuracy — whether the class ordered or the run could
    /// not separate its members. \c defaultHow says how that row was confirmed.
    /// Empty exactly when \c verdict is \c kCannotDetermine.
    std::string recommended;

    /// How \c recommended was reached. \c kOrdered is a measured ordering;
    /// \c kRefined and \c kVote are the refinement runs naming this same row;
    /// \c kChosenAmongEquals is a tie this run could not break, including a
    /// refinement vote that named another row — the default is the row the run's
    /// own figures put first, not the row the vote preferred. \c kOnlyEntry
    /// appears when the caller narrowed the run to a class of one.
    OptionProbeDefaultHow defaultHow = OptionProbeDefaultHow::kNone;

    /// The members of an axis that the class behind \c recommended held no row
    /// for, one line per axis, empty when that class covered every axis whole.
    ///
    /// **A class that names a default for an axis is a class that compared that
    /// axis's members**, and a member no row of the class reached was not in the
    /// comparison. This list is what says so, rather than letting a default read
    /// as a race that was never run: the axis's name, and the members the class
    /// held no row for. The default is still named - the run measured what it
    /// measured - but the report does not present it as chosen among equals it
    /// never met.
    std::vector<std::string> defaultClassAbsent;

    /// The fastest option of the certified lane's reference class by the main run's
    /// own rounds alone, empty when no option of that class was measured. It is what
    /// \c recommended names whenever the class holds more than one measured
    /// option.
    std::string fastestAtReferenceAccuracy;

    /// The fastest option measured, whatever its precision, empty when none was
    /// measured. A row to read beside the classes, never across them.
    std::string fastestOverall;

    /// One sentence saying what the verdict rests on.
    std::string reason;

    /// The options the probe will not order against the recommendation, with their
    /// figures, the band their ratio to the leader fell in, and in how many rounds
    /// each was the slower of the two.
    std::vector<std::string> inseparable;

    /// One line naming how far the recommendation can be trusted, built from the
    /// same numbers the verdict is: how far the widest-moving pair's ratio
    /// travelled between the run's halves, beside the resolution that figure is
    /// read against, with a warning when it went past it. It also says whether the
    /// options compared ran the same arithmetic route, since register widths draw
    /// the clock differently.
    std::string confidence = "not measured";

    /// The refinement stages this run took: one entry per precision whose reference
    /// class the main run left tied. Empty when every reference class was ordered by
    /// the main run or held a single option.
    std::vector<OptionProbeRefinement> refinements;

    /// Whether the report ends with a default for the caller to take: true
    /// exactly when \c recommended names an option.
    bool hasDefault = false;
};

/// Measures every evaluation option this build offers on this machine.
///
/// The options are enumerated from the library rather than from a list: the
/// arithmetic each option runs in is resolved against backend::BoysBackends(), so
/// one whose arithmetic the build does not carry is reported through
/// OptionProbeReport::unoffered; the rungs are found by asking the library what
/// each tier it can name delivers; and the axes a policy carries are walked over
/// the sets the library reports, once per precision lane, so the coverage section
/// can account for every cell of the space, served or refused.
///
/// The pass protocol in ProbeOptions is then run: every pass carries runs of the
/// fixed-work canary beside its rounds, and each round calls every option once in
/// an order shuffled per round. No pass is discarded on the canary's word: a
/// fixed work read by wall clock measures the clock as much as the load, so the
/// canary is reported and never gates.
///
/// The entry allocates (the workload, the samples and the report) and is not a
/// hot path; it is meant to be called once. It is not noexcept, unlike the kernel
/// entries: an allocation failure is a real outcome here.
///
/// \param options the workload and the pass protocol
///
/// \returns the report, which names one default combination whenever the
///          certified lane's reference class was measured at all
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
