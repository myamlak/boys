#pragma once

/// \file boys_probe.hpp
/// The option probe: which of this library's evaluation options is fastest
/// where the caller actually runs it, and whether the measurement is strong
/// enough to name one at all.
///
/// **Why this is a library entry and not a benchmark.** Which option wins is a
/// property of the machine and the build, not of the library. The same entry
/// that is fastest on one host can lose on another: the vector tier is present
/// on x86_64 and absent elsewhere; a bare `a * b + c` is one rounding or two
/// depending on the target and the flags, so an arithmetic with more multiply
/// adds can be the cheap one on a host that fuses and the expensive one on a
/// host that calls out; and a lane that halves the precision of the argument
/// trades accuracy for cost by an amount that depends on how the consumer
/// builds. A ranking printed in documentation is therefore a claim about the
/// machine it was written on. This entry measures the ranking on the machine it
/// is called on and reports what it found, including how confident it is.
///
/// **What it measures.** Every option is asked the library's own question:
/// a set of arguments in which each argument carries its own highest order, and
/// every order from 0 up to it evaluated at that argument — one seed, then the
/// recursion up to that argument's order, which is the shape an integral engine
/// asks for. The arguments are log-uniform over a range that populates every
/// region of the kernel, and the per-argument orders are drawn as the sum of two
/// shell angular momenta, which is the distribution a shell-pair batch presents.
/// A benchmark that measured a shape the library never runs would rank the
/// options by the wrong thing.
///
/// **What it reports per option**: the cost per argument, the spread of that
/// cost over the rounds it was measured in, its ratio to the reference lane, the
/// machine load the rounds were taken under, the axes the option instantiates,
/// and the accuracy the option actually delivered — so a consumer can see whether
/// a faster option was faster at the same accuracy or merely at a lower one.
///
/// **The whole option space, enumerated from the library.** The library lets a
/// caller choose a fit route, an evaluation scheme, a partition of the fitted
/// regions and a packing axis, and each of those choices is a member of a
/// reported set (BoysFitRoutes, BoysEvalSchemes, BoysFitGranularities,
/// BoysPackAxes) crossed with the accuracy rungs the build can name. The probe
/// walks that product rather than a list written beside it, so a member this
/// change did not think of, or one a later change adds, is measured without the
/// probe being edited. A cell the library does not serve is refused where it is
/// named, and the probe states every refused cell with the library's reason, in
/// a coverage section that accounts for the whole product — an unstated
/// omission would read as an option that does not exist.
///
/// **The classes are precisions, and a bound is a column.** fp64, fp32, fp16 and
/// bf16 are separate classes and options are ordered only inside one of them:
/// the precision is the choice the caller has already made from the accuracy
/// their calculation needs, and halving the precision is not a faster answer to
/// the same question. Bounds inside a class differ by row — a relaxed rung, the
/// across-orders lane and the narrow partition are all documented at their own
/// figures — so a class's leader is the fastest option at *some* accuracy in
/// that precision, and the report says so where it prints the ranking. A caller
/// who needs a particular accuracy reads the bound column.
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
/// reported cost of an option is the lower quartile of its per-round costs, and
/// its spread is the ratio of the upper quartile to that. For each rival of a
/// class's leader the probe forms the pair's own within-round ratio, and the
/// rival is ordered only when the middle half of those rounds puts it behind the
/// leader — when the pair's own band clears one. A rival whose band straddles
/// one cannot be placed, and the probe then declines to name a winner, prints
/// the run's resolution, and says which options it could not separate and in how
/// many of the rounds each was the slower of the two. The same refusal happens
/// when too few rounds were run for a band to exist, and when nothing was
/// measured in the class. A wrong recommendation is worse than none, so the
/// refusal path is the one this entry is most careful about, and a refusal now
/// leaves a consumer a default: see the fallback below.
///
/// **The instrument is a diagnostic and gates nothing.** Every pass carries runs
/// of a fixed-work integer spin — the canary — bracketing the timed region and
/// taken between its rounds. The spin holds no floating-point state, so the
/// arithmetic question this probe is about cannot change the instrument's own
/// cost. Its readings are reported beside each pass, and a pass whose spin
/// disagreed with itself by more than ProbeOptions::canarySpreadAlarm is marked
/// as one that ran on a wandering machine — **and is still used**. It has to be:
/// a fixed-work spin measured by wall clock measures the clock as much as the
/// load, so a decaying clock widens the spin's own spread on a machine that is
/// doing nothing else, and a rule that discards on that spread discards the
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
/// **When the measurement cannot separate the options, a heuristic stands in,
/// labelled as one.** A consumer who gets a refusal still needs a default, so a
/// refusal also names the option a static reading of the library's own tables
/// picks: the degree the option's partition stores per piece and the coefficients
/// it stores, counted rather than timed. It is reported in its own section,
/// marked as a heuristic, and never printed beside a measured figure as though it
/// were one — the two answers are different kinds of thing and the report keeps
/// them apart.
///
/// **The accuracy column.** Each option's values are compared against the
/// certified per-order fp64 lane — `BoysSingle` at the reference multiplier —
/// evaluated at the order and the argument the option was actually asked about.
/// That last clause matters for the lanes that narrow the argument: a
/// single-precision lane rounds `x` before it evaluates, so holding it to the
/// double-precision argument would measure the representation of the argument
/// rather than the arithmetic of the lane. It matters for another reason for
/// the batch entries: a batch seeds region A at its own highest order, so its
/// F_k for k below that is a recurrence's value rather than a per-order walk's,
/// and the difference shows up here as the small, bounded error it is instead
/// of being hidden by comparing a batch against a batch. The comparison has a
/// floor: the certified lane is itself documented at a bound, that bound is
/// read from the library and printed, and a difference at or below it means the
/// two are indistinguishable at this comparison's resolution — not that the
/// option is proven to that bound, which is what the committed reference grid
/// and the test suite are for. A row whose measured difference is above its own
/// bound but at or below that floor is reported in a third state rather than as
/// a failure: the difference between two partitions or two rungs carries the
/// certified lane's own error with it, and that error is what the floor is.
///
/// **What this probe does not measure**, and why. The native half lane is left
/// out: it covers region C only, its orders are useful up to 8, and its results
/// are scaled by a power of two, so its cost is not the cost of the same
/// question and a table that put it beside the others would compare different
/// work. The region-A matrix-product transform is left out: it computes region A
/// alone, so measuring it means composing the rest of an evaluation, and that
/// composition would be measured rather than the entry. The CUDA lanes are left
/// out: they are a separate optional build that needs a device, and the options
/// this probe ranks are the CPU ones the caller's own build carries.
///
/// What the option space itself refuses is not left out but named: the cells
/// the library cannot serve are listed with their reasons in the report's
/// coverage section, counted rather than omitted. They are the narrow partition
/// crossed with the rational route (which carries no narrow table), with the
/// across-orders packing axis (whose kernel needs the shipped partition's shared
/// piece shape), and with a relaxed accuracy rung (the narrow degrees are the
/// shipped rung's); and the partition axis on the single-precision lanes, which
/// hold one coefficient set each. Each is unbuilt work — a table to generate or
/// a kernel to write — and a later change that builds one moves the cell out of
/// the refused list and into the measured table.
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
    /// rather than the machine. What decides what the run can order is the
    /// spread of the paired within-round ratios, which the report measures.
    double canarySpreadAlarm = 5.0;

    /// The options to measure, named as the report prints them. Empty measures
    /// every option this build offers, which is what a caller who has not
    /// chosen yet wants.
    ///
    /// Naming a set narrows every figure and every conclusion below to that
    /// set: the fastest option reported is then the fastest of the ones asked
    /// for. A name is answered in one of three ways and the report keeps them
    /// apart, because the three mean different things to a caller: an option this
    /// build serves is measured; a cell of the option space that the library
    /// refuses where it is named is reported with the library's reason, which is
    /// unbuilt work and not a misspelling; and a name that is neither is reported
    /// as no option of this library, so a typo is never read as a machine on
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
/// one class can be documented at different bounds — a relaxed accuracy rung
/// against the certified one, the across-orders lane against the per-argument
/// one — and the report shows each row's own bound beside it, so a class's
/// leader is the fastest option at *some* accuracy in that precision, never
/// "the fastest at your accuracy".
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

/// One option, as this machine measured it.
///
/// \ingroup boys
struct OptionProbeMeasurement {
    /// The option's name, as the report prints it.
    std::string name;

    /// The precision class this option is ranked in, and the only set of options
    /// it is ever ordered against.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The axes the option instantiates, as the library reports them: the route
    /// whose fits it reads, the scheme it sums them with, the partition of the
    /// fitted regions it reads, and the packing axis its entry carries. A
    /// defaulted axis is still printed, so a row states the whole combination
    /// rather than the part that differs from a default.
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme;
    FitGranularity granularity = kDefaultFitGranularity;
    PackAxis pack = PackAxis::kArguments;

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

    /// Cost per argument in nanoseconds at the lower quartile of the paired
    /// rounds: the figure a caller with a long workload meets on a machine whose
    /// clock is not at its peak, and not the best single observation. It is the
    /// reference lane's own lower-quartile cost scaled by this option's ratio
    /// below, so the column is consistent with the ratios it is built from.
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

    /// This option's cost as a fraction of the reference lane's, at the lower
    /// quartile of the per-round ratios between them. One exactly for the
    /// reference lane itself.
    double ratioToReference = 1.0;

    /// The same ratio at the lower and the upper quartile of its rounds: the
    /// band the middle half of the run put it in.
    double ratioLo = 0.0;
    double ratioHi = 0.0;

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
    double ownLo = 0.0;
    double ownHi = 0.0;

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

/// One precision's ranking: that precision's options, fastest measured first.
///
/// A class is one precision and nothing else. The bounds inside it are not
/// equal and are not meant to be: each row carries its own, so the leader is
/// the fastest option at some accuracy in this precision. A caller who needs a
/// particular accuracy reads the bound column and takes the fastest row whose
/// bound is theirs; the probe does not make that choice for them by mixing
/// precisions or by hiding a looser bound.
///
/// \ingroup boys
struct OptionProbeClass {
    /// The precision this class is.
    OptionPrecision precision = OptionPrecision::kFp64;

    /// The name a report prints it under.
    std::string name;

    /// This precision's measured options, fastest first, with their bounds
    /// beside them. Empty when nothing of this precision was measured.
    std::vector<std::string> ranked;

    /// The fastest of them, empty when the class is empty.
    std::string leader;

    /// The leader's cost per argument, nanoseconds.
    double leaderNsPerArgument = 0.0;

    /// Whether every other option of the class was the slower of the two against
    /// the leader in the middle half of the paired rounds, so an order exists
    /// inside this class. False for a class whose run was too short for a band.
    bool ordered = false;

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
/// \ingroup boys
struct OptionProbeCell {
    /// The cell's name, in the report's own option grammar, so a caller can
    /// name it in ProbeOptions::only and be told what this build does with it.
    std::string name;

    /// The axes this cell fixes.
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme;
    FitGranularity granularity = kDefaultFitGranularity;
    PackAxis pack = PackAxis::kArguments;
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
    /// One option leads its precision class and every rival of it in that class
    /// was measured with its own within-round ratio band clear of the leader's.
    kRecommend = 0,
    /// The probe declined: too few rounds for a band, nothing measured in the
    /// class, or a rival whose band straddles the leader's, so the two cannot be
    /// ordered against each other. See the report's reason, and the fallback it
    /// leaves.
    kCannotDetermine,
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

    /// One entry per precision measured, in the classes' own order. The ranking
    /// inside a class is that precision's own and never crosses into another.
    std::vector<OptionProbeClass> classes;

    /// Every cell of the option space, served ones and refused ones alike. The
    /// coverage is the library's, not the caller's selection: it is the same
    /// book whichever set was asked for, so a cell this build does not serve is
    /// never absent from a narrowed run either.
    std::vector<OptionProbeCell> cells;

    /// Options the library offers on other builds but not on this one, because
    /// this build's backend table does not carry the arithmetic they run in.
    std::vector<std::string> unoffered;

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
    /// run, and every pass contributed to every figure above.
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

    /// Whether the probe named a winner.
    OptionProbeVerdict verdict = OptionProbeVerdict::kCannotDetermine;

    /// The option the probe recommends: the leader of the certified double
    /// lane's precision class, empty when it declined or when that class
    /// produced no figure.
    std::string recommended;

    /// The fastest option measured in the double lane's precision whose own
    /// bound is at or below the certified lane's, empty when no such option was
    /// measured. This is the fastest row a caller at the certified accuracy can
    /// take, and it is the narrowest reading of the fp64 class's ranking.
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
    /// the same numbers the verdict is.
    std::string confidence = "not measured";

    /// The option a static reading of the library's own tables picks, when the
    /// measurement could not order the field — empty when the probe named a
    /// measured winner, so a reader never has to work out which kind of answer
    /// they are looking at. **A heuristic and not a measurement**: it is chosen
    /// by counting what the option's partition stores and the degree it
    /// evaluates, never by timing, and the report says so where it prints it.
    std::string heuristicOption;

    /// Why that option, in the library's own numbers: the degrees and stored
    /// coefficients the fallback was chosen from, and what the rule cannot
    /// compare. Empty exactly when \c heuristicOption is.
    std::string heuristicBasis;

    /// Whether a run that could not order the field still left the caller a
    /// default to take. True when the fallback above is named or the verdict is
    /// a recommendation; a false here means neither a measurement nor a static
    /// reading could answer, and the report says which.
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
/// the library reports — BoysFitRoutes, BoysEvalSchemes, BoysFitGranularities
/// and BoysPackAxes — crossed with those rungs, so the option space is the
/// library's own and the coverage section can account for every cell of it,
/// served or refused.
///
/// Then it runs the pass protocol in ProbeOptions: every pass carries runs of the
/// fixed-work canary beside its rounds, each round calls every option once in an
/// order shuffled per round, and the report carries each option's lower-quartile
/// cost with its spread beside it, its ratio to the reference lane with the band
/// that ratio fell in, the drift of that ratio between the run's halves, the
/// resolution the paired ratios support, and the canary and load readings taken
/// along the way as context. No pass is discarded on the canary's word: a fixed
/// work read by wall clock measures the clock as much as the load, so the canary
/// is reported and never gates.
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
/// \returns the report, whose verdict is \c kCannotDetermine whenever the
///          measurement did not support naming an option
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
