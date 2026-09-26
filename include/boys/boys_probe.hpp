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
/// cost over the passes it was measured in, the machine load each pass was
/// taken under, the axes the option instantiates, and the accuracy the option
/// actually delivered — so a consumer can see whether a faster option was faster
/// at the same accuracy or merely at a lower one.
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
/// **It refuses to order noise, and it says how close it could look.** The
/// reported cost of an option is the minimum of its clean passes, and its spread
/// is the ratio of the slowest clean pass to the fastest. The probe's resolution
/// is what the run actually observed: the widest disagreement between the
/// canary's own runs inside an admitted pass, or the leading option's spread
/// across its clean passes, whichever is the larger. A rival closer to the
/// leader than that resolution cannot be placed against it, and the probe then
/// declines to name a winner, prints the resolution, and says which options it
/// could not separate. The same refusal happens when no pass could be admitted
/// at all, when fewer than two were — a spread needs two passes to exist, and a
/// single pass reports a precision the run never measured — and when nothing was
/// measured in the class. A wrong recommendation is worse than none, so the
/// refusal path is the one this entry is most careful about.
///
/// **The instrument, and what admits a pass.** Every pass carries runs of a
/// fixed-work integer spin — the canary — bracketing the timed region and taken
/// between its rounds. The spin holds no floating-point state, so the arithmetic
/// question this probe is about cannot change the instrument's own cost. A pass
/// is admitted when the spin's own runs across it agree within
/// ProbeOptions::canarySpreadThreshold; that is the admission rule because it
/// measures the machine's timing noise directly. A steady load slows every
/// option by the same factor and leaves their order alone — a ratio is what is
/// being compared — while an unsteady one is what corrupts a comparison, and the
/// spread between the spin's runs is that unsteadiness.
///
/// The load readings printed beside each pass are context, not the gate: a
/// background average over the quiet window before the timed region, a point
/// reading either side of it, and the level the spin held while the timed rounds
/// ran. A reading is the percentage by which the spin took longer than the
/// fastest run of that same work ever seen on this machine, so it says how busy
/// the machine looked rather than how well it could be measured. It is not the
/// machine's total processor utilization — the standard library cannot reach
/// another process's processor time, which is what a system-wide counter would
/// need — it is the load a single-threaded, CPU-bound caller actually suffered.
/// A pass the canary cannot vouch for is reported as disturbed and **excluded
/// rather than averaged in**.
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

    /// Timed passes. A disturbed pass is discarded, so this is an upper bound
    /// on the number of passes the reported figures rest on.
    int passes = 5;

    /// Timed rounds per pass. Every option is called once per round, so a pass
    /// is one interleaved sequence of one call per option per round; the
    /// option's figure for that pass is its minimum over the rounds.
    int rounds = 5;

    /// Length of the quiet window the background load average is taken over,
    /// in seconds.
    double backgroundWindowSeconds = 1.0;

    /// Length of the calibration window the spin's quiet floor is taken over,
    /// in seconds.
    double calibrationSeconds = 0.5;

    /// Spread percentage of the canary's own runs across a pass above which the
    /// pass is discarded. This is the admission rule: it is a property of the
    /// instrument's own readings, so a pass is judged on whether fixed work
    /// could be repeated rather than on how busy the machine looked. What the
    /// probe then orders is decided by the measured resolution, not by this
    /// number, so a lenient bar here costs no honesty.
    double canarySpreadThreshold = 5.0;

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

/// One timed pass and the canary readings that decide whether its figures may be
/// reported.
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
    /// the load the pass's own figures were taken under.
    double inPassLoad = 0.0;

    /// Spread percentage of the canary's runs across the pass: the slowest run
    /// over the fastest, less one, in percent. This is what admits or discards
    /// the pass.
    double canarySpread = 0.0;

    /// Whether the canary's runs disagreed by more than the bar. A disturbed
    /// pass is reported here and excluded from every figure below.
    bool disturbed = false;
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

    /// Whether at least one pass of this option was clean, so the cost and
    /// spread below rest on a measurement. False means the option produced no
    /// figure on this run, and the accuracy below is still the measured one.
    bool measured = false;

    /// Cost per argument in the fastest clean pass, nanoseconds.
    double nsPerArgument = 0.0;

    /// Cost per argument in the slowest clean pass, nanoseconds.
    double nsPerArgumentMax = 0.0;

    /// Slowest clean pass divided by the fastest: 1.00 is a perfectly repeatable
    /// measurement and a large value is a result, not a nuisance.
    double spread = 0.0;

    /// Clean passes this figure rests on.
    int cleanPasses = 0;

    /// Passes discarded for this option.
    int disturbedPasses = 0;

    /// Load percentage the canary held inside the pass that produced the fastest
    /// figure above, which is the load the reported number was taken under.
    double loadAtMinimum = 0.0;

    /// Largest absolute difference between this option's values and the
    /// certified per-order fp64 lane's value for the same order at the same
    /// argument, over the workload.
    double maxError = 0.0;

    /// The bound this option is judged against, read from the library: the
    /// accuracy rung's own figure for a rung option, the partition's own
    /// certified figure for a partition option, and the published per-lane
    /// bound for the narrower lanes. A measured error at or below it is the
    /// contract being met on this workload.
    double bound = 0.0;

    /// Whether every value was bit-identical to the certified lane's, at the
    /// same order and the same argument.
    bool bitIdenticalToReference = false;

    /// Whether the measured error is at or below the bound above.
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

    /// Whether the leader is clear of every other option in the class by more
    /// than this run's resolution, so an order exists.
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
    /// One option leads its precision class and every other option in that class
    /// is further behind it than the resolution this run measured.
    kRecommend = 0,
    /// The probe declined: no clean pass, fewer than two, nothing measured in
    /// the class, or a rival inside the resolution, so the two cannot be ordered
    /// against each other. See the report's reason.
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

    /// Whether a floor was established at all. A false here is the whole
    /// measurement being unusable: the canary's readings have nothing to be
    /// relative to, and no pass can be judged, so none is run.
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

    /// Passes the canary vouched for.
    int cleanPasses = 0;

    /// Passes discarded for the canary's own disagreement.
    int disturbedPasses = 0;

    /// Spread percentage of the canary's runs across the widest admitted pass:
    /// the instrument's own measured uncertainty on this run.
    double canarySpread = 0.0;

    /// What this run can order, as a fraction of a cost: a rival closer to the
    /// leader than this is inside the noise. It is the larger of the canary's
    /// widest admitted spread and the leader's own spread across its clean
    /// passes, both measured here, so it moves with the machine instead of being
    /// a bar chosen in advance — an unsteady run orders only large differences,
    /// a steady one orders small ones. Zero when the run could order nothing at
    /// all, which is also when the verdict is the refusal: fewer than two passes
    /// were admitted, or the accuracy class produced no figure.
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
    /// their figures and how far behind the leader they are. Empty when the
    /// recommendation is clear of the field.
    std::vector<std::string> inseparable;

    /// One line naming how far the recommendation can be trusted, built from
    /// the same numbers the verdict is.
    std::string confidence = "not measured";
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
/// Then it runs the pass protocol in ProbeOptions: each pass carries runs of the
/// fixed-work canary, a pass whose canary runs disagree by more than
/// ProbeOptions::canarySpreadThreshold is discarded, and the report carries the
/// minimum of the passes that were admitted with their spread beside it, the
/// resolution those passes support, and the load readings taken along the way as
/// context.
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
