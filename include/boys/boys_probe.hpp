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
/// taken under, and the accuracy the option actually delivered — so a consumer
/// can see whether a faster option was faster at the same accuracy or merely at
/// a lower one.
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
/// measured at the certified accuracy. A wrong recommendation is worse than
/// none, so the refusal path is the one this entry is most careful about.
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
/// and the test suite are for.
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
    /// for. A name that is no option of this library is reported apart from one
    /// this build cannot serve, so a misspelling is told apart from a build
    /// fact rather than read as a machine on which nothing is fast.
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

/// One option, as this machine measured it.
///
/// \ingroup boys
struct OptionProbeMeasurement {
    /// The option's name, as the report prints it.
    std::string name;

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

    /// The bound the option is held to at the reference accuracy multiplier:
    /// read from the library for the fp64 options, and the published per-lane
    /// bound for the narrower ones. A measured error at or below it is the
    /// contract being met on this workload.
    double bound = 0.0;

    /// Whether every value was bit-identical to the certified lane's, at the
    /// same order and the same argument.
    bool bitIdenticalToReference = false;

    /// Whether the measured error is at or below the bound above.
    bool meetsBound = false;

    /// Sum of every value the option returned, so a caller can see that the
    /// timed work ran and ran on the intended arguments.
    double checkedSum = 0.0;
};

/// What the probe concluded, and the two ways it can end.
///
/// \ingroup boys
enum class OptionProbeVerdict : int {
    /// One option leads and every other option in its accuracy class is further
    /// behind it than the resolution this run measured.
    kRecommend = 0,
    /// The probe declined: no clean pass, fewer than two, nothing at the
    /// certified accuracy, or a rival inside the resolution, so the two cannot be
    /// ordered against each other. See the report's reason.
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

    /// One entry per option measured, in the order the report prints them.
    std::vector<OptionProbeMeasurement> measurements;

    /// Options the library offers on other builds but not on this one, because
    /// this build's backend table does not carry the arithmetic they run in.
    std::vector<std::string> unoffered;

    /// Names the caller asked for that are no option of this library at all.
    /// Kept apart from the unoffered list above, which holds options the library
    /// has and this build cannot serve: a name here is a misspelling, and
    /// nothing was measured for it.
    std::vector<std::string> notAnOption;

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
    /// agree to within it cannot be separated by this probe.
    double referenceBound = 0.0;

    /// Whether the probe named a winner.
    OptionProbeVerdict verdict = OptionProbeVerdict::kCannotDetermine;

    /// The option the probe recommends, empty when it declined.
    std::string recommended;

    /// The fastest option measured whose accuracy is indistinguishable from the
    /// certified lane's, empty when no such option was measured.
    std::string fastestAtReferenceAccuracy;

    /// The fastest option measured, whatever its accuracy, empty when no option
    /// was measured.
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
/// not this build's; and the relaxed accuracy tiers are found by asking the
/// library what each tier it can name delivers, so a build serving fewer rungs
/// offers fewer options.
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
