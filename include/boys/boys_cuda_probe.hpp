#pragma once

/// \file boys_cuda_probe.hpp
/// The device option probe: which of this library's CUDA entries is cheapest on
/// the card the caller is running on, and whether the measurement is strong
/// enough to name one at all.
///
/// **Why this is a library entry and not a table in a document.** Which device
/// entry wins is a property of the card, not of the library. Double precision is
/// a thirty-second of single on a small consumer die and a half on a compute
/// part, so the fp64 and fp32 lanes trade places between cards; a card with
/// bf16 tensor support is not the card here; the batch shapes move the cost by
/// how much of the ladder they store and how they classify the argument; and the
/// device-callable entries win or lose against the batch ones by whether the
/// caller's kernel can keep the ladder in registers. A ranking printed in
/// documentation is therefore a claim about the card it was written on. This
/// entry measures the ranking on the card it is called on and reports what it
/// found, including how confident it is.
///
/// **The device is chosen by the caller, and the report names it.** The entry
/// takes the ordinal of the device to measure, establishes that device's context
/// before it allocates anything, and reports the name, compute capability,
/// memory, driver, toolkit and target architecture of the device that answered.
/// A consumer who asked for device 1 has to be able to see that device 1 is what
/// measured their figure. A device that does not exist, or a machine that has no
/// device at all, is a status in the return value and never a crash.
///
/// **Transfer is outside the timed region.** Every buffer the measurement needs
/// — the order array, the arguments in each precision the lane takes, the output
/// — is allocated and filled once, before any clock runs, and nothing is read
/// back until the whole measurement is over. Nothing crosses the bus while the
/// figures are being taken. What the figures are therefore not is an end-to-end
/// call cost: a consumer who uploads a fresh batch per call pays a transfer this
/// probe deliberately excludes, and the two are different questions.
///
/// **Launch is amortised, and what remains of it is measured rather than
/// assumed.** A timed region records a CUDA event, launches the entry many times
/// back to back into the same resident buffers, records a second event and
/// synchronises; `cudaEventElapsedTime` then reports the device timeline between
/// the two, and the per-call figure is that elapsed time divided by the
/// repetition count. The host's submission cost is hidden by queueing the
/// launches ahead of the device, and what a figure still contains is the part of
/// a launch that is genuinely device-side — the grid it takes to start a kernel
/// at all. That part is not assumed to be small: the report carries a control
/// that measures a kernel doing no arithmetic under the same protocol, and
/// states the floor per launch and what fraction of the fastest figure it is.
/// The same control repeats one entry at two very different repetition counts
/// and reports whether the per-call figures agree — a timer that had absorbed
/// the queueing would answer differently at the two counts, because the amount
/// amortised over a call changes by the ratio between them.
///
/// **The device-callable entries are measured by subtraction, inside a kernel.**
/// The entries of boys_cuda_device.hpp exist to be called from the middle of
/// the caller's own kernel, so launching one of ours would measure a shape
/// nobody uses. Those rows are taken by timing a caller-shaped kernel that forms
/// x per thread and calls the entry, timing the same kernel with the call
/// removed and the memory traffic unchanged, and subtracting the second bracket
/// from the first. No launch of this library's is inside either bracket, and the
/// subtraction removes the launch, the indexing and the traffic together. The
/// report names, per row, which of the two methods produced its figure.
///
/// **It refuses to order noise, and it says how close it could look.** Every
/// timed region carries repetitions of a fixed-work integer kernel — the canary
/// — taken between the rounds, and its own spread across a pass is what admits
/// or discards that pass. A disturbed pass is reported and excluded rather than
/// averaged in. An entry's figure is the minimum of its clean passes and its
/// spread is the ratio of the slowest clean pass to the fastest. The run's
/// resolution is what it actually observed: the larger of the canary's widest
/// admitted spread and the leading entry's own spread across its clean passes.
/// A rival closer to the leader than that resolution cannot be placed against
/// it, and the probe declines to name a winner, states the resolution and says
/// which entries it could not separate. The same refusal happens when no pass
/// could be admitted, when fewer than two were — a spread needs two passes to
/// exist — and when nothing was measured at all. A wrong ranking is worse than
/// none, so the refusal path is the one this entry is most careful about.
///
/// **Entries are ordered only against entries asked the same question.** An
/// entry that returns F_n alone and one that returns the whole ladder for the
/// same arguments have not done the same work, and a table that put them in one
/// column would rank the amount of output instead of the cost of the
/// arithmetic. The report therefore carries one ranking per question class —
/// the single order, the ladder to each argument's own order, and the ladder to
/// one common top order — and never a winner across them.
///
/// **The accuracy column is the documented bound, not a measurement.** A faster
/// precision is not a faster option unless it is fast at an accuracy the caller
/// can use, so each row carries the bound its lane documents at full accuracy.
/// It is read from the documentation and stated as such; what an entry actually
/// delivers on this card is the accuracy gate's business, and that gate exists.
///
/// **What this probe does not measure**, and why. The relaxed accuracy
/// multipliers are left out: the device lane fixes the multiplier at the call
/// site as a template argument, so measuring one would need an instantiation
/// per rung, and the CPU probe already ranks the rungs on a host where they are
/// reachable at run time. The ordered-batch precondition is left out: AllN*
/// entries document that their arguments are non-decreasing and this probe does
/// not sort them, so an unsorted batch costs what the classification costs and
/// is not a figure for the sorted shape the entry is for. The fp16 and fp32
/// single entries' region-B option is measured in both of its forms, because
/// both are reachable from a consumer and neither substitutes for the other.
///
/// **This result is about the card it was measured on.** The report says so in
/// its own output, not only here, and it states the card's documented ratio of
/// single- to double-precision throughput, because that ratio is what orders the
/// fp64 and fp32 lanes against each other. A table of costs pasted into a bug
/// report months later would otherwise read as a general claim about the
/// library, and on this die it would be the wrong one.
///
/// \ingroup boys

#include "boys/boys.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace boys {

/// How a call to the device probe ended.
///
/// The probe reports through this status and never throws: a caller that names
/// a device the machine does not have, or runs on a machine with no CUDA device
/// at all, gets that back as a value it can branch on. \c kSuccess is 0.
///
/// \ingroup boys
enum class DeviceProbeStatus : int {
    kSuccess = 0,      ///< the measurement ran
    kNoDevice,         ///< this machine has no CUDA device
    kDeviceNotFound,   ///< the ordinal asked for is not a device this machine has
    kDeviceError,      ///< a CUDA operation failed; the CUDA runtime reports the detail
    kInvalidArgument,  ///< the request named no entry this library has; see the report
};

/// What the probe concluded for one question class, and the two ways it can end.
///
/// \ingroup boys
enum class DeviceProbeVerdict : int {
    /// One entry leads and every other entry in its class is further behind it
    /// than the resolution this run measured.
    kRecommend = 0,
    /// The probe declined: no clean pass, fewer than two, nothing measured in
    /// the class, or a rival inside the resolution, so the two cannot be ordered
    /// against each other. See the class's reason.
    kCannotDetermine,
};

/// The workload and the pass protocol the device probe runs, with the defaults
/// a caller who wants a representative answer should leave alone.
///
/// The knobs exist because the right workload is the caller's, not this
/// library's: a consumer whose basis has a lower highest angular momentum, or
/// whose argument range is narrower, gets a more relevant ranking by saying so.
/// Every one of the values below is printed in the report, so a figure is never
/// read without the protocol that produced it.
///
/// \ingroup boys
struct DeviceProbeOptions {
    /// Ordinal of the device to measure, as the CUDA runtime numbers it. The
    /// probe establishes this device's context before it allocates anything.
    int device = 0;

    /// Arguments per call. It has to be large enough that one call's arithmetic
    /// costs more device time than launching the kernel does, or the figure is
    /// the launch rather than the entry; the report's launch control states the
    /// floor per launch and what fraction of the fastest figure it is, so a run
    /// whose workload was too small says so instead of quietly ranking the
    /// launcher.
    ///
    /// The default is a batch a real integral pass would present, and it is also
    /// the smallest at which the subtraction route's figures were repeatable on a
    /// card of this class: below it the caller-shaped kernel's two halves come out
    /// equal within the clock and the subtraction resolves nothing.
    std::size_t count = 1u << 18;

    /// Highest order any argument carries, 1..kMaxBoysOrder. It is also the
    /// common top order the all-n class is asked for.
    int nmax = kMaxBoysOrder;

    /// Lower end of the log-uniform argument range.
    double xLo = 1e-3;

    /// Upper end of the log-uniform argument range.
    double xHi = 40.0;

    /// Seed of the workload's generator, so a run is reproducible from the
    /// report alone.
    std::uint64_t seed = 47;

    /// Timed passes. A disturbed pass is discarded, so this is an upper bound
    /// on the number of passes the reported figures rest on. It defaults above
    /// the two a resolution needs, because a machine carrying other work will
    /// lose some of them, and a run that admits only one reports nothing.
    int passes = 5;

    /// Timed rounds per pass. Every entry is timed once per round, so a pass is
    /// one interleaved sequence of one timed region per entry per round; the
    /// entry's figure for that pass is its minimum over the rounds.
    ///
    /// Two is the smallest useful value and a reasonable one: the two halves of a
    /// subtraction swap places between the rounds, so each half is timed going
    /// first in one of them, and the pass's short window is itself worth having on
    /// a machine whose timing wanders — admission is decided by how well fixed
    /// work repeated inside the pass, so a longer pass is a longer exposure.
    int rounds = 2;

    /// Launches inside one timed region. This is the number the launch cost is
    /// amortised over: the region brackets all of them together, so the host's
    /// submission cost falls by this factor while the device-side part of a
    /// launch does not, and that difference is what the control measures. The
    /// controls read the same entries at
    /// DeviceProbeOptions::controlRepetitionsLow and its high counterpart, which
    /// sit either side of this value.
    int repetitions = 16;

    /// Repetition counts the repetition controls use. They are far apart on
    /// purpose: a figure that had absorbed a fixed cost per call would differ
    /// between them by roughly the ratio of the two counts.
    int controlRepetitionsLow = 4;

    /// The second count of the repetition controls; see controlRepetitionsLow.
    int controlRepetitionsHigh = 256;

    /// Spread percentage of the canary's own runs across a pass above which the
    /// pass is discarded. This is the admission rule: it is a property of the
    /// instrument's own readings, so a pass is judged on whether fixed work
    /// could be repeated rather than on how busy the card looked. What the probe
    /// then orders is decided by the measured resolution, not by this number, so
    /// a lenient bar here costs no honesty.
    double canarySpreadThreshold = 5.0;

    /// The entries to measure, named as the report prints them. Empty measures
    /// every entry this build offers, which is what a caller who has not chosen
    /// yet wants.
    ///
    /// Naming a set narrows every figure and every conclusion below to that set:
    /// the fastest entry reported is then the fastest of the ones asked for. A
    /// name that is no entry of this library is reported apart from one this
    /// build cannot serve, so a misspelling is told apart from a build fact
    /// rather than read as a card on which nothing is fast.
    std::vector<std::string> only;
};

/// The device the figures were taken on.
///
/// A device figure with no device named is not a measurement, and this is the
/// struct that names it. Every field is read from the runtime before anything is
/// timed, so a consumer running this on another card can tell whether a figure
/// they are reading was taken on theirs.
///
/// \ingroup boys
struct DeviceProbeDevice {
    /// Ordinal the probe measured, which is the one the caller asked for.
    int ordinal = -1;

    /// The runtime's own name for the card.
    std::string name;

    /// Compute capability, major and minor.
    int computeMajor = 0;

    /// Compute capability, minor part.
    int computeMinor = 0;

    /// Total device memory, bytes.
    std::size_t totalMemoryBytes = 0;

    /// Streaming multiprocessors on the card.
    int multiProcessorCount = 0;

    /// Graphics clock the card reports, kHz.
    int clockKHz = 0;

    /// Memory clock the card reports, kHz.
    int memoryClockKHz = 0;

    /// The card's documented ratio of single- to double-precision throughput,
    /// read from the runtime — 32 on a small consumer die, 2 on a compute part.
    /// This is what orders the fp64 and fp32 lanes against each other, and it is
    /// the reason a ranking is a statement about a card.
    int singleToDoublePrecisionPerfRatio = 0;

    /// Driver version, as the runtime packs it.
    int driverVersion = 0;

    /// CUDA runtime version, as the runtime packs it.
    int runtimeVersion = 0;

    /// The CUDA toolkit the library was compiled with.
    std::string toolkitVersion;

    /// The architectures this library was compiled for, as CMake was told them.
    /// A figure for a kernel running a cubin built for another architecture is
    /// not a figure for this card.
    std::string architectures;
};

/// One timed pass and the canary readings that decide whether its figures may be
/// reported.
///
/// \ingroup boys
struct DeviceProbePass {
    /// Wall seconds of the pass's timed rounds, the canary's own runs between
    /// them not counted.
    double seconds = 0.0;

    /// Median canary timing inside the pass, milliseconds. How far this sits
    /// above the calibration floor is what the pass saw of the card.
    double canaryMedianMs = 0.0;

    /// Spread percentage of the canary's runs across the pass: the slowest run
    /// over the fastest, less one, in percent. This is what admits or discards
    /// the pass.
    double canarySpread = 0.0;

    /// Whether the canary's runs disagreed by more than the bar. A disturbed
    /// pass is reported here and excluded from every figure below.
    bool disturbed = false;
};

/// One entry, as this card measured it.
///
/// \ingroup boys
struct DeviceProbeMeasurement {
    /// The entry's name, as the report prints it.
    std::string name;

    /// \c "fp64", \c "fp32" or \c "fp16": the arithmetic the entry runs in.
    std::string precision;

    /// \c "single", \c "all-orders", \c "all-n" or \c "each-order": the shape of
    /// the call.
    std::string shape;

    /// The question class the entry belongs to, which is the set it may be
    /// ordered against and no other.
    std::string question;

    /// How the figure was obtained: \c "launched" for a kernel of this library
    /// bracketed by events, \c "in-kernel" for the subtraction against the
    /// caller's kernel with the call removed.
    std::string route;

    /// Whether the launch of this entry is this library's or the caller's.
    bool launchedByLibrary = false;

    /// The bound the entry's lane documents at full accuracy, read from the
    /// documentation. It is not a measurement — the accuracy gate is what
    /// measures — and it is here so a faster precision is not read as a faster
    /// option at the same accuracy.
    double documentedBound = 0.0;

    /// Whether at least one pass of this entry was clean, so the cost and spread
    /// below rest on a measurement. False means the entry produced no figure on
    /// this run.
    bool measured = false;

    /// Whether the subtraction that produced this row resolved a cost above its
    /// own baseline. False for an in-kernel row whose two halves came out equal
    /// within the clock: at this workload the entry's arithmetic is not
    /// distinguishable from the same caller kernel with the call removed, so the
    /// row is a statement about that kernel and about the card's timing rather
    /// than about the entry. Such a row carries no ordering: a zero here is the
    /// instrument's resolution, not a free entry. Always true for a launched row,
    /// which is not a subtraction.
    bool subtractionResolved = false;

    /// Cost per argument in the fastest clean pass, nanoseconds.
    double nsPerArgument = 0.0;

    /// Cost per argument in the slowest clean pass, nanoseconds.
    double nsPerArgumentMax = 0.0;

    /// For a row taken by subtraction, the per-argument figure of the half the
    /// entry was subtracted against: the same caller-shaped kernel with the call
    /// removed and the traffic kept. Zero for a launched row, which is not a
    /// subtraction. It is here so that the subtraction is a number a reader can
    /// check rather than a method the report asserts — the entry's own figure is
    /// the difference between the region that held it and this.
    double nsPerArgumentBaseline = 0.0;

    /// Slowest clean pass divided by the fastest: 1.00 is a perfectly repeatable
    /// measurement and a large value is a result, not a nuisance.
    double spread = 0.0;

    /// Clean passes this figure rests on.
    int cleanPasses = 0;

    /// Passes discarded for this entry.
    int disturbedPasses = 0;

    /// Sum of every value the entry returned, read back once outside the timer,
    /// so a caller can see that the timed work ran and ran on the intended
    /// arguments in every precision the lane takes.
    double checkedSum = 0.0;

    /// Whether the repetition control was run on this row. Every row a ranking
    /// could recommend carries it: the two rows the report's controls describe,
    /// and every row that led a question class once the rows set aside below were
    /// taken out of the running.
    bool repetitionChecked = false;

    /// What the repetition control said about this row. False means the row's
    /// per-argument figure changed more between two very different counts of
    /// launches than the run can place the row at, so a cost that repeats with
    /// the launch rather than with the call survived into it — for a launched row
    /// that is the launch, and for a subtracted row it is the halves' own noise
    /// amplified by the ratio between the difference and the readings it came
    /// from. It is also false when one of the two counts left no figure to
    /// compare, which establishes nothing either way rather than establishing
    /// agreement. Such a row is not ordered: the class lists it in \c notOrdered
    /// and falls to the next row.
    bool repetitionAgrees = true;

    /// The repetition control's own sentence about this row, empty when it was
    /// not run.
    std::string repetitionNote;
};

/// One question class: what its entries were asked to produce, and what the run
/// could conclude about ordering them.
///
/// \ingroup boys
struct DeviceProbeClass {
    /// The class's name, as the report prints it.
    std::string question;

    /// One sentence saying what every entry in this class produced.
    std::string asked;

    /// What this run can order in this class, as a fraction of a cost: two rows
    /// closer together than this are inside the noise and the probe declines to
    /// order them. It is the larger of the canary's widest admitted spread and
    /// the leader's own spread across its clean passes, both measured here, so it
    /// moves with the card instead of being a bar chosen in advance. Zero when
    /// the run could order nothing at all.
    ///
    /// It describes the leader. Each rival is judged against a threshold built
    /// from that rival's spread as well, since a rival whose own passes wandered
    /// is a rival this run cannot place however steady the leader was; the
    /// strings in \c inseparable carry the threshold each was judged against.
    double resolution = 0.0;

    /// Whether the probe named a winner in this class.
    DeviceProbeVerdict verdict = DeviceProbeVerdict::kCannotDetermine;

    /// The entry the probe recommends in this class, empty when it declined.
    std::string recommended;

    /// The fastest entry measured in this class, empty when no entry was.
    std::string fastestOverall;

    /// The entries this class will not order against the recommendation, with
    /// their figures and how far behind the leader they are. Empty when the
    /// recommendation is clear of the field.
    std::vector<std::string> inseparable;

    /// The rows this class set aside rather than rank, each with the reason it
    /// was set aside. Two reasons arise, and each names itself: an in-kernel row
    /// whose subtraction resolved nothing above the caller kernel's own baseline
    /// — its arithmetic sat inside the traffic and the clock's resolution at this
    /// workload — and a row whose repetition control did not agree, so a figure
    /// is not being placed on it at all. Both are named rather than dropped: what
    /// the first needs is a heavier workload or a leaner caller kernel, and what
    /// the second needs is a steadier machine.
    std::vector<std::string> notOrdered;

    /// One sentence saying what the verdict rests on.
    std::string reason;

    /// One line naming how far the recommendation can be trusted, built from
    /// the same numbers the verdict is.
    std::string confidence = "not measured";
};

/// The repetition-count control: one entry timed at two very different counts of
/// launches inside the region, to show what the timer is and is not measuring.
///
/// A timer that had absorbed a fixed cost per call would not survive this. The
/// figure is the region's elapsed device time divided by the launches in it, so
/// any cost that does not repeat with the call — the region's own setup, the
/// first launch's cold instruction cache, a fixed submission charge that lands on
/// the device timeline once — is divided by a different number at each count and
/// shows up as a difference between the two figures. An entry whose per-call
/// figure holds across a wide ratio of counts has had such a cost divided out.
///
/// Two of these are taken: one on the launched route, over the fastest launched
/// row, and one on the subtraction route, over the fastest in-kernel row. They
/// are the same check applied to the two methods, and a route whose control
/// disagrees has figures the report does not ship silently.
///
/// \ingroup boys
struct DeviceProbeRepetitionControl {
    /// The entry the control used.
    std::string entry;

    /// \c "launched" or \c "in-kernel": which route's figure this controls.
    std::string route;

    /// The low repetition count the entry was timed at.
    int repetitionsLow = 0;

    /// The low count's figure, nanoseconds per argument. On the in-kernel route
    /// this is the difference the subtraction produced, not a region's own time.
    double nsPerArgumentLow = 0.0;

    /// The high repetition count.
    int repetitionsHigh = 0;

    /// The high count's figure, nanoseconds per argument.
    double nsPerArgumentHigh = 0.0;

    /// On the in-kernel route, the low count's baseline: the same caller-shaped
    /// kernel with the entry's call removed. Zero on the launched route, which
    /// subtracts nothing. Both halves are reported because a difference of two
    /// unstable readings can look stable by accident; a reader who has the halves
    /// can see that it did not.
    double nsPerArgumentBaselineLow = 0.0;

    /// The high count's baseline, on the same terms as the low one.
    double nsPerArgumentBaselineHigh = 0.0;

    /// The two figures' disagreement, as a fraction of the faster of them.
    /// Infinite when one of the two counts produced no figure to compare — the
    /// subtraction at that count resolved nothing — which is not a disagreement
    /// of zero and must not be read as one.
    double difference = 0.0;

    /// Whether the two counts agreed within the resolution the run measured —
    /// the larger of the canary's widest admitted spread and this row's own
    /// spread across its clean passes. False means either that a fixed cost per
    /// call survived into the figures at the repetition count they were taken at,
    /// or that one count left nothing to compare; the note says which, and the
    /// report sets the row aside rather than shipping it.
    bool agrees = false;

    /// The resolution the two counts were judged against, as a fraction. Carried
    /// so that the verdict above is a number a reader can check.
    double judgedAgainst = 0.0;

    /// What a kernel that does no Boys arithmetic costs per launch, nanoseconds,
    /// measured under the same event protocol. Filled on the launched route only:
    /// nothing launched by this library is inside either bracket of a subtraction,
    /// so there is no floor to take. The in-kernel route's floor is its own
    /// baseline, reported above.
    double nsPerLaunchFloor = 0.0;

    /// The strongest of the checks above as one sentence, for a reader who reads
    /// only this line.
    std::string note;
};

/// Everything the device probe measured and concluded.
///
/// \ingroup boys
struct DeviceProbeReport {
    /// How the call ended. Every field below is meaningful only when this is
    /// \c kSuccess; on any other value the report carries the reason and the
    /// device information the runtime could supply.
    DeviceProbeStatus status = DeviceProbeStatus::kNoDevice;

    /// The protocol the figures below were taken under.
    DeviceProbeOptions options;

    /// The card the figures were taken on, named in full.
    DeviceProbeDevice device;

    /// The card-specific caveat, in the report's own words: what this card is
    /// and why a ranking taken on it is a statement about it. Built from what
    /// the runtime reports about the device, so it says something true of the
    /// card that answered rather than of the card this was written on.
    std::string caveat;

    /// Why the call ended as it did, when it did not succeed. Empty on success.
    std::string failure;

    /// One entry per measurable entry of the lane, in the order the report
    /// prints them.
    std::vector<DeviceProbeMeasurement> measurements;

    /// Names the caller asked for that are no entry of this library at all.
    /// Nothing was measured for them.
    std::vector<std::string> notAnEntry;

    /// Entries this build does not carry, because the fp16 seam is closed in it.
    std::vector<std::string> unoffered;

    /// Arguments in the workload, and the highest order any of them carries.
    std::size_t workloadCount = 0;

    /// One entry per pass run, in order.
    std::vector<DeviceProbePass> passes;

    /// Passes the canary vouched for.
    int cleanPasses = 0;

    /// Passes discarded for the canary's own disagreement.
    int disturbedPasses = 0;

    /// Spread percentage of the canary's runs across the widest admitted pass:
    /// the instrument's own measured uncertainty on this run.
    double canarySpread = 0.0;

    /// Wall milliseconds of the fastest canary run seen while calibrating: the
    /// floor the canary's own spread is measured against.
    double canaryFloorMs = 0.0;

    /// One conclusion per question class, in the report's own order.
    std::vector<DeviceProbeClass> classes;

    /// The control on the launched route: the fastest launched row timed at two
    /// very different repetition counts.
    DeviceProbeRepetitionControl control;

    /// The same control on the subtraction route, over the fastest in-kernel row
    /// whose subtraction resolved a cost. The method that needs it most: a
    /// difference of two readings is the one figure here that a fixed cost per
    /// call could hide inside, and this is what says it did not.
    DeviceProbeRepetitionControl deviceCallControl;
};

/// Measures every CUDA entry this build offers on the device the caller names.
///
/// Establishes the named device's context, checks it exists, uploads that
/// device's tables, allocates the workload's buffers once and fills them, and
/// then runs the pass protocol in DeviceProbeOptions: every entry is timed by
/// device events over DeviceProbeOptions::repetitions back-to-back launches into
/// the resident buffers, in each of several rounds, inside each of several
/// passes, with runs of the fixed-work canary taken between the rounds. A pass
/// whose canary runs disagree by more than
/// DeviceProbeOptions::canarySpreadThreshold is discarded. The report carries,
/// per entry, the minimum of the passes that were admitted with their spread
/// beside it, one ranking per question class with the resolution those passes
/// support, and the repetition-count controls described on
/// DeviceProbeRepetitionControl — one per route, since the two routes are two
/// methods and a check of one says nothing about the other.
///
/// The device-callable entries are measured by the with-and-without subtraction
/// rather than by launching this library's kernel; DeviceProbeMeasurement::route
/// says which method produced each row.
///
/// The entry allocates, uploads, launches and synchronises, and is not a hot
/// path: it is meant to be called once, from a program the consumer builds and
/// runs where they deploy. It is not noexcept — an allocation failure is a real
/// outcome here and there is no meaningful result to return instead — but it
/// does not throw for a bad device: that is a DeviceProbeStatus.
///
/// The probe saves the calling thread's current device before it starts and
/// restores it before it returns, so a caller's own device is where they left it.
/// A table is uploaded to whichever device is current when it is uploaded, and
/// the upload is guarded on the device it went to, so the copies a caller's
/// device already holds are not written over: measuring device 1 leaves device
/// 0's tables resident and uploads device 1's. See the header preamble of
/// boys_cuda.hpp for what the lane's tables are keyed on.
///
/// \param options the workload, the device ordinal and the pass protocol; the
///                defaults are the ones the preamble describes
///
/// \returns the report, whose status is \c kDeviceNotFound or \c kNoDevice when
///          the caller named a device this machine does not have, and whose
///          per-class verdict is \c kCannotDetermine whenever the measurement
///          did not support naming an entry
///
/// \ingroup boys
DeviceProbeReport RunDeviceOptionProbe(const DeviceProbeOptions& options = {});

/// The report as the text a consumer reads, device statement and all.
///
/// The wording is the report's own: it is plain text, it names the card, it
/// states the protocol, and it says what was discarded and why. It is written
/// for a reader who has nothing but this output.
///
/// \param report a report, from RunDeviceOptionProbe
///
/// \returns the report as text, newline-terminated
///
/// \ingroup boys
std::string FormatDeviceOptionProbe(const DeviceProbeReport& report);

} // namespace boys
