#pragma once

/// \file boys_cuda_probe.hpp
/// The device option probe: which of this library's CUDA entries is cheapest on
/// the card the caller is running on, and whether the measurement is strong
/// enough to name one at all.
///
/// **Why this is a library entry and not a table in a document.** Which device
/// entry wins is a property of the card, not of the library: double precision is
/// a thirty-second of single on a small consumer die and a half on a compute
/// part, so the fp64 and fp32 lanes trade places between cards; a card with bf16
/// tensor support computes a different arithmetic again; the batch shapes move
/// the cost by how much of the ladder they store and how they classify the
/// argument; and the device-callable entries win or lose against the batch ones
/// by whether the caller's kernel can keep the ladder in registers. A ranking
/// printed in documentation would be a claim about the card it was written on;
/// this entry measures the ranking on the card it is called on and reports what
/// it found, including how confident it is.
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
/// **Launch is amortised, and what remains of it is taken out of the figure
/// from the entry's own readings rather than assumed.** A timed region records a
/// CUDA event, launches the entry many times back to back into the same resident
/// buffers, records a second event and synchronises; `cudaEventElapsedTime` then
/// reports the device timeline between the two, and the per-call figure is that
/// elapsed time divided by the repetition count. The host's submission cost is
/// hidden by queueing the launches ahead of the device, and what a region's
/// figure still contains is the part of a launch that is genuinely device-side —
/// the grid, the registers and the occupancy ramp it takes to start that entry's
/// kernel at all. That part is paid once per launch and therefore falls as
/// 1/count, so every row is read at two argument counts in the same round, four
/// times apart, and its figure is the count-independent asymptote its own two
/// readings extrapolate to. Both readings are reported beside the figure, so a
/// reader sees what was taken out and how much it was, and the report states
/// that the extrapolation assumes the launch term falls as 1/count.
///
/// The report also measures a kernel that does no arithmetic at all, launched the
/// same way, and states the floor per launch and what fraction of the fastest
/// figure it is. **It is a diagnostic and not the number that comes out of the
/// figures**: an entry's kernel starts more work than an empty one and its own
/// launch costs more than the floor does, so subtracting the floor would take out
/// part of the term and look like a correction. A control repeats one entry at
/// two very different repetition counts and reports whether the per-call figures
/// agree — a timer that had absorbed the queueing would answer differently at the
/// two counts, because the amount amortised over a call changes by the ratio
/// between them.
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
/// **The comparison is paired, and made inside a round.** Every entry is timed
/// once in every round of a pass, in an order shuffled per round from the
/// workload's own seed, and the comparison between two entries is the ratio of
/// their per-round figures *within the same round*, never a ratio of two figures
/// taken from different rounds. That is what makes the answer survive a card
/// whose clocks move: a graphics clock drops as the die warms and a memory clock
/// steps under load, so the device time of fixed work drifts through a run with
/// no other process involved, and two entries timed in different rounds would
/// then be compared across two different clocks. Inside one round both entries
/// ran under the same clock — the timed regions of a round are adjacent, and
/// each opens and closes its own event pair — so a drift common to the round
/// cancels in the ratio. Every figure below is an aggregate of those paired
/// ratios rather than of absolute times taken across the run.
///
/// **An entry's cost is anchored, and read at the middle of the run.** The
/// reference entry's own figure is the lower quartile of its per-round costs;
/// every other entry's is that figure scaled by its ratio to the reference at the
/// middle of the rounds, with the ratio's upper quartile printed beside it as the
/// spread. The
/// minimum of a run is its earliest and best-clocked observation, which on a card
/// whose clock decays with the die's temperature is a state a caller's workload
/// does not stay in; a mean would let one disturbed round move the figure, and a
/// quartile is what the bulk of a long workload meets and still drops the round a
/// background process stole. A peak-clock column is kept beside it — the fastest
/// single round the entry was seen in — because it bounds what the entry can do,
/// and it is never the reported cost.
///
/// **It refuses to order noise, and it says how close it could look.** For each
/// rival of a shape's fastest entry the probe forms the pair's own within-round
/// ratio, and the rival is ordered only when the middle half of those rounds
/// puts it behind — when the pair's own band clears one. A rival whose band
/// straddles one cannot be placed, and one whose band lies ahead of the leader
/// contradicts the statistic the leader was chosen by; either way the probe
/// declines to name a winner on that evidence, prints the run's resolution and
/// names each unplaced rival with its band and the number of rounds it was the
/// slower of the two in. It is not where a shape ends: the entries the run could
/// not separate are re-run alone at a longer protocol and the one that led the
/// most of those runs is recommended. That answer stands whether the entries
/// could not be separated or the run was too short for a band to exist at all:
/// fewer than four pooled rounds places nothing, and what such a run read is what
/// it re-runs, with the count it took printed beside the name.
/// A shape whose entries produced no figure at all is the one shape that ends
/// without a recommendation, because there is nothing there to re-run.
///
/// **The canary is a diagnostic and gates nothing.** Every pass carries runs of
/// a fixed-work integer kernel — the canary — taken between its rounds, holding
/// no floating-point state, so the arithmetic this probe ranks cannot change the
/// instrument's own cost. Its readings are reported beside each pass, and a pass
/// whose canary runs disagreed with itself by more than
/// DeviceProbeOptions::canarySpreadAlarm is marked as one that ran on a card
/// whose clocks moved — **and is still used**: a fixed work read by a device
/// clock measures the clock as much as the load, so a decaying clock widens the
/// canary's own spread on a card doing nothing else, and a rule that discarded on
/// that spread would discard the measurement rather than the card.
///
/// **The ordering is checked against the card's clocks, not assumed free of
/// them.** Each rival's ratio to its shape's fastest entry is followed between
/// the run's first and second half of rounds, and the report warns where a pair
/// moved by more than the run can order: such a pair is not equally exposed to
/// the card's clocks, and its ordering is a property of this run's clocks as
/// well as of the two entries. Where no pair moved by more than that, the report
/// says so instead.
///
/// **When the measurement cannot separate the entries, they are re-run rather
/// than guessed at.** The entries a shape's own rounds left tied are measured
/// again on their own, at a protocol many times longer and repeated several
/// times, with a fresh shuffle each run; each run names its own fastest entry by
/// the same within-round rule the main run used, and the vote over those runs
/// gives the shape its recommendation. A unanimous result, a majority and a
/// field the runs divided evenly are three different answers and the report
/// prints which one it has: the first two are named with their counts, and the
/// third is named as a choice among entries the evidence cannot separate. A
/// shape holding a single entry names that entry, because one entry is not a
/// ranking and there is no alternative to it. Nothing here is read off the
/// library's tables: a name this report carries is a name some clock produced.
///
/// **The two routes are kept apart and named per row.** An in-kernel row is a
/// subtraction of two adjacent regions of the *same* round, so the pair that
/// produces it is already paired on the clock and survives a drift; the report
/// names the route that produced each row because the two methods are not
/// comparable to each other. The repetition-count control reports whether an
/// entry timed at two very different numbers of launches per region agreed within
/// the resolution the run measured — the same within-round band the ordering is
/// made in — and a row it fails is set aside rather than ranked.
///
/// **Entries are ordered only within one precision and one question shape.**
/// Those two are what the consumer has already fixed when they call: how much
/// precision the result needs and what they are asking for — one F_n, or the
/// ladder. An entry of another precision or another question answers a different
/// call, so no entry is ever ordered against one: the report carries one class
/// per combination it measured, and a class's winner is the fastest entry of
/// that precision for that question and a claim about nothing else. The
/// figure the report states is per argument rather than per value, so the single
/// order, the ladder to each argument's own order and the ladder to one common
/// top order are three questions and never one ranking.
///
/// **The class key is the consumer's; the ranking is over the library's
/// choices.** Inside a class the entries are alternatives the library selects
/// between on the caller's behalf — the fit route, the evaluation scheme, the
/// interval partition, the packing axis — and the class's ranking over them is
/// what produces the default the report names. Two rows that differ only in how
/// the library reads its stored fits are two entries of one class.
///
/// **The bound is a column of a row, not the thing a class is keyed on.** Two
/// rows of one class need not document the same bound: the fp32 lane's single
/// entries are the case in this library, where one of them evaluates the fast
/// region-B exponential and documents the looser bound that buys. That row stays
/// in its class and carries its own bound, so a class's winner is the fastest
/// entry of one shape in one precision at the bound its own row states — the
/// fastest at an accuracy, not the fastest at one accuracy.
///
/// **The accuracy column is the documented bound, not a measurement.** A row
/// that is fast at a looser bound is not a faster option for a caller who needs
/// the tighter one, so each row carries the bound its lane documents, read from
/// the library's option table. What an entry actually delivers on this card is
/// the accuracy gate's business, and that gate certifies every entry.
///
/// **A default is read out of the classes the run measured and nothing else.**
/// The fastest entry of a class is the default that class names, and it is named
/// only where the run's own rounds placed it there: an entry read off the
/// library's tables is never a default.
///
/// **What this probe does not measure**, and why. The ordered-batch precondition
/// is left out: AllN* entries document that their arguments are non-decreasing
/// and this probe does not sort them, so an unsorted batch costs what the
/// classification costs and is not a figure for the sorted shape the entry is
/// for. The fp16 and fp32 single entries' region-B option is measured in both of
/// its forms, because both are reachable from a consumer and neither substitutes
/// for the other, and each carries the bound its own form documents. The accuracy
/// the entries actually deliver is left to the accuracy gate, which certifies
/// every entry: this probe spends its time on cost, and every figure it prints is
/// a device time.
///
/// **A lane whose tables this build cannot make resident is a refusal with its
/// reason and not an absence.** A lane whose degree tables the device refused is
/// reported as such where its classes would have stood, because a reader who
/// expected it there would otherwise take its absence for a fact about the
/// arithmetic.
///
/// **This result is about the card it was measured on.** The report says so in
/// its own output, not only here, and it states the card's documented ratio of
/// single- to double-precision throughput, because that ratio is what orders the
/// fp64 and fp32 lanes against each other. A table of costs pasted into a bug
/// report months later would otherwise read as a general claim about the library.
///
/// \ingroup boys

#include "boys/boys.hpp"
#include "boys/boys_cuda_options.hpp"

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

    kCount, ///< statuses this report defines; one past the last
};

/// What the probe concluded for one question shape, and the two ways it can end.
///
/// \ingroup boys
enum class DeviceProbeVerdict : int {
    /// One entry leads: either every other entry of that shape in that precision
    /// is further behind it than the resolution this run measured, or the shape
    /// holds one entry and there is no alternative to it, or the entries this
    /// run could not place behind the leader were re-run on their own and one of
    /// them led those runs.
    kRecommend = 0,
    /// The probe declined: nothing measured in the shape, or nothing that
    /// resolved above its own instrument, so there was no figure of this shape to
    /// place at all and no tied band to re-run. A run too short for a quartile
    /// band is not this case: its entries produced figures, it places none of
    /// them, and it names one the way a shape its own rounds could not order does.
    /// See the ranking's reason.
    kCannotDetermine,
};

/// How a shape's recommendation was reached.
///
/// It exists so that a consumer who prints a recommended entry prints where the
/// name came from with it, rather than leaving a reader to guess whether the
/// entry was measured or taken by choice. The distinction is the whole of the
/// rule: a measured ordering is a figure, a vote over re-runs is a weaker
/// figure with its count beside it, and a choice among entries the runs divided
/// evenly is not a measurement at all and is named as what it is.
///
/// \ingroup boys
enum class DeviceProbeDefaultHow : int {
    /// No recommendation was reached.
    kNone = 0,
    /// The shape's own rounds placed every rival behind the leader: a measured
    /// ordering of entries the same run timed side by side.
    kOrdered,
    /// The shape holds one entry, or one entry of it is the only one this run
    /// could order, or one entry of it came out of the run with a figure at all.
    /// There is nothing it can be ordered against, so it is named by there being
    /// no alternative — which is an answer, not a refusal, and not a ranking
    /// either. A row of the shape that was not the one named is not dropped: the
    /// class prints it with the check that set it aside.
    kOnlyEntry,
    /// The shape's own rounds left the leader tied with others, those entries
    /// were re-run alone at the refinement protocol, and every one of those runs
    /// was fastest with the entry this shape names — the entry the vote named,
    /// measured again and unanimous. The row the shape's own figures put first is
    /// printed beside it, and where the two differ that difference is what says
    /// the top entries cannot be separated.
    kRefined,
    /// The same re-runs, with a majority rather than all of them leading with the
    /// entry this shape names: the vote had a plurality over the entries it could
    /// not separate, and the entry it named is the one the shape names. The row
    /// the shape's own figures put first is printed beside it.
    kVote,
    /// The shape's top entries could not be separated, and the vote itself did
    /// not settle them either: it was split across several, or no run placed a
    /// leader at all. The entry is the row the shape's own figures put first
    /// among entries it cannot tell apart — named as that, with the vote and both
    /// figures printed, and not as a ranking.
    kChosenAmongEquals,

    kCount, ///< states this report defines; one past the last
};

/// The name of one of those, as one token a script or a report can print beside
/// a recommendation: "ordered", "only-entry", "refined", "vote",
/// "chosen-among-equals" or "none".
///
/// Every enumerator of \c DeviceProbeDefaultHow is named by the definition, and
/// its switch has no default arm: a state added to the enumeration without a
/// name is a compile error rather than a state quietly reported under the name
/// of another. That sentence is load-bearing here and not a formality — the
/// name an unstated state would take, \c "none", is itself a plausible answer a
/// report prints and a reader accepts, so before the check beside the definition
/// an omission was indistinguishable from a true answer.
///
/// \param how the state to name
///
/// \returns the name, which is never empty for an enumerator of the
///          enumeration; \c nullptr only for a value no arm names, which the
///          check beside the definition turns into a compile error
///
/// \ingroup boys
constexpr const char* DeviceProbeDefaultHowName(DeviceProbeDefaultHow how) noexcept;

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

    /// Arguments per call, and the first of the two counts every row is read at.
    /// It has to be large enough that one call's arithmetic costs more device
    /// time than launching the kernel does, or the figure is the launch rather
    /// than the entry; the report's launch control states the floor per launch
    /// and what fraction of the fastest figure it is, so a run whose workload was
    /// too small says so instead of quietly ranking the launcher.
    ///
    /// The default is a batch a real integral pass would present, and it is also
    /// the smallest at which the subtraction route's figures were repeatable on a
    /// card of this class: below it the caller-shaped kernel's two halves come out
    /// equal within the clock and the subtraction resolves nothing.
    std::size_t count = 1u << 18;

    /// The second count every row is read at, as a multiple of \c count. A row's
    /// figure is formed from its own readings at both counts, so this is the
    /// lever that decides how much of the launch term comes out with it.
    ///
    /// The two readings are made in the same round and four times apart by
    /// default, which is what the arithmetic on \c nsPerArgument assumes: with a
    /// fixed cost L per launch, a reading f at C and f' at r*C leave a figure of
    /// (r*f' - f)/(r - 1) — the count-independent cost the entry itself
    /// extrapolates to. Four is large enough that a launch term of a few percent
    /// of a figure moves it by a storable amount and small enough that the second
    /// reading stays inside the same regime on the device.
    ///
    /// **Both counts must be large enough that the grid saturates the card.** At
    /// a count where a heavy per-argument kernel's grid is too small, the figure
    /// carries a second effect that is not the launch at all, and no launch term
    /// explains a swing of that size; the default pair sits well above that
    /// regime. See \c DeviceProbeOptions::count for the other end of the rule.
    int countPairFactor = 4;

    /// Highest order any argument carries, 1..kMaxBoysOrder. It is also the
    /// common top order the all-n shape is asked for.
    int nmax = kMaxBoysOrder;

    /// Lower end of the log-uniform argument range.
    double xLo = 1e-3;

    /// Upper end of the log-uniform argument range.
    double xHi = 40.0;

    /// Seed of the workload's generator, so a run is reproducible from the
    /// report alone.
    std::uint64_t seed = 47;

    /// Timed passes. Every pass is used: no pass is discarded, so the paired
    /// rounds the figures rest on are this number times \c rounds, and the
    /// figures pool all of them.
    int passes = 5;

    /// Timed rounds per pass. Every entry is timed once in every round, so a
    /// pass is one interleaved sequence of one timed region per entry per round
    /// and the comparison between two entries is the ratio of their figures
    /// inside one round. The pooled rounds are \c passes * \c rounds, and a
    /// quartile band needs at least four of them.
    ///
    /// Two is the smallest useful value: the two halves of a subtraction swap
    /// places between the rounds, so each half is timed going first in one of
    /// them, and a pass's short window keeps the rounds inside it close together
    /// on a clock that drifts across a run.
    int rounds = 2;

    /// Launches inside one timed region. This is the number the launch cost is
    /// amortised over: the region brackets all of them together, so the host's
    /// submission cost falls by this factor while the device-side part of a
    /// launch does not, and that difference is what the control measures. The
    /// controls read the same entries at
    /// DeviceProbeOptions::controlRepetitionsLow and its high counterpart, which
    /// sit either side of this value.
    int repetitions = 16;

    /// Repetition counts the repetition controls use. They have to be far enough
    /// apart that a figure which had absorbed a cost paid per region would differ
    /// between them by more than the band the run can order at, and both have to
    /// sit in the regime the reported figures are taken in: four launches is a far
    /// shorter run on the device than sixty-four, so a pair reaching much lower
    /// would be comparing two protocols rather than checking one.
    ///
    /// The range is measured rather than chosen. On the card this probe was
    /// developed against — a Quadro T1000 — the figure the table ships for the
    /// widest all-orders fp64 entry came to 6.772, 6.772, 6.773 and 6.768
    /// ns/argument at four, sixteen, sixty-four and two hundred and fifty-six
    /// launches per region, this probe's own reading at `--passes=3 --rounds=3`.
    /// The figure holds across that whole range, and the repetition count the rows
    /// themselves are taken at (\c repetitions, sixteen) sits inside it, between
    /// this value and its high counterpart.
    int controlRepetitionsLow = 4;

    /// The second count of the repetition controls; see controlRepetitionsLow.
    int controlRepetitionsHigh = 64;

    /// Spread percentage of the canary's own runs across a pass above which the
    /// report marks the pass as one that ran on a card whose clocks moved.
    /// **A diagnostic, and the only thing it decides is a flag.** It admits
    /// nothing and excludes nothing: the canary is a fixed-work kernel read by a
    /// device clock, so a card whose clock decays widens this number with no
    /// other process involved, and a run that discarded on it would discard the
    /// measurement rather than the card. What decides what the run can order is
    /// the spread of the paired within-round ratios, which the report measures.
    double canarySpreadAlarm = 5.0;

    /// The rows to measure, named as the report prints them. Empty measures
    /// every row this build offers, which is what a caller who has not chosen
    /// yet wants.
    ///
    /// A row is an entry crossed with a division form
    /// (\c DeviceProbeMeasurement::form), and the unit of a request is the row:
    /// the name an entry carries unmarked is the default form's row, and its
    /// other two members are named with the form's segment, so a request that
    /// wants one entry at all three forms names all three of its rows — which is
    /// the unit every count below this option is made in, the grid's places and
    /// the rankings' rows alike.
    ///
    /// Naming a set narrows every figure and every conclusion below to that set:
    /// the fastest entry reported is then the fastest of the ones asked for. A
    /// name that is no entry of this library is reported apart from one this
    /// build cannot serve, so a misspelling is told apart from a build fact
    /// rather than read as a card on which nothing is fast.
    std::vector<std::string> only;

    /// Runs the refinement stage takes of a shape whose entries its own rounds
    /// could not separate, and the factor by which each of those runs is longer
    /// than the main protocol: a refinement run is \c passes passes of
    /// \c rounds * this rounds, with its own shuffle, over the tied entries
    /// alone. **One multiplication and not two**: the factor lengthens a run's
    /// rounds, so a run is \c refinementFactor times the main protocol and not
    /// that factor squared, and the stage costs \c refinementRuns such runs over
    /// the tied set alone — the pair or the few entries the answer actually rests
    /// on, and not the whole option space.
    int refinementRuns = 5;

    /// The refinement protocol's multiplier; see refinementRuns.
    int refinementFactor = 1;
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

/// One timed pass and the canary readings taken beside it.
///
/// \ingroup boys
struct DeviceProbePass {
    /// Wall seconds of the pass's timed rounds, the canary's own runs between
    /// them not counted.
    double seconds = 0.0;

    /// Whether the canary was timed in this pass at all. False means no reading
    /// was taken — the two figures below and the flag are then not readings, they
    /// are zero because nothing was read, and this pass is counted in neither
    /// DeviceProbeReport::passesWithinAlarm nor
    /// DeviceProbeReport::passesAboveAlarm. **A canary that did not run is not a
    /// canary that stayed quiet**, so read this before reading any of them: a
    /// spread of zero is a still clock only where a reading was taken.
    bool canaryMeasured = false;

    /// Median canary timing inside the pass, milliseconds. How far this sits
    /// above the calibration floor is what the pass saw of the card. Zero when
    /// DeviceProbePass::canaryMeasured is false, and not a reading then.
    double canaryMedianMs = 0.0;

    /// Spread percentage of the canary's runs across the pass: the slowest run
    /// over the fastest, less one, in percent. **Context, not a gate**: the
    /// canary is fixed work read by a device clock, so this number moves with
    /// the clocks as well as with the load. Zero when
    /// DeviceProbePass::canaryMeasured is false, and not a reading then.
    double canarySpread = 0.0;

    /// Whether the canary's runs disagreed by more than
    /// DeviceProbeOptions::canarySpreadAlarm. The pass is reported with the flag
    /// and excluded by nothing. False when the canary did not run: the flag is
    /// raised by a reading that disagreed, never by the absence of one.
    bool canaryWide = false;

    /// Widest relative width of a within-round paired ratio this pass measured:
    /// the upper quartile of an entry's ratio to the reference over the lower
    /// quartile of the same ratio, less one, in percent, taken at its widest over
    /// the entries. This is the spread the ordering is made in — a ratio inside
    /// one round, where a drift common to the round cancels — so it is the paired
    /// analogue of the canary column above, and it is reported beside it rather
    /// than gating anything itself. Zero when the pass held too few rounds for a
    /// band to exist.
    double pairedSpread = 0.0;
};

/// One entry, as this card measured it.
///
/// \ingroup boys
struct DeviceProbeMeasurement {
    /// The entry's name, as the report prints it. A name appears once in a run's
    /// table, so the name and \c entryIndex identify a row between them.
    ///
    /// The name is the library's own name for the entry where the row runs the
    /// build's default division form, and that name with the form's own segment
    /// (a row of \c BoysDivisionForms(), boys.hpp) where it does not — the
    /// grammar the option probe's cells are named by, on the same reading: the
    /// row a caller who names no form reaches is the default form's, so the form
    /// left unmarked is the form that row really divides in, and the two other
    /// members of the axis are marked so that no two rows of one entry can print
    /// the same name.
    std::string name;

    /// \c "fp64", \c "fp32" or \c "fp16": the arithmetic the entry runs in.
    std::string precision;

    /// The entry's row in the library's own option table: the entry's own facts —
    /// what it documents and what the library says it is — are read through it
    /// rather than looked up by name.
    std::size_t entryIndex = 0;

    /// The entry itself, where \c entryIndex is its position in this run's own
    /// list. A reader needs the entry and not only its index, because the axes an
    /// entry fixes are the entry's own statement (\c DeviceEntryAxesOf,
    /// boys_cuda_options.hpp), and a row is judged against the entry it names.
    DeviceEntry entry = DeviceEntry::kSingleF64;

    /// The division form this row's ladder steps divide in: which member of
    /// \c BoysDivisionForms() (boys.hpp) the row was measured at.
    ///
    /// **A row is an entry crossed with a form and not an entry.** Every launched
    /// entry takes the form as a trailing parameter and every device-callable one
    /// as a template argument, all three of them certified for each
    /// (\c BoysCuda, boys_cuda.hpp), so the same entry is a different row under
    /// each of the three and this field is the coordinate that tells them apart.
    /// It is the form the region was dispatched at
    /// (\c probe_detail::ProbeTimeRequest::form), and it is stated on the row so
    /// that the arithmetic a figure is about is read off the row and not
    /// recovered from its name.
    DivisionForm form = kDefaultDeviceDivisionForm;

    /// \c "single", \c "all-orders", \c "all-n" or \c "each-order": the shape of
    /// the call.
    std::string shape;

    /// The question shape of the call. With the entry's precision it names the one
    /// set the entry may be ordered in and no other: the report ranks one question
    /// shape at one precision, so this is the ranking the row stands in.
    std::string question;

    /// How the figure was obtained: \c "launched" for a kernel of this library
    /// bracketed by events, \c "in-kernel" for the subtraction against the
    /// caller's kernel with the call removed.
    std::string route;

    /// Whether the launch of this entry is this library's or the caller's.
    bool launchedByLibrary = false;

    /// The bound the entry's lane documents, read from the library's own option
    /// table. It is not a measurement — the accuracy gate is what measures — and
    /// it is here so a faster precision is not read as a faster option at the
    /// same accuracy.
    double documentedBound = 0.0;

    /// Whether at least one round of this entry produced a figure, so the cost
    /// and the band below rest on a measurement. False means the entry produced
    /// no figure on this run: no round of it could be timed.
    bool measured = false;

    /// Whether the subtraction that produced this row resolved a cost above its
    /// own baseline. False for an in-kernel row whose two halves came out equal
    /// within the clock at every round: at this workload the entry's arithmetic
    /// is not distinguishable from the same caller kernel with the call removed,
    /// so the row is a statement about that kernel and about the card's timing
    /// rather than about the entry. Such a row carries no ordering: a zero here
    /// is the instrument's resolution, not a free entry. Always true for a
    /// launched row, which is not a subtraction.
    bool subtractionResolved = false;

    /// The row's own reading at the run's first count
    /// (\c DeviceProbeOptions::count), nanoseconds per argument, at the lower
    /// quartile of the paired rounds and anchored to the reference entry the way
    /// \c nsPerArgument is. It is the first of the two readings the figure was
    /// extrapolated from, and it is reported so that the extrapolation can be
    /// checked rather than believed: this reading still contains the launch term
    /// the figure does not.
    double nsPerArgumentAtCount = 0.0;

    /// The row's own reading at the second count (\c DeviceProbeOptions::count
    /// times \c DeviceProbeOptions::countPairFactor), on the same terms as
    /// \c nsPerArgumentAtCount and taken in the same rounds. The two together
    /// are what the figure was formed from.
    double nsPerArgumentAtPairCount = 0.0;

    /// Cost per argument in nanoseconds: the reference entry's own lower-quartile
    /// figure scaled by \c ratioToReference, which is this entry's ratio to the
    /// reference at the middle of the run's rounds — a central ratio and not this
    /// entry's own band; see \c ratioToReference for why that is the fair choice.
    /// **It is the one figure this entry is ranked and recommended by.**
    ///
    /// **It is the count-independent cost the row's own two readings extrapolate
    /// to, on both routes**: each round reads the entry at the run's count and
    /// again at the pair count, four times larger, and takes out the term that
    /// falls as 1/count — see \c DeviceProbeOptions::countPairFactor for the
    /// arithmetic. No assumption about what an empty kernel costs stands in for
    /// the entry's own launch, so the correction is the difference between two
    /// columns of the report and not a number a reader has to take on trust. It
    /// assumes the launch term falls as 1/count, and the report says so where it
    /// prints the columns.
    double nsPerArgument = 0.0;

    /// Whether the row's own two readings left no positive launch term to take
    /// out in any round of the run: at every round the reading at the pair count
    /// was not the cheaper of the two, so the asymptote the figure is formed from
    /// came out at or below zero and the row has no figure at all. Such a row is
    /// set aside by name wherever a shape lists what it could not place — a
    /// zero or a negative there would read as a free call, and a positive one
    /// would be a number the readings do not support. Raising the argument counts
    /// is what answers it: the launch term has to be a measurable part of both
    /// readings before it can be taken out of either.
    bool extrapolationUnresolved = false;

    /// The same cost at the upper quartile of those rounds, so the two ends of
    /// the band come from the same distribution.
    double nsPerArgumentMax = 0.0;

    /// The fastest single round this entry was ever seen in, nanoseconds per
    /// argument, over the rounds that produced a figure. **One round's own
    /// figure under no anchor**, kept as one column because it is the entry's own
    /// floor — no round of it came out below this — and never the reported cost.
    /// It is not comparable with the anchored columns beside it: those are the
    /// reference entry's own figure scaled by this entry's ratio to it, so a
    /// reference whose rounds are heavy-tailed puts the anchored cost below this
    /// entry's own fastest round, and neither bounds the other in general. It is
    /// also not what a caller meets: on a card whose clock decays the fastest
    /// round is the earliest and best-clocked one.
    ///
    /// A round whose cell is zero is left out of it, because such a cell has no
    /// cost in it: for a subtracted row that is the floor a difference which did
    /// not clear its own baseline was held at, and taking it as the peak would
    /// print the instrument's floor as the entry's best round. Zero here means no
    /// round of the entry produced a figure, and the table prints a dash for it.
    double nsPerArgumentPeak = 0.0;

    /// For a row taken by subtraction, the per-argument figure of the half the
    /// entry was subtracted against, at the lower quartile of the same rounds
    /// that produced the figure above: the same caller-shaped kernel with the
    /// call removed and the traffic kept. Zero for a launched row, which is not a
    /// subtraction. It is here so that the subtraction is a number a reader can
    /// check rather than a method the report asserts. Both halves are taken
    /// inside one round, so the difference that produced \c nsPerArgument was
    /// formed round by round and these two columns are not two quartiles of one
    /// round's readings: they are the two ends of the pair, each aggregated over
    /// the run.
    double nsPerArgumentBaseline = 0.0;

    /// Upper quartile over lower quartile of this row's within-round ratio to the
    /// reference: 1.00 is a perfectly repeatable measurement and a large value is
    /// a result, not a nuisance. Both ends are ratios of two entries timed
    /// together, so this spread does not carry a clock drift common to a round.
    double spread = 0.0;

    /// This entry's cost as a fraction of the reference entry's, at the **middle**
    /// of the per-round ratios between them: one exactly for the reference entry
    /// itself, and the figure \c nsPerArgument is scaled by.
    ///
    /// The middle, and not the lower quartile of those ratios, because the two are
    /// not the same kind of reading for the reference and for every other entry:
    /// the reference's ratios are one in every round, so any quantile of them is
    /// one, while a rival's lower quartile sits below its middle by a fraction of
    /// its own round-to-round spread — so scoring every row at a lower quartile
    /// would hand the reference a credit no measurement supports and take it from
    /// its rivals, enough to turn the ranking of a class. The middle is the
    /// quantile that credits the two alike: it is reciprocal under a change of
    /// anchor, element for element over the same rounds, so this entry's figure
    /// against the reference and the reference's figure against this entry always
    /// agree on which of the two is cheaper.
    double ratioToReference = 1.0;

    /// The band the middle half of the run put that ratio in, at the lower and the
    /// upper quartile of the same rounds. **The band and not the figure**: these
    /// are what a pair that cannot be separated is read from, and they bound
    /// \c ratioToReference rather than being equal to its ends.
    double ratioLo = 0.0;
    double ratioHi = 0.0; ///< the band's other end, the ratio at its upper quartile

    /// How far this entry's ratio to the reference moved between the run's first
    /// and second half of rounds: the second half's median ratio over the first
    /// half's, less one. Zero for the reference entry by construction, and
    /// non-zero for another entry only when its cost relative to the reference
    /// changed as the run went on — which is what a clock whose decay falls
    /// differently on the two would look like. **A warning, not a correction**:
    /// a large value here says the comparisons this entry takes part in are not
    /// clock-independent on this run.
    double ratioDrift = 0.0;

    /// Paired rounds the figures above rest on: the rounds of every pass, pooled.
    /// Every round of the run is pooled, so this is the same for every entry that
    /// produced a figure, and a quartile band needs at least four of them.
    int rounds = 0;

    /// The anchor figure this row's cost columns were scaled from: the reference
    /// entry, at the lower quartile of the run's pooled rounds. A ratio between
    /// two rows is therefore a ratio inside one round's readings, and every cost
    /// column of the report is one measurement and a set of ratios rather than a
    /// set of independent times.
    double referenceNsPerArgument = 0.0;

    /// Sum of every value the entry returned, read back once outside the timer,
    /// so a caller can see that the timed work ran and ran on the intended
    /// arguments in every precision the lane takes.
    double checkedSum = 0.0;

    /// Whether the repetition control was run on this row. Every row a ranking
    /// could recommend carries it: the two rows the report's controls describe,
    /// and every row that led a question shape once the rows set aside below were
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

/// The refinement stage: a shape's tied entries, measured alone at the refinement
/// protocol, repeated, and voted on.
///
/// This is what the report does instead of naming an entry from a figure counted
/// off the library's tables. The entries re-measured are the shape's fastest and
/// every entry of it the main run could not place behind the fastest; nothing
/// else in the option space is touched, so the stage's whole cost is spent on
/// the entries the answer actually rests on.
///
/// Each run is a fresh pass over the tied set at a protocol \c passes passes long
/// and \c rounds * DeviceProbeOptions::refinementFactor rounds long, with its own
/// shuffle, and each run is ordered by the same within-round ratio rule the main
/// run used — so a run is a measurement of the same kind and not a different
/// rule. The vote is over the runs, and a run whose own rounds cannot place a
/// rival contributes its leader alone, which is what makes the vote a vote and not
/// a re-run of the main statistic.
///
/// \ingroup boys
struct DeviceProbeRefinement {
    /// Whether the stage ran at all.
    bool ran = false;

    /// Runs taken, the passes each ran, and the rounds each ran. Zero when the
    /// stage did not run.
    int runs = 0;
    int passes = 0; ///< passes each run took
    int rounds = 0; ///< rounds each pass took

    /// The entries re-measured: the shape's leader and every entry of it the main
    /// run could not place behind the leader, by name.
    std::vector<std::string> pool;

    /// The entry that led each run, in the order the runs were taken. Empty
    /// string where a run placed no leader.
    std::vector<std::string> runLeaders;

    /// The entry the stage names, empty when the stage did not run.
    std::string winner;

    /// One sentence saying what the vote was, in the report's own words.
    std::string note;

    /// The run's votes, one line per candidate that led at least one run.
    std::vector<std::string> tally;

    /// Whether every run led with the same entry, and whether the winning entry
    /// led more runs than any other while one run — not all — led elsewhere or
    /// nowhere. Both false means the runs divided evenly and the entry was taken
    /// by choice.
    bool unanimous = false;
    bool plurality = false; ///< the winner led more runs than any other, but not every run
};

/// One question shape inside a class: the entries of one precision that were
/// asked the same question, and what the run could conclude about ordering them
/// against each other.
///
/// A ranking is per shape because the cost the report states is per argument and
/// the shapes return different amounts of output for one argument — the ladder's
/// cost carries order + 1 values where the single order's carries one — so only
/// the entries of one shape, in one precision, have done the same work.
///
/// \ingroup boys
struct DeviceProbeRanking {
    /// The shape's name, as the report prints it: \c "single",
    /// \c "all-orders" or \c "all-n". The entries it holds are those of its
    /// class's precision whose own \c question is this name.
    std::string question;

    /// One sentence saying what every entry in this shape produced.
    std::string asked;

    /// What this run can order in this shape, as a fraction of a cost: two rows
    /// closer together than this are inside the noise and the probe declines to
    /// order them. It is the widest within-round ratio band the shape showed on
    /// this run — the widest of every rival's band against the shape's fastest
    /// row and of that row's own band against the reference — so it moves with
    /// the card and with this run instead of being a bar chosen in advance. It is
    /// literally the number the refusal prints, and the number the repetition
    /// control's two counts are judged against. Zero when the shape measured
    /// nothing it could compare.
    double resolution = 0.0;

    /// Whether the probe named a recommendation in this shape.
    DeviceProbeVerdict verdict = DeviceProbeVerdict::kCannotDetermine;

    /// The entry the probe recommends in this shape, empty when it declined.
    ///
    /// **It is the entry the refinement stage's vote named, where that stage
    /// ran** — entries a shape's own rounds cannot separate are settled by which
    /// was fastest in most of the stage's runs, and how that vote came out is
    /// what \c defaultHow states. The entry this shape's own figures put first,
    /// \c fastestOverall, is the record of the shorter protocol: it is printed
    /// beside the name, and where the two differ the difference is what says the
    /// shape's top entries cannot be separated. A shape the stage did not reach —
    /// one its own rounds ordered, one holding a single entry, one whose rounds
    /// produced no figure to refine — is named by the entry its own figures put
    /// first.
    std::string recommended;

    /// How \c recommended was reached. \c kOrdered is the shape's own rounds.
    /// \c kRefined and \c kVote are the vote over the refinement runs — every run
    /// and a majority of the runs respectively, each leading with the entry the
    /// shape names. \c kChosenAmongEquals is a tie the vote itself could not
    /// break: it was split across several entries, or no run placed a leader at
    /// all, and the entry named is then the one the shape's own figures put first.
    /// \c kOnlyEntry is a shape holding one entry, named by there being no
    /// alternative.
    DeviceProbeDefaultHow defaultHow = DeviceProbeDefaultHow::kNone;

    /// The fastest entry measured in this shape, empty when no entry was.
    std::string fastestOverall;

    /// The entries this shape will not order against the recommendation, with
    /// their figures and how far behind the leader they are. Empty when the
    /// recommendation is clear of the field.
    std::vector<std::string> inseparable;

    /// The rows this shape set aside rather than rank, each with the reason it
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

    /// Paired rounds this shape's verdict rests on: the rounds every pass of the
    /// run pooled together, the same number for every shape. Fewer than four is
    /// a refusal that names the count, because a lower-quartile band over fewer
    /// than four rounds is the round and not a distribution.
    int rounds = 0;

    /// The entries this shape did not place behind one leader, the leader first,
    /// by name. Filled wherever the shape's answer goes to the refinement stage:
    /// the entries its own rounds left unplaced — every shape whose \c inseparable
    /// list is non-empty — and, when the run's own checks left the shape with no
    /// row it could rank at all, the rows it measured that carry a figure. They
    /// are the entries that stage re-runs and votes on.
    std::vector<std::string> tiedEntries;

    /// The refinement stage's own record. \c ran is false for a shape the main
    /// run ordered, for one that holds a single entry, and for one whose rounds
    /// produced no figure to refine.
    DeviceProbeRefinement refinement;
};

/// One class: one precision, one question shape, and the one ranking over the
/// alternatives the library offers at those two.
///
/// The two are the consumer's own choices — the precision their calculation
/// needs and the question they are asking — so a row of another class answers a
/// different call and is never ordered against this one. What the class ranges
/// over is what the library decides: the fit route, the evaluation scheme, the
/// interval partition and the packing axis, each a row of the option table that
/// carries its own documented bound. The class's winner is the fastest of them
/// at the bound its own row states.
///
/// \ingroup boys
struct DeviceProbeClass {
    /// \c "fp64", \c "fp32" or \c "fp16": the arithmetic every entry of this
    /// class runs, and the first of its two key members. How far the fp64 class
    /// sits behind the fp32 one is the card's own ratio of single- to
    /// double-precision throughput, which is a property of the card and not of
    /// the library.
    std::string precision;

    /// \c "single", \c "all-orders" or \c "all-n": which question the entries of
    /// this class answer, and the second key member. Its member is \c question.
    std::string question;

    /// The question said in full: what an entry of this class produces for one
    /// argument, which is what makes the class's entries alternatives to each
    /// other and what a winner here is a claim about.
    std::string asked;

    /// One paragraph saying what this class is: the precision and the question
    /// its rows share, that the bounds they document are their own and need not
    /// agree, and what its winner is therefore a claim about.
    std::string note;

    /// The class's one ranking: the alternatives of this precision and this
    /// question, ordered by cost. Empty of rows when every entry of the class
    /// failed to produce a figure, which the ranking's own reason states.
    DeviceProbeRanking ranking;
};

/// The repetition-count control: one entry timed at two very different counts of
/// launches inside the region, to show what the timer is and is not measuring.
///
/// Each of the two figures is the row's own — a region's device
/// time divided by the launches in it and by the arguments in it, read at both of
/// the run's argument counts and reduced to the count-independent cost they
/// extrapolate to — so a cost the region pays once rather than once per call is
/// divided by a different number of launches at each count and shows up as a
/// difference between the two. An entry whose figure holds across a wide ratio of
/// counts has had such a cost divided out.
///
/// **The two figures are the quantity the report ships**, not a reading beside
/// it: each is the row's own figure formed at that repetition count.
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

    /// The low count's figure, nanoseconds per argument. It is the row's own
    /// figure formed at that count of launches: that count's readings at both of
    /// the run's argument counts, reduced to the count-independent cost they
    /// extrapolate to, which is the quantity the table ships for the row. On the
    /// in-kernel route each reading is the difference the subtraction produced,
    /// not a region's own time.
    double nsPerArgumentLow = 0.0;

    /// The high repetition count.
    int repetitionsHigh = 0;

    /// The high count's figure, nanoseconds per argument, on the same terms as
    /// the low one.
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
    /// Infinite when one of the two counts produced no figure to compare — a
    /// launched row whose two readings left no positive launch term, a subtraction
    /// that resolved nothing — which is not a disagreement of zero and must not be
    /// read as one.
    double difference = 0.0;

    /// Whether the two counts agreed within the resolution this run measured:
    /// the widest within-round ratio band the shape showed, the same number that
    /// shape's refusal prints. False means either that a cost paid per region
    /// survived into the figure at the repetition count it was taken at, or that
    /// one count left nothing to compare; the note says which, and the report sets
    /// the row aside rather than naming it on that figure.
    ///
    /// **This check does not soften as the measurement does.** Its two figures are
    /// the row's own figures rather than a reading beside them, but a run that
    /// cannot resolve makes this control fail rather than pass: an instrument that
    /// cannot tell two counts apart has not shown that the per-region cost is
    /// absent.
    bool agrees = false;

    /// The resolution the two counts were judged against, as a fraction. Carried
    /// so that the verdict above is a number a reader can check.
    double judgedAgainst = 0.0;

    /// What a kernel that does no Boys arithmetic costs per launch, nanoseconds,
    /// measured under the same event protocol. Filled on the launched route only:
    /// nothing launched by this library is inside either bracket of a subtraction,
    /// so there is no floor to take. The in-kernel route's floor is its own
    /// baseline, reported above.
    ///
    /// **A floor and not an entry's launch cost.** A kernel that does no
    /// arithmetic needs no registers and no occupancy ramp, so it starts more
    /// cheaply than any entry's kernel does; this is the cheapest launch the route
    /// can make, and it is reported as that. It is not the number the figures have
    /// taken out of them — the entries' own launch terms are, from each entry's
    /// own two readings — and the fraction of a row's cost printed beside it is a
    /// lower bound on that row's launch share.
    ///
    /// The run times it once, under its own protocol, before any row is put
    /// through a control, and every control the run records carries that one
    /// reading rather than dropping it: this is the run's floor, not a reading of
    /// the control it is reported beside, and a zero here is a floor the run did
    /// not establish rather than a launch that costs nothing.
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

    /// Options of the library's device space that this build does not serve,
    /// as the library reports them, with the reason beside each in
    /// \c refusedBecause. Read from BoysDeviceOptions(), so a row the library
    /// adds to the space and this build cannot serve reaches a report without
    /// an edit here.
    std::vector<std::string> unoffered;

    /// The library's reason for each name in \c unoffered, in the same order.
    std::vector<std::string> refusedBecause;

    /// Arguments in the workload, and the highest order any of them carries.
    std::size_t workloadCount = 0;

    /// One entry per pass run, in order. Every pass is used: the pass table is a
    /// record of the whole run, and a pass the canary ran wide in is flagged
    /// there rather than dropped from the figures.
    std::vector<DeviceProbePass> passes;

    /// Passes whose canary spread stayed within the alarm, passes whose canary
    /// spread did not, and passes in which no canary reading was taken at all.
    /// **Counters of a diagnostic and not of an admission rule**: no pass is
    /// excluded from any figure by which of them it landed in.
    ///
    /// The three are the pass table and account for all of it —
    /// \c passesWithinAlarm + \c passesAboveAlarm + \c passesWithoutCanary equals
    /// DeviceProbeReport::passes.size() — and the first two are the passes a
    /// spread was measured in, so a pass with no reading is placed in neither.
    int passesWithinAlarm = 0;
    int passesAboveAlarm = 0; ///< passes whose canary spread did not stay within it
    int passesWithoutCanary = 0; ///< passes in which no canary reading was taken

    /// Paired rounds every figure in the report rests on: the rounds of every
    /// pass, pooled. This is the number the minimum of four is about, and the
    /// number a refusal names when the run is short of it.
    int pairedRounds = 0;

    /// Spread percentage of the canary's runs across the widest pass that took a
    /// reading: the instrument's own measured uncertainty on this run, reported
    /// beside the figures as context. It bounds nothing and excludes nothing, and
    /// it is zero only where no pass took a reading — a case the text says in
    /// words rather than printing the zero as a spread.
    double canarySpread = 0.0;

    /// Wall milliseconds of the fastest canary run seen while calibrating: the
    /// floor the canary's own spread is measured against.
    double canaryFloorMs = 0.0;

    /// The entry every cost column is anchored to: the option book's first fp64
    /// row, resolved before any round is timed. Each measured row's cost is that
    /// reference figure scaled by the row's ratio to it at the middle of the
    /// run's rounds, so the column is one measurement and a set of ratios rather
    /// than a set of independent times.
    std::string referenceEntry;

    /// The reference entry's own cost per argument at the lower quartile of its
    /// rounds, nanoseconds: the scale every ratio of the report is applied to.
    /// Zero when the reference entry produced no figure. Every row carries its own
    /// copy in DeviceProbeMeasurement::referenceNsPerArgument, which is what that
    /// row's ratio was taken against.
    double referenceNsPerArgument = 0.0;

    /// Whether the run named a recommended entry in any class. False means no
    /// class of the run reached one: every class either ordered its entries on
    /// its own rounds, or held a single entry, or measured no figure at all.
    bool hasDefault = false;

    /// One class per precision and question shape this run's option table
    /// carries, in the report's own order: the rows of the option table in their
    /// own order, so a reader sees the entries of one precision together.
    std::vector<DeviceProbeClass> classes;

    /// Whether this run made the degree tables resident on the device. False
    /// means no row of this report was timed: a device that will not hold the
    /// tables has no entry of this surface measured, and every row is reported as
    /// producing no figure rather than as costing what tables that were never
    /// uploaded would cost.
    bool tablesResident = false;

    /// The library's own answer for a device that would not hold the tables,
    /// empty when \c tablesResident is true. It is data and not only a sentence
    /// because the option space's closure counts the members of such a run from
    /// it (DeviceOptionSpaceClosure): a row of a run whose tables never went
    /// resident is not a row this card can run, and the closure says so rather
    /// than counting it as a row that produced no figure.
    std::string refusedTables;

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
/// the resident buffers, once in each round, inside each of several passes, with
/// runs of the fixed-work canary taken between the rounds. **Every round of every
/// pass is pooled and every pass is used**; the canary's spread flags a pass and
/// excludes none. The degree tables are made resident before the first row is
/// timed, so every entry is measured under one arithmetic: a device that will not
/// hold the tables is a refusal in DeviceProbeReport::refusedTables rather than an
/// absence. Each row's cost is the reference figure scaled by that row's ratio to
/// it at the middle of the run's rounds, and every comparison the report makes is
/// between two rows' ratios taken in the same round. The report carries, per row,
/// that figure with its ratio band and its drift beside it, one class per
/// precision and question shape — each with the widest within-round band it showed
/// as its resolution and a recommended entry wherever its measurement supported
/// one — and the repetition-count controls described on
/// DeviceProbeRepetitionControl, one per route.
///
/// A shape whose own rounds cannot place a rival behind the fastest entry is not
/// left there: the entries it could not separate are re-run alone at
/// DeviceProbeOptions::refinementRuns runs of a protocol
/// DeviceProbeOptions::refinementFactor times the main one, and the entry that
/// led the most of those runs is the recommendation, with the vote printed beside
/// it. A shape holding one entry names that entry, because one entry is not a
/// ranking and there is no alternative to it. Only a shape none of whose entries
/// produced a figure at all — nothing that resolved above its own instrument —
/// ends without a recommendation, and it says which row was missing rather than
/// naming an entry read off a table. A run too short to form a within-round band
/// is not such a shape: its entries produce figures, it places none of them, and
/// it names one all the same, with the count the band needed printed beside the
/// name.
///
/// The device-callable entries are measured by the with-and-without subtraction
/// rather than by launching this library's kernel; DeviceProbeMeasurement::route
/// says which method produced each row. Both halves of that subtraction are taken
/// inside one round, so the difference is a within-round pair and survives a
/// clock that drifts over the run.
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
///          per-shape verdict is \c kCannotDetermine only where the shape
///          produced no figure to place at all
///
/// \ingroup boys
DeviceProbeReport RunDeviceOptionProbe(const DeviceProbeOptions& options = {});

/// The report as the text a consumer reads, device statement and all.
///
/// The wording is the report's own: it is plain text, it names the card, it
/// states the protocol, and it says which passes the canary ran wide in and what
/// a refusal could not place. Every shape that names a recommended entry says
/// how that name was reached, so a reader never has to guess whether the name was
/// measured. It is written for a reader who has nothing but this output.
///
/// **Its last block is the option space's closure** (\c DeviceOptionClosure):
/// the space counted, one count per state a member of it can be in, the
/// arithmetic over those counts and the verdict on it. It is printed on every
/// path, a report whose run measured nothing included, so that the number of
/// members the space has and the number this run accounted for are both in the
/// output rather than one of them being an inference from the other.
///
/// \param report a report, from RunDeviceOptionProbe
///
/// \returns the report as text, newline-terminated
///
/// \ingroup boys
std::string FormatDeviceOptionProbe(const DeviceProbeReport& report);

/// The build-defaults file this run can write, and the classes it writes a row for and
/// refuses.
///
/// The seam it writes into is `boys/boys_build_defaults.hpp`: a replacement carries the seven
/// names - the host's five and this lane's own two - a `BOYS_BUILD_DEFAULT_ROWS` list, and no
/// `BOYS_BUILD_DEFAULTS_SHIPPED`, and a build pointed at it through the `BOYS_BUILD_DEFAULTS`
/// CMake option reads it instead of the committed file. The device lane is the half of that table which no measurement had ever
/// written a row for, and the reason is the surface: a row names the arithmetic its class
/// compiles, and until every device option row stated the route, the scheme and the packing
/// its entry fixes (\c DeviceEntryAxesOf, boys_cuda_options.hpp) a device row would have
/// carried cells nobody could check.
///
/// **One row per class the table keys on, and the device half of that key is three cells
/// wide.** A class is `kDevice` with the lane the entry runs in and the shape, one per question
/// the probe ranks; the device's own three precisions are three lanes, so the rows below are the
/// nine classes of the device half - \c Precision::kFp64Device, \c kFp32Device and
/// \c kFp16Device, which are the three precisions the device surface's entries are built at
/// (\c BoysDeviceLane, boys_device_tables.hpp). A table that gave the device half one precision
/// cell could carry one of those nine per shape and no row for the other six, which is the
/// keying this file's own list is written to be able to state.
///
/// The route, the scheme and the packing axis of a row are the entry's own
/// (\c DeviceEntryAxesOf), the granularity is the partition it reads, and the packing cell is
/// that reading of region A in the axis the seam names \c PackAxis. Two further cells name the
/// row's own coordinates rather than the entry's fields: the division form the winning row was
/// measured at, and the region-B exponential its entry carries
/// (\c DeviceOptionInfo::regionBExp). Both are written because a device option's identity
/// includes them - every entry of the space runs every form, so the form cell has to be the
/// row's own, and the exponent is an axis the entries themselves vary
/// (\c DeviceOptionAxis::kRegionBExp) - so a row holding the build's default in either cell
/// would stand under a figure measured at another combination. The library serves one
/// accuracy and no accuracy cell is written: a row of this table names what the library picks,
/// and the figure behind it is the class's own reading.
///
/// **A row is a measurement only where the run measured one.** A class whose winner was the
/// last entry left standing rather than an ordering's first - \c kOnlyEntry - is written with
/// the marker a choice carries, which is the seam's own rule, and a class this run did not
/// measure, or whose ranking could not determine an entry, gets no row at all and a reason in
/// \c refused.
///
/// **A class the table in force already carries is overlaid and not skipped.** The shipped
/// rows are the base this run writes over: a class this run measured carries this run's row
/// even where the table in force wrote one, and that class is listed in \c overridden with
/// the row it replaced, so which rows the run changed is read off the emission and not
/// inferred from the file. A class the table carries and this run did not measure is written
/// back verbatim and appears in neither list: a replacement is read INSTEAD of the committed
/// file, so one that dropped it would leave its callers with no row at all.
///
/// \ingroup boys
struct DeviceDefaultsEmission {
    /// The replacement header, empty where this run measured no class at all.
    std::string text;

    /// One line per row written: the class, the entry it was taken from, and how that
    /// entry was reached.
    std::vector<std::string> emitted;

    /// One line per row written over a row the table in force already carried: the class
    /// and the entry, so the run's own edits to the shipped table are listed and not left
    /// to be found by reading two files against each other.
    std::vector<std::string> overridden;

    /// One line per class no row is written for, with the reason.
    std::vector<std::string> refused;
};

/// The defaults file this run implies, with the classes it carries and the ones it refuses.
///
/// \param report  a report, from RunDeviceOptionProbe
/// \param takenAt the host and the run, as the report's own first lines state them, or
///                empty where the caller has none to give; written into the file's comments
///
/// \returns the file's text and the two class lists. \c text is empty where the run measured
///          no device class, which is the same rule the option probe's own emission keeps: a
///          file written by a run that ranked nothing would be a transcription of the seam
///          and not a measurement
///
/// \ingroup boys
DeviceDefaultsEmission FormatDeviceBuildDefaults(const DeviceProbeReport& report,
                                                 const std::string& takenAt);

/// The device option space counted: every member of it in the state one run
/// established for it, and the arithmetic that has to close.
///
/// A member is one row of the space crossed with one division form — a row of
/// \c BoysDeviceOptions() (boys_cuda_options.hpp) at a member of
/// \c BoysDivisionForms() (boys.hpp) — so the total is the product of those two
/// tables' own sizes, both read from the library, and a projection of it that
/// cannot fall behind it. Nothing is crossed that the library does not report a
/// member for: every entry of this surface runs every form, which the axis's own
/// rows state (\c DivisionFormInfo, boys.hpp: "every entry this build carries runs
/// every one of them") and which the entries carry in their signatures
/// (\c BoysCuda, boys_cuda.hpp). The states are what the run did with that
/// member, and a member is in exactly one of them:
///
///   * \c measured — the run took a figure for the row;
///   * \c refusedAndOwed — the row is one this build does not serve, refused with the
///     library's own reason (\c DeviceOptionInfo::refusedBecause) and outstanding work;
///   * \c notRunnable — the row is served and this card would not hold the degree
///     tables, so nothing of it was timed (\c DeviceProbeReport::tablesResident is
///     false and \c refusedTables states the library's answer);
///   * \c offeredNoFigure — the run carried a place for the row and no round of it
///     produced a figure;
///   * \c notAsked — this run's own request (\c DeviceProbeOptions::only) named no such
///     row, so the member was never presented to the device; a run that never reached
///     the grid at all places no member here.
///
/// **Nothing else is a state, and \c unaccounted counts the members in none of
/// them.** A member there is a row of the library's own report that no part of the
/// run stands behind: the closure says so rather than counting it into the nearest
/// state, and the report's last line fails on it.
///
/// \c closed is the verdict that line prints: the five states sum to \c total,
/// \c unaccounted is zero, the run's own grid holds exactly the places the request
/// owes (\c gridPlaces equals \c gridPlacesOwed), the report carries every class the
/// space admits (\c classesPrinted equals \c classesAdmitted), and the run succeeded.
/// It is what the probe's driver returns its exit status from, so a closure that does
/// not close is a failed run and not a printed remark.
///
/// \ingroup boys
struct DeviceOptionClosure {
    std::size_t rows = 0; ///< rows of the library's own option table, BoysDeviceOptions()
    std::size_t forms = 0; ///< division forms the library reports, BoysDivisionForms()
    std::size_t total = 0; ///< the members of the space: one per row per form

    std::size_t measured = 0; ///< members this run took a figure for
    std::size_t refusedAndOwed = 0; ///< members of a row this build does not serve
    std::size_t notRunnable = 0; ///< members of a run whose tables the card would not hold
    std::size_t offeredNoFigure = 0; ///< places of this run's grid that produced no figure
    std::size_t notAsked = 0; ///< members this run's request never named

    /// The five states above summed, which is the left-hand side the report prints
    /// and the bar the space's own total is held to.
    std::size_t states = 0;

    /// Members in no state above. Zero for every run whose closure closes, and the
    /// count the verdict fails on.
    std::size_t unaccounted = 0;

    /// Places this run's own measurement table carries, and the number a run of this
    /// request owes the space: one per member — row and form — it serves and the request
    /// named. The two are read from different sources — the run's grid and the library's
    /// tables — and they have to agree.
    std::size_t gridPlaces = 0;
    std::size_t gridPlacesOwed = 0; ///< the places a run of this request owes the space

    /// The classes the space admits, and the classes the report carries.
    ///
    /// A class is one precision and one question, and which of them the space admits is
    /// read off the rows the request named — not off the report those rows are held to.
    /// \c classesPrinted is \c DeviceProbeReport::classes, and the two have to agree: a
    /// class the space admits and the report does not carry is a shape of this surface
    /// that nothing in the run reports on.
    std::size_t classesAdmitted = 0;
    std::size_t classesPrinted = 0; ///< the classes the report carries: DeviceProbeReport::classes

    /// Whether the closure holds; see this struct's own note.
    bool closed = false;
};

/// Counts the option space one report was taken over; see \c DeviceOptionClosure.
///
/// It reads the library's own tables and the report's own record of the run and
/// nothing else, so a report built by hand is counted the same way a measured one is
/// — which is what lets a test hold the counting without a card.
///
/// \param report the report to count
///
/// \returns the counts, with \c closed the verdict on them
///
/// \ingroup boys
DeviceOptionClosure DeviceOptionSpaceClosure(const DeviceProbeReport& report) noexcept;

} // namespace boys
