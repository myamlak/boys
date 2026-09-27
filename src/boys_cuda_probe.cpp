// The option probe's host side: the workload, the protocol, the folding of the
// rounds into figures, the refusal to order noise, and the text a consumer
// reads.
//
// The clock is not here. Every timed region is a CUDA event pair this file
// opens and closes through boys_cuda_probe_kernels.cu, because the host's own
// submission cost must not be in the figure and the events are the only thing
// that keeps it out. What this file decides is what to time, how many times, and
// what the results are allowed to say.
//
// The shape is the CPU option probe's (src/boys_probe.cpp in this library): every
// entry is timed once in every round, a comparison between two entries is the
// ratio of their per-round figures inside one round, the figure reported for an
// entry is a lower quartile of that entry's ratios to a reference entry scaled by
// the reference's own lower-quartile cost, the canary beside each pass is a
// diagnostic that gates nothing, the resolution is what the run measured rather
// than a bar chosen in advance, a rival whose band does not clear one is named as
// unplaced instead of being ordered, and a refusal is accompanied by a static
// fallback labelled as the heuristic it is. The differences are all forced by the
// device: the instrument is a kernel and not a host spin, the card is named
// instead of the host, and the entries that exist to run inside the caller's
// kernel are measured by subtraction rather than launched.
//
// One difference is worth naming here, because it is the one that survives the
// change rather than being forced by it: the two halves of an in-kernel row's
// subtraction are timed adjacent inside the *same round*, with their order
// alternating by round, so the difference that row's figure is made of is a
// within-round pair exactly like a ratio between two entries is, and a clock that
// drifts over the run moves both halves together and cancels in it. That is why
// the subtraction route did not need to be re-founded when the ordering moved to
// paired ratios; it was already paired.

#include "boys/boys_cuda_probe.hpp"

#include "boys/boys_coefficients.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_device_tables.hpp"
#include "boys/f16.hpp"

#include "boys_cuda_probe_entries.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// The device-side boundary (boys_cuda_probe_kernels.cu). Return codes as the
// lane's own exported functions: 0 success, 1 an internal error, 2 a CUDA
// failure.
// ---------------------------------------------------------------------------
extern "C" {
int BoysCudaProbeDeviceCount(int* out);
int BoysCudaProbeSetDevice(int ordinal);
int BoysCudaProbeCurrentDevice(int* out);
int BoysCudaProbeDeviceQuery(int ordinal, boys::probe_detail::ProbeDeviceFacts* out);
int BoysCudaProbeBuildFacts(boys::probe_detail::ProbeBuildFacts* out);
int BoysCudaProbeAlloc(void** out, std::size_t bytes);
int BoysCudaProbeFree(void* p);
int BoysCudaProbeUpload(void* dst, const void* src, std::size_t bytes);
int BoysCudaProbeDownload(void* dst, const void* src, std::size_t bytes);
int BoysCudaProbeSynchronize();
int BoysCudaProbeTime(boys::probe_detail::ProbeTimeRequest* request);
}

namespace boys {
namespace {

using probe_detail::ProbeEntry;
using probe_detail::ProbeQuestion;
using probe_detail::ProbeTimeRequest;
using probe_detail::ProbeWhat;

// ---------------------------------------------------------------------------
// The documented bounds each precision's lane carries at full accuracy. These
// are read from the library's documentation, not measured here: what a lane
// delivers on a card is the accuracy gate's business, and this probe spends its
// time on cost. They are in the report so that a faster precision is not read as
// a faster option at the same accuracy.
// ---------------------------------------------------------------------------
constexpr double kFp64Bound = 5.5e-14;
constexpr double kFp32Bound = 1.5e-7;
constexpr double kFp32FastBound = 1.5e-7 + 8e-8;
constexpr double kFp16Bound = 1e-7 + 0x1p-11;

/// How the workload's orders are drawn: the sum of two shell angular momenta,
// which is the distribution a shell-pair batch presents. The shells are capped
/// so that the sum can reach the workload's own highest order.
constexpr int kShellMax = 7;

std::string Text(const char* literal) {
    return std::string(literal);
}

template <typename... Args>
std::string Text(const char* format, Args... args) {
    const int size = std::snprintf(nullptr, 0, format, args...);

    if (size <= 0)
    {
        return std::string();
    }

    std::string out(static_cast<std::size_t>(size), '\0');
    std::snprintf(out.data(), static_cast<std::size_t>(size) + 1, format, args...);
    return out;
}

/// Slowest over fastest, less one, in percent. Zero for a set that cannot
/// disagree with itself.
double SpreadPercent(const std::vector<double>& samples) {
    if (samples.size() < 2)
    {
        return 0.0;
    }

    const auto ends = std::minmax_element(samples.begin(), samples.end());

    if (!(*ends.first > 0.0))
    {
        return 0.0;
    }

    return 100.0 * (*ends.second / *ends.first - 1.0);
}

double MedianOf(std::vector<double> samples) {
    if (samples.empty())
    {
        return 0.0;
    }

    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

/// The quantile \p p of a set of readings, by copy and sort, interpolating
/// between the two order statistics it falls between.
///
/// This is the statistic the reported cost is: a lower quartile rather than a
/// minimum, because the minimum of a run under a decaying clock is the earliest
/// and fastest observation rather than what a caller with a long workload meets,
/// and a quartile rather than a mean, because one disturbed round must not move
/// it. Its complement, the upper quartile, is what the spread beside it is made
/// of, so the two ends come from the same distribution.
double QuantileOf(std::vector<double> values, double p) {
    if (values.empty())
    {
        return 0.0;
    }

    std::sort(values.begin(), values.end());

    const double position = p * static_cast<double>(values.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(position);
    const std::size_t upper = std::min(lower + 1, values.size() - 1);
    const double weight = position - static_cast<double>(lower);

    return values[lower] * (1.0 - weight) + values[upper] * weight;
}

/// The quantile the reported cost and its ratio to the reference are read at.
///
/// A lower quartile, and not the minimum this probe used to report. On a card
/// whose clock decays under a sustained load - which is every card that boosts
/// opportunistically, and this is a load - the minimum of a run is its earliest
/// and best-clocked round, and a caller whose kernel runs for hours does not meet
/// that state. A quartile is what the bulk of such a run meets, and it still
/// discards the round a background process or a clock step spoiled. It is a
/// quartile and not a mean for the same reason: one disturbed round out of many
/// must not move a figure a consumer will build a default on.
constexpr double kStatisticQuantile = 0.25;

/// The number of paired rounds a quartile band needs before it can exist: the two
/// ends of a band are two order statistics, and below four observations they are
/// the same observation twice.
constexpr std::size_t kMinimumPairedRounds = 4;

/// The widest relative width of a within-round ratio band a set of rounds showed,
/// in percent: for each entry, the upper quartile of its ratio to \p reference
/// over the lower quartile of the same ratio, taken at its widest.
///
/// The reference's own column is skipped, and only it: its ratio is one in every
/// round by construction, so it would contribute a band of zero. A set too short
/// for a band, or one whose reference column holds a non-positive reading, reports
/// zero rather than a width it did not measure.
double PassPairedSpread(const std::vector<std::vector<double>>& rounds, std::size_t reference) {
    double widest = 0.0;

    if (rounds.empty() || reference >= rounds.front().size())
    {
        return 0.0;
    }

    // A round in which either the reference or the row could not be timed holds no
    // ratio, and is left out of every column rather than contributing an infinity:
    // a failed launch is not a slow one.
    for (const std::vector<double>& row : rounds)
    {
        if (!std::isfinite(row[reference]) || row[reference] <= 0.0)
        {
            return 0.0;
        }
    }

    for (std::size_t index = 0; index < rounds.front().size(); ++index)
    {
        if (index == reference)
        {
            continue;
        }

        std::vector<double> ratios;

        for (const std::vector<double>& row : rounds)
        {
            if (!std::isfinite(row[index]))
            {
                continue;
            }

            ratios.push_back(row[index] / row[reference]);
        }

        const double low = QuantileOf(ratios, kStatisticQuantile);
        const double high = QuantileOf(ratios, 1.0 - kStatisticQuantile);

        if (low > 0.0)
        {
            widest = std::max(widest, 100.0 * (high / low - 1.0));
        }
    }

    return widest;
}

/// One rival measured against one leader, inside the rounds.
///
/// Every quantity here is a ratio formed inside a single round: both entries were
/// timed under whatever clock that round ran at, so a drift common to the round
/// is in both terms of the ratio and cancels. This is the comparison the probe is
/// ordered by.
struct PairedOutcome {
    /// Lower and upper quartile of the within-round ratio, rival over leader: the
    /// band the middle half of the run put the pair in.
    double lo = 1.0;
    double hi = 1.0;

    /// Rounds in which the rival was the slower of the two, of the rounds that
    /// could be formed.
    int slowerRounds = 0;
    int rounds = 0;

    /// How far the pair's ratio moved between the run's first and second half of
    /// rounds: the second half's median ratio over the first half's, less one. A
    /// pair whose ratio is a property of the two entries holds still as the card's
    /// clock moves; one that drifts is a pair whose two entries do not carry the
    /// clock alike, which is the one way a paired comparison can still be misled
    /// by a card whose boost decays.
    double drift = 0.0;

    /// Whether the band clears one: the middle half of the run put the rival
    /// behind the leader, and the probe orders only what it saw in that half.
    bool ordered = false;
};

/// Measures one rival against one leader over the run's rounds.
///
/// The ratio is formed inside a round and never across rounds, so the pair is
/// compared under one clock. The band is the lower and upper quartile of those
/// per-round ratios, and the ordering is that lower quartile clearing one: the
/// middle half of the run has to put the rival behind, not merely the run's
/// average.
///
/// \param rounds the round table: one row per round, one column per entry, in
///               cost per argument
/// \param leader column of the entry the rival is measured against
/// \param rival  column of the entry being placed
///
/// \returns the pair's own band, its slower-round count and its drift
PairedOutcome CompareToLeader(const std::vector<std::vector<double>>& rounds, std::size_t leader,
                              std::size_t rival) {
    PairedOutcome outcome;
    std::vector<double> ratios;
    std::vector<double> early;
    std::vector<double> late;

    for (std::size_t round = 0; round < rounds.size(); ++round)
    {
        if (leader >= rounds[round].size() || rival >= rounds[round].size())
        {
            continue;
        }

        const double lead = rounds[round][leader];

        if (!std::isfinite(lead) || lead <= 0.0 || !std::isfinite(rounds[round][rival]))
        {
            continue;
        }

        const double ratio = rounds[round][rival] / lead;
        ratios.push_back(ratio);

        if (ratio > 1.0)
        {
            ++outcome.slowerRounds;
        }

        if (round * 2 < rounds.size())
        {
            early.push_back(ratio);
        } else
        {
            late.push_back(ratio);
        }
    }

    outcome.rounds = static_cast<int>(ratios.size());
    outcome.lo = QuantileOf(ratios, kStatisticQuantile);
    outcome.hi = QuantileOf(ratios, 1.0 - kStatisticQuantile);
    outcome.ordered = outcome.lo > 1.0;

    const double firstHalf = MedianOf(early);
    const double secondHalf = MedianOf(late);

    if (firstHalf > 0.0)
    {
        outcome.drift = secondHalf / firstHalf - 1.0;
    }

    return outcome;
}

/// A CUDA version as the runtime packs it, written the way a consumer reads it.
std::string FormatCudaVersion(int packed) {
    if (packed <= 0)
    {
        return "unknown";
    }

    return Text("%d.%d", packed / 1000, (packed % 1000) / 10);
}

/// Bytes as a consumer reads a memory size.
std::string FormatBytes(std::size_t bytes) {
    const double gib = static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
    return Text("%.1f GiB", gib);
}

/// The status in words, so a caller reading the text does not have to know the
/// enumerators to see why nothing was measured.
const char* StatusName(DeviceProbeStatus status) {
    switch (status)
    {
        case DeviceProbeStatus::kSuccess:
            return "measured";
        case DeviceProbeStatus::kNoDevice:
            return "this machine has no CUDA device";
        case DeviceProbeStatus::kDeviceNotFound:
            return "the device asked for is not one this machine has";
        case DeviceProbeStatus::kInvalidArgument:
            return "the request named no entry this library has";
        default:
            return "a CUDA operation failed";
    }
}

// ---------------------------------------------------------------------------
// The entries this build offers, as the library reports them.
// ---------------------------------------------------------------------------

/// One row of the library's report, with the label a probe row prints.
struct EntryInfo {
    ProbeEntry entry;
    const char* name;
    const char* precision;
    const char* shape;
    ProbeQuestion question;
    bool inKernel;
    double bound;
    /// The degree tables the row's lane reads, and whose seed it amplifies, as
    /// the library's own option book states it. It is carried so that the static
    /// fallback can count a row's arithmetic from the same tables the row runs
    /// on, rather than from a second mapping written here.
    BoysDeviceLane lane;
    /// Which region-B exponential the row evaluates, as the option book states
    /// it, so the fallback can prefer the library's own documented default.
    RegionBExp regionBExp;
};

const char* QuestionName(ProbeQuestion question) {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "single";
        case ProbeQuestion::kAllOrders:
            return "all-orders";
        default:
            return "all-n";
    }
}

/// The order the classes print and rank in: one question per class, and the
/// library's enumerators in the order the report's prose names them.
int QuestionRank(ProbeQuestion question) {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return 0;
        case ProbeQuestion::kAllOrders:
            return 1;
        default:
            return 2;
    }
}

/// The library's question, said in full: what the option produces for one
/// argument, which is what makes two of them the same question.
const char* QuestionAsked(ProbeQuestion question) {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "one value per argument: F_n(x) at that argument's own order n";
        case ProbeQuestion::kAllOrders:
            return "a ladder per argument: F_0(x)..F_n(x) at that argument's own order n";
        default:
            return "one common ladder for the whole batch: F_0(x)..F_nmax(x) for every argument";
    }
}

/// The precision, and the degree-table lane behind it, as the report spells
/// both. The lane is not printed today � every row of a precision class reads
/// one lane's tables in this build � and it is carried so that a probe which
/// later separates the lanes can do it without asking the library again.
const char* PrecisionName(DeviceOptionPrecision precision) {
    switch (precision)
    {
        case DeviceOptionPrecision::kFp64:
            return "fp64";
        case DeviceOptionPrecision::kFp32:
            return "fp32";
        default:
            return "fp16";
    }
}

const char* ShapeName(DeviceOptionShape shape) {
    switch (shape)
    {
        case DeviceOptionShape::kSingle:
            return "single";
        case DeviceOptionShape::kAllOrders:
            return "all-orders";
        case DeviceOptionShape::kAllN:
            return "all-n";
        default:
            return "each-order";
    }
}

const char* GroupName(DeviceOptionGroup group) {
    return group == DeviceOptionGroup::kLaunched ? "launched" : "device";
}

/// The axis a row varies, with the member it is. A row with no axis states the
/// one thing its entry is; a row with one states which member of it the row
/// measured, because that is what its bound is the bound of.
///
/// The member is the one the row's own axis names: a row that varies the
/// scheme carries a scheme and not a region-B exponential, so naming the
/// exponential for it would state a choice the row does not offer.
std::string AxisName(const DeviceOptionInfo& option) {
    switch (option.axis)
    {
        case DeviceOptionAxis::kNone:
            return std::string("-");
        case DeviceOptionAxis::kRegionBExp:
            return Text("region-B:%s",
                        option.regionBExp == RegionBExp::kFast ? "fast" : "accurate");
        case DeviceOptionAxis::kPartition:
            return "partition:narrow";
        case DeviceOptionAxis::kPacking:
            return "packing:per-order";
        case DeviceOptionAxis::kScheme:
            return std::string("scheme:")
                   + (option.scheme == EvalScheme::kHorner ? "horner" : "split-clenshaw");
        default:
            // An axis named by a newer header: the row is still printed, under
            // the enumerator it was written with rather than under a member of
            // some other axis.
            return Text("axis:%d", static_cast<int>(option.axis));
    }
}

/// The degree tables a row reads. The lane is what makes two rows of one
/// precision different arithmetic — the seed amplification it carries is the
/// lane's — so a report that names the precision and not the lane has not said
/// which arithmetic it measured.
const char* LaneName(BoysDeviceLane lane) {
    switch (lane)
    {
        case BoysDeviceLane::kF64Single:
            return "f64-single";
        case BoysDeviceLane::kF64Batch:
            return "f64-batch";
        case BoysDeviceLane::kF32Single:
            return "f32-single";
        case BoysDeviceLane::kF32Batch:
            return "f32-batch";
        case BoysDeviceLane::kF16Single:
            return "f16-single";
        default:
            return "f16-batch";
    }
}

/// The sentence a class opens with: what a class is, and what a ranking inside
/// it therefore ranks.
///
/// The class is one precision, and the reason it is keyed on precision rather
/// than on the accuracy a row achieves is the reason this sentence exists: a
/// caller has already chosen fp64, fp32 or fp16 from the accuracy their
/// calculation needs, so a row that trades accuracy for speed is a faster way to
/// compute the precision they chose rather than a different precision. The
/// bounds the rows document are then their own, and whether one class holds two
/// of them is a fact about the rows, so it is read off them rather than assumed.
std::string PrecisionNote(const std::string& precision, bool oneBound) {
    std::string text = Text("every entry in this class runs %s. The class is one precision and not "
                            "one accuracy: precision is the choice the caller made from the "
                            "accuracy their calculation needs, so it is not traded for speed "
                            "here, and no entry of this class is ordered against an entry of "
                            "another. Each shape is ranked on its own, so the winner of a shape "
                            "is the fastest %s entry of that shape at the bound its own row "
                            "states, and the bound column above is that figure.",
                            precision.c_str(),
                            precision.c_str());

    text += oneBound
                ? std::string(" The rows of this class document one bound between them, so each "
                              "of its rankings is at one accuracy.")
                : std::string(" The rows of this class do not document one bound between them: a "
                              "row of a looser bound is a faster way to compute this precision at "
                              "the accuracy that row names, and both stand in the one shape "
                              "ranking, which is what makes that ranking a ranking by cost.");

    return text;
}

/// The option table, projected from the library's own report: one probe row per
/// report row this build serves, and one unoffered entry per report row it does
/// not, carrying the library's reason.
///
/// Nothing here names an option the library does not report. A row the library
/// adds appears in this probe's table, and a row it refuses appears in the
/// probe's list of what this build cannot serve, both without an edit here.
std::vector<EntryInfo> EnumerateEntries(std::vector<std::string>& unoffered,
                                       std::vector<std::string>& refusedBecause) {
    std::vector<EntryInfo> entries;

    for (const DeviceOptionInfo& option : BoysDeviceOptions())
    {
        if (!option.built)
        {
            unoffered.push_back(option.name);
            refusedBecause.push_back(option.refusedBecause != nullptr ? option.refusedBecause
                                                                     : "not served by this build");
            continue;
        }

        entries.push_back(EntryInfo{option.entry,
                                    option.name,
                                    PrecisionName(option.precision),
                                    ShapeName(option.shape),
                                    option.question,
                                    option.group == DeviceOptionGroup::kDeviceCallable,
                                    option.bound,
                                    option.lane,
                                    option.regionBExp});
    }

    return entries;
}

// ---------------------------------------------------------------------------
// The workload.
// ---------------------------------------------------------------------------

/// The library's own question, as the CPU probe asks it: a set of arguments in
/// which each argument carries its own highest order, drawn as the sum of two
/// shell angular momenta, over a log-uniform range that populates every region
/// of the kernel.
///
/// The pairs are sorted by argument because the uniform-order entries document
/// that their arguments are non-decreasing: this probe does not sort for them,
/// so it has to present them a batch that already satisfies what they promise.
struct Workload {
    std::vector<int> n;
    std::vector<double> x;
    std::vector<float> xf;
    std::vector<std::uint16_t> xh;
};

Workload BuildWorkload(const DeviceProbeOptions& options) {
    Workload work;
    const std::size_t count = options.count;

    work.n.resize(count);
    work.x.resize(count);

    std::uint64_t state = options.seed;

    const auto next = [&state]() {
        state = state * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<std::uint32_t>(state >> 33);
    };

    const double logSpan = std::log(options.xHi / options.xLo);

    for (std::size_t i = 0; i < count; ++i)
    {
        const int shellA = static_cast<int>(next() % static_cast<std::uint32_t>(kShellMax + 1));
        const int shellB = static_cast<int>(next() % static_cast<std::uint32_t>(kShellMax + 1));
        work.n[i] = std::min(shellA + shellB, options.nmax);

        const double u = static_cast<double>(next()) / 4294967296.0;
        work.x[i] = options.xLo * std::exp(u * logSpan);
    }

    // Sort the (order, argument) pairs by argument, so the uniform-order
    // entries' documented precondition holds.
    std::vector<std::size_t> order(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        order[i] = i;
    }

    std::sort(order.begin(), order.end(), [&work](std::size_t a, std::size_t b) {
        return work.x[a] < work.x[b];
    });

    std::vector<int> sortedN(count);
    std::vector<double> sortedX(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        sortedN[i] = work.n[order[i]];
        sortedX[i] = work.x[order[i]];
    }

    work.n.swap(sortedN);
    work.x.swap(sortedX);

    // The narrow lanes' argument arrays. The batch entries of the fp32 and fp16
    // lanes take the double argument and round it themselves; the device-callable
    // entries of those lanes take the narrowed argument, which is what a caller's
    // own kernel would hold in a register.
    work.xf.resize(count);
    work.xh.resize(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        work.xf[i] = static_cast<float>(work.x[i]);
        work.xh[i] = F16(static_cast<float>(work.x[i])).Bits();
    }

    return work;
}

// ---------------------------------------------------------------------------
// Device memory, owned for the length of the measurement. Everything is
// allocated and filled once, before the first clock, and freed after the last.
// ---------------------------------------------------------------------------

struct DeviceBuffer {
    void* pointer = nullptr;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept : pointer(other.pointer) {
        other.pointer = nullptr;
    }
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other)
        {
            if (pointer != nullptr)
            {
                BoysCudaProbeFree(pointer);
            }

            pointer = other.pointer;
            other.pointer = nullptr;
        }

        return *this;
    }
    ~DeviceBuffer() {
        if (pointer != nullptr)
        {
            BoysCudaProbeFree(pointer);
        }
    }
};

/// The calling thread's device, put back where it was however this returns.
struct DeviceGuard {
    int previous = 0;
    bool ok = false;

    DeviceGuard() {
        ok = BoysCudaProbeCurrentDevice(&previous) == 0;
    }

    ~DeviceGuard() {
        if (ok)
        {
            BoysCudaProbeSetDevice(previous);
        }
    }
};

// ---------------------------------------------------------------------------
// One timed region.
// ---------------------------------------------------------------------------

/// Device milliseconds of \p reps back-to-back launches of whatever the request
/// names. A non-positive result is a failure, never a fast entry.
bool TimeRegion(ProbeTimeRequest request, double& outMs) {
    request.outMs = &outMs;
    return BoysCudaProbeTime(&request) == 0 && outMs >= 0.0;
}

/// Nanoseconds per argument, which is the unit every figure in this report is
/// in: one argument's worth of the question the entry was asked.
double PerArgument(double milliseconds, int reps, std::size_t count) {
    return milliseconds * 1.0e6 / static_cast<double>(reps) / static_cast<double>(count);
}

/// What one entry's timed region or regions produced, in device milliseconds.
///
/// A launched row fills \c withMs and leaves \c withoutMs at zero. An in-kernel
/// row fills both: the caller's kernel with the call in it, and the same kernel
/// with the call removed and the traffic kept. Both halves are kept, not only
/// their difference, so the report can print the subtraction rather than assert
/// that one happened.
struct TimedEntry {
    double withMs = 0.0;
    double withoutMs = 0.0;
    bool subtracted = false;
    bool ok = false;
};

/// One entry's timed figure for one round, in milliseconds of device time.
///
/// A launched entry is one region. A device-callable entry is two, taken adjacent
/// in the same round: two adjacent regions inside one round and not two regions
/// in different rounds, so a drift in the card's clocks moves both halves together
/// and cancels in the difference. That is what makes an in-kernel row's figure a
/// within-round pair of the same kind a ratio between two entries is, and it is
/// why the subtraction route did not need re-founding when the ordering moved to
/// paired ratios.
///
/// Which half goes first alternates by round, and both halves of every round are
/// kept, so a per-round difference can be formed from them and each half is timed
/// going first in half the rounds. A region carries a small fixed cost of its own
/// — the event pair, the first launch's cold start, the host's submission of the
/// launch — which is divided by the repetition count rather than by the call, and
/// whichever half pays it lands in the difference as a term that shrinks as the
/// repetition count rises. That is a bias, not noise: it does not average away,
/// and it moves the figure between two repetition counts, which is what the
/// subtraction control found. Alternating the order spreads it over both halves
/// rather than leaving it on one.
TimedEntry TimeEntry(const EntryInfo& info,
                     const ProbeTimeRequest& base,
                     int reps,
                     bool baselineFirst = false) {
    TimedEntry timed;

    if (!info.inKernel)
    {
        ProbeTimeRequest request = base;
        request.what = static_cast<int>(ProbeWhat::kLaunchedEntry);
        request.entry = static_cast<int>(info.entry);
        request.reps = reps;
        timed.ok = TimeRegion(request, timed.withMs);
        return timed;
    }

    ProbeTimeRequest withBoys = base;
    withBoys.what = static_cast<int>(ProbeWhat::kInKernelEntry);
    withBoys.entry = static_cast<int>(info.entry);
    withBoys.reps = reps;
    withBoys.withBoys = 1;

    ProbeTimeRequest withoutBoys = withBoys;
    withoutBoys.withBoys = 0;

    timed.subtracted = true;

    if (baselineFirst)
    {
        timed.ok = TimeRegion(withoutBoys, timed.withoutMs) && TimeRegion(withBoys, timed.withMs);
    } else
    {
        timed.ok = TimeRegion(withBoys, timed.withMs) && TimeRegion(withoutBoys, timed.withoutMs);
    }

    return timed;
}

/// Times one entry at two very different repetition counts and fills a
/// repetition-count control from what the two said.
///
/// The counts have to be far apart for the check to mean anything: a cost that
/// repeats with the launch rather than with the call is divided by a different
/// number of launches at each count, so it changes the per-argument figure by
/// that cost's share of it times the ratio of the counts. An entry whose figure
/// holds across the two has had such a cost divided out.
///
/// Both halves of a subtraction are reported, not only their difference, so that
/// a reader can see the difference was not two unstable readings cancelling.
/// Returns false when either count could not be timed.
///
/// Each count is read over \c kMinimumPairedRounds rounds with the order of the
/// two halves alternating, and each count's figure is the lower quartile of its
/// own rounds' per-argument readings, with an in-kernel round's difference formed
/// inside the round. That is the statistic the reported figures are, so the two
/// counts are compared as figures of one kind rather than as a quartile against a
/// minimum. **The check itself is not softened by it**: a count that resolved no
/// difference leaves no second figure to compare, and a run that cannot resolve
/// the row cannot vouch for its repetition counts either, so both of those fail
/// the control rather than passing it.
bool RunRepetitionControl(const EntryInfo& info,
                          const DeviceProbeMeasurement& row,
                          const ProbeTimeRequest& base,
                          const DeviceProbeOptions& clamped,
                          double judgedAgainst,
                          DeviceProbeRepetitionControl& out) {
    out.entry = row.name;
    out.route = row.route;
    out.repetitionsLow = clamped.controlRepetitionsLow;
    out.repetitionsHigh = clamped.controlRepetitionsHigh;
    out.judgedAgainst = judgedAgainst;

    const int kRounds = static_cast<int>(kMinimumPairedRounds);

    std::vector<double> lowRounds;
    std::vector<double> highRounds;
    std::vector<double> baselineLow;
    std::vector<double> baselineHigh;
    bool subtracted = false;
    bool ok = true;

    for (int round = 0; round < kRounds && ok; ++round)
    {
        // Alternated, so the per-region cost that whichever half goes second
        // pays is not read as a difference between the halves.
        const bool baselineFirst = (round % 2) == 1;
        const TimedEntry low = TimeEntry(info, base, clamped.controlRepetitionsLow, baselineFirst);
        const TimedEntry high =
            TimeEntry(info, base, clamped.controlRepetitionsHigh, baselineFirst);

        ok = low.ok && high.ok;
        subtracted = low.subtracted && high.subtracted;

        if (!ok)
        {
            break;
        }

        // The same shape the figures are, formed inside the round: a launched row
        // is its region, an in-kernel row the difference between the region that
        // held the call and the one that did not, both timed together here.
        const auto Figure = [&](double withMs, double withoutMs, int reps) {
            const double ms = subtracted ? std::max(0.0, withMs - withoutMs) : withMs;
            return PerArgument(ms, reps, clamped.count);
        };

        lowRounds.push_back(Figure(low.withMs, low.withoutMs, clamped.controlRepetitionsLow));
        highRounds.push_back(Figure(high.withMs, high.withoutMs, clamped.controlRepetitionsHigh));

        if (subtracted)
        {
            baselineLow.push_back(
                PerArgument(low.withoutMs, clamped.controlRepetitionsLow, clamped.count));
            baselineHigh.push_back(
                PerArgument(high.withoutMs, clamped.controlRepetitionsHigh, clamped.count));
        }
    }

    if (!ok)
    {
        out.entry.clear();
        out.note = Text("'%s' at %d and %d launches could not be timed on this run, so what the "
                        "%s figures exclude is not established by this run",
                        row.name.c_str(),
                        clamped.controlRepetitionsLow,
                        clamped.controlRepetitionsHigh,
                        row.route.c_str());
        return false;
    }

    out.nsPerArgumentLow = QuantileOf(lowRounds, kStatisticQuantile);
    out.nsPerArgumentHigh = QuantileOf(highRounds, kStatisticQuantile);

    if (subtracted)
    {
        out.nsPerArgumentBaselineLow = QuantileOf(baselineLow, kStatisticQuantile);
        out.nsPerArgumentBaselineHigh = QuantileOf(baselineHigh, kStatisticQuantile);
    }

    const double smaller = std::min(out.nsPerArgumentLow, out.nsPerArgumentHigh);
    // A count at which the subtraction resolved nothing leaves no second figure
    // to compare, and the one outcome that must never come out of that is
    // agreement: the control would be passing without evidence, and this is the
    // check the whole subtraction route rests on. The difference is infinite
    // rather than zero, so the data says what the note says.
    const bool resolved = smaller > 0.0;

    out.difference = resolved
                         ? std::fabs(out.nsPerArgumentHigh - out.nsPerArgumentLow) / smaller
                         : std::numeric_limits<double>::infinity();
    // The widest within-round band this row's own shape showed on this run is
    // what the two counts are judged against - the same number that shape's
    // refusal prints - and not a bar invented here. A run that could not resolve
    // the row has no yardstick for its repetition counts either, and the control
    // says so rather than passing.
    out.agrees = judgedAgainst > 0.0 && out.difference <= judgedAgainst;

    const std::string comparison =
        resolved
            ? Text("a disagreement of %.2f%%, %s the %.2f%% this shape could order at (the widest "
                   "within-round ratio band its own rows showed over this run), which is the check "
                   "that a cost repeating with the launch rather than with the call did not survive "
                   "into the figures at the repetition count they were taken at",
                   100.0 * out.difference,
                   out.agrees ? "inside" : "OUTSIDE",
                   100.0 * judgedAgainst)
            : Text("no disagreement to report, because at %d launches the subtraction resolved no "
                   "cost at all: there is no second figure to set beside the one at %d launches, "
                   "so this control has not established that the figure holds as the launches are "
                   "repeated and the row is not ordered on it",
                   clamped.controlRepetitionsHigh,
                   clamped.controlRepetitionsLow);

    const std::string halves =
        subtracted
            ? Text("the subtraction's own two halves were %.3f ns/argument with the call against "
                   "%.3f without it at %d launches, and %.3f against %.3f at %d, so the difference "
                   "is between two readings that were themselves stable, not two unstable ones "
                   "cancelling",
                   out.nsPerArgumentBaselineLow + out.nsPerArgumentLow,
                   out.nsPerArgumentBaselineLow,
                   clamped.controlRepetitionsLow,
                   out.nsPerArgumentBaselineHigh + out.nsPerArgumentHigh,
                   out.nsPerArgumentBaselineHigh,
                   clamped.controlRepetitionsHigh)
            : Text("the region holds no subtraction, so there is one reading per count rather "
                   "than two");

    out.note = Text("'%s' on the %s route at %d and %d launches gives %.3f against %.3f "
                    "ns/argument: %s; %s",
                    row.name.c_str(),
                    row.route.c_str(),
                    clamped.controlRepetitionsLow,
                    clamped.controlRepetitionsHigh,
                    out.nsPerArgumentLow,
                    out.nsPerArgumentHigh,
                    comparison.c_str(),
                    halves.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// The conclusion for one question shape of one precision.
// ---------------------------------------------------------------------------

/// Folds one shape's live measurements into a verdict, on the CPU probe's rule:
/// the shape's leader is the entry with the lowest ratio to the reference at the
/// lower quartile of the paired rounds, and it stands only when every rival's own
/// within-round band against it has a lower quartile above one. A rival whose band
/// straddles one cannot be placed, and a rival whose band lies wholly below one was
/// ahead in that half, which means the statistic and the rounds disagree about the
/// pair; both end in a refusal that names each unplaced rival with its band and its
/// slower-round count and prints the run's own resolution. The ordering is never
/// weakened into a ranking: no winner is named while any rival is unplaced.
///
/// Every row handed in is already of one precision and one question shape, which
/// is what makes them comparable at all. The bounds those rows document are
/// columns of their own rows and are not what this orders on: two rows at
/// different bounds are still two ways to compute the precision the caller
/// chose, which is why the class holds them together.
///
/// \param clause the ranking to fill in
/// \param live   the shape's measured rows, in the report's own order
/// \param columns each live row's own column in \p rounds, in the same order as
///               \p live
/// \param rounds the round table the report's figures were aggregated from: one
///               row per pooled round, one column per entry in the measurements'
///               own order, in cost per argument, so a pair can be compared
///               inside a round
/// \param pairedRounds rounds the run pooled, which is what a quartile band needs
///               four of
/// The clock check, stated: whether any pair of the shape moved between the run's
/// first and second half of rounds by more than the run can order.
///
/// It is appended to a ranking's confidence line on every path that followed a
/// pair, a refusal included, because a reader of a refusal is asking exactly this:
/// whether the shape could not be ordered because its entries are close or because
/// the clock moved under the run. It carries its own leading punctuation because it
/// is always a sentence added to one already written.
std::string DriftClause(const std::string& driftPair, double widestDrift, double resolution) {
    if (std::abs(widestDrift) > resolution) {
        return Text(". WARNING: the pair %s moved %.1f%% between the run's first and second half of "
                    "rounds, beyond the %.2f%% this run can order, so those two entries are not "
                    "equally exposed to this card's clock and the conclusion above is a property "
                    "of this run's clock as well as of the entries.",
                    driftPair.c_str(),
                    100.0 * widestDrift,
                    100.0 * resolution);
    }

    return Text(". No pair's ratio moved between the run's halves by more than the %.2f%% this run "
                "can order (the widest was %s at %.1f%%), so no measured pair was more exposed to "
                "the clock's drift than the run's own resolution is wide.",
                100.0 * resolution,
                driftPair.empty() ? "none" : driftPair.c_str(),
                100.0 * widestDrift);
}

void Conclude(DeviceProbeRanking& clause,
              const std::vector<DeviceProbeMeasurement*>& live,
              const std::vector<std::size_t>& columns,
              const std::vector<std::vector<double>>& rounds,
              int pairedRounds) {
    clause.verdict = DeviceProbeVerdict::kCannotDetermine;
    clause.rounds = pairedRounds;

    /// A row of this shape with the column its rounds were timed in, which is what
    /// a pair against another of them is formed from.
    struct Candidate {
        DeviceProbeMeasurement* row = nullptr;
        std::size_t column = 0;
    };

    // Two kinds of row may not be ranked: an in-kernel row whose subtraction did
    // not resolve, and a row whose repetition control disagreed. Neither may be
    // silently dropped either, because a reader who expected it in the table
    // would take its absence for a build fact.
    std::vector<Candidate> ordered;

    for (std::size_t index = 0; index < live.size(); ++index)
    {
        DeviceProbeMeasurement* measurement = live[index];
        const std::size_t column = index < columns.size() ? columns[index] : 0u;

        if (!measurement->subtractionResolved)
        {
            clause.notOrdered.push_back(
                Text("%s (in-kernel): the caller kernel with the call in it cost the same as the "
                     "same kernel with the call removed, within this run's clock; its arithmetic "
                     "is inside the traffic that kernel already does at this workload, so no cost "
                     "was resolved for it",
                     measurement->name.c_str()));
        } else if (measurement->repetitionChecked && !measurement->repetitionAgrees)
        {
            clause.notOrdered.push_back(
                Text("%s: %s", measurement->name.c_str(), measurement->repetitionNote.c_str()));
        } else
        {
            ordered.push_back({measurement, column});
        }
    }

    if (ordered.empty())
    {
        clause.reason = Text("no entry of this shape produced a figure that resolved above the "
                             "%s it was measured against and held across its repetition control, "
                             "so there is nothing to order",
                             live.empty() ? "instrument" : "kernel without the call");
        clause.confidence =
            live.empty()
                ? std::string("CANNOT DETERMINE: no entry of this shape produced a figure at all")
                : Text("CANNOT DETERMINE: %zu %s of this shape measured and none resolved",
                       live.size(),
                       live.size() == 1 ? "entry" : "entries");
        return;
    }

    const auto cheaper = [](const Candidate& a, const Candidate& b) {
        return a.row->nsPerArgument < b.row->nsPerArgument;
    };

    const Candidate leader = *std::min_element(ordered.begin(), ordered.end(), cheaper);

    clause.fastestOverall = leader.row->name;

    // Every pair of this shape, measured inside the rounds. Done once and kept,
    // because the resolution, the refusal and the confidence line are all made of
    // it and a second computation could disagree with the first.
    struct Rival {
        Candidate candidate;
        PairedOutcome pair;
    };

    std::vector<Rival> rivals;
    double widestBand = std::max(0.0, leader.row->spread - 1.0);
    double widestDrift = 0.0;
    std::string driftPair;

    for (const Candidate& candidate : ordered)
    {
        if (candidate.row == leader.row)
        {
            continue;
        }

        const PairedOutcome pair = CompareToLeader(rounds, leader.column, candidate.column);
        rivals.push_back({candidate, pair});

        if (pair.lo > 0.0)
        {
            widestBand = std::max(widestBand, pair.hi / pair.lo - 1.0);
        }

        widestBand = std::max(widestBand, candidate.row->spread - 1.0);

        if (std::abs(pair.drift) > std::abs(widestDrift))
        {
            widestDrift = pair.drift;
            driftPair = Text("'%s' against '%s'", candidate.row->name.c_str(), leader.row->name.c_str());
        }
    }

    clause.resolution = widestBand;

    // A quartile band needs four rounds. With fewer, the lower and upper quartiles
    // are two- and three-point order statistics, and a probe that ordered on them
    // would be reporting a resolution it never measured.
    if (pairedRounds < static_cast<int>(kMinimumPairedRounds))
    {
        clause.reason = Text("'%s' is the fastest entry of this shape by the run's own statistic "
                             "(%.3f ns/argument at the lower quartile of its rounds), but the run "
                             "took %d paired round(s), and the lower and upper quartiles of a "
                             "within-round ratio need %zu of them: with fewer, the band is the "
                             "ratio of two or three rounds rather than a quartile of many, and "
                             "this probe will not order entries on it. Raising "
                             "DeviceProbeOptions::passes or DeviceProbeOptions::rounds is what "
                             "this needs",
                             leader.row->name.c_str(),
                             leader.row->nsPerArgument,
                             pairedRounds,
                             kMinimumPairedRounds);
        clause.confidence = Text("CANNOT DETERMINE: %d paired round(s), and the comparison needs %zu",
                                 pairedRounds,
                                 kMinimumPairedRounds);
        return;
    }

    if (!(leader.row->nsPerArgument > 0.0))
    {
        // A per-argument figure of zero is not a fast entry: it is a timed region
        // whose device time rounded away, which happens when the workload is too
        // small for the clock that measured it. Ordering a shape on that would
        // rank the workload rather than the arithmetic.
        clause.reason = Text("no entry of this shape produced a figure above zero: at this "
                             "workload and this repetition count the timed regions' device time "
                             "rounded away, so the shape is not ordered. Raising the argument "
                             "count or the repetition count is what this needs.");
        clause.confidence = "CANNOT DETERMINE: the fastest figure in this shape is zero, which is "
                            "the timer's resolution rather than a cost";
        return;
    }

    if (ordered.size() < 2)
    {
        // One entry is not a ranking. What the one figure says is that the entry
        // is the fastest of the one entry this run measured of its shape; what it
        // does not say is that the entry beats anything, and shipping it as a
        // recommendation would be shipping a field of one as a result.
        clause.reason = Text("'%s' is the fastest entry of this shape (%.3f ns/argument at the "
                             "lower quartile of the %d paired rounds), but it is the only entry of "
                             "it this run could order, so there is nothing to order it against: a "
                             "ranking needs a second entry of the same shape and precision, and "
                             "this run has one",
                             leader.row->name.c_str(),
                             leader.row->nsPerArgument,
                             pairedRounds);
        clause.confidence = Text("CANNOT DETERMINE: one entry of this shape in this class could be "
                                 "ordered, and one entry is not a ranking");
        return;
    }

    // The rivals, each placed by its own band against the leader and by nothing
    // else. A band whose lower quartile does not clear one is a pair this run did
    // not put in order, in either direction, and the probe will not name a winner
    // while one of those stands.
    std::vector<std::string> within;
    std::vector<std::string> ahead;
    const DeviceProbeMeasurement* nearest = nullptr;
    PairedOutcome nearestPair;

    for (const Rival& rival : rivals)
    {
        const DeviceProbeMeasurement* row = rival.candidate.row;

        const std::string line = Text(
            "'%s' at %.3f ns/argument, %.1f%% of the leader's: its within-round ratio to '%s' over "
            "the %d paired rounds fell in %.3f..%.3f, and it was the slower of the two in %d of "
            "them (its own rounds spread %.2fx)",
            row->name.c_str(),
            row->nsPerArgument,
            100.0 * (row->nsPerArgument / leader.row->nsPerArgument),
            leader.row->name.c_str(),
            rival.pair.rounds,
            rival.pair.lo,
            rival.pair.hi,
            rival.pair.slowerRounds,
            row->spread);

        if (nearest == nullptr || row->nsPerArgument < nearest->nsPerArgument)
        {
            nearest = row;
            nearestPair = rival.pair;
        }

        if (rival.pair.ordered)
        {
            continue;
        }

        if (rival.pair.hi < 1.0)
        {
            ahead.push_back(line);
        } else
        {
            within.push_back(line);
        }
    }

    if (!within.empty() || !ahead.empty())
    {
        std::vector<std::string> unplaced = within;
        unplaced.insert(unplaced.end(), ahead.begin(), ahead.end());
        clause.inseparable = unplaced;

        clause.reason = Text(
            "'%s' is the fastest entry of this shape by the run's own statistic - %.3f "
            "ns/argument at the lower quartile of the %d paired rounds, %.3f at the upper - but "
            "%zu of the %zu entry(s) of this shape could not be placed behind it: %s. The band and "
            "the slower-round count of every one of them are listed below. Ordering a pair whose "
            "band straddles one would be ordering noise, and ordering one whose band lies below "
            "one would contradict the run's own statistic, so the probe does neither: it reports "
            "what it could not separate, and the static fallback below stands in for the default",
            leader.row->name.c_str(),
            leader.row->nsPerArgument,
            pairedRounds,
            leader.row->nsPerArgumentMax,
            unplaced.size(),
            ordered.size(),
            !within.empty() ? within.front().c_str() : ahead.front().c_str());
        clause.confidence = Text("CANNOT DETERMINE: %zu of %zu entry(s) of this shape could not be "
                                 "ordered against '%s' over the %d paired rounds; the widest band "
                                 "this shape showed was %.2f%%",
                                 unplaced.size(),
                                 ordered.size(),
                                 leader.row->name.c_str(),
                                 pairedRounds,
                                 100.0 * clause.resolution);

        // The clock check belongs here as much as on a recommendation: a refusal
        // is where a reader wants to know whether the entries were too close to
        // separate or whether the machine moved under the run. Guarded by the same
        // rule as above — this branch is only reached with a quartile band's worth
        // of rounds, and every path short of that has already returned.
        clause.confidence += DriftClause(driftPair, widestDrift, clause.resolution);
        return;
    }

    clause.verdict = DeviceProbeVerdict::kRecommend;
    clause.recommended = leader.row->name;

    clause.reason = Text("'%s' is the fastest entry of this shape: %.3f ns/argument at the lower "
                         "quartile of the %d paired rounds, %.3f at the upper (spread %.2fx), "
                         "%.3f ns/argument at its peak. Every other entry of the shape was the "
                         "slower of the two in the middle half of the same rounds.",
                         leader.row->name.c_str(),
                         leader.row->nsPerArgument,
                         pairedRounds,
                         leader.row->nsPerArgumentMax,
                         leader.row->spread,
                         leader.row->nsPerArgumentPeak);

    // The clock check, done rather than assumed. A pair whose ratio moves between
    // the run's halves is a pair whose two entries do not carry a decaying clock
    // alike, and the report says so instead of implying the ordering holds at any
    // clock. A pair whose ratio held still within the run's own resolution puts no
    // such caveat on the ordering.
    clause.confidence =
        nearest != nullptr
            ? Text("HIGH: the nearest entry of this shape, '%s', took %.1f%% of the leader's cost "
                   "and was the slower of the two in %d of the %d paired rounds, its within-round "
                   "band %.3f..%.3f clearing one",
                   nearest->name.c_str(),
                   100.0 * (nearest->nsPerArgument / leader.row->nsPerArgument),
                   nearestPair.slowerRounds,
                   nearestPair.rounds,
                   nearestPair.lo,
                   nearestPair.hi)
            : std::string("HIGH: it is the only entry this run measured in this shape, so there is "
                          "no rival of it to fall inside a resolution");

    if (nearest == nullptr)
    {
        // Nothing to say: no rival was measured, so no pair was followed.
    } else
    {
        // The clock check, done rather than assumed. A pair whose ratio moves
        // between the run's halves is a pair whose two entries do not carry a
        // decaying clock alike, and the report says so instead of implying the
        // ordering holds at any clock. A pair whose ratio held still within the
        // run's own resolution puts no such caveat on the ordering.
        clause.confidence += DriftClause(driftPair, widestDrift, clause.resolution);
    }
}

/// The widest within-round ratio band one shape showed on this run, as a fraction.
///
/// It is the number the shape's refusal prints and the number the repetition
/// control's two counts are judged against, computed once from the pairs rather
/// than twice from two thresholds: an entry's own band to the reference, every
/// rival's band against the shape's fastest entry, and the fastest entry's own
/// band, taken at their widest.
///
/// A shape with fewer than two rows that could be ordered has no pair to form, and
/// reports the fastest row's own band, which is what one row's rounds can say
/// about themselves.
double ShapeResolution(const std::vector<DeviceProbeMeasurement>& measurements,
                       const std::vector<std::vector<double>>& rounds,
                       const std::string& precision,
                       const std::string& question) {
    std::vector<std::size_t> columns;

    for (std::size_t index = 0; index < measurements.size(); ++index)
    {
        if (measurements[index].measured && measurements[index].subtractionResolved &&
            measurements[index].precision == precision &&
            measurements[index].question == question)
        {
            columns.push_back(index);
        }
    }

    if (columns.empty())
    {
        return 0.0;
    }

    std::size_t leader = columns.front();

    for (const std::size_t column : columns)
    {
        if (measurements[column].nsPerArgument < measurements[leader].nsPerArgument)
        {
            leader = column;
        }
    }

    double widest = std::max(0.0, measurements[leader].spread - 1.0);

    for (const std::size_t column : columns)
    {
        if (column == leader)
        {
            continue;
        }

        const PairedOutcome pair = CompareToLeader(rounds, leader, column);

        if (pair.lo > 0.0)
        {
            widest = std::max(widest, pair.hi / pair.lo - 1.0);
        }

        widest = std::max(widest, measurements[column].spread - 1.0);
    }

    return widest;
}

/// The entry to name when the measurement could not order a shape, and the static
/// reading of the library's device entry book it was chosen from.
struct StaticFallback {
    /// The entry's name, empty when the shape's rows hold nothing the rule can
    /// rank.
    std::string name;

    /// Why that entry, in the library's own numbers, and what the rule does not
    /// compare.
    std::string basis;
};

/// The degrees a row's lane is cut to, read from the library's own degree tables.
///
/// Which table region A is cut from is a property of the lane and is stated by the
/// tables themselves: the double piece table serves kF64Single, kF64Batch,
/// kF32Batch and kF16Batch — a float entry's region-A seed is computed in double
/// whatever precision it returns — and the float piece table serves kF32Single and
/// kF16Single. Region B is the lane's own seed degree, read by the single lanes per
/// order and by the batch lanes at the order-0 entry.
///
/// Read from the host-side tables the device image is built out of
/// (boys_coefficients.hpp), and not from the handle the device entries read:
/// that handle holds addresses in the device's own memory, which a host reader
/// of this report must not follow. The values are the ones the upload copies
/// into those addresses, so the count here and the degree there are one fact.
struct LaneDegrees {
    /// Dependent multiply-adds region A's piece evaluation is cut to at the top
    /// order the shape reaches: the stored degrees of that order's pieces, summed,
    /// which is what the table states for it and no choice of piece.
    int regionA = 0;

    /// The region-B seed's degree.
    int regionB = 0;

    /// Stored coefficients the two together read, at that order.
    int stored = 0;

    /// Whether these came from the double lane's tables, so the basis says which.
    bool doubles = true;
};

LaneDegrees LaneDegreesOf(BoysDeviceLane lane, int topOrder) {
    LaneDegrees degrees;

    const bool floatTable =
        lane == BoysDeviceLane::kF32Single || lane == BoysDeviceLane::kF16Single;
    const bool floatSeed = lane == BoysDeviceLane::kF32Batch || lane == BoysDeviceLane::kF16Batch ||
                           floatTable;

    degrees.doubles = !floatTable;

    if (topOrder < 0 || topOrder > detail::kMaxOrder)
    {
        return degrees;
    }

    if (floatTable)
    {
        const auto& pieces = detail::f32::kPieces;
        const auto& pieceStart = detail::f32::kPieceStart;

        for (int piece = pieceStart[topOrder]; piece < pieceStart[topOrder + 1]; ++piece)
        {
            degrees.regionA += pieces[static_cast<std::size_t>(piece)].deg;
            degrees.stored += pieces[static_cast<std::size_t>(piece)].deg;
        }

        degrees.regionB = detail::f32::kBDeg;
        degrees.stored += detail::f32::kBDeg;
        return degrees;
    }

    const auto& pieces = detail::kPieces;
    const auto& pieceStart = detail::kPieceStart;

    for (int piece = pieceStart[topOrder]; piece < pieceStart[topOrder + 1]; ++piece)
    {
        degrees.regionA += pieces[static_cast<std::size_t>(piece)].deg;
        degrees.stored += pieces[static_cast<std::size_t>(piece)].deg;
    }

    degrees.regionB = floatSeed ? detail::f32::kBDeg : detail::kBDeg;
    degrees.stored += degrees.regionB;
    return degrees;
}

/// Chooses a shape's fallback by counting what the library's entry book and its
/// degree tables say, never by timing anything.
///
/// **This is a heuristic and it is reported as one.** The rule, whole: over the
/// rows this run carries in one shape — one precision and one question, so the
/// candidates answered the same question and produced the same amount of output
/// for the same arguments — the row whose route evaluates the fewer dependent
/// multiply-adds for one argument is named, counted as the region-A piece degrees
/// of the shape's top order summed plus the lane's region-B seed degree, both read
/// from the degree tables the library itself publishes. A tie is broken by the
/// stored coefficients those two read, then by the number of times the entry is
/// invoked per argument — the library's own shape says once for \c single,
/// \c all-orders and \c all-n, and once per order for \c each-order — then by the
/// launches of this library's own kernel the route needs, which is one for a
/// launched row and none for a device-callable one, then by the library's own
/// default region-B exponential, and last by the library's own enumeration order,
/// so a tie-break is a documented choice rather than an accident of enumeration.
///
/// What it cannot do is as much a part of the rule as what it does: it does not
/// compare the two routes' arithmetic, so it cannot say whether the calling kernel
/// can keep the ladder in registers, which is the whole question between a
/// launched row and a device-callable one; it says nothing about any card's
/// arithmetic; and it reads the whole-argument-range bound each row documents
/// rather than measuring what the row delivers.
///
/// \param entries  the rows this run carries, as the library reports them
/// \param device   the card the run measured, named in the basis
/// \param topOrder the shape's top order, which is the run's own highest
/// \param precision one class's precision
/// \param question  one shape's question
///
/// \returns the fallback, empty when the shape holds nothing to rank
StaticFallback StaticFallbackFor(const std::vector<EntryInfo>& entries,
                                 const std::string& device,
                                 int topOrder,
                                 const std::string& precision,
                                 ProbeQuestion question) {
    StaticFallback fallback;

    /// What the rule counts about one candidate, in the order the rule compares
    /// them: the arithmetic, the table it brings into cache, the invocations, the
    /// launches, the library's own default axis and, last, the library's own order.
    struct Counted {
        const EntryInfo* info = nullptr;
        LaneDegrees degrees;
        int calls = 0;
        int launches = 0;
        int axes = 0;
    };

    const auto counted = [topOrder](const EntryInfo& info) {
        Counted out;
        out.info = &info;
        out.degrees = LaneDegreesOf(info.lane, topOrder);
        // The library's own shape says how many times an entry is invoked for one
        // argument: once for single, all-orders and all-n, once per order for
        // each-order. And its own group says whether this library launches its
        // kernel at all.
        out.calls = info.shape == std::string("each-order") ? 2 : 1;
        out.launches = info.inKernel ? 0 : 1;
        // The library's own default region-B exponential, which is the member its
        // default names; the other member of that axis is the exception.
        out.axes = info.regionBExp == kDefaultRegionBExp ? 0 : 1;
        return out;
    };

    const auto lower = [](const Counted& candidate, const Counted& best) {
        const int candidateArithmetic = candidate.degrees.regionA + candidate.degrees.regionB;
        const int bestArithmetic = best.degrees.regionA + best.degrees.regionB;

        if (candidateArithmetic != bestArithmetic)
        {
            return candidateArithmetic < bestArithmetic;
        }

        if (candidate.degrees.stored != best.degrees.stored)
        {
            return candidate.degrees.stored < best.degrees.stored;
        }

        if (candidate.calls != best.calls)
        {
            return candidate.calls < best.calls;
        }

        if (candidate.launches != best.launches)
        {
            return candidate.launches < best.launches;
        }

        if (candidate.axes != best.axes)
        {
            return candidate.axes < best.axes;
        }

        // The library's own enumeration order last, so the tie-break is the book's
        // order rather than this loop's.
        return static_cast<int>(candidate.info->entry) < static_cast<int>(best.info->entry);
    };

    Counted best;
    const EntryInfo* second = nullptr;

    for (const EntryInfo& info : entries)
    {
        if (info.precision != precision || info.question != question)
        {
            continue;
        }

        const Counted candidate = counted(info);

        if (best.info == nullptr)
        {
            best = candidate;
            continue;
        }

        if (lower(candidate, best))
        {
            second = best.info;
            best = candidate;
        } else if (second == nullptr)
        {
            second = &info;
        }
    }

    if (best.info == nullptr)
    {
        return fallback;
    }

    const std::string runnerUp =
        second != nullptr ? Text(", the next being '%s', whose route is counted at %d dependent "
                                 "multiply-adds for one argument",
                                 second->name,
                                 LaneDegreesOf(second->lane, topOrder).regionA +
                                     LaneDegreesOf(second->lane, topOrder).regionB)
                          : std::string();

    fallback.name = best.info->name;
    fallback.basis = Text(
        "counted, not timed: '%s' reads the %s lane's piece table, whose pieces at order %d are cut "
        "to %d dependent multiply-adds with a region-B seed at degree %d — %d together — storing %d "
        "coefficients%s. The route reaches it as a %s row, invoking the entry %s per argument and "
        "launching %s of this library's own kernel%s. The rule takes the fewest counted "
        "multiply-adds first, the fewest stored coefficients second, the fewest invocations per "
        "argument third, the fewest launches fourth, the library's own default region-B "
        "exponential fifth and the library's own enumeration order last, over this run's own rows "
        "of one precision and one question. It does not compare what the two routes' arithmetic "
        "does once it is running — whether a caller's kernel keeps the ladder in registers is the "
        "whole question between a launched row and a device-callable one and is not counted here — "
        "it says nothing about what any card does with these tables, and the %.4g beside this row "
        "is the bound its documentation states over the whole argument range rather than anything "
        "measured. A name below this line is a property of the tables and of the device %s answered "
        "on, and is not evidence that the entry is fast on it.",
        best.info->name,
        best.degrees.doubles ? "double" : "float",
        topOrder,
        best.degrees.regionA,
        best.degrees.regionB,
        best.degrees.regionA + best.degrees.regionB,
        best.degrees.stored,
        runnerUp.c_str(),
        best.info->inKernel ? "device-callable" : "launched",
        best.calls == 1 ? "once" : "once per order",
        best.launches == 1 ? "one" : "none",
        best.launches == 1 ? "" : ", so the calling kernel is the whole of its cost",
        best.info->bound,
        device.c_str());

    return fallback;
}

/// The card-specific caveat, built from what the runtime reports about the card
/// that answered rather than from what was known about the card this was
/// written on.
std::string BuildCaveat(const DeviceProbeDevice& device) {
    std::string text = Text("This ranking is about this card: %s, compute capability %d.%d. ",
                            device.name.c_str(),
                            device.computeMajor,
                            device.computeMinor);

    if (device.singleToDoublePrecisionPerfRatio >= 2)
    {
        text += Text("It documents a ratio of single- to double-precision throughput of %d:1, so "
                     "how far the fp64 lane sits behind the fp32 lane here is this card's ratio "
                     "and not the library's -- a part with a ratio of 2 orders those two "
                     "differently, and a figure taken here would be the wrong one there. ",
                     device.singleToDoublePrecisionPerfRatio);
    } else if (device.singleToDoublePrecisionPerfRatio == 1)
    {
        text += Text("It reports equal single- and double-precision throughput, so the fp64 and "
                     "fp32 lanes here are separated by their arithmetic rather than by a "
                     "throughput ratio. ");
    }

    if (device.computeMajor < 8)
    {
        text += Text("Its compute capability predates the bf16 tensor instructions, so nothing "
                     "here runs on a tensor path and none of these figures is a tensor figure. ");
    }

    text += Text("A different card, a different toolkit or a different set of target "
                 "architectures can rank these entries differently, which is why the probe is run "
                 "where the numbers are used rather than read from documentation.");
    return text;
}

// ---------------------------------------------------------------------------
// The measurement.
// ---------------------------------------------------------------------------

} // namespace

DeviceProbeReport RunDeviceOptionProbe(const DeviceProbeOptions& options) {
    DeviceProbeReport report;
    report.options = options;

    int deviceCount = 0;

    if (BoysCudaProbeDeviceCount(&deviceCount) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the CUDA runtime could not report how many devices this machine has";
        return report;
    }

    if (deviceCount <= 0)
    {
        report.status = DeviceProbeStatus::kNoDevice;
        report.failure = "this machine has no CUDA device, so there is no device option to "
                         "measure";
        return report;
    }

    if (options.device < 0 || options.device >= deviceCount)
    {
        report.status = DeviceProbeStatus::kDeviceNotFound;
        report.failure = Text("device %d was asked for and this machine has %d device(s), numbered "
                              "0..%d",
                              options.device,
                              deviceCount,
                              deviceCount - 1);
        return report;
    }

    // The caller's own device is where they left it, whatever happens below.
    const DeviceGuard guard;

    if (BoysCudaProbeSetDevice(options.device) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the CUDA runtime could not make device %d current", options.device);
        return report;
    }

    probe_detail::ProbeDeviceFacts facts{};

    if (BoysCudaProbeDeviceQuery(options.device, &facts) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("device %d could not be queried", options.device);
        return report;
    }

    probe_detail::ProbeBuildFacts build{};

    if (BoysCudaProbeBuildFacts(&build) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the toolkit and architecture this library was built with could not be "
                         "read";
        return report;
    }

    report.device.ordinal = options.device;
    report.device.name = facts.name;
    report.device.computeMajor = facts.computeMajor;
    report.device.computeMinor = facts.computeMinor;
    report.device.totalMemoryBytes = static_cast<std::size_t>(facts.totalMemoryBytes);
    report.device.multiProcessorCount = facts.multiProcessorCount;
    report.device.clockKHz = facts.clockKHz;
    report.device.memoryClockKHz = facts.memoryClockKHz;
    report.device.singleToDoublePrecisionPerfRatio = facts.singleToDoublePrecisionPerfRatio;
    report.device.driverVersion = facts.driverVersion;
    report.device.runtimeVersion = facts.runtimeVersion;
    report.device.toolkitVersion = build.toolkit;
    report.device.architectures = build.architectures;
    report.caveat = BuildCaveat(report.device);

    // The lane's own table upload, for the device now current. Idempotent per
    // device, so a caller who had already set this device up pays nothing and
    // sees no change.
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the coefficient tables could not be uploaded to device %d",
                              options.device);
        return report;
    }

    BoysDeviceTables handle{};

    if (BoysCuda::DeviceTables(&handle) != BoysStatus::kSuccess)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the device-table handle for device %d could not be filled",
                              options.device);
        return report;
    }

    // Whether this build carries the fp16 rows, which is a fact about the
    // build and not about the device: the handle above names the float lane's
    // tables, and those are resident whatever the fp16 seam is set to, so a
    // handle field cannot answer this. The entries are declared behind the
    // seam, so the seam is what is read.
#if BoysFp16
    const bool fp16 = true;
#else
    const bool fp16 = false;
#endif

    // --- The workload, and the buffers it lives in ---------------------------
    DeviceProbeOptions clamped = options;
    clamped.count = std::max<std::size_t>(options.count, 1u);
    clamped.nmax = std::min(std::max(options.nmax, 1), kMaxBoysOrder);
    clamped.passes = std::max(options.passes, 1);
    clamped.rounds = std::max(options.rounds, 1);
    clamped.repetitions = std::max(options.repetitions, 1);
    clamped.controlRepetitionsLow = std::max(options.controlRepetitionsLow, 1);
    clamped.controlRepetitionsHigh = std::max(options.controlRepetitionsHigh, 1);

    if (!(options.xHi > options.xLo) || !(options.xLo > 0.0))
    {
        clamped.xLo = 1e-3;
        clamped.xHi = 40.0;
    }

    report.options = clamped;

    const Workload work = BuildWorkload(clamped);
    report.workloadCount = clamped.count;

    // The rows are the library's option space, projected: an option this build
    // does not serve is in report.unoffered with the library's reason, and one
    // it serves has a row here whether or not this probe has ever measured it.
    std::vector<EntryInfo> entries = EnumerateEntries(report.unoffered, report.refusedBecause);

    if (!clamped.only.empty())
    {
        std::vector<EntryInfo> narrowed;

        for (const EntryInfo& info : entries)
        {
            bool asked = false;

            for (const std::string& name : clamped.only)
            {
                if (name == info.name)
                {
                    asked = true;
                }
            }

            if (asked)
            {
                narrowed.push_back(info);
            }
        }

        for (const std::string& name : clamped.only)
        {
            bool known = false;

            for (const EntryInfo& info : entries)
            {
                if (name == info.name)
                {
                    known = true;
                }
            }

            for (const std::string& unbuilt : report.unoffered)
            {
                if (name == unbuilt)
                {
                    known = true;
                }
            }

            if (!known)
            {
                report.notAnEntry.push_back(name);
            }
        }

        if (narrowed.empty())
        {
            report.status = DeviceProbeStatus::kInvalidArgument;
            report.failure = "every name asked for is either no entry of this library or an entry "
                             "this build does not carry, so nothing was measured";
            return report;
        }

        entries.swap(narrowed);
    }

    const std::size_t ladderValues = clamped.count * (kMaxBoysOrder + 1);

    DeviceBuffer orders;
    DeviceBuffer args64;
    DeviceBuffer args32;
    DeviceBuffer args16;
    DeviceBuffer output;
    DeviceBuffer canarySink;

    if (BoysCudaProbeAlloc(&orders.pointer, clamped.count * sizeof(int)) != 0 ||
        BoysCudaProbeAlloc(&args64.pointer, clamped.count * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&args32.pointer, clamped.count * sizeof(float)) != 0 ||
        BoysCudaProbeAlloc(&args16.pointer, clamped.count * sizeof(std::uint16_t)) != 0 ||
        BoysCudaProbeAlloc(&output.pointer, ladderValues * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&canarySink.pointer, sizeof(unsigned long long)) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the workload's buffers could not be allocated on device %d",
                              options.device);
        return report;
    }

    if (BoysCudaProbeUpload(orders.pointer, work.n.data(), clamped.count * sizeof(int)) != 0 ||
        BoysCudaProbeUpload(args64.pointer, work.x.data(), clamped.count * sizeof(double)) != 0 ||
        BoysCudaProbeUpload(args32.pointer, work.xf.data(), clamped.count * sizeof(float)) != 0 ||
        BoysCudaProbeUpload(
            args16.pointer, work.xh.data(), clamped.count * sizeof(std::uint16_t)) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the workload could not be uploaded to device %d", options.device);
        return report;
    }

    // Every figure's request differs only in what it names; the buffers are the
    // same resident ones for the whole run, so nothing is transferred while any
    // clock is open.
    ProbeTimeRequest base{};
    base.handle = &handle;
    base.n = static_cast<const int*>(orders.pointer);
    base.x = static_cast<const double*>(args64.pointer);
    base.xf = static_cast<const float*>(args32.pointer);
    base.xh = args16.pointer;
    base.out = output.pointer;
    base.canarySink = static_cast<unsigned long long*>(canarySink.pointer);
    base.count = clamped.count;
    base.nmax = clamped.nmax;

    report.measurements.resize(entries.size());

    for (std::size_t e = 0; e < entries.size(); ++e)
    {
        const EntryInfo& info = entries[e];
        DeviceProbeMeasurement& measurement = report.measurements[e];
        measurement.name = info.name;
        measurement.precision = info.precision;
        measurement.shape = info.shape;
        measurement.question = QuestionName(info.question);
        measurement.route = info.inKernel ? "in-kernel" : "launched";
        measurement.launchedByLibrary = !info.inKernel;
        measurement.documentedBound = info.bound;
    }

    /// The entry every cost column of this run is anchored to: the library's own
    /// first fp64 row, which is the single-order double entry in the book's own
    /// order. It is resolved in this run's own entry list, because a run narrowed
    /// to a set that does not carry it still needs an anchor, and it falls back to
    /// the first fp64 row the run carries and then to its first row.
    ///
    /// The anchor is what makes the costs comparable: every entry's figure is this
    /// entry's own lower-quartile cost scaled by that entry's lower-quartile ratio
    /// to it, so the column is one measurement and a set of ratios taken inside the
    /// rounds rather than a set of independent times taken in whatever clock each
    /// happened to run under.
    const auto ReferenceIndex = [&entries]() {
        for (std::size_t index = 0; index < entries.size(); ++index)
        {
            if (entries[index].name == std::string("single-fp64"))
            {
                return index;
            }
        }

        for (std::size_t index = 0; index < entries.size(); ++index)
        {
            if (entries[index].precision == std::string("fp64"))
            {
                return index;
            }
        }

        return std::size_t{0};
    };

    const std::size_t reference = ReferenceIndex();

    // The pooled round table: one row per round, one column per entry, in cost per
    // argument, plus the same shape for the in-kernel rows' baselines. Every round
    // of every pass is in here; no round is dropped and no pass is excluded.
    std::vector<std::vector<double>> roundCost;
    std::vector<std::vector<double>> roundBaseline;

    // --- Warm-up. The first launch of a kernel pays one-time costs that are the
    // card's and the context's rather than the entry's, so nothing here is timed.
    bool warm = true;

    for (const EntryInfo& info : entries)
    {
        warm = TimeEntry(info, base, 1).ok && warm;
    }

    double canaryWarm = 0.0;
    ProbeTimeRequest canaryBase = base;
    canaryBase.what = static_cast<int>(ProbeWhat::kCanary);
    // One launch per region for the canary, which does its fixed work inside the
    // kernel rather than by being launched many times.
    canaryBase.reps = 1;

    if (TimeRegion(canaryBase, canaryWarm))
    {
        report.canaryFloorMs = canaryWarm;
    } else
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the canary kernel could not be timed, so this run has no instrument to "
                         "judge its own passes with and no figures are reported";
        return report;
    }

    if (!warm)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = "the warm-up launch of at least one entry failed, so the device is not "
                         "running this build's kernels and nothing was timed";
        return report;
    }

    // --- Passes --------------------------------------------------------------
    //
    // Every pass is run and every pass is used. What the canary said beside a pass
    // is reported with it and decides nothing: a fixed work read by a device clock
    // measures the clock as much as the load, and a card whose boost decays widens
    // the spin's own spread while doing nothing else, so discarding on it would
    // discard the measurement rather than the machine.
    //
    // The visit order is shuffled once per round from the run's own seed. A fixed
    // order would put the same entry first in every round, and whatever a position
    // in the round is worth — the first launch touching a table the others then
    // find warm — would be worth the same to that entry every time and would enter
    // its ratio as though it were the entry's own cost. The seed keeps a run
    // reproducible.
    std::mt19937_64 shuffle(clamped.seed);
    std::vector<std::size_t> visit(entries.size());

    for (std::size_t slot = 0; slot < visit.size(); ++slot)
    {
        visit[slot] = slot;
    }

    const int canarySamples = std::max(3, clamped.rounds + 1);

    for (int pass = 0; pass < clamped.passes; ++pass)
    {
        DeviceProbePass record;
        std::vector<double> canaryMs;
        std::vector<std::vector<double>> passRounds;

        const auto takeCanary = [&]() {
            double ms = 0.0;

            if (TimeRegion(canaryBase, ms))
            {
                canaryMs.push_back(ms);
                report.canaryFloorMs = report.canaryFloorMs > 0.0
                                           ? std::min(report.canaryFloorMs, ms)
                                           : ms;
            }
        };

        takeCanary();

        const auto start = std::chrono::steady_clock::now();

        for (int round = 0; round < clamped.rounds; ++round)
        {
            std::shuffle(visit.begin(), visit.end(), shuffle);

            // One row of this pass's own round table: one cell per entry, in cost
            // per argument, timed in this round and in no other. A cell left
            // infinite is a round in which the entry could not be timed, and it is
            // left out of the ratios below rather than counted as a fast one.
            std::vector<double> row(entries.size(),
                                    std::numeric_limits<double>::infinity());
            std::vector<double> baseline(row);

            // Which half of an in-kernel pair goes first alternates by round, so
            // the fixed cost of a timed region — the event pair, the first launch's
            // cold start, the host's submission — lands on each half in half the
            // rounds instead of always on one.
            const bool baselineFirst = (round % 2) == 1;

            for (const std::size_t index : visit)
            {
                const TimedEntry timed =
                    TimeEntry(entries[index], base, clamped.repetitions, baselineFirst);

                if (!timed.ok)
                {
                    continue;
                }

                const double with =
                    PerArgument(timed.withMs, clamped.repetitions, clamped.count);

                row[index] = with;

                if (!timed.subtracted)
                {
                    continue;
                }

                // Both halves were timed in this round and the difference is
                // formed here, inside the round the two were timed in: whatever the
                // card's clock did between rounds is in both halves of this pair
                // and cancels, and this cell is not a cost taken under a clock of
                // its own. A subtraction that comes out at or below zero is the
                // entry's arithmetic inside the noise of its own baseline: the row
                // is a cost and cannot be below zero, and the report says separately
                // that the subtraction did not resolve.
                const double without =
                    PerArgument(timed.withoutMs, clamped.repetitions, clamped.count);

                baseline[index] = without;
                row[index] = entries[index].inKernel ? std::max(0.0, with - without) : with;
            }

            passRounds.push_back(row);
            roundCost.push_back(std::move(row));
            roundBaseline.push_back(std::move(baseline));

            takeCanary();
        }

        const auto finish = std::chrono::steady_clock::now();
        record.seconds = std::chrono::duration<double>(finish - start).count();

        while (static_cast<int>(canaryMs.size()) < canarySamples && !canaryMs.empty())
        {
            takeCanary();
        }

        // The canary is read here and nowhere else decides anything on it. A pass
        // in which no read succeeded took no reading at all, and that is recorded
        // as such: its figures are zero because nothing was read, and a zero
        // spread would otherwise be a machine reported as still by a reading that
        // was never taken.
        record.canaryMeasured = !canaryMs.empty();

        if (record.canaryMeasured)
        {
            record.canaryMedianMs = MedianOf(canaryMs);
            record.canarySpread = SpreadPercent(canaryMs);
            record.canaryWide = record.canarySpread > clamped.canarySpreadAlarm;

            if (record.canaryWide)
            {
                ++report.passesAboveAlarm;
            } else
            {
                ++report.passesWithinAlarm;
            }

            report.canarySpread = std::max(report.canarySpread, record.canarySpread);
        } else
        {
            ++report.passesWithoutCanary;
        }

        // The pass's own paired spread: the within-round ratios of this pass, which
        // is the spread the ordering is made in. Reported beside the canary's, and
        // it gates nothing either.
        record.pairedSpread = PassPairedSpread(passRounds, reference);
        report.passes.push_back(record);
    }

    report.pairedRounds = static_cast<int>(roundCost.size());

    // --- Fold the rounds into figures ---------------------------------------
    //
    // Every figure below is an aggregate of ratios taken inside a round of the
    // table above, which is what makes it a paired comparison: two cells of one
    // round were timed under one clock. The statistic is a lower quartile —
    // see QuantileOf and kStatisticQuantile for why not the minimum, and why not a
    // mean — and the cost is the reference entry's own lower-quartile figure scaled
    // by this entry's lower-quartile ratio to it, so every cost column of the
    // report is anchored to one entry's own measurement.
    if (!roundCost.empty() && reference < entries.size())
    {
        report.referenceEntry = entries[reference].name;

        std::vector<double> referenceCost;

        for (const std::vector<double>& row : roundCost)
        {
            referenceCost.push_back(row[reference]);
        }

        report.referenceNsPerArgument = QuantileOf(referenceCost, kStatisticQuantile);

        const std::size_t half = roundCost.size() / 2;

        for (std::size_t e = 0; e < entries.size(); ++e)
        {
            DeviceProbeMeasurement& measurement = report.measurements[e];
            std::vector<double> ratios;
            std::vector<double> early;
            std::vector<double> late;
            std::vector<double> baselines;
            double peak = std::numeric_limits<double>::infinity();

            for (std::size_t round = 0; round < roundCost.size(); ++round)
            {
                const double anchor = roundCost[round][reference];
                const double cost = roundCost[round][e];

                if (!std::isfinite(anchor) || anchor <= 0.0 || !std::isfinite(cost))
                {
                    continue;
                }

                ratios.push_back(cost / anchor);
                // The peak column is the entry's own fastest single round, and a
                // round whose cell is zero has no cost in it: for a subtracted row
                // that cell is the floor a difference which did not clear its own
                // baseline was held at, so taking it as the peak would print the
                // instrument's floor as the entry's best round. The peak is the
                // fastest round that produced a figure, which for a launched row is
                // every round of the run. Left infinite when no round did, and the
                // table prints a dash for that.
                if (cost > 0.0)
                {
                    peak = std::min(peak, cost);
                }
                if (round < half)
                {
                    early.push_back(cost / anchor);
                } else
                {
                    late.push_back(cost / anchor);
                }

                if (std::isfinite(roundBaseline[round][e]))
                {
                    baselines.push_back(roundBaseline[round][e]);
                }
            }

            if (ratios.size() < 2 || report.referenceNsPerArgument <= 0.0)
            {
                continue;
            }

            measurement.measured = true;
            measurement.rounds = static_cast<int>(roundCost.size());
            measurement.ratioToReference = QuantileOf(ratios, kStatisticQuantile);
            measurement.ratioLo = measurement.ratioToReference;
            measurement.ratioHi = QuantileOf(ratios, 1.0 - kStatisticQuantile);
            measurement.nsPerArgument =
                report.referenceNsPerArgument * measurement.ratioToReference;
            measurement.nsPerArgumentMax = report.referenceNsPerArgument * measurement.ratioHi;
            measurement.nsPerArgumentPeak = std::isfinite(peak) ? peak : 0.0;
            measurement.nsPerArgumentBaseline = QuantileOf(baselines, kStatisticQuantile);
            measurement.spread =
                measurement.ratioLo > 0.0 ? measurement.ratioHi / measurement.ratioLo : 1.0;
            // A launched row is not a subtraction and always carries its figure. An
            // in-kernel row carries one only when the difference it was reduced to
            // stands above its own baseline in the middle half of the run; a row
            // whose figure is the zero the max() above floors it at is the row's
            // arithmetic sitting inside its kernel's traffic at this workload.
            measurement.subtractionResolved =
                !entries[e].inKernel || measurement.nsPerArgument > 0.0;

            // How far the entry's ratio to the reference moved between the run's
            // halves. Zero means the entry and the reference kept pace as the clock
            // moved, which is what a ratio that is genuinely the entry's own cost
            // looks like; a value away from zero says the two do not carry the clock
            // alike, and the report says so where it prints this entry.
            const double firstHalf = MedianOf(early);
            const double secondHalf = MedianOf(late);

            if (firstHalf > 0.0)
            {
                measurement.ratioDrift = secondHalf / firstHalf - 1.0;
            }
        }
    }

    // --- What the timed work actually produced. One read-back per entry,
    // outside every clock, so the report can show that the launches ran and ran
    // on the intended arguments.
    {
        std::vector<double> host(ladderValues, 0.0);

        for (std::size_t e = 0; e < entries.size(); ++e)
        {
            const EntryInfo& info = entries[e];

            // One more launch per entry, and its result read back. It is not
            // timed: what it establishes is that the timed launches produced
            // values, on the arguments the workload holds.
            if (!TimeEntry(info, base, 1).ok)
            {
                continue;
            }

            const std::size_t bytes = info.precision == std::string("fp64")
                                          ? ladderValues * sizeof(double)
                                          : info.precision == std::string("fp32")
                                                ? ladderValues * sizeof(float)
                                                : ladderValues * sizeof(std::uint16_t);

            if (BoysCudaProbeDownload(host.data(), output.pointer, bytes) != 0)
            {
                continue;
            }

            double sum = 0.0;

            if (info.precision == std::string("fp64"))
            {
                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += host[i];
                }
            } else if (info.precision == std::string("fp32"))
            {
                const auto* values = reinterpret_cast<const float*>(host.data());

                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += static_cast<double>(values[i]);
                }
            } else
            {
                const auto* bits = reinterpret_cast<const std::uint16_t*>(host.data());

                for (std::size_t i = 0; i < ladderValues; ++i)
                {
                    sum += static_cast<double>(static_cast<float>(F16::FromBits(bits[i])));
                }
            }

            report.measurements[e].checkedSum = sum;
        }

        (void)BoysCudaProbeSynchronize();
    }

    // --- The classes, one per precision --------------------------------------
    //
    // After the controls, because a row the controls set aside is not ranked, and
    // run to a fixed point: every row a class would recommend is put through the
    // repetition control first, and if it does not agree the class falls to the
    // next row, which is checked in its turn.
    //
    // A class is a precision and holds one ranking per question shape. The class
    // set comes from the table of entries this run was asked for rather than from
    // the rows that happened to measure, so a shape whose every row failed to
    // measure is still reported as a shape of the class with nothing in it, and a
    // class the caller did not ask about is not invented.
    const auto ConcludeClasses = [&]() {
        report.classes.clear();

        for (const EntryInfo& info : entries)
        {
            std::size_t classAt = report.classes.size();

            for (std::size_t c = 0; c < report.classes.size(); ++c)
            {
                if (report.classes[c].precision == info.precision)
                {
                    classAt = c;
                }
            }

            if (classAt == report.classes.size())
            {
                DeviceProbeClass fresh;
                fresh.precision = info.precision;
                report.classes.push_back(fresh);

                // Whether this class holds one bound or two is a fact about its
                // rows, so it is read off them; the sentence says which, and a
                // class whose rows document one bound between them says so rather
                // than carrying the other sentence as a caution it does not need.
                bool oneBound = true;
                double first = 0.0;
                bool haveFirst = false;

                for (const EntryInfo& other : entries)
                {
                    if (other.precision != info.precision)
                    {
                        continue;
                    }

                    if (!haveFirst)
                    {
                        first = other.bound;
                        haveFirst = true;
                    } else if (other.bound != first)
                    {
                        oneBound = false;
                    }
                }

                report.classes[classAt].note = PrecisionNote(info.precision, oneBound);
            }

            const std::string question = QuestionName(info.question);
            bool haveShape = false;

            for (const DeviceProbeRanking& ranking : report.classes[classAt].rankings)
            {
                if (ranking.question == question)
                {
                    haveShape = true;
                }
            }

            if (haveShape)
            {
                continue;
            }

            DeviceProbeRanking ranking;
            ranking.question = question;
            ranking.asked = QuestionAsked(info.question);

            std::vector<DeviceProbeMeasurement*> live;
            std::vector<std::size_t> columns;

            for (std::size_t index = 0; index < report.measurements.size(); ++index)
            {
                DeviceProbeMeasurement& measurement = report.measurements[index];

                if (measurement.measured && measurement.precision == info.precision &&
                    measurement.question == question)
                {
                    live.push_back(&measurement);
                    columns.push_back(index);
                }
            }

            Conclude(ranking, live, columns, roundCost, report.pairedRounds);

            // A refusal is still a report a consumer can act on, so it carries a
            // default read out of the library's own entry book for this shape and
            // precision. It is derived from the tables — which regions a route
            // runs, whether the entry is launched or called inside the caller's
            // kernel, the repetition counts, the degrees the tables state — and
            // from no timing at all, so it is labelled as a heuristic wherever it
            // is printed and it never appears beside a measured figure. A shape
            // the run did order carries none: it has a measurement instead.
            if (ranking.verdict == DeviceProbeVerdict::kCannotDetermine)
            {
                const StaticFallback fallback = StaticFallbackFor(entries,
                                                                  report.device.name,
                                                                  clamped.nmax,
                                                                  info.precision,
                                                                  info.question);

                ranking.heuristicEntry = fallback.name;
                ranking.heuristicBasis = fallback.basis;
                report.hasDefault = report.hasDefault || !fallback.name.empty();
            }

            report.classes[classAt].rankings.push_back(ranking);
        }
    };

    // A row is judged against what this run's own instrument and that row's own
    // passes measured, never against the widest threshold some other class
    // needed: a class whose rival wandered is a fact about that rival.
    const auto RowResolution = [&](const DeviceProbeMeasurement* row) {
        return ShapeResolution(report.measurements, roundCost, row->precision, row->question);
    };

    const auto FindEntry = [&](const std::string& name) -> const EntryInfo* {
        for (const EntryInfo& info : entries)
        {
            if (info.name == name)
            {
                return &info;
            }
        }

        return nullptr;
    };

    /// Puts one row through the repetition control and records the outcome on the
    /// row, so that the class conclusions can see it.
    const auto CheckRow = [&](DeviceProbeMeasurement& row, DeviceProbeRepetitionControl* printed) {
        const EntryInfo* info = FindEntry(row.name);

        if (info == nullptr)
        {
            return false;
        }

        DeviceProbeRepetitionControl outcome;
        const bool timed = RunRepetitionControl(*info,
                                                row,
                                                base,
                                                clamped,
                                                RowResolution(&row),
                                                outcome);

        row.repetitionChecked = true;
        row.repetitionAgrees = timed && outcome.agrees;
        row.repetitionNote = outcome.note;

        if (printed != nullptr)
        {
            *printed = outcome;
        }

        return true;
    };

    // --- The controls --------------------------------------------------------
    //
    // One check per route, both under the protocol the figures were taken under.
    // The check is that one entry timed at two very different repetition counts
    // gives the same per-argument figure: a cost that repeats with the launch
    // rather than with the call is amortised over a different number of launches
    // at each count, so a timer that had absorbed one would not agree with
    // itself. The launched route is judged against the device-side floor as well
    // — a kernel launched the same way that does no Boys arithmetic at all —
    // because that route's figure has a launch inside it by construction.
    {
        DeviceProbeMeasurement* fastestLaunched = nullptr;
        DeviceProbeMeasurement* fastestSubtracted = nullptr;

        for (DeviceProbeMeasurement& measurement : report.measurements)
        {
            if (!measurement.measured)
            {
                continue;
            }

            if (measurement.launchedByLibrary)
            {
                if (fastestLaunched == nullptr ||
                    measurement.nsPerArgument < fastestLaunched->nsPerArgument)
                {
                    fastestLaunched = &measurement;
                }
            } else if (measurement.subtractionResolved)
            {
                // Only a row whose subtraction resolved a cost above its own
                // baseline has a figure for the control to check.
                if (fastestSubtracted == nullptr ||
                    measurement.nsPerArgument < fastestSubtracted->nsPerArgument)
                {
                    fastestSubtracted = &measurement;
                }
            }
        }

        if (fastestLaunched != nullptr &&
            CheckRow(*fastestLaunched, &report.control))
        {
            // The floor, taken once for the route that has a launch inside it.
            ProbeTimeRequest floorRequest = base;
            floorRequest.what = static_cast<int>(ProbeWhat::kFloor);
            floorRequest.reps = clamped.controlRepetitionsHigh;

            double floorMs = 0.0;

            if (report.control.entry.empty())
            {
                // RunRepetitionControl already said why in the note.
            } else if (!TimeRegion(floorRequest, floorMs))
            {
                report.control.note += "; the device-side floor could not be timed on this run, so "
                                       "how much of that figure is launch rather than arithmetic "
                                       "is not established here";
            } else
            {
                report.control.nsPerLaunchFloor =
                    floorMs * 1.0e6 / static_cast<double>(clamped.controlRepetitionsHigh);

                const double perCall = clamped.count * fastestLaunched->nsPerArgument;
                const double share =
                    perCall > 0.0 ? report.control.nsPerLaunchFloor / perCall : 0.0;

                // What the floor is, stated as what it measured rather than as
                // what it was meant to isolate. It is the per-launch cost of a
                // kernel that does no arithmetic, so it contains the device's own
                // launch and the host's submission of it together; on a platform
                // whose submission is expensive that is the larger part.
                //
                // When it is at or above a row's own per-call cost, the row is
                // the floor rather than the arithmetic, and a reader who took it
                // for an evaluation cost would be reading the launcher.
                const std::string floorClause =
                    share >= 1.0
                        ? Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, which is %.0f%% of that "
                               "row's own cost per call. THAT ROW IS THE FLOOR: at this argument "
                               "count a launched entry is not being measured, because %d arguments "
                               "of its arithmetic cost less than one launch does. Raising the "
                               "argument count until the floor is a small fraction of the figure "
                               "is what this needs",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share,
                               static_cast<int>(clamped.count))
                        : Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, %.2f%% of that row's own "
                               "cost per call, and is the part of every launched figure that is "
                               "launch and submission together rather than arithmetic",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share);

                report.control.note += "; ";
                report.control.note += floorClause;
                report.control.note += ".";
            }
        } else
        {
            report.control.note = "no launched entry produced a figure, so there was nothing to "
                                  "put the launched route's repetition control through";
        }

        if (fastestSubtracted != nullptr && CheckRow(*fastestSubtracted, &report.deviceCallControl))
        {
            if (!report.deviceCallControl.entry.empty())
            {
                // Neither bracket of a subtraction holds a launch of this
                // library, so there is no floor to take for this route: what
                // stands in for one is the baseline, reported above, and it is
                // already subtracted out of the figure being checked.
                report.deviceCallControl.note += "; nothing launched by this library is inside "
                                                 "either half of this row, so the floor that "
                                                 "applies to the launched route does not apply "
                                                 "here - the row's own baseline is what it was "
                                                 "subtracted against, and it is reported above.";
            }
        } else
        {
            report.deviceCallControl.note =
                "no in-kernel row of this run resolved a cost above its own baseline, so there was "
                "nothing to put the subtraction route's repetition control through";
        }
    }

    // --- The conclusions, to a fixed point -----------------------------------
    //
    // Every row the report names as a shape's fastest or as its recommendation is
    // put through the repetition control before the report ships, and a row that
    // does not agree is set aside and the shape falls to the next. Looped,
    // because setting a row aside can name a new leader, which then has to be
    // checked in its turn. Bounded by the number of rows: each round checks at
    // least one row not checked before, and the loop leaves when there is none.
    for (;;)
    {
        ConcludeClasses();

        DeviceProbeMeasurement* unchecked = nullptr;

        for (const DeviceProbeClass& clause : report.classes)
        {
            for (const DeviceProbeRanking& ranking : clause.rankings)
            {
                for (DeviceProbeMeasurement& measurement : report.measurements)
                {
                    const bool named = !ranking.recommended.empty()
                                           ? measurement.name == ranking.recommended
                                           : measurement.name == ranking.fastestOverall;

                    if (named && !measurement.repetitionChecked)
                    {
                        unchecked = &measurement;
                    }
                }
            }
        }

        if (unchecked == nullptr)
        {
            break;
        }

        if (!CheckRow(*unchecked, nullptr))
        {
            // No entry of this build is that row, which cannot happen for a row
            // that came out of the measurements; stop rather than loop.
            break;
        }
    }

    report.status = DeviceProbeStatus::kSuccess;
    return report;
}

// ---------------------------------------------------------------------------
// The text.
// ---------------------------------------------------------------------------

/// The library's report of its device option space, row by row, with the row
/// this probe carries for each beside it.
///
/// This block is the probe's coverage statement and not a courtesy: the probe's
/// option table is a projection of the library's, so every row listed here is
/// either a row measured above or a refusal with the library's reason, and a
/// row this probe carries that the library does not report would be a list kept
/// beside the library rather than read from it. `probeRows` is what the run
/// actually carried — the measurement table's own names, in its own order — so
/// the comparison is against the report this run printed and not against a
/// second reading of the same source.
void AppendOptionSpace(std::string& text, const std::vector<std::string>& probeRows) {
    const std::span<const DeviceOptionInfo> space = BoysDeviceOptions();
    std::size_t served = 0;
    std::size_t refused = 0;
    std::size_t launched = 0;
    std::size_t inKernel = 0;

    for (const DeviceOptionInfo& option : space)
    {
        (option.built ? served : refused) += 1;
        (option.group == DeviceOptionGroup::kLaunched ? launched : inKernel) += 1;
    }

    text += "\n\nthe option space - the library's own report of it, and the row this run carries "
            "for each\n";

    if (probeRows.empty())
    {
        text += Text("  (BoysDeviceOptions, include/boys/boys_cuda_options.hpp): %zu option(s) = %zu "
                     "launched + %zu\n  device-callable, %zu this build serves and %zu it "
                     "refuses. No row was measured on\n  this run, so the second column says so "
                     "rather than naming one.\n",
                     space.size(), launched, inKernel, served, refused);
    } else
    {
        text += Text("  (BoysDeviceOptions, include/boys/boys_cuda_options.hpp): %zu option(s) = %zu "
                     "launched + %zu\n  device-callable, %zu this build serves and %zu it "
                     "refuses; %zu row(s) measured here\n",
                     space.size(), launched, inKernel, served, refused, probeRows.size());
    }

    text += "  report row                 probe row                  group     precision  shape    "
            "   question    axis          lane        bound     documented form\n";

    for (const DeviceOptionInfo& option : space)
    {
        std::string carried = "not measured on this run";

        for (const std::string& name : probeRows)
        {
            if (name == option.name)
            {
                carried = name;
            }
        }

        text += Text("  %-26s %-26s %-9s %-10s %-11s %-11s %-13s %-11s %-9.2g %s\n",
                     option.name,
                     carried.c_str(),
                     GroupName(option.group),
                     PrecisionName(option.precision),
                     ShapeName(option.shape),
                     QuestionName(option.question),
                     AxisName(option).c_str(),
                     LaneName(option.lane),
                     option.bound,
                     option.boundForm);
    }

    // The other direction, and the one a hand-written list fails: a row this
    // run carries that the report does not. It is named here rather than passed
    // over, and the count is printed whether or not it is empty.
    std::vector<std::string> unreported;

    for (const std::string& name : probeRows)
    {
        bool reported = false;

        for (const DeviceOptionInfo& option : space)
        {
            reported = reported || option.name == name;
        }

        if (!reported)
        {
            unreported.push_back(name);
        }
    }

    if (unreported.empty())
    {
        text += "  rows this run carries that the library's report does not: none\n";
    } else
    {
        text += "  rows this run carries that the library's report does not:";

        for (const std::string& name : unreported)
        {
            text += Text(" %s", name.c_str());
        }

        text += "\n";
    }

    text += "  a refused row is refused by the library where it is named, and not by this probe: "
            "the\n  option is absent from the table above because no build with this seam serves "
            "it, and the\n  report states the seam. An option added to the library reaches this "
            "table without an edit\n  here; one removed leaves it.\n";
}

std::string FormatDeviceOptionProbe(const DeviceProbeReport& report) {
    std::string text;

    text += "boys device option probe - this result is about this device, this build, this process "
            "and the workload below.\n";

    if (report.status != DeviceProbeStatus::kSuccess)
    {
        if (!report.device.name.empty())
        {
            text += Text("  device: %s (ordinal %d, compute capability %d.%d)\n",
                         report.device.name.c_str(),
                         report.device.ordinal,
                         report.device.computeMajor,
                         report.device.computeMinor);

            if (report.device.singleToDoublePrecisionPerfRatio > 0)
            {
                text += Text("          single:double throughput ratio %d:1\n",
                             report.device.singleToDoublePrecisionPerfRatio);
            }
        }

        text += Text("\n  NOT MEASURED: %s\n", report.failure.c_str());
        text += Text("  status: %s\n", StatusName(report.status));

        for (const std::string& name : report.notAnEntry)
        {
            text += Text("  no such entry: %s\n", name.c_str());
        }

        for (const std::string& name : report.unoffered)
        {
            text += Text("  refused by the library: %s\n", name.c_str());
        }

        // The space is a fact about the option the entry names and not about
        // this host, so it is stated even when no figure could be taken: a
        // reader of a failed run learns which options exist and which this
        // build refuses, and what is missing is the measurement.
        AppendOptionSpace(text, std::vector<std::string>());
        return text;
    }

    text += Text("  device: %s\n", report.device.name.c_str());
    text += Text("    ordinal %d, compute capability %d.%d, %d streaming multiprocessors\n",
                 report.device.ordinal,
                 report.device.computeMajor,
                 report.device.computeMinor,
                 report.device.multiProcessorCount);
    text += Text("    memory %s, graphics clock %.2f GHz, memory clock %.2f GHz\n",
                 FormatBytes(report.device.totalMemoryBytes).c_str(),
                 static_cast<double>(report.device.clockKHz) / 1.0e6,
                 static_cast<double>(report.device.memoryClockKHz) / 1.0e6);
    text += Text("    single:double throughput ratio %d:1\n",
                 report.device.singleToDoublePrecisionPerfRatio);
    text += Text("    driver %s, runtime %s, built with toolkit %s, for architecture(s) %s\n",
                 FormatCudaVersion(report.device.driverVersion).c_str(),
                 FormatCudaVersion(report.device.runtimeVersion).c_str(),
                 report.device.toolkitVersion.c_str(),
                 report.device.architectures.c_str());

    text += "\n  workload:\n";
    text += Text("    %zu arguments, log-uniform over [%.3g, %.3g], each carrying its own "
                 "highest\n",
                 report.workloadCount,
                 report.options.xLo,
                 report.options.xHi);
    text += Text("    order drawn as the sum of two shell angular momenta 0..%d, capped at %d,\n",
                 kShellMax,
                 report.options.nmax);
    text += "    sorted by argument so the uniform-order entries' stated precondition holds.\n";
    text += Text("    the fp32 and fp16 device-callable entries take the argument rounded to "
                 "their own\n    precision, which is what a caller's kernel holds in a register; "
                 "the batch entries\n    of those lanes take the double argument and round it "
                 "themselves.\n");

    text += "\n  protocol:\n";
    text += Text("    %d passes, %d rounds per pass, %d launches per timed region. Every entry "
                 "is timed by\n    a CUDA event pair around those launches, into buffers "
                 "uploaded once before the first\n    clock; the per-argument figure is the "
                 "device time divided by the launches and the\n    arguments.\n",
                 report.options.passes,
                 report.options.rounds,
                 report.options.repetitions);
    text += Text("    every entry is timed once per round and the visit order is shuffled per "
                 "round, so the\n    %d rounds of the run are pooled into one table of %d rows and "
                 "%zu columns, one row\n    per round and one column per entry: a comparison between "
                 "two entries is the ratio\n    of their cells in one row — both were timed under "
                 "whatever clock that round ran at,\n    and a drift common to the round cancels in "
                 "the ratio.\n",
                 report.pairedRounds,
                 report.pairedRounds,
                 report.measurements.size());
    text += "    no round and no pass is dropped. A reported figure is the lower quartile of those "
            "ratios\n    and never the minimum of the run: on a card whose clock decays the "
            "minimum is the\n    earliest and best-clocked round, which is not what a caller's "
            "long workload meets, and\n    the upper quartile is printed beside it so both ends "
            "come from one distribution. The\n    peak column is the entry's own fastest single "
            "round, a raw figure under no anchor and\n    not a bound on the columns beside it.\n";
    text += Text("    the canary beside a pass is a diagnostic and gates nothing: a fixed work read "
                 "by a\n    device clock measures the clock as much as the load, so a pass is "
                 "flagged when the\n    canary's own runs disagreed by more than %.1f%% and is "
                 "used either way.\n",
                 report.options.canarySpreadAlarm);
    text += Text("    in-kernel rows are a subtraction: the caller's kernel with the entry in it, "
                 "less the\n    same kernel with the call removed and the traffic kept, the two "
                 "timed adjacent in\n    the same round, so the difference is formed inside that "
                 "round and survives a drift.\n");

    text += "\n  canary:\n";
    text += Text("    a fixed-work integer kernel, launched between the rounds, holding no "
                 "floating-point\n    state, so the arithmetic this probe ranks cannot change "
                 "the instrument's own cost.\n");

    double widestPaired = 0.0;
    bool anyCanaryMeasured = false;

    for (const DeviceProbePass& pass : report.passes)
    {
        widestPaired = std::max(widestPaired, pass.pairedSpread);
        anyCanaryMeasured = anyCanaryMeasured || pass.canaryMeasured;
    }

    // A canary that did not run is said in words where a spread would be read. A
    // zero spread and an absent reading are different facts about a machine, and
    // printing the second as the first is the report claiming to have looked.
    if (!anyCanaryMeasured)
    {
        text += Text("    no pass %s a canary reading between its rounds, so no spread of the "
                     "instrument's\n    own is reported here: a canary that did not run is not a "
                     "canary that stayed quiet,\n    and the space where its spread would go holds "
                     "those words instead of a zero. The\n    widest within-round paired spread "
                     "across the passes, which is where the ordering\n    below is made, is "
                     "%.2f%%.\n",
                     report.passes.empty() ? "was run to take" : "took",
                     widestPaired);
    } else
    {
        text += Text("    its fastest run seen here is %.4f ms; the widest spread across the passes "
                     "is %.2f%%,\n    beside the widest within-round paired spread of %.2f%%. Both "
                     "are context: the\n    ordering below is made in the paired column, and the "
                     "canary is the same fixed work\n    read by the same clock the entries "
                     "were.\n",
                     report.canaryFloorMs,
                     report.canarySpread,
                     widestPaired);
    }

    text += "\npass  wall_s  canary_ms  canary%  paired%  verdict\n";

    if (report.passes.empty())
    {
        text += "     (no pass was run)\n";
    }

    for (std::size_t i = 0; i < report.passes.size(); ++i)
    {
        const DeviceProbePass& pass = report.passes[i];

        // A pass whose canary was not timed prints no canary figure at all: a dash
        // in both columns and the reason in the verdict column, so the table
        // cannot be read as a reading that was never taken.
        const std::string canaryMs =
            pass.canaryMeasured ? Text("%.4f", pass.canaryMedianMs) : std::string("-");
        const std::string canarySpread =
            pass.canaryMeasured ? Text("%.2f", pass.canarySpread) : std::string("-");
        const char* wideText = pass.canaryMeasured
                                   ? (pass.canaryWide ? "canary wide - reported, used"
                                                      : "canary within alarm")
                                   : "canary not timed - no reading, not a quiet one";

        text += Text("%4zu  %6.3f  %9s  %7s  %7.2f  %s\n",
                     i + 1,
                     pass.seconds,
                     canaryMs.c_str(),
                     canarySpread.c_str(),
                     pass.pairedSpread,
                     wideText);
    }

    text += Text("\n  %d of %d pass(es) ran with the canary's own runs wider than the %.1f%% "
                 "alarm, and every\n  one of them was used: the canary is fixed work read by the "
                 "same clock the entries were,\n  so it cannot separate the two, and discarding "
                 "on it would discard the measurement\n  rather than the machine. What separates "
                 "the entries is the paired column and the\n  refusal below, not this one.\n",
                 report.passesAboveAlarm,
                 report.options.passes,
                 report.options.canarySpreadAlarm);

    if (report.passesWithoutCanary > 0)
    {
        text += Text("  %d of the %d pass(es) took no canary reading at all and are counted in "
                     "neither the\n  within nor the above: a pass the canary did not run in is a "
                     "pass with no reading, and\n  no zero is reported for it.",
                     report.passesWithoutCanary,
                     report.options.passes);
        text += "\n";
    }

    {
        // What the column's own ceiling is, said where a reader meets it: at two
        // rounds per pass the widest a two-reading quartile band can be is three
        // times, so a pass that reads 200.00% is at that ceiling and is not a cap.
        const std::string ceiling =
            report.options.rounds == 2
                ? std::string("A pass holds two rounds and the two ends of a quartile band over "
                              "two\n  readings are three times apart at the extreme, so 200.00% is "
                              "that column's\n  own ceiling: a pass reading it had one entry whose "
                              "two rounds were as far\n  apart as two readings can be.")
                : Text("A pass holds %d rounds, so this column and the run's resolution below\n  are "
                       "the same kind of number taken over different round counts.",
                       report.options.rounds);

        text += Text("\n  The paired column is the widest within-round band any entry of that pass "
                     "showed, so\n  it is the resolution that pass alone can order at.\n  %s\n  The "
                     "run's resolution, below, is the same measurement over all %d pooled "
                     "rounds.\n",
                     ceiling.c_str(),
                     report.pairedRounds);
    }

    text += "\nentries - ns/argument is device time per argument, transfer and host submission "
            "excluded\n";
    text += "  entry                     precision  shape        route      ns/arg      hi      "
            "minus   spread   vs ref band  drift%    peak ns  rounds  bound      method\n";

    for (const DeviceProbeMeasurement& measurement : report.measurements)
    {
        const std::string baseline =
            measurement.nsPerArgumentBaseline > 0.0
                ? Text("%.3f", measurement.nsPerArgumentBaseline)
                : std::string("-");
        const std::string band = measurement.measured
                                     ? Text("%.3f..%.3f", measurement.ratioLo, measurement.ratioHi)
                                     : std::string("-");
        const std::string rounds =
            Text("%2d/%d", measurement.rounds, report.pairedRounds);
        const std::string peak =
            measurement.nsPerArgumentPeak > 0.0
                ? Text("%.3f", measurement.nsPerArgumentPeak)
                : std::string("-");

        if (!measurement.measured)
        {
            text += Text("  %-24s %-10s %-11s %-10s %10s  %-7s  %-7s  %-7s  %-12s  %-7s  %8s  "
                         "%-6s  %8.2g  %s\n",
                         measurement.name.c_str(),
                         measurement.precision.c_str(),
                         measurement.shape.c_str(),
                         measurement.route.c_str(),
                         "-",
                         "-",
                         "-",
                         "-",
                         "-",
                         "-",
                         "-",
                         rounds.c_str(),
                         measurement.documentedBound,
                         measurement.launchedByLibrary ? "library kernel launched"
                                                       : "caller kernel, subtracted");
            continue;
        }

        // An in-kernel row whose subtraction did not clear its own baseline has no
        // figure, and printing the zero it came out at would read as a free call:
        // the class this row belongs to sets it aside, so the table says the same.
        if (!measurement.subtractionResolved)
        {
            text += Text("  %-24s %-10s %-11s %-10s %10s  %-7s  %-7s  %6.2fx  %-12s  %+6.2f%%  "
                         "%8s  %-6s  %8.2g  %s\n",
                         measurement.name.c_str(),
                         measurement.precision.c_str(),
                         measurement.shape.c_str(),
                         measurement.route.c_str(),
                         "unresolved",
                         "-",
                         baseline.c_str(),
                         measurement.spread,
                         band.c_str(),
                         100.0 * measurement.ratioDrift,
                         "-",
                         rounds.c_str(),
                         measurement.documentedBound,
                         "caller kernel, subtracted");
            continue;
        }

        text += Text("  %-24s %-10s %-11s %-10s %10.3f  %8.3f  %-7s  %6.2fx  %-12s  %+6.2f%%  "
                     "%8s  %-6s  %8.2g  %s\n",
                     measurement.name.c_str(),
                     measurement.precision.c_str(),
                     measurement.shape.c_str(),
                     measurement.route.c_str(),
                     measurement.nsPerArgument,
                     measurement.nsPerArgumentMax,
                     baseline.c_str(),
                     measurement.spread,
                     band.c_str(),
                     100.0 * measurement.ratioDrift,
                     peak.c_str(),
                     rounds.c_str(),
                     measurement.documentedBound,
                     measurement.launchedByLibrary ? "library kernel launched"
                                                   : "caller kernel, subtracted");
    }

    text += Text("\n  the ns/arg column is the lower quartile of this entry's %d paired rounds, as "
                 "the\n  reference entry's own lower-quartile figure scaled by this entry's "
                 "lower-quartile ratio\n  to it ('%s', whose own ratio is exactly 1.000 and whose "
                 "drift is exactly 0.00%%); 'hi' is\n  the same at the upper quartile of the same "
                 "rounds and 'spread' is hi over lo, so the two\n  ends of a row come from one "
                 "distribution. The vs-ref band is that ratio over the middle\n  half of the "
                 "rounds, and drift is how far it moved between the run's first and second\n  "
                 "half — a row whose drift is larger than the resolution below is a row whose two\n"
                 "  entries do not carry this card's clock alike. 'peak ns' is the fastest single "
                 "round the\n  entry was ever seen in: a raw figure under no anchor, the entry's own "
                 "floor and not\n  a bound on the columns beside it, which are anchored to the "
                 "reference entry — and a\n  round whose cell was zero is left out of it, because "
                 "such a cell is the floor a\n  subtracted difference was held at rather than a "
                 "cost. 'rounds' is how many of the\n  run's pooled rounds the row rests on, and "
                 "every measured row rests on every one of\n  them — no round and no pass is "
                 "dropped for it.\n",
                 report.pairedRounds,
                 report.referenceEntry.empty() ? "the reference entry" : report.referenceEntry.c_str());
    text += "\n  the minus column is the second half of a subtraction: the same caller-shaped "
            "kernel with\n  the call removed and the traffic kept, timed in the same round as the "
            "half it is\n  subtracted from, so the difference is formed inside that round and a "
            "drift common to\n  the round cancels in it. Both columns are then aggregated over "
            "the same rounds. A\n  launched row has no minus column because nothing was "
            "subtracted from it. The ns/arg\n  column reads unresolved for an in-kernel row whose "
            "difference did not clear its own\n  baseline.\n";

    if (!report.unoffered.empty())
    {
        text += "\noptions of this library's device space that this build does not serve, refused\n"
                "where the library names them:\n";

        for (std::size_t i = 0; i < report.unoffered.size(); ++i)
        {
            const std::string reason = i < report.refusedBecause.size()
                                           ? report.refusedBecause[i]
                                           : std::string("this build serves it and the report "
                                                         "names no reason");
            text += Text("    %s\n      %s\n", report.unoffered[i].c_str(), reason.c_str());
        }
    }

    if (!report.notAnEntry.empty())
    {
        text += "\nnames asked for that are no entry of this library, so nothing was measured "
                "for them:\n";

        for (const std::string& name : report.notAnEntry)
        {
            text += Text("    %s\n", name.c_str());
        }
    }

    text += "\n  the bound column is what each entry's lane documents at full accuracy, read from "
            "the\n  library's own report of the option space and not measured here. What a lane "
            "delivers on this card is the "
            "accuracy\n  gate's business. The bounds in that column are not all the same, and a "
            "row that carries a\n  looser one is a row that bought speed with accuracy: the "
            "rankings below are per precision\n  and per question shape, so no row is ordered "
            "against a row of another precision, and the\n  winner of a shape is the fastest "
            "entry of that shape at the bound its own row states.\n";

    // --- The controls --------------------------------------------------------
    //
    // One per route, since the two routes are two methods and a check of one
    // says nothing about the other. The floor is the launched route's own: a
    // subtraction has no launch of this library's inside either half.
    const auto AppendControl = [&text](const char* heading,
                                       const DeviceProbeRepetitionControl& control,
                                       bool withFloor) {
        text += Text("\n%s\n", heading);

        if (control.note.empty())
        {
            text += "  not run\n";
            return;
        }

        if (control.entry.empty())
        {
            text += Text("  %s\n", control.note.c_str());
            return;
        }

        text += Text("  entry: %s (%s route)\n", control.entry.c_str(), control.route.c_str());
        text += Text("  %d launches: %.3f ns/argument\n",
                     control.repetitionsLow,
                     control.nsPerArgumentLow);
        text += Text("  %d launches: %.3f ns/argument\n",
                     control.repetitionsHigh,
                     control.nsPerArgumentHigh);

        if (control.nsPerArgumentBaselineLow > 0.0 || control.nsPerArgumentBaselineHigh > 0.0)
        {
            text += Text("  the half it was subtracted against: %.3f and %.3f ns/argument\n",
                         control.nsPerArgumentBaselineLow,
                         control.nsPerArgumentBaselineHigh);
        }

        if (std::isfinite(control.difference))
        {
            text += Text("  disagreement %.2f%%: %s the %.2f%% this row can be placed at\n",
                         100.0 * control.difference,
                         control.agrees ? "AGREES within" : "DOES NOT AGREE within",
                         100.0 * control.judgedAgainst);
        } else
        {
            text += Text("  no disagreement: the %d-launch reading resolved no cost, so NOT "
                         "AGREED\n",
                         control.repetitionsHigh);
        }

        if (withFloor)
        {
            text += Text("  floor, a kernel launched the same way with no Boys arithmetic in it: "
                         "%.3f us per launch\n",
                         control.nsPerLaunchFloor / 1000.0);
        }

        text += Text("  %s\n", control.note.c_str());
    };

    AppendControl("launch control", report.control, true);
    AppendControl("subtraction control", report.deviceCallControl, false);

    // --- The rankings --------------------------------------------------------
    //
    // One class per precision and one ranking per question shape inside it, which
    // is the whole of the rule the table above is read under: precision is the
    // choice the caller has already made from the accuracy their calculation
    // needs and is not traded for speed, and two shapes have not produced the
    // same amount of output for the same arguments.
    if (!report.classes.empty())
    {
        std::string precisions;

        for (const DeviceProbeClass& clause : report.classes)
        {
            if (!precisions.empty())
            {
                precisions += ", ";
            }

            precisions += clause.precision;
        }

        text += Text("\nrankings - one class per precision, and one ranking per question shape "
                     "inside a class.\n  This run's option table carries %s: no entry of one class "
                     "is ordered against an entry of\n  another, and no entry of one shape is "
                     "ordered against an entry of another.\n",
                     precisions.c_str());
    }

    /// The rows of the option table that stand in one ranking: the ones of that
    /// class's precision whose question is that ranking's. It is the rows the
    /// table lists, measured or not, because what a row documents is a fact about
    /// the row whether or not this run got a figure out of it.
    const auto ShapeRows = [&report](const DeviceProbeClass& clause,
                                     const DeviceProbeRanking& ranking) {
        std::vector<const DeviceProbeMeasurement*> rows;

        for (const DeviceProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.precision == clause.precision &&
                measurement.question == ranking.question)
            {
                rows.push_back(&measurement);
            }
        }

        return rows;
    };

    /// The static fallback's own section, which is what a consumer is left with
    /// when the run refused to order a shape.
    ///
    /// It is printed on a refusal and on nothing else, under a heading that says
    /// what it is, and its basis line begins with the words that say where the
    /// name came from: the library's own tables, counted. A shape the run did order
    /// has a measurement instead and gets a line saying no fallback is needed,
    /// which keeps the two kinds of answer from ever being read as one.
    const auto AppendFallback = [&text](const DeviceProbeRanking& ranking) {
        if (ranking.verdict == DeviceProbeVerdict::kRecommend)
        {
            text += "\n  static fallback: none needed, this shape was ordered by measurement\n";
            return;
        }

        text += "\n  static fallback — a heuristic, not a measurement\n";

        if (ranking.heuristicEntry.empty())
        {
            text += "    this shape's rows carry nothing the library's own tables can count, so no "
                    "default\n    is named for it\n";
            return;
        }

        text += Text("    default: %s\n", ranking.heuristicEntry.c_str());
        text += Text("    basis: %s\n", ranking.heuristicBasis.c_str());
        text += "    this name is a property of the library's own entry book and degree tables for "
                "this\n    device, this build and this shape. It is not a measurement of "
                "anything, it took no\n    part in the run above, and it is not to be read beside "
                "a measured figure.\n";
    };

    for (const DeviceProbeClass& clause : report.classes)
    {
        text += Text("\nclass %s\n", clause.precision.c_str());
        text += Text("  %s\n", clause.note.c_str());

        for (const DeviceProbeRanking& ranking : clause.rankings)
        {
            const std::vector<const DeviceProbeMeasurement*> rows = ShapeRows(clause, ranking);

            text += Text("\n  shape %s\n", ranking.question.c_str());
            text += Text("    every entry here was asked for %s\n", ranking.asked.c_str());

            // The question's name is the grouping key and the shape column of a
            // row need not spell it the same way — the ladder to each argument's
            // own order is produced by one all-orders call or by one call per
            // order — so the shapes the rows of this ranking carry are stated
            // where the two are not the same word.
            std::vector<std::string> rowShapes;

            for (const DeviceProbeMeasurement* row : rows)
            {
                if (std::find(rowShapes.begin(), rowShapes.end(), row->shape) == rowShapes.end())
                {
                    rowShapes.push_back(row->shape);
                }
            }

            if (rowShapes.size() != 1 || rowShapes.front() != ranking.question)
            {
                std::string listed;

                for (const std::string& shape : rowShapes)
                {
                    if (!listed.empty())
                    {
                        listed += ", ";
                    }

                    listed += shape;
                }

                text += Text("    the shape column above reports these rows as: %s\n",
                             listed.c_str());
            }

            if (ranking.fastestOverall.empty())
            {
                // Nothing here is ordered, and the two ways that happens are not
                // the same fact: a shape whose every row failed to measure has no
                // figures at all, and a shape whose rows all failed a gate has
                // figures that this run would not place. The rows' own lines in
                // the table say which, and so do the set-aside reasons below.
                std::size_t measuredRows = 0;

                for (const DeviceProbeMeasurement* row : rows)
                {
                    if (row->measured)
                    {
                        ++measuredRows;
                    }
                }

                text += measuredRows == 0 ? "    no entry of this shape produced a figure\n"
                                          : "    no entry of this shape could be ordered\n";
                text += Text("    reason: %s\n", ranking.reason.c_str());

                for (const std::string& entry : ranking.notOrdered)
                {
                    text += Text("    set aside: %s\n", entry.c_str());
                }

                text += Text("    verdict: %s\n",
                             ranking.verdict != DeviceProbeVerdict::kRecommend ? "CANNOT DETERMINE"
                                                                               : "RECOMMEND");
                text += Text("    confidence: %s\n", ranking.confidence.c_str());
                AppendFallback(ranking);
                continue;
            }

            text += Text("    fastest measured: %s\n", ranking.fastestOverall.c_str());

            // A resolution is a distance between two order statistics of the run's
            // ratios, and a run with fewer paired rounds than a quartile band needs
            // has none: the widest band it showed is a band of two or three
            // readings, printed beside a refusal for being too short, and a reader
            // would take it for what this run can order. The field is not printed
            // there, and the line says which count is missing instead.
            if (ranking.resolution > 0.0 &&
                ranking.rounds >= static_cast<int>(kMinimumPairedRounds))
            {
                // The canary is named beside the resolution only where it was read.
                // A run whose canary never ran has no spread to put there, and a
                // zero printed in this line would read as a still machine.
                const std::string canaryContext =
                    anyCanaryMeasured
                        ? Text("The run's canary spread, %.2f%%,\n                is context beside "
                               "it and is not what sets it.\n",
                               report.canarySpread)
                        : std::string("No pass took a canary reading on this run,\n                "
                                      "so there is no canary spread beside it.\n");

                text += Text("    resolution: entries whose within-round ratio band is narrower "
                             "than %.2f%%\n                cannot be ordered on this run. The "
                             "number is measured, not assumed: it\n                is the widest "
                             "ratio band this shape showed, taken over the leader\n"
                             "                against each rival and over each entry against the "
                             "reference, all of\n                them ratios formed inside a "
                             "round. %s",
                             100.0 * ranking.resolution,
                             canaryContext.c_str());
            } else if (ranking.rounds < static_cast<int>(kMinimumPairedRounds))
            {
                text += Text("    resolution: not measurable on this run: it took %d paired round(s) "
                             "and a lower\n                and an upper quartile need %zu of them, "
                             "so the widest band it showed is not\n                a resolution and "
                             "is not printed as one.\n",
                             ranking.rounds,
                             kMinimumPairedRounds);
            } else
            {
                text += "    resolution: not measurable on this run, so this shape is not ordered\n";
            }

            text += Text("    verdict: %s\n",
                         ranking.verdict != DeviceProbeVerdict::kRecommend ? "CANNOT DETERMINE"
                                                                           : "RECOMMEND");

            if (!ranking.recommended.empty())
            {
                text += Text("    recommended entry: %s\n", ranking.recommended.c_str());

                // The bound is a column of a row and not the thing the class is
                // keyed on, so the bound of the row being named is repeated here
                // and the looser or tighter one beside it is named with it: this
                // is the line that gets quoted, and it is the line a reader would
                // quote as "the fastest" without the accuracy it is fastest at.
                double recommendedBound = 0.0;

                for (const DeviceProbeMeasurement* row : rows)
                {
                    if (row->name == ranking.recommended)
                    {
                        recommendedBound = row->documentedBound;
                    }
                }

                // Whether one shape holds one bound or two, and which bound the
                // tightest row beside the winner documents, are both read off the
                // rows rather than assumed from the precision they share.
                bool oneBound = true;
                double otherBound = 0.0;

                for (const DeviceProbeMeasurement* row : rows)
                {
                    if (row->name == ranking.recommended || !(row->documentedBound > 0.0) ||
                        row->documentedBound == recommendedBound)
                    {
                        continue;
                    }

                    oneBound = false;

                    if (otherBound == 0.0 || row->documentedBound < otherBound)
                    {
                        otherBound = row->documentedBound;
                    }
                }

                std::string otherNames;

                for (const DeviceProbeMeasurement* row : rows)
                {
                    if (row->name == ranking.recommended || row->documentedBound != otherBound)
                    {
                        continue;
                    }

                    if (!otherNames.empty())
                    {
                        otherNames += ", ";
                    }

                    otherNames += row->name;
                }

                if (oneBound)
                {
                    text += Text("      this row and every other row of this shape document the "
                                 "same bound, %.2g,\n      so the ranking is at one accuracy and "
                                 "this is the fastest entry of the shape\n      in the %s class "
                                 "at it.\n",
                                 recommendedBound,
                                 clause.precision.c_str());
                } else
                {
                    // The named bound is the tightest one beside the winner, so
                    // the direction holds for that row and the looser rows are
                    // looser still.
                    text += Text("      this row documents a bound of %.2g, %s the %.2g of %s.\n"
                                 "      The winner of this shape is the fastest entry of it at the "
                                 "bound its own row\n      states, which is not the same as the "
                                 "fastest at one accuracy. The bound column above\n      is the "
                                 "figure to weigh against this one.\n",
                                 recommendedBound,
                                 otherBound < recommendedBound ? "looser than" : "tighter than",
                                 otherBound,
                                 otherNames.c_str());
                }
            }

            text += Text("    reason: %s\n", ranking.reason.c_str());

            for (const std::string& entry : ranking.inseparable)
            {
                text += Text("    not separable: %s\n", entry.c_str());
            }

            for (const std::string& entry : ranking.notOrdered)
            {
                text += Text("    set aside: %s\n", entry.c_str());
            }

            text += Text("    confidence: %s\n", ranking.confidence.c_str());
            AppendFallback(ranking);
        }
    }

    text += Text("\n%s\n", report.caveat.c_str());

    {
        std::vector<std::string> probeRows;

        for (const DeviceProbeMeasurement& measurement : report.measurements)
        {
            probeRows.push_back(measurement.name);
        }

        AppendOptionSpace(text, probeRows);
    }

    text += "\nnot measured by this probe, by design: the relaxed accuracy multipliers (the "
            "device lane\n  fixes the multiplier at the call site, so ranking them would need an "
            "instantiation per\n  rung, and the host probe ranks the rungs where they are "
            "reachable at run time); the\n  ordered-batch sort (the AllN entries are handed a "
            "batch that already satisfies their\n  stated precondition, so nothing here is a "
            "figure for sorting one); the arithmetic's\n  accuracy (the device accuracy gate "
            "measures that, against the committed reference).\n";

    return text;
}

} // namespace boys
