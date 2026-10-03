// The option probe's host side: the workload, the protocol, the folding of the rounds into figures,
// the refusal to order noise, and the text a consumer reads.
//
// The clock is not here. Every timed region is a CUDA event pair this file opens and closes through
// boys_cuda_probe_kernels.cu, because the host's own submission cost must not be in the figure and
// the events are the only thing that keeps it out. What this file decides is what to time, how many
// times, and what the results are allowed to say.
//
// The shape is the CPU option probe's (src/boys_probe.cpp): every entry is timed once in every
// round, a comparison between two entries is the ratio of their per-round figures inside one round,
// an entry's reported figure is the middle of its ratios to a reference entry scaled by the
// reference's own lower-quartile cost, the canary beside each pass is a diagnostic that gates
// nothing, the resolution is what the run measured rather than a bar chosen in advance, a rival
// whose band does not clear one is named as unplaced instead of being ordered, and the entries a
// shape's own rounds cannot separate are re-measured on their own and voted on. The differences are
// all forced by the device: the instrument is a kernel and not a host spin, the card is named
// instead of the host, and the entries that exist to run inside the caller's kernel are measured by
// subtraction rather than launched.
//
// The two halves of an in-kernel row's subtraction are timed adjacent inside the *same round*, with
// their order alternating by round, so that row's difference is a within-round pair exactly like a
// ratio between two entries is, and a clock that drifts over the run cancels in it.

#include "boys/boys_cuda_probe.hpp"

#include "boys/boys_coefficients.hpp"
#include "boys/boys_cuda.hpp"
#include "boys/boys_device_tables.hpp"
#include "boys/f16.hpp"

#include "boys_cuda_probe_entries.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <utility>
#include <vector>

// --------------------------------------------------------------------------- The device-side
// boundary (boys_cuda_probe_kernels.cu). Return codes as the lane's own exported functions: 0
// success, 1 an internal error, 2 a CUDA failure.
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
/// between the two order statistics it falls between. Its complement is what the
/// spread beside every figure is made of, so the two ends of a row come from one
/// distribution.
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

/// The quantile the reported cost's ratio to the reference, and the band a pair is placed by, are
/// read at.
///
/// A lower quartile rather than the minimum or the mean: on a card whose clock decays under a
/// sustained load the minimum of a run is its earliest and best-clocked round, which a caller's
/// long workload does not meet, and a mean would let one disturbed round move a figure a consumer
/// builds a default on.
///
/// This is the quantile of the **band** a pair is placed by; an entry's own figure is formed at
/// \c kFigureQuantile.
constexpr double kStatisticQuantile = 0.25;

/// The quantile an entry's own figure is formed at: the middle of its rounds.
///
/// **Not the lower quartile.** Every cost column of the run is its reference entry's own cost scaled
/// by that row's ratio to it, and the reference's ratio is one in every round, so its ratio's lower
/// quartile is one too: at a lower quartile the reference would be scored at the middle of its own
/// rounds while every other entry is scored at the twenty-fifth percentile of its ratio, a credit
/// that belongs to the anchor rather than to the entries.
///
/// The middle is the quantile that treats the two alike, and it is exactly reciprocal under a
/// change of anchor: the set of ratios to the anchor is the set against it, element for element, so
/// two entries agree on which of them is cheaper.
constexpr double kFigureQuantile = 0.5;

/// The number of paired rounds a quartile band needs before it can exist: the two
/// ends of a band are two order statistics, and below four observations they are
/// the same observation twice.
constexpr std::size_t kMinimumPairedRounds = 4;

/// The widest relative width of a within-round ratio band a set of rounds showed, in percent: for
/// each entry, the upper quartile of its ratio to \p reference over the lower quartile of the same
/// ratio, taken at its widest.
///
/// The reference's own column is skipped, and only it: its ratio is one in every round by
/// construction, so it would contribute a band of zero. A set too short for a band, or one whose
/// reference column holds a non-positive reading, reports zero rather than a width it did not
/// measure.
///
/// Every column of the round table is read: a ratio is a ratio inside one set of
/// rounds, and the width is what that set showed.
///
/// \param rounds the pass's round table, one row per round, one column per row of
/// the measurement table, in cost per argument
/// \param reference the anchor's column
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

/// One rival measured against one leader, inside the rounds: every quantity here is a ratio formed
/// inside a single round, so a drift common to the round cancels. This is the comparison the probe
/// is ordered by.
struct PairedOutcome {
    /// Lower and upper quartile of the within-round ratio, rival over leader: the
    /// band the middle half of the run put the pair in.
    double lo = 1.0;
    double hi = 1.0;

    /// Rounds in which the rival was the slower of the two, of the rounds that
    /// could be formed.
    int slowerRounds = 0;
    int rounds = 0;

    /// How far the pair's ratio moved between the run's first and second half of rounds: the second
    /// half's median ratio over the first half's, less one. A pair whose ratio is a property of the
    /// two entries holds still as the card's clock moves; one that drifts is a pair whose two
    /// entries do not carry the clock alike.
    double drift = 0.0;

    /// Whether the band clears one: the middle half of the run put the rival
    /// behind the leader, and the probe orders only what it saw in that half.
    bool ordered = false;
};

/// Measures one rival against one leader over the run's rounds.
///
/// The band is the lower and upper quartile of the per-round ratios, and the ordering is that lower
/// quartile clearing one: the middle half of the run has to put the rival behind, not merely the
/// run's average.
///
/// /// \\param rounds the round table: one row per round, one column per entry, in cost per argument
/// /// \\param leader column of the entry the rival is measured against
/// /// \\param rival column of the entry being placed
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

/// Every value of an enumeration, against one check per value.
///
/// The namers below are covered by a class template each rather than by a predicate over the
/// enumeration's count, and this is what instantiates them. The reason is the diagnostic: a
/// static_assert's message is a string literal, and no C++23 construct turns an enumerator into
/// text, so the message cannot name the value that failed it. An instantiation can - the compiler
/// prints its argument - and on MSVC that note is the difference between "name it in StatusName"
/// and a name to go and find, because MSVC emits no -Wswitch and the message is otherwise the
/// only diagnostic there is. The explicit instantiation beside each namer is what asks the
/// question and is not decoration: a check that nothing instantiates checks nothing.
template <template <auto> class Check, typename Enum, typename Index>
struct EveryValueOf;

template <template <auto> class Check, typename Enum, std::size_t... Index>
struct EveryValueOf<Check, Enum, std::index_sequence<Index...>>
    : Check<static_cast<Enum>(Index)>... {};

/// The named check, over every value the enumeration counts.
template <template <auto> class Check, typename Enum>
struct EveryValue
    : EveryValueOf<Check, Enum, std::make_index_sequence<static_cast<std::size_t>(Enum::kCount)>> {
};

/// The status in words, so a caller reading the text does not have to know the
/// enumerators to see why nothing was measured.
///
/// Every enumerator of \c DeviceProbeStatus is named below and the switch has no default
/// arm, so a status added to the enumeration without words of its own is a compile error
/// rather than a status reported under the sentence belonging to another. That check is
/// what does the work here: MSVC emits no -Wswitch, and the sentence an unstated status
/// would take is itself a plausible one a reader accepts.
constexpr const char* StatusName(DeviceProbeStatus status) noexcept {
    switch (status)
    {
        case DeviceProbeStatus::kSuccess:
            return "measured";
        case DeviceProbeStatus::kNoDevice:
            return "this machine has no CUDA device";
        case DeviceProbeStatus::kDeviceNotFound:
            return "the device asked for is not one this machine has";
        case DeviceProbeStatus::kDeviceError:
            return "a CUDA operation failed";
        case DeviceProbeStatus::kInvalidArgument:
            return "the request named no entry this library has";
        case DeviceProbeStatus::kCount:
            break;
    }

    // A value no arm above names, which the check below turns into a compile error.
    return nullptr;
}

/// One enumerator of \c DeviceProbeStatus that the switch above names no status for. The
/// compilation of this file is what keeps the two in step, and the failed instantiation names
/// the enumerator it stopped on.
template <DeviceProbeStatus Status>
struct StatusIsNamed {
    static_assert(StatusName(Status) != nullptr,
                  "an enumerator of DeviceProbeStatus names no status: name it in StatusName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct EveryValue<StatusIsNamed, DeviceProbeStatus>;

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

    /// The bound the library documents for this entry's precision, read from the library's own
    /// tables rather than measured here: what a lane delivers on a card is the accuracy gate's
    /// business, and this probe spends its time on cost. The report carries it so that a faster
    /// precision is not read as a faster option at the same accuracy.
    double bound;
};

/// Makes the degree tables resident on the current device and fills \p handle from them.
///
/// It is the library's own call (\c BoysCuda::DeviceTables, boys_cuda.hpp): one upload serves every
/// entry of this surface, so it is made once, before any row is timed, and it is host work that
/// belongs outside every timed region. A device that will not hold the tables is a refusal the
/// report carries with the library's own answer rather than an absence: the entries of this surface
/// were never presented to the device, and nothing of them is timed.
///
/// \param handle the table handle the in-kernel rows read
///
/// \returns whether the tables are resident when the call returns
bool TablesAreResident(BoysDeviceTables& handle) {
    return BoysCuda::DeviceTables(&handle) == BoysStatus::kSuccess;
}

constexpr const char* QuestionName(ProbeQuestion question) noexcept {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "single";
        case ProbeQuestion::kAllOrders:
            return "all-orders";
        case ProbeQuestion::kAllN:
            return "all-n";
        case ProbeQuestion::kCount:
            break;
    }

    return nullptr;
}

/// The library's question, said in full: what the option produces for one
/// argument, which is what makes two of them the same question.
constexpr const char* QuestionAsked(ProbeQuestion question) noexcept {
    switch (question)
    {
        case ProbeQuestion::kSingle:
            return "one value per argument: F_n(x) at that argument's own order n";
        case ProbeQuestion::kAllOrders:
            return "a ladder per argument: F_0(x)..F_n(x) at that argument's own order n";
        case ProbeQuestion::kAllN:
            return "one common ladder for the whole batch: F_0(x)..F_nmax(x) for every argument";
        case ProbeQuestion::kCount:
            break;
    }

    return nullptr;
}

/// One enumerator of \c DeviceOptionQuestion that \c QuestionName states no token for.
///
/// Both namers of a question are covered, as one predicate covered them before: a question
/// named by one of them and not the other is the same defect twice, and each instantiation
/// now says which of the two namers and which enumerator it is.
template <DeviceOptionQuestion Question>
struct QuestionIsNamed {
    static_assert(QuestionName(Question) != nullptr,
                  "an enumerator of DeviceOptionQuestion names no question: name it in "
                  "QuestionName (src/boys_cuda_probe.cpp)");
};

/// One enumerator of \c DeviceOptionQuestion that \c QuestionAsked states no sentence for.
template <DeviceOptionQuestion Question>
struct QuestionIsAsked {
    static_assert(QuestionAsked(Question) != nullptr,
                  "an enumerator of DeviceOptionQuestion is asked no question: name it in "
                  "QuestionAsked (src/boys_cuda_probe.cpp)");
};

template struct EveryValue<QuestionIsNamed, DeviceOptionQuestion>;
template struct EveryValue<QuestionIsAsked, DeviceOptionQuestion>;

/// The precision as the report spells it.
///
/// Every enumerator of \c DeviceOptionPrecision is named, and the switch has no default
/// arm. The default this replaced returned the half lane's own name, so the next format
/// added to that lane would have been reported as the one already there — and a bound is
/// read off the precision a row prints, which makes a wrong name here a wrong bound one
/// step later.
constexpr const char* PrecisionName(DeviceOptionPrecision precision) noexcept {
    switch (precision)
    {
        case DeviceOptionPrecision::kFp64:
            return "fp64";
        case DeviceOptionPrecision::kFp32:
            return "fp32";
        case DeviceOptionPrecision::kFp16:
            return "fp16";
        case DeviceOptionPrecision::kCount:
            break;
    }

    return nullptr;
}

constexpr const char* ShapeName(DeviceOptionShape shape) noexcept {
    switch (shape)
    {
        case DeviceOptionShape::kSingle:
            return "single";
        case DeviceOptionShape::kAllOrders:
            return "all-orders";
        case DeviceOptionShape::kAllN:
            return "all-n";
        case DeviceOptionShape::kEachOrder:
            return "each-order";
        case DeviceOptionShape::kCount:
            break;
    }

    return nullptr;
}

/// One enumerator of \c DeviceOptionPrecision that \c PrecisionName spells no name for.
template <DeviceOptionPrecision Precision>
struct PrecisionIsNamed {
    static_assert(PrecisionName(Precision) != nullptr,
                  "an enumerator of DeviceOptionPrecision names no member: name it in "
                  "PrecisionName (src/boys_cuda_probe.cpp)");
};

/// One enumerator of \c DeviceOptionShape that \c ShapeName spells no name for.
template <DeviceOptionShape Shape>
struct ShapeIsNamed {
    static_assert(ShapeName(Shape) != nullptr,
                  "an enumerator of DeviceOptionShape names no member: name it in ShapeName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct EveryValue<PrecisionIsNamed, DeviceOptionPrecision>;
template struct EveryValue<ShapeIsNamed, DeviceOptionShape>;

/// The group as the report spells it. Written as a switch rather than the two-way test it
/// was, which answered "device" to every group but the launched one and would have gone on
/// answering it to a third.
constexpr const char* GroupName(DeviceOptionGroup group) noexcept {
    switch (group)
    {
        case DeviceOptionGroup::kLaunched:
            return "launched";
        case DeviceOptionGroup::kDeviceCallable:
            return "device";
        case DeviceOptionGroup::kCount:
            break;
    }

    return nullptr;
}

/// The fit route a row states, as the report spells it; nullptr for a value outside the
/// enumerators, which the checks below turn into a compile error.
constexpr const char* RouteName(FitRoute route) noexcept {
    switch (route)
    {
        case FitRoute::kChebyshev:
            return "chebyshev";
        case FitRoute::kRationalMinimax:
            return "rational";
    }

    return nullptr;
}

/// One enumerator of \c FitRoute that \c RouteName spells no name for.
template <FitRoute Route>
struct RouteIsNamed {
    static_assert(RouteName(Route) != nullptr,
                  "an enumerator of FitRoute names no route: name it in RouteName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct RouteIsNamed<FitRoute::kChebyshev>;
template struct RouteIsNamed<FitRoute::kRationalMinimax>;

/// The summation a row states, as the report spells it; nullptr for a value outside the
/// enumerators.
constexpr const char* SchemeName(EvalScheme scheme) noexcept {
    switch (scheme)
    {
        case EvalScheme::kSplitClenshaw:
            return "split-clenshaw";
        case EvalScheme::kHorner:
            return "horner";
    }

    return nullptr;
}

/// One enumerator of \c EvalScheme that \c SchemeName spells no name for.
template <EvalScheme Scheme>
struct SchemeIsNamed {
    static_assert(SchemeName(Scheme) != nullptr,
                  "an enumerator of EvalScheme names no summation: name it in SchemeName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct SchemeIsNamed<EvalScheme::kSplitClenshaw>;
template struct SchemeIsNamed<EvalScheme::kHorner>;

/// The reading of region A a row states, as the report spells it; nullptr for a value outside
/// the enumerators.
constexpr const char* PackingName(DevicePacking packing) noexcept {
    switch (packing)
    {
        case DevicePacking::kLadder:
            return "ladder";
        case DevicePacking::kPerOrder:
            return "per-order";
        case DevicePacking::kNotApplicable:
            return "n/a";
        case DevicePacking::kUnstated:
            // Never a row's value: DeviceEntryAxesOf states the three axes for every
            // enumerator of DeviceEntry, and the assertion under that switch stops the
            // build for an enumerator it has not been taught. Named here so the switch
            // is total, and printed as itself if a value ever did arrive.
            return "unstated";
    }

    return nullptr;
}

/// One enumerator of \c DevicePacking that \c PackingName spells no name for.
template <DevicePacking Packing>
struct PackingIsNamed {
    static_assert(PackingName(Packing) != nullptr,
                  "an enumerator of DevicePacking names no reading: name it in PackingName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct PackingIsNamed<DevicePacking::kLadder>;
template struct PackingIsNamed<DevicePacking::kPerOrder>;
template struct PackingIsNamed<DevicePacking::kNotApplicable>;
template struct PackingIsNamed<DevicePacking::kUnstated>;

/// The axes of a row: the member of the axis it varies, where it varies one, and then the route,
/// the summation and the region-A packing its entry runs.
///
/// The three axes last are stated on every row and not only on the rows that move them, because
/// that is what places a row against its neighbours: a row whose axis is kNone is the entry, and
/// before this column carried the three, the arithmetic it runs was recoverable from its name and
/// from nothing the report printed. A row of the packing, scheme or route axis carries its member
/// among the three, so nothing is said twice.
std::string AxisName(const DeviceOptionInfo& option) {
    std::string text;

    switch (option.axis)
    {
        case DeviceOptionAxis::kRegionBExp:
            text = Text("region-B:%s ",
                        option.regionBExp == RegionBExp::kFast ? "fast" : "accurate");
            break;
        case DeviceOptionAxis::kPartition:
            text = Text("partition:%s ", DevicePartitionName(option.entry));
            break;
        case DeviceOptionAxis::kNone:
        case DeviceOptionAxis::kPacking:
        case DeviceOptionAxis::kScheme:
        case DeviceOptionAxis::kRoute:
            break;
        default:
            // An axis named by a newer header: the row is still printed, under
            // the enumerator it was written with rather than under a member of
            // some other axis.
            text = Text("axis:%d ", static_cast<int>(option.axis));
            break;
    }

    text += Text("route:%s scheme:%s packing:%s",
                 RouteName(option.route),
                 SchemeName(option.scheme),
                 PackingName(option.packing));
    return text;
}

/// The degree tables a row reads. The lane is what makes two rows of one precision different
/// arithmetic - the seed amplification it carries is the lane's - so a report that names the
/// precision and not the lane has not said which arithmetic it measured.
constexpr const char* LaneName(BoysDeviceLane lane) noexcept {
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
        case BoysDeviceLane::kF16Batch:
            return "f16-batch";
        case BoysDeviceLane::kCount:
            break;
    }

    return nullptr;
}

/// One enumerator of \c DeviceOptionGroup that \c GroupName spells no name for.
template <DeviceOptionGroup Group>
struct GroupIsNamed {
    static_assert(GroupName(Group) != nullptr,
                  "an enumerator of DeviceOptionGroup names no member: name it in GroupName "
                  "(src/boys_cuda_probe.cpp)");
};

/// One enumerator of \c BoysDeviceLane that \c LaneName spells no name for.
template <BoysDeviceLane Lane>
struct LaneIsNamed {
    static_assert(LaneName(Lane) != nullptr,
                  "an enumerator of BoysDeviceLane names no member: name it in LaneName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct EveryValue<GroupIsNamed, DeviceOptionGroup>;
template struct EveryValue<LaneIsNamed, BoysDeviceLane>;

/// The sentence a class opens with: what a class is, and what its winner is therefore a claim
/// about.
///
/// The class is one precision and one question shape, and it is keyed on those two rather than on
/// anything the library picks because a caller has already chosen fp64, fp32 or fp16 from the
/// accuracy their calculation needs, and the question they are asking. So no entry of this class is
/// a substitute for one of another class - it is a faster or slower way to answer the same call.
/// The bounds the rows document are then their own, and whether one class holds two of them is read
/// off the rows rather than assumed.
///
/// /// \\param clause a row of the class, for the precision and the question it is keyed on
/// /// \\param asked the class's question said in full
/// /// \\param members how many rows the class holds, measured or not
/// /// \\param oneBound whether every row of it documents one bound
std::string ClassNote(const DeviceProbeMeasurement& clause,
                      const std::string& asked,
                      std::size_t members,
                      bool oneBound) {
    std::string text = Text(
        "every entry in this class runs %s, and answers the question \"%s\". The class "
        "is one precision and one question shape, because those two are what "
        "the caller has already chosen when they make the call: no entry of this class is an "
        "alternative to one of another class, and no entry of it was ordered against one. What "
        "varies inside the class is what the library picks on the caller's behalf — the fit route, "
        "the evaluation scheme, the partition and the packing axis — so its winner is the fastest "
        "%s entry of this question, at the bound its own row states, and the bound "
        "column above is that figure.",
        clause.precision.c_str(),
        asked.c_str(),
        clause.precision.c_str());

    text += Text(" This class holds %zu row(s) of the option table here.", members);

    text += oneBound
                ? std::string(" They document one bound between them, so the ranking is at one "
                              "accuracy.")
                : std::string(" They do not document one bound between them: a row of a looser "
                              "bound is a faster way to answer this question at that accuracy, and "
                              "both stand in the one ranking, which is what makes it a ranking by "
                              "cost.");

    return text;
}

/// The option table, projected from the library's own report: one probe row per report row this
/// build serves, and one unoffered entry per report row it does not, carrying the library's reason.
///
/// Nothing here names an option the library does not report: a row the library adds appears in this
/// probe's table, and a row it refuses appears in the probe's list of what this build cannot serve,
/// both without an edit here.
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
                                    option.bound});
    }

    return entries;
}

// ---------------------------------------------------------------------------
// The workload.
// ---------------------------------------------------------------------------

/// The library's own question, as the CPU probe asks it: a set of arguments in which each argument
/// carries its own highest order, drawn as the sum of two shell angular momenta, over a log-uniform
/// range that populates every region of the kernel.
///
/// The pairs are sorted by argument because the uniform-order entries document that their arguments
/// are non-decreasing: this probe does not sort for them, so it has to present them a batch that
/// already satisfies what they promise.
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

    // Sorted by argument, so the uniform-order entries' stated precondition
    // holds.
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

    // The narrow lanes' argument arrays. The batch entries of the fp32 and fp16 lanes take the
    // double argument and round it themselves; the device-callable entries of those lanes take the
    // narrowed argument, which is what a caller's own kernel would hold in a register.
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
/// A launched row fills \c withMs and leaves \c withoutMs at zero. An in-kernel row fills both: the
/// caller's kernel with the call in it, and the same kernel with the call removed and the traffic
/// kept. Both halves are kept, not only their difference, so the report can print the subtraction
/// rather than assert that one happened.
struct TimedEntry {
    double withMs = 0.0;
    double withoutMs = 0.0;
    bool subtracted = false;
    bool ok = false;
};

/// One entry's timed figure for one round, in milliseconds of device time.
///
/// A launched entry is one region. A device-callable entry is two, taken adjacent in the same
/// round, so a drift in the card's clocks moves both halves together and cancels in the difference.
///
/// Which half goes first alternates by round, and both halves of every round are kept. A region
/// carries a small fixed cost of its own - the event pair, the first launch's cold start, the
/// host's submission of the launch - divided by the repetition count rather than by the call, so
/// whichever half pays it lands in the difference as a term that shrinks as the repetition count
/// rises. That is a bias, not noise: it does not average away, and it moves the figure between two
/// repetition counts, which is what the subtraction control found. Alternating the order spreads it
/// over both halves.
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

/// Times one entry at two very different repetition counts and fills a repetition-count control
/// from what the two said.
///
/// The counts have to be far apart for the check to mean anything: a cost paid per region rather
/// than per call is divided by a different number of launches at each count, so it changes the
/// per-argument figure by that cost's share of it times the ratio of the counts. An entry whose
/// figure holds across the two has had such a cost divided out.
///
/// **What the two counts are compared on is the figure the report ships**: each count's figure is
/// that count's own readings at both of the run's argument counts, reduced to the count-independent
/// cost they extrapolate to - the same quantity, formed the same way, as the figure the row carries
/// in the table. A control reading a row at one argument count while the report ships the
/// extrapolation of two would be checking a number beside the figure.
///
/// Both halves of a subtraction are reported, not only their difference, so that a reader can see
/// the difference was not two unstable readings cancelling. Returns false when any of the four
/// regions could not be timed.
///
/// Each count is read over \c kMinimumPairedRounds rounds with the order of the two halves
/// alternating, and each count's figure is the lower quartile of its own rounds' figures, with an
/// in-kernel round's difference formed inside the round. **The check fails closed**: a count whose
/// readings left no figure to form leaves no second figure to compare, and a run that cannot
/// resolve the row cannot vouch for its repetition counts either.
///
/// /// \\param base the run's own request at the first of its two argument counts
/// /// \\param pairBase the same request at the second count, which is what makes each figure the row's
///                 own rather than a reading beside it
bool RunRepetitionControl(const EntryInfo& info,
                          const DeviceProbeMeasurement& row,
                          const ProbeTimeRequest& base,
                          const ProbeTimeRequest& pairBase,
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

    // The ratio between the two argument counts the rows are read at, as their own
    // figures use it: every region the control takes is reduced to the
    // count-independent cost its pair extrapolates to.
    const double pairRatio =
        static_cast<double>(pairBase.count) / static_cast<double>(base.count);

    for (int round = 0; round < kRounds && ok; ++round)
    {
        // Alternated, so the per-region cost that whichever half goes second
        // pays is not read as a difference between the halves.
        const bool baselineFirst = (round % 2) == 1;
        // Each count of launches is read at both of the run's argument counts, and
        // the four regions are timed adjacently inside one round, so both figures
        // are taken under one clock and one workload state.
        const TimedEntry lowAtCount =
            TimeEntry(info, base, clamped.controlRepetitionsLow, baselineFirst);
        const TimedEntry lowAtPair =
            TimeEntry(info, pairBase, clamped.controlRepetitionsLow, baselineFirst);
        const TimedEntry highAtCount =
            TimeEntry(info, base, clamped.controlRepetitionsHigh, baselineFirst);
        const TimedEntry highAtPair =
            TimeEntry(info, pairBase, clamped.controlRepetitionsHigh, baselineFirst);

        ok = lowAtCount.ok && lowAtPair.ok && highAtCount.ok && highAtPair.ok;
        subtracted = lowAtCount.subtracted && lowAtPair.subtracted && highAtCount.subtracted &&
                     highAtPair.subtracted;

        if (!ok)
        {
            break;
        }

        // One region, at the count it was taken at, reduced to a per-argument
        // figure: a launched row is its region, an in-kernel row the difference
        // between the region that held the call and the one that did not.
        const auto Reading = [&](const TimedEntry& timed, std::size_t count, int reps) {
            const double with = PerArgument(timed.withMs, reps, count);

            if (!timed.subtracted)
            {
                return with;
            }

            return with - PerArgument(timed.withoutMs, reps, count);
        };

        // One count of launches reduced to the row's own figure: the extrapolation
        // the table's figure is, from the same two argument counts.
        const auto Figure = [&](const TimedEntry& atCount, const TimedEntry& atPair, int reps) {
            const double first = Reading(atCount, base.count, reps);
            const double second = Reading(atPair, pairBase.count, reps);

            return (pairRatio * second - first) / (pairRatio - 1.0);
        };

        lowRounds.push_back(Figure(lowAtCount, lowAtPair, clamped.controlRepetitionsLow));
        highRounds.push_back(Figure(highAtCount, highAtPair, clamped.controlRepetitionsHigh));

        if (subtracted)
        {
            baselineLow.push_back(Reading(lowAtCount, base.count, clamped.controlRepetitionsLow));
            baselineHigh.push_back(
                Reading(highAtCount, base.count, clamped.controlRepetitionsHigh));
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
    // A count whose readings left no figure to form - a launched row whose two readings left no
    // positive launch term, a subtraction that resolved nothing - leaves no second figure to
    // compare, and the one outcome that must never come out of that is agreement: the control would
    // be passing without evidence, and this is the check the whole subtraction route rests on. The
    // difference is infinite rather than zero, so the data says what the note says.
    const bool resolved = smaller > 0.0;

    out.difference = resolved
                         ? std::fabs(out.nsPerArgumentHigh - out.nsPerArgumentLow) / smaller
                         : std::numeric_limits<double>::infinity();
    // The widest within-round band this row's own shape showed on this run is what the two counts
    // are judged against - the same number that shape's refusal prints - and not a bar invented
    // here. A run that could not resolve the row has no yardstick for its repetition counts either,
    // and the control says so rather than passing.
    out.agrees = judgedAgainst > 0.0 && out.difference <= judgedAgainst;

    const std::string comparison =
        resolved
            ? Text("a disagreement of %.2f%%, %s the %.2f%% this shape could order at (the widest "
                   "within-round ratio band its own rows showed over this run), which is the check "
                   "that a cost paid per region rather than per call did not survive into the "
                   "figure at the repetition count it was taken at. Both figures are the row's own "
                   "figure, each formed from that count's readings at %d and at %d arguments, so "
                   "what is judged here is the number the table ships and not a reading beside it",
                   100.0 * out.difference,
                   out.agrees ? "inside" : "OUTSIDE",
                   100.0 * judgedAgainst,
                   static_cast<int>(base.count),
                   static_cast<int>(pairBase.count))
            : Text("no disagreement to report, because at %d launches the row's own readings left "
                   "no figure to form - a launched row whose two readings left no positive launch "
                   "term to take out, or a subtraction that resolved nothing - so there is no "
                   "second figure to set beside the one at %d launches. This control has not "
                   "established that the figure holds as the launches are repeated, and the row is "
                   "not ordered on it",
                   clamped.controlRepetitionsHigh,
                   clamped.controlRepetitionsLow);

    const std::string halves =
        subtracted
            ? Text("the subtraction's own two halves at the lower argument count were %.3f "
                   "ns/argument with the call against %.3f without it at %d launches, and %.3f "
                   "against %.3f at %d, so the difference is between two readings that were "
                   "themselves stable, not two unstable ones cancelling",
                   out.nsPerArgumentBaselineLow + out.nsPerArgumentLow,
                   out.nsPerArgumentBaselineLow,
                   clamped.controlRepetitionsLow,
                   out.nsPerArgumentBaselineHigh + out.nsPerArgumentHigh,
                   out.nsPerArgumentBaselineHigh,
                   clamped.controlRepetitionsHigh)
            : Text("the launched row holds no subtraction: each of its two figures is a region's "
                   "own device time at that count of launches, read at both of the run's argument "
                   "counts and reduced to the count-independent cost they extrapolate to, so the "
                   "two are figures of one kind and both are the row's own");

    out.note = Text("'%s' on the %s route at %d and at %d launches gives %.3f against %.3f "
                    "ns/argument, each formed from that count's readings at this run's %d and %d "
                    "arguments as the table's own figure is: %s; %s",
                    row.name.c_str(),
                    row.route.c_str(),
                    clamped.controlRepetitionsLow,
                    clamped.controlRepetitionsHigh,
                    out.nsPerArgumentLow,
                    out.nsPerArgumentHigh,
                    static_cast<int>(base.count),
                    static_cast<int>(pairBase.count),
                    comparison.c_str(),
                    halves.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// The conclusion for one question shape of one precision.
// ---------------------------------------------------------------------------

/// Folds one shape's live measurements into a verdict, on the CPU probe's rule: the shape's leader
/// is the entry with the lowest figure the report prints for it, and it stands only when every
/// rival's own within-round band against it has a lower quartile above one. A rival whose band
/// straddles one cannot be placed, and a rival whose band lies wholly below one was ahead in that
/// half, which means the statistic and the rounds disagree about the pair; both end in a refusal
/// that names each unplaced rival with its band and its slower-round count and prints the run's own
/// resolution. The ordering is never weakened into a ranking: no winner is named while any rival is
/// unplaced.
///
/// Every row handed in is already of one precision and one question shape, which is what makes them
/// comparable at all. The bounds those rows document are columns of their own rows and are not what
/// this orders on: two rows at different bounds are still two ways to compute the precision the
/// caller chose.
///
/// /// \\param clause the ranking to fill in
/// /// \\param live the shape's measured rows, in the report's own order
/// /// \\param columns each live row's own column in \p rounds, in the same order as \p live
/// /// \\param rounds \param rounds the round table the report's figures were aggregated from: one row
///               per pooled round, one column per entry in the measurements' own order, in cost per
///               argument
/// /// \\param pairedRounds rounds the run pooled, which is what a quartile band needs four of
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

/// The sentence a run too short to form a within-round band owes the reader of the
/// name it still reaches: the count it took and the count a band needs.
///
/// Below four paired rounds the two ends of a band are the same reading twice, so
/// no pair of a shape can be placed on it and every name such a run reaches is an
/// answer without a comparison behind it. The name is still owed — a shape whose
/// entries produced figures ends with one — and this is what its reader is told
/// about the rounds it came from. Empty when the run was long enough for a band.
///
/// /// \\param pairedRounds rounds the run pooled
std::string ShortRunClause(int pairedRounds) {
    if (pairedRounds >= static_cast<int>(kMinimumPairedRounds))
    {
        return {};
    }

    return Text(" The run took %d paired round(s), and the lower and upper quartiles of a "
                "within-round ratio need %zu of them: with fewer, the band is the ratio of two or "
                "three rounds rather than a quartile of many, and this probe will not order "
                "entries on it. Raising DeviceProbeOptions::passes or DeviceProbeOptions::rounds "
                "is what this needs.",
                pairedRounds,
                kMinimumPairedRounds);
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
                             "so there is nothing to order.",
                             live.empty() ? "instrument" : "kernel without the call");

        // A run below the band's floor is why nothing here could be placed, and the
        // name the shape reaches below is read differently because of it, so the
        // count is said rather than left for the reader to infer from a name.
        clause.reason += ShortRunClause(pairedRounds);

        // The widest band the rows' own spreads against the reference showed, so
        // that a refusal does not print a zero that reads as a resolution never
        // measured.
        double widestOwnBand = 0.0;

        for (DeviceProbeMeasurement* measurement : live)
        {
            if (measurement->measured)
            {
                widestOwnBand = std::max(widestOwnBand, measurement->spread - 1.0);
            }
        }

        clause.resolution = std::max(0.0, widestOwnBand);

        // A shape the run's own checks emptied is not left without an answer:
        // these rows are the band the refinement stage re-runs and votes on. A
        // shape that measured no figure at all has nothing to re-run, and that is
        // the one case this shape ends with no entry.
        std::vector<DeviceProbeMeasurement*> figures;

        for (DeviceProbeMeasurement* measurement : live)
        {
            if (measurement->measured && measurement->nsPerArgument > 0.0)
            {
                figures.push_back(measurement);
            }
        }

        std::sort(figures.begin(), figures.end(),
                  [](const DeviceProbeMeasurement* a, const DeviceProbeMeasurement* b) {
                      return a->nsPerArgument < b->nsPerArgument;
                  });

        if (figures.empty())
        {
            clause.confidence =
                live.empty()
                    ? std::string("CANNOT DETERMINE: no entry of this shape produced a figure at "
                                  "all")
                    : Text("CANNOT DETERMINE: %zu %s of this shape measured and none resolved",
                           live.size(),
                           live.size() == 1 ? "entry" : "entries");
            return;
        }

        clause.fastestOverall = figures.front()->name;

        if (figures.size() > 1)
        {
            // The band the stage re-runs, the fastest first.
            for (DeviceProbeMeasurement* measurement : figures)
            {
                clause.tiedEntries.push_back(measurement->name);
            }

            clause.confidence =
                Text("%zu entries of this shape produced a figure and the run's own checks placed "
                     "none of them, so the answer below comes from the refinement stage's own runs "
                     "over the %zu of them that did",
                     figures.size(),
                     figures.size());
            return;
        }

        // One figure is not a ranking and the stage has nothing to vote between;
        // the check that did not place it is recorded beside it rather than
        // dropped.
        clause.verdict = DeviceProbeVerdict::kRecommend;
        clause.recommended = figures.front()->name;
        clause.defaultHow = DeviceProbeDefaultHow::kOnlyEntry;
        clause.reason +=
            Text(" One entry of this shape produced a figure - '%s' at %.3f ns/argument - and it "
                 "is the entry this shape names: with nothing else in the shape to order it "
                 "against, the name rests on there being no alternative to it and not on a "
                 "comparison",
                 figures.front()->name.c_str(),
                 figures.front()->nsPerArgument);
        clause.confidence =
            Text("one entry of this shape produced a figure at all, and the run's own checks did "
                 "not place it, so the shape names it by there being no alternative to it; its "
                 "figure is the entry's own cost and is not a margin over anything");
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
            driftPair =
                Text("'%s' against '%s'", candidate.row->name.c_str(), leader.row->name.c_str());
        }
    }

    clause.resolution = widestBand;

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

    // A quartile band needs four rounds. With fewer, the lower and upper quartiles
    // are two- and three-point order statistics, and a probe that ordered on them
    // would be reporting a resolution it never measured. The shape is not left
    // without an answer because of it: the entries this run read a figure for are
    // the band it could not form, and they are what the refinement stage re-runs at
    // a longer protocol. A shape with one such entry has nothing to vote between,
    // and it is named by there being no alternative to it.
    if (pairedRounds < static_cast<int>(kMinimumPairedRounds))
    {
        std::vector<const Candidate*> read;

        for (const Candidate& candidate : ordered)
        {
            if (candidate.row->measured && candidate.row->nsPerArgument > 0.0)
            {
                read.push_back(&candidate);
            }
        }

        clause.reason = Text("'%s' is the fastest entry of this shape by the run's own statistic "
                             "(%.3f ns/argument, its within-round ratio to the reference read at "
                             "the middle of the run's rounds), but the run took %d paired "
                             "round(s), and the lower and upper quartiles of a "
                             "within-round ratio need %zu of them: with fewer, the band is the "
                             "ratio of two or three rounds rather than a quartile of many, and "
                             "this probe will not order entries on it. Raising "
                             "DeviceProbeOptions::passes or DeviceProbeOptions::rounds is what "
                             "this needs.",
                             leader.row->name.c_str(),
                             leader.row->nsPerArgument,
                             pairedRounds,
                             kMinimumPairedRounds);

        if (read.size() < 2)
        {
            // The leader is named, not won: it is the only row of the shape the run
            // read a figure for.
            clause.verdict = DeviceProbeVerdict::kRecommend;
            clause.recommended = leader.row->name;
            clause.defaultHow = DeviceProbeDefaultHow::kOnlyEntry;
            clause.reason += Text(" One entry of this shape and precision produced a figure here, "
                                  "and it is the entry this shape names: there was no second entry "
                                  "to place it against and no band to place it on, so the name "
                                  "rests on there being no alternative to it and not on a "
                                  "comparison");
            clause.confidence =
                Text("one entry of this shape and precision produced a figure, and the run took "
                     "%d paired round(s), fewer than the %zu a band needs, so the shape names it "
                     "by there being no alternative to it; its figure is the entry's own cost and "
                     "is not a margin over anything",
                     pairedRounds,
                     kMinimumPairedRounds);
            return;
        }

        // Every entry the run read a figure for is a candidate it did not place,
        // which is the band the refinement stage re-runs.
        for (const Candidate* candidate : read)
        {
            clause.tiedEntries.push_back(candidate->row->name);
        }

        clause.confidence =
            Text("%zu entries of this shape produced a figure, and the run took %d paired "
                 "round(s), fewer than the %zu a quartile band needs, so the answer below comes "
                 "from the refinement stage's own runs over every one of them",
                 read.size(),
                 pairedRounds,
                 kMinimumPairedRounds);
        return;
    }

    if (ordered.size() < 2)
    {
        // One entry is not a ranking and not a failure: the reason line names it by
        // there being no alternative to it rather than as the winner of a
        // comparison.
        clause.verdict = DeviceProbeVerdict::kRecommend;
        clause.recommended = leader.row->name;
        clause.defaultHow = DeviceProbeDefaultHow::kOnlyEntry;
        clause.reason = Text("'%s' is the only entry of this shape and precision this run could "
                             "order, and it is the entry this shape names: %.3f ns/argument, its "
                             "within-round ratio to the reference read at the middle of the %d "
                             "paired rounds. One entry is not a ranking - "
                             "there is nothing here to order it against, and this run measured "
                             "none - so the name rests on there being no alternative to it and not "
                             "on a comparison",
                             leader.row->name.c_str(),
                             leader.row->nsPerArgument,
                             pairedRounds);
        clause.confidence = Text("one entry of this shape in this class could be ordered, so the "
                                 "shape is that entry with no rival to place behind it; the figure "
                                 "is the entry's own cost and is not a margin over anything");
        clause.confidence += DriftClause(driftPair, widestDrift, clause.resolution);
        return;
    }

    // The rivals, each placed by its own band against the leader and by nothing
    // else. A band whose lower quartile does not clear one is a pair this run did
    // not put in order, in either direction, and the probe will not name a winner
    // while one of those stands.
    std::vector<std::string> within;
    std::vector<std::string> ahead;
    // The same two lists as names, for the pool the refinement stage re-runs:
    // the prose above is what the report prints, and these are what the stage
    // needs, and both are built from one pass over the rivals so the two cannot
    // name different entries.
    std::vector<std::string> withinNames;
    std::vector<std::string> aheadNames;
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
            aheadNames.push_back(row->name);
        } else
        {
            within.push_back(line);
            withinNames.push_back(row->name);
        }
    }

    if (!within.empty() || !ahead.empty())
    {
        std::vector<std::string> unplaced = within;
        unplaced.insert(unplaced.end(), ahead.begin(), ahead.end());
        clause.inseparable = unplaced;

        // The entries this shape's own rounds could not separate, the leader
        // first: the pool the refinement stage re-runs and votes on, recorded as
        // rows and not only as prose.
        clause.tiedEntries.push_back(leader.row->name);
        clause.tiedEntries.insert(clause.tiedEntries.end(), withinNames.begin(), withinNames.end());
        clause.tiedEntries.insert(clause.tiedEntries.end(), aheadNames.begin(), aheadNames.end());

        clause.reason = Text(
            "'%s' is the fastest entry of this shape by the run's own statistic - %.3f "
            "ns/argument, its within-round ratio to the reference read at the middle of the %d "
            "paired rounds, %.3f with that ratio at the upper - but "
            "%zu of the %zu entry(s) of this shape could not be placed behind it: %s. The band and "
            "the slower-round count of every one of them are listed below. Ordering a pair whose "
            "band straddles one would be ordering noise, and ordering one whose band lies below "
            "one would contradict the run's own statistic, so the probe does neither: the entries "
            "this shape could not separate are re-run on their own, below, and the entry that led "
            "those runs is the one this shape names",
            leader.row->name.c_str(),
            leader.row->nsPerArgument,
            pairedRounds,
            leader.row->nsPerArgumentMax,
            unplaced.size(),
            ordered.size(),
            !within.empty() ? within.front().c_str() : ahead.front().c_str());
        clause.confidence = Text("the shape's own rounds left %zu of %zu entry(s) unplaced against "
                                 "'%s' over the %d paired rounds, so the widest band this shape "
                                 "showed, %.2f%%, is what it could not separate and not a "
                                 "resolution it ordered by",
                                 unplaced.size(),
                                 ordered.size(),
                                 leader.row->name.c_str(),
                                 pairedRounds,
                                 100.0 * clause.resolution);

        // Guarded like the recommendation below: this branch is only reached with
        // a quartile band's worth of rounds.
        clause.confidence += DriftClause(driftPair, widestDrift, clause.resolution);
        return;
    }

    clause.verdict = DeviceProbeVerdict::kRecommend;
    clause.recommended = leader.row->name;
    clause.defaultHow = DeviceProbeDefaultHow::kOrdered;

    clause.reason = Text("'%s' is the fastest entry of this shape: %.3f ns/argument, its "
                         "within-round ratio to the reference read at the middle of the %d paired "
                         "rounds, %.3f with that ratio at the upper (spread %.2fx), %.3f "
                         "ns/argument at its peak. Every other entry of the shape was the "
                         "slower of the two in the middle half of the same rounds.",
                         leader.row->name.c_str(),
                         leader.row->nsPerArgument,
                         pairedRounds,
                         leader.row->nsPerArgumentMax,
                         leader.row->spread,
                         leader.row->nsPerArgumentPeak);

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
/// It is the number the shape's refusal prints and the number the repetition control's two counts
/// are judged against, computed once from the pairs rather than twice from two thresholds: an
/// entry's own band to the reference, every rival's band against the shape's fastest entry, and the
/// fastest entry's own band, taken at their widest.
///
/// A shape with fewer than two rows that could be ordered has no pair to form, and reports the
/// fastest row's own band, which is what one row's rounds can say about themselves.
///
/// The rows formed into pairs are the ones of one class: one precision and one question shape.
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

/// The refinement stage: a shape's tied entries, measured alone at a larger protocol, repeated, and
/// voted on.
///
/// The entries re-measured are the shape's fastest entry and every entry of it the main run could
/// not place behind the fastest. Each run is a fresh pass over the tied set at \c passes * the
/// refinement factor passes of \c rounds * the same factor rounds, with its own shuffle, ordered by
/// the same within-round ratio rule the main run used. A run whose own rounds cannot place a rival
/// contributes its leader alone, which is what makes the vote a vote rather than a re-run of the
/// main statistic.
///
/// **The figure it ranks is the main run's figure**: each entry is read at both of the run's
/// argument counts inside the round and the cell is the count-independent cost the two readings
/// extrapolate to, formed exactly as the main run forms it. A refinement that ranked a different
/// quantity from the one the main run could not separate would be answering a question the shape
/// did not ask.
///
/// The stage settles **which entry the shape names** where the shape's own rounds could not: the
/// entry the vote named is the recommendation, and the entry the shape's own figures put first stays
/// the record of what the shorter protocol said. The two are printed together, and where they differ
/// the difference is what says the shape's top entries cannot be separated. A stage that placed no
/// leader at all leaves the shape with the entry its own figures put first.
///
/// /// \\param clause the shape whose tie is to be refined, with its tiedEntries filled
/// /// \\param entries the rows this run measured, in the report's own order
/// /// \\param base the request every timed region of this run is made from, at the run's first count
/// /// \\param pairBase the same request at the run's second count
/// /// \\param clamped the protocol in force, read for the run count and the factor
/// /// \\param live the class's own rows, the set the tied names are resolved in
void RefineShape(DeviceProbeRanking& clause,
                 const std::vector<DeviceProbeMeasurement*>& live,
                 const std::vector<EntryInfo>& entries,
                 const ProbeTimeRequest& base,
                 const ProbeTimeRequest& pairBase,
                 const DeviceProbeOptions& clamped,
                 BoysDeviceTables& handle) {
    // A name is fixed here, before the stage can run, so that a report on a run where
    // the stage cannot be taken still names a row rather than none. It is the fallback
    // and not the answer: where the stage does run, the vote names the default and this
    // one becomes the record of what the shape's own figures put first.
    if (!clause.fastestOverall.empty())
    {
        clause.verdict = DeviceProbeVerdict::kRecommend;
        clause.recommended = clause.fastestOverall;
        clause.defaultHow = DeviceProbeDefaultHow::kChosenAmongEquals;
    }

    // The degree tables, resident before the stage's first launch: a vote taken
    // under no tables at all is no vote, so a device that will not hold them leaves
    // the class with the name above and the route it already carries rather than
    // with a vote.
    if (!TablesAreResident(handle))
    {
        return;
    }

    // The stage times with the run's own requests: one arithmetic, so there is
    // nothing to re-point them at between the main run and the stage.
    const ProbeTimeRequest& runBase = base;
    const ProbeTimeRequest& runPairBase = pairBase;

    DeviceProbeRefinement stage;
    stage.runs = std::max(1, clamped.refinementRuns);
    stage.passes = std::max(1, clamped.passes) * std::max(1, clamped.refinementFactor);
    stage.rounds = std::max(1, clamped.rounds) * std::max(1, clamped.refinementFactor);

    // A name this class's own rows do not carry is not measured: the stage re-runs
    // the rows the main run timed.
    std::vector<std::size_t> pool;

    for (const std::string& name : clause.tiedEntries)
    {
        for (const DeviceProbeMeasurement* row : live)
        {
            if (row->name == name)
            {
                pool.push_back(row->entryIndex);
                break;
            }
        }
    }

    if (pool.size() < 2)
    {
        // Nothing to vote between: the stage cannot refine what the main run
        // could not place, and the shape keeps the refusal it already has.
        return;
    }

    stage.ran = true;

    for (const std::size_t index : pool)
    {
        stage.pool.push_back(entries[index].name);
    }

    std::vector<std::size_t> visit(pool.size());

    for (std::size_t slot = 0; slot < pool.size(); ++slot)
    {
        visit[slot] = slot;
    }

    std::vector<std::string> winners;

    for (int run = 0; run < stage.runs; ++run)
    {
        // A fresh seed per run: repetitions of one shuffle would be one run taken
        // several times, and whatever a position in the round is worth would be
        // worth the same to the same entry every time.
        std::mt19937_64 shuffle(clamped.seed +
                                static_cast<std::uint64_t>(run + 1) * 0x9e3779b97f4a7c15ull);
        std::vector<std::vector<double>> table;

        for (int pass = 0; pass < stage.passes; ++pass)
        {
            for (int round = 0; round < stage.rounds; ++round)
            {
                std::shuffle(visit.begin(), visit.end(), shuffle);

                // Which half of an in-kernel pair goes first alternates by round,
                // as it does in the main run: the fixed cost of a timed region
                // then lands on each half in half the rounds of this run too.
                const bool baselineFirst = ((pass * stage.rounds + round) % 2) == 1;

                std::vector<double> row(pool.size(),
                                        std::numeric_limits<double>::infinity());

                // The pair ratio of this run's own counts, from the count the
                // requests carry rather than from a second reading of the options:
                // a stage that extrapolated with a different factor from the
                // figures it is refining would be ranking a different quantity.
                const double pairRatio =
                    static_cast<double>(runPairBase.count) / static_cast<double>(runBase.count);

                for (const std::size_t slot : visit)
                {
                    const EntryInfo& info = entries[pool[slot]];
                    const TimedEntry atCount =
                        TimeEntry(info, runBase, clamped.repetitions, baselineFirst);
                    const TimedEntry atPair =
                        TimeEntry(info, runPairBase, clamped.repetitions, baselineFirst);

                    if (!atCount.ok || !atPair.ok)
                    {
                        continue;
                    }

                    const auto Reading = [&](const TimedEntry& timed, std::size_t count) {
                        const double with = PerArgument(timed.withMs, clamped.repetitions, count);

                        if (!timed.subtracted)
                        {
                            return with;
                        }

                        const double without =
                            PerArgument(timed.withoutMs, clamped.repetitions, count);

                        return with - without;
                    };

                    const double extrapolated =
                        (pairRatio * Reading(atPair, runPairBase.count) -
                         Reading(atCount, runBase.count)) /
                        (pairRatio - 1.0);

                    // The main run's rule for the two routes: a launched cell at
                    // or below zero is a round with no figure in it and stays
                    // infinite, and a subtraction at or below zero is the
                    // instrument's resolution and is held at zero.
                    if (info.inKernel)
                    {
                        row[slot] = std::max(0.0, extrapolated);
                    } else if (extrapolated > 0.0)
                    {
                        row[slot] = extrapolated;
                    }
                }

                table.push_back(std::move(row));
            }
        }

        // The run's own leader: the entry with the lowest figure at the lower
        // quartile of its own rounds. A run that timed too little of an entry
        // leaves it out rather than ranking an infinity.
        std::size_t leaderSlot = pool.size();
        double leaderCost = 0.0;

        for (std::size_t slot = 0; slot < pool.size(); ++slot)
        {
            std::vector<double> cells;

            for (const std::vector<double>& row : table)
            {
                if (std::isfinite(row[slot]) && row[slot] > 0.0)
                {
                    cells.push_back(row[slot]);
                }
            }

            if (cells.size() < static_cast<std::size_t>(kMinimumPairedRounds))
            {
                continue;
            }

            const double cost = QuantileOf(cells, kStatisticQuantile);

            if (leaderSlot == pool.size() || cost < leaderCost)
            {
                leaderSlot = slot;
                leaderCost = cost;
            }
        }

        // A run whose pool is too thin for a band contributes no name rather than
        // a guess, and the vote counts only the runs that placed one.
        winners.push_back(leaderSlot == pool.size() ? std::string()
                                                    : entries[pool[leaderSlot]].name);
        stage.runLeaders.push_back(winners.back());
    }

    // The vote, in the order the candidates first led a run.
    std::vector<std::string> candidates;
    std::vector<int> counts;

    for (const std::string& winner : winners)
    {
        if (winner.empty())
        {
            continue;
        }

        const auto found = std::find(candidates.begin(), candidates.end(), winner);

        if (found == candidates.end())
        {
            candidates.push_back(winner);
            counts.push_back(1);
        } else
        {
            ++counts[static_cast<std::size_t>(found - candidates.begin())];
        }
    }

    int best = 0;

    for (const int count : counts)
    {
        best = std::max(best, count);
    }

    std::size_t bestIndex = 0;
    int tiedAtBest = 0;

    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        if (counts[index] == best)
        {
            if (tiedAtBest == 0)
            {
                bestIndex = index;
            }

            ++tiedAtBest;
        }
    }

    for (std::size_t index = 0; index < candidates.size(); ++index)
    {
        stage.tally.push_back(
            Text("%s led %d of %d run(s)", candidates[index].c_str(), counts[index], stage.runs));
    }

    if (candidates.empty())
    {
        // No run placed a leader: the pool was too fast or too small for the
        // statistic to form, so the shape's own fastest stands.
        stage.winner = clause.fastestOverall;
        stage.note = Text("no run placed a leader at all, so the shape's own fastest, '%s', was "
                          "kept and the runs are reported above",
                          stage.winner.c_str());
    } else
    {
        stage.winner = candidates[bestIndex];
        stage.unanimous = (tiedAtBest == 1 && best == stage.runs);
        stage.plurality = (tiedAtBest == 1);

        if (stage.unanimous)
        {
            stage.note = Text("every one of the %d run(s) led with '%s', so the vote was unanimous",
                              stage.runs,
                              stage.winner.c_str());
        } else if (stage.plurality)
        {
            stage.note = Text("'%s' led %d of the %d run(s), more than any other candidate, so the "
                              "vote had a plurality and not a unanimous result",
                              stage.winner.c_str(),
                              best,
                              stage.runs);
        } else
        {
            stage.note = Text("the vote was split: %d entry(s) led %d run(s) each, so '%s' was "
                              "taken from among them by choice - any of them is equally good on "
                              "this evidence, and this one is named so that the shape ends with "
                              "one recommendation",
                              tiedAtBest,
                              best,
                              stage.winner.c_str());
        }
    }

    // The vote decides the name where it ran. The rule is that options a class cannot
    // separate are settled by which was fastest in most runs, so the row the vote named
    // is the default and the row this shape's own figures put first is the record of what
    // the shorter protocol said. The name fixed above stands only where no vote was
    // taken, which is what keeps a report from naming nothing.
    const bool voted = !stage.winner.empty();

    if (voted)
    {
        clause.recommended = stage.winner;
    }

    clause.defaultHow =
        voted ? (stage.unanimous ? DeviceProbeDefaultHow::kRefined
                                 : (stage.plurality ? DeviceProbeDefaultHow::kVote
                                                    : DeviceProbeDefaultHow::kChosenAmongEquals))
              : DeviceProbeDefaultHow::kChosenAmongEquals;

    clause.reason += Text(" The entries this shape did not place behind one leader were then "
                          "re-run on their own, %zu of them, %d run(s) at %d passes by %d rounds: "
                          "%s. The default is the entry the vote named, '%s', named with the way it "
                          "was reached - %s. This shape's own figures put '%s' first; where those "
                          "two differ, the difference is what says the shape's top entries cannot "
                          "be separated, and the vote is what settles a tie this shape's own rounds "
                          "could not.",
                          stage.pool.size(),
                          stage.runs,
                          stage.passes,
                          stage.rounds,
                          stage.note.c_str(),
                          clause.recommended.c_str(),
                          DeviceProbeDefaultHowName(clause.defaultHow),
                          clause.fastestOverall.empty() ? "no entry" : clause.fastestOverall.c_str());

    clause.confidence += Text(". The refinement stage re-ran the %zu entry(s) it was given, alone, "
                              "at %d passes by %d rounds, %d time(s), and %s",
                              stage.pool.size(),
                              stage.passes,
                              stage.rounds,
                              stage.runs,
                              stage.note.c_str());

    clause.refinement = std::move(stage);
}
/// The card-specific caveat, built from what the runtime reports about the card
/// that answered.
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

constexpr const char* DeviceProbeDefaultHowName(DeviceProbeDefaultHow how) noexcept {
    switch (how)
    {
        case DeviceProbeDefaultHow::kOrdered:
            return "ordered";
        case DeviceProbeDefaultHow::kOnlyEntry:
            return "only-entry";
        case DeviceProbeDefaultHow::kRefined:
            return "refined";
        case DeviceProbeDefaultHow::kVote:
            return "vote";
        case DeviceProbeDefaultHow::kChosenAmongEquals:
            return "chosen-among-equals";
        case DeviceProbeDefaultHow::kNone:
            return "none";
        case DeviceProbeDefaultHow::kCount:
            break;
    }

    // A value no arm above names, which the check below turns into a compile error.
    return nullptr;
}

namespace {

/// One enumerator of \c DeviceProbeDefaultHow that the switch above names no state for.
template <DeviceProbeDefaultHow How>
struct DefaultHowIsNamed {
    static_assert(DeviceProbeDefaultHowName(How) != nullptr,
                  "an enumerator of DeviceProbeDefaultHow names no state: name it in "
                  "DeviceProbeDefaultHowName (src/boys_cuda_probe.cpp)");
};

template struct EveryValue<DefaultHowIsNamed, DeviceProbeDefaultHow>;

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

    // Whether this build carries the fp16 rows is a fact about the build and not
    // about the device: the handle above names the float lane's tables, which are
    // resident whatever the fp16 seam is set to. BoysDeviceOptions() answers it
    // instead, and EnumerateEntries below takes that answer.

    // --- The workload, and the buffers it lives in ---------------------------
    DeviceProbeOptions clamped = options;
    clamped.count = std::max<std::size_t>(options.count, 1u);
    // The pair count has to be a genuinely larger count for the two readings to
    // fix the launch term between them: a factor of one would make the two
    // readings the same count and the extrapolation a division by zero, so it is
    // clamped to two rather than accepted as a degenerate protocol.
    clamped.countPairFactor = std::max(options.countPairFactor, 2);
    const std::size_t pairCount =
        clamped.count * static_cast<std::size_t>(clamped.countPairFactor);
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

    // The workload is built at the pair count, the wider of the two counts this
    // run reads every row at: the same buffers serve both readings, so they have
    // to be wide enough for the larger one and the arguments the smaller one uses
    // are its first slice.
    DeviceProbeOptions workloadOptions = clamped;
    workloadOptions.count = pairCount;
    const Workload work = BuildWorkload(workloadOptions);
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

    const std::size_t ladderValues = pairCount * (kMaxBoysOrder + 1);

    DeviceBuffer orders;
    DeviceBuffer args64;
    DeviceBuffer args32;
    DeviceBuffer args16;
    DeviceBuffer output;
    DeviceBuffer canarySink;

    if (BoysCudaProbeAlloc(&orders.pointer, pairCount * sizeof(int)) != 0 ||
        BoysCudaProbeAlloc(&args64.pointer, pairCount * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&args32.pointer, pairCount * sizeof(float)) != 0 ||
        BoysCudaProbeAlloc(&args16.pointer, pairCount * sizeof(std::uint16_t)) != 0 ||
        BoysCudaProbeAlloc(&output.pointer, ladderValues * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&canarySink.pointer, sizeof(unsigned long long)) != 0)
    {
        report.status = DeviceProbeStatus::kDeviceError;
        report.failure = Text("the workload's buffers could not be allocated on device %d",
                              options.device);
        return report;
    }

    if (BoysCudaProbeUpload(orders.pointer, work.n.data(), pairCount * sizeof(int)) != 0 ||
        BoysCudaProbeUpload(args64.pointer, work.x.data(), pairCount * sizeof(double)) != 0 ||
        BoysCudaProbeUpload(args32.pointer, work.xf.data(), pairCount * sizeof(float)) != 0 ||
        BoysCudaProbeUpload(args16.pointer, work.xh.data(), pairCount * sizeof(std::uint16_t)) != 0)
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

    // The same request at the pair count: every row is read at both counts, in
    // the same round, and only the count differs between the two requests. The
    // buffers above are sized for this one, so nothing has to be moved to take
    // the second reading.
    ProbeTimeRequest pairBase = base;
    pairBase.count = pairCount;

    report.measurements.resize(entries.size());

    // One row of this run's measurement table per entry of the option book this build serves, in
    // the book's own order, so a round visits them in that order.
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        const EntryInfo& info = entries[index];
        DeviceProbeMeasurement& measurement = report.measurements[index];

        measurement.name = info.name;
        measurement.precision = info.precision;
        measurement.entryIndex = index;
        measurement.entry = info.entry;
        measurement.shape = info.shape;
        measurement.question = QuestionName(info.question);
        measurement.route = info.inKernel ? "in-kernel" : "launched";
        measurement.launchedByLibrary = !info.inKernel;
        measurement.documentedBound = info.bound;
    }

    /// The entry every cost column is anchored to: the library's own first fp64
    /// row, which is the single-order double entry in the book's own order. It is
    /// resolved in this run's own entry list, because a run narrowed to a set that
    /// does not carry it still needs an anchor, and it falls back to the first fp64
    /// row the run carries and then to its first row.
    ///
    /// The anchor is what makes the costs comparable: every row's figure is this
    /// entry's own lower-quartile cost scaled by that row's ratio to it, taken
    /// inside the rounds.
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

    const std::size_t referenceAt = ReferenceIndex();

    // The pooled round table: one row per round, one column per row of the
    // measurement table, in cost per argument, plus the same shape for the
    // in-kernel rows' baselines. Every round of every pass is in here; no round is
    // dropped and no pass is excluded.
    std::vector<std::vector<double>> roundCost;
    std::vector<std::vector<double>> roundBaseline;
    // The same shape for the two readings every cell was formed from: one column
    // per row, one row per round, both taken in the round the cell was taken in.
    std::vector<std::vector<double>> roundAtCount;
    std::vector<std::vector<double>> roundAtPairCount;

    // The degree tables, made resident before any row is timed. One upload serves every entry of
    // this surface and the device holds it for the whole run, so it is one call and it is host work
    // outside every timed region. A device that will not hold them measures nothing: the refusal is
    // carried with the library's own answer, and every row stands as producing no figure rather
    // than as costing what tables nobody uploaded would cost.
    const bool tablesResident = TablesAreResident(handle);

    report.tablesResident = tablesResident;

    if (!tablesResident)
    {
        report.refusedTables =
            "the device would not hold the degree tables (BoysCuda::DeviceTables), so no row of "
            "this run measured anything: the classes stand with this reason and no figure";
    }

    // --- Warm-up. The first launch of a kernel pays one-time costs that are the
    // card's and the context's rather than the entry's, so nothing here is timed.
    // It uploads nothing itself: the tables are resident above, and the first
    // launch reading them is what pays for their being touched.
    bool warm = true;

    if (tablesResident)
    {
        for (const EntryInfo& info : entries)
        {
            // Both counts: the pair count is a different grid, and the first launch
            // on a grid pays the card's one-time cost for it as much as the first
            // launch on the run's count does.
            warm = TimeEntry(info, base, 1).ok && TimeEntry(info, pairBase, 1).ok && warm;
        }
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

    // --- The device-side launch floor, as the diagnostic it is ---------------
    //
    // A kernel launched the same way that does no Boys arithmetic at all gives the cheapest launch
    // this route can make, and the report states it per launch and as a fraction of the fastest
    // row, so a run whose workload was too small says so instead of quietly ranking the launcher.
    //
    // **It is not what the figures have taken out of them.** An entry's kernel needs registers and
    // an occupancy ramp an empty kernel never pays, so the floor is a lower bound on an entry's own
    // launch cost and subtracting it would take out part of the term. What comes out of a figure is
    // the entry's own launch term, fixed by that entry's readings at the two counts below.
    //
    // So a floor that cannot be read costs this run its diagnostic and not its figures: the
    // measurement continues and the control says the floor was not established.
    bool floorTimed = false;

    {
        ProbeTimeRequest floorRequest = base;
        floorRequest.what = static_cast<int>(ProbeWhat::kFloor);
        floorRequest.reps = clamped.repetitions;

        double floorMs = 0.0;

        if (TimeRegion(floorRequest, floorMs))
        {
            report.control.nsPerLaunchFloor =
                floorMs * 1.0e6 / static_cast<double>(clamped.repetitions);
        } else
        {
            report.control.nsPerLaunchFloor = 0.0;
        }

        // A floor of zero is a floor this run did not establish, whatever the
        // timer said: it is the same reading the report prints as "could not be
        // timed", and it is read here from the one value both sites print so that
        // no line of this report can call the floor timed while another calls it
        // missing. A region whose device time rounds away at this repetition count
        // reads as zero and is reported as no reading rather than as a free launch.
        floorTimed = report.control.nsPerLaunchFloor > 0.0;
    }

    // --- Passes --------------------------------------------------------------
    //
    // Every pass is run and every pass is used; what the canary said beside one is reported with it
    // and decides nothing.
    //
    // The visit order is shuffled once per round from the run's own seed. A fixed order would put
    // the same entry first in every round, and whatever a position in the round is worth - the first
    // launch touching a table the others then find warm - would enter that entry's ratio as though
    // it were the entry's own cost. The seed keeps a run reproducible.
    //
    // Every row of a round is timed with the same resident tables, which is what makes the two rows
    // of any comparison this report makes a comparison of one arithmetic.
    std::mt19937_64 shuffle(clamped.seed);
    std::vector<std::size_t> visit(report.measurements.size());

    for (std::size_t slot = 0; slot < visit.size(); ++slot)
    {
        visit[slot] = slot;
    }

    // How many rounds of the run a launched row's own two readings left no
    // positive launch term in, so that no figure could be extrapolated for it. A
    // row whose every round did is a row this workload cannot measure at these
    // counts, and the count is what tells that apart from a row that was simply
    // never timed.
    std::vector<int> unresolvedRounds(report.measurements.size(), 0);

    // How many rounds of the run both counts were actually read in. It is what
    // tells a row the two readings could not be extrapolated from apart from a
    // row that was never read at all: the first has rounds to its name and no
    // figure, the second has neither.
    std::vector<int> readRounds(report.measurements.size(), 0);

    // The ratio between the two counts every row is read at, and the factor the
    // two readings are combined with: with a fixed cost L per launch, a reading f
    // at C and f' at r*C give f = a + L/C and f' = a + L/(r*C), so the
    // count-independent a is (r*f' - f)/(r - 1).
    const double pairRatio = static_cast<double>(pairCount) / static_cast<double>(clamped.count);

    const int canarySamples = std::max(3, clamped.rounds + 1);

    // A run whose tables never went resident has no round to time: every pass would launch into
    // tables this device does not hold, and a figure from that is a figure for an arithmetic nobody
    // offered it. The refusal is reported and the rows stand unmeasured.
    const int timedPasses = tablesResident ? clamped.passes : 0;

    for (int pass = 0; pass < timedPasses; ++pass)
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
            // One row of this pass's own round table, in cost per argument. A cell
            // left infinite is a round in which the entry could not be timed, and it
            // is left out of the ratios below rather than counted as a fast one.
            std::vector<double> row(report.measurements.size(), std::numeric_limits<double>::infinity());
            std::vector<double> baseline(row);
            // The row's own two readings, kept per round as well as the figure
            // formed from them: the report prints them beside the figure, and a
            // reader checking the extrapolation needs them at the same rounds the
            // figure rests on rather than as one summary number.
            std::vector<double> firstReading(row);
            std::vector<double> secondReading(row);

            // Which half of an in-kernel pair goes first alternates by round, so
            // the fixed cost of a timed region — the event pair, the first launch's
            // cold start, the host's submission — lands on each half in half the
            // rounds instead of always on one.
            const bool baselineFirst = (round % 2) == 1;

            // One timed entry reduced to its own reading at the count it was taken
            // at, in cost per argument: a launched entry is its region, an in-kernel
            // entry the difference between the region that held the call and the one
            // that did not, formed inside the round both halves were timed in. Both
            // halves are the same caller kernel launched the same number of times on
            // the same grid, so the per-launch cost cancels in the difference
            // exactly, whatever the count.
            const auto Reading = [&](const TimedEntry& timed, std::size_t count) {
                const double with = PerArgument(timed.withMs, clamped.repetitions, count);

                if (!timed.subtracted)
                {
                    return with;
                }

                const double without = PerArgument(timed.withoutMs, clamped.repetitions, count);

                return with - without;
            };

            // The visit order is drawn here, inside the round: the set walked is every row of
            // this run, and which of them is timed first is what the shuffle decides.
            std::shuffle(visit.begin(), visit.end(), shuffle);

            for (std::size_t local = 0; local < visit.size(); ++local)
            {
                const std::size_t index = visit[local];
                const EntryInfo& info = entries[index];

                // The row's own two readings, in this round and at both counts,
                // timed adjacently so that both sit under one clock and one workload
                // state. One reading alone cannot say how much of it is the launch;
                // the two together can, without an empty kernel standing in for what
                // this entry's kernel costs to start.
                const TimedEntry atCount =
                    TimeEntry(info, base, clamped.repetitions, baselineFirst);
                const TimedEntry atPair =
                    TimeEntry(info, pairBase, clamped.repetitions, baselineFirst);

                if (!atCount.ok || !atPair.ok)
                {
                    // A count of the pair that could not be timed leaves the cell
                    // infinite and uncounted: the row was not measured in this
                    // round, which is a different fact from a row the two
                    // readings could not be extrapolated from.
                    continue;
                }

                const double first = Reading(atCount, clamped.count);
                const double second = Reading(atPair, pairCount);

                firstReading[index] = first;
                secondReading[index] = second;
                ++readRounds[index];

                // The cell is the count-independent figure the row's own two
                // readings extrapolate to. What it removes is the term that repeats
                // with the launch rather than with the call, which is paid once per
                // launch and falls as 1/count.
                const double extrapolated = (pairRatio * second - first) / (pairRatio - 1.0);

                if (atCount.subtracted)
                {
                    // An in-kernel row: the difference came out at or below zero
                    // in this round, which is the entry's arithmetic inside the
                    // noise of its own baseline rather than a negative cost. The
                    // cell is held at zero and the fold reads that as the row's
                    // subtraction not having resolved at this workload.
                    row[index] = std::max(0.0, extrapolated);
                    baseline[index] =
                        PerArgument(atCount.withoutMs, clamped.repetitions, clamped.count);
                    continue;
                }

                if (!(extrapolated > 0.0))
                {
                    // A launched row whose two readings left no positive launch
                    // term to take out: the pair-count reading was not the
                    // cheaper of the two, so the asymptote is at or below zero
                    // and this round produced no figure for the row. The cell is
                    // left infinite and counted, and a row whose every round did
                    // this is set aside by name below rather than printed at a
                    // zero that reads as a free call or at a negative one.
                    ++unresolvedRounds[index];
                    continue;
                }

                row[index] = extrapolated;
            }

            passRounds.push_back(row);
            roundCost.push_back(std::move(row));
            roundBaseline.push_back(std::move(baseline));
            roundAtCount.push_back(std::move(firstReading));
            roundAtPairCount.push_back(std::move(secondReading));

            takeCanary();
        }

        const auto finish = std::chrono::steady_clock::now();
        record.seconds = std::chrono::duration<double>(finish - start).count();

        while (static_cast<int>(canaryMs.size()) < canarySamples && !canaryMs.empty())
        {
            takeCanary();
        }

        // A pass in which no read succeeded took no reading at all: its figures are
        // zero because nothing was read, and a zero spread would otherwise be a
        // machine reported as still by a reading that was never taken.
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

        // The pass's own paired spread: the spread the ordering is made in, taken at
        // its widest over the run's rows, every one of which was timed against the same
        // anchor column.
        record.pairedSpread = PassPairedSpread(passRounds, referenceAt);
        report.passes.push_back(record);
    }

    report.pairedRounds = static_cast<int>(roundCost.size());

    // --- Fold the rounds into figures ---------------------------------------
    //
    // Every figure below is an aggregate of ratios taken inside a round of the table above, which
    // is what makes it a paired comparison: two cells of one round were timed under one clock. The
    // statistic a pair is placed by is a lower quartile - see QuantileOf and kStatisticQuantile -
    // and a row's ratio to the reference is read at the middle (\c kFigureQuantile), so the cost is
    // the reference entry's own lower-quartile figure scaled by that ratio.
    //
    // Every row is folded against the one anchor: the reference entry's own figure, which every row
    // of the run was timed beside, so a ratio is always between two rows measured under one
    // arithmetic.
    if (!roundCost.empty())
    {
        const std::size_t reference = referenceAt;

        std::vector<double> referenceCost;

        for (const std::vector<double>& row : roundCost)
        {
            referenceCost.push_back(row[reference]);
        }

        const double referenceNs = QuantileOf(referenceCost, kStatisticQuantile);

        report.referenceEntry = entries[referenceAt].name;
        report.referenceNsPerArgument = referenceNs;

        const std::size_t half = roundCost.size() / 2;

        for (std::size_t index = 0; index < report.measurements.size(); ++index)
        {
            DeviceProbeMeasurement& measurement = report.measurements[index];
            measurement.referenceNsPerArgument = referenceNs;

            std::vector<double> ratios;
            std::vector<double> early;
            std::vector<double> late;
            std::vector<double> baselines;
            std::vector<double> firsts;
            std::vector<double> seconds;
            double peak = std::numeric_limits<double>::infinity();

            for (std::size_t round = 0; round < roundCost.size(); ++round)
            {
                const double anchor = roundCost[round][reference];
                const double cost = roundCost[round][index];

                if (!std::isfinite(anchor) || anchor <= 0.0 || !std::isfinite(cost))
                {
                    continue;
                }

                ratios.push_back(cost / anchor);

                // The two readings the cell was formed from, on the same rounds
                // and against the same anchor as the cell itself: the report
                // prints them beside the figure so that what the extrapolation
                // took out is the difference between three columns of one table
                // rather than a number a reader has to take on trust.
                if (std::isfinite(roundAtCount[round][index]) &&
                    std::isfinite(roundAtPairCount[round][index]))
                {
                    firsts.push_back(roundAtCount[round][index] / anchor);
                    seconds.push_back(roundAtPairCount[round][index] / anchor);
                }

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

                if (std::isfinite(roundBaseline[round][index]))
                {
                    baselines.push_back(roundBaseline[round][index]);
                }
            }

            if (ratios.size() < 2 || referenceNs <= 0.0)
            {
                // A launched row whose every round left no positive launch term has
                // no figure at all, and is named as that wherever the report lists
                // what a shape could not place: a row that was never timed and one
                // the two readings could not be extrapolated from are different
                // facts about the run, and only the second is answered by raising
                // the counts.
                measurement.extrapolationUnresolved =
                    !entries[index].inKernel && readRounds[index] > 0 &&
                    unresolvedRounds[index] == readRounds[index];
                continue;
            }

            measurement.measured = true;
            measurement.rounds = static_cast<int>(roundCost.size());
            measurement.ratioToReference = QuantileOf(ratios, kFigureQuantile);
            measurement.ratioLo = QuantileOf(ratios, kStatisticQuantile);
            measurement.ratioHi = QuantileOf(ratios, 1.0 - kStatisticQuantile);
            measurement.nsPerArgument = referenceNs * measurement.ratioToReference;
            measurement.nsPerArgumentMax = referenceNs * measurement.ratioHi;
            measurement.nsPerArgumentAtCount =
                firsts.empty() ? 0.0 : referenceNs * QuantileOf(firsts, kStatisticQuantile);
            measurement.nsPerArgumentAtPairCount =
                seconds.empty() ? 0.0 : referenceNs * QuantileOf(seconds, kStatisticQuantile);
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
                !entries[index].inKernel || measurement.nsPerArgument > 0.0;

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

    // --- What the timed work actually produced. One read-back per row, outside
    // every clock, so the report can show that the launches ran and ran on the
    // intended arguments.
    {
        std::vector<double> host(ladderValues, 0.0);

        // A device that does not hold the tables reads nothing back: the sums stay
        // zero rather than carrying the values of tables it never held.
        if (tablesResident)
        {
            for (std::size_t index = 0; index < report.measurements.size(); ++index)
            {
                const EntryInfo& info = entries[index];

                // One more launch per row, and its result read back. It is not
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

                report.measurements[index].checkedSum = sum;
            }
        }

        (void)BoysCudaProbeSynchronize();
    }

    // --- The classes, one per precision and question shape -------------------
    //
    // After the controls, and run to a fixed point: every row a class would recommend goes through
    // the repetition control first, and a row that does not agree sets the class back to the next
    // one.
    //
    // The class set comes from the rows this run was asked for rather than from the rows that
    // happened to measure, so a class whose every row failed to measure is still reported with its
    // reason. It is walked over the measurement table, which is the option table this build serves,
    // in the book's own order.
    /// What one class's conclusion is a function of, and the conclusion it had.
    ///
    /// A class is concluded from the rows it holds and from what this run's checks
    /// have said about them, and from nothing else, so a class whose rows and flags
    /// are unchanged reaches the same conclusion. The convergence loop below names
    /// one row at a time, and without this it would re-run every class's refinement
    /// — the expensive part of a conclusion — once per named row.
    struct ClassMemo {
        std::string key;
        std::vector<std::size_t> live;
        std::vector<std::uint32_t> flags;
        DeviceProbeRanking ranking;
    };

    std::vector<ClassMemo> memo;

    const auto ConcludeClasses = [&]() {
        report.classes.clear();

        for (const DeviceProbeMeasurement& clause : report.measurements)
        {
            std::size_t classAt = report.classes.size();

            for (std::size_t c = 0; c < report.classes.size(); ++c)
            {
                if (report.classes[c].precision == clause.precision &&
                    report.classes[c].question == clause.question)
                {
                    classAt = c;
                }
            }

            if (classAt != report.classes.size())
            {
                continue;
            }

            DeviceProbeClass fresh;
            fresh.precision = clause.precision;
            fresh.question = clause.question;
            fresh.asked = QuestionAsked(entries[clause.entryIndex].question);
            report.classes.push_back(fresh);

            // Whether this class holds one bound or two is a fact about its rows,
            // read off them: the bound each row documents for itself.
            bool oneBound = true;
            double first = 0.0;
            bool haveFirst = false;
            std::size_t memberCount = 0;

            for (const DeviceProbeMeasurement& other : report.measurements)
            {
                if (other.precision != clause.precision || other.question != clause.question)
                {
                    continue;
                }

                ++memberCount;

                if (!haveFirst)
                {
                    first = other.documentedBound;
                    haveFirst = true;
                } else if (other.documentedBound != first)
                {
                    oneBound = false;
                }
            }

            report.classes[classAt].note =
                ClassNote(clause, report.classes[classAt].asked, memberCount, oneBound);

            DeviceProbeRanking ranking;
            ranking.question = clause.question;
            ranking.asked = report.classes[classAt].asked;

            std::vector<DeviceProbeMeasurement*> live;
            std::vector<std::size_t> columns;

            for (std::size_t index = 0; index < report.measurements.size(); ++index)
            {
                DeviceProbeMeasurement& measurement = report.measurements[index];

                if (measurement.measured && measurement.precision == clause.precision &&
                    measurement.question == clause.question)
                {
                    live.push_back(&measurement);
                    columns.push_back(index);
                }
            }

            // This class's signature: the rows it holds, in the table's own order,
            // and what the run's checks have said about each.
            std::vector<std::uint32_t> flags;

            for (const DeviceProbeMeasurement* row : live)
            {
                flags.push_back(static_cast<std::uint32_t>((row->measured ? 1u : 0u) |
                                                           (row->subtractionResolved ? 2u : 0u) |
                                                           (row->repetitionChecked ? 4u : 0u) |
                                                           (row->repetitionAgrees ? 8u : 0u)));
            }

            const std::string memoKey = clause.precision + "/" + clause.question;

            if (classAt < memo.size() && memo[classAt].key == memoKey &&
                memo[classAt].live == columns && memo[classAt].flags == flags)
            {
                report.classes[classAt].ranking = memo[classAt].ranking;
                report.hasDefault =
                    report.hasDefault || !report.classes[classAt].ranking.recommended.empty();
                continue;
            }

            Conclude(ranking, live, columns, roundCost, report.pairedRounds);

            // A class of a run whose tables the device would not hold measured
            // nothing, and that is why its rows produced no figure: the residency
            // failure is put in front of whatever the conclusion wrote, because a
            // reader who expected the entry here would otherwise take its absence for
            // a fact about the arithmetic rather than about this device and this run.
            if (!tablesResident)
            {
                ranking.reason = Text("the device would not hold the degree tables, so no entry of "
                                      "this class was timed. %s",
                                      ranking.reason.c_str());
                ranking.confidence = Text("CANNOT DETERMINE: the degree tables are what this device "
                                          "could not hold");
            }

            // A launched row the two readings could not be extrapolated from was
            // not placed and is not silently absent either: a reader who expected
            // it in the table would take its absence for a build fact. It is named
            // here with what its own readings did, beside the rows Conclude set
            // aside for its own reasons.
            for (const DeviceProbeMeasurement& measurement : report.measurements)
            {
                if (!measurement.extrapolationUnresolved ||
                    measurement.precision != clause.precision ||
                    measurement.question != clause.question)
                {
                    continue;
                }

                ranking.notOrdered.push_back(Text(
                    "%s (launched): read at %d and at %d arguments in every round, its two "
                    "readings left no positive launch term to take out - the reading at the larger "
                    "count did not come out the cheaper of the two - so the figure it extrapolates "
                    "to came out at or below zero and this row has no figure rather than one this "
                    "run cannot stand behind. Raising both counts is what answers it: the launch "
                    "term has to be a measurable part of the two readings before it can be taken "
                    "out of either",
                    measurement.name.c_str(),
                    static_cast<int>(clamped.count),
                    static_cast<int>(pairCount)));
            }

            // A class whose own rounds could not place a rival behind the leader is
            // not left without an answer: the entries they could not separate are
            // re-run on their own at a longer protocol and voted on, and the entry
            // the vote names is the class's recommendation with the way it was
            // reached recorded beside it.
            if (!ranking.tiedEntries.empty())
            {
                RefineShape(ranking, live, entries, base, pairBase, clamped, handle);
            }

            report.hasDefault = report.hasDefault || !ranking.recommended.empty();

            // The conclusion and its signature, for the next sweep to reuse if
            // nothing the conclusion rests on has moved.
            ClassMemo reached;
            reached.key = memoKey;
            reached.live = columns;
            reached.flags = std::move(flags);
            reached.ranking = ranking;

            if (classAt < memo.size())
            {
                memo[classAt] = std::move(reached);
            } else
            {
                memo.push_back(std::move(reached));
            }

            report.classes[classAt].ranking = std::move(ranking);
        }
    };

    // A row is judged against what this run's own instrument and that row's own
    // passes measured, never against the widest threshold some other class
    // needed: a class whose rival wandered is a fact about that rival.
    const auto RowResolution = [&](const DeviceProbeMeasurement* row) {
        return ShapeResolution(report.measurements, roundCost, row->precision, row->question);
    };

    /// Puts one row through the repetition control and records the outcome on the
    /// row, so that the class conclusions can see it. The degree tables are made
    /// resident first: the control re-times the row at two repetition counts, and
    /// a row timed under no tables at all would be a check of a different
    /// arithmetic than the figure it is checking.
    const auto CheckRow = [&](DeviceProbeMeasurement& row, DeviceProbeRepetitionControl* printed) {
        const EntryInfo& info = entries[row.entryIndex];

        if (!tablesResident)
        {
            row.repetitionChecked = true;
            row.repetitionAgrees = false;
            row.repetitionNote = Text("the degree tables could not be made resident for the "
                                      "repetition control, so this row was not checked");
            return true;
        }

        DeviceProbeRepetitionControl outcome;

        // The floor belongs to the run, not to this control: the run timed it once,
        // under its own protocol, before any row was put through a check, and the
        // assignment below replaces the whole struct. Carrying it in here is what
        // keeps one report from saying a floor could not be timed while a line of it
        // prints the floor as a measured zero - the zero this struct's own default
        // would put in the note.
        if (printed != nullptr)
        {
            outcome.nsPerLaunchFloor = printed->nsPerLaunchFloor;
        }

        const bool timed = RunRepetitionControl(info,
                                                row,
                                                base,
                                                pairBase,
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
    // itself. The launched route carries the device-side floor beside its
    // verdict as well — a kernel launched the same way that does no Boys
    // arithmetic at all — because that route's figure has a launch inside it by
    // construction; the floor says how much of the row's cost a bare launch
    // accounts for, and it is a diagnostic rather than what was taken out.
    {
        DeviceProbeMeasurement* fastestLaunched = nullptr;
        DeviceProbeMeasurement* fastestSubtracted = nullptr;

        for (DeviceProbeMeasurement& measurement : report.measurements)
        {
            // The two printed controls are taken over the row a default is read
            // from: that is the row whose figure a reader is most likely to quote,
            // and running the check on some other row instead would leave the
            // default's own figure uncontrolled. Every other row a class names is
            // checked in its own turn by the fixed-point loop below, and its class
            // says whether it agreed.
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
            // The floor is a diagnostic read beside the row the control put through
            // its two counts, taken under the protocol the figures were taken under.
            // A run that could not read it says so here rather than losing a
            // measurement that does not rest on it.
            if (report.control.entry.empty())
            {
                // RunRepetitionControl already said why in the note.
            } else
            {
                const double perCall = clamped.count * fastestLaunched->nsPerArgument;
                const double share = perCall > 0.0 && floorTimed
                                         ? report.control.nsPerLaunchFloor / perCall
                                         : 0.0;

                // The floor is the per-launch cost of a kernel that does no
                // arithmetic, so it holds the device's launch and the host's
                // submission of it together, and on a platform whose submission is
                // expensive that is the larger part. **It is not the entry's own
                // launch cost and not what came out of the figures**: an entry's
                // kernel needs registers and an occupancy ramp the empty one never
                // pays, so this is the cheapest launch the route can make, and a row
                // whose whole figure is at or below it is launch-bound.
                const std::string floorClause =
                    !floorTimed
                        ? Text("the device-side launch floor - a kernel launched the same way that "
                               "does no Boys arithmetic - could not be timed at %d launches over "
                               "%d arguments, so this run states no floor and no fraction of a "
                               "row's cost against it. The figures above do not rest on it: each "
                               "row's launch term was taken out from that row's own readings at %d "
                               "and at %d arguments",
                               clamped.repetitions,
                               static_cast<int>(clamped.count),
                               static_cast<int>(clamped.count),
                               static_cast<int>(pairCount))
                    : share >= 1.0
                        ? Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, which is %.0f%% of that "
                               "row's own cost per call. THAT ROW IS LAUNCH-BOUND: the cheapest "
                               "launch this route can make costs as much per call as the row's "
                               "whole figure, and an entry's own kernel costs more to start than "
                               "this empty one does, so the figure is a small difference between "
                               "two readings that starting kernels dominate. Raising both argument "
                               "counts until the floor is a small fraction of the figure is what "
                               "this needs",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share)
                        : Text("the floor - a kernel launched the same way that does no Boys "
                               "arithmetic - costs %.3f us per launch, %.2f%% of that row's own "
                               "cost per call. It is a diagnostic and not what came out of the "
                               "figures: an entry's own kernel costs more to start than this empty "
                               "one does, and each row's launch term was taken out from that row's "
                               "own readings at %d and at %d arguments (the two columns beside "
                               "each figure), so this fraction is a lower bound on the share of "
                               "that row's cost a bare launch accounts for",
                               report.control.nsPerLaunchFloor / 1000.0,
                               100.0 * share,
                               static_cast<int>(clamped.count),
                               static_cast<int>(pairCount));

                report.control.note += "; ";
                report.control.note += floorClause;
                report.control.note += ".";
            }
        } else
        {
            report.control.note = "no launched entry produced a "
                                  "figure, so there was nothing to put the launched route's "
                                  "repetition control through";
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
                "no in-kernel row of the full-accuracy block resolved a cost above its own "
                "baseline, so there was nothing to put the subtraction route's repetition "
                "control through";
        }
    }

    // --- The conclusions, to a fixed point -----------------------------------
    //
    // Every row the report names as a class's fastest or as its recommendation is
    // put through the repetition control before the report ships, and a row that
    // does not agree is set aside and the class falls to the next. Looped,
    // because setting a row aside can name a new leader, which then has to be
    // checked in its turn. Bounded by the number of rows: each round checks at
    // least one row not checked before, and the loop leaves when there is none.
    //
    // The row is resolved by its own index in the option table and not by its
    // name: the run's measurement table holds one row per entry, and a class
    // names the row it concluded on.
    for (;;)
    {
        ConcludeClasses();

        DeviceProbeMeasurement* unchecked = nullptr;

        for (const DeviceProbeClass& clause : report.classes)
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            const std::string& named =
                !ranking.recommended.empty() ? ranking.recommended : ranking.fastestOverall;

            if (named.empty())
            {
                continue;
            }

            for (DeviceProbeMeasurement& measurement : report.measurements)
            {
                if (measurement.name == named && !measurement.repetitionChecked)
                {
                    unchecked = &measurement;
                }
            }
        }

        if (unchecked == nullptr)
        {
            break;
        }

        if (!CheckRow(*unchecked, nullptr))
        {
            // The tables could not be made resident, or the row is not one this run
            // measured; either way it is not checked here. Its own line in the
            // class says which, and the loop must not spin on it.
            unchecked->repetitionChecked = true;
        }
    }

    report.status = DeviceProbeStatus::kSuccess;
    return report;
}

// --------------------------------------------------------------------------- The text.
// ---------------------------------------------------------------------------

/// The place this run's grid carries for one member of the space, or nullptr where it
/// carries none.
///
/// The member is identified by its entry, which is what one row of this run's table is:
/// a run measures one entry once, so a name and an entry are the same place.
const DeviceProbeMeasurement* GridPlaceOf(const DeviceProbeReport& report, DeviceEntry entry) {
    for (const DeviceProbeMeasurement& place : report.measurements)
    {
        if (place.entry == entry)
        {
            return &place;
        }
    }

    return nullptr;
}

DeviceOptionClosure DeviceOptionSpaceClosure(const DeviceProbeReport& report) noexcept {
    const std::span<const DeviceOptionInfo> space = BoysDeviceOptions();

    DeviceOptionClosure closure;
    closure.rows = space.size();
    closure.total = closure.rows;

    // One flag per (precision, question) the space admits: a table and not a list of
    // keys, so that the third reading of the space below allocates nothing. A class the
    // space admits and the report does not carry is a shape of this surface that nothing
    // reports on.
    constexpr std::size_t kPrecisions = static_cast<std::size_t>(DeviceOptionPrecision::kCount);
    constexpr std::size_t kQuestions = static_cast<std::size_t>(DeviceOptionQuestion::kCount);
    std::array<bool, kPrecisions * kQuestions> classes = {};

    // Whether the run reached the device's own tables at all. A run that did not is one
    // that never presented a member to a device, and a member it never presented is in no
    // state rather than in the state nearest to one.
    const bool reachedTheGrid = report.tablesResident || !report.refusedTables.empty() ||
                                !report.measurements.empty();

    for (const DeviceOptionInfo& row : space)
    {
        // Whether this run's own request named the row. A request that names nothing
        // asks for the whole space; one that names entries asks for those.
        const bool asked = report.options.only.empty() ||
                           std::find(report.options.only.begin(),
                                     report.options.only.end(),
                                     std::string(row.name)) != report.options.only.end();

        // The order is the space's own: a row the build does not serve is refused with
        // the library's reason and owed, whatever the run did with the others.
        if (!row.built)
        {
            ++closure.refusedAndOwed;
            continue;
        }

        if (asked)
        {
            closure.gridPlacesOwed += 1;

            // A member the space serves and this request named is a member of one class
            // of the report, keyed on the precision and the question: this is the space's
            // own count of the classes a run of this request has to print, and it is read
            // off the rows rather than off the report it is held to.
            const std::size_t slot = static_cast<std::size_t>(row.precision) * kQuestions +
                                     static_cast<std::size_t>(row.question);
            classes[slot] = true;
        }

        // The card would not hold the degree tables: the row was never presented to it,
        // and that is this card's answer rather than the row's absence.
        if (!report.tablesResident && !report.refusedTables.empty())
        {
            ++closure.notRunnable;
            continue;
        }

        if (!asked && reachedTheGrid)
        {
            ++closure.notAsked;
            continue;
        }

        const DeviceProbeMeasurement* place = GridPlaceOf(report, row.entry);

        if (place != nullptr)
        {
            (place->measured ? closure.measured : closure.offeredNoFigure) += 1;
            continue;
        }

        ++closure.unaccounted;
    }

    closure.states = closure.measured + closure.offeredNoFigure + closure.refusedAndOwed +
                     closure.notRunnable + closure.notAsked;
    closure.gridPlaces = report.measurements.size();

    for (const bool admitted : classes)
    {
        closure.classesAdmitted += admitted ? 1u : 0u;
    }

    closure.classesPrinted = report.classes.size();
    closure.closed = closure.unaccounted == 0 && closure.states == closure.total &&
                     closure.gridPlaces == closure.gridPlacesOwed &&
                     closure.classesPrinted == closure.classesAdmitted &&
                     report.status == DeviceProbeStatus::kSuccess;
    return closure;
}

/// The rows this build refuses, grouped by the library's own reason: one entry per
/// distinct reason, with the rows it refuses, so the closure block states a reason once
/// and every row it covers.
std::vector<std::pair<std::string, std::vector<std::string>>> RefusalsByReason() {
    std::vector<std::pair<std::string, std::vector<std::string>>> groups;

    for (const DeviceOptionInfo& row : BoysDeviceOptions())
    {
        if (row.built)
        {
            continue;
        }

        const std::string reason =
            row.refusedBecause != nullptr ? std::string(row.refusedBecause)
                                          : std::string("this build serves it and the row names no "
                                                        "reason");

        bool placed = false;

        for (auto& group : groups)
        {
            if (group.first == reason)
            {
                group.second.push_back(row.name);
                placed = true;
                break;
            }
        }

        if (!placed)
        {
            groups.push_back({reason, {row.name}});
        }
    }

    return groups;
}

/// A comma-separated list of names, wrapped into lines no wider than the report's own
/// column so that a list of every row one reason refuses is a paragraph and not a line
/// off the edge of a terminal.
std::string WrappedNames(const std::vector<std::string>& names, const std::string& indent) {
    constexpr std::size_t kWidth = 96;
    std::string text;
    std::size_t lineWidth = 0;

    for (const std::string& name : names)
    {
        const std::string piece = text.empty() ? name : ", " + name;

        if (!text.empty() && lineWidth + piece.size() > kWidth)
        {
            text += ",\n" + indent + name;
            lineWidth = indent.size() + name.size();
            continue;
        }

        text += piece;
        lineWidth += piece.size();
    }

    return text;
}

/// The option space's closure, printed: the space, one count per state a member of it
/// can be in, the arithmetic over those counts, and the verdict.
///
/// It is the last block of the report on every path, a run that measured nothing
/// included: what a run did with the space is a fact about the run, and a run that
/// reached none of it says so here rather than leaving a reader to infer it from the
/// absence of figures. `DeviceOptionClosure` is what the two readings of one total are,
/// and the counts below are printed with the tables they were read from.
void AppendOptionClosure(std::string& text, const DeviceProbeReport& report) {
    const DeviceOptionClosure closure = DeviceOptionSpaceClosure(report);
    const std::vector<std::pair<std::string, std::vector<std::string>>> refusals =
        RefusalsByReason();

    text += "\n\nthe closure - the space above counted, every member of it in one state of this "
            "run and no\n  member in two, so that a member the report does not account for is "
            "visible as a missing\n  number rather than as an absence:\n";
    text += Text("  the space: %zu row(s) of this library's own option table (BoysDeviceOptions,\n"
                 "  include/boys/boys_cuda_options.hpp), one member per row: %zu member(s). The "
                 "count is\n  the library's own and is not listed here.\n",
                 closure.rows,
                 closure.total);

    text += Text("\n  MEMBERS: %zu of %zu member(s) of the option space are measured on this card "
                 "and\n                published\n",
                 closure.measured,
                 closure.total);
    text += Text("                %zu refused with the library's own reason and owed\n",
                 closure.refusedAndOwed);

    for (const auto& group : refusals)
    {
        text += Text("                    %s: %s\n",
                     group.first.c_str(),
                     WrappedNames(group.second, "                      ").c_str());
    }

    text += Text("                %zu not runnable on this card, counted apart and not against the "
                 "library: the\n                device would not hold the degree tables, so no row "
                 "was presented to it. The\n                library's own answer to that upload is "
                 "printed above\n",
                 closure.notRunnable);
    text += Text("                %zu offered and this run carried a place for, producing no "
                 "figure\n",
                 closure.offeredNoFigure);
    text += Text("                %zu not asked for by this run's request: the space is stated "
                 "whole and this\n                run measured the entries its request named\n",
                 closure.notAsked);
    text += Text("  the arithmetic: %zu + %zu + %zu + %zu + %zu = %zu\n",
                 closure.measured,
                 closure.refusedAndOwed,
                 closure.notRunnable,
                 closure.offeredNoFigure,
                 closure.notAsked,
                 closure.states);
    text += Text("                 the space's own total: %zu member(s) = the %zu above + %zu in "
                 "no state\n",
                 closure.total,
                 closure.states,
                 closure.unaccounted);
    text += Text("                 the run's own grid: %zu place(s), against the %zu a run of this "
                 "request owes\n                 the space (one place per row it serves and this "
                 "request named)\n",
                 closure.gridPlaces,
                 closure.gridPlacesOwed);
    text += Text("                 the classes the space admits: %zu, one per (precision, "
                 "question) the\n                 rows this request named fall into; this report "
                 "carries %zu class(es)\n",
                 closure.classesAdmitted,
                 closure.classesPrinted);

    if (closure.closed)
    {
        text += Text("  the verdict: PASS - every one of the space's %zu member(s) is in one state "
                     "above and\n                none is in two\n",
                     closure.total);
        return;
    }

    text += "  the verdict: FAIL - the closure does not hold, and this run's exit status says "
            "so:\n";

    if (report.status != DeviceProbeStatus::kSuccess)
    {
        text += Text("                the run did not succeed: %s\n"
                     "                The members above are the space this build serves and this "
                     "run did not present\n                to a device\n",
                     StatusName(report.status));
    }

    if (closure.states != closure.total)
    {
        text += Text("                the states sum to %zu and the space has %zu member(s)\n",
                     closure.states,
                     closure.total);
    }

    if (closure.gridPlaces != closure.gridPlacesOwed)
    {
        text += Text("                the run's own grid carries %zu place(s) where a run of this "
                     "request\n                owes %zu\n",
                     closure.gridPlaces,
                     closure.gridPlacesOwed);
    }

    if (closure.classesPrinted != closure.classesAdmitted)
    {
        text += Text("                the report carries %zu class(es) where the space admits %zu "
                     "over the\n                rows this request named\n",
                     closure.classesPrinted,
                     closure.classesAdmitted);
    }

    if (closure.unaccounted > 0)
    {
        text += Text("                %zu member(s) are in no state above: rows of the library's "
                     "own report\n                that no part of this run stands behind\n",
                     closure.unaccounted);
    }
}

/// The library's report of its device option space, row by row, with the row this probe carries for
/// each beside it.
///
/// This block is the probe's coverage statement and not a courtesy: the probe's option table is a
/// projection of the library's, so every row listed here is either a row measured above or a
/// refusal with the library's reason. `probeRows` is what the run actually carried - the
/// measurement table's own names, in its own order - so the comparison is against the report this
/// run printed and not against a second reading of the same source.
/// /// \\param probeRows the entries the run carried a figure for
/// /// \\param emptyRows the offered cells that produced no figure
void AppendOptionSpace(std::string& text,
                       const std::vector<std::string>& probeRows,
                       std::size_t emptyRows) {
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

    text += Text("  One member per row of those tables, and %zu of the %zu row(s) this build serves "
                 "produced no\n  figure here.\n",
                 emptyRows,
                 served);

    text += "  report row                 probe row                  group     precision  shape       question    axes                                                                       lane        bound     documented form\n";

    for (const DeviceOptionInfo& option : space)
    {
        // Whether this run carried a row for the option, by name: the run's measurement
        // table holds one row per entry it measured, so a name is a place.
        const std::string carried =
            std::find(probeRows.begin(), probeRows.end(), std::string(option.name)) !=
                    probeRows.end()
                ? std::string("measured on this run")
                : std::string("not measured on this run");

        text += Text("  %-26s %-26s %-9s %-10s %-11s %-11s %-74s %-11s %-9.2g %s\n",
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

    // The other direction: a row this run carries that the library's report does
    // not. A name is looked for once — an entry missing from the report is one fact.
    std::vector<std::string> unreported;

    for (const std::string& name : probeRows)
    {
        if (std::find(unreported.begin(), unreported.end(), name) != unreported.end())
        {
            continue;
        }

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

// ---------------------------------------------------------------------------
// The seam this run implies: the device half of the build-defaults table, as
// FormatDeviceBuildDefaults (boys/boys_cuda_probe.hpp) hands it to the probe command.
// ---------------------------------------------------------------------------

namespace {

/// The shape cell of a row: the question a class answers, as the default-policy table keys
/// it; nullptr for an enumerator the table keys no shape by.
constexpr const char* SeamShapeCell(DeviceOptionQuestion question) noexcept {
    switch (question)
    {
        case DeviceOptionQuestion::kSingle:
            return "kSingle";
        case DeviceOptionQuestion::kAllOrders:
            return "kAllOrders";
        case DeviceOptionQuestion::kAllN:
            return "kAllN";
        case DeviceOptionQuestion::kCount:
            break;
    }

    return nullptr;
}

/// The route cell of a row: the fit route the entry runs, as the table spells it.
constexpr const char* SeamRouteCell(FitRoute route) noexcept {
    switch (route)
    {
        case FitRoute::kChebyshev:
            return "FitRoute::kChebyshev";
        case FitRoute::kRationalMinimax:
            return "FitRoute::kRationalMinimax";
    }

    return nullptr;
}

/// The scheme cell of a row: the summation the entry runs, as the table spells it.
constexpr const char* SeamSchemeCell(EvalScheme scheme) noexcept {
    switch (scheme)
    {
        case EvalScheme::kSplitClenshaw:
            return "EvalScheme::kSplitClenshaw";
        case EvalScheme::kHorner:
            return "EvalScheme::kHorner";
    }

    return nullptr;
}

/// The granularity cell of a row: the partition the entry reads, which is the axis the CPU
/// lane names \c FitGranularity and this lane names \c DeviceOptionAxis::kPartition.
///
/// An entry that reads no member of either partition - the single-order shapes, the shipped
/// ladders, the all-N and each-order shapes and the handle-reading generics - carries the
/// coarsest cut, which is the partition those entries are built from; \c DevicePartitionName
/// (boys_cuda_options.hpp) states which entries those are. The cell is therefore a statement
/// about the arithmetic of this build and not a choice the entry made.
const char* SeamGranularityCell(const DeviceOptionInfo& row) noexcept {
    const char* const partition = DevicePartitionName(row.entry);

    if (partition != nullptr && std::strcmp(partition, "narrow") == 0)
    {
        return "FitGranularity::kNarrow";
    }

    if (partition != nullptr && std::strcmp(partition, "uniform") == 0)
    {
        return "FitGranularity::kUniform";
    }

    return "FitGranularity::kCoarsest";
}

// The one row list this run read, expanded twice: once as the rows written back into the file
// it emits, cell for cell and stringized rather than re-derived - so a row of the committed
// file or of another replacement reaches the file this run writes with the tokens it was
// written with - and once as the (device, precision, shape) classes those rows already carry.
#define BOYS_DEVICE_PROBE_SEAM_ROW(device, precision, shape, route, scheme, budget, pack,          \
                                   granularity, division)                                          \
    "    X(" #device ", " #precision ", " #shape ", " #route ", " #scheme ", " #budget ", "        \
    #pack ", " #granularity ", " #division ")\\\n"

#define BOYS_DEVICE_PROBE_SEAM_CLASS(device, precision, shape, route, scheme, budget, pack,        \
                                     granularity, division)                                        \
    {#device, #precision, #shape},

/// One class of the table in force, as that table's own list spells it.
struct SeamClassKey {
    const char* device;
    const char* precision;
    const char* shape;
};

#if defined(BOYS_BUILD_DEFAULT_ROWS)
/// The classes the table in force already carries. A class already there is left exactly as
/// it is: writing a second row for it would be two explicit specializations of one template,
/// which does not compile.
const SeamClassKey kTableClasses[] = {BOYS_BUILD_DEFAULT_ROWS(BOYS_DEVICE_PROBE_SEAM_CLASS)};
#else
// A build that carries only the five names lists no class, so none is already carried. The one
// entry is the empty key rather than an empty array, which C++ does not have: a name of ""
// matches no class below.
const SeamClassKey kTableClasses[] = {{"", "", ""}};
#endif

#undef BOYS_DEVICE_PROBE_SEAM_CLASS

// Two steps, so a macro is stringized as its value and not as its own name.
#define BOYS_DEVICE_PROBE_STRING(text) #text
#define BOYS_DEVICE_PROBE_VALUE(text) BOYS_DEVICE_PROBE_STRING(text)

} // namespace

DeviceDefaultsEmission FormatDeviceBuildDefaults(const DeviceProbeReport& report,
                                                 const std::string& takenAt) {
    DeviceDefaultsEmission emission;

    constexpr DeviceOptionQuestion kQuestions[] = {DeviceOptionQuestion::kSingle,
                                                   DeviceOptionQuestion::kAllOrders,
                                                   DeviceOptionQuestion::kAllN};

    std::string rows;
    bool measured = false;

    for (const DeviceOptionQuestion question : kQuestions)
    {
        const std::string klass = Text("kFp32Device %s", QuestionName(question));
        const char* const shapeCell = SeamShapeCell(question);
        bool carried = false;

        // A class the table in force already carries is left exactly as it is: writing a
        // second row for it would be two specializations of one template, which does not
        // compile. It is a class this run has nothing to add to.
        for (const SeamClassKey& known : kTableClasses)
        {
            carried = carried || (std::strcmp(known.device, "kDevice") == 0 &&
                                  std::strcmp(known.precision, "kFp32Device") == 0 &&
                                  std::strcmp(known.shape, shapeCell) == 0);
        }

        if (carried)
        {
            emission.refused.push_back(Text("%s: the table in force already carries this class, "
                                            "so its row is left as that file wrote it",
                                            klass.c_str()));
            continue;
        }

        // The class this row would come from: the fp32 device lane's, which is the one
        // precision cell the default-policy table gives the device lane. A row states what the
        // library picks for a class, so it is the entry that class's own readings placed first.
        const DeviceProbeClass* founder = nullptr;

        for (const DeviceProbeClass& candidate : report.classes)
        {
            if (candidate.precision == PrecisionName(DeviceOptionPrecision::kFp32) &&
                candidate.question == QuestionName(question))
            {
                founder = &candidate;
            }
        }

        if (founder == nullptr)
        {
            emission.refused.push_back(
                Text("%s: this run carried no fp32 class of that question", klass.c_str()));
            continue;
        }

        const std::string& winner = founder->ranking.recommended;

        if (winner.empty() || founder->ranking.verdict == DeviceProbeVerdict::kCannotDetermine)
        {
            emission.refused.push_back(
                Text("%s: %s", klass.c_str(),
                     founder->ranking.reason.empty()
                         ? "the class's ranking named no entry and gave no reason"
                         : founder->ranking.reason.c_str()));
            continue;
        }

        const DeviceOptionInfo* row = nullptr;

        for (const DeviceOptionInfo& candidate : BoysDeviceOptions())
        {
            if (winner == candidate.name)
            {
                row = &candidate;
            }
        }

        if (row == nullptr)
        {
            emission.refused.push_back(
                Text("%s: the entry it named, '%s', is no row of the library's option space",
                     klass.c_str(), winner.c_str()));
            continue;
        }

        // A winner reached by being the last entry left standing is a choice and not a
        // measurement: the seam's own header asks for the two markers to differ, and a walkover
        // recorded as a win would be a default nobody measured.
        const bool walkover = founder->ranking.defaultHow == DeviceProbeDefaultHow::kOnlyEntry;

        rows += walkover
                    ? Text("    /* a choice, not a measurement: '%s' was the last entry standing "
                           "in\n"
                           "       this class, so the row is an answer and not the winner of a\n"
                           "       comparison */\\\n",
                           winner.c_str())
                    : Text("    /* measured: the fp32 %s class, '%s', reached by %s */\\\n",
                           QuestionName(question), winner.c_str(),
                           DeviceProbeDefaultHowName(founder->ranking.defaultHow));

        rows += Text("    X(kDevice, kFp32Device, %s, %s, %s,\\\n"
                     "      BoysBudget::kFloat, PackAxis::kArguments, %s,\\\n"
                     "      DivisionForm::kRefinedReciprocal)\\\n",
                     shapeCell, SeamRouteCell(row->route), SeamSchemeCell(row->scheme),
                     SeamGranularityCell(*row));

        measured = measured || !walkover;

        const std::string how =
            Text("reached by %s, a measurement of this class's own runs",
                 DeviceProbeDefaultHowName(founder->ranking.defaultHow));

        emission.emitted.push_back(
            Text("%s: '%s', %s", klass.c_str(), winner.c_str(),
                 walkover ? "the last entry standing - written as a choice and not as a "
                            "measurement"
                          : how.c_str()));
    }

    // A file written by a run that measured nothing would be a transcription of the seam and
    // not a measurement, which is the rule the option probe's own emission keeps.
    if (!measured)
    {
        emission.text.clear();
        return emission;
    }

    std::string text;
    text += "#pragma once\n\n";
    text += "/// \\file\n";
    text += "/// This build's default-policy seam, the device half, written by the device option\n";
    text += "/// probe from the classes it measured: the device classes it ranked carry the entry\n";
    text += "/// that won them, and the classes the table in force already carried are written\n";
    text += "/// back beside them, because a replacement is read INSTEAD of the committed file and\n";
    text += "/// one that dropped them would leave their callers with no row at all.\n";
    text += "///\n";
    text += "/// The command is `boys-device-probe --emit-defaults <file>`, and the surface it\n";
    text += "/// reports is `boys/boys_cuda_probe.hpp`. A row is a measurement taken on one card,\n";
    text += "/// which is why the card, the run and the date are in the comments below.\n";
    text += "///\n";

    if (!takenAt.empty())
    {
        text += Text("/// The card and the run: %s\n///\n", takenAt.c_str());
    }

    text += "/// **What a device row's cells are, and what they are not.** The class is `kDevice`\n";
    text += "/// with `kFp32Device`, the one precision cell the table gives the device lane, and the\n";
    text += "/// shape of one question the probe ranks. The route and the scheme are the entry's own\n";
    text += "/// (`DeviceEntryAxesOf`, boys_cuda_options.hpp), read from the lane and the body its\n";
    text += "/// kernels name, and the granularity is the partition it reads. The packing-axis and\n";
    text += "/// division-form cells are the seam's own values: this lane's packing axis is region A's\n";
    text += "/// reading - one ladder, or one fit per order - which is a different axis from the CPU\n";
    text += "/// lane's `PackAxis`, and its lane names no division form at all, so neither cell is a\n";
    text += "/// device choice and a reader must not read one as that. The budget is the float budget,\n";
    text += "/// which is the budget every non-half lane's default policy carries.\n";
    text += "///\n";
    text += "/// **A row a probe named by there being no rival is a choice, not a measurement.** A\n";
    text += "/// device class whose winner won an ordering carries the marker a measurement carries;\n";
    text += "/// one whose winner was the last entry left standing carries the marker a choice\n";
    text += "/// carries, and a class this run did not measure carries no row at all.\n";
    text += "\n";
    text += "/// The five a class the list below carries no row for resolves to: the build's own\n";
    text += "/// values, which is what this build compiled before this file existed.\n";
    text += Text("#define BOYS_BUILD_DEFAULT_FIT_ROUTE %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_FIT_ROUTE));
    text += Text("#define BOYS_BUILD_DEFAULT_EVAL_SCHEME %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_EVAL_SCHEME));
    text += Text("#define BOYS_BUILD_DEFAULT_PACK_AXIS %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_PACK_AXIS));
    text += Text("#define BOYS_BUILD_DEFAULT_DIVISION_FORM %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_DIVISION_FORM));
    text += Text("#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_FIT_GRANULARITY));

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    text += "/// The classes this file sets a default for: **one row per class**, in the table's own\n";
    text += "/// format. The rows first are the table in force's own, written back verbatim; the\n";
    text += "/// device rows after them are this run's.\n";
    text += "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n";
    text += BOYS_BUILD_DEFAULT_ROWS(BOYS_DEVICE_PROBE_SEAM_ROW);
#else
    // A build that carries only the five names has no classes to write back: the rows below are
    // the whole list, which is the one statement this run can make about a table it does not
    // have.
    text += "/// The classes this file sets a default for: **one row per class**, in the table's own\n";
    text += "/// format. This build's seam carries no row list, so the device rows below are the\n";
    text += "/// whole of it.\n";
    text += "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n";
#endif

    text += rows;
    text += "\n";
    emission.text = text;
    return emission;
}

#undef BOYS_DEVICE_PROBE_VALUE
#undef BOYS_DEVICE_PROBE_STRING
#undef BOYS_DEVICE_PROBE_SEAM_ROW

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
        // reader of a failed run learns which options exist, which this build
        // refuses and which cells of the space they are, and what is missing is
        // the measurement.
        AppendOptionSpace(text, std::vector<std::string>(), 0);
        // And counted, which is where a failed run's coverage stands: the members
        // this run did not present to the device are in no state, and the verdict
        // names that rather than leaving the space uncounted.
        AppendOptionClosure(text, report);
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
    text += Text("    %zu arguments per call, and %zu at the pair count (the wider of the two "
                 "every row\n    is read at), log-uniform over [%.3g, %.3g], each carrying its "
                 "own highest\n",
                 report.workloadCount,
                 report.workloadCount * static_cast<std::size_t>(report.options.countPairFactor),
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
                 "uploaded once before the first\n    clock; the per-argument reading is the "
                 "device time divided by the launches and the\n    arguments.\n",
                 report.options.passes,
                 report.options.rounds,
                 report.options.repetitions);
    text += Text("    each entry is read twice per round, at %zu and at %zu arguments, and its "
                 "figure is the\n    count-independent cost the two readings extrapolate to: a "
                 "launch term L paid once per\n    launch enters a reading at C as L/C, so with "
                 "the second count r times the first the two\n    readings fix the term and the "
                 "cost the entry is left with is (r*f' - f)/(r - 1). Both readings are\n    "
                 "reported beside the figure. **Nothing an empty kernel costs is subtracted**: "
                 "the\n    extrapolation is the entry's own, and the floor the control states is "
                 "the cheaper\n    launch a kernel that does no arithmetic makes.\n",
                 report.workloadCount,
                 report.workloadCount * static_cast<std::size_t>(report.options.countPairFactor));
    text += Text("    every row is timed once per round and the visit order is shuffled per round, "
                 "so the %d\n    rounds of the run are pooled into one table of %d rows and %zu "
                 "columns — one row per\n    round and one column per row of the measurement table. "
                 "A comparison between two entries is\n    the ratio of their cells in one row: both "
                 "were timed under whatever clock that round ran\n    at, and a drift common to the "
                 "round cancels in the ratio. Every row of the run is timed under\n    the one set "
                 "of degree tables this run made resident before any row was timed, so any\n    two "
                 "entries of a comparison were timed under one arithmetic.\n",
                 report.pairedRounds,
                 report.pairedRounds,
                 report.measurements.size());
    text += "    no round and no pass is dropped. A reported figure is the middle of those "
            "ratios\n    and never the minimum of the run: on a card whose clock decays the "
            "minimum is\n    the earliest and best-clocked round, which is not what a caller's "
            "long workload\n    meets, and a central ratio credits the reference entry and its "
            "rivals alike; the\n    upper quartile of that ratio is printed beside the figure so "
            "both ends come from\n    one distribution. The peak column is the entry's own fastest "
            "single round, a raw\n    figure under no anchor and not a bound on the columns beside "
            "it.\n";
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

    const std::size_t pairCount =
        report.options.count * static_cast<std::size_t>(report.options.countPairFactor);

    const std::string floorSentence =
        report.control.nsPerLaunchFloor > 0.0
            ? Text("The device-side launch\n  floor - a kernel launched the same way that does no "
                   "Boys arithmetic - is %.3f us per\n  launch here, %.4f ns/argument at the first "
                   "of those counts. It is the cheapest launch\n  this route can make, and it is "
                   "not what came out of the figures: an entry's own\n  kernel starts more work "
                   "than an empty one, so its launch term is the larger one the\n  two readings "
                   "fixed for it.",
                   report.control.nsPerLaunchFloor / 1000.0,
                   report.control.nsPerLaunchFloor / static_cast<double>(report.options.count))
            : std::string("The device-side launch\n  floor could not be timed on this run, so no "
                          "floor is stated; the figures do not rest on it.");

    text += Text("\nentries - ns/argument is the arithmetic's own device time per argument, "
                 "transfer and\n  host submission excluded, and each row's figure is formed from "
                 "that row's own two\n  readings: the column headed %zu is the row at %zu "
                 "arguments per call, the column\n  headed %zu is the same row at %zu arguments, "
                 "and 'ns/arg' is the count-independent\n  cost the two extrapolate to. A launch "
                 "costs what it costs once per launch, so the\n  share of a reading that is the "
                 "launcher falls as 1/count: with the pair count r times\n  the first, readings f "
                 "and f' leave the entry (r*f' - f)/(r - 1). That the launch term\n  falls as "
                 "1/count is the one assumption the extrapolation makes, and both readings are\n  "
                 "printed so that what came out of a figure can be checked rather than taken on "
                 "trust.\n  %s\n",
                 report.options.count,
                 report.options.count,
                 pairCount,
                 pairCount,
                 floorSentence.c_str());

    text += Text("  %-24s %-10s %-11s %-10s %10zu  %10zu  %10s  %8s  %-7s  %7s  %-12s  %7s  "
                 "%8s  %-6s  %8s  %s\n",
                 "entry",
                 "precision",
                 "shape",
                 "route",
                 report.options.count,
                 pairCount,
                 "ns/arg",
                 "hi",
                 "minus",
                 "spread",
                 "vs ref band",
                 "drift%",
                 "peak ns",
                 "rounds",
                 "bound",
                 "method");

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
            text += Text("  %-24s %-10s %-11s %-10s %10s  %10s  %10s  %-7s  %-7s  %-7s  %-12s  "
                         "%-7s  %8s  %-6s  %8.2g  %s\n",
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
            text += Text("  %-24s %-10s %-11s %-10s %10s  %10s  %10s  %-7s  %-7s  %6.2fx  %-12s  "
                         "%+6.2f%%  %8s  %-6s  %8.2g  %s\n",
                         measurement.name.c_str(),
                         measurement.precision.c_str(),
                         measurement.shape.c_str(),
                         measurement.route.c_str(),
                         "-",
                         "-",
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

        text += Text("  %-24s %-10s %-11s %-10s %10.3f  %10.3f  %10.3f  %8.3f  %-7s  %6.2fx  "
                     "%-12s  %+6.2f%%  %8s  %-6s  %8.2g  %s\n",
                     measurement.name.c_str(),
                     measurement.precision.c_str(),
                     measurement.shape.c_str(),
                     measurement.route.c_str(),
                     measurement.nsPerArgumentAtCount,
                     measurement.nsPerArgumentAtPairCount,
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

    text += Text("\n  the three cost columns and the two readings are the reference entry's own "
                 "lower-quartile\n  cost over this entry's %d paired rounds, scaled by this "
                 "entry's ratio to it: the two\n  readings at the lower quartile of that ratio, "
                 "the figure in ns/arg at its middle\n  ('%s', whose own ratio is exactly 1.000 "
                 "and whose drift is exactly 0.00%%) — so the two\n  readings and the figure are "
                 "anchored alike from the same rounds, and the figure is the\n  reading column it "
                 "sits beside with the launch term the two readings fixed taken out of it. A\n"
                 "  launched row's readings both contain that term, the second with a\n  "
                 "quarter of it, which is why the second is the cheaper of the two; an\n  "
                 "in-kernel row's readings are differences between two halves that both carried "
                 "the\n  launch, so its two columns sit on top of each other and its figure barely "
                 "moves. 'hi' is\n  the figure at the upper quartile of the same rounds and "
                 "'spread' is hi over lo, so the\n  two ends of a row come from one distribution. "
                 "The vs-ref band is that ratio over the\n  middle half of the rounds, and drift "
                 "is how far it moved between the run's first and\n  second half — a row whose "
                 "drift is larger than the resolution below is a row whose\n  two entries do not "
                 "carry this card's clock alike. 'peak ns' is the fastest single round\n  the entry "
                 "was ever seen in: one round's own figure under no anchor, the entry's own\n  "
                 "floor and not a bound on the columns beside it, which are anchored to the\n  "
                 "reference entry — and a round whose cell was zero is left out of it, because "
                 "such a\n  cell is not a cost. 'rounds' is how many of the run's pooled rounds the "
                 "row rests on, and\n  every measured row rests on every one of them — no round "
                 "and no pass is dropped for it.\n",
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

    text += "\n  the bound column is what each entry's lane documents, read from the library's own "
            "report\n  of the option space and not measured here. What a lane delivers on this card "
            "is the accuracy\n  gate's business. The bounds in that column are not all the same, and "
            "a row that carries a\n  looser one is a row that bought speed with accuracy: the "
            "classes below are per precision\n  and per question shape, so no row is ordered against "
            "a row of another class, and the name a class\n  carries is reached among its own rows "
            "and read at the bound each of them states.\n";

    // --- The controls --------------------------------------------------------
    //
    // One per route, since the two routes are two methods and a check of one
    // says nothing about the other. The floor is the launched route's own: a
    // subtraction has no launch of this library's inside either half.
    const auto AppendControl = [&text, &report](const char* heading,
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
            // The floor as the diagnostic it is: what a kernel with no Boys
            // arithmetic in it costs to launch here, not the term that came out
            // of the figures. Each row's own launch term is the larger one its
            // two readings fixed, so this line states a lower bound and says so.
            text += control.nsPerLaunchFloor > 0.0
                        ? Text("  floor, a kernel launched the same way with no Boys arithmetic "
                               "in it: %.3f us per launch\n  (%.4f ns/argument at this run's "
                               "count). A diagnostic, not what came out of the figures:\n  an "
                               "entry's own kernel costs more to start than this empty one, and "
                               "each row's launch\n  term is the one its own two readings fixed. A "
                               "row whose whole figure is below this\n  line is launch-bound.\n",
                               control.nsPerLaunchFloor / 1000.0,
                               control.nsPerLaunchFloor /
                                   static_cast<double>(report.options.count))
                        : std::string("  the floor could not be timed on this run, so no floor is "
                                      "stated. The figures above do\n  not rest on it: each row's "
                                      "launch term is the one its own two readings fixed.\n");
        }

        text += Text("  %s\n", control.note.c_str());
    };

    AppendControl("launch control", report.control, true);
    AppendControl("subtraction control", report.deviceCallControl, false);

    // --- The rankings --------------------------------------------------------
    //
    // One class per precision and question shape: neither is traded for speed here,
    // and what varies inside a class is what the library picks on the caller's
    // behalf.
    if (!report.classes.empty())
    {
        text += Text("\nrankings - one class per precision and question shape.\n"
                     "  A class is what the caller has already fixed when they make the call: how "
                     "much\n  precision the result needs, and what they are asking for. This run's "
                     "option table\n  carries %zu row(s) across %zu class(es), and no entry of one "
                     "class was ordered against\n  an entry of another. What varies inside a class "
                     "is what the library picks on the\n  caller's behalf.\n",
                     report.measurements.size(),
                     report.classes.size());
    }

    /// The rows of the option table that stand in one class: the ones of that
    /// class's precision whose question is the class's. It is the rows the table
    /// lists, measured or not, because what a row documents is a fact about the row
    /// whether or not this run got a figure out of it.
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

    /// How the shape's recommendation was reached, in the report's own words: the one thing a
    /// consumer reading a name has to be able to tell, since a name measured over the shape's own
    /// rounds and a name taken from a vote over re-runs are answers of different strength, and a
    /// name taken by choice among entries the runs divided evenly is not a measurement at all.
    ///
    /// A shape that reached no recommendation gets the line saying so, and the reason above it says
    /// which count or which row was missing.
    ///
    /// /// \\param ranking the ranking whose name is being explained
    /// /// \\param clause the class the ranking sits in, which is what its shape's rows are read out
    ///                of: a name taken because the shape had nothing to order it against is one
    ///                answer when the shape holds one row and another when the run's own checks
    ///                left one row of several standing, and the line says which it was
    const auto AppendReached = [&text, &ShapeRows](const DeviceProbeRanking& ranking,
                                                  const DeviceProbeClass& clause) {
        if (ranking.recommended.empty())
        {
            text += "\n  reached by: nothing — this shape named no entry\n";
            return;
        }

        switch (ranking.defaultHow)
        {
            case DeviceProbeDefaultHow::kOrdered:
                text += "\n  reached by: a measured ordering on this shape's own rounds — every "
                        "entry the\n    shape holds was the slower of the two against the leader "
                        "in the middle half of\n    the same pooled rounds\n";
                return;
            case DeviceProbeDefaultHow::kOnlyEntry:
                if (ShapeRows(clause, ranking).size() <= 1)
                {
                    text +=
                        "\n  reached by: the shape holding one entry. There was nothing to order it "
                        "against\n    and this run measured no rival, so it is named by there "
                        "being no alternative —\n    which is an answer and not a ranking\n";
                    return;
                }

                // The shape holds several rows and the run could place only one of
                // them: the name is the row that was left, not the row that won,
                // and the reader is told which of the two it is holding.
                text += "\n  reached by: the shape holding no rival this run could place. Every "
                        "other entry\n    of it was set aside by the run's own checks or produced "
                        "no figure, so the entry\n    named is the one it had left to name — an "
                        "answer, and not the winner of a\n    comparison\n";
                return;
            default:
                break;
        }

        const DeviceProbeRefinement& stage = ranking.refinement;

        // The entry the vote named is the entry this shape names, so the line says how
        // the vote came out. The entry the shape's own figures put first is printed
        // beside the name below, and where the two differ the difference is what says
        // the shape's top entries cannot be separated.
        text += Text("\n  reached by: the refinement stage — %s\n",
                     stage.unanimous   ? "a unanimous re-run of the entries this shape could not "
                                         "separate"
                     : stage.plurality ? "a majority vote over re-runs of the entries this shape "
                                         "could not separate"
                     : !stage.winner.empty()
                         ? "a choice among entries the re-runs divided evenly; the shape's top\n"
                           "    entries are entries this stage could not separate"
                         : "no run placed a leader, so the entry the shape's own figures put\n"
                           "    first stands");
        text += Text("    the entries the shape's own rounds could not separate, re-run alone: "
                     "%zu\n",
                     stage.pool.size());

        for (const std::string& name : stage.pool)
        {
            text += Text("      %s\n", name.c_str());
        }

        text += Text("    protocol: %d run(s), each %d passes by %d rounds\n",
                     stage.runs,
                     stage.passes,
                     stage.rounds);

        for (std::size_t index = 0; index < stage.runLeaders.size(); ++index)
        {
            text += Text("      run %zu led with %s\n",
                         index + 1,
                         stage.runLeaders[index].empty() ? "no entry"
                                                         : stage.runLeaders[index].c_str());
        }

        for (const std::string& line : stage.tally)
        {
            text += Text("    %s\n", line.c_str());
        }

        text += Text("    vote: %s\n", stage.note.c_str());
        text += Text("    vote named: %s\n",
                     stage.winner.empty() ? "no entry" : stage.winner.c_str());

        const std::string why =
            ranking.recommended == ranking.fastestOverall
                ? std::string("the entry the vote named, which the shape's own figures put first "
                              "too")
                : Text("the entry the vote named; this shape's own figures put %s first, which is "
                       "what\n      says the shape's top entries cannot be separated",
                       ranking.fastestOverall.empty() ? "no entry"
                                                      : ranking.fastestOverall.c_str());

        text += Text("    default: %s — %s\n", ranking.recommended.c_str(), why.c_str());
    };

    // The classes, in full: one per precision and question shape, each the class a
    // default is read from. There is no second block: the library serves one
    // arithmetic and a class is the whole of it.
    for (const DeviceProbeClass& clause : report.classes)
    {
        const DeviceProbeRanking& ranking = clause.ranking;

        text += Text("\nclass %s, %s\n",
                     clause.precision.c_str(),
                     clause.question.c_str());
        text += Text("  %s\n", clause.note.c_str());

        {
            const std::vector<const DeviceProbeMeasurement*> rows = ShapeRows(clause, ranking);

            text += Text("\n  every entry here was asked for %s\n", ranking.asked.c_str());

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
                AppendReached(ranking, clause);
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
                             "than %.2f%%\n                cannot be ordered on this run's own "
                             "rounds. The number is measured,\n                not assumed: it is "
                             "the widest ratio band this shape showed, taken over\n"
                             "                the leader against each rival and over each entry "
                             "against the reference,\n                all of them ratios formed "
                             "inside a round. %s",
                             100.0 * ranking.resolution,
                             canaryContext.c_str());

                if (ranking.refinement.ran)
                {
                    // A shape the resolution could not order is a shape whose tie
                    // was carried to the refinement stage, and a reader of this
                    // line has to see that the band was answered rather than left
                    // standing as the last word.
                    text += Text("                This shape's tie was carried past it: see the "
                                 "refinement\n                stage below.\n");
                }
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
                // keyed on: this line gets quoted as "the fastest" without the
                // accuracy it is fastest at, so the row's own bound is repeated
                // here.
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
                                 "same bound, %.2g,\n      so the ranking is at one accuracy: the "
                                 "name above rests on entries of the\n      shape all measured at "
                                 "it, in the %s class.\n",
                                 recommendedBound,
                                 clause.precision.c_str());
                } else
                {
                    // The named bound is the tightest one beside the winner, so
                    // the direction holds for that row and the looser rows are
                    // looser still.
                    text += Text("      this row documents a bound of %.2g, %s the %.2g of %s.\n"
                                 "      The entry this shape names is named among entries that "
                                 "document two bounds,\n      so its figure is read at the bound "
                                 "its own row states and not as the fastest at\n      one accuracy. "
                                 "The bound column above is the figure to weigh against this one.\n",
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
            AppendReached(ranking, clause);
        }
    }

    text += Text("\n%s\n", report.caveat.c_str());

    {
        std::vector<std::string> probeRows;

        for (const DeviceProbeMeasurement& measurement : report.measurements)
        {
            // The rows this run has a figure for, and not every row of the space
            // it holds a place for: a row the run carried and got no figure from
            // is a place and not a measurement. The appendix counts what it names
            // — 'row(s) measured here' — and a count that included the empty
            // places would say a run carried an entry it never timed.
            if (!measurement.measured)
            {
                continue;
            }

            probeRows.push_back(measurement.name);
        }

        // The places this run offered and got no figure from, which is a different
        // fact from a row the space does not offer: the first was offered and
        // produced nothing, the second is refused by the library's own contract
        // and is counted where the rows are.
        std::size_t emptyRows = 0;

        for (const DeviceProbeMeasurement& measurement : report.measurements)
        {
            if (!measurement.measured)
            {
                ++emptyRows;
            }
        }

        AppendOptionSpace(text, probeRows, emptyRows);
    }

    // The degree tables a run could not make resident, if any: a refusal with the
    // library's own reason and not a silence. It follows the option space, which is
    // where a reader looks for what the build carries, and the closure below counts
    // the rows it covers apart from the ones the library refuses.
    if (!report.refusedTables.empty())
    {
        text += Text("\nnot measured on this run: %s\n", report.refusedTables.c_str());
    }

    text += "\nnot measured by this probe, by design: the ordered-batch sort (the AllN entries "
            "are\n  handed a batch that already satisfies their stated precondition, so nothing "
            "here is a\n  figure for sorting one); the arithmetic's accuracy (the device accuracy "
            "gate measures\n  that, against the committed reference). The accuracy figures "
            "themselves are not absent\n  from this report: each row's bound column states the "
            "figure its own lane documents, read\n  from the library's table rather than measured "
            "here.\n";

    // The last block of the report, on every path: the space counted, so that what
    // this run did with it is a number a reader can check rather than an inference
    // from the sections above.
    AppendOptionClosure(text, report);

    return text;
}

} // namespace boys
