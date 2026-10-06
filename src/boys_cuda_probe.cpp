// The option probe's host side: the workload, the protocol, the folding of the rounds into
// figures, the refusal to order noise, and the text a consumer reads. The clock is not here -
// every timed region is a CUDA event pair opened and closed through the kernels file, so the
// host's submission stays out. An in-kernel row's two halves are timed adjacent, alternating.

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
    std::string name;
    const char* precision;
    const char* shape;
    ProbeQuestion question;
    bool inKernel;

    /// The division form this row's ladder steps divide in: the coordinate that
    /// makes two rows of one entry different rows, and the value every region of
    /// this row is dispatched at (\c ProbeTimeRequest::form).
    DivisionForm form;

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
        case DeviceOptionPrecision::kBf16:
            return "bf16";
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

/// The division form a row's ladder steps divide in, as the report spells it; nullptr for a value
/// outside the enumerators.
///
/// The library spells these too (\c DivisionFormName, boys/backend.hpp), and this table is the
/// report's own the way the route, the scheme and the packing are: an enumerator added to
/// \c DivisionForm and not named here stops this translation unit at
/// \c DivisionIsNamed below, where a table keyed by the library's name function would have
/// printed its fallback for it.
constexpr const char* DivisionName(DivisionForm form) noexcept {
    switch (form)
    {
        case DivisionForm::kExactDivision:
            return "exact-division";
        case DivisionForm::kPlainReciprocal:
            return "plain-reciprocal";
        case DivisionForm::kRefinedReciprocal:
            return "refined-reciprocal";
    }

    return nullptr;
}

/// One enumerator of \c DivisionForm that \c DivisionName spells no name for.
template <DivisionForm Form>
struct DivisionIsNamed {
    static_assert(DivisionName(Form) != nullptr,
                  "an enumerator of DivisionForm names no form: name it in DivisionName "
                  "(src/boys_cuda_probe.cpp)");
};

template struct DivisionIsNamed<DivisionForm::kExactDivision>;
template struct DivisionIsNamed<DivisionForm::kPlainReciprocal>;
template struct DivisionIsNamed<DivisionForm::kRefinedReciprocal>;

/// The axes of a row: the member of the axis it varies, where it varies one, and then the route,
/// the summation, the region-A packing and the division form its entry runs.
///
/// The four axes last are stated on every row and not only on the rows that move them, because
/// that is what places a row against its neighbours: a row whose axis is kNone is the entry, and
/// before this column carried them, the arithmetic it runs was recoverable from its name and
/// from nothing the report printed. A row of the packing, scheme, route or division axis carries
/// its member among the four, so nothing is said twice.
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
        case DeviceOptionAxis::kDivision:
            text = Text("division:%s ", DivisionName(option.division));
            break;
        case DeviceOptionAxis::kNone:
        case DeviceOptionAxis::kPacking:
        case DeviceOptionAxis::kScheme:
        case DeviceOptionAxis::kRoute:
            break;
        default:
            // An axis no arm above names - a value cast in from outside the enumeration, or a member a newer
            // header carries. The row is still printed, because a row the library carries that this probe
            // cannot place is a fact about the space, but the cell refuses to write the value down as a member.
            text = "axis:(not one this probe names) ";
            break;
    }

    text += Text("route:%s scheme:%s packing:%s division:%s",
                 RouteName(option.route),
                 SchemeName(option.scheme),
                 PackingName(option.packing),
                 DivisionName(option.division));
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
        "the evaluation scheme, the partition and the packing axis — and the division form, which "
        "the caller may name and which the build's own default supplies when they name none. So "
        "its winner is the fastest %s row of this question, at the bound its own row states, and "
        "the bound column above is that figure.",
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

/// The name one row of this probe's option table is printed under: the library's own name for the
/// entry, with the division form's own name appended where the row does not run the build's default
/// form.
///
/// The row that carries no form segment is the row a caller who names no form reaches — every
/// launched entry's trailing parameter defaults to \c kDefaultDivisionForm and every
/// device-callable entry's template argument does too (boys_cuda.hpp) — so the form left
/// unmarked is the form that row really divides in, and the two other members of the axis are
/// marked. That is the option probe's own grammar for its cells (src/boys_probe.cpp,
/// \c CellName) and it is read here for the same reason it is read there: without the segment,
/// three rows of one entry would print one name, and a name is what the rankings beside this
/// table conclude on.
///
/// The segment is the library's own spelling of the member (\c DivisionFormInfo::name,
/// \c BoysDivisionForms) and not a second list here, so a form renamed in the library renames its
/// rows.
///
/// \param option the library's row this probe row is a member of
/// \param form   the division form this probe row is the member at
///
/// \returns the name, which is the entry's own where \p form is the build's default
std::string ProbeRowName(const DeviceOptionInfo& option, DivisionForm form) {
    std::string name = option.name;

    if (form == kDefaultDivisionForm)
    {
        return name;
    }

    for (const DivisionFormInfo& member : BoysDivisionForms())
    {
        if (member.form == form)
        {
            name += "-";
            name += member.name;
            return name;
        }
    }

    // A form outside the library's own enumeration cannot arrive: BoysDivisionForms() is
    // what the crossing below reads the axis off, and this function is called with a
    // member of it. Named rather than left to fall through, so that a value that did
    // arrive is visible in the name it produces.
    name += Text("-form-%d", static_cast<int>(form));
    return name;
}

/// The option table, projected from the library's own report: one probe row per report row this
/// build serves crossed with one member of the library's division-form axis, and one unoffered
/// entry per report row this build does not serve, carrying the library's reason.
///
/// **The cross is the whole space and not a sample of it.** Every entry of this surface runs every
/// form — the form is a trailing parameter of every launched entry and a template argument of every
/// device-callable one, the axis's own rows say every entry this build carries runs every one of
/// them (\c DivisionFormInfo, boys.hpp), and the kernel bodies are instantiated from it
/// (src/boys_cuda_probe_kernels.cu, \c LaunchInKernel and \c LaunchLaunched) — so a row of the
/// library's table is three rows here and none of the three is a spelling of another. A row this
/// build does not serve is refused once and not three times: it is not an arithmetic this build can
/// run at any form, so it stands in \c unoffered under the library's own name and carries the
/// library's own reason.
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

        for (const DivisionFormInfo& member : BoysDivisionForms())
        {
            entries.push_back(EntryInfo{option.entry,
                                        ProbeRowName(option, member.form),
                                        PrecisionName(option.precision),
                                        ShapeName(option.shape),
                                        option.question,
                                        option.group == DeviceOptionGroup::kDeviceCallable,
                                        member.form,
                                        option.bound});
        }
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
    std::vector<std::uint16_t> xb;
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
    work.xb.resize(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        work.xf[i] = static_cast<float>(work.x[i]);
        work.xh[i] = F16(static_cast<float>(work.x[i])).Bits();
        work.xb[i] = Bf16(static_cast<float>(work.x[i])).Bits();
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

    // The form this row is measured at: the row's own coordinate and not a value chosen here, so the
    // region dispatched below runs the arithmetic the row's axes column reports. Reading it off the
    // entry would measure the default form three times under three names.
    const int form = static_cast<int>(info.form);

    if (!info.inKernel)
    {
        ProbeTimeRequest request = base;
        request.what = static_cast<int>(ProbeWhat::kLaunchedEntry);
        request.entry = static_cast<int>(info.entry);
        request.form = form;
        request.reps = reps;
        timed.ok = TimeRegion(request, timed.withMs);
        return timed;
    }

    ProbeTimeRequest withBoys = base;
    withBoys.what = static_cast<int>(ProbeWhat::kInKernelEntry);
    withBoys.entry = static_cast<int>(info.entry);
    withBoys.form = form;
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
    // A count whose readings left no figure to form leaves no second figure to compare, and the one
    // outcome that must never come out of that is agreement: the control would be passing without
    // evidence. The difference is infinite rather than zero, so the data says what the note says.
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

    // A quartile band needs four rounds: with fewer, the quartiles are two- and three-point order
    // statistics and a probe that ordered on them would report a resolution it never measured. The
    // entries this run read a figure for are the band it could not form, and the refinement stage
    // re-runs them.
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
        // The clock check, done rather than assumed: a pair whose ratio moves between the run's halves
        // does not carry a decaying clock alike, and the report says so instead of implying the ordering
        // holds at any clock. A pair that held still within the resolution puts no caveat on the ordering.
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

/// The refinement stage: a shape's tied entries, measured alone at the refinement protocol,
/// repeated, and voted on.
///
/// The entries re-measured are the shape's fastest entry and every entry of it the main run could
/// not place behind the fastest. Each run is a fresh pass over the tied set at \c passes passes of
/// \c rounds * the refinement factor rounds - the factor lengthens the run's rounds and not its
/// passes, so a run is the refinement factor times the protocol it refines and not that factor
/// squared - with its own shuffle, ordered by the same within-round ratio rule the main run used. A
/// run whose own rounds cannot place a rival contributes its leader alone, which is what makes the
/// vote a vote rather than a re-run of the main statistic.
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
    // ONE multiplication, not two - the same defect the host probe carried, at the same line of its
    // own stage. The stage refines a tie by asking whether the leader holds up over more ROUNDS of the
    // same comparison, so the factor lengthens the rounds and leaves the passes alone. Multiplying
    // both made a run the factor SQUARED and the whole stage 125 times the protocol it refines.
    stage.passes = std::max(1, clamped.passes);
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
    const auto stageStarted = std::chrono::steady_clock::now();

    for (int run = 0; run < stage.runs; ++run)
    {
        std::fprintf(stderr,
                     "boys-device-probe: refinement run %d of %d over %zu tied entry(s), %.1fs "
                     "into the stage\n",
                     run + 1, stage.runs, pool.size(),
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - stageStarted)
                         .count());
        std::fflush(stderr);
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

    // The vote decides the name where it ran: options a class cannot separate are settled by which was
    // fastest in most runs, so the row the vote named is the default and the row this shape's own
    // figures put first is the record of the shorter protocol. The name above stands only with no vote.
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
    DeviceBuffer argsBf16;
    DeviceBuffer output;
    DeviceBuffer canarySink;

    if (BoysCudaProbeAlloc(&orders.pointer, pairCount * sizeof(int)) != 0 ||
        BoysCudaProbeAlloc(&args64.pointer, pairCount * sizeof(double)) != 0 ||
        BoysCudaProbeAlloc(&args32.pointer, pairCount * sizeof(float)) != 0 ||
        BoysCudaProbeAlloc(&args16.pointer, pairCount * sizeof(std::uint16_t)) != 0 ||
        BoysCudaProbeAlloc(&argsBf16.pointer, pairCount * sizeof(std::uint16_t)) != 0 ||
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
        BoysCudaProbeUpload(args16.pointer, work.xh.data(), pairCount * sizeof(std::uint16_t)) != 0 ||
        BoysCudaProbeUpload(argsBf16.pointer, work.xb.data(), pairCount * sizeof(std::uint16_t)) != 0)
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
    base.xb = argsBf16.pointer;
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

    // One row of this run's measurement table per member of the option space this build serves —
    // one per entry per division form, the entries in the book's own order and the forms in the
    // axis's — so a round visits them in that order.
    for (std::size_t index = 0; index < entries.size(); ++index)
    {
        const EntryInfo& info = entries[index];
        DeviceProbeMeasurement& measurement = report.measurements[index];

        measurement.name = info.name;
        measurement.precision = info.precision;
        measurement.entryIndex = index;
        measurement.entry = info.entry;
        measurement.form = info.form;
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

    // The degree tables, made resident before any row is timed: one upload serves every entry of this
    // surface, host work outside every timed region. A device that will not hold them measures nothing,
    // and every row then stands as producing no figure rather than as costing what no table would.
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

    // A kernel launched the same way that does no Boys arithmetic gives the cheapest launch this
    // route can make, reported per launch and as a fraction of the fastest row. It is NOT what the
    // figures have taken out of them: an entry's kernel needs registers and an occupancy ramp an
    // empty kernel never pays, so subtracting it would take out part of the term.
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

        // A floor of zero is a floor this run did not establish, whatever the timer said: it is read here
        // from the one value both sites print, so no line can call the floor timed while another calls it
        // missing. A region whose device time rounds away reads as no reading rather than a free launch.
        floorTimed = report.control.nsPerLaunchFloor > 0.0;
    }

    // Every pass is run and every pass is used; what the canary said beside one is reported with it
    // and decides nothing. The visit order is shuffled once per round from the run's own seed, so a
    // position in the round does not enter an entry's ratio as though it were its own cost, and every
    // row of a round is timed with the same resident tables.
    std::mt19937_64 shuffle(clamped.seed);
    std::vector<std::size_t> visit(report.measurements.size());

    for (std::size_t slot = 0; slot < visit.size(); ++slot)
    {
        visit[slot] = slot;
    }

    // How many rounds of the run a launched row's own two readings left no positive launch term in,
    // so that no figure could be extrapolated for it: a row whose every round did is one this
    // workload cannot measure at these counts, which the count tells apart from a row never timed.
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

    const auto probeStarted = std::chrono::steady_clock::now();

    for (int pass = 0; pass < timedPasses; ++pass)
    {
        // Progress on stderr, not stdout: the report on stdout is what a recorded closure is made
        // from, and a run that printed progress there would be a run whose report no longer matches
        // its own format. Without this a reader cannot tell "measuring" from "hung".
        std::fprintf(stderr, "boys-device-probe: pass %d of %d, %.1fs in\n", pass + 1, timedPasses,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - probeStarted)
                         .count());
        std::fflush(stderr);
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

            // One timed entry reduced to its own reading at the count it was taken at, in cost per argument:
            // a launched entry is its region, an in-kernel entry the difference between the region that held
            // the call and the one that did not, both halves being the same kernel launched the same number of
            // times on the same grid, so the per-launch cost cancels.
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

                // The row's own two readings, in this round and at both counts, timed adjacently so both sit under
                // one clock and one workload state: one reading alone cannot say how much of it is the launch, and
                // the two together can without an empty kernel standing in for this entry's kernel cost.
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
                    // An in-kernel row: the difference came out at or below zero in this round, which is the entry's
                    // arithmetic inside the noise of its own baseline rather than a negative cost. The cell is held at
                    // zero and the fold reads that as the row's subtraction not having resolved at this workload.
                    row[index] = std::max(0.0, extrapolated);
                    baseline[index] =
                        PerArgument(atCount.withoutMs, clamped.repetitions, clamped.count);
                    continue;
                }

                if (!(extrapolated > 0.0))
                {
                    // A launched row whose two readings left no positive launch term to take out: the pair-count
                    // reading was not the cheaper of the two, so the asymptote is at or below zero and this round
                    // produced no figure. The cell is left infinite and counted, and a row whose every round did this
                    // is set aside by name rather than printed at a zero that reads as a free call.
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

    // Every figure below is an aggregate of ratios taken inside a round of the table above, which is
    // what makes it a paired comparison: two cells of one round were timed under one clock. The pair
    // statistic is a lower quartile and a row's ratio to the reference is read at the middle, so the
    // cost is the reference's own lower-quartile figure scaled by that ratio.
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

                // The two readings the cell was formed from, on the same rounds and against the same anchor: the
                // report prints them beside the figure so that what the extrapolation took out is the difference
                // between three columns of one table rather than a number a reader has to take on trust.
                if (std::isfinite(roundAtCount[round][index]) &&
                    std::isfinite(roundAtPairCount[round][index]))
                {
                    firsts.push_back(roundAtCount[round][index] / anchor);
                    seconds.push_back(roundAtPairCount[round][index] / anchor);
                }

                // The peak column is the entry's own fastest single round, and a round whose cell is zero has no
                // cost in it: for a subtracted row that cell is the floor a difference which did not clear its own
                // baseline was held at, so taking it would print the instrument's floor as the entry's best round.
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
                // A launched row whose every round left no positive launch term has no figure at all, named where
                // the report lists what a shape could not place: a row never timed and one the readings could not
                // be extrapolated from are different facts, and only the second is answered by raising the counts.
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
            // Whether the subtraction this row is came out of its own noise: an in-kernel row's difference is
            // formed between two halves that both carry the run's clock, and a difference that is not positive
            // at its lower quartile was not resolved, so the column reads unresolved and the class sets it aside.
            measurement.subtractionResolved =
                !entries[index].inKernel ||
                (measurement.nsPerArgument > 0.0 && measurement.nsPerArgumentAtCount > 0.0 &&
                 measurement.nsPerArgumentAtPairCount > 0.0);

            // How far the entry's ratio to the reference moved between the run's halves: zero means the two
            // kept pace as the clock moved, and a value away from zero says they do not carry the clock alike,
            // which the report states where it prints this entry.
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

        /// What one class's conclusion is a function of, and the conclusion it had.
        ///
        /// A class is concluded from the rows it holds and from what this run's checks have said about
        /// them, and from nothing else, so a class whose rows and flags are unchanged reaches the same
        /// conclusion. The convergence loop below names one row at a time, and without this it would
        /// re-run every class's refinement - the expensive part of a conclusion - once per named row.
        /// The class set comes from the rows this run was asked for, so a class whose every row failed
        /// to measure is still reported with its reason.
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

            // A class of a run whose tables the device would not hold measured nothing, and that is why its
            // rows produced no figure: the residency failure is put in front of whatever the conclusion wrote,
            // because a reader who expected the entry would take its absence for a fact about the arithmetic.
            if (!tablesResident)
            {
                ranking.reason = Text("the device would not hold the degree tables, so no entry of "
                                      "this class was timed. %s",
                                      ranking.reason.c_str());
                ranking.confidence = Text("CANNOT DETERMINE: the degree tables are what this device "
                                          "could not hold");
            }

            // A launched row the two readings could not be extrapolated from was not placed and is not
            // silently absent either - a reader who expected it in the table would take its absence for a build
            // fact - so it is named here with what its own readings did.
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

            // A class whose own rounds could not place a rival behind the leader is not left without an answer:
            // the entries they could not separate are re-run on their own at a longer protocol and voted on,
            // and the entry the vote names is the recommendation with the way it was reached recorded beside it.
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

        // The floor belongs to the run, not to this control: the run timed it once before any row was put
        // through a check, and the assignment below replaces the whole struct. Carrying it in here keeps one
        // report from saying a floor could not be timed while a line prints the floor as a measured zero.
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

    // One check per route, both under the protocol the figures were taken under: one entry timed at two
    // very different repetition counts must give the same per-argument figure, since a cost that
    // repeats with the launch rather than with the call would be amortised differently at each count.
    // The launched route carries the device-side floor beside its verdict as a diagnostic.
    {
        DeviceProbeMeasurement* fastestLaunched = nullptr;
        DeviceProbeMeasurement* fastestSubtracted = nullptr;

        for (DeviceProbeMeasurement& measurement : report.measurements)
        {
            // The two printed controls are taken over the row a default is read from: that is the row a reader
            // is most likely to quote, and running the check on some other row would leave the default's own
            // figure uncontrolled. Every other row is checked by the fixed-point loop below.
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

                // The floor is the per-launch cost of a kernel that does no arithmetic, so it holds the device's
                // launch and the host's submission of it together. It is not the entry's own launch cost: an
                // entry's kernel needs registers and an occupancy ramp the empty one never pays.
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

    // Every row the report names as a class's fastest or as its recommendation is put through the
    // repetition control before the report ships, and a row that does not agree is set aside and the
    // class falls to the next. Looped, because setting a row aside can name a new leader, and bounded
    // by the number of rows. The row is resolved by its index in the option table, not by its name.
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
/// The member is identified by its entry and its form, which is what one row of this
/// run's table is: a run measures one entry at one form once, so the pair and the place
/// are the same member. The name is deliberately not the key — a name is what the
/// ranking concludes on and what a request selects by, and a member's identity is the
/// arithmetic it runs rather than the spelling a report gives it.
const DeviceProbeMeasurement* GridPlaceOf(const DeviceProbeReport& report,
                                          DeviceEntry entry,
                                          DivisionForm form) {
    for (const DeviceProbeMeasurement& place : report.measurements)
    {
        if (place.entry == entry && place.form == form)
        {
            return &place;
        }
    }

    return nullptr;
}

DeviceOptionClosure DeviceOptionSpaceClosure(const DeviceProbeReport& report) noexcept {
    const std::span<const DeviceOptionInfo> space = BoysDeviceOptions();
    const std::span<const DivisionFormInfo> forms = BoysDivisionForms();

    DeviceOptionClosure closure;
    closure.rows = space.size();
    closure.forms = forms.size();
    closure.total = closure.rows * closure.forms;

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
        // The row is one member per form the library reports, and every count below is
        // made once per member: the space this closure closes over is the cross and not
        // its first column.
        for (const DivisionFormInfo& member : forms)
        {
            // Whether this run's own request named the row. A request that names nothing
            // asks for the whole space; one that names rows asks for those. The name is
            // the member's own, so a request that named one form of an entry does not
            // stand behind the other two.
            const bool asked =
                report.options.only.empty() ||
                std::find(report.options.only.begin(),
                          report.options.only.end(),
                          ProbeRowName(row, member.form)) != report.options.only.end();

            // The order is the space's own: a row the build does not serve is refused with
            // the library's reason and owed, whatever the run did with the others, and it
            // is owed at every form — one absence is one member of the cross per form.
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

            // The card would not hold the degree tables: the member was never presented to
            // it, and that is this card's answer rather than the row's absence.
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

            const DeviceProbeMeasurement* place = GridPlaceOf(report, row.entry, member.form);

            if (place != nullptr)
            {
                (place->measured ? closure.measured : closure.offeredNoFigure) += 1;
                continue;
            }

            ++closure.unaccounted;
        }
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
                 "  include/boys/boys_cuda_options.hpp) crossed with the %zu division form(s) the\n"
                 "  library reports (BoysDivisionForms, boys/boys.hpp), one member per (row, form): "
                 "%zu\n  member(s). Every entry of this surface runs every form, so no cell of the "
                 "cross is\n  empty. Both counts are the library's own and neither is listed "
                 "here.\n",
                 closure.rows,
                 closure.forms,
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
                 "request owes\n                 the space (one place per member — row and form — it "
                 "serves and this\n                 request named)\n",
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
/// refusal with the library's reason. The report is the run's own measurement table, so the
/// comparison is against the report this run printed and not against a second reading of the same
/// source.
///
/// **The second column counts members and does not flag a row.** A row of the library's table is
/// one member of the cross per form the library reports, so the place this run carries for it is a
/// number: a row carried at all three forms, one carried at one and not the others, and one this
/// run carried no place for at all are three different facts, and a column that said "measured"
/// for the first two would be reporting a projection of this run rather than this run.
void AppendOptionSpace(std::string& text, const DeviceProbeReport& report) {
    const std::span<const DeviceOptionInfo> space = BoysDeviceOptions();
    const std::span<const DivisionFormInfo> forms = BoysDivisionForms();
    std::size_t served = 0;
    std::size_t refused = 0;
    std::size_t launched = 0;
    std::size_t inKernel = 0;

    for (const DeviceOptionInfo& option : space)
    {
        (option.built ? served : refused) += 1;
        (option.group == DeviceOptionGroup::kLaunched ? launched : inKernel) += 1;
    }

    const std::size_t members = space.size() * forms.size();

    // The places this run carried a figure for, and the ones it carried and got no figure
    // from: a place is a member the run offered to the device, and the two are different
    // facts about it.
    std::size_t measuredPlaces = 0;
    std::size_t emptyPlaces = 0;

    for (const DeviceProbeMeasurement& measurement : report.measurements)
    {
        (measurement.measured ? measuredPlaces : emptyPlaces) += 1;
    }

    text += "\n\nthe option space - the library's own report of it, crossed with the division forms "
            "the\n  library reports, and the place this run carries for each member\n";

    if (report.measurements.empty())
    {
        text += Text("  (BoysDeviceOptions, include/boys/boys_cuda_options.hpp): %zu option(s) = %zu "
                     "launched + %zu\n  device-callable, %zu this build serves and %zu it refuses; "
                     "the entry's own row\n  crossed with the %zu division form(s) of "
                     "BoysDivisionForms(): %zu member(s). No\n  place was measured on this run, so "
                     "the second column says so rather than naming one.\n",
                     space.size(), launched, inKernel, served, refused, forms.size(), members);
    } else
    {
        text += Text("  (BoysDeviceOptions, include/boys/boys_cuda_options.hpp): %zu option(s) = %zu "
                     "launched + %zu\n  device-callable, %zu this build serves and %zu it refuses; "
                     "the entry's own row\n  crossed with the %zu division form(s) of "
                     "BoysDivisionForms(): %zu member(s); %zu place(s)\n  measured here\n",
                     space.size(), launched, inKernel, served, refused, forms.size(), members,
                     measuredPlaces);
    }

    text += Text("  One member per (row, form) of those tables, and %zu of the %zu place(s) this run "
                 "carried\n  produced no figure here.\n",
                 emptyPlaces,
                 report.measurements.size());

    text += "  report row                 forms carried of 3          group     precision  shape       question    axes                                                                                                  lane        bound     documented form\n";

    for (const DeviceOptionInfo& option : space)
    {
        // How many of the row's own members this run carried a place for, and how many of
        // those produced a figure: the row's entry and the member's form are the member's
        // identity, so the count is read off the run's own table and not off its names.
        std::size_t places = 0;
        std::size_t withAFigure = 0;

        for (const DivisionFormInfo& member : forms)
        {
            for (const DeviceProbeMeasurement& place : report.measurements)
            {
                if (place.entry == option.entry && place.form == member.form)
                {
                    ++places;
                    withAFigure += place.measured ? 1u : 0u;
                }
            }
        }

        std::string carried;

        if (places == 0)
        {
            carried = "not carried by this run";
        } else if (places < forms.size())
        {
            carried = Text("%zu of %zu carried here", places, forms.size());
        } else
        {
            carried = Text("%zu of %zu measured here", withAFigure, forms.size());
        }

        text += Text("  %-26s %-26s %-9s %-10s %-11s %-11s %-101s %-11s %-9.2g %s\n",
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

    // The other direction: a member this run carries that the library's report is not under. The entry
    // is what the library's table names - its own name carries the form's segment where the row does
    // not run the build's default form - and a name is looked for once, one fact per missing entry.
    std::vector<std::string> unreported;

    for (const DeviceProbeMeasurement& place : report.measurements)
    {
        if (std::find(unreported.begin(), unreported.end(), place.name) != unreported.end())
        {
            continue;
        }

        bool reported = false;

        for (const DeviceOptionInfo& option : space)
        {
            reported = reported || option.entry == place.entry;
        }

        if (!reported)
        {
            unreported.push_back(place.name);
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

/// The division-form cell of a row: the form the row's ladder steps divide in, as the
/// default-policy table spells it.
///
/// Read off the winning row and never listed beside it: a member of the option space is an entry
/// crossed with a form (\c DeviceProbeMeasurement::form, boys_cuda_probe.hpp), and the figure the
/// seam row carries was taken at the form that member states, so the cell follows the arithmetic
/// the measurement ran rather than a form chosen where the row is written. An entry's own
/// \c DeviceOptionInfo::division is \c DeviceEntryAxesOf's statement of the form its body performs
/// its divisions in, which is the build's default (\c kDefaultDivisionForm, boys_cuda_options.hpp)
/// and is the member of the cross that carries no form segment - so reading that field here would
/// write the default under a figure measured at another form.
///
/// \returns the cell, or \c nullptr for a value outside \c DivisionForm's enumerators - which the
///          crossing states for no member, since it reads the axis off \c BoysDivisionForms
constexpr const char* SeamDivisionCell(DivisionForm form) noexcept {
    switch (form)
    {
        case DivisionForm::kExactDivision:
            return "DivisionForm::kExactDivision";
        case DivisionForm::kPlainReciprocal:
            return "DivisionForm::kPlainReciprocal";
        case DivisionForm::kRefinedReciprocal:
            return "DivisionForm::kRefinedReciprocal";
    }

    return nullptr;
}

/// The exponential cell of a row: the member of the region-B axis the winning entry carries.
///
/// It is read off the library's own row for that entry and not written from the build's default,
/// for the reason the division-form cell above gives: where an entry varies this axis, the member
/// it varies is part of that entry's identity (\c DeviceOptionInfo::regionBExp, which
/// \c DeviceOptionAxis::kRegionBExp enumerates), so a cell holding the default would stand under
/// a figure measured at another member. Where an entry varies no member, the table states
/// \c RegionBExp::kAccurate on its row — that field's own reading for a row with no axis, which
/// is also this lane's default member (\c kDefaultRegionBExp, boys/boys_device_tables.hpp) and
/// what the device's accuracy gate pairs such a row with. So the cell is the table's statement
/// for the row in both cases, and this probe never writes a member the table did not state.
///
/// \param exp the member the winning entry's row in the library's table carries
///
/// \returns the cell, or \c nullptr for a value outside \c RegionBExp's enumerators - which the
///          crossing states for no member, since it reads the axis off the rows themselves
constexpr const char* SeamExpCell(RegionBExp exp) noexcept {
    switch (exp)
    {
        case RegionBExp::kAccurate:
            return "RegionBExp::kAccurate";
        case RegionBExp::kFast:
            return "RegionBExp::kFast";
    }

    return nullptr;
}

/// The precision cell of a row: the lane the class runs in, as the default-policy table spells
/// it.
///
/// The device's four precisions are four lanes of the table and not one lane named four ways: a
/// row is keyed by (device, precision, shape), so a table that gave the device half one
/// precision cell would carry one class per shape and none of the other three precisions'
/// shapes. The twelve classes the probe ranks - four precisions by three shapes - need the four
/// cells below, and each names the lane whose entries that class's winner is an entry of
/// (boys/boys_device_tables.hpp, \c BoysDeviceLane, which is where the device's own lanes are
/// enumerated).
///
/// The half lane's two stores are two cells for the reason the host's two are two classes: the
/// class is keyed by the format a return carries, so an fp16 recommendation and a bf16 one are
/// two answers and a table that folded them into one cell would write a bf16 entry's name into
/// the fp16 class's row - the defect this key exists to make impossible.
///
/// \param precision the class's precision, as the probe's option table names it
///
/// \returns the cell, or \c nullptr for an enumerator this switch has not been taught - which
///          the assertions below turn into a compile error rather than a row quietly keyed to
///          another lane's class
constexpr const char* SeamPrecisionCell(DeviceOptionPrecision precision) noexcept {
    switch (precision)
    {
        case DeviceOptionPrecision::kFp64:
            return "kFp64Device";
        case DeviceOptionPrecision::kFp32:
            return "kFp32Device";
        case DeviceOptionPrecision::kFp16:
            return "kFp16Device";
        case DeviceOptionPrecision::kBf16:
            return "kBf16Device";
        case DeviceOptionPrecision::kCount:
            break;
    }

    return nullptr;
}

/// One enumerator of \c DeviceOptionPrecision that \c SeamPrecisionCell spells no cell for.
template <DeviceOptionPrecision Precision> constexpr bool PrecisionCellIsNamed() noexcept {
    return SeamPrecisionCell(Precision) != nullptr;
}

static_assert(PrecisionCellIsNamed<DeviceOptionPrecision::kFp64>() &&
                  PrecisionCellIsNamed<DeviceOptionPrecision::kFp32>() &&
                  PrecisionCellIsNamed<DeviceOptionPrecision::kFp16>() &&
                  PrecisionCellIsNamed<DeviceOptionPrecision::kBf16>(),
              "an enumerator of DeviceOptionPrecision names no cell: name it in "
              "SeamPrecisionCell (src/boys_cuda_probe.cpp)");

/// The budget cell of a row: the budget the class's lane carries, which is the axis
/// \c BoysBudget names and the one \c LaneFallbackBudget (boys/boys.hpp) states for that lane.
///
/// The half lane is the one that moves it: the fp16 and bf16 device entries run the float
/// lane's bodies under the half budget, whose degree tables are not the float lane's, and every
/// other device lane carries the float budget. The cell is read from the lane rather than chosen
/// here, and a lane added to the device without a statement of its budget is a compile error
/// below rather than a row that carries the float budget by accident.
constexpr const char* SeamBudgetCell(DeviceOptionPrecision precision) noexcept {
    switch (precision)
    {
        case DeviceOptionPrecision::kFp64:
        case DeviceOptionPrecision::kFp32:
            return "BoysBudget::kFloat";
        case DeviceOptionPrecision::kFp16:
        case DeviceOptionPrecision::kBf16:
            return "BoysBudget::kFp16";
        case DeviceOptionPrecision::kCount:
            break;
    }

    return nullptr;
}

/// One enumerator of \c DeviceOptionPrecision that \c SeamBudgetCell spells no cell for.
template <DeviceOptionPrecision Precision> constexpr bool BudgetCellIsNamed() noexcept {
    return SeamBudgetCell(Precision) != nullptr;
}

static_assert(BudgetCellIsNamed<DeviceOptionPrecision::kFp64>() &&
                  BudgetCellIsNamed<DeviceOptionPrecision::kFp32>() &&
                  BudgetCellIsNamed<DeviceOptionPrecision::kFp16>() &&
                  BudgetCellIsNamed<DeviceOptionPrecision::kBf16>(),
              "an enumerator of DeviceOptionPrecision names no budget cell: name it in "
              "SeamBudgetCell (src/boys_cuda_probe.cpp)");

/// The packing cell of a row: the entry's own reading of region A, in the axis the seam names
/// \c PackAxis.
///
/// The two axes are not one axis, and this is where they meet: the device states how region A's
/// fits are read (\c DevicePacking, boys_cuda_options.hpp) and the seam states which axis a
/// packed lane vectorises over (\c PackAxis, boys/backend.hpp). The translation below is the one
/// the library's own words use - the ladder is "the per-argument ladder every one of those
/// entries has" and the per-order reading is "the orders reading of the same stored fits"
/// (src/boys.cpp, the fp32 device lane's carrier) - so a reader comparing this cell with
/// \c DeviceEntryAxesOf for the entry the row names finds one fact stated twice rather than two
/// facts:
///
///   * \c kLadder, the top order's fit seeded and every lower order brought back down the
///     recurrence, is a reading taken per argument, so its cell is \c PackAxis::kArguments;
///   * \c kPerOrder, every order's own fit located and summed where it lies, is the orders
///     reading of the same fits, so its cell is \c PackAxis::kOrders;
///   * \c kNotApplicable is the single-order shape's own statement that it fixes nothing on the
///     axis, and the axis a one-order call has is the arguments one - there is no second order
///     to pack - which is the cell the committed file gives every host single-order row.
///
/// The derivation is the entry's own (\c DeviceEntryAxesOf) and never the entry's name: the
/// names of this space spell "orders" for the per-order rows, and a cell written from a name
/// would be a statement about a string rather than about the arithmetic the row compiles.
///
/// \param packing the entry's own reading of region A
///
/// \returns the cell, or \c nullptr for an enumerator this switch has not been taught
constexpr const char* SeamPackCell(DevicePacking packing) noexcept {
    switch (packing)
    {
        case DevicePacking::kLadder:
            return "PackAxis::kArguments";
        case DevicePacking::kPerOrder:
            return "PackAxis::kOrders";
        case DevicePacking::kNotApplicable:
            return "PackAxis::kArguments";
        case DevicePacking::kUnstated:
            // Never a row's value: DeviceEntryAxesOf answers it for an entry its switch has not
            // been taught, which that header's own assertion turns into a compile error. A row
            // written from it would state the argument axis for an entry that stated none.
            break;
    }

    return nullptr;
}

/// One enumerator of \c DevicePacking that \c SeamPackCell spells no cell for. \c kUnstated is
/// excluded: it is not a reading and never a row's value (boys_cuda_options.hpp).
template <DevicePacking Packing> constexpr bool PackingCellIsNamed() noexcept {
    return SeamPackCell(Packing) != nullptr;
}

static_assert(PackingCellIsNamed<DevicePacking::kLadder>() &&
                  PackingCellIsNamed<DevicePacking::kPerOrder>() &&
                  PackingCellIsNamed<DevicePacking::kNotApplicable>(),
              "an enumerator of DevicePacking names no cell: name it in SeamPackCell "
              "(src/boys_cuda_probe.cpp)");

// The one row list this run read, expanded into the table in force's own rows: each entry keeps the
// class it carries and the row as the tokens that class's file wrote it with, so a row of the
// committed file reaches the file this run writes exactly as it arrived.
#define BOYS_DEVICE_PROBE_SEAM_ROW(device, precision, shape, route, scheme, budget, pack,          \
                                   granularity, division, exp)                                     \
    {#device, #precision, #shape,                                                                  \
     "    X(" #device ", " #precision ", " #shape ", " #route ", " #scheme ", " #budget ", "       \
     #pack ", " #granularity ", " #division ", " #exp ")\\\n"},

/// One row of the table in force: the class it carries, as that table's own list spells it,
/// and the row itself, as that table's own list writes it.
struct SeamRow {
    const char* device;
    const char* precision;
    const char* shape;
    const char* text;
};

#if defined(BOYS_BUILD_DEFAULT_ROWS)
/// The rows the table in force carries, verbatim.
const SeamRow kTableRows[] = {BOYS_BUILD_DEFAULT_ROWS(BOYS_DEVICE_PROBE_SEAM_ROW)};
#else
// A build that carries only the seven names lists no row, so the base is empty. The one entry is
// the empty key rather than an empty array, which C++ does not have: a device name of "" is no
// class this run writes, so nothing of the base is written back.
const SeamRow kTableRows[] = {{"", "", "", ""}};
#endif

// Two steps, so a macro is stringized as its value and not as its own name.
#define BOYS_DEVICE_PROBE_STRING(text) #text
#define BOYS_DEVICE_PROBE_VALUE(text) BOYS_DEVICE_PROBE_STRING(text)

} // namespace

/// The classes of the device half of the seam, as the two axes the table is keyed on: the four
/// precisions this lane's entries are built at by the three questions the probe ranks, in the
/// table's own order. One statement of the twelve, read by the emitter and by the publisher
/// alike, so a class this list gains or loses is a class both write.
constexpr DeviceOptionPrecision kDevicePrecisions[] = {DeviceOptionPrecision::kFp64,
                                                       DeviceOptionPrecision::kFp32,
                                                       DeviceOptionPrecision::kFp16,
                                                       DeviceOptionPrecision::kBf16};
constexpr DeviceOptionQuestion kDeviceQuestions[] = {DeviceOptionQuestion::kSingle,
                                                     DeviceOptionQuestion::kAllOrders,
                                                     DeviceOptionQuestion::kAllN};

/// Whether a class's fastest group holds the row the class named: the row its own figures put
/// first, or one of the rows its rounds could not place behind that one. Nothing else in a
/// ranking is a measurement of a row that no ordering placed first, so this is the whole of what
/// such a row may claim - and it is one predicate rather than two, read by the emitter for the
/// marker it writes and by the publisher for the class it places, because the claim and the check
/// on it have to be the same statement.
bool InFastestGroup(const DeviceProbeRanking& ranking, const std::string& name) {
    if (name == ranking.fastestOverall)
    {
        return true;
    }

    for (const std::string& tied : ranking.inseparable)
    {
        if (name == tied)
        {
            return true;
        }
    }

    return false;
}

DeviceDefaultsEmission FormatDeviceBuildDefaults(const DeviceProbeReport& report,
                                                 const std::string& takenAt) {
    DeviceDefaultsEmission emission;

    std::string rows;
    bool measured = false;

    // The classes this run's own rows carry, as the seam spells them. The file this run writes
    // holds one row per class - the base's own for a class it did not measure, this run's for one
    // it did - so the base's list below is written minus these.
    struct SeamClass {
        const char* precision;
        const char* shape;
    };
    std::vector<SeamClass> writtenClasses;

    const auto thisRunWroteClass = [&writtenClasses](const char* precision, const char* shape) {
        for (const SeamClass& written : writtenClasses)
        {
            if (std::strcmp(written.precision, precision) == 0 &&
                std::strcmp(written.shape, shape) == 0)
            {
                return true;
            }
        }

        return false;
    };

    // One row per class the probe ranks: the four device precisions by the three questions, in the
    // table's own order. A class is a (device, precision, shape) triple, so the twelve rows below are
    // the twelve classes of the device half of the seam.
    for (const DeviceOptionPrecision precision : kDevicePrecisions)
    {
        for (const DeviceOptionQuestion question : kDeviceQuestions)
        {
            const char* const precisionCell = SeamPrecisionCell(precision);
            const std::string klass = Text("%s %s", precisionCell, QuestionName(question));
            const char* const shapeCell = SeamShapeCell(question);
            bool carried = false;

            // The table in force is the BASE this run writes over and not a fence: a class it already carries
            // is one this run replaces, and one it carries that this run did not measure is written back
            // verbatim. So the check here decides the marker and the list, never whether the row is written.
            for (const SeamRow& known : kTableRows)
            {
                carried = carried || (std::strcmp(known.device, "kDevice") == 0 &&
                                      std::strcmp(known.precision, precisionCell) == 0 &&
                                      std::strcmp(known.shape, shapeCell) == 0);
            }

            // The class this row would come from: this run's own class of this precision and this
            // question, which is the class the library's answer for this key is taken from. A row
            // states what the library picks for a class, so it is the entry that class's own
            // readings placed first.
            const DeviceProbeClass* founder = nullptr;

            for (const DeviceProbeClass& candidate : report.classes)
            {
                if (candidate.precision == PrecisionName(precision) &&
                    candidate.question == QuestionName(question))
                {
                    founder = &candidate;
                }
            }

            // What a refused class is left with: a class the table in force carries keeps that
            // file's own row, which the list below writes back verbatim, and one it does not carry
            // has no row in the file this run writes. The two are different answers and the reason
            // line says which one this is.
            const char* const stands =
                carried ? "; the table in force's own row for it is written back unchanged" : "";

            if (founder == nullptr)
            {
                emission.refused.push_back(Text("%s: this run carried no %s class of that question%s",
                                                klass.c_str(), PrecisionName(precision), stands));
                continue;
            }

            const std::string& winner = founder->ranking.recommended;

            if (winner.empty() || founder->ranking.verdict == DeviceProbeVerdict::kCannotDetermine)
            {
                emission.refused.push_back(
                    Text("%s: %s%s", klass.c_str(),
                         founder->ranking.reason.empty()
                             ? "the class's ranking named no entry and gave no reason"
                             : founder->ranking.reason.c_str(),
                         stands));
                continue;
            }

            // The row the class named: this run's own measurement of it, the row the ranking ordered and the
            // figure beside this seam row was taken at. The name is looked up in the run's own table and not
            // the library's, because a winner at a non-default form carries that form's segment.
            const DeviceProbeMeasurement* won = nullptr;

            for (const DeviceProbeMeasurement& candidate : report.measurements)
            {
                if (winner == candidate.name)
                {
                    won = &candidate;
                }
            }

            if (won == nullptr)
            {
                emission.refused.push_back(
                    Text("%s: the row it named, '%s', is no row of this run's own option table%s",
                         klass.c_str(), winner.c_str(), stands));
                continue;
            }

            // The library's own row for that member's entry: the route, the scheme and the
            // packing cell are the entry's, which is the library's statement of what it runs
            // rather than this run's.
            const DeviceOptionInfo* row = nullptr;

            for (const DeviceOptionInfo& candidate : BoysDeviceOptions())
            {
                if (candidate.entry == won->entry)
                {
                    row = &candidate;
                }
            }

            if (row == nullptr)
            {
                emission.refused.push_back(
                    Text("%s: the entry it named, '%s', is no row of the library's option space%s",
                         klass.c_str(), winner.c_str(), stands));
                continue;
            }

            // What a row of this class may claim, decided by how its winner was reached: an ordering placed
            // every rival behind it and the row is the winner of a comparison; a vote over entries the class
            // could not separate establishes less, and something; a walkover establishes neither. Three markers.
            const DeviceProbeDefaultHow reached = founder->ranking.defaultHow;
            const bool walkover = reached == DeviceProbeDefaultHow::kOnlyEntry;
            const bool ordered = reached == DeviceProbeDefaultHow::kOrdered;

            // The class's fastest group - the row its own figures put first and every row the run
            // could not place behind that one - is the whole of what a row reached without an
            // ordering may claim, so a row outside it is refused rather than written with a
            // marker no class of this run would make true of it.
            if (!ordered && !walkover && !InFastestGroup(founder->ranking, winner))
            {
                emission.refused.push_back(
                    Text("%s: the entry it named, '%s', is neither the fastest row of the class "
                         "nor one the class could not place behind it, so no marker of this file "
                         "is true of it%s",
                         klass.c_str(), winner.c_str(), stands));
                continue;
            }

            // The class's own resolution and the size of its fastest group, as the ranking states them: the two
            // figures a row reached without an ordering is stated with, and the evidence a reader of the file
            // has for it. Neither is recomputed here.
            const std::size_t group = founder->ranking.inseparable.size() + 1;
            const std::string at =
                founder->ranking.resolution > 0.0
                    ? Text("the %.2f%% it could order at", 100.0 * founder->ranking.resolution)
                    : std::string("a resolution it did not measure");

            // The packing cell is the entry's own reading of region A, translated once here (SeamPackCell).
            // DeviceEntryAxesOf answers kUnstated only for an entry its switch has not been taught, which that
            // header's own assertion turns into a compile error, so this arm reads a value that cannot arrive.
            const char* const packCell = SeamPackCell(row->packing);

            if (packCell == nullptr)
            {
                emission.refused.push_back(
                    Text("%s: the entry it named, '%s', states no reading of region A, so no packing "
                         "cell of the seam is this row's%s",
                         klass.c_str(), winner.c_str(), stands));
                continue;
            }

            // The division-form cell is the form the winning row was measured at (SeamDivisionCell), the
            // member's own coordinate and not the entry's field: reading DeviceOptionInfo::division here would
            // write the build's default under a figure measured at another form. A row stating none cannot arrive.
            const char* const formCell = SeamDivisionCell(won->form);

            if (formCell == nullptr)
            {
                emission.refused.push_back(
                    Text("%s: the row it named, '%s', states no division form, so no "
                         "division-form cell of the seam is this row's%s",
                         klass.c_str(), winner.c_str(), stands));
                continue;
            }

            // The region-B exponential cell is the entry's own coordinate in the library's table (SeamExpCell),
            // read off the library's row for the winning entry and not off the build's default. A row of this
            // space that varies no member of the axis carries the table's own statement for it.
            const char* const expCell = SeamExpCell(row->regionBExp);

            if (expCell == nullptr)
            {
                emission.refused.push_back(
                    Text("%s: the entry it named, '%s', states no region-B exponential, so no "
                         "region-B cell of the seam is this row's",
                         klass.c_str(), winner.c_str()));
                continue;
            }

            // A row written over one the table in force already carries says so in its own
            // marker: the emitted file is read on its own, and a row that silently replaced the
            // file's own would leave a reader comparing two files to find out what this run
            // changed.
            const char* const replaces =
                carried ? "; replaces the row the table in force carries" : "";

            // The marker is the claim, and a consumer of the file reads it beside the row. Every line of it is a
            // complete comment ending in the macro's own continuation, which is what the tool that splices these
            // rows reads a marker as (tools/splice_default_rows.py): a comment left open across two lines
            // detaches the provenance from the row.
            if (ordered)
            {
                rows += Text("    /* measured: '%s', reached by ordered; this class's own rounds "
                             "placed every rival of it behind it%s */\\\n",
                             winner.c_str(), replaces);
            } else if (walkover)
            {
                rows += Text("    /* a choice, not a measurement: '%s' was the last entry standing "
                             "in this class */\\\n",
                             winner.c_str());
                rows += Text("    /* the row is an answer and not the winner of a comparison%s */\\\n",
                             replaces);
            } else
            {
                rows += Text("    /* measured: '%s', reached by %s; no entry of this class was "
                             "measured faster than it */\\\n",
                             winner.c_str(), DeviceProbeDefaultHowName(reached));
                rows += Text("    /* the class could not be ordered: its %zu fastest entries could "
                             "not be separated at %s%s */\\\n",
                             group, at.c_str(), replaces);
            }

            rows += Text("    X(kDevice, %s, %s, %s, %s,\\\n"
                         "      %s, %s, %s,\\\n"
                         "      %s, %s)\\\n",
                         precisionCell, shapeCell, SeamRouteCell(row->route),
                         SeamSchemeCell(row->scheme), SeamBudgetCell(precision), packCell,
                         SeamGranularityCell(*row), formCell, expCell);

            measured = measured || !walkover;
            writtenClasses.push_back(SeamClass{precisionCell, shapeCell});

            // The emission's own list says the same thing the marker does, in the same three ways: a class the
            // run ordered is a measurement of a comparison, a class it could not order is a measurement of what
            // no entry beat, and a walkover is not a measurement at all.
            const std::string how =
                ordered
                    ? Text("reached by %s, a measurement of this class's own runs",
                           DeviceProbeDefaultHowName(reached))
                : walkover
                    ? std::string("the last entry standing - written as a choice and not as a "
                                  "measurement")
                    : Text("reached by %s; no entry of this class was measured faster than it, and "
                           "its %zu fastest entries could not be separated at %s",
                           DeviceProbeDefaultHowName(reached), group, at.c_str());

            emission.emitted.push_back(
                Text("%s: '%s', %s", klass.c_str(), winner.c_str(), how.c_str()));

            // What this row did to the table in force: a class that file already carries has had its
            // row replaced, and the replacement is listed here so the run's own edits to the shipped
            // table are read off the emission rather than found by comparing two files.
            if (carried)
            {
                emission.overridden.push_back(
                    Text("%s: now '%s', %s - the row the table in force carried for this class is "
                         "replaced by this one",
                         klass.c_str(), winner.c_str(), how.c_str()));
            }
        }
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
    text += "/// The device half of this build's default-policy seam, written by `boys-device-probe\n";
    text += "/// --emit-defaults <file>` from the classes this run measured. Each class it ranked\n";
    text += "/// carries the entry that won it, over the row the table in force wrote for it where it\n";
    text += "/// wrote one. A class that file carries and this run did not measure is written back\n";
    text += "/// unchanged: a replacement is read INSTEAD of the committed file, so dropping it would\n";
    text += "/// leave its callers with no row. A row is a measurement taken on one card, so the card,\n";
    text += "/// the run and the date are named below; the probe that writes it is\n";
    text += "/// `boys/boys_cuda_probe.hpp`.\n";
    text += "///\n";

    if (!takenAt.empty())
    {
        text += Text("/// The card and the run: %s\n///\n", takenAt.c_str());
    }

    text += "/// **One row per class, and a class is a (device, precision, shape) triple.** The rows\n";
    text += "/// below are the twelve classes of the device half: `kFp64Device`, `kFp32Device`,\n";
    text += "/// `kFp16Device` and `kBf16Device` - the four precisions this lane's entries are built\n";
    text += "/// at - by the three questions the probe ranks. A table keyed by the triple holds the\n";
    text += "/// twelve; one keyed by a single device precision cell holds one of them per shape, and\n";
    text += "/// one that folded the half lane's two stores into one cell would write a bf16 entry's\n";
    text += "/// name into the fp16 class's row.\n";
    text += "///\n";
    text += "/// **What a device row's cells are.** The route, the scheme and the packing are the\n";
    text += "/// entry's own (`DeviceEntryAxesOf`, boys_cuda_options.hpp), read from the lane and the\n";
    text += "/// body its kernels name; the granularity is the partition it reads; the budget is the\n";
    text += "/// lane's - the float budget on fp64 and fp32, the half budget on fp16 and bf16, whose\n";
    text += "/// degree tables are not the float lane's. The packing cell is the entry's own reading of\n";
    text += "/// region A in the axis `PackAxis`: the ladder - the top order's fit seeded and every lower\n";
    text += "/// order brought back down the recurrence, the per-argument reading - is\n";
    text += "/// `PackAxis::kArguments`; the per-order reading is `PackAxis::kOrders`, and a\n";
    text += "/// single-order class carries the arguments axis, because a call that produces one order\n";
    text += "/// has no second order to pack. The division-form cell is the form the winning row was\n";
    text += "/// measured at, and not the entry's `DeviceOptionInfo::division`: the axis is\n";
    text += "/// `DeviceOptionAxis::kDivision` on this lane and `DivisionForm` in the seam. Every entry\n";
    text += "/// runs every form, so a row is an (entry, form) pair. The region-B exponential is the\n";
    text += "/// winning entry's own coordinate on `RegionBExp` (`DeviceOptionInfo::regionBExp`): the\n";
    text += "/// member its recurrence seeds with where the entry varies the axis, the lane's own\n";
    text += "/// default where it varies none. Either cell written from the build's default would put a\n";
    text += "/// figure measured at another member under this row.\n";
    text += "///\n";
    text += "/// **A row's marker states what the class's own rounds established, and no more.**\n";
    text += "/// Three statements are possible and the marker carries the one that is true of its\n";
    text += "/// row. A class an ordering placed an entry first in carries the marker of a\n";
    text += "/// comparison: every rival of the row was measured slower than it. A class whose own\n";
    text += "/// rounds could not separate its top entries carries a marker of what WAS measured -\n";
    text += "/// that no entry of the class was measured faster than the row named - together with\n";
    text += "/// the size of that top group and the ratio band the class could order inside, because\n";
    text += "/// which member of the group is fastest is what the run did not establish. A class\n";
    text += "/// holding one entry, or one the run's own checks left alone, carries a choice and not\n";
    text += "/// a measurement. A row whose class established none of the three is refused rather\n";
    text += "/// than written. A class this run did not measure keeps the row the table in force\n";
    text += "/// carries, written back by the list below.\n";
    text += "\n";
    text += "/// The seven a class the list below carries no row for resolves to: the build's own\n";
    text += "/// values, which is what this build compiled before this file existed. Five are the\n";
    text += "/// host lane's and two are the device lane's own, and a class of the device lane\n";
    text += "/// carries the device pair and not the host's.\n";
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
    text += "/// The device lane's two, which are the host's names' own siblings: a device class\n";
    text += "/// resolves its division form and its region-B exponential to these and never to the\n";
    text += "/// host's, so an unnamed device call reads this lane's own seam (boys_cuda.hpp,\n";
    text += "/// accuracy.hpp, boys_device_tables.hpp).\n";
    text += Text("#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM));
    text += Text("#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP %s\n\n",
                 BOYS_DEVICE_PROBE_VALUE(BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP));

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    text += "/// The classes this file sets a default for: **one row per class**, in the table's own\n";
    text += "/// format. First the table in force's own rows, for the classes this run did not measure\n";
    text += "/// and written back verbatim; then this run's, one per class it measured, over the row\n";
    text += "/// that table carried where it carried one.\n";
    text += "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n";

    for (const SeamRow& base : kTableRows)
    {
        // The base is written over and not beside: a class this run measured is carried by this
        // run's own row below, and writing the base's row for it as well would be two rows for one
        // class, which is two explicit specializations of one template and does not compile.
        if (std::strcmp(base.device, "kDevice") == 0 &&
            thisRunWroteClass(base.precision, base.shape))
        {
            continue;
        }

        text += base.text;
    }
#else
    // A build that carries only the seven names has no classes to write back: the rows below are
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

namespace {

/// The report's lines, with the carriage return a console redirect may leave at the end of one
/// taken off: the report's own text is written with '\n' and this reads it back from a file, and
/// a reader that kept the '\r' would match none of the markers below.
std::vector<std::string> ReportLines(const std::string& text) {
    std::vector<std::string> lines;
    std::size_t at = 0;

    for (;;)
    {
        const std::size_t end = text.find('\n', at);
        std::string line = text.substr(at, end == std::string::npos ? end : end - at);

        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }

        lines.push_back(std::move(line));

        if (end == std::string::npos)
        {
            return lines;
        }

        at = end + 1;
    }
}

/// Whether a line opens with the report's own marker for what it states. A marker is the
/// report's wording and not a name invented here, so a report that says something else is
/// refused rather than matched to the nearest marker.
bool OpensWith(const std::string& line, const char* marker) {
    return line.rfind(marker, 0) == 0;
}

/// One class's block of the report: the lines from the `class` heading for it to the next such
/// heading, which is where one class ends and the next begins. The report's own layout, read as
/// it writes it rather than by counting lines.
bool ClassBlock(const std::vector<std::string>& lines, const std::string& heading, std::size_t& first,
                std::size_t& last) {
    for (std::size_t at = 0; at < lines.size(); ++at)
    {
        if (lines[at] != heading)
        {
            continue;
        }

        first = at + 1;
        last = lines.size();

        for (std::size_t next = first; next < lines.size(); ++next)
        {
            if (OpensWith(lines[next], "class "))
            {
                last = next;
                break;
            }
        }

        return true;
    }

    return false;
}

/// Every line of a block that states what \p marker introduces, as the value the report writes
/// after the marker. Zero, one and more than one of them are three different facts to a reader,
/// so the count is the caller's to read and this does not collapse them.
std::vector<std::string> StatedValues(const std::vector<std::string>& lines, std::size_t first,
                                      std::size_t last, const char* marker) {
    std::vector<std::string> values;

    for (std::size_t at = first; at < last; ++at)
    {
        if (OpensWith(lines[at], marker))
        {
            values.push_back(lines[at].substr(std::strlen(marker)));
        }
    }

    return values;
}

/// Several statements of what one class is wrong with, as one line.
std::string SemicolonJoined(const std::vector<std::string>& parts) {
    std::string joined;

    for (const std::string& part : parts)
    {
        if (!joined.empty())
        {
            joined += "; ";
        }

        joined += part;
    }

    return joined;
}

/// The library's own row and division form a name the report prints belongs to.
///
/// The name is not taken apart: the grammar that produced it is \c ProbeRowName's, which is the
/// one statement of what a name is, so a name is resolved by running that function over the
/// library's own option table and form axis rather than by looking for a form's segment at the
/// end of it. A name no row of that cross prints is a name this build does not serve, which is a
/// refusal and not a row.
bool RowNamed(const std::string& name, DeviceEntry& entry, DivisionForm& form) {
    for (const DeviceOptionInfo& option : BoysDeviceOptions())
    {
        for (const DivisionFormInfo& member : BoysDivisionForms())
        {
            if (ProbeRowName(option, member.form) != name)
            {
                continue;
            }

            entry = option.entry;
            form = member.form;
            return true;
        }
    }

    return false;
}

/// The way the report says an entry was reached, as the one token it prints for it.
///
/// A class the refinement stage decided states the way on its `reason:` line, and it states it in
/// the enumeration's own names (\c DeviceProbeDefaultHowName), so a token this reads is the name
/// a writer would have written. Empty where the report states none: a class the stage never
/// reached states its way on the `reached by:` line instead, and one whose ranking carries a
/// value outside the enumeration states none at all.
std::string StatedHowToken(const std::vector<std::string>& lines, std::size_t first,
                           std::size_t last) {
    const char* const kMarker = "named with the way it was reached - ";

    for (const std::string& reason : StatedValues(lines, first, last, "    reason: "))
    {
        const std::size_t at = reason.find(kMarker);

        if (at == std::string::npos)
        {
            continue;
        }

        std::string token = reason.substr(at + std::strlen(kMarker));
        const std::size_t stop = token.find('.');

        if (stop != std::string::npos)
        {
            token.erase(stop);
        }

        return token;
    }

    return std::string();
}

/// The card the report names, as the report's own `device:` line states it, or empty where the
/// report names none. A row is a figure taken on one card, and the file a report is published
/// into says which.
std::string StatedCard(const std::vector<std::string>& lines) {
    for (const std::string& line : lines)
    {
        if (OpensWith(line, "  device: "))
        {
            return line.substr(std::strlen("  device: "));
        }
    }

    return std::string();
}

/// The run the report came from, as the report's own opening line states it, or empty where the
/// report carries none.
std::string StatedRunStart(const std::vector<std::string>& lines) {
    for (const std::string& line : lines)
    {
        if (OpensWith(line, "started "))
        {
            return line.substr(std::strlen("started "));
        }
    }

    return std::string();
}

/// The way one class's entry was reached, as the report states it, and where the report states
/// none, the sentence saying so. One value and not two, because a class whose way is unstated is
/// a class no row is written for.
struct StatedHow {
    DeviceProbeDefaultHow how = DeviceProbeDefaultHow::kNone;
    std::string refusal; ///< empty where the report states a way
};

/// The report's own two statements of how a class's entry was reached, read together.
///
/// The report states it in one of two places and each place carries the arms the other does not:
/// the `reached by:` line names the shape's own ordering and the lone-entry case, and a class the
/// refinement stage decided carries that line's statement that the way is not one its vocabulary
/// names, with the way itself on the `reason:` line. So both are read, and the two have to agree:
/// a report whose two lines name different ways is refused rather than placed under either.
StatedHow HowStatedBy(const std::vector<std::string>& lines, std::size_t first, std::size_t last) {
    StatedHow stated;

    const std::vector<std::string> reached = StatedValues(lines, first, last, "  reached by: ");

    if (reached.size() != 1)
    {
        stated.refusal = reached.empty() ? "the report states no 'reached by:' line for it"
                                         : "the report states more than one 'reached by:' line for it";
        return stated;
    }

    const std::string& line = reached.front();
    const std::string token = StatedHowToken(lines, first, last);

    // The arms the line has, as the report's own words: the shape's own ordering, the shape holding one
    // entry, the shape holding no rival this run could place, the refinement stage's block, and the
    // line's own statement that the ranking carries a value its vocabulary does not name.
    struct Arm {
        const char* opens;         ///< how the report's line opens
        DeviceProbeDefaultHow how; ///< the way it names, where it names one
        bool namesIt;              ///< whether the line is itself the statement of the way
    };

    const Arm kArms[] = {
        {"a measured ordering on this shape's own rounds", DeviceProbeDefaultHow::kOrdered, true},
        {"the shape holding one entry.", DeviceProbeDefaultHow::kOnlyEntry, true},
        {"the shape holding no rival this run could place.", DeviceProbeDefaultHow::kOnlyEntry, true},
        {"the refinement stage", DeviceProbeDefaultHow::kNone, false},
        {"(not a way this revision names)", DeviceProbeDefaultHow::kNone, false},
    };

    // The one arm that is not a way at all: the shape named no entry, so there is no name for a
    // row to be written from whatever the class's own lines say elsewhere.
    if (OpensWith(line, "nothing "))
    {
        stated.refusal = "the report says the shape named no entry";
        return stated;
    }

    for (const Arm& arm : kArms)
    {
        if (!OpensWith(line, arm.opens))
        {
            continue;
        }

        if (arm.namesIt)
        {
            if (!token.empty() && token != DeviceProbeDefaultHowName(arm.how))
            {
                stated.refusal = Text("the report's 'reached by:' line names '%s' and its "
                                      "'reason:' line names '%s'; the two lines disagree",
                                      DeviceProbeDefaultHowName(arm.how), token.c_str());
                return stated;
            }

            stated.how = arm.how;
            return stated;
        }

        // The way is on the reason line here, and only a way the refinement stage reaches can be:
        // a report naming one of the two the line names itself has contradicted itself.
        if (token.empty())
        {
            stated.refusal =
                Text("the report states no way its entry was reached: the 'reached by:' line "
                     "names none this revision defines and the 'reason:' line names none either");
            return stated;
        }

        for (int at = 0; at < static_cast<int>(DeviceProbeDefaultHow::kCount); ++at)
        {
            const DeviceProbeDefaultHow candidate = static_cast<DeviceProbeDefaultHow>(at);

            if (token != DeviceProbeDefaultHowName(candidate))
            {
                continue;
            }

            if (candidate != DeviceProbeDefaultHow::kRefined &&
                candidate != DeviceProbeDefaultHow::kVote &&
                candidate != DeviceProbeDefaultHow::kChosenAmongEquals)
            {
                stated.refusal = Text("the report's 'reached by:' line names no way this revision "
                                      "defines and its 'reason:' line names '%s'; the two lines "
                                      "disagree",
                                      token.c_str());
                return stated;
            }

            stated.how = candidate;
            return stated;
        }

        stated.refusal = Text("the report names '%s' as the way its entry was reached, and this "
                              "revision's enumeration defines no way of that name",
                              token.c_str());
        return stated;
    }

    stated.refusal = Text("the report's 'reached by:' line states no way this revision defines: it "
                          "reads '%s'",
                          line.c_str());
    return stated;
}

/// What one class states about the rows its own rounds could not order: the band it could not
/// order inside, the rows it lists as ones it left unplaced, and where the block states neither,
/// the sentence saying why it could not be read.
struct StatedResolution {
    double fraction = 0.0;             ///< the band's width as a fraction of a cost
    std::vector<std::string> unplaced; ///< the rows it left unplaced, as the report states them
    std::string refusal;               ///< empty where the block could be read
};

/// The report's own two statements of what a class could not order.
///
/// A class that measured a resolution prints it as a percentage of a ratio band, and states one
/// `not separable:` line per row its own rounds left unplaced against its leader. A class whose
/// resolution was too short to measure states no percentage, and that is not a fault: the marker
/// says the same thing either way. Two percentages are: one block states one class, and two bands
/// would leave a reader unable to tell which of them the row beside them was written under.
StatedResolution ResolutionStatedBy(const std::vector<std::string>& lines, std::size_t first,
                                    std::size_t last) {
    const char* const kBand =
        "    resolution: entries whose within-round ratio band is narrower than ";

    StatedResolution stated;
    const std::vector<std::string> bands = StatedValues(lines, first, last, kBand);

    if (bands.size() > 1)
    {
        stated.refusal = Text("the report states %zu 'resolution:' lines for it", bands.size());
        return stated;
    }

    if (bands.size() == 1)
    {
        stated.fraction = std::strtod(bands.front().c_str(), nullptr) / 100.0;
    }

    stated.unplaced = StatedValues(lines, first, last, "    not separable: ");

    // The line states a row and then its figures, so what the membership tests below need is the
    // quoted name and not the sentence: keeping the sentence made each test compare a row's name
    // against a sentence about it and find nothing, which read as six device classes whose
    // recommended entry no marker of the seam is true of.
    for (std::string& value : stated.unplaced)
    {
        const std::size_t opening = value.find('\'');

        if (opening == std::string::npos)
        {
            continue;
        }

        const std::size_t closing = value.find('\'', opening + 1);

        if (closing != std::string::npos)
        {
            value = value.substr(opening + 1, closing - opening - 1);
        }
    }

    return stated;
}

} // namespace

DeviceDefaultsPublication PublishDeviceBuildDefaults(const std::string& reportText) {
    DeviceDefaultsPublication publication;

    const std::vector<std::string> lines = ReportLines(reportText);

    // The card and the run the report names, and not this process's moment: the rows of the file
    // are that run's, and dating them to their publishing would name a run that measured nothing.
    const std::string card = StatedCard(lines);
    const std::string started = StatedRunStart(lines);
    const std::string takenAt =
        card.empty() ? started : (started.empty() ? card : card + ", " + started);

    DeviceProbeReport report;

    for (const DeviceOptionPrecision precision : kDevicePrecisions)
    {
        for (const DeviceOptionQuestion question : kDeviceQuestions)
        {
            const std::string klass = Text("%s %s", SeamPrecisionCell(precision),
                                           QuestionName(question));
            const std::string key =
                Text("%s, %s", PrecisionName(precision), QuestionName(question));
            std::vector<std::string> faults;
            std::size_t first = 0;
            std::size_t last = 0;

            // The heading the report writes for a class is the library's own key for it, so a
            // report of this revision's classes is read without a name being invented here.
            if (!ClassBlock(lines, Text("class %s", key.c_str()), first, last))
            {
                faults.push_back(Text("the report states no '%s' class", key.c_str()));
            }

            std::string recommended;

            if (faults.empty())
            {
                const std::vector<std::string> named =
                    StatedValues(lines, first, last, "    recommended entry: ");

                if (named.size() != 1)
                {
                    faults.push_back(
                        named.empty()
                            ? "the report states no 'recommended entry:' line for it"
                            : Text("the report states %zu 'recommended entry:' lines for it",
                                   named.size()));
                }
                else
                {
                    recommended = named.front();
                }
            }

            DeviceProbeDefaultHow how = DeviceProbeDefaultHow::kNone;

            if (faults.empty())
            {
                // The verdict the class's own ranking reached, which a row is written from only
                // where the run recommended one: a block stating CANNOT DETERMINE prints the name
                // it could not act on beside it, and a row written from that name would be a
                // default the run declined to name.
                const std::vector<std::string> verdicts =
                    StatedValues(lines, first, last, "    verdict: ");

                if (verdicts.size() != 1)
                {
                    faults.push_back(verdicts.empty()
                                         ? "the report states no 'verdict:' line for it"
                                         : Text("the report states %zu 'verdict:' lines for it",
                                                verdicts.size()));
                }
                else if (verdicts.front() != "RECOMMEND")
                {
                    faults.push_back(Text("the report's verdict for it is '%s', so the class named "
                                          "no entry the run recommended",
                                          verdicts.front().c_str()));
                }
            }

            if (faults.empty())
            {
                const StatedHow stated = HowStatedBy(lines, first, last);

                if (stated.refusal.empty())
                {
                    how = stated.how;
                } else
                {
                    faults.push_back(stated.refusal);
                }
            }

            DeviceEntry entry = DeviceEntry::kSingleF64;
            DivisionForm form = kDefaultDivisionForm;
            StatedResolution resolution;
            std::string fastestMeasured;

            if (faults.empty())
            {
                resolution = ResolutionStatedBy(lines, first, last);

                if (!resolution.refusal.empty())
                {
                    faults.push_back(resolution.refusal);
                }
            }

            if (faults.empty())
            {
                // The row the class's own figures put first, as the report states it: a row reached without an
                // ordering is written with a claim about this row and the group it leads, and the claim has to be
                // checkable against the report rather than assumed of it.
                const std::vector<std::string> fastest =
                    StatedValues(lines, first, last, "    fastest measured: ");

                if (fastest.size() > 1)
                {
                    faults.push_back(
                        Text("the report states %zu 'fastest measured:' lines for it",
                             fastest.size()));
                }
                else if (fastest.size() == 1)
                {
                    fastestMeasured = fastest.front();
                }
            }

            if (faults.empty() && !RowNamed(recommended, entry, form))
            {
                faults.push_back(Text("'%s' is no row of the library's own option space at any "
                                      "division form this build carries",
                                      recommended.c_str()));
            }

            if (!faults.empty())
            {
                publication.refusals.push_back(Text("%s (the report's '%s' class): %s", klass.c_str(),
                                                    key.c_str(), SemicolonJoined(faults).c_str()));
                continue;
            }

            DeviceProbeClass clause;
            clause.precision = PrecisionName(precision);
            clause.question = QuestionName(question);
            clause.ranking.recommended = recommended;
            clause.ranking.defaultHow = how;
            clause.ranking.verdict = DeviceProbeVerdict::kRecommend;

            // What the class states it could not order, and the row its own figures put first,
            // carried into the ranking: the marker a row reached without an ordering is written
            // with claims about those, and both are read from the report's own lines rather than
            // recomputed here.
            clause.ranking.fastestOverall = fastestMeasured;
            clause.ranking.resolution = resolution.fraction;
            clause.ranking.inseparable = resolution.unplaced;

            // The membership that marker claims, checked against the report rather than assumed
            // of it: a report naming a default that neither its own fastest row leads nor its own
            // rounds left unplaced states no row this file could write a true marker for.
            if (how != DeviceProbeDefaultHow::kOrdered && how != DeviceProbeDefaultHow::kOnlyEntry &&
                !InFastestGroup(clause.ranking, recommended))
            {
                publication.refusals.push_back(
                    Text("%s (the report's '%s' class): its 'recommended entry:' line names '%s', "
                         "which is neither the row its 'fastest measured:' line names nor one of "
                         "the rows its own rounds left unplaced, so no marker of the seam is true "
                         "of it",
                         klass.c_str(), key.c_str(), recommended.c_str()));
                continue;
            }

            report.classes.push_back(std::move(clause));

            // The one row of the run's own table the emitter reads, filled from the name: the
            // entry and the form it was resolved to, so the seam's cells are the entry's own and
            // the division-form cell is the form the row the report ranked was measured at.
            DeviceProbeMeasurement row;
            row.name = recommended;
            row.precision = PrecisionName(precision);
            row.question = QuestionName(question);
            row.entry = entry;
            row.form = form;
            report.measurements.push_back(std::move(row));
        }
    }

    // Published whole or not at all. The file is read INSTEAD of the committed seam, so one that
    // carried eleven rows would leave the twelfth class resolving to a row this report did not
    // state; the splice the consumer runs refuses the same file for the same reason, by position.
    if (!publication.refusals.empty())
    {
        return publication;
    }

    publication.emission = FormatDeviceBuildDefaults(report, takenAt);

    // The emitter's own refusals, which a report that placed every class reaches only where an
    // axis the seam names carries a member this revision's tables answer nothing for. Nothing is
    // written for as long as one of those stands either.
    if (!publication.emission.refused.empty())
    {
        publication.refusals = publication.emission.refused;
        publication.emission = DeviceDefaultsEmission{};
        return publication;
    }

    publication.complete = true;
    return publication;
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

        // The space is a fact about the option the entry names and not about this host, so it is stated even
        // when no figure could be taken: a reader of a failed run learns which options exist and what is
        // missing is the measurement.
        AppendOptionSpace(text, report);
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
            "difference is not positive at\nthe lower quartile of its own rounds - the quantile "
            "the two readings beside it are\n  stated at - because what such a row would print "
            "there is the instrument's floor\n  and not a cost.\n";

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
    // One per route, since the two routes are two methods and a check of one says nothing about the
    // other. The floor is the launched route's own: a subtraction has no launch of this library's inside
    // either half.
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
    // One class per precision and question shape: neither is traded for speed here, and what varies
    // inside a class is what the library picks on the caller's behalf plus the division form.
    if (!report.classes.empty())
    {
        text += Text("\nrankings - one class per precision and question shape.\n"
                     "  A class is what the caller has already fixed when they make the call: how "
                     "much\n  precision the result needs, and what they are asking for. This run's "
                     "option table\n  carries %zu row(s) across %zu class(es), and no entry of one "
                     "class was ordered against\n  an entry of another. What varies inside a class "
                     "is what the library picks on the\n  caller's behalf, and the division form, "
                     "which the caller may name and the build's\n  own default supplies when they "
                     "name none.\n",
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
            case DeviceProbeDefaultHow::kRefined:
            case DeviceProbeDefaultHow::kVote:
            case DeviceProbeDefaultHow::kChosenAmongEquals:
                // The three ways the refinement stage reaches, and ways this revision names: the block below the
                // switch is their statement. Falling through to it is the whole of this arm - every class the stage
                // decided took the arm below instead and had its way withheld from a reader.
                break;
            case DeviceProbeDefaultHow::kNone:
            case DeviceProbeDefaultHow::kCount:
                // The two values no stage reaches: the enumeration's own value for a shape that reached nothing,
                // which the empty-name line above already answers, and the enumerator that closes the enumeration.
                // A value cast in from outside arrives here too.
                text += "\n  reached by: (not a way this revision names) — this shape's ranking "
                        "names no way\n    its entry was reached, so how it was reached cannot be "
                        "stated here\n";
                return;
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

            // The question's name is the grouping key and the shape column of a row need not spell it the same
            // way - the ladder to each argument's own order is produced by one all-orders call or by one call per
            // order - so the shapes the rows carry are stated where the two are not the same word.
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
                // Nothing here is ordered, and the two ways that happens are not the same fact: a shape whose every
                // row failed to measure has no figures at all, and a shape whose rows all failed a gate has figures
                // this run would not place.
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

            // A resolution is a distance between two order statistics of the run's ratios, and a run with fewer
            // paired rounds than a quartile band needs has none: the field is not printed there, and the line
            // says which count is missing instead.
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

    // The run's own measurement table, so the appendix reads the members this run
    // carried rather than a second reading of the space: a row of the library's
    // table is one member of the cross per form, and only the table says which of
    // its members this run took a place on.
    AppendOptionSpace(text, report);

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
