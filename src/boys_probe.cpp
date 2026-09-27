#include "boys/boys_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

// The option probe. Three things live here and nowhere else in the library:
//
//   1. A load instrument that needs no platform API, and a comparison that does
//      not depend on it. Every pass carries a reading of how much of a core this
//      process actually got: a fixed-work integer spin, whose fastest time here
//      is the machine's quiet floor and whose reading is the percentage by which
//      a run of it exceeded that floor. An integer chain is used deliberately -
//      it holds no floating-point state, so the arithmetic question the probe is
//      about cannot change the cost of the instrument itself. That reading is
//      reported and never gates: a fixed work read by wall clock measures the
//      clock as much as the load, so a machine whose boost decays widens the
//      spin's own spread while doing nothing else. The options are compared
//      inside a round instead - both were timed under the same clock, and a
//      drift common to the round cancels in their ratio - and the run is ordered
//      from those paired ratios.
//
//   2. The workload: the library's real call shape, per-argument orders over a
//      log-uniform argument range, built once and shared by every option so no
//      option is measured on a different question.
//
//   3. One body per option, written against a callable so the timed checksum
//      and the untimed accuracy comparison consume the same values from the
//      same calls. Two implementations of one option would be two chances to
//      measure something other than what is recommended.

namespace boys {

const char* PrecisionName(OptionPrecision precision) noexcept {
    switch (precision)
    {
    case OptionPrecision::kFp64:
        return "fp64";
    case OptionPrecision::kFp32:
        return "fp32";
    case OptionPrecision::kFp16:
        return "fp16";
    case OptionPrecision::kBf16:
        return "bf16";
    }

    return "unknown";
}

namespace {

// --- clocks -----------------------------------------------------------------

using Clock = std::chrono::steady_clock;

double SecondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double>(to - from).count();
}

double MillisecondsBetween(Clock::time_point from, Clock::time_point to) {
    return std::chrono::duration<double, std::milli>(to - from).count();
}

double NowSeconds() {
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}

// --- text -------------------------------------------------------------------

/// One line with no substitutions. A separate overload rather than the
/// variadic one with an empty pack: a format that is not a literal and carries
/// no arguments is what -Wformat-security rejects, and it is right to - the
/// call cannot be checked against its arguments because there are none. This
/// overload makes the no-argument case a copy and leaves the checked case the
/// only one that reaches snprintf.
std::string Text(const char* text) {
    return std::string(text);
}

/// One formatted line, so the report can be built without an iostream.
///
/// A call with no arguments passes its text through `%s`: gcc and clang reject
/// a non-literal format carrying no arguments (-Wformat-security), and a
/// caller with nothing to substitute is a caller whose text is the line.
template <typename... Args>
std::string Text(const char* format, Args... args) {
    char buffer[1024];

    if constexpr (sizeof...(args) == 0)
    {
        std::snprintf(buffer, sizeof(buffer), "%s", format);
    }
    else
    {
        std::snprintf(buffer, sizeof(buffer), format, args...);
    }

    return std::string(buffer);
}

// --- the load instrument ----------------------------------------------------

/// Iterations of the canary spin. Fixed, because the reading is a ratio of two
/// times of the same work and the work must not vary between them.
constexpr std::uint32_t kCanaryRounds = 1u << 23;

/// The spin: a xorshift chain, one dependent step per iteration, no memory
/// beyond the accumulator and no floating point.
double CanaryMilliseconds() {
    volatile std::uint64_t sink = 0;
    const Clock::time_point t0 = Clock::now();
    std::uint64_t state = 0x243F6A8885A308D3ull;

    for (std::uint32_t i = 0; i < kCanaryRounds; ++i)
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
    }

    // Inside the timed region on purpose: the store is the observable use of
    // the chain, so neither the chain nor the region can be moved past the
    // other.
    sink = state;
    const Clock::time_point t1 = Clock::now();
    (void)sink;
    return MillisecondsBetween(t0, t1);
}

/// The slowest run over the fastest, less one, in percent: how much a set of
/// runs of the same fixed work disagreed with itself. Zero for a set too small
/// to disagree at all.
double SpreadPercent(const std::vector<double>& milliseconds) {
    if (milliseconds.size() < 2)
    {
        return 0.0;
    }

    const auto ends = std::minmax_element(milliseconds.begin(), milliseconds.end());

    if (*ends.first <= 0.0)
    {
        return 0.0;
    }

    return 100.0 * (*ends.second / *ends.first - 1.0);
}

/// The middle value of a set of readings, by copy and sort: the sets here are
/// one pass long, and the sort costs nothing beside the runs it holds.
double MedianMilliseconds(std::vector<double> milliseconds) {
    if (milliseconds.empty())
    {
        return 0.0;
    }

    std::sort(milliseconds.begin(), milliseconds.end());
    return milliseconds[milliseconds.size() / 2];
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

/// The load the machine put on a single-threaded, CPU-bound workload, as a
/// percentage above the fastest run of the canary this machine has shown, with
/// the canary's own run-to-run agreement as the thing that decides whether a
/// pass may be reported at all.
///
/// Not the machine's total processor utilization: the standard library exposes
/// no other process's processor time, so the system-wide counter a
/// utilization reading would need is out of reach here. What is in reach is the
/// quantity that utilization stands for - whether a pass had a core to itself -
/// and it is measured directly, which is the stronger reading of the two.
///
/// A load percentage is a level and a level is not what corrupts a comparison:
/// a machine that is steadily busy slows every option by the same factor, and
/// the ratio the report orders on survives that. A machine whose speed wanders
/// is the one that corrupts a comparison, and the spread between the canary's
/// own runs measures exactly that, in the units the comparison is made in.
class LoadMeter {
public:
    /// Establishes the quiet floor over a window of \p seconds, and refuses the
    /// whole measurement when the window was not quiet enough to establish one:
    /// a floor taken on a busy machine is too high, and every later reading
    /// against it would look clean. What the window found is kept, so the
    /// report can say how steady the machine was rather than asserting it.
    ///
    /// \param seconds   length of the calibration window
    /// \param tolerance percentage above the minimum within which the window's
    ///                  median must sit
    /// \returns true when a floor was established
    bool Calibrate(double seconds, double tolerance) {
        std::vector<double> samples;
        const double begin = NowSeconds();

        while (NowSeconds() - begin < seconds)
        {
            samples.push_back(CanaryMilliseconds());
        }

        _runs = static_cast<int>(samples.size());

        if (samples.size() < 8)
        {
            return false;
        }

        std::sort(samples.begin(), samples.end());
        _floorMs = samples.front();
        _medianMs = samples[samples.size() / 2];
        _spread = SpreadPercent(samples);
        _calibrated = (_medianMs <= _floorMs * (1.0 + tolerance / 100.0));
        return _calibrated;
    }

    /// \returns the quiet floor, in milliseconds
    double FloorMs() const
    {
        return _floorMs;
    }

    /// \returns the number of runs the calibration window fitted in
    int Runs() const
    {
        return _runs;
    }

    /// \returns the calibration window's median run, in milliseconds
    double MedianMs() const
    {
        return _medianMs;
    }

    /// \returns the calibration window's own spread, in percent
    double CalibrationSpreadPercent() const
    {
        return _spread;
    }

    /// \returns whether a floor was established
    bool Calibrated() const
    {
        return _calibrated;
    }

    /// \returns one raw run of the canary, in milliseconds
    double SampleMs() const
    {
        return CanaryMilliseconds();
    }

    /// \param milliseconds one canary run
    /// \returns that run as a percentage above the floor
    double Percent(double milliseconds) const {
        if (_floorMs <= 0.0)
        {
            return 0.0;
        }

        const double above = 100.0 * (milliseconds / _floorMs - 1.0);
        return above > 0.0 ? above : 0.0;
    }

    /// \param seconds length of the window
    /// \returns the mean of the readings taken until the window elapsed
    double Average(double seconds) const {
        const double begin = NowSeconds();
        double total = 0.0;
        std::size_t readings = 0;

        do
        {
            total += Percent(CanaryMilliseconds());
            ++readings;
        } while (NowSeconds() - begin < seconds);

        return total / static_cast<double>(readings);
    }

private:
    double _floorMs = 0.0;
    double _medianMs = 0.0;
    double _spread = 0.0;
    int _runs = 0;
    bool _calibrated = false;
};

/// Percentage above the minimum within which the calibration window's median
/// must sit for its floor to mean anything at all.
///
/// Loose on purpose, and not an admission rule — there is none. The floor is the
/// denominator of the load readings, which are context, so a busy calibration
/// window costs a little accuracy in those numbers and nothing else. A machine
/// that is busy throughout still gets measured and told what its own wandering
/// allows it to order, because what decides that is the spread of the paired
/// within-round ratios and not the canary. This bar is here only to catch a
/// window that never settled into a floor at all: at 100 the median run may take
/// twice the fastest, and beyond that the fastest run is a glitch rather than a
/// floor.
constexpr double kCalibrationTolerancePercent = 100.0;

/// The quantile the reported cost and its ratio to the reference are read at.
///
/// A lower quartile, and not the minimum the probe used to report. Under a
/// decaying clock the minimum of a run is its earliest observation, taken at the
/// highest clock the machine will reach that day, and a caller whose workload
/// runs for hours does not meet that state; a quartile is what the bulk of a
/// long workload meets, and it still discards the round a background process
/// stole. It is a quartile and not a mean for the same reason: one disturbed
/// round out of many must not move a figure a consumer will build on.
constexpr double kStatisticQuantile = 0.25;

/// The number of rounds a quartile band needs before it can exist: two ends of a
/// band are two order statistics, and below four observations they are the same
/// observation twice.
constexpr std::size_t kMinimumPairedRounds = 4;

/// The lane whose option space this probe enumerates. The fitted double lane is
/// the one the library's own default is defined in and the one every other lane
/// is judged against, so its cells are the space the coverage book walks and the
/// space a default is chosen from.
constexpr Precision kCellLane = Precision::kFp64;

// --- the workload -----------------------------------------------------------

/// The arguments, their own orders, and the grouping the grouped entries need,
/// built once and read by every option.
struct Workload {
    ProbeOptions options;

    /// One argument per entry.
    std::vector<double> x;

    /// Each argument's own highest order, 0..nmax.
    std::vector<int> order;

    /// Argument indices by (order, x): each run of equal order is a batch one
    /// all-N call can take, and each run is non-decreasing in x, which is what
    /// makes the sorted overload's declaration true of it.
    std::vector<std::size_t> sorted;

    /// Run boundaries into `sorted`; one more entry than there are runs.
    std::vector<std::size_t> runBegin;

    /// Each run's order.
    std::vector<int> runOrder;

    /// Arguments in the largest run.
    std::size_t largestRun = 0;
};

Workload BuildWorkload(const ProbeOptions& options) {
    Workload work;
    work.options = options;
    work.x.resize(options.count);
    work.order.resize(options.count);

    std::mt19937_64 generator(options.seed);
    std::uniform_real_distribution<double> logX(
        std::log10(options.xLo), std::log10(options.xHi));
    std::uniform_int_distribution<int> shell(0, options.nmax / 2);

    for (std::size_t i = 0; i < options.count; ++i)
    {
        work.x[i] = std::pow(10.0, logX(generator));

        // The order a shell pair presents: the sum of two shell angular
        // momenta, which is what an integral engine's batches are grouped by.
        const int first = shell(generator);
        const int second = shell(generator);
        work.order[i] = std::min(options.nmax, first + second);
    }

    work.sorted.resize(options.count);
    std::iota(work.sorted.begin(), work.sorted.end(), std::size_t{0});
    std::sort(work.sorted.begin(), work.sorted.end(),
              [&work](std::size_t a, std::size_t b) {
                  if (work.order[a] != work.order[b])
                  {
                      return work.order[a] < work.order[b];
                  }

                  return work.x[a] < work.x[b];
              });

    work.runBegin.push_back(0);

    for (std::size_t i = 1; i < options.count; ++i)
    {
        if (work.order[work.sorted[i]] != work.order[work.sorted[i - 1]])
        {
            work.runBegin.push_back(i);
            work.runOrder.push_back(work.order[work.sorted[i - 1]]);
        }
    }

    work.runOrder.push_back(work.order[work.sorted[options.count - 1]]);
    work.runBegin.push_back(options.count);

    for (std::size_t r = 0; r + 1 < work.runBegin.size(); ++r)
    {
        work.largestRun = std::max(work.largestRun, work.runBegin[r + 1] - work.runBegin[r]);
    }

    return work;
}

/// The scratch the grouped entries need, allocated once so the timed region
/// never pays for an allocation.
struct Buffers {
    std::vector<double> runX;
    std::vector<double> runOut;
    std::vector<std::size_t> workspace;
};

Buffers MakeBuffers(const Workload& work) {
    Buffers buffers;
    const std::size_t run = std::max<std::size_t>(work.largestRun, 1);
    buffers.runX.resize(run);
    buffers.runOut.resize(run * static_cast<std::size_t>(work.options.nmax + 1));
    buffers.workspace.resize(BoysAllNWorkspaceSize(run));
    return buffers;
}

// --- the options ------------------------------------------------------------

/// Which entries an option is, and therefore which arithmetic it runs in and
/// which call shape it measures.
enum class OptionKind {
    kBatchFp64, ///< one all-orders call per argument, fp64
    kTierFp64, ///< the same, at a run-time accuracy tier
    kFp64Cell, ///< one cell of the route x scheme x partition x axis x rung space
    kGroupedFp64, ///< one all-N call per order run, fp64
    kTaggedFp64, ///< the same, with the arguments declared already sorted
    kBatchFp32, ///< one all-orders fp32 call per argument
    kBatchF16, ///< one all-orders call per argument, fp16 I/O
    kBatchBf16, ///< one all-orders call per argument, bf16 I/O
};

struct Option {
    std::string name;
    std::string arithmetic;
    bool contracts = false;
    OptionKind kind = OptionKind::kBatchFp64;
    OptionPrecision precision = OptionPrecision::kFp64;
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme;
    FitGranularity granularity = kDefaultFitGranularity;
    PackAxis pack = PackAxis::kArguments;
    AccuracyTier tier = AccuracyTier::kReference;

    /// The figure the row is judged against, and it is a whole-domain one for
    /// every option: the figure the library documents for the entry the option
    /// reaches. Naming a partition changes which tables that entry reads, not
    /// the entry's own figure, so a partition option carries the same figure as
    /// the option that reads the shipped tables.
    double bound = 0.0;

    /// The figure the option's own tables are certified at, where it has any of
    /// its own, and the interval that figure holds on. Zero for an option that
    /// reads the shipped tables, and then the interval is not printed. A
    /// partition's figures are its stored fits', and they say nothing about the
    /// arguments outside the interval, where the entry runs its own arithmetic —
    /// so they inform the row rather than judge it.
    double ownBound = 0.0;
    double ownLo = 0.0;
    double ownHi = std::numeric_limits<double>::infinity();
};

/// The packed backends' names, which the library's own backend test pins. The
/// scalar names are not written here: they are the types' own kName.
constexpr const char* kPackedFp64Name = "avx2-fp64";
constexpr const char* kPackedFp32Name = "avx2-fp32";

/// The entry every ratio is formed against: the library's default
/// double-precision all-orders call, which is what a caller who chooses no
/// policy already runs. Naming it here is what makes the anchor a documented
/// choice rather than the winner of the comparison it anchors.
constexpr const char* kReferenceOptionName = "batch-fp64";

/// The option every ratio is formed against, resolved from the option book: the
/// entry named above when this run measured it, else the first option of the
/// default precision, else the first option measured.
///
/// Resolved before any round is timed, so the anchor cannot be chosen to suit
/// the answer. It is the library's own default lane, so every ratio reads as
/// "this option against what a caller would have got anyway".
std::size_t ReferenceIndex(const std::vector<Option>& options) {
    for (std::size_t index = 0; index < options.size(); ++index)
    {
        if (options[index].name == kReferenceOptionName)
        {
            return index;
        }
    }

    for (std::size_t index = 0; index < options.size(); ++index)
    {
        if (options[index].precision == OptionPrecision::kFp64)
        {
            return index;
        }
    }

    return 0;
}

/// The widest relative width of a within-round ratio band a set of rounds
/// showed, in percent: for each option, the upper quartile of its ratio to the
/// reference over the lower quartile of the same ratio, less one, taken at its
/// widest.
///
/// The reference's own column is skipped, and only it: its ratio is one in every
/// round by construction, so it would contribute a band of zero. A set too short
/// for a band reports zero rather than a width it did not measure.
double PassPairedSpread(const std::vector<std::vector<double>>& rounds, std::size_t reference) {
    double widest = 0.0;

    if (rounds.empty() || reference >= rounds.front().size())
    {
        return 0.0;
    }

    for (const std::vector<double>& row : rounds)
    {
        if (row[reference] <= 0.0)
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

/// The published bounds of the narrower lanes at the reference multiplier. The
/// fp64 options do not appear here: their bound is read from the library
/// through QueryTier, which reports the batch lane's own budget.
constexpr double kFloatLaneBound = 1.5e-7;

/// 1e-7 plus the half-ULP representation term, taken at the top of the range
/// the function returns in: |F_n(x)| <= 1, so 2^-11 is the largest half-ULP a
/// binary16 return can carry there, and 2^-9 the largest a bfloat16 one can.
/// Both belong to the entries the fp16 seam declares, so they are compiled with
/// them and a build with the seam closed carries no bound for a lane it has.
#if BoysFp16
constexpr double kHalfIoLaneBound = 1e-7 + 0x1p-11;
constexpr double kBf16IoLaneBound = 1e-7 + 0x1p-9;
#endif // BoysFp16

/// The largest error a tier can deliver in any region, read from the library.
///
/// The batch lane's budget is region-independent for the fp64 entries this
/// probe measures, and taking the maximum over the regions rather than naming
/// one keeps that a measured fact about this build instead of an assumption.
double TierBound(AccuracyTier tier) {
    double bound = 0.0;

    for (const AccuracyRegion region :
         {AccuracyRegion::kA, AccuracyRegion::kB, AccuracyRegion::kC})
    {
        bound = std::max(bound, QueryTier(tier, region, 0.0).reachable);
    }

    return bound;
}

/// The accuracy rungs this build serves, the reference first.
///
/// Walked over the tier enumeration rather than written here: a tier this build
/// does not serve reports the reference multiplier's reachable error, which is
/// the rung the library evaluates it at, and the rungs this build does serve
/// report their own, so the tiers whose report differs from the reference's are
/// exactly the rungs the build carries. A build that adds, drops or spaces out a
/// rung is measured as it is.
std::vector<AccuracyTier> ServedTiers() {
    std::vector<AccuracyTier> tiers{AccuracyTier::kReference};
    const double referenceReachable =
        QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable;
    constexpr int kMaxTierProbe = 32;

    for (int raw = 1; raw <= kMaxTierProbe; ++raw)
    {
        const auto tier = static_cast<AccuracyTier>(raw);

        if (QueryTier(tier, AccuracyRegion::kA, 0.0).reachable != referenceReachable)
        {
            tiers.push_back(tier);
        }
    }

    return tiers;
}

/// The name a cell's option is printed under, in one grammar for all of them.
///
/// The name states the cell's own axes and omits the defaults, so the names the
/// probe printed before this change still name the same options: a defaulted
/// route, scheme, partition and axis leave no segment behind. The partition and
/// the rung share the first segment — `batch` for the shipped partition at the
/// reference rung, `tier-<m>` for it at a rung, `narrow` for the other partition
/// — and the precision closes every name, because a name is only ever read
/// inside its class.
std::string CellName(FitGranularity granularity, PackAxis pack, FitRoute route,
                     EvalScheme scheme, AccuracyTier tier) {
    std::string name = "batch";
    const bool rungNamed = tier != AccuracyTier::kReference;

    if (granularity == FitGranularity::kNarrow)
    {
        name = "narrow";
    } else if (rungNamed)
    {
        name = Text("tier-%g", AccuracyMultiplier(tier));
    }

    if (granularity == FitGranularity::kNarrow && rungNamed)
    {
        name += Text("-%g", AccuracyMultiplier(tier));
    }

    if (pack == PackAxis::kOrders)
    {
        name += "-pack-orders";
    }

    if (route == FitRoute::kRationalMinimax)
    {
        name += "-rational";
    }

    if (scheme == EvalScheme::kHorner)
    {
        name += "-horner";
    }

    return name + "-fp64";
}

/// The arithmetic of one precision as this build's table names it: the packed
/// arithmetic when the vector tier is live, the scalar arithmetic otherwise.
///
/// \returns the table entry, or nullptr when the table carries neither, which
///          is how a build without that precision's arithmetic offers none of
///          the options that run in it
const backend::BackendInfo* ResolveArithmetic(std::span<const backend::BackendInfo> table,
                                              bool fp64) {
    const char* packed = fp64 ? kPackedFp64Name : kPackedFp32Name;
    const char* scalar = fp64 ? backend::ScalarFp64::kName : backend::ScalarFp32::kName;
    const char* wanted = BoysAvx2Available() ? packed : scalar;

    for (const backend::BackendInfo& entry : table)
    {
        if (entry.name != nullptr && std::strcmp(entry.name, wanted) == 0)
        {
            return &entry;
        }
    }

    const char* fallback = (std::strcmp(wanted, packed) == 0) ? scalar : packed;

    for (const backend::BackendInfo& entry : table)
    {
        if (entry.name != nullptr && std::strcmp(entry.name, fallback) == 0)
        {
            return &entry;
        }
    }

    return nullptr;
}

/// The name the library prints a route under, read from its own route table
/// rather than written here, so a coverage line cannot name a route the library
/// does not.
const char* RouteName(std::span<const FitRouteInfo> routes, FitRoute route) {
    for (const FitRouteInfo& row : routes)
    {
        if (row.route == route)
        {
            return row.name;
        }
    }

    return "unknown";
}

/// The routes a partition's tables carry, as the library names them, joined for
/// a report line — so a refusal can say what the partition does hold rather
/// than only what it lacks.
std::string PartitionRouteNames(const FitGranularityInfo& partition) {
    std::string text;

    for (const FitRouteInfo& row : BoysFitRoutes())
    {
        if (!FitGranularityHasRoute(partition, row.route))
        {
            continue;
        }

        const std::string name = RouteName(BoysFitRoutes(), row.route);

        if (text.find(name) != std::string::npos)
        {
            continue;
        }

        if (!text.empty())
        {
            text += ", ";
        }

        text += name;
    }

    return text;
}

/// The names of the routes a fit table carries, each taken once.
///
/// A table has one row per route and region, so the same route's name appears
/// more than once; the enumeration takes the route once, because a cell names
/// the route and not the region its fit covers.
std::vector<std::string> DistinctRouteNames(std::span<const FitRouteInfo> routes) {
    std::vector<std::string> names;
    std::vector<FitRoute> seen;

    for (const FitRouteInfo& row : routes)
    {
        if (std::find(seen.begin(), seen.end(), row.route) != seen.end())
        {
            continue;
        }

        seen.push_back(row.route);
        names.push_back(RouteName(routes, row.route));
    }

    return names;
}

/// The whole option space this build defines, cell by cell, with the library's
/// reason for every cell it does not serve.
///
/// The space is the product the library reports: the routes of the double lane's
/// fit table, the evaluation schemes, the partitions of the fitted regions, the
/// packing axes, and the accuracy rungs this build serves. Walking the product
/// rather than a list written here is what makes the coverage a statement about
/// the library: a member a later change adds is enumerated, and a cell the
/// library refuses is named here with the reason the library refuses it, so it is
/// counted as the unbuilt work it is instead of being absent from the report.
///
/// **Each cell's state is the library's answer to the cell's own combination**,
/// asked through \c BoysAccuracyGuaranteed: a cell is served exactly when that
/// call answers a figure for it, and a refused cell carries the reason that call
/// gives, in its own words. Nothing here restates a rule about routes, axes,
/// partitions or rungs - a second copy of the library's refusals is a second
/// answer, and it drifts: this file used to keep one, and it went on refusing
/// cells of the narrow rational partition after the library had begun serving
/// them, which shrank the set a default could be chosen from without saying so.
std::vector<OptionProbeCell> EnumerateCells() {
    std::vector<OptionProbeCell> cells;
    const std::span<const FitRouteInfo> routes = BoysFitRoutes();
    const std::vector<AccuracyTier> tiers = ServedTiers();

    // A route has one row per region it supplies, so the routes are taken once:
    // a cell names the route, not the region its fit covers.
    std::vector<FitRoute> seenRoutes;

    for (const FitRouteInfo& route : routes)
    {
        if (std::find(seenRoutes.begin(), seenRoutes.end(), route.route) != seenRoutes.end())
        {
            continue;
        }

        seenRoutes.push_back(route.route);

        for (const EvalSchemeInfo& scheme : BoysEvalSchemes())
        {
            for (const FitGranularityInfo& partition : BoysFitGranularities())
            {
                for (const PackAxisInfo& axis : BoysPackAxes())
                {
                    for (const AccuracyTier tier : tiers)
                    {
                        OptionProbeCell cell;
                        cell.name = CellName(
                            partition.granularity, axis.axis, route.route, scheme.scheme, tier);
                        cell.route = route.route;
                        cell.scheme = scheme.scheme;
                        cell.granularity = partition.granularity;
                        cell.pack = axis.axis;
                        cell.tier = tier;

                        // One question, one answer: the library is asked about this
                        // cell's own combination, and its reply is both the state and
                        // the reason. The lane is the one whose space this book
                        // enumerates, and the scheme is inert on it for the same
                        // reason it is inert in the option book above.
                        const AccuracyFigure carriage = BoysAccuracyGuaranteed(
                            kCellLane, cell.route, cell.scheme, cell.pack, cell.granularity,
                            cell.tier);

                        cell.served = carriage.available;

                        if (!carriage.available)
                        {
                            cell.reason = carriage.reason;
                        }

                        cells.push_back(cell);
                    }
                }
            }
        }
    }

    return cells;
}

/// Every option this build offers, enumerated from the library.
///
/// A cell the library serves becomes an option, named by the cell's own
/// grammar, and the entry that evaluates it is chosen by the cell's axes: the
/// run-time tier entry carries the shipped partition on the arguments axis at
/// any route and scheme, the across-orders lane carries every route and scheme
/// at any rung, and the narrow partition is reachable only as a policy — which
/// is the whole point of listing it, since a consumer reaches it the same way.
///
/// The entries whose call shape is not one of the five axes are named here,
/// because a shape is a function and no table of them exists to read: the
/// all-N per-run shape and its sorted overload, and the narrower lanes. What is
/// not written here is what this build carries: each option's arithmetic is
/// resolved against backend::BoysBackends(), so an option whose arithmetic the
/// table lacks is reported as not offered rather than measured under a name that
/// is not this build's. The half-precision lanes are behind the same BoysFp16
/// seam that declares them, so a build whose seam is closed carries no entry to
/// name and says so through notCarried rather than by leaving the names
/// unmentioned.
std::vector<Option> EnumerateOptions(std::span<const backend::BackendInfo> table,
                                     std::span<const FitGranularityInfo> partitions,
                                     const std::vector<OptionProbeCell>& cells,
                                     std::vector<std::string>& unoffered,
                                     std::vector<std::string>& notCarried) {
    std::vector<Option> options;

    const backend::BackendInfo* fp64 = ResolveArithmetic(table, true);
    const backend::BackendInfo* fp32 = ResolveArithmetic(table, false);

    const auto append = [&](std::string name, OptionKind kind, OptionPrecision precision,
                            FitRoute route, EvalScheme scheme, FitGranularity granularity,
                            PackAxis pack, AccuracyTier tier,
                            const backend::BackendInfo* arithmetic, double bound) {
        if (arithmetic == nullptr)
        {
            unoffered.push_back(std::move(name));
            return;
        }

        Option option;
        option.name = std::move(name);
        option.arithmetic = arithmetic->name;
        option.contracts = arithmetic->contracts;
        option.kind = kind;
        option.precision = precision;
        option.route = route;
        option.scheme = scheme;
        option.granularity = granularity;
        option.pack = pack;
        option.tier = tier;
        option.bound = bound;
        options.push_back(std::move(option));
    };

    append("batch-fp64", OptionKind::kBatchFp64, OptionPrecision::kFp64, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments, AccuracyTier::kReference,
           fp64, TierBound(AccuracyTier::kReference));
    append("grouped-fp64", OptionKind::kGroupedFp64, OptionPrecision::kFp64, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments, AccuracyTier::kReference,
           fp64, TierBound(AccuracyTier::kReference));
    append("tagged-fp64", OptionKind::kTaggedFp64, OptionPrecision::kFp64, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments, AccuracyTier::kReference,
           fp64, TierBound(AccuracyTier::kReference));
    append("batch-fp32", OptionKind::kBatchFp32, OptionPrecision::kFp32, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments, AccuracyTier::kReference,
           fp32, kFloatLaneBound);

#if BoysFp16
    // The half lanes run the fp32 engine, so their arithmetic is the fp32 one.
    append("f16-io", OptionKind::kBatchF16, OptionPrecision::kFp16, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments,
           AccuracyTier::kReference, fp32, kHalfIoLaneBound);
    append("bf16-io", OptionKind::kBatchBf16, OptionPrecision::kBf16, kDefaultFitRoute,
           kDefaultEvalScheme, kDefaultFitGranularity, PackAxis::kArguments,
           AccuracyTier::kReference, fp32, kBf16IoLaneBound);

    // A build that carries both lanes has nothing to report as not carried, and
    // the register stays a parameter either way so its caller reads one list in
    // both builds.
    (void)notCarried;
#else
    // The seam is closed in this build, so the two lanes are not options it can
    // serve. They are named as ones this build does not carry rather than
    // dropped from both lists: a caller asking for one is told the build has no
    // such lane, instead of being told the name is no option of this library.
    notCarried.emplace_back("f16-io");
    notCarried.emplace_back("bf16-io");
#endif

    // Every served cell becomes an option, under the cell's own name, except the
    // one cell the shape rows above already carry: the shipped partition on the
    // arguments axis at the default route and scheme, whose rungs are the
    // `tier-<m>-fp64` rows the walk below adds. Those rows keep their names, so a
    // caller who read an earlier report still names the same options.
    for (const OptionProbeCell& cell : cells)
    {
        if (!cell.served)
        {
            continue;
        }

        if (cell.granularity == kDefaultFitGranularity && cell.pack == PackAxis::kArguments &&
            cell.route == kDefaultFitRoute && cell.scheme == kDefaultEvalScheme)
        {
            continue;
        }

        append(cell.name, OptionKind::kFp64Cell, OptionPrecision::kFp64, cell.route, cell.scheme,
               cell.granularity, cell.pack, cell.tier, fp64, 0.0);
    }

    // The relaxed rungs of the default policy, read off the same served-tier
    // list the coverage book uses, so the rows and the book cannot disagree about
    // which rungs this build carries.
    for (const AccuracyTier tier : ServedTiers())
    {
        if (tier == AccuracyTier::kReference)
        {
            continue;
        }

        append(Text("tier-%g-fp64", AccuracyMultiplier(tier)), OptionKind::kTierFp64,
               OptionPrecision::kFp64, kDefaultFitRoute, kDefaultEvalScheme,
               kDefaultFitGranularity, PackAxis::kArguments, tier, fp64, TierBound(tier));
    }

    // The bound of every cell option is the figure the library documents for the
    // entry the cell reaches, which is what the run-time tier entry reports
    // through QueryTier whatever partition it reads: naming a partition replaces
    // the fitted tables the entry reads and leaves the entry's own figure where
    // it was. The partition's own figures are recorded beside it, with the
    // interval they hold on, because they are figures for the fitted interval
    // alone and the row prints them as such.
    for (Option& option : options)
    {
        if (option.kind != OptionKind::kFp64Cell)
        {
            continue;
        }

        option.bound = TierBound(option.tier);

        if (option.granularity == FitGranularity::kNarrow)
        {
            for (const FitGranularityInfo& partition : partitions)
            {
                if (partition.granularity == FitGranularity::kNarrow)
                {
                    option.ownBound = partition.bound;
                    option.ownLo = partition.lo;
                    option.ownHi = partition.hi;
                }
            }
        }
    }

    return options;
}

// --- one option's values ----------------------------------------------------

/// One cell's own entry at one rung, as the cell's routes, schemes, partition
/// and packing axis select it.
///
/// A cell that is not the default policy's own shape exists only as an
/// instantiation: the partition and the packing axis are template arguments of
/// the policy and have no run-time entry, so a consumer reaches them the way
/// this does, by naming them. The rung is one branch per multiplier for the same
/// reason - the multiplier is the first template argument of the entry - and the
/// dispatch is the same five-way cross the library's own accuracy gate measures,
/// so a cell this probe reports as served is a cell some entry of this build
/// really runs.
template <FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kGran>
void CellRung(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    constexpr BoysBudget kBudget = BoysBudget::kFloat;
    using Policy = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran>;

    switch (tier)
    {
    case AccuracyTier::kRelaxed64:
        BoysAllOrders<64.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed256:
        BoysAllOrders<256.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed1024:
        BoysAllOrders<1024.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed4096:
        BoysAllOrders<4096.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed16384:
        BoysAllOrders<16384.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed65536:
        BoysAllOrders<65536.0, Policy>(nmax, x, out);
        return;

    default:
        BoysAllOrders<kBoysFullAccuracyMultiplier, Policy>(nmax, x, out);
        return;
    }
}

/// The same, for a cell named at run time: four run-time branches, one per axis,
/// each narrowing to the template argument it names, so a cell is measured
/// through its own policy and never through another cell's.
void CellPolicy(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity granularity,
                AccuracyTier tier, int nmax, double x, double* out) noexcept {
    const auto with_partition = [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack>() {
        if (granularity == FitGranularity::kNarrow)
        {
            CellRung<kRoute, kScheme, kPack, FitGranularity::kNarrow>(tier, nmax, x, out);
        } else
        {
            CellRung<kRoute, kScheme, kPack, FitGranularity::kShipped>(tier, nmax, x, out);
        }
    };

    const auto with_pack = [&]<FitRoute kRoute, EvalScheme kScheme>() {
        if (pack == PackAxis::kOrders)
        {
            with_partition.template operator()<kRoute, kScheme, PackAxis::kOrders>();
        } else
        {
            with_partition.template operator()<kRoute, kScheme, PackAxis::kArguments>();
        }
    };

    const auto with_scheme = [&]<FitRoute kRoute>() {
        if (scheme == EvalScheme::kHorner)
        {
            with_pack.template operator()<kRoute, EvalScheme::kHorner>();
        } else
        {
            with_pack.template operator()<kRoute, EvalScheme::kSplitClenshaw>();
        }
    };

    if (route == FitRoute::kRationalMinimax)
    {
        with_scheme.template operator()<FitRoute::kRationalMinimax>();
    } else
    {
        with_scheme.template operator()<FitRoute::kChebyshev>();
    }
}

/// One cell of the option space, evaluated as the entry its axes select.
///
/// The default policy's own shape is answered by the run-time tier entry — the
/// dispatch a consumer reaches without naming a template argument — and every
/// other cell by its own policy instantiation, which is how a consumer reaches a
/// partition or a packing axis. Both are branches on the same run-time values,
/// so a cell's cost includes its own selection, as the run-time tier options'
/// cost already does.
void CellValues(const Option& option, int nmax, double x, double* out) noexcept {
    if (option.granularity == kDefaultFitGranularity &&
        option.pack == PackAxis::kArguments)
    {
        BoysAllOrdersAtTier(option.tier, option.route, option.scheme, nmax, x, out);
        return;
    }

    CellPolicy(option.route, option.scheme, option.pack, option.granularity, option.tier, nmax, x,
               out);
}

/// Runs one option's own calls over the workload and hands every value it
/// returns to `visit(x, order, value)`, with `x` the argument the option was
/// actually asked about — the narrowed one for a lane that rounds its argument
/// before evaluating.
///
/// The single body shared by the timed checksum and the untimed accuracy
/// comparison: a second implementation of an option would be a second chance to
/// measure something other than what the report recommends.
template <typename Visit>
void VisitValues(const Workload& work, Buffers& buffers, const Option& option, Visit&& visit) {
    switch (option.kind)
    {
    case OptionKind::kBatchFp64:
    case OptionKind::kTierFp64:
    {
        double values[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];

            if (option.kind == OptionKind::kBatchFp64)
            {
                BoysAllOrders(n, work.x[i], values);
            } else
            {
                BoysAllOrdersAtTier(option.tier, n, work.x[i], values);
            }

            for (int k = 0; k <= n; ++k)
            {
                visit(work.x[i], k, values[k]);
            }
        }

        break;
    }

    case OptionKind::kFp64Cell:
    {
        double values[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];
            CellValues(option, n, work.x[i], values);

            for (int k = 0; k <= n; ++k)
            {
                visit(work.x[i], k, values[k]);
            }
        }

        break;
    }

    case OptionKind::kBatchFp32:
    {
        float values[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];
            const float x = static_cast<float>(work.x[i]);
            BoysAllOrdersF32(n, x, values);

            for (int k = 0; k <= n; ++k)
            {
                visit(static_cast<double>(x), k, static_cast<double>(values[k]));
            }
        }

        break;
    }

    case OptionKind::kBatchF16:
    case OptionKind::kBatchBf16:
    {
        // Both kinds are declared in every build, so both are labelled in every
        // build: a switch that left them out would be refused by -Werror=switch,
        // and a label with nothing under it would answer in silence. The seam
        // decides what stands under the label.
#if BoysFp16
        if (option.kind == OptionKind::kBatchF16)
        {
            F16 values[kMaxBoysOrder + 1];

            for (std::size_t i = 0; i < work.x.size(); ++i)
            {
                const int n = work.order[i];
                const F16 x = static_cast<F16>(static_cast<float>(work.x[i]));
                BoysAllOrdersF16(n, x, values);

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k, static_cast<double>(values[k]));
                }
            }
        } else
        {
            Bf16 values[kMaxBoysOrder + 1];

            for (std::size_t i = 0; i < work.x.size(); ++i)
            {
                const int n = work.order[i];
                const Bf16 x = static_cast<Bf16>(static_cast<float>(work.x[i]));
                BoysAllOrdersBf16(n, x, values);

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k, static_cast<double>(values[k]));
                }
            }
        }
#else
        // A build whose seam is closed constructs no option of either kind - the
        // enumeration below offers neither and registers them as not carried - so
        // this arm is unreachable here. It stops rather than returning nothing,
        // because a value visitor that answered no values would read as an option
        // this build measured and found empty.
        std::fprintf(stderr,
                     "boys-probe: %s names a half-precision lane this build does not carry "
                     "(BoysFp16 = 0)\n",
                     option.name.c_str());
        std::abort();
#endif // BoysFp16

        break;
    }

    case OptionKind::kGroupedFp64:
    case OptionKind::kTaggedFp64:
    {
        for (std::size_t r = 0; r + 1 < work.runBegin.size(); ++r)
        {
            const std::size_t from = work.runBegin[r];
            const std::size_t to = work.runBegin[r + 1];
            const std::size_t run = to - from;
            const int n = work.runOrder[r];

            for (std::size_t j = 0; j < run; ++j)
            {
                buffers.runX[j] = work.x[work.sorted[from + j]];
            }

            if (option.kind == OptionKind::kGroupedFp64)
            {
                // The allocation-free form the entry documents for a hot loop:
                // the caller's own workspace, so the timed region measures the
                // grouping and not the allocator.
                BoysAllN(n, buffers.runX.data(), buffers.runOut.data(), run,
                         buffers.workspace.data());
            } else
            {
                // Every run is non-decreasing in x by construction, which is
                // exactly what this overload declares.
                BoysAllN(n, buffers.runX.data(), buffers.runOut.data(), run, BoysSortedArgs{});
            }

            for (std::size_t j = 0; j < run; ++j)
            {
                for (int k = 0; k <= n; ++k)
                {
                    visit(buffers.runX[j], k,
                          buffers.runOut[static_cast<std::size_t>(k) * run + j]);
                }
            }
        }

        break;
    }
    }
}

/// The timed consumer: one addition per returned value, so the clock reads the
/// option's own work.
double Checksum(const Workload& work, Buffers& buffers, const Option& option) {
    double sum = 0.0;
    VisitValues(work, buffers, option,
                [&sum](double, int, double value) { sum += value; });
    return sum;
}

/// The untimed consumer: the worst difference from the certified lane at the
/// order and argument the option was asked about, and whether the two ever
/// differed at all.
///
/// The reference is the certified per-order fp64 entry, evaluated one order at
/// a time. That is the question a consumer is really asking — this option
/// returned a number for order k at argument x, and F_k(x) is certified — and
/// it is the only anchor that is the same value for every option: the batch
/// entries seed region A at the batch's own highest order, so their F_k for
/// k below it is a recurrence's value rather than a per-order walk's, and a
/// batch lane taken as the reference would hide that difference behind the
/// label "reference" while showing it for everything else.
///
/// The reference calls are memoized per argument: a lane's whole order list at
/// one argument costs the orders it asks for, walked upward. That is also why
/// the grouped options are visited argument-major even though their output is
/// plane-major — the order the values are visited does not change a maximum,
/// and it decides whether the comparison is one reference call per argument or
/// one per value.
class AccuracyWalker {
public:
    void operator()(double x, int k, double value) {
        const double expected = Reference(x, k);
        const double difference = std::fabs(value - expected);

        _maxError = std::max(_maxError, difference);
        _bitIdentical = _bitIdentical && (value == expected);
    }

    double MaxError() const
    {
        return _maxError;
    }

    bool BitIdentical() const
    {
        return _bitIdentical;
    }

private:
    double Reference(double x, int k) {
        if (x != _lastX) {
            _lastX = x;
            _filled = 0;
        }

        if (k >= _filled) {
            for (int j = _filled; j <= k; ++j) {
                _reference[j] = BoysSingle(j, x);
            }
            _filled = k + 1;
        } else {
            _reference[k] = BoysSingle(k, x);
        }

        return _reference[k];
    }

    double _reference[kMaxBoysOrder + 1] = {};
    double _lastX = -1.0; // arguments are >= 0, so this is never a match
    int _filled = 0;
    double _maxError = 0.0;
    bool _bitIdentical = true;
};

// --- the conclusion ---------------------------------------------------------

/// One rival measured against its class's leader, inside the rounds.
///
/// Every quantity here is a ratio formed inside a single round: both options
/// were timed under whatever clock that round ran at, so a drift common to the
/// round is in both terms of the ratio and cancels. This is the comparison the
/// probe is ordered by.
struct PairedOutcome {
    /// Lower and upper quartile of the within-round ratio, rival over leader:
    /// the band the middle half of the run put the pair in.
    double lo = 1.0;
    double hi = 1.0;

    /// Rounds in which the rival was the slower of the two, of the rounds that
    /// could be formed.
    int slowerRounds = 0;
    int rounds = 0;

    /// How far the pair's ratio moved between the run's first and second half of
    /// rounds: the second half's median ratio over the first half's, less one. A
    /// pair whose ratio is a property of the two options holds still as the clock
    /// moves; one that drifts is a pair the two do not carry the clock alike,
    /// which is the one way a paired comparison can still be misled by a machine
    /// whose boost decays.
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
/// \param rounds the round table: one row per round, one column per option, in
///               cost per argument
/// \param leader column of the option the rival is measured against
/// \param rival  column of the option being placed
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

        if (lead <= 0.0)
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

    const double firstHalf = MedianMilliseconds(early);
    const double secondHalf = MedianMilliseconds(late);

    if (firstHalf > 0.0)
    {
        outcome.drift = secondHalf / firstHalf - 1.0;
    }

    return outcome;
}

/// One run's ordering of a set of options, read from that run's own round table.
struct RunOutcome {
    /// The options ordered by the run's statistic, fastest first, as columns of
    /// the table they were read from.
    std::vector<std::size_t> order;

    /// The fastest of them. Meaningless when \c order is empty.
    std::size_t leader = 0;

    /// The options the run's own rounds could not place behind the leader: their
    /// within-round ratio to it did not clear one in the middle half of the
    /// rounds. Empty when the run ordered the whole set.
    std::vector<std::size_t> unplaced;

    /// The widest relative width of a band the run showed, an option's own band
    /// against the anchor or a rival's band against the leader.
    double widestBand = 0.0;

    /// How far the ratio of the run's widest-moving pair travelled between its
    /// two halves, and which pair that was, as columns of the table the run was
    /// read from.
    double widestDrift = 0.0;
    std::size_t driftLeader = 0;
    std::size_t driftRival = 0;

    /// Whether the run ordered the whole set: false when a pair was left
    /// unplaced, and false when the run held too few rounds for a band to exist
    /// at all.
    bool ordered = false;
};

/// Reads one run's ordering of the columns it was given.
///
/// The statistic is the run's own, and where the columns carry a printed figure
/// it is that figure: the row a class names as its leader is then the row the
/// class's own listing shows first, so the name and the figures under it are one
/// quantity and cannot part company when the machine is loaded. Such a figure is
/// formed inside a round, as the column's ratio to the certified lane scaled by
/// that lane's own cost per argument, so the statistic is a paired one and a
/// drift common to a round is in both terms of it and cancels. A table whose
/// columns carry no printed figure - a refinement run's own, which holds no lane
/// to pair against - falls back to the lower quartile of the column's cost per
/// argument over the rounds.
///
/// The ordering is then the paired comparison - every other column's within-round
/// ratio to the leader, banded, and placed only when the band clears one. This is
/// the whole of the probe's rule, in one function, so the classes the report
/// prints and the refinement runs that decide a tied default cannot drift apart.
///
/// \param rounds  the round table: one row per round, one column per option
/// \param columns the columns to order, in any order
/// \param figures the figure the report prints for each column, indexed the same
///                way, or empty when the table carries no printed figures
///
/// \returns the run's ordering of them, with the pairs it could not place
RunOutcome OrderColumns(const std::vector<std::vector<double>>& rounds,
                        const std::vector<std::size_t>& columns,
                        const std::vector<double>& figures = {}) {
    RunOutcome outcome;
    outcome.order = columns;

    if (columns.empty())
    {
        return outcome;
    }

    const auto statistic = [&rounds, &figures](std::size_t column) {
        if (column < figures.size() && figures[column] > 0.0)
        {
            return figures[column];
        }

        std::vector<double> costs;

        for (const std::vector<double>& row : rounds)
        {
            if (column < row.size())
            {
                costs.push_back(row[column]);
            }
        }

        return QuantileOf(costs, kStatisticQuantile);
    };

    std::sort(outcome.order.begin(), outcome.order.end(),
              [&statistic](std::size_t a, std::size_t b) {
                  return statistic(a) < statistic(b);
              });

    outcome.leader = outcome.order.front();
    outcome.widestBand = 0.0;

    for (std::size_t index = 1; index < outcome.order.size(); ++index)
    {
        const std::size_t rival = outcome.order[index];
        const PairedOutcome pair = CompareToLeader(rounds, outcome.leader, rival);

        if (pair.lo > 0.0)
        {
            outcome.widestBand = std::max(outcome.widestBand, pair.hi / pair.lo - 1.0);
        }

        if (std::abs(pair.drift) > std::abs(outcome.widestDrift))
        {
            outcome.widestDrift = pair.drift;
            outcome.driftLeader = outcome.leader;
            outcome.driftRival = rival;
        }

        // Two ways a rival is not placed behind the leader: a band that straddles
        // one, which is a pair this run cannot order, and a band that lies below
        // one, which is a pair the run's own statistic contradicts. Neither is an
        // ordering, and the probe makes neither.
        if (!pair.ordered)
        {
            outcome.unplaced.push_back(rival);
        }
    }

    outcome.ordered = outcome.unplaced.empty() && rounds.size() >= kMinimumPairedRounds;

    return outcome;
}

/// A comma-separated run of items on one line, for the short lists the classes
/// carry. Declared here rather than beside the report's other list helpers
/// because the conclusion builds its notes from it and the report is printed
/// after it.
std::string JoinNames(const std::vector<std::string>& items) {
    std::string text;

    for (const std::string& item : items)
    {
        if (!text.empty())
        {
            text += ", ";
        }

        text += item;
    }

    return text;
}

/// The columns of the measurements that are one precision's reference class: the
/// options of that precision built at the library's own full-accuracy
/// multiplier, the rung the report prints as m = 1.
///
/// The key is the rung the option was built at, never a comparison of documented
/// figures. A figure belongs to one lane: reading one lane's bound against
/// another lane's empties the other lane's class by construction — a single
/// precision's 5.5e-14 floor admits no single-precision option at all — and it
/// answers a different question besides, since two options are alternatives only
/// when they were built at the same rung.
///
/// \param report    the report whose measurements are read
/// \param precision the lane whose reference class is wanted
///
/// \returns the columns of that class's measured options, in the report's own
///          order
std::vector<std::size_t> ReferencePoolColumns(const OptionProbeReport& report,
                                              OptionPrecision precision) {
    std::vector<std::size_t> columns;

    for (std::size_t index = 0; index < report.measurements.size(); ++index)
    {
        const OptionProbeMeasurement& measurement = report.measurements[index];

        if (measurement.measured && measurement.precision == precision &&
            measurement.tier == AccuracyTier::kReference)
        {
            columns.push_back(index);
        }
    }

    return columns;
}

/// How an entry was reached, as one short phrase a reader can weigh.
const char* HowText(OptionProbeDefaultHow how) {
    switch (how)
    {
    case OptionProbeDefaultHow::kOrdered:
        return "a measured ordering of equals";
    case OptionProbeDefaultHow::kOnlyEntry:
        return "the only entry of its class, named by there being no alternative";
    case OptionProbeDefaultHow::kRefined:
        return "the vote over the refinement runs, which was unanimous";
    case OptionProbeDefaultHow::kVote:
        return "the vote over the refinement runs, which had a plurality";
    case OptionProbeDefaultHow::kChosenAmongEquals:
        return "a choice among options the run could not separate";
    case OptionProbeDefaultHow::kNone:
        break;
    }

    return "nothing named";
}

/// The figure the report prints for each column of the round table, indexed by
/// the column, and zero for a column the run formed no figure for.
///
/// This is the quantity an ordering is made in: a class's leader is the row whose
/// figure is the smallest of the class's, which is the row its listing shows
/// first, and the default is the same figure taken over the class the default is
/// read from.
std::vector<double> PrintedFigures(const OptionProbeReport& report) {
    std::vector<double> figures(report.measurements.size(), 0.0);

    for (std::size_t index = 0; index < report.measurements.size(); ++index)
    {
        if (report.measurements[index].measured)
        {
            figures[index] = report.measurements[index].nsPerArgument;
        }
    }

    return figures;
}

/// Fills in the classes, the single default and the refusal from the figures
/// already in the report.
///
/// The rule, in one place. Options are grouped into classes of one precision and
/// one rung of the accuracy axis — the multiplier an option was built at — and a
/// class is the only set this probe orders inside: two options are alternatives
/// only if they answer the same question, and one of another precision or another
/// rung answers a different one. A class of one is not a ranking and does not
/// pretend to be one: it names its entry by there being no alternative.
///
/// Inside a class the run's own statistic — the figure the report prints for each
/// option, a within-round ratio to the certified lane scaled by that lane's own
/// cost per argument — names a leader, and a rival is placed behind it only when
/// the pair's own within-round ratio clears one in the middle half of the rounds.
/// A rival whose band straddles one cannot be placed, and the class reports it
/// rather than ordering noise.
///
/// The default comes from one class: the certified double lane's precision at the
/// library's own full-accuracy multiplier, whose members were all built at that
/// rung, so choosing among them trades nothing but speed. Where that class orders
/// its members the default is its leader and the report says it is a measured
/// ordering; where the class is left tied, the default comes from the refinement
/// stage RunOptionProbe ran over the tied options alone — a vote over repeated
/// runs at a larger protocol, whose winner is reported with the vote that made it
/// — and where even that vote is split the report takes one of the tied options
/// and says that it did. What the report never does is leave the caller without a
/// default or name one chosen by counting the library's tables.
///
/// \param report the report to conclude on, whose measurements and refinement
///               stages carry the figures
/// \param rounds the round table the measurements were aggregated from, one row
///               per round and one column per option in the measurements' own
///               order, so a pair can be compared inside a round
void Conclude(OptionProbeReport& report, const std::vector<std::vector<double>>& rounds) {
    report.verdict = OptionProbeVerdict::kCannotDetermine;
    report.recommended.clear();
    report.defaultHow = OptionProbeDefaultHow::kNone;
    report.hasDefault = false;

    // The comparison is not made in the load readings, and a run whose
    // calibration window never settled still measures; the load columns then have
    // nothing to be relative to, which is what this says.
    const std::string loadNote =
        report.calibrated
            ? std::string()
            : ". The load instrument never established a floor on this machine, so the load "
              "columns of this report have nothing to be relative to; no figure above is made of "
              "them.";

    const auto refuse = [&report, &loadNote](const std::string& reason,
                                             const std::string& confidence) {
        report.reason = reason + loadNote;
        report.confidence = confidence;
    };

    std::vector<const OptionProbeMeasurement*> live;

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured)
        {
            live.push_back(&measurement);
        }
    }

    // The columns are ranked by the figures the report prints, which are formed
    // before this call from the same round table.
    const std::vector<double> figures = PrintedFigures(report);

    if (live.empty())
    {
        refuse(Text("no option produced a figure: the run took %d paired round(s), and a cost per "
                    "argument needs one reading of each option from the same round, so nothing "
                    "here is a measurement. Raise ProbeOptions::passes or ProbeOptions::rounds and "
                    "run again",
                    report.pairedRounds),
               "CANNOT DETERMINE: nothing was measured");
        return;
    }

    const OptionProbeMeasurement* overall =
        *std::min_element(live.begin(), live.end(),
                          [](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
                              return a->nsPerArgument < b->nsPerArgument;
                          });
    report.fastestOverall = overall->name;

    // The columns of one option, which is its index in the measurements: the
    // round table and the measurement rows are built from the same option book,
    // in the same order.
    const auto column_of = [&report](const OptionProbeMeasurement* measurement) {
        return static_cast<std::size_t>(measurement - report.measurements.data());
    };

    const auto name_of = [&report](std::size_t column) {
        return report.measurements[column].name;
    };

    // ---- the classes: one precision at one rung ------------------------------
    struct ClassRecord {
        OptionProbeClass entry;
        std::vector<std::size_t> columns;
        RunOutcome outcome;
    };

    std::vector<ClassRecord> records;

    for (const OptionPrecision precision :
         {OptionPrecision::kFp64, OptionPrecision::kFp32, OptionPrecision::kFp16,
          OptionPrecision::kBf16})
    {
        bool anyMeasured = false;

        for (const AccuracyTier tier : ServedTiers())
        {
            ClassRecord record;
            record.entry.precision = precision;
            record.entry.tier = tier;
            record.entry.name =
                Text("%s m=%g", PrecisionName(precision), AccuracyMultiplier(tier));

            for (const OptionProbeMeasurement* measurement : live)
            {
                if (measurement->precision == precision && measurement->tier == tier)
                {
                    record.columns.push_back(column_of(measurement));
                }
            }

            if (record.columns.empty())
            {
                continue;
            }

            anyMeasured = true;
            record.outcome = OrderColumns(rounds, record.columns, figures);

            // One entry is not a ranking: a class holding a single option is
            // measured and not ordered, however many rounds the run took, since
            // there was nothing in it to measure that option against.
            record.entry.ordered = record.outcome.ordered && record.columns.size() > 1;

            const OptionProbeMeasurement& leader = report.measurements[record.outcome.leader];
            record.entry.bound = leader.bound;
            record.entry.leader = leader.name;
            record.entry.leaderNsPerArgument = leader.nsPerArgument;

            for (const std::size_t column : record.outcome.order)
            {
                record.entry.ranked.push_back(name_of(column));
            }

            for (const std::size_t column : record.outcome.unplaced)
            {
                record.entry.unplaced.push_back(name_of(column));
            }

            // A class is keyed on the rung and the precision, not on a bound, so
            // a row of the same precision and rung that documents a looser figure
            // is in the class: it is reported by name here rather than ranked as
            // an equal silently.
            for (const std::size_t column : record.columns)
            {
                const OptionProbeMeasurement& member = report.measurements[column];

                if (member.name != leader.name && member.bound != leader.bound)
                {
                    record.entry.differingBounds.push_back(
                        Text("%s documents %.3g", member.name.c_str(), member.bound));
                }
            }

            records.push_back(std::move(record));
        }

        if (!anyMeasured)
        {
            OptionProbeClass entry;
            entry.precision = precision;
            entry.tier = AccuracyTier::kReference;
            entry.name = Text("%s m=1", PrecisionName(precision));
            entry.note = "no option of this precision produced a figure on this run";
            report.classes.push_back(entry);
        }
    }

    const auto refinement_of = [&report](OptionPrecision precision) {
        for (const OptionProbeRefinement& stage : report.refinements)
        {
            if (stage.precision == precision)
            {
                return &stage;
            }
        }

        return static_cast<const OptionProbeRefinement*>(nullptr);
    };

    const auto class_of = [&records](OptionPrecision precision, AccuracyTier tier) {
        for (const ClassRecord& record : records)
        {
            if (record.entry.precision == precision && record.entry.tier == tier)
            {
                return &record;
            }
        }

        return static_cast<const ClassRecord*>(nullptr);
    };

    // What this run can order, as the widest relative width of a within-round
    // band the certified lane's classes showed - a rival's band against its own
    // class's leader. It is reported beside the figures; the ordering itself is
    // made pair by pair from each pair's own band, so this number bars nothing.
    if (report.pairedRounds >= static_cast<int>(kMinimumPairedRounds))
    {
        for (const ClassRecord& record : records)
        {
            if (record.entry.precision == OptionPrecision::kFp64)
            {
                report.resolution = std::max(report.resolution, record.outcome.widestBand);
            }
        }
    }

    // ---- the default: one combination, from the certified lane's own rung -----
    const std::vector<std::size_t> pool = ReferencePoolColumns(report, OptionPrecision::kFp64);
    const ClassRecord* doubles = class_of(OptionPrecision::kFp64, AccuracyTier::kReference);

    if (pool.empty() || doubles == nullptr)
    {
        refuse(Text("no option of the certified double lane's precision at the library's own "
                    "full-accuracy multiplier was measured, and that class is the only pool this "
                    "probe takes a default from: naming any other option would be naming one built "
                    "at a different rung or in a different arithmetic, which is an answer to a "
                    "different question. The run measured %zu option(s); the fastest of them is "
                    "'%s' at %.2f ns/argument, documented at %.3g",
                    live.size(), overall->name.c_str(), overall->nsPerArgument, overall->bound),
               "CANNOT DETERMINE: the certified lane's reference class is empty");
        return;
    }

    const RunOutcome pooled = OrderColumns(rounds, pool, figures);
    report.fastestAtReferenceAccuracy = name_of(pooled.leader);

    const OptionProbeMeasurement& fastest = report.measurements[pooled.leader];
    const OptionProbeRefinement* stage = refinement_of(OptionPrecision::kFp64);

    if (pool.size() == 1)
    {
        // Read before the ordering branch, not after it: a class holding one
        // option holds no rival to be placed behind it, so the ordering test
        // would pass on any run long enough for a band and the entry would be
        // reported as the winner of a comparison that never happened.
        report.recommended = fastest.name;
        report.defaultHow = OptionProbeDefaultHow::kOnlyEntry;
    } else if (pooled.ordered)
    {
        report.recommended = fastest.name;
        report.defaultHow = OptionProbeDefaultHow::kOrdered;
    } else if (stage != nullptr && stage->ran && !stage->winner.empty())
    {
        report.recommended = stage->winner;
        report.defaultHow = stage->unanimous
                                ? OptionProbeDefaultHow::kRefined
                                : (stage->plurality ? OptionProbeDefaultHow::kVote
                                                    : OptionProbeDefaultHow::kChosenAmongEquals);
    } else
    {
        // No refinement ran on the pool — which happens when the caller narrowed
        // the run to a set the stage could not re-run, or when the run was too
        // short for one — so the class's own fastest is the default and the
        // choice among the tied options is reported as one.
        report.recommended = fastest.name;
        report.defaultHow = OptionProbeDefaultHow::kChosenAmongEquals;
    }

    // ---- the classes' own entries, now that the default is known --------------
    for (ClassRecord& record : records)
    {
        const bool referenceRung = record.entry.tier == AccuracyTier::kReference;
        const OptionProbeRefinement* own = refinement_of(record.entry.precision);

        if (record.entry.precision == OptionPrecision::kFp64 && referenceRung)
        {
            // The class the default is taken from reports the default's own how,
            // whichever way it was reached.
            record.entry.how = report.defaultHow;
        } else if (!referenceRung)
        {
            record.entry.how = record.entry.ordered ? OptionProbeDefaultHow::kOrdered
                                                    : OptionProbeDefaultHow::kChosenAmongEquals;
        } else if (record.columns.size() == 1)
        {
            record.entry.how = OptionProbeDefaultHow::kOnlyEntry;
        } else if (record.outcome.ordered)
        {
            record.entry.how = OptionProbeDefaultHow::kOrdered;
        } else if (own != nullptr && own->ran && !own->winner.empty())
        {
            record.entry.how = own->unanimous
                                   ? OptionProbeDefaultHow::kRefined
                                   : (own->plurality ? OptionProbeDefaultHow::kVote
                                                     : OptionProbeDefaultHow::kChosenAmongEquals);
        } else
        {
            record.entry.how = OptionProbeDefaultHow::kChosenAmongEquals;
        }

        const std::string named =
            (record.entry.precision == OptionPrecision::kFp64 && referenceRung)
                ? report.recommended
                : (own != nullptr && own->ran && !own->winner.empty() ? own->winner
                                                                     : record.entry.leader);

        std::string note;

        if (record.columns.size() == 1)
        {
            note = Text("one entry, so this is not a ranking: nothing in this class was measured "
                        "against it, and it is the entry this class names by there being no "
                        "alternative. It was built at m=%g and documents %.3g",
                        AccuracyMultiplier(record.entry.tier), record.entry.bound);
        } else if (report.pairedRounds < static_cast<int>(kMinimumPairedRounds))
        {
            note = Text("not ordered: the run took %d paired round(s), and a band over the lower "
                        "and upper quartiles of the within-round ratios needs %zu, so nothing in "
                        "this class was placed. The refinement stage is where an entry is reached "
                        "from here",
                        report.pairedRounds, kMinimumPairedRounds);
        } else if (record.outcome.unplaced.empty() && record.outcome.ordered)
        {
            note = Text("ordered: every one of the %zu option(s) behind '%s' was the slower of the "
                        "two in the middle half of the %d paired rounds, and the widest band in "
                        "the class was %.2f%%. Every row of this class was built at m=%g, so this "
                        "is the fastest option at that rung and nothing in the class traded "
                        "accuracy for its place",
                        record.columns.size() - 1, record.entry.leader.c_str(), report.pairedRounds,
                        100.0 * record.outcome.widestBand,
                        AccuracyMultiplier(record.entry.tier));
        } else
        {
            note = Text("not ordered: %zu of the %zu option(s) of this class could not be placed "
                        "behind '%s' - their within-round ratio to it did not clear one in the "
                        "middle half of the rounds - and the widest band the class showed was "
                        "%.2f%%",
                        record.entry.unplaced.size(), record.columns.size(),
                        record.entry.leader.c_str(), 100.0 * record.outcome.widestBand);
        }

        if (named != record.entry.leader)
        {
            note += Text(". The entry this class names is '%s', reached by %s", named.c_str(),
                         HowText(record.entry.how));
        }

        if (!record.entry.differingBounds.empty())
        {
            note += Text(". Not every row of this class documents the leader's %.3g: %s - a class "
                         "is keyed on the rung and the precision, so a row of another figure is in "
                         "it and is reported here rather than being named the class's entry by "
                         "its cost alone",
                         record.entry.bound, JoinNames(record.entry.differingBounds).c_str());
        }

        record.entry.note = std::move(note);
        report.classes.push_back(record.entry);
    }

    // ---- the answer -----------------------------------------------------------
    //
    // The clock check, done rather than assumed, and attached to whatever the
    // default turns out to be. A pair whose ratio moves between the run's halves
    // is a pair whose two options do not carry a decaying clock alike, and a
    // report that ordered without saying so would imply the ordering holds at any
    // clock. Which clock an option draws is a property of the registers it runs
    // in, so the clause also says whether the pool this choice compares put two
    // different arithmetic routes - two vector register widths - against each
    // other.
    const auto clock_clause = [&]() -> std::string {
        if (pool.size() < 2 || report.pairedRounds < static_cast<int>(kMinimumPairedRounds))
        {
            return std::string();
        }

        std::string clause;

        if (std::abs(pooled.widestDrift) > report.resolution)
        {
            clause = Text(". WARNING: the pair '%s' against '%s' moved %.1f%% between the run's "
                          "first and second half of rounds, beyond the %.2f%% this run can order, "
                          "so those two options are not equally exposed to this machine's clock "
                          "and the entry above is a property of this run's clock as well as of the "
                          "options",
                          name_of(pooled.driftRival).c_str(), name_of(pooled.driftLeader).c_str(),
                          100.0 * pooled.widestDrift, 100.0 * report.resolution);
        } else
        {
            clause = Text(". No pair of the class moved between the run's halves by more than the "
                          "%.2f%% this run can order (the widest was '%s' against '%s' at %.1f%%), "
                          "so no measured pair was more exposed to the clock's drift than the "
                          "comparison's own precision",
                          100.0 * report.resolution, name_of(pooled.driftRival).c_str(),
                          name_of(pooled.driftLeader).c_str(), 100.0 * pooled.widestDrift);
        }

        const std::string& route = report.measurements[pooled.leader].arithmetic;
        const bool one_route = std::all_of(
            pool.begin(), pool.end(), [&report, &route](std::size_t column) {
                return report.measurements[column].arithmetic == route;
            });

        if (one_route)
        {
            clause += Text(". Every option of that class runs the same arithmetic - %s for all %zu "
                           "of them - so no pair here sets a wider vector register against a "
                           "narrower one, and the frequency a wider register draws is not a "
                           "difference between these options: what this run measured, above, is "
                           "what is reported in place of assuming it",
                           route.c_str(), pool.size());
        } else
        {
            clause += ". The class does not run one arithmetic throughout, so a pair of it may set "
                      "a wider vector register against a narrower one and the two may draw "
                      "different clocks; the drift above is the whole of what this run measured "
                      "about that, and nothing here sets it aside";
        }

        return clause;
    };

    // The rival lines: what the class could not place behind its fastest, each
    // with its own band and round counts. They are reported for a default reached
    // by ordering as well as for one reached by the vote, because the reader needs
    // to see how far the class itself got.
    for (const std::size_t column : pooled.unplaced)
    {
        const OptionProbeMeasurement& rival = report.measurements[column];
        const PairedOutcome pair = CompareToLeader(rounds, pooled.leader, column);

        report.inseparable.push_back(Text(
            "'%s' at %.2f ns/argument, %.1f%% of the fastest's: its within-round ratio to '%s' "
            "over the %d paired rounds fell in %.3f..%.3f, and it was the slower of the two in %d "
            "of them (its own rounds spread %.2fx)",
            rival.name.c_str(), rival.nsPerArgument,
            100.0 * (rival.nsPerArgument / fastest.nsPerArgument), fastest.name.c_str(),
            pair.rounds, pair.lo, pair.hi, pair.slowerRounds, rival.spread));
    }

    switch (report.defaultHow)
    {
    case OptionProbeDefaultHow::kOrdered:
        report.reason = Text(
            "'%s' is the default: it is the fastest option of the certified double lane's "
            "reference class on this machine - %.2f ns/argument at the lower quartile of the %d "
            "paired rounds, %.2f at the upper - and every one of the other %zu option(s) of that "
            "class was the slower of the two in the middle half of those rounds, so this is a "
            "measured ordering of equals and not a choice. The class is one precision at one rung "
            "- every row of it was built at m=1 - so nothing in it traded accuracy for speed. It "
            "measured %.3g against the certified lane over this workload",
            report.recommended.c_str(), fastest.nsPerArgument, report.pairedRounds,
            fastest.nsPerArgumentMax, pool.size() - 1, fastest.maxError);
        break;
    case OptionProbeDefaultHow::kOnlyEntry:
        report.reason = Text(
            "'%s' is the default: it is the only option of the certified double lane's reference "
            "class this run measured, at %.2f ns/argument over the %d paired rounds. One entry is "
            "not a ranking - there was nothing to measure it against - and it is named by there "
            "being no alternative, not as the winner of a comparison. It measured %.3g against "
            "the certified lane over this workload",
            report.recommended.c_str(), fastest.nsPerArgument, report.pairedRounds,
            fastest.maxError);
        break;
    default:
        report.reason = Text(
            "'%s' is the default: the certified double lane's reference class holds %zu measured "
            "option(s) built at m=1, and its fastest by this run's own statistic is '%s' at %.2f "
            "ns/argument. ",
            report.recommended.c_str(), pool.size(), fastest.name.c_str(), fastest.nsPerArgument);

        if (report.pairedRounds < static_cast<int>(kMinimumPairedRounds))
        {
            report.reason += Text(
                "Nothing in that class could be ordered: the run took %d paired round(s) and a "
                "band over the lower and upper quartiles of the within-round ratios needs %zu, so "
                "no pair of it was placed either way and the class's widest band does not exist. ",
                report.pairedRounds, kMinimumPairedRounds);
        } else
        {
            report.reason += Text(
                "%zu of the others could not be placed behind it - their within-round ratio to it "
                "did not clear one in the middle half of the %d paired rounds, and ordering a pair "
                "whose band straddles one would be ordering noise. ",
                pooled.unplaced.size(), report.pairedRounds);
        }

        if (stage != nullptr && stage->ran)
        {
            report.reason += Text(
                "The %zu option(s) the class left tied were then re-run on their own, %d time(s) "
                "at %d passes by %d rounds, and %s. ",
                stage->pool.size(), stage->runs, stage->passes, stage->rounds,
                stage->note.c_str());
        } else
        {
            report.reason += "No refinement stage ran on this set, so the choice among the tied "
                             "options was made without one. ";
        }

        report.reason += "The default is named with the way it was reached - " +
                         std::string(HowText(report.defaultHow)) +
                         " - and not as a measured ordering, which this run did not establish.";
        break;
    }

    std::string confidence;

    switch (report.defaultHow)
    {
    case OptionProbeDefaultHow::kOrdered:
        confidence = Text(
            "HIGH: the nearest option of the class, '%s', took %.1f%% of the default's cost and "
            "was the slower of the two in %d of the %d paired rounds, its within-round band "
            "%.3f..%.3f clearing one",
            pooled.order.size() > 1 ? name_of(pooled.order[1]).c_str() : "",
            pooled.order.size() > 1
                ? 100.0 * (report.measurements[pooled.order[1]].nsPerArgument /
                           fastest.nsPerArgument)
                : 0.0,
            pooled.order.size() > 1
                ? CompareToLeader(rounds, pooled.leader, pooled.order[1]).slowerRounds
                : 0,
            report.pairedRounds,
            pooled.order.size() > 1 ? CompareToLeader(rounds, pooled.leader, pooled.order[1]).lo
                                    : 1.0,
            pooled.order.size() > 1 ? CompareToLeader(rounds, pooled.leader, pooled.order[1]).hi
                                    : 1.0);
        break;
    case OptionProbeDefaultHow::kRefined:
        confidence = Text(
            "HIGH: the default is the class's own fastest, and the refinement runs agreed with "
            "this run: one of them led with '%s' in every one of the %d runs, at a protocol %d "
            "times the main one, so no rival of the tied band led a single run",
            report.recommended.c_str(), stage != nullptr ? stage->runs : 0,
            report.options.refinementFactor);
        break;
    case OptionProbeDefaultHow::kVote:
        confidence = Text(
            "MEDIUM: the class's own rounds could not separate the tied options and neither did "
            "every refinement run; '%s' led %d of the %d runs. A majority over runs is weaker "
            "evidence than a measured ordering, and this is one",
            report.recommended.c_str(),
            report.recommended == fastest.name ? 1 : 0, stage != nullptr ? stage->runs : 0);
        break;
    case OptionProbeDefaultHow::kChosenAmongEquals:
        confidence = Text(
            "LOW: this run could not separate the options of the class, the vote over the "
            "refinement runs had no plurality, and '%s' is one of the tied options taken by "
            "choice - the way it was reached is stated above and it is not a ranking",
            report.recommended.c_str());
        break;
    case OptionProbeDefaultHow::kOnlyEntry:
        confidence = Text(
            "LOW: one entry is not a ranking and this class holds one. The option is the default "
            "by there being no alternative in its precision at its rung, and its cost above is "
            "the whole of what this run measured about it");
        break;
    case OptionProbeDefaultHow::kNone:
        break;
    }

    confidence += Text(". The class's own rounds left %zu of %zu option(s) unplaced and its "
                       "widest band was %.2f%%",
                       pooled.unplaced.size(), pool.size(), 100.0 * pooled.widestBand);

    confidence += clock_clause();

    if (overall != &fastest && overall->bound > fastest.bound)
    {
        confidence += Text(
            ". The fastest option measured overall is '%s' at %.2f ns/argument, documented at "
            "%.3g, a looser accuracy: faster, not faster at the same accuracy",
            overall->name.c_str(), overall->nsPerArgument, overall->bound);
    }

    if (report.calibrated)
    {
        confidence += Text(". The passes ran at a median load of %.1f%% inside their timed rounds",
                           report.loadMedian);

        if (report.passesAboveAlarm > 0)
        {
            confidence += Text(
                ", and %d of %zu pass(es) ran with the canary's own runs wider than the %.1f%% "
                "alarm - reported, used, and not what the entry rests on",
                report.passesAboveAlarm, report.passes.size(), report.options.canarySpreadAlarm);
        }
    } else
    {
        confidence += ". The load instrument never established a floor on this machine, so no load "
                      "reading and no canary run went beside this run's paired rounds, and nothing "
                      "here is read from one";
    }

    report.confidence = confidence;
    report.hasDefault = !report.recommended.empty();
    report.verdict =
        report.hasDefault ? OptionProbeVerdict::kRecommend : OptionProbeVerdict::kCannotDetermine;
}

/// The refinement stage: the tied options of one precision's reference class,
/// measured alone at a larger protocol, repeated, and voted on.
///
/// This is what the report does instead of naming an entry from a figure counted
/// off the library's tables. The options re-measured are the class's fastest and
/// every option of it the main run could not place behind the fastest; nothing
/// else in the option space is touched, so the stage's whole cost is spent on the
/// pair the answer actually rests on.
///
/// Each run is a fresh pass over the tied set at ProbeOptions::passes times
/// ProbeOptions::refinementFactor passes of ProbeOptions::rounds times the same
/// factor rounds, with its own shuffle, and each run is ordered by the same
/// OrderColumns the main run used. A refinement run's table holds the tied set
/// alone - no certified lane to pair against - so its columns carry no printed
/// figure and the run is ordered on its own cost per argument, which is the one
/// quantity that table has; nothing else about the rule changes, so a run is a
/// smaller measurement of the same kind and not a different one. The vote is over
/// the runs, and a run whose own rounds cannot place a rival contributes its
/// leader alone, which is what makes the vote a vote and not a re-run of the main
/// statistic.
///
/// \param report    the report the stage is recorded in
/// \param book      the options the run measured, in the report's own order
/// \param work      the workload, already built and warmed
/// \param buffers   the scratch the calls need
/// \param options   the protocol in force, read for the run count and the factor
/// \param precision the lane whose reference class is refined
/// \param pool      the columns of the class's measured options
/// \param mainLeader the class's fastest by the main run's own rounds, which is
///                  the entry the stage falls back to when no run places one
void Refine(OptionProbeReport& report, const std::vector<Option>& book, const Workload& work,
            Buffers& buffers, const ProbeOptions& options, OptionPrecision precision,
            const std::vector<std::size_t>& pool, const std::string& mainLeader) {
    OptionProbeRefinement stage;
    stage.precision = precision;
    stage.ran = true;
    stage.runs = std::max(1, options.refinementRuns);
    stage.passes = std::max(1, options.passes) * std::max(1, options.refinementFactor);
    stage.rounds = std::max(1, options.rounds) * std::max(1, options.refinementFactor);

    for (const std::size_t column : pool)
    {
        stage.pool.push_back(report.measurements[column].name);
    }

    std::vector<std::size_t> visit(pool.size());
    std::vector<std::size_t> columns(pool.size());

    for (std::size_t slot = 0; slot < pool.size(); ++slot)
    {
        visit[slot] = slot;
        columns[slot] = slot;
    }

    std::vector<std::string> winners;

    for (int run = 0; run < stage.runs; ++run)
    {
        // A fresh seed per run: repetitions of one shuffle would be one run taken
        // several times, and whatever a position in the round is worth would be
        // worth the same to the same option every time.
        std::mt19937_64 shuffle(options.seed +
                                static_cast<std::uint64_t>(run + 1) * 0x9e3779b97f4a7c15ull);
        std::vector<std::vector<double>> table;

        for (int pass = 0; pass < stage.passes; ++pass)
        {
            for (int round = 0; round < stage.rounds; ++round)
            {
                std::shuffle(visit.begin(), visit.end(), shuffle);

                std::vector<double> row(pool.size(), 0.0);

                for (const std::size_t slot : visit)
                {
                    const Clock::time_point t0 = Clock::now();
                    const double sum = Checksum(work, buffers, book[pool[slot]]);
                    const Clock::time_point t1 = Clock::now();
                    (void)sum;

                    row[slot] =
                        MillisecondsBetween(t0, t1) * 1e6 / static_cast<double>(options.count);
                }

                table.push_back(std::move(row));
            }
        }

        const RunOutcome outcome = OrderColumns(table, columns);

        winners.push_back(outcome.order.empty() ? std::string()
                                                : report.measurements[pool[outcome.leader]].name);
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
        stage.tally.push_back(Text("%s won %d of %d run(s)", candidates[index].c_str(),
                                   counts[index], stage.runs));
    }

    if (candidates.empty())
    {
        // No run placed a leader at all: the tied set was too small or too fast
        // for the statistic to form, so the class's own fastest stands and the
        // stage says that rather than inventing a winner.
        stage.winner = mainLeader;
        stage.note = Text("no run placed a leader at all, so the class's own fastest, '%s', was "
                          "kept and the runs are reported above",
                          mainLeader.c_str());
    } else
    {
        stage.winner = candidates[bestIndex];
        stage.unanimous = (tiedAtBest == 1 && best == stage.runs);
        stage.plurality = (tiedAtBest == 1);

        if (stage.unanimous)
        {
            stage.note = Text("every one of the %d run(s) led with '%s', so the vote was unanimous",
                              stage.runs, stage.winner.c_str());
        } else if (stage.plurality)
        {
            stage.note = Text("'%s' led %d of the %d run(s), more than any other candidate, so the "
                              "vote had a plurality and not a unanimous result",
                              stage.winner.c_str(), best, stage.runs);
        } else
        {
            stage.note = Text(
                "the vote was split: %d option(s) led %d run(s) each, so '%s' was taken from among "
                "them by choice - any of them is equally good on this evidence, and this one is "
                "named so that the report ends with one default",
                tiedAtBest, best, stage.winner.c_str());
        }
    }

    report.refinements.push_back(std::move(stage));
}
} // namespace

OptionProbeReport RunOptionProbe(const ProbeOptions& requested) {
    OptionProbeReport report;
    ProbeOptions options = requested;

    if (options.count == 0)
    {
        options.count = 1;
    }

    if (options.nmax < 1)
    {
        options.nmax = 1;
    }

    if (options.nmax > kMaxBoysOrder)
    {
        options.nmax = kMaxBoysOrder;
    }

    if (options.passes < 1)
    {
        options.passes = 1;
    }

    if (options.rounds < 1)
    {
        options.rounds = 1;
    }

    if (!(options.xHi > options.xLo) || !(options.xLo > 0.0))
    {
        options.xLo = 1e-3;
        options.xHi = 40.0;
    }

    report.options = options;
    report.logicalProcessors = static_cast<int>(std::thread::hardware_concurrency());
    report.avx2 = BoysAvx2Available();
    report.backends = backend::BoysBackends();
    report.granularities = BoysFitGranularities();
    report.referenceBound = TierBound(AccuracyTier::kReference);

    const Workload work = BuildWorkload(options);
    Buffers buffers = MakeBuffers(work);
    report.orderRuns = work.runOrder.size();
    report.largestRun = work.largestRun;

    // The coverage book is built before the selection is applied: it is the
    // library's option space, so a narrowed run accounts for every cell of it
    // just as a full one does.
    const std::vector<OptionProbeCell> cells = EnumerateCells();
    report.cells = cells;
    std::vector<Option> options_ = EnumerateOptions(report.backends, report.granularities,
                                                    cells, report.unoffered, report.notCarried);

    // A caller who names a set is answered about that set, and every conclusion
    // below is drawn from what was measured, so the set is narrowed before
    // anything is measured rather than filtered out of the report afterwards.
    // A name is one of three things and the report keeps them apart: an option
    // this build serves, a cell of the option space the library refuses where it
    // is named, and a name that is no cell of the space at all. The last is
    // recorded rather than silently measuring nothing, because an empty report
    // otherwise reads as a machine on which nothing is fast; the middle one is
    // recorded with the library's reason, because it is unbuilt work rather than
    // a misspelling.
    if (!options.only.empty())
    {
        const auto asked = [&](const std::string& name) {
            return std::find(options.only.begin(), options.only.end(), name) != options.only.end();
        };

        for (const std::string& name : options.only)
        {
            const bool known =
                std::any_of(options_.begin(),
                            options_.end(),
                            [&](const Option& tried) { return tried.name == name; }) ||
                std::find(report.unoffered.begin(), report.unoffered.end(), name) !=
                    report.unoffered.end() ||
                std::find(report.notCarried.begin(), report.notCarried.end(), name) !=
                    report.notCarried.end();

            if (known)
            {
                continue;
            }

            const auto cell = std::find_if(cells.begin(), cells.end(), [&](const OptionProbeCell& c) {
                return c.name == name;
            });

            if (cell != cells.end() && !cell->served)
            {
                report.refused.push_back(Text("%s — %s", name.c_str(), cell->reason.c_str()));
                continue;
            }

            report.notAnOption.push_back(name);
        }

        options_.erase(std::remove_if(options_.begin(),
                                      options_.end(),
                                      [&](const Option& tried) { return !asked(tried.name); }),
                       options_.end());
        report.unoffered.erase(
            std::remove_if(report.unoffered.begin(),
                           report.unoffered.end(),
                           [&](const std::string& name) { return !asked(name); }),
            report.unoffered.end());
        report.notCarried.erase(
            std::remove_if(report.notCarried.begin(),
                           report.notCarried.end(),
                           [&](const std::string& name) { return !asked(name); }),
            report.notCarried.end());
    }

    report.measurements.resize(options_.size());

    // Warm-up: every option once, so the tables, the stack pages and the
    // workspace are resident before the first timed round, and the load
    // instrument has seen the machine with this workload's own working set
    // live.
    for (const Option& option : options_)
    {
        const double warm = Checksum(work, buffers, option);
        (void)warm;
    }

    LoadMeter meter;
    report.calibrated = meter.Calibrate(options.calibrationSeconds, kCalibrationTolerancePercent);
    report.canaryFloorMs = meter.FloorMs();
    report.canaryCalibrationRuns = meter.Runs();
    report.canaryCalibrationMedianMs = meter.MedianMs();
    report.canaryCalibrationSpread = meter.CalibrationSpreadPercent();

    // The round table: one row per timed round in the order the rounds ran, one
    // column per option, each cell the cost per argument of that option's call
    // in that round. Every figure below is an aggregate of ratios taken inside a
    // row of this table, which is what makes it a paired comparison: two cells
    // of one row were timed under one clock.
    std::vector<std::vector<double>> roundCost;

    // The option every ratio is formed against, fixed here from the option book
    // rather than chosen from the figures afterwards, so the anchor is a
    // documented choice and not the winner of the comparison it anchors.
    const std::size_t reference = ReferenceIndex(options_);
    std::vector<double> referenceCost;
    std::vector<double> passLoads;

    // The canary's runs inside a pass: one before the first round and one after
    // every round, so the set brackets the timed work and spans it. Each is
    // taken between rounds, never inside a timed region, so the instrument can
    // never land in a figure. At least three, because a spread needs a set that
    // can disagree with itself.
    const int canarySamples = std::max(3, options.rounds + 1);

    // The visit order is shuffled once per round from the workload's own seed.
    // A fixed order would put the same option first in every round, and whatever
    // a position in the round is worth - the first call touching a table the
    // others then find warm - would be worth the same to that option every time
    // and would enter its ratio as though it were the option's own cost.
    // Shuffling makes a position average out instead of accumulating, and the
    // seed keeps a run reproducible.
    std::mt19937_64 shuffle(options.seed);
    std::vector<std::size_t> visit(options_.size());

    for (std::size_t slot = 0; slot < visit.size(); ++slot)
    {
        visit[slot] = slot;
    }

    // Every pass is run and every pass is used. The canary beside it is read and
    // reported; it decides nothing, because a fixed work read by wall clock
    // measures the clock as much as the load, and a machine whose boost decays
    // widens the spin's own spread while doing nothing else. Discarding on that
    // would discard the measurement rather than the machine.
    for (int pass = 0; pass < options.passes; ++pass)
    {
        OptionProbePass record;
        std::vector<double> canaryMs;
        std::vector<std::vector<double>> passRounds;

        if (report.calibrated)
        {
            record.backgroundLoad = meter.Average(options.backgroundWindowSeconds);
            canaryMs.push_back(meter.SampleMs());
            record.beforeLoad = meter.Percent(canaryMs.back());
        }

        for (int round = 0; round < options.rounds; ++round)
        {
            std::shuffle(visit.begin(), visit.end(), shuffle);

            std::vector<double> row(options_.size(), 0.0);

            for (const std::size_t index : visit)
            {
                const Clock::time_point t0 = Clock::now();
                const double sum = Checksum(work, buffers, options_[index]);
                const Clock::time_point t1 = Clock::now();

                const double milliseconds = MillisecondsBetween(t0, t1);
                row[index] = milliseconds * 1e6 / static_cast<double>(options.count);
                record.seconds += SecondsBetween(t0, t1);
                report.measurements[index].checkedSum = sum;
            }

            passRounds.push_back(row);
            roundCost.push_back(std::move(row));

            if (report.calibrated)
            {
                canaryMs.push_back(meter.SampleMs());
            }
        }

        if (report.calibrated)
        {
            while (static_cast<int>(canaryMs.size()) < canarySamples)
            {
                canaryMs.push_back(meter.SampleMs());
            }

            record.afterLoad = meter.Percent(canaryMs.back());
            record.inPassLoad = meter.Percent(MedianMilliseconds(canaryMs));
            record.canarySpread = SpreadPercent(canaryMs);
            record.canaryWide = record.canarySpread > options.canarySpreadAlarm;

            if (record.canaryWide)
            {
                ++report.passesAboveAlarm;
            } else
            {
                ++report.passesWithinAlarm;
            }

            report.canarySpread = std::max(report.canarySpread, record.canarySpread);
            passLoads.push_back(record.inPassLoad);
        }

        // The pass's own paired spread: the within-round ratios of this pass,
        // which is the spread the ordering is made in. Reported beside the
        // canary's, and it gates nothing either - the resolution below is what
        // says whether a pair could be separated.
        record.pairedSpread = PassPairedSpread(passRounds, reference);
        report.passes.push_back(record);
    }

    report.loadMedian = MedianMilliseconds(passLoads);
    report.pairedRounds = static_cast<int>(roundCost.size());

    // The accuracy column, measured once per option outside the clock. It does
    // not depend on a pass: the values a lane returns for an argument are a
    // property of the build, not of the machine's load.
    for (std::size_t index = 0; index < options_.size(); ++index)
    {
        AccuracyWalker walker;
        VisitValues(work, buffers, options_[index], walker);

        OptionProbeMeasurement& measurement = report.measurements[index];
        measurement.name = options_[index].name;
        measurement.arithmetic = options_[index].arithmetic;
        measurement.contracts = options_[index].contracts;
        measurement.precision = options_[index].precision;
        measurement.tier = options_[index].tier;
        measurement.route = options_[index].route;
        measurement.scheme = options_[index].scheme;
        measurement.granularity = options_[index].granularity;
        measurement.pack = options_[index].pack;
        measurement.bound = options_[index].bound;
        measurement.ownBound = options_[index].ownBound;
        measurement.ownLo = options_[index].ownLo;
        measurement.ownHi = options_[index].ownHi;
        measurement.maxError = walker.MaxError();
        measurement.bitIdenticalToReference = walker.BitIdentical();

        // One figure judges the whole row, and it is a whole-domain one: the
        // entry's own, which the library reports through QueryTier whatever
        // partition the entry reads. A partition's own figures are certified for
        // the fitted interval alone, and the measured column is a difference from
        // the certified lane rather than an error against the true function, so
        // judging the wider measurement by the narrower figure would report the
        // partition as missing a promise it never made — the row carries that
        // figure and its interval as information instead.
        measurement.meetsBound = measurement.maxError <= measurement.bound;
        measurement.withinReferenceFloor =
            !measurement.meetsBound && measurement.maxError <= report.referenceBound;
    }

    // The paired figures. Each option's ratios to the reference are formed
    // inside the round its two calls were timed in, so a drift common to the
    // round - a clock that decayed between this round and the last, a background
    // process that stole time from both - is in both terms of the ratio and
    // cancels. The statistic is a lower quartile: the minimum of these rounds
    // would be the earliest, highest-clock observation, which is not what a
    // caller with a long workload meets, and a mean would let one disturbed
    // round move the figure.
    if (!roundCost.empty() && reference < options_.size())
    {
        report.referenceOption = options_[reference].name;

        for (const std::vector<double>& row : roundCost)
        {
            referenceCost.push_back(row[reference]);
        }

        report.referenceNsPerArgument = QuantileOf(referenceCost, kStatisticQuantile);

        const std::size_t half = roundCost.size() / 2;

        for (std::size_t index = 0; index < options_.size(); ++index)
        {
            OptionProbeMeasurement& measurement = report.measurements[index];
            std::vector<double> ratios;
            std::vector<double> early;
            std::vector<double> late;
            double peak = std::numeric_limits<double>::infinity();

            for (std::size_t round = 0; round < roundCost.size(); ++round)
            {
                const double anchor = roundCost[round][reference];

                if (anchor <= 0.0)
                {
                    continue;
                }

                const double cost = roundCost[round][index];
                ratios.push_back(cost / anchor);
                peak = std::min(peak, cost);

                if (round < half)
                {
                    early.push_back(cost / anchor);
                } else
                {
                    late.push_back(cost / anchor);
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
            measurement.nsPerArgument = report.referenceNsPerArgument * measurement.ratioToReference;
            measurement.nsPerArgumentMax = report.referenceNsPerArgument * measurement.ratioHi;
            measurement.nsPerArgumentPeak = peak;
            measurement.spread = measurement.ratioLo > 0.0
                                     ? measurement.ratioHi / measurement.ratioLo
                                     : 1.0;

            // How far the option's ratio to the lane moved between the run's
            // halves. Zero means the option and the lane kept pace as the clock
            // moved, which is what a ratio that is genuinely the option's own
            // cost looks like; a value away from zero is a warning that the two
            // do not carry the clock alike, and the report says so where it
            // prints this option.
            const double firstHalf = MedianMilliseconds(early);
            const double secondHalf = MedianMilliseconds(late);

            if (firstHalf > 0.0)
            {
                measurement.ratioDrift = secondHalf / firstHalf - 1.0;
            }
        }
    }

    // The refinement stages, taken before the conclusion so that the conclusion
    // can read their votes. A reference class the main run could not order - and
    // which holds more than one option, since one entry is not a ranking to
    // refine - is re-measured on its own at a larger protocol and voted on. A
    // class the main run ordered outright is left alone: it is already a
    // measurement, and a refinement invented for it would be a re-run of an
    // answer that stands.
    //
    // The tied set is read the same way the conclusion reads it: off the figures
    // the report prints, so the options re-run here are the ones the classes
    // below will report as the pair that was left unplaced.
    const std::vector<double> printed = PrintedFigures(report);

    for (const OptionPrecision precision :
         {OptionPrecision::kFp64, OptionPrecision::kFp32, OptionPrecision::kFp16,
          OptionPrecision::kBf16})
    {
        const std::vector<std::size_t> pool = ReferencePoolColumns(report, precision);

        if (pool.size() < 2)
        {
            continue;
        }

        const RunOutcome mainRun = OrderColumns(roundCost, pool, printed);

        if (mainRun.ordered)
        {
            continue;
        }

        // Only the options that could not be placed behind the leader: the
        // others were already ordered by the main run, and re-measuring them
        // would spend the longer protocol on pairs the run had settled.
        std::vector<std::size_t> tied;
        tied.push_back(mainRun.leader);

        for (const std::size_t column : mainRun.unplaced)
        {
            tied.push_back(column);
        }

        Refine(report, options_, work, buffers, options, precision, tied,
               report.measurements[mainRun.leader].name);
    }

    Conclude(report, roundCost);

    return report;
}

namespace {

/// The report's own column width. A line longer than this is wrapped, so a
/// printed report survives being read in a terminal, a diff or a log.
constexpr std::size_t kReportWidth = 96;

/// A comma-separated run of items, wrapped and indented under its heading.
///
/// Used for the lists this change made long — a precision class's ranked
/// members, the cells a build serves — where one line per item would bury the
/// structure the list exists to show.
std::string WrappedList(const std::vector<std::string>& items, const std::string& indent) {
    std::string text;
    std::string line = indent;

    for (const std::string& item : items)
    {
        std::string piece = (line.size() > indent.size() ? ", " : "") + item;

        if (line.size() + piece.size() > kReportWidth && line.size() > indent.size())
        {
            text += line + "\n";
            line = indent + item;
        } else
        {
            line += piece;
        }
    }

    if (line.size() > indent.size())
    {
        text += line + "\n";
    }

    return text;
}

/// A comma-separated run of items on one line, for a list short enough to read
/// inline.
std::string Joined(const std::vector<std::string>& items) {
    std::string text;

    for (const std::string& item : items)
    {
        if (!text.empty())
        {
            text += ", ";
        }

        text += item;
    }

    return text;
}

/// One cell's name and the library's reason, the reason wrapped under the name
/// so a long sentence reads as a paragraph rather than running off the page.
std::string WrappedReason(const std::string& name, const std::string& reason,
                          std::size_t nameWidth) {
    std::string text = Text("    %-*s  ", static_cast<int>(nameWidth), name.c_str());
    const std::string indent(text.size(), ' ');
    std::size_t column = text.size();
    std::size_t start = 0;

    while (start < reason.size())
    {
        std::size_t end = reason.find(' ', start);

        if (end == std::string::npos)
        {
            end = reason.size();
        }

        const std::string word = reason.substr(start, end - start);
        start = end + 1;

        if (word.empty())
        {
            continue;
        }

        if (column + 1 + word.size() > kReportWidth && column > indent.size())
        {
            text += "\n" + indent;
            column = indent.size();
        }

        if (column > indent.size())
        {
            text += ' ';
            ++column;
        }

        text += word;
        column += word.size();
    }

    return text + "\n";
}

} // namespace

std::string OptionProbeDefaultHowName(OptionProbeDefaultHow how) {
    switch (how)
    {
    case OptionProbeDefaultHow::kOrdered:
        return "ordered";
    case OptionProbeDefaultHow::kOnlyEntry:
        return "only-entry";
    case OptionProbeDefaultHow::kRefined:
        return "refined";
    case OptionProbeDefaultHow::kVote:
        return "vote";
    case OptionProbeDefaultHow::kChosenAmongEquals:
        return "chosen-among-equals";
    case OptionProbeDefaultHow::kNone:
        break;
    }

    return "none";
}

std::string FormatOptionProbe(const OptionProbeReport& report) {
    std::string text;

    // How far the calibration window's median run sat above its fastest run:
    // what the calibration found about this machine. Guarded, because a window
    // that never produced a floor has no ratio to report.
    const double medianAboveFloor =
        report.canaryFloorMs > 0.0
            ? 100.0 * (report.canaryCalibrationMedianMs / report.canaryFloorMs - 1.0)
            : 0.0;

    text += "boys option probe — this result is about this machine, this build and this process.\n";
    text += Text("  machine: %d logical processors | AVX2+FMA tier: %s\n",
                 report.logicalProcessors,
                 report.avx2 ? "present" : "absent");
    text += "  arithmetic backends this build carries, and whether a bare a*b+c written in each\n";
    text += "  is a single rounding here:\n";

    for (const backend::BackendInfo& entry : report.backends)
    {
        text += Text("    %-14s contracts=%s\n",
                     entry.name != nullptr ? entry.name : "(unnamed)",
                     entry.contracts ? "yes" : "no");
    }

    text += Text("  workload: %zu arguments | x log-uniform on [%.0e, %.0e] | each argument's own",
                 report.options.count, report.options.xLo, report.options.xHi);
    text += " highest\n";
    text += Text("            order drawn as the sum of two shell angular momenta over 0..%d "
                 "(nmax %d):\n",
                 report.options.nmax / 2, report.options.nmax);
    text += Text("            %zu order runs, the largest %zu arguments — what the grouped "
                 "options batch on\n",
                 report.orderRuns, report.largestRun);
    text += Text("  protocol: %d passes | %d interleaved rounds per pass | background window "
                 "%.3f s |\n",
                 report.options.passes, report.options.rounds,
                 report.options.backgroundWindowSeconds);
    text += Text("            calibration %.3f s | canary alarm %.1f%% (a flag, not a gate)\n",
                 report.options.calibrationSeconds, report.options.canarySpreadAlarm);
    text += "  comparison: paired. Every option is called once in every round, and the comparison\n";
    text += "            between two options is the ratio of their times inside the round both "
            "were called\n";
    text += "            in, never a ratio of figures from different rounds, so a clock drift "
            "common to\n";
    text += "            the round cancels in it. The visit order is shuffled per round from the\n";
    text += Text("            workload's seed %llu, so a position in a round is not a systematic "
                 "advantage.\n",
                 static_cast<unsigned long long>(report.options.seed));

    if (!report.options.only.empty())
    {
        text += "  measured set: named by the caller, so the fastest option below is the fastest\n";
        text += "            of these and not of the library:\n";

        for (const std::string& name : report.options.only)
        {
            text += Text("              %s\n", name.c_str());
        }
    }

    text += "  canary: a fixed-work integer spin with no floating point, so the arithmetic\n";
    text += "            question above cannot change the instrument's own cost. ";
    text += Text("Its quiet floor\n            here is the fastest of the %d runs of it in the "
                 "calibration window:\n",
                 report.canaryCalibrationRuns);
    text += Text("            %.3f ms. Those runs spread %.1f%% and their median sat %.1f%% above "
                 "the floor.\n",
                 report.canaryFloorMs, report.canaryCalibrationSpread, medianAboveFloor);
    text += "            It is a diagnostic and it gates nothing. A fixed work read by wall clock\n";
    text += "            measures the clock as much as the load, so a machine whose boost decays\n";
    text += "            widens the spin's own spread while doing nothing else; a rule that\n";
    text += "            discarded a pass on that would discard the measurement rather than the\n";
    text += "            machine. The paired% column below is the spread of the quantity the\n";
    text += "            comparison is actually made in, and it is what the ordering uses.\n";
    text += "  load readings: context, and not a bar. A reading is the percentage by which the\n";
    text += "            canary took longer than its floor, so it says how busy the machine "
            "looked,\n";
    text += "            not how well it could be measured. It is not the machine's total\n";
    text += "            utilization: no standard-library call can read another process's "
            "processor\n";
    text += "            time.\n";
    text += Text("  reference: the certified per-order fp64 lane, BoysSingle, documented at "
                 "%.3g,\n",
                 report.referenceBound);
    text += "            evaluated at the order and the argument each option was actually asked\n";
    text += "            about. A difference at or below that bound is not separable from the\n";
    text += "            certified lane by this comparison; the committed reference grid in the\n";
    text += "            test suite is what proves the bound itself.\n";

    text += "\npass  wall_s  background%  before%  after%  in-pass%  canary%  paired%  note\n";

    for (std::size_t i = 0; i < report.passes.size(); ++i)
    {
        const OptionProbePass& pass = report.passes[i];

        if (report.calibrated)
        {
            text += Text("%4zu  %6.3f  %11.1f  %7.1f  %6.1f  %8.1f  %7.2f  %7.2f  %s\n", i + 1,
                         pass.seconds, pass.backgroundLoad, pass.beforeLoad, pass.afterLoad,
                         pass.inPassLoad, pass.canarySpread, pass.pairedSpread,
                         pass.canaryWide ? "canary wide — reported, used"
                                         : "canary within alarm");
        } else
        {
            text += Text("%4zu  %6.3f  %11s  %7s  %6s  %8s  %7s  %7.2f  %s\n", i + 1, pass.seconds,
                         "-", "-", "-", "-", "-", pass.pairedSpread,
                         "load instrument never ran");
        }
    }

    if (report.passes.empty())
    {
        text += "     (no pass was run)\n";
    }

    if (report.calibrated)
    {
        text += Text("  %d of %d pass(es) ran with the canary's own runs wider than the %.1f%% "
                     "alarm, and every\n",
                     report.passesAboveAlarm, static_cast<int>(report.passes.size()),
                     report.options.canarySpreadAlarm);
        text += Text("  one of them was used. The figures below pool all %d paired rounds, and the "
                     "paired%%\n",
                     report.pairedRounds);
        text += "  column is the widest within-round band that pass's own rounds showed: the "
                "spread the\n";
        text += "  ordering is made in, in the units the comparison is made in.\n";
    } else
    {
        text += "  The load instrument never established a floor on this machine, so the load and\n";
        text += "  canary columns have nothing to be relative to and the spin was not run. The\n";
        text += Text("  comparison does not use them: the %d paired round(s) below are timed "
                     "rounds.\n",
                     report.pairedRounds);
    }

    text += Text("\noptions — every option evaluates F_0..F_n for one argument, with n that "
                 "argument's own highest order\n");

    // The name column is as wide as the longest name this run printed, so the
    // columns line up whether the option set is the old handful of shapes or the
    // whole cell space.
    std::size_t nameWidth = 19;

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        nameWidth = std::max(nameWidth, measurement.name.size());
    }

    text += Text("  %-*s %-6s %-13s %9s %9s %7s %15s %7s %9s  %11s  %10s  %s\n",
                 static_cast<int>(nameWidth), "option", "prec", "arithmetic", "ns/arg", "hi",
                 "spread", "vs ref band", "drift%", "peak ns", "max error", "bound", "verdict");

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        // The verdict is the accuracy one and it does not depend on a single
        // pass being admitted: the values a lane returns for an argument are a
        // property of the build, and a busy machine that widened every canary
        // reading must not be able to blank the accuracy column with it.
        std::string verdict = measurement.meetsBound ? "within bound"
                              : measurement.withinReferenceFloor
                                  ? "within the floor"
                                  : "ABOVE BOUND";

        if (measurement.bitIdenticalToReference)
        {
            verdict = "reference";
        }

        // An option whose own tables are certified at a figure of their own says
        // so here, in the row and not only in the prose below. The column is the
        // entry's whole-domain figure, which is a different promise from the
        // option's own fitted-interval one, and the interval is printed with it:
        // without both, a reader would take the fitted interval's figure for the
        // figure the row was judged at, or the judged figure for the size of the
        // partition's own error.
        if (measurement.ownBound > 0.0)
        {
            verdict += Text(" (whole-domain figure; the partition's own fits are certified at "
                            "%.3g, on x in [%.4g, %.4g) alone)",
                            measurement.ownBound, measurement.ownLo, measurement.ownHi);
        }

        if (measurement.measured)
        {
            text += Text("  %-*s %-6s %-13s %9.2f %9.2f %6.2fx %6.3f..%-6.3f %6.1f%% %9.2f  "
                         "%11.3e  %10.3g  %s\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         PrecisionName(measurement.precision), measurement.arithmetic.c_str(),
                         measurement.nsPerArgument, measurement.nsPerArgumentMax,
                         measurement.spread, measurement.ratioLo, measurement.ratioHi,
                         100.0 * measurement.ratioDrift, measurement.nsPerArgumentPeak,
                         measurement.maxError, measurement.bound, verdict.c_str());
        } else
        {
            text += Text("  %-*s %-6s %-13s %9s %9s %7s %15s %7s %9s  %11.3e  %10.3g  %s\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         PrecisionName(measurement.precision), measurement.arithmetic.c_str(),
                         "not measured", "-", "-", "-", "-", "-", measurement.maxError,
                         measurement.bound, verdict.c_str());
        }
    }

    text += "\n  (prec is the precision class the row is ranked in, and the only set of options it "
            "is ever\n";
    text += "   ordered against: a lane that computes in single precision is not a faster answer to "
            "the\n";
    text += "   double lane's question, it is an answer to a different one, so no column here is "
            "read\n";
    text += "   across precisions. ns/arg is the lower quartile of the option's per-round costs and "
            "hi the\n";
    text += "   upper, both from within-round ratios to the reference lane, so neither carries a "
            "drift\n";
    text += "   common to a round; spread is hi over ns/arg. 'vs ref band' is the range the "
            "option's ratio\n";
    text += "   to the reference lane fell in over the run's rounds, at its lower and upper "
            "quartiles, and\n";
    text += "   drift% is how far that ratio moved between the run's first and second half — a "
            "pair whose\n";
    text += "   ratio drifts does not carry a decaying clock alike, and the confidence line below "
            "says\n";
    text += "   whether any pair drifted further than this run can order. 'peak ns' is the single "
            "fastest\n";
    text += "   round the option was seen in; it bounds what the option can do at a peak clock and "
            "is\n";
    text += "   never the option's cost, because under a decaying clock the fastest round is the\n";
    text += "   earliest one, which a caller with a long workload does not meet. The verdict column "
            "is the\n";
    text += "   accuracy one, against the bound in the row's own bound column: 'within bound' met "
            "it,\n";
    text += Text("   'within the floor' is above its own bound but at or below the certified lane's "
                 "%.3g,\n",
                 report.referenceBound);
    text += "   which this comparison cannot separate, and 'ABOVE BOUND' means the option did not "
            "deliver\n";
    text += "   what its own lane documents.\n";
    text += "   Every bound in the column is read from the library, and every one of them applies "
            "over\n";
    text += "   every argument the row was measured on: the certified lane's own figure for the "
            "certified\n";
    text += "   rows, the accuracy rung's for a rung, and the figure the library documents for the "
            "entry\n";
    text += "   the row reaches for the narrower lanes and for the partitions — naming a partition "
            "changes\n";
    text += "   which fitted tables that entry reads, not the entry's own figure. A partition also "
            "publishes\n";
    text += "   a figure of its own, for its stored fits over the interval those fits cover and for "
            "nothing\n";
    text += "   outside it, and that figure is not the row's judgement: the measured column is a "
            "difference\n";
    text += "   from the certified lane rather than an error against the true function, and past "
            "the fitted\n";
    text += "   interval the entry is running the certified lane's own arithmetic. A row reading a "
            "partition\n";
    text += "   therefore names that figure and the interval it holds on beside its verdict, and "
            "the column\n";
    text += "   stays the entry's whole-domain figure. The measured column itself is the worst "
            "difference\n";
    text += "   over the arguments this run sampled, so it moves with --xrange and --count: a "
            "figure from\n";
    text += "   one range is not comparable with one from another, while the verdict is, because "
            "the figure\n";
    text += "   it is judged against does not move. The verdict does not wait on a pass:\n";
    text += "   the values a lane returns are a property of the build, so a run on a machine whose\n";
    text += "   clock wandered still reports which rows delivered what they document, and its cost\n";
    text += "   columns read 'not measured' only where no figure could be formed at all.)\n";

    if (!report.refused.empty())
    {
        text += "\nnames asked for that are cells of this library's option space, which the "
                "library refuses\nwhere they are named — unbuilt work, counted, and not measured "
                "here:\n";

        for (const std::string& name : report.refused)
        {
            text += Text("  %s\n", name.c_str());
        }
    }

    if (!report.notAnOption.empty())
    {
        text += "\nnames asked for that are no option of this library, so nothing was measured "
                "for them:\n";

        for (const std::string& name : report.notAnOption)
        {
            text += Text("  %s\n", name.c_str());
        }
    }

    if (!report.unoffered.empty())
    {
        text += "\noptions this build does not offer, because its backend table carries no "
                "arithmetic to run them in:\n";

        for (const std::string& name : report.unoffered)
        {
            text += Text("  %s\n", name.c_str());
        }
    }

    text += "\n\nthe accuracy classes — one precision at one rung, the only sets this probe orders "
            "inside\n";
    text += "  A class is one precision and one rung of the accuracy axis: the multiplier an "
            "option was\n";
    text += "  built at, m = 1, 64, 256 and so on. Membership is decided by that rung, never by "
            "comparing\n";
    text += "  one lane's documented figure against another's — a figure belongs to one lane, and "
            "reading\n";
    text += "  it across lanes is how a class ends up empty by construction. Every row of a class "
            "was built\n";
    text += "  at the same multiplier, so nothing inside it traded accuracy for speed and the "
            "entry it names\n";
    text += "  is the fastest option at that rung. Nothing here is ordered across classes: another "
            "precision\n";
    text += "  or another rung is an answer to a different question, not a slower answer to this "
            "one.\n";

    for (const OptionProbeClass& entry : report.classes)
    {
        if (entry.leader.empty())
        {
            text += Text("  %-10s %s\n", entry.name.c_str(), entry.note.c_str());
            continue;
        }

        // The members of this class alone, fastest first, each with the figure its
        // own row documents — the row that says which entries the entry named here
        // was measured against.
        std::vector<const OptionProbeMeasurement*> members;

        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.tier == entry.tier)
            {
                members.push_back(&measurement);
            }
        }

        std::sort(members.begin(), members.end(),
                  [](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
                      return a->nsPerArgument < b->nsPerArgument;
                  });

        text += Text("  %-10s %zu measured | fastest %s at %.2f ns/argument, documented at %.3g | "
                     "%s\n",
                     entry.name.c_str(), entry.ranked.size(), entry.leader.c_str(),
                     entry.leaderNsPerArgument, entry.bound,
                     entry.ordered ? "ordered" : "not ordered");
        text += Text("             %zu could not be placed behind it | the class's entry is reached "
                     "by %s\n",
                     entry.unplaced.size(), HowText(entry.how));

        std::vector<std::string> rankedLines;

        for (const OptionProbeMeasurement* member : members)
        {
            rankedLines.push_back(Text("%s %.2f ns (%.3g)", member->name.c_str(),
                                       member->nsPerArgument, member->bound));
        }

        text += WrappedList(rankedLines, "             ");
        text += Text("             %s\n", entry.note.c_str());

        if (entry.precision == OptionPrecision::kFp64 && entry.tier == AccuracyTier::kReference)
        {
            text += "             (this is the class the default below is taken from: the library's "
                    "own\n";
            text += "             default precision at its own full-accuracy multiplier)\n";
        }
    }

    // The coverage: the library's own option space, every cell of it, so a
    // combination this build does not carry is counted and given the library's
    // reason instead of being absent from the report.
    std::size_t served = 0;
    std::size_t refused = 0;

    for (const OptionProbeCell& cell : report.cells)
    {
        (cell.served ? served : refused) += 1;
    }

    text += Text("\n\nthe option space — the axes this library reports, every cell of their "
                 "product, and what\n");
    text += Text("  this build does with each: %zu served + %zu refused = %zu cells\n", served,
                 refused, report.cells.size());

    {
        // Every axis and every member is read from the library's own reporting
        // API, so this line cannot name an axis or a member this build lacks.
        const std::vector<std::string> routes = DistinctRouteNames(BoysFitRoutes());
        std::vector<std::string> schemes;
        std::vector<std::string> axes;
        std::vector<std::string> partitions;

        for (const EvalSchemeInfo& scheme : BoysEvalSchemes())
        {
            schemes.push_back(scheme.name != nullptr ? scheme.name : "(unnamed)");
        }

        for (const PackAxisInfo& axis : BoysPackAxes())
        {
            axes.push_back(axis.name != nullptr ? axis.name : "(unnamed)");
        }

        for (const FitGranularityInfo& partition : report.granularities)
        {
            partitions.push_back(partition.name);
        }

        text += Text("  axes: %zu route(s) (%s) | %zu scheme(s) (%s) | %zu partition(s) (%s) |\n",
                     routes.size(), Joined(routes).c_str(), schemes.size(),
                     Joined(schemes).c_str(), partitions.size(), Joined(partitions).c_str());
        text += Text("        %zu packing axis/axes (%s) | %zu accuracy rung(s) this build serves\n",
                     axes.size(), Joined(axes).c_str(), ServedTiers().size());
    }

    text += "  a refused cell is refused by the library where it is named, not by this probe: it "
            "is\n";
    text += "  unbuilt work, counted here, and no cell of the product is absent from this book. A "
            "served\n";
    text += "  cell has an option row above unless the run was narrowed by the names the caller "
            "gave.\n";
    text += "  partitions this build ships, read from the library:\n";

    for (const FitGranularityInfo& partition : report.granularities)
    {
        text += Text("    %-8s rungs=%d axes=%s%d | region A %d piece(s) deg %d (%d stored) | "
                     "region B %d piece(s) deg %d (%d stored)\n",
                     partition.name, partition.rungs,
                     FitGranularityHasAxis(partition, PackAxis::kArguments) ? "arguments+" : "",
                     FitGranularityHasAxis(partition, PackAxis::kOrders) ? 1 : 0,
                     partition.regionAPieces, partition.regionADeg, partition.regionAStored,
                     partition.regionBPieces, partition.regionBDeg, partition.regionBStored);
        text += Text("             route(s) %s | stored fits: delivered %.3g | bound %.3g, on "
                     "x in [%.4g, %.4g) alone\n",
                     PartitionRouteNames(partition).c_str(), partition.delivered, partition.bound,
                     partition.lo, partition.hi);
    }

    std::vector<std::string> servedCells;

    for (const OptionProbeCell& cell : report.cells)
    {
        if (cell.served)
        {
            servedCells.push_back(cell.name);
        }
    }

    text += Text("  served cells (%zu):\n", servedCells.size());
    text += WrappedList(servedCells, "    ");

    if (refused > 0)
    {
        std::size_t reasonWidth = 0;

        for (const OptionProbeCell& cell : report.cells)
        {
            if (!cell.served)
            {
                reasonWidth = std::max(reasonWidth, cell.name.size());
            }
        }

        text += Text("  refused cells (%zu), each with the library's reason:\n", refused);

        for (const OptionProbeCell& cell : report.cells)
        {
            if (!cell.served)
            {
                text += WrappedReason(cell.name, cell.reason, reasonWidth);
            }
        }
    }
    if (!report.notCarried.empty())
    {
        text += "\noptions this build does not carry, because its fp16 seam is closed (BoysFp16 "
                "= 0):\n";

        for (const std::string& name : report.notCarried)
        {
            text += Text("  %s\n", name.c_str());
        }
    }

    // The class the default is taken from, so the resolution can be reported
    // against the class it is measured in rather than against one row of it.
    const OptionProbeClass* doublesClass = nullptr;

    for (const OptionProbeClass& entry : report.classes)
    {
        if (entry.precision == OptionPrecision::kFp64 && entry.tier == AccuracyTier::kReference)
        {
            doublesClass = &entry;
        }
    }

    text += "\nrecommendation\n";
    text += Text("  the %s class — the certified double lane's precision at the library's own\n",
                 doublesClass != nullptr ? doublesClass->name.c_str() : "fp64 m=1");
    text += "  full-accuracy multiplier, every row of it built at that rung, so nothing in it "
            "traded\n";
    text += "  accuracy for speed. This is the pool the default is taken from, and these are its "
            "rows:\n";

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.precision != OptionPrecision::kFp64 ||
            measurement.tier != AccuracyTier::kReference)
        {
            continue;
        }

        if (measurement.measured)
        {
            text += Text("    %-*s %8.2f ns/argument  band %.3f..%.3f  spread %.2fx  %d paired "
                         "round(s)  peak %.2f\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         measurement.nsPerArgument, measurement.ratioLo, measurement.ratioHi,
                         measurement.spread, measurement.rounds, measurement.nsPerArgumentPeak);
        } else
        {
            text += Text("    %-*s cost not measured on this run; delivered %.3g against %.3g\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(), measurement.maxError,
                         measurement.bound);
        }
    }

    if (!report.fastestAtReferenceAccuracy.empty())
    {
        text += Text("  fastest of that class by the run's own rounds: %s\n",
                     report.fastestAtReferenceAccuracy.c_str());
    }

    // The default's margin over the nearest rival inside the class it was taken
    // from. Which row the default is and how far ahead of the next one it is are
    // two questions, and a reader who is told only the first cannot see whether
    // the class had a rival at all. The rival is the class's own second-fastest
    // row by this run's statistic: a figure from another rung or another
    // precision belongs to a different class and is not a rival to a row of this
    // one.
    if (doublesClass != nullptr && !report.recommended.empty())
    {
        const OptionProbeMeasurement* chosen = nullptr;

        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.name == report.recommended &&
                measurement.nsPerArgument > 0.0)
            {
                chosen = &measurement;
            }
        }

        const OptionProbeMeasurement* nearest = nullptr;

        if (chosen != nullptr)
        {
            for (const OptionProbeMeasurement& measurement : report.measurements)
            {
                if (!measurement.measured || measurement.precision != OptionPrecision::kFp64 ||
                    measurement.tier != AccuracyTier::kReference ||
                    measurement.nsPerArgument <= 0.0 || measurement.name == chosen->name)
                {
                    continue;
                }

                if (nearest == nullptr || measurement.nsPerArgument < nearest->nsPerArgument)
                {
                    nearest = &measurement;
                }
            }
        }

        if (nearest != nullptr)
        {
            const double margin = nearest->nsPerArgument / chosen->nsPerArgument;

            text += Text("    its nearest rival in that class: %s at %.2f ns/argument, %.1f%% of the "
                         "default's figure\n",
                         nearest->name.c_str(), nearest->nsPerArgument, 100.0 * margin);

            if (report.defaultHow == OptionProbeDefaultHow::kOrdered)
            {
                text += Text("      so the default is %.1f%% ahead of the fastest other row of its "
                             "own class\n",
                             100.0 * (margin - 1.0));
            } else
            {
                text += Text("      the two are figures of this run and not an ordering of them: the "
                             "default is %.1f%%\n      ahead of that row here, and the class's own "
                             "rounds could not place one behind the other,\n      which is why the "
                             "default was reached the way the line above says\n",
                             100.0 * (margin - 1.0));
            }
        } else if (chosen != nullptr)
        {
            text += "    its nearest rival in that class: none — this class holds one measured "
                    "row and no\n    other, so the entry it names was forced by there being no "
                    "alternative and not chosen\n";
        }
    }

    if (!report.fastestOverall.empty())
    {
        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.name == report.fastestOverall)
            {
                const bool sameClass =
                    measurement.precision == OptionPrecision::kFp64 &&
                    measurement.tier == AccuracyTier::kReference;

                text += Text("  fastest measured overall: %s at %.2f ns/argument, documented at "
                             "%.3g%s\n",
                             measurement.name.c_str(), measurement.nsPerArgument, measurement.bound,
                             sameClass ? ""
                                       : " — another precision or another rung: faster, not "
                                         "faster at the same accuracy");
            }
        }
    }

    // What a relaxed rung buys, per precision. The default is the full-accuracy
    // class's own winner, so a cheaper row of a looser rung is a trade a caller
    // may make and never a candidate for the default; it is printed here, under
    // its own heading and after the default, so that a reader sees what the trade
    // costs in accuracy and buys in time without either figure being confusable
    // with the default. Every figure is the fastest row of its own precision at
    // its own rung, read from the same reference lane the classes above are read
    // from, so a ratio between two of them is between rows measured against one
    // anchor.
    text += "\n  what a relaxed rung buys, kept apart from the default\n";
    text += "    The default above is the m = 1 class's own winner and nothing else. A row of a\n";
    text += "    looser rung is a looser bound for a cheaper call: that is the caller's trade to\n";
    text += "    make, it is not an answer at the default's accuracy, and no row of this block is\n";
    text += "    offered as a default. Each figure is the fastest row of its precision at that\n";
    text += "    rung on this run; each ratio is that figure against the fastest row of the same\n";
    text += "    precision's m = 1 class:\n";

    for (const OptionPrecision precision : {OptionPrecision::kFp64, OptionPrecision::kFp32,
                                            OptionPrecision::kFp16, OptionPrecision::kBf16})
    {
        std::vector<const OptionProbeClass*> measured;
        const OptionProbeClass* reference = nullptr;

        for (const OptionProbeClass& entry : report.classes)
        {
            if (entry.precision != precision)
            {
                continue;
            }

            if (entry.tier == AccuracyTier::kReference)
            {
                reference = &entry;
            }

            if (!entry.leader.empty() && entry.leaderNsPerArgument > 0.0)
            {
                measured.push_back(&entry);
            }
        }

        if (measured.empty())
        {
            continue;
        }

        const double base = reference != nullptr ? reference->leaderNsPerArgument : 0.0;

        for (const OptionProbeClass* entry : measured)
        {
            const bool atReference = entry->tier == AccuracyTier::kReference;

            const bool defaultRung = atReference && precision == OptionPrecision::kFp64;
            const char* tag = defaultRung ? ", the default's rung"
                              : (atReference && measured.size() == 1)
                                  ? ", the only rung of this precision that produced a figure on "
                                    "this run"
                                  : "";

            if (entry == measured.front())
            {
                text += Text("    %-5s m=%g: %.2f ns/argument, documents %.3g%s\n",
                             PrecisionName(precision), AccuracyMultiplier(entry->tier),
                             entry->leaderNsPerArgument, entry->bound, tag);
            } else
            {
                text += Text("          m=%g: %.2f ns/argument, documents %.3g",
                             AccuracyMultiplier(entry->tier), entry->leaderNsPerArgument,
                             entry->bound);
                text += "\n";
            }

            if (!atReference)
            {
                if (base > 0.0)
                {
                    text += Text("          (%.3fx the m = 1 figure of this precision — %.2fx "
                                 "cheaper, at a bound this\n           rung documents as %.3g "
                                 "against the m = 1 row's %.3g)\n",
                                 entry->leaderNsPerArgument / base, base / entry->leaderNsPerArgument,
                                 entry->bound, reference->bound);
                } else
                {
                    text += "          (the m = 1 class of this precision produced no figure on "
                            "this run, so there is\n           nothing here to read this rung "
                            "against)\n";
                }
            }
        }
    }

    // The refinement stage and its vote, printed in full: a default reached this
    // way rests on the runs, so the runs are what the reader is given.
    if (!report.refinements.empty())
    {
        text += "\n  the refinement stage — the options the class left tied, re-run alone and "
                "voted on\n";

        for (const OptionProbeRefinement& stage : report.refinements)
        {
            text += Text("    %s: %d run(s) of %d passes by %d rounds each, over %zu tied "
                         "option(s)\n",
                         PrecisionName(stage.precision), stage.runs, stage.passes, stage.rounds,
                         stage.pool.size());
            text += Text("      options re-measured: %s\n", Joined(stage.pool).c_str());
            text += Text("      leader of each run, in the order the runs were taken: %s\n",
                         Joined(stage.runLeaders).c_str());

            for (const std::string& line : stage.tally)
            {
                text += Text("      %s\n", line.c_str());
            }

            text += Text("      vote: %s — %s\n", stage.winner.c_str(), stage.note.c_str());
        }
    }

    if (doublesClass == nullptr || doublesClass->leader.empty() ||
        !doublesClass->leaderNsPerArgument)
    {
        text += "\n  resolution: not measurable on this run — the certified double lane's "
                "precision produced\n";
        text += "              no measured option, so there was nothing to order\n";
    } else if (report.pairedRounds < static_cast<int>(kMinimumPairedRounds))
    {
        text += Text("\n  resolution: not measurable on this run — %d paired round(s) is fewer than "
                     "the %zu a\n",
                     report.pairedRounds, kMinimumPairedRounds);
        text += "              quartile band needs, so no band was formed and nothing was "
                "ordered\n";
    } else
    {
        text += "\n  resolution: the widest within-round band anything in the certified double "
                "lane's\n";
        text += Text("              precision showed over the %d paired rounds was %.2f%%. It is "
                     "measured in the\n",
                     report.pairedRounds, 100.0 * report.resolution);
        text += "              paired ratios, which is the quantity the ordering is made of, and "
                "not in\n";
        text += "              absolute times: every pair's band is formed inside a round and the\n";
        text += "              ordering is made pair by pair from that pair's own band, so this "
                "number states\n";
        text += "              how coarse the class's own measurement got and bars nothing. The\n";
        text += Text("              canary's widest pass spread beside it was %.2f%% — context, and "
                     "not the\n",
                     report.canarySpread);
        text += "              bar: a clock that decays widens a fixed work read by wall clock and\n";
        text += "              cancels in a ratio, which is why the two columns differ and why the\n";
        text += "              second is the one the ordering uses.\n";
    }

    text += Text("  verdict: %s\n", report.verdict == OptionProbeVerdict::kRecommend
                                        ? "RECOMMEND"
                                        : "CANNOT DETERMINE");

    if (!report.recommended.empty())
    {
        text += Text("  default: %s\n", report.recommended.c_str());
        text += Text("    reached by: %s. %s\n", HowText(report.defaultHow),
                     report.defaultHow == OptionProbeDefaultHow::kOrdered
                         ? "This is a measured ordering, and the default is the fastest option of "
                           "the class by it"
                         : (report.defaultHow == OptionProbeDefaultHow::kOnlyEntry
                                ? "This is not a ranking: nothing in the class was measured "
                                  "against it, and it is named by there being no alternative"
                                : "This is not a measured ordering: the options the class left tied "
                                  "were re-run alone and the run above says which of them this is, "
                                  "and on what vote"));
    }

    if (!report.reason.empty())
    {
        text += Text("  reason: %s\n", report.reason.c_str());
    }

    if (!report.inseparable.empty())
    {
        text += "\n  not placed behind the class's fastest by the paired rounds:\n";

        for (const std::string& entry : report.inseparable)
        {
            text += Text("    * %s\n", entry.c_str());
        }
    }

    if (!report.confidence.empty())
    {
        text += Text("  confidence: %s\n", report.confidence.c_str());
    }

    text += "\n  this ranking is this machine's: a different host, a different compiler or a "
            "different set of\n";
    text += "  build flags can rank these options differently, which is why the probe is run where "
            "the numbers\n";
    text += "  are used rather than read from documentation. A figure above is the lower quartile "
            "of its\n";
    text += "  paired rounds — not the minimum, which is the earliest and highest-clock "
            "observation, and\n";
    text += "  not the mean, which one disturbed round moves — and its spread is printed beside "
            "it so the\n";
    text += "  reader can see what the figure rests on. The peak column bounds what each option can "
            "do at a\n";
    text += "  peak clock and is never the option's cost. The resolution above is this run's: a "
            "steadier\n";
    text += "  machine would order pairs this one could not.\n";

    // The refusals, grouped by the axis that refuses them, counted from the
    // coverage book rather than written here: the three ways a cell of the
    // product can be unserved are read off the partition rows the library
    // publishes, so a build that grows one of them says so by itself.
    const auto partition_of = [&report](FitGranularity granularity) -> const FitGranularityInfo* {
        for (const FitGranularityInfo& partition : report.granularities)
        {
            if (partition.granularity == granularity)
            {
                return &partition;
            }
        }

        return nullptr;
    };

    std::size_t refusedRoute = 0;
    std::size_t refusedAxis = 0;
    std::size_t refusedRung = 0;
    std::size_t refusedFiner = 0;

    for (const OptionProbeCell& cell : report.cells)
    {
        const FitGranularityInfo* partition = cell.served ? nullptr : partition_of(cell.granularity);

        if (partition == nullptr)
        {
            continue;
        }

        if (!FitGranularityHasRoute(*partition, cell.route))
        {
            ++refusedRoute;
        } else if (!FitGranularityHasAxis(*partition, cell.pack))
        {
            ++refusedAxis;
        } else if (partition->rungs == 1 && cell.tier != AccuracyTier::kReference)
        {
            ++refusedRung;
        } else
        {
            // A limit finer than the partition row's own fields can state. The row
            // says which routes and axes the partition carries and how many rungs it
            // serves; it cannot say that one lane of a route it carries is
            // instantiated over another partition's fits, or that a rung of that
            // route is derived from them. Those cells carry the library's own
            // reason in the coverage above, and this count keeps the summary's
            // arithmetic equal to the refusals it describes.
            ++refusedFiner;
        }
    }

    text += "\nnot measured by this probe, by design — listed so that nothing above is read as a "
            "complete\n";
    text += "  account of the library:\n";
    text += "  * the native half lane (region C only, orders 0..8, results scaled by a power of "
            "two — a\n";
    text += "    different question from this workload's);\n";
    text += "  * the region-A matrix-product transform (region A alone, so measuring it means "
            "composing the\n";
    text += "    rest of an evaluation, and the composition would be what got measured);\n";
    text += "  * the CUDA lanes (a separate optional build that needs a device);\n";
    text += "  * a partition's stored fits read directly: the figure a partition row prints beside "
            "its\n";
    text += "    verdict is the library's published figure for that partition's fits, and every row "
            "above\n";
    text += "    is an entry read end to end, so nothing here measures a fit's own error against "
            "the true\n";
    text += "    function over the interval it is cut for — the library's accuracy gate measures "
            "that book;\n";
    if (refused > 0)
    {
        text += Text("  * the %zu cells of the option space this build refuses, of %zu, counted by "
                     "what the\n    partition rows state of themselves: %zu on the rational route "
                     "(a table to generate),\n    %zu on the across-orders packing axis (a kernel "
                     "to write), %zu at the relaxed rungs\n",
                     refused, report.cells.size(), refusedRoute, refusedAxis, refusedRung);
        text += Text("    (a degree table to derive) and %zu on a limit finer than those fields "
                     "state — an\n    across-orders packed lane instantiated over the shipped fits "
                     "rather than the\n    partition's own (a body to write), and a relaxed rung of "
                     "the rational route derived\n    from those same fits (a table to derive). "
                     "Each is named with the library's own reason in\n",
                     refusedFiner);
        text += "    coverage above: they are unbuilt work, counted rather than absent, and no "
                "cell of the\n";
        text += "    library's product is missing from this report;\n";
    }
    text += "  * the call shapes other than this workload's all-orders-per-argument one, which the "
            "five axes\n";
    text += "    are not crossed with: the all-N grouping, the sorted-argument overload, the "
            "single-order\n";
    text += Text("    entry and the single-precision lane's own route table (%zu row(s) this build "
                 "carries) are\n",
                 BoysFitRoutesF32().size());
    text += "    measured at their default policy alone above. Crossing them with the axes is "
            "outstanding\n";
    text += "    work, not an impossibility.\n";

    return text;
}

} // namespace boys
