#include "boys/boys_probe.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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
constexpr double kHalfIoLaneBound = 1e-7 + 0x1p-11;
constexpr double kBf16IoLaneBound = 1e-7 + 0x1p-9;

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
/// The reasons are read off the partition rows rather than restated: a route the
/// row's tables do not hold, a packing axis the row has no kernel for, and a rung
/// the row's degrees are not certified at are the three ways a cell of this
/// product can be unserved, and each is a fact of the row the library publishes.
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
                        cell.served = true;

                        if (!FitGranularityHasRoute(partition, route.route))
                        {
                            cell.served = false;
                            cell.reason = Text(
                                "the %s partition's tables carry the %s route, and the %s route "
                                "is not one of them — a table to generate",
                                partition.name, PartitionRouteNames(partition).c_str(), route.name);
                        } else if (!FitGranularityHasAxis(partition, axis.axis))
                        {
                            cell.served = false;
                            cell.reason = Text(
                                "the %s partition has no kernel on the %s axis: the across-orders "
                                "lane steps one order's coefficients to the next at a fixed "
                                "stride, which needs the pieces to share their shape from order "
                                "to order, and the %s pieces are cut per order — a kernel to "
                                "write",
                                partition.name, axis.name, partition.name);
                        } else if (partition.rungs == 1 && tier != AccuracyTier::kReference)
                        {
                            cell.served = false;
                            cell.reason = Text(
                                "the %s partition is certified at the reference rung alone (%g of "
                                "the %zu rungs this build serves): its degrees are the shipped "
                                "rung's, so a relaxed rung has no %s table to truncate — a degree "
                                "table to derive",
                                partition.name, AccuracyMultiplier(tier), tiers.size(),
                                partition.name);
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
/// seam that declares them.
std::vector<Option> EnumerateOptions(std::span<const backend::BackendInfo> table,
                                     std::span<const FitGranularityInfo> partitions,
                                     const std::vector<OptionProbeCell>& cells,
                                     std::vector<std::string>& unoffered) {
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

/// The across-orders packed lane at one rung, for one pair of the other axes.
///
/// The lane's entry is a template over the scheme, the rung and the route, so
/// the probe names them here and the library instantiates them: the same
/// instantiation a consumer compiles when they name this axis, and the one the
/// library's own packed unit already instantiates for every rung.
template <FitRoute kRoute, EvalScheme kScheme>
void PackedOrdersRung(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    constexpr PackAxis kPack = PackAxis::kOrders;
    constexpr BoysBudget kBudget = BoysBudget::kFloat;

    switch (tier)
    {
    case AccuracyTier::kRelaxed64:
        BoysAllOrders<64.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed256:
        BoysAllOrders<256.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed1024:
        BoysAllOrders<1024.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed4096:
        BoysAllOrders<4096.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed16384:
        BoysAllOrders<16384.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed65536:
        BoysAllOrders<65536.0, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(nmax, x, out);
        return;

    default:
        BoysAllOrders<kBoysFullAccuracyMultiplier, EvalPolicy<kRoute, kScheme, kBudget, kPack>>(
            nmax, x, out);
        return;
    }
}

/// The across-orders packed lane for any pair of axes the library carries.
void PackedOrdersCell(FitRoute route, EvalScheme scheme, AccuracyTier tier, int nmax, double x,
                      double* out) noexcept {
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            PackedOrdersRung<FitRoute::kRationalMinimax, EvalScheme::kHorner>(tier, nmax, x, out);
        } else
        {
            PackedOrdersRung<FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw>(
                tier, nmax, x, out);
        }

        return;
    }

    if (scheme == EvalScheme::kHorner)
    {
        PackedOrdersRung<FitRoute::kChebyshev, EvalScheme::kHorner>(tier, nmax, x, out);
    } else
    {
        PackedOrdersRung<FitRoute::kChebyshev, EvalScheme::kSplitClenshaw>(tier, nmax, x, out);
    }
}

/// The narrow partition at one rung, which only exists as a policy: the library
/// reports the partition, and the probe reaches it the way a consumer does, by
/// naming it in the policy's axes. The partition's tables are cut at every rung
/// the tier enumeration declares, so the rung is part of the instantiation.
template <EvalScheme kScheme>
void NarrowRung(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    constexpr FitRoute kRoute = FitRoute::kChebyshev;
    constexpr PackAxis kPack = PackAxis::kArguments;
    constexpr BoysBudget kBudget = BoysBudget::kFloat;

    switch (tier)
    {
    case AccuracyTier::kRelaxed64:
        BoysAllOrders<64.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                       FitGranularity::kNarrow>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed256:
        BoysAllOrders<256.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                        FitGranularity::kNarrow>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed1024:
        BoysAllOrders<1024.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                         FitGranularity::kNarrow>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed4096:
        BoysAllOrders<4096.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                         FitGranularity::kNarrow>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed16384:
        BoysAllOrders<16384.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                          FitGranularity::kNarrow>>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed65536:
        BoysAllOrders<65536.0, EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                          FitGranularity::kNarrow>>(nmax, x, out);
        return;

    default:
        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                 FitGranularity::kNarrow>>(nmax, x, out);
        return;
    }
}

/// The narrow partition for either scheme at one rung.
void NarrowCell(EvalScheme scheme, AccuracyTier tier, int nmax, double x,
                double* out) noexcept {
    if (scheme == EvalScheme::kHorner)
    {
        NarrowRung<EvalScheme::kHorner>(tier, nmax, x, out);
    } else
    {
        NarrowRung<EvalScheme::kSplitClenshaw>(tier, nmax, x, out);
    }
}

/// One cell of the option space, evaluated as the entry its axes select.
///
/// A cell that the run-time tier entries carry is answered by them — the same
/// dispatch a consumer reaches — and the two axis members that exist only as
/// policies are compiled by the probe the way a consumer compiles them. The
/// choice is per call rather than per option, so a cell's cost includes its own
/// selection, as the run-time tier options' cost already does.
void CellValues(const Option& option, int nmax, double x, double* out) noexcept {
    if (option.granularity == FitGranularity::kNarrow)
    {
        NarrowCell(option.scheme, option.tier, nmax, x, out);
        return;
    }

    if (option.pack == PackAxis::kOrders)
    {
        PackedOrdersCell(option.route, option.scheme, option.tier, nmax, x, out);
        return;
    }

    BoysAllOrdersAtTier(option.tier, option.route, option.scheme, nmax, x, out);
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

#if BoysFp16
    case OptionKind::kBatchF16:
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

        break;
    }

    case OptionKind::kBatchBf16:
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

        break;
    }
#endif // BoysFp16

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

/// The option to name when the measurement could not order the field, and the
/// static reading of the library's tables it was chosen from.
struct StaticFallback {
    /// The option's name, empty when the run's set holds nothing the rule can
    /// rank.
    std::string name;

    /// Why that option, in the library's own numbers, and what the rule does not
    /// compare.
    std::string basis;
};

/// Chooses the fallback by counting what the library's tables say, never by
/// timing anything.
///
/// **This is a heuristic and it is reported as one.** The rule, whole: over the
/// options this run carries at the certified bound, restricted to the certified
/// fit route and evaluation scheme - the family the certified lane itself is
/// defined in, because a count of multiply-adds does not rank a Chebyshev
/// polynomial against a rational one, whose evaluation is a numerator and a
/// denominator and a division - the option whose partition evaluates the fewer
/// operations is named: the degree a region-A piece is evaluated at plus the
/// degree region B's seed is, both read from the partition's own row in the
/// library's table. A tie is broken by the coefficients the partition stores,
/// which is the table it brings into cache, and then by the library's own
/// default axis values, so a tie-break is a documented choice rather than an
/// accident of enumeration order. The set is narrowed to the certified double
/// lane's own precision where that set holds something rankable.
///
/// What it cannot do is as much a part of the rule as what it does: it does not
/// compare the routes or the schemes it excluded, it says nothing about any
/// machine's arithmetic, and a partition's own fits cover the interval its row
/// states and not the whole argument range.
///
/// \param report a report whose measurements and partition rows are filled in
///
/// \returns the fallback, empty when the run's option set holds nothing to rank
StaticFallback StaticFallbackFor(const OptionProbeReport& report) {
    StaticFallback fallback;

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

    const auto rankable = [&report](const OptionProbeMeasurement& measurement,
                                    bool doublesOnly) {
        if (measurement.bound > report.referenceBound ||
            measurement.route != kDefaultFitRoute || measurement.scheme != kDefaultEvalScheme)
        {
            return false;
        }

        return !doublesOnly || measurement.precision == OptionPrecision::kFp64;
    };

    const OptionProbeMeasurement* best = nullptr;
    const OptionProbeMeasurement* second = nullptr;
    const FitGranularityInfo* bestPartition = nullptr;
    bool doublesOnly = true;

    // The certified lane's own precision first; a run narrowed to another
    // precision still gets a fallback, and the basis says which set was ranked.
    for (const bool narrowed : {true, false})
    {
        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (!rankable(measurement, narrowed))
            {
                continue;
            }

            const FitGranularityInfo* partition = partition_of(measurement.granularity);

            if (partition == nullptr)
            {
                continue;
            }

            const int degree = partition->regionADeg + partition->regionBDeg;
            const int stored = partition->regionAStored + partition->regionBStored;
            const int axes = (measurement.granularity == kDefaultFitGranularity ? 0 : 1) +
                             (measurement.pack == PackAxis::kArguments ? 0 : 1);

            if (best == nullptr)
            {
                best = &measurement;
                bestPartition = partition;
                doublesOnly = narrowed;
                continue;
            }

            const int bestDegree = bestPartition->regionADeg + bestPartition->regionBDeg;
            const int bestStored = bestPartition->regionAStored + bestPartition->regionBStored;
            const int bestAxes = (best->granularity == kDefaultFitGranularity ? 0 : 1) +
                                 (best->pack == PackAxis::kArguments ? 0 : 1);

            const bool better = degree < bestDegree ||
                                (degree == bestDegree && stored < bestStored) ||
                                (degree == bestDegree && stored == bestStored && axes < bestAxes);

            if (better)
            {
                second = best;
                best = &measurement;
                bestPartition = partition;
            } else if (second == nullptr)
            {
                second = &measurement;
            }
        }

        if (best != nullptr)
        {
            break;
        }
    }

    if (best == nullptr || bestPartition == nullptr)
    {
        return fallback;
    }

    const std::string runnerUp =
        second != nullptr && partition_of(second->granularity) != nullptr
            ? Text(", the next being '%s', whose partition evaluates %d dependent multiply-adds",
                   second->name.c_str(), partition_of(second->granularity)->regionADeg +
                                              partition_of(second->granularity)->regionBDeg)
            : std::string();

    fallback.name = best->name;
    fallback.basis = Text(
        "counted, not timed: '%s' reads a partition whose region-A piece is evaluated at degree %d "
        "and whose region-B seed is at degree %d, %d dependent multiply-adds, storing %d "
        "coefficients%s. The rule takes the lowest degree first, the fewest stored coefficients "
        "second and the library's own default axes last, over this build's options at the "
        "certified bound on the certified fit route and evaluation scheme%s. It does not compare "
        "the routes or the schemes it excluded, it says nothing about this machine's arithmetic, "
        "and a partition's own fits cover the interval its row states and not the whole range.",
        best->name.c_str(), bestPartition->regionADeg, bestPartition->regionBDeg,
        bestPartition->regionADeg + bestPartition->regionBDeg,
        bestPartition->regionAStored + bestPartition->regionBStored, runnerUp.c_str(),
        doublesOnly ? ""
                    : " (no option of the certified double lane's precision was rankable, so this "
                      "is the widest precision set the run carried)");

    return fallback;
}

/// Fills in the recommendation, or the refusal, from the figures already in the
/// report.
///
/// The rule, in one place. Inside one precision class the run's own statistic -
/// the lower quartile of each option's ratio to the reference lane over the
/// paired rounds - names a leader, and the leader stands only when every other
/// member of its class was behind it in the middle half of those rounds: the
/// comparison is the pair's own within-round ratio and the pair's lower quartile
/// has to clear one. A rival whose band straddles one cannot be placed, and a
/// rival whose band lies below one was ahead of the leader in that half, which
/// means the statistic and the paired rounds disagree about the pair; both end
/// in the refusal, with the band and the round counts printed beside each. The
/// same happens when the run is too short for a quartile band to exist and when
/// nothing of the class was measured.
///
/// What the rule is not is a threshold on absolute time. A ratio inside one
/// round cancels a clock drift common to the round, so a machine whose speed
/// wanders - which every machine that boosts opportunistically does - still
/// orders options through it. The canary beside each pass is reported and gates
/// nothing, and the load a pass ran under is context rather than a bar.
///
/// \param report the report to conclude on, whose measurements carry the figures
/// \param rounds the round table the measurements were aggregated from, one row
///               per round and one column per option in the measurements' own
///               order, so a pair can be compared inside a round
void Conclude(OptionProbeReport& report, const std::vector<std::vector<double>>& rounds) {
    report.verdict = OptionProbeVerdict::kCannotDetermine;
    report.heuristicOption.clear();
    report.heuristicBasis.clear();
    report.hasDefault = false;

    // The comparison is not made in the load readings, and a run whose
    // calibration window never settled still measures; the load columns then have
    // nothing to be relative to, which is what this says.
    const std::string loadNote =
        report.calibrated
            ? std::string()
            : " The load instrument never established a floor on this machine, so the load columns "
              "of this report have nothing to be relative to; no figure above is made of them.";

    // Every refusal leaves the caller a default: the static reading of the
    // library's own tables. It is filled in one place, so no refusal path can
    // return without it and none of them prints it beside a measured winner.
    const auto refuse = [&report, &loadNote](const std::string& reason,
                                             const std::string& confidence) {
        report.reason = reason + loadNote;
        report.confidence = confidence;

        const StaticFallback fallback = StaticFallbackFor(report);

        if (!fallback.name.empty())
        {
            report.heuristicOption = fallback.name;
            report.heuristicBasis = fallback.basis;
            report.hasDefault = true;
        }
    };

    // The column an option's calls were timed in, which is its index in the
    // measurements: the round table and the measurement rows are built from the
    // same option book, in the same order.
    const auto column_of = [&report](const OptionProbeMeasurement* measurement) {
        return static_cast<std::size_t>(measurement - report.measurements.data());
    };

    const auto cheaper = [](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
        return a->nsPerArgument < b->nsPerArgument;
    };

    std::vector<const OptionProbeMeasurement*> live;

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured)
        {
            live.push_back(&measurement);
        }
    }

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

    const OptionProbeMeasurement* overall = *std::min_element(live.begin(), live.end(), cheaper);
    report.fastestOverall = overall->name;

    // The fastest option of the double lane's precision whose own bound is the
    // certified lane's or tighter: the fastest row a caller at the certified
    // accuracy can take, which is a narrower reading than the class's leader.
    const OptionProbeMeasurement* certified = nullptr;

    for (const OptionProbeMeasurement* measurement : live)
    {
        if (measurement->precision == OptionPrecision::kFp64 &&
            measurement->bound <= report.referenceBound &&
            (certified == nullptr || cheaper(measurement, certified)))
        {
            certified = measurement;
        }
    }

    if (certified != nullptr)
    {
        report.fastestAtReferenceAccuracy = certified->name;
    }

    // What one precision class's own rounds said. The members are ranked by the
    // run's statistic; every other member is then measured against the leader
    // pair by pair, and each pair is placed by its own band rather than by any
    // threshold on time.
    struct ClassOutcome {
        std::vector<const OptionProbeMeasurement*> members;
        std::vector<std::string> behind;
        std::vector<std::string> within;
        std::vector<std::string> ahead;
        const OptionProbeMeasurement* nearest = nullptr;
        PairedOutcome nearestPair;
        double widestBand = 0.0;
        double widestDrift = 0.0;
        std::string driftPair;
    };

    const auto outcome_of = [&rounds, &column_of, &cheaper, &live](OptionPrecision precision) {
        ClassOutcome outcome;

        for (const OptionProbeMeasurement* measurement : live)
        {
            if (measurement->precision == precision)
            {
                outcome.members.push_back(measurement);
            }
        }

        std::sort(outcome.members.begin(), outcome.members.end(),
                  [&cheaper](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
                      return cheaper(a, b);
                  });

        if (outcome.members.empty())
        {
            return outcome;
        }

        const OptionProbeMeasurement* leader = outcome.members.front();
        const std::size_t leaderColumn = column_of(leader);

        // The leader's own band is part of what the class's measurement showed,
        // so a class of one option still reports the width it was measured to.
        outcome.widestBand = std::max(0.0, leader->spread - 1.0);

        for (std::size_t index = 1; index < outcome.members.size(); ++index)
        {
            const OptionProbeMeasurement* rival = outcome.members[index];
            const PairedOutcome pair = CompareToLeader(rounds, leaderColumn, column_of(rival));
            const double width = pair.lo > 0.0 ? pair.hi / pair.lo - 1.0 : 0.0;

            outcome.widestBand = std::max(outcome.widestBand, width);
            outcome.widestBand = std::max(outcome.widestBand, rival->spread - 1.0);

            if (index == 1)
            {
                outcome.nearest = rival;
                outcome.nearestPair = pair;
            }

            if (std::abs(pair.drift) > std::abs(outcome.widestDrift))
            {
                outcome.widestDrift = pair.drift;
                outcome.driftPair =
                    Text("'%s' against '%s'", rival->name.c_str(), leader->name.c_str());
            }

            const std::string line = Text(
                "'%s' at %.2f ns/argument, %.1f%% of the leader's: its within-round ratio to the "
                "leader over the %d paired rounds fell in %.3f..%.3f, and it was the slower of the "
                "two in %d of them (its own rounds spread %.2fx)",
                rival->name.c_str(), rival->nsPerArgument,
                100.0 * (rival->nsPerArgument / leader->nsPerArgument), pair.rounds, pair.lo,
                pair.hi, pair.slowerRounds, rival->spread);

            if (pair.ordered)
            {
                outcome.behind.push_back(line);
            } else if (pair.hi < 1.0)
            {
                outcome.ahead.push_back(line);
            } else
            {
                outcome.within.push_back(line);
            }
        }

        return outcome;
    };

    // The classes: one per precision, each ordered inside itself. This is the
    // only partition the probe orders across, because it is the only one whose
    // members answer the same question - a lane that computes in single precision
    // is not a faster answer to the double lane's question, it is an answer to a
    // different one. Nothing below ever compares a row of one class with a row of
    // another.
    ClassOutcome doubles;
    const bool tooFewRounds = report.pairedRounds < static_cast<int>(kMinimumPairedRounds);

    for (const OptionPrecision precision :
         {OptionPrecision::kFp64, OptionPrecision::kFp32, OptionPrecision::kFp16,
          OptionPrecision::kBf16})
    {
        const ClassOutcome outcome = outcome_of(precision);

        OptionProbeClass entry;
        entry.precision = precision;
        entry.name = PrecisionName(precision);

        if (outcome.members.empty())
        {
            entry.note = "no option of this precision produced a figure on this run";
            report.classes.push_back(entry);
            continue;
        }

        entry.leader = outcome.members.front()->name;
        entry.leaderNsPerArgument = outcome.members.front()->nsPerArgument;

        double lowest = outcome.members.front()->bound;
        double highest = outcome.members.front()->bound;

        for (const OptionProbeMeasurement* member : outcome.members)
        {
            entry.ranked.push_back(member->name);
            lowest = std::min(lowest, member->bound);
            highest = std::max(highest, member->bound);
        }

        entry.ordered = tooFewRounds ? false
                                     : outcome.within.empty() && outcome.ahead.empty();

        if (tooFewRounds)
        {
            entry.note = Text("not ordered: the run took %d paired round(s), and a band over the "
                              "lower and upper quartiles of the within-round ratios needs %zu, so "
                              "nothing in this class was placed",
                              report.pairedRounds, kMinimumPairedRounds);
        } else if (!entry.ordered)
        {
            entry.note = Text("not ordered: %s",
                              !outcome.within.empty() ? outcome.within.front().c_str()
                                                      : outcome.ahead.front().c_str());
        } else if (outcome.members.size() == 1)
        {
            entry.note = Text("ordered: this class is one precision and one option of it was "
                              "measured, so it has no rival in its class to be ordered against. Its "
                              "own rounds spread %.2f, and %.3g is the bound it is documented at: a "
                              "bound is a column of this table and not the class's key",
                              outcome.members.front()->spread, lowest);
        } else
        {
            entry.note = Text(
                "ordered: this class is one precision, and each of the %zu option(s) behind '%s' "
                "was the slower of the two in the middle half of the %d paired rounds (the widest "
                "band in the class was %.2f%%). The bounds inside the class differ by row (%.3g to "
                "%.3g here), so the leader is the fastest option at some accuracy in this "
                "precision and not the fastest at yours: read the bound column.",
                outcome.members.size() - 1, entry.leader.c_str(), report.pairedRounds,
                100.0 * outcome.widestBand, lowest, highest);
        }

        report.classes.push_back(entry);

        if (precision == OptionPrecision::kFp64)
        {
            doubles = outcome;
        }
    }

    // A quartile band needs four rounds. With fewer, the lower and upper
    // quartiles are two- and three-point order statistics, and a probe that
    // ordered on them would be reporting a resolution it never measured.
    if (tooFewRounds)
    {
        refuse(Text("the run took %d paired round(s), and the lower and upper quartiles of a "
                    "within-round ratio need %zu of them: with fewer, the band is the ratio of two "
                    "or three rounds rather than a quartile of many, and this probe will not order "
                    "options on it. Raise ProbeOptions::passes or ProbeOptions::rounds and run "
                    "again",
                    report.pairedRounds, kMinimumPairedRounds),
               Text("CANNOT DETERMINE: %d paired rounds, and the comparison needs %zu",
                    report.pairedRounds, kMinimumPairedRounds));
        return;
    }

    if (doubles.members.empty())
    {
        refuse(Text("no option of the certified double lane's precision was measured cleanly, and "
                    "that is the class this verdict is made in; the fastest measured option "
                    "overall is '%s' at %.2f ns/argument, which is another precision's answer and "
                    "is not ordered against the double lane's",
                    overall->name.c_str(), overall->nsPerArgument),
               "CANNOT DETERMINE: the class the verdict is made in is empty");
        return;
    }

    const OptionProbeMeasurement* leader = doubles.members.front();

    // What this run can order, as the widest relative width of a within-round
    // band the class's own measurement showed - an option's own band or a
    // rival's band against the leader. It is reported beside the figures; the
    // ordering itself is made pair by pair from each pair's own band, so this
    // number bars nothing.
    report.resolution = doubles.widestBand;

    // The clock check, done rather than assumed, and attached to whatever the
    // verdict turns out to be. A pair whose ratio moves between the run's halves
    // is a pair the two options do not carry a decaying clock alike, and the two
    // differ in how exposed they are to the clock; a report that ordered without
    // saying so would imply the ordering holds at any clock. A pair whose ratio
    // held still inside the run's own resolution puts no such caveat on the
    // ordering. Which clock an option draws is a property of the registers it
    // runs in, so the clause also says whether the class this ordering compares
    // put two different arithmetic routes - two vector register widths - against
    // each other, read from the run's own rows rather than assumed.
    const auto clock_clause = [&]() -> std::string {
        if (doubles.members.size() < 2 || report.pairedRounds < kMinimumPairedRounds)
        {
            return std::string();
        }

        std::string clause;

        if (std::abs(doubles.widestDrift) > report.resolution)
        {
            clause = Text(". WARNING: the pair %s moved %.1f%% between the run's first and second "
                          "half of rounds, beyond the %.2f%% this run can order, so those two "
                          "options are not equally exposed to this machine's clock and the ordering "
                          "above is a property of this run's clock as well as of the options",
                          doubles.driftPair.c_str(), 100.0 * doubles.widestDrift,
                          100.0 * report.resolution);
        } else
        {
            clause = Text(". No pair's ratio moved between the run's halves by more than the %.2f%% "
                          "this run can order (the widest was %s at %.1f%%), so no measured pair was "
                          "more exposed to the clock's drift than the ordering's own precision",
                          100.0 * report.resolution, doubles.driftPair.c_str(),
                          100.0 * doubles.widestDrift);
        }

        const std::string& route = doubles.members.front()->arithmetic;
        const bool one_route = std::all_of(
            doubles.members.begin(), doubles.members.end(),
            [&route](const OptionProbeMeasurement* member) { return member->arithmetic == route; });

        if (one_route)
        {
            clause += Text(". Every option this comparison puts against another runs the same "
                           "arithmetic - %s for all %zu of them - so no pair here sets a wider vector "
                           "register against a narrower one, and the frequency a wider register draws "
                           "is not a difference between these options: what this run measured, above, "
                           "is what is reported in place of assuming it",
                           route.c_str(), doubles.members.size());
        } else
        {
            clause += Text(". This class does not run one arithmetic throughout, so a pair of it may "
                           "set a wider vector register against a narrower one and the two may draw "
                           "different clocks; the drift above is the whole of what this run measured "
                           "about that, and nothing here sets it aside");
        }

        return clause;
    };

    if (!doubles.within.empty() || !doubles.ahead.empty())
    {
        std::vector<std::string> unplaced = doubles.within;
        unplaced.insert(unplaced.end(), doubles.ahead.begin(), doubles.ahead.end());
        report.inseparable = unplaced;

        refuse(
            Text("'%s' is the fastest option of the certified double lane's precision by this "
                 "run's own statistic - %.2f ns/argument at the lower quartile of the %d paired "
                 "rounds, %.2f at the upper - but %zu of the %zu option(s) in that precision "
                 "could not be placed behind it: %s. The band and the round count of every one of "
                 "them are listed below. Ordering a pair whose band straddles one would be "
                 "ordering noise, and ordering one whose band lies below one would contradict this "
                 "run's own statistic, so the probe does neither; the static fallback below is what "
                 "a caller who needs a default takes instead",
                 leader->name.c_str(), leader->nsPerArgument, report.pairedRounds,
                 leader->nsPerArgumentMax, unplaced.size(), doubles.members.size(),
                 !doubles.within.empty() ? doubles.within.front().c_str()
                                         : doubles.ahead.front().c_str()),
            Text("CANNOT DETERMINE: %zu of %zu option(s) in the certified double lane's precision "
                 "could not be ordered against '%s' over the %d paired rounds; the widest band the "
                 "class showed was %.2f%%",
                 unplaced.size(), doubles.members.size(), leader->name.c_str(),
                 report.pairedRounds, 100.0 * report.resolution) +
                clock_clause());
        return;
    }

    report.verdict = OptionProbeVerdict::kRecommend;
    report.recommended = leader->name;
    report.hasDefault = true;

    const std::string shape =
        doubles.members.size() == 1
            ? "It is the only option of that precision this run measured, so it has no rival in "
              "its class to be ordered against."
            : Text("Each of the other %zu option(s) of the same precision was the slower of the two "
                   "in the middle half of the same rounds.",
                   doubles.members.size() - 1);

    report.reason = Text(
        "'%s' is the fastest option of the certified double lane's precision: %.2f ns/argument at "
        "the lower quartile of the %d paired rounds, %.2f at the upper (spread %.2fx), documented "
        "at %.3g and measured at %.3g over this workload. ",
        leader->name.c_str(), leader->nsPerArgument, report.pairedRounds,
        leader->nsPerArgumentMax, leader->spread, leader->bound, leader->maxError) +
                    shape;

    std::string confidence;

    if (doubles.nearest != nullptr)
    {
        confidence = Text(
            "HIGH: the nearest option of this precision, '%s', took %.1f%% of the leader's cost "
            "and was the slower of the two in %d of the %d paired rounds, its within-round band "
            "%.3f..%.3f clearing one",
            doubles.nearest->name.c_str(),
            100.0 * (doubles.nearest->nsPerArgument / leader->nsPerArgument),
            doubles.nearestPair.slowerRounds, doubles.nearestPair.rounds, doubles.nearestPair.lo,
            doubles.nearestPair.hi);
    } else
    {
        confidence = "HIGH: it is the only option this run measured in the certified double lane's "
                     "precision, so there is no rival of it to fall inside a resolution";
    }

    // The clock check, already built above so that a refusal carries it too.
    confidence += clock_clause();

    if (leader->bound > report.referenceBound)
    {
        confidence += Text(
            ". This leader's own bound is %.3g, looser than the certified lane's %.3g: it is the "
            "fastest option in this precision, at some accuracy in it",
            leader->bound, report.referenceBound);
    }

    if (overall != leader)
    {
        confidence += Text(
            ". The fastest option measured overall is '%s' at %.2f ns/argument, a different "
            "precision's answer: faster, not faster at this precision",
            overall->name.c_str(), overall->nsPerArgument);
    }

    confidence += Text(". The passes ran at a median load of %.1f%% inside their timed rounds",
                       report.loadMedian);

    if (report.passesWithinAlarm < static_cast<int>(report.passes.size()))
    {
        confidence += Text(
            ", and %d of %d pass(es) ran with the canary's own runs wider than the %.1f%% alarm - "
            "reported, used, and not what the ordering rests on",
            report.passesAboveAlarm, static_cast<int>(report.passes.size()),
            report.options.canarySpreadAlarm);
    }

    report.confidence = confidence + loadNote;
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
    std::vector<Option> options_ =
        EnumerateOptions(report.backends, report.granularities, cells, report.unoffered);

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
                    report.unoffered.end();

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

    text += "\n\nthe precision classes — one class per precision, the only sets this probe orders "
            "inside\n";
    text += "  A class is one precision, and the bounds inside it differ by row, so a class's "
            "leader is the\n";
    text += "  fastest option at some accuracy in that precision and not the fastest at yours: the "
            "bound\n";
    text += "  column above is what says which one a row is. Nothing here is ordered across "
            "classes — a\n";
    text += "  lane of another precision is an answer to a different question, not a faster answer "
            "to this\n";
    text += "  one.\n";

    for (const OptionProbeClass& entry : report.classes)
    {
        if (entry.leader.empty())
        {
            text += Text("  %-6s %s\n", entry.name.c_str(), entry.note.c_str());
            continue;
        }

        const OptionProbeMeasurement* leaderRow = nullptr;

        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.name == entry.leader)
            {
                leaderRow = &measurement;
            }
        }

        text += Text("  %-6s %zu option(s) measured | leader %s at %.2f ns/argument, documented at "
                     "%.3g | %s\n",
                     entry.name.c_str(), entry.ranked.size(), entry.leader.c_str(),
                     entry.leaderNsPerArgument,
                     leaderRow != nullptr ? leaderRow->bound : 0.0,
                     entry.ordered ? "ordered" : "NOT ORDERED");

        // Every member of the class, fastest first, each with the bound its own
        // lane documents — the row that says whether the leader is the fastest
        // at the accuracy the reader needs.
        std::vector<const OptionProbeMeasurement*> members;

        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.precision == entry.precision)
            {
                members.push_back(&measurement);
            }
        }

        std::sort(members.begin(), members.end(),
                  [](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
                      return a->nsPerArgument < b->nsPerArgument;
                  });

        std::vector<std::string> rankedLines;

        for (const OptionProbeMeasurement* member : members)
        {
            rankedLines.push_back(Text("%s %.2f ns (%.3g)", member->name.c_str(),
                                       member->nsPerArgument, member->bound));
        }

        text += WrappedList(rankedLines, "         ");
        text += Text("         %s\n", entry.note.c_str());

        if (entry.precision == OptionPrecision::kFp64)
        {
            text += "         (this is the class the verdict below is made in: the library's own "
                    "default\n";
            text += "         precision, and the one a caller who names no precision is asking "
                    "about)\n";
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
            "is unbuilt\n";
    text += "  work, counted here, and no cell of the product is absent from this book. A served "
            "cell has\n";
    text += "  an option row above unless the run was narrowed by the names the caller gave.\n";
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

    // The certified rows of the double lane's precision, and the class's own
    // entry so the resolution can be reported against the class it is measured
    // in rather than against one row of it.
    const OptionProbeMeasurement* leader = nullptr;
    const OptionProbeClass* doublesClass = nullptr;

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured && measurement.name == report.fastestAtReferenceAccuracy)
        {
            leader = &measurement;
        }
    }

    for (const OptionProbeClass& entry : report.classes)
    {
        if (entry.precision == OptionPrecision::kFp64)
        {
            doublesClass = &entry;
        }
    }

    text += "\nrecommendation\n";
    text += Text("  the rows of the %s class whose own bound is the certified lane's %.3g or "
                 "tighter, measured\n",
                 PrecisionName(OptionPrecision::kFp64), report.referenceBound);
    text += "  cleanly. This is the narrowest reading of that class, not a class of its own: the "
            "class holds\n";
    text += "  the relaxed rungs and the narrow partition beside these rows, each showing its own "
            "bound\n";
    text += "  above.\n";

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.bound > report.referenceBound)
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

    if (!report.fastestOverall.empty())
    {
        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.name == report.fastestOverall)
            {
                text += Text("  fastest measured overall: %s at %.2f ns/argument, documented at "
                             "%.3g%s\n",
                             measurement.name.c_str(), measurement.nsPerArgument, measurement.bound,
                             (measurement.bound > report.referenceBound)
                                 ? " — a looser accuracy class: faster, not faster at the same "
                                   "accuracy"
                                 : "");
            }
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
        text += Text("  recommended option: %s\n", report.recommended.c_str());
    }

    if (!report.reason.empty())
    {
        text += Text("  reason: %s\n", report.reason.c_str());
    }

    if (!report.inseparable.empty())
    {
        text += "\n  not placed behind the leader by the paired rounds:\n";

        for (const std::string& entry : report.inseparable)
        {
            text += Text("    * %s\n", entry.c_str());
        }
    }

    if (!report.confidence.empty())
    {
        text += Text("  confidence: %s\n", report.confidence.c_str());
    }

    // The fallback, in its own section and labelled where a consumer reads it.
    // It is a different kind of answer from everything above, and printing it in
    // the same list as a measured cost is the one thing this must not do.
    if (!report.heuristicOption.empty())
    {
        text += "\nstatic fallback — a heuristic, not a measurement\n";
        text += "  The measurement above did not order this field, so the default below is chosen "
                "by counting\n";
        text += "  the library's own tables and never by timing anything:\n";
        text += Text("    option: %s\n", report.heuristicOption.c_str());
        text += Text("    basis:  %s\n", report.heuristicBasis.c_str());
        text += "  It is a property of the tables this build ships and not of this machine, so it "
                "holds\n";
        text += "  anywhere this build runs — and it is not evidence that the option is fast here. "
                "No figure\n";
        text += "  above this line is derived from it.\n";
    } else if (report.verdict == OptionProbeVerdict::kCannotDetermine)
    {
        text += "\nstatic fallback: none. The measurement did not order the field and this run's "
                "option set\n";
        text += "  holds nothing the static rule can rank against the certified bound, so neither "
                "answered.\n";
    } else
    {
        text += "\nstatic fallback: none needed. The verdict above is a measurement, and no "
                "heuristic is\n";
        text += "  printed beside it.\n";
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
    text += Text("  * the %zu cells of the option space this build refuses, of %zu: %zu on the "
                 "rational route\n",
                 refused, report.cells.size(), refusedRoute);
    text += Text("    (a table to generate), %zu on the across-orders packing axis (a kernel to "
                 "write) and %zu\n",
                 refusedAxis, refusedRung);
    text += "    at the relaxed rungs (a degree table to derive). Each is named with the library's "
            "own reason\n";
    text += "    in the coverage above: they are unbuilt work, counted rather than absent, and no "
            "cell of the\n";
    text += "    library's product is missing from this report;\n";
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
