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
//   1. A load instrument that needs no platform API. The options are ranked by
//      a wall-clock minimum, and a wall clock is only worth reading on a
//      machine nobody else is using, so every pass carries a reading of how
//      much of a core this process actually got. The instrument is a fixed-work
//      integer spin: the fastest time that same work has ever taken here is the
//      machine's quiet floor, and a reading is the percentage by which a run of
//      it exceeded that floor. An integer chain is used deliberately - it holds
//      no floating-point state, so the arithmetic question the probe is about
//      cannot change the cost of the instrument itself.
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
/// Loose on purpose, and not the admission rule. The floor is the denominator of
/// the load readings, which are context, so a busy calibration window costs a
/// little accuracy in those numbers and nothing else. What admits a pass is the
/// canary's spread *inside* that pass, so a machine that is busy throughout still
/// gets measured and told what its own wandering allows it to order. This bar is
/// here only to catch a window that never settled into a floor at all: at 100 the
/// median run may take twice the fastest, and beyond that the fastest run is a
/// glitch rather than a floor.
constexpr double kCalibrationTolerancePercent = 100.0;

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
    double bound = 0.0;
};

/// The packed backends' names, which the library's own backend test pins. The
/// scalar names are not written here: they are the types' own kName.
constexpr const char* kPackedFp64Name = "avx2-fp64";
constexpr const char* kPackedFp32Name = "avx2-fp32";

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

    // The bound of a cell option is the bound its own lane documents: the
    // partition's certified figure where the cell reads a partition's tables, and
    // the rung's own budget everywhere else. Read from the library either way, so
    // a regeneration that moved a figure moves the row.
    for (Option& option : options)
    {
        if (option.kind != OptionKind::kFp64Cell)
        {
            continue;
        }

        if (option.granularity == FitGranularity::kNarrow)
        {
            for (const FitGranularityInfo& partition : partitions)
            {
                if (partition.granularity == FitGranularity::kNarrow)
                {
                    option.bound = partition.bound;
                }
            }
        } else
        {
            option.bound = TierBound(option.tier);
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

/// The narrow partition, which only exists as a policy: the library reports the
/// partition and refuses the combinations it has no tables for, and the probe
/// reaches it the way a consumer does, by naming it in the policy's axes.
void NarrowCell(EvalScheme scheme, int nmax, double x, double* out) noexcept {
    constexpr FitRoute kRoute = FitRoute::kChebyshev;
    constexpr PackAxis kPack = PackAxis::kArguments;
    constexpr BoysBudget kBudget = BoysBudget::kFloat;

    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      EvalPolicy<kRoute, EvalScheme::kHorner, kBudget, kPack,
                                 FitGranularity::kNarrow>>(nmax, x, out);
    } else
    {
        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      EvalPolicy<kRoute, EvalScheme::kSplitClenshaw, kBudget, kPack,
                                 FitGranularity::kNarrow>>(nmax, x, out);
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
        NarrowCell(option.scheme, nmax, x, out);
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

/// Fills in the recommendation, or the refusal, from the figures already in the
/// report.
///
/// The rule, in one place: the recommendation is the fastest option documented
/// at the certified lane's own accuracy, and it is only made when every rival in
/// that accuracy class is further behind it than this run's resolution. The
/// resolution is measured, not chosen: it is the larger of the widest spread the
/// canary showed inside an admitted pass and the leader's own spread across its
/// clean passes, so a machine whose speed wanders orders only large differences
/// and a steady one orders small ones. Everything else - a rival inside the
/// resolution, no clean pass, no option in the accuracy class, an instrument
/// that never found a floor - ends in the refusal, because ordering noise is the
/// one outcome this tool exists to avoid.
void Conclude(OptionProbeReport& report) {
    report.verdict = OptionProbeVerdict::kCannotDetermine;

    if (!report.calibrated)
    {
        if (report.canaryCalibrationRuns == 0)
        {
            report.reason =
                "the load instrument could not be calibrated: its calibration window was empty, "
                "so the fixed-work canary never ran and no pass has anything to be judged "
                "against; no pass was run";
        } else
        {
            const double medianAboveFloor =
                report.canaryFloorMs > 0.0
                    ? 100.0 * (report.canaryCalibrationMedianMs / report.canaryFloorMs - 1.0)
                    : 0.0;

            report.reason = Text(
                "the load instrument could not be calibrated: %d runs of the fixed-work canary "
                "over %.3f s spread %.1f%% and their median sat %.1f%% above the fastest of them, "
                "so this machine never repeated fixed work closely enough for a pass to be judged; "
                "no pass was run",
                report.canaryCalibrationRuns, report.options.calibrationSeconds,
                report.canaryCalibrationSpread, medianAboveFloor);
        }

        report.confidence = "CANNOT DETERMINE: nothing was measured";
        return;
    }

    const auto faster = [](const OptionProbeMeasurement& a, const OptionProbeMeasurement& b) {
        return a.nsPerArgument < b.nsPerArgument;
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
        report.reason = report.cleanPasses == 0
                            ? "every pass was discarded: the canary's own runs disagreed by more "
                              "than the bar inside each of them, so this probe has no figure it "
                              "is willing to report. The canary column beside each pass says how "
                              "much that pass wandered; a quieter machine is what would let this "
                              "run order anything"
                            : "no option produced a figure on a pass the canary vouched for";
        report.confidence = "CANNOT DETERMINE";
        return;
    }

    // The fastest of a set, by the cost the report orders on.
    const auto cheapest = [&faster](const OptionProbeMeasurement* a,
                                    const OptionProbeMeasurement* b) { return faster(*a, *b); };

    const OptionProbeMeasurement* overall = *std::min_element(live.begin(), live.end(), cheapest);
    report.fastestOverall = overall->name;

    // The classes: one per precision, each ranked inside itself. This is the
    // only partition the probe orders across, because it is the only one whose
    // members answer the same question — a lane that computes in single
    // precision is not a faster answer to the double lane's question, it is an
    // answer to a different one. Nothing below ever compares a row of one class
    // against a row of another.
    const auto members_of = [&live, &cheapest](OptionPrecision precision) {
        std::vector<const OptionProbeMeasurement*> members;

        for (const OptionProbeMeasurement* measurement : live)
        {
            if (measurement->precision == precision)
            {
                members.push_back(measurement);
            }
        }

        std::sort(members.begin(), members.end(), cheapest);
        return members;
    };

    // The double lane's own certified reading: the options whose bound is the
    // certified lane's or tighter. It is a reading of the fp64 class rather than
    // a class of its own — a relaxed rung and the narrow partition still stand
    // in that class beside the certified rows, showing their own bounds.
    const OptionProbeMeasurement* certified = nullptr;

    for (const OptionProbeMeasurement* measurement : live)
    {
        if (measurement->precision == OptionPrecision::kFp64 &&
            measurement->bound <= report.referenceBound &&
            (certified == nullptr || faster(*measurement, *certified)))
        {
            certified = measurement;
        }
    }

    if (certified != nullptr)
    {
        report.fastestAtReferenceAccuracy = certified->name;
    }

    // The verdict is made in the double lane's precision: it is the library's
    // default lane, the one the certified reference is in, and the lane whose
    // class a caller who names no precision is asking about.
    const std::vector<const OptionProbeMeasurement*> doubles = members_of(OptionPrecision::kFp64);

    if (doubles.empty())
    {
        report.reason = Text(
            "no option of the certified double lane's precision was measured cleanly; the fastest "
            "measured option overall is '%s' at %.2f ns/argument, which is another precision's "
            "answer and is not ordered against the double lane's",
            overall->name.c_str(), overall->nsPerArgument);
        report.confidence = "CANNOT DETERMINE";
        return;
    }

    const OptionProbeMeasurement* leader = doubles.front();

    // An ordering rests on how the admitted passes disagreed with each other. A
    // single admitted pass has nothing to disagree with - its spread is 1 by
    // construction - so a resolution built on it would report a precision the run
    // never measured, which is the one thing this entry must not do.
    if (report.cleanPasses < 2)
    {
        report.reason = Text(
            "'%s' is the fastest option of the certified double lane's precision (%.2f "
            "ns/argument), but only %d of %d passes was admitted, and how far two options can be "
            "ordered apart rests on how the admitted passes disagreed; a second pass the canary "
            "vouches for is what this needs",
            leader->name.c_str(), leader->nsPerArgument, report.cleanPasses,
            static_cast<int>(report.passes.size()));
        report.confidence =
            Text("CANNOT DETERMINE: %d of %d passes admitted, and the resolution needs two",
                 report.cleanPasses, static_cast<int>(report.passes.size()));
        return;
    }

    // What this run can order. Both terms are measured here: the instrument's
    // own disagreement with itself, and how far the leader's clean passes
    // disagreed with each other. A machine that is steadily busy does not inflate
    // either - it slows every option by the same factor, and a ratio is what is
    // compared - so a loaded but steady run still orders small differences, and
    // an unsteady one stops at large ones.
    report.resolution = std::max(report.canarySpread / 100.0, leader->spread - 1.0);

    // Every class's own ranking, at the resolution this run measured. A class
    // whose leader is clear of its own field is ordered; one whose field is
    // inside the resolution is not, and says so, rather than being folded into
    // another class's answer.
    for (const OptionPrecision precision :
         {OptionPrecision::kFp64, OptionPrecision::kFp32, OptionPrecision::kFp16,
          OptionPrecision::kBf16})
    {
        OptionProbeClass entry;
        entry.precision = precision;
        entry.name = PrecisionName(precision);

        const std::vector<const OptionProbeMeasurement*> members = members_of(precision);

        if (members.empty())
        {
            entry.note = "no option of this precision produced a figure on this run";
            report.classes.push_back(entry);
            continue;
        }

        entry.leader = members.front()->name;
        entry.leaderNsPerArgument = members.front()->nsPerArgument;

        double lowest = members.front()->bound;
        double highest = members.front()->bound;

        for (const OptionProbeMeasurement* member : members)
        {
            entry.ranked.push_back(member->name);
            lowest = std::min(lowest, member->bound);
            highest = std::max(highest, member->bound);
        }

        std::vector<std::string> within;

        for (std::size_t i = 1; i < members.size(); ++i)
        {
            if (members[i]->nsPerArgument <=
                members.front()->nsPerArgument * (1.0 + report.resolution))
            {
                within.push_back(Text(
                    "%s %.2f ns/argument (%.2f%% behind the leader, inside the %.2f%% this run "
                    "can order)",
                    members[i]->name.c_str(), members[i]->nsPerArgument,
                    100.0 * (members[i]->nsPerArgument / members.front()->nsPerArgument - 1.0),
                    100.0 * report.resolution));
            }
        }

        entry.ordered = within.empty();
        entry.note =
            within.empty()
                ? Text("ordered: this class is one precision, %zu option(s) of it were measured "
                       "cleanly, and the nearest of them is beyond the %.2f%% this run can order. "
                       "The bounds inside the class differ by row (%.3g to %.3g here), so the "
                       "leader is the fastest option at some accuracy in this precision and not "
                       "the fastest at yours: read the bound column.",
                       members.size(), 100.0 * report.resolution, lowest, highest)
                : Text("not ordered: %s, and this run can only order %.2f%%",
                       within.front().c_str(), 100.0 * report.resolution);

        if (precision == OptionPrecision::kFp64)
        {
            report.inseparable = within;
        }

        report.classes.push_back(entry);
    }

    if (!report.inseparable.empty())
    {
        report.reason = Text(
            "'%s' is the fastest option of the certified double lane's precision (%.2f "
            "ns/argument), but '%s' (%.2f ns/argument) is %.2f%% behind it, inside the %.2f%% "
            "this run can order; the probe does not order noise",
            leader->name.c_str(), leader->nsPerArgument, doubles[1]->name.c_str(),
            doubles[1]->nsPerArgument,
            100.0 * (doubles[1]->nsPerArgument / leader->nsPerArgument - 1.0),
            100.0 * report.resolution);
        report.confidence = Text("CANNOT DETERMINE: %zu of %zu options in the certified double "
                                 "lane's precision are inside the %.2f%% this run can order, which "
                                 "is what the canary and the leader's own passes measured",
                                 report.inseparable.size(), doubles.size(),
                                 100.0 * report.resolution);
        return;
    }

    report.verdict = OptionProbeVerdict::kRecommend;
    report.recommended = leader->name;

    const OptionProbeMeasurement* nearest = doubles.size() > 1 ? doubles[1] : nullptr;

    report.reason = Text(
        "'%s' is the fastest option of the certified double lane's precision, documented at %.3g "
        "and measured cleanly: %.2f ns/argument over %d admitted of %d passes, spread %.2fx, "
        "taken at a load of %.1f%% inside the pass",
        leader->name.c_str(), leader->bound, leader->nsPerArgument, leader->cleanPasses,
        leader->cleanPasses + leader->disturbedPasses, leader->spread, leader->loadAtMinimum);

    report.confidence = Text("HIGH: the nearest option of this precision is at least %.2f%% "
                             "behind, beyond the %.2f%% this run can order (canary spread %.2f%%, "
                             "leader's own passes spread %.2f%%)",
                             nearest != nullptr
                                 ? 100.0 * (nearest->nsPerArgument / leader->nsPerArgument - 1.0)
                                 : 0.0,
                             100.0 * report.resolution, report.canarySpread,
                             100.0 * (leader->spread - 1.0));

    if (leader->bound > report.referenceBound)
    {
        report.confidence += Text(". This leader's own bound is %.3g, looser than the certified "
                                  "lane's %.3g: it is the fastest option in this precision, at some "
                                  "accuracy in it",
                                  leader->bound, report.referenceBound);
    }

    if (overall != leader)
    {
        report.confidence += Text(". The fastest option measured overall is '%s' at %.2f "
                                  "ns/argument, a different precision's answer: faster, not faster "
                                  "at this precision",
                                  overall->name.c_str(), overall->nsPerArgument);
    }
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
    std::vector<std::vector<double>> passCost(options.passes);

    for (std::vector<double>& column : passCost)
    {
        column.assign(options_.size(), std::numeric_limits<double>::infinity());
    }

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

    // The canary's runs inside a pass: one before the first round and one after
    // every round, so the set brackets the timed work and spans it. Each is
    // taken between rounds, never inside a timed region, so the instrument can
    // never land in a figure. At least three, because a spread needs a set that
    // can disagree with itself.
    const int canarySamples = std::max(3, options.rounds + 1);

    if (report.calibrated)
    {
        for (int pass = 0; pass < options.passes; ++pass)
        {
            OptionProbePass record;
            std::vector<double> canaryMs;

            record.backgroundLoad = meter.Average(options.backgroundWindowSeconds);
            canaryMs.push_back(meter.SampleMs());
            record.beforeLoad = meter.Percent(canaryMs.back());

            for (int round = 0; round < options.rounds; ++round)
            {
                for (std::size_t index = 0; index < options_.size(); ++index)
                {
                    const Clock::time_point t0 = Clock::now();
                    const double sum = Checksum(work, buffers, options_[index]);
                    const Clock::time_point t1 = Clock::now();

                    const double milliseconds = MillisecondsBetween(t0, t1);
                    const double perArgument =
                        milliseconds * 1e6 / static_cast<double>(options.count);
                    passCost[static_cast<std::size_t>(pass)][index] =
                        std::min(passCost[static_cast<std::size_t>(pass)][index], perArgument);
                    record.seconds += SecondsBetween(t0, t1);
                    report.measurements[index].checkedSum = sum;
                }

                canaryMs.push_back(meter.SampleMs());
            }

            while (static_cast<int>(canaryMs.size()) < canarySamples)
            {
                canaryMs.push_back(meter.SampleMs());
            }

            record.afterLoad = meter.Percent(canaryMs.back());
            record.inPassLoad = meter.Percent(MedianMilliseconds(canaryMs));
            record.canarySpread = SpreadPercent(canaryMs);
            record.disturbed = record.canarySpread > options.canarySpreadThreshold;

            if (record.disturbed)
            {
                ++report.disturbedPasses;
            } else
            {
                ++report.cleanPasses;
                report.canarySpread = std::max(report.canarySpread, record.canarySpread);
            }

            report.passes.push_back(record);
        }
    }

    // The accuracy column, measured once per option outside the clock. It does
    // not depend on a clean pass: the values a lane returns for an argument are
    // a property of the build, not of the machine's load.
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
        measurement.maxError = walker.MaxError();
        measurement.bitIdenticalToReference = walker.BitIdentical();
        measurement.meetsBound = measurement.maxError <= measurement.bound;
        measurement.withinReferenceFloor =
            !measurement.meetsBound && measurement.maxError <= report.referenceBound;
    }

    // The figures: the minimum of the clean passes, with the ratio of the
    // slowest clean pass to it as the spread. A disturbed pass is not averaged
    // in and does not take part in either end.
    for (std::size_t pass = 0; pass < report.passes.size(); ++pass)
    {
        if (report.passes[pass].disturbed)
        {
            continue;
        }

        for (std::size_t index = 0; index < options_.size(); ++index)
        {
            OptionProbeMeasurement& measurement = report.measurements[index];
            const double cost = passCost[pass][index];

            if (!measurement.measured || cost < measurement.nsPerArgument)
            {
                measurement.nsPerArgument = cost;
                measurement.loadAtMinimum = report.passes[pass].inPassLoad;
            }

            measurement.measured = true;
            measurement.nsPerArgumentMax = std::max(measurement.nsPerArgumentMax, cost);
            ++measurement.cleanPasses;
        }
    }

    for (std::size_t pass = 0; pass < report.passes.size(); ++pass)
    {
        if (!report.passes[pass].disturbed)
        {
            continue;
        }

        for (OptionProbeMeasurement& measurement : report.measurements)
        {
            ++measurement.disturbedPasses;
        }
    }

    for (OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured && measurement.nsPerArgument > 0.0)
        {
            measurement.spread = measurement.nsPerArgumentMax / measurement.nsPerArgument;
        }
    }

    Conclude(report);

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
    text += Text("            calibration %.3f s | a pass is admitted while the canary's own "
                 "runs across it\n",
                 report.options.calibrationSeconds);
    text += Text("            stay within %.1f%% of each other\n",
                 report.options.canarySpreadThreshold);

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
    text += "            A pass is admitted on the canary's own spread, not on how busy the "
            "machine\n";
    text += "            looked: a steady load slows every option by the same factor and leaves\n";
    text += "            their order alone, because a ratio is what is compared, while a load "
            "that\n";
    text += "            wanders is what corrupts a comparison, and the spread between the\n";
    text += "            canary's own runs is that wandering, measured directly.\n";
    text += "  load readings: context, printed beside each pass and not the gate. A reading is "
            "the\n";
    text += "            percentage by which the canary took longer than its floor, so it says "
            "how\n";
    text += "            busy the machine looked, not how well it could be measured. It is not "
            "the\n";
    text += "            machine's total utilization: no standard-library call can read another\n";
    text += "            process's processor time. A pass the canary cannot vouch for is "
            "reported\n";
    text += "            as disturbed and excluded from every figure rather than averaged "
            "in.\n";
    text += Text("  reference: the certified per-order fp64 lane, BoysSingle, documented at "
                 "%.3g,\n",
                 report.referenceBound);
    text += "            evaluated at the order and the argument each option was actually asked\n";
    text += "            about. A difference at or below that bound is not separable from the\n";
    text += "            certified lane by this comparison; the committed reference grid in the\n";
    text += "            test suite is what proves the bound itself.\n";

    text += "\npass  wall_s  background%  before%  after%  in-pass%  canary%  verdict\n";

    for (std::size_t i = 0; i < report.passes.size(); ++i)
    {
        const OptionProbePass& pass = report.passes[i];
        text += Text("%4zu  %6.3f  %11.1f  %7.1f  %6.1f  %8.1f  %7.2f  %s\n", i + 1, pass.seconds,
                     pass.backgroundLoad, pass.beforeLoad, pass.afterLoad, pass.inPassLoad,
                     pass.canarySpread, pass.disturbed ? "DISTURBED (excluded)" : "admitted");
    }

    if (report.passes.empty())
    {
        text += "     (no pass was run: the canary never settled, so no pass could be judged — "
                "see the reason below)\n";
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

    text += Text("  %-*s %-6s %-13s %-9s %12s  %7s  %5s  %6s  %11s  %10s  %s\n",
                 static_cast<int>(nameWidth), "option", "prec", "arithmetic", "contracts", "ns/arg",
                 "spread", "clean", "load%", "max error", "bound", "verdict");

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        // The verdict is the accuracy one and it does not depend on a single
        // pass being admitted: the values a lane returns for an argument are a
        // property of the build, and a busy machine that discards every pass
        // must not be able to blank the accuracy column with it.
        std::string verdict = measurement.meetsBound ? "within bound"
                              : measurement.withinReferenceFloor
                                  ? "within the floor"
                                  : "ABOVE BOUND";

        if (measurement.bitIdenticalToReference)
        {
            verdict = "reference";
        }

        if (measurement.measured)
        {
            text += Text("  %-*s %-6s %-13s %-9s %12.2f  %6.2fx  %2d/%d  %6.1f  %11.3e  %10.3g  %s\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         PrecisionName(measurement.precision), measurement.arithmetic.c_str(),
                         measurement.contracts ? "yes" : "no",
                         measurement.nsPerArgument,
                         measurement.spread,
                         measurement.cleanPasses,
                         measurement.cleanPasses + measurement.disturbedPasses,
                         measurement.loadAtMinimum,
                         measurement.maxError,
                         measurement.bound,
                         verdict.c_str());
        } else
        {
            text += Text("  %-*s %-6s %-13s %-9s %12s  %7s  %2d/%d  %6s  %11.3e  %10.3g  %s\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         PrecisionName(measurement.precision), measurement.arithmetic.c_str(),
                         measurement.contracts ? "yes" : "no", "not measured", "-",
                         measurement.cleanPasses,
                         measurement.cleanPasses + measurement.disturbedPasses, "-",
                         measurement.maxError, measurement.bound, verdict.c_str());
        }
    }

    text += "\n  (prec is the precision class the row is ranked in, and the only set of options it "
            "is ever\n";
    text += "   ordered against: a lane that computes in single precision is not a faster answer to "
            "the\n";
    text += "   double lane's question, it is an answer to a different one, so no column here is "
            "read\n";
    text += "   across precisions. load% is the canary's reading inside the pass each figure was "
            "taken in —\n";
    text += "   the load that number was taken under. The verdict column is the accuracy one, "
            "against the\n";
    text += "   bound in the row's own bound column: 'within bound' met it, 'within the floor' is "
            "above its\n";
    text += Text("   own bound but at or below the certified lane's %.3g, which this comparison "
                 "cannot\n",
                 report.referenceBound);
    text += "   separate, and 'ABOVE BOUND' means the option did not deliver what its own lane "
            "documents.\n";
    text += "   Every bound in the column is read from the library: the certified lane's own figure "
            "for\n";
    text += "   the certified rows, the accuracy rung's for a rung, the partition's for a "
            "partition, and\n";
    text += "   the published per-lane figure for the narrower lanes. The verdict does not wait on "
            "a pass:\n";
    text += "   the values a lane returns are a property of the build, so a run whose passes were "
            "all\n";
    text += "   discarded by the load instrument still reports which rows delivered what they "
            "document,\n";
    text += "   and its cost columns read 'not measured' where no figure could be stood behind.)\n";

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

        text += Text("  %-6s %zu option(s) measured cleanly | leader %s at %.2f ns/argument, "
                     "documented at %.3g | %s\n",
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
        text += Text("             route(s) %s | delivered %.3g | bound %.3g\n",
                     PartitionRouteNames(partition).c_str(), partition.delivered, partition.bound);
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

    // What this run could order, and which of the two measured terms set it, so
    // a reader can see the number was not chosen in advance.
    const OptionProbeMeasurement* leader = nullptr;

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured && measurement.name == report.fastestAtReferenceAccuracy)
        {
            leader = &measurement;
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
            text += Text("    %-*s %8.2f ns/argument  spread %.2fx  %d clean pass(es)\n",
                         static_cast<int>(nameWidth), measurement.name.c_str(),
                         measurement.nsPerArgument, measurement.spread, measurement.cleanPasses);
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

    if (leader == nullptr)
    {
        text += "\n  resolution: not measurable on this run — the accuracy class produced no "
                "figure, so\n";
        text += "              there was nothing to order\n";
    } else if (report.cleanPasses < 2)
    {
        text += Text("\n  resolution: not measurable on this run — an ordering rests on how the "
                     "admitted\n");
        text += Text("              passes disagreed with each other, and %d of %d was admitted\n",
                     report.cleanPasses, static_cast<int>(report.passes.size()));
    } else
    {
        text += Text("\n  resolution: options closer than %.2f%% cannot be ordered on this run. "
                     "The number is\n",
                     100.0 * report.resolution);
        text += "              measured, not assumed: it is the wider of the canary's own spread "
                "inside\n";
        text += Text("              an admitted pass (%.2f%%) and the leading option's own "
                     "admitted\n",
                     report.canarySpread);
        text += Text("              passes (%.2f%%), whichever was wider. A steady load widens "
                     "neither — it\n",
                     100.0 * (leader->spread - 1.0));
        text += "              slows every option alike — so this number moves with how steady "
                "the machine\n";
        text += "              was, not with how busy it was.\n";
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

    for (const std::string& entry : report.inseparable)
    {
        text += Text("  not separable: %s\n", entry.c_str());
    }

    if (!report.confidence.empty())
    {
        text += Text("  confidence: %s\n", report.confidence.c_str());
    }

    text += "\n  this ranking is this machine's: a different host, a different compiler or a "
            "different set of\n";
    text += "  build flags can rank these options differently, which is why the probe is run where "
            "the numbers\n";
    text += "  are used rather than read from documentation. A figure above is a minimum of "
            "admitted\n";
    text += "  passes, not a mean, and its spread is what the passes it rests on disagreed by. "
            "The\n";
    text += "  resolution above is this run's: a steadier or busier machine would give a "
            "different one.\n";

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
