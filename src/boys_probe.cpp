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

/// Which entries an option is, and therefore which arithmetic it runs in.
enum class OptionKind {
    kBatchFp64, ///< one all-orders call per argument, fp64
    kTierFp64, ///< the same, at a run-time accuracy tier
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

/// Every option this build offers, enumerated from the library.
///
/// The options that exist as entries are named here, because an entry is a
/// function and no table of them exists to read. What is not written here is
/// what this build carries: each option's arithmetic is resolved against
/// backend::BoysBackends(), so an option whose arithmetic the table lacks is
/// reported as not offered rather than measured under a name that is not this
/// build's; and the relaxed tiers are found by asking the library what each
/// tier it can name delivers, so a build serving fewer rungs offers fewer
/// options. The half-precision lanes are behind the same BoysFp16 seam that
/// declares them.
std::vector<Option> EnumerateOptions(std::span<const backend::BackendInfo> table,
                                     std::vector<std::string>& unoffered) {
    std::vector<Option> options;

    const backend::BackendInfo* fp64 = ResolveArithmetic(table, true);
    const backend::BackendInfo* fp32 = ResolveArithmetic(table, false);

    const auto append = [&](const char* name, OptionKind kind, AccuracyTier tier,
                            const backend::BackendInfo* arithmetic, double bound) {
        if (arithmetic == nullptr)
        {
            unoffered.emplace_back(name);
            return;
        }

        Option option;
        option.name = name;
        option.arithmetic = arithmetic->name;
        option.contracts = arithmetic->contracts;
        option.kind = kind;
        option.tier = tier;
        option.bound = bound;
        options.push_back(option);
    };

    append("batch-fp64", OptionKind::kBatchFp64, AccuracyTier::kReference, fp64,
           TierBound(AccuracyTier::kReference));
    append("grouped-fp64", OptionKind::kGroupedFp64, AccuracyTier::kReference, fp64,
           TierBound(AccuracyTier::kReference));
    append("tagged-fp64", OptionKind::kTaggedFp64, AccuracyTier::kReference, fp64,
           TierBound(AccuracyTier::kReference));
    append("batch-fp32", OptionKind::kBatchFp32, AccuracyTier::kReference, fp32,
           kFloatLaneBound);

#if BoysFp16
    // The half lanes run the fp32 engine, so their arithmetic is the fp32 one.
    append("f16-io", OptionKind::kBatchF16, AccuracyTier::kReference, fp32, kHalfIoLaneBound);
    append("bf16-io", OptionKind::kBatchBf16, AccuracyTier::kReference, fp32, kBf16IoLaneBound);
#endif

    // The relaxed rungs, walked over the enum's integer range rather than over
    // the enumerators: a tier this build does not serve reports the reference
    // multiplier's reachable error, which is the rung the library evaluates it
    // at, and the rungs this build does serve report their own, so the ones
    // whose report differs from the reference's are exactly the tiers the build
    // serves. A build that adds, drops or spaces out a rung is measured as it
    // is, and each option is named after the multiplier the library reports for
    // its tier rather than after a multiplier written here.
    constexpr int kMaxTierProbe = 32;
    const double referenceReachable = QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0)
                                          .reachable;

    for (int raw = 1; raw <= kMaxTierProbe; ++raw)
    {
        const auto tier = static_cast<AccuracyTier>(raw);

        if (QueryTier(tier, AccuracyRegion::kA, 0.0).reachable == referenceReachable)
        {
            continue;
        }

        const std::string name = Text("tier-%g-fp64", AccuracyMultiplier(tier));
        append(name.c_str(), OptionKind::kTierFp64, tier, fp64, TierBound(tier));
    }

    return options;
}

// --- one option's values ----------------------------------------------------

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

    // The accuracy class: the certified lane's own documented bound, read from
    // the library. An option in it is documented at an accuracy this probe
    // cannot separate from the reference's.
    std::vector<const OptionProbeMeasurement*> sameClass;

    for (const OptionProbeMeasurement* measurement : live)
    {
        if (measurement->bound <= report.referenceBound)
        {
            sameClass.push_back(measurement);
        }
    }

    if (sameClass.empty())
    {
        report.reason = Text(
            "no option documented at or below the certified lane's %.3g was measured cleanly; the "
            "fastest measured option overall is '%s' at %.2f ns/argument, documented at %.3g",
            report.referenceBound, overall->name.c_str(), overall->nsPerArgument, overall->bound);
        report.confidence = "CANNOT DETERMINE";
        return;
    }

    const OptionProbeMeasurement* leader =
        *std::min_element(sameClass.begin(), sameClass.end(), cheapest);
    report.fastestAtReferenceAccuracy = leader->name;

    // An ordering rests on how the admitted passes disagreed with each other. A
    // single admitted pass has nothing to disagree with - its spread is 1 by
    // construction - so a resolution built on it would report a precision the run
    // never measured, which is the one thing this entry must not do.
    if (report.cleanPasses < 2)
    {
        report.reason = Text(
            "'%s' is the fastest option documented at the certified lane's accuracy (%.2f "
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

    std::vector<const OptionProbeMeasurement*> inseparable;

    for (const OptionProbeMeasurement* rival : sameClass)
    {
        if (rival == leader)
        {
            continue;
        }

        if (rival->nsPerArgument <= leader->nsPerArgument * (1.0 + report.resolution))
        {
            inseparable.push_back(rival);
        }
    }

    for (const OptionProbeMeasurement* rival : inseparable)
    {
        report.inseparable.push_back(Text(
            "%s %.2f ns/argument (%.2f%% behind the leader, inside the %.2f%% this run can order)",
            rival->name.c_str(), rival->nsPerArgument,
            100.0 * (rival->nsPerArgument / leader->nsPerArgument - 1.0),
            100.0 * report.resolution));
    }

    if (!inseparable.empty())
    {
        report.reason = Text(
            "'%s' is the fastest option documented at the certified lane's accuracy (%.2f "
            "ns/argument), but '%s' (%.2f ns/argument) is %.2f%% behind it, inside the %.2f%% "
            "this run can order; the probe does not order noise",
            leader->name.c_str(), leader->nsPerArgument, inseparable.front()->name.c_str(),
            inseparable.front()->nsPerArgument,
            100.0 * (inseparable.front()->nsPerArgument / leader->nsPerArgument - 1.0),
            100.0 * report.resolution);
        report.confidence = Text("CANNOT DETERMINE: %zu of %zu options in the accuracy class are "
                                 "inside the %.2f%% this run can order, which is what the canary "
                                 "and the leader's own passes measured",
                                 inseparable.size(), sameClass.size(), 100.0 * report.resolution);
        return;
    }

    report.verdict = OptionProbeVerdict::kRecommend;
    report.recommended = leader->name;

    const OptionProbeMeasurement* nearest = nullptr;

    for (const OptionProbeMeasurement* rival : sameClass)
    {
        if (rival != leader &&
            (nearest == nullptr || rival->nsPerArgument < nearest->nsPerArgument))
        {
            nearest = rival;
        }
    }

    report.reason = Text(
        "'%s' is the fastest option documented at the certified lane's accuracy: %.2f "
        "ns/argument over %d admitted of %d passes, spread %.2fx, taken at a load of %.1f%% inside "
        "the pass",
        leader->name.c_str(), leader->nsPerArgument, leader->cleanPasses,
        leader->cleanPasses + leader->disturbedPasses, leader->spread, leader->loadAtMinimum);

    report.confidence = Text("HIGH: the nearest rival in the accuracy class is at least %.2f%% "
                             "behind, beyond the %.2f%% this run can order (canary spread %.2f%%, "
                             "leader's own passes spread %.2f%%)",
                             nearest != nullptr
                                 ? 100.0 * (nearest->nsPerArgument / leader->nsPerArgument - 1.0)
                                 : 0.0,
                             100.0 * report.resolution, report.canarySpread,
                             100.0 * (leader->spread - 1.0));

    if (overall != leader && overall->bound > report.referenceBound)
    {
        report.confidence += Text(". The fastest option measured overall is '%s' at %.2f "
                                  "ns/argument, documented at %.3g, a looser accuracy class: "
                                  "faster, not faster at the same accuracy",
                                  overall->name.c_str(), overall->nsPerArgument, overall->bound);
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
    report.referenceBound = TierBound(AccuracyTier::kReference);

    const Workload work = BuildWorkload(options);
    Buffers buffers = MakeBuffers(work);
    report.orderRuns = work.runOrder.size();
    report.largestRun = work.largestRun;
    const std::vector<Option> options_ = EnumerateOptions(report.backends, report.unoffered);
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
        measurement.bound = options_[index].bound;
        measurement.maxError = walker.MaxError();
        measurement.bitIdenticalToReference = walker.BitIdentical();
        measurement.meetsBound = measurement.maxError <= measurement.bound;
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
    text += Text("  machine: %d logical processors | AVX2+FMA tier: %s\n", report.logicalProcessors,
                 report.avx2 ? "present" : "absent");
    text += "  arithmetic backends this build carries, and whether a bare a*b+c written in each\n";
    text += "  is a single rounding here:\n";

    for (const backend::BackendInfo& entry : report.backends)
    {
        text += Text("    %-14s contracts=%s\n", entry.name != nullptr ? entry.name : "(unnamed)",
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
    text += "  option              arithmetic    contracts   ns/arg   spread  clean   load%   "
            "max error      bound  verdict\n";

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        std::string verdict = "not measured";

        if (measurement.measured)
        {
            verdict = measurement.meetsBound ? "within bound" : "ABOVE BOUND";
        }

        if (measurement.bitIdenticalToReference && measurement.measured)
        {
            verdict = "reference";
        }

        if (measurement.measured)
        {
            text += Text("  %-19s %-13s %-9s %8.2f  %6.2fx  %2d/%d  %6.1f  %11.3e  %10.3g  %s\n",
                         measurement.name.c_str(), measurement.arithmetic.c_str(),
                         measurement.contracts ? "yes" : "no", measurement.nsPerArgument,
                         measurement.spread, measurement.cleanPasses,
                         measurement.cleanPasses + measurement.disturbedPasses,
                         measurement.loadAtMinimum, measurement.maxError, measurement.bound,
                         verdict.c_str());
        } else
        {
            text += Text("  %-19s %-13s %-9s %8s  %7s  %2d/%d  %6s  %11.3e  %10.3g  %s\n",
                         measurement.name.c_str(), measurement.arithmetic.c_str(),
                         measurement.contracts ? "yes" : "no", "-", "-", measurement.cleanPasses,
                         measurement.cleanPasses + measurement.disturbedPasses, "-",
                         measurement.maxError, measurement.bound, verdict.c_str());
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

    text += "\n  (load% is the canary's reading inside the pass each figure was taken in — the "
            "load that\n";
    text += "   number was taken under. The verdict column is the accuracy one: whether the "
            "option met\n";
    text += "   the bound it is held to on this workload.)\n";

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
    text += Text("  documented at the certified lane's %.3g, measured cleanly:\n",
                 report.referenceBound);

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.measured && measurement.bound <= report.referenceBound)
        {
            text += Text("    %-19s %8.2f ns/argument  spread %.2fx  %d clean pass(es)\n",
                         measurement.name.c_str(), measurement.nsPerArgument, measurement.spread,
                         measurement.cleanPasses);
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

    text += "\nnot measured by this probe, by design: the native half lane (region C only, orders "
            "0..8, results\n";
    text += "  scaled by a power of two — a different question from this workload's); the\n";
    text += "  region-A matrix-product transform (region A alone, so measuring it means composing "
            "the\n";
    text += "  rest of an evaluation, and the composition would be what got measured); the CUDA\n";
    text += "  lanes (a separate optional build that needs a device).\n";

    return text;
}

} // namespace boys
