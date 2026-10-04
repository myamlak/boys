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
//   1. The load instrument, and the comparison that does not depend on it: every
//      pass carries a reading of how much of a core this process got, and the
//      options are compared by their ratio inside one round, where a drift
//      common to the round cancels.
//
//   2. The workload: the library's real call shape, per-argument orders over a
//      log-uniform argument range, built once and shared by every option so no
//      option is measured on a different question.
//
//   3. One body per option, written against a callable so the timed checksum
//      and the untimed accuracy comparison consume the same values from the
//      same calls.

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
    case OptionPrecision::kFp32Device:
        return "fp32-device";
    }

    return "unknown";
}

const char* OptionProbeShapeName(OptionProbeShape shape) noexcept {
    switch (shape)
    {
    case OptionProbeShape::kSingle:
        return "single";
    case OptionProbeShape::kAllOrders:
        return "all-orders";
    case OptionProbeShape::kFixedN:
        return "fixed-n";
    case OptionProbeShape::kAllN:
        return "all-n";
    case OptionProbeShape::kAllNAtOrders:
        return "all-n-at-orders";
    }

    return "unknown";
}

const char* OptionProbeShapeQuestion(OptionProbeShape shape) noexcept {
    switch (shape)
    {
    case OptionProbeShape::kSingle:
        return "one call per order per argument: F_n(x) for one order at one argument";
    case OptionProbeShape::kAllOrders:
        return "one argument per call: F_0(x)..F_n(x) at that argument's own order n";
    case OptionProbeShape::kFixedN:
        return "one call per order over the array: F_n(x_i) at one fixed order n";
    case OptionProbeShape::kAllN:
        return "one call per order run: F_0(x)..F_nmax(x) for every argument of the array";
    case OptionProbeShape::kAllNAtOrders:
        return "one call over the array: F_0(x_i)..F_n[i](x_i) at each argument's own top order";
    }

    return "an unknown question";
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

/// One line with no substitutions. A separate overload rather than the variadic
/// one with an empty pack: a format that is not a literal and carries no
/// arguments is what -Wformat-security rejects, and it is right to - there is
/// nothing to check the call against.
std::string Text(const char* text) {
    return std::string(text);
}

/// One formatted line, so the report can be built without an iostream.
///
/// A call with no arguments passes its text through `%s`, for the same reason.
///
/// The buffer is sized for the longest line this report forms, which is a class's note: a class
/// holds one option per served cell of its lane, so its note enumerates them, and 144 cells at
/// roughly ten characters a name is why a kilobyte is not enough. A line longer than this would
/// be cut mid-word, and the newline that ends it with it - so the class after it would be
/// printed on the same line, as a name with no line of its own.
template <typename... Args>
std::string Text(const char* format, Args... args) {
    char buffer[8192];

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
/// runs of the same fixed work disagreed with itself.
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

/// The middle value of a set of readings, by copy and sort: the sets are short
/// enough that the sort costs nothing beside the runs they hold.
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

/// The quantile the band a pair is placed by, and the ratio it is ordered on, are
/// read at.
///
/// A lower quartile and not the minimum: under a decaying clock the minimum of a
/// run is its earliest observation, taken at the highest clock the machine will
/// reach that day, and a caller whose workload runs for hours does not meet that
/// state. A quartile and not a mean for the same reason - one disturbed round
/// must not move a figure a consumer will build on.
///
/// It is not the quantile a row's figure is formed at; that is \c kFigureQuantile,
/// and the difference between them is what keeps the figure a fair reading of the
/// row.
constexpr double kStatisticQuantile = 0.25;

/// The quantile a row's own figure is formed at: the middle of its rounds.
///
/// **Not the lower quartile, because a ratio's lower quartile is not the same
/// kind of number for the anchor as for its rivals.** Every cost column of a
/// class is the anchor's own cost scaled by that row's ratio to it, so the
/// anchor's ratio is one in every round and its lower quartile is one too —
/// the anchor is scored at the middle of its own rounds while every rival is
/// scored at the twenty-fifth percentile of its ratio, which sits below the
/// rival's middle by a fraction of its own round-to-round spread. That credit is
/// a property of which row the run anchored on, not of the options.
///
/// The middle is the quantile that treats the two alike, and it is exactly
/// reciprocal under a change of anchor — the set of ratios to the anchor is the
/// set against it, element for element — so a row's figure against the anchor
/// and the anchor's figure against that row agree on which of the two is faster.
/// The anchor, whose central ratio is one by construction, is therefore credited
/// exactly as its rivals are.
constexpr double kFigureQuantile = 0.5;

/// The number of rounds a quartile band needs before it can exist: two ends of a
/// band are two order statistics, and below four observations they are the same
/// observation twice.
constexpr std::size_t kMinimumPairedRounds = 4;

/// The precision classes this probe measures, in the order it walks them: the
/// double lane first, because that is the lane the library's own default is
/// defined in and the one every other is judged against, and the three remaining
/// host lanes after it.
///
/// The device lane's single precision is not one of them, and its book is
/// enumerated and counted beside theirs rather than left out of the report: this
/// probe has no device arm - `benchmarks/boys_option_probe.cpp`, the driver, names
/// no CUDA entry and this translation unit calls none - so no cell of that class
/// has a row here, and a class dropped from every count would be a class whose
/// refusals nothing in this report accounts for.
constexpr OptionPrecision kCellPrecisions[] = {OptionPrecision::kFp64, OptionPrecision::kFp32,
                                               OptionPrecision::kFp16, OptionPrecision::kBf16};

/// The question shapes this probe measures, in the order it walks their classes:
/// the seam's own five, in the seam's own order (\c Shape, boys/boys.hpp), so the
/// classes here and the classes `BOYS_BUILD_DEFAULT_ROWS` names are one list. A
/// shape no option answers produces no class, and the report says so instead of
/// leaving its key unmentioned; which shapes those are is a property of this
/// revision's entries rather than of the list.
constexpr OptionProbeShape kProbeShapes[] = {
    OptionProbeShape::kSingle, OptionProbeShape::kAllOrders, OptionProbeShape::kFixedN,
    OptionProbeShape::kAllN, OptionProbeShape::kAllNAtOrders};

/// The seam's own key, as the seam itself states it: the shapes its default-policy rows can be
/// written for (\c Shape, boys/boys.hpp), in its own order. This is the vocabulary the emitted
/// file's rows are drawn from, which is not the same list as the shapes this probe walks: a
/// shape the seam gains and this probe is not taught is a class the seam's key reaches and no
/// row of the emitted file can be written for, and the report names it as that rather than
/// dropping it, which is why the two lists are kept apart rather than derived one from the
/// other.
constexpr Shape kSeamShapes[] = {Shape::kSingle, Shape::kAllOrders, Shape::kFixedN, Shape::kAllN,
                                 Shape::kAllNAtOrders};

/// The shape of the question this probe's workload asks, and so the shape whose
/// reference class the default is taken from.
///
/// The workload walks the arguments one at a time, each with its own highest
/// order: the all-orders shape. Every other shape's options are handed the same
/// arguments - the runs the all-N entries take, the fixed order the fixed-N entry
/// sweeps, the per-argument tops the all-n-at-orders entry reads - so every shape
/// is measured on one workload, and the default is named from the class whose
/// question that workload is.
constexpr OptionProbeShape kWorkloadShape = OptionProbeShape::kAllOrders;

/// The library precision whose lane answers for one of the classes above.
///
/// The two half formats are one lane at one budget in the library: its entries
/// are declared once, under the fp16 name, and the bf16 entries are that engine
/// with the other format's store. Both classes are enumerated from the fp16
/// lane. The device lane's single precision is a lane of its own rather than a
/// second class of the host's float lane: its entries are the device's, and it
/// carries its own figure.
///
/// \param precision the class
///
/// \returns the lane the class's cells are enumerated from and asked about
constexpr Precision LaneOf(OptionPrecision precision) noexcept {
    switch (precision)
    {
    case OptionPrecision::kFp32:
        return Precision::kFp32;
    case OptionPrecision::kFp16:
    case OptionPrecision::kBf16:
        return Precision::kFp16;
    case OptionPrecision::kFp32Device:
        return Precision::kFp32Device;
    case OptionPrecision::kFp64:
        break;
    }

    return Precision::kFp64;
}

/// The route table a lane's cells are enumerated from: the double lane's own fit
/// table, or the single-precision engine's, which is the table the float and half
/// lanes' fits are reported in. Read from the library rather than restated here.
///
/// The device lane's cells are enumerated from the double lane's table: that is
/// the table the device lane's single-precision entries read their own region-A
/// fits from, and it is the table the library's own reading of a device
/// combination resolves routes in.
std::span<const FitRouteInfo> LaneRoutes(Precision lane) noexcept {
    return (lane == Precision::kFp32 || lane == Precision::kFp16) ? BoysFitRoutesF32()
                                                                  : BoysFitRoutes();
}

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

    /// One order across every argument: the fixed-N entry's own output, and its
    /// own argument array, so the sweep is not handed a compacted copy of the
    /// workload's - the shape it answers is one order at every argument.
    std::vector<double> fixedN;

    /// The order-major planes the all-n-at-orders entry writes: count * (nmax +
    /// 1) doubles, out[k * count + i].
    std::vector<double> planesFp64;

    /// The float lanes' own scratch: the run's arguments after the lane's own
    /// narrowing, and the planes that lane's all-N entry writes.
    std::vector<float> runXF32;
    std::vector<float> runOutF32;
};

Buffers MakeBuffers(const Workload& work) {
    Buffers buffers;
    const std::size_t run = std::max<std::size_t>(work.largestRun, 1);
    const std::size_t count = work.x.size();
    const std::size_t planes = count * static_cast<std::size_t>(work.options.nmax + 1);
    const std::size_t runPlanes = run * static_cast<std::size_t>(work.options.nmax + 1);

    buffers.runX.resize(run);
    buffers.runOut.resize(runPlanes);
    buffers.workspace.resize(BoysAllNWorkspaceSize(run));
    buffers.fixedN.resize(count);
    buffers.planesFp64.resize(planes);
    buffers.runXF32.resize(run);
    buffers.runOutF32.resize(runPlanes);
    return buffers;
}

// --- the options ------------------------------------------------------------

/// Which entries an option is, and therefore which arithmetic it runs in and
/// which call shape it measures.
enum class OptionKind {
    kBatchFp64, ///< one all-orders call per argument, fp64
    kFp64Cell, ///< one cell of the double lane's route x scheme x partition x axis space
    kSingleCell, ///< one cell of the single-precision engine's same space, at its own budget
    kGroupedFp64, ///< one all-N call per order run, fp64
    kTaggedFp64, ///< the same, with the arguments declared already sorted
    kBatchFp32, ///< one all-orders fp32 call per argument
    kBatchF16, ///< one all-orders call per argument, fp16 I/O
    kBatchBf16, ///< one all-orders call per argument, bf16 I/O
    kSingleFp64, ///< one single-order call per order per argument, fp64
    kSingleFp32, ///< the same, fp32 I/O
    kSingleHalf, ///< the same, at whichever half format the option's precision names
    kFixedNFp64, ///< one fixed-N call per order, sweeping the array, fp64
    kAllNAtOrdersFp64, ///< one all-N-at-orders call over the array, fp64
    kGroupedFp32, ///< one all-N call per order run, fp32
};

/// The question one kind answers: what the entry under it hands back for a call.
///
/// \param kind the entry an option is
///
/// \returns the shape of the question that entry is asked
constexpr OptionProbeShape ShapeOf(OptionKind kind) noexcept {
    switch (kind)
    {
    case OptionKind::kGroupedFp64:
    case OptionKind::kTaggedFp64:
    case OptionKind::kGroupedFp32:
        return OptionProbeShape::kAllN;
    case OptionKind::kSingleFp64:
    case OptionKind::kSingleFp32:
    case OptionKind::kSingleHalf:
        return OptionProbeShape::kSingle;
    case OptionKind::kFixedNFp64:
        return OptionProbeShape::kFixedN;
    case OptionKind::kAllNAtOrdersFp64:
        return OptionProbeShape::kAllNAtOrders;
    case OptionKind::kBatchFp64:
    case OptionKind::kFp64Cell:
    case OptionKind::kSingleCell:
    case OptionKind::kBatchFp32:
    case OptionKind::kBatchF16:
    case OptionKind::kBatchBf16:
        break;
    }

    return OptionProbeShape::kAllOrders;
}

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

    /// The division form the option's recurrence steps end in: the sixth axis of
    /// the policy, and the one a cell carries as a value where the others are
    /// narrowed at run time.
    DivisionForm division = kDefaultDivisionForm;

    /// The exponential the option's region-B ladders are seeded with: the seventh
    /// axis of the policy, carried as a value for the same reason the form is.
    RegionBExp regionBExp = kDefaultHostRegionBExp;

    /// The question this option answers, and the second part of the class it is
    /// ranked in: the shape ShapeOf gives its kind.
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

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

/// The entry every ratio is formed against, by default: the double lane's own
/// shape row, which is the cell that names no axis at all — the shipped
/// partition, on the arguments axis, at the shipped route and scheme. Naming it
/// here is what makes the anchor a documented choice
/// rather than the winner of the comparison it anchors.
///
/// **It is not the call a caller who names no policy gets**, and the difference
/// is the point of the anchor being a *unit* rather than a participant: the
/// library's default reads \c kDefaultFitGranularity and \c kDefaultEvalScheme,
/// which the option probe's own measurement sets, so tying the anchor to them
/// would make the probe's denominator move with the answer it is used to decide.
/// The cell a caller who names no policy gets is measured like every other, and
/// \c ProbeOptions::reference is how a reader takes the same run in that row's
/// units instead.
constexpr const char* kReferenceOptionName = "batch-fp64";

/// The option every ratio is formed against, resolved from the option book: the
/// entry named above, or the one the caller asked for, when this run measured
/// it; else the first option of the default precision; else the first option
/// measured.
///
/// Resolved before any round is timed, so the anchor cannot be chosen from the
/// figures to suit the answer. It is one row's own cost that every column is
/// scaled by and no row's score, so which row it is changes the units the
/// absolute columns are printed in and not the rank any two rows take.
std::size_t ReferenceIndex(const std::vector<Option>& options, const std::string& requested) {
    // A caller's own anchor first, and only when the run measured that name:
    // anchoring on a row this run did not measure would leave every ratio without
    // a denominator, and the run says so and falls back rather than measuring
    // nothing.
    if (!requested.empty())
    {
        for (std::size_t index = 0; index < options.size(); ++index)
        {
            if (options[index].name == requested)
            {
                return index;
            }
        }
    }

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
/// round by construction, so it would contribute a band of zero.
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

/// The base the half lanes' rows are judged against: the figure the library
/// publishes for the lane. |F_n(x)| <= 1, so the largest half-ULP a return can
/// carry is 2^-11 for a binary16 return and 2^-9 for a bfloat16 one.
double HalfLaneBase() noexcept
{
    return BoysLaneContracts()[static_cast<std::size_t>(Precision::kFp16)].bound;
}

/// The half-ULP representation term of each half format, as above.
constexpr double kHalfFormatTerm = 0x1p-11;
constexpr double kBf16FormatTerm = 0x1p-9;

/// The term a half format's row carries beside the base its lane publishes: half
/// of the last representable digit of the returned value, which that lane's own
/// source sentence claims and the accessor's figure does not add. Zero for every
/// other class.
///
/// \param precision the class
///
/// \returns the term, in the units the lane's figure is in
constexpr double HalfFormatTermOf(OptionPrecision precision) noexcept {
    switch (precision)
    {
    case OptionPrecision::kFp16:
        return kHalfFormatTerm;
    case OptionPrecision::kBf16:
        return kBf16FormatTerm;
    case OptionPrecision::kFp32:
    case OptionPrecision::kFp32Device:
    case OptionPrecision::kFp64:
        break;
    }

    return 0.0;
}

/// The bar a row that calls its class's own default-policy entry is judged
/// against: the figure that policy publishes, asked of the library by the class
/// rather than rebuilt from the axes the call resolves to.
///
/// The batch entries of the float and the two half formats take a policy that
/// defaults to their class's own row of this build's seam — at this revision the
/// narrow partition in a form the row's own figure carries a term for — so a bar
/// written out as an axis tuple is a figure no combination of that policy
/// publishes, and it parts company with the policy the moment the seam moves.
/// \c DefaultGuarantee asks through the policy's own axes and its own division
/// form, which is what makes the bar and the policy one number.
///
/// \param precision the class's precision: the float lane, or one of the two half
///                  formats, which are one lane at one budget
///
/// \returns the figure, the half formats' own term beside it
double DefaultPolicyBound(OptionPrecision precision) {
    const bool floatLane = precision == OptionPrecision::kFp32;
    const double formatTerm = HalfFormatTermOf(precision);

    // The two half formats are one lane and one row of the table: both entries
    // resolve their policy on the fp16 row.
    const AccuracyFigure figure =
        floatLane ? DefaultGuarantee<Precision::kFp32, Shape::kAllOrders>()
                  : DefaultGuarantee<Precision::kFp16, Shape::kAllOrders>();

    if (figure.available)
    {
        return figure.value + formatTerm;
    }

    // Unreachable from a build whose seam names a row the library carries, which
    // is the only shape of table a class with an entry can have. The fallback is
    // the lane's own base, the figure every entry of this revision is built at,
    // rather than a zero: a zero bar
    // would report every row of the class as one that missed a bound it was never
    // given.
    const LaneContractInfo& lane = BoysLaneContracts()[static_cast<std::size_t>(
        floatLane ? Precision::kFp32 : Precision::kFp16)];

    return kBoysFullAccuracyMultiplier * lane.bound + formatTerm;
}

/// The figure one row of one precision is judged against, read from the
/// library.
///
/// The lanes document different figures and none substitutes for another: the
/// double lane's is the error its entries can reach over the
/// whole domain, the single-precision lane's is its own row of
/// \c BoysLaneContracts, the half lanes' is the
/// engine budget's base plus the largest half of a
/// representable digit their returns can carry, and the device lane's is its own
/// row of that table plus the term its row states beside
/// the base. The lane is an argument here because a row judged against another
/// lane's figure is worse than no row.
///
/// The form is an argument for the same reason: the plain reciprocal rounds once
/// more per step on the lanes whose rows publish a term for it, so a form-aware
/// figure asked at another form is a bar the row's own arithmetic is not held
/// to. Every lane reads the form, and the lanes that carry no term are unchanged
/// by it.
///
/// \param precision   the class the row is ranked in
/// \param route       the row's fit route
/// \param scheme      the row's evaluation scheme
/// \param axis        the row's packing axis
/// \param granularity the row's partition
/// \param form        the division form the row's arithmetic runs in
///
/// \returns the figure, per value of F_n
double LaneCellBound(OptionPrecision precision,
                     FitRoute route,
                     EvalScheme scheme,
                     PackAxis axis,
                     FitGranularity granularity,
                     DivisionForm form) {
    const double multiplier = kBoysFullAccuracyMultiplier;

    switch (precision)
    {
    case OptionPrecision::kFp32:
    {
        const AccuracyFigure figure =
            BoysAccuracyGuaranteed(Precision::kFp32, route, scheme, axis, granularity, form);

        if (figure.available)
        {
            return figure.value;
        }

        // Unreachable from this probe's own book, which enumerates a cell only
        // where the same call answered a figure; the fallback is the same
        // arithmetic that answer is formed from, so a cell that somehow reached
        // here is judged at its lane's figure rather than at a zero.
        return multiplier * BoysLaneContracts()[static_cast<std::size_t>(Precision::kFp32)].bound;
    }

    case OptionPrecision::kFp32Device:
    {
        const AccuracyFigure figure =
            BoysAccuracyGuaranteed(Precision::kFp32Device, route, scheme, axis, granularity, form);

        if (figure.available)
        {
            return figure.value;
        }

        // Unreachable for the same reason the float lane's fallback is: a cell is
        // enumerated only where the call above answers a figure, and no cell of
        // this class becomes a row at all. The fallback is the device lane's own
        // row and not the double lane's figure, which is what the arm below
        // returns: a row of this lane judged against another lane's arithmetic
        // would state a promise no device entry makes.
        const LaneContractInfo& row =
            BoysLaneContracts()[static_cast<std::size_t>(Precision::kFp32Device)];

        return multiplier * row.bound + row.additive;
    }

    case OptionPrecision::kFp16:
    case OptionPrecision::kBf16:
    {
        // The half lanes run the single-precision engine and are one lane at one
        // budget, so the figure is that lane's own row at the form the row
        // evaluates in, plus the format's half digit, which that lane's sentence
        // adds beside the base and the accessor's figure does not carry. Reading
        // the row rather than rebuilding it is what keeps the plain reciprocal's
        // own term in the figure of a row that runs it.
        const AccuracyFigure figure =
            BoysAccuracyGuaranteed(Precision::kFp16, route, scheme, axis, granularity, form);

        return (figure.available ? figure.value : multiplier * HalfLaneBase()) +
               HalfFormatTermOf(precision);
    }

    case OptionPrecision::kFp64:
    {
        // The double lane's row, read through the same accessor every other class
        // reads: its figure is the lane's own documented one, and a cell of it is
        // judged at the lane's figure rather than at the batch entry's alone.
        const AccuracyFigure figure =
            BoysAccuracyGuaranteed(Precision::kFp64, route, scheme, axis, granularity, form);

        if (figure.available)
        {
            return figure.value;
        }

        break;
    }
    }

    // Unreachable from this probe's own book, which enumerates a cell only where
    // the calls above answered a figure, and kept for the reason the float lane's
    // fallback is: the lane's own row rather than a zero. The multiplication by
    // the multiplier keeps the expression the report prints this figure's
    // provenance with, and the multiplier this revision carries is one.
    return multiplier * BoysLaneContracts()[static_cast<std::size_t>(Precision::kFp64)].bound;
}

/// The name a cell's option is printed under, in one grammar for all of them.
///
/// The name states the cell's own axes and omits the defaults, so a defaulted
/// route, scheme, partition, axis and division form leave no segment behind. The
/// partition takes the first segment — `batch` for the shipped one, and `narrow`
/// or `uniform` for either of the other two — and the precision closes every
/// name, because a name is only ever read inside its class.
///
/// The division form is named only where it is not the one the library's default
/// policy runs, and that is the one reading of this axis under which the names
/// above it keep meaning what they say: the shape rows carry the name this
/// grammar gives the default form's cell, and the entries those rows call are the
/// default policy's, so the form left unmarked is the form they really divide in.
/// The other two members are named so that no two cells of one combination can
/// print the same name — without the segment a reader would take the plain
/// reciprocal's row for the default's, which is exactly the confusion the two
/// forms' arithmetic differs by.
///
/// The region-B exponential is named on the same reading and for the same reason:
/// the member left unmarked is the host default, which is the member the entries
/// the shape rows call really seed with, and the other member is named so that a
/// cell of one combination cannot print the other's name.
///
/// The precision in the closing segment is the class the row is ranked in, not
/// the lane its cells were enumerated from: the two half formats are one lane and
/// two classes, so the same cell of that lane is named once for each format it is
/// measured in.
std::string CellName(OptionPrecision precision,
                     FitGranularity granularity,
                     PackAxis pack,
                     FitRoute route,
                     EvalScheme scheme,
                     DivisionForm division,
                     RegionBExp regionBExp) {
    std::string name = "batch";

    if (granularity == FitGranularity::kUniform)
    {
        name = "uniform";
    } else if (granularity == FitGranularity::kNarrow)
    {
        name = "narrow";
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

    if (division != kDefaultDivisionForm)
    {
        // The library's own spelling of the form, so the segment a report prints
        // is the name the library answers for the enumerator and not a second
        // way of writing it here.
        name += "-";
        name += DivisionFormName(division);
    }

    if (regionBExp != kDefaultHostRegionBExp)
    {
        // The library's own spelling of the member, on the same reading as the
        // form's segment above.
        name += "-";
        name += RegionBExpName(regionBExp);
    }

    return name + "-" + PrecisionName(precision);
}

/// The name this precision's own shape row is printed under.
///
/// Every lane has one row for the cell that names no axis at all — the shipped
/// partition on the arguments axis at the shipped route and scheme — and on the
/// double, float and device lanes that row is named
/// in the cells' own grammar, so the cell and the row carry one name. The half
/// lanes' rows are named for the format boundary they cross, \c f16-io and
/// \c bf16-io: those entries are the single-precision engine with a store on
/// either side, and they take no policy argument for a caller to name.
///
/// The name is read by the coverage and by the option book, so a served cell is
/// measured under the name the report prints for it.
///
/// \param precision the class
///
/// \returns the row's name, which is also the name of the cell it carries
std::string LaneShapeName(OptionPrecision precision) {
    switch (precision)
    {
    case OptionPrecision::kFp32:
        return "batch-fp32";
    case OptionPrecision::kFp32Device:
        return "batch-fp32-device";
    case OptionPrecision::kFp16:
        return "f16-io";
    case OptionPrecision::kBf16:
        return "bf16-io";
    case OptionPrecision::kFp64:
        break;
    }

    return "batch-fp64";
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

/// The whole option space this build defines at one precision, cell by cell,
/// with the library's reason for every cell it does not serve.
///
/// The space is the product the library reports **for that precision**: the
/// routes of the lane's own fit table, the evaluation schemes, the partitions of
/// the fitted regions, the packing axes, the division forms and the region-B
/// exponentials. It is walked
/// rather than listed, so a member a later change adds is enumerated and a cell
/// the library refuses is counted as the
/// unbuilt work it is instead of being absent from the report.
///
/// The lane is part of the question and not a filter applied afterwards: the
/// single-precision engine's fits are its own table, and a cell the double lane
/// refuses can be one the engine serves.
///
/// **Each cell's state is the library's answer to the cell's own combination**,
/// asked through \c BoysAccuracyGuaranteed: a cell is served exactly when that
/// call answers a figure for it, and a refused cell carries the reason that call
/// gives, in its own words. Nothing here restates a rule about routes, axes or
/// partitions: the library's answer is the only copy.
///
/// \param precision the class this book is the option space of
///
/// \returns every cell of that space, served ones and refused ones alike
std::vector<OptionProbeCell> EnumerateCells(OptionPrecision precision) {
    std::vector<OptionProbeCell> cells;
    const Precision lane = LaneOf(precision);
    const std::span<const FitRouteInfo> routes = LaneRoutes(lane);

    // A route has one row per region it supplies, so it is taken once here.
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
                    for (const DivisionFormInfo& form : BoysDivisionForms())
                    {
                        for (const RegionBExpInfo& exp : BoysRegionBExps())
                        {
                            OptionProbeCell cell;
                            // The form and the exponential are part of the test
                            // and not only of the name: the shape row's name is
                            // the one this grammar gives the *default* members'
                            // cell, and the entry that row measures runs those
                            // members. A cell of the same combination at another
                            // member is its own cell and keeps its own name - two
                            // cells printing one name is the collision the
                            // segments exist to stop.
                            const bool shapesRow =
                                partition.granularity == FitGranularity::kCoarsest &&
                                axis.axis == PackAxis::kArguments &&
                                route.route == FitRoute::kChebyshev &&
                                scheme.scheme == EvalScheme::kSplitClenshaw &&
                                form.form == kDefaultDivisionForm &&
                                exp.exp == kDefaultHostRegionBExp;
                            cell.name = shapesRow ? LaneShapeName(precision)
                                                  : CellName(precision,
                                                             partition.granularity,
                                                             axis.axis,
                                                             route.route,
                                                             scheme.scheme,
                                                             form.form,
                                                             exp.exp);
                            cell.precision = precision;
                            cell.lane = lane;
                            cell.route = route.route;
                            cell.scheme = scheme.scheme;
                            cell.granularity = partition.granularity;
                            cell.pack = axis.axis;
                            cell.division = form.form;
                            cell.regionBExp = exp.exp;

                            // Neither the form nor the exponential is asked
                            // about: they are how a step divides and how a seed
                            // is evaluated, not which entry runs, so the
                            // library's carriage answer is the same for every
                            // member of either axis and the cell is served or
                            // refused on its other five.
                            const AccuracyFigure carriage =
                                BoysAccuracyGuaranteed(lane, cell.route, cell.scheme, cell.pack,
                                                       cell.granularity);

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
    }

    return cells;
}

/// Every option this build offers, enumerated from the library.
///
/// A cell the library serves becomes an option, named by the cell's own
/// grammar, and the entry that evaluates it is chosen by the cell's axes: the
/// run-time selector carries the shipped partition on the arguments axis at
/// any route and scheme, the across-orders lane carries every route and scheme,
/// and the narrow partition is reachable only as a policy.
///
/// The division form of the rows named here is the library's default, and it is
/// written as that constant rather than as an enumerator because it is a
/// property of the entries the rows call: \c BoysAllOrders, \c BoysAllN and the
/// narrow lanes' entries take no policy and divide in the default form, so a
/// row that recorded another member would be claiming an arithmetic no entry it
/// calls runs. Every other member of the axis is reached through the cell loop
/// below, which is where a cell's own form becomes an instantiation.
///
/// The region-B exponential of those rows is derived the same way and for the
/// same reason, from the default policy of the precision the row runs rather
/// than from a constant: the row's entry takes no policy, so the member it
/// seeds its region-B ladders with is the default policy's, and the row records
/// what its own implementation does. It is written per precision because the
/// default policy is per precision - the half lanes run the float engine, so
/// their rows carry the float policy's member.
///
/// The entries whose call shape is not one of the axes are named here,
/// because a shape is a function and no table of them exists to read: the
/// all-N per-run shape and its sorted overload, and the narrower lanes. Each
/// option's arithmetic is resolved against backend::BoysBackends(), so an option
/// the table cannot give an arithmetic to is reported as not offered, and the
/// half-precision lanes are behind the BoysFp16 seam that declares them, so a
/// build whose seam is closed reports them through notCarried.
///
/// The cells handed in are the ones this machine measures, and no others: the
/// device lane's book is enumerated beside this one and never reaches here,
/// because a row's body is chosen by the row's axes, so a row of that class
/// would be run by another lane's arithmetic and reported under the device
/// lane's name.
std::vector<Option> EnumerateOptions(std::span<const backend::BackendInfo> table,
                                     std::span<const FitGranularityInfo> partitions,
                                     const std::vector<OptionProbeCell>& cells,
                                     std::vector<std::string>& unoffered,
                                     std::vector<std::string>& notCarried) {
    std::vector<Option> options;

    const backend::BackendInfo* fp64 = ResolveArithmetic(table, true);
    const backend::BackendInfo* fp32 = ResolveArithmetic(table, false);

    const auto append = [&](std::string name,
                            OptionKind kind,
                            OptionPrecision precision,
                            FitRoute route,
                            EvalScheme scheme,
                            FitGranularity granularity,
                            PackAxis pack,
                            DivisionForm division,
                            RegionBExp regionBExp,
                            const backend::BackendInfo* arithmetic,
                            double bound) {
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
        option.division = division;
        option.regionBExp = regionBExp;
        option.shape = ShapeOf(kind);
        option.bound = bound;
        options.push_back(std::move(option));
    };

    // The figure the double lane's own cell rows are judged against, read from the
    // library's lane table through the same accessor every other class reads.
    const double fp64Bound = LaneCellBound(OptionPrecision::kFp64, FitRoute::kChebyshev,
                                           EvalScheme::kSplitClenshaw, PackAxis::kArguments,
                                           FitGranularity::kCoarsest, kDefaultDivisionForm);

    append(LaneShapeName(OptionPrecision::kFp64),
           OptionKind::kBatchFp64,
           OptionPrecision::kFp64,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp64::kRegionBExp,
           fp64,
           fp64Bound);
    append("grouped-fp64",
           OptionKind::kGroupedFp64,
           OptionPrecision::kFp64,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp64::kRegionBExp,
           fp64,
           fp64Bound);
    append("tagged-fp64",
           OptionKind::kTaggedFp64,
           OptionPrecision::kFp64,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp64::kRegionBExp,
           fp64,
           fp64Bound);
    append(LaneShapeName(OptionPrecision::kFp32),
           OptionKind::kBatchFp32,
           OptionPrecision::kFp32,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp32::kRegionBExp,
           fp32,
           DefaultPolicyBound(OptionPrecision::kFp32));

    // The rows for the shapes the seam names and no cell of the axes carries.
    //
    // A shape row's axes are the entry's own and not a cell's: every row below
    // calls its entry with no policy argument, so what runs is the class's own row
    // of the seam in force (`DefaultPolicy`, boys/boys.hpp), and the axes written
    // here are read from that row rather than chosen. A row whose axes were
    // written out by hand would state a combination its own body does not
    // instantiate the moment the seam moved, and this report names the arithmetic
    // each row ran.
    //
    // One row per class: the library carries one entry for each of these shapes at
    // each lane below, so there is no alternative to rank - the class ranks its
    // single option, which the seam's own header writes as a choice and not as a
    // comparison's winner.
    {
        using Fp64SingleRow = DefaultPolicy<Precision::kFp64, Shape::kSingle>;

        append("single-fp64", OptionKind::kSingleFp64, OptionPrecision::kFp64,
               Fp64SingleRow::kRoute, Fp64SingleRow::kScheme, Fp64SingleRow::kGranularity,
               Fp64SingleRow::kPack, Fp64SingleRow::kDivision, Fp64SingleRow::kRegionBExp, fp64,
               fp64Bound);
    }

    {
        using Fp64FixedNRow = DefaultPolicy<Precision::kFp64, Shape::kFixedN>;

        append("fixed-n-fp64", OptionKind::kFixedNFp64, OptionPrecision::kFp64,
               Fp64FixedNRow::kRoute, Fp64FixedNRow::kScheme, Fp64FixedNRow::kGranularity,
               Fp64FixedNRow::kPack, Fp64FixedNRow::kDivision, Fp64FixedNRow::kRegionBExp, fp64,
               fp64Bound);
    }

    {
        using Fp64AtOrdersRow = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>;

        append("all-n-at-orders-fp64", OptionKind::kAllNAtOrdersFp64, OptionPrecision::kFp64,
               Fp64AtOrdersRow::kRoute, Fp64AtOrdersRow::kScheme, Fp64AtOrdersRow::kGranularity,
               Fp64AtOrdersRow::kPack, Fp64AtOrdersRow::kDivision, Fp64AtOrdersRow::kRegionBExp,
               fp64, fp64Bound);
    }

    {
        using Fp32SingleRow = DefaultPolicy<Precision::kFp32, Shape::kSingle>;

        append("single-fp32", OptionKind::kSingleFp32, OptionPrecision::kFp32,
               Fp32SingleRow::kRoute, Fp32SingleRow::kScheme, Fp32SingleRow::kGranularity,
               Fp32SingleRow::kPack, Fp32SingleRow::kDivision, Fp32SingleRow::kRegionBExp, fp32,
               DefaultPolicyBound(OptionPrecision::kFp32));
    }

    {
        using Fp32AllNRow = DefaultPolicy<Precision::kFp32, Shape::kAllN>;

        append("grouped-fp32", OptionKind::kGroupedFp32, OptionPrecision::kFp32,
               Fp32AllNRow::kRoute, Fp32AllNRow::kScheme, Fp32AllNRow::kGranularity,
               Fp32AllNRow::kPack, Fp32AllNRow::kDivision, Fp32AllNRow::kRegionBExp, fp32,
               DefaultPolicyBound(OptionPrecision::kFp32));
    }

#if BoysFp16
    // The half lanes run the fp32 engine, so their arithmetic is the fp32 one.
    append(LaneShapeName(OptionPrecision::kFp16),
           OptionKind::kBatchF16,
           OptionPrecision::kFp16,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp32::kRegionBExp,
           fp32,
           DefaultPolicyBound(OptionPrecision::kFp16));
    append(LaneShapeName(OptionPrecision::kBf16),
           OptionKind::kBatchBf16,
           OptionPrecision::kBf16,
           FitRoute::kChebyshev,
           EvalScheme::kSplitClenshaw,
           FitGranularity::kCoarsest,
           PackAxis::kArguments,
           kDefaultDivisionForm,
           DefaultPolicyFp32::kRegionBExp,
           fp32,
           DefaultPolicyBound(OptionPrecision::kBf16));

    // The half lane's single-order row, one per format and for the reason the
    // single-precision rows above carry: the shape's entry is the class's own, so
    // there is nothing to rank it against. The two half formats are one lane and
    // two classes, and each is measured under the format it returns.
    {
        using HalfSingleRow = DefaultPolicy<Precision::kFp16, Shape::kSingle>;

        append("single-fp16", OptionKind::kSingleHalf, OptionPrecision::kFp16,
               HalfSingleRow::kRoute, HalfSingleRow::kScheme, HalfSingleRow::kGranularity,
               HalfSingleRow::kPack, HalfSingleRow::kDivision, HalfSingleRow::kRegionBExp, fp32,
               DefaultPolicyBound(OptionPrecision::kFp16));
        append("single-bf16", OptionKind::kSingleHalf, OptionPrecision::kBf16,
               HalfSingleRow::kRoute, HalfSingleRow::kScheme, HalfSingleRow::kGranularity,
               HalfSingleRow::kPack, HalfSingleRow::kDivision, HalfSingleRow::kRegionBExp, fp32,
               DefaultPolicyBound(OptionPrecision::kBf16));
    }

    // A build that carries both lanes has nothing to report as not carried, and
    // the register stays a parameter either way so its caller reads one list in
    // both builds.
    (void)notCarried;
#else
    // The seam is closed in this build, so the two lanes are not options it can
    // serve. They are named as not carried rather than dropped from both lists: a
    // caller asking for one is told the build has no such lane, not that the name
    // is no option of this library.
    notCarried.push_back(LaneShapeName(OptionPrecision::kFp16));
    notCarried.push_back(LaneShapeName(OptionPrecision::kBf16));
#endif

    // Every served cell of the classes handed in becomes an option, under the
    // cell's own name, except the one cell per precision the shape rows above
    // already carry: the shipped partition on the arguments axis at the shipped
    // route and scheme. On the narrower lanes that cell is an entry of its own; on
    // the double lane the walk adds every cell of the product, its own shape row
    // excepted.
    for (const OptionProbeCell& cell : cells)
    {
        if (!cell.served)
        {
            continue;
        }

        // The form and the exponential are part of the test for the same reason
        // they are part of the shape row's: the row this skips is the one the
        // named rows above carry at the default members, and it carries it at one
        // member of each axis only. Skipping the cell without asking about them
        // would drop the plain and exact members of that combination, and the
        // other member of the exponential axis at every one of the three forms,
        // out of the book entirely - a cell of the product with no row and no
        // reason, which is the omission the coverage exists to prevent.
        if (cell.granularity == FitGranularity::kCoarsest && cell.pack == PackAxis::kArguments &&
            cell.route == FitRoute::kChebyshev && cell.scheme == EvalScheme::kSplitClenshaw &&
            cell.division == kDefaultDivisionForm && cell.regionBExp == kDefaultHostRegionBExp)
        {
            continue;
        }

        if (cell.precision == OptionPrecision::kFp64)
        {
            append(cell.name,
                   OptionKind::kFp64Cell,
                   cell.precision,
                   cell.route,
                   cell.scheme,
                   cell.granularity,
                   cell.pack,
                   cell.division,
                   cell.regionBExp,
                   fp64,
                   0.0);
            continue;
        }

        append(cell.name,
               OptionKind::kSingleCell,
               cell.precision,
               cell.route,
               cell.scheme,
               cell.granularity,
               cell.pack,
               cell.division,
               cell.regionBExp,
               fp32,
               LaneCellBound(cell.precision, cell.route, cell.scheme, cell.pack, cell.granularity,
                             cell.division));
    }

    // The bound of every cell option is the figure the library documents for the
    // entry the cell reaches, read through \c BoysAccuracyGuaranteed at the cell's
    // own axes. The partition's own figures are recorded beside it, with the
    // interval they hold on.
    //
    // The row is the option's own, found by the value it names rather than by
    // position in the table: every partition of that table certifies its own fits
    // at a figure for the interval they cover, and a lookup that carried one
    // partition's figures and left the others at zero would report a cell whose
    // own tables have a figure as one whose tables have none.
    for (Option& option : options)
    {
        if (option.kind != OptionKind::kFp64Cell)
        {
            continue;
        }

        option.bound = LaneCellBound(option.precision, option.route, option.scheme, option.pack,
                                     option.granularity, option.division);

        for (const FitGranularityInfo& partition : partitions)
        {
            if (partition.granularity == option.granularity)
            {
                option.ownBound = partition.bound;
                option.ownLo = partition.lo;
                option.ownHi = partition.hi;
            }
        }
    }

    return options;
}

// --- one option's values ----------------------------------------------------

/// One cell's own entry, as the cell's routes, schemes, partition,
/// packing axis and division form select it.
///
/// A cell that is not the default policy's own shape exists only as an
/// instantiation: the partition, the packing axis, the division form and the
/// region-B exponential are template arguments of the policy and have no
/// run-time entry, so a consumer
/// reaches them the way this does, by naming them. Every entry is built at the
/// library's full-accuracy multiplier, which is the one accuracy this revision
/// carries, and the dispatch is the same cross the library's own accuracy
/// gate measures, so a cell this probe reports as served is a cell some entry of
/// this build really runs.
///
/// \tparam kDivision the form the cell's recurrence steps divide in
/// \tparam kExp      the exponential the cell's region-B seed is evaluated with
template <FitRoute kRoute,
          EvalScheme kScheme,
          PackAxis kPack,
          FitGranularity kGran,
          DivisionForm kDivision,
          RegionBExp kExp>
void CellEntry(int nmax, double x, double* out) noexcept {
    constexpr BoysBudget kBudget = BoysBudget::kFloat;
    using Policy = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran, kDivision, kExp>;

    BoysAllOrders<Policy>(nmax, x, out);
}

/// One uniform-partition cell of the option space, at either route and on either
/// packing axis.
///
/// Both routes are served and the table stores both members: a Chebyshev fit per
/// order per interval and, beside it, one numerator/denominator pair per
/// interval. Both packing axes are served as well - the across-orders packed
/// lane carries the Chebyshev member as the arguments axis does, and hands the
/// rational member's rows to the scalar orders lane, which has no stride to step
/// - so both the route and the axis are read off the cell rather than assumed.
///
/// Every cell of this partition is served and evaluated through \c CellEntry, the
/// one place this file names the policy's multiplier, so a cell named here and a
/// cell of any other partition are built by the same reading.
///
/// A cell outside the partition's own lattice is not one any book of this probe
/// can name - the enumeration asks \c BoysAccuracyGuaranteed before it registers
/// a cell - and it stops rather than being evaluated under a combination it did
/// not name.
/// That is the failure this arm exists to make impossible rather than to answer:
/// another partition's fits under the uniform name is what the derived
/// partitions' own lattice did with this value before it was given an arm.
///
/// \tparam kDivision the form the cell's recurrence steps divide in
/// \tparam kExp      the exponential the cell's region-B seed is evaluated with
template <DivisionForm kDivision, RegionBExp kExp>
void CellUniform(FitRoute route,
                 EvalScheme scheme,
                 PackAxis pack,
                 int nmax,
                 double x,
                 double* out) noexcept {
    // The rational member is served like every other cell's: the library
    // derives the member over the grid's own intervals, stores its pairs and
    // reads them, so there is no second path here and no arm to stop on. It was
    // the one cell of this partition a build did not answer, and this file
    // stopped rather than evaluating it under a combination it had not named -
    // the failure that arm existed to make impossible. **The arm is gone because
    // the state it guarded is gone.**
    //
    // The axis the book names is the axis the cell is measured at: the across-orders
    // packed lane carries the uniform grid as well, so a cell named on either axis
    // is evaluated through a policy naming that axis and never through the other.
    // The route reaches the entry the same way, through the policy: the two
    // members are stored in one partition and the entry dispatches on the route
    // it is handed rather than on a second branch here.
    //
    // The policy is \c CellEntry's and not a second dispatch written here, so both
    // members of this partition are built by the one reading of the multiplier.
    const auto with_axis = [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack>() {
        CellEntry<kRoute, kScheme, kPack, FitGranularity::kUniform, kDivision, kExp>(nmax, x, out);
    };

    const auto with_pack = [&]<FitRoute kRoute, EvalScheme kScheme>() {
        if (pack == PackAxis::kOrders)
        {
            with_axis.template operator()<kRoute, kScheme, PackAxis::kOrders>();
        } else
        {
            with_axis.template operator()<kRoute, kScheme, PackAxis::kArguments>();
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

/// The same, at one division form named as a compile-time value: the axes other
/// than the form narrow at run time inside it, and every one of them reaches the
/// instantiation it names, so a cell is measured through its own policy and never
/// through another cell's.
///
/// The uniform partition is dispatched before the lattice rather than inside it,
/// and that is the shape the partition's own coverage has: the three axes the
/// lattice crosses are the three this partition is served on one member of each,
/// so a lattice arm for it would have to
/// instantiate that partition's combinations twice over to answer cells that are
/// never any. The
/// route and the axis the lattice would have crossed are not among them, and
/// \c CellUniform carries both.
///
/// \tparam kDivision the form this call's cells divide in
/// \tparam kExp      the exponential this call's cells seed their region-B ladders with
template <DivisionForm kDivision, RegionBExp kExp>
void CellFormPolicy(FitRoute route,
                    EvalScheme scheme,
                    PackAxis pack,
                    FitGranularity granularity,
                    int nmax,
                    double x,
                    double* out) noexcept {
    if (granularity == FitGranularity::kUniform)
    {
        CellUniform<kDivision, kExp>(route, scheme, pack, nmax, x, out);

        return;
    }

    const auto with_partition = [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack>() {
        if (granularity == FitGranularity::kNarrow)
        {
            CellEntry<kRoute, kScheme, kPack, FitGranularity::kNarrow, kDivision, kExp>(nmax, x,
                                                                                         out);
        } else
        {
            CellEntry<kRoute, kScheme, kPack, FitGranularity::kCoarsest, kDivision, kExp>(nmax, x,
                                                                                          out);
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

/// The same, at one region-B exponential named as a compile-time value and the
/// division form dispatched at run time inside it: the form is a template argument
/// of the policy like the partition and the packing axis, so a cell that runs one
/// of the non-default forms needs an instantiation of its own to run, and the three
/// cases below are that instantiation each, at the member this call names.
///
/// The switch names its three enumerators and leaves no default label: a fourth
/// member of the axis is a decision to make here rather than arithmetic to pick
/// silently, and the build says so at this line instead of a cell quietly
/// measuring another form's steps.
///
/// The form is a value of the cell and never a filter: every one of the three is
/// served on every combination of the other axes, so no case here refuses.
///
/// \tparam kExp the exponential this call's cells seed their region-B ladders with
template <RegionBExp kExp>
void CellExpPolicy(FitRoute route,
                   EvalScheme scheme,
                   PackAxis pack,
                   FitGranularity granularity,
                   DivisionForm division,
                   int nmax,
                   double x,
                   double* out) noexcept {
    switch (division)
    {
    case DivisionForm::kExactDivision:
        CellFormPolicy<DivisionForm::kExactDivision, kExp>(route, scheme, pack, granularity, nmax,
                                                           x, out);
        return;
    case DivisionForm::kPlainReciprocal:
        CellFormPolicy<DivisionForm::kPlainReciprocal, kExp>(route, scheme, pack, granularity,
                                                             nmax, x, out);
        return;
    case DivisionForm::kRefinedReciprocal:
        CellFormPolicy<DivisionForm::kRefinedReciprocal, kExp>(route, scheme, pack, granularity,
                                                               nmax, x, out);
        return;
    }
}

/// The same, for a cell named at run time, with the region-B exponential
/// dispatched here: the member is a template argument of the policy like the
/// division form and the partition, so a cell that runs the non-default member
/// needs an instantiation of its own to run, and the two cases below are that
/// instantiation each.
///
/// The switch names its two enumerators and leaves no default label, on the same
/// reading as the form's: a third member of this axis is a decision to make here
/// rather than arithmetic to pick silently, and the build says so at this line
/// instead of a cell quietly seeding its ladder with the other member.
///
/// The member is a value of the cell and never a filter: both are served on every
/// combination of the other axes, so no case here refuses.
void CellPolicy(FitRoute route,
                EvalScheme scheme,
                PackAxis pack,
                FitGranularity granularity,
                DivisionForm division,
                RegionBExp regionBExp,
                int nmax,
                double x,
                double* out) noexcept {
    switch (regionBExp)
    {
    case RegionBExp::kAccurate:
        CellExpPolicy<RegionBExp::kAccurate>(route, scheme, pack, granularity, division, nmax, x,
                                             out);
        return;
    case RegionBExp::kFast:
        CellExpPolicy<RegionBExp::kFast>(route, scheme, pack, granularity, division, nmax, x, out);
        return;
    }
}

/// The engine budget a precision class runs at.
///
/// It is a property of the class and not a choice inside it: the single-precision
/// lane's entries are compiled at \c BoysBudget::kFloat and the half lanes' at
/// the tighter \c BoysBudget::kFp16, which is the axis that makes their bound 1e-7
/// rather than the float lane's 1.5e-7 and the reason the two are not one class.
///
/// \param precision the class
///
/// \returns the budget its entries are built at
constexpr BoysBudget BudgetOf(OptionPrecision precision) noexcept {
    return (precision == OptionPrecision::kFp16 || precision == OptionPrecision::kBf16)
               ? BoysBudget::kFp16
               : BoysBudget::kFloat;
}

/// One single-precision cell's own entry: the same shape as the
/// double lane's \c CellEntry, with the budget as a further template argument a
/// consumer names, because it is a template argument of the engine's entries
/// too.
///
/// \tparam kDivision the form the cell's recurrence steps divide in
/// \tparam kExp      the member of the region-B exponential axis the cell's
///                   ladders seed with
template <BoysBudget kBudget,
          FitRoute kRoute,
          EvalScheme kScheme,
          PackAxis kPack,
          FitGranularity kGran,
          DivisionForm kDivision,
          RegionBExp kExp>
void CellEntrySingle(int nmax, float x, float* out) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran, kDivision, kExp>;

    BoysAllOrdersF32<Policy>(nmax, x, out);
}

/// A single-precision uniform-partition cell.
///
/// This arm refuses the rational route, and it refuses it for this lane's own
/// reasons rather than the double
/// lane's arm's. No book of this probe names a cell of that route,
/// because \c CarriesSingle answers the whole partition unserved on this lane.
/// The rational route has a member over the *float* grid of its own: one
/// numerator/denominator pair per interval, stored in this lane's width and read
/// through the route dispatch in UniformOrderAtF32. A cell naming it is
/// evaluated through that dispatch like any other cell of the partition, on both
/// budgets; the route reaches the policy rather than being
/// read off the cell here, which is what keeps a cell named on one family from
/// being answered by another's fits.
///
/// The budget is an axis of its own on this side and not a route: the
/// fp16 class runs the same engine at its own region-B boundary, so a cell of
/// either class is evaluated through a policy naming its own budget and never
/// through the other's.
///
/// \tparam kDivision the form the cell's recurrence steps divide in
/// \tparam kExp      the member of the region-B exponential axis the cell's
///                   ladders seed with
template <DivisionForm kDivision, RegionBExp kExp>
void CellUniformSingle(BoysBudget budget,
                       FitRoute route,
                       EvalScheme scheme,
                       PackAxis pack,
                       int nmax,
                       float x,
                       float* out) noexcept {
    // Both routes of the uniform partition are served here and neither is an arm:
    // the rational member is a pair per interval of this lane's own grid, stored
    // in its width and read through the route dispatch in UniformOrderAtF32, so a
    // cell naming it is evaluated like any other and not stopped. The arm this
    // replaces stopped on the rational route *because the member was not stored
    // then*, and that is what changed.
    const auto with_budget = [&]<FitRoute kRoute, BoysBudget kBudget>() {
        const auto with_pack = [&]<PackAxis kPack, EvalScheme kScheme>() {
            using Policy = EvalPolicy<kRoute, kScheme, kBudget, kPack,
                                      FitGranularity::kUniform, kDivision, kExp>;

            BoysAllOrdersF32<Policy>(nmax, x, out);
        };

        const auto with_scheme = [&]<EvalScheme kScheme>() {
            if (pack == PackAxis::kOrders)
            {
                with_pack.template operator()<PackAxis::kOrders, kScheme>();
            } else
            {
                with_pack.template operator()<PackAxis::kArguments, kScheme>();
            }
        };

        if (scheme == EvalScheme::kHorner)
        {
            with_scheme.template operator()<EvalScheme::kHorner>();
        } else
        {
            with_scheme.template operator()<EvalScheme::kSplitClenshaw>();
        }
    };

    if (route == FitRoute::kRationalMinimax)
    {
        if (budget == BoysBudget::kFp16)
        {
            with_budget.template operator()<FitRoute::kRationalMinimax, BoysBudget::kFp16>();
        } else
        {
            with_budget.template operator()<FitRoute::kRationalMinimax, BoysBudget::kFloat>();
        }
    } else
    {
        if (budget == BoysBudget::kFp16)
        {
            with_budget.template operator()<FitRoute::kChebyshev, BoysBudget::kFp16>();
        } else
        {
            with_budget.template operator()<FitRoute::kChebyshev, BoysBudget::kFloat>();
        }
    }
}

/// The same, with the budget and the division form as two further axes: each
/// narrows to the template argument it names, so a cell is measured through its
/// own policy and never through another cell's.
///
/// The uniform partition reaches its own arm above rather than the lattice
/// below: this lane carries the grid on one route, and the arm is where that
/// route and the axes this lane holds it on are read.
///
/// \tparam kDivision the form this call's cells divide in
/// \tparam kExp      the member of the region-B exponential axis this call's
///                   cells seed their ladders with
template <DivisionForm kDivision, RegionBExp kExp>
void CellFormPolicySingle(BoysBudget budget,
                          FitRoute route,
                          EvalScheme scheme,
                          PackAxis pack,
                          FitGranularity granularity,
                          int nmax,
                          float x,
                          float* out) noexcept {
    if (granularity == FitGranularity::kUniform)
    {
        CellUniformSingle<kDivision, kExp>(budget, route, scheme, pack, nmax, x, out);

        return;
    }

    const auto with_partition = [&]<BoysBudget kBudget,
                                    FitRoute kRoute,
                                    EvalScheme kScheme,
                                    PackAxis kPack>() {
        if (granularity == FitGranularity::kNarrow)
        {
            CellEntrySingle<kBudget, kRoute, kScheme, kPack, FitGranularity::kNarrow, kDivision,
                            kExp>(nmax, x, out);
        } else
        {
            CellEntrySingle<kBudget, kRoute, kScheme, kPack, FitGranularity::kCoarsest, kDivision,
                            kExp>(nmax, x, out);
        }
    };

    const auto with_pack = [&]<BoysBudget kBudget, FitRoute kRoute, EvalScheme kScheme>() {
        if (pack == PackAxis::kOrders)
        {
            with_partition.template operator()<kBudget, kRoute, kScheme, PackAxis::kOrders>();
        } else
        {
            with_partition.template operator()<kBudget, kRoute, kScheme, PackAxis::kArguments>();
        }
    };

    const auto with_scheme = [&]<BoysBudget kBudget, FitRoute kRoute>() {
        if (scheme == EvalScheme::kHorner)
        {
            with_pack.template operator()<kBudget, kRoute, EvalScheme::kHorner>();
        } else
        {
            with_pack.template operator()<kBudget, kRoute, EvalScheme::kSplitClenshaw>();
        }
    };

    const auto with_route = [&]<BoysBudget kBudget>() {
        if (route == FitRoute::kRationalMinimax)
        {
            with_scheme.template operator()<kBudget, FitRoute::kRationalMinimax>();
        } else
        {
            with_scheme.template operator()<kBudget, FitRoute::kChebyshev>();
        }
    };

    if (budget == BoysBudget::kFp16)
    {
        with_route.template operator()<BoysBudget::kFp16>();
    } else
    {
        with_route.template operator()<BoysBudget::kFloat>();
    }
}

/// The same, at one region-B exponential named as a compile-time value and the
/// division form dispatched at run time inside it, for the same reason and in the
/// same shape as the double lane's \c CellExpPolicy: the form is a template
/// argument of the engine's entries, so each of the three cases below is the
/// instantiation a cell of that form runs, at the member this call names.
///
/// \tparam kExp the member of the region-B exponential axis this call's cells
///              seed their ladders with
template <RegionBExp kExp>
void CellExpPolicySingle(BoysBudget budget,
                         FitRoute route,
                         EvalScheme scheme,
                         PackAxis pack,
                         FitGranularity granularity,
                         DivisionForm division,
                         int nmax,
                         float x,
                         float* out) noexcept {
    switch (division)
    {
    case DivisionForm::kExactDivision:
        CellFormPolicySingle<DivisionForm::kExactDivision, kExp>(budget, route, scheme, pack,
                                                                 granularity, nmax, x, out);
        return;
    case DivisionForm::kPlainReciprocal:
        CellFormPolicySingle<DivisionForm::kPlainReciprocal, kExp>(budget, route, scheme, pack,
                                                                   granularity, nmax, x, out);
        return;
    case DivisionForm::kRefinedReciprocal:
        CellFormPolicySingle<DivisionForm::kRefinedReciprocal, kExp>(budget, route, scheme, pack,
                                                                     granularity, nmax, x, out);
        return;
    }
}

/// The same, for a cell named at run time, with the region-B exponential
/// dispatched here: both members are served on this lane as on the double lane's,
/// so the switch below names the two enumerators and leaves no default label, and
/// a cell that runs the non-default member runs its own instantiation rather than
/// being answered by the other member's.
void CellPolicySingle(BoysBudget budget,
                      FitRoute route,
                      EvalScheme scheme,
                      PackAxis pack,
                      FitGranularity granularity,
                      DivisionForm division,
                      RegionBExp regionBExp,
                      int nmax,
                      float x,
                      float* out) noexcept {
    switch (regionBExp)
    {
    case RegionBExp::kAccurate:
        CellExpPolicySingle<RegionBExp::kAccurate>(budget, route, scheme, pack, granularity,
                                                   division, nmax, x, out);
        return;
    case RegionBExp::kFast:
        CellExpPolicySingle<RegionBExp::kFast>(budget, route, scheme, pack, granularity, division,
                                               nmax, x, out);
        return;
    }
}

/// One single-precision cell of the option space, evaluated as the entry its own
/// axes and its class's budget select.
///
/// There is no run-time convenience entry on this side to fall back on, unlike
/// the double lane's \c BoysAllOrdersWithRoute: every cell of the single-precision
/// space is an instantiation a consumer reaches by naming its policy, and the
/// probe reaches it the same way.
void CellValuesSingle(const Option& option, int nmax, float x, float* out) noexcept {
    CellPolicySingle(BudgetOf(option.precision),
                     option.route,
                     option.scheme,
                     option.pack,
                     option.granularity,
                     option.division,
                     option.regionBExp,
                     nmax,
                     x,
                     out);
}

/// One cell of the option space, evaluated as the entry its axes select: the
/// route-and-scheme entry for the default policy's own shape, which is the dispatch
/// a consumer reaches without naming a template argument, and its own policy
/// instantiation for every other cell, which is how a consumer reaches a
/// partition, a packing axis or a division form. Both are branches on the same
/// run-time values, so a cell's cost includes its own selection, as the run-time
/// selector options' cost already does.
///
/// The run-time entry is the class's own default's - the row this build's seam
/// gives \c BoysAllOrders, or the five where it gives none - and therefore
/// divides in that row's form and seeds its region-B ladders with that row's
/// exponential, whatever it is handed: it takes neither as an argument, so a cell
/// at one of the other two form members or at the other member of the exponential
/// axis would be measured through arithmetic it did not name. The form, the member
/// and the packing axis are part of the shortcut's own condition for that reason,
/// and a cell of the class's shape at another member of any of them takes its own
/// instantiation below like any other cell.
///
/// The condition is read off the name the run-time entry stands in for and not
/// off the five: in a build whose seam carries a row for the class, a cell at the
/// five's own combination is a cell at another arithmetic, and measuring it
/// through the selector would report one combination's cost for another's.
void CellValues(const Option& option, int nmax, double x, double* out) noexcept {
    using SelectorClass = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>;

    if (option.granularity == SelectorClass::kGranularity && option.pack == SelectorClass::kPack &&
        option.division == SelectorClass::kDivision &&
        option.regionBExp == SelectorClass::kRegionBExp)
    {
        BoysAllOrdersWithRoute(option.route, option.scheme, nmax, x, out);
        return;
    }

    CellPolicy(option.route,
               option.scheme,
               option.pack,
               option.granularity,
               option.division,
               option.regionBExp,
               nmax,
               x,
               out);
}

/// Runs one option's own calls over the workload and hands every value it
/// returns to `visit(x, order, value)`, with `x` the argument the option was
/// actually asked about — the narrowed one for a lane that rounds its argument
/// before evaluating.
///
/// The single body shared by the timed checksum and the untimed accuracy
/// comparison.
template <typename Visit>
void VisitValues(const Workload& work, Buffers& buffers, const Option& option, Visit&& visit) {
    switch (option.kind)
    {
    case OptionKind::kBatchFp64:
    {
        double values[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];

            BoysAllOrders(n, work.x[i], values);

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

    case OptionKind::kSingleCell:
    {
        // The lane narrows the argument before it evaluates, and the row is
        // measured on the argument it was actually asked about: the float lane's
        // entries take a float, and the half lanes' take a half and return one.
        float values[kMaxBoysOrder + 1];

        // The device lane has no host body at all. The option book is enumerated
        // over the classes this machine measures, so no row of that class exists;
        // were one to reach here it would be run by one of the bodies below and
        // its figures reported as the device lane's arithmetic, which is a
        // substitution this stops rather than makes.
        if (option.precision == OptionPrecision::kFp32Device)
        {
            std::fprintf(stderr,
                         "boys-probe: %s names the device lane's single precision, which this "
                         "machine has no entry to run\n",
                         option.name.c_str());
            std::abort();
        }

        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];

            if (option.precision == OptionPrecision::kFp32)
            {
                const float x = static_cast<float>(work.x[i]);
                CellValuesSingle(option, n, x, values);

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k, static_cast<double>(values[k]));
                }

                continue;
            }

#if BoysFp16
            if (option.precision == OptionPrecision::kFp16)
            {
                const F16 x = static_cast<F16>(static_cast<float>(work.x[i]));
                CellValuesSingle(option, n, static_cast<float>(x), values);

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k,
                          static_cast<double>(static_cast<F16>(values[k])));
                }
            } else
            {
                const Bf16 x = static_cast<Bf16>(static_cast<float>(work.x[i]));
                CellValuesSingle(option, n, static_cast<float>(x), values);

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k,
                          static_cast<double>(static_cast<Bf16>(values[k])));
                }
            }
#else
            // A build whose seam is closed constructs no cell of either half
            // class - the enumeration offers none and registers the two lanes as
            // not carried - so this arm is unreachable here. It stops rather than
            // returning nothing, because a value visitor that answered no values
            // would read as an option this build measured and found empty.
            std::fprintf(stderr,
                         "boys-probe: %s names a half-precision lane this build does not carry "
                         "(BoysFp16 = 0)\n",
                         option.name.c_str());
            std::abort();
#endif // BoysFp16
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
        // Unreachable in this build, as in the single-cell arm above: the
        // enumeration offers no option of either kind.
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

    case OptionKind::kGroupedFp32:
    {
        // The float lane's all-N entry, reached the way the double lane's is: one
        // call per order run, over that run's own arguments. The lane narrows the
        // argument before it evaluates, so the values are visited on the argument
        // the lane was asked about.
        for (std::size_t r = 0; r + 1 < work.runBegin.size(); ++r)
        {
            const std::size_t from = work.runBegin[r];
            const std::size_t to = work.runBegin[r + 1];
            const std::size_t run = to - from;
            const int n = work.runOrder[r];

            for (std::size_t j = 0; j < run; ++j)
            {
                buffers.runXF32[j] = static_cast<float>(work.x[work.sorted[from + j]]);
            }

            BoysAllNF32(n, buffers.runXF32.data(), buffers.runOutF32.data(), run);

            for (std::size_t j = 0; j < run; ++j)
            {
                const float x = buffers.runXF32[j];

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k,
                          static_cast<double>(
                              buffers.runOutF32[static_cast<std::size_t>(k) * run + j]));
                }
            }
        }

        break;
    }

    case OptionKind::kSingleFp64:
    {
        // One order at one argument, called once for each order the ladder asks
        // for: the same values the all-orders row returns, bought a call at a
        // time. That is what the shape costs a caller who has one (n, x) pair.
        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];

            for (int k = 0; k <= n; ++k)
            {
                visit(work.x[i], k, BoysSingle(k, work.x[i]));
            }
        }

        break;
    }

    case OptionKind::kSingleFp32:
    {
        for (std::size_t i = 0; i < work.x.size(); ++i)
        {
            const int n = work.order[i];
            const float x = static_cast<float>(work.x[i]);

            for (int k = 0; k <= n; ++k)
            {
                visit(static_cast<double>(x), k,
                      static_cast<double>(BoysSingleF32(k, x)));
            }
        }

        break;
    }

    case OptionKind::kSingleHalf:
    {
#if BoysFp16
        if (option.precision == OptionPrecision::kFp16)
        {
            for (std::size_t i = 0; i < work.x.size(); ++i)
            {
                const int n = work.order[i];
                const F16 x = static_cast<F16>(static_cast<float>(work.x[i]));

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k,
                          static_cast<double>(BoysSingleF16(k, x)));
                }
            }
        } else
        {
            for (std::size_t i = 0; i < work.x.size(); ++i)
            {
                const int n = work.order[i];
                const Bf16 x = static_cast<Bf16>(static_cast<float>(work.x[i]));

                for (int k = 0; k <= n; ++k)
                {
                    visit(static_cast<double>(x), k,
                          static_cast<double>(BoysSingleBf16(k, x)));
                }
            }
        }
#else
        // Unreachable in this build, as in the arms above: the enumeration offers
        // no option of this kind where the seam is closed.
        std::fprintf(stderr,
                     "boys-probe: %s names a half-precision lane this build does not carry "
                     "(BoysFp16 = 0)\n",
                     option.name.c_str());
        std::abort();
#endif // BoysFp16

        break;
    }

    case OptionKind::kFixedNFp64:
    {
        // One order at every argument of the array: one call per order, each
        // sweeping the whole array, and the argument's own value is read out of
        // the sweep of its own order. Every order the workload asks for is
        // computed once for the whole array, which is what the fixed-N entry
        // costs a caller who needs one order per element.
        const std::size_t count = work.x.size();

        for (int k = 0; k <= work.options.nmax; ++k)
        {
            BoysFixedN(k, work.x.data(), buffers.fixedN.data(), count, 1);

            for (std::size_t i = 0; i < count; ++i)
            {
                if (work.order[i] >= k)
                {
                    visit(work.x[i], k, buffers.fixedN[i]);
                }
            }
        }

        break;
    }

    case OptionKind::kAllNAtOrdersFp64:
    {
        // The workload's whole answer in one call: the tops arrive as an array
        // and each column stops at its own, so the arguments are neither padded
        // up to the batch's largest order nor split into runs.
        const std::size_t count = work.x.size();

        BoysAllNAtOrders(work.order.data(), work.x.data(), buffers.planesFp64.data(), count);

        for (std::size_t i = 0; i < count; ++i)
        {
            for (int k = 0; k <= work.order[i]; ++k)
            {
                visit(work.x[i], k,
                      buffers.planesFp64[static_cast<std::size_t>(k) * count + i]);
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
/// a time: it is the only anchor that is the same value for every option, where
/// a batch lane seeds region A at the batch's own highest order and would hide
/// that difference behind the label "reference".
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
    /// moves; one that drifts is a pair the two do not carry the clock alike.
    double drift = 0.0;

    /// Whether the band clears one: the middle half of the run put the rival
    /// behind the leader, and the probe orders only what it saw in that half.
    bool ordered = false;
};

/// Measures one rival against one leader over the run's rounds: the ratio is
/// formed inside a round and never across rounds, and the ordering is that
/// pair's lower quartile clearing one, so the middle half of the run has to put
/// the rival behind rather than the run's average.
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
/// it is that figure, so the row a class names as its leader is the row the
/// class's own listing shows first. Such a figure is formed inside a round, as
/// the column's ratio to the certified lane scaled by that lane's own cost per
/// argument, so the statistic is a paired one and a drift common to a round is
/// in both terms of it and cancels. A table whose columns carry no printed
/// figure - a refinement run's own, which holds no lane to pair against - falls
/// back to the lower quartile of the column's cost per argument over the rounds.
///
/// The ordering is then the paired comparison - every other column's within-round
/// ratio to the leader, banded, and placed only when the band clears one.
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
        // one, which this run cannot order, and a band that lies below one, which
        // the run's own statistic contradicts. The probe makes neither an ordering.
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

/// The columns of the measurements that are one class: the options of one
/// precision, built at the library's own full-accuracy multiplier, and
/// answering one question shape.
///
/// Membership is the shape, never a comparison of documented
/// figures: a figure belongs to one lane, and reading one lane's bound against
/// another's empties the other lane's class by construction.
///
/// \param report    the report whose measurements are read
/// \param precision the lane whose class is wanted
/// \param shape     the question shape whose class is wanted
///
/// \returns the columns of that class's measured options, in the report's own
///          order
std::vector<std::size_t> ReferencePoolColumns(const OptionProbeReport& report,
                                              OptionPrecision precision, OptionProbeShape shape) {
    std::vector<std::size_t> columns;

    for (std::size_t index = 0; index < report.measurements.size(); ++index)
    {
        const OptionProbeMeasurement& measurement = report.measurements[index];

        if (measurement.measured && measurement.precision == precision &&
            measurement.shape == shape)
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
        return "nothing named";
    }

    // A value no arm above names: a member a newer header carries and this revision has not been
    // taught, or a value cast in from outside the enumeration. "nothing named" is one member's
    // own phrase, so it is not the answer here: a reader of it cannot tell that value from the
    // member that means this run named no default.
    return "(not a way this revision names)";
}

/// The figure the report prints for each column of the round table, indexed by
/// the column, and zero for a column the run formed no figure for.
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
/// one question shape — what the option hands back — and a class is the only set
/// this
/// probe orders inside: two options are alternatives only if they answer the same
/// question, and one of another precision or another shape answers
/// a different one. A class of one is not a ranking: it names its entry by there
/// being no alternative.
///
/// Inside a class the run's own statistic — the figure the report prints for each
/// option, a within-round ratio to the certified lane scaled by that lane's own
/// cost per argument — names a leader, and a rival is placed behind it only when
/// the pair's own within-round ratio clears one in the middle half of the rounds.
/// A rival whose band straddles one cannot be placed, and the class reports it.
///
/// The default comes from one class: the certified double lane's precision at the
/// library's own full-accuracy multiplier, for the all-orders shape this probe's
/// workload asks, whose members were all built at that multiplier and all hand
/// back the
/// same thing, so choosing among them trades nothing but speed. Where the class's
/// own rounds order it, the default is that class's fastest row by the figure this
/// report prints; where they cannot place one row behind another, the tie is
/// re-run alone at a larger protocol and the row that led most of those runs is
/// the default, with the row the printed figures put first kept beside it as the
/// record of what the shorter protocol said. Both rows are re-measured by the
/// vote, and where they differ the difference is what says the class's top entries
/// cannot be separated. The report never leaves the caller without a
/// default, and never names one by counting the library's tables.
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

    // The column of one option is its index in the measurements: the round table
    // and the measurement rows are built from the same option book in the same
    // order.
    const auto column_of = [&report](const OptionProbeMeasurement* measurement) {
        return static_cast<std::size_t>(measurement - report.measurements.data());
    };

    const auto name_of = [&report](std::size_t column) {
        return report.measurements[column].name;
    };

    // ---- the classes: one precision, one question shape ----------------------
    struct ClassRecord {
        OptionProbeClass entry;
        std::vector<std::size_t> columns;
        RunOutcome outcome;
    };

    std::vector<ClassRecord> records;

    for (const OptionPrecision precision : kCellPrecisions)
    {
        // Shapes are the walk, so a precision's questions are listed beside
        // each other: a class is ordered among its own shapes and never across
        // them.
        for (const OptionProbeShape shape : kProbeShapes)
        {
            ClassRecord record;
            record.entry.precision = precision;
            record.entry.shape = shape;
            record.entry.name =
                Text("%s %s", PrecisionName(precision), OptionProbeShapeName(shape));

            for (const OptionProbeMeasurement* measurement : live)
            {
                if (measurement->precision == precision && measurement->shape == shape)
                {
                    record.columns.push_back(column_of(measurement));
                }
            }

            if (record.columns.empty())
            {
                // A key this build has no option for is still a key the probe
                // enumerates: the entry is made
                // where the classes are put in order so the key stands among
                // the classes it belongs between.
                continue;
            }

            record.outcome = OrderColumns(rounds, record.columns, figures);

            // A class holding one option is measured, not ordered, however
            // many rounds the run took.
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

            // A class is keyed on the precision and the shape, not
            // on a bound: a member of the same key that documents a looser
            // figure is reported by name here rather than ranked as an equal.
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
    }

    // The stage that refines one class and no other. The key is the class's whole
    // key: a stage re-runs the class's own pool, and no other class may be handed
    // its vote — the option such a vote
    // names answers another question or belongs to another lane and is not even a
    // member of that class.
    const auto refinement_of = [&report](OptionPrecision precision, OptionProbeShape shape) {
        for (const OptionProbeRefinement& stage : report.refinements)
        {
            if (stage.precision == precision && stage.shape == shape)
            {
                return &stage;
            }
        }

        return static_cast<const OptionProbeRefinement*>(nullptr);
    };

    const auto class_of = [&records](OptionPrecision precision,
                                     OptionProbeShape shape) -> ClassRecord* {
        for (ClassRecord& record : records)
        {
            if (record.entry.precision == precision &&
                record.entry.shape == shape)
            {
                return &record;
            }
        }

        return nullptr;
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

    // ---- the default: one combination, from the certified lane's own class ----
    const std::vector<std::size_t> pool =
        ReferencePoolColumns(report, OptionPrecision::kFp64, kWorkloadShape);
    const ClassRecord* doubles = class_of(OptionPrecision::kFp64, kWorkloadShape);

    if (pool.empty() || doubles == nullptr)
    {
        refuse(Text("no option of the certified double lane's precision at the library's own "
                    "full-accuracy multiplier answering this workload's all-orders question was "
                    "measured, and that class is the only pool this probe takes a default from: "
                    "naming any other option would be naming one in a "
                    "different arithmetic or for a caller asking a different question, and none "
                    "of those is this default. The run measured %zu option(s); the fastest of them "
                    "is '%s' at %.2f ns/argument, documented at %.3g",
                    live.size(), overall->name.c_str(), overall->nsPerArgument, overall->bound),
               "CANNOT DETERMINE: the certified lane's reference class is empty");
        return;
    }

    // ---- what the default's own class actually compared ----------------------
    //
    // The class names a default for every axis a caller may leave unnamed, and a
    // class that names one is a class that compared that axis's members. A member
    // it held no row for was not in the comparison, and a default read as a race
    // that was never run is the defect this records: the runs that set this
    // library's shipped partition default predated the uniform partition, so the
    // comparison naming 'narrow' had a three-member axis and measured one of them.
    //
    // The partition is the axis checked here because it is the one this library's
    // default states as a comparison. The members are read off the class's own
    // rows and the axis is read off the library's table, so what the line reports
    // is a member this run did not compare rather than one the library lacks.
    {
        std::vector<std::string> compared;

        for (const std::size_t column : doubles->columns)
        {
            for (const FitGranularityInfo& partition : report.granularities)
            {
                if (partition.granularity == report.measurements[column].granularity)
                {
                    compared.push_back(partition.name);
                }
            }
        }

        std::string missing;

        for (const FitGranularityInfo& partition : report.granularities)
        {
            if (std::find(compared.begin(), compared.end(), partition.name) != compared.end())
            {
                continue;
            }

            missing += missing.empty() ? "" : ", ";
            missing += partition.name;
        }

        if (!missing.empty())
        {
            report.defaultClassAbsent.push_back(
                Text("partition: the class held no row for %s, so the default it names was not "
                     "chosen against %s",
                     missing.c_str(), missing.find(',') == std::string::npos ? "it" : "them"));
        }
    }

    const RunOutcome pooled = OrderColumns(rounds, pool, figures);
    report.fastestAtReferenceAccuracy = name_of(pooled.leader);

    const OptionProbeMeasurement& fastest = report.measurements[pooled.leader];
    const OptionProbeRefinement* stage =
        refinement_of(OptionPrecision::kFp64, kWorkloadShape);

    // Where the class's own rounds ordered it, the default is the row they put first;
    // where they could not place one row behind another, the vote over the refinement
    // stage RunOptionProbe ran over the tied rows alone decides, and the row the
    // printed figures put first is kept beside it as the record of the shorter protocol.
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
    } else
    {
        // The class could not be ordered, so the run has a tie to report and not a
        // ranking. Where the longer protocol ran, it decides: options this run cannot
        // separate are settled by which was fastest in most runs, so the vote's row is
        // the default and the row the printed figures put first is the record of what
        // the shorter protocol said. With no vote to consult, the printed figure is
        // all there is to go on.
        const bool voted = stage != nullptr && stage->ran && !stage->winner.empty();

        report.recommended = voted ? stage->winner : fastest.name;

        report.defaultHow =
            voted ? (stage->unanimous
                         ? OptionProbeDefaultHow::kRefined
                         : (stage->plurality ? OptionProbeDefaultHow::kVote
                                             : OptionProbeDefaultHow::kChosenAmongEquals))
                  : OptionProbeDefaultHow::kChosenAmongEquals;
    }

    // Whether the vote named the row the report names or the printed figures did,
    // the reason and the confidence say so: the class's top entries could not be
    // separated, and what the other evidence preferred is on the record beside the
    // figure the default is printed with.
    const bool row_differs = stage != nullptr && stage->ran && !stage->winner.empty() &&
                             fastest.name != report.recommended;

    const OptionProbeMeasurement* other_row = nullptr;

    if (row_differs)
    {
        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.name == fastest.name)
            {
                other_row = &measurement;
            }
        }
    }

    // ---- the classes' own entries, now that the default is known --------------
    const auto materialise = [&](ClassRecord& record) {
        const bool defaultClass = record.entry.precision == OptionPrecision::kFp64 &&
                                  record.entry.shape == kWorkloadShape;

        // The stage that refines this class and no other: the class's own tie is
        // the only one its vote may settle.
        const OptionProbeRefinement* own =
            refinement_of(record.entry.precision, record.entry.shape);

        if (defaultClass)
        {
            // The class the default is taken from reports the default's own how,
            // whichever way it was reached.
            record.entry.how = report.defaultHow;
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
            defaultClass
                ? report.recommended
                : (own != nullptr && own->ran && !own->winner.empty() ? own->winner
                                                                     : record.entry.leader);

        std::string note;

        if (record.columns.size() == 1)
        {
            note = Text("one entry, so this is not a ranking: nothing in this class was measured "
                        "against it, and it is the entry this class names by there being no "
                        "alternative. It was built at m=%g and documents %.3g",
                        kBoysFullAccuracyMultiplier, record.entry.bound);
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
                        "the class was %.2f%%. Every row of this class was built at m=%g and "
                        "answers the same question - %s - so this is the fastest option for that "
                        "question, and nothing in the class traded accuracy for "
                        "its place",
                        record.columns.size() - 1, record.entry.leader.c_str(), report.pairedRounds,
                        100.0 * record.outcome.widestBand,
                        kBoysFullAccuracyMultiplier,
                        OptionProbeShapeQuestion(record.entry.shape));
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
        else if (defaultClass && row_differs && stage != nullptr)
        {
            // The class the default is taken from, where the vote named the default and
            // the class's own printed figures put another of its tied rows first: the
            // class says here that the two could not be separated.
            note += Text(". The report's own figures put '%s' first instead, and those two rows "
                         "cannot be separated by this run: what is measured about them is that no "
                         "pair of the class placed one behind the other, and neither is reported as "
                         "the fastest",
                         fastest.name.c_str());
        }

        if (!record.entry.differingBounds.empty())
        {
            note += Text(". Not every row of this class documents the leader's %.3g: %s - a class "
                         "is keyed on the precision and the question shape, so a row of "
                         "another figure is in it and is reported here rather than being named the "
                         "class's entry by its cost alone",
                         record.entry.bound, JoinNames(record.entry.differingBounds).c_str());
        }

        record.entry.note = std::move(note);
        report.classes.push_back(record.entry);
    };

    // The classes in the report's own order: each precision's question shapes
    // beside each other. A key this build measured nothing in still stands in that
    // order.
    for (const OptionPrecision precision : kCellPrecisions)
    {
        for (const OptionProbeShape shape : kProbeShapes)
        {
            ClassRecord* record = class_of(precision, shape);

            if (record != nullptr)
            {
                materialise(*record);
                continue;
            }

            OptionProbeClass empty;
            empty.precision = precision;
            empty.shape = shape;
            empty.name = Text("%s %s", PrecisionName(precision), OptionProbeShapeName(shape));
            empty.note = "no option of this precision and shape produced a figure on this run";
            report.classes.push_back(std::move(empty));
        }
    }

    // ---- the answer -----------------------------------------------------------
    //
    // The clock check, done rather than assumed, and attached to whatever the
    // default turns out to be. A pair whose ratio moves between the run's halves
    // is a pair whose two options do not carry a decaying clock alike. Which clock
    // an option draws is a property of the registers it runs in, so the clause
    // also says whether the pool this choice compares put two different arithmetic
    // routes - two vector register widths - against each other.
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
    // with its own band and round counts.
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
            "reference class on this machine - %.2f ns/argument, its within-round ratio to the "
            "reference lane read at the middle of the %d paired rounds, %.2f with that ratio at "
            "the upper - and every one of the other %zu option(s) of that "
            "class was the slower of the two in the middle half of those rounds, so this is a "
            "measured ordering of equals and not a choice. The class is one precision for one "
            "question shape - every row of it was built at m=1 and answers the same one - "
            "so nothing in it traded accuracy for speed and nothing in it answers something else. "
            "It measured %.3g against the certified lane over this workload",
            report.recommended.c_str(), fastest.nsPerArgument, report.pairedRounds,
            fastest.nsPerArgumentMax, pool.size() - 1, fastest.maxError);
        break;
    case OptionProbeDefaultHow::kOnlyEntry:
        report.reason = Text(
            "'%s' is the default: it is the only option of the certified double lane's reference "
            "class this run measured, at %.2f ns/argument, its within-round ratio to the reference "
            "lane read at the middle of the %d paired rounds. One entry is "
            "not a ranking - there was nothing to measure it against - and it is named by there "
            "being no alternative, not as the winner of a comparison. It measured %.3g against "
            "the certified lane over this workload",
            report.recommended.c_str(), fastest.nsPerArgument, report.pairedRounds,
            fastest.maxError);
        break;
    case OptionProbeDefaultHow::kRefined:
    case OptionProbeDefaultHow::kVote:
    case OptionProbeDefaultHow::kChosenAmongEquals:
    case OptionProbeDefaultHow::kNone:
        report.reason = Text(
            "'%s' is the default: the certified double lane's own class - built at m=1, for "
            "the all-orders question this workload asks - holds %zu measured option(s), and its "
            "fastest by this run's own statistic is '%s' at %.2f ns/argument. ",
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

        if (row_differs && other_row != nullptr)
        {
            report.reason += Text(
                "The report's own figures put '%s' first, at %.2f ns/argument; the vote over the "
                "refinement runs named '%s', and that is the default. Those two cannot be separated "
                "by this run, so the class's top entries are reported as what they are - entries no "
                "pair of the class placed one behind the other - and neither is offered as the "
                "fastest: what this run measured about them is that it could not tell them "
                "apart. ",
                fastest.name.c_str(), other_row->nsPerArgument, stage->winner.c_str());
        }

        report.reason += "The default is named with the way it was reached - " +
                         std::string(HowText(report.defaultHow)) +
                         " - and not as a measured ordering, which this run did not establish.";
        break;
    default:
        // A way of reaching a default that no arm above names: a value cast in from outside
        // the enumeration, or an enumerator this revision has not been taught. Neither is a
        // way this run walked, so the reason above is not stated for it: naming the class's
        // fastest entry "the default" here would be a sentence about a choice this run did
        // not make, which is the one thing this field exists not to say.
        report.reason = Text(
            "No default is named for this class and no reason for one is stated: the way this "
            "run records for its own figure is not one the enumeration names, so there is no "
            "way to state here and nothing this run measured about one. The %zu option(s) of "
            "the class are this run's own measurements and are reported as what they are; "
            "which of them is the default is not something this run settles.",
            pool.size());
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
        // The count is the vote's own: how many of the runs led with the row this
        // report names.
        confidence = Text(
            "MEDIUM: the class's own rounds could not separate the tied options and neither did "
            "every refinement run; '%s' led %d of the %d runs. A majority over runs is weaker "
            "evidence than a measured ordering, and this is one",
            report.recommended.c_str(),
            stage != nullptr
                ? static_cast<int>(std::count(stage->runLeaders.begin(), stage->runLeaders.end(),
                                              stage->winner))
                : 0,
            stage != nullptr ? stage->runs : 0);
        break;
    case OptionProbeDefaultHow::kChosenAmongEquals: {
        // Three ways this run can end in a tie it cannot break, and they are not
        // the same evidence: no stage ran, the stage's runs split, or the stage
        // named another row - the last told in figures, since the default is then
        // not the row the vote preferred.
        std::string voteClause;

        if (stage != nullptr && stage->ran && !stage->winner.empty())
        {
            if (row_differs)
            {
                voteClause = Text("; the report's own figures put '%s' first instead, at %.2f "
                                  "ns/argument against its %.2f, and those two cannot be separated "
                                  "by this run",
                                  fastest.name.c_str(),
                                  other_row != nullptr ? other_row->nsPerArgument : 0.0,
                                  fastest.nsPerArgument);
            }
            else
            {
                voteClause = "; the vote over the refinement runs was split across several of them";
            }
        }

        confidence = Text(
            "LOW: this run could not separate the options of the class%s, so '%s' is one of the rows "
            "it cannot tell apart - the way it was reached is "
            "stated above and it is not a ranking",
            voteClause.c_str(), report.recommended.c_str());
        break;
    }
    case OptionProbeDefaultHow::kOnlyEntry:
        confidence = Text(
            "LOW: one entry is not a ranking and this class holds one. The option is the default "
            "by there being no alternative in its precision, and its cost above is "
            "the whole of what this run measured about it");
        break;
    case OptionProbeDefaultHow::kNone:
        break;
    default:
        // The same value the reason above refuses, and refused here in the same terms: a
        // confidence is about a default this run named, and this run names none for a way it
        // did not walk. Left unsaid, this field would open with the measured sentence below
        // and read as a run that concluded something.
        confidence = Text(
            "No confidence is stated: this run names no default for the class - the way it "
            "reached its own figure is not one the enumeration names - so there is nothing a "
            "confidence here would be about");
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

/// The refinement stage: the tied options of one class, measured alone at a
/// larger protocol, repeated, and voted on.
///
/// The options re-measured are the class's fastest and every option of it the
/// main run could not place behind the fastest; nothing else in the option space
/// is touched.
///
/// Each run is a fresh pass over the tied set at ProbeOptions::passes times
/// ProbeOptions::refinementFactor passes of ProbeOptions::rounds times the same
/// factor rounds, with its own shuffle, and each run is ordered by the same
/// OrderColumns the main run used. A refinement run's table holds the tied set
/// alone - no certified lane to pair against - so its anchor is the tied set's own
/// fastest row and its figures are each option's cost against that row, formed
/// inside a single round. That is the quantity the report orders by, so a run is a
/// longer measurement of the same figure and not a second one: the vote says
/// whether the row the report names as the class's fastest holds up over more
/// rounds of the comparison, and a vote for any other row is the run saying it
/// could not separate the two.
///
/// \param report    the report the stage is recorded in
/// \param book      the options the run measured, in the report's own order
/// \param work      the workload, already built and warmed
/// \param buffers   the scratch the calls need
/// \param options   the protocol in force, read for the run count and the factor
/// \param precision the lane whose reference class is refined
/// \param shape     the question shape of the class refined, which with the
///                  precision is the class this stage's vote is about
/// \param pool      the columns of the class's measured options
/// \param mainLeader the class's fastest by the main run's own rounds, which is
///                  the entry the stage falls back to when no run places one
void Refine(OptionProbeReport& report, const std::vector<Option>& book, const Workload& work,
            Buffers& buffers, const ProbeOptions& options, OptionPrecision precision,
            OptionProbeShape shape, const std::vector<std::size_t>& pool,
            const std::string& mainLeader) {
    OptionProbeRefinement stage;
    stage.precision = precision;
    stage.shape = shape;
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

        // The runs are ordered in the quantity the report orders by: each option's
        // cost against the tied set's own fastest, formed inside a single round so
        // that whatever the clock did in that round is in both terms of the ratio
        // and cancels. Slot zero is the anchor, as the reference lane is in the
        // report's own table: it is the main run's leader - the row the report names
        // unless a vote confirms another - its ratio is one in every round, and the
        // figure is taken at the quantile the report's own figure is taken at.
        std::vector<double> runFigures(pool.size(), 0.0);

        if (!table.empty() && pool.size() > 0)
        {
            std::vector<double> anchors;

            for (const std::vector<double>& row : table)
            {
                anchors.push_back(row.front());
            }

            const double anchorCost = QuantileOf(anchors, kStatisticQuantile);

            for (std::size_t slot = 0; slot < pool.size(); ++slot)
            {
                std::vector<double> ratios;

                for (const std::vector<double>& row : table)
                {
                    if (row.front() > 0.0)
                    {
                        ratios.push_back(row[slot] / row.front());
                    }
                }

                if (ratios.size() >= 2 && anchorCost > 0.0)
                {
                    runFigures[slot] = anchorCost * QuantileOf(ratios, kFigureQuantile);
                }
            }
        }

        const RunOutcome outcome = OrderColumns(table, columns, runFigures);

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
    report.referenceBound = LaneCellBound(OptionPrecision::kFp64, kDefaultFitRoute,
                                          kDefaultEvalScheme, kDefaultPackAxis,
                                          kDefaultFitGranularity, kDefaultDivisionForm);

    const Workload work = BuildWorkload(options);
    Buffers buffers = MakeBuffers(work);
    report.orderRuns = work.runOrder.size();
    report.largestRun = work.largestRun;

    // The coverage book is built before the selection is applied, so a narrowed
    // run accounts for every cell of the library's option space just as a full one
    // does.
    //
    // The device lane's book is enumerated beside the measured classes' and kept
    // out of them. A refusal of that lane is work the library owes, so it belongs
    // in the account; no cell of it has a row here, because this probe has no
    // device arm, and a row of it would be timed by another
    // class's body and reported as that lane's arithmetic. The report counts and
    // lists it in its own block, and no cell of it reaches the option book below.
    std::vector<OptionProbeCell> cells;

    for (const OptionPrecision precision : kCellPrecisions)
    {
        const std::vector<OptionProbeCell> book = EnumerateCells(precision);
        cells.insert(cells.end(), book.begin(), book.end());
    }

    report.cells = cells;

    // The device lane's book, enumerated the same way and carried apart from the
    // cells above. It is counted so that the space this library has is the space
    // the report accounts for, and it is measured nowhere, so it is not handed to
    // the option book: every cell in that vector becomes a timed row.
    report.deviceCells = EnumerateCells(OptionPrecision::kFp32Device);

    std::vector<Option> options_ = EnumerateOptions(report.backends, report.granularities, cells,
                                                    report.unoffered, report.notCarried);

    // A caller who names a set is answered about that set, so the set is narrowed
    // before anything is measured rather than filtered out of the report
    // afterwards. A name is one of three things and the report keeps them apart: an
    // option this build serves, a cell of the option space the library refuses
    // where it is named, and a name that is no cell of the space at all. An unknown
    // name is recorded rather than silently measuring nothing, because an empty
    // report otherwise reads as a machine on which nothing is fast, and a refused
    // cell is recorded with the library's reason.
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
    const std::size_t reference = ReferenceIndex(options_, options.reference);
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
    // reported; it decides nothing, because a fixed spin read by wall clock
    // measures the clock as much as the load.
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
        measurement.shape = options_[index].shape;
        measurement.route = options_[index].route;
        measurement.scheme = options_[index].scheme;
        measurement.granularity = options_[index].granularity;
        measurement.pack = options_[index].pack;
        measurement.division = options_[index].division;
        measurement.regionBExp = options_[index].regionBExp;
        measurement.bound = options_[index].bound;
        measurement.ownBound = options_[index].ownBound;
        measurement.ownLo = options_[index].ownLo;
        measurement.ownHi = options_[index].ownHi;
        measurement.maxError = walker.MaxError();
        measurement.bitIdenticalToReference = walker.BitIdentical();

        // One figure judges the whole row, and it is a whole-domain one: the
        // entry's own, which the library reports through BoysAccuracyGuaranteed
        // whatever partition the entry reads. A partition's own figures are certified for
        // the fitted interval alone, and the measured column is a difference from
        // the certified lane rather than an error against the true function — so
        // the row carries that figure and its interval as information instead.
        measurement.meetsBound = measurement.maxError <= measurement.bound;
        measurement.withinReferenceFloor =
            !measurement.meetsBound && measurement.maxError <= report.referenceBound;
    }

    // The paired figures. Each option's ratios to the reference are formed
    // inside the round its two calls were timed in, so a drift common to the
    // round - a clock that decayed between this round and the last, a background
    // process that stole time from both - is in both terms of the ratio and
    // cancels. The figure is the middle of those rounds rather than the quartile
    // the band above is made of, which credits each row alike and is reciprocal
    // between a row and the anchor.
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
            measurement.ratioToReference = QuantileOf(ratios, kFigureQuantile);
            measurement.ratioLo = QuantileOf(ratios, kStatisticQuantile);
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
            // do not carry the clock alike.
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
    // measurement.
    //
    // The tied set is read the same way the conclusion reads it: off the figures
    // the report prints, so the options re-run here are the ones the classes
    // below will report as the pair that was left unplaced.
    const std::vector<double> printed = PrintedFigures(report);

    // One stage per class that needs one, so a stage's tied set is the class's
    // own: a precision's shapes are one class each and get one vote each, none of
    // which can decide an entry of another's question.
    for (const OptionPrecision precision :
         {OptionPrecision::kFp64, OptionPrecision::kFp32, OptionPrecision::kFp16,
          OptionPrecision::kBf16})
    {
        for (const OptionProbeShape shape : kProbeShapes)
        {
            const std::vector<std::size_t> pool = ReferencePoolColumns(report, precision, shape);

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
            // others were already ordered by the main run.
            std::vector<std::size_t> tied;
            tied.push_back(mainRun.leader);

            for (const std::size_t column : mainRun.unplaced)
            {
                tied.push_back(column);
            }

            Refine(report, options_, work, buffers, options, precision, shape, tied,
                   report.measurements[mainRun.leader].name);
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
/// Used for the lists that span more than a line — a precision class's ranked
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

/// One line of the report, its text wrapped at the report's own width and continued under the
/// prefix it started on.
///
/// A class's note enumerates the members of its class - a class of a 144-cell lane holds about
/// six thousand characters of names - and a note printed as one line is a line no terminal, a
/// diff or a log reads, which is the promise `kReportWidth` states. It is also why the line is
/// not left to a fixed buffer: a note cut mid-word is a list that ends where the buffer did, and
/// the class printed after it would be printed on the same line.
std::string WrappedAfter(const std::string& prefix, const std::string& text) {
    const std::string indent(prefix.size(), ' ');
    std::string out;
    std::string line = prefix;
    std::size_t start = 0;

    while (start < text.size())
    {
        std::size_t end = text.find(' ', start);

        if (end == std::string::npos)
        {
            end = text.size();
        }

        const std::string word = text.substr(start, end - start);
        start = end + 1;

        if (word.empty())
        {
            continue;
        }

        if (line.size() + 1 + word.size() > kReportWidth && line.size() > indent.size())
        {
            out += line + "\n";
            line = indent;
        } else if (line.size() > indent.size())
        {
            line += ' ';
        }

        line += word;
    }

    return out + line + "\n";
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

// --- the closure ------------------------------------------------------------

/// The classes this build's option space is spread over, in the order the coverage
/// walks them: the precision classes this machine measures, and the device lane's
/// book beside them.
///
/// The device lane is a class of the space and not a filter over it: its cells are
/// enumerated from that lane's own fit table, and it is counted
/// where the others are. It is not one of \c kCellPrecisions, which are the classes
/// this machine can run.
std::vector<OptionPrecision> SpaceClasses() {
    std::vector<OptionPrecision> classes;

    for (const OptionPrecision precision : kCellPrecisions)
    {
        classes.push_back(precision);
    }

    classes.push_back(OptionPrecision::kFp32Device);
    return classes;
}

/// The routes a lane's own fit table reports, each taken once: one factor of that
/// lane's option space.
///
/// A route has one row per region it supplies, so the same route appears in the table
/// more than once; a cell names the route and not the region its fit covers, so this
/// counts each route once — the same reading \c EnumerateCells makes when it walks
/// that table.
std::size_t LaneRouteCount(Precision lane) {
    std::vector<FitRoute> seen;

    for (const FitRouteInfo& route : LaneRoutes(lane))
    {
        if (std::find(seen.begin(), seen.end(), route.route) == seen.end())
        {
            seen.push_back(route.route);
        }
    }

    return seen.size();
}

/// The cells one class carries: the product of the axes the library reports for that
/// class's own lane.
///
/// Every factor is read from the library's own reporting API, which is what makes this
/// the space's own product and not the report's: it is the reading
/// \c OptionProbeSpaceClosure holds its own walk of the axes and the report's books to,
/// and the three are one total or the closure fails.
std::size_t CellsPerClass(OptionPrecision precision) {
    return LaneRouteCount(LaneOf(precision)) * BoysEvalSchemes().size() *
           BoysFitGranularities().size() * BoysPackAxes().size() * BoysDivisionForms().size() *
           BoysRegionBExps().size();
}

/// The cells the library's own axes admit for one class.
std::size_t AdmittedCells(OptionPrecision precision) {
    return CellsPerClass(precision);
}

/// Whether a row of this run's own table is a cell of the space.
///
/// The row's name is the cell's identity in this probe: the report's own option grammar
/// carries every axis of the cell — the division form included, whose segment exists so
/// that two cells cannot print one name — so a match on the name is a match on the cell.
/// A row that matches no cell of either book is one of the call shapes the axes are not
/// crossed with, measured at its own default policy alone.
///
/// \param report the run's report
/// \param row a row of it
///
/// \returns true when a cell of either book is that row's cell
bool RowIsACell(const OptionProbeReport& report, const OptionProbeMeasurement& row) {
    for (const OptionProbeCell& cell : report.cells)
    {
        if (cell.name == row.name)
        {
            return true;
        }
    }

    for (const OptionProbeCell& cell : report.deviceCells)
    {
        if (cell.name == row.name)
        {
            return true;
        }
    }

    return false;
}

/// The rows this run carries that are no cell of the space, in the table's own order:
/// the call shapes the axes are not crossed with. Named so the block can state which
/// they are rather than only how many.
std::vector<std::string> RowsOutsideTheSpace(const OptionProbeReport& report) {
    std::vector<std::string> names;

    for (const OptionProbeMeasurement& row : report.measurements)
    {
        if (!RowIsACell(report, row))
        {
            names.push_back(row.name);
        }
    }

    return names;
}

/// The option space's closure, printed: the space, one count per state a cell of it can
/// be in, the arithmetic over those counts, the run's own rows against the places the
/// space owes it, and the verdict.
///
/// It is the last block of the report on every path: what a run did with the space is a
/// fact about the run, and a run that reached none of it says so here rather than
/// leaving a reader to infer it from the absence of figures. \c OptionProbeClosure is
/// what the readings of one total are, and the counts below are printed beside the
/// tables they were read from.
///
/// The closure is handed in rather than counted here: the section above names the debt
/// this block counts, and both read the one counting so that the prose and the
/// arithmetic cannot state two different numbers.
void AppendOptionClosure(std::string& text,
                         const OptionProbeReport& report,
                         const OptionProbeClosure& closure) {
    const std::vector<OptionPrecision> classes = SpaceClasses();
    const std::vector<std::string> outside = RowsOutsideTheSpace(report);

    text += "\n\nthe closure — the space above counted, every cell of it in one state of this "
            "run and no\n  cell in two, so that a cell the report does not account for is "
            "visible as a missing\n  number rather than as an absence:\n";

    text += "  the space: the axes this library reports, crossed once per class this build\n";
    text += "  carries — the precision classes this machine measures and the device lane's "
            "book\n  beside them, which this host cannot run. One class's cells are the product "
            "of its own\n  lane's axes, every factor read from the library:\n";
    text += Text("    %-11s %8s %8s %11s %13s %15s %13s %7s\n", "class", "routes", "schemes",
                 "partitions", "packing axes", "division forms", "exponentials", "cells");

    std::string product;

    for (const OptionPrecision precision : classes)
    {
        const std::size_t admitted = AdmittedCells(precision);

        text += Text("    %-11s %8zu %8zu %11zu %13zu %15zu %13zu %7zu\n",
                     PrecisionName(precision), LaneRouteCount(LaneOf(precision)),
                     BoysEvalSchemes().size(), BoysFitGranularities().size(),
                     BoysPackAxes().size(), BoysDivisionForms().size(), BoysRegionBExps().size(),
                     admitted);

        if (!product.empty())
        {
            product += " + ";
        }

        product += Text("%zu", admitted);
    }

    text += Text("  the axes' own product over the %zu class(es): %s = %zu cell(s), and the walk "
                 "of them\n  this closure makes enumerates %zu — the library's own factors twice, "
                 "so a class\n  added to the axes and not to the walk is a number here. The two "
                 "books this run\n  carries hold %zu cell(s), the third reading of the same total, "
                 "and all three have\n  to agree.\n",
                 closure.classes, product.c_str(), closure.admitted, closure.walked,
                 closure.enumerated);

    text += Text("\n  MEMBERS: %zu of %zu cell(s) of the space are measured on this machine and\n"
                 "                published\n",
                 closure.measured, closure.total);
    text += Text("                %zu offered at a place this run carried a row for and producing "
                 "no figure\n",
                 closure.offeredNoFigure);
    text += Text("                %zu not asked for by this run's request: the space is stated "
                 "whole and this\n                run measured the cells its request named\n",
                 closure.notAsked);
    text += Text("                %zu served by the library and owed an arithmetic this build's "
                 "backend table\n                does not carry\n",
                 closure.unoffered);
    text += Text("                %zu served by the library and owed a body this build's own seam "
                 "does not\n                declare\n",
                 closure.notCarried);
    text += Text("                %zu of the device lane's book, counted apart and not against this "
                 "build:\n                this probe has no device arm — `benchmarks/"
                 "boys_option_probe.cpp`\n                names no CUDA entry and this probe calls "
                 "none — so no cell of that\n                lane has a row here, whatever "
                 "arithmetic the library documents for it\n",
                 closure.deviceNotRun);
    text += Text("                %zu refused with the library's own reason, each named in the "
                 "coverage above:\n                unbuilt work the library owes\n",
                 closure.refused);
    text += Text("  the arithmetic: %zu + %zu + %zu + %zu + %zu + %zu + %zu = %zu\n",
                 closure.measured, closure.offeredNoFigure, closure.notAsked, closure.unoffered,
                 closure.notCarried, closure.deviceNotRun, closure.refused, closure.states);
    text += Text("                 the space's own total: %zu cell(s) = the %zu above + %zu in no "
                 "state\n",
                 closure.total, closure.states, closure.unaccounted);
    text += Text("                 the run's own table: %zu row(s) carried, against the %zu place(s) "
                 "the\n                 space owes this request — the served cells of the classes "
                 "this machine\n                 measures that this build carries a body for and "
                 "the request named;\n                 the other %zu row(s) are no cell of it\n",
                 closure.rows, closure.rowsOwed, closure.shapesNotCrossed);

    if (!outside.empty())
    {
        text += Text("                 the call shapes the axes are not crossed with, each measured "
                     "at its own\n                 default policy alone: %s. Each stands for the "
                     "cell(s) of its own class that\n                 crossing it with the axes "
                     "would add, less the cell the row itself stands\n                 at: %zu cell(s) "
                     "of work named in the paragraph above\n",
                     Joined(outside).c_str(), closure.crossedOwed);
    }

    if (closure.closed)
    {
        text += Text("  the verdict: PASS — every one of the space's %zu cell(s) is in one state "
                     "above and\n                none is in two\n",
                     closure.total);
        return;
    }

    text += "  the verdict: FAIL — the closure does not hold, and this run's exit status says "
            "so:\n";

    if (closure.states != closure.total)
    {
        text += Text("                the states sum to %zu and the space has %zu cell(s)\n",
                     closure.states, closure.total);
    }

    if (closure.walked != closure.admitted)
    {
        text += Text("                the walk enumerates %zu cell(s) where the library's axes "
                     "admit %zu\n",
                     closure.walked, closure.admitted);
    }

    if (closure.enumerated != closure.admitted)
    {
        text += Text("                the run's own books carry %zu cell(s) where the library's "
                     "axes admit\n                %zu\n",
                     closure.enumerated, closure.admitted);
    }

    if (closure.rows != closure.rowsOwed + closure.shapesNotCrossed)
    {
        text += Text("                the run's own table carries %zu row(s) where the space owes "
                     "it %zu\n                place(s) and %zu row(s) are no cell of it\n",
                     closure.rows, closure.rowsOwed, closure.shapesNotCrossed);
    }

    if (closure.unaccounted > 0)
    {
        text += Text("                %zu cell(s) are in no state above: served by the library, "
                     "carried by\n                this build and named by this request, and with "
                     "no row in the run's own\n                table\n",
                     closure.unaccounted);
    }
}

} // namespace

OptionProbeClosure OptionProbeSpaceClosure(const OptionProbeReport& report) {
    OptionProbeClosure closure;

    const std::vector<OptionPrecision> classes = SpaceClasses();
    closure.classes = classes.size();

    // The space, walked here rather than read off the report: the members the states below
    // are placed on are this build's own option space, so a run that carried no book at all
    // still owes every cell it did not present, and the count in no state is that number
    // rather than an absence. The walk is the one the report's books are enumerated by,
    // which is what makes the readings below one space and not three.
    std::vector<OptionProbeCell> space;

    for (const OptionPrecision precision : classes)
    {
        closure.admitted += AdmittedCells(precision);

        const std::vector<OptionProbeCell> book = EnumerateCells(precision);
        space.insert(space.end(), book.begin(), book.end());
    }

    closure.walked = space.size();
    closure.enumerated = report.cells.size() + report.deviceCells.size();
    closure.total = closure.admitted;

    // Where one cell of a book is placed. The questions are asked in the order the report
    // answers them: the library's own carriage first — a cell the library refuses is
    // refused whatever this build carries — then whether this build can run it at all,
    // then whether this run's request named it, and last where its place in the run's own
    // table is.
    const auto place = [&](const OptionProbeCell& cell) {
        if (!cell.served)
        {
            ++closure.refused;
            return;
        }

        if (cell.precision == OptionPrecision::kFp32Device)
        {
            ++closure.deviceNotRun;
            return;
        }

        if (std::find(report.unoffered.begin(), report.unoffered.end(), cell.name) !=
            report.unoffered.end())
        {
            ++closure.unoffered;
            return;
        }

        if (std::find(report.notCarried.begin(), report.notCarried.end(), cell.name) !=
            report.notCarried.end())
        {
            ++closure.notCarried;
            return;
        }

        const bool asked = report.options.only.empty() ||
                           std::find(report.options.only.begin(), report.options.only.end(),
                                     cell.name) != report.options.only.end();

        if (!asked)
        {
            ++closure.notAsked;
            return;
        }

        ++closure.rowsOwed;

        for (const OptionProbeMeasurement& row : report.measurements)
        {
            if (row.name == cell.name)
            {
                (row.measured ? closure.measured : closure.offeredNoFigure) += 1;
                return;
            }
        }

        // A cell this build serves, carries a body for and was asked for, with no row in
        // the run's own table: the run owes the space a place here and does not carry one.
        // Counted as unaccounted rather than into the nearest state, and the verdict fails
        // on it.
        ++closure.unaccounted;
    };

    for (const OptionProbeCell& cell : space)
    {
        place(cell);
    }

    closure.rows = report.measurements.size();

    for (const OptionProbeMeasurement& row : report.measurements)
    {
        if (RowIsACell(report, row))
        {
            continue;
        }

        // A row the run carries that is no cell of the space: one of the call shapes the
        // axes are not crossed with, measured at its own default policy alone. The cells
        // its crossing with the axes would add are its own class's cells, less the one
        // cell the row itself stands at.
        closure.shapesNotCrossed += 1;

        const std::size_t perClass = CellsPerClass(row.precision);
        closure.crossedOwed += perClass > 0 ? perClass - 1 : 0;
    }

    closure.states = closure.measured + closure.offeredNoFigure + closure.notAsked +
                     closure.unoffered + closure.notCarried + closure.deviceNotRun +
                     closure.refused;

    // The verdict: the states partition the space, the walk and the run's own books are both
    // the axes' product, and the run's own table carries exactly the places the space owes
    // this request plus the rows that are no cell of it. A run that measured nothing closes
    // on nothing and fails — a closure that could not fail would be a decoration.
    closure.closed = closure.unaccounted == 0 && closure.states == closure.total &&
                     closure.walked == closure.admitted && closure.enumerated == closure.admitted &&
                     closure.rows == closure.rowsOwed + closure.shapesNotCrossed;

    return closure;
}

// --- the seam this run implies ----------------------------------------------

namespace {

/// The seam's own spelling of one cell of the row it writes, axis by axis.
///
/// The spelling is text because the file it is written into is text, and the case label
/// beside it is the enumerator, so the two cannot be renamed apart in silence: an
/// enumerator renamed in the library leaves the label unable to compile, and a cell the
/// switch does not name reaches the last return, which is a spelling no build carries —
/// loud where the emitted file is compiled rather than silent where it is written.
constexpr const char* RouteCell(FitRoute route) noexcept {
    switch (route)
    {
    case FitRoute::kChebyshev:
        return "FitRoute::kChebyshev";
    case FitRoute::kRationalMinimax:
        return "FitRoute::kRationalMinimax";
    }

    return "(a route this probe names no cell for)";
}

constexpr const char* SchemeCell(EvalScheme scheme) noexcept {
    switch (scheme)
    {
    case EvalScheme::kSplitClenshaw:
        return "EvalScheme::kSplitClenshaw";
    case EvalScheme::kHorner:
        return "EvalScheme::kHorner";
    }

    return "(a scheme this probe names no cell for)";
}

constexpr const char* PackCell(PackAxis axis) noexcept {
    switch (axis)
    {
    case PackAxis::kArguments:
        return "PackAxis::kArguments";
    case PackAxis::kOrders:
        return "PackAxis::kOrders";
    }

    return "(a packing axis this probe names no cell for)";
}

constexpr const char* GranularityCell(FitGranularity granularity) noexcept {
    switch (granularity)
    {
    case FitGranularity::kCoarsest:
        return "FitGranularity::kCoarsest";
    case FitGranularity::kNarrow:
        return "FitGranularity::kNarrow";
    case FitGranularity::kUniform:
        return "FitGranularity::kUniform";
    }

    return "(a partition this probe names no cell for)";
}

constexpr const char* DivisionCell(DivisionForm form) noexcept {
    switch (form)
    {
    case DivisionForm::kExactDivision:
        return "DivisionForm::kExactDivision";
    case DivisionForm::kPlainReciprocal:
        return "DivisionForm::kPlainReciprocal";
    case DivisionForm::kRefinedReciprocal:
        return "DivisionForm::kRefinedReciprocal";
    }

    return "(a division form this probe names no cell for)";
}

constexpr const char* BudgetCell(BoysBudget budget) noexcept {
    switch (budget)
    {
    case BoysBudget::kFloat:
        return "BoysBudget::kFloat";
    case BoysBudget::kFp16:
        return "BoysBudget::kFp16";
    }

    return "(a budget this probe names no cell for)";
}

/// The seam's own spelling of the seventh axis cell: which exponential a region-B ladder is
/// seeded with.
///
/// It is a cell like the other six because the seam's row macro takes it like the other six,
/// and the two members are two certified arithmetics rather than one member with a faster
/// spelling: the accurate member is the exponential itself and the fast member the corrected
/// seed's own series. A row that cannot name the member it was measured at cannot state the
/// arithmetic it means, which is why this cell exists rather than being left to a default.
constexpr const char* ExpCell(RegionBExp exp) noexcept {
    switch (exp)
    {
    case RegionBExp::kAccurate:
        return "RegionBExp::kAccurate";
    case RegionBExp::kFast:
        return "RegionBExp::kFast";
    }

    return "(a region-B exponential this probe names no cell for)";
}

/// The seam's own token for a lane's precision cell, and for a shape's cell: the two keys
/// its class list is written in. They are matched by name because the seam's list names its
/// cells as tokens rather than as values, and this is the spelling that list uses.
constexpr const char* PrecisionToken(Precision lane) noexcept {
    switch (lane)
    {
    case Precision::kFp64:
        return "kFp64";
    case Precision::kFp32:
        return "kFp32";
    case Precision::kFp16:
        return "kFp16";
    case Precision::kFp32Device:
        return "kFp32Device";
    case Precision::kFp64Device:
        return "kFp64Device";
    case Precision::kFp16Device:
        return "kFp16Device";
    }

    return "(a lane this probe names no cell for)";
}

constexpr const char* ShapeToken(Shape shape) noexcept {
    switch (shape)
    {
    case Shape::kSingle:
        return "kSingle";
    case Shape::kAllOrders:
        return "kAllOrders";
    case Shape::kFixedN:
        return "kFixedN";
    case Shape::kAllN:
        return "kAllN";
    case Shape::kAllNAtOrders:
        return "kAllNAtOrders";
    }

    return "(a shape this probe names no cell for)";
}

/// The probe's own classes for one of the seam's precision cells.
///
/// The two keys differ on purpose and the difference is the formats: the seam keys a class
/// by the LANE a call resolves in, and the probe keys a class by the FORMAT it times, so the
/// one half-lane cell is answered by whichever of the two half classes this run ranked. A
/// run that ranked both says in its report whether the two winners were one combination,
/// because a lane carries one row and a difference between them is a finding about the
/// keying rather than a row to fold.
std::vector<OptionPrecision> ProbeClassesOf(const std::string& precisionToken) {
    if (precisionToken == PrecisionToken(Precision::kFp16))
    {
        return {OptionPrecision::kFp16, OptionPrecision::kBf16};
    }

    for (const OptionPrecision precision : SpaceClasses())
    {
        if (precisionToken == PrecisionToken(LaneOf(precision)))
        {
            return {precision};
        }
    }

    return {};
}

/// The probe's own class shape for one of the seam's shape cells. The two vocabularies name the
/// same five questions and this is where they are held to each other: the switch is exhaustive
/// over the seam's shapes, so a shape the seam gains and the probe does not is a warning here
/// rather than a class the probe silently stops ranking.
constexpr OptionProbeShape ProbeShapeFor(Shape shape) noexcept {
    switch (shape)
    {
    case Shape::kSingle:
        return OptionProbeShape::kSingle;
    case Shape::kAllOrders:
        return OptionProbeShape::kAllOrders;
    case Shape::kFixedN:
        return OptionProbeShape::kFixedN;
    case Shape::kAllN:
        return OptionProbeShape::kAllN;
    case Shape::kAllNAtOrders:
        return OptionProbeShape::kAllNAtOrders;
    }

    return OptionProbeShape::kAllOrders;
}

/// The probe's own shape for one of the seam's shape cells, and whether the seam names one at
/// all: the probe carries a class for each of the seam's five shapes, so only a token outside
/// that set has no measurement behind it, and that answer is about the token and not the shape
/// vocabulary.
bool ProbeShapeOf(const std::string& shapeToken, OptionProbeShape& shape) {
    for (const Shape candidate :
         {Shape::kSingle, Shape::kAllOrders, Shape::kFixedN, Shape::kAllN, Shape::kAllNAtOrders})
    {
        if (shapeToken == ShapeToken(candidate))
        {
            shape = ProbeShapeFor(candidate);
            return true;
        }
    }

    return false;
}

/// The spelling a class is named by in this report's own words, which is what the block
/// naming the classes it emitted and refused is written in.
constexpr const char* LaneSpelling(Precision lane) noexcept {
    switch (lane)
    {
    case Precision::kFp64:
        return "fp64";
    case Precision::kFp32:
        return "fp32";
    case Precision::kFp16:
        return "fp16";
    case Precision::kFp32Device:
        return "fp32-device";
    case Precision::kFp64Device:
        return "fp64-device";
    case Precision::kFp16Device:
        return "fp16-device";
    }

    return "unknown";
}

constexpr const char* ShapeSpelling(Shape shape) noexcept {
    switch (shape)
    {
    case Shape::kSingle:
        return "single";
    case Shape::kAllOrders:
        return "all-orders";
    case Shape::kFixedN:
        return "fixed-n";
    case Shape::kAllN:
        return "all-n";
    case Shape::kAllNAtOrders:
        return "all-n-at-orders";
    }

    return "unknown";
}

/// One class of the seam's table, as the seam's own list names it.
///
/// A class is a (device, precision, shape) triple, so the device cell is part of the key and not
/// a constant of the writer's: the list this is read from carries both halves of the interface,
/// and a row written back with a device cell the list did not name would be a row for another
/// class than the one read.
struct SeamClass {
    std::string device; ///< the seam's device cell, e.g. "kHost" or "kDevice"
    std::string precision; ///< the seam's precision cell, e.g. "kFp64"
    std::string shape; ///< the seam's shape cell, e.g. "kAllOrders"
};

#define BOYS_PROBE_SEAM_CLASS(device, precision, shape, route, scheme, budget, pack, granularity,  \
                              division, exp)                                                     \
    {#device, #precision, #shape},

/// The classes the seam in force carries, read from its own `BOYS_BUILD_DEFAULT_ROWS` list
/// rather than written here: a class the seam adds is emitted without an edit here, and one
/// it drops stops being emitted, which is what keeps the file this writes a replacement for
/// the file it read.
///
/// The macro names every cell the row format carries, the region-B exponential included,
/// because it is expanded from the same list the row macro is: a row that dropped a cell
/// would otherwise be read here as a class and refused there, and the two readings of one
/// list have to be one reading. Only the key cells are used; the rest are named so that the
/// expansion is the row's own.
///
/// A build whose seam carries no list has no classes to write and the emitted file carries
/// none either - which is that seam's own shape, and not an omission here.
std::vector<SeamClass> SeamClasses() {
    std::vector<SeamClass> classes;

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    const SeamClass listed[] = {BOYS_BUILD_DEFAULT_ROWS(BOYS_PROBE_SEAM_CLASS)};
    classes.assign(std::begin(listed), std::end(listed));
#endif

    return classes;
}

#undef BOYS_PROBE_SEAM_CLASS

/// One row the emitted file carries: the class it is for, the `X(...)` call as the seam
/// writes it, and the comment line above it.
struct EmittedSeamRow {
    std::string klass; ///< the class as the report names it, e.g. "fp64 all-orders"
    std::string cells; ///< the X(...) call, wrapped where the seam wraps it
    std::string marker; ///< the comment above it: a measurement, or a choice and why
    bool measured = false; ///< whether the row is this run's own winning combination

    /// Whether the row states one entry this run measured of the class, that entry standing
    /// alone: a class the library carries one entry for has nothing to compare, and the seam's
    /// own header writes such a row as a choice rather than as a comparison's winner. It is
    /// neither a measurement nor a row the run failed to rank, and the block that accounts for
    /// the rows has to say which one it is.
    bool choiceAlone = false;

    /// Whether the cells written are the file's own five rather than a combination this run
    /// measured: true of a class the run ranked no cell of. A class of the device half is one of
    /// these by construction - this probe ranks the host's cells and no device cell - and its
    /// row states the same five at its own lane's budget together with the device lane's own
    /// two names, so the block that accounts for the rows names it with the host's fallbacks
    /// rather than leaving it in a group of its own.
    bool fromFive = false;
};

/// The row this run measured for one class of the seam's table, with what it was reached by.
///
/// There is no second candidate beside \c row and no flag saying the winner could not be
/// written: every axis a policy carries has a cell in the row format, so the combination that
/// won a class is a combination the file can state, and the row written is the winner. A
/// class whose winner the format could not name would be one whose row stated another
/// arithmetic under this class's key, which is the reading this struct no longer carries a
/// field for.
struct SeamWinner {
    const OptionProbeMeasurement* row = nullptr;
    OptionProbeDefaultHow how = OptionProbeDefaultHow::kNone;
    bool formatsDisagree = false; ///< the two half classes were both ranked and differed
};

/// The row one name belongs to in this run's own table, or nothing where no row carries it.
const OptionProbeMeasurement* RowNamed(const OptionProbeReport& report, const std::string& name) {
    for (const OptionProbeMeasurement& row : report.measurements)
    {
        if (row.name == name)
        {
            return &row;
        }
    }

    return nullptr;
}

/// The winner of one seam class's class, or nothing where this run ranked no cell of it.
/// The two half classes are both asked, and a disagreement between them is reported rather
/// than folded.
SeamWinner WinnerOf(const OptionProbeReport& report,
                    const std::string& precisionToken,
                    const std::string& shapeToken) {
    SeamWinner winner;
    OptionProbeShape shape = OptionProbeShape::kAllOrders;

    if (!ProbeShapeOf(shapeToken, shape))
    {
        return winner;
    }

    for (const OptionPrecision precision : ProbeClassesOf(precisionToken))
    {
        for (const OptionProbeClass& clause : report.classes)
        {
            if (clause.precision != precision || clause.shape != shape || clause.leader.empty())
            {
                continue;
            }

            const OptionProbeMeasurement* row = RowNamed(report, clause.leader);

            if (row == nullptr)
            {
                continue;
            }

            // The class's leader, written as it stands: the row format carries every axis the
            // leader's combination names, the region-B exponential included, so the fastest
            // cell of this class is a combination the file it writes can state.
            if (winner.row == nullptr)
            {
                winner.row = row;
                winner.how = clause.how;
                continue;
            }

            // The two half formats are one lane and one row: where both were ranked and their
            // leaders were not one combination, the difference is one the seam's key cannot
            // carry, and every axis of the policy is compared - the exponential included,
            // because the row format names it and a difference in it is a difference in the
            // arithmetic the lane would run.
            winner.formatsDisagree =
                winner.formatsDisagree || winner.row->route != row->route ||
                winner.row->scheme != row->scheme || winner.row->granularity != row->granularity ||
                winner.row->pack != row->pack || winner.row->division != row->division ||
                winner.row->regionBExp != row->regionBExp;
        }
    }

    return winner;
}

/// The five the emitted file states as its fallback: **the build's own five**, and not this
/// run's winner for any class.
///
/// The five are the point a class the table carries no row for resolves to, so they have to
/// be a combination every class of every lane compiles - and a run's winner is not: a class
/// whose entry evaluates one order refuses the orders axis by construction, and
/// `boys_impl.hpp` asserts it, so a file whose fallback named a batch winner stops compiling
/// for the shapes the list does not carry. That is measured and not assumed: the first
/// version of this writer used the run's own default, and a build pointed at the file it
/// wrote failed on three of those assertions (`boys_impl.hpp` 3687, 3937 and 4154) before a
/// single row of it was read.
///
/// The rows are this run's measurements and the five are the build's own choices; the two
/// answer different questions, which is why only the rows are written as measured.
struct SeamFive {
    FitRoute route = kDefaultFitRoute;
    EvalScheme scheme = kDefaultEvalScheme;
    PackAxis pack = kDefaultPackAxis;
    FitGranularity granularity = kDefaultFitGranularity;
    DivisionForm division = kDefaultDivisionForm;
};

SeamFive FileFive() {
    return SeamFive{};
}

/// A lane the seam's own precision cell can name, the device it runs on, and the budget
/// the library states for it.
///
/// The budget is resolved here, at compile time and beside the lane it belongs to, because the
/// library states one per lane of the enumeration and has none for a value outside it: the
/// table is keyed by the enumeration, so a row this file writes carries the same budget cell
/// the library's own macro would write for that lane, and a lane no enumerator names has no
/// budget to be written rather than the nearest one to it.
///
/// The device is stated besides it because a row is written for a class and a class is a
/// (device, precision, shape) triple: the classes this run measures are the host's, so a
/// class of the device half is not this run's to restate at the host's choices, and telling
/// the two halves apart is a reading of the table rather than of a spelling.
struct SeamLaneBudget {
    Precision lane;   ///< the precision lane, as the library's own enumeration names it
    BoysBudget budget; ///< the budget a class of that lane falls back to
    Device device; ///< the device that lane runs on: the seam's first key
};

/// The seam lanes, in the order the library enumerates them.
constexpr SeamLaneBudget kSeamLaneBudgets[] = {
    {Precision::kFp64, detail::LaneFallbackBudget<Precision::kFp64>(), Device::kHost},
    {Precision::kFp32, detail::LaneFallbackBudget<Precision::kFp32>(), Device::kHost},
    {Precision::kFp16, detail::LaneFallbackBudget<Precision::kFp16>(), Device::kHost},
    {Precision::kFp32Device, detail::LaneFallbackBudget<Precision::kFp32Device>(),
     Device::kDevice},
    {Precision::kFp64Device, detail::LaneFallbackBudget<Precision::kFp64Device>(),
     Device::kDevice},
    {Precision::kFp16Device, detail::LaneFallbackBudget<Precision::kFp16Device>(),
     Device::kDevice},
};

/// The `X(...)` call as the seam writes it: the cells in the seam's own order, broken after
/// the budget cell and continued under it, each line ending in the macro's backslash.
///
/// The class's own three cells are written back as the seam spelled them rather than re-spelled
/// from the enumerator they named - the device cell included, which is what keeps a device class
/// a device class: a row written with the host's cell would be a row for another class, and the
/// two would be two specializations of one template, which does not compile. A class the probe
/// reads and does not understand is still a class the file it replaces carries, and a row
/// written from the seam's own tokens cannot disagree with the seam about which class it is for.
///
/// Every axis the policy carries has a cell here, the region-B exponential included. A cell
/// the row format leaves out is a cell the seam's row macro takes and this writer must
/// therefore write: the row it emits is read back by the seam's own macro, and a call short
/// of one cell is an error where that macro expands rather than a row that resolves to the
/// default of the axis nobody wrote.
std::string SeamRowCall(const std::string& deviceToken,
                        const std::string& precisionToken,
                        const std::string& shapeToken,
                        BoysBudget budget,
                        FitRoute route,
                        EvalScheme scheme,
                        PackAxis pack,
                        FitGranularity granularity,
                        DivisionForm division,
                        RegionBExp exp) {
    return Text("    X(%s, %s, %s, %s, %s, %s,\\\n"
                "      %s, %s, %s,\\\n"
                "      %s)\\\n",
                deviceToken.c_str(), precisionToken.c_str(), shapeToken.c_str(),
                RouteCell(route), SchemeCell(scheme), BudgetCell(budget), PackCell(pack),
                GranularityCell(granularity), DivisionCell(division), ExpCell(exp));
}

/// The rows this run's own rankings imply, one per class the seam in force carries, in the
/// seam's own order.
///
/// A class this run ranked carries the combination that won its class, with
/// the figure beside it; one whose class held a single entry carries the same combination
/// under the marker a choice carries, because an entry that stood alone was not compared
/// and the seam's own header asks for the two to be written differently. A class the run
/// ranked no cell of carries the file's five at that lane's budget and says so - and a class
/// of the device half is one this run ranks no cell of by construction, because the classes
/// this run measures are the host's, so it carries those five beside the device lane's own
/// division form and its own region-B exponential. A row keyed to a device class with the
/// host's members under it is the one thing this writer may not produce: a build pointed at
/// it would resolve an unnamed device call through the host's seam.
std::vector<EmittedSeamRow> SeamRows(const OptionProbeReport& report) {
    const SeamFive five = FileFive();
    std::vector<EmittedSeamRow> rows;

    for (const SeamClass& klass : SeamClasses())
    {
        EmittedSeamRow row;
        bool laneFound = false;
        Precision lane = Precision::kFp64;
        Device laneDevice = Device::kHost;
        BoysBudget budget = detail::LaneFallbackBudget<Precision::kFp64>();
        Shape shape = Shape::kAllOrders;

        for (const SeamLaneBudget& candidate : kSeamLaneBudgets)
        {
            if (klass.precision == PrecisionToken(candidate.lane))
            {
                lane = candidate.lane;
                budget = candidate.budget;
                laneDevice = candidate.device;
                laneFound = true;
            }
        }

        for (const Shape candidate : {Shape::kSingle, Shape::kAllOrders, Shape::kFixedN, Shape::kAllN,
                                      Shape::kAllNAtOrders})
        {
            if (klass.shape == ShapeToken(candidate))
            {
                shape = candidate;
            }
        }

        if (!laneFound)
        {
            continue;
        }

        row.klass = Text("%s %s", LaneSpelling(lane), ShapeSpelling(shape));

        // A class of the device half: a class this run has no cell of to write, because the
        // classes this run measures are the host's. The row is the file's own point at the
        // device lane's own names - the five above at this lane's budget beside
        // `kDefaultDeviceDivisionForm` and `kDefaultDeviceRegionBExp`, which is what a device
        // call that names no policy resolves to - and not the host's two, which would leave a
        // build resolving an unnamed device call through the host's seam.
        if (laneDevice == Device::kDevice)
        {
            row.fromFive = true;
            row.marker = "    /* a choice, not a measurement: this run ranks no cell of a\n"
                         "       class of the device half, so the row states the five above at this\n"
                         "       lane's budget beside the device lane's own two names */\\\n";
            row.cells = SeamRowCall(klass.device, klass.precision, klass.shape, budget,
                                    five.route, five.scheme, five.pack, five.granularity,
                                    kDefaultDeviceDivisionForm, kDefaultDeviceRegionBExp);
            rows.push_back(std::move(row));
            continue;
        }

        const SeamWinner winner = WinnerOf(report, klass.precision, klass.shape);

        // The row written is the class's own winner, exactly as the run ranked it. Every axis
        // the policy carries has a cell in this format, so there is no combination the run
        // measured that this file cannot state, and no second candidate to fall back to: a row
        // written at any other member would carry this class's key over another arithmetic.
        const OptionProbeMeasurement* written = winner.row;

        if (written != nullptr)
        {
            // A class reached by one entry standing alone is a choice and not a comparison:
            // the seam's own header asks for the marker the two carry to differ.
            const bool walkover = winner.how == OptionProbeDefaultHow::kOnlyEntry;

            row.measured = !walkover;
            row.choiceAlone = walkover;

            if (walkover)
            {
                row.marker = "    /* a choice, not a measurement: one entry of this class was "
                             "measured\n"
                             "       and it is the last one standing, so this row is an answer and "
                             "not the\n"
                             "       winner of a comparison */\\\n";
            }
            else
            {
                row.marker = Text("    /* measured: m = 1, the %s class, %.2f ns per\n"
                                  "       argument on this host; the entry was reached by %s */\\\n",
                                  ShapeSpelling(shape), written->nsPerArgument,
                                  OptionProbeDefaultHowName(winner.how).c_str());
            }

            row.cells = SeamRowCall(klass.device, klass.precision, klass.shape, budget,
                                    written->route, written->scheme, written->pack,
                                    written->granularity, written->division,
                                    written->regionBExp);
            rows.push_back(std::move(row));
            continue;
        }

        // A class the run ranked no cell of: the fallback is stated, and it is the file's own
        // point rather than a measurement of this machine. The exponential cell is the
        // library's host default, which is the member the fallback's arithmetic has always
        // been; the five above do not carry that axis.
        row.fromFive = true;
        row.marker = "    /* a choice, not a measurement: this run ranked no cell of this\n"
                     "       class, so the row states the five above at this lane's budget and the\n"
                     "       library's own region-B exponential */\\\n";
        row.cells = SeamRowCall(klass.device, klass.precision, klass.shape, budget, five.route,
                                five.scheme, five.pack, five.granularity, five.division,
                                kDefaultHostRegionBExp);
        rows.push_back(std::move(row));
    }

    return rows;
}

/// The host lanes the seam's key is written for: this probe's own measurable classes folded to
/// the lanes they run in. The two half formats are one lane - the library declares the bf16
/// entries under the fp16 name, so a row written for one is the row both resolve to - and the
/// fold is `ProbeClassesOf`'s reading of that and not a lane list written out here.
std::vector<Precision> SeamLanes() {
    std::vector<Precision> lanes;

    for (const OptionPrecision precision : kCellPrecisions)
    {
        const Precision lane = LaneOf(precision);

        if (std::find(lanes.begin(), lanes.end(), lane) == lanes.end())
        {
            lanes.push_back(lane);
        }
    }

    return lanes;
}

/// The number of classes the seam's own key reaches: its lanes, by the shapes it states.
std::size_t SeamClassCount() {
    return SeamLanes().size() * std::size(kSeamShapes);
}

/// The seam this run would write, as the report's own account of it: the classes it emits
/// from this run's rankings, the classes it carries at the file's five because the run ranked
/// no cell of them, the classes the seam's own key reaches and the file carries no row for at
/// all, and any disagreement between the two half formats.
///
/// There is no set here for a class whose winner the file could not state: every axis a policy
/// carries has a cell in the row format, so the combination that won a class is one the file
/// writes. The region-B exponential is what that cost while the format carried six cells.
///
/// The block exists because the file is the point and the account is what makes it
/// checkable: a reader sees which rows a run measured before pointing a build at it, and a
/// class of the seam's key that no row of the file is written for is named as that rather
/// than left as a class nobody asked about.
void AppendDefaultsBlock(std::string& text, const OptionProbeReport& report) {
    const std::vector<EmittedSeamRow> rows = SeamRows(report);
    std::size_t measured = 0;

    for (const EmittedSeamRow& row : rows)
    {
        measured += row.measured ? 1 : 0;
    }

    text += "\n\nthe build-defaults seam this run implies — the rows a replacement "
            "(`--emit-defaults\n  <file>`, the command `boys/boys_build_defaults.hpp` names) "
            "would carry:\n";

    if (measured == 0)
    {
        text += "  no row and no file: this run measured no class, and a file of rows this run\n";
        text += "  did not rank would be a transcription of the seam rather than a measurement\n";
        text += "  of this machine.\n";
        return;
    }

    text += Text("  %zu of the %zu class(es) this file's row list carries - the seam's own key "
                 "reaches\n  %zu - carry a measured row, one row per class:\n",
                 measured, rows.size(), SeamClassCount());

    for (const EmittedSeamRow& row : rows)
    {
        if (row.measured)
        {
            text += Text("    %s\n", row.klass.c_str());
        }
    }

    std::size_t alone = 0;
    std::size_t fromFive = 0;

    for (const EmittedSeamRow& row : rows)
    {
        alone += row.choiceAlone ? 1 : 0;
        fromFive += row.fromFive ? 1 : 0;
    }

    if (alone > 0)
    {
        // A class the library carries one entry for has nothing to compare, and the seam's
        // own header asks for such a row to be marked a choice. It is neither of the two
        // groups beside it: the run did measure the class's one entry - the row's own figure
        // is that measurement - and it did rank a cell of the class, so the block below is
        // not its place either.
        text += Text("  %zu class(es) carry the one entry this run measured of them, which stood "
                     "alone:\n  a class the library carries one entry for has nothing to compare, "
                     "so the row is an\n  answer and not the winner of a comparison — the marker "
                     "a choice carries, at the\n  entry's own axes:\n",
                     alone);

        for (const EmittedSeamRow& row : rows)
        {
            if (row.choiceAlone)
            {
                text += Text("    %s\n", row.klass.c_str());
            }
        }
    }

    if (fromFive > 0)
    {
        text += Text("  %zu class(es) carry the file's own five, because this run ranked no cell "
                     "of\n  them — a class with no cell ranked has no figure of its own to carry, "
                     "and a\n  class of the device half carries the five beside the device lane's "
                     "own two names:\n",
                     fromFive);

        for (const EmittedSeamRow& row : rows)
        {
            if (row.fromFive)
            {
                text += Text("    %s\n", row.klass.c_str());
            }
        }
    }

    // The seam's own vocabulary of classes, which is what the file above is a row list for: the
    // host lanes the seam keys by, crossed with the shapes it can name. A class of it the file
    // carries no row for is named here rather than left out, because the block above is read as
    // an account of the seam and a class the seam's own key reaches and the file does not is
    // exactly what a reader cannot see for themselves.
    //
    // The lanes are this probe's own classes folded to their lanes - the two half formats are
    // one lane, `ProbeClassesOf` - and the shapes are the ones the seam states, so the
    // vocabulary here is read from the enumerations rather than written out a second time in
    // this file. The spelling is the same two helpers the rows above are spelled with, so a
    // class found here is a class the file does not carry and never a spelling that missed.
    const std::vector<Precision> lanes = SeamLanes();

    std::vector<std::string> noFigure;
    std::vector<std::string> noRow;

    for (const Precision lane : lanes)
    {
        for (const Shape shape : kSeamShapes)
        {
            const std::string klass = Text("%s %s", LaneSpelling(lane), ShapeSpelling(shape));
            bool carried = false;

            for (const EmittedSeamRow& row : rows)
            {
                carried = carried || row.klass == klass;
            }

            if (carried)
            {
                continue;
            }

            bool ranked = false;

            for (const OptionPrecision probe : ProbeClassesOf(PrecisionToken(lane)))
            {
                for (const OptionProbeClass& member : report.classes)
                {
                    if (member.precision == probe && member.shape == ProbeShapeFor(shape) &&
                        !member.leader.empty())
                    {
                        ranked = true;
                    }
                }
            }

            (ranked ? noRow : noFigure).push_back(klass);
        }
    }

    if (!noFigure.empty())
    {
        text += Text("  %zu class(es) the seam's own key reaches and no row above carries: this\n"
                     "  build's library carries no entry of that shape on that lane, so this "
                     "probe's\n  own enumeration of the class finds nothing to call and the class "
                     "has no cell\n  to rank. A row for one would state a combination nobody "
                     "measured; writing the\n  entry is the work owed, and not a refusal here. "
                     "The fixed-order call and the\n  per-argument-order array entry are declared "
                     "for the double lane alone, and the\n  half lane's array shape is its native "
                     "entry, a different question from this\n  workload's. The seam's key is %zu "
                     "host lanes by %zu shapes, %zu class(es), and this\n  file's row list is the "
                     "list it replaces — a class that list does not name is one\n  no row below can "
                     "be written for:\n",
                     noFigure.size(), lanes.size(), std::size(kSeamShapes), SeamClassCount());

        for (const std::string& klass : noFigure)
        {
            text += Text("    %s\n", klass.c_str());
        }
    }

    if (!noRow.empty())
    {
        text += Text("  %zu class(es) the seam's own key reaches and no row above carries, "
                     "though this run\n  ranked cells of them: the rows are the list this file "
                     "replaces, and a replacement\n  states the rows it read:\n",
                     noRow.size());

        for (const std::string& klass : noRow)
        {
            text += Text("    %s\n", klass.c_str());
        }
    }

    // The two half formats are one lane and one row of the seam's table. A run whose two
    // winners were one combination loses nothing by folding them; one whose winners
    // differed has found a difference the seam's key cannot carry, and says so.
    for (const SeamClass& klass : SeamClasses())
    {
        const SeamWinner winner = WinnerOf(report, klass.precision, klass.shape);

        if (winner.formatsDisagree)
        {
            text += Text("  the two half formats' winners were not one combination in the %s "
                         "class:\n  the lane carries the fp16 row, and a format-specific default "
                         "would need a\n  format key `Precision` does not have.\n",
                         klass.shape.c_str());
        }
    }
}

} // namespace

std::string FormatBuildDefaults(const OptionProbeReport& report, const std::string& takenAt) {
    const std::vector<EmittedSeamRow> rows = SeamRows(report);

    if (std::none_of(rows.begin(), rows.end(),
                     [](const EmittedSeamRow& row) { return row.measured; }))
    {
        return std::string();
    }

    const SeamFive five = FileFive();

    std::string text;
    text += "#pragma once\n\n";
    text += "/// \\file\n";
    text += "/// This build's default-policy seam, written by the option probe from the rankings it\n";
    text += Text("/// measured%s: the classes it ranked carry the combination that won the\n",
                 takenAt.empty() ? "" : Text(" on this host, %s", takenAt.c_str()).c_str());
    text += "/// class (see the row list below), and the classes it ranked no cell of carry the five\n";
    text += "/// below at their own lane's budget - a class of the device half beside the device lane's\n";
    text += "/// own two names, which are written below with the five.\n";
    text += "///\n";
    text += "/// A row is a measurement taken on one machine and not a choice of the library's, so\n";
    text += "/// this file belongs to the host that produced it: a report from a build pointed at it\n";
    text += "/// describes that host. The probe that wrote it is `boys/boys_probe.hpp`;\n";
    text += "/// `boys-option-probe --emit-defaults <file>` is the command.\n";
    text += "///\n";
    text += Text("/// The run: %d logical processor(s), the AVX2+FMA tier %s, seed %llu, %d pass(es) of\n",
                 report.logicalProcessors, report.avx2 ? "present" : "absent",
                 static_cast<unsigned long long>(report.options.seed), report.options.passes);
    text += Text("/// %d round(s) over %zu argument(s) to order %d, every option called once in every\n",
                 report.options.rounds, report.options.count, report.options.nmax);
    text += "/// round and compared through the ratio of two times inside the round they were taken\n";
    text += "/// in, so a clock common to the round cancels.\n";
    text += "///\n";
    text += "/// WHAT A REPLACEMENT CARRIES, and this file is one: the seven names below - the host's\n";
    text += "/// five and the device lane's two - this row list, and no\n";
    text += "/// `BOYS_BUILD_DEFAULTS_SHIPPED`. A build pointed at it through the\n";
    text += "/// `BOYS_BUILD_DEFAULTS` CMake option defines `BOYS_BUILD_DEFAULTS_REPLACED` and reads\n";
    text += "/// it instead of the committed file.\n";
    text += "\n";
    text += "/// The five a class the list below carries no row for resolves to. **They are the\n";
    text += "/// build's own five and not this run's winner**: the fallback has to be a combination\n";
    text += "/// every class compiles, and a run's winner for one lane's batch shape is not - a class\n";
    text += "/// whose entry evaluates one order refuses the orders axis, so this line would stop the\n";
    text += "/// file compiling if it named one. The rows below are the measurements; these five are\n";
    text += "/// the choices the file already carried.\n";

    text += Text("#define BOYS_BUILD_DEFAULT_FIT_ROUTE %s\n\n", RouteCell(five.route));
    text += Text("#define BOYS_BUILD_DEFAULT_EVAL_SCHEME %s\n\n", SchemeCell(five.scheme));
    text += Text("#define BOYS_BUILD_DEFAULT_PACK_AXIS %s\n\n", PackCell(five.pack));
    text += Text("#define BOYS_BUILD_DEFAULT_DIVISION_FORM %s\n\n", DivisionCell(five.division));
    text += Text("#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY %s\n\n", GranularityCell(five.granularity));
    text += "/// The device lane's own two names, read from the seam this run replaced and written back\n";
    text += "/// beside the host's five. A device class resolves its division form and its region-B\n";
    text += "/// exponential to these and never to the host's members above: the two lanes' published\n";
    text += "/// figures are two sets, so one name for both targets could only be wrong on one of them.\n";
    text += "/// A replacement that leaves either name out does not compile - boys/accuracy.hpp reads\n";
    text += "/// both, and the committed seam states them beside the host's five for the same reason.\n";
    text += Text("#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM %s\n\n",
                 DivisionCell(kDefaultDeviceDivisionForm));
    text += Text("#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP %s\n\n",
                 ExpCell(kDefaultDeviceRegionBExp));
    text += "/// The classes this file sets a default for: **one row per class**, in the table's own\n";
    text += "/// format. A measured row is one this run's rounds placed first in its class: the\n";
    text += "/// combination below is the winner's own, cell for cell, and every axis of the policy\n";
    text += "/// has a cell here — the region-B exponential included, so the row states the\n";
    text += "/// arithmetic the class's first place was measured at. A row marked a choice is one the\n";
    text += "/// run did not rank, and it states the five above with the lane's own exponential: the\n";
    text += "/// host's on a host class, and the device lane's beside its own division form on a class\n";
    text += "/// of the device half.\n";
    text += "#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n";

    for (const EmittedSeamRow& row : rows)
    {
        text += row.marker;
        text += row.cells;
    }

    text += "\n";
    return text;
}

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
        return "none";
    }

    // A value no arm above names: a member a newer header carries and this revision has not been
    // taught, or a value cast in from outside the enumeration. "none" is one member's own name,
    // so it is not the answer here: a reader of it cannot tell that value from the member that
    // means no default was named.
    return "(not a way this revision names)";
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

    // The anchor, named whenever the caller chose one: it is a choice of the
    // reporting and not of the measurement, so a reader comparing two runs has to
    // be able to see which two units they are in. A request the run could not
    // honour is reported as not honoured rather than as the anchor it fell back to.
    if (!report.options.reference.empty())
    {
        text += Text("  anchor: named by the caller, so the cost columns below are in units of\n"
                     "            '%s'. A rank is a ratio and does not move with the anchor;\n"
                     "            the absolute columns do.\n",
                     report.referenceOption.c_str());

        if (report.options.reference != report.referenceOption)
        {
            text += Text("            the anchor asked for, '%s', is no option this run measured,\n"
                         "            so it fell back to the entry above and says so rather than\n"
                         "            leaving every ratio without a denominator.\n",
                         report.options.reference.c_str());
        }
    }

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
    // columns line up whatever the option set is.
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
        // so here, in the row and not only in the prose below: the column is the
        // entry's whole-domain figure, a different promise from the option's own
        // fitted-interval one, and the interval is printed with it.
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
    text += "   across precisions. ns/arg is the reference lane's own lower-quartile cost scaled by "
            "this\n";
    text += "   option's ratio to it at the middle of the run's rounds — a central ratio and not "
            "the\n";
    text += "   lower quartile of the same ratios, because the reference's ratio to itself is one "
            "in\n";
    text += "   every round and any quantile of it is one, so a lower quartile there would credit "
            "the\n";
    text += "   reference with the middle of its rounds and score every other option below the "
            "middle of\n";
    text += "   its own — a credit for whichever row the run anchored on, and worth more than the\n";
    text += "   margin between the fastest options. hi is the same scaled by the upper quartile of "
            "those\n";
    text += "   ratios, lo the same scaled by their lower quartile; both are formed inside a round "
            "and carry\n";
    text += "   no drift common to one, and spread is hi over lo — the ratio between the two ends "
            "of that\n";
    text += "   row's own band, a factor and not a cost. 'vs ref band' is the range the option's "
            "ratio to\n";
    text += "   the reference lane fell in\n";
    text += "   over the run's rounds, at its lower and upper quartiles, and\n";
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
    text += "   rows, and the figure the library documents for the entry\n";
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

    text += "\n\nthe accuracy classes — one precision and one question shape, the only sets "
            "this\n";
    text += "  probe orders inside\n";
    text += "  A class is the set of options that are alternatives for one need a caller has: one "
            "precision\n";
    text += "  (fp64, fp32, fp16 or bf16), built at the library's full-accuracy multiplier, and one "
            "question\n";
    text += "  shape — what the\n";
    text += "  option hands back. The five shapes here are, in the seam's own order:\n";

    // The list is read from the shapes the probe walks and the questions are the
    // statements those shapes carry, so the headings here and the class keys
    // below cannot part company with the code that made them.
    for (const OptionProbeShape shape : kProbeShapes)
    {
        text += Text("    %-10s  %s\n", OptionProbeShapeName(shape), OptionProbeShapeQuestion(shape));
    }

    text += "  The shape is part of the key for the reason the precision is: it "
            "changes what\n";
    text += "  the caller gets back, and the figure below is per argument, so ranking an option "
            "that returns\n";
    text += "  one argument's ladder against one that returns a whole array's would rank the amount "
            "of output\n";
    text += "  against the arithmetic. Everything a caller does not choose — the fit route, the "
            "evaluation\n";
    text += "  scheme, the partition of the fitted regions, the packing axis, the division form "
            "the recursion\n";
    text += "  ends in, the exponential a region-B ladder is seeded with, and whether a sorted "
            "array is declared\n";
    text += "  so that the all-N entry skips its sort — is a\n";
    text += "  way of computing the same answer and is\n";
    text += "  a column inside the class, so those options compete in one ranking rather than "
            "dividing it.\n";
    text += "  Membership is decided by the precision and the shape, never by comparing "
            "one lane's\n";
    text += "  documented figure against another's — a figure belongs to one lane, and reading it "
            "across lanes\n";
    text += "  is how a class ends up empty by construction. Every row of a class was built at the "
            "same\n";
    text += "  multiplier and answers the same question, so nothing inside it traded accuracy for "
            "speed and\n";
    text += "  nothing inside it answers something else: the entry each class names is its fastest "
            "option at\n";
    text += "  the library's full accuracy, for that question, on this machine and no more. Nothing "
            "here is ordered\n";
    text += "  across classes: another precision or another shape is an answer to a "
            "different\n";
    text += "  question, not a slower answer to this one.\n";

    for (const OptionProbeClass& entry : report.classes)
    {
        if (entry.leader.empty())
        {
            // The note is wrapped rather than printed on the class's own line: it is a sentence
            // about the class, and the class's name is the line's prefix and not its first word.
            text += WrappedAfter(Text("  %-21s ", entry.name.c_str()), entry.note);
            continue;
        }

        // The members of this class alone, fastest first, each with the figure its
        // own row documents.
        std::vector<const OptionProbeMeasurement*> members;

        for (const OptionProbeMeasurement& measurement : report.measurements)
        {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.shape == entry.shape)
            {
                members.push_back(&measurement);
            }
        }

        std::sort(members.begin(), members.end(),
                  [](const OptionProbeMeasurement* a, const OptionProbeMeasurement* b) {
                      return a->nsPerArgument < b->nsPerArgument;
                  });

        text += Text("  %-21s %zu measured | fastest %s at %.2f ns/argument, documented at %.3g | "
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
        text += WrappedAfter("             ", entry.note);

        if (entry.precision == OptionPrecision::kFp64 && entry.shape == kWorkloadShape)
        {
            text += "             (this is the class the default below is taken from: the library's "
                    "own\n";
            text += "             default precision at its own full-accuracy multiplier, "
                    "answering this\n";
            text += "             workload's own question shape. The other shape's class above "
                    "answers a\n";
            text += "             different question and its winner is a claim about that question "
                    "alone)\n";
        }
    }

    // The coverage: the library's own option space, so a combination this build
    // does not carry is counted and given the library's reason rather than being
    // absent from the report. One book per precision class this machine measures,
    // because the lanes do not answer alike: the counts are per class, and the
    // total is their sum. The device lane's book is counted in the block below
    // them and kept out of these, because no cell of it is measured.
    std::size_t served = 0;
    std::size_t refused = 0;

    for (const OptionProbeCell& cell : report.cells)
    {
        (cell.served ? served : refused) += 1;
    }

    text += Text("\n\nthe option space — the axes this library reports, every cell of their "
                 "product at every\n");
    text += Text("  precision this build measures, and what this build does with each: %zu "
                 "served + %zu\n",
                 served, refused);
    text += Text("  refused = %zu cells over %zu precision class(es)\n", report.cells.size(),
                 std::size(kCellPrecisions));

    for (const OptionPrecision precision : kCellPrecisions)
    {
        std::size_t laneServed = 0;
        std::size_t laneRefused = 0;

        for (const OptionProbeCell& cell : report.cells)
        {
            if (cell.precision == precision)
            {
                (cell.served ? laneServed : laneRefused) += 1;
            }
        }

        const std::size_t laneIndex = static_cast<std::size_t>(LaneOf(precision));
        const std::span<const LaneContractInfo> lanes = BoysLaneContracts();
        const char* laneName = laneIndex < lanes.size() ? lanes[laneIndex].name : "unknown";

        text += Text("    %-4s (lane %s): %zu served + %zu refused = %zu cells\n",
                     PrecisionName(precision), laneName, laneServed, laneRefused,
                     laneServed + laneRefused);
    }

    {
        // The device lane's book, counted beside the classes above and measured
        // nowhere. It is enumerated and counted like theirs, because a refusal of
        // that lane is work the library owes, and no cell of it has a row here:
        // this probe has no device arm, so a row of that lane would have to be
        // filled by a host entry under a device name. The block says both things
        // in its own words, so its served count cannot be read as a count of rows
        // this probe ran.
        std::size_t deviceServed = 0;
        std::size_t reasonWidth = 0;

        for (const OptionProbeCell& cell : report.deviceCells)
        {
            if (cell.served)
            {
                deviceServed += 1;
            } else
            {
                reasonWidth = std::max(reasonWidth, cell.name.size());
            }
        }

        const std::size_t deviceRefused = report.deviceCells.size() - deviceServed;
        const std::size_t deviceLane =
            static_cast<std::size_t>(LaneOf(OptionPrecision::kFp32Device));
        const std::span<const LaneContractInfo> lanes = BoysLaneContracts();
        const char* deviceLaneName = deviceLane < lanes.size() ? lanes[deviceLane].name : "unknown";

        text += Text("    %-4s (lane %s): %zu served + %zu refused = %zu cells, "
                     "counted here\n",
                     PrecisionName(OptionPrecision::kFp32Device), deviceLaneName, deviceServed,
                     deviceRefused, report.deviceCells.size());
        text += Text("      and not measured: the %zu served cells of this book are what this "
                     "library\n",
                     deviceServed);
        text += "      serves for that lane, and none of them has a row here, because this probe\n";
        text += "      has no device arm: `benchmarks/boys_option_probe.cpp` names no CUDA entry\n";
        text += "      and this probe calls none, so a row of that lane would be a host entry's\n";
        text += "      figures under a device name. That lane's refusals are unbuilt work of the\n";
        text += "      same kind as any other class's:\n";

        if (deviceRefused > 0)
        {
            text += Text("      refused cells (%zu), each with the library's reason:\n",
                         deviceRefused);

            for (const OptionProbeCell& cell : report.deviceCells)
            {
                if (!cell.served)
                {
                    text += WrappedReason(cell.name, cell.reason, reasonWidth);
                }
            }
        }
    }

    {
        // Every axis and every member is read from the library's own reporting
        // API, so this line cannot name an axis or a member this build lacks.
        const std::vector<std::string> routes = DistinctRouteNames(BoysFitRoutes());
        std::vector<std::string> schemes;
        std::vector<std::string> axes;
        std::vector<std::string> partitions;
        std::vector<std::string> forms;
        std::vector<std::string> exponentials;

        for (const EvalSchemeInfo& scheme : BoysEvalSchemes())
        {
            schemes.push_back(scheme.name != nullptr ? scheme.name : "(unnamed)");
        }

        for (const PackAxisInfo& axis : BoysPackAxes())
        {
            axes.push_back(axis.name != nullptr ? axis.name : "(unnamed)");
        }

        for (const DivisionFormInfo& form : BoysDivisionForms())
        {
            forms.push_back(form.name != nullptr ? form.name : "(unnamed)");
        }

        for (const RegionBExpInfo& exp : BoysRegionBExps())
        {
            exponentials.push_back(exp.name != nullptr ? exp.name : "(unnamed)");
        }

        for (const FitGranularityInfo& partition : report.granularities)
        {
            partitions.push_back(partition.name);
        }

        text += Text("  axes: %zu route(s) (%s) | %zu scheme(s) (%s) | %zu partition(s) (%s) |\n",
                     routes.size(),
                     Joined(routes).c_str(),
                     schemes.size(),
                     Joined(schemes).c_str(),
                     partitions.size(),
                     Joined(partitions).c_str());
        text += Text("        %zu packing axis/axes (%s) | %zu division form(s) (%s) | "
                     "%zu exponential(s) (%s),\n",
                     axes.size(),
                     Joined(axes).c_str(),
                     forms.size(),
                     Joined(forms).c_str(),
                     exponentials.size(),
                     Joined(exponentials).c_str());
        text += "        each class's cells counted above\n";
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
        // A partition that stores one table over the whole of the interval it
        // serves has no region-B seed beside it, and its region-A fields are
        // that one table's own: printing them under the derived partitions'
        // labels would report a grid's intervals as region A's pieces and a
        // table that is not there as a region-B count of zero. The row says
        // which of the two shapes it is, and the interval it holds on, here
        // rather than in the prose below.
        const bool oneTable = partition.regionBPieces == 0 && partition.regionBStored == 0;

        if (oneTable)
        {
            text += Text("    %-8s axes=%s%d | one table over x in [%.4g, %.4g): "
                         "%d interval(s) deg %d (%d stored) | no region-B table of its own\n",
                         partition.name,
                         FitGranularityHasAxis(partition, PackAxis::kArguments) ? "arguments+" : "",
                         FitGranularityHasAxis(partition, PackAxis::kOrders) ? 1 : 0,
                         partition.lo, partition.hi, partition.regionAPieces, partition.regionADeg,
                         partition.regionAStored);
        }
        else
        {
            text += Text("    %-8s axes=%s%d | region A %d piece(s) deg %d (%d stored) | "
                         "region B %d piece(s) deg %d (%d stored)\n",
                         partition.name,
                         FitGranularityHasAxis(partition, PackAxis::kArguments) ? "arguments+" : "",
                         FitGranularityHasAxis(partition, PackAxis::kOrders) ? 1 : 0,
                         partition.regionAPieces, partition.regionADeg, partition.regionAStored,
                         partition.regionBPieces, partition.regionBDeg, partition.regionBStored);
        }

        text += Text("             route(s) %s | stored fits: delivered %.3g | bound %.3g, on "
                     "x in [%.4g, %.4g) alone\n",
                     PartitionRouteNames(partition).c_str(), partition.delivered, partition.bound,
                     partition.lo, partition.hi);
    }

    for (const OptionPrecision precision : kCellPrecisions)
    {
        std::vector<std::string> servedCells;

        for (const OptionProbeCell& cell : report.cells)
        {
            if (cell.precision == precision && cell.served)
            {
                servedCells.push_back(cell.name);
            }
        }

        text += Text("  served cells of the %s book (%zu):\n", PrecisionName(precision),
                     servedCells.size());
        text += WrappedList(servedCells, "    ");
    }

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
        if (entry.precision == OptionPrecision::kFp64 && entry.shape == kWorkloadShape)
        {
            doublesClass = &entry;
        }
    }

    text += "\nrecommendation\n";
    text += Text("  the %s class — the certified double lane's precision at the library's own\n",
                 doublesClass != nullptr ? doublesClass->name.c_str() : "fp64 all-orders");
    text += "  full-accuracy multiplier, answering this workload's own question shape (one "
            "argument per\n";
    text += "  call, that argument's own ladder), every row of it built at that multiplier, so "
            "nothing in it\n";
    text += "  traded accuracy for speed and nothing in it answers a different question. This is "
            "the pool\n";
    text += "  the default is taken from, and these are its rows:\n";

    for (const OptionProbeMeasurement& measurement : report.measurements)
    {
        if (measurement.precision != OptionPrecision::kFp64 || measurement.shape != kWorkloadShape)
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
    // from: that class's own second-fastest row by this run's statistic. A figure
    // from another precision or another question shape belongs to a different
    // class, so none of them is a rival to a row of this one.
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
                    measurement.shape != kWorkloadShape || measurement.nsPerArgument <= 0.0 ||
                    measurement.name == chosen->name)
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
                const bool sameClass = measurement.precision == OptionPrecision::kFp64 &&
                                       measurement.shape == kWorkloadShape;

                text += Text("  fastest measured overall: %s at %.2f ns/argument, documented at "
                             "%.3g%s\n",
                             measurement.name.c_str(), measurement.nsPerArgument, measurement.bound,
                             sameClass ? ""
                                       : " — another precision or another question shape: faster, "
                                         "not faster at the same accuracy for the same question");
            }
        }
    }

    // The refinement stage and its vote, printed in full: a default reached this
    // way rests on the runs, so the runs are what the reader is given.
    if (!report.refinements.empty())
    {
        text += "\n  the refinement stages — the options a class left tied, re-run alone and "
                "voted on\n";
        text += "    One stage per class whose own rounds could not order its pool, so a\n";
        text += "    stage's tied set is the class's own and its vote settles that class and no "
                "other.\n";

        for (const OptionProbeRefinement& stage : report.refinements)
        {
            text += Text("    %s %s: %d run(s) of %d passes by %d rounds each, over %zu tied "
                         "option(s)\n",
                         PrecisionName(stage.precision), OptionProbeShapeName(stage.shape),
                         stage.runs, stage.passes, stage.rounds, stage.pool.size());
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
        // The row the class's own refinement stage named, when that is not the row
        // this report names: the two differ exactly when the run could not separate
        // them.
        const OptionProbeRefinement* tiedStage = nullptr;

        for (const OptionProbeRefinement& stage : report.refinements)
        {
            if (stage.precision == OptionPrecision::kFp64 && stage.shape == kWorkloadShape)
            {
                tiedStage = &stage;
            }
        }

        const bool voteNamesAnother = tiedStage != nullptr && tiedStage->ran &&
                                      !tiedStage->winner.empty() &&
                                      tiedStage->winner != report.recommended;

        text += Text("  default: %s\n", report.recommended.c_str());

        // Printed before the way-it-was-reached, because it qualifies it: a
        // default named from a class that held no row for a member of an axis is
        // not the winner of that axis, and the reader has to meet that before
        // reading how the row was reached rather than after.
        for (const std::string& absent : report.defaultClassAbsent)
        {
            text += Text("    NOT COMPARED: %s\n", absent.c_str());
        }

        if (report.defaultHow == OptionProbeDefaultHow::kOrdered)
        {
            text += "    reached by: a measured ordering of equals. This is the fastest option of "
                    "the class\n";
            text += "                by it, and nothing in the class traded accuracy for its place\n";
        } else if (report.defaultHow == OptionProbeDefaultHow::kOnlyEntry)
        {
            text += "    reached by: the only entry of its class, named by there being no "
                    "alternative. This is\n";
            text += "                not a ranking: nothing in the class was measured against it\n";
        } else if (voteNamesAnother)
        {
            double votedFigure = 0.0;

            for (const OptionProbeMeasurement& measurement : report.measurements)
            {
                if (measurement.measured && measurement.name == tiedStage->winner)
                {
                    votedFigure = measurement.nsPerArgument;
                }
            }

            text += Text("    reached by: %s. The class's top entries could not be separated by "
                         "this run: the\n",
                         HowText(report.defaultHow));
            text += Text("                refinement runs named '%s' at %.2f ns/argument, and the "
                         "report's own figures put\n",
                         tiedStage->winner.c_str(), votedFigure);
            text += Text("                '%s' first at %.2f. Neither is offered as the "
                         "fastest — what this run measured\n",
                         report.recommended.c_str(),
                         doublesClass != nullptr ? doublesClass->leaderNsPerArgument : 0.0);
            text += "                about the two is that it could not tell them apart\n";
        } else if (report.defaultHow == OptionProbeDefaultHow::kRefined ||
                   report.defaultHow == OptionProbeDefaultHow::kVote ||
                   report.defaultHow == OptionProbeDefaultHow::kChosenAmongEquals)
        {
            text += Text("    reached by: %s. This is not a measured ordering: the options the "
                         "class left tied\n",
                         HowText(report.defaultHow));
            text += "                were re-run alone and the run above says which of them this "
                    "is\n";
        } else
        {
            // The sentence above states a tie and a re-run of it. A report whose own field is
            // none of the ways this revision names has neither, and printing it here would state
            // a run that did not happen.
            text += "    reached by: (not a way this revision names) — this report's own field is "
                    "not one of\n";
            text += "                the enumeration's members, so there is no way to state here\n";
        }
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
    text += "  are used rather than read from documentation. A figure above is the option's "
            "within-round\n";
    text += "  ratio to the reference lane at the middle of the run's rounds, scaled by that lane's "
            "own cost:\n";
    text += "  a paired reading, so no clock drift common to a round is in it, and a central one, "
            "so it\n";
    text += "  does not credit whichever row the run anchored on. Its spread is printed beside it, "
            "and the\n";
    text += "  peak column bounds what each option can do at a peak clock and is never the option's "
            "cost.\n";
    text += "  The resolution above is this run's: a steadier machine would order pairs this one "
            "could not.\n";

    // The refusals, grouped by the axis that refuses them, counted from the
    // coverage book rather than written here: the three ways a cell of the
    // product can be unserved are read off the partition rows the library
    // publishes, so a build that grows one of them says so by itself.
    //
    // The rows are the double lane's, so the census is taken over that lane's
    // book alone; the other lanes' refusals are named with the library's own
    // reason in the coverage above.
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
    std::size_t refusedFiner = 0;
    std::size_t doubleCells = 0;

    for (const OptionProbeCell& cell : report.cells)
    {
        if (cell.precision != OptionPrecision::kFp64)
        {
            continue;
        }

        ++doubleCells;

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
        } else
        {
            // A limit finer than the partition row's own fields can state. The row
            // says which routes and axes the partition carries; it cannot say that one
            // lane of a route it carries is instantiated over another partition's fits.
            // Those cells carry the library's own reason in the coverage above, and this
            // count keeps the summary's arithmetic equal to the refusals it describes.
            ++refusedFiner;
        }
    }

    // The space counted, taken before the list below so that the debt this section names in
    // prose carries the number the closure prints below rather than a sentence about it. It
    // is the same closure the last block of the report prints.
    const OptionProbeClosure closure = OptionProbeSpaceClosure(report);

    // And the seam this run implies, named before the space is closed: what a run would
    // write is a reading of the same rankings the closure counts.
    AppendDefaultsBlock(text, report);

    text += "\nnot measured by this probe, by design — listed so that nothing above is read as a "
            "complete\n";
    text += "  account of the library:\n";
    text += "  * the native half lane (region C only, orders 0..8, results scaled by a power of "
            "two — a\n";
    text += "    different question from this workload's);\n";
    text += "  * the region-A matrix-product transform (region A alone, so measuring it means "
            "composing the\n";
    text += "    rest of an evaluation, and the composition would be what got measured);\n";
    text += "  * the CUDA lanes (a separate optional build, and a lane this probe reaches with no "
            "arm at\n";
    text += "    all: `benchmarks/boys_option_probe.cpp` names no CUDA entry and this probe calls\n";
    text += "    none, so a row of that lane would be a host entry's figures under a device "
            "name);\n";
    text += "  * a partition's stored fits read directly: the figure a partition row prints beside "
            "its\n";
    text += "    verdict is the library's published figure for that partition's fits, and every row "
            "above\n";
    text += "    is an entry read end to end, so nothing here measures a fit's own error against "
            "the true\n";
    text += "    function over the interval it is cut for — the library's accuracy gate measures "
            "that book;\n";
    if (refusedRoute + refusedAxis + refusedFiner > 0)
    {
        text += Text("  * the %zu cells of the double lane's book this build refuses, of %zu, "
                     "counted by what\n    that lane's partition rows state of themselves: %zu on "
                     "the rational route (a table\n    to generate), %zu on the across-orders "
                     "packing axis (a kernel to write), and %zu on a\n",
                     refusedRoute + refusedAxis + refusedFiner, doubleCells,
                     refusedRoute, refusedAxis, refusedFiner);
        text += "    limit finer than those fields state — an across-orders packed lane "
                "instantiated over\n    the shipped fits rather than the partition's own (a body to "
                "write). Each is named\n    with the library's own reason in the coverage above: "
                "they are unbuilt work, counted\n    rather than absent, and no cell of the "
                "library's product is missing from this report;\n";
    }
    text += "  * the call shapes other than this workload's all-orders-per-argument one, which the "
            "seven axes\n";
    text += "    are not crossed with. Every other shape the seam's key names is measured at its "
            "class's\n";
    text += "    own default policy alone - one row per class, and no crossing of the axes - so "
            "the\n";
    text += "    all-N grouping and its sorted-argument overload, the fixed-order call that "
            "returns one\n";
    text += "    order across an array, the array entry that reads a per-argument order, and the\n";
    text += "    single-order call at each format the library declares it for are all measured "
            "above.\n";
    text += "    Those rows are no cell of the space, and they are counted in the closure below,\n";
    text += Text("    which names them and counts the crossing they stand for: %zu row(s),\n"
                 "    standing for the %zu cell(s) of their own class that crossing them\n"
                 "    with the axes would add beyond the cells they stand at.\n",
                 closure.shapesNotCrossed, closure.crossedOwed);
    text += "    Crossing them with the axes is outstanding work and not an impossibility: an "
            "entry is a\n";
    text += "    function, and no table of entries exists in this library for one to be counted "
            "from. A\n";
    text += "    class the seam's key reaches that this run produced no figure for is named in the\n";
    text += "    build-defaults block above, with the reason this run has for it. What the closure\n";
    text += "    does count is every row this report carries and the cells it stands for. The axes "
            "are\n";
    text += "    crossed on every precision class: the routes each class's own lane reports its "
            "fits\n";
    text += "    in, the schemes, partitions, packing axes and division forms of that lane and the\n";
    text += "    exponentials its region-B ladders can be seeded with, one row per served cell of a\n";
    text += "    class this machine can run.\n";

    // The last block of the report, on every path: the space counted, so that what this run
    // did with it is a number a reader can check rather than an inference from the sections
    // above.
    AppendOptionClosure(text, report, closure);

    return text;
}

} // namespace boys
