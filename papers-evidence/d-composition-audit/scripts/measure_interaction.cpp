// Composed-bounds interaction measurement against a pinned boys revision.
//
// For every host precision, every served combination of the six call axes, and
// every (n, x) on the argument grid, this driver records the delivered value,
// the high-precision reference, the published figure and the two normalized
// errors the summary analysis needs. It is a measurement, not a test: it prints
// no verdict, and a figure the measured error exceeds is recorded, not judged.
//
// The grid mixes exactly representable arguments with generic decimals that no
// measured format represents, because exact arguments can suppress the rounding
// interaction the probe exists to measure. The lane's actual input is not the
// decimal: each lane narrows it to its own format first. The driver therefore
// emits that narrowed input per lane (--emit-arguments), the reference generator
// computes F_n at each lane's exact narrowed value, and the measurement looks the
// reference up by (lane, x). That keeps argument narrowing out of the arithmetic
// comparison instead of confounding it.
//
// The cell dispatch mirrors the option probe's own (src/boys_probe.cpp,
// CellEntry and CellEntrySingle): the axes are compile-time policy arguments,
// so a cell is evaluated through its own policy and never through another
// cell's arithmetic.
//
// The canary witnesses machine quietness for a timing-sensitive run. It says
// nothing about numerical validity: the error rows here are deterministic and
// do not depend on the canary.

// The measurement names every table group the library stores, so the umbrella
// comes first: its coefficient group headers must precede the policy
// instantiations in the headers below. This is the option probe's own include
// order (src/boys_probe.cpp).
#include "boys/boys_all.hpp"

#include "boys/backend.hpp"
#include "boys/boys.hpp"
#include "boys/policy.hpp"
#include "boys/version.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <iterator>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace boys;

// ---------------------------------------------------------------------------
// The argument grid: exact values and generic decimals.
// ---------------------------------------------------------------------------

constexpr int kOrders = kMaxBoysOrder;
constexpr std::size_t kStride = static_cast<std::size_t>(kOrders) + 1;

struct GridValue {
    double nominal;
    const char* kind;
};

// The runner may name a path whose directory does not exist yet (the manual
// command sequence in the plan writes into results/). Creating it here keeps
// the driver's own commands runnable without a preceding mkdir; a failure is
// left to fopen, which reports the path.
void EnsureParentDirectory(const char* path) noexcept
{
    std::error_code error;
    const std::filesystem::path parent = std::filesystem::path(path).parent_path();

    if (!parent.empty())
    {
        std::filesystem::create_directories(parent, error);
    }
}

// The exact values have been the probe's grid since the design began; the
// generic decimals are not exactly representable in any measured format, so
// they exercise the rounding a composition of bounds must survive. They span
// region A, the region-A/region-B boundary (kX0), and region B.
constexpr GridValue kGrid[] = {
    {0.0, "exact"},     {0.125, "exact"}, {0.25, "exact"},  {0.5, "exact"},
    {1.0, "exact"},     {1.125, "exact"}, {2.0, "exact"},   {4.0, "exact"},
    {8.0, "exact"},     {11.5, "exact"},  {12.0, "exact"},  {16.0, "exact"},
    {24.0, "exact"},    {29.0, "exact"},  {30.5, "exact"},  {32.0, "exact"},
    {0.1, "generic"},   {0.7, "generic"}, {1.3, "generic"}, {2.7, "generic"},
    {7.3, "generic"},   {11.7, "generic"}, {12.7, "generic"}, {17.3, "generic"},
    {24.7, "generic"},  {30.3, "generic"},
};

// The smoke grid exists so the driver and the reference generator can be
// exercised end to end without a quiet machine. It is four arguments, not the
// probe's 26, and a smoke run is a functional check of the harness: it reads no
// canary, its rows are not a measurement, and the window run never passes
// --smoke.
constexpr GridValue kSmokeGrid[] = {
    {1.0, "exact"}, {11.7, "generic"}, {12.7, "generic"}, {24.0, "exact"},
};

struct ArgumentRow {
    double nominal = 0.0;
    double x = 0.0;
    std::string kind;
};

struct RefGroup {
    double x = 0.0;
    std::array<double, kStride> hi{};
    std::array<double, kStride> lo{};
    std::array<bool, kStride> seen{};
};

struct LaneData {
    Precision precision = Precision::kFp64;
    const char* label = "";
    std::vector<ArgumentRow> rows;
    std::vector<RefGroup> refs;
};

struct Cell {
    Precision precision;
    FitRoute route;
    EvalScheme scheme;
    FitGranularity granularity;
    PackAxis pack;
    DivisionForm form;
    RegionBExp exp;
};

// Six cells per host lane: both routes, both schemes, all three partitions,
// both packing axes, all three forms and both exponential members appear. Not a
// fractional design and not a result; a coverage check for the driver's
// instantiation and dispatch paths.
constexpr Cell kSmokeCells[] = {
    {Precision::kFp64, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,
     FitGranularity::kCoarsest, PackAxis::kArguments, DivisionForm::kExactDivision,
     RegionBExp::kAccurate},
    {Precision::kFp64, FitRoute::kRationalMinimax, EvalScheme::kHorner,
     FitGranularity::kNarrow, PackAxis::kOrders, DivisionForm::kPlainReciprocal,
     RegionBExp::kFast},
    {Precision::kFp64, FitRoute::kChebyshev, EvalScheme::kHorner, FitGranularity::kUniform,
     PackAxis::kArguments, DivisionForm::kRefinedReciprocal, RegionBExp::kFast},
    {Precision::kFp32, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,
     FitGranularity::kCoarsest, PackAxis::kArguments, DivisionForm::kExactDivision,
     RegionBExp::kAccurate},
    {Precision::kFp32, FitRoute::kRationalMinimax, EvalScheme::kHorner,
     FitGranularity::kNarrow, PackAxis::kOrders, DivisionForm::kPlainReciprocal,
     RegionBExp::kFast},
    {Precision::kFp32, FitRoute::kChebyshev, EvalScheme::kHorner, FitGranularity::kUniform,
     PackAxis::kArguments, DivisionForm::kRefinedReciprocal, RegionBExp::kFast},
    {Precision::kFp16, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,
     FitGranularity::kCoarsest, PackAxis::kArguments, DivisionForm::kExactDivision,
     RegionBExp::kAccurate},
    {Precision::kBf16, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,
     FitGranularity::kCoarsest, PackAxis::kArguments, DivisionForm::kExactDivision,
     RegionBExp::kAccurate},
};

struct Context {
    std::FILE* out = nullptr;
    std::size_t rows = 0;
    std::size_t refused = 0;
    std::size_t over_bound = 0;
};

// ---------------------------------------------------------------------------
// Lane tags: the argument type and the all-orders entry of each host lane.
// ---------------------------------------------------------------------------

template <Precision kPrecision>
struct Lane;

template <>
struct Lane<Precision::kFp64> {
    using Io = double;

    static Io Narrow(double x) noexcept { return x; }
    static double Widen(Io v) noexcept { return v; }

    template <class Policy>
    static void AllOrders(int n, Io x, Io* out) noexcept
    {
        BoysAllOrders<Policy>(n, x, out);
    }
};

template <>
struct Lane<Precision::kFp32> {
    using Io = float;

    static Io Narrow(double x) noexcept { return static_cast<float>(x); }
    static double Widen(Io v) noexcept { return static_cast<double>(v); }

    template <class Policy>
    static void AllOrders(int n, Io x, Io* out) noexcept
    {
        BoysAllOrdersF32<Policy>(n, x, out);
    }
};

#if BoysFp16
template <>
struct Lane<Precision::kFp16> {
    using Io = F16;

    static Io Narrow(double x) noexcept { return static_cast<Io>(static_cast<float>(x)); }
    static double Widen(Io v) noexcept { return static_cast<double>(static_cast<float>(v)); }

    template <class Policy>
    static void AllOrders(int n, Io x, Io* out) noexcept
    {
        BoysAllOrdersF16<Policy>(n, x, out);
    }
};

template <>
struct Lane<Precision::kBf16> {
    using Io = Bf16;

    static Io Narrow(double x) noexcept { return static_cast<Io>(static_cast<float>(x)); }
    static double Widen(Io v) noexcept { return static_cast<double>(static_cast<float>(v)); }

    template <class Policy>
    static void AllOrders(int n, Io x, Io* out) noexcept
    {
        BoysAllOrdersBf16<Policy>(n, x, out);
    }
};
#endif // BoysFp16

template <Precision kPrecision>
constexpr const char* PrecisionLabel() noexcept;

template <>
constexpr const char* PrecisionLabel<Precision::kFp64>() noexcept
{
    return "fp64";
}

template <>
constexpr const char* PrecisionLabel<Precision::kFp32>() noexcept
{
    return "fp32";
}

#if BoysFp16
template <>
constexpr const char* PrecisionLabel<Precision::kFp16>() noexcept
{
    return "fp16";
}

template <>
constexpr const char* PrecisionLabel<Precision::kBf16>() noexcept
{
    return "bf16";
}
#endif

constexpr const char* RouteLabel(FitRoute route) noexcept
{
    switch (route)
    {
    case FitRoute::kChebyshev:
        return "chebyshev";
    case FitRoute::kRationalMinimax:
        return "rational-minimax";
    }

    return "unknown";
}

const char* PrecisionLabelForRuntime(Precision precision) noexcept
{
    switch (precision)
    {
    case Precision::kFp64:
        return "fp64";
    case Precision::kFp32:
        return "fp32";
    case Precision::kFp16:
        return "fp16";
    case Precision::kBf16:
        return "bf16";
    case Precision::kFp32Device:
        return "fp32-device";
    case Precision::kFp64Device:
        return "fp64-device";
    case Precision::kFp16Device:
        return "fp16-device";
    case Precision::kBf16Device:
        return "bf16-device";
    }

    return "unknown";
}

// ---------------------------------------------------------------------------
// Arguments: emission per lane, and loading.
// ---------------------------------------------------------------------------

bool PrecisionFromLabel(const std::string& label, Precision& out) noexcept
{
    if (label == "fp64")
    {
        out = Precision::kFp64;
        return true;
    }
    if (label == "fp32")
    {
        out = Precision::kFp32;
        return true;
    }
    if (label == "fp16")
    {
        out = Precision::kFp16;
        return true;
    }
    if (label == "bf16")
    {
        out = Precision::kBf16;
        return true;
    }
    return false;
}

void EmitArguments(const char* path, bool smoke)
{
    EnsureParentDirectory(path);
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "measure-interaction: cannot write arguments '%s'\n", path);
        std::exit(2);
    }

    std::fputs("lane,x_nominal,x,kind\n", f);

    auto emit_lane = [f, smoke]<Precision kP>() {
        for (const GridValue& g : smoke ? std::span<const GridValue>(kSmokeGrid)
                                        : std::span<const GridValue>(kGrid))
        {
            const double narrowed = Lane<kP>::Widen(Lane<kP>::Narrow(g.nominal));
            std::fprintf(f, "%s,%.17g,%.17g,%s\n", PrecisionLabel<kP>(), g.nominal, narrowed,
                         g.kind);
        }
    };

    emit_lane.template operator()<Precision::kFp64>();
    emit_lane.template operator()<Precision::kFp32>();
#if BoysFp16
    emit_lane.template operator()<Precision::kFp16>();
    emit_lane.template operator()<Precision::kBf16>();
#endif

    std::fclose(f);
}

std::vector<LaneData> LoadArguments(const char* path)
{
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "measure-interaction: cannot open arguments '%s'\n", path);
        std::exit(2);
    }

    std::array<LaneData, 4> lanes{};
    std::array<bool, 4> present{};
    constexpr Precision kOrder[4] = {Precision::kFp64, Precision::kFp32, Precision::kFp16,
                                     Precision::kBf16};

    for (int i = 0; i < 4; ++i)
    {
        lanes[static_cast<std::size_t>(i)].precision = kOrder[i];
        lanes[static_cast<std::size_t>(i)].label = PrecisionLabelForRuntime(kOrder[i]);
    }

    char line[512];
    bool header = true;

    while (std::fgets(line, sizeof(line), f) != nullptr)
    {
        if (header)
        {
            header = false;
            continue;
        }

        char lane_label[32] = {};
        char kind[32] = {};
        double nominal = 0.0;
        double x = 0.0;
        const int read = std::sscanf(line, "%31[^,],%lf,%lf,%31s", lane_label, &nominal, &x, kind);

        if (read != 4)
        {
            std::fprintf(stderr, "measure-interaction: bad arguments row '%s'\n", line);
            std::exit(2);
        }

        Precision precision{};
        if (!PrecisionFromLabel(lane_label, precision))
        {
            std::fprintf(stderr, "measure-interaction: unknown lane '%s'\n", lane_label);
            std::exit(2);
        }
        if (!(x >= 0.0) || !std::isfinite(x) || !std::isfinite(nominal))
        {
            std::fprintf(stderr, "measure-interaction: invalid argument %.17g\n", x);
            std::exit(2);
        }

        const std::string kind_text = kind;
        if (kind_text != "exact" && kind_text != "generic")
        {
            std::fprintf(stderr, "measure-interaction: unknown argument kind '%s'\n", kind);
            std::exit(2);
        }

        std::size_t index = 0;
        while (index < 4 && kOrder[index] != precision)
        {
            ++index;
        }

        LaneData& lane = lanes[index];
        for (const ArgumentRow& row : lane.rows)
        {
            if (row.x == x)
            {
                std::fprintf(stderr,
                             "measure-interaction: duplicate argument in lane %s: %.17g\n",
                             lane.label, x);
                std::exit(2);
            }
        }

        lane.rows.push_back(ArgumentRow{nominal, x, kind_text});
        present[index] = true;
    }

    std::fclose(f);

    if (!present[0] || !present[1])
    {
        std::fprintf(stderr, "measure-interaction: arguments must carry fp64 and fp32 lanes\n");
        std::exit(2);
    }

    const std::size_t count = lanes[0].rows.size();

    // Validate every present lane against the fp64 grid before any lane is moved
    // out: the move empties its source, so a validation loop that interleaved
    // the two would read a moved-from vector.
    for (int i = 0; i < 4; ++i)
    {
        const LaneData& lane = lanes[static_cast<std::size_t>(i)];

        if (!present[static_cast<std::size_t>(i)])
        {
            continue;
        }
        if (lane.rows.size() != count)
        {
            std::fprintf(stderr, "measure-interaction: lane %s has %zu arguments, fp64 has %zu\n",
                         lane.label, lane.rows.size(), count);
            std::exit(2);
        }
        for (std::size_t k = 0; k < count; ++k)
        {
            if (lane.rows[k].nominal != lanes[0].rows[k].nominal ||
                lane.rows[k].kind != lanes[0].rows[k].kind)
            {
                std::fprintf(stderr,
                             "measure-interaction: lane %s argument %zu is not the fp64 grid "
                             "entry\n",
                             lane.label, k);
                std::exit(2);
            }
        }
    }

    std::vector<LaneData> out;
    out.reserve(4);
    for (int i = 0; i < 4; ++i)
    {
        if (present[static_cast<std::size_t>(i)])
        {
            out.push_back(std::move(lanes[static_cast<std::size_t>(i)]));
        }
    }

    return out;
}

// ---------------------------------------------------------------------------
// Canary: the machine rule requires a quiet machine, alone. This is a
// fixed-work spin, not a library timing; its spread across repeated samples is
// the quietness witness, and a wide spread stops the run before any cell is
// measured. The 5% alarm is the library probe's own default
// (include/boys/boys_probe.hpp:133, canarySpreadAlarm = 5.0). The canary
// witnesses timing/quietness only: the error rows are deterministic and do not
// depend on it.
// ---------------------------------------------------------------------------

constexpr int kCanarySamples = 8;
constexpr std::uint64_t kCanaryIterations = 20000000;
constexpr double kCanarySpreadAlarmPct = 5.0;

struct Canary {
    double min_ms = 0.0;
    double median_ms = 0.0;
    double max_ms = 0.0;
    double spread_pct = 0.0;
    double sink = 0.0;
};

double CanarySpin(std::uint64_t iterations) noexcept
{
    double a = 1.0;
    double b = 0.5;

    for (std::uint64_t i = 0; i < iterations; ++i)
    {
        a = a * 1.0000000000000002 + b * 1e-17;
        b = b * 0.9999999999999999 + a * 1e-17;
    }

    return a + b;
}

Canary RunCanary(int samples, std::uint64_t iterations)
{
    std::vector<double> ms;
    ms.reserve(static_cast<std::size_t>(samples));

    Canary c;
    for (int s = 0; s < samples; ++s)
    {
        const auto start = std::chrono::steady_clock::now();
        c.sink += CanarySpin(iterations);
        const auto stop = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    std::sort(ms.begin(), ms.end());
    c.min_ms = ms.front();
    c.max_ms = ms.back();
    c.median_ms = ms[ms.size() / 2];
    c.spread_pct = c.median_ms > 0.0 ? (c.max_ms - c.min_ms) / c.median_ms * 100.0 : 0.0;
    return c;
}

void PrintCanary(const char* when, const Canary& c, bool wide)
{
    std::printf("canary %s: min %.3f ms, median %.3f ms, max %.3f ms, spread %.2f%%%s\n",
                when, c.min_ms, c.median_ms, c.max_ms, c.spread_pct,
                wide ? " (WIDE, above the 5% alarm)" : "");
}

// ---------------------------------------------------------------------------
// Reference: F_n at each lane's exact narrowed argument.
// ---------------------------------------------------------------------------

RefGroup& FindOrAddGroup(LaneData& lane, double x)
{
    for (RefGroup& group : lane.refs)
    {
        if (group.x == x)
        {
            return group;
        }
    }

    lane.refs.push_back(RefGroup{});
    lane.refs.back().x = x;
    return lane.refs.back();
}

void AttachReference(const char* path, std::vector<LaneData>& lanes)
{
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "measure-interaction: cannot open reference '%s'\n", path);
        std::exit(2);
    }

    char line[512];
    bool header = true;

    while (std::fgets(line, sizeof(line), f) != nullptr)
    {
        if (header)
        {
            header = false;
            continue;
        }

        char lane_label[32] = {};
        int n = 0;
        double x = 0.0;
        double hi = 0.0;
        double lo = 0.0;
        const int read =
            std::sscanf(line, "%31[^,],%d,%lf,%lf,%lf", lane_label, &n, &x, &hi, &lo);

        if (read != 5)
        {
            std::fprintf(stderr, "measure-interaction: bad reference row '%s'\n", line);
            std::exit(2);
        }
        if (n < 0 || n > kOrders)
        {
            std::fprintf(stderr, "measure-interaction: reference order %d is outside 0..%d\n", n,
                         kOrders);
            std::exit(2);
        }

        Precision precision{};
        if (!PrecisionFromLabel(lane_label, precision))
        {
            std::fprintf(stderr, "measure-interaction: unknown reference lane '%s'\n", lane_label);
            std::exit(2);
        }

        LaneData* lane = nullptr;
        for (LaneData& candidate : lanes)
        {
            if (candidate.precision == precision)
            {
                lane = &candidate;
                break;
            }
        }
        if (lane == nullptr)
        {
            std::fprintf(stderr,
                         "measure-interaction: reference carries lane %s that the arguments "
                         "do not\n",
                         lane_label);
            std::exit(2);
        }

        RefGroup& group = FindOrAddGroup(*lane, x);
        if (group.seen[static_cast<std::size_t>(n)])
        {
            std::fprintf(stderr, "measure-interaction: duplicate reference at (%s, %.17g, %d)\n",
                         lane_label, x, n);
            std::exit(2);
        }
        group.seen[static_cast<std::size_t>(n)] = true;
        group.hi[static_cast<std::size_t>(n)] = hi;
        group.lo[static_cast<std::size_t>(n)] = lo;
    }

    std::fclose(f);

    std::size_t values = 0;
    for (LaneData& lane : lanes)
    {
        for (const ArgumentRow& row : lane.rows)
        {
            const RefGroup* group = nullptr;
            for (const RefGroup& candidate : lane.refs)
            {
                if (candidate.x == row.x)
                {
                    group = &candidate;
                    break;
                }
            }
            if (group == nullptr)
            {
                std::fprintf(stderr,
                             "measure-interaction: no reference for lane %s argument %.17g\n",
                             lane.label, row.x);
                std::exit(2);
            }
            for (std::size_t n = 0; n < kStride; ++n)
            {
                if (!group->seen[n])
                {
                    std::fprintf(stderr,
                                 "measure-interaction: reference lacks order %zu at (%s, %.17g)\n",
                                 n, lane.label, row.x);
                    std::exit(2);
                }
            }
            values += kStride;
        }
    }

    std::printf("reference: %zu values across %zu lane-arguments\n", values,
                [&] {
                    std::size_t total = 0;
                    for (const LaneData& lane : lanes)
                    {
                        total += lane.rows.size();
                    }
                    return total;
                }());
}

const RefGroup& FindGroup(const LaneData& lane, double x)
{
    for (const RefGroup& group : lane.refs)
    {
        if (group.x == x)
        {
            return group;
        }
    }

    std::fprintf(stderr, "measure-interaction: no reference group at %.17g\n", x);
    std::exit(2);
}

void WriteEnvironment(const char* path, int canary_samples, std::uint64_t canary_iterations,
                      const Canary& pre, const Canary& post, bool smoke)
{
    EnsureParentDirectory(path);
    std::FILE* f = std::fopen(path, "wb");
    if (f == nullptr)
    {
        std::fprintf(stderr, "measure-interaction: cannot write environment '%s'\n", path);
        std::exit(2);
    }

    const std::time_t now = std::time(nullptr);
    char stamp[64] = {};
#if defined(_MSC_VER)
    std::tm utc{};
    gmtime_s(&utc, &now);
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", &utc);
#else
    std::tm* utc = std::gmtime(&now);
    if (utc != nullptr)
    {
        std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%SZ", utc);
    }
#endif

    std::fprintf(f, "boys_version=%s\n", VersionString());
    std::fprintf(f, "seam_identity=%s\n", BuildDefaultsSeamIdentity());
#if BoysFp16
    std::fprintf(f, "boys_fp16_seam=1\n");
#else
    std::fprintf(f, "boys_fp16_seam=0\n");
#endif
    // The route in force is what the arithmetic delivers; the selection is what
    // the build asked for. They differ exactly when a contracting build compiled
    // the separate route into fused steps, which is the fact a stability run must
    // be able to read back rather than infer from the option's spelling.
    std::fprintf(f, "muladd_fp64=%s\n",
                 backend::MulAddRouteName(backend::detail::RouteInForce<double>()));
    std::fprintf(f, "muladd_fp32=%s\n",
                 backend::MulAddRouteName(backend::detail::RouteInForce<float>()));
    std::fprintf(f, "muladd_selected_fp64=%s\n",
                 backend::MulAddRouteName(backend::Scalar<double>::kRoute));
    std::fprintf(f, "muladd_selected_fp32=%s\n",
                 backend::MulAddRouteName(backend::Scalar<float>::kRoute));
#if defined(BOYS_SCALAR_CONTRACTS)
    std::fprintf(f, "scalar_contracts=%d\n", BOYS_SCALAR_CONTRACTS);
#else
    std::fprintf(f, "scalar_contracts=unmeasured\n");
#endif
    std::fprintf(f, "smoke=%d\n", smoke ? 1 : 0);
    if (smoke)
    {
        std::fprintf(f, "arguments=smoke grid, 4 values (2 exact + 2 generic); not the probe\n");
        std::fprintf(f, "canary=not run (functional smoke, no quiet machine required)\n");
        std::fprintf(f, "canary_verdict=not-run-smoke\n");
        std::fprintf(f, "canary_witnesses=none; this file records a harness smoke, not a "
                        "measurement\n");
        std::fprintf(f, "machine_rule=smoke only; the window run never passes --smoke\n");
    }
    else
    {
        std::fprintf(f, "arguments=16 exact + 10 generic (see arguments.csv)\n");
    }
    std::fprintf(f, "orders=0..%d\n", kMaxBoysOrder);
    std::fprintf(f, "run_started_utc=%s\n", stamp);
    if (!smoke)
    {
        std::fprintf(f, "machine_rule=quiet machine, alone; no build, lane, generator or other "
                        "timing run concurrently; canary is a hint under "
                        "AMENDMENT-2026-10-20\n");
        std::fprintf(f, "canary_samples=%d\n", canary_samples);
        std::fprintf(f, "canary_iterations=%llu\n",
                     static_cast<unsigned long long>(canary_iterations));
        std::fprintf(f, "canary_alarm_pct=%.1f\n", kCanarySpreadAlarmPct);
        std::fprintf(f, "canary_pre_min_ms=%.3f\ncanary_pre_median_ms=%.3f\n"
                        "canary_pre_max_ms=%.3f\ncanary_pre_spread_pct=%.2f\n",
                     pre.min_ms, pre.median_ms, pre.max_ms, pre.spread_pct);
        std::fprintf(f, "canary_post_min_ms=%.3f\ncanary_post_median_ms=%.3f\n"
                        "canary_post_max_ms=%.3f\ncanary_post_spread_pct=%.2f\n",
                     post.min_ms, post.median_ms, post.max_ms, post.spread_pct);
        std::fprintf(f, "canary_verdict=%s\n",
                     (pre.spread_pct <= kCanarySpreadAlarmPct &&
                      post.spread_pct <= kCanarySpreadAlarmPct)
                         ? "quiet"
                         : "wide");
        std::fprintf(f, "canary_witnesses=timing/quietness only; error rows are deterministic "
                        "and do not depend on the canary\n");
        std::fprintf(f, "canary_sink=%.17g\n", pre.sink + post.sink);
    }
#if defined(_MSC_VER)
    std::fprintf(f, "compiler=msvc\ncompiler_version=%d\n", _MSC_VER);
#elif defined(__VERSION__)
    std::fprintf(f, "compiler=gnu-or-clang\ncompiler_version=%s\n", __VERSION__);
#else
    std::fprintf(f, "compiler=unknown\ncompiler_version=unknown\n");
#endif
#if defined(__AVX2__)
    std::fprintf(f, "avx2_defined=1\n");
#else
    std::fprintf(f, "avx2_defined=0\n");
#endif

    std::fclose(f);
}

// ---------------------------------------------------------------------------
// One cell
// ---------------------------------------------------------------------------

// The half lanes' rows state their base and axis terms plus half of the last
// representable digit of the returned value, claimed only where the value
// exceeds the sum (src/boys.cpp:634-644 and :710-725). The accessor returns the
// base and its two axis terms; the digit is a term of the returned value and
// stays in the row's sentence. Safety is judged against the row's whole
// statement, so the digit is carried as its own column and the limit is the
// sum. Adding it where the value does not exceed the sum is weaker than the
// sentence requires, never stronger.
double HalfDigitTerm(Precision precision, double value)
{
    constexpr int kFp16LastDigit = 10;  // explicit mantissa bits
    constexpr int kBf16LastDigit = 7;
    constexpr int kFp16MinExponent = -14;   // smallest normal fp16 exponent
    constexpr int kBf16MinExponent = -126;  // smallest normal bf16 exponent

    int last_digit = 0;
    int min_exponent = 0;
    if (precision == Precision::kFp16)
    {
        last_digit = kFp16LastDigit;
        min_exponent = kFp16MinExponent;
    } else if (precision == Precision::kBf16)
    {
        last_digit = kBf16LastDigit;
        min_exponent = kBf16MinExponent;
    } else
    {
        return 0.0;
    }

    const double magnitude = std::fabs(value);
    const int exponent =
        magnitude > 0.0 ? std::ilogb(magnitude) : min_exponent - 1;
    const int half_exponent =
        std::max(exponent - last_digit - 1, min_exponent - last_digit - 1);
    return std::ldexp(1.0, half_exponent);
}

template <Precision kP, FitRoute kR, EvalScheme kS, FitGranularity kG, PackAxis kA,
          DivisionForm kD, RegionBExp kE>
void Measure(Context& ctx, const LaneData& lane)
{
    constexpr BoysBudget kBudget =
        (kP == Precision::kFp16 || kP == Precision::kBf16) ? BoysBudget::kFp16
                                                           : BoysBudget::kFloat;
    using Policy = SeamPolicy<kR, kS, kBudget, kA, kG, kD, kE>;
    using Io = typename Lane<kP>::Io;

    std::array<Io, kStride> values{};
    const char* precision = PrecisionLabel<kP>();

    for (const ArgumentRow& argument : lane.rows)
    {
        const Io xin = Lane<kP>::Narrow(argument.x);
        if (Lane<kP>::Widen(xin) != argument.x)
        {
            std::fprintf(stderr,
                         "measure-interaction: argument %.17g does not round-trip in lane %s\n",
                         argument.x, precision);
            std::exit(3);
        }

        const RefGroup& ref = FindGroup(lane, argument.x);
        Lane<kP>::template AllOrders<Policy>(kOrders, xin, values.data());

        const double f0 = ref.hi[0] + ref.lo[0];

        int pack_live = 0;
        for (const PackAxisInfo& row : BoysPackAxes())
        {
            if (row.axis == kA)
            {
                pack_live = (argument.x >= row.lo && argument.x < row.hi) ? 1 : 0;
            }
        }

        int partition_in_domain = 0;
        for (const FitGranularityInfo& row : BoysFitGranularities())
        {
            if (row.granularity == kG)
            {
                partition_in_domain = (argument.x >= row.lo && argument.x < row.hi) ? 1 : 0;
            }
        }

        for (int n = 0; n <= kOrders; ++n)
        {
            const std::size_t index = static_cast<std::size_t>(n);
            const double value = Lane<kP>::Widen(values[index]);
            const double abs_err = std::fabs((value - ref.hi[index]) - ref.lo[index]);
            const AccuracyFigure figure =
                BoysAccuracyGuaranteed(kP, n, argument.x, kR, kS, kA, kG, kD, kE);

            const double fn = ref.hi[index] + ref.lo[index];
            const double f0_ratio = f0 > 0.0 ? abs_err / f0 : 0.0;

            // `figure` is what the accessor returns. `half_digit` is the term
            // the half lanes' rows state beside it; `limit` is the row's whole
            // statement and is the safety denominator. The accessor ratio is
            // reported apart: on a half lane it is expected to exceed 1 where
            // the digit binds, and that is the contract's shape, not a bug.
            const double half_digit = HalfDigitTerm(kP, value);
            const double limit = figure.available ? figure.value + half_digit : 0.0;
            const double accessor_ratio =
                (figure.available && figure.value > 0.0) ? abs_err / figure.value : -1.0;
            const double figure_ratio =
                (figure.available && limit > 0.0) ? abs_err / limit : -1.0;

            if (!figure.available)
            {
                ++ctx.refused;
            }
            if (figure_ratio > 1.0 + 1e-9)
            {
                ++ctx.over_bound;
            }

            std::fprintf(
                ctx.out,
                "%s,%s,%s,%s,%s,%s,%s,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,"
                "%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%d,%d,%s,%s\n",
                precision,
                RouteLabel(kR),
                EvalSchemeName(kS),
                GranularityName(kG),
                PackAxisName(kA),
                DivisionFormName(kD),
                RegionBExpName(kE),
                n,
                argument.nominal,
                argument.x,
                value,
                ref.hi[index],
                ref.lo[index],
                abs_err,
                f0,
                fn,
                figure.available ? figure.value : 0.0,
                half_digit,
                limit,
                accessor_ratio,
                figure_ratio,
                pack_live,
                partition_in_domain,
                argument.kind.c_str(),
                figure.available ? "ok" : "refused");

            ++ctx.rows;
        }
    }
}

// ---------------------------------------------------------------------------
// Runtime dispatch over the six axes, in the option probe's own shape.
// ---------------------------------------------------------------------------

template <Precision kP>
void RunCellFor(Context& ctx, const Cell& c, const LaneData& lane)
{
    auto with_route = [&]<FitRoute kR>() {
        auto with_scheme = [&]<EvalScheme kS>() {
            auto with_granularity = [&]<FitGranularity kG>() {
                auto with_pack = [&]<PackAxis kA>() {
                    auto with_form = [&]<DivisionForm kD>() {
                        if (c.exp == RegionBExp::kFast)
                        {
                            Measure<kP, kR, kS, kG, kA, kD, RegionBExp::kFast>(ctx, lane);
                        } else
                        {
                            Measure<kP, kR, kS, kG, kA, kD, RegionBExp::kAccurate>(ctx, lane);
                        }
                    };

                    switch (c.form)
                    {
                    case DivisionForm::kExactDivision:
                        with_form.template operator()<DivisionForm::kExactDivision>();
                        return;
                    case DivisionForm::kPlainReciprocal:
                        with_form.template operator()<DivisionForm::kPlainReciprocal>();
                        return;
                    case DivisionForm::kRefinedReciprocal:
                        with_form.template operator()<DivisionForm::kRefinedReciprocal>();
                        return;
                    }
                };

                switch (c.pack)
                {
                case PackAxis::kArguments:
                    with_pack.template operator()<PackAxis::kArguments>();
                    return;
                case PackAxis::kOrders:
                    with_pack.template operator()<PackAxis::kOrders>();
                    return;
                }
            };

            switch (c.granularity)
            {
            case FitGranularity::kCoarsest:
                with_granularity.template operator()<FitGranularity::kCoarsest>();
                return;
            case FitGranularity::kNarrow:
                with_granularity.template operator()<FitGranularity::kNarrow>();
                return;
            case FitGranularity::kUniform:
                with_granularity.template operator()<FitGranularity::kUniform>();
                return;
            }
        };

        switch (c.scheme)
        {
        case EvalScheme::kSplitClenshaw:
            with_scheme.template operator()<EvalScheme::kSplitClenshaw>();
            return;
        case EvalScheme::kHorner:
            with_scheme.template operator()<EvalScheme::kHorner>();
            return;
        }
    };

    switch (c.route)
    {
    case FitRoute::kChebyshev:
        with_route.template operator()<FitRoute::kChebyshev>();
        return;
    case FitRoute::kRationalMinimax:
        with_route.template operator()<FitRoute::kRationalMinimax>();
        return;
    }
}

void RunCell(Context& ctx, const Cell& c, const std::vector<LaneData>& lanes)
{
    const LaneData* lane = nullptr;
    for (const LaneData& candidate : lanes)
    {
        if (candidate.precision == c.precision)
        {
            lane = &candidate;
            break;
        }
    }
    if (lane == nullptr)
    {
        return;
    }

    switch (c.precision)
    {
    case Precision::kFp64:
        RunCellFor<Precision::kFp64>(ctx, c, *lane);
        return;
    case Precision::kFp32:
        RunCellFor<Precision::kFp32>(ctx, c, *lane);
        return;
#if BoysFp16
    case Precision::kFp16:
        RunCellFor<Precision::kFp16>(ctx, c, *lane);
        return;
    case Precision::kBf16:
        RunCellFor<Precision::kBf16>(ctx, c, *lane);
        return;
#endif
    case Precision::kFp32Device:
    case Precision::kFp64Device:
    case Precision::kFp16Device:
    case Precision::kBf16Device:
        // The device lanes need a CUDA build and a device; they are a separate
        // run and are not silently mixed into the host figures.
        return;
    }

    return;
}

} // namespace

int main(int argc, char** argv)
{
    const char* emit_path = nullptr;
    const char* arguments_path = nullptr;
    const char* reference_path = nullptr;
    const char* out_path = nullptr;
    const char* env_path = nullptr;
    bool canary_only = false;
    bool smoke = false;
    int canary_samples = kCanarySamples;
    std::uint64_t canary_iterations = kCanaryIterations;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--emit-arguments" && i + 1 < argc)
        {
            emit_path = argv[++i];
        } else if (arg == "--arguments" && i + 1 < argc)
        {
            arguments_path = argv[++i];
        } else if (arg == "--reference" && i + 1 < argc)
        {
            reference_path = argv[++i];
        } else if (arg == "--out" && i + 1 < argc)
        {
            out_path = argv[++i];
        } else if (arg == "--env" && i + 1 < argc)
        {
            env_path = argv[++i];
        } else if (arg == "--canary-only")
        {
            canary_only = true;
        } else if (arg == "--smoke")
        {
            smoke = true;
        } else if (arg == "--canary-samples" && i + 1 < argc)
        {
            canary_samples = std::atoi(argv[++i]);
        } else if (arg == "--canary-iterations" && i + 1 < argc)
        {
            canary_iterations = std::strtoull(argv[++i], nullptr, 10);
        } else
        {
            std::fprintf(stderr,
                         "usage: measure-interaction --emit-arguments <csv> [--smoke]\n"
                         "       measure-interaction --arguments <csv> --reference <csv> "
                         "--out <csv> [--env <txt>] [--smoke]\n"
                         "       measure-interaction --canary-only [--canary-samples N] "
                         "[--canary-iterations N]\n"
                         "  --smoke  a few cells, no canary: a functional check of the harness, "
                         "never a measurement\n");
            return 2;
        }
    }

    if (smoke && canary_only)
    {
        std::fprintf(stderr, "measure-interaction: --smoke and --canary-only are opposite "
                             "requests; pick one\n");
        return 2;
    }

    if (emit_path != nullptr)
    {
        EmitArguments(emit_path, smoke);
        std::printf("arguments: written to %s%s\n", emit_path,
                    smoke ? " (smoke grid, not the probe)" : "");
        return 0;
    }

    if (canary_samples < 1 || canary_iterations < 1)
    {
        std::fprintf(stderr, "measure-interaction: canary samples and iterations must be "
                             "at least 1\n");
        return 2;
    }

    // Owner ruling 2026-10-20: the canary is a HINT.  With this variable set,
    // a wide reading is printed and recorded but does not stop the probe; the
    // rows keep the canary beside them.  Without it the strict default stands.
    const char* hint_env = std::getenv("BOYS_CANARY_HINT_ONLY");
    const bool canary_hint_only = (hint_env != nullptr && hint_env[0] == '1');

    if (canary_only)
    {
        const Canary c = RunCanary(canary_samples, canary_iterations);
        const bool wide = c.spread_pct > kCanarySpreadAlarmPct;
        PrintCanary("gate", c, wide);
        if (wide && canary_hint_only)
        {
            std::printf("canary gate: WIDE — recorded as a hint only (owner ruling 2026-10-20); "
                        "proceeding\n");
            return 0;
        }
        std::printf("canary gate: %s\n", wide ? "WIDE, do not measure" : "quiet, proceed");
        return wide ? 6 : 0;
    }

    if (arguments_path == nullptr || reference_path == nullptr || out_path == nullptr)
    {
        std::fprintf(stderr,
                     "usage: measure-interaction --arguments <csv> --reference <csv> "
                     "--out <csv> [--env <txt>]\n");
        return 2;
    }

    // Machine rule: the coordinator classifies the probe as timing-sensitive.
    // Read the canary first; a wide pre-canary means the machine is not quiet
    // and no cell is measured. The smoke path is not the probe: it reads no
    // canary, measures a fixed handful of cells and says so on every line that
    // names it.
    Canary pre{};
    bool pre_wide = false;

    if (!smoke)
    {
        pre = RunCanary(canary_samples, canary_iterations);
        pre_wide = pre.spread_pct > kCanarySpreadAlarmPct;
        PrintCanary("before measurement", pre, pre_wide);
        if (pre_wide && !canary_hint_only)
        {
            std::printf("canary gate: WIDE, no cell measured; rerun on a quiet machine\n");
            return 6;
        }
        if (pre_wide)
        {
            std::printf("canary gate: WIDE — recorded as a hint only (owner ruling 2026-10-20); "
                        "measuring\n");
        }
    } else
    {
        std::printf("SMOKE: functional harness check only; no canary; %zu fixed cells; "
                    "these rows are not a measurement\n",
                    std::size(kSmokeCells));
        std::fflush(stdout);
    }

    std::vector<LaneData> lanes = LoadArguments(arguments_path);
    AttachReference(reference_path, lanes);

    EnsureParentDirectory(out_path);
    std::FILE* out = std::fopen(out_path, "wb");
    if (out == nullptr)
    {
        std::fprintf(stderr, "measure-interaction: cannot write '%s'\n", out_path);
        return 2;
    }

    std::fputs(
        "precision,route,scheme,partition,pack,form,exp,n,x_nominal,x,value,ref_hi,ref_lo,"
        "abs_err,f0,fn,figure,half_digit,limit,accessor_ratio,figure_ratio,pack_live,"
        "partition_in_domain,argument_kind,status\n",
        out);

    Context ctx{out};

    constexpr Precision kPrecisions[] = {Precision::kFp64, Precision::kFp32, Precision::kFp16,
                                         Precision::kBf16};
    constexpr FitRoute kRoutes[] = {FitRoute::kChebyshev, FitRoute::kRationalMinimax};
    constexpr EvalScheme kSchemes[] = {EvalScheme::kSplitClenshaw, EvalScheme::kHorner};
    constexpr FitGranularity kGranularities[] = {FitGranularity::kCoarsest,
                                                 FitGranularity::kNarrow,
                                                 FitGranularity::kUniform};
    constexpr PackAxis kPacks[] = {PackAxis::kArguments, PackAxis::kOrders};
    constexpr DivisionForm kForms[] = {DivisionForm::kExactDivision,
                                       DivisionForm::kPlainReciprocal,
                                       DivisionForm::kRefinedReciprocal};
    constexpr RegionBExp kExps[] = {RegionBExp::kAccurate, RegionBExp::kFast};

    if (smoke)
    {
        for (const Cell& cell : kSmokeCells)
        {
            RunCell(ctx, cell, lanes);
        }
    } else
    {
        for (const Precision precision : kPrecisions)
        {
#if !BoysFp16
            if (precision == Precision::kFp16 || precision == Precision::kBf16)
            {
                continue;
            }
#endif
            for (const FitRoute route : kRoutes)
            {
                for (const EvalScheme scheme : kSchemes)
                {
                    for (const FitGranularity granularity : kGranularities)
                    {
                        for (const PackAxis pack : kPacks)
                        {
                            for (const DivisionForm form : kForms)
                            {
                                for (const RegionBExp exp : kExps)
                                {
                                    RunCell(ctx, Cell{precision, route, scheme, granularity, pack,
                                                      form, exp},
                                            lanes);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    std::fclose(out);

    Canary post{};
    bool post_wide = false;

    if (!smoke)
    {
        post = RunCanary(canary_samples, canary_iterations);
        post_wide = post.spread_pct > kCanarySpreadAlarmPct;
        PrintCanary("after measurement", post, post_wide);
    }

    if (env_path != nullptr)
    {
        WriteEnvironment(env_path, canary_samples, canary_iterations, pre, post, smoke);
    }

    std::printf(
        "measure-interaction: %zu rows, %zu refused cells-readings, %zu readings over "
        "the published figure%s\n",
        ctx.rows, ctx.refused, ctx.over_bound,
        smoke ? " (SMOKE: not a measurement)" : "");

    if (post_wide)
    {
        if (canary_hint_only)
        {
            std::printf("canary: WIDE after measurement — recorded as a hint only "
                        "(owner ruling 2026-10-20); the rows are timing-provisional\n");
        }
        else
        {
            std::printf("canary: WIDE after measurement; the rows are provisional and nothing "
                        "ships from them, rerun on a quiet machine\n");
            return 6;
        }
    }

    return 0;
}
