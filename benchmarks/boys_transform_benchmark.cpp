// The region-A transform lane's cost: one row per arithmetic mode and band, on
// one argument set.
//
// WHAT A ROW IS. BoysRegionAProduct evaluates one band's per-order fits as a
// single matrix product per band, in the mode named at compile time. Each row
// below is one call of the shipped entry at one (mode, band) pair, timed; the
// two bands are the lane's whole domain (the entry takes region-A arguments and
// does not sort, group or fall back), so the rows are its entire option space.
//
// WHY THE MODES SEPARATE. The modes differ in exactly two things: how many
// products per operand they take (ProductModeInfo::parts: 1, 2 or 3, which is
// parts * (parts + 1) / 2 products per degree pair) and how many operand
// roundings they pay, since every mode but fp64 rounds each operand to its
// format through detail::RoundSignificand. fp64 against tf32 isolates the
// second one: one product per degree in both, so the difference between those
// two rows is the rounding and nothing else. The split modes then pay it twice
// or three times per operand on top of their extra products.
//
// PROTOCOL. Every row is timed once per round, the rounds interleaved, so a
// machine doing something else inflates every row together and the ratios
// survive it. Each row reports its minimum over the rounds with the max/min
// spread beside it; the ratio to the same band's fp64 row is the result.
// ns_per_value is that minimum in nanoseconds per output value, the call
// writing count * (nmax + 1) values as out[k * count + i] = F_k(x[i]).
//
// WORKLOAD. Arguments log-uniform across the band, from a fixed formula so two
// runs of one build hand the entry the same arguments. The lane's work does not
// depend on where inside a band an argument sits - the whole band is one
// product, taken by every argument of the batch - so what the spread buys is
// that no single point of the interval is what a figure describes. Every row
// writes into one output buffer: the entry is pure and touches only its own
// out, so reusing it leaves the rows differing in arithmetic and nothing else.
//
// Usage:
//   boys-transform-benchmark [--band=kA1|kA2] [--mode=NAME] [--nmax=N]
//                            [--count=N] [--rounds=N]
//   boys-transform-benchmark --list
#include "boys/boys_transform.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kCount = 4096;
constexpr int kOrder = boys::kMaxBoysOrder;
constexpr int kRounds = 11;

static_assert(kCount % boys::detail::kProductTile == 0,
              "a count off the entry's tile multiple would make every row measure its ragged "
              "last tile as well as its steady state");

using Clock = std::chrono::steady_clock;

struct Config {
    std::size_t count = kCount;
    int nmax = kOrder;
    int rounds = kRounds;
    std::string band; // empty: both bands
    std::string mode; // empty: every mode the build reports
    bool list = false;
};

struct Row {
    const boys::ProductModeInfo* info;
    boys::RegionABand band;
    std::size_t slot; // this band's arguments, in the order the bands were built
    double minMs = 0.0;
    double maxMs = 0.0;
};

const char* BandName(boys::RegionABand band) {
    return band == boys::RegionABand::kA1 ? "kA1" : "kA2";
}

std::string Timestamp() {
    const std::time_t now = std::time(nullptr);
    std::array<char, 32> buffer{};
    std::tm utc{};

#ifdef _MSC_VER
    gmtime_s(&utc, &now);
#else
    gmtime_r(&now, &utc);
#endif

    std::strftime(buffer.data(), buffer.size(), "%Y-%m-%dT%H:%M:%SZ", &utc);
    return std::string(buffer.data());
}

// One band's arguments, log-uniform across it. Both ends are held strictly
// inside the half-open band: the lower nudge keeps exp(log(a)) from landing a
// rounding below a itself, which the entry asserts against.
std::vector<double> MakeArguments(boys::RegionABand band, std::size_t count) {
    const double lower = (band == boys::RegionABand::kA1) ? 1e-3 : boys::kRegionA1Edge;
    const double upper = (band == boys::RegionABand::kA1) ? boys::kRegionA1Edge : boys::kRegionAEnd;
    const double lo = std::log(lower * (1.0 + 1e-12));
    const double hi = std::log(upper * (1.0 - 1e-9));
    std::vector<double> x(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        const double u =
            (count > 1) ? static_cast<double>(i) / static_cast<double>(count - 1) : 0.0;
        x[i] = std::exp(lo + u * (hi - lo));
    }

    return x;
}

// One row's call: the shipped entry at the row's mode and band. The switch names
// every mode the enumeration carries, so a build that adds one is a warning
// here and a refusal below, never a row timed as some other mode.
bool Call(const Row& row, int nmax, const double* x, double* out, std::size_t count) {
    switch (row.info->mode)
    {
    case boys::ProductMode::kFp64:
        boys::BoysRegionAProduct<boys::ProductMode::kFp64>(row.band, nmax, x, out, count);
        return true;

    case boys::ProductMode::kTf32x3:
        boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(row.band, nmax, x, out, count);
        return true;

    case boys::ProductMode::kBf16x6:
        boys::BoysRegionAProduct<boys::ProductMode::kBf16x6>(row.band, nmax, x, out, count);
        return true;

    case boys::ProductMode::kTf32:
        boys::BoysRegionAProduct<boys::ProductMode::kTf32>(row.band, nmax, x, out, count);
        return true;

    case boys::ProductMode::kBf16:
        boys::BoysRegionAProduct<boys::ProductMode::kBf16>(row.band, nmax, x, out, count);
        return true;

    case boys::ProductMode::kFp16:
        boys::BoysRegionAProduct<boys::ProductMode::kFp16>(row.band, nmax, x, out, count);
        return true;
    }

    return false;
}

// The row a ratio divides by: the same band's fp64 row, which pays no operand
// rounding at all. Zero when this run times none.
double Baseline(const std::vector<Row>& rows, boys::RegionABand band) {
    for (const Row& row : rows)
    {
        if (row.band == band && row.info->mode == boys::ProductMode::kFp64)
        {
            return row.minMs;
        }
    }

    return 0.0;
}

} // namespace

int main(int argc, char** argv) {
    Config config;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--list")
        {
            config.list = true;
        } else if (arg.rfind("--band=", 0) == 0)
        {
            config.band = arg.substr(7);
        } else if (arg.rfind("--mode=", 0) == 0)
        {
            config.mode = arg.substr(7);
        } else if (arg.rfind("--nmax=", 0) == 0)
        {
            config.nmax = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--count=", 0) == 0)
        {
            config.count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 8, nullptr, 10));
        } else if (arg.rfind("--rounds=", 0) == 0)
        {
            config.rounds = std::atoi(arg.c_str() + 9);
        } else
        {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    // The mode list is the library's own, not one written out here.
    const std::span<const boys::ProductModeInfo> modes = boys::BoysProductModes();

    if (config.list)
    {
        for (const boys::ProductModeInfo& info : modes)
        {
            std::printf("%-9s parts %d | %s\n", info.name, info.parts, info.model);
        }

        return 0;
    }

    if (config.nmax < 0 || config.nmax > boys::kMaxBoysOrder || config.count == 0 ||
        config.rounds <= 0)
    {
        std::fprintf(stderr,
                     "nmax must be 0..%d, count at least 1, rounds at least 1\n",
                     boys::kMaxBoysOrder);
        return 2;
    }

    std::vector<boys::RegionABand> bands = {boys::RegionABand::kA1, boys::RegionABand::kA2};

    if (!config.band.empty())
    {
        if (config.band == "kA1")
        {
            bands = {boys::RegionABand::kA1};
        } else if (config.band == "kA2")
        {
            bands = {boys::RegionABand::kA2};
        } else
        {
            std::fprintf(
                stderr, "unknown band: %s (--band=kA1 or --band=kA2)\n", config.band.c_str());
            return 2;
        }
    }

    std::vector<Row> rows;
    bool named = config.mode.empty();

    for (const boys::ProductModeInfo& info : modes)
    {
        if (!config.mode.empty() && config.mode != info.name)
        {
            continue;
        }

        named = true;

        for (std::size_t slot = 0; slot < bands.size(); ++slot)
        {
            rows.push_back(Row{&info, bands[slot], slot, 0.0, 0.0});
        }
    }

    if (!named || rows.empty())
    {
        std::fprintf(stderr, "no mode named %s in this build; --list shows the ones it reports\n",
                     config.mode.c_str());
        return 2;
    }

    // One argument set per band, in the order the rows' slots name.
    std::vector<std::vector<double>> arguments;

    for (const boys::RegionABand band : bands)
    {
        arguments.push_back(MakeArguments(band, config.count));
    }

    const std::size_t values = config.count * (static_cast<std::size_t>(config.nmax) + 1);
    std::vector<double> out(values);
    const std::string started = Timestamp();

    // Warm-up: every row once, so the tables and the pages are resident before
    // the first timed round - and so a mode this harness cannot time is refused
    // before any clock is read, rather than printed as a row it never called.
    for (const Row& row : rows)
    {
        if (!Call(row, config.nmax, arguments[row.slot].data(), out.data(), config.count))
        {
            std::fprintf(stderr, "this harness has no call for mode %s: add its line to Call()\n",
                         row.info->name);
            return 2;
        }
    }

    // The interleaved rounds, and the sum of every row's last call with them: a
    // call that wrote nothing, or wrote another plane's values, moves it.
    double checksum = 0.0;

    for (int round = 0; round < config.rounds; ++round)
    {
        for (Row& row : rows)
        {
            const double* x = arguments[row.slot].data();
            const auto t0 = Clock::now();
            Call(row, config.nmax, x, out.data(), config.count);
            const auto t1 = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (round == 0)
            {
                row.minMs = ms;
                row.maxMs = ms;
            } else
            {
                row.minMs = std::min(row.minMs, ms);
                row.maxMs = std::max(row.maxMs, ms);
            }

            if (round == config.rounds - 1)
            {
                for (const double value : out)
                {
                    checksum += value;
                }
            }
        }
    }

    std::printf("workload: log-uniform-bands | rows: %d | nmax: %d | count: %zu | rounds: %d | "
                "min_of_rounds | checked_sum: %.6e\n",
                static_cast<int>(rows.size()),
                config.nmax,
                config.count,
                config.rounds,
                checksum);
    std::printf("started: %s | finished: %s\n", started.c_str(), Timestamp().c_str());

    for (const Row& row : rows)
    {
        std::printf("  %-9s %s | parts %d | min_ms %.3f | max_ms %.3f | spread %.2fx | "
                    "ns_per_value %.1f\n",
                    row.info->name,
                    BandName(row.band),
                    row.info->parts,
                    row.minMs,
                    row.maxMs,
                    row.maxMs / row.minMs,
                    row.minMs * 1e6 / static_cast<double>(values));
    }

    std::printf("ratios (mode / the same band's fp64 row, >1 means the mode costs more):\n");

    for (const Row& row : rows)
    {
        const double base = Baseline(rows, row.band);
        std::printf("  %-9s %s  ", row.info->name, BandName(row.band));

        if (base > 0.0)
        {
            std::printf("%.2fx\n", row.minMs / base);
        } else
        {
            std::printf("n/a (this run times no fp64 row for %s)\n", BandName(row.band));
        }
    }

    return 0;
}
