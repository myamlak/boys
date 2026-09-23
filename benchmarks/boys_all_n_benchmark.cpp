// The all-orders batch entry (BoysAllN) against the per-argument loop it
// replaces: same arguments, same order-major planes, and the sort's own cost
// separated from the grouped-path win.
//
// Five variants over ONE argument set:
//   perarg          BoysAllOrders per argument, scattered into the planes
//                   (the baseline a caller writes today; the C API's
//                   BoysDoubleBatch is this loop)
//   shuffled        BoysAllN on the arguments in generation order - the sort is
//                   paid
//   ascending       BoysAllN on the same arguments sorted - the sort is still
//                   walked, but the run memory is sequential
//   tag             BoysAllN with BoysSortedArgs on the sorted arguments - the
//                   sort is skipped
//   tag_workspace   the same call with a caller-supplied workspace
//
// The honest separations: tag vs perarg is the win the entry exists for; a
// measurement of the sort by itself is ascending minus tag (same arguments,
// same runs, and the only difference is the classification pass, the index
// scatter and the indexed plane writes); shuffled vs ascending isolates the
// input-order effect at a fixed sort cost.
//
// Protocol: rounds of one call per variant, interleaved, so a busy machine
// inflates every variant together and the reported ratios survive it; per
// variant the minimum over the rounds is what is reported, with the round
// count and the max/min spread printed so a reader can judge the noise. The
// ratios, not the milliseconds, are the result.
//
// Workload: x log-uniform on [1e-3, 40] by default (a synthetic stand-in for a
// molecular argument set: it populates region A, the extended band, region B and
// region C in one array); --xrange=lo,hi, --nmax=N, --count=N and --rounds=N
// move it to the shapes a consumer actually asks for (a low angular momentum
// shell quartet wants a small nmax over region-A arguments).
#include "boys/boys.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <ctime>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kCount = 1u << 14; // 16384 arguments
constexpr int kOrder = boys::kMaxBoysOrder;
constexpr int kRounds = 41;

using Clock = std::chrono::steady_clock;

struct Variant {
    const char* name;
    double minMs = 0.0;
    double maxMs = 0.0;
};

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

// The baseline: one per-argument call per element, scattered into the same
// order-major planes the entry writes.
void RunPerArgument(const double* x, double* out, std::size_t count, int order) {
    double batch[boys::kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        boys::BoysAllOrders(order, x[i], batch);

        for (int k = 0; k <= order; ++k)
        {
            out[static_cast<std::size_t>(k) * count + i] = batch[k];
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    int order = kOrder;
    double rangeLo = 1e-3;
    double rangeHi = 40.0;
    std::size_t count = kCount;
    int rounds = kRounds;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg.rfind("--nmax=", 0) == 0)
        {
            order = std::atoi(arg.c_str() + 7);
        } else if (arg.rfind("--count=", 0) == 0)
        {
            count = static_cast<std::size_t>(std::strtoull(arg.c_str() + 8, nullptr, 10));
        } else if (arg.rfind("--rounds=", 0) == 0)
        {
            rounds = std::atoi(arg.c_str() + 9);
        } else if (arg.rfind("--xrange=", 0) == 0)
        {
            rangeLo = std::strtod(arg.c_str() + 9, nullptr);
            const char* comma = std::strchr(arg.c_str() + 9, ',');
            rangeHi = (comma != nullptr) ? std::strtod(comma + 1, nullptr) : rangeLo;
        }
    }

    std::mt19937_64 rng(47);
    std::uniform_real_distribution<double> logX(std::log10(rangeLo), std::log10(rangeHi));
    std::vector<double> x(count);

    for (double& value : x)
    {
        value = std::pow(10.0, logX(rng));
    }

    std::vector<double> ascending = x;
    std::sort(ascending.begin(), ascending.end());
    std::vector<double> out(count * (static_cast<std::size_t>(order) + 1));
    std::vector<std::size_t> workspace(boys::BoysAllNWorkspaceSize(count));

    Variant variants[] = {
        {"perarg", 0.0, 0.0},
        {"shuffled", 0.0, 0.0},
        {"ascending", 0.0, 0.0},
        {"tag", 0.0, 0.0},
        {"tag_workspace", 0.0, 0.0},
    };

    const auto call = [&](std::size_t which) {
        switch (which)
        {
        case 0:
            RunPerArgument(x.data(), out.data(), count, order);
            break;

        case 1:
            boys::BoysAllN(order, x.data(), out.data(), count);
            break;

        case 2:
            boys::BoysAllN(order, ascending.data(), out.data(), count);
            break;

        case 3:
            boys::BoysAllN(order, ascending.data(), out.data(), count, boys::BoysSortedArgs{});
            break;

        default:
            boys::BoysAllN(order, ascending.data(), out.data(), count, workspace.data());
            break;
        }
    };

    // Warm-up: every variant once, so the tables and the workspace pages are
    // resident before the first timed round.
    const std::string started = Timestamp();

    for (std::size_t v = 0; v < std::size(variants); ++v)
    {
        call(v);
    }

    double checksum = 0.0;

    for (int round = 0; round < rounds; ++round)
    {
        for (std::size_t v = 0; v < std::size(variants); ++v)
        {
            const auto t0 = Clock::now();
            call(v);
            const auto t1 = Clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

            if (round == 0)
            {
                variants[v].minMs = ms;
                variants[v].maxMs = ms;
            } else
            {
                variants[v].minMs = std::min(variants[v].minMs, ms);
                variants[v].maxMs = std::max(variants[v].maxMs, ms);
            }
        }
    }

    for (const double value : out)
    {
        checksum += value;
    }

    std::printf("host: avx2=%d | workload: log-uniform-x-%.0e-%.0e | nmax: %d | count: %zu | "
                "rounds: %d | min_of_rounds | checked_sum: %.6e\n",
                boys::BoysAvx2Available() ? 1 : 0,
                rangeLo,
                rangeHi,
                order,
                count,
                rounds,
                checksum);
    std::printf("started: %s | finished: %s\n", started.c_str(), Timestamp().c_str());

    for (const Variant& variant : variants)
    {
        std::printf("  %-14s min_ms %.3f | max_ms %.3f | spread %.2fx\n",
                    variant.name,
                    variant.minMs,
                    variant.maxMs,
                    variant.maxMs / variant.minMs);
    }

    const double perarg = variants[0].minMs;
    std::printf("ratios (perarg / variant, >1 means the entry is faster):\n");
    std::printf("  entry_shuffled   %.2fx\n", perarg / variants[1].minMs);
    std::printf("  entry_ascending  %.2fx\n", perarg / variants[2].minMs);
    std::printf("  entry_tag        %.2fx\n", perarg / variants[3].minMs);
    std::printf("  entry_tag_ws     %.2fx\n", perarg / variants[4].minMs);
    std::printf("  sort_cost        %.3f ms (ascending - tag) | shuffled/ascending %.2fx\n",
                variants[2].minMs - variants[3].minMs,
                variants[1].minMs / variants[2].minMs);
    return 0;
}
