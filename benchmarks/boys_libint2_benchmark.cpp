// The comparison benchmark: this library's default Boys entry against libint2's
// Boys evaluator, on the same argument streams and at full accuracy on both
// sides.
//
// **The two sides.** This library's side is BoysAllOrders instantiated at
// boys::DefaultPolicyFp64 - the batch entry at the library's own default
// policy, which is the instantiation a call site that names no route, scheme,
// granularity or packing reaches, because that policy IS the entries' template
// default. The benchmark names the policy rather than restating its axes, so
// the comparison follows the default if the default moves; the banner prints
// what the default selects, under the names the library's own tables give it.
// The other side is libint2::FmEval_Chebyshev7<double>, libint2's
// Chebyshev-table Boys evaluator, constructed with its own default precision.
//
// libint2 is v2.13.1 (https://github.com/evaleev/libint). Its Boys surface is
// header-only - libint2/boys.h is compiled against here and no libint2 object
// is linked - and those headers are LGPL-3.0-or-later, which is why they are
// consumed at build time only and never vendored or redistributed with this
// library. This target exists only where BOYS_LIBINT2_INCLUDE_DIR names a
// libint2 include directory; libint2 is not a dependency of the library, and a
// tree without it builds everything else unchanged.
//
// **Both sides evaluate the same ladder.** Per argument this library fills
// F_0(x)..F_n(x) and libint2 fills F_0(x)..F_m_max(x), both at the plain
// definition F_m(x) = the integral of t^(2m) exp(-x t^2) over t in [0, 1], with
// no prefactor on either side. One pass over a stream therefore does the same
// work on both sides, and every value of both ladders is compared.
//
// **Accuracy.** Both sides run at their full documented accuracy: this library
// at the multiplier m = 1, whose batch bound is 5.5e-14 absolute per value (the
// accuracy contract in the boys/boys.hpp preamble), and libint2 at its
// Chebyshev table's documented relative precision, one unit in the last place
// of a double. The two errors are independent, so their difference is bounded
// by their sum, and the largest difference measured over each stream is printed
// against that sum. The check is worth running because the two libraries define
// F_m(x) with the same normalisation: a mismatch of convention would show there
// as an O(1) difference rather than as the small one this reports.
//
// **Streams.** The two streams and their generators are the ones the library's
// own benchmark uses (benchmarks/boys_benchmark.cpp): arguments uniform on
// [0, 40] with n uniform on [0, maxima], from mt19937_64(42); and the benzene
// 6-31G(d) nuclear-attraction stream from mt19937_64(43), whose n is drawn
// geometrically so that the low orders dominate as they do in real integral
// work. Reusing them keeps these rows comparable with the rows that benchmark
// records. Each stream's own extent - its largest n and its largest x - is
// printed with its numbers, because the extent is part of what the two sides
// had to agree on.
//
// **Protocol.** Per stream: one warm-up pass per side, then kPasses recorded
// passes with the two sides alternating inside each pass, so both sides see the
// same machine state. A recorded pass reports nanoseconds per argument - the
// pass's wall time divided by the stream's length - and the min, median and max
// over the recorded passes are printed per side, with the median as the figure
// to read. The ratio of the two medians is printed beside them: it is the
// comparison that survives a busy machine, since a load that inflates one pass
// inflates the other in the same round.
#include "boys/boys.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

// libint2's headers reach for the SIMD types through <intrin.h> on MSVC and key
// their vector sections on the compiler's own feature macros, which MSVC spells
// differently; the shim below supplies the ones the 4-wide Chebyshev path reads
// and is inert on every other compiler.
#if defined(_MSC_VER) && defined(__AVX__)
#include <immintrin.h>
#ifndef __SSE2__
#define __SSE2__ 1
#endif
#ifndef __SSE__
#define __SSE__ 1
#endif
#endif

#include <libint2/boys.h>

// libint2 initialises its static tables from a hook that its compiled library
// defines. A Boys-only consumer links no libint2 object and has no such state
// to set up, so it supplies the empty pair.
extern "C" void libint2_static_init() {}

extern "C" void libint2_static_cleanup() {}

namespace {

constexpr std::size_t kInputCount = 1u << 22; // 4,194,304 arguments per stream
constexpr int kPasses = 5;
constexpr std::size_t kLadderSize = static_cast<std::size_t>(boys::kMaxBoysOrder) + 1;

// The two sides' documented errors over one value: this library's batch bound,
// and libint2's table, whose error is relative to a value no larger than
// F_0(x) <= 1 and so no larger than one double epsilon.
constexpr double kDefaultEntryBound = 5.5e-14;
constexpr double kLibint2Bound = std::numeric_limits<double>::epsilon();
constexpr double kAgreementBudget = kDefaultEntryBound + kLibint2Bound;

struct Item {
    int n;
    double x;
};

struct Agreement {
    double maxDiff = 0.0;
    Item worst = Item{0, 0.0};
};

struct Spread {
    double min = 0.0;
    double median = 0.0;
    double max = 0.0;
};

// Arguments uniform on [0, 40] with n uniform on [0, kMaxBoysOrder], from one
// mt19937_64(42): the generator of the library's own uniform benchmark stream.
std::vector<Item> UniformInputs() {
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> xd(0.0, 40.0);
    std::uniform_int_distribution<int> nd(0, boys::kMaxBoysOrder);
    std::vector<Item> items(kInputCount);

    for (auto& item : items)
    {
        item.n = nd(rng);
        item.x = xd(rng);
    }

    return items;
}

// Benzene at 6-31G(d): primitive exponents and geometry (C-C 1.39 A, C-H 1.09
// A), sampled primitive pairs with x = p*|P - C|^2 against a third primitive's
// center C, exactly as the library's own molecular benchmark stream builds it.
std::vector<Item> MolecularInputs() {
    constexpr double kBohr = 1.8897261246257702;
    constexpr double kR = 1.39;
    constexpr double kCH = 1.09;
    constexpr double kCExp[] = {3047.52490,
                                457.369510,
                                103.948690,
                                29.2101550,
                                9.28666300,
                                3.16392700,
                                7.86827240,
                                1.88128850,
                                0.54424930,
                                0.16871440,
                                0.80000000};
    constexpr double kHExp[] = {18.7311370, 2.8253937, 0.6401217, 0.1612778};

    struct Primitive {
        double exponent;
        double x, y, z;
    };

    std::vector<Primitive> primitives;
    constexpr double kPi = 3.14159265358979323846;

    for (int k = 0; k < 6; ++k)
    {
        const double angle = k * kPi / 3.0;
        const double cx = kR * std::cos(angle);
        const double cy = kR * std::sin(angle);

        for (double exponent : kCExp)
        {
            primitives.push_back({exponent, cx * kBohr, cy * kBohr, 0.0});
        }
    }

    for (int k = 0; k < 6; ++k)
    {
        const double angle = k * kPi / 3.0;
        const double hx = (kR + kCH) * std::cos(angle);
        const double hy = (kR + kCH) * std::sin(angle);

        for (double exponent : kHExp)
        {
            primitives.push_back({exponent, hx * kBohr, hy * kBohr, 0.0});
        }
    }

    std::mt19937_64 rng(43);
    std::vector<Item> items;
    items.reserve(kInputCount);

    while (items.size() < kInputCount)
    {
        const Primitive& a = primitives[rng() % primitives.size()];
        const Primitive& b = primitives[rng() % primitives.size()];
        const double p = a.exponent + b.exponent;
        const double px = (a.exponent * a.x + b.exponent * b.x) / p;
        const double py = (a.exponent * a.y + b.exponent * b.y) / p;
        const double pz = (a.exponent * a.z + b.exponent * b.z) / p;
        const Primitive& c = primitives[rng() % primitives.size()];
        const double d2 =
            (px - c.x) * (px - c.x) + (py - c.y) * (py - c.y) + (pz - c.z) * (pz - c.z);
        // Geometric n: the low orders dominate real integral workloads.
        int n = 0;

        while (n < boys::kMaxBoysOrder && (rng() & 1u) == 0)
        {
            ++n;
        }

        items.push_back({n, p * d2});
    }

    return items;
}

// One pass of this library over a stream. The policy named here is the
// library's default for the double lane, which is also the entries' template
// default, so this is the instantiation a caller who names nothing reaches; the
// argument's own row feeds the sink so the call cannot be eliminated.
void DefaultEntryPass(const std::vector<Item>& items, volatile double& sink) {
    std::array<double, kLadderSize> ladder{};

    for (const Item& item : items)
    {
        boys::BoysAllOrders<boys::kBoysFullAccuracyMultiplier, boys::DefaultPolicyFp64>(
            item.n, item.x, ladder.data());
        sink += ladder[static_cast<std::size_t>(item.n)];
    }
}

// One pass of libint2 over the same stream, at the same ladder depth.
void Libint2Pass(const libint2::FmEval_Chebyshev7<double>& fm,
                 const std::vector<Item>& items,
                 volatile double& sink) {
    std::array<double, kLadderSize> ladder{};

    for (const Item& item : items)
    {
        fm.eval(ladder.data(), item.x, item.n);
        sink += ladder[static_cast<std::size_t>(item.n)];
    }
}

// The largest difference between the two ladders over one stream, and the
// argument that produced it.
Agreement MeasureAgreement(const libint2::FmEval_Chebyshev7<double>& fm,
                           const std::vector<Item>& items) {
    std::array<double, kLadderSize> mine{};
    std::array<double, kLadderSize> theirs{};
    Agreement result;

    for (const Item& item : items)
    {
        boys::BoysAllOrders<boys::kBoysFullAccuracyMultiplier, boys::DefaultPolicyFp64>(
            item.n, item.x, mine.data());
        fm.eval(theirs.data(), item.x, item.n);

        for (int k = 0; k <= item.n; ++k)
        {
            const double diff =
                std::abs(mine[static_cast<std::size_t>(k)] - theirs[static_cast<std::size_t>(k)]);

            if (diff > result.maxDiff)
            {
                result.maxDiff = diff;
                result.worst = item;
            }
        }
    }

    return result;
}

Spread Summarise(const std::vector<double>& samples) {
    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    return Spread{sorted.front(), sorted[sorted.size() / 2], sorted.back()};
}

// The name the library itself prints for the route its default policy selects,
// read from the route table rather than restated here.
const char* DefaultRouteName() {
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        if (row.route == boys::DefaultPolicyFp64::kRoute)
        {
            return row.name;
        }
    }

    return "unnamed";
}

int MaxOrderOf(const std::vector<Item>& items) {
    int nmax = 0;

    for (const Item& item : items)
    {
        nmax = std::max(nmax, item.n);
    }

    return nmax;
}

double MaxArgumentOf(const std::vector<Item>& items) {
    double xmax = 0.0;

    for (const Item& item : items)
    {
        xmax = std::max(xmax, item.x);
    }

    return xmax;
}

void PrintBanner() {
    std::printf("comparison: this library's default Boys entry vs libint2 Boys\n");
    std::printf("this library entry: BoysAllOrders (batch) at boys::DefaultPolicyFp64"
                " - the library's default, and the entries' template default\n");
    std::printf("default selectors: route %s | scheme %s | granularity %s | packing %s\n",
                DefaultRouteName(),
                boys::EvalSchemeName(boys::DefaultPolicyFp64::kScheme),
                boys::GranularityName(boys::DefaultPolicyFp64::kGranularity),
                boys::PackAxisName(boys::DefaultPolicyFp64::kPack));
    std::printf("default accuracy: multiplier m = 1 (full), bound %.3e absolute per value\n",
                kDefaultEntryBound);
    std::printf("other side: libint2 v2.13.1 FmEval_Chebyshev7<double>, header-only, "
                "at its table's own precision (bound %.3e relative)\n",
                kLibint2Bound);
    std::printf("agreement budget: %.4e (the two sides' documented errors, summed)\n",
                kAgreementBudget);
}

void PrintStreamRow(const char* name, const std::vector<Item>& items) {
    std::printf("stream: %s | count: %zu | nmax: %d | xmax: %.6e\n",
                name,
                items.size(),
                MaxOrderOf(items),
                MaxArgumentOf(items));
}

void PrintTimingRow(const char* side, const Spread& spread) {
    std::printf("  %-22s ns_per_arg min: %9.3f | median: %9.3f | max: %9.3f\n",
                side,
                spread.min,
                spread.median,
                spread.max);
}

// One stream end to end: the agreement measurement, then the warm-up pass and
// the recorded passes, then the two sides' rows.
void CompareStream(const char* name,
                   const std::vector<Item>& items,
                   const libint2::FmEval_Chebyshev7<double>& fm) {
    PrintStreamRow(name, items);

    const Agreement agreement = MeasureAgreement(fm, items);
    const bool within = agreement.maxDiff <= kAgreementBudget;
    std::printf("agreement: %s | max_abs_diff: %.3e | budget: %.3e | within_budget: %s | "
                "worst at n: %d | worst at x: %.9f\n",
                name,
                agreement.maxDiff,
                kAgreementBudget,
                within ? "yes" : "no",
                agreement.worst.n,
                agreement.worst.x);

    const auto elapsedNs = [&items](const auto& pass) {
        const auto t0 = std::chrono::steady_clock::now();
        pass();
        const auto t1 = std::chrono::steady_clock::now();

        return std::chrono::duration<double, std::nano>(t1 - t0).count() /
               static_cast<double>(items.size());
    };

    // One sink per side: each carries that side's own sum, so the two numbers
    // printed at the end are each side's and not a shared running total.
    volatile double librarySink = 0.0;
    volatile double libint2Sink = 0.0;
    const auto libraryPass = [&items, &librarySink]() { DefaultEntryPass(items, librarySink); };
    const auto libint2Pass = [&items, &fm, &libint2Sink]() { Libint2Pass(fm, items, libint2Sink); };

    // Warm-up: one pass per side, so both sides' tables and pages are resident
    // before the first timed pass.
    libraryPass();
    libint2Pass();

    std::vector<double> libraryPasses;
    std::vector<double> libint2Passes;
    libraryPasses.reserve(kPasses);
    libint2Passes.reserve(kPasses);

    for (int pass = 0; pass < kPasses; ++pass)
    {
        libraryPasses.push_back(elapsedNs(libraryPass));
        libint2Passes.push_back(elapsedNs(libint2Pass));
    }

    const Spread library = Summarise(libraryPasses);
    const Spread libint2 = Summarise(libint2Passes);
    std::printf("timing: %s | warmup_passes: 1 | recorded_passes: %d\n", name, kPasses);
    PrintTimingRow("default-entry", library);
    PrintTimingRow("libint2-cheb7", libint2);
    std::printf("  median ns_per_arg ratio (libint2 / default-entry): %.3fx\n",
                libint2.median / library.median);
    std::printf("  sink: default-entry %.17g | libint2 %.17g\n",
                static_cast<double>(librarySink),
                static_cast<double>(libint2Sink));
}

} // namespace

int main() {
    // The stream arguments are identical on both sides: one input list is built
    // and handed to each.
    const std::vector<Item> uniform = UniformInputs();
    const std::vector<Item> molecular = MolecularInputs();
    const libint2::FmEval_Chebyshev7<double> fm(boys::kMaxBoysOrder);

    PrintBanner();
    CompareStream("uniform-n32-x40", uniform, fm);
    CompareStream("molecular-benzene-631gd", molecular, fm);
    return 0;
}
