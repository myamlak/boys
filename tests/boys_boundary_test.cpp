// Recursion-boundaries harness: where the erf-seeded upward recursion leaves the 5e-14 window of
// the [VikhamarSandberg2026] eq. 26 series reference, per kmax in {4, 8, 16, 32}, pinning the four
// cells (first failing samples of the 1e-4 descending sweep; kThresholdRows). The library has no
// std::erf (region B seeds from the Chebyshev fit), so the harness carries the seed itself.

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace {

// The boundary threshold: the figure the accuracy contract is stated at, the
// absolute window the recursion must stay within for all n <= kmax.
constexpr double kBoundaryThreshold = 5e-14;

// The measurement resolution. The pin is the largest failing sample of a descending
// sweep at this step, so a different step is a different measurement, and 1e-4 is
// the resolution the cells below were taken at. The step is absolute, not scaled by
// the cell. Cost of the sweep: ~1.3 s at kmax = 32, ~0.8 s at kmax = 4.
constexpr double kMeasurementResolution = 1e-4;

// The band stays 20% and is not widened to cover scatter: it is set from this measurement's own
// scatter, measured below. INJECTED-REGRESSION EVIDENCE (glibc x86_64): a seed off by 1e-15
// relative (~10 ulp) moves the kmax = 4 cell to 0.6151 = +33.0% -> RED, as do a single-precision
// seed, a dropped +0.5 in the upward step and a typo'd reference denominator -> sweep-start guard RED.
constexpr double kMeasuredTolerance = 0.2;

// Series term cap and truncation floor, mirroring the committed generator
// (tools/gen_boys_coefficients.py, boys_ref: range(300), break at term < 1e-26 after l > 20). A
// floor of 1e-12 in its place is not a regression at this gate: it moves the reference in long
// double but not in the double the predicate compares.
constexpr int kMaxSeriesTerms = 300;
constexpr long double kSeriesTailFloor = 1e-26L;

// Inline series vs committed grid (relative, cross-check test below): ~3.7x the measured worst on
// MSVC's 64-bit long double (1.340e-15 at (n=6, x=40), the grid regenerated at 45 digits). The grid
// it replaced measured 3.200e-15 at (n=31, x=38.3119) - that grid's own argument rounding, F_n at
// the full-precision argument vs the x column read as a double (~2.5e-15 at d ln F / d ln x ~ 31).
constexpr double kCsvAgreementTolerance = 5e-15;

// Toolchain premise pin: MSVC's long double is 64-bit (a synonym for double), GCC/Clang's the x87
// 80-bit type. The reference series degrades to ~1e-16 relative on MSVC instead of ~1e-19, but F_n
// is O(1) or smaller in the sweep, so the reference error stays 2-3 orders below kBoundaryThreshold
// and the measurement is unaffected (the CSV cross-check's tolerance reflects this).
static_assert(sizeof(long double) == 8 || sizeof(long double) == 16);

// F0(x) = 0.5 sqrt(pi/x) erf(sqrt(x)) - the erf seed, implemented in-test
// because std::erf is deliberately not added to the production library.
double ErfSeedF0(double x) {
    return 0.5 * std::sqrt(std::numbers::pi / x) * std::erf(std::sqrt(x));
}

// F_n(x) via the provably stable series ([VikhamarSandberg2026] eq. 26), all terms positive:
// F_n(x) = 0.5 e^-x sum_l x^l / prod_{j=0..l} (n + j + 0.5), the series the committed grid is
// generated from (gen_boys_coefficients.py boys_ref) in long double: (n, x) are order, argument.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
long double SeriesReference(int n, double x) {
    if (x == 0.0)
    {
        return 1.0L / (2.0L * n + 1.0L);
    }

    const long double xL = static_cast<long double>(x);
    long double sum = 0.0L;
    long double term = 1.0L / (static_cast<long double>(n) + 0.5L);

    for (int l = 0; l < kMaxSeriesTerms; ++l)
    {
        sum += term;

        if (l > 20 && term < kSeriesTailFloor)
        {
            break;
        }

        term *= xL / (static_cast<long double>(n + l) + 1.5L);
    }

    // std::exp, not std::expl: C99 math names are not required in namespace std
    // (libstdc++ has no std::expl); the argument selects the long double overload.
    return std::exp(-xL) * 0.5L * sum;
}

// The shipped region-B upward step (expx = 0.5 e^-x hoisted once, then f = ((l + 0.5) f - expx) / x),
// seeded from the erf definition rather than the Chebyshev fit. Runs in double: the boundary is a
// property of fp64 arithmetic. (n, x) are the order and the argument - convertible, never swapped.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
double ErfSeededUpward(int n, double x) {
    double f = ErfSeedF0(x);
    const double expx = 0.5 * std::exp(-x);

    for (int l = 0; l < n; ++l)
    {
        f = ((l + 0.5) * f - expx) / x;
    }

    return f;
}

// Pass criterion: |recursion(n, x) - reference(n, x)| <= 5e-14 for all n in
// 0..kmax. (kmax, x) are the order cap and the argument - convertible, never swapped.
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
bool PassesThreshold(int kmax, double x) {
    for (int n = 0; n <= kmax; ++n)
    {
        const double recursion = ErfSeededUpward(n, x);
        const double reference = static_cast<double>(SeriesReference(n, x));

        if (std::abs(recursion - reference) > kBoundaryThreshold)
        {
            return false;
        }
    }

    return true;
}

// The failure top: the largest failing sample of a descending sweep from the sweep start at
// kMeasurementResolution - an existence witness and lower bound on the failure top, resolution-limited
// by construction. Stable over 12 lattice variants (glibc x86_64; steps 1e-4/1.3e-4/7e-5/5e-5): 6.1% /
// 3.8% / 2.3% / 0.5% of the cell for kmax = 4/8/16/32, worst -12.3% over 48 draws, inside the 20% band.
struct FailureTop {
    bool startPasses = false; // the sweep start is inside the 5e-14 window
    bool found = false; // a failing sample exists above the sweep floor
    double x = 0.0; // the largest failing sample found (the cell)
};

FailureTop MeasureFailureTop(int kmax) {
    constexpr double kSweepStart = 12.0;
    constexpr double kSweepFloor = 0.001;

    FailureTop result;

    // The pass region must contain the sweep start: at x = 12.0 the upward
    // amplification is O(1) for all n <= 32 (seed error ~1e-16 -> ~2e-16), far
    // below the threshold, and without a passing start there is no failure to find.
    if (!PassesThreshold(kmax, kSweepStart))
    {
        return result;
    }

    result.startPasses = true;

    // Descend at the measured resolution and stop at the FIRST failing sample.
    // Starting from the sweep start is what keeps the result independent of the
    // pinned cell: the search window is never chosen from the value it checks.
    // Bisection was rejected: the predicate is not monotone over the transition.
    for (int i = 0;; ++i)
    {
        const double x = kSweepStart - static_cast<double>(i) * kMeasurementResolution;

        if (x < kSweepFloor)
        {
            break;
        }

        if (!PassesThreshold(kmax, x))
        {
            result.found = true;
            result.x = x;
            break;
        }
    }

    return result;
}

// Per kmax: x0Measured (the asserted pin, the failure top of the 1e-4 descending sweep, MSVC) and
// formula ([VikhamarSandberg2026] eq. 25/13, literature, not measured); margins = formula / x0 are
// computed ratios, not asserted. Not a pass-run draw: the lattice alone moves that draw over
// 0.3157..0.4010 at kmax = 4 (27.0%) / 13.6% at kmax = 8, against 4.4% / 1.2% for the failure top (0.361 was no aarch64 evidence).
struct ThresholdRow {
    int kmax;
    double x0Measured;
    double formula;
    double resolution; // the step of the sweep that produced x0Measured
};

const std::array<ThresholdRow, 4> kThresholdRows = {
    ThresholdRow{4, 0.4625, 1.60, kMeasurementResolution},
    ThresholdRow{8, 1.6373, 3.07, kMeasurementResolution},
    ThresholdRow{16, 4.2367, 6.01, kMeasurementResolution},
    ThresholdRow{32, 10.0492, 11.9, kMeasurementResolution},
};

// The committed reference grid (tools/gen_boys_coefficients.py, 45-digit
// mpmath values of F_n at the double in each row's x column) - cross-checked
// against the inline series as a self-test.
struct CsvRow {
    int n;
    double x;
    double value;
};

std::vector<CsvRow> LoadCsvRows() {
    const std::string path = std::string(BoysDataDir) + "/boys_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<CsvRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream stream(line);
        std::string cell;
        CsvRow row{};
        std::getline(stream, cell, ',');
        row.n = std::stoi(cell);
        std::getline(stream, cell, ',');
        row.x = std::strtod(cell.c_str(), nullptr);
        std::getline(stream, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        // The committed grid ends at x = 100: beyond it every F_n is covered by
        // region C's asymptotic form (see boys_test.cpp). The filter guards against
        // a future grid extension past the series' limit, ~x = 250 in the generator.
        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

} // namespace

TEST(BoysBoundaryTest, ErfSeededUpwardRecursionThresholds) {
    for (const ThresholdRow& row : kThresholdRows)
    {
        const FailureTop top = MeasureFailureTop(row.kmax);

        // The sweep start must lie inside the window and the sweep must find a
        // failing sample above the floor: either failure means the measurement
        // found no boundary where the pin has one.
        ASSERT_TRUE(top.startPasses)
            << "kmax=" << row.kmax << ": no pass at the sweep start x = 12.0";
        ASSERT_TRUE(top.found) << "kmax=" << row.kmax
                               << ": no failing sample above the sweep floor (x = 0.001)";

        // The measured cell is the failure top at row.resolution - a first
        // failing sample, the same kind of measurement as the pin - so this is a
        // like-for-like comparison and the band covers the seed/libm spread.
        const double relativeDeviation = std::abs(top.x - row.x0Measured) / row.x0Measured;
        EXPECT_LE(relativeDeviation, kMeasuredTolerance)
            << "kmax=" << row.kmax << ": measured failure top x0 = " << top.x << " at resolution "
            << row.resolution << " vs pinned " << row.x0Measured;
        std::printf("BoysBoundary kmax=%2d: failure top x0 = %.6f at resolution %.0e, pinned %.4f, "
                    "formula %.2f ([VikhamarSandberg2026] eq. 25/13), margin %.1fx, deviation "
                    "vs pinned %.2e\n",
                    row.kmax,
                    top.x,
                    row.resolution,
                    row.x0Measured,
                    row.formula,
                    row.formula / top.x,
                    relativeDeviation);
    }
}

TEST(BoysBoundaryTest, InlineReferenceMatchesCommittedGrid) {
    // Self-test of the inline series: it and the CSV evaluate the same [VikhamarSandberg2026] eq. 26
    // series (CSV at 30 mpmath digits, rounded to double), so two roundings bound their agreement.
    // The "~1e-17" assumed the 80-bit premise, MSVC's 64-bit long double giving ~1e-16 instead, which
    // kCsvAgreementTolerance reflects; it catches a transcription error, 1-2 orders below the 5e-14.
    const std::vector<CsvRow> rows = LoadCsvRows();
    ASSERT_GT(rows.size(), 500u);

    double worstRelative = 0.0;
    int worstN = -1;
    double worstX = 0.0;

    for (const CsvRow& row : rows)
    {
        const double inlineValue = static_cast<double>(SeriesReference(row.n, row.x));
        const double relativeError = std::abs(inlineValue - row.value) / row.value;
        EXPECT_LE(relativeError, kCsvAgreementTolerance) << "n=" << row.n << " x=" << row.x;

        if (relativeError > worstRelative)
        {
            worstRelative = relativeError;
            worstN = row.n;
            worstX = row.x;
        }
    }

    std::printf("BoysBoundary: inline long-double series vs committed grid: "
                "worst relative |error| = %.3e at (n=%d, x=%g)\n",
                worstRelative,
                worstN,
                worstX);
}
