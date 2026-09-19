// Recursion-boundaries harness (the boundary record):
// measures where the erf-seeded upward recursion leaves the
// 5e-14 absolute window of the V&S eq. 26 series reference, per kmax in
// {4, 8, 16, 32}, and pins the four recorded cells within
// kMeasuredTolerance (the recorded cells adopted the
// first-failure-at-0.0001 record - see the reconciliation note on
// kThresholdRows). Both halves of the recursion are implemented here: no
// std::erf exists in the library's sources (region B seeds from the
// Chebyshev fit), so the harness carries the erf seed itself and mirrors
// the shipped upward step exactly.

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

// The boundary threshold: the recursion stays within this absolute window of
// the reference for all n <= kmax (the record's 5e-14, shared with the
// accuracy cells).
constexpr double kBoundaryThreshold = 5e-14;

// The measurement resolution. The pin is the largest failing sample of a
// descending sweep at this step, so the step is part of the cell, not an
// implementation detail: a different step is a different measurement. 1e-4 is
// the record's own resolution - the cells ARE its first failing
// samples - so the measurement and the record stay like for like. The step is
// absolute, not scaled by the cell, matching how the record was made. Cost of
// the sweep at this step: ~1.3 s at kmax = 32 (the swath from the sweep start
// down to the first failure), ~0.8 s at kmax = 4.
constexpr double kMeasurementResolution = 1e-4;

// Asserted agreement of the measured cell with the pin recorded below (the
// recording machine's first failing sample of the 1e-4 descending sweep, see
// kThresholdRows; the boundary record's cells carry the same record as
// x0Recorded). Both sides of the comparison are the same KIND of value now - a
// first failing sample at the same stated resolution - so the band absorbs
// the ulp-level seed/libm spread only, and no longer a resolution difference.
//
// The band is unchanged at 20% and is NOT widened to cover scatter. What it
// must cover is the measurement's own scatter, MEASURED over 12 lattice
// variants per cell (glibc x86_64): 6.1% / 3.8% / 2.3% / 0.5% of the cell for
// kmax = 4/8/16/32, and the deviations from the recorded cells over all 48
// draws run -12.3%..+1.7%. So the band keeps room over the scatter it must
// absorb while still failing on a broken seed/step/reference, which is the
// gate's purpose. INJECTED-REGRESSION EVIDENCE (glibc x86_64, through this
// measurement): a seed biased by 1e-15 relative (~10 ulp) moves the kmax = 4
// cell to 0.6151 = +33.0% -> RED; a single-precision seed, a dropped +0.5 in
// the upward step's coefficient, and a typo'd denominator in the reference
// series each break the sweep-start guard -> RED. A tail floor of 1e-12 in
// place of 1e-26 is NOT a regression at this gate: it changes the reference in
// long double but not in the double the predicate compares, measured directly.
constexpr double kMeasuredTolerance = 0.2;

// Series term cap and truncation floor, mirroring the committed generator
// (tools/gen_boys_coefficients.py, boys_ref: range(300), break at
// term < 1e-26 after l > 20).
constexpr int kMaxSeriesTerms = 300;
constexpr long double kSeriesTailFloor = 1e-26L;

// Inline-series vs committed-grid agreement (relative; see the cross-check
// test): ~3.7x the measured worst on MSVC's 64-bit long double
// (1.340e-15 at (n=6, x=40) - measured on this machine against the grid
// regenerated at 45 digits). The grid this replaced measured 3.200e-15 at
// (n=31, x=38.3119): that was the GRID's own argument rounding (F_n was
// evaluated at the full-precision argument while the reader parses the x
// column as a double, worth ~2.5e-15 at d ln F / d ln x ~ 31), not the
// series'. The tolerance is unchanged - the reference got stronger, the bar
// did not move.
constexpr double kCsvAgreementTolerance = 5e-15;

// Toolchain premise pin (the "(80-bit on MSVC)" premise is false): MSVC's long
// double is 64-bit - a synonym for double - while GCC/Clang carry the x87
// 80-bit type. The reference series degrades to double precision on MSVC
// (~1e-16 relative instead of ~1e-19); F_n values are O(1) or smaller in
// the sweep, so the reference error stays 2-3 orders below kBoundaryThreshold
// and the boundary measurement is unaffected (verified by the CSV
// cross-check, whose tolerance reflects this).
static_assert(sizeof(long double) == 8 || sizeof(long double) == 16);

// F0(x) = 0.5 sqrt(pi/x) erf(sqrt(x)) - the erf seed, implemented in-test
// (the measurement definition; std::erf is deliberately not added to
// the production library).
double ErfSeedF0(double x) {
    return 0.5 * std::sqrt(std::numbers::pi / x) * std::erf(std::sqrt(x));
}

// F_n(x) via the provably stable series (V&S eq. 26): all terms positive,
// F_n(x) = 0.5 e^-x sum_l x^l / prod_{j=0..l} (n + j + 0.5). The same series
// the committed reference grid is generated from (gen_boys_coefficients.py
// boys_ref), evaluated here in long double (see the static_assert above).
// (n, x) are the order and the argument - convertible, never swapped.
//
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
    // (libstdc++ has no std::expl); the long double overload
    // is selected by the argument.
    return std::exp(-xL) * 0.5L * sum;
}

// The shipped region-B upward step (expx = 0.5 e^-x hoisted once, then
// f = ((l + 0.5) f - expx) / x), seeded from the erf definition instead of
// the Chebyshev fit. Runs in double: the boundary is a property of fp64
// arithmetic.
// (n, x) are the order and the argument - convertible, never swapped.
//
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

// Pass criterion: |recursion(n, x) - reference(n, x)| <= 5e-14 for all
// n in 0..kmax. (kmax, x) are the order cap and the argument - convertible,
// never swapped.
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

// THE MEASUREMENT: the largest failing sample of a descending sweep from the
// sweep start at kMeasurementResolution. The value is an EXISTENCE WITNESS -
// the criterion provably fails there, so it is a rigorous lower bound on the
// top of the failure set - bracketed above by the sampled point one step
// higher, which passes. It is a resolution-limited record by construction, and
// the resolution travels with the pin (ThresholdRow::resolution).
//
// The two-phase sweep this replaced (0.01 down to the first failure, then
// 0.001 below it) was ILL-CONDITIONED, not merely coarse: its value was a
// lattice draw, not a property of the function, because it reported the first
// point of a CONTIGUOUS PASS RUN at a coarse step. The failure set is a sparse
// mixture, so a pass run ends wherever the lattice happens to cross a failure
// band, and a coarser lattice stops earlier. MEASURED on one fixed machine
// (glibc x86_64, gcc 15.2, same libm, same function): moving ONLY the sweep
// start by 1e-4 moved the kmax = 4 cell over 0.3157..0.4010 - a 27.0% spread,
// larger than the whole 20% band - and kmax = 8 over 1.3317..1.5121 (13.6%).
// That is why glibc-aarch64's 0.361 (21.9%, red) needs no platform theory: it
// is an ordinary draw of this machine's own spread (offsets 0.0002/0.0005 give
// 0.3582/0.3605 here).
//
// Bisection to a fixed tolerance was tried and REJECTED as the fix: the pass
// predicate is not monotone over the transition (the amplified seed rounding
// oscillates through the 5e-14 line; 402-3351 alternations per band at 1e-4),
// so there is no crossing to bisect to. MEASURED: bisection over the same
// lattice-phase sweep gave a 15.85% spread with deviations -26%..-13.5% - the
// same distribution as the sweep it would have replaced. It moves the failure;
// it does not fix it.
//
// Reporting a FAILING point instead is what makes the value a property of the
// function: no run of samples is required to pass, only that one sample lies
// outside the window. MEASURED stability of THIS measurement (the scan from the
// sweep start; 12 lattice variants each - steps 1e-4/1.3e-4/7e-5/5e-5 x starts
// 12.0/12.0+1e-7/12.0+1e-3, glibc x86_64): the value spreads 6.1% / 3.8% / 2.3%
// / 0.5% of the cell for kmax = 4/8/16/32, with deviations from the recorded
// cells of -12.3%..-6.2% / -7.2%..-3.4% / -0.5%..+1.7% / -0.4%..+0.1%. Worst
// deviation over all 48 draws: -12.3%, inside the band with ~8 points of
// margin (the retired draw reds this same machine's kmax = 4 at -31.7%).
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
    // amplification is O(1) for all n <= 32 (seed error ~1e-16 -> ~2e-16),
    // far below the threshold. This guard is kept from the sweep it replaced:
    // without a passing start there is no failure region to find.
    if (!PassesThreshold(kmax, kSweepStart))
    {
        return result;
    }

    result.startPasses = true;

    // Descend at the measured resolution and stop at the FIRST failing sample.
    // Starting from the sweep start (not from a coarse first failure) is what
    // keeps the result independent of the recorded cell: the search window is
    // never chosen from the value it checks.
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

// Boundary rows: per kmax, x0Measured (the asserted pin - the
// recording machine's first failing sample of the 0.0001 descending sweep,
// within kMeasuredTolerance), x0Recorded (the recorded cell - the same record),
// and the published formula (V&S eq. 25/13, arXiv:2512.10059 - literature
// input, hardcoded, not measured). Margins = formula / x0 are computed
// ratios in the test, not asserted independently.
//
// RECONCILIATION RECORD - THREE GENERATIONS (the record-and-investigate
// protocol: "if a machine deviates beyond it, record and investigate before
// editing the record"): (1) the ORIGINAL cells (0.244/0.754/3.82/9.70)
// were
// measurements of the design-study harness's coarse x*1.02 geometric grid
// (first passing grid point), reproduced
// bit-for-bit by simulating that exact grid (0.244/0.754/2.069/3.823/5.910/
// 9.697). The recursion error is NOT monotone over the transition (the
// "empirically monotone" premise is false): the amplified seed
// rounding error oscillates through the 5e-14 line hundreds of times, so
// the coarse grid skipped over failure bands and its cells are grid
// artifacts. (2) The fine two-phase sweep (0.01 then 0.001
// steps) then measured 0.393/1.484/4.150/9.866 on this machine's pinned
// builds (fresh builds draw 0.401/1.435 for kmax = 4/8), and the writing
// pass adopted those cells - but the 0.0001-resolution post-check refuted
// them too: failing 1e-4 samples sit above the
// cells (+17.7%/+10.3%/+2.1%/+1.9% for kmax = 4/8/16/32), with 402-3351
// pass/fail alternations per band, so the 0.001 cells are themselves lattice
// artifacts of the two-phase sweep. (3) CURRENT: the record presents the cells
// as first-failure-at-resolution values
// - the first failing samples of the 0.0001 descending sweep (0.4625/1.6373/
// 4.2367/10.0492 - the x0Recorded column below, which the record adopted), with
// the passing sample one 1e-4 step above each (0.4626/1.6374/
// 4.2368/10.0493); the record is explicitly resolution-limited, not a
// stability threshold. The recording machine's 1e-5 scans over the
// envelope-top neighborhoods extend the largest failing samples to
// ~0.4625/1.64959/4.29768/10.05917 (the extended record). The pins
// below guard these cells with the same ±20% band: this machine's own
// two-phase measurement is a coarser (0.001) lattice draw of the same
// oscillating envelope, and the band absorbs both the libm roundings and
// the resolution difference while still failing on a broken seed/step/
// reference.
//
// (4) CURRENT MEASUREMENT (this change): generations (1)-(3) all measured a
// LATTICE DRAW - the first point of a contiguous PASS RUN at a chosen step -
// and a draw is not reproducible across machines. The gate now measures the
// FAILURE TOP instead: the largest failing sample of a 1e-4 descending sweep
// from the sweep start (see MeasureFailureTop). MEASURED on one fixed machine,
// holding libm and function still and moving only the lattice: the retired
// two-phase draw spread 27.0% at kmax = 4 (0.3157..0.4010) and 13.6% at
// kmax = 8, while the failure top spreads 4.4% and 1.2%. The retired draw is
// what reddened glibc-aarch64 (0.361 = 21.9%) - and that value sits inside the
// retired measurement's own 27% spread on x86_64, so it was never evidence
// about aarch64. The recorded cells are unchanged: they were always first
// failing samples at 1e-4, which is exactly what is now measured.
struct ThresholdRow {
    int kmax;
    double x0Measured;
    double x0Recorded;
    double formula;
    double resolution; // the step of the sweep that produced x0Measured/x0Recorded
};

const std::array<ThresholdRow, 4> kThresholdRows = {
    ThresholdRow{4, 0.4625, 0.4625, 1.60, kMeasurementResolution},
    ThresholdRow{8, 1.6373, 1.6373, 3.07, kMeasurementResolution},
    ThresholdRow{16, 4.2367, 4.2367, 6.01, kMeasurementResolution},
    ThresholdRow{32, 10.0492, 10.0492, 11.9, kMeasurementResolution},
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
        // The committed grid ends at x = 100: beyond it every F_n is covered
        // by region C's asymptotic form (see boys_test.cpp; the filter guards
        // against a future grid extension past the series' convergence
        // limit, ~x = 250 in the generator).
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
        // found no boundary where the record has one.
        ASSERT_TRUE(top.startPasses)
            << "kmax=" << row.kmax << ": no pass at the sweep start x = 12.0";
        ASSERT_TRUE(top.found) << "kmax=" << row.kmax
                               << ": no failing sample above the sweep floor (x = 0.001)";

        // The measured cell is the failure top at row.resolution - a first
        // failing sample, the same kind of record as the pin - so this is a
        // like-for-like comparison and the band covers the seed/libm spread.
        const double relativeDeviation = std::abs(top.x - row.x0Measured) / row.x0Measured;
        EXPECT_LE(relativeDeviation, kMeasuredTolerance)
            << "kmax=" << row.kmax << ": measured failure top x0 = " << top.x << " at resolution "
            << row.resolution << " vs pinned " << row.x0Measured;
        const double recordedDeviation = std::abs(top.x - row.x0Recorded) / row.x0Recorded;
        std::printf("BoysBoundary kmax=%2d: failure top x0 = %.6f at resolution %.0e, paper %.4f, "
                    "formula %.2f (V&S eq. 25/13), margin %.1fx, deviation vs paper %.2e\n",
                    row.kmax,
                    top.x,
                    row.resolution,
                    row.x0Recorded,
                    row.formula,
                    row.formula / top.x,
                    recordedDeviation);
    }
}

TEST(BoysBoundaryTest, InlineReferenceMatchesCommittedGrid) {
    // Self-test of the inline series: both it and the CSV evaluate the same
    // V&S eq. 26 series (the CSV at 30 mpmath digits, rounded to double), so
    // their agreement is bounded by the two roundings. The "agree to
    // ~1e-17" assumed the 80-bit long-double premise; with MSVC's 64-bit
    // long double the inline error is ~1e-16 relative, so the asserted
    // agreement (kCsvAgreementTolerance, ~5x the measured worst on MSVC)
    // reflects that - the gate's purpose is only to catch a transcription
    // error in the series, 1-2 orders below the 5e-14 boundary threshold.
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
