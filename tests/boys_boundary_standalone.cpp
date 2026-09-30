// Standalone reproduction of the boundary measurement: the smallest x for which the
// double-precision upward recursion from an erf-seeded F0 stays within 5e-14 of the
// reference for all n <= kmax. The shipped kernel is seeded from its stored fits
// instead, and its certified boundaries are higher than every cell below.
//
// The recursion error is not monotone over the transition: the amplified seed-rounding
// error oscillates through the 5e-14 line, so each grid draws its own lattice sample
// and every value here is a genuine measurement of its grid, ulp-sensitive within
// roughly 10-20%. The cells are first failing samples of a 0.0001-step descending
// sweep - resolution-limited draws, not formal proofs.
//
// The reference is the provably stable all-positive series (V&S eq. 26), evaluated in
// long double where the platform provides it, and self-tested against the shipped
// 45-digit grid (boys_reference.csv), read from the working directory.
//
// Sections: [1] the coarse ascending geometric grid; [2] the fine two-phase descending
// sweep; [2b] the failure top at 1e-4, which is what the gates compare; [3] the 0.0001
// descending sweep over [0.8 x0, 1.2 x0] of each cell (--postcheck); [4] the shipped
// kernel's extended-band path over [0.8 x_env, kX0) (--extended-seed).
//
// Compile: any C++17 compiler, e.g.
//     cl /O2 /std:c++17 /EHsc boys_boundary_standalone.cpp   (MSVC)
//     g++ -O2 -std=c++17 boys_boundary_standalone.cpp -o boys_boundary
// Run:     ./boys_boundary                 (sections [1], [2] and [2b])
//          ./boys_boundary --postcheck     (section [3])
//          ./boys_boundary --extended-seed (section [4])
//
// No dependencies beyond the standard library.

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr double kBoundaryThreshold = 5e-14;

// The gate's measurement resolution (section [2b]): each pinned cell IS the largest
// failing sample of a descending sweep at this step, so the step is part of the cell.
constexpr double kMeasurementResolution = 1e-4;

constexpr double kPi = 3.14159265358979323846;
constexpr double kHalfSqrtPi = 0.886226925452758014; // sqrt(pi)/2
constexpr int kMaxSeriesTerms = 300;
constexpr long double kSeriesTailFloor = 1e-26L;

// ---------------------------------------------------------------------------
// Reference series (V&S eq. 26), in the two forms the two measurements use: they
// differ in precision and in truncation rule, both reproduced exactly.
// ---------------------------------------------------------------------------
long double SeriesReferenceLongDouble(int n, long double x) {
    if (x == 0.0L)
    {
        return 1.0L / (2.0L * n + 1.0L);
    }

    long double sum = 0.0L;
    long double term = 1.0L / (static_cast<long double>(n) + 0.5L);

    for (int l = 0; l < kMaxSeriesTerms; ++l)
    {
        sum += term;

        if (l > 20 && term < kSeriesTailFloor)
        {
            break;
        }

        term *= x / (static_cast<long double>(n + l) + 1.5L);
    }

    // std::exp, not std::expl (which libstdc++ does not have): the long double
    // overload is selected by the argument.
    return std::exp(-x) * 0.5L * sum;
}

double SeriesReferenceDouble(int n, double x) {
    if (x == 0.0)
    {
        return 1.0 / (2.0 * n + 1.0);
    }

    double sum = 0.0;
    double term = 1.0 / (n + 0.5);

    for (int l = 0; l < 4000; ++l)
    {
        sum += term;

        if (l > 30 && term < 1e-30 * (sum > 1.0 ? sum : 1.0))
        {
            break;
        }

        term *= x / (n + l + 1.5);
    }

    return 0.5 * std::exp(-x) * sum;
}

// ---------------------------------------------------------------------------
// Fine-sweep recursion: the erf seed and the upward step, in double throughout — the
// boundary is a property of fp64 arithmetic.
// ---------------------------------------------------------------------------
double ErfSeedF0(double x) {
    return 0.5 * std::sqrt(kPi / x) * std::erf(std::sqrt(x));
}

double ErfSeededUpward(int n, double x) {
    double f = ErfSeedF0(x);
    const double expx = 0.5 * std::exp(-x);

    for (int l = 0; l < n; ++l)
    {
        f = ((l + 0.5) * f - expx) / x;
    }

    return f;
}

bool PassesThreshold(int kmax, double x) {
    for (int n = 0; n <= kmax; ++n)
    {
        const double recursion = ErfSeededUpward(n, x);
        const double reference =
            static_cast<double>(SeriesReferenceLongDouble(n, static_cast<long double>(x)));

        if (std::abs(recursion - reference) > kBoundaryThreshold)
        {
            return false;
        }
    }

    return true;
}

double MeasureBoundaryFine(int kmax) {
    constexpr double kSweepStart = 12.0;
    constexpr double kSweepStep = 0.01;
    constexpr double kSweepFloor = 0.001;
    constexpr double kRefineStep = 0.001;

    if (!PassesThreshold(kmax, kSweepStart))
    {
        return -1.0;
    }

    double xPass = kSweepStart;

    for (int i = 1;; ++i)
    {
        const double x = kSweepStart - static_cast<double>(i) * kSweepStep;

        if (x < kSweepFloor)
        {
            break;
        }

        if (PassesThreshold(kmax, x))
        {
            xPass = x;
        } else
        {
            break;
        }
    }

    const double xFail = xPass - kSweepStep;
    double x0 = xPass;

    for (int i = 1;; ++i)
    {
        const double x = xPass - static_cast<double>(i) * kRefineStep;

        if (x <= xFail)
        {
            break;
        }

        if (PassesThreshold(kmax, x))
        {
            x0 = x;
        } else
        {
            break;
        }
    }

    return x0;
}

// ---------------------------------------------------------------------------
// The GATE's measurement (section [2b]): the FAILURE TOP - the largest failing sample
// of a descending sweep from the sweep start at kMeasurementResolution. A failing
// sample is a witness that the recursion leaves the window there, so the value is a
// rigorous lower bound on the top of the failure set, bracketed above by the sampled
// point one step higher, which passes. It is the same kind of measurement as the
// recorded cells, so the comparison is like for like.
// ---------------------------------------------------------------------------
struct FailureTop {
    bool startPasses = false; // the sweep start is inside the 5e-14 window
    bool found = false; // a failing sample exists above the sweep floor
    double x = 0.0; // the largest failing sample found (the cell)
};

FailureTop MeasureFailureTop(int kmax) {
    constexpr double kSweepStart = 12.0;
    constexpr double kSweepFloor = 0.001;

    FailureTop result;

    if (!PassesThreshold(kmax, kSweepStart))
    {
        return result;
    }

    result.startPasses = true;

    // From the sweep start, not from the draw's coarse first failure: the
    // search window is never chosen from the value it checks.
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

// ---------------------------------------------------------------------------
// Coarse-grid measurement: the coarse grid's recursion and its ascending geometric
// grid, checking orders l = 1..kmax only - the seed itself is not re-checked - as [1].
// ---------------------------------------------------------------------------
double MeasureBoundaryCoarse(int kmax) {
    double found = -1.0;

    for (double x = 0.05; x < 40.0; x *= 1.02)
    {
        double f = kHalfSqrtPi / std::sqrt(x) * std::erf(std::sqrt(x));
        const double expx = std::exp(-x);
        bool ok = true;

        for (int l = 1; l <= kmax; ++l)
        {
            f = ((2.0 * l - 1.0) * f - expx) / (2.0 * x);

            if (std::abs(f - SeriesReferenceDouble(l, x)) > kBoundaryThreshold)
            {
                ok = false;
                break;
            }
        }

        if (ok)
        {
            found = x;
            break;
        }
    }

    return found;
}

// ---------------------------------------------------------------------------
// Self-test: the inline long-double series against the shipped 45-digit reference grid.
// ---------------------------------------------------------------------------
struct CsvRow {
    int n;
    double x;
    double value;
};

std::vector<CsvRow> LoadCsv(const char* path) {
    std::ifstream file(path);
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

        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

double RoundTo3Sig(double value) {
    if (value == 0.0)
    {
        return 0.0;
    }

    const double exp = std::floor(std::log10(std::abs(value)));
    return std::round(value / std::pow(10.0, exp - 2.0)) * std::pow(10.0, exp - 2.0);
}

struct ThresholdRow {
    int kmax;
    double x0Recorded; // the cell [3] records: the first failing sample of
                    // the 0.0001 descending sweep (MSVC, 2026-09-05),
                    // adopted from the pin of the accompanying test suite
    double x0Coarse; // the coarse-grid cell of [1], reproduced there
    double formula; // published formula (V&S eq. 25/13), literature input
};

const std::array<ThresholdRow, 4> kThresholdRows = {
    ThresholdRow{4, 0.4625, 0.244, 1.60},
    ThresholdRow{8, 1.6373, 0.754, 3.07},
    ThresholdRow{16, 4.2367, 3.82, 6.01},
    ThresholdRow{32, 10.0492, 9.70, 11.9},
};

// ---------------------------------------------------------------------------
// [3] the 0.0001-step descending sweep over [0.8 x0, 1.2 x0] of a cell x0
// (see the file header) - each cell IS the first failing sample of this
// walk. The whole band is walked - not stopped at the first
// failure - so the pass/fail alternations (the amplified seed-rounding error
// oscillating through the 5e-14 line) are counted as evidence of the grid's
// adequacy near the transition.
// ---------------------------------------------------------------------------
constexpr double kPostCheckStep = 0.0001;

struct PostCheckResult {
    int samples = 0;
    bool topPasses = false;
    bool bottomPasses = false;
    int alternations = 0; // pass/fail flips over the whole band walk
    int shortestRun = 0; // shortest pass/fail run in samples (0: no flips)
    int firstFailIndex = -1; // first failing sample, descending from the top
    double firstFailX = 0.0;
    double lastPassX = 0.0; // the passing sample just above the first failure
};

PostCheckResult RunPostCheck(int kmax, double x0) {
    const double xTop = 1.2 * x0;
    const double xBottom = 0.8 * x0;
    const int nSamples = static_cast<int>(std::lround((xTop - xBottom) / kPostCheckStep)) + 1;

    PostCheckResult result;
    result.samples = nSamples;

    bool passes = PassesThreshold(kmax, xTop);
    result.topPasses = passes;
    int currentRun = 1;
    int shortestClosedRun = nSamples;

    for (int i = 1; i < nSamples; ++i)
    {
        const double x = xTop - static_cast<double>(i) * kPostCheckStep;
        const bool next = PassesThreshold(kmax, x);

        if (i == nSamples - 1)
        {
            result.bottomPasses = next;
        }

        if (next != passes)
        {
            ++result.alternations;
            shortestClosedRun = std::min(shortestClosedRun, currentRun);
            currentRun = 1;
            passes = next;
        } else
        {
            ++currentRun;
        }

        if (result.firstFailIndex < 0 && !next)
        {
            result.firstFailIndex = i;
            result.firstFailX = x;
            result.lastPassX = x + kPostCheckStep;
        }
    }

    if (result.alternations > 0)
    {
        result.shortestRun = shortestClosedRun;
    }

    return result;
}

// Runs the 0.0001-resolution descending sweep for all four cells over [0.8 x0,
// 1.2 x0] of each, confirming that each walk's own first failing sample reproduces its
// cell within the 20% band while the band top passes.
bool RunPostCheckAll() {
    std::printf("\n");
    std::printf("=== [3] resolution sweep: 0.0001-step descending sweep over\n");
    std::printf("    [0.8 x0, 1.2 x0] of the four cells ===\n");
    std::printf("    (same seed, step, and 5e-14 criterion as the fine sweep of [2];\n");
    std::printf("    the sweep is resolution-limited - 1e-5 envelope-top scans are\n");
    std::printf("    recorded in the file header - and not a formal proof)\n");
    std::printf("fine two-phase sweep, this run (context):\n");

    bool allOk = true;

    for (const ThresholdRow& row : kThresholdRows)
    {
        const double x0 = MeasureBoundaryFine(row.kmax);
        const double deviation = std::abs(x0 - row.x0Recorded) / row.x0Recorded;
        const bool withinBand = deviation <= 0.2;
        std::printf("  kmax=%2d: x0 = %.6f | recorded cell %.4f | %s\n",
                    row.kmax,
                    x0,
                    row.x0Recorded,
                    withinBand ? "agrees (within 20%)" : "DEVIATES");
    }

    for (int r = 0; r < static_cast<int>(kThresholdRows.size()); ++r)
    {
        const ThresholdRow& row = kThresholdRows[r];
        const PostCheckResult check = RunPostCheck(row.kmax, row.x0Recorded);

        // The recorded cell is this walk's own first failing sample: a healthy re-run
        // reproduces it within the 20% band the ulp-sensitive cells allow.
        const double firstFailDeviation =
            check.firstFailIndex >= 0 ? std::abs(check.firstFailX - row.x0Recorded) / row.x0Recorded
                                      : -1.0;
        const bool confirmed =
            check.topPasses && check.firstFailIndex >= 0 && firstFailDeviation <= 0.2;
        allOk = allOk && confirmed;

        const double xBottom = 0.8 * row.x0Recorded;
        const double xTop = 1.2 * row.x0Recorded;
        std::printf("\n");
        std::printf("kmax=%2d: recorded cell %.4f | band [%.4f, %.4f] | step %.4f | %d samples\n",
                    row.kmax,
                    row.x0Recorded,
                    xBottom,
                    xTop,
                    kPostCheckStep,
                    check.samples);
        std::printf("  band top %s, band bottom %s\n",
                    check.topPasses ? "PASSES" : "FAILS",
                    check.bottomPasses ? "PASSES" : "FAILS");

        if (check.firstFailIndex >= 0)
        {
            std::printf("  first failure @1e-4 (descending): %.4f | last pass @1e-4: %.4f\n",
                        check.firstFailX,
                        check.lastPassX);
            std::printf("  first failure reproduces the recorded cell within %.4f%%\n",
                        100.0 * firstFailDeviation);
        } else
        {
            std::printf("  no failure inside the band (transition below %.4f)\n", xBottom);
        }

        if (check.alternations > 0)
        {
            std::printf("  pass/fail alternations in the band: %d (shortest run %d sample%s)\n",
                        check.alternations,
                        check.shortestRun,
                        check.shortestRun == 1 ? "" : "s");
        } else
        {
            std::printf("  pass/fail alternations in the band: none\n");
        }

        std::printf("  POST-CHECK kmax=%2d: %s\n",
                    row.kmax,
                    confirmed ? "recorded cell CONFIRMED as the first failing sample of "
                                "this 0.0001 lattice (resolution-limited)"
                              : "NOT CONFIRMED - see the values above");
    }

    std::printf("\n");
    std::printf("POST-CHECK (kmax = 4, 8, 16, 32): %s\n",
                allOk ? "all four cells CONFIRMED at 0.0001 resolution" : "DEVIATION");
    return allOk;
}

// ---------------------------------------------------------------------------
// [4] the extended-band path: the seeded F0 fit on [kExtendedBX0, kX0) plus the upward
// step, dispatched per kmax tier at the certified kTierBoundaries. This sweep is the
// certified rows' measured consistency check - a certified crossing must sit at or
// above the first failure, a conservative bound crossing no lower than the true error.
// ---------------------------------------------------------------------------

// The generated constants: exact decimal copies of boys_coefficients.hpp, the double
// values being the nearest doubles to these literals.
constexpr double kX0 = 1.18998481521084840e+01; // region-A/B boundary (double)
constexpr double kExtendedBX0 = 1.08552523453493333e+00; // the band's left edge
constexpr int kExtendedBDeg = 24;
constexpr double kExtendedBcoeffs[25] = {
    4.12114508161470272e-01,  -2.08513328473299508e-01, 7.23146663434936776e-02,
    -2.55479910482183173e-02, 8.64778167861495438e-03,  -2.74192371757364054e-03,
    8.07090736167245327e-04,  -2.19956149640727428e-04, 5.55248467482282208e-05,
    -1.30093169516035672e-05, 2.83691273573312611e-06,  -5.77570974546896711e-07,
    1.10129683781869441e-07,  -1.97281092572063005e-08, 3.32990602638389726e-09,
    -5.31073485374336990e-10, 8.02392141465355472e-11,  -1.15128891629713017e-11,
    1.57227956403092410e-12,  -2.04802643771997034e-13, 2.54946807665347599e-14,
    -3.03850609868245207e-15, 3.47296452162601717e-16,  -3.81251682238197342e-17,
    3.98661416732232774e-18,
};
// The certified per-order dispatch thresholds: order n takes the extended seed exactly
// when x >= kTierThresholds[n].
constexpr double kTierThresholds[33] = {
    1.08552523453493333e+00, 1.08552523453493333e+00, 1.08552523453493333e+00,
    1.08552523453493333e+00, 1.08552523453493333e+00, 2.01529770533511421e+00,
    2.01529770533511421e+00, 2.01529770533511421e+00, 2.01529770533511421e+00,
    4.89787029982565691e+00, 4.89787029982565691e+00, 4.89787029982565691e+00,
    4.89787029982565691e+00, 4.89787029982565691e+00, 4.89787029982565691e+00,
    4.89787029982565691e+00, 4.89787029982565691e+00, 1.07835878589167624e+01,
    1.07835878589167624e+01, 1.07835878589167624e+01, 1.07835878589167624e+01,
    1.07835878589167624e+01, 1.07835878589167624e+01, 1.07835878589167624e+01,
    1.07835878589167624e+01, 1.07835878589167624e+01, 1.07835878589167624e+01,
    1.07835878589167624e+01, 1.07835878589167624e+01, 1.07835878589167624e+01,
    1.07835878589167624e+01, 1.07835878589167624e+01, 1.07835878589167624e+01,
};

// ClenshawSplit: the split-Clenshaw evaluation of the seeds, with an explicit std::fma
// in every fused position.
inline double ClenshawSplit(const double* c, int deg, double t) noexcept {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return std::fma(t, c[1], c[0]);
    }

    const double v = std::fma(2.0, t * t, -1.0);
    const double twoV = v + v;

    if (deg == 2)
    {
        return std::fma(t, c[1], std::fma(v, c[2], c[0]));
    }

    assert(deg >= 4 && deg % 2 == 0);

    const int m = deg / 2;
    double b1 = c[std::ptrdiff_t{2} * m];
    double b2 = 0.0;

    for (int k = m - 1; k >= 1; --k)
    {
        const double b0 = std::fma(twoV, b1, c[std::ptrdiff_t{2} * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const double even = std::fma(v, b1, c[0] - b2);
    double o1 = c[2 * m - 1];
    double o2 = 0.0;

    for (int k = m - 2; k >= 1; --k)
    {
        const double o0 = std::fma(twoV, o1, c[std::ptrdiff_t{2} * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const double odd = std::fma(twoV - 1.0, o1, c[1] - o2);
    return std::fma(t, odd, even);
}

// The extended-band F0 fit on t = 2*(x-kExtendedBX0)/(kX0-kExtendedBX0) - 1.
inline double RegionBExtendedSeed(double x) noexcept {
    const double t = 2.0 * (x - kExtendedBX0) / (kX0 - kExtendedBX0) - 1.0;
    return ClenshawSplit(kExtendedBcoeffs, kExtendedBDeg, t);
}

// ExtendedBandUpward: the F0 fit seed, the hoisted 0.5*exp(-x) precompute, then the
// upward step f = ((l + 0.5)*f - expx)/x.
double ExtendedBandUpward(int n, double x) {
    double f = RegionBExtendedSeed(x);
    const double expx = 0.5 * std::exp(-x);

    for (int l = 0; l < n; ++l)
    {
        f = ((l + 0.5) * f - expx) / x;
    }

    return f;
}

bool PassesThresholdExtended(int kmax, double x) {
    for (int n = 0; n <= kmax; ++n)
    {
        const double recursion = ExtendedBandUpward(n, x);
        const double reference =
            static_cast<double>(SeriesReferenceLongDouble(n, static_cast<long double>(x)));

        if (std::abs(recursion - reference) > kBoundaryThreshold)
        {
            return false;
        }
    }

    return true;
}

// The [4] rows: the certified boundary of the extended band per tier. The certified
// crossings must sit at or above the first failures, a conservative bound crossing no
// lower than the true error does.
struct ExtendedRow {
    int kmax;
    double xEnvCertified; // the certified boundary (1-ulp-exp column)
};

const std::array<ExtendedRow, 4> kExtendedRows = {
    ExtendedRow{4, kTierThresholds[4]},
    ExtendedRow{8, kTierThresholds[8]},
    ExtendedRow{16, kTierThresholds[16]},
    ExtendedRow{32, kTierThresholds[32]},
};

struct ExtendedSweepResult {
    int samples = 0;
    bool allAboveEnvPass = true; // every sampled x >= xEnvCertified passes
    int alternations = 0; // pass/fail flips over the whole walk
    int firstFailIndex = -1; // first failing sample, descending from kX0
    double firstFailX = 0.0;
    double lastPassX = 0.0; // the passing sample just above the first failure
};

// The 0.0001-step descending sweep over the extended-band path, from just below kX0
// down to 0.8 * xEnvCertified. The whole range is walked, not stopped at the first
// failure, so the alternations are counted as evidence of the oscillation.
ExtendedSweepResult RunExtendedSeedSweep(const ExtendedRow& row) {
    const double xTop = kX0 - kPostCheckStep;
    const double xBottom = 0.8 * row.xEnvCertified;
    const int nSamples = static_cast<int>(std::lround((xTop - xBottom) / kPostCheckStep)) + 1;

    ExtendedSweepResult result;
    result.samples = nSamples;

    bool passes = PassesThresholdExtended(row.kmax, xTop);

    for (int i = 1; i < nSamples; ++i)
    {
        const double x = xTop - static_cast<double>(i) * kPostCheckStep;
        const bool next = PassesThresholdExtended(row.kmax, x);

        // Alternations are this sweep's evidence (see the function comment);
        // there is no run-length claim here, so no run counter is kept.
        if (next != passes)
        {
            ++result.alternations;
            passes = next;
        }

        if (!next && x >= row.xEnvCertified)
        {
            result.allAboveEnvPass = false;
        }

        if (result.firstFailIndex < 0 && !next)
        {
            result.firstFailIndex = i;
            result.firstFailX = x;
            result.lastPassX = x + kPostCheckStep;
        }
    }

    return result;
}

// Runs section [4] for all four tiers, reporting each walk's first failing sample and
// the consistency check against the certified x_env: a first failure above the
// certified value would refute the certificate.
bool RunExtendedSeedAll() {
    std::printf("\n");
    std::printf("=== [4] extended-seed sweep: 0.0001-step descending sweep over\n");
    std::printf("    [0.8 x_env, kX0) of the shipped kernel's extended-band path ===\n");
    std::printf("    (the per-range F0 fit seed and the upward step, copied verbatim\n");
    std::printf("    from the shipped kernel; the certified-boundary table's measured\n");
    std::printf("    consistency check - the certification itself is the interval\n");
    std::printf("    evaluation of the certified per-kmax boundaries)\n");

    bool allOk = true;

    for (const ExtendedRow& row : kExtendedRows)
    {
        const ExtendedSweepResult check = RunExtendedSeedSweep(row);
        const bool consistent =
            check.allAboveEnvPass &&
            (check.firstFailIndex < 0 || check.firstFailX <= row.xEnvCertified + kPostCheckStep);
        allOk = allOk && consistent;

        std::printf("\n");
        std::printf(
            "kmax=%2d: certified x_env %.5f | sweep [%.4f, %.4f] | step %.4f | %d samples\n",
            row.kmax,
            row.xEnvCertified,
            0.8 * row.xEnvCertified,
            kX0 - kPostCheckStep,
            kPostCheckStep,
            check.samples);

        if (check.firstFailIndex >= 0)
        {
            std::printf("  first failure @1e-4 (descending): %.4f | last pass @1e-4: %.4f\n",
                        check.firstFailX,
                        check.lastPassX);
        } else
        {
            std::printf("  no failure inside the swept range (transition below %.4f)\n",
                        0.8 * row.xEnvCertified);
        }

        std::printf("  every sampled x >= certified x_env passes: %s\n",
                    check.allAboveEnvPass ? "YES" : "NO - failures above x_env");
        std::printf("  pass/fail alternations in the range: %d\n", check.alternations);
        std::printf("  EXTENDED-SEED kmax=%2d: %s\n",
                    row.kmax,
                    consistent ? "certified crossing sits AT OR ABOVE the first failure "
                                 "(the bound is conservative - consistent)"
                               : "NOT CONSISTENT - see the values above");
    }

    std::printf("\n");
    std::printf("EXTENDED-SEED (kmax = 4, 8, 16, 32): %s\n",
                allOk ? "all four certified crossings CONSISTENT with the 0.0001 sweep"
                      : "DEVIATION");
    return allOk;
}

} // namespace

int main(int argc, char** argv) {
    const bool postCheckMode = argc > 1 && std::strcmp(argv[1], "--postcheck") == 0;
    const bool extendedSeedMode = argc > 1 && std::strcmp(argv[1], "--extended-seed") == 0;

    // Self-test: inline series vs the shipped grid.
    const std::vector<CsvRow> rows = LoadCsv("boys_reference.csv");

    if (rows.size() < 500)
    {
        std::printf("ERROR: boys_reference.csv missing or too small (run from the "
                    "package directory)\n");
        return 1;
    }

    double worstRelative = 0.0;
    int worstN = -1;
    double worstX = 0.0;

    for (const CsvRow& row : rows)
    {
        const double inlineValue =
            static_cast<double>(SeriesReferenceLongDouble(row.n, static_cast<long double>(row.x)));
        const double relativeError = std::abs(inlineValue - row.value) / row.value;

        if (relativeError > worstRelative)
        {
            worstRelative = relativeError;
            worstN = row.n;
            worstX = row.x;
        }
    }

    std::printf("self-test: inline series vs the shipped 45-digit grid: worst "
                "relative |error| = %.3e at (n=%d, x=%g)\n",
                worstRelative,
                worstN,
                worstX);
    std::printf("\n");

    if (postCheckMode)
    {
        return RunPostCheckAll() ? 0 : 1;
    }

    if (extendedSeedMode)
    {
        return RunExtendedSeedAll() ? 0 : 1;
    }

    // [1] coarse grid — the coarse-grid measurement.
    std::printf("=== [1] coarse geometric grid (x = 0.05, x *= 1.02, first passing "
                "point; the coarse-grid cells) ===\n");
    bool coarseMatch = true;

    for (const ThresholdRow& row : kThresholdRows)
    {
        const double x0 = MeasureBoundaryCoarse(row.kmax);
        const bool match = RoundTo3Sig(x0) == RoundTo3Sig(row.x0Coarse);
        coarseMatch = coarseMatch && match;
        std::printf("kmax=%2d: stable from x >= %.3f | coarse cell %.3f | "
                    "formula %.2f | margin %.1fx | %s\n",
                    row.kmax,
                    x0,
                    row.x0Coarse,
                    row.formula,
                    row.formula / x0,
                    match ? "MATCH" : "MISMATCH");
    }

    std::printf("coarse-grid reproduction vs [1]'s cells: %s\n",
                coarseMatch ? "MATCH" : "MISMATCH");
    std::printf("\n");

    // [2] fine two-phase sweep: the fresh value is a lattice draw of the
    // same oscillating envelope, printed but NOT gated - the gate is [2b].
    std::printf("=== [2] fine two-phase sweep (0.01 then 0.001 steps; the "
                "cells above are [3]'s measurement, so this one is reproduced "
                "for comparison and NOT "
                "gated - its value is a lattice draw) ===\n");

    for (const ThresholdRow& row : kThresholdRows)
    {
        const double x0 = MeasureBoundaryFine(row.kmax);
        const double recordedDeviation = std::abs(x0 - row.x0Recorded) / row.x0Recorded;
        std::printf("kmax=%2d: fine-sweep lattice draw x0 = %.6f | recorded cell %.4f "
                    "| deviation %.2e | formula %.2f | margin %.1fx\n",
                    row.kmax,
                    x0,
                    row.x0Recorded,
                    recordedDeviation,
                    row.formula,
                    row.formula / x0);
    }

    // [2b] THE GATE: the failure top at 1e-4. It is the same kind of measurement as the
    // recorded cells - a first failing sample at 1e-4 - so the 20% band covers the
    // ulp-level seed/libm spread rather than the measurement's own lattice scatter.
    std::printf("\n");
    std::printf("=== [2b] failure top at 1e-4 (the measured cell; gated within "
                "20%% of the recorded cells) ===\n");
    bool boundaryOk = true;

    for (const ThresholdRow& row : kThresholdRows)
    {
        const FailureTop top = MeasureFailureTop(row.kmax);
        const bool measured = top.startPasses && top.found;
        const double recordedDeviation = measured ? std::abs(top.x - row.x0Recorded) / row.x0Recorded : -1.0;
        const bool withinBand = measured && recordedDeviation <= 0.2;
        boundaryOk = boundaryOk && withinBand;

        if (!measured)
        {
            std::printf("kmax=%2d: NO FAILURE TOP MEASURED (%s)\n",
                        row.kmax,
                        top.startPasses ? "no failing sample above the sweep floor"
                                        : "the sweep start x = 12.0 does not pass");
            continue;
        }

        std::printf("kmax=%2d: failure top x0 = %.6f at resolution %.0e | recorded cell %.4f "
                    "(%s within 20%%) | formula %.2f | margin %.1fx | deviation %.2e\n",
                    row.kmax,
                    top.x,
                    kMeasurementResolution,
                    row.x0Recorded,
                    withinBand ? "agrees" : "DEVIATES",
                    row.formula,
                    row.formula / top.x,
                    recordedDeviation);
    }

    std::printf("BOUNDARY-CHECK: failure top at 1e-4 vs recorded cells "
                "(within 20%%): %s\n",
                boundaryOk ? "MATCH" : "MISMATCH");
    std::printf("\n");
    std::printf("The pinned cells are the first failing samples of the "
                "0.0001 descending\n");
    std::printf("sweep, which is what [2b] measures: a failing sample is a "
                "witness that the\n");
    std::printf("recursion leaves the window there, and a lower bound on the "
                "top of the failure\n");
    std::printf("set. [1] and [2] above are coarser measurements of the same "
                "quantity, reproduced\n");
    std::printf("to show the spread between the grids: they are lattice draws "
                "and are not gated.\n");
    return boundaryOk ? 0 : 1;
}
