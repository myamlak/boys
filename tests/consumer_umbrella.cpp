// Consumer check: does the umbrella header reach everything the library
// documents, and does every returned value meet the bound its entry states?
//
// This translation unit is a CONSUMER of the library, not one of its tests:
//
//  * <boys/boys.hpp> is its first include and its only library include, and no
//    internal header is named. The library's own suite includes headers from
//    src/ and compiles with that directory on its include path, so a public
//    header a consumer cannot reach - or an entry whose definition never left
//    a .cpp file - passes the suite and fails a consumer. The probe below makes
//    that distinction explicit: this build fails if src/ ever reaches this
//    file's include path, because every other property of this check rests on
//    it;
//
//  * every documented public choice is exercised: the seven accuracy tiers,
//    both QueryTier overloads, the two fit routes of the double lane, the three
//    product modes of the region-A transform over both bands, the
//    sorted-argument and workspace forms of the many-argument entry, its
//    per-element-order form in both precisions, the fp16/bf16 I/O lanes, the
//    native packed half lane, and the lane templates at
//    multipliers the library does not pre-instantiate - the case that is a link
//    error when a definition lives in a .cpp file rather than in the header its
//    declaration ships in;
//
//  * every returned value is judged against the bound its entry documents,
//    with the committed 45-digit reference grid as the reference for the
//    arguments it carries. Where a format conversion sits between the reference
//    and the entry (the fp32 and fp16 lanes round their argument, the native
//    half lane returns 2^15 F_k), the comparison is made against the certified
//    double lane at the same converted argument, and the rule's name states the
//    bound that composition carries;
//
//  * the claims that hold on one arithmetic and not on another are compiled
//    against the answer this build's configure measured rather than against the
//    host's architecture, which does not decide the question: MSVC on aarch64
//    does not contract a bare product-plus-add and gcc on the same architecture
//    does, while on x86-64 no compiler measured contracts one without -mfma.
//    Two entries that evaluate the same recurrence from the same source are not
//    thereby guaranteed the same bits, because whether that bare form is one
//    rounding or two is a licence the compiler holds per call site. The two
//    entries' values agree at every cell swept here on every host measured, and
//    that is asserted wherever the run happens. The strided shape's offsets are
//    exact by construction only where the bare form is two roundings, so there
//    the bound is asserted on every build and the exact offset on the builds
//    whose configure measured that. Both counts are printed either way, so an
//    abstention is a visible figure rather than a silent one.
//
// The output is one line per rule - cells judged, the worst measured error as a
// fraction of the bound, and where that cell is - so a green run states what it
// covered and a red one names the cells that exceeded.
//
// Run:  cmake --build <build> --target boys-consumer-umbrella
//       <build>/boys-consumer-umbrella
//       ctest --test-dir <build> -R boys-consumer-umbrella

#include <algorithm>
#include <array>
#include <boys/boys.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <span>
#include <string>
#include <vector>

// The library's private src/ directory is deliberately NOT on this file's
// include path: a consumer gets include/ and nothing else. Prove it here rather
// than assume it. Each name below resolves only if a directory holding that
// file is on the include path, and those files are the library's sources; five
// names, so that renaming one does not quietly retire the probe.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it no longer proves that a consumer can build against the public headers alone"
#endif

// The build's own version, so the check can assert that the header a caller
// reads agrees with the project() call rather than the two being two answers.
// A build that drops the definition would otherwise skip the check silently.
#ifndef BoysExpectedVersion
#error "this consumer check needs BoysExpectedVersion (the project version of the build it runs in); the boys-consumer-umbrella target carries it"
#endif

namespace {

// --- the check's framework --------------------------------------------------

/// One named rule: the cells it judged and the worst ratio it saw. A ratio at
/// or below 1 is a cell inside its documented bound.
struct Rule {
    std::string name;
    std::size_t cells = 0;
    std::size_t exceeded = 0;
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;
};

/// Assertions that are not a ratio: a constant's value, two entries agreeing, a
/// buffer the caller declared empty staying untouched.
struct Report {
    std::size_t assertions = 0;
    std::size_t failed = 0;

    /// The bit-for-bit comparisons this run made of an entry's exact form, and
    /// how many held. Printed on every build; a count below the other is how a
    /// host that parts company with the exact form is named by a figure rather
    /// than by a silent pass.
    std::size_t exactCompared = 0;
    std::size_t exactHeld = 0;
};

/// A deque, not a vector: the checks hold a reference to a rule while they
/// register the next one, and a deque keeps those references valid.
std::deque<Rule> gRules;
std::size_t gPrinted = 0;
constexpr std::size_t kPrintedPerRule = 5;

void Fail(const Rule& rule, int n, double x, double measured, double reference, double bound) {
    if (gPrinted < kPrintedPerRule)
    {
        std::printf("  EXCEEDED %s n=%d x=%.17g measured=%.17g reference=%.17g bound=%.6g\n",
                    rule.name.c_str(),
                    n,
                    x,
                    measured,
                    reference,
                    bound);
        ++gPrinted;
    }
}

/// Judge one cell: |measured - reference| against the bound documented for it.
/// A value that is not a finite number is counted as exceeded rather than
/// compared, because a ratio against a bound is false for every NaN and would
/// otherwise slip through the test.
void Judge(Rule& rule, double measured, double reference, double bound, int n, double x) {
    const double error = std::abs(measured - reference);
    const double ratio = bound > 0.0 ? error / bound : 0.0;

    ++rule.cells;

    if (!std::isfinite(measured) || error > bound)
    {
        ++rule.exceeded;
        Fail(rule, n, x, measured, reference, bound);
    }

    if (ratio > rule.worst)
    {
        rule.worst = ratio;
        rule.worstN = n;
        rule.worstX = x;
    }
}

void Require(Report& report, bool ok, const char* what) {
    ++report.assertions;

    if (!ok)
    {
        ++report.failed;
        std::printf("  FAILED %s\n", what);
    }
}

/// Count one bit-for-bit comparison of an entry's exact form.
void Compare(Report& report, bool holds) {
    ++report.exactCompared;
    report.exactHeld += holds ? 1 : 0;
}

/// A rule registered up front, so the verdict lists every rule the check
/// intended to run even if it judged no cell.
Rule& NewRule(const std::string& name) {
    gRules.push_back(Rule{name, 0, 0, 0.0, -1, 0.0});
    return gRules.back();
}

/// The documented entries this run reached. The verdict prints the count, so
/// "what of the surface is covered" is a statement about the run rather than a
/// claim about this file.
std::vector<std::string> gCovered;

void Covered(const char* name) {
    if (std::find(gCovered.begin(), gCovered.end(), name) == gCovered.end())
    {
        gCovered.emplace_back(name);
    }
}

std::string RuleName(const char* format, const char* multiplier) {
    std::string name(format);
    const std::size_t at = name.find('%');
    name.replace(at, 1, multiplier);
    return name;
}

void PrintRules() {
    for (const Rule& rule : gRules)
    {
        std::printf("  %-56s %7zu cells  worst %.4g of the bound",
                    rule.name.c_str(),
                    rule.cells,
                    rule.worst);

        if (rule.worstN >= 0)
        {
            std::printf(" (n=%d, x=%.6g)", rule.worstN, rule.worstX);
        }

        std::printf("  exceeded %zu\n", rule.exceeded);
    }
}

// --- the committed reference grid -------------------------------------------

/// One cell of the committed 45-digit grid: F_n(x) to more digits than any lane
/// here returns.
struct Cell {
    int n = 0;
    double x = 0.0;
    double value = 0.0;
};

std::vector<Cell> LoadGrid(const char* path) {
    std::vector<Cell> cells;
    std::ifstream in(path);

    if (!in)
    {
        return cells;
    }

    std::string line;
    std::getline(in, line); // the header row

    while (std::getline(in, line))
    {
        const std::size_t first = line.find(',');
        const std::size_t second =
            first == std::string::npos ? std::string::npos : line.find(',', first + 1);

        if (first == std::string::npos || second == std::string::npos)
        {
            continue;
        }

        Cell cell{};
        cell.n = std::atoi(line.substr(0, first).c_str());
        cell.x = std::atof(line.substr(first + 1, second - first - 1).c_str());
        cell.value = std::atof(line.substr(second + 1).c_str());

        if (cell.value >= 0.0 && std::isfinite(cell.x) && cell.n >= 0 &&
            cell.n <= boys::kMaxBoysOrder)
        {
            cells.push_back(cell);
        }
    }

    return cells;
}

/// The grid's cell for one (order, argument), or nullptr when the grid does not
/// carry it. The grid is n-major and its argument sets are per order.
const Cell* Find(const std::vector<Cell>& cells, int n, double x) {
    for (const Cell& cell : cells)
    {
        if (cell.n == n && cell.x == x)
        {
            return &cell;
        }
    }

    return nullptr;
}

/// The distinct arguments of the grid, ascending: the batch entries take an
/// argument array, and the grid holds one cell per order at each argument.
std::vector<double> DistinctArgs(const std::vector<Cell>& cells) {
    std::vector<double> args;
    args.reserve(cells.size());

    for (const Cell& cell : cells)
    {
        args.push_back(cell.x);
    }

    std::sort(args.begin(), args.end());
    args.erase(std::unique(args.begin(), args.end()), args.end());
    return args;
}

// --- the documented bounds --------------------------------------------------

/// The double single lane's bound at x, from the published figures: m*1e-15
/// below the tightest per-order end of region A, m*3e-14 below the end of
/// region A (the extended band's figure, which covers region A as well), and
/// m*5.5e-14 past it.
double SingleBound(double x, double m) {
    if (x < 1.0855)
    {
        return m * 1e-15;
    }

    if (x < boys::kRegionAEnd)
    {
        return m * 3e-14;
    }

    return m * 5.5e-14;
}

/// The double batch lane's bound: one figure over the whole argument line.
double BatchBound(double m) {
    return m * 5.5e-14;
}

/// The quantum (one ULP) of a binary16 or bfloat16 value of the given
/// significand width: 2^(e - bits) for a normal value 1.f * 2^e, and zero at
/// zero, where no quantum is defined.
double QuantumOf(double v, int significandBits) {
    if (v == 0.0)
    {
        return 0.0;
    }

    int exponent = 0;
    (void)std::frexp(std::abs(v), &exponent);
    return std::ldexp(1.0, exponent - 1 - significandBits);
}

/// The smallest positive normal binary16 value: the floor of the documented
/// domain of the native half lane, whose claim holds where the returned value
/// is a normal half.
constexpr double kHalfMinNormal = 6.103515625e-05; // 2^-14

/// The fp16 lane's bound at a returned value: m*1e-7 plus the representation
/// term, one half-ULP of the returned half.
double F16IoBound(double returned, double m) {
    return m * 1e-7 + 0.5 * QuantumOf(returned, 10);
}

/// The bf16 lane's bound: m*1e-7 plus one half-ULP of the returned bf16, whose
/// significand is 8 bits wide.
double Bf16IoBound(double returned, double m) {
    return m * 1e-7 + 0.5 * QuantumOf(returned, 7);
}

/// The region-A product's bound per mode: the fit term at m, plus the two split
/// modes' 32-bit accumulator floor, which no multiplier moves.
double ProductBound(Report& report, boys::ProductMode mode, double m) {
    // Read out of the library's own report rather than restated here: the
    // bounds are what BoysProductModes answers, so a mode added to the
    // enumeration arrives in this sweep with its own bound instead of the
    // 2.5e-7 the split modes carry.
    for (const boys::ProductModeInfo& row : boys::BoysProductModes())
    {
        if (row.mode == mode)
        {
            return m * row.fitTerm + row.floor;
        }
    }

    Require(report, false, "every mode the sweep names has a BoysProductModes row");
    return 0.0;
}

/// One ULP of a normal half value: 2^(e - 10) for a value 1.f * 2^e.
double HalfUlpOf(double v) {
    return QuantumOf(v, 10);
}

/// The reference lane's own bound, carried by every composed comparison: the
/// certified double lane at m = 1 stands in for the grid wherever a format
/// conversion sits between the grid and the entry, so the composed bound is the
/// entry's figure plus this one.
constexpr double kOracleBound = 5.5e-14;

/// A value no Boys entry returns: F_n(x) is positive and at most 1 for every
/// supported order and argument, so a negative marker says "not written".
constexpr double kUnwritten = -1.0;

/// The multipliers these checks name. The library pre-instantiates m = 1 and
/// the sampled rungs; 3, 8 and 100 are inside the documented 1..65536 and are
/// not among them, which is the point: an entry whose definition lives in a
/// .cpp file is a link error here.
const char* MultiplierName(double m) {
    if (m == 1.0)
    {
        return "1, the default";
    }

    if (m == 3.0)
    {
        return "3";
    }

    if (m == 8.0)
    {
        return "8";
    }

    return "100";
}

// --- the entries, each named at the multiplier the rules use -----------------

double Single(double m, int n, double x) {
    if (m == 1.0)
    {
        return boys::BoysSingle<1.0>(n, x);
    }

    if (m == 3.0)
    {
        return boys::BoysSingle<3.0>(n, x);
    }

    if (m == 8.0)
    {
        return boys::BoysSingle<8.0>(n, x);
    }

    return boys::BoysSingle<100.0>(n, x);
}

void AllOrders(double m, int nmax, double x, double* out) {
    if (m == 1.0)
    {
        boys::BoysAllOrders<1.0>(nmax, x, out);
    } else if (m == 3.0)
    {
        boys::BoysAllOrders<3.0>(nmax, x, out);
    } else if (m == 8.0)
    {
        boys::BoysAllOrders<8.0>(nmax, x, out);
    } else
    {
        boys::BoysAllOrders<100.0>(nmax, x, out);
    }
}

void FixedN(double m, int n, const double* x, double* out, std::size_t count, std::size_t stride) {
    if (m == 1.0)
    {
        boys::BoysFixedN<1.0>(n, x, out, count, stride);
    } else if (m == 3.0)
    {
        boys::BoysFixedN<3.0>(n, x, out, count, stride);
    } else if (m == 8.0)
    {
        boys::BoysFixedN<8.0>(n, x, out, count, stride);
    } else
    {
        boys::BoysFixedN<100.0>(n, x, out, count, stride);
    }
}

void AllN(double m, int nmax, const double* x, double* out, std::size_t count, std::size_t* ws) {
    if (m == 1.0)
    {
        boys::BoysAllN<1.0>(nmax, x, out, count, ws);
    } else if (m == 3.0)
    {
        boys::BoysAllN<3.0>(nmax, x, out, count, ws);
    } else if (m == 8.0)
    {
        boys::BoysAllN<8.0>(nmax, x, out, count, ws);
    } else
    {
        boys::BoysAllN<100.0>(nmax, x, out, count, ws);
    }
}

void AllNSorted(double m, int nmax, const double* x, double* out, std::size_t count) {
    if (m == 1.0)
    {
        boys::BoysAllN<1.0>(nmax, x, out, count, boys::BoysSortedArgs{});
    } else if (m == 3.0)
    {
        boys::BoysAllN<3.0>(nmax, x, out, count, boys::BoysSortedArgs{});
    } else if (m == 8.0)
    {
        boys::BoysAllN<8.0>(nmax, x, out, count, boys::BoysSortedArgs{});
    } else
    {
        boys::BoysAllN<100.0>(nmax, x, out, count, boys::BoysSortedArgs{});
    }
}

void AllNAtOrders(double m, const int* n, const double* x, double* out, std::size_t count) {
    if (m == 1.0)
    {
        boys::BoysAllNAtOrders<1.0>(n, x, out, count);
    } else if (m == 3.0)
    {
        boys::BoysAllNAtOrders<3.0>(n, x, out, count);
    } else if (m == 8.0)
    {
        boys::BoysAllNAtOrders<8.0>(n, x, out, count);
    } else
    {
        boys::BoysAllNAtOrders<100.0>(n, x, out, count);
    }
}

void AllNF32(double m, int nmax, const float* x, float* out, std::size_t count) {
    if (m == 1.0)
    {
        boys::BoysAllNF32<1.0>(nmax, x, out, count);
    } else if (m == 3.0)
    {
        boys::BoysAllNF32<3.0>(nmax, x, out, count);
    } else if (m == 8.0)
    {
        boys::BoysAllNF32<8.0>(nmax, x, out, count);
    } else
    {
        boys::BoysAllNF32<100.0>(nmax, x, out, count);
    }
}

float SingleF32(double m, int n, float x) {
    if (m == 1.0)
    {
        return boys::BoysSingleF32<1.0>(n, x);
    }

    if (m == 3.0)
    {
        return boys::BoysSingleF32<3.0>(n, x);
    }

    if (m == 8.0)
    {
        return boys::BoysSingleF32<8.0>(n, x);
    }

    return boys::BoysSingleF32<100.0>(n, x);
}

void AllOrdersF32(double m, int nmax, float x, float* out) {
    if (m == 1.0)
    {
        boys::BoysAllOrdersF32<1.0>(nmax, x, out);
    } else if (m == 3.0)
    {
        boys::BoysAllOrdersF32<3.0>(nmax, x, out);
    } else if (m == 8.0)
    {
        boys::BoysAllOrdersF32<8.0>(nmax, x, out);
    } else
    {
        boys::BoysAllOrdersF32<100.0>(nmax, x, out);
    }
}

boys::F16 SingleF16(double m, int n, boys::F16 x) {
    if (m == 1.0)
    {
        return boys::BoysSingleF16<1.0>(n, x);
    }

    if (m == 3.0)
    {
        return boys::BoysSingleF16<3.0>(n, x);
    }

    if (m == 8.0)
    {
        return boys::BoysSingleF16<8.0>(n, x);
    }

    return boys::BoysSingleF16<100.0>(n, x);
}

void AllOrdersF16(double m, int nmax, boys::F16 x, boys::F16* out) {
    if (m == 1.0)
    {
        boys::BoysAllOrdersF16<1.0>(nmax, x, out);
    } else if (m == 3.0)
    {
        boys::BoysAllOrdersF16<3.0>(nmax, x, out);
    } else if (m == 8.0)
    {
        boys::BoysAllOrdersF16<8.0>(nmax, x, out);
    } else
    {
        boys::BoysAllOrdersF16<100.0>(nmax, x, out);
    }
}

boys::Bf16 SingleBf16(double m, int n, boys::Bf16 x) {
    if (m == 1.0)
    {
        return boys::BoysSingleBf16<1.0>(n, x);
    }

    if (m == 3.0)
    {
        return boys::BoysSingleBf16<3.0>(n, x);
    }

    if (m == 8.0)
    {
        return boys::BoysSingleBf16<8.0>(n, x);
    }

    return boys::BoysSingleBf16<100.0>(n, x);
}

void AllOrdersBf16(double m, int nmax, boys::Bf16 x, boys::Bf16* out) {
    if (m == 1.0)
    {
        boys::BoysAllOrdersBf16<1.0>(nmax, x, out);
    } else if (m == 3.0)
    {
        boys::BoysAllOrdersBf16<3.0>(nmax, x, out);
    } else if (m == 8.0)
    {
        boys::BoysAllOrdersBf16<8.0>(nmax, x, out);
    } else
    {
        boys::BoysAllOrdersBf16<100.0>(nmax, x, out);
    }
}

void RegionAProduct(boys::ProductMode mode,
                    double m,
                    boys::RegionABand band,
                    int nmax,
                    const double* x,
                    double* out,
                    std::size_t count) {
    if (mode == boys::ProductMode::kFp64 && m == 1.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kFp64, 1.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kFp64 && m == 3.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kFp64, 3.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kFp64)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kFp64, 8.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kTf32x3 && m == 8.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kTf32x3, 8.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kTf32x3)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kTf32x3, 100.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kBf16x6 && m == 8.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kBf16x6, 8.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kBf16x6)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kBf16x6, 100.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kTf32 && m == 8.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kTf32, 8.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kTf32)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kTf32, 100.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kBf16 && m == 8.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kBf16, 8.0>(band, nmax, x, out, count);
    } else if (mode == boys::ProductMode::kBf16)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kBf16, 100.0>(band, nmax, x, out, count);
    } else if (m == 8.0)
    {
        boys::BoysRegionAProduct<boys::ProductMode::kFp16, 8.0>(band, nmax, x, out, count);
    } else
    {
        boys::BoysRegionAProduct<boys::ProductMode::kFp16, 100.0>(band, nmax, x, out, count);
    }
}

// --- the checks -------------------------------------------------------------

void CheckConstants(Report& report) {
    Require(report, boys::kMaxBoysOrder == 32, "kMaxBoysOrder is the documented 32");
    Require(report,
            boys::kBoysFullAccuracyMultiplier == 1.0,
            "kBoysFullAccuracyMultiplier is the documented 1.0");
    Require(report,
            boys::kRegionA1Edge == 5.94992407605424223,
            "kRegionA1Edge is the documented join of the two bands");
    Require(report,
            boys::kRegionAEnd == 11.899848152108484,
            "kRegionAEnd is the documented end of region A");
    Require(report,
            boys::kHalfNativeScaleExponent == 15,
            "kHalfNativeScaleExponent is the documented 2^15 scale");

    // The version a caller reads is the version the build was configured at.
    // The build carries that value (BoysExpectedVersion, from the project()
    // call), so a release that bumps one and not the other fails here rather
    // than shipping two answers to "which version is this?".
    // Parsed by hand rather than with sscanf: MSVC deprecates sscanf and this
    // tree builds with warnings as errors, so a portable parser is cheaper than
    // a suppression. The format is the project() call's, three dot-separated
    // decimal components and nothing else.
    int major = 0;
    int minor = 0;
    int patch = 0;
    {
        int* parts[] = {&major, &minor, &patch};
        const char* p = BoysExpectedVersion;

        for (int i = 0; i < 3 && *p != '\0'; ++i)
        {
            while (*p == '.')
            {
                ++p;
            }

            while (*p >= '0' && *p <= '9')
            {
                *parts[i] = *parts[i] * 10 + (*p - '0');
                ++p;
            }
        }
    }
    Require(report,
            boys::kVersionMajor == major && boys::kVersionMinor == minor &&
                boys::kVersionPatch == patch,
            "the version constants are the version the build was configured at");
    char spelled[64];
    std::snprintf(spelled, sizeof(spelled), "%d.%d.%d", major, minor, patch);
    Require(report,
            std::string(boys::VersionString()) == spelled,
            "VersionString() spells the same version as the constants");
    Covered("boys::VersionString");
    Covered("boys::kVersionMajor");
    Covered("boys::kMaxBoysOrder");
    Covered("boys::kBoysFullAccuracyMultiplier");
    Covered("boys::RegionABand");
    Covered("boys::kRegionA1Edge");
    Covered("boys::kRegionAEnd");
    Covered("boys::kHalfNativeScaleExponent");
    Covered("boys::BoysAllNWorkspaceSize");

    // The capability predicate is documented as a pure function of the
    // processor, and as reporting false on every non-x86_64 target.
    const bool first = boys::BoysAvx2Available();
    Require(
        report, boys::BoysAvx2Available() == first, "BoysAvx2Available() is stable across calls");
#if !defined(__x86_64__) && !defined(_M_X64)
    Require(report, !first, "BoysAvx2Available() reports false on a non-x86_64 target");
#endif
    Covered("boys::BoysAvx2Available");

    // The library publishes whether this build contracts a bare
    // product-plus-add, measured once when the build was configured, and the
    // lane claims below are compiled against that answer. It is a statement
    // about one translation unit's arithmetic, so it is checked here in the
    // unit that makes those claims: a build whose flags reached this file but
    // not the measurement would otherwise assert the other arithmetic's claim
    // and be green for it.
#if defined(BOYS_SCALAR_CONTRACTS)
    Require(report,
            boys::backend::ScalarFp64::Contracts() == (BOYS_SCALAR_CONTRACTS != 0),
            "the configure-time contraction measurement is this translation unit's");
#endif
}

struct TierSpec {
    boys::AccuracyTier tier;
    double multiplier;
    const char* name;
};

const TierSpec kTiers[] = {
    {boys::AccuracyTier::kReference, 1.0, "kReference"},
    {boys::AccuracyTier::kRelaxed64, 64.0, "kRelaxed64"},
    {boys::AccuracyTier::kRelaxed256, 256.0, "kRelaxed256"},
    {boys::AccuracyTier::kRelaxed1024, 1024.0, "kRelaxed1024"},
    {boys::AccuracyTier::kRelaxed4096, 4096.0, "kRelaxed4096"},
    {boys::AccuracyTier::kRelaxed16384, 16384.0, "kRelaxed16384"},
    {boys::AccuracyTier::kRelaxed65536, 65536.0, "kRelaxed65536"},
};

constexpr std::size_t kTierCount = sizeof(kTiers) / sizeof(kTiers[0]);

void CheckTiers(Report& report) {
    for (const TierSpec& spec : kTiers)
    {
        Require(report,
                boys::AccuracyMultiplier(spec.tier) == spec.multiplier,
                "AccuracyMultiplier names the tier's documented m");
    }
    Covered("boys::AccuracyTier");
    Covered("boys::AccuracyMultiplier");

    // Documented: a value outside the enumerators is not a tier, and every
    // entry treats it as the reference tier rather than guessing a rung.
    const boys::AccuracyTier outside = static_cast<boys::AccuracyTier>(99);
    Require(report,
            boys::AccuracyMultiplier(outside) == boys::kBoysFullAccuracyMultiplier,
            "an out-of-range tier names the reference multiplier");

    const boys::AccuracyRegion kRegions[] = {
        boys::AccuracyRegion::kA, boys::AccuracyRegion::kB, boys::AccuracyRegion::kC};
    const double kTolerances[] = {0.0, 1e-17, 1e-15, 1e-8, 1e-6};

    // Region C's reachable error is the reference tier's at every tier, and the
    // reachable error is monotone in m; both are stated over the whole tier
    // list, so the figures the comparisons need are carried across the tiers
    // rather than recomputed inside one.
    double referenceReachable = 0.0;
    double previousReachable[kTierCount] = {};
    const std::size_t kRegionIndex[] = {0, 1, 2};

    for (std::size_t i = 0; i < kTierCount; ++i)
    {
        for (std::size_t r = 0; r < 3; ++r)
        {
            const boys::AccuracyRegion region = kRegions[r];

            for (const double tolerance : kTolerances)
            {
                const boys::TierCoverage coverage =
                    boys::QueryTier(kTiers[i].tier, region, tolerance);

                // Documented: meets reports whether the tier delivers an error
                // at or below the request, and limiting names the component
                // that stops it when a tighter error is asked for.
                Require(report,
                        coverage.meets == (tolerance >= coverage.reachable),
                        "TierCoverage.meets reports whether the request is reachable");
                Require(report,
                        coverage.limiting == boys::AccuracyComponent::kRegionASeed ||
                            coverage.limiting == boys::AccuracyComponent::kRegionBFit ||
                            coverage.limiting == boys::AccuracyComponent::kRegionCAsymptotic,
                        "TierCoverage.limiting names one of the three components");
                Require(
                    report, coverage.reachable >= 0.0, "TierCoverage.reachable is not negative");

                // Documented: region C has no relaxable resource, so its
                // reachable error is the reference tier's at every m.
                if (region == boys::AccuracyRegion::kC)
                {
                    if (kTiers[i].tier == boys::AccuracyTier::kReference)
                    {
                        referenceReachable = coverage.reachable;
                    } else
                    {
                        Require(report,
                                coverage.reachable == referenceReachable,
                                "region C reaches the reference tier's error at every tier");
                    }
                }
            }

            // Documented: the contract is monotone in m, so a coarser tier
            // never reaches a tighter error than the tier below it.
            if (region != boys::AccuracyRegion::kC)
            {
                const double reachable = boys::QueryTier(kTiers[i].tier, region, 1e-12).reachable;

                if (i > 0)
                {
                    Require(report,
                            reachable >= previousReachable[kRegionIndex[r]],
                            "the reachable error is monotone in the tier");
                }

                previousReachable[kRegionIndex[r]] = reachable;
            }
        }
    }
    Covered("boys::QueryTier (tier, region, tolerance)");
    Covered("boys::TierCoverage");
    Covered("boys::AccuracyRegion");
    Covered("boys::AccuracyComponent");

    // Documented: the second overload takes the region from x rather than from
    // the caller, so it must answer what the named-region form answers for the
    // region that argument falls in. The four arguments sit inside the three
    // documented intervals rather than near a boundary.
    struct ArgSpec {
        double x;
        boys::AccuracyRegion region;
    };

    const ArgSpec kArgs[] = {{0.5, boys::AccuracyRegion::kA},
                             {5.0, boys::AccuracyRegion::kA},
                             {15.0, boys::AccuracyRegion::kB},
                             {40.0, boys::AccuracyRegion::kC}};

    for (const TierSpec& spec : kTiers)
    {
        for (const ArgSpec& arg : kArgs)
        {
            for (const double tolerance : kTolerances)
            {
                const boys::TierCoverage byArg = boys::QueryTier(spec.tier, arg.x, tolerance);
                const boys::TierCoverage byRegion =
                    boys::QueryTier(spec.tier, arg.region, tolerance);

                Require(report,
                        byArg.meets == byRegion.meets && byArg.reachable == byRegion.reachable &&
                            byArg.limiting == byRegion.limiting,
                        "QueryTier(x) answers what QueryTier(region) answers");
            }
        }
    }
    Covered("boys::QueryTier (tier, x, tolerance)");

    // Documented: the run-time tier entry writes nmax + 1 values at every tier,
    // including a tier this build does not serve, and a value outside the
    // enumerators evaluates as the reference tier - bit for bit, since the
    // entry is a pure function of its arguments and both calls select the same
    // rung.
    double out[boys::kMaxBoysOrder + 1];
    double relaxed[boys::kMaxBoysOrder + 1];

    std::fill(std::begin(out), std::end(out), kUnwritten);
    boys::BoysAllOrdersAtTier(outside, 7, 3.0, out);

    for (int n = 0; n <= 7; ++n)
    {
        Require(report, out[n] != kUnwritten, "the tier entry writes every value below nmax");
    }

    Require(report, out[8] == kUnwritten, "the tier entry writes nmax + 1 values and no more");

    boys::BoysAllOrdersAtTier(boys::AccuracyTier::kReference, boys::kMaxBoysOrder, 3.0, out);
    boys::BoysAllOrdersAtTier(outside, boys::kMaxBoysOrder, 3.0, relaxed);

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        Require(report, out[n] != kUnwritten, "the reference tier writes every order");
        Require(report,
                out[n] == relaxed[n],
                "an out-of-range tier evaluates as the reference tier, bit for bit");
    }
    Covered("boys::BoysAllOrdersAtTier");

    // Documented: the tier and the route are two selectors of two different
    // things, so the entry that names both answers the rung of the route it was
    // given rather than the default route's rung. Judged against the boundary
    // the route's own row reports: over an interval the rational route serves,
    // naming it has to change the values, and the scheme-carrying and
    // reference-scheme overloads have to agree with each other and with a
    // direct call at the rung's own multiplier.
    {
        const auto& rows = boys::BoysFitRoutes();
        double rationalFrom = 0.0;
        double rationalHi = 0.0;

        for (const boys::FitRouteInfo& row : rows)
        {
            if (row.route == boys::FitRoute::kRationalMinimax &&
                row.region == boys::AccuracyRegion::kA)
            {
                rationalFrom = row.servesFrom;
                rationalHi = row.hi;
            }
        }

        const double x = 0.5 * (rationalFrom + rationalHi);

        boys::BoysAllOrdersAtTier(boys::AccuracyTier::kRelaxed64,
                                  boys::FitRoute::kRationalMinimax,
                                  boys::kMaxBoysOrder,
                                  x,
                                  relaxed);
        boys::BoysAllOrdersAtTier(boys::AccuracyTier::kRelaxed64, boys::kMaxBoysOrder, x, out);

        std::size_t differs = 0;

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            Require(report,
                    relaxed[n] != kUnwritten,
                    "the route-carrying tier entry writes every order");
            differs += relaxed[n] != out[n];
        }

        Require(report,
                differs > 0,
                "the route-carrying tier entry answers the route it was given, not the default");

        boys::BoysAllOrdersAtTier(boys::AccuracyTier::kRelaxed64,
                                  boys::FitRoute::kRationalMinimax,
                                  boys::EvalScheme::kSplitClenshaw,
                                  boys::kMaxBoysOrder,
                                  x,
                                  out);

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            Require(report,
                    relaxed[n] == out[n],
                    "the two- and three-selector tier overloads agree, bit for bit");
        }

        // The reference rung of the route is the uncut route's own entry: the
        // tier that names no rung and the entry that names no rung are the same
        // call, and the route table's figure is that call's.
        boys::BoysAllOrdersAtTier(boys::AccuracyTier::kReference,
                                  boys::FitRoute::kRationalMinimax,
                                  boys::kMaxBoysOrder,
                                  x,
                                  relaxed);
        boys::BoysAllOrdersWithRoute(
            boys::FitRoute::kRationalMinimax, boys::kMaxBoysOrder, x, out);

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            Require(report,
                    relaxed[n] == out[n],
                    "the reference rung of the rational route is the uncut route, bit for bit");
        }
    }

    Covered("boys::BoysAllOrdersAtTier (tier, route, scheme)");
    Covered("boys::BoysAllOrdersAtTier (tier, route)");
}

/// BoysSingle, BoysAllOrders and BoysFixedN over the whole grid, at the default
/// multiplier, at one the library does not pre-instantiate, and at one past the
/// largest rung the tier list samples.
void CheckDoubleLanes(Report& report, const std::vector<Cell>& cells) {
    for (const double m : {1.0, 8.0, 100.0})
    {
        Rule& single = NewRule(RuleName("BoysSingle<m = %> (grid sweep)", MultiplierName(m)));
        Rule& batch = NewRule(RuleName("BoysAllOrders<m = %> (grid sweep)", MultiplierName(m)));
        Rule& fixed = NewRule(RuleName("BoysFixedN<m = %> (strided)", MultiplierName(m)));

        for (const Cell& cell : cells)
        {
            Judge(single,
                  Single(m, cell.n, cell.x),
                  cell.value,
                  SingleBound(cell.x, m),
                  cell.n,
                  cell.x);
        }

        for (const double x : DistinctArgs(cells))
        {
            double all[boys::kMaxBoysOrder + 1] = {};
            AllOrders(m, boys::kMaxBoysOrder, x, all);

            for (const Cell& cell : cells)
            {
                if (cell.x == x)
                {
                    Judge(batch, all[cell.n], cell.value, BatchBound(m), cell.n, cell.x);
                }
            }

            // Documented: each output element of the fixed-order entry carries
            // the single lane's per-region bound at the same multiplier, order
            // and argument, and is that entry's value bit for bit.
            //
            // Same recurrence, same source - which is why the values agree - but
            // the same source is not the same bits: whether a bare
            // product-plus-add in it is one rounding or two is a licence the
            // compiler holds per call site. The equality is asserted here on
            // every build, because it has held at every cell this check has swept
            // on every host measured; the bound is asserted beside it so a host
            // where the equality stops holding is still judged on what the lane
            // promises there.
            const std::size_t comparedBefore = report.exactCompared;
            const std::size_t heldBefore = report.exactHeld;
            std::size_t outsideBound = 0;

            for (const Cell& cell : cells)
            {
                if (cell.x != x)
                {
                    continue;
                }

                double plain = kUnwritten;
                FixedN(m, cell.n, &x, &plain, 1, 1);
                Judge(fixed, plain, cell.value, SingleBound(cell.x, m), cell.n, cell.x);

                const double singleValue = Single(m, cell.n, x);

                if (std::abs(plain - singleValue) > SingleBound(cell.x, m))
                {
                    ++outsideBound;
                }

                Compare(report, plain == singleValue);
            }

            Require(report,
                    outsideBound == 0,
                    "BoysFixedN returns what BoysSingle returns inside the single lane's bound");
            Require(report,
                    report.exactHeld - heldBefore == report.exactCompared - comparedBefore,
                    "BoysFixedN returns what BoysSingle returns, bit for bit");

            // Documented: out[i * stride] = F_n(x[i]), so the same value lands
            // at offset 0 of a stride-1 call and offset 0 and 3 of a stride-3
            // one, and the slots between them are the caller's, untouched. The
            // value itself is the single lane's by the paragraph above, so the
            // two claims below are split the same way: the bound on every build,
            // the exact offset on the builds whose bare product-plus-add is two
            // roundings. This shape is the one place the difference has been
            // measured: a call with two arguments and a gap is not the same
            // generated code as a call with one, and on a contracting build the
            // compiler fused the recurrence at one of the two and not at the
            // other, which moves the value by one unit in the last place.
            const double two[2] = {x, x * 0.5 + 0.25};
            const double one = Single(m, 3, x);
            const double other = Single(m, 3, two[1]);
            double strided[7];
            std::fill(std::begin(strided), std::end(strided), kUnwritten);
            FixedN(m, 3, two, strided, 2, 3);
            Require(report,
                    std::abs(strided[0] - one) <= SingleBound(x, m),
                    "a stride of 3 writes the first value inside the single lane's bound");
            Require(report,
                    strided[1] == kUnwritten && strided[2] == kUnwritten,
                    "a stride of 3 leaves the padding between values untouched");
            Require(report,
                    std::abs(strided[3] - other) <= SingleBound(two[1], m),
                    "a stride of 3 writes the second value inside the single lane's bound");
            Compare(report, strided[0] == one);
            Compare(report, strided[3] == other);
#if defined(BOYS_SCALAR_CONTRACTS) && BOYS_SCALAR_CONTRACTS == 0
            Require(report, strided[0] == one, "a stride of 3 writes the first value at offset 0");
            Require(
                report, strided[3] == other, "a stride of 3 writes the second value at offset 3");
#endif

            // Documented: count may be 0, and then nothing is written.
            double untouched[2] = {kUnwritten, kUnwritten};
            FixedN(m, 3, two, untouched, 0, 1);
            Require(report,
                    untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                    "BoysFixedN writes nothing for count = 0");
        }

        Covered("boys::BoysSingle<m>");
        Covered("boys::BoysAllOrders<m>");
        Covered("boys::BoysFixedN<m>");
    }
}

/// The run-time fit routes. A consumer reaches the choice through the public
/// report alone: which routes exist, the interval each fit covers, the argument
/// each selector takes over at and the bar each is certified against all come
/// from BoysFitRoutes(), so every row here is judged against the figures that
/// row states rather than against numbers this file carries.
void CheckFitRoutes(Report& report, const std::vector<Cell>& cells) {
    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();
    const std::vector<double> args = DistinctArgs(cells);

    Require(report, !routes.empty(), "BoysFitRoutes reports the routes this build carries");

    bool chebyshev = false;
    bool rational = false;
    bool regionA = false;
    bool regionB = false;

    for (const boys::FitRouteInfo& row : routes)
    {
        Require(report, row.name != nullptr && row.name[0] != '\0', "a route row names its route");
        Require(report, row.lo < row.hi, "a route row states a non-empty interval");
        Require(report,
                row.servesFrom >= row.lo && row.servesFrom < row.hi,
                "a route row's served domain is inside the interval its fit covers");
        Require(report, row.stored > 0, "a route row states the coefficients its fit stores");
        Require(report,
                row.delivered <= row.bound,
                "a route row's bar covers the error it reports delivering");

        chebyshev = chebyshev || row.route == boys::FitRoute::kChebyshev;
        rational = rational || row.route == boys::FitRoute::kRationalMinimax;
        regionA = regionA || row.region == boys::AccuracyRegion::kA;
        regionB = regionB || row.region == boys::AccuracyRegion::kB;

        const std::string name =
            std::string("BoysAllOrdersWithRoute(") + row.name +
            (row.region == boys::AccuracyRegion::kA ? ", region A)" : ", region B)");
        Rule& rule = NewRule(name);
        std::size_t changed = 0;

        for (const double x : args)
        {
            // Inside the domain the row serves, and nowhere else: a row that
            // hands its fit over per order states the argument from which it
            // does, so this is the domain the row's own figures are a promise
            // over.
            if (x < row.servesFrom || x >= row.hi)
            {
                continue;
            }

            double plain[boys::kMaxBoysOrder + 1] = {};
            double out[boys::kMaxBoysOrder + 1] = {};
            AllOrders(1.0, boys::kMaxBoysOrder, x, plain);
            boys::BoysAllOrdersWithRoute(row.route, boys::kMaxBoysOrder, x, out);

            for (const Cell& cell : cells)
            {
                if (cell.x != x)
                {
                    continue;
                }

                Judge(rule, out[cell.n], cell.value, row.bound, cell.n, cell.x);

                if (out[cell.n] != plain[cell.n])
                {
                    ++changed;
                }
            }
        }

        // A route a caller cannot reach a difference through is not an option,
        // whatever its row reports. The default route is the other half of the
        // same statement: naming it is the default entry, so it changes nothing
        // anywhere, and a consumer reads that as the route being the shipped
        // fits rather than as a route that does nothing.
        char message[160];

        if (row.route == boys::FitRoute::kRationalMinimax)
        {
            std::snprintf(message,
                          sizeof(message),
                          "naming the rational route changes values inside %s's domain",
                          name.c_str());
            Require(report, changed > 0, message);
        } else
        {
            std::snprintf(message,
                          sizeof(message),
                          "naming the default route over %s is the default entry, bit for bit",
                          name.c_str());
            Require(report, changed == 0, message);
        }
    }

    Require(report,
            chebyshev && rational,
            "the report carries both the default and the rational route");
    Require(report, regionA && regionB, "the report covers region A and region B");

    // Documented: outside the intervals its rows report, naming a route runs the
    // default entry's own code, so a caller who names one and a caller who does
    // not receive the same numbers bit for bit.
    std::size_t outside = 0;
    std::size_t changedOutside = 0;

    for (const double x : args)
    {
        bool served = false;

        for (const boys::FitRouteInfo& row : routes)
        {
            served = served || (x >= row.servesFrom && x < row.hi);
        }

        if (served)
        {
            continue;
        }

        ++outside;
        double plain[boys::kMaxBoysOrder + 1] = {};
        double out[boys::kMaxBoysOrder + 1] = {};
        AllOrders(1.0, boys::kMaxBoysOrder, x, plain);
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kRationalMinimax, boys::kMaxBoysOrder, x, out);

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            if (out[n] != plain[n])
            {
                ++changedOutside;
            }
        }
    }

    Require(report, outside > 0, "the grid reaches arguments outside every route's served domain");
    Require(report,
            changedOutside == 0,
            "outside a route's served domain the entry is the default entry, bit for bit");

    // Documented: a value outside the enumerators is not a route, and every
    // entry on this surface treats it as the default rather than guessing.
    std::size_t changedUnknown = 0;

    for (const double x : args)
    {
        double plain[boys::kMaxBoysOrder + 1] = {};
        double out[boys::kMaxBoysOrder + 1] = {};
        AllOrders(1.0, boys::kMaxBoysOrder, x, plain);
        boys::BoysAllOrdersWithRoute(static_cast<boys::FitRoute>(99), boys::kMaxBoysOrder, x, out);

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            if (out[n] != plain[n])
            {
                ++changedUnknown;
            }
        }
    }

    Require(report,
            changedUnknown == 0,
            "a route value outside the enumeration evaluates at the default, bit for bit");

    // The two axes compose at run time as they do at compile time: the entry
    // names both, and a caller holds one fixed to see the other move. If one
    // axis did not reach the call, one of the three counts below would be zero -
    // which is what makes this a check of the pair rather than of two options
    // printed beside each other.
    std::size_t routeAtClenshaw = 0;
    std::size_t routeAtHorner = 0;
    std::size_t schemeOnRational = 0;

    for (const double x : args)
    {
        std::array<double, 33> rationalClenshaw = {};
        std::array<double, 33> rationalHorner = {};
        std::array<double, 33> chebyshevClenshaw = {};
        std::array<double, 33> chebyshevHorner = {};
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kRationalMinimax,
                                     boys::EvalScheme::kSplitClenshaw,
                                     boys::kMaxBoysOrder,
                                     x,
                                     rationalClenshaw.data());
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kRationalMinimax,
                                     boys::EvalScheme::kHorner,
                                     boys::kMaxBoysOrder,
                                     x,
                                     rationalHorner.data());
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kChebyshev,
                                     boys::EvalScheme::kSplitClenshaw,
                                     boys::kMaxBoysOrder,
                                     x,
                                     chebyshevClenshaw.data());
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kChebyshev,
                                     boys::EvalScheme::kHorner,
                                     boys::kMaxBoysOrder,
                                     x,
                                     chebyshevHorner.data());

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const std::size_t j = static_cast<std::size_t>(n);

            if (rationalClenshaw[j] != chebyshevClenshaw[j])
            {
                ++routeAtClenshaw;
            }

            if (rationalHorner[j] != chebyshevHorner[j])
            {
                ++routeAtHorner;
            }

            if (rationalClenshaw[j] != rationalHorner[j])
            {
                ++schemeOnRational;
            }
        }
    }

    Require(report,
            routeAtClenshaw > 0,
            "naming the rational route changes values with the scheme held at the default");
    Require(report,
            routeAtHorner > 0,
            "naming the rational route changes values with the Horner scheme held");
    Require(report,
            schemeOnRational > 0,
            "naming a scheme on the rational route reaches the parts its own fits do not serve");

    Covered("boys::FitRoute");
    Covered("boys::FitRouteInfo");
    Covered("boys::BoysFitRoutes");
    Covered("boys::BoysAllOrdersWithRoute");
}

/// The float lane's fit routes, reached the same way: every figure this file
/// judges a row by comes from BoysFitRoutesF32's own row rather than from a
/// number carried here, so the report and the entry cannot agree with each
/// other and both be wrong.
void CheckFitRoutesF32(Report& report, const std::vector<Cell>& cells) {
    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutesF32();
    const std::vector<double> args = DistinctArgs(cells);

    Require(report, !routes.empty(), "BoysFitRoutesF32 reports the routes this build carries");

    bool chebyshev = false;
    bool rational = false;

    for (const boys::FitRouteInfo& row : routes)
    {
        Require(report, row.name != nullptr && row.name[0] != '\0', "a float route row names its route");
        Require(report, row.lo < row.hi, "a float route row states a non-empty interval");
        Require(report,
                row.servesFrom >= row.lo && row.servesFrom < row.hi,
                "a float route row's served domain is inside the interval its fit covers");
        Require(report, row.stored > 0, "a float route row states the coefficients its fit stores");
        Require(report,
                row.delivered <= row.bound,
                "a float route row's bar covers the error it reports delivering");

        chebyshev = chebyshev || row.route == boys::FitRoute::kChebyshev;
        rational = rational || row.route == boys::FitRoute::kRationalMinimax;

        const std::string name =
            std::string("BoysSingleF32WithRoute(") + row.name +
            (row.region == boys::AccuracyRegion::kA ? ", region A)" : ", region B)");
        NewRule(name);
        std::size_t changed = 0;
        std::size_t unknownDiff = 0;

        for (const double x : args)
        {
            if (x < row.servesFrom || x >= row.hi)
            {
                continue;
            }

            const float xf = static_cast<float>(x);

            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                const float plain = boys::BoysSingleF32(n, xf);
                const float selected = boys::BoysSingleF32WithRoute(row.route, n, xf);
                const float unknown =
                    boys::BoysSingleF32WithRoute(static_cast<boys::FitRoute>(99), n, xf);

                Require(report,
                        std::isfinite(selected),
                        "a float route row's entry returns a finite value over the domain it "
                        "serves");

                if (selected != plain)
                {
                    ++changed;
                }

                if (unknown != plain)
                {
                    ++unknownDiff;
                }
            }
        }

        // The default route is the default entry by construction, so only a
        // row that names a different fit is required to change a value.
        Require(report,
                row.route == boys::FitRoute::kChebyshev || changed > 0,
                "naming a float route other than the default changes values inside the domain "
                "its own row serves");
        Require(report,
                unknownDiff == 0,
                "an unnamed route value is the default entry, bit for bit, on the float lane");
    }

    Require(report, chebyshev, "BoysFitRoutesF32 carries the default route");
    Require(report, rational, "BoysFitRoutesF32 carries the rational route");

    Covered("boys::BoysSingleF32WithRoute");
    Covered("boys::BoysFitRoutesF32");
}

/// The float lane's policy path, which a consumer reaches as a template argument
/// on the entries themselves rather than through the run-time selector. Three
/// readings make that path an option rather than a name: a policy naming the
/// shipped pair is the entry naming no policy, bit for bit; naming the other
/// scheme changes values the shipped scheme answers with; and each route's policy
/// answers exactly what that route's run-time selector answers, which is one body
/// reached two ways rather than two wirings that happen to agree.
void CheckFloatPolicies(Report& report, const std::vector<Cell>& cells) {
    using Shipped = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw>;
    using Horner = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;
    using Rational = boys::EvalPolicy<boys::FitRoute::kRationalMinimax>;

    static_assert(Shipped{}.kRoute == boys::kDefaultFitRoute &&
                      Shipped{}.kScheme == boys::kDefaultEvalScheme,
                  "the pair this check calls the shipped one is the library's own default pair");

    std::size_t sameAsDefault = 0;
    std::size_t sameAsSelector[2] = {0, 0};
    std::size_t changedByScheme = 0;
    std::size_t changedByRoute = 0;
    bool allFinite = true;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const float byDefault = boys::BoysSingleF32(cell.n, xf);
        const float shipped =
            boys::BoysSingleF32<boys::kBoysFullAccuracyMultiplier, Shipped>(cell.n, xf);
        const float horner =
            boys::BoysSingleF32<boys::kBoysFullAccuracyMultiplier, Horner>(cell.n, xf);
        const float rational =
            boys::BoysSingleF32<boys::kBoysFullAccuracyMultiplier, Rational>(cell.n, xf);
        const float bySelector[2] = {
            boys::BoysSingleF32WithRoute(boys::FitRoute::kChebyshev, cell.n, xf),
            boys::BoysSingleF32WithRoute(boys::FitRoute::kRationalMinimax, cell.n, xf)};

        sameAsDefault += (shipped == byDefault) ? 1 : 0;
        sameAsSelector[0] += (shipped == bySelector[0]) ? 1 : 0;
        sameAsSelector[1] += (rational == bySelector[1]) ? 1 : 0;
        changedByScheme += (horner != shipped) ? 1 : 0;
        changedByRoute += (rational != shipped) ? 1 : 0;
        allFinite = allFinite && std::isfinite(horner) && std::isfinite(rational);
    }

    Require(report,
            allFinite,
            "a float policy this build stores answers a finite value at every cell of the "
            "reference grid");
    Require(report,
            cells.size() > 0 && sameAsDefault == cells.size(),
            "naming the shipped route and scheme is the entry naming no policy, bit for bit, at "
            "every cell of the reference grid");
    Require(report,
            sameAsSelector[0] == cells.size() && sameAsSelector[1] == cells.size(),
            "each route's policy answers what that route's run-time selector answers, bit for "
            "bit, so the two ways in read one body");
    Require(report,
            changedByScheme > 0,
            "naming the Horner scheme on the float lane changes values the shipped scheme "
            "answers with");
    Require(report,
            changedByRoute > 0,
            "naming the rational route on the float lane changes values the shipped route "
            "answers with somewhere on the reference grid");

    // The same pair on the all-orders shape. Its seeds are not the single
    // entry's - region A's is the double lane's fit at the policy's route and
    // scheme, region B's is this lane's - so the two entries answer the same fit
    // by different roads and the reading here is reachability, not identity: the
    // pair reaches this entry too, and it is the shipped pair by default.
    std::size_t batchSameAsDefault = 0;
    std::size_t batchChangedByScheme = 0;
    std::size_t batchChangedByRoute = 0;
    std::size_t batchArgs = 0;
    bool batchFinite = true;

    for (const double x : DistinctArgs(cells))
    {
        const float xf = static_cast<float>(x);
        std::array<float, boys::kMaxBoysOrder + 1> plain = {};
        std::array<float, boys::kMaxBoysOrder + 1> named = {};
        std::array<float, boys::kMaxBoysOrder + 1> horner = {};
        std::array<float, boys::kMaxBoysOrder + 1> rational = {};

        boys::BoysAllOrdersF32(boys::kMaxBoysOrder, xf, plain.data());
        boys::BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, Shipped>(
            boys::kMaxBoysOrder, xf, named.data());
        boys::BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, Horner>(
            boys::kMaxBoysOrder, xf, horner.data());
        boys::BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, Rational>(
            boys::kMaxBoysOrder, xf, rational.data());

        ++batchArgs;
        batchSameAsDefault +=
            std::memcmp(plain.data(), named.data(), sizeof(plain)) == 0 ? 1 : 0;
        batchChangedByScheme +=
            std::memcmp(plain.data(), horner.data(), sizeof(plain)) == 0 ? 0 : 1;
        batchChangedByRoute +=
            std::memcmp(plain.data(), rational.data(), sizeof(plain)) == 0 ? 0 : 1;

        for (std::size_t k = 0; k < plain.size(); ++k)
        {
            batchFinite = batchFinite && std::isfinite(horner[k]) && std::isfinite(rational[k]);
        }
    }

    Require(report,
            batchFinite,
            "a policy this build stores answers a finite value at every order of the batch entry");
    Require(report,
            batchArgs > 0 && batchSameAsDefault == batchArgs,
            "naming the shipped pair on the batch entry is the call naming no policy, bit for "
            "bit, at every argument of the reference grid");
    Require(report,
            batchChangedByScheme > 0 && batchChangedByRoute > 0,
            "the batch entry reads the pair a caller names: another route or scheme changes the "
            "values it answers with");

    Covered("boys::BoysSingleF32");
    Covered("boys::BoysAllOrdersF32");
}

/// The many-argument entry in all four of its documented shapes: the
/// workspace-supplied call, the internally allocated one, the sorted-argument
/// overload, and arguments in the wrong order.
void CheckManyArgumentLanes(Report& report, const std::vector<Cell>& cells) {
    const std::vector<double> args = DistinctArgs(cells);
    const std::size_t count = args.size();
    const std::size_t plane = boys::kMaxBoysOrder + 1;
    std::vector<double> planes(count * plane);
    std::vector<double> again(count * plane);
    std::vector<double> reversed(count);
    std::vector<std::size_t> workspace(boys::BoysAllNWorkspaceSize(count));

    std::reverse_copy(args.begin(), args.end(), reversed.begin());

    for (const double m : {1.0, 8.0})
    {
        Rule& rule =
            NewRule(RuleName("BoysAllN<m = %> (grid sweep, four shapes)", MultiplierName(m)));

        std::fill(planes.begin(), planes.end(), kUnwritten);
        AllN(m, boys::kMaxBoysOrder, args.data(), planes.data(), count, nullptr);

        std::fill(again.begin(), again.end(), kUnwritten);
        AllN(m, boys::kMaxBoysOrder, args.data(), again.data(), count, workspace.data());
        Require(report,
                std::equal(planes.begin(), planes.end(), again.begin()),
                "the caller's workspace returns the same planes as the internal one");

        std::fill(again.begin(), again.end(), kUnwritten);
        AllNSorted(m, boys::kMaxBoysOrder, args.data(), again.data(), count);
        Require(report,
                std::equal(planes.begin(), planes.end(), again.begin()),
                "the sorted-argument overload returns the same planes as the sorting one");

        // Documented: arguments may arrive in any order; the entry classifies
        // and groups them and returns the results in the caller's order, so
        // plane k, position i holds F_k(reversed[i]).
        std::fill(again.begin(), again.end(), kUnwritten);
        AllN(m, boys::kMaxBoysOrder, reversed.data(), again.data(), count, nullptr);

        std::size_t missing = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                const Cell* sorted = Find(cells, n, args[i]);
                const Cell* shuffled = Find(cells, n, reversed[i]);
                const std::size_t index = static_cast<std::size_t>(n) * count + i;

                if (sorted == nullptr || shuffled == nullptr)
                {
                    ++missing;
                    continue;
                }

                Judge(rule, planes[index], sorted->value, BatchBound(m), n, args[i]);
                Judge(rule, again[index], shuffled->value, BatchBound(m), n, reversed[i]);
            }
        }

        Require(report, missing == 0, "the grid carries every cell this rule looks up");

        // Documented: count may be 0, and then nothing is written.
        double untouched[4] = {kUnwritten, kUnwritten, kUnwritten, kUnwritten};
        AllN(m, boys::kMaxBoysOrder, args.data(), untouched, 0, nullptr);
        AllNSorted(m, boys::kMaxBoysOrder, args.data(), untouched, 0);
        Require(report,
                untouched[0] == kUnwritten,
                "BoysAllN writes nothing for count = 0, in both shapes");
        Covered("boys::BoysAllN<m>");
        Covered("boys::BoysAllN<m> (BoysSortedArgs)");
        Covered("boys::BoysSortedArgs");
    }
}

/// The lanes whose per-element order is not the batch's: the double order-array
/// batch, where each argument carries its own top order, and the float all-N
/// batch, whose shape is the double one in single precision.
void CheckPerElementOrderLanes(Report& report, const std::vector<Cell>& cells) {
    const std::vector<double> args = DistinctArgs(cells);
    const std::size_t count = args.size();
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t plane = static_cast<std::size_t>(nmax) + 1;
    std::vector<int> raggeds(count);
    std::vector<int> commons(count, nmax);
    std::vector<double> planes(count * plane);
    std::vector<double> row(plane);
    std::vector<float> argsF(count);
    std::vector<float> planesF(count * plane);
    std::vector<float> rowF(plane);

    // A ragged top order per argument, so the entry runs where the planes above
    // an argument's own top order are the caller's to keep, and with arguments
    // at the highest order as well, so a ragged batch still carries full columns.
    for (std::size_t i = 0; i < count; ++i)
    {
        raggeds[i] = static_cast<int>(i % plane);
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        argsF[i] = static_cast<float>(args[i]);
    }

    for (const double m : {1.0, 8.0})
    {
        Rule& ragged = NewRule(
            RuleName("BoysAllNAtOrders<m = %> (ragged top order per argument)", MultiplierName(m)));
        Rule& common = NewRule(
            RuleName("BoysAllNAtOrders<m = %> (one top order for the batch)", MultiplierName(m)));

        std::fill(planes.begin(), planes.end(), kUnwritten);
        AllNAtOrders(m, raggeds.data(), args.data(), planes.data(), count);

        std::size_t missing = 0;
        std::size_t above = 0;
        std::size_t kept = 0;
        std::size_t differs = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            // Documented: the column stops at the argument's own top order and
            // is, bit for bit, the per-argument all-orders entry's value at that
            // argument and that top order.
            AllOrders(m, raggeds[i], args[i], row.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const std::size_t index = static_cast<std::size_t>(n) * count + i;

                if (n > raggeds[i])
                {
                    ++above;
                    kept += planes[index] == kUnwritten ? 1 : 0;
                    continue;
                }

                const Cell* cell = Find(cells, n, args[i]);

                if (cell == nullptr)
                {
                    ++missing;
                    continue;
                }

                Judge(ragged, planes[index], cell->value, BatchBound(m), n, args[i]);
                differs += planes[index] == row[static_cast<std::size_t>(n)] ? 0 : 1;
            }
        }

        Require(report, missing == 0, "the grid carries every cell this rule looks up");
        Require(report,
                above > 0 && kept == above,
                "BoysAllNAtOrders leaves the cells above each argument's top order untouched");
        Require(report,
                differs == 0,
                "BoysAllNAtOrders returns the per-argument entry's values bit for bit");

        // One top order for the whole batch is the shape BoysAllN has, and the
        // order-array entry answers it at the same documented bound.
        std::fill(planes.begin(), planes.end(), kUnwritten);
        AllNAtOrders(m, commons.data(), args.data(), planes.data(), count);

        for (std::size_t i = 0; i < count; ++i)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                const Cell* cell = Find(cells, n, args[i]);

                if (cell != nullptr)
                {
                    Judge(common,
                          planes[static_cast<std::size_t>(n) * count + i],
                          cell->value,
                          BatchBound(m),
                          n,
                          args[i]);
                }
            }
        }

        // Documented: count may be 0, and then nothing is written.
        double untouched[2] = {kUnwritten, kUnwritten};
        AllNAtOrders(m, raggeds.data(), args.data(), untouched, 0);
        Require(report,
                untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                "BoysAllNAtOrders writes nothing for count = 0");

        Covered("boys::BoysAllNAtOrders<m>");
    }

    for (const double m : {1.0, 8.0})
    {
        Rule& rule =
            NewRule(RuleName("BoysAllNF32<m = %> (the float all-N batch)", MultiplierName(m)));
        const double bound = m * 1.5e-7 + kOracleBound;

        std::fill(planesF.begin(), planesF.end(), static_cast<float>(kUnwritten));
        AllNF32(m, nmax, argsF.data(), planesF.data(), count);

        std::size_t differs = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            // The float lane rounds its argument to float before evaluating, so
            // the reference is the certified double entry at that rounded
            // argument, in the same shape.
            boys::BoysAllOrders<1.0>(nmax, static_cast<double>(argsF[i]), row.data());
            AllOrdersF32(m, nmax, argsF[i], rowF.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const std::size_t index = static_cast<std::size_t>(n) * count + i;

                Judge(rule, static_cast<double>(planesF[index]), row[n], bound, n, args[i]);
                differs += planesF[index] == rowF[static_cast<std::size_t>(n)] ? 0 : 1;
            }
        }

        Require(report, differs == 0, "BoysAllNF32 returns BoysAllOrdersF32's values bit for bit");

        // Documented: count may be 0, and then nothing is written.
        float untouched[2] = {-1.0f, -1.0f};
        AllNF32(m, nmax, argsF.data(), untouched, 0);
        Require(report,
                untouched[0] == -1.0f && untouched[1] == -1.0f,
                "BoysAllNF32 writes nothing for count = 0");

        Covered("boys::BoysAllNF32<m>");
    }
}

/// The run-time tier entry over the whole grid, at every tier.
void CheckTierLane(const std::vector<Cell>& cells) {
    for (const TierSpec& spec : kTiers)
    {
        char multiplier[32] = {};
        std::snprintf(multiplier, sizeof(multiplier), "%.6g", spec.multiplier);
        const std::string name =
            std::string("BoysAllOrdersAtTier (") + spec.name + ", m = " + multiplier + ")";
        Rule& rule = NewRule(name);

        for (const double x : DistinctArgs(cells))
        {
            double out[boys::kMaxBoysOrder + 1] = {};
            boys::BoysAllOrdersAtTier(spec.tier, boys::kMaxBoysOrder, x, out);

            for (const Cell& cell : cells)
            {
                if (cell.x == x)
                {
                    Judge(
                        rule, out[cell.n], cell.value, BatchBound(spec.multiplier), cell.n, cell.x);
                }
            }
        }
    }
}

/// The fp32 lane. Its argument is a float, so the reference is the certified
/// double lane at that same float argument, and the composed bound is the
/// lane's own figure plus the reference lane's.
void CheckFloatLane(const std::vector<Cell>& cells) {
    for (const double m : {1.0, 8.0})
    {
        Rule& single = NewRule(
            RuleName("BoysSingleF32<m = %> (vs the double lane at float(x))", MultiplierName(m)));
        Rule& batch = NewRule(RuleName("BoysAllOrdersF32<m = %> (vs the double lane at float(x))",
                                       MultiplierName(m)));
        const double bound = m * 1.5e-7 + kOracleBound;

        for (const Cell& cell : cells)
        {
            const float xf = static_cast<float>(cell.x);
            const double oracle = boys::BoysSingle<1.0>(cell.n, static_cast<double>(xf));
            Judge(single,
                  static_cast<double>(SingleF32(m, cell.n, xf)),
                  oracle,
                  bound,
                  cell.n,
                  cell.x);
        }

        for (const double x : DistinctArgs(cells))
        {
            const float xf = static_cast<float>(x);
            float out[boys::kMaxBoysOrder + 1] = {};
            AllOrdersF32(m, boys::kMaxBoysOrder, xf, out);

            for (const Cell& cell : cells)
            {
                if (cell.x == x)
                {
                    const double oracle = boys::BoysSingle<1.0>(cell.n, static_cast<double>(xf));
                    Judge(batch, static_cast<double>(out[cell.n]), oracle, bound, cell.n, cell.x);
                }
            }
        }

        Covered("boys::BoysSingleF32<m>");
        Covered("boys::BoysAllOrdersF32<m>");
    }
}

/// The fp16 and bf16 I/O lanes. Both round their argument to the 16-bit format
/// before evaluating, so the reference is the certified double lane at that
/// rounded argument. The bound is the lane's own figure, claimed only where the
/// value exceeds it; the cells past that ceiling are counted rather than
/// judged, because no accuracy is claimed there.
void CheckHalfIo(const std::vector<Cell>& cells) {
    for (const double m : {1.0, 8.0})
    {
        Rule& f16Single =
            NewRule(RuleName("BoysSingleF16<m = %> (cells above its bound)", MultiplierName(m)));
        Rule& bf16Single =
            NewRule(RuleName("BoysSingleBf16<m = %> (cells above its bound)", MultiplierName(m)));
        Rule& f16Batch =
            NewRule(RuleName("BoysAllOrdersF16<m = %> (cells above its bound)", MultiplierName(m)));
        Rule& bf16Batch = NewRule(
            RuleName("BoysAllOrdersBf16<m = %> (cells above its bound)", MultiplierName(m)));
        std::size_t f16Past = 0;
        std::size_t bf16Past = 0;

        for (const Cell& cell : cells)
        {
            const float xf = static_cast<float>(cell.x);
            const boys::F16 h = boys::F16(xf);
            const boys::Bf16 b = boys::Bf16(xf);
            const double oracleF16 = boys::BoysSingle<1.0>(cell.n, static_cast<double>(h));
            const double oracleBf16 = boys::BoysSingle<1.0>(cell.n, static_cast<double>(b));
            const double gotF16 = static_cast<double>(SingleF16(m, cell.n, h));
            const double gotBf16 = static_cast<double>(SingleBf16(m, cell.n, b));
            const double boundF16 = F16IoBound(gotF16, m);
            const double boundBf16 = Bf16IoBound(gotBf16, m);

            if (std::abs(oracleF16) > boundF16)
            {
                Judge(f16Single, gotF16, oracleF16, boundF16, cell.n, cell.x);
            } else
            {
                ++f16Past;
            }

            if (std::abs(oracleBf16) > boundBf16)
            {
                Judge(bf16Single, gotBf16, oracleBf16, boundBf16, cell.n, cell.x);
            } else
            {
                ++bf16Past;
            }
        }

        for (const double x : DistinctArgs(cells))
        {
            const float xf = static_cast<float>(x);
            const boys::F16 h = boys::F16(xf);
            const boys::Bf16 b = boys::Bf16(xf);
            boys::F16 outF16[boys::kMaxBoysOrder + 1] = {};
            boys::Bf16 outBf16[boys::kMaxBoysOrder + 1] = {};
            AllOrdersF16(m, boys::kMaxBoysOrder, h, outF16);
            AllOrdersBf16(m, boys::kMaxBoysOrder, b, outBf16);

            for (const Cell& cell : cells)
            {
                if (cell.x != x)
                {
                    continue;
                }

                const double oracleF16 = boys::BoysSingle<1.0>(cell.n, static_cast<double>(h));
                const double oracleBf16 = boys::BoysSingle<1.0>(cell.n, static_cast<double>(b));
                const double gotF16 = static_cast<double>(outF16[cell.n]);
                const double gotBf16 = static_cast<double>(outBf16[cell.n]);
                const double boundF16 = F16IoBound(gotF16, m);
                const double boundBf16 = Bf16IoBound(gotBf16, m);

                if (std::abs(oracleF16) > boundF16)
                {
                    Judge(f16Batch, gotF16, oracleF16, boundF16, cell.n, cell.x);
                }

                if (std::abs(oracleBf16) > boundBf16)
                {
                    Judge(bf16Batch, gotBf16, oracleBf16, boundBf16, cell.n, cell.x);
                }
            }
        }

        std::printf("  %-56s fp16 %zu cells past its bound, bf16 %zu\n",
                    "fp16/bf16 I/O lanes (cells at or below their bound)",
                    f16Past,
                    bf16Past);

        Covered("boys::BoysSingleF16<m>");
        Covered("boys::BoysAllOrdersF16<m>");
        Covered("boys::BoysSingleBf16<m>");
        Covered("boys::BoysAllOrdersBf16<m>");
        Covered("boys::F16");
        Covered("boys::Bf16");
    }
}

/// The native packed half lane: region C only, arguments at or above the fp16
/// value of the region-C boundary, results scaled by 2^15, and a bound of 8 ULP
/// of the returned value, claimed only where that value is a normal half.
void CheckNativeHalf(Report& report, const std::vector<Cell>& cells) {
    constexpr double kHalfRegionCBoundary = 28.984375;
    Rule& pair = NewRule("BoysAllOrdersHalf2 (cells at or above a normal half)");
    Rule& array = NewRule("BoysAllNF16Native (cells at or above a normal half)");
    std::size_t inDomain = 0;
    std::size_t outOfDomain = 0;

    for (const double x : DistinctArgs(cells))
    {
        if (x < kHalfRegionCBoundary)
        {
            continue;
        }

        const boys::F16 h = boys::F16(static_cast<float>(x));
        const boys::Half2 packed(h, h);
        boys::Half2 results[boys::kMaxBoysOrder + 1] = {};
        boys::BoysAllOrdersHalf2(boys::kMaxBoysOrder, packed, results);

        const boys::F16 arrayIn[2] = {h, h};
        boys::F16 arrayOut[2 * (boys::kMaxBoysOrder + 1)] = {};
        boys::BoysAllNF16Native(boys::kMaxBoysOrder, arrayIn, arrayOut, 2);

        for (const Cell& cell : cells)
        {
            if (cell.x != x)
            {
                continue;
            }

            // Documented: the output is 2^kHalfNativeScaleExponent * F_k(x),
            // for both halves of the pair and for both entries of the array
            // form's plane, which the same argument fills.
            const double want = std::ldexp(1.0, boys::kHalfNativeScaleExponent) * cell.value;
            const double low = static_cast<double>(results[cell.n].Low());
            const double high = static_cast<double>(results[cell.n].High());
            const double first =
                static_cast<double>(arrayOut[static_cast<std::size_t>(cell.n) * 2]);
            const double second =
                static_cast<double>(arrayOut[static_cast<std::size_t>(cell.n) * 2 + 1]);

            // Documented: the claim holds where the returned value is a normal
            // half, and orders 9 to 32 have no argument at all where this lane
            // returns a value inside its bound.
            if (cell.n >= 9)
            {
                Require(report,
                        low < kHalfMinNormal && high < kHalfMinNormal && first < kHalfMinNormal &&
                            second < kHalfMinNormal,
                        "orders 9 and above return no value inside the lane's bound");
                ++outOfDomain;
            } else if (low >= kHalfMinNormal)
            {
                ++inDomain;
                Judge(pair, low, want, 8.0 * HalfUlpOf(low), cell.n, cell.x);
                Judge(pair, high, want, 8.0 * HalfUlpOf(high), cell.n, cell.x);
                Judge(array, first, want, 8.0 * HalfUlpOf(first), cell.n, cell.x);
                Judge(array, second, want, 8.0 * HalfUlpOf(second), cell.n, cell.x);
            } else
            {
                ++outOfDomain;
            }
        }
    }

    // Documented: count may be 0, and then nothing is written.
    const boys::F16 countZeroIn[2] = {boys::F16(30.0f), boys::F16(30.0f)};
    boys::F16 countZeroOut[4];
    std::fill(std::begin(countZeroOut), std::end(countZeroOut), boys::F16(-1.0f));
    boys::BoysAllNF16Native(boys::kMaxBoysOrder, countZeroIn, countZeroOut, 0);
    Require(report,
            static_cast<double>(countZeroOut[0]) == -1.0 &&
                static_cast<double>(countZeroOut[3]) == -1.0,
            "the native half array form writes nothing for count = 0");

    std::printf("  %-56s %zu cells in the domain, %zu past it\n",
                "native half lane (x >= 28.984375)",
                inDomain,
                outOfDomain);
    Covered("boys::BoysAllOrdersHalf2");
    Covered("boys::BoysAllNF16Native");
    Covered("boys::Half2");
}

/// The region-A transform, in all three of its arithmetic modes, over both
/// bands, at a multiplier the library does not pre-instantiate. The entry is a
/// lane of its own: its bound is the mode's, and its domain is region A.
void CheckProductModes(Report& report, const std::vector<Cell>& cells) {
    struct ModeSpec {
        boys::ProductMode mode;
        const char* name;
    };

    const ModeSpec kModes[] = {{boys::ProductMode::kFp64, "kFp64"},
                               {boys::ProductMode::kTf32x3, "kTf32x3"},
                               {boys::ProductMode::kBf16x6, "kBf16x6"},
                               {boys::ProductMode::kTf32, "kTf32"},
                               {boys::ProductMode::kBf16, "kBf16"},
                               {boys::ProductMode::kFp16, "kFp16"}};

    // Every enumerator has exactly one report row, and every row is an
    // enumerator: a mode a caller can name but cannot ask about is the gap
    // this check exists for, and so is a row for a mode that is not in the
    // enumeration.
    {
        const std::span<const boys::ProductModeInfo> rows = boys::BoysProductModes();
        Require(report,
                rows.size() == std::size(kModes),
                "BoysProductModes has one row per ProductMode enumerator");

        for (const ModeSpec& mode : kModes)
        {
            std::size_t seen = 0;

            for (const boys::ProductModeInfo& row : rows)
            {
                seen += row.mode == mode.mode ? 1u : 0u;
            }

            Require(report, seen == 1, "BoysProductModes has exactly one row per mode");
        }

        Covered("boys::ProductModeInfo");
        Covered("boys::ModeCertification");
        Covered("boys::BoysProductModes");

        // The report is the only place a caller learns which rows a card is
        // held to, so the class has to be reported rather than implied: the
        // fp64 mode is the one whose arithmetic is an ordinary double sum.
        int certified = 0;

        for (const boys::ProductModeInfo& row : rows)
        {
            Require(report,
                    row.model != nullptr && row.model[0] != '\0',
                    "every mode row names what its bound is a claim about");

            if (row.certification == boys::ModeCertification::kCertified)
            {
                ++certified;
                Require(report,
                        row.mode == boys::ProductMode::kFp64,
                        "only the fp64 mode is certified, and it is");
            }
        }

        Require(report,
                certified == 1,
                "exactly one product mode is certified by a measurement of hardware");
    }
    const std::vector<double> all = DistinctArgs(cells);

    for (const ModeSpec& mode : kModes)
    {
        for (const boys::RegionABand band : {boys::RegionABand::kA1, boys::RegionABand::kA2})
        {
            const bool lower = band == boys::RegionABand::kA1;
            std::vector<double> args;

            for (const double x : all)
            {
                const bool inBand = lower ? x < boys::kRegionA1Edge
                                          : x >= boys::kRegionA1Edge && x < boys::kRegionAEnd;

                if (inBand)
                {
                    args.push_back(x);
                }
            }

            const std::string name = std::string("BoysRegionAProduct<") + mode.name +
                                     (lower ? ", kA1, m = 8>" : ", kA2, m = 8>");
            Rule& rule = NewRule(name);
            std::vector<double> out(args.size() * (boys::kMaxBoysOrder + 1));
            std::fill(out.begin(), out.end(), kUnwritten);
            RegionAProduct(
                mode.mode, 8.0, band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());

            for (std::size_t i = 0; i < args.size(); ++i)
            {
                for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
                {
                    const Cell* cell = Find(cells, n, args[i]);

                    if (cell != nullptr)
                    {
                        Judge(rule,
                              out[static_cast<std::size_t>(n) * args.size() + i],
                              cell->value,
                              ProductBound(report, mode.mode, 8.0),
                              n,
                              args[i]);
                    }
                }
            }

            // Documented: count may be 0, and then nothing is written.
            double untouched[2] = {kUnwritten, kUnwritten};
            const double one[1] = {args.empty() ? 0.0 : args[0]};
            RegionAProduct(mode.mode, 8.0, band, 0, one, untouched, 0);
            Require(report,
                    untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                    "BoysRegionAProduct writes nothing for count = 0");
        }

        Covered("boys::ProductMode");
        Covered("boys::BoysRegionAProduct (three modes, both bands)");
    }

    // The default multiplier is a documented choice too, and it is the one the
    // library's own translation unit instantiates: a consumer that names it
    // links against that instantiation rather than compiling a second copy.
    std::vector<double> low;

    for (const double x : all)
    {
        if (x < boys::kRegionA1Edge)
        {
            low.push_back(x);
        }
    }

    Rule& defaultRule = NewRule("BoysRegionAProduct<kFp64, kA1, m = 1> (the default)");
    std::vector<double> out(low.size() * (boys::kMaxBoysOrder + 1));
    RegionAProduct(boys::ProductMode::kFp64,
                   1.0,
                   boys::RegionABand::kA1,
                   boys::kMaxBoysOrder,
                   low.data(),
                   out.data(),
                   low.size());

    for (std::size_t i = 0; i < low.size(); ++i)
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const Cell* cell = Find(cells, n, low[i]);

            if (cell != nullptr)
            {
                Judge(defaultRule,
                      out[static_cast<std::size_t>(n) * low.size() + i],
                      cell->value,
                      ProductBound(report, boys::ProductMode::kFp64, 1.0),
                      n,
                      low[i]);
            }
        }
    }
}

/// The packed half type's own surface: construction, the accessors, and the
/// arithmetic, on values whose results are exactly representable in binary16 -
/// so a correct operation is asserted exactly, not within a tolerance.
void CheckHalf2Surface(Report& report) {
    const boys::F16 kOneAndAHalf(1.5f);
    const boys::F16 kQuarter(0.25f);
    const boys::F16 kTwo(2.0f);
    const boys::F16 kThree(3.0f);
    const boys::F16 kFour(4.0f);
    const boys::F16 kNine(9.0f);
    const boys::Half2 a(kOneAndAHalf, kTwo);
    const boys::Half2 b(kQuarter, boys::F16(0.5f));

    Require(report,
            static_cast<double>(a.Low()) == 1.5 && static_cast<double>(a.High()) == 2.0,
            "Half2(low, high) packs the halves it is given");
    Require(report, boys::Half2::FromBits(a.Bits()) == a, "Half2::FromBits inverts Bits()");
    Require(report,
            boys::Half2::Broadcast(kOneAndAHalf).Low() == kOneAndAHalf &&
                boys::Half2::Broadcast(kOneAndAHalf).High() == kOneAndAHalf,
            "Half2::Broadcast puts one value in both halves");

    const boys::Half2 sum = boys::Half2Add(a, b);
    const boys::Half2 difference = boys::Half2Sub(a, b);
    const boys::Half2 product = boys::Half2Mul(a, boys::Half2(kTwo, kThree));
    const boys::Half2 quotient =
        boys::Half2Div(boys::Half2(kThree, kFour), boys::Half2(kTwo, kTwo));
    const boys::Half2 root = boys::Half2Sqrt(boys::Half2(kFour, kNine));

    Require(report,
            static_cast<double>(sum.Low()) == 1.75 && static_cast<double>(sum.High()) == 2.5,
            "Half2Add adds both halves");
    Require(report,
            static_cast<double>(difference.Low()) == 1.25 &&
                static_cast<double>(difference.High()) == 1.5,
            "Half2Sub subtracts both halves");
    Require(report,
            static_cast<double>(product.Low()) == 3.0 && static_cast<double>(product.High()) == 6.0,
            "Half2Mul multiplies both halves");
    Require(report,
            static_cast<double>(quotient.Low()) == 1.5 &&
                static_cast<double>(quotient.High()) == 2.0,
            "Half2Div divides both halves");
    Require(report,
            static_cast<double>(root.Low()) == 2.0 && static_cast<double>(root.High()) == 3.0,
            "Half2Sqrt takes the root of both halves");

    // The widening conversions, and NextUp as the successor of a half value.
    Require(report,
            static_cast<double>(kOneAndAHalf) == 1.5 && static_cast<float>(kQuarter) == 0.25f,
            "the half types widen to float and double exactly");
    Require(report,
            static_cast<double>(boys::NextUp(kQuarter)) > static_cast<double>(kQuarter),
            "NextUp returns the successor of a half value");
    Require(report,
            static_cast<double>(boys::F16(boys::NextUp(kQuarter))) ==
                static_cast<double>(boys::NextUp(kQuarter)),
            "the successor of a half value is itself a half value");
    Require(report,
            static_cast<double>(boys::Bf16(1.5f)) == 1.5,
            "the bf16 type round-trips a value its format holds");

    Covered("boys::Half2");
    Covered("boys::Half2Add");
    Covered("boys::Half2Sub");
    Covered("boys::Half2Mul");
    Covered("boys::Half2Div");
    Covered("boys::Half2Sqrt");
    Covered("boys::NextUp");
}

/// The evaluation schemes a consumer can ask about and ask for. What a
/// consumer reads here is the whole of the option: which arithmetic is in
/// force, what each scheme promises on each stored fit, and that the entries
/// answer under the scheme that was named - which is the part a reading of the
/// enumeration alone cannot show.
void CheckEvalSchemes(Report& report, const std::vector<Cell>& cells) {
    const std::span<const boys::EvalSchemeInfo> schemes = boys::BoysEvalSchemes();
    const std::span<const boys::EvalFitInfo> fits = boys::BoysEvalSchemeFits();

    Require(report, !schemes.empty(), "BoysEvalSchemes reports at least one scheme");
    Require(report, !fits.empty(), "BoysEvalSchemeFits reports at least one stored fit");

    // The route the bounds are stated in, taken from the table a consumer reads
    // for the arithmetic rather than from a compile-time guess.
    boys::backend::MulAddRoute route = boys::backend::MulAddRoute::kFused;
    bool routeFound = false;

    for (const boys::backend::BackendInfo& info : boys::backend::BoysBackends())
    {
        if (std::strcmp(info.name, "scalar-fp64") == 0)
        {
            route = info.route;
            routeFound = true;
        }
    }

    Require(report,
            routeFound,
            "BoysBackends names the scalar-fp64 arithmetic the scheme bounds are in");

    for (const boys::EvalSchemeInfo& info : schemes)
    {
        Require(report, info.name != nullptr && *info.name != '\0', "a scheme row carries a name");
        Require(report,
                std::strcmp(boys::EvalSchemeName(info.scheme), info.name) == 0,
                "EvalSchemeName agrees with the name the row carries");
        Require(report, info.lanes > 0, "a scheme row serves at least one stored fit");
        Require(report, info.deg > 0 && info.stored > info.deg,
                "a scheme row carries a degree and the stored count that degree needs");
        Require(report, info.delivered > 0.0, "a scheme row promises a positive bound");
        Require(report,
                info.route == route,
                "a scheme row's bound is stated in the arithmetic this build runs");

        bool anyFit = false;

        for (const boys::EvalFitInfo& fit : fits)
        {
            if (fit.scheme != info.scheme)
            {
                continue;
            }

            anyFit = true;
            const double bound = boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane);
            Require(report, bound > 0.0, "every stored fit has a positive bound");
            Require(report, bound <= info.delivered, "the aggregate is the worst fit's bound");
            Require(report,
                    bound == (route == boys::backend::MulAddRoute::kSeparate ? fit.separate
                                                                            : fit.fused),
                    "the bound is the one the route in force carries");
        }

        Require(report, anyFit, "every scheme row has its fits");
    }

    // A pair no scheme carries answers zero rather than a guess.
    Require(report,
            boys::BoysEvalSchemeDelivered(static_cast<boys::EvalScheme>(99),
                                          boys::EvalLane::kRegionA) == 0.0,
            "a scheme this build does not carry delivers no bound");

    const auto schemeBound = [&](boys::EvalScheme scheme) {
        double worst = 0.0;

        for (const boys::EvalFitInfo& fit : fits)
        {
            if (fit.scheme == scheme)
            {
                worst = std::max(worst, boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane));
            }
        }

        return worst;
    };

    const double worstSchemeBound = schemeBound(boys::EvalScheme::kHorner);
    Require(report, worstSchemeBound > 0.0, "the Horner scheme publishes a bound");

    // The two axes compose into one selection: a policy names the fit route and
    // the scheme together, and every templated entry takes that policy. The
    // default policy is the certified pair, so a call site that names neither
    // axis is the call this library has always answered.
    using DefaultPolicy = boys::EvalPolicy<>;
    using ClenshawPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                           boys::EvalScheme::kSplitClenshaw>;
    using HornerPolicy =
        boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;

    // The policy's fields read back what it was named with, so a consumer can
    // ask a policy which pair it carries rather than reading its type.
    constexpr DefaultPolicy kDefaultPolicy{};
    static_assert(kDefaultPolicy.kRoute == boys::FitRoute::kChebyshev &&
                      kDefaultPolicy.kScheme == boys::EvalScheme::kSplitClenshaw &&
                      kDefaultPolicy.kBudget == boys::BoysBudget::kFloat,
                  "the default policy is the Chebyshev route by the split Clenshaw recurrence "
                  "at the float lane's budget");
    static_assert(HornerPolicy{}.kScheme == boys::EvalScheme::kHorner,
                  "a policy carries the scheme it was named with");

    // The two schemes sum one polynomial, so the entries that name them differ
    // by no more than their own bounds and the Horner entry is inside the
    // certified entry's error plus twice that. A scheme that reached the wrong
    // fit, or no fit, would be wrong by orders of magnitude rather than by a
    // bound, which is what this separates.
    for (const Cell& cell : cells)
    {
        const double byDefault =
            boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(cell.n, cell.x);
        const double byName =
            boys::BoysSingle<boys::kBoysFullAccuracyMultiplier, ClenshawPolicy>(cell.n, cell.x);
        Require(report, byDefault == byName, "a call naming no policy is the certified route");

        std::array<double, 33> un = {};
        std::array<double, 33> named = {};
        boys::BoysAllOrders<boys::kBoysFullAccuracyMultiplier>(cell.n, cell.x, un.data());
        boys::BoysAllOrders<boys::kBoysFullAccuracyMultiplier, ClenshawPolicy>(
            cell.n, cell.x, named.data());
        Require(report,
                std::memcmp(un.data(), named.data(), sizeof(un)) == 0,
                "a batch call naming no policy is the certified route, bit for bit");

        // The other scheme is reachable from both entries, and they agree with
        // each other: the batch entry is not a second, differently-wired way in.
        const double horner =
            boys::BoysSingle<boys::kBoysFullAccuracyMultiplier, HornerPolicy>(cell.n, cell.x);
        Require(report, std::isfinite(horner), "the Horner scheme answers a finite value");

        std::array<double, 33> hornerOut = {};
        boys::BoysAllOrders<boys::kBoysFullAccuracyMultiplier, HornerPolicy>(
            cell.n, cell.x, hornerOut.data());
        Require(report,
                hornerOut[static_cast<std::size_t>(cell.n)] == horner,
                "the batch entry answers the Horner scheme the same value as the single entry");

        Require(report,
                std::abs(horner - cell.value) <=
                    std::abs(byDefault - cell.value) + 2.0 * worstSchemeBound,
                "the Horner entry is inside the certified entry's error plus the "
                "two schemes' own bounds");
    }

    Covered("boys::EvalScheme");
    Covered("boys::kDefaultEvalScheme");
    Covered("boys::EvalSchemeName");
    Covered("boys::EvalSchemeInfo");
    Covered("boys::EvalFitInfo");
    Covered("boys::EvalLane");
    Covered("boys::BoysEvalSchemes");
    Covered("boys::BoysEvalSchemeFits");
    Covered("boys::BoysEvalSchemeDelivered");
    Covered("boys::EvalPolicy");
    Covered("boys::FitRoute");
    Covered("boys::kDefaultFitRoute");
}

// The interval-granularity axis, reached the way a consumer reaches it: by
// naming the partition on the policy and calling the entries.
//
// The member is a second partition of the fitted domain - region A's pieces and
// region B's seed - so the things a consumer has to be able to read from it are
// that naming it changes the values over the fitted domain at both of the
// regions the two tables are cut in, so that the member is a partition and not
// the shipped tables under another name; that it changes nothing at or above the
// fitted domain's end, the axis being a selection between two stored tables and
// not a second arithmetic path; that naming the shipped member is the default
// call bit for bit, so the default is a member of the axis rather than a third
// reading beside it; and that every value it returns is inside the lane's
// published bound, so the member does not widen the contract a caller already
// relies on.
//
// The counts the partition costs - the coefficients one evaluation reads and
// the coefficients the table stores - are the generated header's own
// static_assert and the gate's narrow rows; what is asserted here is that the
// policy carries the partition it was named with, so a call site that names one
// is not silently handed the other.
void CheckGranularityLane(Report& report, const std::vector<Cell>& cells) {
    using NarrowPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                          boys::EvalScheme::kSplitClenshaw,
                                          boys::BoysBudget::kFloat,
                                          boys::kDefaultPackAxis,
                                          boys::FitGranularity::kNarrow>;
    using ShippedPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                           boys::EvalScheme::kSplitClenshaw,
                                           boys::BoysBudget::kFloat,
                                           boys::kDefaultPackAxis,
                                           boys::FitGranularity::kShipped>;

    static_assert(NarrowPolicy{}.kGranularity == boys::FitGranularity::kNarrow &&
                      ShippedPolicy{}.kGranularity == boys::FitGranularity::kShipped,
                  "a policy carries the partition it was named with");
    static_assert(!std::is_same_v<NarrowPolicy::Fit, ShippedPolicy::Fit>,
                  "the two partitions are different fits: neither is the other under a second "
                  "name");

    Require(report,
            std::strcmp(boys::GranularityName(boys::FitGranularity::kShipped),
                        boys::GranularityName(boys::FitGranularity::kNarrow)) != 0,
            "the two partitions are reported under different names rather than one blank");

    Rule& rule = NewRule("granularity: the narrow partition through the entries");

    // x1, where the fitted domain ends and the asymptotic path takes over, as
    // the umbrella header publishes it - region B runs x0 <= x < x1 and region C
    // is x >= x1. The public surface names no constant for it, so it is
    // transcribed the way the lane bounds at the head of this file are.
    constexpr double kFittedDomainEnd = 28.98933773882074;

    std::size_t inA = 0;
    std::size_t changedInA = 0;
    std::size_t inB = 0;
    std::size_t changedInB = 0;
    std::size_t aboveDomain = 0;
    std::size_t changedAboveDomain = 0;
    std::size_t shippedDiffering = 0;

    for (const Cell& cell : cells)
    {
        const double byDefault = boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(cell.n, cell.x);
        const double narrow =
            boys::BoysSingle<boys::kBoysFullAccuracyMultiplier, NarrowPolicy>(cell.n, cell.x);
        const double named =
            boys::BoysSingle<boys::kBoysFullAccuracyMultiplier, ShippedPolicy>(cell.n, cell.x);

        if (cell.x < boys::kRegionAEnd)
        {
            ++inA;

            if (narrow != byDefault)
            {
                ++changedInA;
            }
        } else if (cell.x < kFittedDomainEnd)
        {
            ++inB;

            if (narrow != byDefault)
            {
                ++changedInB;
            }
        } else
        {
            ++aboveDomain;

            if (narrow != byDefault)
            {
                ++changedAboveDomain;
            }
        }

        if (named != byDefault)
        {
            ++shippedDiffering;
        }

        Judge(rule, narrow, cell.value, SingleBound(cell.x, 1.0), cell.n, cell.x);
    }

    // Each rule says which cells it measured rather than passing on an empty
    // sweep: a grid that carries no argument of a region would otherwise leave
    // the reading vacuous.
    Require(report,
            inA > 0 && changedInA > 0,
            "naming the narrow partition changes region A's values: the member cuts region A's "
            "pieces as well as region B's seed, and the change is visible through the entry");
    Require(report,
            inB > 0 && changedInB > 0,
            "naming the narrow partition changes region B's values: the member is a partition "
            "and not the shipped seed under another name");
    Require(report,
            aboveDomain > 0 && changedAboveDomain == 0,
            "naming the narrow partition changes nothing at or above the fitted domain's end: "
            "above it the entry reads the asymptotic path, which no partition of the stored "
            "fits is part of");
    Require(report,
            shippedDiffering == 0,
            "naming the shipped partition is the default call bit for bit, so the default is "
            "that member and not a third reading beside the two");

    std::printf("  %-56s %7zu cells  %zu of %zu in region A changed, %zu of %zu in region B, "
                "%zu of %zu above the fitted domain\n",
                rule.name.c_str(),
                rule.cells,
                changedInA,
                inA,
                changedInB,
                inB,
                changedAboveDomain,
                aboveDomain);

    Covered("boys::FitGranularity");
    Covered("boys::kDefaultFitGranularity");
    Covered("boys::GranularityName");
}

} // namespace

int main(int argc, char** argv) {
    const char* gridPath = argc > 1 ? argv[1] : BoysConsumerReference;
    const std::vector<Cell> cells = LoadGrid(gridPath);

    if (cells.empty())
    {
        std::printf("consumer check: cannot read the reference grid at %s\n", gridPath);
        return 1;
    }

    Report report;
    CheckConstants(report);
    CheckTiers(report);
    CheckEvalSchemes(report, cells);
    CheckGranularityLane(report, cells);
    CheckDoubleLanes(report, cells);
    CheckFitRoutes(report, cells);
    CheckFitRoutesF32(report, cells);
    CheckFloatPolicies(report, cells);
    CheckManyArgumentLanes(report, cells);
    CheckPerElementOrderLanes(report, cells);
    CheckTierLane(cells);
    CheckFloatLane(cells);
    CheckHalfIo(cells);
    CheckNativeHalf(report, cells);
    CheckProductModes(report, cells);
    CheckHalf2Surface(report);

    std::printf("consumer check through <boys/boys.hpp>: %zu grid cells\n", cells.size());
    PrintRules();
    std::printf("  %zu assertions, %zu failed; %zu documented entries covered\n",
                report.assertions,
                report.failed,
                gCovered.size());
    // The two entries that share a recurrence, compared bit for bit and counted
    // rather than only asserted, so a host that parts company with the exact
    // form on a build this run did not assert it for is named here by a figure.
    std::printf("  fixed-order against single-lane, bit for bit: %zu compared, %zu agreed\n",
                report.exactCompared,
                report.exactHeld);

    return report.failed == 0 ? 0 : 1;
}
