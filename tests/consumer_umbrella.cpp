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
//    file's include path;
//
//  * every documented public choice is exercised: the two fit routes of the
//    double lane, the two evaluation schemes, the product modes of the region-A
//    transform over both bands, the sorted-argument and workspace forms of the
//    many-argument entry, its per-element-order form in both precisions, the
//    fp16/bf16 I/O lanes, the native packed half lane, and the lane templates at
//    a policy the library does not pre-instantiate - the case that is a link
//    error when a definition lives in a .cpp file rather than in the header its
//    declaration ships in;
//
//  * every returned value is judged against the bound its entry documents, with
//    the committed 45-digit reference grid as the reference for the arguments it
//    carries. Where a format conversion sits between the reference and the entry
//    (the fp32 and fp16 lanes round their argument, the native half lane returns
//    2^15 F_k), the comparison is made against the certified double lane at the
//    same converted argument, and the rule's name states the bound that
//    composition carries;
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
    "this consumer check is compiled with src/ on its include path; it does not prove that a consumer can build against the public headers alone"
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

    /// The bit-for-bit comparisons this run made of an entry's exact form, and how
    /// many held. Printed on every build; a count below the other names a host that
    /// parts company with the exact form by a figure rather than a silent pass.
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
/// A value that is not a finite number is counted as exceeded rather than compared,
/// because a ratio against a bound is false for every NaN and would slip through.
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

/// A rule registered up front, so the verdict lists every rule it intended to run.
Rule& NewRule(const std::string& name) {
    gRules.push_back(Rule{name, 0, 0, 0.0, -1, 0.0});
    return gRules.back();
}

/// The documented entries this run reached. The verdict prints the count, so "what
/// of the surface is covered" is a statement about the run, not about this file.
std::vector<std::string> gCovered;

void Covered(const char* name) {
    if (std::find(gCovered.begin(), gCovered.end(), name) == gCovered.end())
    {
        gCovered.emplace_back(name);
    }
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

/// The double single lane's bound at x, from the published figures: 1e-15
/// below the tightest per-order end of region A, 3e-14 below the end of
/// region A (the extended band's figure, which covers region A as well), and
/// 5.5e-14 past it.
double SingleBound(double x) {
    if (x < 1.0855)
    {
        return 1e-15;
    }

    if (x < boys::kRegionAEnd)
    {
        return 3e-14;
    }

    return 5.5e-14;
}

/// The double batch lane's bound: one figure over the whole argument line.
double BatchBound() {
    return 5.5e-14;
}

/// The quantum (one ULP) of a binary16 or bfloat16 value of the given
/// significand width: 2^(e - bits) for a normal value 1.f * 2^e, and zero at
/// zero, where no quantum is defined. The half lanes' bounds below call it.
#if BoysFp16
double QuantumOf(double v, int significandBits) {
    if (v == 0.0)
    {
        return 0.0;
    }

    int exponent = 0;
    (void)std::frexp(std::abs(v), &exponent);
    return std::ldexp(1.0, exponent - 1 - significandBits);
}
#endif // BoysFp16

// The half lanes' ceilings, which belong to the lanes the BoysFp16 seam
// declares and leave the build with them: the smallest positive normal
// binary16 value, the floor of the native half lane's documented domain, and
// the two I/O lanes' bounds. main states the lanes this build does not carry.
#if BoysFp16
/// The smallest positive normal binary16 value: the floor of the documented domain
/// of the native half lane, whose claim holds where the returned value is a normal
/// half.
constexpr double kHalfMinNormal = 6.103515625e-05; // 2^-14

/// The fp16 lane's bound at a returned value: 1e-7 plus the representation
/// term, one half-ULP of the returned half.
double F16IoBound(double returned) {
    return 1e-7 + 0.5 * QuantumOf(returned, 10);
}

/// The bf16 lane's bound: 1e-7 plus one half-ULP of the returned bf16, whose
/// significand is 8 bits wide.
double Bf16IoBound(double returned) {
    return 1e-7 + 0.5 * QuantumOf(returned, 7);
}

/// One ULP of a normal half value: 2^(e - 10) for a value 1.f * 2^e. The
/// native half lane's bound is stated in these, so it belongs to that lane.
double HalfUlpOf(double v) {
    return QuantumOf(v, 10);
}
#endif // BoysFp16

/// The region-A product's bound per mode: the fit term, plus the two split
/// modes' 32-bit accumulator floor.
double ProductBound(Report& report, boys::ProductMode mode) {
    // Read out of the library's own report rather than restated here: the bounds are
    // what BoysProductModes answers, so a mode added to the enumeration arrives in
    // this sweep with its own bound instead of the 2.5e-7 the split modes carry.
    for (const boys::ProductModeInfo& row : boys::BoysProductModes())
    {
        if (row.mode == mode)
        {
            return row.fitTerm + row.floor;
        }
    }

    Require(report, false, "every mode the sweep names has a BoysProductModes row");
    return 0.0;
}

/// The reference lane's own bound, carried by every composed comparison: the
/// certified double lane stands in for the grid wherever a format conversion
/// sits between the grid and the entry.
constexpr double kOracleBound = 5.5e-14;

/// A value no Boys entry returns: F_n(x) is positive and at most 1 for every
/// supported order and argument, so a negative marker says "not written".
constexpr double kUnwritten = -1.0;

/// The figure the float lane's contract publishes for one class's division form:
/// the base the lane documents, plus the term the plain reciprocal adds beside it
/// where that form is the one the class runs. README's contract row states the
/// figure per form - "float single / batch | <= 1.5e-7, or <= 2.5e-7 in the
/// plain-reciprocal form" - and the form a call runs is its own class's row, so a
/// rule judged at one form's figure while the entry divides in the other measures
/// the call against an arithmetic it did not run. Both numbers are read from
/// BoysLaneContracts rather than written here: the base is the row's own
/// documented figure and the term beside it is the row's own plain-form term.
template <boys::Precision kPrecision, boys::Shape kShape>
double FloatFigure() {
    using Class = boys::DefaultPolicy<kPrecision, kShape>;
    double base = 0.0;
    double plainTerm = 0.0;

    for (const boys::LaneContractInfo& row : boys::BoysLaneContracts())
    {
        if (row.precision == kPrecision)
        {
            base = row.bound;
            plainTerm = row.plainAdditive;
        }
    }

    return base + (Class::kDivision == boys::DivisionForm::kPlainReciprocal ? plainTerm : 0.0);
}

// --- the entries the rules sweep --------------------------------------------
//
// Each helper names its entry at the lane's own default policy - the call an
// entry naming no policy compiles. A rule that reads a policy of its own calls
// the entry itself, so the policy is named where the reader meets it.

double Single(int n, double x) {
    return boys::BoysSingle<>(n, x);
}

void AllOrders(int nmax, double x, double* out) {
    boys::BoysAllOrders<>(nmax, x, out);
}

void FixedN(int n, const double* x, double* out, std::size_t count, std::size_t stride) {
    boys::BoysFixedN<>(n, x, out, count, stride);
}

void AllN(int nmax, const double* x, double* out, std::size_t count, std::size_t* ws) {
    boys::BoysAllN<>(nmax, x, out, count, ws);
}

void AllNSorted(int nmax, const double* x, double* out, std::size_t count) {
    boys::BoysAllN<>(nmax, x, out, count, boys::BoysSortedArgs{});
}

void AllNAtOrders(const int* n, const double* x, double* out, std::size_t count) {
    boys::BoysAllNAtOrders<>(n, x, out, count);
}

void AllNF32(int nmax, const float* x, float* out, std::size_t count) {
    boys::BoysAllNF32<>(nmax, x, out, count);
}

float SingleF32(int n, float x) {
    return boys::BoysSingleF32<>(n, x);
}

void AllOrdersF32(int nmax, float x, float* out) {
    boys::BoysAllOrdersF32<>(nmax, x, out);
}

// The fp16 and bf16 I/O lanes' call sites: the entries the BoysFp16 seam
// declares, so they are compiled with the seam and CheckHalfIo, the check that
// names them, is too.
#if BoysFp16
boys::F16 SingleF16(int n, boys::F16 x) {
    return boys::BoysSingleF16<>(n, x);
}

void AllOrdersF16(int nmax, boys::F16 x, boys::F16* out) {
    boys::BoysAllOrdersF16<>(nmax, x, out);
}

boys::Bf16 SingleBf16(int n, boys::Bf16 x) {
    return boys::BoysSingleBf16<>(n, x);
}

void AllOrdersBf16(int nmax, boys::Bf16 x, boys::Bf16* out) {
    boys::BoysAllOrdersBf16<>(nmax, x, out);
}
#endif // BoysFp16

void RegionAProduct(boys::ProductMode mode,
                    boys::RegionABand band,
                    int nmax,
                    const double* x,
                    double* out,
                    std::size_t count) {
    switch (mode)
    {
    case boys::ProductMode::kFp64:
        boys::BoysRegionAProduct<boys::ProductMode::kFp64>(band, nmax, x, out, count);
        return;
    case boys::ProductMode::kTf32x3:
        boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(band, nmax, x, out, count);
        return;
    case boys::ProductMode::kBf16x6:
        boys::BoysRegionAProduct<boys::ProductMode::kBf16x6>(band, nmax, x, out, count);
        return;
    case boys::ProductMode::kTf32:
        boys::BoysRegionAProduct<boys::ProductMode::kTf32>(band, nmax, x, out, count);
        return;
    case boys::ProductMode::kBf16:
        boys::BoysRegionAProduct<boys::ProductMode::kBf16>(band, nmax, x, out, count);
        return;
    case boys::ProductMode::kFp16:
        boys::BoysRegionAProduct<boys::ProductMode::kFp16>(band, nmax, x, out, count);
        return;
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
#if BoysFp16
    Require(report,
            boys::kHalfNativeScaleExponent == 15,
            "kHalfNativeScaleExponent is the documented 2^15 scale");
#endif

    // The version a caller reads is the version the build was configured at: the
    // build carries BoysExpectedVersion (from project()), so a release that bumps
    // one and not the other fails here rather than shipping two answers. Parsed by
    // hand rather than with sscanf, which MSVC deprecates and this tree builds with
    // warnings as errors; the format is project()'s, three dot-separated decimals.
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
    Covered("boys::BoysAllNWorkspaceSize");

#if BoysFp16
    Covered("boys::kHalfNativeScaleExponent");
#endif

    // The capability predicate is documented as a pure function of the
    // processor, and as reporting false on every non-x86_64 target.
    const bool first = boys::BoysAvx2Available();
    Require(
        report, boys::BoysAvx2Available() == first, "BoysAvx2Available() is stable across calls");
#if !defined(__x86_64__) && !defined(_M_X64)
    Require(report, !first, "BoysAvx2Available() reports false on a non-x86_64 target");
#endif
    Covered("boys::BoysAvx2Available");

    // The library publishes whether this build contracts a bare product-plus-add,
    // measured when the build was configured, and the lane claims below are compiled
    // against that answer. It is a statement about one translation unit's arithmetic,
    // so it is checked here in the unit that makes those claims: a build whose flags
    // reached this file but not the measurement would assert the other arithmetic's
    // claim and be green for it.
#if defined(BOYS_SCALAR_CONTRACTS)
    Require(report,
            boys::backend::ScalarFp64::Contracts() == (BOYS_SCALAR_CONTRACTS != 0),
            "the configure-time contraction measurement is this translation unit's");
#endif
}

/// BoysSingle, BoysAllOrders and BoysFixedN over the whole grid, at the double
/// lane's own default policy.
void CheckDoubleLanes(Report& report, const std::vector<Cell>& cells) {
    // The three entries at one policy, judged cell by cell.
    {
        Rule& single = NewRule("BoysSingle (grid sweep)");
        Rule& batch = NewRule("BoysAllOrders (grid sweep)");
        Rule& fixed = NewRule("BoysFixedN (strided)");

        for (const Cell& cell : cells)
        {
            Judge(single,
                  Single(cell.n, cell.x),
                  cell.value,
                  SingleBound(cell.x),
                  cell.n,
                  cell.x);
        }

        for (const double x : DistinctArgs(cells))
        {
            double all[boys::kMaxBoysOrder + 1] = {};
            AllOrders(boys::kMaxBoysOrder, x, all);

            for (const Cell& cell : cells)
            {
                if (cell.x == x)
                {
                    Judge(batch, all[cell.n], cell.value, BatchBound(), cell.n, cell.x);
                }
            }

            // Documented: each output element of the fixed-order entry carries the
            // single lane's per-region bound at the same order and argument, and is
            // that entry's value bit for bit.
            //
            // Same recurrence, same source - but the same source is not the same
            // bits: whether a bare product-plus-add in it is one rounding or two is
            // a licence the compiler holds per call site. The equality is asserted
            // on every build, because it has held at every cell swept on every host
            // measured; the bound is asserted beside it so a host where the equality
            // stops holding is still judged on what the lane promises there.
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
                FixedN(cell.n, &x, &plain, 1, 1);
                Judge(fixed, plain, cell.value, SingleBound(cell.x), cell.n, cell.x);

                const double singleValue = Single(cell.n, x);

                if (std::abs(plain - singleValue) > SingleBound(cell.x))
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

            // Documented: out[i * stride] = F_n(x[i]), so the same value lands at
            // offset 0 of a stride-1 call and at offsets 0 and 3 of a stride-3 one,
            // and the slots between them are the caller's, untouched. The value is
            // the single lane's by the paragraph above, so the two claims are split
            // the same way: the bound on every build, the exact offset on the builds
            // whose bare product-plus-add is two roundings. This shape is the one
            // place the difference has been measured: a call with two arguments and
            // a gap is not the same generated code as a call with one, and on a
            // contracting build the compiler fused the recurrence at one of the two
            // and not at the other, moving the value by one unit in the last place.
            const double two[2] = {x, x * 0.5 + 0.25};
            const double one = Single(3, x);
            const double other = Single(3, two[1]);
            double strided[7];
            std::fill(std::begin(strided), std::end(strided), kUnwritten);
            FixedN(3, two, strided, 2, 3);
            Require(report,
                    std::abs(strided[0] - one) <= SingleBound(x),
                    "a stride of 3 writes the first value inside the single lane's bound");
            Require(report,
                    strided[1] == kUnwritten && strided[2] == kUnwritten,
                    "a stride of 3 leaves the padding between values untouched");
            Require(report,
                    std::abs(strided[3] - other) <= SingleBound(two[1]),
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
            FixedN(3, two, untouched, 0, 1);
            Require(report,
                    untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                    "BoysFixedN writes nothing for count = 0");
        }

        Covered("boys::BoysSingle<>");
        Covered("boys::BoysAllOrders<>");
        Covered("boys::BoysFixedN<>");
    }
}

/// The run-time fit routes. A consumer reaches the choice through the public report
/// alone: which routes exist, the interval each fit covers, the argument each
/// selector takes over at and the bar each is certified against all come from
/// BoysFitRoutes(), so every row here is judged against the figures that row states.
void CheckFitRoutes(Report& report, const std::vector<Cell>& cells) {
    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();
    const std::vector<double> args = DistinctArgs(cells);

    // The class every unnamed call in this check is made through, and the domain
    // the table that class reads covers. A row of BoysFitRoutes states the
    // intervals of the per-order partitions' fits; a class row naming the grid
    // reads one fixed table instead, whose domain is its own row of
    // BoysFitGranularities - the grid covers [0, kFlatHi) whole, region A's and
    // region B's arguments alike, and the body answers the whole of that domain
    // from the table and returns before the region tests. Where the class row
    // names that partition, naming a route reaches every argument the partition
    // covers, because the route's own member over the grid is what answers there.
    using AllOrdersDefault =
        boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;
    const boys::FitGranularityInfo* partition = nullptr;

    for (const boys::FitGranularityInfo& row : boys::BoysFitGranularities())
    {
        if (row.granularity == AllOrdersDefault::kGranularity)
        {
            partition = &row;
        }
    }

    const bool readsGrid =
        partition != nullptr && partition->granularity == boys::FitGranularity::kUniform;

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
            // Inside the domain the row serves, and nowhere else: a row that hands its
            // fit over per order states the argument from which it does, so this is the
            // domain the row's own figures are a promise over.
            if (x < row.servesFrom || x >= row.hi)
            {
                continue;
            }

            double plain[boys::kMaxBoysOrder + 1] = {};
            double out[boys::kMaxBoysOrder + 1] = {};
            AllOrders(boys::kMaxBoysOrder, x, plain);
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
        // whatever its row reports. The default route is the other half of the same
        // statement: naming it is the default entry, so it changes nothing anywhere.
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

    // Documented: outside the intervals a route's own fits cover, naming a route
    // runs the default entry's own code, so a caller who names one and one who
    // does not agree. Which intervals those are is the class row's: a per-order
    // partition leaves the route rows' own boundaries, and a class row naming the
    // grid leaves the partition's cover, where the route's member over the table
    // answers every argument. Read from the two reports rather than written here,
    // so a class row that moves the partition moves the domain with it.
    std::size_t outside = 0;
    std::size_t changedOutside = 0;

    for (const double x : args)
    {
        bool served = readsGrid && x >= partition->lo && x < partition->hi;

        if (!readsGrid)
        {
            for (const boys::FitRouteInfo& row : routes)
            {
                served = served || (x >= row.servesFrom && x < row.hi);
            }
        }

        if (served)
        {
            continue;
        }

        ++outside;
        double plain[boys::kMaxBoysOrder + 1] = {};
        double out[boys::kMaxBoysOrder + 1] = {};
        AllOrders(boys::kMaxBoysOrder, x, plain);
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
        AllOrders(boys::kMaxBoysOrder, x, plain);
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

    // The two axes compose at run time as they do at compile time: the entry names
    // both, and a caller holds one fixed to see the other move. If one axis did not
    // reach the call, one of the counts below would be zero - which makes this
    // a check of the pair rather than of two options printed beside each other.
    //
    // What the scheme reaches on the rational route is the library's own account of
    // it: "the scheme reaches the parts of the call that route's fits do not serve"
    // (boys/boys.hpp, the three-selector overload), and where the call reads those
    // parts it reads the Chebyshev lane's own tables at the policy's scheme
    // (boys_impl.hpp, PolicyRegionAValue: "below each route's fits-first crossover
    // the value a policy answers with is the Chebyshev lane's"). Which parts those
    // are is the class row's partition: a per-order partition leaves the arguments
    // below the route's own takeover, and a class row naming the grid leaves none
    // inside the fitted domain - the grid's branch answers them from its own
    // member, one fit under either scheme. Both readings are taken from the two
    // reports rather than written here, so a class row that moves the partition
    // moves them with it.
    std::size_t routeAtClenshaw = 0;
    std::size_t routeAtHorner = 0;
    std::size_t schemeOnRational = 0;
    std::size_t schemeInsideTheFits = 0;
    std::size_t partsTheRationalFitsDoNotServe = 0;

    for (const double x : args)
    {
        // Inside the fitted domain the class reads, and inside the rational
        // route's own part of it, at this argument.
        const bool fitted = partition != nullptr && x >= partition->lo && x < partition->hi;
        bool rationalServes = readsGrid && fitted;

        if (!readsGrid)
        {
            for (const boys::FitRouteInfo& row : routes)
            {
                if (row.route == boys::FitRoute::kRationalMinimax)
                {
                    rationalServes = rationalServes || (x >= row.servesFrom && x < row.hi);
                }
            }
        }

        if (fitted && !rationalServes)
        {
            ++partsTheRationalFitsDoNotServe;
        }

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
                schemeInsideTheFits += rationalServes ? 1 : 0;
            }
        }
    }

    Require(report,
            routeAtClenshaw > 0,
            "naming the rational route changes values with the scheme held at the default");
    Require(report,
            routeAtHorner > 0,
            "naming the rational route changes values with the Horner scheme held");
    // The two counts behind the reading above, printed rather than left in the
    // assertions: the differences inside the rational route's own domain are what
    // the region-A read at the policy's scheme reaches when the class row names a
    // per-order partition, and the argument count outside it is the domain the
    // axis is live over on this build.
    std::printf("  scheme on the rational route: %zu differing cell(s), %zu of them inside the "
                "domain that route's own fits answer, over %zu swept argument(s) outside it\n",
                schemeOnRational,
                schemeInsideTheFits,
                partsTheRationalFitsDoNotServe);
    // The half the sentence states, read off the partition: where the class row
    // leaves the route parts its own fits do not serve, naming a scheme reaches
    // them. A class row naming the grid leaves none - the shape it names is the
    // one fit under either scheme - so there the axis is live on this route
    // nowhere, and the count above states that rather than asserting it away.
    Require(report,
            partsTheRationalFitsDoNotServe == 0 || schemeOnRational > 0,
            "naming a scheme on the rational route reaches the parts its own fits do not serve");

    // Documented: the two-selector overload names a route and no scheme, so the
    // call it is one with is the three-selector call that names the build's
    // default scheme - the scheme is the library's on an axis the caller leaves
    // unnamed, and the two overloads answer one call rather than two. The
    // argument is inside the rational route's served domain, where naming the
    // route reaches values rather than the default entry.
    {
        double rationalFrom = 0.0;
        double rationalHi = 0.0;

        for (const boys::FitRouteInfo& row : routes)
        {
            if (row.route == boys::FitRoute::kRationalMinimax &&
                row.region == boys::AccuracyRegion::kA)
            {
                rationalFrom = row.servesFrom;
                rationalHi = row.hi;
            }
        }

        const double x = 0.5 * (rationalFrom + rationalHi);

        std::array<double, boys::kMaxBoysOrder + 1> twoSelectors = {};
        std::array<double, boys::kMaxBoysOrder + 1> threeSelectors = {};
        boys::BoysAllOrdersWithRoute(
            boys::FitRoute::kRationalMinimax, boys::kMaxBoysOrder, x, twoSelectors.data());
        boys::BoysAllOrdersWithRoute(boys::FitRoute::kRationalMinimax,
                                     boys::kDefaultEvalScheme,
                                     boys::kMaxBoysOrder,
                                     x,
                                     threeSelectors.data());

        std::size_t differing = 0;

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const std::size_t j = static_cast<std::size_t>(n);
            differing += twoSelectors[j] == threeSelectors[j] ? 0 : 1;
        }

        Require(report,
                differing == 0,
                "naming the default scheme on the three-selector overload is the two-selector "
                "call, bit for bit");
    }

    Covered("boys::FitRoute");
    Covered("boys::FitRouteInfo");
    Covered("boys::BoysFitRoutes");
    Covered("boys::BoysAllOrdersWithRoute");
}

/// The float lane's fit routes, reached the same way: every figure this file judges
/// a row by comes from BoysFitRoutesF32's own row rather than from a number carried
/// here, so the report and the entry cannot agree with each other and both be wrong.
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

/// The float lane's policy path, which a consumer reaches as a template argument on
/// the entries themselves rather than through the run-time selector. Three readings
/// make that path an option rather than a name: a policy naming the default pair
/// (the route and the scheme this build's defaults header names, read off the
/// library's own default rather than written here) is the entry naming no policy,
/// bit for bit; naming the OTHER scheme - whichever of the two this build does not
/// default to - changes values the default scheme answers with; and each route's
/// policy answers exactly what that route's run-time selector answers, which is one
/// body reached two ways rather than two wirings that happen to agree.
///
/// The other scheme is read off the default rather than written here as Horner: a
/// build that moves the scheme moves which word the other one is, and both readings
/// above are the same two readings at either setting of the axis.
void CheckFloatPolicies(Report& report, const std::vector<Cell>& cells) {
    using ByDefault = boys::DefaultPolicyFp32;

    // The name each entry read below actually resolves to when no policy is named: its
    // own class's row, which is `ByDefault` where this build's seam carries no row for
    // the class and the row where it does. The two entries are held to this name for
    // the reason CheckEvalSchemes states: a name that happens to agree with the entry's
    // default at this revision is not the entry's default.
    using SingleDefault = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kSingle>;
    using AllOrdersDefault =
        boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>;

    // The scheme this build does not default to. The axis has two members and this
    // names the one EvalPolicy<> leaves at the other, so what the readings below
    // separate is the scheme and not the word a build compiled it at.
    constexpr boys::EvalScheme kOtherScheme =
        boys::kDefaultEvalScheme == boys::EvalScheme::kHorner ? boys::EvalScheme::kSplitClenshaw
                                                              : boys::EvalScheme::kHorner;
    using OtherScheme = boys::EvalPolicy<boys::kDefaultFitRoute, kOtherScheme>;
    using Rational = boys::EvalPolicy<boys::FitRoute::kRationalMinimax>;

    static_assert(ByDefault{}.kRoute == boys::kDefaultFitRoute &&
                      ByDefault{}.kScheme == boys::kDefaultEvalScheme &&
                      ByDefault{}.kGranularity == boys::kDefaultFitGranularity,
                  "the policy this check calls the default one names every axis the library "
                  "defaults, so none of them is a second value written here");

    // Naming the other scheme moves the scheme and nothing else, which is what
    // makes the reading below a reading about the scheme rather than about a
    // pair of values that differ in two places.
    static_assert(OtherScheme{}.kRoute == ByDefault{}.kRoute &&
                      OtherScheme{}.kGranularity == ByDefault{}.kGranularity &&
                      OtherScheme{}.kScheme != ByDefault{}.kScheme,
                  "the policy this check calls the other-scheme one differs from the default in "
                  "its scheme alone, at either setting of the axis");

    std::size_t sameAsDefault = 0;
    std::size_t sameAsSelector[2] = {0, 0};
    std::size_t changedByScheme = 0;
    std::size_t changedByRoute = 0;
    bool allFinite = true;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const float byDefault = boys::BoysSingleF32(cell.n, xf);
        const float byItsDefault = boys::BoysSingleF32<SingleDefault>(cell.n, xf);
        const float otherScheme = boys::BoysSingleF32<OtherScheme>(cell.n, xf);
        const float rational = boys::BoysSingleF32<Rational>(cell.n, xf);
        const float bySelector[2] = {
            boys::BoysSingleF32WithRoute(boys::FitRoute::kChebyshev, cell.n, xf),
            boys::BoysSingleF32WithRoute(boys::FitRoute::kRationalMinimax, cell.n, xf)};

        sameAsDefault += (byItsDefault == byDefault) ? 1 : 0;
        sameAsSelector[0] += (byItsDefault == bySelector[0]) ? 1 : 0;
        sameAsSelector[1] += (rational == bySelector[1]) ? 1 : 0;
        changedByScheme += (otherScheme != byItsDefault) ? 1 : 0;
        changedByRoute += (rational != byItsDefault) ? 1 : 0;
        allFinite = allFinite && std::isfinite(otherScheme) && std::isfinite(rational);
    }

    Require(report,
            allFinite,
            "a float policy this build stores answers a finite value at every cell of the "
            "reference grid");
    Require(report,
            cells.size() > 0 && sameAsDefault == cells.size(),
            "naming the default route and scheme is the entry naming no policy, bit for bit, at "
            "every cell of the reference grid");
    Require(report,
            sameAsSelector[0] == cells.size() && sameAsSelector[1] == cells.size(),
            "each route's policy answers what that route's run-time selector answers, bit for "
            "bit, so the two ways in read one body");
    Require(report,
            changedByScheme > 0,
            "naming the scheme this build does not default to changes values the default "
            "scheme answers with, so the unnamed call is not reading that scheme's tables");
    Require(report,
            changedByRoute > 0,
            "naming the rational route on the float lane changes values the default route "
            "answers with somewhere on the reference grid");

    // The same pair on the all-orders shape. Its seeds are not the single entry's -
    // region A's is the double lane's fit at the policy's route and scheme, region
    // B's is this lane's - so the reading here is reachability, not identity: the
    // pair reaches this entry too.
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
        std::array<float, boys::kMaxBoysOrder + 1> otherScheme = {};
        std::array<float, boys::kMaxBoysOrder + 1> rational = {};

        boys::BoysAllOrdersF32(boys::kMaxBoysOrder, xf, plain.data());
        boys::BoysAllOrdersF32<AllOrdersDefault>(boys::kMaxBoysOrder, xf, named.data());
        boys::BoysAllOrdersF32<OtherScheme>(boys::kMaxBoysOrder, xf, otherScheme.data());
        boys::BoysAllOrdersF32<Rational>(boys::kMaxBoysOrder, xf, rational.data());

        ++batchArgs;
        batchSameAsDefault +=
            std::memcmp(plain.data(), named.data(), sizeof(plain)) == 0 ? 1 : 0;
        batchChangedByScheme +=
            std::memcmp(plain.data(), otherScheme.data(), sizeof(plain)) == 0 ? 0 : 1;
        batchChangedByRoute +=
            std::memcmp(plain.data(), rational.data(), sizeof(plain)) == 0 ? 0 : 1;

        for (std::size_t k = 0; k < plain.size(); ++k)
        {
            batchFinite =
                batchFinite && std::isfinite(otherScheme[k]) && std::isfinite(rational[k]);
        }
    }

    Require(report,
            batchFinite,
            "a policy this build stores answers a finite value at every order of the batch entry");
    Require(report,
            batchArgs > 0 && batchSameAsDefault == batchArgs,
            "naming the default pair on the batch entry is the call naming no policy, bit for "
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

    // The four shapes at the double lane's own default policy.
    {
        Rule& rule = NewRule("BoysAllN (grid sweep, four shapes)");

        std::fill(planes.begin(), planes.end(), kUnwritten);
        AllN(boys::kMaxBoysOrder, args.data(), planes.data(), count, nullptr);

        std::fill(again.begin(), again.end(), kUnwritten);
        AllN(boys::kMaxBoysOrder, args.data(), again.data(), count, workspace.data());
        Require(report,
                std::equal(planes.begin(), planes.end(), again.begin()),
                "the caller's workspace returns the same planes as the internal one");

        std::fill(again.begin(), again.end(), kUnwritten);
        AllNSorted(boys::kMaxBoysOrder, args.data(), again.data(), count);
        Require(report,
                std::equal(planes.begin(), planes.end(), again.begin()),
                "the sorted-argument overload returns the same planes as the sorting one");

        // Documented: arguments may arrive in any order; the entry classifies
        // and groups them and returns the results in the caller's order, so
        // plane k, position i holds F_k(reversed[i]).
        std::fill(again.begin(), again.end(), kUnwritten);
        AllN(boys::kMaxBoysOrder, reversed.data(), again.data(), count, nullptr);

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

                Judge(rule, planes[index], sorted->value, BatchBound(), n, args[i]);
                Judge(rule, again[index], shuffled->value, BatchBound(), n, reversed[i]);
            }
        }

        Require(report, missing == 0, "the grid carries every cell this rule looks up");

        // Documented: count may be 0, and then nothing is written.
        double untouched[4] = {kUnwritten, kUnwritten, kUnwritten, kUnwritten};
        AllN(boys::kMaxBoysOrder, args.data(), untouched, 0, nullptr);
        AllNSorted(boys::kMaxBoysOrder, args.data(), untouched, 0);
        Require(report,
                untouched[0] == kUnwritten,
                "BoysAllN writes nothing for count = 0, in both shapes");
        Covered("boys::BoysAllN<>");
        Covered("boys::BoysAllN<> (BoysSortedArgs)");
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

    // The name each entry below resolves to when no policy is named: its own
    // class's row, which is the five where this build's seam carries no row for the
    // class and the row where it does. Both entries document the identity each
    // check reads as one with a body of the all-orders shape - "the per-argument
    // all-orders body run at that argument's own top order" for the order-array
    // entry, and "the all-orders body this entry calls" for the float all-N one -
    // and that body is read at this entry's own name: two classes whose rows differ
    // answer two arithmetics, and reading one class's row on the body would measure
    // the seam's choice reaching two shapes rather than the shape reaching a body.
    using AllNAtOrdersDefault =
        boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>;
    using AllNF32Default = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>;

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

    // The per-argument entry, and the same entry at one common top order, at the
    // double lane's own default policy.
    {
        Rule& ragged = NewRule("BoysAllNAtOrders (ragged top order per argument)");
        Rule& common = NewRule("BoysAllNAtOrders (one top order for the batch)");

        std::fill(planes.begin(), planes.end(), kUnwritten);
        AllNAtOrders(raggeds.data(), args.data(), planes.data(), count);

        std::size_t missing = 0;
        std::size_t above = 0;
        std::size_t kept = 0;
        std::size_t differs = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            // Documented: the column stops at the argument's own top order and
            // is, bit for bit, the per-argument all-orders entry's value at that
            // argument and that top order - that entry at this entry's own
            // class row, which is the name the call above resolves to.
            boys::BoysAllOrders<AllNAtOrdersDefault>(raggeds[i], args[i], row.data());

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

                Judge(ragged, planes[index], cell->value, BatchBound(), n, args[i]);
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
        AllNAtOrders(commons.data(), args.data(), planes.data(), count);

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
                          BatchBound(),
                          n,
                          args[i]);
                }
            }
        }

        // Documented: count may be 0, and then nothing is written.
        double untouched[2] = {kUnwritten, kUnwritten};
        AllNAtOrders(raggeds.data(), args.data(), untouched, 0);
        Require(report,
                untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                "BoysAllNAtOrders writes nothing for count = 0");

        Covered("boys::BoysAllNAtOrders<>");
    }

    // The float all-N batch at the float lane's own default policy.
    {
        Rule& rule = NewRule("BoysAllNF32 (the float all-N batch)");
        const double bound =
            FloatFigure<boys::Precision::kFp32, boys::Shape::kAllN>() + kOracleBound;

        std::fill(planesF.begin(), planesF.end(), static_cast<float>(kUnwritten));
        AllNF32(nmax, argsF.data(), planesF.data(), count);

        std::size_t differs = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            // The float lane rounds its argument to float before evaluating, so
            // the reference is the certified double entry at that rounded
            // argument, in the same shape.
            boys::BoysAllOrders<>(nmax, static_cast<double>(argsF[i]), row.data());
            // Documented: each column of this entry is that entry's value at the
            // same (nmax, x[i]) bit for bit - the float all-orders entry at this
            // entry's own class row, which is the name the call above resolves to.
            boys::BoysAllOrdersF32<AllNF32Default>(nmax, argsF[i], rowF.data());

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
        AllNF32(nmax, argsF.data(), untouched, 0);
        Require(report,
                untouched[0] == -1.0f && untouched[1] == -1.0f,
                "BoysAllNF32 writes nothing for count = 0");

        Covered("boys::BoysAllNF32<>");
    }
}

/// The fp32 lane. Its argument is a float, so the reference is the certified
/// double lane at that same float argument, and the composed bound is the
/// lane's own figure plus the reference lane's.
void CheckFloatLane(const std::vector<Cell>& cells) {
    // The float lane at its own default policy, judged against the double lane at
    // the same rounded argument.
    {
        Rule& single = NewRule("BoysSingleF32 (vs the double lane at float(x))");
        Rule& batch = NewRule("BoysAllOrdersF32 (vs the double lane at float(x))");
        // Each rule's own figure: the two entries are two classes, and this build
        // may name them two forms.
        const double singleBound =
            FloatFigure<boys::Precision::kFp32, boys::Shape::kSingle>() + kOracleBound;
        const double batchBound =
            FloatFigure<boys::Precision::kFp32, boys::Shape::kAllOrders>() + kOracleBound;

        for (const Cell& cell : cells)
        {
            const float xf = static_cast<float>(cell.x);
            const double oracle = boys::BoysSingle<>(cell.n, static_cast<double>(xf));
            Judge(single,
                  static_cast<double>(SingleF32(cell.n, xf)),
                  oracle,
                  singleBound,
                  cell.n,
                  cell.x);
        }

        for (const double x : DistinctArgs(cells))
        {
            const float xf = static_cast<float>(x);
            float out[boys::kMaxBoysOrder + 1] = {};
            AllOrdersF32(boys::kMaxBoysOrder, xf, out);

            for (const Cell& cell : cells)
            {
                if (cell.x == x)
                {
                    const double oracle = boys::BoysSingle<>(cell.n, static_cast<double>(xf));
                    Judge(batch,
                          static_cast<double>(out[cell.n]),
                          oracle,
                          batchBound,
                          cell.n,
                          cell.x);
                }
            }
        }

        Covered("boys::BoysSingleF32<>");
        Covered("boys::BoysAllOrdersF32<>");
    }
}

// The two half lanes below name entries the BoysFp16 seam declares, so both checks
// are compiled with the entries they measure. A consumer that builds this tree with
// the seam closed has no such entry to name, and main prints those lanes as ones
// this build does not carry.
#if BoysFp16
/// The fp16 and bf16 I/O lanes. Both round their argument to the 16-bit format
/// before evaluating, so the reference is the certified double lane at that rounded
/// argument. The bound is the lane's own figure, claimed only where the value
/// exceeds it; cells past that ceiling are counted rather than judged.
void CheckHalfIo(const std::vector<Cell>& cells) {
    // Both I/O lanes at their own default policy, judged where the reference
    // exceeds the lane's bound.
    {
        Rule& f16Single = NewRule("BoysSingleF16 (cells above its bound)");
        Rule& bf16Single = NewRule("BoysSingleBf16 (cells above its bound)");
        Rule& f16Batch = NewRule("BoysAllOrdersF16 (cells above its bound)");
        Rule& bf16Batch = NewRule("BoysAllOrdersBf16 (cells above its bound)");
        std::size_t f16Past = 0;
        std::size_t bf16Past = 0;

        for (const Cell& cell : cells)
        {
            const float xf = static_cast<float>(cell.x);
            const boys::F16 h = boys::F16(xf);
            const boys::Bf16 b = boys::Bf16(xf);
            const double oracleF16 = boys::BoysSingle<>(cell.n, static_cast<double>(h));
            const double oracleBf16 = boys::BoysSingle<>(cell.n, static_cast<double>(b));
            const double gotF16 = static_cast<double>(SingleF16(cell.n, h));
            const double gotBf16 = static_cast<double>(SingleBf16(cell.n, b));
            const double boundF16 = F16IoBound(gotF16);
            const double boundBf16 = Bf16IoBound(gotBf16);

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
            AllOrdersF16(boys::kMaxBoysOrder, h, outF16);
            AllOrdersBf16(boys::kMaxBoysOrder, b, outBf16);

            for (const Cell& cell : cells)
            {
                if (cell.x != x)
                {
                    continue;
                }

                const double oracleF16 = boys::BoysSingle<>(cell.n, static_cast<double>(h));
                const double oracleBf16 = boys::BoysSingle<>(cell.n, static_cast<double>(b));
                const double gotF16 = static_cast<double>(outF16[cell.n]);
                const double gotBf16 = static_cast<double>(outBf16[cell.n]);
                const double boundF16 = F16IoBound(gotF16);
                const double boundBf16 = Bf16IoBound(gotBf16);

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

        Covered("boys::BoysSingleF16<>");
        Covered("boys::BoysAllOrdersF16<>");
        Covered("boys::BoysSingleBf16<>");
        Covered("boys::BoysAllOrdersBf16<>");
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
#endif // BoysFp16

/// The region-A transform, in every arithmetic mode the enumeration carries,
/// over both bands. The entry is a lane of its own: its bound is the mode's, and
/// its domain is region A. Every mode is instantiated in the library's own
/// translation unit, so these calls link against those instantiations rather
/// than compiling second copies.
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

    // Every enumerator has exactly one report row, and every row is an enumerator: a
    // mode a caller can name but cannot ask about is the gap this check exists for,
    // and so is a row for a mode that is not in the enumeration.
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

        // The report is the only place a caller learns which rows a card is held to,
        // so the class has to be reported rather than implied.
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

            const std::string name =
                std::string("BoysRegionAProduct<") + mode.name + (lower ? ", kA1>" : ", kA2>");
            Rule& rule = NewRule(name);
            std::vector<double> out(args.size() * (boys::kMaxBoysOrder + 1));
            std::fill(out.begin(), out.end(), kUnwritten);
            RegionAProduct(
                mode.mode, band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());

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
                              ProductBound(report, mode.mode),
                              n,
                              args[i]);
                    }
                }
            }

            // Documented: count may be 0, and then nothing is written.
            double untouched[2] = {kUnwritten, kUnwritten};
            const double one[1] = {args.empty() ? 0.0 : args[0]};
            RegionAProduct(mode.mode, band, 0, one, untouched, 0);
            Require(report,
                    untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                    "BoysRegionAProduct writes nothing for count = 0");
        }

        Covered("boys::ProductMode");
        Covered("boys::BoysRegionAProduct (every mode, both bands)");
    }
}

// The packed half type's surface. This file includes only <boys/boys.hpp>, so the
// type arrives with the seam: a closed-seam consumer has no boys::Half2, boys::F16
// or boys::Bf16 to name, and main prints the surface as one this build lacks.
#if BoysFp16
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
#endif // BoysFp16

/// The evaluation schemes a consumer can ask about and ask for. What a consumer
/// reads here is the whole of the option: which arithmetic is in force, what each
/// scheme promises on each stored fit, and that the entries answer under the scheme
/// that was named - which a reading of the enumeration alone cannot show.
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

    // The two axes compose into one selection: a policy names the fit route and the
    // scheme together, and every templated entry takes that policy. The default
    // policy is the Chebyshev route at the scheme this build's defaults header
    // names, read from the constants rather than written here.
    using DefaultPolicy = boys::EvalPolicy<>;

    // The name a call that names no policy actually resolves to, per class: its own
    // class's row, which is the five above where this build's seam carries no row for
    // the class and the row where it does (boys/boys.hpp, DefaultPolicy: "the policy a
    // class compiles when its call site names no policy: the name an entry's policy
    // parameter defaults to"). The two entries read below are held to this name rather
    // than to `DefaultPolicy`, because `DefaultPolicy` is the five and agrees with the
    // class's row only while the seam carries none: a name that happens to agree with
    // the entry's default at this revision is not the entry's default.
    using SingleDefault = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>;
    using AllOrdersDefault =
        boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;

    using SplitClenshawPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                                 boys::EvalScheme::kSplitClenshaw>;
    using HornerPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;

    // The scheme this build does not default to, named the same way the float lane's
    // section names it: which of the two words is the other one is the build's, and
    // the reading below is the same reading at either setting of the axis.
    constexpr boys::EvalScheme kOtherScheme =
        boys::kDefaultEvalScheme == boys::EvalScheme::kHorner ? boys::EvalScheme::kSplitClenshaw
                                                              : boys::EvalScheme::kHorner;
    using OtherSchemePolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev, kOtherScheme>;

    // The policy's fields read back what it was named with, so a consumer can ask a
    // policy which pair it carries rather than reading its type. What is pinned about
    // the default one is that it names the library's own defaults - read from the
    // constants rather than written here.
    constexpr DefaultPolicy kDefaultPolicy{};
    static_assert(kDefaultPolicy.kRoute == boys::kDefaultFitRoute &&
                      kDefaultPolicy.kScheme == boys::kDefaultEvalScheme &&
                      kDefaultPolicy.kGranularity == boys::kDefaultFitGranularity &&
                      kDefaultPolicy.kBudget == boys::BoysBudget::kFloat,
                  "the policy this check calls the default one names every axis the library "
                  "defaults, at the float lane's budget");
    static_assert(HornerPolicy{}.kScheme == boys::EvalScheme::kHorner &&
                      SplitClenshawPolicy{}.kScheme == boys::EvalScheme::kSplitClenshaw,
                  "a policy carries the scheme it was named with");

    // The two schemes sum one polynomial, so the entries that name them differ by no
    // more than their own bounds and either entry is inside the other's error plus
    // twice that. A scheme that reached the wrong fit would be wrong by orders of
    // magnitude rather than by a bound, which is what this separates.
    std::size_t schemePartsFromOther = 0;

    for (const Cell& cell : cells)
    {
        const double byDefault = boys::BoysSingle<>(cell.n, cell.x);
        const double byDefaultNamed = boys::BoysSingle<SingleDefault>(cell.n, cell.x);
        Require(report,
                byDefault == byDefaultNamed,
                "a call naming no policy is the pair the library's defaults name, bit for bit");

        std::array<double, 33> un = {};
        std::array<double, 33> named = {};
        boys::BoysAllOrders<>(cell.n, cell.x, un.data());
        boys::BoysAllOrders<AllOrdersDefault>(cell.n, cell.x, named.data());
        Require(report,
                std::memcmp(un.data(), named.data(), sizeof(un)) == 0,
                "a batch call naming no policy is the pair the library's defaults name, bit "
                "for bit");

        // The unnamed call reads the scheme this build's defaults header names, and
        // the reading that says so is the other scheme's: the two sum one fit, so they
        // agree to within their own bounds and part somewhere, and the part is what
        // makes the unnamed call's scheme a reading rather than the only one there is.
        // Which word the other scheme is belongs to the build, so this names the
        // member the defaults header leaves at the other rather than the split
        // Clenshaw recurrence; a build whose default is that recurrence reads the same
        // three lines below about Horner's rule.
        const double byOtherScheme = boys::BoysSingle<OtherSchemePolicy>(cell.n, cell.x);
        schemePartsFromOther += (byDefault != byOtherScheme) ? 1 : 0;
        Require(report, std::isfinite(byOtherScheme), "the other scheme answers a finite value");

        std::array<double, 33> otherSchemeOut = {};
        boys::BoysAllOrders<OtherSchemePolicy>(cell.n, cell.x, otherSchemeOut.data());
        Require(report,
                otherSchemeOut[static_cast<std::size_t>(cell.n)] == byOtherScheme,
                "the batch entry answers the other scheme the same value as the single entry");

        Require(report,
                std::abs(byOtherScheme - cell.value) <=
                    std::abs(byDefault - cell.value) + 2.0 * worstSchemeBound,
                "the other scheme's entry is inside the default entry's error plus the "
                "two schemes' own bounds");
    }

    Require(report,
            schemePartsFromOther > 0,
            "naming the scheme this build does not default to changes values the unnamed call "
            "answers with somewhere on the reference grid, so the unnamed call reads the "
            "defaults header's scheme and not the other one's tables");

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

// The interval-granularity axis, reached the way a consumer reaches it: by naming
// the partition on the policy and calling the entries.
//
// The member is a second partition of the fitted domain - region A's pieces and
// region B's seed - so the things a consumer has to be able to read from it are
// that naming it changes the values over the fitted domain at both regions the two
// tables are cut in, so that the member is a partition and not the default's tables
// under another name; that it changes nothing at or above the fitted domain's end,
// the axis being a selection between two stored tables and not a second arithmetic
// path; that naming the default member is the default call bit for bit, so the
// default is a member of the axis; and that every value it returns is inside the
// lane's published bound, so the member does not widen the contract a caller relies
// on. The counts the partition costs are the generated header's own static_assert
// and the gate's narrow rows; what is asserted here is that the policy carries the
// partition it was named with.
//
// The member named is the one this build's defaults header leaves at the other, so
// the readings hold at either setting: they separate the two partitions, they do not
// pin which of the two a given build compiles as its default.
void CheckGranularityLane(Report& report, const std::vector<Cell>& cells) {
    // The axis' default member, named: the library's own policy, so the reading below
    // is that the member a call reaches by naming nothing is the member the default names.
    using DefaultPolicy = boys::EvalPolicy<>;

    // The other member, named: every axis the default policy carries, with the
    // partition named to the value the default does not name. Which of the two words
    // that is belongs to the build - the shipped partition is the default of a build
    // whose defaults header names it - so what the readings below separate is the two
    // partitions rather than which one a given build calls the default.
    constexpr boys::FitGranularity kOtherGranularity =
        boys::kDefaultFitGranularity == boys::FitGranularity::kCoarsest
            ? boys::FitGranularity::kNarrow
            : boys::FitGranularity::kCoarsest;
    using OtherPolicy = boys::EvalPolicy<boys::kDefaultFitRoute,
                                         boys::kDefaultEvalScheme,
                                         boys::BoysBudget::kFloat,
                                         boys::kDefaultPackAxis,
                                         kOtherGranularity>;

    static_assert(DefaultPolicy{}.kGranularity == boys::kDefaultFitGranularity &&
                      OtherPolicy{}.kGranularity == kOtherGranularity &&
                      kOtherGranularity != boys::kDefaultFitGranularity,
                  "a policy carries the partition it was named with: the one this check calls "
                  "the default carries the library's default, and the other carries a member "
                  "that is not it");
    static_assert(OtherPolicy{}.kRoute == DefaultPolicy{}.kRoute &&
                      OtherPolicy{}.kScheme == DefaultPolicy{}.kScheme &&
                      OtherPolicy{}.kBudget == DefaultPolicy{}.kBudget &&
                      OtherPolicy{}.kPack == DefaultPolicy{}.kPack &&
                      OtherPolicy{}.kGranularity != DefaultPolicy{}.kGranularity,
                  "the policy this check calls the other-partition one differs from the default "
                  "in its partition alone, so what the readings below separate is the partition");
    static_assert(!std::is_same_v<DefaultPolicy::Fit, OtherPolicy::Fit>,
                  "the two partitions are different fits: neither is the other under a second "
                  "name");

    Require(report,
            std::strcmp(boys::GranularityName(boys::FitGranularity::kCoarsest),
                        boys::GranularityName(boys::FitGranularity::kNarrow)) != 0,
            "the two partitions are reported under different names rather than one blank");

    Rule& rule = NewRule("granularity: both partitions through the entries");

    // x1, where the fitted domain ends and the asymptotic path takes over, as the
    // umbrella header publishes it - region B runs x0 <= x < x1 and region C is
    // x >= x1. The public surface names no constant for it, so it is transcribed here.
    constexpr double kFittedDomainEnd = 28.98933773882074;

    std::size_t inA = 0;
    std::size_t changedInA = 0;
    std::size_t inB = 0;
    std::size_t changedInB = 0;
    std::size_t aboveDomain = 0;
    std::size_t changedAboveDomain = 0;
    std::size_t defaultDiffering = 0;

    for (const Cell& cell : cells)
    {
        const double byDefault = boys::BoysSingle<>(cell.n, cell.x);
        // The default member, named, and the other member, named: the call that names
        // neither is the default one of them. The default is named by its own class
        // rather than by `DefaultPolicy`, which is the five and is the class's row only
        // where this build's seam carries none for it - the same reason the entries in
        // CheckEvalSchemes above are held to their class's name.
        const double byDefaultNamed =
            boys::BoysSingle<boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>>(
                cell.n, cell.x);
        const double other = boys::BoysSingle<OtherPolicy>(cell.n, cell.x);

        if (cell.x < boys::kRegionAEnd)
        {
            ++inA;

            if (other != byDefault)
            {
                ++changedInA;
            }
        } else if (cell.x < kFittedDomainEnd)
        {
            ++inB;

            if (other != byDefault)
            {
                ++changedInB;
            }
        } else
        {
            ++aboveDomain;

            if (other != byDefault)
            {
                ++changedAboveDomain;
            }
        }

        if (byDefaultNamed != byDefault)
        {
            ++defaultDiffering;
        }

        Judge(rule, other, cell.value, SingleBound(cell.x), cell.n, cell.x);
        Judge(rule, byDefaultNamed, cell.value, SingleBound(cell.x), cell.n, cell.x);
    }

    // Each rule says which cells it measured rather than passing on an empty sweep:
    // a grid that carries no argument of a region would leave the reading vacuous.
    Require(report,
            inA > 0 && changedInA > 0,
            "naming the partition this build does not default to changes region A's values "
            "against the default call: the member cuts region A's pieces as well as region B's "
            "seed, and the change is visible through the entry");
    Require(report,
            inB > 0 && changedInB > 0,
            "naming the partition this build does not default to changes region B's values "
            "against the default call: the member is a partition and not the default seed under "
            "another name");
    Require(report,
            aboveDomain > 0 && changedAboveDomain == 0,
            "naming the partition this build does not default to changes nothing at or above the "
            "fitted domain's end: above it the entry reads the asymptotic path, which no "
            "partition of the stored fits is part of");
    Require(report,
            defaultDiffering == 0,
            "naming the default partition is the default call bit for bit, so the default is "
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

// --- the accuracy a combination carries, and the tolerance a caller names ----

/// The three entries a consumer picks an option with: the bound a combination
/// carries, the figure it was measured to deliver, and the entry that answers
/// whether either is at the tolerance the caller names.
///
/// The first two are different numbers and the library says which is which in the
/// `reading` field of each answer, so this check reads them apart rather than
/// comparing them. The third is the question a caller with a target has - *is this
/// at what I need* - and this check reads it the way such a caller does, at a
/// request the run itself derives.
///
/// The whole cross is walked rather than a chosen pair, so the counts below are of
/// every combination this build's tables name: the query's figures are required to
/// be the accessors' figures, and its verdict the comparison of the request with them.
void CheckOptionAccuracy(Report& report) {
    const auto laneName = [](boys::Precision precision) {
        for (const boys::LaneContractInfo& row : boys::BoysLaneContracts())
        {
            if (row.precision == precision)
            {
                return std::string(row.name);
            }
        }

        return std::string("no lane of this library");
    };

    const auto routeName = [](boys::FitRoute route) {
        for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
        {
            if (row.route == route)
            {
                return std::string(row.name);
            }
        }

        return std::string("no fit route of this library");
    };

    const auto axesName = [&laneName, &routeName](boys::Precision precision,
                                                  boys::FitRoute route,
                                                  boys::EvalScheme scheme,
                                                  boys::PackAxis axis,
                                                  boys::FitGranularity granularity) {
        char text[256];
        std::snprintf(text,
                      sizeof(text),
                      "%s, %s, %s, %s, %s",
                      laneName(precision).c_str(),
                      routeName(route).c_str(),
                      boys::EvalSchemeName(scheme),
                      boys::PackAxisName(axis),
                      boys::GranularityName(granularity));
        return std::string(text);
    };

    // A combination this build serves, taken as the first the cross names so
    // that a lane's own default policy is the one the reader is shown.
    bool haveServed = false;
    boys::Precision servedPrecision = boys::Precision::kFp64;
    boys::FitRoute servedRoute = boys::FitRoute::kChebyshev;
    boys::EvalScheme servedScheme = boys::EvalScheme::kSplitClenshaw;
    boys::PackAxis servedAxis = boys::PackAxis::kArguments;
    boys::FitGranularity servedGranularity = boys::FitGranularity::kCoarsest;

    // A combination this build refuses, and the reason it gives: the first the cross
    // refuses, so the example moves with the tables rather than being hard-coded.
    bool haveRefused = false;
    boys::Precision refusedPrecision = boys::Precision::kFp64;
    boys::FitRoute refusedRoute = boys::FitRoute::kChebyshev;
    boys::EvalScheme refusedScheme = boys::EvalScheme::kSplitClenshaw;
    boys::PackAxis refusedAxis = boys::PackAxis::kArguments;
    boys::FitGranularity refusedGranularity = boys::FitGranularity::kCoarsest;

    std::size_t carried = 0;
    std::size_t refused = 0;
    std::size_t answeredOutside = 0;
    std::size_t answeredOnTheMeasurement = 0;
    std::size_t deliveredAbsent = 0;
    std::size_t disagreements = 0;

    for (const boys::LaneContractInfo& lane : boys::BoysLaneContracts())
    {
        for (const boys::FitRouteInfo& route : boys::BoysFitRoutes())
        {
            for (const boys::EvalSchemeInfo& scheme : boys::BoysEvalSchemes())
            {
                for (const boys::FitGranularityInfo& partition : boys::BoysFitGranularities())
                {
                    for (const boys::PackAxisInfo& axis : boys::BoysPackAxes())
                    {
                        const boys::AccuracyFigure guaranteed = boys::BoysAccuracyGuaranteed(
                            lane.precision, route.route, scheme.scheme, axis.axis,
                            partition.granularity);
                        const boys::AccuracyFigure delivered = boys::BoysAccuracyDelivered(
                            lane.precision, route.route, scheme.scheme, axis.axis,
                            partition.granularity);
                        const boys::CombinationCoverage asked = boys::QueryCombination(
                            lane.precision, route.route, scheme.scheme, axis.axis,
                            partition.granularity, guaranteed.value);

                        if (!guaranteed.available)
                        {
                            ++refused;

                            if (!haveRefused)
                            {
                                haveRefused = true;
                                refusedPrecision = lane.precision;
                                refusedRoute = route.route;
                                refusedScheme = scheme.scheme;
                                refusedAxis = axis.axis;
                                refusedGranularity = partition.granularity;
                            }

                            disagreements +=
                                asked.verdict != boys::ToleranceVerdict::kNotCarried ||
                                        asked.bound != 0.0 || asked.delivered != 0.0 ||
                                        asked.deliveredKnown || asked.reason[0] == '\0' ||
                                        guaranteed.value != 0.0 || delivered.value != 0.0
                                    ? 1
                                    : 0;

                            continue;
                        }

                        ++carried;

                        if (!haveServed)
                        {
                            haveServed = true;
                            servedPrecision = lane.precision;
                            servedRoute = route.route;
                            servedScheme = scheme.scheme;
                            servedAxis = axis.axis;
                            servedGranularity = partition.granularity;
                        }

                        // The query is the comparison of the request with the two
                        // figures, and the figures are the accessors': both are
                        // required on every row, not only the row this check prints.
                        if (asked.bound != guaranteed.value ||
                            asked.deliveredKnown != delivered.available ||
                            (delivered.available && asked.delivered != delivered.value) ||
                            asked.requested != guaranteed.value ||
                            asked.verdict != boys::ToleranceVerdict::kGuaranteedInside ||
                            guaranteed.reading != boys::AccuracyReading::kGuaranteed ||
                            delivered.reading != boys::AccuracyReading::kDelivered)
                        {
                            ++disagreements;

                            continue;
                        }

                        const boys::CombinationCoverage half = boys::QueryCombination(
                            lane.precision, route.route, scheme.scheme, axis.axis,
                            partition.granularity, guaranteed.value * 0.5);

                        if (half.verdict == boys::ToleranceVerdict::kGuaranteedInside)
                        {
                            ++disagreements;
                        } else if (half.verdict == boys::ToleranceVerdict::kDeliveredInside)
                        {
                            ++answeredOnTheMeasurement;
                        } else
                        {
                            ++answeredOutside;
                        }

                        deliveredAbsent += delivered.available ? 0 : 1;
                    }
                }
            }
        }
    }

    // The cross refuses combinations, because the uniform partition is served at one
    // route, one packing axis and its other cells are refused where they
    // are named: the first refusal the walk reaches is the example printed below. The
    // fallback is for a revision that serves the whole space, and what it names then
    // is a value outside the enumerations, which names no combination at all.
    if (!haveRefused)
    {
        refusedPrecision = boys::Precision::kFp32;
        refusedRoute = static_cast<boys::FitRoute>(97);
        haveRefused = true;
    }

    Require(report, haveServed, "some combination of this build is served");
    Require(report, haveRefused, "nothing reached the refusal path");
    Require(report, disagreements == 0,
            "the tolerance query answers the two accessors' figures and their comparison on every "
            "combination of the cross");

    // A caller with a target, read at requests on one combination: the numbers each
    // answer was made on are printed beside it, so the "yes" and the "no" here are the
    // caller's request against the library's own figures rather than a claim about
    // them. The requests run from above the bound to below both figures, so every
    // verdict but a refusal is reached; the request at the measured figure is made
    // only where that figure is the tighter of the two, the case a single number
    // could not have answered.
    const boys::AccuracyFigure servedBound = boys::BoysAccuracyGuaranteed(
        servedPrecision, servedRoute, servedScheme, servedAxis, servedGranularity);
    const boys::AccuracyFigure servedDelivered = boys::BoysAccuracyDelivered(
        servedPrecision, servedRoute, servedScheme, servedAxis, servedGranularity);
    const double smaller = std::min(servedBound.value,
                                    servedDelivered.available ? servedDelivered.value
                                                              : servedBound.value);

    std::printf("\n  a combination's accuracy, read from <boys/boys.hpp> alone: %s\n",
                axesName(servedPrecision, servedRoute, servedScheme, servedAxis, servedGranularity)
                    .c_str());
    std::printf("    the bound the lane documents, to rely on: %.6g, from %s (%s)\n",
                servedBound.value,
                servedBound.source,
                servedBound.reading == boys::AccuracyReading::kGuaranteed ? "the guarantee"
                                                                         : "not the guarantee");

    if (servedDelivered.available)
    {
        std::printf("    the figure it was measured to deliver, to rank options by: %.6g, from %s "
                    "(%s)\n",
                    servedDelivered.value,
                    servedDelivered.source,
                    servedDelivered.reading == boys::AccuracyReading::kDelivered
                        ? "a measurement"
                        : "not a measurement");
    } else
    {
        std::printf("    the figure it was measured to deliver: this build holds none (%s)\n",
                    servedDelivered.reason);
    }

    std::vector<double> requests{servedBound.value * 2.0, servedBound.value, smaller * 0.5};
    std::vector<const char*> requestNames{
        "twice the bound", "the bound itself", "half of the smaller of the two figures"};

    if (servedDelivered.available && servedDelivered.value < servedBound.value)
    {
        requests.insert(requests.begin() + 2, servedDelivered.value);
        requestNames.insert(requestNames.begin() + 2, "the figure it was measured to deliver");
    }

    for (std::size_t i = 0; i < requests.size(); ++i)
    {
        const boys::CombinationCoverage asked = boys::QueryCombination(
            servedPrecision, servedRoute, servedScheme, servedAxis, servedGranularity,
            requests[i]);
        const char* const verdict = asked.verdict == boys::ToleranceVerdict::kGuaranteedInside
                                        ? "yes, at the bound"
                                    : asked.verdict == boys::ToleranceVerdict::kDeliveredInside
                                        ? "yes, at the measurement - not at the bound"
                                    : asked.verdict == boys::ToleranceVerdict::kOutside
                                        ? "no"
                                        : "no verdict (refused)";

        std::printf("    asked for %.6g (%s): %s [bound %.6g", asked.requested, requestNames[i],
                    verdict, asked.bound);

        if (asked.deliveredKnown)
        {
            std::printf(", measured to deliver %.6g]\n", asked.delivered);
        } else
        {
            std::printf(", no figure measured]\n");
        }
    }

    const boys::CombinationCoverage above =
        boys::QueryCombination(servedPrecision, servedRoute, servedScheme, servedAxis,
                               servedGranularity, servedBound.value * 2.0);
    const boys::CombinationCoverage below = boys::QueryCombination(
        servedPrecision, servedRoute, servedScheme, servedAxis, servedGranularity,
        smaller * 0.5);

    Require(report, above.verdict == boys::ToleranceVerdict::kGuaranteedInside && above.bound != 0.0,
            "a request the bound is inside answers yes");
    Require(report, below.verdict == boys::ToleranceVerdict::kOutside && below.bound != 0.0 &&
                        below.deliveredKnown == servedDelivered.available,
            "a request below both figures answers no, with the figures still carried");

    // A combination the library refuses: no figure, no verdict, and the reason
    // is the library's own sentence rather than a second vocabulary.
    const boys::CombinationCoverage refusedAsk =
        boys::QueryCombination(refusedPrecision, refusedRoute, refusedScheme, refusedAxis,
                               refusedGranularity, 1e-12);

    std::printf("  a combination this build refuses, asked the same question: %s\n",
                axesName(refusedPrecision, refusedRoute, refusedScheme, refusedAxis,
                         refusedGranularity)
                    .c_str());
    std::printf("    verdict %s, bound %g, delivered %s, and the library's own reason: %s\n",
                refusedAsk.verdict == boys::ToleranceVerdict::kNotCarried ? "no verdict"
                                                                          : "a verdict",
                refusedAsk.bound,
                refusedAsk.deliveredKnown ? "held" : "not held",
                refusedAsk.reason);

    // And a precision no lane of this library answers for, which is the other
    // kind of refusal: not work this library has not done, but a combination
    // that is not a member of the space at all. The sentence says which.
    const boys::CombinationCoverage noLane =
        boys::QueryCombination(static_cast<boys::Precision>(99), servedRoute, servedScheme,
                               servedAxis, servedGranularity, 1e-12);
    const boys::AccuracyFigure noLaneFigure = boys::BoysAccuracyGuaranteed(
        static_cast<boys::Precision>(99), servedRoute, servedScheme, servedAxis, servedGranularity);

    std::printf("    a precision with no lane: the accessor returns no figure (%s), and the "
                "query agrees: %s\n",
                noLaneFigure.reason,
                noLane.verdict == boys::ToleranceVerdict::kNotCarried ? "no verdict" : "a verdict");

    Require(report, noLaneFigure.value == 0.0 && !noLaneFigure.available && noLane.bound == 0.0 &&
                        noLane.verdict == boys::ToleranceVerdict::kNotCarried,
            "a combination this build does not carry returns no figure and no verdict");

    std::printf("  the accuracy accessors over the whole cross: %zu combination(s) carried, %zu "
                "refused;\n    every carried one asked at the figure its lane publishes "
                "answered inside it, and asked\n    at half of that figure %zu more answered "
                "outside while %zu were inside on the figure\n    they were measured to deliver; "
                "%zu carried rows hold no measured figure to rank by;\n    %zu disagreement(s) "
                "against the two accessors\n",
                carried,
                refused,
                answeredOutside,
                answeredOnTheMeasurement,
                deliveredAbsent,
                disagreements);

    Covered("boys::AccuracyFigure");
    Covered("boys::AccuracyReading");
    Covered("boys::BoysAccuracyGuaranteed");
    Covered("boys::BoysAccuracyDelivered");
    Covered("boys::BoysLaneContracts");
    Covered("boys::LaneContractInfo");
    Covered("boys::CombinationCoverage");
    Covered("boys::ToleranceVerdict");
    Covered("boys::QueryCombination");
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
    CheckEvalSchemes(report, cells);
    CheckGranularityLane(report, cells);
    CheckDoubleLanes(report, cells);
    CheckFitRoutes(report, cells);
    CheckFitRoutesF32(report, cells);
    CheckFloatPolicies(report, cells);
    CheckManyArgumentLanes(report, cells);
    CheckPerElementOrderLanes(report, cells);
    CheckFloatLane(cells);
#if BoysFp16
    CheckHalfIo(cells);
    CheckNativeHalf(report, cells);
#else
    // Stated, not skipped: these are lanes this build does not carry, and a
    // reader of this report is told so beside the lanes that were checked.
    std::printf("  %-56s not carried by this build (BoysFp16 = 0)\n",
                "fp16/bf16 I/O and native half lanes (not checked)");
#endif
    CheckProductModes(report, cells);
#if BoysFp16
    CheckHalf2Surface(report);
#else
    // Stated, not skipped: the packed half type comes to this file through the
    // umbrella header, so a closed-seam build carries none of it.
    std::printf("  %-56s not carried by this build (BoysFp16 = 0)\n",
                "packed half type surface (not checked)");
#endif
    CheckOptionAccuracy(report);

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
