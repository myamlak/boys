// Consumer check: are the lane templates' definitions reachable from the public
// headers alone, with no library on the link line? The library pre-instantiates
// the default policy's entries, so only a policy outside that set proves
// anything: this target links nothing (CMakeLists.txt asserts the empty link
// line) and names two it does not export.
//
// The two policies flip one structural axis away from the build's own default and name no
// division form, so each combination named here is one the lane carries and runs the form the
// build's own seam chose. The division form is the one axis that moves a lane's guaranteed
// figure, so the figure a rule is judged against is the build's and not the shipped revision's:
// each rule below reads the form in force and names the lane's own figures for it.
//
// Values are judged against the committed 45-digit grid, with the double lane at
// the policy this file names as the oracle, so a composed bound includes the
// oracle's own figure.
//
// Run:  cmake --build <build> --target boys-consumer-header-only, then
//       <build>/boys-consumer-header-only, or ctest --test-dir <build> -R boys-consumer-header-only

#include <algorithm>
#include <boys/boys.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

// The library's private src/ is deliberately not on this file's include path;
// each name below resolves only if a directory holding that file is on it.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it does not prove that a consumer can build against the public headers alone"
#endif

namespace {

// --- the check's framework --------------------------------------------------

struct Rule {
    std::string name;
    std::size_t cells = 0;
    std::size_t exceeded = 0;
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;
};

struct Report {
    std::size_t assertions = 0;
    std::size_t failed = 0;
};

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

Rule& NewRule(const std::string& name) {
    gRules.push_back(Rule{name, 0, 0, 0.0, -1, 0.0});
    return gRules.back();
}

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

/// The bound is the lane's and not the combination's, so the named policy is judged against the
/// same figures the default policy's entry documents.
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

double BatchBound() {
    return 5.5e-14;
}

// The ULP of a representation at a magnitude, for the half-ULP ceilings below.
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

// fp16/bf16 I/O lane ceilings: 1e-7 of the certified float engine, plus one half-ULP per value.
#if BoysFp16
double F16IoBound(double returned) {
    return 1e-7 + 0.5 * QuantumOf(returned, 10);
}

double Bf16IoBound(double returned) {
    return 1e-7 + 0.5 * QuantumOf(returned, 7);
}
#endif // BoysFp16

// The two policies this check names. Neither is one the library pre-instantiates
// (the `<>` entries are), so every instantiation below must link from the headers
// alone. Each flips one structural axis away from the build's own default and
// leaves the rest - so the combination is one the lane carries - and neither
// flips the division form, the one axis that moves a lane's guaranteed figure.
constexpr boys::FitRoute kOtherRoute = boys::kDefaultFitRoute == boys::FitRoute::kChebyshev
                                           ? boys::FitRoute::kRationalMinimax
                                           : boys::FitRoute::kChebyshev;
constexpr boys::EvalScheme kOtherScheme = boys::kDefaultEvalScheme == boys::EvalScheme::kHorner
                                              ? boys::EvalScheme::kSplitClenshaw
                                              : boys::EvalScheme::kHorner;

using First = boys::EvalPolicy<kOtherRoute, boys::kDefaultEvalScheme, boys::BoysBudget::kFloat,
                               boys::kDefaultPackAxis, boys::kDefaultFitGranularity>;
using Second = boys::EvalPolicy<boys::kDefaultFitRoute, kOtherScheme, boys::BoysBudget::kFloat,
                                boys::kDefaultPackAxis, boys::kDefaultFitGranularity>;
// The half lanes run the fp16 engine budget, so their policies name it.
using FirstHalf = boys::EvalPolicy<kOtherRoute, boys::kDefaultEvalScheme, boys::BoysBudget::kFp16,
                                   boys::kDefaultPackAxis, boys::kDefaultFitGranularity>;
using SecondHalf = boys::EvalPolicy<boys::kDefaultFitRoute, kOtherScheme, boys::BoysBudget::kFp16,
                                    boys::kDefaultPackAxis, boys::kDefaultFitGranularity>;

// The claim every call below rests on: a named policy is not the type its entry was
// pre-instantiated at, so the call is an instantiation this translation unit owes a definition for,
// and one it can only get from the headers. A build whose default row named the combination above
// would fold the call onto the library's own instantiation and this check would prove nothing.
static_assert(
    !std::is_same_v<First, boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>>);
static_assert(
    !std::is_same_v<First, boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>>);
static_assert(
    !std::is_same_v<Second, boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>>);
static_assert(
    !std::is_same_v<Second, boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>>);
static_assert(
    !std::is_same_v<FirstHalf, boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kSingle>>);
static_assert(
    !std::is_same_v<FirstHalf,
                    boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>>);
static_assert(
    !std::is_same_v<SecondHalf, boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kSingle>>);
static_assert(
    !std::is_same_v<SecondHalf,
                    boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>>);

/// The label a rule carries for a named policy: the axis that policy flips away from the build's
/// own default, which is what makes it an instantiation the library does not pre-instantiate.
template <class Policy>
const char* PolicyLabel() {
    return Policy::kRoute != boys::kDefaultFitRoute ? "the other route" : "the other scheme";
}

constexpr double kOracleBound = 5.5e-14;
constexpr double kUnwritten = -1.0;

// --- the checks -------------------------------------------------------------

/// The double lane's whole family at one named policy: the single, all-orders and fixed-N entries
/// over the grid, and the fixed-N entry's values against the single entry's bit for bit. The
/// instantiation is this file's: the policy is not the type the library pre-instantiates.
template <class Policy>
void CheckDoubleLanes(Report& report, const std::vector<Cell>& cells) {
    char name[160] = {};

    std::snprintf(name, sizeof(name), "BoysSingle<%s> (grid sweep, no library)",
                  PolicyLabel<Policy>());
    Rule& single = NewRule(name);
    std::snprintf(name, sizeof(name), "BoysAllOrders<%s> (grid sweep, no library)",
                  PolicyLabel<Policy>());
    Rule& batch = NewRule(name);
    std::snprintf(name, sizeof(name), "BoysFixedN<%s> (grid sweep, no library)",
                  PolicyLabel<Policy>());
    Rule& fixed = NewRule(name);

    for (const Cell& cell : cells)
    {
        const double got = boys::BoysSingle<Policy>(cell.n, cell.x);
        Judge(single, got, cell.value, SingleBound(cell.x), cell.n, cell.x);
    }

    std::size_t differing = 0;

    for (const double x : DistinctArgs(cells))
    {
        double all[boys::kMaxBoysOrder + 1] = {};
        boys::BoysAllOrders<Policy>(boys::kMaxBoysOrder, x, all);

        for (const Cell& cell : cells)
        {
            if (cell.x != x)
            {
                continue;
            }

            double plain = kUnwritten;
            boys::BoysFixedN<Policy>(cell.n, &x, &plain, 1, 1);

            Judge(batch, all[cell.n], cell.value, BatchBound(), cell.n, cell.x);
            Judge(fixed, plain, cell.value, SingleBound(cell.x), cell.n, cell.x);

            const double one = boys::BoysSingle<Policy>(cell.n, x);

            if (plain != one)
            {
                ++differing;
            }
        }
    }

    Require(report, differing == 0, "BoysFixedN returns what BoysSingle returns, bit for bit");

    Covered("boys::BoysSingle<Policy>");
    Covered("boys::BoysAllOrders<Policy>");
    Covered("boys::BoysFixedN<Policy>");
}

void CheckManyArgumentLanes(Report& report, const std::vector<Cell>& cells) {
    const std::vector<double> args = DistinctArgs(cells);
    const std::size_t count = args.size();
    const std::size_t plane = boys::kMaxBoysOrder + 1;
    std::vector<double> planes(count * plane);
    std::vector<double> workspacePlanes(count * plane);
    std::vector<double> sortedPlanes(count * plane);
    std::vector<double> shuffledPlanes(count * plane);
    std::vector<std::size_t> workspace(boys::BoysAllNWorkspaceSize(count));
    std::vector<double> reversed(count);
    std::reverse_copy(args.begin(), args.end(), reversed.begin());

    Rule& withWorkspace = NewRule("BoysAllN<the other route> (caller's workspace, no library)");
    Rule& sorted = NewRule("BoysAllN<the other route> (BoysSortedArgs, no library)");
    Rule& unsorted = NewRule("BoysAllN<the other route> (arguments in the caller's order)");

    boys::BoysAllN<First>(boys::kMaxBoysOrder, args.data(), planes.data(), count, nullptr);
    boys::BoysAllN<First>(
        boys::kMaxBoysOrder, args.data(), workspacePlanes.data(), count, workspace.data());
    Require(report,
            std::equal(planes.begin(), planes.end(), workspacePlanes.begin()),
            "the caller's workspace returns the same planes as the internal one");

    boys::BoysAllN<First>(
        boys::kMaxBoysOrder, args.data(), sortedPlanes.data(), count, boys::BoysSortedArgs{});
    Require(report,
            std::equal(planes.begin(), planes.end(), sortedPlanes.begin()),
            "the sorted-argument overload returns the same planes as the sorting one");

    boys::BoysAllN<First>(
        boys::kMaxBoysOrder, reversed.data(), shuffledPlanes.data(), count, nullptr);

    std::size_t missing = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const Cell* firstCell = Find(cells, n, args[i]);
            const Cell* last = Find(cells, n, reversed[i]);
            const std::size_t index = static_cast<std::size_t>(n) * count + i;

            if (firstCell == nullptr || last == nullptr)
            {
                ++missing;
                continue;
            }

            Judge(withWorkspace,
                  workspacePlanes[index],
                  firstCell->value,
                  BatchBound(),
                  n,
                  args[i]);
            Judge(sorted, sortedPlanes[index], firstCell->value, BatchBound(), n, args[i]);
            Judge(unsorted, shuffledPlanes[index], last->value, BatchBound(), n, reversed[i]);
        }
    }

    Require(report, missing == 0, "the grid carries every cell this check looks up");
    Covered("boys::BoysAllN<Policy>");
    Covered("boys::BoysAllN<Policy> (BoysSortedArgs)");
    Covered("boys::BoysSortedArgs");
    Covered("boys::BoysAllNWorkspaceSize");
}

/// The float lane's two entries at one named policy, judged against the double lane at the same
/// converted argument: the composed bound is the float lane's own figure plus the oracle's.
///
/// The lane's figure is the one it publishes for the division form these entries run, which is
/// not one figure: the lane's row carries a base of 1.5e-7 and, beside it, the term the plain
/// reciprocal adds - 1.5e-7 plus 1e-7 under that form - because the form rounds once more per
/// step (include/boys/boys.hpp, LaneContractInfo; include/boys/backend.hpp, DivisionForm states
/// the same two figures, measured, for this lane of the three the axis has). Neither policy
/// below names a form, so the form in force is the build's own default one, and a build whose
/// seam moved that axis publishes and runs the second figure rather than the first. This target
/// links no library, so `BoysLaneContracts()` is not reachable here and the two figures are
/// written as the row states them; what sweeps the lane against an independent reference and
/// holds it inside them is the accuracy gate's float book, tests/boys_accuracy_gate.cpp.
///
/// The term is what the plain form spends on the downward ladder, which is inside the batch
/// shape and not inside the single one, so the batch rule is judged at the sum and the single
/// rule at the base - the reading tests/boys_accuracy_test.cpp states for the same two shapes.
template <class Policy>
void CheckFloatLane(const std::vector<Cell>& cells) {
    Rule& single = NewRule("BoysSingleF32<the other route> (vs the double lane at float(x))");
    Rule& batch = NewRule("BoysAllOrdersF32<the other route> (vs the double lane, float(x))");

    constexpr double kFloatLaneBase = 1.5e-7;
    constexpr double kFloatPlainTerm = 1e-7;
    const double plainTerm =
        boys::kDefaultDivisionForm == boys::DivisionForm::kPlainReciprocal ? kFloatPlainTerm : 0.0;
    const double singleBound = kFloatLaneBase + kOracleBound;
    const double batchBound = kFloatLaneBase + plainTerm + kOracleBound;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const double oracle = boys::BoysSingle<First>(cell.n, static_cast<double>(xf));
        Judge(single,
              static_cast<double>(boys::BoysSingleF32<Policy>(cell.n, xf)),
              oracle,
              singleBound,
              cell.n,
              cell.x);
    }

    for (const double x : DistinctArgs(cells))
    {
        const float xf = static_cast<float>(x);
        float out[boys::kMaxBoysOrder + 1] = {};
        boys::BoysAllOrdersF32<Policy>(boys::kMaxBoysOrder, xf, out);

        for (const Cell& cell : cells)
        {
            if (cell.x == x)
            {
                const double oracle = boys::BoysSingle<First>(cell.n, static_cast<double>(xf));
                Judge(batch, static_cast<double>(out[cell.n]), oracle, batchBound, cell.n, cell.x);
            }
        }
    }

    Covered("boys::BoysSingleF32<Policy>");
    Covered("boys::BoysAllOrdersF32<Policy>");
}

// The fp16/bf16 I/O lanes the BoysFp16 seam declares; with the seam closed main
// reports them as not carried by this build rather than dropping them in silence.
#if BoysFp16
template <class Policy>
void CheckHalfIo(const std::vector<Cell>& cells) {
    Rule& f16Single = NewRule("BoysSingleF16<the other route> (cells above its bound)");
    Rule& bf16Single = NewRule("BoysSingleBf16<the other route> (cells above its bound)");
    Rule& f16Batch = NewRule("BoysAllOrdersF16<the other route> (cells above its bound)");
    Rule& bf16Batch = NewRule("BoysAllOrdersBf16<the other route> (cells above its bound)");
    std::size_t past = 0;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const boys::F16 h = boys::F16(xf);
        const boys::Bf16 b = boys::Bf16(xf);
        const double oracleF16 = boys::BoysSingle<First>(cell.n, static_cast<double>(h));
        const double oracleBf16 = boys::BoysSingle<First>(cell.n, static_cast<double>(b));
        const double gotF16 = static_cast<double>(boys::BoysSingleF16<Policy>(cell.n, h));
        const double gotBf16 = static_cast<double>(boys::BoysSingleBf16<Policy>(cell.n, b));
        const double boundF16 = F16IoBound(gotF16);
        const double boundBf16 = Bf16IoBound(gotBf16);

        // The ceiling the header states, widened by the oracle's own bound: the
        // reference here is the double lane's named policy, whose own figure
        // composes with the half lane's.
        const double ceilingF16 = boundF16 + kOracleBound;
        const double ceilingBf16 = boundBf16 + kOracleBound;

        if (std::abs(oracleF16) > ceilingF16)
        {
            Judge(f16Single, gotF16, oracleF16, ceilingF16, cell.n, cell.x);
        } else
        {
            ++past;
        }

        if (std::abs(oracleBf16) > ceilingBf16)
        {
            Judge(bf16Single, gotBf16, oracleBf16, ceilingBf16, cell.n, cell.x);
        }
    }

    for (const double x : DistinctArgs(cells))
    {
        const float xf = static_cast<float>(x);
        const boys::F16 h = boys::F16(xf);
        const boys::Bf16 b = boys::Bf16(xf);
        boys::F16 outF16[boys::kMaxBoysOrder + 1] = {};
        boys::Bf16 outBf16[boys::kMaxBoysOrder + 1] = {};
        boys::BoysAllOrdersF16<Policy>(boys::kMaxBoysOrder, h, outF16);
        boys::BoysAllOrdersBf16<Policy>(boys::kMaxBoysOrder, b, outBf16);

        for (const Cell& cell : cells)
        {
            if (cell.x != x)
            {
                continue;
            }

            const double oracleF16 = boys::BoysSingle<First>(cell.n, static_cast<double>(h));
            const double oracleBf16 = boys::BoysSingle<First>(cell.n, static_cast<double>(b));
            const double gotF16 = static_cast<double>(outF16[cell.n]);
            const double gotBf16 = static_cast<double>(outBf16[cell.n]);
            const double ceilingF16 = F16IoBound(gotF16) + kOracleBound;
            const double ceilingBf16 = Bf16IoBound(gotBf16) + kOracleBound;

            if (std::abs(oracleF16) > ceilingF16)
            {
                Judge(f16Batch, gotF16, oracleF16, ceilingF16, cell.n, cell.x);
            }

            if (std::abs(oracleBf16) > ceilingBf16)
            {
                Judge(bf16Batch, gotBf16, oracleBf16, ceilingBf16, cell.n, cell.x);
            }
        }
    }

    std::printf("  %-56s %zu cells at or below the fp16 lane's bound\n",
                "fp16/bf16 lanes (outside their bound's reach)",
                past);
    Covered("boys::BoysSingleF16<Policy>");
    Covered("boys::BoysAllOrdersF16<Policy>");
    Covered("boys::BoysSingleBf16<Policy>");
    Covered("boys::BoysAllOrdersBf16<Policy>");
}
#endif // BoysFp16

} // namespace

int main(int argc, char** argv) {
    const char* gridPath = argc > 1 ? argv[1] : BoysConsumerReference;
    const std::vector<Cell> cells = LoadGrid(gridPath);

    if (cells.empty())
    {
        std::printf("consumer check: cannot read the reference grid at %s\n", gridPath);
        return 1;
    }

    std::printf("header-only consumer check: this binary links no library\n");
    Report report;
    CheckDoubleLanes<First>(report, cells);
    CheckDoubleLanes<Second>(report, cells);
    CheckManyArgumentLanes(report, cells);
    CheckFloatLane<First>(cells);
#if BoysFp16
    CheckHalfIo<FirstHalf>(cells);
#else
    // Stated rather than skipped: this build does not carry the lane.
    std::printf("  %-56s not carried by this build (BoysFp16 = 0)\n",
                "fp16/bf16 lanes (not checked)");
#endif

    std::printf("consumer check through <boys/boys.hpp> alone: %zu grid cells\n", cells.size());
    PrintRules();

    // The violations PrintRules has just listed are this check's own verdict. A
    // sweep whose cells may exceed the bound their lane documents and still exit 0
    // is a check that cannot fail, so every rule's count is summed into the one
    // assertion that decides the exit code. The float and the half lanes hold no
    // Report of their own and reach no other assertion, so this is also the only
    // place their bounds are answered for.
    std::size_t exceeded = 0;

    for (const Rule& rule : gRules)
    {
        exceeded += rule.exceeded;
    }

    Require(report, exceeded == 0, "every grid cell is within the bound its lane documents");

    std::printf("  %zu assertions, %zu failed; %zu documented entries covered\n",
                report.assertions,
                report.failed,
                gCovered.size());

    return report.failed == 0 ? 0 : 1;
}
