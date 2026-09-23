// Consumer check: are the lane templates' definitions reachable from the
// public headers alone, with no library on the link line?
//
// The library pre-instantiates the default multiplier and the sampled rungs,
// so a consumer that names one of those can link against the library's copy
// whatever the header does. Every other multiplier is compiled at the call
// site, and that only works while the definition is in the header the
// declaration ships in: a definition that lives in a .cpp file is a link error
// for every value except the ones the library exports. This check is the
// consumer for those values, and its whole meaning is what it does NOT link:
// the target links nothing, and it names multipliers the library does not
// pre-instantiate. If the library ever reaches its link line the check still
// compiles and still runs - and proves nothing - so the build asserts that the
// link line is empty (CMakeLists.txt) rather than trusting it here.
//
// It judges values the same way the umbrella consumer does: the committed
// 45-digit grid is the reference, and every returned value is compared with
// the bound its entry documents. The oracle here is the double lane at the
// multiplier this file names, since the certified m = 1 lane is one of the
// pre-instantiated ones - so the bound a composed comparison carries includes
// the oracle's own figure, and the rule's name says so.
//
// Run:  cmake --build <build> --target boys-consumer-header-only
//       <build>/boys-consumer-header-only
//       ctest --test-dir <build> -R boys-consumer-header-only

#include <algorithm>
#include <boys/boys.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

// The library's private src/ directory is deliberately NOT on this file's
// include path. Prove it here rather than assume it: each name below resolves
// only if a directory holding that file is on the include path, and those files
// are the library's sources.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it no longer proves that a consumer can build against the public headers alone"
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

double BatchBound(double m) {
    return m * 5.5e-14;
}

double QuantumOf(double v, int significandBits) {
    if (v == 0.0)
    {
        return 0.0;
    }

    int exponent = 0;
    (void)std::frexp(std::abs(v), &exponent);
    return std::ldexp(1.0, exponent - 1 - significandBits);
}

double F16IoBound(double returned, double m) {
    return m * 1e-7 + 0.5 * QuantumOf(returned, 10);
}

double Bf16IoBound(double returned, double m) {
    return m * 1e-7 + 0.5 * QuantumOf(returned, 7);
}

double ProductBound(boys::ProductMode mode, double m) {
    return mode == boys::ProductMode::kFp64 ? m * 1e-15 : m * 1e-15 + 2.5e-7;
}

/// The multipliers this check names. Neither is one the library pre-instantiates
/// - m = 1, 64, 256, 1024, 4096, 16384 and 65536 are - so every instantiation
/// below is compiled here, from the headers, and must link with no library.
constexpr double kFirst = 3.0;
constexpr double kSecond = 100.0;
constexpr double kOracleBound = kFirst * 5.5e-14;
constexpr double kUnwritten = -1.0;

// --- the checks -------------------------------------------------------------

void CheckDoubleLanes(Report& report, const std::vector<Cell>& cells) {
    char name[160] = {};

    for (const double m : {kFirst, kSecond})
    {
        std::snprintf(name, sizeof(name), "BoysSingle<m = %g> (grid sweep, no library)", m);
        Rule& single = NewRule(name);
        std::snprintf(name, sizeof(name), "BoysAllOrders<m = %g> (grid sweep, no library)", m);
        Rule& batch = NewRule(name);
        std::snprintf(name, sizeof(name), "BoysFixedN<m = %g> (grid sweep, no library)", m);
        Rule& fixed = NewRule(name);

        for (const Cell& cell : cells)
        {
            const double got = m == kFirst ? boys::BoysSingle<kFirst>(cell.n, cell.x)
                                           : boys::BoysSingle<kSecond>(cell.n, cell.x);
            Judge(single, got, cell.value, SingleBound(cell.x, m), cell.n, cell.x);
        }

        std::size_t differing = 0;

        for (const double x : DistinctArgs(cells))
        {
            double all[boys::kMaxBoysOrder + 1] = {};

            if (m == kFirst)
            {
                boys::BoysAllOrders<kFirst>(boys::kMaxBoysOrder, x, all);
            } else
            {
                boys::BoysAllOrders<kSecond>(boys::kMaxBoysOrder, x, all);
            }

            for (const Cell& cell : cells)
            {
                if (cell.x != x)
                {
                    continue;
                }

                double plain = kUnwritten;

                if (m == kFirst)
                {
                    boys::BoysFixedN<kFirst>(cell.n, &x, &plain, 1, 1);
                } else
                {
                    boys::BoysFixedN<kSecond>(cell.n, &x, &plain, 1, 1);
                }

                Judge(batch, all[cell.n], cell.value, BatchBound(m), cell.n, cell.x);
                Judge(fixed, plain, cell.value, SingleBound(cell.x, m), cell.n, cell.x);

                const double one = m == kFirst ? boys::BoysSingle<kFirst>(cell.n, x)
                                               : boys::BoysSingle<kSecond>(cell.n, x);

                if (plain != one)
                {
                    ++differing;
                }
            }
        }

        Require(report, differing == 0, "BoysFixedN returns what BoysSingle returns, bit for bit");

        Covered("boys::BoysSingle<m>");
        Covered("boys::BoysAllOrders<m>");
        Covered("boys::BoysFixedN<m>");
    }
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

    Rule& withWorkspace = NewRule("BoysAllN<m = 3> (caller's workspace, no library)");
    Rule& sorted = NewRule("BoysAllN<m = 3> (BoysSortedArgs, no library)");
    Rule& unsorted = NewRule("BoysAllN<m = 3> (arguments in the caller's order)");

    boys::BoysAllN<kFirst>(boys::kMaxBoysOrder, args.data(), planes.data(), count, nullptr);
    boys::BoysAllN<kFirst>(
        boys::kMaxBoysOrder, args.data(), workspacePlanes.data(), count, workspace.data());
    Require(report,
            std::equal(planes.begin(), planes.end(), workspacePlanes.begin()),
            "the caller's workspace returns the same planes as the internal one");

    boys::BoysAllN<kFirst>(
        boys::kMaxBoysOrder, args.data(), sortedPlanes.data(), count, boys::BoysSortedArgs{});
    Require(report,
            std::equal(planes.begin(), planes.end(), sortedPlanes.begin()),
            "the sorted-argument overload returns the same planes as the sorting one");

    boys::BoysAllN<kFirst>(
        boys::kMaxBoysOrder, reversed.data(), shuffledPlanes.data(), count, nullptr);

    std::size_t missing = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const Cell* first = Find(cells, n, args[i]);
            const Cell* last = Find(cells, n, reversed[i]);
            const std::size_t index = static_cast<std::size_t>(n) * count + i;

            if (first == nullptr || last == nullptr)
            {
                ++missing;
                continue;
            }

            Judge(withWorkspace,
                  workspacePlanes[index],
                  first->value,
                  BatchBound(kFirst),
                  n,
                  args[i]);
            Judge(sorted, sortedPlanes[index], first->value, BatchBound(kFirst), n, args[i]);
            Judge(unsorted, shuffledPlanes[index], last->value, BatchBound(kFirst), n, reversed[i]);
        }
    }

    Require(report, missing == 0, "the grid carries every cell this check looks up");
    Covered("boys::BoysAllN<m>");
    Covered("boys::BoysAllN<m> (BoysSortedArgs)");
    Covered("boys::BoysSortedArgs");
    Covered("boys::BoysAllNWorkspaceSize");
}

void CheckFloatLane(const std::vector<Cell>& cells) {
    Rule& single = NewRule("BoysSingleF32<m = 3> (vs the double lane at float(x), no library)");
    Rule& batch = NewRule("BoysAllOrdersF32<m = 3> (vs the double lane at float(x), no library)");
    const double bound = 3.0 * 1.5e-7 + kOracleBound;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const double oracle = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(xf));
        Judge(single,
              static_cast<double>(boys::BoysSingleF32<kFirst>(cell.n, xf)),
              oracle,
              bound,
              cell.n,
              cell.x);
    }

    for (const double x : DistinctArgs(cells))
    {
        const float xf = static_cast<float>(x);
        float out[boys::kMaxBoysOrder + 1] = {};
        boys::BoysAllOrdersF32<kFirst>(boys::kMaxBoysOrder, xf, out);

        for (const Cell& cell : cells)
        {
            if (cell.x == x)
            {
                const double oracle = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(xf));
                Judge(batch, static_cast<double>(out[cell.n]), oracle, bound, cell.n, cell.x);
            }
        }
    }

    Covered("boys::BoysSingleF32<m>");
    Covered("boys::BoysAllOrdersF32<m>");
}

void CheckHalfIo(const std::vector<Cell>& cells) {
    Rule& f16Single = NewRule("BoysSingleF16<m = 3> (cells above its bound, no library)");
    Rule& bf16Single = NewRule("BoysSingleBf16<m = 3> (cells above its bound, no library)");
    Rule& f16Batch = NewRule("BoysAllOrdersF16<m = 3> (cells above its bound, no library)");
    Rule& bf16Batch = NewRule("BoysAllOrdersBf16<m = 3> (cells above its bound, no library)");
    std::size_t past = 0;

    for (const Cell& cell : cells)
    {
        const float xf = static_cast<float>(cell.x);
        const boys::F16 h = boys::F16(xf);
        const boys::Bf16 b = boys::Bf16(xf);
        const double oracleF16 = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(h));
        const double oracleBf16 = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(b));
        const double gotF16 = static_cast<double>(boys::BoysSingleF16<kFirst>(cell.n, h));
        const double gotBf16 = static_cast<double>(boys::BoysSingleBf16<kFirst>(cell.n, b));
        const double boundF16 = F16IoBound(gotF16, 3.0);
        const double boundBf16 = Bf16IoBound(gotBf16, 3.0);

        // The ceiling the header states, widened by the oracle's own bound:
        // the reference here is the lane at m = 3, not the certified one.
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
        boys::BoysAllOrdersF16<kFirst>(boys::kMaxBoysOrder, h, outF16);
        boys::BoysAllOrdersBf16<kFirst>(boys::kMaxBoysOrder, b, outBf16);

        for (const Cell& cell : cells)
        {
            if (cell.x != x)
            {
                continue;
            }

            const double oracleF16 = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(h));
            const double oracleBf16 = boys::BoysSingle<kFirst>(cell.n, static_cast<double>(b));
            const double gotF16 = static_cast<double>(outF16[cell.n]);
            const double gotBf16 = static_cast<double>(outBf16[cell.n]);
            const double ceilingF16 = F16IoBound(gotF16, 3.0) + kOracleBound;
            const double ceilingBf16 = Bf16IoBound(gotBf16, 3.0) + kOracleBound;

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
    Covered("boys::BoysSingleF16<m>");
    Covered("boys::BoysAllOrdersF16<m>");
    Covered("boys::BoysSingleBf16<m>");
    Covered("boys::BoysAllOrdersBf16<m>");
}

/// The region-A transform: its definition has to be in the header for any
/// multiplier outside the sampled set, in all three modes.
void CheckProductModes(Report& report, const std::vector<Cell>& cells) {
    struct ModeSpec {
        boys::ProductMode mode;
        double multiplier;
        const char* name;
    };

    const ModeSpec kModes[] = {{boys::ProductMode::kFp64, kFirst, "kFp64, m = 3"},
                               {boys::ProductMode::kFp64, kSecond, "kFp64, m = 100"},
                               {boys::ProductMode::kTf32x3, kFirst, "kTf32x3, m = 3"},
                               {boys::ProductMode::kBf16x6, kFirst, "kBf16x6, m = 3"}};
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
                                     (lower ? ", kA1, no library>" : ", kA2, no library>");
            Rule& rule = NewRule(name);
            std::vector<double> out(args.size() * (boys::kMaxBoysOrder + 1));
            std::fill(out.begin(), out.end(), kUnwritten);

            if (mode.mode == boys::ProductMode::kFp64 && mode.multiplier == kFirst)
            {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64, kFirst>(
                    band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());
            } else if (mode.mode == boys::ProductMode::kFp64)
            {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64, kSecond>(
                    band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());
            } else if (mode.mode == boys::ProductMode::kTf32x3)
            {
                boys::BoysRegionAProduct<boys::ProductMode::kTf32x3, kFirst>(
                    band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());
            } else
            {
                boys::BoysRegionAProduct<boys::ProductMode::kBf16x6, kFirst>(
                    band, boys::kMaxBoysOrder, args.data(), out.data(), args.size());
            }

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
                              ProductBound(mode.mode, mode.multiplier),
                              n,
                              args[i]);
                    }
                }
            }

            // Documented: count may be 0, and then nothing is written.
            double untouched[2] = {kUnwritten, kUnwritten};
            const double one[1] = {args.empty() ? 0.0 : args[0]};

            if (mode.mode == boys::ProductMode::kFp64)
            {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64, kFirst>(
                    band, 0, one, untouched, 0);
            } else if (mode.mode == boys::ProductMode::kTf32x3)
            {
                boys::BoysRegionAProduct<boys::ProductMode::kTf32x3, kFirst>(
                    band, 0, one, untouched, 0);
            } else
            {
                boys::BoysRegionAProduct<boys::ProductMode::kBf16x6, kFirst>(
                    band, 0, one, untouched, 0);
            }

            Require(report,
                    untouched[0] == kUnwritten && untouched[1] == kUnwritten,
                    "BoysRegionAProduct writes nothing for count = 0");
        }

        Covered("boys::ProductMode");
        Covered("boys::BoysRegionAProduct (three modes, no library)");
    }
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

    std::printf("header-only consumer check: this binary links no library\n");
    Report report;
    CheckDoubleLanes(report, cells);
    CheckManyArgumentLanes(report, cells);
    CheckFloatLane(cells);
    CheckHalfIo(cells);
    CheckProductModes(report, cells);

    std::printf("consumer check through <boys/boys.hpp> alone: %zu grid cells\n", cells.size());
    PrintRules();
    std::printf("  %zu assertions, %zu failed; %zu documented entries covered\n",
                report.assertions,
                report.failed,
                gCovered.size());

    return report.failed == 0 ? 0 : 1;
}
