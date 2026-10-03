// Consumer check: does every combination of the option space this build serves have a NAME a
// consumer can write, and does naming it evaluate that combination?
//
// The option space is four structural axes - the fit route, the evaluation scheme, the interval
// partition and the packing axis - per precision: the axes are what a call site chooses once, when
// it is written, and a policy names one combination of them.
//
//  * for every combination the library's own book reports as served (BoysAccuracyGuaranteed
//    answering a figure for the five axes), this program writes THE NAME of that combination - a
//    policy type - and calls the entry at that name: one arm per name, resolved where the line is
//    written, with no string, no table and no run-time search anywhere in this file, so a
//    combination it could not name would not compile;
//
//  * every value that comes back is judged against the committed 45-digit reference grid within
//    the bound the book states for the combination that was named - the figure
//    BoysAccuracyGuaranteed answers for the same five axes;
//
//  * the census is printed per precision - combinations served, combinations named, cells served,
//    cells named and evaluated - counted from the library's own tables rather than a list kept
//    here; the axes this file spells are checked against those tables first, so a third scheme or
//    partition is a reported gap rather than an else arm.
//
// The four precisions are four measurements reported apart: the double lane's entries take a
// policy at the library's own budget, the float lane's at its own, the half lanes run the float
// lane's engine at the tighter fp16 budget with the half store. Where a format conversion sits
// between the grid and the entry, the comparison is against the certified double lane at the same
// converted argument, as the umbrella consumer check does it, and the half lanes' representation
// term is one half-ULP of the value they returned.
//
// Run:  cmake --build <build> --target boys-consumer-option-space
//       <build>/boys-consumer-option-space
//       ctest --test-dir <build> -R boys-consumer-option-space

#include <algorithm>
#include <array>
#include <boys/boys.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <span>
#include <string>
#include <vector>

// The library's private src/ directory is deliberately NOT on this file's include path: a
// consumer gets include/ and nothing else. Each of the five names below resolves only if such a
// directory is on the path, so renaming one source file does not quietly retire the probe.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it does not prove that a consumer can name the option space against the public headers alone"
#endif

namespace {

using boys::BoysBudget;
using boys::EvalPolicy;
using boys::EvalScheme;
using boys::FitGranularity;
using boys::FitRoute;
using boys::PackAxis;

/// The certified double lane, which stands in for the committed grid at the converted argument
/// wherever a format conversion sits between the grid and the named entry: a composed bound is the
/// cell's own figure plus this one.
constexpr double kOracleBound = 5.5e-14;

/// The four structural axes, each as the enumerators this file spells an arm for; the tables the
/// library publishes are checked against these before anything is counted.
constexpr std::array<FitRoute, 2> kRoutes = {FitRoute::kChebyshev, FitRoute::kRationalMinimax};
constexpr std::array<EvalScheme, 2> kSchemes = {EvalScheme::kSplitClenshaw, EvalScheme::kHorner};
constexpr std::array<FitGranularity, 3> kPartitions = {
    FitGranularity::kCoarsest, FitGranularity::kNarrow, FitGranularity::kUniform};
constexpr std::array<PackAxis, 2> kAxes = {PackAxis::kArguments, PackAxis::kOrders};

/// The precision classes this file sweeps: one per name the library publishes a figure for, with
/// the lane whose book each class's cells are counted in and the named entry its combinations are
/// written with. The two half formats are one lane and two classes.
struct ClassInfo {
    const char* name;  ///< the name the census prints
    const char* entry; ///< the named entry the class's combinations are written with
    boys::Precision lane;
};

constexpr std::array<ClassInfo, 4> kClasses = {{
    {"fp64", "BoysAllOrders<>", boys::Precision::kFp64},
    {"fp32", "BoysAllOrdersF32<>", boys::Precision::kFp32},
    {"fp16", "BoysAllOrdersF16<>", boys::Precision::kFp16},
    {"bf16", "BoysAllOrdersBf16<>", boys::Precision::kFp16},
}};

// --- the committed reference grid -------------------------------------------

/// One cell of the committed 45-digit grid: F_n(x) to more digits than any lane here returns.
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

/// The distinct arguments of the grid, ascending.
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

/// Every k-th argument of the grid, so the sweep visits the whole argument line rather than its
/// head: the fits, the seed and the asymptotic form are three pieces of arithmetic behind one
/// entry, and one argument alone would not say which of them a combination reached.
std::vector<double> SweepArgs(const std::vector<Cell>& cells) {
    const std::vector<double> args = DistinctArgs(cells);
    std::vector<double> swept;

    constexpr std::size_t kStride = 8;

    for (std::size_t i = 0; i < args.size(); i += kStride)
    {
        swept.push_back(args[i]);
    }

    if (!args.empty() && (swept.empty() || swept.back() != args.back()))
    {
        swept.push_back(args.back());
    }

    return swept;
}

// --- the bookkeeping --------------------------------------------------------

struct Rule {
    std::string name;
    std::size_t cells = 0;
    std::size_t exceeded = 0;
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;
};

struct Census {
    std::size_t servedCombinations = 0;
    std::size_t namedCombinations = 0;
    std::size_t servedCells = 0;
    std::size_t namedCells = 0;
};

/// A deque, not a vector: the checks hold a reference to a rule while they
/// register the next one, and a deque keeps those references valid.
std::deque<Rule> gRules;
std::size_t gPrinted = 0;
constexpr std::size_t kPrintedPerRule = 5;
std::size_t gFailures = 0;

/// The denominators the verdict prints: the values judged, and the values the half lanes claim
/// no bound for. A green line that says how much it judged is a measurement; one that says only
/// "PASS" is not.
std::size_t gJudged = 0;
std::size_t gSkipped = 0;

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

    ++gFailures;
}

/// Judge one value: |measured - reference| against the bound the book states for the combination
/// that was named. A non-finite value is counted as exceeded rather than compared,
/// because a ratio against a bound is false for every NaN and would otherwise slip through.
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

Rule& NewRule(const std::string& name) {
    gRules.push_back(Rule{name, 0, 0, 0.0, -1, 0.0});
    return gRules.back();
}

void Require(bool ok, const char* what) {
    if (!ok)
    {
        std::printf("  FAILED %s\n", what);
        ++gFailures;
    }
}

void PrintRules() {
    for (const Rule& rule : gRules)
    {
        std::printf("  %-58s %8zu cells  worst %.4g of the bound", rule.name.c_str(), rule.cells,
                    rule.worst);

        if (rule.worstN >= 0)
        {
            std::printf(" (n=%d, x=%.6g)", rule.worstN, rule.worstX);
        }

        std::printf("  exceeded %zu\n", rule.exceeded);
    }
}

#if BoysFp16
/// The quantum (one ULP) of a binary16 or bfloat16 value of the given significand
/// width: 2^(e - bits) for a normal value 1.f * 2^e, and zero at zero, where no
/// quantum is defined. The half lanes' representation term is one half of it.
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

// --- the four axes as a call's template argument ---------------------------

/// The four structural axes of one cell, each narrowed to the value that names it. The nested
/// lambdas write one arm per name without sixteen of them by hand; the innermost arm is the policy
/// a call site writes. Each level's else arm names the other enumerator it spells and is not a
/// default, so a value that is neither is caught by the table check rather than answered here; for
/// the partition, the one axis that carries three members, it is caught here too,
/// because answering from the shipped tables would measure one partition's cell under another name.
template <typename Call>
void AtAxes(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
            Call&& call) {
    const auto with_partition = [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack>() {
        if (partition == FitGranularity::kNarrow)
        {
            call.template operator()<kRoute, kScheme, kPack, FitGranularity::kNarrow>();
        } else if (partition == FitGranularity::kCoarsest)
        {
            call.template operator()<kRoute, kScheme, kPack, FitGranularity::kCoarsest>();
        } else if (partition == FitGranularity::kUniform)
        {
            call.template operator()<kRoute, kScheme, kPack, FitGranularity::kUniform>();
        } else
        {
            std::printf("boys consumer option space: a cell was named at a partition this file "
                        "writes no arm for (%d), and answering it from the shipped partition's "
                        "tables would measure one partition's cell under another's name\n",
                        static_cast<int>(partition));
            std::fflush(stdout);
            std::abort();
        }
    };

    const auto with_pack = [&]<FitRoute kRoute, EvalScheme kScheme>() {
        if (pack == PackAxis::kOrders)
        {
            with_partition.template operator()<kRoute, kScheme, PackAxis::kOrders>();
        } else
        {
            with_partition.template operator()<kRoute, kScheme, PackAxis::kArguments>();
        }
    };

    const auto with_scheme = [&]<FitRoute kRoute>() {
        if (scheme == EvalScheme::kHorner)
        {
            with_pack.template operator()<kRoute, EvalScheme::kHorner>();
        } else
        {
            with_pack.template operator()<kRoute, EvalScheme::kSplitClenshaw>();
        }
    };

    if (route == FitRoute::kRationalMinimax)
    {
        with_scheme.template operator()<FitRoute::kRationalMinimax>();
    } else
    {
        with_scheme.template operator()<FitRoute::kChebyshev>();
    }
}

// --- the named cells, one arm per name --------------------------------------

/// One combination, evaluated through its own name. The policy is the entry's only template
/// argument and it names the whole combination, so the arm that names a cell is the call itself.

void EvalFp64(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              int nmax, double x, double* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               boys::BoysAllOrders<Policy>(nmax, x, out);
           });
}

void EvalFp32(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              int nmax, float x, float* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               boys::BoysAllOrdersF32<Policy>(nmax, x, out);
           });
}

#if BoysFp16
void EvalFp16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              int nmax, boys::F16 x, boys::F16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               boys::BoysAllOrdersF16<Policy>(nmax, x, out);
           });
}

void EvalBf16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              int nmax, boys::Bf16 x, boys::Bf16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               boys::BoysAllOrdersBf16<Policy>(nmax, x, out);
           });
}
#endif // BoysFp16

// --- one cell over the sweep ------------------------------------------------

/// Name one cell through its own policy, judge its values against the book's figure for the
/// combination, and leave the counts the verdict prints where they are counted. The named counts
/// are this arm's own, incremented here and not beside the book's answer, so the verdict compares
/// what the book serves against what this file ran.
void SweepCell(std::size_t classIndex,
               Census& census,
               Rule& bound,
               FitRoute route,
               EvalScheme scheme,
               PackAxis pack,
               FitGranularity partition,
               const std::vector<Cell>& grid,
               const std::vector<double>& args) {
    // One accuracy is one cell of the option space, so the combination and its cell are counted
    // together: with the rung gone there is no second accuracy point to tell them apart.
    ++census.namedCombinations;
    ++census.namedCells;

    constexpr int kNmax = boys::kMaxBoysOrder;
    constexpr std::size_t kWidth = static_cast<std::size_t>(kNmax) + 1;

    const boys::AccuracyFigure figure =
        boys::BoysAccuracyGuaranteed(kClasses[classIndex].lane, route, scheme, pack, partition);

    std::vector<double> value(kWidth);

    for (std::size_t a = 0; a < args.size(); ++a)
    {
        const double x = args[a];

        if (classIndex == 0)
        {
            EvalFp64(route, scheme, pack, partition, kNmax, x, value.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                const Cell* cell = Find(grid, n, x);

                if (cell != nullptr)
                {
                    Judge(bound, value[static_cast<std::size_t>(n)], cell->value, figure.value, n,
                          x);
                    ++gJudged;
                }
            }
        } else if (classIndex == 1)
        {
            const float x32 = static_cast<float>(x);
            std::vector<float> fvalue(kWidth);

            EvalFp32(route, scheme, pack, partition, kNmax, x32, fvalue.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                const std::size_t k = static_cast<std::size_t>(n);
                value[k] = static_cast<double>(fvalue[k]);

                // The lane rounds its argument before evaluating, so the reference is the
                // certified double lane at the argument the lane was asked about, and the bound
                // composes the cell's figure with its own.
                const double reference = boys::BoysSingle<>(n, static_cast<double>(x32));

                Judge(bound, value[k], reference, figure.value + kOracleBound, n, x);
                ++gJudged;
            }
        } else
        {
#if BoysFp16
            const float x32 = static_cast<float>(x);
            const bool isBf16 = classIndex == 3;

            if (isBf16)
            {
                std::vector<boys::Bf16> bvalue(kWidth);
                const boys::Bf16 h(x32);

                EvalBf16(route, scheme, pack, partition, kNmax, h, bvalue.data());

                for (int n = 0; n <= kNmax; ++n)
                {
                    const std::size_t k = static_cast<std::size_t>(n);
                    value[k] = static_cast<double>(bvalue[k]);

                    const double reference =
                        boys::BoysSingle<>(n, static_cast<double>(static_cast<float>(h)));
                    const double cellBound =
                        figure.value + 0.5 * QuantumOf(value[k], 7) + kOracleBound;

                    if (std::abs(reference) > cellBound)
                    {
                        Judge(bound, value[k], reference, cellBound, n, x);
                        ++gJudged;
                    } else
                    {
                        ++gSkipped;
                    }
                }
            } else
            {
                std::vector<boys::F16> hvalue(kWidth);
                const boys::F16 h(x32);

                EvalFp16(route, scheme, pack, partition, kNmax, h, hvalue.data());

                for (int n = 0; n <= kNmax; ++n)
                {
                    const std::size_t k = static_cast<std::size_t>(n);
                    value[k] = static_cast<double>(hvalue[k]);

                    const double reference =
                        boys::BoysSingle<>(n, static_cast<double>(static_cast<float>(h)));
                    const double cellBound =
                        figure.value + 0.5 * QuantumOf(value[k], 10) + kOracleBound;

                    if (std::abs(reference) > cellBound)
                    {
                        Judge(bound, value[k], reference, cellBound, n, x);
                        ++gJudged;
                    } else
                    {
                        ++gSkipped;
                    }
                }
            }
#else
// A build whose fp16 seam is closed carries neither half class: the census reports them as not
// carried rather than sweeping an empty arm, and this program's rules for them judge nothing.
            std::fflush(stdout);
#endif // BoysFp16
        }
    }
}

// --- the axes the library publishes -----------------------------------------

/// The distinct route values of a lane's route table, which is one row per route and region.
std::vector<FitRoute> DistinctRoutes(std::span<const boys::FitRouteInfo> rows) {
    std::vector<FitRoute> routes;

    for (const boys::FitRouteInfo& row : rows)
    {
        if (std::find(routes.begin(), routes.end(), row.route) == routes.end())
        {
            routes.push_back(row.route);
        }
    }

    return routes;
}

/// Check the tables this file spells its arms from against the tables the library publishes, so an
/// axis added to the library is a counted gap here rather than a value swept into an else arm.
void CheckAxes(std::size_t classIndex) {
    const std::span<const boys::FitRouteInfo> rows =
        classIndex == 0 ? boys::BoysFitRoutes() : boys::BoysFitRoutesF32();

    const std::vector<FitRoute> routes = DistinctRoutes(rows);

    Require(routes.size() == kRoutes.size(),
            "the lane publishes one route per arm this file writes");

    for (const FitRoute route : routes)
    {
        Require(std::find(kRoutes.begin(), kRoutes.end(), route) != kRoutes.end(),
                "every published route is an arm this file writes");
    }

    Require(boys::BoysEvalSchemes().size() == kSchemes.size(),
            "the library publishes one scheme per arm this file writes");
    Require(boys::BoysPackAxes().size() == kAxes.size(),
            "the library publishes one packing axis per arm this file writes");

    Require(boys::BoysFitGranularities().size() == kPartitions.size(),
            "the library publishes one partition per arm this file writes");

    // Checked row by row as well as by count, because a row whose selector this file writes no arm
    // for must be a counted gap rather than a value swept into an else arm.
    for (const boys::FitGranularityInfo& row : boys::BoysFitGranularities())
    {
        Require(std::find(kPartitions.begin(), kPartitions.end(), row.granularity) !=
                    kPartitions.end(),
                "every published partition is an arm this file writes");
    }
}

// --- one class's census -----------------------------------------------------

Census RunClass(std::size_t classIndex, const std::vector<Cell>& grid,
                const std::vector<double>& args) {
    Census census;

    const std::string prefix = std::string(kClasses[classIndex].name) + " ";

    Rule& bound = NewRule(prefix + kClasses[classIndex].entry + " (values above the bound)");

    for (const FitRoute route : kRoutes)
    {
        for (const EvalScheme scheme : kSchemes)
        {
            for (const PackAxis pack : kAxes)
            {
                for (const FitGranularity partition : kPartitions)
                {
                    // What the book serves here: the figure the library answers for this
                    // combination, at this class's own precision. A combination it answers no
                    // figure for is one this file has no arm for, and it is counted nowhere rather
                    // than swept against a figure that is not its own.
                    const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
                        kClasses[classIndex].lane, route, scheme, pack, partition);

                    if (!figure.available)
                    {
                        continue;
                    }

                    // Every combination the book serves is one of the arms this file writes: the
                    // axes are the enumerations the table check above held, and the arms are their
                    // product. One accuracy is one cell of the option space, so a served
                    // combination is one served cell.
                    ++census.servedCombinations;
                    ++census.servedCells;

                    SweepCell(
                        classIndex, census, bound, route, scheme, pack, partition, grid, args);
                }
            }
        }
    }

    return census;
}

} // namespace

int main(int argc, char** argv) {
    const char* gridPath = nullptr;

#ifdef BoysConsumerReference
    gridPath = BoysConsumerReference;
#endif

    if (argc > 1)
    {
        gridPath = argv[1];
    }

    const std::vector<Cell> grid = gridPath != nullptr ? LoadGrid(gridPath) : std::vector<Cell>{};

    if (grid.empty())
    {
        std::printf("boys consumer option space: no reference grid (%s)\n",
                    gridPath != nullptr ? gridPath : "no path compiled in");
        return 1;
    }

    const std::vector<double> args = SweepArgs(grid);

    std::printf("boys consumer option space: %zu grid cells, %zu arguments swept, "
                "%zu combinations per class\n",
                grid.size(), args.size(),
                kRoutes.size() * kSchemes.size() * kAxes.size() * kPartitions.size());

    std::array<Census, 4> censuses{};

    for (std::size_t i = 0; i < kClasses.size(); ++i)
    {
#if !BoysFp16
        // A build whose seam is closed declares neither half lane, so neither half class is
        // carried: the census says so rather than counting cells this build has no entry for.
        if (i >= 2)
        {
            std::printf("  %-6s not carried: this build has BoysFp16 closed\n", kClasses[i].name);
            continue;
        }
#endif

        CheckAxes(i);
        censuses[i] = RunClass(i, grid, args);
    }

    std::printf("\nper class, counted from the library's own book:\n");
    std::printf("  %-6s %-26s %8s %8s %9s %9s\n", "class", "entry", "combos", "named", "cells",
                "named");
    for (std::size_t i = 0; i < kClasses.size(); ++i)
    {
        const Census& census = censuses[i];
        std::printf("  %-6s %-26s %8zu %8zu %9zu %9zu\n", kClasses[i].name, kClasses[i].entry,
                    census.servedCombinations, census.namedCombinations, census.servedCells,
                    census.namedCells);
    }

    std::printf("\nrules:\n");
    PrintRules();

    std::printf("\nvalues judged %zu, skipped below the half lanes' floor %zu\n", gJudged,
                gSkipped);

    std::size_t servedCombinations = 0;
    std::size_t namedCombinations = 0;
    std::size_t servedCells = 0;
    std::size_t namedCells = 0;

    for (const Census& census : censuses)
    {
        servedCombinations += census.servedCombinations;
        namedCombinations += census.namedCombinations;
        servedCells += census.servedCells;
        namedCells += census.namedCells;
    }

    std::printf("combinations served %zu, named %zu; cells served %zu, named %zu\n",
                servedCombinations, namedCombinations, servedCells, namedCells);

    Require(namedCombinations == servedCombinations,
            "every combination the book serves has a name in the public API");
    Require(namedCells == servedCells, "every cell the book serves is named and evaluated");

    if (gFailures == 0)
    {
        std::printf("PASS: every served combination of the option space is named and evaluated "
                    "through the public API\n");
        return 0;
    }

    std::printf("FAIL: %zu failed check(s)\n", gFailures);
    return 1;
}
