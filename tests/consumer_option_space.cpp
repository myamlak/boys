// Consumer check: does every combination of the option space this build serves
// have a NAME a consumer can write, and does naming it evaluate that
// combination?
//
// The library's option space is four structural axes - the fit route, the
// evaluation scheme, the interval partition and the packing axis - on one
// accuracy rung, per precision. The axes are what a call site chooses once, when
// it is written; the rung is what a caller may decide per call. This check is the
// consumer's side of that split:
//
//  * for every combination the library's own book reports as served
//    (BoysAccuracyGuaranteed answering a figure for the five axes), this program
//    writes THE NAME of that combination - a policy type - and passes the rung as
//    the call's own argument. There is no string, no table, no name-to-function
//    map and no search of anything at run time anywhere in this file: one arm per
//    name, resolved where the line is written. A combination this program could
//    not name would not compile, which is why this is a consumer check and not a
//    reading. The one row of the partition axis this file writes no arm for is
//    served at every rung on one of its two routes and at the reference rung on
//    the other: an arm here names a partition and not a member of one, so it
//    would have to name the route whose rungs the library refuses beside the one
//    whose rungs it serves, and the refused one is not an instantiation that
//    exists. It is named in kUnspelledPartitions with that reason and printed
//    with the census rather than left out of it;
//
//  * every value that comes back is judged twice. It is compared against the
//    committed 45-digit reference grid within the bound the book states for the
//    combination and rung that were named - the figure BoysAccuracyGuaranteed
//    answers for the same five axes - and it is compared BIT FOR BIT against the
//    same combination reached the other way, through the entry whose multiplier is
//    its template argument. The second comparison is what catches a named entry
//    that ignores the rung it was handed: if a rung were dropped, the two
//    spellings would part company wherever that rung changes the arithmetic. The
//    number of cells where the rung moved the values is printed beside the number
//    where it did not, so a green line is not a statement about a comparison that
//    could never have failed;
//
//  * the census is printed per precision - combinations served, combinations
//    named, cells served, cells named and evaluated - and it is counted from the
//    library's own tables rather than from a list kept here. The axes this file
//    spells are checked against those tables first, so a third scheme or a third
//    partition would be reported as a gap rather than swept into an else arm.
//
// The four precisions are four different measurements and are reported apart: the
// double lane's entries take a policy at the library's own budget, the float
// lane's at its own, and the half lanes run the float lane's engine at the tighter
// fp16 budget with the half store. Where a format conversion sits between the grid
// and the entry, the comparison is made against the certified double lane at the
// same converted argument, exactly as the umbrella consumer check does it, and the
// half lanes' representation term is one half-ULP of the value they returned.
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

// The library's private src/ directory is deliberately NOT on this file's
// include path: a consumer gets include/ and nothing else. Prove it here rather
// than assume it. Each name below resolves only if a directory holding that file
// is on the include path, and those files are the library's sources; five names,
// so that renaming one does not quietly retire the probe.
#if __has_include("boys.cpp") ||                                                                   \
                  __has_include("boys_simd.cpp") ||                                                \
                                __has_include("boys_transform.cpp") ||                             \
                                              __has_include("boys_c.cpp") ||                       \
                                                            __has_include("boys_half_native.cpp")
#error                                                                                             \
    "this consumer check is compiled with src/ on its include path; it does not prove that a consumer can name the option space against the public headers alone"
#endif

namespace {

using boys::AccuracyTier;
using boys::BoysBudget;
using boys::EvalPolicy;
using boys::EvalScheme;
using boys::FitGranularity;
using boys::FitRoute;
using boys::PackAxis;

/// The reference lane's own bound, carried wherever a format conversion sits
/// between the committed grid and the entry that was named: the certified double
/// lane at m = 1 stands in for the grid at the converted argument, so a composed
/// bound is the cell's own figure plus this one.
constexpr double kOracleBound = 5.5e-14;

/// The rungs this file names, one arm per enumerator of the library's own
/// enumeration, in enumerator order.
constexpr std::array<AccuracyTier, 7> kTiers = {
    AccuracyTier::kReference,   AccuracyTier::kRelaxed64,     AccuracyTier::kRelaxed256,
    AccuracyTier::kRelaxed1024, AccuracyTier::kRelaxed4096,   AccuracyTier::kRelaxed16384,
    AccuracyTier::kRelaxed65536,
};

/// The raw enumerator value past which this file stops looking for rungs the book
/// might serve: the library's own enumeration ends well below it, and a rung
/// outside the enumeration is one no arm here could spell.
constexpr int kMaxTierProbe = 32;

/// The four structural axes, each as the enumerators this file spells an arm
/// for. The tables the library publishes are checked against these before
/// anything is counted.
constexpr std::array<FitRoute, 2> kRoutes = {FitRoute::kChebyshev, FitRoute::kRationalMinimax};
constexpr std::array<EvalScheme, 2> kSchemes = {EvalScheme::kSplitClenshaw, EvalScheme::kHorner};
constexpr std::array<FitGranularity, 2> kPartitions = {FitGranularity::kShipped,
                                                       FitGranularity::kNarrow};
constexpr std::array<PackAxis, 2> kAxes = {PackAxis::kArguments, PackAxis::kOrders};

/// The partitions the library publishes and this file writes no arm for, each
/// with the reason, because the reason is the whole of what makes the second
/// list honest rather than a list of the rows that were inconvenient.
///
/// An arm here names one cell through the entry whose *rung* is the call's own
/// argument (\c BoysAllOrdersAtTier and its single-precision sibling) beside the
/// same cell reached through the compile-time multiplier, and compares the two
/// bit for bit. The uniform partition has no such entry: its table stores one
/// degree for every order and every interval, so the rung the entry would take at
/// run time has no second value to be handed - the library refuses every rung of
/// it but the reference one where the call is named. Its cell is reached by
/// naming the multiplier at compile time, which is the other half of every pair
/// this file compares and not a pair. A partition added to the library that is
/// not one of these two is caught by the check below rather than swept into the
/// census.
struct Unspelled {
    FitGranularity partition; ///< the row this file writes no arm for
    const char* why;          ///< why no arm here can name a cell of it
};

constexpr std::array<Unspelled, 1> kUnspelledPartitions = {{
    {FitGranularity::kUniform,
     "its two routes' rungs do not agree: the Chebyshev member is served at every rung and the "
     "rational member at the reference rung alone, and an arm here names the partition rather "
     "than a member of it, so no arm could reach the served rungs without naming the refused "
     "ones - which is not an instantiation that exists"},
}};

/// The precision classes this file sweeps: one per name the library publishes a
/// figure for, with the lane whose book each class's cells are counted in and the
/// named entry its combinations are written with. The two half formats are one
/// lane and two classes, as the library's own enumeration has them.
struct ClassInfo {
    const char* name;  ///< the name the census prints
    const char* entry; ///< the named entry the class's combinations are written with
    boys::Precision lane;
};

constexpr std::array<ClassInfo, 4> kClasses = {{
    {"fp64", "BoysAllOrdersAtTier", boys::Precision::kFp64},
    {"fp32", "BoysAllOrdersF32AtTier", boys::Precision::kFp32},
    {"fp16", "BoysAllOrdersF16AtTier", boys::Precision::kFp16},
    {"bf16", "BoysAllOrdersBf16AtTier", boys::Precision::kFp16},
}};

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

/// Every k-th argument of the grid, so the sweep visits the whole of the argument
/// line rather than its head: the fits, the seed and the asymptotic form are three
/// different pieces of arithmetic behind one entry, and a combination named at one
/// argument only would not say which of them it reached.
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
    std::size_t discriminating = 0; ///< cells where the rung moved the values
};

/// A deque, not a vector: the checks hold a reference to a rule while they
/// register the next one, and a deque keeps those references valid.
std::deque<Rule> gRules;
std::size_t gPrinted = 0;
constexpr std::size_t kPrintedPerRule = 5;
std::size_t gFailures = 0;

/// The denominators the verdict prints: the values judged, the values the half
/// lanes claim no bound for, and the bit-for-bit comparisons with how many held.
/// A green line that says how much it judged is a measurement; one that says only
/// "PASS" is not.
std::size_t gJudged = 0;
std::size_t gSkipped = 0;
std::size_t gCompared = 0;
std::size_t gHeld = 0;

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

/// Judge one value: |measured - reference| against the bound the book states for
/// the combination and rung that were named. A value that is not a finite number
/// is counted as exceeded rather than compared, because a ratio against a bound
/// is false for every NaN and would otherwise slip through the check.
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

// --- the two spellings of one cell ------------------------------------------

/// The rung as the multiplier the other spelling takes as its template argument:
/// one arm per enumerator, and the multiplier in each arm is the one the library
/// publishes for that rung (AccuracyMultiplier) rather than a literal written
/// here. The two spellings of a cell are therefore compared through the library's
/// own statement of what a rung is, and a rung the ladder below reads at a
/// multiplier the library does not publish for it shows up as a disagreement.
template <typename Call>
void AtRung(AccuracyTier tier, Call&& call) noexcept {
    switch (tier)
    {
    case AccuracyTier::kReference:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kReference)>();
        return;
    case AccuracyTier::kRelaxed64:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed64)>();
        return;
    case AccuracyTier::kRelaxed256:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed256)>();
        return;
    case AccuracyTier::kRelaxed1024:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed1024)>();
        return;
    case AccuracyTier::kRelaxed4096:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed4096)>();
        return;
    case AccuracyTier::kRelaxed16384:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed16384)>();
        return;
    case AccuracyTier::kRelaxed65536:
        call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kRelaxed65536)>();
        return;
    }

    call.template operator()<boys::AccuracyMultiplier(AccuracyTier::kReference)>();
}

/// The four structural axes of one cell, each narrowed to the value that names
/// it. The nested lambdas are how this file writes one arm per name without
/// writing sixteen of them by hand; the innermost arm is the policy a call site
/// writes. Each level's else arm names the other enumerator it spells and is not
/// a default, so a value that is neither is caught by the table check rather
/// than answered here — and for the partition, which is the one axis whose third
/// member this file has no arm for, it is caught here as well: a value that
/// reached this call and is not one of the two arms is not a cell of the census
/// this dispatch may answer from the shipped partition's tables.
template <typename Call>
void AtAxes(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
            Call&& call) {
    const auto with_partition = [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack>() {
        if (partition == FitGranularity::kNarrow)
        {
            call.template operator()<kRoute, kScheme, kPack, FitGranularity::kNarrow>();
        } else if (partition == FitGranularity::kShipped)
        {
            call.template operator()<kRoute, kScheme, kPack, FitGranularity::kShipped>();
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

/// One combination named at a rung passed as the call's own argument, and the
/// same combination named at the compile-time multiplier. The pair is what this
/// file compares: the first is the name a consumer writes, the second is the same
/// cell reached through the entry whose multiplier is its template argument.

void NameFp64(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              AccuracyTier tier, int nmax, double x, double* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               boys::BoysAllOrdersAtTier<Policy>(tier, nmax, x, out);
           });
}

void SpellFp64(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
               AccuracyTier tier, int nmax, double x, double* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               AtRung(tier, [&]<double m>() { boys::BoysAllOrders<m, Policy>(nmax, x, out); });
           });
}

void NameFp32(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              AccuracyTier tier, int nmax, float x, float* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               boys::BoysAllOrdersF32AtTier<Policy>(tier, nmax, x, out);
           });
}

void SpellFp32(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
               AccuracyTier tier, int nmax, float x, float* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kPack, kPartition>;
               AtRung(tier, [&]<double m>() { boys::BoysAllOrdersF32<m, Policy>(nmax, x, out); });
           });
}

#if BoysFp16
void NameFp16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              AccuracyTier tier, int nmax, boys::F16 x, boys::F16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               boys::BoysAllOrdersF16AtTier<Policy>(tier, nmax, x, out);
           });
}

void SpellFp16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
               AccuracyTier tier, int nmax, boys::F16 x, boys::F16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               AtRung(tier, [&]<double m>() { boys::BoysAllOrdersF16<m, Policy>(nmax, x, out); });
           });
}

void NameBf16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
              AccuracyTier tier, int nmax, boys::Bf16 x, boys::Bf16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               boys::BoysAllOrdersBf16AtTier<Policy>(tier, nmax, x, out);
           });
}

void SpellBf16(FitRoute route, EvalScheme scheme, PackAxis pack, FitGranularity partition,
               AccuracyTier tier, int nmax, boys::Bf16 x, boys::Bf16* out) {
    AtAxes(route, scheme, pack, partition,
           [&]<FitRoute kRoute, EvalScheme kScheme, PackAxis kPack, FitGranularity kPartition>() {
               using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFp16, kPack, kPartition>;
               AtRung(tier, [&]<double m>() { boys::BoysAllOrdersBf16<m, Policy>(nmax, x, out); });
           });
}
#endif // BoysFp16

// --- one cell over the sweep ------------------------------------------------

/// One cell's values over the whole sweep, argument-major: values[a * (nmax + 1) + n].
struct SweepResult {
    std::vector<double> values;
};

/// Name one cell at the rung that was passed, evaluate it at every argument of the
/// sweep, judge every value against the book's figure for the combination and the
/// rung, and compare every value with the same cell's compile-time spelling.
SweepResult SweepCell(std::size_t classIndex,
                      Rule& bound,
                      Rule& spellings,
                      FitRoute route,
                      EvalScheme scheme,
                      PackAxis pack,
                      FitGranularity partition,
                      AccuracyTier tier,
                      const std::vector<Cell>& grid,
                      const std::vector<double>& args) {
    constexpr int kNmax = boys::kMaxBoysOrder;
    constexpr std::size_t kWidth = static_cast<std::size_t>(kNmax) + 1;

    SweepResult result;
    result.values.assign(args.size() * kWidth, 0.0);

    const boys::AccuracyFigure figure =
        boys::BoysAccuracyGuaranteed(kClasses[classIndex].lane, route, scheme, pack, partition,
                                     tier);

    std::vector<double> value(kWidth);
    std::vector<double> spelled(kWidth);

    for (std::size_t a = 0; a < args.size(); ++a)
    {
        const double x = args[a];

        if (classIndex == 0)
        {
            NameFp64(route, scheme, pack, partition, tier, kNmax, x, value.data());
            SpellFp64(route, scheme, pack, partition, tier, kNmax, x, spelled.data());

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
            std::vector<float> fspelled(kWidth);

            NameFp32(route, scheme, pack, partition, tier, kNmax, x32, fvalue.data());
            SpellFp32(route, scheme, pack, partition, tier, kNmax, x32, fspelled.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                const std::size_t k = static_cast<std::size_t>(n);
                value[k] = static_cast<double>(fvalue[k]);
                spelled[k] = static_cast<double>(fspelled[k]);

                // The lane rounds its argument before it evaluates, so the
                // reference is the certified double lane at the argument the lane
                // was actually asked about, and the composed bound is the cell's
                // figure plus that lane's own.
                const double reference =
                    boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(n, static_cast<double>(x32));

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
                std::vector<boys::Bf16> bspelled(kWidth);
                const boys::Bf16 h(x32);

                NameBf16(route, scheme, pack, partition, tier, kNmax, h, bvalue.data());
                SpellBf16(route, scheme, pack, partition, tier, kNmax, h, bspelled.data());

                for (int n = 0; n <= kNmax; ++n)
                {
                    const std::size_t k = static_cast<std::size_t>(n);
                    value[k] = static_cast<double>(bvalue[k]);
                    spelled[k] = static_cast<double>(bspelled[k]);

                    const double reference = boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(
                        n, static_cast<double>(static_cast<float>(h)));
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
                std::vector<boys::F16> hspelled(kWidth);
                const boys::F16 h(x32);

                NameFp16(route, scheme, pack, partition, tier, kNmax, h, hvalue.data());
                SpellFp16(route, scheme, pack, partition, tier, kNmax, h, hspelled.data());

                for (int n = 0; n <= kNmax; ++n)
                {
                    const std::size_t k = static_cast<std::size_t>(n);
                    value[k] = static_cast<double>(hvalue[k]);
                    spelled[k] = static_cast<double>(hspelled[k]);

                    const double reference = boys::BoysSingle<boys::kBoysFullAccuracyMultiplier>(
                        n, static_cast<double>(static_cast<float>(h)));
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
            // A build whose fp16 seam is closed carries neither half class: the
            // census reports them as not carried rather than sweeping an empty
            // arm, and this program's rules for them judge nothing.
            std::fflush(stdout);
#endif // BoysFp16
        }

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t k = static_cast<std::size_t>(n);

            ++gCompared;
            ++spellings.cells;

            if (value[k] == spelled[k])
            {
                ++gHeld;
            } else
            {
                ++gFailures;
                ++spellings.exceeded;
                Fail(spellings, n, x, value[k], spelled[k], 0.0);
            }

            result.values[a * kWidth + k] = value[k];
        }
    }

    return result;
}

// --- the axes the library publishes -----------------------------------------

/// The distinct route values of a lane's route table, which is one row per route
/// and region.
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

/// Check the tables this file spells its arms from against the tables the library
/// publishes, so that an axis added to the library is a counted gap here rather
/// than a value swept into an else arm.
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

    // The partitions are checked row by row rather than by count, because the
    // two lists are not the same kind of list any more: every row the library
    // publishes is either an arm this file writes or a row named in
    // kUnspelledPartitions with the reason no arm here can name a cell of it,
    // and a third kind of row - one this file neither spells nor accounts for -
    // fails the check instead of quietly leaving the census short of the
    // library.
    for (const boys::FitGranularityInfo& row : boys::BoysFitGranularities())
    {
        const bool spelled =
            std::find(kPartitions.begin(), kPartitions.end(), row.granularity) != kPartitions.end();
        const bool accounted =
            std::any_of(kUnspelledPartitions.begin(), kUnspelledPartitions.end(),
                        [granularity = row.granularity](const Unspelled& unspelled) {
                            return unspelled.partition == granularity;
                        });

        Require(spelled || accounted,
                "every published partition is an arm this file writes or a row it accounts for");

        Require(!(spelled && accounted),
                "no published partition is both an arm this file writes and an accounted row");
    }
}

// --- one class's census -----------------------------------------------------

Census RunClass(std::size_t classIndex, const std::vector<Cell>& grid,
                const std::vector<double>& args) {
    Census census;

    const std::string prefix = std::string(kClasses[classIndex].name) + " ";

    Rule& bound = NewRule(prefix + kClasses[classIndex].entry + " (values above the bound)");
    Rule& spellings =
        NewRule(prefix + "rung honoured (named value vs the same cell at compile time)");

    for (const FitRoute route : kRoutes)
    {
        for (const EvalScheme scheme : kSchemes)
        {
            for (const PackAxis pack : kAxes)
            {
                for (const FitGranularity partition : kPartitions)
                {
                    // What the book serves for this combination: every rung of the
                    // enumeration the library answers a figure for, and every rung
                    // outside it that the library serves and no arm here can spell.
                    // The two are counted apart, because only one of them has a
                    // name.
                    std::size_t servedHere = 0;

                    for (int raw = 0; raw <= kMaxTierProbe; ++raw)
                    {
                        const auto tier = static_cast<AccuracyTier>(raw);

                        if (boys::BoysAccuracyGuaranteed(kClasses[classIndex].lane, route, scheme,
                                                         pack, partition, tier)
                                .available)
                        {
                            ++servedHere;
                        }
                    }

                    if (servedHere > 0)
                    {
                        ++census.servedCombinations;

                        // Every combination the book serves is one of the arms this
                        // file writes: the axes are the enumerations the table check
                        // above held, and the count of arms is their product.
                        ++census.namedCombinations;
                    }

                    census.servedCells += servedHere;

                    std::vector<double> referenceRung;

                    for (const AccuracyTier tier : kTiers)
                    {
                        const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
                            kClasses[classIndex].lane, route, scheme, pack, partition, tier);

                        if (!figure.available)
                        {
                            continue;
                        }

                        ++census.namedCells;

                        const SweepResult result = SweepCell(classIndex, bound, spellings, route,
                                                             scheme, pack, partition, tier, grid,
                                                             args);

                        if (tier == AccuracyTier::kReference)
                        {
                            referenceRung = result.values;
                        } else if (!referenceRung.empty())
                        {
                            for (std::size_t i = 0; i < result.values.size(); ++i)
                            {
                                if (result.values[i] != referenceRung[i])
                                {
                                    ++census.discriminating;
                                }
                            }
                        }
                    }
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

    std::printf("boys consumer option space: %zu grid cells, %zu arguments swept, %zu rungs, "
                "%zu combinations per class\n",
                grid.size(), args.size(), kTiers.size(),
                kRoutes.size() * kSchemes.size() * kAxes.size() * kPartitions.size());

    std::array<Census, 4> censuses{};

    for (std::size_t i = 0; i < kClasses.size(); ++i)
    {
#if !BoysFp16
        // A build whose seam is closed declares neither half lane, so neither
        // half class is carried here and there is nothing to name: the census says
        // so rather than counting cells this build has no entry for.
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

    // The partitions the counts above do not cover, named with the reason, so
    // that "every cell the book serves is named" is not read as a claim about
    // cells this file writes no arm for. A row neither counted above nor listed
    // here failed the table check rather than being dropped from the census.
    std::printf("\npartitions this file writes no arm for, and why:\n");

    for (const Unspelled& unspelled : kUnspelledPartitions)
    {
        for (const boys::FitGranularityInfo& row : boys::BoysFitGranularities())
        {
            if (row.granularity == unspelled.partition)
            {
                std::printf("  %-8s %s\n", row.name, unspelled.why);
            }
        }
    }

    std::printf("\nrules:\n");
    PrintRules();

    std::printf("\nvalues judged %zu, skipped below the half lanes' floor %zu\n", gJudged,
                gSkipped);
    std::printf("bit-for-bit: %zu comparisons, %zu held, over cells where the rung moved the "
                "values: ",
                gCompared, gHeld);
    std::size_t discriminating = 0;

    for (const Census& census : censuses)
    {
        discriminating += census.discriminating;
    }

    std::printf("%zu\n", discriminating);

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
