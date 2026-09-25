// The run-time accuracy tier's contract tests: AccuracyTier, QueryTier,
// TierCoverage, AccuracyRegion, AccuracyComponent and BoysAllOrdersAtTier -
// and, in section 5, the same surface's fit routes: FitRoute, FitRouteInfo,
// BoysFitRoutes() and BoysAllOrdersWithRoute.
//
// The tier is a selector: it names one of the multipliers this kernel already
// instantiates and routes a call to that instantiation at run time. Four
// things can go wrong with a selector, and each gets its own test:
//
//  1. MISMAPPING. A requested rung silently reaches a different multiplier.
//     The tier entry must be BIT-IDENTICAL to the compile-time instantiation at
//     the rung's own multiplier, everywhere, for every order. One exact
//     comparison and no tolerance: a tolerance here is exactly the slack a
//     wrong rung could hide in. Asserted twice - against the library's own
//     instantiation (routed by the extern-template declarations below, so the
//     comparison carries no per-translation-unit codegen confound) and against
//     an instantiation this TU compiled for itself.
//
//  2. BOUND NOT MET. Each rung must hold the bound the code declares for it.
//     The declaration is transcribed below with its source and re-derived from
//     QueryTier at run time, so a change to either fails this file instead of
//     drifting.
//
//  3. DISHONEST QUERY. QueryTier exists so a caller can ask in advance what a
//     tier delivers, so a report claiming coverage the code does not have is
//     the defect this file is sharpest about: the query's `reachable` is
//     asserted to be an upper bound on the MEASURED error, and its `meets` is
//     asserted against the delivered value at the tolerance it answers for.
//
//  4. SILENT SUBSTITUTION OR SILENT NON-WRITE. An unavailable tier must not
//     return a value at an accuracy the caller did not ask for, and must not
//     leave the output unwritten. The documented fallback is the reference
//     multiplier, which is never coarser than any tier a caller could name, so
//     a garbage enumerator is asserted to select the reference rung exactly -
//     and to leave no sentinel standing in the output.
//
// Two guards against a green result that proves nothing. Both are counted and
// printed rather than folded into a pass total:
//  * a cell whose bound exceeds the function's own magnitude is BOUND-COVERED:
//    returning zero would pass it, so it cannot distinguish the tier from a
//    stub, and it is counted as a vacuous pass;
//  * a cell where two rungs agree bitwise cannot distinguish those rungs. The
//    rungs are m-invariant in region C by construction (the asymptotic form has
//    no coefficients to truncate), so a region-C-only sweep can never detect a
//    mis-mapping. RungsAreDistinguishableWhereTheTierVaries measures where they
//    do differ.
//
// Reference: the committed high-precision grid is located at run time
// (BoysTierReference, falling back to BoysDataDir/boys_reference.csv) and its
// provenance is printed, so every claim below is attributed to a named
// reference rather than to the library under test.

#include "boys/boys.hpp"
#include "boys/boys_impl.hpp" // kX0/kX1, the region kernels, BoysAllOrdersImpl

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <vector>

namespace boys {

// The six relaxed instantiations are defined in src/boys.cpp (commit 2f81bb8).
// Declaring them extern here routes the reference calls in this file to the
// library's own compiled code, so the bit-identity test compares the library's
// dispatch against the library's instantiation rather than against a second
// copy this translation unit compiled for itself.
extern template void BoysAllOrders<64.0>(int nmax, double x, double* out) noexcept;
extern template void BoysAllOrders<256.0>(int nmax, double x, double* out) noexcept;
extern template void BoysAllOrders<1024.0>(int nmax, double x, double* out) noexcept;
extern template void BoysAllOrders<4096.0>(int nmax, double x, double* out) noexcept;
extern template void BoysAllOrders<16384.0>(int nmax, double x, double* out) noexcept;
extern template void BoysAllOrders<65536.0>(int nmax, double x, double* out) noexcept;

} // namespace boys

namespace {

using boys::AccuracyComponent;
using boys::AccuracyMultiplier;
using boys::AccuracyRegion;
using boys::AccuracyTier;
using boys::BoysAllOrders;
using boys::BoysAllOrdersAtTier;
using boys::BoysAllOrdersWithRoute;
using boys::QueryTier;
using boys::TierCoverage;
using boys::detail::BoysAllOrdersImpl;
using boys::detail::kX0;
using boys::detail::kX1;

// ---------------------------------------------------------------------------
// The declared contract, transcribed from include/boys/boys.hpp.
// ---------------------------------------------------------------------------

// BoysAllOrdersAtTier's own declaration: the batch entry's contract - the
// contract table's "double batch" row - which is m*5.5e-14 in all three
// regions. Not the three-value per-region column (m*1e-15 in region A), which
// belongs to the single lane: the tier entry dispatches to BoysAllOrders, and
// TheEntryHoldsTheBatchRowAndNotThePerRegionColumn measures it exceeding that
// column by 3.2x at m = 1 alone.
//
// Not taken on trust: TheTranscribedBoundIsTheOneQueryTierDeclares re-derives
// the base from QueryTier, which restates it as its own reachable. If either
// declaration moves, the two disagree and that test fails.
constexpr double kDeclaredBatchBase = 5.5e-14;

// The region C entry of that table, restated by QueryTier as an m-independent
// reachable: the asymptotic branch has no coefficients to truncate.
constexpr double kDeclaredAsymptotic = 5.5e-14;

// The seven rungs and the multiplier each enumerator names.
struct Rung {
    AccuracyTier tier;
    double multiplier;
    const char* name;
};

constexpr std::array<Rung, 7> kRungs{{
    {AccuracyTier::kReference, boys::kBoysFullAccuracyMultiplier, "kReference"},
    {AccuracyTier::kRelaxed64, 64.0, "kRelaxed64"},
    {AccuracyTier::kRelaxed256, 256.0, "kRelaxed256"},
    {AccuracyTier::kRelaxed1024, 1024.0, "kRelaxed1024"},
    {AccuracyTier::kRelaxed4096, 4096.0, "kRelaxed4096"},
    {AccuracyTier::kRelaxed16384, 16384.0, "kRelaxed16384"},
    {AccuracyTier::kRelaxed65536, 65536.0, "kRelaxed65536"},
}};

constexpr std::array<AccuracyRegion, 3> kRegions{{
    AccuracyRegion::kA,
    AccuracyRegion::kB,
    AccuracyRegion::kC,
}};

// The compile-time path at a multiplier, compiled by the library and reached
// through the library's exported instantiation.
template <double M> void LibraryPath(int nmax, double x, double* out) noexcept {
    BoysAllOrders<M>(nmax, x, out);
}

// The same code compiled in this translation unit, for the cross-TU check.
template <double M> void LocalPath(int nmax, double x, double* out) noexcept {
    BoysAllOrdersImpl<M, boys::EvalPolicy<>>(nmax, x, out);
}

using PathFn = void (*)(int, double, double*) noexcept;

// Indexed by rung, in kRungs order.
constexpr std::array<PathFn, 7> kLibraryPaths{{
    &LibraryPath<boys::kBoysFullAccuracyMultiplier>,
    &LibraryPath<64.0>,
    &LibraryPath<256.0>,
    &LibraryPath<1024.0>,
    &LibraryPath<4096.0>,
    &LibraryPath<16384.0>,
    &LibraryPath<65536.0>,
}};

constexpr std::array<PathFn, 7> kLocalPaths{{
    &LocalPath<boys::kBoysFullAccuracyMultiplier>,
    &LocalPath<64.0>,
    &LocalPath<256.0>,
    &LocalPath<1024.0>,
    &LocalPath<4096.0>,
    &LocalPath<16384.0>,
    &LocalPath<65536.0>,
}};

// ---------------------------------------------------------------------------
// The reference grid
// ---------------------------------------------------------------------------

struct Grid {
    std::vector<double> xs; // the distinct arguments, ascending
    // values[n][i] = F_n(xs[i]); dense, so a sweep lookup is O(1).
    std::array<std::vector<double>, boys::kMaxBoysOrder + 1> values{};
    std::string provenance;
    std::size_t rows = 0;
    std::size_t missing = 0;

    bool Load(const std::string& path, const std::string& label) {
        std::ifstream file(path);

        if (!file)
        {
            return false;
        }

        std::vector<std::array<double, 3>> raw; // {n, x, value}
        std::string line;
        std::getline(file, line); // header

        while (std::getline(file, line))
        {
            std::stringstream ss(line);
            std::string cell;
            std::array<double, 3> row{};
            bool complete = true;

            for (int c = 0; c < 3; ++c)
            {
                if (!std::getline(ss, cell, ','))
                {
                    complete = false;
                    break;
                }

                row[static_cast<std::size_t>(c)] = std::strtod(cell.c_str(), nullptr);
            }

            if (complete)
            {
                raw.push_back(row);
            }
        }

        if (raw.empty())
        {
            return false;
        }

        provenance = label + " [" + path + "]";
        rows = raw.size();

        xs.reserve(raw.size());

        for (const std::array<double, 3>& row : raw)
        {
            xs.push_back(row[1]);
        }

        std::sort(xs.begin(), xs.end());
        xs.erase(std::unique(xs.begin(), xs.end()), xs.end());

        for (auto& v : values)
        {
            v.assign(xs.size(), std::nan(""));
        }

        for (const std::array<double, 3>& row : raw)
        {
            const std::size_t n = static_cast<std::size_t>(row[0]);

            if (n > static_cast<std::size_t>(boys::kMaxBoysOrder))
            {
                continue;
            }

            const auto it = std::lower_bound(xs.begin(), xs.end(), row[1]);
            values[n][static_cast<std::size_t>(it - xs.begin())] = row[2];
        }

        for (const auto& v : values)
        {
            for (double value : v)
            {
                if (std::isnan(value))
                {
                    ++missing;
                }
            }
        }

        return true;
    }
};

// Loaded once per process. The gate lane's grid when the tree carries it, else
// the suite's; the provenance is printed rather than assumed.
const Grid& Reference() {
    static const Grid grid = [] {
        Grid g;

        if (g.Load(BoysTierReference, "accuracy-gate grid"))
        {
            return g;
        }

        g = Grid{};
        const std::string suite = std::string(BoysDataDir) + "/boys_reference.csv";

        if (!g.Load(suite, "suite grid"))
        {
            ADD_FAILURE() << "no reference grid: tried " << BoysTierReference << " and " << suite;
        }

        return g;
    }();

    return grid;
}

AccuracyRegion RegionOf(double x) {
    if (x < kX0)
    {
        return AccuracyRegion::kA;
    }

    return x < kX1 ? AccuracyRegion::kB : AccuracyRegion::kC;
}

const char* RegionName(AccuracyRegion region) {
    switch (region)
    {
    case AccuracyRegion::kA:
        return "A";

    case AccuracyRegion::kB:
        return "B";

    case AccuracyRegion::kC:
        return "C";
    }

    return "?";
}

// What QueryTier states for a rung and region, at a tolerance the comparison
// cannot itself satisfy: reachable is the reach of the report.
double DeclaredReachable(AccuracyTier tier, AccuracyRegion region) {
    return QueryTier(tier, region, 0.0).reachable;
}

// The floor under which a region's reference coverage cannot support the bound
// claim it is being used for. Region A carries the per-order Chebyshev pieces,
// region B the F0 seed's truncation range and region C the asymptotic onset;
// each needs many arguments and a wrong tier shows up at specific ones (a piece
// boundary, the seed's amplification maximum, the onset). The accuracy gate's
// grid carries 646 / 248 / 544 arguments. The suite grid's x set carries
// 73 / 5 / 10 - which is why a region below this floor is reported as NOT
// CHECKED rather than counted as a pass.
constexpr std::size_t kMinArgumentsPerRegion = 32;

std::size_t ArgumentCount(AccuracyRegion region) {
    const Grid& grid = Reference();
    std::size_t count = 0;

    for (double x : grid.xs)
    {
        if (RegionOf(x) == region)
        {
            ++count;
        }
    }

    return count;
}

bool RegionIsCovered(AccuracyRegion region) {
    return ArgumentCount(region) >= kMinArgumentsPerRegion;
}

// ---------------------------------------------------------------------------
// The sweeps
// ---------------------------------------------------------------------------

// One measurement cell: the error, where the worst one is, and whether the
// cell can distinguish anything.
struct Sweep {
    std::size_t cells = 0;
    std::size_t met = 0;
    std::size_t bound_covered = 0; // met, but returning zero would pass too
    double worst = 0.0;
    int worst_n = -1;
    double worst_x = 0.0;

    void Note(double error, int n, double x, double magnitude, double bound) {
        ++cells;

        if (error > worst)
        {
            worst = error;
            worst_n = n;
            worst_x = x;
        }

        if (error <= bound)
        {
            ++met;

            if (bound > std::fabs(magnitude))
            {
                ++bound_covered;
            }
        }
    }
};

// Every argument of the reference grid in the region, every nmax in 0..32,
// every order of the returned batch, against the reference value. No sampling:
// a gap here is a cell a wrong rung could hide in.
Sweep RunSweep(AccuracyTier tier, AccuracyRegion region, double bound) {
    const Grid& grid = Reference();
    Sweep sweep;

    if (grid.xs.empty())
    {
        return sweep;
    }

    std::vector<double> out(boys::kMaxBoysOrder + 1, std::nan(""));

    for (std::size_t i = 0; i < grid.xs.size(); ++i)
    {
        const double x = grid.xs[i];

        if (RegionOf(x) != region)
        {
            continue;
        }

        for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
        {
            std::fill(out.begin(), out.end(), std::nan(""));
            BoysAllOrdersAtTier(tier, nmax, x, out.data());

            for (int k = 0; k <= nmax; ++k)
            {
                const double reference = grid.values[static_cast<std::size_t>(k)][i];

                if (std::isnan(reference))
                {
                    continue;
                }

                sweep.Note(std::fabs(out[static_cast<std::size_t>(k)] - reference),
                           k,
                           x,
                           reference,
                           bound);
            }
        }
    }

    return sweep;
}

// The sweeps are shared by the accuracy test, the reachable-upper-bound test
// and the honesty test; computed once, against the bound the code declares.
const Sweep& SweepFor(std::size_t rung, AccuracyRegion region) {
    static std::vector<Sweep> cache; // [rung * 3 + region]

    if (cache.empty())
    {
        cache.resize(kRungs.size() * kRegions.size());

        for (std::size_t r = 0; r < kRungs.size(); ++r)
        {
            for (std::size_t g = 0; g < kRegions.size(); ++g)
            {
                cache[r * kRegions.size() + g] = RunSweep(
                    kRungs[r].tier, kRegions[g], DeclaredReachable(kRungs[r].tier, kRegions[g]));
            }
        }
    }

    return cache[rung * kRegions.size() + static_cast<std::size_t>(region)];
}

bool BitwiseEqual(const double* a, const double* b, int count) {
    for (int k = 0; k <= count; ++k)
    {
        if (std::bit_cast<std::uint64_t>(a[k]) != std::bit_cast<std::uint64_t>(b[k]))
        {
            return false;
        }
    }

    return true;
}

std::size_t CountDifferingCells(PathFn a, PathFn b, AccuracyRegion region) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        return 0;
    }

    std::vector<double> outa(boys::kMaxBoysOrder + 1, std::nan(""));
    std::vector<double> outb(boys::kMaxBoysOrder + 1, std::nan(""));
    std::size_t differ = 0;

    for (double x : grid.xs)
    {
        if (RegionOf(x) != region)
        {
            continue;
        }

        for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
        {
            std::fill(outa.begin(), outa.end(), std::nan(""));
            std::fill(outb.begin(), outb.end(), std::nan(""));
            a(nmax, x, outa.data());
            b(nmax, x, outb.data());

            if (!BitwiseEqual(outa.data(), outb.data(), nmax))
            {
                ++differ;
            }
        }
    }

    return differ;
}

// ---------------------------------------------------------------------------
// The declared-bound transcription, checked against the declaration
// ---------------------------------------------------------------------------

TEST(Tier, TheTranscribedBoundIsTheOneQueryTierDeclares) {
    EXPECT_DOUBLE_EQ(DeclaredReachable(AccuracyTier::kReference, AccuracyRegion::kA),
                     kDeclaredBatchBase)
        << "QueryTier's region A reachable no longer matches the transcribed contract "
           "table entry - re-read include/boys/boys.hpp";
    EXPECT_DOUBLE_EQ(DeclaredReachable(AccuracyTier::kReference, AccuracyRegion::kB),
                     kDeclaredBatchBase)
        << "QueryTier's region B reachable no longer matches the transcribed contract "
           "table entry";
    EXPECT_DOUBLE_EQ(DeclaredReachable(AccuracyTier::kReference, AccuracyRegion::kC),
                     kDeclaredAsymptotic)
        << "QueryTier's region C reachable no longer matches the transcribed contract "
           "table entry";
}

// ---------------------------------------------------------------------------
// 1. The rung-to-instantiation mapping
// ---------------------------------------------------------------------------

TEST(Tier, MappingMatchesTheDeclaredMultiplier) {
    // AccuracyMultiplier is documented as the multiplier a tier names and the
    // enumerators state the rungs. A disagreement is the mis-mapping caught
    // before any number is computed.
    for (const Rung& rung : kRungs)
    {
        EXPECT_DOUBLE_EQ(AccuracyMultiplier(rung.tier), rung.multiplier)
            << rung.name << " names m = " << rung.multiplier << " but AccuracyMultiplier reports "
            << AccuracyMultiplier(rung.tier);
        EXPECT_GT(AccuracyMultiplier(rung.tier), 0.0) << rung.name;
    }
}

TEST(Tier, TierIsBitIdenticalToTheCompileTimePathAtItsMultiplier) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::printf("\nreference: %s\n%zu rows, %zu arguments, %zu absent (n, x) cells\n",
                grid.provenance.c_str(),
                grid.rows,
                grid.xs.size(),
                grid.missing);
    std::printf("%-15s %-7s %10s %10s %10s %10s\n",
                "rung",
                "region",
                "mismatch",
                "cells",
                "cross-TU",
                "cells");

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const Rung& rung = kRungs[r];
        std::vector<double> got(boys::kMaxBoysOrder + 1, std::nan(""));
        std::vector<double> want(boys::kMaxBoysOrder + 1, std::nan(""));

        for (AccuracyRegion region : kRegions)
        {
            std::size_t mismatch = 0;
            std::size_t cells = 0;
            std::size_t cross_tu = 0;
            std::size_t cross_cells = 0;
            double first_x = 0.0;
            int first_nmax = -1;

            for (double x : grid.xs)
            {
                if (RegionOf(x) != region)
                {
                    continue;
                }

                for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
                {
                    std::fill(got.begin(), got.end(), std::nan(""));
                    std::fill(want.begin(), want.end(), std::nan(""));
                    BoysAllOrdersAtTier(rung.tier, nmax, x, got.data());
                    kLibraryPaths[r](nmax, x, want.data());
                    ++cells;

                    if (!BitwiseEqual(got.data(), want.data(), nmax))
                    {
                        if (mismatch == 0)
                        {
                            first_nmax = nmax;
                            first_x = x;
                        }

                        ++mismatch;
                    }

                    std::fill(want.begin(), want.end(), std::nan(""));
                    kLocalPaths[r](nmax, x, want.data());
                    ++cross_cells;

                    if (!BitwiseEqual(got.data(), want.data(), nmax))
                    {
                        ++cross_tu;
                    }
                }
            }

            std::printf("%-15s %-7s %10zu %10zu %10zu %10zu\n",
                        rung.name,
                        RegionName(region),
                        mismatch,
                        cells,
                        cross_tu,
                        cross_cells);

            EXPECT_EQ(mismatch, 0U)
                << rung.name << " (m = " << rung.multiplier << ") is not bit-identical to the "
                << "compile-time instantiation in region " << RegionName(region)
                << "; first mismatch at nmax = " << first_nmax << ", x = " << first_x;
            EXPECT_EQ(cross_tu, 0U)
                << rung.name << " dispatch disagrees with the instantiation compiled in this "
                << "translation unit in region " << RegionName(region)
                << " - the two TUs' codegen differs, so the comparison above needs re-reading";
        }
    }
}

TEST(Tier, RungsAreDistinguishableWhereTheTierVaries) {
    // The identity test above proves nothing where two rungs produce the same
    // bits. Region C is m-invariant by construction - the asymptotic form has
    // no coefficients to truncate - so a region-C-only sweep could not tell a
    // correct dispatch from a fixed rung. This measures where the rungs
    // actually differ, per region, so the identity test's reach is stated
    // rather than assumed.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::printf("\n%-7s %-24s %20s\n", "region", "cell pairs that differ", "reference vs m=65536");

    std::size_t total_differ_ab = 0;

    for (AccuracyRegion region : kRegions)
    {
        const std::size_t differ = CountDifferingCells(kLibraryPaths[0], kLibraryPaths[6], region);
        std::size_t cells = 0;

        for (double x : grid.xs)
        {
            if (RegionOf(x) == region)
            {
                cells += boys::kMaxBoysOrder + 1;
            }
        }

        std::printf("%-7s %10zu / %-10zu %20zu\n", RegionName(region), differ, cells, cells);

        if (region != AccuracyRegion::kC)
        {
            total_differ_ab += differ;
        }
    }

    // Regions A and B carry the relaxable resource, so the rungs must differ
    // there - otherwise the identity test cannot distinguish any two of them.
    EXPECT_GT(total_differ_ab, 0U)
        << "no rung pair differs in regions A or B: the identity test cannot tell the "
           "rungs apart, so its pass would be vacuous";
}

// ---------------------------------------------------------------------------
// 2. The declared per-rung bound
// ---------------------------------------------------------------------------

TEST(Tier, EveryRungMeetsItsDeclaredBoundOnTheReferenceGrid) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::printf("\n%-15s %-7s %12s %12s %10s %10s %10s\n",
                "rung",
                "region",
                "worst error",
                "declared",
                "ratio",
                "cells",
                "bound-cov");

    std::printf("reference: %s\n", grid.provenance.c_str());

    for (AccuracyRegion region : kRegions)
    {
        std::printf("region %s coverage: %zu arguments%s\n",
                    RegionName(region),
                    ArgumentCount(region),
                    RegionIsCovered(region) ? "" : "  <-- NOT CHECKED (below the floor of 32)");
    }

    std::size_t not_checked = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const Rung& rung = kRungs[r];

        for (AccuracyRegion region : kRegions)
        {
            const double bound = DeclaredReachable(rung.tier, region);
            const Sweep& sweep = SweepFor(r, region);
            const double ratio = sweep.worst > 0.0 ? sweep.worst / bound : 0.0;

            std::printf("%-15s %-7s %12.4e %12.4e %10.4f %10zu %10zu%s\n",
                        rung.name,
                        RegionName(region),
                        sweep.worst,
                        bound,
                        ratio,
                        sweep.cells,
                        sweep.bound_covered,
                        RegionIsCovered(region) ? "" : "  NOT CHECKED");

            if (!RegionIsCovered(region))
            {
                // Not a pass and not a failure: the reference in the tree
                // cannot check this region's claim, so the region is reported
                // as unchecked rather than counted either way.
                ++not_checked;
                continue;
            }

            EXPECT_LE(sweep.worst, bound)
                << rung.name << " region " << RegionName(region) << ": worst error " << sweep.worst
                << " at n = " << sweep.worst_n << ", x = " << sweep.worst_x
                << " exceeds the declared bound " << bound << " (m = " << rung.multiplier << ")";
        }
    }

    std::printf("%zu of %zu (rung, region) bound claims were not checked: the reference in the "
                "tree is too thin in those regions\n",
                not_checked,
                kRungs.size() * kRegions.size());
}

// ---------------------------------------------------------------------------
// 3. The query surface, against the delivered values
// ---------------------------------------------------------------------------

TEST(Tier, QueryReachableIsAnUpperBoundOnTheDeliveredError) {
    // `reachable` is documented as the largest error the tier can deliver in
    // the region. If the code delivers more, the query has told the caller it
    // reaches an accuracy it does not have - a report claiming coverage that
    // is not there, which is the defect this test is for.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (AccuracyRegion region : kRegions)
        {
            const TierCoverage coverage = QueryTier(kRungs[r].tier, region, 0.0);
            const Sweep& sweep = SweepFor(r, region);

            // A thin region can only understate the sweep's maximum, so an
            // upper-bound check there is a vacuous pass rather than evidence.
            if (!RegionIsCovered(region))
            {
                std::printf("NOT CHECKED: %s region %s (reference coverage %zu arguments)\n",
                            kRungs[r].name,
                            RegionName(region),
                            ArgumentCount(region));
                continue;
            }

            EXPECT_LE(sweep.worst, coverage.reachable)
                << kRungs[r].name << " region " << RegionName(region)
                << ": QueryTier reports reachable = " << coverage.reachable << " but the code "
                << "delivers " << sweep.worst << " at n = " << sweep.worst_n
                << ", x = " << sweep.worst_x << " - the report claims coverage it does not have";
        }
    }
}

TEST(Tier, QueryMeetsIsTheToleranceComparisonAndItsAnswerIsHonest) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    // Tolerances chosen to land on, just under and just over each rung's
    // reachable, plus the extremes of the documented range.
    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (AccuracyRegion region : kRegions)
        {
            const TierCoverage at_zero = QueryTier(kRungs[r].tier, region, 0.0);
            const double reachable = at_zero.reachable;
            const Sweep& sweep = SweepFor(r, region);

            // The delivered-error side of this comparison is only as strong as
            // the region's reference coverage; a thin region is not counted.
            const bool region_checked = RegionIsCovered(region);

            const std::array<double, 9> tolerances{{
                0.0,
                reachable * 0.25,
                reachable * 0.999999,
                reachable,
                reachable * 1.000001,
                reachable * 4.0,
                1e-9,
                1e-7,
                1.0,
            }};

            for (double tolerance : tolerances)
            {
                const TierCoverage coverage = QueryTier(kRungs[r].tier, region, tolerance);

                // The answer is the threshold comparison, not a second rule.
                EXPECT_EQ(coverage.meets, reachable <= tolerance)
                    << kRungs[r].name << " region " << RegionName(region)
                    << ": meets = " << coverage.meets << " at tolerance " << tolerance
                    << " does not follow from reachable = " << reachable;
                EXPECT_DOUBLE_EQ(coverage.reachable, reachable)
                    << "reachable must not depend on the tolerance asked about";

                // The honesty direction: a report of coverage is only as good
                // as the delivered error behind it.
                if (region_checked && coverage.meets && tolerance < reachable)
                {
                    EXPECT_LE(sweep.worst, tolerance)
                        << kRungs[r].name << " region " << RegionName(region)
                        << ": QueryTier reports the tier meets a tolerance of " << tolerance
                        << " but the delivered error reaches " << sweep.worst;
                }
            }
        }
    }
}

// The worst error at one argument, over every order of the returned batch.
// Cached per rung so the per-argument honesty test below is one pass.
const std::vector<double>& PerArgumentWorst(std::size_t rung) {
    static std::vector<std::vector<double>> cache;

    if (cache.empty())
    {
        const Grid& grid = Reference();
        cache.resize(kRungs.size());

        for (std::size_t r = 0; r < kRungs.size(); ++r)
        {
            std::vector<double>& worst = cache[r];

            if (grid.xs.empty())
            {
                continue;
            }

            worst.assign(grid.xs.size(), 0.0);
            std::vector<double> out(boys::kMaxBoysOrder + 1, std::nan(""));

            for (std::size_t i = 0; i < grid.xs.size(); ++i)
            {
                BoysAllOrdersAtTier(kRungs[r].tier, boys::kMaxBoysOrder, grid.xs[i], out.data());

                for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
                {
                    const double reference = grid.values[static_cast<std::size_t>(k)][i];

                    if (std::isnan(reference))
                    {
                        continue;
                    }

                    worst[i] =
                        std::max(worst[i], std::fabs(out[static_cast<std::size_t>(k)] - reference));
                }
            }
        }
    }

    return cache[rung];
}

TEST(Tier, QueryAnsweredForAnArgumentIsHonestAboutThatArgument) {
    // The sharpest form of the query's promise, and the one that catches a
    // report answered for the wrong region: asking about THIS argument must
    // give a reachable that bounds THIS argument's delivered error, and a
    // `meets` that the delivered error actually satisfies. A caller who names
    // the region by hand can produce a report about a different region; the
    // argument-taking entry cannot.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::size_t overclaimed = 0;
    std::size_t dishonest_meets = 0;
    std::size_t cells = 0;
    double first_overclaim_x = 0.0;
    int first_overclaim_rung = -1;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const std::vector<double>& worst = PerArgumentWorst(r);

        for (std::size_t i = 0; i < grid.xs.size(); ++i)
        {
            const double x = grid.xs[i];
            ++cells;

            const TierCoverage coverage = QueryTier(kRungs[r].tier, x, 0.0);

            if (worst[i] > coverage.reachable)
            {
                if (overclaimed == 0)
                {
                    first_overclaim_x = x;
                    first_overclaim_rung = static_cast<int>(r);
                }

                ++overclaimed;
            }

            // The tolerance the report itself says is reachable must be met by
            // the value delivered at this argument.
            const TierCoverage at_reach = QueryTier(kRungs[r].tier, x, coverage.reachable);
            EXPECT_TRUE(at_reach.meets) << kRungs[r].name << " at x = " << x
                                        << ": the report named a reachable error it does not "
                                        << "meet at its own tolerance";

            if (at_reach.meets && worst[i] > at_reach.reachable)
            {
                ++dishonest_meets;
            }
        }
    }

    std::printf("\nper-argument query: %zu (rung, argument) cells, %zu reachable overclaims, "
                "%zu dishonest meets\n",
                cells,
                overclaimed,
                dishonest_meets);

    EXPECT_EQ(overclaimed, 0U)
        << "QueryTier(tier, x, ...) reported a reachable error smaller than the code delivers "
        << "at x = " << first_overclaim_x << " (rung index " << first_overclaim_rung
        << ") - the report claims coverage it does not have";
    EXPECT_EQ(dishonest_meets, 0U);
}

TEST(Tier, NamingTheWrongRegionOverclaimsAndTheArgumentEntryRemovesTheGuess) {
    // Why the argument-taking entry exists. The region-taking form trusts the
    // caller's classification, and the region boundaries are internal, so a
    // caller who guesses is not being conservative: region C's reachable is the
    // reference tier's at every m, so naming region C for an argument that is
    // really in region A or B reports a tier reaching m times better than it
    // does. This measures that hazard per wrong name - the evidence for the
    // overload, printed rather than asserted at, because the hazard belongs to
    // a caller's guess and not to the code.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::printf("\n%-15s", "rung");

    for (AccuracyRegion named : kRegions)
    {
        std::printf("  overclaim-if-named-%s", RegionName(named));
    }

    std::printf("\n");

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const std::vector<double>& worst = PerArgumentWorst(r);
        std::printf("%-15s", kRungs[r].name);

        for (AccuracyRegion named : kRegions)
        {
            const double reachable = DeclaredReachable(kRungs[r].tier, named);
            std::size_t overclaims = 0;

            for (std::size_t i = 0; i < grid.xs.size(); ++i)
            {
                if (worst[i] > reachable)
                {
                    ++overclaims;
                }
            }

            std::printf("  %10zu / %-6zu", overclaims, grid.xs.size());
        }

        std::printf("\n");
    }

    // The argument-taking entry cannot overclaim: it derives the region, so
    // every cell it answers for is answered at its own region's bound.
    std::size_t overclaims = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const std::vector<double>& worst = PerArgumentWorst(r);

        for (std::size_t i = 0; i < grid.xs.size(); ++i)
        {
            if (worst[i] > QueryTier(kRungs[r].tier, grid.xs[i], 0.0).reachable)
            {
                ++overclaims;
            }
        }
    }

    EXPECT_EQ(overclaims, 0U);
}

TEST(Tier, TheEntryHoldsTheBatchRowAndNotThePerRegionColumn) {
    // The tier entry dispatches to BoysAllOrders - the batch entry - so the
    // bound it can hold is the batch row of the contract table, m*5.5e-14 in
    // every region, NOT the three-value per-region column (m*1e-15 in region
    // A) that the single lane holds. The two differ by up to 5.5e-14/1e-15 =
    // 55 at m = 1, and by 55*m at a relaxed rung, so a caller who reads the
    // sentence's "B_region" as the per-region column is told a bound up to
    // tens of times tighter than the code enforces.
    //
    // This is decided by measurement, not by preference: if the per-region
    // column were this entry's contract, the entry would have to hold m*1e-15
    // in region A, and at m = 1 it does not. The cell the batch lane is worst
    // at is the smallest argument, where the extended-band seed serves the
    // whole recursion, so that is where the two bounds are told apart.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    // The reference tier's own contract, as the header states it for the
    // single lane and for the batch lane.
    constexpr double kPerRegionColumnA = 1e-15;
    constexpr double kBatchRow = 5.5e-14;

    // The smallest positive argument: the cell the batch lane's extended-band
    // seed serves the whole recursion at. Not x = 0, where every tier returns
    // 1/(2l+1) exactly and the two bounds cannot be told apart.
    const double denorm = std::numeric_limits<double>::denorm_min();

    std::printf("\n%-15s %14s %12s %13s %14s\n",
                "rung",
                "worst region A",
                "of m*1e-15",
                "of m*5.5e-14",
                "at denorm_min");

    std::vector<double> out(boys::kMaxBoysOrder + 1, std::nan(""));
    std::size_t outside_per_region = 0;
    std::size_t outside_batch = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        const Rung& rung = kRungs[r];
        const double m = rung.multiplier;
        const double worst = SweepFor(r, AccuracyRegion::kA).worst;

        double at_denorm = 0.0;
        BoysAllOrdersAtTier(rung.tier, boys::kMaxBoysOrder, denorm, out.data());

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            // F_k(denorm_min) = 1/(2k+1) to every digit a double holds: the
            // integrand's x*t^2 term is below the representation of the result.
            const double exact = 1.0 / (2.0 * k + 1.0);
            at_denorm = std::max(at_denorm, std::fabs(out[static_cast<std::size_t>(k)] - exact));
        }

        std::printf("%-15s %14.4e %12.4f %13.4f %14.4e\n",
                    rung.name,
                    worst,
                    worst / (m * kPerRegionColumnA),
                    worst / (m * kBatchRow),
                    at_denorm);

        if (worst > m * kPerRegionColumnA)
        {
            ++outside_per_region;
        }

        if (worst > m * kBatchRow)
        {
            ++outside_batch;
        }
    }

    std::printf(
        "region A: %zu of %zu rungs exceed the per-region column, %zu exceed the batch row\n",
        outside_per_region,
        kRungs.size(),
        outside_batch);

    // The decision, pinned: the entry does NOT hold the per-region column, so
    // the per-region table cannot be this entry's contract...
    EXPECT_GT(outside_per_region, 0U)
        << "no rung exceeds the per-region column in region A, so this reference cannot tell "
        << "the two readings apart - re-check before changing the documented bound";
    // ...and it does hold the batch row, which is what QueryTier reports. If
    // this fails, QueryTier IS under-reporting and the code needs the fix, not
    // the sentence.
    EXPECT_EQ(outside_batch, 0U)
        << "a rung exceeds the batch row, so QueryTier under-reports the delivered error";
}

TEST(Tier, QueryLimitingNamesTheComponentThatBoundsTheRegion) {
    const std::array<AccuracyComponent, 3> expected{{
        AccuracyComponent::kRegionASeed,
        AccuracyComponent::kRegionBFit,
        AccuracyComponent::kRegionCAsymptotic,
    }};

    for (const Rung& rung : kRungs)
    {
        for (std::size_t g = 0; g < kRegions.size(); ++g)
        {
            // Asking for an error no tier can reach: the caller is told what
            // stops it, and the name must be the region's own component.
            const TierCoverage coverage = QueryTier(rung.tier, kRegions[g], 0.0);
            EXPECT_EQ(coverage.limiting, expected[g])
                << rung.name << " region " << RegionName(kRegions[g]);
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Refusal by name: an unavailable tier, and an argument no tier refuses
// ---------------------------------------------------------------------------

TEST(Tier, UnavailableTierSelectsTheReferenceRungAndWritesEveryOutput) {
    // There is no named error channel on this surface (void, noexcept), so the
    // documented refusal is the conservative fallback: an enumerator the
    // selector does not know evaluates at the reference multiplier, which is
    // never coarser than any tier the caller could have named. What must never
    // happen is a SILENT NON-WRITE - the caller reading whatever its buffer
    // held - so the output is pre-filled with a sentinel that a real evaluation
    // cannot produce, and the sentinel is asserted gone.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    constexpr double kSentinel = std::numeric_limits<double>::quiet_NaN();
    const std::array<AccuracyTier, 3> unavailable{{
        static_cast<AccuracyTier>(7), // one past the last enumerator
        static_cast<AccuracyTier>(-1), // below the first
        static_cast<AccuracyTier>(1000),
    }};

    std::vector<double> got(boys::kMaxBoysOrder + 1);
    std::vector<double> reference(boys::kMaxBoysOrder + 1);

    for (AccuracyTier tier : unavailable)
    {
        for (double x : grid.xs)
        {
            for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
            {
                std::fill(got.begin(), got.end(), kSentinel);
                BoysAllOrdersAtTier(tier, nmax, x, got.data());

                for (int k = 0; k <= nmax; ++k)
                {
                    ASSERT_FALSE(std::isnan(got[static_cast<std::size_t>(k)]))
                        << "an unavailable tier (" << static_cast<int>(tier) << ") left out[" << k
                        << "] unwritten at x = " << x << ", nmax = " << nmax
                        << ": the caller reads uninitialized memory and no refusal happened";
                }

                std::fill(reference.begin(), reference.end(), kSentinel);
                BoysAllOrders<boys::kBoysFullAccuracyMultiplier>(nmax, x, reference.data());

                ASSERT_TRUE(BitwiseEqual(got.data(), reference.data(), nmax))
                    << "an unavailable tier (" << static_cast<int>(tier)
                    << ") did not fall back to the reference rung at x = " << x
                    << ", nmax = " << nmax;
            }
        }
    }
}

TEST(Tier, UnavailableTierIsReportedAsTheReferenceByTheWholeQuerySurface) {
    // The query surface and the selector must answer the same question the
    // same way, or a caller that records the accuracy it asked for beside the
    // numbers it got records an accuracy it did not get.
    const std::array<AccuracyTier, 3> unavailable{{
        static_cast<AccuracyTier>(7),
        static_cast<AccuracyTier>(-1),
        static_cast<AccuracyTier>(1000),
    }};

    for (AccuracyTier tier : unavailable)
    {
        EXPECT_DOUBLE_EQ(AccuracyMultiplier(tier), AccuracyMultiplier(AccuracyTier::kReference))
            << "AccuracyMultiplier(" << static_cast<int>(tier) << ") must name the multiplier "
            << "the selector actually reaches";

        for (AccuracyRegion region : kRegions)
        {
            const TierCoverage got = QueryTier(tier, region, 1e-12);
            const TierCoverage want = QueryTier(AccuracyTier::kReference, region, 1e-12);

            EXPECT_DOUBLE_EQ(got.reachable, want.reachable) << static_cast<int>(tier);
            EXPECT_EQ(got.meets, want.meets) << static_cast<int>(tier);
            EXPECT_EQ(got.limiting, want.limiting) << static_cast<int>(tier);
        }
    }
}

TEST(Tier, NoArgumentIsOutsideAnyTiersRegion) {
    // The tier entry has no per-tier region restriction: every rung serves all
    // three regions, and the two regions with a relaxable resource are the ones
    // the rungs differ in. So there is no "argument outside the tier's region"
    // to refuse, and the sharpest way to say so is that region C is
    // BIT-IDENTICAL across every rung - asking any tier for a region C value
    // gets the same bits, which is the m-invariance the query surface reports
    // as a constant reachable.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::vector<double> base(boys::kMaxBoysOrder + 1);
    std::vector<double> other(boys::kMaxBoysOrder + 1);
    std::size_t differs = 0;
    std::size_t cells = 0;

    for (double x : grid.xs)
    {
        if (RegionOf(x) != AccuracyRegion::kC)
        {
            continue;
        }

        for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
        {
            BoysAllOrdersAtTier(AccuracyTier::kReference, nmax, x, base.data());
            ++cells;

            for (std::size_t r = 1; r < kRungs.size(); ++r)
            {
                BoysAllOrdersAtTier(kRungs[r].tier, nmax, x, other.data());

                if (!BitwiseEqual(base.data(), other.data(), nmax))
                {
                    ++differs;
                }
            }
        }
    }

    std::printf(
        "\nregion C: %zu cells, %zu rung disagreements with the reference rung\n", cells, differs);

    EXPECT_EQ(differs, 0U)
        << "region C is documented as having no relaxable resource, so every rung must "
        << "return the reference rung's bits there; " << differs << " cells disagree";
}

TEST(Tier, DeliveredErrorRatiosArePinnedSoASilentRegressionIsVisible) {
    // The declared bound is loose enough that a regression can hide under it:
    // at kRelaxed16384 in region B the code delivers 0.21 of m*5.5e-14, so a
    // rung that got four times worse would still pass the bound test above.
    // This pins the ratio the code actually delivers, as an UPPER bound at 5%
    // above the recorded value - a further improvement passes, a regression
    // does not.
    //
    // The ratios are specific to this reference and to this sweep, so they are
    // pinned only when the accuracy gate's grid is the reference; on the suite
    // grid the sweep is too thin and the pin is not asserted. Re-baselining is
    // mechanical: the measured table is printed beside the recorded one.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    // Measured with the accuracy gate's grid; column order matches kRegions.
    struct Recorded {
        const char* rung;
        double ratio[3];
    };

    constexpr std::array<Recorded, 7> kRecorded{{
        {"kReference", {0.0585, 0.1807, 0.9091}},
        {"kRelaxed64", {0.7049, 0.0743, 0.9091}},
        {"kRelaxed256", {0.7841, 0.5211, 0.9091}},
        {"kRelaxed1024", {0.9759, 0.1303, 0.9091}},
        {"kRelaxed4096", {0.8015, 0.8419, 0.9091}},
        {"kRelaxed16384", {0.6249, 0.2105, 0.9091}},
        {"kRelaxed65536", {0.9786, 0.0526, 0.9091}},
    }};

    // 5% above the recorded value: an improvement is a smaller ratio and
    // passes; a regression beyond 5% is a change to report, not to absorb.
    constexpr double kPin = 1.05;
    const bool pinned = grid.provenance.find("accuracy-gate") != std::string::npos;

    std::printf("\n%-15s %-7s %12s %12s %8s\n", "rung", "region", "measured", "recorded", "of pin");

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (std::size_t g = 0; g < kRegions.size(); ++g)
        {
            const double declared = DeclaredReachable(kRungs[r].tier, kRegions[g]);
            const double measured =
                declared > 0.0 ? SweepFor(r, kRegions[g]).worst / declared : 0.0;
            const double recorded = kRecorded[r].ratio[g];

            std::printf("%-15s %-7s %12.4f %12.4f %8.4f\n",
                        kRungs[r].name,
                        RegionName(kRegions[g]),
                        measured,
                        recorded,
                        measured / recorded);

            if (!pinned)
            {
                continue;
            }

            EXPECT_LE(measured, recorded * kPin)
                << kRungs[r].name << " region " << RegionName(kRegions[g]) << " now delivers "
                << measured << " of its declared bound, against a recorded " << recorded
                << ". If the reference grid or the sweep changed, re-baseline the "
                << "table in this test; otherwise a rung got worse inside a bound loose enough "
                << "to hide it.";
        }
    }
}

TEST(Tier, BoundCoveredCellsAreCountedAndNamedRatherThanPassed) {
    // The vacuous-pass audit: how much of each rung's green total is a cell
    // where the declared bound is larger than the function's own magnitude, so
    // returning zero would pass it too. Printed in full and asserted to be
    // counted, not hidden - a rung whose whole sweep is bound-covered has not
    // been tested at all.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::printf("\n%-15s %-7s %10s %10s %10s\n", "rung", "region", "cells", "met", "bound-cov");

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (AccuracyRegion region : kRegions)
        {
            const Sweep& sweep = SweepFor(r, region);

            std::printf("%-15s %-7s %10zu %10zu %10zu\n",
                        kRungs[r].name,
                        RegionName(region),
                        sweep.cells,
                        sweep.met,
                        sweep.bound_covered);

            EXPECT_LE(sweep.bound_covered, sweep.cells) << kRungs[r].name;
        }
    }

    // The reference rung in region A is the certified lane's own pin: its
    // bound-covered count is the floor the audit is measured against, and it is
    // reported rather than asserted at - but a rung that is bound-covered
    // EVERYWHERE in a region would mean no cell of that region distinguishes a
    // correct implementation from a stub, which is worth failing on.
    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (AccuracyRegion region : kRegions)
        {
            const Sweep& sweep = SweepFor(r, region);

            EXPECT_LT(sweep.bound_covered, sweep.cells)
                << kRungs[r].name << " region " << RegionName(region)
                << ": every cell is bound-covered, so no cell of this region distinguishes a "
                << "correct implementation from one that returns zero";
        }
    }
}

// ---------------------------------------------------------------------------
// 5. The certified fit routes: FitRoute, FitRouteInfo, BoysFitRoutes() and
//    BoysAllOrdersWithRoute()
// ---------------------------------------------------------------------------
// A route is a way of serving a region rather than a rung of one design, so the
// selector's contract is read off the report rather than transcribed: the tests
// below take every interval, bound and served-domain boundary from
// BoysFitRoutes() and assert the entry against it. Transcribing them here would
// let the report drift from the code and stay green.
//
// What is asserted:
//  * the report enumerates the routes, one row per route and region, with the
//    figures a consumer reads - and no two rows describe the same pair twice;
//  * every row delivers its own bound over the domain its selector serves,
//    measured against the reference grid, with the bound-covered cells counted
//    so a row that passes vacuously is visible;
//  * naming a route changes nothing outside the domain its row states it serves
//    - in particular nothing below the boundary a row names above its own left
//    edge, which is the documented fallback for a fit that reaches further than
//    its selector takes over;
//  * the default route is the default entry bit for bit, and a value the
//    enumeration does not name is the default route, so no caller is handed a
//    fit they did not ask for.

// The domain a row's selector serves: the fit may reach further left than this.
double ServedFrom(const boys::FitRouteInfo& row) {
    return std::max(row.lo, row.servesFrom);
}

TEST(Route, TheReportEnumeratesEveryRouteOverEveryRegionItServes) {
    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();

    ASSERT_FALSE(routes.empty()) << "no route is enumerated: a consumer cannot ask what exists";

    for (const boys::FitRouteInfo& row : routes)
    {
        EXPECT_NE(row.name, nullptr);
        EXPECT_GT(row.stored, 0) << row.name;
        EXPECT_LT(row.lo, row.hi) << row.name;
        EXPECT_GT(row.bound, 0.0) << row.name;
        EXPECT_LE(row.delivered, row.bound)
            << row.name << " region " << RegionName(row.region)
            << ": the row reports a delivered figure above the bar it is certified against";
        EXPECT_GE(row.servesFrom, row.lo)
            << row.name << " region " << RegionName(row.region)
            << ": a row's served domain cannot begin before its own fit does";
        EXPECT_LT(row.servesFrom, row.hi) << row.name << " region " << RegionName(row.region);
    }

    // One row per route and region, and each region that has a fit has both
    // routes: a consumer comparing two routes over one region needs both rows
    // to exist. Region C has no stored fit - the asymptotic branch is a closed
    // form with no coefficients to choose between - so it has no rows.
    const std::array<AccuracyRegion, 2> fitted{{AccuracyRegion::kA, AccuracyRegion::kB}};

    for (AccuracyRegion region : fitted)
    {
        std::size_t chebyshev = 0;
        std::size_t rational = 0;

        for (const boys::FitRouteInfo& row : routes)
        {
            if (row.region != region)
            {
                continue;
            }

            if (row.route == boys::FitRoute::kChebyshev)
            {
                ++chebyshev;
            }

            if (row.route == boys::FitRoute::kRationalMinimax)
            {
                ++rational;
            }
        }

        EXPECT_EQ(chebyshev, 1u) << "region " << RegionName(region);
        EXPECT_EQ(rational, 1u) << "region " << RegionName(region);
    }

    for (const boys::FitRouteInfo& row : routes)
    {
        EXPECT_NE(row.region, AccuracyRegion::kC)
            << row.name << ": region C has no stored fit for a route to serve";
    }

    // The region-A rational row states a served domain that begins above its own
    // fit's left edge, and no other row does. That row is the reason
    // FitRouteInfo::servesFrom exists: below that argument the lane reads every
    // order from its own fit and documents a tighter figure than the rational
    // fits hold, so the row claims the narrower domain rather than the interval
    // its table covers. The argument it names is the lowest of the per-order
    // boundaries its selector hands orders over at, which is the extended band's
    // left edge.
    std::size_t narrowed = 0;

    for (const boys::FitRouteInfo& row : routes)
    {
        if (row.servesFrom > row.lo)
        {
            ++narrowed;

            EXPECT_EQ(row.region, AccuracyRegion::kA) << row.name;
            EXPECT_EQ(row.route, boys::FitRoute::kRationalMinimax) << row.name;
            EXPECT_DOUBLE_EQ(row.lo, 0.0) << row.name;
            EXPECT_DOUBLE_EQ(row.servesFrom, boys::detail::kExtendedBX0)
                << "the boundary the region-A rational row names is not the extended band's edge";
        }
    }

    EXPECT_EQ(narrowed, 1u) << "the report should narrow exactly one row's served domain";
}

TEST(Route, EveryRowDeliversItsOwnBoundAtEveryReferenceArgumentItServes) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();
    std::vector<double> out(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    std::printf("\n%-20s %-7s %10s %10s %10s %14s\n",
                "route",
                "region",
                "cells",
                "met",
                "bound-cov",
                "worst/bound");
    std::printf("%-20s %-7s %10s %10s %10s %14s\n", "", "", "", "", "", "(at n, x)");

    for (const boys::FitRouteInfo& row : routes)
    {
        Sweep sweep;

        for (std::size_t i = 0; i < grid.xs.size(); ++i)
        {
            const double x = grid.xs[i];

            if (!(x >= ServedFrom(row) && x < row.hi))
            {
                continue;
            }

            BoysAllOrdersWithRoute(row.route, boys::kMaxBoysOrder, x, out.data());

            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                const double ref = grid.values[static_cast<std::size_t>(n)][i];

                if (std::isnan(ref))
                {
                    continue;
                }

                sweep.Note(std::fabs(out[static_cast<std::size_t>(n)] - ref), n, x, ref, row.bound);
            }
        }

        std::printf("%-20s %-7s %10zu %10zu %10zu %14.3g (n=%d, x=%.6g)\n",
                    row.name,
                    RegionName(row.region),
                    sweep.cells,
                    sweep.met,
                    sweep.bound_covered,
                    sweep.worst / row.bound,
                    sweep.worst_n,
                    sweep.worst_x);

        ASSERT_GT(sweep.cells, 0u)
            << row.name << " region " << RegionName(row.region)
            << ": no grid argument lies in the domain this row serves, so nothing was measured";
        EXPECT_EQ(sweep.met, sweep.cells)
            << row.name << " region " << RegionName(row.region) << ": worst " << sweep.worst
            << " against " << row.bound << " at n = " << sweep.worst_n << ", x = " << sweep.worst_x;
        EXPECT_LT(sweep.bound_covered, sweep.cells)
            << row.name << " region " << RegionName(row.region)
            << ": every cell is bound-covered, so returning zero would pass this row too";
    }
}

TEST(Route, NamingARouteChangesNothingOutsideTheDomainItsRowServes) {
    // The confinement contract, read off the report: outside the intervals its
    // rows state, naming a route hands the caller the default entry's values bit
    // for bit. Below the boundary the region-A rational row names, that is what
    // keeps the tighter bound the lane documents there.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();
    std::vector<double> plain(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    std::vector<double> got(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    std::size_t outside = 0;
    std::size_t inside = 0;
    std::size_t belowBoundary = 0;

    for (double x : grid.xs)
    {
        boys::BoysAllOrders(boys::kMaxBoysOrder, x, plain.data());
        BoysAllOrdersWithRoute(
            boys::FitRoute::kRationalMinimax, boys::kMaxBoysOrder, x, got.data());

        bool served = false;

        for (const boys::FitRouteInfo& row : routes)
        {
            if (x >= ServedFrom(row) && x < row.hi)
            {
                served = true;
            }
        }

        const bool equal = BitwiseEqual(plain.data(), got.data(), boys::kMaxBoysOrder);

        if (!served)
        {
            ++outside;
            EXPECT_TRUE(equal) << "naming the rational route changed the values at x = " << x
                               << ", which no row of BoysFitRoutes reports it serves";
        } else
        {
            ++inside;
        }

        // The narrow statement, measured on its own so a failure names which one
        // of the two moved: below the region-A rational row's boundary the
        // entry's values are the default's.
        for (const boys::FitRouteInfo& row : routes)
        {
            if (row.servesFrom <= row.lo || x >= row.servesFrom || x < row.lo)
            {
                continue;
            }

            ++belowBoundary;
            EXPECT_TRUE(equal) << "naming the " << row.name
                               << " route changed the values at x = " << x
                               << ", below the boundary " << row.servesFrom << " its row names";
        }
    }

    EXPECT_GT(outside, 0u) << "no grid argument lies outside every route's served domain";
    EXPECT_GT(inside, 0u) << "no grid argument lies inside a route's served domain";
    EXPECT_GT(belowBoundary, 0u)
        << "the narrowed row's served-domain boundary was never exercised from below";
}

TEST(Route, AnOrderInsideTheServedDomainKeepsTheDefaultValueUntilItsOwnBoundary) {
    // The region-A rational row serves per order, so the boundary it states is
    // the lowest of a set. Above that boundary an order whose own region-A range
    // has not ended yet is still read from its own fit by the lane, and the
    // entry hands back the default's value for it rather than the route's - the
    // property that keeps the lane's tighter per-order figure intact right up to
    // the argument where the lane itself stops reading the order from its fit.
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    std::vector<double> plain(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    std::vector<double> got(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    std::size_t handed = 0;
    std::size_t held = 0;

    for (double x : grid.xs)
    {
        if (x < boys::detail::kRatARouteLo || x >= boys::detail::kRatARouteHi)
        {
            continue;
        }

        boys::BoysAllOrders(boys::kMaxBoysOrder, x, plain.data());
        BoysAllOrdersWithRoute(
            boys::FitRoute::kRationalMinimax, boys::kMaxBoysOrder, x, got.data());

        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            const auto j = static_cast<std::size_t>(n);

            if (x >= boys::detail::kTierThresholds[j])
            {
                ++handed;
                continue;
            }

            ++held;
            EXPECT_TRUE(got[j] == plain[j])
                << "at x = " << x << ", order " << n
                << " is still inside its own region-A range, and the entry changed it";
        }
    }

    EXPECT_GT(handed, 0u) << "no grid argument handed any order to the route's own fit";
    EXPECT_GT(held, 0u) << "no grid argument reached an order before its own region-A end";
}

TEST(Route, TheDefaultRouteAndAnUnnamedRouteAreTheDefaultEntry) {
    const Grid& grid = Reference();

    if (grid.xs.empty())
    {
        GTEST_SKIP() << "no reference grid";
    }

    // The default route is today's certified code, so it must be bit-identical
    // to the default entry rather than merely close: a route selector that
    // perturbed the default path would move every lane's measured figure.
    const std::array<boys::FitRoute, 3> fallbacks{{
        boys::FitRoute::kChebyshev,
        static_cast<boys::FitRoute>(99),
        static_cast<boys::FitRoute>(-1),
    }};
    std::vector<double> plain(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    std::vector<double> got(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    for (double x : grid.xs)
    {
        boys::BoysAllOrders(boys::kMaxBoysOrder, x, plain.data());

        for (boys::FitRoute route : fallbacks)
        {
            std::fill(got.begin(), got.end(), std::nan(""));
            BoysAllOrdersWithRoute(route, boys::kMaxBoysOrder, x, got.data());

            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                ASSERT_FALSE(std::isnan(got[static_cast<std::size_t>(n)]))
                    << "route " << static_cast<int>(route) << " left out[" << n
                    << "] unwritten at x = " << x;
            }

            EXPECT_TRUE(BitwiseEqual(plain.data(), got.data(), boys::kMaxBoysOrder))
                << "route " << static_cast<int>(route) << " at x = " << x
                << " is not the default entry's values";
        }
    }
}

TEST(Route, EveryOrderIsWrittenWhateverNmaxAndRouteAreAsked) {
    // The entry's shape contract: nmax + 1 values, every one of them written.
    // Above the extended band's left edge the batch entry reads each order from
    // that order's own fit, so its values carry no nmax: an nmax smaller than
    // the order asked about must not truncate or change anything, and the route
    // selector only picks fits, so it keeps that property. Below that edge the
    // batch entry seeds one downward recursion at nmax (the documented
    // fallback), so there the values do depend on nmax by design and only the
    // shape is asserted.
    const std::span<const boys::FitRouteInfo> routes = boys::BoysFitRoutes();
    std::vector<double> batch(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    std::vector<double> single(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    const std::array<double, 8> xs{{
        0.25,
        1.5,
        3.0,
        6.5,
        9.5,
        11.5,
        15.0,
        40.0,
    }};

    for (boys::FitRoute route : {boys::FitRoute::kChebyshev, boys::FitRoute::kRationalMinimax})
    {
        for (double x : xs)
        {
            std::fill(batch.begin(), batch.end(), std::nan(""));
            BoysAllOrdersWithRoute(route, boys::kMaxBoysOrder, x, batch.data());

            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                EXPECT_FALSE(std::isnan(batch[static_cast<std::size_t>(n)]))
                    << "route " << static_cast<int>(route) << " at x = " << x << ": out[" << n
                    << "] was left unwritten";
            }

            const bool per_order = x >= boys::detail::kExtendedBX0;

            for (int nmax = 0; nmax <= boys::kMaxBoysOrder; ++nmax)
            {
                std::fill(single.begin(), single.end(), std::nan(""));
                BoysAllOrdersWithRoute(route, nmax, x, single.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    EXPECT_FALSE(std::isnan(single[static_cast<std::size_t>(n)]))
                        << "route " << static_cast<int>(route) << " at x = " << x
                        << ", nmax = " << nmax << ": out[" << n << "] was left unwritten";

                    if (!per_order)
                    {
                        continue;
                    }

                    EXPECT_EQ(std::bit_cast<std::uint64_t>(single[static_cast<std::size_t>(n)]),
                              std::bit_cast<std::uint64_t>(batch[static_cast<std::size_t>(n)]))
                        << "route " << static_cast<int>(route) << " at x = " << x
                        << ", nmax = " << nmax << ": out[" << n << "] depends on nmax";
                }
            }
        }
    }

    // The routes the loop above covers are the ones the report names: a route
    // added to the library without a row here would be exercised by nothing.
    for (const boys::FitRouteInfo& row : routes)
    {
        EXPECT_TRUE(row.route == boys::FitRoute::kChebyshev ||
                    row.route == boys::FitRoute::kRationalMinimax)
            << row.name << ": a route this test does not sweep";
    }
}

// ---------------------------------------------------------------------------
// The rational route's relaxed rung
// ---------------------------------------------------------------------------
// The rung and the route are two selectors of two different things, and this
// section is where their product is measured: the pair criterion's own bound
// (which is the derivation, and is a claim about a bound rather than about any
// value a table happens to hold), the run-time entry's mapping onto the
// compile-time instantiation, the rung's declared bound on the reference grid,
// and the route's carriage - that naming the pair is answered with the route's
// fits rather than with the default entry's.
//
// What is NOT pinned here is which cut the criterion certifies. That is the
// criterion's output and not its contract: a coefficient table that moved would
// move it, and a test that pinned it would fail on a change that is correct.
// ThePairCriterionBoundsEveryCutOfEveryRationalPiece is the contract - the
// bound holds whatever the cut is - and it is measured on every cut of every
// piece, with the cuts the criterion refuses counted beside the ones it admits.

// The mapped argument's own grid for the piece-level bound sweep: dense enough
// that a bound missed between points would have to be missed narrowly.
constexpr int kPairSweep = 20001;

// The rational route's rung compiled at a multiplier and a scheme: the path the
// run-time entry has to reach, compiled here for the comparison.
template <double M, boys::EvalScheme S>
void RationalRungPath(int nmax, double x, double* out) noexcept {
    BoysAllOrders<M, boys::EvalPolicy<boys::FitRoute::kRationalMinimax, S>>(nmax, x, out);
}

// Indexed by rung, in kRungs order (the reference rung first).
constexpr std::array<PathFn, 7> kRationalSplitPaths{{
    &RationalRungPath<boys::kBoysFullAccuracyMultiplier, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<64.0, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<256.0, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<1024.0, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<4096.0, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<16384.0, boys::EvalScheme::kSplitClenshaw>,
    &RationalRungPath<65536.0, boys::EvalScheme::kSplitClenshaw>,
}};

constexpr std::array<PathFn, 7> kRationalHornerPaths{{
    &RationalRungPath<boys::kBoysFullAccuracyMultiplier, boys::EvalScheme::kHorner>,
    &RationalRungPath<64.0, boys::EvalScheme::kHorner>,
    &RationalRungPath<256.0, boys::EvalScheme::kHorner>,
    &RationalRungPath<1024.0, boys::EvalScheme::kHorner>,
    &RationalRungPath<4096.0, boys::EvalScheme::kHorner>,
    &RationalRungPath<16384.0, boys::EvalScheme::kHorner>,
    &RationalRungPath<65536.0, boys::EvalScheme::kHorner>,
}};

TEST(ThePairCriterion, BoundsEveryCutOfEveryRationalPiece) {
    std::size_t cuts = 0;
    std::size_t inadmissible = 0;
    double tightest = 0.0;
    int tightest_piece = -1;
    int tightest_cut = -1;

    for (std::size_t piece = 0; piece < boys::detail::kPieces.size(); ++piece) {
        const std::size_t offset = static_cast<std::size_t>(boys::detail::kRatAOffset[piece]);
        const int numDeg = boys::detail::kRatANumDeg[piece];
        const int denDeg = boys::detail::kRatADenDeg[piece];
        const int full = numDeg > denDeg ? numDeg : denDeg;

        for (int cut = 0; cut <= full; ++cut) {
            if (cut == 3 || (cut > 2 && cut % 2 != 0)) {
                continue;
            }

            bool admissible = false;
            const std::size_t denOffset =
                offset + static_cast<std::size_t>(numDeg) + 1;
            const double tail = boys::detail::RationalPairTail(boys::detail::kRatACoeffs,
                                                               offset,
                                                               boys::detail::kRatACoeffs,
                                                               denOffset,
                                                               numDeg,
                                                               denDeg,
                                                               cut,
                                                               admissible);

            if (!admissible) {
                ++inadmissible;
                continue;
            }

            double worst = 0.0;

            for (int i = 0; i < kPairSweep; ++i) {
                const double t = -1.0 + 2.0 * i / (kPairSweep - 1);
                const double stored = boys::detail::RationalPieceAtCut(piece, numDeg, denDeg, t);
                const double cutPair = boys::detail::RationalPieceAtCut(piece,
                                                                       cut < numDeg ? cut : numDeg,
                                                                       cut < denDeg ? cut : denDeg,
                                                                       t);
                worst = std::max(worst, std::fabs(stored - cutPair));
            }

            ++cuts;

            // The bound is the criterion's whole claim: a cut it admits is a cut
            // whose move the budget is asserted to carry.
            EXPECT_LE(worst, tail)
                << "piece " << piece << " (F" << piece / 2 << ") cut " << cut << ": measured "
                << worst << " against the criterion's " << tail;

            const double ratio = tail > 0.0 ? worst / tail : 0.0;

            if (ratio > tightest) {
                tightest = ratio;
                tightest_piece = static_cast<int>(piece);
                tightest_cut = cut;
            }
        }
    }

    // A criterion every one of whose cuts is inadmissible measures nothing: its
    // fallback would be reached by every scan and the scan itself untested.
    EXPECT_GT(cuts, 0u) << "no cut of any rational piece is admissible";
    EXPECT_GT(inadmissible, 0u) << "no cut was refused, so the denominator's floor never bit";
    EXPECT_LT(tightest, 1.0) << "the tightest cut is the bound itself, which would say the "
                                "measure is not a bound at all";

    std::printf("\nthe pair criterion: %zu admissible cut(s), %zu refused by the denominator's "
                "floor; tightest measured/bound %.4g at piece %d, cut %d\n",
                cuts,
                inadmissible,
                tightest,
                tightest_piece,
                tightest_cut);
}

TEST(Rung, TheRationalRungIsBitIdenticalToItsCompileTimeInstantiation) {
    const std::array<AccuracyTier, 7> tiers{AccuracyTier::kReference,
                                            AccuracyTier::kRelaxed64,
                                            AccuracyTier::kRelaxed256,
                                            AccuracyTier::kRelaxed1024,
                                            AccuracyTier::kRelaxed4096,
                                            AccuracyTier::kRelaxed16384,
                                            AccuracyTier::kRelaxed65536};

    const std::array<double, 12> xs{1e-12, 0.3,    1.0,  1.0855252345349333,
                                    2.5,   6.0,    11.0, 11.899848152108484,
                                    14.0,  20.0,   28.9, 40.0};

    for (std::size_t r = 0; r < tiers.size(); ++r) {
        for (const boys::EvalScheme scheme :
             {boys::EvalScheme::kSplitClenshaw, boys::EvalScheme::kHorner}) {
            const PathFn path = scheme == boys::EvalScheme::kHorner ? kRationalHornerPaths[r]
                                                                    : kRationalSplitPaths[r];

            for (const double x : xs) {
                std::array<double, 33> routed{};
                std::array<double, 33> direct{};

                BoysAllOrdersAtTier(tiers[r], boys::FitRoute::kRationalMinimax, scheme, 32, x,
                                    routed.data());
                path(32, x, direct.data());

                for (int n = 0; n <= 32; ++n) {
                    EXPECT_EQ(std::bit_cast<std::uint64_t>(routed[static_cast<std::size_t>(n)]),
                              std::bit_cast<std::uint64_t>(direct[static_cast<std::size_t>(n)]))
                        << "rung " << r << ", scheme " << static_cast<int>(scheme)
                        << ", x = " << x << ": out[" << n << "] is not the instantiation's value";
                }
            }
        }
    }
}

TEST(Rung, EveryRationalRungHoldsItsDeclaredBoundOnTheReferenceGrid) {
    const Grid& grid = Reference();

    if (grid.xs.empty()) {
        GTEST_SKIP() << "no reference grid";
    }

    std::vector<double> out(static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    std::printf("\n%-14s %-16s %10s %10s %10s %14s\n",
                "rung",
                "scheme",
                "cells",
                "met",
                "bound-cov",
                "worst/bound");

    for (int t = static_cast<int>(AccuracyTier::kReference);
         t <= static_cast<int>(AccuracyTier::kRelaxed65536);
         ++t) {
        const AccuracyTier tier = static_cast<AccuracyTier>(t);
        const double m = AccuracyMultiplier(tier);
        const double bound = m * kDeclaredBatchBase;

        for (const boys::EvalScheme scheme :
             {boys::EvalScheme::kSplitClenshaw, boys::EvalScheme::kHorner}) {
            Sweep sweep;

            for (std::size_t i = 0; i < grid.xs.size(); ++i) {
                BoysAllOrdersAtTier(tier, boys::FitRoute::kRationalMinimax, scheme,
                                    boys::kMaxBoysOrder, grid.xs[i], out.data());

                for (int n = 0; n <= boys::kMaxBoysOrder; ++n) {
                    const double ref = grid.values[static_cast<std::size_t>(n)][i];

                    if (std::isnan(ref)) {
                        continue;
                    }

                    sweep.Note(std::fabs(out[static_cast<std::size_t>(n)] - ref),
                               n,
                               grid.xs[i],
                               ref,
                               bound);
                }
            }

            std::printf("%-14g %-16s %10zu %10zu %10zu %14.3g (n=%d, x=%.6g)\n",
                        m,
                        boys::EvalSchemeName(scheme),
                        sweep.cells,
                        sweep.met,
                        sweep.bound_covered,
                        sweep.worst / bound,
                        sweep.worst_n,
                        sweep.worst_x);

            ASSERT_GT(sweep.cells, 0u) << "the reference grid measured nothing at m = " << m;
            EXPECT_EQ(sweep.met, sweep.cells)
                << "m = " << m << " at " << boys::EvalSchemeName(scheme) << ": worst "
                << sweep.worst << " against " << bound << " at n = " << sweep.worst_n
                << ", x = " << sweep.worst_x;
            EXPECT_LT(sweep.bound_covered, sweep.cells)
                << "m = " << m << ": every cell is bound-covered, so returning zero would pass";
        }
    }
}

TEST(Rung, TheRationalRungIsTheRoutesOwnAnswerAndNotTheDefaultEntries) {
    // Naming the route at a rung has to change the answer, or the route was not
    // carried: the default entry's rung is the Chebyshev route's, and the two
    // routes' fits are different numbers over the interval the rational row
    // serves. Measured and counted rather than asserted, because a route that
    // agreed everywhere would be a selection that does nothing.
    std::size_t cells = 0;
    std::size_t differ = 0;

    for (int t = static_cast<int>(AccuracyTier::kRelaxed64);
         t <= static_cast<int>(AccuracyTier::kRelaxed65536);
         ++t) {
        const AccuracyTier tier = static_cast<AccuracyTier>(t);

        for (double x = 1.09; x < kX0; x += 0.05) {
            std::array<double, 33> rational{};
            std::array<double, 33> shipped{};

            BoysAllOrdersAtTier(tier, boys::FitRoute::kRationalMinimax,
                                boys::EvalScheme::kSplitClenshaw, 32, x, rational.data());
            BoysAllOrdersAtTier(tier, 32, x, shipped.data());

            for (int n = 0; n <= 32; ++n) {
                ++cells;
                differ += rational[static_cast<std::size_t>(n)] !=
                          shipped[static_cast<std::size_t>(n)];
            }
        }
    }

    EXPECT_GT(differ, 0u) << "the rational rung answers the default route's values everywhere: "
                             "the route is not carried";
    std::printf("\nthe rational rung differs from the default entry's rung in %zu of %zu cell(s)\n",
                differ,
                cells);
}

// ---------------------------------------------------------------------------
// 3. The same tier on the single-order shape
// ---------------------------------------------------------------------------
// An engine that reads one order at a time cannot reach a rung through an entry
// that computes every order, so the single-order shape has its own run-time
// entry. What is held here is that it selects, not that it exists: the value the
// run-time entry returns is the compile-time lane's at the multiplier the tier
// names, bit for bit, because a switch that reached the wrong body would still
// deliver something inside every bound in this file.

namespace {

using SingleFn = double (*)(int, double) noexcept;

template <double M> double LibrarySingle(int n, double x) noexcept {
    return boys::BoysSingle<M>(n, x);
}

template <double M> double LibrarySingleRational(int n, double x) noexcept {
    return boys::BoysSingle<M, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(n, x);
}

constexpr std::array<SingleFn, 7> kLibrarySingles{{
    &LibrarySingle<boys::kBoysFullAccuracyMultiplier>,
    &LibrarySingle<64.0>,
    &LibrarySingle<256.0>,
    &LibrarySingle<1024.0>,
    &LibrarySingle<4096.0>,
    &LibrarySingle<16384.0>,
    &LibrarySingle<65536.0>,
}};

constexpr std::array<SingleFn, 7> kLibraryRationalSingles{{
    &LibrarySingleRational<boys::kBoysFullAccuracyMultiplier>,
    &LibrarySingleRational<64.0>,
    &LibrarySingleRational<256.0>,
    &LibrarySingleRational<1024.0>,
    &LibrarySingleRational<4096.0>,
    &LibrarySingleRational<16384.0>,
    &LibrarySingleRational<65536.0>,
}};

constexpr std::array<double, 9> kSingleArgs{{
    0.0, 1e-8, 0.001, 0.5, 1.0855252345349333, 3.0, 11.899848152108484, 20.0, 200.0,
}};

} // namespace

TEST(Tier, TheSingleOrderRungIsTheCompileTimeLaneAtItsMultiplier) {
    std::size_t cells = 0;
    std::size_t mismatch = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (const double x : kSingleArgs)
        {
            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                ++cells;
                const double got = boys::BoysSingleAtTier(kRungs[r].tier, n, x);
                const double want = kLibrarySingles[r](n, x);

                if (std::memcmp(&got, &want, sizeof(double)) != 0)
                {
                    ++mismatch;
                }
            }
        }
    }

    EXPECT_GT(cells, 0u);
    EXPECT_EQ(mismatch, 0u)
        << "BoysSingleAtTier is not the compile-time lane at the multiplier its tier names";
}

TEST(Tier, TheSingleOrderRungCarriesTheRouteAtRunTime) {
    std::size_t cells = 0;
    std::size_t mismatch = 0;
    std::size_t differ = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (const double x : kSingleArgs)
        {
            for (int n = 0; n <= boys::kMaxBoysOrder; n += 4)
            {
                ++cells;
                const double got = boys::BoysSingleAtTier(
                    kRungs[r].tier, boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw, n, x);
                const double want = kLibraryRationalSingles[r](n, x);

                if (std::memcmp(&got, &want, sizeof(double)) != 0)
                {
                    ++mismatch;
                }

                const double shipped = kLibrarySingles[r](n, x);

                if (std::memcmp(&want, &shipped, sizeof(double)) != 0)
                {
                    ++differ;
                }            }
        }
    }

    EXPECT_GT(cells, 0u);
    EXPECT_EQ(mismatch, 0u) << "the run-time route does not select the route's own fits";
    EXPECT_GT(differ, 0u)
        << "the rational route is not reachable at run time on the single-order shape: it "
           "answers the default route's values everywhere";
}

TEST(Tier, TheSingleOrderEntryAgreesWithTheBatchEntryAtTheSameRung) {
    // The two shapes are different calls and are not required to be equal - the
    // batch entry seeds at the top order and recurses down where this one reads
    // its own fit - so what is held is only that both stay inside the contract
    // at the same rung, which is the claim a caller reading one order cares
    // about.
    std::size_t over = 0;
    std::size_t cells = 0;

    for (std::size_t r = 0; r < kRungs.size(); ++r)
    {
        for (const double x : kSingleArgs)
        {
            std::array<double, 33> batch{};
            boys::BoysAllOrdersAtTier(kRungs[r].tier, boys::kMaxBoysOrder, x, batch.data());
            const double bound = kRungs[r].multiplier * DeclaredReachable(kRungs[r].tier, AccuracyRegion::kA);

            if (x >= boys::detail::kX0)
            {
                continue;
            }

            for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
            {
                ++cells;

                if (std::abs(boys::BoysSingleAtTier(kRungs[r].tier, n, x) - batch[static_cast<std::size_t>(n)]) >
                    bound)
                {
                    ++over;
                }
            }
        }
    }

    EXPECT_GT(cells, 0u);
    EXPECT_EQ(over, 0u) << "the two shapes have parted by more than the rung's own bound";
}

} // namespace
