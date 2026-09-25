// The all-orders batch entry (BoysAllN) contract tests: F_0(x_i)..F_nmax(x_i)
// over an array of arguments, with the per-argument dispatch and the grouping
// done internally - the batch shape a shell-quartet consumer needs.
//
// The entry's documented bound is the double batch lane's per-region budget
// (m * 5.5e-14 everywhere), the bound the per-argument BoysAllOrders call
// meets, and the suite pins both sides of it over the committed reference grid:
// the grid sweep at every sampled multiplier, and the difference against the
// per-argument path. The paths the scalar bodies serve (the zero path, region B,
// region C) hold that difference at exactly zero - the entry calls the
// per-argument path's own bodies; the region-A path is served by the region-A
// lane at m = 1 on an AVX2 host, below the band within that lane's own 1e-15 and
// inside the band within the entry's bound, and the suite reports the observed
// maxima both ways.
//
// The sorted overload's declaration is exercised as a merging overload too: the
// same argument set shuffled and passed to the unsorted entry, against the
// sorted entry's own result, with the observed difference reported - the
// grouping is a performance promise, so equal values are not claimed bitwise
// where a lane's tail can serve a run of a different length.
//
// The sampled-m instantiations compile from the internal headers, like the rest
// of the accuracy suite; the m = 1 call sites below route to the library's
// certified instantiation (extern-template surface in boys.hpp, explicit
// instantiation in boys.cpp).

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using boys::BoysAllN;
using boys::BoysAllNWorkspaceSize;
using boys::BoysAllOrders;
using boys::BoysSortedArgs;
using boys::detail::kTierThresholds;
using boys::detail::kX0;
using boys::detail::kX1;

// The entry's documented budget: the double batch lane's per-region bound, the
// same value in every region (the batch row of the contract table).
constexpr double kBatchBound = 5.5e-14;

// The lanes' budget on the paths they serve (the region-A lane's 1e-15).
constexpr double kGroupedBudget = 1e-15;

// The order at which region A stops being the region-A lane's run: at and below
// it the entry hands a homogeneous run to the lane, above it the lane's
// per-order cost outweighs the scalar body's single seed and downward
// recursion. The lane route is what this suite's low-order cases exercise.
constexpr int kLaneMaxOrder = 4;

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid (tools/gen_boys_coefficients.py, 45-digit
// mpmath values of F_n at the double in each row's x column) - the same
// loader as the other suites.
std::vector<ReferenceRow> LoadReference() {
    const std::string path = std::string(BoysDataDir) + "/boys_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<ReferenceRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        ReferenceRow row{};
        std::getline(ss, cell, ',');
        row.n = std::stoi(cell);
        std::getline(ss, cell, ',');
        row.x = std::strtod(cell.c_str(), nullptr);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);

        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

// The grid as an argument array: every distinct x once, ascending, with the
// row -> argument index map the sweeps read their columns through.
struct Grid {
    std::vector<ReferenceRow> rows;
    std::vector<double> xs;
    std::unordered_map<double, std::size_t> at;
};

Grid BuildGrid() {
    Grid grid;
    grid.rows = LoadReference();

    for (const ReferenceRow& row : grid.rows)
    {
        grid.xs.push_back(row.x);
    }

    std::sort(grid.xs.begin(), grid.xs.end());
    grid.xs.erase(std::unique(grid.xs.begin(), grid.xs.end()), grid.xs.end());

    for (std::size_t i = 0; i < grid.xs.size(); ++i)
    {
        grid.at[grid.xs[i]] = i;
    }

    return grid;
}

const Grid gGrid = BuildGrid();

// The entry's sub-regions, for the reported worsts: the band split is the
// m = 1 dispatch's own, and x = kX0 belongs to region B (the region-A test is
// strict) while x = kX1 belongs to region C.
enum class Sub : std::uint8_t { kZero, kA, kBand, kB, kC };

Sub SubOf(double x) {
    if (x == 0.0)
    {
        return Sub::kZero;
    }

    if (x < kTierThresholds[0])
    {
        return Sub::kA;
    }

    if (x < kX0)
    {
        return Sub::kBand;
    }

    if (x < kX1)
    {
        return Sub::kB;
    }

    return Sub::kC;
}

// The bound the entry's difference against the per-argument path is held to.
//
// At m > 1 no lane is entered: both sides are relaxed scalar bodies, each
// documented at m * 5.5e-14 against the true value, so the difference between
// them is held to the sum of the two.
//
// At m = 1 the entry calls the per-argument path's own bodies for every path
// except region A below the crossover order kLaneMaxOrder, where it hands the
// run to the region-A lane; on every other path the difference is exactly zero.
// On the lane route region A is the lane's, below the band within that lane's
// own 1e-15 and inside the band within the entry's 5.5e-14 - the lane evaluates
// the argument rather than the band's scalar body, and the lane's own 1e-15 does
// not hold over the band.
double DifferenceBudget(double x, double m, int nmax = boys::kMaxBoysOrder) {
    if (m != 1.0)
    {
        return 2.0 * m * kBatchBound;
    }

    if (!boys::BoysAvx2Available() || nmax > kLaneMaxOrder)
    {
        return 0.0;
    }

    switch (SubOf(x))
    {
    case Sub::kA:
        return kGroupedBudget;

    case Sub::kBand:
        return kBatchBound;

    default:
        return 0.0;
    }
}

struct Worsts {
    std::array<double, 5> bySub{};

    void Update(Sub sub, double error) {
        double& slot = bySub[static_cast<std::size_t>(sub)];
        slot = std::max(slot, error);
    }

    void Print(const char* label, double m) const {
        std::printf("%s m=%.0e avx2=%d: worst zero %.3e A %.3e band %.3e B %.3e C %.3e\n",
                    label,
                    m,
                    boys::BoysAvx2Available() ? 1 : 0,
                    bySub[0],
                    bySub[1],
                    bySub[2],
                    bySub[3],
                    bySub[4]);
    }
};

// The grid's arguments in a shuffled order, with the argument -> position map.
struct Shuffle {
    std::vector<double> xs;
    std::vector<std::size_t> position;
};

Shuffle MakeShuffle(const Grid& grid) {
    const std::size_t count = grid.xs.size();
    std::vector<std::size_t> place(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        place[i] = i;
    }

    std::mt19937_64 rng(20260921);
    std::shuffle(place.begin(), place.end(), rng);

    Shuffle shuffle;
    shuffle.xs.resize(count);
    shuffle.position.resize(count);

    for (std::size_t t = 0; t < count; ++t)
    {
        shuffle.xs[t] = grid.xs[place[t]];
        shuffle.position[place[t]] = t;
    }

    return shuffle;
}

const Shuffle gShuffle = MakeShuffle(gGrid);

// One entry call over the grid's arguments: the sorted overload on the
// ascending array, or the merging overload on the shuffled one.
template <double kM>
std::vector<double> RunEntry(const Grid& grid,
                             bool shuffled,
                             bool sortedOverload,
                             int nmax = boys::kMaxBoysOrder) {
    const std::size_t count = grid.xs.size();
    std::vector<double> out(count * (static_cast<std::size_t>(nmax) + 1));

    if (sortedOverload)
    {
        BoysAllN<kM>(nmax, grid.xs.data(), out.data(), count, BoysSortedArgs{});
    } else if (shuffled)
    {
        BoysAllN<kM>(nmax, gShuffle.xs.data(), out.data(), count);
    } else
    {
        BoysAllN<kM>(nmax, grid.xs.data(), out.data(), count);
    }

    return out;
}

std::size_t ArgumentIndex(const Grid& grid, double x, bool shuffled) {
    const std::size_t j = grid.at.at(x);
    return shuffled ? gShuffle.position[j] : j;
}

// ---------------------------------------------------------------------------
// Grid accuracy: |entry value - reference| <= m * B_region per element, at the
// order-major planes offset the layout contract names
// ---------------------------------------------------------------------------

template <double kM>
void SweepGrid(const Grid& grid, bool shuffled, bool sortedOverload, const char* label) {
    const std::size_t count = grid.xs.size();
    const std::vector<double> out = RunEntry<kM>(grid, shuffled, sortedOverload);
    Worsts worst;

    for (const ReferenceRow& row : grid.rows)
    {
        const std::size_t i = ArgumentIndex(grid, row.x, shuffled);
        const double got = out[static_cast<std::size_t>(row.n) * count + i];
        const double error = std::abs(got - row.value);
        EXPECT_LE(error, kM * kBatchBound)
            << label << " m=" << kM << " n=" << row.n << " x=" << row.x << " got=" << got
            << " want=" << row.value;
        worst.Update(SubOf(row.x), error);
    }

    worst.Print(label, kM);
}

// ---------------------------------------------------------------------------
// Difference against the per-argument path: exactly zero where a scalar body
// serves the path, the serving lane's budget where a region lane does
// ---------------------------------------------------------------------------

template <double kM>
void CheckAgainstPerArgument(const Grid& grid,
                             bool shuffled,
                             bool sortedOverload,
                             const char* label,
                             int nmax = boys::kMaxBoysOrder) {
    const std::size_t count = grid.xs.size();
    const std::vector<double> out = RunEntry<kM>(grid, shuffled, sortedOverload, nmax);
    double groupedWorst = 0.0;
    double scalarWorst = 0.0;

    for (const ReferenceRow& row : grid.rows)
    {
        if (row.n > nmax)
        {
            continue;
        }

        const std::size_t i = ArgumentIndex(grid, row.x, shuffled);
        double want[boys::kMaxBoysOrder + 1];

        // The per-argument path at the batch's own order: region A's body seeds
        // at nmax and recurses down, so the batch's F_k for k < nmax is the
        // recurrence's, not the one a per-order call at nmax = k walks.
        BoysAllOrders<kM>(nmax, row.x, want);
        const double diff =
            std::abs(out[static_cast<std::size_t>(row.n) * count + i] - want[row.n]);

        const double budget = DifferenceBudget(row.x, kM, nmax);

        if (budget > 0.0)
        {
            groupedWorst = std::max(groupedWorst, diff);
            EXPECT_LE(diff, budget) << label << " m=" << kM << " n=" << row.n << " x=" << row.x;
        } else
        {
            scalarWorst = std::max(scalarWorst, diff);
            EXPECT_EQ(diff, 0.0) << label << " m=" << kM << " n=" << row.n << " x=" << row.x;
        }
    }

    std::printf("%s m=%.0e vs per-argument: bounded worst %.3e, exact-path worst %.3e\n",
                label,
                kM,
                groupedWorst,
                scalarWorst);
}

// The sampled-m set of the accuracy suite.
template <typename Fn> void ForEachSampledMultiplier(Fn&& fn) {
    fn.template operator()<1.0>();
    fn.template operator()<2.0>();
    fn.template operator()<10.0>();
    fn.template operator()<100.0>();
    fn.template operator()<1e4>();
    fn.template operator()<1e8>();
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

TEST(BoysAllNTest, SortedOverloadGridSweepMatchesReferenceAtM1) {
    SweepGrid<1.0>(gGrid, false, true, "sorted overload");
}

TEST(BoysAllNTest, MergingOverloadGridSweepMatchesReferenceAtM1) {
    SweepGrid<1.0>(gGrid, false, false, "merging overload");
}

TEST(BoysAllNTest, ShuffledGridSweepMatchesReferenceAtM1) {
    SweepGrid<1.0>(gGrid, true, false, "shuffled");
}

TEST(BoysAllNTest, GridSweepMatchesReferenceAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        if constexpr (kM != 1.0)
        {
            SweepGrid<kM>(gGrid, false, true, "sorted overload");
            SweepGrid<kM>(gGrid, true, false, "shuffled");
        }
    });
}

TEST(BoysAllNTest, DifferenceFromThePerArgumentPathAtM1) {
    CheckAgainstPerArgument<1.0>(gGrid, false, true, "sorted overload");
    CheckAgainstPerArgument<1.0>(gGrid, false, false, "merging overload");
    CheckAgainstPerArgument<1.0>(gGrid, true, false, "shuffled");
}

TEST(BoysAllNTest, DifferenceFromThePerArgumentPathAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        if constexpr (kM != 1.0)
        {
            CheckAgainstPerArgument<kM>(gGrid, false, true, "sorted overload");
            CheckAgainstPerArgument<kM>(gGrid, true, false, "shuffled");
        }
    });
}

// The purity contract: the values depend on the arguments and nmax alone, with
// no state carried between calls. Three different shapes are run in sequence and
// then again, and each array's second result must equal its first, bit for bit -
// a cache or scratch buffer shared across calls would show up here as an
// argument set's value changing with what the previous call looked like.
TEST(BoysAllNTest, RepeatedCallsWithDifferentShapesAreBitIdentical) {
    const std::vector<double> gridSorted = RunEntry<1.0>(gGrid, false, true);
    const std::vector<double> gridShuffled = RunEntry<1.0>(gGrid, true, false);
    const std::vector<double> boundary = RunEntry<1.0>(gGrid, false, true, 1);

    EXPECT_EQ(RunEntry<1.0>(gGrid, false, true), gridSorted);
    EXPECT_EQ(RunEntry<1.0>(gGrid, true, false), gridShuffled);
    EXPECT_EQ(RunEntry<1.0>(gGrid, false, true, 1), boundary);
}

// The lane route. A batch at or below kLaneMaxOrder hands its region-A runs to
// the region-A lane, so this is the shape that serves them: it is held to the
// lane's budgets rather than to bit-identity, with the crossover order's two
// sides both covered - kLaneMaxOrder on the lane, the order above it on the
// scalar body, which must be exact.
TEST(BoysAllNTest, LaneRouteHoldsTheLaneBudgetAndItsNeighbourIsExact) {
    for (const int nmax : {0, 1, kLaneMaxOrder, kLaneMaxOrder + 1})
    {
        std::array<char, 48> label{};
        std::snprintf(label.data(), label.size(), "lane route nmax=%d", nmax);
        CheckAgainstPerArgument<1.0>(gGrid, false, true, label.data(), nmax);
        CheckAgainstPerArgument<1.0>(gGrid, true, false, label.data(), nmax);
    }
}

// The lane route's chunking. The grouped kernel stages a fixed number of
// arguments at a time, so a region-A run that is not a whole number of chunks
// takes that loop round more than once, with a short chunk last and the lane's
// own scalar tail inside every chunk. The committed grid's region-A run is
// shorter than one chunk, so these lengths are what cover the boundaries: the
// chunk size less one, the chunk size, one past it, and a run of several chunks.
// Every argument is inside region A below the band, so the whole array is one
// run of one path.
TEST(BoysAllNTest, LaneRouteChunkBoundariesOverALongRun) {
    constexpr int kNmax = kLaneMaxOrder;

    for (const std::size_t count : {std::size_t{1},
                                    std::size_t{5},
                                    std::size_t{127},
                                    std::size_t{128},
                                    std::size_t{129},
                                    std::size_t{300},
                                    std::size_t{1024}})
    {
        std::vector<double> x(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            x[i] = 1e-3 + (1.0 - 1e-3) * static_cast<double>(i) / static_cast<double>(count);
        }

        std::vector<double> out(count * (static_cast<std::size_t>(kNmax) + 1));
        BoysAllN(kNmax, x.data(), out.data(), count);
        double worst = 0.0;

        for (std::size_t i = 0; i < count; ++i)
        {
            double want[boys::kMaxBoysOrder + 1];
            BoysAllOrders(kNmax, x[i], want);

            for (int k = 0; k <= kNmax; ++k)
            {
                const double got = out[static_cast<std::size_t>(k) * count + i];
                worst = std::max(worst, std::abs(got - want[k]));
            }
        }

        EXPECT_LE(worst, kGroupedBudget) << "count=" << count;
        std::printf("lane route count=%zu: worst %.3e (budget %.0e)\n",
                    count,
                    worst,
                    kGroupedBudget);
    }
}

// The grouping is a performance promise: the entry's value for an argument does
// not depend on the argument's position in the input array, so the same set in
// shuffled order and in ascending order agree per argument, and the observed
// maximum is reported rather than assumed to be zero - the lane's own last
// count % 4 values are scalar, so a value can change with the position it is
// served at, by no more than the serving body's own bound.
TEST(BoysAllNTest, ShuffledAndAscendingAgreeWithinTheLaneBudget) {
    const std::size_t count = gGrid.xs.size();
    const std::vector<double> ascending = RunEntry<1.0>(gGrid, false, true);
    const std::vector<double> shuffled = RunEntry<1.0>(gGrid, true, false);
    double worst = 0.0;
    int worstN = -1;
    double worstX = 0.0;

    for (const ReferenceRow& row : gGrid.rows)
    {
        const std::size_t a = ArgumentIndex(gGrid, row.x, false);
        const std::size_t s = ArgumentIndex(gGrid, row.x, true);
        const double difference =
            std::abs(shuffled[static_cast<std::size_t>(row.n) * count + s]
                     - ascending[static_cast<std::size_t>(row.n) * count + a]);

        if (difference > worst)
        {
            worst = difference;
            worstN = row.n;
            worstX = row.x;
        }
    }

    std::printf("shuffled vs ascending: worst difference %.3e (n=%d x=%.17g)\n",
                worst,
                worstN,
                worstX);
    EXPECT_LE(worst, kBatchBound);
}

// The region boundaries of the dispatch, exact: x = 0, the band edge, x0 and
// x1 as the grid's own boundary rows, and their immediate neighbours - the
// argument that must fall on the far side of each comparison.
TEST(BoysAllNTest, ExactBoundaryArgumentsMatchThePerArgumentPath) {
    const double bandEdge = kTierThresholds[0];
    const std::array<double, 8> xs = {
        0.0,
        bandEdge,
        std::nextafter(bandEdge, 0.0),
        std::nextafter(bandEdge, kX0),
        kX0,
        std::nextafter(kX0, 0.0),
        kX1,
        std::nextafter(kX1, 2.0 * kX1),
    };
    std::vector<double> out(xs.size() * (boys::kMaxBoysOrder + 1));

    BoysAllN(boys::kMaxBoysOrder, xs.data(), out.data(), xs.size());

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        double want[boys::kMaxBoysOrder + 1];
        BoysAllOrders(boys::kMaxBoysOrder, xs[i], want);

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const double got = out[static_cast<std::size_t>(k) * xs.size() + i];

            const double budget = DifferenceBudget(xs[i], 1.0);

            if (budget > 0.0)
            {
                EXPECT_LE(std::abs(got - want[k]), budget) << "x=" << xs[i] << " k=" << k;
            } else
            {
                EXPECT_EQ(got, want[k]) << "x=" << xs[i] << " k=" << k;
            }
        }
    }

    // The exact boundary rows of the committed grid itself.
    for (const double x : {kTierThresholds[0], kX0, kX1})
    {
        EXPECT_NE(gGrid.at.find(x), gGrid.at.end()) << "grid boundary row missing: " << x;
    }
}

// count = 0 writes nothing at either overload; count = 1 gives exactly the
// per-argument column at nmax = 0 and at nmax = kMaxBoysOrder.
TEST(BoysAllNTest, DegenerateCountsAndOrders) {
    constexpr double kSentinel = 1.2345678901234567e100;
    const std::array<double, 4> args = {0.0, 1.5, 12.5, 30.0};
    std::vector<double> empty(args.size() * (boys::kMaxBoysOrder + 1), kSentinel);

    BoysAllN(boys::kMaxBoysOrder, args.data(), empty.data(), 0);
    BoysAllN(boys::kMaxBoysOrder, args.data(), empty.data(), 0, BoysSortedArgs{});

    for (const double value : empty)
    {
        EXPECT_EQ(value, kSentinel);
    }

    for (const double x : args)
    {
        for (const int nmax : {0, boys::kMaxBoysOrder})
        {
            const double budget = DifferenceBudget(x, 1.0, nmax);
            std::vector<double> one(static_cast<std::size_t>(nmax) + 1);
            std::vector<double> want(static_cast<std::size_t>(nmax) + 1);
            BoysAllN(nmax, &x, one.data(), 1);
            BoysAllOrders(nmax, x, want.data());

            for (int k = 0; k <= nmax; ++k)
            {
                const std::size_t slot = static_cast<std::size_t>(k);

                if (budget > 0.0)
                {
                    EXPECT_LE(std::abs(one[slot] - want[slot]), budget)
                        << "x=" << x << " nmax=" << nmax << " k=" << k;
                } else
                {
                    EXPECT_EQ(one[slot], want[slot]) << "x=" << x << " nmax=" << nmax << " k=" << k;
                }
            }
        }
    }
}

// The caller's workspace: the same values as the internal allocation, and the
// entry stays inside the words the size function reports.
TEST(BoysAllNTest, CallerWorkspaceIsEquivalentAndRespected) {
    const std::size_t count = gGrid.xs.size();
    const std::size_t words = BoysAllNWorkspaceSize(count);
    constexpr std::size_t kGuard = 8;
    constexpr std::size_t kGuardWord = 0x5a5a5a5a5a5a5a5aULL;
    std::vector<std::size_t> buffer(words + 2 * kGuard, kGuardWord);
    const std::vector<double> internal = RunEntry<1.0>(gGrid, false, false);
    std::vector<double> out(count * (boys::kMaxBoysOrder + 1));
    std::vector<double> shuffledOut(count * (boys::kMaxBoysOrder + 1));

    BoysAllN(boys::kMaxBoysOrder, gGrid.xs.data(), out.data(), count, buffer.data() + kGuard);
    BoysAllN(
        boys::kMaxBoysOrder, gShuffle.xs.data(), shuffledOut.data(), count, buffer.data() + kGuard);

    for (std::size_t w = 0; w < kGuard; ++w)
    {
        EXPECT_EQ(buffer[w], kGuardWord) << "guard word below the workspace";
        EXPECT_EQ(buffer[words + kGuard + w], kGuardWord) << "guard word above the workspace";
    }

    const std::vector<double> expectShuffled = RunEntry<1.0>(gGrid, true, false);

    for (std::size_t slot = 0; slot < out.size(); ++slot)
    {
        EXPECT_EQ(out[slot], internal[slot]) << "slot " << slot;
        EXPECT_EQ(shuffledOut[slot], expectShuffled[slot]) << "slot " << slot;
    }
}

// The mapped-m surface: the relaxed bodies are the per-argument entry's, so a
// workspace call and an internal-allocation call agree bit for bit at every
// sampled multiplier on both argument orders.
TEST(BoysAllNTest, WorkspaceEquivalenceAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        const std::size_t count = gGrid.xs.size();
        std::vector<std::size_t> workspace(BoysAllNWorkspaceSize(count));
        std::vector<double> out(count * (boys::kMaxBoysOrder + 1));
        std::vector<double> want = RunEntry<kM>(gGrid, true, false);

        BoysAllN<kM>(boys::kMaxBoysOrder,
                     gShuffle.xs.data(),
                     out.data(),
                     count,
                     workspace.data());

        for (std::size_t slot = 0; slot < out.size(); ++slot)
        {
            EXPECT_EQ(out[slot], want[slot]) << "m=" << kM << " slot " << slot;
        }
    });
}

// ---------------------------------------------------------------------------
// The orders axis on this entry
// ---------------------------------------------------------------------------
// The plane entry's call shape has both wide dimensions: an argument's whole
// order vector (out[k * count + i] is F_k(x[i])) and an order's whole argument
// array. A packed lane keeps four doubles in a register, so which of the two it
// holds is the packing axis, and this entry carries both. The axis reaches the
// per-argument path, whose body is the all-orders entry's own, so what the
// tests below hold is that the surface and that path are the same code - and
// that the region grouping, which exists to feed the arguments-axis lane, is
// not taken when the axis named packs orders instead.

namespace {

// The policy the orders axis is named with on this entry.
template <boys::EvalScheme kScheme>
using OrdersAxisPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

bool SameBits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// The entry's values under the orders axis, over one argument array, in the
// plane layout.
template <boys::EvalScheme kScheme>
std::vector<double> RunOrdersAxis(const std::vector<double>& xs, int nmax, bool sortedOverload) {
    const std::size_t count = xs.size();
    std::vector<double> out(count * (static_cast<std::size_t>(nmax) + 1));

    if (sortedOverload)
    {
        BoysAllN<1.0, OrdersAxisPolicy<kScheme>>(
            nmax, xs.data(), out.data(), count, BoysSortedArgs{});
    } else
    {
        BoysAllN<1.0, OrdersAxisPolicy<kScheme>>(nmax, xs.data(), out.data(), count);
    }

    return out;
}

} // namespace

// The default policy still names the shipped axis, so a call site that names no
// axis compiles the entry it always did.
static_assert(boys::EvalPolicy<>{}.kPack == boys::PackAxis::kArguments,
              "the default axis moved: a call site that names no axis must compile the "
              "shipped path");
static_assert(OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>::kPack == boys::PackAxis::kOrders,
              "the policy does not name the orders axis");

// The surface and the engine are the same code: every plane cell is the
// all-orders entry's value at that argument, bit for bit, under either scheme
// and on both overloads.
TEST(BoysAllNTest, OrdersAxisIsThePerArgumentEntryBitForBit) {
    const std::vector<double> xs = gGrid.xs;
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = xs.size();

    for (const bool sortedOverload : {false, true})
    {
        const std::vector<double> planes =
            RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(xs, nmax, sortedOverload);
        const std::vector<double> horner =
            RunOrdersAxis<boys::EvalScheme::kHorner>(xs, nmax, sortedOverload);
        std::size_t differingClenshaw = 0;
        std::size_t differingHorner = 0;

        for (std::size_t i = 0; i < count; ++i)
        {
            std::vector<double> row(static_cast<std::size_t>(nmax) + 1);
            boys::BoysAllOrders<1.0, OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>>(
                nmax, xs[i], row.data());

            for (int l = 0; l <= nmax; ++l)
            {
                const std::size_t slot = static_cast<std::size_t>(l) * count + i;

                if (!SameBits(planes[slot], row[static_cast<std::size_t>(l)]))
                {
                    ++differingClenshaw;
                }
            }

            std::vector<double> rowH(static_cast<std::size_t>(nmax) + 1);
            boys::BoysAllOrders<1.0, OrdersAxisPolicy<boys::EvalScheme::kHorner>>(
                nmax, xs[i], rowH.data());

            for (int l = 0; l <= nmax; ++l)
            {
                const std::size_t slot = static_cast<std::size_t>(l) * count + i;

                if (!SameBits(horner[slot], rowH[static_cast<std::size_t>(l)]))
                {
                    ++differingHorner;
                }
            }
        }

        EXPECT_EQ(differingClenshaw, 0u)
            << "sorted overload = " << sortedOverload
            << ": the plane entry and the all-orders entry have parted on the orders axis";
        EXPECT_EQ(differingHorner, 0u) << "sorted overload = " << sortedOverload;
    }
}

// Past the packed lane's own interval the axis runs the certified scalar single
// lane one order at a time, and that is asserted exactly rather than against a
// tolerance: the fallback's whole claim is that its values are that lane's.
TEST(BoysAllNTest, OrdersAxisIsDefinedPastItsOwnDomain) {
    const std::vector<double> xs = {0.0, kX0, kX0 + 1e-9, 20.0, kX1, 31.0, 200.0};
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = xs.size();
    const std::vector<double> planes =
        RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(xs, nmax, false);
    std::size_t differing = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            const double single = boys::BoysSingle(l, xs[i]);
            const std::size_t slot = static_cast<std::size_t>(l) * count + i;

            if (!SameBits(planes[slot], single))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the axis's fallback is not the certified scalar single lane";
}

// The axis is reachable: naming it changes the values a caller receives in
// region A, which is the whole of what the option is. The two answers differ
// because the shipped entry reaches most region-A orders by a recursion from
// the batch seed where this axis evaluates each order's own fit - and both are
// inside the entry's own bound, so what moves is which of two certified values
// a caller gets.
TEST(BoysAllNTest, OrdersAxisChangesTheRegionAValuesAndStaysInsideTheBound) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = gGrid.xs.size();
    const std::vector<double> shipped = RunEntry<1.0>(gGrid, false, true);
    const std::vector<double> axes =
        RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(gGrid.xs, nmax, true);
    std::size_t differingInA = 0;
    std::size_t cellsInA = 0;
    double worstAlongTheAxis = 0.0;

    for (const ReferenceRow& row : gGrid.rows)
    {
        if (!(row.x < kX0))
        {
            continue;
        }

        const std::size_t i = ArgumentIndex(gGrid, row.x, false);
        const std::size_t slot = static_cast<std::size_t>(row.n) * count + i;
        ++cellsInA;

        if (!SameBits(axes[slot], shipped[slot]))
        {
            ++differingInA;
        }

        // Both values are certified, so the axis's own error is measured here
        // rather than assumed: against the committed reference, at the packed
        // lane's per-order bar over the interval the lane evaluates.
        worstAlongTheAxis = std::max(worstAlongTheAxis, std::abs(axes[slot] - row.value));
    }

    EXPECT_GT(cellsInA, 0u);
    EXPECT_GT(differingInA, 0u)
        << "the axis names a lane whose values are the shipped entry's: the option is not "
           "reachable";
    EXPECT_LE(worstAlongTheAxis, kGroupedBudget)
        << "worst delivered " << worstAlongTheAxis << " over the packed lane's own bar";
}

// The region grouping is the arguments axis's and is not taken here: the same
// arguments shuffled, in ascending order, and through the sorted overload all
// return the same planes, because an orders-axis call evaluates each argument
// on its own and has nothing to group.
TEST(BoysAllNTest, OrdersAxisTakesNoRegionGrouping) {
    const std::size_t count = gGrid.xs.size();
    const int nmax = boys::kMaxBoysOrder;
    const std::vector<double> ascending = RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(
        gGrid.xs, nmax, false);

    std::vector<double> shuffledOut(count * (static_cast<std::size_t>(nmax) + 1));
    BoysAllN<1.0, OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>>(
        nmax, gShuffle.xs.data(), shuffledOut.data(), count);
    std::size_t differingShuffled = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        const std::size_t j = ArgumentIndex(gGrid, gGrid.xs[i], true);

        for (int l = 0; l <= nmax; ++l)
        {
            const std::size_t slot = static_cast<std::size_t>(l) * count;

            if (!SameBits(ascending[slot + i], shuffledOut[slot + j]))
            {
                ++differingShuffled;
            }
        }
    }

    EXPECT_EQ(differingShuffled, 0u) << "the axis's values moved with the argument order";
}

} // namespace
