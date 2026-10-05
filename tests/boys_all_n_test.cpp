// The all-orders batch entry (BoysAllN) contract tests: F_0(x_i)..F_nmax(x_i)
// over an array of arguments, with the per-argument dispatch and the grouping
// done internally - the batch shape a shell-quartet consumer needs.
//
// The entry's documented bound is 5.5e-14, the double batch lane's
// per-region budget and the bound the per-argument BoysAllOrders call meets.
// The suite pins it over the committed reference grid and against the
// per-argument path, both as observed maxima.

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

// The band's own bar. x < x0 is not one interval: region A carries the tight bar
// above and the band this one, because the band's arithmetic is a different fit.
constexpr double kBandBudget = 3e-14;

// The order at and below which a homogeneous region-A run goes to the region-A lane:
// above it the lane's per-order cost outweighs the body's single seed and recursion.
constexpr int kLaneMaxOrder = 4;

// The policy an unnamed BoysAllN call compiles: this entry's own class row.
//
// The sweeps below compare this entry against the per-argument BoysAllOrders entry, and
// the identity they pin is between the two ENTRIES - the batch's bodies are that entry's
// own, restructured onto this layout (include/boys/boys_impl.hpp). It is an identity at
// ONE policy and not across the class table: a replacement header that carries its own
// rows may give the fp64 all-n class a different combination from the fp64 all-orders
// class (boys/boys.hpp expands the one table the header carries), and two entries at two
// policies return two arithmetics by construction. So the per-argument side below is asked
// at THIS entry's class policy, which is the policy the unnamed call here compiles.
using AllNPolicy = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllN>;

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid: 45-digit mpmath values of F_n at each row's double x.
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

// The arguments as an array: every distinct x once, ascending, with the x -> index map.
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

// The entry's sub-regions, for the reported worsts: x = kX0 belongs to region B
// and x = kX1 to region C, because the region-A and region-B tests are strict.
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

// The bound the difference against the per-argument path is held to. The entry
// calls the per-argument path's own bodies on every path except region A below
// kLaneMaxOrder, where a lane serves it: exactly zero elsewhere, and there the
// lane's own 1e-15 below the band, the entry's 5.5e-14 over it - the lane
// evaluating the argument, not the band's scalar body.
double DifferenceBudget(double x, int nmax = boys::kMaxBoysOrder) {
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

    void Print(const char* label) const {
        std::printf("%s avx2=%d: worst zero %.3e A %.3e band %.3e B %.3e C %.3e\n",
                    label,
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

// One entry call: the sorted overload on ascending input, the merging overload otherwise.
std::vector<double> RunEntry(const Grid& grid,
                             bool shuffled,
                             bool sortedOverload,
                             int nmax = boys::kMaxBoysOrder) {
    const std::size_t count = grid.xs.size();
    std::vector<double> out(count * (static_cast<std::size_t>(nmax) + 1));

    if (sortedOverload)
    {
        BoysAllN(nmax, grid.xs.data(), out.data(), count, BoysSortedArgs{});
    } else if (shuffled)
    {
        BoysAllN(nmax, gShuffle.xs.data(), out.data(), count);
    } else
    {
        BoysAllN(nmax, grid.xs.data(), out.data(), count);
    }

    return out;
}

std::size_t ArgumentIndex(const Grid& grid, double x, bool shuffled) {
    const std::size_t j = grid.at.at(x);
    return shuffled ? gShuffle.position[j] : j;
}

// Grid accuracy: |entry value - reference| <= B_region per element, at the
// order-major plane offsets the layout contract names.

void SweepGrid(const Grid& grid, bool shuffled, bool sortedOverload, const char* label) {
    const std::size_t count = grid.xs.size();
    const std::vector<double> out = RunEntry(grid, shuffled, sortedOverload);
    Worsts worst;

    for (const ReferenceRow& row : grid.rows)
    {
        const std::size_t i = ArgumentIndex(grid, row.x, shuffled);
        const double got = out[static_cast<std::size_t>(row.n) * count + i];
        const double error = std::abs(got - row.value);
        EXPECT_LE(error, kBatchBound) << label << " n=" << row.n << " x=" << row.x
                                      << " got=" << got << " want=" << row.value;
        worst.Update(SubOf(row.x), error);
    }

    worst.Print(label);
}

// Difference against the per-argument path: exactly zero where a scalar body
// serves the path, the serving lane's budget where a region lane does.

void CheckAgainstPerArgument(const Grid& grid,
                             bool shuffled,
                             bool sortedOverload,
                             const char* label,
                             int nmax = boys::kMaxBoysOrder) {
    const std::size_t count = grid.xs.size();
    const std::vector<double> out = RunEntry(grid, shuffled, sortedOverload, nmax);
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

        // The per-argument path at the batch's own order: region A's body seeds at
        // nmax and recurses down, so the batch's F_k for k < nmax is the recurrence's.
        // Asked at the batch entry's own class policy (AllNPolicy above): the two entries
        // are two classes in the build's table, and a replacement may move one without
        // the other.
        BoysAllOrders<AllNPolicy>(nmax, row.x, want);
        const double diff =
            std::abs(out[static_cast<std::size_t>(row.n) * count + i] - want[row.n]);

        const double budget = DifferenceBudget(row.x, nmax);

        if (budget > 0.0)
        {
            groupedWorst = std::max(groupedWorst, diff);
            EXPECT_LE(diff, budget) << label << " n=" << row.n << " x=" << row.x;
        } else
        {
            scalarWorst = std::max(scalarWorst, diff);
            EXPECT_EQ(diff, 0.0) << label << " n=" << row.n << " x=" << row.x;
        }
    }

    std::printf("%s vs per-argument: bounded worst %.3e, exact-path worst %.3e\n",
                label,
                groupedWorst,
                scalarWorst);
}

// The tests.

TEST(BoysAllNTest, SortedOverloadGridSweepMatchesReferenceAtM1) {
    SweepGrid(gGrid, false, true, "sorted overload");
}

TEST(BoysAllNTest, MergingOverloadGridSweepMatchesReferenceAtM1) {
    SweepGrid(gGrid, false, false, "merging overload");
}

TEST(BoysAllNTest, ShuffledGridSweepMatchesReferenceAtM1) {
    SweepGrid(gGrid, true, false, "shuffled");
}

TEST(BoysAllNTest, DifferenceFromThePerArgumentPathAtM1) {
    CheckAgainstPerArgument(gGrid, false, true, "sorted overload");
    CheckAgainstPerArgument(gGrid, false, false, "merging overload");
    CheckAgainstPerArgument(gGrid, true, false, "shuffled");
}

// The purity contract: the values depend on the arguments and nmax alone, with no
// state carried between calls - a scratch buffer shared across calls would show here.
TEST(BoysAllNTest, RepeatedCallsWithDifferentShapesAreBitIdentical) {
    const std::vector<double> gridSorted = RunEntry(gGrid, false, true);
    const std::vector<double> gridShuffled = RunEntry(gGrid, true, false);
    const std::vector<double> boundary = RunEntry(gGrid, false, true, 1);

    EXPECT_EQ(RunEntry(gGrid, false, true), gridSorted);
    EXPECT_EQ(RunEntry(gGrid, true, false), gridShuffled);
    EXPECT_EQ(RunEntry(gGrid, false, true, 1), boundary);
}

// The lane route: a batch at or below kLaneMaxOrder hands its region-A runs to the
// region-A lane, so this shape is held to that lane's budgets, not to bit-identity;
// the crossover is covered both ways: kLaneMaxOrder on the lane, above it exact on the body.
TEST(BoysAllNTest, LaneRouteHoldsTheLaneBudgetAndItsNeighbourIsExact) {
    for (const int nmax : {0, 1, kLaneMaxOrder, kLaneMaxOrder + 1})
    {
        std::array<char, 48> label{};
        std::snprintf(label.data(), label.size(), "lane route nmax=%d", nmax);
        CheckAgainstPerArgument(gGrid, false, true, label.data(), nmax);
        CheckAgainstPerArgument(gGrid, true, false, label.data(), nmax);
    }
}

// The lane route's chunking: the grouped kernel stages a fixed number of arguments
// at a time, so a run that is not a whole number of chunks goes round that loop
// more than once, short chunk last, with a scalar tail inside every chunk. The grid's
// region-A run is shorter than one chunk, so these lengths cover the boundaries - every
// argument here is inside region A below the band, so the array is one run of one path.
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

// The grouping is a performance promise: an argument's value does not depend on its
// position in the input array. The worst difference is reported, not assumed zero:
// the lane serves its last count % 4 values scalar, so a value can move with its
// position, though by no more than the serving body's own bound.
TEST(BoysAllNTest, ShuffledAndAscendingAgreeWithinTheLaneBudget) {
    const std::size_t count = gGrid.xs.size();
    const std::vector<double> ascending = RunEntry(gGrid, false, true);
    const std::vector<double> shuffled = RunEntry(gGrid, true, false);
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

// The region boundaries of the dispatch, exact: x = 0, the band edge, x0 and x1 with
// their immediate neighbours, each argument on the far side of its comparison.
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
        BoysAllOrders<AllNPolicy>(boys::kMaxBoysOrder, xs[i], want);

        for (int k = 0; k <= boys::kMaxBoysOrder; ++k)
        {
            const double got = out[static_cast<std::size_t>(k) * xs.size() + i];

            const double budget = DifferenceBudget(xs[i]);

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
            const double budget = DifferenceBudget(x, nmax);
            std::vector<double> one(static_cast<std::size_t>(nmax) + 1);
            std::vector<double> want(static_cast<std::size_t>(nmax) + 1);
            BoysAllN(nmax, &x, one.data(), 1);
            BoysAllOrders<AllNPolicy>(nmax, x, want.data());

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

// The caller's workspace: the internal allocation's values, within the reported words.
TEST(BoysAllNTest, CallerWorkspaceIsEquivalentAndRespected) {
    const std::size_t count = gGrid.xs.size();
    const std::size_t words = BoysAllNWorkspaceSize(count);
    constexpr std::size_t kGuard = 8;
    constexpr std::size_t kGuardWord = 0x5a5a5a5a5a5a5a5aULL;
    std::vector<std::size_t> buffer(words + 2 * kGuard, kGuardWord);
    const std::vector<double> internal = RunEntry(gGrid, false, false);
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

    const std::vector<double> expectShuffled = RunEntry(gGrid, true, false);

    for (std::size_t slot = 0; slot < out.size(); ++slot)
    {
        EXPECT_EQ(out[slot], internal[slot]) << "slot " << slot;
        EXPECT_EQ(shuffledOut[slot], expectShuffled[slot]) << "slot " << slot;
    }
}

// The orders axis on this entry. The plane entry's call shape has both wide
// dimensions: an argument's whole order vector (out[k * count + i] is F_k(x[i])) and an
// order's whole argument array. A packed lane holds four doubles, so which of the two it
// packs is the axis, and this entry carries both. The tests below hold that the axis's
// values are the all-orders entry's own and that the region grouping, which exists for
// the arguments axis, is not taken here.

namespace {

// The policy the orders axis is named with on this entry.
template <boys::EvalScheme kScheme>
using OrdersAxisPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

// The certified scalar single lane the axis's fallback runs: the axis's own policy with
// the packing axis set to this shape's, the one cell the per-order entry carries
// (boys_impl.hpp asserts it, one order at one argument having no four orders to fill a
// lane with) and the cell ScalarOrders names when it hands its orders over one at a
// time - src/boys_orders_simd.cpp, "the certified scalar single lane at the policy the
// axis names". Every other cell is the axis's, so a rung and a form fall back to their
// own arithmetic and not to another's.
template <boys::EvalScheme kScheme>
using OrdersAxisSingleLane =
    boys::EvalPolicy<OrdersAxisPolicy<kScheme>::kRoute,
                     kScheme,
                     OrdersAxisPolicy<kScheme>::kBudget,
                     boys::PackAxis::kArguments,
                     OrdersAxisPolicy<kScheme>::kGranularity,
                     OrdersAxisPolicy<kScheme>::kDivision,
                     OrdersAxisPolicy<kScheme>::kRegionBExp>;

bool SameBits(double a, double b) {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// The entry's values under the orders axis, over one argument array, in the plane layout.
template <boys::EvalScheme kScheme>
std::vector<double> RunOrdersAxis(const std::vector<double>& xs, int nmax, bool sortedOverload) {
    const std::size_t count = xs.size();
    std::vector<double> out(count * (static_cast<std::size_t>(nmax) + 1));

    if (sortedOverload)
    {
        BoysAllN<OrdersAxisPolicy<kScheme>>(nmax, xs.data(), out.data(), count, BoysSortedArgs{});
    } else
    {
        BoysAllN<OrdersAxisPolicy<kScheme>>(nmax, xs.data(), out.data(), count);
    }

    return out;
}

} // namespace

// The default policy still names the committed axis, so a call site that names no
// axis compiles the entry it always did.
static_assert(boys::EvalPolicy<>{}.kPack == boys::PackAxis::kArguments,
              "the default axis moved: a call site that names no axis must compile the "
              "committed path");
static_assert(OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>::kPack == boys::PackAxis::kOrders,
              "the policy does not name the orders axis");

// The surface and the engine are the same code: every plane cell is the all-orders
// entry's value at that argument, bit for bit - either scheme, both overloads.
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
            boys::BoysAllOrders<OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>>(
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
            boys::BoysAllOrders<OrdersAxisPolicy<boys::EvalScheme::kHorner>>(
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

// Past the packed lane's own interval the axis runs the certified scalar single lane
// one order at a time, asserted exactly: the fallback claims that lane's values. The
// lane is the one at the policy the axis names - the policy this test asks the batch
// entry with - so the per-order side is asked at that same policy rather than at the
// single-order entry's own class row, which a replacement seam may move apart from it.
TEST(BoysAllNTest, OrdersAxisIsDefinedPastItsOwnDomain) {
    const std::vector<double> xs = {0.0, kX0, kX0 + 1e-9, 20.0, kX1, 31.0, 200.0};
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = xs.size();
    using AxisLane = OrdersAxisSingleLane<boys::EvalScheme::kSplitClenshaw>;
    const std::vector<double> planes =
        RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(xs, nmax, false);
    std::size_t differing = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            const double single = boys::BoysSingle<AxisLane>(l, xs[i]);
            const std::size_t slot = static_cast<std::size_t>(l) * count + i;

            if (!SameBits(planes[slot], single))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the axis's fallback is not the certified scalar single lane";
}

// The axis is reachable: naming it changes the values a caller receives in region A,
// since the committed entry reaches most region-A orders by a recursion from the batch
// seed where this axis evaluates each order's own fit. Both are inside the entry's
// bound, so what moves is which certified value a caller gets.
TEST(BoysAllNTest, OrdersAxisChangesTheRegionAValuesAndStaysInsideTheBound) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = gGrid.xs.size();
    const std::vector<double> committed = RunEntry(gGrid, false, true);
    const std::vector<double> axes =
        RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(gGrid.xs, nmax, true);
    std::size_t differingInA = 0;
    std::size_t cellsInA = 0;
    double worstInA = 0.0;
    double worstInBand = 0.0;

    for (const ReferenceRow& row : gGrid.rows)
    {
        if (!(row.x < kX0))
        {
            continue;
        }

        const std::size_t i = ArgumentIndex(gGrid, row.x, false);
        const std::size_t slot = static_cast<std::size_t>(row.n) * count + i;
        ++cellsInA;

        if (!SameBits(axes[slot], committed[slot]))
        {
            ++differingInA;
        }

        // Both values are certified, so the axis's own error is measured here against the
        // committed reference at the bar the argument's sub-region carries: region A and the
        // extended band sit inside x < x0 under different bars, worsts kept per sub-region.
        const double error = std::abs(axes[slot] - row.value);

        if (SubOf(row.x) == Sub::kBand)
        {
            worstInBand = std::max(worstInBand, error);
        }
        else
        {
            worstInA = std::max(worstInA, error);
        }
    }

    EXPECT_GT(cellsInA, 0u);
    EXPECT_GT(differingInA, 0u)
        << "the axis names a lane whose values are the committed entry's: the option is not "
           "reachable";
    EXPECT_LE(worstInA, kGroupedBudget)
        << "worst delivered " << worstInA << " in region A, against its own bar";
    EXPECT_LE(worstInBand, kBandBudget)
        << "worst delivered " << worstInBand << " in the extended band, against its own bar";
}

// The region grouping is the arguments axis's and is not taken here: an orders-axis call
// evaluates each argument on its own, so shuffling the arguments changes no plane.
TEST(BoysAllNTest, OrdersAxisTakesNoRegionGrouping) {
    const std::size_t count = gGrid.xs.size();
    const int nmax = boys::kMaxBoysOrder;
    const std::vector<double> ascending = RunOrdersAxis<boys::EvalScheme::kSplitClenshaw>(
        gGrid.xs, nmax, false);

    std::vector<double> shuffledOut(count * (static_cast<std::size_t>(nmax) + 1));
    BoysAllN<OrdersAxisPolicy<boys::EvalScheme::kSplitClenshaw>>(
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

// The route is carried on this entry too, by the same shape the orders axis takes: the
// per-argument path, whose body is the all-orders entry's own. A plane call naming the
// rational route returns that entry's planes bit for bit, and they differ from the committed
// ones over the intervals its rows cover: the carriage is a measurement, not a sentence.
TEST(BoysAllNTest, TheRationalRouteIsCarriedAndIsThePerArgumentEntry) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = gGrid.xs.size();
    std::vector<double> got(count * (static_cast<std::size_t>(nmax) + 1));
    boys::BoysAllN<boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
        nmax, gGrid.xs.data(), got.data(), count);
    std::size_t differingFromEntry = 0;
    std::size_t differingFromCommitted = 0;

    for (std::size_t i = 0; i < count; ++i)
    {
        std::array<double, 33> rational{};
        std::array<double, 33> committed{};
        boys::BoysAllOrders<boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
            nmax, gGrid.xs[i], rational.data());
        boys::BoysAllOrders<boys::EvalPolicy<>>(nmax, gGrid.xs[i], committed.data());

        for (int l = 0; l <= nmax; ++l)
        {
            const std::size_t slot = static_cast<std::size_t>(l) * count + i;
            const std::size_t n = static_cast<std::size_t>(l);

            if (!SameBits(got[slot], rational[n]))
            {
                ++differingFromEntry;
            }

            if (!SameBits(rational[n], committed[n]))
            {
                ++differingFromCommitted;
            }
        }
    }

    EXPECT_EQ(differingFromEntry, 0u)
        << "the plane entry's route carriage is not the per-argument all-orders body";
    EXPECT_GT(differingFromCommitted, 0u)
        << "the rational route returns the committed values: the carriage is not reachable";
}

TEST(BoysAllNTest, TheRouteNamedIsTheRowsAndNotTheSeams) {
    const int nmax = boys::kMaxBoysOrder;
    const std::size_t count = gGrid.xs.size();
    std::vector<double> got(count * (static_cast<std::size_t>(nmax) + 1));
    boys::BoysAllN<boys::EvalPolicy<boys::kDefaultFitRoute>>(
        nmax, gGrid.xs.data(), got.data(), count);
    const std::vector<double> plain = RunEntry(gGrid, false, true);
    std::size_t differing = 0;

    for (std::size_t k = 0; k < got.size(); ++k)
    {
        if (!SameBits(got[k], plain[k]))
        {
            ++differing;
        }
    }

    // Which arithmetic this entry's unnamed call compiles: the class's own row, which is
    // the seam's five exactly where the row spells them. The five-composed table (the
    // committed header, and every fixture that names no rows) spells them for every class,
    // and a replacement that carries its own row list need not.
    constexpr bool kRowIsTheSeamFive =
        AllNPolicy::kRoute == boys::kDefaultFitRoute &&
        AllNPolicy::kScheme == boys::kDefaultEvalScheme &&
        AllNPolicy::kPack == boys::kDefaultPackAxis &&
        AllNPolicy::kGranularity == boys::kDefaultFitGranularity &&
        AllNPolicy::kDivision == boys::kDefaultDivisionForm;

    std::printf("naming the seam's route over %zu values: %zu differ, and this class's row %s "
                "the seam's five\n",
                got.size(),
                differing,
                kRowIsTheSeamFive ? "is" : "is not");

    // The claim both ways: naming the seam's route (with the seam's other four axes) is the
    // unnamed call exactly where the class's row is the seam's five, and is a different
    // arithmetic where the row moves the class. An entry resolving through the five while
    // its row names something else agrees here when it must not.
    EXPECT_EQ(differing == 0u, kRowIsTheSeamFive)
        << "naming the route the seam's five name does not move the unnamed call where this "
           "class's row is those five, or moves it nowhere where the row names another "
           "combination: the entry is not resolving through the row its class carries";
}

} // namespace
