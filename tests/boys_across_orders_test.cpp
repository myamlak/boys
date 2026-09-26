// The across-orders packed lane's contract tests: BoysAllOrdersSimd, the
// entry that vectorises four ORDERS of one argument where the across-arguments
// lane vectorises four arguments of one order.
//
// What can go wrong with a lane that re-sums a stored fit in a different
// order, and what each test is for:
//
//  1. A TRANSCRIPTION SLIP. The split Clenshaw body is a second copy of the
//     one boys_impl.hpp holds, and a reordered step, a fused operation turned
//     into two, or a coefficient read one index out would all still return a
//     plausible number. The lane's answer for order l is therefore asserted
//     BIT-IDENTICAL to the across-arguments lane's answer for the same order,
//     which runs the library's own body over four copies of the same argument.
//     One exact comparison, no tolerance: a tolerance is the slack a slip
//     would hide in.
//
//  2. A BOUND NOT MET. The direct sum and Horner are different summations of
//     the same fits and have their own error. Each scheme is measured against
//     the committed reference grid - not against the library - over every
//     region-A cell of it.
//
//  3. A LAYOUT THAT SILENTLY TRANSPOSES. The entry writes at out[l * stride]
//     so the plane entry's shape can use it; the strided write is asserted to
//     carry the same bits as the contiguous one.
//
//  4. A ROUTE THAT DISAGREES WITH THE SHIPPED ONE. The lane reaches no order
//     by recursion where BoysAllOrders reaches most of them that way, so the
//     two differ by their routes' arithmetic. The difference is measured and
//     the file states it rather than leaving it to be discovered.
//
//  5. A CALL THAT ANSWERS WITH THE WRONG TABLE. A policy naming the rational
//     route has to return that route's own region-A fits, and a call naming a
//     relaxed multiplier has to return the fits cut to that rung. The first is
//     compared with the routed per-argument entry and held to it at every cell
//     where the two routes' readings part by more than a four-lane sum can
//     explain - the cells that tell the tables apart; the second is held to the
//     degrees the multiplier's criterion certifies for the table its scheme
//     sums, which is not the same answer for both schemes: the Chebyshev table
//     has a droppable tail at the first rung and the monomial table has none,
//     so there the rung's values are the reference multiplier's bit for bit.
//
//  6. A FIGURE THAT IS NOT MET. Both opened calls are measured on the committed
//     reference grid at the figure the library publishes for them: the route at
//     the batch budget its fits hold, the rung at the multiplier times the
//     m = 1 contract of the table it reads. The worst cell is printed.
//
//  7. A LANE THAT FETCHES THE WRONG PARTITION'S PIECES. The narrow partition
//     cuts region A per order, so the axis cannot step one piece's
//     coefficients at a fixed stride and fetches each packed order its own
//     piece instead. Its values are therefore measured as a DIFFERENCE from the
//     per-order narrow lane - the worst cell and the count that differ at all,
//     printed - and held to that lane wherever the difference is wider than the
//     two mappings' own arithmetic. A second count says which table it read:
//     the shipped-table axis is the same body at the same scheme, so the cells
//     where the two part are the cells the partition argument reached.

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"
#include "boys_orders_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace {

constexpr int kNmax = boys::kMaxBoysOrder;
constexpr double kX0 = boys::detail::kX0;

// The loudest fp64 budget the library documents (the 5.5e-14 per-region
// budget of the shipped entries). The scheme rows promise far tighter for
// region A; this test asserts the shipped budget and prints the measured
// worst so a reader sees the margin.
constexpr double kRegionBudget = 5.5e-14;

// The scale of the difference between reading a fit four orders at a time and
// reading it once: measured at 2.3e-16 over the region-A grid, so this is that
// with room to spare. It sits far below the distance between the two routes'
// own region-A readings, so a lane that kept one route's table while carrying
// the other route's name cannot hide inside it.
constexpr double kArithmeticSlack = 1e-15;

// The two figures a region-A cell of the double single lane can be documented
// at: 1e-15 below the extended band, 3e-14 inside it. The orders axis carries
// the tighter of the two across the whole of region A, because that is the
// figure its own section states for the lane at m = 1.
constexpr double kSingleBarA = boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleSingle);
constexpr double kSingleBarBand = 3e-14;

// The bar a cell of the single lane is judged at, by the cell's own region.
double SingleBar(double x) {
    return x < boys::detail::kExtendedBX0 ? kSingleBarA : kSingleBarBand;
}

bool SameBits(double a, double b) noexcept {
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}

// A sweep of region A that reaches every part of it: the small-x end where the
// fits are widest, the piece boundary both sides of it, and the right edge at
// kX0. Deterministic, so the assertion count is a property of the file.
std::vector<double> RegionAGrid() {
    std::vector<double> grid;
    const std::size_t logSteps = 3000;

    for (std::size_t i = 0; i <= logSteps; ++i)
    {
        const double u = static_cast<double>(i) / static_cast<double>(logSteps);
        grid.push_back(std::exp(std::log(1e-8) + u * (std::log(kX0) - std::log(1e-8))));
    }

    const double boundary = boys::detail::kPieces[0].b;

    for (double d : {-1e-12, -1e-9, -1e-6, 0.0, 1e-6, 1e-9, 1e-12})
    {
        grid.push_back(boundary + d);
    }

    grid.push_back(kX0 - 1e-9);
    grid.push_back(std::nextafter(kX0, 0.0));
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());

    // The exp/log sweep's last point lands within an ULP of kX0 and can land
    // above it, where the across-arguments lane has no piece to blend and
    // returns zero: that lane's domain is [0, kX0), and this grid has to be
    // inside it for the comparison to mean anything.
    grid.erase(std::remove_if(grid.begin(), grid.end(), [](double x) { return !(x < kX0); }),
               grid.end());
    return grid;
}

// --- The committed reference grid -------------------------------------------
//
// The gate's 80-digit mpmath grid, three columns of the thirteen read: n, x
// and the double value. Measured against, never regenerated.
struct ReferenceCell {
    int n;
    double x;
    double value;
};

std::vector<ReferenceCell> LoadReference() {
    std::vector<ReferenceCell> cells;

#ifdef BoysTierReference
    std::ifstream file(BoysTierReference);

    if (!file)
    {
        return cells;
    }

    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        // The first three columns of the thirteen: n, x, value.
        const char* p = line.c_str();
        char* end = nullptr;
        const double order = std::strtod(p, &end);

        if (end == p || *end != ',')
        {
            continue;
        }

        p = end + 1;
        const double arg = std::strtod(p, &end);

        if (end == p || *end != ',')
        {
            continue;
        }

        p = end + 1;
        const double value = std::strtod(p, &end);

        if (end != p)
        {
            cells.push_back(ReferenceCell{static_cast<int>(order), arg, value});
        }
    }
#endif

    return cells;
}

// The region-A cells, which is the domain the lane is defined on.
std::vector<ReferenceCell> RegionAReference() {
    std::vector<ReferenceCell> cells;

    for (const ReferenceCell& cell : LoadReference())
    {
        if (cell.n >= 0 && cell.n <= kNmax && cell.x > 0.0 && cell.x < kX0)
        {
            cells.push_back(cell);
        }
    }

    return cells;
}

bool VectorTier() {
    return boys::BoysAvx2Available();
}

// The worst absolute error one scheme reaches on the reference grid, and the
// cell it reaches it at.
struct WorstError {
    double error = 0.0;
    int n = 0;
    double x = 0.0;
};

WorstError SchemeAgainstReference(boys::detail::OrdersScheme scheme,
                                  const std::vector<ReferenceCell>& cells) {
    WorstError worst;
    std::vector<double> out(static_cast<std::size_t>(kNmax) + 1);

    for (const ReferenceCell& cell : cells)
    {
        boys::detail::BoysAllOrdersSimd(scheme, kNmax, cell.x, out.data(), 1);
        const double error = std::abs(out[static_cast<std::size_t>(cell.n)] - cell.value);

        if (error > worst.error)
        {
            worst = WorstError{error, cell.n, cell.x};
        }
    }

    return worst;
}

// The worst absolute error one public policy reaches on the committed reference
// grid, and the cell it reaches it at. The measure the lane test above takes,
// through a policy rather than through the lane: what a caller naming this
// policy is handed.
template <double kMultiplier, class Policy>
WorstError PolicyAgainstReference(const std::vector<ReferenceCell>& cells) {
    WorstError worst;
    std::vector<double> out(static_cast<std::size_t>(kNmax) + 1);

    for (const ReferenceCell& cell : cells)
    {
        boys::BoysAllOrders<kMultiplier, Policy>(kNmax, cell.x, out.data());
        const double error = std::abs(out[static_cast<std::size_t>(cell.n)] - cell.value);

        if (error > worst.error)
        {
            worst = WorstError{error, cell.n, cell.x};
        }
    }

    return worst;
}

} // namespace

// The axis's own correctness claim: with the certified scheme the lane's value
// for an order is the across-arguments lane's value for that order, bit for
// bit. That lane computes it once per order over four copies of the argument,
// which is the same work the across-orders lane does in one pass.
TEST(BoysAcrossOrders, SplitClenshawIsTheAcrossArgumentsLaneBitForBit) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();
    std::vector<double> ours(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> lanes(4);
    std::vector<double> duplicate(4);
    std::size_t compared = 0;
    std::size_t differing = 0;

    for (double x : grid)
    {
        boys::detail::BoysAllOrdersSimd(
            boys::detail::OrdersScheme::kSplitClenshaw, kNmax, x, ours.data(), 1);

        for (int l = 0; l <= kNmax; ++l)
        {
            duplicate.assign(4, x);
            boys::detail::BoysRegionASimd<boys::kBoysFullAccuracyMultiplier>(
                l, duplicate.data(), lanes.data(), 4);
            ++compared;

            if (!SameBits(ours[static_cast<std::size_t>(l)], lanes[0]))
            {
                if (differing == 0)
                {
                    std::printf("  first differing cell: n = %d, x = %.17g, ours = %.17g, "
                                "lane = %.17g\n",
                                l,
                                x,
                                ours[static_cast<std::size_t>(l)],
                                lanes[0]);
                }

                ++differing;
            }
        }
    }

    std::printf("\n  across-orders split Clenshaw against the across-arguments lane: "
                "%zu of %zu order values bit-identical over %zu arguments\n",
                compared - differing,
                compared,
                grid.size());
    EXPECT_EQ(differing, 0u) << "the two bodies have parted";
}

// The two coefficient fetches are the same lane: the arithmetic is identical,
// so a difference between them would be a fetch that read the wrong bytes, and
// no tolerance is needed to see it.
TEST(BoysAcrossOrders, TheComposedFetchIsTheGatheredFetchBitForBit) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();
    std::vector<double> gathered(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> composed(static_cast<std::size_t>(kNmax) + 1);
    std::size_t compared = 0;
    std::size_t differing = 0;

    for (double x : grid)
    {
        for (boys::detail::OrdersScheme scheme : {boys::detail::OrdersScheme::kSplitClenshaw,
                                                  boys::detail::OrdersScheme::kDirectSum,
                                                  boys::detail::OrdersScheme::kHorner})
        {
            boys::detail::BoysAllOrdersSimd(scheme, kNmax, x, gathered.data(), 1);
            boys::detail::BoysAllOrdersSimdComposed(scheme, kNmax, x, composed.data(), 1);

            for (int l = 0; l <= kNmax; ++l)
            {
                ++compared;

                if (!SameBits(gathered[static_cast<std::size_t>(l)],
                              composed[static_cast<std::size_t>(l)]))
                {
                    ++differing;
                }
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "of " << compared << " values";
}

// The layout claim: the plane entry's order stride carries the contiguous
// write's bits.
TEST(BoysAcrossOrders, StridedWriteCarriesTheContiguousBits) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    constexpr std::size_t kStride = 7;
    const std::vector<double> grid = RegionAGrid();
    std::vector<double> contiguous(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> plane((static_cast<std::size_t>(kNmax) + 1) * kStride);
    std::size_t compared = 0;
    std::size_t differing = 0;

    for (double x : grid)
    {
        for (boys::detail::OrdersScheme scheme : {boys::detail::OrdersScheme::kSplitClenshaw,
                                                  boys::detail::OrdersScheme::kDirectSum,
                                                  boys::detail::OrdersScheme::kHorner})
        {
            boys::detail::BoysAllOrdersSimd(scheme, kNmax, x, contiguous.data(), 1);
            boys::detail::BoysAllOrdersSimd(scheme, kNmax, x, plane.data(), kStride);

            for (int l = 0; l <= kNmax; ++l)
            {
                ++compared;

                if (!SameBits(contiguous[static_cast<std::size_t>(l)],
                              plane[static_cast<std::size_t>(l) * kStride]))
                {
                    ++differing;
                }
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "of " << compared << " values";
}

// The accuracy claim, measured against the committed reference rather than
// against the library, over every region-A cell of it.
TEST(BoysAcrossOrders, EverySchemeMeetsTheRegionBudgetOnTheReferenceGrid) {
    const std::vector<ReferenceCell> cells = RegionAReference();

    if (cells.empty())
    {
        GTEST_SKIP() << "the reference grid is not present in this tree";
    }

    const struct {
        boys::detail::OrdersScheme scheme;
        const char* name;
    } schemes[] = {
        {boys::detail::OrdersScheme::kSplitClenshaw, "split clenshaw"},
        {boys::detail::OrdersScheme::kDirectSum, "direct sum"},
        {boys::detail::OrdersScheme::kHorner, "horner"},
    };

    for (const auto& entry : schemes)
    {
        const WorstError worst = SchemeAgainstReference(entry.scheme, cells);

        std::printf("  across-orders %-14s worst %.6e at n = %d, x = %.12g  (budget %.2e)\n",
                    entry.name,
                    worst.error,
                    worst.n,
                    worst.x,
                    kRegionBudget);
        EXPECT_LE(worst.error, kRegionBudget) << entry.name;
    }

    std::printf("  cells: %zu region-A cells of the committed reference grid\n", cells.size());
    // The zero is excluded: it is the closed form, not a fit, and a cell there
    // cannot discriminate between the schemes.
    EXPECT_EQ(cells.size(), 21285u) << "the reference grid's region-A cells moved";
}

// The route's difference from the shipped entry, stated rather than assumed:
// the shipped all-orders entry reaches most of its orders by a recursion from
// one fit, the lane reaches every order from its own fit, so the two agree to
// the recursion's rounding and not to the bit.
TEST(BoysAcrossOrders, AgreesWithTheShippedAllOrdersEntryWithinItsBudget) {
    const std::vector<double> grid = RegionAGrid();
    std::vector<double> ours(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> shipped(static_cast<std::size_t>(kNmax) + 1);
    double worst = 0.0;
    double worstRelative = 0.0;
    std::size_t differing = 0;
    std::size_t compared = 0;

    for (double x : grid)
    {
        boys::BoysAllOrders(kNmax, x, shipped.data());
        boys::detail::BoysAllOrdersSimd(
            boys::detail::OrdersScheme::kSplitClenshaw, kNmax, x, ours.data(), 1);

        for (int l = 0; l <= kNmax; ++l)
        {
            const double a = ours[static_cast<std::size_t>(l)];
            const double b = shipped[static_cast<std::size_t>(l)];
            ++compared;

            if (!SameBits(a, b))
            {
                ++differing;
            }

            worst = std::max(worst, std::abs(a - b));
            worstRelative = std::max(worstRelative, std::abs(a - b) / std::abs(b));
        }
    }

    std::printf("  across-orders against BoysAllOrders: worst absolute %.6e, "
                "worst relative %.6e, %zu of %zu values differing\n",
                worst,
                worstRelative,
                differing,
                compared);
    EXPECT_LE(worst, kRegionBudget);
}

// ---------------------------------------------------------------------------
// The axis on the public surface
// ---------------------------------------------------------------------------
// The lane is reached from outside through PackAxis::kOrders on the all-orders
// entry. What the tests below hold is that the surface and the lane are the
// same code: the entry's values are the packed lane's values, and the axis a
// policy names is the axis the report prints.

namespace {

// The policy the public axis is named with.
template <boys::EvalScheme kScheme>
using OrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

// The same axis with the rational route named: the first of the two calls the
// axis answers beyond its shipped reading.
template <boys::EvalScheme kScheme>
using RationalOrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kRationalMinimax, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

// The per-order lane the entry falls back to outside the packed interval, at
// the same rung: the certified scalar single entry, which is the one lane the
// axis cannot be formed on.
template <boys::EvalScheme kScheme>
using PerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments>;

// The narrow partition on the axis. Its region-A pieces are cut per order, so
// the packed lane has no shared stride to step and fetches each of the four
// orders it packs its own piece and coefficients. The shipped-table axis is
// the same body at the same scheme, which is what makes the two comparable
// cell for cell: the partition is the only thing that differs between them.
template <boys::EvalScheme kScheme>
using NarrowOrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders, boys::FitGranularity::kNarrow>;

// The same partition read one order at a time by the certified scalar single
// entry: the reading the packed lane's gathered fetch has to reproduce, and
// the lane a call would fall back to if the axis had no narrow kernel.
template <boys::EvalScheme kScheme>
using NarrowPerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, boys::FitGranularity::kNarrow>;

// The first relaxed rung, named as a caller names it: a tier, through the
// library's own mapping from the tier to its multiplier.
constexpr double kRungMultiplier = boys::AccuracyMultiplier(boys::AccuracyTier::kRelaxed64);

} // namespace

// The default policy still names the shipped axis, so a call site that names
// no axis compiles the entry it always did.
static_assert(boys::EvalPolicy<>{}.kPack == boys::PackAxis::kArguments,
              "the default axis moved: a call site that names no axis must compile the "
              "shipped path");
static_assert(OrdersPolicy<boys::EvalScheme::kSplitClenshaw>::kPack == boys::PackAxis::kOrders,
              "the orders policy does not name the orders axis");

TEST(BoysAcrossOrders, ThePublicAxisIsThePackedLane) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();
    std::vector<double> viaEntry(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> viaLane(static_cast<std::size_t>(kNmax) + 1);
    std::size_t differing = 0;

    for (double x : grid)
    {
        boys::BoysAllOrders<1.0, OrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
            kNmax, x, viaEntry.data());
        boys::detail::BoysAllOrdersSimdComposed(
            boys::detail::OrdersScheme::kSplitClenshaw, kNmax, x, viaLane.data(), 1);

        for (int l = 0; l <= kNmax; ++l)
        {
            if (!SameBits(viaEntry[static_cast<std::size_t>(l)],
                          viaLane[static_cast<std::size_t>(l)]))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the entry and the lane have parted";
}

TEST(BoysAcrossOrders, ThePublicAxisIsDefinedPastItsOwnDomain) {
    // Past the lane's interval the entry runs the certified scalar single lane
    // one order at a time, so it is defined for every argument the library
    // accepts and its values are that lane's, bit for bit. That is the whole
    // claim the fallback makes, and it is asserted exactly rather than against
    // a tolerance: at these arguments the orders above 28 are below the
    // region-C bound, where the returned value is the recurrence's rounding and
    // carries no structure a monotonicity check could test.
    const double arguments[] = {0.0,
                                boys::detail::kX0,
                                boys::detail::kX0 + 1e-9,
                                20.0,
                                boys::detail::kX1,
                                31.0,
                                200.0};
    std::vector<double> out(static_cast<std::size_t>(kNmax) + 1);
    std::size_t differing = 0;

    for (double x : arguments)
    {
        boys::BoysAllOrders<1.0, OrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
            kNmax, x, out.data());

        for (int l = 0; l <= kNmax; ++l)
        {
            const double single = boys::BoysSingle(l, x);

            if (!SameBits(out[static_cast<std::size_t>(l)], single))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the entry's fallback is not the certified scalar single lane";
}

// The rational route on the axis: a policy names the route, the entry answers,
// and the values are that route's own region-A fits. The routed per-argument
// entry is the second reading that says which values those are - it reads the
// same stored pairs by the route's own body - and the two are compared over the
// whole grid. They differ by the arithmetic of a four-lane group sum against a
// scalar fit evaluation and by nothing else, so where the two routes' readings
// part by more than that arithmetic can explain, the axis has to be the routed
// one: that is the cell that says which table it read.
TEST(BoysAcrossOrders, TheRationalRouteOnTheAxisIsTheRoutesOwnReading) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();

    const auto compareWithTheRoutedEntry = [&]<boys::EvalScheme kScheme>(const char* name) {
        std::vector<double> ours(static_cast<std::size_t>(kNmax) + 1);
        std::vector<double> routed(static_cast<std::size_t>(kNmax) + 1);
        std::vector<double> shipped(static_cast<std::size_t>(kNmax) + 1);
        std::size_t compared = 0;
        std::size_t differing = 0;
        std::size_t discriminating = 0;
        std::size_t violations = 0;
        double worst = 0.0;

        for (double x : grid)
        {
            boys::BoysAllOrders<1.0, RationalOrdersPolicy<kScheme>>(kNmax, x, ours.data());
            boys::BoysAllOrdersWithRoute(
                boys::FitRoute::kRationalMinimax, kScheme, kNmax, x, routed.data());
            boys::BoysAllOrders<1.0, OrdersPolicy<kScheme>>(kNmax, x, shipped.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double a = ours[static_cast<std::size_t>(l)];
                const double b = routed[static_cast<std::size_t>(l)];
                const double s = shipped[static_cast<std::size_t>(l)];
                const double fromRouted = std::abs(a - b);
                ++compared;
                worst = std::max(worst, fromRouted);

                if (!SameBits(a, b))
                {
                    ++differing;
                }

                if (std::abs(b - s) > kArithmeticSlack)
                {
                    ++discriminating;

                    if (fromRouted > kArithmeticSlack)
                    {
                        ++violations;
                    }
                }
            }
        }

        std::printf("  rational route on the axis, %-14s: worst %.6e from the routed entry over "
                    "%zu order values, %zu of them differing; %zu cells tell the two routes apart "
                    "and %zu of those kept the shipped table\n",
                    name,
                    worst,
                    compared,
                    differing,
                    discriminating,
                    violations);
        EXPECT_EQ(violations, 0u) << name << ": the axis is not the route's own reading";
        EXPECT_GT(discriminating, 0u) << name << ": no cell of the grid tells the routes apart";
    };

    compareWithTheRoutedEntry.template operator()<boys::EvalScheme::kSplitClenshaw>(
        "split clenshaw");
    compareWithTheRoutedEntry.template operator()<boys::EvalScheme::kHorner>("horner");
}

// The narrow partition on the axis. The shipped lane's fetch steps from one
// order's coefficients to the next at a fixed stride, which the shipped
// region-A table has because every order's pieces share their intervals and
// degrees. The narrow partition's pieces are cut per order, so there is no such
// stride to step: the lane fetches each of the four orders it packs its own
// piece and coefficients, and one group is four different pieces evaluated
// together rather than one piece read four orders deep.
//
// What that costs is measured here as a DIFFERENCE IN VALUES from the
// per-order narrow lane, not asserted to agree with it. The two are two
// mappings of one argument into one piece - the scalar entry's
// `2 (x - a) / (b - a) - 1` against the packed lane's
// `fma(x - a, 2 / (b - a), -1)`, which are the same number up to the rounding
// of those operations - so the two can part by that arithmetic. The worst cell,
// the count of order values that differ at all and the count outside the slack
// are printed, so the cost of the gathered fetch is a number in the log rather
// than a promise.
//
// Which table the lane read is a second measurement, and the shipped-table axis
// is the lane to measure it against: the same body at the same scheme, so its
// only difference from this call is the partition the policy names. The count of
// order values whose bits differ from that lane's has to be nonzero for the row
// to mean anything - a lane that ignored the partition argument would return
// that lane's bits exactly.
//
// Where the two narrow lanes part by more than the slack, this test does not
// decide which of them is right: the next one does, against the committed
// reference. Here the parting is a measurement, and the two assertions are the
// ones this grid can carry - that the axis is not the per-order lane's bits and
// not the shipped axis's.
TEST(BoysAcrossOrders, TheNarrowPartitionOnTheAxisIsTheNarrowLanesOwnReading) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();

    const auto compareWithThePerOrderLane = [&]<boys::EvalScheme kScheme>(const char* name) {
        std::vector<double> packed(static_cast<std::size_t>(kNmax) + 1);
        std::vector<double> perOrder(static_cast<std::size_t>(kNmax) + 1);
        std::vector<double> shippedLane(static_cast<std::size_t>(kNmax) + 1);
        std::size_t compared = 0;
        std::size_t differing = 0;
        std::size_t discriminating = 0;
        std::size_t parting = 0;
        double worst = 0.0;
        double worstX = 0.0;
        int worstOrder = 0;

        for (double x : grid)
        {
            boys::BoysAllOrders<1.0, NarrowOrdersPolicy<kScheme>>(kNmax, x, packed.data());
            boys::BoysAllOrders<1.0, OrdersPolicy<kScheme>>(kNmax, x, shippedLane.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                perOrder[static_cast<std::size_t>(l)] =
                    boys::BoysSingle<1.0, NarrowPerOrderPolicy<kScheme>>(l, x);
            }

            for (int l = 0; l <= kNmax; ++l)
            {
                const std::size_t at = static_cast<std::size_t>(l);
                const double fromPerOrder = std::abs(packed[at] - perOrder[at]);
                ++compared;

                if (fromPerOrder > worst)
                {
                    worst = fromPerOrder;
                    worstX = x;
                    worstOrder = l;
                }

                if (!SameBits(packed[at], perOrder[at]))
                {
                    ++differing;
                }

                if (!SameBits(packed[at], shippedLane[at]))
                {
                    ++discriminating;
                }

                if (fromPerOrder > kArithmeticSlack)
                {
                    ++parting;
                }
            }
        }

        std::printf("  narrow partition on the axis, %-14s: worst %.6e from the per-order narrow "
                    "lane at n = %d, x = %.12g, over %zu order values; %zu differ at all, %zu read "
                    "a table the shipped axis did not, %zu outside the arithmetic slack\n",
                    name,
                    worst,
                    worstOrder,
                    worstX,
                    compared,
                    differing,
                    discriminating,
                    parting);
        EXPECT_GT(differing, 0u)
            << name << ": the axis returned the per-order lane's bits exactly, which is the "
                       "scalar lane's arithmetic and not a packed group's";
        EXPECT_GT(discriminating, 0u)
            << name << ": the axis returned the shipped-table axis's bits exactly, so the "
                       "partition it named is not the one it read";
    };

    compareWithThePerOrderLane.template operator()<boys::EvalScheme::kSplitClenshaw>(
        "split clenshaw");
    compareWithThePerOrderLane.template operator()<boys::EvalScheme::kHorner>("horner");
}

// The axis and the per-order narrow lane are two readings of one fit, and where
// they part by more than the slack this test asks a third party which reading is
// right: the committed 80-digit reference. It is a difference in values resolved
// against the function rather than an agreement asserted between the two lanes.
//
// Each cell is judged at the figure the lane that produced it documents. The
// axis carries the orders lane's own 1e-15 over the whole of region A. The
// per-order narrow lane is the single lane, so its figure is 1e-15 below the
// extended band and 3e-14 inside it - the two readings are held to different
// budgets because they are different lanes, and the count of cells where the
// per-order lane is outside its own budget is printed rather than assumed.
TEST(BoysAcrossOrders, WhereTheNarrowAxisPartsFromThePerOrderLaneTheReferenceIsOnTheAxis) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<ReferenceCell> cells = RegionAReference();

    if (cells.empty())
    {
        GTEST_SKIP() << "the reference grid is not present in this tree";
    }

    const auto referee = [&]<boys::EvalScheme kScheme>(const char* name) {
        std::vector<double> packed(static_cast<std::size_t>(kNmax) + 1);
        std::size_t part = 0;
        std::size_t partInRegionA = 0;
        std::size_t closer = 0;
        std::size_t inBar = 0;
        std::size_t perOrderOver = 0;
        double axisWorst = 0.0;
        double perOrderWorst = 0.0;
        double worstPart = 0.0;
        double worstPartAxis = 0.0;
        double worstPartPerOrder = 0.0;
        double worstPartX = 0.0;
        int worstPartOrder = 0;

        for (const ReferenceCell& cell : cells)
        {
            boys::BoysAllOrders<1.0, NarrowOrdersPolicy<kScheme>>(kNmax, cell.x, packed.data());
            const double axis = packed[static_cast<std::size_t>(cell.n)];
            const double perOrder =
                boys::BoysSingle<1.0, NarrowPerOrderPolicy<kScheme>>(cell.n, cell.x);
            const double axisError = std::abs(axis - cell.value);
            const double perOrderError = std::abs(perOrder - cell.value);
            const double partBy = std::abs(axis - perOrder);

            axisWorst = std::max(axisWorst, axisError);
            perOrderWorst = std::max(perOrderWorst, perOrderError);

            if (partBy > kArithmeticSlack)
            {
                ++part;
                partInRegionA += cell.x < boys::detail::kExtendedBX0;
                closer += axisError < perOrderError;
                inBar += axisError <= kSingleBarA;
                perOrderOver += perOrderError > SingleBar(cell.x);

                if (partBy > worstPart)
                {
                    worstPart = partBy;
                    worstPartAxis = axisError;
                    worstPartPerOrder = perOrderError;
                    worstPartX = cell.x;
                    worstPartOrder = cell.n;
                }
            }
        }

        std::printf("  narrow axis against the per-order narrow lane, %-14s: %zu of %zu cells "
                    "part by more than %.0e (%zu of those in region A proper); the reference is "
                    "on the axis's side at %zu, the axis is inside %.0e at %zu, the per-order "
                    "lane is outside its own figure at %zu\n",
                    name,
                    part,
                    cells.size(),
                    kArithmeticSlack,
                    partInRegionA,
                    closer,
                    kSingleBarA,
                    inBar,
                    perOrderOver);
        std::printf("    worst of them n = %d, x = %.12g: axis %.6e, per-order lane %.6e from the "
                    "reference\n",
                    worstPartOrder,
                    worstPartX,
                    worstPartAxis,
                    worstPartPerOrder);
        std::printf("    worst over the whole grid: axis %.6e (figure %.0e), per-order lane %.6e\n",
                    axisWorst,
                    kSingleBarA,
                    perOrderWorst);

        EXPECT_GT(part, 0u) << name << ": the two readings never part, so this grid cannot say "
                                        "which of them the reference agrees with";
        EXPECT_EQ(closer, part)
            << name << ": at a cell where the axis parts from the per-order lane, the per-order "
                       "lane is the reading the reference agrees with";
        EXPECT_EQ(inBar, part)
            << name << ": the axis left its own region-A figure at a cell where it parts from "
                       "the per-order lane";
        EXPECT_LE(axisWorst, kSingleBarA)
            << name << ": the axis is outside the region-A figure its own section states";
    };

    referee.template operator()<boys::EvalScheme::kSplitClenshaw>("split clenshaw");
    referee.template operator()<boys::EvalScheme::kHorner>("horner");
}

// The rung on the axis: a call naming a relaxed multiplier reads the degrees
// that multiplier's criterion certifies for the table its scheme sums, and
// outside the packed interval it is the rung's own per-order lane. What the
// criterion certifies is not the same answer for the two schemes - the
// Chebyshev table has a droppable tail at this rung, the monomial table does
// not - so each scheme is held to the reading its own table supports: the
// values move where a degree was cut, and are the reference multiplier's bit
// for bit where none was.
//
// One interval is outside that rule, and it is the extended band. There the
// reference rung answers from the band's seed and its upward recursion while a
// relaxed one keeps the region-A fit, so the two are two readings of two
// different things and part by their own size - by design, and stated in the
// band's own comment in boys_impl.hpp as the m = 1 lane's alone. So the band's
// cells are counted apart from the rest, the identity is asserted over the
// arguments where both readings really are the fit, and the band joins it only
// where the orders axis is the packed lane and both calls are that one body.
//
// The fallback is that per-order lane's body, compiled in the packed unit
// rather than in this file, and region B's recurrence ends each step in a
// multiply and a subtract that a compiler may round as one operation or as two.
// Where the two units choose differently, the readings part by the last bits of
// that recurrence - so the fallback's identity is held bit for bit where the
// arithmetic fixes it and to a stated slack where the build decides it, with the
// count and the worst printed either way.
TEST(BoysAcrossOrders, TheRelaxedRungOnTheAxisIsTheRungsOwnReading) {
    const std::vector<double> grid = RegionAGrid();
    std::vector<double> rung(static_cast<std::size_t>(kNmax) + 1);
    std::vector<double> reference(static_cast<std::size_t>(kNmax) + 1);

    const auto countMoved = [&]<boys::EvalScheme kScheme>(const char* name) {
        constexpr auto kCut = boys::detail::RegionADegrees<kRungMultiplier,
                                                           boys::detail::BoysRole::kDoubleSingle,
                                                           boys::detail::SchemeTailBasis<kScheme>()>();
        std::size_t cutPieces = 0;

        for (std::size_t p = 0; p < kCut.size(); ++p)
        {
            if (kCut[p] < boys::detail::kPieces[p].deg)
            {
                ++cutPieces;
            }
        }

        std::size_t differing = 0;
        std::size_t differingInBand = 0;
        double worst = 0.0;

        for (double x : grid)
        {
            boys::BoysAllOrders<kRungMultiplier, OrdersPolicy<kScheme>>(kNmax, x, rung.data());
            boys::BoysAllOrders<1.0, OrdersPolicy<kScheme>>(kNmax, x, reference.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double a = rung[static_cast<std::size_t>(l)];
                const double b = reference[static_cast<std::size_t>(l)];

                if (!SameBits(a, b))
                {
                    // The extended band is the one interval where the two
                    // multipliers are two readings rather than one table read
                    // twice: the reference rung answers [kExtendedBX0, kX0) from
                    // the band's seed and its upward recursion, and the relaxed
                    // branch keeps the region-A fit there. The two are counted
                    // apart so the identity below is claimed only where both
                    // readings really are the fit.
                    if (x >= boys::detail::kExtendedBX0)
                    {
                        ++differingInBand;
                    } else
                    {
                        ++differing;
                    }

                    worst = std::max(worst, std::abs(a - b));
                }
            }
        }

        std::printf("  rung m = %g on the axis, %-14s: the criterion cuts %zu of %zu piece "
                    "degrees, %zu order values moved outside the band, %zu inside it, worst %e\n",
                    kRungMultiplier,
                    name,
                    cutPieces,
                    kCut.size(),
                    differing,
                    differingInBand,
                    worst);

        if (cutPieces == 0)
        {
            EXPECT_EQ(differing, 0u) << name << ": the table is certified whole at this rung, so "
                                     << "outside the band the values have to be the reference "
                                     << "rung's";

            if (VectorTier())
            {
                EXPECT_EQ(differingInBand, 0u)
                    << name << ": on a host whose orders axis is the packed lane both readings "
                    << "are that lane's body, so no cut means the band's values are the "
                    << "reference rung's too";
            }
        } else
        {
            EXPECT_GT(differing, 0u) << name << ": the multiplier did not reach the fits";
        }
    };

    const auto compareFallback = [&]<boys::EvalScheme kScheme>(const char* name) {
        // Past the lane's interval the entry runs the certified scalar single
        // lane at the same multiplier, one order at a time, so a rung is defined
        // for every argument the library accepts and its values there are the
        // rung's per-order lane's. Past kX1 that recurrence reads no fit at all
        // and only multiplies and divides, so no build can round one of its steps
        // two ways and the two readings have to agree bit for bit. Inside region
        // B the step is a multiply and a subtract, which a compiler may fuse into
        // one rounding, and the packed unit and this file make that choice
        // separately - the packed unit's flags only make it likelier to differ.
        // There the same comparison is bounded instead, and the count and the
        // worst are printed rather than assumed away.
        constexpr double kFallbackArithmeticSlack = 1e-14;
        const double pastRegionB[] = {boys::detail::kX1, 31.0, 200.0};
        const double insideRegionB[] = {boys::detail::kX0, boys::detail::kX0 + 1e-9, 20.0};

        std::size_t compared = 0;
        std::size_t differing = 0;

        for (double x : pastRegionB)
        {
            boys::BoysAllOrders<kRungMultiplier, OrdersPolicy<kScheme>>(kNmax, x, rung.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double single =
                    boys::BoysSingle<kRungMultiplier, PerOrderPolicy<kScheme>>(l, x);
                ++compared;

                if (!SameBits(rung[static_cast<std::size_t>(l)], single))
                {
                    ++differing;
                }
            }
        }

        std::printf("  rung m = %g fallback past kX1, %-14s: %zu of %zu order values bit-identical "
                    "to the rung's per-order lane\n",
                    kRungMultiplier,
                    name,
                    compared - differing,
                    compared);

        EXPECT_EQ(differing, 0u) << name << ": past kX1 this recurrence multiplies and divides and "
                                 << "never adds, so a build cannot round it two ways and the "
                                 << "fallback has to be the rung's per-order lane bit for bit";

        // Inside region B the slack stands in for the rounding the two units may
        // disagree about, measured at 1.6e-16 over both schemes here and set an
        // order above that. What tells a wrong reading from a licence is the
        // moved count, not the slack: both its readings come from one unit, so no
        // rounding stands between them, and the rung's own region-B table parts
        // from the reference multiplier's by 1.5e-12 at kX0 and 1.4e-13 at 20.
        constexpr auto kCutB =
            boys::detail::RegionBDegrees<kRungMultiplier,
                                         boys::detail::BoysRole::kDoubleSingle,
                                         boys::detail::SchemeTailBasis<kScheme>()>();
        std::size_t cutOrders = 0;

        for (int order = 0; order <= kNmax; ++order)
        {
            if (kCutB[static_cast<std::size_t>(order)] < boys::detail::kBDeg)
            {
                ++cutOrders;
            }
        }

        std::size_t insideCompared = 0;
        std::size_t insideDiffering = 0;
        std::size_t moved = 0;
        double worst = 0.0;

        for (double x : insideRegionB)
        {
            boys::BoysAllOrders<kRungMultiplier, OrdersPolicy<kScheme>>(kNmax, x, rung.data());
            boys::BoysAllOrders<1.0, OrdersPolicy<kScheme>>(kNmax, x, reference.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double atRung = rung[static_cast<std::size_t>(l)];
                const double single =
                    boys::BoysSingle<kRungMultiplier, PerOrderPolicy<kScheme>>(l, x);
                ++insideCompared;

                if (!SameBits(atRung, reference[static_cast<std::size_t>(l)]))
                {
                    ++moved;
                }

                if (!SameBits(atRung, single))
                {
                    ++insideDiffering;
                    worst = std::max(worst, std::abs(atRung - single));
                }
            }
        }

        std::printf("  rung m = %g fallback inside region B, %-14s: %zu of %zu order values "
                    "bit-identical to the rung's per-order lane, worst %e; the rung cut %zu of %d "
                    "seed degrees and moved %zu of %zu values against the reference rung\n",
                    kRungMultiplier,
                    name,
                    insideCompared - insideDiffering,
                    insideCompared,
                    worst,
                    cutOrders,
                    kNmax + 1,
                    moved,
                    insideCompared);

        EXPECT_LE(worst, kFallbackArithmeticSlack)
            << name << ": the fallback inside region B is further from the rung's per-order lane "
            << "than the rounding two compilations of it may differ by";

        if (cutOrders == 0)
        {
            EXPECT_EQ(moved, 0u) << name << ": the region-B seed table is certified whole at this "
                                 << "rung, so the values have to be the reference rung's";
        } else
        {
            EXPECT_GT(moved, 0u) << name << ": the multiplier did not reach the region-B seed";
        }
    };

    countMoved.template operator()<boys::EvalScheme::kSplitClenshaw>("split clenshaw");
    countMoved.template operator()<boys::EvalScheme::kHorner>("horner");
    compareFallback.template operator()<boys::EvalScheme::kSplitClenshaw>("split clenshaw");
    compareFallback.template operator()<boys::EvalScheme::kHorner>("horner");
}

// The two opened calls' figures, measured where this library measures every
// other row: against the committed 80-digit reference grid, over its region-A
// cells, at the figure the library states for the call. The route's fits hold
// the batch budget in region A; the rung's truncated fits are cut against the
// multiplier times the m = 1 contract of the table they are read from, so the
// rung's figure is that product. Both figures are read off the library's own
// tables rather than repeated here.
TEST(BoysAcrossOrders, TheOpenedCallsMeetTheirFiguresOnTheReferenceGrid) {
    const std::vector<ReferenceCell> cells = RegionAReference();

    if (cells.empty())
    {
        GTEST_SKIP() << "the reference grid is not present in this tree";
    }

    constexpr double kRouteBound =
        boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleBatch);
    constexpr double kRungBound = kRungMultiplier *
                                  boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleSingle);

    // The narrow partition's pieces are read for their own value one order at a
    // time, so the bar that covers them is the single lane's per-order one and
    // not the batch lane's, and it is the figure the packing book judges the
    // narrow rows at as well.
    constexpr double kPerOrderBound = kSingleBarA;

    const auto measure = [&]<double kMultiplier, class Policy>(const char* label, double bound) {
        const WorstError worst = PolicyAgainstReference<kMultiplier, Policy>(cells);

        std::printf("  %-34s worst %.6e at n = %d, x = %.12g  (figure %.2e)\n",
                    label,
                    worst.error,
                    worst.n,
                    worst.x,
                    bound);
        EXPECT_LE(worst.error, bound) << label;
    };

    measure.template operator()<1.0, RationalOrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
        "rational route, split clenshaw", kRouteBound);
    measure.template operator()<1.0, RationalOrdersPolicy<boys::EvalScheme::kHorner>>(
        "rational route, horner", kRouteBound);
    measure.template operator()<kRungMultiplier, OrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
        "rung m=64 shipped route, split clenshaw", kRungBound);
    measure.template operator()<kRungMultiplier, OrdersPolicy<boys::EvalScheme::kHorner>>(
        "rung m=64 shipped route, horner", kRungBound);
    measure.template operator()<1.0, NarrowOrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
        "narrow partition on the axis, split clenshaw", kPerOrderBound);
    measure.template operator()<1.0, NarrowOrdersPolicy<boys::EvalScheme::kHorner>>(
        "narrow partition on the axis, horner", kPerOrderBound);

    std::printf("  cells: %zu region-A cells of the committed reference grid\n", cells.size());
}

TEST(BoysAcrossOrders, TheAxisReportNamesBothMembers) {
    const std::span<const boys::PackAxisInfo> axes = boys::BoysPackAxes();

    ASSERT_EQ(axes.size(), 2u);

    for (std::size_t i = 0; i < axes.size(); ++i)
    {
        const boys::PackAxisInfo& row = axes[i];

        EXPECT_EQ(static_cast<std::size_t>(row.axis), i) << "the rows are not in enumerator order";
        EXPECT_STREQ(row.name, boys::PackAxisName(row.axis));
        EXPECT_EQ(row.width, 4);
        EXPECT_DOUBLE_EQ(row.lo, 0.0);
        EXPECT_DOUBLE_EQ(row.hi, boys::detail::kX0);
        EXPECT_GT(row.bound, 0.0);
    }
}
