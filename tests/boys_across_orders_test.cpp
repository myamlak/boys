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

// The rung on the axis: a call naming a relaxed multiplier reads the degrees
// that multiplier's criterion certifies for the table its scheme sums, and
// outside the packed interval it is the rung's own per-order lane. What the
// criterion certifies is not the same answer for the two schemes - the
// Chebyshev table has a droppable tail at this rung, the monomial table does
// not - so each scheme is held to the reading its own table supports: the
// values move where a degree was cut, and are the reference multiplier's bit
// for bit where none was.
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
                    ++differing;
                    worst = std::max(worst, std::abs(a - b));
                }
            }
        }

        std::printf("  rung m = %g on the axis, %-14s: the criterion cuts %zu of %zu piece "
                    "degrees, %zu order values moved against the reference rung, worst %e\n",
                    kRungMultiplier,
                    name,
                    cutPieces,
                    kCut.size(),
                    differing,
                    worst);

        if (cutPieces == 0)
        {
            EXPECT_EQ(differing, 0u) << name << ": the table is certified whole at this rung, so "
                                     << "the values have to be the reference rung's";
        } else
        {
            EXPECT_GT(differing, 0u) << name << ": the multiplier did not reach the fits";
        }
    };

    const auto compareFallback = [&]<boys::EvalScheme kScheme>(const char* name) {
        // Past the lane's interval the entry runs the certified scalar single
        // lane at the same multiplier, one order at a time, so a rung is defined
        // for every argument the library accepts and its values there are the
        // rung's per-order lane's, bit for bit.
        const double arguments[] = {boys::detail::kX0,
                                    boys::detail::kX0 + 1e-9,
                                    20.0,
                                    boys::detail::kX1,
                                    31.0,
                                    200.0};
        std::size_t differing = 0;

        for (double x : arguments)
        {
            boys::BoysAllOrders<kRungMultiplier, OrdersPolicy<kScheme>>(kNmax, x, rung.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double single =
                    boys::BoysSingle<kRungMultiplier, PerOrderPolicy<kScheme>>(l, x);

                if (!SameBits(rung[static_cast<std::size_t>(l)], single))
                {
                    ++differing;
                }
            }
        }

        EXPECT_EQ(differing, 0u) << name << ": the rung's fallback is not the certified scalar "
                                 << "single lane at the rung";
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
