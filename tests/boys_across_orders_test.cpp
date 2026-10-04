// The across-orders packed lane's contract tests: BoysAllOrdersSimd, the entry
// that vectorises four ORDERS of one argument where the across-arguments lane
// vectorises four arguments of one order.
//
// What can go wrong with a lane that re-sums a stored fit in a different order,
// and what each test is for:
//
//  1. A TRANSCRIPTION SLIP. The split Clenshaw body is a second copy of the one
//     boys_impl.hpp holds, and a reordered step, a fused operation turned into
//     two, or a coefficient read one index out would all still return a plausible
//     number: the lane's answer for order l is asserted BIT-IDENTICAL to the
//     across-arguments lane's - that lane runs the library's own body over four
//     copies of the argument - with no tolerance, a tolerance being the slack a
//     slip would hide in.
//
//  2. A BOUND NOT MET. Each scheme is measured against the committed reference
//     grid, not against the library, over every region-A cell of it.
//
//  3. A LAYOUT THAT SILENTLY TRANSPOSES. The entry writes at out[l * stride];
//     the strided write must carry the contiguous write's bits.
//
//  4. A ROUTE THAT DISAGREES WITH THE SHIPPED ONE. The lane reaches no order by
//     recursion where BoysAllOrders reaches most of them that way, so the two
//     differ by their routes' arithmetic - stated here, not left to be discovered.
//
//  5. A CALL THAT ANSWERS WITH THE WRONG TABLE. A policy naming the rational
//     route must return that route's own region-A fits, and a policy naming the
//     narrow or the uniform partition must read that partition's own pieces
//     rather than another's under the name it was given.
//
//  6. A FIGURE THAT IS NOT MET. The opened call is measured on the committed
//     reference grid at the figure the library publishes for it: the route at
//     the batch budget its fits hold.
//
//  7. A LANE THAT FETCHES THE WRONG PARTITION'S PIECES. The narrow partition cuts
//     region A per order, so the axis fetches each packed order its own piece
//     instead of stepping one piece's coefficients at a fixed stride. Its values
//     are measured as a DIFFERENCE from the per-order narrow lane and held to it
//     wherever the difference is wider than the two mappings' own arithmetic.

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

// The loudest fp64 budget the library documents: the 5.5e-14 per-region budget
// of the shipped entries, asserted here although the scheme rows promise tighter.
constexpr double kRegionBudget = 5.5e-14;

// The scale of the difference between reading a fit four orders at a time and
// reading it once, measured at 2.3e-16 over the region-A grid - far below the
// distance between the two routes' readings, so a lane carrying one route's table
// under the other's name cannot hide inside it.
constexpr double kArithmeticSlack = 1e-15;

// The two figures a region-A cell of the double single lane is documented at:
// 1e-15 below the extended band, 3e-14 inside it. The band is answered from its
// seed and its upward recursion, so each cell is judged at its own region's figure.
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

    // The exp/log sweep's last point can land an ULP above kX0, where the across-arguments
    // lane has no piece to blend and returns zero; its domain is [0, kX0).
    grid.erase(std::remove_if(grid.begin(), grid.end(), [](double x) { return !(x < kX0); }),
               grid.end());
    return grid;
}

// --- The committed reference grid -------------------------------------------
//
// The gate's 80-digit mpmath grid, three of its thirteen columns read: n, x and
// the double value. Measured against, never regenerated.
struct ReferenceCell {
    int n;
    double x;
    double value;
};

std::vector<ReferenceCell> LoadReference() {
    std::vector<ReferenceCell> cells;

    const std::string path = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
    std::ifstream file(path);

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

// The worst absolute error one scheme reaches on the reference grid, and where.
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

// The cell one public policy sits highest above its own figure at, the error it
// delivers there, and that cell's documented figure: what a caller naming this
// policy is handed. The figure travels with the cell because a region-dependent
// bar - 1e-15 below the extended band, 3e-14 inside it - puts the worst
// shortfall at a different cell from the worst error.
struct BarShortfall {
    double error = 0.0;
    double figure = 0.0;
    int n = 0;
    double x = 0.0;
};

template <class Policy, class Bar>
BarShortfall PolicyAgainstReference(const std::vector<ReferenceCell>& cells, Bar bar) {
    BarShortfall worst;
    double worstFraction = -1.0;
    std::vector<double> out(static_cast<std::size_t>(kNmax) + 1);

    for (const ReferenceCell& cell : cells)
    {
        boys::BoysAllOrders<Policy>(kNmax, cell.x, out.data());
        const double error = std::abs(out[static_cast<std::size_t>(cell.n)] - cell.value);
        const double figure = bar(cell.x);

        if (error / figure > worstFraction)
        {
            worstFraction = error / figure;
            worst = BarShortfall{error, figure, cell.n, cell.x};
        }
    }

    return worst;
}

} // namespace

// The axis's own correctness claim: with the certified scheme the lane's value
// for an order is the across-arguments lane's for that order.
//
// At the fused route the two bodies run one arithmetic and the claim is bit for
// bit. At the separate route they are two arithmetics: both lanes carry the route
// the build selected, and they sum the same stored fit in different orders. The
// claim there is how far the two part, which is a rounding's — each body is
// inside the region's budget, so the two are within twice it — and the count and
// the parting print either way.
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
    double worst = 0.0;

    for (double x : grid)
    {
        boys::detail::BoysAllOrdersSimd(
            boys::detail::OrdersScheme::kSplitClenshaw, kNmax, x, ours.data(), 1);

        for (int l = 0; l <= kNmax; ++l)
        {
            duplicate.assign(4, x);
            boys::detail::BoysRegionASimd(l, duplicate.data(), lanes.data(), 4);
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

                const double gap = std::abs(ours[static_cast<std::size_t>(l)] - lanes[0]);

                worst = gap > worst ? gap : worst;
                ++differing;
            }
        }
    }

    std::printf("\n  across-orders split Clenshaw against the across-arguments lane: "
                "%zu of %zu order values bit-identical over %zu arguments, worst parting "
                "%.3e\n",
                compared - differing,
                compared,
                grid.size(),
                worst);

#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
    EXPECT_LE(worst, 2.0 * boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleSingle))
        << "the two bodies have parted further than two roundings of one scheme: worst "
        << worst << " over " << compared << " values";
#else
    EXPECT_EQ(differing, 0u) << "the two bodies have parted";
#endif
}

// The two coefficient fetches are the same lane: a difference between them is a
// fetch that read the wrong bytes, which needs no tolerance to see.
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

// The layout claim: the plane entry's order stride carries the contiguous bits.
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

// The accuracy claim, measured against the committed reference, not the library.
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
    // The zero is excluded: it is the closed form, not a fit, so it cannot discriminate.
    EXPECT_EQ(cells.size(), 21285u) << "the reference grid's region-A cells moved";
}

// The lane reaches every order from its own fit where the shipped entry reaches
// most of them by a recursion from one, so the two agree to the recursion's
// rounding and not to the bit. Stated, not assumed.
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
// entry: the entry's values are the packed lane's, and the axis a policy names
// is the axis the report prints.

namespace {

// The policy the public axis is named with, on the shipped tables.
template <boys::EvalScheme kScheme>
using OrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders, boys::FitGranularity::kCoarsest>;

// The same axis with the rational route named: one of the two opened calls.
template <boys::EvalScheme kScheme>
using RationalOrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kRationalMinimax, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders, boys::FitGranularity::kCoarsest>;

// The rational route read one order at a time: the lane the routed axis's own reading has to be.
template <boys::EvalScheme kScheme>
using RationalPerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kRationalMinimax, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, boys::FitGranularity::kCoarsest>;

// The per-order lane the entry falls back to outside the packed interval: the
// certified scalar single entry, the one lane the axis cannot be formed on.
template <boys::EvalScheme kScheme>
using PerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, boys::FitGranularity::kCoarsest>;

// The narrow partition on the axis: region-A pieces cut per order, so the
// packed lane fetches each of the four orders it packs its own piece. The
// shipped-table axis is the same body at the same scheme, so the partition is
// the only difference and the two are comparable cell for cell.
template <boys::EvalScheme kScheme>
using NarrowOrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders, boys::FitGranularity::kNarrow>;

// The same partition read one order at a time by the certified scalar single
// entry: the reading the packed lane's gathered fetch has to reproduce.
template <boys::EvalScheme kScheme>
using NarrowPerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, boys::FitGranularity::kNarrow>;

// The uniform partition on the axis: one fixed grid over the whole fitted
// domain, not a cut of region A, so pieces are of one width and every order of a
// cell is stored at one degree. The grid is interval-major - every order of one
// interval lies one stride from the next order's, the shape the shipped lane's
// fetch already steps - with the interval found by a multiply and a truncation
// rather than a scan of piece edges. The rational member over the same grid
// stores one pair per interval and has no stride between orders, so its cell is
// served by the certified scalar orders lane (boys_orders_simd.cpp).
template <boys::EvalScheme kScheme>
using UniformOrdersPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders, boys::FitGranularity::kUniform>;

// The same partition read one order at a time by the certified scalar single
// entry: the lane the entry reproduces in region A and falls back to past the grid.
template <boys::EvalScheme kScheme>
using UniformPerOrderPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, boys::FitGranularity::kUniform>;

} // namespace

// The default policy still names the shipped axis, so a call site naming none is unchanged.
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
        boys::BoysAllOrders<OrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
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
    // Past the lane's interval the entry runs the certified scalar single lane one
    // order at a time, so it is defined for every argument the library accepts, and
    // its values there are that lane's, bit for bit - asserted exactly rather than
    // against a tolerance, because above order 28 these arguments are below the
    // region-C bound, where the value is the recurrence's rounding and carries no
    // structure a check could test.
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
        boys::BoysAllOrders<OrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
            kNmax, x, out.data());

        for (int l = 0; l <= kNmax; ++l)
        {
            // The certified scalar single lane at the policy the entry was called
            // with: the default entry is a different policy, and a measurement moves it.
            const double single =
                boys::BoysSingle<PerOrderPolicy<boys::EvalScheme::kSplitClenshaw>>(l, x);

            if (!SameBits(out[static_cast<std::size_t>(l)], single))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the entry's fallback is not the certified scalar single lane";
}

// The class the run-time route selector stands in for: BoysAllOrders, the double
// lane's ladder entry, which is the entry its unnamed calls are made through.
using RoutedClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;

// The axis at the combination the selector builds: the class's own row, with the
// route and the scheme named where the selector takes them (src/boys.cpp,
// SelectorClass and SelectorPolicy - "the axes a selector does not take are read off
// the class's own default"). Spelling the other axes as the library's five instead
// would compare the axis against a combination this build's class does not compile:
// the routed entry reads the class row, so the two would differ wherever the seam
// carries a row for the class, and the reading below would measure the seam rather
// than the route.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme>
using RoutedAxisPolicy = boys::EvalPolicy<kRoute,
                                          kScheme,
                                          RoutedClass::kBudget,
                                          RoutedClass::kPack,
                                          RoutedClass::kGranularity,
                                          RoutedClass::kDivision,
                                          RoutedClass::kRegionBExp>;

// The rational route on the axis: a policy names the route and the entry answers
// with that route's own region-A fits. The routed per-argument entry reads the same
// stored pairs by the route's own body at the same combination, so the two agree bit
// for bit - and the cell that says which table the axis read is the count of bits it
// does not share with the other route's axis, the reading the narrow partition's test
// above states: "the count of differing bits has to be nonzero, since a lane ignoring
// the route would return that axis's bits exactly". The value difference is printed
// beside it and not asserted: at a partition whose two routes are both accurate on
// the grid, the routes' values agree well inside the arithmetic slack while their
// bits do not, so a slack would measure the two fits' accuracy rather than which
// table was read.
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
            boys::BoysAllOrders<RoutedAxisPolicy<boys::FitRoute::kRationalMinimax, kScheme>>(
                kNmax, x, ours.data());
            boys::BoysAllOrdersWithRoute(
                boys::FitRoute::kRationalMinimax, kScheme, kNmax, x, routed.data());
            boys::BoysAllOrders<RoutedAxisPolicy<boys::FitRoute::kChebyshev, kScheme>>(
                kNmax, x, shipped.data());

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

                // Which table the axis read, in bits: a cell the two routes' axes
                // answer with the same bits is a cell this reading cannot tell
                // apart, whichever route was named.
                if (!SameBits(b, s))
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
                    "and %zu of those kept the default route's table\n",
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

// The narrow partition on the axis. The shipped lane's fetch steps from one order's
// coefficients to the next at a fixed stride, which the double lane's shipped region-A
// table has because every order's pieces share their intervals and degrees. The narrow
// partition's pieces are cut per order, so there is no such stride: the lane fetches
// each of the four orders it packs its own piece and coefficients.
//
// The cost is measured as a DIFFERENCE IN VALUES from the per-order narrow lane, not
// asserted to agree - the two map one argument into one piece by different arithmetic
// (the scalar `2 (x - a) / (b - a) - 1` against the packed
// `fma(x - a, 2 / (b - a), -1)`, the same number up to those roundings) - and the
// worst cell and both counts are printed.
//
// Which table the lane read is a second measurement against the shipped-table axis:
// the same body at the same scheme, so only the partition differs - and the count of
// differing bits has to be nonzero, since a lane ignoring the partition would return
// that axis's bits exactly. Where the two narrow lanes part by more than the slack
// this test does not decide which is right; the next one does, against the committed
// reference.
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
            boys::BoysAllOrders<NarrowOrdersPolicy<kScheme>>(kNmax, x, packed.data());
            boys::BoysAllOrders<OrdersPolicy<kScheme>>(kNmax, x, shippedLane.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                perOrder[static_cast<std::size_t>(l)] =
                    boys::BoysSingle<NarrowPerOrderPolicy<kScheme>>(l, x);
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

// The axis and the per-order narrow lane are two readings of one fit, and where they
// part by more than the slack the committed 80-digit reference decides which is right:
// a difference in values resolved against the function, not an agreement asserted
// between two lanes.
//
// Each cell is judged at the figure the lane that produced it documents - the axis at
// the orders lane's 1e-15 over region A, the per-order lane at the single lane's
// 1e-15 below the extended band and 3e-14 inside it - with the count outside the
// per-order lane's own budget printed.
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
            boys::BoysAllOrders<NarrowOrdersPolicy<kScheme>>(kNmax, cell.x, packed.data());
            const double axis = packed[static_cast<std::size_t>(cell.n)];
            const double perOrder =
                boys::BoysSingle<NarrowPerOrderPolicy<kScheme>>(cell.n, cell.x);
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

// The uniform grid on the axis: the lane's third table, whose domain is not region A.
// Held here is the axis's own claim - the entry's values are the certified per-order
// lane's, bit for bit - over that whole domain, including the join at the grid's end,
// past which the entry runs that same lane. Its value against the committed reference
// is not asserted: the figure that would judge it is the entry's documented one for
// this combination, which the partition's row does not state yet.
TEST(BoysAcrossOrders, TheUniformGridOnTheAxisIsThePerOrderLaneBitForBit) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    // Every cell edge of the grid, one interior point of each cell, the edge
    // approached from below, and the grid's end and its far side.
    std::vector<double> sweep;

    for (int cell = 0; cell < boys::detail::kFlatIntervals; ++cell)
    {
        const double edge =
            static_cast<double>(cell) / (1.0 / boys::detail::kFlatWidth);

        sweep.push_back(edge);
        sweep.push_back(edge + 0.5 * boys::detail::kFlatWidth);
        sweep.push_back(std::nextafter(edge, 0.0));
    }

    sweep.push_back(0.0);
    sweep.push_back(std::nextafter(boys::detail::kFlatHi, 0.0));
    sweep.push_back(boys::detail::kFlatHi);
    sweep.push_back(boys::detail::kFlatHi + 1.0);
    sweep.push_back(200.0);

    std::vector<double> packed(static_cast<std::size_t>(kNmax) + 1);
    std::size_t compared = 0;
    std::size_t differing = 0;
    double worst = 0.0;

    const auto sweepScheme = [&]<boys::EvalScheme kScheme>(const char* name) {
        std::size_t mineCompared = 0;
        std::size_t mineDiffering = 0;
        double mineWorst = 0.0;

        for (const double x : sweep)
        {
            boys::BoysAllOrders<UniformOrdersPolicy<kScheme>>(kNmax, x, packed.data());

            for (int l = 0; l <= kNmax; ++l)
            {
                const double lane = boys::BoysSingle<UniformPerOrderPolicy<kScheme>>(l, x);
                const double delta = std::abs(packed[static_cast<std::size_t>(l)] - lane);

                ++mineCompared;
                mineWorst = delta > mineWorst ? delta : mineWorst;

                if (!SameBits(packed[static_cast<std::size_t>(l)], lane))
                {
                    ++mineDiffering;
                }
            }
        }

        std::printf("  uniform grid, %-16s %zu values, %zu differ | worst from the per-order "
                    "lane %.3e\n",
                    name,
                    mineCompared,
                    mineDiffering,
                    mineWorst);

        compared += mineCompared;
        differing += mineDiffering;
        worst = mineWorst > worst ? mineWorst : worst;
    };

    sweepScheme.template operator()<boys::EvalScheme::kSplitClenshaw>("split clenshaw");
    sweepScheme.template operator()<boys::EvalScheme::kHorner>("horner");

    EXPECT_GT(compared, 0u) << "the sweep measured nothing";

#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
    // This build's scalar route is the two-rounding one, so the lane and the per-order
    // entry are two arithmetics and what is asserted is the region's budget: the packed
    // lane runs the build's own two-rounding route, summed in its own order. The
    // count above still prints, so a reader sees how far apart the two arithmetics are.
    static_cast<void>(differing);
    EXPECT_LE(worst, boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleSingle))
        << "the uniform grid's cell is outside the single lane's region-A budget against the "
           "certified per-order lane, worst "
        << worst << " over " << compared << " values";
#else
    EXPECT_EQ(differing, 0u) << "the uniform grid's cell is not the certified scalar lane's value "
                                "at every order and argument of its domain, worst "
                             << worst << " over " << compared << " values";
#endif
}

// The rational route on the axis, split by which fit answers the cell. A route is a
// selector: BoysFitRoutes() reports the domain each takes over, which the library
// calls the place where "naming it may change a value". Region A's rational route hands
// an order over at that order's own end of the region (kTierThresholds); below that end
// the partition's per-order fit answers the cell, under either route's name.
//
// This test holds that as two counts: a fallback cell is the shipped lane's value, and
// a served cell is the route's own pair. The second count is required nonzero so a lane
// ignoring the route it named cannot pass, and the count of fallback cells that are the
// per-order rational lane's reading is printed beside it, a reader needing both.
//
// The comparison is exact where the build's scalar multiply-add is one-rounding (every
// release and CI leg but the `BOYS_MULADD_SEPARATE` leg); elsewhere the two are two
// arithmetics and each half's agreement is bounded at the figure its lane publishes.
TEST(BoysAcrossOrders, TheRationalRoutesFallbackCellsAreTheShippedLanesReading) {
    if (!VectorTier())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<double> grid = RegionAGrid();

    const auto sweep = [&]<boys::EvalScheme kScheme>(const char* name) {
        std::vector<double> routed(static_cast<std::size_t>(kNmax) + 1);
        std::vector<double> shipped(static_cast<std::size_t>(kNmax) + 1);
        std::size_t servedCells = 0;
        std::size_t servedDiffering = 0;
        std::size_t discriminating = 0;
        std::size_t fallbackCells = 0;
        std::size_t fallbackDiffering = 0;
        std::size_t fallbackPerOrderReading = 0;
        double worstServed = 0.0;
        double worstFallback = 0.0;

        for (double x : grid)
        {
            boys::BoysAllOrders<RationalOrdersPolicy<kScheme>>(kNmax, x, routed.data());
            boys::BoysAllOrders<OrdersPolicy<kScheme>>(kNmax, x, shipped.data());

            // The route's own handover rule, at the entry's nmax: the orders
            // whose own end of region A this argument has reached, and no more.
            int served = 0;

            while (served <= kNmax &&
                   x >= boys::detail::kTierThresholds[static_cast<std::size_t>(served)])
            {
                ++served;
            }

            for (int l = 0; l <= kNmax; ++l)
            {
                const std::size_t at = static_cast<std::size_t>(l);
                const double fromPerOrder =
                    boys::BoysSingle<RationalPerOrderPolicy<kScheme>>(l, x);

                if (l >= served)
                {
                    // The cell the route's selector does not answer: the partition's
                    // own fit, which is the shipped lane's value here.
                    ++fallbackCells;

                    const double shippedHere = boys::BoysSingle<PerOrderPolicy<kScheme>>(l, x);

                    if (!SameBits(routed[at], shippedHere))
                    {
                        ++fallbackDiffering;
                        const double delta = std::abs(routed[at] - shippedHere);
                        worstFallback = delta > worstFallback ? delta : worstFallback;
                    }

                    if (!SameBits(routed[at], fromPerOrder))
                    {
                        ++fallbackPerOrderReading;
                    }

                    continue;
                }

                ++servedCells;

                if (!SameBits(routed[at], fromPerOrder))
                {
                    ++servedDiffering;
                    const double delta = std::abs(routed[at] - fromPerOrder);
                    worstServed = delta > worstServed ? delta : worstServed;
                }

                if (!SameBits(routed[at], shipped[at]))
                {
                    ++discriminating;
                }
            }
        }

        std::printf("  rational route on the axis, %-14s: %zu served cells, %zu differ from the "
                    "per-order rational lane, worst %.3e; %zu fallback cells, %zu differ from the "
                    "shipped lane's value, worst %.3e, of which %zu are the per-order rational "
                    "lane's reading; %zu served cells tell the two routes apart\n",
                    name,
                    servedCells,
                    servedDiffering,
                    worstServed,
                    fallbackCells,
                    fallbackDiffering,
                    worstFallback,
                    fallbackPerOrderReading,
                    discriminating);

#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
        // This build's scalar route is the two-rounding one, so the axis and the
        // per-order entry are two arithmetics: what is asserted is that each half is
        // inside the figure its lane publishes, counts still printed.
        EXPECT_LE(worstServed, boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleBatch))
            << name << ": the routed half is outside the route's own figure";
        EXPECT_LE(worstFallback, kSingleBarA)
            << name << ": the fallback half is outside the shipped lane's figure";
#else
        EXPECT_EQ(servedDiffering, 0u)
            << name << ": the cells the route's selector takes over are not the rational route's "
                       "own reading, worst "
            << worstServed;
        EXPECT_EQ(fallbackDiffering, 0u)
            << name << ": the cells the route does not answer are not the shipped lane's value, "
                       "worst "
            << worstFallback;
#endif
        EXPECT_GT(discriminating, 0u)
            << name << ": no cell of the grid tells the two routes apart, so this sweep cannot "
                       "say which table the axis read";
    };

    sweep.template operator()<boys::EvalScheme::kSplitClenshaw>("split clenshaw");
    sweep.template operator()<boys::EvalScheme::kHorner>("horner");
}

// The opened call's figure, measured where every other row is: against the
// committed 80-digit reference grid, over its region-A cells, at the figure the library
// states. The route's fits hold the batch budget in region A, read off the library's own
// table.
TEST(BoysAcrossOrders, TheOpenedCallMeetsItsFigureOnTheReferenceGrid) {
    const std::vector<ReferenceCell> cells = RegionAReference();

    if (cells.empty())
    {
        GTEST_SKIP() << "the reference grid is not present in this tree";
    }

    constexpr double kRouteBound =
        boys::detail::RegionABudget(boys::detail::BoysRole::kDoubleBatch);

    // A batch-route cell is documented at one figure wherever it sits, so its bar is flat.
    const auto routeBar = [](double) { return kRouteBound; };

    const auto measure = [&]<class Policy, class Bar>(const char* label, Bar bar) {
        const BarShortfall worst = PolicyAgainstReference<Policy>(cells, bar);

        std::printf("  %-34s worst %.6e at n = %d, x = %.12g  (figure %.2e)\n",
                    label,
                    worst.error,
                    worst.n,
                    worst.x,
                    worst.figure);
        EXPECT_LE(worst.error, worst.figure) << label;
    };

    measure.template operator()<RationalOrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
        "rational route, split clenshaw", routeBar);
    measure.template operator()<RationalOrdersPolicy<boys::EvalScheme::kHorner>>(
        "rational route, horner", routeBar);
    // The narrow partition reads the single lane's own fits one order at a time, so
    // its rows carry the lane's figures, each cell judged at its own region's.
    measure.template operator()<NarrowOrdersPolicy<boys::EvalScheme::kSplitClenshaw>>(
        "narrow partition on the axis, split clenshaw", SingleBar);
    measure.template operator()<NarrowOrdersPolicy<boys::EvalScheme::kHorner>>(
        "narrow partition on the axis, horner", SingleBar);

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
