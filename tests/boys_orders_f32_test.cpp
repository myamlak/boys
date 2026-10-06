// The single-precision across-orders packed lane's contract tests: eight orders of one argument in one vector
// register, where the double lane holds four. It is a second body over the float lane's own region-A tables,
// which cut each order's cover where that order needs it (order 0 two pieces, order 14 three, no break shared),
// so a fixed argument selects a different piece in every lane and the group runs at its lanes' largest degree.

// Each sweep is measured twice, against the library's per-order lane at the same policy and against the double
// lane; which is the assertion is kScalarIsFused, and on the two-rounding route the two part, where the per-order
// rational lane reads 1.7277e-07 against the double lane - over the 1.5e-07 the route is certified against - and
// the packed lane reads 1.3093e-07, inside it. The bound is the float lane's documented one, from the header.

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"
#include "boys_orders_simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

namespace {

constexpr int kNmax = boys::kMaxBoysOrder;
constexpr double kX0 = boys::detail::kX0;

// The float lane's documented bar for its region-A fits, read from the
// generated header so a regeneration moves the test with the claim.
constexpr double kF32Bar = boys::detail::f32::kRegionAFitBar;

bool SameBitsF32(float a, float b) noexcept {
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// Whether the two lanes this file compares run one arithmetic: both read the build's multiply-add
// selection, but the packed lane spells its own steps from it (kSelectedRoute, src/boys_orders_simd.cpp)
// while the scalar lane takes it through the build's contraction (include/boys/backend.hpp, RouteInForce).
// The bit-identity is asserted where the fused route is selected and printed where it is not.
constexpr bool kScalarIsFused() noexcept {
#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
    return false;
#else
    return true;
#endif
}

// A region-A sweep reaching every part of the interval the float table covers: the small-x end where
// order 0's first piece lies, each break the table declares, and the right edge at kX0. The largest
// float below kX0 casts to the same float and dedupes away, so that edge is answered from the fallback
// rather than from the vector (the lane's own test is x < (float)kX0), and that handover is measured here.
std::vector<float> RegionAGridF32() {
    std::vector<float> grid;
    const std::size_t logSteps = 1200;

    for (std::size_t i = 0; i <= logSteps; ++i)
    {
        const double u = static_cast<double>(i) / static_cast<double>(logSteps);
        grid.push_back(
            static_cast<float>(std::exp(std::log(1e-6) + u * (std::log(kX0) - std::log(1e-6)))));
    }

    for (const auto& piece : boys::detail::f32::kPieces)
    {
        for (double d : {-1e-6, -1e-9, 0.0, 1e-9, 1e-6})
        {
            const float at = static_cast<float>(static_cast<double>(piece.b) + d);

            if (at >= 0.0f && static_cast<double>(at) < kX0)
            {
                grid.push_back(at);
            }
        }
    }

    grid.push_back(static_cast<float>(std::nextafter(kX0, 0.0)));
    std::sort(grid.begin(), grid.end());
    grid.erase(std::unique(grid.begin(), grid.end()), grid.end());
    grid.erase(std::remove_if(grid.begin(),
                              grid.end(),
                              [](float x) { return !(static_cast<double>(x) < kX0); }),
                grid.end());
    return grid;
}

// The per-order lane the packed lane must reproduce: the certified float single
// entry at the same policy with the shipped packing axis, the axis the default
// policy names.
template <class Policy> float PerOrder(int n, float x) noexcept {
    using ArgsAxis = boys::EvalPolicy<Policy::kRoute,
                                       Policy::kScheme,
                                       Policy::kBudget,
                                       boys::PackAxis::kArguments>;
    return boys::BoysSingleF32<ArgsAxis>(n, x);
}

// The double lane at the same order and argument: the function the float lane is a
// fit to. Its own error is a double lane's, so it is a reference down to the bar the
// float lane publishes.
double DoubleLaneValue(int n, float x) noexcept {
    return boys::BoysSingle(n, static_cast<double>(x));
}

// What one sweep measured. The two worst figures are the same values read against
// two references: the per-order lane at the same policy, and the double lane. They
// are carried together because which of the two is the assertion depends on the
// build, and a reader has to see the one that is not.
struct SweepTotals {
    std::size_t compared = 0;
    std::size_t differing = 0;
    double worstFromPerOrder = 0.0;
    double worstFromTruth = 0.0;
    float perOrderAt = 0.0f;
    int perOrderOrder = 0;
    float truthAt = 0.0f;
    int truthOrder = 0;
};

// One policy's sweep: every order from 0 to kNmax, every argument of the grid.
// `differing` counts the values that are not the per-order value bit for bit, and
// both measurements are this sweep's own, added to the caller's totals.
template <class Policy> void SweepPolicy(const char* name, SweepTotals& totals) {
    const std::vector<float> grid = RegionAGridF32();
    std::vector<float> packed(static_cast<std::size_t>(kNmax) + 1, 0.0f);
    std::size_t mineDiffering = 0;
    double mineFromPerOrder = 0.0;
    double mineFromTruth = 0.0;
    float fromPerOrderAt = 0.0f;
    int fromPerOrderOrder = 0;
    float fromTruthAt = 0.0f;
    int fromTruthOrder = 0;

    for (float x : grid)
    {
        boys::BoysAllOrdersF32<Policy>(kNmax, x, packed.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const float one = PerOrder<Policy>(n, x);
            const float mine = packed[static_cast<std::size_t>(n)];
            const double fromPerOrder =
                std::abs(static_cast<double>(mine) - static_cast<double>(one));
            const double fromTruth = std::abs(static_cast<double>(mine) - DoubleLaneValue(n, x));

            if (!SameBitsF32(mine, one))
            {
                ++mineDiffering;
            }

            if (fromPerOrder > mineFromPerOrder)
            {
                mineFromPerOrder = fromPerOrder;
                fromPerOrderAt = x;
                fromPerOrderOrder = n;
            }

            if (fromTruth > mineFromTruth)
            {
                mineFromTruth = fromTruth;
                fromTruthAt = x;
                fromTruthOrder = n;
            }
        }
    }

    std::printf("  %-34s %6zu values, %6zu differ | from per-order %.3e (x=%g n=%d) | from the "
                "double lane %.3e (x=%g n=%d) | bar %.3e\n",
                name,
                grid.size() * (static_cast<std::size_t>(kNmax) + 1),
                mineDiffering,
                mineFromPerOrder,
                static_cast<double>(fromPerOrderAt),
                fromPerOrderOrder,
                mineFromTruth,
                static_cast<double>(fromTruthAt),
                fromTruthOrder,
                kF32Bar);

    totals.differing += mineDiffering;
    totals.compared += grid.size() * (static_cast<std::size_t>(kNmax) + 1);

    if (mineFromPerOrder > totals.worstFromPerOrder)
    {
        totals.worstFromPerOrder = mineFromPerOrder;
        totals.perOrderAt = fromPerOrderAt;
        totals.perOrderOrder = fromPerOrderOrder;
    }

    if (mineFromTruth > totals.worstFromTruth)
    {
        totals.worstFromTruth = mineFromTruth;
        totals.truthAt = fromTruthAt;
        totals.truthOrder = fromTruthOrder;
    }
}

// The policy families the lane accepts. The budget is a template parameter because
// the fp16 fits are certified against a tighter bar than the kFloat ones, so an fp16
// policy answered with kFloat degrees would deliver the looser figure under the
// tighter name.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme, boys::BoysBudget kBudget>
using Orders = boys::EvalPolicy<kRoute, kScheme, kBudget, boys::PackAxis::kOrders>;

} // namespace

// The axis is the one the caller named, and the default is still the shipped one.
static_assert(Orders<boys::FitRoute::kChebyshev,
                     boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat>::kPack == boys::PackAxis::kOrders,
              "the orders policy does not name the orders axis");
static_assert(boys::EvalPolicy<>{}.kPack == boys::PackAxis::kArguments,
              "the default axis moved");

// Every configuration the lane carries: three schemes on the shipped route and
// the rational route, at both budgets.
TEST(BoysOrdersF32, ThePackedLaneIsThePerOrderValue) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    SweepTotals totals{};

    SweepPolicy<Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("cheb/split kFloat", totals);
    SweepPolicy<Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kHorner,
                       boys::BoysBudget::kFloat>>("cheb/horner kFloat", totals);
    SweepPolicy<Orders<boys::FitRoute::kRationalMinimax,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("rational/split kFloat", totals);
    SweepPolicy<Orders<boys::FitRoute::kRationalMinimax,
                       boys::EvalScheme::kHorner,
                       boys::BoysBudget::kFloat>>("rational/horner kFloat", totals);
    SweepPolicy<Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFp16>>("cheb/split kFp16", totals);

    EXPECT_GT(totals.compared, 0u);

    if (kScalarIsFused())
    {
        EXPECT_EQ(totals.differing, 0u)
            << "the packed lane and the per-order lane have parted: the eight-lane group is not "
               "reproducing the per-order arithmetic. Worst "
            << totals.worstFromPerOrder << " at x=" << totals.perOrderAt
            << " n=" << totals.perOrderOrder;
    }

    // The lane's accuracy, on every build. On the fused build it is the same number
    // the per-order lane delivers, the row above having said they are bit-identical;
    // on the two-rounding build it is the only reading that means anything, the
    // per-order rational lane being outside this bar there.
    EXPECT_LE(totals.worstFromTruth, kF32Bar)
        << "the packed lane is outside the float lane's documented bar: worst "
        << totals.worstFromTruth << " at x=" << totals.truthAt << " n=" << totals.truthOrder;
}

// The gathered fetch and the composed fetch are one lane and must return the same bits.
TEST(BoysOrdersF32, TheComposedFetchIsTheGatheredFetch) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::vector<float> grid = RegionAGridF32();
    std::vector<float> gathered(static_cast<std::size_t>(kNmax) + 1, 0.0f);
    std::vector<float> composed(static_cast<std::size_t>(kNmax) + 1, 0.0f);
    std::size_t differing = 0;

    for (float x : grid)
    {
        boys::detail::BoysAllOrdersF32Simd(boys::detail::OrdersScheme::kSplitClenshaw,
                                           boys::FitRoute::kChebyshev,
                                           kNmax,
                                           x,
                                           gathered.data());
        boys::detail::BoysAllOrdersF32SimdComposed(boys::detail::OrdersScheme::kSplitClenshaw,
                                                   boys::FitRoute::kChebyshev,
                                                   kNmax,
                                                   x,
                                                   composed.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            if (!SameBitsF32(gathered[static_cast<std::size_t>(n)],
                             composed[static_cast<std::size_t>(n)]))
            {
                ++differing;
            }
        }
    }

    EXPECT_EQ(differing, 0u) << "the two fetches have parted";
}

// Past the packed lane's own interval the entry is defined and answers from the lane that can: region B
// seeds a downward recurrence, region C is three closed-form lines, and the packed entry reaches both
// through the library's translation unit while the reference here is compiled into this one. Both name
// their rounding (fit fused multiply-add; backend two-rounding multiply-subtract), so the bits agree.
TEST(BoysOrdersF32, TheAxisIsDefinedPastItsOwnDomain) {
    using Policy = Orders<boys::FitRoute::kChebyshev,
                          boys::EvalScheme::kSplitClenshaw,
                          boys::BoysBudget::kFloat>;

    std::vector<float> packed(static_cast<std::size_t>(kNmax) + 1, 0.0f);
    std::size_t differing = 0;
    std::size_t compared = 0;
    double worst = 0.0;
    float worstAt = 0.0f;
    int worstOrder = 0;

    const float grid[] = {static_cast<float>(kX0),
                          static_cast<float>(kX0) + 1.0f,
                          20.0f,
                          static_cast<float>(boys::detail::kX1),
                          1e6f};

    for (float x : grid)
    {
        boys::BoysAllOrdersF32<Policy>(kNmax, x, packed.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const float one = PerOrder<Policy>(n, x);
            const float mine = packed[static_cast<std::size_t>(n)];
            const double difference =
                std::abs(static_cast<double>(mine) - static_cast<double>(one));
            ++compared;

            if (!SameBitsF32(mine, one))
            {
                ++differing;
            }

            if (difference > worst)
            {
                worst = difference;
                worstAt = x;
                worstOrder = n;
            }
        }
    }

    std::printf("  past the ledger: %zu values over %zu arguments, %zu differ, worst |d| %.3e "
                "at x=%g n=%d\n",
                compared,
                sizeof(grid) / sizeof(grid[0]),
                differing,
                worst,
                static_cast<double>(worstAt),
                worstOrder);

    EXPECT_GT(compared, 0u);
    EXPECT_EQ(differing, 0u) << "the fallback past the packed lane's interval is not the "
                                "policy's own lane";
}

// The partial last group: nmax + 1 is not a multiple of eight for most nmax, so a
// lane dropped at the boundary would show up as one order that is not per-order.
TEST(BoysOrdersF32, EveryGroupTailIsThePerOrderValue) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    using Policy = Orders<boys::FitRoute::kChebyshev,
                          boys::EvalScheme::kSplitClenshaw,
                          boys::BoysBudget::kFloat>;

    const std::vector<float> grid = RegionAGridF32();
    std::vector<float> packed(static_cast<std::size_t>(kNmax) + 1, 0.0f);
    std::size_t differing = 0;
    std::size_t compared = 0;
    double worstFromPerOrder = 0.0;
    double worstFromTruth = 0.0;

    for (int nmax = 0; nmax <= kNmax; ++nmax)
    {
        for (float x : grid)
        {
            boys::BoysAllOrdersF32<Policy>(nmax, x, packed.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const float one = PerOrder<Policy>(n, x);
                const float mine = packed[static_cast<std::size_t>(n)];
                ++compared;

                if (!SameBitsF32(mine, one))
                {
                    ++differing;
                }

                worstFromPerOrder = std::max(
                    worstFromPerOrder,
                    std::abs(static_cast<double>(mine) - static_cast<double>(one)));
                worstFromTruth =
                    std::max(worstFromTruth,
                             std::abs(static_cast<double>(mine) - DoubleLaneValue(n, x)));
            }
        }
    }

    std::printf("  every nmax 0..%d: %zu values, %zu differ | from per-order %.3e | from the "
                "double lane %.3e | bar %.3e\n",
                kNmax,
                compared,
                differing,
                worstFromPerOrder,
                worstFromTruth,
                kF32Bar);

    EXPECT_GT(compared, 0u);

    if (kScalarIsFused())
    {
        EXPECT_EQ(differing, 0u) << "a partial last group is not carrying the per-order value";
    }

    EXPECT_LE(worstFromTruth, kF32Bar)
        << "a partial last group is outside the float lane's documented bar";
}

namespace {

// The single-order entry at one scheme and one partition, at the arguments the
// uniform partition's own domain ends inside of.
template <boys::EvalScheme kScheme, boys::FitGranularity kGranularity>
using SingleF32 =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments, kGranularity, boys::DivisionForm::kExactDivision>;

// What the uniform partition answers over the band between the grid's join and
// kX1, read against the shipped (coarsest) partition's own value there.
template <boys::EvalScheme kScheme>
void UniformPastJoinSweep(std::size_t& compared, std::size_t& fromShipped, float& firstAt,
                          int& firstOrder) {
    using Uniform = SingleF32<kScheme, boys::FitGranularity::kUniform>;
    using Shipped = SingleF32<kScheme, boys::FitGranularity::kCoarsest>;

    const float lo = boys::detail::f32::kFlatHiF32;
    const float hi = static_cast<float>(boys::detail::kX1);

    for (int step = 0; step <= 64; ++step)
    {
        const float x = lo + (hi - lo) * (static_cast<float>(step) / 64.0f);

        for (int n = 0; n <= 8; ++n)
        {
            const float shipped = boys::BoysSingleF32<Shipped>(n, x);
            const float got = boys::BoysSingleF32<Uniform>(n, x);

            ++compared;

            if (got != shipped)
            {
                ++fromShipped;
                firstAt = x;
                firstOrder = n;
            }
        }
    }
}

} // namespace

// Above the uniform grid's join (kFlatHiF32) the entry's region path answers, and it must answer with
// the partition the caller named: a fit that resolves every partition but the shipped one to the NARROW
// pieces would answer a uniform policy with the narrow member's region-B seed bit for bit over the whole
// of [kFlatHiF32, kX1). The test is stated on values rather than on a table: uniform must return shipped.
TEST(BoysOrdersF32, TheUniformPartitionsAnswerPastItsJoinIsTheShippedMember) {
    std::size_t compared = 0;
    std::size_t fromShipped = 0;
    float firstAt = 0.0f;
    int firstOrder = -1;

    UniformPastJoinSweep<boys::EvalScheme::kSplitClenshaw>(
        compared, fromShipped, firstAt, firstOrder);
    UniformPastJoinSweep<boys::EvalScheme::kHorner>(compared, fromShipped, firstAt, firstOrder);

    std::printf("  uniform partition over [kFlatHiF32, kX1): %zu values, %zu differ from the "
                "shipped member's value\n",
                compared,
                fromShipped);

    EXPECT_GT(compared, 0u);

    EXPECT_EQ(fromShipped, 0u)
        << "the uniform partition's value past the grid's join is not the shipped member's, first "
           "at x="
        << firstAt << " n=" << firstOrder;
}
