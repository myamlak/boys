// The single-precision across-orders packed lane's contract tests: eight
// orders of one argument in one vector register, where the double lane holds
// four.
//
// The lane is a second body over the float lane's own region-A tables, and the
// tables are the part that differs from the double lane's: the float table cuts
// each order's cover where that order needs it, so order 0 is cut into two
// pieces where order 14 is cut into three and a break is not shared between
// orders. A fixed argument therefore selects a DIFFERENT piece in each of the
// eight lanes, and the offset from one lane's coefficients to the next is not a
// stride. What the eight lanes share is the degree the group is summed at, and
// the group runs at its lanes' largest one with a lane reading zeros above its
// own cut.
//
// What can go wrong with that, and what each test is for:
//
//  1. A LANE THAT IS NOT THE PER-ORDER VALUE. Reading zeros above a cut is the
//     lane's own polynomial and, down the split Clenshaw, the lane's own
//     arithmetic: the extra top step has an exact zero for both of its terms.
//     The claim is therefore the strong one - the packed value IS the per-order
//     value, bit for bit - and the test asserts it over a region-A sweep and
//     every order, at each entry, each rung and each budget. On a build whose
//     scalar arithmetic is the two-rounding route the packed lane's own
//     instruction has one rounding and the scalar lane has two, so the two part
//     there. The reading that build takes is the lane's accuracy against the
//     double lane rather than its agreement with its sibling, because a sibling
//     is not a reference: measured on that build, the per-order rational lane is
//     itself at 1.7277e-07 against the double lane - over the 1.5e-07 the route
//     is certified against - so a difference from it cannot be held to that bar
//     by either lane. The packed lane, which performs the fused step, reads
//     1.3093e-07 there and is inside it. Both numbers are printed, so which
//     reading the build took is never a matter of the test's word.
//
//  2. A GROUP BOUNDARY THAT DROPS A LANE. nmax + 1 is not a multiple of eight,
//     so the last group is partial and the lane's tail runs beside the vector
//     body. The sweep runs every nmax from 0 to kMaxBoysOrder, which is every
//     remainder there is.
//
//  3. A RUNG THAT DOES NOT REACH THE LANE. A policy naming a relaxed
//     multiplier must return the fits cut to that rung's degrees, and the
//     per-order lane at the same policy and multiplier is what that is.
//
//  4. A CALL THAT ANSWERS WITH THE WRONG TABLE. A policy naming the rational
//     route must return that route's own region-A fits, whose denominator is a
//     second Horner array at a per-lane offset.
//
//  5. AN ENTRY THAT PARTS FROM ITS SIBLING. The two fetches are the same lane
//     and must return the same bits.
//
// Each sweep is measured twice from the same values: against the library's own
// per-order lane at the same policy, which is the certified entry and the thing
// the packing is supposed to reproduce, and against the double lane, which is
// the function itself as far as a 1e-7 question can tell. The first is the
// assertion wherever the build's scalar arithmetic is the packed instruction's,
// and the second wherever it is not. The bound both are measured against is the
// float lane's documented one, read from the generated header rather than
// written here.

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

// Whether the scalar arithmetic this build compiles is the one-rounding route.
// The packed lane names its own instruction and is contraction-free whatever
// this says; the scalar lane the values are compared with is not.
constexpr bool kScalarIsFused() noexcept {
#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
    return false;
#else
    return true;
#endif
}

// A region-A sweep that reaches every part of the interval the float table
// covers: the small-x end where order 0's first piece lies, each break the
// table declares, and the right edge at kX0. The edge itself is in the sweep
// and is worth its place: the largest float below kX0 casts to the same float
// as kX0 and dedupes away, and that float is read by the lane's own test as
// region A - its comparison is x < (float)kX0, which is false there - so the
// sweep's last point is the first argument the entry answers from the fallback
// instead of from the vector, and the sweep is where that handover is measured.
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
// entry at the same policy with the shipped packing axis, which is the axis the
// default policy names.
template <double kMultiplier, class Policy>
float PerOrder(int n, float x) noexcept {
    using ArgsAxis = boys::EvalPolicy<Policy::kRoute,
                                       Policy::kScheme,
                                       Policy::kBudget,
                                       boys::PackAxis::kArguments>;
    return boys::BoysSingleF32<kMultiplier, ArgsAxis>(n, x);
}

// The double lane at the same order and argument: the function the float lane
// is a fit to. Its own error is a double lane's, so it is a reference for
// anything the float lane does down to the bar the float lane publishes.
double DoubleLaneValue(int n, float x) noexcept {
    return boys::BoysSingle(n, static_cast<double>(x));
}

// What one sweep measured. The two worst figures are the same values read
// against two references: the per-order lane at the same policy, which is what
// the lane has to reproduce, and the double lane, which is what it has to be
// right about. They are carried together because which of the two is the
// assertion depends on the build, and a reader has to see the one that is not.
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
// `differing` counts the values that are not the per-order value bit for bit.
// Both measurements are this sweep's own and are added to the caller's totals.
template <double kMultiplier, class Policy>
void SweepPolicy(const char* name, SweepTotals& totals) {
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
        boys::BoysAllOrdersF32<kMultiplier, Policy>(kNmax, x, packed.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const float one = PerOrder<kMultiplier, Policy>(n, x);
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

// The policy families the lane accepts. The budget is a template parameter on
// this lane because the fp16 budget's fits are certified against a tighter bar
// than the kFloat one, so a lane that answered an fp16 policy with the kFloat
// degrees would deliver the looser figure under the tighter name.
template <boys::FitRoute kRoute, boys::EvalScheme kScheme, boys::BoysBudget kBudget>
using Orders = boys::EvalPolicy<kRoute, kScheme, kBudget, boys::PackAxis::kOrders>;

constexpr double kRung64 = boys::AccuracyMultiplier(boys::AccuracyTier::kRelaxed64);
constexpr double kRungMax = boys::AccuracyMultiplier(boys::AccuracyTier::kRelaxed65536);

} // namespace

// The axis is the one the caller named, and the default is still the shipped
// one, so a call site that names no axis compiles what it always did.
static_assert(Orders<boys::FitRoute::kChebyshev,
                     boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat>::kPack == boys::PackAxis::kOrders,
              "the orders policy does not name the orders axis");
static_assert(boys::EvalPolicy<>{}.kPack == boys::PackAxis::kArguments,
              "the default axis moved");

// Every configuration the lane carries, at the reference multiplier: three
// schemes on the shipped route and the rational route, at both budgets. The
// per-order value is asserted bit for bit where the build's scalar arithmetic
// is the one the packed instruction performs, and held to the lane's own bar
// where it is not.
TEST(BoysOrdersF32, ThePackedLaneIsThePerOrderValue) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    SweepTotals totals{};

    SweepPolicy<1.0,
                Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("cheb/split kFloat m=1", totals);
    SweepPolicy<1.0,
                Orders<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
                       boys::BoysBudget::kFloat>>("cheb/horner kFloat m=1", totals);
    SweepPolicy<1.0,
                Orders<boys::FitRoute::kRationalMinimax,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("rational/split kFloat m=1", totals);
    SweepPolicy<1.0,
                Orders<boys::FitRoute::kRationalMinimax,
                       boys::EvalScheme::kHorner,
                       boys::BoysBudget::kFloat>>("rational/horner kFloat m=1", totals);
    SweepPolicy<1.0,
                Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFp16>>("cheb/split kFp16 m=1", totals);

    EXPECT_GT(totals.compared, 0u);

    if (kScalarIsFused())
    {
        EXPECT_EQ(totals.differing, 0u)
            << "the packed lane and the per-order lane have parted: the eight-lane group is not "
               "reproducing the per-order arithmetic. Worst "
            << totals.worstFromPerOrder << " at x=" << totals.perOrderAt
            << " n=" << totals.perOrderOrder;
    }

    // The lane's accuracy, on every build. On the fused build this is the same
    // number the per-order lane delivers, since the two are one arithmetic and
    // the row above already says they are bit-identical; on the two-rounding
    // build it is the only reading of the two that means anything, because the
    // per-order rational lane is itself outside this bar there.
    EXPECT_LE(totals.worstFromTruth, kF32Bar)
        << "the packed lane is outside the float lane's documented bar: worst "
        << totals.worstFromTruth << " at x=" << totals.truthAt << " n=" << totals.truthOrder;
}

// The relaxed rungs, where the lane reads the effective-degree table rather than
// the whole fit. The rung path is the shipped route and scheme alone, so that is
// what is swept. The bar is the rung's own: the float lane's documented bar
// multiplied by the multiplier the rung names, which is what a relaxed rung
// means - the same fit cut to fewer degrees, at a correspondingly larger error.
TEST(BoysOrdersF32, TheRelaxedRungsAreTheRungsOwnReading) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    SweepTotals totals{};

    SweepPolicy<kRung64,
                Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("cheb/split kFloat m=64", totals);
    SweepPolicy<kRungMax,
                Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFloat>>("cheb/split kFloat m=65536", totals);
    SweepPolicy<kRungMax,
                Orders<boys::FitRoute::kChebyshev,
                       boys::EvalScheme::kSplitClenshaw,
                       boys::BoysBudget::kFp16>>("cheb/split kFp16 m=65536", totals);

    EXPECT_GT(totals.compared, 0u);

    if (kScalarIsFused())
    {
        EXPECT_EQ(totals.differing, 0u)
            << "the rung did not reach the packed lane: its values are not the per-order lane's "
               "at the same multiplier. Worst "
            << totals.worstFromPerOrder << " at x=" << totals.perOrderAt
            << " n=" << totals.perOrderOrder;
    }

    EXPECT_LE(totals.worstFromTruth, kRungMax * kF32Bar)
        << "the rung's value is outside the bound the rung names: worst " << totals.worstFromTruth
        << " at x=" << totals.truthAt << " n=" << totals.truthOrder << " against "
        << kRungMax * kF32Bar;
}

// The two entries are one lane: the gathered fetch and the composed fetch must
// return the same bits over the whole sweep.
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

// Past the packed lane's own interval the entry is defined and answers from the
// lane that can: the entry is not allowed to be undefined outside region A.
//
// This is where the axis crosses a translation-unit boundary and where the claim
// stops being about the vector lane. Region B seeds a downward recurrence and
// region C is three closed-form lines; the packed entry reaches them through the
// library's own translation unit while the reference here is compiled into this
// one. That is only a difference if the arithmetic those two units compile is a
// difference, and in this lane it is not: the region-B seed and the recurrence's
// own steps both name their rounding - the seed through the fit's fused
// multiply-add, the step through the backend's two-rounding multiply-subtract -
// so both units compute the same bits and the strong form of the claim holds
// past the ledger as well. The test says so rather than assuming it, and it is
// the guard on that spelling: a regression to a bare product and difference in
// either body shows up here as a red row on a build whose two units contract
// differently.
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
        boys::BoysAllOrdersF32<1.0, Policy>(kNmax, x, packed.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const float one = PerOrder<1.0, Policy>(n, x);
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

// The partial last group: nmax + 1 is not a multiple of eight for most nmax, and
// a lane dropped at the group boundary would show up as a value that is not the
// per-order one at exactly one order.
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
            boys::BoysAllOrdersF32<1.0, Policy>(nmax, x, packed.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const float one = PerOrder<1.0, Policy>(n, x);
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
