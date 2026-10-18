// The default coefficient set's reachability: this unit includes the front door only, so the nine
// groups stated present are what <boys/boys.hpp> opens and the twelve absent are those a caller
// must open by name. Every host entry is called below, holding the set to its question: no default
// entry reads a group the front door leaves out.

#include <boys/boys.hpp>

#include <array>
#include <cstddef>
#include <gtest/gtest.h>

namespace {

// The nine groups the front door opens: the committed class rows, plus the double narrow
// Chebyshev table the half-format rows' batch seeds read before they round to single.
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::UniformMonoGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::UniformMonoTransposedGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::F32UniformMonoGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::NarrowMonoGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::NarrowRatGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::NarrowChebGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::F32NarrowChebGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::F32NarrowMonoGroup>);
static_assert(boys::detail::kCoefficientGroupPresent<boys::detail::F32NarrowRatGroup>);

// The twelve groups a policy outside the class rows names: the fit's assertion prints the
// header that defines each of these. A build carrying a replacement seam may open some of them
// by name, so the absence is stated of the committed seam and not of every build.
#if !defined(BOYS_BUILD_DEFAULTS_REPLACED)
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::UniformChebGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::UniformChebTransposedGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::UniformRatGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::F32UniformChebGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::F32UniformRatGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::ScaledFarGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::CoarseChebGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::CoarseMonoGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::F32CoarseChebGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::F32CoarseMonoGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::F32CoarseRatGroup>);
static_assert(!boys::detail::kCoefficientGroupPresent<boys::detail::CoarseRatGroup>);
#endif // !defined(BOYS_BUILD_DEFAULTS_REPLACED)

TEST(DefaultTables, EveryHostEntryRunsOnTheDefaultSet) {
    constexpr int kNmax = boys::kMaxBoysOrder;
    constexpr std::size_t kCount = 3;
    const double x[kCount] = {0.25, 3.5, 40.0};
    const float xf[kCount] = {0.25f, 3.5f, 40.0f};
    const int orders[kCount] = {0, 4, kNmax};

    std::array<double, boys::kMaxBoysOrder + 1> out{};
    std::array<double, kCount * (boys::kMaxBoysOrder + 1)> batch{};
    std::array<std::size_t, boys::BoysAllNWorkspaceSize(kCount)> workspace{};
    std::array<float, boys::kMaxBoysOrder + 1> outf{};
    std::array<float, kCount * (boys::kMaxBoysOrder + 1)> batchf{};

    EXPECT_GT(boys::BoysSingle(3, 0.5), 0.0);
    EXPECT_EQ(boys::BoysSingleChecked(3, 0.5, out.data()), boys::BoysStatus::kSuccess);
    boys::BoysAllOrders(kNmax, 0.5, out.data());
    EXPECT_EQ(boys::BoysAllOrdersChecked(kNmax, 0.5, out.data()), boys::BoysStatus::kSuccess);
    boys::BoysFixedN(3, x, batch.data(), kCount);
    EXPECT_EQ(boys::BoysFixedNChecked(3, x, batch.data(), kCount, 1),
              boys::BoysStatus::kSuccess);
    boys::BoysAllN(kNmax, x, batch.data(), kCount, workspace.data());
    EXPECT_EQ(boys::BoysAllNChecked(kNmax, x, batch.data(), kCount, workspace.data()),
              boys::BoysStatus::kSuccess);
    boys::BoysAllN(kNmax, x, batch.data(), kCount, boys::BoysSortedArgs{});
    boys::BoysAllNAtOrders(orders, x, batch.data(), kCount);
    EXPECT_EQ(boys::BoysAllNAtOrdersChecked(orders, x, batch.data(), kCount),
              boys::BoysStatus::kSuccess);

    EXPECT_GT(boys::BoysSingleF32(3, 0.5f), 0.0f);
    boys::BoysAllOrdersF32(kNmax, 0.5f, outf.data());
    boys::BoysFixedNF32(3, xf, batchf.data(), kCount);
    boys::BoysAllNF32(kNmax, xf, batchf.data(), kCount);
    boys::BoysAllNAtOrdersF32(orders, xf, batchf.data(), kCount);

#if BoysFp16
    std::array<boys::F16, boys::kMaxBoysOrder + 1> out16{};
    std::array<boys::Bf16, boys::kMaxBoysOrder + 1> outbf{};
    std::array<boys::F16, kCount * (boys::kMaxBoysOrder + 1)> batch16{};
    std::array<boys::Bf16, kCount * (boys::kMaxBoysOrder + 1)> batchbf{};
    std::array<boys::F16, kCount> x16{boys::F16(0.25f), boys::F16(3.5f), boys::F16(40.0f)};
    std::array<boys::Bf16, kCount> xbf{boys::Bf16(0.25f), boys::Bf16(3.5f), boys::Bf16(40.0f)};

    EXPECT_GT(static_cast<float>(boys::BoysSingleF16(3, boys::F16(0.5f))), 0.0f);
    boys::BoysAllOrdersF16(kNmax, boys::F16(0.5f), out16.data());
    boys::BoysFixedNF16(3, x16.data(), batch16.data(), kCount);
    boys::BoysAllNF16(kNmax, x16.data(), batch16.data(), kCount);
    boys::BoysAllNAtOrdersF16(orders, x16.data(), batch16.data(), kCount);

    EXPECT_GT(static_cast<float>(boys::BoysSingleBf16(3, boys::Bf16(0.5f))), 0.0f);
    boys::BoysAllOrdersBf16(kNmax, boys::Bf16(0.5f), outbf.data());
    boys::BoysFixedNBf16(3, xbf.data(), batchbf.data(), kCount);
    boys::BoysAllNBf16(kNmax, xbf.data(), batchbf.data(), kCount);
    boys::BoysAllNAtOrdersBf16(orders, xbf.data(), batchbf.data(), kCount);
#endif
}

TEST(DefaultTables, EveryDefaultCallAgreesOnTheScalarValue) {
    std::array<double, boys::kMaxBoysOrder + 1> out{};
    boys::BoysAllOrders(6, 0.5, out.data());
    EXPECT_DOUBLE_EQ(boys::BoysSingle(6, 0.5), out[6]);
}

} // namespace
