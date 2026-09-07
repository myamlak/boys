// The C linkage surface contract (boys_c.h): scalar/batch entries match the
// C++ lanes, the multiplier dispatch is exact over the sampled set, and the
// validation rules return the documented status codes.
#include "boys/boys.hpp"
#include "boys/boys_c.h"
#include "boys_effective_degrees.hpp"
#include "boys_impl.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {

constexpr double kX0 = boys::detail::kX0;
constexpr double kX1 = boys::detail::kX1;

const double kSampleXs[] = {
    0.0, 1e-12, 1e-6, 0.5, 1.25, 7.0, kX0, kX0 + 1.0, 15.0, 23.0, kX1, 30.0, 60.0, 100.0, 1e6};
constexpr int kSampleOrders[] = {0, 1, 2, 3, 7, 16, 31, 32};

double RefSingle(int n, double x) {
    double value = 0.0;
    EXPECT_EQ(BoysDouble(n, x, &value), BOYS_SUCCESS);
    return value;
}

TEST(BoysCTest, DoubleMatchesCppLane) {
    for (int n : kSampleOrders)
    {
        for (double x : kSampleXs)
        {
            EXPECT_DOUBLE_EQ(RefSingle(n, x), boys::BoysSingle(n, x)) << "n=" << n << " x=" << x;
        }
    }
}

TEST(BoysCTest, FloatMatchesCppLane) {
    for (int n : kSampleOrders)
    {
        for (double x : kSampleXs)
        {
            const float xf = static_cast<float>(x);
            float value = 0.0f;
            ASSERT_EQ(BoysFloat(n, xf, &value), BOYS_SUCCESS);
            EXPECT_FLOAT_EQ(value, boys::BoysSingleF32(n, xf)) << "n=" << n << " x=" << x;
        }
    }
}

// The multiplier dispatch is exact over the sampled set; each relaxed entry
// must be bit-identical to the direct C++ instantiation at that m.
template <double kM> void CheckMultiplierLane(int n, double x) {
    double viaC = 0.0;
    ASSERT_EQ(BoysDoubleWithMultiplier(kM, n, x, &viaC), BOYS_SUCCESS);
    EXPECT_DOUBLE_EQ(viaC, boys::BoysSingle<kM>(n, x)) << "m=" << kM << " n=" << n << " x=" << x;
}

TEST(BoysCTest, DoubleMultiplierDispatchMatchesCpp) {
    for (int n : kSampleOrders)
    {
        for (double x : kSampleXs)
        {
            CheckMultiplierLane<1.0>(n, x);
            CheckMultiplierLane<2.0>(n, x);
            CheckMultiplierLane<10.0>(n, x);
            CheckMultiplierLane<100.0>(n, x);
            CheckMultiplierLane<1e4>(n, x);
            CheckMultiplierLane<1e8>(n, x);
        }
    }
}

TEST(BoysCTest, FloatMultiplierDispatchMatchesCpp) {
    for (int n : kSampleOrders)
    {
        for (double x : kSampleXs)
        {
            const float xf = static_cast<float>(x);
            float viaC = 0.0f;
            ASSERT_EQ(BoysFloatWithMultiplier(1.0, n, xf, &viaC), BOYS_SUCCESS);
            EXPECT_FLOAT_EQ(viaC, boys::BoysSingleF32(n, xf));
            ASSERT_EQ(BoysFloatWithMultiplier(2.0, n, xf, &viaC), BOYS_SUCCESS);
            EXPECT_FLOAT_EQ(viaC, boys::BoysSingleF32<2.0>(n, xf));
            ASSERT_EQ(BoysFloatWithMultiplier(1e8, n, xf, &viaC), BOYS_SUCCESS);
            EXPECT_FLOAT_EQ(viaC, boys::BoysSingleF32<1e8>(n, xf));
        }
    }
}

TEST(BoysCTest, DoubleBatchMatchesCppPerElement) {
    // Layout: out[k * count + i] = F_k(x[i]), checked against the C++ batch
    // lane per element. Exercises x spanning all three regions and x = 0.
    std::vector<double> xs;
    xs.push_back(0.0);
    xs.push_back(1e-9);
    xs.push_back(0.3);
    xs.push_back(kX0 - 0.5);
    xs.push_back(kX0);
    xs.push_back(14.5);
    xs.push_back(kX1);
    xs.push_back(kX1 + 5.0);
    xs.push_back(1e4);

    const int nmax = 32;
    std::vector<double> out(static_cast<std::size_t>(xs.size()) * (nmax + 1));
    ASSERT_EQ(BoysDoubleBatch(nmax, static_cast<int>(xs.size()), xs.data(), out.data()),
              BOYS_SUCCESS);

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        std::vector<double> row(nmax + 1);
        boys::BoysBatch(nmax, xs[i], row.data());

        for (int k = 0; k <= nmax; ++k)
        {
            EXPECT_DOUBLE_EQ(out[static_cast<std::size_t>(k) * xs.size() + i],
                             row[static_cast<std::size_t>(k)])
                << "i=" << i << " k=" << k << " x=" << xs[i];
        }
    }
}

TEST(BoysCTest, FloatBatchMatchesCppPerElement) {
    const std::vector<float> xs = {
        0.0f, 0.3f, 7.5f, static_cast<float>(kX0), static_cast<float>(kX1), 1e4f};
    const int nmax = 16;
    std::vector<float> out(static_cast<std::size_t>(xs.size()) * (nmax + 1));
    ASSERT_EQ(BoysFloatBatch(nmax, static_cast<int>(xs.size()), xs.data(), out.data()),
              BOYS_SUCCESS);

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        std::vector<float> row(nmax + 1);
        boys::BoysBatchF32(nmax, xs[i], row.data());

        for (int k = 0; k <= nmax; ++k)
        {
            EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(k) * xs.size() + i],
                            row[static_cast<std::size_t>(k)])
                << "i=" << i << " k=" << k << " x=" << xs[i];
        }
    }
}

TEST(BoysCTest, RejectsInvalidArguments) {
    double value = 0.0;
    float fvalue = 0.0f;
    const double nan = std::numeric_limits<double>::quiet_NaN();

    EXPECT_EQ(BoysDouble(-1, 0.5, &value), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDouble(boys::kMaxBoysOrder + 1, 0.5, &value), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDouble(0, -1.0, &value), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDouble(0, nan, &value), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDouble(0, 0.5, nullptr), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysFloat(0, -1.0f, &fvalue), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysFloat(0, 0.5f, nullptr), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatch(-1, 4, nullptr, nullptr), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatch(boys::kMaxBoysOrder + 1, 4, nullptr, nullptr),
              BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatch(0, -1, nullptr, nullptr), BOYS_ERROR_INVALID_ARGUMENT);

    double single = 0.0;
    EXPECT_EQ(BoysDoubleBatch(0, 1, nullptr, &single), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatch(0, 1, &single, nullptr), BOYS_ERROR_INVALID_ARGUMENT);
    const double negative = -0.5;
    EXPECT_EQ(BoysDoubleBatch(0, 1, &negative, &single), BOYS_ERROR_INVALID_ARGUMENT);
    const double nanX = nan;
    EXPECT_EQ(BoysDoubleBatch(0, 1, &nanX, &single), BOYS_ERROR_INVALID_ARGUMENT);

    // Zero count is a no-op, even with null pointers.
    EXPECT_EQ(BoysDoubleBatch(32, 0, nullptr, nullptr), BOYS_SUCCESS);
    EXPECT_EQ(BoysFloatBatch(32, 0, nullptr, nullptr), BOYS_SUCCESS);
}

TEST(BoysCTest, RejectsUnsupportedMultipliers) {
    double value = 0.0;
    float fvalue = 0.0f;
    EXPECT_EQ(BoysDoubleWithMultiplier(0.5, 0, 0.5, &value), BOYS_ERROR_UNSUPPORTED_MULTIPLIER);
    EXPECT_EQ(BoysDoubleWithMultiplier(3.0, 0, 0.5, &value), BOYS_ERROR_UNSUPPORTED_MULTIPLIER);
    EXPECT_EQ(BoysDoubleWithMultiplier(-1.0, 0, 0.5, &value), BOYS_ERROR_UNSUPPORTED_MULTIPLIER);
    EXPECT_EQ(BoysFloatWithMultiplier(1.5, 0, 0.5f, &fvalue), BOYS_ERROR_UNSUPPORTED_MULTIPLIER);
    // Argument validation precedes the multiplier dispatch.
    EXPECT_EQ(BoysDoubleWithMultiplier(1.5, -1, 0.5, &value), BOYS_ERROR_INVALID_ARGUMENT);
}

} // namespace
