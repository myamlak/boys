// The C linkage surface contract (boys_c.h): scalar/batch entries match the
// C++ lanes, and the validation rules return the documented status codes.
#include "boys/boys.hpp"
#include "boys/boys_c.h"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <vector>

namespace {

constexpr double kX0 = boys::detail::kX0;
constexpr double kX1 = boys::detail::kX1;

const double kSampleXs[] = {
    0.0, 1e-12, 1e-6, 0.5, 1.25, 7.0, kX0, kX0 + 1.0, 15.0, 23.0, kX1, 30.0, 60.0, 100.0, 1e6};
constexpr int kSampleOrders[] = {0, 1, 2, 3, 7, 16, 31, 32};

// A value no Boys entry returns: F_n(x) is positive and at most 1 for every
// supported order and argument, so a negative marker says "not written".
constexpr double kUnwritten = -1.0;

// Each batch entry below wraps one C++ entry, and that entry resolves through the
// default-policy row of its own class - the row, which is not the seam's five once a
// replacement header moves a class. So a reference asked without naming that class
// reads a different row wherever the two disagree, and the comparison would state an
// identity between two classes instead of the identity between the batch shape and the
// per-argument shape this file is checking. Naming the entry's class on the reference
// side keeps every comparison the statement the entries make: one policy, two shapes.
using DoubleAtOrdersClass =
    boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>;
using FloatBatchClass = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>;

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

TEST(BoysCTest, DoubleBatchMatchesCppPerElement) {
    // Layout: out[k * count + i] = F_k(x[i]), checked against the C++ batch lane
    // per element. The samples span all three regions, plus x = 0.
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
        // Unnamed on both sides, and rightly so: this C entry itself runs the fp64
        // all-orders entry per argument, so the two names compile the same class's row.
        boys::BoysAllOrders(nmax, xs[i], row.data());

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

    // The C entry routes to the library's float all-N batch, so the two agree bit for bit.
    std::vector<float> allN(out.size());
    boys::BoysAllNF32(nmax, xs.data(), allN.data(), xs.size());

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        std::vector<float> row(nmax + 1);
        // At the class the C entry compiles (FloatBatchClass above): the per-argument
        // shape and the batch shape at one policy, which is the identity being checked.
        boys::BoysAllOrdersF32<FloatBatchClass>(nmax, xs[i], row.data());

        for (int k = 0; k <= nmax; ++k)
        {
            EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(k) * xs.size() + i],
                            row[static_cast<std::size_t>(k)])
                << "i=" << i << " k=" << k << " x=" << xs[i];
            EXPECT_FLOAT_EQ(out[static_cast<std::size_t>(k) * xs.size() + i],
                            allN[static_cast<std::size_t>(k) * xs.size() + i])
                << "i=" << i << " k=" << k << " x=" << xs[i];
        }
    }
}

TEST(BoysCTest, DoubleBatchAtOrdersMatchesCppPerElement) {
    // Layout: out[k * count + i] = F_k(x[i]) for k = 0..n[i], the top order taken
    // per argument, over samples in all three regions and top orders at both ends.
    const std::vector<double> xs = {0.0, 1e-9, 0.3, kX0 - 0.5, kX0, 14.5, kX1, kX1 + 5.0, 1e4};
    const std::vector<int> tops = {0, 3, 32, 8, 16, 1, 32, 5, 12};
    const int nmax = 32;
    const std::size_t count = xs.size();
    const std::size_t cells = count * (nmax + 1);

    std::vector<double> viaC(cells, kUnwritten);
    ASSERT_EQ(BoysDoubleBatchAtOrders(tops.data(), static_cast<int>(count), xs.data(), viaC.data()),
              BOYS_SUCCESS);

    std::vector<double> viaCpp(cells, kUnwritten);
    boys::BoysAllNAtOrders(tops.data(), xs.data(), viaCpp.data(), count);

    for (std::size_t i = 0; i < count; ++i)
    {
        // The reference is this argument's OWN top order, which is what the entry
        // documents - not the batch's nmax: below the first tier threshold an
        // all-orders body seeds its downward recursion at the order it is called.
        std::vector<double> row(static_cast<std::size_t>(tops[i]) + 1);
        // At the class the C entry compiles (DoubleAtOrdersClass above): the per-argument
        // shape and the at-orders batch shape at one policy.
        boys::BoysAllOrders<DoubleAtOrdersClass>(tops[i], xs[i], row.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const std::size_t index = static_cast<std::size_t>(k) * count + i;

            // Documented: a cell above the argument's own top order is left as
            // the caller left it, by both surfaces.
            if (k > tops[i])
            {
                EXPECT_EQ(viaC[index], kUnwritten) << "i=" << i << " k=" << k;
                EXPECT_EQ(viaCpp[index], kUnwritten) << "i=" << i << " k=" << k;
                continue;
            }

            EXPECT_DOUBLE_EQ(viaC[index], row[static_cast<std::size_t>(k)])
                << "i=" << i << " k=" << k << " x=" << xs[i];
            EXPECT_DOUBLE_EQ(viaC[index], viaCpp[index]) << "i=" << i << " k=" << k;
        }
    }
}

TEST(BoysCTest, DoubleBatchAtOrdersRejectsTheWholeBatchBeforeWriting) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> xs = {0.5, 0.5, 0.5};
    const std::size_t cells = xs.size() * (boys::kMaxBoysOrder + 1);

    // A rejected batch leaves the caller's buffer as it found it, at any position.
    const auto rejects = [&](const std::vector<int>& tops, const std::vector<double>& args) {
        std::vector<double> out(cells, kUnwritten);
        EXPECT_EQ(BoysDoubleBatchAtOrders(
                      tops.data(), static_cast<int>(args.size()), args.data(), out.data()),
                  BOYS_ERROR_INVALID_ARGUMENT);

        for (const double v : out)
        {
            EXPECT_EQ(v, kUnwritten);
        }
    };

    rejects({1, boys::kMaxBoysOrder + 1, 1}, xs);
    rejects({1, -1, 1}, xs);
    rejects({1, 1, 1}, {0.5, -0.5, 0.5});
    rejects({1, 1, 1}, {0.5, nan, 0.5});

    std::vector<double> out(cells, kUnwritten);
    EXPECT_EQ(BoysDoubleBatchAtOrders(nullptr, 3, xs.data(), out.data()),
              BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatchAtOrders(nullptr, -1, nullptr, nullptr), BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatchAtOrders(nullptr, 3, nullptr, out.data()),
              BOYS_ERROR_INVALID_ARGUMENT);
    EXPECT_EQ(BoysDoubleBatchAtOrders(nullptr, 3, xs.data(), nullptr), BOYS_ERROR_INVALID_ARGUMENT);

    // Documented: count may be 0, and then nothing is written.
    std::vector<double> untouched(2, kUnwritten);
    EXPECT_EQ(BoysDoubleBatchAtOrders(nullptr, 0, nullptr, untouched.data()), BOYS_SUCCESS);
    EXPECT_EQ(untouched[0], kUnwritten);
    EXPECT_EQ(untouched[1], kUnwritten);
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

} // namespace
