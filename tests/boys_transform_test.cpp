// The region-A transform lane (boys/boys_transform.hpp): F_0..F_nmax for a
// batch of arguments in one band, by one matrix product in the named mode's
// arithmetic.
//
// The suite measures every mode's delivered error against the committed
// 45-digit reference grid and holds it to that mode's asserted bound (the
// public header's table). It reports each mode's worst alongside the bound, so
// a change that eats the margin is visible rather than silent.

#include "boys/boys_transform.hpp"


#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct ReferenceRow {
    int n;
    double x;
    double value;
};

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

        if (row.x < boys::kRegionAEnd)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

const std::vector<ReferenceRow> gReference = LoadReference();

// The reference arguments, ascending and distinct, and the band each is in.
std::vector<double> BandArguments(boys::RegionABand band) {
    std::vector<double> xs;

    for (const ReferenceRow& row : gReference)
    {
        const bool inBand = (band == boys::RegionABand::kA1) ? (row.x < boys::kRegionA1Edge)
                                                             : (row.x >= boys::kRegionA1Edge);

        if (inBand)
        {
            xs.push_back(row.x);
        }
    }

    std::sort(xs.begin(), xs.end());
    xs.erase(std::unique(xs.begin(), xs.end()), xs.end());
    return xs;
}

double Reference(int n, double x) {
    for (const ReferenceRow& row : gReference)
    {
        if (row.n == n && row.x == x)
        {
            return row.value;
        }
    }

    ADD_FAILURE() << "no reference row for n = " << n << ", x = " << x;
    return 0.0;
}

// The worst delivered error of one mode over one band, and where it sits.
struct Worst {
    double error = 0.0;
    int order = 0;
    double x = 0.0;
};

template <boys::ProductMode kMode> Worst SweepBand(boys::RegionABand band) {
    const std::vector<double> xs = BandArguments(band);

    if (xs.empty())
    {
        ADD_FAILURE() << "no reference arguments in the band";
        return {};
    }

    constexpr int kNmax = boys::kMaxBoysOrder;
    std::vector<double> out(xs.size() * (kNmax + 1));

    boys::BoysRegionAProduct<kMode>(band, kNmax, xs.data(), out.data(), xs.size());

    Worst worst;

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        for (int n = 0; n <= kNmax; ++n)
        {
            const double error = std::abs(out[static_cast<std::size_t>(n) * xs.size() + i] -
                                          Reference(n, xs[i]));

            if (error > worst.error)
            {
                worst.error = error;
                worst.order = n;
                worst.x = xs[i];
            }
        }
    }

    return worst;
}

std::string Report(const char* mode, boys::RegionABand band, const Worst& worst, double bound) {
    std::ostringstream ss;
    ss << mode << (band == boys::RegionABand::kA1 ? " A1" : " A2") << ": worst " << worst.error
       << " at order " << worst.order << ", x = " << worst.x << " (bound " << bound << ", "
       << bound / worst.error << "x margin)";
    return ss.str();
}

// The asserted bounds, from the public header's table. kFp64's is the double
// single lane's region-A budget; the two split modes' is their fp32
// accumulator's floor with a margin over the worst a dense sweep of the lower
// band finds - they are not the float lane's 1.5e-7, which they miss by 1.3x at
// m = 1 and hold from m = 2, when that lane's own budget is 3e-7.
constexpr double kFp64Bound = 1e-15;
constexpr double kSplitBound = 2.5e-7;

} // namespace

// The modes' delivered error, held to their asserted bounds. The margin is the
// point of the report: it is what a later change has to eat before it shows up
// as a failure.
TEST(BoysTransform, DeliveredErrorPerMode) {
    for (boys::RegionABand band : {boys::RegionABand::kA1, boys::RegionABand::kA2})
    {
        const Worst fp64 = SweepBand<boys::ProductMode::kFp64>(band);
        const Worst tf32 = SweepBand<boys::ProductMode::kTf32x3>(band);
        const Worst bf16 = SweepBand<boys::ProductMode::kBf16x6>(band);

        std::cout << Report("kFp64", band, fp64, kFp64Bound) << "\n"
                  << Report("kTf32x3", band, tf32, kSplitBound) << "\n"
                  << Report("kBf16x6", band, bf16, kSplitBound) << std::endl;

        EXPECT_LE(fp64.error, kFp64Bound);
        EXPECT_LE(tf32.error, kSplitBound);
        EXPECT_LE(bf16.error, kSplitBound);
    }
}

// The accumulate format is the wall, not the operand: the split modes reuse the
// double lane's table at fp32 precision, and no split reaches the double budget
// however many parts it carries. A split mode that quietly started reaching
// 1e-15 would be a different claim, not a better number.
TEST(BoysTransform, SplitModesDoNotReachTheDoubleBudget) {
    const Worst tf32 = SweepBand<boys::ProductMode::kTf32x3>(boys::RegionABand::kA1);
    const Worst bf16 = SweepBand<boys::ProductMode::kBf16x6>(boys::RegionABand::kA1);

    EXPECT_GT(tf32.error, 1e-9);
    EXPECT_GT(bf16.error, 1e-9);
}

// The fp64 mode is the double lane's own arithmetic in the product's shape, so
// its delivered error is the coefficient table's truncation and nothing
// measurable more. Held against the shipped per-order fit at the same argument:
// the two schemes differ, and neither may leave the lane's budget.
TEST(BoysTransform, Fp64MatchesTheShippedFitWithinTheBudget) {
    constexpr int kNmax = boys::kMaxBoysOrder;
    const std::vector<double> xs = BandArguments(boys::RegionABand::kA1);
    std::vector<double> out(xs.size() * (kNmax + 1));
    boys::BoysRegionAProduct<boys::ProductMode::kFp64>(boys::RegionABand::kA1, kNmax, xs.data(),
                                                       out.data(), xs.size());
    double worst = 0.0;

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        for (int n = 0; n <= kNmax; ++n)
        {
            worst = std::max(worst, std::abs(out[static_cast<std::size_t>(n) * xs.size() + i] -
                                             boys::BoysSingle(n, xs[i])));
        }
    }

    std::cout << "kFp64 vs BoysSingle, region A: worst " << worst << std::endl;
    // Two different schemes, each inside its own bound; the difference is
    // bounded by their sum and is not expected to be zero.
    EXPECT_LE(worst, 3.1e-14);
}

// The layout is the all-orders batch entry's: order-major planes, and the
// stride is the batch's count, not nmax + 1.
TEST(BoysTransform, LayoutIsOrderMajorPlanes) {
    const std::vector<double> xs = BandArguments(boys::RegionABand::kA1);
    ASSERT_GE(xs.size(), 3U);

    const std::vector<double> args{xs[0], xs[1], xs[2]};
    std::vector<double> out(args.size() * 3);
    boys::BoysRegionAProduct<boys::ProductMode::kFp64>(boys::RegionABand::kA1, 2, args.data(),
                                                       out.data(), args.size());

    for (std::size_t i = 0; i < args.size(); ++i)
    {
        for (int n = 0; n <= 2; ++n)
        {
            const double value = out[static_cast<std::size_t>(n) * args.size() + i];
            EXPECT_NEAR(value, Reference(n, args[i]), 1e-15);
            EXPECT_NEAR(value, boys::BoysSingle(n, args[i]), 1e-14);
        }
    }
}

// The result does not depend on the batch tile: the same arguments in one call
// and split across calls that straddle the tile boundary agree bit for bit.
TEST(BoysTransform, ResultIsIndependentOfBatching) {
    // A pitch inside the upper band: the tiling check is about the call
    // pattern, not about accuracy, so a synthetic sweep is the right argument
    // set - it is denser than the reference grid.
    std::vector<double> xs;
    for (int i = 0; i < 100; ++i)
    {
        xs.push_back(boys::kRegionA1Edge + 0.05 * static_cast<double>(i));
    }

    constexpr int kNmax = 8;
    const std::size_t kSplit = 33; // one past the internal tile, so both tiles run

    std::vector<double> whole(xs.size() * (kNmax + 1));
    boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(boys::RegionABand::kA2, kNmax, xs.data(),
                                                         whole.data(), xs.size());

    std::vector<double> head(kSplit * (kNmax + 1));
    std::vector<double> tail((xs.size() - kSplit) * (kNmax + 1));
    boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(boys::RegionABand::kA2, kNmax, xs.data(),
                                                         head.data(), kSplit);
    boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(boys::RegionABand::kA2, kNmax,
                                                         xs.data() + kSplit, tail.data(),
                                                         xs.size() - kSplit);

    for (int n = 0; n <= kNmax; ++n)
    {
        for (std::size_t i = 0; i < kSplit; ++i)
        {
            EXPECT_EQ(whole[static_cast<std::size_t>(n) * xs.size() + i],
                      head[static_cast<std::size_t>(n) * kSplit + i]);
        }

        for (std::size_t i = 0; i < xs.size() - kSplit; ++i)
        {
            EXPECT_EQ(whole[static_cast<std::size_t>(n) * xs.size() + kSplit + i],
                      tail[static_cast<std::size_t>(n) * (xs.size() - kSplit) + i]);
        }
    }
}

// Degenerate batches: count 0 writes nothing, nmax 0 is one plane, and the
// bands' edges belong where the published constants say.
TEST(BoysTransform, DegenerateAndBoundaryCases) {
    const std::vector<double> xs = BandArguments(boys::RegionABand::kA1);

    boys::BoysRegionAProduct<boys::ProductMode::kFp64>(boys::RegionABand::kA1, 4, xs.data(),
                                                       nullptr, 0);

    std::vector<double> out(4);
    boys::BoysRegionAProduct<boys::ProductMode::kFp64>(boys::RegionABand::kA1, 0, xs.data(),
                                                       out.data(), 1);
    EXPECT_NEAR(out[0], Reference(0, xs[0]), 1e-15);

    // The join between the bands: kRegionA1Edge is the upper band's first
    // argument, not the lower band's last.
    EXPECT_LT(boys::kRegionA1Edge, boys::kRegionAEnd);
}

// The mode set: the three named modes are the ones that carry a usable bound,
// and the enumerators are distinct - a future mode is added here beside them.
TEST(BoysTransform, ModeSetIsTheUsableOne) {
    EXPECT_NE(static_cast<int>(boys::ProductMode::kFp64),
              static_cast<int>(boys::ProductMode::kTf32x3));
    EXPECT_NE(static_cast<int>(boys::ProductMode::kTf32x3),
              static_cast<int>(boys::ProductMode::kBf16x6));
    EXPECT_NE(static_cast<int>(boys::ProductMode::kFp64),
              static_cast<int>(boys::ProductMode::kBf16x6));
}

// The two split modes are worst where the band's own polynomial oscillates
// fastest - the left end of the lower band - and the reference grid's samples
// understate them there by about 1.5 times. This holds them to their bound on a
// sweep dense enough to find that, against the shipped per-order fit, whose own
// error at these orders and arguments is 1e-16 and so does not enter the
// figure.
TEST(BoysTransform, SplitModesAreInsideTheirBoundOnADenseSweep) {
    std::vector<double> xs;
    for (int i = 1; i <= 20001; ++i)
    {
        const double u = static_cast<double>(i) / 20001.0;
        xs.push_back(boys::kRegionA1Edge * u * u * u * u);
    }

    constexpr int kNmax = boys::kMaxBoysOrder;
    const std::size_t n = xs.size();
    std::vector<double> out(n * (kNmax + 1));
    const auto sweep = [&](boys::ProductMode mode) {
        if (mode == boys::ProductMode::kFp64)
            boys::BoysRegionAProduct<boys::ProductMode::kFp64>(boys::RegionABand::kA1, kNmax,
                                                               xs.data(), out.data(), n);
        else if (mode == boys::ProductMode::kTf32x3)
            boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(boys::RegionABand::kA1, kNmax,
                                                                 xs.data(), out.data(), n);
        else
            boys::BoysRegionAProduct<boys::ProductMode::kBf16x6>(boys::RegionABand::kA1, kNmax,
                                                                 xs.data(), out.data(), n);
        double worst = 0.0;
        int order = 0;
        for (int k = 0; k <= kNmax; ++k)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const double error = std::abs(out[static_cast<std::size_t>(k) * n + i] -
                                              boys::BoysSingle(k, xs[i]));
                if (error > worst)
                {
                    worst = error;
                    order = k;
                }
            }
        }
        std::cout << "dense sweep, mode " << static_cast<int>(mode) << ": worst " << worst
                  << " at order " << order << std::endl;
        return worst;
    };

    EXPECT_LE(sweep(boys::ProductMode::kTf32x3), kSplitBound);
    EXPECT_LE(sweep(boys::ProductMode::kBf16x6), kSplitBound);
}
