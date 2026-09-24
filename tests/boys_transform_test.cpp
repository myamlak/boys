// The region-A transform lane (boys/boys_transform.hpp): F_0..F_nmax for a
// batch of arguments in one band, by one matrix product in the named mode's
// arithmetic.
//
// The suite measures every mode's delivered error against the committed
// 45-digit reference grid and holds it to that mode's asserted bound (the
// public header's table). It reports each mode's worst alongside the bound, so
// a change that eats the margin is visible rather than silent.

#include "boys/boys.hpp"
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
// The single-pass narrow modes' floors, from the same table. They are three
// orders looser than the split modes' and the header says so: one product per
// degree at an 11- or 8-bit operand is what a card computes without a split,
// and what the split exists to avoid.
constexpr double kTf32Bound = 5e-4;
constexpr double kBf16Bound = 3e-3;
constexpr double kFp16Bound = 5e-4;

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

// The single-pass narrow modes, held to their own floors in the same sweep. The
// floors are three orders looser than the split modes' and are what the modes
// cost: one product per degree instead of three or six. The mode is offered at
// that error, not at the split's, and the header's table says which is which -
// so a row here that read like the split modes' would be the failure.
TEST(BoysTransform, DeliveredErrorPerSinglePassMode) {
    for (boys::RegionABand band : {boys::RegionABand::kA1, boys::RegionABand::kA2})
    {
        const Worst tf32 = SweepBand<boys::ProductMode::kTf32>(band);
        const Worst bf16 = SweepBand<boys::ProductMode::kBf16>(band);
        const Worst fp16 = SweepBand<boys::ProductMode::kFp16>(band);

        std::cout << Report("kTf32", band, tf32, kTf32Bound) << "\n"
                  << Report("kBf16", band, bf16, kBf16Bound) << "\n"
                  << Report("kFp16", band, fp16, kFp16Bound) << std::endl;

        EXPECT_LE(tf32.error, kTf32Bound);
        EXPECT_LE(bf16.error, kBf16Bound);
        EXPECT_LE(fp16.error, kFp16Bound);
    }
}

// tf32 and fp16 carry the same significand, and the band's operands are all
// well inside binary16's exponent range, so the two modes are the same
// arithmetic here and must return the same bits. The header claims the
// coincidence in prose; this is the artifact behind it. A difference means
// either the operand rounding or the accumulate format stopped being the same,
// and either would make one of the two published floors wrong.
TEST(BoysTransform, Tf32AndFp16CoincideOnRegionA) {
    for (boys::RegionABand band : {boys::RegionABand::kA1, boys::RegionABand::kA2})
    {
        const std::vector<double> xs = BandArguments(band);
        ASSERT_FALSE(xs.empty());

        constexpr int kNmax = boys::kMaxBoysOrder;
        std::vector<double> tf32(xs.size() * (kNmax + 1));
        std::vector<double> fp16(tf32.size());

        boys::BoysRegionAProduct<boys::ProductMode::kTf32>(
            band, kNmax, xs.data(), tf32.data(), xs.size());
        boys::BoysRegionAProduct<boys::ProductMode::kFp16>(
            band, kNmax, xs.data(), fp16.data(), xs.size());

        EXPECT_EQ(tf32, fp16);
    }
}

// The mode report a consumer reads, held to the header's table and to the
// enumeration: one row per enumerator, in order, with the delivered figure and
// the floor the header publishes, and the certification column saying which
// rows a card is held to. The single-pass modes and the split modes are
// uncertified - their bounds are arithmetic on a model of an fp32 accumulator -
// and kFp64 alone is certified, because an fp64 accumulator is an ordinary IEEE
// double sum and that is the arithmetic the row's measurement ran.
TEST(BoysTransform, ModeReportMatchesTheHeaderTable) {
    const std::span<const boys::ProductModeInfo> rows = boys::BoysProductModes();

    ASSERT_EQ(rows.size(), 6u);

    struct Expected {
        boys::ProductMode mode;
        const char* name;
        boys::ModeCertification certification;
        int parts;
        double fitTerm;
        double floor;
        double delivered;
    };

    const Expected expected[] = {
        {boys::ProductMode::kFp64,
         "fp64",
         boys::ModeCertification::kCertified,
         1,
         1e-15,
         0.0,
         1.110e-16},
        {boys::ProductMode::kTf32x3,
         "3xtf32",
         boys::ModeCertification::kUncertified,
         2,
         1e-15,
         2.5e-7,
         1.916e-07},
        {boys::ProductMode::kBf16x6,
         "6xbf16",
         boys::ModeCertification::kUncertified,
         3,
         1e-15,
         2.5e-7,
         1.946e-07},
        {boys::ProductMode::kTf32,
         "tf32",
         boys::ModeCertification::kUncertified,
         1,
         1e-15,
         kTf32Bound,
         4.4184e-04},
        {boys::ProductMode::kBf16,
         "bf16",
         boys::ModeCertification::kUncertified,
         1,
         1e-15,
         kBf16Bound,
         2.7893e-03},
        {boys::ProductMode::kFp16,
         "fp16",
         boys::ModeCertification::kUncertified,
         1,
         1e-15,
         kFp16Bound,
         4.4184e-04},
    };

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(rows[i].mode, expected[i].mode);
        EXPECT_STREQ(rows[i].name, expected[i].name);
        EXPECT_EQ(rows[i].certification, expected[i].certification);
        EXPECT_EQ(rows[i].parts, expected[i].parts);
        EXPECT_EQ(rows[i].fitTerm, expected[i].fitTerm);
        EXPECT_EQ(rows[i].floor, expected[i].floor);
        EXPECT_EQ(rows[i].delivered, expected[i].delivered);
        EXPECT_NE(rows[i].model, nullptr);
    }

    // The rows are the enumeration's order and one per enumerator, so a mode
    // added to the enumeration without a row is caught here rather than by a
    // consumer finding it missing.
    EXPECT_EQ(static_cast<int>(rows.back().mode) + 1, 6);
    EXPECT_EQ(static_cast<int>(boys::ProductMode::kFp16) + 1, 6);

    // Exactly one row is certified, and it is the fp64 one.
    int certified = 0;

    for (const boys::ProductModeInfo& row : rows)
    {
        if (row.certification == boys::ModeCertification::kCertified)
        {
            ++certified;
            EXPECT_EQ(row.mode, boys::ProductMode::kFp64);
        }
    }

    EXPECT_EQ(certified, 1);
}

// The accumulate format is the wall, not the operand: the split modes reuse the
// double lane's table at fp32 precision, and no split reaches the double budget
// however many parts it carries. A split mode that quietly started reaching
// 1e-15 would be a different claim, not a better number.
TEST(BoysTransform, SplitModesDoNotReachTheDoubleBudget) {    const Worst tf32 = SweepBand<boys::ProductMode::kTf32x3>(boys::RegionABand::kA1);
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
    // u stays below 1: the band is half-open, so its right end belongs to the
    // next band and is not an argument of this one.
    for (int i = 1; i < 20001; ++i)
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

// The multiplier's path: the band's fits truncate to one width every order in
// the band admits, and the relaxed product stays inside the same bound. For
// kFp64 the multiplier is live from m = 2 upward, so this is where a relaxed
// width can be seen at all; the two split modes' floors are four orders above
// the fit term and no documented multiplier moves them.
TEST(BoysTransform, RelaxedMultiplierStaysInsideTheBound) {
    constexpr double kMultiplier = 1024.0;
    const std::vector<double> xs = BandArguments(boys::RegionABand::kA1);
    constexpr int kNmax = boys::kMaxBoysOrder;

    std::vector<double> out(xs.size() * (kNmax + 1));
    boys::BoysRegionAProduct<boys::ProductMode::kFp64, kMultiplier>(
        boys::RegionABand::kA1, kNmax, xs.data(), out.data(), xs.size());

    double worst = 0.0;

    for (std::size_t i = 0; i < xs.size(); ++i)
    {
        for (int n = 0; n <= kNmax; ++n)
        {
            worst = std::max(worst, std::abs(out[n * xs.size() + i] - Reference(n, xs[i])));
        }
    }

    std::cout << "kFp64 at m = " << kMultiplier << ": worst " << worst << " (bound "
              << kMultiplier * kFp64Bound << ")" << std::endl;
    EXPECT_LE(worst, kMultiplier * kFp64Bound);

    // The relaxed width is a compile-time fact of the instantiation, not a
    // runtime switch: the two instantiations are different code.
    EXPECT_LT(boys::detail::BandDegreeAtMultiplier<0>(kMultiplier),
              boys::detail::BandDegreeAtMultiplier<0>(1.0));
}
