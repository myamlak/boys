// The fixed-n vector entry (BoysFixedN) contract tests: F_n(x[i]) over an
// array of arguments at one fixed order, the batch shape of
// angular-momentum-grouped integral-engine inner loops.
//
// The entry's per-element contract is the scalar single lane's: each
// element runs the BoysSingle region bodies verbatim (m = 1: the
// certified path, bit-identical by construction; m > 1: the same bodies
// at the single-lane effective degrees), so the accuracy assertions reuse
// the double-single per-region bounds (1e-15 / 3e-14 / 5.5e-14) over the
// committed reference grid, and the identity tests assert bitwise
// agreement with BoysSingle at every sampled multiplier. The layout tests
// pin the strided surface (out[i * stride] = F_n(x[i]), stride >= 1 in
// doubles, default 1) and the alignment contract (natural double
// alignment only - the entry is scalar; buffers over-aligned like the
// AVX2 lanes' are accepted unchanged).
//
// The sampled-m instantiations compile from the internal headers, like
// the rest of the accuracy suite; the m = 1 call sites below route to the
// library's certified instantiation (extern-template surface in boys.hpp,
// explicit instantiation in boys.cpp).

#include "boys/boys.hpp"
#include "boys_effective_degrees.hpp"
#include "boys_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using boys::BoysFixedN;
using boys::BoysSingle;
using boys::detail::kExtendedBX0;
using boys::detail::kX0;
using boys::detail::kX1;

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid (tools/gen_boys_coefficients.py, 30-digit
// mpmath values rounded to double) - the same loader as the other suites.
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

        if (row.x <= 100.0)
        {
            rows.push_back(row);
        }
    }

    return rows;
}

// The grid grouped per order: every order's own x set and reference values
// (the fixed-n entry is swept one whole x array per order).
const std::vector<ReferenceRow> gReference = LoadReference();

struct GridColumns {
    std::array<std::vector<double>, boys::kMaxBoysOrder + 1> x;
    std::array<std::vector<double>, boys::kMaxBoysOrder + 1> want;
};

GridColumns BuildColumns(const std::vector<ReferenceRow>& rows) {
    GridColumns grid;

    for (const ReferenceRow& row : rows)
    {
        const std::size_t n = static_cast<std::size_t>(row.n);
        grid.x[n].push_back(row.x);
        grid.want[n].push_back(row.value);
    }

    return grid;
}

// Region bucketing per the paper's tab:accuracy caption (x = kX1 rows land
// in region C), the shipped kernel's own boundaries. The extended band
// [kExtendedBX0, kX0) is its own region: the per-range F0 seed + upward
// recursion serves it per kmax tier with the region-B budget.
enum class BoysRegion : std::uint8_t { A, B, C, E };

BoysRegion RegionOf(double x) {
    if (x < kX0)
    {
        return x >= kExtendedBX0 ? BoysRegion::E : BoysRegion::A;
    }

    if (x < kX1)
    {
        return BoysRegion::B;
    }

    return BoysRegion::C;
}

// The double-single per-region bound (the fixed-n entry mirrors the scalar
// single lane per element).
double RegionBound(BoysRegion region) {
    switch (region)
    {
    case BoysRegion::A:
        return 1e-15;

    case BoysRegion::B:
    case BoysRegion::E:
        return 3e-14;

    case BoysRegion::C:
        return 5.5e-14;
    }

    return 0.0; // unreachable
}

// The sampled-m set of the accuracy suite.
template <typename Fn> void ForEachSampledMultiplier(Fn&& fn) {
    fn.template operator()<1.0>();
    fn.template operator()<2.0>();
    fn.template operator()<10.0>();
    fn.template operator()<100.0>();
    fn.template operator()<1e4>();
    fn.template operator()<1e8>();
}

struct RegionWorsts {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double e = 0.0;

    void Update(double error, double x) {
        switch (RegionOf(x))
        {
        case BoysRegion::A:
            a = std::max(a, error);
            break;

        case BoysRegion::B:
            b = std::max(b, error);
            break;

        case BoysRegion::C:
            c = std::max(c, error);
            break;

        case BoysRegion::E:
            e = std::max(e, error);
            break;
        }
    }
};

void PrintWorsts(const char* lane, double m, const RegionWorsts& worst) {
    std::printf("%s m=%.0e: worst region A %.3e, B %.3e, C %.3e, extended band %.3e\n",
                lane,
                m,
                worst.a,
                worst.b,
                worst.c,
                worst.e);
}

// The x sets each order is swept over: the order's own grid x's plus the
// region-boundary and edge probes.
std::vector<double> SweepXOf(const GridColumns& grid, int n) {
    std::vector<double> xs = grid.x[static_cast<std::size_t>(n)];

    const std::array<double, 8> probes = {
        0.0,
        kX0,
        kX1,
        0.5 * kX0,
        2.0 * kX1,
        1.0e-12,
        3.0e1,
        1.0e2,
    };
    xs.insert(xs.end(), probes.begin(), probes.end());
    return xs;
}

// ---------------------------------------------------------------------------
// Grid accuracy: |BoysFixedN value - reference| <= m * B_region per element
// ---------------------------------------------------------------------------

template <double kM> void SweepGridAccuracy(const GridColumns& grid) {
    RegionWorsts worst;
    std::vector<double> out;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        const std::vector<double>& xs = grid.x[static_cast<std::size_t>(n)];
        const std::vector<double>& want = grid.want[static_cast<std::size_t>(n)];
        out.resize(xs.size());
        BoysFixedN<kM>(n, xs.data(), out.data(), xs.size());

        for (std::size_t i = 0; i < xs.size(); ++i)
        {
            const double bound = kM * RegionBound(RegionOf(xs[i]));
            const double error = std::abs(out[i] - want[i]);
            EXPECT_LE(error, bound) << "m=" << kM << " n=" << n << " x=" << xs[i]
                                    << " got=" << out[i] << " want=" << want[i];
            worst.Update(error, xs[i]);
        }
    }

    PrintWorsts("fixed-n vector", kM, worst);
}

// ---------------------------------------------------------------------------
// Identity: bitwise agreement with BoysSingle per element (the region
// bodies are the single lane's verbatim at every m)
// ---------------------------------------------------------------------------

template <double kM> void SweepSingleIdentity(const GridColumns& grid) {
    std::vector<double> out;

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        const std::vector<double> xs = SweepXOf(grid, n);
        out.resize(xs.size());
        BoysFixedN<kM>(n, xs.data(), out.data(), xs.size());

        for (std::size_t i = 0; i < xs.size(); ++i)
        {
            EXPECT_EQ(out[i], BoysSingle<kM>(n, xs[i]))
                << "m=" << kM << " n=" << n << " x=" << xs[i];
        }
    }
}

// ---------------------------------------------------------------------------
// The tests
// ---------------------------------------------------------------------------

TEST(BoysFixedNTest, GridSweepMatchesReferenceAtM1) {
    const GridColumns grid = BuildColumns(gReference);
    SweepGridAccuracy<1.0>(grid);
}

TEST(BoysFixedNTest, GridSweepMatchesReferenceAtSampledMultipliers) {
    const GridColumns grid = BuildColumns(gReference);
    ForEachSampledMultiplier([&grid]<double kM>() {
        if (kM != 1.0)
        {
            SweepGridAccuracy<kM>(grid);
        }
    });
}

TEST(BoysFixedNTest, ElementWiseBitIdentityWithBoysSingle) {
    const GridColumns grid = BuildColumns(gReference);
    ForEachSampledMultiplier([&grid]<double kM>() { SweepSingleIdentity<kM>(grid); });
}

// The strided output layout: out[i * stride] = F_n(x[i]); only the stride
// slots are written (the gaps keep their sentinel), stride = 1 equals the
// default-argument call, count = 0 writes nothing.
TEST(BoysFixedNTest, StridedLayoutWritesOnlyStrideSlots) {
    const GridColumns grid = BuildColumns(gReference);
    const std::vector<double>& xs = grid.x[0];
    ASSERT_FALSE(xs.empty());
    const std::size_t count = xs.size();
    constexpr double kSentinel = 1.2345678901234567e100;

    for (const std::size_t stride :
         {std::size_t{1}, std::size_t{2}, std::size_t{3}, std::size_t{7}, std::size_t{16}})
    {
        std::vector<double> out((count - 1) * stride + 1, kSentinel);
        BoysFixedN(0, xs.data(), out.data(), count, stride);

        for (std::size_t i = 0; i < count; ++i)
        {
            EXPECT_EQ(out[i * stride], BoysSingle(0, xs[i])) << "stride=" << stride << " i=" << i;
        }

        for (std::size_t j = 0; j < out.size(); ++j)
        {
            if (j % stride != 0)
            {
                EXPECT_EQ(out[j], kSentinel) << "stride=" << stride << " gap at " << j;
            }
        }
    }

    // Explicit stride 1 (contiguous) matches the default-argument call.
    std::vector<double> explicitOut(count);
    std::vector<double> defaultOut(count);
    BoysFixedN(0, xs.data(), explicitOut.data(), count, 1);
    BoysFixedN(0, xs.data(), defaultOut.data(), count);

    for (std::size_t i = 0; i < count; ++i)
    {
        EXPECT_EQ(explicitOut[i], defaultOut[i]) << "i=" << i;
    }

    // count = 0 writes nothing for any stride.
    std::vector<double> empty(4, kSentinel);
    BoysFixedN(0, xs.data(), empty.data(), 0, 3);

    for (const double value : empty)
    {
        EXPECT_EQ(value, kSentinel);
    }
}

// The alignment contract: natural double alignment is the only requirement;
// x and out placed at 8/16/32/64-byte alignments all deliver the same
// values (the over-aligned placements are what the AVX2 region lanes need,
// so one buffer can serve both surfaces).
TEST(BoysFixedNTest, AlignmentContractHoldsFromNaturalUp) {
    const GridColumns grid = BuildColumns(gReference);
    alignas(64) std::array<double, 512> xStore{};
    alignas(64) std::array<double, 512> outStore{};

    for (int n = 0; n <= boys::kMaxBoysOrder; n += 8)
    {
        const std::vector<double>& xs = grid.x[static_cast<std::size_t>(n)];
        ASSERT_LE(xs.size() + 8, xStore.size());

        for (std::size_t i = 0; i < xs.size(); ++i)
        {
            xStore[i] = xs[i];
        }

        for (const std::size_t xOffset : {std::size_t{0}, std::size_t{1}})
        {
            for (const std::size_t outOffset :
                 {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{4}})
            {
                BoysFixedN(n, xStore.data() + xOffset, outStore.data() + outOffset, xs.size());

                // The value each slot received is F_n at the shifted input
                // xStore[xOffset + i] (xStore is zero-initialized, so the
                // tail slot the xOffset = 1 pass reads is 0.0, not garbage).
                for (std::size_t i = 0; i < xs.size(); ++i)
                {
                    EXPECT_EQ(outStore[outOffset + i], BoysSingle(n, xStore[xOffset + i]))
                        << "n=" << n << " xOff=" << xOffset << " outOff=" << outOffset
                        << " x=" << xStore[xOffset + i];
                }
            }
        }
    }
}

} // namespace
