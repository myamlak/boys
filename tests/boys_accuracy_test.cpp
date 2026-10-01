// The accuracy-parametrization contract tests.
//
// Sampled-m per-region contract: for m in {1, 2, 10, 1e2, 1e4, 1e8} x {double single,
// double batch, float single, float batch}, assert |F_hat_m - ref| <= m * B_region per
// region on the committed reference grid (region bucketing: A x < kX0, B kX0 <= x < kX1,
// C x >= kX1), with the asserted bounds B: double single 1e-15/3e-14/5.5e-14, double
// batch 5.5e-14 per region, float single 1.5e-7 per region, and float batch the figure
// the lane publishes for the division form the entries divide in - 1.5e-7 at exact
// division and at the refined reciprocal, plus the row's plain-reciprocal term where the
// build compiles that form, which is the term that form spends on the downward ladder the
// batch shape reads. The m = 1 rows re-assert the certified pins.
//
// The effective-degree tables are non-increasing in m (a larger budget may only truncate
// further) over all six lane roles and both regions; the m = 1 tables are the full
// degrees.
//
// The fp16/Bf16 lanes forward the multiplier to the F32 engine (I/O-only wrappers, no
// fp16-specific degree tables): assert |F_hat - F(x16)| <= m * 1e-7 + 1/2 ULP per value
// on the reference grid at sampled m, F(x16) being the certified double lane evaluated at
// the fp16-rounded argument.
//
// The sampled-m instantiations are compiled from the shipped headers (boys/boys_impl.hpp,
// boys/boys_effective_degrees.hpp); the m = 1 call sites below still route to the
// library's certified instantiations, which the extern-template declarations in
// boys/boys.hpp name. The relaxed SIMD lanes consume the same constexpr degree tables as
// their scalar twins (region A: kDoubleSingle; region B: kDoubleBatch) and the same
// Clenshaw recursions with runtime degrees, so the grid contract below pins the mechanism
// the SIMD lanes share.
//
// The file lives alongside boys_test.cpp rather than inside it to keep that file's
// existing tests untouched (same test binary, same contract).

#include "boys/boys.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

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

using boys::BoysAllOrders;
using boys::BoysAllOrdersF32;
using boys::BoysSingle;
using boys::BoysSingleF32;
using boys::detail::BoysRole;
using boys::detail::kExtendedBX0;
using boys::detail::kX0;
using boys::detail::kX1;
using boys::detail::RegionADegrees;
using boys::detail::RegionBDegrees;

struct ReferenceRow {
    int n;
    double x;
    double value;
};

// The committed reference grid (tools/gen_boys_coefficients.py, 45-digit mpmath values of
// F_n at the double in each row's x column) - the same loader as boys_test.cpp.
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
        rows.push_back(row);
    }

    return rows;
}

// The grid holds every (n, x) pair of its x set (the generator emits all orders per x),
// so a batch lane's lookup of F_k(x) for k <= nmax is exact. Indexed once at startup.
struct ReferenceGrid {
    std::array<std::vector<ReferenceRow>, boys::kMaxBoysOrder + 1> byN{};

    explicit ReferenceGrid(const std::vector<ReferenceRow>& rows) {
        for (const ReferenceRow& row : rows)
        {
            byN[static_cast<std::size_t>(row.n)].push_back(row);
        }

        for (auto& v : byN)
        {
            std::sort(v.begin(), v.end(), [](const ReferenceRow& a, const ReferenceRow& b) {
                return a.x < b.x;
            });
        }
    }

    // The committed value F_n(x); present because the grid is complete in n per x.
    double Value(int n, double x) const {
        const std::vector<ReferenceRow>& v = byN[static_cast<std::size_t>(n)];
        const auto it = std::lower_bound(
            v.begin(), v.end(), x, [](const ReferenceRow& row, double xv) { return row.x < xv; });
        EXPECT_TRUE(it != v.end() && it->x == x) << "missing grid pair n=" << n << " x=" << x;
        return it != v.end() ? it->value : 0.0;
    }
};

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

enum class LaneKind : std::uint8_t { kDoubleSingle, kDoubleBatch, kFloatSingle, kFloatBatch };

// The asserted per-region bounds of the m = 1 contract. The extended band carries the
// region-B budget: at m = 1 its per-range F0 seed serves the certified upward recursion
// (the band is the m = 1 lane's; the m > 1 branch keeps the region-A treatment there,
// whose tighter budget implies this cell at every m).
double RegionBound(BoysRegion region, LaneKind lane) {
    switch (lane)
    {
    case LaneKind::kDoubleSingle:

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

        break;

    case LaneKind::kDoubleBatch:
        return 5.5e-14;

    case LaneKind::kFloatSingle:
    case LaneKind::kFloatBatch:
        return 1.5e-7;
    }

    return 0.0; // unreachable
}

// The figure the float lane publishes for the division form the entries below run. Those
// entries name no policy, so the form they divide in is the build's default one, and the
// figure to read is the one the lane publishes for that form: the lane's own contract row
// - the row the README's table, BoysAccuracyGuaranteed and both gates read - plus the term
// the row carries beside its base for the plain reciprocal where that is the form in force.
// The row's own scaling is the multiplier times that sum, this lane's additive term being
// zero, so the sweeps below read the figure the accessor would answer at each sampled m.
//
// It is not the same number as the float base above, and the difference is the point: the
// base is what the lane's two other forms deliver, and the row's term is what the plain
// form spends on the downward ladder, which is inside the batch shape and not inside the
// single one. So the batch sweeps are read at this figure and the single shape at the base.
double FloatLanePublishedFigure() {
    for (const boys::LaneContractInfo& row : boys::BoysLaneContracts())
    {
        if (row.precision == boys::Precision::kFp32)
        {
            return row.bound + (boys::kDefaultDivisionForm == boys::DivisionForm::kPlainReciprocal
                                    ? row.plainAdditive
                                    : 0.0);
        }
    }

    return 0.0;
}

// Runs the callable once per sampled multiplier, the multiplier passed as a compile-time
// constant - the NTTP surface is the contract under test.
template <typename Fn> void ForEachSampledMultiplier(Fn&& fn) {
    fn.template operator()<1.0>();
    fn.template operator()<2.0>();
    fn.template operator()<10.0>();
    fn.template operator()<100.0>();
    fn.template operator()<1e4>();
    fn.template operator()<1e8>();
}

const std::vector<ReferenceRow> gReference = LoadReference();
const ReferenceGrid gGrid(gReference);

struct RegionWorsts {
    double a = 0.0;
    double b = 0.0;
    double c = 0.0;
    double e = 0.0;

    // (error, x): the candidate error and its x, the per-region max accumulator's pair.
    //
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
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

// ---------------------------------------------------------------------------
// The sampled-m grid contract, one templated sweep per lane
// ---------------------------------------------------------------------------

template <double kM> void SweepDoubleSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = BoysSingle<kM>(row.n, row.x);
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single", kM, worst);
}

template <double kM> void SweepDoubleBatch() {
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrders<kM>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch", kM, worst);
}

template <double kM> void SweepFloatSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32<kM>(row.n, static_cast<float>(row.x));
        // The base and not the lane's published figure for the form in force: this shape
        // evaluates one order from one fit and reads no downward step, which is where the
        // row's plain-reciprocal term is spent (FloatLanePublishedFigure above).
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kFloatSingle);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("float single", kM, worst);
}

// The half lanes' engine budget, as the f32 entries now take it: the policy is
// a type, so naming the budget is naming a policy.
using Fp16Budget = boys::EvalPolicy<boys::kDefaultFitRoute,
                                    boys::kDefaultEvalScheme,
                                    boys::BoysBudget::kFp16>;

// The float lane with the half lanes' engine budget named. The budget picks the degree
// table the relaxation truncates to, so it selects nothing at the reference multiplier -
// the lane evaluates its full fits either way - and the promise is the float lane's own
// bound at every multiplier: the option buys fewer coefficients read, not a different bar.
// The m = 1 case is checked against the default entry bit for bit, the row that would catch
// the option having reached the certified lane's arithmetic rather than only its truncation.
template <double kM> void SweepFloatSingleFp16Budget() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32<kM, Fp16Budget>(row.n, static_cast<float>(row.x));
        // The base, for the reason the default entry's sweep above states: the single shape
        // reads no downward step, and the budget the policy names moves no division form.
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kFloatSingle);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);

        // The multiplier is a template parameter, so this is tested as a compile-time
        // condition: MSVC reports a constant conditional expression under /W4, and this
        // tree makes that an error.
        if constexpr (kM == 1.0)
        {
            EXPECT_EQ(value, BoysSingleF32<1.0>(row.n, static_cast<float>(row.x)))
                << "the reference multiplier: naming the budget must not move the certified "
                   "lane, which reads no degree table there";
        }
    }

    PrintWorsts("float single, fp16 budget", kM, worst);
}

template <double kM> void SweepFloatBatchFp16Budget() {
    RegionWorsts worst;
    std::vector<float> batch(boys::kMaxBoysOrder + 1);
    std::vector<float> plain(batch.size());

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32<kM, Fp16Budget>(
            row.n, static_cast<float>(row.x), batch.data());
        BoysAllOrdersF32<kM>(row.n, static_cast<float>(row.x), plain.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            // The lane's published figure for the form in force and not the shape's base:
            // this entry's downward ladder divides in the form the build compiles, and the
            // budget the policy names moves no division form.
            const double bound = kM * FloatLanePublishedFigure();
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound) << "m=" << kM << " batch F" << k << " at x=" << row.x;
            worst.Update(error, row.x);

            if constexpr (kM == 1.0)
            {
                EXPECT_EQ(batch[static_cast<std::size_t>(k)],
                          plain[static_cast<std::size_t>(k)]);
            }
        }
    }

    PrintWorsts("float batch, fp16 budget", kM, worst);
}

template <double kM> void SweepFloatBatch() {    RegionWorsts worst;
    std::vector<float> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32<kM>(row.n, static_cast<float>(row.x), batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            // The lane's published figure for the form in force and not the shape's base:
            // this entry's downward ladder divides in the form the build compiles, so the
            // bar is the one the lane states for that form.
            const double bound = kM * FloatLanePublishedFigure();
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("float batch", kM, worst);
}

// The Horner rung, measured against the committed reference: the same sweep as the default
// scheme's above, over the table the monomial degree rule is certified on. The gate's
// stored-fit rows cover the m = 1 reading; this is the reading a relaxed rung delivers,
// which is the one the truncation decides.
template <double kM> void SweepDoubleBatchHorner() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        boys::BoysAllOrders<kM, Policy>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " horner batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch, horner", kM, worst);
}

// The uniform route against the same committed reference at the same bar: a fixed grid,
// every order from its own coefficients rather than from a seed and a recursion, so what
// this measures is the whole of that route's arithmetic - the interval index, the mapped
// argument, and the block a ladder is read from - and not the fit alone. The last two are
// part of the table as much as its numbers are: a wrong stride reads a correct table
// wrongly, delivering a wrong value no check of the coefficients would catch.
// Both schemes, because the table stores both coefficient forms and only the Horner one had
// ever been swept. That omission was not hypothetical: the table was first fitted at an odd
// degree, which ClenshawSplit cannot read at all (it asserts an even degree and says so), so
// the Clenshaw path asserted in debug and computed silently wrong values in release while a
// bound for it sat published. A scheme the table carries but nothing sweeps is a scheme
// nothing has checked.
template <double kM, boys::EvalScheme kScheme = boys::EvalScheme::kHorner>
void SweepDoubleBatchUniform() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                    kScheme,
                                    boys::BoysBudget::kFloat,
                                    boys::PackAxis::kArguments,
                                    boys::FitGranularity::kUniform>;
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        boys::BoysAllOrders<kM, Policy>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "m=" << kM << " uniform batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts(kScheme == boys::EvalScheme::kHorner ? "double batch, uniform horner"
                                                     : "double batch, uniform clenshaw",
                kM,
                worst);
}

// The uniform route's single-order path, on the same grid and at the same bar - the
// reading where the route is cheapest rather than dearest. It shares the index arithmetic
// with the ladder above through one FlatLocate, so what this separates is the order lookup.
template <double kM, boys::EvalScheme kScheme = boys::EvalScheme::kHorner>
void SweepDoubleSingleUniform() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                    kScheme,
                                    boys::BoysBudget::kFloat,
                                    boys::PackAxis::kArguments,
                                    boys::FitGranularity::kUniform>;
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = boys::BoysSingle<kM, Policy>(row.n, row.x);
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "m=" << kM << " uniform single n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts(kScheme == boys::EvalScheme::kHorner ? "double single, uniform horner"
                                                     : "double single, uniform clenshaw",
                kM,
                worst);
}

template <double kM> void SweepDoubleSingleHorner() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = boys::BoysSingle<kM, Policy>(row.n, row.x);
        const double bound = kM * RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "horner m=" << kM << " n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single, horner", kM, worst);
}

TEST(BoysAccuracyTest, DoubleBatchHornerRungsAgainstTheReferenceGrid) {
    SweepDoubleBatchUniform<1.0>();
    SweepDoubleSingleUniform<1.0>();
    SweepDoubleBatchUniform<1.0, boys::EvalScheme::kSplitClenshaw>();
    SweepDoubleSingleUniform<1.0, boys::EvalScheme::kSplitClenshaw>();
    SweepDoubleBatchHorner<1.0>();
    SweepDoubleBatchHorner<64.0>();
    SweepDoubleBatchHorner<256.0>();
    SweepDoubleBatchHorner<1024.0>();
    SweepDoubleBatchHorner<4096.0>();
    SweepDoubleBatchHorner<16384.0>();
    SweepDoubleBatchHorner<65536.0>();
}

TEST(BoysAccuracyTest, DoubleSingleHornerRungsAgainstTheReferenceGrid) {
    SweepDoubleSingleHorner<1.0>();
    SweepDoubleSingleHorner<64.0>();
    SweepDoubleSingleHorner<256.0>();
    SweepDoubleSingleHorner<1024.0>();
    SweepDoubleSingleHorner<4096.0>();
    SweepDoubleSingleHorner<16384.0>();
    SweepDoubleSingleHorner<65536.0>();
}

TEST(BoysAccuracyTest, DoubleSingleSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepDoubleSingle<kM>(); });
}

TEST(BoysAccuracyTest, DoubleBatchSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepDoubleBatch<kM>(); });
}

TEST(BoysAccuracyTest, FloatSingleSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatSingle<kM>(); });
}

TEST(BoysAccuracyTest, FloatBatchSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatBatch<kM>(); });
}

// The same two sweeps with the half lanes' engine budget named. Kept as their own tests, so
// the default-multiplier sweeps above stay the reading of the certified lane that they were.
TEST(BoysAccuracyTest, FloatSingleSampledMultipliersFp16Budget) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatSingleFp16Budget<kM>(); });
}

TEST(BoysAccuracyTest, FloatBatchSampledMultipliersFp16Budget) {
    ForEachSampledMultiplier([]<double kM>() { SweepFloatBatchFp16Budget<kM>(); });
}

// ---------------------------------------------------------------------------
// The effective degrees: non-increasing in m, full at m = 1, inside
// the evaluator domain {0, 1, 2, 4, 6, ...} at every (m, role, point)
// ---------------------------------------------------------------------------

constexpr bool InEvaluatorDomain(int d) {
    return d == 0 || d == 1 || d == 2 || (d >= 4 && d % 2 == 0);
}

template <BoysRole kRole> void AssertRoleTables() {
    constexpr auto a1 = RegionADegrees<1.0, kRole>();
    constexpr auto b1 = RegionBDegrees<1.0, kRole>();

    for (std::size_t i = 0; i < a1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(a1[i])) << "region-A degree " << a1[i] << " at index " << i;
    }

    for (std::size_t i = 0; i < b1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(b1[i])) << "region-B degree " << b1[i] << " at index " << i;
    }

    // The m = 1 tables are the full degrees: the criterion's budget
    // (m - 1)*B is zero, so only the full degree has an empty tail.
    if constexpr (boys::detail::RoleUsesDoubleTables(kRole))
    {
        for (std::size_t p = 0; p < boys::detail::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], boys::detail::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree at m = 1";
        }
    } else
    {
        for (std::size_t p = 0; p < boys::detail::f32::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], boys::detail::f32::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree at m = 1";
        }
    }

    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], boys::detail::kBDeg)
                << "region-B order " << n << " must keep its full degree at m = 1";
        }
    } else
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], boys::detail::f32::kBDeg)
                << "region-B order " << n << " must keep its full degree at m = 1";
        }
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesAtM1AreFullAndInDomain) {
    AssertRoleTables<BoysRole::kDoubleSingle>();
    AssertRoleTables<BoysRole::kDoubleBatch>();
    AssertRoleTables<BoysRole::kF32Single>();
    AssertRoleTables<BoysRole::kF32Batch>();
    AssertRoleTables<BoysRole::kF32Fp16Single>();
    AssertRoleTables<BoysRole::kF32Fp16Batch>();
}

template <double kMLo, double kMHi, BoysRole kRole> void AssertRoleNonIncreasing() {
    static_assert(kMLo < kMHi);
    constexpr auto loA = RegionADegrees<kMLo, kRole>();
    constexpr auto hiA = RegionADegrees<kMHi, kRole>();
    static_assert(loA.size() == hiA.size());

    for (std::size_t i = 0; i < loA.size(); ++i)
    {
        EXPECT_GE(loA[i], hiA[i]) << "region-A index " << i << " degree must not grow with m ("
                                  << kMLo << " -> " << kMHi << ")";
    }

    constexpr auto loB = RegionBDegrees<kMLo, kRole>();
    constexpr auto hiB = RegionBDegrees<kMHi, kRole>();

    for (std::size_t i = 0; i < loB.size(); ++i)
    {
        EXPECT_GE(loB[i], hiB[i]) << "region-B index " << i << " degree must not grow with m ("
                                  << kMLo << " -> " << kMHi << ")";
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesNonIncreasingInMultiplier) {
    // Non-increasing in m over every pair of sampled multipliers and every role.
    ForEachSampledMultiplier([]<double kMLo>() {
        ForEachSampledMultiplier([]<double kMHi>() {
            if constexpr (kMLo < kMHi)
            {
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kDoubleSingle>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kDoubleBatch>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Single>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Batch>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Fp16Single>();
                AssertRoleNonIncreasing<kMLo, kMHi, BoysRole::kF32Fp16Batch>();
            }
        });
    });
}

// ---------------------------------------------------------------------------
// A degree table is certified against the table its scheme sums
// ---------------------------------------------------------------------------
// A rung truncates one stored table, and what the truncation costs is the 1-norm of the
// coefficients above d' *in that table*: |T_k(t)| <= 1 and |t^k| <= 1 over the mapped
// interval, so either basis bounds its own dropped series, and the two tables hold
// different numbers for the same fit. A degree chosen on one table's tail and spent on the
// other's is not certified at all - its dropped tail is whatever the other table holds,
// which for the monomial end of a Chebyshev fit runs orders of magnitude larger. The rows
// below are that statement as a measurement: for every shipped rung, piece and order,
// either the entry is the full degree (it drops nothing) or the tail of the table the basis
// names, times the path's amplification, is inside the rung's budget.
// ---------------------------------------------------------------------------

using boys::detail::RegionAAmplification;
using boys::detail::RegionABudget;
using boys::detail::RegionBAmplification;
using boys::detail::RegionBBudget;
using boys::detail::RoleUsesBatchAmplification;
using boys::detail::TailBasis;

/// The region-A coefficient table a basis names.
template <TailBasis kBasis> constexpr const auto& BasisTableA() {
    if constexpr (kBasis == TailBasis::kMonomial)
    {
        return boys::detail::kMonoCoeffs;
    } else
    {
        return boys::detail::kCoeffs;
    }
}

/// The region-B coefficient table a basis names.
template <TailBasis kBasis> constexpr const auto& BasisTableB() {
    if constexpr (kBasis == TailBasis::kMonomial)
    {
        return boys::detail::kMonoBcoeffs;
    } else
    {
        return boys::detail::kBcoeffs;
    }
}

template <double kM, BoysRole kRole, TailBasis kBasis> void AssertBasisCertifiedA() {
    constexpr auto degrees = RegionADegrees<kM, kRole, kBasis>();
    constexpr const auto& table = BasisTableA<kBasis>();
    const double budget = (kM - 1.0) * RegionABudget(kRole);

    for (int order = 0; order <= boys::kMaxBoysOrder; ++order)
    {
        for (int p = boys::detail::kPieceStart[order]; p < boys::detail::kPieceStart[order + 1];
             ++p)
        {
            const auto& piece = boys::detail::kPieces[static_cast<std::size_t>(p)];
            const int d = degrees[static_cast<std::size_t>(p)];
            EXPECT_TRUE(InEvaluatorDomain(d)) << "region-A piece " << p << " at m = " << kM;
            EXPECT_LE(d, piece.deg) << "region-A piece " << p << " at m = " << kM;

            if (d == piece.deg)
            {
                continue; // the fallback drops nothing
            }

            const double amplification =
                RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
            EXPECT_LE(boys::detail::CoefficientTail(
                          table, static_cast<std::size_t>(piece.offset), piece.deg, d) *
                          amplification,
                      budget)
                << "region-A piece " << p << " (order " << order << ", b = " << piece.b
                << ") at m = " << kM << " truncates to " << d << ", and the tail of the "
                << (kBasis == TailBasis::kMonomial ? "monomial" : "chebyshev")
                << " table above it does not fit the budget";
        }
    }
}

template <double kM, BoysRole kRole, TailBasis kBasis> void AssertBasisCertifiedB() {
    constexpr auto degrees = RegionBDegrees<kM, kRole, kBasis>();
    constexpr const auto& table = BasisTableB<kBasis>();
    const double budget = (kM - 1.0) * RegionBBudget(kRole);

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        const int d = degrees[static_cast<std::size_t>(n)];
        EXPECT_TRUE(InEvaluatorDomain(d)) << "region-B order " << n << " at m = " << kM;
        EXPECT_LE(d, boys::detail::kBDeg) << "region-B order " << n << " at m = " << kM;

        if (d == boys::detail::kBDeg)
        {
            continue; // the fallback drops nothing
        }

        EXPECT_LE(boys::detail::CoefficientTail(table, 0, boys::detail::kBDeg, d) *
                      RegionBAmplification(n),
                  budget)
            << "region-B order " << n << " at m = " << kM << " truncates to " << d
            << ", and the tail of the "
            << (kBasis == TailBasis::kMonomial ? "monomial" : "chebyshev")
            << " table above it does not fit the budget (amplification " << RegionBAmplification(n)
            << ")";
    }
}

template <double kM> void AssertEveryRungCertified() {
    AssertBasisCertifiedA<kM, BoysRole::kDoubleSingle, TailBasis::kChebyshev>();
    AssertBasisCertifiedA<kM, BoysRole::kDoubleSingle, TailBasis::kMonomial>();
    AssertBasisCertifiedA<kM, BoysRole::kDoubleBatch, TailBasis::kChebyshev>();
    AssertBasisCertifiedA<kM, BoysRole::kDoubleBatch, TailBasis::kMonomial>();
    AssertBasisCertifiedB<kM, BoysRole::kDoubleSingle, TailBasis::kChebyshev>();
    AssertBasisCertifiedB<kM, BoysRole::kDoubleSingle, TailBasis::kMonomial>();
    AssertBasisCertifiedB<kM, BoysRole::kDoubleBatch, TailBasis::kChebyshev>();
    AssertBasisCertifiedB<kM, BoysRole::kDoubleBatch, TailBasis::kMonomial>();
}

TEST(BoysAccuracyTest, EveryRungDegreesFitTheTableItsSchemeSums) {
    AssertEveryRungCertified<64.0>();
    AssertEveryRungCertified<256.0>();
    AssertEveryRungCertified<1024.0>();
    AssertEveryRungCertified<4096.0>();
    AssertEveryRungCertified<16384.0>();
    AssertEveryRungCertified<65536.0>();
}

TEST(BoysAccuracyTest, BasisTablesAreNotInterchangeable) {
    // The carriage of the basis through the degree tables: a rung's region-A and region-B
    // tables must differ from the other basis' at m = 64, and both bases must give the full
    // degrees at m = 1. A choice of basis that reached no table - one table shared by both,
    // or a rule that always read the Chebyshev coefficients - would leave the Horner rung
    // uncertified while every number above it stayed green.
    constexpr auto chebA = RegionADegrees<64.0, BoysRole::kDoubleSingle, TailBasis::kChebyshev>();
    constexpr auto monoA = RegionADegrees<64.0, BoysRole::kDoubleSingle, TailBasis::kMonomial>();
    constexpr auto chebA1 = RegionADegrees<1.0, BoysRole::kDoubleSingle, TailBasis::kChebyshev>();
    constexpr auto monoA1 = RegionADegrees<1.0, BoysRole::kDoubleSingle, TailBasis::kMonomial>();
    constexpr auto chebB = RegionBDegrees<64.0, BoysRole::kDoubleBatch, TailBasis::kChebyshev>();
    constexpr auto monoB = RegionBDegrees<64.0, BoysRole::kDoubleBatch, TailBasis::kMonomial>();
    constexpr auto chebB1 = RegionBDegrees<1.0, BoysRole::kDoubleBatch, TailBasis::kChebyshev>();
    constexpr auto monoB1 = RegionBDegrees<1.0, BoysRole::kDoubleBatch, TailBasis::kMonomial>();

    ASSERT_EQ(chebA.size(), monoA.size());
    ASSERT_EQ(chebB.size(), monoB.size());

    std::size_t differA = 0;
    std::size_t differB = 0;

    for (std::size_t i = 0; i < chebA.size(); ++i)
    {
        differA += (chebA[i] != monoA[i]) ? 1U : 0U;
        EXPECT_EQ(chebA1[i], monoA1[i])
            << "m = 1 region-A index " << i << " must be the full degree in either basis";
    }

    for (std::size_t i = 0; i < chebB.size(); ++i)
    {
        differB += (chebB[i] != monoB[i]) ? 1U : 0U;
        EXPECT_EQ(chebB1[i], monoB1[i])
            << "m = 1 region-B index " << i << " must be the full degree in either basis";
    }

    EXPECT_GT(differA, 0U) << "at m = 64 the two region-A bases must not return one table";
    EXPECT_GT(differB, 0U) << "at m = 64 the two region-B bases must not return one table";
}

// ---------------------------------------------------------------------------
// Delivered-path consistency: the relaxed single and batch lanes (separate
// arithmetic paths — per-order fits vs. seed + recursion) must agree within
// the sum of their m-scaled region bounds, and the exact x = 0 values must
// survive at every sampled m.
// ---------------------------------------------------------------------------

template <double kM> void CheckSingleBatchAgree() {
    std::mt19937_64 rng(424242);
    std::uniform_real_distribution<double> xd(1e-4, 40.0);
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const double x = xd(rng);
        const BoysRegion region = RegionOf(x);
        const int nmax = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
        BoysAllOrders<kM>(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const double single = BoysSingle<kM>(k, x);
            const double bound = kM * (RegionBound(region, LaneKind::kDoubleSingle) +
                                       RegionBound(region, LaneKind::kDoubleBatch));
            EXPECT_LE(std::abs(batch[static_cast<std::size_t>(k)] - single), bound)
                << "m=" << kM << " n=" << k << " x=" << x;
        }
    }
}

template <double kM> void CheckSingleBatchAgreeF32() {
    std::mt19937_64 rng(424243);
    std::uniform_real_distribution<float> xd(1e-4f, 40.0f);
    std::vector<float> batch(boys::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const float x = xd(rng);
        const BoysRegion region = RegionOf(static_cast<double>(x));
        const int nmax = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
        BoysAllOrdersF32<kM>(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const float single = BoysSingleF32<kM>(k, x);
            const double bound = kM * (RegionBound(region, LaneKind::kFloatSingle) +
                                       RegionBound(region, LaneKind::kFloatBatch));
            EXPECT_LE(std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) -
                               static_cast<double>(single)),
                      bound)
                << "m=" << kM << " n=" << k << " x=" << x;
        }
    }
}

template <double kM> void CheckZeroArgument() {
    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        EXPECT_DOUBLE_EQ(BoysSingle<kM>(n, 0.0), 1.0 / (2.0 * n + 1.0));
        EXPECT_FLOAT_EQ(BoysSingleF32<kM>(n, 0.0f), 1.0f / (2.0f * static_cast<float>(n) + 1.0f));
    }

    double batch[boys::kMaxBoysOrder + 1];
    BoysAllOrders<kM>(8, 0.0, batch);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_DOUBLE_EQ(batch[k], 1.0 / (2.0 * k + 1.0));
    }

    float batchF32[boys::kMaxBoysOrder + 1];
    BoysAllOrdersF32<kM>(8, 0.0f, batchF32);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_FLOAT_EQ(batchF32[k], 1.0f / (2.0f * static_cast<float>(k) + 1.0f));
    }
}

TEST(BoysAccuracyTest, SingleBatchAgreeAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        CheckSingleBatchAgree<kM>();
        CheckSingleBatchAgreeF32<kM>();
    });
}

TEST(BoysAccuracyTest, ZeroArgumentExactAtSampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() { CheckZeroArgument<kM>(); });
}

// ---------------------------------------------------------------------------
// The fp16/Bf16 lanes at sampled m: |F_hat - F(x16)| <= m*1e-7 + 1/2
// ULP per value, F(x16) the certified double lane at the fp16-rounded
// argument (the reference lane). The multiplier forwards to the F32
// engine's kFp16-budget roles (no fp16-specific degree tables).
// ---------------------------------------------------------------------------
#if BoysFp16

double HalfUlp(boys::F16 x) {
    return 0.5 * (static_cast<double>(boys::NextUp(x)) - static_cast<double>(x));
}

double HalfUlp(boys::Bf16 x) {
    return 0.5 * (static_cast<double>(boys::NextUp(x)) - static_cast<double>(x));
}

template <typename Half,
          double kM,
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept>
void RunHalfSampledCheck(const char* label) {
    double worstSingle = 0.0;
    double worstBatch = 0.0;
    std::vector<Half> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        // Two steps on purpose: the half types take the value at float width
        // first (the F16/Bf16 fallback constructors are float-taking), so the
        // narrowing is spelled out rather than left to the compiler.
        const Half x = static_cast<Half>(static_cast<float>(row.x));
        const double reference = BoysSingle<1.0>(row.n, static_cast<double>(x));
        const Half half = static_cast<Half>(static_cast<float>(reference));
        const double tolerance = kM * 1e-7 + HalfUlp(half);
        const double single = static_cast<double>(SingleFn(row.n, x));
        const double singleError = std::abs(single - reference);
        EXPECT_LE(singleError, tolerance) << "m=" << kM << " n=" << row.n << " x=" << row.x
                                          << " got=" << single << " want=" << reference;
        worstSingle = std::max(worstSingle, singleError);
        BatchFn(row.n, x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double batchReference = BoysSingle<1.0>(k, static_cast<double>(x));
            const double batchTolerance =
                kM * 1e-7 + HalfUlp(static_cast<Half>(static_cast<float>(batchReference)));
            const double batchError =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - batchReference);
            EXPECT_LE(batchError, batchTolerance)
                << "m=" << kM << " batch F" << k << " at x=" << row.x
                << " got=" << static_cast<double>(batch[static_cast<std::size_t>(k)])
                << " want=" << batchReference;
            worstBatch = std::max(worstBatch, batchError);
        }
    }

    std::printf("%s m=%.0e: worst single |error| %.3e, worst batch |error| %.3e "
                "(bound m*1e-7 + 1/2 ULP per value)\n",
                label,
                kM,
                worstSingle,
                worstBatch);
}

TEST(BoysAccuracyTest, F16SampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        RunHalfSampledCheck<boys::F16, kM, boys::BoysSingleF16<kM>, boys::BoysAllOrdersF16<kM>>(
            "BoysF16");
    });
}

TEST(BoysAccuracyTest, Bf16SampledMultipliers) {
    ForEachSampledMultiplier([]<double kM>() {
        RunHalfSampledCheck<boys::Bf16, kM, boys::BoysSingleBf16<kM>, boys::BoysAllOrdersBf16<kM>>(
            "BoysBf16");
    });
}

#endif // BoysFp16

} // namespace
