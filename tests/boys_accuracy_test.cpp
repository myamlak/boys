// The accuracy contract tests.
//
// Per-region contract: for {double single, double batch, float single, float batch},
// assert |F_hat - ref| <= B_region per region on the committed reference grid (region
// bucketing: A x < kX0, B kX0 <= x < kX1, C x >= kX1), with the asserted bounds B: double
// single 1e-15/3e-14/5.5e-14, double batch 5.5e-14 per region, float single 1.5e-7 per
// region, and float batch the figure the lane publishes for the division form the entries
// divide in - 1.5e-7 at exact division and at the refined reciprocal, plus the row's
// plain-reciprocal term where the build compiles that form, which is the term that form
// spends on the downward ladder the batch shape reads.
//
// The effective-degree tables carry the full degrees of the table they are read from,
// over all six lane roles and both regions, and are inside the evaluator domain.
//
// The fp16/Bf16 lanes forward to the F32 engine (I/O-only wrappers, no fp16-specific
// degree tables): assert |F_hat - F(x16)| <= 1e-7 + 1/2 ULP per value on the reference
// grid, F(x16) being the certified double lane evaluated at the fp16-rounded argument.
//
// The call sites below route to the library's certified instantiations, which the
// extern-template declarations in boys/boys.hpp name.
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
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <span>
#include <sstream>
#include <string>
#include <utility>
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
using boys::detail::TailBasis;

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

// The asserted per-region bounds. The extended band carries the region-B budget: its
// per-range F0 seed serves the certified upward recursion.
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

// The policy the float batch entry's unnamed call compiles: the fp32 all-orders class's
// row. The sweeps below call BoysAllOrdersF32 naming no policy, and the form that entry's
// downward ladder divides in is this class's and not the seam's five - a replacement
// header may move the class (boys/boys.hpp expands the one table the header carries), and
// the seam's five are then the point a class with no row falls to and not this one's form.
using FloatBatchClass = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>;

// The figure the float lane publishes for the division form the entries below run: the
// lane's own contract row - the row the README's table, BoysAccuracyGuaranteed and both
// gates read - plus the term the row carries beside its base for the plain reciprocal
// where that is the form in force. The row's own scaling is that sum, this lane's additive
// term being zero, so the sweeps below read the figure the accessor answers.
//
// The form is the caller's argument and not a seam read, because the answer belongs to the
// class the entry under test compiles: an entry whose class row names the plain reciprocal
// spends that term however the seam's five are set, and one whose row names another form
// spends nothing.
//
// It is not the same number as the float base above, and the difference is the point: the
// base is what the lane's two other forms deliver, and the row's term is what the plain
// form spends on the downward ladder, which is inside the batch shape and not inside the
// single one. So the batch sweeps are read at this figure and the single shape at the base.
double FloatLanePublishedFigure(boys::DivisionForm kForm) {
    for (const boys::LaneContractInfo& row : boys::BoysLaneContracts())
    {
        if (row.precision == boys::Precision::kFp32)
        {
            return row.bound +
                   (kForm == boys::DivisionForm::kPlainReciprocal ? row.plainAdditive : 0.0);
        }
    }

    return 0.0;
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

void PrintWorsts(const char* lane, const RegionWorsts& worst) {
    std::printf("%s: worst region A %.3e, B %.3e, C %.3e, extended band %.3e\n",
                lane,
                worst.a,
                worst.b,
                worst.c,
                worst.e);
}

// ---------------------------------------------------------------------------
// The grid contract, one sweep per lane
// ---------------------------------------------------------------------------

void SweepDoubleSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = BoysSingle(row.n, row.x);
        const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single", worst);
}

void SweepDoubleBatch() {
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrders(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch", worst);
}

void SweepFloatSingle() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32(row.n, static_cast<float>(row.x));
        // The base and not the lane's published figure for the form in force: this shape
        // evaluates one order from one fit and reads no downward step, which is where the
        // row's plain-reciprocal term is spent (FloatLanePublishedFigure above).
        const double bound = RegionBound(RegionOf(row.x), LaneKind::kFloatSingle);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("float single", worst);
}

// The half lanes' engine budget, as the f32 entries now take it: the policy is a type,
// so naming the budget is naming a policy.
//
// Each policy is composed from the class the entry under test resolves to - the row the
// build's table carries for it - rather than from the seam's five, so the budget is the
// one cell the pairs below differ in. A replacement header carries its own row per class
// (boys/boys.hpp expands BOYS_BUILD_DEFAULT_ROWS in place of the five-composed table, and
// DefaultPolicy is that row), so a policy composed from the five is another arithmetic
// from the class's own the moment a row spells anything but the five: the two calls would
// then differ in their route and in their division form as well, and the equality below
// would read a difference the budget did not make as one it did.
using FloatSingleClass = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kSingle>;
using Fp16SingleBudget = boys::EvalPolicy<FloatSingleClass::kRoute,
                                          FloatSingleClass::kScheme,
                                          boys::BoysBudget::kFp16,
                                          FloatSingleClass::kPack,
                                          FloatSingleClass::kGranularity,
                                          FloatSingleClass::kDivision,
                                          FloatSingleClass::kRegionBExp>;
using Fp16BatchBudget = boys::EvalPolicy<FloatBatchClass::kRoute,
                                         FloatBatchClass::kScheme,
                                         boys::BoysBudget::kFp16,
                                         FloatBatchClass::kPack,
                                         FloatBatchClass::kGranularity,
                                         FloatBatchClass::kDivision,
                                         FloatBatchClass::kRegionBExp>;

// The float lane with the half lanes' engine budget named. The budget is what the halves'
// fits are cut against, and at the accuracy this library serves naming it selects nothing:
// the entry is checked against the default entry bit for bit, the row that would catch the
// option having reached the certified lane's arithmetic rather than only its table.
void SweepFloatSingleFp16Budget() {
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const float value = BoysSingleF32<Fp16SingleBudget>(row.n, static_cast<float>(row.x));
        // The base, for the reason the default entry's sweep above states: the single shape
        // reads no downward step, and the budget the policy names moves no division form.
        const double bound = RegionBound(RegionOf(row.x), LaneKind::kFloatSingle);
        const double error = std::abs(static_cast<double>(value) - row.value);
        EXPECT_LE(error, bound) << "n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);

        EXPECT_EQ(value, BoysSingleF32(row.n, static_cast<float>(row.x)))
            << "naming the budget must not move the certified lane";
    }

    PrintWorsts("float single, fp16 budget", worst);
}

void SweepFloatBatchFp16Budget() {
    RegionWorsts worst;
    std::vector<float> batch(boys::kMaxBoysOrder + 1);
    std::vector<float> plain(batch.size());

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32<Fp16BatchBudget>(row.n, static_cast<float>(row.x), batch.data());
        BoysAllOrdersF32(row.n, static_cast<float>(row.x), plain.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            // The lane's published figure for the form in force and not the shape's base:
            // this entry's downward ladder divides in the form this class's row names, and
            // the budget the policy names moves no division form.
            const double bound = FloatLanePublishedFigure(FloatBatchClass::kDivision);
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound) << "batch F" << k << " at x=" << row.x;
            worst.Update(error, row.x);

            EXPECT_EQ(batch[static_cast<std::size_t>(k)], plain[static_cast<std::size_t>(k)])
                << "naming the budget must not move the certified lane: n=" << row.n
                << " x=" << row.x << " order " << k;
        }
    }

    PrintWorsts("float batch, fp16 budget", worst);
}

void SweepFloatBatch() {
    RegionWorsts worst;
    std::vector<float> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        BoysAllOrdersF32(row.n, static_cast<float>(row.x), batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            // The lane's published figure for the form in force and not the shape's base:
            // this entry's downward ladder divides in the form this class's row names, so
            // the bar is the one the lane states for that form.
            const double bound = FloatLanePublishedFigure(FloatBatchClass::kDivision);
            const double error =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - reference);
            EXPECT_LE(error, bound)
                << "batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("float batch", worst);
}

// The Horner scheme, measured against the committed reference: the same sweep as the
// default scheme's above, over the table the monomial degree rule is certified on - the
// reading that separates the Horner coefficients and their arithmetic from the shipped
// split-Clenshaw lane.
void SweepDoubleBatchHorner() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        boys::BoysAllOrders<Policy>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "horner batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts("double batch, horner", worst);
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
template <boys::EvalScheme kScheme = boys::EvalScheme::kHorner> void SweepDoubleBatchUniform() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                    kScheme,
                                    boys::BoysBudget::kFloat,
                                    boys::PackAxis::kArguments,
                                    boys::FitGranularity::kUniform>;
    RegionWorsts worst;
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        boys::BoysAllOrders<Policy>(row.n, row.x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double reference = gGrid.Value(k, row.x);
            const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleBatch);
            const double error = std::abs(batch[static_cast<std::size_t>(k)] - reference);
            EXPECT_LE(error, bound)
                << "uniform batch F" << k << " at x=" << row.x
                << " got=" << batch[static_cast<std::size_t>(k)] << " want=" << reference;
            worst.Update(error, row.x);
        }
    }

    PrintWorsts(kScheme == boys::EvalScheme::kHorner ? "double batch, uniform horner"
                                                     : "double batch, uniform clenshaw",
                worst);
}

// The uniform route's single-order path, on the same grid and at the same bar - the
// reading where the route is cheapest rather than dearest. It shares the index arithmetic
// with the ladder above through one FlatLocate, so what this separates is the order lookup.
template <boys::EvalScheme kScheme = boys::EvalScheme::kHorner> void SweepDoubleSingleUniform() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                    kScheme,
                                    boys::BoysBudget::kFloat,
                                    boys::PackAxis::kArguments,
                                    boys::FitGranularity::kUniform>;
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = boys::BoysSingle<Policy>(row.n, row.x);
        const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "uniform single n=" << row.n << " x=" << row.x
                                << " got=" << value << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts(kScheme == boys::EvalScheme::kHorner ? "double single, uniform horner"
                                                     : "double single, uniform clenshaw",
                worst);
}

void SweepDoubleSingleHorner() {
    using Policy = boys::EvalPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>;
    RegionWorsts worst;

    for (const ReferenceRow& row : gReference)
    {
        const double value = boys::BoysSingle<Policy>(row.n, row.x);
        const double bound = RegionBound(RegionOf(row.x), LaneKind::kDoubleSingle);
        const double error = std::abs(value - row.value);
        EXPECT_LE(error, bound) << "horner n=" << row.n << " x=" << row.x << " got=" << value
                                << " want=" << row.value;
        worst.Update(error, row.x);
    }

    PrintWorsts("double single, horner", worst);
}

TEST(BoysAccuracyTest, DoubleBatchUniformAndHornerMeetTheReferenceGrid) {
    SweepDoubleBatchUniform();
    SweepDoubleSingleUniform();
    SweepDoubleBatchUniform<boys::EvalScheme::kSplitClenshaw>();
    SweepDoubleSingleUniform<boys::EvalScheme::kSplitClenshaw>();
    SweepDoubleBatchHorner();
}

TEST(BoysAccuracyTest, DoubleSingleHornerMeetsTheReferenceGrid) {
    SweepDoubleSingleHorner();
}

TEST(BoysAccuracyTest, DoubleSingleMeetsTheRegionBudget) {
    SweepDoubleSingle();
}

TEST(BoysAccuracyTest, DoubleBatchMeetsTheRegionBudget) {
    SweepDoubleBatch();
}

TEST(BoysAccuracyTest, FloatSingleMeetsTheRegionBudget) {
    SweepFloatSingle();
}

TEST(BoysAccuracyTest, FloatBatchMeetsTheRegionBudget) {
    SweepFloatBatch();
}

// The same two sweeps with the half lanes' engine budget named. Kept as their own tests, so
// the sweeps above stay the reading of the certified lane that they were.
TEST(BoysAccuracyTest, FloatSingleWithTheFp16BudgetMeetsTheRegionBudget) {
    SweepFloatSingleFp16Budget();
}

TEST(BoysAccuracyTest, FloatBatchWithTheFp16BudgetMeetsTheRegionBudget) {
    SweepFloatBatchFp16Budget();
}

// ---------------------------------------------------------------------------
// The effective degrees: the full degrees of the table read, inside
// the evaluator domain {0, 1, 2, 4, 6, ...} at every (role, point)
// ---------------------------------------------------------------------------

constexpr bool InEvaluatorDomain(int d) {
    return d == 0 || d == 1 || d == 2 || (d >= 4 && d % 2 == 0);
}

template <BoysRole kRole> void AssertRoleTables() {
    constexpr auto a1 = RegionADegrees<kRole>();
    constexpr auto b1 = RegionBDegrees<kRole>();

    for (std::size_t i = 0; i < a1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(a1[i])) << "region-A degree " << a1[i] << " at index " << i;
    }

    for (std::size_t i = 0; i < b1.size(); ++i)
    {
        EXPECT_TRUE(InEvaluatorDomain(b1[i])) << "region-B degree " << b1[i] << " at index " << i;
    }

    // The degree tables carry the full degrees: the criterion's budget is
    // zero at the accuracy this library serves, so only the full degree has
    // an empty tail.
    if constexpr (boys::detail::RoleUsesDoubleTables(kRole))
    {
        for (std::size_t p = 0; p < boys::detail::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], boys::detail::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree";
        }
    } else
    {
        for (std::size_t p = 0; p < boys::detail::f32::kPieces.size(); ++p)
        {
            EXPECT_EQ(a1[p], boys::detail::f32::kPieces[p].deg)
                << "region-A piece " << p << " must keep its full degree";
        }
    }

    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], boys::detail::kBDeg)
                << "region-B order " << n << " must keep its full degree";
        }
    } else
    {
        for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
        {
            EXPECT_EQ(b1[static_cast<std::size_t>(n)], boys::detail::f32::kBDeg)
                << "region-B order " << n << " must keep its full degree";
        }
    }
}

TEST(BoysAccuracyTest, EffectiveDegreesAreFullAndInDomain) {
    AssertRoleTables<BoysRole::kDoubleSingle>();
    AssertRoleTables<BoysRole::kDoubleBatch>();
    AssertRoleTables<BoysRole::kF32Single>();
    AssertRoleTables<BoysRole::kF32Batch>();
    AssertRoleTables<BoysRole::kF32Fp16Single>();
    AssertRoleTables<BoysRole::kF32Fp16Batch>();
}

TEST(BoysAccuracyTest, BothBasesReadTheirOwnTableToItsFullDegree) {
    // The carriage of the basis through the degree tables: the two bases must return the
    // degree table's own (full) degrees, region A and region B. The criterion's budget is
    // zero at the accuracy this library serves, so no degree is cut in either basis - and a
    // choice of basis that reached no table (one table shared by both, or a rule that always
    // read the Chebyshev coefficients) would show here as a basis whose degrees are not the
    // table's own.
    constexpr auto chebA = RegionADegrees<BoysRole::kDoubleSingle, TailBasis::kChebyshev>();
    constexpr auto monoA = RegionADegrees<BoysRole::kDoubleSingle, TailBasis::kMonomial>();
    constexpr auto chebB = RegionBDegrees<BoysRole::kDoubleBatch, TailBasis::kChebyshev>();
    constexpr auto monoB = RegionBDegrees<BoysRole::kDoubleBatch, TailBasis::kMonomial>();

    ASSERT_EQ(chebA.size(), boys::detail::kPieces.size());
    ASSERT_EQ(chebB.size(), static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
    ASSERT_EQ(chebA.size(), monoA.size());
    ASSERT_EQ(chebB.size(), monoB.size());

    for (std::size_t i = 0; i < chebA.size(); ++i)
    {
        EXPECT_EQ(chebA[i], boys::detail::kPieces[i].deg)
            << "region-A index " << i << " must be the table's own degree in either basis";
        EXPECT_EQ(chebA[i], monoA[i])
            << "region-A index " << i << " must be the full degree in either basis";
    }

    for (std::size_t i = 0; i < chebB.size(); ++i)
    {
        EXPECT_EQ(chebB[i], boys::detail::kBDeg)
            << "region-B index " << i << " must be the table's own degree in either basis";
        EXPECT_EQ(chebB[i], monoB[i])
            << "region-B index " << i << " must be the full degree in either basis";
    }
}

// ---------------------------------------------------------------------------
// Delivered-path consistency: the single and batch lanes (separate
// arithmetic paths — per-order fits vs. seed + recursion) must agree within
// the sum of their region bounds, and the exact x = 0 values must survive.
// ---------------------------------------------------------------------------

void CheckSingleBatchAgree() {
    std::mt19937_64 rng(424242);
    std::uniform_real_distribution<double> xd(1e-4, 40.0);
    std::vector<double> batch(boys::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const double x = xd(rng);
        const BoysRegion region = RegionOf(x);
        const int nmax = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
        BoysAllOrders(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const double single = BoysSingle(k, x);
            const double bound = RegionBound(region, LaneKind::kDoubleSingle) +
                                 RegionBound(region, LaneKind::kDoubleBatch);
            EXPECT_LE(std::abs(batch[static_cast<std::size_t>(k)] - single), bound)
                << "n=" << k << " x=" << x;
        }
    }
}

void CheckSingleBatchAgreeF32() {
    std::mt19937_64 rng(424243);
    std::uniform_real_distribution<float> xd(1e-4f, 40.0f);
    std::vector<float> batch(boys::kMaxBoysOrder + 1);

    for (int sample = 0; sample < 200; ++sample)
    {
        const float x = xd(rng);
        const BoysRegion region = RegionOf(static_cast<double>(x));
        const int nmax = static_cast<int>(rng() % (boys::kMaxBoysOrder + 1));
        BoysAllOrdersF32(nmax, x, batch.data());

        for (int k = 0; k <= nmax; ++k)
        {
            const float single = BoysSingleF32(k, x);
            const double bound = RegionBound(region, LaneKind::kFloatSingle) +
                                 RegionBound(region, LaneKind::kFloatBatch);
            EXPECT_LE(std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) -
                               static_cast<double>(single)),
                      bound)
                << "n=" << k << " x=" << x;
        }
    }
}

void CheckZeroArgument() {
    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        EXPECT_DOUBLE_EQ(BoysSingle(n, 0.0), 1.0 / (2.0 * n + 1.0));
        EXPECT_FLOAT_EQ(BoysSingleF32(n, 0.0f), 1.0f / (2.0f * static_cast<float>(n) + 1.0f));
    }

    double batch[boys::kMaxBoysOrder + 1];
    BoysAllOrders(8, 0.0, batch);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_DOUBLE_EQ(batch[k], 1.0 / (2.0 * k + 1.0));
    }

    float batchF32[boys::kMaxBoysOrder + 1];
    BoysAllOrdersF32(8, 0.0f, batchF32);

    for (int k = 0; k <= 8; ++k)
    {
        EXPECT_FLOAT_EQ(batchF32[k], 1.0f / (2.0f * static_cast<float>(k) + 1.0f));
    }
}

TEST(BoysAccuracyTest, SingleBatchAgreeWithinTheBoundSum) {
    CheckSingleBatchAgree();
    CheckSingleBatchAgreeF32();
}

TEST(BoysAccuracyTest, ZeroArgumentIsExact) {
    CheckZeroArgument();
}

// ---------------------------------------------------------------------------
// The fp16/Bf16 lanes: |F_hat - F(x16)| <= 1e-7 + 1/2
// ULP per value, F(x16) the certified double lane at the fp16-rounded
// argument (the reference lane). The half entries forward to the F32
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
          Half (*SingleFn)(int, Half) noexcept,
          void (*BatchFn)(int, Half, Half*) noexcept>
void RunHalfCheck(const char* label) {
    double worstSingle = 0.0;
    double worstBatch = 0.0;
    std::vector<Half> batch(boys::kMaxBoysOrder + 1);

    for (const ReferenceRow& row : gReference)
    {
        // Two steps on purpose: the half types take the value at float width
        // first (the F16/Bf16 fallback constructors are float-taking), so the
        // narrowing is spelled out rather than left to the compiler.
        const Half x = static_cast<Half>(static_cast<float>(row.x));
        const double reference = BoysSingle(row.n, static_cast<double>(x));
        const Half half = static_cast<Half>(static_cast<float>(reference));
        const double tolerance = 1e-7 + HalfUlp(half);
        const double single = static_cast<double>(SingleFn(row.n, x));
        const double singleError = std::abs(single - reference);
        EXPECT_LE(singleError, tolerance)
            << "n=" << row.n << " x=" << row.x << " got=" << single << " want=" << reference;
        worstSingle = std::max(worstSingle, singleError);
        BatchFn(row.n, x, batch.data());

        for (int k = 0; k <= row.n; ++k)
        {
            const double batchReference = BoysSingle(k, static_cast<double>(x));
            const double batchTolerance =
                1e-7 + HalfUlp(static_cast<Half>(static_cast<float>(batchReference)));
            const double batchError =
                std::abs(static_cast<double>(batch[static_cast<std::size_t>(k)]) - batchReference);
            EXPECT_LE(batchError, batchTolerance)
                << "batch F" << k << " at x=" << row.x
                << " got=" << static_cast<double>(batch[static_cast<std::size_t>(k)])
                << " want=" << batchReference;
            worstBatch = std::max(worstBatch, batchError);
        }
    }

    std::printf("%s: worst single |error| %.3e, worst batch |error| %.3e "
                "(bound 1e-7 + 1/2 ULP per value)\n",
                label,
                worstSingle,
                worstBatch);
}

TEST(BoysAccuracyTest, F16MeetsTheHalfBudget) {
    RunHalfCheck<boys::F16, boys::BoysSingleF16<>, boys::BoysAllOrdersF16<>>("BoysF16");
}

TEST(BoysAccuracyTest, Bf16MeetsTheHalfBudget) {
    RunHalfCheck<boys::Bf16, boys::BoysSingleBf16<>, boys::BoysAllOrdersBf16<>>("BoysBf16");
}

#endif // BoysFp16

// ---------------------------------------------------------------------------
// The compile-time guarantee entry, held to the library's own tables.
//
// `BoysAccuracyGuaranteedStated` is the guarantee a *stated* seven-tuple carries:
// the figure `BoysAccuracyGuaranteed` answers with, read from the same table at
// the same seven axes, plus an assertion that every axis is a member of its own
// enumeration (`detail::GuaranteeAxesAreEnumerators`). Nothing in this repository
// instantiated it, and one bound in that assertion sat one member behind its
// enumeration - `Precision` bounded at `kFp16Device` while the enumeration had
// grown to `kBf16`, and again when `kBf16Device` was appended - with every suite green:
// the entry was refused at compile time
// for a tuple the run-time accessor served. The checks below hold the two sides
// against each other, and they fail in both directions: a bound left behind by a
// member appended to an enumeration is a compile error here, and a boundary this
// file names that the library's own table does not carry is a failure at run time.
//
// The members of each axis are read from the library's own rows - BoysLaneContracts
// for the lanes, BoysFitRoutes and BoysFitRoutesF32 for the fit routes,
// BoysEvalSchemes, BoysPackAxes, BoysFitGranularities, BoysDivisionForms and
// BoysRegionBExps for the rest. What this file names from an enumeration is its
// last member and nothing else: that is the boundary the assertion is about, and
// the two static_asserts per axis pin the assertion to it. A member appended to an
// enumeration without a row in its table - or a row without a member - fails
// `ExpectAxisIsTheEnumeration`; a bound left behind by such a member fails the
// boundary asserts below and the whole cross with them.
//
// WHAT IT CATCHES, AND WHAT IT DOES NOT
//
// It catches drift between an axis's enumeration and (a) the assertion's bound and
// (b) the row table the run-time accessor reads, and it catches a compile-time
// answer that disagrees with the run-time one at any tuple of the cross.
//
// It does NOT catch a defect the two sides share. A figure both the stated entry and
// the accessor read from one wrong row is one answer to this comparison, however wrong
// it is; what is held against the arithmetic here is nothing at all.
//
// It does NOT catch a bound that is right for its enumeration and wrong for the
// mathematics, and no check of this shape can: whether 1.5e-7 is the figure a lane
// may promise is a question about the arithmetic, not about which members an
// enumeration has. That question belongs to the accuracy gate
// (tests/boys_accuracy_gate.cpp, which measures every lane's cells against the
// committed high-precision grid) and to the bound instrument, and nothing here
// answers it.
// ---------------------------------------------------------------------------

// The last member of each axis, named by the enumeration that carries it. Each is
// the value the assertion's own bound is stated against - `Precision::kBf16Device`
// is what `GuaranteeAxesAreEnumerators` bounds `Precision` at - so this is where the
// boundary the entry is supposed to have is written down, and the rows the run-time
// accessor reads are what hold it to the library's carried set.
constexpr std::size_t kLastPrecision = static_cast<std::size_t>(boys::Precision::kBf16Device);
constexpr std::size_t kLastRoute = static_cast<std::size_t>(boys::FitRoute::kRationalMinimax);
constexpr std::size_t kLastScheme = static_cast<std::size_t>(boys::EvalScheme::kHorner);
constexpr std::size_t kLastAxis = static_cast<std::size_t>(boys::PackAxis::kOrders);
constexpr std::size_t kLastGranularity = static_cast<std::size_t>(boys::FitGranularity::kUniform);
constexpr std::size_t kLastForm = static_cast<std::size_t>(boys::DivisionForm::kRefinedReciprocal);
constexpr std::size_t kLastExp = static_cast<std::size_t>(boys::RegionBExp::kFast);

// The first enumerator of each axis. The assertion is a conjunction of independent
// per-axis comparisons, so holding the other six here isolates the one axis a check
// is about; 0 is inside every one of the bounds by construction (each is a `<=`
// against a value of at least 0), which is what makes the isolation exact.
constexpr boys::Precision kFirstPrecision = static_cast<boys::Precision>(0);
constexpr boys::FitRoute kFirstRoute = static_cast<boys::FitRoute>(0);
constexpr boys::EvalScheme kFirstScheme = static_cast<boys::EvalScheme>(0);
constexpr boys::PackAxis kFirstAxis = static_cast<boys::PackAxis>(0);
constexpr boys::FitGranularity kFirstGranularity = static_cast<boys::FitGranularity>(0);
constexpr boys::DivisionForm kFirstForm = static_cast<boys::DivisionForm>(0);
constexpr boys::RegionBExp kFirstExp = static_cast<boys::RegionBExp>(0);

// The assertion with one axis left free, the other six at their first enumerator.
template <boys::Precision kPrecision>
constexpr bool PrecisionIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kPrecision, kFirstRoute, kFirstScheme, kFirstAxis,
                                                     kFirstGranularity, kFirstForm, kFirstExp>();
}

template <boys::FitRoute kRoute>
constexpr bool RouteIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kRoute, kFirstScheme, kFirstAxis,
                                                     kFirstGranularity, kFirstForm, kFirstExp>();
}

template <boys::EvalScheme kScheme>
constexpr bool SchemeIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kFirstRoute, kScheme, kFirstAxis,
                                                     kFirstGranularity, kFirstForm, kFirstExp>();
}

template <boys::PackAxis kAxis>
constexpr bool AxisIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kFirstRoute, kFirstScheme, kAxis,
                                                     kFirstGranularity, kFirstForm, kFirstExp>();
}

template <boys::FitGranularity kGranularity>
constexpr bool GranularityIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kFirstRoute, kFirstScheme, kFirstAxis,
                                                     kGranularity, kFirstForm, kFirstExp>();
}

template <boys::DivisionForm kForm>
constexpr bool FormIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kFirstRoute, kFirstScheme, kFirstAxis,
                                                     kFirstGranularity, kForm, kFirstExp>();
}

template <boys::RegionBExp kExp>
constexpr bool ExpIsAnEnumerator() noexcept {
    return boys::detail::GuaranteeAxesAreEnumerators<kFirstPrecision, kFirstRoute, kFirstScheme, kFirstAxis,
                                                     kFirstGranularity, kFirstForm, kExp>();
}

// The boundary, at compile time and per axis: the last member of the enumeration is
// taken and a value one past it is refused. The refusal half is the one that carries
// the weight - an assertion that takes everything is not a bound but a deletion, and
// it would look like a fix - and it is proven end to end, on the entry itself rather
// than on this predicate, by the scratch translation units under .claude/tmp/
// (guarstatee-refusal-*.cpp), which are compiled as a control and must fail.
//
// What these two asserts do not say is which enumeration the boundary is the last
// member of: that is `ExpectAxisIsTheEnumeration` below, at run time, against the
// rows the library itself carries.
static_assert(PrecisionIsAnEnumerator<static_cast<boys::Precision>(kLastPrecision)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of Precision: its bound sits "
              "behind the enumeration it bounds, which is the defect that stood at kFp16Device "
              "while the enumeration had grown to kBf16, and at kBf16 while it had grown to "
              "kBf16Device");
static_assert(!PrecisionIsAnEnumerator<static_cast<boys::Precision>(kLastPrecision + 1)>(),
              "GuaranteeAxesAreEnumerators takes a Precision one past the last member: a bound "
              "that accepts everything bounds nothing");

static_assert(RouteIsAnEnumerator<static_cast<boys::FitRoute>(kLastRoute)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of FitRoute: its bound sits "
              "behind the enumeration it bounds");
static_assert(!RouteIsAnEnumerator<static_cast<boys::FitRoute>(kLastRoute + 1)>(),
              "GuaranteeAxesAreEnumerators takes a FitRoute one past the last member: a bound "
              "that accepts everything bounds nothing");

static_assert(SchemeIsAnEnumerator<static_cast<boys::EvalScheme>(kLastScheme)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of EvalScheme: its bound sits "
              "behind the enumeration it bounds");
static_assert(!SchemeIsAnEnumerator<static_cast<boys::EvalScheme>(kLastScheme + 1)>(),
              "GuaranteeAxesAreEnumerators takes an EvalScheme one past the last member: a bound "
              "that accepts everything bounds nothing");

static_assert(AxisIsAnEnumerator<static_cast<boys::PackAxis>(kLastAxis)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of PackAxis: its bound sits "
              "behind the enumeration it bounds");
static_assert(!AxisIsAnEnumerator<static_cast<boys::PackAxis>(kLastAxis + 1)>(),
              "GuaranteeAxesAreEnumerators takes a PackAxis one past the last member: a bound "
              "that accepts everything bounds nothing");

static_assert(GranularityIsAnEnumerator<static_cast<boys::FitGranularity>(kLastGranularity)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of FitGranularity: its bound "
              "sits behind the enumeration it bounds");
static_assert(!GranularityIsAnEnumerator<static_cast<boys::FitGranularity>(kLastGranularity + 1)>(),
              "GuaranteeAxesAreEnumerators takes a FitGranularity one past the last member: a "
              "bound that accepts everything bounds nothing");

static_assert(FormIsAnEnumerator<static_cast<boys::DivisionForm>(kLastForm)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of DivisionForm: its bound "
              "sits behind the enumeration it bounds");
static_assert(!FormIsAnEnumerator<static_cast<boys::DivisionForm>(kLastForm + 1)>(),
              "GuaranteeAxesAreEnumerators takes a DivisionForm one past the last member: a "
              "bound that accepts everything bounds nothing");

static_assert(ExpIsAnEnumerator<static_cast<boys::RegionBExp>(kLastExp)>(),
              "GuaranteeAxesAreEnumerators refuses the last member of RegionBExp: its bound sits "
              "behind the enumeration it bounds");
static_assert(!ExpIsAnEnumerator<static_cast<boys::RegionBExp>(kLastExp + 1)>(),
              "GuaranteeAxesAreEnumerators takes a RegionBExp one past the last member: a bound "
              "that accepts everything bounds nothing");

// The selector values the rows of one of the library's axis tables carry.
template <typename Row, typename Selector>
std::vector<unsigned> SelectorsOf(std::span<const Row> rows, Selector selector) {
    std::vector<unsigned> values;
    values.reserve(rows.size());

    for (const Row& row : rows)
    {
        values.push_back(selector(row));
    }

    return values;
}

// The fit routes: both lanes' route tables name the axis, and the members are the
// union of the two - the route is one enumeration over both.
std::vector<unsigned> RouteSelectors() {
    std::vector<unsigned> values = SelectorsOf(
        boys::BoysFitRoutes(), [](const boys::FitRouteInfo& row) { return static_cast<unsigned>(row.route); });
    const std::vector<unsigned> single = SelectorsOf(
        boys::BoysFitRoutesF32(), [](const boys::FitRouteInfo& row) { return static_cast<unsigned>(row.route); });
    values.insert(values.end(), single.begin(), single.end());

    return values;
}

// The axis the library carries, held to the boundary this file names: one row per
// enumerator, in enumerator order, and no member the enumeration does not have. This
// is what makes the boundary above the library's own rather than this file's: a row
// added for a member this file has not named, or a member named here that no row
// carries, is a failure with the axis's name on it.
void ExpectAxisIsTheEnumeration(const char* axis, std::vector<unsigned> values, std::size_t last) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());

    ASSERT_FALSE(values.empty()) << axis << ": the library's own table carries no row for this axis";

    EXPECT_EQ(values.size(), last + 1)
        << axis << ": the library's table carries " << values.size()
        << " members and this file names a boundary at " << last + 1
        << " - the table and the boundary have drifted apart";

    const std::size_t common = std::min(values.size(), last + 1);

    for (std::size_t i = 0; i < common; ++i)
    {
        EXPECT_EQ(values[i], i) << axis << ": member " << i << " of the library's table is "
                                << values[i] << ", so the axis is not the enumerator set 0.." << last;
    }
}

// The name the library gives a lane and a route, for a failure message; a value
// outside the tables is answered "unknown", which the cross never names.
const char* LaneName(boys::Precision precision) {
    const std::size_t index = static_cast<std::size_t>(precision);
    const std::span<const boys::LaneContractInfo> lanes = boys::BoysLaneContracts();

    return index < lanes.size() ? lanes[index].name : "unknown";
}

const char* RouteName(boys::FitRoute route) {
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        if (row.route == route)
        {
            return row.name;
        }
    }

    return "unknown";
}

// The seven axes of one cell, as a reader names them.
std::string CellLabel(boys::Precision precision,
                      boys::FitRoute route,
                      boys::EvalScheme scheme,
                      boys::PackAxis axis,
                      boys::FitGranularity granularity,
                      boys::DivisionForm form,
                      boys::RegionBExp exp) {
    char buffer[192];
    std::snprintf(buffer,
                  sizeof buffer,
                  "precision=%s route=%s scheme=%s axis=%s granularity=%s form=%s exp=%s",
                  LaneName(precision),
                  RouteName(route),
                  boys::EvalSchemeName(scheme),
                  boys::PackAxisName(axis),
                  boys::GranularityName(granularity),
                  boys::DivisionFormName(form),
                  boys::RegionBExpName(exp));

    return buffer;
}

// The axes the library carries, one count per axis: what the cross below is the
// product of.
constexpr std::size_t kPrecisionCount = kLastPrecision + 1;
constexpr std::size_t kRouteCount = kLastRoute + 1;
constexpr std::size_t kSchemeCount = kLastScheme + 1;
constexpr std::size_t kAxisCount = kLastAxis + 1;
constexpr std::size_t kGranularityCount = kLastGranularity + 1;
constexpr std::size_t kFormCount = kLastForm + 1;
constexpr std::size_t kExpCount = kLastExp + 1;

// Every combination of every axis: 8 lanes x 2 fit routes x 2 schemes x 2 packing
// axes x 3 partitions x 3 division forms x 2 region-B exponentials = 1152 cells,
// each one instantiated at compile time and each one held to the run-time accessor
// at the same seven axes. The product is written as a product and printed by the
// test, so the count is the compiler's arithmetic and the run's own, not this
// file's.
constexpr std::size_t kCombinationCount = kPrecisionCount * kRouteCount * kSchemeCount * kAxisCount *
                                          kGranularityCount * kFormCount * kExpCount;

// One axis's digit of a cell index, the last axis running fastest: the mixed radix
// the product above is, decoded one axis at a time.
template <std::size_t kAxis>
constexpr std::size_t CellDigit(std::size_t cell) noexcept {
    constexpr std::size_t kCounts[7] = {kPrecisionCount,
                                        kRouteCount,
                                        kSchemeCount,
                                        kAxisCount,
                                        kGranularityCount,
                                        kFormCount,
                                        kExpCount};
    std::size_t remaining = cell;

    for (std::size_t i = 6; i > kAxis; --i)
    {
        remaining /= kCounts[i];
    }

    return remaining % kCounts[kAxis];
}

// The seven axes of one cell, as values: what a message names a cell by.
struct CellAxes {
    boys::Precision precision{};
    boys::FitRoute route{};
    boys::EvalScheme scheme{};
    boys::PackAxis axis{};
    boys::FitGranularity granularity{};
    boys::DivisionForm form{};
    boys::RegionBExp exp{};
};

CellAxes AxesOf(std::size_t cell) {
    CellAxes axes{};

    axes.precision = static_cast<boys::Precision>(CellDigit<0>(cell));
    axes.route = static_cast<boys::FitRoute>(CellDigit<1>(cell));
    axes.scheme = static_cast<boys::EvalScheme>(CellDigit<2>(cell));
    axes.axis = static_cast<boys::PackAxis>(CellDigit<3>(cell));
    axes.granularity = static_cast<boys::FitGranularity>(CellDigit<4>(cell));
    axes.form = static_cast<boys::DivisionForm>(CellDigit<5>(cell));
    axes.exp = static_cast<boys::RegionBExp>(CellDigit<6>(cell));

    return axes;
}

// One cell's two answers, as the compile-time entry and the run-time accessor gave
// them. The struct is what one instantiation produces, so the cross is a vector of
// these rather than one assertion site per cell: the comparison is a loop at the end,
// and a cell that agrees costs a call and two figures.
struct CellAnswers {
    boys::AccuracyFigure stated{};
    boys::AccuracyFigure answered{};
};

// One cell: the compile-time entry instantiated at the tuple, and the same tuple asked
// of the run-time accessor. The two are the defect class this whole block is for, so
// they are read back and compared rather than trusted: the assertion inside the stated
// entry is the part a compiler settles, and the accessor's answer is what a caller gets.
template <boys::Precision kPrecision,
          boys::FitRoute kRoute,
          boys::EvalScheme kScheme,
          boys::PackAxis kAxis,
          boys::FitGranularity kGranularity,
          boys::DivisionForm kForm,
          boys::RegionBExp kExp>
CellAnswers AskBoth() {
    return CellAnswers{
        boys::BoysAccuracyGuaranteedStated<kPrecision, kRoute, kScheme, kAxis, kGranularity, kForm, kExp>(),
        boys::BoysAccuracyGuaranteed(kPrecision, kRoute, kScheme, kAxis, kGranularity, kForm, kExp)};
}

template <std::size_t kCell>
void AskOneCell(std::vector<CellAnswers>& answers) {
    constexpr std::size_t kPrecision = CellDigit<0>(kCell);
    constexpr std::size_t kRoute = CellDigit<1>(kCell);
    constexpr std::size_t kScheme = CellDigit<2>(kCell);
    constexpr std::size_t kAxis = CellDigit<3>(kCell);
    constexpr std::size_t kGranularity = CellDigit<4>(kCell);
    constexpr std::size_t kForm = CellDigit<5>(kCell);
    constexpr std::size_t kExp = CellDigit<6>(kCell);

    answers.push_back(AskBoth<static_cast<boys::Precision>(kPrecision),
                              static_cast<boys::FitRoute>(kRoute),
                              static_cast<boys::EvalScheme>(kScheme),
                              static_cast<boys::PackAxis>(kAxis),
                              static_cast<boys::FitGranularity>(kGranularity),
                              static_cast<boys::DivisionForm>(kForm),
                              static_cast<boys::RegionBExp>(kExp)>());
}

// The cells are asked a chunk at a time rather than in one fold. A fold over all
// kCombinationCount of them nests one level per cell, and clang refuses a fold
// expression past 256 arguments (-fbracket-depth): the count is a property of the
// library's axes, and the suite must not stop compiling when an axis grows. Chunking
// bounds the nesting by the chunk, and the halving below adds only its logarithm.
constexpr std::size_t kCellsPerFold = 64;

template <std::size_t kFirst, std::size_t kLast>
void AskCellRange(std::vector<CellAnswers>& answers) {
    if constexpr (kLast - kFirst <= kCellsPerFold) {
        [&answers]<std::size_t... kOffsets>(std::index_sequence<kOffsets...>)
        { (AskOneCell<kFirst + kOffsets>(answers), ...); }(std::make_index_sequence<kLast - kFirst>{});
    } else {
        constexpr std::size_t kMiddle = kFirst + (kLast - kFirst) / 2;
        AskCellRange<kFirst, kMiddle>(answers);
        AskCellRange<kMiddle, kLast>(answers);
    }
}

std::vector<CellAnswers> AskEveryCombination() {
    std::vector<CellAnswers> answers;
    answers.reserve(kCombinationCount);

    AskCellRange<0, kCombinationCount>(answers);

    return answers;
}

// The two answers' disagreement, with all seven axes and both figures in it. Built
// only where a cell disagrees, so the cells that agree pay nothing for it.
std::string MismatchReport(std::size_t cell, const CellAnswers& answer) {
    const CellAxes axes = AxesOf(cell);
    char buffer[768];
    std::snprintf(buffer,
                  sizeof buffer,
                  "%s: stated available=%d value=%.17g reading=%d source=\"%s\" reason=\"%s\" | "
                  "accessor available=%d value=%.17g reading=%d source=\"%s\" reason=\"%s\"",
                  CellLabel(axes.precision,
                            axes.route,
                            axes.scheme,
                            axes.axis,
                            axes.granularity,
                            axes.form,
                            axes.exp)
                      .c_str(),
                  answer.stated.available ? 1 : 0,
                  answer.stated.value,
                  static_cast<int>(answer.stated.reading),
                  answer.stated.source,
                  answer.stated.reason,
                  answer.answered.available ? 1 : 0,
                  answer.answered.value,
                  static_cast<int>(answer.answered.reading),
                  answer.answered.source,
                  answer.answered.reason);

    return buffer;
}

// Whether the two answers are one answer: the same availability, the same figure, the
// same reading, and the same two sentences. The figures are read from one table by one
// expression, so they are compared exactly rather than within a tolerance.
bool AnswersAgree(const CellAnswers& answer) {
    return answer.stated.available == answer.answered.available &&
           answer.stated.value == answer.answered.value &&
           answer.stated.reading == answer.answered.reading &&
           std::strcmp(answer.stated.source, answer.answered.source) == 0 &&
           std::strcmp(answer.stated.reason, answer.answered.reason) == 0;
}

// The boundary this file names, held to the rows the library itself carries. It fails
// when a member is appended to an enumeration and given a row without this file's
// boundary moving - the direction that would otherwise let the entry fall behind
// silently - and when a row is added for a member the enumeration does not have.
TEST(BoysAccuracyTest, TheGuaranteeEntrysBoundaryIsTheLastMemberOfEveryAxis) {
    ExpectAxisIsTheEnumeration(
        "Precision",
        SelectorsOf(boys::BoysLaneContracts(),
                    [](const boys::LaneContractInfo& row) { return static_cast<unsigned>(row.precision); }),
        kLastPrecision);
    ExpectAxisIsTheEnumeration("FitRoute", RouteSelectors(), kLastRoute);
    ExpectAxisIsTheEnumeration(
        "EvalScheme",
        SelectorsOf(boys::BoysEvalSchemes(),
                    [](const boys::EvalSchemeInfo& row) { return static_cast<unsigned>(row.scheme); }),
        kLastScheme);
    ExpectAxisIsTheEnumeration(
        "PackAxis",
        SelectorsOf(boys::BoysPackAxes(),
                    [](const boys::PackAxisInfo& row) { return static_cast<unsigned>(row.axis); }),
        kLastAxis);
    ExpectAxisIsTheEnumeration(
        "FitGranularity",
        SelectorsOf(boys::BoysFitGranularities(),
                    [](const boys::FitGranularityInfo& row) { return static_cast<unsigned>(row.granularity); }),
        kLastGranularity);
    ExpectAxisIsTheEnumeration(
        "DivisionForm",
        SelectorsOf(boys::BoysDivisionForms(),
                    [](const boys::DivisionFormInfo& row) { return static_cast<unsigned>(row.form); }),
        kLastForm);
    ExpectAxisIsTheEnumeration(
        "RegionBExp",
        SelectorsOf(boys::BoysRegionBExps(),
                    [](const boys::RegionBExpInfo& row) { return static_cast<unsigned>(row.exp); }),
        kLastExp);
}

// Every cell of the cross, instantiated and answered. The compile-time entry is the
// one the library never exercised: a cell whose stated tuple the assertion refuses is
// a build error here, and a cell whose compile-time answer differs from the run-time
// one is a failure with the seven axes named.
TEST(BoysAccuracyTest, TheStatedGuaranteeIsTheAccessorsAnswerOnEveryCombination) {
    const std::vector<CellAnswers> answers = AskEveryCombination();
    ASSERT_EQ(answers.size(), kCombinationCount)
        << "the cross did not visit every combination it says it does";

    std::size_t carried = 0;
    std::size_t mismatched = 0;

    for (std::size_t cell = 0; cell < answers.size(); ++cell)
    {
        carried += answers[cell].stated.available ? 1u : 0u;

        if (!AnswersAgree(answers[cell]))
        {
            ++mismatched;
            ADD_FAILURE() << MismatchReport(cell, answers[cell]);
        }
    }

    std::printf("stated guarantee: %zu combinations instantiated and held to "
                "BoysAccuracyGuaranteed (%zu carried, %zu refused, %zu disagreeing)\n",
                answers.size(),
                carried,
                answers.size() - carried,
                mismatched);

    EXPECT_EQ(mismatched, 0u);
}

} // namespace
