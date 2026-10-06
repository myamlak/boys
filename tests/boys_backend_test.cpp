// The arithmetic-backend contract (boys/backend.hpp): which arithmetic the build carries, and what
// each one's multiply-adds round. MulAdd is fused and MulSub is not - the split-precision lane
// evaluates its recurrence at the two-rounding reading on purpose, and a build that fused the bare
// form would move it. Contracts() is asserted against a measurement in this translation unit.

#include "boys/backend.hpp"

#include "boys/boys.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <span>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using boys::backend::ArithmeticBackend;
using boys::backend::BackendInfo;
using boys::backend::BoysBackends;
using boys::backend::MulAddRoute;
using boys::backend::MulAddRouteName;
using boys::backend::Scalar;
using boys::backend::ScalarFp32;
using boys::backend::ScalarFp64;

static_assert(ArithmeticBackend<ScalarFp64>);
static_assert(ArithmeticBackend<ScalarFp32>);

// Operands whose product is inexact and whose addend cancels the leading term,
// so the fused and two-rounding readings land on different values.
template <typename T>
struct Probe {
    T a;
    T b;
    T c;
    T lost; // the bit the product's rounding removes
};

template <typename T>
Probe<T> MakeProbe() {
    constexpr int kShift = std::numeric_limits<T>::digits / 2 + 1;
    const T unit{1};
    const T small = static_cast<T>(std::ldexp(1.0, -kShift));
    const T half = static_cast<T>(std::ldexp(1.0, -(kShift - 1)));
    return Probe<T>{static_cast<T>(unit + small), static_cast<T>(unit + small),
                    static_cast<T>(-(unit + half)), static_cast<T>(small * small)};
}

// Whether a bare product-plus-add is one rounding here. The operands are volatile so the expression
// is evaluated rather than folded away; whether the multiply-add is contracted is the question, and
// the addend makes the two answers differ. The fused value comes from the standard function rather
// than the backend, whose own multiply-add is under test: asking it would have it confirm itself.
template <typename T>
bool BareIsFused() {
    const Probe<T> p = MakeProbe<T>();
    volatile const T va = p.a;
    volatile const T vb = p.b;
    volatile const T vc = p.c;
    const T bare = va * vb + vc;
    return bare == std::fma(va, vb, vc);
}

TEST(BackendTest, ScalarWidthAndValueType) {
    EXPECT_EQ(ScalarFp64::kWidth, 1u);
    EXPECT_EQ(ScalarFp32::kWidth, 1u);
    EXPECT_TRUE((std::is_same_v<ScalarFp64::Value, double>));
    EXPECT_TRUE((std::is_same_v<ScalarFp32::Value, float>));
    EXPECT_STREQ(ScalarFp64::kName, "scalar-fp64");
    EXPECT_STREQ(ScalarFp32::kName, "scalar-fp32");
}

// The transport round-trips; a broadcast is that value in the one lane of a scalar backend.
TEST(BackendTest, ScalarTransportRoundTrips) {
    const double stored = 3.25;
    const double loaded = ScalarFp64::Load(&stored);
    double out = 0.0;
    ScalarFp64::Store(&out, ScalarFp64::Mul(loaded, ScalarFp64::Broadcast(2.0)));
    EXPECT_EQ(out, 6.5);
}

// MulAdd is the selected route's arithmetic, and these operands tell the two routes
// apart: the fused step keeps the bit the product's rounding drops, the separate
// step loses it. MulSub is outside the route and two-rounding on every build.
TEST(BackendTest, MulAddIsTheSelectedRouteAndMulSubIsNot) {
    const Probe<double> p = MakeProbe<double>();

    const double mulAdd = ScalarFp64::MulAdd(p.a, p.b, p.c);
    const double mulSub = ScalarFp64::MulSub(p.a, p.b, p.c);

    // The two readings of the sum, spelled out: one rounding through the
    // standard function, and the product rounded before the sum rounds.
    const double fused = std::fma(p.a, p.b, p.c);
    const double separate = ScalarFp64::Add(ScalarFp64::Mul(p.a, p.b), p.c);

    // The two readings differ, so which one MulAdd returned says which arithmetic it ran.
    EXPECT_NE(fused, separate);
    EXPECT_EQ(fused, p.lost);

    // MulAdd is the route in force, which is the one the report states.
    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_GE(backends.size(), 1u);
    EXPECT_EQ(mulAdd, backends[0].route == MulAddRoute::kFused ? fused : separate);

    // MulSub is outside the route: product rounds, then difference rounds, on every build.
    EXPECT_EQ(mulSub, ScalarFp64::Sub(ScalarFp64::Mul(p.a, p.b), p.c));
    EXPECT_NE(mulSub, mulAdd);
}

// The reported route agrees with the arithmetic measured in this translation unit, for
// each scalar precision: a route reported as separate is one whose fused step is really
// two roundings here.
TEST(BackendTest, ReportedMulAddRouteMatchesTheArithmetic) {
    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_GE(backends.size(), 2u);

    EXPECT_EQ(backends[0].route, boys::backend::detail::RouteInForce<double>());
    EXPECT_EQ(backends[1].route, boys::backend::detail::RouteInForce<float>());

    // A route in force is the selection unless the build contracts the bare form, and that filtering is
    // the SCALAR pair's, whose separate route is a bare expression the build may contract. The packed
    // pair names two instructions instead, running its separate route either way, so what its rows owe
    // is that the route they report is the one they run, asserted below.
    for (std::size_t i = 0; i < 2; ++i) {
        if (backends[i].route == MulAddRoute::kSeparate) {
            EXPECT_FALSE(backends[i].contracts) << backends[i].name;
        }
    }
}

// The packed pair runs the route this build selected. The two routes are two spellings there, not two
// readings of one bare expression, so the reported route is the selection with nothing measured away
// from it: a build asked for the separate route whose packed lanes reported fused would be reporting
// an arithmetic whose kernels were never compiled into it.
TEST(BackendTest, ThePackedLanesRunTheSelectedRoute) {
    if (!boys::BoysAvx2Available()) {
        GTEST_SKIP() << "the AVX2 tier is not available on this target";
    }

    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_EQ(backends.size(), 4u);

    EXPECT_EQ(backends[2].route, boys::backend::detail::kSelectedRoute)
        << backends[2].name << " reports " << MulAddRouteName(backends[2].route)
        << ", the build selected " << MulAddRouteName(boys::backend::detail::kSelectedRoute);
    EXPECT_EQ(backends[3].route, boys::backend::detail::kSelectedRoute)
        << backends[3].name << " reports " << MulAddRouteName(backends[3].route)
        << ", the build selected " << MulAddRouteName(boys::backend::detail::kSelectedRoute);
}

// The route is a name a report can print, and no row answers with the placeholder.
TEST(BackendTest, MulAddRouteNamesArePrintable) {
    EXPECT_STREQ(MulAddRouteName(MulAddRoute::kFused), "fused");
    EXPECT_STREQ(MulAddRouteName(MulAddRoute::kSeparate), "separate");

    for (const BackendInfo& info : BoysBackends()) {
        EXPECT_STRNE(MulAddRouteName(info.route), "unknown") << info.name;
    }
}

// MulSub is the two-rounding reading on every build: the product and the difference are
// written separately, which no contraction pass can merge into one step.
TEST(BackendTest, MulSubAgreesWithSeparateProductAndDifference) {
    const double lattice[] = {0.0,           1.0,        -1.0,        2.5,
                              -3.75,         1e-8,       1e8,         6.25e-2,
                              0.333333333,   -72631.0,   1023.5,      -1e-300,
                              4.71238898038, 12345.6789, -0.0009765,  9.99999e13};
    std::size_t compared = 0;
    for (const double a : lattice) {
        for (const double b : lattice) {
            for (const double c : lattice) {
                EXPECT_EQ(ScalarFp64::MulSub(a, b, c),
                          ScalarFp64::Sub(ScalarFp64::Mul(a, b), c))
                    << "a=" << a << " b=" << b << " c=" << c;
                ++compared;
            }
        }
    }
    EXPECT_EQ(compared, std::size(lattice) * std::size(lattice) * std::size(lattice));
}

// Contracts() answers what this build does with a bare product-plus-add, asserted against
// the measurement itself in both precisions.
TEST(BackendTest, ContractsMatchesAMeasurementInThisTranslationUnit) {
    EXPECT_EQ(ScalarFp64::Contracts(), BareIsFused<double>());
    EXPECT_EQ(ScalarFp32::Contracts(), BareIsFused<float>());
}

// A lane is a value a report can print: every backend named once, by the name its type states.
TEST(BackendTest, TheCarriedBackendsAreEnumeratedAndNamed) {
    const std::span<const BackendInfo> backends = BoysBackends();

    ASSERT_GE(backends.size(), 2u);
    EXPECT_LE(backends.size(), 4u);
    EXPECT_STREQ(backends[0].name, ScalarFp64::kName);
    EXPECT_STREQ(backends[1].name, ScalarFp32::kName);

    std::vector<std::string> names;
    for (const BackendInfo& info : backends) {
        ASSERT_NE(info.name, nullptr);
        EXPECT_GT(std::string(info.name).size(), 0u);
        names.emplace_back(info.name);
    }
    std::sort(names.begin(), names.end());
    EXPECT_EQ(std::adjacent_find(names.begin(), names.end()), names.end())
        << "a backend name repeats";
}

// The reported contraction flag is the one the header gives for the arithmetic carrying that name.
// Contraction belongs to a compiled translation unit, and the library has more than one flag context:
// src/boys_simd.cpp carries the packed flags, everything else does not. A table measured in the packed
// unit would print that unit's answer beside the scalar lanes' values, so the comparison is made here.
TEST(BackendTest, ReportedContractionMatchesTheArithmetic) {
    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_GE(backends.size(), 2u);
    EXPECT_EQ(backends[0].contracts, ScalarFp64::Contracts());
    EXPECT_EQ(backends[1].contracts, ScalarFp32::Contracts());
}

// The packed pair appears exactly when the build has the AVX2 + FMA tier,
// which is the condition the library dispatches on.
TEST(BackendTest, ThePackedPairAppearsExactlyWithTheVectorTier) {
    const std::span<const BackendInfo> backends = BoysBackends();
    if (boys::BoysAvx2Available()) {
        ASSERT_EQ(backends.size(), 4u);
        EXPECT_STREQ(backends[2].name, "avx2-fp64");
        EXPECT_STREQ(backends[3].name, "avx2-fp32");
    } else {
        EXPECT_EQ(backends.size(), 2u);
    }
}

} // namespace

// Every axis the entries select is one field of EvalPolicy with its own default, and the pins below are
// the committed names - the configuration every bound in this repository was measured at - where
// boys/boys_build_defaults.hpp is the committed header. All five axes are read from it, the packing
// axis, division form and partition included (BOYS_BUILD_DEFAULTS, CONTRIBUTING.md).
static_assert(boys::EvalPolicy<>{}.kBudget == boys::BoysBudget::kFloat,
              "the default engine budget moved");

#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
static_assert(boys::EvalPolicy<>{}.kRoute == boys::FitRoute::kChebyshev,
              "the default fit route moved");
static_assert(boys::EvalPolicy<>{}.kScheme == boys::EvalScheme::kHorner,
              "the default evaluation scheme moved");

static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the default fit route moved: an unnamed call evaluates the Chebyshev fits every "
              "bound in this repository was measured at");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kHorner,
              "the default evaluation scheme moved");
#else
// A replacement is read instead of the committed file, not beside it. Where it carries
// BOYS_BUILD_DEFAULT_ROWS the table expands from those rows instead of from the five, never both
// (boys/boys.hpp, one branch or the other): a committed five with a row that moves a class has
// chosen the class policy, which a guard reading only the five refuses.
constexpr bool kCoarsestFiveInForce =
    boys::kDefaultFitRoute == boys::FitRoute::kChebyshev &&
    boys::kDefaultEvalScheme == boys::EvalScheme::kHorner &&
    boys::kDefaultPackAxis == boys::PackAxis::kArguments &&
    boys::kDefaultDivisionForm == boys::DivisionForm::kRefinedReciprocal &&
    boys::kDefaultFitGranularity == boys::FitGranularity::kNarrow;

// Whether the class table answers this class with the combination the committed build composes from
// the committed five: every axis compared, the region-B exponential included, one member per half
// (BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP). A class the table carries no row for answers true; the host
// bf16 classes are excepted - their committed rows are measured ones that name another combination.
template <boys::Precision kLane, boys::Shape kShape, boys::Device kDevice = boys::Device::kHost>
constexpr bool ClassIsTheCommittedCombination() noexcept
{
    using Row = boys::detail::DefaultPolicyRow<kDevice, kLane, kShape>;

    if constexpr (!Row::kCarried)
    {
        return true;
    }
    else
    {
        using Policy = typename Row::Type;

        constexpr boys::RegionBExp kCommittedRegionBExp =
            kDevice == boys::Device::kHost ? boys::kDefaultHostRegionBExp
                                           : boys::RegionBExp::kAccurate;

        return Policy::kRoute == boys::FitRoute::kChebyshev &&
               Policy::kScheme == boys::EvalScheme::kHorner &&
               Policy::kBudget == boys::detail::LaneFallbackBudget<kLane>() &&
               Policy::kPack == boys::PackAxis::kArguments &&
               Policy::kGranularity == boys::FitGranularity::kNarrow &&
               Policy::kDivision == boys::DivisionForm::kRefinedReciprocal &&
               Policy::kRegionBExp == kCommittedRegionBExp;
    }
}

constexpr bool kCoarsestClassTableInForce =
    ClassIsTheCommittedCombination<boys::Precision::kFp64, boys::Shape::kSingle>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64, boys::Shape::kFixedN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64, boys::Shape::kAllN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64, boys::Shape::kAllOrders>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32, boys::Shape::kSingle>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32, boys::Shape::kFixedN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32, boys::Shape::kAllN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32, boys::Shape::kAllNAtOrders>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32, boys::Shape::kAllOrders>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16, boys::Shape::kSingle>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16, boys::Shape::kFixedN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16, boys::Shape::kAllN>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16, boys::Shape::kAllNAtOrders>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16, boys::Shape::kAllOrders>() &&
    // The device half, the same twelve classes the seam's own list carries: the four device
    // lanes by the three questions a device entry answers. A replacement that left these at
    // the committed combination has chosen nothing for them, which is what this reads.
    ClassIsTheCommittedCombination<boys::Precision::kFp64Device, boys::Shape::kSingle, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64Device, boys::Shape::kAllOrders, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp64Device, boys::Shape::kAllN, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32Device, boys::Shape::kSingle, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32Device, boys::Shape::kAllOrders, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp32Device, boys::Shape::kAllN, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16Device, boys::Shape::kSingle, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16Device, boys::Shape::kAllOrders, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kFp16Device, boys::Shape::kAllN, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kBf16Device, boys::Shape::kSingle, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kBf16Device, boys::Shape::kAllOrders, boys::Device::kDevice>() &&
    ClassIsTheCommittedCombination<boys::Precision::kBf16Device, boys::Shape::kAllN, boys::Device::kDevice>();

static_assert(!(kCoarsestFiveInForce && kCoarsestClassTableInForce),
              "the defaults header in force names all five committed values and answers every class "
              "with the committed combination, so this build has chosen nothing: point "
              "BOYS_BUILD_DEFAULTS at a header that moves at least one of the seven or that carries "
              "a row moving a class, or unset it to build the committed configuration");

#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)
// The test's own override (tests/build_defaults_tuned.hpp), pinned by value so a configure that
// delivered a header claiming this name fails here rather than passing; the single-axis fixtures
// define neither this guard nor BOYS_BUILD_DEFAULTS_COMMITTED. One assertion per axis: pinning
// only the axes a fixture moves is how three macros came to be documented, replaced and read by nothing.
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the fixture's fit route is not in force");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kSplitClenshaw,
              "the fixture's evaluation scheme is not in force");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the fixture's packing axis is not in force");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kPlainReciprocal,
              "the fixture's division form is not in force");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kCoarsest,
              "the fixture's fit granularity is not in force");
#endif
#endif

// Address identity: two function addresses compare equal in a constant expression only when the two
// names are one instantiation, so the assertion below is a claim about the entry's declaration - its
// policy parameter defaulting to its own class's row - and not about two calls that agree today.
// Only the yes direction is constant there: distinct addresses are not, under the sanitizer.
template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

// The unnamed call is the build's policy, and that policy is its CLASS'S ROW: the table
// boys/boys_build_defaults.hpp carries, read through DefaultPolicy<Precision, Shape>. It returns that
// row's bits, and equals the seam's own five exactly where the row spells them, and not otherwise.
// One entry only: the double lane's all-orders ladder, the rest in tests/boys_build_defaults_test.cpp.
TEST(BackendTest, TheUnnamedCallIsTheDefaultThisBuildWasCompiledWith) {
    using boys::BoysAllOrders;

    // The class this entry belongs to, as the policy the build's table resolves it to.
    // Named through the library's own alias rather than spelled out: this is the name the
    // entry's policy parameter defaults to, so the comparison below is a claim about the
    // build's table and not about this line.
    using Class = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;

    // The point the seam's own five compose: the fallback a class the table carries no row
    // for resolves to, and the type the library names EvalPolicy<>. Not what an entry
    // resolves to in general - an entry resolves to its class's row, which is this
    // combination only where that row spells it.
    using Fallback = boys::EvalPolicy<>;

    static_assert(SameCall<&BoysAllOrders<>, &BoysAllOrders<Class>>,
                  "an entry that names no policy does not name its own class's row: the class is "
                  "the one the build's table carries for (kFp64, all-orders), and the unnamed "
                  "call is not that policy's call");

    // Which of the two the table answers this class with, read off the class's own policy so that it is
    // the build's answer and not this line's. Every axis is read, the region-B exponential included:
    // that is the axis a row can move while spelling the seam's five, and a flag reading five axes would
    // call such a row the seam's own combination and require the calls to agree where they must not.
    constexpr bool kClassRowIsTheSeamFive =
        Class::kRoute == Fallback::kRoute && Class::kScheme == Fallback::kScheme &&
        Class::kBudget == Fallback::kBudget && Class::kPack == Fallback::kPack &&
        Class::kGranularity == Fallback::kGranularity && Class::kDivision == Fallback::kDivision &&
        Class::kRegionBExp == Fallback::kRegionBExp;

    // Arguments across the domain's regions, at orders that span the ladder: 0 is the seed
    // every higher order is reached from, and kMaxBoysOrder the widest call the entries serve.
    constexpr int kOrders[] = {0, 1, 4, 12, boys::kMaxBoysOrder};
    constexpr double kArguments[] = {0.0, 1e-12, 1e-3, 0.5, 1.0, 3.0, 11.9, 12.0, 60.0, 120.0};

    std::size_t classMoved = 0;
    std::size_t fallbackMoved = 0;
    std::size_t cells = 0;

    for (const int nmax : kOrders) {
        for (const double x : kArguments) {
            std::array<double, boys::kMaxBoysOrder + 1> unnamed{};
            std::array<double, boys::kMaxBoysOrder + 1> classDefault{};
            std::array<double, boys::kMaxBoysOrder + 1> fallback{};

            BoysAllOrders(nmax, x, unnamed.data());
            BoysAllOrders<Class>(nmax, x, classDefault.data());
            BoysAllOrders<Fallback>(nmax, x, fallback.data());

            for (int order = 0; order <= nmax; ++order) {
                const std::size_t sorder = static_cast<std::size_t>(order);
                ++cells;

                if (std::bit_cast<std::uint64_t>(unnamed[sorder]) !=
                    std::bit_cast<std::uint64_t>(classDefault[sorder])) {
                    ++classMoved;
                }

                if (std::bit_cast<std::uint64_t>(unnamed[sorder]) !=
                    std::bit_cast<std::uint64_t>(fallback[sorder])) {
                    ++fallbackMoved;
                }
            }
        }
    }

    std::printf("boys: the fp64 all-orders class resolves to (route %d, %s, budget %d, %s, %s, "
                "%s, %s); that row %s the seam's five, and the unnamed call differs from the "
                "class's row's call in %zu of %zu values and from the seam's five's call in %zu "
                "of them\n",
                static_cast<int>(Class::kRoute),
                boys::EvalSchemeName(Class::kScheme),
                static_cast<int>(Class::kBudget),
                boys::PackAxisName(Class::kPack),
                boys::GranularityName(Class::kGranularity),
                boys::DivisionFormName(Class::kDivision),
                boys::RegionBExpName(Class::kRegionBExp),
                kClassRowIsTheSeamFive ? "is" : "is not",
                classMoved,
                cells,
                fallbackMoved);

    EXPECT_EQ(classMoved, 0u)
        << "an unnamed call to BoysAllOrders and a call at the policy the build's table carries "
           "for its class differ in "
        << classMoved << " of " << cells
        << " values: the entry's default argument is not its own class's row, so a caller that "
           "names no policy is answered by a policy the build's table does not name for that "
           "class";

    EXPECT_EQ(fallbackMoved == 0u, kClassRowIsTheSeamFive)
        << "the unnamed call and the seam's five's call differ in " << fallbackMoved
        << " value(s) while the build's table row for the class "
        << (kClassRowIsTheSeamFive ? "spells" : "does not spell")
        << " those five: an entry resolving through the seam's point where the table has moved "
           "the class agrees where it must not; one resolving through a row that spells the point "
           "differs where it must not; and a row naming a cell this class's entry does not carry "
           "is a cell the table states and the build does not honour, which reads here as the two "
           "calls agreeing";
}

// Naming the narrow partition is answered from its own tables, and a combination with no table for the
// named partition is refused where it is named rather than answered from another partition's fits -
// refusals that are static_asserts inside `RouteFit`, which a compiling test cannot exercise. What is
// pinned here is the default; the member itself is measured in the accuracy gate.
TEST(BackendTest, ThePartitionNamesRoundTrip) {
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kCoarsest), "shipped");
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kNarrow), "narrow");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kCoarsest),
                 boys::GranularityName(boys::FitGranularity::kNarrow));

    // A partition this build serves is a row of the enumeration: a name the enumeration does
    // not carry would print "unknown", which is the name of no partition at all.
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kUniform), "uniform");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kUniform), "unknown");
}

// The uniform partition is a row of that table, not a value the header names on the side; the row's
// declarations are checked against what the build answers. The last check calls the entry rather than
// reading the row: a partition served from another partition's tables is invisible in the row's own
// fields, as this build found once at RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kUniform>.
TEST(BackendTest, TheUniformPartitionDeclaresWhatTheBuildServes) {
    const std::span<const boys::FitGranularityInfo> rows = boys::BoysFitGranularities();

    ASSERT_GE(rows.size(), 3u) << "the enumeration does not carry the uniform partition";

    const boys::FitGranularityInfo* uniform = nullptr;

    for (std::size_t i = 0; i < rows.size(); ++i)
    {
        EXPECT_EQ(static_cast<std::size_t>(rows[i].granularity), i)
            << "the rows are not in enumerator order, which is the order a report reads them in";

        if (rows[i].granularity == boys::FitGranularity::kUniform)
        {
            uniform = &rows[i];
        }
    }

    ASSERT_NE(uniform, nullptr) << "the uniform partition names no row of the enumeration";
    EXPECT_STREQ(uniform->name, boys::GranularityName(boys::FitGranularity::kUniform));
    EXPECT_STRNE(uniform->name, "unknown");

    // Both routes and both packing axes, as the row states them.
    EXPECT_TRUE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kChebyshev));
    EXPECT_TRUE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kRationalMinimax));
    EXPECT_TRUE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kArguments));
    EXPECT_TRUE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kOrders));

    // The row's fields against the accessor that answers a caller, over every combination of
    // the other axes: a combination the accessor serves and the row does not claim is a claim
    // the row is missing, and one the row claims and the accessor refuses is a route or an
    // axis the build does not have.
    std::size_t served = 0;
    std::size_t claimedCount = 0;

    for (const boys::FitRouteInfo& route : boys::BoysFitRoutes())
    {
        for (const boys::EvalSchemeInfo& scheme : boys::BoysEvalSchemes())
        {
            for (const boys::PackAxisInfo& axis : boys::BoysPackAxes())
            {
                const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
                    boys::Precision::kFp64, route.route, scheme.scheme, axis.axis,
                    boys::FitGranularity::kUniform);
                const bool claimed =
                    boys::FitGranularityHasRoute(*uniform, route.route) &&
                    boys::FitGranularityHasAxis(*uniform, axis.axis);

                EXPECT_EQ(figure.available, claimed)
                    << "the uniform row says " << (claimed ? "served" : "refused")
                    << " at route " << static_cast<int>(route.route) << ", scheme "
                    << static_cast<int>(scheme.scheme) << ", axis "
                    << static_cast<int>(axis.axis) << " and the accessor answers "
                    << (figure.available ? "served" : "no") << ": " << figure.reason;

                if (claimed)
                {
                    ++claimedCount;
                }

                if (figure.available)
                {
                    ++served;
                    EXPECT_GT(figure.value, 0.0) << "a served combination carries no figure";
                    EXPECT_NE(figure.source[0], '\0')
                        << "a served combination does not say where its figure comes from";
                    EXPECT_EQ(figure.reason[0], '\0')
                        << "a served combination carries the reason of a refusal: "
                        << figure.reason;
                }
                else
                {
                    EXPECT_NE(figure.reason[0], '\0')
                        << "a refused combination is refused without a reason";
                }
            }
        }
    }

    // The count is held to the rows' own claim rather than to a literal - it was 64, and it
    // went stale as soon as the rational member's cells were served. `claimed` comes from the
    // row's own declarations, whose routes and axes are asserted above, so the equality
    // below is the same claim without the snapshot.
    EXPECT_GT(claimedCount, 0u) << "the uniform row claims nothing at all";
    EXPECT_EQ(served, claimedCount)
        << "the accessor and the uniform row disagree over the cross: the rows claim "
        << claimedCount << " combination(s) and the accessor serves " << served;

    // And the entry the row claims runs, reads its own table, and is not the narrow one renamed.
    constexpr int kNmax = boys::kMaxBoysOrder;
    constexpr std::array<double, 7> kArguments = {0.5, 1.0, 2.0, 6.0, 12.0, 24.0, 34.0};

    using UniformPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                           boys::EvalScheme::kHorner,
                                           boys::BoysBudget::kFloat,
                                           boys::PackAxis::kArguments,
                                           boys::FitGranularity::kUniform>;
    using NarrowPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                          boys::EvalScheme::kHorner,
                                          boys::BoysBudget::kFloat,
                                          boys::PackAxis::kArguments,
                                          boys::FitGranularity::kNarrow>;

    std::array<double, kNmax + 1> values{};
    std::array<double, kNmax + 1> narrow{};
    std::size_t moved = 0;

    for (const double x : kArguments)
    {
        boys::BoysAllOrders<UniformPolicy>(kNmax, x, values.data());
        boys::BoysAllOrders<NarrowPolicy>(kNmax, x, narrow.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            EXPECT_TRUE(std::isfinite(values[static_cast<std::size_t>(n)]))
                << "the uniform entry returned " << values[static_cast<std::size_t>(n)]
                << " at n = " << n << ", x = " << x;

            moved += values[static_cast<std::size_t>(n)] != narrow[static_cast<std::size_t>(n)]
                         ? 1u
                         : 0u;
        }
    }

    EXPECT_GT(moved, 0u)
        << "every value of the uniform entry is the narrow partition's bit for bit, over "
        << kArguments.size() << " arguments and " << (kNmax + 1)
        << " orders: a uniform policy is being answered from another partition's tables";

    // The same check for the rational route, whose substitution risk is real: its pairs are per
    // interval and the narrow member's per narrow piece, so an entry answering a uniform rational
    // policy from the narrow pairs would return certified numbers under the grid's name. The two are
    // the same family read over different partitions, which is what makes the comparison meaningful.
    using UniformRatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                              boys::EvalScheme::kHorner,
                                              boys::BoysBudget::kFloat,
                                              boys::PackAxis::kArguments,
                                              boys::FitGranularity::kUniform>;
    using NarrowRatPolicy = boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                                             boys::EvalScheme::kHorner,
                                             boys::BoysBudget::kFloat,
                                             boys::PackAxis::kArguments,
                                             boys::FitGranularity::kNarrow>;

    std::array<double, kNmax + 1> rational{};
    std::array<double, kNmax + 1> narrowRational{};
    std::size_t movedRational = 0;

    for (const double x : kArguments)
    {
        boys::BoysAllOrders<UniformRatPolicy>(kNmax, x, rational.data());
        boys::BoysAllOrders<NarrowRatPolicy>(kNmax, x, narrowRational.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            EXPECT_TRUE(std::isfinite(rational[static_cast<std::size_t>(n)]))
                << "the uniform rational entry returned " << rational[static_cast<std::size_t>(n)]
                << " at n = " << n << ", x = " << x;

            movedRational +=
                rational[static_cast<std::size_t>(n)] != narrowRational[static_cast<std::size_t>(n)]
                    ? 1u
                    : 0u;
        }
    }

    EXPECT_GT(movedRational, 0u)
        << "every value of the uniform rational entry is the narrow partition's bit for bit, over "
        << kArguments.size() << " arguments and " << (kNmax + 1)
        << " orders: the grid's rational member is being answered from the narrow partition's "
           "pairs";
}

// The division forms report themselves as the other axes do: one row per member, in
// enumerator order, named by the library's own function, with the default one of the rows.
// A row this accessor does not carry is a member no report can print.
TEST(BackendTest, TheDivisionFormAxisNamesItsMembers) {
    const std::span<const boys::DivisionFormInfo> forms = boys::BoysDivisionForms();

    ASSERT_EQ(forms.size(), 3u);

    bool carriesDefault = false;

    for (std::size_t i = 0; i < forms.size(); ++i)
    {
        const boys::DivisionFormInfo& row = forms[i];

        EXPECT_EQ(static_cast<std::size_t>(row.form), i) << "the rows are not in enumerator order";
        EXPECT_STREQ(row.name, boys::DivisionFormName(row.form));
        EXPECT_STRNE(row.name, "unknown");

        if (row.form == boys::kDefaultDivisionForm)
        {
            carriesDefault = true;
        }
    }

    EXPECT_TRUE(carriesDefault) << "the default form is not one of the rows this build reports";
}

// What each division form governs, measured not read off the source: the refined form is bit-identical
// to exact division on both lanes - the claim every published per-region figure rests on. The plain
// form reaches the downward ladder (divisor: the step's constant, not the argument) and the single-
// precision lane's too, where its published figure for the form covers every cell the sweep read.
TEST(BackendTest, TheDivisionFormReachesTheLaddersItDocuments) {
    using boys::BoysAllOrders;
    using boys::BoysAllOrdersF32;
    using boys::DivisionForm;
    using boys::EvalPolicy;

    constexpr auto kRoute = boys::kDefaultFitRoute;
    constexpr auto kScheme = boys::kDefaultEvalScheme;
    constexpr auto kBudget = boys::BoysBudget::kFloat;
    constexpr auto kPack = boys::kDefaultPackAxis;
    constexpr auto kGran = boys::kDefaultFitGranularity;

    using DExact = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran, DivisionForm::kExactDivision>;
    using DPlain = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran, DivisionForm::kPlainReciprocal>;
    using DRefined =
        EvalPolicy<kRoute, kScheme, kBudget, kPack, kGran, DivisionForm::kRefinedReciprocal>;

    // Arguments below each lane's kX0, where the downward recursion runs, and above it, where the upward
    // ladders do. The downward set carries both of the accuracy gate's own cells where the plain form's
    // reciprocal reaches furthest on the single-precision lane - x = 7 and x = 9.74054909 - so the figure
    // that form is held to is read where it was found rather than at a convenient point.
    constexpr double kDown[] = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 9.74054909, 11.5};
    constexpr double kUp[] = {12.0, 15.0, 20.0, 28.9, 29.0, 40.0, 60.0, 120.0};
    constexpr int kNmax = boys::kMaxBoysOrder;

    // The figure a lane publishes for the plain reciprocal, read off the library's own contract row -
    // the row the accuracy accessor, the accessor's documentation and the gates all read - plus the term
    // that row carries beside its base for this form. It is not a tolerance chosen here: it is the number
    // the library publishes for the arithmetic the sweep below runs.
    const auto plainFormFigure = [](boys::Precision precision) {
        for (const boys::LaneContractInfo& row : boys::BoysLaneContracts()) {
            if (row.precision == precision) {
                return row.bound + row.plainAdditive;
            }
        }

        return 0.0;
    };

    const double kFloatPlainFigure = plainFormFigure(boys::Precision::kFp32);

    std::size_t refinedMoved = 0;
    std::size_t doubleDownMoved = 0;
    std::size_t doubleUpMoved = 0;
    std::size_t floatDownMoved = 0;
    std::size_t floatUpMoved = 0;
    std::size_t floatOutsideFigure = 0;
    std::size_t cells = 0;

    std::array<double, kNmax + 1> dExact{};
    std::array<double, kNmax + 1> dPlain{};
    std::array<double, kNmax + 1> dRefined{};
    std::array<float, kNmax + 1> fExact{};
    std::array<float, kNmax + 1> fPlain{};
    std::array<float, kNmax + 1> fRefined{};
    std::array<double, kNmax + 1> dReference{};

    const auto sweepDouble = [&](double x, bool downward) {
        BoysAllOrders<DExact>(kNmax, x, dExact.data());
        BoysAllOrders<DPlain>(kNmax, x, dPlain.data());
        BoysAllOrders<DRefined>(kNmax, x, dRefined.data());

        for (int n = 0; n <= kNmax; ++n) {
            const std::size_t sn = static_cast<std::size_t>(n);
            ++cells;

            if (std::bit_cast<std::uint64_t>(dRefined[sn]) !=
                std::bit_cast<std::uint64_t>(dExact[sn])) {
                ++refinedMoved;
            }

            if (std::bit_cast<std::uint64_t>(dPlain[sn]) !=
                std::bit_cast<std::uint64_t>(dExact[sn])) {
                (downward ? doubleDownMoved : doubleUpMoved) += 1;
            }
        }
    };

    const auto sweepFloat = [&](float x, bool downward) {
        BoysAllOrdersF32<DExact>(kNmax, x, fExact.data());
        BoysAllOrdersF32<DPlain>(kNmax, x, fPlain.data());
        BoysAllOrdersF32<DRefined>(kNmax, x, fRefined.data());

        // The reference this lane's plain form is measured against: the double lane at the same argument,
        // whose own published figure is 5.5e-14 - five to six digits inside the bar the single lane is read
        // at, so a difference between the two is this lane's error and not the reference's. The argument is
        // the float one widened, not the double the sweep walked in with, so the float's own rounding is out.
        BoysAllOrders<DExact>(kNmax, static_cast<double>(x), dReference.data());

        for (int n = 0; n <= kNmax; ++n) {
            const std::size_t sn = static_cast<std::size_t>(n);
            ++cells;

            if (std::bit_cast<std::uint32_t>(fRefined[sn]) !=
                std::bit_cast<std::uint32_t>(fExact[sn])) {
                ++refinedMoved;
            }

            if (std::bit_cast<std::uint32_t>(fPlain[sn]) !=
                std::bit_cast<std::uint32_t>(fExact[sn])) {
                (downward ? floatDownMoved : floatUpMoved) += 1;
            }

            // The plain form's own cells, held to the figure the lane publishes for that form:
            // measured on every cell and not only on the cells the form moved, because the
            // claim is that what the form returns is inside the number the lane states for it,
            // and a cell the form did not move is a cell it still answered.
            if (std::abs(static_cast<double>(fPlain[sn]) - dReference[sn]) > kFloatPlainFigure)
            {
                ++floatOutsideFigure;
            }
        }
    };

    for (const double x : kDown) {
        sweepDouble(x, true);
        sweepFloat(static_cast<float>(x), true);
    }

    for (const double x : kUp) {
        sweepDouble(x, false);
        sweepFloat(static_cast<float>(x), false);
    }

    std::printf("boys: the division form over %zu cell(s) per pair - refined against exact moved "
                "%zu;\n  the plain form against exact moved %zu in the double lane's downward "
                "ladder and %zu\n  in its upward ones, %zu in the single lane's downward ladder "
                "and %zu in its upward ones;\n  the single lane's plain form is outside the %g "
                "the lane publishes for it at %zu cell(s)\n",
                cells,
                refinedMoved,
                doubleDownMoved,
                doubleUpMoved,
                floatDownMoved,
                floatUpMoved,
                kFloatPlainFigure,
                floatOutsideFigure);

    EXPECT_EQ(refinedMoved, 0u)
        << "the refined form is documented as bit-identical to exact division, and it is not";

    EXPECT_GT(doubleDownMoved, 0u)
        << "the plain form reaches no cell of the double lane's downward ladder, whose divisor is "
           "the step's constant: a caller naming it is served exact division there and nothing "
           "reports it";

    EXPECT_GT(doubleUpMoved, 0u) << "the plain form reaches no cell of the double lane's upward "
                                    "ladders, so the axis selects no arithmetic there";

    EXPECT_GT(floatDownMoved, 0u)
        << "the plain form reaches no cell of the single lane's downward ladder, which is the "
           "carve-out that ladder used to keep: that lane names the form it divides in, and a "
           "caller naming this one was served exact division there with nothing reporting it";

    EXPECT_GT(floatUpMoved, 0u)
        << "the plain form reaches no cell of the single lane's upward ladders either, so the two "
           "counts above would both be zero for want of an axis rather than for a carve-out";

    EXPECT_EQ(floatOutsideFigure, 0u)
        << "the single lane's plain form delivered outside the figure the lane publishes for it ("
        << kFloatPlainFigure << ") at " << floatOutsideFigure
        << " cell(s): the form is served on that lane, so what it returns has to be inside the "
           "lane's own figure for it, and that figure is what a caller reads";
}

// The partition question, asked of a fit rather than of a run: do the derived families carry the
// grid's own table? They are written against two partitions - their granularity parameter a two-case
// conditional with no answer for the grid - so a path resolving a partition through one of them answers
// the grid's policy from the narrow member, unreported: the question detail::FitAnswersPartition asks once.
static_assert(boys::detail::kFitCarriesUniform<boys::detail::UniformFit<boys::EvalScheme::kHorner>>,
              "the Chebyshev member over the grid is the one fit that carries it");
static_assert(boys::detail::kFitCarriesUniform<boys::detail::RationalFitUniform>,
              "the rational member over the grid is the other fit that carries it");

static_assert(!boys::detail::kFitCarriesUniform<
                  boys::detail::ChebyshevFit<boys::EvalScheme::kHorner,
                                             boys::FitGranularity::kCoarsest>>,
              "the derived Chebyshev family has no grid table: its granularity parameter is a "
              "two-case conditional, and an answer for the grid here is the substitution the "
              "trait exists to catch");
static_assert(!boys::detail::kFitCarriesUniform<boys::detail::RationalFit>,
              "the shipped rational member has no grid table");
static_assert(!boys::detail::kFitCarriesUniform<boys::detail::RationalFitNarrow>,
              "the narrow rational member has no grid table");

namespace {

using GridFit = boys::detail::UniformFit<boys::EvalScheme::kHorner>;
using DerivedFit =
    boys::detail::ChebyshevFit<boys::EvalScheme::kHorner, boys::FitGranularity::kCoarsest>;

static_assert(boys::detail::FitAnswersPartition<GridFit>(boys::FitGranularity::kUniform),
              "the grid's member answers the grid");

// A grid member answers the grid and nothing else: reading it as the shipped or the narrow
// partition would be a substitution in the other direction, and the same one.
static_assert(!boys::detail::FitAnswersPartition<GridFit>(boys::FitGranularity::kCoarsest));
static_assert(!boys::detail::FitAnswersPartition<GridFit>(boys::FitGranularity::kNarrow));

static_assert(boys::detail::FitAnswersPartition<DerivedFit>(boys::FitGranularity::kCoarsest),
              "the derived family's own two partitions are what it answers");
static_assert(boys::detail::FitAnswersPartition<DerivedFit>(boys::FitGranularity::kNarrow));
static_assert(!boys::detail::FitAnswersPartition<DerivedFit>(boys::FitGranularity::kUniform),
              "the derived family does not answer the grid, and a path resolving a policy "
              "naming the grid through it must say so rather than read the narrow member");

// A value outside the enumeration fails closed at both members. A two-case conditional would
// hand it to the arm its `else` names - the narrow member for the derived families, the
// shipped one for the float lane's twins - so the predicate's answer is the one that turns
// such a value into a refusal wherever a partition is resolved rather than into a fit.
static_assert(!boys::detail::FitAnswersPartition<DerivedFit>(
                  static_cast<boys::FitGranularity>(200)),
              "a granularity outside the enumeration is answered by no fit");
static_assert(!boys::detail::FitAnswersPartition<GridFit>(static_cast<boys::FitGranularity>(200)),
              "a granularity outside the enumeration is answered by no fit");

} // namespace

// The guard's bite: a policy naming the grid is served by the batched shapes only because each routes
// it to the per-argument body, which reads the grid's table, so this file fails to compile if that
// routing is removed or the predicate weakened - the call below then hits the partitioned path, the
// one property a run-time test cannot have: a substitution that compiles is measured as a value.
TEST(BackendTest, TheGuardedBatchedEntriesReadTheGridsOwnTable) {
    using UniformPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                           boys::EvalScheme::kHorner,
                                           boys::BoysBudget::kFloat,
                                           boys::PackAxis::kArguments,
                                           boys::FitGranularity::kUniform>;
    using NarrowPolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                          boys::EvalScheme::kHorner,
                                          boys::BoysBudget::kFloat,
                                          boys::PackAxis::kArguments,
                                          boys::FitGranularity::kNarrow>;

    // One argument under the grid's join, one inside region B below it, and two above it -
    // where the double lane's grid reaches past the end of region B, so the entry there is
    // the asymptotic form and reads no fit at all.
    constexpr std::array<double, 5> kArguments = {0.25, 1.0, 5.5, 12.0, 27.0};
    constexpr int kNmax = boys::kMaxBoysOrder;

    std::vector<double> planes(kArguments.size() * static_cast<std::size_t>(kNmax + 1), 0.0);
    std::vector<std::size_t> workspace(boys::BoysAllNWorkspaceSize(kArguments.size()));

    boys::BoysAllN<UniformPolicy>(kNmax,
                                  kArguments.data(),
                                  planes.data(),
                                  kArguments.size(),
                                  workspace.data());

    std::size_t moved = 0;

    for (std::size_t i = 0; i < kArguments.size(); ++i)
    {
        std::array<double, kNmax + 1> perOrder{};
        std::array<double, kNmax + 1> narrow{};

        boys::BoysAllOrders<UniformPolicy>(kNmax, kArguments[i], perOrder.data());
        boys::BoysAllOrders<NarrowPolicy>(kNmax, kArguments[i], narrow.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const double plane = planes[static_cast<std::size_t>(n) * kArguments.size() + i];

            EXPECT_EQ(plane, perOrder[static_cast<std::size_t>(n)])
                << "the batched entry's grid path is the per-argument body's own reading, and "
                << "at n = " << n << ", x = " << kArguments[i] << " the two part: the entry a "
                << "policy naming the grid is served by is not the body that reads the grid";

            moved += plane != narrow[static_cast<std::size_t>(n)] ? 1u : 0u;
        }
    }

    EXPECT_GT(moved, 0u)
        << "every value of the batched entry's grid path is the narrow partition's bit for bit, "
        << "over " << kArguments.size() << " arguments and " << (kNmax + 1)
        << " orders: a policy naming the grid is being answered from another partition's tables";

    // The fixed-order entry, whose routing asks the same question of the same read: one order
    // at every argument of the array.
    std::vector<double> fixed(kArguments.size(), 0.0);

    boys::BoysFixedN<UniformPolicy>(kNmax / 2,
                                    kArguments.data(),
                                    fixed.data(),
                                    kArguments.size());

    for (std::size_t i = 0; i < kArguments.size(); ++i)
    {
        // Parenthesised: the template argument list carries a comma, which the assertion's
        // own macro would otherwise read as a second argument of its own.
        EXPECT_EQ(fixed[i], (boys::BoysSingle<UniformPolicy>(kNmax / 2, kArguments[i])))
            << "the fixed-order entry's grid path is the single-order body's own reading, and "
            << "they part at x = " << kArguments[i];
    }
}
