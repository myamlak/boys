// The arithmetic-backend contract (boys/backend.hpp): which arithmetic the build
// carries, and what each one's multiply-adds round. MulAdd is fused and MulSub is
// not - the split-precision lane evaluates its recurrence at the two-rounding
// reading on purpose, and a build that fused the bare form would move it.
// Contracts() is asserted against a measurement in this translation unit.

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

// Whether a bare product-plus-add is one rounding here. The operands are volatile
// so the expression is evaluated rather than folded away; whether the multiply-add
// is contracted is the question, and the addend makes the two answers differ.
//
// The fused value comes from the standard function rather than the backend, whose
// own multiply-add is under test: asking it here would have it confirm itself.
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

    // A route in force is the selection unless the build contracts the bare form, so a
    // reported separate route is printed only where the two roundings happen.
    for (const BackendInfo& info : backends) {
        if (info.route == MulAddRoute::kSeparate) {
            EXPECT_FALSE(info.contracts) << info.name;
        }
    }
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

// The reported contraction flag is the one the header gives for the arithmetic carrying
// that name. Contraction belongs to a compiled translation unit, and the library has more
// than one flag context: src/boys_simd.cpp carries the packed flags, everything else does
// not. A table measured in the packed unit would print that unit's answer beside the
// scalar lanes' values, so the comparison is made where a scalar kernel would be compiled,
// which is what this file is.
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

// ---------------------------------------------------------------------------
// The selection axes on the policy, and the five names the build fixes
// ---------------------------------------------------------------------------
// Every axis the entries select is one field of EvalPolicy with its own default. The pins
// say which member each default names, so a move of one is a decision this test states
// rather than a value that follows silently, and each axis reports itself by name.
//
// The engine budget is the library's: what a single-precision engine computes at, and no
// build replaces it. The names an unnamed call resolves to are the BUILD's, which is what
// boys/boys_build_defaults.hpp exists for: a build that has measured its own machine
// replaces that header with its own set (the CMake option is BOYS_BUILD_DEFAULTS,
// CONTRIBUTING.md). The pins below are the shipped names where the shipped header is in
// force - the configuration every bound in this repository was measured at - and what has
// to hold instead where it is not.
//
// ALL FIVE AXES ARE READ FROM THAT HEADER. Two of them always were - the fit route and the
// evaluation scheme - and the other three, the packing axis, the division form and the
// partition, were literals in backend.hpp until they were wired to the same seam. The pins
// below cover all five, because a pin on some of them is how three macros came to be
// documented, replaced and read by nothing.
static_assert(boys::EvalPolicy<>{}.kBudget == boys::BoysBudget::kFloat,
              "the default engine budget moved");

#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
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
// A replacement is read instead of the committed file rather than beside it, so the names
// it carries are this build's. A replacement naming the shipped set, and a seam that
// stopped delivering the file, both come out as the committed values.
constexpr bool kShippedDefaultsInForce =
    boys::kDefaultFitRoute == boys::FitRoute::kChebyshev &&
    boys::kDefaultEvalScheme == boys::EvalScheme::kHorner;

static_assert(!kShippedDefaultsInForce,
              "the defaults header in force names the shipped route and scheme, so this build "
              "has chosen nothing: point BOYS_BUILD_DEFAULTS at a header that moves at least one "
              "axis, or unset it to build the shipped configuration");

#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)
// The test's own override (tests/build_defaults_tuned.hpp), pinned by value so a configure
// that delivered some other header fails here rather than passing.
// One assertion per axis, whichever value the fixture sets it to. Pinning only
// the axes it happens to move is how the seam's three dead macros went unnoticed:
// the fixture moved the scheme, copied the library's literals for the rest, and
// nothing here could tell a move that took effect from a macro nothing read.
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the fixture's fit route is not in force");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kSplitClenshaw,
              "the fixture's evaluation scheme is not in force");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the fixture's packing axis is not in force");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kPlainReciprocal,
              "the fixture's division form is not in force");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kShipped,
              "the fixture's fit granularity is not in force");
#endif
#endif

// The unnamed call is the build's policy, and its values are the shipped policy's only
// where the shipped header is in force. The comparison is made through the entry a caller
// writes - BoysAllOrders with no template argument against the same entry called at the
// shipped policy explicitly - so what is measured is what a call site gets rather than
// what a constant holds.
TEST(BackendTest, TheUnnamedCallIsTheDefaultThisBuildWasCompiledWith) {
    using boys::BoysAllOrders;
    using Shipped = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                     boys::EvalScheme::kHorner,
                                     boys::BoysBudget::kFloat,
                                     boys::PackAxis::kArguments,
                                     boys::FitGranularity::kNarrow,
                                     boys::DivisionForm::kRefinedReciprocal>;

    // Arguments across the domain's regions, at orders that span the ladder: 0 is the seed
    // every higher order is reached from, and kMaxBoysOrder the widest call the entries serve.
    constexpr int kOrders[] = {0, 1, 4, 12, boys::kMaxBoysOrder};
    constexpr double kArguments[] = {0.0, 1e-12, 1e-3, 0.5, 1.0, 3.0, 11.9, 12.0, 60.0, 120.0};

    std::size_t moved = 0;
    std::size_t cells = 0;

    for (const int nmax : kOrders) {
        for (const double x : kArguments) {
            std::array<double, boys::kMaxBoysOrder + 1> built{};
            std::array<double, boys::kMaxBoysOrder + 1> shipped{};

            BoysAllOrders(nmax, x, built.data());
            BoysAllOrders<boys::kBoysFullAccuracyMultiplier, Shipped>(nmax, x, shipped.data());

            for (int order = 0; order <= nmax; ++order) {
                ++cells;

                if (std::bit_cast<std::uint64_t>(built[order]) !=
                    std::bit_cast<std::uint64_t>(shipped[order])) {
                    ++moved;
                }
            }
        }
    }

    std::printf("boys: this build's default policy is %s the shipped policy type; the unnamed "
                "call and the shipped policy's call differ in %zu of %zu values\n",
                std::is_same_v<boys::DefaultPolicyFp64, Shipped> ? "the same as" : "different from",
                moved,
                cells);

#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
    // One call under this header, so a value that differs here is a translation
    // unit and a library instantiation that did not resolve the same defaults.
    EXPECT_EQ(moved, 0u) << "the committed defaults header is in force, so an unnamed call is the "
                            "shipped policy's call and no value may differ";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)
    // The fixture moves the scheme, so the values are not the shipped ones and the
    // assertion says an unnamed call follows the build's header.
    EXPECT_GT(moved, 0u) << "the fixture moves the scheme, so an unnamed call cannot hand back "
                            "the shipped policy's values";
#endif
}

// Naming the narrow partition is answered from its own tables; combinations with no narrow
// table are refused where they are named rather than answered from the shipped one. Those
// refusals are static_asserts inside RouteFit, the relaxed rungs' RequireShippedPartition
// and the single-precision lanes, so a test that has to compile cannot exercise one: what
// is pinned here is the default. The member itself is measured in the accuracy gate.
TEST(BackendTest, ThePartitionNamesRoundTrip) {
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kShipped), "shipped");
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kNarrow), "narrow");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kShipped),
                 boys::GranularityName(boys::FitGranularity::kNarrow));

    // A partition this build serves is a row of the enumeration: a name the enumeration does
    // not carry would print "unknown", which is the name of no partition at all.
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kUniform), "uniform");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kUniform), "unknown");
}

// The uniform partition is a row of that table rather than a value the header names on the
// side; what the row declares is checked against what the build answers, member by member.
//
// The last check calls the entry rather than reading the row: a partition served from
// another partition's tables is a defect the row's own fields cannot show, because those
// fields would be the ones the substitution was made to satisfy. This build refused that
// shape once, at the fit selector in backend.hpp, RouteFit<FitRoute::kRationalMinimax,
// kScheme, FitGranularity::kUniform>, whose first version the narrow partition answered;
// what is pinned is that each of the two members is the one it names, not a second name for
// the narrow fits.
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

    // Both routes, both packing axes and every rung of the enumeration, as the row states them.
    EXPECT_TRUE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kChebyshev));
    EXPECT_TRUE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kRationalMinimax));
    EXPECT_TRUE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kArguments));
    EXPECT_TRUE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kOrders));
    EXPECT_EQ(uniform->rungs, static_cast<int>(boys::AccuracyTier::kRelaxed65536) + 1)
        << "a rung of the uniform partition is the stored cells read uncut, so the row claims "
           "every rung the enumeration names";

    // The row's fields against the accessor that answers a caller, over every combination of
    // the other axes: a combination the accessor serves and the row does not claim is a claim
    // the row is missing, and one the row claims and the accessor refuses is a route or an
    // axis the build does not have.
    //
    // The rung count is the row's own, for either route: a rung of this partition is the
    // stored cells read uncut, and the rational member's pairs are stored and read at every
    // multiplier by the same reading.
    std::size_t served = 0;
    std::size_t claimedCount = 0;

    for (const boys::FitRouteInfo& route : boys::BoysFitRoutes())
    {
        for (const boys::EvalSchemeInfo& scheme : boys::BoysEvalSchemes())
        {
            for (const boys::PackAxisInfo& axis : boys::BoysPackAxes())
            {
                for (int raw = 0; raw <= static_cast<int>(boys::AccuracyTier::kRelaxed65536);
                     ++raw)
                {
                    const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
                        boys::Precision::kFp64, route.route, scheme.scheme, axis.axis,
                        boys::FitGranularity::kUniform, static_cast<boys::AccuracyTier>(raw));
                    const bool claimed =
                        boys::FitGranularityHasRoute(*uniform, route.route) &&
                        boys::FitGranularityHasAxis(*uniform, axis.axis) && raw < uniform->rungs;

                    EXPECT_EQ(figure.available, claimed)
                        << "the uniform row says " << (claimed ? "served" : "refused")
                        << " at route " << static_cast<int>(route.route) << ", scheme "
                        << static_cast<int>(scheme.scheme) << ", axis "
                        << static_cast<int>(axis.axis) << ", m = "
                        << boys::AccuracyMultiplier(static_cast<boys::AccuracyTier>(raw))
                        << " and the accessor answers " << (figure.available ? "served" : "no")
                        << ": " << figure.reason;

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
    }

    // The count is held to the rows' own claim rather than to a literal - it was 64, and it
    // went stale as soon as the rational member's rungs were served. `claimed` comes from the
    // row's own declarations, whose rungs, routes and axes are asserted above, so the equality
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
        boys::BoysAllOrders<1.0, UniformPolicy>(kNmax, x, values.data());
        boys::BoysAllOrders<1.0, NarrowPolicy>(kNmax, x, narrow.data());

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

    // The same check for the rational route, whose substitution risk is real: its pairs are
    // per interval and the narrow member's per narrow piece, so an entry answering a uniform
    // rational policy from the narrow pairs would return certified numbers under the grid's
    // name. The two members are the same family read over different partitions, which is what
    // makes the comparison meaningful.
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
        boys::BoysAllOrders<1.0, UniformRatPolicy>(kNmax, x, rational.data());
        boys::BoysAllOrders<1.0, NarrowRatPolicy>(kNmax, x, narrowRational.data());

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

// What each division form actually governs, measured rather than read off the source:
//
//  - the refined form is bit-identical to exact division, at every order and every argument,
//    on both lanes. That is the claim the default form rests on: every published per-region
//    figure is stated for the exact arithmetic, and the refined form has to deliver those
//    values to carry that figure;
//  - the plain form reaches the downward ladder, whose divisor is the step's constant rather
//    than the argument. A caller naming the plain form was once served exact division there
//    and nothing reported it, so a zero count here is that back again;
//  - the plain form does NOT reach the single-precision lane's downward ladder. That lane's
//    figure is one number for every form, and the plain form's reciprocal at that step takes
//    it outside that number. The lane's upward ladders do take the form (the fourth count),
//    so the two together separate "not applied on this lane" from "not applied on this
//    ladder".
//
// The counts are printed because a count is the measurement; the assertions are on relations
// between them and not on the values.
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

    // Arguments below each lane's kX0, where the downward recursion runs, and above it, where
    // the upward ladders do. The largest downward argument is the accuracy gate's own cell
    // where the plain form's reciprocal leaves the float lane's figure, so the carve-out is
    // asserted where it was found rather than at a convenient point.
    constexpr double kDown[] = {0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 7.0, 9.74054909, 11.5};
    constexpr double kUp[] = {12.0, 15.0, 20.0, 28.9, 29.0, 40.0, 60.0, 120.0};
    constexpr int kNmax = boys::kMaxBoysOrder;

    std::size_t refinedMoved = 0;
    std::size_t doubleDownMoved = 0;
    std::size_t doubleUpMoved = 0;
    std::size_t floatDownMoved = 0;
    std::size_t floatUpMoved = 0;
    std::size_t cells = 0;

    std::array<double, kNmax + 1> dExact{};
    std::array<double, kNmax + 1> dPlain{};
    std::array<double, kNmax + 1> dRefined{};
    std::array<float, kNmax + 1> fExact{};
    std::array<float, kNmax + 1> fPlain{};
    std::array<float, kNmax + 1> fRefined{};

    const auto sweepDouble = [&](double x, bool downward) {
        BoysAllOrders<boys::kBoysFullAccuracyMultiplier, DExact>(kNmax, x, dExact.data());
        BoysAllOrders<boys::kBoysFullAccuracyMultiplier, DPlain>(kNmax, x, dPlain.data());
        BoysAllOrders<boys::kBoysFullAccuracyMultiplier, DRefined>(kNmax, x, dRefined.data());

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
        BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, DExact>(kNmax, x, fExact.data());
        BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, DPlain>(kNmax, x, fPlain.data());
        BoysAllOrdersF32<boys::kBoysFullAccuracyMultiplier, DRefined>(kNmax, x, fRefined.data());

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
                "and %zu in its upward ones\n",
                cells,
                refinedMoved,
                doubleDownMoved,
                doubleUpMoved,
                floatDownMoved,
                floatUpMoved);

    EXPECT_EQ(refinedMoved, 0u)
        << "the refined form is documented as bit-identical to exact division, and it is not";

    EXPECT_GT(doubleDownMoved, 0u)
        << "the plain form reaches no cell of the double lane's downward ladder, whose divisor is "
           "the step's constant: a caller naming it is served exact division there and nothing "
           "reports it";

    EXPECT_GT(doubleUpMoved, 0u) << "the plain form reaches no cell of the double lane's upward "
                                    "ladders, so the axis selects no arithmetic there";

    EXPECT_EQ(floatDownMoved, 0u)
        << "the plain form has reached the single lane's downward ladder, which the axis's "
           "documentation says it does not - and does not because serving it there takes the "
           "ladder outside the 1.5e-7 the lane publishes. Removing this carve-out is a "
           "measurement and not an edit: the lane's figure has to gain a form dimension first";

    EXPECT_GT(floatUpMoved, 0u)
        << "the plain form reaches no cell of the single lane either, so the two counts above "
           "would both be zero for want of an axis rather than for a carve-out";
}
