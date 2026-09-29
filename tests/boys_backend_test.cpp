// The arithmetic-backend contract (boys/backend.hpp): which arithmetic the
// build carries, and what each one's multiply-adds round.
//
// The two facts that matter to a caller holding a number are here, because
// they are what a lane's published figure rests on and neither is visible in
// the value itself. The first is that MulAdd is fused and MulSub is not — the
// split-precision lane evaluates its recurrence at the two-rounding reading
// on purpose, and a build that fused the bare form would move it. The second
// is Contracts(), which reports whether a bare product-plus-add in this
// arithmetic is one rounding in this build; it is asserted against a
// measurement made in this translation unit, not against an expectation.

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

// The concept admits both scalar arithmetics: the declaration is the check.
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

// Whether a bare product-plus-add is one rounding in this translation unit.
// The operands are volatile so the expression is evaluated rather than folded
// away; the multiply-add itself may still be contracted, which is the
// question, and the addend makes the two answers differ.
//
// The fused value is spelled out with the standard function rather than taken
// from the backend, because the backend's own multiply-add is what is under
// test: it follows the build's route, so asking it here would have the
// measurement confirm itself.
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

// The transport round-trips, and a broadcast of a value is that value in the
// one lane the scalar backend has.
TEST(BackendTest, ScalarTransportRoundTrips) {
    const double stored = 3.25;
    const double loaded = ScalarFp64::Load(&stored);
    double out = 0.0;
    ScalarFp64::Store(&out, ScalarFp64::Mul(loaded, ScalarFp64::Broadcast(2.0)));
    EXPECT_EQ(out, 6.5);
}

// MulAdd is the selected route's arithmetic, and this pair of operands tells
// the two routes apart: the fused step keeps the bit the product's rounding
// drops and the separate step loses it. MulSub is outside the route and is the
// two-rounding value on every build.
TEST(BackendTest, MulAddIsTheSelectedRouteAndMulSubIsNot) {
    const Probe<double> p = MakeProbe<double>();

    const double mulAdd = ScalarFp64::MulAdd(p.a, p.b, p.c);
    const double mulSub = ScalarFp64::MulSub(p.a, p.b, p.c);

    // The two readings of the sum, spelled out: one rounding through the
    // standard function, and the product rounded before the sum rounds.
    const double fused = std::fma(p.a, p.b, p.c);
    const double separate = ScalarFp64::Add(ScalarFp64::Mul(p.a, p.b), p.c);

    // The operands differ between the two readings, so which one MulAdd
    // returned says which arithmetic it ran.
    EXPECT_NE(fused, separate);
    EXPECT_EQ(fused, p.lost);

    // MulAdd is the route in force, which is the one the report states.
    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_GE(backends.size(), 1u);
    EXPECT_EQ(mulAdd, backends[0].route == MulAddRoute::kFused ? fused : separate);

    // MulSub is outside the route: the product rounds, then the difference
    // rounds, on every build.
    EXPECT_EQ(mulSub, ScalarFp64::Sub(ScalarFp64::Mul(p.a, p.b), p.c));
    EXPECT_NE(mulSub, mulAdd);
}

// The route the report states is the arithmetic the lane delivers, for each
// scalar precision: the reported route agrees with the one measured in this
// translation unit, and a route reported as separate is one whose fused step is
// really two roundings here. This is the check that a caller reading the table
// can attribute a value to the arithmetic that produced it.
TEST(BackendTest, ReportedMulAddRouteMatchesTheArithmetic) {
    const std::span<const BackendInfo> backends = BoysBackends();
    ASSERT_GE(backends.size(), 2u);

    EXPECT_EQ(backends[0].route, boys::backend::detail::RouteInForce<double>());
    EXPECT_EQ(backends[1].route, boys::backend::detail::RouteInForce<float>());

    // A route in force is the selection unless the build contracts the bare
    // form, so a reported separate route is only ever printed beside a
    // contraction measurement that says the two roundings happen.
    for (const BackendInfo& info : backends) {
        if (info.route == MulAddRoute::kSeparate) {
            EXPECT_FALSE(info.contracts) << info.name;
        }
    }
}

// The route is a name a report can print, and every route in the table has
// one that is not the placeholder.
TEST(BackendTest, MulAddRouteNamesArePrintable) {
    EXPECT_STREQ(MulAddRouteName(MulAddRoute::kFused), "fused");
    EXPECT_STREQ(MulAddRouteName(MulAddRoute::kSeparate), "separate");

    for (const BackendInfo& info : BoysBackends()) {
        EXPECT_STRNE(MulAddRouteName(info.route), "unknown") << info.name;
    }
}

// MulSub is the two-rounding reading on every build: it agrees with the
// product and the difference written separately, neither of which a
// contraction pass can merge into one step.
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

// Contracts() answers what this build does with a bare product-plus-add. The
// assertion is against the measurement itself, in both precisions, so a build
// that changes its mind is caught rather than assumed.
TEST(BackendTest, ContractsMatchesAMeasurementInThisTranslationUnit) {
    EXPECT_EQ(ScalarFp64::Contracts(), BareIsFused<double>());
    EXPECT_EQ(ScalarFp32::Contracts(), BareIsFused<float>());
}

// A lane is a value a report can print: every backend the build carries is
// named, the names are the ones the types state, and none repeats.
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

// The reported contraction flag is the one the header gives for the
// arithmetic carrying that name, so the report and the arithmetic agree.
//
// This is the check that the report describes the arithmetic the scalar lanes
// actually run. Contraction belongs to a compiled translation unit, and the
// library has more than one flag context: src/boys_simd.cpp carries the packed
// flags, everything else does not. A table measured in the packed unit would
// print that unit's answer beside the scalar lanes' values, so this compares
// the report against the same arithmetic measured where a scalar kernel would
// be compiled — which is what this file is.
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
// Every axis the entries select is one field of EvalPolicy, and every field has
// its own default. What the entries do with an axis is their own business; what
// is pinned here is which member each default names, so that a move of one is a
// decision this test states rather than a value that follows silently, and that
// the axes report themselves by name.
//
// The engine budget is the library's: it is what a single-precision engine
// computes at, and no build replaces it. The names an unnamed call resolves to
// are the BUILD's rather than the library's, which is what
// boys/boys_build_defaults.hpp exists for: a build that has measured its own
// machine replaces that header with its own set (that header states what a
// replacement carries; the CMake option is BOYS_BUILD_DEFAULTS, CONTRIBUTING.md).
// So the pins below are the shipped names where the shipped header is in force
// — the configuration every bound in this repository was measured at — and
// where it is not, they state what has to hold instead.
//
// TWO OF THE FIVE AXES ARE READ FROM THAT HEADER TODAY, the fit route and the
// evaluation scheme, and the pins below cover those two. The packing axis, the
// division form and the partition follow the same seam and are read by the
// constants in backend.hpp, so their pins are asserted in the change that makes
// those constants read this file rather than a literal: asserting them here
// would be asserting a value no header governs yet.
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
// A replacement is in force, and it is read instead of the committed file rather
// than beside it, so the names it carries are this build's. A replacement that
// names the shipped set has chosen nothing, and so does a seam that stopped
// delivering the file — both come out here as the committed values, which is
// what this refuses.
constexpr bool kShippedDefaultsInForce =
    boys::kDefaultFitRoute == boys::FitRoute::kChebyshev &&
    boys::kDefaultEvalScheme == boys::EvalScheme::kHorner;

static_assert(!kShippedDefaultsInForce,
              "the defaults header in force names the shipped route and scheme, so this build "
              "has chosen nothing: point BOYS_BUILD_DEFAULTS at a header that moves at least one "
              "axis, or unset it to build the shipped configuration");

#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)
// The test's own override (tests/build_defaults_tuned.hpp), pinned by value so
// that a configure which set the option and delivered some other header is a
// failure here rather than a green run of some other build's choices.
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the fixture's fit route is not in force");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kSplitClenshaw,
              "the fixture's evaluation scheme is not in force");
#endif
#endif

// The unnamed call is the build's policy, and the values it hands back are the
// shipped policy's only where the shipped header is in force. The comparison is
// made through the entry a caller writes — BoysAllOrders with no template
// argument against the same entry called at the shipped policy explicitly, over
// the domain's regions — so what is measured is what a call site gets and not
// what a constant holds, and it prints the count it moved rather than a
// checkmark.
TEST(BackendTest, TheUnnamedCallIsTheDefaultThisBuildWasCompiledWith) {
    using boys::BoysAllOrders;
    using Shipped = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                     boys::EvalScheme::kHorner,
                                     boys::BoysBudget::kFloat,
                                     boys::PackAxis::kArguments,
                                     boys::FitGranularity::kNarrow,
                                     boys::DivisionForm::kRefinedReciprocal>;

    // Arguments across the domain's regions, at orders that span the ladder: 0
    // is the seed every higher order is reached from, and kMaxBoysOrder is the
    // widest call the entries serve.
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
    // The fixture moves the scheme, so the values are not the shipped ones: the
    // count above is measured rather than expected, and this is the pin that
    // says an unnamed call follows the build's header.
    EXPECT_GT(moved, 0u) << "the fixture moves the scheme, so an unnamed call cannot hand back "
                            "the shipped policy's values";
#endif
}

// Naming the narrow partition is answered from its own tables; the combinations
// that have no narrow table are refused where they are named rather than
// answered from the shipped one. Those refusals are static_asserts inside
// RouteFit, the relaxed rungs' RequireShippedPartition and the single-precision
// lanes, and a refusal cannot be exercised by a test that has to compile: what
// is pinned here is the default, and the refusals are stated in the headers and
// in the contract document. The member itself is measured in the accuracy gate
// and reached through the public entries in the consumer umbrella.
TEST(BackendTest, ThePartitionNamesRoundTrip) {
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kShipped), "shipped");
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kNarrow), "narrow");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kShipped),
                 boys::GranularityName(boys::FitGranularity::kNarrow));

    // A partition this build serves is a row of the enumeration that says which
    // partitions exist: a name for a partition the enumeration does not carry is
    // a route a report cannot print and a caller cannot find, and the value it
    // prints instead is "unknown", which is the name of no partition at all.
    EXPECT_STREQ(boys::GranularityName(boys::FitGranularity::kUniform), "uniform");
    EXPECT_STRNE(boys::GranularityName(boys::FitGranularity::kUniform), "unknown");
}

// The uniform partition is one of the rows of that table and not a value the
// header names on the side, and what the row declares is checked against what
// the build answers, member by member.
//
// The check that matters is the last one, and it is made by calling the entry
// rather than by reading the row: a partition served by another partition's
// tables is the defect the row's own fields cannot show, because the fields
// would be the ones the substitution was made to satisfy. This build has
// refused that shape once already, at the fit selector (backend.hpp,
// RouteFit<FitRoute::kRationalMinimax, kScheme, FitGranularity::kUniform>, whose
// first version was answered by the narrow partition), so what is pinned here is
// that the uniform kernel is the uniform one and not a second name for the
// narrow fits.
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

    // One route, one packing axis and one rung, as the row states them.
    EXPECT_TRUE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kChebyshev));
    EXPECT_FALSE(boys::FitGranularityHasRoute(*uniform, boys::FitRoute::kRationalMinimax));
    EXPECT_TRUE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kArguments));
    EXPECT_FALSE(boys::FitGranularityHasAxis(*uniform, boys::PackAxis::kOrders));
    EXPECT_EQ(uniform->rungs, 1) << "the uniform table is stored at one degree and takes no rung";

    // The row's fields against the accessor that answers a caller, over every
    // combination of the other axes: a combination the accessor serves and the
    // row does not claim is a claim the row is missing, and one the row claims
    // and the accessor refuses is a route or an axis the build does not have.
    std::size_t served = 0;

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

    EXPECT_EQ(served, 4u)
        << "the partition is served at the reference rung on the arguments axis and the "
           "Chebyshev route alone: two evaluation schemes over the two rows the route table names "
           "that route by - one per region - and nothing else";

    // And the entry the row claims runs, reads its own table, and is not the
    // narrow partition's under another name.
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
}

// The division forms report themselves the way the other axes do: one row per
// member, in enumerator order, named by the library's own function, with the
// default one of the rows rather than a fourth thing beside them. A report that
// has to name a form it measured reads the name from here, so a row this
// accessor does not carry is a member no report can print.
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
