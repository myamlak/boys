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
#include <cmath>
#include <cstddef>
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
// The selection axes on the policy
// ---------------------------------------------------------------------------
// Every axis the entries select is one field of EvalPolicy, and every field has
// its own default. What the entries do with an axis is their own business; what
// is pinned here is that the defaults are the shipped ones, so a call site that
// names no axis compiles the code it always did, and that the axes report
// themselves by name.
static_assert(boys::EvalPolicy<>{}.kRoute == boys::FitRoute::kChebyshev,
              "the default fit route moved");
static_assert(boys::EvalPolicy<>{}.kScheme == boys::EvalScheme::kSplitClenshaw,
              "the default evaluation scheme moved");
static_assert(boys::EvalPolicy<>{}.kBudget == boys::BoysBudget::kFloat,
              "the default engine budget moved");
static_assert(boys::EvalPolicy<>{}.kGranularity == boys::FitGranularity::kShipped,
              "the default partition moved: a call site that names none must compile the "
              "committed tables");

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
}
