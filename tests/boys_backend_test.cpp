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
template <typename T>
bool BareIsFused() {
    const Probe<T> p = MakeProbe<T>();
    volatile const T va = p.a;
    volatile const T vb = p.b;
    volatile const T vc = p.c;
    const T bare = va * vb + vc;
    return bare == Scalar<T>::MulAdd(p.a, p.b, p.c);
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

// The two multiply-adds are different operations, and this pair of operands
// shows it: the fused form keeps the bit the product's rounding drops.
TEST(BackendTest, MulAddIsFusedAndMulSubIsNot) {
    const Probe<double> p = MakeProbe<double>();

    const double fused = ScalarFp64::MulAdd(p.a, p.b, p.c);
    const double twoRoundings = ScalarFp64::MulSub(p.a, p.b, p.c);

    EXPECT_EQ(fused, std::fma(p.a, p.b, p.c));
    EXPECT_EQ(fused, p.lost);
    EXPECT_NE(fused, twoRoundings);
    // The two-rounding form is the product rounded and then the difference
    // rounded, each a single rounded operation of its own.
    EXPECT_EQ(twoRoundings, ScalarFp64::Sub(ScalarFp64::Mul(p.a, p.b), p.c));
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
