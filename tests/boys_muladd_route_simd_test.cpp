// The packed backends' route liveness: the half of the regression that needs
// the intrinsics, and so a translation unit of its own.
//
// tests/boys_muladd_route_test.cpp carries the argument this file is the second
// half of: a lane built to read BOYS_MULADD_SEPARATE has to DELIVER that route's
// values, and the route is a build fact a lane can stop reading while every
// suite stays green. What is held here is boys_backend_simd.hpp's pair -
// Avx2Fp64<Route> and Avx2Fp32<Route> - at both named routes and at the
// default, against the reference arithmetic of
// tests/boys_muladd_route_reference.hpp, in the LANES the vector holds.
//
// This unit carries the intrinsics' flags the way src/boys_simd.cpp does:
// CMakeLists.txt pins the same set onto it, contraction flag included, so the
// two spellings mean here what they mean in the library's own unit.
//
// WHICH ROUTE A STEP DELIVERS HERE. The fused spelling is the fused
// instruction. The separate spelling is a product and a sum, and a build that
// contracts a bare product-plus-add contracts this spelling too
// (boys_backend_simd.hpp:9-15, which names the compiler and the flags), so
// where THIS unit measures contraction the separate spelling is the fused
// arithmetic and the expected value is the fused one. The expectation is
// therefore read off the measurement rather than assumed from the name, and a
// lane that stopped reading the selection still fails: with contraction off the
// expected value is the two-rounding one and a hard-coded fused step delivers
// the other.

#include <gtest/gtest.h>

#include "boys/backend.hpp"
#include "boys/boys.hpp"

#include "boys_muladd_route_reference.hpp"

// --- Architecture guard -----------------------------------------------------
//
// As src/boys_simd.cpp: the build's answer when it states one, the compiler's
// predefines otherwise. Unlike that unit this one does not refuse an unstated
// non-x86_64 target - the library's SIMD unit is the one that must not compile
// the tier out silently - so a target without the tier carries a single
// skipping test and the rest of the suite is unchanged.
#ifdef BOYS_SIMD_X86

// The build answered; nothing to detect.

#elif defined(__x86_64__) || defined(_M_X64)

#define BOYS_SIMD_X86 1

#else

#define BOYS_SIMD_X86 0

#endif

#if BOYS_SIMD_X86

#include "boys_backend_simd.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace bdetail = boys::backend::detail;

using boys::backend::Avx2Fp32;
using boys::backend::Avx2Fp64;
using boys::backend::MulAddRoute;
using boys::backend::MulAddRouteName;
using route_reference::RouteForced;
using route_reference::StepProbes;
using route_reference::TwoRoundingSub;

/// What one sweep measured; the counts carry the verdict and are printed beside
/// it, as the scalar half's rows are.
struct PackedRow {
    const char* backend = "";
    const char* step = "";
    MulAddRoute named = MulAddRoute::kFused;
    MulAddRoute delivered = MulAddRoute::kFused;

    /// What the sweep's `routes-differ` count compares where the operation is
    /// not a route's (the multiply-subtract), and what the row's `delivered`
    /// column says where no route name is the answer.
    const char* differ_subject = "the two routes";
    const char* delivered_note = nullptr;

    bool contracts = false;
    std::size_t cells = 0;
    std::size_t routes_differ = 0;
    std::size_t off_route = 0;
    std::size_t on_the_other_route = 0;
    bool first_off_route = false;
    double first_delivered = 0.0;
    double first_expected = 0.0;
};

void Print(const PackedRow& row) {
    std::printf("  route-liveness family=%s step=%s named=%s delivered=%s contracts=%d cells=%zu "
                "routes-differ=%zu delivered-off-route=%zu delivered-on-the-other-route=%zu\n",
                row.backend,
                row.step,
                MulAddRouteName(row.named),
                row.delivered_note != nullptr ? row.delivered_note
                                              : MulAddRouteName(row.delivered),
                row.contracts ? 1 : 0,
                row.cells,
                row.routes_differ,
                row.off_route,
                row.on_the_other_route);

    if (row.first_off_route)
    {
        std::printf("    first off-route lane: delivered = %.17g, the delivered route's value = "
                    "%.17g\n",
                    row.first_delivered,
                    row.first_expected);
    }

    std::fflush(stdout);
}

void Hold(const PackedRow& row) {
    Print(row);

    EXPECT_EQ(row.off_route, std::size_t{0})
        << row.backend << ", step " << row.step << ": " << row.off_route << " of " << row.cells
        << " lanes are not the value this arithmetic's own route delivers here - the backend is "
           "not running the multiply-add route its spelling names";

    EXPECT_EQ(row.on_the_other_route, std::size_t{0})
        << row.backend << ", step " << row.step << ": " << row.on_the_other_route
        << " lanes are the OTHER route's at cells where the two routes differ";

    if (!row.contracts)
    {
        EXPECT_GT(row.routes_differ, std::size_t{0})
            << row.backend << ", step " << row.step
            << ": this unit does not contract a bare product-plus-add, so " << row.differ_subject
            << " are two arithmetics here, and no lane of this sweep told them apart: the sweep "
               "cannot see the difference it exists to see, and the counts above prove nothing";
    }
}

/// The route a packed instantiation's separate spelling really delivers here:
/// the route it names where this unit's contraction leaves the two spellings
/// apart, and the fused route where the build folds them into one instruction.
template <typename T>
MulAddRoute DeliveredRoute(MulAddRoute named) noexcept {
    return bdetail::MeasureContraction<T>() ? MulAddRoute::kFused : named;
}

/// One group of probe lanes through one instantiation's step, each lane held
/// against the value its own route delivers.
template <typename B, MulAddRoute kNamed>
PackedRow SweepPackedStep(const char* backend_name, const char* step_name) {
    using T = typename B::Value;

    PackedRow row;
    row.backend = backend_name;
    row.step = step_name;
    row.named = kNamed;
    row.delivered = DeliveredRoute<T>(kNamed);
    row.contracts = bdetail::MeasureContraction<T>();

    const std::vector<std::array<T, 3>> probes = StepProbes<T>();
    const std::size_t width = B::kWidth;

    for (std::size_t i = 0; i + width <= probes.size(); i += width)
    {
        std::array<T, 8> a{};
        std::array<T, 8> b{};
        std::array<T, 8> c{};
        std::array<T, 8> out{};

        for (std::size_t j = 0; j < width; ++j)
        {
            a[j] = probes[i + j][0];
            b[j] = probes[i + j][1];
            c[j] = probes[i + j][2];
        }

        B::Store(out.data(), B::MulAdd(B::Load(a.data()), B::Load(b.data()), B::Load(c.data())));

        for (std::size_t j = 0; j < width; ++j)
        {
            const T at_fused = RouteForced<MulAddRoute::kFused, T>::MulAdd(a[j], b[j], c[j]);
            const T at_separate = RouteForced<MulAddRoute::kSeparate, T>::MulAdd(a[j], b[j], c[j]);
            const bool differ = std::memcmp(&at_fused, &at_separate, sizeof(T)) != 0;
            const T expected = row.delivered == MulAddRoute::kFused ? at_fused : at_separate;
            const bool off = std::memcmp(&out[j], &expected, sizeof(T)) != 0;

            ++row.cells;
            if (differ)
            {
                ++row.routes_differ;
            }

            if (off)
            {
                ++row.off_route;

                if (!row.first_off_route)
                {
                    row.first_off_route = true;
                    row.first_delivered = static_cast<double>(out[j]);
                    row.first_expected = static_cast<double>(expected);
                }
            }

            const T other = expected == at_fused ? at_separate : at_fused;
            if (differ && std::memcmp(&out[j], &other, sizeof(T)) == 0)
            {
                ++row.on_the_other_route;
            }
        }
    }

    return row;
}

/// The multiply-subtract at one instantiation: two roundings on every build,
/// which is the contract boys_backend_simd.hpp states for it and the reason the
/// route does not reach it.
template <typename B, MulAddRoute kNamed>
PackedRow SweepPackedSub(const char* backend_name) {
    using T = typename B::Value;

    PackedRow row;
    row.backend = backend_name;
    row.step = "step-sub";
    row.named = kNamed;
    row.delivered = kNamed;
    row.differ_subject = "the fused step and the two-rounding step";
    row.delivered_note = "two-roundings-on-every-build";
    row.contracts = bdetail::MeasureContraction<T>();

    const std::vector<std::array<T, 3>> probes = StepProbes<T>();
    const std::size_t width = B::kWidth;

    for (std::size_t i = 0; i + width <= probes.size(); i += width)
    {
        std::array<T, 8> a{};
        std::array<T, 8> b{};
        std::array<T, 8> c{};
        std::array<T, 8> out{};

        for (std::size_t j = 0; j < width; ++j)
        {
            a[j] = probes[i + j][0];
            b[j] = probes[i + j][1];
            c[j] = probes[i + j][2];
        }

        B::Store(out.data(), B::MulSub(B::Load(a.data()), B::Load(b.data()), B::Load(c.data())));

        for (std::size_t j = 0; j < width; ++j)
        {
            // The fused step at this operation - one instruction, one rounding -
            // against the two-rounding difference the backend's MulSub promises
            // on every build. The route does not reach the operation, so a lane
            // that ran the route anyway would deliver the first where the two
            // part, and the count of those cells is what this sweep reads.
            const T fused = std::fma(a[j], b[j], -c[j]);
            const T expected = TwoRoundingSub(a[j], b[j], c[j]);

            ++row.cells;
            if (std::memcmp(&fused, &expected, sizeof(T)) != 0)
            {
                ++row.routes_differ;
            }

            if (std::memcmp(&out[j], &expected, sizeof(T)) != 0)
            {
                ++row.off_route;

                if (!row.first_off_route)
                {
                    row.first_off_route = true;
                    row.first_delivered = static_cast<double>(out[j]);
                    row.first_expected = static_cast<double>(expected);
                }

                if (std::memcmp(&out[j], &fused, sizeof(T)) == 0)
                {
                    ++row.on_the_other_route;
                }
            }
        }
    }

    return row;
}

TEST(BoysMulAddRouteSimd, PackedBackendsDeliverTheirRoute) {
    Hold(SweepPackedStep<Avx2Fp64<MulAddRoute::kFused>, MulAddRoute::kFused>("avx2-fp64",
                                                                           "step-fused"));
    Hold(SweepPackedStep<Avx2Fp64<MulAddRoute::kSeparate>, MulAddRoute::kSeparate>("avx2-fp64",
                                                                                  "step-separate"));
    Hold(SweepPackedStep<Avx2Fp64<>, bdetail::kSelectedRoute>("avx2-fp64", "step-default"));

    Hold(SweepPackedStep<Avx2Fp32<MulAddRoute::kFused>, MulAddRoute::kFused>("avx2-fp32",
                                                                            "step-fused"));
    Hold(SweepPackedStep<Avx2Fp32<MulAddRoute::kSeparate>, MulAddRoute::kSeparate>("avx2-fp32",
                                                                                   "step-separate"));
    Hold(SweepPackedStep<Avx2Fp32<>, bdetail::kSelectedRoute>("avx2-fp32", "step-default"));

    Hold(SweepPackedSub<Avx2Fp64<MulAddRoute::kFused>, MulAddRoute::kFused>("avx2-fp64"));
    Hold(SweepPackedSub<Avx2Fp64<MulAddRoute::kSeparate>, MulAddRoute::kSeparate>("avx2-fp64"));
    Hold(SweepPackedSub<Avx2Fp64<>, bdetail::kSelectedRoute>("avx2-fp64"));

    Hold(SweepPackedSub<Avx2Fp32<MulAddRoute::kFused>, MulAddRoute::kFused>("avx2-fp32"));
    Hold(SweepPackedSub<Avx2Fp32<MulAddRoute::kSeparate>, MulAddRoute::kSeparate>("avx2-fp32"));
    Hold(SweepPackedSub<Avx2Fp32<>, bdetail::kSelectedRoute>("avx2-fp32"));
}

TEST(BoysMulAddRouteSimd, TheReportNamesTheseInstantiations) {
    // The default instantiation is the build's selection, which is what a call
    // site that names no route gets (boys_backend_simd.hpp: the Route parameter
    // of each backend), and it is the route the library reports for the pair.
    EXPECT_EQ(Avx2Fp64<>::kRoute, bdetail::kSelectedRoute);
    EXPECT_EQ(Avx2Fp32<>::kRoute, bdetail::kSelectedRoute);

    const boys::backend::BackendInfo* fp64 = nullptr;
    const boys::backend::BackendInfo* fp32 = nullptr;

    for (const boys::backend::BackendInfo& info : boys::backend::BoysBackends())
    {
        if (std::strcmp(info.name, Avx2Fp64<>::kName) == 0)
        {
            fp64 = &info;
        }

        if (std::strcmp(info.name, Avx2Fp32<>::kName) == 0)
        {
            fp32 = &info;
        }
    }

    if (fp64 == nullptr || fp32 == nullptr)
    {
        GTEST_SKIP() << "the packed pair is not listed on this host - the tier is not available at "
                        "run time, so the library reports the scalar pair alone and the rows "
                        "above are the whole of this unit's verdict";
    }

    // The pair's rows are the instantiation's own (src/boys_simd.cpp:
    // AppendPackedBackends), and the contraction answer each carries is the one
    // this unit measures for itself: a unit whose flags told it a different
    // answer from the library's would be a scale whose two halves disagree.
    EXPECT_EQ(MulAddRouteName(fp64->route), MulAddRouteName(Avx2Fp64<>::kRoute));
    EXPECT_EQ(MulAddRouteName(fp32->route), MulAddRouteName(Avx2Fp32<>::kRoute));
    EXPECT_EQ(fp64->contracts, Avx2Fp64<>::Contracts());
    EXPECT_EQ(fp32->contracts, Avx2Fp32<>::Contracts());
    EXPECT_EQ(Avx2Fp64<>::Contracts(), bdetail::MeasureContraction<double>());
    EXPECT_EQ(Avx2Fp32<>::Contracts(), bdetail::MeasureContraction<float>());
}

#else

TEST(BoysMulAddRouteSimd, PackedBackendsAreNotBuiltOnThisTarget) {
    GTEST_SKIP() << "this target has no AVX2 + FMA tier: src/boys_simd.cpp defines the region "
                    "kernels against the certified scalar lanes here and there is no packed "
                    "backend to hold to a route";
}

#endif // BOYS_SIMD_X86

} // namespace
