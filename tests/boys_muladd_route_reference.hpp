#pragma once

// The reference arithmetic at a NAMED multiply-add route.
//
// The library's own certified summations (boys_impl.hpp: ClenshawSplit,
// HornerMono, FitSum) are templates over any arithmetic backend, and the routes
// are a build fact (boys/backend.hpp: kSelectedRoute). The two backends below
// make the route a template argument instead, so one binary holds either
// arithmetic, and a lane's delivered value can be held against the route it was
// built to run rather than against what it reports. See
// tests/boys_muladd_route_test.cpp for the argument and the sweeps, and
// tests/boys_muladd_route_simd_test.cpp for the packed backends' half.
//
// Both live here rather than in one test file because two translation units
// need them and the two must be the same arithmetic: a second copy that drifted
// would be a reference that agrees with nothing.

#include "boys/backend.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace route_reference {

namespace bdetail = boys::backend::detail;

/// The scalar backend's contract at a route a test names: `a * b + c` at that
/// route, and `a * b - c` with two roundings on every build, which is what
/// boys/backend.hpp's Scalar<T> promises of the pair. It differs from the
/// build's Scalar<T> in one thing only - the route is the argument here and the
/// selection there.
template <boys::backend::MulAddRoute kForcedRoute, typename T>
struct RouteForced {
    using Value = T;
    using Storage = T;
    using Packed = T;

    static constexpr std::size_t kWidth = 1;
    static constexpr const char* kName = "route-forced";
    static constexpr boys::backend::MulAddRoute kRoute = kForcedRoute;

    static Packed Load(const Storage* p) noexcept { return *p; }
    static void Store(Storage* p, Packed v) noexcept { *p = v; }
    static Packed Broadcast(Value v) noexcept { return v; }
    static Packed Mul(Packed a, Packed b) noexcept { return a * b; }
    static Packed Add(Packed a, Packed b) noexcept { return a + b; }
    static Packed Sub(Packed a, Packed b) noexcept { return a - b; }

    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kForcedRoute == boys::backend::MulAddRoute::kFused)
        {
            return bdetail::Fused(a, b, c);
        } else
        {
            return bdetail::Separate(a, b, c);
        }
    }

    /// The backend's own two-rounding contract, which the route does not reach
    /// (boys/backend.hpp: Scalar<T>::MulSub).
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return bdetail::Fused(a, b, Packed{0}) - c;
    }

    static bool Contracts() noexcept { return bdetail::MeasureContraction<T>(); }
};

/// The orders lane's STEP arithmetic at a route a test names:
/// src/boys_orders_simd.cpp's StepMulAdd/StepMulSub as a backend. Its
/// multiply-subtract sits on the route, unlike the backend's above: the fused
/// form is one instruction and one rounding, whose scalar spelling is
/// `fma(a, b, -c)`, and the separate form is the bare product and difference,
/// which contracts where the build contracts - exactly as the lane's two named
/// instructions do.
template <boys::backend::MulAddRoute kForcedRoute, typename T>
struct RouteStep {
    using Value = T;
    using Storage = T;
    using Packed = T;

    static constexpr std::size_t kWidth = 1;
    static constexpr const char* kName = "route-step";
    static constexpr boys::backend::MulAddRoute kRoute = kForcedRoute;

    static Packed Load(const Storage* p) noexcept { return *p; }
    static void Store(Storage* p, Packed v) noexcept { *p = v; }
    static Packed Broadcast(Value v) noexcept { return v; }
    static Packed Mul(Packed a, Packed b) noexcept { return a * b; }
    static Packed Add(Packed a, Packed b) noexcept { return a + b; }
    static Packed Sub(Packed a, Packed b) noexcept { return a - b; }

    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kForcedRoute == boys::backend::MulAddRoute::kFused)
        {
            return bdetail::Fused(a, b, c);
        } else
        {
            return bdetail::Separate(a, b, c);
        }
    }

    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kForcedRoute == boys::backend::MulAddRoute::kFused)
        {
            return std::fma(a, b, -c);
        } else
        {
            return a * b - c;
        }
    }

    static bool Contracts() noexcept { return bdetail::MeasureContraction<T>(); }
};

/// Triples whose product and sum are inexact, so the fused and the two-rounding
/// steps part: `(1 + 2^-s)^2 - (1 + 2^-(s-1))`, the shape the library's own
/// contraction measurement uses (boys/backend.hpp: MeasureContraction), at
/// several shifts, signs and forms. Read through volatile where they are used,
/// so the compiler evaluates the steps rather than the constants.
template <typename T>
std::vector<std::array<T, 3>> StepProbes() {
    constexpr int kCentral = std::numeric_limits<T>::digits / 2 + 1;
    const T one{1};
    std::vector<std::array<T, 3>> probes;

    for (int j = 0; j < 6; ++j)
    {
        const int s = kCentral - 2 + j;
        const T small = static_cast<T>(std::ldexp(1.0, -s));
        const T half = static_cast<T>(std::ldexp(1.0, -(s - 1)));

        probes.push_back({one + small, one + small, -(one + half)});
        probes.push_back({one - small, one + small, -(one - half)});
        probes.push_back({one + small, one - small, one});
        probes.push_back({one + half, one - small, small});
    }

    return probes;
}

/// `a * b - c` with two roundings, spelled so that no contraction setting can
/// fuse it: the product is stored through a volatile and the difference is
/// taken from the stored value.
template <typename T>
T TwoRoundingSub(T a, T b, T c) noexcept {
    volatile T product = a * b;
    return static_cast<T>(product) - c;
}

} // namespace route_reference
