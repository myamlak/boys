#pragma once

// The template definitions behind the accuracy-multiplier surface of
// boys/boys.hpp. Internal header: callers all use the default m = 1, which the
// extern-template
// declarations in boys.hpp route to the explicit instantiations in
// boys.cpp / boys_simd.cpp; the sampled-m instantiations (the contract
// tests) include this header directly for their definitions.
//
// Every entry is compiled twice under if constexpr: the m = 1 branch is
// today's certified body VERBATIM (the bit-identity pin — the
// discarded relaxed branch adds no instruction, branch, or load to the
// m = 1 path), and the m > 1 branch evaluates the seed fits at the
// compile-time effective degrees of boys_effective_degrees.hpp.
// The relaxed region-C paths are m-invariant (the asymptotic form has no
// coefficients to truncate); only their scalar tails carry the multiplier.

#include "boys/boys.hpp"
#include "boys_coefficients.hpp"
#include "boys_effective_degrees.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <new>

namespace boys {
namespace detail {

constexpr double kBoysHalfSqrtPi = 0.886226925452758014; // sqrt(pi)/2
constexpr float kBoysHalfSqrtPiF32 = 0.88622693f; // sqrt(pi)/2, single lane

// ---------------------------------------------------------------------------
// Scalar double lane helpers (today's anonymous-namespace helpers, verbatim)
// ---------------------------------------------------------------------------
// Piece containing x for this order; the pieces partition [0, kX0).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::OrderPiece& FindPiece(int order, double x) noexcept {
    const int first = detail::kPieceStart[order];
    const int last = detail::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::kPieces[i].b)
        {
            return detail::kPieces[i];
        }
    }

    return detail::kPieces[last - 1];
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
inline double ClenshawSplit(const double* c, int deg, double t) noexcept {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return std::fma(t, c[1], c[0]);
    }

    const double v = std::fma(2.0, t * t, -1.0);
    const double twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4.
        return std::fma(t, c[1], std::fma(v, c[2], c[0]));
    }

    // The odd part below is the Clenshaw finalization for the D family with
    // the top odd coefficient c[2m-1]; it is valid only for even deg >= 4
    // (deg == 2 is handled above). The generator
    // (tools/gen_boys_coefficients.py) only emits such degrees.
    assert(deg >= 4 && deg % 2 == 0);

    // Even part: coefficients c[0], c[2], ..., c[2m].
    const int m = deg / 2;
    double b1 = c[std::ptrdiff_t{2} * m];
    double b2 = 0.0;

    for (int k = m - 1; k >= 1; --k)
    {
        const double b0 = std::fma(twoV, b1, c[std::ptrdiff_t{2} * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const double even = std::fma(v, b1, c[0] - b2);

    // Odd part: coefficients c[1], c[3], ..., c[2m-1] with the D recurrence.
    double o1 = c[2 * m - 1];
    double o2 = 0.0;

    for (int k = m - 2; k >= 1; --k)
    {
        const double o0 = std::fma(twoV, o1, c[std::ptrdiff_t{2} * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const double odd = std::fma(twoV - 1.0, o1, c[1] - o2);
    return std::fma(t, odd, even);
}

// Region-A seed F_order(x) via the split Clenshaw evaluation.
inline double ChebyshevValue(int order, double x) noexcept {
    const detail::OrderPiece& piece = FindPiece(order, x);
    const double* c = detail::kCoeffs.data() + piece.offset;
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return ClenshawSplit(c, piece.deg, t);
}

// Region-B seed F_0(x), valid on [kX0, kX1).
inline double RegionBSeed(double x) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
    return ClenshawSplit(detail::kBcoeffs.data(), detail::kBDeg, t);
}

// The extended band's per-(n, x) dispatch (the pure-function rule): an
// order n takes the extended seed exactly when x >= kTierThresholds[n],
// regardless of the entry point or the other orders of a batch call. The
// threshold table is n-indexed (structurally like the per-order degree
// tables): each order's entry is the certified boundary of the smallest
// kmax row that covers it (the rows 4/8/16/32), so
// BoysAllOrders(n,x)[n] == BoysSingle(n,x) == BoysAllOrders(m>=n,x)[n] exactly.
// The band is the m = 1 lane's: the m > 1 branch keeps today's
// region-A dispatch in the band (the per-order amplification of the
// band's upward recursion needs its own derivation - named future work,
// not ported here), and the float lane's dispatch is untouched (its
// double-seeded region-A branch already serves the band at the float
// budget).

// The extended-band seed (the per-range seed design): F_0(x) on
// [kExtendedBX0, kX0) via the same split Clenshaw evaluation as the
// region-B seed. It serves the upward recursion below kX0 in the m = 1
// double lanes only, dispatched per (n, x) at the certified per-order
// thresholds kTierThresholds; the m > 1 branch and the float lanes keep
// their existing dispatch untouched.
inline double RegionBExtendedSeed(double x) noexcept {
    const double t = 2.0 * (x - kExtendedBX0) / (kX0 - kExtendedBX0) - 1.0;
    return ClenshawSplit(detail::kExtendedBcoeffs.data(), detail::kExtendedBDeg, t);
}

// ---------------------------------------------------------------------------
// Scalar float lane helpers (today's anonymous-namespace helpers, verbatim)
// ---------------------------------------------------------------------------
// Float-lane piece lookup; see FindPiece.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::f32::OrderPiece& FindPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kPieceStart[order];
    const int last = detail::f32::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kPieces[i].b)
        {
            return detail::f32::kPieces[i];
        }
    }

    return detail::f32::kPieces[last - 1];
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
inline float ClenshawSplitF32(const float* c, int deg, float t) noexcept {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return std::fmaf(t, c[1], c[0]);
    }

    const float v = std::fmaf(2.0f, t * t, -1.0f);
    const float twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4.
        return std::fmaf(t, c[1], std::fmaf(v, c[2], c[0]));
    }

    assert(deg >= 4 && deg % 2 == 0); // see ClenshawSplit

    const int m = deg / 2;
    float b1 = c[std::ptrdiff_t{2} * m];
    float b2 = 0.0f;

    for (int k = m - 1; k >= 1; --k)
    {
        const float b0 = std::fmaf(twoV, b1, c[std::ptrdiff_t{2} * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const float even = std::fmaf(v, b1, c[0] - b2);

    float o1 = c[2 * m - 1];
    float o2 = 0.0f;

    for (int k = m - 2; k >= 1; --k)
    {
        const float o0 = std::fmaf(twoV, o1, c[std::ptrdiff_t{2} * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const float odd = std::fmaf(twoV - 1.0f, o1, c[1] - o2);
    return std::fmaf(t, odd, even);
}

// Float-lane region-A seed; see ChebyshevValue.
// Forced inline: the m = 1 F32-single engine must keep the full-accuracy
// code shape (piece scan inlined); with two call sites (the kFloat and the
// kFp16 budget instantiations) MSVC's size heuristic keeps this helper
// out-of-line otherwise. Semantics are unaffected — inline never changes the
// bit-identity pin.
// __forceinline is MSVC-only; GCC/Clang spell the same intent with
// always_inline (plain inline is a hint there, not a requirement).
#if defined(_MSC_VER)
#define BoysForceInline __forceinline
#else
#define BoysForceInline inline __attribute__((always_inline))
#endif
BoysForceInline float ChebyshevValueF32(int order, float x) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const float* c = detail::f32::kCoeffs.data() + piece.offset;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return ClenshawSplitF32(c, piece.deg, t);
}

#undef BoysForceInline

// Float-lane region-B seed; see RegionBSeed.
inline float RegionBSeedF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return ClenshawSplitF32(detail::f32::kBcoeffs.data(), detail::f32::kBDeg, t);
}

// ---------------------------------------------------------------------------
// Degree-parameterized variants (the relaxation path)
// ---------------------------------------------------------------------------
// The region-A seed at the piece's effective degree; the piece index is the
// flat kPieces index (the degrees tables are indexed the same way).
template <typename DegreesArray>
double ChebyshevValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    const detail::OrderPiece& piece = FindPiece(order, x);
    const std::ptrdiff_t index = &piece - detail::kPieces.data();
    const double* c = detail::kCoeffs.data() + piece.offset;
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return ClenshawSplit(c, degrees[static_cast<std::size_t>(index)], t);
}

template <typename DegreesArray>
float ChebyshevValueF32WithDegrees(int order, float x, const DegreesArray& degrees) noexcept {
    const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
    const std::ptrdiff_t index = &piece - detail::f32::kPieces.data();
    const float* c = detail::f32::kCoeffs.data() + piece.offset;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    return ClenshawSplitF32(c, degrees[static_cast<std::size_t>(index)], t);
}

inline double RegionBSeedWithDegrees(double x, int degree) noexcept {
    const double t = 2.0 * (x - kX0) / (kX1 - kX0) - 1.0;
    return ClenshawSplit(detail::kBcoeffs.data(), degree, t);
}

inline float RegionBSeedF32WithDegrees(float x, int degree) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    return ClenshawSplitF32(detail::f32::kBcoeffs.data(), degree, t);
}

// ---------------------------------------------------------------------------
// The lane engines (compiled twice: the m = 1 branch is today's body
// verbatim; the relaxed branch evaluates at the effective degrees)
// ---------------------------------------------------------------------------
// The F32 engine's computation budget: the float lane's 1.5e-7, or the
// fp16/bf16 lanes' tighter 1e-7 (the fp16 formula; the fp16 entries are
// I/O around this engine, so the budget is the only difference between the
// roles at m > 1 — at m = 1 the branch is identical regardless).
enum class BoysBudget {
    kFloat,
    kFp16,
};

template <double kAccuracyMultiplier> double BoysSingleImpl(int n, double x) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0)
        {
            return 1.0 / (2.0 * n + 1.0);
        }

        if (x < kX0)
        {
            if (x >= detail::kTierThresholds[static_cast<std::size_t>(n)])
            {
                double f = RegionBExtendedSeed(x);
                const double expx = 0.5 * std::exp(-x);

                for (int l = 0; l < n; ++l)
                {
                    f = ((l + 0.5) * f - expx) / x;
                }

                return f;
            }

            return ChebyshevValue(n, x);
        }

        double f = RegionBSeed(x);

        if (x < kX1)
        {
            const double expx = 0.5 * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((l + 0.5) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPi / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (l + 0.5) * f / x;
        }

        return f;
    } else
    {
        static constexpr auto kDegreesA =
            RegionADegrees<kAccuracyMultiplier, BoysRole::kDoubleSingle>();
        static constexpr auto kDegreesB =
            RegionBDegrees<kAccuracyMultiplier, BoysRole::kDoubleSingle>();

        if (x == 0.0)
        {
            return 1.0 / (2.0 * n + 1.0);
        }

        if (x < kX0)
        {
            return ChebyshevValueWithDegrees(n, x, kDegreesA);
        }

        double f = RegionBSeedWithDegrees(x, kDegreesB[static_cast<std::size_t>(n)]);

        if (x < kX1)
        {
            const double expx = 0.5 * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((l + 0.5) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPi / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (l + 0.5) * f / x;
        }

        return f;
    }
}

template <double kAccuracyMultiplier> void BoysAllOrdersImpl(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0 / (2.0 * l + 1.0);
            }

            return;
        }

        if (x < kX0)
        {
            // The pure per-(n, x) dispatch, driven by the n-indexed
            // threshold table: the upward recursion from the extended seed
            // serves exactly the orders k with x >= kTierThresholds[k] - a
            // prefix (the thresholds are non-decreasing in n); the tail
            // orders come from the per-order region-A fits. The per-order
            // result out[k] is then bit-identical to BoysSingle(k, x) for
            // every k (the single entry applies the same per-(n, x) rule).
            int served = 0;

            while (served < nmax &&
                   x >= detail::kTierThresholds[static_cast<std::size_t>(served + 1)])
            {
                ++served;
            }

            if (x >= detail::kTierThresholds[0])
            {
                double f = RegionBExtendedSeed(x);
                out[0] = f;
                const double expx = 0.5 * std::exp(-x);

                for (int l = 1; l <= served; ++l)
                {
                    f = ((l - 0.5) * f - expx) / x;
                    out[l] = f;
                }

                for (int l = served + 1; l <= nmax; ++l)
                {
                    out[l] = ChebyshevValue(l, x);
                }

                return;
            }

            double f = ChebyshevValue(nmax, x);
            out[nmax] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (l + 0.5);
                out[l] = f;
            }

            return;
        }

        if (x < kX1)
        {
            double f = RegionBSeed(x);
            out[0] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((l - 0.5) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        double f = kBoysHalfSqrtPi / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (l - 0.5) * f / x;
            out[l] = f;
        }
    } else
    {
        static constexpr auto kDegreesA =
            RegionADegrees<kAccuracyMultiplier, BoysRole::kDoubleBatch>();
        static constexpr auto kDegreesB =
            RegionBDegrees<kAccuracyMultiplier, BoysRole::kDoubleBatch>();

        if (x == 0.0)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0 / (2.0 * l + 1.0);
            }

            return;
        }

        if (x < kX0)
        {
            double f = ChebyshevValueWithDegrees(nmax, x, kDegreesA);
            out[nmax] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (l + 0.5);
                out[l] = f;
            }

            return;
        }

        if (x < kX1)
        {
            // The F0 seed's truncation error reaches EVERY output order with
            // gain A_B(l) = prod_{j=0..l-1} (j+1/2)/x0^l, the same quantity as
            // prod_{j=1..l} (2j-1)/(2 x0): the upward step's factor is (2l-1)
            // over 2x, and kX0 sits at the value that makes A_B(32) exactly
            // one. Shipped, it is 1 + 1.846e-17 at l = 32 --- above one by a
            // last-digit excess, and the maximum over l <= 32 --- so the
            // seed's error is not amplified: the order-0 entry is the seed
            // degree, the one whose criterion is
            // Delta(d') * 1 <= (m-1) * B. Written with the product running
            // j = 0..l-1 as above; run from j = 1 it carries one extra factor,
            // (2l+1) = 65 at l = 32, which is not the gain. The per-order
            // entries kDegreesB[n] (gain far below one at small n) bound only
            // the order-n single-style path; seeding with kDegreesB[nmax]
            // leaves F0's error at Delta(d'(nmax)) — up to
            // (m-1)*B/A_B(nmax) — unbounded.
            double f = RegionBSeedWithDegrees(x, kDegreesB[0]);
            out[0] = f;
            const double expx = 0.5 * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((l - 0.5) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        double f = kBoysHalfSqrtPi / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (l - 0.5) * f / x;
            out[l] = f;
        }
    }
}

// The fixed-n vector engine: F_n at every argument of an array, one fixed
// order (the batch shape of angular-momentum-grouped inner loops; the
// strided output layout belongs to the public surface in boys.hpp). The
// region bodies below mirror BoysSingleImpl's verbatim - the m = 1 branch
// is the certified scalar single-lane code (the bit-identity pin), the
// relaxed branch the same bodies at the single-lane effective degrees
// (BoysRole::kDoubleSingle) - so every output element is bit-identical to
// the corresponding BoysSingle call by construction; keep the two engines'
// bodies in lockstep. The region dispatch is per element: mixed-region
// arguments need no pre-partitioning (the portable shape). The
// dispatch-once-per-batch region structure lives in the AVX2 region-sorted
// lanes (boys_simd.cpp), whose callers partition by region first.
template <double kAccuracyMultiplier>
void BoysFixedNImpl(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x != nullptr);
    assert(out != nullptr);
    assert(stride >= 1);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const double xi = x[i];
            assert(xi >= 0.0);

            if (xi == 0.0)
            {
                out[i * stride] = 1.0 / (2.0 * n + 1.0);
                continue;
            }

            if (xi < kX0)
            {
                if (xi >= detail::kTierThresholds[static_cast<std::size_t>(n)])
                {
                    double f = RegionBExtendedSeed(xi);
                    const double expx = 0.5 * std::exp(-xi);

                    for (int l = 0; l < n; ++l)
                    {
                        f = ((l + 0.5) * f - expx) / xi;
                    }

                    out[i * stride] = f;
                    continue;
                }

                out[i * stride] = ChebyshevValue(n, xi);
                continue;
            }

            double f = RegionBSeed(xi);

            if (xi < kX1)
            {
                const double expx = 0.5 * std::exp(-xi);

                for (int l = 0; l < n; ++l)
                {
                    f = ((l + 0.5) * f - expx) / xi;
                }

                out[i * stride] = f;
                continue;
            }

            f = kBoysHalfSqrtPi / std::sqrt(xi);

            for (int l = 0; l < n; ++l)
            {
                f = (l + 0.5) * f / xi;
            }

            out[i * stride] = f;
        }
    } else
    {
        static constexpr auto kDegreesA =
            RegionADegrees<kAccuracyMultiplier, BoysRole::kDoubleSingle>();
        static constexpr auto kDegreesB =
            RegionBDegrees<kAccuracyMultiplier, BoysRole::kDoubleSingle>();

        for (std::size_t i = 0; i < count; ++i)
        {
            const double xi = x[i];
            assert(xi >= 0.0);

            if (xi == 0.0)
            {
                out[i * stride] = 1.0 / (2.0 * n + 1.0);
                continue;
            }

            if (xi < kX0)
            {
                out[i * stride] = ChebyshevValueWithDegrees(n, xi, kDegreesA);
                continue;
            }

            // The order-n single-lane seed entry mirrors the relaxed single
            // path; the per-order amplification analysis of BoysSingleImpl
            // applies unchanged.
            double f = RegionBSeedWithDegrees(xi, kDegreesB[static_cast<std::size_t>(n)]);

            if (xi < kX1)
            {
                const double expx = 0.5 * std::exp(-xi);

                for (int l = 0; l < n; ++l)
                {
                    f = ((l + 0.5) * f - expx) / xi;
                }

                out[i * stride] = f;
                continue;
            }

            f = kBoysHalfSqrtPi / std::sqrt(xi);

            for (int l = 0; l < n; ++l)
            {
                f = (l + 0.5) * f / xi;
            }

            out[i * stride] = f;
        }
    }
}

// The float lanes' scope (the per-range seed design is fp64-only v1): the
// float dispatch is untouched, still keyed to kX0/kX1, so the carved band
// [kExtendedBX0, kX0) stays EXACTLY today's float path - the per-order
// region-A fits, double-seeded in the batch form (ChebyshevValue's double
// evaluation), serving the band at the float budget. The certified table
// is the double recursion's; the float band is measured, not certified.
template <double kAccuracyMultiplier, BoysBudget kBudget>
float BoysSingleF32Impl(int n, float x) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= 0.0f);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0f)
        {
            return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            return ChebyshevValueF32(n, x);
        }

        float f = RegionBSeedF32(x);

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((static_cast<float>(l) + 0.5f) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (static_cast<float>(l) + 0.5f) * f / x;
        }

        return f;
    } else
    {
        constexpr BoysRole kRole =
            (kBudget == BoysBudget::kFloat) ? BoysRole::kF32Single : BoysRole::kF32Fp16Single;
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier, kRole>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier, kRole>();

        if (x == 0.0f)
        {
            return 1.0f / (2.0f * static_cast<float>(n) + 1.0f);
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            return ChebyshevValueF32WithDegrees(n, x, kDegreesA);
        }

        float f = RegionBSeedF32WithDegrees(x, kDegreesB[static_cast<std::size_t>(n)]);

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 0; l < n; ++l)
            {
                f = ((static_cast<float>(l) + 0.5f) * f - expx) / x;
            }

            return f;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);

        for (int l = 0; l < n; ++l)
        {
            f = (static_cast<float>(l) + 0.5f) * f / x;
        }

        return f;
    }
}

template <double kAccuracyMultiplier, BoysBudget kBudget>
void BoysAllOrdersF32Impl(int nmax, float x, float* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            // The seed must be double precision: the downward recursion
            // amplifies a float seed error by up to x0^n / prod(j+1/2)
            // (5e4 at nmax=8), far beyond the certified 1.5e-7 float budget;
            // one double evaluation per batch is negligible; the recursion
            // itself stays in float.
            const double seed = ChebyshevValue(nmax, static_cast<double>(x));
            out[nmax] = static_cast<float>(seed);
            float f = out[nmax];
            const float expx = 0.5f * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (static_cast<float>(l) + 0.5f);
                out[l] = f;
            }

            return;
        }

        float f = RegionBSeedF32(x);
        out[0] = f;

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((static_cast<float>(l) - 0.5f) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (static_cast<float>(l) - 0.5f) * f / x;
            out[l] = f;
        }
    } else
    {
        constexpr BoysRole kRole =
            (kBudget == BoysBudget::kFloat) ? BoysRole::kF32Batch : BoysRole::kF32Fp16Batch;
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier, kRole>();
        static constexpr auto kDegreesB = RegionBDegrees<kAccuracyMultiplier, kRole>();

        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        const float x0 = static_cast<float>(kX0);
        const float x1 = static_cast<float>(kX1);

        if (x < x0)
        {
            // The seed must be double precision (see the m = 1 branch); the
            // recursion itself stays in float.
            const double seed = ChebyshevValueWithDegrees(nmax, static_cast<double>(x), kDegreesA);
            out[nmax] = static_cast<float>(seed);
            float f = out[nmax];
            const float expx = 0.5f * std::exp(-x);

            for (int l = nmax - 1; l >= 0; --l)
            {
                f = (x * f + expx) / (static_cast<float>(l) + 0.5f);
                out[l] = f;
            }

            return;
        }

        // See the double batch branch: the F0 seed's error reaches every
        // output with gain <= 1 + 1.846e-17 (the ladder's gain at the band
        // edge, one at the last order), so the order-0 entry is
        // the one that bounds the whole batch (kDegreesB[nmax] is loose:
        // the gain is far below 1 at small nmax, and the entry is 0 for many
        // orders at m >= 1e4 — a degree-0 seed error ~5e-2 on F0).
        float f = RegionBSeedF32WithDegrees(x, kDegreesB[0]);
        out[0] = f;

        if (x < x1)
        {
            const float expx = 0.5f * std::exp(-x);

            for (int l = 1; l <= nmax; ++l)
            {
                f = ((static_cast<float>(l) - 0.5f) * f - expx) / x;
                out[l] = f;
            }

            return;
        }

        f = kBoysHalfSqrtPiF32 / std::sqrt(x);
        out[0] = f;

        for (int l = 1; l <= nmax; ++l)
        {
            f = (static_cast<float>(l) - 0.5f) * f / x;
            out[l] = f;
        }
    }
}

// ---------------------------------------------------------------------------
// The region kernels (internal: defined in boys_simd.cpp)
// ---------------------------------------------------------------------------
// One region of the argument line per kernel, same order across an array of
// arguments, AVX2 with a scalar tail. They are the vector tier the entries
// dispatch to; they are not the public surface, because a caller reaching them
// directly has to partition its arguments by region itself, which is the work
// the batch entries exist to do.
//
// The certified per-value bounds: A |F̂ − F| ≤ m·1e-15 over [0, kX0), B and C
// ≤ m·5.5e-14. Measured against the committed reference grid (this tree's test
// suite): A holds that bound below the extended band and lands 1.4e-15 at
// n = 32 on the band itself, B lands 2.2e-13 at n = 32, x = kX0 — four times
// its own bound, which is why the batch entry serves region B with the scalar
// body instead — and C lands 5.0e-14, inside its bound with 10% slack.
//
// On a non-x86_64 target these are defined against the certified scalar lanes
// (see the guard in boys_simd.cpp) and BoysAvx2Available() reports false.
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for arguments in region B; out[order * count + i].
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept;

template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
// The fp16/bf16 lanes: the same kernels on half-precision I/O types, per order.
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

extern template void BoysRegionASimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionBSimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionCSimd<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
extern template void BoysRegionASimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionBSimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionCSimdF16<kBoysFullAccuracyMultiplier>(
    int n, const F16* x, F16* out, std::size_t count) noexcept;
extern template void BoysRegionASimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
extern template void BoysRegionBSimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
extern template void BoysRegionCSimdBf16<kBoysFullAccuracyMultiplier>(
    int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

// ---------------------------------------------------------------------------
// The all-orders batch over an argument array (BoysAllN)
// ---------------------------------------------------------------------------
// Every dispatch path is an interval of the argument line and the intervals are
// ordered, so the classification is monotone in x: a non-decreasing array is
// already contiguous by path, which is what the BoysSortedArgs overload
// declares and why one comparison per argument is the honest statement of it.
//
// A path is served by one of two kernel shapes. The ungrouped shape walks one
// argument at a time and writes the caller's planes; its bodies are the
// per-argument entry's own (BoysAllOrdersImpl) called at the batch's nmax,
// restructured onto the caller's layout and kept in lockstep with it, so at
// m = 1 the batch values ARE that entry's values, bit for bit, on every path it
// serves. (Region A's body seeds at nmax and recurses down, so a batch's F_k for
// k < nmax is that recurrence's value, not the one a per-order call at nmax = k
// walks; both are inside the entry's bound.) The grouped shape runs the
// region-A lane over a homogeneous run, one order at a time (the lane's shape is
// per order), staged through a fixed stack frame whatever count is.
//
// Which shape serves region A is decided by measurement, because the lane pays
// for every order separately: it evaluates each order from that order's own
// fit, while the scalar body evaluates one seed and recurses down. On a
// 65536-argument run, entry against a plain per-argument loop over the same
// arguments, the lane is 2.6x faster serving F_0, at parity at n = 4, and 10 to
// 20% slower from n = 8 up. kBoysAllNLaneMaxOrder is where the two cross.
//
// The zero path and region C run scalar in both shapes: the asymptotic form is
// a seed and nmax recursion steps written into the caller's planes, which is
// less work than the per-order lane, and the scalar body is the per-argument
// path's own.
constexpr std::size_t kBoysAllNChunk = 128;
constexpr int kBoysAllNLaneMaxOrder = 4;

enum class BoysPath : std::uint8_t {
    kZero = 0,
    kTabulated = 1,
    kExtended = 2,
    kMiddle = 3,
    kAsymptotic = 4,
};

constexpr int kBoysPathCount = 5;

// The path of one argument, in the per-argument entry's dispatch order. The
// region-A band split arrives as the caller's tierSplit: at m = 1 the scalar
// dispatch splits the band at the first extended-seed tier and that split is a
// path of its own, at m > 1 the band does not exist and the whole region is one
// path.
inline BoysPath BoysClassifyPath(double x, double tierSplit) noexcept {
    if (x == 0.0)
    {
        return BoysPath::kZero;
    }

    if (x < tierSplit)
    {
        return BoysPath::kTabulated;
    }

    if (x < kX0)
    {
        return BoysPath::kExtended;
    }

    if (x < kX1)
    {
        return BoysPath::kMiddle;
    }

    return BoysPath::kAsymptotic;
}

// The band split of the multiplier in use: the classification has to reproduce
// the per-argument entry's dispatch, band and all, and to stay a split of the
// argument line whatever the multiplier is.
template <double kAccuracyMultiplier> constexpr double BoysAllNTierSplit() noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return kTierThresholds[0];
    } else
    {
        return kX0;
    }
}

// out[order * count + i] = values[t], for the run's t-th argument. The
// unsorted entry's runs carry their permutation; the sorted overload's runs are
// the caller's own order, so its plane writes are sequential.
inline void BoysAllNScatter(int order,
                            std::size_t count,
                            const std::size_t* index,
                            std::size_t begin,
                            std::size_t block,
                            const double* values,
                            double* out) noexcept {
    double* plane = out + static_cast<std::size_t>(order) * count;

    if (index == nullptr)
    {
        for (std::size_t t = 0; t < block; ++t)
        {
            plane[begin + t] = values[t];
        }

        return;
    }

    for (std::size_t t = 0; t < block; ++t)
    {
        plane[index[begin + t]] = values[t];
    }
}

// One argument's column of the caller's planes: plane[k * count] = F_k(x) for
// k = 0..nmax. These bodies are the per-argument entry's, restructured onto the
// caller's layout; keep the two in lockstep.
inline void BoysAllNBodyZero(int nmax, std::size_t count, double* plane) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * count] = 1.0 / (2.0 * l + 1.0);
    }
}

// Region A below the band: the downward recursion from the per-order fit.
template <double kAccuracyMultiplier>
inline void BoysAllNBodyRegionADown(int nmax, double x, std::size_t count, double* plane) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        double f = ChebyshevValue(nmax, x);
        plane[static_cast<std::size_t>(nmax) * count] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    } else
    {
        static constexpr auto kDegreesA =
            RegionADegrees<kAccuracyMultiplier, BoysRole::kDoubleBatch>();
        double f = ChebyshevValueWithDegrees(nmax, x, kDegreesA);
        plane[static_cast<std::size_t>(nmax) * count] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = nmax - 1; l >= 0; --l)
        {
            f = (x * f + expx) / (l + 0.5);
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    }
}

// Region A's extended band: the per-range seed + upward recursion, with the
// per-order fits for the orders the tier does not reach. The classification
// admits only x >= kTierThresholds[0] here, which is the per-argument entry's
// guard on that condition - the band is this path.
template <double kAccuracyMultiplier>
inline void BoysAllNBodyRegionAExtended(int nmax,
                                        double x,
                                        std::size_t count,
                                        double* plane) noexcept {
    static_assert(kAccuracyMultiplier == 1.0,
                  "the extended band is a path of the m = 1 dispatch only");

    int served = 0;

    while (served < nmax && x >= kTierThresholds[static_cast<std::size_t>(served + 1)])
    {
        ++served;
    }

    double f = RegionBExtendedSeed(x);
    plane[0] = f;
    const double expx = 0.5 * std::exp(-x);

    for (int l = 1; l <= served; ++l)
    {
        f = ((l - 0.5) * f - expx) / x;
        plane[static_cast<std::size_t>(l) * count] = f;
    }

    for (int l = served + 1; l <= nmax; ++l)
    {
        plane[static_cast<std::size_t>(l) * count] = ChebyshevValue(l, x);
    }
}

// Region B: the F0 seed + upward recursion.
template <double kAccuracyMultiplier>
inline void BoysAllNBodyRegionB(int nmax, double x, std::size_t count, double* plane) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        double f = RegionBSeed(x);
        plane[0] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((l - 0.5) * f - expx) / x;
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    } else
    {
        // The seed degree must be the order-0 entry: the seed's truncation
        // error reaches every output order with amplification A_B(l), at most
        // 1 + 1.846e-17 over the supported orders (its maximum at l = 32).
        static constexpr auto kDegreesB =
            RegionBDegrees<kAccuracyMultiplier, BoysRole::kDoubleBatch>();
        double f = RegionBSeedWithDegrees(x, kDegreesB[0]);
        plane[0] = f;
        const double expx = 0.5 * std::exp(-x);

        for (int l = 1; l <= nmax; ++l)
        {
            f = ((l - 0.5) * f - expx) / x;
            plane[static_cast<std::size_t>(l) * count] = f;
        }
    }
}

// Region C: the asymptotic form (m-invariant).
inline void BoysAllNBodyRegionC(int nmax, double x, std::size_t count, double* plane) noexcept {
    double f = kBoysHalfSqrtPi / std::sqrt(x);
    plane[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        f = (l - 0.5) * f / x;
        plane[static_cast<std::size_t>(l) * count] = f;
    }
}

// The ungrouped kernel: one argument at a time over the run, the path's body
// per argument, the caller's planes.
template <double kAccuracyMultiplier, BoysPath kPath>
void BoysAllNRunUngrouped(int nmax,
                          const double* x,
                          double* out,
                          std::size_t count,
                          const std::size_t* index,
                          std::size_t begin,
                          std::size_t end) noexcept {
    for (std::size_t j = begin; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;
        const double xi = x[i];
        double* plane = out + i;

        if constexpr (kPath == BoysPath::kZero)
        {
            BoysAllNBodyZero(nmax, count, plane);
        }
        else if constexpr (kPath == BoysPath::kTabulated)
        {
            BoysAllNBodyRegionADown<kAccuracyMultiplier>(nmax, xi, count, plane);
        }
        else if constexpr (kPath == BoysPath::kExtended)
        {
            BoysAllNBodyRegionAExtended<kAccuracyMultiplier>(nmax, xi, count, plane);
        }
        else if constexpr (kPath == BoysPath::kMiddle)
        {
            BoysAllNBodyRegionB<kAccuracyMultiplier>(nmax, xi, count, plane);
        }
        else
        {
            BoysAllNBodyRegionC(nmax, xi, count, plane);
        }
    }
}

// The grouped kernel: the region-A lane over a homogeneous run, chunked so the
// staging planes stay inside a fixed stack frame. Both region-A paths take this
// shape when they take it at all - the lane covers [0, kX0), so which of the
// two scalar bodies the sort's path split assigned an argument to does not
// choose the kernel; the lane serves the band as well, at its band accuracy.
//
// Region B keeps its scalar body. BoysRegionBSimd does not hold its own
// documented bound across the region - at n = 32, x = kX0 it lands 2.2e-13 from
// the reference against its 5.5e-14 - and the scalar body is the per-argument
// path's own, exact to the bit.
inline void BoysAllNRunGrouped(int nmax,
                               const double* x,
                               double* out,
                               std::size_t count,
                               const std::size_t* index,
                               std::size_t begin,
                               std::size_t end) noexcept {
    std::array<double, (kMaxBoysOrder + 1) * kBoysAllNChunk> stage;
    std::array<double, kBoysAllNChunk> args;

    for (std::size_t base = begin; base < end; base += kBoysAllNChunk)
    {
        const std::size_t block = std::min(kBoysAllNChunk, end - base);

        for (std::size_t t = 0; t < block; ++t)
        {
            args[t] = x[(index != nullptr) ? index[base + t] : base + t];
        }

        // The region-A lane is per order, so the entry asks it for one order at
        // a time, which is the shape the region-A fits have.
        for (int l = 0; l <= nmax; ++l)
        {
            BoysRegionASimd(l, args.data(), stage.data(), block);
            BoysAllNScatter(l, count, index, base, block, stage.data(), out);
        }
    }
}

// One run of one path, by the shape the tier can serve.
template <double kAccuracyMultiplier>
void BoysAllNRun(int nmax,
                 const double* x,
                 const std::size_t* index,
                 double* out,
                 std::size_t count,
                 bool vectorTier,
                 BoysPath path,
                 std::size_t begin,
                 std::size_t end) noexcept {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        // The lane serves region A below the crossover order; at and above it
        // the scalar body is the faster of the two.
        if (vectorTier && nmax <= kBoysAllNLaneMaxOrder &&
            (path == BoysPath::kTabulated || path == BoysPath::kExtended))
        {
            BoysAllNRunGrouped(nmax, x, out, count, index, begin, end);
            return;
        }
    }

    switch (path)
    {
    case BoysPath::kZero:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kZero>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kTabulated:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kTabulated>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kExtended:
        if constexpr (kAccuracyMultiplier == 1.0)
        {
            BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kExtended>(
                nmax, x, out, count, index, begin, end);
        }
        break;

    case BoysPath::kMiddle:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kMiddle>(
            nmax, x, out, count, index, begin, end);
        break;

    case BoysPath::kAsymptotic:
        BoysAllNRunUngrouped<kAccuracyMultiplier, BoysPath::kAsymptotic>(
            nmax, x, out, count, index, begin, end);
        break;
    }
}

// The per-argument path, in the caller's planes - the shape the C entry point's
// batch loop already has, and the batch entry's total fallback.
template <double kAccuracyMultiplier>
void BoysAllNRunPerArgument(int nmax, const double* x, double* out, std::size_t count) noexcept {
    for (std::size_t i = 0; i < count; ++i)
    {
        double row[kMaxBoysOrder + 1];
        BoysAllOrdersImpl<kAccuracyMultiplier>(nmax, x[i], row);

        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * count + i] = row[static_cast<std::size_t>(l)];
        }
    }
}

template <double kAccuracyMultiplier>
void BoysAllNImpl(int nmax,
                  const double* x,
                  double* out,
                  std::size_t count,
                  std::size_t* workspace) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    const bool vectorTier = (kAccuracyMultiplier == 1.0) && BoysAvx2Available();
    const double tierSplit = BoysAllNTierSplit<kAccuracyMultiplier>();

    std::size_t* owned = nullptr;

    if (workspace == nullptr)
    {
        owned = new (std::nothrow) std::size_t[count];
        workspace = owned;
    }

    if (workspace == nullptr)
    {
        // Total by construction: the per-argument path, the same values at the
        // same bound, without the grouping.
        BoysAllNRunPerArgument<kAccuracyMultiplier>(nmax, x, out, count);
        return;
    }

    // The counting sort over the dispatch paths. The paths are ordered
    // intervals, so this both groups a shuffled array and leaves a non-decreasing
    // one in place - there is no separate already-grouped case to detect.
    std::size_t counts[kBoysPathCount] = {};

    for (std::size_t i = 0; i < count; ++i)
    {
        assert(x[i] >= 0.0);
        ++counts[static_cast<std::size_t>(BoysClassifyPath(x[i], tierSplit))];
    }

    std::size_t starts[kBoysPathCount + 1] = {};
    std::size_t cursor[kBoysPathCount] = {};

    for (int p = 0; p < kBoysPathCount; ++p)
    {
        starts[static_cast<std::size_t>(p) + 1] =
            starts[static_cast<std::size_t>(p)] + counts[static_cast<std::size_t>(p)];
        cursor[static_cast<std::size_t>(p)] = starts[static_cast<std::size_t>(p)];
    }

    for (std::size_t i = 0; i < count; ++i)
    {
        const BoysPath path = BoysClassifyPath(x[i], tierSplit);
        workspace[cursor[static_cast<std::size_t>(path)]++] = i;
    }

    for (int p = 0; p < kBoysPathCount; ++p)
    {
        const std::size_t from = starts[static_cast<std::size_t>(p)];
        const std::size_t to = starts[static_cast<std::size_t>(p) + 1];

        if (from != to)
        {
            BoysAllNRun<kAccuracyMultiplier>(
                nmax, x, workspace, out, count, vectorTier, static_cast<BoysPath>(p), from, to);
        }
    }

    delete[] owned;
}

template <double kAccuracyMultiplier>
void BoysAllNSortedImpl(int nmax, const double* x, double* out, std::size_t count) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(count == 0 || x != nullptr);
    assert(count == 0 || out != nullptr);

    if (count == 0)
    {
        return;
    }

    const bool vectorTier = (kAccuracyMultiplier == 1.0) && BoysAvx2Available();
    const double tierSplit = BoysAllNTierSplit<kAccuracyMultiplier>();

    assert(x[0] >= 0.0);

    // The runs are found by the classification itself rather than declared, so
    // every run is homogeneous whatever the array is: a caller that declared an
    // order it did not have pays the run boundaries, not a wrong value.
    std::size_t begin = 0;

    while (begin < count)
    {
        const BoysPath path = BoysClassifyPath(x[begin], tierSplit);
        std::size_t end = begin + 1;

        while (end < count && BoysClassifyPath(x[end], tierSplit) == path)
        {
            assert(x[end] >= x[end - 1]); // the BoysSortedArgs declaration
            ++end;
        }

        BoysAllNRun<kAccuracyMultiplier>(nmax, x, nullptr, out, count, vectorTier, path, begin, end);
        begin = end;
    }
}

} // namespace detail

// ---------------------------------------------------------------------------
// The public template entries (declared in boys/boys.hpp)
// ---------------------------------------------------------------------------

template <double kAccuracyMultiplier> double BoysSingle(int n, double x) noexcept {
    return detail::BoysSingleImpl<kAccuracyMultiplier>(n, x);
}

template <double kAccuracyMultiplier> void BoysAllOrders(int nmax, double x, double* out) noexcept {
    detail::BoysAllOrdersImpl<kAccuracyMultiplier>(nmax, x, out);
}

template <double kAccuracyMultiplier>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    detail::BoysFixedNImpl<kAccuracyMultiplier>(n, x, out, count, stride);
}

template <double kAccuracyMultiplier>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace) noexcept {
    detail::BoysAllNImpl<kAccuracyMultiplier>(nmax, x, out, count, workspace);
}

template <double kAccuracyMultiplier>
void BoysAllN(int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept {
    detail::BoysAllNSortedImpl<kAccuracyMultiplier>(nmax, x, out, count);
}

template <double kAccuracyMultiplier> float BoysSingleF32(int n, float x) noexcept {
    return detail::BoysSingleF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFloat>(n, x);
}

template <double kAccuracyMultiplier> void BoysAllOrdersF32(int nmax, float x, float* out) noexcept {
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFloat>(nmax, x, out);
}

#if BoysFp16
// The fp16/bf16 lanes forward the multiplier to the F32 engine with the
// fp16 computation budget (the m*1e-7 + 1/2-ULP formula); at m = 1 the
// engine branch is the certified F32 path verbatim, so the lanes are
// bit-unchanged. The half-ULP representation term is m-independent.

template <double kAccuracyMultiplier> F16 BoysSingleF16(int n, F16 x) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    return static_cast<F16>(
        detail::BoysSingleF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
            n, static_cast<float>(x)));
}

template <double kAccuracyMultiplier> void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
        nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<F16>(scratch[l]);
    }
}

template <double kAccuracyMultiplier> Bf16 BoysSingleBf16(int n, Bf16 x) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    return static_cast<Bf16>(
        detail::BoysSingleF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
            n, static_cast<float>(x)));
}

template <double kAccuracyMultiplier> void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysAllOrdersF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
        nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<Bf16>(scratch[l]);
    }
}
#endif // BoysFp16

} // namespace boys
