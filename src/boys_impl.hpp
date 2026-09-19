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

#include <cassert>
#include <cmath>
#include <cstddef>

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
// BoysBatch(n,x)[n] == BoysSingle(n,x) == BoysBatch(m>=n,x)[n] exactly.
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

template <double kAccuracyMultiplier> void BoysBatchImpl(int nmax, double x, double* out) noexcept {
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
            // amplification A_B(l) = prod(j+1/2)/x0^l <= A_B(0) = 1 (the
            // maximum over l <= 32; A_B(32) ~ 1). The seed degree must
            // therefore be the order-0 entry — the one whose criterion is
            // Delta(d') * 1 <= (m-1) * B. The per-order entries kDegreesB[n]
            // (amplification A_B(n) < 1 at small n) bound only the order-n
            // single-style path; seeding with kDegreesB[nmax] leaves F0's
            // error at Delta(d'(nmax)) — up to (m-1)*B/A_B(nmax) — unbounded.
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
void BoysBatchF32Impl(int nmax, float x, float* out) noexcept {
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
        // output with amplification <= A_B(0) = 1, so the order-0 entry is
        // the one that bounds the whole batch (kDegreesB[nmax] is loose:
        // A_B(nmax) < 1 at small nmax, and the entry is 0 for many orders
        // at m >= 1e4 — a degree-0 seed error ~5e-2 on F0).
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

} // namespace detail

// ---------------------------------------------------------------------------
// The public template entries (declared in boys/boys.hpp)
// ---------------------------------------------------------------------------

template <double kAccuracyMultiplier> double BoysSingle(int n, double x) noexcept {
    return detail::BoysSingleImpl<kAccuracyMultiplier>(n, x);
}

template <double kAccuracyMultiplier> void BoysBatch(int nmax, double x, double* out) noexcept {
    detail::BoysBatchImpl<kAccuracyMultiplier>(nmax, x, out);
}

template <double kAccuracyMultiplier>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    detail::BoysFixedNImpl<kAccuracyMultiplier>(n, x, out, count, stride);
}

template <double kAccuracyMultiplier> float BoysSingleF32(int n, float x) noexcept {
    return detail::BoysSingleF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFloat>(n, x);
}

template <double kAccuracyMultiplier> void BoysBatchF32(int nmax, float x, float* out) noexcept {
    detail::BoysBatchF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFloat>(nmax, x, out);
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

template <double kAccuracyMultiplier> void BoysBatchF16(int nmax, F16 x, F16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<F16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysBatchF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
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

template <double kAccuracyMultiplier> void BoysBatchBf16(int nmax, Bf16 x, Bf16* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= static_cast<Bf16>(0.0f));
    assert(out != nullptr);

    float scratch[kMaxBoysOrder + 1];
    detail::BoysBatchF32Impl<kAccuracyMultiplier, detail::BoysBudget::kFp16>(
        nmax, static_cast<float>(x), scratch);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = static_cast<Bf16>(scratch[l]);
    }
}
#endif // BoysFp16

} // namespace boys
