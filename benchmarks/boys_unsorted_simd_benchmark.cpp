// The unsorted-SIMD lane: one kernel evaluating all three region paths per vector, blended per lane
// — the cost of an unsorted input stream against the region-sorted lanes of the companion
// benchmark. No such lane is shipped: the library partitions arguments by region first, so every
// 4-lane vector is homogeneous, and the parts measured here are the shipped kernel's own.
#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"

// The shipped AVX2 unit, compiled into this one so that the table, Eval4 and Clenshaw4Split below
// ARE those definitions and not lookalikes. Four of that unit's definitions are the library's in
// this build, which this target links as well, so they are renamed for the length of the include;
// a new external definition there needs the same treatment, and the duplicate-symbol error asks.
#define BoysAvx2Available BoysAvx2AvailableFromShippedUnit
#define AppendPackedBackends AppendPackedBackendsFromShippedUnit
#define DetectF16c DetectF16cFromShippedUnit
#define F16cAvailable F16cAvailableFromShippedUnit
// The nine region entries stopped being templates when the accuracy rung was removed, so
// they no longer reach this unit as instantiations it may leave to the library: each is a
// definition of the library's own name now, and is renamed here for the length of the
// include, which is the treatment the note above asks for.
#define BoysRegionASimd BoysRegionASimdFromShippedUnit
#define BoysRegionASimdF16 BoysRegionASimdF16FromShippedUnit
#define BoysRegionASimdBf16 BoysRegionASimdBf16FromShippedUnit
#define BoysRegionBSimd BoysRegionBSimdFromShippedUnit
#define BoysRegionBSimdF16 BoysRegionBSimdF16FromShippedUnit
#define BoysRegionBSimdBf16 BoysRegionBSimdBf16FromShippedUnit
#define BoysRegionCSimd BoysRegionCSimdFromShippedUnit
#define BoysRegionCSimdF16 BoysRegionCSimdF16FromShippedUnit
#define BoysRegionCSimdBf16 BoysRegionCSimdBf16FromShippedUnit
#include "boys_simd.cpp"
#undef BoysRegionASimdF16
#undef BoysRegionASimdBf16
#undef BoysRegionASimd
#undef BoysRegionBSimdF16
#undef BoysRegionBSimdBf16
#undef BoysRegionBSimd
#undef BoysRegionCSimdF16
#undef BoysRegionCSimdBf16
#undef BoysRegionCSimd
#undef F16cAvailable
#undef DetectF16c
#undef AppendPackedBackends
#undef BoysAvx2Available

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <random>
#include <vector>

namespace {

constexpr std::size_t kInputCount = 1u << 22; // 4,194,304 values
constexpr int kOrder = 8; // the order the whole array is evaluated at
constexpr int kPasses = 3;

// All three region paths per vector, blended per lane: the divergence penalty.
void ChebSimdMixed(int n,
                   const double* x,
                   double* out,
                   std::size_t count,
                   const boys::detail::ExpTable& expTable) {
    using namespace boys::detail;

    for (std::size_t i = 0; i + 3 < count; i += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + i);
        const __m256d mA = _mm256_cmp_pd(xv, _mm256_set1_pd(kX0), _CMP_LT_OQ);
        const __m256d mB = _mm256_cmp_pd(xv, _mm256_set1_pd(kX1), _CMP_LT_OQ);
        const __m256d mBonly = _mm256_andnot_pd(mA, mB);

        __m256d fA = _mm256_setzero_pd();
        {
            const int s = kPieceStart[n];
            const int e = kPieceStart[n + 1];

            for (int p = s; p < e; ++p)
            {
                const OrderPiece& piece = kPieces[p];
                const __m256d mask = _mm256_and_pd(
                    _mm256_and_pd(mA, _mm256_cmp_pd(xv, _mm256_set1_pd(piece.a), _CMP_GE_OQ)),
                    _mm256_cmp_pd(xv, _mm256_set1_pd(piece.b), _CMP_LT_OQ));
                fA = _mm256_blendv_pd(fA, boys::detail::Clenshaw4Split(piece, xv), mask);
            }
        }

        __m256d fB = _mm256_setzero_pd();
        {
            // Split Clenshaw (even/odd) on kBcoeffs, the shipped region-B F0 seed shape.
            const double* c = kBcoeffs.data();
            const int m = kBDeg / 2;
            __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(kX0));
            t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (kX1 - kX0)), _mm256_set1_pd(-1.0));
            const __m256d v =
                _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
            const __m256d twoV = _mm256_add_pd(v, v);
            __m256d b1 = _mm256_set1_pd(c[2 * m]);
            __m256d b2 = _mm256_setzero_pd();

            for (int k = m - 1; k >= 1; --k)
            {
                const __m256d b0 =
                    _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k]), b2));
                b2 = b1;
                b1 = b0;
            }

            const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));
            __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
            __m256d o2 = _mm256_setzero_pd();

            for (int k = m - 2; k >= 1; --k)
            {
                const __m256d o0 =
                    _mm256_fmadd_pd(twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[2 * k + 1]), o2));
                o2 = o1;
                o1 = o0;
            }

            const __m256d odd = _mm256_fmadd_pd(_mm256_sub_pd(twoV, _mm256_set1_pd(1.0)),
                                                o1,
                                                _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
            __m256d f = _mm256_fmadd_pd(t, odd, even);
            const __m256d expx = expTable.Eval4(xv);
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_fmadd_pd(
                    _mm256_set1_pd(l + 0.5), f, _mm256_mul_pd(_mm256_set1_pd(-0.5), expx));
                f = _mm256_mul_pd(f, invx);
            }

            fB = f;
        }

        __m256d fC = _mm256_setzero_pd();
        {
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
            __m256d f = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_sqrt_pd(invx));

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_mul_pd(_mm256_mul_pd(f, _mm256_set1_pd(l + 0.5)), invx);
            }

            fC = f;
        }

        __m256d res = _mm256_blendv_pd(fC, fB, mBonly);
        res = _mm256_blendv_pd(res, fA, mA);
        _mm256_storeu_pd(out + i, res);
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = boys::BoysSingle(n, x[i]);
    }
}

double MaxAbsError(const double* out, const double* x, std::size_t count, double* worstX) {
    double worst = 0.0;
    double xAtWorst = 0.0;

    for (std::size_t i = 0; i < count; ++i)
    {
        const double err = std::abs(out[i] - boys::BoysSingle(kOrder, x[i]));

        if (err > worst)
        {
            worst = err;
            xAtWorst = x[i];
        }
    }

    *worstX = xAtWorst;
    return worst;
}

} // namespace

int main(int argc, char** argv) {
    const bool selfCheck = argc > 1 && std::strcmp(argv[1], "--self-check") == 0;

    // The unsorted workload: x uniform in [0, 40], rng(46), order 8 throughout.
    std::mt19937_64 rng(46);
    std::uniform_real_distribution<double> xd(0.0, 40.0);
    std::vector<double> x(kInputCount);

    for (auto& value : x)
    {
        value = xd(rng);
    }

    std::vector<double> out(kInputCount);
    const boys::detail::ExpTable expTable;

    if (selfCheck)
    {
        ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
        double xAtWorst = 0.0;
        const double maxErr = MaxAbsError(out.data(), x.data(), kInputCount, &xAtWorst);
        const bool pass = maxErr <= 5.5e-14;
        std::printf(
            "self-check: cheb-simd-unsorted-n8 | max_abs_err: %.3e | budget: 5.50e-14 | %s\n",
            maxErr,
            pass ? "PASS" : "FAIL");
        std::printf("  worst at x = %.9f\n", xAtWorst);
        return pass ? 0 : 1;
    }

    // Measurement protocol: warmup + 3 passes, min/median/max.
    ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
    std::vector<double> passes(kPasses);

    for (int p = 0; p < kPasses; ++p)
    {
        const auto t0 = std::chrono::steady_clock::now();
        ChebSimdMixed(kOrder, x.data(), out.data(), kInputCount, expTable);
        const auto t1 = std::chrono::steady_clock::now();
        passes[p] = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }

    std::sort(passes.begin(), passes.end());
    const double median = passes[1];
    // count/median is items per millisecond, i.e. 1e3 items/s; the /1e3 below
    // makes the printed value the unit its field names (Mvals/s).
    std::printf(
        "kernel: cheb-simd-unsorted-n8 | workload: uniform-x40-n8 | count: %zu | passes: %d | "
        "min_ms: %.3f | median_ms: %.3f | max_ms: %.3f | median_Mvals_per_s: %.3f\n",
        kInputCount,
        kPasses,
        passes.front(),
        median,
        passes.back(),
        kInputCount / median / 1e3);
    return 0;
}
