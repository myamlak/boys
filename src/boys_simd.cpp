#include "boys/boys.hpp"
#include "boys_coefficients.hpp"
#include "boys_effective_degrees.hpp"
#include "boys_impl.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// --- Architecture guard -----------------------------------------------------
//
// The AVX2 tier is x86_64-only *by construction*: the kernels are AVX2/FMA
// intrinsics and CMake compiles this TU with /arch:AVX2 (MSVC) or
// -mavx2 -mfma -mf16c (GCC/Clang). Everything below the #if is that tier.
//
// BOYS_SIMD_X86 answers one question — does this TU compile the vector tier?
// — and there are two ways to reach that answer, in this order:
//
//   1. The build states it. CMakeLists.txt derives it from a configure-time
//      compiler probe (check_cxx_source_compiles) and defines it on the
//      `boys` target, so the flag set and this guard are one decision rather
//      than two that can drift apart. An explicit answer, 1 or 0, is final.
//   2. Nobody states it, so the guard detects it from the compiler's own
//      predefines — the two macros the CMakeLists probe asks about. This is
//      the case for a consumer that compiles this source into a target of its
//      own instead of linking `boys` (a consumer target does exactly
//      that), and it is why such a consumer no longer has to repeat the
//      build's macro. It is detection, not a guess: the two spellings below
//      are the whole x86_64 question, answered by the compiler that is doing
//      the compiling. Detection answers the architecture question only — a
//      consumer that compiles this TU itself still owns the AVX2/FMA/F16C
//      flag set.
//
// Anything neither stated nor x86_64 is an ERROR rather than a quiet 0. This
// TU cannot tell a genuine non-x86 target from an x86 target whose predefines
// it has not been taught, and the failure this guard exists to prevent is a
// vector tier that goes silently dead: on a target that IS x86_64, answering
// 0 would compile the intrinsics out, leave every SIMD correctness test
// soft-skipping, and leave CI green. A build that knows it is not x86_64 says
// so — CMakeLists.txt does, on the same probe — and gets the scalar lanes.
//
// On a non-x86 target the same twelve entry points and BoysAvx2Available()
// below are defined against the certified scalar lanes instead: same
// signatures, same contracts, same numerics as the scalar tails the x86
// entries already run for their last count % 4 elements — the vector engine
// is absent, the results are not. BoysAvx2Available() reports false there,
// and the CI legs assert that per architecture.
#ifdef BOYS_SIMD_X86

// The build answered; nothing to detect.

#elif defined(__x86_64__) || defined(_M_X64)

// Unstated, but the compiler says x86_64 — the same two macros the
// CMakeLists probe tests for, so the detected answer and the probed one agree.

#define BOYS_SIMD_X86 1

#else

#error "BOYS_SIMD_X86 is unset and this is not x86_64: define it (1 on x86_64, 0 elsewhere)"

#endif

#if BOYS_SIMD_X86

#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// AVX2 region-sorted lanes. The engine pattern (region-first; the
// unsorted variant's divergence penalty is reported in the accompanying
// paper): partition the arguments by region FIRST so every
// 4-lane vector is homogeneous; the unsorted variant pays a measured 2.3x
// divergence penalty.
//
// Callers must check BoysAvx2Available() before invoking these; the
// translation unit is compiled with /arch:AVX2 and the kernels are FMA
// chains, so the predicate requires the FMA feature bit as well.
//
// Accuracy-multiplier treatment: every region entry is a
// template on kAccuracyMultiplier, compiled twice under if constexpr — the
// m = 1 branch is today's body verbatim (the bit-identity pin; the only
// m = 1 differences are the scalar tails calling the templated scalar
// entries at m = 1, which resolve to the same functions); the relaxed
// branch passes the per-piece / per-order effective degrees into the
// degree-parameterized Clenshaw variants (the loop-bound load shape is
// unchanged — only the degree constant source differs). The relaxed
// region-B exp-Taylor table is untouched (its error 1.83e-17 sits ~2700x
// below the 5e-14 target and is m-independent). The default m = 1
// instantiations of the region entries live at the bottom of this TU (the
// extern-template declarations in boys.hpp).

namespace boys::detail {
namespace {

using detail::kX0;
using detail::kX1;

constexpr double kHalfSqrtPi = 0.886226925452758014;

// CPUID + OSXSAVE detection of the AVX2 + FMA scope (the F64/F32 lanes'
// engine). FMA belongs to the scope, it is not an optional extra: this
// translation unit is compiled with /arch:AVX2 (MSVC) or -mavx2 -mfma
// (GCC/Clang) and its kernels are FMA chains (the split-Clenshaw recurrences,
// the region-B upward recursion, the e^{-x} Horner), so a processor that has
// AVX2 without FMA would be dispatched into an unimplemented instruction.
bool DetectAvx2() noexcept {
#ifdef _MSC_VER
    int cpuInfo[4] = {};
    int leaf1[4] = {};
    __cpuidex(cpuInfo, 7, 0);
    __cpuid(leaf1, 1);
    const bool osXsave = (leaf1[2] & (1u << 27)) != 0; // OSXSAVE
    const bool fma = (leaf1[2] & (1u << 12)) != 0; // FMA
    const bool avx2 = (cpuInfo[1] & (1u << 5)) != 0;
    return osXsave && fma && avx2;
#else
    // GCC/Clang builds: the kernel enables the AVX XCR0 state whenever
    // the CPU supports it, so the leaf-1 feature bits are authoritative (the
    // same check the MSVC branch performs).
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    const bool osXsave = (ecx & (1u << 27)) != 0; // OSXSAVE
    const bool fma = (ecx & (1u << 12)) != 0; // FMA

    if (!osXsave || !fma)
    {
        return false;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    const bool avx2 = (ebx & (1u << 5)) != 0;
    return avx2;
#endif
}

// ---------------------------------------------------------------------------
// e^{-x} on [0, 30]: a degree-4 Taylor table, one row per grid abscissa
// x_i = i * kStep, rows padded to 8 doubles so the gathers can use the legal
// scale 8. On [x0, x1] the measured worst absolute error is 1.83e-17
// (x ~= 11.99) - still ~2700x below the 5e-14 target.
//
// A row holds the quartic Taylor polynomial of e^{-x} at x_i, written in the
// monomial basis of the ABSOLUTE argument x so that Eval4 is a plain Horner
// chain. Split e^{-x} = e^{-x_i} e^{-h} at h = x - x_i and expand the binomial
// powers of h = x - x_i; the coefficient of x^k is
//
//     a_k = e^{-x_i} * (-1)^k * S_{4-k} / k!,   S_m = sum_{j=0..m} x_i^j / j!.
//
// The row stores (-1)^k a_k = e^{-x_i} S_{4-k} / k!, i.e. the alternating sign
// is folded into the table and taken back out by the sign pattern of Eval4's
// FMA chain. Elementary Taylor expansion of the exponential; the table seeds
// the Boys kernel [Boys1950].
// ---------------------------------------------------------------------------
class ExpTable {
public:
    static constexpr double kStep = 0.01;
    static constexpr int kNumPoints = 3000;
    static constexpr int kDegree = 4;

    ExpTable() noexcept {
        for (int i = 0; i <= kNumPoints; ++i)
        {
            const double x = i * kStep;
            const double decay = std::exp(-x);

            // partialSum[m] = sum_{j=0..m} x^j / j!, built upward on the
            // running term x^j / j! (one multiply-and-divide per step).
            double term = 1.0;
            double running = 1.0;
            double partialSum[kDegree + 1];
            partialSum[0] = 1.0;

            for (int m = 1; m <= kDegree; ++m)
            {
                term *= x / m;
                running += term;
                partialSum[m] = running;
            }

            // Row entry k = e^{-x_i} * S_{4-k} / k!.
            double factorial = 1.0;

            for (int k = 0; k <= kDegree; ++k)
            {
                _coefficients[i][k] = decay * partialSum[kDegree - k] / factorial;
                factorial *= k + 1;
            }
        }
    }

    __m256d Eval4(__m256d x) const noexcept {
        __m128i index = _mm256_cvtpd_epi32(_mm256_mul_pd(x, _mm256_set1_pd(1.0 / kStep)));
        index = _mm_min_epi32(index, _mm_set1_epi32(kNumPoints));
        // Rows are 8 doubles apart; gather indices are double offsets, so the
        // grid index scales by 8 (scale 8 bytes x index = row address).
        index = _mm_slli_epi32(index, 3);
        __m256d c0 = _mm256_i32gather_pd(&_coefficients[0][0], index, 8);
        __m256d c1 = _mm256_i32gather_pd(&_coefficients[0][1], index, 8);
        __m256d c2 = _mm256_i32gather_pd(&_coefficients[0][2], index, 8);
        __m256d c3 = _mm256_i32gather_pd(&_coefficients[0][3], index, 8);
        __m256d c4 = _mm256_i32gather_pd(&_coefficients[0][4], index, 8);
        // The stored coefficients are the ALTERNATING-sign monomial form of
        // the Taylor sum sum_k (z - x)^k / k!: evaluate
        // c4*x^4 - c3*x^3 + c2*x^2 - c1*x + c0 (the stored coefficient order).
        __m256d result = _mm256_fmsub_pd(c4, x, c3);
        result = _mm256_fmadd_pd(result, x, c2);
        result = _mm256_fmsub_pd(result, x, c1);
        result = _mm256_fmadd_pd(result, x, c0);
        return result;
    }

private:
    double _coefficients[kNumPoints + 1][8]{};
};

// Split Clenshaw (even/odd), 4-wide, half-depth FMA chains. See boys.cpp for
// the scalar derivation; T_{2j+1}(t) = t * D_j(v) with the D recurrence.
// 4-wide split Clenshaw for one piece; even deg >= 4 only (see boys.cpp).
__m256d Clenshaw4SplitDeg(const detail::OrderPiece& piece, int deg, __m256d xv) noexcept {
    const double* c = detail::kCoeffs.data() + piece.offset;

    __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(piece.a));
    t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (piece.b - piece.a)), _mm256_set1_pd(-1.0));

    if (deg == 0)
    {
        return _mm256_set1_pd(c[0]);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_pd(t, _mm256_set1_pd(c[1]), _mm256_set1_pd(c[0]));
    }

    const __m256d v =
        _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
    const __m256d twoV = _mm256_add_pd(v, v);

    const int m = deg / 2;
    __m256d b1 = _mm256_set1_pd(c[std::ptrdiff_t{2} * m]);
    __m256d b2 = _mm256_setzero_pd();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256d b0 =
            _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));

    __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
    __m256d o2 = _mm256_setzero_pd();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256d o0 = _mm256_fmadd_pd(
            twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256d odd = _mm256_fmadd_pd(
        _mm256_sub_pd(twoV, _mm256_set1_pd(1.0)), o1, _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
    return _mm256_fmadd_pd(t, odd, even);
}

// The m = 1 entry: the piece's full degree, as before the parametrization.
__m256d Clenshaw4Split(const detail::OrderPiece& piece, __m256d xv) noexcept {
    return Clenshaw4SplitDeg(piece, piece.deg, xv);
}

// 4-wide region-B F0 seed at a compile-time degree. The m = 1 entry uses the
// full fit degree kBDeg through this template: the constant bound lets MSVC
// unroll the odd/even recurrences and inline the kernel into the RegionB loop
// (the full-accuracy code shape; the bit-identity pin).
template <int kDeg> __m256d ClenshawB4K(__m256d xv) noexcept {
    const double* c = detail::kBcoeffs.data();

    __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(kX0));
    t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (kX1 - kX0)), _mm256_set1_pd(-1.0));

    if constexpr (kDeg == 0)
    {
        return _mm256_set1_pd(c[0]);
    }

    if constexpr (kDeg == 1)
    {
        return _mm256_fmadd_pd(t, _mm256_set1_pd(c[1]), _mm256_set1_pd(c[0]));
    }

    const __m256d v =
        _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
    const __m256d twoV = _mm256_add_pd(v, v);

    const int m = kDeg / 2;
    __m256d b1 = _mm256_set1_pd(c[std::ptrdiff_t{2} * m]);
    __m256d b2 = _mm256_setzero_pd();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256d b0 =
            _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));

    __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
    __m256d o2 = _mm256_setzero_pd();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256d o0 = _mm256_fmadd_pd(
            twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256d odd = _mm256_fmadd_pd(
        _mm256_sub_pd(twoV, _mm256_set1_pd(1.0)), o1, _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
    return _mm256_fmadd_pd(t, odd, even);
}

// 4-wide region-B F0 seed at a runtime degree. Used by the relaxed-m
// (kAccuracyMultiplier > 1) RegionB loop only; the data-dependent loop is not
// inlined, which is the documented relaxed-path cost.
//
// The library instantiates the SIMD region entry points at m = 1 only (the
// relaxed set is compiled for the scalar entries in boys_c.cpp), so in this
// TU that caller sits in the discarded arm of an `if constexpr` and GCC/Clang
// see a defined-but-unused internal function. It is kept because it is the
// relaxed SIMD path's seed — an instantiation added here would use it — and
// it is marked so the discard stays visible as a fact about the instantiation
// set rather than a warning this tree suppresses wholesale.
[[maybe_unused]] __m256d ClenshawB4Deg(int deg, __m256d xv) noexcept {
    const double* c = detail::kBcoeffs.data();

    __m256d t = _mm256_sub_pd(xv, _mm256_set1_pd(kX0));
    t = _mm256_fmadd_pd(t, _mm256_set1_pd(2.0 / (kX1 - kX0)), _mm256_set1_pd(-1.0));

    if (deg == 0)
    {
        return _mm256_set1_pd(c[0]);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_pd(t, _mm256_set1_pd(c[1]), _mm256_set1_pd(c[0]));
    }

    const __m256d v =
        _mm256_fmsub_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(1.0));
    const __m256d twoV = _mm256_add_pd(v, v);

    const int m = deg / 2;
    __m256d b1 = _mm256_set1_pd(c[std::ptrdiff_t{2} * m]);
    __m256d b2 = _mm256_setzero_pd();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256d b0 =
            _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(_mm256_set1_pd(c[0]), b2));

    __m256d o1 = _mm256_set1_pd(c[2 * m - 1]);
    __m256d o2 = _mm256_setzero_pd();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256d o0 = _mm256_fmadd_pd(
            twoV, o1, _mm256_sub_pd(_mm256_set1_pd(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256d odd = _mm256_fmadd_pd(
        _mm256_sub_pd(twoV, _mm256_set1_pd(1.0)), o1, _mm256_sub_pd(_mm256_set1_pd(c[1]), o2));
    return _mm256_fmadd_pd(t, odd, even);
}

// The m = 1 entry: the fit's full degree.
__m256d ClenshawB4(__m256d xv) noexcept {
    return ClenshawB4K<detail::kBDeg>(xv);
}

} // namespace

} // namespace boys::detail

namespace boys {

bool BoysAvx2Available() noexcept {
    static const bool available = detail::DetectAvx2();
    return available;
}

} // namespace boys

namespace boys::detail {

template <double kAccuracyMultiplier>
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    const int first = detail::kPieceStart[n];
    const int last = detail::kPieceStart[n + 1];

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i + 3 < count; i += 4)
        {
            const __m256d xv = _mm256_loadu_pd(x + i);
            __m256d acc = _mm256_setzero_pd();

            for (int p = first; p < last; ++p)
            {
                const detail::OrderPiece& piece = detail::kPieces[p];
                const __m256d mask =
                    _mm256_and_pd(_mm256_cmp_pd(xv, _mm256_set1_pd(piece.a), _CMP_GE_OQ),
                                  _mm256_cmp_pd(xv, _mm256_set1_pd(piece.b), _CMP_LT_OQ));
                acc = _mm256_blendv_pd(acc, Clenshaw4Split(piece, xv), mask);
            }

            _mm256_storeu_pd(out + i, acc);
        }
    } else
    {
        static constexpr auto kDegrees =
            detail::RegionADegrees<kAccuracyMultiplier, detail::BoysRole::kDoubleSingle>();

        for (std::size_t i = 0; i + 3 < count; i += 4)
        {
            const __m256d xv = _mm256_loadu_pd(x + i);
            __m256d acc = _mm256_setzero_pd();

            for (int p = first; p < last; ++p)
            {
                const detail::OrderPiece& piece = detail::kPieces[p];
                const __m256d mask =
                    _mm256_and_pd(_mm256_cmp_pd(xv, _mm256_set1_pd(piece.a), _CMP_GE_OQ),
                                  _mm256_cmp_pd(xv, _mm256_set1_pd(piece.b), _CMP_LT_OQ));
                acc = _mm256_blendv_pd(
                    acc, Clenshaw4SplitDeg(piece, kDegrees[static_cast<std::size_t>(p)], xv), mask);
            }

            _mm256_storeu_pd(out + i, acc);
        }
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = BoysSingle<kAccuracyMultiplier>(n, x[i]);
    }
}

template <double kAccuracyMultiplier>
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    static const ExpTable expTable;

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i + 3 < count; i += 4)
        {
            const __m256d xv = _mm256_loadu_pd(x + i);
            __m256d f = ClenshawB4(xv);
            const __m256d expx = expTable.Eval4(xv);
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
            _mm256_storeu_pd(out + i, f);

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_fmadd_pd(
                    _mm256_set1_pd(l + 0.5), f, _mm256_mul_pd(_mm256_set1_pd(-0.5), expx));
                f = _mm256_mul_pd(f, invx);
                _mm256_storeu_pd(out + (l + 1) * count + i, f);
            }
        }
    } else
    {
        static constexpr auto kDegreesB =
            detail::RegionBDegrees<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch>();

        for (std::size_t i = 0; i + 3 < count; i += 4)
        {
            const __m256d xv = _mm256_loadu_pd(x + i);
            __m256d f = ClenshawB4Deg(kDegreesB[static_cast<std::size_t>(n)], xv);
            const __m256d expx = expTable.Eval4(xv);
            const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
            _mm256_storeu_pd(out + i, f);

            for (int l = 0; l < n; ++l)
            {
                f = _mm256_fmadd_pd(
                    _mm256_set1_pd(l + 0.5), f, _mm256_mul_pd(_mm256_set1_pd(-0.5), expx));
                f = _mm256_mul_pd(f, invx);
                _mm256_storeu_pd(out + (l + 1) * count + i, f);
            }
        }
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        double batch[kMaxBoysOrder + 1];
        BoysAllOrders<kAccuracyMultiplier>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <double kAccuracyMultiplier>
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    for (std::size_t i = 0; i + 3 < count; i += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + i);
        const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
        __m256d f = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_sqrt_pd(invx));

        for (int l = 0; l < n; ++l)
        {
            f = _mm256_mul_pd(_mm256_mul_pd(f, _mm256_set1_pd(l + 0.5)), invx);
        }

        _mm256_storeu_pd(out + i, f);
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = BoysSingle<kAccuracyMultiplier>(n, x[i]);
    }
}

#if BoysFp16
// ---------------------------------------------------------------------------
// fp16 / bf16 lanes, AVX2 scope (8 lanes; the fp32 engine, F16C/bit-trick
// I/O). Same region-partitioned engine pattern as the F64 lanes above; the
// lanes are the certified mixed-precision extension (fp16 I/O around the
// certified fp32 fits of detail::f32). The relaxed branches use the fp16
// computation budget (1e-7) with the F32 piece tables.
// ---------------------------------------------------------------------------
// CPUID detection of the F16C feature bit (the fp16 lane's conversions).
bool DetectF16c() noexcept {
#ifdef _MSC_VER
    int cpuInfo[4] = {};
    __cpuidex(cpuInfo, 1, 0);
    return (cpuInfo[2] & (1u << 29)) != 0; // F16C
#else
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    return (ecx & (1u << 29)) != 0; // F16C
#endif
}

// 8-wide fp32 Clenshaw over one piece, at the given degree; the float mirror
// of Clenshaw4SplitDeg (same even/odd split, FMA chains — the generator only
// emits even degrees, see boys.cpp).
__m256 Clenshaw8SplitF32Deg(const detail::f32::OrderPiece& piece, int deg, __m256 xv) noexcept {
    const float* c = detail::f32::kCoeffs.data() + piece.offset;

    __m256 t = _mm256_sub_ps(xv, _mm256_set1_ps(piece.a));
    t = _mm256_fmadd_ps(t, _mm256_set1_ps(2.0f / (piece.b - piece.a)), _mm256_set1_ps(-1.0f));

    if (deg == 0)
    {
        return _mm256_set1_ps(c[0]);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_ps(t, _mm256_set1_ps(c[1]), _mm256_set1_ps(c[0]));
    }

    const __m256 v =
        _mm256_fmsub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(t, t), _mm256_set1_ps(1.0f));
    const __m256 twoV = _mm256_add_ps(v, v);

    const int m = deg / 2;
    __m256 b1 = _mm256_set1_ps(c[std::ptrdiff_t{2} * m]);
    __m256 b2 = _mm256_setzero_ps();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256 b0 =
            _mm256_fmadd_ps(twoV, b1, _mm256_sub_ps(_mm256_set1_ps(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256 even = _mm256_fmadd_ps(v, b1, _mm256_sub_ps(_mm256_set1_ps(c[0]), b2));

    __m256 o1 = _mm256_set1_ps(c[2 * m - 1]);
    __m256 o2 = _mm256_setzero_ps();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256 o0 = _mm256_fmadd_ps(
            twoV, o1, _mm256_sub_ps(_mm256_set1_ps(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256 odd = _mm256_fmadd_ps(
        _mm256_sub_ps(twoV, _mm256_set1_ps(1.0f)), o1, _mm256_sub_ps(_mm256_set1_ps(c[1]), o2));
    return _mm256_fmadd_ps(t, odd, even);
}

// The m = 1 entry: the piece's full degree.
__m256 Clenshaw8SplitF32(const detail::f32::OrderPiece& piece, __m256 xv) noexcept {
    return Clenshaw8SplitF32Deg(piece, piece.deg, xv);
}

// 8-wide region-B F0 seed (float lane), at the given degree.
__m256 ClenshawB8F32Deg(int deg, __m256 xv) noexcept {
    const float* c = detail::f32::kBcoeffs.data();

    const float x0 = static_cast<float>(kX0);
    const float span = static_cast<float>(kX1 - kX0);
    __m256 t = _mm256_sub_ps(xv, _mm256_set1_ps(x0));
    t = _mm256_fmadd_ps(t, _mm256_set1_ps(2.0f / span), _mm256_set1_ps(-1.0f));

    if (deg == 0)
    {
        return _mm256_set1_ps(c[0]);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_ps(t, _mm256_set1_ps(c[1]), _mm256_set1_ps(c[0]));
    }

    const __m256 v =
        _mm256_fmsub_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(t, t), _mm256_set1_ps(1.0f));
    const __m256 twoV = _mm256_add_ps(v, v);

    const int m = deg / 2;
    __m256 b1 = _mm256_set1_ps(c[std::ptrdiff_t{2} * m]);
    __m256 b2 = _mm256_setzero_ps();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256 b0 =
            _mm256_fmadd_ps(twoV, b1, _mm256_sub_ps(_mm256_set1_ps(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256 even = _mm256_fmadd_ps(v, b1, _mm256_sub_ps(_mm256_set1_ps(c[0]), b2));

    __m256 o1 = _mm256_set1_ps(c[2 * m - 1]);
    __m256 o2 = _mm256_setzero_ps();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256 o0 = _mm256_fmadd_ps(
            twoV, o1, _mm256_sub_ps(_mm256_set1_ps(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256 odd = _mm256_fmadd_ps(
        _mm256_sub_ps(twoV, _mm256_set1_ps(1.0f)), o1, _mm256_sub_ps(_mm256_set1_ps(c[1]), o2));
    return _mm256_fmadd_ps(t, odd, even);
}

// The m = 1 entry: the fit's full degree.
__m256 ClenshawB8F32(__m256 xv) noexcept {
    return ClenshawB8F32Deg(detail::f32::kBDeg, xv);
}

// e^{-x} for 8 floats from the double Taylor table (two 4-wide evaluations):
// the table error 1.83e-17 is far below the certified 1.5e-7 float budget
// after the float conversion.
__m256 Eval8Exp(const ExpTable& table, __m256 xv) noexcept {
    const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
    const __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
    const __m128 loF = _mm256_cvtpd_ps(table.Eval4(lo));
    const __m128 hiF = _mm256_cvtpd_ps(table.Eval4(hi));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(loF), hiF, 1);
}

// Per-half-type I/O: F16C load/store for fp16, the RNE bit trick for bf16
// (no AVX-512_BF16 in the AVX2 scope). The loads/stores round-trip through a
// uint16_t buffer (memcpy) instead of reinterpreting the half pointers: the
// lane works identically whether HalfT is a stdfloat alias or the
// self-contained F16/Bf16 wrapper (f16.hpp), and the compiler folds the
// memcpy into the same vector load/store.
struct F16Lane {
    using HalfT = F16;

    static __m256 Load(const HalfT* p) noexcept {
        std::uint16_t raw[8];
        std::memcpy(raw, p, sizeof(raw));
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(raw)));
    }

    static void Store(HalfT* p, __m256 v) noexcept {
        std::uint16_t raw[8];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw),
                         _mm256_cvtps_ph(v, _MM_FROUND_TO_NEAREST_INT));
        std::memcpy(p, raw, sizeof(raw));
    }

    template <double kAccuracyMultiplier> static HalfT Single(int n, HalfT x) noexcept {
        return BoysSingleF16<kAccuracyMultiplier>(n, x);
    }

    template <double kAccuracyMultiplier> static void Batch(int n, HalfT x, HalfT* out) noexcept {
        BoysAllOrdersF16<kAccuracyMultiplier>(n, x, out);
    }
};

struct Bf16Lane {
    using HalfT = Bf16;

    static __m256 Load(const HalfT* p) noexcept {
        std::uint16_t raw[8];
        std::memcpy(raw, p, sizeof(raw));
        const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(lo), 16));
    }

    static void Store(HalfT* p, __m256 v) noexcept {
        // fp32 -> bf16 with round-to-nearest-even:
        //   bits += 0x7FFF + ((bits >> 16) & 1);  bits >>= 16
        std::uint16_t raw[8];
        __m256i bits = _mm256_castps_si256(v);
        const __m256i lsb = _mm256_and_si256(_mm256_srli_epi32(bits, 16), _mm256_set1_epi32(1));
        bits = _mm256_add_epi32(_mm256_add_epi32(bits, _mm256_set1_epi32(0x7fff)), lsb);
        // The 256-bit packus takes the low 128 of each operand, so a single
        // 8-wide operand would duplicate elements 0-3 and drop elements 4-7.
        // Split the shifted words into the two 128-bit halves and use the
        // 128-bit pack, which assembles {lo0-3, hi4-7}.
        const __m256i hi = _mm256_srli_epi32(bits, 16);
        const __m128i packed =
            _mm_packus_epi32(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw), packed);
        std::memcpy(p, raw, sizeof(raw));
    }

    template <double kAccuracyMultiplier> static HalfT Single(int n, HalfT x) noexcept {
        return BoysSingleBf16<kAccuracyMultiplier>(n, x);
    }

    template <double kAccuracyMultiplier> static void Batch(int n, HalfT x, HalfT* out) noexcept {
        BoysAllOrdersBf16<kAccuracyMultiplier>(n, x, out);
    }
};

// Region-partitioned 8-wide kernels shared by the fp16 and bf16 lanes.
// The scalar tails call the half-lane scalar entries, and the vector bodies
// hold the same documented bound as those entries (they are not bit-identical
// to them: the bodies run the fp32 fit at full degree where the scalar lane
// truncates it, so the two differ by quanta of the half format, never by a
// value the bound cannot absorb).
template <typename Lane, double kAccuracyMultiplier>
void RegionASimdHalf(int n,
                     const typename Lane::HalfT* x,
                     typename Lane::HalfT* out,
                     std::size_t count) noexcept {
    const int first = detail::f32::kPieceStart[n];
    const int last = detail::f32::kPieceStart[n + 1];

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i + 7 < count; i += 8)
        {
            const __m256 xv = Lane::Load(x + i);
            __m256 acc = _mm256_setzero_ps();

            for (int p = first; p < last; ++p)
            {
                const detail::f32::OrderPiece& piece = detail::f32::kPieces[p];
                const __m256 mask =
                    _mm256_and_ps(_mm256_cmp_ps(xv, _mm256_set1_ps(piece.a), _CMP_GE_OQ),
                                  _mm256_cmp_ps(xv, _mm256_set1_ps(piece.b), _CMP_LT_OQ));
                acc = _mm256_blendv_ps(acc, Clenshaw8SplitF32(piece, xv), mask);
            }

            Lane::Store(out + i, acc);
        }
    } else
    {
        static constexpr auto kDegrees =
            detail::RegionADegrees<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Single>();

        for (std::size_t i = 0; i + 7 < count; i += 8)
        {
            const __m256 xv = Lane::Load(x + i);
            __m256 acc = _mm256_setzero_ps();

            for (int p = first; p < last; ++p)
            {
                const detail::f32::OrderPiece& piece = detail::f32::kPieces[p];
                const __m256 mask =
                    _mm256_and_ps(_mm256_cmp_ps(xv, _mm256_set1_ps(piece.a), _CMP_GE_OQ),
                                  _mm256_cmp_ps(xv, _mm256_set1_ps(piece.b), _CMP_LT_OQ));
                acc = _mm256_blendv_ps(
                    acc,
                    Clenshaw8SplitF32Deg(piece, kDegrees[static_cast<std::size_t>(p)], xv),
                    mask);
            }

            Lane::Store(out + i, acc);
        }
    }

    for (std::size_t i = count - (count % 8); i < count; ++i)
    {
        out[i] = Lane::template Single<kAccuracyMultiplier>(n, x[i]);
    }
}

template <typename Lane, double kAccuracyMultiplier>
void RegionBSimdHalf(int n,
                     const typename Lane::HalfT* x,
                     typename Lane::HalfT* out,
                     std::size_t count) noexcept {
    static const ExpTable expTable;

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        for (std::size_t i = 0; i + 7 < count; i += 8)
        {
            const __m256 xv = Lane::Load(x + i);
            __m256 f = ClenshawB8F32(xv);
            const __m256 expx = Eval8Exp(expTable, xv);
            Lane::Store(out + i, f);

            for (int l = 0; l < n; ++l)
            {
                // Divide by x rather than multiply by the rounded 1/x: the
                // upward recurrence amplifies a relative perturbation by
                // ((l + 1/2)/x) per step, and past l = x that factor exceeds
                // one, so the reciprocal's 6e-8 rounding compounds to tens of
                // per cent at the highest orders (measured at n = 32,
                // x = 11.9453: 2.5e-8 by multiplication, 8.4e-8 by division,
                // against 1.539e-7 exact). The certified scalar lane divides.
                f = _mm256_div_ps(
                    _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(l) + 0.5f),
                                    f,
                                    _mm256_mul_ps(_mm256_set1_ps(-0.5f), expx)),
                    xv);
                Lane::Store(out + (l + 1) * count + i, f);
            }
        }
    } else
    {
        static constexpr auto kDegreesB =
            detail::RegionBDegrees<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Single>();

        for (std::size_t i = 0; i + 7 < count; i += 8)
        {
            const __m256 xv = Lane::Load(x + i);
            __m256 f = ClenshawB8F32Deg(kDegreesB[static_cast<std::size_t>(n)], xv);
            const __m256 expx = Eval8Exp(expTable, xv);
            Lane::Store(out + i, f);

            for (int l = 0; l < n; ++l)
            {
                // Division, not a rounded reciprocal: see the m = 1 arm.
                f = _mm256_div_ps(
                    _mm256_fmadd_ps(_mm256_set1_ps(static_cast<float>(l) + 0.5f),
                                    f,
                                    _mm256_mul_ps(_mm256_set1_ps(-0.5f), expx)),
                    xv);
                Lane::Store(out + (l + 1) * count + i, f);
            }
        }
    }

    for (std::size_t i = count - (count % 8); i < count; ++i)
    {
        typename Lane::HalfT batch[kMaxBoysOrder + 1];
        Lane::template Batch<kAccuracyMultiplier>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <typename Lane, double kAccuracyMultiplier>
void RegionCSimdHalf(int n,
                     const typename Lane::HalfT* x,
                     typename Lane::HalfT* out,
                     std::size_t count) noexcept {
    for (std::size_t i = 0; i + 7 < count; i += 8)
    {
        const __m256 xv = Lane::Load(x + i);
        const __m256 invx = _mm256_div_ps(_mm256_set1_ps(1.0f), xv);
        __m256 f =
            _mm256_mul_ps(_mm256_set1_ps(static_cast<float>(kHalfSqrtPi)), _mm256_sqrt_ps(invx));

        for (int l = 0; l < n; ++l)
        {
            f = _mm256_mul_ps(_mm256_mul_ps(f, _mm256_set1_ps(static_cast<float>(l) + 0.5f)), invx);
        }

        Lane::Store(out + i, f);
    }

    for (std::size_t i = count - (count % 8); i < count; ++i)
    {
        out[i] = Lane::template Single<kAccuracyMultiplier>(n, x[i]);
    }
}

// F16C is implied by AVX2 on every shipping x86 CPU; the CPUID check below
// is defensive and routes to the certified scalar lane if it ever fires.
bool F16cAvailable() noexcept {
    static const bool available = DetectF16c();
    return available;
}

template <double kAccuracyMultiplier>
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleF16<kAccuracyMultiplier>(n, x[i]);
        }

        return;
    }

    RegionASimdHalf<F16Lane, kAccuracyMultiplier>(n, x, out, count);
}

template <double kAccuracyMultiplier>
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        F16 batch[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < count; ++i)
        {
            BoysAllOrdersF16<kAccuracyMultiplier>(n, x[i], batch);

            for (int l = 0; l <= n; ++l)
            {
                out[l * count + i] = batch[l];
            }
        }

        return;
    }

    RegionBSimdHalf<F16Lane, kAccuracyMultiplier>(n, x, out, count);
}

template <double kAccuracyMultiplier>
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleF16<kAccuracyMultiplier>(n, x[i]);
        }

        return;
    }

    RegionCSimdHalf<F16Lane, kAccuracyMultiplier>(n, x, out, count);
}

template <double kAccuracyMultiplier>
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleBf16<kAccuracyMultiplier>(n, x[i]);
        }

        return;
    }

    RegionASimdHalf<Bf16Lane, kAccuracyMultiplier>(n, x, out, count);
}

template <double kAccuracyMultiplier>
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        Bf16 batch[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < count; ++i)
        {
            BoysAllOrdersBf16<kAccuracyMultiplier>(n, x[i], batch);

            for (int l = 0; l <= n; ++l)
            {
                out[l * count + i] = batch[l];
            }
        }

        return;
    }

    RegionBSimdHalf<Bf16Lane, kAccuracyMultiplier>(n, x, out, count);
}

template <double kAccuracyMultiplier>
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleBf16<kAccuracyMultiplier>(n, x[i]);
        }

        return;
    }

    RegionCSimdHalf<Bf16Lane, kAccuracyMultiplier>(n, x, out, count);
}
#endif // BoysFp16

} // namespace boys::detail

#else // BOYS_SIMD_X86

// ---------------------------------------------------------------------------
// Non-x86 targets: the same entry points, defined against the certified scalar
// lanes. The vector engine does not exist here, so BoysAvx2Available() is
// false and no entry asserts it; the region contract (inputs partitioned by
// region) is still a *sufficient* precondition, it is simply no longer
// required — the scalar lanes accept any argument >= 0.
//
// The bodies mirror, element for element, the scalar tails the x86 entries run
// for their last count % 4 elements (see the #if branch above), so a caller
// gets the same numbers it would get from the x86 entry on a machine without
// AVX2. Region B keeps its transposed layout out[l * count + i].
// ---------------------------------------------------------------------------
namespace boys::detail {

} // namespace boys::detail

namespace boys {

bool BoysAvx2Available() noexcept {
    return false;
}

} // namespace boys

namespace boys::detail {

template <double kAccuracyMultiplier>
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingle<kAccuracyMultiplier>(n, x[i]);
    }
}

template <double kAccuracyMultiplier>
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    double batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrders<kAccuracyMultiplier>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <double kAccuracyMultiplier>
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingle<kAccuracyMultiplier>(n, x[i]);
    }
}

#if BoysFp16
template <double kAccuracyMultiplier>
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleF16<kAccuracyMultiplier>(n, x[i]);
    }
}

template <double kAccuracyMultiplier>
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    F16 batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrdersF16<kAccuracyMultiplier>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <double kAccuracyMultiplier>
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleF16<kAccuracyMultiplier>(n, x[i]);
    }
}

template <double kAccuracyMultiplier>
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleBf16<kAccuracyMultiplier>(n, x[i]);
    }
}

template <double kAccuracyMultiplier>
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    Bf16 batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrdersBf16<kAccuracyMultiplier>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <double kAccuracyMultiplier>
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleBf16<kAccuracyMultiplier>(n, x[i]);
    }
}
#endif // BoysFp16

} // namespace boys::detail

#endif // BOYS_SIMD_X86

// The default (m = 1) instantiations behind the extern-template declarations
// in boys.hpp (the m = 1 branches are today's bodies verbatim).
namespace boys::detail {
template void BoysRegionASimd<kBoysFullAccuracyMultiplier>(int n,
                                                           const double* x,
                                                           double* out,
                                                           std::size_t count) noexcept;
template void BoysRegionBSimd<kBoysFullAccuracyMultiplier>(int n,
                                                           const double* x,
                                                           double* out,
                                                           std::size_t count) noexcept;
template void BoysRegionCSimd<kBoysFullAccuracyMultiplier>(int n,
                                                           const double* x,
                                                           double* out,
                                                           std::size_t count) noexcept;
#if BoysFp16
template void BoysRegionASimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                              const F16* x,
                                                              F16* out,
                                                              std::size_t count) noexcept;
template void BoysRegionBSimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                              const F16* x,
                                                              F16* out,
                                                              std::size_t count) noexcept;
template void BoysRegionCSimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                              const F16* x,
                                                              F16* out,
                                                              std::size_t count) noexcept;
template void BoysRegionASimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                               const Bf16* x,
                                                               Bf16* out,
                                                               std::size_t count) noexcept;
template void BoysRegionBSimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                               const Bf16* x,
                                                               Bf16* out,
                                                               std::size_t count) noexcept;
template void BoysRegionCSimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                               const Bf16* x,
                                                               Bf16* out,
                                                               std::size_t count) noexcept;
#endif // BoysFp16

} // namespace boys::detail
