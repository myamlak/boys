#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/kernels.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// The AVX2 tier is x86_64-only by construction: AVX2/FMA intrinsics, compiled /arch:AVX2 (MSVC)
// or -mavx2 -mfma -mf16c. BOYS_SIMD_X86 answers whether this TU compiles that tier - a CMake
// configure-time probe states it, and the guard detects it from the two predefines otherwise.
// Unstated and not x86_64 is an #error rather than a quiet 0. A non-x86 target gets scalar lanes.
#ifdef BOYS_SIMD_X86

// The build answered; nothing to detect.

#elif defined(__x86_64__) || defined(_M_X64)

// Unstated, but the compiler says x86_64 — the same two macros the
// CMakeLists probe tests for, so the detected answer and the probed one agree.

#define BOYS_SIMD_X86 1

#else

#error "BOYS_SIMD_X86 is unset and this is not x86_64: define it (1 on x86_64, 0 elsewhere)"

#endif

// The packed arithmetic backends this TU's kernels are written against.
#include "boys_backend_registry.hpp"
#include "boys_backend_simd.hpp"

#if BOYS_SIMD_X86

#include <immintrin.h>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// AVX2 region-sorted lanes, region-first: partition the arguments by region so every 4-lane
// vector is homogeneous - the unsorted variant pays a measured 2.3x divergence penalty. Callers
// must check BoysAvx2Available() first; the FMA chains require the FMA feature bit too.

namespace boys::detail {

// CPUID + OSXSAVE detection of the AVX2 + FMA scope (the F64/F32 lanes' engine).
// FMA belongs to the scope, not an optional extra: a processor with AVX2 but no
// FMA would be dispatched into an unimplemented instruction. Defined here, not in
// the header, so the one copy serves the inline test in every caller's TU.
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
    // GCC/Clang builds: the kernel enables the AVX XCR0 state whenever the CPU
    // supports it, so the leaf-1 feature bits are authoritative.
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

namespace {

using detail::kX0;
using detail::kX1;

constexpr double kHalfSqrtPi = 0.886226925452758014;

// e^{-x} on [0, 30]: a degree-4 Taylor table, one row per abscissa x_i = i * kStep, padded to 8
// doubles for the gathers' legal scale 8; worst absolute error 1.83e-17. A row holds the quartic
// Taylor polynomial at x_i in the monomial basis of the ABSOLUTE argument - a_k = e^{-x_i} *
// (-1)^k * S_{4-k} / k! - and stores (-1)^k a_k, the sign taken back out by Eval4's FMA chain.
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

            // partialSum[m] = sum_{j=0..m} x^j / j!.
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
        // Rows are 8 doubles apart: a grid index scaled by 8 is a row address.
        index = _mm_slli_epi32(index, 3);
        __m256d c0 = _mm256_i32gather_pd(&_coefficients[0][0], index, 8);
        __m256d c1 = _mm256_i32gather_pd(&_coefficients[0][1], index, 8);
        __m256d c2 = _mm256_i32gather_pd(&_coefficients[0][2], index, 8);
        __m256d c3 = _mm256_i32gather_pd(&_coefficients[0][3], index, 8);
        __m256d c4 = _mm256_i32gather_pd(&_coefficients[0][4], index, 8);
        // The stored coefficients carry the alternating sign: evaluate
        // c4*x^4 - c3*x^3 + c2*x^2 - c1*x + c0.
        __m256d result = _mm256_fmsub_pd(c4, x, c3);
        result = _mm256_fmadd_pd(result, x, c2);
        result = _mm256_fmsub_pd(result, x, c1);
        result = _mm256_fmadd_pd(result, x, c0);
        return result;
    }

private:
    double _coefficients[kNumPoints + 1][8]{};
};

// Split Clenshaw (even/odd), packed, half-depth FMA chains: T_{2j+1}(t) = t * D_j(v) with the D
// recurrence. One body for every packed width and precision - which multiply-add a step uses is
// the backend's. What stays here is the mapped argument, in the packed lanes' own form.
template <backend::ArithmeticBackend B, typename Piece>
typename B::Packed RegionAClenshaw(const typename B::Value* c,
                                   const Piece& piece,
                                   int deg,
                                   typename B::Packed xv) noexcept {
    using V = typename B::Value;
    const typename B::Packed t = B::MulAdd(B::Sub(xv, B::Broadcast(piece.a)),
                                           B::Broadcast(V{2} / (piece.b - piece.a)),
                                           B::Broadcast(-V{1}));
    return ClenshawSplit<B>(c + piece.offset, deg, t);
}

// The region-B seed, at a compile-time degree where kDeg >= 0 and at the
// caller's where it is not. The constant bound is what lets MSVC unroll the
// odd/even recurrences and inline the kernel into the RegionB loop.
template <backend::ArithmeticBackend B, int kDeg>
typename B::Packed RegionBClenshaw(const typename B::Value* c,
                                   int deg,
                                   typename B::Packed xv) noexcept {
    using V = typename B::Value;
    const V lo = static_cast<V>(kX0);
    const V span = static_cast<V>(kX1 - kX0);
    const typename B::Packed t = B::MulAdd(B::Sub(xv, B::Broadcast(lo)),
                                           B::Broadcast(V{2} / span), B::Broadcast(-V{1}));
    const int d = (kDeg >= 0) ? kDeg : deg;
    return ClenshawSplit<B>(c, d, t);
}

// 4-wide split Clenshaw for one piece; even deg >= 4 only.
inline __m256d Clenshaw4SplitDeg(const detail::OrderPiece& piece, int deg, __m256d xv) noexcept {
    return RegionAClenshaw<backend::Avx2Fp64<>>(detail::kCoeffs.data(), piece, deg, xv);
}

// The m = 1 entry: the piece's full degree.
inline __m256d Clenshaw4Split(const detail::OrderPiece& piece, __m256d xv) noexcept {
    return Clenshaw4SplitDeg(piece, piece.deg, xv);
}

// The m = 1 entry: the fit's full degree, known at compile time.
inline __m256d ClenshawB4(__m256d xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp64<>, detail::kBDeg>(detail::kBcoeffs.data(), 0, xv);
}

// 4-wide region-B F0 seed at a runtime degree, for the relaxed RegionB loop only. The library
// instantiates the SIMD region entries at m = 1 only, so here that caller sits in the discarded
// arm of an `if constexpr` and GCC/Clang see a defined-but-unused internal function.
[[maybe_unused]] inline __m256d ClenshawB4Deg(int deg, __m256d xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp64<>, -1>(detail::kBcoeffs.data(), deg, xv);
}

} // namespace

} // namespace boys::detail

namespace boys::detail {

void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    const int first = detail::kPieceStart[n];
    const int last = detail::kPieceStart[n + 1];

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
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}

// No entry point calls this lane; its callers are the tests and the benchmark. It takes no
// policy, only the multiplier, so a caller cannot name it with a division form, and its vector
// body forms 1/x once - the plain reciprocal, not the kRefinedReciprocal the entries default to.
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    static const ExpTable expTable;

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
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        double batch[kMaxBoysOrder + 1];
        BoysAllOrders<>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

// No entry point calls this lane; its callers are the tests and the benchmark. It takes no policy,
// forms 1/x once as the plain reciprocal, and runs the shipped far tail's three pieces - the
// corrected closed-form seed, then the step carrying the exp(-x)/2 term the bare asymptotic step
// drops. That exp is the library routine: region B's table stops at 30, the term reaches near 60.
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    for (std::size_t i = 0; i + 3 < count; i += 4)
    {
        const __m256d xv = _mm256_loadu_pd(x + i);
        const __m256d invx = _mm256_div_pd(_mm256_set1_pd(1.0), xv);
        __m256d f = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_sqrt_pd(invx));

        // H_0 = 1 + correction, where the correction's fit is defined.
        const __m256d t = _mm256_sub_pd(
            _mm256_div_pd(
                _mm256_mul_pd(_mm256_set1_pd(2.0),
                              _mm256_sub_pd(xv, _mm256_set1_pd(kSeedCorrectionX0))),
                _mm256_set1_pd(kSeedCorrectionX1 - kSeedCorrectionX0)),
            _mm256_set1_pd(1.0));
        const __m256d corrected = _mm256_mul_pd(
            f,
            _mm256_add_pd(_mm256_set1_pd(1.0),
                          ClenshawSplit<backend::Avx2Fp64<>>(kSeedCorrectionCoeffs.data(),
                                                            kSeedCorrectionDeg,
                                                            t)));
        f = _mm256_blendv_pd(
            f, corrected, _mm256_cmp_pd(xv, _mm256_set1_pd(kSeedCorrectionX1), _CMP_LT_OQ));

        double lanes[4];
        _mm256_storeu_pd(lanes, xv);

        for (double& argument : lanes)
        {
            argument = RegionBHalfExp<RegionBExp::kAccurate>(argument);
        }

        const __m256d expv = _mm256_loadu_pd(lanes);

        for (int l = 0; l < n; ++l)
        {
            f = _mm256_mul_pd(
                _mm256_sub_pd(_mm256_mul_pd(f, _mm256_set1_pd(l + 0.5)), expv), invx);
        }

        _mm256_storeu_pd(out + i, f);
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}

// The far region's ladder in the arguments lane: four arguments of ONE ladder at once, the whole
// body - the seed, its correction and every step - written through the scalar body's own
// operations in the scalar body's own order. Nothing here is a second arithmetic, so a value the
// lane returns is the per-argument body's value at every bit.
namespace {

// Whether this build's scalar arithmetic contracts a bare product-minus-add. The step below is
// written that way in the scalar body (UpwardStep's `(l - 1/2) * f - expx`), and
// BOYS_SCALAR_CONTRACTS is the answer this build measured at its own flags - the same answer the
// CMake pins onto this TU's -ffp-contract, because the lane and the body it reproduces have to
// round alike. An intrinsic pair never contracts, so the lane has to spell the answer rather than
// inherit it. A cross-compiling configure states no answer, and the separate route is then the
// lane's: it is what the build's own flags give the scalar body there.
#if defined(BOYS_SCALAR_CONTRACTS) && BOYS_SCALAR_CONTRACTS
constexpr bool kFarStepContracts = true;
#else
constexpr bool kFarStepContracts = false;
#endif

// The multiply-add route the seed's correction runs. The scalar body reads it through
// ScalarFp64::MulAdd, which spells the fused route as the runtime's `fma` and the separate route
// as a bare product-plus-add - so a build that selected the separate route and contracts runs the
// FUSED arithmetic there, and the lane has to name the route in force rather than the selection.
constexpr backend::MulAddRoute kFarCorrectionRoute =
    (backend::detail::kSelectedRoute == backend::MulAddRoute::kFused || kFarStepContracts)
        ? backend::MulAddRoute::kFused
        : backend::MulAddRoute::kSeparate;

// One step of the ladder: F_l = ((l - 1/2) F_(l-1) - exp(-x)/2) / x. The product-minus-
// exponential is the scalar body's own expression, at the route that body's build rounds it in.
inline __m256d FarLadderStep(__m256d lHalf, __m256d f, __m256d expx) noexcept {
    if constexpr (kFarStepContracts)
    {
        return _mm256_fmsub_pd(lHalf, f, expx);
    }
    else
    {
        return _mm256_sub_pd(_mm256_mul_pd(lHalf, f), expx);
    }
}

// The seed: the closed form sqrt(pi)/(2 sqrt(x)) and, where the argument is inside the
// correction's span, times 1 + the correction. FarSeedAt, lane for lane: the lanes at or above
// the span's end keep the closed form's own bits, which is what the scalar branch leaves them,
// and the comparison is the ordered one the scalar `<` is.
inline __m256d FarLadderSeed(__m256d xv) noexcept {
    const __m256d f = _mm256_div_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_sqrt_pd(xv));
    const __m256d t = _mm256_sub_pd(
        _mm256_div_pd(
            _mm256_mul_pd(_mm256_set1_pd(2.0),
                          _mm256_sub_pd(xv, _mm256_set1_pd(kSeedCorrectionX0))),
            _mm256_set1_pd(kSeedCorrectionX1 - kSeedCorrectionX0)),
        _mm256_set1_pd(1.0));
    const __m256d correction = ClenshawSplit<backend::Avx2Fp64<kFarCorrectionRoute>>(
        kSeedCorrectionCoeffs.data(), kSeedCorrectionDeg, t);
    const __m256d corrected = _mm256_mul_pd(f, _mm256_add_pd(_mm256_set1_pd(1.0), correction));
    const __m256d inside = _mm256_cmp_pd(xv, _mm256_set1_pd(kSeedCorrectionX1), _CMP_LT_OQ);

    return _mm256_blendv_pd(f, corrected, inside);
}

// DivideStep at each form, four-wide. A quotient and a square root are correctly rounded in
// either width, so the exact form is the scalar spelling read four-wide and the plain form is the
// product through the reciprocal. The refined form's infinite divisor is the scalar body's own
// branch: the product, which for an infinite divisor is the signed zero the plain quotient gives,
// and a blend rather than the two fmas the recovery needs - those form a NaN there.
__m256d DivideRefinedLanes(__m256d a, __m256d x, __m256d invx) noexcept {
    const __m256d quotient = _mm256_mul_pd(a, invx);
    const __m256d negated = _mm256_xor_pd(quotient, _mm256_set1_pd(-0.0));
    const __m256d refined = _mm256_fmadd_pd(_mm256_fmadd_pd(negated, x, a), invx, quotient);
    const __m256d infinite = _mm256_cmp_pd(_mm256_andnot_pd(_mm256_set1_pd(-0.0), x),
                                           _mm256_set1_pd(HUGE_VAL),
                                           _CMP_EQ_OQ);

    return _mm256_blendv_pd(refined, quotient, infinite);
}

template <DivisionForm kForm>
__m256d DivideStepLanes(__m256d a, __m256d x, __m256d invx) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(invx);
        return _mm256_div_pd(a, x);
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        static_cast<void>(x);
        return _mm256_mul_pd(a, invx);
    } else
    {
        return DivideRefinedLanes(a, x, invx);
    }
}

// StepReciprocal, four-wide: the exact form forms none, as the scalar spelling does not.
template <DivisionForm kForm>
__m256d StepReciprocalLanes(__m256d x) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(x);
        return _mm256_setzero_pd();
    } else
    {
        return _mm256_div_pd(_mm256_set1_pd(1.0), x);
    }
}

// One order's four values into the four columns the writer named. Four consecutive doubles are
// one register store - the sorted entry's groups, which is the shape the tile exists for - and a
// permutation's four addresses share no store, so each lane's value lands on its own.
template <bool kContiguous>
void StoreLanes(int order, std::size_t stride, double* const* cols, __m256d f) noexcept {
    const std::size_t at = static_cast<std::size_t>(order) * stride;

    if constexpr (kContiguous)
    {
        _mm256_storeu_pd(cols[0] + at, f);
    } else
    {
        alignas(32) double lane[4];
        _mm256_store_pd(lane, f);

        for (std::size_t j = 0; j < 4; ++j)
        {
            cols[j][at] = lane[j];
        }
    }
}

template <DivisionForm kForm, bool kContiguous>
void FarLadderLanesAt(int nmax,
                      const double* x,
                      const double* halfExp,
                      std::size_t stride,
                      double* const* cols) noexcept {
    const __m256d xv = _mm256_loadu_pd(x);
    const __m256d expx = _mm256_loadu_pd(halfExp);
    const __m256d invx = StepReciprocalLanes<kForm>(xv);
    __m256d f = FarLadderSeed(xv);

    StoreLanes<kContiguous>(0, stride, cols, f);

    for (int l = 1; l <= nmax; ++l)
    {
        const __m256d step =
            FarLadderStep(_mm256_set1_pd(static_cast<double>(l) - 0.5), f, expx);
        f = DivideStepLanes<kForm>(step, xv, invx);
        StoreLanes<kContiguous>(l, stride, cols, f);
    }
}

/// The run's writers, the entry's two shapes: see the declaration for which one a run takes and
/// why. Both are the per-argument body's own arithmetic - the lane's groups and that body's rows.
template <DivisionForm kForm>
void FarRunDirect(int nmax,
                  const double* x,
                  const std::size_t* index,
                  double* out,
                  std::size_t count,
                  std::size_t begin,
                  std::size_t end) noexcept {
    double xs[4];
    double halfExp[4];
    double* cols[4];
    std::size_t j = begin;

    for (; j + 4 <= end; j += 4)
    {
        for (std::size_t t = 0; t < 4; ++t)
        {
            const std::size_t i = (index != nullptr) ? index[j + t] : j + t;
            xs[t] = x[i];
            halfExp[t] = RegionBHalfExp<RegionBExp::kAccurate>(xs[t]);
            cols[t] = out + i;
        }

        const bool contiguous = cols[1] == cols[0] + 1 && cols[2] == cols[0] + 2 &&
                                cols[3] == cols[0] + 3;

        if (contiguous)
        {
            FarLadderLanesAt<kForm, true>(nmax, xs, halfExp, count, cols);
        }
        else
        {
            FarLadderLanesAt<kForm, false>(nmax, xs, halfExp, count, cols);
        }
    }

    for (; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;
        BoysAllNBodyRegionC<kForm>(nmax, x[i], count, out + i);
    }
}

template <DivisionForm kForm>
void FarRunTiled(int nmax,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t begin,
                 std::size_t end) noexcept {
    BoysAllNTile(nmax, count, nullptr, out, begin, end,
                 [&](std::size_t base, std::size_t block, double* stage) {
                     double xs[4];
                     double halfExp[4];
                     double* cols[4];
                     std::size_t t = 0;

                     for (; t + 4 <= block; t += 4)
                     {
                         for (std::size_t q = 0; q < 4; ++q)
                         {
                             xs[q] = x[base + t + q];
                             halfExp[q] = RegionBHalfExp<RegionBExp::kAccurate>(xs[q]);
                             cols[q] = stage + t + q;
                         }

                         FarLadderLanesAt<kForm, true>(nmax, xs, halfExp, block, cols);
                     }

                     for (; t < block; ++t)
                     {
                         BoysAllNBodyRegionC<kForm>(nmax, x[base + t], block, stage + t);
                     }
                 });
}

template <DivisionForm kForm>
void FarRunAt(int nmax,
              const double* x,
              const std::size_t* index,
              double* out,
              std::size_t count,
              std::size_t begin,
              std::size_t end) noexcept {
    if (index == nullptr)
    {
        FarRunTiled<kForm>(nmax, x, out, count, begin, end);
        return;
    }

    FarRunDirect<kForm>(nmax, x, index, out, count, begin, end);
}

} // namespace

void BoysFarLadderLanes(int nmax,
                        const double* x,
                        const double* halfExp,
                        std::size_t stride,
                        double* const* cols,
                        DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(BoysAvx2Available());
    assert(x[0] >= kX1 && x[1] >= kX1 && x[2] >= kX1 && x[3] >= kX1);

    // Which of the two stores a group takes is a property of the columns and not of the shape:
    // the sorted entry's groups are four consecutive doubles of a plane - or of the tile, which
    // is the same four addresses - and the unsorted entry's are wherever its permutation put
    // them. Decided once per group, and the values either way are the same.
    const bool contiguous = cols[1] == cols[0] + 1 && cols[2] == cols[0] + 2 &&
                            cols[3] == cols[0] + 3;

    if (contiguous)
    {
        switch (form)
        {
        case DivisionForm::kExactDivision:
            FarLadderLanesAt<DivisionForm::kExactDivision, true>(nmax, x, halfExp, stride, cols);
            break;

        case DivisionForm::kPlainReciprocal:
            FarLadderLanesAt<DivisionForm::kPlainReciprocal, true>(nmax, x, halfExp, stride, cols);
            break;

        case DivisionForm::kRefinedReciprocal:
            FarLadderLanesAt<DivisionForm::kRefinedReciprocal, true>(
                nmax, x, halfExp, stride, cols);
            break;
        }

        return;
    }

    switch (form)
    {
    case DivisionForm::kExactDivision:
        FarLadderLanesAt<DivisionForm::kExactDivision, false>(nmax, x, halfExp, stride, cols);
        break;

    case DivisionForm::kPlainReciprocal:
        FarLadderLanesAt<DivisionForm::kPlainReciprocal, false>(nmax, x, halfExp, stride, cols);
        break;

    case DivisionForm::kRefinedReciprocal:
        FarLadderLanesAt<DivisionForm::kRefinedReciprocal, false>(nmax, x, halfExp, stride, cols);
        break;
    }
}

void BoysFarLadderRun(int nmax,
                      const double* x,
                      const std::size_t* index,
                      double* out,
                      std::size_t count,
                      std::size_t begin,
                      std::size_t end,
                      DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(BoysAvx2Available());
    assert(begin < end && end - begin >= 4);
    assert(index == nullptr || (index[begin] < count && index[end - 1] < count));

    switch (form)
    {
    case DivisionForm::kExactDivision:
        FarRunAt<DivisionForm::kExactDivision>(nmax, x, index, out, count, begin, end);
        break;

    case DivisionForm::kPlainReciprocal:
        FarRunAt<DivisionForm::kPlainReciprocal>(nmax, x, index, out, count, begin, end);
        break;

    case DivisionForm::kRefinedReciprocal:
        FarRunAt<DivisionForm::kRefinedReciprocal>(nmax, x, index, out, count, begin, end);
        break;
    }
}

#if BoysFp16
// fp16 / bf16 lanes, AVX2 scope (8 lanes; the fp32 engine, F16C/bit-trick I/O): the same
// region-partitioned engine pattern as the F64 lanes, around the certified fp32 fits of
// detail::f32. The relaxed branches use the fp16 budget (1e-7) with the F32 piece tables.
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

// 8-wide fp32 Clenshaw over one piece, at the given degree: the float
// instantiation of the same body the double lane runs. The generator only emits
// even degrees.
inline __m256 Clenshaw8SplitF32Deg(const detail::f32::OrderPiece& piece,
                                   int deg,
                                   __m256 xv) noexcept {
    return RegionAClenshaw<backend::Avx2Fp32<>>(detail::f32::kCoeffs.data(), piece, deg, xv);
}

// The m = 1 entry: the piece's full degree.
inline __m256 Clenshaw8SplitF32(const detail::f32::OrderPiece& piece, __m256 xv) noexcept {
    return Clenshaw8SplitF32Deg(piece, piece.deg, xv);
}

// The m = 1 entry: the fit's full degree, known at compile time.
inline __m256 ClenshawB8F32(__m256 xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp32<>, detail::f32::kBDeg>(
        detail::f32::kBcoeffs.data(), 0, xv);
}

// The relaxed entry: the caller's degree; see ClenshawB4Deg.
[[maybe_unused]] inline __m256 ClenshawB8F32Deg(int deg, __m256 xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp32<>, -1>(
        detail::f32::kBcoeffs.data(), deg, xv);
}

// e^{-x} for 8 floats from the double Taylor table (two 4-wide evaluations).
__m256 Eval8Exp(const ExpTable& table, __m256 xv) noexcept {
    const __m256d lo = _mm256_cvtps_pd(_mm256_castps256_ps128(xv));
    const __m256d hi = _mm256_cvtps_pd(_mm256_extractf128_ps(xv, 1));
    const __m128 loF = _mm256_cvtpd_ps(table.Eval4(lo));
    const __m128 hiF = _mm256_cvtpd_ps(table.Eval4(hi));
    return _mm256_insertf128_ps(_mm256_castps128_ps256(loF), hiF, 1);
}

// Per-half-type I/O: F16C load/store for fp16, the RNE bit trick for bf16
// (no AVX-512_BF16 in the AVX2 scope). The memcpy round-trip through a
// uint16_t buffer works whether HalfT is a stdfloat alias or the self-contained
// F16/Bf16 wrapper, and the compiler folds it into the same vector load/store.
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

    static HalfT Single(int n, HalfT x) noexcept {
        return BoysSingleF16<>(n, x);
    }

    static void Batch(int n, HalfT x, HalfT* out) noexcept {
        BoysAllOrdersF16<>(n, x, out);
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
        // The 256-bit packus takes the low 128 of each operand, so an 8-wide
        // operand would duplicate elements 0-3 and drop 4-7; the 128-bit pack
        // of the two halves assembles {lo0-3, hi4-7}.
        const __m256i hi = _mm256_srli_epi32(bits, 16);
        const __m128i packed =
            _mm_packus_epi32(_mm256_castsi256_si128(hi), _mm256_extracti128_si256(hi, 1));
        _mm_storeu_si128(reinterpret_cast<__m128i*>(raw), packed);
        std::memcpy(p, raw, sizeof(raw));
    }

    static HalfT Single(int n, HalfT x) noexcept {
        return BoysSingleBf16<>(n, x);
    }

    static void Batch(int n, HalfT x, HalfT* out) noexcept {
        BoysAllOrdersBf16<>(n, x, out);
    }
};

// Region-partitioned 8-wide kernels shared by the fp16 and bf16 lanes. Their
// scalar tails call the half-lane scalar entries; the vector bodies hold the
// same documented bound as those entries but are not bit-identical to them (the
// bodies run the fp32 fit at full degree where the scalar lane truncates it).
template <typename Lane>
void RegionASimdHalf(int n,
                     const typename Lane::HalfT* x,
                     typename Lane::HalfT* out,
                     std::size_t count) noexcept {
    const int first = detail::f32::kPieceStart[n];
    const int last = detail::f32::kPieceStart[n + 1];

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
    }

    for (std::size_t i = count - (count % 8); i < count; ++i)
    {
        out[i] = Lane::Single(n, x[i]);
    }
}

template <typename Lane>
void RegionBSimdHalf(int n,
                     const typename Lane::HalfT* x,
                     typename Lane::HalfT* out,
                     std::size_t count) noexcept {
    static const ExpTable expTable;

    {
        for (std::size_t i = 0; i + 7 < count; i += 8)
        {
            const __m256 xv = Lane::Load(x + i);
            __m256 f = ClenshawB8F32(xv);
            const __m256 expx = Eval8Exp(expTable, xv);
            Lane::Store(out + i, f);

            for (int l = 0; l < n; ++l)
            {
                // Divide by x rather than multiply by the rounded 1/x: the upward recurrence amplifies a relative
                // perturbation by ((l + 1/2)/x) per step, so past l = x the 6e-8 reciprocal rounding compounds to
                // tens of per cent at the highest orders. The certified scalar lane divides.
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
        Lane::Batch(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

template <typename Lane>
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

        // The step's exp(-x)/2 term, which the scalar entries this lane is compared against
        // carry: the whole grid's worth of it cannot move a half, and its absence would leave
        // the two shapes solving different recurrences.
        float arguments[8];
        _mm256_storeu_ps(arguments, xv);

        for (float& argument : arguments)
        {
            argument = RegionBHalfExpF32<RegionBExp::kAccurate>(argument);
        }

        const __m256 expv = _mm256_loadu_ps(arguments);

        for (int l = 0; l < n; ++l)
        {
            f = _mm256_mul_ps(
                _mm256_sub_ps(_mm256_mul_ps(f, _mm256_set1_ps(static_cast<float>(l) + 0.5f)),
                              expv),
                invx);
        }

        Lane::Store(out + i, f);
    }

    for (std::size_t i = count - (count % 8); i < count; ++i)
    {
        out[i] = Lane::Single(n, x[i]);
    }
}

// F16C is implied by AVX2 on every shipping x86 CPU; this check is defensive
// and routes to the certified scalar lane if it ever fires.
bool F16cAvailable() noexcept {
    static const bool available = DetectF16c();
    return available;
}

void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleF16<>(n, x[i]);
        }

        return;
    }

    RegionASimdHalf<F16Lane>(n, x, out, count);
}

void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        F16 batch[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < count; ++i)
        {
            BoysAllOrdersF16<>(n, x[i], batch);

            for (int l = 0; l <= n; ++l)
            {
                out[l * count + i] = batch[l];
            }
        }

        return;
    }

    RegionBSimdHalf<F16Lane>(n, x, out, count);
}

void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleF16<>(n, x[i]);
        }

        return;
    }

    RegionCSimdHalf<F16Lane>(n, x, out, count);
}

void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleBf16<>(n, x[i]);
        }

        return;
    }

    RegionASimdHalf<Bf16Lane>(n, x, out, count);
}

void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        Bf16 batch[kMaxBoysOrder + 1];

        for (std::size_t i = 0; i < count; ++i)
        {
            BoysAllOrdersBf16<>(n, x[i], batch);

            for (int l = 0; l <= n; ++l)
            {
                out[l * count + i] = batch[l];
            }
        }

        return;
    }

    RegionBSimdHalf<Bf16Lane>(n, x, out, count);
}

void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);
    assert(BoysAvx2Available());

    if (!F16cAvailable())
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            out[i] = BoysSingleBf16<>(n, x[i]);
        }

        return;
    }

    RegionCSimdHalf<Bf16Lane>(n, x, out, count);
}
#endif // BoysFp16

} // namespace boys::detail

#else // BOYS_SIMD_X86

// Non-x86 targets: the same entry points, defined against the certified scalar lanes. The vector
// engine does not exist here, so BoysAvx2Available() is false and no entry asserts it; the region
// contract stays a sufficient precondition, just not a required one. The bodies mirror the scalar
// tails the x86 entries run for their last count % 4 elements.
namespace boys::detail {

// The AVX2 tier does not exist on a non-x86_64 target, so the inline
// availability test reads false and every entry below is the scalar lane.
bool DetectAvx2() noexcept {
    return false;
}

void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}

// No entry point calls this lane here either, and it takes no policy, only the
// multiplier, so wiring it is a signature change. It forms no reciprocal of its own:
// the run goes to the scalar lane at that lane's unnamed division form, the build
// default rather than a caller's choice.
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    double batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrders<>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

// No entry point calls this lane here either, and it takes no policy, only the
// multiplier, so wiring it is a signature change. It forms no reciprocal of its own:
// the run goes to the scalar lane at that lane's unnamed division form, the build
// default rather than a caller's choice.
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}

// The far ladder's lane on a target without the vector engine: the region's certified scalar
// body, one column at a time. It is the body the entry's own scalar path runs, at the same
// arguments and the same stride, so the lane returns the values the entry returns on a host whose
// tier is absent - and BoysAvx2Available() is false there, so no caller reaches it anyway.
void BoysFarLadderLanes(int nmax,
                        const double* x,
                        const double* halfExp,
                        std::size_t stride,
                        double* const* cols,
                        DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    static_cast<void>(halfExp); // the scalar body forms its own, at the same expression

    for (std::size_t j = 0; j < kBoysFarLaneWidth; ++j)
    {
        switch (form)
        {
        case DivisionForm::kExactDivision:
            BoysAllNBodyRegionC<DivisionForm::kExactDivision>(nmax, x[j], stride, cols[j]);
            break;

        case DivisionForm::kPlainReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kPlainReciprocal>(nmax, x[j], stride, cols[j]);
            break;

        case DivisionForm::kRefinedReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kRefinedReciprocal>(nmax, x[j], stride, cols[j]);
            break;
        }
    }
}

// The far run's kernel on a target without the vector engine: the region's certified scalar body,
// one argument of the run at a time, in the caller's own planes. It is the body the entry's scalar
// path runs, so the kernel returns what the entry returns on a host whose tier is absent - and
// BoysAvx2Available() is false there, so no caller reaches it anyway.
void BoysFarLadderRun(int nmax,
                      const double* x,
                      const std::size_t* index,
                      double* out,
                      std::size_t count,
                      std::size_t begin,
                      std::size_t end,
                      DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(begin < end);

    for (std::size_t j = begin; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;

        switch (form)
        {
        case DivisionForm::kExactDivision:
            BoysAllNBodyRegionC<DivisionForm::kExactDivision>(nmax, x[i], count, out + i);
            break;

        case DivisionForm::kPlainReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kPlainReciprocal>(nmax, x[i], count, out + i);
            break;

        case DivisionForm::kRefinedReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kRefinedReciprocal>(nmax, x[i], count, out + i);
            break;
        }
    }
}

#if BoysFp16
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleF16<>(n, x[i]);
    }
}

void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    F16 batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrdersF16<>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleF16<>(n, x[i]);
    }
}

void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleBf16<>(n, x[i]);
    }
}

void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    Bf16 batch[kMaxBoysOrder + 1];

    for (std::size_t i = 0; i < count; ++i)
    {
        BoysAllOrdersBf16<>(n, x[i], batch);

        for (int l = 0; l <= n; ++l)
        {
            out[l * count + i] = batch[l];
        }
    }
}

void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingleBf16<>(n, x[i]);
    }
}
#endif // BoysFp16

} // namespace boys::detail

#endif // BOYS_SIMD_X86

namespace boys::detail {
#if BoysFp16
#endif // BoysFp16

} // namespace boys::detail

// The packed half of the backend table: this unit's flags are what make its contraction answer
// differ from the scalar one, so the pair is measured here, and it is listed only where the tier
// is both compiled in and available at run time. The route reported is the instantiation's own,
// unfiltered by the contraction question: the packed steps name two instructions.
namespace boys::backend {
namespace detail {

std::size_t AppendPackedBackends(BackendInfo* out) noexcept {
#if BOYS_SIMD_X86
    if (!BoysAvx2Available())
    {
        return 0;
    }

    out[0] = BackendInfo{Avx2Fp64<>::kName, Avx2Fp64<>::Contracts(), Avx2Fp64<>::kRoute};
    out[1] = BackendInfo{Avx2Fp32<>::kName, Avx2Fp32<>::Contracts(), Avx2Fp32<>::kRoute};
    return 2;
#else
    (void)out;
    return 0;
#endif
}

} // namespace detail
} // namespace boys::backend
