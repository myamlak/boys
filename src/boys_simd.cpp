#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

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
// BOYS_SIMD_X86 answers one question - does this TU compile the vector tier?
// CMake derives it from a configure-time probe and states it on the `boys`
// target; when nobody states it the guard detects it from the same two
// predefines, which is the case for a consumer compiling this source itself.
//
// Unstated and not x86_64 is an #error rather than a quiet 0: this TU cannot
// tell a real non-x86 target from an x86 target whose predefines it has not
// been taught, and answering 0 on an x86_64 target would compile the intrinsics
// out, soft-skip every SIMD test and leave CI green.
//
// On a non-x86 target the same entry points are defined against the certified
// scalar lanes instead - same signatures, same contracts, same numerics.
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

// AVX2 region-sorted lanes. The engine pattern (region-first): partition the
// arguments by region FIRST so every 4-lane vector is homogeneous; the unsorted
// variant pays a measured 2.3x divergence penalty.
//
// Callers must check BoysAvx2Available() before invoking these; the kernels are
// FMA chains, so the predicate requires the FMA feature bit as well.
//
// Every region entry is a template on kAccuracyMultiplier, compiled twice under
// if constexpr: the m = 1 branch is the full-accuracy body verbatim (the
// bit-identity pin), the relaxed branch passes the per-piece / per-order
// effective degrees into the degree-parameterized Clenshaw variants. The
// default m = 1 instantiations live at the bottom of this TU.

namespace boys::detail {
namespace {

using detail::kX0;
using detail::kX1;

constexpr double kHalfSqrtPi = 0.886226925452758014;

// CPUID + OSXSAVE detection of the AVX2 + FMA scope (the F64/F32 lanes'
// engine). FMA belongs to the scope, it is not an optional extra: a processor
// with AVX2 but no FMA would be dispatched into an unimplemented instruction.
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

// ---------------------------------------------------------------------------
// e^{-x} on [0, 30]: a degree-4 Taylor table, one row per grid abscissa
// x_i = i * kStep, rows padded to 8 doubles so the gathers can use the legal
// scale 8. Worst absolute error 1.83e-17.
//
// A row holds the quartic Taylor polynomial of e^{-x} at x_i in the monomial
// basis of the ABSOLUTE argument x, so Eval4 is a plain Horner chain: splitting
// e^{-x} = e^{-x_i} e^{-h} at h = x - x_i gives, for the coefficient of x^k,
//
//     a_k = e^{-x_i} * (-1)^k * S_{4-k} / k!,   S_m = sum_{j=0..m} x_i^j / j!.
//
// The row stores (-1)^k a_k, i.e. the alternating sign is folded into the table
// and taken back out by the sign pattern of Eval4's FMA chain.
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

// Split Clenshaw (even/odd), packed, half-depth FMA chains:
// T_{2j+1}(t) = t * D_j(v) with the D recurrence.
//
// One body for every packed width and precision: which multiply-add a step
// uses, and how many roundings it makes, is the backend's, and the body below
// is written once against it. What stays here is the mapped argument, in the
// packed lanes' own form - the interval reached with a single fused step rather
// than the scalar lanes' two.
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
    return RegionAClenshaw<backend::Avx2Fp64>(detail::kCoeffs.data(), piece, deg, xv);
}

// The m = 1 entry: the piece's full degree.
inline __m256d Clenshaw4Split(const detail::OrderPiece& piece, __m256d xv) noexcept {
    return Clenshaw4SplitDeg(piece, piece.deg, xv);
}

// The m = 1 entry: the fit's full degree, known at compile time.
inline __m256d ClenshawB4(__m256d xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp64, detail::kBDeg>(detail::kBcoeffs.data(), 0, xv);
}

// 4-wide region-B F0 seed at a runtime degree, for the relaxed RegionB loop
// only; the data-dependent loop is not inlined.
//
// The library instantiates the SIMD region entries at m = 1 only, so in this TU
// that caller sits in the discarded arm of an `if constexpr` and GCC/Clang see
// a defined-but-unused internal function; a relaxed instantiation would use it.
[[maybe_unused]] inline __m256d ClenshawB4Deg(int deg, __m256d xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp64, -1>(detail::kBcoeffs.data(), deg, xv);
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

// No entry point calls this lane; its callers are the tests and the benchmark. It
// takes no policy, only the multiplier, so a caller cannot name it with a division
// form and wiring it is a signature change. Its vector body forms 1/x once and
// multiplies it through the ladder - the plain reciprocal, not the kRefinedReciprocal
// the entries default to - so wired as it stands it would divide in a form no caller
// named.
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

// No entry point calls this lane either, and unlike region B it holds its bound - C
// lands 5.0e-14 against the 5.5e-14 the contract table states - so the omission is
// not an accuracy one. This definition records no reason for it; the tree's one
// remark on the choice is the batch entry's note that region C runs scalar there.
//
// It takes no policy, only the multiplier, so a caller cannot name it with a division
// form and wiring it is a signature change, and its vector body forms 1/x once and
// multiplies it through the ladder - the plain reciprocal, not the kRefinedReciprocal
// the entries default to.
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
// fp16 / bf16 lanes, AVX2 scope (8 lanes; the fp32 engine, F16C/bit-trick I/O).
// Same region-partitioned engine pattern as the F64 lanes above, around the
// certified fp32 fits of detail::f32. The relaxed branches use the fp16
// computation budget (1e-7) with the F32 piece tables.
// ---------------------------------------------------------------------------
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
    return RegionAClenshaw<backend::Avx2Fp32>(detail::f32::kCoeffs.data(), piece, deg, xv);
}

// The m = 1 entry: the piece's full degree.
inline __m256 Clenshaw8SplitF32(const detail::f32::OrderPiece& piece, __m256 xv) noexcept {
    return Clenshaw8SplitF32Deg(piece, piece.deg, xv);
}

// The m = 1 entry: the fit's full degree, known at compile time.
inline __m256 ClenshawB8F32(__m256 xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp32, detail::f32::kBDeg>(
        detail::f32::kBcoeffs.data(), 0, xv);
}

// The relaxed entry: the caller's degree; see ClenshawB4Deg.
[[maybe_unused]] inline __m256 ClenshawB8F32Deg(int deg, __m256 xv) noexcept {
    return RegionBClenshaw<backend::Avx2Fp32, -1>(
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
        // The 256-bit packus takes the low 128 of each operand, so an 8-wide
        // operand would duplicate elements 0-3 and drop 4-7; the 128-bit pack
        // of the two halves assembles {lo0-3, hi4-7}.
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

// Region-partitioned 8-wide kernels shared by the fp16 and bf16 lanes. Their
// scalar tails call the half-lane scalar entries; the vector bodies hold the
// same documented bound as those entries but are not bit-identical to them (the
// bodies run the fp32 fit at full degree where the scalar lane truncates it).
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
                // ((l + 1/2)/x) per step, so past l = x that factor exceeds one
                // and the reciprocal's 6e-8 rounding compounds to tens of per
                // cent at the highest orders. The certified scalar lane divides.
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

// F16C is implied by AVX2 on every shipping x86 CPU; this check is defensive
// and routes to the certified scalar lane if it ever fires.
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
// lanes. The vector engine does not exist here, so BoysAvx2Available() is false
// and no entry asserts it; the region contract stays a sufficient precondition,
// just not a required one. The bodies mirror, element for element, the scalar
// tails the x86 entries run for their last count % 4 elements, so a caller gets
// the same numbers it would from the x86 entry; region B keeps its transposed
// layout out[l * count + i].
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
// No entry point calls this lane here either, and it takes no policy, only the
// multiplier, so wiring it is a signature change. It forms no reciprocal of its own:
// the run goes to the scalar lane at that lane's unnamed division form, the build
// default rather than a caller's choice.
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
// No entry point calls this lane here either, and it takes no policy, only the
// multiplier, so wiring it is a signature change. It forms no reciprocal of its own:
// the run goes to the scalar lane at that lane's unnamed division form, the build
// default rather than a caller's choice.
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
// in boys.hpp.
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

// The packed half of the backend table. This unit's flags are what make its
// contraction answer different from the scalar one, so the pair is measured
// here; it is listed only where the tier is both compiled in and available at
// run time, because the probe executes the instructions it measures.
namespace boys::backend {
namespace detail {

std::size_t AppendPackedBackends(BackendInfo* out) noexcept {
#if BOYS_SIMD_X86
    if (!BoysAvx2Available())
    {
        return 0;
    }

    out[0] = BackendInfo{Avx2Fp64::kName, Avx2Fp64::Contracts(), Avx2Fp64::kRoute};
    out[1] = BackendInfo{Avx2Fp32::kName, Avx2Fp32::Contracts(), Avx2Fp32::kRoute};
    return 2;
#else
    (void)out;
    return 0;
#endif
}

} // namespace detail
} // namespace boys::backend
