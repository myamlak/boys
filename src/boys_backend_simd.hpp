#pragma once

// The packed arithmetic backends of the AVX2 + FMA tier.
//
// Private to the library: the intrinsics' header is not part of the public
// surface, so the packed backends are named here and reported through
// boys::backend::BoysBackends rather than declared in include/boys/backend.hpp.
//
// Both packed backends are contraction-free by construction: every multiply,
// add and fused step names its instruction, so no setting of the contraction
// flag changes what they compute. What Contracts() reports for them is
// therefore whether a kernel that wrote a bare product-plus-add in this
// arithmetic would fuse — which is a fact about the build, and is measured.

#include "boys/backend.hpp"

#if BOYS_SIMD_X86

#include <immintrin.h>

namespace boys::backend {

/// The 4-wide double arithmetic of the AVX2 + FMA tier.
struct Avx2Fp64 {
    using Value = double;
    using Storage = double;
    using Packed = __m256d;

    static constexpr std::size_t kWidth = 4;
    static constexpr const char* kName = "avx2-fp64";

    static Packed Load(const Storage* p) noexcept { return _mm256_loadu_pd(p); }
    static void Store(Storage* p, Packed v) noexcept { _mm256_storeu_pd(p, v); }
    static Packed Broadcast(Value v) noexcept { return _mm256_set1_pd(v); }
    static Packed Mul(Packed a, Packed b) noexcept { return _mm256_mul_pd(a, b); }
    static Packed Add(Packed a, Packed b) noexcept { return _mm256_add_pd(a, b); }
    static Packed Sub(Packed a, Packed b) noexcept { return _mm256_sub_pd(a, b); }

    /// `a * b + c`, fused: one instruction, one rounding.
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        return _mm256_fmadd_pd(a, b, c);
    }

    /// `a * b - c` with two roundings: the fused step with a zero addend
    /// rounds the product and nothing else, and the subtraction rounds after
    /// it. See the scalar backend for why the bare form will not do.
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return _mm256_sub_pd(_mm256_fmadd_pd(a, b, _mm256_setzero_pd()), c);
    }

    static bool Contracts() noexcept;
};

/// The 8-wide single-precision arithmetic of the AVX2 + FMA tier. It carries
/// the float lane and, through the half-format transports of f16.hpp, the
/// fp16 and bf16 lanes: their I/O is narrower, their arithmetic is this one.
struct Avx2Fp32 {
    using Value = float;
    using Storage = float;
    using Packed = __m256;

    static constexpr std::size_t kWidth = 8;
    static constexpr const char* kName = "avx2-fp32";

    static Packed Load(const Storage* p) noexcept { return _mm256_loadu_ps(p); }
    static void Store(Storage* p, Packed v) noexcept { _mm256_storeu_ps(p, v); }
    static Packed Broadcast(Value v) noexcept { return _mm256_set1_ps(v); }
    static Packed Mul(Packed a, Packed b) noexcept { return _mm256_mul_ps(a, b); }
    static Packed Add(Packed a, Packed b) noexcept { return _mm256_add_ps(a, b); }
    static Packed Sub(Packed a, Packed b) noexcept { return _mm256_sub_ps(a, b); }

    /// `a * b + c`, fused: one instruction, one rounding.
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        return _mm256_fmadd_ps(a, b, c);
    }

    /// `a * b - c` with two roundings; see the double backend.
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return _mm256_sub_ps(_mm256_fmadd_ps(a, b, _mm256_setzero_ps()), c);
    }

    static bool Contracts() noexcept;
};

// Whether a bare product-plus-add in this arithmetic would fuse is a property
// of the value type, the target and the flags, not of the width: the same
// measurement answers for the packed backend and for the scalar backend of
// the same precision.
inline bool Avx2Fp64::Contracts() noexcept {
    return detail::MeasureContraction<double>();
}

inline bool Avx2Fp32::Contracts() noexcept {
    return detail::MeasureContraction<float>();
}

} // namespace boys::backend

#endif // BOYS_SIMD_X86
