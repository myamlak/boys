#pragma once

// The packed arithmetic backends of the AVX2 + FMA tier.
//
// Private to the library: the intrinsics' header is not part of the public
// surface, so the packed backends are named here and reported through
// boys::backend::BoysBackends rather than declared in include/boys/backend.hpp.
//
// Both are contraction-free by construction: every step names its instruction,
// so no setting of the contraction flag changes what they compute. What
// Contracts() reports for them is the separate question of whether a bare
// product-plus-add in this arithmetic would fuse.
//
// Each carries its route as a template parameter rather than as a constant, so
// one build instantiates either arithmetic and a test holds the two against
// each other. The default is the build's selection, which is where the scalar
// pair reads its route from too, so a call site that names none gets the
// arithmetic the library reports. Being contraction-free by construction is
// what lets the separate route mean two roundings here on every build: the
// product and the sum are two named instructions, so no propagation of the
// contraction flag into this arithmetic can make them one.

#include "boys/backend.hpp"

#if BOYS_SIMD_X86

#include <immintrin.h>

namespace boys::backend {

/// The 4-wide double arithmetic of the AVX2 + FMA tier, at one multiply-add
/// route.
///
/// \tparam Route the route this instantiation runs. The default is the route
///         the build selected, so an instantiation that names none is the
///         arithmetic BoysBackends() reports for this build; naming one
///         explicitly is what puts both arithmetics in one binary.
template <MulAddRoute Route = detail::kSelectedRoute>
struct Avx2Fp64 {
    using Value = double;
    using Storage = double;
    using Packed = __m256d;

    static constexpr std::size_t kWidth = 4;
    static constexpr const char* kName = "avx2-fp64";

    /// The route this instantiation runs: the selection, or the argument a
    /// caller named. Unlike the scalar pair this is not the route in force
    /// filtered through the build's contraction, because the two routes are
    /// two spellings here rather than two readings of one bare expression.
    static constexpr MulAddRoute kRoute = Route;

    static Packed Load(const Storage* p) noexcept { return _mm256_loadu_pd(p); }
    static void Store(Storage* p, Packed v) noexcept { _mm256_storeu_pd(p, v); }
    static Packed Broadcast(Value v) noexcept { return _mm256_set1_pd(v); }
    static Packed Mul(Packed a, Packed b) noexcept { return _mm256_mul_pd(a, b); }
    static Packed Add(Packed a, Packed b) noexcept { return _mm256_add_pd(a, b); }
    static Packed Sub(Packed a, Packed b) noexcept { return _mm256_sub_pd(a, b); }

    /// `a * b + c` at this instantiation's route.
    ///
    /// Fused, one instruction and one rounding. Separate, the product rounds
    /// and then the sum rounds: two named instructions, and so two roundings on
    /// every build the flag is in force on. The packed lane pays no call for
    /// the separate route, which the scalar lane's does on a target without the
    /// fused instruction.
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kRoute == MulAddRoute::kFused)
        {
            return _mm256_fmadd_pd(a, b, c);
        } else
        {
            return _mm256_add_pd(_mm256_mul_pd(a, b), c);
        }
    }

    /// `a * b - c` with two roundings: the fused step with a zero addend rounds
    /// the product and nothing else, and the subtraction rounds after it.
    ///
    /// Outside the route, as it is for the scalar pair: the operation's own
    /// contract is two roundings on every build, and the spelling is what keeps
    /// it. The zero addend is exact, so it is the product rounded once; a
    /// contraction pass has nothing left to fuse.
    ///
    /// One transformation could defeat that spelling: a compiler licensed to
    /// ignore the sign of zero may fold `fma(a, b, +0)` into `a * b` and then
    /// contract the product into the subtraction. No flag set this tree builds
    /// with grants that licence: MSVC builds at /fp:precise, and the GCC and
    /// clang sets add -ffp-contract and never -ffast-math, which is the flag
    /// the licence lives under.
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return _mm256_sub_pd(_mm256_fmadd_pd(a, b, _mm256_setzero_pd()), c);
    }

    static bool Contracts() noexcept;
};

/// The 8-wide single-precision arithmetic of the AVX2 + FMA tier. It carries
/// the float lane and, through the half-format transports of f16.hpp, the
/// fp16 and bf16 lanes: their I/O is narrower, their arithmetic is this one.
///
/// \tparam Route the route this instantiation runs; see the double backend.
template <MulAddRoute Route = detail::kSelectedRoute>
struct Avx2Fp32 {
    using Value = float;
    using Storage = float;
    using Packed = __m256;

    static constexpr std::size_t kWidth = 8;
    static constexpr const char* kName = "avx2-fp32";

    /// The route this instantiation runs; see the double backend.
    static constexpr MulAddRoute kRoute = Route;

    static Packed Load(const Storage* p) noexcept { return _mm256_loadu_ps(p); }
    static void Store(Storage* p, Packed v) noexcept { _mm256_storeu_ps(p, v); }
    static Packed Broadcast(Value v) noexcept { return _mm256_set1_ps(v); }
    static Packed Mul(Packed a, Packed b) noexcept { return _mm256_mul_ps(a, b); }
    static Packed Add(Packed a, Packed b) noexcept { return _mm256_add_ps(a, b); }
    static Packed Sub(Packed a, Packed b) noexcept { return _mm256_sub_ps(a, b); }

    /// `a * b + c` at this instantiation's route; see the double backend.
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kRoute == MulAddRoute::kFused)
        {
            return _mm256_fmadd_ps(a, b, c);
        } else
        {
            return _mm256_add_ps(_mm256_mul_ps(a, b), c);
        }
    }

    /// `a * b - c` with two roundings; see the double backend.
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return _mm256_sub_ps(_mm256_fmadd_ps(a, b, _mm256_setzero_ps()), c);
    }

    static bool Contracts() noexcept;
};

// A bare product-plus-add fuses by value type, target and flags, not by width:
// the same measurement answers for both backends of one precision. The answer
// is a property of the translation unit that asks, and both routes share it,
// because what it measures is a bare expression rather than what these kernels
// are written with.
template <MulAddRoute Route>
inline bool Avx2Fp64<Route>::Contracts() noexcept {
    return detail::MeasureContraction<double>();
}

template <MulAddRoute Route>
inline bool Avx2Fp32<Route>::Contracts() noexcept {
    return detail::MeasureContraction<float>();
}

} // namespace boys::backend

#endif // BOYS_SIMD_X86
