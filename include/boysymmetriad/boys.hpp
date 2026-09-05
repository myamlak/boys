#pragma once

#include <cstddef>

#if BoysFp16
#include "boysymmetriad/f16.hpp"
#endif

/// \defgroup boysymmetriad Boys-function kernel
///
/// Self-contained evaluation of the Boys function family F_n(x),
/// n = 0..32, x >= 0 — the accuracy-critical building block of
/// Gaussian-basis integral recursion, validated against a committed
/// 30-digit reference grid.

/// \file
/// The Boys function kernel.
///
/// F_n(x) = the integral of t^(2n) e^(-x t^2) over t in [0, 1], n >= 0, x >= 0
///
/// Every one- and two-electron integral over Gaussian primitives reduces to
/// evaluations of this family (Boys 1950, Proc. R. Soc. A 200, 542); the
/// standard upward/downward recursion between orders is from Shavitt
/// (1963, Methods in Computational Physics 2, 1).
///
/// Design (measured, not assumed — full study in the accompanying paper):
///  - region A  [0, x0): per-order piecewise Chebyshev fits evaluated by a
///    split Clenshaw recurrence (division-free, FMA-friendly), seeded from
///    F_n and recursed downward for batches;
///  - region B  [x0, x1): a single F0 fit plus upward recursion;
///  - region C  [x1, inf): the asymptotic form 1/2 sqrt(pi/x) and upward
///    recursion;
/// with x0/x1 chosen so that the maximum absolute error stays at or below
/// 5e-14 (double) / 1.5e-7 (float, the certified F32 bound) across n = 0..32
/// — validated against a 30-digit mpmath reference grid, and reproduced by
/// tools/gen_boys_coefficients.py.
///
/// The lane split is deliberate: consumer GPUs run double precision at 1/32
/// of single-precision throughput (measured on a Quadro T1000: the float
/// kernel is 8.4x faster than the fastest double-precision table kernel and
/// 5.2x faster than the fastest double kernel overall), so callers that
/// tolerate the certified 1.5e-7 absolute error should prefer the F32 lane
/// there.
///
/// Behind the BoysFp16 build-time seam (default ON) the fp16 lane extends
/// the certified mixed-precision boundary: F16/Bf16 inputs and outputs
/// around the certified fp32 engine, so the lane delivers the F32 lane's
/// certified bound (1.5e-7 absolute) up to one half-ULP of representation
/// (in fact the fp16 roles run the engine at the tighter 1e-7 region
/// budgets). F16/Bf16 alias
/// std::float16_t / std::bfloat16_t where the toolchain ships them (GCC 13+,
/// Clang 17+); the MSVC STL does not (see f16.hpp), so this library
/// supplies the self-contained wrappers there. The F16/Bf16 SIMD entries
/// are AVX2-only and follow the same region-partitioned engine pattern as
/// the F64 lanes above.
///
/// **Accuracy contract.** Every lane entry is templated on
/// ONE compile-time multiplier \c kAccuracyMultiplier >= 1.0, default
/// \c kBoysFullAccuracyMultiplier (the full static accuracy). The contract
/// per lane and region, with m = kAccuracyMultiplier:
///
/// | Lane | region A (x < x0) | region B (x0 <= x < x1) | region C (x >= x1) |
/// |---|---|---|---|
/// | double single | \|F̂ − F\| ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·5.5e-14 |
/// | double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
/// | float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
/// | fp16 / bf16 | ≤ m·1e-7 + ½ULP | ≤ m·1e-7 + ½ULP | ≤ m·1e-7 + ½ULP |
///
/// i.e. |F̂_n(x) − F_n(x)| ≤ m·B_region per lane and region, with B_region the
/// asserted per-region bounds above (measured — full study in the
/// accompanying paper). **m = 1 is bit-identical to the certified lanes**:
/// the default instantiations select the full-accuracy bodies verbatim — no
/// branch, indirection, or runtime dispatch anywhere on the m = 1 path, and
/// all existing call sites compile unchanged. Relaxation (m > 1) truncates
/// the Chebyshev seed fits to the certified effective degree d'(m) = min{d' :
/// Δ(d')·A ≤ (m−1)·B_region}, Δ(d') = Σ_{k>d'}|c_k| the dropped-coefficient
/// tail and A the path's seed-error amplification — a-priori bounded, never
/// tuned (D-F3); the delivered error ≤ m·B_region follows from the m = 1
/// asserted bound plus the tail bound. The contract and the work are
/// **monotone in m** (d' is non-increasing in m); pointwise error is
/// explicitly NOT guaranteed monotone — a larger m may occasionally change a
/// pointwise error, but never beyond the m·B_region envelope. Region C has
/// no relaxable resource; its contract holds with slack at every m.
/// Instantiations with kAccuracyMultiplier < 1.0 are compile-time errors.
/// The CUDA lane's relaxed path holds one degree table per process (filled
/// once per m, the InitializeTables thread-safety contract). The fp16/bf16
/// m·1e-7 + ½ULP base above is the suite-asserted bound — strictly stronger
/// than the accompanying manuscript's error-bounded 1.5e-7 + ½ULP: the
/// suite tolerance's 1e-7 base is the fp16 lanes' asserted base, not the
/// float budget.
///
/// \ingroup boysymmetriad

namespace boysymmetriad {

/// Highest Boys order supported by the kernel.
inline constexpr int kMaxBoysOrder = 32;

/// The default accuracy multiplier of every lane: m = 1 is full static
/// accuracy, bit-identical to the certified lanes (the accompanying paper's
/// contract).
/// Larger m values trade certified accuracy for work via compile-time degree
/// truncation (see the contract table in the file preamble).
inline constexpr double kBoysFullAccuracyMultiplier = 1.0;

/// F_n(x) in double precision, |F̂ − F| ≤ m·B_region per region (contract
/// table in the file preamble; m = kAccuracyMultiplier).
///
/// \tparam kAccuracyMultiplier the accuracy multiplier m >= 1.0; 1.0 = full
///         static accuracy, bit-identical to the certified lane; relaxes the
///         per-region bound to m·B_region (compile-time degree truncation,
///         monotone in m)
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
double BoysSingle(int n, double x) noexcept;

/// F_0(x)..F_nmax(x) in double precision, |F̂ − F| ≤ m·B_region per value.
///
/// The batch is the pattern real integral engines use (McMurchie-Davidson /
/// Obara-Saika recursions consume all orders at once) and is considerably
/// cheaper than nmax + 1 single evaluations.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysBatch(int nmax, double x, double* out) noexcept;

/// F_n(x) in single precision, |F̂ − F| ≤ m·1.5e-7.
///
/// Same piecewise structure as the double lane with per-order float fits.
/// This is the recommended lane for GPU integral evaluation.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
float BoysSingleF32(int n, float x) noexcept;

/// F_0(x)..F_nmax(x) in single precision, |F̂ − F| ≤ m·1.5e-7 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysBatchF32(int nmax, float x, float* out) noexcept;

/// True if the CPU executes AVX2; the SIMD entry points below require it.
///
/// \returns true when AVX2 is available on the processor and exposed by the OS (OSXSAVE)
///
/// \ingroup boysymmetriad
bool BoysAvx2Available() noexcept;

/// F_n(x) for an array of arguments in region A (x < x0), same n, AVX2,
/// |F̂ − F| ≤ m·1e-15 per value.
///
/// The engine pattern: inputs are partitioned by region first (Tsuji-style),
/// so every 4-lane vector is homogeneous and no lane divergence is paid.
/// Arguments outside region A are undefined behavior — partition first.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of arguments in [0, x0)
/// \param out    receives F_n(x[i])
/// \param count  number of elements; x and out must hold count values
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for an array of arguments in region B (x0 <= x < x1), same n,
/// AVX2, |F̂ − F| ≤ m·5.5e-14 per value. e^{-x} comes from a small Taylor
/// table (gather + FMA) — no scalar special-function calls inside the vector
/// loop.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of arguments in [x0, x1)
/// \param out    receives n + 1 contiguous blocks of count values
///               (out[order * count + i] = F_order(x[i]))
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimd(int n, const double* x, double* out, std::size_t count) noexcept;

/// F_n(x) for an array of arguments in region C (x >= x1), same n, AVX2,
/// |F̂ − F| ≤ m·5.5e-14 per value (the region has no relaxable resource; the
/// contract holds with slack at every m). Pure asymptotic form — the
/// cheapest path.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of arguments >= x1
/// \param out    receives F_n(x[i])
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimd(int n, const double* x, double* out, std::size_t count) noexcept;

#if BoysFp16
/// F_n(x) in IEEE half precision — the fp16 lane of the certified
/// mixed-precision boundary (not a standalone fp16 API): fp16 I/O around
/// the certified fp32 engine. The argument is rounded to fp16 before
/// evaluation (the lane evaluates at the fp16 value), the computation is
/// the F32 lane's, and the result is the correctly rounded fp16 of it —
/// |result - F_n(x16)| <= m·1e-7 + one half-ULP of representation (the
/// representation term is m-independent).
///
/// \tparam kAccuracyMultiplier see BoysSingle (forwards to the F32 engine)
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \returns     F_n(x) in fp16
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
F16 BoysSingleF16(int n, F16 x) noexcept;

/// F_0(x)..F_nmax(x) in fp16, same certified-boundary contract as
/// BoysSingleF16 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysBatchF16(int nmax, F16 x, F16* out) noexcept;

/// F_n(x) in bfloat16 — the Bf16 lane of the certified mixed-precision
/// boundary, same
/// contract as BoysSingleF16 (bf16 representation term, 8-bit mantissa).
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \returns     F_n(x) in bf16
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
Bf16 BoysSingleBf16(int n, Bf16 x) noexcept;

/// F_0(x)..F_nmax(x) in bf16, same contract as BoysBatchF16 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysBatchBf16(int nmax, Bf16 x, Bf16* out) noexcept;

/// F_n(x) for an array of fp16 arguments in region A (x < x0), same n,
/// AVX2 (8 lanes, F16C unpack to the fp32 engine).
///
/// Same engine pattern and preconditions as BoysRegionASimd: inputs must
/// be partitioned by region first; BoysAvx2Available() must hold (every
/// AVX2 CPU ships F16C; a CPUID F16C check routes to the scalar lane when
/// absent). Arguments outside region A are undefined behavior.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of fp16 arguments in [0, x0)
/// \param out    receives F_n(x[i]) in fp16
/// \param count  number of elements; x and out must hold count values
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for an array of fp16 arguments in region B
/// (x0 <= x < x1), same n, AVX2. Batch layout as BoysRegionBSimd
/// (out[order * count + i] = F_order(x[i])).
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of fp16 arguments in [x0, x1)
/// \param out    receives n + 1 contiguous blocks of count fp16 values
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;

/// F_n(x) for an array of fp16 arguments in region C (x >= x1), same n,
/// AVX2. Pure asymptotic form, like BoysRegionCSimd.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of fp16 arguments >= x1
/// \param out    receives F_n(x[i]) in fp16
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdF16(int n, const F16* x, F16* out, std::size_t count) noexcept;

/// F_n(x) for an array of bf16 arguments in region A (x < x0), same n,
/// AVX2. Same engine pattern and preconditions as BoysRegionASimdF16.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of bf16 arguments in [0, x0)
/// \param out    receives F_n(x[i]) in bf16
/// \param count  number of elements; x and out must hold count values
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionASimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;

/// F_0(x)..F_n(x) for an array of bf16 arguments in region B
/// (x0 <= x < x1), same n, AVX2. Batch layout as BoysRegionBSimd.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of bf16 arguments in [x0, x1)
/// \param out    receives n + 1 contiguous blocks of count bf16 values
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionBSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;

/// F_n(x) for an array of bf16 arguments in region C (x >= x1), same n,
/// AVX2. Pure asymptotic form, like BoysRegionCSimd.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n      order, 0..kMaxBoysOrder
/// \param x      array of bf16 arguments >= x1
/// \param out    receives F_n(x[i]) in bf16
/// \param count  number of elements
///
/// \ingroup boysymmetriad
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionCSimdBf16(int n, const Bf16* x, Bf16* out, std::size_t count) noexcept;
#endif // BoysFp16

// The default (m = 1) instantiations are provided by boys.cpp / boys_simd.cpp;
// the declarations below route every call site (which uses the
// default full-accuracy multiplier) to those instantiations — no implicit
// instantiation, no code duplication across TUs.
/// \brief The m = 1 double single instantiation (see the primary template for the contract).
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      argument >= 0
/// \returns     F_n(x)
extern template double BoysSingle<kBoysFullAccuracyMultiplier>(int n, double x) noexcept;
/// \brief The m = 1 double batch instantiation.
/// \param nmax   highest order; out must hold nmax + 1 values
/// \param x      argument >= 0
/// \param out    receives F_0(x)..F_nmax(x)
extern template void BoysBatch<kBoysFullAccuracyMultiplier>(int nmax,
                                                            double x,
                                                            double* out) noexcept;
/// \brief The m = 1 float single instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      argument >= 0
/// \returns     F_n(x)
extern template float BoysSingleF32<kBoysFullAccuracyMultiplier>(int n, float x) noexcept;
/// \brief The m = 1 float batch instantiation.
/// \param nmax   highest order; out must hold nmax + 1 values
/// \param x      argument >= 0
/// \param out    receives F_0(x)..F_nmax(x)
extern template void BoysBatchF32<kBoysFullAccuracyMultiplier>(int nmax,
                                                               float x,
                                                               float* out) noexcept;
/// \brief The m = 1 double region-A SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region A
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionASimd<kBoysFullAccuracyMultiplier>(int n,
                                                                  const double* x,
                                                                  double* out,
                                                                  std::size_t count) noexcept;
/// \brief The m = 1 double region-B SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region B
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionBSimd<kBoysFullAccuracyMultiplier>(int n,
                                                                  const double* x,
                                                                  double* out,
                                                                  std::size_t count) noexcept;
/// \brief The m = 1 double region-C SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region C
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionCSimd<kBoysFullAccuracyMultiplier>(int n,
                                                                  const double* x,
                                                                  double* out,
                                                                  std::size_t count) noexcept;
#if BoysFp16
/// \brief The m = 1 fp16 single instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      argument >= 0
/// \returns     F_n(x) in fp16
extern template F16 BoysSingleF16<kBoysFullAccuracyMultiplier>(int n, F16 x) noexcept;
/// \brief The m = 1 fp16 batch instantiation.
/// \param nmax   highest order; out must hold nmax + 1 values
/// \param x      argument >= 0
/// \param out    receives F_0(x)..F_nmax(x)
extern template void BoysBatchF16<kBoysFullAccuracyMultiplier>(int nmax, F16 x, F16* out) noexcept;
/// \brief The m = 1 bf16 single instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      argument >= 0
/// \returns     F_n(x) in bf16
extern template Bf16 BoysSingleBf16<kBoysFullAccuracyMultiplier>(int n, Bf16 x) noexcept;
/// \brief The m = 1 bf16 batch instantiation.
/// \param nmax   highest order; out must hold nmax + 1 values
/// \param x      argument >= 0
/// \param out    receives F_0(x)..F_nmax(x)
extern template void BoysBatchBf16<kBoysFullAccuracyMultiplier>(int nmax,
                                                                Bf16 x,
                                                                Bf16* out) noexcept;
/// \brief The m = 1 fp16 region-A SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region A
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionASimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                                     const F16* x,
                                                                     F16* out,
                                                                     std::size_t count) noexcept;
/// \brief The m = 1 fp16 region-B SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region B
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionBSimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                                     const F16* x,
                                                                     F16* out,
                                                                     std::size_t count) noexcept;
/// \brief The m = 1 fp16 region-C SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region C
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionCSimdF16<kBoysFullAccuracyMultiplier>(int n,
                                                                     const F16* x,
                                                                     F16* out,
                                                                     std::size_t count) noexcept;
/// \brief The m = 1 bf16 region-A SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region A
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionASimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                                      const Bf16* x,
                                                                      Bf16* out,
                                                                      std::size_t count) noexcept;
/// \brief The m = 1 bf16 region-B SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region B
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionBSimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                                      const Bf16* x,
                                                                      Bf16* out,
                                                                      std::size_t count) noexcept;
/// \brief The m = 1 bf16 region-C SIMD instantiation.
/// \param n      Boys order in [0, kMaxBoysOrder]
/// \param x      arguments in region C
/// \param out    receives the results
/// \param count  element count
extern template void BoysRegionCSimdBf16<kBoysFullAccuracyMultiplier>(int n,
                                                                      const Bf16* x,
                                                                      Bf16* out,
                                                                      std::size_t count) noexcept;
#endif // BoysFp16

} // namespace boysymmetriad
