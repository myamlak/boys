#pragma once

#include <cstddef>

#if BoysFp16
#include "boys/f16.hpp"
#include "boys/half2.hpp"
#endif

/// \defgroup boys Boys-function kernel
///
/// Self-contained evaluation of the Boys function family F_n(x),
/// n = 0..32, x >= 0 — the accuracy-critical building block of
/// Gaussian-basis integral recursion, validated against a committed
/// 45-digit reference grid.

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
/// Design (measured, not assumed):
///  - region A  [0, x0): per-order piecewise Chebyshev fits evaluated by a
///    split Clenshaw recurrence (division-free, FMA-friendly), seeded from
///    F_n and recursed downward for batches;
///  - region B  [x0, x1): a single F0 fit plus upward recursion;
///  - region C  [x1, inf): the asymptotic form 1/2 sqrt(pi/x) and upward
///    recursion;
/// with x0/x1 chosen so that the maximum absolute error stays at or below
/// 5e-14 (double) / 1.5e-7 (float, the certified F32 bound) across n = 0..32
/// — validated against a 45-digit mpmath reference grid, and reproduced by
/// tools/gen_boys_coefficients.py.
///
/// The lane split is deliberate: consumer GPUs run double precision at 1/32
/// of single-precision throughput (measured on a Quadro T1000: the float
/// kernel is 8.4x faster than the fastest double-precision table kernel and
/// 5.2x faster than the fastest double kernel overall), so callers that
/// tolerate the certified 1.5e-7 absolute error should prefer the F32 lane
/// there.
///
/// **Threading.** Every entry is single-threaded and a pure function of its
/// arguments: it spawns no threads, and reads no shared mutable state (the CUDA
/// lane's device tables are the one exception, and its InitializeTables warm-up
/// is the documented contract there). Two calls with the same arguments return
/// the same values whatever else the process is doing, and a call made from
/// inside a caller's own parallel region neither nests nor waits on any lock —
/// which is what makes that region safe to nest this library inside. There is no
/// threaded entry by design: a caller that wants the work spread over threads
/// divides its argument array into batches and calls the batch entry from its
/// own threads, since the per-argument value depends only on that argument and
/// nmax; distinct batches share nothing and may be called concurrently.
///
/// Behind the BoysFp16 build-time seam (default ON) the fp16 lane extends
/// the certified mixed-precision boundary: F16/Bf16 inputs and outputs
/// around the certified fp32 engine, so the lane delivers the F32 lane's
/// certified bound (1.5e-7 absolute) up to one half-ULP of representation
/// (in fact the fp16 roles run the engine at the tighter 1e-7 region
/// budgets). F16/Bf16 alias
/// std::float16_t / std::bfloat16_t where the toolchain ships them (GCC 13+,
/// Clang 17+); the MSVC STL does not (see f16.hpp), so this library
/// supplies the self-contained wrappers there. The F16/Bf16 region kernels
/// are AVX2-only and follow the same region-partitioned engine pattern as
/// the F64 lanes above.
///
/// The native half lane (BoysAllOrdersHalf2, BoysAllNF16Native, half2.hpp) is
/// a third thing again: not I/O around an engine but region C's ladder
/// itself in packed binary16 — a half2.hpp operation per step, correctly
/// rounded to half, two arguments to a register. It has no region but region
/// C (no table of regions A and B is representable in half), no accuracy
/// multiplier (region C carries no truncatable resource), and a bound of its
/// own, an order of magnitude looser than the fp16 I/O lane's: the I/O lane
/// rounds once per value, this one once per operation. Its tensor-core
/// relation is stated where it belongs — nowhere in this lane: the ladder is
/// a scalar recurrence, so there is no matrix product for a tensor core to
/// take.
///
/// **Accuracy contract.** Every lane entry is templated on
/// ONE compile-time multiplier \c kAccuracyMultiplier >= 1.0, default
/// \c kBoysFullAccuracyMultiplier (the full static accuracy). The contract
/// per lane and region, with m = kAccuracyMultiplier:
///
/// | Lane | region A | extended band | region B (x0 <= x < x1) | region C (x >= x1) |
/// |---|---|---|---|---|
/// | double single | \|F̂ − F\| ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·3e-14 | ≤ m·5.5e-14 |
/// | double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
/// | float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
/// | fp16 / bf16 | ≤ m·1e-7 + ½ULP | ≤ m·1e-7 + ½ULP | ≤ m·1e-7 + ½ULP | ≤ m·1e-7 + ½ULP |
/// | native half | — | — | — | ≤ 8 ULP of the returned value |
///
/// Region A is the per-order Chebyshev fits' own argument range, the extended
/// band runs from the end of that range to x0 = 11.899848152108484 and is
/// evaluated by upward recursion from a band seed, x0 <= x < x1 =
/// 28.98933773882074 is region B, and x >= x1 is region C. The per-order end
/// of region A is the largest argument that order's fit is evaluated at, and
/// it is order-dependent; the two rows that are not collapsed — the single
/// lane's fits and its band — are the figures to read when that boundary
/// matters, and the band's 3e-14 is the looser of the two. The native half
/// lane is region C only.
///
/// i.e. |F̂_n(x) − F_n(x)| ≤ m·B_region per lane and region, with B_region the
/// asserted per-region bounds above (measured). **m = 1 is bit-identical
/// to the certified lanes**:
/// the default instantiations select the full-accuracy bodies verbatim — no
/// branch, indirection, or runtime dispatch anywhere on the m = 1 path, and
/// all existing call sites compile unchanged. Relaxation (m > 1) truncates
/// the Chebyshev seed fits to the certified effective degree d'(m) = min{d' :
/// Δ(d')·A ≤ (m−1)·B_region}, Δ(d') = Σ_{k>d'}|c_k| the
/// dropped-coefficient tail and A the path's seed-error amplification — a-priori bounded, never
/// tuned; the delivered error ≤ m·B_region follows from the m = 1
/// asserted bound plus the tail bound. The contract and the work are
/// **monotone in m** (d' is non-increasing in m); pointwise error is
/// explicitly NOT guaranteed monotone — a larger m may occasionally change a
/// pointwise error, but never beyond the m·B_region envelope. Region C has
/// no relaxable resource; its contract holds with slack at every m.
/// Instantiations with kAccuracyMultiplier < 1.0 are compile-time errors.
/// The CUDA lane's relaxed path holds one degree table per process (filled
/// once per m, the InitializeTables thread-safety contract). The fp16/bf16
/// m·1e-7 + ½ULP base above is the suite-asserted bound — strictly stronger
/// than the float lanes' documented 1.5e-7 + ½ULP: the suite tolerance's
/// 1e-7 base is the fp16 lanes' asserted base, not the float budget.
///
/// **The fp16/bf16 bound is a claim only where the value exceeds it.** With u
/// the format's quantum at the returned value, the half lanes bind where
/// |F_n(x)| > m·1e-7 + ½u — there the return is within that distance of the
/// value — and over no other arguments, and **no accuracy is claimed** past
/// that ceiling: a caller that needs |F_n(x)| at or below m·1e-7 wants the
/// float or the double lane, which carry no such floor. What the return is past
/// the ceiling differs between the two formats, because their exponent ranges
/// do. In f16 the ceiling and the format's floor coincide: every argument past
/// it returns a subnormal half, then exactly zero, with the bound met by that
/// floor rather than by the arithmetic — the range begins just above x0 at the
/// highest orders, and every return at or above x1 from order 3 upward is
/// subnormal or zero. In bf16 the exponent range equals float's, so the 1e-7
/// base is met by the arithmetic down to |F_n(x)| of about 2^-126, and only
/// below that does the return become the format's zero. Either way the
/// arguments past the ceiling are counted, not passed, wherever this bound is
/// reported.
///
/// \ingroup boys

namespace boys {

/// Highest Boys order supported by the kernel.
inline constexpr int kMaxBoysOrder = 32;

/// The default accuracy multiplier of every lane: m = 1 is full static
/// accuracy, bit-identical to the certified lanes (the documented
/// contract).
/// Larger m values trade certified accuracy for work via compile-time degree
/// truncation (see the contract table in the file preamble).
inline constexpr double kBoysFullAccuracyMultiplier = 1.0;

/// A run-time accuracy tier: one of the multipliers this kernel
/// instantiates, chosen per call rather than fixed at build time.
///
/// The tier belongs to the call and to nothing else. Nothing is carried
/// between calls, so a coarse tier picked for one call is never reused by a
/// later call that did not ask for it.
///
/// The relaxed enumerators are consecutive rungs of one design family — the
/// same a-priori degree truncation, monotone in m — spaced so that their
/// certified bounds stay distinct.
///
/// The enumerators listed are the tiers this build serves. A value outside
/// them — cast in from outside the enum, or named by a newer header — is not a
/// tier, and every entry on this surface treats it as \c kReference rather
/// than guessing a rung: the fallback is never coarser than any tier this
/// build can name, so a caller can never be handed a value at an accuracy it
/// did not ask for.
///
/// \ingroup boys
enum class AccuracyTier : int {
    /// m = kBoysFullAccuracyMultiplier: the certified lanes, unchanged.
    kReference = 0,
    /// m = 64
    kRelaxed64,
    /// m = 256
    kRelaxed256,
    /// m = 1024
    kRelaxed1024,
    /// m = 4096
    kRelaxed4096,
    /// m = 16384
    kRelaxed16384,
    /// m = 65536
    kRelaxed65536,
};

/// The three evaluation regions of the kernel (contract table in the file
/// preamble).
///
/// \ingroup boys
enum class AccuracyRegion : int {
    kA = 0, ///< x < x0: per-order Chebyshev fits
    kB, ///< x0 <= x < x1: F0 fit plus upward recursion
    kC, ///< x >= x1: asymptotic plus upward recursion
};

/// The part of an evaluation that fixes its accuracy, so a request a tier
/// cannot meet can name what stops it.
///
/// \ingroup boys
enum class AccuracyComponent : int {
    kRegionASeed = 0,
    kRegionBFit,
    kRegionCAsymptotic,
};

/// The accuracy multiplier a tier names, so a caller can record the accuracy
/// it asked for beside the numbers it got.
///
/// A tier this build does not serve names the reference multiplier, which is
/// the rung the rest of this surface evaluates it at.
///
/// \param tier the tier
/// \returns    m >= 1.0
///
/// \ingroup boys
constexpr double AccuracyMultiplier(AccuracyTier tier) noexcept {
    switch (tier)
    {
    case AccuracyTier::kReference:
        return kBoysFullAccuracyMultiplier;
    case AccuracyTier::kRelaxed64:
        return 64.0;
    case AccuracyTier::kRelaxed256:
        return 256.0;
    case AccuracyTier::kRelaxed1024:
        return 1024.0;
    case AccuracyTier::kRelaxed4096:
        return 4096.0;
    case AccuracyTier::kRelaxed16384:
        return 16384.0;
    case AccuracyTier::kRelaxed65536:
        return 65536.0;
    }

    return kBoysFullAccuracyMultiplier;
}

/// What a tier delivers in one region, and what limits it when a tighter
/// error than that is asked for.
///
/// \ingroup boys
struct TierCoverage {
    bool meets = false; ///< the tier delivers an error at or below the request
    double reachable = 0.0; ///< the largest error the tier can deliver here
    AccuracyComponent limiting = AccuracyComponent::kRegionCAsymptotic;
};

/// The accuracy a tier delivers for arguments in \p region, and the component
/// that limits it when \p tolerance is tighter than that.
///
/// Region C is evaluated by its asymptotic form plus upward recursion and has
/// no coefficients to truncate, so its reachable error is the reference
/// tier's at every m: no tier meets a request tighter than it.
///
/// \param tier      the tier
/// \param region    the region the arguments fall in
/// \param tolerance the absolute error asked for
/// \returns         the coverage; \c meets is false when the tier cannot reach it
///
/// \ingroup boys
TierCoverage QueryTier(AccuracyTier tier, AccuracyRegion region, double tolerance) noexcept;

/// The same report for one argument, with the region taken from \p x rather
/// than named by the caller.
///
/// The region boundaries are internal and are not part of the stable surface
/// (README, "Public function signatures and supported domains"), so a caller
/// cannot in general say which region an argument falls in. Naming the wrong
/// one is not a conservative error: region C has no relaxable resource, so its
/// reachable error is the reference tier's at every m — a caller who names
/// region C for an argument that is really in region A or B is told the tier
/// reaches m times better than it does. This overload takes the guess away.
///
/// \param tier      the tier
/// \param x         the argument, >= 0
/// \param tolerance the absolute error asked for
/// \returns         the coverage for arguments in \p x's region
///
/// \ingroup boys
TierCoverage QueryTier(AccuracyTier tier, double x, double tolerance) noexcept;

/// F_0(x)..F_nmax(x) in double precision at a run-time-selected tier; the
/// batch entry's contract — the \c "double batch" row of the table in the
/// file preamble: |F̂ − F| ≤ m·5.5e-14 per value in every region, with
/// m = AccuracyMultiplier(tier).
///
/// The row is named because the table's per-region column is a different
/// number for a different lane: this entry dispatches to \c BoysAllOrders, and
/// its region-A error already reaches 3.2·m·1e-15 at m = 1 — so reading
/// \c B_region as the per-region column would promise up to 54 times tighter
/// than the code delivers. \c QueryTier reports the batch row for the same
/// reason.
///
/// One branch selects the rung, then the rung's own body runs. The reference
/// tier is the template default, so it is the same code a direct call
/// reaches.
///
/// A tier this build does not serve evaluates at the reference multiplier —
/// the same fallback \c AccuracyMultiplier reports, so the number a caller
/// records beside these values is the accuracy they were computed at.
///
/// \param tier the tier, a property of this call only
/// \param nmax highest order, 0..kMaxBoysOrder
/// \param x    argument, >= 0
/// \param out  receives nmax + 1 values, out[k] = F_k(x); every value is
///             written at every tier, including a tier this build does not
///             serve
///
/// \ingroup boys
void BoysAllOrdersAtTier(AccuracyTier tier, int nmax, double x, double* out) noexcept;


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
/// \ingroup boys
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
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllOrders(int nmax, double x, double* out) noexcept;

/// F_n(x_i) for an array of arguments at one fixed order n, double
/// precision; |F_hat - F| <= m*B_region per value (the BoysSingle
/// per-region contract, region table in the file preamble).
///
/// The fixed-n vector entry is the batch shape of integral-engine inner
/// loops that group shell pairs by angular momentum: each element needs
/// exactly one order, so no unused cross-order recursion is paid. Each
/// output element is bit-identical to BoysSingle<kAccuracyMultiplier> at
/// the same (n, x) - the m = 1 path runs the certified scalar single-lane
/// region bodies verbatim (the bit-identity pin), the relaxed path the
/// same bodies at the single-lane effective degrees. Arguments need no
/// pre-partitioning: the region dispatch is per element, the portable
/// shape. The vector tier is reached through the batch entries (BoysAllN),
/// which group the arguments once for the whole call.
///
/// Layout: out[i * stride] = F_n(x[i]), i = 0..count-1; stride is measured
/// in doubles and defaults to 1 (contiguous). The output span must hold
/// (count - 1) * stride + 1 doubles; count may be 0 (no writes). The entry
/// is scalar and portable: x and out need no alignment beyond
/// alignof(double), and the two arrays must not overlap.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param n      order, 0..kMaxBoysOrder - the batch's single fixed order
/// \param x      array of count arguments, each >= 0
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in doubles, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride = 1) noexcept;

/// Tag for the many-argument entries' already-grouped overload: the caller
/// states that the arguments are in non-decreasing order.
///
/// Every dispatch path of this library is an interval of the argument line and
/// the intervals are ordered, so the classification is monotone in x — the
/// arguments of a non-decreasing array are already contiguous by path, the
/// grouping is free, and the entry skips the sort it would otherwise pay for.
/// That is the whole precondition, stated as a property of the caller's array:
/// a caller checks it without knowing anything about the library's regions.
///
/// \ingroup boys
struct BoysSortedArgs {};

/// Workspace size, in std::size_t words, that BoysAllN uses when the caller
/// supplies one.
///
/// \param count number of arguments of the call the workspace serves
/// \returns     the required workspace size
///
/// \ingroup boys
constexpr std::size_t BoysAllNWorkspaceSize(std::size_t count) noexcept
{
    return count;
}

/// F_0(x_i)..F_nmax(x_i) for an array of arguments, double precision — every
/// order at every argument, with the per-argument dispatch and the grouping
/// done internally.
///
/// This is the batch shape a shell-quartet consumer needs: many arguments, all
/// nmax + 1 orders each, in one call. The arguments may arrive in any order:
/// the entry classifies them, groups them by dispatch path, runs the path's
/// kernel over each group, and returns the results in the caller's argument
/// order. The grouping is a documented promise, not an implementation detail —
/// a caller pays the grouped-path performance whatever its ordering, and
/// without having to know what the library groups by. A caller whose arguments
/// are already non-decreasing states it with the BoysSortedArgs overload and
/// skips the sort entirely.
///
/// Layout: order-major planes, out[k * count + i] = F_k(x[i]) — all F_0
/// contiguous, then all F_1, and so on: the layout a contraction consuming one
/// order across many arguments, or a vectorised kernel, wants. The output
/// holds count * (nmax + 1) doubles.
///
/// Accuracy: every returned value satisfies the double batch lane's documented
/// per-region bound, |F̂ − F| ≤ m·5.5e-14 (the contract table in the file
/// preamble), which is the bound the per-argument BoysAllOrders call meets —
/// the same contract, not necessarily the same bits. At m = 1 on a host with
/// AVX2 a low-order batch (where the region-A lane's one-order-at-a-time cost is
/// amortized) hands its region-A runs to that lane and differs from
/// BoysAllOrders only within that lane's own region-A budget; every other case
/// runs the per-argument path's own bodies at the batch's nmax, and is
/// bit-identical to BoysAllOrders called at that same nmax. (Region A's body
/// seeds at nmax and recurses down, so a batch's F_k for k < nmax is the
/// recurrence's value, not the one a per-order call at nmax = k walks; both are
/// inside the bound.) The suite measures the observed maximum difference on both
/// routes and reports it.
///
/// Threading: single-threaded and pure, like every entry in this library — see
/// the threading paragraph in the file preamble. A caller that wants the work
/// spread over threads divides its argument array into batches and calls this
/// entry from its own threads; distinct batches share nothing.
///
/// Workspace: the caller may supply one — BoysAllNWorkspaceSize(count)
/// std::size_t words — to keep a hot loop allocation-free; the default
/// nullptr allocates internally. The entry is total: if that allocation fails
/// it falls back to the per-argument path, at the same bound and without the
/// grouped-path speed-up.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param nmax      highest order, 0..kMaxBoysOrder
/// \param x         array of count arguments, each >= 0
/// \param out       receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count     number of arguments; may be 0 (no writes)
/// \param workspace grouping scratch of BoysAllNWorkspaceSize(count) words, or
///                  nullptr to allocate internally; its contents on return are
///                  unspecified
///
/// \pre x and out must hold count and count * (nmax + 1) values respectively,
///      and must not overlap
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace = nullptr) noexcept;

/// BoysAllN for arguments the caller states are already in non-decreasing
/// order — the sort is skipped rather than paid for.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      array of count arguments, non-decreasing, each >= 0
/// \param out    receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count  number of arguments; may be 0 (no writes)
///
/// \pre x[i - 1] <= x[i] for every i in [1, count) — the declaration this
///      overload exists for, and the caller's responsibility. The runs this
///      overload serves are re-derived from the classification rather than
///      taken from the declaration, so a caller that declared an order it did
///      not have gets the ungrouped path's performance, never a wrong value;
///      the assertion below reports the violation in a build with assertions
///      and this entry has no other error channel.
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllN(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;

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
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
float BoysSingleF32(int n, float x) noexcept;

/// F_0(x)..F_nmax(x) in single precision, |F̂ − F| ≤ m·1.5e-7 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept;

/// True if the CPU executes AVX2 with FMA; the SIMD entry points below require it.
///
/// The SIMD kernels are FMA chains, so AVX2 is gated together with the FMA
/// extension: a processor with AVX2 but without FMA reports false and stays on
/// the scalar lanes.
///
/// The AVX2 tier is x86_64-only by construction, so on any other target this
/// reports false unconditionally — the region entry points below are then
/// defined against the certified scalar lanes rather than being unavailable.
///
/// \returns true when AVX2 and FMA are available on the processor and exposed by the OS (OSXSAVE);
/// false on every non-x86_64 target
///
/// \ingroup boys
bool BoysAvx2Available() noexcept;


#if BoysFp16
/// F_n(x) in IEEE half precision — the fp16 lane of the certified
/// mixed-precision boundary (not a standalone fp16 API): fp16 I/O around
/// the certified fp32 engine. The argument is rounded to fp16 before
/// evaluation (the lane evaluates at the fp16 value), the computation is
/// the F32 lane's, and the result is the correctly rounded fp16 of it —
/// |result - F_n(x16)| <= m·1e-7 + one half-ULP of representation (the
/// representation term is m-independent). The bound is claimed only over the
/// arguments where |F_n(x16)| exceeds it, and no accuracy is claimed past that
/// ceiling: in f16 the return there is a subnormal number, then exactly zero,
/// with the bound met by the format's floor rather than by the lane — see the
/// contract table in the file preamble.
///
/// \tparam kAccuracyMultiplier see BoysSingle (forwards to the F32 engine)
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \returns     F_n(x) in fp16
///
/// \ingroup boys
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
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept;

/// F_n(x) in bfloat16 — the Bf16 lane of the certified mixed-precision
/// boundary, same
/// contract as BoysSingleF16 (bf16 representation term, 8-bit mantissa).
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \returns     F_n(x) in bf16
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
Bf16 BoysSingleBf16(int n, Bf16 x) noexcept;

/// F_0(x)..F_nmax(x) in bf16, same contract as BoysAllOrdersF16 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingleF16
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept;

/// The power of two the native half lane scales its results by: 15, so 2^15 —
/// the largest power of two binary16 holds, and an exact scale in it. A
/// caller recovers F_k(x) from the entry below with an exact power-of-two
/// division by 2^kHalfNativeScaleExponent.
inline constexpr int kHalfNativeScaleExponent = 15;

/// F_0(x)..F_nmax(x) for a packed pair of arguments, evaluated in packed half
/// arithmetic — the native half lane.
///
/// Not the fp16 lane above under another signature. BoysAllOrdersF16 rounds
/// once, around the certified fp32 engine; this entry *is* region C's
/// asymptotic ladder in binary16 — one square root and one divide for the
/// seed, then one packed multiply and one packed divide per order, one
/// rounding per operation, two arguments to a register, correctly rounded to
/// half throughout (half2.hpp). Its error is therefore the arithmetic's
/// rather than the engine's: about one ULP, an order of magnitude above the
/// I/O lane's budget, and it grows with the order because a ladder of 2n + 2
/// roundings has 2n + 2 chances to round.
///
/// **Bound.** With S_k = 2^kHalfNativeScaleExponent · F_k(x) the scaled value
/// this entry returns:
///
///     |out[k] − S_k| ≤ 8 ULP(out[k])
///
/// ULP being the binary16 quantum at the returned value, 2^(e−10) for a normal
/// half — the domain the bound is claimed over, below. The constant is the next
/// power of two above the measured maximum (4.243 ULP, at order 6, a swept
/// maximum rather than a proof bound), not a mean.
///
/// **Domain, and the ceiling that ends it.** The precondition is x at or above
/// the fp16 value of the region-C boundary (kX1 rounds to 28.984375 in
/// binary16); there is no fallback to another region. The claim is made over
/// the arguments whose returned value is a normal half, that is where
/// F_k(x) ≥ 2^-29: the smallest normal value 2^-14 lifted by the largest exact
/// scale. That is the whole of the format's argument range at orders 0 and 1, x
/// up to about 2590 at order 2, and x up to 359 / 128 / 30 at orders 3 / 4 / 8.
/// **Past those arguments the claim stops: no accuracy is claimed there.** The
/// entry returns a subnormal half and then exactly zero by design — at order 8
/// and x = 400, a zero — and a caller that has to be right at those orders and
/// arguments wants the double or float lane instead. The ceiling is not a
/// shortfall a better scale could move: 2^15 is the largest power of two
/// binary16 holds, and at order 8 the ladder's own span F_0/F_8 passes the
/// format's normal range at x ≈ 37.6, so the room any power-of-two scale works
/// in is spent by then. The scale itself costs nothing: it is exact, and
/// multiply and divide are exact under a power-of-two rescale.
///
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     packed pair of arguments, both >= the fp16 region-C boundary
/// \param out   receives nmax + 1 packed pairs, out[k] = 2^15 · (F_k(x.low),
///              F_k(x.high))
///
/// \ingroup boys
void BoysAllOrdersHalf2(int nmax, Half2 x, Half2* out) noexcept;

/// The native half lane over an argument array: region C's ladder in packed
/// half, two arguments to a register, same bound and range as
/// BoysAllOrdersHalf2 per value.
///
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      array of count arguments, each >= the fp16 region-C boundary
/// \param out    receives count * (nmax + 1) halves, out[k * count + i] =
///               2^15 F_k(x[i])
/// \param count  number of arguments; may be 0 (no writes)
///
/// \pre x and out must hold count and count * (nmax + 1) values respectively,
///      and must not overlap
///
/// \ingroup boys
void BoysAllNF16Native(int nmax, const F16* x, F16* out, std::size_t count) noexcept;

#endif // BoysFp16

} // namespace boys
