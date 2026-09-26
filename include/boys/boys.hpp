#pragma once

#include <cstddef>
#include <cstdint>
#include <span>

#if BoysFp16
#include "boys/f16.hpp"
#include "boys/half2.hpp"
#endif

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/boys_transform.hpp"
#include "boys/version.hpp"

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
/// with x0/x1 and the fit degrees placed against a 5e-14 target (double) /
/// 1e-7 (float) across n = 0..32 — validated against a 45-digit mpmath
/// reference grid, and reproduced by tools/gen_boys_coefficients.py. Those
/// targets are what the placement is chosen against, not what a caller
/// receives: the delivered bounds are the contract table's below, 5.5e-14 and
/// 1.5e-7, which carry the headroom the targets do not. The double lane's
/// measured region-C worst sits on its target rather than under it —
/// 5.0000e-14, order 32 at the region-C boundary x1, which is why the table
/// states 5.5e-14 and not 5e-14.
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
/// asserted per-region bounds above (measured). **m = 1 costs nothing extra and
/// matches the certified lanes bit for bit** — the default instantiations are
/// the full-accuracy bodies, and every existing call site compiles unchanged.
/// **Larger m trades a looser bound for less work**: the rung is a separate
/// instantiation of the same entry, cut to a degree the criterion certifies for
/// that multiplier, and the full degree is always admissible, so every rung is
/// served at every scheme. The contract and the work are **monotone in m**;
/// pointwise error is explicitly NOT guaranteed monotone — a larger m may
/// occasionally change a pointwise error, but never beyond the m·B_region
/// envelope. Region C has no relaxable resource; its contract holds with slack
/// at every m. Instantiations with kAccuracyMultiplier < 1.0 are compile-time
/// errors.
///
/// **How a rung is derived** — the degree criterion, the dropped-coefficient
/// tail it is spent on, the per-path seed-error amplification it is certified
/// against, and why that tail is the one the scheme's own summation reads — is
/// stated beside the machinery that applies it, in boys_effective_degrees.hpp.
/// The CUDA lane's relaxed path holds one degree table per process (filled once
/// per m, the InitializeTables thread-safety contract). The fp16/bf16
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

/// The stored fit an evaluated value came from.
///
/// A lane is not a region: the extended band and the per-order fits both serve
/// region A, and they are different data with different measured bounds. Naming
/// the lane is what lets a caller ask what one piece of the kernel promises
/// rather than what a region does.
///
/// \ingroup boys
enum class EvalLane : std::uint8_t {
    kRegionA = 0, ///< the per-order piecewise fits below kX0
    kRegionB, ///< the region-B seed F_0, on [kX0, kX1)
    kExtendedBand, ///< the extended-band seed F_0, on [kExtendedBX0, kX0)
};

/// What one evaluation scheme delivers on one stored fit.
///
/// The two delivered figures are measurements, not derivations: each is the
/// largest error the scheme was found to make over that fit's own interval,
/// swept against the reference, in the multiply-add route named. Neither is a
/// bound read off the fit's residual, and neither is inferred from the other
/// route's figure. Region C is evaluated by its asymptotic form and has no
/// stored fit, so it has no row here and is the same arithmetic under either
/// scheme.
///
/// \ingroup boys
struct EvalFitInfo {
    EvalScheme scheme = EvalScheme::kSplitClenshaw; ///< the scheme
    EvalLane lane = EvalLane::kRegionA; ///< the stored fit it sums
    AccuracyRegion region = AccuracyRegion::kA; ///< the region that fit serves
    int deg = 0; ///< the degree of the fit (the largest, where a lane is piecewise)
    int stored = 0; ///< coefficients stored per fit
    double fused = 0.0; ///< measured delivered bound, fused multiply-add route
    double separate = 0.0; ///< measured delivered bound, separate multiply-add route
};

/// What one evaluation scheme is, and what it promises.
///
/// A scheme is not a lane: the same scheme may be offered on some stored fits
/// and not others, so the promise is the worst of the fits it is offered on and
/// the arithmetic it was measured in is carried beside it rather than assumed.
///
/// \ingroup boys
struct EvalSchemeInfo {
    EvalScheme scheme = EvalScheme::kSplitClenshaw; ///< the scheme
    const char* name = ""; ///< the name a report prints it under
    int lanes = 0; ///< stored fits this scheme is offered on
    int deg = 0; ///< the largest degree it is offered at
    int stored = 0; ///< the largest coefficient count it stores
    double delivered = 0.0; ///< the worst measured bound over those fits, in the route in force
    backend::MulAddRoute route = backend::MulAddRoute::kFused; ///< the arithmetic \c delivered was measured in
};

/// The name a report prints a scheme under.
///
/// \param scheme the scheme
/// \returns a string literal naming it
///
/// \ingroup boys
const char* EvalSchemeName(EvalScheme scheme) noexcept;

/// The evaluation schemes this build carries, as a report prints them.
///
/// Which scheme can be named on an entry is the entries' own business; this
/// answers what exists and what each promises without a caller reading the
/// kernel.
///
/// \returns one row per scheme, in enumerator order
///
/// \ingroup boys
std::span<const EvalSchemeInfo> BoysEvalSchemes() noexcept;

/// The per-fit detail behind BoysEvalSchemes: one row per (scheme, stored fit).
///
/// \returns the rows, in scheme order then lane order
///
/// \ingroup boys
std::span<const EvalFitInfo> BoysEvalSchemeFits() noexcept;

/// The bound the named scheme was measured to deliver on the named stored fit,
/// in whatever multiply-add route this build is in force.
///
/// The route is reported rather than assumed, and it is a property of the build
/// rather than of the pair: the same call answers differently under the two
/// routes, so a caller that records this number should record the arithmetic
/// beside it (BoysBackends carries the same fact per lane).
///
/// \param scheme the scheme
/// \param lane   the stored fit
/// \returns the measured bound, or 0.0 for a pair this build does not carry
///
/// \ingroup boys
double BoysEvalSchemeDelivered(EvalScheme scheme, EvalLane lane) noexcept;

/// What one packing axis vectorises over, and what it promises.
///
/// The axis is a property of a call shape rather than of a kernel: a packed
/// lane keeps four doubles in a register and the call has to supply four of
/// something. The rows below are the two somethings this library packs, with
/// the interval each one's packed lane itself evaluates and the bar its values
/// are certified against. Outside that interval an entry carrying the axis runs
/// the certified scalar lanes, which is a defined answer inside the entry's own
/// bound rather than a value the packed lane produced.
///
/// \ingroup boys
struct PackAxisInfo {
    PackAxis axis = PackAxis::kArguments; ///< the selector value this row describes
    const char* name = ""; ///< the name a report prints it under
    int width = 0; ///< doubles one packed vector holds on this axis
    double lo = 0.0; ///< left edge of the interval the packed lane evaluates, inclusive
    double hi = 0.0; ///< right edge, exclusive
    double bound = 0.0; ///< the bar the packed lane's per-value error is certified against
};

/// The packing axes this build carries, as a report prints them.
///
/// Which axis an entry accepts is the entries' own business and is refused
/// where it is named; this answers what exists and what each member's packed
/// lane promises without a caller reading a kernel.
///
/// \returns one row per axis, in enumerator order
///
/// \ingroup boys
std::span<const PackAxisInfo> BoysPackAxes() noexcept;

/// What a tier delivers in one region, and what limits it when a tighter
/// error than that is asked for.
///
/// \ingroup boys
struct TierCoverage {
    bool meets = false; ///< the tier delivers an error at or below the request
    double reachable = 0.0; ///< the largest error the tier can deliver here
    AccuracyComponent limiting = AccuracyComponent::kRegionCAsymptotic; ///< the component that limits the tier when a tighter error is asked for
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

/// One certified fit route as a report states it.
///
/// The figures are the route's own rather than a lane's: a route supplies the
/// fits of one region, and what a caller receives from a lane entry is those
/// fits carried through the region's recurrence. \c stored is what the
/// evaluation reads; \c delivered is the worst error a sweep measured over
/// \c [lo, hi), in the arithmetic the kernel evaluates the fit in and against
/// the high-precision reference the fits themselves are validated against; and
/// \c bound is the bar the route is certified against, at or above
/// \c delivered. A delivered figure is a swept maximum and not a bound — the
/// two are stated apart for the reason the contract table's measured column and
/// published column are.
///
/// A route that serves more than one region has one row per region, so two rows
/// can carry the same \c route and differ in \c region, \c lo, \c hi, \c stored,
/// \c delivered and \c bound.
///
/// \c lo..hi is the domain of the route's fit, and \c servesFrom is the lowest
/// argument from which naming the route changes the values a caller receives.
/// The two are the same for every route whose selector takes over at its fit's
/// left edge, and they are stated apart because they can differ: a route whose
/// fit covers more than the selector hands it says so here, rather than
/// claiming a domain it does not serve. Region A's rational route is the case
/// that rule exists for, and its boundary is the lowest of a set rather than a
/// single one: each order is handed to the route from that order's own
/// argument, so a caller above \c servesFrom but below an order's own boundary
/// still receives the default route's value for that order.
///
/// \ingroup boys
struct FitRouteInfo {
    FitRoute route; ///< the selector value this row describes
    const char* name; ///< the name a report prints for the route
    AccuracyComponent component; ///< the component the route's fit supplies
    AccuracyRegion region; ///< the region it supplies it for
    double lo; ///< the fit's left edge, inclusive (region B's kX0)
    double hi; ///< the fit's right edge, exclusive (region B's kX1)
    double servesFrom; ///< the argument from which naming the route changes values
    int stored; ///< the route's stored coefficients over \c [lo, hi)
    double delivered; ///< the swept worst |F̂ − F| over \c [lo, hi)
    double bound; ///< the bar \c delivered is certified against
};

/// The certified fit routes this build carries, one row per route and region, so
/// a caller can ask what options exist, what each promises and over what
/// interval without reading this header's tables.
///
/// Every route in the fixed order below is served by this build: the routes are
/// table-driven and a build that carries the tables carries all of them, so a
/// row is never a promise this revision cannot keep.
///
/// \returns the routes, in a fixed order: \c kChebyshev over region A, then
///          \c kRationalMinimax over region A, then the same two over region B.
///
/// \ingroup boys
std::span<const FitRouteInfo> BoysFitRoutes() noexcept;

/// F_0(x)..F_nmax(x) in double precision at a run-time-selected fit route — the
/// batch entry's contract (the \c "double batch" row of the table in the file
/// preamble: |F̂ − F| ≤ 5.5e-14 per value in every region), with the named
/// route's fits serving the intervals they cover.
///
/// The route selects fits and changes nothing else. Outside the intervals a
/// route reports in BoysFitRoutes this entry runs the default route's own code
/// and returns the default entry's values bit for bit, so a caller who names a
/// route and a caller who does not are handed the same numbers wherever the
/// route does not reach. Inside them the named route's fits produce the values,
/// at the same bound, over the arguments the report's row says its selector
/// takes them over - and a route served per order takes over per order, so a
/// caller below an order's own boundary receives the default's value for it.
///
/// The multiplier is the reference one: this entry selects a fit, not a rung.
/// A caller that wants a relaxed budget wants \c BoysAllOrdersAtTier, whose
/// tiers are defined against the default route.
///
/// A route this build does not serve evaluates at the default route, the same
/// fallback BoysFitRoutes' table and the enumeration's contract describe.
///
/// \param route the fit route, a property of this call only
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x); every value is
///              written whatever route is named
///
/// \ingroup boys
void BoysAllOrdersWithRoute(FitRoute route, int nmax, double x, double* out) noexcept;

/// The same entry with the evaluation scheme named as well as the route: the
/// two axes of the compile-time selection, at run time.
///
/// The route and the scheme are the two fields of the policy the templated
/// entries carry (\c EvalPolicy), and they select different things: the route
/// names the fits that serve the regions, and the scheme names the summation
/// the Chebyshev family's coefficients are read in. A route whose own fit has
/// one stored form - the rational minimax family's monomial numerator and
/// denominator - evaluates that fit the same way under either scheme, and the
/// scheme reaches the parts of the call that route's fits do not serve. So
/// naming a scheme changes the values only outside the served intervals a
/// route's rows report, where the route has already handed the argument back to
/// the shipped family, and inside them the route's own fits answer at the bar
/// its row states. A scheme outside the enumeration is the reference scheme's
/// body, the same fallback \c BoysAllOrdersAtTier takes for a scheme it does
/// not carry.
///
/// \param route   the fit route
/// \param scheme  the evaluation scheme
/// \param nmax    highest order, 0..kMaxBoysOrder
/// \param x       argument, >= 0
/// \param out     receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
void BoysAllOrdersWithRoute(
    FitRoute route, EvalScheme scheme, int nmax, double x, double* out) noexcept;

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

/// The same batch entry with the evaluation scheme named as well as the tier.
///
/// The two selectors are different kinds of thing: a tier is a run-time value
/// and a scheme is a compile-time one, so this overload carries the tier and
/// names the scheme for the rung it dispatches to. A scheme this build does
/// not carry is the reference scheme's body, the same fallback a tier this
/// build does not serve takes.
///
/// \param tier   the tier
/// \param scheme the evaluation scheme
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      argument, >= 0
/// \param out    receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
void BoysAllOrdersAtTier(
    AccuracyTier tier, EvalScheme scheme, int nmax, double x, double* out) noexcept;

/// The same batch entry with the fit route named as well as the tier and the
/// scheme: every axis of the compile-time selection, at run time.
///
/// The rung and the route are two selectors of two different things, and until
/// this overload there was no way to name both: a rung truncates the route's
/// own fits to the degrees its criterion certifies, and the criterion reads the
/// table the route evaluates. So a caller that wants the rational route at a
/// relaxed budget wants this entry, and a caller that names only a tier gets
/// the default route's rung, which is what \c BoysAllOrdersAtTier has always
/// answered.
///
/// The rung is a property of the route rather than of the multiplier: the
/// Chebyshev family's rung is a cut of its stored coefficients, and the
/// rational family's is a cut of its stored numerator and denominator pair. The
/// two cuts are derived by the same criterion and neither is a value the other
/// route's table can express, so the pair is a combination this library serves
/// rather than one of its axes alone.
///
/// \param tier   the tier
/// \param route  the fit route
/// \param scheme the evaluation scheme
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      argument, >= 0
/// \param out    receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
void BoysAllOrdersAtTier(
    AccuracyTier tier, FitRoute route, EvalScheme scheme, int nmax, double x, double* out) noexcept;

/// The same entry with the reference scheme named for the route; see the
/// scheme-carrying overload above.
///
/// \param tier  the tier
/// \param route the fit route
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
void BoysAllOrdersAtTier(
    AccuracyTier tier, FitRoute route, int nmax, double x, double* out) noexcept;

/// F_n(x) in double precision at a run-time-selected tier, and optionally a
/// run-time-selected route and scheme; the single-order entry's contract — the
/// \c "double single" rows of the table in the file preamble, |F̂ − F| ≤
/// m·B_region per region with m = AccuracyMultiplier(tier).
///
/// The batch entry's rung is a run-time choice already; this is the same choice
/// on the single-order shape, and it exists because the shape is a different
/// call: an integral engine that reads one order at a time cannot reach the
/// rung through an entry that computes every order. The route and the scheme
/// are named here for the same reason they are named on the batch entry — a
/// rung is a cut of the named route's own fits, so a caller who wants the
/// rational route's rung has to be able to say both.
///
/// One branch selects the rung, then the rung's own body runs, exactly as in
/// the templated entry: the values are those of \c BoysSingle<m, Policy> at the
/// multiplier the tier names, bit for bit, and the reference tier is the
/// template default. A tier or a route this build does not serve is the
/// fallback the rest of the surface takes for it, so a caller who records the
/// multiplier \c AccuracyMultiplier reported beside these values is recording
/// the accuracy they were computed at.
///
/// \param tier   the tier
/// \param route  the fit route; the default route by default
/// \param scheme the evaluation scheme; the reference scheme by default
/// \param n      order, 0..kMaxBoysOrder
/// \param x      argument, >= 0
/// \returns      F_n(x)
///
/// \ingroup boys
double BoysSingleAtTier(AccuracyTier tier, FitRoute route, EvalScheme scheme, int n,
                        double x) noexcept;

/// The same entry with the reference scheme named for the route; see the
/// scheme-carrying overload above.
///
/// \param tier  the tier
/// \param route the fit route
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
double BoysSingleAtTier(AccuracyTier tier, FitRoute route, int n, double x) noexcept;

/// The same entry with the default route and a named scheme.
///
/// \param tier   the tier
/// \param scheme the evaluation scheme
/// \param n      order, 0..kMaxBoysOrder
/// \param x      argument, >= 0
/// \returns      F_n(x)
///
/// \ingroup boys
double BoysSingleAtTier(AccuracyTier tier, EvalScheme scheme, int n, double x) noexcept;

/// F_n(x) at a run-time-selected tier, on the default route and scheme.
///
/// \param tier the tier
/// \param n    order, 0..kMaxBoysOrder
/// \param x    argument, >= 0
/// \returns    F_n(x)
///
/// \ingroup boys
double BoysSingleAtTier(AccuracyTier tier, int n, double x) noexcept;

/// F_n(x) in double precision, |F̂ − F| ≤ m·B_region per region (contract
/// table in the file preamble; m = kAccuracyMultiplier).
///
/// \tparam kAccuracyMultiplier the accuracy multiplier m >= 1.0; 1.0 = full
///         static accuracy, bit-identical to the certified lane; relaxes the
///         per-region bound to m·B_region (compile-time degree truncation,
///         monotone in m). The rung is a cut of the named route's own fits: the
///         Chebyshev route's stored coefficients are cut to the degree table of
///         boys_effective_degrees.hpp, certified against the table the policy's
///         scheme sums, and the rational route's stored numerator and
///         denominator pair is cut by the same criterion, which reads a pair's
///         dropped orders as the two terms a quotient's perturbation has rather
///         than as one coefficient sum. Both routes are carried, at either
///         scheme; the entries of this lane that reach their values through
///         the shipped fits' own path rather than through the policy are the
///         ones that refuse the rational route, and they say so where the call
///         is named
/// \tparam Policy the evaluation policy (\c EvalPolicy): the fit route, the
///         scheme its coefficients are summed in, and a single-precision
///         engine's budget, selected together. The default is the Chebyshev
///         route summed by the split Clenshaw recurrence, so a call site that
///         names neither axis compiles the certified route's code path
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
double BoysSingle(int n, double x) noexcept;

/// F_0(x)..F_nmax(x) in double precision, |F̂ − F| ≤ m·B_region per value.
///
/// The batch is the pattern real integral engines use (McMurchie-Davidson /
/// Obara-Saika recursions consume all orders at once) and is considerably
/// cheaper than nmax + 1 single evaluations.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingle
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllOrders(int nmax, double x, double* out) noexcept;

/// F_n(x_i) for an array of arguments at one fixed order n, double
/// precision; |F_hat - F| <= m*B_region per value (the BoysSingle
/// per-region contract, region table in the file preamble).
///
/// The fixed-n vector entry is the batch shape of integral-engine inner
/// loops that group shell pairs by angular momentum: each element needs
/// exactly one order, so no unused cross-order recursion is paid. Each
/// output element returns BoysSingle<kAccuracyMultiplier>'s value at the
/// same (n, x) and carries the single lane's per-region bound with it: the
/// m = 1 path runs the certified scalar single-lane region bodies
/// verbatim, the relaxed path the same bodies at the single-lane effective
/// degrees. The two agree bit for bit on a build that does not contract a
/// bare product-plus-add, which is what x86-64 without -mfma and MSVC
/// everywhere deliver. A build that does contract one decides per call site
/// whether to fuse that form, so an element can differ from the single
/// entry's in the last place and still be inside the bound; what does not
/// move is the bound. Arguments need no pre-partitioning: the region
/// dispatch is per element, the portable shape. The vector tier is reached
/// through the batch entries (BoysAllN), which group the arguments once for
/// the whole call.
///
/// Layout: out[i * stride] = F_n(x[i]), i = 0..count-1; stride is measured
/// in doubles and defaults to 1 (contiguous). The output span must hold
/// (count - 1) * stride + 1 doubles; count may be 0 (no writes). The entry
/// is scalar and portable: x and out need no alignment beyond
/// alignof(double), and the two arrays must not overlap.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingle. The fixed-order entry carries the fit route:
///         a call naming a route other than the shipped one is answered by the
///         per-argument single entry, once per argument, which is the body this
///         entry's own m = 1 path already mirrors region for region - so on that
///         route the entry runs the single entry's own body rather than a second
///         copy of it. The shaped path below stays the shipped route's, which is
///         the path the entry's own paragraph above is about
///
///         The packing axis is not an axis of this shape, and that is the
///         entry's signature rather than a body nobody built. A packed lane
///         keeps four doubles in a register, and this entry produces ONE order
///         at every argument of the array: there are not four orders here to
///         fill a lane with, and the wide dimension the call does have - count
///         - is a different axis, served by the vector tier the batch entries
///         reach. Naming PackAxis::kOrders is therefore rejected at the call
///         site, which is what the assertion says
/// \param n      order, 0..kMaxBoysOrder - the batch's single fixed order
/// \param x      array of count arguments, each >= 0
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in doubles, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
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
/// \tparam Policy see BoysSingle. The many-argument entry carries the fit route
///         as well as the packing axis, and by the same shape: the route is a
///         property of the fit a value is read from, so a call naming the
///         rational route takes the entry's per-argument path, whose body is
///         the all-orders entry's own and takes its fit from the policy. What
///         the partitioned shape does not carry is the route and not the
///         value - both shapes answer inside this entry's own bound, and the
///         per-argument path is the one that reads each order from its own fit
///         where the partitioned shape reaches most orders by a recursion
///
///         The packing axis is carried here too, and the orders axis is the
///         shape this entry's layout already has: out[k * count + i] is F_k of
///         one argument, so the entry's per-argument path is its body, and the
///         packed lane that fills a register with four orders of that argument
///         is the all-orders entry's. Naming the axis trades the region
///         grouping for it - the grouping exists to feed a lane that packs four
///         arguments, which an orders-axis call has no use for - so a call
///         naming it takes the per-argument path and that axis's own lane. The
///         axis carries every rung the tier enumeration declares and either
///         route, at m·B_region as this entry does
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
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace = nullptr) noexcept;

/// BoysAllN for arguments the caller states are already in non-decreasing
/// order — the sort is skipped rather than paid for.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingle. The many-argument entry carries the fit route
///         as well as the packing axis, and by the same shape: the route is a
///         property of the fit a value is read from, so a call naming the
///         rational route takes the entry's per-argument path, whose body is
///         the all-orders entry's own and takes its fit from the policy. What
///         the partitioned shape does not carry is the route and not the
///         value - both shapes answer inside this entry's own bound, and the
///         per-argument path is the one that reads each order from its own fit
///         where the partitioned shape reaches most orders by a recursion
///
///         The packing axis is carried here too, and the orders axis is the
///         shape this entry's layout already has: out[k * count + i] is F_k of
///         one argument, so the entry's per-argument path is its body, and the
///         packed lane that fills a register with four orders of that argument
///         is the all-orders entry's. Naming the axis trades the region
///         grouping for it - the grouping exists to feed a lane that packs four
///         arguments, which an orders-axis call has no use for - so a call
///         naming it takes the per-argument path and that axis's own lane. The
///         axis carries every rung the tier enumeration declares and either
///         route, at m·B_region as this entry does
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
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllN(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;

/// F_0(x_i)..F_n[i](x_i) for an array of arguments, double precision — every
/// order up to each argument's OWN top order, the tops arriving as an array.
///
/// This is the shape a shell-quartet consumer actually has. A quartet carries
/// its own highest order, so a batch of quartets is a batch of differing tops,
/// and BoysAllN — one common nmax for the whole batch — makes such a caller
/// choose between padding every quartet up to the batch's largest top order,
/// paying for the orders nobody asked for, and calling the library once per
/// quartet. This entry takes the tops as an array, so the caller does neither.
/// It is the CPU spelling of the device lane's per-element-order batch
/// (BoysCuda::AllOrdersF64), which is what lets one kernel be written against
/// both lanes.
///
/// Layout: order-major planes as BoysAllN — out[k * count + i] = F_k(x[i]) —
/// with each column stopping at its own order. Every cell of argument i's
/// column at or below n[i] is written, and **every cell above it is left
/// untouched**: out[k * count + i] for k > n[i] keeps whatever the caller put
/// there, so a caller may pre-fill those cells with the value its own
/// contraction wants to multiply by (a zero, or an earlier group's result) and
/// know it survives the call. Writing only the orders that were asked for is
/// the entry's point rather than a saving inside the recursion: a plane above
/// an argument's own top order has no reader.
///
/// The output holds count * (nmax + 1) doubles, nmax being the largest of the
/// n[i]. That is the caller's own array, so the caller sizes the buffer from
/// what it already has, and no padding of either the arguments or the output is
/// involved.
///
/// Accuracy: every returned value satisfies the double batch lane's documented
/// per-region bound, |F̂ − F| ≤ m·5.5e-14 (the contract table in the file
/// preamble). Each column is the per-argument all-orders body run at that
/// argument's own top order: out[k * count + i] is the value, bit for bit, that
/// BoysAllOrders(n[i], x[i], out) returns at out[k].
///
/// Threading: single-threaded and pure, like every entry in this library — see
/// the threading paragraph in the file preamble. A caller that wants the work
/// spread over threads divides its argument array and calls this entry from its
/// own threads; distinct batches share nothing.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingle. This entry runs the per-argument all-orders
///         body, so it takes the same policies that entry takes
/// \param n     array of count top orders, each 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives the planes: out[k * count + i] = F_k(x[i]) for
///              k = 0..n[i], every cell above n[i] untouched
/// \param count number of arguments; may be 0 (no writes)
///
/// \pre x must hold count values and n count orders; out must hold
///      count * (1 + max_i n[i]) doubles when count > 0; x and out must not
///      overlap
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept;

/// Whether the packed region-A lane serves a call whose policy names this
/// scheme.
///
/// The many-argument entries hand their low-order region-A runs to the packed
/// AVX2 lane where the build has one. That lane holds the shipped Chebyshev
/// coefficients and the split Clenshaw recurrence, so it serves the shipped
/// scheme and no other: a call naming another scheme is answered on the scalar
/// body, at the same bound and with the same values the per-argument entry
/// returns. What is not offered is the lane, not the value - and a lane is a
/// performance property, so nothing in the returned values can show which one
/// ran.
///
/// This is where a caller asks which it got, rather than inferring it from a
/// measurement of their own. It is the packed lane's scope and not the
/// entry's: the scheme is served on every entry that takes a policy, and the
/// packed lane is an implementation of region A that serves one of them.
///
/// \param scheme the evaluation scheme a call names
///
/// \returns true if the packed region-A lane serves that scheme
///
/// \ingroup boys
constexpr bool BoysPackedLaneServes(EvalScheme scheme) noexcept
{
    return scheme == kDefaultEvalScheme;
}

/// F_n(x) in single precision, |F̂ − F| ≤ m·1.5e-7.
///
/// Same piecewise structure as the double lane with per-order float fits.
/// This is the recommended lane for GPU integral evaluation.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingle. At the reference multiplier this lane reads
///         the policy's fit route and its evaluation scheme: the route selects
///         which family supplies the lane's own fits - the shipped Chebyshev
///         table or the rational minimax one - and the scheme selects which of
///         the Chebyshev family's two parallel tables, the Chebyshev form or the
///         monomial form of the same fits, is summed. The budget selects
///         nothing there. From \c m = 2 upward the lane serves the shipped
///         route and scheme alone and the budget is the axis it reads, picking
///         the degree table the truncation targets - which is what tells the
///         float lane's 1.5e-7 apart from the half lanes' 1e-7.
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
float BoysSingleF32(int n, float x) noexcept;

/// F_0(x)..F_nmax(x) in single precision, |F̂ − F| ≤ m·1.5e-7 per value.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingleF32: at the reference multiplier the route and
///         the scheme select the fits - the route for this lane's region-B seed
///         and for the double lane's fit that seeds region A - and the budget
///         selects nothing
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept;

/// F_0(x_i)..F_nmax(x_i) for an array of arguments, single precision — the
/// float lane's entry of the shape BoysAllN has in the double lane.
///
/// Layout and totality as BoysAllN: out[k * count + i] = F_k(x[i]), order-major
/// planes, the output holding count * (nmax + 1) floats. The same shape exists on
/// the device (BoysCuda::AllNF32), and before this entry it existed on the CPU
/// only as a loop a caller wrote for itself — which is what the C surface's
/// BoysFloatBatch was — so a consumer porting a batch between the lanes had a
/// call on one side and its own loop on the other.
///
/// What it is not is a grouping entry. The packed region-A lane the double batch
/// hands its low-order runs to is an AVX2 kernel over doubles; the float lane has
/// no packed region kernel of its own, so there is no run for this entry to
/// group and it evaluates the per-argument all-orders body at each argument. The
/// entry is the shape, and no speed is claimed for it over the same loop written
/// at the call site.
///
/// Accuracy: |F̂ − F| ≤ m·1.5e-7 per value, the float lane's bound, which is the
/// same in every region (the contract table in the file preamble) — the bound
/// BoysAllOrdersF32 meets, and each column is that entry's value at the same
/// (nmax, x[i]) bit for bit.
///
/// Threading: single-threaded and pure, like every entry in this library — see
/// the threading paragraph in the file preamble. A caller that wants the work
/// spread over threads divides its argument array and calls this entry from its
/// own threads; distinct batches share nothing.
///
/// \tparam kAccuracyMultiplier see BoysSingle
/// \tparam Policy see BoysSingleF32: at the reference multiplier the route and
///         the scheme select the fits, and the budget selects nothing
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives count * (nmax + 1) floats, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \pre x must hold count values, out count * (nmax + 1), and the two must not
///      overlap
///
/// \ingroup boys
template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier,
          EvalPolicyLike Policy = EvalPolicy<>>
void BoysAllNF32(int nmax, const float* x, float* out, std::size_t count) noexcept;

/// F_n(x) in single precision at a run-time-selected fit route — the
/// single-precision single entry's contract (|F̂ − F| ≤ 1.5e-7), with the
/// named route's fits serving the intervals they cover.
///
/// This is \c BoysSingleF32 with one thing changed: which fit supplies the
/// lane's region-A seed and its region-B seed. The route is a property of the
/// float lane's own tables, so it is offered on the entry that reads them, at
/// run time and without a scheme. A caller who templates on a policy names the
/// same route there, and \c BoysAllOrdersF32 reads it for the double lane's
/// fit that seeds its region-A recursion.
///
/// The route selects fits and changes nothing else. Region C holds no
/// coefficient under either route, and an argument there is the default
/// entry's value bit for bit. Outside the intervals the route reports in
/// \c BoysFitRoutesF32 the same holds, so a caller who names a route and a
/// caller who does not are handed the same numbers wherever the route does
/// not reach.
///
/// The two routes are alternatives and not rungs: the rational route holds
/// half the stored coefficients over region A and delivers more error than
/// the Chebyshev route at the same bar, so which of the two is cheaper is a
/// property of the caller's machine rather than of the tables. Both are
/// certified against the lane's own bound.
///
/// The multiplier is the reference one: this entry selects a fit, not a rung.
///
/// A route this build does not serve evaluates at the default route, the same
/// fallback the \c FitRoute enumeration's contract describes.
///
/// \param route the fit route, a property of this call only
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
float BoysSingleF32WithRoute(FitRoute route, int n, float x) noexcept;

/// The certified fit routes this build carries for the single-precision lane,
/// one row per route and region, so a caller can ask what options exist, what
/// each promises and over what interval without reading this header's tables.
///
/// The rows are the float lane's own and not the double lane's: that lane's
/// region-A table is its own and its route covers the whole of region A,
/// where the double lane's rational route takes over from a boundary above
/// zero. Every route in the fixed order below is served by this build.
///
/// \returns the routes, in a fixed order: \c kChebyshev over region A, then
///          \c kRationalMinimax over region A, then the same two over region B.
///
/// \ingroup boys
std::span<const FitRouteInfo> BoysFitRoutesF32() noexcept;

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

/// \cond
// Hidden from the API reference: each of these is an instantiation of the
// entries declared above at the default multiplier and the default policy, not
// an entry of its own. They are what the default call sites link against
// instead of compiling the kernel again in their own translation unit; a
// caller that names another multiplier, or another policy, compiles the rung
// or the route it asks for from the definition below.
extern template double BoysSingle<kBoysFullAccuracyMultiplier>(int n, double x) noexcept;
extern template void BoysAllOrders<kBoysFullAccuracyMultiplier>(
    int nmax, double x, double* out) noexcept;
extern template void BoysFixedN<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept;
extern template void BoysAllN<kBoysFullAccuracyMultiplier>(
    int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept;
extern template void BoysAllN<kBoysFullAccuracyMultiplier>(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;
extern template void BoysAllNAtOrders<kBoysFullAccuracyMultiplier>(
    const int* n, const double* x, double* out, std::size_t count) noexcept;
extern template float BoysSingleF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(
    int n, float x) noexcept;
extern template void BoysAllOrdersF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(
    int nmax, float x, float* out) noexcept;
extern template void BoysAllNF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(
    int nmax, const float* x, float* out, std::size_t count) noexcept;

#if BoysFp16
extern template F16 BoysSingleF16<kBoysFullAccuracyMultiplier>(int n, F16 x) noexcept;
extern template void BoysAllOrdersF16<kBoysFullAccuracyMultiplier>(
    int nmax, F16 x, F16* out) noexcept;
extern template Bf16 BoysSingleBf16<kBoysFullAccuracyMultiplier>(int n, Bf16 x) noexcept;
extern template void BoysAllOrdersBf16<kBoysFullAccuracyMultiplier>(
    int nmax, Bf16 x, Bf16* out) noexcept;
#endif // BoysFp16
/// \endcond

} // namespace boys

// The kernel behind the entries above, shipped as a header so that every
// multiplier a caller names is instantiable at the call site. The reference
// documents the entries; this is their implementation. It is included here
// rather than at the top of this file because its definitions name the
// declarations above.
#include "boys/boys_impl.hpp"

// The option probe closes the surface: it measures the entries above on the
// machine it is called on. Included last, because it is written against every
// declaration in this file.
#include "boys/boys_probe.hpp"
