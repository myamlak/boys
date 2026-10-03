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
/// evaluations of this family ([Boys1950]); the standard upward and downward
/// recursions between orders, and the large-argument asymptotic form region C
/// uses, are from Shavitt ([Shavitt1963]).
///
/// Design (measured, not assumed):
///  - region A  [0, x0): per-order piecewise Chebyshev fits evaluated by a
///    split Clenshaw recurrence (division-free, FMA-friendly), seeded from
///    F_n and recursed downward for batches;
///  - region B  [x0, x1): a single F0 fit plus upward recursion;
///  - region C  [x1, inf): the asymptotic form 1/2 sqrt(pi/x) and upward
///    recursion;
/// with x0/x1 and the fit degrees placed against a 5e-14 target (double) /
/// 1e-7 (float) across n = 0..32, validated against a 45-digit mpmath reference
/// grid and reproduced by tools/gen_boys_coefficients.py. A target is what the
/// placement is chosen against, not what a caller receives: the delivered bounds
/// are the contract table's below, 5.5e-14 and 1.5e-7, which carry headroom the
/// targets do not. The double lane's measured region-C worst sits on its target
/// rather than under it — 5.0000e-14, order 32 at the region-C boundary x1,
/// which is why the table states 5.5e-14 and not 5e-14.
///
/// The lane split follows the hardware: consumer GPUs run double precision at
/// 1/32 of single-precision throughput (measured on a Quadro T1000: the float
/// kernel is 8.4x faster than the fastest double-precision table kernel and
/// 5.2x faster than the fastest double kernel overall), so callers that tolerate
/// the certified 1.5e-7 absolute error should prefer the F32 lane there.
///
/// **Threading.** Every entry is single-threaded and a pure function of its
/// arguments: it spawns no threads and reads no shared mutable state (the CUDA
/// lane's device tables are the one exception, and its InitializeTables warm-up
/// is the documented contract there). Two calls with the same arguments return
/// the same values whatever else the process is doing, and a call made from
/// inside a caller's own parallel region neither nests nor waits on any lock.
/// There is no threaded entry by design: a caller that wants the work spread over
/// threads divides its argument array into batches and calls the batch entry
/// from its own threads, since the per-argument value depends only on that
/// argument and nmax; distinct batches share nothing and may be called
/// concurrently.
///
/// Behind the BoysFp16 build-time seam (default ON) the fp16 lane extends the
/// certified mixed-precision boundary: F16/Bf16 inputs and outputs around the
/// certified fp32 engine, so the lane delivers the F32 lane's certified bound
/// (1.5e-7 absolute) up to one half-ULP of representation (in fact the fp16
/// roles run the engine at the tighter 1e-7 region budgets). F16/Bf16 alias
/// std::float16_t / std::bfloat16_t where the toolchain ships them (GCC 13+,
/// Clang 17+); the MSVC STL does not (see f16.hpp), so this library supplies the
/// self-contained wrappers there. The F16/Bf16 region kernels are AVX2-only and
/// follow the same region-partitioned engine pattern as the F64 lanes above.
///
/// The native half lane (BoysAllOrdersHalf2, BoysAllNF16Native, half2.hpp) is a
/// third thing again: not I/O around an engine but region C's ladder itself in
/// packed binary16 — a half2.hpp operation per step, correctly rounded to half,
/// two arguments to a register. It has no region but region C (no table of
/// regions A and B is representable in half), no stored fit (region C's
/// asymptotic form has none), and a bound of its own, an order of magnitude
/// looser than the fp16 I/O lane's: the I/O lane rounds once per value, this
/// one once per operation. It has no tensor-core relation: the
/// ladder is a scalar recurrence, with no matrix product for a tensor core to
/// take.
///
/// **Accuracy contract.** Every lane entry evaluates at the one accuracy this
/// library carries, the full static accuracy of the certified lane.
/// The contract per lane and region:
///
/// | Lane | region A | extended band | region B (x0 <= x < x1) | region C (x >= x1) |
/// |---|---|---|---|---|
/// | double single | \|F̂ − F\| ≤ 1e-15 | ≤ 3e-14 | ≤ 3e-14 | ≤ 5.5e-14 |
/// | double batch | ≤ 5.5e-14 | ≤ 5.5e-14 | ≤ 5.5e-14 | ≤ 5.5e-14 |
/// | float single / batch | ≤ 1.5e-7 | ≤ 1.5e-7 | ≤ 1.5e-7 | ≤ 1.5e-7 |
/// | fp16 / bf16 | ≤ 1.5e-7 + ½ULP | ≤ 1.5e-7 + ½ULP | ≤ 1.5e-7 + ½ULP | ≤ 1.5e-7 + ½ULP |
/// | native half | — | — | — | ≤ 8 ULP of the returned value |
///
/// These are the figures at exact division and at the refined reciprocal. The
/// plain reciprocal rounds once more per step, and on the two single-precision
/// lanes that costs accuracy: those lanes publish 2.5e-7 for it, which is
/// 1.5e-7 plus the term their \c BoysLaneContracts() row carries. Name the form
/// and \c BoysAccuracyGuaranteed answers the figure for it.
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
/// i.e. |F̂_n(x) − F_n(x)| ≤ B_region per lane and region, with B_region the
/// asserted per-region bounds above (measured). **These are the certified
/// lanes' figures and every entry is bit-identical to them.** A lane whose
/// bound is not one number carries the difference as its own additive term
/// beside the base rather than inside it: the bound is the base plus that term.
/// Region C has no stored fit; its contract holds with slack.
///
/// The fp16/bf16 base is
/// the float lane's own figure — the half lanes run that lane's arithmetic and
/// store what it returns — plus the half-ULP term the store adds. The suite
/// asserts them at the tighter 1e-7 the region targets are placed against, so a
/// pass there is a stronger result than the published figure requires.
///
/// **The fp16/bf16 bound is a claim only where the value exceeds it.** With u
/// the format's quantum at the returned value, the half lanes bind where
/// |F_n(x)| > 1.5e-7 + ½u — there the return is within that distance of the
/// value — and over no other arguments, and **no accuracy is claimed** past
/// that ceiling: a caller that needs |F_n(x)| at or below 1.5e-7 wants the
/// double lane, or the float lane with the half format's exponent range in
/// mind, since below the ceiling it is the format's floor that answers and not
/// the arithmetic. What the return is past
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

/// The three evaluation regions of the kernel (contract table in the file
/// preamble).
///
/// \ingroup boys
enum class AccuracyRegion : int {
    kA = 0, ///< x < x0: per-order Chebyshev fits
    kB, ///< x0 <= x < x1: F0 fit plus upward recursion
    kC, ///< x >= x1: asymptotic plus upward recursion
};

/// The part of an evaluation that fixes its accuracy, so a request that cannot
/// be met can name what stops it.
///
/// \ingroup boys
enum class AccuracyComponent : int {
    kRegionASeed = 0,
    kRegionBFit,
    kRegionCAsymptotic,
};

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
/// The two delivered figures are measurements, not derivations: each is the largest
/// error the scheme was found to make over that fit's own interval, swept against
/// the reference, in the multiply-add route named. Region C is evaluated by its
/// asymptotic form and has no stored fit, so it has no row here and is the same
/// arithmetic under either scheme.
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

/// The name a report prints a scheme under, and never null.
///
/// \param scheme the scheme
/// \returns a string literal naming it: "split-clenshaw" or "horner", and
///          "unknown" for a value outside the enumerators
///
/// A value outside the enumerators is answered rather than refused. This
/// function names a scheme; the schemes this build carries, and the fits each
/// reaches, are BoysEvalSchemes and BoysEvalSchemeFits, so a caller that needs
/// the distinction asks those rather than reading this string.
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
/// The axis is a property of a call shape rather than of a kernel: a packed lane
/// keeps four doubles in a register and the call has to supply four of something.
/// The rows below are the two somethings this library packs, with the interval each
/// one's packed lane itself evaluates and the bar its values are certified against.
/// Outside that interval an entry carrying the axis runs the certified scalar lanes,
/// which is a defined answer inside the entry's own bound rather than a value the
/// packed lane produced.
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

/// What one division form is, as a report names it.
///
/// The form is how every recursive step of an evaluation ends, so it is a property
/// of the arithmetic and not of a call shape: a row states which spelling of the
/// per-order division it is, and every entry this build carries runs every one of
/// them. That is why these rows carry no coverage fields where the packing axis and
/// the partition rows do - there is no cell this axis is refused on - and what each
/// member costs on a given host is what the option probe measures there.
///
/// \ingroup boys
struct DivisionFormInfo {
    DivisionForm form = DivisionForm::kExactDivision; ///< the selector value this row describes
    const char* name = ""; ///< the name a report prints it under
};

/// The division forms this build carries, as a report prints them.
///
/// Which form a call runs is a template argument of its policy, so a caller choosing
/// one names an enumerator; this answers which enumerators this build has, in the
/// library's own spelling, without a caller writing the list out again.
///
/// \returns one row per form, in enumerator order
///
/// \ingroup boys
std::span<const DivisionFormInfo> BoysDivisionForms() noexcept;

/// What one region-B exponential is, as a report names it.
///
/// The member is the arithmetic a region-B ladder's seed is evaluated in, so it is
/// a property of the arithmetic and not of a call shape: a row states which of the
/// two arithmetics it is, and every entry that has a region-B ladder runs both of
/// them. That is why these rows carry no coverage fields where the packing axis and
/// the partition rows do - there is no cell this axis is refused on - and what each
/// member costs on a given host is what the option probe measures there.
///
/// \ingroup boys
struct RegionBExpInfo {
    RegionBExp exp = RegionBExp::kAccurate; ///< the selector value this row describes
    const char* name = ""; ///< the name a report prints it under
};

/// The region-B exponentials this build carries, as a report prints them.
///
/// Which member a call runs is a template argument of its policy, so a caller
/// choosing one names an enumerator; this answers which enumerators this build has,
/// in the library's own spelling, without a caller writing the list out again.
///
/// \returns one row per member, in enumerator order
///
/// \ingroup boys
std::span<const RegionBExpInfo> BoysRegionBExps() noexcept;

/// One partition of the fitted regions, and what this build promises about it.
///
/// The partition is how narrowly the fitted domain is cut into pieces, and it is
/// a property of the stored tables rather than of a call shape: a row states
/// what the partition's tables hold — pieces, degree and coefficients per fitted
/// region — and which packing axes the partition has kernels for. A combination
/// the row does not cover is refused where it is named rather than
/// answered from another partition's tables, and the row is where those refusals
/// can be counted from instead of being discovered one compile at a time.
///
/// Every row describes a table the double lane's entries read. The uniform row
/// describes a grid rather than a region's cut: its region-A fields are that
/// grid's intervals, the one degree every order of it is stored at, and the
/// coefficients it stores, and its region-B fields are zero because the grid is
/// one table over the whole of the fitted domain rather than a region-A table
/// beside a region-B seed. The single-precision lanes hold one
/// coefficient set each, so they take no partition and naming one is refused
/// where it is named.
///
/// \c delivered is the worst figure the partition's certification measured over
/// its pieces, and \c bound is the figure its tables are certified against — for
/// the shipped partition the bar its fits are cut at, for the narrow one the
/// per-piece round-up its certification publishes, and for the uniform grid the
/// round-up its own rows publish over the whole table. Both are figures for the
/// stored fits, measured where a fit is read directly, and they are stated apart
/// because a figure a sweep found is not the figure a caller may rely on.
///
/// \c lo and \c hi are the interval those fits cover: \c bound holds on
/// \c [lo, hi) and says nothing about any argument outside it.
/// Naming a partition replaces the fitted tables of that interval and leaves the
/// rest of the domain to what the entry does without them, so this is not the
/// figure for an entry's error: the entry that reads a partition carries its own
/// documented accuracy, and where the values of two partitions are compared the
/// difference is bounded by the two fits' figures together, one of them the
/// other partition's. Above \c hi the same asymptotic arithmetic runs under
/// either partition, so a figure covering the whole line is the entry's and
/// never this one.
///
/// \ingroup boys
struct FitGranularityInfo {
    FitGranularity granularity = FitGranularity::kCoarsest; ///< the selector value this row describes
    const char* name = ""; ///< the name a report prints it under
    unsigned routes = 0; ///< bit (1u << route) set per fit route the partition's tables hold
    unsigned axes = 0; ///< bit (1u << axis) set per packing axis the partition has a kernel for
    int regionAPieces = 0; ///< pieces region A's per-order tables are cut into
    int regionADeg = 0; ///< highest degree an evaluation of a region-A piece reads
    int regionAStored = 0; ///< coefficients region A's tables store
    int regionBPieces = 0; ///< pieces region B's seed is cut into
    int regionBDeg = 0; ///< highest degree an evaluation of a region-B piece reads
    int regionBStored = 0; ///< coefficients region B's tables store
    double delivered = 0.0; ///< worst error the partition's fits were measured to deliver
    double bound = 0.0; ///< the figure the partition's tables are certified against
    double lo = 0.0; ///< lowest argument the partition's own tables serve
    double hi = 0.0; ///< one past the highest argument the partition's own tables serve
};

/// Whether the named partition carries the named fit route's tables.
///
/// A partition and a route are separate choices: a partition's tables are cut
/// for whichever routes it holds, and a route with no table in it is a
/// combination a build either carries or refuses. This answers the row's own
/// coverage without a caller reading a kernel.
///
/// \param partition a row of BoysFitGranularities
/// \param route     the fit route
/// \returns         true when the partition's tables carry that route
///
/// \ingroup boys
constexpr bool FitGranularityHasRoute(const FitGranularityInfo& partition,
                                      FitRoute route) noexcept {
    return (partition.routes & (1u << static_cast<unsigned>(route))) != 0u;
}

/// Whether the named partition has a kernel for the named packing axis.
///
/// The axis is a property of a call shape and the partition of the tables, so
/// the pair is a combination a build either carries or refuses. This answers the
/// row's own coverage without a caller reading a kernel.
///
/// \param partition a row of BoysFitGranularities
/// \param axis      the packing axis
/// \returns         true when the partition has a kernel for that axis
///
/// \ingroup boys
constexpr bool FitGranularityHasAxis(const FitGranularityInfo& partition, PackAxis axis) noexcept {
    return (partition.axes & (1u << static_cast<unsigned>(axis))) != 0u;
}

/// The partitions of the fitted regions this build ships, as a report prints
/// them.
///
/// Which partition an entry accepts is the entries' own business and is refused
/// where it is named; this answers what exists, what each partition's tables
/// hold, and which of the other axes - the packing axes and the fit routes - it
/// is offered on.
///
/// \returns one row per partition, in enumerator order
///
/// \ingroup boys
std::span<const FitGranularityInfo> BoysFitGranularities() noexcept;

/// The precision lane a call runs in.
///
/// A lane is a precision and an arithmetic, not a region and not a route: the
/// same combination of the other axes exists in every lane this build carries,
/// and the figure it delivers differs by lane because the fits and the
/// arithmetic do. \c kFp16 is the fp16 and bfloat16 entries, which round a
/// 32-bit engine's result to the format at the boundary and are compiled behind
/// this build's fp16 seam. \c kFp64Device, \c kFp32Device and \c kFp16Device are
/// the device lane's own three, whose entries a host without a CUDA device
/// cannot run - every one of them is named here because a caller choosing a
/// combination has to be able to name the combination it chose, and
/// \c BoysLaneContracts says where each lane's figure comes from.
///
/// **The device's three are three lanes and not one lane named three ways.**
/// The key of the default-policy table is a (device, precision, shape) class,
/// and a device class is one of those three precisions: the device probe ranks a
/// class per (precision, question) pair, so a table whose device half carried one
/// precision member could state a row for one of the device's three classes per
/// shape and none for the other two. The three members are the three lanes the
/// device surface declares - a double lane, a float lane and a half lane
/// (`boys/boys_device_tables.hpp`, \c BoysDeviceLane), which read different
/// degree tables and are built at different budgets - and not three spellings of
/// one lane.
///
/// **The members are appended, and the order is the history rather than the
/// family.** \c kFp32Device was the device lane's only member before the other
/// two existed and its value is read by name in the accuracy gate, so it keeps
/// the value it had. \c BoysLaneContracts is indexed by this enumeration with one
/// row per member, so a member added here without its row there reads a row that
/// does not exist.
///
/// \ingroup boys
enum class Precision : std::uint8_t {
    kFp64 = 0, ///< double precision, the certified lane every other is measured against
    kFp32, ///< single precision, the host's fp32 engine
    kFp16, ///< half precision: the fp16 and bfloat16 entries, whose figure carries a term of the format
    kFp32Device, ///< single precision as the device lane runs it: the float lane, under the fast exponential's term
    kFp64Device, ///< the device's double lane: the double pieces, region A read by the seeded recurrence
    kFp16Device, ///< the device's half lane: the float lane's bodies, stored half and under the half budget
};

/// The device a call runs on: the first key of the default-policy table.
///
/// The host lanes are the entries of this header; the device lane is the CUDA
/// surface (`boys/boys_cuda.hpp`), which a build carries only where it is
/// configured for it. A row of the table names one of these, because the two
/// lanes do not choose the same things: a host row names the five axes an
/// \c EvalPolicy holds, and the device lane's own choice - its region-B
/// exponential (`boys/boys_device_tables.hpp`) - is not among them.
///
/// \ingroup boys
enum class Device : std::uint8_t {
    kHost = 0, ///< the CPU lanes: every entry of this header
    kDevice, ///< the CUDA lane: \c BoysCuda, where a build carries it
};

/// The question an entry answers: how much of the ladder a call produces, and
/// for how many arguments.
///
/// A shape is what makes two calls comparable questions or not, and it is the
/// entry rather than a parameter of it: \c BoysAllOrders *is* the ladder shape
/// at one argument and \c BoysAllN *is* the ladder shape over an array, so a
/// caller never names a shape separately - the entry supplies it. It is named
/// here because the default-policy table is keyed by it: an entry that produces
/// one order has no second order to pack into a vector lane, so the packing
/// axis is a property of the call shape and a per-shape row is what states it
/// per shape rather than once for the whole build.
///
/// \ingroup boys
enum class Shape : std::uint8_t {
    kSingle = 0, ///< one order at one argument: \c BoysSingle, \c BoysSingleF32, ...
    kAllOrders, ///< the ladder at one argument, to that argument's own top order
    kFixedN, ///< one order at every argument of an array: \c BoysFixedN
    kAllN, ///< the ladder at every argument of an array, at one common top order
    kAllNAtOrders, ///< the ladder at every argument, each stopping at its own top order
};

namespace detail {

/// The budget a class of this lane falls back to when the table names no row
/// for it: the half lanes' own budget, and the float budget every other lane's
/// default policy carries.
///
/// It is why the fallback is not one tuple imposed on every lane: a class of
/// the half lane that no row names compiles the half budget, which is the axis
/// those lanes' degrees are cut for and their figures stated at.
///
/// \param lane the precision lane
/// \returns the budget a class of that lane falls back to
constexpr BoysBudget LaneFallbackBudget(Precision lane) noexcept
{
    switch (lane)
    {
    case Precision::kFp64:
    case Precision::kFp32:
    case Precision::kFp32Device:
    case Precision::kFp64Device:
        return BoysBudget::kFloat;
    case Precision::kFp16:
    case Precision::kFp16Device:
        return BoysBudget::kFp16;
    }

    return BoysBudget::kFloat; // no enumerator reaches this
}

/// One row of the default-policy table, resolved: the policy a (device,
/// precision, shape) class compiles when its call site names no policy.
///
/// A row the table carries is an explicit specialization of this template. The
/// primary template below carries **no** \c Type, so a class the table does not
/// name has no default policy at all, and asking for one is a compile error
/// rather than a call answered by choices nobody made. There is nothing to fall
/// back to: the table *is* the defaults, and a class it omits is a row someone
/// has not written yet. **Nothing here is looked up**: the class is a template
/// argument, the row is the specialization the compiler selects for it, and no
/// run-time search, table read, function pointer or branch on anything a caller
/// chose is in it.
template <Device kDevice, Precision kPrecision, Shape kShape>
struct DefaultPolicyRow {
    /// No row names this class. The absence of \c Type is the whole of what is
    /// said here, and it is what turns a missing row into a build error.
    static constexpr bool kCarried = false;
};

// The seam's rows, one explicit specialization per row. The cells are names the
// compiler resolves against the enumerators of the axes they belong to at this
// point, so a cell naming an enumerator another axis owns is an error here
// rather than a default that is read as something else. A row naming a
// combination its shape cannot carry is caught where that class's entries are
// instantiated, which is the reading that makes a row which cannot compile a
// build error rather than a surprise at a consumer's call site.
/// \cond
#define BOYS_DEFAULT_POLICY_ROW(kDevice, kPrecision, kShape, kRoute, kScheme, kBudget, kPack,  \
                                kGranularity, kDivision)                                       \
    template <>                                                                                \
    struct DefaultPolicyRow<Device::kDevice, Precision::kPrecision, Shape::kShape> {            \
        using Type = EvalPolicy<kRoute, kScheme, kBudget, kPack, kGranularity, kDivision>;      \
        static_assert(EvalPolicyLike<Type>,                                                     \
                      "a row of the default-policy table does not name an evaluation policy: " \
                      "one of its cells is not an axis of the combination it is read as");      \
        static constexpr bool kCarried = true;                                                  \
    };

#if defined(BOYS_BUILD_DEFAULT_ROWS)
BOYS_BUILD_DEFAULT_ROWS(BOYS_DEFAULT_POLICY_ROW)
#else
// A build that names only the five axes - a fixture overriding one of them, say - carries the table
// those five make: one row per host class, every cell the build's own choice. This is not a
// fallback for a class a table omits. The table here is complete by construction, because the rows
// are written out by this list rather than looked up, and a build that writes its own table and
// omits a class still fails to compile for it. The distinction is what the row *is*: a written row
// whose cells come from the build's own five names, not an absent row answered by something else.
#define BOYS_DEFAULT_POLICY_BUILD_ROW(kPrecision, kShape)                                           BOYS_DEFAULT_POLICY_ROW(kHost, kPrecision, kShape, kDefaultFitRoute, kDefaultEvalScheme,                                LaneFallbackBudget(Precision::kPrecision), kDefaultPackAxis,                                               kDefaultFitGranularity, kDefaultDivisionForm)
#define BOYS_DEFAULT_POLICY_BUILD_ROWS(X)                                                           X(kFp64, kSingle) X(kFp64, kFixedN) X(kFp64, kAllN) X(kFp64, kAllNAtOrders)                     X(kFp64, kAllOrders) X(kFp32, kSingle) X(kFp32, kFixedN) X(kFp32, kAllN)                        X(kFp32, kAllNAtOrders) X(kFp32, kAllOrders) X(kFp16, kSingle) X(kFp16, kFixedN)                X(kFp16, kAllN) X(kFp16, kAllNAtOrders) X(kFp16, kAllOrders)
BOYS_DEFAULT_POLICY_BUILD_ROWS(BOYS_DEFAULT_POLICY_BUILD_ROW)
#undef BOYS_DEFAULT_POLICY_BUILD_ROWS
#undef BOYS_DEFAULT_POLICY_BUILD_ROW
#endif
/// \endcond

#undef BOYS_DEFAULT_POLICY_ROW

} // namespace detail

/// The default policy of a class, resolved at compile time from the table the
/// build's seam file carries.
///
/// The class is a (device, precision, shape) triple. The entry supplies the
/// precision and the shape of the call it is declared on; \c kDevice defaults
/// to \c Device::kHost, which is what a name that does not spell one resolves
/// on, and a device entry spells \c Device::kDevice. The table itself is
/// `include/boys/boys_build_defaults.hpp` - one row per class, hand-editable,
/// and replaceable whole through the \c BOYS_BUILD_DEFAULTS CMake option.
///
/// **A class the table does not name has no default, and that is a build
/// error.** There is no fallback row and no tuple to fall back to: a table with
/// a hole in it would otherwise answer that class's callers with choices nobody
/// made, silently, while a reader saw a complete table. The assertion below is
/// what makes the hole loud, and it is deliberately a hard error in this class
/// body rather than a constraint on the alias - a substitution failure in a
/// default template argument would drop the entry from overload resolution
/// instead, and a different overload would answer the call.
///
/// **The library's own choice is the key, and the caller's question is not.** A
/// caller chooses what to evaluate before the call; what this table answers is
/// what the library then picks for that choice, and a class is keyed by that
/// choice alone.
///
/// **This costs nothing at run time.** \c Type is a type, selected by the
/// compiler for a class it knows; there is no registry in it, no string key, no
/// function pointer, and no branch on anything a caller chose. A call that
/// names no policy compiles to one fully specialised body, exactly as it did
/// when the four \c DefaultPolicy* names were the whole of the seam.
///
/// \tparam kPrecision the precision lane
/// \tparam kShape the question shape
///
/// \ingroup boys
template <Precision kPrecision, Shape kShape, Device kDevice = Device::kHost>
struct DefaultPolicyFor {
    static_assert(detail::DefaultPolicyRow<kDevice, kPrecision, kShape>::kCarried,
                  "this build's default-policy table carries no row for this class. Add the row "
                  "to BOYS_BUILD_DEFAULT_ROWS in boys/boys_build_defaults.hpp, or name the policy "
                  "explicitly at the call site. A class the table does not name has no default: "
                  "answering it with one nobody chose is the defect this table exists to remove.");

    /// The policy this class compiles when its call site names no policy.
    using Type = typename detail::DefaultPolicyRow<kDevice, kPrecision, kShape>::Type;

    /// Whether the table named this class. Always true where this class is
    /// instantiated at all - the assertion above is what a false would hit -
    /// and read by the report that prints what a build will do.
    static constexpr bool kCarried =
        detail::DefaultPolicyRow<kDevice, kPrecision, kShape>::kCarried;
};

/// The policy a class compiles when its call site names no policy: the name an
/// entry's policy parameter defaults to, and the name a caller writes to ask
/// for what it would have got by naming none.
///
/// \tparam kPrecision the precision lane
/// \tparam kShape the question shape
/// \tparam kDevice which side of the interface the class is on; \c Device::kHost
///         where a call site does not spell one
///
/// \ingroup boys
template <Precision kPrecision, Shape kShape, Device kDevice = Device::kHost>
using DefaultPolicy = typename DefaultPolicyFor<kPrecision, kShape, kDevice>::Type;
/// One precision lane's contract, as a report states it.
///
/// \c bound is the base figure the lane documents for one value, over the whole
/// of x >= 0, and it is the guarantee rather than a measurement. \c additive is a
/// term the lane documents beside it - zero where the lane has none - so that
/// the figure a lane promises is \c bound plus \c additive, which
/// is what \c BoysAccuracyGuaranteed answers with. \c source states the terms
/// the base figure does not carry, because a row whose bound is not a flat
/// constant over the whole of its domain has to say so rather than let a reader
/// take the number for the whole claim: the half-precision rows carry half of
/// the last representable digit of the returned value beside the base, and that
/// term is a property of the value the caller receives, not of the call.
///
/// Every figure here is the one the README's contract table publishes for the
/// lane, and the four are stated together so that the table, the accuracy gate
/// and the accessor above them are one number rather than four transcriptions
/// of one.
///
/// \c plainAdditive is the third figure a row may carry, and it exists because
/// the division form is an arithmetic and not a spelling. The plain reciprocal
/// rounds once more per step than exact division and the refined reciprocal, so
/// on the lanes where that costs accuracy the plain form's figure is the base
/// plus this term, and \c BoysAccuracyGuaranteed answers it when the caller
/// names that form. It is 0.0 on every lane whose forms deliver one figure: the
/// double lane's plain form stays inside the base everywhere, and on the device
/// lane, which carries the axis but has no figure measured per form, the row's
/// one figure is read at that lane's own default form.
///
/// \ingroup boys
struct LaneContractInfo {
    Precision precision = Precision::kFp64; ///< the lane this row describes
    const char* name = ""; ///< the name a report prints it under
    double bound = 0.0; ///< the documented base figure per value
    double additive = 0.0; ///< a term the lane adds beside the base, 0.0 where it has none
    double plainAdditive = 0.0; ///< a term the plain reciprocal adds beside the base, 0.0 where the forms share one figure
    const char* source = ""; ///< the figures beside the base, empty where the base is the whole claim
    /// The sentence for the figure the plain reciprocal gives, where that figure is not the base and
    /// `source` does not already describe it. A figure and its sentence are handed out together, so a
    /// row whose `source` names the forms its base is for - the float row reads "every region, at
    /// exact division and the refined reciprocal" - must not hand that sentence beside a figure that
    /// carries the plain form's own term. Empty where `source` already covers every form, which is
    /// what the half lane's row does.
    const char* plainSource = "";
};

/// The precision lanes this build carries, each with the figure it documents.
///
/// \returns one row per lane, in enumerator order
///
/// \ingroup boys
std::span<const LaneContractInfo> BoysLaneContracts() noexcept;

/// Which of a combination's two accuracy figures a reading is.
///
/// The two are different numbers and neither substitutes for the other: a caller
/// deciding whether a calculation is safe needs the guarantee, a caller ranking two
/// combinations needs what each was measured to deliver.
///
/// \ingroup boys
enum class AccuracyReading : std::uint8_t {
    kGuaranteed = 0, ///< an upper bound the lane documents: what a caller may rely on
    kDelivered, ///< the figure the combination was measured to deliver: what ranks options
};

/// The accuracy one combination of this library's option space provides.
///
/// \c value is the figure, as an absolute error per value of F_n, and a number only
/// where \c available is true; where the library does not carry the combination
/// there is no figure to read and \c value is 0.0 with \c reason set, so a caller
/// cannot mistake a refusal for an accuracy.
///
/// \ingroup boys
struct AccuracyFigure {
    double value = 0.0; ///< the figure, or 0.0 where none is available
    bool available = false; ///< whether this revision carries the combination
    AccuracyReading reading = AccuracyReading::kGuaranteed; ///< which of the two figures this is
    const char* source = ""; ///< the table the figure was read from
    const char* reason = ""; ///< why no figure is available, empty where one is
};

/// The accuracy a combination is guaranteed: an upper bound the lane documents
/// for it.
///
/// The figure to rely on: the lane's contract table publishes it, plus the
/// lane's own additive term where it documents one. A combination the library
/// does not carry has no figure and says so.
///
/// The bound is the lane's and not the axes': a way to read any of the five axes
/// that narrowed it would be a bound this build does not certify, and the
/// per-axis figures are the ones each axis's own row publishes. What the axes
/// change is the *delivered* figure - see \c BoysAccuracyDelivered - the one to
/// rank two combinations by.
///
/// **The division form is the one argument that does move the figure**, because
/// it is an arithmetic rather than a spelling. On a lane whose
/// plain reciprocal rounds once more per step, the figure that form is
/// guaranteed is the lane's base plus the term its row publishes for it. Name
/// the form you will evaluate in and this answers the figure for that form; a
/// caller who names no form gets the figure of the library's default form.
///
/// \param precision   the lane
/// \param route       the fit route
/// \param scheme      the evaluation scheme
/// \param axis        the packing axis
/// \param granularity the interval partition
/// \param form        how the recursion divides
/// \returns the figure, and whether this revision carries the combination
///
/// \ingroup boys
AccuracyFigure BoysAccuracyGuaranteed(Precision precision,
                                      FitRoute route,
                                      EvalScheme scheme,
                                      PackAxis axis,
                                      FitGranularity granularity,
                                      DivisionForm form = kDefaultDivisionForm) noexcept;

/// The bound a class's default policy carries: what a caller holding the
/// default has, without reconstructing the axes to ask about it.
///
/// The axes it asks the accessor above with are the ones
/// \c DefaultPolicy<kPrecision, kShape, kDevice> resolves to - the row the table
/// carries for the class, or the seam's own five where it carries none - so this
/// is the same figure, from the same table, as a caller gets by naming the
/// policy's five axes by hand. It is a table read and not a measurement: nothing
/// is evaluated and nothing is timed.
///
/// **A class is a (device, precision, shape) triple, and this reads the class the
/// caller names.** \c kDevice defaults to \c Device::kHost, which is what a name
/// that does not spell a device resolves on, so a host class is read as it was;
/// a caller naming \c Device::kDevice and one of the device's lanes reads that
/// class's own row, and the figure is that lane's - the device's three precisions
/// are three lanes of this table (\c Precision::kFp64Device, \c kFp32Device and
/// \c kFp16Device), and each carries its own row of \c BoysLaneContracts.
///
/// \tparam kPrecision the precision lane
/// \tparam kShape the question shape
/// \tparam kDevice which side of the interface the class is on; \c Device::kHost
///         where a caller does not spell one
/// \returns the figure, and whether this build carries the combination
///
/// \ingroup boys
template <Precision kPrecision, Shape kShape, Device kDevice = Device::kHost>
inline AccuracyFigure DefaultGuarantee() noexcept
{
    using Policy = DefaultPolicy<kPrecision, kShape, kDevice>;

    return BoysAccuracyGuaranteed(kPrecision, Policy::kRoute, Policy::kScheme, Policy::kPack,
                                  Policy::kGranularity, Policy::kDivision);
}

/// The accuracy a combination was measured to deliver, which is the figure that
/// ranks two combinations against each other.
///
/// It is the worst figure over the rows the combination names - the fit route's
/// row, the partition's row and the scheme's row, each of which publishes what it
/// was measured to deliver - and those are the figures of the *fits* the
/// combination names. A call adds its own recurrences over those fits, so this is
/// a floor on the error a whole call delivers and not the whole call's figure: it
/// is the number to compare two combinations by, and not a number to quote as
/// what a call achieves. What a call achieves is what the accuracy gate measures,
/// over a committed reference grid, and the gate's report is where that figure
/// lives for every combination.
///
/// This is a swept maximum and not a bound. It is available where a row carries
/// a measurement: a delivered figure is a measurement, so where no row carries
/// one the answer is that there is none rather than a number derived from a
/// bound.
///
/// It is absent for the half-precision lanes, whose error is dominated by the
/// format's own quantum at the returned value: no row of this library measured
/// a half-typed return, and the guaranteed figure above is the one to use.
///
/// \param precision   the lane
/// \param route       the fit route
/// \param scheme      the evaluation scheme
/// \param axis        the packing axis
/// \param granularity the interval partition
/// \returns the figure, and whether this revision carries the combination and
///          measured it
///
/// \ingroup boys
AccuracyFigure BoysAccuracyDelivered(Precision precision,
                                     FitRoute route,
                                     EvalScheme scheme,
                                     PackAxis axis,
                                     FitGranularity granularity) noexcept;

/// Which of a combination's two figures met a tolerance a caller named.
///
/// The two figures are different numbers and this says which one decided. A caller
/// deciding whether a calculation is safe wants the state the guarantee stands
/// behind; a caller choosing between two combinations at a target wants the state
/// that says whether the combination is at that target at all. \c kDeliveredInside
/// is where those two questions part company, and it is a state of its own rather
/// than a yes: the figure that decided it is a measurement of the fits the
/// combination names, and a measurement is not a guarantee that a whole call stays
/// inside it.
///
/// \ingroup boys
enum class ToleranceVerdict : std::uint8_t {
    kNotCarried = 0, ///< this revision does not carry the combination: no verdict and no figures
    kGuaranteedInside, ///< the bound the combination carries is at or below the request
    kDeliveredInside, ///< the bound is above the request and the measured figure is at or below it
    kOutside, ///< no figure this library holds for the combination is at or below the request
};

/// What a combination answers when the caller names the error it needs, and the
/// figures the answer was made on.
///
/// Both figures are stated beside the verdict so that the answer can be read rather
/// than taken: \c bound is the figure the lane documents for the combination, which
/// is the one a calculation's safety rests on, and \c delivered is the
/// figure the combination's own rows were measured to deliver, which is the one that
/// ranks two combinations against each other. The verdict says which of the two met
/// \c requested. \c deliveredKnown is false where no measured figure is held for the
/// combination - the half lanes - so a
/// zero \c delivered is never taken for a measurement of nought.
///
/// A combination this revision does not carry returns no figure at all: \c verdict
/// is \c kNotCarried, both figures are 0.0, \c deliveredKnown is false, and
/// \c reason carries the library's own sentence for the refusal. That sentence is
/// also where the two kinds of refusal stay apart, because it names the work rather
/// than the outcome: a refusal names either a table or kernel this library has
/// not built, which is work owed, or a shape the call itself cannot have, which no
/// revision lifts.
///
/// \ingroup boys
struct CombinationCoverage {
    ToleranceVerdict verdict = ToleranceVerdict::kNotCarried; ///< which figure met the request
    double requested = 0.0; ///< the absolute error the caller asked for
    double bound = 0.0; ///< the figure the lane documents for the combination, 0.0 where refused
    double delivered = 0.0; ///< the figure the combination was measured to deliver, 0.0 where none
    bool deliveredKnown = false; ///< whether a measured figure is held for the combination
    const char* source = ""; ///< the table the figure that decided the verdict was read from
    const char* reason = ""; ///< why there is no verdict, empty where there is one
};

/// Whether a combination provides the accuracy the caller needs, asked at the
/// tolerance the caller names rather than answered as a figure to compare.
///
/// The two figures a combination has answer different questions, and this is the
/// entry for a caller who has a target rather than a comparison: it picks the
/// right figure of the two, where comparing by hand is the mistake the verdict
/// removes.
///
/// The verdict is decided by the bound first: a combination whose bound is at
/// or below the request is \c kGuaranteedInside, which is the only state a
/// calculation's safety can rest on. Where the bound is above it the
/// measurement decides, and a combination measured at or below the request is
/// \c kDeliveredInside - at the target in what it delivers, and not covered by
/// its guarantee. Where neither is, the answer is \c kOutside. Both figures
/// and the request are returned, so the answer can always be checked against
/// the numbers it was made on, and \c source names the table the figure that
/// decided it came from rather than always the bound's.
///
/// The figures are the same ones the two accessors answer with, read from the
/// library's own tables: this entry computes nothing of its own and holds no
/// second list of numbers.
///
/// \param precision   the lane
/// \param route       the fit route
/// \param scheme      the evaluation scheme
/// \param axis        the packing axis
/// \param granularity the interval partition
/// \param tolerance   the absolute error the caller needs, > 0
/// \returns           the verdict and the figures it was made on; \c
///                    kNotCarried, no figures and a reason where this revision
///                    does not carry the combination
/// \pre               \p tolerance is positive and finite. A request that is
///                    zero, negative or not a number is answered \c kOutside
///                    where the combination is carried, as no figure is at or
///                    below it, and \c kNotCarried where it is not; a request
///                    of positive infinity is answered \c kGuaranteedInside,
///                    as every figure this library holds is at or below it.
///
/// \ingroup boys
CombinationCoverage QueryCombination(Precision precision,
                                     FitRoute route,
                                     EvalScheme scheme,
                                     PackAxis axis,
                                     FitGranularity granularity,
                                     double tolerance) noexcept;

/// One certified fit route as a report states it.
///
/// The figures are the route's own rather than a lane's: a route supplies the fits of
/// one region, and what a caller receives from a lane entry is those fits carried
/// through the region's recurrence. \c stored is what the evaluation reads;
/// \c delivered is the worst error a sweep measured over \c [lo, hi), in the
/// arithmetic the kernel evaluates the fit in and against the high-precision
/// reference the fits themselves are validated against; and \c bound is the bar the
/// route is certified against, at or above \c delivered. A delivered figure is a
/// swept maximum and not a bound.
///
/// A route that serves more than one region has one row per region, so two rows can
/// carry the same \c route and differ in \c region, \c lo, \c hi, \c stored,
/// \c delivered and \c bound.
///
/// \c lo..hi is the domain of the route's fit, and \c servesFrom is the lowest
/// argument from which naming the route changes the values a caller receives. The two
/// are the same for every route whose selector takes over at its fit's left edge, and
/// they are stated apart because they can differ: a route whose fit covers more than
/// the selector hands it says so here, rather than claiming a domain it does not
/// serve. Region A's rational route is the case that rule exists for, and its
/// boundary is the lowest of a set rather than a single one: each order is handed to
/// the route from that order's own argument, so a caller above \c servesFrom but
/// below an order's own boundary still receives the default route's value for that
/// order.
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
/// The route selects fits and changes nothing else. Outside the intervals a route
/// reports in BoysFitRoutes this entry runs the default route's own code and returns
/// the default entry's values bit for bit. Inside them the named route's fits produce
/// the values, at the same bound, over the arguments the report's row says its
/// selector takes them over - and a route served per order takes over per order, so
/// a caller below an order's own boundary receives the default's value for it.
///
/// This entry names one axis and no other: the route selects a fit, and the
/// accuracy is the one every entry of this header carries.
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
/// The route and the scheme are the two fields of the policy the templated entries
/// carry (\c EvalPolicy), and they select different things: the route names the fits
/// that serve the regions, and the scheme names the summation the Chebyshev family's
/// coefficients are read in. A route whose own fit has one stored form - the rational
/// minimax family's monomial numerator and denominator - evaluates that fit the same
/// way under either scheme, and the scheme reaches the parts of the call that route's
/// fits do not serve: naming a scheme changes the values only outside the served
/// intervals a route's rows report, where the route has already handed the argument
/// back to the shipped family, and inside them the route's own fits answer at the bar
/// its row states. A scheme outside the enumeration is the reference scheme's body,
/// the same fallback this entry takes for a route it does not serve.
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










/// F_n(x) in double precision, |F̂ − F| ≤ B_region per region (contract
/// table in the file preamble).
///
/// \tparam Policy the evaluation policy (\c EvalPolicy): the fit route, the
///         scheme its coefficients are summed in, the partition of the fitted
///         regions, and a single-precision engine's budget, selected together.
///         The default, \c DefaultPolicyFp64, names every axis the library
///         defaults, so a call site that names no axis compiles that policy's code
///         path, and naming any axis is how a caller asks for another
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kSingle>>
double BoysSingle(int n, double x) noexcept;

/// F_0(x)..F_nmax(x) in double precision; the batch entry's contract — the
/// \c "double batch" row of the table in the file preamble:
/// |F̂ − F| ≤ 5.5e-14 per value in every region.
///
/// The row is named rather than \c B_region because this entry's delivered
/// error is the batch row's and not the per-region column's: its region-A error
/// already reaches 3.2·1e-15, so a caller reading \c B_region as the
/// per-region column would hold this entry to up to 54 times tighter than its
/// code delivers.
///
/// The batch is the pattern real integral engines use (the McMurchie-Davidson
/// [McMurchie1978] and Obara-Saika [ObaraSaika1986] recursions consume all
/// orders at once) and is considerably
/// cheaper than nmax + 1 single evaluations.
///
/// \tparam Policy see BoysSingle
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>
void BoysAllOrders(int nmax, double x, double* out) noexcept;

/// F_n(x_i) for an array of arguments at one fixed order n, double
/// precision; |F_hat - F| <= B_region per value (the BoysSingle
/// per-region contract, region table in the file preamble).
///
/// The batch shape of integral-engine inner loops that group shell pairs by
/// angular momentum: each element needs exactly one order, so no unused
/// cross-order recursion is paid. Each output element returns BoysSingle's
/// value at the same (n, x) and carries the single lane's per-region bound
/// with it: the entry runs the certified scalar single-lane region bodies
/// verbatim. An element agrees with the single entry bit for bit on a build
/// that does not contract a bare product-plus-add, which is what x86-64 without
/// -mfma and MSVC everywhere deliver. A build that does contract one decides per
/// call site whether to fuse that form, so an element can differ from the single
/// entry's in the last place and still be inside the bound; what does not move is
/// the bound. Arguments need no pre-partitioning: the region dispatch is per
/// element, the portable shape. The vector tier is reached through the batch
/// entries (BoysAllN), which group the arguments once for the whole call.
///
/// Layout: out[i * stride] = F_n(x[i]), i = 0..count-1; stride is measured
/// in doubles and defaults to 1 (contiguous). The output span must hold
/// (count - 1) * stride + 1 doubles; count may be 0 (no writes). The entry
/// is scalar and portable: x and out need no alignment beyond
/// alignof(double), and the two arrays must not overlap.
///
/// \tparam Policy see BoysSingle. A call naming a route other than the shipped
///         one is answered by the per-argument single entry, once per argument,
///         whose body this entry's own path already mirrors region for region.
///         A call naming the uniform grid is answered the same way, for the
///         same reason: the grid is a table read one order at a time, and
///         the shaped body below is a walk over the regions the grid does not
///         have. Both keep this entry's documented accuracy - the path costs the
///         shaped body's region dispatch, not a value
///
///         The packing axis is not an axis of this shape, and that is the
///         entry's signature rather than a body nobody built: a packed lane keeps
///         four doubles in a register, and this entry produces ONE order at every
///         argument of the array, so there are not four orders to fill a lane
///         with. The wide dimension the call does have - count - is a different
///         axis, served by the vector tier the batch entries reach. Naming
///         PackAxis::kOrders is therefore rejected at the call site, which is
///         what the assertion says
/// \param n      order, 0..kMaxBoysOrder - the batch's single fixed order
/// \param x      array of count arguments, each >= 0
/// \param out    receives F_n(x[i]) at out[i * stride]
/// \param count  number of arguments
/// \param stride output stride in doubles, >= 1 (default 1 = contiguous)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
void BoysFixedN(
    int n, const double* x, double* out, std::size_t count, std::size_t stride = 1) noexcept;

/// Tag for the many-argument entries' already-grouped overload: the caller
/// states that the arguments are in non-decreasing order.
///
/// Every dispatch path of this library is an interval of the argument line and the
/// intervals are ordered, so the classification is monotone in x: the arguments of a
/// non-decreasing array are already contiguous by path, the grouping is free, and the
/// entry skips the sort it would otherwise pay for. The precondition is a property of
/// the caller's array, checkable without knowing anything about the library's regions.
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
/// The batch shape a shell-quartet consumer needs: many arguments, all nmax + 1
/// orders each, in one call, in any argument order. The entry classifies,
/// groups by dispatch path, runs each group's kernel and returns the results in
/// the caller's order — a documented performance promise, so a caller pays the
/// grouped path whatever its ordering, without knowing what the library groups
/// by. BoysSortedArgs states that the arguments are already non-decreasing and
/// skips the sort entirely.
///
/// Layout: order-major planes, out[k * count + i] = F_k(x[i]) — all F_0
/// contiguous, then all F_1, and so on: the layout a contraction consuming one
/// order across many arguments, or a vectorised kernel, wants. The output holds
/// count * (nmax + 1) doubles.
///
/// Accuracy: every returned value satisfies the double batch lane's documented
/// per-region bound, |F̂ − F| ≤ 5.5e-14 (the contract table in the file
/// preamble), which is the bound the per-argument BoysAllOrders call meets —
/// the same contract, not necessarily the same bits. On a host with AVX2 a
/// low-order batch (where the region-A lane's one-order-at-a-time cost is
/// amortized) hands its region-A runs to that lane and differs from
/// BoysAllOrders only within that lane's own region-A budget; every other case
/// runs the per-argument path's own bodies at the batch's nmax, and is
/// bit-identical to BoysAllOrders called at that same nmax. (Region A's body
/// seeds at nmax and recurses down, so a batch's F_k for k < nmax is the
/// recurrence's value, not the one a per-order call at nmax = k walks; both are
/// inside the bound.) The suite measures the observed maximum difference on both
/// routes and reports it.
///
/// Threading: single-threaded and pure like every entry here — divide the array
/// over your own threads; distinct batches share nothing.
///
/// Workspace: BoysAllNWorkspaceSize(count) std::size_t words from the caller
/// keeps a hot loop allocation-free; the default nullptr allocates internally.
/// The entry is total: a failed allocation falls back to the per-argument path,
/// at the same bound and without the grouped-path speed-up.
///
/// \tparam Policy see BoysSingle. The route is a property of the fit a value is
///         read from, so a call naming the rational route takes the
///         per-argument path, whose body is the all-orders entry's own and takes
///         its fit from the policy; both shapes answer inside this entry's own
///         bound. The partition is a property of the table a value is read from
///         in the same way, so a call naming the uniform grid takes that same
///         path: its body reads the grid below the join and the asymptotic form
///         above it, while the partitioned path - which reads the shipped and
///         narrow pieces through a recursion the grid has no walk for - is never
///         handed the grid
///
///         The orders axis is the shape this entry's layout already has:
///         out[k * count + i] is F_k of one argument, so the per-argument path is
///         its body, and the packed lane that fills a register with four orders
///         of that argument is the all-orders entry's. Naming the axis trades
///         the region grouping (which exists to feed a lane that packs four
///         arguments) for that lane. The axis carries either route, at B_region
///         as this entry does
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
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
void BoysAllN(int nmax,
              const double* x,
              double* out,
              std::size_t count,
              std::size_t* workspace = nullptr) noexcept;

/// BoysAllN for arguments the caller states are already in non-decreasing
/// order — the sort is skipped rather than paid for.
///
/// \tparam Policy see BoysAllN: the same entry and the same paths, reached
///         without the sort
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      array of count arguments, non-decreasing, each >= 0
/// \param out    receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count  number of arguments; may be 0 (no writes)
///
/// \pre x[i - 1] <= x[i] for every i in [1, count) — the declaration this
///      overload exists for, and the caller's responsibility. The runs served
///      are re-derived from the classification rather than taken from the
///      declaration, so a caller that declared an order it did not have gets the
///      ungrouped path's performance, never a wrong value; the assertion below
///      reports the violation in a build with assertions, and this entry has no
///      other error channel.
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
void BoysAllN(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;

/// F_0(x_i)..F_n[i](x_i) for an array of arguments, double precision — every
/// order up to each argument's OWN top order, the tops arriving as an array.
///
/// The shape a shell-quartet consumer actually has: a quartet carries its own
/// highest order, so a batch of quartets is a batch of differing tops, and
/// BoysAllN — one common nmax — makes such a caller either pad every quartet up
/// to the batch's largest top order, paying for orders nobody asked for, or call
/// the library once per quartet. Taking the tops as an array does neither. It is
/// the CPU spelling of the device lane's per-element-order batch
/// (BoysCuda::AllOrdersF64), which is what lets one kernel be written against
/// both lanes.
///
/// Layout: order-major planes as BoysAllN — out[k * count + i] = F_k(x[i]) —
/// with each column stopping at its own order. Every cell of argument i's column
/// at or below n[i] is written, and **every cell above it is left untouched**:
/// out[k * count + i] for k > n[i] keeps whatever the caller put there, so a
/// caller may pre-fill those cells with the value its own contraction wants to
/// multiply by (a zero, or an earlier group's result) and know it survives the
/// call.
///
/// The output holds count * (nmax + 1) doubles, nmax being the largest of the
/// n[i] — the caller's own array, sized from what the caller already has, with no
/// padding of the arguments or the output.
///
/// Accuracy: every returned value satisfies the double batch lane's documented
/// per-region bound, |F̂ − F| ≤ 5.5e-14 (the contract table in the file
/// preamble). Each column is the per-argument all-orders body run at that
/// argument's own top order: out[k * count + i] is the value, bit for bit, that
/// BoysAllOrders(n[i], x[i], out) returns at out[k].
///
/// Threading: single-threaded and pure like every entry here — divide the array
/// over your own threads; distinct batches share nothing.
///
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
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>
void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept;

/// Whether the packed region-A lane serves a call whose policy names this
/// scheme.
///
/// The many-argument entries hand their low-order region-A runs to the packed
/// AVX2 lane where the build has one. That lane holds the Chebyshev
/// coefficients and the split Clenshaw recurrence, so it serves the split
/// Clenshaw scheme and no other - which is not the default scheme at this
/// revision, so the lane is what a call naming \c kSplitClenshaw reaches and
/// not what a call naming no scheme reaches: a call naming another scheme is
/// answered on the scalar
/// body, at the same bound and with the same values the per-argument entry
/// returns. What is not offered is the lane, not the value - and a lane is a
/// performance property, so nothing in the returned values can show which one
/// ran.
///
/// This answers which lane a call got, without a caller measuring it. It is the
/// packed lane's scope and not the entry's: the scheme is served on every entry that
/// takes a policy, and the packed lane is an implementation of region A that serves
/// one of them.
///
/// \param scheme the evaluation scheme a call names
///
/// \returns true if the packed region-A lane serves that scheme
///
/// \ingroup boys
constexpr bool BoysPackedLaneServes(EvalScheme scheme) noexcept
{
    // The scheme the lane's own recurrence is, named rather than read from the
    // default: the lane serves one summation and the default is a value a
    // measurement may move, so answering with it would report a lane the
    // caller does not get.
    return scheme == EvalScheme::kSplitClenshaw;
}

/// F_n(x) in single precision, |F̂ − F| ≤ 1.5e-7.
///
/// Same piecewise structure as the double lane with per-order float fits.
/// This is the recommended lane for GPU integral evaluation.
///
/// \tparam Policy see BoysSingle. The lane reads the policy's fit route and its
///         evaluation scheme: the route selects which family supplies the
///         lane's own fits - the shipped Chebyshev table or the rational
///         minimax one - and the scheme selects which of the Chebyshev family's
///         two parallel tables, the Chebyshev form or the monomial form of the
///         same fits, is summed. The packing axis is not an axis of this shape
///         and is refused where it is named: this entry answers one order at
///         one argument, so the value is a single float and there is no lane to
///         fill
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \returns     F_n(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kSingle>>
float BoysSingleF32(int n, float x) noexcept;

/// F_0(x)..F_nmax(x) in single precision, |F̂ − F| ≤ 1.5e-7 per value.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits -
///         the route for this lane's region-B seed and for the double lane's
///         fit that seeds region A, the scheme for which of the route's tables
///         each of those reads. Region A's seed is the double lane's fit at the
///         policy's pair, so its degrees are the double lane's; region B's is
///         this lane's own. The packing axis is read here too: the orders axis
///         packs eight orders of the single argument and reads the tables the
///         policy names, so a policy naming the narrow partition reads each
///         order's own piece of it (each partition holds its own family's
///         coefficients, so a reading of either is a reading of the family
///         named), and it carries either route. Naming the arguments axis
///         names no lane this shape has to fill: the values are the per-argument
///         body's either way, and the batch entry is where an array of arguments
///         is answered, one argument at a time
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kAllOrders>>
void BoysAllOrdersF32(int nmax, float x, float* out) noexcept;

/// F_0(x_i)..F_nmax(x_i) for an array of arguments, single precision — the
/// float lane's entry of the shape BoysAllN has in the double lane.
///
/// Layout and totality as BoysAllN: out[k * count + i] = F_k(x[i]), order-major
/// planes, the output holding count * (nmax + 1) floats. The same shape exists on
/// the device (BoysCuda::AllNF32), so a consumer porting a batch between the two
/// lanes has a call on both.
///
/// It is not a grouping entry: the packed region-A lane the double batch hands
/// its low-order runs to is an AVX2 kernel over doubles, the float lane has no
/// packed region kernel of its own, and so this entry evaluates the per-argument
/// all-orders body at each argument. No speed is claimed for it over the same loop
/// written at the call site.
///
/// Accuracy: |F̂ − F| ≤ 1.5e-7 per value, the float lane's bound, which is the
/// same in every region (the contract table in the file preamble) — the bound
/// BoysAllOrdersF32 meets, and each column is that entry's value at the same
/// (nmax, x[i]) bit for bit.
///
/// Threading: single-threaded and pure like every entry here — divide the array
/// over your own threads; distinct batches share nothing.
///
/// \tparam Policy see BoysSingleF32: the route and the scheme select the fits.
///         The packing axis is read at every argument - the all-orders body this
///         entry calls packs eight orders of one argument under the orders
///         axis - and what stays at the caller's loop is the region partitioning
///         of the arguments axis
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each >= 0
/// \param out   receives count * (nmax + 1) floats, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \pre x must hold count values, out count * (nmax + 1), and the two must not
///      overlap
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp32, Shape::kAllN>>
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
/// The route selects fits and changes nothing else. Region C holds no coefficient
/// under either route and an argument there is the default entry's value bit for
/// bit; outside the intervals the route reports in \c BoysFitRoutesF32 the same
/// holds.
///
/// The two routes are alternatives: the rational route holds half
/// the stored coefficients over region A and delivers more error than the
/// Chebyshev route at the same bar, so which of the two is cheaper is a property
/// of the caller's machine rather than of the tables. Both are certified against
/// the lane's own bound.
///
/// This entry selects a fit and nothing else, and a caller who wants a
/// compile-time route names it in the policy it templates on instead.
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
/// |result - F_n(x16)| <= 1.5e-7 + one half-ULP of representation (the
/// representation term is the format's, and the base is the float lane's own
/// figure: this lane runs that lane's arithmetic). The bound is claimed only over the
/// arguments where |F_n(x16)| exceeds it, and no accuracy is claimed past that
/// ceiling: in f16 the return there is a subnormal number, then exactly zero,
/// with the bound met by the format's floor rather than by the lane — see the
/// contract table in the file preamble.
///
/// The axes are this lane's entries' as they are the float lane's: every
/// combination of the route, the scheme, the partition and the packing axis this
/// lane's fits are carried at is a policy a call site can name. The budget is what
/// the lane fixes - a property of the half lane rather than of a call - so a
/// policy named here is read for its four other axes and the engine is the fp16
/// one; naming none runs \c DefaultPolicyFp16, this lane's own default and the
/// combination its figures are stated for.
///
/// \tparam Policy the evaluation policy (\c EvalPolicy): the fit route, the
///         scheme its coefficients are summed in, the partition of the fitted
///         regions and the packing axis, selected together; \c DefaultPolicyFp16
///         by default, which is the combination the lane's own figures are
///         stated for
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \returns     F_n(x) in fp16
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kSingle>>
F16 BoysSingleF16(int n, F16 x) noexcept;

/// F_0(x)..F_nmax(x) in fp16, same certified-boundary contract as
/// BoysSingleF16 per value.
///
/// \tparam Policy see BoysSingleF16
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (fp16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kAllOrders>>
void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept;

/// F_n(x) in bfloat16 — the Bf16 lane of the certified mixed-precision
/// boundary, same
/// contract as BoysSingleF16 (bf16 representation term, 8-bit mantissa).
///
/// \tparam Policy see BoysSingleF16. The two half formats are one lane at one
///         budget, so this is the fp16 lane's policy type: \c DefaultPolicyBf16
///         is \c DefaultPolicyFp16
/// \param n     order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \returns     F_n(x) in bf16
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kSingle>>
Bf16 BoysSingleBf16(int n, Bf16 x) noexcept;

/// F_0(x)..F_nmax(x) in bf16, same contract as BoysAllOrdersF16 per value.
///
/// \tparam Policy see BoysSingleBf16
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     argument, >= 0 (bf16)
/// \param out   receives nmax + 1 values, out[k] = F_k(x)
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp16, Shape::kAllOrders>>
void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept;



/// The power of two the native half lane scales its results by: 15, so 2^15 —
/// the largest power of two binary16 holds, and an exact scale in it. A
/// caller recovers F_k(x) from the entry below with an exact power-of-two
/// division by 2^kHalfNativeScaleExponent.
inline constexpr int kHalfNativeScaleExponent = 15;

/// F_0(x)..F_nmax(x) for a packed pair of arguments, evaluated in packed half
/// arithmetic — the native half lane.
///
/// Not the fp16 lane above under another signature: BoysAllOrdersF16 rounds once
/// around the certified fp32 engine, where this entry *is* region C's asymptotic
/// ladder in binary16 — one square root and one divide for the seed, then one
/// packed multiply and one packed divide per order, one rounding per operation,
/// two arguments to a register, correctly rounded to half throughout (half2.hpp).
/// Its error is therefore the arithmetic's rather than the engine's: about one
/// ULP, an order of magnitude above the I/O lane's budget, and it grows with the
/// order because a ladder of 2n + 2 roundings has 2n + 2 chances to round.
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
// entries declared above at the default policy, not an entry of its own. They
// are what the default call sites link against instead of compiling the kernel
// again in their own translation unit; a caller that names another policy
// compiles the combination it asks for from the definition below.
//
// Declared only for the table this build carries. A build that supplies its own
// through BOYS_BUILD_DEFAULTS has its own policy types - the table is read by
// value, so a different row is a different instantiation with a different
// mangled name - and the library carries no definition for those. Declaring
// them here would promise a definition that does not exist and turn a working
// consumer build into an unresolved external.
#if !defined(BOYS_BUILD_DEFAULTS_REPLACED)
extern template double BoysSingle<>(int n, double x) noexcept;
extern template void BoysAllOrders<>(
    int nmax, double x, double* out) noexcept;
extern template void BoysFixedN<>(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept;
extern template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept;
extern template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;
extern template void BoysAllNAtOrders<>(
    const int* n, const double* x, double* out, std::size_t count) noexcept;
// The single-precision entries are NOT declared extern here, and the reason is the
// one the paragraph above gives. An extern declaration PROMISES that this translation
// unit's instantiation is defined elsewhere; the library defines a fixed list of them
// (src/boys_orders_simd.cpp), and since a default call site now resolves its policy
// from the table, the instantiation it selects is not knowably in that list.
// Declaring it anyway turned four consumer targets into unresolved externals on gcc.
// Their definitions are in this header, so a consumer instantiates what it names.

#if BoysFp16
extern template F16 BoysSingleF16<>(int n, F16 x) noexcept;
extern template void BoysAllOrdersF16<>(
    int nmax, F16 x, F16* out) noexcept;
extern template Bf16 BoysSingleBf16<>(int n, Bf16 x) noexcept;
extern template void BoysAllOrdersBf16<>(
    int nmax, Bf16 x, Bf16* out) noexcept;
#endif // BoysFp16
#endif // !BOYS_BUILD_DEFAULTS_REPLACED
/// \endcond

} // namespace boys

// The kernel behind the entries above, shipped as a header so that every policy
// a caller names is instantiable at the call site. Included here rather than at the
// top of this file because its definitions name the declarations above.
#include "boys/boys_impl.hpp"

// The option probe closes the surface: it measures the entries above on the
// machine it is called on. Included last, because it is written against every
// declaration in this file.
#include "boys/boys_probe.hpp"
