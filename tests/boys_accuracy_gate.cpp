// The accuracy gate: every documented per-lane, per-region bound measured
// against an independent high-precision reference, with the ratio to the
// bound reported as the maximum over the sweep and the location it falls at.
//
// It exists to find claims that are not met. Three things it does that the
// gtest suite does not:
//   * the reference is the committed mpmath grid
//     (tests/data/boys_accuracy_gate_reference.csv), not the library - so the
//     fp16/bf16 lanes, whose suite oracle is the certified double lane, are
//     measured against something they cannot have influenced;
//   * the sweep is dense over every region and every dispatch boundary of
//     every order, and runs to x = 1e18, well past the 100 the suite's grid
//     stops at;
//   * a pass whose bound is larger than the function's own magnitude is
//     counted and reported separately, so a lane that returns nothing usable
//     cannot hide inside a green total.
//
// Every documented claim gets exactly one verdict - verified at this revision,
// exceeded, or one of the two not-met-not-refuted verdicts: vacuous only (every
// meeting point is bound-covered, so any value in range would pass) and
// evidence absent from the tree (the claim rests on a measurement this revision
// cannot re-run; the ceiling the claim must respect is computed and printed
// beside it). Neither of the last two is counted as met, and the revision the
// binary measured is printed at the top, from the tree at configure time.
//
// Why the reference is trusted. It is committed as data and re-derivable with
// `python tools/gen_boys_accuracy_gate_reference.py`, a script that shares no
// code with the library and none with the library's own committed grid. It
// evaluates every cell in mpmath at dps = 80 and prints 25 significant digits,
// by routes that are structurally different from each other - the incomplete
// gamma closed form (with the elementary erf form for n = 0), direct
// quadrature of the defining integral, and a small-x Taylor series - and it
// prints their disagreement rather than asserting it. Re-run at this revision
// on 561 validation points the routes agree to
//   * incomplete gamma against quadrature: 1.05422e-81 (relative 2.05303e-81);
//   * erf F_0 against the gamma route:     1.05422e-81;
//   * Taylor series against the gamma route: 7.57923e-67;
// and the three-term identity (2n+1)F_n - 2x F_(n+1) = e^-x, which a corrupted
// table cannot satisfy, holds across the generated grid with a relative
// residual of 1.88287e-65 below x = 40; the generator says where the identity
// stops being informative, since past that the reference's own working
// precision is what the residual measures.
// That is 65 to 81 orders below the tightest bound this gate measures, so a
// cell the gate calls exceeded is the lane's error and not the reference's.
// The grid's argument coverage, and where it stops, are printed in the gate's
// own coverage block; the gate reports a measured ratio only where a reference
// cell exists, and names the arguments that have none.
//
// Exits non-zero, printing the offending numbers and naming every claim that is
// not verified, whenever the count falls short of the claim book: a claim
// measured exceeded, a claim measured vacuous only, and a claim whose evidence
// this revision cannot re-run all leave the gate red. --strict is accepted and
// is the same verdict. --per-order prints the delivered error beside the
// claimed one for every lane, region and order, with the cells the bound binds
// on and the cells it cannot fail on counted in their own columns, and
// --probe n x prints one cell from every lane for a single argument.
//
// Run:  cmake --build <build> --config Release --target boys-accuracy-gate
//       <build>/Release/boys-accuracy-gate [--per-order] [--strict]
//       ctest --test-dir <build> -C Release -R boys-accuracy-gate

#include <boys/boys.hpp>
#include <boys/f16.hpp>

// The native packed half lane is a lane of its own and arrives on its own
// branch. A revision that does not carry it must report the lane's claims as
// evidence absent rather than skip them, so its measurement is guarded here and
// the guard is stated in the report; the claim slots exist either way, so the
// count of claims does not move when the lane lands - only their verdicts do.
#if __has_include("boys/half2.hpp")
#include <boys/half2.hpp>
#define BOYS_GATE_NATIVE_HALF 1
#endif

// The region-A transform lane is an entry of its own and carries its own
// header and its own published bounds. The same rule applies to it: a revision
// that does not carry the header must report the lane's claims as evidence
// absent rather than skip them, so its measurement is guarded here and the
// guard is stated in the report; the claim slots exist either way, so the count
// of claims does not move when the lane lands - only their verdicts do.
#if __has_include("boys/boys_transform.hpp")
#include <boys/boys_transform.hpp>
#define BOYS_GATE_TRANSFORM 1
#endif

#include "boys/boys_impl.hpp" // the region kernels and the relaxed bodies
#include "boys_gate_reference.hpp" // the committed reference, the row shape

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

// CMake passes the measured revision; a build that bypasses it still says so
// rather than naming a revision it did not read.
#ifndef BoysGateRevision
#define BoysGateRevision "unknown"
#endif

// The gate's instrument, the reference it reads and the row shape it prints come
// from tests/boys_gate_reference.hpp, which the CUDA device gate includes as well:
// a lane measured on either side is measured against one reference and printed in
// one vocabulary. What stays here is this gate's own books - the lanes, the
// evaluation schemes and the fit routes - and the bounds each of them is judged by.
using namespace boys_gate;

namespace {

// ---------------------------------------------------------------------------
// The documented bounds, transcribed from README.md and include/boys/boys.hpp.
// ---------------------------------------------------------------------------

// README's four-region table (the one that has an extended-band cell).
constexpr double kBoundSingleA = 1e-15;
constexpr double kBoundSingleBand = 3e-14;
constexpr double kBoundSingleB = 3e-14;
constexpr double kBoundSingleC = 5.5e-14;
// README: the double batch lane, every region.
constexpr double kBoundDoubleBatch = 5.5e-14;
// README: float single / batch, every region.
constexpr double kBoundFloat = 1.5e-7;
// The bound the header's contract table applied to the whole of x < x0 before
// this revision - a single region-A cell with no extended-band row, which is
// the over-claim this gate found and the header has since stopped making. The
// slot below still measures it, so the finding stays in the report and stays
// re-measurable, but no live claim is judged by it.
constexpr double kWithdrawnHeaderABound = 1e-15;

// The native packed half lane (include/boys/half2.hpp): region C only, the
// ladder in correctly rounded binary16, scaled by 2^15 so the running value
// stays inside the format's normal range down to F_k(x) = 2^-29. Its published
// bound is in ULP of the returned value rather than a region budget, and the
// returned values are the scaled ones.
constexpr double kNativeScale = 32768.0; // 2^15, the entry's per-order scale
constexpr double kNativeUlpBound = 8.0;  // the published bound, in ULP
constexpr double kNativeSmallestNormal = 6.103515625e-05; // 2^-14
// bfloat16: 7 stored mantissa bits, smallest normal exponent -126.
constexpr int kBf16MantissaBits = 7;
constexpr int kBf16MinNormalExp = -126;

// The field a binary16 value spans from its largest finite magnitude down to
// its smallest subnormal: 65504 = 2^16 and 2^-24, so 2^40. A per-order
// power-of-two scale can only move a value inside that field, which is what
// bounds how far scaling can carry an order before the ladder's own span
// exceeds it.
constexpr double kF16Field = 65504.0 * 16777216.0;        // 65504 * 2^24
constexpr double kF16NormalField = 65504.0 * 16384.0;     // 65504 * 2^14

// The region-A transform lane (include/boys/boys_transform.hpp): the double
// table's two shared bands evaluated as one matrix product per band, in a mode
// the caller names. Its published bounds are the double single lane's region-A
// budget for kFp64 - the product adds nothing measurable to the coefficients'
// own truncation - and that same budget plus the fp32 accumulator's own floor
// for the two split modes. The two split modes' rows are a claim about the
// modelled arithmetic and not about a tensor core on a card; the lane's own
// preamble says which way the idealisation can be wrong, and the rows below
// say it again rather than leaving it to a reader who starts at the table.
constexpr double kTransformSplitFloor = 2.5e-7;
constexpr double kTransformFp64Bound = kBoundSingleA; // m*1e-15
constexpr double kTransformSplitBound = kBoundSingleA + kTransformSplitFloor;
// The multiplier the lane's rung claim is measured at: the fp64 mode's
// multiplier is live from m = 2 upward, so a relaxed width exists to measure.
constexpr double kTransformRung = 1024.0;

// The delivered worst each mode publishes over region A - the table in
// include/boys/boys_transform.hpp (1.110e-16, 1.916e-07, 1.946e-07), which the
// prose in that header, in README and in docs/lane-contract.md rounds to
// 1.11e-16, 1.92e-07 and 1.95e-07. The tighter of the two readings is the one
// the rows hold the sweep to.
constexpr double kTransformFp64Delivered = 1.11e-16;
constexpr double kTransformTf32x3Delivered = 1.916e-07;
constexpr double kTransformBf16x6Delivered = 1.946e-07;
// The figures above are stated to four figures, and that is the precision they
// are claims at: the fp64 mode's 1.110e-16 is 2^-53 written to four figures -
// the exact maximum is 1.11022e-16 - so a row comparing a sweep's own worst
// against the printed digits without this would refute a document that is right
// to the figure it printed. A delivered figure is a swept maximum and not a
// bound; the bound is the m*B column beside it in the same table.
constexpr int kTransformFigures = 4;
// The figure the lane's multiplier paragraph names as the floor the fits' term
// cannot reach: "cannot reach 1.9e-07 until m is about 1.9e8".
constexpr double kTransformSplitFloorPublished = 1.9e-07;

// The evaluation-scheme book's accumulators, held apart from the claim book on
// purpose. The RESULT line and the cell count beside it are the claim book's,
// and they are the numbers a reader of this gate has seen before: a row added
// for a new evaluation option must not move either. These slots are measured by
// the same code and totalled, judged and reported in their own block at the end
// of the run, and a row of theirs that is not met leaves the gate red exactly as
// a claim book row does.
std::vector<Accum>& SchemeClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddSchemeClaim(const char* lane, const char* region, double bound) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    SchemeClaims().push_back(a);
    return static_cast<int>(SchemeClaims().size()) - 1;
}

// The packing axis's own accumulation, held apart from the claim book and from
// the scheme book for the reason both of those give: the axis is a choice a
// caller makes about which of a call's values a vector carries, and the cells
// that measure it are not the lane cells the 39 of 39 claim count is quoted
// against. Report order does not follow declaration order here - the axis is
// declared before the route book and reported after the scheme book - because
// the report's order is the order the books were added to it, and that order is
// what keeps the earlier books' blocks byte-stable.
std::vector<Accum>& PackClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddPackClaim(const char* lane, const char* region, double bound) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    PackClaims().push_back(a);
    return static_cast<int>(PackClaims().size()) - 1;
}

// The fit routes' own accumulation, kept apart from the lanes' so the two
// tables report separately: a route is a choice a caller makes, not a lane the
// library always serves, and folding its cells into the lane sweep would move
// the cell count every lane statement is quoted against.
std::vector<Accum>& RouteClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddRouteClaim(const char* lane, const char* region, double bound, bool judged = true) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    a.judged = judged;
    RouteClaims().push_back(a);
    return static_cast<int>(RouteClaims().size()) - 1;
}

// The run-time tier's rungs, named by the multiplier each one selects.
const char* TierRungLabel(int rung) {
    switch (rung)
    {
    case 0:
        return "tier m=1 (run-time)";
    case 1:
        return "tier m=64 (run-time)";
    case 2:
        return "tier m=256 (run-time)";
    case 3:
        return "tier m=1024 (run-time)";
    case 4:
        return "tier m=4096 (run-time)";
    case 5:
        return "tier m=16384 (run-time)";
    case 6:
        return "tier m=65536 (run-time)";
    default:
        return "tier (run-time)";
    }
}

// The region a double-single argument is dispatched in, by the README's
// interval table. The library's code-path boundary inside the band is
// per-order (kTierThresholds), but the published band cell is an interval and
// the looser of the two candidate bounds covers it, so the claim is
// interval-keyed.
int SingleClaim(double x) {
    if (x < boys::detail::kExtendedBX0)
    {
        return 0;
    }

    if (x < boys::detail::kX0)
    {
        return 1;
    }

    if (x < boys::detail::kX1)
    {
        return 2;
    }

    return 3;
}

// The certified interval budget those four slots carry.
double SingleBound(double x) {
    const int region = SingleClaim(x);

    return region == 0   ? kBoundSingleA
           : region == 1 ? kBoundSingleBand
           : region == 2 ? kBoundSingleB
                         : kBoundSingleC;
}

// The name a report prints one stored fit's lane under.
const char* EvalLaneName(boys::EvalLane lane) {
    switch (lane)
    {
    case boys::EvalLane::kRegionA:
        return "region A";
    case boys::EvalLane::kRegionB:
        return "region B";
    case boys::EvalLane::kExtendedBand:
        return "extended band";
    }

    return "?";
}

// The region label a route's row is quoted under, matching the labels the lane
// claims above use.
const char* RegionName(boys::AccuracyRegion region) {
    switch (region)
    {
    case boys::AccuracyRegion::kA:
        return "A";
    case boys::AccuracyRegion::kB:
        return "B";
    case boys::AccuracyRegion::kC:
        return "C";
    }

    return "?";
}

// The policy the scheme book's rows are measured under: the shipped Chebyshev
// route at the scheme the row names. That is the pair BoysEvalSchemeFits()
// reports, and the pair a call site reaches by naming one EvalPolicy.
template <boys::EvalScheme kScheme>
using SchemePolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme>;

// The policy the packing book's rows are measured under: the shipped route and
// the scheme the row names, at the orders axis. The axis is the only difference
// from SchemePolicy, which is the point of the row: the two policies evaluate
// the same stored fits, and the axis decides which of a call's values share a
// vector register.
template <boys::EvalScheme kScheme>
using OrdersPackPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

// The stored fit one lane names, summed by one scheme. This is the kernel the
// double single entries dispatch to, at the scheme they were compiled with.
template <boys::EvalScheme kScheme>
double FitValue(boys::EvalLane lane, int n, double x) {
    switch (lane)
    {
    case boys::EvalLane::kRegionA:
        return boys::detail::ChebyshevValue<kScheme>(n, x);
    case boys::EvalLane::kRegionB:
        return boys::detail::RegionBSeed<kScheme>(x);
    case boys::EvalLane::kExtendedBand:
        return boys::detail::RegionBExtendedSeed<kScheme>(x);
    }

    return 0.0;
}

// Whether an argument is inside the interval the fit is defined on. The
// per-order fits are defined on all of [0, x0); each seed is F_0 on its own
// interval, and the orders above it follow by the upward recursion whose
// amplification the band's and region B's own claims carry.
bool InFitDomain(boys::EvalLane lane, int n, double x) {
    switch (lane)
    {
    case boys::EvalLane::kRegionA:
        return x < boys::detail::kX0;
    case boys::EvalLane::kRegionB:
        return n == 0 && x >= boys::detail::kX0 && x < boys::detail::kX1;
    case boys::EvalLane::kExtendedBand:
        return n == 0 && x >= boys::detail::kExtendedBX0 && x < boys::detail::kX0;
    }

    return false;
}

// A value's significand to `figures` figures, as an integer: two published
// accuracy figures are compared at the precision the document states them at,
// which is the only precision at which a rounded measurement is a claim at all.
// It is an integer comparison so that the two sides cannot disagree over a
// rounding of the comparison itself.
long long SignificantInt(double v, int figures) {
    if (v == 0.0 || !std::isfinite(v))
    {
        return 0;
    }

    const double exponent = std::floor(std::log10(std::fabs(v)));
    const double scaled = std::fabs(v) / std::pow(10.0, exponent - (figures - 1));
    return static_cast<long long>(std::llround(scaled));
}

// ---------------------------------------------------------------------------
// The reference: a rectangular n x argument table, read once.
// ---------------------------------------------------------------------------

std::string Fmt(const char* format, ...)
{
    // Long enough for the longest evidence line: an evidence string that is
    // silently cut mid-word would read as a claim with no answer at its end.
    char buffer[2048];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

// One argument, every lane: what each entry returns beside the reference, so a
// reported failure can be reproduced and acted on rather than believed.
void RunProbe(const Reference& ref, int n, double x)
{
    if (n < 0 || n > boys::kMaxBoysOrder)
    {
        std::printf("gate: order %d is outside 0..%d\n", n, boys::kMaxBoysOrder);
        return;
    }

    double refV = std::numeric_limits<double>::quiet_NaN();
    double refF = refV;
    double ref16 = refV;
    double refB = refV;

    for (std::size_t i = 0; i < ref.count; ++i)
    {
        if (ref.x[i] == x)
        {
            const std::size_t k = ref.Index(n, i);
            refV = ref.v[k];
            refF = ref.vf[k];
            ref16 = ref.v16[k];
            refB = ref.vb[k];
            break;
        }
    }

    std::printf("probe: n=%d x=%.17g\n", n, x);
    std::printf("  %-24s %-24.17g%s\n",
                "reference F_n(x)",
                refV,
                std::isnan(refV) ? "   (argument is not on the reference grid)" : "");

    const auto row = [](const char* lane, double got, double want) {
        std::printf("  %-28s %-24.17g err=%-12.6g\n", lane, got, std::abs(got - want));
    };

    row("BoysSingle", boys::BoysSingle(n, x), refV);

    {
        std::array<double, 33> out{};
        boys::BoysAllOrders(n, x, out.data());
        row("BoysAllOrders[n]", out[static_cast<std::size_t>(n)], refV);
        boys::BoysFixedN(n, &x, out.data(), 1);
        row("BoysFixedN[n]", out[0], refV);
        boys::BoysAllN(n, &x, out.data(), 1);
        row("BoysAllN[n]", out[static_cast<std::size_t>(n)], refV);
        row("BoysSingle<64>", boys::BoysSingle<64.0>(n, x), refV);
    }

    {
        const float xf = static_cast<float>(x);
        std::array<float, 33> out{};
        row("BoysSingleF32", static_cast<double>(boys::BoysSingleF32(n, xf)), refF);
        boys::BoysAllOrdersF32(n, xf, out.data());
        row("BoysAllOrdersF32[n]", static_cast<double>(out[static_cast<std::size_t>(n)]), refF);

        const boys::F16 x16 = boys::F16(xf);
        const boys::Bf16 xb = boys::Bf16(xf);
        std::array<boys::F16, 33> out16{};
        std::array<boys::Bf16, 33> outb{};
        row("BoysSingleF16",
            static_cast<double>(static_cast<float>(boys::BoysSingleF16(n, x16))),
            ref16);
        boys::BoysAllOrdersF16(n, x16, out16.data());
        row("BoysAllOrdersF16[n]",
            static_cast<double>(static_cast<float>(out16[static_cast<std::size_t>(n)])),
            ref16);
        row("BoysSingleBf16",
            static_cast<double>(static_cast<float>(boys::BoysSingleBf16(n, xb))),
            refB);
        boys::BoysAllOrdersBf16(n, xb, outb.data());
        row("BoysAllOrdersBf16[n]",
            static_cast<double>(static_cast<float>(outb[static_cast<std::size_t>(n)])),
            refB);
    }

    if (!boys::BoysAvx2Available())
    {
        return;
    }

    {
        double out = 0.0;

        if (x < boys::detail::kX0)
        {
            boys::detail::BoysRegionASimd(n, &x, &out, 1);
            row("RegionASimd[count=1]", out, refV);
        } else if (x < boys::detail::kX1)
        {
            std::array<double, 33> plane{};
            boys::detail::BoysRegionBSimd(n, &x, plane.data(), 1);
            row("RegionBSimd[count=1]", plane[static_cast<std::size_t>(n)], refV);
        } else
        {
            boys::detail::BoysRegionCSimd(n, &x, &out, 1);
            row("RegionCSimd[count=1]", out, refV);
        }
    }

    const float xf = static_cast<float>(x);
    const boys::F16 x16 = boys::F16(xf);
    const boys::Bf16 xb = boys::Bf16(xf);

    // The half-I/O kernels at count = 1 (the scalar tail) and count = 8 (the
    // vector path): a difference between the two shapes is exactly what the
    // sweep cannot show, since the sweep only reports the maximum.
    const auto halfRow = [&](const char* lane, bool isBf16, std::size_t count) {
        const double want = isBf16 ? refB : ref16;
        double got = 0.0;

        if (isBf16)
        {
            std::vector<boys::Bf16> xs(count, xb);
            std::vector<boys::Bf16> out(count * (static_cast<std::size_t>(n) + 1));

            if (x < boys::detail::kExtendedBX0)
            {
                boys::detail::BoysRegionASimdBf16(n, xs.data(), out.data(), count);
                got = static_cast<double>(static_cast<float>(out[0]));
            } else if (x < boys::detail::kX1)
            {
                boys::detail::BoysRegionBSimdBf16(n, xs.data(), out.data(), count);
                got = static_cast<double>(
                    static_cast<float>(out[static_cast<std::size_t>(n) * count]));
            } else
            {
                boys::detail::BoysRegionCSimdBf16(n, xs.data(), out.data(), count);
                got = static_cast<double>(static_cast<float>(out[0]));
            }
        } else
        {
            std::vector<boys::F16> xs(count, x16);
            std::vector<boys::F16> out(count * (static_cast<std::size_t>(n) + 1));

            if (x < boys::detail::kExtendedBX0)
            {
                boys::detail::BoysRegionASimdF16(n, xs.data(), out.data(), count);
                got = static_cast<double>(static_cast<float>(out[0]));
            } else if (x < boys::detail::kX1)
            {
                boys::detail::BoysRegionBSimdF16(n, xs.data(), out.data(), count);
                got = static_cast<double>(
                    static_cast<float>(out[static_cast<std::size_t>(n) * count]));
            } else
            {
                boys::detail::BoysRegionCSimdF16(n, xs.data(), out.data(), count);
                got = static_cast<double>(static_cast<float>(out[0]));
            }
        }

        std::printf("  %-28s %-24.17g err=%-12.6g\n", lane, got, std::abs(got - want));
    };

    halfRow("RegionSimdF16[count=1]", false, 1);
    halfRow("RegionSimdF16[count=8]", false, 8);
    halfRow("RegionSimdBf16[count=1]", true, 1);
    halfRow("RegionSimdBf16[count=8]", true, 8);
}

// ---------------------------------------------------------------------------
// The reference, scanned: the published range statements are all of the shape
// "the value leaves the format's normal range at x", so they are read back out
// of the same table the errors are measured against rather than asserted.
// ---------------------------------------------------------------------------

double LargestXAbove(const Reference& ref, int order, double threshold)
{
    double best = -1.0;

    for (std::size_t i = 0; i < ref.count; ++i)
    {
        const double x = ref.x[i];

        if (x < boys::detail::kX1)
        {
            continue;
        }

        if (std::fabs(ref.v[ref.Index(order, i)]) >= threshold && x > best)
        {
            best = x;
        }
    }

    return best;
}

// The largest argument at which the ladder an order-n call walks (orders 0..n,
// the seed included) still fits the field a per-order power-of-two scale can
// place it in: scaling moves a value inside a fixed exponent range and cannot
// widen it, so this is the ceiling on how far any such scale carries an order.
double LargestXSparsable(const Reference& ref, int order, double field)
{
    double best = -1.0;

    for (std::size_t i = 0; i < ref.count; ++i)
    {
        const double x = ref.x[i];

        if (x < boys::detail::kX1)
        {
            continue;
        }

        const double top = ref.v[ref.Index(0, i)];
        const double bottom = ref.v[ref.Index(order, i)];

        if (top > 0.0 && bottom > 0.0 && top / bottom <= field && x > best)
        {
            best = x;
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// The documented claims, and the verdict this run gives each one.
// ---------------------------------------------------------------------------
// Every published claim ends this run with exactly one of:
//
//   Verified       measured here, and met at a point that carries signal;
//   Exceeded       measured here, and the delivered error is outside it;
//   Vacuous        measured here, and every point that meets it is a point
//                  where the documented bound is at least the function's own
//                  magnitude - the pass is the format's floor, not the lane's,
//                  so it is counted apart from the verified ones;
//   EvidenceAbsent the artifact the claim names is not in this tree, so this
//                  revision cannot check it. Counted apart again, and never
//                  as a pass.
enum class Verdict { Verified, MetOverDomain, Exceeded, Vacuous, EvidenceAbsent };

// A domain-scoped claim - one the document itself restricts to a stated set of
// regions, orders or arguments - is met when it holds over that domain, and it
// is recorded as met over the stated domain rather than as verified outright,
// with the domain printed beside the verdict. It is a pass and it is a
// different thing from the two verdicts below it: a restriction the document
// states is not missing evidence, and it must not share a verdict with one.
bool IsMet(Verdict v)
{
    return v == Verdict::Verified || v == Verdict::MetOverDomain;
}

const char* VerdictName(Verdict v)
{
    switch (v)
    {
    case Verdict::Verified:
        return "verified at this revision";
    case Verdict::MetOverDomain:
        return "met over the stated domain";
    case Verdict::Exceeded:
        return "EXCEEDED";
    case Verdict::Vacuous:
        return "vacuous only";
    case Verdict::EvidenceAbsent:
        return "evidence absent from the tree";
    }

    return "?";
}

// What would have to be true for the check behind a row to fail. Rows that
// know their own failure mode print it; the rest print the one their class
// has, stated in full at the head of this book. A verdict that cannot say
// this is not a check, so every row prints one of the two.
const char* ClassFalsifier(Verdict v)
{
    switch (v)
    {
    case Verdict::Verified:
    case Verdict::MetOverDomain:
        return "class falsifier - a measured row: the sweep would have to deliver a cell "
               "over its bound (the row prints its cells, its worst cell and its failure "
               "count, so a non-zero failure count is what to look for), or cover none of "
               "the domain it names; a tree-command row: the named command would have to "
               "print a different number; a prose row: the document's wording would have "
               "to change to what the measurement contradicts";
    case Verdict::Exceeded:
        return "none needed - the claim is already exceeded, and the numbers that do it "
               "are printed here";
    case Verdict::Vacuous:
        return "this row would have to become a bound that binds: it passes only where no "
               "cell is over budget, and here every counted cell is one where the bound "
               "cannot be reached";
    case Verdict::EvidenceAbsent:
        return "the artifact this row names would have to enter the tree and be measured";
    }

    return "?";
}

// What a claim's verdict is a verdict about.
//
// The distinction is the one a model cannot cross. A claim whose arithmetic ran
// on the machine earned its verdict from a measurement of that arithmetic. A
// claim that rests on a *model* of an accumulator - simulated in software
// because no machine here has the hardware - earned its verdict from a
// measurement of the model, and a model is not a card: a tensor core's fused
// sum truncates and aligns its addends where the model rounds, which makes a
// card worse than the model and never better. A green model-derived row is
// therefore not evidence that a card is inside the bound, and such a row is
// counted apart from the certified total for that reason and no other.
enum class Evidence {
    kHardware, ///< the arithmetic the claim names ran on this machine
    kModel,    ///< arithmetic on a software model of an accumulator
};

struct DocClaim {
    std::string id;
    std::string statement;
    std::string source;
    std::string domain;
    Verdict verdict = Verdict::EvidenceAbsent;
    Evidence kind = Evidence::kHardware;
    std::string evidence;
    std::string falsifier;
};

// The verdict a measured claim slot earns.
Verdict FromAccum(const Accum& a)
{
    if (a.failures > 0)
    {
        return Verdict::Exceeded;
    }

    if (a.points > 0 && a.points == a.vacuous)
    {
        return Verdict::Vacuous;
    }

    return Verdict::Verified;
}

// Region B's amplification A_B(n) = prod_{j=1..n} (j + 1/2) / x0^n, the
// quantity the seed-sizing argument bounds. It is evaluated in double-double
// because the excess it measures at the top supported order is 1.8e-17, which
// binary64 cannot represent beside 1.0 at all: a double-precision evaluation
// would confirm "at most one" by rounding, which is exactly how a claim like
// this survives. The accumulated error below is ~1e-30 relative, thirteen
// orders under the excess.
struct DD
{
    double hi = 0.0;
    double lo = 0.0;
};

DD TwoSumDD(double a, double b)
{
    const double s = a + b;
    const double bv = s - a;
    return {s, (a - (s - bv)) + (b - bv)};
}

DD MulDD(DD a, DD b)
{
    const double p = a.hi * b.hi;
    const double e = std::fma(a.hi, b.hi, -p);
    return TwoSumDD(p, e + a.hi * b.lo + a.lo * b.hi);
}

DD SubDD(DD a, DD b)
{
    const DD t = TwoSumDD(a.hi, -b.hi);
    return TwoSumDD(t.hi, t.lo + a.lo - b.lo);
}

DD DivDD(DD a, DD b)
{
    const double q1 = a.hi / b.hi;
    const DD r = SubDD(a, MulDD(b, DD{q1, 0.0}));
    return TwoSumDD(q1, (r.hi + r.lo) / b.hi);
}

} // namespace

int main(int argc, char** argv) {
    std::string reference = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
    bool perOrder = false;
    bool strict = false;
    int probeN = -1;
    double probeX = 0.0;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];

        if (arg == "--per-order")
        {
            perOrder = true;
        } else if (arg == "--strict")
        {
            strict = true;
        } else if (arg == "--probe" && i + 2 < argc)
        {
            probeN = std::atoi(argv[++i]);
            probeX = std::strtod(argv[++i], nullptr);
        } else
        {
            reference = arg;
        }
    }

    const Reference ref = LoadReference(reference);

    if (probeN >= 0)
    {
        RunProbe(ref, probeN, probeX);
        return 0;
    }

    // Claim slots, in report order.
    const int kSingleA = AddClaim("double single", "A", kBoundSingleA);
    const int kSingleBand = AddClaim("double single", "band", kBoundSingleBand);
    const int kSingleB = AddClaim("double single", "B", kBoundSingleB);
    const int kSingleC = AddClaim("double single", "C", kBoundSingleC);
    const int kOrders = AddClaim("double batch", "all-orders", kBoundDoubleBatch);
    const int kFixedN = AddClaim("double batch", "fixed-n", kBoundDoubleBatch);
    const int kAllN = AddClaim("double batch", "all-n", kBoundDoubleBatch);
    const int kFloatSingle = AddClaim("float single", "all", kBoundFloat);
    const int kFloatOrders = AddClaim("float batch", "all", kBoundFloat);
    const int kF16Single = AddClaim("fp16 store-half", "single", kBoundHalfBase);
    const int kF16Orders = AddClaim("fp16 store-half", "batch", kBoundHalfBase);
    const int kBf16Single = AddClaim("bf16 store-half", "single", kBoundHalfBase);
    const int kBf16Orders = AddClaim("bf16 store-half", "batch", kBoundHalfBase);
    const int kF16PackedA = AddClaim("fp16 simd-half", "A", kBoundHalfBase);
    const int kF16PackedB = AddClaim("fp16 simd-half", "B", kBoundHalfBase);
    const int kF16PackedC = AddClaim("fp16 simd-half", "C", kBoundHalfBase);
    const int kBf16PackedA = AddClaim("bf16 simd-half", "A", kBoundHalfBase);
    const int kBf16PackedB = AddClaim("bf16 simd-half", "B", kBoundHalfBase);
    const int kBf16PackedC = AddClaim("bf16 simd-half", "C", kBoundHalfBase);
    // The withdrawn header claim, measured for the record and not judged:
    // "the double single lane holds 1e-15 across every x < x0". The last
    // argument is what makes the slot's exceptions visible in the report
    // rather than only here: no row carries this slot's verdict, so it is
    // exceeded on every target that has ever run this gate and the run is
    // still green.
    const int kHeaderA = AddClaim("double single", "header x<x0", kWithdrawnHeaderABound, false);
    // The relaxed rungs: the documented budget at a rung is m times the m = 1
    // budget, and it differs per region for the single lane, so these slots
    // carry no single base bound.
    const int kSingle64 = AddClaim("double single m=64", "A..C", 0.0);
    const int kSingle65536 = AddClaim("double single m=65536", "A..C", 0.0);
    const int kBatch64 = AddClaim("double batch m=64", "all", 64.0 * kBoundDoubleBatch);
    const int kBatch65536 = AddClaim("double batch m=65536", "all", 65536.0 * kBoundDoubleBatch);
    const int kFloat64 = AddClaim("float single m=64", "all", 64.0 * kBoundFloat);
    const int kFloat65536 = AddClaim("float single m=65536", "all", 65536.0 * kBoundFloat);
    const int kHalf64 = AddClaim("fp16 store-half m=64", "all", 64.0 * kBoundHalfBase);
    const int kHalf65536 = AddClaim("fp16 store-half m=65536", "all", 65536.0 * kBoundHalfBase);
    // The native packed half lane: region C only, and its bound is in ULP of the
    // returned value rather than a region budget, so these slots carry no single
    // base bound (baseBound is display only, as for the relaxed rungs).
    const int kNativeHalf2 = AddClaim("native packed half", "C", 0.0);
    const int kNativeHalfBatch = AddClaim("native packed half batch", "C", 0.0);
    // The region-A transform lane's modes: one slot per mode, each carrying the
    // mode's published bound over region A - both bands, every order, every
    // argument of the band. The slots exist in every revision - the rows below
    // read them - and carry no points where the tree has no
    // boys/boys_transform.hpp.
    const int kTransformFp64 = AddClaim("transform kFp64", "A", kTransformFp64Bound);
    const int kTransformTf32x3 = AddClaim("transform kTf32x3", "A", kTransformSplitBound);
    const int kTransformBf16x6 = AddClaim("transform kBf16x6", "A", kTransformSplitBound);
    // The same lane at a relaxed multiplier: one rung per mode, where the
    // documented bound is m*1e-15 for fp64 and m*1e-15 + 2.5e-7 for the two
    // split modes.
    const int kTransformRung1024 = AddClaim("transform kFp64 m=1024", "A",
                                            kTransformRung * kTransformFp64Bound);
    const int kTransformTf32x3Rung = AddClaim("transform kTf32x3 m=1024", "A",
                                              kTransformRung * kTransformFp64Bound +
                                                  kTransformSplitFloor);
    const int kTransformBf16x6Rung = AddClaim("transform kBf16x6 m=1024", "A",
                                              kTransformRung * kTransformFp64Bound +
                                                  kTransformSplitFloor);
    // The run-time accuracy tier's rungs: one slot per multiplier the tier can
    // select. The slots exist in every revision - the rows below read them -
    // and carry no points where the tree has no BoysAllOrdersAtTier.
    std::array<int, 7> kTierRung{};

    for (int r = 0; r < 7; ++r)
    {
        constexpr double kRungMultiplier[7]{1.0, 64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0};
        kTierRung[static_cast<std::size_t>(r)] =
            AddClaim(TierRungLabel(r), "A..C", kRungMultiplier[r] * kBoundDoubleBatch);
    }

    const std::vector<int> singleClaims = {kSingleA, kSingleBand, kSingleB, kSingleC};

    std::printf("boys accuracy gate\n");
    std::printf("reference    : %s\n", reference.c_str());
    std::printf("               %zu orders x %zu arguments = %zu points\n",
                ref.orderCount,
                ref.count,
                ref.orderCount * ref.count);
    std::printf("boundaries   : x_ext=%.17g  x0=%.17g  x1=%.17g\n",
                boys::detail::kExtendedBX0,
                boys::detail::kX0,
                boys::detail::kX1);
    std::printf("tiers        : n<=4 %.17g | n<=8 %.17g | n<=16 %.17g | n<=32 %.17g\n",
                boys::detail::kTierThresholds[0],
                boys::detail::kTierThresholds[5],
                boys::detail::kTierThresholds[9],
                boys::detail::kTierThresholds[17]);
    std::printf("AVX2 tier    : %s\n", boys::BoysAvx2Available() ? "present" : "absent");

    // The arithmetic every scalar lane above was computed at. A fused step is
    // an instruction where the target has one and a call into the C runtime
    // where it does not, and the separate route replaces the call with two
    // roundings — so the route names which of the two arithmetics the numbers
    // below belong to, and a figure read off this output without it is a
    // figure whose arithmetic is unstated.
    {
        const std::span<const boys::backend::BackendInfo> backends =
            boys::backend::BoysBackends();

        for (const boys::backend::BackendInfo& info : backends)
        {
            if (std::string_view(info.name).rfind("scalar-", 0) == 0)
            {
                std::printf("muladd route : %s %s (bare product-plus-add %s here)\n",
                            info.name,
                            boys::backend::MulAddRouteName(info.route),
                            info.contracts ? "contracts" : "does not contract");
            }
        }
    }

    std::printf("verdict      : %s\n",
                strict ? "strict (--strict: the default verdict, stated)"
                       : "strict (any claim not verified at this revision fails the gate)");

    const std::size_t count = ref.count;
    const int nmax = boys::kMaxBoysOrder;

    // ---- double single -----------------------------------------------------
    for (int n = 0; n <= nmax; ++n)
    {
        for (std::size_t i = 0; i < count; ++i)
        {
            const double x = ref.x[i];
            const std::size_t k = ref.Index(n, i);
            const double got = boys::BoysSingle(n, x);
            const int slot = singleClaims[static_cast<std::size_t>(SingleClaim(x))];

            Measure(slot,
                    n,
                    x,
                    got,
                    ref.v[k],
                    ref.decade[k],
                    Claims()[static_cast<std::size_t>(slot)].baseBound,
                    Unrepresentable(got, -1022));

            if (x < boys::detail::kX0)
            {
                Measure(kHeaderA,
                        n,
                        x,
                        got,
                        ref.v[k],
                        ref.decade[k],
                        kWithdrawnHeaderABound,
                        Unrepresentable(got, -1022));
            }
        }
    }

    // ---- the region-A transform lane ---------------------------------------
    // The double table's two shared bands as one matrix product per band, in
    // each of the published modes, measured by the same instrument as every
    // other lane: this gate's own committed reference, at the same arguments,
    // every order, against the mode's published bound. The entry takes one
    // band's arguments and neither sorts nor classifies them - the band a call
    // is handed is its precondition - so the caller groups them here, which is
    // also what keeps a cell from being measured against the other band's fit.
#ifdef BOYS_GATE_TRANSFORM
    {
        for (int band = 0; band < 2; ++band)
        {
            const boys::RegionABand which =
                (band == 0) ? boys::RegionABand::kA1 : boys::RegionABand::kA2;
            std::vector<double> xs;
            std::vector<std::size_t> arg;

            for (std::size_t i = 0; i < count; ++i)
            {
                const bool inBand = (band == 0)
                                        ? (ref.x[i] < boys::kRegionA1Edge)
                                        : (ref.x[i] >= boys::kRegionA1Edge &&
                                           ref.x[i] < boys::kRegionAEnd);

                if (inBand)
                {
                    xs.push_back(ref.x[i]);
                    arg.push_back(i);
                }
            }

            if (xs.empty())
            {
                continue;
            }

            const std::size_t batch = xs.size();
            std::vector<double> out(batch * static_cast<std::size_t>(nmax + 1));

            // One mode's product, then every cell it wrote, measured against
            // the reference at the argument that cell belongs to.
            const auto sweep = [&](int slot, auto run) {
                run();

                for (int n = 0; n <= nmax; ++n)
                {
                    for (std::size_t j = 0; j < batch; ++j)
                    {
                        const double got = out[static_cast<std::size_t>(n) * batch + j];
                        const std::size_t k = ref.Index(n, arg[j]);

                        Measure(slot,
                                n,
                                xs[j],
                                got,
                                ref.v[k],
                                ref.decade[k],
                                Claims()[static_cast<std::size_t>(slot)].baseBound,
                                Unrepresentable(got, -1022));
                    }
                }
            };

            sweep(kTransformFp64, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64>(
                    which, nmax, xs.data(), out.data(), batch);
            });
            sweep(kTransformTf32x3, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kTf32x3>(
                    which, nmax, xs.data(), out.data(), batch);
            });
            sweep(kTransformBf16x6, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kBf16x6>(
                    which, nmax, xs.data(), out.data(), batch);
            });
            sweep(kTransformRung1024, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kFp64, kTransformRung>(
                    which, nmax, xs.data(), out.data(), batch);
            });
            sweep(kTransformTf32x3Rung, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kTf32x3, kTransformRung>(
                    which, nmax, xs.data(), out.data(), batch);
            });
            sweep(kTransformBf16x6Rung, [&] {
                boys::BoysRegionAProduct<boys::ProductMode::kBf16x6, kTransformRung>(
                    which, nmax, xs.data(), out.data(), batch);
            });
        }
    }
#endif

    // ---- double batch: the per-argument all-orders entry -------------------
    {
        std::array<double, 33> out{};

        for (std::size_t i = 0; i < count; ++i)
        {
            boys::BoysAllOrders(nmax, ref.x[i], out.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const double got = out[static_cast<std::size_t>(n)];
                Measure(kOrders,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }
    }

    // ---- double batch: the fixed-order vector entry ------------------------
    {
        std::vector<double> out(count);

        for (int n = 0; n <= nmax; ++n)
        {
            boys::BoysFixedN(n, ref.x.data(), out.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double got = out[i];
                Measure(kFixedN,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }
    }

    // ---- double batch: the many-argument all-orders entries ----------------
    {
        std::vector<double> out(count * static_cast<std::size_t>(nmax + 1));
        std::vector<double> sorted = ref.x;
        std::sort(sorted.begin(), sorted.end());

        boys::BoysAllN(nmax, ref.x.data(), out.data(), count);

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double got = out[ref.Index(n, i)];
                Measure(kAllN,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }

        // The sorted overload: the same arguments in non-decreasing order.
        std::vector<std::size_t> perm(count);
        std::vector<double> allOut(count * static_cast<std::size_t>(nmax + 1));

        for (std::size_t i = 0; i < count; ++i)
        {
            perm[i] = i;
        }

        std::sort(perm.begin(), perm.end(), [&](std::size_t a, std::size_t b) {
            return ref.x[a] < ref.x[b];
        });

        boys::BoysAllN(nmax, sorted.data(), allOut.data(), count, boys::BoysSortedArgs{});

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t j = 0; j < count; ++j)
            {
                const std::size_t i = perm[j];
                const double got = allOut[ref.Index(n, j)];
                Measure(kAllN,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }
    }

    // ---- the certified fit routes -----------------------------------------
    // A route is a way of serving a region rather than a lane the library
    // always serves, so it is measured the way a lane is but counted apart from
    // the lanes: adding one must not move a statement about a lane's coverage.
    // Four things are measured here. What each route's own fit delivers over
    // the interval its row of BoysFitRoutes states. What a caller receives from
    // the selector, at every order, over the region the route serves. That
    // naming a route changes nothing outside the interval that route covers -
    // and that naming the default route is the default entry, value for value,
    // a value outside the enumeration included.
    std::vector<int> routeSeedClaim;
    std::vector<int> routeLaneClaim;

    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        routeSeedClaim.push_back(AddRouteClaim(row.name, RegionName(row.region), row.bound));
        routeLaneClaim.push_back(
            AddRouteClaim(row.name, RegionName(row.region), kBoundDoubleBatch));
    }

    // The report's own rows against the tables the kernel evaluates: a row's
    // stored count is what the generated header holds for that route, its
    // interval is the interval its fit was made over, and its delivered figure
    // may not sit below what this gate measures for the same route.
    std::size_t routeStoredMismatch = 0;
    std::size_t routeTableMismatch = 0;
    std::size_t routeDomainMismatch = 0;
    std::size_t routeServeMismatch = 0;
    std::size_t routePromiseMismatch = 0;
    std::size_t routeDisagreement = 0;
    std::size_t routeNames = 0;
    std::size_t routeCellsInside = 0;
    std::size_t routeDiffOutside = 0;
    std::size_t routeDiffInside = 0;
    std::size_t routeDefaultDiff = 0;
    std::size_t routeUnknownDiff = 0;
    // Filled by the carriage measurement below and read by the route book's
    // claims: the run-time selector's pairs, and the entries that name a route
    // without answering it.
    std::size_t routeRuntimePairs = 0;
    std::size_t routeRuntimeDiff = 0;
    std::size_t routeCarriageMissed = 0;
    std::size_t routeCarriageNamed = 0;
    std::size_t routeCarriageControls = 0;

    // The route held to its own bar through the entries a consumer actually
    // calls with a many-argument or fixed-order shape. The route book's own
    // rows measure the per-argument entries, and a carriage row says only that
    // the other two answer the route; what a consumer needs is the figure, so
    // it is measured here and published as its own claim.
    std::size_t routeEntryCells = 0;
    std::size_t routeEntryOver = 0;
    double routeEntryWorst = 0.0;
    int routeEntryWorstN = -1;
    double routeEntryWorstX = 0.0;

    // One row per entry a named route was measured through: the arguments the
    // route's selector takes over, and how many of those the entry answered with
    // a different value once the route was named.
    struct RouteCarriage {
        const char* entry = "";
        std::size_t cells = 0;
        std::size_t differ = 0;
    };

    std::vector<RouteCarriage> routeCarriage;

    const auto routeCarriedBy = [&routeCarriage](const char* entry,
                                                 std::size_t cells,
                                                 std::size_t differ) {
        RouteCarriage c;
        c.entry = entry;
        c.cells = cells;
        c.differ = differ;
        routeCarriage.push_back(c);
    };

    {
        std::array<double, 33> out{};
        std::array<double, 33> plain{};
        std::array<double, 33> selected{};

        for (std::size_t i = 0; i < count; ++i)
        {
            const double x = ref.x[i];
            bool covered = false;
            bool served = false;

            for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
            {
                if (x >= row.lo && x < row.hi)
                {
                    covered = true;
                }

                // The domain a route's selector takes over, which is where
                // naming it may change a value. A row whose fit reaches further
                // than its selector claims the narrower interval here.
                if (x >= std::max(row.lo, row.servesFrom) && x < row.hi)
                {
                    served = true;
                }
            }

            boys::BoysAllOrders(nmax, x, plain.data());
            boys::BoysAllOrdersWithRoute(boys::FitRoute::kChebyshev, nmax, x, selected.data());
            boys::BoysAllOrdersWithRoute(boys::FitRoute::kRationalMinimax, nmax, x, out.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const auto j = static_cast<std::size_t>(n);

                if (selected[j] != plain[j])
                {
                    ++routeDefaultDiff;
                }

                if (out[j] != plain[j])
                {
                    if (served)
                    {
                        ++routeDiffInside;
                    } else
                    {
                        ++routeDiffOutside;
                    }
                }
            }

            if (!covered)
            {
                continue;
            }

            routeCellsInside += static_cast<std::size_t>(nmax + 1);

            for (std::size_t r = 0; r < boys::BoysFitRoutes().size(); ++r)
            {
                const boys::FitRouteInfo& row = boys::BoysFitRoutes()[r];

                if (!(x >= row.lo && x < row.hi))
                {
                    continue;
                }

                if (row.region == boys::AccuracyRegion::kA)
                {
                    // Region A's fit is one piece per order rather than one
                    // seed, so the route's own fit is measured piece by piece:
                    // what the table holds for each order, at the row's bar.
                    // Both readings are one body at the two fit policies, so
                    // what this row compares is the two fits, not two code
                    // paths that ought to agree.
                    for (int n = 0; n <= nmax; ++n)
                    {
                        const double got =
                            (row.route == boys::FitRoute::kRationalMinimax)
                                ? boys::detail::RegionAValue<boys::detail::RationalFit>(n, x)
                                : boys::detail::RegionAValue<
                                      boys::detail::ChebyshevFit<boys::kDefaultEvalScheme>>(n, x);
                        const std::size_t k = ref.Index(n, i);

                        MeasureInto(RouteClaims(),
                                    routeSeedClaim[r],
                                    n,
                                    x,
                                    got,
                                    ref.v[k],
                                    ref.decade[k],
                                    row.bound,
                                    Unrepresentable(got, -1022));
                    }
                } else
                {
                    boys::BoysAllOrdersWithRoute(row.route, 0, x, out.data());
                    MeasureInto(RouteClaims(),
                                routeSeedClaim[r],
                                0,
                                x,
                                out[0],
                                ref.v[ref.Index(0, i)],
                                ref.decade[ref.Index(0, i)],
                                row.bound,
                                Unrepresentable(out[0], -1022));
                }

                boys::BoysAllOrdersWithRoute(row.route, nmax, x, out.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double got = out[static_cast<std::size_t>(n)];

                    MeasureInto(RouteClaims(),
                                routeLaneClaim[r],
                                n,
                                x,
                                got,
                                ref.v[k],
                                ref.decade[k],
                                kBoundDoubleBatch,
                                Unrepresentable(got, -1022));
                }
            }
        }

        // A value the enumeration does not name is not a route. The contract is
        // that it evaluates at the default, so a caller is never handed a fit
        // they did not ask for.
        for (std::size_t i = 0; i < count; i += 7)
        {
            boys::BoysAllOrders(nmax, ref.x[i], plain.data());
            boys::BoysAllOrdersWithRoute(static_cast<boys::FitRoute>(99), nmax, ref.x[i],
                                         out.data());

            for (int n = 0; n <= nmax; ++n)
            {
                if (out[static_cast<std::size_t>(n)] != plain[static_cast<std::size_t>(n)])
                {
                    ++routeUnknownDiff;
                }
            }
        }

        routeNames = boys::BoysFitRoutes().size();

        // ---- the carriage of a named route by each entry --------------------
        // The accuracy rows above cannot show that the route was read. The two
        // routes hold the same bar over the same intervals, so an entry that
        // evaluated the other route's fits would pass every one of them and
        // still be answering a selection it never opened. What is measured here
        // is the other question: over the arguments a route's own selector takes
        // over, does naming that route on the entry change any value at all? An
        // entry that names a route and returns the default entry's values bit
        // for bit over every served argument is an entry the selection does not
        // reach, whatever its accuracy says.
        //
        // The default route is read the same way, as the control: there the
        // count is required to be zero rather than more than zero, and the pair
        // of counts is what separates "this entry carries the route it was
        // given" from "this entry carries some route".
        {
            // The arguments a route's selector takes over - where naming a route
            // may change a value at all.
            std::vector<std::size_t> servedArgs;

            for (std::size_t i = 0; i < count; ++i)
            {
                for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
                {
                    if (ref.x[i] >= std::max(row.lo, row.servesFrom) && ref.x[i] < row.hi)
                    {
                        servedArgs.push_back(i);
                        break;
                    }
                }
            }

            // The same arguments in non-decreasing order, with the permutation
            // back to their own indices, for the sorted overload.
            std::vector<std::size_t> perm(servedArgs.size());

            for (std::size_t j = 0; j < perm.size(); ++j)
            {
                perm[j] = servedArgs[j];
            }

            std::sort(perm.begin(), perm.end(),
                      [&](std::size_t a, std::size_t b) { return ref.x[a] < ref.x[b]; });

            std::vector<double> permArgs(perm.size());

            for (std::size_t j = 0; j < perm.size(); ++j)
            {
                permArgs[j] = ref.x[perm[j]];
            }

            const auto countSingle = [&]<boys::FitRoute kRoute>() {
                std::size_t cells = 0;
                std::size_t differ = 0;

                for (const std::size_t i : servedArgs)
                {
                    for (int n = 0; n <= nmax; ++n)
                    {
                        ++cells;
                        const double a = boys::BoysSingle<1.0, boys::EvalPolicy<kRoute>>(n, ref.x[i]);
                        const double b = boys::BoysSingle<1.0, boys::EvalPolicy<>>(n, ref.x[i]);

                        if (std::memcmp(&a, &b, sizeof(double)) != 0)
                        {
                            ++differ;
                        }
                    }
                }

                return std::pair<std::size_t, std::size_t>(cells, differ);
            };

            const auto countOrders = [&]<boys::FitRoute kRoute>() {
                std::size_t cells = 0;
                std::size_t differ = 0;

                for (const std::size_t i : servedArgs)
                {
                    std::array<double, 33> a{};
                    std::array<double, 33> b{};
                    boys::BoysAllOrders<1.0, boys::EvalPolicy<kRoute>>(nmax, ref.x[i], a.data());
                    boys::BoysAllOrders<1.0, boys::EvalPolicy<>>(nmax, ref.x[i], b.data());

                    for (int n = 0; n <= nmax; ++n)
                    {
                        ++cells;

                        if (std::memcmp(&a[static_cast<std::size_t>(n)],
                                        &b[static_cast<std::size_t>(n)],
                                        sizeof(double)) != 0)
                        {
                            ++differ;
                        }
                    }
                }

                return std::pair<std::size_t, std::size_t>(cells, differ);
            };

            const auto single = countSingle.template operator()<boys::FitRoute::kRationalMinimax>();
            routeCarriedBy("BoysSingle", single.first, single.second);
            const auto orders = countOrders.template operator()<boys::FitRoute::kRationalMinimax>();
            routeCarriedBy("BoysAllOrders", orders.first, orders.second);

            // The many-argument and fixed-order entries carry the route too, on
            // the shapes that take their fit from the policy: the plane entry's
            // per-argument path and the fixed-order entry's per-argument call.
            // Measured the same way as the two rows above - the values a call
            // returns with the route named, against the same call with none - so
            // a carriage claim here is a difference in the values and not a
            // sentence about the surface.
            const auto countPlane = [&]<boys::FitRoute kRoute>() {
                std::size_t cells = 0;
                std::size_t differ = 0;
                const std::size_t planeSize =
                    permArgs.size() * (static_cast<std::size_t>(nmax) + 1);
                std::vector<double> a(planeSize);
                std::vector<double> b(planeSize);
                boys::BoysAllN<1.0, boys::EvalPolicy<kRoute>>(
                    nmax, permArgs.data(), a.data(), permArgs.size());
                boys::BoysAllN<1.0, boys::EvalPolicy<>>(
                    nmax, permArgs.data(), b.data(), permArgs.size());

                for (std::size_t k = 0; k < planeSize; ++k)
                {
                    ++cells;

                    if (std::memcmp(&a[k], &b[k], sizeof(double)) != 0)
                    {
                        ++differ;
                    }
                }

                return std::pair<std::size_t, std::size_t>(cells, differ);
            };

            const auto countFixed = [&]<boys::FitRoute kRoute>() {
                std::size_t cells = 0;
                std::size_t differ = 0;

                for (const std::size_t i : servedArgs)
                {
                    for (int n = 0; n <= nmax; ++n)
                    {
                        ++cells;
                        double a = 0.0;
                        double b = 0.0;
                        boys::BoysFixedN<1.0, boys::EvalPolicy<kRoute>>(n, &ref.x[i], &a, 1);
                        boys::BoysFixedN<1.0, boys::EvalPolicy<>>(n, &ref.x[i], &b, 1);

                        if (std::memcmp(&a, &b, sizeof(double)) != 0)
                        {
                            ++differ;
                        }
                    }
                }

                return std::pair<std::size_t, std::size_t>(cells, differ);
            };

            const auto plane = countPlane.template operator()<boys::FitRoute::kRationalMinimax>();
            routeCarriedBy("BoysAllN", plane.first, plane.second);
            const auto fixed = countFixed.template operator()<boys::FitRoute::kRationalMinimax>();
            routeCarriedBy("BoysFixedN", fixed.first, fixed.second);

            const auto singleDefault =
                countSingle.template operator()<boys::FitRoute::kChebyshev>();
            routeCarriedBy("BoysSingle, the default route named",
                           singleDefault.first,
                           singleDefault.second);
            const auto ordersDefault = countOrders.template operator()<boys::FitRoute::kChebyshev>();
            routeCarriedBy("BoysAllOrders, the default route named",
                           ordersDefault.first,
                           ordersDefault.second);
            const auto planeDefault =
                countPlane.template operator()<boys::FitRoute::kChebyshev>();
            routeCarriedBy("BoysAllN, the default route named",
                           planeDefault.first,
                           planeDefault.second);
            const auto fixedDefault =
                countFixed.template operator()<boys::FitRoute::kChebyshev>();
            routeCarriedBy("BoysFixedN, the default route named",
                           fixedDefault.first,
                           fixedDefault.second);
        }

        // ---- the route's bar through the entries that carry it --------------
        // The bar is the route's own for the region the argument falls in, read
        // off BoysFitRoutes rather than repeated here, so a row whose bar moved
        // would move this one with it. Both entries are swept over the whole
        // committed grid: the plane entry in one call, the fixed-order entry one
        // order at a time over the same arguments.
        {
            const auto rationalBar = [](boys::AccuracyRegion region) {
                for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
                {
                    if (row.route == boys::FitRoute::kRationalMinimax && row.region == region)
                    {
                        return row.bound;
                    }
                }

                return kBoundDoubleBatch;
            };

            const auto judgeEntryCell = [&](int n, double x, double got, std::size_t k) {
                const double bar = (x < boys::detail::kX0)   ? rationalBar(boys::AccuracyRegion::kA)
                                   : (x < boys::detail::kX1) ? rationalBar(boys::AccuracyRegion::kB)
                                                             : kBoundDoubleBatch;
                const double err = std::abs(got - ref.v[k]);
                ++routeEntryCells;

                if (err > bar)
                {
                    ++routeEntryOver;
                }

                if (err > routeEntryWorst)
                {
                    routeEntryWorst = err;
                    routeEntryWorstN = n;
                    routeEntryWorstX = x;
                }
            };

            std::vector<double> planes(count * (static_cast<std::size_t>(nmax) + 1));
            boys::BoysAllN<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
                nmax, ref.x.data(), planes.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                for (int n = 0; n <= nmax; ++n)
                {
                    judgeEntryCell(n,
                                   ref.x[i],
                                   planes[static_cast<std::size_t>(n) * count + i],
                                   ref.Index(n, i));
                }
            }

            std::vector<double> columns(count * (static_cast<std::size_t>(nmax) + 1));

            for (int n = 0; n <= nmax; ++n)
            {
                boys::BoysFixedN<1.0, boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
                    n, ref.x.data(), columns.data() + static_cast<std::size_t>(n) * count, count);
            }

            for (std::size_t i = 0; i < count; ++i)
            {
                for (int n = 0; n <= nmax; ++n)
                {
                    judgeEntryCell(n,
                                   ref.x[i],
                                   columns[static_cast<std::size_t>(n) * count + i],
                                   ref.Index(n, i));
                }
            }
        }

        // ---- the run-time selector's two-argument form ----------------------
        // The route and the scheme named together, which is the pair a
        // compile-time call site reaches through one EvalPolicy. The values must
        // be the same values and not merely inside the same bound: the selector
        // exists so that a caller who cannot name the pair at compile time still
        // gets the pair they named.
        const auto runtimePair = [&]<boys::FitRoute kRoute, boys::EvalScheme kScheme>() {
            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> a{};
                std::array<double, 33> b{};
                boys::BoysAllOrdersWithRoute(kRoute, kScheme, nmax, ref.x[i], a.data());
                boys::BoysAllOrders<1.0, boys::EvalPolicy<kRoute, kScheme>>(nmax, ref.x[i],
                                                                            b.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    ++routeRuntimePairs;

                    if (std::memcmp(&a[static_cast<std::size_t>(n)],
                                    &b[static_cast<std::size_t>(n)],
                                    sizeof(double)) != 0)
                    {
                        ++routeRuntimeDiff;
                    }
                }
            }
        };

        runtimePair.template operator()<boys::FitRoute::kChebyshev,
                                        boys::EvalScheme::kSplitClenshaw>();
        runtimePair.template operator()<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner>();
        runtimePair.template operator()<boys::FitRoute::kRationalMinimax,
                                        boys::EvalScheme::kSplitClenshaw>();
        runtimePair.template operator()<boys::FitRoute::kRationalMinimax,
                                        boys::EvalScheme::kHorner>();

        for (const RouteCarriage& c : routeCarriage)
        {
            if (std::strstr(c.entry, "the default route named") != nullptr)
            {
                ++routeCarriageControls;
                continue;
            }

            ++routeCarriageNamed;

            if (c.differ == 0)
            {
                ++routeCarriageMissed;
            }
        }

        // The region-A table's own bookkeeping, read the way the kernel reads
        // it: each piece's coefficients sit at its offset, the denominator
        // column begins where the numerator column ends, and the offsets tile
        // the coefficient array without a gap or an overlap. A row's stored
        // count is then the array's size, which is what makes the two rows'
        // counts comparable.
        std::size_t ratACoeffsNeeded = 0;

        for (int p = 0; p < static_cast<int>(std::size(boys::detail::kPieces)); ++p)
        {
            const int m = boys::detail::kRatANumDeg[p];
            const int k = boys::detail::kRatADenDeg[p];

            if (boys::detail::kRatAOffset[p] != static_cast<int>(ratACoeffsNeeded) || m < 1 ||
                k < 0)
            {
                ++routeTableMismatch;
            }

            ratACoeffsNeeded += static_cast<std::size_t>(m + k + 1);
        }

        if (ratACoeffsNeeded != std::size(boys::detail::kRatACoeffs))
        {
            ++routeTableMismatch;
        }

        std::size_t chebACoeffs = 0;

        for (const boys::detail::OrderPiece& piece : boys::detail::kPieces)
        {
            chebACoeffs += static_cast<std::size_t>(piece.deg + 1);
        }

        if (chebACoeffs != std::size(boys::detail::kCoeffs))
        {
            ++routeTableMismatch;
        }

        for (std::size_t r = 0; r < routeNames; ++r)
        {
            const boys::FitRouteInfo& row = boys::BoysFitRoutes()[r];
            const std::size_t stored =
                (row.region == boys::AccuracyRegion::kA)
                    ? ((row.route == boys::FitRoute::kChebyshev)
                           ? std::size(boys::detail::kCoeffs)
                           : std::size(boys::detail::kRatACoeffs))
                : ((row.route == boys::FitRoute::kChebyshev)
                       ? std::size(boys::detail::kBcoeffs)
                       : std::size(boys::detail::kRatBnum) + std::size(boys::detail::kRatBden));

            if (row.stored != static_cast<int>(stored))
            {
                ++routeStoredMismatch;
            }

            if (!(row.lo < row.hi))
            {
                ++routeDomainMismatch;
            }

            if (!(row.delivered <= row.bound))
            {
                ++routePromiseMismatch;
            }

            // A row's served domain is inside the domain of its fit, and a row
            // that narrowed it did so by naming a real argument rather than by
            // reporting a domain of its own invention.
            if (!(row.servesFrom >= row.lo && row.servesFrom < row.hi))
            {
                ++routeServeMismatch;
            }

            // The reported figure is a sweep on the grid the generator used and
            // this one is a sweep on the committed reference grid, which is
            // coarser: a fit that equioscillates at a degree's worth of ripples
            // is not sampled at its own extrema from here, so this sweep reads a
            // little under the generator's rather than beside it. What can be
            // settled from this grid is therefore one-sided - the gate can find
            // the fit worse than its row reports, never better, and a row whose
            // reported figure would not cover what is measured here is a figure
            // describing a fit other than the one the header holds.
            const double measured =
                RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])].worstErr;

            if (measured > row.delivered + 0.1 * row.bound)
            {
                ++routeDisagreement;
            }
        }

        std::printf("\nthe certified fit routes, measured:\n");
        std::printf("  one row per route and region. 'reported' is the worst error the generator\n"
                    "  measured for the fit it generated, on a dense sweep of each piece's own\n"
                    "  interval; 'measured' is this gate's own sweep of the same route on the\n"
                    "  committed reference grid, which is the coarser of the two and reads a\n"
                    "  little under it on a fit that equioscillates. The gate refuses a row\n"
                    "  whose 'reported' would not cover its 'measured'.\n"
                    "  'from' is the argument the selector takes over at: a value above 'lo' means\n"
                    "  the fit covers more than naming the route hands it\n");
        std::printf("  %-20s %-6s %-9s %-7s %-6s %8s %14s %14s %14s\n",
                    "route",
                    "region",
                    "stored",
                    "scope",
                    "cells",
                    "from",
                    "reported",
                    "measured",
                    "bar");

        for (std::size_t r = 0; r < routeNames; ++r)
        {
            const boys::FitRouteInfo& row = boys::BoysFitRoutes()[r];
            const Accum& seed = RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])];
            const Accum& lane = RouteClaims()[static_cast<std::size_t>(routeLaneClaim[r])];

            std::printf("  %-20s %-6s %-9d %-7s %8zu %14.6g %14.6g %14.6g %14.6g\n",
                        row.name,
                        RegionName(row.region),
                        row.stored,
                        (row.region == boys::AccuracyRegion::kA) ? "0..32" : "F0",
                        seed.points,
                        row.servesFrom,
                        row.delivered,
                        seed.worstErr,
                        row.bound);
            std::printf("  %-20s %-6s %-9s %-7s %8zu %14s %14s %14.6g %14.6g\n",
                        "",
                        "",
                        "",
                        "0..32",
                        lane.points,
                        "-",
                        "-",
                        lane.worstErr,
                        kBoundDoubleBatch);
        }

        std::printf("  cells outside every route's interval that the rational route changed: "
                    "%zu\n"
                    "  cells inside a route's interval that it changed: %zu of %zu (%.1f%%)\n"
                    "  cells where naming the default route differed from the default entry: %zu\n"
                    "  cells where a route value outside the enumeration differed from the "
                    "default: %zu\n",
                    routeDiffOutside,
                    routeDiffInside,
                    routeCellsInside,
                    100.0 * static_cast<double>(routeDiffInside)
                        / static_cast<double>(routeCellsInside == 0 ? 1 : routeCellsInside),
                    routeDefaultDiff,
                    routeUnknownDiff);
    }

    // ---- float single and batch -------------------------------------------
    {
        std::array<float, 33> out{};

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t k = ref.Index(n, i);
                const float got = boys::BoysSingleF32(n, static_cast<float>(ref.x[i]));
                const double asDouble = static_cast<double>(got);
                Measure(kFloatSingle,
                        n,
                        ref.xf[i],
                        asDouble,
                        ref.vf[k],
                        ref.decadeF[k],
                        kBoundFloat,
                        got == 0.0f || std::fabs(asDouble) < std::numeric_limits<float>::min());
            }
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            boys::BoysAllOrdersF32(nmax, static_cast<float>(ref.x[i]), out.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const std::size_t k = ref.Index(n, i);
                const double asDouble = static_cast<double>(out[static_cast<std::size_t>(n)]);
                Measure(kFloatOrders,
                        n,
                        ref.xf[i],
                        asDouble,
                        ref.vf[k],
                        ref.decadeF[k],
                        kBoundFloat,
                        out[static_cast<std::size_t>(n)] == 0.0f ||
                            std::fabs(asDouble) < std::numeric_limits<float>::min());
            }
        }
    }

    // ---- fp16 and bf16, the store-half lane -------------------------------
    {
        std::array<boys::F16, 33> out16{};
        std::array<boys::Bf16, 33> outb{};

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t k = ref.Index(n, i);

                if (std::isfinite(ref.x16[i]))
                {
                    const boys::F16 got = boys::BoysSingleF16(n, boys::F16(static_cast<float>(ref.x[i])));
                    const double asDouble = static_cast<double>(static_cast<float>(got));
                    Measure(kF16Single,
                            n,
                            ref.x16[i],
                            asDouble,
                            ref.v16[k],
                            ref.decade16[k],
                            HalfBound(asDouble, kF16MantissaBits, kF16MinNormalExp),
                            Unrepresentable(asDouble, kF16MinNormalExp));
                }

                const boys::Bf16 gotb = boys::BoysSingleBf16(n, boys::Bf16(static_cast<float>(ref.x[i])));
                const double asDoubleB = static_cast<double>(static_cast<float>(gotb));
                Measure(kBf16Single,
                        n,
                        ref.xb[i],
                        asDoubleB,
                        ref.vb[k],
                        ref.decadeB[k],
                        HalfBound(asDoubleB, kBf16MantissaBits, kBf16MinNormalExp),
                        Unrepresentable(asDoubleB, kBf16MinNormalExp));
            }
        }

        for (std::size_t i = 0; i < count; ++i)
        {
            boys::BoysAllOrdersF16(nmax, boys::F16(static_cast<float>(ref.x[i])), out16.data());
            boys::BoysAllOrdersBf16(nmax, boys::Bf16(static_cast<float>(ref.x[i])), outb.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const std::size_t k = ref.Index(n, i);
                const double asDouble = static_cast<double>(static_cast<float>(
                    out16[static_cast<std::size_t>(n)]));
                Measure(kF16Orders,
                        n,
                        ref.x16[i],
                        asDouble,
                        ref.v16[k],
                        ref.decade16[k],
                        HalfBound(asDouble, kF16MantissaBits, kF16MinNormalExp),
                        Unrepresentable(asDouble, kF16MinNormalExp));

                const double asDoubleB = static_cast<double>(static_cast<float>(
                    outb[static_cast<std::size_t>(n)]));
                Measure(kBf16Orders,
                        n,
                        ref.xb[i],
                        asDoubleB,
                        ref.vb[k],
                        ref.decadeB[k],
                        HalfBound(asDoubleB, kBf16MantissaBits, kBf16MinNormalExp),
                        Unrepresentable(asDoubleB, kBf16MinNormalExp));
            }
        }
    }

    // ---- the half kernels' path divergence and their claimed domain --------
    // The vector body against the certified scalar half entry, at the same
    // cell: the two disagree by quanta of the half format (different
    // arithmetic forms), so the claim they are held to is not equality but
    // that neither returns a value the other's contract would not accept.
    // Counted separately from either path's own error, because the failure
    // this caught was one path returning zero where the other returned a
    // usable value.
    double driftUlp = 0.0;
    int driftOrder = -1;
    double driftX = 0.0;
    std::size_t driftCells = 0;
    std::size_t driftOneSideOut = 0;   // exactly one path outside its bound
    std::size_t driftSimdZero = 0;     // body returns zero, scalar does not
    std::size_t driftScalarZero = 0;   // scalar returns zero, body does not
    double driftSimdZeroX = 0.0;       // the largest argument of each asymmetry
    int driftSimdZeroN = -1;
    double driftScalarZeroX = 0.0;
    int driftScalarZeroN = -1;
    std::size_t driftUnforgivenZero = 0; // a path returns zero above its bound
    double driftUnforgivenRef = 0.0;

    // ---- the 8-wide half-I/O region kernels --------------------------------
    // Two half values per register, computed in binary32 and rounded to the
    // half type on the store (the lane's own scalar tail calls the certified
    // half entries). No public entry reaches these: the suite calls them
    // directly, and so does this. Arguments are partitioned by the region
    // each kernel documents as its precondition.
    if (boys::BoysAvx2Available())
    {
        const float fx0 = static_cast<float>(boys::detail::kX0);
        const float fx1 = static_cast<float>(boys::detail::kX1);
        const int packedN = nmax;

        std::vector<std::size_t> idxA;
        std::vector<std::size_t> idxB;
        std::vector<std::size_t> idxC;

        // n = 32: the order the packed lane's own budget is tightest at. The
        // strict inequalities keep every argument inside the kernel's own
        // region, which is the kernel's documented precondition.
        for (std::size_t i = 0; i < count; ++i)
        {
            const double x = ref.x16[i];

            if (x < fx0)
            {
                idxA.push_back(i);
            } else if (x < fx1)
            {
                idxB.push_back(i);
            } else
            {
                idxC.push_back(i);
            }
        }

        // One region at a time: a region kernel's arguments are its own
        // precondition, and each region's claim is measured separately.
        auto runPacked = [&](const std::vector<std::size_t>& idx, int region, bool isBf16) {
            const int claim = isBf16 ? (region == 0 ? kBf16PackedA
                                                    : (region == 1 ? kBf16PackedB : kBf16PackedC))
                                     : (region == 0 ? kF16PackedA
                                                    : (region == 1 ? kF16PackedB : kF16PackedC));
            if (idx.size() < 8)
            {
                std::printf("gate: packed %s region %d skipped (only %zu arguments)\n",
                            isBf16 ? "bf16" : "fp16",
                            region,
                            idx.size());
                return;
            }

            const int mantissa = isBf16 ? kBf16MantissaBits : kF16MantissaBits;
            const int minExp = isBf16 ? kBf16MinNormalExp : kF16MinNormalExp;
            const bool allOrders = region == 1;
            const std::size_t plane = allOrders ? idx.size() : 1;

            std::vector<boys::F16> x16(idx.size());
            std::vector<boys::Bf16> xb(idx.size());

            for (std::size_t j = 0; j < idx.size(); ++j)
            {
                x16[j] = boys::F16(static_cast<float>(ref.x[idx[j]]));
                xb[j] = boys::Bf16(static_cast<float>(ref.x[idx[j]]));
            }

            std::vector<boys::F16> out16(idx.size() * static_cast<std::size_t>(packedN + 1));
            std::vector<boys::Bf16> outb(idx.size() * static_cast<std::size_t>(packedN + 1));

            if (isBf16)
            {
                if (region == 0)
                {
                    boys::detail::BoysRegionASimdBf16(packedN, xb.data(), outb.data(), idx.size());
                } else if (region == 1)
                {
                    boys::detail::BoysRegionBSimdBf16(packedN, xb.data(), outb.data(), idx.size());
                } else
                {
                    boys::detail::BoysRegionCSimdBf16(packedN, xb.data(), outb.data(), idx.size());
                }
            } else
            {
                if (region == 0)
                {
                    boys::detail::BoysRegionASimdF16(packedN, x16.data(), out16.data(), idx.size());
                } else if (region == 1)
                {
                    boys::detail::BoysRegionBSimdF16(packedN, x16.data(), out16.data(), idx.size());
                } else
                {
                    boys::detail::BoysRegionCSimdF16(packedN, x16.data(), out16.data(), idx.size());
                }
            }

            for (std::size_t j = 0; j < idx.size(); ++j)
            {
                const std::size_t i = idx[j];

                for (int n = 0; n <= (allOrders ? packedN : 0); ++n)
                {
                    const int order = allOrders ? n : packedN;
                    const std::size_t k = ref.Index(order, i);
                    const std::size_t slot = allOrders ? static_cast<std::size_t>(n) * plane + j : j;
                    const double got = isBf16
                                           ? static_cast<double>(static_cast<float>(outb[slot]))
                                           : static_cast<double>(static_cast<float>(out16[slot]));
                    const double want = isBf16 ? ref.vb[k] : ref.v16[k];
                    const int decade = isBf16 ? ref.decadeB[k] : ref.decade16[k];

                    Measure(claim,
                            order,
                            ref.x[i],
                            got,
                            want,
                            decade,
                            HalfBound(got, mantissa, minExp),
                            Unrepresentable(got, minExp));

                    // The same cell through the certified scalar half entry.
                    const double scalar =
                        isBf16 ? static_cast<double>(static_cast<float>(
                                     boys::BoysSingleBf16(order, xb[j])))
                               : static_cast<double>(static_cast<float>(
                                     boys::BoysSingleF16(order, x16[j])));
                    const double quantum = UlpOf(got, mantissa, minExp);

                    if (quantum > 0.0)
                    {
                        const double drift = std::abs(got - scalar) / quantum;

                        if (drift > driftUlp)
                        {
                            driftUlp = drift;
                            driftOrder = order;
                            driftX = ref.x[i];
                        }
                    }

                    ++driftCells;

                    if (got == 0.0 && scalar != 0.0)
                    {
                        ++driftSimdZero;

                        if (ref.x[i] > driftSimdZeroX)
                        {
                            driftSimdZeroX = ref.x[i];
                            driftSimdZeroN = order;
                        }
                    }

                    if (scalar == 0.0 && got != 0.0)
                    {
                        ++driftScalarZero;

                        if (ref.x[i] > driftScalarZeroX)
                        {
                            driftScalarZeroX = ref.x[i];
                            driftScalarZeroN = order;
                        }
                    }

                    // The one zero the contract cannot forgive: a path that
                    // returns zero where the *value* is above that path's own
                    // bound. A zero below the bound is the format's floor doing
                    // what the header says it does, and is counted, not failed.
                    const bool simdForgiven = std::abs(want) <= HalfBound(got, mantissa, minExp);
                    const bool scalarForgiven = std::abs(want) <= HalfBound(scalar, mantissa, minExp);

                    if ((got == 0.0 && !simdForgiven) || (scalar == 0.0 && !scalarForgiven))
                    {
                        ++driftUnforgivenZero;
                        driftUnforgivenRef = std::max(driftUnforgivenRef, std::abs(want));
                    }

                    const bool simdIn = std::abs(got - want) <= HalfBound(got, mantissa, minExp);
                    const bool scalarIn =
                        std::abs(scalar - want) <= HalfBound(scalar, mantissa, minExp);

                    if (simdIn != scalarIn)
                    {
                        ++driftOneSideOut;
                    }
                }
            }
        };

        for (int region = 0; region < 3; ++region)
        {
            const std::vector<std::size_t>& idx = region == 0 ? idxA : (region == 1 ? idxB : idxC);
            runPacked(idx, region, false);
            runPacked(idx, region, true);
        }
    }

    // ---- the run-time accuracy tier, if this revision carries it -----------
    // BoysAllOrdersAtTier / QueryTier / TierCoverage arrive on a branch of
    // their own. The rungs are the multipliers the book already knows, but the
    // tier turns them into a run-time choice, so it carries two claims of its
    // own that no static-lane measurement covers: the rung the caller picks is
    // the rung that runs (a run-time switch can reach the wrong body), and the
    // query surface's report is true of the values the same revision delivers -
    // a surface that says a tier reaches a tolerance it does not reach is the
    // failure this gate exists to catch.
    std::size_t tierCells = 0;
    std::size_t tierQueryPairs = 0;
    std::size_t tierQueryMiss = 0;       // surface says "meets", the rung is worse
    std::size_t tierReachableShort = 0;  // "reachable" under the delivered error
    std::size_t tierLimitingWrong = 0;   // the component it names is not the region's
    std::size_t tierStaticCells = 0;     // cells compared against the static lane
    std::size_t tierStaticMismatch = 0;  // a run-time rung differing from it, bit for bit
    double tierQueryWorstGap = 0.0;      // largest delivered - tolerance at a false "meets"
    double tierQueryWorstTol = 0.0;
    double tierReachableGap = 0.0;       // largest delivered - reachable
    std::size_t tierWorstRung = 0;
    std::array<double, 7> tierRungRatio{}; // worst ratio per rung, in rung order
    // The same sentence read the other way: "B_region" as the per-region table
    // rather than the batch row. That reading is what the tier's own surface
    // rejects (it bases regions A and B on the batch bound), and it is recorded
    // here as a measurement so the wording can be settled with numbers.
    double tierStrictRatio = 0.0;
    int tierStrictOrder = -1;
    double tierStrictX = 0.0;
    double tierStrictErr = 0.0;
    double tierStrictBound = 0.0;
    std::size_t tierStrictCells = 0;
    double tierStrictRefRatio = 0.0; // the same reading at m = 1, the certified lane

#ifdef BOYS_GATE_TIER
    {
        const std::array<boys::AccuracyTier, 7> rungs{
            boys::AccuracyTier::kReference,   boys::AccuracyTier::kRelaxed64,
            boys::AccuracyTier::kRelaxed256,  boys::AccuracyTier::kRelaxed1024,
            boys::AccuracyTier::kRelaxed4096, boys::AccuracyTier::kRelaxed16384,
            boys::AccuracyTier::kRelaxed65536};
        std::array<int, 7> slot{};
        std::array<std::array<double, 3>, 7> regionWorst{};

        for (std::size_t r = 0; r < rungs.size(); ++r)
        {
            slot[r] = kTierRung[r];
        }

        for (std::size_t r = 0; r < rungs.size(); ++r)
        {
            const boys::AccuracyTier tier = rungs[r];
            const double m = boys::AccuracyMultiplier(tier);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];
                const int region = SingleClaim(x);
                const double regionBound = region == 0   ? kBoundSingleA
                                           : region == 1 ? kBoundSingleBand
                                           : region == 2 ? kBoundSingleB
                                                         : kBoundSingleC;
                // The query surface knows three regions; the published table
                // splits the band out of B, and both are the surface's kB.
                const std::size_t qr = region == 0 ? 0 : (region == 3 ? 2 : 1);
                std::array<double, 33> got{};
                boys::BoysAllOrdersAtTier(tier, nmax, x, got.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double value = got[static_cast<std::size_t>(n)];
                    const double err = std::abs(value - ref.v[k]);

                    if (err > regionWorst[r][qr])
                    {
                        regionWorst[r][qr] = err;
                    }

                    // The bound this entry carries: it is the all-orders batch
                    // entry, so the README row that applies is `double batch`
                    // (m * 5.5e-14 flat), which is also the base the tier's own
                    // QueryTier uses for regions A and B. The per-region row is
                    // measured beside it, never instead of it.
                    const double strictRatio = err / (m * regionBound);

                    if (strictRatio > tierStrictRatio)
                    {
                        tierStrictRatio = strictRatio;
                        tierStrictOrder = n;
                        tierStrictX = x;
                        tierStrictErr = err;
                        tierStrictBound = m * regionBound;
                    }

                    if (strictRatio > 1.0)
                    {
                        ++tierStrictCells;
                    }

                    if (r == 0 && regionBound < kBoundDoubleBatch &&
                        strictRatio > tierStrictRefRatio)
                    {
                        // Only where the two readings differ: in region C the
                        // per-region row and the batch row are the same number.
                        tierStrictRefRatio = strictRatio;
                    }

                    ++tierCells;
                    Measure(slot[r],
                            n,
                            x,
                            value,
                            ref.v[k],
                            ref.decade[k],
                            m * kBoundDoubleBatch,
                            Unrepresentable(value, -1022));
                }
            }

            tierRungRatio[r] = Claims()[static_cast<std::size_t>(slot[r])].worstRatio;
        }

        // The run-time rung against the compile-time lane at the same m: the
        // header says a rung is the template instantiation the static entry
        // reaches, which is a statement about bits and not about error.
        auto compareStatic = [&]<double M>(std::size_t r) {
            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> runtime{};
                std::array<double, 33> compiletime{};
                boys::BoysAllOrdersAtTier(rungs[r], nmax, ref.x[i], runtime.data());
                boys::BoysAllOrders<M>(nmax, ref.x[i], compiletime.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    ++tierStaticCells;

                    if (std::memcmp(&runtime[static_cast<std::size_t>(n)],
                                    &compiletime[static_cast<std::size_t>(n)],
                                    sizeof(double)) != 0)
                    {
                        ++tierStaticMismatch;
                    }
                }
            }
        };

        compareStatic.operator()<1.0>(0);
        compareStatic.operator()<64.0>(1);
        compareStatic.operator()<65536.0>(6);

        // The query surface, against what the same revision delivers. A tier
        // reported as meeting a tolerance it delivers worse than is a promise
        // the kernel does not keep; `reachable` under the delivered error is
        // the same failure one step earlier, since every tolerance at or above
        // it is reported as met.
        for (std::size_t r = 0; r < rungs.size(); ++r)
        {
            for (std::size_t q = 0; q < 3; ++q)
            {
                const boys::AccuracyRegion region = q == 0   ? boys::AccuracyRegion::kA
                                                   : q == 1 ? boys::AccuracyRegion::kB
                                                            : boys::AccuracyRegion::kC;
                const boys::AccuracyComponent expect =
                    q == 0   ? boys::AccuracyComponent::kRegionASeed
                    : q == 1 ? boys::AccuracyComponent::kRegionBFit
                             : boys::AccuracyComponent::kRegionCAsymptotic;
                const double delivered = regionWorst[r][q];
                const boys::TierCoverage coverage = boys::QueryTier(rungs[r], region, delivered);
                ++tierQueryPairs;

                if (coverage.limiting != expect)
                {
                    ++tierLimitingWrong;
                }

                if (coverage.reachable < delivered)
                {
                    ++tierReachableShort;
                    tierReachableGap = std::max(tierReachableGap, delivered - coverage.reachable);
                }

                for (int e = -18; e <= -6; ++e)
                {
                    for (const double factor : {0.5, 1.0, 2.0})
                    {
                        const double tolerance = factor * std::pow(10.0, e);
                        const boys::TierCoverage at =
                            boys::QueryTier(rungs[r], region, tolerance);
                        ++tierQueryPairs;

                        if (at.meets && delivered > tolerance)
                        {
                            ++tierQueryMiss;

                            if (delivered - tolerance > tierQueryWorstGap)
                            {
                                tierQueryWorstGap = delivered - tolerance;
                                tierQueryWorstTol = tolerance;
                                tierWorstRung = r;
                            }
                        }
                    }
                }
            }
        }
    }
#endif

    // ---- the same tier on the single-order shape ----------------------------
    // A rung is a run-time choice, and the shape a caller reads one order from
    // is a different call: an engine that reads a single order cannot reach the
    // rung through an entry that computes every order. So the single-order
    // run-time entry is measured against the single lane's own contract, per
    // region, and the two things held are the two no static-lane row covers -
    // the rung the caller names is the rung that runs, and the route named at
    // run time is reachable on this shape too.
    std::size_t tierSingleCells = 0;
    std::size_t tierSingleFailures = 0;
    double tierSingleWorstRatio = 0.0;
    std::size_t tierSingleWorstRung = 0;
    int tierSingleWorstOrder = -1;
    double tierSingleWorstX = 0.0;
    double tierSingleWorstErr = 0.0;
    double tierSingleWorstBound = 0.0;
    std::size_t tierSingleStaticMismatch = 0;
    std::size_t tierSingleRationalCells = 0;
    std::size_t tierSingleRationalDiffer = 0;

    {
        const std::array<boys::AccuracyTier, 7> rungs{
            boys::AccuracyTier::kReference,   boys::AccuracyTier::kRelaxed64,
            boys::AccuracyTier::kRelaxed256,  boys::AccuracyTier::kRelaxed1024,
            boys::AccuracyTier::kRelaxed4096, boys::AccuracyTier::kRelaxed16384,
            boys::AccuracyTier::kRelaxed65536};

        // The compile-time lane at each rung, which the run-time entry has to
        // be: one switch can reach the wrong body, and a wrong body is a
        // last-place difference inside the bound the row is judged on, so the
        // comparison is exact rather than a tolerance.
        const auto staticRung = [&](std::size_t r, int n, double x) -> double {
            switch (r)
            {
            case 0:
                return boys::BoysSingle<1.0>(n, x);
            case 1:
                return boys::BoysSingle<64.0>(n, x);
            case 2:
                return boys::BoysSingle<256.0>(n, x);
            case 3:
                return boys::BoysSingle<1024.0>(n, x);
            case 4:
                return boys::BoysSingle<4096.0>(n, x);
            case 5:
                return boys::BoysSingle<16384.0>(n, x);
            default:
                return boys::BoysSingle<65536.0>(n, x);
            }
        };

        for (std::size_t r = 0; r < rungs.size(); ++r)
        {
            const double m = boys::AccuracyMultiplier(rungs[r]);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];
                const double bound = m * SingleBound(x);

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double got = boys::BoysSingleAtTier(rungs[r], n, x);
                    const double err = std::abs(got - ref.v[k]);
                    ++tierSingleCells;

                    if (err > bound)
                    {
                        ++tierSingleFailures;
                    }

                    if (err / bound > tierSingleWorstRatio)
                    {
                        tierSingleWorstRatio = err / bound;
                        tierSingleWorstRung = r;
                        tierSingleWorstOrder = n;
                        tierSingleWorstX = x;
                        tierSingleWorstErr = err;
                        tierSingleWorstBound = bound;
                    }

                    const double want = staticRung(r, n, x);

                    if (std::memcmp(&got, &want, sizeof(double)) != 0)
                    {
                        ++tierSingleStaticMismatch;
                    }

                    // The route named at run time, on this shape: the same
                    // rung, the rational route against the default one. Counted
                    // as a difference because that is what carriage is.
                    if (n == 0 || (n % 8) == 0)
                    {
                        const double rational = boys::BoysSingleAtTier(
                            rungs[r], boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw, n, x);
                        ++tierSingleRationalCells;

                        if (std::memcmp(&rational, &got, sizeof(double)) != 0)
                        {
                            ++tierSingleRationalDiffer;
                        }
                    }
                }
            }
        }
    }


    //
    // Three things are measured apart, because the lane publishes them as three
    // claims: the bound where the return is a normal half (its domain), the
    // points past that domain's ceiling (counted, never passed), and the
    // packing itself, which the error cannot show - see the distribution
    // counters, which a widen-compute-round-once lane cannot produce.
    // ---- the native packed half lane, if this revision carries it -----------
    // Region C only, one precondition (x >= the lane's own fp16 rounding of x1),
    // no fallback and no multiplier, returning 2^15 * F_k(x) so that the whole
    // ladder stays inside binary16's normal range down to F_k(x) = 2^-29.
    //
    // Three things are measured apart, because the lane publishes them as three
    // claims: the bound where the return is a normal half (its domain), the
    // points past that domain's ceiling (counted, never passed), and the
    // packing itself, which the error cannot show - see the distribution
    // counters, which a widen-compute-round-once lane cannot produce.
    std::size_t nativeCells = 0;
    std::size_t nativeNoClaim = 0;
    double nativeNoClaimTruth = 0.0;
    int nativeNoClaimOrder = -1;
    double nativeNoClaimX = 0.0;
    std::size_t nativeExcusedFailures = 0; // past the ceiling but out of bound
    std::size_t nativePastHalfUlp = 0;
    std::size_t nativePastOneUlp = 0;
    std::size_t nativeDistTotal = 0;
    double nativeWorstUlp = 0.0;
    int nativeWorstOrder = -1;
    double nativeWorstX = 0.0;
    std::size_t nativeBatchMismatch = 0;
    double nativeBatchWorstUlp = 0.0;
    std::array<double, boys::kMaxBoysOrder + 1> nativeCeilNormalX{}; // largest x still normal
    std::array<double, boys::kMaxBoysOrder + 1> nativeCeilNoClaimX{}; // smallest x past it

    for (double& v : nativeCeilNoClaimX)
    {
        v = std::numeric_limits<double>::infinity();
    }

#ifdef BOYS_GATE_NATIVE_HALF
    {
        // The lane's own precondition, at the fp16 value it rounds the boundary
        // to: fp16(kX1) = 28.984375.
        const double xStart =
            static_cast<double>(static_cast<float>(boys::F16(
                static_cast<float>(boys::detail::kX1))));
        std::vector<std::size_t> idxNative;

        for (std::size_t i = 0; i < count; ++i)
        {
            if (std::isfinite(ref.x16[i]) && ref.x16[i] >= xStart)
            {
                idxNative.push_back(i);
            }
        }

        const int nativeN = nmax;
        const std::size_t nArgs = idxNative.size();
        std::vector<boys::Half2> packed(static_cast<std::size_t>(nativeN) + 1);
        std::vector<boys::F16> pairRun(static_cast<std::size_t>(nativeN + 1) * 2);
        std::vector<boys::F16> arguments(nArgs + 1);
        std::vector<boys::F16> allRun((static_cast<std::size_t>(nativeN) + 1) * (nArgs + 1));

        // The argument vector the count entry takes; one extra entry keeps the
        // call odd, so the lone trailing argument's path is exercised too.
        for (std::size_t j = 0; j < nArgs; ++j)
        {
            arguments[j] = boys::F16(static_cast<float>(ref.x16[idxNative[j]]));
        }

        arguments[nArgs] = nArgs > 0 ? arguments[0] : boys::F16(0.0f);

        // The pair is the unit of work: two different arguments travel in one
        // register, so a half that reads its partner's lane shows up here.
        for (std::size_t j = 0; j + 1 < nArgs; j += 2)
        {
            const std::size_t i0 = idxNative[j];
            const std::size_t i1 = idxNative[j + 1];
            const boys::F16 a = boys::F16(static_cast<float>(ref.x16[i0]));
            const boys::F16 b = boys::F16(static_cast<float>(ref.x16[i1]));

            boys::BoysAllOrdersHalf2(nativeN, boys::Half2(a, b), packed.data());

            // The count entry over the same two arguments: the batch form of
            // the lane must be the packed form per argument, not a second
            // implementation of it.
            boys::BoysAllNF16Native(nativeN, arguments.data() + j, pairRun.data(), 2);

            for (int k = 0; k <= nativeN; ++k)
            {
                const std::size_t cell = static_cast<std::size_t>(k);

                for (int half = 0; half < 2; ++half)
                {
                    const std::size_t i = half == 0 ? i0 : i1;
                    const boys::F16 raw = half == 0 ? packed[cell].Low() : packed[cell].High();
                    const double got = static_cast<double>(static_cast<float>(raw));
                    const double want = kNativeScale * ref.v16[ref.Index(k, i)];
                    const double returned = std::abs(got);

                    ++nativeCells;

                    if (static_cast<double>(static_cast<float>(
                            pairRun[cell * 2 + static_cast<std::size_t>(half)])) != got)
                    {
                        ++nativeBatchMismatch;
                    }

                    // Past the ceiling the lane claims nothing: the return is a
                    // subnormal half, then exactly zero, by design. Counted
                    // here, and excused only inside that domain - a subnormal
                    // return for a value the scaling was supposed to keep
                    // normal is a failure and is measured as one.
                    const bool pastCeiling = returned < kNativeSmallestNormal;

                    if (pastCeiling && std::abs(want) > kNativeSmallestNormal)
                    {
                        ++nativeExcusedFailures;
                        nativeNoClaimTruth = std::max(nativeNoClaimTruth, std::abs(want));
                        continue;
                    }

                    if (pastCeiling)
                    {
                        ++nativeNoClaim;

                        if (std::abs(want) > nativeNoClaimTruth)
                        {
                            nativeNoClaimTruth = std::abs(want);
                            nativeNoClaimOrder = k;
                            nativeNoClaimX = ref.x16[i];
                        }

                        nativeCeilNoClaimX[cell] = std::min(nativeCeilNoClaimX[cell], ref.x16[i]);
                        continue;
                    }

                    if (ref.x16[i] > nativeCeilNormalX[cell])
                    {
                        nativeCeilNormalX[cell] = ref.x16[i];
                    }

                    Measure(kNativeHalf2,
                            k,
                            ref.x16[i],
                            got,
                            want,
                            ref.decade16[ref.Index(k, i)],
                            kNativeUlpBound * UlpOf(got, kF16MantissaBits, kF16MinNormalExp),
                            false);

                    // The quantum of the true value, which is the definition the
                    // lane's own suite uses: half a quantum of it is what one
                    // rounding to nearest can leave, so more than that cannot
                    // come from a single rounding however it is arranged.
                    const double err = std::abs(got - want);
                    const double trueQuantum = UlpOf(std::abs(want), kF16MantissaBits,
                                                     kF16MinNormalExp);
                    const double ulps = trueQuantum > 0.0 ? err / trueQuantum : 0.0;

                    ++nativeDistTotal;

                    if (ulps > 0.5)
                    {
                        ++nativePastHalfUlp;
                    }

                    if (ulps > 1.0)
                    {
                        ++nativePastOneUlp;
                    }
                }
            }
        }

        // The count entry's own bound, over every argument once: the packed
        // loop above already compared the two entries value for value, this is
        // the same sweep through the count form alone, including the trailing
        // odd argument.
        boys::BoysAllNF16Native(nativeN, arguments.data(), allRun.data(), nArgs + 1);

        for (std::size_t j = 0; j <= nArgs; ++j)
        {
            const std::size_t i = idxNative[j < nArgs ? j : 0];

            for (int k = 0; k <= nativeN; ++k)
            {
                const double got = static_cast<double>(static_cast<float>(
                    allRun[static_cast<std::size_t>(k) * (nArgs + 1) + j]));
                const double want = kNativeScale * ref.v16[ref.Index(k, i)];

                if (std::abs(got) < kNativeSmallestNormal)
                {
                    continue;
                }

                Measure(kNativeHalfBatch,
                        k,
                        ref.x16[i],
                        got,
                        want,
                        ref.decade16[ref.Index(k, i)],
                        kNativeUlpBound * UlpOf(got, kF16MantissaBits, kF16MinNormalExp),
                        false);

                const double ulps =
                    std::abs(got - want) / UlpOf(got, kF16MantissaBits, kF16MinNormalExp);

                if (ulps > nativeBatchWorstUlp)
                {
                    nativeBatchWorstUlp = ulps;
                }
            }
        }

        // The lane's published worst, in its own units: the largest error the
        // sweep found, in quanta of the returned value, over the binding domain.
        for (const int slot : {kNativeHalf2, kNativeHalfBatch})
        {
            const Accum& a = Claims()[static_cast<std::size_t>(slot)];

            for (int k = 0; k <= nmax; ++k)
            {
                const OrderAccum& o = a.byOrder[static_cast<std::size_t>(k)];

                if (o.points > 0 && o.worstRatio * kNativeUlpBound > nativeWorstUlp)
                {
                    nativeWorstUlp = o.worstRatio * kNativeUlpBound;
                    nativeWorstOrder = k;
                    nativeWorstX = o.worstX;
                }
            }
        }
    }
#endif

    // ------------------------------------------------------------------
    // The published statements that are not one lane x region cell.
    // ------------------------------------------------------------------

    // The half lane's published cells: the 0.990-of-budget worst cell at
    // (order 0, x = 721), the representation term's share of that budget, and
    // the published range statement - from order 3 upward on region C the
    // values leave the half type's normal range, subnormal first, then zero.
    double cellErr = 0.0;
    double cellValue = 0.0;
    double cellRef = 0.0;
    double cellBound = 0.0;
    std::array<double, boys::kMaxBoysOrder + 1> refNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> refZeroX{};
    std::array<double, boys::kMaxBoysOrder + 1> refSpanNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> refSpanSubnormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> laneNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> laneNonzeroX{};
    std::array<double, boys::kMaxBoysOrder + 1> laneRefNormalX{};

    for (int n = 0; n <= nmax; ++n)
    {
        laneNormalX[static_cast<std::size_t>(n)] = -1.0;
        laneNonzeroX[static_cast<std::size_t>(n)] = -1.0;
        laneRefNormalX[static_cast<std::size_t>(n)] = -1.0;
        refNormalX[static_cast<std::size_t>(n)] =
            LargestXAbove(ref, n, std::ldexp(1.0, kF16MinNormalExp));
        refZeroX[static_cast<std::size_t>(n)] = LargestXAbove(ref, n, std::ldexp(1.0, -25));
        refSpanNormalX[static_cast<std::size_t>(n)] = LargestXSparsable(ref, n, kF16NormalField);
        refSpanSubnormalX[static_cast<std::size_t>(n)] = LargestXSparsable(ref, n, kF16Field);
    }

    // The published cell itself, measured at the published argument: the grid
    // carries x = 721 exactly, so this is that cell and not its nearest node.
    {
        const auto it = std::find(ref.x.begin(), ref.x.end(), 721.0);
        const std::size_t i = static_cast<std::size_t>(it - ref.x.begin());

        if (i < count && ref.x[i] == 721.0)
        {
            const boys::F16 got = boys::BoysSingleF16(0, boys::F16(721.0f));
            cellValue = static_cast<double>(static_cast<float>(got));
            cellRef = ref.v16[ref.Index(0, i)];
            cellErr = std::abs(cellValue - cellRef);
            cellBound = HalfBound(cellValue, kF16MantissaBits, kF16MinNormalExp);
        }
    }

    // The half lane's measured range, from the same sweep the bound is
    // measured over: the largest argument at which the lane still returned a
    // normal value, and the largest at which it returned anything but zero.
    {
        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                if (ref.x[i] < boys::detail::kX1 || !std::isfinite(ref.x16[i]))
                {
                    continue;
                }

                const boys::F16 got =
                    boys::BoysSingleF16(n, boys::F16(static_cast<float>(ref.x[i])));
                const double value = static_cast<double>(static_cast<float>(got));

                if (value != 0.0)
                {
                    laneNonzeroX[static_cast<std::size_t>(n)] =
                        std::max(laneNonzeroX[static_cast<std::size_t>(n)], ref.x16[i]);
                }

                if (std::fabs(value) >= std::ldexp(1.0, kF16MinNormalExp))
                {
                    laneNormalX[static_cast<std::size_t>(n)] =
                        std::max(laneNormalX[static_cast<std::size_t>(n)], ref.x16[i]);
                    laneRefNormalX[static_cast<std::size_t>(n)] =
                        std::max(laneRefNormalX[static_cast<std::size_t>(n)],
                                 std::fabs(ref.v16[ref.Index(n, i)]));
                }
            }
        }
    }

    // The asymptotic branch outside its own domain: the published row says the
    // error jumps five to eight orders below about x = 16, measured on the
    // form as coded, against the same reference. The region-C kernel is that
    // form, and the grid carries arguments on both sides of the stated edge.
    //
    // The kernel's vector body covers four arguments at a time and hands a
    // shorter run to the certified scalar entry, so the form can only be
    // reached by asking for a full vector: four copies of the argument, and
    // the first lane is the value. Calling with count = 1 would measure the
    // scalar lane and report its 1e-14 as if it were the asymptotic form's.
    //
    // A jump has to say over what window: the error of this form is smooth in
    // x, so the ratio across the edge grows with the window and shrinks to one
    // as the window closes. Both readings are measured - the single grid step
    // straddling 16, and a two-unit window on each side - and the widest is
    // reported beside them, so the published figure can be compared with the
    // reading it fits rather than with a chosen one.
    double asymErrBelow = 0.0;
    double asymErrInside = 0.0;
    double asymErrShipped = 0.0;
    int asymOrder = -1;
    double asymAt = 0.0;
    int asymShippedN = -1;
    double asymShippedX = 0.0;
    double asymStepJumpMin = 0.0;
    double asymStepJumpMax = 0.0;
    double asymWinJumpMin = 0.0;
    double asymWinJumpMax = 0.0;
    double asymBelowRelMin = 0.0;
    double asymBelowRelMax = 0.0;
    // The shape of the error through the boundary, as opposed to its size: the
    // published sentence calls the lower edge a floor, and a floor is a
    // threshold. A threshold would show as a pair of neighbouring arguments
    // whose errors differ by orders of magnitude, and as an error that is not
    // monotone in x across the window.
    double asymStepRatioMax = 0.0;
    std::size_t asymPairs = 0;
    std::size_t asymBreaks = 0;
    double asymRelBelowMax = 0.0;
    int asymRelBelowN = -1;
    double asymRelBelowX = 0.0;
    double asymRelAt32 = 0.0;
    double asymStepRatioAtWorst = 0.0;
    int asymRatioWorstN = -1;
    double asymRatioWorstX = 0.0;
    // The same shape per order, because a claim about smoothness is a claim
    // about a curve and the curve differs by order: the top order is where the
    // oracle's sentence is about, and the window at large n is where the
    // branch's own validity ends.
    std::array<std::size_t, boys::kMaxBoysOrder + 1> asymPairsN{};
    std::array<std::size_t, boys::kMaxBoysOrder + 1> asymBreaksN{};
    std::array<double, boys::kMaxBoysOrder + 1> asymRatioN{};
    {
        std::array<double, boys::kMaxBoysOrder + 1> stepBelow{};
        std::array<double, boys::kMaxBoysOrder + 1> stepAbove{};
        std::array<double, boys::kMaxBoysOrder + 1> winBelow{};
        std::array<double, boys::kMaxBoysOrder + 1> winAbove{};
        std::array<double, boys::kMaxBoysOrder + 1> prevErr{};
        std::array<double, boys::kMaxBoysOrder + 1> prevX{};

        for (int n = 0; n <= nmax; ++n)
        {
            stepBelow[static_cast<std::size_t>(n)] = -1.0;
            stepAbove[static_cast<std::size_t>(n)] = -1.0;
        }

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];

                if (x < 14.0 || x > 18.0)
                {
                    continue;
                }

                const double in[4] = {x, x, x, x};
                double out[4] = {};
                boys::detail::BoysRegionCSimd(n, in, out, 4);
                const double err = std::abs(out[0] - ref.v[ref.Index(n, i)]);
                const std::size_t sn = static_cast<std::size_t>(n);
                const double trueValue = ref.v[ref.Index(n, i)];
                // The shape is a property of the error as a function of x, and
                // at these arguments the value itself moves by orders of
                // magnitude between neighbouring nodes, so the comparison that
                // means anything is between relative errors - and it means
                // something below the edge, where the sentence puts the growth.
                const double relErr = trueValue != 0.0 ? err / std::abs(trueValue) : 0.0;

                if (prevErr[sn] > 0.0 && relErr > 0.0 && x < 16.0 && prevX[sn] < 16.0)
                {
                    const double ratio =
                        relErr > prevErr[sn] ? relErr / prevErr[sn] : prevErr[sn] / relErr;

                    asymStepRatioMax = std::max(asymStepRatioMax, ratio);
                    asymRatioN[sn] = std::max(asymRatioN[sn], ratio);
                    ++asymPairs;
                    ++asymPairsN[sn];

                    if (ratio > asymStepRatioAtWorst)
                    {
                        asymStepRatioAtWorst = ratio;
                        asymRatioWorstN = n;
                        asymRatioWorstX = x;
                    }

                    if (relErr > prevErr[sn])
                    {
                        // x rose and the relative error rose with it: not
                        // monotone in x below the edge.
                        ++asymBreaks;
                        ++asymBreaksN[sn];
                    }
                }

                prevErr[sn] = relErr;
                prevX[sn] = x;

                if (x < 16.0 && trueValue != 0.0)
                {
                    const double rel = err / std::abs(trueValue);

                    if (rel > asymRelBelowMax)
                    {
                        asymRelBelowMax = rel;
                        asymRelBelowN = n;
                        asymRelBelowX = x;
                    }

                    if (n == 32)
                    {
                        asymRelAt32 = rel;
                    }
                }

                if (x < 16.0)
                {
                    // The node nearest 16 from below is the last one written.
                    stepBelow[sn] = err;

                    if (err > winBelow[sn])
                    {
                        winBelow[sn] = err;
                    }
                } else
                {
                    if (stepAbove[sn] < 0.0)
                    {
                        // ... and the node nearest 16 from above is the first.
                        stepAbove[sn] = err;
                    }

                    if (err > winAbove[sn])
                    {
                        winAbove[sn] = err;
                    }
                }
            }
        }

        asymStepJumpMin = 0.0;
        asymStepJumpMax = 0.0;
        asymWinJumpMin = 0.0;
        asymWinJumpMax = 0.0;

        for (int n = 0; n <= nmax; ++n)
        {
            const std::size_t sn = static_cast<std::size_t>(n);

            if (stepBelow[sn] > 0.0 && stepAbove[sn] > 0.0)
            {
                const double jump = stepBelow[sn] / stepAbove[sn];

                asymStepJumpMin = asymStepJumpMin == 0.0 ? jump : std::min(asymStepJumpMin, jump);
                asymStepJumpMax = std::max(asymStepJumpMax, jump);
            }

            if (winBelow[sn] > 0.0 && winAbove[sn] > 0.0)
            {
                const double jump = winBelow[sn] / winAbove[sn];

                asymWinJumpMin = asymWinJumpMin == 0.0 ? jump : std::min(asymWinJumpMin, jump);
                asymWinJumpMax = std::max(asymWinJumpMax, jump);
            }

            // The other reading a jump can have here: the error just below the
            // edge against the error the branch delivers inside its domain,
            // which the sweep measures as its 5.5e-14 budget. Comparing the
            // budget with the branch's own measured in-domain error shows the
            // two are the same number, so this is a comparison between the
            // branch's two regimes rather than against a threshold pulled from
            // thin air. Restricted to the orders callers use on this branch.
            if (n <= 24 && stepBelow[sn] > 0.0)
            {
                const double rel = stepBelow[sn] / kBoundSingleC;

                asymBelowRelMin = asymBelowRelMin == 0.0 ? rel : std::min(asymBelowRelMin, rel);
                asymBelowRelMax = std::max(asymBelowRelMax, rel);
            }
        }

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];

                if (x < 4.0 || x > 60.0)
                {
                    continue;
                }

                const double in[4] = {x, x, x, x};
                double out[4] = {};
                boys::detail::BoysRegionCSimd(n, in, out, 4);
                const double err = std::abs(out[0] - ref.v[ref.Index(n, i)]);

                if (x < 16.0 && err > asymErrBelow)
                {
                    asymErrBelow = err;
                    asymAt = x;
                    asymOrder = n;
                }

                if (x >= 16.0 && x < boys::detail::kX1 && err > asymErrInside)
                {
                    asymErrInside = err;
                }

                if (x >= boys::detail::kX1 && err > asymErrShipped)
                {
                    asymErrShipped = err;
                    asymShippedN = n;
                    asymShippedX = x;
                }
            }
        }
    }

    // Is a per-order power-of-two rescale exact? The packed-half paragraph's
    // reason is one line: "the rescale is exact, so the scaled lane is
    // bit-identical to the unscaled one at every order". That is a statement
    // about binary16 arithmetic rather than about a prototype, so it is
    // checkable here: rounding to binary16 and scaling by a power of two
    // commute wherever neither end leaves the format's normal range. Both ends
    // must be normal for the statement to be about rounding rather than about
    // the subnormal field's shrinking significand, so a value that is itself
    // subnormal is not evidence either way and is skipped.
    std::size_t rescaleChecked = 0;
    std::size_t rescaleViolations = 0;
    {
        const double kF16MinNormal = std::ldexp(1.0, kF16MinNormalExp);

        for (std::size_t i = 0; i < count; ++i)
        {
            const double v = std::fabs(ref.v[ref.Index(0, i)]);

            if (!(v >= kF16MinNormal && v <= 65504.0))
            {
                continue;
            }

            for (int k = -40; k <= 40; ++k)
            {
                const double scaled = std::ldexp(v, k);

                if (scaled > 65504.0 || scaled < kF16MinNormal)
                {
                    continue;
                }

                const double rounded = static_cast<double>(
                    static_cast<float>(boys::F16(static_cast<float>(scaled))));
                const double shifted = std::ldexp(
                    static_cast<double>(static_cast<float>(boys::F16(static_cast<float>(v)))), k);
                ++rescaleChecked;

                if (rounded != shifted)
                {
                    ++rescaleViolations;
                }
            }
        }
    }

    // The rung family: the published range runs from m = 64 to m = 65536, and
    // the budget moves with m, so a rung is measured against m times the m = 1
    // budget rather than against the m = 1 budget. Work moves with m as well;
    // no timing is taken here, so only the budget half of that sentence is
    // checked.
    {
        auto sweepRung = [&]<double M>(int claimSingle, int claimBatch, int claimFloat, int claimHalf) {
            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double x = ref.x[i];
                    const std::size_t k = ref.Index(n, i);
                    const int region = SingleClaim(x);
                    const double regionBound = region == 0   ? kBoundSingleA
                                               : region == 1 ? kBoundSingleBand
                                               : region == 2 ? kBoundSingleB
                                                             : kBoundSingleC;
                    const double got = boys::BoysSingle<M>(n, x);
                    Measure(claimSingle,
                            n,
                            x,
                            got,
                            ref.v[k],
                            ref.decade[k],
                            M * regionBound,
                            Unrepresentable(got, -1022));
                }

                if (n == nmax)
                {
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        std::array<double, 33> out{};
                        boys::BoysAllOrders<M>(nmax, ref.x[i], out.data());

                        for (int m = 0; m <= nmax; ++m)
                        {
                            const std::size_t km = ref.Index(m, i);
                            Measure(claimBatch,
                                    m,
                                    ref.x[i],
                                    out[static_cast<std::size_t>(m)],
                                    ref.v[km],
                                    ref.decade[km],
                                    M * kBoundDoubleBatch,
                                    Unrepresentable(out[static_cast<std::size_t>(m)], -1022));
                        }
                    }
                }
            }

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::size_t k = ref.Index(n, i);
                    const float got = boys::BoysSingleF32<M>(n, static_cast<float>(ref.x[i]));
                    Measure(claimFloat,
                            n,
                            static_cast<float>(ref.x[i]),
                            static_cast<double>(got),
                            ref.vf[k],
                            ref.decadeF[k],
                            M * kBoundFloat,
                            got == 0.0f);

                    if (!std::isfinite(ref.x16[i]))
                    {
                        continue;
                    }

                    const boys::F16 goth =
                        boys::BoysSingleF16<M>(n, boys::F16(static_cast<float>(ref.x[i])));
                    const double asDouble = static_cast<double>(static_cast<float>(goth));
                    Measure(claimHalf,
                            n,
                            ref.x16[i],
                            asDouble,
                            ref.v16[k],
                            ref.decade16[k],
                            M * kBoundHalfBase +
                                0.5 * UlpOf(asDouble, kF16MantissaBits, kF16MinNormalExp),
                            Unrepresentable(asDouble, kF16MinNormalExp));
                }
            }
        };

        sweepRung.template operator()<64.0>(kSingle64, kBatch64, kFloat64, kHalf64);
        sweepRung.template operator()<65536.0>(
            kSingle65536, kBatch65536, kFloat65536, kHalf65536);
    }

    // The float lane's own floor, relative rather than absolute: the published
    // ceiling paragraph says the 24-bit significand resolves about 6e-8 and
    // that the m = 1 budget of 1.5e-7 absolute is within a factor of a few of
    // it. The absolute sweep cannot see that floor - an absolute budget is
    // generous wherever |F| is small - so it is measured where |F| >= 0.5,
    // which is where a relative floor is what the caller gets.
    double floatWorstRel = 0.0;
    int floatWorstRelN = -1;
    double floatWorstRelX = 0.0;
    {
        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t k = ref.Index(n, i);
                const double exact = ref.vf[k];

                if (std::abs(exact) < 0.5)
                {
                    continue;
                }

                const float got = boys::BoysSingleF32(n, static_cast<float>(ref.x[i]));
                const double rel = std::abs(static_cast<double>(got) - exact) / std::abs(exact);

                if (rel > floatWorstRel)
                {
                    floatWorstRel = rel;
                    floatWorstRelN = n;
                    floatWorstRelX = ref.x[i];
                }
            }
        }
    }

    // Region C is m-invariant: the branch is one closed form with no
    // coefficients, so the published row says its budget holds with slack at
    // every rung. Measured at three rungs spanning the published range, over
    // region C's own arguments, against the branch's own base budget.
    std::array<double, 3> mInvariantErr{};
    double mInvariantSpread = 0.0;
    double mInvariantMax = 0.0;
    {
        const auto sweepM = [&]<double M>(double& sink) {
            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double x = ref.x[i];

                    if (SingleClaim(x) != 3)
                    {
                        continue;
                    }

                    const double err =
                        std::abs(boys::BoysSingle<M>(n, x) - ref.v[ref.Index(n, i)]);

                    if (err > sink)
                    {
                        sink = err;
                    }

                    if (err > mInvariantMax)
                    {
                        mInvariantMax = err;
                    }
                }
            }
        };

        sweepM.template operator()<1.0>(mInvariantErr[0]);
        sweepM.template operator()<64.0>(mInvariantErr[1]);
        sweepM.template operator()<65536.0>(mInvariantErr[2]);

        double lo = mInvariantErr[0];
        double hi = mInvariantErr[0];

        for (const double e : mInvariantErr)
        {
            lo = std::min(lo, e);
            hi = std::max(hi, e);
        }

        mInvariantSpread = lo > 0.0 ? hi / lo : 0.0;
    }

    // ---- the evaluation-scheme book -----------------------------------------
    // One accumulator per row the public surface enumerates and per entry that
    // reads a scheme: every (scheme, stored fit) pair BoysEvalSchemeFits()
    // reports, then every public double-precision entry whose policy names a
    // scheme, at the shipped rung and at a relaxed one. A fit or a scheme added
    // to either enumeration is measured and judged here without this block
    // changing shape; the two schemes are the closed set the entries themselves
    // switch over, so the dispatchers below are the entries' own.
    //
    // Each fit row is that fit over the whole interval it is defined on, at
    // every order it serves, against the committed reference, judged against
    // the bound the public surface reports for that pair in this build's
    // multiply-add route. The row therefore holds the library to what it tells
    // a consumer, in the arithmetic this build runs, and not to a number
    // written beside the measurement.
    //
    // Each entry row is one public entry over the whole reference grid at one
    // scheme and one rung, judged against the bound that entry documents. The
    // entries are the ones a scheme reaches: the per-argument entries through
    // the fits' summation, the many-argument and fixed-order entries through
    // the region bodies they run, and the relaxed rungs through the
    // effective-degree variants the m = 1 bodies never call. A fit row cannot
    // show that an option is reachable from an entry, and an option that is
    // implemented but unreachable is not delivered.
    //
    // Sweeping the entries is not on its own enough, and the carriage table
    // below is why. Every one of these rows would pass at either scheme if an
    // entry stopped passing the scheme to its call: the two schemes' fits hold
    // the same bar, so dropping one for the other moves a value by a last-place
    // digit and no bound can see it. So each entry is read at both schemes in
    // the same pass and the cells where the two readings differ are counted. A
    // row whose readings agree on every cell it covers is a row the selection
    // does not reach, whatever its accuracy says - which is the check the
    // accuracy column cannot make.
    const std::span<const boys::EvalFitInfo> schemeFitRows = boys::BoysEvalSchemeFits();
    std::vector<int> schemeFitSlots;
    schemeFitSlots.reserve(schemeFitRows.size());

    // The cells each row was read at under both schemes, and the cells where
    // the two readings differed - per region, because a scheme reaches a row
    // region by region: the batch entries read it in the region-A and region-B
    // bodies and not in the asymptotic one, and a row that lost the argument in
    // one of those bodies still differs in the others. Counted per region, a
    // drop of one region's argument is a row of zeros beside three rows of
    // numbers rather than a total that still looks healthy.
    struct SchemeCarriage {
        const char* entry = "";
        const char* rung = "";
        std::array<std::size_t, 4> cells{};
        std::array<std::size_t, 4> differ{};
    };

    // The cells each stored fit was read at under both schemes, and the cells
    // where the two readings differed - one row per lane, because the two
    // schemes' fits of a lane are the same fit read two ways and the question
    // the count answers is one question per lane.
    std::array<SchemeCarriage, 3> schemeFitCarriage{};

    for (const boys::EvalFitInfo& fit : schemeFitRows)
    {
        SchemeCarriage& car = schemeFitCarriage[static_cast<std::size_t>(fit.lane)];
        car.entry = EvalLaneName(fit.lane);
        car.rung = "m = 1";
        schemeFitSlots.push_back(AddSchemeClaim(boys::EvalSchemeName(fit.scheme),
                                                EvalLaneName(fit.lane),
                                                boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane)));
    }

    // The call shape a row is one of. Each is a public entry, and each reads the
    // scheme its policy names somewhere between the call site and the fit.
    enum class SchemeEntryKind : std::uint8_t {
        kSingle,     // BoysSingle: one order at one argument
        kOrders,     // BoysAllOrders: every order at one argument
        kFixedN,     // BoysFixedN: one order over the arguments of an array
        kAllN,       // BoysAllN: every order over an array, the grouping done inside
        kAllNSorted, // BoysAllN with the BoysSortedArgs overload
        kTierEntry,  // BoysAllOrdersAtTier: the scheme named at run time
    };

    struct SchemeEntry {
        boys::EvalScheme scheme = boys::EvalScheme::kSplitClenshaw;
        const char* entry = "";
        const char* rung = "";
        SchemeEntryKind kind = SchemeEntryKind::kSingle;
        double multiplier = 1.0;
        int slot = -1;
        int carriage = -1;
    };

    // The rows of the entry table. Written once and instantiated per scheme the
    // enumeration reports, so a scheme added to the enumeration is measured
    // without this list changing. Each row is a public entry at a rung, and a
    // relaxed rung is the same entry reading the effective-degree tables its
    // m = 1 body never touches.
    struct EntryRow {
        const char* entry;
        const char* rung;
        SchemeEntryKind kind;
        double multiplier;
    };

    std::vector<EntryRow> entryRows{
        {"single entry", "m = 1", SchemeEntryKind::kSingle, 1.0},
        {"orders entry", "m = 1", SchemeEntryKind::kOrders, 1.0},
        {"fixed-n entry", "m = 1", SchemeEntryKind::kFixedN, 1.0},
        {"all-n entry", "m = 1", SchemeEntryKind::kAllN, 1.0},
        {"all-n sorted", "m = 1", SchemeEntryKind::kAllNSorted, 1.0},
        {"single entry", "m = 64", SchemeEntryKind::kSingle, 64.0},
        {"orders entry", "m = 64", SchemeEntryKind::kOrders, 64.0},
        {"fixed-n entry", "m = 64", SchemeEntryKind::kFixedN, 64.0},
        {"all-n entry", "m = 64", SchemeEntryKind::kAllN, 64.0},
#ifdef BOYS_GATE_TIER
        // The run-time selector, whose scheme is read before any body runs.
        {"tier entry", "m = 64", SchemeEntryKind::kTierEntry, 64.0},
#endif
    };

    std::vector<SchemeEntry> schemeEntries;
    std::vector<SchemeCarriage> schemeEntryCarriage;

    for (const EntryRow& row : entryRows)
    {
        SchemeCarriage car;
        car.entry = row.entry;
        car.rung = row.rung;
        schemeEntryCarriage.push_back(car);

        const double baseBound = row.kind == SchemeEntryKind::kSingle ? kBoundSingleC
                                                                     : kBoundDoubleBatch;

        // Both schemes share the one carriage slot: the pair is the same call
        // with one template argument, and the question the slot answers - does
        // this row read the scheme it names - is one question per row.
        for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
        {
            SchemeEntry e;
            e.scheme = info.scheme;
            e.entry = row.entry;
            e.rung = row.rung;
            e.kind = row.kind;
            e.multiplier = row.multiplier;
            e.slot = AddSchemeClaim(info.name, row.entry, row.multiplier * baseBound);
            e.carriage = static_cast<int>(schemeEntryCarriage.size()) - 1;
            schemeEntries.push_back(e);
        }
    }

    // The reference grid in non-decreasing order, with the permutation that
    // takes a position in it back to the argument's own index: the sorted
    // overload is measured by that map rather than by a re-sorted reference.
    std::vector<std::size_t> sortedPerm(count);

    for (std::size_t i = 0; i < count; ++i)
    {
        sortedPerm[i] = i;
    }

    std::sort(sortedPerm.begin(), sortedPerm.end(),
              [&](std::size_t a, std::size_t b) { return ref.x[a] < ref.x[b]; });

    std::vector<double> refSorted(count);

    for (std::size_t j = 0; j < count; ++j)
    {
        refSorted[j] = ref.x[sortedPerm[j]];
    }

    const auto sweepSchemeFits = [&]<boys::EvalScheme kScheme>() {
        constexpr boys::EvalScheme kOther = kScheme == boys::EvalScheme::kSplitClenshaw
                                                ? boys::EvalScheme::kHorner
                                                : boys::EvalScheme::kSplitClenshaw;
        constexpr bool kFirst = kScheme == boys::EvalScheme::kSplitClenshaw;

        for (std::size_t row = 0; row < schemeFitRows.size(); ++row)
        {
            const boys::EvalFitInfo& fit = schemeFitRows[row];

            if (fit.scheme != kScheme)
            {
                continue;
            }

            const double bound = boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane);
            SchemeCarriage& car = schemeFitCarriage[static_cast<std::size_t>(fit.lane)];

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double x = ref.x[i];

                    if (!InFitDomain(fit.lane, n, x))
                    {
                        continue;
                    }

                    const std::size_t k = ref.Index(n, i);
                    const double got = FitValue<kScheme>(fit.lane, n, x);
                    MeasureAt(SchemeClaims()[static_cast<std::size_t>(schemeFitSlots[row])],
                              n,
                              x,
                              got,
                              ref.v[k],
                              ref.decade[k],
                              bound,
                              false);

                    if constexpr (kFirst)
                    {
                        const double other = FitValue<kOther>(fit.lane, n, x);
                        const std::size_t r = static_cast<std::size_t>(SingleClaim(x));
                        ++car.cells[r];

                        if (std::memcmp(&got, &other, sizeof(double)) != 0)
                        {
                            ++car.differ[r];
                        }
                    }
                }
            }
        }
    };

    // One entry row at one scheme and one rung. The other scheme's reading is
    // taken in the same pass and only to count the cells where the two differ:
    // the count is the whole reason the entry table exists, and it is taken
    // once per (entry, rung) rather than once per scheme row.
    const auto sweepEntryScheme = [&]<boys::EvalScheme kScheme, double kM>(SchemeEntry& e) {
        constexpr bool kFirst = kScheme == boys::EvalScheme::kSplitClenshaw;
        constexpr boys::EvalScheme kOther = kScheme == boys::EvalScheme::kSplitClenshaw
                                                ? boys::EvalScheme::kHorner
                                                : boys::EvalScheme::kSplitClenshaw;
        SchemeCarriage& car = schemeEntryCarriage[static_cast<std::size_t>(e.carriage)];
        Accum& acc = SchemeClaims()[static_cast<std::size_t>(e.slot)];

        const auto note = [&car](double a, double b, double x) {
            const std::size_t r = static_cast<std::size_t>(SingleClaim(x));
            ++car.cells[r];

            if (std::memcmp(&a, &b, sizeof(double)) != 0)
            {
                ++car.differ[r];
            }
        };

        switch (e.kind)
        {
        case SchemeEntryKind::kSingle:
            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const double x = ref.x[i];
                    const std::size_t k = ref.Index(n, i);
                    const double got = boys::BoysSingle<kM, SchemePolicy<kScheme>>(n, x);
                    MeasureAt(acc,
                              n,
                              x,
                              got,
                              ref.v[k],
                              ref.decade[k],
                              e.multiplier * SingleBound(x),
                              Unrepresentable(got, -1022));

                    if constexpr (kFirst)
                    {
                        note(got, boys::BoysSingle<kM, SchemePolicy<kOther>>(n, x), x);
                    }
                }
            }

            break;

        case SchemeEntryKind::kOrders:
            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> a{};
                std::array<double, 33> b{};
                boys::BoysAllOrders<kM, SchemePolicy<kScheme>>(nmax, ref.x[i], a.data());

                if constexpr (kFirst)
                {
                    boys::BoysAllOrders<kM, SchemePolicy<kOther>>(nmax, ref.x[i], b.data());
                }

                for (int m = 0; m <= nmax; ++m)
                {
                    const std::size_t km = ref.Index(m, i);
                    MeasureAt(acc,
                              m,
                              ref.x[i],
                              a[static_cast<std::size_t>(m)],
                              ref.v[km],
                              ref.decade[km],
                              e.multiplier * kBoundDoubleBatch,
                              Unrepresentable(a[static_cast<std::size_t>(m)], -1022));

                    if constexpr (kFirst)
                    {
                        note(a[static_cast<std::size_t>(m)], b[static_cast<std::size_t>(m)], ref.x[i]);
                    }
                }
            }

            break;

        case SchemeEntryKind::kFixedN:
            for (int n = 0; n <= nmax; ++n)
            {
                std::vector<double> a(count);
                std::vector<double> b(count);
                boys::BoysFixedN<kM, SchemePolicy<kScheme>>(n, ref.x.data(), a.data(), count);

                if constexpr (kFirst)
                {
                    boys::BoysFixedN<kM, SchemePolicy<kOther>>(n, ref.x.data(), b.data(), count);
                }

                for (std::size_t i = 0; i < count; ++i)
                {
                    const std::size_t k = ref.Index(n, i);
                    MeasureAt(acc,
                              n,
                              ref.x[i],
                              a[i],
                              ref.v[k],
                              ref.decade[k],
                              e.multiplier * kBoundDoubleBatch,
                              Unrepresentable(a[i], -1022));

                    if constexpr (kFirst)
                    {
                        note(a[i], b[i], ref.x[i]);
                    }
                }
            }

            break;

        case SchemeEntryKind::kAllN:
        case SchemeEntryKind::kAllNSorted:
        {
            const bool sorted = e.kind == SchemeEntryKind::kAllNSorted;
            const std::vector<double>& args = sorted ? refSorted : ref.x;
            std::vector<double> a(count * static_cast<std::size_t>(nmax + 1));
            std::vector<double> b(count * static_cast<std::size_t>(nmax + 1));

            if (sorted)
            {
                boys::BoysAllN<kM, SchemePolicy<kScheme>>(nmax, args.data(), a.data(), count,
                                                          boys::BoysSortedArgs{});

                if constexpr (kFirst)
                {
                    boys::BoysAllN<kM, SchemePolicy<kOther>>(nmax, args.data(), b.data(), count,
                                                             boys::BoysSortedArgs{});
                }
            } else
            {
                boys::BoysAllN<kM, SchemePolicy<kScheme>>(nmax, args.data(), a.data(), count);

                if constexpr (kFirst)
                {
                    boys::BoysAllN<kM, SchemePolicy<kOther>>(nmax, args.data(), b.data(), count);
                }
            }

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t j = 0; j < count; ++j)
                {
                    // The sorted call lays its planes out in its own argument
                    // order, so a position in them maps back through the
                    // permutation rather than being the reference's own index.
                    const std::size_t i = sorted ? sortedPerm[j] : j;
                    const std::size_t k = ref.Index(n, i);
                    const std::size_t p = ref.Index(n, j);
                    MeasureAt(acc,
                              n,
                              ref.x[i],
                              a[p],
                              ref.v[k],
                              ref.decade[k],
                              e.multiplier * kBoundDoubleBatch,
                              Unrepresentable(a[p], -1022));

                    if constexpr (kFirst)
                    {
                        note(a[p], b[p], ref.x[i]);
                    }
                }
            }

            break;
        }

#ifdef BOYS_GATE_TIER
        case SchemeEntryKind::kTierEntry:
            // The run-time selector, which reads the scheme and dispatches on it
            // before any body runs; both readings are named at run time, so both
            // are taken here rather than one per instantiation.
            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> a{};
                std::array<double, 33> b{};
                boys::BoysAllOrdersAtTier(
                    boys::AccuracyTier::kRelaxed64, e.scheme, nmax, ref.x[i], a.data());
                boys::BoysAllOrdersAtTier(
                    boys::AccuracyTier::kRelaxed64, kOther, nmax, ref.x[i], b.data());

                for (int m = 0; m <= nmax; ++m)
                {
                    const std::size_t km = ref.Index(m, i);
                    MeasureAt(acc,
                              m,
                              ref.x[i],
                              a[static_cast<std::size_t>(m)],
                              ref.v[km],
                              ref.decade[km],
                              e.multiplier * kBoundDoubleBatch,
                              Unrepresentable(a[static_cast<std::size_t>(m)], -1022));
                    note(a[static_cast<std::size_t>(m)], b[static_cast<std::size_t>(m)], ref.x[i]);
                }
            }

            break;
#endif
        }
    };

    const auto sweepEntry = [&]<double kM>(SchemeEntry& e) {
        if (e.scheme == boys::EvalScheme::kSplitClenshaw)
        {
            sweepEntryScheme.template operator()<boys::EvalScheme::kSplitClenshaw, kM>(e);
        } else
        {
            sweepEntryScheme.template operator()<boys::EvalScheme::kHorner, kM>(e);
        }
    };

    sweepSchemeFits.template operator()<boys::EvalScheme::kSplitClenshaw>();
    sweepSchemeFits.template operator()<boys::EvalScheme::kHorner>();

    for (SchemeEntry& e : schemeEntries)
    {
        if (e.multiplier == 1.0)
        {
            sweepEntry.template operator()<1.0>(e);
        } else
        {
            sweepEntry.template operator()<64.0>(e);
        }
    }

    // ---- the packed region-A lane, by scheme -------------------------------
    // The many-argument entry hands its low-order region-A runs to the packed
    // AVX2 lane where the build has one. That lane holds the shipped Chebyshev
    // coefficients and the split Clenshaw recurrence, so it serves the shipped
    // scheme and no other: a call naming another scheme is answered on the
    // scalar body instead. What the caller loses there is a lane and not a
    // value - the bound is the same and the values are the per-argument entry's
    // own - so no accuracy row in any book can see it and no returned value can
    // show it. The library states the scope in BoysPackedLaneServes instead, and
    // what is measured here is that statement against the values: with the
    // shipped scheme the entry's region-A runs differ from the per-argument
    // entry's (that is what the lane is), and with the other scheme they agree
    // with it bit for bit (that is the scalar body, which is that entry's own).
    struct LaneTier {
        boys::EvalScheme scheme = boys::EvalScheme::kSplitClenshaw;
        const char* name = "";
        bool reached = false;
        std::size_t cells = 0;
        std::size_t differ = 0;
    };

    std::vector<LaneTier> laneTiers;

    if (boys::BoysAvx2Available())
    {
        const auto laneTierOf = [&]<boys::EvalScheme kScheme>() {
            constexpr int kLaneNmax = boys::detail::kBoysAllNLaneMaxOrder;
            LaneTier t;
            t.scheme = kScheme;
            t.name = boys::EvalSchemeName(kScheme);
            std::vector<double> batch(count * static_cast<std::size_t>(kLaneNmax + 1));
            boys::BoysAllN<1.0, SchemePolicy<kScheme>>(
                kLaneNmax, ref.x.data(), batch.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                // Region A is the lane's scope; the other regions are served by
                // bodies both entries share, so a difference there would be a
                // finding of another kind.
                if (!(ref.x[i] < boys::detail::kX0))
                {
                    continue;
                }

                std::array<double, 33> per{};
                boys::BoysAllOrders<1.0, SchemePolicy<kScheme>>(kLaneNmax, ref.x[i], per.data());

                for (int n = 0; n <= kLaneNmax; ++n)
                {
                    ++t.cells;

                    if (std::memcmp(&batch[ref.Index(n, i)], &per[static_cast<std::size_t>(n)],
                                    sizeof(double)) != 0)
                    {
                        ++t.differ;
                    }
                }
            }

            t.reached = t.differ > 0;
            laneTiers.push_back(t);
        };

        laneTierOf.template operator()<boys::EvalScheme::kSplitClenshaw>();
        laneTierOf.template operator()<boys::EvalScheme::kHorner>();
    }

    // What a scheme reaches at each rung, region by region, read off the
    // per-argument entry. That entry reads the scheme through the fits the lane
    // is certified on - the route's own family at the scheme's own summation -
    // and it is the reference every carriage row above is judged against: a row
    // is required to differ where this one does, and nowhere is a region
    // written down as one a scheme ought to reach. Where this reference shows no
    // difference over a region, the region is one no scheme reaches through this
    // entry, and no row is held to it.
    std::array<std::size_t, 4> schemeRefDiffer{};
    std::array<std::size_t, 4> schemeRefDifferRelaxed{};

    const auto referenceCarriage = [&]<double kM>(std::array<std::size_t, 4>& sink) {
        for (std::size_t i = 0; i < count; ++i)
        {
            std::array<double, 33> a{};
            std::array<double, 33> b{};
            boys::BoysAllOrders<kM, SchemePolicy<boys::EvalScheme::kSplitClenshaw>>(
                nmax, ref.x[i], a.data());
            boys::BoysAllOrders<kM, SchemePolicy<boys::EvalScheme::kHorner>>(
                nmax, ref.x[i], b.data());

            const std::size_t r = static_cast<std::size_t>(SingleClaim(ref.x[i]));

            for (int m = 0; m <= nmax; ++m)
            {
                if (std::memcmp(&a[static_cast<std::size_t>(m)], &b[static_cast<std::size_t>(m)],
                                sizeof(double)) != 0)
                {
                    ++sink[r];
                    break;
                }
            }
        }
    };

    referenceCarriage.template operator()<1.0>(schemeRefDiffer);
    referenceCarriage.template operator()<64.0>(schemeRefDifferRelaxed);

    // ---- the packing book ---------------------------------------------------
    // Two rows per scheme the orders axis carries, because the axis's answer
    // and the entry's answer cover different intervals and only one of them is
    // the packed lane:
    //
    //   "its own domain"  the arguments the packed lane evaluates, x < kX0,
    //                     against the per-order region-A bar the fits are
    //                     certified at. This is the row the axis is a claim
    //                     about.
    //   "whole grid"      every argument of the reference grid, against the
    //                     single lane's per-region budgets. Past kX0 the entry
    //                     runs the certified scalar single lane one order at a
    //                     time, so this row measures the fallback rather than
    //                     the lane, and it is here so that the fallback is
    //                     measured rather than assumed.
    //
    // The axis itself has no cell of its own: it is a property of which values
    // a call produces, so a row is judged on what the entry returned, at the
    // bound the entry documents.
    std::vector<int> packSlots;
    std::vector<boys::EvalScheme> packSchemes;

    for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
    {
        packSchemes.push_back(info.scheme);
        packSlots.push_back(AddPackClaim(info.name, "orders axis, region A", kBoundSingleA));
        packSlots.push_back(AddPackClaim(info.name, "orders axis, A..C", kBoundSingleC));
    }

    const auto sweepPackAxis = [&]<boys::EvalScheme kScheme>() {
        for (std::size_t row = 0; row < packSchemes.size(); ++row)
        {
            if (packSchemes[row] != kScheme)
            {
                continue;
            }

            const std::size_t regionASlot = static_cast<std::size_t>(packSlots[2 * row]);
            const std::size_t wholeGridSlot = static_cast<std::size_t>(packSlots[2 * row + 1]);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];
                std::array<double, 33> out{};
                boys::BoysAllOrders<1.0, OrdersPackPolicy<kScheme>>(nmax, x, out.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double got = out[static_cast<std::size_t>(n)];

                    if (x < boys::detail::kX0)
                    {
                        // The axis is served by the packed AVX2 lane where the host has
                        // one and by the certified scalar single lane where it has not.
                        // Two arithmetics over one domain deliver two figures, so the row
                        // is judged against the figure for the host it ran on: the packed
                        // lane's per-order bar where the lane is present, and the scalar
                        // single lane's own per-region budget where the axis falls back to
                        // it. Below the extended boundary the two figures are the same.
                        const double regionABound =
                            boys::BoysAvx2Available() ? kBoundSingleA : SingleBound(x);

                        MeasureAt(PackClaims()[regionASlot],
                                  n,
                                  x,
                                  got,
                                  ref.v[k],
                                  ref.decade[k],
                                  regionABound,
                                  Unrepresentable(got, -1022));
                    }

                    MeasureAt(PackClaims()[wholeGridSlot],
                              n,
                              x,
                              got,
                              ref.v[k],
                              ref.decade[k],
                              SingleBound(x),
                              Unrepresentable(got, -1022));
                }
            }
        }
    };

    sweepPackAxis.template operator()<boys::EvalScheme::kSplitClenshaw>();
    sweepPackAxis.template operator()<boys::EvalScheme::kHorner>();

    // The same axis on the plane entry, which carries it too. The rows are the
    // all-orders entry's rows at the same bounds, and deliberately so: naming
    // the axis on either entry returns the same values, because the plane
    // entry's per-argument path IS the all-orders entry's body. What the two
    // rows measure is therefore not the arithmetic - there is one arithmetic -
    // but that the axis is carried on the second call shape at all, which is
    // the thing that was refused before and is a claim about the surface.
    //
    // The row is judged on the plane entry's own contract, which is the double
    // batch row: m*5.5e-14 in every region, and inside the packed lane's
    // interval the tighter per-order bar the lane's fits are certified at, the
    // same two figures the all-orders rows use. The cells are the same grid
    // cells, reached through the plane layout instead of the order vector.
    std::vector<int> packPlaneSlots;

    for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
    {
        packPlaneSlots.push_back(
            AddPackClaim(info.name, "orders axis on the plane entry, region A", kBoundSingleA));
        packPlaneSlots.push_back(
            AddPackClaim(info.name, "orders axis on the plane entry, A..C", kBoundDoubleBatch));
    }

    const auto sweepPlanePackAxis = [&]<boys::EvalScheme kScheme>() {
        for (std::size_t row = 0; row < packSchemes.size(); ++row)
        {
            if (packSchemes[row] != kScheme)
            {
                continue;
            }

            const std::size_t regionASlot = static_cast<std::size_t>(packPlaneSlots[2 * row]);
            const std::size_t wholeGridSlot = static_cast<std::size_t>(packPlaneSlots[2 * row + 1]);
            std::vector<double> planes(count * (static_cast<std::size_t>(nmax) + 1));

            boys::BoysAllN<1.0, OrdersPackPolicy<kScheme>>(
                nmax, ref.x.data(), planes.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double got = planes[static_cast<std::size_t>(n) * count + i];

                    if (x < boys::detail::kX0)
                    {
                        // Same two-arithmetic split as the all-orders rows: the
                        // packed lane where the host has one, the certified
                        // scalar single lane where it has not, and the packed
                        // lane's own bar where it ran.
                        const double regionABound =
                            boys::BoysAvx2Available() ? kBoundSingleA : SingleBound(x);

                        MeasureAt(PackClaims()[regionASlot],
                                  n,
                                  x,
                                  got,
                                  ref.v[k],
                                  ref.decade[k],
                                  regionABound,
                                  Unrepresentable(got, -1022));
                    }

                    // The plane entry's whole-grid row is the batch bound, not
                    // the single lane's per-region table: a plane call promises
                    // the batch row everywhere, and past the lane's interval it
                    // runs the certified scalar single lane one order at a
                    // time, which is far inside that.
                    MeasureAt(PackClaims()[wholeGridSlot],
                              n,
                              x,
                              got,
                              ref.v[k],
                              ref.decade[k],
                              kBoundDoubleBatch,
                              Unrepresentable(got, -1022));
                }
            }
        }
    };

    sweepPlanePackAxis.template operator()<boys::EvalScheme::kSplitClenshaw>();
    sweepPlanePackAxis.template operator()<boys::EvalScheme::kHorner>();

    std::printf("\naccuracy gate, revision %s\n", BoysGateRevision);
    std::printf("  reference: %s (%zu arguments per order, %zu orders, %s)\n",
                reference.c_str(),
                ref.x.size(),
                static_cast<std::size_t>(nmax) + 1,
                "independent of the code under test");
    std::printf("per-lane, per-region: delivered error / documented bound\n");
    std::printf("  %-24s %-9s %8s %8s  %-24s %-24s %8s %8s\n",
                "lane",
                "region",
                "points",
                "real",
                "delivered / claimed",
                "max ratio (n, x)",
                "vacuous",
                "no value");
    std::printf("  %s\n", std::string(126, '-').c_str());

    const int firstClaim = kSingleA;
    // Through the last slot: the native half lane's two entries and the run-time
    // tier's rungs are created after the static lanes, and a slot with no points
    // (a lane this revision does not carry) prints as a row of zeros rather than
    // silently missing from the table.
    const int lastClaim = kTierRung[6] + 1;

    for (int i = firstClaim; i < lastClaim; ++i)
    {
        PrintClaim(Claims()[static_cast<std::size_t>(i)]);
    }

    std::printf("\n  delivered / claimed = |F_hat(n,x) - F(n,x)| and the bound it was judged\n"
                "             against, at the worst cell of the sweep; a claim whose bound is\n"
                "             per-region or per-value (the half lanes' half-ULP term, the rungs'\n"
                "             m times B_region) shows the pair at that cell, not the base bound\n"
                "  vacuous  = points where the documented bound is at least as large as\n"
                "             |F_n(x)| itself, so any returned value in range passes\n"
                "  no value = of those, points where the lane returned zero or a subnormal\n");

    // ---- the published cells and ranges, measured ---------------------------
    std::printf("\npublished cells and ranges, measured:\n");

    const double cellRatio = cellBound > 0.0 ? cellErr / cellBound : -1.0;
    const double cellShare =
        cellBound > 0.0 ? (0.5 * UlpOf(cellValue, kF16MantissaBits, kF16MinNormalExp)) / cellBound
                        : -1.0;

    std::printf("  half worst cell (n=0, x=721): documented 0.990 of budget\n"
                "    delivered %.6g vs reference %.6g -> err %.6g, bound %.6g, ratio %.4f\n"
                "    representation term share of that budget: measured %.4f, documented 0.99\n",
                cellValue,
                cellRef,
                cellErr,
                cellBound,
                cellRatio,
                cellShare);

    const Accum& f16Single = Claims()[static_cast<std::size_t>(kF16Single)];
    const Accum& bf16Single = Claims()[static_cast<std::size_t>(kBf16Single)];
    std::printf("  half lane, sweep worst: fp16 %.4g at (n=%d, x=%.6g); bf16 %.4g at (n=%d, x=%.6g)\n",
                f16Single.worstRatio,
                f16Single.worstN,
                f16Single.worstX,
                bf16Single.worstRatio,
                bf16Single.worstN,
                bf16Single.worstX);

    std::printf("  half lane range, region C, per order (reference | lane):\n"
                "    a -1 is 'no argument of the branch on the grid', not 'x = -1'\n");
    std::printf("    %5s %14s %14s %14s %14s %14s\n",
                "order",
                "ref normal to",
                "lane normal to",
                "ref zero to",
                "lane != 0 to",
                "scaled limit");

    for (int n = 0; n <= nmax; ++n)
    {
        if (n > 8 && n != nmax)
        {
            continue;
        }

        std::printf("    %5d %14.6g %14.6g %14.6g %14.6g %14.6g\n",
                    n,
                    refNormalX[static_cast<std::size_t>(n)],
                    laneNormalX[static_cast<std::size_t>(n)],
                    refZeroX[static_cast<std::size_t>(n)],
                    laneNonzeroX[static_cast<std::size_t>(n)],
                    refSpanNormalX[static_cast<std::size_t>(n)]);
    }

    std::printf("    (scaled limit = the largest argument at which the order's whole ladder,\n"
                "     seed included, still fits what a per-order power-of-two scale can hold in\n"
                "     binary16's normal range; the subnormal-field limit is, per order, %s)\n",
                (refSpanSubnormalX[8] > 0.0 ? "printed for order 8 below" : "-"));

    std::printf("  asymptotic branch across its stated domain edge (x = 16):\n"
                "    worst err below 16 (n=%d, x=%.6g)  = %.6g\n"
                "    worst err in [16, kX1)              = %.6g\n"
                "    worst err in the shipped domain (n=%d, x=%.6g) = %.6g\n"
                "    jump across the edge, one grid step apart: %.4g to %.4g\n"
                "    jump across the edge, two-unit windows:     %.4g to %.4g\n"
                "    error just below the edge over the branch's own 5.5e-14 (orders 0..24):\n"
                "      %.4g to %.4g  (%.2f to %.2f orders)\n",
                asymOrder,
                asymAt,
                asymErrBelow,
                asymErrInside,
                asymShippedN,
                asymShippedX,
                asymErrShipped,
                asymStepJumpMin,
                asymStepJumpMax,
                asymWinJumpMin,
                asymWinJumpMax,
                asymBelowRelMin,
                asymBelowRelMax,
                std::log10(asymBelowRelMin),
                std::log10(asymBelowRelMax));

    std::printf("  power-of-two rescale in binary16: %zu (value, power) pairs checked in the\n"
                "    normal range, %zu pairs where rounding and scaling disagreed\n",
                rescaleChecked,
                rescaleViolations);

    std::printf("  region C across the rung family (the branch is one closed form, so the\n"
                "    published row says its budget is m-invariant): worst err %.4g at m=1, "
                "%.4g at m=64, %.4g at m=65536, spread %.3gx, against 5.5e-14\n",
                mInvariantErr[0],
                mInvariantErr[1],
                mInvariantErr[2],
                mInvariantSpread);

    std::printf("  float lane's own floor where |F| >= 0.5: worst relative error %.4g at "
                "(n=%d, x=%.6g);\n"
                "    the documented ceiling is about 6e-8, the m=1 budget 1.5e-7 absolute\n",
                floatWorstRel,
                floatWorstRelN,
                floatWorstRelX);

    // The trailing marker of a per-order row. A slot no row judges carries the
    // reading's status on every row it prints, so a ratio above 1.0 in this
    // table cannot be read as a failing claim: what the RESULT line leaves a
    // reader to infer by not counting the slot, the row says outright.
    const auto RowMark = [](const Accum& a, const OrderAccum& o) -> const char* {
        if (!a.judged)
        {
            return o.failures > 0 ? "  EXCEEDED (record, not judged)" : "  record, not judged";
        }

        return o.failures > 0 ? "  EXCEEDED" : "";
    };

    if (perOrder)
    {
        std::printf("\nper lane, per region, per order (delivered error beside the bound; "
                    "binds = cells where the bound is tighter than |F_n(x)|, vacuous its "
                    "complement):\n");
        std::printf("  %-24s %-9s %5s %8s %8s %8s %14s %14s %10s %14s\n",
                    "lane",
                    "region",
                    "order",
                    "points",
                    "binds",
                    "vacuous",
                    "delivered",
                    "claimed",
                    "ratio",
                    "worst x");

        for (int i = firstClaim; i < lastClaim; ++i)
        {
            const Accum& a = Claims()[static_cast<std::size_t>(i)];

            for (int n = 0; n <= nmax; ++n)
            {
                const OrderAccum& o = a.byOrder[static_cast<std::size_t>(n)];

                if (o.points == 0)
                {
                    continue;
                }

                std::printf("  %-24s %-9s %5d %8zu %8zu %8zu %14.6g %14.6g %10.4g %14.6g%s\n",
                            a.lane.c_str(),
                            a.region.c_str(),
                            n,
                            o.points,
                            o.domainPoints,
                            o.vacuous,
                            o.worstErr,
                            o.worstBound,
                            o.worstRatio,
                            o.worstX,
                            RowMark(a, o));
            }
        }
    }

    // ---- the documented claims, each with one verdict -----------------------
    std::vector<DocClaim> book;

    const auto add = [&book](const char* id,
                             const char* statement,
                             std::string source,
                             Verdict verdict,
                             std::string evidence,
                             std::string domain = {},
                             std::string falsifier = {},
                             Evidence kind = Evidence::kHardware) {
        DocClaim c;
        c.id = id;
        c.statement = statement;
        c.source = std::move(source);
        c.verdict = verdict;
        c.kind = kind;
        c.evidence = std::move(evidence);
        c.domain = std::move(domain);
        c.falsifier = std::move(falsifier);
        book.push_back(std::move(c));
    };

    const auto worstOf = [&](std::initializer_list<int> slots) {
        double worst = 0.0;
        int at = -1;

        for (const int s : slots)
        {
            const Accum& a = Claims()[static_cast<std::size_t>(s)];

            if (a.worstRatio > worst)
            {
                worst = a.worstRatio;
                at = s;
            }
        }

        const Accum& a = Claims()[static_cast<std::size_t>(at < 0 ? 0 : at)];
        return Fmt("worst %.3g of budget at (n=%d, x=%.6g): delivered %.4g against %.4g",
                   worst,
                   a.worstN,
                   a.worstX,
                   a.worstErr,
                   a.worstBound);
    };

    // The half lanes' two sides, totalled across the cells a claim covers: the
    // domain where the bound is tighter than the value, and the return's state
    // inside it. A subnormal return inside the bound is allowed - the claim is
    // about the error, not about the exponent - but a zero return is not, and
    // neither is a value outside the bound.
    struct Domain {
        std::size_t points = 0;
        std::size_t subnormal = 0;
        std::size_t zero = 0;
        std::size_t outside = 0;
        double zeroRef = 0.0;
    };

    const auto domainOf = [&](std::initializer_list<int> slots) {
        Domain d;

        for (const int s : slots)
        {
            const Accum& a = Claims()[static_cast<std::size_t>(s)];
            d.points += a.domainPoints;
            d.subnormal += a.domainSubnormal;
            d.zero += a.domainZero;
            d.outside += a.domainOutside;
            d.zeroRef = std::max(d.zeroRef, a.domainZeroRef);
        }

        return d;
    };

    const auto verdictOf = [&](std::initializer_list<int> slots) {
        std::vector<Verdict> vs;

        for (const int s : slots)
        {
            vs.push_back(FromAccum(Claims()[static_cast<std::size_t>(s)]));
        }

        bool anyExceeded = false;
        bool anyVerified = false;
        bool anyVacuous = false;

        for (const Verdict v : vs)
        {
            anyExceeded = anyExceeded || v == Verdict::Exceeded;
            anyVerified = anyVerified || v == Verdict::Verified;
            anyVacuous = anyVacuous || v == Verdict::Vacuous;
        }

        if (anyExceeded)
        {
            return Verdict::Exceeded;
        }

        return anyVerified ? Verdict::Verified : (anyVacuous ? Verdict::Vacuous : Verdict::EvidenceAbsent);
    };

    add("README.double.single",
        "double single at m = 1: 1e-15 on A, 3e-14 on the extended band and B, 5.5e-14 on C",
        "README accuracy contract (four-region table)",
        verdictOf({kSingleA, kSingleBand, kSingleB, kSingleC}),
        worstOf({kSingleA, kSingleBand, kSingleB, kSingleC}));

    add("README.double.batch",
        "double batch: 5.5e-14 in every region, all entries",
        "README accuracy contract (four-region table)",
        verdictOf({kOrders, kFixedN, kAllN}),
        worstOf({kOrders, kFixedN, kAllN}));

    add("README.float",
        "float single and batch: 1.5e-7 absolute at m = 1, the same in every region",
        "README accuracy contract",
        verdictOf({kFloatSingle, kFloatOrders}),
        worstOf({kFloatSingle, kFloatOrders}));

    add("README.half",
        "fp16 and bf16 store-half lanes: m*1e-7 + one half-ULP, single and batch",
        "README accuracy contract and include/boys/boys.hpp",
        verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}) == Verdict::Verified
            ? Verdict::MetOverDomain
            : verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        worstOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        "the arguments where |F_n(x)| > m*1e-7 + one half-ULP of the returned value, and no "
        "others - the restriction the half lanes' paragraph in docs/lane-contract.md states",
        "the row fails if the budget is read as claimed over the whole argument range: the "
        "counted-apart cells in LC.half.vacuous_floor are exactly the cells that reading would "
        "turn into failures");

    {
        // The header's contract table as it stands: the double single lane's
        // region-A cell (m*1e-15 over the per-order fits' own range) and its
        // extended-band cell (m*3e-14), the two rows the table does not
        // collapse, measured against the slots that hold them.
        const Accum& headerA = Claims()[static_cast<std::size_t>(kHeaderA)];

        add("header.double.region_A_and_band",
            "double single: m*1e-15 on region A and m*3e-14 on the extended band, the two "
            "cells the header's table keeps apart",
            "include/boys/boys.hpp preamble table (four-region, with the extended-band column)",
            verdictOf({kSingleA, kSingleBand}),
            Fmt("%s. The record of what this row used to test: the same table read as a "
                "three-region one - a single region-A cell of 1e-15 with no band row, so "
                "1e-15 across the whole of x < x0 - is not a bound the lane meets anywhere in "
                "the band. Measured over the same sweep it is exceeded by %.3g at (n=%d, "
                "x=%.6g), delivered %.4g against 1e-15; that argument is the band cell the "
                "README sizes at 3e-14, where the same delivered error is %.3g of the band "
                "bound, and the band's own worst cell across all orders is the same %.3g of "
                "3e-14. The header now states the band cell separately and both live cells "
                "are met, which is what this row's verdict reports; the withdrawn reading is "
                "still measured, above and in the per-order table under lane \"double single\", "
                "region \"header x<x0\", so the finding stays visible and re-runnable",
                worstOf({kSingleA, kSingleBand}).c_str(),
                headerA.worstRatio,
                headerA.worstN,
                headerA.worstX,
                headerA.worstErr,
                headerA.worstErr / kBoundSingleBand,
                Claims()[static_cast<std::size_t>(kSingleBand)].worstRatio));
    }

    add("LC.double.rungs",
        "double single and batch: m = 64 up to m = 65536, work and budget both move with m",
        "docs/lane-contract.md, double",
        verdictOf({kSingle64, kSingle65536, kBatch64, kBatch65536}),
        Fmt("%s; only the budget half of the sentence is measurable here (no timing taken)",
            worstOf({kSingle64, kSingle65536, kBatch64, kBatch65536}).c_str()));

    add("LC.double.ceiling",
        "the stored coefficients are correctly rounded and the fit carries 5e-19 of "
        "truncation under doubles holding 1e-16 to 2e-16, so no rung goes below that floor",
        "docs/lane-contract.md, double",
        Verdict::Verified,
        "tree command, run by the operator, not by this binary: "
        "`python tools/gen_boys_coefficients.py --check`");

    add("LC.float.rungs",
        "float: the same rungs from m = 1 to m = 65536",
        "docs/lane-contract.md, float",
        verdictOf({kFloat64, kFloat65536}),
        worstOf({kFloat64, kFloat65536}));

    add("LC.float.ceiling",
        "the ceiling is the format: a 24-bit significand resolves about 6e-8 relative and "
        "the m = 1 budget is 1.5e-7 absolute - within a factor of a few of each other",
        "docs/lane-contract.md, float",
        Verdict::Verified,
        Fmt("measured floor: worst relative error of the float lane where |F| >= 0.5 is "
            "%.3g at (n=%d, x=%.6g); the ratio 1.5e-7 / 6e-8 = 2.5 is the documented "
            "'factor of a few'",
            floatWorstRel,
            floatWorstRelN,
            floatWorstRelX));

    // ---- the region-A transform lane ---------------------------------------
    // The lane's own header and the three published documents state its
    // accuracy claims; these seven rows are those claims. Three carry the bound
    // a caller is held to, three the delivered worst the documents publish, and
    // one the multiplier's rung where the bound is allowed to move. The two
    // split modes' rows say again what those bounds are - an idealisation of a
    // 32-bit tensor-core accumulator, measured here in software - so a green
    // row is not read as a statement about a card. No row here says anything
    // about speed, and the lane's own preamble claims none.
    {
        const Accum& fp64 = Claims()[static_cast<std::size_t>(kTransformFp64)];
        const Accum& tf32 = Claims()[static_cast<std::size_t>(kTransformTf32x3)];
        const Accum& bf16 = Claims()[static_cast<std::size_t>(kTransformBf16x6)];
        const Accum& rung = Claims()[static_cast<std::size_t>(kTransformRung1024)];
        const Accum& tf32Rung = Claims()[static_cast<std::size_t>(kTransformTf32x3Rung)];
        const Accum& bf16Rung = Claims()[static_cast<std::size_t>(kTransformBf16x6Rung)];
        const Accum& splitRung = (tf32Rung.worstErr >= bf16Rung.worstErr) ? tf32Rung : bf16Rung;
        const std::size_t splitRungCells = tf32Rung.points + bf16Rung.points;
        const std::size_t splitRungFailures = tf32Rung.failures + bf16Rung.failures;
        const std::string kTransformDomain =
            "region A - both bands, x < kRegionA1Edge and kRegionA1Edge <= x < kRegionAEnd - "
            "every order 0..32, every argument of the band. The band is the entry's "
            "precondition rather than a convenience: the coefficient matrix is the band's, so "
            "the sweep groups the arguments by band itself and a cell is never measured against "
            "the other band's fits";
        const std::string kSplitIdealisation =
            " The bound is an idealisation of an fp32 tensor-core accumulator and not a "
            "measurement of a card: this sweep measures the software model the lane's preamble "
            "describes, which accumulates fp32 with round-to-nearest and a full roll-over, and "
            "a card's fused sum truncates and aligns its addends instead, so a card's error can "
            "be worse than this row's and never better. A green row here is not evidence that a "
            "card is inside the bound, and the lane's own paragraph says the same";

        add("transform.fp64.bound",
            "the region-A transform's fp64 mode: |F_hat - F| <= m*1e-15 over region A - the "
            "double single lane's region-A budget, both bands, every order, every argument",
            "include/boys/boys_transform.hpp preamble (the bounds table); docs/lane-contract.md, "
            "the region-A transform lane; README, the region-A transform paragraph",
            fp64.points == 0 ? Verdict::EvidenceAbsent : verdictOf({kTransformFp64}),
            fp64.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp, so no mode of "
                              "the lane is measured here and none of its bounds is either "
                              "confirmed or refuted")
                : worstOf({kTransformFp64}),
            kTransformDomain);

        add("transform.fp64.delivered",
            "the fp64 mode's delivered worst over region A is 1.110e-16 - the per-order fits' "
            "own truncation, which the product adds nothing measurable to",
            "include/boys/boys_transform.hpp preamble (the delivered column of its bounds "
            "table); docs/lane-contract.md; README, both of which round it to 1.11e-16",
            fp64.points == 0
                ? Verdict::EvidenceAbsent
                : ((SignificantInt(fp64.worstErr, kTransformFigures) <=
                    SignificantInt(kTransformFp64Delivered, kTransformFigures) &&
                    kTransformFp64Delivered <= kTransformFp64Bound)
                       ? Verdict::Verified
                       : Verdict::Exceeded),
            fp64.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : Fmt("this gate's grid: worst %.6g at (n=%d, x=%.6g), which is the published "
                      "figure at the %d figures the table states it to - the exact maximum is "
                      "2^-53 - against the mode's budget %.4g. The published figure is a swept "
                      "maximum, over the committed reference grid and a dense sweep of the "
                      "lower band's left end, so this row holds the code to it at the precision "
                      "the document states and cannot re-find the maximum that produced it: a "
                      "regression the four figures would carry - the fifth digit moving by one "
                      "- is what this row is sized to catch",
                      fp64.worstErr,
                      fp64.worstN,
                      fp64.worstX,
                      kTransformFigures,
                      kTransformFp64Bound),
            kTransformDomain,
            "the row fails in either direction: if a cell of the grid delivers more than the "
            "published figure the document's own number is refuted by this instrument, and if "
            "the published figure ever exceeds the mode's budget the document would be "
            "publishing a measurement outside the bound it states for the mode");

        add("transform.tf32x3.bound",
            "the region-A transform's 3xTF32 mode: |F_hat - F| <= m*1e-15 + 2.5e-7 over "
            "region A, the fp32 accumulator's own floor plus the fits' term",
            "include/boys/boys_transform.hpp preamble (the bounds table); docs/lane-contract.md, "
            "the region-A transform lane; README, the region-A transform paragraph",
            tf32.points == 0 ? Verdict::EvidenceAbsent : verdictOf({kTransformTf32x3}),
            tf32.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : worstOf({kTransformTf32x3}) + kSplitIdealisation,
            kTransformDomain,
            {},
            Evidence::kModel);

        add("transform.tf32x3.delivered",
            "the 3xTF32 mode's delivered worst over region A is 1.916e-07 - its 32-bit "
            "accumulator's floor, not the operand split's - which the prose rounds to 1.92e-07",
            "include/boys/boys_transform.hpp preamble (the delivered column of its bounds "
            "table); docs/lane-contract.md and README, which round it to 1.92e-07",
            tf32.points == 0
                ? Verdict::EvidenceAbsent
                : ((SignificantInt(tf32.worstErr, kTransformFigures) <=
                    SignificantInt(kTransformTf32x3Delivered, kTransformFigures) &&
                    kTransformTf32x3Delivered <= kTransformSplitBound)
                       ? Verdict::Verified
                       : Verdict::Exceeded),
            tf32.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : Fmt("this gate's grid: worst %.6g at (n=%d, x=%.6g), inside the published "
                      "%.4g by %.3g and inside the mode's budget %.4g. The published figure is "
                      "the maximum over the committed reference grid and a dense sweep of the "
                      "lower band's left end, where these modes are worst, and the grid alone "
                      "understates it - the lane's own preamble says so and gives the two grid "
                      "samples - so this row holds the code to the published figure at every "
                      "argument of the grid and cannot re-find the maximum that produced it: a "
                      "regression narrower than the %.2g between the two would hide from it",
                      tf32.worstErr,
                      tf32.worstN,
                      tf32.worstX,
                      kTransformTf32x3Delivered,
                      kTransformTf32x3Delivered - tf32.worstErr,
                      kTransformSplitBound,
                      kTransformTf32x3Delivered - tf32.worstErr),
            kTransformDomain + "." + kSplitIdealisation,
            {},
            Evidence::kModel);

        add("transform.bf16x6.bound",
            "the region-A transform's bf16x6 mode: |F_hat - F| <= m*1e-15 + 2.5e-7 over "
            "region A, the fp32 accumulator's own floor plus the fits' term",
            "include/boys/boys_transform.hpp preamble (the bounds table); docs/lane-contract.md, "
            "the region-A transform lane; README, the region-A transform paragraph",
            bf16.points == 0 ? Verdict::EvidenceAbsent : verdictOf({kTransformBf16x6}),
            bf16.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : worstOf({kTransformBf16x6}) + kSplitIdealisation,
            kTransformDomain,
            {},
            Evidence::kModel);

        add("transform.bf16x6.delivered",
            "the bf16x6 mode's delivered worst over region A is 1.946e-07 - its 32-bit "
            "accumulator's floor - which the prose rounds to 1.95e-07",
            "include/boys/boys_transform.hpp preamble (the delivered column of its bounds "
            "table); docs/lane-contract.md and README, which round it to 1.95e-07",
            bf16.points == 0
                ? Verdict::EvidenceAbsent
                : ((SignificantInt(bf16.worstErr, kTransformFigures) <=
                    SignificantInt(kTransformBf16x6Delivered, kTransformFigures) &&
                    kTransformBf16x6Delivered <= kTransformSplitBound)
                       ? Verdict::Verified
                       : Verdict::Exceeded),
            bf16.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : Fmt("this gate's grid: worst %.6g at (n=%d, x=%.6g), inside the published "
                      "%.4g by %.3g and inside the mode's budget %.4g. The published figure is "
                      "a swept maximum - over the committed reference grid and a dense sweep of "
                      "the lower band's left end - so this row holds the code to it at every "
                      "argument of the grid and cannot re-find the maximum that produced it: a "
                      "regression narrower than the %.2g between the two would hide from it",
                      bf16.worstErr,
                      bf16.worstN,
                      bf16.worstX,
                      kTransformBf16x6Delivered,
                      kTransformBf16x6Delivered - bf16.worstErr,
                      kTransformSplitBound,
                      kTransformBf16x6Delivered - bf16.worstErr),
            kTransformDomain + "." + kSplitIdealisation,
            {},
            Evidence::kModel);

        // The multiplier's rung. The lane's paragraph says the fp64 mode's
        // bound is the double lane's m*1e-15 at every m with the multiplier
        // live from m = 2 upward, and that the two split modes' bound is their
        // accumulator's floor over the whole documented range because the fits'
        // term cannot reach 1.9e-07 until m is about 1.9e8. One rung measures
        // both halves: the three modes' bounds at m = 1024, the width that is
        // what the multiplier buys, and the cross-over the split modes' half of
        // the sentence rests on.
        const bool widthRelaxed =
            boys::detail::BandDegreeAtMultiplier<0>(kTransformRung) <
            boys::detail::BandDegreeAtMultiplier<0>(1.0);
        const double crossOver = (kTransformSplitFloorPublished / kTransformFp64Bound) + 1.0;

        add("transform.multiplier.rung",
            "the multiplier's two halves on this lane: the fp64 mode's bound is m*1e-15 with "
            "its width relaxed from m = 2 upward, and the two split modes' bound is their "
            "accumulator's floor over the whole documented range, the fits' term not reaching "
            "1.9e-07 until m is about 1.9e8",
            "include/boys/boys_transform.hpp preamble (the multiplier paragraph); "
            "docs/lane-contract.md, the region-A transform lane",
            rung.points == 0
                ? Verdict::EvidenceAbsent
                : ((rung.failures == 0 && splitRungFailures == 0 && widthRelaxed &&
                    crossOver >= 1.0e8 && crossOver <= 3.0e8)
                       ? Verdict::Verified
                       : Verdict::Exceeded),
            rung.points == 0
                ? std::string("this revision carries no boys/boys_transform.hpp")
                : Fmt("at m = %.0f: fp64 %zu cells, worst %.6g at (n=%d, x=%.6g) against its "
                      "budget %.6g, %zu outside it; the two split modes %zu cells, worst %.6g "
                      "at (n=%d, x=%.6g) against theirs. The lower band's width is %d "
                      "coefficients at m = 1 and %d at m = %.0f, so the multiplier moves the "
                      "arithmetic and not the budget alone. The split modes' half of the "
                      "sentence is arithmetic on two published numbers rather than a "
                      "measurement, and this row carries it as such: their floor %.4g divided "
                      "by the fits' term %.4g puts the cross-over at m = %.6g, four orders "
                      "past the largest multiplier the rest of this surface samples",
                      kTransformRung,
                      rung.points,
                      rung.worstErr,
                      rung.worstN,
                      rung.worstX,
                      rung.worstBound,
                      rung.failures,
                      splitRungCells,
                      splitRung.worstErr,
                      splitRung.worstN,
                      splitRung.worstX,
                      boys::detail::BandDegreeAtMultiplier<0>(1.0),
                      boys::detail::BandDegreeAtMultiplier<0>(kTransformRung),
                      kTransformRung,
                      kTransformSplitFloorPublished,
                      kTransformFp64Bound,
                      crossOver),
            kTransformDomain,
            "the row fails if a cell at the rung is outside its mode's bound - the sentence is "
            "about every m, so m = 1024 is a cell of it and not a sample of the m = 1 row - or "
            "if the width does not relax at m = 1024, when the multiplier would buy nothing, or "
            "if the cross-over the split modes' half rests on leaves the order of magnitude the "
            "document states for it");
    }

    add("LC.half.budget",
        "the half lane stays inside its budget at every order tested",
        "docs/lane-contract.md, half",
        verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}) == Verdict::Verified
            ? Verdict::MetOverDomain
            : verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        worstOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        "the arguments where |F_n(x)| > m*1e-7 + one half-ULP of the returned value, and no "
        "others",
        "the row fails if a cell inside that domain delivers more than the bound, or if the "
        "domain's edge moves down to arguments where the return is the format's floor: the "
        "counts are in LC.half.vacuous_floor");

    {
        const auto worstSlotOf = [&](std::initializer_list<int> slots) -> const Accum& {
            const Accum* best = &Claims()[static_cast<std::size_t>(*slots.begin())];

            for (const int s : slots)
            {
                const Accum& a = Claims()[static_cast<std::size_t>(s)];

                if (a.worstRatio > best->worstRatio)
                {
                    best = &a;
                }
            }

            return *best;
        };
        const Accum& f16W = worstSlotOf({kF16Single, kF16Orders});
        const Accum& bf16W = worstSlotOf({kBf16Single, kBf16Orders});

        add("LC.half.worst_cell",
            "[corrected this revision] the lane's worst cell is not 0.990 of budget at order 0, "
            "x = 721 with a margin of 1%: that cell reproduces, but the sweep's own worst cell "
            "is within a thousandth of the budget in both formats, so the margin is thousandths "
            "of a per cent - a fragility, not a margin of 1%",
            "docs/lane-contract.md, half (read before this revision: \"a worst cell of 0.990 of "
            "budget at order 0, x = 721 - a margin of 1%\")",
            (std::abs(cellRatio - 0.990) <= 0.05 && f16W.worstRatio >= 0.999
             && bf16W.worstRatio >= 0.999)
                ? Verdict::Verified
                : Verdict::Exceeded,
            Fmt("measured at the published cell: %.5f of budget (delivered %.6g, reference "
                "%.6g, bound %.6g), which reproduces it. The sweep's worst cell is %.6f at "
                "(n=%d, x=%.6g) for fp16 - a margin of %.4g%% of budget - and %.6f at (n=%d, "
                "x=%.6g) for bf16, a margin of %.4g%%. Both are inside the bound and both are "
                "within one half-quantum of it: at that distance the 1e-7 base of the budget "
                "has been spent to the last digit, and the next representable argument, or a "
                "float engine one rounding worse underneath, spends the rest. The published "
                "0.990 understates the worst cell by a factor of a hundred, which is why the "
                "sentence now reads as fragility rather than as margin. A denser independent "
                "sweep, on its own grid, reaches 0.999225 for fp16 and 0.999915 for bf16 - "
                "margins of 0.0775%% and 0.0085%% - so the numbers here are a lower bound on "
                "the worst cell and the two instruments agree on what the sentence must say",
                cellRatio,
                cellValue,
                cellRef,
                cellBound,
                f16W.worstRatio,
                f16W.worstN,
                f16W.worstX,
                100.0 * (1.0 - f16W.worstRatio),
                bf16W.worstRatio,
                bf16W.worstN,
                bf16W.worstX,
                100.0 * (1.0 - bf16W.worstRatio)),
            {},
            "the claim would have to be wrong in the other direction: the row fails if the two "
            "measured worst cells are not within a thousandth of their budget, or if the "
            "published cell stops reproducing - and it turns red the moment either format's "
            "worst ratio reaches 1.0, which is one half-quantum away");
    }

    add("LC.half.representation_share",
        "at that magnitude the half-ULP representation term is 99% of the budget",
        "docs/lane-contract.md, half",
        std::abs(cellShare - 0.99) <= 0.02 ? Verdict::Verified : Verdict::Exceeded,
        Fmt("measured %.4f at (n=0, x=721)", cellShare));

    add("LC.half.vacuous_floor",
        "the half lanes' ceiling is a domain restriction: past it the return is the format's "
        "floor, no accuracy is claimed, and the points are counted rather than passed",
        "docs/lane-contract.md, half; include/boys/boys.hpp, the same paragraph",
        Verdict::MetOverDomain,
        Fmt("the counted side, per lane and region cell. Where the bound exceeds the value: "
            "fp16 single %zu of %zu swept points, %zu of those returning no usable value "
            "(largest discarded |F| = %.4g at n=%d, x=%.6g); fp16 batch %zu of %zu, %zu no "
            "value; bf16 single %zu of %zu, %zu no value - and the last of those is the "
            "difference the two formats' exponent ranges make, since %zu of bf16's "
            "bound-covered points return a normal bf16 and meet the bound by arithmetic. The "
            "binding side is the separate row code.half_domain_stop",
            f16Single.vacuous,
            f16Single.points,
            f16Single.vacuousZero,
            f16Single.lostSignal,
            f16Single.lostSignalN,
            f16Single.lostSignalX,
            Claims()[static_cast<std::size_t>(kF16Orders)].vacuous,
            Claims()[static_cast<std::size_t>(kF16Orders)].points,
            Claims()[static_cast<std::size_t>(kF16Orders)].vacuousZero,
            bf16Single.vacuous,
            bf16Single.points,
            bf16Single.vacuousZero,
            bf16Single.vacuous - bf16Single.vacuousZero),
        "the arguments where |F_n(x)| > m*1e-7 + one half-ULP of the returned value - the "
        "domain the half lanes' paragraph claims over, and no others",
        "the row fails if the document ever reads as claiming accuracy past that ceiling - a "
        "claim over the whole argument range would make every point past the ceiling a "
        "failure, and the counters above are the number of them; it also fails if a point "
        "past the ceiling returns a usable value the count does not carry");

    // The half lane binds where |F_n(x)| exceeds its ceiling, and F_n falls
    // with x, so region C is entirely past the ceiling for an order exactly
    // when the branch's left edge is. That is the onset order the paragraph
    // claims, so the row measures the onset rather than a proxy for it. Every
    // return at that magnitude is subnormal, and a subnormal half has one
    // quantum, so each of those orders sits against the same ceiling.
    const double subnormalCeiling =
        kBoundHalfBase + 0.5 * UlpOf(0.0, kF16MantissaBits, kF16MinNormalExp);
    std::size_t edgeIndex = count;

    for (std::size_t i = 0; i < count; ++i)
    {
        if (ref.x[i] == boys::detail::kX1)
        {
            edgeIndex = i;
            break;
        }
    }

    const bool edgeOnGrid = edgeIndex < count;

    const auto pastCeilingAtX1 = [&](int order) {
        if (!edgeOnGrid)
        {
            return false;
        }

        const double got = static_cast<double>(static_cast<float>(
            boys::BoysSingleF16(order, boys::F16(static_cast<float>(ref.x[edgeIndex])))));

        return std::fabs(ref.v16[ref.Index(order, edgeIndex)]) <=
               kBoundHalfBase + 0.5 * UlpOf(got, kF16MantissaBits, kF16MinNormalExp);
    };

    const auto edgeValue = [&](int order) {
        return edgeOnGrid ? std::fabs(ref.v16[ref.Index(order, edgeIndex)]) : 0.0;
    };

    // The order the whole of region C is past the ceiling from: 3, 4 and 5
    // carry cells the lane does claim over, so the onset is not their
    // neighbour.
    int ceilingOnset = nmax + 1;

    for (int order = 0; order <= nmax; ++order)
    {
        if (pastCeilingAtX1(order))
        {
            ceilingOnset = order;
            break;
        }
    }

    add("LC.half.range_from_order3",
        "from order 3 upward on region C every return is a subnormal half or zero, and from "
        "order 6 upward every argument of the branch is past the ceiling, so no accuracy is "
        "claimed there. Orders 3, 4 and 5 are inside the range the lane does claim over: "
        "they still carry cells whose value exceeds the ceiling",
        "docs/lane-contract.md, half",
        (refNormalX[3] < boys::detail::kX1 && f16Single.failures == 0 && ceilingOnset == 6)
            ? Verdict::Verified
            : Verdict::Exceeded,
        Fmt("reference order 3 reaches the half type's normal range nowhere at or above "
            "region C's start x=%.6g (no such argument on the grid; the largest argument "
            "of the branch that is normal is %.6g at order 2 and %.6g at order 1), so no "
            "argument of the branch is normal unscaled; the lane's largest normal return "
            "is x=%.6g at order 2 and its largest nonzero return x=%.6g at order 6 "
            "(reference %.6g); at the branch's left edge, fp16(%.6g) = %.6g, order 3 has "
            "%.4g against the subnormal ceiling %.4g, 4 has %.4g, 5 has %.4g and 6 has "
            "%.4g, so the onset order is %d; delivered worst is %.3g of budget",
            boys::detail::kX1,
            refNormalX[2],
            refNormalX[1],
            laneNormalX[2],
            laneNonzeroX[6],
            refZeroX[6],
            boys::detail::kX1,
            edgeOnGrid ? ref.x16[edgeIndex] : 0.0,
            edgeValue(3),
            subnormalCeiling,
            edgeValue(4),
            edgeValue(5),
            edgeValue(6),
            ceilingOnset,
            f16Single.worstRatio),
        "orders 3 to 32 on region C (x >= kX1), the part of the half lane's argument range "
        "where the returns are subnormal or zero",
        "the row fails if the onset order is not 6 - that is, if an order below 6 is already "
        "entirely past the ceiling, or if one from 6 upward still carries a cell inside it, "
        "which is exactly what the whole-branch reading asserts - if an argument of the "
        "branch at order 3 or above returns a normal half that the sweep does not account "
        "for, or if the reference's own order-3 value becomes normal somewhere on the branch");

    // Withdrawn: "a per-order power-of-two scale carries order 3 to x ~ 361,
    // order 4 to 129, order 8 to 30". The reaches themselves rest on the
    // packed-half prototype, which is not in this tree; what is in this tree is
    // the ceiling any such scale runs into, which LC.half.no_scaling_past_38
    // and LC.half.range_from_order3 measure from the reference. A row whose
    // evidence is absent is exactly the defect this gate exists to find, so the
    // claim goes rather than staying visible but unverified.

    add("LC.half.no_scaling_past_38",
        "beyond about x ~ 35-38 no scaling reaches order 8 at all",
        "docs/lane-contract.md, half (evidence: the same prototype)",
        (refSpanNormalX[8] >= 30.0 && refSpanNormalX[8] <= 50.0) ? Verdict::Verified
                                                                 : Verdict::Exceeded,
        Fmt("computed from the reference, no prototype needed: the ladder span F_0/F_8 exceeds "
            "binary16's normal field at x=%.6g and its whole subnormal field at x=%.6g",
            refSpanNormalX[8],
            refSpanSubnormalX[8]));

    // Withdrawn: "packed half (two per register, every operation correctly
    // rounded to half): 2.39x its budget at order 0 failing on 25.1% of the
    // branch, 2.35x at order 1 (5.7%)". Those numbers are the prototype's, and
    // the prototype is not in this tree. The technique now is: LC.packed.bound
    // below measures the native packed lane that does ship, against its own
    // published ceiling, and the store-half lanes it replaces are the rows
    // above. The prototype's number and this lane's number are not the same
    // measurement and the row says so rather than substituting one for the
    // other.

    add("LC.packed.rescale_exact",
        "a per-order power-of-two rescale does not repair the packed lane because the rescale "
        "is exact, leaving the scaled lane bit-identical to the unscaled one at every order",
        "docs/lane-contract.md, packed half",
        rescaleViolations == 0 ? Verdict::Verified : Verdict::Exceeded,
        Fmt("%zu (value, power) pairs in binary16's normal range, %zu disagreements between "
            "rounding the scaled value and scaling the rounded value",
            rescaleChecked,
            rescaleViolations));

    // The 8-wide half kernels exist only where the AVX2 tier is compiled, which
    // is x86_64 by construction. On any other target the claim has no subject,
    // so it is scoped to the targets that carry it rather than reported as
    // evidence this revision failed to produce.
#if defined(__x86_64__) || defined(_M_X64)
    constexpr bool kSimdTierTarget = true;
#else
    constexpr bool kSimdTierTarget = false;
#endif

    add("code.half_simd_budget",
        "the 8-wide half-I/O region kernels never drift from the certified scalar half lane "
        "by a value the bound cannot absorb, on the x86_64 targets that carry those kernels",
        "src/boys_simd.cpp comment on the shared half lanes",
        driftCells != 0
            ? ((driftOneSideOut == 0 && driftUnforgivenZero == 0) ? Verdict::Verified
                                                                  : Verdict::Exceeded)
            : (kSimdTierTarget ? Verdict::EvidenceAbsent : Verdict::MetOverDomain),
        Fmt("body against the certified scalar half entry at the same cell, %zu cells over "
            "three regions, both formats, every order: worst divergence %.3g half-quanta at "
            "(n=%d, x=%.6g), which is a quanta count and not an error - at these magnitudes "
            "the quantum is the format's floor. On the contract itself: %zu cells where "
            "exactly one of the two is outside the bound, and %zu where a path returned zero "
            "above its own bound (largest |F_n(x)| so lost %.4g; the contract cannot forgive "
            "those). %zu / %zu cells have one path at zero and the other not - body / scalar, "
            "the largest argument of each being x=%.6g (n=%d) and x=%.6g (n=%d) - and every "
            "one of them is a cell where the bound is looser than the value, so the zero is "
            "the format's floor and not a lost value, which is why they are counted apart "
            "from the contract test. Before the region-B kernel was changed the body returned "
            "0 at (n=32, x=11.944741727643702) where the scalar returned 5.96046e-08 and the "
            "reference is 1.53979e-07, and that cell was in the binding domain",
            driftCells,
            driftUlp,
            driftOrder,
            driftX,
            driftOneSideOut,
            driftUnforgivenZero,
            driftUnforgivenRef,
            driftSimdZero,
            driftScalarZero,
            driftSimdZeroX,
            driftSimdZeroN,
            driftScalarZeroX,
            driftScalarZeroN));

    add("code.half_simd_cells",
        "the 8-wide half-I/O region kernels hold the half lane's budget on the region each "
        "documents as its precondition",
        "src/boys_simd.cpp, region-partitioned kernels",
        verdictOf({kF16PackedA, kF16PackedB, kF16PackedC, kBf16PackedA, kBf16PackedB, kBf16PackedC}),
        worstOf({kF16PackedA, kF16PackedB, kF16PackedC, kBf16PackedA, kBf16PackedB, kBf16PackedC}));

    {
        const Domain halfDomain = domainOf({kF16Single,
                                            kF16Orders,
                                            kBf16Single,
                                            kBf16Orders,
                                            kF16PackedA,
                                            kF16PackedB,
                                            kF16PackedC,
                                            kBf16PackedA,
                                            kBf16PackedB,
                                            kBf16PackedC});

        add("code.half_domain_stop",
            "the fp16/bf16 budget is a claim only where |F_n(x)| exceeds it: there the lane "
            "returns a value inside the bound, while below it the return is a subnormal "
            "number or the format's zero, the bound is met by the format's floor, and no "
            "accuracy is claimed",
            "include/boys/boys.hpp, the half lanes' budget paragraph",
            halfDomain.points == 0
                ? Verdict::EvidenceAbsent
                : ((halfDomain.outside == 0 && halfDomain.zero == 0) ? Verdict::Verified
                                                                     : Verdict::Exceeded),
            Fmt("the ten half lane x region cells over the whole sweep: %zu points in the "
                "binding domain, %zu of them returned the format's zero (largest |F_n(x)| so "
                "discarded %.4g) and %zu landed outside the bound; %zu returns inside the "
                "domain were subnormal yet inside the bound, which the claim permits. The "
                "complement is the vacuous set counted under LC.half.vacuous_floor and never "
                "as met",
                halfDomain.points,
                halfDomain.zero,
                halfDomain.zeroRef,
                halfDomain.outside,
                halfDomain.subnormal));
    }

    // The run-time accuracy tier, in the two shapes its own documentation
    // promises: a rung delivers what its multiplier says it delivers, and the
    // query surface's report is true of the values this revision returns.
    {
        std::size_t tierFailures = 0;
        double tierWorstRatio = 0.0;
        double tierWorstAt = 0.0;
        int tierWorstOrder = -1;
        double tierWorstBound = 0.0;
        double tierWorstErr = 0.0;
        std::size_t tierWorstRungOf = 0;

        for (std::size_t r = 0; r < kTierRung.size(); ++r)
        {
            const Accum& a = Claims()[static_cast<std::size_t>(kTierRung[r])];

            if (a.points == 0)
            {
                continue;
            }

            tierFailures += a.failures;

            if (a.worstRatio > tierWorstRatio)
            {
                tierWorstRatio = a.worstRatio;
                tierWorstAt = a.worstX;
                tierWorstOrder = a.worstN;
                tierWorstBound = a.worstBound;
                tierWorstErr = a.worstErr;
                tierWorstRungOf = r;
            }
        }

        add("tier.rung.bound",
            "the run-time accuracy tier: |F_hat - F| <= m * B_region per value, m = "
            "AccuracyMultiplier(tier), for every rung from kReference to kRelaxed65536 - on the "
            "all-orders entry against its own row, m * 5.5e-14 flat, and on the single-order "
            "entry against the single lane's per-region table times m, with the rung the caller "
            "names being the rung that runs on both",
            "include/boys/boys.hpp, BoysAllOrdersAtTier and BoysSingleAtTier; README, the "
            "double batch and double single rows",
            tierCells == 0 ? Verdict::EvidenceAbsent
                           : ((tierFailures == 0 && tierSingleFailures == 0 &&
                               tierSingleStaticMismatch == 0)
                                  ? Verdict::Verified
                                  : Verdict::Exceeded),
            tierCells == 0
                ? std::string("this revision carries no BoysAllOrdersAtTier, so no rung of the "
                              "run-time tier is measured here; the compile-time rungs at m=64 "
                              "and m=65536 are separate rows above")
                : Fmt("seven rungs over %zu cells against m * 5.5e-14, the bound an all-orders "
                      "entry carries: %.4g of budget at the worst (rung %zu, n=%d, x=%.6g: "
                      "delivered %.6g against %.6g), %zu cells outside it. Per rung: m=1 %.4g, "
                      "m=64 %.4g, m=256 %.4g, m=1024 %.4g, m=4096 %.4g, m=16384 %.4g, "
                      "m=65536 %.4g. Read instead as the per-region table the same sentence "
                      "names by 'B_region', the strict worst is %.4g at (n=%d, x=%.6g; "
                      "delivered %.6g against %.6g) with %zu cells outside, and the m = 1 lane - "
                      "the non-tier batch entry, bit-identical to the first rung - is inside "
                      "that reading at %.4g, so it is the relaxed rungs and not the certified "
                      "one that the strict reading fails; the sentence, not the kernel, is what "
                      "names the wrong table: the tier's own QueryTier bases regions A and B on "
                      "the batch bound, which is the reading judged here. On the single-order "
                      "shape, which is a different call and needs its own entry: seven rungs "
                      "over %zu cells against the single lane's per-region table times m, %.4g "
                      "of budget at the worst (rung %zu, n=%d, x=%.6g: delivered %.6g against "
                      "%.6g), %zu cell(s) outside it, and %zu cell(s) where the run-time entry "
                      "differs from BoysSingle at the multiplier its tier names - the last "
                      "count is what says the rung that was asked for is the rung that ran, "
                      "since a switch that reached the wrong body would still land inside "
                      "every bound here. The route named at run time is reachable on that "
                      "shape too: %zu of %zu sampled cells differ between the rational route "
                      "and the default one at the same rung",
                      tierCells,
                      tierWorstRatio,
                      tierWorstRungOf,
                      tierWorstOrder,
                      tierWorstAt,
                      tierWorstErr,
                      tierWorstBound,
                      tierFailures,
                      tierRungRatio[0],
                      tierRungRatio[1],
                      tierRungRatio[2],
                      tierRungRatio[3],
                      tierRungRatio[4],
                      tierRungRatio[5],
                      tierRungRatio[6],
                      tierStrictRatio,
                      tierStrictOrder,
                      tierStrictX,
                      tierStrictErr,
                      tierStrictBound,
                      tierStrictCells,
                      tierStrictRefRatio,
                      tierSingleCells,
                      tierSingleWorstRatio,
                      tierSingleWorstRung,
                      tierSingleWorstOrder,
                      tierSingleWorstX,
                      tierSingleWorstErr,
                      tierSingleWorstBound,
                      tierSingleFailures,
                      tierSingleStaticMismatch,
                      tierSingleRationalDiffer,
                      tierSingleRationalCells));

        add("tier.query.sound",
            "QueryTier's report is true of the values this revision delivers: a tier reported as "
            "meeting a tolerance does not deliver an error above it, and `reachable` is not "
            "below the error the tier delivers",
            "include/boys/boys.hpp, TierCoverage and QueryTier",
            tierQueryPairs == 0
                ? Verdict::EvidenceAbsent
                : ((tierQueryMiss == 0 && tierReachableShort == 0 && tierLimitingWrong == 0)
                       ? Verdict::Verified
                       : Verdict::Exceeded),
            tierQueryPairs == 0
                ? std::string("this revision carries no QueryTier, so the surface is not "
                              "measured against the delivered values here")
                : Fmt("%zu (rung, region, tolerance) questions, each answered by the surface and "
                      "checked against the worst error this revision delivers for that rung and "
                      "region: %zu answers said a rung meets a tolerance it delivers worse than "
                      "(largest such excess %.6g at tolerance %.6g, rung %zu), %zu said "
                      "`reachable` below the delivered error (largest shortfall %.6g), and %zu "
                      "named a limiting component that is not that region's. The delivered worst "
                      "per rung and region, which is what the surface was held to, is in the "
                      "table above (one row per rung)",
                      tierQueryPairs,
                      tierQueryMiss,
                      tierQueryWorstGap,
                      tierQueryWorstTol,
                      tierWorstRung,
                      tierReachableShort,
                      tierReachableGap,
                      tierLimitingWrong));

        add("tier.runtime.static",
            "the header's claim that a run-time rung is the same code a compile-time call "
            "reaches: BoysAllOrdersAtTier's output is bit-identical to the compile-time "
            "instantiation at the same m",
            "include/boys/boys.hpp, BoysAllOrdersAtTier",
            tierStaticCells == 0
                ? Verdict::EvidenceAbsent
                : (tierStaticMismatch == 0 ? Verdict::Verified : Verdict::Exceeded),
            tierStaticCells == 0
                ? std::string("this revision carries no BoysAllOrdersAtTier")
                : Fmt("%zu values compared bit for bit at m=1, m=64 and m=65536 over the whole "
                      "sweep: %zu differ. A run-time rung that reaches a different body than "
                      "the lane its multiplier names would show here even where both are inside "
                      "the bound",
                      tierStaticCells,
                      tierStaticMismatch));
    }

    // The native packed half lane, in the shape its own contract uses: one
    // domain, one bound, one verdict each. Its suite already measures all three
    // against the certified double lane and its own sweep; what these rows add
    // is the same three claims against this gate's committed mpmath reference,
    // on this gate's grid, over the region-C arguments the grid carries in
    // binary16 - and the packing claim, which no accuracy oracle can see, is
    // refuted rather than confirmed: see the distribution numbers in its row.
    {
        const Accum& packedAcc = Claims()[static_cast<std::size_t>(kNativeHalf2)];
        const Accum& batchAcc = Claims()[static_cast<std::size_t>(kNativeHalfBatch)];

        add("native.half.bound",
            "packed half (two per register, one correctly rounded half operation per ladder "
            "step): |out[k] - 2^15 F_k(x)| <= 8 ULP of the returned value, region C only, over "
            "the arguments whose returned value is a normal half",
            "docs/lane-contract.md, packed half - the native lane",
            packedAcc.points == 0
                ? Verdict::EvidenceAbsent
                : ((packedAcc.failures == 0 && batchAcc.failures == 0) ? Verdict::MetOverDomain
                                                                      : Verdict::Exceeded),
            packedAcc.points == 0
                ? std::string("the lane is not in this revision: no boys/half2.hpp, so its "
                              "bound is neither confirmed nor refuted here")
                : Fmt("worst %.4g ULP at (n=%d, x=%.6g), over %zu cells of the packed entry "
                      "and %zu of the count entry (the count entry's own worst is %.4g ULP); "
                      "the lane's published worst is 4.243 ULP, a swept maximum and not a "
                      "proof bound, and this sweep neither upgrades nor contradicts it. "
                      "Counted apart: %zu cells of %zu where the return is a subnormal half "
                      "and the claim is the ceiling's, not the bound's",
                      nativeWorstUlp,
                      nativeWorstOrder,
                      nativeWorstX,
                      packedAcc.points,
                      batchAcc.points,
                      nativeBatchWorstUlp,
                      nativeNoClaim,
                      nativeCells),
            "region C, x >= 28.984375 = fp16(kX1), over the arguments whose returned value is a "
            "normal half - the lane offers no other region and claims no accuracy outside it",
            "the row fails if a cell inside the domain delivers more than its 8 ULP, or if the "
            "domain widens: the lane's own suite prints the same per-order maxima, and a region "
            "outside C is where the two would disagree");

        add("native.half.ceiling",
            "the ceiling is a domain restriction and past it the lane claims nothing: the "
            "return is a subnormal half, then exactly zero, by design - the crossings are the "
            "whole argument range at orders 0 and 1, x ~ 2590 at order 2, and x ~ 359 / 128 / "
            "30 at orders 3 / 4 / 8",
            "docs/lane-contract.md, packed half - the native lane",
            nativeCells == 0
                ? Verdict::EvidenceAbsent
                : (nativeExcusedFailures == 0 ? Verdict::MetOverDomain : Verdict::Exceeded),
            nativeCells == 0
                ? std::string("the lane is not in this revision")
                : Fmt("the largest scaled value the sweep left unclaimed is %.6g (at n=%d, "
                      "x=%.6g) against the smallest normal half %.6g, so every unclaimed cell "
                      "is a cell where the scaling was meant to run out: %zu of %zu cells are "
                      "past the ceiling and are counted here rather than passed. The "
                      "crossings this sweep sees: the largest argument whose return is still "
                      "normal is x=%.6g at order 2, %.6g at order 3, %.6g at order 4 and "
                      "%.6g at order 8, the smallest past the ceiling being %.6g, %.6g, "
                      "%.6g and %.6g. %zu cells returned a subnormal half for a value the "
                      "scaling should have kept normal, which is the one thing this domain "
                      "must not excuse",
                      nativeNoClaimTruth,
                      nativeNoClaimOrder,
                      nativeNoClaimX,
                      kNativeSmallestNormal,
                      nativeNoClaim,
                      nativeCells,
                      nativeCeilNormalX[2],
                      nativeCeilNormalX[3],
                      nativeCeilNormalX[4],
                      nativeCeilNormalX[8],
                      nativeCeilNoClaimX[2],
                      nativeCeilNoClaimX[3],
                      nativeCeilNoClaimX[4],
                      nativeCeilNoClaimX[8],
                      nativeExcusedFailures),
            "region C, x >= 28.984375 = fp16(kX1), where the ceiling is what the lane's own "
            "scaling reaches",
            "the row fails if a cell past the ceiling returns a value the scaling should have "
            "kept normal - counted above as nativeExcusedFailures - or if the crossings move "
            "much: they are read from the sweep, not assumed");

        add("native.half.packed",
            "the packing is real: every operation is correctly rounded to binary16 once per "
            "operation, both halves of one register, with no widening for the sequence as a "
            "whole",
            "include/boys/half2.hpp; the lane's own suite measures the same claim",
            nativeCells == 0
                ? Verdict::EvidenceAbsent
                : ((nativeBatchMismatch == 0 && nativePastOneUlp > 0) ? Verdict::MetOverDomain
                                                                     : Verdict::Exceeded),
            nativeCells == 0
                ? std::string("the lane is not in this revision")
                : Fmt("verified against the oracle only in part, and said so: this gate's "
                      "reference can say what the value is, never how the arithmetic that "
                      "produced it was arranged, so the packing is separated from the bound "
                      "and tested two other ways. (1) The error distribution: %zu of %zu "
                      "cells in the binding domain carry more than half a quantum of the true "
                      "value and %zu carry more than a whole one, which is the accumulated "
                      "signature of rounding inside the ladder - a lane that widened, "
                      "evaluated at higher precision and rounded once at the end cannot leave "
                      "more than half a quantum however it is arranged, and the wider engine "
                      "contributes about 1e-16 relative on top. That refutes the "
                      "widen-compute-round-once reading rather than confirming this one. "
                      "(2) The two entries: the count entry disagrees with the packed entry "
                      "at %zu of %zu paired cells, and the pair test runs two different "
                      "arguments through one register, so a half that read its partner's "
                      "lane would show up as a mismatch. What is not established here is the "
                      "hardware claim - that a target's packed instructions are what these "
                      "operations become - and that is the lane's own structural evidence, "
                      "not this gate's",
                      nativePastHalfUlp,
                      nativeDistTotal,
                      nativePastOneUlp,
                      nativeBatchMismatch,
                      nativeCells),
            "region C, x >= 28.984375 = fp16(kX1), the only region the lane evaluates in",
            "the row fails in the direction of its own doubt: the oracle measures values and "
            "never how they were arranged, so if the lane's structural test is wrong the number "
            "that would move is nativeBatchMismatch, and a widen-compute-round-once lane would "
            "put every cell under half a quantum - both are printed above");
    }

    {
        // The packed technique's cost, carried on the lane that ships instead
        // of on the prototype that does not.
        add("LC.packed.bound",
            "packed half spends more than the half-quantum of representation the store-half "
            "budget leaves it (the prototype's finding, measured on the lane that is here)",
            "docs/lane-contract.md, packed half - the native lane",
            nativeCells == 0
                ? Verdict::EvidenceAbsent
                : (nativeWorstUlp > 0.5 ? Verdict::Verified : Verdict::Exceeded),
            nativeCells == 0
                ? std::string("the lane is not in this revision")
                : Fmt("the native packed lane's worst cell is %.4g ULP of the returned value at "
                      "(n=%d, x=%.6g), which is %.4g times the 0.5 ULP the store-half budget "
                      "leaves for representing the result - the prototype's 'above its budget' "
                      "finding, on this lane's own numbers - and inside the lane's published "
                      "ceiling of 4.243 ULP, a swept maximum and not a proof bound. The "
                      "prototype's 2.39x at order 0 and 2.35x at order 1 are withdrawn with it: "
                      "they were measured against the store-half budget at the prototype's own "
                      "arguments, and that prototype is not in this tree",
                      nativeWorstUlp,
                      nativeWorstOrder,
                      nativeWorstX,
                      nativeWorstUlp / 0.5),
            "region C, x >= fp16(kX1) = 28.984375, and only there",
            "the row fails if the packed lane's worst cell lands at or below half a quantum - "
            "it would then be as accurate as a store-half lane and the finding would not exist "
            "- or if the lane's published ceiling moves below the measured worst");
    }

    {
        // Region B's amplification at every supported order, in both readings.
        // The sentence in the tree writes it as prod(j+1/2)/x0^l, whose value
        // at l = 32 is 65 by this measurement, not one: the only reading under
        // which "<= A_B(0) = 1" is even nearly true is the gain normalised by
        // the ladder - prod(2j-1)/(2 x0)^l - which the band edge x0 is chosen
        // to make exactly 1 at l = 32. That is the reading the independent
        // oracle used, and it is 1.846e-17 above one with the shipped double.
        double amplLiteralTop = 0.0;
        double amplNormExcess = 0.0;
        int amplNormMaxAt = -1;
        {
            DD a{1.0, 0.0};
            DD g{1.0, 0.0};
            const DD x0{boys::detail::kX0, 0.0};
            const DD twoX0{2.0 * boys::detail::kX0, 0.0};
            double amplNormMaxExcess = 0.0;

            for (int n = 1; n <= nmax; ++n)
            {
                a = DivDD(MulDD(a, DD{static_cast<double>(n) + 0.5, 0.0}), x0);
                g = DivDD(MulDD(g, DD{2.0 * static_cast<double>(n) - 1.0, 0.0}), twoX0);

                const double excess = (g.hi - 1.0) + g.lo;

                if (excess > amplNormMaxExcess)
                {
                    amplNormMaxExcess = excess;
                    amplNormMaxAt = n;
                }

                if (n == nmax)
                {
                    amplLiteralTop = a.hi + a.lo;
                    amplNormExcess = excess;
                }
            }
        }

        add("LC.regionB.amplification",
            "[corrected this revision] region B's amplification is one plus 1.8e-17 at order 32 "
            "- above one, not at most one - and that is its maximum over the supported orders",
            "include/boys/boys_impl.hpp (\"A_B(l) = prod(j+1/2)/x0^l <= A_B(0) = 1\"); "
            "docs/consumer-perspective.md (\"below one for every supported order\")",
            (amplNormExcess > 0.0 && amplNormExcess < 1e-15 && amplNormMaxAt == nmax)
                ? Verdict::Verified
                : Verdict::Exceeded,
            Fmt("the gain the band edge is chosen to make one, prod(2j-1)/(2 x0)^l with the "
                "shipped x0 = %.17g, is 1 + %.4g at l = 32 and that is its maximum over l <= 32 "
                "- evaluated in double-double, whose accumulated relative error ~1e-30 is "
                "thirteen orders under the excess it measures, and the independent oracle "
                "reaches the same 1.846e-17 by exact rational arithmetic. So the two sentences "
                "that said \"<= 1\" and \"below one\" are refuted, by 1.8e-17, and the "
                "consequence is bounded: %.3g of the tightest budget this library claims, so "
                "the sizing conclusion the bound supports - the order-0 seed degree is the one "
                "that bounds the whole batch - is unaffected. The same sentence's formula as "
                "written is a second finding: prod(j+1/2)/x0^l at l = 32 is %.6g, not 1, so "
                "read literally the sentence is wrong by a factor of 65 rather than by 1.8e-17, "
                "and the wording needs the normalisation it now carries",
                boys::detail::kX0,
                amplNormExcess,
                amplNormExcess / kBoundSingleC,
                amplLiteralTop),
            {},
            "the row fails if the excess is not positive - the old sentence would then be right "
            "- or if it is large enough to matter: at 1e-15, three per cent of the tightest "
            "budget, the sizing argument would need redoing, and this prints the number; the "
            "literal reading of the formula fails if prod(j+1/2)/x0^32 is not 65");
    }

    {
        // The coefficient-rounding figure: half a ULP of a double in [0.5, 1) is
        // 2^-54, and that is what the shipped leading coefficient can be off by.
        const double leadCoeff = boys::detail::kCoeffs[0];
        const double halfUlpLead = std::ldexp(1.0, -54);

        add("LC.coeff.rounding",
            "[corrected this revision] rounding the coefficients to double contributes about "
            "5.551e-17 - the half-ULP of the leading coefficient alone - and not 3e-17",
            "docs/consumer-perspective.md (read before this revision: \"rounding the "
            "coefficients to double contributes about 3e-17\")",
            (leadCoeff >= 0.5 && leadCoeff < 1.0 && halfUlpLead == 5.551115123125783e-17)
                ? Verdict::Verified
                : Verdict::Exceeded,
            Fmt("the shipped table's leading coefficient is %.17g (include/boys/boys_coefficients.hpp, "
                "kCoeffs[0]); its binade is [0.5, 1), whose ULP is 2^-53, so a correctly "
                "rounded double there is off by at most half of one, %.6g = 2^-54. The figure "
                "the sentence carried, 3e-17, is the half-ULP of no double in that binade (the "
                "candidates are 2^-55 = %.6g and 2^-54 = %.6g); the independent oracle's exact "
                "arithmetic on the shipped coefficient gives 5.551e-17. The fit's own truncation "
                "is 5e-19, which the row LC.double.ceiling carries",
                leadCoeff,
                halfUlpLead,
                std::ldexp(1.0, -55),
                halfUlpLead),
            {},
            "the row fails if the leading coefficient's magnitude leaves [0.5, 1) - its "
            "half-ULP would then be a different power of two, and the corrected figure would be "
            "wrong - or if the document goes back to a figure that is not the half-ULP of that "
            "binade");
    }

    {
        // What the sweep's arguments cover, per format, and the rule that keeps
        // a lane from outgrowing the grid. The rule is the generator's, stated
        // in grid(): every format narrower than binary64 is carried to its own
        // largest finite value, with every representable value in its last band
        // when the format's spacing leaves one wider than a grid step.
        std::size_t halfFinite = 0;
        std::size_t regionCArgs = 0;
        double halfArgMax = 0.0;
        double halfValueMax = 0.0;
        double gridMax = 0.0;
        std::vector<double> halfSeen;

        for (std::size_t i = 0; i < ref.x.size(); ++i)
        {
            gridMax = std::max(gridMax, ref.x[i]);

            if (ref.x[i] >= boys::detail::kX1)
            {
                ++regionCArgs;
            }

            if (std::isfinite(ref.x16[i]))
            {
                ++halfFinite;
                halfArgMax = std::max(halfArgMax, ref.x[i]);
                halfValueMax = std::max(halfValueMax, ref.x16[i]);

                if (std::find(halfSeen.begin(), halfSeen.end(), ref.x16[i]) == halfSeen.end())
                {
                    halfSeen.push_back(ref.x16[i]);
                }
            }
        }

        const double f32Max = static_cast<double>(std::numeric_limits<float>::max());

        add("code.coverage.argument_range",
            "no lane's claim is unchecked at the top of its own argument range: the sweep "
            "reaches binary16's largest finite value for the half lanes and binary32's for the "
            "floats, and nothing above binary32's maximum is a range any lane of this library "
            "evaluates in",
            "tools/gen_boys_accuracy_gate_reference.py (grid(), the rule); the coverage counts "
            "here",
            (halfValueMax >= 65504.0 && gridMax <= f32Max && halfArgMax >= 65504.0)
                ? Verdict::Verified
                : Verdict::Exceeded,
            Fmt("of the %zu arguments the sweep carries per order, %zu cast to a finite binary16 "
                "(%zu distinct fp16 values; the largest argument that does is %.6g, and the "
                "largest value reached is %.6g = the format's maximum), %zu are region-C "
                "arguments at or above kX1, and the largest argument of all is %.6g = binary32's "
                "maximum. The rule the generator applies, so a format added later does not "
                "reopen it: for every format narrower than binary64 the grid carries its largest "
                "finite value, a geometric ladder of that format's values from the old maximum "
                "up to it, and every value of the format in its last band when the spacing "
                "there leaves a band wider than a grid step - which is how 60672 to 65504 "
                "stopped being an unchecked seven per cent of the half range. Beyond binary32's "
                "maximum no narrower lane can receive an argument at all, and binary64 is never "
                "representation-limited, because the grid is a list of doubles",
                ref.x.size(),
                halfFinite,
                halfSeen.size(),
                halfArgMax,
                halfValueMax,
                regionCArgs,
                gridMax),
            {},
            "the row fails if a documented claim is ever scoped to arguments above the largest "
            "one carried for that claim's format - that region would be unchecked, and the "
            "counts above are what would say so - or if the half lanes' largest representable "
            "argument stops being reached");
    }

    add("LC.regionC.m_invariance",
        "region C is m-invariant: a single closed form with no coefficients, so its budget "
        "holds with slack at every m",
        "docs/lane-contract.md, region C",
        (mInvariantMax <= kBoundSingleC * 1.0 && mInvariantSpread <= 4.0) ? Verdict::Verified
                                                                        : Verdict::Exceeded,
        Fmt("worst region-C error %.4g at m=1, %.4g at m=64, %.4g at m=65536 (spread %.3gx); "
            "the branch's budget is 5.5e-14 at m=1",
            mInvariantErr[0],
            mInvariantErr[1],
            mInvariantErr[2],
            mInvariantSpread));

    add("LC.regionC.domain",
        "[corrected this revision] the branch's lower boundary is not a floor: through x = 16 "
        "the error varies smoothly, with no threshold and no step of orders of magnitude, "
        "growing continuously as x falls below the edge",
        "docs/lane-contract.md, region C (read before this revision: \"the error jumps five to "
        "eight orders - a hard floor, not a taper\")",
        (asymStepRatioMax < 10.0) ? Verdict::Verified : Verdict::Exceeded,
        Fmt("the shape, on the form as coded, over the arguments below x = 16 - where the "
            "sentence puts the growth - at every order: %zu neighbouring pairs of relative "
            "error, %zu of them rising as x rises, so the curve is not monotone everywhere "
            "below the edge (%zu breaks), and the largest factor between neighbours anywhere "
            "below it is %.4g at (n=%d, x=%.6g). A threshold between two regimes - which is "
            "what a floor is - would put orders of magnitude between two adjacent arguments; "
            "%.4g is %.1f orders, so the boundary is a continuation of the branch's error "
            "rather than a step at it. At the top order, where the oracle's sentence and this "
            "row's size reading both sit: %zu pairs, %zu breaks, largest neighbouring factor "
            "%.4g. Its size, for the record: the largest error relative to the value just "
            "below the edge is %.4g, at (n=%d, x=%.6g), and at order 32 the same reading gives "
            "%.4g. The two readings of the old sentence are carried here rather than in a "
            "claim: crossing the edge the jump is %.4g to %.4g over one grid step and %.4g to "
            "%.4g over two-unit windows - %.1f to %.1f orders, below the five to eight it "
            "published - and against the branch's own in-domain error %.4g, which is the "
            "5.5e-14 budget the other reading divides by, the same cells give %.4g to %.4g "
            "(%.1f to %.1f orders over orders 0..24)",
            asymPairs,
            asymPairs - asymBreaks,
            asymBreaks,
            asymStepRatioAtWorst,
            asymRatioWorstN,
            asymRatioWorstX,
            asymStepRatioAtWorst,
            std::log10(asymStepRatioAtWorst),
            asymPairsN[static_cast<std::size_t>(nmax)],
            asymBreaksN[static_cast<std::size_t>(nmax)],
            asymRatioN[static_cast<std::size_t>(nmax)],
            asymRelBelowMax,
            asymRelBelowN,
            asymRelBelowX,
            asymRelAt32,
            asymStepJumpMin,
            asymStepJumpMax,
            asymWinJumpMin,
            asymWinJumpMax,
            std::log10(asymStepJumpMin),
            std::log10(asymStepJumpMax),
            asymErrShipped,
            asymBelowRelMin,
            asymBelowRelMax,
            std::log10(asymBelowRelMin),
            std::log10(asymBelowRelMax)),
        {},
        Fmt("the row fails if any pair of neighbouring arguments below the edge differs in "
            "relative error by more than an order of magnitude - that is what a threshold "
            "between two regimes would look like, and this measurement's largest such factor is "
            "%.4g; the strictly monotone reading of the old sentence would fail on the %zu "
            "breaks the same measurement counts (%.1f%% of the pairs below the edge, %.1f%% at "
            "the top order)",
            asymStepRatioMax,
            asymBreaks,
            100.0 * static_cast<double>(asymBreaks) / static_cast<double>(asymPairs == 0 ? 1 : asymPairs),
            100.0 * static_cast<double>(asymBreaksN[static_cast<std::size_t>(nmax)])
                / static_cast<double>(asymPairsN[static_cast<std::size_t>(nmax)] == 0
                                          ? 1
                                          : asymPairsN[static_cast<std::size_t>(nmax)])));

    add("LC.evidence.ctest",
        "the m = 1 budgets and the rung family are re-checkable with ctest",
        "docs/lane-contract.md, where the numbers come from",
        Verdict::Verified,
        "tree command, run by the operator, not by this binary: "
        "`ctest --test-dir build --output-on-failure`");

    // Withdrawn: "orders 0 through 3 are 95.2% of the calls measured in a
    // production integral engine". The counts are not in this tree, and a
    // distribution measured elsewhere describes the interface that produced it
    // rather than this library's argument range; a row resting on it would be
    // unverifiable here by construction.

    // ---- the verdict table --------------------------------------------------
    std::printf("\ndocumented claims, each with one verdict:\n");
    std::printf("  thresholds a claim labels approximate rather than exact - \"about x = 16\",\n"
                "  \"about x ~ 35-38\", \"about 6e-8\" - are judged against the measured\n"
                "  value with the label kept in view, and the measured number is printed\n"
                "  beside them rather than rounded to the claim; every other threshold\n"
                "  here is a floor the document states exactly, and is compared exactly\n"
                "  every row ends with what would have to be true for its check to fail,\n"
                "  so a green row says which reading of it is still open, not only that it\n"
                "  passed;\n"
                "  a row the document itself scopes - to a region, a set of orders, a\n"
                "  range of arguments - is met over that stated domain and prints the\n"
                "  domain with its verdict: a restriction the document states is not the\n"
                "  same finding as evidence missing from the tree, and never shares its\n"
                "  verdict\n");

    int verified = 0;
    int metOverDomain = 0;
    int exceeded = 0;
    int vacuousOnly = 0;
    int absent = 0;
    int modelDerived = 0;

    for (const DocClaim& c : book)
    {
        if (c.kind == Evidence::kModel && (c.verdict == Verdict::Verified ||
                                           c.verdict == Verdict::MetOverDomain))
        {
            ++modelDerived;
        }

        switch (c.verdict)
        {
        case Verdict::Verified:
            ++verified;
            break;
        case Verdict::MetOverDomain:
            ++metOverDomain;
            break;
        case Verdict::Exceeded:
            ++exceeded;
            break;
        case Verdict::Vacuous:
            ++vacuousOnly;
            break;
        case Verdict::EvidenceAbsent:
            ++absent;
            break;
        }

        // A row resting on a model says so on its own verdict line rather than
        // leaving a reader to carry the class from the RESULT line down to the
        // row: the two verdicts look identical otherwise, and they are not.
        const std::string verdictField =
            c.kind == Evidence::kModel
                ? std::string(VerdictName(c.verdict)) + " (model)"
                : std::string(VerdictName(c.verdict));

        if (c.domain.empty())
        {
            std::printf("  [%-16s] %s\n      %s\n      %s\n      falsified by: %s\n",
                        verdictField.c_str(),
                        c.statement.c_str(),
                        c.source.c_str(),
                        c.evidence.c_str(),
                        c.falsifier.empty() ? ClassFalsifier(c.verdict) : c.falsifier.c_str());
        } else
        {
            std::printf("  [%-16s] domain: %s\n      %s\n      %s\n      %s\n"
                        "      falsified by: %s\n",
                        verdictField.c_str(),
                        c.domain.c_str(),
                        c.statement.c_str(),
                        c.source.c_str(),
                        c.evidence.c_str(),
                        c.falsifier.empty() ? ClassFalsifier(c.verdict) : c.falsifier.c_str());
        }
    }

    std::size_t gateCells = 0;
    std::size_t gateNonDiscriminating = 0;

    for (const Accum& a : Claims())
    {
        gateCells += a.points;
        gateNonDiscriminating += a.vacuous;
    }

    if (gateCells > 0)
    {
        std::printf("\n  cells: %zu comparison cells across every lane, region and order, "
                    "of which %zu (%.1f%%) carry a bound at least as large as the value "
                    "itself, so any return in range passes there and the cell cannot "
                    "discriminate; the no-cell-over-budget statement above is carried by "
                    "the remaining %zu (%.1f%%). This is the vacuous column aggregated, not "
                    "a separate finding, and it is the number to read before reading the "
                    "green rows\n",
                    gateCells,
                    gateNonDiscriminating,
                    100.0 * static_cast<double>(gateNonDiscriminating) / static_cast<double>(gateCells),
                    gateCells - gateNonDiscriminating,
                    100.0 * static_cast<double>(gateCells - gateNonDiscriminating)
                        / static_cast<double>(gateCells));
    }

    std::printf("\n  RESULT: %d of %zu claims met at this revision (%d verified outright, "
                "%d met over a stated domain, %d exceeded, %d vacuous only, %d evidence "
                "absent; neither of the last two counted as met)\n",
                verified + metOverDomain,
                book.size(),
                verified,
                metOverDomain,
                exceeded,
                vacuousOnly,
                absent);

    // The certified total, stated apart from the book's. A row whose arithmetic
    // ran on this machine is certified by its measurement; a row resting on a
    // software model of an accumulator is not, however green it is, because the
    // model is not the hardware the bound would be relied on against. The book
    // line above counts both, and this line is the number a reader who wants
    // only what hardware confirmed should quote.
    if (modelDerived > 0)
    {
        const int certified = verified + metOverDomain - modelDerived;
        std::printf("          of those, %d are certified by a measurement of this machine's own "
                    "arithmetic and %d rest on a software model of an accumulator and are NOT "
                    "certified: a model is not a card, so a green row of the second kind is no "
                    "evidence that a card is inside the bound it states\n",
                    certified,
                    modelDerived);
    }

    // The count in the RESULT line is only as wide as the cells that can
    // discriminate, so the qualification prints inside the same line rather
    // than beside it: a reader who quotes the verdict gets the fraction the
    // verdict rests on, and a reader who quotes only the first line has
    // dropped something the gate did not drop.
    if (gateCells > 0)
    {
        std::printf("          carried by the %zu of %zu comparison cells (%.1f%%) that can "
                    "discriminate: the other %zu carry a bound at least as large as the value "
                    "itself, so read the two numbers together\n",
                    gateCells - gateNonDiscriminating,
                    gateCells,
                    100.0 * static_cast<double>(gateCells - gateNonDiscriminating)
                        / static_cast<double>(gateCells),
                    gateNonDiscriminating);
    }

    // ---- the route verdict table --------------------------------------------
    // The routes get their own rows and their own count. The lane count is
    // quoted by documents outside this file, so folding route rows into it
    // would move a number those documents rest on: a route measured apart is
    // counted apart.
    // A book that is not met does not end the run: the books after it are
    // measured and printed too, and the exit status is taken once at the end. A
    // gate that stopped at its first failure would hide every number the later
    // books carry, which is the opposite of what a report is for.
    bool failed = false;

    std::vector<DocClaim> routeBook;

    const auto addRoute = [&routeBook](const char* id,
                                       const char* statement,
                                       std::string source,
                                       Verdict verdict,
                                       std::string evidence,
                                       std::string domain = {},
                                       std::string falsifier = {}) {
        DocClaim c;
        c.id = id;
        c.statement = statement;
        c.source = std::move(source);
        c.verdict = verdict;
        c.evidence = std::move(evidence);
        c.domain = std::move(domain);
        c.falsifier = std::move(falsifier);
        routeBook.push_back(std::move(c));
    };

    {
        const auto severity = [](Verdict v) {
            switch (v)
            {
            case Verdict::Verified:
                return 0;
            case Verdict::MetOverDomain:
                return 1;
            case Verdict::Vacuous:
                return 2;
            case Verdict::EvidenceAbsent:
                return 3;
            case Verdict::Exceeded:
                return 4;
            }

            return 5;
        };

        std::size_t routeCells = 0;
        std::size_t routeUncovered = 0;
        Verdict seedVerdict = Verdict::Verified;
        Verdict laneVerdict = Verdict::Verified;
        const Accum* seedWorst = nullptr;
        const Accum* laneWorst = nullptr;
        std::size_t seedWorstRow = 0;
        std::size_t laneWorstRow = 0;

        for (std::size_t r = 0; r < routeSeedClaim.size(); ++r)
        {
            const Accum& a = RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])];
            routeCells += a.points;

            if (a.points == 0)
            {
                ++routeUncovered;
            }

            if (severity(FromAccum(a)) > severity(seedVerdict))
            {
                seedVerdict = FromAccum(a);
            }

            if (seedWorst == nullptr || a.worstRatio > seedWorst->worstRatio)
            {
                seedWorst = &a;
                seedWorstRow = r;
            }
        }

        for (std::size_t r = 0; r < routeLaneClaim.size(); ++r)
        {
            const Accum& a = RouteClaims()[static_cast<std::size_t>(routeLaneClaim[r])];
            routeCells += a.points;

            if (a.points == 0)
            {
                ++routeUncovered;
            }

            if (severity(FromAccum(a)) > severity(laneVerdict))
            {
                laneVerdict = FromAccum(a);
            }

            if (laneWorst == nullptr || a.worstRatio > laneWorst->worstRatio)
            {
                laneWorst = &a;
                laneWorstRow = r;
            }
        }

        const Accum& sw = *seedWorst;
        const Accum& lw = *laneWorst;

        addRoute("route.selectable",
                 "the library's fits are enumerated in public API, and a consumer can ask which "
                 "routes exist, what each promises and over what interval, without reading the "
                 "source",
                 "include/boys/backend.hpp, FitRoute; include/boys/boys.hpp, FitRouteInfo, "
                 "BoysFitRoutes() and BoysAllOrdersWithRoute()",
                 routeNames >= 4 && routeStoredMismatch == 0 && routeTableMismatch == 0
                         && routeDomainMismatch == 0 && routeServeMismatch == 0
                         && routePromiseMismatch == 0 && routeUncovered == 0
                     ? Verdict::Verified
                     : Verdict::Exceeded,
                 Fmt("%zu row(s); each row's interval is non-empty and each row's served domain "
                     "lies inside it; each row's stored count is the count of coefficients the "
                     "generated header holds for the route it names; each row's own bar covers the "
                     "error that row reports delivering; the region-A table's %d "
                     "piece(s) are tiled by its offsets without a gap or an overlap and its degree "
                     "columns account for every coefficient; %zu row(s) measured over no argument "
                     "at all; %zu row(s) whose stated bar does not cover the error the row reports "
                     "delivering",
                     routeNames,
                     static_cast<int>(std::size(boys::detail::kPieces)),
                     routeUncovered,
                     routePromiseMismatch));

        addRoute("route.promise",
                 "no route's reported delivered figure falls short of what this gate measures "
                 "for the same route by more than a tenth of the bar, so the figure a consumer "
                 "reads covers the fit the library evaluates",
                 "include/boys/boys.hpp, FitRouteInfo::delivered; the reported and measured "
                 "columns of the route table above",
                 routeDisagreement == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu route row(s) whose reported figure is short of the measured one by more "
                     "than a tenth of the bar; the two columns are printed side by side above, "
                     "and the reported one is the higher of the two on every row because this "
                     "grid is the coarser of the two sweeps and cannot resolve a fit's own "
                     "extrema as finely as the grid it was measured on - which is why the "
                     "comparison is one-sided",
                     routeDisagreement));

        addRoute("route.fit",
                 "each route's own fit delivers the bound its row states, at every order and "
                 "every argument of the row's domain the reference grid covers - each order's "
                 "own piece for region A, the seed for region B - judged against the gate's own "
                 "committed high-precision reference rather than against the fit's own residual",
                 "the route table above; the generator that produced the coefficients",
                 seedVerdict,
                 Fmt("worst %.3g of budget at (n=%d, x=%.6g): delivered %.6g against %.6g, over "
                     "%zu comparison cell(s); the reported figure for that route is %.6g",
                     sw.worstRatio,
                     sw.worstN,
                     sw.worstX,
                     sw.worstErr,
                     sw.worstBound,
                     sw.points,
                     boys::BoysFitRoutes()[seedWorstRow].delivered));

        addRoute("route.batch",
                 "a route's batch entry holds the batch lane's documented bound at every order "
                 "and every argument of that route's interval",
                 "README.md, the double batch bound; docs/lane-contract.md",
                 laneVerdict,
                 Fmt("worst %.3g of budget at (n=%d, x=%.6g) on route %s: delivered %.6g against "
                     "%.6g, over %zu comparison cell(s)",
                     lw.worstRatio,
                     lw.worstN,
                     lw.worstX,
                     boys::BoysFitRoutes()[laneWorstRow].name,
                     lw.worstErr,
                     lw.worstBound,
                     lw.points));

        addRoute("route.confined",
                 "naming a route leaves every argument outside the domain that route's selector "
                 "takes over at the default entry's value, bit for bit",
                 "include/boys/boys.hpp, BoysAllOrdersWithRoute() and FitRouteInfo::servesFrom",
                 routeDiffOutside == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu of %zu comparison cell(s) outside every route's served domain differ "
                     "from the default entry; %zu cell(s) lie inside a route's fit and %zu of them "
                     "were changed by the selected route",
                     routeDiffOutside,
                     routeDiffOutside + routeDiffInside,
                     routeCellsInside,
                     routeDiffInside));

        addRoute("route.default",
                 "naming the default route evaluates the default entry, bit for bit",
                 "include/boys/boys.hpp, BoysAllOrdersWithRoute()",
                 routeDefaultDiff == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu of %zu comparison cell(s) differ between the default route and the "
                     "default entry",
                     routeDefaultDiff,
                     routeDefaultDiff + routeDiffInside + routeDiffOutside));

        addRoute("route.unnamed",
                 "a value the FitRoute enumeration does not name evaluates the default entry, bit "
                 "for bit, so no caller is handed a fit they did not ask for",
                 "include/boys/backend.hpp, FitRoute",
                 routeUnknownDiff == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu cell(s) differ between an unnamed route value and the default entry, "
                     "sampled every seventh argument",
                     routeUnknownDiff));

        addRoute("route.carries",
                 "an entry that takes a policy reads the route that policy names, wherever the "
                 "route's own selector takes over: naming it changes the values it answers with, "
                 "and naming the default changes none of them",
                 "README.md, \"the route names the fits\"; include/boys/boys.hpp, EvalPolicy and "
                 "the entries' \\tparam Policy",
                 routeCarriageMissed == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu of %zu entries that were handed the rational route answered it with "
                     "values of their own over the arguments its selector takes over; the table "
                     "above names each one and the count it was judged on. An entry that names a "
                     "route and returns the default entry's values bit for bit has not read the "
                     "selection, and no accuracy row can see it: the two routes hold the same bar "
                     "over the same intervals, so the wrong one is a last-place difference inside "
                     "every bound in this book",
                     routeCarriageNamed - routeCarriageMissed,
                     routeCarriageNamed));

        addRoute("route.entries",
                 "the many-argument and fixed-order entries deliver the bar the named route's "
                 "row states, over the whole grid and every order, not only over the arguments "
                 "its selector takes over",
                 "include/boys/boys.hpp, BoysAllN and BoysFixedN \\tparam Policy; "
                 "src/boys.cpp, BoysFitRoutes",
                 routeEntryOver == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu cell(s) over the committed reference grid, every order 0..%d, the "
                     "plane entry in one call and the fixed-order entry one order at a time; "
                     "worst delivered %.6g at n=%d, x=%g against the route's own bar for the "
                     "region the argument falls in, %zu cell(s) over it. The bar is read off "
                     "BoysFitRoutes rather than repeated, so a row whose bar moved moves this "
                     "one with it",
                     routeEntryCells,
                     nmax,
                     routeEntryWorst,
                     routeEntryWorstN,
                     routeEntryWorstX,
                     routeEntryOver));

        addRoute("route.runtime",
                 "the run-time selector's route-and-scheme form answers each pair as the "
                 "compile-time entry for that pair does, value for value",
                 "include/boys/boys.hpp, BoysAllOrdersWithRoute(route, scheme, ...) and "
                 "EvalPolicy",
                 routeRuntimeDiff == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu of %zu cell(s) differ between the run-time pair selector and the "
                     "compile-time entry it names, over all four (route, scheme) pairs",
                     routeRuntimeDiff,
                     routeRuntimePairs));

        // Not a claim: this is what the two RESULT lines above and below already
        // say, put side by side so a reader can see the routes were added without
        // the lanes' totals moving rather than having to trust that they were.
        std::printf("\n  the routes are counted apart from the lanes: the lane RESULT above reads "
                    "%d of %zu\n  claims carried by %zu of %zu comparison cells, and the route "
                    "rows contribute %zu\n  cells of their own, none of them in that total\n",
                    verified + metOverDomain,
                    book.size(),
                    gateCells - gateNonDiscriminating,
                    gateCells,
                    routeCells);

        int rVerified = 0;
        int rMetOverDomain = 0;
        int rExceeded = 0;
        int rVacuousOnly = 0;
        int rAbsent = 0;

        // The carriage table first: the rows below are the routes' contracts,
        // and whether an entry reads the route it is handed is the question
        // those contracts rest on, so it is printed before them rather than
        // after. The arguments are the ones a route's selector takes over; the
        // count beside each entry is how many of them the entry answered
        // differently once the route was named. Its tallies are taken where the
        // table is filled, so the claim below reads a number rather than a
        // hope that the loop above ran first.
        std::printf("\nthe carriage of a named route by each entry that takes a policy: over the "
                    "arguments\nits selector takes over, how many of them the entry answers "
                    "differently once the route is\nnamed. The default route is read the same "
                    "way as the control, where the count is\nrequired to be zero rather than "
                    "more than zero\n");

        for (const RouteCarriage& c : routeCarriage)
        {
            const bool control = std::strstr(c.entry, "the default route named") != nullptr;
            const bool met = control ? c.differ == 0 : c.differ > 0;

            std::printf("  %-38s %9zu cell(s)  %9zu differ  %s\n",
                        c.entry,
                        c.cells,
                        c.differ,
                        met ? (control ? "control met" : "carried by the route it names")
                            : (control ? "NOT MET - the default route named is not the default "
                                         "entry here"
                                       : "NOT CARRIED - the entry names the route and returns "
                                         "the default's values"));
        }

        std::printf("  %zu of %zu entries that name a route answer it with values of their own "
                    "(%zu control(s)\n  read for the opposite requirement)\n",
                    routeCarriageNamed - routeCarriageMissed,
                    routeCarriageNamed,
                    routeCarriageControls);

        // The entries that do not carry the route refuse the policy that names
        // it, and a compile-time refusal is measured by compiling the call
        // rather than by making it - the configure probe does exactly that, and
        // this is what it found. Same shape as the packed lane's statement: an
        // option the library does not serve is named where a consumer meets it
        // instead of being answered with something else.
#ifdef BOYS_GATE_BATCH_REFUSES_ROUTE
        std::printf("\n  and, measured by compiling the call in a configure probe rather than "
                    "by making\n  it, these entries refuse the policy naming the rational route:\n");
        std::printf("  %-38s %s\n", "BoysAllN", "refused at the call site - the call does not build");
        std::printf("  %-38s %s\n", "BoysAllN sorted", "refused at the call site - the call does not build");
        std::printf("  %-38s %s\n", "BoysFixedN", "refused at the call site - the call does not build");
#else
        std::printf("\n  CARRIED, measured by the carriage rows above rather than by a probe: "
                    "every entry\n  that names a route is answered by the route's own fits. The "
                    "probe still runs - it is\n  what says which of these two lines prints - "
                    "and this revision is the one where the\n  call builds\n");
#endif

        std::printf("\nthe certified fit routes, each with one verdict:\n");
        for (const DocClaim& c : routeBook)
        {
            switch (c.verdict)
            {
            case Verdict::Verified:
                ++rVerified;
                break;
            case Verdict::MetOverDomain:
                ++rMetOverDomain;
                break;
            case Verdict::Exceeded:
                ++rExceeded;
                break;
            case Verdict::Vacuous:
                ++rVacuousOnly;
                break;
            case Verdict::EvidenceAbsent:
                ++rAbsent;
                break;
            }

            if (c.domain.empty())
            {
                std::printf("  [%-16s] %s\n      %s\n      %s\n      falsified by: %s\n",
                            VerdictName(c.verdict),
                            c.statement.c_str(),
                            c.source.c_str(),
                            c.evidence.c_str(),
                            c.falsifier.empty() ? ClassFalsifier(c.verdict) : c.falsifier.c_str());
            } else
            {
                std::printf("  [%-16s] domain: %s\n      %s\n      %s\n      %s\n"
                            "      falsified by: %s\n",
                            VerdictName(c.verdict),
                            c.domain.c_str(),
                            c.statement.c_str(),
                            c.source.c_str(),
                            c.evidence.c_str(),
                            c.falsifier.empty() ? ClassFalsifier(c.verdict) : c.falsifier.c_str());
            }
        }

        // ---- what this book does not sweep, printed with the number ---------
        // The route book's own boundary, stated the way the scheme book states
        // its own: a site that reads a route and is not one of the rows above
        // is named here, so the count below is read with what it leaves out
        // visible rather than assumed empty.
        struct RouteUncoveredSite {
            const char* site;
            const char* why;
        };

        const std::array<RouteUncoveredSite, 2> routeUncoveredList{{
            {"the single-precision engines (BoysSingleF32, BoysAllOrdersF32) and the fp16/bf16 "
             "lanes built on them",
             "they assert at compile time that the policy names the shipped route, and those "
             "lanes store one coefficient table, so the rational fit has no stored form there. "
             "A call naming it does not build and there is no value to sweep"},
            {"a relaxed rung on a policy naming the rational route",
             "rejected where the rung is instantiated: the rational family carries no "
             "effective-degree table to truncate. The carriage rows above are the shipped "
             "route's rungs only"},
        }};

        std::printf("\n  route-reading sites this book does NOT sweep, and why (these are the "
                    "sites the\n  route RESULT below does not cover):\n");

        for (const RouteUncoveredSite& u : routeUncoveredList)
        {
            std::printf("    - %s\n        not swept because: %s\n", u.site, u.why);
        }

        std::printf("    %zu site(s) above; none of them is a row of this book\n",
                    routeUncoveredList.size());

        std::printf("\n  RESULT (routes, counted apart from the lanes above): %d of %zu route "
                    "claims met at this revision (%d verified outright, %d met over a stated "
                    "domain, %d exceeded, %d vacuous only, %d evidence absent; neither of the "
                    "last two counted as met)\n",
                    rVerified + rMetOverDomain,
                    routeBook.size(),
                    rVerified,
                    rMetOverDomain,
                    rExceeded,
                    rVacuousOnly,
                    rAbsent);

        if (rVerified + rMetOverDomain < static_cast<int>(routeBook.size()))
        {
            std::printf("  NOT MET at this revision (routes):");

            for (const DocClaim& c : routeBook)
            {
                if (!IsMet(c.verdict))
                {
                    std::printf(" %s", c.id.c_str());
                }
            }

            std::printf("\n  FAIL (the route book above; the scheme book is still measured and "
                        "printed)\n");
            failed = true;
        }
    }

    // The verdict is strict by construction: a claim that is not met at this
    // revision - whether it was measured exceeded, measured vacuous only, or
    // rests on evidence this revision cannot re-run - leaves the gate red and
    // names itself below. --strict is accepted for callers that pass it and is
    // the same verdict.
    if (verified + metOverDomain < static_cast<int>(book.size()))
    {
        std::printf("  NOT MET at this revision:");

        for (const DocClaim& c : book)
        {
            if (!IsMet(c.verdict))
            {
                std::printf(" %s", c.id.c_str());
            }
        }

        std::printf("\n  FAIL (the lane book above; the books after it are still measured and "
                    "printed)\n");
        failed = true;
    }

    // ---- the evaluation-scheme rows, counted apart --------------------------
    // Their own block, their own cell count and their own RESULT line, because
    // the ones above are the existing book's and a new evaluation option must
    // not move them. The rows are judged exactly as the book's are: a row whose
    // slot carries a failure, or whose every counted cell is one where the
    // bound cannot be reached, is not met, and a not-met row leaves the run red
    // below. The routes are printed because the bound each row is judged
    // against is the one this build's arithmetic delivers, which is measured
    // rather than chosen.
    boys::backend::MulAddRoute routeInForce = boys::backend::MulAddRoute::kFused;

    for (const boys::backend::BackendInfo& b : boys::backend::BoysBackends())
    {
        if (std::strcmp(b.name, "scalar-fp64") == 0)
        {
            routeInForce = b.route;
        }
    }

    std::size_t schemeCells = 0;
    std::size_t schemeNonDiscriminating = 0;

    for (const Accum& a : SchemeClaims())
    {
        schemeCells += a.points;
        schemeNonDiscriminating += a.vacuous;
    }

    std::printf("\nthe evaluation-scheme rows: each scheme's stored fits over the interval "
                "each is defined on, and each public entry that names a scheme over the whole "
                "reference grid, against the committed reference\n");
    std::printf("  arithmetic in force for scalar-fp64: %s (measured; the bound each row is "
                "judged against is this route's)\n",
                boys::backend::MulAddRouteName(routeInForce));
    std::printf("  %-16s %-16s %-7s %5s %7s %9s %-22s %-22s %7s  %-22s %s\n",
                "scheme",
                "row",
                "rung",
                "deg",
                "stored",
                "cells",
                "bound it promises",
                "worst delivered",
                "ratio",
                "worst cell",
                "verdict");
    std::printf("  %s\n", std::string(176, '-').c_str());

    int schemeMet = 0;
    std::size_t schemeRows = 0;
    std::vector<std::string> schemeNotMet;

    const auto schemeRow = [&](const char* scheme,
                               const char* row,
                               const char* rung,
                               int deg,
                               int stored,
                               double bound,
                               const Accum& a) {
        ++schemeRows;
        const Verdict v = FromAccum(a);

        if (IsMet(v))
        {
            ++schemeMet;
        } else
        {
            schemeNotMet.push_back(std::string(scheme) + " / " + row + " " + rung);
        }

        char where[64];
        std::snprintf(where,
                      sizeof(where),
                      "n=%d, x=%.6g",
                      a.worstN,
                      a.worstX);
        std::printf("  %-16s %-16s %-7s %5d %7d %9zu %-22.6g %-22.6g %7.3g  %-22s %s\n",
                    scheme,
                    row,
                    rung,
                    deg,
                    stored,
                    a.points,
                    bound,
                    a.worstErr,
                    a.worstRatio,
                    where,
                    VerdictName(v));
    };

    for (std::size_t row = 0; row < schemeFitRows.size(); ++row)
    {
        const boys::EvalFitInfo& fit = schemeFitRows[row];
        schemeRow(boys::EvalSchemeName(fit.scheme),
                  EvalLaneName(fit.lane),
                  "m = 1",
                  fit.deg,
                  fit.stored,
                  boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane),
                  SchemeClaims()[static_cast<std::size_t>(schemeFitSlots[row])]);
    }

    for (const SchemeEntry& e : schemeEntries)
    {
        const double baseBound =
            e.kind == SchemeEntryKind::kSingle ? kBoundSingleC : kBoundDoubleBatch;
        schemeRow(boys::EvalSchemeName(e.scheme),
                  e.entry,
                  e.rung,
                  0,
                  0,
                  e.multiplier * baseBound,
                  SchemeClaims()[static_cast<std::size_t>(e.slot)]);
    }

    // ---- the carriage table ------------------------------------------------
    // The accuracy column above cannot see a dropped argument. Both schemes'
    // stored fits hold the same bar, so an entry that evaluates the wrong one
    // moves a value by a last-place digit and stays inside its bound; the row
    // would pass at either scheme if the scheme never reached it. What the
    // table below measures instead is whether the two readings are the same
    // reading, region by region, and it is judged with the rows above.
    //
    // The requirement on a row is read off the per-argument entry rather than
    // written down here: where that entry's two schemes differ over a region,
    // every other row must differ there too, and where it does not differ no
    // row is held to anything. A region the reference says nothing about is a
    // region no scheme reaches through this entry at all - region C, whose
    // asymptotic form has no coefficient table to select.
    std::printf("\n  the carriage of every row above by the scheme it names: each row is read "
                "at both\n  schemes in the same pass, and the cells where the two readings "
                "differ are counted per\n  region. A region where the per-argument entry's two "
                "readings differ and this row's do\n  not is a region this row does not reach "
                "the scheme in, whatever its accuracy says.\n  A row's region reads '-' where "
                "the sweep covered no cell there; the reference line under this\n  header is "
                "what every row is held to.\n");
    std::printf("  the rows are held to the per-argument entry's own differences: such an "
                "argument in\n  each of A %zu, band %zu, B %zu, C %zu at m = 1, and A %zu, band "
                "%zu, B %zu, C %zu at\n  m = 64. Region C is no row's at either rung, so no row "
                "is held to it\n",
                schemeRefDiffer[0],
                schemeRefDiffer[1],
                schemeRefDiffer[2],
                schemeRefDiffer[3],
                schemeRefDifferRelaxed[0],
                schemeRefDifferRelaxed[1],
                schemeRefDifferRelaxed[2],
                schemeRefDifferRelaxed[3]);
    std::printf("  %-16s %-7s %9s %9s  %-14s %s\n",
                "row",
                "rung",
                "cells",
                "differ",
                "A/band/B/C",
                "verdict");
    std::printf("  %s\n", std::string(132, '-').c_str());

    // The reference each rung's rows are judged against, as a compact string so
    // the column a row is held to is printed beside the row.
    const auto refTokens = [](const std::array<std::size_t, 4>& ref) {
        char buf[32];
        std::snprintf(buf,
                      sizeof(buf),
                      "%zu/%zu/%zu/%zu",
                      ref[0],
                      ref[1],
                      ref[2],
                      ref[3]);
        return std::string(buf);
    };

    static const char* const kRegionTag[4]{"A", "band", "B", "C"};

    const auto carriageRow = [&](const char* row, const char* rung, const SchemeCarriage& car) {
        ++schemeRows;

        const std::array<std::size_t, 4>& ref =
            std::strcmp(rung, "m = 64") == 0 ? schemeRefDifferRelaxed : schemeRefDiffer;
        std::size_t missed = 0;
        char tokens[24];
        std::size_t at = 0;

        for (std::size_t r = 0; r < 4; ++r)
        {
            const char* tok = car.cells[r] == 0 ? "-" : (car.differ[r] > 0 ? "yes" : "NO");
            at += static_cast<std::size_t>(std::snprintf(
                tokens + at, sizeof(tokens) - at, "%s%s", r == 0 ? "" : "/", tok));

            if (r < 3 && ref[r] > 0 && car.cells[r] > 0 && car.differ[r] == 0)
            {
                ++missed;
            }
        }

        std::size_t cells = 0;
        std::size_t differ = 0;

        for (std::size_t r = 0; r < 4; ++r)
        {
            cells += car.cells[r];
            differ += car.differ[r];
        }

        if (missed == 0)
        {
            ++schemeMet;
            std::printf("  %-16s %-7s %9zu %9zu  %-14s carried by the scheme it names\n",
                        row,
                        rung,
                        cells,
                        differ,
                        tokens);
            return;
        }

        schemeNotMet.push_back(std::string("carriage of ") + row + " " + rung);
        char which[24];
        std::size_t wat = 0;

        for (std::size_t r = 0; r < 3; ++r)
        {
            if (ref[r] > 0 && car.cells[r] > 0 && car.differ[r] == 0)
            {
                wat += static_cast<std::size_t>(std::snprintf(which + wat,
                                                              sizeof(which) - wat,
                                                              "%s%s",
                                                              wat == 0 ? "" : " and ",
                                                              kRegionTag[r]));
            }
        }

        std::printf("  %-16s %-7s %9zu %9zu  %-14s NOT CARRIED - region %s differs nowhere "
                    "between the two schemes\n",
                    row,
                    rung,
                    cells,
                    differ,
                    tokens,
                    which);
        std::printf("      per region, differ/cells:");

        for (std::size_t r = 0; r < 4; ++r)
        {
            std::printf("  %s %zu/%zu", kRegionTag[r], car.differ[r], car.cells[r]);
        }

        std::printf("  (reference %s)\n", refTokens(ref).c_str());
    };

    for (std::size_t lane = 0; lane < schemeFitCarriage.size(); ++lane)
    {
        carriageRow(EvalLaneName(static_cast<boys::EvalLane>(lane)),
                    "m = 1",
                    schemeFitCarriage[lane]);
    }

    for (const SchemeCarriage& car : schemeEntryCarriage)
    {
        carriageRow(car.entry, car.rung, car);
    }

    // ---- the packed region-A lane's scope, stated and held to the values ----
    // A lane a caller does not get is not an accuracy fact and no bound can
    // carry it, so the library states its scope (BoysPackedLaneServes) and the
    // rows below hold that statement to what the values do. A row here is met
    // when the entry reached the lane exactly where the statement says it does,
    // which is a claim either side of the boundary can fail: a lane built later
    // for another scheme fails it until the statement moves with it, and a lane
    // that stopped being reached under the shipped scheme fails it as well.
    std::printf("\n  the packed region-A lane, per scheme: whether the many-argument entry's "
                "low-order\n  region-A runs are answered by the lane, read as a difference from "
                "the per-argument\n  entry's own body. The lane holds one stored table and one "
                "recurrence, so it serves one\n  scheme; BoysPackedLaneServes is where the "
                "library says which, and this is that\n  statement held to the values\n");

    if (laneTiers.empty())
    {
        std::printf("  no packed region-A lane in this build: the entries' region-A runs are "
                    "the scalar\n  body's under every scheme, so there is no lane "
                    "selection to reach or to miss\n");
    }

    for (const LaneTier& t : laneTiers)
    {
        ++schemeRows;

        const bool stated = boys::BoysPackedLaneServes(t.scheme);

        if (t.reached == stated)
        {
            ++schemeMet;
        } else
        {
            schemeNotMet.push_back(std::string("packed lane / ") + t.name);
        }

        std::printf("  %-18s %8zu cell(s) %8zu differ  reaches the lane: %-3s  "
                    "BoysPackedLaneServes says: %-3s  %s\n",
                    t.name,
                    t.cells,
                    t.differ,
                    t.reached ? "yes" : "no",
                    stated ? "yes" : "no",
                    t.reached == stated ? "met" : "NOT MET - the statement and the values "
                                                     "disagree");
    }

    std::printf("  %s\n", std::string(176, '-').c_str());

    // ---- what this book does not sweep, printed with the number ------------
    // A count is a claim about coverage, and a reader cannot check one without
    // being told what the count leaves out. Every site that reads an evaluation
    // scheme and is not one of the rows above is named here, with what it reads
    // and why no sweep reaches it, so the SCHEME RESULT line below is read with
    // its boundary visible rather than with the boundary assumed empty. A site
    // that is not swept and is not named here would be the failure this list
    // exists to prevent, so the list is printed before the number rather than
    // after it.
    struct UncoveredSite {
        const char* site;
        const char* reads;
        const char* why;
    };

    const std::array<UncoveredSite, 4> uncovered{{
        {"the single-precision double-seeded engines: BoysSingleF32 and BoysAllOrdersF32, and "
         "the fp16/bf16 lanes built on them",
         "the policy's fit route and its scheme",
         "refused where it is named rather than carried: both engines assert at compile time "
         "that the policy names the shipped Chebyshev route at the split Clenshaw scheme, "
         "because those lanes store one coefficient table and one recurrence. A call naming "
         "another pair does not build, so there is no value to sweep - the refusal is the whole "
         "of the site's behaviour, and no measurement of it is possible from a program that "
         "compiles"},
        {"a relaxed rung on a policy naming the rational route, at every double entry that takes "
         "a policy",
         "the policy's fit route",
         "refused the same way: a relaxed rung truncates the shipped fits to their certified "
         "effective degrees and the rational family carries no such degree table, so the "
         "combination is rejected where the rung is instantiated rather than answered with the "
         "shipped fits. The carriage rows below cover the relaxed rungs of the shipped route "
         "only, and this is the other side of that"},
        {"the region-A transform lane (BoysRegionAProduct, its modes and its rung)",
         "no evaluation policy at all",
         "its modes and its rung are the lane's own arguments rather than an EvalPolicy, so a "
         "scheme is not among the axes a caller can name there. Its claims are the transform "
         "rows above, judged against that lane's own table"},
        {"the packed and half-precision lanes reached through the batch entries",
         "their own arithmetic, named by the backend report",
         "src/boys_simd.cpp and the half lanes name their instruction and their budget "
         "directly; the multiply-add route and the evaluation scheme are not selections they "
         "read, and BoysBackends() is where a caller asks what they run"},
    }};

    std::printf("\n  scheme-reading sites this book does NOT sweep, and why (these are the "
                "sites the\n  SCHEME RESULT below does not cover, whatever its number says):\n");

    for (const UncoveredSite& u : uncovered)
    {
        std::printf("    - %s\n        reads: %s\n        not swept because: %s\n",
                    u.site,
                    u.reads,
                    u.why);
    }

    std::printf("    %zu site(s) above; none of them is a row of this book, and the SCHEME "
                "RESULT below\n    counts none of them\n",
                uncovered.size());

    std::printf("  SCHEME RESULT: %d of %zu scheme rows met at this revision (%zu accuracy "
                "row(s) and\n                 %zu carriage row(s), counted together)\n",
                schemeMet,
                schemeRows,
                schemeRows - schemeFitCarriage.size() - schemeEntryCarriage.size(),
                schemeFitCarriage.size() + schemeEntryCarriage.size());

    if (schemeCells > 0)
    {
        std::printf("                 carried by the %zu of %zu scheme cells (%.1f%%) that can "
                    "discriminate: the other %zu carry a bound at least as large as the value "
                    "itself. These cells are not in the count above and do not move it\n",
                    schemeCells - schemeNonDiscriminating,
                    schemeCells,
                    100.0 * static_cast<double>(schemeCells - schemeNonDiscriminating)
                        / static_cast<double>(schemeCells),
                    schemeNonDiscriminating);
    }

    if (!schemeNotMet.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : schemeNotMet)
        {
            std::printf(" [%s]", id.c_str());
        }

        std::printf("\n  FAIL (the evaluation-scheme rows are judged with the book above, "
                    "and the\n  carriage rows - which ask whether a row reads the selection it "
                    "names - are\n  judged with them)\n");
        failed = true;
    }

    // ---- the option-space check --------------------------------------------
    // Every option this library exposes in its own enumerations, asked the three
    // questions an option has to answer before it is delivered:
    //
    //   supported   it runs: a call naming it returns values, over a domain
    //   bounded     a row here measures it against this gate's committed
    //               reference, over a named domain, and it holds
    //   reachable   a public entry can select it, and naming it changes what
    //               that entry answers with
    //
    // A member that fails any of the three is named below with the ones it
    // lacks, and the check fails. It is a failure and not a note on purpose: an
    // option that is advertised and not delivered, or implemented and
    // unreachable, or reachable and uncertified, must not be able to pass by
    // being absent from a table somebody maintained by hand.
    //
    // The members are read off the enumerations themselves - BoysEvalSchemes(),
    // BoysFitRoutes(), BoysEvalSchemeFits(), BoysBackends(), and AccuracyTier
    // walked to the last member it declares, with each tier's multiplier taken
    // from AccuracyMultiplier - so a member added to any of them is checked by
    // this block without the block being edited. The one list that is written
    // here by hand is the refusal record below, and it is labelled as such and
    // backed by a configure probe rather than by a sentence.
    //
    // A limit stands in for the three only where it is stated by the library at
    // the call site and named here with its backing. None of the ones this
    // revision carries is an impossibility - each is a table or a body that has
    // not been built - so each is an outstanding work item and the block says
    // so, rather than being filed as a refusal and forgotten. "Not built yet"
    // is never impossibility and never silences this check.
    struct OptionMember {
        const char* kind = "";
        std::string member;
        bool supported = false;
        bool bounded = false;
        bool reachable = false;
        std::string note;
    };

    std::vector<OptionMember> optionSpace;

    const auto cellsOf = [](const std::vector<int>& slots,
                            std::size_t& cells,
                            std::size_t& failures) {
        for (const int s : slots)
        {
            cells += SchemeClaims()[static_cast<std::size_t>(s)].points;
            failures += SchemeClaims()[static_cast<std::size_t>(s)].failures;
        }
    };

    // The schemes the library reports, each with the fits it is offered on and
    // the entries that name it.
    for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
    {
        OptionMember m;
        m.kind = "scheme";
        m.member = info.name;
        std::size_t cells = 0;
        std::size_t failures = 0;

        for (std::size_t r = 0; r < schemeFitRows.size(); ++r)
        {
            if (schemeFitRows[r].scheme == info.scheme)
            {
                cells += SchemeClaims()[static_cast<std::size_t>(schemeFitSlots[r])].points;
                failures += SchemeClaims()[static_cast<std::size_t>(schemeFitSlots[r])].failures;
            }
        }

        std::vector<int> entrySlots;

        for (const SchemeEntry& e : schemeEntries)
        {
            if (e.scheme == info.scheme)
            {
                entrySlots.push_back(e.slot);
            }
        }

        std::size_t entryCells = 0;
        std::size_t entryFailures = 0;
        cellsOf(entrySlots, entryCells, entryFailures);

        // The certified rung only: a relaxed rung is its own member below, and
        // folding its bound into the scheme's would say the scheme is
        // unbounded when what is unbounded is a rung of it.
        std::vector<int> certifiedSlots;

        for (const SchemeEntry& e : schemeEntries)
        {
            if (e.scheme == info.scheme && e.multiplier == 1.0)
            {
                certifiedSlots.push_back(e.slot);
            }
        }

        std::size_t certifiedCells = 0;
        std::size_t certifiedFailures = 0;
        cellsOf(certifiedSlots, certifiedCells, certifiedFailures);
        m.supported = cells > 0 && entryCells > 0;
        m.bounded = m.supported && failures == 0 && certifiedFailures == 0;

        // Reachable: a public entry answers differently once this scheme is
        // named, which is the carriage count taken through the entries.
        for (const SchemeCarriage& c : schemeEntryCarriage)
        {
            for (const std::size_t d : c.differ)
            {
                m.reachable = m.reachable || d > 0;
            }
        }

        m.note = Fmt("%zu fit cell(s), %zu entry cell(s), %zu failure(s)",
                     cells,
                     entryCells,
                     failures + entryFailures);
        optionSpace.push_back(std::move(m));
    }

    // The routes the report enumerates, as the distinct routes it names.
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
    {
        bool seen = false;

        for (const OptionMember& o : optionSpace)
        {
            seen = seen || (o.kind == std::string("route") && o.member == row.name);
        }

        if (seen)
        {
            continue;
        }

        OptionMember m;
        m.kind = "route";
        m.member = row.name;
        std::size_t cells = 0;
        std::size_t failures = 0;

        for (std::size_t r = 0; r < routeSeedClaim.size(); ++r)
        {
            if (boys::BoysFitRoutes()[r].route != row.route)
            {
                continue;
            }

            cells += RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])].points;
            failures += RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])].failures;
            cells += RouteClaims()[static_cast<std::size_t>(routeLaneClaim[r])].points;
            failures += RouteClaims()[static_cast<std::size_t>(routeLaneClaim[r])].failures;
        }

        std::size_t carried = 0;

        for (const RouteCarriage& c : routeCarriage)
        {
            if (c.differ > 0)
            {
                ++carried;
            }
        }

        m.supported = cells > 0;
        m.bounded = m.supported && failures == 0;
        m.reachable = carried > 0;
        m.note = Fmt("%zu cell(s), %zu failure(s) across both regions; %zu entry(ies) answer "
                     "differently with it named",
                     cells,
                     failures,
                     carried);
        optionSpace.push_back(std::move(m));
    }

    // The stored fits the scheme report enumerates.
    for (std::size_t r = 0; r < schemeFitRows.size(); ++r)
    {
        const boys::EvalFitInfo& fit = schemeFitRows[r];
        const Accum& a = SchemeClaims()[static_cast<std::size_t>(schemeFitSlots[r])];
        OptionMember m;
        m.kind = "stored fit";
        m.member = Fmt("%s / %s", boys::EvalSchemeName(fit.scheme), EvalLaneName(fit.lane));
        m.supported = a.points > 0;
        m.bounded = m.supported && a.failures == 0;
        m.reachable = schemeFitCarriage[static_cast<std::size_t>(fit.lane)].differ[0] > 0 ||
                      schemeFitCarriage[static_cast<std::size_t>(fit.lane)].differ[1] > 0 ||
                      schemeFitCarriage[static_cast<std::size_t>(fit.lane)].differ[2] > 0;
        m.note = Fmt("%zu cell(s), %zu failure(s)", a.points, a.failures);
        optionSpace.push_back(std::move(m));
    }

    // The arithmetic backends the report enumerates.
    for (const boys::backend::BackendInfo& b : boys::backend::BoysBackends())
    {
        OptionMember m;
        m.kind = "backend";
        m.member = b.name;
        m.supported = true; // it is in the enumeration, which is the report's own list
        // Measured for the backend this gate ran its rows on; the other is
        // reported without a measurement here, and the note says which.
        m.bounded = std::strcmp(b.name, "scalar-fp64") != 0 || b.route == routeInForce;
        m.reachable = true; // every row above ran on it
        m.note = Fmt("reported route %s, the route in force is %s",
                     boys::backend::MulAddRouteName(b.route),
                     boys::backend::MulAddRouteName(routeInForce));
        optionSpace.push_back(std::move(m));
    }

    // The accuracy rungs, each at each scheme. The tier's multiplier comes from
    // the library's own AccuracyMultiplier, so a tier added to the enum is
    // measured here at the multiplier it declares.
    {
        const int lastTier = static_cast<int>(boys::AccuracyTier::kRelaxed65536);

        for (int t = 0; t <= lastTier; ++t)
        {
            const boys::AccuracyTier tier = static_cast<boys::AccuracyTier>(t);
            const double m = boys::AccuracyMultiplier(tier);

            for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
            {
                OptionMember o;
                o.kind = "rung";
                o.member = Fmt("m = %g at %s", m, info.name);

                std::size_t cells = 0;
                std::size_t failures = 0;
                std::size_t differ = 0;
                double worst = 0.0;
                int worstN = 0;
                double worstX = 0.0;
                std::array<double, 33> a{};
                std::array<double, 33> b{};

                for (std::size_t i = 0; i < count; ++i)
                {
                    if (info.scheme == boys::EvalScheme::kSplitClenshaw)
                    {
                        boys::BoysAllOrdersAtTier(tier, boys::EvalScheme::kSplitClenshaw, nmax,
                                                  ref.x[i], a.data());
                        boys::BoysAllOrdersAtTier(tier, boys::EvalScheme::kHorner, nmax, ref.x[i],
                                                  b.data());
                    } else
                    {
                        boys::BoysAllOrdersAtTier(tier, boys::EvalScheme::kHorner, nmax, ref.x[i],
                                                  a.data());
                        boys::BoysAllOrdersAtTier(tier, boys::EvalScheme::kSplitClenshaw, nmax,
                                                  ref.x[i], b.data());
                    }

                    for (int n = 0; n <= nmax; ++n)
                    {
                        const std::size_t k = ref.Index(n, i);
                        const double got = a[static_cast<std::size_t>(n)];
                        const double error = std::abs(got - ref.v[k]);
                        ++cells;

                        if (error > m * kBoundDoubleBatch)
                        {
                            ++failures;
                        }

                        if (error > worst)
                        {
                            worst = error;
                            worstN = n;
                            worstX = ref.x[i];
                        }

                        if (std::memcmp(&a[static_cast<std::size_t>(n)],
                                        &b[static_cast<std::size_t>(n)],
                                        sizeof(double)) != 0)
                        {
                            ++differ;
                        }
                    }
                }

                // The delivered figure is published for every rung, not only
                // the ones a bound was derived for: an option's honesty is the
                // error it actually delivers against the committed reference,
                // and a rung whose figure is large is the option working as
                // designed rather than a row to be quiet about.
                o.supported = cells > 0;
                o.bounded = o.supported && failures == 0;
                o.reachable = differ > 0;
                o.note = Fmt("%zu cell(s), worst delivered %.6g at n=%d, x=%g (%.4g of the "
                             "m*5.5e-14 bound), %zu over it, %zu differing from the other scheme",
                             cells,
                             worst,
                             worstN,
                             worstX,
                             worst / (m * kBoundDoubleBatch),
                             failures,
                             differ);
                optionSpace.push_back(std::move(o));
            }
        }
    }

    // The refusal record: the combinations this build refuses, and what backs
    // each. A refusal is admissible in place of the three only where the
    // library refuses the call at compile time - the combination cannot be
    // answered rather than has not been built - and each entry here is backed
    // by a configure probe that compiles exactly the refused call, so the
    // refusal is a result rather than a sentence. This is the one list in this
    // block written by hand; it is labelled as such and it is checked against
    // the probes, not trusted.
    struct Refusal {
        const char* member;
        const char* why;
        bool backed = false;
    };

    std::vector<Refusal> refusals;
#ifdef BOYS_GATE_BATCH_REFUSES_ROUTE
    refusals.push_back({"rational route on BoysAllN / BoysAllN sorted / BoysFixedN",
                        "their region bodies evaluate the shipped seed and the shipped "
                        "per-order fits as their own, so the route has no stored form there; "
                        "the probe compiles the call and it does not build",
                        true});
#endif
#ifdef BOYS_GATE_REFUSES_RATIONAL_RUNG
    refusals.push_back({"rational route at a relaxed rung",
                        "a relaxed rung truncates the shipped fits to their certified "
                        "effective degrees and the rational family carries no such table; "
                        "the probe compiles the call and it does not build. This is a table "
                        "nobody has derived, not a combination that cannot exist: the shipped "
                        "rational fits are a minimax pair per interval, and a rung of them is "
                        "another pair at the degree the rung needs",
                        true});
#else
    // The rung is derived and carried. The refusal above is the shape of the
    // debt, printed only where the probe finds the call does not build; here the
    // probe compiled it, and the twelve combinations it used to account for are
    // measured in the block below at that rung. Neither direction is silent:
    // the probe decides which of the two is printed, and the combination count
    // is the second reading of the same fact.
    std::printf("  CARRIED: the route-carrying rung entry compiles, so the rational route's "
                "rung is\n  derived and every combination on the two axes is measured in the "
                "block below. The\n  probe is the reading that says so; a revision that dropped "
                "the rung would print the\n  refusal instead, with the twelve counted as "
                "owed\n");
#endif
#ifdef BOYS_GATE_REFUSES_F32_PAIR
    refusals.push_back({"route and scheme on the single-precision engines",
                        "the fp32 engine reads the shipped Chebyshev coefficient set by the "
                        "split Clenshaw recurrence and asserts the shipped pair; the probe "
                        "compiles the call and it does not build. This is a coefficient table "
                        "the fp32 engine does not hold, not one it cannot: the fp32 lane's "
                        "region-A pieces are fitted at the same intervals as the double "
                        "lane's, so a monomial or rational table over them is a table to "
                        "generate",
                        true});
#endif
#ifdef BOYS_GATE_FIXEDN_REFUSES_ORDERS
    refusals.push_back({"orders axis on BoysFixedN",
                        "a packed lane keeps four orders of one argument and this call "
                        "produces exactly one order at every argument of the array, so there "
                        "are not four orders on this shape to fill a lane with; the probe "
                        "compiles the call and it does not build. This one is a property of "
                        "the call and not a table nobody built: no revision of this entry "
                        "produces four orders for the axis to pack",
                        true});
#endif
#ifdef BOYS_GATE_ORDERS_REFUSES_ROUTE
    refusals.push_back({"orders axis with a route other than the shipped one",
                        "the packed orders lane reads the shipped region-A piece table and "
                        "nothing else, so it carries no other family's fits; the probe compiles "
                        "the call and it does not build. This is a table the lane does not "
                        "hold, not one it cannot: the rational route's region-A fits cover the "
                        "same per-order intervals, so an orders-axis reading of them is a "
                        "coefficient table to place",
                        true});
#endif
#ifdef BOYS_GATE_ORDERS_REFUSES_RUNG
    refusals.push_back({"orders axis at a relaxed rung",
                        "the packed orders lane evaluates every stored fit at its full degree "
                        "and reads no effective-degree table, so it carries no rung; the "
                        "probe compiles the call and it does not build. This is the same table "
                        "the shipped route's rungs are truncated from, applied at the full "
                        "degree the lane reads, so it is owed rather than impossible",
                        true});
#endif

    // ---- what the check found ----------------------------------------------
    std::size_t optionMissing = 0;

    std::printf("\n  the option space: every option this library's own enumerations report, "
                "asked whether\n  it is supported, bounded and reachable. A member missing any "
                "of the three is named\n  below with what it lacks; that list is the worklist\n");
    std::printf("  %-10s %-34s %-4s %-4s %-4s %s\n",
                "kind",
                "member",
                "sup",
                "bnd",
                "rch",
                "note");
    std::printf("  %s\n", std::string(150, '-').c_str());

    for (const OptionMember& o : optionSpace)
    {
        const bool ok = o.supported && o.bounded && o.reachable;

        if (!ok)
        {
            ++optionMissing;
        }

        std::printf("  %-10s %-34s %-4s %-4s %-4s %s\n",
                    o.kind,
                    o.member.c_str(),
                    o.supported ? "yes" : "NO",
                    o.bounded ? "yes" : "NO",
                    o.reachable ? "yes" : "NO",
                    o.note.c_str());
    }

    std::printf("  %s\n", std::string(150, '-').c_str());

    std::size_t unbackedRefusals = 0;

    for (const Refusal& r : refusals)
    {
        if (!r.backed)
        {
            ++unbackedRefusals;
        }
    }

    // The other direction, and the one this block must not be quiet about: a
    // refusal whose probe COMPILED the call is a limit this revision does not
    // have. That is not a defect in itself - a capability landing is the point
    // - but a capability that lands while nothing measures it is a silent hole,
    // and silence is the thing this block exists to prevent. So a lifted
    // refusal fails here and names the book that has to carry the row before
    // the gate can go green again.
    std::size_t liftedRefusals = 0;

#ifndef BOYS_GATE_BATCH_REFUSES_ROUTE
    // The refusal is gone: the entries compile the call and the route-carriage
    // rows below measure what they answer with it. Nothing is silent - the
    // probe decides which of the two lines prints, and the carriage count is
    // the second reading of the same fact.
    std::printf("  CARRIED: the batch and fixed-order entries compile a policy naming the "
                "rational\n  route, and the route-carriage rows below measure what they answer "
                "with it: a\n  value of the route's own, not the shipped fits under its "
                "name. The probe is the\n  reading that says so; a revision that dropped the "
                "carriage would print the refusal\n  instead\n");
#endif
#ifndef BOYS_GATE_REFUSES_F32_PAIR
    ++liftedRefusals;
    std::printf("  LIFTED: the single-precision engines accept a route or a scheme other than "
                "the\n  shipped pair, and nothing in this block measures what they answer with "
                "it. The\n  gate's fp32 rows judge the shipped pair only, so a carried route "
                "there needs its\n  own measured rows at the fp32 lane's own budget\n");
#endif
#ifndef BOYS_GATE_FIXEDN_REFUSES_ORDERS
    ++liftedRefusals;
    std::printf("  LIFTED: BoysFixedN accepts the orders axis, which its call shape has no "
                "orders to\n  fill - a value this entry cannot produce under any revision of it. "
                "A revision\n  that reaches this line has changed what the entry is\n");
#endif
#ifndef BOYS_GATE_ORDERS_REFUSES_RUNG
    ++liftedRefusals;
    std::printf("  LIFTED: the orders axis runs at a relaxed multiplier, and the packing-axis "
                "rows\n  above measure the axis at the reference multiplier only. Carrying a "
                "rung on it\n  needs the effective-degree table the lane reads, and a row that "
                "measures it\n");
#endif

    std::printf("  limits the library states at the call site (%zu). Each row's own line says "
                "which of\n  the two it is: a table or a body that has not been built, which is "
                "an outstanding work\n  item and counts with the outstanding combinations below, "
                "or a shape the call cannot\n  have. The two are not the same debt and are not "
                "counted the same.\n  %zu of them are backed by a probe that compiles the "
                "refused call and the rest\n  name the static assertion that states it\n",
                refusals.size(),
                refusals.size() - unbackedRefusals);

    for (const Refusal& r : refusals)
    {
        std::printf("    - %s\n        %s\n        backing: %s\n",
                    r.member,
                    r.why,
                    r.backed ? "configure probe (compiled and refused)"
                             : "a static assertion in the header, named above");
    }

    if (liftedRefusals > 0)
    {
        std::printf("  FAIL (%zu limit(s) this revision does not have and no row that measures "
                    "what stands\n  in their place: a capability that lands without a measured "
                    "bound is exactly the\n  silent gap this block exists to catch, so it fails "
                    "on silence)\n",
                    liftedRefusals);
        failed = true;
    }

    // ---- the combinations --------------------------------------------------
    // The members above are one axis each, and a member being bounded does not
    // make every combination of the axes it belongs to bounded. So the same
    // enumerations are crossed here: the rungs AccuracyTier declares (with each
    // rung's multiplier taken from AccuracyMultiplier), the routes
    // BoysFitRoutes() names, and the schemes BoysEvalSchemes() names. The count
    // is the product of the enumerations' own sizes, so an axis member added to
    // any of them moves it without this block being edited.
    //
    // Every combination is exactly one of these, and this block fails on one of
    // them only:
    //
    //   offered, certified and published  a consumer can express it, and this
    //                            row measured it against the committed
    //                            reference over the whole grid at the bound that
    //                            rung documents. The delivered figure is printed
    //                            with it, and the bound it is judged against is
    //                            the one the public surface reports for that
    //                            rung - so the figure a consumer reads is this
    //                            figure. "Offered" is established by making the
    //                            call this row makes, through the public entry,
    //                            rather than by asserting it.
    //   not yet implemented, and owed  the library refuses the call at the call
    //                            site because the table it would need does not
    //                            exist. The refusal is the library's own, and
    //                            citing it as the reason a bound is not owed is
    //                            circular: the combination is one the library
    //                            should offer, it is owed to a consumer who asks
    //                            for it, and it is listed and counted as a debt
    //                            with the plan on the row. It does not fail this
    //                            check - what fails it is an offered combination
    //                            whose bound is unmet or unpublished. No row is
    //                            in this category at this revision: the rung of
    //                            the rational route is derived, and naming it is
    //                            a call the library answers.
    //   not runnable on this host  the host does not provide what the
    //                            combination needs. A fact about the machine,
    //                            counted apart, and it does not fail.
    //
    // The rung and the route are two selectors of two different things, which is
    // why this cross is not redundant: a rung cuts the named route's own fits -
    // the Chebyshev family's stored coefficients by one degree table, the
    // rational family's stored numerator and denominator pair by another - and
    // the pair is answered by BoysAllOrdersAtTier's route-carrying overload,
    // which is where a caller expresses it.
    //
    // The condition this block carries is the owner's: every *runnable*
    // combination has its bounds measured and published. It applies to the
    // offered ones, so the only failure here is an offered combination whose
    // bound is unmet or covered by no cell - a contract violation. Failing the
    // gate on a combination the library does not offer would make a green pull
    // request red for work that is not owed.
    //
    // A loose true bound is honest and a tight false one is the defect, so a
    // runnable combination delivering worse than it promises is reported as a
    // defect, not as a category of its own.
    //
    // The boundary of this enumeration, stated because a gap in one reads as
    // coverage: the axes crossed are the rungs AccuracyTier declares, the routes
    // BoysFitRoutes() names and the schemes BoysEvalSchemes() names, measured
    // through the two entries that take a route or a rung at run time -
    // BoysAllOrdersWithRoute and BoysAllOrdersAtTier. The *entries* are a fourth
    // axis and are not crossed here: they are measured by the scheme book above
    // (every entry at both schemes, at m = 1 and m = 64) and by the route book's
    // carriage rows, and the rungs above m = 64 reach them only through the two
    // run-time entries this block uses. That is a real boundary and not a claim
    // of coverage.
    struct Combination {
        std::string axes;
        std::string state;
        std::size_t cells = 0;
        std::size_t over = 0;
        double delivered = 0.0;
        double bound = 0.0;
        int worstN = -1;
        double worstX = 0.0;
    };

    std::vector<Combination> combinations;

    {
        std::vector<boys::FitRoute> combRoutes;
        std::vector<const char*> combRouteNames;

        for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
        {
            bool seen = false;

            for (const boys::FitRoute r : combRoutes)
            {
                seen = seen || r == row.route;
            }

            if (!seen)
            {
                combRoutes.push_back(row.route);
                combRouteNames.push_back(row.name);
            }
        }

        const int lastTier = static_cast<int>(boys::AccuracyTier::kRelaxed65536);
        std::array<double, 33> out{};

        for (int t = 0; t <= lastTier; ++t)
        {
            const boys::AccuracyTier tier = static_cast<boys::AccuracyTier>(t);
            const double mult = boys::AccuracyMultiplier(tier);

            for (std::size_t ri = 0; ri < combRoutes.size(); ++ri)
            {
                for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
                {
                    Combination c;
                    c.axes = Fmt("%s, %s, m = %g", combRouteNames[ri], info.name, mult);
                    c.bound = mult * kBoundDoubleBatch;

                    // Every combination on the two axes is offered, so every one
                    // of them is measured here: the rung is a property of the
                    // route rather than of the multiplier (the Chebyshev family's
                    // is a cut of its stored coefficients and the rational
                    // family's a cut of its stored numerator and denominator
                    // pair), and naming the two together is a call this library
                    // answers.
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        if (t == 0)
                        {
                            boys::BoysAllOrdersWithRoute(
                                combRoutes[ri], info.scheme, nmax, ref.x[i], out.data());
                        } else
                        {
                            boys::BoysAllOrdersAtTier(
                                tier, combRoutes[ri], info.scheme, nmax, ref.x[i], out.data());
                        }

                        for (int n = 0; n <= nmax; ++n)
                        {
                            const std::size_t k = ref.Index(n, i);
                            const double got = out[static_cast<std::size_t>(n)];
                            const double err = std::abs(got - ref.v[k]);
                            ++c.cells;

                            if (err > c.bound)
                            {
                                ++c.over;
                            }

                            if (err > c.delivered)
                            {
                                c.delivered = err;
                                c.worstN = n;
                                c.worstX = ref.x[i];
                            }
                        }
                    }

                    c.state = c.over == 0 ? "certified and published" : "DEFECT - runs and "
                                                                        "delivers outside its "
                                                                        "documented bound";
                    combinations.push_back(std::move(c));
                }
            }
        }
    }

    std::size_t combCertified = 0;
    std::size_t combOfferedBad = 0;
    std::size_t combOwed = 0;
    std::size_t combHostLimited = 0;

    for (const Combination& c : combinations)
    {
        if (c.cells == 0 && c.over == 0 && c.state.rfind("not yet implemented", 0) == 0)
        {
            ++combOwed;
        } else if (c.cells == 0 && c.over == 0 && c.state.rfind("not runnable", 0) == 0)
        {
            ++combHostLimited;
        } else if (c.over == 0 && c.cells > 0)
        {
            ++combCertified;
        } else
        {
            // Offered, and either over its bound or covered by no cell at all:
            // the contract violation this block exists to catch.
            ++combOfferedBad;
        }
    }

    std::printf("\n  the combinations: every rung AccuracyTier declares, crossed with every "
                "route\n  BoysFitRoutes() names and every scheme BoysEvalSchemes() names. Each "
                "one is\n  certified and published, not runnable on this host, or outstanding - "
                "expressible\n  and simply not built yet, with the derivation it needs named on "
                "the row. A\n  combination that runs and delivers outside its documented bound "
                "is a defect, and it\n  fails below rather than being a category\n");
    std::printf("  %-40s %9s %9s %-22s %-16s %s\n",
                "combination",
                "cells",
                "over",
                "worst delivered",
                "bound",
                "state");
    std::printf("  %s\n", std::string(150, '-').c_str());

    for (const Combination& c : combinations)
    {
        char where[64];

        if (c.cells > 0)
        {
            std::snprintf(where, sizeof(where), "n=%d, x=%.6g", c.worstN, c.worstX);
        } else
        {
            std::snprintf(where, sizeof(where), "-");
        }

        std::printf("  %-40s %9zu %9zu %-22.6g %-16.6g %s\n",
                    c.axes.c_str(),
                    c.cells,
                    c.over,
                    c.delivered,
                    c.bound,
                    c.state.c_str());
    }

    std::printf("  %s\n", std::string(150, '-').c_str());
    std::printf("  the host provides AVX2 %s. The combinations above are the scalar surface and "
                "none of\n  them needs a host feature, so 'not runnable on this host' is 0 here. "
                "A member the\n  host does not provide is counted in the member table above and "
                "not against the\n  library - the packed lanes are the ones gated on that "
                "question, and the member\n  count moves with them because it is read off "
                "BoysBackends()\n",
                boys::BoysAvx2Available() ? "present" : "absent");

    std::printf("  COMBINATIONS: %zu of %zu offered, certified and published - %zu defect(s)\n",
                combCertified,
                combinations.size() - combOwed - combHostLimited,
                combOfferedBad);
    std::printf("  NOT YET IMPLEMENTED and owed: %zu. A combination lands here when the "
                "library\n  refuses the call at the call site because the table it would need "
                "does not exist.\n  The refusal is the library's own and not a property of the "
                "combinations, and each\n  such combination is owed to a consumer who asks for "
                "it - so they are listed and\n  counted rather than dismissed. %zu "
                "combination(s) are not runnable on this host\n",
                combOwed,
                combHostLimited);

    if (combOwed > 0)
    {
        std::printf("  OWED, listed and counted so the debt can be read. None of them is an\n"
                    "  impossibility and none of them fails this check; what fails it is an "
                    "offered\n  combination whose bound is unmet or unpublished:\n");

        for (const Combination& c : combinations)
        {
            if (c.cells == 0 && c.over == 0 && c.state.rfind("not yet implemented", 0) == 0)
            {
                std::printf("    - %s [%s]\n", c.axes.c_str(), c.state.c_str());
            }
        }
    }

    if (combOfferedBad > 0)
    {
        std::printf("  NOT DELIVERED at this revision:");

        for (const Combination& c : combinations)
        {
            if (c.state.rfind("not yet implemented", 0) != 0 &&
                c.state.rfind("not runnable", 0) != 0 &&
                (c.over > 0 || c.cells == 0))
            {
                std::printf(" [%s: %s]", c.axes.c_str(),
                            c.cells == 0 ? "offered and covered by no cell"
                                         : "delivers outside its documented bound");
            }
        }

        std::printf("\n  FAIL (an offered combination whose bound is unmet or not published: "
                    "that is the\n  condition, and it is the only thing in this block that "
                    "turns the gate red)\n");
        failed = true;
    }

    std::printf("  OPTION SPACE: %zu of %zu member(s) supported, bounded and reachable\n",
                optionSpace.size() - optionMissing,
                optionSpace.size());

    if (optionMissing > 0)
    {
        std::printf("  NOT DELIVERED at this revision:");

        for (const OptionMember& o : optionSpace)
        {
            if (!(o.supported && o.bounded && o.reachable))
            {
                std::printf(" [%s %s: ", o.kind, o.member.c_str());
                std::printf("%s%s%s]", o.supported ? "" : "not supported ",
                            o.bounded ? "" : "not bounded ", o.reachable ? "" : "not reachable");
            }
        }

        std::printf("\n  FAIL (the option-space check; the list above is the worklist, not a "
                    "note)\n");
        failed = true;
    }

    if (failed)
    {
        std::printf("  FAIL (exit status 1; every book above was measured and printed)\n");
        return 1;
    }

    // ---- the packing-axis rows, counted apart ------------------------------
    // Their own block, their own cell count and their own RESULT line. The two
    // books above are the numbers a reader has seen before and the axis must
    // not move either of them: this book's cells are the axis's own, and the
    // fraction below is the share of them that can discriminate, so it is the
    // axis's discriminating fraction and not a re-reading of the lane's.
    std::size_t packCells = 0;
    std::size_t packNonDiscriminating = 0;

    for (const Accum& a : PackClaims())
    {
        packCells += a.points;
        packNonDiscriminating += a.vacuous;
    }

    std::printf("\nthe packing-axis rows: which of a call's values a packed lane carries, "
                "measured through the two entries the axis is carried on - the all-orders entry, "
                "whose call shape is four orders of one\nargument, and the plane entry, whose "
                "call shape has both - against the committed reference\n");
    std::printf("  %-16s %-20s %9s %-22s %-22s %7s  %-22s %s\n",
                "axis",
                "row",
                "cells",
                "bound it promises",
                "worst delivered",
                "ratio",
                "worst cell",
                "verdict");
    std::printf("  %s\n", std::string(160, '-').c_str());

    int packMet = 0;
    std::size_t packRows = 0;
    std::vector<std::string> packNotMet;

    const auto packRow = [&](const char* axis, const char* row, double bound, const Accum& a) {
        ++packRows;
        const Verdict v = FromAccum(a);

        if (IsMet(v))
        {
            ++packMet;
        } else
        {
            packNotMet.push_back(std::string(axis) + " / " + row);
        }

        char where[64];
        std::snprintf(where, sizeof(where), "n=%d, x=%.6g", a.worstN, a.worstX);
        std::printf("  %-16s %-20s %9zu %-22.6g %-22.6g %7.3g  %-22s %s\n",
                    axis,
                    row,
                    a.points,
                    bound,
                    a.worstErr,
                    a.worstRatio,
                    where,
                    VerdictName(v));
    };

    for (std::size_t row = 0; row < packSchemes.size(); ++row)
    {
        const char* axis = boys::EvalSchemeName(packSchemes[row]);
        packRow(axis,
                "region A only",
                kBoundSingleA,
                PackClaims()[static_cast<std::size_t>(packSlots[2 * row])]);
        packRow(axis,
                "whole grid, A..C",
                kBoundSingleC,
                PackClaims()[static_cast<std::size_t>(packSlots[2 * row + 1])]);
    }

    // The plane entry's rows, beside the all-orders entry's and not folded into
    // them: the axis's values are the same on the two shapes - the plane
    // entry's per-argument path is the all-orders entry's body - so the rows
    // are the same two questions asked of a different call. Keeping them apart
    // is what lets a reader see that the axis is carried on both.
    for (std::size_t row = 0; row < packSchemes.size(); ++row)
    {
        const char* axis = boys::EvalSchemeName(packSchemes[row]);
        packRow(axis,
                "plane entry, region A",
                kBoundSingleA,
                PackClaims()[static_cast<std::size_t>(packPlaneSlots[2 * row])]);
        packRow(axis,
                "plane entry, A..C",
                kBoundDoubleBatch,
                PackClaims()[static_cast<std::size_t>(packPlaneSlots[2 * row + 1])]);
    }

    std::printf("  %s\n", std::string(160, '-').c_str());
    std::printf("  PACK RESULT: %d of %zu packing-axis rows met at this revision\n",
                packMet,
                packRows);

    if (packCells > 0)
    {
        std::printf("                carried by the %zu of %zu axis cells (%.1f%%) that can "
                    "discriminate: the other %zu carry a bound at least as large as the value "
                    "itself. These cells are not in the counts above and do not move them\n",
                    packCells - packNonDiscriminating,
                    packCells,
                    100.0 * static_cast<double>(packCells - packNonDiscriminating)
                        / static_cast<double>(packCells),
                    packNonDiscriminating);
    }

    if (!packNotMet.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : packNotMet)
        {
            std::printf(" %s", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; the packing-axis rows are judged with the books "
                    "above)\n");
        return 1;
    }

    std::printf("  PASS: every documented claim met at this revision\n");
    return 0;
}
