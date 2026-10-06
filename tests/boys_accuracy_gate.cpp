// The accuracy gate: every documented per-lane, per-region bound measured against the committed
// mpmath grid (tests/data/boys_accuracy_gate_reference.csv, re-derived by
// tools/gen_boys_accuracy_gate_reference.py, whose routes agree to 1e-81 - 65 to 81 orders below
// the tightest bound here). Any claim not verified at this revision exits non-zero.

#include <boys/boys.hpp>
#include <boys/f16.hpp>

// The store-half lanes and the packed simd-half kernels are declared behind the BoysFp16
// build-time seam, so a build whose seam is closed has no entry to call; the claim slots exist
// either way, so only their verdicts move with the seam.
#if BoysFp16
#define BOYS_GATE_FP16 1
#endif

// The native packed half lane arrives on its own branch; a revision without it reports the
// lane's claims as evidence absent rather than skipping them. boys/half2.hpp is unconditional
// while the two entries that take it are behind the BoysFp16 seam, and the two facts are told
// apart below rather than reported as one.
#if __has_include("boys/half2.hpp")
#include <boys/half2.hpp>
#define BOYS_GATE_NATIVE_HALF_HEADER 1
#endif

#if defined(BOYS_GATE_NATIVE_HALF_HEADER) && BoysFp16
#define BOYS_GATE_NATIVE_HALF 1
#endif

// Why the native packed half lane is not measured here; empty when it is. The
// row prints this instead of a measurement, so a reader of either build is told
// which of the two reasons applies rather than left to infer one.
constexpr const char* kNativeHalfAbsent =
#if !defined(BOYS_GATE_NATIVE_HALF_HEADER)
    "the lane is not in this revision: no boys/half2.hpp, so neither the packed type nor the "
    "entries that take it are here, and the lane's claims are neither confirmed nor refuted";
#elif !defined(BOYS_GATE_NATIVE_HALF)
    "the lane is not in this build: boys/half2.hpp is here and defines the packed type, but "
    "BoysAllOrdersHalf2 and BoysAllNF16Native are declared behind the BoysFp16 seam, which this "
    "build has closed, so there is no entry to call and the lane's claims are neither confirmed "
    "nor refuted here";
#else
    "";
#endif

// The region-A transform lane ships its own header; a revision without it reports the lane's
// claims as evidence absent rather than skipping them.
#if __has_include("boys/boys_transform.hpp")
#include <boys/boys_transform.hpp>
#define BOYS_GATE_TRANSFORM 1
#endif

// The device lane's arm is compiled only where CMake states BOYS_GATE_CUDA and links
// boys-cuda, so a build without the lane carries no arm and no device call, and its cross counts
// the device members apart with the build's own reason.
#ifdef BOYS_GATE_CUDA

#include <boys/boys_cuda.hpp>
#endif

#include "boys/boys_impl.hpp" // the region kernels
#include "boys_gate_reference.hpp" // the committed reference, the row shape

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#ifdef BOYS_GATE_CUDA
#include <cuda_runtime.h>
#endif
#include <fstream>
#include <iterator>
#include <limits>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// CMake passes the measured revision; a build that bypasses it still says so
// rather than naming a revision it did not read.
#ifndef BoysGateRevision
#define BoysGateRevision "unknown"
#endif

// The instrument, the reference and the row shape come from tests/boys_gate_reference.hpp,
// shared with the CUDA device gate; this file's own books are the lanes, schemes, routes and the
// bounds each is judged by.
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
// The withdrawn header claim (1e-15 over every x < x0): its slot is still measured so the
// finding stays re-measurable, but no live claim is judged by it.
constexpr double kWithdrawnHeaderABound = 1e-15;

// The native packed half lane (include/boys/half2.hpp): region C only, the ladder in binary16
// scaled by 2^15 so it stays normal down to F_k(x) = 2^-29, and its published bound is in ULP of
// the returned (scaled) value.
constexpr double kNativeScale = 32768.0; // 2^15, the entry's per-order scale
constexpr double kNativeUlpBound = 8.0;  // the published bound, in ULP
constexpr double kNativeSmallestNormal = 6.103515625e-05; // 2^-14
// bfloat16: 7 stored mantissa bits, smallest normal exponent -126.
constexpr int kBf16MantissaBits = 7;
constexpr int kBf16MinNormalExp = -126;

// The field a binary16 value spans, largest finite magnitude to smallest subnormal: 65504 = 2^16
// down to 2^-24, so 2^40. A per-order power-of-two scale can only move a value inside it.
constexpr double kF16Field = 65504.0 * 16777216.0;        // 65504 * 2^24
constexpr double kF16NormalField = 65504.0 * 16384.0;     // 65504 * 2^14

// The region-A transform lane (include/boys/boys_transform.hpp): the double table's two shared
// bands as one matrix product per band, in a caller-named mode. kFp64 carries the double single
// region-A budget, the two split modes that plus the fp32 accumulator's own floor (2.5e-7); their
// rows model that arithmetic rather than measuring a tensor core.
constexpr double kTransformSplitFloor = 2.5e-7;
constexpr double kTransformFp64Bound = kBoundSingleA; // 1e-15
constexpr double kTransformSplitBound = kBoundSingleA + kTransformSplitFloor;

// The delivered worst each mode publishes over region A: the header's table gives 1.110e-16,
// 1.916e-07 and 1.946e-07, which README and docs/lane-contract.md round to 1.11e-16, 1.92e-07 and
// 1.95e-07. The rows hold the sweep to the tighter reading.
constexpr double kTransformFp64Delivered = 1.11e-16;
constexpr double kTransformTf32x3Delivered = 1.916e-07;
constexpr double kTransformBf16x6Delivered = 1.946e-07;
// The figures above are claims to four figures: the fp64 mode's 1.110e-16 is 2^-53 - the exact
// maximum is 1.11022e-16 - written that way. A delivered figure is a swept maximum, not a bound:
// the bound is the B column beside it in the same table.
constexpr int kTransformFigures = 4;
// The evaluation-scheme book's accumulators, held apart from the claim book so that a row added
// for a new evaluation option cannot move the RESULT line or the cell count; they are judged in
// their own block, and a row of theirs that is not met leaves the gate red just the same.
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

// The packing axis's own book, apart from the claim and scheme books for the reason both give.
// It is reported after them because the report's order is the order the books were added, which
// keeps the earlier blocks byte-stable.
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

// The granularity axis's own book, apart from the three above: its cells are not the lane cells
// the claim book's count is quoted against, and its block is printed after theirs.
std::vector<Accum>& GranularityClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddGranularityClaim(const char* lane, const char* region, double bound) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    GranularityClaims().push_back(a);
    return static_cast<int>(GranularityClaims().size()) - 1;
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

// The float lane's policy combinations, counted apart from the books above: a policy is a pair
// a call site names, and the cells that measure one are not the lane cells the claim count is
// quoted against.
std::vector<Accum>& F32PolicyClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddF32PolicyClaim(const char* lane, const char* region, double bound) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    F32PolicyClaims().push_back(a);
    return static_cast<int>(F32PolicyClaims().size()) - 1;
}

// The entry book, apart from the five books above: every other book reaches its combinations
// through one entry per lane, so a combination these axes admit and another entry refuses is in
// no book at all until it is counted here.
std::vector<Accum>& EntryClaims() {
    static std::vector<Accum> claims;
    return claims;
}

int AddEntryClaim(const char* lane, const char* region, double bound) {
    Accum a;
    a.lane = lane;
    a.region = region;
    a.baseBound = bound;
    EntryClaims().push_back(a);
    return static_cast<int>(EntryClaims().size()) - 1;
}

// The region a double-single argument is dispatched in, by the README interval table: the
// library's own boundary inside the band is per-order, but the published band cell is an interval
// and the looser of the two candidate bounds covers it.
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

// The float lane's route rows are quoted under their own label: the two lanes
// carry their own tables, so a reader must not fold a float row's cells into a
// double route's row of the same (route, region) name.
const char* RouteLaneF32(boys::FitRoute route, boys::AccuracyRegion region) {
    const bool rational = (route == boys::FitRoute::kRationalMinimax);

    switch (region)
    {
    case boys::AccuracyRegion::kA:
        return rational ? "float rational A" : "float chebyshev A";
    case boys::AccuracyRegion::kB:
        return rational ? "float rational B" : "float chebyshev B";
    case boys::AccuracyRegion::kC:
        break;
    }

    return "?";
}

// The policy the scheme book's rows are measured under: the shipped Chebyshev
// route at the scheme the row names. That is the pair BoysEvalSchemeFits()
// reports, and the pair a call site reaches by naming one EvalPolicy.
template <boys::EvalScheme kScheme>
using SchemePolicy = boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme>;

// The same pair with the partition named: the packed region-A lane holds one stored table and
// one recurrence, so it is the shipped partition's and no other's - a probe that left the
// partition to the default would read the partition while claiming to read the scheme.
template <boys::EvalScheme kScheme>
using SchemeLanePolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev,
                     kScheme,
                     boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;

// The granularity book's policy: the shipped route at the scheme the row names, with the
// partition named, which is the whole of the axis.
template <boys::EvalScheme kScheme, boys::FitGranularity kGranularity>
using GranularityPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev,
                     kScheme,
                     boys::BoysBudget::kFloat,
                     boys::PackAxis::kArguments,
                     kGranularity>;

// The asymptotic form as boys_impl.hpp spells it: sqrt(pi)/2 over sqrt(x), then one
// multiply-and-divide per order. It is evaluated here rather than read back out of
// BoysRegionCSimd, whose vector body applies the form to four lanes whatever the arguments
// are while a host without AVX2 falls back to the scalar entry - so below kX1 neither defines it.
inline double RegionCForm(int n, double x) noexcept {
    double f = boys::detail::kBoysHalfSqrtPi / std::sqrt(x);

    for (int l = 0; l < n; ++l)
    {
        f = (l + 0.5) * f / x;
    }

    return f;
}

// The stored fit one lane names at one partition, summed by one scheme. It is
// FitValue with the partition named rather than assumed, and the shipped member
// reads what FitValue reads: the same pieces, the same coefficients, the same
// recurrence.
template <boys::EvalScheme kScheme, boys::FitGranularity kGranularity>
double PartitionFitValue(boys::EvalLane lane, int n, double x) {
    using Fit = boys::detail::ChebyshevFit<kScheme, kGranularity>;

    switch (lane)
    {
    case boys::EvalLane::kRegionA:
        return boys::detail::RegionAValue<Fit>(n, x);
    case boys::EvalLane::kRegionB:
        return Fit::RegionBSeed(x);
    case boys::EvalLane::kExtendedBand:
        return boys::detail::RegionBExtendedSeed<kScheme>(x);
    }

    return 0.0;
}

// The packing book's policy: the shipped route and the scheme the row names at the orders axis,
// which is the axis's only difference from SchemePolicy.
template <boys::EvalScheme kScheme>
using OrdersPackPolicy =
    boys::EvalPolicy<boys::FitRoute::kChebyshev, kScheme, boys::BoysBudget::kFloat,
                     boys::PackAxis::kOrders>;

// The stored fit one lane names, summed by one scheme, on the shipped partition.
// This is the kernel the double single entries dispatch to, at the scheme they
// were compiled with; the partition is named rather than left to the default, so
// that what this reads does not move when the default does.
template <boys::EvalScheme kScheme>
double FitValue(boys::EvalLane lane, int n, double x) {
    return PartitionFitValue<kScheme, boys::FitGranularity::kCoarsest>(lane, n, x);
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

// A value's significand to `figures` figures as an integer, so two published figures can be
// compared at the precision the document states them at, without the comparison itself rounding.
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

// One line for a lane this build does not carry, named with its reason rather than omitted: a
// lane missing from the list reads as a lane that was measured.
#ifndef BOYS_GATE_FP16
void PrintNotCarried(const char* lane)
{
    std::printf("  %-28s %s\n", lane, "not carried: this build's BoysFp16 seam is closed");
}
#endif // BOYS_GATE_FP16

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
#ifdef BOYS_GATE_FP16
    double ref16 = refV;
    double refB = refV;
#endif

    for (std::size_t i = 0; i < ref.count; ++i)
    {
        if (ref.x[i] == x)
        {
            const std::size_t k = ref.Index(n, i);
            refV = ref.v[k];
            refF = ref.vf[k];
#ifdef BOYS_GATE_FP16
            ref16 = ref.v16[k];
            refB = ref.vb[k];
#endif
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
    }

    {
        const float xf = static_cast<float>(x);
        std::array<float, 33> out{};
        row("BoysSingleF32", static_cast<double>(boys::BoysSingleF32(n, xf)), refF);
        boys::BoysAllOrdersF32(n, xf, out.data());
        row("BoysAllOrdersF32[n]", static_cast<double>(out[static_cast<std::size_t>(n)]), refF);

#ifdef BOYS_GATE_FP16
        const boys::F16 x16 = boys::F16(xf);
        const boys::Bf16 xb = boys::Bf16(xf);
        std::array<boys::F16, 33> out16{};
        std::array<boys::Bf16, 33> outb{};
        // The division-form control for code.half_simd_budget: the 8-wide half-I/O bodies form a
        // plain reciprocal and answer zero where the certified scalar half entry, running the float
        // engine, divides through the form the policy names. Naming each form here separates a
        // formula from the entry that reaches it - the engine's answer would make all three agree.
        {
            using PExact = boys::EvalPolicy<boys::kDefaultFitRoute,
                                            boys::kDefaultEvalScheme,
                                            boys::BoysBudget::kFloat,
                                            boys::kDefaultPackAxis,
                                            boys::kDefaultFitGranularity,
                                            boys::DivisionForm::kExactDivision>;
            using PPlain = boys::EvalPolicy<boys::kDefaultFitRoute,
                                            boys::kDefaultEvalScheme,
                                            boys::BoysBudget::kFloat,
                                            boys::kDefaultPackAxis,
                                            boys::kDefaultFitGranularity,
                                            boys::DivisionForm::kPlainReciprocal>;
            using PRefined = boys::EvalPolicy<boys::kDefaultFitRoute,
                                              boys::kDefaultEvalScheme,
                                              boys::BoysBudget::kFloat,
                                              boys::kDefaultPackAxis,
                                              boys::kDefaultFitGranularity,
                                              boys::DivisionForm::kRefinedReciprocal>;
            const float inf = std::numeric_limits<float>::infinity();
            row("BoysSingleF32 exact division [inf]",
                static_cast<double>(boys::BoysSingleF32< PExact>(n, inf)),
                ref16);
            row("BoysSingleF32 plain reciprocal [inf]",
                static_cast<double>(boys::BoysSingleF32< PPlain>(n, inf)),
                ref16);
            row("BoysSingleF32 refined reciprocal [inf]",
                static_cast<double>(boys::BoysSingleF32< PRefined>(n, inf)),
                ref16);
        }
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
#else
        PrintNotCarried("BoysSingleF16");
        PrintNotCarried("BoysAllOrdersF16[n]");
        PrintNotCarried("BoysSingleBf16");
        PrintNotCarried("BoysAllOrdersBf16[n]");
#endif // BOYS_GATE_FP16
    }


    if (!boys::BoysAvx2Available())
    {
        return;
    }

    {
        // The fp64 region kernels at count = 1 (the scalar tail) and count = 4 (the vector path).
        // Each shape is read against the reference and not against the other: the two bodies are
        // different arithmetic - the vector body applies the form to four lanes and its region-C
        // seed is sqrt(1/x) - and above kX1 both are the same form judged by the same budget.
        const auto simdRow = [&](const char* lane, std::size_t count) {
            std::vector<double> xs(count, x);
            std::vector<double> out(count);
            std::vector<double> plane(count * (static_cast<std::size_t>(n) + 1));
            double got = 0.0;

            if (x < boys::detail::kX0)
            {
                boys::detail::BoysRegionASimd(n, xs.data(), out.data(), count);
                got = out[0];
            } else if (x < boys::detail::kX1)
            {
                boys::detail::BoysRegionBSimd(n, xs.data(), plane.data(), count);
                got = plane[static_cast<std::size_t>(n) * count];
            } else
            {
                boys::detail::BoysRegionCSimd(n, xs.data(), out.data(), count);
                got = out[0];
            }

            row(lane, got, refV);
        };

        simdRow("RegionASimd[count=1]", 1);
        simdRow("RegionASimd[count=4]", 4);
        simdRow("RegionBSimd[count=1]", 1);
        simdRow("RegionBSimd[count=4]", 4);
        simdRow("RegionCSimd[count=1]", 1);
        simdRow("RegionCSimd[count=4]", 4);
    }

#ifdef BOYS_GATE_FP16
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
#else
    PrintNotCarried("RegionSimdF16[count=1]");
    PrintNotCarried("RegionSimdF16[count=8]");
    PrintNotCarried("RegionSimdBf16[count=1]");
    PrintNotCarried("RegionSimdBf16[count=8]");
#endif // BOYS_GATE_FP16
}

// ---------------------------------------------------------------------------
// The reference, scanned: the published range statements are all of the shape "the value
// leaves the format's normal range at x", so they are read out of this table, not asserted.
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

// The documented claims, and the verdict this run gives each one. Vacuous: every point that
// meets a claim is one where the bound is at least the function's own magnitude - the format's
// floor, not the lane's. EvidenceAbsent: the artifact named is not in this tree. NotCarried:
// this build closed the seam the claim's entries are behind, which is neither pass nor failure.
enum class Verdict
{
    Verified,
    MetOverDomain,
    Exceeded,
    Vacuous,
    EvidenceAbsent,
    NotCarried
};

// What the rows about the native packed half lane report when it was not measured here: a
// revision with no boys/half2.hpp has not carried the lane's evidence, while a build that has
// the type and a closed fp16 seam does not carry its subject - two different verdicts.
constexpr Verdict kNativeHalfNotMeasured =
#if defined(BOYS_GATE_NATIVE_HALF) || !defined(BOYS_GATE_NATIVE_HALF_HEADER)
    Verdict::EvidenceAbsent;
#else
    Verdict::NotCarried;
#endif

// A domain-scoped claim - one the document itself restricts to a stated set of regions,
// orders or arguments - is met over that domain and recorded as such, with the domain printed
// beside it: a restriction the document states is not missing evidence.
bool IsMet(Verdict v)
{
    return v == Verdict::Verified || v == Verdict::MetOverDomain;
}

// A row leaves this run red when it is not met and this build carries its subject: a claim the
// build carries and cannot verify at this revision is the defect the gate exists to find, and it
// fails whatever the reason it could not be verified. Not-carried rows are named in full.
bool FailsTheGate(Verdict v)
{
    return !IsMet(v) && v != Verdict::NotCarried;
}

// The order the verdicts worsen in, for an accumulation that takes the worst
// of several rows. Exceeded is the worst because it is the only one a
// measurement produced; the rest say what the evidence did.
int VerdictRank(Verdict v)
{
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
    case Verdict::NotCarried:
        // Not a verdict an accumulation can reach - FromAccum has no path to it
        // - and deliberately below every measured state: if one ever did reach
        // an accumulation, a row the build does not carry must not outrank a
        // row the build measured and failed.
        return -1;
    }

    return 5;
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
    case Verdict::NotCarried:
        return "not carried by this build";
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
    case Verdict::NotCarried:
        return "nothing here can falsify this row: the build under test does not carry the "
               "thing the claim is about, so there is no entry to call and no cell to judge. "
               "It becomes a check, and can then fail, in a build that carries the entry - one "
               "whose fp16 seam is open - and the reason this build does not is printed with "
               "the row";
    }

    return "?";
}

// What a claim's verdict is a verdict about. A model-derived row earned its verdict from a
// software model of an accumulator rather than from a card: a tensor core's fused sum truncates
// and aligns where the model rounds, which can make a card worse and never better - so a green
// model row is counted apart from the certified total for that reason and no other.
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

// Region B's amplification A_B(n) = prod_{j=1..n} (j + 1/2) / x0^n, the quantity the seed-sizing
// argument bounds. Double-double is required: the excess it measures at the top supported order
// is 1.8e-17, which binary64 cannot represent beside 1.0 at all, so a double evaluation would
// confirm "at most one" by rounding - which is exactly how a claim like this survives.
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

#ifdef BOYS_GATE_CUDA
// --- the device lane's arm: the entry a member is read through --------------
// The lane is a surface of entries, one name per (route, partition, packing axis), mapped here to
// the cross by the library's own statements (src/boys.cpp, CarriesDevice; boys_cuda_options.hpp).
// Where it stores one arithmetic under two scheme names, both members read that one entry.

/// The division form the carriage question below is asked at. Whether a lane
/// carries a member is a property of the lane and the region and not of the
/// form: every carrier this library answers with takes no form at all
/// (src/boys.cpp, CarriesDevice, CarriesDeviceF64, CarriesDeviceF16,
/// CarriesSingle), so one call answers it for all three of them. The figure a
/// carried member is judged by does depend on the form, and the device arms
/// below read each of their cells at every form the axis answers rather than at
/// this one.
constexpr boys::DivisionForm kGateDivisionForm = boys::kDefaultDivisionForm;

/// The entry one member of the device lane's cross is measured through, read at
/// each entry's default policy.
///
/// \param route     the member's fit route: the shipped float ladder or the
///                  rational pair over the same pieces
/// \param scheme    the member's summation; the names reach one entry on the
///                  rows the library states are one row for both
/// \param partition the member's partition: the shipped ladder, the narrow
///                  pieces or the uniform grid
/// \param axis      the member's packing axis: per argument or across orders
///
/// \returns the launched entry that serves the member, or \c nullptr for a
///          partition this library does not carry - which the cross never
///          names, because it enumerates BoysFitGranularities()
constexpr auto GateDeviceEntry(boys::FitRoute route,
                               boys::EvalScheme scheme,
                               boys::FitGranularity partition,
                               boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const double*, float*, std::size_t, void*,
                            boys::DivisionForm) {
    const bool orders = axis == boys::PackAxis::kOrders;
    const bool horner = scheme == boys::EvalScheme::kHorner;
    const bool rational = route == boys::FitRoute::kRationalMinimax;

    if (partition == boys::FitGranularity::kCoarsest)
    {
        // Both scheme names reach one entry on each packing axis of this
        // partition: AllOrdersF32 and its orders-axis sibling are the shipped
        // float ladder, summed the one way the library stores it.
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF32OrdersRatHorner
                              : &boys::BoysCuda::AllOrdersF32OrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF32RatHorner
                          : &boys::BoysCuda::AllOrdersF32Rat;
        }

        return orders ? &boys::BoysCuda::AllOrdersF32Orders
                      : &boys::BoysCuda::AllOrdersF32;
    }

    if (partition == boys::FitGranularity::kNarrow)
    {
        // The narrow pieces are stored in two bases and the scheme names the
        // one a row sums (the monomial name), and the route's pair is stored in
        // one and reached by two names of its own.
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF32NarrowOrdersRatHorner
                              : &boys::BoysCuda::AllOrdersF32NarrowOrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF32NarrowRatHorner
                          : &boys::BoysCuda::AllOrdersF32NarrowRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF32NarrowOrdersMono
                          : &boys::BoysCuda::AllOrdersF32NarrowOrders;
        }

        return horner ? &boys::BoysCuda::AllOrdersF32NarrowMono
                      : &boys::BoysCuda::AllOrdersF32Narrow;
    }

    if (partition == boys::FitGranularity::kUniform)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF32OrdersUniformRatHorner
                              : &boys::BoysCuda::AllOrdersF32OrdersUniformRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF32UniformRatHorner
                          : &boys::BoysCuda::AllOrdersF32UniformRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF32OrdersUniformHorner
                          : &boys::BoysCuda::AllOrdersF32OrdersUniform;
        }

        return horner ? &boys::BoysCuda::AllOrdersF32UniformHorner
                      : &boys::BoysCuda::AllOrdersF32Uniform;
    }

    return nullptr;
}

/// The entry one member of the double device lane's cross is measured through,
/// read at each entry's default policy.
///
/// The lane's surface is the double lane's own family of entries, one name per
/// (route, partition, packing axis), and which name serves a member is the
/// library's statement about its own arithmetic rather than this gate's: the
/// fp64 rows of BoysDeviceOptions() state each entry's route, scheme, packing
/// axis and partition (src/boys_cuda.cpp, read from DeviceEntryAxesOf,
/// boys_cuda_options.hpp), and the one collapse in the map below is that
/// statement's - on the coarsest ladder and on the narrow pieces the rational
/// route is one function per pair and not one per name: its -Horner row names
/// the same entry as its plain row, at the other scheme (boys_cuda_options.hpp,
/// kAllOrdersF64Rat and kAllOrdersF64RatHorner, and the sentence over those
/// cases, "the two scheme names a caller may use reach one kernel and each row
/// states the name it was reached by"). The pair's one arithmetic is
/// AllOrdersF64Rat's: "the pair is stored once, so both scheme names select this
/// arithmetic and the scheme axis is inert here" (boys_cuda.hpp,
/// AllOrdersF64Rat). Both members of the cross are measured and published all
/// the same, as the float lane's note above states for the members it collapses
/// the same way. The uniform grid's rational pair is two names at this
/// revision, AllOrdersF64UniformRat and its -Horner twin, and each member is
/// reached by the name its own scheme names.
///
/// \param route     the member's fit route: the Chebyshev pieces or the
///                  rational pair over the same pieces and intervals
/// \param scheme    the member's summation; where the library stores one
///                  arithmetic under both names the second name reaches it too
/// \param partition the member's partition: the shipped ladder, the narrow
///                  pieces or the uniform grid
/// \param axis      the member's packing axis: per argument or across orders
///
/// \returns the launched entry that serves the member, or \c nullptr for a
///          member this map has not been taught - which the arm below reports
///          and fails on, rather than measuring it under another name
constexpr auto GateDeviceEntryF64(boys::FitRoute route,
                                  boys::EvalScheme scheme,
                                  boys::FitGranularity partition,
                                  boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const double*, double*, std::size_t, void*,
                            boys::DivisionForm) {
    const bool orders = axis == boys::PackAxis::kOrders;
    const bool horner = scheme == boys::EvalScheme::kHorner;
    const bool rational = route == boys::FitRoute::kRationalMinimax;

    if (partition == boys::FitGranularity::kCoarsest)
    {
        if (rational)
        {
            // One function carries both scheme names of this pair (boys_cuda_options.hpp,
            // kAllOrdersF64Rat and kAllOrdersF64RatHorner), so both members of the cross are
            // read through it.
            return orders ? &boys::BoysCuda::AllOrdersF64OrdersRat
                          : &boys::BoysCuda::AllOrdersF64Rat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF64OrdersMono
                          : &boys::BoysCuda::AllOrdersF64Orders;
        }

        return horner ? &boys::BoysCuda::AllOrdersF64Mono : &boys::BoysCuda::AllOrdersF64;
    }

    if (partition == boys::FitGranularity::kNarrow)
    {
        // The narrow pieces, summed in the two bases the scheme names select -
        // and again with the route's pair stored once and reached by both of
        // its names.
        if (rational)
        {
            return orders ? &boys::BoysCuda::AllOrdersF64NarrowOrdersRat
                          : &boys::BoysCuda::AllOrdersF64NarrowRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF64NarrowOrdersMono
                          : &boys::BoysCuda::AllOrdersF64NarrowOrders;
        }

        return horner ? &boys::BoysCuda::AllOrdersF64NarrowMono
                      : &boys::BoysCuda::AllOrdersF64Narrow;
    }

    if (partition == boys::FitGranularity::kUniform)
    {
        // The grid, whose rows state one packing axis for both of its names -
        // a grid stored one fit per order and interval has no seeded ladder to
        // step - and whose rational pair is two names at this revision.
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF64OrdersUniformRatHorner
                              : &boys::BoysCuda::AllOrdersF64OrdersUniformRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF64UniformRatHorner
                          : &boys::BoysCuda::AllOrdersF64UniformRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF64OrdersUniformHorner
                          : &boys::BoysCuda::AllOrdersF64OrdersUniform;
        }

        return horner ? &boys::BoysCuda::AllOrdersF64UniformHorner
                      : &boys::BoysCuda::AllOrdersF64Uniform;
    }

    return nullptr;
}

#ifdef BOYS_GATE_FP16
/// The entry one member of the half device lane's cross is measured through,
/// read at each entry's default policy.
///
/// The lane's surface is the fp16 rows of the same option table the double
/// lane's map above reads (src/boys_cuda.cpp, the fp16 rows and the columns
/// DeviceEntryAxesOf states for them), and the names this map returns are those
/// rows': the lane's 24 members are read through the half lane's own stored
/// tables, one name per member beside the one collapse below. So the map
/// answers an entry for each member rather than for one corner of the cross,
/// and the members it cannot serve are the ones the lane's own carrier refuses
/// - which the arm below asks about first (src/boys.cpp, CarriesDeviceF16) and
/// which this map is never asked for.
///
/// The one collapse in the map is the option table's and not this gate's: the
/// half lane's coarsest partition is one row for both scheme names, its
/// per-argument row and its orders-axis counterpart alike (boys_cuda_options.hpp,
/// "the coarsest partition's row is one row for both scheme names as
/// kAllOrdersF16 is"), where the double table's same route carries a second row
/// per axis on kScheme (src/boys_cuda.cpp, all-orders-fp64-mono and
/// all-orders-fp64-orders-mono). So both scheme names of this cross reach that
/// one row's entry, and the map returns it for both. Everywhere else the two
/// scheme names are two rows, and the map returns the name its own scheme
/// names; on the rational route those two names are one kernel, the Horner name
/// a forwarder to the other (boys_cuda.hpp, AllOrdersF16Rat: "both scheme names
/// reach this one entry", and AllOrdersF16RatHorner: "A forwarder and not a
/// second arithmetic").
///
/// \param route     the member's fit route: the Chebyshev pieces or the
///                  rational pair over the same pieces and intervals
/// \param scheme    the member's summation; where the library stores one
///                  arithmetic under both names the second name reaches it too
/// \param partition the member's partition: the shipped ladder, the narrow
///                  pieces or the uniform grid
/// \param axis      the member's packing axis: per argument or across orders
///
/// \returns the launched entry that serves the member, or \c nullptr for a
///          member this lane's surface has no entry for
constexpr auto GateDeviceEntryF16(boys::FitRoute route,
                                  boys::EvalScheme scheme,
                                  boys::FitGranularity partition,
                                  boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const boys::F16*, boys::F16*, std::size_t, void*,
                            boys::DivisionForm) {
    const bool orders = axis == boys::PackAxis::kOrders;
    const bool horner = scheme == boys::EvalScheme::kHorner;
    const bool rational = route == boys::FitRoute::kRationalMinimax;

    if (partition == boys::FitGranularity::kCoarsest)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF16OrdersRatHorner
                              : &boys::BoysCuda::AllOrdersF16OrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF16RatHorner
                          : &boys::BoysCuda::AllOrdersF16Rat;
        }

        return orders ? &boys::BoysCuda::AllOrdersF16Orders : &boys::BoysCuda::AllOrdersF16;
    }

    if (partition == boys::FitGranularity::kNarrow)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF16NarrowOrdersRatHorner
                              : &boys::BoysCuda::AllOrdersF16NarrowOrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF16NarrowRatHorner
                          : &boys::BoysCuda::AllOrdersF16NarrowRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF16NarrowOrdersMono
                          : &boys::BoysCuda::AllOrdersF16NarrowOrders;
        }

        return horner ? &boys::BoysCuda::AllOrdersF16NarrowMono
                      : &boys::BoysCuda::AllOrdersF16Narrow;
    }

    if (partition == boys::FitGranularity::kUniform)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersF16OrdersUniformRatHorner
                              : &boys::BoysCuda::AllOrdersF16OrdersUniformRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersF16UniformRatHorner
                          : &boys::BoysCuda::AllOrdersF16UniformRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersF16OrdersUniformHorner
                          : &boys::BoysCuda::AllOrdersF16OrdersUniform;
        }

        return horner ? &boys::BoysCuda::AllOrdersF16UniformHorner
                      : &boys::BoysCuda::AllOrdersF16Uniform;
    }

    return nullptr;
}

/// The same map for the half lane's other store: the bfloat16 entries of the same
/// option table, one name per member, read at each entry's own default policy.
///
/// It is the map above with every entry name's \c F16 spelled \c Bf16, which is the
/// relation between the two surfaces themselves (boys_cuda_options.hpp, the bfloat16
/// block: "each of these is the same spelling with F16 -> Bf16"). The lane is one
/// lane and the class is not: the members are the lane's 24 and the entries are this
/// store's, so a cell of this class judged through the map above would be judged on
/// an entry that returns the other format.
///
/// The one collapse in the map is the option table's and not this gate's: the
/// half lane's coarsest partition is one row for both scheme names, its
/// per-argument row and its orders-axis counterpart alike (boys_cuda_options.hpp,
/// "the coarsest partition's row is one row for both scheme names as
/// kAllOrdersBf16 is"), where the double table's same route carries a second row
/// per axis on kScheme (src/boys_cuda.cpp, all-orders-fp64-mono and
/// all-orders-fp64-orders-mono). So both scheme names of this cross reach that
/// one row's entry, and the map returns it for both. Everywhere else the two
/// scheme names are two rows, and the map returns the name its own scheme
/// names; on the rational route those two names are one kernel, the Horner name
/// a forwarder to the other (boys_cuda.hpp, AllOrdersBf16Rat: "both scheme names
/// reach this one entry", and AllOrdersBf16RatHorner: "A forwarder and not a
/// second arithmetic").
///
/// \param route     the member's fit route: the Chebyshev pieces or the
///                  rational pair over the same pieces and intervals
/// \param scheme    the member's summation; where the library stores one
///                  arithmetic under both names the second name reaches it too
/// \param partition the member's partition: the shipped ladder, the narrow
///                  pieces or the uniform grid
/// \param axis      the member's packing axis: per argument or across orders
///
/// \returns the launched entry that serves the member, or \c nullptr for a
///          member this lane's surface has no entry for
constexpr auto GateDeviceEntryBf16(boys::FitRoute route,
                                  boys::EvalScheme scheme,
                                  boys::FitGranularity partition,
                                  boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const boys::Bf16*, boys::Bf16*, std::size_t, void*,
                            boys::DivisionForm) {
    const bool orders = axis == boys::PackAxis::kOrders;
    const bool horner = scheme == boys::EvalScheme::kHorner;
    const bool rational = route == boys::FitRoute::kRationalMinimax;

    if (partition == boys::FitGranularity::kCoarsest)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersBf16OrdersRatHorner
                              : &boys::BoysCuda::AllOrdersBf16OrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersBf16RatHorner
                          : &boys::BoysCuda::AllOrdersBf16Rat;
        }

        return orders ? &boys::BoysCuda::AllOrdersBf16Orders : &boys::BoysCuda::AllOrdersBf16;
    }

    if (partition == boys::FitGranularity::kNarrow)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersBf16NarrowOrdersRatHorner
                              : &boys::BoysCuda::AllOrdersBf16NarrowOrdersRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersBf16NarrowRatHorner
                          : &boys::BoysCuda::AllOrdersBf16NarrowRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersBf16NarrowOrdersMono
                          : &boys::BoysCuda::AllOrdersBf16NarrowOrders;
        }

        return horner ? &boys::BoysCuda::AllOrdersBf16NarrowMono
                      : &boys::BoysCuda::AllOrdersBf16Narrow;
    }

    if (partition == boys::FitGranularity::kUniform)
    {
        if (rational)
        {
            if (orders)
            {
                return horner ? &boys::BoysCuda::AllOrdersBf16OrdersUniformRatHorner
                              : &boys::BoysCuda::AllOrdersBf16OrdersUniformRat;
            }

            return horner ? &boys::BoysCuda::AllOrdersBf16UniformRatHorner
                          : &boys::BoysCuda::AllOrdersBf16UniformRat;
        }

        if (orders)
        {
            return horner ? &boys::BoysCuda::AllOrdersBf16OrdersUniformHorner
                          : &boys::BoysCuda::AllOrdersBf16OrdersUniform;
        }

        return horner ? &boys::BoysCuda::AllOrdersBf16UniformHorner
                      : &boys::BoysCuda::AllOrdersBf16Uniform;
    }

    return nullptr;
}
#endif // BOYS_GATE_FP16

/// The launched entry the fast member of the float device lane is measured through.
///
/// The region-B exponential is not a policy argument on a device lane: the launched
/// entries are distinct functions and the option table states the member each one
/// runs (DeviceOptionInfo::regionBExp, boys/boys_cuda_options.hpp). This revision's
/// table carries the float lane's all-orders family at \c RegionBExp::kAccurate - so
/// the cross's arms above read that member and no other - and carries the lane's
/// \c RegionBExp::kFast at one entry of its own, the single-order launch
/// (`kSingleF32Fast`, whose C++ name is `BoysCuda::SingleF32` at that member). The
/// double lane's table carries no row at \c kFast in either group at this revision,
/// and the arm below names that member rather than judging it at the other member's
/// figure.
///
/// The four parameters are accepted and unused so that this entry is reached through
/// the same call shape as the family above: the fast member's launch reads one ladder
/// of the lane's own, and the axes the cross crosses select the fits the accurate
/// member's family is cut from, so the cell a caller asks about does not move it.
///
/// \param route     the member's fit route, which selects no part of this entry
/// \param scheme    the member's summation, which selects no part of this entry
/// \param partition the member's partition, which selects no part of this entry
/// \param axis      the member's packing axis, which selects no part of this entry
///
/// \returns the launched entry the float lane runs at \c RegionBExp::kFast
constexpr auto GateDeviceEntryFast(boys::FitRoute route,
                                   boys::EvalScheme scheme,
                                   boys::FitGranularity partition,
                                   boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const double*, float*, std::size_t, void*,
                            boys::DivisionForm) {
    (void)route;
    (void)scheme;
    (void)partition;
    (void)axis;

    return &boys::BoysCuda::SingleF32<boys::RegionBExp::kFast>;
}

/// The double lane's launched entry at the other region-B exponential: its own fast
/// member, whose name is the lane's (`BoysCuda::SingleF64Fast`, boys/boys_cuda.hpp,
/// defined in src/boys_cuda.cpp beside the float lane's).
///
/// The four parameters are unused for the reason the float lane's map states above:
/// this member's launch is one arithmetic of the lane and not one per cell.
///
/// \param route     the member's fit route, which selects no part of this entry
/// \param scheme    the member's summation, which selects no part of this entry
/// \param partition the member's partition, which selects no part of this entry
/// \param axis      the member's packing axis, which selects no part of this entry
///
/// \returns the launched entry the double lane runs at \c RegionBExp::kFast
constexpr auto GateDeviceEntryF64Fast(boys::FitRoute route,
                                      boys::EvalScheme scheme,
                                      boys::FitGranularity partition,
                                      boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const double*, double*, std::size_t, void*,
                            boys::DivisionForm) {
    (void)route;
    (void)scheme;
    (void)partition;
    (void)axis;

    return &boys::BoysCuda::SingleF64Fast;
}

#ifdef BOYS_GATE_FP16
/// The half lane's launched entry at the other region-B exponential: its own fast
/// member, whose parameter list takes fp16 arguments and whose name is the lane's
/// (`BoysCuda::SingleF16Fast`, boys/boys_cuda.hpp).
///
/// The four parameters are unused for the reason the float lane's map states above:
/// this member's launch is one arithmetic of the lane and not one per cell.
///
/// \param route     the member's fit route, which selects no part of this entry
/// \param scheme    the member's summation, which selects no part of this entry
/// \param partition the member's partition, which selects no part of this entry
/// \param axis      the member's packing axis, which selects no part of this entry
///
/// \returns the launched entry the half lane runs at \c RegionBExp::kFast
constexpr auto GateDeviceEntryF16Fast(boys::FitRoute route,
                                      boys::EvalScheme scheme,
                                      boys::FitGranularity partition,
                                      boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const boys::F16*, boys::F16*, std::size_t, void*,
                            boys::DivisionForm) {
    (void)route;
    (void)scheme;
    (void)partition;
    (void)axis;

    return &boys::BoysCuda::SingleF16Fast;
}

/// The half lane's other store's launched entry at the other region-B exponential: the
/// same member of this class, whose parameter list takes bfloat16 arguments and whose
/// name is the format's (`BoysCuda::SingleBf16Fast`, boys/boys_cuda.hpp). The lane's
/// table carries it at that member for the reason the fp16 class's does, and the class
/// is the one this arm's cells are judged for.
///
/// The four parameters are unused for the reason the float lane's map states above:
/// this member's launch is one arithmetic of the lane and not one per cell.
///
/// \param route     the member's fit route, which selects no part of this entry
/// \param scheme    the member's summation, which selects no part of this entry
/// \param partition the member's partition, which selects no part of this entry
/// \param axis      the member's packing axis, which selects no part of this entry
///
/// \returns the launched entry the half lane runs at \c RegionBExp::kFast
constexpr auto GateDeviceEntryBf16Fast(boys::FitRoute route,
                                      boys::EvalScheme scheme,
                                      boys::FitGranularity partition,
                                      boys::PackAxis axis) noexcept
    -> boys::BoysStatus (*)(const int*, const boys::Bf16*, boys::Bf16*, std::size_t, void*,
                            boys::DivisionForm) {
    (void)route;
    (void)scheme;
    (void)partition;
    (void)axis;

    return &boys::BoysCuda::SingleBf16Fast;
}
#endif // BOYS_GATE_FP16

/// One device allocation the arm owns for the length of the sweep: allocated
/// on construction, freed on the way out, and reporting what it could not do
/// rather than ending the run - a host whose device refuses the allocation
/// measures no member of the lane, and the cross counts them apart with the
/// reason the arm records.
///
/// The shape is tests/boys_cuda_accuracy_gate.cpp's buffer, which is this same
/// allocation for the same reason and is not shared because the two gates are
/// separate programs.
template <typename T> class GateDeviceBuffer {
public:
    explicit GateDeviceBuffer(std::size_t count) : mCount(count) {
        if (cudaMalloc(reinterpret_cast<void**>(&mPtr), count * sizeof(T)) != cudaSuccess)
        {
            mPtr = nullptr;
        }
    }

    ~GateDeviceBuffer() {
        if (mPtr != nullptr)
        {
            cudaFree(mPtr);
        }
    }

    GateDeviceBuffer(const GateDeviceBuffer&) = delete;
    GateDeviceBuffer& operator=(const GateDeviceBuffer&) = delete;

    T* get() const {
        return mPtr;
    }

    bool ok() const {
        return mPtr != nullptr;
    }

    bool Upload(const std::vector<T>& src) {
        return mPtr != nullptr &&
               cudaMemcpy(mPtr, src.data(), mCount * sizeof(T), cudaMemcpyHostToDevice) ==
                   cudaSuccess;
    }

    bool Download(std::vector<T>& dst) const {
        return mPtr != nullptr &&
               cudaMemcpy(dst.data(), mPtr, mCount * sizeof(T), cudaMemcpyDeviceToHost) ==
                   cudaSuccess;
    }

private:
    T* mPtr = nullptr;
    std::size_t mCount = 0;
};

// The device-callable arm of the fast-member reading, one kernel calling the lane's in-kernel
// entry at RegionBExp::kFast (tests/boys_accuracy_gate_fast_device.cu). Declared here because a
// __device__ entry may only be included by a .cu and this gate's extension is not one.
extern "C" int BoysGateFastDeviceSingleF32(const boys::BoysDeviceTables* tables,
                                           const int* n,
                                           const double* x,
                                           float* out,
                                           std::size_t count,
                                           void* stream,
                                           boys::DivisionForm form);
#endif // BOYS_GATE_CUDA

} // namespace

namespace {

// The combination block's evaluation grid: two region-B members and three division forms per
// cell. At namespace scope because inside the nested template lambdas a variable of the enclosing
// function is a captured reference and not a constant expression (MSVC C2131).
constexpr std::size_t kCombMemberSlots = 2;
constexpr std::size_t kCombFormsPerMember = 3;
constexpr std::size_t kCombEvalReadings = kCombMemberSlots * kCombFormsPerMember;

/// One class's ladder at one argument: F_0(x)..F_nmax(x) in `out`, produced by the entry
/// that answers for the class (kFamily, kShape) and widened to double.
///
/// The combination block's rows are keyed by the lane because the accessor is: the figure
/// BoysAccuracyGuaranteed answers is the lane's. The classes are not keyed that way - each
/// entry answers for the shape its own default policy names - so a row measured through one
/// class's entry is a statement about that class and about no other, and the block measures
/// every class through its own. This is the one place a class's entry is called, so the
/// entry a class's cells came from is the entry named in its `if constexpr` branch, and the
/// name the class book prints is the name called here.
///
/// \tparam kFamily 0 the double lane, 1 the float lane, 2 the fp16 lane, 3 the bf16 lane:
///                 the four host families, one per host lane of BoysLaneContracts(). The
///                 entry, the value type it is handed and returns, and the reference column
///                 its cells are judged against all follow from it.
/// \tparam kShape  the class's shape; the five shapes of the library's own enumeration.
/// \tparam P       the policy, whose packing axis is the row's. A one-order shape's entry
///                 refuses PackAxis::kOrders where it is named, so the caller instantiates
///                 that pair only at the axis the shape carries.
template <int kFamily, boys::Shape kShape, boys::EvalPolicyLike P>
void GateClassLadder(int nmax, double x, double* out) noexcept
{
    if constexpr (kFamily == 0)
    {
        if constexpr (kShape == boys::Shape::kSingle)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                out[n] = boys::BoysSingle<P>(n, x);
            }
        }
        else if constexpr (kShape == boys::Shape::kFixedN)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                double v = 0.0;
                boys::BoysFixedN<P>(n, &x, &v, 1, 1);
                out[n] = v;
            }
        }
        else if constexpr (kShape == boys::Shape::kAllN)
        {
            boys::BoysAllN<P>(nmax, &x, out, 1);
        }
        else if constexpr (kShape == boys::Shape::kAllNAtOrders)
        {
            boys::BoysAllNAtOrders<P>(&nmax, &x, out, 1);
        }
        else
        {
            boys::BoysAllOrders<P>(nmax, x, out);
        }
    }
    else if constexpr (kFamily == 1)
    {
        const float xf = static_cast<float>(x);

        if constexpr (kShape == boys::Shape::kSingle)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                out[n] = static_cast<double>(boys::BoysSingleF32<P>(n, xf));
            }
        }
        else if constexpr (kShape == boys::Shape::kFixedN)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                float v = 0.0f;
                boys::BoysFixedNF32<P>(n, &xf, &v, 1, 1);
                out[n] = static_cast<double>(v);
            }
        }
        else if constexpr (kShape == boys::Shape::kAllN)
        {
            std::array<float, 33> buf{};
            boys::BoysAllNF32<P>(nmax, &xf, buf.data(), 1);

            for (int n = 0; n <= nmax; ++n)
            {
                out[n] = static_cast<double>(buf[static_cast<std::size_t>(n)]);
            }
        }
        else if constexpr (kShape == boys::Shape::kAllNAtOrders)
        {
            std::array<float, 33> buf{};
            boys::BoysAllNAtOrdersF32<P>(&nmax, &xf, buf.data(), 1);

            for (int n = 0; n <= nmax; ++n)
            {
                out[n] = static_cast<double>(buf[static_cast<std::size_t>(n)]);
            }
        }
        else
        {
            std::array<float, 33> buf{};
            boys::BoysAllOrdersF32<P>(nmax, xf, buf.data());

            for (int n = 0; n <= nmax; ++n)
            {
                out[n] = static_cast<double>(buf[static_cast<std::size_t>(n)]);
            }
        }
    }
    else if constexpr (kFamily == 2)
    {
        const boys::F16 xh = boys::F16(static_cast<float>(x));
        std::array<boys::F16, 33> buf{};

        if constexpr (kShape == boys::Shape::kSingle)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                buf[static_cast<std::size_t>(n)] = boys::BoysSingleF16<P>(n, xh);
            }
        }
        else if constexpr (kShape == boys::Shape::kFixedN)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                boys::F16 v = boys::F16(0.0f);
                boys::BoysFixedNF16<P>(n, &xh, &v, 1, 1);
                buf[static_cast<std::size_t>(n)] = v;
            }
        }
        else if constexpr (kShape == boys::Shape::kAllN)
        {
            boys::BoysAllNF16<P>(nmax, &xh, buf.data(), 1);
        }
        else if constexpr (kShape == boys::Shape::kAllNAtOrders)
        {
            boys::BoysAllNAtOrdersF16<P>(&nmax, &xh, buf.data(), 1);
        }
        else
        {
            boys::BoysAllOrdersF16<P>(nmax, xh, buf.data());
        }

        for (int n = 0; n <= nmax; ++n)
        {
            out[n] = static_cast<double>(static_cast<float>(buf[static_cast<std::size_t>(n)]));
        }
    }
    else
    {
        const boys::Bf16 xh = boys::Bf16(static_cast<float>(x));
        std::array<boys::Bf16, 33> buf{};

        if constexpr (kShape == boys::Shape::kSingle)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                buf[static_cast<std::size_t>(n)] = boys::BoysSingleBf16<P>(n, xh);
            }
        }
        else if constexpr (kShape == boys::Shape::kFixedN)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                boys::Bf16 v = boys::Bf16(0.0f);
                boys::BoysFixedNBf16<P>(n, &xh, &v, 1, 1);
                buf[static_cast<std::size_t>(n)] = v;
            }
        }
        else if constexpr (kShape == boys::Shape::kAllN)
        {
            boys::BoysAllNBf16<P>(nmax, &xh, buf.data(), 1);
        }
        else if constexpr (kShape == boys::Shape::kAllNAtOrders)
        {
            boys::BoysAllNAtOrdersBf16<P>(&nmax, &xh, buf.data(), 1);
        }
        else
        {
            boys::BoysAllOrdersBf16<P>(nmax, xh, buf.data());
        }

        for (int n = 0; n <= nmax; ++n)
        {
            out[n] = static_cast<double>(static_cast<float>(buf[static_cast<std::size_t>(n)]));
        }
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string reference = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
    bool perOrder = false;
    bool strict = false;
    bool driftDump = false;
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
        } else if (arg == "--half-drift-dump")
        {
            driftDump = true;
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

    // Claim slots, in report order. The term the float lane's contract row states for the plain
    // reciprocal, read from the library's own row rather than transcribed (README: "<= 1.5e-7, or
    // <= 2.5e-7 in the plain-reciprocal form"); 0.0 where the lane's row states none.
    const double floatPlainTerm = [&] {
        for (const boys::LaneContractInfo& row : boys::BoysLaneContracts())
        {
            if (row.precision == boys::Precision::kFp32)
            {
                return row.plainAdditive;
            }
        }

        return 0.0;
    }();

    // The figure the float lane publishes for one division form: the transcribed base plus the
    // form's own term, so a row judged at one form's figure while dividing in the other - the
    // mistake the granularity book's rows state for their policies - cannot happen.
    const auto floatFigureFor = [&](boys::DivisionForm form) {
        return kBoundFloat +
               (form == boys::DivisionForm::kPlainReciprocal ? floatPlainTerm : 0.0);
    };

    // The form each claim below divides in: BoysSingleF32, BoysAllOrdersF32, BoysAllNF32 and the
    // orders-axis policy name no form at their call site, so each divides in the one its own
    // class's default carries - read from that default, not from kDefaultDivisionForm alone.
    const double floatSingleFigure =
        floatFigureFor(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kSingle>::kDivision);
    const double floatOrdersFigure = floatFigureFor(
        boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>::kDivision);
    const double floatAllNFigure =
        floatFigureFor(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>::kDivision);

    // The entry the orders-axis claim measures: the float ladder over the
    // orders packing axis, named once here because the claim's figure is read
    // off it and the sweep below instantiates it.
    using FloatOrdersAxis = boys::EvalPolicy<boys::FitRoute::kChebyshev,
                                             boys::EvalScheme::kSplitClenshaw,
                                             boys::BoysBudget::kFloat,
                                             boys::PackAxis::kOrders>;

    const int kSingleA = AddClaim("double single", "A", kBoundSingleA);
    const int kSingleBand = AddClaim("double single", "band", kBoundSingleBand);
    const int kSingleB = AddClaim("double single", "B", kBoundSingleB);
    const int kSingleC = AddClaim("double single", "C", kBoundSingleC);
    const int kOrders = AddClaim("double batch", "all-orders", kBoundDoubleBatch);
    const int kFixedN = AddClaim("double batch", "fixed-n", kBoundDoubleBatch);
    const int kAllN = AddClaim("double batch", "all-n", kBoundDoubleBatch);
    const int kAllNAtOrders = AddClaim("double batch", "all-n at-orders", kBoundDoubleBatch);
    const int kFloatSingle = AddClaim("float single", "all", floatSingleFigure);
    const int kFloatOrders = AddClaim("float batch", "all", floatOrdersFigure);
    const int kFloatOrdersPacked =
        AddClaim("float batch", "all, orders axis", floatFigureFor(FloatOrdersAxis::kDivision));
    const int kFloatAllN = AddClaim("float batch", "all-n", floatAllNFigure);
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
    // The withdrawn header claim ("1e-15 across every x < x0"), measured for the record and not
    // judged: the last argument keeps its slots out of the report's verdicts.
    const int kHeaderA = AddClaim("double single", "header x<x0", kWithdrawnHeaderABound, false);
    // The native packed half lane: region C only, and its bound is in ULP of the
    // returned value rather than a region budget, so these slots carry no single
    // base bound (baseBound is display only).
    const int kNativeHalf2 = AddClaim("native packed half", "C", 0.0);
    const int kNativeHalfBatch = AddClaim("native packed half batch", "C", 0.0);
    // The region-A transform lane's modes: one slot per mode, carrying the mode's published
    // bound over both bands, every order and argument. The slots exist in every revision.
    const int kTransformFp64 = AddClaim("transform kFp64", "A", kTransformFp64Bound);
    const int kTransformTf32x3 = AddClaim("transform kTf32x3", "A", kTransformSplitBound);
    const int kTransformBf16x6 = AddClaim("transform kBf16x6", "A", kTransformSplitBound);

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

    // The arithmetic every scalar lane above was computed at: a fused step is an instruction
    // where the target has one and a call into the C runtime where it does not, so a figure read
    // off this output without the route beside it is a figure whose arithmetic is unstated.
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

    // --strict is the default verdict, accepted so a caller can name what they get either way.
    // It scopes over the claims this build carries: a carried row this revision cannot verify
    // fails the gate, while a row whose subject the build does not carry is named with its reason
    // and fails nothing here.
    std::printf("verdict      : %s\n",
                strict ? "strict (--strict: the default verdict, stated) - a claim this build "
                         "carries and cannot verify at this revision fails the gate, while a "
                         "claim whose subject this build does not carry is named with its "
                         "reason and fails nothing here"
                       : "strict (a claim this build carries and does not verify at this "
                         "revision fails the gate; a claim whose subject this build does not "
                         "carry is named with its reason and fails nothing here)");

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
    // The double table's two shared bands as one matrix product per band, in each published mode,
    // measured against this gate's reference at every order and argument. The entry takes one
    // band's arguments and does not sort them, so the caller groups them here.
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

    // ---- double batch: the per-element top order entry ---------------------
    // Two patterns: the ragged one gives each argument its own top order, so the columns stop at
    // different depths and the planes above a column are the ones the entry must leave alone; the
    // uniform one gives every argument the grid's nmax, which puts the entry on the all-n cells.
    {
        std::vector<int> tops(count);
        std::vector<double> out(count * static_cast<std::size_t>(nmax + 1));

        for (std::size_t i = 0; i < count; ++i)
        {
            tops[i] = nmax - static_cast<int>(i % static_cast<std::size_t>(nmax + 1));
        }

        boys::BoysAllNAtOrders(tops.data(), ref.x.data(), out.data(), count);

        for (std::size_t i = 0; i < count; ++i)
        {
            for (int n = 0; n <= tops[i]; ++n)
            {
                const double got = out[ref.Index(n, i)];
                Measure(kAllNAtOrders,
                        n,
                        ref.x[i],
                        got,
                        ref.v[ref.Index(n, i)],
                        ref.decade[ref.Index(n, i)],
                        kBoundDoubleBatch,
                        Unrepresentable(got, -1022));
            }
        }

        std::fill(tops.begin(), tops.end(), nmax);
        boys::BoysAllNAtOrders(tops.data(), ref.x.data(), out.data(), count);

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double got = out[ref.Index(n, i)];
                Measure(kAllNAtOrders,
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
    // A route is a way of serving a region, not a lane the library always serves, so it is
    // measured as a lane is and counted apart: each route's own fit over its row's interval, the
    // selector's values over the region, and that naming a route changes nothing outside it.
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
    double routeShortfallWorst = 0.0;
    std::size_t routeNames = 0;
    std::size_t routeCellsInside = 0;
    std::size_t routeDiffOutside = 0;
    std::size_t routeDiffInside = 0;
    std::size_t routeDefaultDiff = 0;
    std::size_t routeUnknownDiff = 0;

    // The class the unnamed calls of this section are made through: the combination this build's
    // seam gives BoysAllOrders, which is what a call naming no policy compiles (boys/boys.hpp,
    // DefaultPolicy).
    using RouteClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;

    // The domain the class's entries read, and the one a route's selector takes over in this
    // build. A class row naming the grid reads one fixed table whose domain is its own row of
    // BoysFitGranularities: the grid covers [0, kFlatHi) whole and the body answers the whole of
    // it from the table before the region tests (boys_impl.hpp, AllOrdersBody).
    const boys::FitGranularityInfo* routePartition = nullptr;

    for (const boys::FitGranularityInfo& partition : boys::BoysFitGranularities())
    {
        if (partition.granularity == RouteClass::kGranularity)
        {
            routePartition = &partition;
        }
    }

    const bool routeReadsGrid = routePartition != nullptr &&
                                routePartition->granularity == boys::FitGranularity::kUniform;

    // Filled by the carriage measurement below and read by the route book's
    // claims: the run-time selector's pairs, and the entries that name a route
    // without answering it.
    std::size_t routeRuntimePairs = 0;
    std::size_t routeRuntimeDiff = 0;
    std::size_t routeCarriageMissed = 0;
    std::size_t routeCarriageNamed = 0;
    std::size_t routeCarriageControls = 0;

    // The route held to its own bar through the entries a consumer actually calls: the route book
    // measures the per-argument entries, so the figure for these two is measured here.
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

    std::size_t routeCells = 0;

    {
        std::array<double, 33> out{};
        std::array<double, 33> plain{};
        std::array<double, 33> selected{};

        for (std::size_t i = 0; i < count; ++i)
        {
            const double x = ref.x[i];
            bool covered = false;
            bool served =
                routeReadsGrid && x >= routePartition->lo && x < routePartition->hi;

            for (const boys::FitRouteInfo& row : boys::BoysFitRoutes())
            {
                if (x >= row.lo && x < row.hi)
                {
                    covered = true;
                }

                // The domain a route's selector takes over, where naming it may change a value.
                // A row whose fit reaches further than its selector claims the narrower interval;
                // where the class row reads the grid, the partition's own cover is the domain.
                if (!routeReadsGrid && x >= std::max(row.lo, row.servesFrom) && x < row.hi)
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
                    // Region A's fit is one piece per order rather than a seed, so the route's
                    // fit is measured piece by piece. Both readings are one body at two fit
                    // policies, so what this row compares is the two fits.
                    for (int n = 0; n <= nmax; ++n)
                    {
                        const double got =
                            (row.route == boys::FitRoute::kRationalMinimax)
                                ? boys::detail::RegionAValue<boys::detail::RationalFit>(n, x)
                                : boys::detail::RegionAValue<boys::detail::ChebyshevFit<
                                      boys::kDefaultEvalScheme,
                                      boys::kDefaultFitGranularity>>(n, x);
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
        // The accuracy rows cannot show that the route was read: both routes hold the same bar over
        // the same intervals, so an entry evaluating the other route's fits would pass every one.
        // Measure instead whether naming it changes any value; the default route is the control.
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
                        const double a = boys::BoysSingle< boys::EvalPolicy<kRoute>>(n, ref.x[i]);
                        const double b = boys::BoysSingle< boys::EvalPolicy<>>(n, ref.x[i]);

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
                    boys::BoysAllOrders< boys::EvalPolicy<kRoute>>(nmax, ref.x[i], a.data());
                    boys::BoysAllOrders< boys::EvalPolicy<>>(nmax, ref.x[i], b.data());

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

            // The many-argument and fixed-order entries carry the route too, on the shapes that
            // take their fit from the policy; measured the same way - a difference in the values
            // and not a sentence about the surface.
            const auto countPlane = [&]<boys::FitRoute kRoute>() {
                std::size_t cells = 0;
                std::size_t differ = 0;
                const std::size_t planeSize =
                    permArgs.size() * (static_cast<std::size_t>(nmax) + 1);
                std::vector<double> a(planeSize);
                std::vector<double> b(planeSize);
                boys::BoysAllN< boys::EvalPolicy<kRoute>>(
                    nmax, permArgs.data(), a.data(), permArgs.size());
                boys::BoysAllN< boys::EvalPolicy<>>(
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
                        boys::BoysFixedN< boys::EvalPolicy<kRoute>>(n, &ref.x[i], &a, 1);
                        boys::BoysFixedN< boys::EvalPolicy<>>(n, &ref.x[i], &b, 1);

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

        // ---- the route's bar through the entries that carry it: the route's own bar for the
        // region the argument falls in, read off BoysFitRoutes so a moved bar moves this row with
        // it. Both entries are swept over the whole committed grid.
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
            boys::BoysAllN< boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
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
                boys::BoysFixedN< boys::EvalPolicy<boys::FitRoute::kRationalMinimax>>(
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

        // ---- the run-time selector's two-argument form: route and scheme named together, the
        // pair a compile-time call site reaches through one EvalPolicy. The values must be the same
        // values, not merely inside the same bound.
        const auto runtimePair = [&]<boys::FitRoute kRoute, boys::EvalScheme kScheme>() {
            // The entry a call site writes for this pair: the policy names the route and the
            // scheme and leaves the other axes to the class, which is the reading the selector
            // itself makes for the axes its caller did not name (src/boys.cpp, SelectorPolicy).
            using Counterpart = boys::EvalPolicy<kRoute,
                                                 kScheme,
                                                 RouteClass::kBudget,
                                                 RouteClass::kPack,
                                                 RouteClass::kGranularity,
                                                 RouteClass::kDivision,
                                                 RouteClass::kRegionBExp>;

            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> a{};
                std::array<double, 33> b{};
                boys::BoysAllOrdersWithRoute(kRoute, kScheme, nmax, ref.x[i], a.data());
                boys::BoysAllOrders<Counterpart>(nmax, ref.x[i], b.data());

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

        // The region-A table's own bookkeeping, read the way the kernel reads it: each piece's
        // coefficients sit at its offset, and the offsets tile the coefficient array without a gap
        // or an overlap.
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

            // The reported figure is a sweep on the generator's grid and this one is a sweep on
            // the committed reference grid, which is coarser: a fit that equioscillates is not
            // sampled at its own extrema from here, so the reading is one-sided - the gate can
            // find the fit worse than its row reports, never better.
            const double measured =
                RouteClaims()[static_cast<std::size_t>(routeSeedClaim[r])].worstErr;

            if (measured > row.delivered + 0.1 * row.bound)
            {
                ++routeDisagreement;
            }

            // Printed rather than only tested, because the direction is not
            // promised: a fit that equioscillates is not sampled at its own
            // extrema from this grid, so the measured figure can sit above the
            // reported one and still be inside the tolerance.
            const double shortfall = (measured - row.delivered) / row.bound;

            if (shortfall > routeShortfallWorst)
            {
                routeShortfallWorst = shortfall;
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
                        floatSingleFigure,
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
                        floatOrdersFigure,
                        out[static_cast<std::size_t>(n)] == 0.0f ||
                            std::fabs(asDouble) < std::numeric_limits<float>::min());
            }
        }

        // The same cells again through the entry with the orders packing axis named. The bound
        // and the reference are the same, because a packing axis chooses which lane evaluates and
        // not which fit.
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                boys::BoysAllOrdersF32<FloatOrdersAxis>(nmax,
                                                        static_cast<float>(ref.x[i]),
                                                        out.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double asDouble = static_cast<double>(out[static_cast<std::size_t>(n)]);
                    Measure(kFloatOrdersPacked,
                            n,
                            ref.xf[i],
                            asDouble,
                            ref.vf[k],
                            ref.decadeF[k],
                            floatFigureFor(FloatOrdersAxis::kDivision),
                            out[static_cast<std::size_t>(n)] == 0.0f ||
                                std::fabs(asDouble) < std::numeric_limits<float>::min());
                }
            }
        }

        // The float lane's all-N batch: the same cells as the all-orders entry
        // above, through the entry that carries the array shape on this lane.
        std::vector<float> argsF(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            argsF[i] = static_cast<float>(ref.xf[i]);
        }

        std::vector<float> allN(count * static_cast<std::size_t>(nmax + 1));
        boys::BoysAllNF32(nmax, argsF.data(), allN.data(), count);

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const std::size_t k = ref.Index(n, i);
                const double asDouble = static_cast<double>(allN[k]);
                Measure(kFloatAllN,
                        n,
                        ref.xf[i],
                        asDouble,
                        ref.vf[k],
                        ref.decadeF[k],
                        floatAllNFigure,
                        allN[k] == 0.0f || std::fabs(asDouble) < std::numeric_limits<float>::min());
            }
        }
    }

    // ---- the float lane's fit routes --------------------------------------
    // The float lane's region-A table is its own, so its route rows state what
    // BoysSingleF32WithRoute delivers over the interval each row names, carry their own lane label
    // and are counted apart for the reason the double lane's are.
    std::vector<int> routeSeedClaimF32;

    for (const boys::FitRouteInfo& row : boys::BoysFitRoutesF32())
    {
        routeSeedClaimF32.push_back(
            AddRouteClaim(RouteLaneF32(row.route, row.region), "rows", row.bound));
    }

    std::size_t routeStoredMismatchF32 = 0;
    std::size_t routeDiffOutsideF32 = 0;
    std::size_t routeDiffInsideF32 = 0;
    std::size_t routeDefaultDiffF32 = 0;
    std::size_t routeNonPositiveF32 = 0;
    std::size_t f32RouteCells = 0;
    std::size_t f32RouteOver = 0;
    double f32RouteWorst = 0.0;
    int f32RouteWorstN = -1;
    double f32RouteWorstX = 0.0;
    Verdict f32DeliveredVerdict = Verdict::Verified;
    Verdict f32ReportVerdict = Verdict::Verified;
    std::size_t f32RouteShortOver = 0;
    double f32RouteShortWorst = 0.0;

    {
        const std::span<const boys::FitRouteInfo> rows = boys::BoysFitRoutesF32();

        for (std::size_t r = 0; r < rows.size(); ++r)
        {
            const boys::FitRouteInfo& row = rows[r];
            const auto lo = static_cast<float>(row.lo);
            const auto hi = static_cast<float>(row.hi);

            // The row's stored count against the table the kernel reads. A row
            // that undercounts its route is the one way the report can promise
            // less than the entry costs.
            int stored = 0;

            if (row.region == boys::AccuracyRegion::kA)
            {
                const bool cheb = (row.route == boys::FitRoute::kChebyshev);
                const int first =
                    cheb ? boys::detail::f32::kPieceStart[0] : boys::detail::f32::kRatAPieceStart[0];
                const int last = cheb
                                     ? boys::detail::f32::kPieceStart[boys::kMaxBoysOrder + 1]
                                     : boys::detail::f32::kRatAPieceStart[boys::kMaxBoysOrder + 1];

                for (int i = first; i < last; ++i)
                {
                    const std::size_t pi = static_cast<std::size_t>(i);
                    stored += cheb ? boys::detail::f32::kPieces[pi].deg + 1
                                   : boys::detail::f32::kRatAPieces[pi].numdeg +
                                         boys::detail::f32::kRatAPieces[pi].dendeg + 1;
                }
            } else
            {
                stored = row.route == boys::FitRoute::kChebyshev
                             ? boys::detail::f32::kBDeg + 1
                             : boys::detail::f32::kRatBnumDeg + boys::detail::f32::kRatBdenDeg + 1;
            }

            if (stored != row.stored)
            {
                ++routeStoredMismatchF32;
            }

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const float xf = static_cast<float>(ref.xf[i]);

                    if (!(xf >= lo && xf < hi))
                    {
                        continue;
                    }

                    const std::size_t k = ref.Index(n, i);
                    const float got = boys::BoysSingleF32WithRoute(
                        row.route, n, xf);
                    const double asDouble = static_cast<double>(got);
                    const bool unrepresentable =
                        got == 0.0f || std::fabs(asDouble) < std::numeric_limits<float>::min();

                    MeasureInto(RouteClaims(),
                                routeSeedClaimF32[r],
                                n,
                                static_cast<double>(xf),
                                asDouble,
                                ref.vf[k],
                                ref.decadeF[k],
                                row.bound,
                                unrepresentable);

                    if (unrepresentable)
                    {
                        ++routeNonPositiveF32;
                    }
                }
            }
        }

        // What naming a route changes, cell by cell: the default route, a value outside the
        // enumeration and the Chebyshev route must all be the default entry's value bit for bit,
        // and the rational route may differ only where a row of BoysFitRoutesF32 says its selector
        // takes over.
        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const float xf = static_cast<float>(ref.xf[i]);
                const float plain = boys::BoysSingleF32(n, xf);
                const float namedDefault =
                    boys::BoysSingleF32WithRoute(boys::FitRoute::kChebyshev, n, xf);
                const float namedUnknown = boys::BoysSingleF32WithRoute(
                    static_cast<boys::FitRoute>(99), n, xf);
                const float namedRational =
                    boys::BoysSingleF32WithRoute(boys::FitRoute::kRationalMinimax, n, xf);

                if (namedDefault != plain || namedUnknown != plain)
                {
                    ++routeDefaultDiffF32;
                }

                bool inside = false;

                for (const boys::FitRouteInfo& row : rows)
                {
                    if (row.route == boys::FitRoute::kRationalMinimax &&
                        xf >= static_cast<float>(row.lo) && xf < static_cast<float>(row.hi))
                    {
                        inside = true;
                    }
                }

                if (namedRational == plain)
                {
                    continue;
                }

                if (inside)
                {
                    ++routeDiffInsideF32;
                } else
                {
                    ++routeDiffOutsideF32;
                }
            }
        }
    }

    {
        for (std::size_t r = 0; r < routeSeedClaimF32.size(); ++r)
        {
            const Accum& a = RouteClaims()[static_cast<std::size_t>(routeSeedClaimF32[r])];
            f32RouteCells += a.points;
            f32RouteOver += a.failures;

            if (a.worstErr > f32RouteWorst)
            {
                f32RouteWorst = a.worstErr;
                f32RouteWorstN = a.worstN;
                f32RouteWorstX = a.worstX;
            }

            if (a.points == 0)
            {
                f32DeliveredVerdict = Verdict::EvidenceAbsent;
            }

            if (VerdictRank(FromAccum(a)) > VerdictRank(f32DeliveredVerdict))
            {
                f32DeliveredVerdict = FromAccum(a);
            }

            // The same one-sided comparison the double lane's route book is
            // held to, read on this book's rows so that the figure a consumer
            // reads beside a float row is checked against the fit that row
            // serves.
            const boys::FitRouteInfo& f32row = boys::BoysFitRoutesF32()[r];
            const double shortfall = (a.worstErr - f32row.delivered) / f32row.bound;

            if (shortfall > f32RouteShortWorst)
            {
                f32RouteShortWorst = shortfall;
            }

            if (a.worstErr > f32row.delivered + 0.1 * f32row.bound)
            {
                ++f32RouteShortOver;
            }
        }

        if (routeStoredMismatchF32 != 0 || routeDefaultDiffF32 != 0)
        {
            f32ReportVerdict = Verdict::Exceeded;
        }

        std::printf("\n  the float lane's fit routes, BoysFitRoutesF32 and\n"
                    "  BoysSingleF32WithRoute, each row over the interval it names, at every\n"
                    "  order, against the reference this gate uses (%zu rows, %zu cells,\n"
                    "  %zu of them outside the row's bar):\n",
                    routeSeedClaimF32.size(),
                    f32RouteCells,
                    f32RouteOver);
        std::printf("  %-24s %-9s %-6s %7s %14s %14s %14s %4s %13s\n",
                    "route",
                    "stored",
                    "scope",
                    "cells",
                    "reported",
                    "measured",
                    "bar",
                    "n",
                    "at x");

        for (std::size_t r = 0; r < routeSeedClaimF32.size(); ++r)
        {
            const boys::FitRouteInfo& row = boys::BoysFitRoutesF32()[r];
            const Accum& a = RouteClaims()[static_cast<std::size_t>(routeSeedClaimF32[r])];

            std::printf("  %-24s %-9d %-6s %7zu %14.6g %14.6g %14.6g %4d %13.6g\n",
                        RouteLaneF32(row.route, row.region),
                        row.stored,
                        (row.region == boys::AccuracyRegion::kA) ? "0..32" : "F0",
                        a.points,
                        row.delivered,
                        a.worstErr,
                        row.bound,
                        a.worstN,
                        a.worstX);
        }

        std::printf("  rows whose stored count disagrees with the table the kernel reads: "
                    "%zu\n"
                    "  cells where naming the default route or a route outside the enumeration\n"
                    "  differed from the default entry: %zu\n"
                    "  cells outside a rational row's interval that naming the rational route\n"
                    "  changed: %zu; inside one: %zu\n"
                    "  cells of the routes' sweep where the value is at or below the format's\n"
                    "  own floor: %zu\n",
                    routeStoredMismatchF32,
                    routeDefaultDiffF32,
                    routeDiffOutsideF32,
                    routeDiffInsideF32,
                    routeNonPositiveF32);

    }

    // ---- the float lane's route and scheme combinations --------------------
    // The lane stores both forms of its Chebyshev family and one of the rational family, so every
    // (route, scheme) pair the double lane accepts is reachable on its own engines. A row is one
    // (entry, policy, region): its error, and its count of cells differing from the shipped pair's.
    std::size_t f32PolicyCells = 0;
    std::size_t f32PolicyOver = 0;
    std::size_t f32PolicyUncovered = 0;
    std::size_t f32PolicyNotCarried = 0;
    std::size_t f32PolicySeedCells[4] = {0, 0, 0, 0};
    std::size_t f32PolicySeedDiffer[4] = {0, 0, 0, 0};
    std::size_t f32PolicyShortOver = 0;
    std::size_t f32PolicyFloor = 0;
    double f32PolicyShortWorst = 0.0;
    // The worst cell of each entry, kept apart for the same reason its verdict
    // is: the two claims are about different things and neither's worst cell
    // is the other's evidence. Held as a fraction of the worst cell's own bar,
    // because the worst cell of an entry can lie in either region.
    double f32PolicyWorstRatio[2] = {0.0, 0.0};
    double f32PolicyWorstErr[2] = {0.0, 0.0};
    double f32PolicyWorstBar[2] = {0.0, 0.0};
    int f32PolicyWorstN[2] = {-1, -1};
    double f32PolicyWorstX[2] = {0.0, 0.0};
    // Judged per entry: the single-order entry's rows are values of a fit and
    // the batch entry's are a recursion's, so the two are claims about
    // different things and a row of one must not decide the other.
    Verdict f32PolicyVerdict[2] = {Verdict::Verified, Verdict::Verified};

    {
        // Every axis is named, the reference pair included, so each other policy is that pair
        // with one axis moved. THE DIVISION FORM IS NAMED AS THE BUILD'S OWN: the lane publishes
        // 2.5e-7 for the plain reciprocal against 1.5e-7 for the others, and the bar beneath is
        // read from this form, so a tuned configure's move of that default is measured here.
        constexpr boys::DivisionForm kPolicyForm = boys::kDefaultDivisionForm;

        using ShippedPair =
            boys::EvalPolicy<boys::FitRoute::kChebyshev,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kCoarsest,
                             kPolicyForm>;
        using HornerPair =
            boys::EvalPolicy<boys::FitRoute::kChebyshev,
                             boys::EvalScheme::kHorner,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kCoarsest,
                             kPolicyForm>;
        using RationalPair =
            boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kCoarsest,
                             kPolicyForm>;
        // The same two fits at the narrow partition of region A and its own
        // region-B seed: the partition is the only difference from the pairs
        // above, and naming it is the whole of that axis.
        using NarrowPair =
            boys::EvalPolicy<boys::FitRoute::kChebyshev,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kNarrow,
                             kPolicyForm>;
        using NarrowRationalPair =
            boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kNarrow,
                             kPolicyForm>;

        constexpr int kEntries = 2;
        constexpr int kPolicies = 5;
        constexpr int kRegions = 2;
        constexpr int kSingleEntry = 0;
        constexpr int kBatchEntry = 1;

        // One slot per (entry, policy, region), so a row can be printed from
        // the accumulator that measured it.
        const auto slot = [](int entry, int policy, int region) {
            return (entry * kPolicies + policy) * kRegions + region;
        };
        const char* const entryName[kEntries] = {"single", "batch"};
        const char* const policyName[kPolicies] = {"chebyshev x Clenshaw",
                                                   "chebyshev x Horner",
                                                   "rational minimax",
                                                   "chebyshev x Clenshaw x narrow",
                                                   "rational minimax x narrow"};
        const char* const regionName[kRegions] = {"A", "B"};
        // The bar a row is judged by: the region's bar for the two base forms plus the contract
        // term the lane states for the plain reciprocal (2.5e-7 against 1.5e-7 on this lane), read
        // from kPolicyForm so a row's arithmetic and its figure cannot name two forms.
        const double formAdd =
            kPolicyForm == boys::DivisionForm::kPlainReciprocal ? floatPlainTerm : 0.0;
        const double bar[kRegions] = {boys::detail::f32::kRegionAFitBar + formAdd,
                                      boys::detail::f32::kRegionBFitBar + formAdd};
        // The narrow partition's rows carry a figure per multiply-add route, and this build
        // evaluates in one of them: the row is read against that route's own figure rather than
        // the worse of the two.
        const auto narrowPick = [](double fused, double separate) {
            return boys::backend::detail::kSelectedRoute == boys::backend::MulAddRoute::kFused
                       ? fused
                       : separate;
        };
        // The figure the generated header publishes for the fit each policy
        // reads, per policy and region. It is a figure for the single-order
        // entry, whose value is that fit; this is where a consumer's copy of it
        // meets a measurement.
        const double reported[kPolicies][kRegions] = {
            {boys::detail::f32::kRegionAFitChebDelivered,
             boys::detail::f32::kRegionBFitChebDelivered},
            {boys::detail::f32::kRegionAFitChebDeliveredHorner,
             boys::detail::f32::kRegionBFitChebDeliveredHorner},
            {boys::detail::f32::kRegionAFitRatDelivered,
             boys::detail::f32::kRegionBFitRatDelivered},
            {narrowPick(boys::detail::f32::kNarrowARowsF32[0].fused,
                        boys::detail::f32::kNarrowARowsF32[0].separate),
             narrowPick(boys::detail::f32::kNarrowBRowsF32[0].fused,
                        boys::detail::f32::kNarrowBRowsF32[0].separate)},
            {narrowPick(boys::detail::f32::kNarrowRatADeliveredFusedF32,
                        boys::detail::f32::kNarrowRatADeliveredSeparateF32),
             narrowPick(boys::detail::f32::kNarrowRatBDeliveredFusedF32,
                        boys::detail::f32::kNarrowRatBDeliveredSeparateF32)}};

        for (int e = 0; e < kEntries; ++e)
        {
            for (int p = 0; p < kPolicies; ++p)
            {
                for (int r = 0; r < kRegions; ++r)
                {
                    AddF32PolicyClaim(entryName[e], regionName[r], bar[r]);
                }
            }
        }

        std::vector<std::size_t> differ(kEntries * kPolicies * kRegions, 0);

        // The interval each region's cells lie in, read on the float argument
        // the lane evaluates at rather than on the double it came from.
        const auto inRegion = [](int region, float xf) {
            const auto x0 = static_cast<float>(boys::detail::kX0);
            const auto x1 = static_cast<float>(boys::detail::kX1);

            return region == 0 ? (xf < x0) : (xf >= x0 && xf < x1);
        };

        // One cell, three policies, measured in the same pass: the shipped
        // pair's value is on hand to compare the other two against, and the
        // comparison is bitwise because the question a carriage count answers
        // is whether the same call produced a different value at all.
        const auto sweepCell = [&](int entry,
                                   int n,
                                   float xf,
                                   std::size_t k,
                                   int region,
                                   const float got[kPolicies]) {
            for (int p = 0; p < kPolicies; ++p)
            {
                const double asDouble = static_cast<double>(got[p]);
                const bool unrepresentable =
                    got[p] == 0.0f ||
                    std::fabs(asDouble) < std::numeric_limits<float>::min();

                MeasureInto(F32PolicyClaims(),
                            slot(entry, p, region),
                            n,
                            static_cast<double>(xf),
                            asDouble,
                            ref.vf[k],
                            ref.decadeF[k],
                            bar[region],
                            unrepresentable);

                if (unrepresentable)
                {
                    ++f32PolicyFloor;
                }

                if (p > 0 && std::memcmp(&got[p], &got[0], sizeof(float)) != 0)
                {
                    ++differ[static_cast<std::size_t>(slot(entry, p, region))];
                }
            }
        };

        // The single-order entry: one call per cell per policy.
        for (int r = 0; r < kRegions; ++r)
        {
            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    const float xf = static_cast<float>(ref.xf[i]);

                    if (!inRegion(r, xf))
                    {
                        continue;
                    }

                    const float got[kPolicies] = {
                        boys::BoysSingleF32< ShippedPair>(n, xf),
                        boys::BoysSingleF32< HornerPair>(n, xf),
                        boys::BoysSingleF32< RationalPair>(n, xf),
                        boys::BoysSingleF32< NarrowPair>(n, xf),
                        boys::BoysSingleF32< NarrowRationalPair>(n, xf)};

                    sweepCell(kSingleEntry, n, xf, ref.Index(n, i), r, got);
                }
            }
        }

        // The batch entry: one call per argument produces every order, and the seed is what the
        // bar is asked of here - region A's from the double lane's fit at the policy's route and
        // scheme, region B's from this lane's.
        for (int r = 0; r < kRegions; ++r)
        {
            std::array<float, 33> shipped{};
            std::array<float, 33> horner{};
            std::array<float, 33> rational{};
            std::array<float, 33> narrow{};
            std::array<float, 33> narrowRational{};

            for (std::size_t i = 0; i < count; ++i)
            {
                const float xf = static_cast<float>(ref.xf[i]);

                if (!inRegion(r, xf))
                {
                    continue;
                }

                boys::BoysAllOrdersF32< ShippedPair>(nmax, xf, shipped.data());
                boys::BoysAllOrdersF32< HornerPair>(nmax, xf, horner.data());
                boys::BoysAllOrdersF32< RationalPair>(nmax, xf, rational.data());
                boys::BoysAllOrdersF32< NarrowPair>(nmax, xf, narrow.data());
                boys::BoysAllOrdersF32< NarrowRationalPair>(nmax, xf,
                                                                narrowRational.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const float got[kPolicies] = {shipped[static_cast<std::size_t>(n)],
                                                  horner[static_cast<std::size_t>(n)],
                                                  rational[static_cast<std::size_t>(n)],
                                                  narrow[static_cast<std::size_t>(n)],
                                                  narrowRational[static_cast<std::size_t>(n)]};

                    sweepCell(kBatchEntry, n, xf, ref.Index(n, i), r, got);
                }
            }
        }

        // Totalled, judged and printed. The single-order rows are the ones a fit's published
        // figure can be held to, so the route book's one-sided comparison is made here too: a
        // measured figure further above a row's own reported figure than a tenth of the row's bar
        // is that figure to correct, not a bound the row misses.
        for (int e = 0; e < kEntries; ++e)
        {
            std::printf("\n  the float lane's %s entry at each policy it accepts, every order,\n"
                        "  over the interval the policy's fits serve, against the reference this\n"
                        "  gate uses:\n",
                        entryName[e]);

            if (e == kSingleEntry)
            {
                std::printf("  reported is the figure include/boys/boys_coefficients.hpp publishes\n"
                            "  for the fit the policy reads in that region, and the row is the\n"
                            "  measurement that figure is read against.\n");
                std::printf("  %-20s %-5s %7s %14s %14s %14s %8s %4s %13s %8s\n",
                            "policy",
                            "reg",
                            "cells",
                            "reported",
                            "measured",
                            "bar",
                            "of bar",
                            "n",
                            "at x",
                            "differs");
            } else
            {
                std::printf("  The %s entry's value at order n is its recursion's and not a single\n"
                            "  fit's, so no fit's published figure is judged here: the bar column\n"
                            "  is what the row is held to. The differs column is the count of the\n"
                            "  row's cells where this policy's value is not the shipped pair's.\n",
                            entryName[e]);
                std::printf("  %-20s %-5s %7s %14s %14s %8s %4s %13s %8s\n",
                            "policy",
                            "reg",
                            "cells",
                            "measured",
                            "bar",
                            "of bar",
                            "n",
                            "at x",
                            "differs");
            }

            for (int p = 0; p < kPolicies; ++p)
            {
                for (int r = 0; r < kRegions; ++r)
                {
                    const std::size_t s = static_cast<std::size_t>(slot(e, p, r));
                    const Accum& a = F32PolicyClaims()[s];
                    f32PolicyCells += a.points;
                    f32PolicyOver += a.failures;

                    if (a.points == 0)
                    {
                        ++f32PolicyUncovered;
                    }

                    if (VerdictRank(FromAccum(a)) > VerdictRank(f32PolicyVerdict[e]))
                    {
                        f32PolicyVerdict[e] = FromAccum(a);
                    }

                    if (a.points > 0 && a.worstRatio > f32PolicyWorstRatio[e])
                    {
                        f32PolicyWorstRatio[e] = a.worstRatio;
                        f32PolicyWorstErr[e] = a.worstErr;
                        f32PolicyWorstBar[e] = a.worstBound;
                        f32PolicyWorstN[e] = a.worstN;
                        f32PolicyWorstX[e] = a.worstX;
                    }

                    if (p > 0 && e == kBatchEntry && r == 0)
                    {
                        f32PolicySeedCells[p - 1] += a.points;
                        f32PolicySeedDiffer[p - 1] += differ[s];
                    } else if (p > 0 && differ[s] == 0)
                    {
                        ++f32PolicyNotCarried;
                    }

                    if (e == kSingleEntry)
                    {
                        const double shortfall = (a.worstErr - reported[p][r]) / bar[r];

                        if (shortfall > f32PolicyShortWorst)
                        {
                            f32PolicyShortWorst = shortfall;
                        }

                        if (a.worstErr > reported[p][r] + 0.1 * bar[r])
                        {
                            ++f32PolicyShortOver;
                        }

                        std::printf("  %-20s %-5s %7zu %14.6g %14.6g %14.6g %8.3f",
                                    policyName[p],
                                    regionName[r],
                                    a.points,
                                    reported[p][r],
                                    a.worstErr,
                                    bar[r],
                                    a.worstErr / bar[r]);
                    } else
                    {
                        std::printf(
                            "  %-20s %-5s %7zu %14.6g %14.6g %8.3f",
                            policyName[p],
                            regionName[r],
                            a.points,
                            a.worstErr,
                            bar[r],
                            a.worstErr / bar[r]);
                    }

                    std::printf(" %4d %13.6g", a.worstN, a.worstX);

                    if (p == 0)
                    {
                        std::printf(" %8s\n", "n/a");
                    } else
                    {
                        std::printf(" %8zu\n", differ[s]);
                    }
                }
            }
        }

        std::printf("  the float lane's policy rows: %zu cell(s), %zu of them outside the row's bar,\n"
                    "  %zu row(s) measured over no argument at all, %zu cell(s) at or below the\n"
                    "  format's own floor\n",
                    f32PolicyCells,
                    f32PolicyOver,
                    f32PolicyUncovered,
                    f32PolicyFloor);
        std::printf("  rows of a policy other than the shipped pair whose value never differed from\n"
                    "  the shipped pair's on any cell they cover: %zu of the 12 rows the count is\n"
                    "  required for. The batch entry's region-A rows are counted apart, because\n"
                    "  they seed from the double lane's fit at the policy's route, scheme and\n"
                    "  partition rather than from a table this lane stores: the scheme's row covers\n"
                    "  %zu cell(s) and %zu of them differ, the route's row %zu and %zu, the narrow\n"
                    "  partition's %zu and %zu, and the narrow partition's rational route %zu and "
                    "%zu\n",
                    f32PolicyNotCarried,
                    f32PolicySeedCells[0],
                    f32PolicySeedDiffer[0],
                    f32PolicySeedCells[1],
                    f32PolicySeedDiffer[1],
                    f32PolicySeedCells[2],
                    f32PolicySeedDiffer[2],
                    f32PolicySeedCells[3],
                    f32PolicySeedDiffer[3]);
        std::printf("  single-order rows whose published figure is short of the measured one by more\n"
                    "  than a tenth of the bar: %zu of 10, the worst by %.3g of the bar\n",
                    f32PolicyShortOver,
                    f32PolicyShortWorst);
    }

    // ---- the narrow partition's rational route -----------------------------
    // A table of its own: one numerator/denominator pair per narrow piece in each region, read at
    // that piece's own interval and mapped argument. The rows are held to the same bars the route
    // book holds the shipped partition's rows to, and the pieces are measured directly.
    std::size_t narrowRatCells = 0;
    std::size_t narrowRatDiffers = 0;
    double narrowRatWorst = 0.0;
    int narrowRatClaims[3] = {-1, -1, -1};
    {
        using NarrowRational =
            boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kNarrow>;
        using ShippedRational =
            boys::EvalPolicy<boys::FitRoute::kRationalMinimax,
                             boys::EvalScheme::kSplitClenshaw,
                             boys::BoysBudget::kFloat,
                             boys::PackAxis::kArguments,
                             boys::FitGranularity::kCoarsest>;
        using NarrowFit = boys::detail::RouteFit<boys::FitRoute::kRationalMinimax,
                                                 boys::EvalScheme::kSplitClenshaw,
                                                 boys::FitGranularity::kNarrow>::Type;

        const int claimA = AddGranularityClaim("rational x narrow",
                                               "region A pieces",
                                               boys::detail::kRegionAFitBar);
        const int claimB = AddGranularityClaim("rational x narrow",
                                               "region B seed",
                                               boys::detail::kRegionBFitBar);
        const int claimEntry = AddGranularityClaim("rational x narrow",
                                                   "batch entry, A..C",
                                                   kBoundDoubleBatch);

        narrowRatClaims[0] = claimA;
        narrowRatClaims[1] = claimB;
        narrowRatClaims[2] = claimEntry;

        std::array<double, 33> narrow{};
        std::array<double, 33> shipped{};

        for (std::size_t i = 0; i < count; ++i)
        {
            const double x = ref.x[i];

            boys::BoysAllOrders< NarrowRational>(nmax, x, narrow.data());
            boys::BoysAllOrders< ShippedRational>(nmax, x, shipped.data());

            for (int n = 0; n <= nmax; ++n)
            {
                const std::size_t k = ref.Index(n, i);
                const double got = narrow[static_cast<std::size_t>(n)];

                ++narrowRatCells;

                if (got != shipped[static_cast<std::size_t>(n)])
                {
                    ++narrowRatDiffers;
                }

                MeasureInto(GranularityClaims(),
                            claimEntry,
                            n,
                            x,
                            got,
                            ref.v[k],
                            ref.decade[k],
                            kBoundDoubleBatch,
                            Unrepresentable(got, -1022));

                if (x < boys::detail::kX0 && x >= NarrowFit::kRegionAFitsFrom)
                {
                    const double piece = boys::detail::RegionAValue<NarrowFit>(n, x);

                    narrowRatWorst = std::max(narrowRatWorst, std::fabs(piece - ref.v[k]));
                    MeasureInto(GranularityClaims(),
                                claimA,
                                n,
                                x,
                                piece,
                                ref.v[k],
                                ref.decade[k],
                                boys::detail::kRegionAFitBar,
                                Unrepresentable(piece, -1022));
                }
            }

            if (x >= boys::detail::kX0 && x < boys::detail::kX1)
            {
                const std::size_t k = ref.Index(0, i);

                MeasureInto(GranularityClaims(),
                            claimB,
                            0,
                            x,
                            narrow[0],
                            ref.v[k],
                            ref.decade[k],
                            boys::detail::kRegionBFitBar,
                            Unrepresentable(narrow[0], -1022));
            }
        }

        std::printf("\n  the rational route over the narrow partition, measured against the "
                    "same reference\n  the route rows are: %zu cell(s) through the batch entry, "
                    "%zu of them a value the\n  shipped partition's route does not answer with "
                    "(the carriage that makes naming it an\n  option rather than a spelling), "
                    "worst piece value %.6g away from the reference.\n",
                    narrowRatCells,
                    narrowRatDiffers,
                    narrowRatWorst);
    }

    // ---- fp16 and bf16, the store-half lane -------------------------------
#ifdef BOYS_GATE_FP16
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
#endif // BOYS_GATE_FP16

    // ---- the half kernels' path divergence and their claimed domain --------
    // The vector body against the certified scalar half entry at the same cell: they disagree by
    // quanta of the half format, so the claim is not equality but that neither returns a value the
    // other's contract would not accept - the failure this caught was one returning zero.
#ifdef BOYS_GATE_FP16
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
    // The aggregate above says a cell has exactly one path out of bound but not which one, and
    // the two sides are not the same finding: the body being out is the kernel's error, the scalar
    // being out is the certified entry's. They are counted apart, per side and per (region, format).
    std::size_t driftSimdOut = 0;   // the body outside its bound, the scalar inside
    std::size_t driftScalarOut = 0; // the scalar outside its bound, the body inside
    std::array<std::size_t, 6> driftSimdOutCell{};   // [region * 2 + isBf16]
    std::array<std::size_t, 6> driftScalarOutCell{};
    // What the out-of-bound return actually is, per side: a number the bound is too tight for, or
    // a value that is not a number at all, which no bound can hold. The second is the one this row
    // has fired on, so it is counted, and the argument each set sits at is counted too.
    std::size_t driftSimdOutNaN = 0;
    std::size_t driftScalarOutNaN = 0;
    std::size_t driftSimdOutInfArg = 0;    // half argument is not a finite half
    std::size_t driftScalarOutInfArg = 0;
    std::size_t driftSimdOutRefZero = 0;   // reference is exactly zero there
    std::size_t driftScalarOutRefZero = 0;
    double driftSimdOutRatio = 0.0; // the largest the offending side went over
    double driftScalarOutRatio = 0.0;
    double driftSimdOutX = 0.0;     // the largest argument of each side's set
    double driftScalarOutX = 0.0;
    int driftSimdOutN = -1;
    int driftScalarOutN = -1;
    int driftSimdOutRegion = -1;
    int driftScalarOutRegion = -1;
    bool driftSimdOutBf16 = false;
    bool driftScalarOutBf16 = false;
#endif // BOYS_GATE_FP16

    // ---- the 8-wide half-I/O region kernels --------------------------------
    // Two half values per register, computed in binary32 and rounded to the half type on the store
    // (the lane's own scalar tail calls the certified half entries). No public entry reaches these.
    // Arguments are partitioned by the region each kernel documents as its precondition.
#ifdef BOYS_GATE_FP16
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

                        const std::size_t sideSlot =
                            static_cast<std::size_t>(region) * 2 + (isBf16 ? 1 : 0);
                        const double halfArg =
                            isBf16 ? static_cast<double>(static_cast<float>(xb[j]))
                                   : static_cast<double>(static_cast<float>(x16[j]));

                        if (!simdIn)
                        {
                            ++driftSimdOut;
                            ++driftSimdOutCell[sideSlot];

                            if (!std::isfinite(got))
                            {
                                ++driftSimdOutNaN;
                            }

                            if (!std::isfinite(halfArg))
                            {
                                ++driftSimdOutInfArg;
                            }

                            if (want == 0.0)
                            {
                                ++driftSimdOutRefZero;
                            }

                            const double ratio =
                                std::abs(got - want) / HalfBound(got, mantissa, minExp);

                            if (ratio > driftSimdOutRatio)
                            {
                                driftSimdOutRatio = ratio;
                                driftSimdOutRegion = region;
                                driftSimdOutBf16 = isBf16;
                            }

                            if (ref.x[i] > driftSimdOutX)
                            {
                                driftSimdOutX = ref.x[i];
                                driftSimdOutN = order;
                            }
                        } else
                        {
                            ++driftScalarOut;
                            ++driftScalarOutCell[sideSlot];

                            if (!std::isfinite(scalar))
                            {
                                ++driftScalarOutNaN;
                            }

                            if (!std::isfinite(halfArg))
                            {
                                ++driftScalarOutInfArg;
                            }

                            if (want == 0.0)
                            {
                                ++driftScalarOutRefZero;
                            }

                            const double ratio =
                                std::abs(scalar - want) / HalfBound(scalar, mantissa, minExp);

                            if (ratio > driftScalarOutRatio)
                            {
                                driftScalarOutRatio = ratio;
                                driftScalarOutRegion = region;
                                driftScalarOutBf16 = isBf16;
                            }

                            if (ref.x[i] > driftScalarOutX)
                            {
                                driftScalarOutX = ref.x[i];
                                driftScalarOutN = order;
                            }
                        }

                        if (driftDump)
                        {
                            std::printf("  drift-one-side-out %s region %d n=%d x=%.17g got=%.9g "
                                        "scalar=%.9g ref=%.9g |got-ref|=%.6g bound(got)=%.6g "
                                        "|scalar-ref|=%.6g bound(scalar)=%.6g %s\n",
                                        isBf16 ? "bf16" : "fp16",
                                        region,
                                        order,
                                        ref.x[i],
                                        got,
                                        scalar,
                                        want,
                                        std::abs(got - want),
                                        HalfBound(got, mantissa, minExp),
                                        std::abs(scalar - want),
                                        HalfBound(scalar, mantissa, minExp),
                                        !simdIn ? "BODY-OUT" : "SCALAR-OUT");
                        }
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
#endif // BOYS_GATE_FP16

    // ---- the native packed half lane, if this revision carries it -----------
    // Region C only, one precondition (x >= the lane's own fp16 rounding of x1), no fallback,
    // returning 2^15 * F_k(x) so the ladder stays normal down to F_k(x) = 2^-29. Measured apart:
    // the bound where the return is a normal half, the points past that ceiling, and the packing.
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

                    // Past the ceiling the lane claims nothing: the return is a subnormal half,
                    // then exactly zero, by design. Excused only inside that domain - a subnormal
                    // for a value the scaling should have kept normal fails.
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

    // The reference grid's own columns of that statement are read whatever the seam is; the
    // lane's side of it needs an entry to call, so it is measured only where this build carries
    // one.
#ifdef BOYS_GATE_FP16
    double cellErr = 0.0;
    double cellValue = 0.0;
    double cellRef = 0.0;
    double cellBound = 0.0;
#endif
    std::array<double, boys::kMaxBoysOrder + 1> refNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> refZeroX{};
    std::array<double, boys::kMaxBoysOrder + 1> refSpanNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> refSpanSubnormalX{};
#ifdef BOYS_GATE_FP16
    std::array<double, boys::kMaxBoysOrder + 1> laneNormalX{};
    std::array<double, boys::kMaxBoysOrder + 1> laneNonzeroX{};
    std::array<double, boys::kMaxBoysOrder + 1> laneRefNormalX{};
#endif

    for (int n = 0; n <= nmax; ++n)
    {
#ifdef BOYS_GATE_FP16
        laneNormalX[static_cast<std::size_t>(n)] = -1.0;
        laneNonzeroX[static_cast<std::size_t>(n)] = -1.0;
        laneRefNormalX[static_cast<std::size_t>(n)] = -1.0;
#endif
        refNormalX[static_cast<std::size_t>(n)] =
            LargestXAbove(ref, n, std::ldexp(1.0, kF16MinNormalExp));
        refZeroX[static_cast<std::size_t>(n)] = LargestXAbove(ref, n, std::ldexp(1.0, -25));
        refSpanNormalX[static_cast<std::size_t>(n)] = LargestXSparsable(ref, n, kF16NormalField);
        refSpanSubnormalX[static_cast<std::size_t>(n)] = LargestXSparsable(ref, n, kF16Field);
    }

#ifdef BOYS_GATE_FP16
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
#endif // BOYS_GATE_FP16

    // The asymptotic branch outside its own domain: the published row says the error jumps five
    // to eight orders below about x = 16, measured on the form as coded (RegionCForm, where the
    // reason it is evaluated rather than read out of the kernel is stated). A jump has to say over
    // what window, and this form's error is smooth in x, so both readings are reported.
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
    // The shape of the error through the boundary, as opposed to its size: the published sentence
    // calls the lower edge a floor, and a threshold would show as neighbouring arguments whose
    // errors differ by orders of magnitude and as an error not monotone in x.
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

                const double err = std::abs(RegionCForm(n, x) - ref.v[ref.Index(n, i)]);
                const std::size_t sn = static_cast<std::size_t>(n);
                const double trueValue = ref.v[ref.Index(n, i)];
                // The shape is a property of the error as a function of x, and at these
                // arguments the value itself moves by orders of magnitude between neighbouring
                // nodes, so the comparison that means anything is between relative errors.
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

            // The other reading a jump can have here: the error just below the edge against the
            // error the branch delivers inside its domain, which the sweep measures as its 5.5e-14
            // budget. The two are the same number, so this compares the branch's two regimes rather
            // than against a threshold pulled from thin air.
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

                const double err = std::abs(RegionCForm(n, x) - ref.v[ref.Index(n, i)]);

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

    // The packed-half paragraph's reason, checkable here: "the rescale is exact, so the scaled
    // lane is bit-identical to the unscaled one at every order". Rounding to binary16 and scaling
    // by a power of two commute wherever neither end leaves the format's normal range; a value that
    // is itself subnormal is not evidence either way and is skipped.
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


    // The float lane's own floor, relative rather than absolute: the published ceiling paragraph
    // says the 24-bit significand resolves about 6e-8 and the 1.5e-7 absolute budget is within a
    // factor of a few of it. The absolute sweep cannot see that floor, so it is measured where
    // |F| >= 0.5, which is where a relative floor is what the caller gets.
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


    // The evaluation-scheme book: one accumulator per row the public surface enumerates and per
    // entry that reads a scheme. A fit row is that fit over its interval at the bound the surface
    // reports in this build's multiply-add route; an entry row is one public entry over the grid at
    // one scheme. Both schemes hold the same bar, so each entry is read at both and cells counted.
    const std::span<const boys::EvalFitInfo> schemeFitRows = boys::BoysEvalSchemeFits();
    std::vector<int> schemeFitSlots;
    schemeFitSlots.reserve(schemeFitRows.size());

    // The cells each row was read at under both schemes and the cells where the two readings
    // differed, per region: a scheme reaches a row region by region, so a drop in one body is a row
    // of zeros beside three rows of numbers rather than a total that still looks healthy.
    struct SchemeCarriage {
        const char* entry = "";
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
        schemeFitSlots.push_back(AddSchemeClaim(boys::EvalSchemeName(fit.scheme),
                                                EvalLaneName(fit.lane),
                                                boys::BoysEvalSchemeDelivered(fit.scheme, fit.lane)));
    }

    // The call shape a row is one of. Each is a public entry, and each reads the
    // scheme its policy names somewhere between the call site and the fit.
    enum class SchemeEntryKind : std::uint8_t {
        kSingle,       // BoysSingle: one order at one argument
        kOrders,       // BoysAllOrders: every order at one argument
        kFixedN,       // BoysFixedN: one order over the arguments of an array
        kAllN,         // BoysAllN: every order over an array, the grouping done inside
        kAllNSorted,   // BoysAllN with the BoysSortedArgs overload
        kAllNAtOrders, // BoysAllNAtOrders: every order over an array, a top order per argument
    };

    struct SchemeEntry {
        boys::EvalScheme scheme = boys::EvalScheme::kSplitClenshaw;
        const char* entry = "";
        SchemeEntryKind kind = SchemeEntryKind::kSingle;
        int slot = -1;
        int carriage = -1;
    };

    // The rows of the entry table. Written once and instantiated per scheme the
    // enumeration reports, so a scheme added to the enumeration is measured
    // without this list changing. Each row is a public entry.
    struct EntryRow {
        const char* entry;
        SchemeEntryKind kind;
    };

    std::vector<EntryRow> entryRows{
        {"single entry", SchemeEntryKind::kSingle},
        {"orders entry", SchemeEntryKind::kOrders},
        {"fixed-n entry", SchemeEntryKind::kFixedN},
        {"all-n entry", SchemeEntryKind::kAllN},
        {"all-n sorted", SchemeEntryKind::kAllNSorted},
        {"all-n at-orders entry", SchemeEntryKind::kAllNAtOrders},
    };

    std::vector<SchemeEntry> schemeEntries;
    std::vector<SchemeCarriage> schemeEntryCarriage;

    for (const EntryRow& row : entryRows)
    {
        SchemeCarriage car;
        car.entry = row.entry;
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
            e.kind = row.kind;
            e.slot = AddSchemeClaim(info.name, row.entry, baseBound);
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

    // One entry row at one scheme. The other scheme's reading is taken in the
    // same pass and only to count the cells where the two differ: the count is
    // the whole reason the entry table exists, and it is taken once per entry
    // rather than once per scheme row.
    const auto sweepEntryScheme = [&]<boys::EvalScheme kScheme>(SchemeEntry& e) {
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
                    const double got = boys::BoysSingle<SchemePolicy<kScheme>>(n, x);
                    MeasureAt(acc,
                              n,
                              x,
                              got,
                              ref.v[k],
                              ref.decade[k],
                              SingleBound(x),
                              Unrepresentable(got, -1022));

                    if constexpr (kFirst)
                    {
                        note(got, boys::BoysSingle<SchemePolicy<kOther>>(n, x), x);
                    }
                }
            }

            break;

        case SchemeEntryKind::kOrders:
            for (std::size_t i = 0; i < count; ++i)
            {
                std::array<double, 33> a{};
                std::array<double, 33> b{};
                boys::BoysAllOrders<SchemePolicy<kScheme>>(nmax, ref.x[i], a.data());

                if constexpr (kFirst)
                {
                    boys::BoysAllOrders<SchemePolicy<kOther>>(nmax, ref.x[i], b.data());
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
                              kBoundDoubleBatch,
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
                boys::BoysFixedN<SchemePolicy<kScheme>>(n, ref.x.data(), a.data(), count);

                if constexpr (kFirst)
                {
                    boys::BoysFixedN<SchemePolicy<kOther>>(n, ref.x.data(), b.data(), count);
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
                              kBoundDoubleBatch,
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
                boys::BoysAllN<SchemePolicy<kScheme>>(nmax, args.data(), a.data(), count,
                                                          boys::BoysSortedArgs{});

                if constexpr (kFirst)
                {
                    boys::BoysAllN<SchemePolicy<kOther>>(nmax, args.data(), b.data(), count,
                                                             boys::BoysSortedArgs{});
                }
            } else
            {
                boys::BoysAllN<SchemePolicy<kScheme>>(nmax, args.data(), a.data(), count);

                if constexpr (kFirst)
                {
                    boys::BoysAllN<SchemePolicy<kOther>>(nmax, args.data(), b.data(), count);
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
                              kBoundDoubleBatch,
                              Unrepresentable(a[p], -1022));

                    if constexpr (kFirst)
                    {
                        note(a[p], b[p], ref.x[i]);
                    }
                }
            }

            break;
        }

        case SchemeEntryKind::kAllNAtOrders: {
            // The entry's own shape: a top order per argument, so the planes a
            // scheme reaches are read on the columns that run that deep. The
            // two readings share the top array, which is what makes them a
            // carriage comparison rather than two different calls.
            std::vector<int> tops(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                tops[i] = nmax - static_cast<int>(i % static_cast<std::size_t>(nmax + 1));
            }

            std::vector<double> a(count * static_cast<std::size_t>(nmax + 1));
            std::vector<double> b(count * static_cast<std::size_t>(nmax + 1));
            boys::BoysAllNAtOrders<SchemePolicy<kScheme>>(
                tops.data(), ref.x.data(), a.data(), count);

            if constexpr (kFirst)
            {
                boys::BoysAllNAtOrders<SchemePolicy<kOther>>(
                    tops.data(), ref.x.data(), b.data(), count);
            }

            for (std::size_t i = 0; i < count; ++i)
            {
                for (int n = 0; n <= tops[i]; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    MeasureAt(acc,
                              n,
                              ref.x[i],
                              a[k],
                              ref.v[k],
                              ref.decade[k],
                              kBoundDoubleBatch,
                              Unrepresentable(a[k], -1022));

                    if constexpr (kFirst)
                    {
                        note(a[k], b[k], ref.x[i]);
                    }
                }
            }

            break;
        }

        }
    };

    const auto sweepEntry = [&](SchemeEntry& e) {
        if (e.scheme == boys::EvalScheme::kSplitClenshaw)
        {
            sweepEntryScheme.template operator()<boys::EvalScheme::kSplitClenshaw>(e);
        } else
        {
            sweepEntryScheme.template operator()<boys::EvalScheme::kHorner>(e);
        }
    };

    sweepSchemeFits.template operator()<boys::EvalScheme::kSplitClenshaw>();
    sweepSchemeFits.template operator()<boys::EvalScheme::kHorner>();

    for (SchemeEntry& e : schemeEntries)
    {
        sweepEntry(e);
    }

    // The packed region-A lane, by scheme: the many-argument entry hands its low-order region-A
    // runs to the packed AVX2 lane, which holds the shipped coefficients and recurrence and so
    // serves the shipped scheme alone; a call naming another scheme gets the scalar body, a lane
    // lost and not a value - the scope BoysPackedLaneServes states, held here to the values.
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
            boys::BoysAllN< SchemeLanePolicy<kScheme>>(
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
                boys::BoysAllOrders< SchemeLanePolicy<kScheme>>(kLaneNmax,
                                                                    ref.x[i],
                                                                    per.data());

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

    // What a scheme reaches, region by region, read off the per-argument entry. It is the
    // reference every carriage row above is judged against, and a region where it shows no
    // difference is a region no scheme reaches through it, so no row is held to one.
    std::array<std::size_t, 4> schemeRefDiffer{};

    const auto referenceCarriage = [&](std::array<std::size_t, 4>& sink) {
        for (std::size_t i = 0; i < count; ++i)
        {
            std::array<double, 33> a{};
            std::array<double, 33> b{};
            boys::BoysAllOrders<SchemePolicy<boys::EvalScheme::kSplitClenshaw>>(
                nmax, ref.x[i], a.data());
            boys::BoysAllOrders<SchemePolicy<boys::EvalScheme::kHorner>>(
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

    referenceCarriage(schemeRefDiffer);

    // ---- the packing book ---------------------------------------------------
    // Two rows per scheme the orders axis carries, because the axis's answer and the entry's cover
    // different intervals: "its own domain" is x < kX0 at the bar the fits are certified at - the
    // row the axis is a claim about - and "whole grid" runs to the scalar fallback past kX0.
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
                boys::BoysAllOrders< OrdersPackPolicy<kScheme>>(nmax, x, out.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const double got = out[static_cast<std::size_t>(n)];

                    if (x < boys::detail::kX0)
                    {
                        // The axis is served by the packed AVX2 lane where the host has one and
                        // by the certified scalar single lane where it has not, so the row is judged
                        // against the figure for the host it ran on; below the extended boundary the
                        // two figures are the same.
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

    // The same axis on the plane entry, which carries it too. The bounds are the all-orders
    // rows' at the same figures, and deliberately so: naming the axis on either entry returns the
    // same values, because the plane entry's per-argument path IS the all-orders entry's body - so
    // what the two rows measure is that the axis is carried on the second call shape at all.
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

            boys::BoysAllN< OrdersPackPolicy<kScheme>>(
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

                    // The plane entry's whole-grid row is the batch bound, not the single
                    // lane's per-region table: a plane call promises the batch row everywhere, and
                    // past the lane's interval it runs the certified scalar single lane.
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

    // ---- the rest of the axis: the other route, and the other partition -----
    // The reference rows above are the axis on the shipped route over the shipped partition; the
    // axis answers more, and an option a caller can name with no measured row beside it is the
    // silent gap this book exists to catch. Rows are judged at the figure the entry documents.
    struct OpenedRow {
        const char* axis = "";
        std::string row;
        double bound = 0.0;
        std::size_t slot = 0;
    };

    std::vector<OpenedRow> openedRows;

    // The four rows one combination publishes, in the order the slots below are
    // addressed by: the all-orders entry over the packed lane's own interval and
    // over the whole committed grid, then the plane entry's same two. "pl" marks
    // the plane entry's rows in the report.
    const char* const kRowKind[4] = {"A", "A..C", "pl A", "pl A..C"};

    // The route a combination reads its region-A fits from. Every row above this
    // block measured the shipped route; the rational route is the one the orders
    // axis did not carry.
    const char* const kRouteTag[2] = {"cheb", "rat"};

    // The partition a row's region-A fits were read from. The narrow partition is the one the
    // orders axis did not carry, and its pieces are cut per order, so the axis reaches them by
    // fetching each order's own piece rather than by stepping one piece at a fixed stride.
    const char* const kPartitionTag[2] = {"", "narrow "};

    // The flat index of one combination's four labels: partition, route and row
    // kind, in that order.
    const auto openedLabel = [](std::size_t partIdx,
                                std::size_t routeIdx,
                                std::size_t kind) {
        return ((partIdx * 2u + routeIdx) * 4u + kind);
    };

    // A claim records the pointers it is handed rather than copying them, so a
    // row's label has to outlive the loop that makes it. A fixed array of
    // strings does: its elements are built once and never moved afterwards.
    std::array<std::string, 2u * 2u * 4u> openedLabels{};

    for (std::size_t partIdx = 0; partIdx < 2; ++partIdx)
    {
        for (std::size_t routeIdx = 0; routeIdx < 2; ++routeIdx)
        {
            for (std::size_t kind = 0; kind < 4; ++kind)
            {
                openedLabels[openedLabel(partIdx, routeIdx, kind)] =
                    Fmt("%s%s %s", kPartitionTag[partIdx], kRouteTag[routeIdx], kRowKind[kind]);
            }
        }
    }

    // One slot per combination, addressed by the tuple the entry names: route, partition, scheme,
    // entry and region, in that order. A sweep finds the row it fills through this table, so a
    // tuple that grows a member is measured where it is published. The reference rows' own eight
    // stay at -1, as do the narrow partition's rational slots, which it does not have.
    std::vector<int> openedSlots(2u * 2u * 2u * 2u * 2u, -1);

    const auto openedSlot = [](std::size_t routeIdx,
                               std::size_t partIdx,
                               std::size_t schemeIdx,
                               std::size_t entry,
                               std::size_t region) {
        return static_cast<std::size_t>(
                   ((((routeIdx * 2u + partIdx) * 2u + schemeIdx) * 2u + entry) * 2u + region));
    };

    for (std::size_t partIdx = 0; partIdx < 2; ++partIdx)
    {
        for (std::size_t routeIdx = 0; routeIdx < 2; ++routeIdx)
        {
            if (partIdx == 0 && routeIdx == 0)
            {
                continue; // the shipped route over the shipped partition: the rows above
            }

            if (partIdx == 1 && routeIdx == 1)
            {
                continue; // the narrow partition carries no rational region-A route
            }

            // The figure a row is published at, the largest any of its own cells is judged at:
            // inside the packed lane's interval the bar the fits are cut for, and the entry's
            // documented 5.5e-14 over the whole grid. The narrow pieces are cut under a narrower
            // bar, so one figure covers both partitions' rows.
            const double packedBar = (routeIdx == 1) ? kBoundDoubleBatch : kBoundSingleA;
            const double regionARow = boys::BoysAvx2Available()
                                          ? packedBar
                                          : std::max(packedBar, kBoundSingleBand);
            const double rowBound[4] = {regionARow,
                                        kBoundSingleC,
                                        regionARow,
                                        kBoundDoubleBatch};

            for (std::size_t schemeIdx = 0; schemeIdx < 2; ++schemeIdx)
            {
                const char* axis =
                    boys::EvalSchemeName(static_cast<boys::EvalScheme>(schemeIdx));

                for (std::size_t entry = 0; entry < 2; ++entry)
                {
                    for (std::size_t region = 0; region < 2; ++region)
                    {
                        const std::size_t at = entry * 2u + region;
                        const std::size_t label = openedLabel(partIdx, routeIdx, at);
                        const int slot =
                            AddPackClaim(axis, openedLabels[label].c_str(), rowBound[at]);

                        openedSlots[openedSlot(routeIdx, partIdx, schemeIdx, entry, region)] =
                            slot;
                        openedRows.push_back(OpenedRow{axis,
                                                       openedLabels[label],
                                                       rowBound[at],
                                                       static_cast<std::size_t>(slot)});
                    }
                }
            }
        }
    }

    // What a cell of an opened row is judged against, by route and by host: the shipped route's
    // rows as the rows above are - the packed lane's per-order bar inside its own interval and the
    // scalar single lane's per-region budgets everywhere else - and the rational route at its own
    // documented 5.5e-14.
    const auto openedRegionABound = [&](double x, std::size_t routeIdx) {
        if (routeIdx == 1)
        {
            return kBoundDoubleBatch;
        }

        return boys::BoysAvx2Available() ? kBoundSingleA : SingleBound(x);
    };

    const auto openedGridBound = [&](double x, std::size_t routeIdx) {
        return (routeIdx == 1) ? kBoundDoubleBatch : SingleBound(x);
    };

    // One combination, measured through both entries that can carry the axis. Each answers the
    // same two questions about its own layout - the packed lane's interval at the bar the fits are
    // cut for, and every grid argument at the entry's documented figure - so a combination is four
    // rows, and the partition is part of it because it decides what the axis reads.
    const auto measureOpened =
        [&]<boys::EvalScheme kScheme,
            boys::FitRoute kRoute,
            boys::FitGranularity kPart =
                boys::FitGranularity::kCoarsest>() {
            constexpr std::size_t kRouteIdx = static_cast<std::size_t>(kRoute);
            constexpr std::size_t kPartIdx = static_cast<std::size_t>(kPart);
            constexpr std::size_t kSchemeIdx = static_cast<std::size_t>(kScheme);
            using Policy = boys::EvalPolicy<kRoute,
                                            kScheme,
                                            boys::BoysBudget::kFloat,
                                            boys::PackAxis::kOrders,
                                            kPart>;

            // The claim a combination fills is the one the table above holds for its tuple.
            // Reading it back rather than deriving it from the reading order is what keeps a sweep
            // and its row together: the two orders are different and only the table knows the
            // second.
            const auto claimSlot = [&](std::size_t partIdx,
                                       std::size_t schemeIdx,
                                       std::size_t entry,
                                       std::size_t region) {
                const std::size_t at =
                    openedSlot(kRouteIdx, partIdx, schemeIdx, entry, region);
                return static_cast<std::size_t>(openedSlots[at]);
            };

            const std::size_t ordersRegionA = claimSlot(kPartIdx, kSchemeIdx, 0, 0);
            const std::size_t ordersGrid = claimSlot(kPartIdx, kSchemeIdx, 0, 1);
            const std::size_t planeRegionA = claimSlot(kPartIdx, kSchemeIdx, 1, 0);
            const std::size_t planeGrid = claimSlot(kPartIdx, kSchemeIdx, 1, 1);
            std::vector<double> planes(count * (static_cast<std::size_t>(nmax) + 1));

            boys::BoysAllN<Policy>(nmax, ref.x.data(), planes.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];
                std::array<double, 33> out{};
                boys::BoysAllOrders<Policy>(nmax, x, out.data());

                const double ordersGridBound = openedGridBound(x, kRouteIdx);
                const double planeGridBound = kBoundDoubleBatch;

                for (int n = 0; n <= nmax; ++n)
                {
                    const std::size_t k = ref.Index(n, i);
                    const std::size_t j = static_cast<std::size_t>(n);
                    const double viaOrders = out[j];
                    const double viaPlane = planes[j * count + i];

                    if (x < boys::detail::kX0)
                    {
                        const double regionABound = openedRegionABound(x, kRouteIdx);

                        MeasureAt(PackClaims()[ordersRegionA],
                                  n,
                                  x,
                                  viaOrders,
                                  ref.v[k],
                                  ref.decade[k],
                                  regionABound,
                                  Unrepresentable(viaOrders, -1022));
                        MeasureAt(PackClaims()[planeRegionA],
                                  n,
                                  x,
                                  viaPlane,
                                  ref.v[k],
                                  ref.decade[k],
                                  regionABound,
                                  Unrepresentable(viaPlane, -1022));
                    }

                    // The plane entry's whole-grid row is the batch figure rather than the
                    // single lane's per-region table: a plane call promises the batch row in every
                    // region.
                    MeasureAt(PackClaims()[ordersGrid],
                              n,
                              x,
                              viaOrders,
                              ref.v[k],
                              ref.decade[k],
                              ordersGridBound,
                              Unrepresentable(viaOrders, -1022));
                    MeasureAt(PackClaims()[planeGrid],
                              n,
                              x,
                              viaPlane,
                              ref.v[k],
                              ref.decade[k],
                              planeGridBound,
                              Unrepresentable(viaPlane, -1022));
                }
            }
        };

    // Every combination the axis answers beyond the rows above: the other route, and the other
    // partition. Named rather than looped, because the route, the scheme and the partition are
    // template arguments a reader comparing two rows can see.
    constexpr auto kSplit = boys::EvalScheme::kSplitClenshaw;
    constexpr auto kHorner = boys::EvalScheme::kHorner;
    constexpr auto kCheb = boys::FitRoute::kChebyshev;
    constexpr auto kRat = boys::FitRoute::kRationalMinimax;

    measureOpened.template operator()<kSplit, kRat>();

    measureOpened.template operator()<kHorner, kRat>();

    // The same axis over the other partition, at each scheme. The shipped route
    // is the only one it carries: the narrow partition's region-A route set is
    // the chebyshev table, and the rational route over it is a combination this
    // library does not have, so its rational slots are the unread ones.
    measureOpened.template operator()<kSplit, kCheb, boys::FitGranularity::kNarrow>();

    measureOpened.template operator()<kHorner, kCheb, boys::FitGranularity::kNarrow>();

    // ---- the granularity axis ----------------------------------------------
    // Which of two partitions a call reads. The uniform grid is not a member: its fits are one
    // table over [0, kFlatHi) rather than a region-A cut beside a region-B seed, so the arm reading
    // the members here has no third branch - work, not impossibility. Entry rows take the table's cell.
    enum class GranKind : std::uint8_t {
        kFitsA,       // region A's stored fits, read directly, over x < kX0
        kSeedB,       // region B's stored seed, read directly
        kOrdersBelow, // BoysAllOrders below the band, where the seeding fallback is taken
        kOrdersA,     // BoysAllOrders, the call shape with one argument, over x < kX0
        kOrdersB,     // BoysAllOrders over region B, where the seed is carried up
        kSingleA,     // BoysSingle, one order at one argument, over the table's region-A cell
        kSingleWhole, // BoysSingle over the whole grid, at the lane's per-region bars
        kFixedN,      // BoysFixedN, one order over the arguments of the array
        kPlaneA,      // BoysAllN, every order over the array, over x < kX0
    };

    // The bar a row's cells are judged at: one published number, or the
    // single-order lane's own bar for the region an argument falls in, which is
    // what that lane's documented rows are judged at.
    enum class GranBar : std::uint8_t { kFixed, kPerRegion };

    struct GranRow {
        const char* row;
        GranKind kind;
        GranBar bar;
        double bound; // the bar, where bar is kFixed
    };

    constexpr GranRow granRows[] = {
        {"stored fits, region A", GranKind::kFitsA, GranBar::kFixed, kBoundSingleA},
        {"stored seed, region B", GranKind::kSeedB, GranBar::kFixed, kBoundSingleB},
        {"orders entry, below band",
         GranKind::kOrdersBelow,
         GranBar::kFixed,
         kBoundDoubleBatch},
        {"orders entry, band and below", GranKind::kOrdersA, GranBar::kFixed, kBoundDoubleBatch},
        {"orders entry, region B", GranKind::kOrdersB, GranBar::kFixed, kBoundDoubleBatch},
        {"single entry, region A", GranKind::kSingleA, GranBar::kFixed, kBoundSingleA},
        {"single entry, A..C", GranKind::kSingleWhole, GranBar::kPerRegion, 0.0},
        // The fixed-order entry - the shape a caller that needs one order over an array of
        // arguments reaches for - has its own branch on the partition, and this row says that branch
        // reads the table its policy names. Judged at the single-order lane's per-region figure.
        {"fixed-order entry, A..C", GranKind::kFixedN, GranBar::kPerRegion, 0.0},
        {"plane entry, band and below", GranKind::kPlaneA, GranBar::kFixed, kBoundDoubleBatch},
    };
    constexpr std::size_t kGranRowCount = std::size(granRows);
    constexpr std::size_t kGranMembers = 2;
    constexpr std::size_t kGranSchemeCount = 2;

    static_assert(static_cast<std::size_t>(boys::FitGranularity::kCoarsest) == 0
                      && static_cast<std::size_t>(boys::FitGranularity::kNarrow) == 1,
                  "the tables below are indexed by the enumerator, so the enumerators are the "
                  "order they are read in");

    // The schemes, in the order the report walks them, read from the library's
    // own report so a scheme added there is measured here at both partitions.
    std::array<boys::EvalScheme, kGranSchemeCount> granSchemes{};
    std::size_t granSchemeCount = 0;

    for (const boys::EvalSchemeInfo& info : boys::BoysEvalSchemes())
    {
        if (granSchemeCount < kGranSchemeCount)
        {
            granSchemes[granSchemeCount++] = info.scheme;
        }
    }

    // One row of the book is one (partition, scheme, row) tuple: a member, a
    // scheme, and one of the questions the row list above asks.

    // The flat index of one cell of the tables below, addressed by the tuple a
    // row names: partition, scheme, row, in that order.
    const auto granIndex = [](std::size_t member, std::size_t scheme, std::size_t row) {
        return (member * kGranSchemeCount + scheme) * kGranRowCount + row;
    };

    std::vector<int> granSlots(kGranMembers * kGranSchemeCount * kGranRowCount, -1);

    for (std::size_t g = 0; g < kGranMembers; ++g)
    {
        const char* member = boys::GranularityName(static_cast<boys::FitGranularity>(g));

        for (std::size_t s = 0; s < granSchemeCount; ++s)
        {
            for (std::size_t r = 0; r < kGranRowCount; ++r)
            {
                granSlots[granIndex(g, s, r)] =
                    AddGranularityClaim(member, granRows[r].row, granRows[r].bound);
            }
        }
    }

    // The cells each row was read at, and the cells where the two members'
    // readings differed, one row per (granularity, scheme, row).
    std::vector<std::size_t> granCells(granSlots.size(), 0);
    std::vector<std::size_t> granDiffer(granSlots.size(), 0);

    // Both members' readings of the whole grid, held between the members' passes so that the
    // second member's reading of a cell is an array read and not a second call: one array per
    // member per call shape.
    std::array<std::vector<double>, kGranMembers> ordersGrid;
    std::array<std::vector<double>, kGranMembers> planeGrid;
    std::array<std::vector<double>, kGranMembers> singleGrid;
    std::array<std::vector<double>, kGranMembers> fixednGrid;

    // The stored fit one lane names at one partition and one scheme, read
    // directly: PartitionFitValue above, which is the certified fit and the
    // path the rows above already read.
    const auto granStoredFit = [&]<boys::EvalScheme kScheme,
                                   boys::FitGranularity kGranularity>(boys::EvalLane lane,
                                                                      int n,
                                                                      double x) -> double {
        return PartitionFitValue<kScheme, kGranularity>(lane, n, x);
    };

    const auto sweepGranularity =
        [&]<std::size_t kSchemeIndex, boys::EvalScheme kScheme>() {
        // The whole grid at one member, one entry at a time. The two passes
        // over the members are what makes the tables above the entry's own
        // numbers at both partitions; a member read alone would leave the
        // carriage count with nothing to compare against.
        const auto fill = [&]<boys::FitGranularity kGranularity>() {
            constexpr std::size_t kMember =
                kGranularity == boys::FitGranularity::kCoarsest ? 0 : 1;
            const std::size_t grid = count * (static_cast<std::size_t>(nmax) + 1);

            ordersGrid[kMember].assign(grid, 0.0);
            planeGrid[kMember].assign(grid, 0.0);
            singleGrid[kMember].assign(grid, 0.0);
            fixednGrid[kMember].assign(grid, 0.0);

            std::array<double, 33> orders{};

            for (std::size_t i = 0; i < count; ++i)
            {
                boys::BoysAllOrders<GranularityPolicy<kScheme, kGranularity>>(
                    nmax, ref.x[i], orders.data());

                for (int n = 0; n <= nmax; ++n)
                {
                    ordersGrid[kMember][ref.Index(n, i)] = orders[static_cast<std::size_t>(n)];
                }
            }

            boys::BoysAllN<GranularityPolicy<kScheme, kGranularity>>(
                nmax, ref.x.data(), planeGrid[kMember].data(), count);

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < count; ++i)
                {
                    singleGrid[kMember][ref.Index(n, i)] =
                        boys::BoysSingle<GranularityPolicy<kScheme, kGranularity>>(n, ref.x[i]);
                }
            }

            std::vector<double> column(count);

            for (int n = 0; n <= nmax; ++n)
            {
                boys::BoysFixedN<GranularityPolicy<kScheme, kGranularity>>(
                    n, ref.x.data(), column.data(), count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    fixednGrid[kMember][ref.Index(n, i)] = column[i];
                }
            }
        };

        fill.template operator()<boys::FitGranularity::kCoarsest>();
        fill.template operator()<boys::FitGranularity::kNarrow>();

        const auto measure = [&]<boys::FitGranularity kGranularity>() {
            constexpr std::size_t kMember =
                kGranularity == boys::FitGranularity::kCoarsest ? 0 : 1;

            // The other member, for the carriage count. The relation is
            // symmetric, so the count is taken once, in the shipped member's
            // pass, and written to both members' rows.
            constexpr bool kFirst = kMember == 0;
            constexpr boys::FitGranularity kOther =
                kGranularity == boys::FitGranularity::kCoarsest ? boys::FitGranularity::kNarrow
                                                               : boys::FitGranularity::kCoarsest;

            for (std::size_t r = 0; r < kGranRowCount; ++r)
            {
                Accum& acc = GranularityClaims()[static_cast<std::size_t>(
                    granSlots[granIndex(kMember, kSchemeIndex, r)])];
                const std::size_t index = granIndex(kMember, kSchemeIndex, r);

                for (int n = 0; n <= nmax; ++n)
                {
                    // The seed row is one value per argument - order 0, the
                    // value the piece is the seed for - so it is read once
                    // rather than once per order.
                    if (granRows[r].kind == GranKind::kSeedB && n != 0)
                    {
                        continue;
                    }

                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const double x = ref.x[i];
                        const std::size_t k = ref.Index(n, i);
                        double got = 0.0;
                        double other = 0.0;

                        switch (granRows[r].kind)
                        {
                        case GranKind::kFitsA:
                        {
                            if (x >= boys::detail::kX0)
                            {
                                continue;
                            }

                            got = granStoredFit.template operator()<kScheme, kGranularity>(
                                boys::EvalLane::kRegionA, n, x);
                            other = granStoredFit.template operator()<kScheme, kOther>(
                                boys::EvalLane::kRegionA, n, x);
                            break;
                        }

                        case GranKind::kSeedB:
                        {
                            if (x < boys::detail::kX0 || x >= boys::detail::kX1)
                            {
                                continue;
                            }

                            got = granStoredFit.template operator()<kScheme, kGranularity>(
                                boys::EvalLane::kRegionB, 0, x);
                            other = granStoredFit.template operator()<kScheme, kOther>(
                                boys::EvalLane::kRegionB, 0, x);
                            break;
                        }

                        case GranKind::kOrdersBelow:
                        {
                            if (x >= boys::detail::kExtendedBX0)
                            {
                                continue;
                            }

                            got = ordersGrid[kMember][k];
                            other = ordersGrid[1 - kMember][k];
                            break;
                        }

                        case GranKind::kOrdersA:
                        case GranKind::kPlaneA:
                        {
                            if (x >= boys::detail::kX0)
                            {
                                continue;
                            }

                            const bool orders = granRows[r].kind == GranKind::kOrdersA;
                            got = orders ? ordersGrid[kMember][k] : planeGrid[kMember][k];
                            other = orders ? ordersGrid[1 - kMember][k] : planeGrid[1 - kMember][k];
                            break;
                        }

                        case GranKind::kOrdersB:
                        {
                            // Region B's row: the batch entry seeds its upward recursion from
                            // the order-0 seed and carries it to the top order, so the seed's own
                            // partition is amplified here by A_B(n).
                            if (x < boys::detail::kX0 || x >= boys::detail::kX1)
                            {
                                continue;
                            }

                            got = ordersGrid[kMember][k];
                            other = ordersGrid[1 - kMember][k];
                            break;
                        }

                        case GranKind::kSingleA:
                        case GranKind::kSingleWhole:
                        {
                        // The published table changes cell at the band's left edge: below it a
                        // piece is documented at the lane's own 1e-15, above it the band's 3e-14
                        // holds. So this row is the piece read at its own size under its own bar.
                            if (granRows[r].kind == GranKind::kSingleA &&
                                x >= boys::detail::kExtendedBX0)
                            {
                                continue;
                            }

                            got = singleGrid[kMember][k];
                            other = singleGrid[1 - kMember][k];
                            break;
                        }

                        case GranKind::kFixedN:
                        {
                            got = fixednGrid[kMember][k];
                            other = fixednGrid[1 - kMember][k];
                            break;
                        }
                        }

                        ++granCells[index];

                        if (std::memcmp(&got, &other, sizeof(double)) != 0)
                        {
                            if constexpr (kFirst)
                            {
                                ++granDiffer[index];
                                ++granDiffer[granIndex(1 - kMember, kSchemeIndex, r)];
                            }
                        }

                        // The bar this cell is judged at: the row's own figure,
                        // or the single-order lane's per-region formula.
                        const double bar = (granRows[r].bar == GranBar::kFixed)
                                               ? granRows[r].bound
                                               : SingleBound(x);

                        MeasureAt(acc,
                                  n,
                                  x,
                                  got,
                                  ref.v[k],
                                  ref.decade[k],
                                  bar,
                                  Unrepresentable(got, -1022));
                    }
                }
            }
        };

        measure.template operator()<boys::FitGranularity::kCoarsest>();
        measure.template operator()<boys::FitGranularity::kNarrow>();
    };

    // Both schemes, in the order the tables above are addressed.
    sweepGranularity.template operator()<0, boys::EvalScheme::kSplitClenshaw>();
    sweepGranularity.template operator()<1, boys::EvalScheme::kHorner>();

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
    // Through the last slot: the native half lane's two entries are created after
    // the static lanes, and a slot with no points (a lane this revision does not
    // carry) prints as a row of zeros rather than silently missing from the table.
    const int lastClaim = static_cast<int>(Claims().size());

    for (int i = firstClaim; i < lastClaim; ++i)
    {
        PrintClaim(Claims()[static_cast<std::size_t>(i)]);
    }

    std::printf("\n  delivered / claimed = |F_hat(n,x) - F(n,x)| and the bound it was judged\n"
                "             against, at the worst cell of the sweep; a claim whose bound is\n"
                "             per-region or per-value (the half lanes' half-ULP term) shows the\n"
                "             pair at that cell, not the base bound\n"
                "  vacuous  = points where the documented bound is at least as large as\n"
                "             |F_n(x)| itself, so any returned value in range passes\n"
                "  no value = of those, points where the lane returned zero or a subnormal\n");

    // ---- the entry book: the batched entries on the uniform partition --------
    // The books above reach their cells through one entry per lane, so a combination the axes
    // admit and an entry refuses has no cell in any of them. Four batched entries are crossed with
    // the uniform partition, each measured, refused and owed, or not applicable to its shape.
    enum class BatchEntry : std::uint8_t {
        kPlane,       // BoysAllN: every order over an array of arguments
        kPlaneSorted, // BoysAllN, the BoysSortedArgs overload
        kAtOrders,    // BoysAllNAtOrders: each argument's own top order
        kFixedN,      // BoysFixedN: one order over the same array
    };

    // The state one combination is in. Exactly one of the three, and every
    // combination carries one: a row that fits none of them is not a state this
    // book may leave a combination in.
    enum class EntryState : std::uint8_t {
        kMeasured,   // the entry serves the call, and the sweep below ran it
        kRefused,    // the entry's body refuses the policy where it is named
        kShapeLimit, // the call shape cannot form the axis at all
    };

    struct EntryBody {
        const char* name;
        BatchEntry kind;
    };

    // The entry list, written here by hand and labelled as such: the library has no accessor that
    // names its entries, so this list is the coverage this book claims. Every name in it is a public
    // entry of boys.hpp, and the states are decided off the entry's own branch, not off this list.
    constexpr std::array<EntryBody, 4> entryBodies{{
        {"plane entry", BatchEntry::kPlane},
        {"plane entry, sorted", BatchEntry::kPlaneSorted},
        {"per-element tops", BatchEntry::kAtOrders},
        {"fixed-order entry", BatchEntry::kFixedN},
    }};

    constexpr std::size_t kEntryBodyCount = std::size(entryBodies);

    struct EntryBookRow {
        std::string axes;
        const char* entry = "";
        // The three selections the row names, taken from the library's own
        // tables (the route names, the scheme names, the axis names), so a table
        // that renames a member renames the row with it.
        std::string route;
        std::string scheme;
        std::string axis;
        BatchEntry kind = BatchEntry::kPlane;
        EntryState state = EntryState::kShapeLimit;
        const char* reason = "";
        double bound = 0.0;
        int slot = -1;
    };

    // The assertion a shape-limit row prints, quoted from the library (boys_impl.hpp,
    // BoysFixedNImpl, :3209) rather than written here, so a reader of such a row reads the reason
    // the library states where the call is named.
    constexpr const char* kEntryOrdersShapeLimit =
        "the orders axis cannot be formed on this entry: a packed lane keeps four "
        "orders of one argument in a register, and this call produces exactly one "
        "order at every argument of the array, so there are not four orders here to "
        "fill a lane with - the wide dimension it does have is count, and that is the "
        "arguments axis"
        " (boys_impl.hpp, BoysFixedNImpl, the assertion at :3209)";

    // What a refused row prints here is the library's own assertion and nothing else. The six
    // configure probes that stood here went because these rows call the entries: a revision that put
    // one of those guards back would fail to build this file rather than print a refusal.

    // One combination at one policy. The refused and shape-limit states are
    // decided by `if constexpr` on the entry's own branch condition - the same two
    // policy members the entries test - and the served ones are run.
    const auto sweepEntryBook =
        [&]<boys::FitRoute kRoute, boys::EvalScheme kScheme, boys::PackAxis kAxis>(
            BatchEntry kind, EntryBookRow& row) {
            using Policy = boys::EvalPolicy<kRoute, kScheme, boys::BoysBudget::kFloat, kAxis,
                                            boys::FitGranularity::kUniform>;

            const auto open = [&row]() {
                row.state = EntryState::kMeasured;
                row.slot = AddEntryClaim(row.entry, "uniform grid, whole committed grid",
                                         kBoundDoubleBatch);
            };

            switch (kind)
            {
            case BatchEntry::kPlane:
            case BatchEntry::kPlaneSorted:
                // Every combination of this entry is served at this revision, the uniform
                // partition among them. The partitioned path is the shipped route's arguments axis
                // and carries no uniform branch - its region bodies reach ChebyshevFit<scheme,
                // kUniform>, whose else branch is NARROW - so the per-argument path takes it.
            {
                open();

                const bool sorted = kind == BatchEntry::kPlaneSorted;
                const std::vector<double>& args = sorted ? refSorted : ref.x;
                std::vector<double> planes(count * (static_cast<std::size_t>(nmax) + 1));

                if (sorted)
                {
                    boys::BoysAllN< Policy>(nmax, args.data(), planes.data(), count,
                                                boys::BoysSortedArgs{});
                }
                else
                {
                    boys::BoysAllN< Policy>(nmax, args.data(), planes.data(), count);
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
                        const double got = planes[static_cast<std::size_t>(n) * count + j];

                        MeasureAt(EntryClaims()[static_cast<std::size_t>(row.slot)],
                                  n,
                                  ref.x[i],
                                  got,
                                  ref.v[k],
                                  ref.decade[k],
                                  kBoundDoubleBatch,
                                  Unrepresentable(got, -1022));
                    }
                }
            }

            break;

            case BatchEntry::kAtOrders:
            {
                // No guard reaches this entry at any route, scheme or axis: it hands every
                // element to BoysAllOrdersImpl at that element's own top order, and that body
                // reads the grid where the policy names it. The row is measured, not excused.
                open();

                std::vector<int> tops(count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    tops[i] = nmax - static_cast<int>(i % static_cast<std::size_t>(nmax + 1));
                }

                std::vector<double> planes(count * (static_cast<std::size_t>(nmax) + 1));
                boys::BoysAllNAtOrders< Policy>(
                    tops.data(), ref.x.data(), planes.data(), count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    for (int n = 0; n <= tops[i]; ++n)
                    {
                        const std::size_t k = ref.Index(n, i);
                        const double got = planes[k];

                        MeasureAt(EntryClaims()[static_cast<std::size_t>(row.slot)],
                                  n,
                                  ref.x[i],
                                  got,
                                  ref.v[k],
                                  ref.decade[k],
                                  kBoundDoubleBatch,
                                  Unrepresentable(got, -1022));
                    }
                }

                break;
            }

            case BatchEntry::kFixedN:
                if constexpr (kAxis == boys::PackAxis::kOrders)
                {
                    // The shape limit, and it is the entry's own assertion rather
                    // than a partition's: this call produces one order at every
                    // argument, so there are not four orders on it to fill a lane
                    // with. Naming the grid beside it changes nothing about that.
                    row.state = EntryState::kShapeLimit;
                    row.reason = kEntryOrdersShapeLimit;
                }
                else
                {
                    // Every remaining combination of this entry is served at this revision, the
                    // uniform partition among them. The Chebyshev branch resolves the partition to
                    // ChebyshevFit, whose else branch is NARROW, so the entry hands a uniform policy
                    // to BoysSingleImpl, which reads the grid below the join.
                    open();

                    std::vector<double> values(count);

                    for (int n = 0; n <= nmax; ++n)
                    {
                        boys::BoysFixedN< Policy>(
                            n, ref.x.data(), values.data(), count);

                        for (std::size_t i = 0; i < count; ++i)
                        {
                            const std::size_t k = ref.Index(n, i);
                            const double got = values[i];

                            MeasureAt(EntryClaims()[static_cast<std::size_t>(row.slot)],
                                      n,
                                      ref.x[i],
                                      got,
                                      ref.v[k],
                                      ref.decade[k],
                                      kBoundDoubleBatch,
                                      Unrepresentable(got, -1022));
                        }
                    }
                }

                break;
            }
        };

    // The arms this book writes its policies from: the two routes, two schemes and two packing
    // axes the entries' own branches test, reconciled in both directions against the tables that
    // report those axes. An arm no table names cannot be swept, and a table member no arm covers
    // would be left unswept; either turns the run red.
    struct EntryArm {
        boys::FitRoute route;
        boys::EvalScheme scheme;
        boys::PackAxis axis;
    };

    const std::array<EntryArm, 8> entryArms{{
        {boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw, boys::PackAxis::kArguments},
        {boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw, boys::PackAxis::kOrders},
        {boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner, boys::PackAxis::kArguments},
        {boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner, boys::PackAxis::kOrders},
        {boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
         boys::PackAxis::kArguments},
        {boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
         boys::PackAxis::kOrders},
        {boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner, boys::PackAxis::kArguments},
        {boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner, boys::PackAxis::kOrders},
    }};

    // The name each table reports for one arm's route and one arm's axis: the
    // rows are labelled with the library's own names, and an arm the table does
    // not name comes back empty and is counted as such.
    const auto entryArmRouteName = [](boys::FitRoute route) {
        for (const boys::FitRouteInfo& tableRow : boys::BoysFitRoutes())
        {
            if (tableRow.route == route)
            {
                return std::string(tableRow.name);
            }
        }

        return std::string();
    };

    const auto entryArmAxisName = [](boys::PackAxis axis) {
        for (const boys::PackAxisInfo& tableRow : boys::BoysPackAxes())
        {
            if (tableRow.axis == axis)
            {
                return std::string(tableRow.name);
            }
        }

        return std::string();
    };

    std::size_t entryArmsUnnamed = 0;

    for (const EntryArm& arm : entryArms)
    {
        if (entryArmRouteName(arm.route).empty() || entryArmAxisName(arm.axis).empty())
        {
            ++entryArmsUnnamed;
        }
    }

    // The other direction: what the tables name and no arm covers. A route the
    // table names twice is one route, so the route walk de-duplicates the way the
    // cross below does.
    std::size_t entryTableUncovered = 0;
    std::string entryTableUncoveredNames;
    std::vector<boys::FitRoute> entryTableRoutes;

    for (const boys::FitRouteInfo& tableRow : boys::BoysFitRoutes())
    {
        bool seen = false;

        for (const boys::FitRoute known : entryTableRoutes)
        {
            seen = seen || known == tableRow.route;
        }

        if (seen)
        {
            continue;
        }

        entryTableRoutes.push_back(tableRow.route);

        bool covered = false;

        for (const EntryArm& arm : entryArms)
        {
            covered = covered || arm.route == tableRow.route;
        }

        if (!covered)
        {
            ++entryTableUncovered;
            entryTableUncoveredNames += Fmt(" %s", tableRow.name);
        }
    }

    for (const boys::PackAxisInfo& tableRow : boys::BoysPackAxes())
    {
        bool covered = false;

        for (const EntryArm& arm : entryArms)
        {
            covered = covered || arm.axis == tableRow.axis;
        }

        if (!covered)
        {
            ++entryTableUncovered;
            entryTableUncoveredNames += Fmt(" %s", tableRow.name);
        }
    }

    for (const boys::EvalSchemeInfo& tableRow : boys::BoysEvalSchemes())
    {
        bool covered = false;

        for (const EntryArm& arm : entryArms)
        {
            covered = covered || arm.scheme == tableRow.scheme;
        }

        if (!covered)
        {
            ++entryTableUncovered;
            entryTableUncoveredNames += Fmt(" %s", tableRow.name);
        }
    }

    std::vector<EntryBookRow> entryBook;

    for (const EntryBody& body : entryBodies)
    {
        for (std::size_t a = 0; a < entryArms.size(); ++a)
        {
            const EntryArm& arm = entryArms[a];
            EntryBookRow row;
            row.entry = body.name;
            row.route = entryArmRouteName(arm.route);
            row.scheme = boys::EvalSchemeName(arm.scheme);
            row.axis = entryArmAxisName(arm.axis);
            row.kind = body.kind;
            row.bound = kBoundDoubleBatch;
            row.axes = Fmt("%s, %s, %s, %s, uniform",
                           body.name,
                           row.route.c_str(),
                           row.scheme.c_str(),
                           row.axis.c_str());

            // The dispatch: one arm per combination of the three selections the entries' branches
            // test, flat rather than nested, so that a member added to any axis is an arm this list
            // does not have rather than a nested fall-through measuring it under another's policy.
            switch (a)
            {
            case 0:
                sweepEntryBook.template operator()<boys::FitRoute::kChebyshev,
                                                    boys::EvalScheme::kSplitClenshaw,
                                                    boys::PackAxis::kArguments>(body.kind, row);

                break;
            case 1:
                sweepEntryBook.template operator()<boys::FitRoute::kChebyshev,
                                                    boys::EvalScheme::kSplitClenshaw,
                                                    boys::PackAxis::kOrders>(body.kind, row);

                break;
            case 2:
                sweepEntryBook.template operator()<boys::FitRoute::kChebyshev,
                                                    boys::EvalScheme::kHorner,
                                                    boys::PackAxis::kArguments>(body.kind, row);

                break;
            case 3:
                sweepEntryBook.template operator()<boys::FitRoute::kChebyshev,
                                                    boys::EvalScheme::kHorner,
                                                    boys::PackAxis::kOrders>(body.kind, row);

                break;
            case 4:
                sweepEntryBook.template operator()<boys::FitRoute::kRationalMinimax,
                                                    boys::EvalScheme::kSplitClenshaw,
                                                    boys::PackAxis::kArguments>(body.kind, row);

                break;
            case 5:
                sweepEntryBook.template operator()<boys::FitRoute::kRationalMinimax,
                                                    boys::EvalScheme::kSplitClenshaw,
                                                    boys::PackAxis::kOrders>(body.kind, row);

                break;
            case 6:
                sweepEntryBook.template operator()<boys::FitRoute::kRationalMinimax,
                                                    boys::EvalScheme::kHorner,
                                                    boys::PackAxis::kArguments>(body.kind, row);

                break;
            case 7:
                sweepEntryBook.template operator()<boys::FitRoute::kRationalMinimax,
                                                    boys::EvalScheme::kHorner,
                                                    boys::PackAxis::kOrders>(body.kind, row);

                break;
            default:
                // An arm with no case above: the row is not built, and the arithmetic below
                // counts the rows that exist against the arms the tables report, so such an arm
                // turns the run red instead of printing a state the row does not have.
                continue;
            }

            entryBook.push_back(std::move(row));
        }
    }

    // The book's own counters, taken here rather than where the rows are printed
    // because the combination block below reads them into the run's arithmetic.
    // A row is in exactly one of the three states and the three add to the book's
    // own total: that is the arithmetic the entry factor is carried by.
    std::size_t entryBookMeasured = 0;
    std::size_t entryBookRefused = 0;
    std::size_t entryBookShapeLimit = 0;

    for (const EntryBookRow& row : entryBook)
    {
        switch (row.state)
        {
        case EntryState::kMeasured:
            ++entryBookMeasured;

            break;
        case EntryState::kRefused:
            ++entryBookRefused;

            break;
        case EntryState::kShapeLimit:
            ++entryBookShapeLimit;

            break;
        }
    }

    // The same total a second way, off the sizes the tables report. A member
    // added to any of the three axes the library reports moves this and the
    // arms together; a row the enumeration above drops moves only one of them,
    // and the difference is what the combination block fails on.
    const std::size_t entryBookRows = entryBook.size();
    const std::size_t entryBookClaimed = kEntryBodyCount * entryTableRoutes.size() *
                                         boys::BoysEvalSchemes().size() *
                                         boys::BoysPackAxes().size();

    // ---- the published cells and ranges, measured ---------------------------
    std::printf("\npublished cells and ranges, measured:\n");

#ifdef BOYS_GATE_FP16
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
#else
    std::printf("  half worst cell (n=0, x=721): not carried by this build (BoysFp16 = 0)\n"
                "  half lane, sweep worst      : not carried by this build (BoysFp16 = 0)\n"
                "  half lane range, region C   : not measured here - the lane's columns below\n"
                "    would be a lane's own returns, and this build has no half entry to call.\n"
                "    The reference grid's columns of the same statement are:\n");
    std::printf("    %5s %14s %14s %14s\n",
                "order",
                "ref normal to",
                "ref zero to",
                "scaled limit");

    for (int n = 0; n <= nmax; ++n)
    {
        if (n > 8 && n != nmax)
        {
            continue;
        }

        std::printf("    %5d %14.6g %14.6g %14.6g\n",
                    n,
                    refNormalX[static_cast<std::size_t>(n)],
                    refZeroX[static_cast<std::size_t>(n)],
                    refSpanNormalX[static_cast<std::size_t>(n)]);
    }

    std::printf("    (scaled limit = the largest argument at which the order's whole ladder,\n"
                "     seed included, still fits what a per-order power-of-two scale can hold in\n"
                "     binary16's normal range; the subnormal-field limit is, per order, %s)\n",
                (refSpanSubnormalX[8] > 0.0 ? "printed for order 8 below" : "-"));
#endif // BOYS_GATE_FP16

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

    // The half lanes' two sides, totalled across the cells a claim covers: the domain where the
    // bound is tighter than the value, and the return's state inside it. A subnormal return inside
    // the bound is allowed - the claim is about the error, not the exponent - but a zero return is
    // not, and neither is a value outside the bound.
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
        "double single: 1e-15 on A, 3e-14 on the extended band and B, 5.5e-14 on C",
        "README accuracy contract (four-region table)",
        verdictOf({kSingleA, kSingleBand, kSingleB, kSingleC}),
        worstOf({kSingleA, kSingleBand, kSingleB, kSingleC}));

    add("README.double.batch",
        "double batch: 5.5e-14 in every region, all entries",
        "README accuracy contract (four-region table)",
        verdictOf({kOrders, kFixedN, kAllN, kAllNAtOrders}),
        worstOf({kOrders, kFixedN, kAllN, kAllNAtOrders}));

    add("README.float",
        "float single and batch: 1.5e-7 absolute, the same in every region",
        "README accuracy contract",
        verdictOf({kFloatSingle, kFloatOrders, kFloatOrdersPacked, kFloatAllN}),
        worstOf({kFloatSingle, kFloatOrders, kFloatOrdersPacked, kFloatAllN}));

    add("README.half",
        "fp16 and bf16 store-half lanes: 1.5e-7 + one half-ULP, single and batch - the float "
        "lane's figure, which is the arithmetic these lanes run, plus the half-ULP term their "
        "store adds",
        "README accuracy contract and include/boys/boys.hpp",
#ifdef BOYS_GATE_FP16
        verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}) == Verdict::Verified
            ? Verdict::MetOverDomain
            : verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        worstOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        "the arguments where |F_n(x)| > 1e-7 + one half-ULP of the returned value, and no "
        "others - the restriction the half lanes' paragraph in docs/lane-contract.md states. "
        "The row judges at that base, which is the region target the half lanes' fits are cut "
        "for and is tighter than the 1.5e-7 the library publishes for them, so its domain is "
        "wider than the published claim's and a pass here is the stronger result",
        "the row fails if the budget is read as claimed over the whole argument range: the "
        "counted-apart cells in LC.half.vacuous_floor are exactly the cells that reading would "
        "turn into failures");
#else
        Verdict::NotCarried,
        "no lane was called and no cell was measured: the two lanes are declared behind the "
        "BoysFp16 seam, which this build has closed, so the contract this row is about has no "
        "subject here. The claim is not withdrawn - it is carried by the builds that close "
        "nothing, and this one neither confirms nor denies it",
        {},
        "the row's subject is the fp16 and bf16 lanes; a build that carries neither has no cell "
        "of it to judge, which is reported rather than counted as a pass");
#endif // BOYS_GATE_FP16

    {
        // The header's contract table as it stands: the double single lane's
        // region-A cell (1e-15 over the per-order fits' own range) and its
        // extended-band cell (3e-14), the two rows the table does not
        // collapse, measured against the slots that hold them.
        const Accum& headerA = Claims()[static_cast<std::size_t>(kHeaderA)];

        add("header.double.region_A_and_band",
            "double single: 1e-15 on region A and 3e-14 on the extended band, the two "
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

    add("LC.double.ceiling",
        "the stored coefficients are correctly rounded and the fit carries 5e-19 of "
        "truncation under doubles holding 1e-16 to 2e-16, so no fit goes below that floor",
        "docs/lane-contract.md, double",
        Verdict::Verified,
        "tree command, run by the operator, not by this binary: "
        "`python tools/gen_boys_coefficients.py --check`");

    add("LC.float.ceiling",
        "the ceiling is the format: a 24-bit significand resolves about 6e-8 relative and "
        "the budget is 1.5e-7 absolute - within a factor of a few of each other",
        "docs/lane-contract.md, float",
        Verdict::Verified,
        Fmt("measured floor: worst relative error of the float lane where |F| >= 0.5 is "
            "%.3g at (n=%d, x=%.6g); the ratio 1.5e-7 / 6e-8 = 2.5 is the documented "
            "'factor of a few'",
            floatWorstRel,
            floatWorstRelN,
            floatWorstRelX));

    // ---- the region-A transform lane ---------------------------------------
    // The six rows are that lane's published accuracy claims: three carry the bound a caller is
    // held to and three the delivered worst the documents publish. The two split modes' rows say
    // again that their bounds idealise a 32-bit tensor-core accumulator, measured here in software.
    {
        const Accum& fp64 = Claims()[static_cast<std::size_t>(kTransformFp64)];
        const Accum& tf32 = Claims()[static_cast<std::size_t>(kTransformTf32x3)];
        const Accum& bf16 = Claims()[static_cast<std::size_t>(kTransformBf16x6)];
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
            "the region-A transform's fp64 mode: |F_hat - F| <= 1e-15 over region A - the "
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
            "the region-A transform's 3xTF32 mode: |F_hat - F| <= 1e-15 + 2.5e-7 over "
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
            "the region-A transform's bf16x6 mode: |F_hat - F| <= 1e-15 + 2.5e-7 over "
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

    }

    add("LC.half.budget",
        "the half lane stays inside its budget at every order tested",
        "docs/lane-contract.md, half",
#ifdef BOYS_GATE_FP16
        verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}) == Verdict::Verified
            ? Verdict::MetOverDomain
            : verdictOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        worstOf({kF16Single, kF16Orders, kBf16Single, kBf16Orders}),
        "the arguments where |F_n(x)| > 1e-7 + one half-ULP of the returned value, and no "
        "others - the base the half lanes' fits are cut for, tighter than the 1.5e-7 the "
        "library publishes for them",
        "the row fails if a cell inside that domain delivers more than the bound, or if the "
        "domain's edge moves down to arguments where the return is the format's floor: the "
        "counts are in LC.half.vacuous_floor");
#else
        Verdict::NotCarried,
        "not measured: the lane this row is about is declared behind the BoysFp16 seam, which "
        "this build has closed, so there is no entry to call and no cell to judge",
        {},
        "the row's subject is the fp16 and bf16 lanes; a build that carries neither reports the "
        "claim as one it does not carry rather than passing it");
#endif // BOYS_GATE_FP16

#ifdef BOYS_GATE_FP16
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
#else
    // The published cell, the sweep's worst cell and their margin are all readings
    // of the half lane, so a build whose seam is closed has none of them. The row
    // stays in the book with the reason stated: a reader of this build is told the
    // claim is not carried here, and never that it was measured and held.
    add("LC.half.worst_cell",
        "[corrected this revision] the lane's worst cell is not 0.990 of budget at order 0, "
        "x = 721 with a margin of 1%: that cell reproduces, but the sweep's own worst cell "
        "is within a thousandth of the budget in both formats, so the margin is thousandths "
        "of a per cent - a fragility, not a margin of 1%",
        "docs/lane-contract.md, half (read before this revision: \"a worst cell of 0.990 of "
        "budget at order 0, x = 721 - a margin of 1%\")",
        Verdict::NotCarried,
        "not measured: neither the published cell nor either format's sweep exists in this "
        "build, because the half lane is declared behind the BoysFp16 seam it has closed. The "
        "row's claim is about a lane, and this build carries no lane to measure it on",
        {},
        "the row's subject is the fp16 and bf16 lanes; a build that carries neither reports the "
        "claim as one it does not carry rather than passing it");
#endif // BOYS_GATE_FP16

    add("LC.half.representation_share",
        "at that magnitude the half-ULP representation term is 99% of the budget",
        "docs/lane-contract.md, half",
#ifdef BOYS_GATE_FP16
        std::abs(cellShare - 0.99) <= 0.02 ? Verdict::Verified : Verdict::Exceeded,
        Fmt("measured %.4f at (n=0, x=721)", cellShare));
#else
        Verdict::NotCarried,
        "not measured: the share is the representation term of a value the half lane returned "
        "at x = 721, and this build carries no half lane to return it");
#endif // BOYS_GATE_FP16

    add("LC.half.vacuous_floor",
        "the half lanes' ceiling is a domain restriction: past it the return is the format's "
        "floor, no accuracy is claimed, and the points are counted rather than passed",
        "docs/lane-contract.md, half; include/boys/boys.hpp, the same paragraph",
#ifdef BOYS_GATE_FP16
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
        "the arguments where |F_n(x)| > 1e-7 + one half-ULP of the returned value - the "
        "domain a base the half lanes' fits are cut for carves, tighter than the 1.5e-7 the "
        "library publishes and so wider than the published claim, and no others",
        "the row fails if the document ever reads as claiming accuracy past that ceiling - a "
        "claim over the whole argument range would make every point past the ceiling a "
        "failure, and the counters above are the number of them; it also fails if a point "
        "past the ceiling returns a usable value the count does not carry");
#else
        Verdict::NotCarried,
        "not measured: the counted cells are returns of the fp16 and bf16 lanes, which this "
        "build does not carry - its BoysFp16 seam is closed - so there is no return to count "
        "on either side of the ceiling. The domain restriction the row states is a property of "
        "those lanes and is neither confirmed nor denied here",
        {},
        "the row's subject is the fp16 and bf16 lanes' returns past their ceiling; a build that "
        "carries no such lane reports the claim as one it does not carry");
#endif // BOYS_GATE_FP16

    // The half lane binds where |F_n(x)| exceeds its ceiling, and F_n falls with x, so region C
    // is entirely past the ceiling for an order exactly when the branch's left edge is - the onset
    // order the paragraph claims. Every return at that magnitude is subnormal.
#ifdef BOYS_GATE_FP16
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
#else
    // The onset is read off the lane's own returns, so a build whose seam is
    // closed has no onset to report. The reference side of the same statement is
    // still readable - it is the grid's - and LC.half.no_scaling_past_38 below
    // measures it, which is why that row stays unguarded.
    add("LC.half.range_from_order3",
        "from order 3 upward on region C every return is a subnormal half or zero, and from "
        "order 6 upward every argument of the branch is past the ceiling, so no accuracy is "
        "claimed there. Orders 3, 4 and 5 are inside the range the lane does claim over: "
        "they still carry cells whose value exceeds the ceiling",
        "docs/lane-contract.md, half",
        Verdict::NotCarried,
        "not measured: the onset order is the first order whose every region-C return is "
        "subnormal or zero, and this build has no half lane to return one - its BoysFp16 seam "
        "is closed. The reference side of the same statement is measured by "
        "LC.half.no_scaling_past_38",
        {},
        "the row's subject is the fp16 lane's own returns; a build that carries no such lane "
        "reports the claim as one it does not carry");
#endif // BOYS_GATE_FP16

    // Withdrawn: "a per-order power-of-two scale carries order 3 to x ~ 361, order 4 to 129, order
    // 8 to 30". The reaches rest on a prototype that is not in this tree; the ceiling any such scale
    // runs into is what LC.half.no_scaling_past_38 and LC.half.range_from_order3 measure instead.

    add("LC.half.no_scaling_past_38",
        "beyond about x ~ 35-38 no scaling reaches order 8 at all",
        "docs/lane-contract.md, half (evidence: the same prototype)",
        (refSpanNormalX[8] >= 30.0 && refSpanNormalX[8] <= 50.0) ? Verdict::Verified
                                                                 : Verdict::Exceeded,
        Fmt("computed from the reference, no prototype needed: the ladder span F_0/F_8 exceeds "
            "binary16's normal field at x=%.6g and its whole subnormal field at x=%.6g",
            refSpanNormalX[8],
            refSpanSubnormalX[8]));

    // Withdrawn: "packed half ... 2.39x its budget at order 0 failing on 25.1% of the branch,
    // 2.35x at order 1 (5.7%)" - the prototype's numbers, and the prototype is not in this tree.
    // LC.packed.bound below measures the native packed lane that does ship, against its own ceiling.

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
#ifdef BOYS_GATE_FP16
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
            "from the contract test. Which side those cells put outside its bound, which the "
            "aggregate does not say: %zu have the body outside and the scalar inside (A fp16 "
            "%zu, A bf16 %zu, B fp16 %zu, B bf16 %zu, C fp16 %zu, C bf16 %zu), the furthest "
            "over %.4g of its own bound, in region %d on %s, and %zu have the scalar outside "
            "and the body inside (A fp16 %zu, A bf16 %zu, B fp16 %zu, B bf16 %zu, C fp16 %zu, "
            "C bf16 %zu), the furthest over %.4g, in region %d on %s; the largest argument of "
            "each set is x=%.6g (n=%d) and x=%.6g (n=%d). What the out-of-bound return is, "
            "which is the difference between a tight bound and no value at all: of the body's "
            "set %zu returned a value that is not a number, %zu sit at a half argument that is "
            "not a finite half and %zu at a cell whose reference is exactly zero; of the "
            "scalar's set %zu returned a value that is not a number, %zu sit at a half "
            "argument that is not a finite half and %zu at a cell whose reference is exactly "
            "zero. The body's set is empty at this revision and the scalar's is not, so on "
            "every cell this row fails on the 8-wide body returns the reference's own value "
            "and the return outside the bound is the certified scalar entry's - which is what "
            "makes this row's subject and this row's evidence different objects, and is why "
            "the side is printed here rather than left in the aggregate",
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
            driftScalarZeroN,
            driftSimdOut,
            driftSimdOutCell[0],
            driftSimdOutCell[1],
            driftSimdOutCell[2],
            driftSimdOutCell[3],
            driftSimdOutCell[4],
            driftSimdOutCell[5],
            driftSimdOutRatio,
            driftSimdOutRegion,
            driftSimdOutBf16 ? "bf16" : "fp16",
            driftScalarOut,
            driftScalarOutCell[0],
            driftScalarOutCell[1],
            driftScalarOutCell[2],
            driftScalarOutCell[3],
            driftScalarOutCell[4],
            driftScalarOutCell[5],
            driftScalarOutRatio,
            driftScalarOutRegion,
            driftScalarOutBf16 ? "bf16" : "fp16",
            driftSimdOutX,
            driftSimdOutN,
            driftScalarOutX,
            driftScalarOutN,
            driftSimdOutNaN,
            driftSimdOutInfArg,
            driftSimdOutRefZero,
            driftScalarOutNaN,
            driftScalarOutInfArg,
            driftScalarOutRefZero));
#else
        Verdict::NotCarried,
        // The seam is the reason here and not the target: with it closed this build has neither
        // the 8-wide kernels nor the certified scalar half entry they are compared against, on any
        // target. The tier is printed anyway, because with the seam open it decides the row.
        Fmt("this build's BoysFp16 seam is closed, so it carries neither the 8-wide half-I/O "
            "region kernels nor the certified scalar half entry they are compared against: the "
            "claim has no subject in this build, and no cell of it was measured. The reference "
            "grid's fp16 and bf16 columns are read from the build but no lane is called to "
            "produce the values they are compared against. The AVX2 half tier is %s on this "
            "target, which is what would scope the row rather than judge it if the seam were "
            "open",
            kSimdTierTarget ? "compiled in" : "not compiled in"));
#endif // BOYS_GATE_FP16

    add("code.half_simd_cells",
        "the 8-wide half-I/O region kernels hold the half lane's budget on the region each "
        "documents as its precondition",
        "src/boys_simd.cpp, region-partitioned kernels",
#ifdef BOYS_GATE_FP16
        verdictOf({kF16PackedA, kF16PackedB, kF16PackedC, kBf16PackedA, kBf16PackedB, kBf16PackedC}),
        worstOf({kF16PackedA, kF16PackedB, kF16PackedC, kBf16PackedA, kBf16PackedB, kBf16PackedC}));
#else
        Verdict::NotCarried,
        "not measured: the six cells are the 8-wide half-I/O kernels' own, and this build "
        "carries no half entry and no half kernel - its BoysFp16 seam is closed. The claim has "
        "no subject here rather than a cell that passed");
#endif // BOYS_GATE_FP16

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
#ifdef BOYS_GATE_FP16
            halfDomain.points == 0
                ? Verdict::EvidenceAbsent
                : ((halfDomain.outside == 0 && halfDomain.zero == 0) ? Verdict::Verified
                                                                     : Verdict::Exceeded),
#else
            // The seam is closed, so the ten slots were never swept. The zero
            // their accumulations carry is this build's shape and not a domain
            // that came out empty, and the row says which of the two it is
            // rather than leaving a reader to guess from a zero.
            Verdict::NotCarried,
#endif
#ifdef BOYS_GATE_FP16
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
#else
            Fmt("not measured: the ten cells are cells of the fp16 and bf16 lanes, and this "
                "build carries neither - its BoysFp16 seam is closed - so the sweep produced "
                "no point on either side of the ceiling: the ten slots' accumulations hold %zu "
                "swept points between them, which is what a closed seam leaves and not a "
                "domain that came out empty. The domain restriction the claim states is a "
                "property of those lanes and is neither confirmed nor denied here",
                halfDomain.points));
#endif // BOYS_GATE_FP16
    }

    // The native packed half lane, in the shape its own contract uses: one domain, one bound, one
    // verdict each, measured against this gate's committed mpmath reference over the region-C
    // arguments the grid carries in binary16 - and the packing claim, which no accuracy oracle can
    // see, refuted rather than confirmed.
    {
        const Accum& packedAcc = Claims()[static_cast<std::size_t>(kNativeHalf2)];
        const Accum& batchAcc = Claims()[static_cast<std::size_t>(kNativeHalfBatch)];

        add("native.half.bound",
            "packed half (two per register, one correctly rounded half operation per ladder "
            "step): |out[k] - 2^15 F_k(x)| <= 8 ULP of the returned value, region C only, over "
            "the arguments whose returned value is a normal half",
            "docs/lane-contract.md, packed half - the native lane",
            packedAcc.points == 0
                ? kNativeHalfNotMeasured
                : ((packedAcc.failures == 0 && batchAcc.failures == 0) ? Verdict::MetOverDomain
                                                                      : Verdict::Exceeded),
            packedAcc.points == 0
                ? std::string(kNativeHalfAbsent)
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
                ? kNativeHalfNotMeasured
                : (nativeExcusedFailures == 0 ? Verdict::MetOverDomain : Verdict::Exceeded),
            nativeCells == 0
                ? std::string(kNativeHalfAbsent)
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
                ? kNativeHalfNotMeasured
                : ((nativeBatchMismatch == 0 && nativePastOneUlp > 0) ? Verdict::MetOverDomain
                                                                     : Verdict::Exceeded),
            nativeCells == 0
                ? std::string(kNativeHalfAbsent)
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
                ? kNativeHalfNotMeasured
                : (nativeWorstUlp > 0.5 ? Verdict::Verified : Verdict::Exceeded),
            nativeCells == 0
                ? std::string(kNativeHalfAbsent)
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
        // Region B's amplification at every supported order, in both readings: the tree's sentence
        // writes prod(j+1/2)/x0^l, which measures 65 at l = 32 and not one. The reading under which
        // "<= A_B(0) = 1" is even nearly true is the ladder-normalised prod(2j-1)/(2 x0)^l, which x0
        // makes exactly 1 at l = 32 - the independent oracle's reading, 1.846e-17 above one here.
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
        // What the sweep's arguments cover, per format, and the generator's rule that keeps a lane
        // from outgrowing the grid: every format narrower than binary64 is carried to its own largest
        // finite value, with every representable value in its last band where the spacing leaves one.
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

    // Every division form the axis carries, at region C's own budget. This was a pin - the
    // library's own value on region C had to be this spelling's, bit for bit - and it cannot be one
    // any more: the recurrence's division is a policy field. What a consumer relies on is that
    // every form the library offers is inside the branch's bound, which is what is measured here.
    std::size_t formChecked = 0;
    std::size_t formOutside = 0;
    double formWorst = 0.0;
    int formWorstN = 0;
    double formWorstX = 0.0;

    {
        using PExact = boys::EvalPolicy<boys::kDefaultFitRoute,
                                        boys::kDefaultEvalScheme,
                                        boys::BoysBudget::kFloat,
                                        boys::kDefaultPackAxis,
                                        boys::kDefaultFitGranularity,
                                        boys::DivisionForm::kExactDivision>;
        using PPlain = boys::EvalPolicy<boys::kDefaultFitRoute,
                                        boys::kDefaultEvalScheme,
                                        boys::BoysBudget::kFloat,
                                        boys::kDefaultPackAxis,
                                        boys::kDefaultFitGranularity,
                                        boys::DivisionForm::kPlainReciprocal>;
        using PRefined = boys::EvalPolicy<boys::kDefaultFitRoute,
                                          boys::kDefaultEvalScheme,
                                          boys::BoysBudget::kFloat,
                                          boys::kDefaultPackAxis,
                                          boys::kDefaultFitGranularity,
                                          boys::DivisionForm::kRefinedReciprocal>;

        const auto measure = [&](double got, int n, double x, std::size_t i) {
            ++formChecked;

            const double err = std::abs(got - ref.v[ref.Index(n, i)]);

            if (err > formWorst)
            {
                formWorst = err;
                formWorstN = n;
                formWorstX = x;
            }

            if (err > kBoundSingleC)
            {
                ++formOutside;
            }
        };

        for (int n = 0; n <= nmax; ++n)
        {
            for (std::size_t i = 0; i < count; ++i)
            {
                const double x = ref.x[i];

                if (x < boys::detail::kX1)
                {
                    continue;
                }

                measure(boys::BoysSingle<PExact>(n, x), n, x, i);
                measure(boys::BoysSingle<PPlain>(n, x), n, x, i);
                measure(boys::BoysSingle<PRefined>(n, x),
                        n,
                        x,
                        i);
            }
        }
    }

    add("LC.regionC.domain",
        "[corrected this revision] the branch's lower boundary is not a floor: through x = 16 "
        "the error varies smoothly, with no threshold and no step of orders of magnitude, "
        "growing continuously as x falls below the edge",
        "docs/lane-contract.md, region C (read before this revision: \"the error jumps five to "
        "eight orders - a hard floor, not a taper\")",
        (asymStepRatioMax < 10.0 && formOutside == 0) ? Verdict::Verified : Verdict::Exceeded,
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
            "(%.1f to %.1f orders over orders 0..24). The branch is evaluated here through "
            "the library at every division form the axis carries: %zu region-C cell(s) over "
            "the three forms, %zu of them outside the branch's bound, the worst %.6g at "
            "(n=%d, x=%.17g); the shape above is the form's error and "
            "not the stored fit's on every host, which it was not before this revision - a "
            "host without AVX2 has no vector body to reach the form through, so the same "
            "call measured the region-B fit there and reported its smoothness as the form's",
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
            std::log10(asymBelowRelMax),
            formChecked,
            formOutside,
            formWorst,
            formWorstN,
            formWorstX),
        {},
        Fmt("the row fails if any pair of neighbouring arguments below the edge differs in "
            "relative error by more than an order of magnitude - that is what a threshold "
            "between two regimes would look like, and this measurement's largest such factor is "
            "%.4g; the strictly monotone reading of the old sentence would fail on the %zu "
            "breaks the same measurement counts (%.1f%% of the pairs below the edge, %.1f%% at "
            "the top order); or if the form this row evaluates stops agreeing with the "
            "library's own value for it on region C - the two counts at the end of the figure "
            "are that pin, and it is what makes this row a measurement of the library's form "
            "rather than of a transcription of it",
            asymStepRatioMax,
            asymBreaks,
            100.0 * static_cast<double>(asymBreaks) / static_cast<double>(asymPairs == 0 ? 1 : asymPairs),
            100.0 * static_cast<double>(asymBreaksN[static_cast<std::size_t>(nmax)])
                / static_cast<double>(asymPairsN[static_cast<std::size_t>(nmax)] == 0
                                          ? 1
                                          : asymPairsN[static_cast<std::size_t>(nmax)])));

    add("LC.evidence.ctest",
        "the documented budgets are re-checkable with ctest",
        "docs/lane-contract.md, where the numbers come from",
        Verdict::Verified,
        "tree command, run by the operator, not by this binary: "
        "`ctest --test-dir build --output-on-failure`");

    // Withdrawn: "orders 0 through 3 are 95.2% of the calls measured in a production integral
    // engine". The counts are not in this tree, and a distribution measured elsewhere describes the
    // interface that produced it rather than this library's argument range.

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
    int notCarried = 0;
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
        case Verdict::NotCarried:
            ++notCarried;
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
                "absent, %d not carried by this build; none of the last four counted as met)\n",
                verified + metOverDomain,
                book.size(),
                verified,
                metOverDomain,
                exceeded,
                vacuousOnly,
                absent,
                notCarried);

    // The rows this build does not carry, named in full with the reason: a row that vanishes with
    // the seam is how a claim stops being checked without anyone deciding to stop. Each prints the
    // same reason in place of a measurement, and none of them is a pass.
    if (notCarried > 0)
    {
        std::printf("          %d of those rows are not carried by this build: the entries "
                    "they are about are declared behind the fp16 seam this build has closed, "
                    "so the row has no entry to call and no cell to judge. That is a fact "
                    "about this build and not about the tree - no measurement is missing and "
                    "the claim is neither confirmed nor denied here. A build that carries the "
                    "entry is where the claim is checked, and the rows there are the same "
                    "rows, at the same count\n",
                    notCarried);
        std::printf("  NOT CARRIED by this build (named so a reader can see which claims this "
                    "build does not have; neither met nor failed here):");

        for (const DocClaim& c : book)
        {
            if (c.verdict == Verdict::NotCarried)
            {
                std::printf(" %s", c.id.c_str());
            }
        }

        std::printf("\n");
    }

    // The certified total, stated apart from the book's: a row measured on this machine is
    // certified by that measurement, a row resting on a software model of an accumulator is not,
    // however green, because the model is not the hardware the bound would be relied on against.
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

    // The count in the RESULT line is only as wide as the cells that can discriminate, so the
    // qualification prints inside the same line rather than beside it: a reader who quotes the
    // verdict gets the fraction the verdict rests on.
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
    // The routes get their own rows and their own count: the lane count is quoted by documents
    // outside this file, so a route measured apart is counted apart. A book that is not met does
    // not end the run - the books after it are printed too and the exit status is taken at the end.
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
            case Verdict::NotCarried:
                // Below every measured state, for the same reason as VerdictRank's
                // entry: not reachable here, and if it ever were it must not
                // outrank a row this build measured and failed.
                return -1;
            }

            return 5;
        };

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

        routeCells += f32RouteCells;
        // The float lane's policy rows are the route book's too: a policy names a route and a
        // scheme, so their cells are the route axis's cells and not the lane sweep's.
        routeCells += f32PolicyCells;

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
                 "no row of the double lane's route book has a reported delivered figure short "
                 "of what this gate measures for the same route by more than a tenth of the bar, "
                 "so the figure a consumer reads covers the fit the library evaluates",
                 "include/boys/boys.hpp, FitRouteInfo::delivered; the reported and measured "
                 "columns of the route table above",
                 routeDisagreement == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu route row(s) of the double lane's %zu whose reported figure is short of "
                     "the measured one by more than a tenth of the bar; the two columns are "
                     "printed side by side above. The measured column is this gate's own sweep on "
                     "the committed reference grid and the reported one is the generator's "
                     "fit-time sweep of each piece's own interval, so the two are not required to "
                     "agree in either direction, and they do not: the largest amount by which a "
                     "measured figure here sits above its own row's reported one is %.3g of that "
                     "row's bar. The comparison is one-sided for that reason. The float lane's "
                     "rows are a separate book, measured against their own rows' bounds in "
                     "float.route.delivered; the same one-sided comparison read on them leaves "
                     "%zu of %zu row(s) short by more than a tenth of the bar, the worst by %.3g "
                     "of it, which is inside the bar the row states and is that lane's reported "
                     "figure to correct rather than a bound it misses",
                     routeDisagreement,
                     routeNames,
                     routeShortfallWorst,
                     f32RouteShortOver,
                     routeSeedClaimF32.size(),
                     f32RouteShortWorst));

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
                 "include/boys/boys.hpp, BoysAllOrdersWithRoute() and FitRouteInfo::servesFrom; "
                 "src/boys.cpp, BoysFitGranularities() where the class row names the grid",
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
                 "EvalPolicy; src/boys.cpp, SelectorPolicy for the axes the pair does not name",
                 routeRuntimeDiff == 0 ? Verdict::Verified : Verdict::Exceeded,
                 Fmt("%zu of %zu cell(s) differ between the run-time pair selector and the "
                     "compile-time entry it names, over all four (route, scheme) pairs",
                     routeRuntimeDiff,
                     routeRuntimePairs));

        addRoute("float.route.delivered",
                 "the float lane's route report is true of the entry that reads its "
                 "tables: BoysSingleF32WithRoute delivers, at every order over each "
                 "interval BoysFitRoutesF32 names, no worse than the bound that row states",
                 "include/boys/boys.hpp (BoysSingleF32WithRoute, BoysFitRoutesF32)",
                 f32DeliveredVerdict,
                 Fmt("%zu route row(s), %zu cell(s) of the committed reference grid, each "
                     "compared against the bound its own row states, %zu of them over it; "
                     "worst delivered %.6g at n=%d, x=%g; the worst cell of each row is "
                     "printed beside the table above",
                     routeSeedClaimF32.size(),
                     f32RouteCells,
                     f32RouteOver,
                     f32RouteWorst,
                     f32RouteWorstN,
                     f32RouteWorstX));

        addRoute("float.route.report",
                 "BoysFitRoutesF32's rows are true of the tables this revision ships, and a "
                 "route value this build does not serve evaluates the default entry",
                 "include/boys/boys.hpp (FitRouteInfo), include/boys/boys_coefficients.hpp",
                 f32ReportVerdict,
                 Fmt("%zu row(s) whose stored count disagrees with the table the kernel "
                     "reads; %zu cell(s) where naming the default route or a value outside "
                     "the enumeration differed from the default entry; %zu cell(s) outside a "
                     "rational row's interval that naming the rational route changed",
                     routeStoredMismatchF32,
                     routeDefaultDiffF32,
                     routeDiffOutsideF32));

        addRoute("float.policy.single",
                 "the float lane's single-order entry delivers, at every policy it accepts and at "
                 "every order over the interval that policy's fits serve, no worse than the bar "
                 "the lane publishes, and no worse than the figure the generated header publishes "
                 "for the fit that policy reads there by more than a tenth of that bar",
                 "include/boys/boys.hpp, BoysSingleF32; include/boys/boys_coefficients.hpp, "
                 "kRegionAFitChebDelivered, kRegionAFitChebDeliveredHorner and "
                 "kRegionAFitRatDelivered with the three region-B figures beside them",
                 f32PolicyVerdict[0],
                 Fmt("worst %.3g of the bar at (n=%d, x=%.6g): delivered %.6g against %.6g, over "
                     "%zu comparison cell(s) across the six rows; %zu row(s) whose published "
                     "figure is short of this sweep's by more than a tenth of the bar, the worst "
                     "by %.3g of it. The sweep and the generator's fit-time sweep run over "
                     "different argument grids, so the two figures are not required to agree in "
                     "either direction and the comparison is one-sided for that reason",
                     f32PolicyWorstRatio[0],
                     f32PolicyWorstN[0],
                     f32PolicyWorstX[0],
                     f32PolicyWorstErr[0],
                     f32PolicyWorstBar[0],
                     f32PolicyCells / 2,
                     f32PolicyShortOver,
                     f32PolicyShortWorst));

        addRoute("float.policy.batch",
                 "the float lane's batch entry delivers, at every policy it accepts and every "
                 "order over the interval that policy's seeds serve, no worse than the bar the "
                 "lane publishes for the region the argument falls in - whether the seed is the "
                 "double lane's fit at the policy's route and scheme, as region A's is, or this "
                 "lane's, as region B's is",
                 "include/boys/boys.hpp, BoysAllOrdersF32 and BoysLaneContracts(), whose row "
                 "for the float lane carries the plain reciprocal's own term; "
                 "include/boys/boys_coefficients.hpp, kRegionAFitBar and kRegionBFitBar",
                 f32PolicyVerdict[1],
                 Fmt("worst %.3g of the bar at (n=%d, x=%.6g): delivered %.6g against %.6g, over "
                     "%zu comparison cell(s) across the six rows, %zu of them outside the bar. "
                     "Each row of the table above names its own worst cell",
                     f32PolicyWorstRatio[1],
                     f32PolicyWorstN[1],
                     f32PolicyWorstX[1],
                     f32PolicyWorstErr[1],
                     f32PolicyWorstBar[1],
                     f32PolicyCells - f32PolicyCells / 2,
                     f32PolicyOver));

        addRoute("float.policy.carries",
                 "each policy the float lane's engines accept is read: on every row whose value "
                 "a table of this lane's decides, naming a route or a scheme other than the "
                 "shipped pair changes the float the engine answers with, over the cells the two "
                 "policies cover in common; the batch entry's region-A rows, whose value comes "
                 "from the double lane's fit at the pair the policy names, are reported apart "
                 "rather than required to change",
                 "include/boys/boys.hpp, EvalPolicy and the float entries' \\tparam Policy; "
                 "boys_impl.hpp, FloatBatchRegionASeed",
                 (f32PolicyNotCarried == 0 && f32PolicyUncovered == 0) ? Verdict::Verified
                                                                        : Verdict::Exceeded,
                 Fmt("%zu of the 12 row(s) whose value a table of this lane's decides never "
                     "differed from the shipped pair's value on any cell the row covers, which is "
                     "what an engine that ignored the policy would answer; %zu row(s) of the "
                     "sixteen were measured over no argument at all. The batch entry's region-A "
                     "rows are reported apart: the scheme's row differs on %zu of %zu cell(s), the "
                     "route's on %zu of %zu, the narrow partition's on %zu of %zu and the narrow "
                     "partition's rational route on %zu of %zu, the last two differing where the "
                     "lane's narrow fits take over from the shipped ones. The count is bitwise on "
                     "the returned float, and it is the reading that makes the accuracy rows above "
                     "evidence for the option rather than for the table",
                     f32PolicyNotCarried,
                     f32PolicyUncovered,
                     f32PolicySeedDiffer[0],
                     f32PolicySeedCells[0],
                     f32PolicySeedDiffer[1],
                     f32PolicySeedCells[1],
                     f32PolicySeedDiffer[2],
                     f32PolicySeedCells[2],
                     f32PolicySeedDiffer[3],
                     f32PolicySeedCells[3]));

        // Not a claim: this is what the two RESULT lines above and below already
        // say, put side by side so a reader can see the routes were added without
        // the lanes' totals moving rather than having to trust that they were.
        std::printf("\n  the routes are counted apart from the lanes: the lane RESULT above reads "
                    "%d of %zu\n  claims carried by %zu of %zu comparison cells, and the route "
                    "rows and the float lane's\n  policy rows contribute %zu cells of their "
                    "own, none of them in that total\n",
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
        int rNotCarried = 0;

        // The carriage table first: the rows below are the routes' contracts, and whether an entry
        // reads the route it is handed is the question those contracts rest on. The count beside
        // each entry is how many served arguments it answered differently once the route was named.
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

        // The entries that do not carry the route refuse the policy that names it, and a
        // compile-time refusal is measured by compiling the call rather than by making it: the
        // configure probe does exactly that, and this is what it found.
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
            case Verdict::NotCarried:
                ++rNotCarried;
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
        // A site that reads a route and is not one of the rows above is named here, so the count
        // below is read with what it leaves out visible rather than assumed empty.
        struct RouteUncoveredSite {
            const char* site;
            const char* why;
        };

        const std::array<RouteUncoveredSite, 0> routeUncoveredList{};

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
                    "domain, %d exceeded, %d vacuous only, %d evidence absent, %d not carried "
                    "by this build; none of the last four counted as met)\n",
                    rVerified + rMetOverDomain,
                    routeBook.size(),
                    rVerified,
                    rMetOverDomain,
                    rExceeded,
                    rVacuousOnly,
                    rAbsent,
                    rNotCarried);

        if (rVerified + rMetOverDomain + rNotCarried < static_cast<int>(routeBook.size()))
        {
            std::printf("  NOT MET at this revision (routes):");

            for (const DocClaim& c : routeBook)
            {
                if (FailsTheGate(c.verdict))
                {
                    std::printf(" %s", c.id.c_str());
                }
            }

            std::printf("\n  FAIL (the route book above; the scheme book is still measured and "
                        "printed)\n");
            failed = true;
        } else if (rNotCarried > 0)
        {
            std::printf("  NOT CARRIED by this build (routes, named neither met nor failed "
                        "here):");

            for (const DocClaim& c : routeBook)
            {
                if (c.verdict == Verdict::NotCarried)
                {
                    std::printf(" %s", c.id.c_str());
                }
            }

            std::printf("\n");
        }
    }

    // Strict over the claims this build carries: a carried claim that is not met at this revision
    // - measured exceeded, measured vacuous only, or resting on evidence this revision cannot
    // re-run - leaves the gate red and names itself below. A claim whose subject the build does not
    // carry is named above with its reason and still counted in the RESULT line.
    if (verified + metOverDomain + notCarried < static_cast<int>(book.size()))
    {
        std::printf("  NOT MET at this revision:");

        for (const DocClaim& c : book)
        {
            if (FailsTheGate(c.verdict))
            {
                std::printf(" %s", c.id.c_str());
            }
        }

        std::printf("\n  FAIL (the lane book above; the books after it are still measured and "
                    "printed)\n");
        failed = true;
    } else if (notCarried > 0)
    {
        std::printf("  OK, for the claims this build carries (the lane book above): every row "
                    "the build carries is met at this revision, and the %d it does not carry "
                    "are named above rather than counted for it\n",
                    notCarried);
    }

    // ---- the evaluation-scheme rows, counted apart --------------------------
    // Their own block, their own cell count and their own RESULT line, because the rows above are
    // the existing book's and a new evaluation option must not move them. Judged exactly as that
    // book's rows are: a failure or a wholly bound-covered row is not met and leaves the run red.
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
    std::printf("  %-16s %-16s %5s %7s %9s %-22s %-22s %7s  %-22s %s\n",
                "scheme",
                "row",
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
            schemeNotMet.push_back(std::string(scheme) + " / " + row);
        }

        char where[64];
        std::snprintf(where,
                      sizeof(where),
                      "n=%d, x=%.6g",
                      a.worstN,
                      a.worstX);
        std::printf("  %-16s %-16s %5d %7d %9zu %-22.6g %-22.6g %7.3g  %-22s %s\n",
                    scheme,
                    row,
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
                  0,
                  0,
                  baseBound,
                  SchemeClaims()[static_cast<std::size_t>(e.slot)]);
    }

    // ---- the carriage table ------------------------------------------------
    // The accuracy column cannot see a dropped argument: both schemes' stored fits hold the same
    // bar, so an entry that evaluates the wrong one moves a value by a last-place digit and stays
    // inside its bound. Measured instead: whether the two readings are one reading, region by region.
    std::printf("\n  the carriage of every row above by the scheme it names: each row is read "
                "at both\n  schemes in the same pass, and the cells where the two readings "
                "differ are counted per\n  region. A region where the per-argument entry's two "
                "readings differ and this row's do\n  not is a region this row does not reach "
                "the scheme in, whatever its accuracy says.\n  A row's region reads '-' where "
                "the sweep covered no cell there; the reference line under this\n  header is "
                "what every row is held to.\n");
    std::printf("  the rows are held to the per-argument entry's own differences: such an "
                "argument in\n  each of A %zu, band %zu, B %zu, C %zu. Region C is no row's, so\n"
                "  no row is held to it\n",
                schemeRefDiffer[0],
                schemeRefDiffer[1],
                schemeRefDiffer[2],
                schemeRefDiffer[3]);
    std::printf("  %-16s %9s %9s  %-14s %s\n",
                "row",
                "cells",
                "differ",
                "A/band/B/C",
                "verdict");
    std::printf("  %s\n", std::string(132, '-').c_str());

    // The reference every row is judged against, as a compact string so the
    // column a row is held to is printed beside the row.
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

    const auto carriageRow = [&](const char* row, const SchemeCarriage& car) {
        ++schemeRows;

        const std::array<std::size_t, 4>& ref = schemeRefDiffer;
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
            std::printf("  %-16s %9zu %9zu  %-14s carried by the scheme it names\n",
                        row,
                        cells,
                        differ,
                        tokens);
            return;
        }

        schemeNotMet.push_back(std::string("carriage of ") + row);
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

        std::printf("  %-16s %9zu %9zu  %-14s NOT CARRIED - region %s differs nowhere "
                    "between the two schemes\n",
                    row,
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
                    schemeFitCarriage[lane]);
    }

    for (const SchemeCarriage& car : schemeEntryCarriage)
    {
        carriageRow(car.entry, car);
    }

    // ---- the packed region-A lane's scope, stated and held to the values ----
    // A lane a caller does not get is not an accuracy fact and no bound can carry it, so the
    // library states its scope (BoysPackedLaneServes) and these rows hold that statement to what
    // the values do - a claim either side of the boundary can fail.
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
    // A count is a claim about coverage, and a reader cannot check one without being told what it
    // leaves out: every site that reads an evaluation scheme and is not a row above is named here
    // with what it reads, so the SCHEME RESULT line is read with its boundary visible.
    struct UncoveredSite {
        const char* site;
        const char* reads;
        const char* why;
    };

    const std::array<UncoveredSite, 3> uncovered{{
        {"a route or a scheme other than the shipped pair, on the single-precision engines "
         "(BoysSingleF32, BoysAllOrdersF32) and the fp16/bf16 lanes built on them",
         "the policy's fit route and its scheme",
         "not swept by this book, and not refused: a policy names a route and a scheme together, "
         "so this lane's rows are keyed on the pair and belong to the route book - "
         "float.policy.single, float.policy.batch and float.policy.carries, each judged "
         "against the bar the row documents. This book's rows are the scheme axis "
         "read on its own, through the entries that name a scheme, and no entry on this lane "
         "takes one: the engines take a policy. No entry on the fp16/bf16 lanes names a route at "
         "all. A revision that dropped that pair of the route book's rows would leave this "
         "site unswept and unrefused, which is what the run-time probe is read for"},
        {"the region-A transform lane (BoysRegionAProduct and its modes)",
         "no evaluation policy at all",
         "its modes are the lane's own arguments rather than an EvalPolicy, so a "
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
    // Every option the library's own enumerations report, asked whether it is supported (it runs),
    // bounded (measured here against the committed reference) and reachable (a public entry selects
    // it, and naming it changes what that entry answers with). A member failing any fails the check.
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

        m.supported = cells > 0 && entryCells > 0;
        m.bounded = m.supported && failures == 0 && entryFailures == 0;

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

    // The granularity axis, walked from its enumerator because the library reports no table of
    // this axis, so a member added to it is counted here without this block being edited. The cells
    // are the granularity book's own, and the reachability figure is that book's carriage count.
    for (int g = 0; g <= static_cast<int>(boys::FitGranularity::kNarrow); ++g)
    {
        const boys::FitGranularity gran = static_cast<boys::FitGranularity>(g);
        OptionMember o;
        o.kind = "granularity";
        o.member = boys::GranularityName(gran);
        std::size_t cells = 0;
        std::size_t failures = 0;
        std::size_t differ = 0;

        for (std::size_t s = 0; s < granSchemeCount; ++s)
        {
            for (std::size_t r = 0; r < kGranRowCount; ++r)
            {
                const std::size_t index = granIndex(static_cast<std::size_t>(g), s, r);
                cells += granCells[index];
                differ += granDiffer[index];
                failures += GranularityClaims()[static_cast<std::size_t>(granSlots[index])]
                                .failures;
            }
        }

        o.supported = cells > 0;
        o.bounded = o.supported && failures == 0;
        o.reachable = differ > 0;
        o.note = Fmt("%zu cell(s) at both schemes, %zu of them differing from the other "
                     "partition's reading, %zu over the bar its row is judged at",
                     cells,
                     differ,
                     failures);
        optionSpace.push_back(std::move(o));
    }

    // The refusal record: the combinations this build refuses, and what backs each. A refusal
    // stands in for the three questions only where the library refuses the call at compile time,
    // and each entry is backed by a configure probe that compiles exactly the refused call.
    // `unbuilt` separates a body nobody has written - owed work - from a limit no revision lifts.
    struct Refusal {
        const char* member;
        const char* why;
        bool backed = false;
        bool unbuilt = false;
    };

    // The other direction, and the one this block must not be quiet about: a refusal whose probe
    // COMPILED the call is a limit this revision does not have. That is not a defect in itself -
    // a capability landing is the point - but one that lands while nothing measures it is a silent
    // hole, so a lifted refusal fails here and names the book that has to carry the row.
    std::size_t liftedRefusals = 0;

    std::vector<Refusal> refusals;
#ifdef BOYS_GATE_BATCH_REFUSES_ROUTE
    refusals.push_back({"rational route on BoysAllN / BoysAllN sorted / BoysFixedN",
                        "their region bodies evaluate the shipped seed and the shipped "
                        "per-order fits as their own, so the route has no stored form there; "
                        "the probe compiles the call and it does not build",
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
#ifdef BOYS_GATE_F32_SINGLE_REFUSES_ORDERS
    // A one-order entry has one order, so it has nothing to pack and refuses the axis for that
    // reason alone: a property of the call and not of the axis, so it is booked as a shape limit
    // and the debt count is left to the rows naming a table or body nobody has built.
    refusals.push_back({"orders axis on the one-order single-precision entry",
                        "this entry produces one order at one argument, so there are not four "
                        "orders on this shape to fill a vector lane with; the probe compiles the "
                        "call and it does not build. This one is a property of the call and not a "
                        "lane nobody wrote: no revision of this entry produces four orders for the "
                        "axis to pack, and the same lane's all-orders and many-argument entries "
                        "carry the axis",
                        true});
#else
    // The shape refuses the axis and the probe compiled it, which would mean the
    // limit is not what this block says it is. Nothing is silent in either
    // direction: the probe decides which line prints.
    ++liftedRefusals;
    std::printf("  LIFTED: the one-order entry on the single-precision engines accepts the "
                "orders\n  axis, which a call producing one order has nothing to fill - so the "
                "shape limit this\n  block books is not the shape's. A revision that reaches this "
                "line has changed what\n  the entry is\n");
#endif
#ifdef BOYS_GATE_F32_REFUSES_ORDERS
    // The axis itself, on the shapes that can carry it. This is the row the
    // probe above measures directly: the all-orders entry and the many-argument
    // entry, so a failure here is a fact about the axis.
    refusals.push_back({"orders axis on the single-precision engines",
                        "a packed lane reads one coefficient stride and the float table gives "
                        "each order its own cover, so a fixed argument selects a different piece "
                        "in each lane and the eight bases are not a stride apart; the probe "
                        "compiles the call and it does not build. The packed lane is built at "
                        "this revision, so this row prints only where a revision has lost it - "
                        "and a lost packed lane is a body to write, not a shape the call cannot "
                        "have: the eight lanes share the degree the group is summed at, and a "
                        "lane whose own fit is cut shorter reads zeros above its own cut, which "
                        "is that lane's own polynomial read in that lane's own arithmetic",
                        true,
                        true});
#else
    // The axis is implemented and measured: the probe compiled the call, and the row it owes is
    // carried by the float book under "all, orders axis", measured against the committed reference
    // grid at the lane's own bar. Neither direction is silent.
    std::printf("  CARRIED: the single-precision engines compile a policy naming the orders "
                "packing axis,\n  and the float book above carries the row it owes at the lane's "
                "own bar, measured\n  against the committed reference grid beside the per-order "
                "entry's row over the same\n  cells. The probe is the reading that says so; a "
                "revision that dropped the axis would\n  print the refusal instead\n");
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

    // The members of the granularity axis this book's own slots do not hold: the uniform partition
    // is named here rather than left out of the list above in silence. Both of its members are
    // measured in the cross below, which enumerates the partition from BoysFitGranularities.
    for (const boys::FitGranularityInfo& row : boys::BoysFitGranularities())
    {
        if (static_cast<std::size_t>(row.granularity) < kGranMembers)
        {
            continue;
        }

        std::printf("  %-10s %-34s %-4s %-4s %-4s %s\n",
                    "granularity",
                    row.name,
                    "n/a",
                    "n/a",
                    "n/a",
                    "this book's rows hold the shipped and narrow partitions, and this "
                    "partition's members are named here rather than left out of the list above "
                    "in silence. The combination book below measures both, at both schemes, on "
                    "the cross's own partition axis");
    }

    std::printf("  %s\n", std::string(150, '-').c_str());

    std::size_t unbackedRefusals = 0;
    std::size_t unbuiltLimits = 0;

    for (const Refusal& r : refusals)
    {
        if (!r.backed)
        {
            ++unbackedRefusals;
        }

        if (r.unbuilt)
        {
            ++unbuiltLimits;
        }
    }

    // A refusal whose probe COMPILED the call is a limit this revision does not have; a capability
    // that lands while nothing measures it is a silent hole, so a lifted refusal fails here and
    // names the book that has to carry the row before the gate can go green again.
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
    std::printf("  CARRIED: the single-precision engines accept a route or a scheme other "
                "than the\n  shipped pair, and the fp32 policy rows above measure what they "
                "answer with it: one\n  row per policy, region and entry, each judged against "
                "the bar the lane publishes\n  for that region, with the count of the row's "
                "cells that differ from the shipped\n  pair's value beside it\n");
#ifndef BOYS_GATE_FIXEDN_REFUSES_ORDERS
    ++liftedRefusals;
    std::printf("  LIFTED: BoysFixedN accepts the orders axis, which its call shape has no "
                "orders to\n  fill - a value this entry cannot produce under any revision of it. "
                "A revision\n  that reaches this line has changed what the entry is\n");
#endif
#ifndef BOYS_GATE_F32_REFUSES_ORDERS
    // The orders axis was a limit on the single-precision engines until this revision, and it is
    // not one any more: the call compiles and runs, and the row that landing owes is measured rather
    // than promised. The float book's "all, orders axis" claim above is that row.
    std::printf("  CARRIED: the single-precision engines accept an orders-axis policy, and the "
                "float\n  book above carries the row it owes at the lane's own bar, measured "
                "against the\n  committed reference grid beside the per-order entry's row over "
                "the same cells. The\n  probe is the reading that says so; a revision that "
                "dropped the axis would print the\n  refusal instead\n");
#endif
    // The orders axis carried two limits until this revision, and both are gone: it answers a route
    // other than the shipped one, and the narrow partition of region A. Neither landing is silent -
    // the packing-axis rows above measure both routes and both partitions, through both entries.
    std::printf("  CARRIED: the orders axis reads a route other than the shipped one: the\n"
                "  rational route's region-A fits cover the same per-order intervals as the\n"
                "  shipped piece table, and the packing-axis rows above measure them on both\n"
                "  entries the axis is carried on. The single-precision entries are not among\n"
                "  them: the axis is refused on that lane, and the limits list below carries\n"
                "  the reason and counts it\n");
    std::printf("  CARRIED: the orders axis reads the narrow partition of region A: its pieces\n"
                "  are cut per order, so the lane fetches each of the four orders it packs its\n"
                "  own piece and coefficients instead of stepping one piece's coefficients at\n"
                "  a fixed stride, and the packing-axis rows above measure both schemes, on\n"
                "  both entries, each against the shipped lane's own region-A\n"
                "  figure - the bar the narrow pieces are cut under is the narrower of the\n"
                "  two\n");

    std::printf("  limits the library states at the call site (%zu). Each row's own line says "
                "which of\n  the two it is: a table or a body that has not been built, which is "
                "an outstanding work\n  item and counts with the outstanding combinations below, "
                "or a shape the call cannot\n  have. The two are not the same debt and are not "
                "counted the same.\n  %zu of them are backed by a probe that compiles the "
                "refused call and the rest\n  name the static assertion that states it\n",
                refusals.size(),
                refusals.size() - unbackedRefusals);
    std::printf("  of these, %zu name a body or a table this library has not built, and %zu are "
                "shapes the\n  call cannot have; the first count is the debt and it is owed the "
                "way the outstanding\n  combinations below are, which is what keeps a refusal "
                "from reading as a boundary\n",
                unbuiltLimits,
                refusals.size() - unbuiltLimits);

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

    // The combinations: the whole option space, crossed and counted. Five axes - precision
    // (BoysLaneContracts), route (BoysFitRoutes / BoysFitRoutesF32), scheme, partition and packing
    // axis - every member read off a table the library publishes. Each combination is certified and
    // published, refused and owed (debt, not impossibility), or not runnable on this host.
    constexpr std::size_t kCombForms = 3;

    // The plain reciprocal as a column index, for the places that read the bar
    // one form was judged by: the arms compose that bar under this form's name
    // and the rows below compare the accessor's answer for it against the same
    // column.
    constexpr std::size_t kPlainFormIndex =
        static_cast<std::size_t>(boys::DivisionForm::kPlainReciprocal);

    struct Combination {
        std::string axes;
        std::string state;
        std::string source;
        std::size_t cells = 0;
        std::size_t below = 0;
        std::size_t over = 0;
        double delivered = 0.0;
        double bound = 0.0;
        // The widest bar the row was judged against, which is the figure it is certified under: it
        // differs from `bound` exactly where a lane publishes the plain reciprocal's own figure
        // beside its base, and it is printed so a figure above the base cannot read as a defect.
        double judgedBound = 0.0;
        double accessorBound = 0.0;
        // The same answer for the form this build's unnamed calls divide in, which is what the
        // tolerance query and BoysAccuracyDelivered read: neither names a form, so both read the
        // build's default, and the query's `bound` is this figure, not `accessorBound`.
        double accessorFormBound = 0.0;
        // The accessor's figure for the plain reciprocal, and the bar the arms judged that form
        // by: a row that publishes a term for that form publishes a second figure, and these two
        // say the figure the arms held the form to is the one a consumer reads from the accessor.
        double accessorPlainBound = 0.0;
        double plainBar = 0.0;
        double accessorDelivered = 0.0;
        bool accessorDeliveredKnown = false;
        boys::ToleranceVerdict atBound = boys::ToleranceVerdict::kNotCarried;
        boys::ToleranceVerdict atHalf = boys::ToleranceVerdict::kNotCarried;
        double toleranceBound = 0.0;
        double toleranceDelivered = 0.0;
        bool toleranceDeliveredKnown = false;
        std::string toleranceReason;
        int worstN = -1;
        double worstX = 0.0;
        int worstForm = -1; // the division form that delivered the worst value
        // Each form's own worst, the point it was found at, and the count of
        // forms the row was read at: a row whose lane states a figure per form
        // prints the figures its forms delivered.
        std::size_t forms = 0;
        double formWorst[kCombForms] = {};
        int formWorstN[kCombForms] = {-1, -1, -1};
        double formWorstX[kCombForms] = {};
    };

    // The lanes, in the order BoysLaneContracts() reports them: four of them are the device's, and
    // the arms below launch all four. The half lane's two stores are two lanes here for the reason
    // they are two classes in the seam - an fp16 arm cannot measure a bf16 cell.
    const std::span<const boys::LaneContractInfo> combLaneRows = boys::BoysLaneContracts();
    const int combLaneCount = static_cast<int>(combLaneRows.size());
    // maybe_unused: only the CUDA arms below read this one, and a build with
    // BUILD_CUDA=OFF compiles none of them.
    [[maybe_unused]] const int combDeviceLane = static_cast<int>(boys::Precision::kFp32Device);
    const int combHalfLane = static_cast<int>(boys::Precision::kFp16);

    /// Whether a lane is one of the device's four, which are the four precisions the device
    /// surface's entries are built at (boys/boys_device_tables.hpp, BoysDeviceLane). A cell of
    /// one of them is a cell no host entry answers, so the four are classified together.
    const auto combIsDeviceLane = [](int lane) {
        return lane == static_cast<int>(boys::Precision::kFp32Device) ||
               lane == static_cast<int>(boys::Precision::kFp64Device) ||
               lane == static_cast<int>(boys::Precision::kFp16Device) ||
               lane == static_cast<int>(boys::Precision::kBf16Device);
    };

    // Each lane's route axis: the distinct routes that lane's own report names, in the order the
    // report first names them. The device's lanes read the double lane's table for region A, so they
    // enumerate their routes from BoysFitRoutes() with it.
    const auto combRoutesFor = [](int lane) {
        return lane == static_cast<int>(boys::Precision::kFp64) ||
                       lane == static_cast<int>(boys::Precision::kFp32Device) ||
                       lane == static_cast<int>(boys::Precision::kFp64Device) ||
                       lane == static_cast<int>(boys::Precision::kFp16Device) ||
                       lane == static_cast<int>(boys::Precision::kBf16Device)
                   ? boys::BoysFitRoutes()
                   : boys::BoysFitRoutesF32();
    };

    std::vector<std::vector<boys::FitRoute>> combRoutes(static_cast<std::size_t>(combLaneCount));

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        for (const boys::FitRouteInfo& row : combRoutesFor(lane))
        {
            bool seen = false;

            for (const boys::FitRoute carried : combRoutes[static_cast<std::size_t>(lane)])
            {
                seen = seen || carried == row.route;
            }

            if (!seen)
            {
                combRoutes[static_cast<std::size_t>(lane)].push_back(row.route);
            }
        }
    }

    const std::size_t combSchemes = boys::BoysEvalSchemes().size();
    const std::size_t combPartitions = boys::BoysFitGranularities().size();
    const std::size_t combAxes = boys::BoysPackAxes().size();

    // The division forms, read off the accessor that names them for the reason the axis sizes above
    // are read off theirs: a member added moves this number rather than leaving a list to drift. It
    // is not a factor of the cross's arithmetic - DivisionFormInfo carries no coverage field, "there
    // is no cell this axis is refused on" - so it arrives in the measurement and not in the count.
    const std::span<const boys::DivisionFormInfo> combFormRows = boys::BoysDivisionForms();
    const std::size_t combForms = combFormRows.size();

    // The product taken a second way, off the tables' own sizes. A member added
    // to any axis moves this and the enumeration below together; a cell the
    // enumeration drops moves only one of them, and the difference is what the
    // reader sees.
    std::size_t combClaimed = 0;

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        combClaimed += combRoutes[static_cast<std::size_t>(lane)].size() * combSchemes *
                       combPartitions * combAxes;
    }

    // The measurement side: one row per cell this revision's library carries, measured over the
    // whole committed grid through the entry that carries all of the cell's axes at once. A list of
    // explicit instantiations rather than a loop, because the refusals are compile-time ones.
    struct CombCell {
        int lane;
        int route;
        int scheme;
        int partition;
        int axis;
        std::size_t cells;
        std::size_t below;
        std::size_t over;
        double delivered;
        double bound;
        double judgedTo;      // the widest bar any form of this cell was judged against
        double plainTerm;     // beside the base, the part of that bar the plain form's own figure added
        double valueTerm;     // beside the base, the part of that bar that is a term of the value returned
        int worstN;
        double worstX;
        int worstForm;        // which division form delivered the worst value
        std::size_t forms;    // the division forms this cell was read at
        std::size_t moved;    // values a form past the first delivered differently
        std::size_t compared; // values a form past the first was asked for
        std::size_t movedByForm[kCombForms]; // the same, one count per form
        // Each form's own worst over the points this cell was read at, and the point it was found
        // at. On a lane whose row states a figure for a form past the first, this is the figure that
        // form's own statement is held to.
        double formWorst[kCombForms];
        int formWorstN[kCombForms];
        double formWorstX[kCombForms];
        // The bar this cell's plain-reciprocal form was judged by, as the arm composed it: the
        // lane's base plus the term the row publishes for that form plus the member term. The
        // accessor's own composition is held to it below.
        double plainBar;
    };

    std::vector<CombCell> combMeasured;

    struct CombAccum {
        std::size_t cells = 0;
        std::size_t below = 0;
        std::size_t over = 0;
        double worst = 0.0;
        double bound = 0.0;
        // The bar a form is judged against where its lane publishes a figure of
        // its own for it; zero means the lane publishes one figure for every
        // form, which is the case on every row whose forms deliver one value.
        double formBar[kCombForms] = {};
        // The widest bar this cell was judged against: `bound` on every lane whose forms publish
        // one figure, and the plain form's own figure where a lane publishes one for it - the
        // figure the row is certified under, printed so a delivered figure above the base reads.
        double judgedTo = 0.0;
        // What set that bar, kept as the two terms it added beside the lane's base: a report
        // printing the bar is owed the terms that are in the number beside it and no others, and
        // which they are is a fact of the cell that produced the bar rather than of the lane's row.
        double judgedPlain = 0.0;
        double judgedValue = 0.0;
        double ceiling = 0.0; // above this magnitude the bound is claimed
        int worstN = -1;
        double worstX = 0.0;
        int worstForm = -1;       // which division form delivered `worst`
        std::size_t forms = 0;    // the forms each point of this cell was read at
        std::size_t moved = 0;    // values a form past the first delivered differently
        std::size_t compared = 0; // values a form past the first was asked for

        // The same count kept per form, in the accessor's order, because which
        // form moved is the question: the library documents the refined form as
        // bit-identical to exact division, and a single total cannot say whether
        // the values that moved were that form's or the plain one's.
        std::size_t movedByForm[kCombForms] = {};

        // Each form's own worst, kept beside the worst over all of them: a row
        // whose forms deliver different figures states one figure per form, and
        // the figure a form's statement has to bound is that form's own worst
        // rather than the worst of the three.
        double formWorst[kCombForms] = {};
        int formWorstN[kCombForms] = {-1, -1, -1};
        double formWorstX[kCombForms] = {};

        void add(int n, double x, double got, double want, double ulp = 0.0,
                 int form = 0) noexcept
        {
            const double err = std::abs(got - want);
            const double magnitude = std::abs(want);
            // The form's own figure where the lane publishes one for it: a lane
            // whose plain reciprocal rounds once more than its other forms
            // states that term apart, and judging the plain form against the
            // other forms' figure would fail the lane for a figure it publishes.
            const double laneBar =
                formBar[static_cast<std::size_t>(form)] > 0.0
                    ? formBar[static_cast<std::size_t>(form)]
                    : bound;
            const double bar = laneBar + ulp;

            if (bar > judgedTo)
            {
                // The terms this bar added beside the lane's base, recorded with
                // it so that a report of the widest bar names the terms in it.
                judgedTo = bar;
                judgedPlain = laneBar - bound;
                judgedValue = ulp;
            }

            ++cells;

            if (ceiling > 0.0 && magnitude <= bar)
            {
                // At or below the format's floor the lane claims nothing, and
                // the cell is counted rather than judged.
                ++below;

                return;
            }

            if (err > bar)
            {
                ++over;
            }

            if (err > worst)
            {
                worst = err;
                worstN = n;
                worstX = x;
                worstForm = form;
            }

            const std::size_t sf = static_cast<std::size_t>(form);

            if (err > formWorst[sf])
            {
                formWorst[sf] = err;
                formWorstN[sf] = n;
                formWorstX[sf] = x;
            }
        }

        // One (order, argument) as every form the axis carries delivered it: the first form is
        // judged and kept as the baseline, each form past it judged against the same figure and
        // compared with the baseline, so the pair of counts says how far this lane's arithmetic
        // moved when the form changed. The per-form ulp is the half lane's.
        void addAtForms(int n, double x, const double* got, std::size_t countForms, double want,
                        const double* ulp = nullptr) noexcept
        {
            forms = countForms;
            add(n, x, got[0], want, ulp == nullptr ? 0.0 : ulp[0], 0);

            for (std::size_t f = 1; f < countForms; ++f)
            {
                add(n, x, got[f], want, ulp == nullptr ? 0.0 : ulp[f],
                    static_cast<int>(f));
                ++compared;

                if (got[f] != got[0])
                {
                    ++moved;
                    movedByForm[f] += 1;
                }
            }
        }
    };

    // ---- the host lanes' cross: every class, every member -------------------
    // One row per (lane, route, scheme, axis, partition), keyed by the lane because
    // BoysAccuracyGuaranteed takes no Shape. Each shape is measured through the entry GateClassLadder
    // names, and every cell is read at both region-B members and at every division form.

    // The member book: per lane, the cells this cross read at each region-B member,
    // how many of them the two delivered differently, and each member's own worst.
    struct CombMemberBook {
        std::size_t cells[2] = {0, 0};
        std::size_t compared = 0;
        std::size_t differed = 0;
        double worst[2] = {0.0, 0.0};
    };

    std::vector<CombMemberBook> combMemberBook(static_cast<std::size_t>(combLaneCount));

    // The class book's accumulation, per (lane, shape): every cell this cross read through that
    // class's own entry, and the combinations it measured there - a class whose entry this cross
    // does not name is a class with no cells and is printed as such.
    std::vector<std::array<CombAccum, 5>> combClassCells(static_cast<std::size_t>(combLaneCount));
    std::vector<std::array<std::size_t, 5>> combClassCombos(static_cast<std::size_t>(combLaneCount));

    // The region-B members, off the table the library publishes. The two policies
    // the cross instantiates are named here and the table's own size is checked
    // against them, so a member added to the library is a sentence in the class
    // book rather than a cell that quietly went unmeasured.
    const std::span<const boys::RegionBExpInfo> combMemberRows = boys::BoysRegionBExps();
    const std::size_t combMembers = combMemberRows.size();

    const auto combMemberIndexOf = [&](boys::RegionBExp member) {
        for (std::size_t m = 0; m < combMemberRows.size(); ++m)
        {
            if (combMemberRows[m].exp == member)
            {
                return m;
            }
        }

        return combMemberRows.size();
    };

    const std::size_t combIdxAccurate = combMemberIndexOf(boys::RegionBExp::kAccurate);
    const std::size_t combIdxFast = combMemberIndexOf(boys::RegionBExp::kFast);

    // The class space: what each (lane, shape) is measured against, the lane's whole space for
    // every shape rather than the one-order shapes' with the orders half already taken out, so
    // combinations those shapes cannot carry are counted as refused rather than by shrinking the
    // space until the arithmetic closes.
    std::vector<std::array<std::size_t, 5>> combClassSpace(static_cast<std::size_t>(combLaneCount));

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        for (std::size_t s = 0; s < 5; ++s)
        {
            combClassSpace[static_cast<std::size_t>(lane)][s] =
                combRoutes[static_cast<std::size_t>(lane)].size() * combSchemes * combAxes *
                combPartitions * combForms * combMembers;
        }
    }

    // Half of the last representable digit of a value in a format that keeps
    // kMantissaBits explicit mantissa bits, at the value's own exponent. The digit
    // is 2^(exponent - kMantissaBits) and the figure a half lane states beside its
    // base is half of it.
    const auto combHalfDigit = [](double value, int kMantissaBits) {
        const int exponent = std::ilogb(value);

        return exponent == FP_ILOGB0 || exponent == FP_ILOGBNAN || exponent < -1074
                   ? 0.0
                   : std::ldexp(1.0, exponent - 1 - kMantissaBits);
    };

    // The mantissa widths the two half formats keep, beside the families they belong
    // to: binary16 has 10 explicit mantissa bits after its leading one, bfloat16 has
    // 7 (kBf16MantissaBits, at the head of this file). Both are the format's
    // definition and not a figure off a lane's row.

    // One family's cross: the four host lanes, one family each. kFamily indexes the
    // entry family of GateClassLadder and the reference column its cells are judged
    // against; the lane, the budget and the column all follow from it.
    const auto combHostCross =
        [&]<int kFamily, boys::FitRoute kRoute, boys::EvalScheme kScheme, boys::PackAxis kAxis,
            boys::FitGranularity kGran>() {
            constexpr int kLane = kFamily == 0   ? static_cast<int>(boys::Precision::kFp64)
                                  : kFamily == 1 ? static_cast<int>(boys::Precision::kFp32)
                                  : kFamily == 2 ? static_cast<int>(boys::Precision::kFp16)
                                                 : static_cast<int>(boys::Precision::kBf16);
            constexpr boys::BoysBudget kBudget =
                kFamily < 2 ? boys::BoysBudget::kFloat : boys::BoysBudget::kFp16;

            using PExactA = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                            boys::DivisionForm::kExactDivision,
                                            boys::RegionBExp::kAccurate>;
            using PPlainA = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                            boys::DivisionForm::kPlainReciprocal,
                                            boys::RegionBExp::kAccurate>;
            using PRefinedA = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                              boys::DivisionForm::kRefinedReciprocal,
                                              boys::RegionBExp::kAccurate>;
            using PExactF = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                            boys::DivisionForm::kExactDivision,
                                            boys::RegionBExp::kFast>;
            using PPlainF = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                            boys::DivisionForm::kPlainReciprocal,
                                            boys::RegionBExp::kFast>;
            using PRefinedF = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis, kGran,
                                              boys::DivisionForm::kRefinedReciprocal,
                                              boys::RegionBExp::kFast>;

            const boys::LaneContractInfo& laneRow = combLaneRows[static_cast<std::size_t>(kLane)];
            const double laneBound = laneRow.bound;
            const double laneAdd = laneRow.additive;
            const double lanePlainAdd = laneRow.plainAdditive;

            // The figure this lane answers at each member: the row's term beside the
            // base is the member's own, so a reading at the other member is judged
            // without it (LaneContractInfo::additiveMember).
            const double memberAdd[kCombMemberSlots] = {
                static_cast<int>(combIdxAccurate) == static_cast<int>(laneRow.additiveMember)
                    ? laneAdd
                    : 0.0,
                static_cast<int>(combIdxFast) == static_cast<int>(laneRow.additiveMember)
                    ? laneAdd
                    : 0.0};

            CombAccum a;
            a.bound = laneBound + laneAdd;
            a.formBar[kPlainFormIndex] = laneBound + lanePlainAdd + laneAdd;
            if constexpr (kFamily >= 2)
            {
                // The half formats' floor: at or below the magnitude the lane states
                // as its base the format itself is the limit and the lane claims
                // nothing, which is the rule the half lane's cells were already
                // judged under. Both half families carry it.
                a.ceiling = laneBound;
            }

            for (std::size_t s = 0; s < 5; ++s)
            {
                CombAccum& classCells = combClassCells[static_cast<std::size_t>(kLane)][s];
                classCells.bound = a.bound;
                classCells.formBar[kPlainFormIndex] = a.formBar[kPlainFormIndex];
                classCells.ceiling = a.ceiling;
            }

            // The column of arguments this family's entries are handed and the column of
            // correctly-rounded values their cells are judged against; the half pairs are the
            // half-rounded ones, so a half cell is judged against the value the format can hold.
            const double* argCol = ref.x.data();
            const double* wantCol = ref.v.data();

            if constexpr (kFamily == 1)
            {
                argCol = ref.xf.data();
                wantCol = ref.vf.data();
            }
            else if constexpr (kFamily == 2)
            {
                argCol = ref.x16.data();
                wantCol = ref.v16.data();
            }
            else if constexpr (kFamily == 3)
            {
                argCol = ref.xb.data();
                wantCol = ref.vb.data();
            }

            // One class's cells: the ladder of that class's own entry at every point
            // of the grid, at the three forms and both members, judged at the
            // figure this lane answers for the member read.
            const auto oneClass = [&]<boys::Shape kShape>() {
                constexpr std::size_t kShapeIndex =
                    kShape == boys::Shape::kSingle         ? 0
                    : kShape == boys::Shape::kFixedN       ? 1
                    : kShape == boys::Shape::kAllN         ? 2
                    : kShape == boys::Shape::kAllNAtOrders ? 3
                                                           : 4;

                CombAccum& classCells =
                    combClassCells[static_cast<std::size_t>(kLane)][kShapeIndex];
                std::array<std::array<double, 33>, kCombEvalReadings> got{};

                for (std::size_t i = 0; i < count; ++i)
                {
                    const double xd = argCol[i];

                    GateClassLadder<kFamily, kShape, PExactA>(nmax, xd, got[0].data());
                    GateClassLadder<kFamily, kShape, PPlainA>(nmax, xd, got[1].data());
                    GateClassLadder<kFamily, kShape, PRefinedA>(nmax, xd, got[2].data());
                    GateClassLadder<kFamily, kShape, PExactF>(nmax, xd, got[3].data());
                    GateClassLadder<kFamily, kShape, PPlainF>(nmax, xd, got[4].data());
                    GateClassLadder<kFamily, kShape, PRefinedF>(nmax, xd, got[5].data());

                    for (int n = 0; n <= nmax; ++n)
                    {
                        const std::size_t sn = static_cast<std::size_t>(n);
                        const double want = wantCol[ref.Index(n, i)];
                        const double gotA[kCombFormsPerMember] = {got[0][sn], got[1][sn], got[2][sn]};
                        const double gotB[kCombFormsPerMember] = {got[3][sn], got[4][sn], got[5][sn]};
                        double ulpA[kCombFormsPerMember] = {0.0, 0.0, 0.0};
                        double ulpB[kCombFormsPerMember] = {0.0, 0.0, 0.0};

                        if constexpr (kFamily >= 2)
                        {
                            // The format's own width: binary16 keeps 10 explicit
                            // mantissa bits after the leading one, bfloat16 keeps 7.
                            constexpr int kWidth = kFamily == 2 ? kF16MantissaBits : kBf16MantissaBits;

                            for (std::size_t f = 0; f < kCombFormsPerMember; ++f)
                            {
                                ulpA[f] = combHalfDigit(gotA[f], kWidth);
                                ulpB[f] = combHalfDigit(gotB[f], kWidth);
                            }
                        }

                        // Each member at the figure the accessor answers for it: the lane's
                        // base plus the term its row states under the member read, plus the plain
                        // reciprocal's own term at the plain form. The row's own figure is restored
                        // after the loop.
                        for (std::size_t m = 0; m < kCombMemberSlots; ++m)
                        {
                            const double* const gotM = m == 0 ? gotA : gotB;
                            const double* const ulpM = m == 0 ? ulpA : ulpB;
                            const double base = laneBound + memberAdd[m];
                            const double plainBar = laneBound + lanePlainAdd + memberAdd[m];

                            a.bound = base;
                            a.formBar[kPlainFormIndex] = plainBar;
                            classCells.bound = base;
                            classCells.formBar[kPlainFormIndex] = plainBar;

                            a.addAtForms(n, xd, gotM, kCombFormsPerMember, want, ulpM);
                            classCells.addAtForms(n, xd, gotM, kCombFormsPerMember, want, ulpM);
                        }

                        CombMemberBook& memberBook = combMemberBook[static_cast<std::size_t>(kLane)];

                        for (std::size_t f = 0; f < kCombFormsPerMember; ++f)
                        {
                            if (combIdxAccurate < 2 && combIdxFast < 2)
                            {
                                ++memberBook.cells[combIdxAccurate];
                                ++memberBook.cells[combIdxFast];
                                ++memberBook.compared;

                                if (gotA[f] != gotB[f])
                                {
                                    ++memberBook.differed;
                                }

                                const double errA = std::abs(gotA[f] - want);
                                const double errB = std::abs(gotB[f] - want);

                                if (errA > memberBook.worst[combIdxAccurate])
                                {
                                    memberBook.worst[combIdxAccurate] = errA;
                                }

                                if (errB > memberBook.worst[combIdxFast])
                                {
                                    memberBook.worst[combIdxFast] = errB;
                                }
                            }
                        }
                    }
                }
            };

            const std::size_t measuredCombos = combForms * combMembers;

            if constexpr (kAxis == boys::PackAxis::kArguments)
            {
                oneClass.template operator()<boys::Shape::kSingle>();
                oneClass.template operator()<boys::Shape::kFixedN>();
                combClassCombos[static_cast<std::size_t>(kLane)][0] += measuredCombos;
                combClassCombos[static_cast<std::size_t>(kLane)][1] += measuredCombos;
            }

            oneClass.template operator()<boys::Shape::kAllN>();
            oneClass.template operator()<boys::Shape::kAllNAtOrders>();
            oneClass.template operator()<boys::Shape::kAllOrders>();
            combClassCombos[static_cast<std::size_t>(kLane)][2] += measuredCombos;
            combClassCombos[static_cast<std::size_t>(kLane)][3] += measuredCombos;
            combClassCombos[static_cast<std::size_t>(kLane)][4] += measuredCombos;

            // The row's own figure, restored after the per-member judgements above:
            // the lane's base plus the term its row states under the member that row
            // names, which is the figure the accessor answers for the combination
            // this row stands for and the figure the row prints.
            a.bound = laneBound + laneAdd;
            a.formBar[kPlainFormIndex] = laneBound + lanePlainAdd + laneAdd;

            combMeasured.push_back({kLane,
                                    static_cast<int>(kRoute),
                                    static_cast<int>(kScheme),
                                    static_cast<int>(kGran),
                                    static_cast<int>(kAxis),
                                    a.cells,
                                    a.below,
                                    a.over,
                                    a.worst,
                                    a.bound,
                                    a.judgedTo,
                                    a.judgedPlain,
                                    a.judgedValue,
                                    a.worstN,
                                    a.worstX,
                                    a.worstForm,
                                    a.forms,
                                    a.moved,
                                    a.compared,
                                    {a.movedByForm[0], a.movedByForm[1], a.movedByForm[2]},
                                    {a.formWorst[0], a.formWorst[1], a.formWorst[2]},
                                    {a.formWorstN[0], a.formWorstN[1], a.formWorstN[2]},
                                    {a.formWorstX[0], a.formWorstX[1], a.formWorstX[2]},
                                    a.formBar[kPlainFormIndex]});
        };

    const int kFamilyFp64 = 0;
    const int kFamilyFp32 = 1;
    const int kFamilyFp16 = 2;
    const int kFamilyBf16 = 3;

    // The float lane's index, the lane a budget of BoysBudget::kFloat belongs to
    // beside the half lane's: read by the partition-carriage arms below, which are
    // keyed by budget rather than by family.
    const int kLaneSingle = static_cast<int>(boys::Precision::kFp32);

    // The four host families: both routes, both schemes, both axes and both
    // partitions, each family's own entry at each class of its own lane.

    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kCoarsest>();

    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kNarrow>();

    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kChebyshev,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kSplitClenshaw,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kArguments,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp64, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp32, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyFp16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();
    combHostCross.template operator()<kFamilyBf16, boys::FitRoute::kRationalMinimax,
                                      boys::EvalScheme::kHorner,
                                      boys::PackAxis::kOrders,
                                      boys::FitGranularity::kUniform>();

    // Which of the four device lanes an arm of this build measured here, and the sentence a member
    // of a lane is counted apart with where none did. Held per lane: a lane whose arm ran and which
    // holds no cell for a member of its cross is the hole the cross exists to find, and that is a
    // different statement from this host having no device at all.
    std::vector<char> combDeviceLaneArmed(static_cast<std::size_t>(combLaneCount), 0);
    std::vector<std::string> combDeviceLaneReason(static_cast<std::size_t>(combLaneCount));

#ifdef BOYS_GATE_CUDA

    // Half of the last representable digit of the returned value, the term a half lane's own
    // figure carries beside its base: binary16 has ten significand bits, which puts a digit of it
    // at 2^-11 of the binade, and bfloat16 has seven, which puts its at 2^-8.
    const auto halfUlp = [](double value, int mantissaBits) {
        const int exponent = std::ilogb(value);

        return exponent == FP_ILOGB0 || exponent == FP_ILOGBNAN || exponent < -1074
                   ? 0.0
                   : std::ldexp(1.0, exponent - mantissaBits - 1);
    };
    // The region-B member each device lane's arms read, off the library's own option table: every
    // arm launches an entry of the family its map returns, and those rows state the member they run.
    // A lane row's term beside the base belongs to the member that row names, so the two members are
    // two figures - 1.5e-07 against 2.3e-07 on fp32-device - and not one figure under two names.
    const auto combDeviceOptionPrecision = [](int lane) {
        switch (static_cast<boys::Precision>(lane))
        {
        case boys::Precision::kFp64Device:
            return boys::DeviceOptionPrecision::kFp64;
        case boys::Precision::kFp32Device:
            return boys::DeviceOptionPrecision::kFp32;
        case boys::Precision::kFp16Device:
            return boys::DeviceOptionPrecision::kFp16;
        case boys::Precision::kBf16Device:
            return boys::DeviceOptionPrecision::kBf16;
        case boys::Precision::kFp64:
        case boys::Precision::kFp32:
        case boys::Precision::kFp16:
        case boys::Precision::kBf16:
            break;
        }

        return boys::DeviceOptionPrecision::kCount;
    };

    // Per lane: the members the launched all-orders rows of that lane name, and how
    // many such rows this build serves - the count the set is read over.
    std::vector<std::vector<boys::RegionBExp>> combDeviceLaneMembers(
        static_cast<std::size_t>(combLaneCount));
    std::vector<std::size_t> combDeviceLaneMemberRows(static_cast<std::size_t>(combLaneCount), 0);

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        if (!combIsDeviceLane(lane))
        {
            continue;
        }

        const boys::DeviceOptionPrecision precision = combDeviceOptionPrecision(lane);

        for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
        {
            if (row.group != boys::DeviceOptionGroup::kLaunched ||
                row.shape != boys::DeviceOptionShape::kAllOrders || row.precision != precision ||
                !row.built)
            {
                continue;
            }

            ++combDeviceLaneMemberRows[static_cast<std::size_t>(lane)];

            std::vector<boys::RegionBExp>& members =
                combDeviceLaneMembers[static_cast<std::size_t>(lane)];

            if (std::find(members.begin(), members.end(), row.regionBExp) == members.end())
            {
                members.push_back(row.regionBExp);
            }
        }
    }

    // The region-B member the cross's device rows are read at: the member the four maps above
    // launch, all of which name a launched all-orders entry of the accurate family. Stated once,
    // because it is a statement about the maps and not a reading of the library's table - the
    // family names two members since the fast entries landed.
    constexpr boys::RegionBExp kCombDeviceRowMember = boys::RegionBExp::kAccurate;

    // The term beside the base for that member: the lane row's own where the row names
    // it, and 0.0 where the row's term belongs to the other member - the composition the
    // accessor makes, applied to the member the rows were read at rather than to the
    // member the row's term is under.
    const std::vector<double> combDeviceLaneMemberAdd = [&] {
        std::vector<double> term(static_cast<std::size_t>(combLaneCount), 0.0);

        for (int lane = 0; lane < combLaneCount; ++lane)
        {
            if (kCombDeviceRowMember ==
                combLaneRows[static_cast<std::size_t>(lane)].additiveMember)
            {
                term[static_cast<std::size_t>(lane)] =
                    combLaneRows[static_cast<std::size_t>(lane)].additive;
            }
        }

        return term;
    }();

    // ---- the device lanes' arms ---------------------------------------------
    // The four device lanes' reading, taken on the card this build was compiled for and through the
    // lanes' own entries: these have no host entry to call, and a host lane standing in would measure
    // another arithmetic under its name. A host that answers no device measures none of the four.
    bool deviceLaneUsable = false;

    // Why no arm of this build could run on this host: null while the device is usable, and set on
    // every path that leaves it unusable. The state is a pair and not a default sentence because a
    // lane that is not measured is counted apart with this sentence, so it is a claim about a run.
    const char* deviceUnusable = nullptr;

    {
        int combDeviceCount = 0;

        if (cudaGetDeviceCount(&combDeviceCount) != cudaSuccess)
        {
            deviceUnusable =
                "this build carries the CUDA lane (BUILD_CUDA=ON) and the CUDA runtime did not "
                "report a device on this host (cudaGetDeviceCount failed), so the entries that "
                "would have been measured on the card were not run";
        }
        else if (combDeviceCount == 0)
        {
            deviceUnusable =
                "this build carries the CUDA lane (BUILD_CUDA=ON) and this host answers no CUDA "
                "device, so the lanes' entries are in the binary and there is no card to run "
                "them on";
        }
        else if (cudaSetDevice(0) != cudaSuccess ||
                 boys::BoysCuda::InitializeTables() != boys::BoysStatus::kSuccess)
        {
            deviceUnusable =
                "this build carries the CUDA lane (BUILD_CUDA=ON) and the lanes' degree tables "
                "could not be made resident on this host's device (BoysCuda::InitializeTables), "
                "so no entry of the lanes was measured";
        }
        else
        {
            deviceLaneUsable = true;
        }
    }

    // The device answered: every one of the four lanes' arms is armed, and
    // each of them clears its own flag if its own allocations or its own entry
    // turn out not to run. A lane that is not armed is counted apart with this
    // host's reason, which is why the two are written together here.
    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        if (!combIsDeviceLane(lane))
        {
            continue;
        }

        combDeviceLaneArmed[static_cast<std::size_t>(lane)] = deviceLaneUsable ? 1 : 0;

        if (deviceUnusable != nullptr)
        {
            combDeviceLaneReason[static_cast<std::size_t>(lane)] = deviceUnusable;
        }
    }

    // Which device lanes this build has a second arm for, read from the revision's own option
    // table: a lane whose table carries no built launched single-order row at RegionBExp::kFast is
    // a lane no arm here can measure that member on.
    const std::vector<char> combDeviceLaneFastArmed = [&] {
        std::vector<char> armed(static_cast<std::size_t>(combLaneCount), 0);

        for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
        {
            if (!row.built || row.group != boys::DeviceOptionGroup::kLaunched ||
                row.shape != boys::DeviceOptionShape::kSingle ||
                row.regionBExp != boys::RegionBExp::kFast)
            {
                continue;
            }

            for (int lane = 0; lane < combLaneCount; ++lane)
            {
                if (!combIsDeviceLane(lane) || row.precision != combDeviceOptionPrecision(lane))
                {
                    continue;
                }

                if (lane == static_cast<int>(boys::Precision::kFp32Device) ||
                    lane == static_cast<int>(boys::Precision::kFp64Device))
                {
                    armed[static_cast<std::size_t>(lane)] = 1;
                }

#ifdef BOYS_GATE_FP16
                if (lane == static_cast<int>(boys::Precision::kFp16Device) ||
                    lane == static_cast<int>(boys::Precision::kBf16Device))
                {
                    armed[static_cast<std::size_t>(lane)] = 1;
                }
#endif
            }
        }

        return armed;
    }();

    // The arms per lane are a reading of the members above: the cross's arm reads the member the
    // launched all-orders family's rows run, and the fast-member arm reads the other where this
    // build's surface carries the lane's single-order entry at it. A member no arm reads fails here.
    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        if (!combIsDeviceLane(lane) || !deviceLaneUsable)
        {
            continue;
        }

#ifndef BOYS_GATE_FP16
        // The half lane's two stores' entries are behind the seam this build has
        // closed, so their families hold no built row here and the two lanes are
        // counted apart below with that sentence rather than failed on for it.
        if (lane == static_cast<int>(boys::Precision::kFp16Device) ||
            lane == static_cast<int>(boys::Precision::kBf16Device))
        {
            continue;
        }
#endif

        const std::vector<boys::RegionBExp>& members =
            combDeviceLaneMembers[static_cast<std::size_t>(lane)];
        const std::size_t rows = combDeviceLaneMemberRows[static_cast<std::size_t>(lane)];

        if (rows == 0)
        {
            std::printf("  the %s lane: this build serves no launched all-orders entry of it, so "
                        "the arm below has nothing to launch and its members are measured by no "
                        "cell here\n",
                        combLaneRows[static_cast<std::size_t>(lane)].name);
            failed = true;
        }
        else if (std::find(members.begin(), members.end(), kCombDeviceRowMember) == members.end())
        {
            std::printf("  the %s lane: the launched all-orders entries of this build do not name "
                        "the region-B member this cross's device maps launch, so the figure these "
                        "rows are judged at is not one of the members the table states - the maps "
                        "and the table have parted company\n",
                        combLaneRows[static_cast<std::size_t>(lane)].name);
            failed = true;
        }
        else if (members.size() > 1 &&
                 !combDeviceLaneFastArmed[static_cast<std::size_t>(lane)])
        {
            std::printf("  the %s lane: the launched all-orders entries of this build name %zu "
                        "region-B member(s) of it and this build's surface carries no entry of "
                        "that lane at the second that an arm of this gate launches, so the member "
                        "is measured by no cell here\n",
                        combLaneRows[static_cast<std::size_t>(lane)].name,
                        members.size());
            failed = true;
        }
    }

    const std::size_t combDeviceCells = ref.count * static_cast<std::size_t>(nmax + 1);
    const std::vector<int> combDeviceTops(ref.count, nmax);
    GateDeviceBuffer<int> combDeviceN(ref.count);
    GateDeviceBuffer<double> combDeviceX(ref.count);
    GateDeviceBuffer<float> combDeviceValues(combDeviceCells);

    // The buffers are the sweep's own and the arm below reads them, so what they could not do is
    // asked once, here, rather than inside the sweep: a device that refuses an allocation or an
    // upload is a device this lane measures nothing on, and the cross says so about this lane.
    if (deviceLaneUsable &&
        (!combDeviceN.Upload(combDeviceTops) || !combDeviceX.Upload(ref.xf) ||
         !combDeviceValues.ok()))
    {
        combDeviceLaneArmed[static_cast<std::size_t>(combDeviceLane)] = 0;
        combDeviceLaneReason[static_cast<std::size_t>(combDeviceLane)] =
            "this build carries the CUDA lane (BUILD_CUDA=ON) and the device refused an "
            "allocation or the argument upload this lane's measurement needs, so no entry of it "
            "was measured";
    }

    // The forms these arms launch their cells at: the three the axis answers, written out for the
    // reason the host sweeps' three are - the form is a template argument of the device engine, so a
    // form an arm names is an instantiation. Each cell is read at every one and the worst judged
    // against the figure that form's own row states; the list is reconciled with the accessor.
    constexpr std::array<boys::DivisionForm, kCombForms> kDeviceForms = {
        boys::DivisionForm::kExactDivision,
        boys::DivisionForm::kPlainReciprocal,
        boys::DivisionForm::kRefinedReciprocal};

    // The float lane's arm: one reading per (route, scheme, partition, packing
    // axis), the same shape as the two arms above with the entry selected by the
    // map rather than by a policy template, and each cell read at every form the
    // axis answers.
    const auto combDeviceSweep =
        [&]<boys::FitRoute kRoute, boys::EvalScheme kScheme, boys::PackAxis kAxis,
            boys::FitGranularity kGran>(int lane) {
            const double laneBound = combLaneRows[static_cast<std::size_t>(lane)].bound;

            // The term beside the base for the member this lane's entries run, and
            // not the term its row publishes: the two are one number on a lane whose
            // row names that member, and on fp32-device they are 0.0 against 8e-8.
            const double laneMemberAdd = combDeviceLaneMemberAdd[static_cast<std::size_t>(lane)];
            const double lanePlainAdd =
                combLaneRows[static_cast<std::size_t>(lane)].plainAdditive;

            constexpr auto kEntry = GateDeviceEntry(kRoute, kScheme, kGran, kAxis);

            if (kEntry == nullptr)
            {
                std::printf("  the device lane: no entry of this build's CUDA surface "
                            "serves a member the cross names, so no cell here "
                            "measures it\n");
                failed = true;

                return;
            }

            CombAccum a;
            std::array<std::vector<float>, kCombForms> devOut;

            // One download target per form, sized like the arm's own buffer: the
            // download writes through the vector's data, so an empty one is a
            // refused copy rather than an empty reading.
            for (std::vector<float>& out : devOut)
            {
                out.resize(combDeviceCells);
            }

            // The figure the row is judged by, computed the way the accessor computes it: the
            // lane's base, plus the term the lane adds beside it where the member read is the one
            // its row names the term under, plus the plain reciprocal's own figure where stated.
            a.bound = laneBound + laneMemberAdd;
            a.formBar[static_cast<std::size_t>(boys::DivisionForm::kPlainReciprocal)] =
                laneBound + lanePlainAdd + laneMemberAdd;

            for (std::size_t f = 0; f < kCombForms; ++f)
            {
                const boys::BoysStatus status = kEntry(combDeviceN.get(),
                                                       combDeviceX.get(),
                                                       combDeviceValues.get(),
                                                       ref.count,
                                                       nullptr,
                                                       kDeviceForms[f]);

                if (status != boys::BoysStatus::kSuccess ||
                    cudaDeviceSynchronize() != cudaSuccess ||
                    !combDeviceValues.Download(devOut[f]))
                {
                    std::printf("  the device lane: the entry for one member of the cross "
                                "did not run (BoysStatus %d), so that member "
                                "is measured by no cell here\n",
                                static_cast<int>(status));
                    failed = true;

                    return;
                }
            }

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t i = 0; i < ref.count; ++i)
                {
                    const std::size_t e = ref.Index(n, i);
                    const double got[kCombForms] = {
                        static_cast<double>(devOut[0][e]),
                        static_cast<double>(devOut[1][e]),
                        static_cast<double>(devOut[2][e])};

                    a.addAtForms(n, ref.xf[i], got, kCombForms, ref.vf[e]);
                }
            }

            combMeasured.push_back({lane,
                                    static_cast<int>(kRoute),
                                    static_cast<int>(kScheme),
                                    static_cast<int>(kGran),
                                    static_cast<int>(kAxis),
                                    a.cells,
                                    a.below,
                                    a.over,
                                    a.worst,
                                    a.bound,
                                    a.judgedTo,
                                    a.judgedPlain,
                                    a.judgedValue,
                                    a.worstN,
                                    a.worstX,
                                    a.worstForm,
                                    a.forms,
                                    a.moved,
                                    a.compared,
                                    {a.movedByForm[0], a.movedByForm[1],
                                     a.movedByForm[2]},
                                    {a.formWorst[0], a.formWorst[1], a.formWorst[2]},
                                    {a.formWorstN[0], a.formWorstN[1], a.formWorstN[2]},
                                    {a.formWorstX[0], a.formWorstX[1], a.formWorstX[2]},
                                    a.formBar[kPlainFormIndex]});
        };

    // The float lane's arm, over every member of the cross that lane claims: four axes of the
    // space read off the same tables the cross enumerates them from. The count is the cross's own
    // arithmetic for this lane and the claim side below is what holds this list to it.
    if (combDeviceLaneArmed[static_cast<std::size_t>(combDeviceLane)])
    {
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kCoarsest>(combDeviceLane);

        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kNarrow>(combDeviceLane);

        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kArguments,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kChebyshev,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kSplitClenshaw,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
        combDeviceSweep.template operator()<boys::FitRoute::kRationalMinimax,
                                            boys::EvalScheme::kHorner,
                                            boys::PackAxis::kOrders,
                                            boys::FitGranularity::kUniform>(combDeviceLane);
    }

/// The two lanes beside the float one, read the way the float lane's arm above reads its own:
/// one reading per member of the lane's own cross, over the whole committed grid, judged cell by
/// cell against the figure the committed reference holds for the arithmetic the lane runs. A member
/// the lane's own carrier refuses is not a member of this arm, and a member the carrier carries and
/// this arm cannot serve is printed and failed on rather than passed over.
///
/// \tparam TVal the value type the lane's entries return
///
/// \param lane     the lane being measured
/// \param entryOf  the lane's map from a member of the cross to its entry
/// \param args     the arguments the lane's entries take, per reference argument, in the
///                 reference's own order
/// \param want     the reference column the cells are judged against, indexed
///                 Reference::Index(n, i)
/// \param halfLane whether the lane's figure is the half lane's - the base plus half a
///                 representable digit of the value the entry returned, with the cells at or
///                 below that bar counted below it rather than judged
/// \param mantissaBits the significand width of the format the lane stores in: half a digit of
///                 the returned value is 2^(ilogb(value) - mantissaBits - 1)
    const auto combDeviceRunLane =
        [&]<typename TVal>(int lane, auto entryOf, const auto& args,
                           const std::vector<double>& want, bool halfLane,
                           int mantissaBits) {
            using TArg = typename std::decay_t<decltype(args)>::value_type;

            const char* const laneName = combLaneRows[static_cast<std::size_t>(lane)].name;
            const double laneBound = combLaneRows[static_cast<std::size_t>(lane)].bound;
            const double laneMemberAdd = combDeviceLaneMemberAdd[static_cast<std::size_t>(lane)];
            const double lanePlainAdd =
                combLaneRows[static_cast<std::size_t>(lane)].plainAdditive;

            // This lane's own allocations, asked for here rather than shared
            // with the arm above: a device that refuses one of them leaves this
            // lane unmeasured and its neighbours measured, and the sentence the
            // cross counts this lane's members apart with says exactly that.
            GateDeviceBuffer<int> laneN(ref.count);
            GateDeviceBuffer<TArg> laneArgs(args.size());
            GateDeviceBuffer<TVal> laneValues(combDeviceCells);
            std::array<std::vector<TVal>, kCombForms> laneOut;

            // One download target per form, sized like the buffer above and for
            // the reason the float arm's targets are.
            for (std::vector<TVal>& out : laneOut)
            {
                out.resize(combDeviceCells);
            }

            if (!laneN.Upload(combDeviceTops) || !laneArgs.Upload(args) || !laneValues.ok())
            {
                combDeviceLaneArmed[static_cast<std::size_t>(lane)] = 0;
                combDeviceLaneReason[static_cast<std::size_t>(lane)] = Fmt(
                    "this build carries the CUDA lane (BUILD_CUDA=ON) and the device refused an "
                    "allocation or the argument upload the %s lane's measurement needs, so no "
                    "entry of it was measured",
                    laneName);

                return;
            }

            for (const boys::FitRoute route : combRoutes[static_cast<std::size_t>(lane)])
            {
                const char* routeName = "";

                for (const boys::FitRouteInfo& row : combRoutesFor(lane))
                {
                    if (row.route == route)
                    {
                        routeName = row.name;
                    }
                }

                for (const boys::EvalSchemeInfo& schemeRow : boys::BoysEvalSchemes())
                {
                    for (const boys::FitGranularityInfo& partitionRow :
                         boys::BoysFitGranularities())
                    {
                        for (const boys::PackAxisInfo& axisRow : boys::BoysPackAxes())
                        {
                            // What this lane's own carrier says about the
                            // member: a refusal is the cross's owed row, and
                            // this arm is not the instrument that decides it.
                            const boys::AccuracyFigure guaranteed = boys::BoysAccuracyGuaranteed(
                                static_cast<boys::Precision>(lane),
                                route,
                                schemeRow.scheme,
                                axisRow.axis,
                                partitionRow.granularity,
                                kGateDivisionForm);

                            if (!guaranteed.available)
                            {
                                continue;
                            }

                            const auto entry = entryOf(route,
                                                       schemeRow.scheme,
                                                       partitionRow.granularity,
                                                       axisRow.axis);

                            if (entry == nullptr)
                            {
                                std::printf(
                                    "  the %s lane: the library states it carries (%s, %s, %s, "
                                    "%s) and this gate has no entry of the lane's surface for "
                                    "it, so that member is measured by no cell here\n",
                                    laneName,
                                    routeName,
                                    boys::EvalSchemeName(schemeRow.scheme),
                                    partitionRow.name,
                                    axisRow.name);
                                failed = true;

                                continue;
                            }

                            CombAccum a;

                            // The figure the row is judged by, computed the way the accessor
                            // computes it: the lane's base, plus the term the lane adds where the
                            // member read is the one its row names the term under - and on the half
                            // lane plus half a representable digit of the value the cell returned.
                            a.bound = laneBound + laneMemberAdd;
                            a.formBar[static_cast<std::size_t>(
                                boys::DivisionForm::kPlainReciprocal)] =
                                laneBound + lanePlainAdd + laneMemberAdd;
                            a.ceiling = halfLane ? a.bound : 0.0;

                            bool ran = false;

                            for (std::size_t f = 0; f < kCombForms; ++f)
                            {
                                const boys::BoysStatus status =
                                    entry(laneN.get(),
                                          laneArgs.get(),
                                          laneValues.get(),
                                          ref.count,
                                          nullptr,
                                          kDeviceForms[f]);

                                if (status != boys::BoysStatus::kSuccess ||
                                    cudaDeviceSynchronize() != cudaSuccess ||
                                    !laneValues.Download(laneOut[f]))
                                {
                                    std::printf("  the %s lane: the entry for one member of the "
                                                "cross did not run (BoysStatus %d), so that "
                                                "member is measured by no cell here\n",
                                                laneName,
                                                static_cast<int>(status));
                                    failed = true;

                                    break;
                                }

                                ran = true;
                            }

                            if (!ran)
                            {
                                continue;
                            }

                            for (int n = 0; n <= nmax; ++n)
                            {
                                for (std::size_t i = 0; i < ref.count; ++i)
                                {
                                    const std::size_t e = ref.Index(n, i);
                                    const double got[kCombForms] = {
                                        static_cast<double>(laneOut[0][e]),
                                        static_cast<double>(laneOut[1][e]),
                                        static_cast<double>(laneOut[2][e])};
                                    const double ulp[kCombForms] = {
                                        halfLane ? halfUlp(got[0], mantissaBits) : 0.0,
                                        halfLane ? halfUlp(got[1], mantissaBits) : 0.0,
                                        halfLane ? halfUlp(got[2], mantissaBits) : 0.0};

                                    a.addAtForms(n,
                                                 static_cast<double>(args[i]),
                                                 got,
                                                 kCombForms,
                                                 want[e],
                                                 halfLane ? ulp : nullptr);
                                }
                            }

                            combMeasured.push_back({lane,
                                                    static_cast<int>(route),
                                                    static_cast<int>(schemeRow.scheme),
                                                    static_cast<int>(partitionRow.granularity),
                                                    static_cast<int>(axisRow.axis),
                                                    a.cells,
                                                    a.below,
                                                    a.over,
                                                    a.worst,
                                                    a.bound,
                                                    a.judgedTo,
                                                    a.judgedPlain,
                                                    a.judgedValue,
                                                    a.worstN,
                                                    a.worstX,
                                                    a.worstForm,
                                                    a.forms,
                                                    a.moved,
                                                    a.compared,
                                                    {a.movedByForm[0], a.movedByForm[1],
                                                     a.movedByForm[2]},
                                                    {a.formWorst[0], a.formWorst[1],
                                                     a.formWorst[2]},
                                                    {a.formWorstN[0], a.formWorstN[1],
                                                     a.formWorstN[2]},
                                                    {a.formWorstX[0], a.formWorstX[1],
                                                     a.formWorstX[2]},
                                                    a.formBar[kPlainFormIndex]});
                        }
                    }
                }
            }
        };

    // The double lane: the reference's own double argument and double value,
    // which is the pair its entries take and the pair the sibling gate measures
    // these same entries at (tests/boys_cuda_accuracy_gate.cpp, the
    // AllOrdersF64 rows), judged at the lane's own figure.
    if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp64Device)])
    {
        combDeviceRunLane.template operator()<double>(
            static_cast<int>(boys::Precision::kFp64Device),
            GateDeviceEntryF64,
            ref.x,
            ref.v,
            false,
            0);
    }

    // The half lane: the argument is the half the reference's `x16` column is, and the cells are
    // judged against `v16`. The reference loader checks x16 is the half the double rounds to for
    // every finite entry, so the rounding is the grid's own statement rather than this arm's.
#ifdef BOYS_GATE_FP16
    if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp16Device)])
    {
        std::vector<boys::F16> halfArgs(ref.count);

        for (std::size_t i = 0; i < ref.count; ++i)
        {
            halfArgs[i] = boys::F16(static_cast<float>(ref.x[i]));
        }

        combDeviceRunLane.template operator()<boys::F16>(
            static_cast<int>(boys::Precision::kFp16Device),
            GateDeviceEntryF16,
            halfArgs,
            ref.v16,
            true,
            kF16MantissaBits);
    }

    // The half lane's other store: the argument rounded to this format, judged against `vb`, the
    // column that goes with it - the pair every other bf16 arm of this gate reads. Judging it
    // against the finer column's `v16` would charge the lane for the distance between two arguments.
    if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kBf16Device)])
    {
        std::vector<boys::Bf16> bf16Args(ref.count);

        for (std::size_t i = 0; i < ref.count; ++i)
        {
            bf16Args[i] = boys::Bf16(static_cast<float>(ref.x[i]));
        }

        combDeviceRunLane.template operator()<boys::Bf16>(
            static_cast<int>(boys::Precision::kBf16Device),
            GateDeviceEntryBf16,
            bf16Args,
            ref.vb,
            true,
            kBf16MantissaBits);
    }
#else
    // The CUDA surface's fp16 entries are declared behind the BoysFp16 seam this build has closed,
    // so the half lane's carried member has no entry in this binary to measure - an entry missing,
    // not the arithmetic's existence - and the member is counted apart with that sentence.
    combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp16Device)] = 0;
    combDeviceLaneReason[static_cast<std::size_t>(boys::Precision::kFp16Device)] =
        "this build's CUDA surface carries its fp16 entries behind the BoysFp16 seam, which this "
        "build has closed, so the half lane's carried member has no entry in this binary to "
        "measure; a build with the seam open measures that member on the card";
    combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kBf16Device)] = 0;
    combDeviceLaneReason[static_cast<std::size_t>(boys::Precision::kBf16Device)] =
        "this build's CUDA surface carries its bfloat16 entries behind the BoysFp16 seam, which "
        "this build has closed, so the half lane's other store has no entry in this binary to "
        "measure; a build with the seam open measures that member on the card";
#endif // BOYS_GATE_FP16

    // The region-B member the arms above do not read is launched here, through the lane's
    // single-order fast entry; the double lane is only named, because this build declares
    // BoysCuda::SingleF64Fast and defines it nowhere, so a call of it is a link error. Each cell is
    // judged at the accessor's own figure for the member read - 2.3e-07 on fp32-device.
    const auto combDeviceFastMember =
        [&]<typename TArg, typename TVal>(int lane, const char* entryName, auto entry,
                                          const std::vector<TArg>& args,
                                          const std::vector<double>& want, bool halfLane,
                                          int mantissaBits) {
            const boys::LaneContractInfo& laneRow = combLaneRows[static_cast<std::size_t>(lane)];
            const double laneBound = laneRow.bound;
            const double memberAdd =
                laneRow.additiveMember == boys::RegionBExp::kFast ? laneRow.additive : 0.0;

            GateDeviceBuffer<int> laneN(ref.count);
            GateDeviceBuffer<TArg> laneArgs(args.size());
            GateDeviceBuffer<TVal> laneValues(ref.count);
            std::vector<int> tops(ref.count, 0);
            std::vector<TVal> laneOut(ref.count);

            if (!laneN.Upload(tops) || !laneArgs.Upload(args) || !laneValues.ok())
            {
                std::printf("  the %s lane's fast member: the device refused an allocation or the "
                            "argument upload this arm needs, so the member is measured by no cell "
                            "here\n",
                            laneRow.name);
                failed = true;

                return;
            }

            CombAccum a;
            a.bound = laneBound + memberAdd;
            a.formBar[static_cast<std::size_t>(boys::DivisionForm::kPlainReciprocal)] =
                laneBound + laneRow.plainAdditive + memberAdd;
            a.ceiling = halfLane ? a.bound : 0.0;

            for (std::size_t f = 0; f < kCombForms; ++f)
            {
                for (int n = 0; n <= nmax; ++n)
                {
                    std::fill(tops.begin(), tops.end(), n);

                    if (!laneN.Upload(tops))
                    {
                        std::printf("  the %s lane's fast member: the order upload this arm needs "
                                    "was refused, so the member is measured by no cell here\n",
                                    laneRow.name);
                        failed = true;

                        return;
                    }

                    const boys::BoysStatus status =
                        entry(laneN.get(),
                              laneArgs.get(),
                              laneValues.get(),
                              ref.count,
                              nullptr,
                              kDeviceForms[f]);

                    if (status != boys::BoysStatus::kSuccess ||
                        cudaDeviceSynchronize() != cudaSuccess || !laneValues.Download(laneOut))
                    {
                        std::printf("  the %s lane's fast member: the entry did not run "
                                    "(BoysStatus %d), so the member is measured by no cell here\n",
                                    laneRow.name,
                                    static_cast<int>(status));
                        failed = true;

                        return;
                    }

                    for (std::size_t i = 0; i < ref.count; ++i)
                    {
                        const double got = static_cast<double>(laneOut[i]);

                        a.add(n,
                              static_cast<double>(args[i]),
                              got,
                              want[ref.Index(n, i)],
                              halfLane ? halfUlp(got, mantissaBits) : 0.0,
                              static_cast<int>(f));
                    }
                }
            }

            std::printf("  the %s lane's fast member (%s): %zu cell(s) over the committed grid at "
                        "each of the %zu form(s), worst %.6g at n=%d, x=%.6g, judged at %.6g, "
                        "%zu outside it\n",
                        laneRow.name,
                        entryName,
                        a.cells,
                        kCombForms,
                        a.worst,
                        a.worstN,
                        a.worstX,
                        a.judgedTo,
                        a.over);

            if (a.over != 0)
            {
                failed = true;
            }
        };

    if (deviceLaneUsable)
    {
        if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp32Device)])
        {
            combDeviceFastMember.template operator()<double, float>(
                static_cast<int>(boys::Precision::kFp32Device),
                "BoysCuda::SingleF32<RegionBExp::kFast>",
                GateDeviceEntryFast(boys::FitRoute::kChebyshev,
                                    boys::EvalScheme::kSplitClenshaw,
                                    boys::FitGranularity::kCoarsest,
                                    boys::PackAxis::kArguments),
                ref.xf,
                ref.vf,
                false,
                0);

            // The same member through the surface's other group: the in-kernel entry,
            // called from a kernel of this gate's own. A handle the card would not take
            // is a reason for the arm not to run rather than a set of values judged
            // under the launch's name, and it is stated here beside the call.
            boys::BoysDeviceTables fastDeviceTables;

            if (boys::BoysCuda::DeviceTables(&fastDeviceTables) == boys::BoysStatus::kSuccess)
            {
                // The handle is this file's, because the translation unit that holds the
                // kernel may include the device header and not the host one; and the
                // launcher answers its status as an int for the same reason, so the two
                // are made one call shape here rather than in either file.
                const auto fastDeviceEntry = [&fastDeviceTables](const int* n, const double* x,
                                                                 float* out, std::size_t count,
                                                                 void* stream,
                                                                 boys::DivisionForm form) {
                    return BoysGateFastDeviceSingleF32(&fastDeviceTables, n, x, out, count, stream,
                                                       form) == 0
                               ? boys::BoysStatus::kSuccess
                               : boys::BoysStatus::kDeviceError;
                };

                combDeviceFastMember.template operator()<double, float>(
                    static_cast<int>(boys::Precision::kFp32Device),
                    "BoysDeviceSingleF32<kForm, RegionBExp::kFast>",
                    fastDeviceEntry,
                    ref.xf,
                    ref.vf,
                    false,
                    0);
            }
            else
            {
                std::printf("  the fp32-device lane's fast member, in the caller's own kernel: "
                            "the device handle this arm's kernels read could not be filled "
                            "(BoysCuda::DeviceTables), so the in-kernel entry's cells are "
                            "measured by no cell here\n");
                failed = true;
            }
        }

        if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp64Device)] &&
            combDeviceLaneFastArmed[static_cast<std::size_t>(boys::Precision::kFp64Device)])
        {
            combDeviceFastMember.template operator()<double, double>(
                static_cast<int>(boys::Precision::kFp64Device),
                "BoysCuda::SingleF64Fast",
                GateDeviceEntryF64Fast(boys::FitRoute::kChebyshev,
                                       boys::EvalScheme::kSplitClenshaw,
                                       boys::FitGranularity::kCoarsest,
                                       boys::PackAxis::kArguments),
                ref.x,
                ref.v,
                false,
                0);
        }

#ifdef BOYS_GATE_FP16
        if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kFp16Device)])
        {
            std::vector<boys::F16> fastHalfArgs(ref.count);

            for (std::size_t i = 0; i < ref.count; ++i)
            {
                fastHalfArgs[i] = boys::F16(static_cast<float>(ref.x[i]));
            }

            combDeviceFastMember.template operator()<boys::F16, boys::F16>(
                static_cast<int>(boys::Precision::kFp16Device),
                "BoysCuda::SingleF16Fast",
                GateDeviceEntryF16Fast(boys::FitRoute::kChebyshev,
                                       boys::EvalScheme::kSplitClenshaw,
                                       boys::FitGranularity::kCoarsest,
                                       boys::PackAxis::kArguments),
                fastHalfArgs,
                ref.v16,
                true,
                kF16MantissaBits);
        }

        if (combDeviceLaneArmed[static_cast<std::size_t>(boys::Precision::kBf16Device)])
        {
            std::vector<boys::Bf16> fastBf16Args(ref.count);

            for (std::size_t i = 0; i < ref.count; ++i)
            {
                fastBf16Args[i] = boys::Bf16(static_cast<float>(ref.x[i]));
            }

            combDeviceFastMember.template operator()<boys::Bf16, boys::Bf16>(
                static_cast<int>(boys::Precision::kBf16Device),
                "BoysCuda::SingleBf16Fast",
                GateDeviceEntryBf16Fast(boys::FitRoute::kChebyshev,
                                        boys::EvalScheme::kSplitClenshaw,
                                        boys::FitGranularity::kCoarsest,
                                        boys::PackAxis::kArguments),
                fastBf16Args,
                ref.vb,
                true,
                kBf16MantissaBits);
        }
#endif
    }

    if (!combDeviceLaneFastArmed[static_cast<std::size_t>(boys::Precision::kFp64Device)])
    {
        std::size_t fp64FastRows = 0;

        for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
        {
            if (row.precision == boys::DeviceOptionPrecision::kFp64 &&
                row.regionBExp == boys::RegionBExp::kFast)
            {
                ++fp64FastRows;
            }
        }

        std::printf("  the fp64-device lane's fast member: this build's table carries %zu row(s) of "
                    "that lane at RegionBExp::kFast and none of them is a built launched "
                    "single-order row, which is the entry this gate reaches that member through - "
                    "so the member's cells are measured by no cell here rather than judged at the "
                    "accurate member's figure\n",
                    fp64FastRows);
    }
#endif // BOYS_GATE_CUDA

    // Whether each device lane was measured here, and the sentence its members are counted apart
    // with where it was not, are the two vectors the arms above wrote into - per lane, because a
    // member is either a row this run measured on the card or one counted apart for a reason this
    // build and this host can be asked about, and never for the lane's identity.
#ifndef BOYS_GATE_CUDA
    const char* const combDeviceLaneAbsent =
        "this build has no CUDA lane (BUILD_CUDA=OFF), so the device lanes' entries are not in "
        "this binary and no cell here can run them. A build with BUILD_CUDA=ON carries the lanes "
        "and measures these members on the card, and this sentence is replaced there by the "
        "card's own reading";

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        if (combIsDeviceLane(lane))
        {
            combDeviceLaneReason[static_cast<std::size_t>(lane)] = combDeviceLaneAbsent;
        }
    }
#endif

    // How much of the device surface this record answers for: taken here, after the last arm,
    // because a lane the card is present for and whose entry this build does not define is a lane
    // this record did not measure.
    {
        std::size_t combDeviceLanesMeasured = 0;
        std::size_t combDeviceLanes = 0;

        for (int lane = 0; lane < combLaneCount; ++lane)
        {
            if (!combIsDeviceLane(lane))
            {
                continue;
            }

            ++combDeviceLanes;
            combDeviceLanesMeasured +=
                combDeviceLaneArmed[static_cast<std::size_t>(lane)] ? 1 : 0;
        }

        std::printf("device lanes : %zu of %zu measured on a card\n",
                    combDeviceLanesMeasured,
                    combDeviceLanes);
    }

    // ---- the cross, judged against what the accessor answers ----------------
    std::vector<Combination> combinations;
    std::size_t combClaimedCarried = 0;

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        const std::span<const boys::FitRouteInfo> laneRoutes = combRoutesFor(lane);

        for (int ri = 0; ri < static_cast<int>(combRoutes[static_cast<std::size_t>(lane)].size());
             ++ri)
        {
            const boys::FitRoute route = combRoutes[static_cast<std::size_t>(lane)]
                [static_cast<std::size_t>(ri)];
            const char* routeName = "";

            for (const boys::FitRouteInfo& row : laneRoutes)
            {
                if (row.route == route && routeName[0] == '\0')
                {
                    routeName = row.name;
                }
            }

            for (int si = 0; si < static_cast<int>(combSchemes); ++si)
            {
                const boys::EvalScheme scheme = boys::BoysEvalSchemes()
                    [static_cast<std::size_t>(si)].scheme;

                for (int gi = 0; gi < static_cast<int>(combPartitions); ++gi)
                {
                    const boys::FitGranularityInfo& partition =
                        boys::BoysFitGranularities()[static_cast<std::size_t>(gi)];

                    for (int ai = 0; ai < static_cast<int>(combAxes); ++ai)
                    {
                        const boys::PackAxisInfo& axisRow =
                            boys::BoysPackAxes()[static_cast<std::size_t>(ai)];
                        // The division form is named, and it is the one argument this call was
                        // leaving to the build: the accessor answers the figure for the form it is
                        // given, and the figure this row is judged by is the lane's base, so the two
                        // are one number. The region-B member is named for the same reason.
#ifdef BOYS_GATE_CUDA
                        // The member the device arm read this row at, which is the member
                        // its figure was taken at (kCombDeviceRowMember above). A build
                        // without the lane has no arm and no member read: its device rows
                        // are counted apart under the build's own default.
                        const boys::RegionBExp rowExp = kCombDeviceRowMember;
#else
                        const boys::RegionBExp rowExp = boys::kDefaultHostRegionBExp;
#endif
                        const boys::AccuracyFigure guaranteed = boys::BoysAccuracyGuaranteed(
                            static_cast<boys::Precision>(lane),
                            route,
                            scheme,
                            axisRow.axis,
                            partition.granularity,
                            boys::DivisionForm::kRefinedReciprocal,
                            rowExp);
                        // The same figure for the form this build's unnamed calls divide in,
                        // which is what the two entries below answer for: neither BoysAccuracy
                        // Delivered nor QueryCombination takes a form, so both read the build's
                        // default one, and the tolerance query's own `bound` is this figure.
                        const boys::AccuracyFigure guaranteedForm =
                            boys::BoysAccuracyGuaranteed(static_cast<boys::Precision>(lane),
                                                         route,
                                                         scheme,
                                                         axisRow.axis,
                                                         partition.granularity);
                        // The same accessor read at the form a row publishes a figure of its
                        // own for: on a row that states no term for that form this is the figure
                        // above; on one that does, it is the base plus the term, and the row's own
                        // arm judged that form by the bar this figure is held to below.
                        const boys::AccuracyFigure guaranteedPlain = boys::BoysAccuracyGuaranteed(
                            static_cast<boys::Precision>(lane),
                            route,
                            scheme,
                            axisRow.axis,
                            partition.granularity,
                            boys::DivisionForm::kPlainReciprocal,
                            rowExp);
                        const boys::AccuracyFigure delivered = boys::BoysAccuracyDelivered(
                            static_cast<boys::Precision>(lane),
                            route,
                            scheme,
                            axisRow.axis,
                            partition.granularity);

                        Combination c;
                        c.axes = Fmt("%s, %s, %s, %s, %s",
                                     combLaneRows[static_cast<std::size_t>(lane)].name,
                                     routeName,
                                     boys::EvalSchemeName(scheme),
                                     partition.name,
                                     axisRow.name);
                        c.accessorBound = guaranteed.value;
                        c.accessorFormBound = guaranteedForm.value;
                        c.accessorPlainBound = guaranteedPlain.value;
                        c.accessorDelivered = delivered.value;
                        c.accessorDeliveredKnown = delivered.available;

                        // The tolerance query, asked twice on every row: at the figure the row is
                        // judged by, which a bound at or below itself must answer inside, and at
                        // half of it, which nothing at that figure can. The two together say the
                        // entry compares the request with the figure it reports - `guaranteedForm`.
                        const boys::CombinationCoverage askedAtBound =
                            boys::QueryCombination(static_cast<boys::Precision>(lane),
                                                   route,
                                                   scheme,
                                                   axisRow.axis,
                                                   partition.granularity,
                                                   guaranteedForm.value);
                        const boys::CombinationCoverage askedAtHalf =
                            boys::QueryCombination(static_cast<boys::Precision>(lane),
                                                   route,
                                                   scheme,
                                                   axisRow.axis,
                                                   partition.granularity,
                                                   guaranteedForm.value * 0.5);

                        c.atBound = askedAtBound.verdict;
                        c.atHalf = askedAtHalf.verdict;
                        c.toleranceBound = askedAtBound.bound;
                        c.toleranceDelivered = askedAtBound.delivered;
                        c.toleranceDeliveredKnown = askedAtBound.deliveredKnown;
                        c.toleranceReason = askedAtBound.reason;

                        const CombCell* cell = nullptr;

                        for (const CombCell& m : combMeasured)
                        {
                            if (m.lane == lane &&
                                m.route == static_cast<int>(route) &&
                                m.scheme == static_cast<int>(scheme) &&
                                m.partition == static_cast<int>(partition.granularity) &&
                                m.axis == static_cast<int>(axisRow.axis))
                            {
                                cell = &m;

                                break;
                            }
                        }

                        if (cell != nullptr)
                        {
                            c.cells = cell->cells;
                            c.below = cell->below;
                            c.over = cell->over;
                            c.delivered = cell->delivered;
                            c.bound = cell->bound;
                            c.judgedBound = cell->judgedTo > cell->bound ? cell->judgedTo
                                                                        : cell->bound;
                            c.worstN = cell->worstN;
                            c.worstX = cell->worstX;
                            c.worstForm = cell->worstForm;
                            c.forms = cell->forms;
                            c.plainBar = cell->plainBar;

                            for (std::size_t f = 0; f < kCombForms; ++f)
                            {
                                c.formWorst[f] = cell->formWorst[f];
                                c.formWorstN[f] = cell->formWorstN[f];
                                c.formWorstX[f] = cell->formWorstX[f];
                            }
                            c.state = c.over == 0
                                          ? "certified and published"
                                          : "DEFECT: delivers outside its documented bound";
                            // A bar wider than the lane's base is wider for the terms the
                            // cell's widest bar added beside that base, and the sentence names those
                            // and no others: a reason for a number that is not the number beside it.
                            std::string barTerms;

                            if (cell->plainTerm > 0.0)
                            {
                                barTerms += "the plain reciprocal's own rounding, which that "
                                            "form adds beside the base";
                            }

                            if (cell->valueTerm > 0.0)
                            {
                                if (!barTerms.empty())
                                {
                                    barTerms += ", and ";
                                }

                                barTerms += "half of the last representable digit of the value "
                                            "the call returned, which is a term of the value "
                                            "and not of the call";
                            }

                            // A row read at more than one form on a device lane also states the
                            // figure each of them delivered: the device lanes' per-form figures are
                            // ones this gate is the only instrument for, so a row that delivers a
                            // figure per form is owed the numbers they were read from.
                            std::string formFigures;

                            if (combIsDeviceLane(lane) && cell->forms > 1)
                            {
                                const double baseWorst =
                                    std::max(cell->formWorst[0],
                                             cell->formWorst[2]);

                                formFigures = Fmt(
                                    ": %.6g at worst over the forms that share the lane's "
                                    "base, and %.6g under the plain reciprocal at n=%d, "
                                    "x=%.6g",
                                    baseWorst,
                                    cell->formWorst[1],
                                    cell->formWorstN[1],
                                    cell->formWorstX[1]);
                            }

                            const char* formClause =
                                cell->forms > 1 && formFigures.empty()
                                    ? ", and the row was read at every form"
                                    : "";

                            c.source = c.judgedBound > c.bound
                                           ? Fmt("measured here over the whole committed grid. "
                                                 "The bound shown is %.6g, not the lane's base: "
                                                 "it is that base plus %s%s%s",
                                                 c.judgedBound,
                                                 barTerms.c_str(),
                                                 formClause,
                                                 formFigures.c_str())
                                           : Fmt("measured here over the whole committed grid%s",
                                                 formFigures.c_str());
                        } else if (!guaranteed.available)
                        {
                            c.state = "refused - owed";
                            c.source = guaranteed.reason;
                        } else if (combIsDeviceLane(lane))
                        {
                            // A member of a device lane that no measured cell above covers, in
                            // two states read from the lane's own arm: a lane no arm of this build
                            // could run is counted apart with the reason this build and host are
                            // asked about, and a lane whose arm ran and holds no cell here is a hole.
                            if (combDeviceLaneArmed[static_cast<std::size_t>(lane)])
                            {
                                c.state = "OFFERED AND COVERED BY NO CELL";
                                c.source = Fmt(
                                    "BoysAccuracyGuaranteed answers %g from %s, this build's arm "
                                    "for the %s lane ran on this host, and no cell of the cross "
                                    "measured this member",
                                    guaranteed.value,
                                    guaranteed.source,
                                    combLaneRows[static_cast<std::size_t>(lane)].name);
                                c.bound = guaranteed.value;
                            }
                            else
                            {
                                c.state = "not runnable on this host";
                                c.source = combDeviceLaneReason[static_cast<std::size_t>(lane)];
                                c.bound = guaranteed.value;
                            }
                        } else
                        {
                            // The accessor says the library carries this cell
                            // and no measurement on the list covers it. That
                            // is the hole this block exists to find.
                            c.state = "OFFERED AND COVERED BY NO CELL";
                            c.source = Fmt("BoysAccuracyGuaranteed answers %g from %s and no "
                                           "cell of this block measured it",
                                           guaranteed.value,
                                           guaranteed.source);
                        }

                        if (c.state == "certified and published")
                        {
                            ++combClaimedCarried;
                        }

                        combinations.push_back(std::move(c));
                    }
                }
            }
        }
    }

    // ---- the accessor, read against the row and against the measurement ------
    // The accessor answers from the same tables these rows are judged by, so the check below is
    // not that two numbers agree today but that they are one number: the guarantee is compared for
    // exact equality, the delivered figure against what the whole call measured.
    std::size_t combAccessorDisagreeing = 0;
    std::size_t combAccessorRefused = 0;
    std::size_t combAccessorHostOnly = 0;
    std::size_t combAccessorDeliveredShort = 0;
    std::size_t combAccessorDeliveredFloor = 0;
    std::size_t combAccessorDeliveredAbsent = 0;

    for (const Combination& c : combinations)
    {
        if (c.state.rfind("refused", 0) == 0 || c.state.rfind("OFFERED", 0) == 0)
        {
            ++combAccessorRefused;

            if (c.accessorBound != 0.0)
            {
                ++combAccessorDisagreeing;
            }

            // The accessor answers no figure for a refused combination at any
            // form, the plain one included: a second figure here would be a
            // figure for a combination the library does not carry.
            if (c.accessorPlainBound != 0.0)
            {
                ++combAccessorDisagreeing;
            }

            continue;
        }

        if (c.state.rfind("not runnable", 0) == 0)
        {
            ++combAccessorHostOnly;

            if (c.accessorBound != c.bound)
            {
                ++combAccessorDisagreeing;
            }

            continue;
        }

        if (c.accessorBound != c.bound)
        {
            ++combAccessorDisagreeing;
        }

        // The same equality for the form a row publishes its own figure for: the bar the arm
        // judged that form by is the accessor's answer for it, which is what makes the row's
        // per-form figure one a consumer reads rather than one this gate composed.
        if (c.accessorPlainBound != c.plainBar)
        {
            ++combAccessorDisagreeing;
        }

        if (c.accessorDeliveredKnown)
        {
            // The accessor's figure is the worst over the fits the combination names, so it is
            // a floor on what the whole call delivers and must not sit above it: the harmless
            // direction is counted, the other fails.
            if (c.accessorDelivered > c.delivered)
            {
                ++combAccessorDeliveredShort;
                std::printf("  %-58s the accessor answers %g, above the %g the whole call "
                            "measured\n",
                            c.axes.c_str(),
                            c.accessorDelivered,
                            c.delivered);
            } else if (c.accessorDelivered < c.delivered)
            {
                ++combAccessorDeliveredFloor;
            }
        } else if (c.below < c.cells)
        {
            // A delivered figure is absent for the half lane, whose error is the
            // format's. That is stated, and it is not a silent zero.
            ++combAccessorDeliveredAbsent;
        }
    }

    // ---- the tolerance query, judged against the same rows -------------------
    // The query answers the question the two accessors above are the inputs to, so it is judged
    // against them: the figure it reports at a tolerance the row supplies is the accessor's figure,
    // its verdict is the comparison, and a refused row is a refusal with no figure at all.
    std::size_t combToleranceCarried = 0;
    std::size_t combToleranceHalfAsked = 0;
    std::size_t combToleranceHalfUnasked = 0;
    std::size_t combToleranceRefused = 0;
    std::size_t combToleranceDisagreeing = 0;

    for (const Combination& c : combinations)
    {
        if (c.state.rfind("refused", 0) == 0)
        {
            // A refused combination has no figure to compare with a request, so
            // it has no verdict either, and the reason that comes back is the
            // library's own sentence - the one the row is refused with - rather
            // than a second sentence written for this entry.
            ++combToleranceRefused;

            if (c.atBound != boys::ToleranceVerdict::kNotCarried ||
                c.atHalf != boys::ToleranceVerdict::kNotCarried || c.toleranceBound != 0.0 ||
                c.toleranceDelivered != 0.0 || c.toleranceDeliveredKnown ||
                c.toleranceReason.empty())
            {
                ++combToleranceDisagreeing;
                std::printf("  %-58s the tolerance query answered a refused combination with a "
                            "figure, a verdict or no reason\n",
                            c.axes.c_str());
            }

            continue;
        }

        ++combToleranceCarried;

        if (c.atBound != boys::ToleranceVerdict::kGuaranteedInside ||
            c.toleranceBound != c.accessorFormBound ||
            c.toleranceDeliveredKnown != c.accessorDeliveredKnown ||
            c.toleranceDelivered != c.accessorDelivered)
        {
            ++combToleranceDisagreeing;
            std::printf("  %-58s asked at %g the tolerance query answers %g/%g, not the "
                        "accessor's %g/%g\n",
                        c.axes.c_str(),
                        c.accessorFormBound,
                        c.toleranceBound,
                        c.toleranceDelivered,
                        c.accessorFormBound,
                        c.accessorDelivered);
        }

        // Asked at half the figure the row is judged by, nothing that carries
        // that figure is inside the request. The answer may be the measured
        // figure or no - both are honest - and it may not be the guarantee.
        if (c.accessorFormBound > 0.0)
        {
            ++combToleranceHalfAsked;

            if (c.atHalf == boys::ToleranceVerdict::kGuaranteedInside)
            {
                ++combToleranceDisagreeing;
                std::printf("  %-58s asked at half of %g the tolerance query answers inside on "
                            "the guarantee\n",
                            c.axes.c_str(),
                            c.accessorFormBound);
            }
        } else
        {
            // A lane whose bound is zero would be asked nothing by the line
            // above, and a row that is asked nothing is counted rather than
            // passed quietly.
            ++combToleranceHalfUnasked;
        }
    }

    // ---- the accessor beside the figure it is judged by ---------------------
    // One row per lane: `bound` is the figure the lane documents for the combination, `delivered`
    // what the combination's own rows were measured to deliver, and `row bound` and `call measured`
    // are the same two questions asked of this row by this gate. The figures are one number.
    std::printf("\n  the accessor beside the figure the row is judged by, one combination per "
                "lane.\n  `bound` is what a caller may rely on, `delivered` is what the "
                "combination's fits were\n  measured to deliver, and the last two are the same "
                "two questions asked of the row by this\n  gate:\n");
    std::printf("  %-58s %-14s %-14s %-14s %-14s %s\n",
                "combination",
                "bound",
                "delivered",
                "row bound",
                "call measured",
                "state and the source the accessor read");

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        const Combination* shown = nullptr;

        for (const Combination& c : combinations)
        {
            if (c.axes.rfind(combLaneRows[static_cast<std::size_t>(lane)].name, 0) != 0)
            {
                continue;
            }

            if (shown == nullptr ||
                (c.state.rfind("not runnable", 0) == 0 &&
                 shown->state.rfind("not runnable", 0) != 0) ||
                c.state.rfind("certified", 0) == 0)
            {
                shown = &c;
            }

            if (c.state.rfind("certified", 0) == 0)
            {
                break;
            }
        }

        if (shown == nullptr)
        {
            continue;
        }

        const std::string boundText = Fmt("%.6g", shown->accessorBound);
        const std::string deliveredText = shown->accessorDeliveredKnown
                                              ? Fmt("%.6g", shown->accessorDelivered)
                                              : std::string("no figure");
        const std::string rowBoundText =
            shown->bound > 0.0 ? Fmt("%.6g", shown->bound) : std::string("not judged");
        const std::string measuredText =
            shown->cells > 0 ? Fmt("%.6g", shown->delivered) : std::string("not measured");

        std::printf("  %-58s %-14s %-14s %-14s %-14s %s\n",
                    shown->axes.c_str(),
                    boundText.c_str(),
                    deliveredText.c_str(),
                    rowBoundText.c_str(),
                    measuredText.c_str(),
                    // A row this run measured prints its lane's own source for the figures; a row
                    // it did not prints the row's own reason - the reason that row was counted apart
                    // with, and not a sentence about the lane.
                    shown->state.rfind("certified", 0) == 0
                        ? combLaneRows[static_cast<std::size_t>(lane)].source
                        : shown->source.c_str());
    }

    // One combination the library refuses, so that what the accessors do
    // instead of answering is on the record too - and what a caller with a
    // target is told, which is the same nothing plus the reason.
    for (const Combination& c : combinations)
    {
        if (c.state.rfind("refused", 0) == 0)
        {
            std::printf("  %-58s %-14s %-14s %-14s %-14s refused, and neither accessor returns a "
                        "number\n"
                        "  %-58s and the tolerance query returns no verdict: %s\n",
                        c.axes.c_str(),
                        "no figure",
                        "no figure",
                        "not judged",
                        "not measured",
                        "",
                        c.source.c_str());

            break;
        }
    }

    // ---- what the cross found ----------------------------------------------
    std::size_t combCertified = 0;
    std::size_t combOfferedBad = 0;
    std::size_t combOwed = 0;
    std::size_t combHostLimited = 0;
    std::size_t combUncovered = 0;

    for (const Combination& c : combinations)
    {
        if (c.state.rfind("certified", 0) == 0)
        {
            ++combCertified;
        } else if (c.state.rfind("DEFECT", 0) == 0)
        {
            ++combOfferedBad;
        } else if (c.state.rfind("OFFERED", 0) == 0)
        {
            ++combUncovered;
        } else if (c.state.rfind("not runnable", 0) == 0)
        {
            ++combHostLimited;
        } else
        {
            ++combOwed;
        }
    }

    const std::size_t combTotal = combinations.size();
    const std::size_t combAccounted = combCertified + combOfferedBad + combUncovered + combOwed +
                                      combHostLimited;

    std::printf("\n  the combinations: every combination this library's own tables offer, each "
                "one either\n  measured against the committed reference at the figure its lane "
                "publishes, or refused\n  with the library's own reason, or counted as not "
                "runnable on this host. Nothing is\n  absent and nothing is uncounted. Each "
                "measured row is read at every division form\n  the axis carries, so its cells "
                "column counts (order, argument, form) points, its\n  outside column the points "
                "of those that left the bound, and its delivered figure the\n  worst any form "
                "delivered - the table after the cross says which forms moved\n");
    std::printf("  %-58s %9s %9s %-13s %-13s %s\n",
                "combination",
                "cells",
                "outside",
                "delivered",
                "bound",
                "state");
    std::printf("  %s\n", std::string(150, '-').c_str());

    for (const Combination& c : combinations)
    {
        std::printf("  %-58s %9zu %9zu %-13.6g %-13.6g %s\n",
                    c.axes.c_str(),
                    c.cells,
                    c.over,
                    c.delivered,
                    c.judgedBound > 0.0 ? c.judgedBound : c.bound,
                    c.state.c_str());
    }

    std::printf("  %s\n", std::string(150, '-').c_str());

    for (const Combination& c : combinations)
    {
        // A certified row whose bound is not its lane's base prints its reason
        // too: the figure above is the one the lane publishes for the form the
        // row was read at, and a reader meeting 2.5e-07 where the summary table
        // says 1.5e-07 is owed the sentence that says why.
        if ((c.state.rfind("certified", 0) != 0 && c.state.rfind("not runnable", 0) != 0) ||
            c.judgedBound > c.bound)
        {
            std::printf("  %-58s %s\n", c.axes.c_str(), c.source.c_str());
        }
    }

    for (const Combination& c : combinations)
    {
        if (c.over > 0)
        {
            std::printf("  %-58s %zu cell(s) outside %.6g; the worst is n=%d at x=%.6g, "
                        "delivering %.6g in the %s form\n",
                        c.axes.c_str(),
                        c.over,
                        c.bound,
                        c.worstN,
                        c.worstX,
                        c.delivered,
                        c.worstForm < 0
                            ? "unknown"
                            : boys::DivisionFormName(static_cast<boys::DivisionForm>(c.worstForm)));
        }
    }

    for (const Combination& c : combinations)
    {
        if (c.below > 0)
        {
            std::printf("  %-58s %zu of %zu cell(s) sit at or below the figure the lane claims "
                        "over,\n  %-58s where the return is the format's floor rather than the "
                        "arithmetic's, and they\n  %-58s are counted here rather than passed\n",
                        c.axes.c_str(),
                        c.below,
                        c.cells,
                        "",
                        "");
        }
    }

    std::printf("  %s\n", std::string(150, '-').c_str());
    std::printf("  COMBINATIONS: %zu of %zu member(s) of the option space are certified and "
                "published\n",
                combCertified,
                combTotal);
    std::printf("                %zu refused with the library's own reason and owed\n", combOwed);
    std::printf("                %zu call-site limit(s) name unbuilt work and are owed the "
                "same way\n",
                unbuiltLimits);
    std::printf("                %zu not runnable on this host, counted apart and not against the "
                "library\n",
                combHostLimited);
    std::printf("                %zu offered and covered by no cell of this block\n",
                combUncovered);
    std::printf("                %zu delivering outside the bound its lane publishes\n",
                combOfferedBad);
    std::printf("  the arithmetic: %zu + %zu + %zu + %zu + %zu = %zu\n",
                combCertified,
                combOwed,
                combHostLimited,
                combUncovered,
                combOfferedBad,
                combAccounted);
    std::printf("                 the space read off the tables a second way: %zu member(s) over "
                "%d lane(s),\n                 a route axis of",
                combClaimed,
                combLaneCount);

    for (int lane = 0; lane < combLaneCount; ++lane)
    {
        std::printf(" %zu", combRoutes[static_cast<std::size_t>(lane)].size());
    }

    std::printf(" route(s), %zu scheme(s), %zu partition(s),\n                 %zu axis(es)\n",
                combSchemes,
                combPartitions,
                combAxes);
    // The entry factor, which the arithmetic above has none of on its own: the cross reaches
    // every one of its cells through one entry per lane, so a combination these axes admit and
    // another entry refuses is in no book at all until it is counted here.
    std::printf("                 the entry book's own %zu cell(s), the factor this arithmetic "
                "has none of:\n                 %zu measured + %zu refused with the library's own "
                "reason and owed +\n                 %zu not applicable to the entry's shape = "
                "%zu\n",
                entryBookRows,
                entryBookMeasured,
                entryBookRefused,
                entryBookShapeLimit,
                entryBookMeasured + entryBookRefused + entryBookShapeLimit);
    std::printf("                 the entry book's space read off the same tables a second way: "
                "%zu entry(ies) x %zu route(s) x %zu scheme(s),\n                 %zu axis(es) = "
                "%zu\n",
                kEntryBodyCount,
                entryTableRoutes.size(),
                combSchemes,
                combAxes,
                entryBookClaimed);
    std::printf("                 the run's total with the entry factor: %zu + %zu = %zu "
                "cell(s)\n",
                combTotal,
                entryBookRows,
                combTotal + entryBookRows);
    std::printf("  the cross: %zu of %zu member(s) the accessor claims the library carries are "
                "certified and\n                published by the rows above\n",
                combClaimedCarried,
                combTotal);
    std::printf("  the accessor: %zu row(s) refused without a figure, %zu answered for a lane "
                "this host\n                cannot run, %zu whose guarantee differs from the "
                "figure the row is judged by,\n                and %zu whose delivered figure "
                "sits above what the whole call measured\n",
                combAccessorRefused,
                combAccessorHostOnly,
                combAccessorDisagreeing,
                combAccessorDeliveredShort);
    std::printf("                the accessor's delivered figure is the worst over the fits the "
                "combination\n                names, and the call adds its recurrences over "
                "them: on this grid it sits\n                strictly below the whole call's "
                "measurement on %zu row(s), which is that relation\n                and not a "
                "disagreement, and it has no delivered figure at all on %zu row(s) -\n"
                "                the half lane's error is the format's\n",
                combAccessorDeliveredFloor,
                combAccessorDeliveredAbsent);
    std::printf("  the tolerance query: %zu carried row(s) asked at the figure each row is "
                "judged by and\n                answered inside it, %zu of them asked at half "
                "of that figure and answered\n                outside it, %zu row(s) whose lane "
                "publishes no figure to halve, and %zu\n                refused row(s) answered "
                "with no verdict and no figure. %zu disagreement(s)\n                with the "
                "figures the two accessors answer\n",
                combToleranceCarried,
                combToleranceHalfAsked,
                combToleranceHalfUnasked,
                combToleranceRefused,
                combToleranceDisagreeing);

    // The class book: BoysAccuracyGuaranteed takes no Shape, so a figure read off it is a lane's,
    // while every class of that lane answers with an entry of its own - boys::BoysSingle<Policy>,
    // BoysAllN<Policy> and so on, per lane. This book records which class produced which cells, and
    // a class with no cell is printed as such rather than borrowing another class's figure.
    {
        // The entry each class is measured through, in the library's own spelling:
        // the name a caller writes at the call site the cross makes.
        const char* const classEntry[4][5] = {
            {"boys::BoysSingle<Policy>", "boys::BoysFixedN<Policy>", "boys::BoysAllN<Policy>",
             "boys::BoysAllNAtOrders<Policy>", "boys::BoysAllOrders<Policy>"},
            {"boys::BoysSingleF32<Policy>", "boys::BoysFixedNF32<Policy>",
             "boys::BoysAllNF32<Policy>", "boys::BoysAllNAtOrdersF32<Policy>",
             "boys::BoysAllOrdersF32<Policy>"},
            {"boys::BoysSingleF16<Policy>", "boys::BoysFixedNF16<Policy>",
             "boys::BoysAllNF16<Policy>", "boys::BoysAllNAtOrdersF16<Policy>",
             "boys::BoysAllOrdersF16<Policy>"},
            {"boys::BoysSingleBf16<Policy>", "boys::BoysFixedNBf16<Policy>",
             "boys::BoysAllNBf16<Policy>", "boys::BoysAllNAtOrdersBf16<Policy>",
             "boys::BoysAllOrdersBf16<Policy>"}};

        const char* const classShapeName[5] = {"Single", "FixedN", "AllN", "AllNAtOrders",
                                               "AllOrders"};
        const int classLaneOf[4] = {static_cast<int>(boys::Precision::kFp64),
                                    static_cast<int>(boys::Precision::kFp32),
                                    static_cast<int>(boys::Precision::kFp16),
                                    static_cast<int>(boys::Precision::kBf16)};

        std::printf("\n  the class book: every combination of every host row, measured through "
                    "the entry of\n  the class its own shape names. The accessor takes no Shape "
                    "- the figure it answers is\n  the lane's, so nothing in the accessor says "
                    "which class a figure came from, and this\n  is the book that does: the "
                    "entry named beside a class produced every cell counted\n  on that class's "
                    "row. A combination is counted here once per class that carries it, so\n  "
                    "these totals are the lane's space read out per class and are not the lane "
                    "totals\n  of the row book above\n");
        std::printf("    %-26s %9s %9s %9s %12s  %-11s  %s\n", "class", "carried", "measured",
                    "refused", "cell(s)", "worst", "the entry that produced them");

        std::size_t classMeasured = 0;
        std::size_t classRefused = 0;
        std::size_t classCarried = 0;
        std::size_t classCellCount = 0;
        std::size_t classOver = 0;
        std::size_t classRowsMeasured = 0;
        const std::size_t classRowsTotal = 4u * 5u;

        for (int family = 0; family < 4; ++family)
        {
            const std::size_t lane = static_cast<std::size_t>(classLaneOf[family]);

            for (std::size_t s = 0; s < 5; ++s)
            {
                const std::size_t space = combClassSpace[lane][s];
                const std::size_t measured = combClassCombos[lane][s];
                const std::size_t refused = space - measured;
                const CombAccum& cells = combClassCells[lane][s];

                classMeasured += measured;
                classRefused += refused;
                classCarried += space;
                classCellCount += cells.cells;
                classOver += cells.over;

                if (cells.cells > 0)
                {
                    ++classRowsMeasured;
                }

                std::printf("    %-26s %9zu %9zu %9zu %12zu  %-11.6g  %s\n",
                            Fmt("Host %s %s", combLaneRows[lane].name, classShapeName[s]).c_str(),
                            space,
                            measured,
                            refused,
                            cells.cells,
                            cells.worst,
                            classEntry[family][s]);
            }
        }

        std::printf("    %-26s %9zu %9zu %9zu %12zu\n",
                    "total",
                    classCarried,
                    classMeasured,
                    classRefused,
                    classCellCount);
        std::printf("  %zu of %zu host class(es) measured a cell through an entry of their own "
                    "shape;\n  %zu combination(s) are refused, every one of them an axis the "
                    "shape does not carry:\n  PackAxis::kOrders at Shape::kSingle and "
                    "Shape::kFixedN, whose entries refuse it where\n  they are named, so the "
                    "combination is one the library does not carry rather than one\n  this run "
                    "skipped. %zu cell(s) of these classes were judged outside the figure the\n"
                    "  lane answers, which is what a class of this book is for\n",
                    classRowsMeasured,
                    classRowsTotal,
                    classRefused,
                    classOver);

        // The device lanes' classes: three each - Shape::kSingle, kAllN and kAllOrders, the shapes
        // the device surface's entries answer for - and what this record does not carry for them is
        // stated here rather than left to a reader's inference.
        std::printf("  the device lanes: three class(es) each - Shape::kSingle, Shape::kAllN "
                    "and\n  Shape::kAllOrders - and the arms below measure Shape::kAllOrders of "
                    "each through\n  its own entry, the boys::BoysCuda::AllOrdersF32<...>,\n"
                    "  boys::BoysCuda::AllOrdersF64<...> and boys::BoysCuda::AllOrdersF16<...> "
                    "the arms\n  name beside the route- and packing-selected names. Two classes "
                    "of each device\n  lane carry combinations this run measures no cell of: "
                    "Shape::kSingle and\n  Shape::kAllN. Their entries exist - "
                    "boys::BoysCuda::SingleF32<...> and\n  boys::BoysCuda::AllNF32<...> and the "
                    "other lanes' beside them - and the arm that\n  would measure them through "
                    "them is not built in this gate: a work item on this\n  gate, not a limit of "
                    "the library. The region-B member those arms read is\nthe one the launched "
                    "all-orders rows of their lane name, and each cell is\n  judged at the figure "
                    "the accessor answers for that member: the table below\n  names it per lane "
                    "and the other member of these families is a device-\n  callable entry this "
                    "host-only gate has no kernel to run. Every device cell\n  this run does "
                    "carry is counted in the row book above and judged against the\n  figure its "
                    "own lane publishes for the member it read.\n");

#ifdef BOYS_GATE_CUDA
        // The member those device arms read, off the library's own option table: each arm launches
        // an entry of the launched all-orders family its map names, and those rows state the member
        // they run. A lane whose two members answer different figures is judged at the figure of the
        // member its entry runs - the claim that entry's own row publishes.
        std::printf("  the region-B member the device arms above read, off the library's own\n"
                    "  option table: each of them launches an entry of the launched all-orders\n"
                    "  family its map names, and those rows state the member they run\n"
                    "  (boys/boys_cuda_options.hpp, DeviceOptionInfo::regionBExp)\n");

        for (int lane = 0; lane < combLaneCount; ++lane)
        {
            if (!combIsDeviceLane(lane))
            {
                continue;
            }

            const std::size_t rows = combDeviceLaneMemberRows[static_cast<std::size_t>(lane)];
            const char* memberName = "unnamed member";

            for (const boys::RegionBExpInfo& memberRow : combMemberRows)
            {
                if (memberRow.exp == kCombDeviceRowMember)
                {
                    memberName = memberRow.name;
                }
            }

            const double judged =
                combLaneRows[static_cast<std::size_t>(lane)].bound +
                combDeviceLaneMemberAdd[static_cast<std::size_t>(lane)];

            std::printf("    %-26s %6zu  %-14s  %.6g\n",
                        combLaneRows[static_cast<std::size_t>(lane)].name,
                        rows,
                        memberName,
                        judged);
        }

        std::printf("    the figure column is the lane's base plus the term its row states under\n"
                    "    the member named, which is the member the four device maps launch and\n"
                    "    the member these rows are judged at; the half lane adds half a\n"
                    "    representable digit of each returned value beside it, as its own row\n"
                    "    states. The row count is the launched all-orders rows of the lane in\n"
                    "    BoysDeviceOptions(), and that family names a second member where this\n"
                    "    build carries one: those rows are read by the fast-member arm, which\n"
                    "    measures the member rather than folding it into these rows.\n");
#endif // BOYS_GATE_CUDA
    }

    // The member book: every cell of the cross above was read at both members BoysRegionBExps()
    // answers, each judged against the figure the row's own lane answers for that member. The two
    // answer one figure on the host lanes, so what the reader is owed is the count of cells each was
    // read at and of those the two delivered differently at the same order, argument, class and form.
    {
        std::printf("\n  the region-B member: every host cell of the cross above was read at "
                    "each of the %zu\n  member(s) BoysRegionBExps() answers, each reading judged "
                    "against the figure the\n  row's own lane answers for that member - the term "
                    "a row states beside its base is\n  that member's own, so a call naming the "
                    "other is not owed it. The device arms' cells\n  are not in this table: they "
                    "are read at the member their entry names, which is stated\n  under the "
                    "class book above. The compared column is how many host values the two\n  "
                    "members were asked for at the same order, argument, class and form, and\n  "
                    "the differed column how many of those the two delivered differently\n",
                    combMembers);

        if (combMembers != kCombMemberSlots)
        {
            std::printf("    the table answers %zu member(s) and this sweep instantiates %zu: the "
                        "members\n    past the %zu are carried by this library and measured by "
                        "no cell of this cross -\n    a gap in this gate, stated here rather than "
                        "left as a smaller number\n",
                        combMembers,
                        kCombMemberSlots,
                        kCombMemberSlots);
        }

        std::printf("    %-26s %-14s %12s %11s\n", "lane", "member", "cell(s)", "worst");

        for (int lane = 0; lane < combLaneCount; ++lane)
        {
            const CombMemberBook& memberCells = combMemberBook[static_cast<std::size_t>(lane)];

            if (memberCells.cells[0] == 0 && memberCells.cells[1] == 0)
            {
                continue;
            }

            for (std::size_t m = 0; m < combMemberRows.size() && m < kCombMemberSlots; ++m)
            {
                std::printf("    %-26s %-14s %12zu %11.6g\n",
                            combLaneRows[static_cast<std::size_t>(lane)].name,
                            combMemberRows[m].name,
                            memberCells.cells[m],
                            memberCells.worst[m]);
            }

            std::printf("    %-26s the two members were compared at %zu value(s), and delivered "
                        "%zu of\n    %-26s them differently\n",
                        combLaneRows[static_cast<std::size_t>(lane)].name,
                        memberCells.compared,
                        memberCells.differed,
                        "");
        }
    }

    // The division form, on the cells the cross measured: the values a form past the first
    // delivered differently from the first's at the same order and argument, against the values it
    // was asked for. A zero count is a lane the axis selects no arithmetic on - measured here.
    {
        struct FormRow {
            int lane = -1;
            int axis = -1;
            std::size_t cells = 0;
            std::size_t forms = 0;
            std::size_t compared = 0;
            std::array<std::size_t, kCombForms> moved = {};
        };

        std::vector<FormRow> formRows;

        for (const CombCell& m : combMeasured)
        {
            FormRow* row = nullptr;

            for (FormRow& r : formRows)
            {
                if (r.lane == m.lane && r.axis == m.axis)
                {
                    row = &r;

                    break;
                }
            }

            if (row == nullptr)
            {
                formRows.push_back({m.lane, m.axis, 0, 0, 0, {}});
                row = &formRows.back();
            }

            row->cells += m.cells;
            row->forms = std::max(row->forms, m.forms);
            row->compared += m.compared;

            for (std::size_t f = 0; f < kCombForms; ++f)
            {
                row->moved[f] += m.movedByForm[f];
            }
        }

        std::size_t formCompared = 0;
        std::array<std::size_t, kCombForms> formMoved = {};

        for (const FormRow& r : formRows)
        {
            formCompared += r.compared;

            for (std::size_t f = 0; f < kCombForms; ++f)
            {
                formMoved[f] += r.moved[f];
            }
        }

        std::printf("\n  the division form: every cell of the cross above was read at each "
                    "of the %zu\n  form(s) BoysDivisionForms() answers, each form judged "
                    "against the figure the\n  row's own lane publishes for it. The columns "
                    "after the compared count are the\n  values each form past the first "
                    "delivered differently from the first form's at the\n  same order and "
                    "argument, against the values it was asked for. A lane whose\n  column is "
                    "zero for every form is a lane the axis selects no arithmetic on: its\n  "
                    "three names reach one body, and one is what it was certified at\n",
                    combForms);
        std::printf("    %-27s %-11s %10s %9s %11s", "lane", "axis", "cell(s)", "form(s)",
                    "compared");

        // The first form is the baseline the others are compared against, so a
        // column for it would be zero by construction rather than by
        // measurement. Only the forms past it are printed.
        for (std::size_t f = 1; f < kCombForms; ++f)
        {
            std::printf(" %11s", combFormRows[f].name);
        }

        std::printf("\n");

        for (const FormRow& r : formRows)
        {
            std::printf("    %-27s %-11s %10zu %9zu %11zu",
                        combLaneRows[static_cast<std::size_t>(r.lane)].name,
                        boys::PackAxisName(static_cast<boys::PackAxis>(r.axis)),
                        r.cells,
                        r.forms,
                        r.compared);

            for (std::size_t f = 1; f < kCombForms; ++f)
            {
                std::printf(" %11zu", r.moved[f]);
            }

            std::printf("\n");
        }

        std::printf("    %-27s %-11s %10s %9s %11zu", "total", "", "", "", formCompared);

        for (std::size_t f = 1; f < kCombForms; ++f)
        {
            std::printf(" %11zu", formMoved[f]);
        }

        std::printf("\n");

        // The sweep writes its forms out as policies, while the axis is enumerated by an accessor,
        // and the two are reconciled here rather than assumed equal: a fourth form added to
        // BoysDivisionForms() without a fourth policy would leave the axis swept at three.
        if (combForms != kCombForms)
        {
            std::printf("\n  DIVISION-FORM SWEEP FAIL: the axis answers %zu form(s) from "
                        "BoysDivisionForms()\n  and this block writes %zu out as policies, so "
                        "%zu of them were measured and\n  the rest were offered by the library "
                        "and covered by no cell here\n",
                        combForms,
                        kCombForms,
                        kCombForms);
            failed = true;
        }
    }

    // combAccessorDeliveredShort is reported and not fatal: the two figures are two measurements
    // of the same quantity - the accessor's dense sweep of each piece against this gate's coarser
    // grid - so on a row whose fit dominates, the accessor's figure sits above this grid's reading
    // by construction and not by a claim that failed.
    if (combTotal != combClaimed || combTotal != combAccounted || combUncovered > 0 ||
        combOfferedBad > 0 || combAccessorDisagreeing > 0 ||
        combToleranceDisagreeing > 0 ||
        // The entry factor, on both sides of the identity: a member added to any axis of either
        // arithmetic moves one side and not the other, which is the hole the entry book exists to
        // close - a combination a caller can name, refused, with no term for it in the arithmetic.
        entryBookRows != entryBookMeasured + entryBookRefused + entryBookShapeLimit ||
        combTotal + entryBookRows != combClaimed + entryBookClaimed)
    {
        std::printf("\n  COMBINATION COVERAGE FAIL: the option space this library offers is not "
                    "the option space\n  this block accounts for. Each count above is a member "
                    "of the space that has no\n  measured row of its own, and every one of them "
                    "is work - a table, a body or a probe -\n  rather than a combination that "
                    "cannot exist\n");
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
    // Their own block, their own cell count and their own RESULT line: the books above are the
    // numbers a reader has seen before and the axis must not move them, so the fraction below is
    // the axis's own discriminating fraction and not a re-reading of the lane's.
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

    // The plane entry's rows, beside the all-orders entry's and not folded into them: the axis's
    // values are the same on the two shapes - the plane entry's per-argument path is the all-orders
    // entry's body - so keeping them apart is what lets a reader see the axis on both.
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

    // The rest of the axis: the other route and the other partition, at both schemes, on both
    // entries. A row's name is its partition ("narrow" where the region-A fits are the per-order
    // pieces, absent where they are the shipped table), its route (cheb, rat) and which entry and
    // interval it covers - "A" the packed lane's interval, "A..C" the whole grid, "pl" the plane.
    std::printf("  %s\n", std::string(160, '-').c_str());

    for (const OpenedRow& row : openedRows)
    {
        packRow(row.axis, row.row.c_str(), row.bound, PackClaims()[row.slot]);
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

    // ---- the granularity-axis rows, counted apart ---------------------------
    // Their own block, cell count and RESULT line, for the reason the packing axis's block gives:
    // two rows read both members' stored fits directly and six read the entries end to end over the
    // region the axis is cut in, and the trade is printed with the rows and not in place of them.
    std::size_t granCellCount = 0;
    std::size_t granNonDiscriminating = 0;

    for (const Accum& a : GranularityClaims())
    {
        granCellCount += a.points;
        granNonDiscriminating += a.vacuous;
    }

    std::printf("\nthe granularity-axis rows: which partition of the fitted domain a call "
                "reads, measured at both members against the committed reference\n");
    std::printf("  %-10s %-15s %-28s %9s %-22s %-22s %7s  %-22s %s\n",
                "partition",
                "scheme",
                "row",
                "cells",
                "bound it promises",
                "worst delivered",
                "ratio",
                "worst cell",
                "verdict");
    std::printf("  %s\n", std::string(157, '-').c_str());

    int granMet = 0;
    std::size_t granRowsRun = 0;
    std::vector<std::string> granNotMet;

    const auto granRow =
        [&](const char* member, const char* scheme, const char* row, const char* bound,
            const Accum& a) {
            ++granRowsRun;
            const Verdict v = FromAccum(a);

            if (IsMet(v))
            {
                ++granMet;
            } else
            {
                granNotMet.push_back(std::string(member) + " / " + scheme + " / " + row);
            }

            char where[64];
            std::snprintf(where, sizeof(where), "n=%d, x=%.6g", a.worstN, a.worstX);
            std::printf("  %-10s %-15s %-28s %9zu %-22s %-22.6g %7.3g  %-22s %s\n",
                        member,
                        scheme,
                        row,
                        a.points,
                        bound,
                        a.worstErr,
                        a.worstRatio,
                        where,
                        VerdictName(v));
        };

    for (std::size_t g = 0; g < kGranMembers; ++g)
    {
        const char* member = boys::GranularityName(static_cast<boys::FitGranularity>(g));

        for (std::size_t s = 0; s < granSchemeCount; ++s)
        {
            const char* scheme = boys::EvalSchemeName(granSchemes[s]);

            for (std::size_t r = 0; r < kGranRowCount; ++r)
            {
                // The bar a row is judged at, printed as the row was judged:
                // one published number, or the per-region formula the
                // single-order lane's whole-domain rows are judged with.
                char bar[40];

                if (granRows[r].bar == GranBar::kFixed)
                {
                    std::snprintf(bar, sizeof(bar), "%.6g", granRows[r].bound);
                } else
                {
                    std::snprintf(bar, sizeof(bar), "per region (B_region)");
                }

                const Accum& a = GranularityClaims()[static_cast<std::size_t>(
                    granSlots[granIndex(g, s, r)])];

                granRow(member, scheme, granRows[r].row, bar, a);
            }
        }
    }

    // The rational route's rows over the narrow partition are in this book too: the rational family
    // takes no scheme, and what they measure is a partition rather than a member of the axis, so they
    // are judged and counted with the rows above rather than beside them.
    {
        struct NarrowRatRow {
            const char* row;
            double bound;
        };
        const NarrowRatRow rows[3] = {
            {"region A pieces", boys::detail::kRegionAFitBar},
            {"region B seed", boys::detail::kRegionBFitBar},
            {"batch entry, A..C", kBoundDoubleBatch}};

        for (int j = 0; j < 3; ++j)
        {
            char bar[32];
            std::snprintf(bar, sizeof(bar), "%.6g", rows[j].bound);
            granRow("rational x narrow",
                    "-",
                    rows[j].row,
                    bar,
                    GranularityClaims()[static_cast<std::size_t>(narrowRatClaims[j])]);
        }
    }

    std::printf("  %s\n", std::string(157, '-').c_str());
    std::printf("  GRANULARITY RESULT: %d of %zu granularity-axis rows met at this revision\n",
                granMet,
                granRowsRun);

    // The partition's own size, read off the tables the two members are built
    // from: rows to look the piece up in, coefficients stored, and the span of
    // coefficients one evaluation reads.
    {
        int shippedLow = 0;
        int shippedHigh = 0;

        for (const boys::detail::OrderPiece& piece : boys::detail::kPieces)
        {
            shippedLow = shippedLow == 0 || piece.deg < shippedLow ? piece.deg : shippedLow;
            shippedHigh = piece.deg > shippedHigh ? piece.deg : shippedHigh;
        }

        const std::size_t shippedRows = std::size(boys::detail::kPieces);
        const std::size_t shippedStored = std::size(boys::detail::kCoeffs)
                                          + std::size(boys::detail::kBcoeffs);
        const std::size_t narrowRows = std::size(boys::detail::kNarrowAPieces)
                                       + static_cast<std::size_t>(boys::detail::kNarrowBPieces);
        const std::size_t narrowStored = std::size(boys::detail::kNarrowACoeffs)
                                         + std::size(boys::detail::kNarrowBcoeffs);

        std::printf("  the trade the axis names: the shipped partition stores %zu coefficient(s) "
                    "in %zu row(s) over both fitted regions and one evaluation reads %d to %d "
                    "of them;\n"
                    "                            the narrow partition stores %zu in %zu rows and "
                    "one evaluation reads %d. Naming it buys the smaller read at the larger "
                    "table and the piece lookup, and is not a saving\n",
                    shippedStored,
                    shippedRows + 1,
                    shippedLow + 1,
                    shippedHigh + 1,
                    narrowStored,
                    narrowRows,
                    boys::detail::kNarrowADeg + 1);

        // The same partition on the route whose pieces are degree pairs: the rows and the intervals
        // are the partition's, so the table differs in what each row stores - the read is the piece's
        // own numerator p_0..p_m and denominator coefficients.
        int ratReadLow = 0;
        int ratReadHigh = 0;

        for (std::size_t i = 0; i < std::size(boys::detail::kNarrowRatAOffset); ++i)
        {
            const int read = boys::detail::kNarrowRatANumDeg[i]
                             + boys::detail::kNarrowRatADenDeg[i] + 1;
            ratReadLow = ratReadLow == 0 || read < ratReadLow ? read : ratReadLow;
            ratReadHigh = read > ratReadHigh ? read : ratReadHigh;
        }

        std::printf("                            the same partition on the rational route "
                    "stores %zu coefficient(s) in the same %zu row(s), one evaluation reading "
                    "%d to %d of them\n",
                    std::size(boys::detail::kNarrowRatACoeffs)
                        + std::size(boys::detail::kNarrowRatBCoeffs),
                    narrowRows,
                    ratReadLow,
                    ratReadHigh);
    }

    if (granCellCount > 0)
    {
        std::printf("                carried by the %zu of %zu axis cells (%.1f%%) that can "
                    "discriminate: the other %zu carry a bound at least as large as the value "
                    "itself. These cells are not in the counts above and do not move them\n",
                    granCellCount - granNonDiscriminating,
                    granCellCount,
                    100.0 * static_cast<double>(granCellCount - granNonDiscriminating)
                        / static_cast<double>(granCellCount),
                    granNonDiscriminating);
    }

    if (!granNotMet.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : granNotMet)
        {
            std::printf(" %s", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; the granularity-axis rows are judged with the "
                    "books above)\n");
        return 1;
    }

    // ---- the entry book's rows, counted apart -------------------------------
    // This counts the space a caller names, and it is the only place a combination the axes admit
    // and an entry refuses is printed at all. Its own rows and RESULT line; the three states are
    // printed rather than summarised, since a refused row carries the library's own assertion.
    std::printf("\nthe entry book: the batched double-precision entries crossed with the uniform\n"
                "partition, per route, per scheme and per packing axis.\n"
                "Every combination is measured, refused by the library's own guard, or not\n"
                "applicable to the entry's shape; a refused row prints the assertion that\n"
                "refuses it, so a combination no row measures is a named refusal and not an\n"
                "absence\n");
    std::printf("  %-22s %-20s %-16s %-11s %8s %-12s %-12s %7s  %-20s %s\n",
                "entry",
                "route",
                "scheme",
                "axis",
                "cells",
                "bound",
                "delivered",
                "ratio",
                "worst cell",
                "state");
    std::printf("  %s\n", std::string(170, '-').c_str());

    int entryMet = 0;
    std::size_t entryJudged = 0;
    std::vector<std::string> entryNotMet;

    for (const EntryBookRow& row : entryBook)
    {
        if (row.state == EntryState::kMeasured)
        {
            const Accum& a = EntryClaims()[static_cast<std::size_t>(row.slot)];
            const Verdict v = FromAccum(a);
            ++entryJudged;

            if (IsMet(v))
            {
                ++entryMet;
            } else
            {
                entryNotMet.push_back(row.axes);
            }

            char where[64];
            std::snprintf(where, sizeof(where), "n=%d, x=%.6g", a.worstN, a.worstX);
            std::printf("  %-22s %-20s %-16s %-11s %8zu %-12.6g %-12.6g %7.3g  %-20s %s\n",
                        row.entry,
                        row.route.c_str(),
                        row.scheme.c_str(),
                        row.axis.c_str(),
                        a.points,
                        row.bound,
                        a.worstErr,
                        a.worstRatio,
                        where,
                        VerdictName(v));
        } else
        {
            std::printf("  %-22s %-20s %-16s %-11s %8s %-12s %-12s %7s  %-20s %s\n",
                        row.entry,
                        row.route.c_str(),
                        row.scheme.c_str(),
                        row.axis.c_str(),
                        "n/a",
                        "n/a",
                        "n/a",
                        "n/a",
                        "",
                        row.state == EntryState::kRefused
                            ? "refused, and owed"
                            : "not applicable to the entry's shape");
            std::printf("    %s\n", row.axes.c_str());
            std::printf("    %s\n", row.reason);
        }
    }

    // The two states that are not measurements are printed with their counts again, in the book's
    // own arithmetic, so that a reader who takes only the numbers still sees them. The arms are
    // reconciled against the tables that report the three axes, both ways.
    std::printf("  %s\n", std::string(170, '-').c_str());
    std::printf("  ENTRY RESULT: %d of %zu measured entry row(s) met at this revision (%zu "
                "refused and owed,\n                %zu not applicable to the entry's shape; the "
                "three states add to the %zu\n                combination(s) this book crosses)\n",
                entryMet,
                entryJudged,
                entryBookRefused,
                entryBookShapeLimit,
                entryBookRows);

    if (entryArmsUnnamed == 0 && entryTableUncovered == 0)
    {
        std::printf("                the axes are the tables': this book's %zu arm(s) are named "
                    "by the %zu\n                route(s), %zu scheme(s) and %zu axis(es) the "
                    "lane reports, and every one\n                of those members is an arm\n",
                    entryArms.size(),
                    entryTableRoutes.size(),
                    boys::BoysEvalSchemes().size(),
                    boys::BoysPackAxes().size());
    } else
    {
        std::printf("\n  ENTRY BOOK FAIL: the axes this book sweeps are not the library's. %zu "
                    "arm(s) of the\n  %zu it writes are named by no table, and %zu member(s) the "
                    "tables report are no arm of\n  its:%s\n  A member added to any of those three "
                    "axes is a row this book has to be given before the\n  space it sweeps is "
                    "that table's; the arithmetic above counts the rows that exist and\n  not the "
                    "members that are there\n",
                    entryArmsUnnamed,
                    entryArms.size(),
                    entryTableUncovered,
                    entryTableUncoveredNames.c_str());
        return 1;
    }

    if (!entryNotMet.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : entryNotMet)
        {
            std::printf(" [%s]", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; the entry rows are judged with the books above)\n");
        return 1;
    }

    // The uniform grid's carriage on the batched entries: the entry book's rows are accuracy
    // readings, and the two partitions hold the same bound over the same interval - so an entry
    // answering a grid policy out of the narrow tables would print the same numbers and stay green.
    // Each entry is read twice in one pass, once under each partition, differing cells counted.
    std::array<std::size_t, 4> partitionRefDiffer{};

    for (std::size_t i = 0; i < count; ++i)
    {
        for (int n = 0; n <= nmax; ++n)
        {
            const double got =
                boys::BoysSingle<
                                 GranularityPolicy<boys::EvalScheme::kSplitClenshaw,
                                                   boys::FitGranularity::kUniform>>(n, ref.x[i]);
            const double other =
                boys::BoysSingle<
                                 GranularityPolicy<boys::EvalScheme::kSplitClenshaw,
                                                   boys::FitGranularity::kNarrow>>(n, ref.x[i]);

            if (std::memcmp(&got, &other, sizeof(double)) != 0)
            {
                ++partitionRefDiffer[static_cast<std::size_t>(SingleClaim(ref.x[i]))];
            }
        }
    }

    // What one entry answered with, per region, under each of the two policies.
    struct PartitionCarriageRow {
        const char* entry = "";
        std::array<std::size_t, 4> cells{};
        std::array<std::size_t, 4> differ{};
        double worstAbs = 0.0;
        int worstN = -1;
        double worstX = 0.0;
    };

    std::vector<PartitionCarriageRow> partitionRows;

    const auto sweepPartitionCarriage = [&](BatchEntry kind, const char* entryName) {
        using GridReading =
            GranularityPolicy<boys::EvalScheme::kSplitClenshaw, boys::FitGranularity::kUniform>;
        using NarrowReading =
            GranularityPolicy<boys::EvalScheme::kSplitClenshaw, boys::FitGranularity::kNarrow>;

        PartitionCarriageRow row;
        row.entry = entryName;

        const auto record = [&row](int n, double x, double got, double other) {
            const std::size_t region = static_cast<std::size_t>(SingleClaim(x));
            ++row.cells[region];

            if (std::memcmp(&got, &other, sizeof(double)) == 0)
            {
                return;
            }

            ++row.differ[region];

            const double d = std::fabs(got - other);

            if (d > row.worstAbs)
            {
                row.worstAbs = d;
                row.worstN = n;
                row.worstX = x;
            }
        };

        const std::size_t grid = count * (static_cast<std::size_t>(nmax) + 1);

        if (kind == BatchEntry::kPlane || kind == BatchEntry::kPlaneSorted)
        {
            const bool sorted = kind == BatchEntry::kPlaneSorted;
            const std::vector<double>& args = sorted ? refSorted : ref.x;
            std::vector<double> u(grid);
            std::vector<double> v(grid);

            if (sorted)
            {
                boys::BoysAllN< GridReading>(nmax, args.data(), u.data(), count,
                                                 boys::BoysSortedArgs{});
                boys::BoysAllN< NarrowReading>(nmax, args.data(), v.data(), count,
                                                   boys::BoysSortedArgs{});
            } else
            {
                boys::BoysAllN< GridReading>(nmax, args.data(), u.data(), count);
                boys::BoysAllN< NarrowReading>(nmax, args.data(), v.data(), count);
            }

            for (int n = 0; n <= nmax; ++n)
            {
                for (std::size_t j = 0; j < count; ++j)
                {
                    // The sorted call lays its planes out in its own argument
                    // order, so a position maps back through the permutation.
                    const std::size_t i = sorted ? sortedPerm[j] : j;
                    const std::size_t k = static_cast<std::size_t>(n) * count + j;

                    record(n, ref.x[i], u[k], v[k]);
                }
            }
        } else if (kind == BatchEntry::kAtOrders)
        {
            std::vector<int> tops(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                tops[i] = nmax - static_cast<int>(i % static_cast<std::size_t>(nmax + 1));
            }

            std::vector<double> u(grid);
            std::vector<double> v(grid);
            boys::BoysAllNAtOrders< GridReading>(tops.data(), ref.x.data(), u.data(), count);
            boys::BoysAllNAtOrders< NarrowReading>(tops.data(), ref.x.data(), v.data(), count);

            for (std::size_t i = 0; i < count; ++i)
            {
                for (int n = 0; n <= tops[i]; ++n)
                {
                    const std::size_t k = static_cast<std::size_t>(n) * count + i;

                    record(n, ref.x[i], u[k], v[k]);
                }
            }
        } else
        {
            std::vector<double> u(count);
            std::vector<double> v(count);

            for (int n = 0; n <= nmax; ++n)
            {
                boys::BoysFixedN< GridReading>(n, ref.x.data(), u.data(), count);
                boys::BoysFixedN< NarrowReading>(n, ref.x.data(), v.data(), count);

                for (std::size_t i = 0; i < count; ++i)
                {
                    record(n, ref.x[i], u[i], v[i]);
                }
            }
        }

        partitionRows.push_back(std::move(row));
    };

    sweepPartitionCarriage(BatchEntry::kPlane, "plane entry");
    sweepPartitionCarriage(BatchEntry::kPlaneSorted, "plane entry, sorted");
    sweepPartitionCarriage(BatchEntry::kAtOrders, "per-element tops");
    sweepPartitionCarriage(BatchEntry::kFixedN, "fixed-order entry");

    std::printf("\nthe uniform grid's carriage on the batched entries: each entry is read twice "
                "in one\npass, once under a policy naming the grid and once under a policy naming "
                "the narrow\nmember, and the cells where the two readings differ are counted per "
                "region. The two\npartitions hold the same bound over the same interval, so an "
                "accuracy row cannot tell\nthem apart; a difference can, and a region where the "
                "per-argument entry's two readings\ndiffer and this entry's do not is a region "
                "this entry answered from the other\npartition's fits.\n");
    std::printf("  the per-argument entry's own separation, the reference these rows are held "
                "to: A %zu, band %zu, B %zu, C %zu cell(s)\n",
                partitionRefDiffer[0],
                partitionRefDiffer[1],
                partitionRefDiffer[2],
                partitionRefDiffer[3]);
    std::printf("  %-22s %9s %9s  %-14s %-12s %-18s %s\n",
                "entry",
                "cells",
                "differ",
                "A/band/B/C",
                "worst |d|",
                "worst cell",
                "verdict");
    std::printf("  %s\n", std::string(132, '-').c_str());

    std::size_t partitionRowsMet = 0;
    std::vector<std::string> partitionNotMet;

    for (const PartitionCarriageRow& row : partitionRows)
    {
        std::size_t cells = 0;
        std::size_t differ = 0;
        std::size_t missed = 0;
        char tokens[24];
        std::size_t at = 0;

        for (std::size_t r = 0; r < 4; ++r)
        {
            const char* tok = row.cells[r] == 0 ? "-" : (row.differ[r] > 0 ? "yes" : "NO");
            at += static_cast<std::size_t>(std::snprintf(
                tokens + at, sizeof(tokens) - at, "%s%s", r == 0 ? "" : "/", tok));

            if (partitionRefDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0)
            {
                ++missed;
            }

            cells += row.cells[r];
            differ += row.differ[r];
        }

        char where[64];
        std::snprintf(where, sizeof(where), "n=%d, x=%.6g", row.worstN, row.worstX);

        if (missed == 0)
        {
            ++partitionRowsMet;
            std::printf("  %-22s %9zu %9zu  %-14s %-12.6g %-18s %s\n",
                        row.entry,
                        cells,
                        differ,
                        tokens,
                        row.worstAbs,
                        where,
                        "answers with the grid's own values");
            continue;
        }

        partitionNotMet.push_back(std::string("carriage of the grid by the ") + row.entry);
        char which[24];
        std::size_t wat = 0;

        // Every region is named here, region C included: unlike the scheme
        // carriage table's rows, a row of this one is held to C as well, because
        // the two partitions do separate over the cells of C that read a fit.
        for (std::size_t r = 0; r < 4; ++r)
        {
            if (partitionRefDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0)
            {
                wat += static_cast<std::size_t>(std::snprintf(which + wat,
                                                              sizeof(which) - wat,
                                                              "%s%s",
                                                              wat == 0 ? "" : " and ",
                                                              kRegionTag[r]));
            }
        }

        std::printf("  %-22s %9zu %9zu  %-14s %-12.6g %-18s NOT CARRIED - region %s separates "
                    "on the\n      per-argument entry and nowhere on this one, so this entry "
                    "answered it\n      from the narrow member's fits\n",
                    row.entry,
                    cells,
                    differ,
                    tokens,
                    row.worstAbs,
                    where,
                    which);
    }

    std::printf("  %s\n", std::string(132, '-').c_str());
    std::printf("  PARTITION RESULT: %zu of %zu batched entr(ies) answer a policy naming the "
                "uniform grid\n                    with a reading the narrow member does not "
                "answer with, in every region where\nthe two readings can differ\n",
                partitionRowsMet,
                partitionRows.size());

    if (!partitionNotMet.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : partitionNotMet)
        {
            std::printf(" [%s]", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; the uniform grid's carriage on the batched entries "
                    "is judged\n  with the entry book above, and a row that fails it is an entry "
                    "answering a\n  uniform policy from the narrow member's fits - the "
                    "substitution the partition\n  axis exists to prevent)\n");
        return 1;
    }

    // The uniform grid's carriage on the single-precision entries: the block above asks this of
    // the double lane; the float lane's cells on the same partition are measured in the combination
    // book by rows that ask it of nothing. Each entry is read twice in one pass, once under each
    // partition, and the differing cells counted per region against the per-argument entry's own.
    struct PartitionCarriageRowF32 {
        std::string entry;
        std::array<std::size_t, 4> refDiffer{};
        std::array<std::size_t, 4> cells{};
        std::array<std::size_t, 4> differ{};
        double worstAbs = 0.0;
        int worstN = -1;
        float worstX = 0.0f;
    };

    struct PartitionCarriageRefF32 {
        std::string axes;
        std::array<std::size_t, 4> differ{};
    };

    std::vector<PartitionCarriageRowF32> partitionRowsF32;
    std::vector<PartitionCarriageRefF32> partitionRefsF32;

    const auto recordPartitionF32 =
        [](PartitionCarriageRowF32& row, int n, float x, float got, float other) {
            const std::size_t region =
                static_cast<std::size_t>(SingleClaim(static_cast<double>(x)));
            ++row.cells[region];

            if (std::memcmp(&got, &other, sizeof(float)) == 0)
            {
                return;
            }

            ++row.differ[region];

            const double d = std::fabs(static_cast<double>(got) - static_cast<double>(other));

            if (d > row.worstAbs)
            {
                row.worstAbs = d;
                row.worstN = n;
                row.worstX = x;
            }
        };

    const auto laneOfBudget = [](boys::BoysBudget budget) {
        return budget == boys::BoysBudget::kFp16 ? combHalfLane : kLaneSingle;
    };

    const auto sweepPartitionCarriageF32 =
        [&]<boys::BoysBudget kBudget, boys::FitRoute kRoute, boys::EvalScheme kScheme,
            boys::PackAxis kAxis>(const std::array<std::size_t, 4>& refDiffer,
                                  bool plane,
                                  const char* entryName) {
            using GridReading = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis,
                                                 boys::FitGranularity::kUniform>;
            using NarrowReading = boys::EvalPolicy<kRoute, kScheme, kBudget, kAxis,
                                                   boys::FitGranularity::kNarrow>;

            PartitionCarriageRowF32 row;
            row.entry = Fmt("%s, %s, %s, %s, %s",
                            entryName,
                            combLaneRows[static_cast<std::size_t>(laneOfBudget(kBudget))].name,
                            entryArmRouteName(kRoute).c_str(),
                            boys::EvalSchemeName(kScheme),
                            entryArmAxisName(kAxis).c_str());
            row.refDiffer = refDiffer;

            std::vector<float> argsF(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                argsF[i] = static_cast<float>(ref.xf[i]);
            }

            const std::size_t grid = count * (static_cast<std::size_t>(nmax) + 1);

            if (plane)
            {
                std::vector<float> u(grid);
                std::vector<float> v(grid);
                boys::BoysAllNF32< GridReading>(nmax, argsF.data(), u.data(), count);
                boys::BoysAllNF32< NarrowReading>(nmax, argsF.data(), v.data(), count);

                for (int n = 0; n <= nmax; ++n)
                {
                    for (std::size_t i = 0; i < count; ++i)
                    {
                        const std::size_t k = static_cast<std::size_t>(n) * count + i;

                        recordPartitionF32(row, n, argsF[i], u[k], v[k]);
                    }
                }
            } else
            {
                std::array<float, 33> u{};
                std::array<float, 33> v{};

                for (std::size_t i = 0; i < count; ++i)
                {
                    boys::BoysAllOrdersF32< GridReading>(nmax, argsF[i], u.data());
                    boys::BoysAllOrdersF32< NarrowReading>(nmax, argsF[i], v.data());

                    for (int n = 0; n <= nmax; ++n)
                    {
                        const std::size_t sn = static_cast<std::size_t>(n);

                        recordPartitionF32(row, n, argsF[i], u[sn], v[sn]);
                    }
                }
            }

            partitionRowsF32.push_back(std::move(row));
        };

    // One row set per (budget, route, scheme): the reference is that arm's own,
    // and the two entries are swept on both axes under it.
    const auto sweepPartitionF32Arm =
        [&]<boys::BoysBudget kBudget, boys::FitRoute kRoute, boys::EvalScheme kScheme>() {
            using GridReading = boys::EvalPolicy<kRoute, kScheme, kBudget,
                                                 boys::PackAxis::kArguments,
                                                 boys::FitGranularity::kUniform>;
            using NarrowReading = boys::EvalPolicy<kRoute, kScheme, kBudget,
                                                   boys::PackAxis::kArguments,
                                                   boys::FitGranularity::kNarrow>;

            PartitionCarriageRefF32 reference;
            reference.axes = Fmt("%s, %s, %s",
                                 combLaneRows[static_cast<std::size_t>(laneOfBudget(kBudget))].name,
                                 entryArmRouteName(kRoute).c_str(),
                                 boys::EvalSchemeName(kScheme));

            for (std::size_t i = 0; i < count; ++i)
            {
                const float xf = static_cast<float>(ref.xf[i]);
                const std::size_t region =
                    static_cast<std::size_t>(SingleClaim(static_cast<double>(xf)));

                for (int n = 0; n <= nmax; ++n)
                {
                    const float got = boys::BoysSingleF32< GridReading>(n, xf);
                    const float other = boys::BoysSingleF32< NarrowReading>(n, xf);

                    if (std::memcmp(&got, &other, sizeof(float)) != 0)
                    {
                        ++reference.differ[region];
                    }
                }
            }

            partitionRefsF32.push_back(reference);

            sweepPartitionCarriageF32.template operator()<kBudget, kRoute, kScheme,
                                                          boys::PackAxis::kArguments>(
                reference.differ, false, "all-orders entry");
            sweepPartitionCarriageF32.template operator()<kBudget, kRoute, kScheme,
                                                          boys::PackAxis::kOrders>(
                reference.differ, false, "all-orders entry");
            sweepPartitionCarriageF32.template operator()<kBudget, kRoute, kScheme,
                                                          boys::PackAxis::kArguments>(
                reference.differ, true, "plane entry");
            sweepPartitionCarriageF32.template operator()<kBudget, kRoute, kScheme,
                                                          boys::PackAxis::kOrders>(
                reference.differ, true, "plane entry");
        };

    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFloat, boys::FitRoute::kChebyshev,
                                             boys::EvalScheme::kSplitClenshaw>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFloat, boys::FitRoute::kChebyshev,
                                             boys::EvalScheme::kHorner>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFloat,
                                             boys::FitRoute::kRationalMinimax,
                                             boys::EvalScheme::kSplitClenshaw>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFloat,
                                             boys::FitRoute::kRationalMinimax,
                                             boys::EvalScheme::kHorner>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFp16, boys::FitRoute::kChebyshev,
                                             boys::EvalScheme::kSplitClenshaw>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFp16, boys::FitRoute::kChebyshev,
                                             boys::EvalScheme::kHorner>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFp16,
                                             boys::FitRoute::kRationalMinimax,
                                             boys::EvalScheme::kSplitClenshaw>();
    sweepPartitionF32Arm.template operator()<boys::BoysBudget::kFp16,
                                             boys::FitRoute::kRationalMinimax,
                                             boys::EvalScheme::kHorner>();

    std::printf("\nthe uniform grid's carriage on the single-precision entries: each of the "
                "two entries the\nrows carry is read twice in one pass, once under a "
                "policy naming the\ngrid and once under a policy naming the narrow member, at "
                "each budget, route and\nscheme the combination book above holds a row for, and "
                "the cells where the two\nreadings differ are counted per region. A region where "
                "the per-argument entry's\ntwo readings differ and this entry's do not is a "
                "region this entry answered from the\nnarrow member's fits.\n");
    std::printf("  the per-argument entry's own separation, the reference each arm's rows are "
                "held to:\n");

    for (const PartitionCarriageRefF32& armRef : partitionRefsF32)
    {
        std::printf("    %-40s A %zu, band %zu, B %zu, C %zu cell(s)\n",
                    armRef.axes.c_str(),
                    armRef.differ[0],
                    armRef.differ[1],
                    armRef.differ[2],
                    armRef.differ[3]);
    }

    std::printf("  %-58s %9s %9s  %-14s %-12s %-18s %s\n",
                "entry",
                "cells",
                "differ",
                "A/band/B/C",
                "worst |d|",
                "worst cell",
                "verdict");
    std::printf("  %s\n", std::string(150, '-').c_str());

    std::size_t partitionRowsMetF32 = 0;
    std::vector<std::string> partitionNotMetF32;

    for (const PartitionCarriageRowF32& row : partitionRowsF32)
    {
        std::size_t cells = 0;
        std::size_t differ = 0;
        std::size_t missed = 0;
        char tokens[24];
        std::size_t at = 0;

        for (std::size_t r = 0; r < 4; ++r)
        {
            const char* tok = row.cells[r] == 0 ? "-" : (row.differ[r] > 0 ? "yes" : "NO");
            at += static_cast<std::size_t>(std::snprintf(
                tokens + at, sizeof(tokens) - at, "%s%s", r == 0 ? "" : "/", tok));

            if (row.refDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0)
            {
                ++missed;
            }

            cells += row.cells[r];
            differ += row.differ[r];
        }

        char where[64];
        std::snprintf(where, sizeof(where), "n=%d, x=%.6g", row.worstN,
                      static_cast<double>(row.worstX));

        if (missed == 0)
        {
            ++partitionRowsMetF32;
            std::printf("  %-58s %9zu %9zu  %-14s %-12.6g %-18s %s\n",
                        row.entry.c_str(),
                        cells,
                        differ,
                        tokens,
                        row.worstAbs,
                        where,
                        "answers with the grid's own values");
            continue;
        }

        partitionNotMetF32.push_back(row.entry);
        char which[24];
        std::size_t wat = 0;

        for (std::size_t r = 0; r < 4; ++r)
        {
            if (row.refDiffer[r] > 0 && row.cells[r] > 0 && row.differ[r] == 0)
            {
                wat += static_cast<std::size_t>(std::snprintf(which + wat,
                                                              sizeof(which) - wat,
                                                              "%s%s",
                                                              wat == 0 ? "" : " and ",
                                                              kRegionTag[r]));
            }
        }

        std::printf("  %-58s %9zu %9zu  %-14s %-12.6g %-18s NOT CARRIED - region %s separates "
                    "on the\n      per-argument entry and nowhere on this one, so this entry "
                    "answered it\n      from the narrow member's fits\n",
                    row.entry.c_str(),
                    cells,
                    differ,
                    tokens,
                    row.worstAbs,
                    where,
                    which);
    }

    std::printf("  %s\n", std::string(150, '-').c_str());
    std::printf("  PARTITION RESULT: %zu of %zu single-precision entr(ies) answer a policy "
                "naming the\n                    uniform grid with a reading the narrow member "
                "does not answer with, in every\n                    region where the two "
                "readings can differ\n",
                partitionRowsMetF32,
                partitionRowsF32.size());

    if (!partitionNotMetF32.empty())
    {
        std::printf("  NOT MET at this revision:");

        for (const std::string& id : partitionNotMetF32)
        {
            std::printf(" [%s]", id.c_str());
        }

        std::printf("\n  FAIL (exit status 1; the uniform grid's carriage on the "
                    "single-precision entries\n  is judged with the combination book above, and "
                    "a row that fails it is an entry\n  answering a uniform policy from the "
                    "narrow member's fits - the substitution the\n  partition axis exists to "
                    "prevent)\n");
        return 1;
    }

    // The last line, and the only one a caller that reads nothing else sees. It
    // says what was checked and what was not: a build that does not carry some
    // of the book has claims it never judged, and a PASS that read as though it
    // had would be the one sentence here that is false.
    if (notCarried > 0)
    {
        std::printf("  PASS: every documented claim this build carries is met at this revision; "
                    "the %d it does not carry are named above with their reason, and are "
                    "neither met nor failed here\n",
                    notCarried);
        return 0;
    }

    std::printf("  PASS: every documented claim met at this revision\n");
    return 0;
}
