#include "boys/boys.hpp"

#include "boys/boys_impl.hpp"

#include "boys_backend_registry.hpp"

// Boys function kernel. Region structure (fixed kmax=32 boundaries, the
// configuration validated end-to-end against the mpmath reference grid):
//   A: [0, x0)   per-order Chebyshev fits, split Clenshaw (division-free, FMA)
//   B: [x0, x1)  F0 fit + upward recursion
//   C: [x1, inf) asymptotic 1/2 sqrt(pi/x) + upward recursion
//
// The even/odd Clenshaw split halves the dependency-chain depth:
//   T_{2j}(t) = T_j(v), T_{2j+1}(t) = t * D_j(v)  with v = 2t^2 - 1,
//   D_0 = 1, D_1 = 2v - 1 (same three-term recurrence as T_j).
// std::fma is used explicitly - MSVC does not contract without /fp:fast.
//
// The kernel bodies live in boys/boys_impl.hpp — the accuracy-multiplier
// template definitions: every entry is compiled twice under
// if constexpr, the m = 1 branch being today's certified body verbatim
// (the bit-identity pin). This TU provides the default m = 1 explicit
// instantiations that the extern-template declarations in boys.hpp route
// every default call site to — no implicit instantiation, no code
// duplication across TUs, zero cost on the m = 1 path. A call site that
// names any other multiplier compiles its rung from the shipped definition
// instead of linking here.

namespace boys {

template double BoysSingle<kBoysFullAccuracyMultiplier>(int n, double x) noexcept;
template void BoysAllOrders<kBoysFullAccuracyMultiplier>(int nmax, double x, double* out) noexcept;
template void BoysFixedN<kBoysFullAccuracyMultiplier>(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept;
template void BoysAllN<kBoysFullAccuracyMultiplier>(
    int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept;
template void BoysAllN<kBoysFullAccuracyMultiplier>(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;
template void BoysAllNAtOrders<kBoysFullAccuracyMultiplier>(
    const int* n, const double* x, double* out, std::size_t count) noexcept;
template float BoysSingleF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(int n, float x) noexcept;
template void BoysAllOrdersF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(
    int nmax, float x, float* out) noexcept;
template void BoysAllNF32<kBoysFullAccuracyMultiplier, EvalPolicy<>>(
    int nmax, const float* x, float* out, std::size_t count) noexcept;

#if BoysFp16
template F16 BoysSingleF16<kBoysFullAccuracyMultiplier>(int n, F16 x) noexcept;
template void BoysAllOrdersF16<kBoysFullAccuracyMultiplier>(int nmax, F16 x, F16* out) noexcept;
template Bf16 BoysSingleBf16<kBoysFullAccuracyMultiplier>(int n, Bf16 x) noexcept;
template void BoysAllOrdersBf16<kBoysFullAccuracyMultiplier>(int nmax, Bf16 x, Bf16* out) noexcept;
#endif // BoysFp16

// The run-time tiers. Each relaxed rung is instantiated here so the dispatch
// below is a branch over code the library already holds, not a further
// instantiation per call site.
template void BoysAllOrders<64.0>(int nmax, double x, double* out) noexcept;
template void BoysAllOrders<256.0>(int nmax, double x, double* out) noexcept;
template void BoysAllOrders<1024.0>(int nmax, double x, double* out) noexcept;
template void BoysAllOrders<4096.0>(int nmax, double x, double* out) noexcept;
template void BoysAllOrders<16384.0>(int nmax, double x, double* out) noexcept;
template void BoysAllOrders<65536.0>(int nmax, double x, double* out) noexcept;

namespace {

// The m = 1 batch bound in regions A and B, the base the contract scales by m.
constexpr double kRelaxedBatchBound = 5.5e-14;

// The region C bound: the asymptotic branch has no coefficients to truncate,
// so it is the same at every m.
constexpr double kAsymptoticBound = 5.5e-14;

} // namespace

TierCoverage QueryTier(AccuracyTier tier, AccuracyRegion region, double tolerance) noexcept {
    TierCoverage coverage;

    coverage.reachable = (region == AccuracyRegion::kC)
                             ? kAsymptoticBound
                             : AccuracyMultiplier(tier) * kRelaxedBatchBound;
    coverage.meets = coverage.reachable <= tolerance;
    coverage.limiting = (region == AccuracyRegion::kA)   ? AccuracyComponent::kRegionASeed
                        : (region == AccuracyRegion::kB) ? AccuracyComponent::kRegionBFit
                                                         : AccuracyComponent::kRegionCAsymptotic;

    return coverage;
}

TierCoverage QueryTier(AccuracyTier tier, double x, double tolerance) noexcept {
    const AccuracyRegion region = x < detail::kX0   ? AccuracyRegion::kA
                                  : x < detail::kX1 ? AccuracyRegion::kB
                                                    : AccuracyRegion::kC;

    return QueryTier(tier, region, tolerance);
}

namespace {

// The tier dispatch, once per evaluation policy: the policy is a compile-time
// choice and a tier is a run-time one, so the policy is the template parameter
// and the tier stays the switch.
template <EvalPolicyLike Policy>
void AllOrdersAtTier(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    switch (tier)
    {
    case AccuracyTier::kReference:
        BoysAllOrders<kBoysFullAccuracyMultiplier, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed64:
        BoysAllOrders<64.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed256:
        BoysAllOrders<256.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed1024:
        BoysAllOrders<1024.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed4096:
        BoysAllOrders<4096.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed16384:
        BoysAllOrders<16384.0, Policy>(nmax, x, out);
        return;
    case AccuracyTier::kRelaxed65536:
        BoysAllOrders<65536.0, Policy>(nmax, x, out);
        return;

    default:
        break;
    }

    // A tier this build does not serve - a value cast in from outside the
    // enum, or one a newer header named. The selector must not fall through
    // and leave the caller's buffer unwritten, and it must not reach a rung
    // the caller did not ask for: the reference multiplier is the one rung
    // that is never coarser than any tier this build can name, and
    // AccuracyMultiplier reports the same choice, so the accuracy a caller
    // records beside these values is the accuracy they were computed at.
    BoysAllOrders<kBoysFullAccuracyMultiplier, Policy>(nmax, x, out);
}

} // namespace

void BoysAllOrdersAtTier(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    AllOrdersAtTier<EvalPolicy<>>(tier, nmax, x, out);
}

namespace {

// The single-order tier dispatch, the same shape as the batch one above: the
// policy is the compile-time choice and the tier stays the switch. It exists
// because the shape is a different call - an engine that reads one order at a
// time cannot reach a rung through an entry that computes every order - and not
// because the rung means anything different here.
template <EvalPolicyLike Policy>
double SingleAtTier(AccuracyTier tier, int n, double x) noexcept {
    switch (tier)
    {
    case AccuracyTier::kReference:
        return BoysSingle<kBoysFullAccuracyMultiplier, Policy>(n, x);
    case AccuracyTier::kRelaxed64:
        return BoysSingle<64.0, Policy>(n, x);
    case AccuracyTier::kRelaxed256:
        return BoysSingle<256.0, Policy>(n, x);
    case AccuracyTier::kRelaxed1024:
        return BoysSingle<1024.0, Policy>(n, x);
    case AccuracyTier::kRelaxed4096:
        return BoysSingle<4096.0, Policy>(n, x);
    case AccuracyTier::kRelaxed16384:
        return BoysSingle<16384.0, Policy>(n, x);
    case AccuracyTier::kRelaxed65536:
        return BoysSingle<65536.0, Policy>(n, x);

    default:
        break;
    }

    // A tier this build does not serve: the reference multiplier, the same
    // fallback the batch entry takes and for the same reason - it is the one
    // rung that is never coarser than any tier this build can name, and
    // AccuracyMultiplier reports it, so the number a caller records beside this
    // value is the accuracy it was computed at.
    return BoysSingle<kBoysFullAccuracyMultiplier, Policy>(n, x);
}

} // namespace

double BoysSingleAtTier(AccuracyTier tier, int n, double x) noexcept {
    return SingleAtTier<EvalPolicy<>>(tier, n, x);
}

double BoysSingleAtTier(AccuracyTier tier, EvalScheme scheme, int n, double x) noexcept {
    if (scheme == EvalScheme::kHorner)
    {
        return SingleAtTier<EvalPolicy<kDefaultFitRoute, EvalScheme::kHorner>>(tier, n, x);
    }

    return SingleAtTier<EvalPolicy<>>(tier, n, x);
}

double BoysSingleAtTier(AccuracyTier tier, FitRoute route, int n, double x) noexcept {
    return BoysSingleAtTier(tier, route, kDefaultEvalScheme, n, x);
}

double BoysSingleAtTier(
    AccuracyTier tier, FitRoute route, EvalScheme scheme, int n, double x) noexcept {
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            return SingleAtTier<EvalPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                tier, n, x);
        }

        return SingleAtTier<EvalPolicy<FitRoute::kRationalMinimax, kDefaultEvalScheme>>(tier, n, x);
    }

    return BoysSingleAtTier(tier, scheme, n, x);
}

void BoysAllOrdersAtTier(
    AccuracyTier tier, EvalScheme scheme, int nmax, double x, double* out) noexcept {
    if (scheme == EvalScheme::kHorner)
    {
        AllOrdersAtTier<EvalPolicy<kDefaultFitRoute, EvalScheme::kHorner>>(tier, nmax, x, out);
        return;
    }

    AllOrdersAtTier<EvalPolicy<>>(tier, nmax, x, out);
}

void BoysAllOrdersAtTier(
    AccuracyTier tier, FitRoute route, int nmax, double x, double* out) noexcept {
    BoysAllOrdersAtTier(tier, route, kDefaultEvalScheme, nmax, x, out);
}

void BoysAllOrdersAtTier(AccuracyTier tier,
                         FitRoute route,
                         EvalScheme scheme,
                         int nmax,
                         double x,
                         double* out) noexcept {
    // The same shape as BoysAllOrdersWithRoute, one axis wider: the route and
    // the scheme are compile-time choices and the tier is a run-time one, so the
    // pair is the template argument and the tier stays the switch inside. Both
    // selectors are answered as named - the rung's own body for the rung, the
    // named route's fits for the route - so a caller who names the pair is not
    // handed one of them under the other's name.
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            AllOrdersAtTier<EvalPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                tier, nmax, x, out);
            return;
        }

        AllOrdersAtTier<EvalPolicy<FitRoute::kRationalMinimax, kDefaultEvalScheme>>(
            tier, nmax, x, out);
        return;
    }

    BoysAllOrdersAtTier(tier, scheme, nmax, x, out);
}

std::span<const FitRouteInfo> BoysFitRoutes() noexcept {
    // The rows are the generated header's own measured figures rather than
    // numbers repeated here, so a table and the fits it describes cannot drift
    // apart: a regeneration that moved a delivered error moves the row.
    //
    // The region-A rows state the domain of the region's per-order tables -
    // every order's own piece, from zero to kX0 - because that is what the two
    // routes' tables cover and what makes the two rows a comparison.
    // kRatARouteLo is where the rational route's selector takes over: below it
    // the shipped lane is documented at 1e-15 and the rational fits hold the
    // wider bar, so the rational row states that boundary rather than claiming
    // the interval from zero that its selector does not serve.
    static const std::span<const FitRouteInfo> kRoutes = [] {
        static const FitRouteInfo kRows[] = {
            {FitRoute::kChebyshev,
             "chebyshev",
             AccuracyComponent::kRegionASeed,
             AccuracyRegion::kA,
             0.0,
             detail::kX0,
             0.0,
             detail::kRegionAFitChebStored,
             detail::kRegionAFitChebDelivered,
             detail::kRegionAFitBar},
            {FitRoute::kRationalMinimax,
             "rational-minimax",
             AccuracyComponent::kRegionASeed,
             AccuracyRegion::kA,
             0.0,
             detail::kX0,
             detail::kRatARouteLo,
             detail::kRegionAFitRatStored,
             detail::kRegionAFitRatDelivered,
             detail::kRegionAFitBar},
            {FitRoute::kChebyshev,
             "chebyshev",
             AccuracyComponent::kRegionBFit,
             AccuracyRegion::kB,
             detail::kX0,
             detail::kX1,
             detail::kX0,
             detail::kRegionBFitChebStored,
             detail::kRegionBFitChebDelivered,
             detail::kRegionBFitBar},
            {FitRoute::kRationalMinimax,
             "rational-minimax",
             AccuracyComponent::kRegionBFit,
             AccuracyRegion::kB,
             detail::kX0,
             detail::kX1,
             detail::kX0,
             detail::kRegionBFitRatStored,
             detail::kRegionBFitRatDelivered,
             detail::kRegionBFitBar},
        };
        return std::span<const FitRouteInfo>(kRows);
    }();

    return kRoutes;
}

void BoysAllOrdersWithRoute(FitRoute route, int nmax, double x, double* out) noexcept {
    BoysAllOrdersWithRoute(route, kDefaultEvalScheme, nmax, x, out);
}

void BoysAllOrdersWithRoute(
    FitRoute route, EvalScheme scheme, int nmax, double x, double* out) noexcept {
    // One body, instantiated once per (route, scheme) pair through the same
    // entry a compile-time caller uses: the zero argument's closed form, the
    // region split, the recurrences, the per-order rule and the domains are the
    // body's, and the policy names only the fits and the summation they are read
    // in. So naming a route cannot reach a fit the caller did not name, and
    // every argument the named route's rows do not cover is answered by the
    // body's own shipped-family branches - which read no route fit at all, so
    // those values are the default entry's bit for bit, not merely close to
    // them. Every route this build does not serve, a value outside the
    // enumeration included, is the default entry outright.
    //
    // The two axes select different things, so neither is folded into the
    // other: the route names the fits, and the scheme names the summation the
    // shipped family's coefficients are read in - the parts of the call the
    // named route's own fits do not serve. Both are answered as named.
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            BoysAllOrders<kBoysFullAccuracyMultiplier,
                          EvalPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                nmax, x, out);
            return;
        }

        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      EvalPolicy<FitRoute::kRationalMinimax, kDefaultEvalScheme>>(nmax, x, out);
        return;
    }

    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      EvalPolicy<kDefaultFitRoute, EvalScheme::kHorner>>(nmax, x, out);
        return;
    }

    BoysAllOrders<kBoysFullAccuracyMultiplier>(nmax, x, out);
}

// The evaluation-scheme report. Both tables are built once from the generated
// rows; the route in force is a build fact, so which of a row's two figures a
// query answers with is decided here rather than left to the caller.
constexpr double DeliveredIn(const EvalFitInfo& fit, backend::MulAddRoute route) noexcept {
    return route == backend::MulAddRoute::kSeparate ? fit.separate : fit.fused;
}

std::span<const EvalFitInfo> BoysEvalSchemeFits() noexcept {
    static const std::array<EvalFitInfo, std::size(detail::kSchemeRows)> rows = [] {
        std::array<EvalFitInfo, std::size(detail::kSchemeRows)> built{};

        for (std::size_t i = 0; i < built.size(); ++i)
        {
            const detail::SchemeRow& row = detail::kSchemeRows[i];
            EvalFitInfo& info = built[i];
            info.scheme = static_cast<EvalScheme>(row.scheme);
            info.lane = static_cast<EvalLane>(row.lane);
            info.region = static_cast<AccuracyRegion>(row.region);
            info.deg = row.deg;
            info.stored = row.stored;
            info.fused = row.fused;
            info.separate = row.separate;
        }

        return built;
    }();

    return rows;
}

std::span<const EvalSchemeInfo> BoysEvalSchemes() noexcept {
    constexpr std::size_t kSchemeCount =
        static_cast<std::size_t>(EvalScheme::kHorner) + 1;
    static const std::array<EvalSchemeInfo, kSchemeCount> rows = [] {
        std::array<EvalSchemeInfo, kSchemeCount> built{};
        const backend::MulAddRoute route = backend::detail::RouteInForce<double>();

        for (std::size_t s = 0; s < built.size(); ++s)
        {
            EvalSchemeInfo& info = built[s];
            info.scheme = static_cast<EvalScheme>(s);
            info.name = EvalSchemeName(info.scheme);
            info.route = route;

            for (const EvalFitInfo& fit : BoysEvalSchemeFits())
            {
                if (fit.scheme != info.scheme)
                {
                    continue;
                }

                ++info.lanes;
                info.deg = std::max(info.deg, fit.deg);
                info.stored = std::max(info.stored, fit.stored);
                info.delivered = std::max(info.delivered, DeliveredIn(fit, route));
            }
        }

        return built;
    }();

    return rows;
}

double BoysEvalSchemeDelivered(EvalScheme scheme, EvalLane lane) noexcept {
    const backend::MulAddRoute route = backend::detail::RouteInForce<double>();

    for (const EvalFitInfo& fit : BoysEvalSchemeFits())
    {
        if (fit.scheme == scheme && fit.lane == lane)
        {
            return DeliveredIn(fit, route);
        }
    }

    return 0.0;
}

const char* EvalSchemeName(EvalScheme scheme) noexcept {
    switch (scheme)
    {
    case EvalScheme::kSplitClenshaw:
        return "split-clenshaw";
    case EvalScheme::kHorner:
        return "horner";
    }

    return "unknown";
}

const char* PackAxisName(PackAxis axis) noexcept {
    switch (axis)
    {
    case PackAxis::kArguments:
        return "arguments";
    case PackAxis::kOrders:
        return "orders";
    }

    return "unknown";
}

const char* GranularityName(FitGranularity granularity) noexcept {
    switch (granularity)
    {
    case FitGranularity::kShipped:
        return "shipped";
    case FitGranularity::kNarrow:
        return "narrow";
    }

    return "unknown";
}

std::span<const PackAxisInfo> BoysPackAxes() noexcept {
    // Both members evaluate the region-A per-order fits of the double lane, so
    // both are certified against that region's bar. The packed lanes cover
    // region A and nothing else: past it the entries that carry an axis run the
    // certified scalar lanes, and the row's interval says where the packed
    // answer ends rather than implying it continues.
    static const std::array<PackAxisInfo, 2> rows = {{
        {PackAxis::kArguments, PackAxisName(PackAxis::kArguments), 4, 0.0, detail::kX0, 1e-15},
        {PackAxis::kOrders, PackAxisName(PackAxis::kOrders), 4, 0.0, detail::kX0, 1e-15},
    }};

    return rows;
}

} // namespace boys

// The arithmetic backends this build carries, as a report prints them.
//
// This translation unit answers for the scalar pair, because this is where the
// scalar arithmetic is compiled: the same flags a consumer's own code gets, and
// not the packed flags src/boys_simd.cpp carries. A contraction fact measured
// in the SIMD unit would describe that unit's arithmetic and would be printed
// beside the scalar lanes' values, which is the one thing the table must not
// do. The packed entries are appended by the unit that owns them.
namespace boys::backend {

std::span<const BackendInfo> BoysBackends() noexcept {
    static const std::span<const BackendInfo> kBackends = [] {
        static BackendInfo info[4];
        std::size_t n = 0;
        info[n++] = BackendInfo{ScalarFp64::kName, ScalarFp64::Contracts(),
                                detail::RouteInForce<double>()};
        info[n++] = BackendInfo{ScalarFp32::kName, ScalarFp32::Contracts(),
                                detail::RouteInForce<float>()};
        n += detail::AppendPackedBackends(info + n);
        return std::span<const BackendInfo>(info, n);
    }();
    return kBackends;
}

} // namespace boys::backend
