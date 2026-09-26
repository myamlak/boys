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

std::span<const FitRouteInfo> BoysFitRoutesF32() noexcept {
    // The float lane's rows, from the same generated header the double lane's
    // rows read, so a regeneration that moved a delivered error moves these
    // too. The lane's own tables are what the two region-A routes evaluate -
    // its shipped per-order fits and the rational cover - and its route
    // covers the whole of region A, so the rows state no servesFrom boundary
    // above zero: naming either route changes a value at every argument of
    // the interval.
    static const std::span<const FitRouteInfo> kRoutes = [] {
        static const FitRouteInfo kRows[] = {
            {FitRoute::kChebyshev,
             "chebyshev",
             AccuracyComponent::kRegionASeed,
             AccuracyRegion::kA,
             0.0,
             detail::kX0,
             0.0,
             detail::f32::kRegionAFitChebStored,
             detail::f32::kRegionAFitChebDelivered,
             detail::f32::kRegionAFitBar},
            {FitRoute::kRationalMinimax,
             "rational-minimax",
             AccuracyComponent::kRegionASeed,
             AccuracyRegion::kA,
             0.0,
             detail::kX0,
             0.0,
             detail::f32::kRegionAFitRatStored,
             detail::f32::kRegionAFitRatDelivered,
             detail::f32::kRegionAFitBar},
            {FitRoute::kChebyshev,
             "chebyshev",
             AccuracyComponent::kRegionBFit,
             AccuracyRegion::kB,
             detail::kX0,
             detail::kX1,
             detail::kX0,
             detail::f32::kRegionBFitChebStored,
             detail::f32::kRegionBFitChebDelivered,
             detail::f32::kRegionBFitBar},
            {FitRoute::kRationalMinimax,
             "rational-minimax",
             AccuracyComponent::kRegionBFit,
             AccuracyRegion::kB,
             detail::kX0,
             detail::kX1,
             detail::kX0,
             detail::f32::kRegionBFitRatStored,
             detail::f32::kRegionBFitRatDelivered,
             detail::f32::kRegionBFitBar},
        };
        return std::span<const FitRouteInfo>(kRows);
    }();

    return kRoutes;
}

float BoysSingleF32WithRoute(FitRoute route, int n, float x) noexcept {
    // One body instantiated per route, as the double lane's selector is: the
    // closed form at zero, the region split, the two recurrences and the
    // domains are the body's, and the route names only its two fits. So
    // naming a route cannot reach a fit the caller did not name, and region C
    // - which reads no coefficient at all - is answered by the body's own
    // branch, which is the default entry's code for those arguments.
    if (route == FitRoute::kRationalMinimax)
    {
        return detail::SingleOrderF32Body<detail::RationalFit32>(n, x);
    }

    return BoysSingleF32<kBoysFullAccuracyMultiplier>(n, x);
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

std::span<const FitGranularityInfo> BoysFitGranularities() noexcept {
    // The rows are the tables' own counts and the certification's own figures,
    // so a regeneration that moved a piece, a degree or a delivered error moves
    // the row rather than leaving a number here to drift from it.
    //
    // What a row does not cover is as much a part of it as what it does. Both
    // entries below carry the double lane's tables; the single-precision lanes
    // hold one coefficient set each and take no partition. The shipped row
    // carries every rung this build can name - the count is the tier
    // enumeration's own - both packing axes, because the across-orders lane is
    // instantiated for every scheme, rung and route over the shipped region-A
    // pieces, and both fit routes, because the shipped partitions carry a table
    // for each and the seeds are the routes' own. The narrow row carries the
    // reference rung, the arguments axis and the chebyshev route alone: its
    // region-A pieces are cut per order, so the across-orders lane, which steps
    // one order's coefficients to the next at a fixed stride, has no narrow
    // kernel, its degrees are the shipped rung's, so a relaxed rung has no
    // narrow table to truncate, and only the chebyshev route was cut on it, so
    // the rational route has no narrow table to evaluate. All three are unbuilt
    // work rather than unavailable options, and saying so here is what lets a
    // caller count them without reading the kernels.
    static const std::array<FitGranularityInfo, 2> rows = [] {
        static constexpr unsigned kBothAxes = (1u << static_cast<unsigned>(PackAxis::kArguments)) |
                                              (1u << static_cast<unsigned>(PackAxis::kOrders));
        static constexpr unsigned kChebBit = 1u << static_cast<unsigned>(FitRoute::kChebyshev);
        static constexpr unsigned kRatBit =
            1u << static_cast<unsigned>(FitRoute::kRationalMinimax);

        // A route's own figures over a region, read from the route table rather
        // than restated: these are the numbers the route rows publish, so a
        // regeneration moves both readings together.
        const auto routeStored = [](FitRoute route, AccuracyRegion region) {
            int stored = 0;

            for (const FitRouteInfo& row : BoysFitRoutes())
            {
                if (row.route == route && row.region == region)
                {
                    stored = row.stored;
                }
            }

            return stored;
        };

        // The worst a partition delivers and the bar it is certified against,
        // taken over the routes it holds and the regions they cover: a caller
        // who does not name a route may read any of them.
        const auto routeWorst = [](const auto& field) {
            double worst = 0.0;

            for (const FitRouteInfo& row : BoysFitRoutes())
            {
                worst = std::max(worst, field(row));
            }

            return worst;
        };

        int shippedADeg = 0;

        for (const detail::OrderPiece& piece : detail::kPieces)
        {
            shippedADeg = std::max(shippedADeg, piece.deg);
        }

        // The rational route is cut on the same pieces, so its degrees are read
        // over the same rows: numerator and denominator both bound what one
        // evaluation of a piece costs.
        for (const int deg : detail::kRatANumDeg)
        {
            shippedADeg = std::max(shippedADeg, deg);
        }

        for (const int deg : detail::kRatADenDeg)
        {
            shippedADeg = std::max(shippedADeg, deg);
        }

        int narrowADeg = 0;
        int narrowAStored = 0;

        for (const detail::OrderPiece& piece : detail::kNarrowAPieces)
        {
            narrowADeg = std::max(narrowADeg, piece.deg);
            narrowAStored += piece.deg + 1;
        }

        // The interval a partition's own tables serve, read from the fitted
        // routes' own domains: region A's per-order tables from zero to kX0 and
        // region B's seed from kX0 to kX1, under either partition, so the two
        // edges are the lowest left edge and the highest right edge the fitted
        // rows state. It is the domain the partition's figure holds on and no
        // wider one: above the top edge the entry runs region C's asymptotic
        // form, which no partition replaces and whose figure is the certified
        // lane's, so a caller reading the figure against a wider range would be
        // matching a promise about the fitted tables to an error that is not
        // theirs.
        double fittedLo = std::numeric_limits<double>::infinity();
        double fittedHi = 0.0;

        for (const FitRouteInfo& row : BoysFitRoutes())
        {
            fittedLo = std::min(fittedLo, row.lo);
            fittedHi = std::max(fittedHi, row.hi);
        }

        // No fitted row at all leaves no interval to report, and one made up here
        // would be a claim about tables that are not there.
        if (!(fittedLo <= fittedHi))
        {
            fittedLo = 0.0;
            fittedHi = 0.0;
        }

        std::array<FitGranularityInfo, 2> built{};

        built[0].granularity = FitGranularity::kShipped;
        built[0].name = GranularityName(FitGranularity::kShipped);
        built[0].routes = kChebBit | kRatBit;
        built[0].rungs = static_cast<int>(AccuracyTier::kRelaxed65536) + 1;
        built[0].axes = kBothAxes;
        built[0].regionAPieces = static_cast<int>(std::size(detail::kPieces));
        built[0].regionADeg = shippedADeg;
        built[0].regionAStored = routeStored(FitRoute::kChebyshev, AccuracyRegion::kA) +
                                 routeStored(FitRoute::kRationalMinimax, AccuracyRegion::kA);
        built[0].regionBPieces = 1;
        built[0].regionBDeg = std::max({detail::kBDeg, detail::kRatBnumDeg, detail::kRatBdenDeg});
        built[0].regionBStored = routeStored(FitRoute::kChebyshev, AccuracyRegion::kB) +
                                 routeStored(FitRoute::kRationalMinimax, AccuracyRegion::kB);
        built[0].delivered =
            routeWorst([](const FitRouteInfo& row) { return row.delivered; });
        built[0].bound = routeWorst([](const FitRouteInfo& row) { return row.bound; });
        built[0].lo = fittedLo;
        built[0].hi = fittedHi;

        built[1].granularity = FitGranularity::kNarrow;
        built[1].name = GranularityName(FitGranularity::kNarrow);
        built[1].routes = kChebBit;
        built[1].rungs = 1;
        built[1].axes = 1u << static_cast<unsigned>(PackAxis::kArguments);
        built[1].regionAPieces = static_cast<int>(std::size(detail::kNarrowAPieces));
        built[1].regionADeg = narrowADeg;
        built[1].regionAStored = narrowAStored;
        built[1].regionBPieces = detail::kNarrowBPieces;
        built[1].regionBDeg = detail::kNarrowBDeg;
        built[1].regionBStored = static_cast<int>(std::size(detail::kNarrowBcoeffs));
        // The narrow certification publishes a per-piece round-up of what each
        // piece delivers rather than a bar the pieces were cut at, and that
        // round-up is the figure a caller may rely on: it bounds a sweep and not
        // only the one that measured it.
        built[1].delivered = std::max(detail::kNarrowRows[0].fused, detail::kNarrowRows[0].separate);
        built[1].bound = built[1].delivered;
        built[1].lo = fittedLo;
        built[1].hi = fittedHi;

        return built;
    }();

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
