#include "boys/boys.hpp"

#include "boys/boys_cuda_options.hpp"
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
// template definitions: every entry is compiled twice under if constexpr, the
// m = 1 branch being the certified body verbatim (the bit-identity pin). This
// TU provides the default m = 1 explicit instantiations that the
// extern-template declarations in boys.hpp route every default call site to. A
// call site that names any other multiplier compiles its rung from the shipped
// definition instead of linking here.

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

// The policy a run-time selector builds from the axes its caller named. The
// three axes a selector does not take - the budget, the packing axis and the
// granularity - are spelled here at their defaults rather than left to the
// template's own.
template <FitRoute kRoute, EvalScheme kScheme>
using SelectorPolicy =
    EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, kDefaultPackAxis, kDefaultFitGranularity>;

} // namespace

void BoysAllOrdersAtTier(AccuracyTier tier, int nmax, double x, double* out) noexcept {
    BoysAllOrdersAtTier<EvalPolicy<>>(tier, nmax, x, out);
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

    // A tier this build does not serve: the same fallback the batch entry takes
    // and for the same reason.
    return BoysSingle<kBoysFullAccuracyMultiplier, Policy>(n, x);
}

} // namespace

double BoysSingleAtTier(AccuracyTier tier, int n, double x) noexcept {
    return SingleAtTier<EvalPolicy<>>(tier, n, x);
}

double BoysSingleAtTier(AccuracyTier tier, EvalScheme scheme, int n, double x) noexcept {
    // Each arm names the scheme the caller named: the other arm is the other
    // enumerator and not the default, so a move of the default cannot hand a
    // caller naming one scheme the summation of the other.
    if (scheme == EvalScheme::kHorner)
    {
        return SingleAtTier<SelectorPolicy<kDefaultFitRoute, EvalScheme::kHorner>>(tier, n, x);
    }

    return SingleAtTier<SelectorPolicy<kDefaultFitRoute, EvalScheme::kSplitClenshaw>>(tier, n, x);
}

double BoysSingleAtTier(AccuracyTier tier, FitRoute route, int n, double x) noexcept {
    // The default scheme, the axis this overload leaves unnamed. It must answer
    // as BoysAllOrdersWithRoute(route, ...) does - the same shape, a route and
    // no scheme - because the route table's row is the figure for both.
    return BoysSingleAtTier(tier, route, kDefaultEvalScheme, n, x);
}

double BoysSingleAtTier(
    AccuracyTier tier, FitRoute route, EvalScheme scheme, int n, double x) noexcept {
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            return SingleAtTier<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                tier, n, x);
        }

        return SingleAtTier<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw>>(
            tier, n, x);
    }

    // The Chebyshev arms name this route rather than reaching the entry that
    // answers the default route: the caller named a route here.
    if (scheme == EvalScheme::kHorner)
    {
        return SingleAtTier<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kHorner>>(tier, n, x);
    }

    return SingleAtTier<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kSplitClenshaw>>(tier, n, x);
}

void BoysAllOrdersAtTier(
    AccuracyTier tier, EvalScheme scheme, int nmax, double x, double* out) noexcept {
    // Each arm names the scheme the caller named; see BoysSingleAtTier.
    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrdersAtTier<SelectorPolicy<kDefaultFitRoute, EvalScheme::kHorner>>(tier, nmax, x, out);
        return;
    }

    BoysAllOrdersAtTier<SelectorPolicy<kDefaultFitRoute, EvalScheme::kSplitClenshaw>>(tier, nmax, x, out);
}

void BoysAllOrdersAtTier(
    AccuracyTier tier, FitRoute route, int nmax, double x, double* out) noexcept {
    // The default scheme; see BoysSingleAtTier.
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
    // pair is the template argument and the tier stays the switch inside.
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            BoysAllOrdersAtTier<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                tier, nmax, x, out);
            return;
        }

        BoysAllOrdersAtTier<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw>>(
            tier, nmax, x, out);
        return;
    }

    // The Chebyshev arms name this route rather than reaching the entry that
    // answers the default route; see BoysSingleAtTier.
    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrdersAtTier<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kHorner>>(tier, nmax, x, out);
        return;
    }

    BoysAllOrdersAtTier<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kSplitClenshaw>>(
        tier, nmax, x, out);
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
    // wider bar, so the rational row states that boundary.
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
    // The default scheme, not the reference one: the values a route does not
    // serve are the default entry's bit for bit, which requires the scheme the
    // default entry reads.
    BoysAllOrdersWithRoute(route, kDefaultEvalScheme, nmax, x, out);
}

void BoysAllOrdersWithRoute(
    FitRoute route, EvalScheme scheme, int nmax, double x, double* out) noexcept {
    // One body, instantiated once per (route, scheme) pair through the same
    // entry a compile-time caller uses: the zero argument's closed form, the
    // region split, the recurrences, the per-order rule and the domains are the
    // body's, and the policy names only the fits and the summation they are read
    // in. Every argument the named route's rows do not cover is answered by the
    // body's own shipped-family branches, which read no route fit at all, so
    // those values are the default entry's bit for bit. Every route this build
    // does not serve, a value outside the enumeration included, is the default
    // entry outright.
    //
    // The two axes select different things: the route names the fits, and the
    // scheme names the summation the shipped family's coefficients are read in.
    // Both are answered as named.
    if (route == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            BoysAllOrders<kBoysFullAccuracyMultiplier,
                          SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                nmax, x, out);
            return;
        }

        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw>>(nmax, x,
                                                                                        out);
        return;
    }

    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrders<kBoysFullAccuracyMultiplier,
                      SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kHorner>>(nmax, x, out);
        return;
    }

    BoysAllOrders<kBoysFullAccuracyMultiplier,
                  SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kSplitClenshaw>>(nmax, x, out);
}

// The evaluation-scheme report. Both tables are built once from the generated
// rows; the route in force is a build fact, so which of a row's two figures a
// query answers with is decided here.
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
    // domains are the body's, and the route names only its two fits. Region C,
    // which reads no coefficient at all, is answered by the body's own branch -
    // the default entry's code for those arguments.
    if (route == FitRoute::kRationalMinimax)
    {
        return detail::SingleOrderF32Body<detail::RationalFit32<>>(n, x);
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
    case FitGranularity::kUniform:
        return "uniform";
    }

    return "unknown";
}

const char* DivisionFormName(DivisionForm form) noexcept {
    switch (form)
    {
    case DivisionForm::kExactDivision:
        return "exact-division";
    case DivisionForm::kPlainReciprocal:
        return "plain-reciprocal";
    case DivisionForm::kRefinedReciprocal:
        return "refined-reciprocal";
    }

    return "unknown";
}

std::span<const PackAxisInfo> BoysPackAxes() noexcept {
    // Both members evaluate the region-A per-order fits of the double lane, so
    // both are certified against that region's bar. The packed lanes cover
    // region A and nothing else: past it the entries that carry an axis run the
    // certified scalar lanes, and the row's interval says where the packed
    // answer ends.
    static const std::array<PackAxisInfo, 2> rows = {{
        {PackAxis::kArguments, PackAxisName(PackAxis::kArguments), 4, 0.0, detail::kX0, 1e-15},
        {PackAxis::kOrders, PackAxisName(PackAxis::kOrders), 4, 0.0, detail::kX0, 1e-15},
    }};

    return rows;
}

std::span<const DivisionFormInfo> BoysDivisionForms() noexcept {
    // Every row is served by every entry: the form is read inside the
    // recurrence's own step, so naming one selects arithmetic and never an
    // entry, and there is no combination of the other axes it is missing from.
    static const std::array<DivisionFormInfo, 3> rows = {{
        {DivisionForm::kExactDivision, DivisionFormName(DivisionForm::kExactDivision)},
        {DivisionForm::kPlainReciprocal, DivisionFormName(DivisionForm::kPlainReciprocal)},
        {DivisionForm::kRefinedReciprocal, DivisionFormName(DivisionForm::kRefinedReciprocal)},
    }};

    return rows;
}

std::span<const FitGranularityInfo> BoysFitGranularities() noexcept {
    // The rows are the tables' own counts and the certification's own figures,
    // so a regeneration that moved a piece, a degree or a delivered error moves
    // the row rather than leaving a number here to drift from it.
    //
    // Every entry carries the double lane's tables; the single-precision lanes
    // read tables of their own - that engine's pieces, coefficients and rung
    // tables, cut for its own arithmetic and its own budget - so a row here
    // describes the double lane and not them: which partitions, routes, axes and
    // rungs that lane serves is stated by its own carriage where the call is
    // named. The uniform row's table is the double lane's own grid
    // (detail::kFlatCoeffs); the single-precision engine's grid
    // (detail::f32::kFlatCoeffsF32) is the device lane's, and no entry of the
    // host single-precision lane enters a uniform branch, which is why that
    // lane's carriage refuses the partition where it is named. The shipped row
    // carries every rung this build can name - the count is the tier
    // enumeration's own - both packing axes, because the across-orders lane is
    // instantiated for every scheme, rung and route over the shipped region-A
    // pieces, and both fit routes, because the shipped partitions carry a table
    // for each. The narrow row carries every rung this build can name and both
    // packing axes, because each of the two reads a cut of this partition's own
    // pieces - a relaxed rung truncates the partition's stored fits against a
    // degree table derived over them, and the across-orders lane gathers each of
    // the orders it holds its own piece and coefficients - and both fit routes,
    // because this partition was cut for the rational route as well: it stores
    // one numerator/denominator pair over each of its pieces, region A's and
    // region B's alike. The one fact these fields state for the double lane's
    // tables alone is the same axis on the float engine, and there the calls
    // state it: that engine's narrow partition is instantiated at every rung the
    // enumeration names, at both schemes and both routes, one entry per cell at
    // each of the lane's two budgets (the packed lane's narrow rungs,
    // boys_orders_simd.cpp), and the degree tables those entries read are cut
    // from that lane's own narrow pieces per role
    // (boys_effective_degrees.hpp). So a rung of the narrow partition is the
    // float lane's own table and not the double lane's degrees under a float
    // name.
    //
    // The uniform row states the one route, the one packing axis and the one
    // rung this build carries that partition at, and each of the three is a
    // refusal the build makes where the call is named rather than a figure
    // nothing would catch: the rational route is refused at compile time because
    // its fit is a numerator/denominator pair per derived piece and the grid is
    // fixed, the across-orders packing axis has no instantiation over this
    // partition, and a relaxed rung is refused at compile time because the table
    // stores one degree for every order and every interval. Its region-A fields
    // describe the grid rather than a cut of a region: the grid covers
    // [0, kFlatHi) whole, region A's and region B's arguments alike, and its
    // intervals are intervals rather than pieces. Its region-B fields are zero
    // for the same reason and not for want of a figure: there is no second table
    // here, because above the grid's top edge the call reaches the asymptotic
    // every route ends in, which reads no fit at all.
    static const std::array<FitGranularityInfo, 3> rows = [] {
        static constexpr unsigned kBothAxes = (1u << static_cast<unsigned>(PackAxis::kArguments)) |
                                              (1u << static_cast<unsigned>(PackAxis::kOrders));
        static constexpr unsigned kArgumentsBit =
            1u << static_cast<unsigned>(PackAxis::kArguments);
        static constexpr unsigned kChebBit = 1u << static_cast<unsigned>(FitRoute::kChebyshev);
        static constexpr unsigned kRatBit =
            1u << static_cast<unsigned>(FitRoute::kRationalMinimax);

        // The uniform table's certification rows describe the table this row's
        // counts are read from, so the two readings are tied here rather than
        // left to drift: a regeneration that moved the grid, its degree or its
        // stored size moves both.
        static_assert(std::size(detail::kFlatRows) == 2,
                      "the uniform table is certified one row per evaluation scheme");
        static_assert(detail::kFlatRows[0].deg == detail::kFlatDeg &&
                          detail::kFlatRows[1].deg == detail::kFlatDeg &&
                          detail::kFlatRows[0].intervals == detail::kFlatIntervals &&
                          detail::kFlatRows[1].intervals == detail::kFlatIntervals &&
                          detail::kFlatRows[0].stored ==
                              static_cast<int>(std::size(detail::kFlatCoeffs)) &&
                          detail::kFlatRows[1].stored ==
                              static_cast<int>(std::size(detail::kFlatCoeffs)),
                      "the uniform table's certification rows must describe the table this "
                      "row's counts are read from");

        // A route's own figures over a region, read from the route table rather
        // than restated, so a regeneration moves both readings together.
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

        for (const int deg : detail::kNarrowRatANumDeg)
        {
            narrowADeg = std::max(narrowADeg, deg);
        }

        for (const int deg : detail::kNarrowRatADenDeg)
        {
            narrowADeg = std::max(narrowADeg, deg);
        }

        int narrowBDeg = detail::kNarrowBDeg;

        for (const int deg : detail::kNarrowRatBNumDeg)
        {
            narrowBDeg = std::max(narrowBDeg, deg);
        }

        for (const int deg : detail::kNarrowRatBDenDeg)
        {
            narrowBDeg = std::max(narrowBDeg, deg);
        }

        // The interval a partition's own tables serve, read from the fitted
        // routes' own domains: region A's per-order tables from zero to kX0 and
        // region B's seed from kX0 to kX1, under either partition, so the two
        // edges are the lowest left edge and the highest right edge the fitted
        // rows state. Above the top edge the entry runs region C's asymptotic
        // form, which no partition replaces and whose figure is the certified
        // lane's, so the figure holds on this domain and no wider one.
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

        std::array<FitGranularityInfo, 3> built{};

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
        built[1].routes = kChebBit | kRatBit;
        built[1].rungs = static_cast<int>(AccuracyTier::kRelaxed65536) + 1;
        built[1].axes = kBothAxes;
        built[1].regionAPieces = static_cast<int>(std::size(detail::kNarrowAPieces));
        built[1].regionADeg = narrowADeg;
        built[1].regionAStored = narrowAStored + detail::kNarrowRatAStored;
        built[1].regionBPieces = detail::kNarrowBPieces;
        built[1].regionBDeg = narrowBDeg;
        built[1].regionBStored =
            static_cast<int>(std::size(detail::kNarrowBcoeffs)) + detail::kNarrowRatBStored;
        // The narrow certification publishes a per-piece round-up of what each
        // piece delivers rather than a bar the pieces were cut at, and that
        // round-up is the figure a caller may rely on: it bounds a sweep and not
        // only the one that measured it. Both routes the row holds are read into
        // it, for the reason routeWorst gives above.
        built[1].delivered = std::max({detail::kNarrowRows[0].fused,
                                       detail::kNarrowRows[0].separate,
                                       detail::kNarrowRatADeliveredFused,
                                       detail::kNarrowRatADeliveredSeparate,
                                       detail::kNarrowRatBDeliveredFused,
                                       detail::kNarrowRatBDeliveredSeparate});
        built[1].bound = built[1].delivered;
        built[1].lo = fittedLo;
        built[1].hi = fittedHi;

        built[2].granularity = FitGranularity::kUniform;
        built[2].name = GranularityName(FitGranularity::kUniform);
        // One route, one packing axis and one rung, each of them the build's own
        // refusal rather than a figure read from a table: the rational family has
        // no member over a fixed grid, the across-orders lane has no
        // instantiation over it, and its table is stored at one degree for every
        // order and every interval, so no rung of it can be cut. Each refusal is
        // made where the call is named, so a cell this row does not cover is one
        // a compile or a link answers rather than one answered by another
        // partition's tables.
        built[2].routes = kChebBit;
        built[2].rungs = 1;
        built[2].axes = kArgumentsBit;
        // The grid itself, which is what this partition is where the other two
        // are walks: the intervals are the grid's cells rather than pieces the
        // proved bound put where the function needs them, the degree is the one
        // every order and every cell of it is stored at, and what it stores is
        // the whole table - one fit per order per interval.
        built[2].regionAPieces = detail::kFlatIntervals;
        built[2].regionADeg = detail::kFlatDeg;
        built[2].regionAStored = static_cast<int>(std::size(detail::kFlatCoeffs));
        built[2].regionBPieces = 0;
        built[2].regionBDeg = 0;
        built[2].regionBStored = 0;
        // The worse of the table's two evaluation schemes and of its two
        // multiply-add routes, which is the one figure that holds for a caller
        // who names a scheme and a route the table stores both of: the
        // certification's own round-up rather than a number restated here. The
        // row is one figure for the partition and the delivered figure is the
        // bound it is certified against, as the narrow row's is and for the same
        // reason.
        double uniformWorst = 0.0;

        for (const detail::FlatRow& row : detail::kFlatRows)
        {
            uniformWorst = std::max({uniformWorst, row.fused, row.separate});
        }

        built[2].delivered = uniformWorst;
        built[2].bound = uniformWorst;
        built[2].lo = 0.0;
        built[2].hi = detail::kFlatHi;

        return built;
    }();

    return rows;
}

std::span<const LaneContractInfo> BoysLaneContracts() noexcept {
    // The figures the README's contract table publishes, stated here once so
    // that the table, the accuracy gate and BoysAccuracyGuaranteed read one
    // number rather than three transcriptions of one. Each is the figure the
    // lane documents for one value over the whole of x >= 0 at the reference
    // multiplier; the half-precision lane's f32 entries add a term of their own
    // under their region-B exponential option, which is why the row carries it.
    static const std::array<LaneContractInfo, 4> rows = {{
        {Precision::kFp64, "fp64", 5.5e-14, 0.0, "throughout, every region"},
        {Precision::kFp32, "fp32", 1.5e-7, 0.0, "throughout, every region"},
        {Precision::kFp16, "fp16", 1.5e-7, 0.0,
         "the single-precision lane's own figure, plus half of the last representable digit of the "
         "returned value and claimed only where the value exceeds the sum. The half lane computes "
         "in that arithmetic and stores what it returns, so it cannot be more accurate than the "
         "lane whose arithmetic it runs: a bar below that figure is one no conforming host can "
         "keep, and a host whose rounding differs delivers the fit's own error through it"},
        {Precision::kFp32Device, "fp32-device", 1.5e-7, 8e-8,
         "plus 8e-8 under the fast region-B exponential, which is the corrected seed's own "
         "contribution"},
    }};

    return rows;
}

namespace {

// Whether this revision carries a combination, and why not where it does not.
struct Carriage {
    bool carried = false;
    const char* reason = "";
};

// The single-precision lanes' carriage, read off the entries those lanes build.
// Both packing axes are served on the shipped partition: the arguments axis by
// the per-order bodies, the orders axis by the packed lane that steps one
// order's coefficients to the next order's. The narrow partition is stored for
// this lane and is read at the reference multiplier on both axes; a rung of it
// is a cut of the narrow pieces' own degree table, which this lane derives,
// and the bodies below read it - so a narrow rung is served on both axes and
// on every route and scheme, because the packed lane the orders axis names is
// the shipped partition's and a narrow policy takes the scalar path instead.
//
// The uniform partition is not served on this lane at all: this lane's entries
// are the shipped route's recurrences, and no entry of it reads a grid. The
// partition is a row of the host table because the double lane's entries do
// read one, and a row in that table is not a claim about this lane - which is
// why the refusal is here rather than left to the enumeration's size.
Carriage CarriesSingle(FitRoute route,
                       EvalScheme scheme,
                       PackAxis axis,
                       FitGranularity granularity,
                       AccuracyTier tier) noexcept {
    // Every combination this library names is served on both axes and on the
    // two partitions this lane stores, at every rung. A cut has to be read
    // against the table the scheme actually sums: the Chebyshev cut under the
    // Horner scheme's monomial table delivers outside the bound on 1964 of 56694
    // cells at m = 64. With the basis carried through, the same combination is a
    // rung of the family the caller named on either partition, so there is no
    // cell of those two left for this rule to refuse.
    //
    // What it refuses is a value outside the enumerations, which names no
    // combination at all rather than one this revision does not carry, and the
    // uniform partition, whose row describes a grid no entry of this lane reads.
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    const std::size_t t = static_cast<std::size_t>(tier);

    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size() ||
        t > static_cast<std::size_t>(AccuracyTier::kRelaxed65536))
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis, a partition and a rung from "
                "the enumerations this revision publishes"};
    }

    if (granularity == FitGranularity::kUniform)
    {
        return {false,
                "this lane's bodies reach their values through a recurrence over the orders and "
                "enter no uniform branch: the row the partition is enumerated by describes the "
                "double lane's grid, and no entry of this lane reads a grid at all. A caller "
                "naming it here names a table this lane has no body for"};
    }

    return {true, ""};
}

// The device lane's rule. Its accuracy multiplier is a template argument at the
// call site against a certified degree table the lane holds per rung, so every
// rung the lane instantiates is served. Those are the option space's seven
// (AccuracyTier) beside the lane's own six — one union of twelve multipliers,
// kDeviceRungs (boys_cuda_options.hpp) — and this rule reads that table rather
// than a list kept here: a value outside it names a multiplier no entry of the
// lane was compiled for.
//
// What the lane this rule answers for carries, and it is that lane's own entries
// this reads rather than the double lane's beside it: Precision::kFp32Device is
// the device lane's single-precision one, and every entry of it is the float
// engine's arithmetic.
//
// The lane carries one fit route and one packing axis: the shipped Chebyshev
// route's fits, read as a per-argument ladder, which is the shape every one of
// its fp32 entries has (BoysCuda::SingleF32, BoysCuda::AllOrdersF32 and the rung
// form each of the two has). The rational family has no member in it and no
// across-orders entry of it is instantiated, so both are refused below with the
// work they would take - they are lanes this library carries and this lane does
// not, which is a body to write and not a combination that cannot exist.
//
// Its partitions are three: the shipped one at every rung of the lane's table
// (BoysCuda::AllOrdersF32AtRung), and the narrow and uniform ones over the float
// engine's own pieces and grid (BoysCuda::AllOrdersF32Narrow and
// BoysCuda::AllOrdersF32Uniform, with the monomial form of each beside it), each
// served at the full-accuracy multiplier alone. That last is the same thing this
// lane's entries state in their own contracts: the device lane holds no rung
// table cut from the float lane's pieces, so a relaxed call has no degrees to
// read. What the scheme axis reaches on which partition is the lane's own table's
// to state: this function has one scheme field for the whole call, and that
// table is where the two names are paired with the tables they sum.
Carriage CarriesDevice(FitRoute route,
                       EvalScheme scheme,
                       PackAxis axis,
                       FitGranularity granularity,
                       AccuracyTier tier) noexcept {
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    const std::size_t t = static_cast<std::size_t>(tier);

    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size() ||
        t > static_cast<std::size_t>(AccuracyTier::kRelaxed65536))
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis, a partition and a rung from "
                "the enumerations this revision publishes"};
    }

    if (route != FitRoute::kChebyshev)
    {
        return {false,
                "this lane carries the shipped Chebyshev route's fits and no member of the "
                "rational family: its fp32 entries read the Chebyshev route's tables and the "
                "selector has no rational member over them. The family is a table and a body this "
                "lane has not been given rather than a combination that cannot exist"};
    }

    if (axis != PackAxis::kArguments)
    {
        return {false,
                "no across-orders entry of this lane is instantiated over the float engine's "
                "tables: its packed lane carries four orders of one argument and its fp32 entries "
                "are per-argument ladders. The axis names a body this lane has not been given "
                "rather than a shape the call cannot have"};
    }

    if (granularity != FitGranularity::kShipped && tier != AccuracyTier::kReference)
    {
        return {false,
                "this lane's narrow and uniform fp32 entries are served at the full-accuracy "
                "multiplier alone: the device lane holds no degree table cut from the float "
                "engine's own pieces, so a relaxed call over them has no degrees to read. The "
                "lane's entries state that as unbuilt work rather than as a property of the "
                "partition"};
    }

    if (!DeviceRungServed(AccuracyMultiplier(tier)))
    {
        return {false,
                "the device lane's degree tables are cut per rung and its entries are compiled at "
                "each rung it serves, so this multiplier is one no entry of that lane answers at: "
                "the lane serves the option space's rungs beside its own, and a call naming any "
                "other would be reading another rung's cut"};
    }

    return {true, ""};
}

} // namespace

AccuracyFigure BoysAccuracyGuaranteed(Precision precision,
                                      FitRoute route,
                                      EvalScheme scheme,
                                      PackAxis axis,
                                      FitGranularity granularity,
                                      AccuracyTier tier) noexcept
{
    const std::span<const LaneContractInfo> lanes = BoysLaneContracts();
    const std::size_t index = static_cast<std::size_t>(precision);
    AccuracyFigure figure;

    figure.reading = AccuracyReading::kGuaranteed;
    figure.source = "BoysLaneContracts()";

    if (index >= lanes.size())
    {
        figure.reason = "no lane of this library has that precision";

        return figure;
    }

    const LaneContractInfo& lane = lanes[index];
    const Carriage carriage = [&] {
        switch (precision)
        {
        case Precision::kFp32Device:
            return CarriesDevice(route, scheme, axis, granularity, tier);
        case Precision::kFp32:
        case Precision::kFp16:
            return CarriesSingle(route, scheme, axis, granularity, tier);
        case Precision::kFp64:
            break;
        }

        const std::span<const FitGranularityInfo> partitions = BoysFitGranularities();
        const std::size_t p = static_cast<std::size_t>(granularity);
        Carriage c;

        // The enumeration guard the two carriers above make, and it is made here
        // rather than left to the row tables below. A partition's bitmask answers
        // an out-of-enumeration route or axis with a refusal about the partition
        // and answers an out-of-enumeration *scheme* with nothing at all: no row
        // of a partition repeats the scheme axis, so there is no bit to be clear
        // and the combination was read as carried. What the sentence says is what
        // the other lanes say, because a value outside an enumeration names no
        // combination on any lane, and the answer for it is one answer.
        if (static_cast<std::size_t>(route) >= BoysFitRoutes().size() ||
            static_cast<std::size_t>(scheme) >= BoysEvalSchemes().size() ||
            static_cast<std::size_t>(axis) >= BoysPackAxes().size() ||
            p >= partitions.size() ||
            static_cast<std::size_t>(tier) >
                static_cast<std::size_t>(AccuracyTier::kRelaxed65536))
        {
            c.reason =
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis, a partition and a rung from "
                "the enumerations this revision publishes";

            return c;
        }

        const FitGranularityInfo& row = partitions[p];

        if (static_cast<int>(tier) >= row.rungs)
        {
            c.reason =
                "a relaxed rung reads a stored row at a per-order effective degree, and a "
                "partition whose table is fitted at one degree for every order and every "
                "interval has neither a criterion to cut it by nor such a table to read: the "
                "uniform partition is served at the reference multiplier alone, and the build "
                "refuses a rung of it where the call is named rather than answering it from "
                "another partition's fits";
        } else if (!FitGranularityHasRoute(row, route))
        {
            c.reason =
                "the partition's tables do not hold this fit route: the rational family is a "
                "numerator/denominator pair over each piece of a partition a criterion derived, "
                "and the uniform grid is fixed rather than derived, so it has no piece to pair "
                "and no member of that route over it. The combination is refused where it is "
                "named, and the member is a fit to derive rather than a shape the call cannot "
                "have";
        } else if (!FitGranularityHasAxis(row, axis))
        {
            c.reason =
                "the partition has no kernel for this packing axis: the across-orders packed lane "
                "is instantiated over the two derived partitions' own pieces and over no other "
                "table, so the axis names a kernel the uniform grid has no body for. That body is "
                "one to write rather than a shape the call cannot have";
        } else
        {
            c.carried = true;
        }

        return c;
    }();

    if (!carriage.carried)
    {
        figure.reason = carriage.reason;

        return figure;
    }

    figure.available = true;
    figure.value = AccuracyMultiplier(tier) * lane.bound + lane.additive;
    figure.source = lane.source;

    return figure;
}

AccuracyFigure BoysAccuracyDelivered(Precision precision,
                                     FitRoute route,
                                     EvalScheme scheme,
                                     PackAxis axis,
                                     FitGranularity granularity,
                                     AccuracyTier tier) noexcept
{
    // A combination this build refuses has no delivered figure for the same
    // reason it has no bound.
    AccuracyFigure figure = BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity, tier);

    figure.reading = AccuracyReading::kDelivered;
    figure.value = 0.0;

    if (!figure.available)
    {
        return figure;
    }

    if (precision == Precision::kFp16)
    {
        figure.available = false;
        figure.reason =
            "the half-precision lanes' error is dominated by the format's own quantum half a "
            "representable digit of the returned value, which is a property of the value and not "
            "of the call: no row of this library measured a half-typed return, so there is no "
            "delivered figure to read and the guaranteed figure above is the one to use. Where "
            "the value falls at or below the format's floor no accuracy is claimed at all, and a "
            "suite that reports this lane counts those arguments rather than passing them";
        figure.source = "";

        return figure;
    }

    if (tier != AccuracyTier::kReference)
    {
        figure.available = false;
        figure.reason =
            "a delivered figure is a measurement and the rows carry one at the multiplier they "
            "were measured at, which is the reference one: no row publishes what a relaxed rung "
            "delivers, so there is no number to read here. The accuracy gate measures the rungs "
            "and its report is where those figures are";
        figure.source = "";

        return figure;
    }

    // The worst figure over the rows the combination names: a composition is at
    // least as bad as its worst part.
    double worst = 0.0;
    const char* source = "BoysFitRoutesF32()";

    // The half and single lanes' own route table; the double lane's, which the
    // device lane's single-precision entries read their region-A fits from.
    const std::span<const FitRouteInfo> routes =
        (precision == Precision::kFp32 || precision == Precision::kFp16) ? BoysFitRoutesF32()
                                                                         : BoysFitRoutes();

    if (precision != Precision::kFp32 && precision != Precision::kFp16)
    {
        source = "BoysFitRoutes()";
    }

    for (const FitRouteInfo& row : routes)
    {
        if (row.route == route)
        {
            worst = std::max(worst, row.delivered);
        }
    }

    if (precision == Precision::kFp64)
    {
        source = "BoysFitRoutes(), BoysFitGranularities() and BoysEvalSchemes()";

        const std::size_t p = static_cast<std::size_t>(granularity);

        if (p < BoysFitGranularities().size())
        {
            worst = std::max(worst, BoysFitGranularities()[p].delivered);
        }
    }

    for (const EvalSchemeInfo& row : BoysEvalSchemes())
    {
        if (row.scheme == scheme)
        {
            worst = std::max(worst, row.delivered);
        }
    }

    figure.available = true;
    figure.value = worst;
    figure.source = source;
    figure.reason = "";

    return figure;
}

CombinationCoverage QueryCombination(Precision precision,
                                     FitRoute route,
                                     EvalScheme scheme,
                                     PackAxis axis,
                                     FitGranularity granularity,
                                     AccuracyTier tier,
                                     double tolerance) noexcept
{
    // Both figures come from the two accessors above: the tables, the axes and
    // the refusals are theirs.
    CombinationCoverage coverage;
    const AccuracyFigure guaranteed =
        BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity, tier);

    coverage.requested = tolerance;
    coverage.source = guaranteed.source;

    if (!guaranteed.available)
    {
        // A refusal carries no figure and no verdict; the reason is the
        // library's own sentence.
        coverage.reason = guaranteed.reason;

        return coverage;
    }

    const AccuracyFigure delivered =
        BoysAccuracyDelivered(precision, route, scheme, axis, granularity, tier);

    coverage.bound = guaranteed.value;
    coverage.delivered = delivered.available ? delivered.value : 0.0;
    coverage.deliveredKnown = delivered.available;

    // The bound decides first; only where it does not reach the request does the
    // measured figure decide. A request no figure is at or below - including one
    // that is not positive or not finite - is kOutside rather than refused: the
    // combination is carried, so there is a verdict, and the verdict is no.
    if (coverage.bound <= tolerance)
    {
        coverage.verdict = ToleranceVerdict::kGuaranteedInside;
    } else if (coverage.deliveredKnown && coverage.delivered <= tolerance)
    {
        coverage.verdict = ToleranceVerdict::kDeliveredInside;
        // The measurement is the figure that decided this answer, so the source
        // goes with it.
        coverage.source = delivered.source;
    } else
    {
        coverage.verdict = ToleranceVerdict::kOutside;
    }

    return coverage;
}

} // namespace boys

// The arithmetic backends this build carries, as a report prints them.
//
// This translation unit answers for the scalar pair, because this is where the
// scalar arithmetic is compiled: the same flags a consumer's own code gets, and
// not the packed flags src/boys_simd.cpp carries. The packed entries are
// appended by the unit that owns them.
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
