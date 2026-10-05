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
// The kernel bodies live in boys/boys_impl.hpp. This TU provides the default m = 1
// explicit instantiations that the extern-template declarations in boys.hpp route
// every default call site to; a call site naming any other multiplier compiles its
// rung from the shipped definition instead of linking here.

namespace boys {

template double BoysSingle<>(int n, double x) noexcept;
template void BoysAllOrders<>(int nmax, double x, double* out) noexcept;
template void BoysFixedN<>(
    int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept;
template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept;
template void BoysAllN<>(
    int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept;
template void BoysAllNAtOrders<>(
    const int* n, const double* x, double* out, std::size_t count) noexcept;
template float BoysSingleF32<EvalPolicy<>>(int n, float x) noexcept;
template void BoysAllOrdersF32<EvalPolicy<>>(
    int nmax, float x, float* out) noexcept;
template void BoysAllNF32<EvalPolicy<>>(
    int nmax, const float* x, float* out, std::size_t count) noexcept;

#if BoysFp16
template F16 BoysSingleF16<>(int n, F16 x) noexcept;
template void BoysAllOrdersF16<>(int nmax, F16 x, F16* out) noexcept;
template Bf16 BoysSingleBf16<>(int n, Bf16 x) noexcept;
template void BoysAllOrdersBf16<>(int nmax, Bf16 x, Bf16* out) noexcept;
#endif // BoysFp16

namespace {

// The class the run-time selectors stand in for: BoysAllOrders, the double lane's
// ladder entry, which is the entry the unnamed calls below are made through.
using SelectorClass = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>;

// The policy a run-time selector builds from the axes its caller named. The axes a
// selector does not take are read off the class's own default - the row this build's
// seam carries for BoysAllOrders, or the five where it carries none - and not spelled
// here as the five: a five spelled here agrees with the entry the selector stands in
// for only while the seam carries no row for the class, and a seam that carries one
// is a build whose selector would then answer a different arithmetic from the default
// entry it stands in for.
template <FitRoute kRoute, EvalScheme kScheme>
using SelectorPolicy = EvalPolicy<kRoute,
                                  kScheme,
                                  SelectorClass::kBudget,
                                  SelectorClass::kPack,
                                  SelectorClass::kGranularity,
                                  SelectorClass::kDivision,
                                  SelectorClass::kRegionBExp>;

} // namespace

std::span<const FitRouteInfo> BoysFitRoutes() noexcept {
    // The rows are the generated header's own measured figures rather than numbers
    // repeated here, so a table and the fits it describes cannot drift apart.
    //
    // The region-A rows state the domain of the region's per-order tables - every
    // order's own piece, from zero to kX0 - because that is what the two routes'
    // tables cover and what makes the two rows a comparison. kRatARouteLo is where
    // the rational route's selector takes over: below it the shipped lane is
    // documented at 1e-15 and the rational fits hold the wider bar.
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
    // The class's own scheme, not the reference one: the values a route does not
    // serve are the default entry's bit for bit, which requires the scheme that entry
    // reads, and the default entry reads its class's row. The scheme is therefore read
    // off the row (SelectorClass above) rather than off the seam's five: the two agree
    // only while the row carries the combination the five compose.
    BoysAllOrdersWithRoute(route, SelectorClass::kScheme, nmax, x, out);
}

void BoysAllOrdersWithRoute(
    FitRoute route, EvalScheme scheme, int nmax, double x, double* out) noexcept {
    // One body, instantiated once per (route, scheme) pair through the same entry a
    // compile-time caller uses: the zero argument's closed form, the region split, the
    // recurrences, the per-order rule and the domains are the body's, and the policy
    // names only the fits and the summation they are read in. Every argument the named
    // route's rows do not cover is answered by the body's own shipped-family branches,
    // which read no route fit at all, so those values are the default entry's bit for
    // bit. Every route this build does not serve, a value outside the enumeration
    // included, is the default entry outright.
    //
    // The route names the fits and the scheme the summation the shipped family's
    // coefficients are read in. The arm a route the enumeration does not name takes is
    // read off the class row rather than written as "the route that is not the rational
    // one": such a value evaluates at the default route, which is the one the entry
    // naming no route reads, and that is the class's.
    const bool served = route == FitRoute::kChebyshev || route == FitRoute::kRationalMinimax;
    const FitRoute selected = served ? route : SelectorClass::kRoute;

    if (selected == FitRoute::kRationalMinimax)
    {
        if (scheme == EvalScheme::kHorner)
        {
            BoysAllOrders<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kHorner>>(
                nmax, x, out);
            return;
        }

        BoysAllOrders<SelectorPolicy<FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw>>(nmax, x,
                                                                                        out);
        return;
    }

    if (scheme == EvalScheme::kHorner)
    {
        BoysAllOrders<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kHorner>>(nmax, x, out);
        return;
    }

    BoysAllOrders<SelectorPolicy<FitRoute::kChebyshev, EvalScheme::kSplitClenshaw>>(nmax, x, out);
}

// The evaluation-scheme report. Both tables are built once from the generated rows;
// the route in force is a build fact, so which of a row's two figures a query answers
// with is decided here.
constexpr double DeliveredIn(const EvalFitInfo& fit, backend::MulAddRoute route) noexcept {
    return route == backend::MulAddRoute::kSeparate ? fit.separate : fit.fused;
}

std::span<const FitRouteInfo> BoysFitRoutesF32() noexcept {
    // The float lane's rows, from the same generated header the double lane's rows
    // read. The lane's own tables are what the two region-A routes evaluate, and its
    // route covers the whole of region A, so the rows state no servesFrom boundary
    // above zero: naming either route changes a value at every argument of the
    // interval.
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
    // One body instantiated per route, as the double lane's selector is: the closed
    // form at zero, the region split, the two recurrences and the domains are the
    // body's, and the route names only its two fits. Region C, which reads no
    // coefficient at all, is answered by the body's own branch - the default entry's
    // code for those arguments.
    //
    // Every cell but the route is read off the row this entry's default arm runs -
    // BoysSingleF32's own default, the row this build's seam carries for the class, or
    // the five where it carries none - and the route is the one thing a call changes.
    // The partition is one of those cells, and that is why this is the row and not a
    // fit family named here: an arm naming its own family would answer from that row
    // only where the intended partition is absent, and would hand back another
    // partition's tables under the named member's name - the substitution
    // BoysSingleF32Impl refuses in as many words. Read off the row, the two arms and
    // the default entry are one body under three routes: BoysSingleF32 with one thing
    // changed.
    //
    // The scheme, the budget and the packing axis come off the row for the same
    // reason, and the packing axis is the arguments one by the row's own contract (a
    // single-order class has no second order to fill a lane with).
    using SingleOrderClass = DefaultPolicy<Precision::kFp32, Shape::kSingle>;

    using RowCells = EvalPolicy<FitRoute::kChebyshev, SingleOrderClass::kScheme,
                                SingleOrderClass::kBudget, SingleOrderClass::kPack,
                                SingleOrderClass::kGranularity, SingleOrderClass::kDivision,
                                SingleOrderClass::kRegionBExp>;

    if (route == FitRoute::kRationalMinimax)
    {
        using RationalClass = EvalPolicy<FitRoute::kRationalMinimax, SingleOrderClass::kScheme,
                                         SingleOrderClass::kBudget, SingleOrderClass::kPack,
                                         SingleOrderClass::kGranularity, SingleOrderClass::kDivision,
                                         SingleOrderClass::kRegionBExp>;

        return BoysSingleF32<RationalClass>(n, x);
    }

    return BoysSingleF32<RowCells>(n, x);
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
    case FitGranularity::kCoarsest:
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
    // Both members evaluate the region-A per-order fits of the double lane, so both
    // are certified against that region's bar. The packed lanes cover region A and
    // nothing else: past it the entries that carry an axis run the certified scalar
    // lanes, and the row's interval says where the packed answer ends.
    static const std::array<PackAxisInfo, 2> rows = {{
        {PackAxis::kArguments, PackAxisName(PackAxis::kArguments), 4, 0.0, detail::kX0, 1e-15},
        {PackAxis::kOrders, PackAxisName(PackAxis::kOrders), 4, 0.0, detail::kX0, 1e-15},
    }};

    return rows;
}

const char* RegionBExpName(RegionBExp exp) noexcept {
    switch (exp)
    {
    case RegionBExp::kAccurate:
        return "accurate";
    case RegionBExp::kFast:
        return "fast";
    }

    return "unknown";
}

std::span<const RegionBExpInfo> BoysRegionBExps() noexcept {
    // Every row is served by every entry that has a region-B ladder: the member is
    // read inside the ladder's own seed, so naming one selects arithmetic and never an
    // entry.
    static const std::array<RegionBExpInfo, 2> rows = {{
        {RegionBExp::kAccurate, RegionBExpName(RegionBExp::kAccurate)},
        {RegionBExp::kFast, RegionBExpName(RegionBExp::kFast)},
    }};

    return rows;
}

std::span<const DivisionFormInfo> BoysDivisionForms() noexcept {
    // Every row is served by every entry: the form is read inside the recurrence's own
    // step, so naming one selects arithmetic and never an entry.
    static const std::array<DivisionFormInfo, 3> rows = {{
        {DivisionForm::kExactDivision, DivisionFormName(DivisionForm::kExactDivision)},
        {DivisionForm::kPlainReciprocal, DivisionFormName(DivisionForm::kPlainReciprocal)},
        {DivisionForm::kRefinedReciprocal, DivisionFormName(DivisionForm::kRefinedReciprocal)},
    }};

    return rows;
}

std::span<const FitGranularityInfo> BoysFitGranularities() noexcept {
    // The rows are the tables' own counts and the certification's own figures, so a
    // regeneration that moved a piece, a degree or a delivered error moves the row.
    //
    // Every entry carries the double lane's tables; the single-precision lanes read
    // tables of their own, cut for their own arithmetic and budget, so a row here
    // describes the double lane and not them. The uniform row's table is the double
    // lane's own grid (detail::kFlatCoeffs); the single-precision engine's grid
    // (detail::f32::kFlatCoeffsF32) is the device lane's, and no entry of the host
    // single-precision lane enters a uniform branch, which is why that lane's carriage
    // refuses the partition where it is named. The shipped row carries every rung this
    // build can name - the count is the tier enumeration's own - both packing axes and
    // both fit routes; so does the narrow row, because each of the two axes reads a
    // cut of this partition's own pieces and the partition was cut for the rational
    // route as well, one numerator/denominator pair over each of its pieces in both
    // regions. The same axis on the float engine is instantiated at every rung, both
    // schemes and both routes, one entry per cell at each of the lane's two budgets
    // (boys_orders_simd.cpp), and the degree tables those entries read are cut from
    // that lane's own narrow pieces per role (boys_effective_degrees.hpp) - so a rung
    // of the narrow partition is the float lane's own table and not the double lane's
    // degrees under a float name.
    //
    // The uniform row states the one route and the one rung this build carries that
    // partition at, and each is a refusal the build makes where the call is named: the
    // rational route is refused at compile time because its fit is a numerator/
    // denominator pair per derived piece and the grid is fixed, and a relaxed rung is
    // refused at compile time because the table stores one degree for every order and
    // every interval. Both packing axes are carried: the across-orders lane has an
    // instantiation over this grid (boys_orders_simd.cpp), and the grid's interval-major
    // layout makes the stride its fetch steps at the same kind of stride the shipped
    // cover's is. Its region-A fields describe the grid rather than a cut of a region:
    // the grid covers [0, kFlatHi) whole, region A's and region B's arguments alike,
    // and its intervals are intervals rather than pieces. Its region-B fields are zero
    // because there is no second table here: above the grid's top edge the call reaches
    // the asymptotic every route ends in, which reads no fit at all.
    static const std::array<FitGranularityInfo, 3> rows = [] {
        static constexpr unsigned kBothAxes = (1u << static_cast<unsigned>(PackAxis::kArguments)) |
                                              (1u << static_cast<unsigned>(PackAxis::kOrders));
        static constexpr unsigned kChebBit = 1u << static_cast<unsigned>(FitRoute::kChebyshev);
        static constexpr unsigned kRatBit =
            1u << static_cast<unsigned>(FitRoute::kRationalMinimax);

        // The uniform table's certification rows describe the table this row's counts
        // are read from, so the two readings are tied here rather than left to drift. A
        // row's `deg` is the read cap and not a degree any one interval stores: the grid
        // carries a degree per interval (kFlatDegs) and the cap is what the summations
        // are written for, which boys_impl.hpp checks every interval against.
        static_assert(std::size(detail::kFlatRows) == 2,
                      "the uniform table is certified one row per evaluation scheme");
        static_assert(detail::kFlatRows[0].deg == detail::kFlatReadCap &&
                          detail::kFlatRows[1].deg == detail::kFlatReadCap &&
                          detail::kFlatRows[0].intervals == detail::kFlatIntervals &&
                          detail::kFlatRows[1].intervals == detail::kFlatIntervals &&
                          detail::kFlatRows[0].stored ==
                              static_cast<int>(std::size(detail::kFlatCoeffs)) &&
                          detail::kFlatRows[1].stored ==
                              static_cast<int>(std::size(detail::kFlatCoeffs)) &&
                          detail::kFlatOffsets[detail::kFlatIntervals] ==
                              static_cast<int>(std::size(detail::kFlatCoeffs)),
                      "the uniform table's certification rows must describe the table this "
                      "row's counts are read from");

        // A route's own figures over a region, read from the route table rather than
        // restated, so a regeneration moves both readings together.
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

        // The worst a partition delivers and the bar it is certified against, taken over
        // the routes it holds and the regions they cover: a caller who does not name a
        // route may read any of them.
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

        // The rational route is cut on the same pieces, so its degrees are read over the
        // same rows: numerator and denominator both bound what one evaluation of a piece
        // costs.
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

        // The interval a partition's own tables serve, read from the fitted routes' own
        // domains: region A's per-order tables from zero to kX0 and region B's seed from
        // kX0 to kX1, under either partition. Above the top edge the entry runs region
        // C's asymptotic form, which no partition replaces and whose figure is the
        // certified lane's, so the figure holds on this domain and no wider one.
        double fittedLo = std::numeric_limits<double>::infinity();
        double fittedHi = 0.0;

        for (const FitRouteInfo& row : BoysFitRoutes())
        {
            fittedLo = std::min(fittedLo, row.lo);
            fittedHi = std::max(fittedHi, row.hi);
        }

        // No fitted row at all leaves no interval to report, and one made up here would
        // be a claim about tables that are not there.
        if (!(fittedLo <= fittedHi))
        {
            fittedLo = 0.0;
            fittedHi = 0.0;
        }

        std::array<FitGranularityInfo, 3> built{};

        built[0].granularity = FitGranularity::kCoarsest;
        built[0].name = GranularityName(FitGranularity::kCoarsest);
        built[0].routes = kChebBit | kRatBit;
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
        built[1].axes = kBothAxes;
        built[1].regionAPieces = static_cast<int>(std::size(detail::kNarrowAPieces));
        built[1].regionADeg = narrowADeg;
        built[1].regionAStored = narrowAStored + detail::kNarrowRatAStored;
        built[1].regionBPieces = detail::kNarrowBPieces;
        built[1].regionBDeg = narrowBDeg;
        built[1].regionBStored =
            static_cast<int>(std::size(detail::kNarrowBcoeffs)) + detail::kNarrowRatBStored;
        // The narrow certification publishes a per-piece round-up of what each piece
        // delivers rather than a bar the pieces were cut at, and that round-up is the
        // figure a caller may rely on: it bounds a sweep and not only the one that
        // measured it. Both routes the row holds are read into it, for the reason
        // routeWorst gives above.
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
        // Both routes and one rung, and the two are read off different things: the rung
        // is a refusal the build makes where the call is named, and the routes are the two
        // members the partition stores. The table holds a Chebyshev fit per order per
        // interval and, beside it, one numerator/denominator pair per interval - the
        // rational family's answer over the grid's cells, which are intervals like any
        // other partition's pieces - so a cell naming either route is one this build
        // answers.
        //
        // Every rung, because the partition's table is admissible at every one of them:
        // the criterion that cuts a stored row to a rung's degree scans the dropped
        // coefficients' tail and takes the first degree whose tail fits the rung's budget,
        // and the full degree's tail is zero, so the scan always reaches it
        // (boys_effective_degrees.hpp). The one rung this row still has refused for it is
        // the rational member's, and that refusal is the carrier's below.
        //
        // Both packing axes, because both are instantiated over this grid. The grid is
        // interval-major - every order of one interval lies one stride from the next
        // order's - so the across-orders lane steps an order's coefficients to the next
        // order's at a stride, as it does on the shipped cover, and the per-argument lane
        // reads the same layout a whole ladder at a time. The rational member's rows have
        // no such stride - a pair is per interval - so its across-orders cell is served by
        // the certified scalar orders lane rather than by the packed body
        // (boys_orders_simd.cpp).
        built[2].routes = kChebBit | kRatBit;
        built[2].axes = kBothAxes;
        // The grid itself, which is what this partition is where the other two are walks:
        // the intervals are the grid's cells rather than pieces the proved bound put where
        // the function needs them, each interval carries its own degree and offset rather
        // than sharing one stride, and it stores the whole table - one fit per order per
        // interval.
        //
        // The degree the row reports is therefore the greatest of the intervals' own: an
        // evaluation reads the degree of the interval it landed in, and the highest of
        // those is the highest any evaluation reads. The read cap beside it is a ceiling on
        // that number and not the number.
        //
        // The rational member is read the same way and contributes the same field: its
        // pair's two degrees are what one evaluation of a row costs.
        constexpr int kUniformReadDeg = [] {
            int hi = 0;

            for (const int deg : detail::kFlatDegs)
            {
                hi = deg > hi ? deg : hi;
            }

            return hi;
        }();

        constexpr int kUniformRatReadDeg = [] {
            int hi = 0;

            for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatRatIntervals);
                 ++iv)
            {
                hi = std::max({hi, detail::kFlatRatNumDeg[iv], detail::kFlatRatDenDeg[iv]});
            }

            return hi;
        }();

        built[2].regionAPieces = detail::kFlatIntervals;
        built[2].regionADeg = std::max(kUniformReadDeg, kUniformRatReadDeg);
        built[2].regionAStored = static_cast<int>(std::size(detail::kFlatCoeffs)) +
                                 detail::kFlatRatStoredTotal;
        built[2].regionBPieces = 0;
        built[2].regionBDeg = 0;
        built[2].regionBStored = 0;
        // The worse of the partition's three members and of its two multiply-add routes,
        // which is the one figure that holds for a caller who names any of them: the
        // certification's own round-up rather than a number restated here.
        double uniformWorst = 0.0;

        for (const detail::FlatRow& row : detail::kFlatRows)
        {
            uniformWorst = std::max({uniformWorst, row.fused, row.separate});
        }

        for (const detail::FlatRatRow& row : detail::kFlatRatRows)
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
    // The figures the README's contract table publishes, stated here once so that the
    // table, the accuracy gate and BoysAccuracyGuaranteed read one number rather than
    // three transcriptions of one. Each is the figure the lane documents for one value
    // over the whole of x >= 0 at the reference multiplier; the half-precision lane's
    // f32 entries add a term of their own under their region-B exponential option,
    // which is why the row carries it.
    //
    // Where the division forms deliver different figures the row states the plain
    // form's apart, in the member the struct documents for it. How far that axis moves
    // the figures is measured rather than reasoned, and the measurement is the accuracy
    // gate's: it reads every cell of every lane's rows at each of the three forms and
    // judges each form against the figure the row publishes for it.
    //
    // The host fp32 and fp16 lanes' terms are the plain reciprocal's own rounding,
    // measured on the accuracy gate's reference grid at the reference multiplier:
    // that form's worst on the fp32 lane is 1.75140e-07 at n = 0, x = 9.74054909,
    // where the lane's other two forms deliver 1.08354e-07 at worst. The difference
    // is 6.68e-8, and the term published is 1e-7, which bounds it with the margin a
    // guarantee needs. The fp16-device lane's term is its own measurement and not
    // that one: there the plain reciprocal's worst is 1.53895485e-07 at n = 32,
    // x = 11.9453125, a cell at which the value is subnormal in the half format, and
    // the two forms that divide exactly deliver 1.78813934e-07 and 1.1920929e-07
    // there, so the plain form's extra rounding is 5.39e-8 over the base and 1e-7
    // bounds it.
    static const std::array<LaneContractInfo, 7> rows = {{
        {Precision::kFp64, "fp64", 5.5e-14, 0.0, 0.0, RegionBExp::kFast,
         "throughout, every region"},
        {Precision::kFp32, "fp32", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
         "every region, at exact division and the refined reciprocal",
         "every region under the plain reciprocal, whose extra rounding adds that form's own term "
         "beside the base"},
        {Precision::kFp16, "fp16", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
         "the single-precision lane's own figure, plus half of the last representable digit of the "
         "returned value and claimed only where the value exceeds the sum. The half lane computes "
         "in that arithmetic and stores what it returns, so it cannot be more accurate than the "
         "lane whose arithmetic it runs: a bar below that figure is one no conforming host can "
         "keep, and a host whose rounding differs delivers the fit's own error through it. Beside "
         "the base, and under the same division form: the fp16 and bf16 entries run the "
         "single-precision engine's own bodies and round at the boundary (BoysAllOrdersF16, "
         "boys_impl.hpp), so the plain reciprocal's larger figure on that lane - 1.75140e-07 at "
         "n = 0, x = 9.74054909, 6.67e-8 above that lane's own worst of 1.08354e-07 - is this "
         "lane's too, before the format's own half digit is added to it"},
        {Precision::kFp32Device, "fp32-device", 1.5e-7, 8e-8, 0.0, RegionBExp::kFast,
         "plus 8e-8 under the fast region-B exponential, which is the corrected seed's own "
         "contribution. The device lane carries the division axis - the form is a trailing "
         "parameter of every launched entry, and DeviceOptionAxis::kDivision crosses each entry "
         "of the device option space with all three (boys/boys_cuda_options.hpp) - and the "
         "accuracy gate reads every cell of this lane's rows at each of the three forms, "
         "judging each against the figure this row publishes for it. The three forms share that "
         "figure: over the lane's whole grid the plain reciprocal delivers 1.48716e-07 at "
         "worst and the two forms that divide exactly deliver 1.28793e-07, and both are inside "
         "the 2.3e-07 the row states. The axis is live on this lane rather than three names for "
         "one body: the plain reciprocal's values differ from the exact form's in 194120 of the "
         "cells on the arguments axis and 133158 on the orders axis, while the refined "
         "reciprocal's differ in none, which is the bit-identity the library documents of that "
         "form. So the row states one figure for all three forms and no plain term beside it"},
        // The device's double lane carries the axis like every other device lane, and
        // states one figure for it: the double lane's own, because its entries read the
        // double lane's piece tables and its region-B seed and the bound the CUDA surface
        // states for them is 5.5e-14 (boys/boys_cuda.hpp, SingleF64 and AllOrdersF64,
        // each stating "|error| <= 5.5e-14"). It is stated apart from the host row
        // because it is a different lane, and the figure a consumer holding a
        // `kFp64Device` class reads must be this lane's own statement.
        {Precision::kFp64Device, "fp64-device", 5.5e-14, 0.0, 0.0, RegionBExp::kFast,
         "the double lane's figure over the whole of x >= 0 on this lane's own entries, which "
         "publish it as \"|error| <= 5.5e-14\" (boys/boys_cuda.hpp, SingleF64 and AllOrdersF64). "
         "It is the host double lane's number because it is the same arithmetic over the same "
         "pieces; it is this row's because the lane is this lane's - a device class asked for "
         "its bound is answered here and not by the host row beside this one. The lane carries "
         "the division axis like every device lane, and the accuracy gate reads every cell of "
         "its rows at each of the three forms: all three deliver 5e-14 at worst, which is "
         "inside the 5.5e-14 above, so the row states one figure for the axis and no plain "
         "term beside it. The forms are not one body under three names there either - the "
         "plain reciprocal's values differ from the exact form's in 302586 cells on the "
         "arguments axis and 250542 on the orders axis - but the difference stays inside the "
         "figure the row publishes, and the refined reciprocal's values differ in none",
         ""},
        // The device's half lane computes in the float lane's bodies and stores what it
        // returns, so the base is the half lane's and not the float lane's: 1e-7 is the
        // figure the CUDA surface states for this lane (boys/boys_cuda.hpp, SingleF16 and
        // AllOrdersF16), with the value-dependent term kept in the sentence below rather
        // than folded into the number, which is how the device option table states the
        // same form (src/boys_cuda.cpp, kFormF16 against kBoundF16). The plain term
        // beside the base is this lane's own measurement and not the fp32 lane's: the
        // gate found the plain reciprocal delivering zero where the base is 1e-7 and the
        // other two forms were inside it.
        {Precision::kFp16Device, "fp16-device", 1e-7, 0.0, 1e-7, RegionBExp::kFast,
         "plus half of the last representable digit of the returned value, which is a term of "
         "the value the caller receives and not of the call: the lane computes in the float "
         "lane's bodies and stores half, so it cannot be more accurate than the format it stores "
         "in, and where the value falls at or below the format's floor no accuracy is claimed at "
         "all. The base is the figure the device lane's own half entries publish "
         "(boys/boys_cuda.hpp, SingleF16 and AllOrdersF16). The lane carries the division axis "
         "like the other device lanes, and the accuracy gate reads every cell of its rows at "
         "each of the three forms, judging each against the figure this row states for it. Under "
         "the two forms that divide exactly the lane delivers 2.43800e-04 at worst over the "
         "grid, which is the half digit of the sentence above and not a term of the form; under "
         "the plain reciprocal the values differ from the exact form's in 98 cells on the "
         "arguments axis and 82 on the orders axis, and the difference is that form's own "
         "rounding at a result the half format stores as a subnormal: at n = 32, x = 11.9453125 "
         "the value is 1.53895485e-07, the two forms that divide exactly deliver 1.78813934e-07 "
         "and 1.1920929e-07 there, and the plain reciprocal delivers zero. That is 1.53895485e-07 "
         "from the value and outside the 1e-07 base, so this row states the plain form's figure "
         "apart, in the term the struct documents for it and beside the same half-digit term",
         "every region under the plain reciprocal, which at a subnormal result adds its own "
         "rounding beside this lane's base: the accuracy gate measured that form's worst at "
         "1.53895485e-07, at n = 32, x = 11.9453125, where the two forms that divide exactly are "
         "inside the base at 2.5e-8 and 3.5e-8 from the value. The difference over the base is "
         "5.39e-8, and the term published is 1e-7, which bounds it with the margin a guarantee "
         "needs"},
        // This lane runs the fp16 lane's bodies at the fp16 lane's budget and differs from
        // it in the format it stores, so the base and the plain form's term are that
        // lane's and the digit the sentence carries is this format's own. The row stands
        // here rather than folded into the fp16 one because the seam keys a class by the
        // format a return carries (boys/boys_build_defaults.hpp writes each class's row).
        {Precision::kBf16, "bf16", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
         "the single-precision lane's own figure, plus half of the last representable digit of "
         "the returned value, which in this format is 2^-8 = 3.90625e-03, and claimed only where "
         "the value exceeds the sum. The half lane computes in that arithmetic and stores what it "
         "returns, so it cannot be more accurate than the lane whose arithmetic it runs: a bar "
         "below that figure is one no conforming host can keep, and a host whose rounding differs "
         "delivers the fit's own error through it. It is the fp16 lane's figure on this format's "
         "store and not the fp16 class's row: the two are two classes of one lane, each stating "
         "its own format's half digit, and this format's is the coarser of the two - 2^-8 against "
         "2^-11. The fp16 row beside this one states that term in words; this row names its "
         "number. "
         "Beside the base, and under the same division form: the fp16 and bf16 entries run the "
         "single-precision engine's own bodies and round at the boundary (BoysAllOrdersBf16, "
         "boys_impl.hpp), so the plain reciprocal's larger figure on that lane - 1.75140e-07 at "
         "n = 0, x = 9.74054909, 6.67e-8 above that lane's own worst of 1.08354e-07 - is this "
         "lane's too, before the format's own half digit is added to it"},
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
// Both packing axes are served on the shipped partition: the arguments axis by the
// per-order bodies, the orders axis by the packed lane that steps one order's
// coefficients to the next order's. The narrow partition is stored for this lane and
// is read at the reference multiplier on both axes; a rung of it is a cut of the
// narrow pieces' own degree table, which this lane derives, so a narrow rung is
// served on both axes and on every route and scheme.
//
// The uniform partition is not served on this lane at all: no entry of it reads a
// grid. The grid this lane's arithmetic was fitted over is stored - it is the table
// the device lane's fp32 entries read - and the entries this lane publishes refuse a
// policy naming the partition where the call is named rather than answering it from
// another partition's fits. The partition is a row of the host table because the
// double lane's entries do read one, and a row in that table is not a claim about
// this lane.
Carriage CarriesSingle(FitRoute route,
                       EvalScheme scheme,
                       PackAxis axis,
                       FitGranularity granularity) noexcept {
    // Every combination this library names is served on both axes and on the three
    // partitions this lane reads, at every rung, with no exception. A cut has to be read
    // against the table the scheme actually sums: the Chebyshev cut under the Horner
    // scheme's monomial table delivers outside the bound on 1964 of 56694 cells at
    // m = 64. With the basis carried through, the same combination is a rung of the
    // family the caller named on either of the two piecewise partitions, so there is no
    // cell of those left for this rule to refuse. The uniform grid's own table is stored
    // at a degree per interval and every rung of it is the route's own arithmetic rather
    // than a cut of it - its multiplier is not read - so its Chebyshev member is served
    // at every rung too, and the rational member is fitted over this lane's own grid and
    // emitted beside the Chebyshev one (kFlatRatCoeffsF32,
    // tools/gen_boys_coefficients.py), read through the route dispatch in
    // UniformOrderAtF32 below the grid's join on the arguments and orders axes alike.
    //
    // What it refuses is a value outside the enumerations, which names no combination at
    // all rather than one this revision does not carry.
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size())
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis and a partition from "
                "the enumerations this revision publishes"};
    }

    return {true, ""};
}

// The device lane's rule. Its entries are compiled at the one accuracy this library
// carries, so what this rule refuses is a value outside the enumerations this revision
// publishes rather than a rung it did not instantiate.
//
// This reads the lane's own entries, not the double lane's beside it:
// Precision::kFp32Device is the device lane's single-precision one, and every entry of
// it is the float engine's arithmetic.
//
// What the lane carries, route by route and partition by partition, read off the fp32
// entries the lane itself builds:
//
//   * the shipped partition carries both routes: its Chebyshev fits are the float
//     lane's own table (BoysCuda::AllOrdersF32, with SingleF32 beside them), and its
//     rational member is a pair per piece of the same partition (AllOrdersF32Rat and
//     its Horner name) - this lane's own fit in region B and the double lane's in
//     region A;
//   * the narrow partition carries both routes the same way (AllOrdersF32Narrow with
//     its monomial name, AllOrdersF32NarrowRat with its Horner name);
//   * the uniform partition carries the Chebyshev member (AllOrdersF32Uniform and
//     AllOrdersF32UniformHorner) and its rational member (AllOrdersF32UniformRat and
//     its Horner name), the grid's fits of both routes being rows of this lane.
//
// Both packing axes are carried on the same tables: the per-argument ladder every one
// of those entries has, and the orders reading of the same stored fits - one fit per
// order in region A, where the ladder seeds the top order's fit and brings the lower
// orders back down a recurrence (BoysCuda::AllOrdersF32Orders and its siblings, one
// pair per partition and route, mirroring the double lane's). Neither axis reaches a
// table the other does not.
//
// So the accuracy refuses no cell of this lane, and no arm below tests one: a lane that
// answers at m = 1 alone has no rung to refuse. Which scheme reaches which partition is
// the lane's own table's to state.
Carriage CarriesDevice(FitRoute route,
                       EvalScheme scheme,
                       PackAxis axis,
                       FitGranularity granularity) noexcept {
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size())
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis and a partition from "
                "the enumerations this revision publishes"};
    }

    // The uniform grid's rational member is a row of this lane now, on both precisions:
    // the member is fitted over each lane's own grid, emitted beside the Chebyshev one
    // (tools/gen_boys_coefficients.py), uploaded to the device face and read by entries
    // of its own (BoysDeviceAllOrdersF64UniformRat and its float counterpart, the
    // kernels behind BoysCuda::AllOrdersF64UniformRat and its float counterpart). So no
    // arm here refuses it.

    // No arm below tests an accuracy, and none can: this lane answers at m = 1 alone,
    // which is the one accuracy each of its entries is built at.

    return {true, ""};
}

// The device's double lane, whose rule this is. It is a device lane and not the host
// double lane under another name: its entries are the CUDA surface's own
// (BoysCuda::AllOrdersF64 and the family beside it), a host without a CUDA device
// cannot run one of them, and the figure it publishes is stated in its own row of
// BoysLaneContracts.
//
// What this lane carries, read off the fp64 rows of the option table (src/boys_cuda.cpp),
// which are the entries the lane itself builds:
//
//   * both routes over the coarsest partition, on both packing axes and under both
//     scheme names (all-orders-fp64 with all-orders-fp64-orders, the mono pair beside
//     them, and the rational pairs all-orders-fp64-rat with -rat-horner);
//   * the narrow partition the same way (all-orders-fp64-narrow with its orders twin
//     and their monomial and rational names);
//   * the uniform grid the same way (all-orders-fp64-uniform and
//     all-orders-fp64-uniform-horner with their orders twins, and the grid's rational
//     pairs).
//
// Every cell the cross of that table names is therefore a row of the lane, and the arms
// below refuse a value outside the enumerations rather than a cell the lane does not
// build.
//
// The lane names no division form and answers at m = 1 alone, for the reason the fp32
// device lane's rule gives: the rung is not a build's to choose, and the form is a field
// of the host policy, which no CUDA surface header names.
Carriage CarriesDeviceF64(FitRoute route,
                          EvalScheme scheme,
                          PackAxis axis,
                          FitGranularity granularity) noexcept {
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size())
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis and a partition from "
                "the enumerations this revision publishes"};
    }

    return {true, ""};
}

// The device's half lane, whose rule this is. It is a device lane and not the host
// half lane under another name: its entries are the CUDA surface's own
// (BoysCuda::AllOrdersF16 and the family beside it), a host without a CUDA device
// cannot run one of them, and the figure it publishes is stated in its own row of
// BoysLaneContracts, which carries the half format's term.
//
// What this lane carries, read off the fp16 rows of the option table (src/boys_cuda.cpp),
// which are the entries the lane itself builds:
//
//   * both routes over the coarsest partition, on both packing axes and under both
//     scheme names (all-orders-fp16 with all-orders-fp16-orders, and the rational pairs
//     all-orders-fp16-rat with -rat-horner);
//   * the narrow partition the same way (all-orders-fp16-narrow with its orders twin
//     and their monomial and rational names);
//   * the uniform grid the same way (all-orders-fp16-uniform and
//     all-orders-fp16-uniform-horner with their orders twins, and the grid's rational
//     pairs).
//
// Every cell the cross of that table names is therefore a row of the lane, and the arms
// below refuse a value outside the enumerations rather than a cell the lane does not
// build.
//
// The lane names no division form and answers at m = 1 alone, for the reason the fp32
// device lane's rule gives.
Carriage CarriesDeviceF16(FitRoute route,
                          EvalScheme scheme,
                          PackAxis axis,
                          FitGranularity granularity) noexcept {
    const std::size_t r = static_cast<std::size_t>(route);
    const std::size_t s = static_cast<std::size_t>(scheme);
    const std::size_t a = static_cast<std::size_t>(axis);
    const std::size_t g = static_cast<std::size_t>(granularity);
    if (r >= BoysFitRoutes().size() || s >= BoysEvalSchemes().size() ||
        a >= BoysPackAxes().size() || g >= BoysFitGranularities().size())
    {
        return {false,
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis and a partition from "
                "the enumerations this revision publishes"};
    }

    return {true, ""};
}

} // namespace

AccuracyFigure BoysAccuracyGuaranteed(Precision precision,
                                      FitRoute route,
                                      EvalScheme scheme,
                                      PackAxis axis,
                                      FitGranularity granularity,
                                      DivisionForm form,
                                      RegionBExp exp) noexcept
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
            return CarriesDevice(route, scheme, axis, granularity);
        case Precision::kFp64Device:
            return CarriesDeviceF64(route, scheme, axis, granularity);
        case Precision::kFp16Device:
            return CarriesDeviceF16(route, scheme, axis, granularity);
        case Precision::kFp32:
        case Precision::kFp16:
        case Precision::kBf16:
            return CarriesSingle(route, scheme, axis, granularity);
        case Precision::kFp64:
            break;
        }

        const std::span<const FitGranularityInfo> partitions = BoysFitGranularities();
        const std::size_t p = static_cast<std::size_t>(granularity);
        Carriage c;

        // The enumeration guard the two carriers above make, and it is made here rather
        // than left to the row tables below. A partition's bitmask answers an
        // out-of-enumeration route or axis with a refusal about the partition and answers
        // an out-of-enumeration *scheme* with nothing at all: no row of a partition
        // repeats the scheme axis, so there is no bit to be clear and the combination was
        // read as carried.
        if (static_cast<std::size_t>(route) >= BoysFitRoutes().size() ||
            static_cast<std::size_t>(scheme) >= BoysEvalSchemes().size() ||
            static_cast<std::size_t>(axis) >= BoysPackAxes().size() ||
            p >= partitions.size())
        {
            c.reason =
                "the value named is outside the enumeration this library serves, so it names no "
                "combination: name a route, a scheme, a packing axis and a partition from "
                "the enumerations this revision publishes";

            return c;
        }

        const FitGranularityInfo& row = partitions[p];

        if (!FitGranularityHasRoute(row, route))
        {
            c.reason =
                "the partition's tables do not hold this fit route: the row states which "
                "routes this build stores for that partition, and this call names one the "
                "row leaves clear. A member the row does not carry is one to fit rather "
                "than a shape the call cannot have";
        } else if (!FitGranularityHasAxis(row, axis))
        {
            // No row this build writes leaves an axis clear: each of the three partitions
            // carries an instantiation of both lanes, the uniform grid's across-orders body
            // included, at both schemes and at the reference rung (boys_orders_simd.cpp). So
            // this branch is the reading of the field rather than a refusal a caller reaches
            // on this revision, and it is kept as that reading - a row added with an axis
            // left clear is refused here rather than answered by another partition's
            // tables.
            c.reason =
                "the partition's row carries no kernel for this packing axis: the row states "
                "which axes this build is instantiated over for that partition, and this call "
                "names one the row leaves clear. A kernel the row does not carry is one to write "
                "rather than a shape the call cannot have";
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

    // The plain reciprocal rounds once more per step than the other two forms, so on a
    // lane where that costs accuracy the row publishes a term beside its base and this
    // form's figure is the base plus it. Every other lane carries 0.0 here and the figure
    // is the base at every form.
    const double formTerm =
        form == DivisionForm::kPlainReciprocal ? lane.plainAdditive : 0.0;

    // Which exponential seeds a region-B ladder is an axis and not a spelling of the
    // combination (accuracy.hpp: naming the other member is naming another arithmetic). A
    // row states the member its term beside the base is under - the fp32-device row's
    // 8e-8 is the fast member's own contribution, certified there - so a call naming the
    // other member is not owed that term. On a row whose term is 0.0 the two members
    // answer the same figure: below the member's cut those are one evaluation bit for
    // bit, and above it the term the narrowed member adds is a fraction of a rounding the
    // published figure has room for (accuracy.hpp, kDefaultHostRegionBExp).
    const double memberTerm = exp == lane.additiveMember ? lane.additive : 0.0;

    figure.available = true;
    figure.value = lane.bound + formTerm + memberTerm;

    // The sentence moves with the figure. `source` names the forms the base is for, so on
    // a lane whose plain form carries its own term the figure above is not the one that
    // sentence describes, and the row states the plain form's sentence separately. A lane
    // whose source already covers every form states none.
    figure.source =
        (formTerm == 0.0 || lane.plainSource[0] == '\0') ? lane.source : lane.plainSource;

    return figure;
}

AccuracyFigure BoysAccuracyDelivered(Precision precision,
                                     FitRoute route,
                                     EvalScheme scheme,
                                     PackAxis axis,
                                     FitGranularity granularity) noexcept
{
    // A combination this build refuses has no delivered figure for the same reason it has
    // no bound.
    AccuracyFigure figure = BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity);

    figure.reading = AccuracyReading::kDelivered;
    figure.value = 0.0;

    if (!figure.available)
    {
        return figure;
    }

    // Every half lane answers this way: the host's two formats and the device's. Each
    // computes in the float lane's bodies and stores half, so what the caller receives
    // carries the format quantum a half-typed return does - a different quantum of the
    // value in each of the two host formats, which is why the sentence below names the
    // lanes and not one format.
    if (precision == Precision::kFp16 || precision == Precision::kFp16Device ||
        precision == Precision::kBf16)
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

    // The worst figure over the rows the combination names: a composition is at least as
    // bad as its worst part.
    double worst = 0.0;
    const char* source = "BoysFitRoutesF32()";

    // The half and single lanes' own route table; the double lane's is the one the device
    // lane's single-precision entries read their region-A fits from. Both half formats
    // read the single-precision table: they run that lane's bodies.
    const std::span<const FitRouteInfo> routes =
        (precision == Precision::kFp32 || precision == Precision::kFp16 ||
         precision == Precision::kBf16)
            ? BoysFitRoutesF32()
            : BoysFitRoutes();

    if (precision != Precision::kFp32 && precision != Precision::kFp16 &&
        precision != Precision::kBf16)
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

    // The device's double lane reads the double lane's pieces and its partitions are the
    // same threesome cut on those pieces, so the rows this figure is taken over are the
    // double lane's. It is a mirror of that lane's reading and not a second measurement,
    // which is what this accessor is: a figure a combination was measured to deliver.
    if (precision == Precision::kFp64 || precision == Precision::kFp64Device)
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
                                     double tolerance) noexcept
{
    // Both figures come from the two accessors above: the tables, the axes and the
    // refusals are theirs.
    CombinationCoverage coverage;
    const AccuracyFigure guaranteed =
        BoysAccuracyGuaranteed(precision, route, scheme, axis, granularity);

    coverage.requested = tolerance;
    coverage.source = guaranteed.source;

    if (!guaranteed.available)
    {
        // A refusal carries no figure and no verdict; the reason is the library's own
        // sentence.
        coverage.reason = guaranteed.reason;

        return coverage;
    }

    const AccuracyFigure delivered =
        BoysAccuracyDelivered(precision, route, scheme, axis, granularity);

    coverage.bound = guaranteed.value;
    coverage.delivered = delivered.available ? delivered.value : 0.0;
    coverage.deliveredKnown = delivered.available;

    // The bound decides first; only where it does not reach the request does the measured
    // figure decide. A request no figure is at or below - including one that is not
    // positive or not finite - is kOutside rather than refused: the combination is
    // carried, so there is a verdict, and the verdict is no.
    if (coverage.bound <= tolerance)
    {
        coverage.verdict = ToleranceVerdict::kGuaranteedInside;
    } else if (coverage.deliveredKnown && coverage.delivered <= tolerance)
    {
        coverage.verdict = ToleranceVerdict::kDeliveredInside;
        // The measurement is the figure that decided this answer, so the source goes with
        // it.
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
// This translation unit answers for the scalar pair, because this is where the scalar
// arithmetic is compiled: the same flags a consumer's own code gets, and not the packed
// flags src/boys_simd.cpp carries. The packed entries are appended by the unit that
// owns them.
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
