// Route liveness: a lane built to read BOYS_MULADD_SEPARATE must deliver that route's values, not
// report it. One that ignored the build fact (boys/backend.hpp: kSelectedRoute) met every bound and
// stayed green: reverting the across-orders lane left its suites at [PASSED] 19 tests, exit 0.
// Values are held against the certified summations at a forced route; sweeps stop below kX0.

#include <gtest/gtest.h>

#include "boys/backend.hpp"
#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"
#include "boys_orders_simd.hpp"
#include "boys_muladd_route_reference.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace {

namespace bdetail = boys::backend::detail;
namespace detail = boys::detail;

using boys::backend::MulAddRoute;
using boys::backend::MulAddRouteName;
using boys::backend::Scalar;

// The reference arithmetic at a named route, shared with the packed backends'
// own test unit so that the two hold lanes to one arithmetic.
using route_reference::RouteForced;
using route_reference::RouteStep;
using route_reference::StepProbes;
using route_reference::TwoRoundingSub;

// --- The report the counted rows are held to --------------------------------

/// The library's own statement of an arithmetic: its name, whether a bare
/// product-plus-add contracts here, and the route in force for it
/// (boys/backend.hpp: BackendInfo, from src/boys.cpp: BoysBackends).
const boys::backend::BackendInfo* ReportedEntry(const char* name) noexcept {
    for (const boys::backend::BackendInfo& info : boys::backend::BoysBackends())
    {
        if (std::strcmp(info.name, name) == 0)
        {
            return &info;
        }
    }

    return nullptr;
}

// --- The counted row ---------------------------------------------------------

/// What one sweep measured. The counts are what the assertions read, and they
/// are printed so the numbers a build produced are in the log beside the
/// verdict.
struct RouteRow {
    const char* family = "";
    const char* scheme = "";
    MulAddRoute reported = MulAddRoute::kFused;
    bool contracts = false;
    std::size_t cells = 0;
    std::size_t routes_differ = 0;
    std::size_t off_route = 0;
    std::size_t on_the_other_route = 0;
    bool first_off_route = false;
    int first_order = 0;
    double first_x = 0.0;
    double first_delivered = 0.0;
    double first_expected = 0.0;

    /// One cell: the value the lane delivered, and the two routes' reference
    /// values at the same piece and argument, all compared as bit patterns. A
    /// float widened to a double stays exact, so one comparison serves both
    /// precisions.
    void Add(double delivered, double at_fused, double at_separate, int order, double x) noexcept {
        const bool differ = std::memcmp(&at_fused, &at_separate, sizeof(double)) != 0;
        const double expected = reported == MulAddRoute::kFused ? at_fused : at_separate;
        const bool off = std::memcmp(&delivered, &expected, sizeof(double)) != 0;

        ++cells;
        if (differ)
        {
            ++routes_differ;
        }

        if (off)
        {
            ++off_route;

            if (!first_off_route)
            {
                first_off_route = true;
                first_order = order;
                first_x = x;
                first_delivered = delivered;
                first_expected = expected;
            }
        }

        const double other = expected == at_fused ? at_separate : at_fused;
        if (differ && std::memcmp(&delivered, &other, sizeof(double)) == 0)
        {
            ++on_the_other_route;
        }
    }
};

void Print(const RouteRow& row) {
    std::printf("  route-liveness family=%s scheme=%s reported=%s contracts=%d cells=%zu "
                "routes-differ=%zu delivered-off-route=%zu delivered-on-the-other-route=%zu\n",
                row.family,
                row.scheme,
                MulAddRouteName(row.reported),
                row.contracts ? 1 : 0,
                row.cells,
                row.routes_differ,
                row.off_route,
                row.on_the_other_route);

    if (row.first_off_route)
    {
        std::printf("    first off-route cell: order = %d, x = %.17g, delivered = %.17g, the "
                    "reported route's value = %.17g\n",
                    row.first_order,
                    row.first_x,
                    row.first_delivered,
                    row.first_expected);
    }

    std::fflush(stdout);
}

/// The row's verdict, and the bar the sweep itself has to clear: where this
/// unit measures that the build keeps the two routes apart, some cell of the
/// sweep must tell them apart, or the sweep proves nothing.
void Hold(const RouteRow& row) {
    Print(row);

    EXPECT_EQ(row.off_route, std::size_t{0})
        << "family " << row.family << ", scheme " << row.scheme << ": " << row.off_route << " of "
        << row.cells
        << " values are not the route the library reports for this arithmetic - the lane is not "
           "running the multiply-add route it says it runs";

    EXPECT_EQ(row.on_the_other_route, std::size_t{0})
        << "family " << row.family << ", scheme " << row.scheme << ": " << row.on_the_other_route
        << " values are the OTHER route's at cells where the two routes differ";

    if (!row.contracts)
    {
        EXPECT_GT(row.routes_differ, std::size_t{0})
            << "family " << row.family << ", scheme " << row.scheme
            << ": this unit does not contract a bare product-plus-add, so the two routes are two "
               "arithmetics here, and no cell of this sweep told them apart: the sweep cannot see "
               "the difference it exists to see, and the counts above prove nothing";
    }
}

// --- Probe values ------------------------------------------------------------

/// Arguments the lanes' region-A bodies are the ones that answer, and nothing
/// else: strictly below the interval's own end (kX0, where every entry hands a
/// high argument to the region-B and asymptotic lanes and to the certified
/// scalar lane), on either side of the piece edge the shipped double table
/// cuts region A at (5.9499240760542422), and near the end from below. Twenty
/// of them, so the across-arguments sweep's four-wide truncation covers the
/// whole set.
std::vector<double> RegionASweep() {
    const double edge = detail::kPieces[1].a;
    std::vector<double> xs;

    for (const double base : {1e-7, 1e-4, 0.01, 0.2, 1.0, 2.5, 4.0, 5.0, 6.0, 7.5, 9.0, 10.5})
    {
        xs.push_back(base);
    }

    for (const double delta : {-1e-12, -1e-7, -1e-3, 0.0, 1e-3, 1e-7, 1e-12})
    {
        xs.push_back(edge + delta);
    }

    // Below kX0 in this unit's arithmetic and in the single-precision lane's,
    // which casts the same argument to float before it asks: 1e-5 of relative
    // room is four orders of magnitude above that lane's ulp at this argument,
    // so both read it inside the region-A body.
    xs.push_back(detail::kX0 * (1.0 - 1e-5));

    return xs;
}

// --- The scalar lane ---------------------------------------------------------

template <typename T>
void HoldScalarStep(const char* name) {
    const boys::backend::BackendInfo* reported = ReportedEntry(name);
    ASSERT_NE(reported, nullptr) << "the library's report names no arithmetic called " << name;

    RouteRow row;
    row.family = name;
    row.scheme = "step";
    row.reported = reported->route;
    row.contracts = bdetail::MeasureContraction<T>();

    RouteRow sub_row = row;
    sub_row.scheme = "step-sub";

    const std::vector<std::array<T, 3>> probes = StepProbes<T>();

    for (const std::array<T, 3>& probe : probes)
    {
        volatile const T va = probe[0];
        volatile const T vb = probe[1];
        volatile const T vc = probe[2];
        const T a = va;
        const T b = vb;
        const T c = vc;

        row.Add(static_cast<double>(Scalar<T>::MulAdd(a, b, c)),
                static_cast<double>(RouteForced<MulAddRoute::kFused, T>::MulAdd(a, b, c)),
                static_cast<double>(RouteForced<MulAddRoute::kSeparate, T>::MulAdd(a, b, c)),
                0,
                0.0);

        // What the sweep tells apart at the multiply-subtract: the fused step, one instruction, one
        // rounding, from the two-rounding difference the operation's contract names. A lane that
        // let the route reach it would deliver the first where they differ; the count is the bar.
        const T sub_delivered = Scalar<T>::MulSub(a, b, c);
        const T sub_expected = TwoRoundingSub(a, b, c);
        const T sub_fused = std::fma(a, b, -c);

        ++sub_row.cells;
        if (std::memcmp(&sub_fused, &sub_expected, sizeof(T)) != 0)
        {
            ++sub_row.routes_differ;
        }

        if (std::memcmp(&sub_delivered, &sub_expected, sizeof(T)) != 0)
        {
            ++sub_row.off_route;

            if (std::memcmp(&sub_delivered, &sub_fused, sizeof(T)) == 0)
            {
                ++sub_row.on_the_other_route;
            }
        }
    }

    Hold(row);

    // The route does not reach the multiply-subtract: its contract is two
    // roundings on EVERY build (boys/backend.hpp: Scalar<T>::MulSub), and a
    // change that let the route reach it is a change to a certified contract.
    Print(sub_row);
    EXPECT_EQ(sub_row.off_route, std::size_t{0})
        << name << ": the multiply-subtract's contract is two roundings on every build, and "
        << sub_row.off_route << " of " << sub_row.cells << " probes are not that value";
    EXPECT_EQ(sub_row.on_the_other_route, std::size_t{0})
        << name << ": " << sub_row.on_the_other_route
        << " probes are the fused step's value, which is one rounding - the route has been let "
           "into an operation whose contract is two";
    EXPECT_GT(sub_row.routes_differ, std::size_t{0})
        << name << ": no probe of this sweep is a cell where the fused step and the two-rounding "
           "difference part, so the sweep cannot see the difference it exists to see";
}

TEST(BoysMulAddRouteScalar, UnitContractionMeasurementAgreesWithTheReport) {
    const boys::backend::BackendInfo* fp64 = ReportedEntry("scalar-fp64");
    const boys::backend::BackendInfo* fp32 = ReportedEntry("scalar-fp32");
    ASSERT_NE(fp64, nullptr);
    ASSERT_NE(fp32, nullptr);

    // The report's contraction answer is taken in the unit the scalar arithmetic is compiled in
    // (src/boys.cpp), and this unit carries the same flags; a measured answer that disagrees with
    // this unit's measurement means the claims and the arithmetic are not the same statement.
    std::printf("  route-liveness family=scalar report-contracts fp64=%d fp32=%d, this unit "
                "measures fp64=%d fp32=%d, route fp64=%s fp32=%s, build selected=%s\n",
                fp64->contracts ? 1 : 0,
                fp32->contracts ? 1 : 0,
                bdetail::MeasureContraction<double>() ? 1 : 0,
                bdetail::MeasureContraction<float>() ? 1 : 0,
                MulAddRouteName(fp64->route),
                MulAddRouteName(fp32->route),
                MulAddRouteName(bdetail::kSelectedRoute));
    std::fflush(stdout);

    EXPECT_EQ(bdetail::MeasureContraction<double>(), fp64->contracts);
    EXPECT_EQ(bdetail::MeasureContraction<float>(), fp32->contracts);
    EXPECT_EQ(bdetail::RouteInForce<double>(), fp64->route);
    EXPECT_EQ(bdetail::RouteInForce<float>(), fp32->route);
}

TEST(BoysMulAddRouteScalar, StepDeliversTheReportedRoute) {
    HoldScalarStep<double>("scalar-fp64");
    HoldScalarStep<float>("scalar-fp32");
}

/// The certified region-A fit body the scalar lane evaluates, at the shipped
/// partition, over a sweep of orders and arguments. The delivered value is
/// boys_impl.hpp's own read (RegionAValue over ChebyshevFit); the reference is
/// the same numbers, out of the same tables, summed by the library's own FitSum
/// at the named route.
template <boys::EvalScheme kScheme>
RouteRow SweepScalarRegionA(MulAddRoute reported, bool contracts, const char* scheme_name) {
    RouteRow row;
    row.family = "scalar-fp64";
    row.scheme = scheme_name;
    row.reported = reported;
    row.contracts = contracts;

    for (const double x : RegionASweep())
    {
        for (const int order : {0, 1, 2, 5, 9, 17, 32})
        {
            const double delivered = detail::RegionAValue<
                detail::ChebyshevFit<kScheme, boys::FitGranularity::kCoarsest>>(order, x);

            const detail::OrderPiece& piece = detail::FindPiece(order, x);
            const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;

            row.Add(delivered,
                    detail::FitSum<kScheme, RouteStep<MulAddRoute::kFused, double>>(
                        detail::kCoeffs.data() + piece.offset,
                        detail::kMonoCoeffs.data() + piece.offset,
                        piece.deg,
                        t),
                    detail::FitSum<kScheme, RouteStep<MulAddRoute::kSeparate, double>>(
                        detail::kCoeffs.data() + piece.offset,
                        detail::kMonoCoeffs.data() + piece.offset,
                        piece.deg,
                        t),
                    order,
                    x);
        }
    }

    return row;
}

TEST(BoysMulAddRouteScalar, RegionAFitBodyDeliversTheReportedRoute) {
    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp64");
    ASSERT_NE(reported, nullptr);

    EXPECT_EQ(bdetail::RouteInForce<double>(), reported->route);

    Hold(SweepScalarRegionA<boys::EvalScheme::kSplitClenshaw>(
        reported->route, bdetail::MeasureContraction<double>(), "split-clenshaw"));
    Hold(SweepScalarRegionA<boys::EvalScheme::kHorner>(
        reported->route, bdetail::MeasureContraction<double>(), "horner"));
}

// --- The across-arguments packed lane's region-A body ------------------------

/// The lane's own region-A kernel, at a route a test names: the mapped argument
/// in the packed lanes' form (boys_simd.cpp: RegionAClenshaw - one multiply-add
/// of the backend's route, not the scaled division the scalar lane forms it
/// with), then the certified split Clenshaw over the piece.
template <MulAddRoute kForcedRoute>
double ArgumentsLaneFitAtRoute(int order, double x) {
    using B = RouteStep<kForcedRoute, double>;
    const detail::OrderPiece& piece = detail::FindPiece(order, x);
    const double t = B::MulAdd(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);

    return detail::ClenshawSplit<B>(detail::kCoeffs.data() + piece.offset, piece.deg, t);
}

TEST(BoysMulAddRouteArguments, RegionABodyDeliversTheReportedRoute) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this host: the lane's body is not the "
                        "one this test holds";
    }

    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp64");
    ASSERT_NE(reported, nullptr);

    RouteRow row;
    row.family = "avx2-arguments";
    row.scheme = "split-clenshaw";
    row.reported = reported->route;
    row.contracts = bdetail::MeasureContraction<double>();

    const std::vector<double> xs = RegionASweep();

    for (const int order : {0, 1, 4, 11, 23, 32})
    {
        const std::size_t count = (xs.size() / 4) * 4;
        std::vector<double> delivered(count);

        detail::BoysRegionASimd(order, xs.data(), delivered.data(), count);

        for (std::size_t i = 0; i < count; ++i)
        {
            // The lane's first four arguments fill one vector; its tail is the
            // scalar lane's, and the sweep is a multiple of four wide, so every
            // value below is the vector body's.
            row.Add(delivered[i],
                    ArgumentsLaneFitAtRoute<MulAddRoute::kFused>(order, xs[i]),
                    ArgumentsLaneFitAtRoute<MulAddRoute::kSeparate>(order, xs[i]),
                    order,
                    xs[i]);
        }
    }

    Hold(row);
}

// --- The across-orders packed lane -------------------------------------------

/// The shipped table's geometry at one argument, as the lane's shipped body
/// reads it (src/boys_orders_simd.cpp: OrdersBody): order 0's piece, the offset
/// stride between one order's pieces and the next's, and the mapped argument in
/// the lane's own form.
struct ShippedOrdersGeometry {
    const detail::OrderPiece* piece;

    /// The offset from one order's coefficients at this piece to the next
    /// order's, which is what the lane steps the coefficient base by.
    int orderStride;

    /// The stride between the piece TABLE's entries for one order and the next,
    /// which is what the lane steps the degrees by.
    int pieceStride;

    std::size_t flatFirst;
    double t;
};

ShippedOrdersGeometry ShippedGeometryAt(double x) noexcept {
    const detail::OrderPiece& piece = detail::FindPiece(0, x);
    const int index = static_cast<int>(&piece - detail::kPieces.data()) - detail::kPieceStart[0];

    ShippedOrdersGeometry geometry{};
    geometry.piece = &piece;
    geometry.orderStride = detail::kPieces[detail::kPieceStart[1] + index].offset - piece.offset;
    geometry.pieceStride = detail::kPieceStart[1] - detail::kPieceStart[0];
    geometry.flatFirst = static_cast<std::size_t>(detail::kPieceStart[0] + index);
    geometry.t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);

    return geometry;
}

/// Whether the lane reads this order inside a full four-order group, which is
/// the branch its degree comes from.
bool OrderSitsInAGroup(int order, int nmax) noexcept {
    return order - (order % 4) + 3 <= nmax;
}

/// The degree the lane reads one order's fit at (src/boys_orders_simd.cpp:
/// GroupDegree, over StoredDegree): a grouped order at the largest of its
/// group's four stored degrees, a tail order at the lane's own. The four are
/// the same number for a table whose orders share a piece degree, which the
/// sweep checks rather than assumes - the lane's own premise,
/// src/boys_orders_simd.cpp: PiecesShareShape.
int ReadDegree(const ShippedOrdersGeometry& geometry, int order, int nmax) noexcept {
    if (!OrderSitsInAGroup(order, nmax))
    {
        return geometry.piece->deg;
    }

    const std::size_t groupFlat =
        geometry.flatFirst + static_cast<std::size_t>(order - (order % 4)) *
                                 static_cast<std::size_t>(geometry.pieceStride);
    int deg = geometry.piece->deg;

    for (int j = 1; j < 4; ++j)
    {
        const int own =
            detail::kPieces[groupFlat + static_cast<std::size_t>(j) *
                                           static_cast<std::size_t>(geometry.pieceStride)]
                .deg;
        deg = own > deg ? own : deg;
    }

    return deg;
}

/// One order's stored fit, summed by the library's certified scheme at a named
/// route, over the same piece and mapped argument the lane reads it at.
template <boys::EvalScheme kScheme>
double OrdersLaneFitAtRoute(MulAddRoute route, int order, int nmax, const ShippedOrdersGeometry& g) {
    const std::size_t offset = static_cast<std::size_t>(g.piece->offset) +
                               static_cast<std::size_t>(order) *
                                   static_cast<std::size_t>(g.orderStride);
    const int deg = ReadDegree(g, order, nmax);
    const double* cheb = detail::kCoeffs.data() + offset;
    const double* mono = detail::kMonoCoeffs.data() + offset;

    if (route == MulAddRoute::kFused)
    {
        return detail::FitSum<kScheme, RouteStep<MulAddRoute::kFused, double>>(cheb, mono, deg, g.t);
    }

    return detail::FitSum<kScheme, RouteStep<MulAddRoute::kSeparate, double>>(cheb, mono, deg, g.t);
}

/// The direct sum (src/boys_orders_simd.cpp: ChebyshevDirectSum), which no
/// library body sums: the Chebyshev series term by term, T_k by the forward
/// recurrence, written here so that the scheme the lane serves has a reference
/// at a named route. It is a transcription onto scalars, and the sweep is what
/// holds the two to each other.
template <MulAddRoute kForcedRoute>
double DirectSumAtRoute(const double* c, int deg, double t) noexcept {
    using B = RouteStep<kForcedRoute, double>;

    if (deg == 0)
    {
        return c[0];
    }

    double sum = B::MulAdd(t, c[1], c[0]);

    if (deg == 1)
    {
        return sum;
    }

    const double two_t = t + t;
    double prev = 1.0;
    double cur = t;

    for (int k = 2; k <= deg; ++k)
    {
        const double next = B::MulSub(two_t, cur, prev);
        sum = B::MulAdd(c[k], next, sum);
        prev = cur;
        cur = next;
    }

    return sum;
}

double DirectSumRefAtRoute(MulAddRoute route, int order, const ShippedOrdersGeometry& g) {
    const std::size_t offset = static_cast<std::size_t>(g.piece->offset) +
                               static_cast<std::size_t>(order) *
                                   static_cast<std::size_t>(g.orderStride);

    if (route == MulAddRoute::kFused)
    {
        return DirectSumAtRoute<MulAddRoute::kFused>(
            detail::kCoeffs.data() + offset, g.piece->deg, g.t);
    }

    return DirectSumAtRoute<MulAddRoute::kSeparate>(
        detail::kCoeffs.data() + offset, g.piece->deg, g.t);
}

/// One order's reference at a scheme, dispatched from the lane's own run-time
/// scheme (the lane's entries take the scheme as a value, the summation as a
/// type).
double OrdersLaneFitFor(boys::detail::OrdersScheme scheme,
                        MulAddRoute route,
                        int order,
                        int nmax,
                        const ShippedOrdersGeometry& g) {
    switch (scheme)
    {
    case boys::detail::OrdersScheme::kDirectSum:
        return DirectSumRefAtRoute(route, order, g);

    case boys::detail::OrdersScheme::kHorner:
        return OrdersLaneFitAtRoute<boys::EvalScheme::kHorner>(route, order, nmax, g);

    case boys::detail::OrdersScheme::kSplitClenshaw:
    case boys::detail::OrdersScheme::kCount:
        break;
    }

    return OrdersLaneFitAtRoute<boys::EvalScheme::kSplitClenshaw>(route, order, nmax, g);
}

void SweepOrdersEntry(const char* family,
                      boys::detail::OrdersScheme scheme,
                      const char* scheme_name,
                      bool composed,
                      MulAddRoute reported,
                      bool contracts) {
    RouteRow row;
    row.family = family;
    row.scheme = scheme_name;
    row.reported = reported;
    row.contracts = contracts;

    const std::vector<double> xs = RegionASweep();

    for (const int nmax : {1, 3, 7, 14, 32})
    {
        std::vector<double> delivered(static_cast<std::size_t>(nmax) + 1);

        for (const double x : xs)
        {
            if (composed)
            {
                detail::BoysAllOrdersSimdComposed(scheme, nmax, x, delivered.data(), 1);
            } else
            {
                detail::BoysAllOrdersSimd(scheme, nmax, x, delivered.data(), 1);
            }

            const ShippedOrdersGeometry geometry = ShippedGeometryAt(x);

            for (int order = 0; order <= nmax; ++order)
            {
                // The table's premise, asserted before the lane reads a group at one degree for
                // four orders: every order's piece here is cut at the same interval and stored
                // to the same degree, so a regenerated table that broke it fails, not misreads.
                if (OrderSitsInAGroup(order, nmax))
                {
                    const std::size_t groupFlat =
                        geometry.flatFirst + static_cast<std::size_t>(order - (order % 4)) *
                                                 static_cast<std::size_t>(geometry.pieceStride);

                    for (int j = 1; j < 4; ++j)
                    {
                        const detail::OrderPiece& mate =
                            detail::kPieces[groupFlat + static_cast<std::size_t>(j) *
                                                           static_cast<std::size_t>(
                                                               geometry.pieceStride)];
                        ASSERT_EQ(mate.deg, geometry.piece->deg)
                            << "the shipped table's orders no longer share a piece degree, so the "
                               "group degree this reference reads is not the lane's";
                        ASSERT_EQ(mate.a, geometry.piece->a);
                        ASSERT_EQ(mate.b, geometry.piece->b);
                    }
                }

                row.Add(delivered[static_cast<std::size_t>(order)],
                        OrdersLaneFitFor(scheme, MulAddRoute::kFused, order, nmax, geometry),
                        OrdersLaneFitFor(scheme, MulAddRoute::kSeparate, order, nmax, geometry),
                        order,
                        x);
            }
        }
    }

    Hold(row);
}

TEST(BoysMulAddRouteOrders, DoubleEntriesDeliverTheReportedRoute) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this host: the lane's body is not the "
                        "one this test holds";
    }

    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp64");
    ASSERT_NE(reported, nullptr);
    const bool contracts = bdetail::MeasureContraction<double>();

    for (const boys::detail::OrdersScheme scheme :
         {boys::detail::OrdersScheme::kSplitClenshaw,
          boys::detail::OrdersScheme::kHorner,
          boys::detail::OrdersScheme::kDirectSum})
    {
        const char* scheme_name = scheme == boys::detail::OrdersScheme::kHorner
                                      ? "horner"
                                      : (scheme == boys::detail::OrdersScheme::kDirectSum
                                             ? "direct-sum"
                                             : "split-clenshaw");

        for (const bool composed : {false, true})
        {
            std::printf("  route-liveness across-orders fetch=%s\n",
                        composed ? "composed" : "gather");
            SweepOrdersEntry(
                "avx2-orders-fp64", scheme, scheme_name, composed, reported->route, contracts);
        }
    }
}

// --- The negative control for the availability guard -------------------------

/// The guard above is a claim about WHICH BODY this entry is on this target, and
/// a skip is a way of not looking: a guard put on the wrong predicate, or on a
/// build that does have the lane, would turn a failing sweep into a green skip
/// and the route would go unwatched from that day on. What follows is what holds
/// that guard to its premise, in both directions and on every target.
///
/// The premise has two halves and both are read off the library rather than
/// restated here. Where the tier is available the entry is the across-orders
/// group body, which reads the shipped group table at the group's degree and
/// mapped argument, so it must NOT agree with the certified scalar single lane
/// at every cell of the sweep - a guard that skipped there would be skipping a
/// sweep that had something to measure. Where the tier is absent the entry is
/// the certified scalar single lane, because that is what the lane's non-x86
/// branch defines it as (src/boys_orders_simd.cpp: BoysAllOrdersSimd), so it
/// must agree at EVERY cell: the skip then states which body is built instead of
/// postponing the question.
TEST(BoysMulAddRouteOrders, TheAvailabilityGuardStatesWhichBodyIsBuilt) {
    // The predicate the guard reads and the arithmetics this build reports are
    // one statement: AppendPackedBackends lists the packed pair exactly where the
    // tier is available, so the size of BoysBackends() is the same fact told a
    // second way, and a guard reading one of them reads the other.
    const std::size_t listed = boys::backend::BoysBackends().size();
    EXPECT_EQ(listed, boys::BoysAvx2Available() ? std::size_t{4} : std::size_t{2})
        << "the packed pair is listed exactly where the tier is available, so a guard reading "
           "BoysAvx2Available() reads the fact this build's own report prints";

    // Which arithmetic this entry runs has three answers, not two: the group lane's value at the
    // reported route, at the other route (the silent substitution), and neither - a different
    // arithmetic. Counting all three tells a substituted route apart from an entry that is not the
    // group lane at all.
    const boys::backend::BackendInfo* const scalar_report = ReportedEntry("scalar-fp64");
    const boys::backend::BackendInfo* const packed_report = ReportedEntry("avx2-orders-fp64");

    std::size_t cells = 0;
    std::size_t agree_with_the_scalar_lane = 0;
    std::size_t at_the_reported_route = 0;
    std::size_t at_the_other_route = 0;

    for (const double x : RegionASweep())
    {
        for (const int nmax : {1, 7, 32})
        {
            std::vector<double> delivered(static_cast<std::size_t>(nmax) + 1);

            detail::BoysAllOrdersSimd(
                boys::detail::OrdersScheme::kSplitClenshaw, nmax, x, delivered.data(), 1);

            const ShippedOrdersGeometry geometry = ShippedGeometryAt(x);

            for (int order = 0; order <= nmax; ++order)
            {
                const std::size_t l = static_cast<std::size_t>(order);
                const double scalar = boys::BoysSingle<>(order, x);
                const double* const got = &delivered[l];

                const double group_reported =
                    OrdersLaneFitFor(boys::detail::OrdersScheme::kSplitClenshaw,
                                     scalar_report->route,
                                     order,
                                     nmax,
                                     geometry);
                const double group_other = OrdersLaneFitFor(
                    boys::detail::OrdersScheme::kSplitClenshaw,
                    scalar_report->route == MulAddRoute::kFused ? MulAddRoute::kSeparate
                                                                : MulAddRoute::kFused,
                    order,
                    nmax,
                    geometry);

                ++cells;

                if (std::memcmp(got, &scalar, sizeof(double)) == 0)
                {
                    ++agree_with_the_scalar_lane;
                }

                if (std::memcmp(got, &group_reported, sizeof(double)) == 0)
                {
                    ++at_the_reported_route;
                }

                if (std::memcmp(got, &group_other, sizeof(double)) == 0)
                {
                    ++at_the_other_route;
                }
            }
        }
    }

    std::printf("  guard-premise avx2-available=%d contracts=%d scalar-fp64-report=%s "
                "avx2-orders-fp64-report=%s contracts-at-that-arithmetic=%d cells=%zu "
                "at-the-scalar-lane=%zu at-the-reported-routes-group-lane=%zu "
                "at-the-other-routes-group-lane=%zu\n",
                boys::BoysAvx2Available() ? 1 : 0,
                bdetail::MeasureContraction<double>() ? 1 : 0,
                scalar_report == nullptr ? "(no such arithmetic is reported)"
                                         : MulAddRouteName(scalar_report->route),
                packed_report == nullptr ? "(no such arithmetic is reported)"
                                         : MulAddRouteName(packed_report->route),
                scalar_report == nullptr ? 0 : (scalar_report->contracts ? 1 : 0),
                cells,
                agree_with_the_scalar_lane,
                at_the_reported_route,
                at_the_other_route);
    std::fflush(stdout);

    if (boys::BoysAvx2Available())
    {
        EXPECT_LT(agree_with_the_scalar_lane, cells)
            << "the AVX2 tier is available, so this entry is the across-orders group body and not "
               "the certified scalar single lane; it agreed with that lane at all "
            << cells
            << " cells of the sweep, so either the entry is not the group body or the guard above "
               "is skipping a sweep that had something to measure";
    } else
    {
        EXPECT_EQ(agree_with_the_scalar_lane, cells)
            << "the AVX2 tier is absent, so this entry is the certified scalar single lane "
               "(src/boys_orders_simd.cpp: BoysAllOrdersSimd) and must agree with it at all "
            << cells << " cells of the sweep; it agreed at " << agree_with_the_scalar_lane
            << ", so the entry is some other body and the guard above is skipping a sweep that "
               "should have run";
    }
}

// --- The across-orders lane, single precision ---------------------------------

/// The float lane's own cover is cut per order (boys_orders_simd.cpp:
/// FindPieceF32), so one argument lands in a different piece in each of its
/// eight lanes; each order's value is its own piece's fit at its own mapped
/// argument and its own stored degree, which is what the group's masked read at
/// the group's largest degree delivers bit for bit.
template <boys::EvalScheme kScheme>
float F32OrdersLaneFitAtRoute(MulAddRoute route, int order, float x) {
    const detail::f32::OrderPiece& piece = detail::FindPieceF32(order, x);
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    const float* cheb = detail::f32::kCoeffs.data() + piece.offset;
    const float* mono = detail::f32::kMonoCoeffs.data() + piece.offset;

    if (route == MulAddRoute::kFused)
    {
        return detail::FitSum<kScheme, RouteStep<MulAddRoute::kFused, float>>(
            cheb, mono, piece.deg, t);
    }

    return detail::FitSum<kScheme, RouteStep<MulAddRoute::kSeparate, float>>(
        cheb, mono, piece.deg, t);
}

template <boys::EvalScheme kScheme>
RouteRow SweepF32Orders(boys::detail::OrdersScheme scheme,
                        MulAddRoute reported,
                        bool contracts,
                        const char* scheme_name) {
    RouteRow row;
    row.family = "avx2-orders-fp32";
    row.scheme = scheme_name;
    row.reported = reported;
    row.contracts = contracts;

    for (const double xd : RegionASweep())
    {
        const float x = static_cast<float>(xd);

        for (const int nmax : {7, 15, 32})
        {
            std::vector<float> delivered(static_cast<std::size_t>(nmax) + 1);

            detail::BoysAllOrdersF32Simd(
                scheme, boys::FitRoute::kChebyshev, nmax, x, delivered.data());

            for (int order = 0; order <= nmax; ++order)
            {
                row.Add(static_cast<double>(delivered[static_cast<std::size_t>(order)]),
                        static_cast<double>(
                            F32OrdersLaneFitAtRoute<kScheme>(MulAddRoute::kFused, order, x)),
                        static_cast<double>(
                            F32OrdersLaneFitAtRoute<kScheme>(MulAddRoute::kSeparate, order, x)),
                        order,
                        static_cast<double>(x));
            }
        }
    }

    return row;
}

TEST(BoysMulAddRouteOrders, F32EntryDeliversTheReportedRoute) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this host: the single-precision "
                        "lane's body is not the one this test holds";
    }

    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp32");
    ASSERT_NE(reported, nullptr);
    const bool contracts = bdetail::MeasureContraction<float>();

    // The entry's route argument selects a fit family - the certified Chebyshev
    // one here - and its scheme the summation, which the reference follows
    // scheme for scheme out of the same tables.
    Hold(SweepF32Orders<boys::EvalScheme::kSplitClenshaw>(
        boys::detail::OrdersScheme::kSplitClenshaw, reported->route, contracts, "split-clenshaw"));
    Hold(SweepF32Orders<boys::EvalScheme::kHorner>(
        boys::detail::OrdersScheme::kHorner, reported->route, contracts, "horner"));
}

// The across-orders POLICY entries: boys_impl.hpp's detail::BoysAllOrdersPacked and
// detail::BoysAllOrdersF32Packed, each a template electing its body at the build's route. Measured
// before this section was written: a packed-ladder dump at the two route selections moved 2,271 of
// 13,728 cells, held per cell against this file's reference, below kX0 and kFlatHi.

/// The library's own certified summation over one stored fit, at a route the
/// caller names instead of the build's selection. Same two tables, same mapped
/// argument, same order; the step is the only thing the route changes.
template <boys::EvalScheme kScheme>
double PackedFitAtRoute(
    MulAddRoute route, const double* cheb, const double* mono, int deg, double t) noexcept {
    if (route == MulAddRoute::kFused)
    {
        return detail::FitSum<kScheme, RouteStep<MulAddRoute::kFused, double>>(cheb, mono, deg, t);
    }

    return detail::FitSum<kScheme, RouteStep<MulAddRoute::kSeparate, double>>(cheb, mono, deg, t);
}

template <boys::EvalScheme kScheme>
float PackedFitAtRouteF32(
    MulAddRoute route, const float* cheb, const float* mono, int deg, float t) noexcept {
    if (route == MulAddRoute::kFused)
    {
        return detail::FitSum<kScheme, RouteStep<MulAddRoute::kFused, float>>(cheb, mono, deg, t);
    }

    return detail::FitSum<kScheme, RouteStep<MulAddRoute::kSeparate, float>>(cheb, mono, deg, t);
}

/// The same certified summation at the SCALAR backend's contract instead of the
/// orders lane's step: the two differ in what a multiply-subtract does, and one
/// cell of the float entry reaches the scalar backend directly (the uniform
/// grid's Chebyshev member, boys_impl.hpp: UniformOrderAtF32 over
/// backend::ScalarFp32), so its reference has to be that arithmetic.
template <boys::EvalScheme kScheme>
float ScalarFitAtRouteF32(
    MulAddRoute route, const float* cheb, const float* mono, int deg, float t) noexcept {
    if (route == MulAddRoute::kFused)
    {
        return detail::FitSum<kScheme, RouteForced<MulAddRoute::kFused, float>>(cheb, mono, deg, t);
    }

    return detail::FitSum<kScheme, RouteForced<MulAddRoute::kSeparate, float>>(cheb, mono, deg, t);
}

/// One cut numerator/denominator pair at a named route: boys_impl.hpp's
/// RationalPieceAtCut, RationalPieceNarrowAtCut and RationalPieceF32AtCutBody,
/// which differ in their table and their width alone, with the scalar backend's
/// step made a template argument. The cut keeps the low-order terms of both parts,
/// and the denominator's position is the piece's FULL numerator degree's, as the
/// library's three bodies write it.
template <MulAddRoute kRoute, typename T>
T RationalCutAtTheRoute(
    const T* c, int storedNumDeg, int numDeg, int denDeg, T t) noexcept {
    using B = RouteForced<kRoute, T>;
    T num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = B::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    T den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = B::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / B::MulAdd(den, t, T{1});
}

template <typename T>
T RationalCutAtRoute(
    MulAddRoute route, const T* c, int storedNumDeg, int numDeg, int denDeg, T t) noexcept {
    if (route == MulAddRoute::kFused)
    {
        return RationalCutAtTheRoute<MulAddRoute::kFused, T>(c, storedNumDeg, numDeg, denDeg, t);
    }

    return RationalCutAtTheRoute<MulAddRoute::kSeparate, T>(c, storedNumDeg, numDeg, denDeg, t);
}

/// One order off the uniform grid's rational member at a named route:
/// boys_impl.hpp's RationalUniformOrderAt, whose two readers the entry reaches
/// through the certified scalar single lane (SingleOrder, UniformSingleOrder).
template <MulAddRoute kRoute>
double RationalUniformAtRoute(const detail::FlatPoint& at, int order) noexcept {
    using B = RouteForced<kRoute, double>;
    const std::size_t stored = static_cast<std::size_t>(detail::kFlatRatStored[at.iv]);
    const double* c = detail::kFlatRatCoeffs.data() +
                      static_cast<std::size_t>(detail::kFlatRatOffsets[at.iv]) +
                      static_cast<std::size_t>(order) * stored;
    const int m = detail::kFlatRatNumDeg[at.iv];
    const int k = detail::kFlatRatDenDeg[at.iv];
    double num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = B::MulAdd(num, at.t, c[j]);
    }

    double den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = B::MulAdd(den, at.t, c[m + j]);
    }

    return num / B::MulAdd(den, at.t, 1.0);
}

template <MulAddRoute kRoute>
float RationalUniformAtRouteF32(const detail::FlatPointF32& at, int order) noexcept {
    using B = RouteForced<kRoute, float>;
    const std::size_t stored = static_cast<std::size_t>(detail::f32::kFlatRatStoredF32[at.iv]);
    const float* c = detail::f32::kFlatRatCoeffsF32.data() +
                     static_cast<std::size_t>(detail::f32::kFlatRatOffsetsF32[at.iv]) +
                     static_cast<std::size_t>(order) * stored;
    const int m = detail::f32::kFlatRatNumDegF32[at.iv];
    const int k = detail::f32::kFlatRatDenDegF32[at.iv];
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = B::MulAdd(num, at.t, c[j]);
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = B::MulAdd(den, at.t, c[m + j]);
    }

    return num / B::MulAdd(den, at.t, 1.0f);
}

/// The order from which the rational route's own fits are what the lane reads:
/// the route hands an order over at that order's own end of region A, and the
/// ends are non-decreasing in the order, so the orders the route answers are a
/// prefix (boys_orders_simd.cpp: RationalOrdersBody).
int ServedOrders(int nmax, double x) noexcept {
    int served = 0;

    while (served <= nmax && x >= detail::kTierThresholds[static_cast<std::size_t>(served)])
    {
        ++served;
    }

    return served;
}

/// The degree a group of four orders is read at: the largest of the four lanes'
/// own, which is the lane's group rule (boys_orders_simd.cpp: GroupDegree) and is
/// not the same number as the degree any one lane was certified at. The lane
/// states it is free to read above a lane's own cut, so the same order can carry
/// one value in a whole group and another in a tail - which is why the reading
/// below says which of the two an order is in rather than assuming one.
///
/// The four lanes' indices come from a lookup and not from a stride: the narrow
/// partition's pieces are cut per order, so its four lanes are not one stride
/// apart in its table.
template <typename Degrees, typename FlatAt>
int GroupDegreeOf(const Degrees& degrees, int own, FlatAt flat_of_lane) noexcept {
    int deg = own;

    for (int j = 1; j < 4; ++j)
    {
        const int mate = degrees[flat_of_lane(j)];
        deg = mate > deg ? mate : deg;
    }

    return deg;
}

/// The narrow partition's piece index for one order, which is what its group
/// degree rule looks its four lanes up by (boys_orders_simd.cpp: NarrowGeometry).
std::size_t NarrowFlatOf(int order, double x) noexcept {
    return static_cast<std::size_t>(&detail::FindNarrowAPiece(order, x) -
                                    detail::kNarrowAPieces.data());
}

/// Which of a route body's two readings one order takes, and whether it is read
/// with its group: the rule boys_orders_simd.cpp's RationalOrdersBody and
/// NarrowRationalOrdersBody share - a whole group below the handover is the
/// route's own pair, a whole group at or above it is the partition's fit at the
/// group's degree, and a group the handover falls inside, and the tail, are read
/// one order at a time, in those same two readings.
struct PackedReading {
    bool route;

    /// Read at the four lanes' largest degree rather than at the lane's own.
    bool grouped;
};

PackedReading PackedReadingOf(int order, int nmax, int served) noexcept {
    const int l = order - (order % 4);

    if (OrderSitsInAGroup(order, nmax))
    {
        if (l >= served)
        {
            return PackedReading{false, true};
        }

        if (l + 4 <= served)
        {
            return PackedReading{true, false};
        }
    }

    return PackedReading{order < served, false};
}

/// The double entry's region-A value for one order, at a route the caller names:
/// the entry's own body, read off the tables it reads.
template <boys::EvalScheme kScheme,
          boys::FitRoute kFitRoute,
          boys::FitGranularity kGranularity,
          boys::DivisionForm kForm>
double PackedOrderAtRoute(MulAddRoute route, int order, int nmax, double x) {
    if constexpr (kGranularity == boys::FitGranularity::kUniform)
    {
        const detail::FlatPoint at = detail::FlatLocate(x);

        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            // The entry delegates this cell to the certified scalar single lane,
            // whose uniform-rational read is RationalUniformOrderAt.
            if (route == MulAddRoute::kFused)
            {
                return RationalUniformAtRoute<MulAddRoute::kFused>(at, order);
            }

            return RationalUniformAtRoute<MulAddRoute::kSeparate>(at, order);
        } else
        {
            const int deg = detail::kFlatDegs[at.iv];
            const std::size_t base =
                at.block + static_cast<std::size_t>(order) * static_cast<std::size_t>(deg + 1);

            return PackedFitAtRoute<kScheme>(route,
                                             detail::kFlatCoeffs.data() + base,
                                             detail::kFlatMonoCoeffs.data() + base,
                                             deg,
                                             at.t);
        }
    } else if constexpr (kGranularity == boys::FitGranularity::kCoarsest)
    {
        const ShippedOrdersGeometry g = ShippedGeometryAt(x);
        const std::size_t offset = static_cast<std::size_t>(g.piece->offset) +
                                   static_cast<std::size_t>(order) *
                                       static_cast<std::size_t>(g.orderStride);

        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            static constexpr auto kPairs = detail::RationalRegionADegrees();
            static constexpr auto kDegrees =
                detail::RegionADegrees<detail::BoysRole::kDoubleSingle,
                                       detail::SchemeTailBasis<kScheme>()>();
            const std::size_t flat = g.flatFirst + static_cast<std::size_t>(order) *
                                                       static_cast<std::size_t>(g.pieceStride);
            const PackedReading reading = PackedReadingOf(order, nmax, ServedOrders(nmax, x));

            // This body's own mapped argument, 2(x - a)/(b - a) - 1, which the
            // route's pair and the shipped fit are both read at here - not the
            // fixed-piece mapping the shipped body forms.
            const double t = 2.0 * (x - g.piece->a) / (g.piece->b - g.piece->a) - 1.0;

            if (reading.route)
            {
                return RationalCutAtRoute(route,
                                          detail::kRatACoeffs.data() + detail::kRatAOffset[flat],
                                          detail::kRatANumDeg[flat],
                                          kPairs.num[flat],
                                          kPairs.den[flat],
                                          t);
            }

            int deg = kDegrees[flat];

            if (reading.grouped)
            {
                const int l = order - (order % 4);
                deg = GroupDegreeOf(kDegrees, deg, [&](int j) {
                    return static_cast<int>(g.flatFirst) + (l + j) * g.pieceStride;
                });
            }

            // The route's body maps both of its readings with the plain form
            // above (boys_orders_simd.cpp: RationalOrdersBody), not with the
            // shipped body's fused one - the two differ in the last bit at some
            // arguments, and this branch reads what that body reads.
            return PackedFitAtRoute<kScheme>(route,
                                             detail::kCoeffs.data() + offset,
                                             detail::kMonoCoeffs.data() + offset,
                                             deg,
                                             t);
        } else
        {
            // This cell's body is the shipped cover's at its STORED degrees
            // (OrdersByScheme), which is the reading this file's across-orders
            // sweep holds too; the reference reads the same table at the same
            // group rule.
            const int deg = ReadDegree(g, order, nmax);

            return PackedFitAtRoute<kScheme>(route,
                                             detail::kCoeffs.data() + offset,
                                             detail::kMonoCoeffs.data() + offset,
                                             deg,
                                             g.t);
        }
    } else
    {
        const detail::OrderPiece& piece = detail::FindNarrowAPiece(order, x);
        const std::size_t flat = static_cast<std::size_t>(&piece - detail::kNarrowAPieces.data());
        const double t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);

        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            static constexpr auto kPairs = detail::RationalRegionANarrowDegrees();
            static constexpr auto kDegrees =
                detail::NarrowRegionADegrees<detail::BoysRole::kDoubleSingle,
                                             detail::SchemeTailBasis<kScheme>()>();
            const PackedReading reading = PackedReadingOf(order, nmax, ServedOrders(nmax, x));

            if (reading.route)
            {
                return RationalCutAtRoute(route,
                                          detail::kNarrowRatACoeffs.data() +
                                              detail::kNarrowRatAOffset[flat],
                                          detail::kNarrowRatANumDeg[flat],
                                          kPairs.num[flat],
                                          kPairs.den[flat],
                                          t);
            }

            int deg = kDegrees[flat];

            if (reading.grouped)
            {
                const int l = order - (order % 4);
                deg = GroupDegreeOf(
                    kDegrees, deg, [&](int j) { return static_cast<int>(NarrowFlatOf(l + j, x)); });
            }

            return PackedFitAtRoute<kScheme>(route,
                                             detail::kNarrowACoeffs.data() + piece.offset,
                                             detail::kNarrowAMonoCoeffs.data() + piece.offset,
                                             deg,
                                             t);
        } else
        {
            static constexpr auto kDegrees =
                detail::NarrowRegionADegrees<detail::BoysRole::kDoubleSingle,
                                             detail::SchemeTailBasis<kScheme>()>();
            int deg = kDegrees[flat];

            if (OrderSitsInAGroup(order, nmax))
            {
                const int l = order - (order % 4);
                deg = GroupDegreeOf(
                    kDegrees, deg, [&](int j) { return static_cast<int>(NarrowFlatOf(l + j, x)); });
            }

            return PackedFitAtRoute<kScheme>(route,
                                             detail::kNarrowACoeffs.data() + piece.offset,
                                             detail::kNarrowAMonoCoeffs.data() + piece.offset,
                                             deg,
                                             t);
        }
    }
}

/// The float entry's region-A value for one order, at a named route. The float
/// fetch zeroes every coefficient above the degree its lane is read at, so a
/// lane's value is its own fit's at its own degree whatever the group runs at -
/// the group's `degMax` reaches the shape of the recurrence and not the value,
/// which the lane's own suite asserts and this reference reads as stated.
template <boys::EvalScheme kScheme,
          boys::FitRoute kFitRoute,
          boys::BoysBudget kBudget,
          boys::FitGranularity kGranularity,
          boys::DivisionForm kForm>
float PackedOrderAtRouteF32(MulAddRoute route, int order, int nmax, float x) {
    constexpr detail::BoysRole kRole = (kBudget == boys::BoysBudget::kFloat)
                                           ? detail::BoysRole::kF32Single
                                           : detail::BoysRole::kF32Fp16Single;

    if constexpr (kGranularity == boys::FitGranularity::kUniform)
    {
        const detail::FlatPointF32 at = detail::FlatLocateF32(x);

        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            if (route == MulAddRoute::kFused)
            {
                return RationalUniformAtRouteF32<MulAddRoute::kFused>(at, order);
            }

            return RationalUniformAtRouteF32<MulAddRoute::kSeparate>(at, order);
        } else
        {
            const int deg = detail::f32::kFlatDegsF32[at.iv];
            const std::size_t base =
                at.block + static_cast<std::size_t>(order) * static_cast<std::size_t>(deg + 1);

            // This cell is the certified scalar body's own read (boys_impl.hpp:
            // UniformOrderAtF32, over backend::ScalarFp32), not the packed lane's
            // step: the grid's cells are stored per order with nothing to step,
            // so the entry reads them one at a time in the scalar backend.
            return ScalarFitAtRouteF32<kScheme>(route,
                                                detail::f32::kFlatCoeffsF32.data() + base,
                                                detail::f32::kFlatMonoCoeffsF32.data() + base,
                                                deg,
                                                at.t);
        }
    } else if constexpr (kGranularity == boys::FitGranularity::kCoarsest)
    {
        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            static constexpr auto kPairs = detail::RationalRegionAF32Degrees<kRole>();
            const detail::f32::RatPiece& piece = detail::FindRatPieceF32(order, x);
            const std::size_t flat =
                static_cast<std::size_t>(&piece - detail::f32::kRatAPieces.data());
            const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

            // The group reader clamps a lane's denominator cut to the piece's
            // stored one (BuildF32RatGroup); the tail hands the cut through.
            const int cut = kPairs.den[flat];
            const bool grouped = order - (order % 8) + 8 <= nmax + 1;
            const int den = grouped && cut > piece.dendeg ? piece.dendeg : cut;

            return RationalCutAtRoute(route,
                                      detail::f32::kRatACoeffs.data() + piece.offset,
                                      piece.numdeg,
                                      kPairs.num[flat],
                                      den,
                                      t);
        } else
        {
            const detail::f32::OrderPiece& piece = detail::FindPieceF32(order, x);
            const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

            // This cell's body reads the shipped float cover whole
            // (F32StoredDegree), so the degree is the piece's own.
            return PackedFitAtRouteF32<kScheme>(route,
                                                detail::f32::kCoeffs.data() + piece.offset,
                                                detail::f32::kMonoCoeffs.data() + piece.offset,
                                                piece.deg,
                                                t);
        }
    } else
    {
        if constexpr (kFitRoute == boys::FitRoute::kRationalMinimax)
        {
            static constexpr auto kPairs = detail::NarrowRationalRegionAF32Degrees<kRole>();
            const detail::f32::RatPiece& piece = detail::FindNarrowRatPieceF32(order, x);
            const std::size_t flat =
                static_cast<std::size_t>(&piece - detail::f32::kNarrowRatAPiecesF32.data());
            const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
            const int cut = kPairs.den[flat];
            const bool grouped = order - (order % 8) + 8 <= nmax + 1;
            const int den = grouped && cut > piece.dendeg ? piece.dendeg : cut;

            return RationalCutAtRoute(route,
                                      detail::f32::kNarrowRatACoeffsF32.data() + piece.offset,
                                      piece.numdeg,
                                      kPairs.num[flat],
                                      den,
                                      t);
        } else
        {
            static constexpr auto kDegrees =
                detail::NarrowRegionADegrees<kRole, detail::SchemeTailBasis<kScheme>()>();
            const detail::f32::OrderPiece& piece = detail::FindNarrowPieceF32(order, x);
            const std::size_t flat =
                static_cast<std::size_t>(&piece - detail::f32::kNarrowAPiecesF32.data());
            const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

            return PackedFitAtRouteF32<kScheme>(route,
                                                detail::f32::kNarrowACoeffsF32.data() + piece.offset,
                                                detail::f32::kNarrowAMonoCoeffsF32.data() +
                                                    piece.offset,
                                                kDegrees[flat],
                                                t);
        }
    }
}

/// The double entry over the sweep, at one (scheme, route, partition, form) cell:
/// what the entry delivers against both routes' reference values.
template <boys::EvalScheme kScheme,
          boys::FitRoute kFitRoute,
          boys::FitGranularity kGranularity,
          boys::DivisionForm kForm>
RouteRow SweepPackedOrders(
    const char* label, MulAddRoute reported, bool contracts) {
    RouteRow row;
    row.family = "avx2-orders-packed-fp64";
    row.scheme = label;
    row.reported = reported;
    row.contracts = contracts;

    for (const double x : RegionASweep())
    {
        for (const int nmax : {1, 3, 7, 14, 32})
        {
            std::vector<double> delivered(static_cast<std::size_t>(nmax) + 1);

            detail::BoysAllOrdersPacked<kScheme, kFitRoute, kGranularity, kForm>(
                nmax, x, delivered.data());

            for (int order = 0; order <= nmax; ++order)
            {
                row.Add(delivered[static_cast<std::size_t>(order)],
                        PackedOrderAtRoute<kScheme, kFitRoute, kGranularity, kForm>(
                            MulAddRoute::kFused, order, nmax, x),
                        PackedOrderAtRoute<kScheme, kFitRoute, kGranularity, kForm>(
                            MulAddRoute::kSeparate, order, nmax, x),
                        order,
                        x);
            }
        }
    }

    return row;
}

/// The float entry over the same sweep, at one (scheme, route, budget, partition,
/// form) cell.
template <boys::EvalScheme kScheme,
          boys::FitRoute kFitRoute,
          boys::BoysBudget kBudget,
          boys::FitGranularity kGranularity,
          boys::DivisionForm kForm>
RouteRow SweepPackedOrdersF32(
    const char* label, MulAddRoute reported, bool contracts) {
    RouteRow row;
    row.family = "avx2-orders-packed-fp32";
    row.scheme = label;
    row.reported = reported;
    row.contracts = contracts;

    for (const double xd : RegionASweep())
    {
        const float x = static_cast<float>(xd);

        for (const int nmax : {7, 15, 32})
        {
            std::vector<float> delivered(static_cast<std::size_t>(nmax) + 1);

            detail::BoysAllOrdersF32Packed<kScheme, kFitRoute, kBudget, kGranularity, kForm>(
                nmax, x, delivered.data());

            for (int order = 0; order <= nmax; ++order)
            {
                row.Add(static_cast<double>(delivered[static_cast<std::size_t>(order)]),
                        static_cast<double>(
                            PackedOrderAtRouteF32<kScheme, kFitRoute, kBudget, kGranularity, kForm>(
                                MulAddRoute::kFused, order, nmax, x)),
                        static_cast<double>(
                            PackedOrderAtRouteF32<kScheme, kFitRoute, kBudget, kGranularity, kForm>(
                                MulAddRoute::kSeparate, order, nmax, x)),
                        order,
                        static_cast<double>(x));
            }
        }
    }

    return row;
}

/// One (scheme, partition) cell of the double entry at all three division
/// forms: three rows, one per form.
template <boys::EvalScheme kScheme, boys::FitRoute kFitRoute, boys::FitGranularity kGranularity>
void HoldPackedDoubleCell(
    const char* scheme_name, const char* partition_name, MulAddRoute reported, bool contracts) {
    char label[96];

    std::snprintf(label, sizeof label, "%s/%s/exact-division", scheme_name, partition_name);
    Hold(SweepPackedOrders<kScheme, kFitRoute, kGranularity, boys::DivisionForm::kExactDivision>(
        label, reported, contracts));

    std::snprintf(label, sizeof label, "%s/%s/plain-reciprocal", scheme_name, partition_name);
    Hold(SweepPackedOrders<kScheme, kFitRoute, kGranularity,
                           boys::DivisionForm::kPlainReciprocal>(label, reported, contracts));

    std::snprintf(label, sizeof label, "%s/%s/refined-reciprocal", scheme_name, partition_name);
    Hold(SweepPackedOrders<kScheme, kFitRoute, kGranularity,
                           boys::DivisionForm::kRefinedReciprocal>(label, reported, contracts));
}

/// The same for the float entry, one budget at a time.
template <boys::EvalScheme kScheme, boys::FitRoute kFitRoute, boys::BoysBudget kBudget,
          boys::FitGranularity kGranularity>
void HoldPackedF32Cell(const char* scheme_name,
                       const char* partition_name,
                       const char* budget_name,
                       MulAddRoute reported,
                       bool contracts) {
    char label[96];

    std::snprintf(label, sizeof label, "%s/%s/%s/exact-division", scheme_name, partition_name,
                  budget_name);
    Hold(SweepPackedOrdersF32<kScheme, kFitRoute, kBudget, kGranularity,
                              boys::DivisionForm::kExactDivision>(label, reported, contracts));

    std::snprintf(label, sizeof label, "%s/%s/%s/plain-reciprocal", scheme_name, partition_name,
                  budget_name);
    Hold(SweepPackedOrdersF32<kScheme, kFitRoute, kBudget, kGranularity,
                              boys::DivisionForm::kPlainReciprocal>(label, reported, contracts));

    std::snprintf(label, sizeof label, "%s/%s/%s/refined-reciprocal", scheme_name, partition_name,
                  budget_name);
    Hold(SweepPackedOrdersF32<kScheme, kFitRoute, kBudget, kGranularity,
                              boys::DivisionForm::kRefinedReciprocal>(label, reported, contracts));
}

template <boys::BoysBudget kBudget>
void HoldPackedF32Budget(const char* budget_name, MulAddRoute reported, bool contracts) {
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kCoarsest>(
        "split-clenshaw", "coarsest", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kCoarsest>(
        "horner", "coarsest", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kNarrow>(
        "split-clenshaw", "narrow", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kNarrow>(
        "horner", "narrow", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kUniform>(
        "split-clenshaw", "uniform", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev, kBudget,
                      boys::FitGranularity::kUniform>(
        "horner", "uniform", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kCoarsest>(
        "split-clenshaw", "coarsest-rational", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kCoarsest>(
        "horner", "coarsest-rational", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kNarrow>(
        "split-clenshaw", "narrow-rational", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kNarrow>(
        "horner", "narrow-rational", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kUniform>(
        "split-clenshaw", "uniform-rational", budget_name, reported, contracts);
    HoldPackedF32Cell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax, kBudget,
                      boys::FitGranularity::kUniform>(
        "horner", "uniform-rational", budget_name, reported, contracts);
}

TEST(BoysMulAddRouteOrders, PackedEntryDeliversTheReportedRoute) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this host: every cell of these entries "
                        "is the certified scalar single lane here, and this sweep holds the "
                        "vector bodies";
    }

    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp64");
    ASSERT_NE(reported, nullptr);
    const bool contracts = bdetail::MeasureContraction<double>();
    const MulAddRoute r = reported->route;

    std::printf("  route-liveness across-orders-pack packed entry=%s\n", MulAddRouteName(r));

    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kCoarsest>("split-clenshaw", "coarsest", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kCoarsest>("horner", "coarsest", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kNarrow>("split-clenshaw", "narrow", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kNarrow>("horner", "narrow", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kUniform>("split-clenshaw", "uniform", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kChebyshev,
                         boys::FitGranularity::kUniform>("horner", "uniform", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kCoarsest>(
        "split-clenshaw", "coarsest-rational", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kCoarsest>(
        "horner", "coarsest-rational", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kNarrow>(
        "split-clenshaw", "narrow-rational", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kNarrow>("horner", "narrow-rational", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kSplitClenshaw, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kUniform>(
        "split-clenshaw", "uniform-rational", r, contracts);
    HoldPackedDoubleCell<boys::EvalScheme::kHorner, boys::FitRoute::kRationalMinimax,
                         boys::FitGranularity::kUniform>("horner", "uniform-rational", r, contracts);
}

TEST(BoysMulAddRouteOrders, PackedF32EntryDeliversTheReportedRoute) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the AVX2 tier is not available on this host: every cell of these entries "
                        "is the certified scalar single lane here, and this sweep holds the "
                        "vector bodies";
    }

    const boys::backend::BackendInfo* reported = ReportedEntry("scalar-fp32");
    ASSERT_NE(reported, nullptr);
    const bool contracts = bdetail::MeasureContraction<float>();
    const MulAddRoute r = reported->route;

    std::printf("  route-liveness across-orders-pack packed-f32 entry=%s\n", MulAddRouteName(r));

    HoldPackedF32Budget<boys::BoysBudget::kFloat>("kFloat", r, contracts);
    HoldPackedF32Budget<boys::BoysBudget::kFp16>("kFp16", r, contracts);
}

} // namespace
