// The multiply-add route's liveness: a lane built to read BOYS_MULADD_SEPARATE
// has to DELIVER that route's values, not merely report it.
//
// WHY THIS TEST EXISTS. The route is a build fact (boys/backend.hpp:
// kSelectedRoute) and the lanes read it as one, so a lane that stops reading it
// keeps compiling, keeps every published accuracy bound - both routes are
// accurate, and the bounds are the fused route's - and keeps the suites that
// name the route, whose arms differ in what they PRINT rather than in what they
// assert. Measured before this file was written: reverting the across-orders
// lane to its hard-coded fused intrinsic and building at the separate route
// leaves the across-orders suites at [ PASSED ] 19 tests, exit 0. A report can
// be right while the arithmetic is wrong, so what is held here is the VALUES a
// lane delivers.
//
// THE REFERENCE is the library's own certified summation - ClenshawSplit,
// HornerMono, FitSum, the same templates the lanes evaluate - instantiated over
// a backend whose route is a template argument instead of the build's selection
// (RouteForced and RouteStep below). Same piece, same coefficients, same mapped
// argument, same steps, same order; the only difference is which multiply-add
// the step is built from. A lane that ignores the selection therefore delivers
// the OTHER route's values, and the counts below say so.
//
// WHERE IT CAN SEE THE DIFFERENCE, and why the counts are printed. The two
// routes are one arithmetic on a build whose bare `a * b + c` contracts
// (boys/backend.hpp: MeasureContraction), and no value comparison can part them
// there. Each row prints the route the library reports for its family, this
// translation unit's own contraction measurement, and the counts: the cells
// compared, the cells the two routes give different values at, the cells the
// lane delivered off the reported route, and the cells it delivered the OTHER
// route's value at. A non-zero `routes-differ` is required wherever this unit
// measures that the build keeps the routes apart, so a sweep that stopped
// proving anything fails rather than passing quietly.
//
// WHICH LANES THIS COVERS, and which it does not:
//   * the scalar lanes (boys/backend.hpp: Scalar<T>), at the step, at the
//     multiply-subtract the route does NOT reach, and through the certified
//     region-A fit body boys_impl.hpp evaluates (both schemes);
//   * the across-arguments packed lane's region-A body (boys_simd.cpp:
//     BoysRegionASimd), which reaches the packed backends at the selection;
//   * the across-orders lane (boys_orders_simd.cpp): the double entry and its
//     composed fetch, and the single-precision entry, at every scheme this
//     library certifies a summation for.
// The packed backends themselves (boys_backend_simd.hpp: Avx2Fp64/Avx2Fp32) are
// held in tests/boys_muladd_route_simd_test.cpp, whose translation unit carries
// the intrinsics' flags the way the library's SIMD units do.
//
// NOT COVERED, and why: the CUDA device lane, which has its own
// boys-cuda-route-tests and no device in this configuration; the rational fit
// family's own region-A body, which reaches the route through the same
// Scalar<T>::MulAdd the step row holds, over a different summation; the derived
// partitions of the across-orders lane (narrow and uniform), whose bodies run
// the same route-templated StepMulAdd/StepMulSub the shipped body runs and are
// not separately swept; and the arguments at or above kX0, where a lane's
// region-A body stops being what answers and the entry hands the argument to
// the region-B, asymptotic and certified scalar bodies
// (boys_orders_simd.cpp:1076 OrdersLaneApplies, :1744 F32OrdersLaneApplies).
// The sweeps here stay strictly below that cut (RegionASweep), so what they
// hold, and all they claim, is each lane's region-A body at the build's route.

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

        // What this sweep can tell apart for the multiply-subtract: the fused
        // step, one instruction and one rounding, and the two-rounding
        // difference the operation's own contract names. A lane that let the
        // route reach this operation would deliver the first where the two
        // differ, so the count is also the bar the sweep has to clear.
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

    // The report's contraction answer is taken in the unit the scalar
    // arithmetic is compiled in (src/boys.cpp), and this unit carries the same
    // flags. A build whose measured answer and this unit's measurement disagree
    // is a build whose claims and whose arithmetic are not the same statement,
    // and it fails here rather than passing quietly.
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
                // The table's premise, which the lane asserts of itself before
                // it reads a group at one degree for four orders: every order's
                // piece at this index is cut at the same interval and stored to
                // the same degree. A regenerated table that broke it is a
                // failure here rather than a reference read at a wrong degree.
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

    // Which arithmetic this entry runs is a question with three answers, not
    // two: the group lane's value at the reported route, the group lane's value
    // at the OTHER route - which is the silent substitution a lane that stopped
    // reading the selection would deliver - and neither of those, which is a
    // different arithmetic. The sweep counts all three, so a build that reports
    // one route and delivers the other is told apart from one whose entry is not
    // the group lane at all rather than being read as either.
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

} // namespace
