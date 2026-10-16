#pragma once

// The across-orders packed lane's measurement entries: the reference reading the installed path
// (boys/boys_orders_lanes.hpp) keeps out of line, because their value is what the option probe
// measures against the entries a consumer compiles. Private to the library, declared here and
// defined in boys_orders_simd.cpp, over the closure and OrdersScheme that path declares.

#include "boys/backend.hpp"
#include "boys/boys.hpp"
#include "boys/boys_orders_lanes.hpp"

#include <cstddef>

namespace boys::detail {

/// Fills out[l * stride] with F_l(x) for l = 0..nmax, by the named scheme.
///
/// Every value is the stored region-A fit of its own order, evaluated
/// independently: no order is reached from another by a recursion. That is
/// what the across-orders axis buys in place of the recursion.
///
/// \param scheme which summation of the stored fits to use
/// \param nmax   highest order, 0..kMaxBoysOrder
/// \param x      argument; the fits cover 0 <= x < kX0
/// \param out    nmax + 1 values, written at out[l * stride] for l = 0..nmax
/// \param stride the output's order stride, >= 1
/// \pre out holds (nmax + 1) * stride doubles, and x >= 0
///
/// Outside the fits' interval the entry falls back to the certified scalar
/// lanes, one order at a time, so it is defined for every argument the library
/// accepts; only inside the interval does it vectorise.
void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept;

/// The same entry with the other coefficient fetch: the four orders' step
/// coefficients composed from four loads rather than gathered with one
/// instruction.
///
/// The two are the same lane - same schemes, same tables, same arithmetic,
/// same values - and differ only in the instruction the fetch costs. Which is
/// cheaper is a property of the machine, so both are offered and the
/// benchmark measures them rather than either being chosen here.
///
/// \param scheme see BoysAllOrdersSimd
/// \param nmax   see BoysAllOrdersSimd
/// \param x      see BoysAllOrdersSimd
/// \param out    see BoysAllOrdersSimd
/// \param stride see BoysAllOrdersSimd
void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept;

// The same axis where a register holds eight floats rather than four doubles, and one premise
// weaker: the double lane rests on every order's region-A fit being cut at the same boundaries,
// where the float table gives each order its own cover, so a fixed x selects a different piece per
// lane and the offset is not a stride. What the lanes share is the DEGREE the group is summed at.

/// Fills out[l] with F_l(x) for l = 0..nmax in single precision, eight orders to
/// a vector, with the eight coefficient bases gathered one instruction per step.
///
/// The packed lane's own entry, at the scheme and the route the policy named and
/// at the degree the multiplier's effective-degree table cuts each fit to. It is
/// this lane's region-A body, and it covers region A alone: past kX0 the
/// engine's own entry runs the certified scalar single lane one order at a time,
/// exactly as the double lane's does past its own interval.
///
/// \param scheme    which summation of the stored fits to use
/// \param route     which family's region-A fits the lane reads
/// \param nmax      highest order, 0..kMaxBoysOrder
/// \param x         argument; the fits cover 0 <= x < kX0
/// \param out       nmax + 1 values, contiguous
/// \pre out holds nmax + 1 floats, and x >= 0
void BoysAllOrdersF32Simd(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept;

/// The same entry with the other fetch: the eight bases composed into the
/// register from eight loads rather than gathered with one instruction.
///
/// Same lane, same tables, same arithmetic, same values; the two differ in the
/// instruction the fetch costs and in nothing else, which the lane's own test
/// asserts. Which is cheaper is a property of the machine, so both are here,
/// the benchmark measures them, and the engine's own entry takes the one this
/// lane's counters prefer rather than the double lane's answer.
///
/// \param scheme see BoysAllOrdersF32Simd
/// \param route  see BoysAllOrdersF32Simd
/// \param nmax   see BoysAllOrdersF32Simd
/// \param x      see BoysAllOrdersF32Simd
/// \param out    see BoysAllOrdersF32Simd
void BoysAllOrdersF32SimdComposed(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept;

// The two entries above are the reference rung's reading, where every fit is read whole. A relaxed
// rung is the same lane reading the same fits at the degrees a truncation criterion certifies - a
// table rather than a body - so the rung reaches the lane through the pair the public orders axis
// dispatches to: boys_orders_lanes.hpp's BoysAllOrdersPacked and BoysAllOrdersF32Packed.

} // namespace boys::detail
