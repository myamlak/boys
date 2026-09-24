#pragma once

// The across-orders packed lane: one argument's orders in a single pass, four
// orders to a vector.
//
// It is the companion of the across-arguments lane in boys_simd.cpp, not a
// replacement for it. That lane vectorises four ARGUMENTS of one order, which
// is the shape BoysFixedN and the plane entry's region body ask for; this one
// vectorises four ORDERS of one argument, which is the shape
// BoysAllOrders(nmax, x, out) asks for. On that shape the across-arguments
// lane has no argument to put in a vector and runs at width one, while the
// across-orders axis is nmax + 1 wide.
//
// Private to the library, like the packed backends and for the same reason:
// the intrinsics' header is not part of the public surface. The lane itself is
// reachable from outside through the entries that carry the orders axis
// (PackAxis::kOrders), which dispatch to BoysAllOrdersPacked below.
//
// Region A only (argument below kX0), where the per-order piecewise fits are
// stored. The other two regions evaluate one seed and reach every order from
// it by recursion, which is one fit for all of them; the fits are the cost
// this lane exists to spread over a vector.

#include <cstddef>

namespace boys::detail {

/// How the across-orders lane sums a stored fit.
///
/// The axis and the scheme are separate choices, and the lane offers the
/// combinations so that the two can be told apart by measurement: every entry
/// here sums the SAME stored fits, in the same vector lanes, and differs only
/// in the summation.
enum class OrdersScheme {
    /// The certified scheme, summed exactly as boys_impl.hpp's ClenshawSplit
    /// sums it, with the four orders' coefficients gathered a step at a time.
    kSplitClenshaw,
    /// The direct sum: the Chebyshev series term by term, T_k by the forward
    /// recurrence. Two multiply-adds per coefficient against the split
    /// Clenshaw's one, and no dependence between the terms.
    kDirectSum,
    /// Horner over the monomial form of the same fit, the second certified
    /// scheme: one multiply-add per stored coefficient.
    kHorner,
};

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

// The entry the public orders axis dispatches to, BoysAllOrdersPacked, is
// declared in boys/boys_impl.hpp with the other internal entries and defined in
// this translation unit: one declaration, where the entries that call it are.

} // namespace boys::detail
