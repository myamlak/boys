#pragma once

/// \file
/// The many-argument entries taking `std::span`, so a caller passes a container
/// instead of a pointer and a count.
///
/// The library's entries take a pointer and a count, which is what a kernel
/// wants at the call and what the C surface needs. A C++ caller usually holds a
/// container instead, and this header adds an overload of each entry for that
/// case. Every one of them is an inline forwarder passing `data()` and `size()`
/// to the pointer entry, so the call compiles to the same code: at `-O2` both
/// spellings emit a single call to the same instantiation, which the consumer
/// check asserts rather than this comment claiming it.
///
/// `std::span` accepts any contiguous range - `std::array`, `std::vector`, a C
/// array, another span - so one overload covers them all. **Nothing existing
/// changes**: a call passing `p, count` still resolves to the pointer entry
/// exactly, because a `T*` beats a `std::span<T>` conversion, so this header is
/// additive and a caller who does not include it is unaffected.
///
/// This is a separate header rather than part of `boys/boys.hpp` so that the
/// core stays a pointer-and-count kernel: the C surface, the device lane and a
/// caller writing their own loop never compile a `std::span` they do not use.
///
/// The entries not overloaded here are the scalar ones, which take no array,
/// and the single-precision and half-precision families, which take the same
/// shapes at another type - write the same forwarder if you need one, or pass
/// `.data()` and `.size()` to the entry.
///
/// \ingroup boys

#include "boys/boys.hpp"

#include <cstddef>
#include <span>

namespace boys {

/// F_0(x) through F_nmax(x) written into \p out; see \c BoysAllOrders.
///
/// \param nmax the highest order
/// \param x    the argument
/// \param out  receives the ladder, `nmax + 1` values
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>
inline void BoysAllOrders(int nmax, double x, std::span<double> out) noexcept
{
    BoysAllOrders<Policy>(nmax, x, out.data());
}

/// F_n(x_i) for every argument written into \p out; see \c BoysFixedN.
///
/// \param n      the order
/// \param x      the arguments
/// \param out    receives the values, `x.size() * stride` of them
/// \param stride the distance between the elements the values are written to
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
inline void BoysFixedN(int n, std::span<const double> x, std::span<double> out,
                       std::size_t stride = 1) noexcept
{
    BoysFixedN<Policy>(n, x.data(), out.data(), x.size(), stride);
}

/// F_0(x_i) through F_nmax(x_i) for every argument; see \c BoysAllN.
///
/// \param nmax      the highest order
/// \param x         the arguments
/// \param out       receives the planes, `x.size() * (nmax + 1)` values
/// \param workspace scratch the grouped path uses, or empty
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
inline void BoysAllN(int nmax, std::span<const double> x, std::span<double> out,
                     std::span<std::size_t> workspace = {}) noexcept
{
    BoysAllN<Policy>(nmax, x.data(), out.data(), x.size(),
                                          workspace.empty() ? nullptr : workspace.data());
}

/// F_0(x_i) through F_{n[i]}(x_i), each argument at its own top order; see
/// \c BoysAllNAtOrders.
///
/// \param n   the top order of each argument
/// \param x   the arguments
/// \param out receives the planes, `x.size() * (1 + the largest order)` values
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>
inline void BoysAllNAtOrders(std::span<const int> n, std::span<const double> x,
                             std::span<double> out) noexcept
{
    BoysAllNAtOrders<Policy>(n.data(), x.data(), out.data(), x.size());
}

} // namespace boys
