#pragma once

/// \file
/// The many-argument entries taking `std::span`, so a caller passes a container
/// instead of a pointer and a count.
///
/// The library's entries take a pointer and a count, which is what a kernel
/// wants at the call and what the C surface needs. A C++ caller usually holds a
/// container instead, and this header adds an overload of each entry for that
/// case. Every one of them passes `data()` to the pointer entry, so the values
/// and the accuracy are that entry's; and every one of them compares the
/// extents it was handed against what the call writes first, refusing a span
/// that is too short rather than writing past its end. The pointer entries
/// cannot make that comparison - a bare pointer carries no size - so a caller
/// passing `p, count` still owns the buffer it hands over.
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
#include "boys/kernels.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <span>

namespace boys {

namespace detail {

/// Refuses a span shorter than the call writes: one line naming the call, the
/// span and the two extents, then an abort.
///
/// \param entry  the call, as the message names it
/// \param which  the span that is short, as the message names it
/// \param needed the values the call writes
/// \param given  the values the span holds
[[noreturn]] inline void RefuseShortSpan(const char* entry,
                                         const char* which,
                                         std::size_t needed,
                                         std::size_t given) noexcept {
    std::fprintf(stderr,
                 "boys: %s needs its %s to hold at least %zu value(s) and was given %zu; a "
                 "span shorter than the call writes is written past its end, so the call is "
                 "refused\n",
                 entry,
                 which,
                 needed,
                 given);
    std::fflush(stderr);
    std::abort();
}

/// Refuses two spans that have to run together and do not: one line naming the
/// call and the two extents, then an abort.
///
/// \param entry     the call, as the message names it
/// \param first     what the first span holds, one per element of the second
/// \param firstSize the first span's extent
/// \param second    what the second span holds
/// \param secondSize the second span's extent
[[noreturn]] inline void RefuseMismatchedSpans(const char* entry,
                                               const char* first,
                                               std::size_t firstSize,
                                               const char* second,
                                               std::size_t secondSize) noexcept {
    std::fprintf(stderr,
                 "boys: %s takes one %s per %s and was given %zu and %zu; the two run "
                 "together, so the call is refused\n",
                 entry,
                 first,
                 second,
                 firstSize,
                 secondSize);
    std::fflush(stderr);
    std::abort();
}

} // namespace detail

/// F_0(x) through F_nmax(x) written into \p out; see \c BoysAllOrders.
///
/// \param nmax the highest order
/// \param x    the argument
/// \param out  receives the ladder, `nmax + 1` values; a shorter span is
///             refused with a line naming both extents
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>
inline void BoysAllOrders(int nmax, double x, std::span<double> out) noexcept {
    const std::size_t needed = static_cast<std::size_t>(nmax) + 1;

    if (out.size() < needed)
    {
        detail::RefuseShortSpan("BoysAllOrders", "out", needed, out.size());
    }

    BoysAllOrders<Policy>(nmax, x, out.data());
}

/// F_n(x_i) for every argument written into \p out; see \c BoysFixedN.
///
/// \param n      the order
/// \param x      the arguments
/// \param out    receives the values, `(x.size() - 1) * stride + 1` of them; a
///               shorter span is refused with a line naming both extents
/// \param stride the distance between the elements the values are written to
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
inline void BoysFixedN(int n,
                       std::span<const double> x,
                       std::span<double> out,
                       std::size_t stride = 1) noexcept {
    if (x.size() >= 1)
    {
        const std::size_t needed = (x.size() - 1) * stride + 1;

        if (out.size() < needed)
        {
            detail::RefuseShortSpan("BoysFixedN", "out", needed, out.size());
        }
    }

    BoysFixedN<Policy>(n, x.data(), out.data(), x.size(), stride);
}

/// F_0(x_i) through F_nmax(x_i) for every argument; see \c BoysAllN.
///
/// \param nmax      the highest order
/// \param x         the arguments
/// \param out       receives the planes, `x.size() * (nmax + 1)` values; a
///                  shorter span is refused with a line naming both extents
/// \param workspace scratch the grouped path uses, or empty; a non-empty span
///                  shorter than \c BoysAllNWorkspaceSize(x.size()) is refused
///                  the same way
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllN>>
inline void BoysAllN(int nmax,
                     std::span<const double> x,
                     std::span<double> out,
                     std::span<std::size_t> workspace = {}) noexcept {
    const std::size_t planes = x.size() * (static_cast<std::size_t>(nmax) + 1);

    if (out.size() < planes)
    {
        detail::RefuseShortSpan("BoysAllN", "out", planes, out.size());
    }

    const std::size_t scratch = BoysAllNWorkspaceSize(x.size());

    if (!workspace.empty() && workspace.size() < scratch)
    {
        detail::RefuseShortSpan("BoysAllN", "workspace", scratch, workspace.size());
    }

    BoysAllN<Policy>(
        nmax, x.data(), out.data(), x.size(), workspace.empty() ? nullptr : workspace.data());
}

/// F_0(x_i) through F_{n[i]}(x_i), each argument at its own top order; see
/// \c BoysAllNAtOrders.
///
/// \param n   the top order of each argument, one per argument; a different
///            extent is refused with a line naming both
/// \param x   the arguments
/// \param out receives the planes, `x.size() * (1 + the largest order)` values;
///            a shorter span is refused with a line naming both extents
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>
inline void BoysAllNAtOrders(std::span<const int> n,
                             std::span<const double> x,
                             std::span<double> out) noexcept {
    if (n.size() != x.size())
    {
        detail::RefuseMismatchedSpans(
            "BoysAllNAtOrders", "top order", n.size(), "argument", x.size());
    }

    std::size_t top = 0;

    for (const int order : n)
    {
        if (static_cast<std::size_t>(order) > top)
        {
            top = static_cast<std::size_t>(order);
        }
    }

    const std::size_t planes = x.size() * (top + 1);

    if (out.size() < planes)
    {
        detail::RefuseShortSpan("BoysAllNAtOrders", "out", planes, out.size());
    }

    BoysAllNAtOrders<Policy>(n.data(), x.data(), out.data(), x.size());
}

} // namespace boys
