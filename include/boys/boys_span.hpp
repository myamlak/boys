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
/// **`BoysFixedN`'s stride is spelled `BoysStride{}` here.** The pointer entry's
/// fourth argument is the count and this overload's count comes from the argument
/// span, so the stride is a tag rather than a bare number: a count written in that
/// position is a compile error instead of a stride no size-check can see.
///
/// \ingroup boys

#include "boys/boys.hpp"
#include "boys/kernels.hpp"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <span>

namespace boys {

namespace detail {

/// Whether \p outSize values admit \p count values written \p stride apart: the
/// last of them lands at `(count - 1) * stride`, and the question is asked as a
/// division so that a product which would wrap the size type is refused rather
/// than admitted.
///
/// \param count   the values the call writes
/// \param stride  the distance between them
/// \param outSize the values the out span holds
///
/// \returns true when every value the write lands is inside the span, false when
///          one falls past its end
inline bool StridedSpanAdmits(std::size_t count, std::size_t stride,
                              std::size_t outSize) noexcept {
    if (count == 0)
    {
        return true;
    }

    if (outSize == 0)
    {
        return false;
    }

    return stride == 0 || (count - 1) <= (outSize - 1) / stride;
}

/// The extent \p count values written \p stride apart ask of the out span,
/// saturating at the largest representable size so a refusal can name a bound
/// where the true product does not fit one.
///
/// \param count  the values the call writes
/// \param stride the distance between them
///
/// \returns the out-span extent the write asks, saturated at the largest
///          representable size
inline std::size_t StridedSpanNeeded(std::size_t count, std::size_t stride) noexcept {
    if (count == 0)
    {
        return 0;
    }

    const std::size_t kMax = (std::numeric_limits<std::size_t>::max)();

    if (stride != 0 && count - 1 > (kMax - 1) / stride)
    {
        return kMax;
    }

    return (count - 1) * stride + 1;
}

/// Whether an out span of \p outSize values admits the `count * planes` values
/// a plane-shaped call writes; asked as a division so a product that would wrap
/// the size type is refused rather than admitted.
///
/// \param count   the arguments the call walks
/// \param planes  the value planes one argument writes
/// \param outSize the values the out span holds
///
/// \returns true when the span holds `count * planes` values, false otherwise
inline bool PlanesAdmit(std::size_t count, std::size_t planes,
                        std::size_t outSize) noexcept {
    if (count == 0 || planes == 0)
    {
        return true;
    }

    return count <= outSize / planes;
}

/// The extent `count * planes` values ask of the out span, saturating at the
/// largest representable size so a refusal can name a bound where the true
/// product does not fit one.
///
/// \param count  the arguments the call walks
/// \param planes the value planes one argument writes
///
/// \returns the out-span extent the call asks, saturated at the largest
///          representable size
inline std::size_t PlanesNeeded(std::size_t count, std::size_t planes) noexcept {
    const std::size_t kMax = (std::numeric_limits<std::size_t>::max)();

    if (count == 0 || planes == 0)
    {
        return 0;
    }

    if (planes > kMax / count)
    {
        return kMax;
    }

    return count * planes;
}

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
/// \param out    receives the values, `(x.size() - 1) * stride.value + 1` of them; a
///               shorter span is refused with a line naming both extents, and so
///               is one whose stride would carry the last value past the end of
///               the size type
/// \param stride the distance between the elements the values are written to,
///               spelled through \c BoysStride{} because the count comes from
///               \p x here; a bare number in this position is a compile error
///
/// \ingroup boys
template <EvalPolicyLike Policy = DefaultPolicy<Precision::kFp64, Shape::kFixedN>>
inline void BoysFixedN(int n,
                       std::span<const double> x,
                       std::span<double> out,
                       BoysStride stride = {}) noexcept {
    if (!detail::StridedSpanAdmits(x.size(), stride.value, out.size()))
    {
        detail::RefuseShortSpan("BoysFixedN", "out",
                                detail::StridedSpanNeeded(x.size(), stride.value), out.size());
    }

    BoysFixedN<Policy>(n, x.data(), out.data(), x.size(), stride.value);
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
    const std::size_t kMax = (std::numeric_limits<std::size_t>::max)();
    const std::size_t orders = nmax < 0 ? kMax : static_cast<std::size_t>(nmax) + 1;

    if (!detail::PlanesAdmit(x.size(), orders, out.size()))
    {
        detail::RefuseShortSpan(
            "BoysAllN", "out", detail::PlanesNeeded(x.size(), orders), out.size());
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

    const std::size_t kMax = (std::numeric_limits<std::size_t>::max)();
    const std::size_t orders = top == kMax ? kMax : top + 1;

    if (!detail::PlanesAdmit(x.size(), orders, out.size()))
    {
        detail::RefuseShortSpan(
            "BoysAllNAtOrders", "out", detail::PlanesNeeded(x.size(), orders), out.size());
    }

    BoysAllNAtOrders<Policy>(n.data(), x.data(), out.data(), x.size());
}

} // namespace boys
