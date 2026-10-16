#pragma once

/// \file
/// The dependent false, shared by the host implementation and the CUDA lane, so that one
/// definition decides it wherever an arm refuses a combination.
///
/// A header of its own so that neither side reaches into the other for it: the host
/// implementation carries the entry bodies, and a translation unit that includes the CUDA lane's
/// headers must not carry them.

/// \cond

namespace boys::detail {

/// A false that depends on what it is instantiated with.
///
/// An assertion on this is evaluated where the arm it stands in is instantiated and nowhere
/// else, so a fourth partition added to FitGranularity fails the build at every switch that does
/// not name it. A bare `false` would be rejected where the arm is written, which is what the
/// parameter is for.
template <typename>
inline constexpr bool kAlwaysFalse = false;

} // namespace boys::detail

/// \endcond
