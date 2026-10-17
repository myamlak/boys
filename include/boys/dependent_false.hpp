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

/// The group a coefficient marker read names, carried on a parameter of the template that
/// reads it.
///
/// A group header specializes the marker, and a read whose template argument does not depend
/// on the reading template is resolved where that template is written: a group header included
/// afterwards would then specialize a marker already read, which MSVC refuses (C2908). Passing
/// the group through this alias keeps the read dependent, so it is resolved where the reading
/// template is instantiated and the caller's includes have been seen.
template <typename, typename Group>
using DependentGroup = Group;

} // namespace boys::detail

/// \endcond
