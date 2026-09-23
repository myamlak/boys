#pragma once

/// \file
/// The two compile-time facts the rest of the library is written against: the
/// highest order the kernel serves and the multiplier m = 1 that is every
/// lane's default accuracy.
///
/// They stand in their own header because not every header that names them
/// needs the kernel: an entry that takes the multiplier as a template
/// parameter names the two constants and nothing else.

namespace boys {

/// Highest Boys order supported by the kernel.
inline constexpr int kMaxBoysOrder = 32;

/// The default accuracy multiplier of every lane: m = 1 is full static
/// accuracy, bit-identical to the certified lanes (the documented
/// contract).
/// Larger m values trade certified accuracy for work via compile-time degree
/// truncation (see the contract table in the boys/boys.hpp preamble).
inline constexpr double kBoysFullAccuracyMultiplier = 1.0;

} // namespace boys
