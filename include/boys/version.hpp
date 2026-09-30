#pragma once

/// \file
/// The library's version, as a caller can query it.
///
/// The number is written down once, in `project(boys VERSION ...)` in the
/// top-level CMakeLists.txt; this header carries it to a caller who did not
/// build through CMake, or who checks at run time which release it linked
/// against. A test asserts the two agree.

namespace boys {

/// Major version: changes only with an incompatible change to the public
/// surface or the supported domains.
inline constexpr int kVersionMajor = 3;

/// Minor version: a compatible change, which may change bitwise outputs.
inline constexpr int kVersionMinor = 0;

/// Patch version: a compatible change that does not.
inline constexpr int kVersionPatch = 0;

/// The version as a string, "MAJOR.MINOR.PATCH".
inline constexpr const char* kVersionString = "3.0.0";

/// The library's version, as a caller queries it.
///
/// \returns the version as a string, "MAJOR.MINOR.PATCH" — the same value
///          \c kVersionString holds, so it needs no allocation and stays valid
///          for the life of the program
inline constexpr const char* VersionString() noexcept
{
    return kVersionString;
}

} // namespace boys
