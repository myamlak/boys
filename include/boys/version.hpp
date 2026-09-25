#pragma once

/// \file
/// The library's version, as a caller can query it.
///
/// The number itself is written down in one place, `project(boys VERSION ...)`
/// in the top-level CMakeLists.txt; this header carries it to a caller who did
/// not build through CMake, or who needs to check at run time which release it
/// linked against. A test compiled against the CMake value asserts the two
/// agree, so a release that bumps one and not the other fails its own build
/// rather than shipping two answers.

namespace boys {

/// Major version: changes only with an incompatible change to the public
/// surface or the supported domains.
inline constexpr int kVersionMajor = 2;

/// Minor version: a compatible change, which may change bitwise outputs.
inline constexpr int kVersionMinor = 0;

/// Patch version: a compatible change that does not.
inline constexpr int kVersionPatch = 0;

/// The version as a string, "MAJOR.MINOR.PATCH".
inline constexpr const char* kVersionString = "2.0.0";

/// The library's version, as a caller queries it.
///
/// \returns the version as a string, "MAJOR.MINOR.PATCH" — e.g. "2.0.0".
inline constexpr const char* VersionString() noexcept
{
    return kVersionString;
}

} // namespace boys
