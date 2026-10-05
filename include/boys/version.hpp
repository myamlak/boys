#pragma once

/// \file
/// The library's version, and the identity of the defaults seam it was built
/// with, as a caller can query them.
///
/// The number is written down once, in `project(boys VERSION ...)` in the
/// top-level CMakeLists.txt; this header carries it to a caller who did not
/// build through CMake, or who checks at run time which release it linked
/// against. A test asserts the two agree.
///
/// The seam's identity is the build's rather than this header's: a build
/// configured with `BOYS_BUILD_DEFAULTS` is what hashed that file, and a build
/// that replaced nothing carries no hash at all. Both are stated below.

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

/// The seam this build's unnamed calls resolve through: which file supplied the
/// seven choices — the host lane's five, the fit route, the evaluation scheme,
/// the packing axis, the division form and the fit granularity, and the device
/// lane's own division form and region-B exponential — as a caller can query it.
///
/// A build configured with the `BOYS_BUILD_DEFAULTS` CMake option hashed the
/// file the option named and reports that sha256, 64 lowercase hexadecimal
/// digits. A build that replaced nothing reports
/// \c "the committed header (the shipped choices)", which is the file every
/// bound in this repository was measured with.
///
/// **The digest is a content identity, and that is what makes it comparable**:
/// two builds that compiled the same seam file report the same string on any
/// machine, whatever configured them, so a consumer whose regression baseline
/// moved can first ask whether the library it linked is the one the baseline was
/// taken with. The seam file carries the machine, the date and the probe figures
/// its choices were measured on in its own comments (the contract in
/// `boys/boys_build_defaults.hpp`), so two seams that share a digest share those
/// too.
///
/// What the digest does not carry is the choices themselves. It identifies a
/// seam; it does not describe one, and a caller holding a digest of a file it
/// does not have learns that the seam moved and not where. The choices a build
/// resolves are the `boys-defaults` command's report, class by class, and the
/// seam's own seven are what that report's header prints beside them.
///
/// The value is a compile-time constant, fixed when the build was configured: a
/// seam file edited after that is a configure that has to run again, which the
/// copy the targets read is a dependency of. A unit compiled against an installed
/// `include/` tree by hand, without the library target's definitions, carries no
/// digest and reports the committed identity — the words describe the unit that
/// reported them, and a build that replaced its seam is one that carries the
/// digest the target defines.
inline constexpr const char* kBuildDefaultsSeamIdentity =
#if defined(BOYS_BUILD_DEFAULTS_SEAM_SHA256)
    BOYS_BUILD_DEFAULTS_SEAM_SHA256;
#else
    "the committed header (the shipped choices)";
#endif

/// The identity of the defaults seam this build resolved its unnamed calls
/// through.
///
/// \returns the seam's identity as a string — the sha256 of the header
///          `BOYS_BUILD_DEFAULTS` named, or
///          \c "the committed header (the shipped choices)" — which needs no
///          allocation and stays valid for the life of the program
///
/// \sa kBuildDefaultsSeamIdentity, which is the value this returns and states
///     what it does and does not carry
inline constexpr const char* BuildDefaultsSeamIdentity() noexcept
{
    return kBuildDefaultsSeamIdentity;
}

} // namespace boys
