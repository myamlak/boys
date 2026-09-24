#pragma once

// The seam between the two translation units that own arithmetic.
//
// The library compiles its arithmetic in more than one flag context: the SIMD
// unit carries the packed flags, and everything else — the scalar kernels and
// the consumer — is compiled without them. A contraction fact belongs to one
// of those contexts, so the table of backends is built by asking each one where
// its arithmetic is compiled. The scalar entries are written where the scalar
// kernels live (src/boys.cpp); the packed entries are appended from here, and
// this header is the only thing the two units share.

#include "boys/backend.hpp"

#include <cstddef>

namespace boys::backend {
namespace detail {

/// Writes the packed arithmetic backends this build carries to `out`, which
/// must have room for two entries.
///
/// The SIMD translation unit answers, because the packed arithmetic is
/// compiled there and nowhere else; a build without the tier writes none.
///
/// \param out where the entries are written
///
/// \returns how many entries were written, 0 or 2
std::size_t AppendPackedBackends(BackendInfo* out) noexcept;

} // namespace detail
} // namespace boys::backend
