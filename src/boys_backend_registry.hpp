#pragma once

// The seam between the two translation units that own arithmetic: the scalar
// entries are written where the scalar kernels live (src/boys.cpp) and the
// packed ones are appended from here, because a contraction fact belongs to the
// flag context its arithmetic is compiled in. This header is the only thing the
// two units share.

#include "boys/backend.hpp"

#include <cstddef>

namespace boys::backend {
namespace detail {

/// Writes the packed arithmetic backends this build carries to `out`, which
/// must have room for two entries.
///
/// \param out where the entries are written
///
/// \returns how many entries were written, 0 or 2
std::size_t AppendPackedBackends(BackendInfo* out) noexcept;

} // namespace detail
} // namespace boys::backend
