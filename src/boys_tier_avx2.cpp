// The AVX2 + FMA tier of a dispatch build: every default entry, compiled at the vector tier and
// named by this tier's token.
//
// CMake compiles this unit with the vector tier's flags and defines BOYS_ISA_TIER_UNIT, so the
// token derives to isa_avx2 and the guard registers the tier as carried. Each entry is one call
// to the public template at the documented default row: the body is therefore the same body the
// header defines for a compile-time build, instantiated here at this tier's flags and given this
// tier's symbol name. BOYS_NOINLINE keeps it a real function in this unit rather than a candidate
// the compiler may fold into its neighbour.
//
// The contraction pin is part of the flag set and is restated by CMake on this source: the vector
// tier's flags license a bare product-plus-add to fuse, and that licence is not this tier's to
// spend (see the flag block in CMakeLists.txt).

#include "boys/boys.hpp"

#include <cstddef>

namespace boys {

namespace BOYS_DETAIL_ISA_NAMESPACE {

namespace kernel {

#define BOYS_TIER_ENTRY(RETURN, NAME, PARAMS, ARGS, PRECISION, SHAPE) \
    BOYS_NOINLINE RETURN NAME PARAMS noexcept                         \
    {                                                                 \
        return boys::NAME<DefaultPolicy<PRECISION, SHAPE>> ARGS;      \
    }

#include "boys/detail/isa_tier_entries.hpp"

} // namespace kernel

} // namespace BOYS_DETAIL_ISA_NAMESPACE

} // namespace boys
