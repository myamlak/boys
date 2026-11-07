// The baseline tier of a dispatch build: every default entry, compiled at the portable scalar
// tier and named by this tier's token.
//
// CMake compiles this unit without the vector tier's flags and defines BOYS_ISA_TIER_UNIT, so the
// token derives to isa_baseline and the guard registers the tier as carried. Each entry is one
// call to the public template at the documented default row: the body is therefore the same body
// the header defines for a compile-time build, instantiated here at this tier's flags and given
// this tier's symbol name. BOYS_NOINLINE keeps it a real function in this unit rather than a
// candidate the compiler may fold into its neighbour.

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
