// The AVX2 + FMA tier of a dispatch build: every default entry, compiled at the vector tier and
// named by its token. CMake compiles this unit at the vector flags and defines BOYS_ISA_TIER_UNIT;
// each entry is one call to the public template at the documented default row, instantiated here
// at this tier's flags, and BOYS_NOINLINE keeps it a real function in this unit.

#include "boys/boys.hpp"

#include <cstddef>

namespace boys {

inline namespace BOYS_DETAIL_ISA_NAMESPACE {

namespace kernel {

#define BOYS_TIER_ENTRY(RETURN, NAME, PARAMS, ARGS, PRECISION, SHAPE) \
    BOYS_NOINLINE RETURN NAME PARAMS noexcept                         \
    {                                                                 \
        return boys::NAME<DefaultPolicy<PRECISION, SHAPE>> ARGS;      \
    }

#include "boys_tier_entries.hpp"

} // namespace kernel

} // namespace BOYS_DETAIL_ISA_NAMESPACE

} // namespace boys
