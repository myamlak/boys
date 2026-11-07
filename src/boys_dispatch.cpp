// The dispatcher of a dispatch build: the public default entries, defined once, forwarding to the
// carried tier the selection names.
//
// The unit is compiled at the baseline tier like every client unit, and defines
// BOYS_ISA_DISPATCH_DEFINE so the guard registers it as the dispatcher. The selection is one
// function-local static in ActiveIsa below: the first call reads the CPUID cache, checks that the
// image carries what it selected, and caches the answer; every later call is a load and a compare.
// The tiers' entries are called directly from each forwarding body - one call, no function
// pointer, and nothing per order or per argument.
//
// The carried tiers' entries are declared here because a translation unit's headers name only its
// own tier. isa_avx2 is declared inline to match the tier unit that defines it (the token is
// inline there); isa_baseline is this unit's own token.

#define BOYS_ISA_DISPATCH_DEFINE 1

#include "boys/boys.hpp"

#include <cstddef>

// --- The carried tiers' entry declarations -----------------------------------

namespace boys {

inline namespace isa_avx2 {

namespace kernel {

#define BOYS_TIER_ENTRY(RETURN, NAME, PARAMS, ARGS, PRECISION, SHAPE) \
    BOYS_NOINLINE RETURN NAME PARAMS noexcept;

#include "boys/detail/isa_tier_entries.hpp"

} // namespace kernel

} // namespace isa_avx2

namespace isa_baseline {

namespace kernel {

#define BOYS_TIER_ENTRY(RETURN, NAME, PARAMS, ARGS, PRECISION, SHAPE) \
    BOYS_NOINLINE RETURN NAME PARAMS noexcept;

#include "boys/detail/isa_tier_entries.hpp"

} // namespace kernel

} // namespace isa_baseline

// --- The public entries ------------------------------------------------------

inline namespace BOYS_DETAIL_SEAM_NAMESPACE {

#define BOYS_TIER_ENTRY(RETURN, NAME, PARAMS, ARGS, PRECISION, SHAPE) \
    RETURN NAME PARAMS noexcept                                       \
    {                                                                 \
        if (dispatch::ActiveIsa() == dispatch::kIsaAvx2)              \
        {                                                             \
            return isa_avx2::kernel::NAME ARGS;                       \
        }                                                             \
        return isa_baseline::kernel::NAME ARGS;                       \
    }

#include "boys/detail/isa_tier_entries.hpp"

} // namespace BOYS_DETAIL_SEAM_NAMESPACE

} // namespace boys

// --- The selection -----------------------------------------------------------

namespace {

#if BOYS_ISA_DISPATCH && BOYS_ISA_TIER_FORCING
int g_forcedIsa = -1; ///< the tier the forcing gate named, or -1 for the dispatcher's own
#endif

} // namespace

namespace boys::dispatch {

namespace {

/// The selection's first call. It runs once: the answer is cached by ActiveIsa.
///
/// The checks are the adjudication's start-up check. A dispatcher that is not registered means
/// this unit was not linked or its initialisation did not run; a selected tier that is not in the
/// carried set means a tier unit was not linked or was compiled without its flag set. Both abort
/// with the two profiles rather than run a body the image cannot execute.
int ResolveIsa() noexcept {
    const detail::tier_guard::Registry& registry = detail::tier_guard::Announced();

    if (!registry.dispatcher)
    {
        detail::tier_guard::NoDispatcher();
    }

    const int selected = CpuHasAvx2Fma() ? kIsaAvx2 : kIsaBaseline;

    if ((registry.carried & (1 << selected)) == 0)
    {
        detail::tier_guard::SelectedNotCarried(selected, registry.carried);
    }

    return selected;
}

} // namespace

int ActiveIsa() noexcept {
#if BOYS_ISA_DISPATCH && BOYS_ISA_TIER_FORCING
    if (g_forcedIsa >= 0)
    {
        return g_forcedIsa;
    }
#endif

    static const int selected = ResolveIsa();
    return selected;
}

} // namespace boys::dispatch

#if BOYS_ISA_DISPATCH && BOYS_ISA_TIER_FORCING
namespace boys {

bool BoysForceTier(BoysIsaTier tier) noexcept {
    const detail::tier_guard::Registry& registry = detail::tier_guard::Announced();
    const int isa = tier == BoysIsaTier::kAvx2 ? dispatch::kIsaAvx2 : dispatch::kIsaBaseline;

    if (!registry.dispatcher || (registry.carried & (1 << isa)) == 0)
    {
        return false;
    }

    g_forcedIsa = isa;
    return true;
}

void BoysClearForcedTier() noexcept {
    g_forcedIsa = -1;
}

} // namespace boys
#endif
