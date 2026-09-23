#include "boys/boys_transform.hpp"

// The region-A transform lane. The product's structure, the modes' bounds and
// the kernel itself are in the public header, which is what makes every
// multiplier a caller names instantiable at the call site: the kernel is a
// template defined in the headers, so a consumer's translation unit compiles
// the rung it uses and no fixed set of multipliers is the surface.
//
// This TU holds the default multiplier's instantiations, which the
// extern-template declarations in that header route the default call sites to:
// a caller that names no multiplier links against these instead of compiling
// the kernel again in its own translation unit.

namespace boys {

template void BoysRegionAProduct<ProductMode::kFp64, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kTf32x3, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16x6, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

} // namespace boys
