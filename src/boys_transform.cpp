#include "boys/boys_transform.hpp"

#include "boys_transform.hpp"

// The region-A transform lane. The product's structure and the modes' bounds
// are in the public header; this TU provides the default m = 1 explicit
// instantiations that the extern-template declarations route every call site
// to, so no call site instantiates the kernel and no two TUs hold a copy of it.
//
// The m > 1 rungs are not instantiated here: the lane's multiplier is a
// compile-time parameter and its relaxed widths are the caller's to choose, the
// documented surface being the three modes at full accuracy.

namespace boys {

template void BoysRegionAProduct<ProductMode::kFp64, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kTf32x3, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16x6, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

} // namespace boys
