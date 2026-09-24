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

template void BoysRegionAProduct<ProductMode::kTf32, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kFp16, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

std::span<const ProductModeInfo> BoysProductModes() noexcept {
    // The bounds and the floors are the header's own preamble figures, and the
    // certification column is what says which of them a card can be held to:
    // the fp64 mode's bound is a measurement of an ordinary IEEE double sum,
    // and the other five are arithmetic on the fp32 accumulator model the
    // preamble describes. The single-pass floors are the dense-sweep worst
    // measurements of tools_tc/compare.py at the same density the split modes'
    // were taken at, rounded up as those are.
    static const ProductModeInfo kRows[] = {
        {ProductMode::kFp64,
         "fp64",
         ModeCertification::kCertified,
         "an fp64 accumulator: an ordinary IEEE double sum",
         1,
         1e-15,
         0.0,
         1.110e-16},
        {ProductMode::kTf32x3,
         "3xtf32",
         ModeCertification::kUncertified,
         "an fp32 accumulator, round-to-nearest with a full roll-over",
         2,
         1e-15,
         2.5e-7,
         1.916e-07},
        {ProductMode::kBf16x6,
         "6xbf16",
         ModeCertification::kUncertified,
         "an fp32 accumulator, round-to-nearest with a full roll-over",
         3,
         1e-15,
         2.5e-7,
         1.946e-07},
        {ProductMode::kTf32,
         "tf32",
         ModeCertification::kUncertified,
         "an fp32 accumulator, round-to-nearest with a full roll-over",
         1,
         1e-15,
         5e-4,
         4.4184e-04},
        {ProductMode::kBf16,
         "bf16",
         ModeCertification::kUncertified,
         "an fp32 accumulator, round-to-nearest with a full roll-over",
         1,
         1e-15,
         3e-3,
         2.7893e-03},
        {ProductMode::kFp16,
         "fp16",
         ModeCertification::kUncertified,
         "an fp32 accumulator, round-to-nearest with a full roll-over",
         1,
         1e-15,
         5e-4,
         4.4184e-04},
    };

    return std::span<const ProductModeInfo>(kRows);
}

template void BoysRegionAProduct<ProductMode::kFp64, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kTf32x3, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16x6, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

} // namespace boys
