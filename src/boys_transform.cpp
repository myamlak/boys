#include "boys/boys_transform.hpp"

// The entry's explicit instantiations. The extern-template declarations in the
// public header route a default call site here, so a caller that names no mode
// links against these instead of compiling the kernel again in its own
// translation unit.

namespace boys {

template void BoysRegionAProduct<ProductMode::kTf32>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kFp16>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

std::span<const ProductModeInfo> BoysProductModes() noexcept {
    // The bounds and the floors are the header's preamble figures, and the
    // certification column says which of them a card can be held to. The
    // single-pass floors are worst measurements, rounded up.
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

template void BoysRegionAProduct<ProductMode::kFp64>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kTf32x3>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
template void BoysRegionAProduct<ProductMode::kBf16x6>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

} // namespace boys
