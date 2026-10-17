#include "boys_orders_simd.hpp"

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
// The shipped double partition and the float partitions' whole route space: this unit prebuilds
// the orders lanes for both schemes and every route the tables offer, and the coarsest forms sit
// outside the default set.
#include "boys/boys_coefficients_d_coarse_cheb.hpp"
#include "boys/boys_coefficients_d_coarse_mono.hpp"
#include "boys/boys_coefficients_f32_coarse_cheb.hpp"
#include "boys/boys_coefficients_f32_coarse_mono.hpp"
#include "boys/boys_coefficients_f32_coarse_rat.hpp"
// The scaled far table: this unit prebuilds the far ladders' fused leg below, so it names the
// representation `boys/boys.hpp`'s default set does not open.
#include "boys/boys_coefficients_d_far.hpp"
#include "boys/kernels.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>

// --- Architecture guard -----------------------------------------------------
// As boys_simd.cpp, which carries the argument: BOYS_SIMD_X86 answers whether this TU is an x86_64
// target, BOYS_ISA_AVX2_FMA whether the flags compile the vector tier in; the lanes below need
// both, and else this TU holds the scalar twins a build without the vector tier gets.
#ifdef BOYS_SIMD_X86


#elif defined(__x86_64__) || defined(_M_X64)

#define BOYS_SIMD_X86 1

#else

#error "BOYS_SIMD_X86 is unset and this is not x86_64: define it (1 on x86_64, 0 elsewhere)"

#endif

// The packed arithmetic backends this TU's kernels are written against.
#include "boys_backend_simd.hpp"

#if BOYS_SIMD_X86 && BOYS_ISA_AVX2_FMA

#include <immintrin.h>
namespace boys::detail {

// The closure these entries sum from - the coefficient fetch, the group geometry, the region-A
// bodies, the zero in closed form, the fallback past the fits' interval and the lane at a division
// form - lives on the installed path (boys/boys_orders_lanes.hpp), where a consumer's flags
// and policy reach it; what stays here is the reference the option probe measures it against.

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimdAtForm<false, kDefaultDivisionForm>(scheme, nmax, x, out, stride);
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimdAtForm<true, kDefaultDivisionForm>(scheme, nmax, x, out, stride);
}

void BoysAllOrdersF32Simd(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if (!F32OrdersShortcut(nmax, x, out))
    {
        F32OrdersByRoute<backend::detail::kSelectedRoute, false>(scheme, route, nmax, x, out);
    }
}

void BoysAllOrdersF32SimdComposed(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if (!F32OrdersShortcut(nmax, x, out))
    {
        F32OrdersByRoute<backend::detail::kSelectedRoute, true>(scheme, route, nmax, x, out);
    }
}

} // namespace boys::detail

#else // BOYS_SIMD_X86 && BOYS_ISA_AVX2_FMA

// A target that compiles no vector tier: the entries exist and are defined against the certified
// scalar lanes, exactly as the region entries of boys_simd.cpp are. The tier is a question about
// the flags and not about the processor, so this arm serves a non-x86 target and an x86 one built
// without -mavx2 -mfma alike.
namespace boys::detail {

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);
    assert(stride >= 1);

    static_cast<void>(scheme);

    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<>(l, x);
    }
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimd(scheme, nmax, x, out, stride);
}

// The float lane's measurement entries, for the reason above: the vector tier
// is absent, so the lane is the fits it would have vectorised, one order at a
// time.
void BoysAllOrdersF32Simd(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    static_cast<void>(scheme);
    static_cast<void>(route);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<>(l, x);
    }
}

void BoysAllOrdersF32SimdComposed(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    BoysAllOrdersF32Simd(scheme, route, nmax, x, out);
}

} // namespace boys::detail

#endif // BOYS_SIMD_X86 && BOYS_ISA_AVX2_FMA

namespace boys::detail {

// The far region's two ladders at the fused leg: this unit is licensed for the fused instruction
// set, ClenshawSplit's multiply-add being the fused one at every form. The leg is in the name
// (LadderLeg), so a unit compiles the leg it names and the asking leg's copies come from units
// without the fused flag. Measured, one shared definition cost 28.84 instructions per order.
template void CorrectedSeedLadder<DivisionForm::kExactDivision, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;
template void CorrectedSeedLadder<DivisionForm::kPlainReciprocal, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;
template void CorrectedSeedLadder<DivisionForm::kRefinedReciprocal, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;

template void ScaledFarLadder<DivisionForm::kExactDivision, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;
template void ScaledFarLadder<DivisionForm::kPlainReciprocal, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;
template void ScaledFarLadder<DivisionForm::kRefinedReciprocal, LadderLeg::kFusedLeg>(
    int, double, double*) noexcept;

} // namespace boys::detail
