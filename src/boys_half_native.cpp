#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"

#include <cassert>
#include <cstddef>

// Region C's asymptotic ladder in packed correctly rounded half arithmetic: a square root and a
// divide for the seed, then a packed multiply and divide per order. The roundings are relative,
// so the error is about one half ULP carried; the running value is scaled by 2^15 to keep the
// ladder's values normal down to F_n(x) = 2^-29, and the lane has a precondition (x >= kX1).

// The lane is behind the BoysFp16 seam like the entries that declare it: both entries, the packed
// type they take and return, and the scale constant they document leave the public surface
// together, so a build with the seam closed has neither entry to define and this file adds nothing.
#if BoysFp16

namespace boys {
namespace detail {

// The seed (1/2)sqrt(pi/x) is one divide by the square root, so the constant is
// sqrt(pi)/2 rounded to half once here and exact from then on.
constexpr F16 kHalfNativeSeedConstant = F16FromDouble(kBoysHalfSqrtPi);

// The per-order scale: the largest power of two binary16 holds.
constexpr F16 kHalfNativeScaleConstant = F16FromDouble(32768.0); // 2^15

/// F_0(x)..F_nmax(x) for one packed pair of arguments, scaled by 2^15.
///
/// The output is 2^15 * F_k(x) half by half; the scale is the same for every
/// order, so a caller unscales by one exact power-of-two division.
///
/// \param nmax Highest order, 0..kMaxBoysOrder.
/// \param x    Packed pair of arguments, both >= kX1.
/// \param out  Receives nmax + 1 packed pairs, out[k] = 2^15 * (F_k(x.low), F_k(x.high)).
inline void HalfNativeLadder(int nmax, Half2 x, Half2* out) noexcept {
    const Half2 seed = Half2::Broadcast(kHalfNativeSeedConstant);
    const Half2 scale = Half2::Broadcast(kHalfNativeScaleConstant);

    // Seed: F_0(x) = (1/2) sqrt(pi/x), then the exact per-order scale.
    Half2 f = Half2Mul(Half2Div(seed, Half2Sqrt(x)), scale);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        // The shipped region-C body, op for op: f = (l - 1/2) * f / x.
        const F16 factor = F16FromDouble(static_cast<double>(l) - 0.5);
        f = Half2Div(Half2Mul(Half2::Broadcast(factor), f), x);
        out[l] = f;
    }
}

} // namespace detail

void BoysAllOrdersHalf2(int nmax, Half2 x, Half2* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x.Low() >= detail::F16FromDouble(detail::kX1) &&
           x.High() >= detail::F16FromDouble(detail::kX1));
    assert(out != nullptr);

    detail::HalfNativeLadder(nmax, x, out);
}

void BoysAllNF16Native(int nmax, const F16* x, F16* out, std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x != nullptr || count == 0);
    assert(out != nullptr || count == 0);

    for (std::size_t i = 0; i < count; i += 2)
    {
        // The pair is the unit of work; a lone trailing argument rides in both
        // halves of the register and only the low half is written back.
        const bool paired = i + 1 < count;
        const Half2 arguments = Half2(x[i], paired ? x[i + 1] : x[i]);

        assert(arguments.Low() >= detail::F16FromDouble(detail::kX1) &&
               arguments.High() >= detail::F16FromDouble(detail::kX1));

        Half2 values[kMaxBoysOrder + 1];
        detail::HalfNativeLadder(nmax, arguments, values);

        for (int k = 0; k <= nmax; ++k)
        {
            out[static_cast<std::size_t>(k) * count + i] = values[k].Low();

            if (paired)
            {
                out[static_cast<std::size_t>(k) * count + i + 1] = values[k].High();
            }
        }
    }
}

} // namespace boys

#endif // BoysFp16
