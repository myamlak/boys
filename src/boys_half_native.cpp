#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"

#include <cassert>
#include <cstddef>

// --- The native half lane ----------------------------------------------------
//
// Region C's asymptotic ladder evaluated in packed correctly rounded half
// arithmetic: one square root and one divide for the seed, then one packed
// multiply and one packed divide per order. Every value the ladder touches is a
// half, every operation rounds once, and two arguments travel through the same
// register -- this is the lane f16.hpp's I/O types surround the fp32 engine
// for, at half the register traffic per value and none of the engine's
// precision.
//
// Two consequences follow from the arithmetic and are the lane's documented
// contract, both measured (tests/boys_half_native_test.cpp, and the sweep the
// bound was derived from):
//
//   * The ladder's roundings are relative, so its error is about one half ULP
//     of whatever value it carries -- an order of magnitude above the I/O
//     lane's budget, and arithmetic rather than hardware: two correctly rounded
//     roundings per order land at one ULP where that budget allows half of one.
//   * The running value is scaled by 2^15, the largest power of two binary16
//     holds. The scale is exact, so it buys no accuracy; it buys *range*. It is
//     what keeps the ladder's values in the format's normal range (11
//     significand bits) and not merely representable (one subnormal bit at a
//     time) down to F_n(x) = 2^-29, which is 2^-14 * 2^-15: the smallest normal
//     value, lifted by the largest exact scale. Past that value the lane
//     returns a subnormal and then a zero by design and claims nothing there;
//     the ceiling is the end of the claim, not a bound the format's floor
//     happens to satisfy.
//
// The span the ladder has to cross is what bounds the range from the other
// side: F_0/F_n grows like (2x)^n and the format's normal range is a factor
// 2^29 wide, so at order 8 the last arguments a scale can reach are around
// x = 36 whatever scale is chosen -- measured, and reported with the rest of
// the sweep in the test.
//
// Region C only: the tables of regions A and B are not representable in half,
// so this lane has a precondition (x >= kX1) rather than a fallback, and no
// accuracy multiplier (region C carries no truncatable resource).

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

    // Seed: F_0(x) = (1/2) sqrt(pi/x) -- one square root and one divide, each
    // correctly rounded to half -- then the exact per-order scale.
    Half2 f = Half2Mul(Half2Div(seed, Half2Sqrt(x)), scale);
    out[0] = f;

    for (int l = 1; l <= nmax; ++l)
    {
        // The shipped region-C body, op for op: f = (l - 1/2) * f / x, one
        // packed multiply and one packed divide, two roundings per order.
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
