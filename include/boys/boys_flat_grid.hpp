#pragma once

// The uniform grids' derived constants and the bounds they are held to. Every value here
// is a function of the master's metadata; the fits and the lanes read them, and no group
// header is needed to parse them.

/// \cond

#include "boys/accuracy.hpp"
#include "boys/boys_coefficients.hpp"

namespace boys {

namespace detail {

// The reciprocal is taken from the stored width, never pasted beside it: the width belongs to the
// derivation and moves when it moves, so a constant here would read the table one interval off the
// cell it was fitted on.
inline constexpr double kFlatPerUnit = 1.0 / detail::kFlatWidth;

static_assert(detail::kFlatWidth * kFlatPerUnit == 1.0,
              "the uniform table's grid is addressed by multiplying x by "
              "kFlatPerUnit and truncating, so that product must be exactly the "
              "reciprocal of the stored interval width");

static_assert(kFlatPerUnit * detail::kFlatHi ==
                  static_cast<double>(detail::kFlatIntervals),
              "the grid's own identity: the top of the table's domain scaled by the locate's "
              "constant must be exactly its interval count, so that an argument inside the "
              "domain locates no further than the last interval");

// Every interval's degree is checked against the read cap rather than assumed: the
// split Clenshaw scheme's body is written for a degree of at least 4 (0, 1 and 2
// return above it), and a degree above the cap is beyond the coefficients the
// interval stores. Parity is no part of the rule.
constexpr bool FlatDegreesCarried() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatIntervals); ++iv)
    {
        const int deg = detail::kFlatDegs[iv];

        if (deg < 4 || deg > detail::kFlatReadCap)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatDegreesCarried(),
              "every interval of the uniform table must be fitted at a degree between 4 and the "
              "read cap: below 4 the split Clenshaw summation's general body is not written, and "
              "a degree above the cap is beyond the coefficients the interval carries");

static_assert(std::size(detail::kFlatOffsets) ==
                      static_cast<std::size_t>(detail::kFlatIntervals) + 1 &&
                  detail::kFlatOffsets[detail::kFlatIntervals] ==
                      static_cast<int>(std::size(detail::kFlatCoeffs)),
              "the uniform table's offsets must run one per interval and end at the stored "
              "count: the read rule reaches an interval's last coefficient through them, so an "
              "offset past the end reads a neighbouring fit's numbers");

// The two uniformities the locator's arithmetic rests on, proved here so a regenerated grid that
// breaks either refuses at compile time rather than at a caller: every interval is fitted at the
// one degree kFlatDeg, and every offset steps by the one block kFlatBlock = (kMaxOrder + 1) *
// (kFlatDeg + 1). The offsets array keeps its readers; this is what lets the locator skip it.
inline constexpr int kFlatDeg = detail::kFlatDegs[0];
inline constexpr int kFlatBlock = (kMaxOrder + 1) * (kFlatDeg + 1);

constexpr bool FlatDegreesUniform() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatIntervals); ++iv)
    {
        if (detail::kFlatDegs[iv] != kFlatDeg)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatDegreesUniform(),
              "every interval of the uniform table must be fitted at the one degree the "
              "locator's stride and the walk's trip count are compiled from: an interval at "
              "another degree would be summed at the wrong stride into a different polynomial");

constexpr bool FlatBlocksUniform() noexcept
{
    for (std::size_t iv = 0; iv < std::size(detail::kFlatOffsets); ++iv)
    {
        if (detail::kFlatOffsets[iv] != static_cast<int>(iv) * kFlatBlock)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatBlocksUniform(),
              "the uniform table's offsets must step by kFlatBlock, the block the locator "
              "computes instead of reading them: an offset that steps otherwise would address, "
              "silently, a neighbouring interval's fit");

// t's map, 2 (u - iv) - 1, is exact from u's own bits, and no bound on the grid's top takes part
// in it: a power-of-two kFlatPerUnit of at least one moves x's significand and rounds nothing
// into u, f = u - iv is exact there, and 2 f - 1 needs a 54th bit only below u = 1/4 - the
// first cell's own quarter, where the map rounds in its last bit and no constant can prevent it.
constexpr bool FlatMapExact() noexcept
{
    double perUnit = kFlatPerUnit;

    // Refused rather than normalized: a reciprocal below one is a cell wider than the unit, whose
    // product of x and kFlatPerUnit falls subnormal below x = 2^-1022 / kFlatPerUnit and can round
    // there. A zero, a NaN or a negative reciprocal lands here too.
    if (!(perUnit >= 1.0))
    {
        return false;
    }

    while (perUnit >= 2.0)
    {
        perUnit *= 0.5;
    }

    return perUnit == 1.0;
}

static_assert(FlatMapExact(),
              "the locator's mapped argument is taken through one MulAdd, which delivers the "
              "two-rounding spelling's bits wherever the product and the fraction are exact: that "
              "needs kFlatPerUnit to be a power of two no smaller than one, so that x's "
              "significand is only moved and the product can never fall subnormal - the grid's top "
              "is no part of it, 2 * f - 1 holding at most 53 bits for every u at or above 1/4 "
              "however far the grid reaches, and rounding once below it");

// This is the lane's own grid, derived from this lane's bound, format and read cap: reading the
// double lane's under this lane's name would hand a table fitted to a bound this lane's arithmetic
// cannot reach. The width's reciprocal is exact in binary32, so the interval an argument falls in
// is a property of the argument and not of a rounding.
inline constexpr float kFlatPerUnitF32 = 1.0f / detail::f32::kFlatWidthF32;

static_assert(detail::f32::kFlatWidthF32 * kFlatPerUnitF32 == 1.0f,
              "the locate is one multiply by this constant and a truncation, so the product must "
              "be exactly the width's reciprocal in binary32: a width whose reciprocal does not "
              "round back reads the table one interval off the cell it was fitted on, silently");

static_assert(kFlatPerUnitF32 * detail::f32::kFlatHiF32 ==
                  static_cast<float>(detail::f32::kFlatIntervalsF32),
              "the grid's own identity: the top of the table's domain scaled by the locate's "
              "constant must be exactly its interval count, so that an argument inside the "
              "domain locates no further than the last interval");

// Every interval's degree is checked against the read cap rather than assumed: the
// split Clenshaw scheme seeds its odd part at c[2m-1] and so needs an even degree of
// at least 4, and a degree above the cap is beyond the coefficients the interval
// stores. A table violating either would be summed into a different polynomial.
constexpr bool FlatDegreesCarriedF32() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::f32::kFlatIntervalsF32); ++iv)
    {
        const int deg = detail::f32::kFlatDegsF32[iv];

        if (deg < 4 || deg % 2 != 0 || deg > detail::f32::kFlatReadCapF32)
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatDegreesCarriedF32(),
              "every interval of the float uniform table must be fitted at an even degree between "
              "4 and the read cap: an odd degree would drop the leading odd coefficient the split "
              "Clenshaw summation seeds from, and a degree above the cap is beyond the "
              "coefficients the interval carries");

} // namespace detail

} // namespace boys

/// \endcond
