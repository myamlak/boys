#pragma once

/// \file half2.hpp
/// The packed half-precision arithmetic type of the native half lane.
///
/// A `Half2` is two IEEE-754 binary16 values in one 32-bit register — the
/// packed shape a half-throughput consumer wants, where one operation does the
/// work of two. The lane's arithmetic is *defined* by this file: every
/// operation is correctly rounded to half, once per operation, for both halves
/// of the register, with no widening for the sequence as a whole. A lane that
/// widens to single, evaluates, and rounds back once at the end is the I/O lane
/// of f16.hpp, not this one.
///
/// The implementation is portable and toolchain-independent: each operation is
/// evaluated in binary64 and rounded once to binary16. For +, -, and * of
/// binary16 operands the binary64 result is exact (22 significand bits at
/// most), and for / and sqrt binary64 is wide enough for the single rounding to
/// be the correctly rounded one — the intermediate precision only has to reach
/// 2p + 2 = 24 bits for that to hold — so the rounded result is the same
/// correctly rounded half a packed half instruction would produce. No
/// third-party half library, and no arithmetic outside the half domain: the
/// only widening is inside one operation.
///
/// On a target whose ISA has packed binary16 arithmetic, these operations are
/// the packed instructions (`vaddph`/`vdivph`/`vsqrtph` on x86-64 with
/// AVX512-FP16, the `f16` vector forms elsewhere): one instruction per
/// operation for the register's two values. This tree carries no ISA-specific
/// backend — a branch no leg of its test matrix can execute would be a claim
/// with no artifact behind it — so the portable body above is what runs, and
/// the equivalences it rests on are what the suite verifies.

#include "boys/f16.hpp"

#include <bit>
#include <cmath>
#include <cstdint>

namespace boys {

namespace detail {

/// The smallest magnitude that rounds to Inf rather than to the half maximum:
/// the round-to-nearest-even midpoint between 65504 and 65536.
constexpr double kHalfRoundToInfinity = 65520.0;

/// binary64 -> binary16, round-to-nearest-even, by the same in-place exponent
/// conversion the F16 fallback uses (see f16.hpp): the exponent field maps with
/// a 1008 bias difference and the significand rounds at bit 42.
///
/// \param value The value to round.
/// \returns The nearest binary16 bit pattern.
constexpr std::uint16_t HalfBitsFromDouble(double value) noexcept {
    const std::uint64_t b = std::bit_cast<std::uint64_t>(value);
    const std::uint32_t sign = static_cast<std::uint32_t>((b >> 48) & 0x8000u);
    const std::uint64_t a = b & 0x7FFFFFFFFFFFFFFFull;

    if (a >= 0x7FF0000000000000ull)
    {
        // Inf or NaN; the payload bits [62:53] carry over, quiet bit set.
        const std::uint64_t mant =
            a == 0x7FF0000000000000ull ? 0ull : (((a >> 42) & 0x3FFull) | 0x200ull);
        return static_cast<std::uint16_t>(sign | 0x7C00u | mant);
    }

    if (a >= std::bit_cast<std::uint64_t>(kHalfRoundToInfinity))
    {
        // |value| >= 65520: rounds to Inf (the half maximum is 65504; 65520 is
        // the RNE threshold between 65504 and Inf).
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    }

    if (a >= 0x3F10000000000000ull)
    {
        // |value| >= 2^-14: normal half. RNE at bit 42; the exponent field
        // converts in place (half bias 15 vs double bias 1023, 1008 << 52).
        return static_cast<std::uint16_t>(
            sign | ((a - 0x3F00000000000000ull + 0x1FFFFFFFFFFull + ((a >> 42) & 1ull)) >> 42));
    }

    if (a < 0x3E60000000000000ull)
    {
        // |value| < 2^-25: rounds to zero (the 2^-25 tie rounds to even, zero).
        return static_cast<std::uint16_t>(sign);
    }

    // Subnormal half: |value| in [2^-25, 2^-14); M = round(2^24 * |value|) in
    // [0, 1023] (1024 carries into the smallest normal, 0x0400).
    const std::uint32_t shift = 1051u - static_cast<std::uint32_t>(a >> 52); // in [42, 53]
    const std::uint64_t significand = 0x0010000000000000ull | (a & 0x000FFFFFFFFFFFFFull);
    const std::uint64_t m =
        (significand + (1ull << (shift - 1u)) - 1ull + ((significand >> shift) & 1ull)) >> shift;
    return static_cast<std::uint16_t>(sign | m);
}

/// The inverse of HalfBitsFromDouble: binary16 -> binary64, exact.
///
/// \param bits The half bit pattern.
/// \returns The exact value.
constexpr double DoubleFromHalfBits(std::uint16_t bits) noexcept {
    return static_cast<double>(static_cast<float>(detail::F16FromBits(bits)));
}

/// A half from a binary64 rounding (the packed operations' one rounding).
///
/// \param value The value to round to half.
/// \returns The correctly rounded half.
constexpr F16 F16FromDouble(double value) noexcept {
    return detail::F16FromBits(detail::HalfBitsFromDouble(value));
}

} // namespace detail

/// Two binary16 values in one 32-bit register: the packed operand of the native
/// half lane.
///
/// The low half occupies bits [15:0] and the high half bits [31:16]. The type
/// is an I/O and arithmetic container only — it carries no accuracy contract of
/// its own, and the operations below are the lane's arithmetic.
///
/// \ingroup boys
class Half2 {
public:
    /// Value-initializes both halves to 0.0.
    constexpr Half2() noexcept = default;

    /// Packs two halves, `low` into bits [15:0] and `high` into bits [31:16].
    ///
    /// \param low  The half in the register's low bits.
    /// \param high The half in the register's high bits.
    constexpr Half2(F16 low, F16 high) noexcept :
        _mBits(static_cast<std::uint32_t>(detail::F16Bits(low)) |
               (static_cast<std::uint32_t>(detail::F16Bits(high)) << 16)) {}

    /// Both halves from one value (the shape a per-order constant takes).
    ///
    /// \param value The value to replicate.
    /// \returns The packed pair with both halves equal to value.
    static constexpr Half2 Broadcast(F16 value) noexcept {
        const std::uint32_t bits = detail::F16Bits(value);
        return FromBits(bits | (bits << 16));
    }

    /// The half in the register's low bits.
    ///
    /// \returns The low half.
    constexpr F16 Low() const noexcept {
        return detail::F16FromBits(static_cast<std::uint16_t>(_mBits & 0xFFFFu));
    }

    /// The half in the register's high bits.
    ///
    /// \returns The high half.
    constexpr F16 High() const noexcept {
        return detail::F16FromBits(static_cast<std::uint16_t>(_mBits >> 16));
    }

    /// The raw 32-bit pattern (low half in bits [15:0], high in [31:16]).
    ///
    /// \returns The bit pattern.
    constexpr std::uint32_t Bits() const noexcept {
        return _mBits;
    }

    /// A packed pair from a raw 32-bit pattern (the inverse of Bits()).
    ///
    /// \param bits The bit pattern.
    /// \returns The pair with that pattern.
    static constexpr Half2 FromBits(std::uint32_t bits) noexcept {
        Half2 out;
        out._mBits = bits;
        return out;
    }

    /// Equality against another packed pair, compared as exact widened floats
    /// half by half (no rounding in the comparison).
    ///
    /// \param other The pair to compare with.
    /// \returns true when both halves are equal.
    constexpr bool operator==(const Half2& other) const noexcept {
        return Low() == other.Low() && High() == other.High();
    }

    /// Inequality against another packed pair (the negation of operator==).
    ///
    /// \param other The pair to compare with.
    /// \returns true when the pairs differ in either half.
    constexpr bool operator!=(const Half2& other) const noexcept {
        return !(*this == other);
    }

private:
    std::uint32_t _mBits = 0;
};

/// The sum of two packed pairs, each half correctly rounded to half.
///
/// \param a The first operand.
/// \param b The second operand.
/// \returns The packed pair of sums.
/// \ingroup boys
inline Half2 Half2Add(Half2 a, Half2 b) noexcept {
    return Half2(detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.Low())) +
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.Low()))),
                 detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.High())) +
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.High()))));
}

/// The difference of two packed pairs, each half correctly rounded to half.
///
/// \param a The minuend.
/// \param b The subtrahend.
/// \returns The packed pair of differences.
/// \ingroup boys
inline Half2 Half2Sub(Half2 a, Half2 b) noexcept {
    return Half2(detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.Low())) -
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.Low()))),
                 detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.High())) -
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.High()))));
}

/// The product of two packed pairs, each half correctly rounded to half.
///
/// \param a The first operand.
/// \param b The second operand.
/// \returns The packed pair of products.
/// \ingroup boys
inline Half2 Half2Mul(Half2 a, Half2 b) noexcept {
    return Half2(detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.Low())) *
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.Low()))),
                 detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.High())) *
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.High()))));
}

/// The quotient of two packed pairs, each half correctly rounded to half.
///
/// \param a The dividend.
/// \param b The divisor.
/// \returns The packed pair of quotients.
/// \ingroup boys
inline Half2 Half2Div(Half2 a, Half2 b) noexcept {
    return Half2(detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.Low())) /
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.Low()))),
                 detail::F16FromDouble(detail::DoubleFromHalfBits(detail::F16Bits(a.High())) /
                                       detail::DoubleFromHalfBits(detail::F16Bits(b.High()))));
}

/// The square root of a packed pair, each half correctly rounded to half.
///
/// \param a The operand, both halves >= 0.
/// \returns The packed pair of square roots.
/// \ingroup boys
inline Half2 Half2Sqrt(Half2 a) noexcept {
    return Half2(
        detail::F16FromDouble(std::sqrt(detail::DoubleFromHalfBits(detail::F16Bits(a.Low())))),
        detail::F16FromDouble(std::sqrt(detail::DoubleFromHalfBits(detail::F16Bits(a.High())))));
}

} // namespace boys
