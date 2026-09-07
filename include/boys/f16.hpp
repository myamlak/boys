#pragma once

/// \file f16.hpp
/// The fp16 I/O types of the certified mixed-precision Boys lane.
///
/// C++23 `stdfloat` defines the optional extended floating-point types
/// std::float16_t / std::bfloat16_t only on toolchains that implement them
/// (GCC 13+, Clang 17+). The MSVC STL ships `stdfloat` but declares no
/// extended types ("We don't support any optional extended floating-point
/// types" — 14.51 header text), so this library supplies the
/// self-contained F16/Bf16
/// wrappers there: pure I/O types over the IEEE-754 binary16 / bfloat16 bit
/// patterns. No third-party half library, and no arithmetic outside the
/// engine's float domain — the half value only crosses the lane boundary.

#include <bit>
#include <cstdint>

#if __has_include(<stdfloat>)
#include <stdfloat>
#endif

namespace boys {

#if defined(__cpp_lib_stdfloat)
/// The fp16 I/O type of the Boys half lane: the C++23 extended type
/// std::float16_t where the toolchain ships it, else the F16 wrapper below.
using F16 = std::float16_t;
/// The bf16 I/O type of the Boys half lane (std::bfloat16_t or Bf16).
using Bf16 = std::bfloat16_t;
#else

/// Self-contained IEEE-754 binary16 I/O type (the MSVC fallback; see the
/// file doc). Round-to-nearest-even conversions, explicit widening,
/// comparisons — nothing else.
///
/// \ingroup boys
class F16 {
public:
    /// Value-initializes to 0.0.
    constexpr F16() noexcept = default;

    /// Round-to-nearest-even conversion from float.
    ///
    /// \param value The float to convert.
    constexpr F16(float value) noexcept : _mBits(FloatToHalf(value)) {}

    /// The widened value (exact: binary16 is a subset of binary32).
    ///
    /// \returns The exact float value.
    constexpr explicit operator float() const noexcept {
        return HalfToFloat(_mBits);
    }

    /// The widened value in double (exact).
    ///
    /// \returns The exact double value.
    constexpr explicit operator double() const noexcept {
        return static_cast<double>(HalfToFloat(_mBits));
    }

    /// The raw IEEE-754 binary16 bit pattern.
    ///
    /// \returns The bit pattern.
    constexpr std::uint16_t Bits() const noexcept {
        return _mBits;
    }

    /// F16 from a raw bit pattern (the inverse of Bits()).
    ///
    /// \param bits The bit pattern.
    /// \returns The F16 value with that pattern.
    static constexpr F16 FromBits(std::uint16_t bits) noexcept {
        return F16(bits, FromBitsTag{});
    }

    /// Equality against another half value, compared as exact widened
    /// floats (no rounding involved in the comparison).
    ///
    /// \param other The value to compare with.
    /// \returns true when the values are equal.
    constexpr bool operator==(const F16& other) const noexcept {
        return static_cast<float>(*this) == static_cast<float>(other);
    }

    /// Inequality against another half value (the negation of operator==).
    ///
    /// \param other The value to compare with.
    /// \returns true when the values differ.
    constexpr bool operator!=(const F16& other) const noexcept {
        return !(*this == other);
    }

    /// Less-than against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is smaller.
    constexpr bool operator<(const F16& other) const noexcept {
        return static_cast<float>(*this) < static_cast<float>(other);
    }

    /// Less-or-equal against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is not greater.
    constexpr bool operator<=(const F16& other) const noexcept {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }

    /// Greater-than against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is larger.
    constexpr bool operator>(const F16& other) const noexcept {
        return static_cast<float>(*this) > static_cast<float>(other);
    }

    /// Greater-or-equal against another half value, compared as exact
    /// widened floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is not smaller.
    constexpr bool operator>=(const F16& other) const noexcept {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }

private:
    struct FromBitsTag {};

    constexpr F16(std::uint16_t bits, FromBitsTag) noexcept : _mBits(bits) {}

    // float -> binary16, round-to-nearest-even (the F16C conversion
    // semantics). Verified against the F16C instruction on the reference
    // cases: 1.0 -> 0x3C00, 2^-24 -> 0x0001, the 2^-25 tie -> 0, 65504 ->
    // 0x7BFF, 65520 and above -> Inf.
    static constexpr std::uint16_t FloatToHalf(float f) noexcept {
        const std::uint32_t b = std::bit_cast<std::uint32_t>(f);
        const std::uint32_t sign = (b >> 16) & 0x8000u;
        const std::uint32_t a = b & 0x7FFFFFFFu;

        if (a >= 0x7F800000u)
        {
            // Inf or NaN: keep the payload bits [22:13] and set the quiet
            // bit (0x200) for NaN (F16C's cvtps_ph behavior, mirrored so the
            // wrapper and the SIMD lane write identical patterns). Inf keeps
            // the zero mantissa.
            const std::uint32_t mant = a == 0x7F800000u ? 0u : (((a >> 13) & 0x3FFu) | 0x200u);
            return static_cast<std::uint16_t>(sign | 0x7C00u | mant);
        }

        if (a >= 0x477FF000u)
        {
            // |f| >= 65520: rounds to Inf (the half maximum is 65504; 65520
            // is the RNE threshold between 65504 and Inf).
            return static_cast<std::uint16_t>(sign | 0x7C00u);
        }

        if (a >= 0x38800000u)
        {
            // Normal half (|f| >= 2^-14): RNE at bit 13; the exponent field
            // converts in place (half bias 15 vs float bias 127, 112 << 23).
            return static_cast<std::uint16_t>(
                sign | ((a - 0x38000000u + 0x0FFFu + ((a >> 13) & 1u)) >> 13));
        }

        if (a < 0x33000000u)
        {
            // |f| < 2^-25: rounds to zero (the 2^-25 tie rounds to even,
            // which is zero).
            return static_cast<std::uint16_t>(sign);
        }
        // Subnormal half: |f| in [2^-25, 2^-14); M = round(2^24 * |f|) in
        // [0, 1023] (1024 carries into the smallest normal, 0x0400).
        const std::uint32_t shift = 126u - (a >> 23); // in [14, 24]
        const std::uint32_t significand = 0x800000u | (a & 0x7FFFFFu);
        const std::uint32_t m =
            (significand + (1u << (shift - 1u)) - 1u + ((significand >> shift) & 1u)) >> shift;
        return static_cast<std::uint16_t>(sign | m);
    }

    // binary16 -> float, exact (subnormals normalize into float subnormals).
    static constexpr float HalfToFloat(std::uint16_t h) noexcept {
        const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
        const std::uint32_t exp = (h >> 10) & 0x1Fu;
        const std::uint32_t man = h & 0x3FFu;
        std::uint32_t bits;

        if (exp == 0x1Fu)
        {
            bits = sign | 0x7F800000u | (man << 13); // Inf (man == 0) or NaN
        } else if (exp == 0u)
        {
            if (man == 0u)
            {
                bits = sign; // +/-0
            } else
            {
                // Subnormal: value = man * 2^-24. Shift the mantissa up to
                // the implicit-1 position (bit 10), tracking the exponent.
                std::uint32_t m = man;
                int e = -24;

                while ((m & 0x400u) == 0u)
                {
                    m <<= 1;
                    --e;
                }
                // value = 1.xxx * 2^(e + 10)
                bits = sign | (static_cast<std::uint32_t>(e + 137) << 23) | ((m & 0x3FFu) << 13);
            }
        } else
        {
            bits = sign | ((exp + 112u) << 23) | (man << 13);
        }

        return std::bit_cast<float>(bits);
    }

    std::uint16_t _mBits = 0;
};

/// Self-contained bfloat16 I/O type (the MSVC fallback; see the file doc).
/// Same contract as F16 with the 8-bit exponent / 7-bit mantissa layout.
///
/// \ingroup boys
class Bf16 {
public:
    /// Value-initializes to 0.0.
    constexpr Bf16() noexcept = default;

    /// Round-to-nearest-even conversion from float.
    ///
    /// \param value The float to convert.
    constexpr Bf16(float value) noexcept : _mBits(FloatToBf16(value)) {}

    /// The widened value (exact: bf16 is a subset of binary32).
    ///
    /// \returns The exact float value.
    constexpr explicit operator float() const noexcept {
        return Bf16ToFloat(_mBits);
    }

    /// The widened value in double (exact).
    ///
    /// \returns The exact double value.
    constexpr explicit operator double() const noexcept {
        return static_cast<double>(Bf16ToFloat(_mBits));
    }

    /// The raw bf16 bit pattern.
    ///
    /// \returns The bit pattern.
    constexpr std::uint16_t Bits() const noexcept {
        return _mBits;
    }

    /// Bf16 from a raw bit pattern (the inverse of Bits()).
    ///
    /// \param bits The bit pattern.
    /// \returns The Bf16 value with that pattern.
    static constexpr Bf16 FromBits(std::uint16_t bits) noexcept {
        return Bf16(bits, FromBitsTag{});
    }

    /// Equality against another half value, compared as exact widened
    /// floats (no rounding involved in the comparison).
    ///
    /// \param other The value to compare with.
    /// \returns true when the values are equal.
    constexpr bool operator==(const Bf16& other) const noexcept {
        return static_cast<float>(*this) == static_cast<float>(other);
    }

    /// Inequality against another half value (the negation of operator==).
    ///
    /// \param other The value to compare with.
    /// \returns true when the values differ.
    constexpr bool operator!=(const Bf16& other) const noexcept {
        return !(*this == other);
    }

    /// Less-than against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is smaller.
    constexpr bool operator<(const Bf16& other) const noexcept {
        return static_cast<float>(*this) < static_cast<float>(other);
    }

    /// Less-or-equal against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is not greater.
    constexpr bool operator<=(const Bf16& other) const noexcept {
        return static_cast<float>(*this) <= static_cast<float>(other);
    }

    /// Greater-than against another half value, compared as exact widened
    /// floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is larger.
    constexpr bool operator>(const Bf16& other) const noexcept {
        return static_cast<float>(*this) > static_cast<float>(other);
    }

    /// Greater-or-equal against another half value, compared as exact
    /// widened floats.
    ///
    /// \param other The value to compare with.
    /// \returns true when this value is not smaller.
    constexpr bool operator>=(const Bf16& other) const noexcept {
        return static_cast<float>(*this) >= static_cast<float>(other);
    }

private:
    struct FromBitsTag {};

    constexpr Bf16(std::uint16_t bits, FromBitsTag) noexcept : _mBits(bits) {}

    // float -> bf16, round-to-nearest-even (the AVX2 bit trick the SIMD
    // lane uses; Inf/NaN and the subnormal range fall out of the same
    // rounding). Verified: 1.0 -> 0x3F80, -1.0 -> 0xBF80, 0x7F7FFFFF ->
    // 0x7F80 (RNE rounds up to Inf), 0x7F800000 -> 0x7F80.
    static constexpr std::uint16_t FloatToBf16(float f) noexcept {
        const std::uint32_t b = std::bit_cast<std::uint32_t>(f);
        return static_cast<std::uint16_t>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
    }

    // bf16 -> float, exact: the 16-bit pattern occupies the top half of the
    // binary32 word.
    static constexpr float Bf16ToFloat(std::uint16_t h) noexcept {
        return std::bit_cast<float>(static_cast<std::uint32_t>(h) << 16);
    }

    std::uint16_t _mBits = 0;
};
#endif // defined(__cpp_lib_stdfloat)

namespace detail {
/// Bit-pattern access that works for both the stdfloat aliases and the
/// self-contained wrappers.
///
/// \param x The half value to inspect.
/// \returns The raw bit pattern.
constexpr std::uint16_t F16Bits(F16 x) noexcept {
#if defined(__cpp_lib_stdfloat)
    return std::bit_cast<std::uint16_t>(x);
#else
    return x.Bits();
#endif
}

/// The inverse of F16Bits: a half value from a raw bit pattern.
///
/// \param bits The raw bit pattern.
/// \returns The half value with that pattern.
constexpr F16 F16FromBits(std::uint16_t bits) noexcept {
#if defined(__cpp_lib_stdfloat)
    return std::bit_cast<F16>(bits);
#else
    return F16::FromBits(bits);
#endif
}

/// Bit-pattern access for bf16 (same contract as F16Bits).
///
/// \param x The half value to inspect.
/// \returns The raw bit pattern.
constexpr std::uint16_t Bf16Bits(Bf16 x) noexcept {
#if defined(__cpp_lib_stdfloat)
    return std::bit_cast<std::uint16_t>(x);
#else
    return x.Bits();
#endif
}

/// The inverse of Bf16Bits: a half value from a raw bit pattern.
///
/// \param bits The raw bit pattern.
/// \returns The half value with that pattern.
constexpr Bf16 Bf16FromBits(std::uint16_t bits) noexcept {
#if defined(__cpp_lib_stdfloat)
    return std::bit_cast<Bf16>(bits);
#else
    return Bf16::FromBits(bits);
#endif
}
} // namespace detail

/// The next representable half value above x for x >= 0 (the raw-bits
/// increment; it moves toward zero for negative x), widened to float. The Boys
/// accuracy tests use NextUp(x) - float(x) as the half-ULP representation
/// term of the fp16 tolerance (the fp16 output quantizes at ~1e-3 near x = 0,
/// far above any 1e-7 absolute assertion).
///
/// \param x The value to get the successor of.
/// \returns The successor of x, widened to float.
/// \ingroup boys
constexpr float NextUp(F16 x) noexcept {
    return static_cast<float>(detail::F16FromBits(detail::F16Bits(x) + 1));
}

/// The next representable bf16 value above x for x >= 0 (the raw-bits
/// increment; it moves toward zero for negative x), widened to float (same ULP
/// contract as NextUp(F16)).
///
/// \param x The value to get the successor of.
/// \returns The successor of x, widened to float.
/// \ingroup boys
constexpr float NextUp(Bf16 x) noexcept {
    return static_cast<float>(detail::Bf16FromBits(detail::Bf16Bits(x) + 1));
}

} // namespace boys
