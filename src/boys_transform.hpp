#pragma once

// The template definitions behind the region-A transform lane
// (boys/boys_transform.hpp). Internal header: the entries are instantiated in
// boys_transform.cpp.
//
// Region A's double table is two bands shared by all 33 orders - every order
// has exactly one piece in each band, at the same degree - so one band's
// per-order fits are a coefficient matrix C of shape (orders x degrees) and the
// basis is a matrix T of shape (degrees x batch) whose columns are the
// Chebyshev values at the mapped arguments. Evaluating the fits for a batch of
// arguments is then the product F = C . T, which is what this lane computes.
// The recursion between orders is a chain of one multiply and one divide per
// step, not a product, and is not part of this lane.
//
// A mode names three separate facts, because the measurements behind these
// bounds show they are separate:
//
//   * the format each operand is rounded to (the "carry" format the value is
//     held in, and the narrower format its parts are stored in);
//   * the format the products are accumulated in;
//   * the order the products are reduced in.
//
// Of the first two, the accumulate format is the binding one: with an fp32
// accumulator no operand precision reaches the double lane, and splitting an
// fp64 operand into tf32 parts gives the identical error at every split depth
// while the accumulator stays fp32. A mode that needs the double lane needs an
// fp64 accumulator, which means an fp64 tensor core; no split reaches it.
//
// The reduction order is worth more than a factor of three on its own, at the
// same formats. Reducing from the highest degree down, rather than from the
// constant term up, and summing the degrees of one pass pairwise rather than
// into one running total, is what recovers the shipped split Clenshaw's own
// error from the same coefficients - the small coefficients are summed before
// the running total reaches their scale, instead of being dropped below its
// last bit. The software chooses both: permuting the columns of C and the rows
// of T by the same permutation leaves the product unchanged, so the direction
// of the degree loop is a layout decision, and pairwise summation is the
// reduction a blocked degree loop pays for anyway. Neither is a property of the
// card, and the bounds in the public header are measured in the order below.

#include "boys/boys_transform.hpp"
#include "boys_coefficients.hpp"
#include "boys_effective_degrees.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>

namespace boys {
namespace detail {

// ---------------------------------------------------------------------------
// Operand formats
// ---------------------------------------------------------------------------
/// Round v to `bits` significant bits (round-half-to-even): the significand the
/// format would store. frexp and ldexp are exact and nearbyint is a single
/// rounding, so this is correctly rounded at every exponent.
///
/// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (v, bits) reads naturally.
inline double RoundSignificand(double v, int bits) noexcept {
    int exponent = 0;
    const double mantissa = std::frexp(v, &exponent);
    const double scale = std::ldexp(1.0, bits);
    return std::ldexp(std::nearbyint(mantissa * scale) / scale, exponent);
}

/// One mode's arithmetic. `Value` is the type operands and products are carried
/// in - double for fp64, float for the split modes, whose parts are
/// float-representable by construction. `Split` fills the operand's parts in
/// the carry format, each part itself rounded to the operand format; `Combine`
/// is one reduction step, rounded once into the accumulate format.
///
/// A further mode is a further policy and one enumerator; no call site changes.
template <int kOperandBitsIn, int kCarryBitsIn, int kPartsIn, typename AccumulateIn>
struct ProductPolicy {
    static constexpr int kOperandBits = kOperandBitsIn;
    static constexpr int kCarryBits = kCarryBitsIn;
    static constexpr int kParts = kPartsIn;

    /// The mode's accumulate format. It carries the operand parts as well as
    /// the running total, because a mode's operand format is narrower than its
    /// accumulate format by construction: the parts of a split value are
    /// representable in the accumulate format exactly.
    using Value = AccumulateIn;

    /// The operand's parts: the carry format first, then the operand format.
    static void Split(double v, Value* parts) noexcept {
        double rest = (kCarryBits >= 53) ? v : RoundSignificand(v, kCarryBits);

        for (int i = 0; i < kParts; ++i)
        {
            const double part = (kOperandBits >= 53) ? rest : RoundSignificand(rest, kOperandBits);
            parts[i] = static_cast<Value>(part);
            rest -= part;
        }
    }

    /// One reduction step: one rounding into the accumulate format. The sum is
    /// exact in the carrier for the split modes' operand widths, so this is the
    /// single rounding a fused multiply-add would make.
    static Value Combine(const Value& a, const Value& b) noexcept {
        return static_cast<Value>(a + b);
    }
};

/// fp64 operands, fp64 accumulate, one product per degree: the double lane's
/// own arithmetic, in the product's shape.
using Fp64Policy = ProductPolicy<53, 53, 1, double>;

/// An fp32 operand carried as two tf32 parts, three products, fp32 accumulate:
/// the 3xTF32 emulation of an fp32 product on a tf32 tensor core.
using Tf32x3Policy = ProductPolicy<11, 24, 2, float>;

/// An fp32 operand carried as three bf16 parts, six products, fp32 accumulate:
/// the bfloat16 emulation of the same product.
using Bf16x6Policy = ProductPolicy<8, 24, 3, float>;

/// The policy of a mode enumerator.
template <ProductMode kMode> struct PolicyOf;

template <> struct PolicyOf<ProductMode::kFp64> {
    using Type = Fp64Policy;
};

template <> struct PolicyOf<ProductMode::kTf32x3> {
    using Type = Tf32x3Policy;
};

template <> struct PolicyOf<ProductMode::kBf16x6> {
    using Type = Bf16x6Policy;
};

// ---------------------------------------------------------------------------
// The two bands
// ---------------------------------------------------------------------------
/// One band of region A: the interval its fits' mapped argument runs over and
/// the degree every order's fit in it has.
struct BandSpec {
    double a;
    double b;
    int degree;
};

/// The band's description, taken from the table itself rather than restated.
template <int kIndex> constexpr BandSpec BandOf() noexcept {
    const OrderPiece& piece = kPieces[static_cast<std::size_t>(kPieceStart[0] + kIndex)];
    return BandSpec{piece.a, piece.b, piece.deg};
}

/// The flat coefficient offset of one order's fit in one band.
template <int kIndex> constexpr std::size_t BandOffset(int order) noexcept {
    return static_cast<std::size_t>(kPieces[static_cast<std::size_t>(kPieceStart[order] + kIndex)]
                                        .offset);
}

/// Region A's double table is two bands shared by all orders - that is what
/// makes its per-order fits one coefficient matrix per band, and so what makes
/// this lane a product at all. A table that stops being that shape is a
/// different lane's table, not a silently different product.
constexpr bool SharedBandTable() noexcept {
    const OrderPiece& first = kPieces[static_cast<std::size_t>(kPieceStart[0])];
    const OrderPiece& second = kPieces[static_cast<std::size_t>(kPieceStart[0] + 1)];

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        if (kPieceStart[order + 1] - kPieceStart[order] != 2)
        {
            return false;
        }

        const OrderPiece& lower = kPieces[static_cast<std::size_t>(kPieceStart[order])];
        const OrderPiece& upper = kPieces[static_cast<std::size_t>(kPieceStart[order] + 1)];

        if (lower.a != first.a || lower.b != first.b || lower.deg != first.deg)
        {
            return false;
        }

        if (upper.a != second.a || upper.b != second.b || upper.deg != second.deg)
        {
            return false;
        }
    }

    return true;
}

static_assert(SharedBandTable(),
              "the region-A transform lane needs the double table's two shared bands: one piece "
              "per band per order, all orders at the band's degree");

/// The widest band degree, i.e. the coefficient matrix's width before any
/// relaxation.
inline constexpr int kMaxBandDegree = BandOf<0>().degree > BandOf<1>().degree
                                          ? BandOf<0>().degree
                                          : BandOf<1>().degree;

/// The band's degree at a multiplier. Unlike the per-order relaxed lanes, one
/// product carries every order at one width, so the width is the widest
/// truncation every order admits: the smallest d' for which the dropped tail of
/// EVERY order in the band is within the relaxation budget. The amplification
/// is one - this lane returns each order's own fit directly, so no seed error
/// re-enters a recursion to be amplified.
///
/// The scan walks the same evaluation domain as the shipped degree tables
/// (0, 1, 2, then even degrees), so a relaxed product relaxes to a degree the
/// rest of the library also uses.
template <int kIndex> constexpr int BandDegreeAtMultiplier(double m) noexcept {
    constexpr BandSpec spec = BandOf<kIndex>();

    for (int dPrime = 0; dPrime <= spec.degree; ++dPrime)
    {
        if (dPrime == 3 || (dPrime > 2 && dPrime % 2 != 0))
        {
            continue; // outside the evaluator's domain
        }

        bool admissible = true;

        for (int order = 0; order <= kMaxOrder && admissible; ++order)
        {
            admissible = CoefficientTail(kCoeffs, BandOffset<kIndex>(order), spec.degree, dPrime) <=
                         (m - 1.0) * RegionABudget(BoysRole::kDoubleSingle);
        }

        if (admissible)
        {
            return dPrime;
        }
    }

    return spec.degree;
}

// ---------------------------------------------------------------------------
// The product
// ---------------------------------------------------------------------------
/// The batch tile. The result does not depend on it - every (order, argument)
/// pair owns its accumulator and no accumulator is ever split across tiles - so
/// it is a working-set decision only: it bounds the stack the entry uses.
inline constexpr std::size_t kProductTile = 32;

/// The product for one band: out[k * count + i] = F_k(x[i]), every argument in
/// the band, by C . T in the mode's arithmetic.
template <int kIndex, typename Policy, double kAccuracyMultiplier>
void RegionAProductBand(int nmax, const double* x, double* out, std::size_t count) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");

    using Value = typename Policy::Value;
    constexpr int kParts = Policy::kParts;
    constexpr BandSpec kBand = BandOf<kIndex>();
    constexpr int kWidth = BandDegreeAtMultiplier<kIndex>(kAccuracyMultiplier) + 1;
    constexpr int kOrders = kMaxOrder + 1;

    // The kept products: every part pairing whose part indices sum below the
    // part count, in part order - the split emulation's own quadrature.
    int pairs[kParts * kParts][2] = {};
    int pairCount = 0;

    for (int i = 0; i < kParts; ++i)
    {
        for (int j = 0; j < kParts; ++j)
        {
            if (i + j <= kParts - 1)
            {
                pairs[pairCount][0] = i;
                pairs[pairCount][1] = j;
                ++pairCount;
            }
        }
    }

    // The coefficient matrix's parts: one matrix per order and degree, shared
    // by every argument of the batch, which is what makes the work a product
    // rather than a set of independent dot products.
    Value coeffs[kParts][kOrders][kWidth];

    for (int order = 0; order <= nmax; ++order)
    {
        const double* c = kCoeffs.data() + BandOffset<kIndex>(order);

        for (int k = 0; k < kWidth; ++k)
        {
            Value parts[kParts];
            Policy::Split(c[k], parts);

            for (int p = 0; p < kParts; ++p)
            {
                coeffs[p][order][k] = parts[p];
            }
        }
    }

    for (std::size_t base = 0; base < count; base += kProductTile)
    {
        const std::size_t n = std::min(kProductTile, count - base);

        // The basis matrix's operand parts, and the mapped arguments the
        // recurrence runs on. The recurrence itself is always fp64 - it is the
        // reference the shipped split Clenshaw evaluates - and only its values
        // are rounded to the mode's operand format on the way into the product.
        Value basis[kParts][kWidth][kProductTile];
        double mapped[kProductTile];
        double lower[kProductTile];
        double upper[kProductTile];

        for (std::size_t s = 0; s < n; ++s)
        {
            const double xi = x[base + s];
            assert(xi >= kBand.a && xi < kBand.b);
            mapped[s] = 2.0 * (xi - kBand.a) / (kBand.b - kBand.a) - 1.0;
            lower[s] = 1.0;
            Value parts[kParts];
            Policy::Split(1.0, parts);

            for (int p = 0; p < kParts; ++p)
            {
                basis[p][0][s] = parts[p];
            }
        }

        if (kWidth > 1)
        {
            for (std::size_t s = 0; s < n; ++s)
            {
                upper[s] = mapped[s];
                Value parts[kParts];
                Policy::Split(upper[s], parts);

                for (int p = 0; p < kParts; ++p)
                {
                    basis[p][1][s] = parts[p];
                }
            }
        }

        for (int k = 2; k < kWidth; ++k)
        {
            for (std::size_t s = 0; s < n; ++s)
            {
                const double next = 2.0 * mapped[s] * upper[s] - lower[s];
                lower[s] = upper[s];
                upper[s] = next;
                Value parts[kParts];
                Policy::Split(next, parts);

                for (int p = 0; p < kParts; ++p)
                {
                    basis[p][k][s] = parts[p];
                }
            }
        }

        Value pass[kWidth][kProductTile];
        Value acc[kOrders][kProductTile];

        for (int order = 0; order <= nmax; ++order)
        {
            for (std::size_t s = 0; s < n; ++s)
            {
                acc[order][s] = Value{0};
            }
        }

        for (int pair = 0; pair < pairCount; ++pair)
        {
            const int left = pairs[pair][0];
            const int right = pairs[pair][1];

            for (int order = 0; order <= nmax; ++order)
            {
                // One pass: this pair's products for one order, the degree laid
                // out from the highest down.
                for (int k = 0; k < kWidth; ++k)
                {
                    const int degree = kWidth - 1 - k;
                    const Value a = coeffs[left][order][degree];

                    for (std::size_t s = 0; s < n; ++s)
                    {
                        pass[k][s] = static_cast<Value>(a * basis[right][degree][s]);
                    }
                }

                // ... and reduced by pairwise summation: the low coefficients
                // are summed among themselves before any of them meets the
                // scale of the high ones, so they never fall below the running
                // total's last bit. This is the second half of what the degree
                // loop's direction starts: at the same formats, the other
                // direction and a single running total cost up to three times
                // the error. Both halves are the software's to choose -
                // permuting the columns of the coefficient matrix and the rows
                // of the basis by the same permutation leaves the product
                // unchanged - so this is a layout decision and not a property
                // of the card.
                for (int width = kWidth; width > 1; width = (width + 1) / 2)
                {
                    for (int k = 0; 2 * k + 1 < width; ++k)
                    {
                        for (std::size_t s = 0; s < n; ++s)
                        {
                            pass[k][s] = Policy::Combine(pass[2 * k][s], pass[2 * k + 1][s]);
                        }
                    }

                    if (width % 2 != 0)
                    {
                        for (std::size_t s = 0; s < n; ++s)
                        {
                            pass[width / 2][s] = pass[width - 1][s];
                        }
                    }
                }

                for (std::size_t s = 0; s < n; ++s)
                {
                    acc[order][s] = Policy::Combine(acc[order][s], pass[0][s]);
                }
            }
        }

        for (int order = 0; order <= nmax; ++order)
        {
            for (std::size_t s = 0; s < n; ++s)
            {
                out[static_cast<std::size_t>(order) * count + base + s] =
                    static_cast<double>(acc[order][s]);
            }
        }
    }
}

} // namespace detail

template <ProductMode kMode, double kAccuracyMultiplier>
void BoysRegionAProduct(RegionABand band, int nmax, const double* x, double* out,
                        std::size_t count) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x != nullptr || count == 0);
    assert(out != nullptr || count == 0);

    using Policy = typename detail::PolicyOf<kMode>::Type;

    if (band == RegionABand::kA2)
    {
        detail::RegionAProductBand<1, Policy, kAccuracyMultiplier>(nmax, x, out, count);
    } else
    {
        detail::RegionAProductBand<0, Policy, kAccuracyMultiplier>(nmax, x, out, count);
    }
}

} // namespace boys
