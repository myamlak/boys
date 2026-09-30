#pragma once

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_effective_degrees.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

/// \file
/// The region-A transform lane: the per-order Chebyshev fits of the double
/// lane's region A, evaluated for a batch of arguments as one matrix product
/// per band, with the arithmetic mode chosen at compile time.
///
/// Region A's double table is two bands that every order shares, so one band's
/// fits are a coefficient matrix \c C of shape (orders × degrees) and the basis
/// is a matrix \c T of shape (degrees × batch); the lane evaluates \c F = C·T.
///
/// **This is an alternative lane, not a replacement.** No existing entry, path,
/// region boundary, table or bound changes because this lane exists.
///
/// **What is verified on this machine is the arithmetic, not the hardware.** The
/// modes below \c kFp64 evaluate a *model* of an fp32 accumulator, which is not
/// what a card's fused sum does. The two split modes' floor is 1.28 to 1.30 times
/// the float lane's 1.5e-7, so they miss that lane's budget at \c m = 1 and carry
/// it from \c m = 2.
///
/// A caller who feeds one order's value to the shipped downward recursion
/// instead is held to a tighter requirement: that recursion carries order N's
/// error up to order 0 with gain \c w(N,x), so the seed must be accurate to the
/// budget divided by \c w(N,x), and \c w peaks at 1.04e5 (order 12, the band's
/// right end). At that end \c kFp64 may seed at orders 0..2 and 25..32 and
/// nowhere between; the two split modes may seed at no order at all.
///
/// ## What the lane does not do
///
/// It does not evaluate regions B or C, and it does not evaluate the extended
/// band; it takes region-A arguments only. It does not use the recursion between
/// orders. It carries the double lane's coefficient table and no second table.

namespace boys {

/// The two halves of region A: the bands the double table's fits are shared
/// over. \c kA1 is the lower interval, \c kA2 the upper one; they are adjacent
/// with no gap.
///
/// An argument outside the named band is outside the lane.
///
/// \ingroup boys
enum class RegionABand : int {
    kA1 = 0, ///< [0, kRegionA1Edge)
    kA2, ///< [kRegionA1Edge, kRegionAEnd)
};

/// The upper end of the lower band, and the join between the two.
///
/// \ingroup boys
inline constexpr double kRegionA1Edge = 5.94992407605424223;

/// The upper end of region A: the argument at which the per-order fits stop and
/// the rest of the library takes over.
///
/// \ingroup boys
inline constexpr double kRegionAEnd = 11.899848152108484;

/// The arithmetic mode of the region-A transform, as a compile-time parameter.
///
/// \ingroup boys
enum class ProductMode : int {
    /// fp64 operands, fp64 accumulate. The double lane's arithmetic in the
    /// product's shape; the only mode that reaches the double lane's region-A
    /// budget.
    kFp64 = 0,
    /// An fp32 operand carried as two tf32 parts, three products, fp32
    /// accumulate: 3xTF32.
    kTf32x3,
    /// An fp32 operand carried as three bf16 parts, six products, fp32
    /// accumulate.
    kBf16x6,
    /// One tf32 operand, one product, fp32 accumulate: the single-pass mode a
    /// tensor core computes without a split.
    kTf32,
    /// One bf16 operand, one product, fp32 accumulate.
    kBf16,
    /// One fp16 operand, one product, fp32 accumulate.
    kFp16,
};

/// What a mode's bound is a measurement of. **Not a quality ranking**: both
/// classes are measured.
///
/// \ingroup boys
enum class ModeCertification : std::uint8_t {
    /// The bound was measured by the arithmetic the mode names, on a machine
    /// that runs it. An fp64 accumulator is an ordinary IEEE double sum, which
    /// is what makes the fp64 mode's bound a certification.
    kCertified = 0,

    /// The bound is arithmetic on a model of an accumulator this machine does
    /// not have. It is a valid statement about the model and not about a card.
    /// A green sweep of it is not evidence that a card is inside it.
    kUncertified,
};

/// One region-A mode as a report states it.
///
/// \ingroup boys
struct ProductModeInfo {
    ProductMode mode; ///< the mode this row describes
    const char* name; ///< the name a report prints for the mode
    ModeCertification certification; ///< what the row's bound is a measurement of
    const char* model; ///< the arithmetic or accumulator the bound is a claim about
    int parts; ///< products per operand: 1, 2 or 3
    double fitTerm; ///< the multiplier-scaled term of the bound: m * fitTerm
    double floor; ///< the multiplier-independent floor the format adds
    double delivered; ///< swept worst |F̂ − F| over region A at the reference multiplier
};

/// The region-A transform's arithmetic modes this build carries, one row each,
/// with the bound and the certification of each.
///
/// The rows are in the enumeration's order. A row's bound over region A at
/// multiplier \c m is `m * fitTerm + floor`.
///
/// \returns the modes, with their bounds and their certifications
///
/// \ingroup boys
std::span<const ProductModeInfo> BoysProductModes() noexcept;

/// F_0(x[i]) through F_nmax(x[i]) for a batch of arguments, region A, evaluated
/// as one matrix product per band in the named mode's arithmetic.
///
/// The arguments are one band's: the caller states which, and every argument
/// must lie in it. The coefficient matrix is the band's, so a batch that mixes
/// the bands is two calls; the entry does not sort, group, classify or fall back.
///
/// A caller can check the precondition without knowing the tables: membership is
/// \c x < kRegionA1Edge for kA1 and \c kRegionA1Edge <= x < kRegionAEnd for kA2.
///
/// Layout: order-major planes, out[k * count + i] = F_k(x[i]) - all F_0
/// contiguous, then all F_1, and so on, the layout the all-orders batch entry
/// on the rest of this surface uses. The output holds count * (nmax + 1)
/// doubles.
///
/// Accuracy: the named mode's own bound, a per-(order, argument) bound over the
/// whole band. The modes below \c kFp64 are an idealisation of an fp32 tensor
/// core's accumulator.
///
/// Threading: single-threaded and pure, like every entry in this library. The
/// entry allocates nothing and touches no shared state.
///
/// \tparam kMode               the mode whose arithmetic the product is
///                             evaluated in; see ProductMode
/// \tparam kAccuracyMultiplier see BoysSingle. The multiplier truncates the
///                             band's fits to a width every order in the band
///                             admits. It is inert for the two split modes
///                             across the documented range and live for kFp64.
/// \param band  which band every argument of the call lies in
/// \param nmax  highest order, 0..kMaxBoysOrder
/// \param x     array of count arguments, each inside `band`
/// \param out   receives count * (nmax + 1) doubles, out[k * count + i] = F_k(x[i])
/// \param count number of arguments; may be 0 (no writes)
///
/// \pre x and out must hold count and count * (nmax + 1) doubles respectively,
///      and must not overlap. Every x[i] must lie in `band`.
///
/// \ingroup boys
template <ProductMode kMode, double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
void BoysRegionAProduct(RegionABand band,
                        int nmax,
                        const double* x,
                        double* out,
                        std::size_t count) noexcept;

/// \cond
// Explicit instantiations of the entry above at the default multiplier, hidden
// from the API reference: a default call site links against these instead of
// compiling the kernel in its own translation unit.
extern template void BoysRegionAProduct<ProductMode::kFp64, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kTf32x3, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kBf16x6, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kTf32, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kBf16, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kFp16, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

// The kernel behind the entry declared above, defined here so that every
// multiplier a caller names is instantiable at the call site.
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
template <int kOperandBitsIn, int kCarryBitsIn, int kPartsIn, typename AccumulateIn>
struct ProductPolicy {
    static constexpr int kOperandBits = kOperandBitsIn;
    static constexpr int kCarryBits = kCarryBitsIn;
    static constexpr int kParts = kPartsIn;

    /// The mode's accumulate format. A mode's operand format is narrower than
    /// its accumulate format by construction, so the parts are representable in
    /// it exactly.
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
/// the 3xTF32 emulation of an fp32 product on a tf32 tensor core
/// ([OotomoYokota2022]).
using Tf32x3Policy = ProductPolicy<11, 24, 2, float>;

/// An fp32 operand carried as three bf16 parts, six products, fp32 accumulate:
/// the bfloat16 emulation of the same product.
using Bf16x6Policy = ProductPolicy<8, 24, 3, float>;

/// One tf32 operand, one product, fp32 accumulate: the mode a tensor core
/// computes without a split. The operand arrives as fp32 and is rounded once to
/// the format, which is what Split does with one part.
using Tf32Policy = ProductPolicy<11, 24, 1, float>;

/// One bf16 operand, one product, fp32 accumulate.
using Bf16Policy = ProductPolicy<8, 24, 1, float>;

/// One fp16 operand, one product, fp32 accumulate.
///
/// The same arithmetic as Tf32Policy over region A: both formats carry an
/// 11-bit significand and the band's values are well inside binary16's exponent
/// range, so every product rounds identically. The distinct name is what tells
/// a report which card instruction ran, not a distinct precision.
using Fp16Policy = ProductPolicy<11, 24, 1, float>;

/// The policy of a mode enumerator. The primary template is the refusal, not a
/// default: a mode the specializations below do not name has no policy.
template <ProductMode kMode> struct PolicyOf {
    static_assert(kMode == ProductMode::kFp64 || kMode == ProductMode::kTf32x3 ||
                      kMode == ProductMode::kBf16x6,
                  "a product mode outside the ProductMode enumeration is not a mode this "
                  "library carries: name ProductMode::kFp64, ProductMode::kTf32x3 or "
                  "ProductMode::kBf16x6");
};

template <> struct PolicyOf<ProductMode::kFp64> {
    using Type = Fp64Policy;
};

template <> struct PolicyOf<ProductMode::kTf32x3> {
    using Type = Tf32x3Policy;
};

template <> struct PolicyOf<ProductMode::kBf16x6> {
    using Type = Bf16x6Policy;
};

template <> struct PolicyOf<ProductMode::kTf32> {
    using Type = Tf32Policy;
};

template <> struct PolicyOf<ProductMode::kBf16> {
    using Type = Bf16Policy;
};

template <> struct PolicyOf<ProductMode::kFp16> {
    using Type = Fp16Policy;
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

/// Region A's double table is two bands every order shares, which is what makes
/// one band's per-order fits a single coefficient matrix.
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

/// The band's degree at a multiplier. One product carries every order at one
/// width, so the width is the smallest d' for which the dropped tail of every
/// order in the band is within the relaxation budget. Amplification is one: the
/// lane returns each order's own fit, so no seed error re-enters a recursion.
///
/// The scan walks the same evaluation domain as the shipped degree tables
/// (0, 1, 2, then even degrees).
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
/// The batch tile. The result does not depend on it - no accumulator is ever
/// split across tiles - so it bounds only the stack the entry uses.
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
    // by every argument of the batch.
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

        // The basis matrix's operand parts and the mapped arguments. The
        // recurrence below is always fp64; only its values are rounded to the
        // mode's operand format.
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

        // A width of 1 has no second Chebyshev value.
        if constexpr (kWidth > 1)
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
                // MulSub rounds the product and then the difference whatever
                // the target contracts: the two roundings this lane's delivered
                // figure is measured at, not the fused one.
                const double next =
                    backend::ScalarFp64::MulSub(2.0 * mapped[s], upper[s], lower[s]);
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
                // sum among themselves before any of them meets the scale of the
                // high ones.
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

/// \endcond
} // namespace boys
