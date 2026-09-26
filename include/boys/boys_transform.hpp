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
/// is a matrix \c T of shape (degrees × batch); the lane evaluates
/// \c F = C·T. The recursion between orders is a chain, not a product, and is
/// neither used nor changed here.
///
/// **This is an alternative lane, not a replacement.** No existing entry, path,
/// region boundary, table or bound changes because this lane exists; a caller
/// that does not name it pays nothing for it.
///
/// ## The arithmetic modes
///
/// A mode names three separate facts, and the measurements behind the bounds
/// below show they really are separate: the format each operand is rounded to,
/// the format the products are accumulated in, and the order the products are
/// reduced in.
///
/// | Mode | Operand | Accumulate | Products | Reduction |
/// |---|---|---|---|---|
/// | \c kFp64 | fp64 | fp64 | 1 | pairwise, degree high to low |
/// | \c kTf32x3 | fp32, carried as 2 tf32 parts | fp32 | 3 | pairwise, degree high to low |
/// | \c kBf16x6 | fp32, carried as 3 bf16 parts | fp32 | 6 | pairwise, degree high to low |
/// | \c kTf32 | tf32 | fp32 | 1 | pairwise, degree high to low |
/// | \c kBf16 | bf16 | fp32 | 1 | pairwise, degree high to low |
/// | \c kFp16 | fp16 | fp32 | 1 | pairwise, degree high to low |
///
/// **The accumulate format is the binding choice of the two formats.** With an
/// fp32 accumulator no operand precision reaches the double lane: splitting an
/// fp64 operand into tf32 parts gives the identical error at two parts through
/// six. Reaching the double lane's region A needs an fp64 accumulator, that is
/// an fp64 tensor core; no split reaches it.
///
/// **The reduction order is a property of the layout, not of the card.**
/// Permuting the columns of \c C and the rows of \c T by the same permutation
/// leaves the product unchanged, so the order the degree loop runs in, and
/// whether the degrees of one pass are summed in one running total or pairwise,
/// are the software's to choose. They are worth choosing: from the same
/// coefficients at the same formats, running the degrees from the constant term
/// up into one running total costs three times the error at fp64 (3.331e-16
/// against 1.110e-16) and more than three times at \c kBf16x6's format
/// (5.913e-07 against 1.946e-07). The kernel's order is the one these bounds
/// are measured in.
///
/// **No speed is claimed for any mode.** A machine with no tensor core can test
/// a mode's precision, and that is what was verified. What the modes buy is
/// that a card whose arithmetic is one of them can be used without a redesign
/// of this lane.
///
/// ## The bounds
///
/// With \c m the accuracy multiplier, over region A - both bands, every order
/// 0..32, every argument of the band - the delivered error is at most
///
/// | Mode | Bound | Delivered, worst over region A | Certification |
/// |---|---|---|---|
/// | \c kFp64 | \c m·1e-15 | 1.110e-16 | certified |
/// | \c kTf32x3 | \c m·1e-15 + 2.5e-7 | 1.916e-07 | uncertified |
/// | \c kBf16x6 | \c m·1e-15 + 2.5e-7 | 1.946e-07 | uncertified |
/// | \c kTf32 | \c m·1e-15 + 5e-4 | 4.4184e-04 | uncertified |
/// | \c kBf16 | \c m·1e-15 + 3e-3 | 2.7893e-03 | uncertified |
/// | \c kFp16 | \c m·1e-15 + 5e-4 | 4.4184e-04 | uncertified |
///
/// **The certification column is not a quality ranking and the bounds are not
/// comparable across it.** Every row is measured; the column says what the
/// measurement was of. The fp64 row's bound is a measurement of an ordinary
/// IEEE double sum, which is the arithmetic the row names, so a caller may hold
/// a result to it. The other five are arithmetic on a *model* of an fp32
/// accumulator, and a card's fused sum is not that model, so their bounds are
/// claims about the model and never about a card. `BoysProductModes` reports
/// each mode's class so a caller can act on the distinction without reading
/// this table.
///
/// The delivered figures are the worst found, against the committed 45-digit
/// reference grid and against a 200000-point sweep of the lower band's left
/// end, where these modes are worst. The 200000 is that sweep's own size and
/// not the size of the suite's: the dense-sweep test in the tree runs 20000
/// points of the same form, which is a coarser instrument and reports a
/// correspondingly lower worst, so its printed number and the figures above
/// are not the same measurement and the difference is the grid rather than
/// the lane. The \c kFp64 figure is the fitted polynomial's own truncation:
/// the product adds nothing measurable to what the coefficients already cost,
/// and it is the same figure the shipped split Clenshaw delivers. The two
/// split-mode figures are their format's floor. The sweep is what found them -
/// the grid alone samples 1.24e-07 and 1.37e-07 at those modes and understates
/// the worst by about 1.5 times.
///
/// **The multiplier does not move the ceiling, and for the two split modes it
/// does not move the bound either.** A mode's floor is its format's, which no
/// multiplier changes. The fit term \c (m−1)·1e-15 cannot reach 1.9e-07 until
/// \c m is about 1.9e8, four orders past the largest multiplier the rest of
/// this surface samples, so for \c kTf32x3 and \c kBf16x6 the bound is the
/// floor up to there and the fit term past it. For \c kFp64 the floor is
/// 1.11e-16, below the budget's own 1e-15, so its bound is the double single
/// lane's \c m·1e-15 at every \c m, and the multiplier is live from \c m = 2
/// upward as it is on every other lane.
///
/// **The two split modes miss the float lane's budget at \c m = 1 and carry it
/// from \c m = 2.** Their floor is 1.28 to 1.30 times the float lane's 1.5e-7,
/// so a caller who needs 1.5e-7 at \c m = 1 has the float or the double lane.
/// The float lane's budget scales with \c m while a format's floor does not, so
/// at \c m = 2 that lane's contract is 3e-7 and both modes are inside it with
/// about a third of it to spare. That is the whole of what these two modes
/// offer: a float-lane-grade result from a card that has no fp64 arithmetic, at
/// a bound 1.3 times looser than the float lane's at the same \c m.
///
/// **These two bounds are an idealisation, and the accumulate format is what
/// makes them one.** The models here accumulate fp32 with round-to-nearest and
/// a full roll-over, which is what an IEEE fp32 addition does and is not what a
/// tensor core's fused sum does: those truncate the addends and align them with
/// a limited number of extra bits, so an fp32-accumulate card's delivered error
/// can be worse than any figure here, in the direction that matters. The
/// idealisation therefore cannot be used to argue a mode in; it can only
/// overstate how good one is, and hardware is the only way to settle an
/// fp32-accumulate row. The \c kFp64 row is a claim about an fp64 accumulator,
/// which no machine here has either, but an fp64 accumulator is a far narrower
/// assumption than an fp32 tensor core's: it is an ordinary IEEE double sum.
/// **What is verified on this machine is the arithmetic, not the hardware.**
///
/// ## The domain, and the recursion the lane does not run
///
/// The lane returns every order's own fit, so nothing is amplified and the
/// bound above is per (order, argument) over the whole band. That is the direct
/// reading, and it is what a caller of this entry gets.
///
/// A caller who takes one order's value and feeds it to the shipped downward
/// recursion instead is held to a tighter requirement, because that recursion
/// carries order N's error up to order 0 with gain \c w(N,x), the ratio of the
/// orders' magnitudes. The seed must then be accurate to the budget divided by
/// \c w(N,x), and \c w is not monotone: it peaks at the order nearest \c x -
/// 1.04e5 at order 12 at the band's right end - and is 1 at order 32 for every
/// argument of region A. So the orders a mode may seed at are not an interval:
/// at the band's right end \c kFp64 may seed at orders 0..2 and 25..32 and
/// nowhere between, and at smaller arguments the admissible set widens. The two
/// split modes may seed at no order at all, because their floor exceeds the
/// whole budget divided by \c w wherever \c w is 1; a caller who wants the
/// recursion wants \c kFp64.
///
/// ## What the lane does not do
///
/// It does not evaluate regions B or C, and it does not evaluate the extended
/// band; it takes region-A arguments only. It does not use the recursion
/// between orders. It carries the double lane's coefficient table and no second
/// table: a mode below fp64 reuses that table at the fp32 precision its
/// operands carry, which is what the bounds above are measured against. A table
/// fitted at a narrower precision is a further table and a further lane, not a
/// mode here.

namespace boys {

/// The two halves of region A: the bands the double table's fits are shared
/// over. \c kA1 is the lower interval, \c kA2 the upper one; they are adjacent
/// with no gap.
///
/// The bands are the table's own, and the entry asserts membership: an argument
/// outside the named band is outside the lane.
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
/// A mode's bound is a statement about an arithmetic, and where that arithmetic
/// ran is what decides whether the bound is a certification. Each mode is
/// therefore reported with its certification (BoysProductModes) rather than
/// only with its bound: a certified mode's bound was measured by the arithmetic
/// the mode names, and an uncertified mode's bound is arithmetic on a model of
/// an accumulator, which is a narrower claim than a card.
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

/// What a mode's bound is a measurement of.
///
/// The distinction is not a quality ranking: both classes are measured, and
/// neither is a guess. It is a statement about what the number can be held
/// against. A card's fused sum truncates and aligns its addends, which the
/// models behind the uncertified modes do not do, so an uncertified bound can
/// be wrong in the direction that matters and only hardware can settle it.
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
/// with the bound and the certification of each, so a caller can ask what the
/// modes are and which of their bounds a card is held to without reading the
/// kernel.
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
/// must lie in it. The band is the product's precondition rather than a
/// convenience - the coefficient matrix is the band's, so a batch that mixes
/// the bands is two calls, and the caller who has both knows it. A caller whose
/// arguments are not grouped by band gets nothing from passing them anyway: the
/// entry does not sort, group, classify or fall back.
///
/// A caller can check the precondition without knowing the tables: the bands
/// are published as kRegionA1Edge and kRegionAEnd, and membership is
/// \c x < kRegionA1Edge for kA1 and \c kRegionA1Edge <= x < kRegionAEnd for kA2.
///
/// Layout: order-major planes, out[k * count + i] = F_k(x[i]) - all F_0
/// contiguous, then all F_1, and so on, the layout the all-orders batch entry
/// on the rest of this surface uses. The output holds count * (nmax + 1)
/// doubles.
///
/// Accuracy: the named mode's own bound, as in the file preamble. It is a
/// per-(order, argument) bound over the whole band, not an average and not a
/// typical case, and the two split modes' bounds are an idealisation of an
/// fp32 tensor core's accumulator - read that paragraph before relying on them.
///
/// Threading: single-threaded and pure, like every entry in this library. The
/// entry allocates nothing and touches no shared state.
///
/// \tparam kMode               the mode whose arithmetic the product is
///                             evaluated in; see ProductMode
/// \tparam kAccuracyMultiplier see BoysSingle. The multiplier truncates the
///                             band's fits to a width every order in the band
///                             admits, a-priori and never tuned. It is inert
///                             for the two split modes across the documented
///                             range and live for kFp64; see the preamble.
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
// Hidden from the API reference: each of these is an instantiation of the
// entry declared above at the default multiplier, not an entry of its own.
// They are what the default call sites link against instead of compiling the
// kernel again in their own translation unit; a caller that names any other
// multiplier compiles the rung it asks for from the definition below.
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
// multiplier a caller names is instantiable at the call site. The reference
// documents the entry; this is its implementation.
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

/// One tf32 operand, one product, fp32 accumulate: the mode a tensor core
/// computes without a split. The operand arrives as fp32 and is rounded once to
/// the format, which is what Split does with one part.
using Tf32Policy = ProductPolicy<11, 24, 1, float>;

/// One bf16 operand, one product, fp32 accumulate.
using Bf16Policy = ProductPolicy<8, 24, 1, float>;

/// One fp16 operand, one product, fp32 accumulate.
///
/// This is the same arithmetic as Tf32Policy over region A, and the
/// measurement in tools_tc/compare.py says so: both formats carry an 11-bit
/// significand, the band's operands and basis values are all well inside
/// binary16's exponent range, so the two round every product identically and
/// their measured floors agree to every digit the sweep resolves. They are
/// distinct names because a card's two instructions are distinct, and the name
/// is what tells a report which one ran; a caller choosing between them on this
/// lane is choosing an instruction, not a precision.
using Fp16Policy = ProductPolicy<11, 24, 1, float>;

/// The policy of a mode enumerator.
///
/// The primary template is the refusal, not a default: a mode the specializations
/// below do not name has no policy, and this is where that is said in a
/// sentence rather than left to an incomplete type. A mode that fell through to
/// another mode's policy would be evaluated in arithmetic it does not name and
/// would be held to a bound its own row does not state, which is the one thing
/// the mode axis must not do.
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

        // The second Chebyshev value exists only when the width has one: at a
        // width of 1 the whole block is discarded rather than branched over,
        // so the block is never the one a constant condition names.
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
                // Two roundings, stated by the arithmetic rather than left to
                // the build: MulSub rounds the product and then the
                // difference whatever the target contracts. The delivered
                // figure this lane publishes is the one the recurrence is
                // evaluated at, and it is not the fused one.
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

/// \endcond
} // namespace boys
