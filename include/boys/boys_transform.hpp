#pragma once

#include "boys/boys.hpp"

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
/// | Mode | Bound | Delivered, worst over region A |
/// |---|---|---|
/// | \c kFp64 | \c m·1e-15 | 1.110e-16 |
/// | \c kTf32x3 | \c m·1e-15 + 2.5e-7 | 1.916e-07 |
/// | \c kBf16x6 | \c m·1e-15 + 2.5e-7 | 1.946e-07 |
///
/// The delivered figures are the worst found, against the committed 45-digit
/// reference grid and against a 200000-point sweep of the lower band's left
/// end, where these modes are worst. The \c kFp64 figure is the fitted
/// polynomial's own truncation: the product adds nothing measurable to what the
/// coefficients already cost, and it is the same figure the shipped split
/// Clenshaw delivers. The two split-mode figures are their format's floor. The
/// sweep is what found them - the grid alone samples 1.24e-07 and 1.37e-07 at
/// those modes and understates the worst by about 1.5 times.
///
/// **The multiplier does not move the ceiling, and for the two split modes it
/// does not move the bound either.** A mode's floor is its format's, which no
/// multiplier changes. The fit term \c (m−1)·1e-15 cannot reach 1.9e-07 until
/// \c m is about 1.9e8, four orders past the 65536 the multiplier surface
/// offers, so for \c kTf32x3 and \c kBf16x6 the bound is the floor at every
/// multiplier this library names. For \c kFp64 the floor is 1.11e-16, below the
/// budget's own 1e-15, so its bound is the double single lane's \c m·1e-15 at
/// every \c m, and the multiplier is live from \c m = 2 upward as it is on
/// every other lane.
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
/// The modes are the ones that carry a bound a caller can use. The single-pass
/// narrow modes - tf32, bf16 and fp16 operands accumulated in fp32 - are
/// deliberately absent: their error is ten to twelve orders past every budget
/// this library states for the lanes they would serve, and a mode nobody can
/// use is not a mode.
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
};

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

extern template void BoysRegionAProduct<ProductMode::kFp64, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kTf32x3, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;
extern template void BoysRegionAProduct<ProductMode::kBf16x6, kBoysFullAccuracyMultiplier>(
    RegionABand band, int nmax, const double* x, double* out, std::size_t count) noexcept;

} // namespace boys
