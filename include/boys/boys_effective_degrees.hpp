#pragma once

// The compile-time effective-degree machinery behind the accuracy-multiplier
// parametrization. The multiplier m relaxes each lane's asserted per-region
// bound B_region by truncating the seed fits to the effective degree
//
//   d'(m) = min { d' in {0,1,2,4,6,...} : Delta(d') * A <= (m-1) * B_region },
//   Delta(d') = sum_{k=d'+1}^{d} |c_k|   (the dropped-coefficient tail),
//
// with A the path's seed-error amplification: 1 for the single-style lanes,
// w(b) = max(1, b^n / prod(j+1/2)) at the piece's right end for the region-A
// batch downward recursion, and prod(j+1/2)/x0^n for the region-B upward
// recursion (worst at x = x0). The per-(order, piece) d' tables are
// compile-time constants per (m, lane role, basis), a-priori, never tuned.
//
// The tail must come from the table the scheme sums. A fit is carried as a
// Chebyshev table and a monomial table over the same pieces, and the two hold
// different numbers describing it: a Chebyshev fit's coefficients decay with
// the fit's accuracy, while the same fit's monomial coefficients are its
// Taylor coefficients on the piece and their high-order end is larger by about
// 2^k. A degree the Chebyshev tail admits can drop a monomial tail several
// orders of magnitude over budget. TailBasis names which table a scan reads.
//
// The summation's own rounding. Horner at degree d' returns
// sum_{k<=d'} c_k t^k (1 + theta_k) with |theta_k| <= gamma_{k+1},
// gamma_k = k*u/(1-ku): the backward form, in which the coefficient at step k
// carries the perturbation of the k+1 roundings at or below it. So the sum's
// error over the exact truncated polynomial is at most
//   R(d') = sum_{k<=d'} |c_k| gamma_{k+1} <= gamma_{d'+1} sum_{k<=d'} |c_k|.
// R(d') is non-decreasing in d' - every term added is non-negative - so
// R(d') <= R(d) and the rung's summation rounds no more than the m = 1 lane's
// does. It is the m = 1 lane's rounding that the m = 1 base is asserted to
// carry, and the criterion spends nothing on the rung's.
//
// Why no constant rounding term belongs in the criterion. Bounding the rung's
// rounding and the m = 1 lane's independently and adding both would put a term
// in the criterion that does not fall to zero at d' = d, and such a criterion
// would refuse the full degree - which is provably wrong, because at d' = d
// the rung runs the m = 1 summation over the same coefficients bit for bit,
// and its delivered error is the m = 1 lane's, inside B_region by the
// contract. A criterion that can refuse a degree it must admit is not a
// criterion. So the shape is the m = 1 base, asserted and measured, plus the
// one term the truncation adds: Delta(d') * A.
//
// Delta(d) = 0, so the scan always reaches the full degree: no rung is left
// without an admissible one, and the fallback is the m = 1 summation, whose
// error the rung's own bound already covers.
//
// The fp16/bf16 lanes are I/O around the fp32 engine; their 1e-7 region budgets
// are the fp16 bound formula's asserted base, strictly stronger than the float
// lanes' documented 1.5e-7, and the representation half-ULP term is
// m-independent.

/// \cond
// Not API: the degree arithmetic the entries are compiled from.

#include "boys/boys_coefficients.hpp"

#include <array>
#include <cstddef>
#include <utility>

namespace boys {
namespace detail {

/// The lane roles of the accuracy contract (region budgets + amplification).
enum class BoysRole {
    kDoubleSingle,
    kDoubleBatch,
    kF32Single,
    kF32Batch,
    kF32Fp16Single,
    kF32Fp16Batch,
};

/// The coefficient table a truncation drops from. A fit is carried once per
/// scheme that sums it and the two tables hold different numbers over the same
/// pieces, so the basis is part of the degree table's identity.
enum class TailBasis {
    /// The Chebyshev tables — kCoeffs, kBcoeffs and their single-precision
    /// pair — which the split Clenshaw recurrence reads.
    kChebyshev,

    /// The monomial tables — kMonoCoeffs and kMonoBcoeffs — which are the
    /// double lane's Horner form.
    kMonomial,
};

/// Region-A asserted budget of the role (the m = 1 contract).
constexpr double RegionABudget(BoysRole role) noexcept {
    switch (role)
    {
    case BoysRole::kDoubleSingle:
        return 1e-15;
    case BoysRole::kDoubleBatch:
        return 5.5e-14;
    case BoysRole::kF32Single:
    case BoysRole::kF32Batch:
        return 1.5e-7;
    case BoysRole::kF32Fp16Single:
    case BoysRole::kF32Fp16Batch:
        return 1e-7;
    }

    return 0.0; // unreachable
}

/// Region-B asserted budget of the role (the m = 1 contract).
constexpr double RegionBBudget(BoysRole role) noexcept {
    switch (role)
    {
    case BoysRole::kDoubleSingle:
        return 3e-14;
    case BoysRole::kDoubleBatch:
        return 5.5e-14;
    case BoysRole::kF32Single:
    case BoysRole::kF32Batch:
        return 1.5e-7;
    case BoysRole::kF32Fp16Single:
    case BoysRole::kF32Fp16Batch:
        return 1e-7;
    }

    return 0.0; // unreachable
}

/// Whether the role's region-A seed runs the batch downward recursion
/// (amplification w(b) at the piece's right end) or is single-style (A = 1).
constexpr bool RoleUsesBatchAmplification(BoysRole role) noexcept {
    return role == BoysRole::kDoubleBatch || role == BoysRole::kF32Batch ||
           role == BoysRole::kF32Fp16Batch;
}

/// Whether the role's region-A seed evaluates the double piece table.
constexpr bool RoleUsesDoubleTables(BoysRole role) noexcept {
    return role == BoysRole::kDoubleSingle || role == BoysRole::kDoubleBatch ||
           role == BoysRole::kF32Batch || role == BoysRole::kF32Fp16Batch;
}

/// The region-A batch seed-error amplification at the piece's right end:
/// w(b) = max(1, b^n / prod_{j=0}^{n-1}(j+1/2)), increasing in x.
constexpr double RegionAAmplification(int order, double b) noexcept {
    double numerator = 1.0;
    double denominator = 1.0;

    for (int j = 0; j < order; ++j)
    {
        numerator *= b;
        denominator *= (static_cast<double>(j) + 0.5);
    }

    return numerator > denominator ? numerator / denominator : 1.0;
}

/// The region-B upward-recursion amplification, worst at x = kX0:
/// A_B(n) = prod_{j=0}^{n-1}(j+1/2) / kX0^n.
constexpr double RegionBAmplification(int order) noexcept {
    double numerator = 1.0;
    double denominator = 1.0;

    for (int j = 0; j < order; ++j)
    {
        numerator *= (static_cast<double>(j) + 0.5);
        denominator *= kX0;
    }

    return numerator / denominator;
}

// The extended band (below kX0) keeps the region-A treatment: its per-order
// amplification varies per order, unlike the fixed kX0 worst case
// RegionBAmplification assumes, and the m > 1 branch does not carry it.

/// The k-th stored coefficient, read through the table's data pointer: the
/// subscript's check is a constant expression of its own, one per coefficient
/// read, and the element read is the same one either way.
template <typename CoeffArray>
constexpr double CoefficientAt(const CoeffArray& coeffs, std::size_t index) noexcept {
    return static_cast<double>(coeffs.data()[index]);
}

/// The dropped-coefficient tail Delta(d') = sum_{k=d'+1}^{deg} |c_k|. The scan
/// is basis-blind; which table reaches it is the caller's choice.
template <typename CoeffArray>
constexpr double CoefficientTail(const CoeffArray& coeffs,
                                 std::size_t offset,
                                 int deg,
                                 int dPrime) noexcept {
    double tail = 0.0;

    for (int k = dPrime + 1; k <= deg; ++k)
    {
        const double c = CoefficientAt(coeffs, offset + static_cast<std::size_t>(k));
        tail += (c >= 0.0) ? c : -c;
    }

    return tail;
}

/// The effective degree: the smallest admissible truncation in the
/// evaluator's domain {0,1,2,4,6,...}; even degrees >= 4 are the split-Clenshaw
/// domain, and 0/1/2 are special-cased by the evaluator.
///
/// The budget is the one this library carries - m = 1 relaxes nothing, so the
/// criterion is Delta(d') * A <= 0 - and Delta(deg) = 0 is the only zero tail,
/// so the scan reports the full degree. The scan is kept rather than replaced by
/// the degree it always returns: it is the derivation, and the assertion beside
/// each table reads it.
template <typename CoeffArray>
constexpr int EffectiveDegree(const CoeffArray& coeffs,
                              std::size_t offset,
                              int deg,
                              double amplification) noexcept {
    const double budget = 0.0;

    for (int dPrime = 0; dPrime <= deg; ++dPrime)
    {
        if (dPrime == 3 || (dPrime > 2 && dPrime % 2 != 0))
        {
            continue; // outside the evaluator's domain
        }

        if (CoefficientTail(coeffs, offset, deg, dPrime) * amplification <= budget)
        {
            return dPrime;
        }
    }

    return deg;
}

// Which of a piece table's two stored forms the scan reads. Declared above the
// derivations because each names it with explicit template arguments over a
// dependent array, which a compiler would not find if it were declared below.
template <TailBasis kBasis, typename Table>
constexpr const auto& TailTable(const Table& chebyshev, const Table& monomial) noexcept {
    return (kBasis == TailBasis::kChebyshev) ? chebyshev : monomial;
}

// The narrow partition's own effective-degree tables. The narrow pieces carry
// their own coefficients, so the criterion applies to them unchanged: the tail a
// rung drops from *this* piece's stored coefficients, times the path's
// amplification, against the rung's budget. Region A's pieces are cut per order,
// so the table is flat over kNarrowAPieces as the shipped one is over kPieces;
// which lane's narrow table a role scans follows the same split as
// RegionADegrees' two branches (RoleUsesDoubleTables).
template <BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
constexpr auto NarrowRegionADegrees() noexcept {
    if constexpr (RoleUsesDoubleTables(kRole))
    {
        constexpr const auto& coeffs = TailTable<kBasis>(kNarrowACoeffs, kNarrowAMonoCoeffs);
        std::array<int, std::size(kNarrowAPieces)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = kNarrowAPieceStart[order]; p < kNarrowAPieceStart[order + 1]; ++p)
            {
                const OrderPiece& piece = kNarrowAPieces[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece.offset),
                                    piece.deg,
                                    amplification);
            }
        }

        return degrees;
    } else
    {
        // The single-precision lane's own narrow pieces, in whichever of the
        // two forms its scheme sums.
        constexpr const auto& coeffs = TailTable<kBasis>(f32::kNarrowACoeffsF32,
                                                         f32::kNarrowAMonoCoeffsF32);
        std::array<int, std::size(f32::kNarrowAPiecesF32)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = f32::kNarrowAPieceStartF32[order];
                 p < f32::kNarrowAPieceStartF32[order + 1];
                 ++p)
            {
                const f32::OrderPiece& piece = f32::kNarrowAPiecesF32[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole)
                        ? RegionAAmplification(order, static_cast<double>(piece.b))
                        : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece.offset),
                                    piece.deg,
                                    amplification);
            }
        }

        return degrees;
    }
}

// The narrow partition of region B. Its seed is one polynomial per piece, so a
// row is the pair (piece, order): flat, indexed
// piece * (kMaxOrder + 1) + order, as the other tables are flat. The gain is the
// shipped region B one, A_B(n), because the seed is carried up the same
// recursion either partition feeds. The split is by precision and not by the
// region-A rule above: a float return is a float seed, cut at the float
// budget.
template <BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
constexpr auto NarrowRegionBDegrees() noexcept {
    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        constexpr const auto& coeffs = (kBasis == TailBasis::kChebyshev) ? kNarrowBcoeffs
                                                                         : kNarrowBMonoCoeffs;
        std::array<int, static_cast<std::size_t>(kNarrowBPieces) * (kMaxOrder + 1)> degrees{};

        for (int piece = 0; piece < kNarrowBPieces; ++piece)
        {
            for (int order = 0; order <= kMaxOrder; ++order)
            {
                degrees[static_cast<std::size_t>(piece) * (kMaxOrder + 1)
                        + static_cast<std::size_t>(order)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece) * (kNarrowBDeg + 1),
                                    kNarrowBDeg,
                                    RegionBAmplification(order));
            }
        }

        return degrees;
    } else
    {
        constexpr const auto& coeffs = (kBasis == TailBasis::kChebyshev) ? f32::kNarrowBcoeffsF32
                                                                        : f32::kNarrowBMonoCoeffsF32;
        std::array<int, static_cast<std::size_t>(f32::kNarrowBPiecesF32) * (kMaxOrder + 1)>
            degrees{};

        for (int piece = 0; piece < f32::kNarrowBPiecesF32; ++piece)
        {
            for (int order = 0; order <= kMaxOrder; ++order)
            {
                degrees[static_cast<std::size_t>(piece) * (kMaxOrder + 1)
                        + static_cast<std::size_t>(order)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece)
                                        * (static_cast<std::size_t>(f32::kNarrowBDegF32) + 1),
                                    f32::kNarrowBDegF32,
                                    RegionBAmplification(order));
            }
        }

        return degrees;
    }
}

// ---------------------------------------------------------------------------
// The rational pair's own truncation
// ---------------------------------------------------------------------------
// A stored rational piece is a numerator P(t) = sum_{j<=m} p_j t^j and a
// denominator Q(t) = 1 + sum_{1<=j<=k} q_j t^j over one piece of region A's
// partition, and the region-B seed is one such pair over the whole of region B.
// A lower-order pair cuts both at one order d': m' = min(d', m) and
// k' = min(d', k). The value is a quotient, so what the cut costs is not a
// coefficient sum. With dP = P - P' and dQ = Q - Q' the dropped parts,
//
//   R - R' = dP/Q - R' * dQ/Q,
//
// and the pairwise tail is bounded by
//
//   Delta(d') = ( DP(d') + (SP / (Qlo - DQ(d'))) * DQ(d') ) / Qlo,
//
// with DP/DQ the dropped tails of P and Q, SP the numerator's stored value scale
// and Qlo the denominator's floor over the piece (DenominatorFloor). Both
// terms are needed because cutting the denominator moves every value the pair
// returns.
//
// At d' = max(m, k) the cut is the whole pair, so Delta = 0 and the scan always
// has an admissible order: the returned pair is never one the budget cannot
// carry, and at the full cut it returns the shipped pair's value bit for bit.

/// The number of grid intervals the denominator's floor is bounded on: an
/// a-priori constant of the criterion, not of any one piece.
inline constexpr int kPairFloorGrid = 64;

/// SP, the sum of the numerator's stored coefficients, which is the pair's
/// value scale: |P(t)| <= SP on the mapped interval.
template <typename CoeffArray>
constexpr double NumeratorScale(const CoeffArray& coeffs, std::size_t offset, int numDeg) noexcept {
    double scale = 0.0;

    for (int j = 0; j <= numDeg; ++j)
    {
        const double c = CoefficientAt(coeffs, offset + static_cast<std::size_t>(j));
        scale += (c >= 0.0) ? c : -c;
    }

    return scale;
}

/// Qlo: the denominator's floor over the mapped interval, from Q's own values on
/// a grid less what Q can move between grid points. den[denOffset + j - 1] holds
/// q_j for j = 1..denDeg, and Q(t) = 1 + sum_j q_j t^j.
template <typename CoeffArray>
constexpr double DenominatorFloor(const CoeffArray& den,
                                  std::size_t denOffset,
                                  int denDeg) noexcept {
    double slope = 0.0;

    for (int j = 1; j <= denDeg; ++j)
    {
        const double c = CoefficientAt(den, denOffset + static_cast<std::size_t>(j - 1));
        slope += static_cast<double>(j) * ((c >= 0.0) ? c : -c);
    }

    double lowest = 0.0;
    bool seen = false;

    for (int i = 0; i <= kPairFloorGrid; ++i)
    {
        const double t = -1.0 + 2.0 * static_cast<double>(i) / kPairFloorGrid;
        // Q's own Horner: the constant term is the denominator's held one, so it
        // is added after the last coefficient rather than before the first.
        double value = 0.0;

        for (int j = denDeg; j >= 1; --j)
        {
            value = value * t + CoefficientAt(den, denOffset + static_cast<std::size_t>(j - 1));
        }

        value = value * t + 1.0;

        const double magnitude = value >= 0.0 ? value : -value;

        if (!seen || magnitude < lowest)
        {
            lowest = magnitude;
            seen = true;
        }
    }

    return lowest - slope * (2.0 / kPairFloorGrid) * 0.5;
}

/// The pair's dropped-order measure at one cut, with the two cut-independent
/// operands of the criterion passed in: both are properties of the stored pair
/// rather than of the order it is read at, and the floor is a 65-point grid, so
/// hoisting them keeps the whole table inside a compiler's step budget.
template <typename NumArray, typename DenArray>
constexpr double RationalPairTailAt(const NumArray& num,
                                    std::size_t numOffset,
                                    const DenArray& den,
                                    std::size_t denOffset,
                                    int numDeg,
                                    int denDeg,
                                    int dPrime,
                                    double floor,
                                    double scale,
                                    bool& admissible) noexcept {
    const int numPrime = dPrime < numDeg ? dPrime : numDeg;
    const int denPrime = dPrime < denDeg ? dPrime : denDeg;
    double droppedNum = 0.0;

    for (int j = numPrime + 1; j <= numDeg; ++j)
    {
        const double c = CoefficientAt(num, numOffset + static_cast<std::size_t>(j));
        droppedNum += (c >= 0.0) ? c : -c;
    }

    double droppedDen = 0.0;

    for (int j = denPrime + 1; j <= denDeg; ++j)
    {
        const double c = CoefficientAt(den, denOffset + static_cast<std::size_t>(j - 1));
        droppedDen += (c >= 0.0) ? c : -c;
    }

    const double floorAtCut = floor - droppedDen;

    if (floorAtCut <= 0.0)
    {
        admissible = false;
        return 0.0;
    }

    admissible = true;

    return (droppedNum + (scale / floorAtCut) * droppedDen) / (floorAtCut + droppedDen);
}

/// The pair's dropped-order measure at one cut, taking the two operands from the
/// pair itself. A cut whose floor does not clear the denominator's own dropped
/// tail has no bound under this argument at all, and \p admissible reports that
/// rather than a number, so the caller skips the cut.
template <typename NumArray, typename DenArray>
constexpr double RationalPairTail(const NumArray& num,
                                  std::size_t numOffset,
                                  const DenArray& den,
                                  std::size_t denOffset,
                                  int numDeg,
                                  int denDeg,
                                  int dPrime,
                                  bool& admissible) noexcept {
    return RationalPairTailAt(num,
                              numOffset,
                              den,
                              denOffset,
                              numDeg,
                              denDeg,
                              dPrime,
                              DenominatorFloor(den, denOffset, denDeg),
                              NumeratorScale(num, numOffset, numDeg),
                              admissible);
}

/// The pair's effective cut: the smallest admissible order in the evaluator's
/// domain {0,1,2,4,6,...}, truncating the numerator and the denominator together.
/// The full pair is the fallback and is admissible by construction, its dropped
/// parts being both empty.
///
/// The budget is the one this library carries - m = 1 relaxes nothing, so the
/// criterion is its tail against zero - and the full pair is the only zero tail,
/// so the cut reports it.
template <typename NumArray, typename DenArray>
constexpr void RationalPairCut(const NumArray& num,
                               std::size_t numOffset,
                               const DenArray& den,
                               std::size_t denOffset,
                               int numDeg,
                               int denDeg,
                               double amplification,
                               int& outNumDeg,
                               int& outDenDeg) noexcept {
    const double budget = 0.0;
    const int full = numDeg > denDeg ? numDeg : denDeg;

    // The two cut-independent operands, taken once for the pair: every candidate
    // cut below asks the same two questions.
    const double floor = DenominatorFloor(den, denOffset, denDeg);
    const double scale = NumeratorScale(num, numOffset, numDeg);

    for (int dPrime = 0; dPrime <= full; ++dPrime)
    {
        if (dPrime == 3 || (dPrime > 2 && dPrime % 2 != 0))
        {
            continue; // outside the evaluator's domain
        }

        bool admissible = false;
        const double tail = RationalPairTailAt(
            num, numOffset, den, denOffset, numDeg, denDeg, dPrime, floor, scale, admissible);

        if (admissible && tail * amplification <= budget)
        {
            outNumDeg = dPrime < numDeg ? dPrime : numDeg;
            outDenDeg = dPrime < denDeg ? dPrime : denDeg;
            return;
        }
    }

    outNumDeg = numDeg;
    outDenDeg = denDeg;
}

#if !defined(__CUDACC__)
// The per-(m, role, basis) compile-time d' tables. The degree tables are flat
// std::array<int, ...> (one entry per region-A piece / per order for region B;
// the flat form keeps the tables constexpr on MSVC). The NTTP forms are
// instantiation-local constants - zero mutable state on the CPU path.
//
// The region-A degrees of one role and one basis: the branch below is which
// lane's stored piece table the role reads, and the basis picks the form within
// it - the Chebyshev coefficients ClenshawSplit reads, or the monomial ones
// HornerMono reads, over the same pieces at the same degrees.
template <BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
constexpr auto RegionADegrees() noexcept {
    if constexpr (RoleUsesDoubleTables(kRole))
    {
        constexpr const auto& coeffs = TailTable<kBasis>(kCoeffs, kMonoCoeffs);
        std::array<int, std::size(kPieces)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = kPieceStart[order]; p < kPieceStart[order + 1]; ++p)
            {
                const OrderPiece& piece = kPieces[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece.offset),
                                    piece.deg,
                                    amplification);
            }
        }

        return degrees;
    } else
    {
        // The single-precision roles' own table, in whichever of its two stored
        // forms the rung's scheme sums.
        constexpr const auto& coeffs = TailTable<kBasis>(f32::kCoeffs, f32::kMonoCoeffs);
        std::array<int, std::size(f32::kPieces)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = f32::kPieceStart[order]; p < f32::kPieceStart[order + 1]; ++p)
            {
                const f32::OrderPiece& piece = f32::kPieces[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(coeffs,
                                    static_cast<std::size_t>(piece.offset),
                                    piece.deg,
                                    amplification);
            }
        }

        return degrees;
    }
}

// The region-B degrees of one role and one basis; see RegionADegrees.
template <BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
constexpr auto RegionBDegrees() noexcept {
    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        constexpr const auto& coeffs = TailTable<kBasis>(kBcoeffs, kMonoBcoeffs);
        std::array<int, kMaxOrder + 1> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            degrees[static_cast<std::size_t>(order)] = EffectiveDegree(coeffs,
                                                                       0,
                                                                       kBDeg,
                                                                       RegionBAmplification(order));
        }

        return degrees;
    } else
    {
        // The single-precision lane's own seed, in the form its scheme sums;
        // one row, as the double lane's is, so the scan is the same one.
        constexpr const auto& coeffs = TailTable<kBasis>(f32::kBcoeffs, f32::kMonoBcoeffs);
        std::array<int, kMaxOrder + 1> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            degrees[static_cast<std::size_t>(order)] = EffectiveDegree(coeffs,
                                                                       0,
                                                                       f32::kBDeg,
                                                                       RegionBAmplification(order));
        }

        return degrees;
    }
}

// The rational pair tables at a rung: per region-A piece and per region-B
// order, the numerator and denominator degree the pair criterion certifies. Two
// flat int arrays rather than one of pairs, for the same reason the polynomial
// tables are flat.
//
// The amplification each region's cut pays is the path's own: region A reads one
// piece per order and nothing amplifies the cut, so A = 1 rather than the batch
// role's w(b); region B reads its seed at order 0 and carries it up, so its cut
// pays the shipped table's A_B(n). The budget is the batch role's at both, since
// the tier the rung is named by documents m * 5.5e-14 in every region.
struct RationalRegionAPairs {
    std::array<int, std::size(kPieces)> num{};
    std::array<int, std::size(kPieces)> den{};
};

struct RationalRegionBPairs {
    std::array<int, kMaxOrder + 1> num{};
    std::array<int, kMaxOrder + 1> den{};
};

// One order's row of the double lane's stored region-A pairs: the pieces are
// grouped by order, so a row is the order's own pieces. It is cut in a constant
// expression of its own rather than inside the assembly's single evaluation,
// because a compiler's step budget is spent per evaluation; the rows hold the
// same cuts either way.
template <int kOrder, bool kSeedReading> struct RationalARow {
    static constexpr int kFirst = kPieceStart[kOrder];
    static constexpr int kCount = kPieceStart[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <BoysRole kRole, bool kSeedReading, int kOrder>
constexpr RationalARow<kOrder, kSeedReading> CutRationalARow() noexcept {
    RationalARow<kOrder, kSeedReading> row{};

    for (int i = 0; i < RationalARow<kOrder, kSeedReading>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(RationalARow<kOrder, kSeedReading>::kFirst + i);
        const OrderPiece& piece = kPieces[index];
        const int numDeg = kRatANumDeg[index];
        const int denDeg = kRatADenDeg[index];

        RationalPairCut(kRatACoeffs,
                        static_cast<std::size_t>(kRatAOffset[index]),
                        kRatACoeffs,
                        static_cast<std::size_t>(kRatAOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        kSeedReading ? RegionAAmplification(kOrder, piece.b) : 1.0,
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// A variable template's initializer is a constant expression of its own, so each
// row above is cut under a budget of its own rather than the assembly's.
template <BoysRole kRole, bool kSeedReading, int kOrder>
inline constexpr RationalARow<kOrder, kSeedReading> kRationalARow =
    CutRationalARow< kRole, kSeedReading, kOrder>();

template <BoysRole kRole, bool kSeedReading, int kOrder>
constexpr void CopyRationalARow(RationalRegionAPairs& pairs) noexcept {
    const RationalARow<kOrder, kSeedReading> row =
        kRationalARow< kRole, kSeedReading, kOrder>;

    for (int i = 0; i < RationalARow<kOrder, kSeedReading>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(RationalARow<kOrder, kSeedReading>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <BoysRole kRole, bool kSeedReading, std::size_t... kOrders>
constexpr RationalRegionAPairs AssembleRationalA(std::index_sequence<kOrders...>) noexcept {
    RationalRegionAPairs pairs{};
    (CopyRationalARow< kRole, kSeedReading, static_cast<int>(kOrders)>(pairs),
     ...);
    return pairs;
}

constexpr RationalRegionAPairs RationalRegionADegrees() noexcept {
    return AssembleRationalA< BoysRole::kDoubleBatch, false>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

constexpr RationalRegionBPairs RationalRegionBDegrees() noexcept {
    RationalRegionBPairs pairs{};

    // One seed for the region rather than one fit per order, so the cut is the
    // same pair at every order and the table is that pair repeated. It is judged
    // at A_B(0) = 1, the smallest the region pays: the amplification belongs to
    // the order the pair is read for, not to the seed.
    int numDeg = 0;
    int denDeg = 0;

    RationalPairCut(kRatBnum,
                    0,
                    kRatBden,
                    0,
                    kRatBnumDeg,
                    kRatBdenDeg,
                    RegionBAmplification(0),
                    numDeg,
                    denDeg);

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        pairs.num[static_cast<std::size_t>(order)] = numDeg;
        pairs.den[static_cast<std::size_t>(order)] = denDeg;
    }

    return pairs;
}

// ---------------------------------------------------------------------------
// The narrow partition's own pairs at a rung
// ---------------------------------------------------------------------------
// The narrow partition carries its own stored numerator/denominator pairs - a
// different cover, a different fit, different roundings from the shipped
// partition's over the same domain - so a rung of it is a cut of THESE pairs.
// The criterion and the reading are the shipped ones: region A's piece is the
// order's value and nothing amplifies the cut, so A = 1; region B's seed is one
// evaluation carried up from order 0, so its cut is judged at A_B(0) = 1. Both
// spend the batch role's region budget, a property of the region and the
// recursion rather than of the entry that reads the rung.
struct NarrowRationalRegionAPairs {
    std::array<int, std::size(kNarrowAPieces)> num{};
    std::array<int, std::size(kNarrowAPieces)> den{};
};

struct NarrowRationalRegionBPairs {
    std::array<int, kNarrowBPieces> num{};
    std::array<int, kNarrowBPieces> den{};
};

constexpr NarrowRationalRegionAPairs RationalRegionANarrowDegrees() noexcept {
    NarrowRationalRegionAPairs pairs{};

    for (int p = 0; p < static_cast<int>(std::size(kNarrowAPieces)); ++p)
    {
        const std::size_t index = static_cast<std::size_t>(p);
        const int numDeg = kNarrowRatANumDeg[index];
        const int denDeg = kNarrowRatADenDeg[index];

        RationalPairCut(kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index]),
                        kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        1.0,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

constexpr NarrowRationalRegionBPairs RationalRegionBNarrowDegrees() noexcept {
    NarrowRationalRegionBPairs pairs{};

    // The narrow seed is one pair per piece rather than one pair for the whole
    // region, so each piece's pair is cut against its own coefficients; the
    // reading is the shipped seed's, at order 0's amplification.
    for (int p = 0; p < kNarrowBPieces; ++p)
    {
        const std::size_t index = static_cast<std::size_t>(p);
        const int numDeg = kNarrowRatBNumDeg[index];
        const int denDeg = kNarrowRatBDenDeg[index];

        RationalPairCut(kNarrowRatBCoeffs,
                        static_cast<std::size_t>(kNarrowRatBOffset[index]),
                        kNarrowRatBCoeffs,
                        static_cast<std::size_t>(kNarrowRatBOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        RegionBAmplification(0),
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

// The narrow pairs at the batch seed's reading: the same narrow pieces and the
// same stored pairs as the single-order reading above, cut at the amplification
// the recursion pays at the piece's right end rather than at A = 1.
template <BoysRole kRole>
constexpr NarrowRationalRegionAPairs RationalRegionANarrowSeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalRegionANarrowDegrees instead");

    NarrowRationalRegionAPairs pairs{};

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        for (int p = kNarrowAPieceStart[order]; p < kNarrowAPieceStart[order + 1]; ++p)
        {
            const std::size_t index = static_cast<std::size_t>(p);
            const OrderPiece& piece = kNarrowAPieces[index];
            const int numDeg = kNarrowRatANumDeg[index];
            const int denDeg = kNarrowRatADenDeg[index];

            RationalPairCut(kNarrowRatACoeffs,
                            static_cast<std::size_t>(kNarrowRatAOffset[index]),
                            kNarrowRatACoeffs,
                            static_cast<std::size_t>(kNarrowRatAOffset[index] + numDeg + 1),
                            numDeg,
                            denDeg,
                            RegionAAmplification(order, piece.b),
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// ---------------------------------------------------------------------------
// The rational pair at a batch seed's reading
// ---------------------------------------------------------------------------
// The cut is the same pair over the same pieces as the per-order reading, but
// judged at the batch amplification: an engine that seeds a batch from the top
// order's piece and recurses down pays w(b) at that piece's right end. The pair
// is the double lane's stored one - an engine that needs a batch seed's
// precision reads this lane's rational fit, not the float lane's own pair.
template <BoysRole kRole>
constexpr RationalRegionAPairs RationalRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalRegionADegrees instead");

    return AssembleRationalA< kRole, true>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// One order's row of the narrow seed table. The table is cut a row at a time:
// cutting all of them inside one constant expression is longer than a compiler is
// obliged to run one for, and a single row is a small fraction of that budget.
// The rows hold the same cuts either way.
template <int kOrder> struct NarrowRationalSeedRow {
    static constexpr int kFirst = kNarrowAPieceStart[kOrder];
    static constexpr int kCount = kNarrowAPieceStart[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <BoysRole kRole, int kOrder>
constexpr NarrowRationalSeedRow<kOrder> CutNarrowRationalSeedRow() noexcept {
    NarrowRationalSeedRow<kOrder> row{};

    for (int i = 0; i < NarrowRationalSeedRow<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalSeedRow<kOrder>::kFirst + i);
        const OrderPiece& piece = kNarrowAPieces[index];
        const int numDeg = kNarrowRatANumDeg[index];
        const int denDeg = kNarrowRatADenDeg[index];

        RationalPairCut(kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index]),
                        kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        RegionAAmplification(kOrder, piece.b),
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// A variable template is initialized in a constant expression of its own, so the
// row above is cut under a budget of its own rather than the assembly's.
template <BoysRole kRole, int kOrder>
inline constexpr NarrowRationalSeedRow<kOrder> kNarrowRationalSeedRow =
    CutNarrowRationalSeedRow< kRole, kOrder>();

template <BoysRole kRole, int kOrder>
constexpr void CopyNarrowRationalSeedRow(NarrowRationalRegionAPairs& pairs) noexcept {
    const NarrowRationalSeedRow<kOrder> row =
        kNarrowRationalSeedRow< kRole, kOrder>;

    for (int i = 0; i < NarrowRationalSeedRow<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalSeedRow<kOrder>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <BoysRole kRole, std::size_t... kOrders>
constexpr NarrowRationalRegionAPairs AssembleNarrowRationalSeed(
    std::index_sequence<kOrders...>) noexcept {
    NarrowRationalRegionAPairs pairs{};
    (CopyNarrowRationalSeedRow< kRole, static_cast<int>(kOrders)>(pairs), ...);
    return pairs;
}

// The same batch reading over the double lane's narrow pieces, judged at each
// piece's own w(b): a batch role whose policy names the narrow partition seeds
// its recursion from that partition's rational fit.
template <BoysRole kRole>
constexpr NarrowRationalRegionAPairs NarrowRationalRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads the single-order reading instead");

    return AssembleNarrowRationalSeed< kRole>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// ---------------------------------------------------------------------------
// The single-precision lane's own rational pairs at a rung
// ---------------------------------------------------------------------------
// The float lane carries the rational family as its own stored tables, so a rung
// of this family on this lane is a cut of THESE, by the same pair criterion and
// at the role's own budget. Region A's reading is the single-order one, A = 1;
// the float lane's batch seeds from the double lane's pair above and not from
// this table.
struct RationalRegionAF32Pairs {
    std::array<int, std::size(f32::kRatAPieces)> num{};
    std::array<int, std::size(f32::kRatAPieces)> den{};
};

struct RationalRegionBF32Pairs {
    std::array<int, kMaxOrder + 1> num{};
    std::array<int, kMaxOrder + 1> den{};
};

template <BoysRole kRole>
constexpr RationalRegionAF32Pairs RationalRegionAF32Degrees() noexcept {
    RationalRegionAF32Pairs pairs{};

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        for (int p = f32::kRatAPieceStart[order]; p < f32::kRatAPieceStart[order + 1]; ++p)
        {
            const std::size_t index = static_cast<std::size_t>(p);
            const f32::RatPiece& piece = f32::kRatAPieces[index];

            RationalPairCut(f32::kRatACoeffs,
                            static_cast<std::size_t>(piece.offset),
                            f32::kRatACoeffs,
                            static_cast<std::size_t>(piece.offset + piece.numdeg + 1),
                            piece.numdeg,
                            piece.dendeg,
                            RoleUsesBatchAmplification(kRole)
                                ? RegionAAmplification(order, static_cast<double>(piece.b))
                                : 1.0,
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// The float lane's region-B seed pair at a rung, cut once and repeated down the
// table as the double lane's is, judged at A_B(0) = 1: kX0 is the argument at
// which prod_{j<n}(j+1/2)/kX0^n first reaches one, so A_B(n) <= A_B(0) at every
// order the table covers.
template <BoysRole kRole>
constexpr RationalRegionBF32Pairs RationalRegionBF32Degrees() noexcept {
    RationalRegionBF32Pairs pairs{};

    int numDeg = 0;
    int denDeg = 0;

    RationalPairCut(f32::kRatBnum,
                    0,
                    f32::kRatBden,
                    0,
                    f32::kRatBnumDeg,
                    f32::kRatBdenDeg,
                    RegionBAmplification(0),
                    numDeg,
                    denDeg);

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        pairs.num[static_cast<std::size_t>(order)] = numDeg;
        pairs.den[static_cast<std::size_t>(order)] = denDeg;
    }

    return pairs;
}

// The same two tables over the float lane's narrow partition, cut by the same
// criterion: that is the pair the m = 1 call of a narrow policy evaluates.
struct NarrowRationalRegionAF32Pairs {
    std::array<int, std::size(f32::kNarrowRatAPiecesF32)> num{};
    std::array<int, std::size(f32::kNarrowRatAPiecesF32)> den{};
};

struct NarrowRationalRegionBF32Pairs {
    std::array<int, static_cast<std::size_t>(f32::kNarrowRatBPiecesCountF32)> num{};
    std::array<int, static_cast<std::size_t>(f32::kNarrowRatBPiecesCountF32)> den{};
};

// One order's row of the float lane's narrow pairs, cut in a constant expression
// of its own and copied by the assembly below; a row is a small fraction of a
// compiler's step budget. The rows hold the same cuts either way.
template <int kOrder> struct NarrowRationalF32Row {
    static constexpr int kFirst = f32::kNarrowRatAPieceStartF32[kOrder];
    static constexpr int kCount = f32::kNarrowRatAPieceStartF32[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <BoysRole kRole, int kOrder>
constexpr NarrowRationalF32Row<kOrder> CutNarrowRationalF32Row() noexcept {
    NarrowRationalF32Row<kOrder> row{};

    for (int i = 0; i < NarrowRationalF32Row<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalF32Row<kOrder>::kFirst + i);
        const f32::RatPiece& piece = f32::kNarrowRatAPiecesF32[index];

        RationalPairCut(f32::kNarrowRatACoeffsF32,
                        static_cast<std::size_t>(piece.offset),
                        f32::kNarrowRatACoeffsF32,
                        static_cast<std::size_t>(piece.offset + piece.numdeg + 1),
                        piece.numdeg,
                        piece.dendeg,
                        RoleUsesBatchAmplification(kRole)
                            ? RegionAAmplification(kOrder, static_cast<double>(piece.b))
                            : 1.0,
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// A variable template's initializer is a constant expression of its own, so each
// row above is cut under a budget of its own rather than the assembly's.
template <BoysRole kRole, int kOrder>
inline constexpr NarrowRationalF32Row<kOrder> kNarrowRationalF32Row =
    CutNarrowRationalF32Row< kRole, kOrder>();

template <BoysRole kRole, int kOrder>
constexpr void CopyNarrowRationalF32Row(NarrowRationalRegionAF32Pairs& pairs) noexcept {
    const NarrowRationalF32Row<kOrder> row =
        kNarrowRationalF32Row< kRole, kOrder>;

    for (int i = 0; i < NarrowRationalF32Row<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalF32Row<kOrder>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <BoysRole kRole, std::size_t... kOrders>
constexpr NarrowRationalRegionAF32Pairs AssembleNarrowRationalF32(
    std::index_sequence<kOrders...>) noexcept {
    NarrowRationalRegionAF32Pairs pairs{};
    (CopyNarrowRationalF32Row< kRole, static_cast<int>(kOrders)>(pairs), ...);
    return pairs;
}

template <BoysRole kRole>
constexpr NarrowRationalRegionAF32Pairs NarrowRationalRegionAF32Degrees() noexcept {
    return AssembleNarrowRationalF32< kRole>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// Region B's narrow seed is one pair per narrow piece rather than one for the
// region, so the cut is the same reading as the shipped one taken once per
// piece.
template <BoysRole kRole>
constexpr NarrowRationalRegionBF32Pairs NarrowRationalRegionBF32Degrees() noexcept {
    NarrowRationalRegionBF32Pairs pairs{};

    for (int piece = 0; piece < f32::kNarrowRatBPiecesCountF32; ++piece)
    {
        const std::size_t index = static_cast<std::size_t>(piece);
        const f32::RatPiece& row = f32::kNarrowRatBPiecesF32[index];

        RationalPairCut(f32::kNarrowRatBCoeffsF32,
                        static_cast<std::size_t>(row.offset),
                        f32::kNarrowRatBCoeffsF32,
                        static_cast<std::size_t>(row.offset + row.numdeg + 1),
                        row.numdeg,
                        row.dendeg,
                        RegionBAmplification(0),
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}
#endif // !defined(__CUDACC__)


// ---------------------------------------------------------------------------
// The same tables under the names the device lane's entries read them by
// ---------------------------------------------------------------------------
// The narrow partition's own pairs, cut by the criterion above: same layout,
// same criterion, different stored numbers, and a degree certified for one
// table's fit is not a degree certified for the other's. Region A's two readings
// are the shipped route's two - a per-order reading pays A = 1 and a batch seed
// the piece's own w(b) - and region B's seed is one pair per narrow piece,
// judged at order 0 as the shipped route's is.
struct RationalNarrowRegionAPairs {
    std::array<int, std::size(kNarrowAPieces)> num{};
    std::array<int, std::size(kNarrowAPieces)> den{};
};

struct RationalNarrowRegionBPairs {
    std::array<int, kNarrowBPieces> num{};
    std::array<int, kNarrowBPieces> den{};
};

// The per-order reading: one pair per narrow piece, at A = 1.
constexpr RationalNarrowRegionAPairs RationalNarrowRegionADegrees() noexcept {
    RationalNarrowRegionAPairs pairs{};

    for (int p = 0; p < static_cast<int>(std::size(kNarrowAPieces)); ++p)
    {
        const std::size_t index = static_cast<std::size_t>(p);
        const int numDeg = kNarrowRatANumDeg[index];
        const int denDeg = kNarrowRatADenDeg[index];

        RationalPairCut(kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index]),
                        kNarrowRatACoeffs,
                        static_cast<std::size_t>(kNarrowRatAOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        1.0,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

// The batch seed's reading of the same table: the amplification is the piece's
// own w(b) here and 1 above.
template <BoysRole kRole>
constexpr RationalNarrowRegionAPairs RationalNarrowRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the narrow rational pair is the batch recursion's - the cut "
                  "is judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalNarrowRegionADegrees instead");

    RationalNarrowRegionAPairs pairs{};

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        for (int p = kNarrowAPieceStart[order]; p < kNarrowAPieceStart[order + 1]; ++p)
        {
            const std::size_t index = static_cast<std::size_t>(p);
            const OrderPiece& piece = kNarrowAPieces[index];
            const int numDeg = kNarrowRatANumDeg[index];
            const int denDeg = kNarrowRatADenDeg[index];

            RationalPairCut(kNarrowRatACoeffs,
                            static_cast<std::size_t>(kNarrowRatAOffset[index]),
                            kNarrowRatACoeffs,
                            static_cast<std::size_t>(kNarrowRatAOffset[index] + numDeg + 1),
                            numDeg,
                            denDeg,
                            RegionAAmplification(order, piece.b),
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// The narrow region-B seed's pairs, one per piece, at order 0's amplification.
constexpr RationalNarrowRegionBPairs RationalNarrowRegionBDegrees() noexcept {
    RationalNarrowRegionBPairs pairs{};

    for (int piece = 0; piece < kNarrowBPieces; ++piece)
    {
        const std::size_t index = static_cast<std::size_t>(piece);
        const int numDeg = kNarrowRatBNumDeg[index];
        const int denDeg = kNarrowRatBDenDeg[index];

        RationalPairCut(kNarrowRatBCoeffs,
                        static_cast<std::size_t>(kNarrowRatBOffset[index]),
                        kNarrowRatBCoeffs,
                        static_cast<std::size_t>(kNarrowRatBOffset[index] + numDeg + 1),
                        numDeg,
                        denDeg,
                        RegionBAmplification(0),
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

} // namespace detail
} // namespace boys

/// \endcond
