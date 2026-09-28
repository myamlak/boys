#pragma once

// The compile-time effective-degree machinery behind the accuracy-multiplier
// parametrization.
// The multiplier m relaxes each lane's *asserted*
// per-region bounds B_region by truncating the seed fits to the effective
// degree
//
//   d'(m) = min { d' in {0,1,2,4,6,...} : Delta(d') * A <= (m-1) * B_region },
//   Delta(d') = sum_{k=d'+1}^{d} |c_k|   (the dropped-coefficient tail),
//
// with A the path's seed-error amplification: 1 for the single-style lanes,
// w(b) = max(1, b^n / prod(j+1/2)) at the piece's right end for the region-A
// batch downward recursion, and prod(j+1/2)/x0^n for the region-B upward
// recursion (worst at x = x0). Everything here is constexpr over the emitted
// tables: the per-(order,piece) d' tables are compile-time constants per
// (m, lane role, basis), a-priori, never tuned.
//
// Why the tail has to be the one the summation reads. A stored fit is one
// polynomial carried twice over the same pieces - the Chebyshev table a
// Clenshaw recurrence reads and the monomial table Horner's rule reads - and
// the two tables' coefficients are different numbers describing it. What a
// truncation costs is the 1-norm of the coefficients it drops, and that holds
// in either basis for the same reason: every basis function is at most one in
// modulus on the mapped interval (|T_k(t)| <= 1, |t^k| <= 1 for |t| <= 1), so
// the dropped series is at most Delta(d') wherever it is summed. The two
// tables are not interchangeable in that sum. A Chebyshev fit's coefficients
// decay with the fit's accuracy; the same fit's monomial coefficients are its
// Taylor coefficients on the piece and their high-order end is larger by
// about 2^k, so a degree the Chebyshev tail admits can drop a monomial tail
// several orders of magnitude over budget. The tail must therefore come from
// the table the scheme reads: TailBasis names which, and EffectiveDegree is
// the one scan over it.
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
// The lane roles pair the region budgets with the amplification each lane
// pays (the m = 1 asserted bounds):
//   kDoubleSingle      A: double table, A=1,            B=1e-15;  B: double, A_B(n),   B=3e-14
//   kDoubleBatch       A: double table, A=w(b),         B=5.5e-14; B: double, A_B(n),   B=5.5e-14
//   kF32Single         A: float  table, A=1,            B=1.5e-7; B: float,  A_B(n),   B=1.5e-7
//   kF32Batch          A: double table, A=w(b),         B=1.5e-7; B: float,  A_B(n),   B=1.5e-7
//   kF32Fp16Single     A: float  table, A=1,            B=1e-7;  B: float,  A_B(n),   B=1e-7
//   kF32Fp16Batch      A: double table, A=w(b),         B=1e-7;  B: float,  A_B(n),   B=1e-7
// (the fp16/bf16 lanes are I/O around the fp32 engine; their 1e-7 region
// budgets are the suite-asserted fp16 base of the fp16 bound formula —
// strictly stronger than the float lanes' documented 1.5e-7 + ½ULP (the
// suite's 1e-7 is the fp16 lanes' asserted base, not the float budget);
// the representation half-ULP term is m-independent).
//
// The CUDA host side (boys_cuda.cu) fills the device degree tables for a
// runtime m by calling EffectiveDegree directly (the non-template core);
// the per-(m, role) NTTP getters at the bottom are CPU-side only.

/// \cond
// Not API: the degree arithmetic the entries are compiled from. See the
// headers that name it for the entries it serves.

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
/// scheme that sums it and the tables hold different numbers over the same
/// pieces, so the basis is part of the degree table's identity: the tail of
/// kCoeffs says nothing about the value HornerMono returns, and vice versa.
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
/// w(b) = max(1, b^n / prod_{j=0}^{n-1}(j+1/2)), increasing in x, so the
/// worst sits at x = b (the piece's right end).
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

// The extended band (below kX0) is the m = 1
// lane's: its per-order amplification for the m > 1 machinery needs its
// own derivation (the band's effective edge varies per order, unlike the
// fixed kX0 worst case region B's RegionBAmplification assumes), which the
// m > 1 branch does not carry; it keeps the region-A treatment in the band.

/// The k-th stored coefficient. The read goes through the table's data pointer:
/// std::array's subscript is a checked one, and the check is a constant
/// expression of its own - one per coefficient read, which over a table this
/// size is a step cost larger than the sum it feeds. It is no part of the
/// criterion, and the element read is the same one either way.
template <typename CoeffArray>
constexpr double CoefficientAt(const CoeffArray& coeffs, std::size_t index) noexcept {
    return static_cast<double>(coeffs.data()[index]);
}

/// The dropped-coefficient tail Delta(d') = sum_{k=d'+1}^{deg} |c_k|, over
/// whichever table is handed in: the scan is basis-blind, and the basis is
/// which table reaches it.
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

/// The effective degree: the smallest admissible truncation (most
/// relaxation) in the ClenshawSplit evaluation domain {0,1,2,4,6,...} —
/// degree 0/1/2 are special-cased by the evaluator, even degrees >= 4 are
/// the split-Clenshaw domain; the scan returns deg (no truncation) when
/// nothing smaller is admissible, so the m = 1 path is today's path by
/// construction. Delta(deg) = 0, so the fallback is always admissible and the
/// returned degree is never one whose tail the budget cannot carry.
/// kAccuracyMultiplier is a runtime quantity here so the CUDA host side can
/// fill tables for an arbitrary m.
template <typename CoeffArray>
constexpr int EffectiveDegree(const CoeffArray& coeffs,
                              std::size_t offset,
                              int deg,
                              double m,
                              double amplification,
                              double bRegion) noexcept {
    const double budget = (m - 1.0) * bRegion;

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

// Which of a piece table's two stored forms the scan reads: the Chebyshev
// table or the monomial one. Every table this library stores a rung's degrees
// for is carried in both, so the choice is the basis and nothing else, and a
// table that carried one form would be a table to store the other for. It
// stands above the derivations rather than beside them because each of them
// names it with explicit template arguments over a dependent array, and a
// compiler that looks the name up where the derivation is written would not
// find it below.
template <TailBasis kBasis, typename Table>
constexpr const auto& TailTable(const Table& chebyshev, const Table& monomial) noexcept {
    return (kBasis == TailBasis::kChebyshev) ? chebyshev : monomial;
}

// The narrow partition's own effective-degree tables. The narrow pieces carry
// their own coefficients, so the criterion above applies to them unchanged: the
// tail a rung drops from *this* piece's stored coefficients, times the path's
// amplification, against the rung's budget. Nothing about the derivation is the
// shipped partition's - a narrow piece is a different fit over a different
// interval, so the tail it drops at a degree and the gain it is read under are
// its own numbers.
//
// Region A's pieces are cut per order rather than shared across them, so the
// table is flat over `kNarrowAPieces` exactly as the shipped one is over
// `kPieces` - one entry per row of the partition, at that row's own degree -
// and the amplification is the same region A gain the shipped derivation uses
// for the role.
//
// Each lane stores a narrow partition of its own - the double lane's pieces
// cut per order at degree 10, the single-precision lane's cut per order at
// degree 6 - so which table the criterion scans is which lane's role reads its
// region-A value from, exactly as RegionADegrees' two branches split. A batch
// role seeds its recursion from the double lane's fits whatever precision it
// returns (RoleUsesDoubleTables), so the two batch roles beside the two double
// roles scan the double lane's narrow pieces, and the two single roles scan
// theirs.
template <double kAccuracyMultiplier, BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
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
                                    kAccuracyMultiplier,
                                    amplification,
                                    RegionABudget(kRole));
            }
        }

        return degrees;
    } else
    {
        // The single-precision lane's own narrow pieces, in whichever of the
        // two forms its scheme sums; the pieces are cut per order here too, so
        // the table is flat over them and a row is the order's own value.
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
                                    kAccuracyMultiplier,
                                    amplification,
                                    RegionABudget(kRole));
            }
        }

        return degrees;
    }
}

// The narrow partition of region B. Its seed is one polynomial per piece, and
// an argument selects the piece it falls in, so a row is the pair (piece,
// order): the piece whose tail the cut drops, and the order the seed's error
// reaches. Flat, indexed piece * (kMaxOrder + 1) + order, for the reason the
// other tables are flat.
//
// The gain is the shipped region B one, A_B(n), because the seed is carried up
// the same recursion either partition feeds: which pieces the seed was cut
// from is what the partition decides, and no piece of it reaches the output
// orders by another path.
//
// Region B's seed is the lane's own either way - a float return is a float
// seed, cut at the float budget - so the split here is by precision and not by
// the region-A rule above: the double roles scan the double lane's narrow seed
// pieces and the single-precision roles scan theirs.
template <double kAccuracyMultiplier, BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
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
                                    kAccuracyMultiplier,
                                    RegionBAmplification(order),
                                    RegionBBudget(kRole));
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
                                    kAccuracyMultiplier,
                                    RegionBAmplification(order),
                                    RegionBBudget(kRole));
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
// k' = min(d', k). What the cut costs is not a coefficient sum, because the
// value is a quotient. With dP = P - P' and dQ = Q - Q' the dropped parts,
//
//   R - R' = P/Q - P'/Q'
//          = ((P' + dP) Q' - P' (Q' + dQ)) / (Q Q')
//          = dP/Q - R' * dQ/Q,
//
// which is exact. Every basis function of the mapped argument is at most one in
// modulus on [-1, 1], so |dP(t)| <= DP(d') = sum_{j>m'} |p_j| and
// |dQ(t)| <= DQ(d') = sum_{j>k'} |q_j| over the piece; |Q(t)| >= Qlo by the
// floor below; and |R'(t)| <= sum_{j<=m'} |p_j| / inf|Q'| <= SP / (Qlo - DQ(d')),
// using |Q'| >= |Q| - |dQ| >= Qlo - DQ(d') and |P'| <= SP = sum_{j<=m} |p_j|.
// The pairwise tail is therefore
//
//   Delta(d') = ( DP(d') + (SP / (Qlo - DQ(d'))) * DQ(d') ) / Qlo,
//
// two terms because a quotient's perturbation has two: the numerator's own
// dropped tail against the denominator's floor, and the denominator's dropped
// tail carried by the pair's value scale. Neither is a coefficient sum on its
// own, and the second is why a pair cannot be read as one series: cutting the
// denominator moves every value the pair returns, and how far it moves them is
// the pair's own size rather than a stored number.
//
// Qlo, the denominator's floor over the piece. Q does not vanish on these
// intervals - a pair whose denominator did would not approximate anything there
// - but sum_j |q_j| exceeds 1 on the shipped pieces, so the triangle bound
// 1 - sum_j |q_j| says nothing. The floor is taken from Q's own values on a
// grid, less what Q can move between grid points: with h the spacing of an
// N-point grid over [-1, 1] and |Q'| <= sum_j j|q_j| on it,
//
//   Qlo = min_i |Q(t_i)| - (sum_j j |q_j|) * h / 2.
//
// A Qlo at or below DQ(d') makes that cut inadmissible rather than admissible:
// the pair's own value scale is then unbounded by this argument, and the cut is
// skipped like one outside the evaluator's domain.
//
// The cut is in the same domain as the polynomial scan, and the fallback is the
// same shape. At d' = max(m, k) the cut is the whole pair, Delta = 0 exactly
// because dP = dQ = 0, so the scan always has an admissible order and the
// returned pair is never one the budget cannot carry. That is this family's
// answer to the polynomial side's Delta(deg) = 0: the rational equivalent is a
// pair cut to itself, which returns the shipped pair's value bit for bit.

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

/// Qlo: the denominator's floor over the mapped interval, from the grid bound
/// above. den[denOffset + j - 1] holds q_j for j = 1..denDeg, and
/// Q(t) = 1 + sum_j q_j t^j.
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

/// The pair's dropped-order measure at one cut, with the two cut-INDEPENDENT
/// operands of the criterion passed in. `Qlo` and `SP` are properties of the
/// stored pair rather than of the order it is read at, and a caller that scans
/// every candidate cut asks the same two questions once per cut otherwise - the
/// floor is a 65-point grid. Passing them in is what keeps the whole table
/// inside a compiler's constant-expression step budget; the value returned is
/// identical to the three-argument form's, because neither operand depends on
/// \p dPrime.
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

/// The pair's dropped-order measure at one cut; see the block comment above.
/// A cut whose floor does not clear the denominator's own dropped tail has no
/// bound under this argument at all - the pair's value scale is then unbounded
/// by it - and \p admissible reports that rather than a number, so the caller
/// skips the cut instead of accepting a tail that does not mean anything.
///
/// This form takes the two operands from the pair itself, which is what a
/// caller measuring one cut wants; `RationalPairTailAt` is the same measure
/// with them supplied.
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
/// domain {0,1,2,4,6,...}, which truncates the numerator and the denominator
/// together. The full pair is the fallback and is admissible by construction
/// (its dropped parts are both empty, so it costs nothing).
/// kAccuracyMultiplier is a runtime quantity here, as it is on the polynomial
/// side, so a table can be filled for an arbitrary m.
template <typename NumArray, typename DenArray>
constexpr void RationalPairCut(const NumArray& num,
                               std::size_t numOffset,
                               const DenArray& den,
                               std::size_t denOffset,
                               int numDeg,
                               int denDeg,
                               double m,
                               double amplification,
                               double bRegion,
                               int& outNumDeg,
                               int& outDenDeg) noexcept {
    const double budget = (m - 1.0) * bRegion;
    const int full = numDeg > denDeg ? numDeg : denDeg;

    // The two cut-independent operands, taken once for the pair: the
    // denominator's floor and the numerator's scale. Every candidate cut below
    // asks the same two questions, and the floor is a 65-point grid.
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
// std::array<int, ...> (one entry per region-A piece / per order for
// region B; the flat form keeps the tables constexpr on MSVC). The NTTP
// forms are instantiation-local constants — zero mutable state on the CPU
// path.
// The region-A degrees of one role and one basis. The basis selects the table
// the tail is summed from, and each lane stores both forms of its own fits, so
// the branch below is which lane's tables the role reads - the double lane's
// stored piece table or the float lane's - and the basis picks the form within
// it: the Chebyshev coefficients ClenshawSplit reads, or the monomial ones
// HornerMono reads, over the same pieces at the same degrees.
template <double kAccuracyMultiplier, BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
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
                                    kAccuracyMultiplier,
                                    amplification,
                                    RegionABudget(kRole));
            }
        }

        return degrees;
    } else
    {
        // The single-precision roles' own table, in whichever of its two
        // stored forms the rung's scheme sums: the lane holds a Chebyshev
        // table and a monomial one over the same pieces, so a Horner rung on
        // it is cut from the monomial table exactly as the double lane's is.
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
                                    kAccuracyMultiplier,
                                    amplification,
                                    RegionABudget(kRole));
            }
        }

        return degrees;
    }
}

// The region-B degrees of one role and one basis; see RegionADegrees for what
// the basis selects and why each lane's own seed table is the one its role
// reads.
template <double kAccuracyMultiplier, BoysRole kRole, TailBasis kBasis = TailBasis::kChebyshev>
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
                                                                       kAccuracyMultiplier,
                                                                       RegionBAmplification(order),
                                                                       RegionBBudget(kRole));
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
                                                                       kAccuracyMultiplier,
                                                                       RegionBAmplification(order),
                                                                       RegionBBudget(kRole));
        }

        return degrees;
    }
}

// The rational pair tables at a rung: per region-A piece and per region-B
// order, the numerator and the denominator degree the pair criterion certifies.
// Two flat int arrays rather than one of pairs, for the same reason the
// polynomial tables are flat.
//
// The amplification each region's cut pays is the path's own, and the rational
// route's path is not the shipped one:
//
//  - Region A reads one piece per order. The lane contract's finding that a
//    piece fitted for the values alone does not survive the batch downward
//    recursion's gain is why the route is read this way, and it is what makes
//    A = 1 here rather than the batch role's w(b): the cut's error reaches the
//    value the piece was read for with nothing in between, where the shipped
//    route's seed reaches F_0 through the recursion and is amplified by it.
//  - Region B reads its seed at order 0 and carries it up, so its cut pays the
//    shipped table's own A_B(n) = prod_{j<n}(j+1/2) / kX0^n.
//
// The budget is the batch role's region budget at both: the tier the rung is
// named by documents m * 5.5e-14 in every region, so the m = 1 base the
// criterion spends is that bound's own base.
struct RationalRegionAPairs {
    std::array<int, std::size(kPieces)> num{};
    std::array<int, std::size(kPieces)> den{};
};

struct RationalRegionBPairs {
    std::array<int, kMaxOrder + 1> num{};
    std::array<int, kMaxOrder + 1> den{};
};

// One order's row of the double lane's stored region-A pairs. The table above
// is flat over the pieces and the pieces are grouped by order, so a row is the
// order's own pieces and the table is the rows laid end to end.
//
// A row is cut in a constant expression of its own rather than inside the
// assembly's single evaluation, because the budget a compiler allows is spent
// per evaluation and a row is a small fraction of a table. The rows hold the
// same cuts either way: this is how the table is obtained, not what it holds.
//
// The per-order reading and the batch seed's reading differ in one thing - the
// amplification the cut is judged at - so one cutter serves both: a per-order
// reading pays A = 1, and a batch seed pays w(b) at the piece's right end.
template <int kOrder, bool kSeedReading> struct RationalARow {
    static constexpr int kFirst = kPieceStart[kOrder];
    static constexpr int kCount = kPieceStart[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <double kAccuracyMultiplier, BoysRole kRole, bool kSeedReading, int kOrder>
constexpr RationalARow<kOrder, kSeedReading> CutRationalARow() noexcept {
    constexpr double kBudget = RegionABudget(kRole);
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
                        kAccuracyMultiplier,
                        kSeedReading ? RegionAAmplification(kOrder, piece.b) : 1.0,
                        kBudget,
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// A variable template's initializer is a constant expression of its own, so each
// row above is cut under a budget of its own rather than under the one the
// assembly is evaluated with.
template <double kAccuracyMultiplier, BoysRole kRole, bool kSeedReading, int kOrder>
inline constexpr RationalARow<kOrder, kSeedReading> kRationalARow =
    CutRationalARow<kAccuracyMultiplier, kRole, kSeedReading, kOrder>();

template <double kAccuracyMultiplier, BoysRole kRole, bool kSeedReading, int kOrder>
constexpr void CopyRationalARow(RationalRegionAPairs& pairs) noexcept {
    const RationalARow<kOrder, kSeedReading> row =
        kRationalARow<kAccuracyMultiplier, kRole, kSeedReading, kOrder>;

    for (int i = 0; i < RationalARow<kOrder, kSeedReading>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(RationalARow<kOrder, kSeedReading>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <double kAccuracyMultiplier, BoysRole kRole, bool kSeedReading, std::size_t... kOrders>
constexpr RationalRegionAPairs AssembleRationalA(std::index_sequence<kOrders...>) noexcept {
    RationalRegionAPairs pairs{};
    (CopyRationalARow<kAccuracyMultiplier, kRole, kSeedReading, static_cast<int>(kOrders)>(pairs),
     ...);
    return pairs;
}

template <double kAccuracyMultiplier>
constexpr RationalRegionAPairs RationalRegionADegrees() noexcept {
    return AssembleRationalA<kAccuracyMultiplier, BoysRole::kDoubleBatch, false>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

template <double kAccuracyMultiplier>
constexpr RationalRegionBPairs RationalRegionBDegrees() noexcept {
    constexpr double kBudget = RegionBBudget(BoysRole::kDoubleBatch);
    RationalRegionBPairs pairs{};

    // One seed for the region rather than one fit per order, so the cut is the
    // same pair at every order: the table is the seed's pair repeated, and it is
    // kept per order so a reader reads it the way the polynomial table reads.
    // The seed's own cut is judged at order 0's amplification - A_B(0) = 1, the
    // smallest the region pays - because the pair is one evaluation: every
    // order's output carries the same cut, and the amplification is a property
    // of the order it is read for, not of the seed.
    int numDeg = 0;
    int denDeg = 0;

    RationalPairCut(kRatBnum,
                    0,
                    kRatBden,
                    0,
                    kRatBnumDeg,
                    kRatBdenDeg,
                    kAccuracyMultiplier,
                    RegionBAmplification(0),
                    kBudget,
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
// The narrow partition carries a numerator/denominator pair per piece of its
// own region-A cover and one pair per piece for its region-B seed, and they are
// different stored numbers from the shipped partition's pairs over the same
// domain: a different cover, a different fit, different roundings. A rung of
// the narrow partition is therefore a cut of THESE pairs. Cutting the shipped
// partition's pairs and reading them over the narrow pieces would answer with
// the shipped fit under the narrow partition's name, which is the substitution
// this pair of derivations exists to make unnecessary.
//
// The criterion is the one the shipped pairs are cut by, and the reading is the
// same reading: region A's piece is the order's value and nothing amplifies the
// cut, so A = 1; region B's seed is one evaluation carried up from order 0, so
// its cut is judged at A_B(0) = 1. Both spend the batch role's region budget,
// as the shipped pair tables do, because the budget and the reading are
// properties of the region and the recursion rather than of the entry that
// reads the rung.
struct NarrowRationalRegionAPairs {
    std::array<int, std::size(kNarrowAPieces)> num{};
    std::array<int, std::size(kNarrowAPieces)> den{};
};

struct NarrowRationalRegionBPairs {
    std::array<int, kNarrowBPieces> num{};
    std::array<int, kNarrowBPieces> den{};
};

template <double kAccuracyMultiplier>
constexpr NarrowRationalRegionAPairs RationalRegionANarrowDegrees() noexcept {
    constexpr double kBudget = RegionABudget(BoysRole::kDoubleBatch);
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
                        kAccuracyMultiplier,
                        1.0,
                        kBudget,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

template <double kAccuracyMultiplier>
constexpr NarrowRationalRegionBPairs RationalRegionBNarrowDegrees() noexcept {
    constexpr double kBudget = RegionBBudget(BoysRole::kDoubleBatch);
    NarrowRationalRegionBPairs pairs{};

    // The narrow seed is one pair per piece rather than one pair for the whole
    // region, so each piece's pair is cut against its own coefficients. The
    // reading is the shipped seed's - the pair is one evaluation whose output is
    // carried up the recursion, so it is judged at order 0's amplification.
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
                        kAccuracyMultiplier,
                        RegionBAmplification(0),
                        kBudget,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

// The narrow pairs at the batch seed's reading. The shipped partition carries
// both readings - RationalRegionADegrees for a per-order read of one piece and
// RationalRegionASeedDegrees for a seed carried down the recursion - so the
// narrow partition carries both as well, and this is the second: the same
// narrow pieces and the same stored pairs, cut at the amplification the
// recursion pays at the piece's right end rather than at A = 1.
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr NarrowRationalRegionAPairs RationalRegionANarrowSeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalRegionANarrowDegrees instead");

    constexpr double kBudget = RegionABudget(kRole);
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
                            kAccuracyMultiplier,
                            RegionAAmplification(order, piece.b),
                            kBudget,
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// ---------------------------------------------------------------------------
// The rational pair at a batch seed's reading
// ---------------------------------------------------------------------------
// The cut above is the one a per-order reading of the route pays: the piece's
// value is the order's value and nothing amplifies the cut. An engine that
// seeds a batch from the top order's piece and recurses down instead pays the
// batch amplification at that piece's right end, which is the polynomial
// tables' w(b) - the same quantity RegionADegrees spends for a batch role. The
// cut below is the same pair over the same pieces at that reading, so a batch
// engine can seed a rung from the route's fit rather than reading the route
// per order.
//
// The pair is the double lane's stored one: an engine that needs a batch seed's
// precision reads this lane's rational fit, not the float lane's own pair (see
// the float lane's batch body for why a 1.5e-7 seed cannot start a batch).
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr RationalRegionAPairs RationalRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalRegionADegrees instead");

    return AssembleRationalA<kAccuracyMultiplier, kRole, true>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// One order's row of the narrow seed table: the same two flat rows as the whole
// table, sized to the pieces that order owns.
//
// The table is cut a row at a time because the cut of a piece depends on that
// piece and its order alone, and cutting all of them inside one constant
// expression is longer than a compiler is obliged to run one for: clang's
// default step budget refuses the whole-table evaluation at every relaxed
// multiplier, while a single row is a small fraction of that budget. The rows
// hold the same cuts either way - this is how the table is obtained, not what
// it contains.
template <int kOrder> struct NarrowRationalSeedRow {
    static constexpr int kFirst = kNarrowAPieceStart[kOrder];
    static constexpr int kCount = kNarrowAPieceStart[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
constexpr NarrowRationalSeedRow<kOrder> CutNarrowRationalSeedRow() noexcept {
    constexpr double kBudget = RegionABudget(kRole);
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
                        kAccuracyMultiplier,
                        RegionAAmplification(kOrder, piece.b),
                        kBudget,
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// One order's row of the narrow seed table. A variable template is initialized
// in a constant expression of its own, so the row above is cut under a budget
// of its own rather than under the one the assembly below is evaluated with.
template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
inline constexpr NarrowRationalSeedRow<kOrder> kNarrowRationalSeedRow =
    CutNarrowRationalSeedRow<kAccuracyMultiplier, kRole, kOrder>();

template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
constexpr void CopyNarrowRationalSeedRow(NarrowRationalRegionAPairs& pairs) noexcept {
    const NarrowRationalSeedRow<kOrder> row =
        kNarrowRationalSeedRow<kAccuracyMultiplier, kRole, kOrder>;

    for (int i = 0; i < NarrowRationalSeedRow<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalSeedRow<kOrder>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <double kAccuracyMultiplier, BoysRole kRole, std::size_t... kOrders>
constexpr NarrowRationalRegionAPairs AssembleNarrowRationalSeed(
    std::index_sequence<kOrders...>) noexcept {
    NarrowRationalRegionAPairs pairs{};
    (CopyNarrowRationalSeedRow<kAccuracyMultiplier, kRole, static_cast<int>(kOrders)>(pairs), ...);
    return pairs;
}

// The same batch reading over the double lane's NARROW pieces. A batch role
// whose policy names the narrow partition seeds its recursion from that
// partition's rational fit (see the float lane's batch body), so the cut this
// table carries is the one such a seed is read at: the same pair criterion over
// the narrow pieces, judged at each piece's own w(b).
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr NarrowRationalRegionAPairs NarrowRationalRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the rational pair is the batch recursion's - the cut is "
                  "judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads the single-order reading instead");

    return AssembleNarrowRationalSeed<kAccuracyMultiplier, kRole>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// ---------------------------------------------------------------------------
// The single-precision lane's own rational pairs at a rung
// ---------------------------------------------------------------------------
// The float lane carries the rational family as its own stored tables: a
// numerator/denominator pair per piece of its own region-A cover, and one pair
// for its region-B seed. They are different stored numbers from the double
// lane's pair over the same region - their own cover, their own fit, their own
// roundings - so a rung of this family on this lane is a cut of THESE tables,
// derived by the same pair criterion and read at the role's own budget.
//
// Region A's reading is the single-order one: the piece is the order's value
// and nothing amplifies the cut, so A = 1 exactly as the double lane's
// per-order route pays it. (The float lane's batch seeds its recursion from the
// double lane's pair, which is the seed table above and not this one.)
struct RationalRegionAF32Pairs {
    std::array<int, std::size(f32::kRatAPieces)> num{};
    std::array<int, std::size(f32::kRatAPieces)> den{};
};

struct RationalRegionBF32Pairs {
    std::array<int, kMaxOrder + 1> num{};
    std::array<int, kMaxOrder + 1> den{};
};

template <double kAccuracyMultiplier, BoysRole kRole>
constexpr RationalRegionAF32Pairs RationalRegionAF32Degrees() noexcept {
    constexpr double kBudget = RegionABudget(kRole);
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
                            kAccuracyMultiplier,
                            RoleUsesBatchAmplification(kRole)
                                ? RegionAAmplification(order, static_cast<double>(piece.b))
                                : 1.0,
                            kBudget,
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// The float lane's region-B seed pair at a rung; the same shape as the double
// lane's table above and for the same reason - the region holds one seed rather
// than one fit per order, so the cut is one pair repeated down the table, judged
// at order 0's amplification, which is the largest this region pays: every
// order's gain is A_B(n) = prod_{j<n}(j+1/2)/kX0^n, and kX0 is the argument at
// which that product first reaches one, so A_B(n) <= A_B(0) = 1 for every order
// the table covers.
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr RationalRegionBF32Pairs RationalRegionBF32Degrees() noexcept {
    constexpr double kBudget = RegionBBudget(kRole);
    RationalRegionBF32Pairs pairs{};

    int numDeg = 0;
    int denDeg = 0;

    RationalPairCut(f32::kRatBnum,
                    0,
                    f32::kRatBden,
                    0,
                    f32::kRatBnumDeg,
                    f32::kRatBdenDeg,
                    kAccuracyMultiplier,
                    RegionBAmplification(0),
                    kBudget,
                    numDeg,
                    denDeg);

    for (int order = 0; order <= kMaxOrder; ++order)
    {
        pairs.num[static_cast<std::size_t>(order)] = numDeg;
        pairs.den[static_cast<std::size_t>(order)] = denDeg;
    }

    return pairs;
}

// The same two tables over the float lane's narrow partition: its own cover,
// its own stored pairs, cut by the same criterion at the role's budget. A rung
// of the rational route under a policy that names the narrow partition reads
// these, because that is the pair the m = 1 call of the same policy evaluates.
struct NarrowRationalRegionAF32Pairs {
    std::array<int, std::size(f32::kNarrowRatAPiecesF32)> num{};
    std::array<int, std::size(f32::kNarrowRatAPiecesF32)> den{};
};

struct NarrowRationalRegionBF32Pairs {
    std::array<int, static_cast<std::size_t>(f32::kNarrowRatBPiecesCountF32)> num{};
    std::array<int, static_cast<std::size_t>(f32::kNarrowRatBPiecesCountF32)> den{};
};

// One order's row of the float lane's narrow pairs, cut in a constant
// expression of its own. The budget a compiler allows is spent per evaluation,
// and a row is a small fraction of a table, so the rows are cut one at a time
// and the assembly below only copies them; the rows hold the same cuts either
// way, and this is how the table is obtained, not what it contains.
template <int kOrder> struct NarrowRationalF32Row {
    static constexpr int kFirst = f32::kNarrowRatAPieceStartF32[kOrder];
    static constexpr int kCount = f32::kNarrowRatAPieceStartF32[kOrder + 1] - kFirst;

    std::array<int, static_cast<std::size_t>(kCount)> num{};
    std::array<int, static_cast<std::size_t>(kCount)> den{};
};

template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
constexpr NarrowRationalF32Row<kOrder> CutNarrowRationalF32Row() noexcept {
    constexpr double kBudget = RegionABudget(kRole);
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
                        kAccuracyMultiplier,
                        RoleUsesBatchAmplification(kRole)
                            ? RegionAAmplification(kOrder, static_cast<double>(piece.b))
                            : 1.0,
                        kBudget,
                        row.num[static_cast<std::size_t>(i)],
                        row.den[static_cast<std::size_t>(i)]);
    }

    return row;
}

// A variable template's initializer is a constant expression of its own, so each
// row above is cut under a budget of its own rather than under the one the
// assembly is evaluated with.
template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
inline constexpr NarrowRationalF32Row<kOrder> kNarrowRationalF32Row =
    CutNarrowRationalF32Row<kAccuracyMultiplier, kRole, kOrder>();

template <double kAccuracyMultiplier, BoysRole kRole, int kOrder>
constexpr void CopyNarrowRationalF32Row(NarrowRationalRegionAF32Pairs& pairs) noexcept {
    const NarrowRationalF32Row<kOrder> row =
        kNarrowRationalF32Row<kAccuracyMultiplier, kRole, kOrder>;

    for (int i = 0; i < NarrowRationalF32Row<kOrder>::kCount; ++i)
    {
        const std::size_t index =
            static_cast<std::size_t>(NarrowRationalF32Row<kOrder>::kFirst + i);
        pairs.num[index] = row.num[static_cast<std::size_t>(i)];
        pairs.den[index] = row.den[static_cast<std::size_t>(i)];
    }
}

template <double kAccuracyMultiplier, BoysRole kRole, std::size_t... kOrders>
constexpr NarrowRationalRegionAF32Pairs AssembleNarrowRationalF32(
    std::index_sequence<kOrders...>) noexcept {
    NarrowRationalRegionAF32Pairs pairs{};
    (CopyNarrowRationalF32Row<kAccuracyMultiplier, kRole, static_cast<int>(kOrders)>(pairs), ...);
    return pairs;
}

template <double kAccuracyMultiplier, BoysRole kRole>
constexpr NarrowRationalRegionAF32Pairs NarrowRationalRegionAF32Degrees() noexcept {
    return AssembleNarrowRationalF32<kAccuracyMultiplier, kRole>(
        std::make_index_sequence<kMaxOrder + 1>{});
}

// Region B's narrow seed is one pair per narrow piece, and which piece an
// argument falls in is the argument's own business, so the cut is the same
// reading as the shipped one taken once per piece rather than once for the
// region: the region holds f32::kNarrowRatBPiecesF32 seeds where the shipped
// row holds one.
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr NarrowRationalRegionBF32Pairs NarrowRationalRegionBF32Degrees() noexcept {
    constexpr double kBudget = RegionBBudget(kRole);
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
                        kAccuracyMultiplier,
                        RegionBAmplification(0),
                        kBudget,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}
#endif // !defined(__CUDACC__)


// ---------------------------------------------------------------------------
// The same tables under the names the device lane's entries read them by
// ---------------------------------------------------------------------------
// The rational pair at the narrow partition
// ---------------------------------------------------------------------------
// The narrow partition's own pairs, cut by the criterion above. The pieces, the
// intervals and the coverage are NarrowRegionAPartition's — the partition is not
// the family's and this member brings the pairs to it — so the derivation is the
// shipped one's at the other table: same layout, same criterion, different
// stored numbers, and a degree certified for one table's fit is not a degree
// certified for the other's.
//
// Region A's two readings are the shipped route's two, for the same reason: a
// per-order reading pays A = 1, because the cut reaches the value the piece was
// read for with nothing in between, and a batch seed pays the piece's own w(b),
// because the seed is carried down the recursion from the piece's right end.
// Region B's seed is one pair per narrow piece, read at order 0 and carried up
// the upward recursion, whose gain A_B(n) <= A_B(0) = 1 over this region, so the
// cut is judged at order 0 as the shipped route's is.
struct RationalNarrowRegionAPairs {
    std::array<int, std::size(kNarrowAPieces)> num{};
    std::array<int, std::size(kNarrowAPieces)> den{};
};

struct RationalNarrowRegionBPairs {
    std::array<int, kNarrowBPieces> num{};
    std::array<int, kNarrowBPieces> den{};
};

// The per-order reading: one pair per narrow piece, at A = 1.
template <double kAccuracyMultiplier>
constexpr RationalNarrowRegionAPairs RationalNarrowRegionADegrees() noexcept {
    constexpr double kBudget = RegionABudget(BoysRole::kDoubleBatch);
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
                        kAccuracyMultiplier,
                        1.0,
                        kBudget,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

// The batch seed's reading of the same table; see RationalRegionASeedDegrees
// for why the amplification is the piece's own w(b) here and 1 above.
template <double kAccuracyMultiplier, BoysRole kRole>
constexpr RationalNarrowRegionAPairs RationalNarrowRegionASeedDegrees() noexcept {
    static_assert(RoleUsesBatchAmplification(kRole),
                  "this reading of the narrow rational pair is the batch recursion's - the cut "
                  "is judged at the piece's own w(b) because the seed is carried down by it - so "
                  "it belongs to a batch role; a role whose region-A seed is the order's own "
                  "value pays A = 1 and reads RationalNarrowRegionADegrees instead");

    constexpr double kBudget = RegionABudget(kRole);
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
                            kAccuracyMultiplier,
                            RegionAAmplification(order, piece.b),
                            kBudget,
                            pairs.num[index],
                            pairs.den[index]);
        }
    }

    return pairs;
}

// The narrow region-B seed's pairs, one per piece, at order 0's amplification.
template <double kAccuracyMultiplier>
constexpr RationalNarrowRegionBPairs RationalNarrowRegionBDegrees() noexcept {
    constexpr double kBudget = RegionBBudget(BoysRole::kDoubleBatch);
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
                        kAccuracyMultiplier,
                        RegionBAmplification(0),
                        kBudget,
                        pairs.num[index],
                        pairs.den[index]);
    }

    return pairs;
}

} // namespace detail
} // namespace boys

/// \endcond
