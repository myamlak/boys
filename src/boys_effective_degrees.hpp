#pragma once

// The compile-time effective-degree machinery behind the accuracy-multiplier
// parametrization.
// The multiplier m relaxes each lane's *asserted*
// per-region bounds B_region by truncating the Chebyshev seed fits to the
// effective degree
//
//   d'(m) = min { d' in {0,1,2,4,6,...} : Delta(d') * A <= (m-1) * B_region },
//   Delta(d') = sum_{k=d'+1}^{d} |c_k|   (the dropped-coefficient tail),
//
// with A the path's seed-error amplification: 1 for the single-style lanes,
// w(b) = max(1, b^n / prod(j+1/2)) at the piece's right end for the region-A
// batch downward recursion, and prod(j+1/2)/x0^n for the region-B upward
// recursion (worst at x = x0). |T_k(t)| <= 1 on the mapped interval, so the
// truncation delta is <= Delta(d') exactly, and the delivered error is
// <= base(m=1) + Delta*A <= m*B_region (the m = 1 base is the asserted
// contract). Everything here is constexpr over the emitted tables: the
// per-(order,piece) d' tables are compile-time constants per (m, lane role),
// a-priori, never tuned.
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
// strictly stronger than the accompanying manuscript's error-bounded
// 1.5e-7 + ½ULP (the suite's 1e-7 is the fp16 lanes' asserted base, not
// the float budget); the representation half-ULP term is m-independent).
//
// The CUDA host side (boys_cuda.cu) fills the device degree tables for a
// runtime m by calling EffectiveDegree directly (the non-template core);
// the per-(m, role) NTTP getters at the bottom are CPU-side only.

#include "boys_coefficients.hpp"

#include <array>
#include <cstddef>

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

// The extended band (the per-range seed design below kX0) is the m = 1
// lane's: its per-order amplification for the m > 1 machinery needs its
// own derivation (the band's effective edge varies per order, unlike the
// fixed kX0 worst case region B's RegionBAmplification assumes) - named
// future work; the m > 1 branch keeps the region-A treatment in the band.

/// The dropped-coefficient tail Delta(d') = sum_{k=d'+1}^{deg} |c_k|.
template <typename CoeffArray>
constexpr double CoefficientTail(const CoeffArray& coeffs,
                                 std::size_t offset,
                                 int deg,
                                 int dPrime) noexcept {
    double tail = 0.0;

    for (int k = dPrime + 1; k <= deg; ++k)
    {
        const double c = static_cast<double>(coeffs[offset + static_cast<std::size_t>(k)]);
        tail += (c >= 0.0) ? c : -c;
    }

    return tail;
}

/// The effective degree: the smallest admissible truncation (most
/// relaxation) in the ClenshawSplit evaluation domain {0,1,2,4,6,...} —
/// degree 0/1/2 are special-cased by the evaluator, even degrees >= 4 are
/// the split-Clenshaw domain; the scan returns deg (no truncation) when
/// nothing smaller is admissible, so the m = 1 path is today's path by
/// construction. kAccuracyMultiplier is a runtime quantity here so the CUDA
/// host side can fill tables for an arbitrary m.
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

#if !defined(__CUDACC__)
// The per-(m, role) compile-time d' tables. The degree tables are flat
// std::array<int, ...> (one entry per region-A piece / per order for
// region B; the flat form keeps the tables constexpr on MSVC). The NTTP
// forms are instantiation-local constants — zero mutable state on the CPU
// path.
template <double kAccuracyMultiplier, BoysRole kRole> constexpr auto RegionADegrees() noexcept {
    if constexpr (RoleUsesDoubleTables(kRole))
    {
        std::array<int, std::size(kPieces)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = kPieceStart[order]; p < kPieceStart[order + 1]; ++p)
            {
                const OrderPiece& piece = kPieces[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(kCoeffs,
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
        std::array<int, std::size(f32::kPieces)> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            for (int p = f32::kPieceStart[order]; p < f32::kPieceStart[order + 1]; ++p)
            {
                const f32::OrderPiece& piece = f32::kPieces[static_cast<std::size_t>(p)];
                const double amplification =
                    RoleUsesBatchAmplification(kRole) ? RegionAAmplification(order, piece.b) : 1.0;
                degrees[static_cast<std::size_t>(p)] =
                    EffectiveDegree(f32::kCoeffs,
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

template <double kAccuracyMultiplier, BoysRole kRole> constexpr auto RegionBDegrees() noexcept {
    if constexpr (kRole == BoysRole::kDoubleSingle || kRole == BoysRole::kDoubleBatch)
    {
        std::array<int, kMaxOrder + 1> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            degrees[static_cast<std::size_t>(order)] = EffectiveDegree(kBcoeffs,
                                                                       0,
                                                                       kBDeg,
                                                                       kAccuracyMultiplier,
                                                                       RegionBAmplification(order),
                                                                       RegionBBudget(kRole));
        }

        return degrees;
    } else
    {
        std::array<int, kMaxOrder + 1> degrees{};

        for (int order = 0; order <= kMaxOrder; ++order)
        {
            degrees[static_cast<std::size_t>(order)] = EffectiveDegree(f32::kBcoeffs,
                                                                       0,
                                                                       f32::kBDeg,
                                                                       kAccuracyMultiplier,
                                                                       RegionBAmplification(order),
                                                                       RegionBBudget(kRole));
        }

        return degrees;
    }
}
#endif // !defined(__CUDACC__)

} // namespace detail
} // namespace boys
