#pragma once

/// The evaluation policy Paper A names, the recurrence replica both sweeps
/// admit against the shipped entry, and the seed-fit summation the second
/// sweep varies.
///
/// The policy is pinned to the evaluation the paper describes: the Chebyshev
/// fits on the coarsest partition, degree 20 in A1 and 18 in A2, summed by
/// the split Clenshaw recurrence, in the double lane, with the
/// refined-reciprocal division and the fast region-B exponential.  It is not
/// the v5.0.0 build default: an unnamed host fp64 all-orders call resolves to
/// its class row, the uniform partition summed by Horner over the orders
/// axis, and the seam fallback is the narrow partition summed by Horner.  The
/// paper's harness subsection says so and names the cells for exactly that
/// reason.
///
/// The replica repeats the step arithmetic of `boys/boys_fit_core.hpp` in
/// this translation unit, so that "the recurrence the harness runs" is a
/// separate body from the library's.  Admission is bitwise: every cell the
/// shipped entry serves by recursion must match.

#include "boys/boys_coefficients_d_coarse_cheb.hpp"
#include "boys/boys_coefficients_d_far.hpp"
#include "boys/boys.hpp"

#include <bit>
#include <cmath>
#include <cstdint>

namespace paper_a {

using PaperPolicy = boys::StatedEvalPolicy<
    boys::FitRoute::kChebyshev,
    boys::EvalScheme::kSplitClenshaw,
    boys::BoysBudget::kFloat,
    boys::PackAxis::kArguments,
    boys::FitGranularity::kCoarsest,
    boys::DivisionForm::kRefinedReciprocal,
    boys::RegionBExp::kFast>;

static_assert(boys::EvalPolicyLike<PaperPolicy>);

inline constexpr int kNmax = boys::kMaxBoysOrder;

struct Admission {
    const char* direction;
    int checked;
    int matched;
};

inline bool SameBits(double a, double b) noexcept {
    return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
}

/// One division in the refined-reciprocal form the policy names: the plain
/// product, then the classical correction through the divisor's reciprocal.
inline double RefinedDivide(double a, double divisor, double reciprocal) noexcept {
    const double quotient = a * reciprocal;

    if (std::isinf(divisor)) {
        return quotient;
    }

    return std::fma(std::fma(-quotient, divisor, a), reciprocal, quotient);
}

/// The upward step F_l from F_{l-1}, with the divisor and its reciprocal.
inline double UpwardStep(int l, double f, double x, double reciprocal, double half_exp) noexcept {
    return RefinedDivide((static_cast<double>(l) - 0.5) * f - half_exp, x, reciprocal);
}

/// The downward step F_l from F_{l+1}; the divisor l + 1/2 is exact.
inline double DownwardStep(int l, double f, double x, double half_exp) noexcept {
    const double divisor = static_cast<double>(l) + 0.5;
    return RefinedDivide(x * f + half_exp, divisor, 1.0 / divisor);
}

/// Run the shipped policy's own path from its own seed and compare every
/// recursed cell with the library's value, bit for bit.  x == 0 is the exact
/// closed form and is checked separately.
inline Admission AdmitReplica(double x, const double* shipped) noexcept {
    using Fit = typename PaperPolicy::Fit;

    if (x == 0.0) {
        int matched = 0;

        for (int l = 0; l <= kNmax; ++l) {
            matched += SameBits(shipped[l], 1.0 / (2.0 * l + 1.0)) ? 1 : 0;
        }

        return {"exact", kNmax + 1, matched};
    }

    if (x >= Fit::kRegionAFitsFrom) {
        int served = 0;

        while (served < kNmax && x >= boys::detail::kTierThresholds[served + 1]) {
            ++served;
        }

        typename Fit::template BandSource<PaperPolicy::kDivision> source(x);
        double f = source.Next(0, x);
        int checked = 1;
        int matched = SameBits(f, shipped[0]) ? 1 : 0;
        const double half_exp = 0.5 * std::exp(-x);
        const double reciprocal = 1.0 / x;

        for (int l = 1; l <= served; ++l) {
            f = UpwardStep(l, f, x, reciprocal, half_exp);
            ++checked;
            matched += SameBits(f, shipped[l]) ? 1 : 0;
        }

        return {"upward", checked, matched};
    }

    double f = boys::detail::PolicyRegionAValue<PaperPolicy>(kNmax, x);
    int checked = 1;
    int matched = SameBits(f, shipped[kNmax]) ? 1 : 0;
    const double half_exp = 0.5 * std::exp(-x);

    for (int l = kNmax - 1; l >= 0; --l) {
        f = DownwardStep(l, f, x, half_exp);
        ++checked;
        matched += SameBits(f, shipped[l]) ? 1 : 0;
    }

    return {"downward", checked, matched};
}

/// A seed fit's value: its Chebyshev coefficients at the band's own mapped
/// argument, summed by Clenshaw in double.
inline double SeedFit(const double* coefficients, int degree, double x, double left, double right) noexcept {
    const double t = 2.0 * (x - left) / (right - left) - 1.0;
    double b1 = 0.0;
    double b2 = 0.0;

    for (int j = degree; j >= 1; --j) {
        const double b0 = 2.0 * t * b1 - b2 + coefficients[j];
        b2 = b1;
        b1 = b0;
    }

    return t * b1 - b2 + coefficients[0];
}

} // namespace paper_a
