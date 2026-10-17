#pragma once

// The rational minimax route's fit families and the cut reads they serve.
// They are defined here rather than in boys_impl.hpp so that a translation unit parses
// only the group headers its policies need; the markers in each read are what name the
// group header a caller must add.

/// \cond

#include <cassert>
#include <cstddef>

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/policy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/dependent_false.hpp"
#include "boys/boys_fit_core.hpp"

namespace boys {

namespace detail {

// The rational minimax route: a numerator/denominator pair per piece, read by Horner
// in the same mapped argument, the denominator stored as q_1..q_k with q_0 held at
// 1. An alternative to the shipped route rather than a replacement: the same pieces,
// intervals and mapping, the other scheme.
struct RationalFit {
    // The shipped partition. This route's pieces are fitted over the shipped
    // pieces' own intervals - one minimax pair per piece - so there is no
    // second partition for it to read, and a policy naming both is refused
    // where it is named (see RouteFit).
    using Partition = ShippedRegionAPartition;

    // Below this argument the order's own piece is what the lane documents, at
    // the per-order 1e-15; at and above it the lane reads the order from the
    // band and documents the band's bound, which is the bar these fits hold.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    template <class Tag = void>
    static double EvalPiece(std::size_t index, double t) noexcept {
        static_assert(kCoefficientGroupPresent<DependentGroup<Tag, CoarseRatGroup>> ||
                          kAlwaysFalse<Tag>,
                      "this read needs boys/boys_coefficients_d_coarse_rat.hpp: include that "
                      "header before boys/boys.hpp in the translation unit that names the shipped rational route");
        const double* c = detail::kRatACoeffs.data() + detail::kRatAOffset[index];
        const int m = detail::kRatANumDeg[index];
        const int k = detail::kRatADenDeg[index];
        double num = c[m];

        for (int j = m - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, c[j]);
        }

        if (k == 0)
        {
            return num;
        }

        double den = c[m + k];

        for (int j = k - 1; j >= 1; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, c[m + j]);
        }

        return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
    }

    // The region-B seed from the region's own minimax pair.
    static double RegionBSeed(double x) noexcept {
        const double t = 2.0 * (x - kX0) / (kFlatHi - kX0) - 1.0;
        double num = detail::kRatBnum[detail::kRatBnumDeg];

        for (int j = detail::kRatBnumDeg - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, detail::kRatBnum[j]);
        }

        if constexpr (detail::kRatBdenDeg == 0)
        {
            return num;
        }

        double den = detail::kRatBden[detail::kRatBdenDeg - 1];

        for (int j = detail::kRatBdenDeg - 2; j >= 0; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, detail::kRatBden[j]);
        }

        den = backend::ScalarFp64::MulAdd(den, t, 1.0);
        return num / den;
    }

    // The band's orders: this route has no seed to carry up - each of those
    // orders is its own fit - so the source carries no state.
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFit>(l, x); }
    };
};

/// The rational route over the narrow partition: the family's own degree pair per
/// narrow piece, read at that piece's own interval and mapped argument, in the same
/// stored form as the shipped member - the numerator ascending, then the
/// denominator's q_1..q_k with q_0 held at 1.
///
/// The partition is not the family's: it is the narrow Chebyshev partition's cut of
/// both regions (see FitGranularity), so the member brings only the pairs - region
/// A's one per kNarrowAPieces row, region B's one per narrow piece at the mapped
/// argument the Chebyshev seed on that piece uses, so the two routes over a piece
/// read one t.
///
/// Every pair was accepted at the criterion the shipped member's were - the bare
/// delivered error of the stored pair in the kernel's own arithmetic, under the
/// 3e-14 bar in region A and the 5e-14 one in region B - read at BOTH multiply-add
/// routes with the worse taken: a bound taken at one route is not a bound on the
/// other's evaluation.
struct RationalFitNarrow {
    // The narrow cut of region A; the pieces, their intervals and the piece
    // lookup are the Chebyshev narrow partition's own.
    using Partition = NarrowRegionAPartition;

    // The route hands each order over at its own region-A boundary, as the
    // shipped rational family does: below it the shipped lane's value is the one
    // that order is documented at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    template <class Tag = void>
    static double EvalPiece(std::size_t index, double t) noexcept {
        static_assert(kCoefficientGroupPresent<DependentGroup<Tag, NarrowRatGroup>> ||
                          kAlwaysFalse<Tag>,
                      "this read needs boys/boys_coefficients_d_narrow_rat.hpp: include that "
                      "header before boys/boys.hpp in the translation unit that names the narrow rational route");
        const double* c = detail::kNarrowRatACoeffs.data() + detail::kNarrowRatAOffset[index];
        const int m = detail::kNarrowRatANumDeg[index];
        const int k = detail::kNarrowRatADenDeg[index];
        double num = c[m];

        for (int j = m - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, c[j]);
        }

        if (k == 0)
        {
            return num;
        }

        double den = c[m + k];

        for (int j = k - 1; j >= 1; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, c[m + j]);
        }

        return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
    }

    // The region-B seed from the narrow piece the argument falls in, at that
    // piece's own pair and mapped argument.
    template <class Tag = void>
    static double RegionBSeed(double x) noexcept {
        static_assert(kCoefficientGroupPresent<DependentGroup<Tag, NarrowRatGroup>> ||
                          kAlwaysFalse<Tag>,
                      "this read needs boys/boys_coefficients_d_narrow_rat.hpp: include that "
                      "header before boys/boys.hpp in the translation unit that names the narrow rational route");
        const std::size_t index = static_cast<std::size_t>(NarrowBPieceOf(x));
        const double a = detail::kNarrowBEdges[index];
        const double b = detail::kNarrowBEdges[index + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;
        const double* c = detail::kNarrowRatBCoeffs.data() + detail::kNarrowRatBOffset[index];
        const int m = detail::kNarrowRatBNumDeg[index];
        const int k = detail::kNarrowRatBDenDeg[index];
        double num = c[m];

        for (int j = m - 1; j >= 0; --j)
        {
            num = backend::ScalarFp64::MulAdd(num, t, c[j]);
        }

        if (k == 0)
        {
            return num;
        }

        double den = c[m + k];

        for (int j = k - 1; j >= 1; --j)
        {
            den = backend::ScalarFp64::MulAdd(den, t, c[m + j]);
        }

        return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
    }

    // Each of the band's orders is its own fit, as it is on the shipped member.
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double) noexcept {}

        double Next(int l, double x) noexcept { return RegionAValue<RationalFitNarrow>(l, x); }
    };
};

// One rational piece at a cut pair, in the mapped argument the shipped evaluation reads it in. The
// cut keeps the low-order terms of both parts: the numerator's p_0..p_{numDeg} and the
// denominator's q_1..q_{denDeg}. The denominator's coefficients sit above the *stored* numerator,
// so the position of q_j is the piece's full numerator degree's, not the cut's.
template <class Tag = void>
inline double RationalPieceAtCut(std::size_t index, int numDeg, int denDeg, double t) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, CoarseRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_d_coarse_rat.hpp: include that header before boys/boys.hpp "
                  "in the translation unit that names the shipped rational route");
    const double* c = detail::kRatACoeffs.data() + detail::kRatAOffset[index];
    const int storedNumDeg = detail::kRatANumDeg[index];
    double num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    double den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
}

// The same piece read at a cut pair over the double lane's narrow table; see
// RationalFitNarrow::EvalPiece for the full pair this cuts and
// RationalPieceAtCut for the shipped partition's reading of the same shape.
template <class Tag = void>
inline double RationalPieceNarrowAtCut(std::size_t index,
                                       int numDeg,
                                       int denDeg,
                                       double t) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, NarrowRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_d_narrow_rat.hpp: include that header before boys/boys.hpp "
                  "in the translation unit that names the narrow rational route");
    const double* c = detail::kNarrowRatACoeffs.data() + detail::kNarrowRatAOffset[index];
    const int storedNumDeg = detail::kNarrowRatANumDeg[index];
    double num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    double den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
}

// The region-B seed at a cut pair; same reading as RationalFit::RegionBSeed.
inline double RationalSeedAtCut(int numDeg, int denDeg, double t) noexcept {
    double num = detail::kRatBnum[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, t, detail::kRatBnum[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    double den = detail::kRatBden[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, t, detail::kRatBden[j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, t, 1.0);
}

} // namespace detail

} // namespace boys

/// \endcond
