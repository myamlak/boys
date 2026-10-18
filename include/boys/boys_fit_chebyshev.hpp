#pragma once

// The Chebyshev route's fit families and the region-seed reads over them.
// They are defined here rather than in boys_impl.hpp so that a translation unit parses
// only the group headers its policies need; the markers in each read are what name the
// group header a caller must add.

/// \cond

#include <type_traits>

#include <cassert>
#include <cmath>
#include <cstddef>

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/policy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/dependent_false.hpp"
#include "boys/boys_fit_core.hpp"

namespace boys {

namespace detail {

// The Chebyshev route, at the scheme its coefficients are summed in: the shipped family, holding
// both stored forms - the Chebyshev table the split Clenshaw reads and the monomial table Horner
// reads - over the same pieces at the same degrees, and both partitions of both regions, which is
// what the granularity axis selects; see FitGranularity for what that axis is and is not.
template <EvalScheme kScheme, FitGranularity kGranularity>
struct ChebyshevFit {
    // This family stores the shipped and the narrow partition and no uniform one: its region-A read
    // is piece-indexed, and the grid's cells are interval-major, read through UniformFit's own
    // locator, so a uniform instantiation would answer every read out of the narrow pieces under
    // the grid's name - the third name is refused here, at the alias that would otherwise resolve it.
    static_assert(kGranularity != FitGranularity::kUniform,
                  "ChebyshevFit carries the shipped and the narrow partition and no uniform "
                  "table: the grid's cells are not pieces of this family, and a uniform "
                  "instantiation of it answers every read out of the narrow pieces under the "
                  "grid's name. The grid's own read is UniformFit's");

    // The region-A partition this fit reads; the bodies ask for it rather than for
    // the table, so a route over the fit does not have to know which. The two
    // values the assertion above leaves are the two this alias names.
    using Partition = std::conditional_t<kGranularity == FitGranularity::kCoarsest,
                                         ShippedRegionAPartition,
                                         NarrowRegionAPartition>;

    // The band's orders are read from the band's seed, so this route's own
    // answer starts at the band's left edge.
    static constexpr double kRegionAFitsFrom = detail::kTierThresholds[0];

    static double EvalPiece(std::size_t index, double t) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            const detail::OrderPiece& piece = detail::kPieces[index];

            if constexpr (kScheme == EvalScheme::kHorner)
            {
                static_assert(kCoefficientGroupPresent<
                                  DependentGroup<GranularityTag<kGranularity>, CoarseMonoGroup>> ||
                                  kAlwaysFalse<GranularityTag<kGranularity>>,
                              "this branch reads boys/boys_coefficients_d_coarse_mono.hpp: "
                              "include that header before boys/boys.hpp in the translation unit that names this "
                              "scheme and granularity");
                return FitSumMono<backend::ScalarFp64>(detail::kMonoCoeffs.data() + piece.offset,
                                                       piece.deg,
                                                       t);
            }
            else
            {
                static_assert(kCoefficientGroupPresent<
                                  DependentGroup<GranularityTag<kGranularity>, CoarseChebGroup>> ||
                                  kAlwaysFalse<GranularityTag<kGranularity>>,
                              "this branch reads boys/boys_coefficients_d_coarse_cheb.hpp: "
                              "include that header before boys/boys.hpp in the translation unit that names this "
                              "scheme and granularity");
                return FitSumCheb<backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                       piece.deg,
                                                       t);
            }
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            const detail::OrderPiece& piece = detail::kNarrowAPieces[index];

            if constexpr (kScheme == EvalScheme::kHorner)
            {
                static_assert(kCoefficientGroupPresent<
                                  DependentGroup<GranularityTag<kGranularity>, NarrowMonoGroup>> ||
                                  kAlwaysFalse<GranularityTag<kGranularity>>,
                              "this branch reads boys/boys_coefficients_d_narrow_mono.hpp: "
                              "include that header before boys/boys.hpp in the translation unit that names this "
                              "scheme and granularity");
                return FitSumMono<backend::ScalarFp64>(
                    detail::kNarrowAMonoCoeffs.data() + piece.offset, piece.deg, t);
            }
            else
            {
                static_assert(kCoefficientGroupPresent<
                                  DependentGroup<GranularityTag<kGranularity>, NarrowChebGroup>> ||
                                  kAlwaysFalse<GranularityTag<kGranularity>>,
                              "this branch reads boys/boys_coefficients_d_narrow_cheb.hpp: "
                              "include that header before boys/boys.hpp in the translation unit that names this "
                              "scheme and granularity");
                return FitSumCheb<backend::ScalarFp64>(
                    detail::kNarrowACoeffs.data() + piece.offset, piece.deg, t);
            }
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // Region A is where the three partitions differ, and the grid has no table of this
            // shape: it is piece-indexed, and the grid's cells are interval-major with a degree of
            // their own. The arm stands rather than falling through to the narrow pieces, so a
            // uniform instantiation reaching it is a build failure, not a value from other rows.
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "ChebyshevFit's region-A read is piece-indexed and the uniform grid "
                          "is not: its cells are interval-major and are summed at the degree "
                          "each was fitted at, through UniformFit's own locator. Read the grid "
                          "there instead of routing it to this family");
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    static double RegionBSeed(double x) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return ShippedRegionBSeed<kScheme>(x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return NarrowRegionBSeed<kScheme>(x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // The same fit as the shipped arm, by construction and not by fallback:
            // region B's seed is one fit over [kX0, kFlatHi) at every granularity except
            // the narrow one, and the grid has no seed of its own to store.
            return ShippedRegionBSeed<kScheme>(x);
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    // The band's orders: one seed at the band's left edge, then one upward step per
    // order. The state is the step's, so a batch pays one division per order. The
    // source is constructed at the argument, so its reciprocal is the
    // construction's too and the step never divides.
    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        double f;
        double expx;
        double invx;

        explicit BandSource(double x) noexcept
            : f(RegionBExtendedSeed<kScheme>(x)),
              expx(0.5 * std::exp(-x)),
              invx(StepReciprocal<kForm>(x)) {}

        double Next(int l, double x) noexcept {
            return l == 0 ? f : (f = UpwardStep<kForm>(l, f, x, invx, expx));
        }
    };
};

// Region-A seed F_order(x) of the Chebyshev route at a scheme, for the lanes
// and the reports that read one fit rather than a whole route.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double ChebyshevValue(int order, double x) noexcept {
    return RegionAValue<ChebyshevFit<kScheme, kDefaultFitGranularity>>(order, x);
}

// Region-B seed F_0(x) of the Chebyshev route at a scheme, valid on [kX0, kFlatHi).
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeed(double x) noexcept {
    return ChebyshevFit<kScheme, kDefaultFitGranularity>::RegionBSeed(x);
}

// Region B's seed at the partition a policy names: the Chebyshev seed at the policy's scheme and
// granularity, not the route's fit, so a policy's region-B seed is the partition it names rather
// than always the shipped one. It resolves that partition through a fit family, refusing where the
// family does not answer it: the barrier itself, and a later body that reads it fails to compile.
template <EvalPolicyLike Policy>
inline double PolicyRegionBSeed(double x) noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this read resolves the policy's partition through the Chebyshev family, which "
                  "carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one: a policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Give this body a "
                  "branch that reads the grid's own table, or route a policy naming the grid to "
                  "one that has it");
    return ChebyshevFit<Policy::kScheme, Policy::kGranularity>::RegionBSeed(x);
}

/// Region A's per-order value at the partition a policy names, for the bodies that
/// read the Chebyshev route's region-A pieces directly rather than through a
/// policy's fit. It is the Chebyshev partition for the reason PolicyRegionBSeed is -
/// the piece tables are the Chebyshev family's - and the policy's granularity. The
/// rational routes are read here at the policy's partition as well: below each
/// route's fits-first crossover the value a policy answers with is the Chebyshev
/// lane's, which is the lane that region is documented at on both partitions.
///
/// This read carries the same assertion PolicyRegionBSeed carries. It is reached
/// from the `x < kX0` block of SingleOrder and of AllOrdersBody, and those blocks
/// are instantiated for a policy naming the grid even though the grid's own branch
/// has returned for every argument they cover - below kFlatHi the grid's body
/// answers, and kFlatHi is above kX1 and so above kX0. The blocks are conditionally
/// dead on this same question of this same family, so the barrier stands here too.
template <EvalPolicyLike Policy>
inline double PolicyRegionAValue(int order, double x) noexcept {
    static_assert(FitAnswersPartition<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(
                      Policy::kGranularity),
                  "this read resolves the policy's partition through the Chebyshev family, "
                  "which carries the shipped and the narrow partition and reads every other "
                  "granularity as the narrow one: a policy naming the uniform grid here is "
                  "answered from the narrow tables under the grid's name. Give this body a "
                  "branch that reads the grid's own table, or route a policy naming the grid "
                  "to one that has it");
    return RegionAValue<ChebyshevFit<Policy::kScheme, Policy::kGranularity>>(order, x);
}

// The region-A seed at the piece's effective degree; the piece index is the
// flat index of the partition's own piece table, which is how that partition's
// degrees table is indexed.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
double ChebyshevValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::OrderPiece& piece = FindPiece(order, x);
        const std::ptrdiff_t index = &piece - detail::kPieces.data();
        const int deg = degrees[static_cast<std::size_t>(index)];
        const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, CoarseMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_d_coarse_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp64>(detail::kMonoCoeffs.data() + piece.offset,
                                                   deg,
                                                   t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, CoarseChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_d_coarse_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp64>(detail::kCoeffs.data() + piece.offset,
                                                   deg,
                                                   t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::OrderPiece& piece = FindNarrowAPiece(order, x);
        const std::ptrdiff_t index = &piece - detail::kNarrowAPieces.data();
        const int deg = degrees[static_cast<std::size_t>(index)];
        const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, NarrowMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_d_narrow_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp64>(
                detail::kNarrowAMonoCoeffs.data() + piece.offset, deg, t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, NarrowChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_d_narrow_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp64>(
                detail::kNarrowACoeffs.data() + piece.offset, deg, t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A is where the three partitions differ, and this read is piece-indexed off a
        // per-order effective-degree table: the grid has neither. Its cells are interval-major,
        // summed at the degree each was fitted at. The arm refuses rather than reading the
        // narrow pieces under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this region-A read takes a piece index and a per-order degree, and the "
                      "uniform grid has neither: its cells are read interval-major at the "
                      "degree each was fitted at. Route the grid to its own read rather than "
                      "here");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The shipped region-B seed at a cut degree: one polynomial over [kX0, kFlatHi), read at the
// degree the caller's table certifies for the peak order. The shipped and the uniform
// partition read this fit between them (see ShippedRegionBSeed), so the two arms of
// RegionBSeedWithDegrees below reach the table through here.
template <EvalScheme kScheme, typename DegreesArray>
inline double ShippedRegionBSeedWithDegrees(double x,
                                            const DegreesArray& degrees,
                                            int peakOrder) noexcept {
    const double t = 2.0 * (x - kX0) / (kFlatHi - kX0) - 1.0;

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(
            detail::kMonoBcoeffs.data(), degrees[static_cast<std::size_t>(peakOrder)], t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(
            detail::kBcoeffs.data(), degrees[static_cast<std::size_t>(peakOrder)], t);
    }
}

// The region-B seed at the stored degree for the peak order the caller is
// about to reach, over the partition named. The shipped seed is one polynomial over the
// whole region, so its table is indexed by that order; the narrow partition's seed is one
// polynomial per piece, and its table carries the order beside the piece.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
inline double RegionBSeedWithDegrees(double x,
                                     const DegreesArray& degrees,
                                     int peakOrder) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedWithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const std::size_t piece = static_cast<std::size_t>(NarrowBPieceOf(x));
        const double a = detail::kNarrowBEdges[piece];
        const double b = detail::kNarrowBEdges[piece + 1];
        const double t = 2.0 * (x - a) / (b - a) - 1.0;
        const std::size_t offset = piece * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);
        const std::size_t degree =
            degrees[piece * (static_cast<std::size_t>(kMaxOrder) + 1)
                    + static_cast<std::size_t>(peakOrder)];

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            return FitSumMono<backend::ScalarFp64>(detail::kNarrowBMonoCoeffs.data() + offset,
                                                   static_cast<int>(degree),
                                                   t);
        }
        else
        {
            return FitSumCheb<backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                                   static_cast<int>(degree),
                                                   t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region B's
        // seed is one fit over [kX0, kFlatHi) at every granularity except the narrow one, so the
        // degree this read is cut at is the shipped table's own. The arm is written rather
        // than left to an `else`, so a reader sees that the two coincide.
        return ShippedRegionBSeedWithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The shipped seed read at a degree handed in directly, which is how the gate's
// granularity book asks a stored row: the partition-naming form above is what the
// bodies call, and this one is the shipped partition's reading of the same fit.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBSeedWithDegrees(double x, int degree) noexcept {
    const double t = 2.0 * (x - kX0) / (kFlatHi - kX0) - 1.0;
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(detail::kMonoBcoeffs.data(), degree, t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(detail::kBcoeffs.data(), degree, t);
    }
}

// The narrow partition's counterparts of the two above: the same mapped argument and the same
// summation, over the pieces that partition holds and at the degree its own table certifies for
// them. A degree table is indexed the way the partition it cuts is read - one entry per narrow
// region-A piece, one per order for the narrow region-B seed - so tables and helpers move together.
template <EvalScheme kScheme = kDefaultEvalScheme, typename DegreesArray>
double NarrowRegionAValueWithDegrees(int order, double x, const DegreesArray& degrees) noexcept {
    const detail::OrderPiece& piece = FindNarrowAPiece(order, x);
    const std::ptrdiff_t index = &piece - detail::kNarrowAPieces.data();
    const int deg = degrees[static_cast<std::size_t>(index)];
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        static_assert(
            kCoefficientGroupPresent<
                DependentGroup<std::integral_constant<EvalScheme, kScheme>, NarrowMonoGroup>> ||
                kAlwaysFalse<std::integral_constant<EvalScheme, kScheme>>,
            "this read needs boys/boys_coefficients_d_narrow_mono.hpp: include that "
            "header before boys/boys.hpp in the translation unit that names this scheme");
        return FitSumMono<backend::ScalarFp64>(detail::kNarrowAMonoCoeffs.data() + piece.offset,
                                               deg,
                                               t);
    }
    else
    {
        static_assert(
            kCoefficientGroupPresent<
                DependentGroup<std::integral_constant<EvalScheme, kScheme>, NarrowChebGroup>> ||
                kAlwaysFalse<std::integral_constant<EvalScheme, kScheme>>,
            "this read needs boys/boys_coefficients_d_narrow_cheb.hpp: include that "
            "header before boys/boys.hpp in the translation unit that names this scheme");
        return FitSumCheb<backend::ScalarFp64>(detail::kNarrowACoeffs.data() + piece.offset,
                                               deg,
                                               t);
    }
}

template <EvalScheme kScheme = kDefaultEvalScheme>
inline double NarrowRegionBSeedWithDegrees(double x, int degree) noexcept {
    const std::size_t index = static_cast<std::size_t>(NarrowBPieceOf(x));
    const double a = detail::kNarrowBEdges[index];
    const double b = detail::kNarrowBEdges[index + 1];
    const double t = 2.0 * (x - a) / (b - a) - 1.0;
    const std::size_t offset = index * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(detail::kNarrowBMonoCoeffs.data() + offset,
                                               degree,
                                               t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                               degree,
                                               t);
    }
}

} // namespace detail

} // namespace boys

/// \endcond
