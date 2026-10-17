#pragma once

// The single-precision lane's fit families and their reads.
// They are defined here rather than in boys_impl.hpp so that a translation unit parses
// only the group headers its policies need; the markers in each read are what name the
// group header a caller must add.

/// \cond

#include <cassert>
#include <cmath>
#include <cstddef>

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/policy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_flat_grid.hpp"
#include "boys/dependent_false.hpp"
#include "boys/boys_fit_core.hpp"

namespace boys {

namespace detail {

// Float-lane piece lookup; see FindPiece.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::f32::OrderPiece& FindPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kPieceStart[order];
    const int last = detail::f32::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kPieces[i].b)
        {
            return detail::f32::kPieces[i];
        }
    }

    return detail::f32::kPieces[last - 1];
}

// Narrow-partition piece lookup; see FindPieceF32: the narrow pieces are cut per
// order like the shipped ones, so the walk is the same over the narrow
// piece-start table.
inline const detail::f32::OrderPiece& FindNarrowPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kNarrowAPieceStartF32[order];
    const int last = detail::f32::kNarrowAPieceStartF32[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kNarrowAPiecesF32[i].b)
        {
            return detail::f32::kNarrowAPiecesF32[i];
        }
    }

    return detail::f32::kNarrowAPiecesF32[last - 1];
}

// Which narrow region-B piece an argument falls in; see NarrowBPieceOf, whose
// walk this is over the float lane's own edges.
inline int NarrowBPieceF32(float x) noexcept {
    const float* edges = detail::f32::kNarrowBEdgesF32.data();
    const int pieces = detail::f32::kNarrowBPiecesF32;

    for (int i = 0; i + 1 < pieces; ++i)
    {
        if (x < edges[i + 1])
        {
            return i;
        }
    }

    return pieces - 1;
}

// Which rational narrow region-A piece an argument falls in; see FindPieceF32.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::f32::RatPiece& FindNarrowRatPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kNarrowRatAPieceStartF32[order];
    const int last = detail::f32::kNarrowRatAPieceStartF32[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kNarrowRatAPiecesF32[i].b)
        {
            return detail::f32::kNarrowRatAPiecesF32[i];
        }
    }

    return detail::f32::kNarrowRatAPiecesF32[last - 1];
}

// Which narrow rational region-B piece an argument falls in.
inline const detail::f32::RatPiece& FindNarrowRatBPieceF32(float x) noexcept {
    const detail::f32::RatPiece* pieces = detail::f32::kNarrowRatBPiecesF32.data();
    const int count = detail::f32::kNarrowRatBPiecesCountF32;

    for (int i = 0; i + 2 <= count; ++i)
    {
        if (x < pieces[i].b)
        {
            return pieces[i];
        }
    }

    return pieces[count - 1];
}

// Float-lane region-A seed; see ChebyshevValue. The scheme picks which of the two parallel
// coefficient tables the piece is summed from, the pieces matching under either.

// Forced inline so MSVC's size heuristic does not drop the inlined piece scan the F32-single
// engine's full-accuracy code shape needs; semantics are unaffected. __forceinline is MSVC-only;
// GCC/Clang spell the same intent with always_inline.
#if defined(_MSC_VER)
#define BoysForceInline __forceinline
#else
#define BoysForceInline inline __attribute__((always_inline))
#endif
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
BoysForceInline float ChebyshevValueF32(int order, float x) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32CoarseMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_coarse_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp32>(detail::f32::kMonoCoeffs.data() + piece.offset,
                                                   piece.deg,
                                                   t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32CoarseChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_coarse_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp32>(detail::f32::kCoeffs.data() + piece.offset,
                                                   piece.deg,
                                                   t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::OrderPiece& piece = FindNarrowPieceF32(order, x);
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32NarrowMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_narrow_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp32>(
                detail::f32::kNarrowAMonoCoeffsF32.data() + piece.offset, piece.deg, t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32NarrowChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_narrow_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp32>(
                detail::f32::kNarrowACoeffsF32.data() + piece.offset, piece.deg, t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A of a piece-indexed family has no uniform table: the grid's cells
        // are interval-major and read at their own degrees. The arm refuses rather
        // than reading the narrow pieces under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this is the piece-indexed region-A read, and the uniform grid has no "
                      "piece-indexed table: its cells are summed interval-major, at the degree "
                      "each was fitted at, by the grid's own locator. Route the grid there "
                      "rather than to this family");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

#undef BoysForceInline

// The uniform partition's grid as the emitted table names it: one degree and one offset per
// interval, so the coefficient of order l and term k is
// kFlatCoeffsF32[kFlatOffsetsF32[iv] + l * (kFlatDegsF32[iv] + 1) + k], iv = floor(x * kFlatPerUnitF32).
// The degrees are the interval's own - a single-stride walk would read a neighbouring cell's fit.


/// Where an argument sits on the float uniform grid: the interval it falls in, that interval's first
/// coefficient, and the argument mapped into it.
///
/// One copy, as the double lane's FlatLocate is one copy: the ladder and the single-order read must
/// agree on the interval to the bit, and two spellings of the same index arithmetic would be two
/// chances for them to disagree about which interval an argument is in, with nothing reporting it.
struct FlatPointF32 {
    std::size_t iv;    ///< the interval the argument falls in
    std::size_t block; ///< kFlatOffsetsF32[iv], this interval's order-0 coefficient
    float t;           ///< the argument mapped into [-1, 1) on that interval
};

inline FlatPointF32 FlatLocateF32(float x) noexcept {
    const float u = x * kFlatPerUnitF32;
    int iv = static_cast<int>(u);

    if (iv > detail::f32::kFlatIntervalsF32 - 1)
    {
        iv = detail::f32::kFlatIntervalsF32 - 1;
    }

    const std::size_t index = static_cast<std::size_t>(iv);

    return FlatPointF32{index,
                        static_cast<std::size_t>(detail::f32::kFlatOffsetsF32[index]),
                        2.0f * (u - static_cast<float>(iv)) - 1.0f};
}

/// The rational member over the FLOAT lane's uniform grid: one numerator/denominator
/// pair per interval of the same grid, fitted over that interval's own cell and read
/// at the mapped argument FlatLocateF32 builds for it, in the stored form
/// RationalFit32 and RationalFitNarrow read their rows with.
///
/// The double member's shape at this lane's tables, and a fit of its own rather than
/// a reading of that one: the double lane's pairs are stored in binary64 over the
/// double grid's cells, this lane's in binary32 over its own. Like the shipped and
/// narrow members it is one fit under either scheme: its coefficients are a monomial
/// numerator and denominator with no Chebyshev form to sum.
///
/// Its three unserved members refuse as RationalFitUniform's do, for the reason
/// stated there.
struct RationalFitUniformF32 {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    static float EvalPiece(std::size_t index, float t) noexcept {
        assert(!"the float uniform grid's rational member is not piece-indexed: it is "
                        "one pair per interval, reached through the interval's own "
                        "offset. A body reaching here is a body this partition does not "
                        "serve, and it must refuse the partition where it is named");
        return std::nanf("") + static_cast<float>(index) + t;
    }

    static float RegionBSeed(float x, int /*order*/) noexcept {
        assert(!"the float uniform grid's rational member stores no region-B seed: "
                        "region B is one of the walks this route replaces with a fixed "
                        "grid, not a region it reads");
        return std::nanf("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(float x) noexcept
        {
            assert(!"the float uniform grid's rational member has no band source: every "
                            "order is read from its own pair, so there is no seed to step "
                            "from");
            (void)x;
        }

        float Next(int l, float x) noexcept {
            assert(!"the float uniform grid's rational member has no band to step along: "
                            "every order is read from its own pair, so there is no next "
                            "one to build");
            return std::nanf("") + static_cast<float>(l) + x;
        }
    };
};

// Every interval's pair is checked against the read rule here rather than assumed from it, on the
// double member's reading: a row is reached at offsets[iv] + l * stored[iv], the numerator's m + 1
// coefficients and the denominator's k after them, so a count that is not m + 1 + k, a pair with no
// denominator term, or a block that is not one row per order reads a neighbouring interval's numbers.
constexpr bool FlatRatPairsCarriedF32() noexcept
{
    for (std::size_t iv = 0;
         iv < static_cast<std::size_t>(detail::f32::kFlatRatIntervalsF32);
         ++iv)
    {
        const int m = detail::f32::kFlatRatNumDegF32[iv];
        const int k = detail::f32::kFlatRatDenDegF32[iv];

        if (m < 1 || k < 1 || detail::f32::kFlatRatStoredF32[iv] != m + 1 + k ||
            detail::f32::kFlatRatOffsetsF32[iv + 1] -
                    detail::f32::kFlatRatOffsetsF32[iv] !=
                (kMaxBoysOrder + 1) * detail::f32::kFlatRatStoredF32[iv])
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatRatPairsCarriedF32(),
              "the float rational member over the uniform grid must carry, at every "
              "interval, a numerator, a denominator with a non-constant term and one row "
              "per order at the stride its stored count states: the reader reaches a row "
              "at offset + order * stored, so anything else reads another interval's "
              "pair");

/// One order off the float grid's rational member: this interval's row at this
/// order, read by the same steps RationalFit32::EvalOrder reads a shipped
/// piece's row with - the numerator by Horner, then the denominator's q_1..q_k
/// with its constant term held at 1, then one division - in the lane's own
/// width, which is the arithmetic the pairs were fitted and certified in.
template <class Tag = void>
inline float RationalUniformOrderAtF32(const FlatPointF32& at, int l) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32UniformRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_uniform_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float uniform rational route");
    const std::size_t stored =
        static_cast<std::size_t>(detail::f32::kFlatRatStoredF32[at.iv]);
    const float* c = detail::f32::kFlatRatCoeffsF32.data() +
                     static_cast<std::size_t>(detail::f32::kFlatRatOffsetsF32[at.iv]) +
                     static_cast<std::size_t>(l) * stored;
    const int m = detail::f32::kFlatRatNumDegF32[at.iv];
    const int k = detail::f32::kFlatRatDenDegF32[at.iv];
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, at.t, c[j]);
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, at.t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, at.t, 1.0f);
}

/// One order off the float uniform grid, at the route the policy names: the Chebyshev
/// member's cell, summed at that cell's own degree, or the rational member's pair, read
/// at that interval's own degrees. Both are reached through one `FlatPointF32` - the
/// interval the argument fell in and the mapped argument into it are the locator's - so
/// the dispatch is the whole of the difference.
template <typename Policy>
float UniformOrderAtF32(const FlatPointF32& at, int l) noexcept {
    if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        return RationalUniformOrderAtF32(at, l);
    }
    else
    {
        const int deg = detail::f32::kFlatDegsF32[at.iv];
        const std::size_t base =
            at.block + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        if constexpr (Policy::kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, F32UniformMonoGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_f32_uniform_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this policy");
            return FitSumMono<backend::ScalarFp32>(detail::f32::kFlatMonoCoeffsF32.data() + base,
                                                   deg,
                                                   at.t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, F32UniformChebGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_f32_uniform_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this policy");
            return FitSumCheb<backend::ScalarFp32>(detail::f32::kFlatCoeffsF32.data() + base,
                                                   deg,
                                                   at.t);
        }
    }
}

/// One argument's ladder off the float uniform table: every order read from its
/// own coefficients, none from another's, so a lane of the across-orders packed
/// entry carries the per-order value the arguments axis carries - which is the
/// identity that entry's own suite asserts.
template <typename Policy>
void UniformAllOrdersF32(int nmax, float x, float* out) noexcept {
    const FlatPointF32 at = FlatLocateF32(x);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = UniformOrderAtF32<Policy>(at, l);
    }
}

/// One order at one argument, off the same table and the same index. This is where the route is
/// strongest rather than weakest: a call that needs one order pays for one order, and the ladder form
/// above cannot know that only one is wanted.
template <typename Policy>
float UniformSingleOrderF32(int n, float x) noexcept {
    return UniformOrderAtF32<Policy>(FlatLocateF32(x), n);
}

// This lane's shipped region-B seed: one polynomial over [kX0, kX1), the fit the
// shipped and the uniform partition read between them; see ShippedRegionBSeed for why
// the two coincide.
template <EvalScheme kScheme>
inline float ShippedRegionBSeedF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp32>(detail::f32::kMonoBcoeffs.data(),
                                               detail::f32::kBDeg,
                                               t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp32>(detail::f32::kBcoeffs.data(),
                                               detail::f32::kBDeg,
                                               t);
    }
}

// Float-lane region-B seed; see RegionBSeed.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
inline float RegionBSeedF32(float x) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedF32<kScheme>(x);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const int index = NarrowBPieceF32(x);
        const float a = detail::f32::kNarrowBEdgesF32[static_cast<std::size_t>(index)];
        const float b = detail::f32::kNarrowBEdgesF32[static_cast<std::size_t>(index) + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const std::size_t offset =
            static_cast<std::size_t>(index)
            * static_cast<std::size_t>(detail::f32::kNarrowBDegF32 + 1);

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            return FitSumMono<backend::ScalarFp32>(
                detail::f32::kNarrowBMonoCoeffsF32.data() + offset,
                detail::f32::kNarrowBDegF32,
                t);
        }
        else
        {
            return FitSumCheb<backend::ScalarFp32>(
                detail::f32::kNarrowBcoeffsF32.data() + offset,
                detail::f32::kNarrowBDegF32,
                t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region B's seed is one
        // fit over [kX0, kX1) at every granularity except the narrow one, and the grid has no seed of
        // its own to store.
        return ShippedRegionBSeedF32<kScheme>(x);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The float lane's narrow partition at a cut degree: the same reads, over this
// lane's own narrow pieces and at the degree its own table certifies.

template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
float ChebyshevValueF32WithDegrees(int order, float x, const DegreesArray& degrees) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        const detail::f32::OrderPiece& piece = FindPieceF32(order, x);
        const std::ptrdiff_t index = &piece - detail::f32::kPieces.data();
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32CoarseMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_coarse_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp32>(
                detail::f32::kMonoCoeffs.data() + piece.offset,
                degrees[static_cast<std::size_t>(index)],
                t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32CoarseChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_coarse_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp32>(
                detail::f32::kCoeffs.data() + piece.offset,
                degrees[static_cast<std::size_t>(index)],
                t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const detail::f32::OrderPiece& piece = FindNarrowPieceF32(order, x);
        const std::ptrdiff_t index = &piece - detail::f32::kNarrowAPiecesF32.data();
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        if constexpr (kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32NarrowMonoGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_narrow_mono.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumMono<backend::ScalarFp32>(
                detail::f32::kNarrowAMonoCoeffsF32.data() + piece.offset,
                degrees[static_cast<std::size_t>(index)],
                t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<
                              DependentGroup<GranularityTag<kGranularity>, F32NarrowChebGroup>> ||
                              kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this branch reads boys/boys_coefficients_f32_narrow_cheb.hpp: include "
                          "that header before boys/boys.hpp in the translation unit that names this scheme and "
                          "granularity");
            return FitSumCheb<backend::ScalarFp32>(
                detail::f32::kNarrowACoeffsF32.data() + piece.offset,
                degrees[static_cast<std::size_t>(index)],
                t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A is where the three partitions differ, and this read is piece-indexed off a
        // per-order effective-degree table: the grid has neither. The arm refuses rather than
        // reading the narrow pieces under the grid's name.
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

// The region-B seed at the stored degree for the peak order the caller is about to
// reach, over the partition named: one polynomial over the whole region for the shipped partition,
// one per piece for the narrow one. ShippedRegionBSeedF32WithDegrees below is that seed at a cut
// degree, the counterpart of ShippedRegionBSeedWithDegrees.
template <EvalScheme kScheme, typename DegreesArray>
inline float ShippedRegionBSeedF32WithDegrees(float x,
                                              const DegreesArray& degrees,
                                              int peakOrder) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp32>(
            detail::f32::kMonoBcoeffs.data(), degrees[static_cast<std::size_t>(peakOrder)], t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp32>(
            detail::f32::kBcoeffs.data(), degrees[static_cast<std::size_t>(peakOrder)], t);
    }
}

template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity,
          typename DegreesArray>
inline float RegionBSeedF32WithDegrees(float x,
                                       const DegreesArray& degrees,
                                       int peakOrder) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return ShippedRegionBSeedF32WithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        const std::size_t piece = static_cast<std::size_t>(NarrowBPieceF32(x));
        const float a = detail::f32::kNarrowBEdgesF32[piece];
        const float b = detail::f32::kNarrowBEdgesF32[piece + 1];
        const float t = 2.0f * (x - a) / (b - a) - 1.0f;
        const std::size_t offset =
            piece * (static_cast<std::size_t>(detail::f32::kNarrowBDegF32) + 1);
        if constexpr (kScheme == EvalScheme::kHorner)
        {
            return FitSumMono<backend::ScalarFp32>(
                detail::f32::kNarrowBMonoCoeffsF32.data() + offset,
                degrees[piece * (static_cast<std::size_t>(kMaxOrder) + 1)
                        + static_cast<std::size_t>(peakOrder)],
                t);
        }
        else
        {
            return FitSumCheb<backend::ScalarFp32>(
                detail::f32::kNarrowBcoeffsF32.data() + offset,
                degrees[piece * (static_cast<std::size_t>(kMaxOrder) + 1)
                        + static_cast<std::size_t>(peakOrder)],
                t);
        }
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same fit as the shipped arm, by construction and not by fallback: region B's seed
        // is one fit over [kX0, kX1) at every granularity except the narrow one, so the degree
        // this read is cut at is the shipped table's own. See ShippedRegionBSeed for why the two
        // coincide.
        return ShippedRegionBSeedF32WithDegrees<kScheme>(x, degrees, peakOrder);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// A float-lane single-order call reads a coefficient in exactly two places: the region-A seed
// and the region-B seed, and this names which pair of fits those two are. The routes are
// alternatives, not accuracy levels: the rational one changes the coefficients those two
// intervals are evaluated from and nothing else.

// Rational-route piece lookup; see FindPieceF32.
inline const detail::f32::RatPiece& FindRatPieceF32(int order, float x) noexcept {
    const int first = detail::f32::kRatAPieceStart[order];
    const int last = detail::f32::kRatAPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::f32::kRatAPieces[i].b)
        {
            return detail::f32::kRatAPieces[i];
        }
    }

    return detail::f32::kRatAPieces[last - 1];
}

// Region-A seed of the rational route: the stored numerator over one plus t
// times the stored denominator, read by Horner in the lane's own arithmetic.
// The mapped argument is the expression ChebyshevValueF32 evaluates, so the
// two routes' fits are read at one t.
template <class Tag = void>
inline float RationalValueF32(int order, float x) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32CoarseRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_coarse_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float shipped rational route");
    const detail::f32::RatPiece& piece = FindRatPieceF32(order, x);
    const float* c = detail::f32::kRatACoeffs.data() + piece.offset;
    const int m = piece.numdeg;
    const int k = piece.dendeg;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (k == 0)
    {
        return num;
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// Region-B seed of the rational route; see RegionBSeedF32.
inline float RegionBSeedRationalF32(float x) noexcept {
    const float t = 2.0f * (x - static_cast<float>(kX0)) / static_cast<float>(kX1 - kX0) - 1.0f;
    float num = detail::f32::kRatBnum[detail::f32::kRatBnumDeg];

    for (int j = detail::f32::kRatBnumDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, detail::f32::kRatBnum[j]);
    }

    if constexpr (detail::f32::kRatBdenDeg == 0)
    {
        return num;
    }

    float den = detail::f32::kRatBden[detail::f32::kRatBdenDeg - 1];

    for (int j = detail::f32::kRatBdenDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, detail::f32::kRatBden[j]);
    }

    den = backend::ScalarFp32::MulAdd(den, t, 1.0f);
    return num / den;
}

// One rational piece of this lane at a cut pair, in the mapped argument the full pair is
// read in. The cut keeps the low-order terms of both parts: the numerator's p_0..p_numDeg
// and the denominator's q_1..q_denDeg. The denominator's coefficients sit above the *stored*
// numerator, so the position of q_j is the piece's full numerator degree's, not the cut's.
inline float RationalPieceF32AtCutBody(const float* c,
                                       int storedNumDeg,
                                       int numDeg,
                                       int denDeg,
                                       float t) noexcept {
    float num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = c[storedNumDeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[storedNumDeg + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

template <FitGranularity kGranularity = kDefaultFitGranularity>
inline float RationalPieceF32AtCut(std::size_t index, int numDeg, int denDeg, float t) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        static_assert(kCoefficientGroupPresent<
                          DependentGroup<GranularityTag<kGranularity>, F32CoarseRatGroup>> ||
                          kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this branch reads boys/boys_coefficients_f32_coarse_rat.hpp: include "
                      "that header before boys/boys.hpp in the translation unit that names this scheme and "
                      "granularity");
        const detail::f32::RatPiece& piece = detail::f32::kRatAPieces[index];
        return RationalPieceF32AtCutBody(detail::f32::kRatACoeffs.data() + piece.offset,
                                         piece.numdeg,
                                         numDeg,
                                         denDeg,
                                         t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        static_assert(kCoefficientGroupPresent<
                          DependentGroup<GranularityTag<kGranularity>, F32NarrowRatGroup>> ||
                          kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this branch reads boys/boys_coefficients_f32_narrow_rat.hpp: include "
                      "that header before boys/boys.hpp in the translation unit that names this scheme and "
                      "granularity");
        const detail::f32::RatPiece& piece = detail::f32::kNarrowRatAPiecesF32[index];
        return RationalPieceF32AtCutBody(detail::f32::kNarrowRatACoeffsF32.data() + piece.offset,
                                         piece.numdeg,
                                         numDeg,
                                         denDeg,
                                         t);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // Region A of this family, where the three partitions differ and this one has no row of
        // this shape: the grid's numerator and denominator are one pair per interval, reached by
        // the interval's own offset. The arm refuses rather than summing the narrow pieces' rows
        // under the grid's name.
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "the rational family's piece-indexed region-A read has no uniform table: "
                      "the grid's numerator and denominator are one pair per interval, reached "
                      "by the interval's own offset. Read the grid through its own order entry "
                      "rather than here");
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// The region-B seed of this lane at a cut pair; same reading as RegionBSeedRationalF32. The stored
// coefficients are q_1..q_k with the constant term held at 1, and each descent below reads the next
// one down. The shipped region holds one pair over the whole interval, the narrow one a pair per
// piece; this lane's shipped pair at a cut is the fit the shipped and uniform share (ShippedRegionBSeed).
inline float RationalSeedF32ShippedAtCut(int numDeg, int denDeg, float t) noexcept {
    float num = detail::f32::kRatBnum[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, detail::f32::kRatBnum[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = detail::f32::kRatBden[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, detail::f32::kRatBden[j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

template <FitGranularity kGranularity = kDefaultFitGranularity>
inline float RationalSeedF32AtCut(std::size_t index, int numDeg, int denDeg, float t) noexcept {
    if constexpr (kGranularity == FitGranularity::kCoarsest)
    {
        return RationalSeedF32ShippedAtCut(numDeg, denDeg, t);
    }
    else if constexpr (kGranularity == FitGranularity::kNarrow)
    {
        static_assert(kCoefficientGroupPresent<
                          DependentGroup<GranularityTag<kGranularity>, F32NarrowRatGroup>> ||
                          kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this branch reads boys/boys_coefficients_f32_narrow_rat.hpp: include "
                      "that header before boys/boys.hpp in the translation unit that names this scheme and "
                      "granularity");
        const detail::f32::RatPiece& piece = detail::f32::kNarrowRatBPiecesF32[index];
        const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
        float num = c[numDeg];

        for (int j = numDeg - 1; j >= 0; --j)
        {
            num = backend::ScalarFp32::MulAdd(num, t, c[j]);
        }

        if (denDeg == 0)
        {
            return num;
        }

        float den = c[piece.numdeg + denDeg];

        for (int j = denDeg - 1; j >= 1; --j)
        {
            den = backend::ScalarFp32::MulAdd(den, t, c[piece.numdeg + j]);
        }

        return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
    }
    else if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The same pair as the shipped arm, by construction and not by fallback: region B's seed
        // is one fit over [kX0, kX1) at every granularity except the narrow one. The arm is
        // written rather than left to an `else`, so the sharing is visible here.
        return RationalSeedF32ShippedAtCut(numDeg, denDeg, t);
    }
    else
    {
        static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                      "this switch enumerates the three fit partitions: a fourth value added "
                      "to FitGranularity must be given its own arm here rather than inheriting "
                      "the last one's table");
    }
}

// One narrow rational piece of this lane at a degree pair; the pair indexes the piece, whose
// interval and stored degrees are the narrow partition's own. The denominator's
// coefficients sit above the *stored* numerator here as they do in the shipped table.
template <class Tag = void>
inline float RationalPieceNarrowF32AtCut(std::size_t index,
                                         int numDeg,
                                         int denDeg,
                                         float t) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32NarrowRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_narrow_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float narrow rational route");
    const detail::f32::RatPiece& piece = detail::f32::kNarrowRatAPiecesF32[index];
    const float* c = detail::f32::kNarrowRatACoeffsF32.data() + piece.offset;
    float num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = c[piece.numdeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[piece.numdeg + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// The narrow region-B seed of this lane at a cut pair, over the narrow piece
// the argument falls in; same reading as RegionBSeedRationalNarrowF32.
template <class Tag = void>
inline float RationalSeedNarrowF32AtCut(std::size_t index,
                                        int numDeg,
                                        int denDeg,
                                        float t) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32NarrowRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_narrow_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float narrow rational route");
    const detail::f32::RatPiece& piece = detail::f32::kNarrowRatBPiecesF32[index];
    const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
    float num = c[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (denDeg == 0)
    {
        return num;
    }

    float den = c[piece.numdeg + denDeg];

    for (int j = denDeg - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[piece.numdeg + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// Region-A seed of the rational route over the narrow partition: the same numerator over one
// plus t times the stored denominator, at the narrow pieces' own intervals and mapped
// arguments, so the two routes over a piece read one t. The degree pair is per piece here
// rather than per order.
template <class Tag = void>
inline float RationalValueNarrowF32(int order, float x) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32NarrowRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_narrow_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float narrow rational route");
    const detail::f32::RatPiece& piece = FindNarrowRatPieceF32(order, x);
    const float* c = detail::f32::kNarrowRatACoeffsF32.data() + piece.offset;
    const int m = piece.numdeg;
    const int k = piece.dendeg;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (k == 0)
    {
        return num;
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// Region-B seed of the rational route over the narrow partition; see
// RegionBSeedRationalF32 for the shipped seed this is the narrow counterpart of.
template <class Tag = void>
inline float RegionBSeedRationalNarrowF32(float x) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, F32NarrowRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_f32_narrow_rat.hpp: include that "
                  "header before boys/boys.hpp in the translation unit that names the float narrow rational route");
    const detail::f32::RatPiece& piece = FindNarrowRatBPieceF32(x);
    const float* c = detail::f32::kNarrowRatBCoeffsF32.data() + piece.offset;
    const int m = piece.numdeg;
    const int k = piece.dendeg;
    const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
    float num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp32::MulAdd(num, t, c[j]);
    }

    if (k == 0)
    {
        return num;
    }

    float den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp32::MulAdd(den, t, c[m + j]);
    }

    return num / backend::ScalarFp32::MulAdd(den, t, 1.0f);
}

// The two routes as one float-lane call reads them. Each policy forwards to the helper above,
// so a policy's arithmetic is the helper's operation for operation. The granularity names which
// partition of region A and which region-B seed the route reads.
template <EvalScheme kScheme = kDefaultEvalScheme,
          FitGranularity kGranularity = kDefaultFitGranularity>
struct ChebyshevFit32 {
    /// The partition this fit was named with. A body that reads a fit and has to know
    /// whether the fit answers that partition asks the shared question here rather than
    /// re-deriving it from the type.
    static constexpr FitGranularity kPartition = kGranularity;

    static float EvalOrder(int n, float x) noexcept {
        return ChebyshevValueF32<kScheme, kGranularity>(n, x);
    }

    // This family's seed covers the whole region at one degree, so the order is
    // not read.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        return RegionBSeedF32<kScheme, kGranularity>(x);
    }
};

// The rational family takes no scheme: its numerator and denominator are stored in monomial
// form and read by Horner, so there is no second table for a scheme to choose between, and a
// policy composes with either scheme and evaluates the same values under both.
template <FitGranularity kGranularity = kDefaultFitGranularity>
struct RationalFit32 {
    // Where a batch reading of this route hands an order over to the route's
    // own fit rather than to the band's seed: the lane's own constant, the one
    // the double lane's rational route hands over at.
    static constexpr double kRegionAFitsFrom = detail::kRatARouteLo;

    /// The partition this fit was named with; see ChebyshevFit32::kPartition.
    static constexpr FitGranularity kPartition = kGranularity;

    static float EvalOrder(int n, float x) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return RationalValueF32(n, x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return RationalValueNarrowF32(n, x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // Region A of this family, and the grid has no order of this shape: its pairs are one
            // per interval of a fixed grid, reached by the interval's own offset. The arm refuses
            // rather than reading the narrow pairs under the grid's name.
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this is the rational family's piece-indexed region-A read, and the "
                          "uniform grid has no piece-indexed pair: its numerator and "
                          "denominator are one per interval. Read the grid through its own "
                          "order entry rather than here");
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }

    // One pair for the whole region, so the order is not read.
    static float RegionBSeed(float x, int /*order*/) noexcept {
        if constexpr (kGranularity == FitGranularity::kCoarsest)
        {
            return RegionBSeedRationalF32(x);
        }
        else if constexpr (kGranularity == FitGranularity::kNarrow)
        {
            return RegionBSeedRationalNarrowF32(x);
        }
        else if constexpr (kGranularity == FitGranularity::kUniform)
        {
            // The same pair as the shipped arm, by construction and not by fallback:
            // region B's seed is one fit over [kX0, kX1) at every granularity except the
            // narrow one, and this route's shipped region B is one pair over the interval.
            return RegionBSeedRationalF32(x);
        }
        else
        {
            static_assert(kAlwaysFalse<GranularityTag<kGranularity>>,
                          "this switch enumerates the three fit partitions: a fourth value "
                          "added to FitGranularity must be given its own arm here rather than "
                          "inheriting the last one's table");
        }
    }
};

} // namespace detail

} // namespace boys

/// \endcond
