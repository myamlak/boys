#pragma once

// The uniform grid's fit family: its locator, its sums and its entries.
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

// The uniform route: one fixed grid over the fitted domain, every order fitted independently, no
// order built from another - no serial dependency chain to shorten. The table is interval-major
// ([interval][order][coefficient]), so one argument's ladder is contiguous, and locating is one
// multiply and a truncation, never a scan of piece edges. Above kFlatHi the call falls through.


template <EvalScheme kScheme>
struct UniformFit {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    // The uniform route reads one fixed grid and nothing else: no piece-indexed fit, no region-B
    // seed. These members exist because FitPolicy asks every fit for the whole contract; they
    // cannot refuse at compile time (the concept check instantiates them), so they answer with a
    // value no route can produce, and a body reaching one fails a comparison rather than returning.
    static double EvalPiece(std::size_t index, double t) noexcept {
        assert(!"the uniform fit is not piece-indexed: it is one fixed grid read "
                        "interval-major. A body reaching here is a body this partition "
                        "does not serve, and it must refuse the partition where it is named");
        return std::nan("") + static_cast<double>(index) + t;
    }

    static double RegionBSeed(double x) noexcept {
        assert(!"the uniform fit stores no region-B seed: region B is one of the walks "
                        "this route replaces with a fixed grid, not a region it reads");
        return std::nan("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double x) noexcept
        {
            assert(!"the uniform fit has no band source: it steps no upward recursion "
                            "from a stored seed");
            (void)x;
        }

        double Next(int l, double x) noexcept {
            assert(!"the uniform fit has no band to step along: every order is read "
                            "from its own coefficients, so there is no next one to build");
            return std::nan("") + static_cast<double>(l) + x;
        }
    };
};

/// The rational member over the uniform grid: one numerator/denominator pair per
/// interval of the same grid, fitted over that interval's own cell and read at the
/// mapped argument the grid's own locator builds for it, in the same stored form
/// RationalFit and RationalFitNarrow read their rows with - the numerator
/// ascending, then the denominator's q_1..q_k with q_0 held at 1.
///
/// A fit of its own rather than a mode of UniformFit, because the two store
/// different things: UniformFit holds a Chebyshev table and its monomial twin, this
/// one holds pairs. Like the shipped and narrow members it is one fit under either
/// scheme: its coefficients are a monomial numerator and denominator with no
/// Chebyshev form to sum. The partition is the grid's own cells, so the member
/// brings one pair per interval to it.
///
/// The degree is the pair's own, so a reader reaches an interval's rows through the
/// interval and RationalUniformOrderAt below answers a call. The struct exists to
/// satisfy FitPolicy's contract, and its three unserved members refuse as
/// UniformFit's do, for the reason stated there.
struct RationalFitUniform {
    using Partition = NarrowRegionAPartition;

    static constexpr double kRegionAFitsFrom = detail::kX1;

    static double EvalPiece(std::size_t index, double t) noexcept {
        assert(!"the uniform grid's rational member is not piece-indexed: it is one "
                        "pair per interval, reached through the interval's own offset. A "
                        "body reaching here is a body this partition does not serve, and it "
                        "must refuse the partition where it is named");
        return std::nan("") + static_cast<double>(index) + t;
    }

    static double RegionBSeed(double x) noexcept {
        assert(!"the uniform grid's rational member stores no region-B seed: region B is "
                        "one of the walks this route replaces with a fixed grid, not a "
                        "region it reads");
        return std::nan("") + x;
    }

    template <DivisionForm kForm = kDefaultDivisionForm>
    struct BandSource {
        explicit BandSource(double x) noexcept
        {
            assert(!"the uniform grid's rational member has no band source: every order "
                            "is read from its own pair, so there is no seed to step from");
            (void)x;
        }

        double Next(int l, double x) noexcept {
            assert(!"the uniform grid's rational member has no band to step along: every "
                            "order is read from its own pair, so there is no next one to build");
            return std::nan("") + static_cast<double>(l) + x;
        }
    };
};

// Every interval's pair is checked against the read rule rather than assumed from it: a row is
// reached at offsets[iv] + l * stored[iv], the numerator's m + 1 coefficients and the denominator's
// k after them, so a stored count that is not m + 1 + k, a pair with no denominator term, or a
// block that is not one row per order reads a neighbouring interval's coefficients as this one's.
constexpr bool FlatRatPairsCarried() noexcept
{
    for (std::size_t iv = 0; iv < static_cast<std::size_t>(detail::kFlatRatIntervals); ++iv)
    {
        const int m = detail::kFlatRatNumDeg[iv];
        const int k = detail::kFlatRatDenDeg[iv];

        if (m < 1 || k < 1 || detail::kFlatRatStored[iv] != m + 1 + k ||
            detail::kFlatRatOffsets[iv + 1] - detail::kFlatRatOffsets[iv] !=
                (kMaxBoysOrder + 1) * detail::kFlatRatStored[iv])
        {
            return false;
        }
    }

    return true;
}

static_assert(FlatRatPairsCarried(),
              "the rational member over the uniform grid must carry, at every interval, a "
              "numerator, a denominator with a non-constant term and one row per order at the "
              "stride its stored count states: the reader reaches a row at "
              "offset + order * stored, so anything else reads another interval's pair");

/// Where an argument sits on the uniform grid: the interval it falls in, that
/// interval's first coefficient, and the argument mapped into it.
///
/// One copy, because the ladder and the single order must agree on it to the bit:
/// two spellings of the same index arithmetic would be two chances to disagree
/// about which interval an argument falls in. The interval is carried and not only
/// its offset, because the table's degree is one for every interval (FlatDegreesUniform):
/// an order's coefficients are reached at the stride kFlatDeg + 1, which is the stride
/// the single-order read walks; the group reader reads the grid's transposed
/// shape, one coefficient index across the orders.
struct FlatPoint {
    std::size_t iv;    ///< the interval the argument falls in
    std::size_t block; ///< iv * kFlatBlock, this interval's order-0 coefficient (kFlatOffsets[iv])
    double t;          ///< the argument mapped into [-1, 1) on that interval
};

inline FlatPoint FlatLocate(double x) noexcept {
    const double u = x * kFlatPerUnit;
    int iv = static_cast<int>(u);

    if (iv > detail::kFlatIntervals - 1)
    {
        iv = detail::kFlatIntervals - 1;
    }

    const std::size_t index = static_cast<std::size_t>(iv);

    // The block is arithmetic where FlatBlocksUniform proves the table's own value. t is the same
    // map through MulAdd, which FlatMapExact proves returns the two-rounding spelling's bits.
    return FlatPoint{index, static_cast<std::size_t>(iv * kFlatBlock),
                     backend::ScalarFp64::MulAdd(2.0, u - static_cast<double>(iv), -1.0)};
}

/// One order off the rational member over the uniform grid: this interval's row at
/// this order, read as RationalFit::EvalPiece reads a piece's row - the numerator by
/// Horner, then the denominator's q_1..q_k with its constant term held at 1, then
/// one division.
template <class Tag = void>
inline double RationalUniformOrderAt(const FlatPoint& at, int l) noexcept {
    static_assert(kCoefficientGroupPresent<DependentGroup<Tag, UniformRatGroup>> ||
                      kAlwaysFalse<Tag>,
                  "this read needs boys/boys_coefficients_d_uniform_rat.hpp: include that "
                  "header in the translation unit that names the uniform rational route");
    const std::size_t stored = static_cast<std::size_t>(detail::kFlatRatStored[at.iv]);
    const double* c = detail::kFlatRatCoeffs.data() +
                      static_cast<std::size_t>(detail::kFlatRatOffsets[at.iv]) +
                      static_cast<std::size_t>(l) * stored;
    const int m = detail::kFlatRatNumDeg[at.iv];
    const int k = detail::kFlatRatDenDeg[at.iv];
    double num = c[m];

    for (int j = m - 1; j >= 0; --j)
    {
        num = backend::ScalarFp64::MulAdd(num, at.t, c[j]);
    }

    double den = c[m + k];

    for (int j = k - 1; j >= 1; --j)
    {
        den = backend::ScalarFp64::MulAdd(den, at.t, c[m + j]);
    }

    return num / backend::ScalarFp64::MulAdd(den, at.t, 1.0);
}

/// One order off the uniform grid, at the route the policy names: the Chebyshev
/// member's cell at that cell's own degree, or the rational member's pair at that
/// interval's own degrees. Both are reached through one `FlatPoint`, so the
/// dispatch is the whole of the difference.
template <typename Policy>
double UniformOrderAt(const FlatPoint& at, int l) noexcept {
    if constexpr (Policy::kRoute == FitRoute::kRationalMinimax)
    {
        return RationalUniformOrderAt(at, l);
    }
    else
    {
        const std::size_t base =
            at.block + static_cast<std::size_t>(l) *
                           static_cast<std::size_t>(kFlatDeg + 1);

        if constexpr (Policy::kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, UniformMonoGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_d_uniform_mono.hpp: include "
                          "that header in the translation unit that names this policy");
            return FitSumMono<backend::ScalarFp64>(detail::kFlatMonoCoeffs.data() + base,
                                                   kFlatDeg,
                                                   at.t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, UniformChebGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_d_uniform_cheb.hpp: include "
                          "that header in the translation unit that names this policy");
            return FitSumCheb<backend::ScalarFp64>(detail::kFlatCoeffs.data() + base,
                                                   kFlatDeg,
                                                   at.t);
        }
    }
}

// The flat grid's one-order fit as the vector tier reads it: the selected scheme's summation
// over the table handed in, so the value is the portable body's at every bit. The caller picks
// the table, so no instantiation names the other form's array; the scalar tier calls it
// nowhere, so where the tier is absent the name stays a declaration.
#if BOYS_ISA_AVX2_FMA
template <EvalScheme kScheme>
double FlatFitSum(const double* c, int deg, double t) noexcept {
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return HornerMono<backend::ScalarFp64>(c, deg, t);
    }
    else
    {
        return ClenshawSplit<backend::ScalarFp64>(c, deg, t);
    }
}
#else
template <EvalScheme kScheme>
double FlatFitSum(const double* c, int deg, double t) noexcept;
#endif

// One argument's ladder off the uniform table: every order read from its own
// coefficients, none from another's.
template <typename Policy>
void UniformAllOrders(int nmax, double x, double* out) noexcept {
    const FlatPoint at = FlatLocate(x);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = UniformOrderAt<Policy>(at, l);
    }
}

// One order at one argument, off the same table and the same index: a call that
// needs one order pays for one order. The ladder form computes every order it is
// asked for, which is the cost the option probe measures against the recursing
// routes - they seed once and step, so their tail orders are nearly free.
template <typename Policy>
double UniformSingleOrder(int n, double x) noexcept {
    return UniformOrderAt<Policy>(FlatLocate(x), n);
}

// One order off the grid, from this unit's own vector tier where it has one and UniformSingleOrder
// above where it does not. The index, coefficients and walk are that body's own, so the value is
// its value at every bit: what changes is where the fused step comes from, the instruction or a
// call into the C runtime. The tier is this unit's flags, never the host's processor.
template <typename Policy>
double FlatSingleOrder(int n, double x) noexcept {
#if BOYS_ISA_AVX2_FMA
    // Branches of one `if constexpr`, not an early return: this build refuses the
    // unreachable-code warning.
    if constexpr (Policy::kRoute == FitRoute::kChebyshev)
    {
        const FlatPoint at = FlatLocate(x);
        const std::size_t base =
            at.block + static_cast<std::size_t>(n) *
                           static_cast<std::size_t>(kFlatDeg + 1);

        if constexpr (Policy::kScheme == EvalScheme::kHorner)
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, UniformMonoGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_d_uniform_mono.hpp: include "
                          "that header in the translation unit that names this policy");
            return FlatFitSum<Policy::kScheme>(detail::kFlatMonoCoeffs.data() + base, kFlatDeg,
                                               at.t);
        }
        else
        {
            static_assert(kCoefficientGroupPresent<DependentGroup<Policy, UniformChebGroup>> ||
                              kAlwaysFalse<Policy>,
                          "this read needs boys/boys_coefficients_d_uniform_cheb.hpp: include "
                          "that header in the translation unit that names this policy");
            return FlatFitSum<Policy::kScheme>(detail::kFlatCoeffs.data() + base, kFlatDeg, at.t);
        }
    }
    else
    {
        return UniformSingleOrder<Policy>(n, x);
    }
#else
    return UniformSingleOrder<Policy>(n, x);
#endif
}

} // namespace detail

} // namespace boys

/// \endcond
