#pragma once

// The shared fit machinery: partition traits, the two summations, the region-B division forms,
// and the pieces and seeds every route shares. They live here rather than in boys_impl.hpp so a
// translation unit parses only the group headers its policies need; each read's marker names
// the group header a caller must add.

/// \cond

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/policy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/dependent_false.hpp"

namespace boys {

namespace detail {

// Every derived family here (Chebyshev, the rational route's derived members, both stored forms)
// reads any granularity but the shipped one as the narrow; the grid's two members carry the grid
// alone. The trap: resolving a policy's partition through a derived family answers a grid-naming
// policy from the narrow pieces, silently - one question below asks it, so a new path fails there.

/// Whether the fit \c Fit carries the uniform grid's own table.
///
/// The derived families do not: their granularity parameter is a two-case
/// conditional over the shipped and the narrow partition, and the grid is a third
/// value they have no answer for. The specializations below are the two members
/// \c RouteFit resolves a policy naming the grid to, one per route.
template <typename Fit>
inline constexpr bool kFitCarriesUniform = false;

template <EvalScheme kScheme>
inline constexpr bool kFitCarriesUniform<UniformFit<kScheme>> = true;

template <>
inline constexpr bool kFitCarriesUniform<RationalFitUniform> = true;

/// Whether the fit \c Fit answers an argument at the partition \c kGranularity from
/// that partition's own table, rather than from another partition's fit under its
/// name.
///
/// \param kGranularity the partition a policy or a body named
///
/// \returns true where \c Fit's stored tables are a cut of that partition
///
/// The switch carries no `default:` arm on purpose: gcc and clang warn for an
/// enumerator it does not name and this tree builds with -Werror, so a new
/// partition fails the build at every site that asks. MSVC emits no -Wswitch, so
/// there the caller's own assertion on the answer carries the fact.
template <typename Fit>
constexpr bool FitAnswersPartition(FitGranularity kGranularity) noexcept
{
    switch (kGranularity)
    {
    case FitGranularity::kCoarsest:
    case FitGranularity::kNarrow:
        // The derived families' two. The grid's members carry neither.
        return !kFitCarriesUniform<Fit>;

    case FitGranularity::kUniform:
        return kFitCarriesUniform<Fit>;
    }

    // A value outside the enumeration: no family answers it, so the answer is no
    // and such a partition fails closed rather than resolving to the narrow tables.
    return false;
}

/// The granularity a body was named with, carried as a type so that the dependent
/// false below can be instantiated on it at the arm that refuses.
template <FitGranularity kGranularity>
struct GranularityTag {};

// Piece containing x for this order; the pieces partition [0, kX0).
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::OrderPiece& FindPiece(int order, double x) noexcept {
    const int first = detail::kPieceStart[order];
    const int last = detail::kPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::kPieces[i].b)
        {
            return detail::kPieces[i];
        }
    }

    return detail::kPieces[last - 1];
}

// The shipped partition is the per-order piecewise table the lane reads; the narrow one is derived
// from the same region at a lower degree per piece and more pieces. The body asks the fit for its
// partition rather than carrying a lookup of its own, so the two differ only in the tables they
// name.
struct ShippedRegionAPartition {
    static std::size_t PieceIndex(int order, double x) noexcept {
        return static_cast<std::size_t>(&FindPiece(order, x) - detail::kPieces.data());
    }

    static const detail::OrderPiece& PieceAt(std::size_t index) noexcept {
        return detail::kPieces[index];
    }
};

// The narrow partition's piece containing x for this order; the same scan as
// FindPiece, over the second partition's rows for the same order.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (order, x) reads naturally.
inline const detail::OrderPiece& FindNarrowAPiece(int order, double x) noexcept {
    const int first = detail::kNarrowAPieceStart[order];
    const int last = detail::kNarrowAPieceStart[order + 1];

    for (int i = first; i < last - 1; ++i)
    {
        if (x < detail::kNarrowAPieces[i].b)
        {
            return detail::kNarrowAPieces[i];
        }
    }

    return detail::kNarrowAPieces[last - 1];
}

struct NarrowRegionAPartition {
    static std::size_t PieceIndex(int order, double x) noexcept {
        return static_cast<std::size_t>(&FindNarrowAPiece(order, x)
                                        - detail::kNarrowAPieces.data());
    }

    static const detail::OrderPiece& PieceAt(std::size_t index) noexcept {
        return detail::kNarrowAPieces[index];
    }
};

// The even/odd split Clenshaw evaluation of a Chebyshev sum, in backend B's
// arithmetic and width. t is the caller's, because the lanes map it differently.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
template <backend::ArithmeticBackend B>
typename B::Packed
ClenshawSplit(const typename B::Value* c, int deg, typename B::Packed t) noexcept {
    using V = typename B::Value;
    using P = typename B::Packed;

    if (deg == 0)
    {
        return B::Broadcast(c[0]);
    }

    if (deg == 1)
    {
        return B::MulAdd(t, B::Broadcast(c[1]), B::Broadcast(c[0]));
    }

    const P v = B::MulAdd(B::Broadcast(V{2}), B::Mul(t, t), B::Broadcast(-V{1}));
    const P twoV = B::Add(v, v);

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4.
        return B::MulAdd(
            t, B::Broadcast(c[1]), B::MulAdd(v, B::Broadcast(c[2]), B::Broadcast(c[0])));
    }

    // The odd part below is the D family's Clenshaw finalization, seeded at the
    // degree's top odd coefficient c[2 * mo - 1], mo = (deg + 1) / 2: at an even
    // degree that is c[2 * m - 1], at an odd one the degree's own top odd
    // coefficient. deg == 2 returns above, where the split has no loop to run.
    assert(deg >= 4);

    const int m = deg / 2;
    const int mo = (deg + 1) / 2;
    P b1 = B::Broadcast(c[std::ptrdiff_t{2} * m]);
    P b2 = B::Broadcast(V{0});

    for (int k = m - 1; k >= 1; --k)
    {
        const P b0 = B::MulAdd(twoV, b1, B::Sub(B::Broadcast(c[std::ptrdiff_t{2} * k]), b2));
        b2 = b1;
        b1 = b0;
    }

    const P even = B::MulAdd(v, b1, B::Sub(B::Broadcast(c[0]), b2));

    // Odd part: c[1], c[3], ..., c[2 * mo - 1] with the D recurrence.
    P o1 = B::Broadcast(c[2 * mo - 1]);
    P o2 = B::Broadcast(V{0});

    for (int k = mo - 2; k >= 1; --k)
    {
        const P o0 = B::MulAdd(twoV, o1, B::Sub(B::Broadcast(c[std::ptrdiff_t{2} * k + 1]), o2));
        o2 = o1;
        o1 = o0;
    }

    const P odd = B::MulAdd(B::Sub(twoV, B::Broadcast(V{1})), o1, B::Sub(B::Broadcast(c[1]), o2));
    return B::MulAdd(t, odd, even);
}

// Horner's rule over the monomial form of a fit, in B's arithmetic and width, at the caller's
// mapped argument t. The monomial coefficients are safe to sum: t never leaves [-1, 1], so they
// stay within a small factor of the Chebyshev ones however high the degree.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (deg, t) reads naturally.
template <backend::ArithmeticBackend B>
typename B::Packed HornerMono(const typename B::Value* c, int deg, typename B::Packed t) noexcept {
    if (deg == 0)
    {
        return B::Broadcast(c[0]);
    }

    typename B::Packed acc = B::Broadcast(c[deg]);

    for (int k = deg - 1; k >= 0; --k)
    {
        acc = B::MulAdd(acc, t, B::Broadcast(c[k]));
    }

    return acc;
}

// One stored fit, summed by the named scheme. The two coefficient tables are
// parallel - same pieces, intervals, degrees and offsets.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): (cheb, mono) reads naturally.
template <EvalScheme kScheme, backend::ArithmeticBackend B>
typename B::Packed FitSum(const typename B::Value* cheb,
                          const typename B::Value* mono,
                          int deg,
                          typename B::Packed t) noexcept {
    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return HornerMono<B>(mono, deg, t);
    } else
    {
        return ClenshawSplit<B>(cheb, deg, t);
    }
}

// The two summations FitSum names, each taking only the table it reads. A call site that knows
// its scheme chooses one under `if constexpr`, so the other scheme's `.data()` is never formed:
// an argument is evaluated whatever the callee does with it. FitSum itself stays for the callers
// that hold both tables.
template <backend::ArithmeticBackend B>
typename B::Packed FitSumMono(const typename B::Value* mono,
                              int deg,
                              typename B::Packed t) noexcept {
    return HornerMono<B>(mono, deg, t);
}

template <backend::ArithmeticBackend B>
typename B::Packed FitSumCheb(const typename B::Value* cheb,
                              int deg,
                              typename B::Packed t) noexcept {
    return ClenshawSplit<B>(cheb, deg, t);
}

// A region-A route is a coefficient set and the scheme it is read in, named as a type (the
// FitPolicy contract, backend.hpp): one body instantiated per (route, scheme) pair, coefficients,
// degrees and piece intervals being compile-time facts of the generated tables and only the piece
// an (order, argument) falls in run-time. The axes meet in RouteFit, where an uncarried pair fails.

// The extended band's per-(n, x) dispatch, in the double lanes only: an order n takes the
// extended seed exactly when x >= kTierThresholds[n], whatever the entry point or the batch's
// other orders. Each threshold is the certified boundary of the smallest kmax row covering it
// (the rows 4/8/16/32), so BoysAllOrders(n,x)[n] == BoysSingle(n,x) == BoysAllOrders(m>=n,x)[n].

// The extended-band seed: F_0(x) on [kExtendedBX0, kX0), summed by the named
// scheme (see ChebyshevValue), dispatched per (n, x) at kTierThresholds in the
// double lanes. The float lanes keep their own dispatch.
template <EvalScheme kScheme = kDefaultEvalScheme>
inline double RegionBExtendedSeed(double x) noexcept {
    const double t = 2.0 * (x - kExtendedBX0) / (kX0 - kExtendedBX0) - 1.0;

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(detail::kMonoExtendedBcoeffs.data(),
                                               detail::kExtendedBDeg,
                                               t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(detail::kExtendedBcoeffs.data(),
                                               detail::kExtendedBDeg,
                                               t);
    }
}

// The three division forms, one entry each, and the selectors the bodies read. Which is cheapest is
// a property of the host (DivisionForm in accuracy.hpp), so the bodies take the form from the policy
// rather than writing one of the three into a body, and each entry is templated on the value type.

// The exact form: the correctly rounded quotient, and what a published bound
// over a region is stated for.
template <typename T>
inline T DivideExact(T a, T x) noexcept {
    return a / x;
}

// The plain form: the quotient through the divisor's reciprocal. It rounds
// twice where the exact form rounds once, so a step may differ by an ulp and a
// ladder of them accumulates the difference.
template <typename T>
inline T DividePlain(T a, T invx) noexcept {
    return a * invx;
}

// The refined form: the plain product, then the classical refinement carrying the product's error
// back through the reciprocal, correctly rounded whenever invx is the correctly rounded 1/x. An
// infinite divisor is where that recovery cannot run - the signed-zero quotient makes the first
// fma a `0 * inf` NaN - so it returns the product, which is that signed zero; the branch is free.
template <typename T>
inline T DivideByReciprocal(T a, T x, T invx) noexcept {
    const T quotient = a * invx;

    if (std::isinf(x))
    {
        return quotient;
    }

    return std::fma(std::fma(-quotient, x, a), invx, quotient);
}

// One divide in the form named, taking whichever of the divisor and its
// reciprocal that form reads.
template <DivisionForm kForm, typename T>
inline T DivideStep(T a, T x, T invx) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(invx);
        return DivideExact(a, x);
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        static_cast<void>(x);
        return DividePlain(a, invx);
    } else
    {
        return DivideByReciprocal(a, x, invx);
    }
}

// The reciprocal a form needs; the exact form forms none, so a form that divides
// does not also pay for a reciprocal it never reads.
template <DivisionForm kForm, typename T>
inline T StepReciprocal(T x) noexcept {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(x);
        return T{0};
    } else
    {
        return T{1} / x;
    }
}

// The downward step's divisor, l + 1/2, as a table of its reciprocals: the constant is exact in
// both value types for every order the recurrences run to, so its reciprocal is a compile-time one.
// Forming it at the step from `l` would be a division per order, which a form whose whole point is
// not to divide cannot pay; the table leaves the plain form one product, the refined form its fma.
template <typename T>
inline constexpr std::array<T, static_cast<std::size_t>(kMaxBoysOrder) + 1>
    kDownwardReciprocals = [] {
        std::array<T, static_cast<std::size_t>(kMaxBoysOrder) + 1> table{};

        for (int l = 0; l <= kMaxBoysOrder; ++l)
        {
            table[static_cast<std::size_t>(l)] = T{1} / (static_cast<T>(l) + T{0.5});
        }

        return table;
    }();

// The downward step's divide, in the form named. The exact form divides by l + 1/2
// as the recurrence is written; the two reciprocal forms read the constant table
// above, and the refined one recovers the correctly rounded quotient because the
// entry is the correctly rounded 1/(l + 1/2).
template <DivisionForm kForm, typename T>
inline T DivideDownwardStep(int l, T a) noexcept {
    const std::size_t index = static_cast<std::size_t>(l);

    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        return DivideExact(a, static_cast<T>(l) + T{0.5});
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        return DividePlain(a, kDownwardReciprocals<T>[index]);
    } else
    {
        return DivideByReciprocal(a, static_cast<T>(l) + T{0.5}, kDownwardReciprocals<T>[index]);
    }
}

// One step of the upward recursion: F_l(x) from F_{l-1}(x). The band's orders,
// the single-order band entry and region B all ride this step, in the division
// form the policy names.
template <DivisionForm kForm = kDefaultDivisionForm>
inline double UpwardStep(int l, double f, double x, double invx, double expx) noexcept {
    return DivideStep<kForm>((l - 0.5) * f - expx, x, invx);
}

// Region-A F_order(x) in a fit's own scheme, over the piece the fit's own
// partition puts the argument in, at that piece's mapped argument.
template <FitPolicy Fit>
inline double RegionAValue(int order, double x) noexcept {
    using Partition = typename Fit::Partition;
    const std::size_t index = Partition::PieceIndex(order, x);
    const detail::OrderPiece& piece = Partition::PieceAt(index);
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;
    return Fit::EvalPiece(index, t);
}

// Which piece of the narrow partition an argument falls in. The edges are the
// derived partition's own, non-decreasing, and the last is kFlatHi, so the scan
// ends inside the table for any argument region B dispatches here.
inline int NarrowBPieceOf(double x) noexcept {
    int piece = 0;

    while (piece + 1 < detail::kNarrowBPieces && x >= detail::kNarrowBEdges[static_cast<std::size_t>(piece + 1)])
    {
        ++piece;
    }

    return piece;
}

// Region B's seed from the narrow partition: the piece the argument falls in, at
// that piece's own mapped argument. The pieces tile the interval the shipped seed
// serves, at fewer coefficients per evaluation and more table rows.
template <EvalScheme kScheme>
inline double NarrowRegionBSeed(double x) noexcept {
    const int piece = NarrowBPieceOf(x);
    const std::size_t index = static_cast<std::size_t>(piece);
    const double a = detail::kNarrowBEdges[index];
    const double b = detail::kNarrowBEdges[index + 1];
    const double t = 2.0 * (x - a) / (b - a) - 1.0;
    const std::size_t offset =
        index * (static_cast<std::size_t>(detail::kNarrowBDeg) + 1);

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(detail::kNarrowBMonoCoeffs.data() + offset,
                                               detail::kNarrowBDeg,
                                               t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(detail::kNarrowBcoeffs.data() + offset,
                                               detail::kNarrowBDeg,
                                               t);
    }
}

// Region B's seed from the shipped partition: one polynomial over the whole of
// [kX0, kFlatHi) at the degree the table carries. This is the seed at every
// granularity except the narrow one - the grid stores no seed of its own, and the
// extended band is the same fit because nothing amplifies its seed.
template <EvalScheme kScheme>
inline double ShippedRegionBSeed(double x) noexcept {
    const double t = 2.0 * (x - kX0) / (kFlatHi - kX0) - 1.0;

    if constexpr (kScheme == EvalScheme::kHorner)
    {
        return FitSumMono<backend::ScalarFp64>(detail::kMonoBcoeffs.data(), detail::kBDeg, t);
    }
    else
    {
        return FitSumCheb<backend::ScalarFp64>(detail::kBcoeffs.data(), detail::kBDeg, t);
    }
}

} // namespace detail

} // namespace boys

/// \endcond
