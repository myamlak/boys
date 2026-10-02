#pragma once

/// \file
/// The compile-time facts the rest of the library is written against: the
/// highest order the kernel serves, the multiplier m = 1 that is every lane's
/// default accuracy, and the two axes a call site selects an evaluation by -
/// the route its fits come from and the scheme their coefficients are summed
/// in.
///
/// They stand in their own header so that a header naming them need not carry
/// the kernel: the CUDA lane's option rows are read from a header that has to
/// stay free of the library's C++23 ones, and defining the axes here is what
/// lets one name describe an option on both sides of the device boundary.

#include "boys/boys_build_defaults.hpp"

#include <cstdint>

namespace boys {

/// Highest Boys order supported by the kernel.
inline constexpr int kMaxBoysOrder = 32;

/// The default accuracy multiplier of every lane: m = 1 is full static
/// accuracy, bit-identical to the certified lanes (the documented
/// contract).
/// Larger m values trade certified accuracy for work via compile-time degree
/// truncation (see the contract table in the boys/boys.hpp preamble).
inline constexpr double kBoysFullAccuracyMultiplier = 1.0;

/// Which certified fit serves a region, where the library carries more than
/// one for it.
///
/// Region A and region B each carry two tables of fitted coefficients, placed
/// against the same bar over the same interval, and this names which one a call
/// evaluates: region B's is one lowest-order seed per route that the higher
/// orders are reached from, and region A's is a piecewise fit of every order
/// over each of the pieces its lane's tables cut the region into - two equal
/// bands an order on the double lane, each order's own pieces on the
/// single-precision one. They are alternatives rather than rungs of one design:
/// naming the rational one changes the fits that serve the intervals it covers
/// and nothing else.
///
/// What each route promises, over what interval, and from which argument naming
/// it changes anything, is BoysFitRoutes' report rather than this enumeration's:
/// a route whose fit reaches further than its selector takes over says so there
/// instead of being stretched to claim a domain it does not serve. Region A's
/// rational route is that case.
///
/// A value outside the enumerators - cast in from outside the enum, or named by
/// a newer header - is not a route, and every entry on this surface treats it
/// as \c kChebyshev: the default is the route every build carries.
///
/// \ingroup boys
enum class FitRoute : int {
    /// The Chebyshev fits: the default, and the route the certified lanes are
    /// defined by.
    kChebyshev = 0,
    /// The rational minimax fits of region A's pieces and of region B's seed.
    kRationalMinimax,
};

/// The route the entries evaluate when the caller names none.
///
/// The value is the build's rather than this header's: it is
/// \c BOYS_BUILD_DEFAULT_FIT_ROUTE, from `boys/boys_build_defaults.hpp`, whose
/// committed file carries \c FitRoute::kChebyshev and which a build that has
/// measured its own machine replaces with its own choice.
inline constexpr FitRoute kDefaultFitRoute = BOYS_BUILD_DEFAULT_FIT_ROUTE;

/// Which summation a stored fit is evaluated by.
///
/// A fit is a polynomial whatever scheme sums it: the Chebyshev coefficients
/// and the monomial coefficients of one fit describe the same function, at the
/// same degree, to within the rounding of the two tables. The schemes differ
/// in the arithmetic they round in and in the work they do, not in what they
/// approximate, so both are offered and neither replaces the other.
///
/// The scheme reaches the double scalar lane (per-order region-A fits, the
/// extended-band seed and the region-B seed). Region C is evaluated by its
/// asymptotic form and has no stored fit to sum, so it is the same arithmetic
/// under either scheme; the half-precision lanes and the packed region-A lane
/// are the split Clenshaw route's and are not offered under the other scheme.
///
/// The scheme is one axis of the evaluation and the fit route (FitRoute) is
/// another, and the two are named together by an EvalPolicy. They select
/// different things: the route names the fits that serve the regions, and the
/// scheme names the summation a fit's coefficients are read in. A fit whose
/// coefficients have two stored forms is summed by either scheme; a fit whose
/// coefficients have one - the rational family's monomial numerator and
/// denominator - is read in that form whichever scheme is named, and the scheme
/// still reaches the parts of a call the named route's fits do not serve, which
/// are the shipped family's.
///
/// \ingroup boys
enum class EvalScheme : std::uint8_t {
    /// The even/odd split Clenshaw recurrence on the Chebyshev form: the
    /// summation the certified lanes were first written with, and the one a
    /// caller names to read a fit that way.
    kSplitClenshaw = 0,

    /// Horner's rule on the monomial form of the same fit.
    kHorner = 1,
};

/// The scheme the entries evaluate in when the caller names none.
///
/// The value is the build's rather than this header's: it is
/// \c BOYS_BUILD_DEFAULT_EVAL_SCHEME, from `boys/boys_build_defaults.hpp`, whose
/// committed file carries \c EvalScheme::kHorner and which a build that has
/// measured its own machine replaces with its own choice. The paragraphs below
/// are why that shipped choice is the one it is.
///
/// Horner's rule: a call site that names no scheme is compiled as the entry
/// naming \c kHorner, and naming \c kSplitClenshaw is how a caller asks for the
/// other summation. The two sum one fit, so this value is a choice between two
/// ways of computing the same answer rather than between two accuracies.
///
/// **It is not the cheaper of the two at either partition.** On the host these
/// were last measured on, the two schemes came out within 0.6% of each other
/// where the fit they sum is a piece of the narrow partition, while at the
/// shipped partition the split Clenshaw recurrence was 20% to 22% cheaper than
/// Horner's rule. What that measurement establishes is a tie: three rows of the
/// double lane's full-accuracy class for the all-orders shape — the shipped
/// partition summed by the split Clenshaw recurrence, and the narrow partition
/// summed by either scheme — came out within 0.9% of each other, against the 5.6
/// to 7.4 points one of those rows moves by from one run to the next, with the
/// class's next row 11.5% to 12% behind them. This default is one of the three
/// tied rows. A caller who wants this ranking on their own machine runs the
/// option probe (boys_probe.hpp), which measures it there and names the pairs it
/// could and could not separate.
inline constexpr EvalScheme kDefaultEvalScheme = BOYS_BUILD_DEFAULT_EVAL_SCHEME;

} // namespace boys
