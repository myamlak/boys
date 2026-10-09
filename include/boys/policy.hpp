#pragma once

/// \file policy.hpp
/// One cell of an evaluation policy replaced, every other cell derived from a base
/// rather than written down.
///
/// \c EvalPolicy is seven positional template parameters and a parameter list has no
/// holes: a caller who wants the last of them replaced writes the six before it, and
/// those six are then the values that caller wrote and not the build's. The aliases
/// below take the cell the caller has an opinion about and a policy to derive the rest
/// from, so the cells a caller does not name are the base's own and not a copy of them:
///
///     // the build's own point, with one cell named
///     using P = boys::WithDivisionForm<boys::DivisionForm::kPlainReciprocal>;
///
///     // a class's own row of the default-policy table, with one cell named
///     using Q = boys::WithDivisionForm<boys::DivisionForm::kPlainReciprocal,
///                                      boys::DefaultPolicy<boys::Precision::kFp64,
///                                                          boys::Shape::kAllOrders>>;
///
/// **Where a caller names no base, the base is \c EvalPolicy<>, which is the build's
/// own point and not a class's row.** \c EvalPolicy<> is the seven the build's seam
/// header (`boys/boys_build_defaults.hpp`) names; what an entry compiles when its call
/// site names no policy is \c DefaultPolicy<Precision, Shape>, the class's own row, and
/// the cells of that row need not be the seam's. A caller who wants that row with one
/// cell replaced names it as the base, as \c Q does above; a caller who names none gets
/// the point \c EvalPolicy<> is. Neither form writes a cell its caller did not choose.
///
/// Each alias is the type its long spelling is, so a call through one is compiled as
/// the call through the other, to the last bit.
///
/// \ingroup boys

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"

namespace boys {

/// The fit route replaced: the base's policy, with \c kRoute taken from the argument
/// and every other cell left as the base has it.
///
/// The route names which fits are evaluated, and it is read together with the scheme
/// and the granularity: a combination the library does not carry is refused where the
/// caller names it, at the fit the three select.
///
/// \tparam kFitRoute the route the derived policy evaluates
/// \tparam Base      the policy to derive from; \c EvalPolicy<>, the build's own
///                   point, where the caller names none
///
/// \ingroup boys
template <FitRoute kFitRoute, EvalPolicyLike Base = EvalPolicy<>>
using WithFitRoute = StatedEvalPolicy<kFitRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                      Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The evaluation scheme replaced: the base's policy, with \c kScheme taken from the
/// argument and every other cell left as the base has it.
///
/// The scheme is how one stored fit is summed, so it changes the arithmetic of every
/// evaluation and not the fit that is evaluated.
///
/// \tparam kEvalScheme the scheme the derived policy sums its fits in
/// \tparam Base        the policy to derive from; \c EvalPolicy<>, the build's own
///                     point, where the caller names none
///
/// \ingroup boys
template <EvalScheme kEvalScheme, EvalPolicyLike Base = EvalPolicy<>>
using WithEvalScheme = StatedEvalPolicy<Base::kRoute, kEvalScheme, Base::kBudget, Base::kPack,
                                        Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The engine budget replaced: the base's policy, with \c kBudget taken from the
/// argument and every other cell left as the base has it.
///
/// The budget is read by the single-precision and half-precision engines and by no
/// other lane, so on a double-precision policy this cell is inert: two policies that
/// differ in it alone compile the same call.
///
/// \tparam kEngineBudget the budget a single-precision engine runs at
/// \tparam Base          the policy to derive from; \c EvalPolicy<>, the build's own
///                       point, where the caller names none
///
/// \ingroup boys
template <BoysBudget kEngineBudget, EvalPolicyLike Base = EvalPolicy<>>
using WithBudget = StatedEvalPolicy<Base::kRoute, Base::kScheme, kEngineBudget, Base::kPack,
                                    Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The packing axis replaced: the base's policy, with \c kPack taken from the argument
/// and every other cell left as the base has it.
///
/// The axis is a property of the shape a call has: an entry that produces one order
/// has no second order to pack, so the axis an entry reads is the one its own packed
/// lane fills.
///
/// \tparam kPackedAxis the axis a packed evaluation vectorises over
/// \tparam Base        the policy to derive from; \c EvalPolicy<>, the build's own
///                     point, where the caller names none
///
/// \ingroup boys
template <PackAxis kPackedAxis, EvalPolicyLike Base = EvalPolicy<>>
using WithPackAxis = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, kPackedAxis,
                                      Base::kGranularity, Base::kDivision, Base::kRegionBExp>;

/// The fit granularity replaced: the base's policy, with \c kGranularity taken from the
/// argument and every other cell left as the base has it.
///
/// The granularity says how narrowly the fitted domain is cut into pieces, and it
/// selects tables: a piece of another width is another fit at another degree, so this
/// cell moves the stored coefficients a call reads and not only its arithmetic.
///
/// \tparam kFitGranularity how narrowly the derived policy cuts the fitted domain
/// \tparam Base            the policy to derive from; \c EvalPolicy<>, the build's own
///                         point, where the caller names none
///
/// \ingroup boys
template <FitGranularity kFitGranularity, EvalPolicyLike Base = EvalPolicy<>>
using WithFitGranularity = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                            kFitGranularity, Base::kDivision, Base::kRegionBExp>;

/// The division form replaced: the base's policy, with \c kDivision taken from the
/// argument and every other cell left as the base has it.
///
/// The form is how the recursions' per-order division is performed. The three are three
/// arithmetics of a chain and not three spellings of one, and the two reciprocal forms
/// differ from each other in the last bits they deliver as well as in what they cost.
///
/// \tparam kDivisionForm how the derived policy performs the recursion's divisions
/// \tparam Base          the policy to derive from; \c EvalPolicy<>, the build's own
///                       point, where the caller names none
///
/// \ingroup boys
template <DivisionForm kDivisionForm, EvalPolicyLike Base = EvalPolicy<>>
using WithDivisionForm = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                          Base::kGranularity, kDivisionForm, Base::kRegionBExp>;

/// The region-B exponential replaced: the base's policy, with \c kRegionBExp taken from
/// the argument and every other cell left as the base has it.
///
/// The cell selects which exponential seeds a region-B ladder, so it changes the values
/// region B returns and nothing an earlier region returns.
///
/// \tparam kExp which exponential the derived policy's region-B ladder is seeded with
/// \tparam Base the policy to derive from; \c EvalPolicy<>, the build's own point, where
///              the caller names none
///
/// \ingroup boys
template <RegionBExp kExp, EvalPolicyLike Base = EvalPolicy<>>
using WithRegionBExp = StatedEvalPolicy<Base::kRoute, Base::kScheme, Base::kBudget, Base::kPack,
                                        Base::kGranularity, Base::kDivision, kExp>;

}  // namespace boys
