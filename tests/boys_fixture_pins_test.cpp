// The four single-axis build-defaults fixtures, pinned by value: what each one resolves, on all
// five axes, in the configure that points BOYS_BUILD_DEFAULTS at it.
//
// The gap this file closes is named beside the pin block of tests/boys_backend_test.cpp, whose
// five-value block opens only under BOYS_BUILD_DEFAULTS_TEST_FIXTURE - the name
// tests/build_defaults_tuned.hpp gives itself - and not under the four fixtures that move one axis
// each: tests/build_defaults_fit_route.hpp, build_defaults_eval_scheme.hpp,
// build_defaults_division_form.hpp and build_defaults_fit_granularity.hpp. Each of the four was
// pinned by nothing, so a configure carrying one of them resolved five values that no assertion
// stated - the shape that let three of the seam's five macros be read by nothing, because a macro
// dropped from its reader moves a value and a value nothing states is a move nobody reads. Each of
// the four carries an identity macro of its own, one block below opens on each of those names, and
// the block states what that fixture resolves.
//
// WHY THE PINNED VALUES ARE LITERALS, AND WHERE THE TEETH ARE
//
// Every assertion below compares a library constant against an enumerator written out here, and
// not against the seam macro the constant is supposed to read: a macro dropped from its reader
// leaves the constant holding the shipped literal, so a comparison against the macro would still
// hold while this one falls over. The teeth are therefore on the axis the fixture moves - the fit
// route in the first block, the scheme in the second, the division form in the third and the fit
// granularity in the fourth - and a configure whose moved axis resolves anything but the value
// stated here does not compile.
//
// WHAT THE OTHER FOUR ASSERTIONS IN A BLOCK ARE, AND WHAT THEY ARE NOT
//
// On the four axes a fixture leaves at the committed values, the assertion is an equality between
// that value and itself: it holds by construction, and no defect this file can reach moves it. It
// is stated because "the fixture's other four values are the committed ones" is a claim like the
// moved one, and it was the half nothing said before this file existed.
//
// The packing axis is the one of the five no block can have teeth on, for a reason of the
// library's rather than a fixture's: PackAxis::kOrders is refused by the library's own sources
// (include/boys/boys_impl.hpp, BoysSingleImpl - a call that produces one order has no second order
// to put in a vector lane), so kArguments is the only value the axis can hold in a build that
// compiles and every pin on it is an equality no fixture can break.
//
// WHAT THIS FILE CANNOT SEE, so that a configure it passes is read for what it is: a build whose
// fixture was not delivered - the header named but not copied - carries no identity, reads no
// block here and resolves the committed values, and this file cannot see that configure's intent.
// The assertion that catches it is the replaced-choices one in tests/boys_backend_test.cpp, which
// refuses a header naming all five shipped values.
//
// WHERE IT IS BUILT
//
// It belongs in the boys-tests source list beside tests/boys_backend_test.cpp:
//
//   add_executable(boys-tests ... tests/boys_fixture_pins_test.cpp)

#include "boys/backend.hpp"

#include <cstdio>
#include <gtest/gtest.h>

namespace {

// The fixture this build carries, as the line the run prints: a report from a configure that
// failed says which header is in force rather than leaving it to be read off the include path.
// The arms below are exhaustive for this tree, and the last two are the headers a build can be
// reading without carrying a fixture: the committed one, and a consumer's own replacement
// (CONTRIBUTING.md), for which this file has no values to pin.
constexpr const char* kFixtureInForce =
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)
    "the tuned fixture (tests/build_defaults_tuned.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_ROUTE)
    "the fit-route fixture (tests/build_defaults_fit_route.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_EVAL_SCHEME)
    "the eval-scheme fixture (tests/build_defaults_eval_scheme.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_DIVISION_FORM)
    "the division-form fixture (tests/build_defaults_division_form.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_GRANULARITY)
    "the fit-granularity fixture (tests/build_defaults_fit_granularity.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_SHIPPED)
    "none: the committed header, and the shipped choices";
#else
    "none: a replacement header that is not one of the five fixtures";
#endif

} // namespace

TEST(FixturePinsTest, TheFixtureThisBuildCarriesIsNamed) {
    std::printf("boys: the build-defaults fixture in force is %s\n", kFixtureInForce);
}

// The four blocks below are exclusive by construction, since a configure copies one header and a
// header is one fixture. A header naming two of the four identities is not silent either: each
// fixture moves a different axis to a different value, so no two blocks can both be satisfied, and
// the pins of at least one of them fail.

// --- tests/build_defaults_fit_route.hpp: the fit route moves --------------------------------
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_ROUTE)
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kRationalMinimax,
              "the fit-route fixture's fit route is not in force");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kHorner,
              "the fit-route fixture's evaluation scheme is not the committed value");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the fit-route fixture's packing axis is not the committed value");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kRefinedReciprocal,
              "the fit-route fixture's division form is not the committed value");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kNarrow,
              "the fit-route fixture's fit granularity is not the committed value");
#endif

// --- tests/build_defaults_eval_scheme.hpp: the evaluation scheme moves -----------------------
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_EVAL_SCHEME)
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the eval-scheme fixture's fit route is not the committed value");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kSplitClenshaw,
              "the eval-scheme fixture's evaluation scheme is not in force");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the eval-scheme fixture's packing axis is not the committed value");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kRefinedReciprocal,
              "the eval-scheme fixture's division form is not the committed value");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kNarrow,
              "the eval-scheme fixture's fit granularity is not the committed value");
#endif

// --- tests/build_defaults_division_form.hpp: the division form moves -------------------------
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_DIVISION_FORM)
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the division-form fixture's fit route is not the committed value");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kHorner,
              "the division-form fixture's evaluation scheme is not the committed value");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the division-form fixture's packing axis is not the committed value");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kPlainReciprocal,
              "the division-form fixture's division form is not in force");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kNarrow,
              "the division-form fixture's fit granularity is not the committed value");
#endif

// --- tests/build_defaults_fit_granularity.hpp: the fit granularity moves ---------------------
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_GRANULARITY)
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the fit-granularity fixture's fit route is not the committed value");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kHorner,
              "the fit-granularity fixture's evaluation scheme is not the committed value");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the fit-granularity fixture's packing axis is not the committed value");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kRefinedReciprocal,
              "the fit-granularity fixture's division form is not the committed value");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kShipped,
              "the fit-granularity fixture's fit granularity is not in force");
#endif
