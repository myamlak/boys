// The four single-axis build-defaults fixtures, pinned by value: what each resolves, on all five
// axes, in the configure that points BOYS_BUILD_DEFAULTS at it. Each carries an identity macro of
// its own, one block below opens on each of those names, and each of the four resolved five values
// that no assertion stated.

// The four are tests/build_defaults_fit_route.hpp, build_defaults_eval_scheme.hpp,
// build_defaults_division_form.hpp and build_defaults_fit_granularity.hpp; the five-value block in
// tests/boys_backend_test.cpp opens only under tests/build_defaults_tuned.hpp, not under these.

// Every assertion below compares a library constant against an enumerator written out here, not
// against the seam macro the constant is supposed to read: a macro dropped from its reader leaves
// the constant holding the committed literal, so a comparison against the macro would still hold.
// The teeth are on the axis each fixture moves - route, scheme, division form, granularity.

// On the four axes a fixture leaves at the committed values, the assertion is an equality between
// that value and itself: it holds by construction and no defect this file can reach moves it, but
// "the fixture's other four values are the committed ones" is a claim like the moved one.

// The packing axis is the one of the five no block can have teeth on: PackAxis::kOrders is refused
// by the library's own sources (include/boys/boys_impl.hpp, BoysSingleImpl - a call that produces
// one order has no second order to put in a vector lane), so kArguments is the only value the axis
// can hold in a build that compiles, and every pin on it is an equality no fixture can break.

// What this file cannot see, so that a configure it passes is read for what it is: a build whose
// fixture was not delivered - the header named but not copied - carries no identity, reads no block
// here and resolves the committed values. The assertion that catches it is the replaced-choices one
// in tests/boys_backend_test.cpp, which refuses a header naming all five committed values.

// Where it is built: the boys-tests source list, beside tests/boys_backend_test.cpp -
//
//   add_executable(boys-tests ... tests/boys_fixture_pins_test.cpp)

#include "boys/backend.hpp"

#include "boys/boys.hpp"

#include <cstdio>
#include <gtest/gtest.h>

namespace {

// The fixture this build carries, as the line the run prints, so a report from a failed configure
// says which header is in force rather than leaving it to be read off the include path. The arms
// are exhaustive; the last two are the headers a build can read without carrying a fixture, the
// committed one and a consumer's own replacement (CONTRIBUTING.md), which this file cannot pin.
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
#elif defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_UNIFORM_ROW)
    "the uniform-row fixture (tests/build_defaults_uniform.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_TEST_ROWS)
    "the row-list fixture (tests/build_defaults_rows.hpp)";
#elif defined(BOYS_BUILD_DEFAULTS_COMMITTED)
    "none: the committed header, and the committed choices";
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
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kCoarsest,
              "the fit-granularity fixture's fit granularity is not in force");
#endif

// --- tests/build_defaults_uniform.hpp: the granularity moves, on the rows --------------------
#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE_UNIFORM_ROW)
// The five names this fixture writes are the committed ones: what it moves is the fit
// granularity cell of its row list, so these five comparisons hold by construction and are
// stated for the same reason the four blocks above state theirs - a fixture's names are a
// claim like the cells it moves.
static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
              "the uniform-row fixture's fit route is not the committed value");
static_assert(boys::kDefaultEvalScheme == boys::EvalScheme::kHorner,
              "the uniform-row fixture's evaluation scheme is not the committed value");
static_assert(boys::kDefaultPackAxis == boys::PackAxis::kArguments,
              "the uniform-row fixture's packing axis is not the committed value");
static_assert(boys::kDefaultDivisionForm == boys::DivisionForm::kRefinedReciprocal,
              "the uniform-row fixture's division form is not the committed value");
static_assert(boys::kDefaultFitGranularity == boys::FitGranularity::kNarrow,
              "the uniform-row fixture's fit granularity, the five's own, is not the committed value");

// The teeth: the class table is where this fixture moves the axis, and a class the table
// carried no row for resolves the five above - the narrow partition - so a row whose cell was
// dropped, or a table no class resolved through, fails one of these ten assertions. The two
// host lanes' five classes each, which is the whole of what the fixture moves.
static_assert(boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp64 all-orders row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp64 single-order row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kFixedN>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp64 fixed-order row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllN>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp64 all-N row does not name the grid");
static_assert(
    boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>::kGranularity ==
        boys::FitGranularity::kUniform,
    "the uniform-row fixture's fp64 all-N-at-orders row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp32 all-orders row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kSingle>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp32 single-order row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kFixedN>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp32 fixed-order row does not name the grid");
static_assert(boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>::kGranularity ==
                  boys::FitGranularity::kUniform,
              "the uniform-row fixture's fp32 all-N row does not name the grid");
static_assert(
    boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllNAtOrders>::kGranularity ==
        boys::FitGranularity::kUniform,
    "the uniform-row fixture's fp32 all-N-at-orders row does not name the grid");
#endif
