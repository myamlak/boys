// The test's own override of the build defaults: the header the seam test's second configure points
// BOYS_BUILD_DEFAULTS at, so that a build whose choices are not the committed ones is exercised by the
// suite rather than by reading (include/boys/boys_build_defaults.hpp states the contract a replacement
// satisfies). It is a fixture, not a measurement. Each move is quoted where its value is defined.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE 1

// Committed: the fit route is pinned where the seam test reads this file (tests/boys_backend_test.cpp,
// `static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev, ...)`), so naming
// kRationalMinimax here fails that pin rather than compiling a moved route, and moving the route is a
// change to this file and to that pin in one commit.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev

// Moved: the same fit, summed by the split Clenshaw recurrence instead of
// Horner's rule. Different rounding, so the values move in their last places.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kSplitClenshaw

// Committed, and unmovable: the orders axis is not an axis on a shape producing one order, and the
// library refuses it in its own words at include/boys/boys_impl.hpp (BoysSingleImpl) and again in the
// batch entries. Naming kOrders here fails to compile in the library's own sources - src/boys.cpp
// instantiates the single-order entry at EvalPolicy<> - which is the answer and not a default.
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments

// Moved: one reciprocal per divisor and one product per step, in place of the refined form's quotient.
// Which form is cheapest is the host's property, so all three are carried and ranked by the option
// probe; the two not named here are instantiated on every packed lane anyway - boys/boys_impl.hpp's
// BOYS_ORD_* blocks and src/boys_orders_simd.cpp's BOYS_ORDERS_*_INSTANTIATIONS.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kPlainReciprocal

// Moved: kCoarsest, the partition the certified lanes are defined by, in place of the committed
// header's kNarrow; not the member a build cannot move, which is the packing axis (CONTRIBUTING.md).
// FitGranularity::kUniform is named by the row table in tests/build_defaults_uniform.hpp;
// kCoarsest is what tests/boys_accuracy_gate.cpp reads.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kCoarsest

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above,
// so the three axes moved above are the values this build resolves differently and the device
// half of it compiles what the committed file names.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate

// The moves above are the compile-time half of the seam check, and this file's own configure is the
// only reading there is for them: the `Build defaults` step in `.github/workflows/ci.yml` points
// BOYS_BUILD_DEFAULTS at this file on one leg, so "the library reads BOYS_BUILD_DEFAULT_DIVISION_FORM
// and BOYS_BUILD_DEFAULT_FIT_GRANULARITY" is established by that configure building and passing.
