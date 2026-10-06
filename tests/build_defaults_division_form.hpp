// One of the build's five choices, moved on its own: the form the chains' divisions are performed in,
// kRefinedReciprocal to kPlainReciprocal, in a replacement for include/boys/boys_build_defaults.hpp
// under BOYS_BUILD_DEFAULTS, so a failing configure names the axis. The replaced-build branch of
// tests/boys_backend_test.cpp passes exactly when this axis took effect; a fixture, not a measurement.

// The name below is this fixture's identity: a guard says which block of a test is read, while
// this says which fixture the build carries, and tests/boys_fixture_pins_test.cpp opens a block
// on it and pins the five values below to this file's.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_DIVISION_FORM 1

// Moved: one reciprocal per divisor and one product per step, in place of the refined
// form's quotient recovered from the product's error.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kPlainReciprocal

// Committed: the committed file's own values, so that the division form above is the
// single value this build resolves differently from the committed configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above,
// which is why the host's division form above is not the value the device half of this build
// carries: this configure moves one axis on one side of the device boundary.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
