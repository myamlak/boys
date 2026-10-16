// One of the build's five choices, moved on its own: the evaluation scheme, the arithmetic a stored
// fit is summed in when the call site names no policy. A replacement header for
// include/boys/boys_build_defaults.hpp, selected with BOYS_BUILD_DEFAULTS, so a failing configure
// names this axis where tests/build_defaults_tuned.hpp cannot. A fixture, not a measurement.

// The name below is this fixture's identity: a guard says which block of a test is read, while
// this says which fixture the build carries, and tests/boys_fixture_pins_test.cpp opens a block
// on it and pins the five values below to this file's.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_EVAL_SCHEME 1

// Moved: the same fits, summed by the even/odd split Clenshaw recurrence on the
// Chebyshev form instead of by Horner's rule on the monomial form. Different rounding,
// so the values move in their last places rather than by any amount a bound reads.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kSplitClenshaw

// Committed: the committed file's own values, so that the scheme above is the single value
// this build resolves differently from the committed configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above,
// so the evaluation scheme above is the single value this build resolves differently.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
