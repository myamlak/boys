// One of the build's five choices, moved on its own: how narrowly the fitted domain is cut - kNarrow
// to the shipped partition kCoarsest - in a replacement for include/boys/boys_build_defaults.hpp under
// BOYS_BUILD_DEFAULTS, so a failing configure names the axis. kUniform, the third member, is named by
// tests/build_defaults_uniform.hpp's row table instead, and the accuracy gate reads kCoarsest.

// The name below is this fixture's identity: a guard says which block of a test is read, while
// this says which fixture the build carries, and tests/boys_fixture_pins_test.cpp opens a block
// on it and pins the five values below to this file's.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_GRANULARITY 1

// Moved: the partition the certified lanes are defined by, in place of the narrow
// partition the committed file names.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kCoarsest

// Committed: the committed file's own values, so that the fit granularity above is the
// single value this build resolves differently from the committed configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above,
// so the fit granularity above is the single value this build resolves differently.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
