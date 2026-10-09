// One of the build's five choices, moved on its own: the fit route an entry that names no policy
// evaluates. A replacement header for include/boys/boys_build_defaults.hpp, pointed at with the
// BOYS_BUILD_DEFAULTS option, so a configure that fails can say which axis failed:
// tests/build_defaults_tuned.hpp moves three at once - scheme, division form and granularity.

//   Point a build at it: -DBOYS_BUILD_DEFAULTS=tests/build_defaults_fit_route.hpp

// A fixture and not a measurement: the value it moves is one the library carries and certifies for
// every build, and none of it is a claim about any machine. Neither guard a replacement may define
// is carried: not BOYS_BUILD_DEFAULTS_TEST_FIXTURE, whose block in tests/boys_backend_test.cpp pins
// boys::kDefaultFitRoute to boys::FitRoute::kChebyshev; not BOYS_BUILD_DEFAULTS_COMMITTED.

// The name below is this file's identity: tests/boys_fixture_pins_test.cpp opens a block on it and
// pins the five values to this file's.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_FIT_ROUTE 1

// Moved: the rational minimax fits of region A's pieces and of region B's seed, in
// place of the Chebyshev fits every committed bound in this repository was measured at.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kRationalMinimax

// Committed: the committed file's own values, so that the route above is the single value
// this build resolves differently from the committed configuration.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above,
// so the fit route above is the single value this build resolves differently.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
