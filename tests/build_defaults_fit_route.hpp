// One of the build's five choices, moved on its own: the fit route an entry that names
// no policy evaluates. A replacement header for include/boys/boys_build_defaults.hpp,
// pointed at with the BOYS_BUILD_DEFAULTS option, so that a configure which fails can
// say WHICH axis failed: tests/build_defaults_tuned.hpp moves three at once - the
// evaluation scheme, the division form and the fit granularity - and a report from that
// build cannot say which of the three a failure belongs to.
//
//   cmake -S . -B build-defaults-fit-route \
//       -DBOYS_BUILD_DEFAULTS=tests/build_defaults_fit_route.hpp
//
// It is a fixture and not a measurement: the value it moves is one this library carries
// and certifies for every build, and none of it is a claim about any machine. A build
// that has measured its own machine writes the host, the date and the option probe's own
// figures beside its choices (include/boys/boys_build_defaults.hpp states the contract a
// replacement satisfies, and this file satisfies it at the committed values plus one).
//
// WHAT A FAILURE IN THIS CONFIGURE IS A FINDING ABOUT
//
// The fit route, and nothing else: the other four values below are the committed file's
// own, so this build resolves exactly one choice differently from the committed
// configuration. Something the suite reports here is the rational route's, where the
// same report from the tuned fixture could belong to either of the other axes it
// crosses as easily.
//
// WHY THE ROUTE NEEDS A FILE OF ITS OWN
//
// tests/build_defaults_tuned.hpp leaves this axis at the committed value and says a
// fixture naming kRationalMinimax fails
// tests/boys_backend_test.cpp's pin -
//
//   static_assert(boys::kDefaultFitRoute == boys::FitRoute::kChebyshev,
//                 "the fixture's fit route is not in force");
//
// - rather than compiling a moved route. That pin sits inside the block
// `#if defined(BOYS_BUILD_DEFAULTS_TEST_FIXTURE)`, which is the tuned fixture's own name
// for itself, and it pins all five values to THAT fixture's. This file defines neither
// of the two guards a replacement may carry, so the pin is not read in its configure:
//
//  - not BOYS_BUILD_DEFAULTS_TEST_FIXTURE, because the block that reads it holds the
//    five values to the tuned fixture's, so a one-axis replacement carrying that name
//    would fail four of the five assertions instead of compiling the build this file
//    exists to make;
//  - not BOYS_BUILD_DEFAULTS_COMMITTED, which is how the committed header says that it,
//    and not a replacement, is in force. A file carrying it would have the seam test,
//    the backend test and the consumer check read this build as the committed
//    configuration and report the wrong one.
//
// What the pin refuses is therefore a fixture that moves an axis and claims the tuned
// fixture's five values; the route itself is an axis a build can move, and this is the
// file that moves it alone.
//
// The name below is this file's identity and neither of those two: a guard says which block of a
// test is read, while this says which fixture the build carries, and
// tests/boys_fixture_pins_test.cpp opens a block on it and pins the five values to this file's.

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
