// One of the build's five choices, moved on its own: the scheme a stored fit is summed
// in when the call site names no policy. A replacement header for
// include/boys/boys_build_defaults.hpp, pointed at with the BOYS_BUILD_DEFAULTS option,
// so that a configure which fails can say WHICH axis failed: the tuned fixture
// (tests/build_defaults_tuned.hpp) moves this one beside the division form and the fit
// granularity, and a report from that build cannot say which of the three a failure
// belongs to.
//
//   cmake -S . -B build-defaults-eval-scheme \
//       -DBOYS_BUILD_DEFAULTS=tests/build_defaults_eval_scheme.hpp
//
// It is a fixture and not a measurement: the value it moves is one this library carries
// and certifies for every build, and none of it is a claim about any machine. A build
// that has measured its own machine writes the host, the date and the option probe's own
// figures beside its choices (include/boys/boys_build_defaults.hpp states the contract a
// replacement satisfies, and this file satisfies it at the shipped values plus one).
//
// WHAT A FAILURE IN THIS CONFIGURE IS A FINDING ABOUT
//
// The evaluation scheme, and nothing else: the other four values below are the committed
// file's own, so this build resolves exactly one choice differently from the shipped
// configuration. The tuned fixture moves this same axis, but beside the division form
// and the fit granularity; here it is the only value that differs from the shipped
// configuration, so a value that moves in this build moved for this reason.
//
// WHAT THE SCHEME IS, AND WHAT MOVING IT CAN AND CANNOT CHANGE
//
// Both schemes sum the same fit: the Chebyshev coefficients and the monomial
// coefficients of one fit describe the same function to within the rounding of the two
// tables, so this is a choice between two arithmetics and not between two accuracies.
// The scheme reaches the double scalar lane's per-order region-A fits, its extended-band
// seed and its region-B seed. Region C has no stored fit to sum and is the same
// arithmetic under either scheme, and the half-precision lanes and the packed region-A
// lane are the split Clenshaw route's and are not offered under the other one
// (boys/accuracy.hpp states both, beside the enumeration). A configure of this file
// therefore reads the scheme on the cells it reaches and leaves the others where they
// were - which is a fact about the axis and not a hole in the fixture.

// The name below is this fixture's identity: a guard says which block of a test is read, while
// this says which fixture the build carries, and tests/boys_fixture_pins_test.cpp opens a block
// on it and pins the five values below to this file's.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_EVAL_SCHEME 1

// Moved: the same fits, summed by the even/odd split Clenshaw recurrence on the
// Chebyshev form instead of by Horner's rule on the monomial form. Different rounding,
// so the values move in their last places rather than by any amount a bound reads.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kSplitClenshaw

// Shipped: the committed file's own values, so that the scheme above is the single value
// this build resolves differently from the shipped configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow
