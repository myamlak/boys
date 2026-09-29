// The test's own override of the build defaults: the header the seam test's
// second configure points BOYS_BUILD_DEFAULTS at, so that a build whose choices
// are not the shipped ones is exercised by the suite rather than by reading
// (include/boys/boys_build_defaults.hpp states the contract a replacement
// satisfies).
//
// It is a fixture and not a measurement: one choice is moved, to a combination
// this library carries and certifies, and it is not a claim about any machine. A
// build that tunes itself for real carries the machine, the date and the option
// probe's own figures beside its choices.
//
//   cmake -S . -B build-defaults-tuned -DBOYS_BUILD_DEFAULTS=tests/build_defaults_tuned.hpp

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE 1

// Shipped: the route the certified lanes are defined by.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev

// Moved: the same fit, summed by the split Clenshaw recurrence instead of
// Horner's rule. Different rounding, so the values move in their last places.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kSplitClenshaw

// Shipped: the axis BoysAllOrders's own packed lane packs.
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments

// Shipped: the form that is bit-identical to exact division.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal

// Shipped: the partition whose pieces are cut to the proved bound.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow
