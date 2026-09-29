// The negative control for the build-defaults seam: a replacement header that
// names the shipped five and has therefore chosen nothing.
//
// It is here so that the seam test's replaced-choices branch can be shown to
// fail, and to fail on its own assertion rather than on something else: a build
// pointed at this file must not compile tests/boys_backend_test.cpp. Without a
// file like it, "the branch passed" would be consistent with a branch that
// cannot fail, which is the way a check of this kind stops meaning anything.
//
//   cmake -S . -B build-defaults-noop -DBOYS_BUILD_DEFAULTS=tests/build_defaults_noop.hpp

#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow
