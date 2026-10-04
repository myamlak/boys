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
//
// The five values are the committed file's own, copied by value and not read
// from it: include/boys/boys_build_defaults.hpp carries them and a move of one
// there has to move here too, or this configure stops being a control and starts
// being a build with a choice. It is the control for the seam test's "has chosen
// nothing" assertion and for nothing else: the seam's macro check
// (tests/boys_build_defaults_test.cpp) holds each value below to the constant
// that owns it, which a header naming the shipped values satisfies by
// construction - that check has teeth where a value is moved, which is what
// tests/build_defaults_tuned.hpp is for.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// Shipped: the device lane's own two names, which this control does not move either. They are
// here so that this file fails where it is meant to - the seam test's "has chosen nothing"
// assertion - and not on a name a replacement must carry (boys/boys_build_defaults.hpp). The
// device lane resolves its form and its region-B exponential to these and never to the host's
// above, so a file naming the host's five and not these is a build that has stated half its
// defaults.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
