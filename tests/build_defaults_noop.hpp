// The negative control for the seam: a replacement naming the committed five, so a build pointed at
// it has chosen nothing and must not compile tests/boys_backend_test.cpp. The values are
// include/boys/boys_build_defaults.hpp's own, copied by value, so a move there moves here too.
//   cmake -S . -B build-defaults-noop -DBOYS_BUILD_DEFAULTS=tests/build_defaults_noop.hpp
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// Committed: the device lane's own two names, which this control does not move either. A replacement
// must carry them (boys/boys_build_defaults.hpp): the device lane resolves its form and its region-B
// exponential to these and never to the host's five above.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
