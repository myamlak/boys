// The positive control for the build-defaults seam's replaced-choices guard
// (tests/boys_backend_test.cpp, the static_assert under BOYS_BUILD_DEFAULTS_SHIPPED):
// a replacement header whose five names are the shipped five and whose row list moves a
// class. It has chosen something - the class policy - and it must compile and pass its
// suite.
//
// It is one half of a control pair and is meaningless without the other.
// tests/build_defaults_noop.hpp is the replacement that has chosen nothing, at either
// level, and must NOT compile; this one chooses at the class level rather than at the five.
// That is the shape a real replacement has: the option probe's own file
// (boys-option-probe --emit-defaults) carries the build's five as the point a class with no
// row resolves to and its rows as the run's winners (src/boys_probe.cpp), so a guard
// reading only the five refuses exactly the file the probe writes.
//
//   cmake -S . -B build-defaults-rows -DBOYS_BUILD_DEFAULTS=tests/build_defaults_rows.hpp
//
// It is a fixture and not a measurement: the row it moves is moved to a combination this
// library carries and certifies. Nothing here is a claim about any machine, and no figure
// is quoted, because no option probe run produced it.
//
// WHAT IT MOVES, AND WHY THAT CLASS
//
// One class, the double lane's all-orders ladder - the class boys::BoysAllOrders resolves
// to when its call site names no policy - and one axis of it: the scheme, from Horner's
// rule to the split Clenshaw recurrence, the same move tests/build_defaults_tuned.hpp
// makes. Different rounding, so an unnamed call to that entry and a call at the seam's own
// five differ in their last places, which is what the unnamed-call sweep of
// tests/boys_build_defaults_test.cpp reads to tell a build whose rows answer from one whose
// five do.
//
// The other nine rows are the classes the host's entries reach, each the five above at its
// own lane's budget and written out rather than left implicit: the table a replacement
// carries is read INSTEAD of the one the five compose (boys/boys.hpp expands one branch or
// the other), so a class this list omits has no default policy at all and fails to compile
// where it is asked for. Every row names the region-B exponential as its last cell, which is
// a cell of the format and not an extra: a row that leaves one out is a row that does not
// compile (include/boys/boys.hpp, the row macro), so the ten rows below are ten cells each
// like every other row this format carries. The member is `RegionBExp::kFast`, the host's own
// default (`kDefaultHostRegionBExp`), so each row here resolves to the policy it resolved to
// before the cell existed.
#define BOYS_BUILD_DEFAULTS_TEST_ROWS 1

#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

#define BOYS_BUILD_DEFAULT_ROWS(X)\
    /* the class this fixture moves: the scheme, to the split Clenshaw recurrence */\
    X(kHost, kFp64, kAllOrders, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    /* the five above, at each class's own lane budget */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp32, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)
