// The positive control for the build-defaults seam's replaced-choices guard (tests/boys_backend_test.cpp, the
// static_assert under BOYS_BUILD_DEFAULTS_COMMITTED): a replacement whose five names are the committed five and
// whose row list moves a class, so it has chosen something and must compile and pass its suite.
// tests/build_defaults_noop.hpp is the pair's other half - chosen nothing, at either level - and must not compile.

// boys-option-probe --emit-defaults (src/boys_probe.cpp) carries the build's five as the point a class with no
// row resolves to and its rows as the run's winners, so a guard reading only the five refuses exactly the file
// the probe writes. This is a fixture and not a measurement: nothing here is a claim about any machine and no
// figure is quoted, because no probe run produced one.

// It moves boys::BoysAllOrders - the double lane's all-orders ladder when the call site names no policy - on one
// axis: the scheme, Horner's rule to the split Clenshaw recurrence, as tests/build_defaults_tuned.hpp moves.
// Different rounding, so an unnamed call to that entry and a call at the seam's own five differ in their last
// places, which tests/boys_build_defaults_test.cpp's unnamed-call sweep reads.

// The other rows are the classes the host's and the device's entries reach, each the five above at its own lane's
// budget, written out because the list a replacement carries is read INSTEAD of the one the five compose
// (include/boys/boys.hpp expands one branch or the other): a class the list omits has no default policy. A class is
// a (device, precision, shape) triple, so the device's three lanes are the other half, on their own division form.

// The host rows name RegionBExp::kFast, the host's own default (kDefaultHostRegionBExp); every row is twelve cells,
// its eleventh the basis and its twelfth the record, and a row that leaves one out does not compile
// (include/boys/boys.hpp). Every row of this fixture is a stated default and names no run: the combinations are this
// file's, and the marker beside a row says which one it is and not what measured it.

//   cmake -S . -B build-defaults-rows -DBOYS_BUILD_DEFAULTS=tests/build_defaults_rows.hpp
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
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    /* the five above, at each class's own lane budget */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp32, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kChosen, "")\
    /* the device half: the five above at each device lane's budget, beside the device lane's
       own division form and its own region-B member, one row per class */\
    X(kDevice, kFp64Device, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp64Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp64Device, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp32Device, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp32Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp32Device, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kFp16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kBf16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kBf16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")\
    X(kDevice, kBf16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate, RowBasis::kChosen, "")

// Committed: the device lane's own two names, which this fixture does not move. A device class
// resolves its form and its region-B exponential to these and never to the host's five above.
// The device rows above name both as cells, one row per class, so these two are the fallback
// for a device class this list carries no row for and not the rows themselves.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
