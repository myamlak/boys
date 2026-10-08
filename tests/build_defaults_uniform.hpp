// A replacement header whose row list moves ten classes' fit granularity to the grid - the third
// member of that axis, which no build in this tree had named on a host row before this file. It is
// pointed at with BOYS_BUILD_DEFAULTS and is a fixture, not a measurement: the member it names is
// one this library carries and certifies, and none of it is a claim about any machine.

//   Point a build at it: -DBOYS_BUILD_DEFAULTS=tests/build_defaults_uniform.hpp

// A build that has measured its own machine writes the host, the date and the option probe's own
// figures beside its choices; include/boys/boys_build_defaults.hpp states the contract a
// replacement satisfies, and this file satisfies it at the committed values plus one cell.

// The granularity axis offers kNarrow, kCoarsest and kUniform; the two fixtures that moved it off
// the committed kNarrow - tests/build_defaults_tuned.hpp beside two other axes and
// tests/build_defaults_fit_granularity.hpp alone - both named kCoarsest, so a host class resolving
// kUniform was one no configure in this tree or in CI had ever compiled.

// A class-table seam the option probe emits, naming the member on a host row, fails
// boys-consumer-umbrella; reaching the member through the seam macro instead does not compile,
// so this fixture names it in a row.

// Through the seam macro, BOYS_BUILD_DEFAULT_FIT_GRANULARITY = FitGranularity::kUniform, that build
// does not compile: tests/boys_accuracy_gate.cpp instantiates the stored region-A fits at
// ChebyshevFit<kDefaultEvalScheme, kDefaultFitGranularity>, and the Chebyshev family refuses the
// grid (include/boys/boys_impl.hpp, the static_assert above ChebyshevFit).

// A row reaches the member where the emitted seam reaches it - the row is what a class's entries
// resolve to, and what the umbrella's granularity lane reads - so the member is exercised by this
// file rather than by the five. The macro path's refusal is a finding of its own, not this file's
// to fix: the arm the gate's route book would need is the grid's own read, which the gate owes.

// One cell of ten rows: the fit granularity of the host double lane's five classes - BoysSingle,
// BoysFixedN, BoysAllOrders, BoysAllN and BoysAllNAtOrders - and of the host single-precision
// lane's same five, from the narrow partition to the grid. Each lane's five move together, not one
// at a time, because this library documents the values of one shape in terms of another.

// BoysFixedN "returns BoysSingle's value at the same (n, x)" and agrees with it bit for bit on a
// build that does not contract a bare product-plus-add (include/boys/boys.hpp, the entry's
// contract), and the float lane's fixed-N entry says "Every element is BoysSingleF32 at the same
// (n, x[i])" and "each element is that entry's value at the same (n, x[i]) bit for bit".

// A configure that moved one of those rows alone would split its lane and fail those identity
// guards for the fixture's shape rather than the axis. Measured on this tree: a draft of this file
// that moved one double row failed BoysFixedNTest.ElementWiseBitIdentityWithBoysSingle,
// StridedLayoutWritesOnlyStrideSlots and AlignmentContractHoldsFromNaturalUp in boys-tests.

// The same three comparisons failed in the umbrella's double-lane check, and a six-row draft that
// moved five double rows and the single-precision lane's single-order row alone failed
// BoysHostClassesTest.TheNewFixedNEntriesAreTheirPerArgumentLoop, whose comparison is that lane's
// single-order and fixed-order rows.

// The float lane's five move because the two failures the emitted seam was reported with are one
// class's each - the double lane's single-order class carries the granularity lane's boundary
// reading (tests/consumer_umbrella.cpp, CheckGranularityLane), the float class the route-selector
// identity (CheckFloatPolicies) - so moving one alone leaves the other unreachable.

// Every other cell of every row below is the committed file's own, transcribed rather than
// re-derived, so the only axis this build moves off the committed configuration is the granularity
// and a failure under this configure belongs to that axis. The rows are written in the committed
// header's own words, their provenance comments included.

// A first draft transcribed the rows of tests/build_defaults_rows.hpp instead, which is the mistake
// this shape invites: that fixture's rows are not the committed ones. Its device rows carry the
// committed five where the committed table carries each device class's measured cells - the split
// Clenshaw scheme, the coarsest partition and the plain reciprocal among them.

// Its host list also omits ten rows the committed table carries: the half lane's fixed-N, all-N and
// all-N-at-orders shapes and the bf16 lane's five. A build against that draft resolved cells no run
// had measured; a fixture that claims to move one cell is worth exactly the diff between its table
// and the committed one, which is why this file's table is the committed one.

// The class is also the one the emitted seam moved: that run's report and its header are the
// reading this fixture was written to reproduce without the rest of the run's rows.

// The replaced-build branch of tests/boys_backend_test.cpp asks two questions of a replacement:
// whether its five host names are the committed five, and whether every class its row table carries
// resolves to the combination the committed build composes from them. It refuses only when both
// answers are yes, so this file passes the guard on the second question alone.

// That guard exists for exactly this shape, the shape the option probe writes
// (`boys-option-probe --emit-defaults`). A class the table carries no row for resolves to the five,
// so this file's five are the committed ones and its rows move a class; writing the move into the
// five instead would move every class at once, a build this repo's own gate does not compile.

// This fixture carries no measured figure of its own and quotes none: the cell it moves is a member
// this library certifies for every build, and no option probe run produced this file.

// The name below is this fixture's identity, and tests/boys_fixture_pins_test.cpp opens a block on
// it and pins the row it moves to this file's value. It is not BOYS_BUILD_DEFAULTS_TEST_ROWS,
// tests/build_defaults_rows.hpp's own name: that block asserts on the class that fixture moves, and
// this file moves a different class on a different axis.
#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE_UNIFORM_ROW 1

// Committed: the committed file's own five. Every class the table below carries no row for
// resolves to these, and the ten rows this file moves are below and not here.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

// The committed file's own row list, transcribed cell for cell, with the one cell this
// fixture moves marked below. One row per class, ten cells each, the region-B exponential
// last; the comments are the committed file's own provenance for its measured rows.
#define BOYS_BUILD_DEFAULT_ROWS(X)\
    /* measured: m = 1 all-orders ladder, 92.89 ns per argument; the granularity cell of
       this row and of the four below is the one this fixture moves, from the narrow
       partition to the grid over the whole fitted domain. */\
    X(kHost, kFp64, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: m = 1 all-orders ladder, the refinement vote's winner; the granularity
       cell is this fixture's, for the reason the float block below states */\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kUniform,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    /* measured: the half lane's vote winner, from the run whose key carried the
       two half formats as one lane; this is the fp16 class's row */\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* the first cell this fixture moves: the granularity, to the grid over the whole
       fitted domain, in place of the narrow partition the committed file names here.
       Every other cell of this row is the committed file's own. */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* the five macros above: no probe run has ranked this shape on any lane; the
       granularity cell is this fixture's, for the reason the single row states */\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* The float lane's five carry the same cell and move together for the same reason, measured:
       the six-row draft of this file, moving this one row of the five, failed
       BoysHostClassesTest.TheNewFixedNEntriesAreTheirPerArgumentLoop, whose comparison is exactly
       the first and third rows below. */\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* no probe run has ranked these four shapes on this lane; every other cell of them is
       the committed file's own */\
    X(kHost, kFp32, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp32, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp32, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* the same on the half lane: three shapes no probe run has ranked */\
    X(kHost, kFp16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp16, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp16, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: m = 1, the bf16 all-orders class's own entry, 116.06 ns per argument, reached by
       the unanimous vote over the refinement runs - the run of 2026-10-04, a
       12-logical-processor AVX2+FMA host, seed 47, whose key carries the two half formats as two
       classes; the fp16 row above is the earlier run's verdict, one lane for both formats. */\
    X(kHost, kBf16, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* no probe run has ranked these four shapes on this lane: the row carries the
       committed five at the half budget, as the fp16 class's unranked shapes do */\
    X(kHost, kBf16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kBf16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kBf16, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kBf16, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: the fp64 single class, 'device-single-fp64-plain-reciprocal', 1.383 ns per argument, reached by vote */\
    X(kDevice, kFp64Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp64 all-orders class, 'device-all-orders-fp64-narrow-rat-horner-plain-reciprocal', 0.199 ns per argument, reached by ordered */\
    X(kDevice, kFp64Device, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp64 all-n class, 'device-all-n-fp64', 0.107 ns per argument, reached by ordered */\
    X(kDevice, kFp64Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp32 single class, 'device-single-fp32-plain-reciprocal', 0.143 ns per argument, reached by vote */\
    X(kDevice, kFp32Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp32 all-orders class, 'all-orders-fp32-uniform-horner-plain-reciprocal', 0.713 ns per argument, reached by chosen-among-equals */\
    X(kDevice, kFp32Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kOrders, FitGranularity::kUniform,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp32 all-n class, 'all-n-fp32-plain-reciprocal', 1.961 ns per argument, reached by refined */\
    X(kDevice, kFp32Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp16 single class, 'device-single-fp16', 0.178 ns per argument, reached by vote */\
    X(kDevice, kFp16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp16 all-orders class, 'device-all-orders-fp16-plain-reciprocal', 0.349 ns per argument, reached by vote */\
    X(kDevice, kFp16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* measured: the fp16 all-n class, 'device-all-n-fp16', 0.266 ns per argument, reached by refined */\
    X(kDevice, kFp16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    /* a choice, not a measurement: the run this file reproduces ranked no bf16 device class, so
       these three rows are the committed seam's own for them - the same five names at the class's
       half budget and the device lane's own two cells - transcribed cell for cell and carrying no
       figure of this file */\
    X(kDevice, kBf16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    X(kDevice, kBf16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    X(kDevice, kBf16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kAccurate)\
    /* A device row's cells are the winning entry's own axes, at the arithmetic its figure was
       measured at: route, scheme, packing axis and granularity come from the lane's own option
       table (boys/boys_cuda_options.hpp), the form cell is that figure's form, and the exponential
       is RegionBExp::kAccurate, the device seam's own (BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP). */

// Committed: the device lane's own two names, which this fixture does not move. A device
// class resolves its form and its region-B exponential to these and never to the host's
// five above, so the cell this table moves is the granularity of ten host rows and no
// cell any device row carries.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate
