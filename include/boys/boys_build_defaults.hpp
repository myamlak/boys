#pragma once

/// \file
/// The choices a call site that names no policy resolves to, as the build that compiled it fixed
/// them. A class resolves to **seven** settings: the host lane's five - the fit route, the
/// evaluation scheme, the packing axis, the division form and the fit granularity - and, on a
/// device class, the device lane's two - the division form its entries take and the region-B
/// exponential its tables read.
///
/// They are macros rather than constants of their own because the enumerations
/// they choose between are declared in two headers, `boys/accuracy.hpp` and
/// `boys/backend.hpp`, and no one header can see both. Each owning header expands
/// its own name where its own enumeration is visible, so what a replacement hands
/// in is names, and the compiler checks each one against the enumerators of the
/// axis it initialises.
///
/// The committed file carries the committed choices: the ones a build that
/// replaces nothing compiles. A build that has measured its own machine - the
/// option probe (`boys/boys_probe.hpp`) is what measures one - may point the
/// `BOYS_BUILD_DEFAULTS` CMake option (CONTRIBUTING.md) at a header carrying its
/// own seven. Nothing is paid for this at run time: a choice is a template
/// argument either way, so an unnamed call compiles to the call it always
/// compiled to, at whatever values the build named.
///
/// No single combination is cheapest for every question, which is why the table
/// below carries one row per (device, precision, shape) class rather than one
/// winner for the library. The seven names are the fallback: they are what a
/// class the table carries no row for resolves to. A build that replaces this
/// file states the seven for its unnamed classes and the rows for the classes it
/// names.
///
/// A replacement carries the seven names below, `BOYS_BUILD_DEFAULT_ROWS` or not -
/// a replacement with no list is one whose every class takes the seven. It
/// does not define `BOYS_BUILD_DEFAULTS_COMMITTED`; it defines
/// `BOYS_BUILD_DEFAULTS_REPLACED` instead (CMakeLists.txt does, with the option),
/// and this file refuses to compile under that define, because a translation unit
/// that read both would be resolving different defaults from the library's own
/// instantiations. A cross-compiling configure cannot measure the machine it is
/// compiling for, so it keeps this file and prints which of the two is in force.

/// The fit route an unnamed call evaluates.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev

/// The scheme those fits are summed in.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner

/// The axis an unnamed call's packed lanes vectorise over.
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments

/// The form an unnamed call's divisions are performed in.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal

/// How narrowly an unnamed call's fitted domain is cut.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow

/// The form an unnamed DEVICE call's divisions are performed in.
///
/// The device lane's own name and not the host's, because every device entry
/// takes the form as an argument and runs every form. The value is the form the
/// figures this repository publishes for that lane were measured at
/// (`boys/boys_device_tables.hpp`).
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal

/// The region-B exponential an unnamed DEVICE call evaluates.
///
/// The device lane's own name, and its value differs from the host's: this lane's
/// published figures were measured at the library routine
/// (`RegionBExp::kAccurate`) where the host's default is its own reduced-argument
/// polynomial (`RegionBExp::kFast`).
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate

/// The classes this file sets a default for: **one row per class**, in the
/// table's own format
///
///     X(device, precision, shape, route, scheme, budget, packing axis,
///       granularity, division form, region-B exponential, basis, record)
///
/// A class is a (device, precision, shape) triple - the device a call runs on,
/// the precision lane an entry is built at, and the question that entry answers -
/// and a row is the combination that class's entries compile when the call site
/// names no policy. The names above are the **fallback** and not the whole
/// answer: they are the point `boys::SeamPolicy<>` spells. A class this list
/// carries no row for resolves to those seven - the
/// host's five at its own lane's budget, and, on a device class, the device
/// lane's two as well.
///
/// **The precision cell is a LANE, and each half format is a lane of its own.**
/// `kFp16` and `kBf16` are one engine at one budget that store different digits -
/// 2^-11 for a binary16 return, 2^-8 for a bfloat16 one - so they are two classes
/// and this list carries a row for each: one row for both would be a default the
/// other format reads.
///
/// **The rung is not a key, because the rung is the caller's and not the
/// library's.** A caller chooses the accuracy before the call and this table
/// answers what the library then picks, so each class's row is the one its own
/// m = 1 class measured.
///
/// **EVERY CLASS THE LIBRARY CARRIES HAS A ROW**, and a class without one is a
/// build error rather than a call answered by something else: `boys::DefaultPolicyFor`
/// refuses the class it is asked for, and `tests/boys_build_defaults_test.cpp` asks
/// for each of the twelve device classes by name.
/// `boys-option-probe --emit-defaults` and `boys-device-probe --emit-defaults` are
/// what write them all.
///
/// **The host rows are measurements taken on one machine.** Their route, scheme, packing axis,
/// partition and exponential cells are the option probe's winners for this host - 12 logical
/// processors, the AVX2+FMA tier present, seed 47, 5 passes of 5 rounds over 16384 arguments to
/// order 32 - run 2026-10-12, and the figure beside each was that class's time per argument. The
/// float lane's five rows carry a stated division form in place of the run's; the criterion above
/// the row list is that cell's source, and it is the one cell of those rows the run did not
/// decide. A row is a property of the machine it was measured on, so a build on another host has
/// its own winners and the probe is what finds them.
///
/// **The device rows are runs' own winners.** Every member of the space - the
/// library's 352 device option rows crossed with its three division forms - was
/// measured on a Quadro T1000 on 2026-10-06 and published, 1056 of 1056, none
/// refused; a re-run of the same space on 2026-10-12 measured 1056 of 1056 again,
/// placed four of the twelve classes, and those rows carry that day's name, while
/// the other eight keep the 2026-10-06 rows it wrote back because it could not
/// separate their top entries. A row's record name says which run it came from. A
/// run measures that no entry of a class beats the entry its row names; it may not
/// be able to order the fastest entries of a class against each other, so each row
/// is one of the entries that class measured as equally fast and each report states
/// the resolution its run reached beside it. A device report prices a class's
/// entries by subtracting a kernel without the call from the same kernel with it,
/// so an entry whose arithmetic sits inside that kernel's own traffic is listed as
/// set aside rather than priced, and the fastest entry a report names is the
/// fastest of those it resolved. A row belongs to the card it was
/// measured on, so another card has its own winners.
///
/// **What a row's combination is, and what it came from, are its last two cells
/// rather than a sentence beside it.** The basis is `RowBasis::kMeasured` where a
/// run ranked the class and the combination below is a winner that run took, and
/// `RowBasis::kChosen` where the combination is a default this file states: the
/// fallback the seven names make, a class a run ranked no cell of, a class
/// whose one entry stood alone, or a run's winner whose division form the
/// criterion below moved. The record names the run the row came from, as
/// `<machine>-<date>`, and it is empty exactly where no run stands behind the row.
/// A name and not a revision: a revision is a second thing to rewrite on every
/// re-record, and a row that carried one would describe the tree reading it rather
/// than the run that produced it. The record a name points at is the file that run
/// wrote, kept under `tests/data/`, which carries the revision the run was taken
/// at and the paths the measurement depends on.
///
/// The marker above a row is prose and carries the rest: on a measurement, the
/// figure the class's winner took and how it was reached; on a choice, why the run
/// wrote the combination it states.
///
/// **The float lane's rows name the refined reciprocal, and that cell is a stated
/// choice.** The lane's plain reciprocal rounds once more per step and publishes its own 1e-7
/// term beside the lane's 1.5e-7 base, so a row naming it carries a 2.5e-7 figure, while the
/// certified-bound model places the engine's fits at a 1e-7 seed. The refined reciprocal
/// recovers the exact quotient and is bit-identical to exact division, so a row naming it
/// publishes the lane's 1.5e-7 base, and the model's 1e-7 fit-placement figure plus the
/// format's half digit is what remains. The five float rows are `RowBasis::kChosen` with no run
/// behind them: their other cells are the 2026-10-12 run's winners, and the form is this file's
/// statement.
///
/// Cells are written as enumerator names, bare where the row's own key is what the
/// enumeration's name means and qualified where an axis is, and the list is
/// expanded where every enumeration it names is visible - so a cell naming an
/// enumerator another axis owns is a compile error there rather than a default
/// nobody reads. Every row writes all seven axis cells, the exponential included: a
/// row that left one out would be compiled with a template default the row never
/// chose. The two provenance cells are the row macro's own parameters for the same
/// reason, so a row that leaves either out does not compile either - a row whose
/// provenance nothing states is the defect this format exists to remove.
#define BOYS_BUILD_DEFAULT_ROWS(X)\
    /* measured: 42.08 ns per argument on this host, reached by chosen-among-equals */\
    X(kHost, kFp64, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "host-2026-10-12")\
    /* chosen: the form cell is the 1e-7 criterion's - refined reciprocal, bit-identical to exact
       division, publishing the lane's 1.5e-7 base; the other cells are the 2026-10-12 run's */\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate, RowBasis::kChosen, "")\
    /* measured: 165.49 ns per argument on this host, reached by vote */\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 171.88 ns per argument on this host, reached by ordered */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 400.46 ns per argument on this host, reached by refined */\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 50.03 ns per argument on this host, reached by vote through the sorted-arguments call */\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 74.33 ns per argument on this host, reached by refined */\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* chosen: the form cell is the 1e-7 criterion's - refined reciprocal, bit-identical to exact
       division, publishing the lane's 1.5e-7 base; the other cells are the 2026-10-12 run's */\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast, RowBasis::kChosen, "")\
    /* chosen: the form cell is the 1e-7 criterion's - refined reciprocal, bit-identical to exact
       division, publishing the lane's 1.5e-7 base; the other cells are the 2026-10-12 run's */\
    X(kHost, kFp32, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate, RowBasis::kChosen, "")\
    /* chosen: the form cell is the 1e-7 criterion's - refined reciprocal, bit-identical to exact
       division, publishing the lane's 1.5e-7 base; the other cells are the 2026-10-12 run's */\
    X(kHost, kFp32, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate, RowBasis::kChosen, "")\
    /* chosen: the form cell is the 1e-7 criterion's - refined reciprocal, bit-identical to exact
       division, publishing the lane's 1.5e-7 base; the other cells are the 2026-10-12 run's */\
    X(kHost, kFp32, kAllNAtOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate, RowBasis::kChosen, "")\
    /* measured: 330.03 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 755.84 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 141.99 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 167.72 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 98.92 ns per argument on this host, reached by vote */\
    X(kHost, kBf16, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 240.20 ns per argument on this host, reached by refined */\
    X(kHost, kBf16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 554.62 ns per argument on this host, reached by refined */\
    X(kHost, kBf16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 102.27 ns per argument on this host, reached by vote */\
    X(kHost, kBf16, kAllN, FitRoute::kRationalMinimax, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 115.43 ns per argument on this host, reached by chosen-among-equals */\
    X(kHost, kBf16, kAllNAtOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "host-2026-10-12")\
    /* measured: 'device-single-fp64-fast', in its class's fastest group */\
    X(kDevice, kFp64Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* measured: 'device-single-fp32', in its class's fastest group */\
    X(kDevice, kFp32Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* measured: 'device-single-fp16-fast-exact-division', in its class's fastest group */\
    X(kDevice, kFp16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kExactDivision,\
      RegionBExp::kFast, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* measured: 'device-all-orders-fp64-narrow-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kFp64Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* the class could not be ordered: its 6 fastest entries could not be separated at the 32.62% it could order at; replaces the row the table in force carries */\
    X(kDevice, kFp32Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kOrders, FitGranularity::kUniform,\
      DivisionForm::kExactDivision, RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-12")\
    /* measured: 'device-all-orders-fp16-uniform-horner-exact-division', in its class's fastest group */\
    X(kDevice, kFp16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* measured: 'device-all-n-fp64-fast-exact-division', reached by ordered; this class's own rounds placed every rival of it behind it; replaces the row the table in force carries */\
    X(kDevice, kFp64Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kExactDivision, RegionBExp::kFast, RowBasis::kMeasured, "quadro-t1000-2026-10-12")\
    /* measured: 'all-n-fp32-fast-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kFp32Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* the class could not be ordered: its 4 fastest entries could not be separated at the 111.48% it could order at; replaces the row the table in force carries */\
    X(kDevice, kFp16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kCoarsest,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast, RowBasis::kMeasured, "quadro-t1000-2026-10-12")\
    /* measured: 'device-single-bfloat16-exact-division', in its class's fastest group */\
    X(kDevice, kBf16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\
    /* the class could not be ordered: its 3 fastest entries could not be separated at the 4485.48% it could order at; replaces the row the table in force carries */\
    X(kDevice, kBf16Device, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kOrders, FitGranularity::kUniform,\
      DivisionForm::kExactDivision, RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-12")\
    /* measured: 'device-all-n-bfloat16-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kBf16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate, RowBasis::kMeasured, "quadro-t1000-2026-10-06")\

/// Which of the two files this is: the committed one defines it, a replacement
/// does not, and the seam test reads it to know which pins apply. It is not a claim
/// about measured against unmeasured: this file carries the probe's measured rows.
#define BOYS_BUILD_DEFAULTS_COMMITTED 1

// A replacement defines BOYS_BUILD_DEFAULTS_REPLACED PUBLIC. Reaching this file under it means
// include/ precedes the replacement on the include path, so a unit here and the library's own
// instantiations would resolve different defaults. The #error below is the diagnostic.
#if defined(BOYS_BUILD_DEFAULTS_REPLACED)
#error                                                                                             \
    "this build was configured with BOYS_BUILD_DEFAULTS, but this translation unit read the committed defaults header: include/ comes before the build's own copy of it on the include path, so this unit and the library's own instantiations would resolve different defaults"
#endif
