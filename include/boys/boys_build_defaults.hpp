#pragma once

/// \file
/// The choices a call site that names no policy resolves to, as the build that
/// compiled it fixed them: the host lane's five - the fit route, the evaluation
/// scheme, the packing axis, the division form and the fit granularity - beside
/// the device lane's two, the division form its entries take and the region-B
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
/// A replacement carries the seven names below, and `BOYS_BUILD_DEFAULT_ROWS` or
/// not - a replacement with no list is one whose every class takes the five. It
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
///       granularity, division form, region-B exponential)
///
/// A class is a (device, precision, shape) triple - the device a call runs on,
/// the precision lane an entry is built at, and the question that entry answers -
/// and a row is the combination that class's entries compile when the call site
/// names no policy. The names above are the **fallback** and not the whole
/// answer: a class this list carries no row for resolves to the host's five at
/// its own lane's budget, or to the device lane's two beside the four host
/// choices its lane composes.
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
/// **The host rows are measurements taken on one machine.** They are the option
/// probe's winners for this host - 12 logical processors, the AVX2+FMA tier
/// present, seed 47, 5 passes of 5 rounds over 16384 arguments to order 32 - run
/// 2026-10-05T10:19:50Z, and the figure beside each is that class's time per
/// argument. A row is a property of the machine it was measured on, so a build on
/// another host has its own winners and the probe is what finds them.
///
/// **The device rows are a run's own winners.** Every member of the space - the
/// library's 352 device option rows crossed with its three division forms - was
/// measured on a Quadro T1000 on 2026-10-06 and published, 1056 of 1056, none
/// refused. The run measured that no entry of a class beats the entry its row
/// names; it could not order the fastest entries of a class against each other,
/// so each row is one of the entries that class measured as equally fast and the
/// report states the resolution it reached beside it. A row belongs to the card
/// it was measured on, so another card has its own winners.
///
/// A row's marker says which of the two it is: a measurement, with the figure the
/// class's winner took and how it was reached; or a choice, where the run ranked
/// no cell of the class or an entry stood alone. Cells are written as enumerator
/// names, unqualified because the types are not declared yet at the point this
/// file is read, and the list is expanded where every enumeration it names is
/// visible - so a cell naming an enumerator another axis owns is a compile error
/// there rather than a default nobody reads. Every row writes all seven cells,
/// the exponential included: a row that left one out would be compiled with a
/// template default the row never chose.
#define BOYS_BUILD_DEFAULT_ROWS(X)\
    /* measured: 88.66 ns per argument on this host, reached by vote */\
    X(kHost, kFp64, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 94.49 ns per argument on this host, reached by vote */\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 171.74 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 302.68 ns per argument on this host, reached by vote */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 674.53 ns per argument on this host, reached by vote */\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 98.17 ns per argument on this host, reached by vote through the sorted-arguments call */\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 140.56 ns per argument on this host, reached by vote */\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 240.08 ns per argument on this host, reached by ordered */\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 104.60 ns per argument on this host, reached by vote */\
    X(kHost, kFp32, kAllN, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 551.11 ns per argument on this host, reached by refined */\
    X(kHost, kFp32, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 125.34 ns per argument on this host, reached by vote */\
    X(kHost, kFp32, kAllNAtOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 358.79 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 782.96 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 165.25 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 229.58 ns per argument on this host, reached by refined */\
    X(kHost, kFp16, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 107.21 ns per argument on this host, reached by refined */\
    X(kHost, kBf16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 253.84 ns per argument on this host, reached by ordered */\
    X(kHost, kBf16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 577.52 ns per argument on this host, reached by ordered */\
    X(kHost, kBf16, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 116.60 ns per argument on this host, reached by vote */\
    X(kHost, kBf16, kAllN, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 140.28 ns per argument on this host, reached by vote */\
    X(kHost, kBf16, kAllNAtOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 'device-single-fp64-fast', in its class's fastest group */\
    X(kDevice, kFp64Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: 'device-single-fp32', in its class's fastest group */\
    X(kDevice, kFp32Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 'device-single-fp16-fast-exact-division', in its class's fastest group */\
    X(kDevice, kFp16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kExactDivision,\
      RegionBExp::kFast)\
    /* measured: 'device-all-orders-fp64-narrow-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kFp64Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 'device-all-orders-fp32-rat-horner-exact-division', in its class's fastest group */\
    X(kDevice, kFp32Device, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate)\
    /* measured: 'device-all-orders-fp16-uniform-horner-exact-division', in its class's fastest group */\
    X(kDevice, kFp16Device, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate)\
    /* measured: 'device-all-n-fp64', in its class's fastest group */\
    X(kDevice, kFp64Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 'all-n-fp32-fast-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kFp32Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kPlainReciprocal,\
      RegionBExp::kFast)\
    /* measured: 'device-all-n-fp16-fast', in its class's fastest group */\
    X(kDevice, kFp16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: 'device-single-bfloat16-exact-division', in its class's fastest group */\
    X(kDevice, kBf16Device, kSingle, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kExactDivision,\
      RegionBExp::kAccurate)\
    /* measured: 'device-all-orders-bfloat16-uniform-rat-horner-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kBf16Device, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kOrders, FitGranularity::kUniform, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\
    /* measured: 'device-all-n-bfloat16-plain-reciprocal', in its class's fastest group */\
    X(kDevice, kBf16Device, kAllN, FitRoute::kChebyshev, EvalScheme::kSplitClenshaw, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kCoarsest, DivisionForm::kPlainReciprocal,\
      RegionBExp::kAccurate)\

/// Which of the two files this is: the committed one defines it, a replacement
/// does not, and the seam test reads it to know which pins apply. It is not a claim
/// about measured against unmeasured: this file carries the probe's measured rows.
#define BOYS_BUILD_DEFAULTS_COMMITTED 1

// A replacement says so on its own command line, PUBLIC, so that every
// translation unit of it can tell which file it read. Reading this file under
// that define means an include path reaches include/ before the directory the
// replacement was copied into: the compiler is asked to say so here rather than
// to compile a program that mixes the two.
#if defined(BOYS_BUILD_DEFAULTS_REPLACED)
#error                                                                                             \
    "this build was configured with BOYS_BUILD_DEFAULTS, but this translation unit read the committed defaults header: include/ comes before the build's own copy of it on the include path, so this unit and the library's own instantiations would resolve different defaults"
#endif
