#pragma once

/// \file
/// The choices a call site that names no policy resolves to, as the build that
/// compiled it fixed them: the host lane's five - the fit route, the evaluation
/// scheme, the packing axis, the division form and the fit granularity - beside
/// the device lane's two, the division form its entries take and the region-B
/// exponential its tables read.
///
/// They are named here as macros rather than as constants of their own. The
/// enumerations they choose between are declared in two headers,
/// `boys/accuracy.hpp` and `boys/backend.hpp`, and the second includes the
/// first, so no one header could declare constants of all five types and be
/// included by both. Each owning header instead expands its own name where its
/// own enumeration is visible: the constant stays beside the type it is a
/// default for, and what travels is the value. A replacement therefore hands in
/// names, and the compiler checks each name against the enumerators of the axis
/// it initialises — a name from another axis is a compile error rather than a
/// different default.
///
/// The committed file carries the shipped choices: the ones every bound in this
/// repository was measured with, and the ones a build that replaces nothing
/// compiles. A build that has measured its own machine — the option probe
/// (`boys/boys_probe.hpp`) is what measures one — may point the
/// `BOYS_BUILD_DEFAULTS` CMake option (CONTRIBUTING.md) at a header carrying its
/// own seven, and the entries that name no policy then compile those choices.
/// Nothing is paid for this at run time: a choice was a template argument
/// before and is one still, so an unnamed call compiles to the call it compiled
/// to before, at whatever values the build named.
///
/// ONE SET OF CHOICES IS ONE POINT, WHICH IS WHY THE TABLE IS HERE. No single
/// combination is cheapest for every question, which is why the option probe
/// keys its ranking by question shape and precision (`OptionProbeShape`,
/// `OptionProbeCell`) rather than naming one winner. The names below are one
/// point per lane: the host's five are what a host class the table carries no row
/// for resolves to, the device's two are what a device class's same fallback
/// carries, and the table beside them is where a per-class answer lives, one row
/// per (device, precision, shape) class. A build that replaces this file has
/// chosen the seven for the classes it names no row for and the rows for the
/// classes it does; a caller whose mix is a third one names the policy at the
/// call site.
///
/// WHAT A REPLACEMENT CARRIES
///
///  - the seven names below: the host's five and the device lane's two. It is
///    read INSTEAD of this file and not beside it, so one it omits is a compile
///    error in the header that wanted it. A replacement that names the host's
///    five and not the device's two is a build that has stated what its host
///    resolves and not what its card resolves, which is why the two are read by
///    the same file and not by the host's names;
///  - `BOYS_BUILD_DEFAULT_ROWS`, or not. A replacement that carries its own
///    probe output defines the list and the classes it names resolve to those
///    rows; one that carries no list is a build whose every class takes the
///    five, which is the configuration this file was before the list existed,
///    and the header that expands the list reads it only where it is defined;
///  - not `BOYS_BUILD_DEFAULTS_SHIPPED`, which is how the file in force says
///    which of the two it is. A build that replaces this file defines
///    `BOYS_BUILD_DEFAULTS_REPLACED` instead (CMakeLists.txt does, with the
///    option), and this file refuses to compile under that define: a unit that
///    reached it in such a build would be resolving different defaults from the
///    library's own instantiations;
///  - the machine, the date and the probe's own figures, in the comments beside
///    the choices. A choice made here is a measurement taken on one host, and a
///    figure belongs to the host it was taken on: a reader holding a tuned build
///    and a shipped one, with no way to tell them apart, cannot tell what a
///    report from either describes.
///
/// A cross-compiling configure cannot measure the machine it is compiling for,
/// so it keeps this file rather than replacing it, and the configure prints
/// which of the two is in force (CMakeLists.txt).

// The macros in this file carry the choices themselves. Each one's meaning is
// the axis it names; the constant that reads it is documented beside the
// enumeration it belongs to, and names this file back.
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
/// The device lane's own name, and not the host's above, because the two lanes'
/// published figures are two sets: every device entry takes the form as an
/// argument and runs every form, so this is what an unnamed device call runs and
/// not a statement about which forms exist. The value here is the form every
/// figure this repository publishes for that lane was measured at
/// (`boys/boys_device_tables.hpp`), so a build that names nothing keeps the
/// arithmetic its documents state, and one whose card measured another form
/// states it here without moving the host's default.
#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::kRefinedReciprocal

/// The region-B exponential an unnamed DEVICE call evaluates.
///
/// The device lane's own name for the same reason, and the two lanes' values
/// differ: this lane's published figures were measured at the library routine
/// (`RegionBExp::kAccurate`) where the host's default is its own reduced-
/// argument polynomial (`RegionBExp::kFast`, `boys/accuracy.hpp`). One name for
/// both targets could only be wrong on one of them.
#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::kAccurate

/// The classes this file sets a default for: **one row per class**, in the
/// table's own format
///
///     X(device, precision, shape, route, scheme, budget, packing axis,
///       granularity, division form, region-B exponential)
///
/// A class is a (device, precision, shape) triple - the device a call runs on,
/// the precision lane an entry is built at, and the question that entry answers
/// - and a row is the combination that class's entries compile when the call
/// site names no policy. The names above are the **fallback**, not the whole
/// answer: a host class this list carries no row for resolves to the host's five,
/// at its own lane's budget, and a device class to the device's own two beside the
/// four host choices its lane composes - so a file carrying no row at all is the
/// build this file was before the list existed.
///
/// **The precision cell is a LANE, and each half format is a lane of its own.**
/// `boys::Precision` is the lane a call runs in: `kFp16` is the binary16 entries
/// and `kBf16` the bfloat16 ones, which are one engine at one budget and differ
/// in the type they store. A class is keyed by the format a return carries, so
/// the two are two classes and the list below carries a row for each. The option
/// probe ranks the two formats as two classes, because what it times includes
/// the store, and the two rows are what lets each format's own winner be written
/// where its class resolves. One row for both would be a default the other
/// format reads: the figure a half-typed return carries is the format's own half
/// digit - 2^-11 for a binary16 store, 2^-9 for a bfloat16 one - and the two
/// classes' measured winners are not one combination.
///
/// **The rung is not a key, because the rung is the caller's and not the library's.** A caller
/// chooses the accuracy before the call and this table answers what the library then picks, so a
/// table keyed by rung would be seven rows of one answer per class. The row a class is given is
/// the one its own m = 1 class measured. The device's fastest entry does move with the rung, and
/// the option probe keys its classes by it - the two keys differ because they key different
/// things, a measurement there and the library's own choice here.
///
/// **The axes are names, checked where they are read.** Each cell is written as
/// the enumerator it means, unqualified by any prefix because the types are not
/// declared yet at the point this file is read. The list is expanded in the
/// header where every enumeration it names is visible, so a cell naming an
/// enumerator another axis owns is a compile error there rather than a default
/// nobody reads.
///
/// **A ROW CARRIES ONE CELL PER AXIS, AND THE SEVENTH IS NOT OPTIONAL.** The
/// axes a row names are the axes of the policy it resolves to, and the row's
/// macro takes one cell for each of them; a row that leaves one out does not
/// compile where the list expands, because the preprocessor knows how many
/// cells the row takes. That is load-bearing rather than tidy: every axis of
/// `EvalPolicy` has a default, so a row naming six of the seven would compile
/// with the seventh filled in by a value the row never chose, and `DefaultPolicy`
/// would hand that class's callers an arithmetic nobody decided. The exponential
/// cell below was the one this format left out, and both of its members are
/// offered on every host entry. The device rows below carry the same cell, and
/// theirs is the device lane's member rather than the host's: the two targets'
/// figures were measured at two arithmetics, which is why that cell is a
/// different name in the two halves of the list.
///
/// **EVERY CLASS THE LIBRARY CARRIES HAS A ROW**, and a class without one
/// is a build error rather than a call answered by something else. The list
/// below carries both halves of the table: one row per class the host's entries reach - the
/// five shapes (single, fixed-N, all-N, all-N-at-orders and all-orders) on each of the four
/// host lanes, the double lane, the float lane and the two half formats, because a shape no
/// entry carries has no default to state - beside one row per class the
/// device half holds, the three device lanes by the three questions a device entry answers. A
/// list missing one of them stops compiling rather than handing the missing class's callers a
/// combination nobody chose: `boys::DefaultPolicyFor` refuses the class it is asked for, and
/// `tests/boys_build_defaults_test.cpp` asks for each of the nine device classes by name, so a
/// dropped device row breaks that build rather than going quiet.
/// `boys-option-probe --emit-defaults` and `boys-device-probe --emit-defaults` are what write
/// them all.
///
/// **Each row says whether it is a measurement or a choice, and the difference
/// is in the comment above it.** A measured row is the option probe's own
/// winner for that class's m = 1 runs, on the host that produced this file:
/// the run of 2026-10-02, a 12-logical-processor AVX2+FMA host, window run 2,
/// seed 47. The double all-orders row measured 92.89 ns per argument where the
/// seam's own combination measures 172.88, and the float and fp16 all-orders
/// rows are the winners of the refinement vote that settled each of those
/// classes. The bf16 all-orders row is the run of 2026-10-04's own verdict for
/// that class - the run whose key carries the two half formats as two classes -
/// and its comment names that run. A row is a figure taken on one host, which is
/// why the machine, the date and the run are here.
///
/// **The device rows are one card's run, and the figure beside each is that run's own.** The nine
/// device rows are `boys-device-probe --emit-defaults` output from the run of 2026-10-04 on a
/// Quadro T1000 (compute capability 7.5, 14 streaming multiprocessors, driver 596.86, built with
/// toolkit 13.3.73 for architecture 75): 262144 arguments per call, 1048576 at the pair count, seed
/// 47, five passes of two rounds, every one of the space's 324 member(s) measured and none in two
/// states (the run's closure reads PASS, 324 of 324). Each figure is the winning entry's own
/// count-independent device time per argument - the arithmetic's own, transfer and host submission
/// excluded - in the units the lane's own table prints. A row is a figure taken on one card, which
/// is why that card, that date and that run are here.
///
/// **That run is provisional, by its own canary.** The same run reports its "fastest run seen here
/// is 2.7930 ms; the widest spread across the passes is 76.02%, beside the widest within-round
/// paired spread of 200.00%", and its pass table reads "5 of 5 pass(es) ran with the canary's own
/// runs wider than the 5.0% alarm" with the paired column at that column's own 200.00% ceiling on
/// every pass. Several of the nine classes were therefore settled by plurality vote and not by a
/// separated ordering. The rows state what that run measured on that card; they are not a
/// statement of what the card is shown to do, and a re-run on a quiet machine is what retires
/// this paragraph.
///
/// The host rows below the measured four name the five choices above, at their
/// lane's budget, because the probe ranks two of the five host shapes at this
/// revision - the all-orders ladder and the batch - so a class of a shape it
/// does not rank has no measurement to carry. The measured four are the four
/// host lanes' all-orders classes, each row its own class's winner and not a
/// neighbour's. They are written out rather than
/// left to anything implicit: a row written here is a default that has been
/// decided and can be read, and the marker above it says on what.
///
/// **The exponential cell of every host row that states the five is
/// `RegionBExp::kFast`, and it is written rather than left to the policy's
/// default.** `kFast` is
/// `kDefaultHostRegionBExp`, which is the member `EvalPolicy` filled in for
/// these rows while this format carried no cell for that axis, so every one of
/// them denotes the same policy it denoted before the cell existed and no figure
/// in this file moved. A row that states a measured winner carries the member
/// that winner was measured at, which is `RegionBExp::kAccurate` in the bf16
/// all-orders row below. It is written out because the alternative is a row whose
/// arithmetic is decided by a template default rather than by the table, and a
/// reader of the table cannot see which member a class runs. A row that wants
/// the other member names `RegionBExp::kAccurate`; both are certified and
/// neither substitutes for the other.
///
/// **A row a probe named by there being no rival is a choice, not a measurement.** The option
/// probes separate the two in the data they emit - a class whose winner won an ordering and a class
/// whose winner was simply the last entry left standing are different values of the probe's own
/// `how`, and its report says of the second that it is "an answer, and not the winner of a
/// comparison". A row written from the second is written with the marker a choice carries, never
/// with the one a measurement carries: a walkover recorded as a win is a default nobody measured.
///
/// What each row resolves to, and the bound it carries, is `boys::DefaultPolicy`
/// and `boys::DefaultGuarantee` in `boys/boys.hpp`. The probe writes this block
/// in this format - `boys-option-probe --emit-defaults <file>` - so the measured
/// rows are one run's output checked in rather than a transcription of it.
#define BOYS_BUILD_DEFAULT_ROWS(X)\
    /* measured: m = 1 all-orders ladder, 92.89 ns per argument */\
    X(kHost, kFp64, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* measured: m = 1 all-orders ladder, the refinement vote's winner */\
    X(kHost, kFp32, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner,\
      BoysBudget::kFloat, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kRefinedReciprocal, RegionBExp::kFast)\
    /* measured: the half lane's vote winner, from the run whose key carried the
       two half formats as one lane; this is the fp16 class's row */\
    X(kHost, kFp16, kAllOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* the five macros above: no probe run has ranked this shape on any lane */\
    X(kHost, kFp64, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp64, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp64, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp64, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp32, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp32, kAllN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* no probe run has ranked these two shapes on this lane: the row carries the shipped five */\
    X(kHost, kFp32, kFixedN, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    X(kHost, kFp32, kAllNAtOrders, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFloat,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
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
    /* measured: m = 1, the bf16 all-orders class's own entry, 116.06 ns per
       argument, reached by the unanimous vote over the refinement runs - the run
       of 2026-10-04, a 12-logical-processor AVX2+FMA host, seed 47, whose key
       carries the two half formats as two classes. The fp16 row above is the
       earlier run's verdict for the lane as that run keyed it - one lane for
       both formats; here the two rows are two classes' defaults */\
    X(kHost, kBf16, kAllOrders, FitRoute::kRationalMinimax, EvalScheme::kHorner,\
      BoysBudget::kFp16, PackAxis::kArguments, FitGranularity::kNarrow,\
      DivisionForm::kPlainReciprocal, RegionBExp::kAccurate)\
    /* no probe run has ranked these four shapes on this lane: the row carries the
       shipped five at the half budget, as the fp16 class's unranked shapes do */\
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
    /* A device row's cells are the winning entry's own axes, at the arithmetic the figure beside it
       was measured at: the route, the scheme, the packing axis and the granularity are the body the
       entry's kernels name, read from the lane's own option table (`boys/boys_cuda_options.hpp`),
       the form cell is the form that figure was taken at, and the exponential cell is
       `RegionBExp::kAccurate` - the device lane's member and not the host's, which is what the
       device seam names (`BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP`) and what every device figure this
       repository publishes was measured at. */

/// Which of the two files this is: the committed one defines it, a replacement
/// does not, and the seam test reads it to know which pins apply.
#define BOYS_BUILD_DEFAULTS_SHIPPED 1

// A build that replaces this file says so on its own command line, PUBLIC, so
// that every translation unit of it can tell which file it read. Reading this
// file under that define means an include path reaches include/ before the
// directory the replacement was copied into: the two files would then disagree
// about the same call in the same program, and the compiler is asked to say so
// here rather than to compile a program that mixes them.
#if defined(BOYS_BUILD_DEFAULTS_REPLACED)
#error                                                                                             \
    "this build was configured with BOYS_BUILD_DEFAULTS, but this translation unit read the committed defaults header: include/ comes before the build's own copy of it on the include path, so this unit and the library's own instantiations would resolve different defaults"
#endif
