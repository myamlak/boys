#pragma once

/// \file
/// The five choices a call site that names no policy resolves to, as the build
/// that compiled it fixed them: the fit route, the evaluation scheme, the
/// packing axis, the division form and the fit granularity.
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
/// own five, and the entries that name no policy then compile those choices.
/// Nothing is paid for this at run time: a choice was a template argument
/// before and is one still, so an unnamed call compiles to the call it compiled
/// to before, at whatever values the build named.
///
/// ONE SET OF FIVE IS ONE POINT, WHICH IS WHY THE TABLE IS HERE. No single
/// combination is cheapest for every question, which is why the option probe
/// keys its ranking by question shape and precision (`OptionProbeShape`,
/// `OptionProbeCell`) rather than naming one winner. The five names below are
/// one point: they are what a class the table carries no row for resolves to,
/// and the table beside them is where a per-class answer lives, one row per
/// (device, precision, shape) class. A build that replaces this file has chosen
/// the five for the classes it names no row for and the rows for the classes it
/// does; a caller whose mix is a third one names the policy at the call site.
///
/// WHAT A REPLACEMENT CARRIES
///
///  - the five names below. It is read INSTEAD of this file and not beside it,
///    so one it omits is a compile error in the header that wanted it;
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

/// The classes this file sets a default for: **one row per class**, in the
/// table's own format
///
///     X(device, precision, shape, route, scheme, budget, packing axis,
///       granularity, division form, region-B exponential)
///
/// A class is a (device, precision, shape) triple - the device a call runs on,
/// the precision lane an entry is built at, and the question that entry answers
/// - and a row is the combination that class's entries compile when the call
/// site names no policy. The five names above are the **fallback**, not the
/// whole answer: a class this list carries no row for resolves to those five, at
/// its own lane's budget, so a file carrying no row at all is the build this
/// file was before the list existed.
///
/// **The precision cell is a LANE, not a format.** `boys::Precision` is the lane
/// a call runs in - `kFp16` is the fp16 *and* bfloat16 entries, which are one
/// engine at one budget and differ only in the type they store - so a row
/// written for `kFp16` is the row both half-format entries resolve to, and there
/// is no `kBf16` cell to write. The option probe ranks the two formats as two
/// classes, because what it times includes the store; where its two winners were
/// one combination, as they are at this revision, the lane carries one row and
/// nothing is lost. Where they were not, that is a finding about the keying
/// rather than a row to fold: a lane is what the library can resolve, and a
/// format-specific default would need a format key that `Precision` does not
/// have.
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
/// offered on every host entry.
///
/// **EVERY HOST CLASS THE LIBRARY CARRIES HAS A ROW**, and a class without one
/// is a build error rather than a call answered by something else. The list
/// below is one row per class the host's entries reach: five shapes on the double lane, three on
/// the single-precision lane and two on the half lane, because a shape no entry carries has no
/// default to state. A list missing one of them stops compiling rather than handing
/// the missing class's callers a combination nobody chose.
/// `tests/boys_default_policy_test.cpp` asks for each class by name, and the
/// emission below is what writes them all.
///
/// **Each row says whether it is a measurement or a choice, and the difference
/// is in the comment above it.** A measured row is the option probe's own
/// winner for that class's m = 1 runs, on the host that produced this file:
/// 2026-10-02, a 12-logical-processor AVX2+FMA host, window run 2, seed 47. The
/// double all-orders row measured 92.89 ns per argument where the seam's own
/// combination measures 172.88, and the float and half all-orders rows are the
/// winners of the refinement vote that settled each of those classes. A row is a
/// figure taken on one host, which is why the machine, the date and the run are
/// here.
///
/// The rows below the measured three name the five choices above, at their
/// lane's budget, because the probe ranks two of the five host shapes at this
/// revision - the all-orders ladder and the batch - so a class of a shape it
/// does not rank has no measurement to carry. They are written out rather than
/// left to anything implicit: a row written here is a default that has been
/// decided and can be read, and the marker above it says on what.
///
/// **The exponential cell of every row below is `RegionBExp::kFast`, and it is
/// written rather than left to the policy's default.** `kFast` is
/// `kDefaultHostRegionBExp`, which is the member `EvalPolicy` filled in for
/// these rows while this format carried no cell for that axis, so every row here
/// denotes the same policy it denoted before the cell existed and no figure in
/// this file moved. It is written out because the alternative is a row whose
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
    /* measured: the vote's winner; the one lane both half formats run */\
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
    X(kHost, kFp16, kSingle, FitRoute::kChebyshev, EvalScheme::kHorner, BoysBudget::kFp16,\
      PackAxis::kArguments, FitGranularity::kNarrow, DivisionForm::kRefinedReciprocal,\
      RegionBExp::kFast)\
    /* the half lane is one lane for both half formats: `Precision::kFp16` is
       where the fp16 and bfloat16 entries live, so there is one row per shape
       and no second format to key. A format-specific default would need a
       format key that `Precision` does not have. */

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
