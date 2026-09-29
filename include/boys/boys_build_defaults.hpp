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
/// ONE SET OF FIVE IS ONE POINT. No single combination is cheapest for every
/// question, which is why the option probe keys its ranking by question shape
/// and precision (`OptionProbeShape`, `OptionProbeCell`) rather than naming one
/// winner. A build that replaces this file has chosen for the shapes its own
/// probe weighed; a caller whose mix is a different one names the policy at the
/// call site, which is where a per-shape answer lives.
///
/// WHAT A REPLACEMENT CARRIES
///
///  - the five names below. It is read INSTEAD of this file and not beside it,
///    so one it omits is a compile error in the header that wanted it;
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
