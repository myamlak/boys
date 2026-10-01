// One of the build's five choices, moved on its own: how narrowly the fitted domain is
// cut when the call site names no policy. A replacement header for
// include/boys/boys_build_defaults.hpp, pointed at with the BOYS_BUILD_DEFAULTS option,
// so that a configure which fails can say WHICH axis failed: tests/build_defaults_tuned.hpp
// moves this one beside the evaluation scheme and the division form, and a report from
// that build cannot say which of the three a failure belongs to.
//
//   cmake -S . -B build-defaults-fit-granularity \
//       -DBOYS_BUILD_DEFAULTS=tests/build_defaults_fit_granularity.hpp
//
// It is a fixture and not a measurement: the value it moves is one this library carries
// and certifies for every build, and none of it is a claim about any machine. A build
// that has measured its own machine writes the host, the date and the option probe's own
// figures beside its choices (include/boys/boys_build_defaults.hpp states the contract a
// replacement satisfies, and this file satisfies it at the shipped values plus one).
//
// WHICH MEMBER MOVES, AND WHICH ONE NO BUILD CAN NAME
//
// Moved: the narrow partition, the committed file's value, to the shipped partition -
// region A's two bands per order and region B's single seed, at the degrees the committed
// tables carry. It is the value the tuned fixture names on this axis too, so a configure
// of this file reads one of the three axes that fixture crosses.
//
// The member no build can name is kUniform and not this one: the batched bodies have no
// branch for it (RefuseUniformPartition, include/boys/boys_impl.hpp) and the rational
// route's rung selector has no pair to cut for it (the static_assert in
// RationalRouteFitAtRung, the same header). What that refuses is one of the three values
// this axis offers, and not the axis: kNarrow and kShipped are both reachable, and
// CONTRIBUTING.md states the same two refusals where it says which choices a build cannot
// move. The shipped partition is also the one the library reads back most often - the
// accuracy gate is defined at it, the packed lane carries it at every rung beside the
// narrow one, and the entries' own fallbacks name it - so a build resolving it is a
// configuration the library already exercises rather than a path this file invents.
//
// WHAT THIS CONFIGURE MEETS BEFORE IT MEETS THE AXIS
//
// This build names the shipped route and the shipped scheme, and the replaced-build
// branch of tests/boys_backend_test.cpp refuses a replacement on exactly those two:
//
//   constexpr bool kShippedDefaultsInForce =
//       boys::kDefaultFitRoute == boys::FitRoute::kChebyshev &&
//       boys::kDefaultEvalScheme == boys::EvalScheme::kHorner;
//   static_assert(!kShippedDefaultsInForce, "the defaults header in force names the
//   shipped route and scheme, so this build has chosen nothing: point BOYS_BUILD_DEFAULTS
//   at a header that moves at least one axis, or unset it to build the shipped
//   configuration");
//
// The message says "at least one axis"; the predicate reads the route and the scheme.
// A build that moves only this axis satisfies the message and not the predicate, so that
// file does not compile under this fixture, and nothing this file can carry changes it:
// it moves one axis of five, and the guard reads two of the other four. What such a
// configure reports first is therefore the guard's answer and not this axis's, and the
// reading for the axis itself needs the predicate widened to the five - or the route or
// the scheme moved as well, which is what the tuned fixture does and is exactly why its
// failures cannot be attributed.

// Moved: the partition the certified lanes are defined by, in place of the narrow
// partition the committed file names.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kShipped

// Shipped: the committed file's own values, so that the fit granularity above is the
// single value this build resolves differently from the shipped configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kRefinedReciprocal
