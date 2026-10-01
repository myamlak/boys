// One of the build's five choices, moved on its own: the form the chains' divisions are
// performed in. A replacement header for include/boys/boys_build_defaults.hpp, pointed at
// with the BOYS_BUILD_DEFAULTS option, so that a configure which fails can say WHICH axis
// failed: tests/build_defaults_tuned.hpp moves this one beside the evaluation scheme and
// the fit granularity, and a report from that build cannot say which of the three a
// failure belongs to.
//
//   cmake -S . -B build-defaults-division-form \
//       -DBOYS_BUILD_DEFAULTS=tests/build_defaults_division_form.hpp
//
// It is a fixture and not a measurement: the value it moves is one this library carries
// and certifies for every build, and none of it is a claim about any machine. A build
// that has measured its own machine writes the host, the date and the option probe's own
// figures beside its choices (include/boys/boys_build_defaults.hpp states the contract a
// replacement satisfies, and this file satisfies it at the shipped values plus one).
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
//
// WHAT A FAILURE IN THIS CONFIGURE IS A FINDING ABOUT
//
// The division form, and nothing else: the other four values below are the committed
// file's own. The axis exists to be moved - which of the three forms is cheapest is a
// property of the host, so all three are carried and ranked by the option probe
// (boys/boys_probe.hpp) - and the two forms this file does not name are instantiated
// beside it on every lane the packed entries serve, so no lane loses a form by this file
// naming one (boys/boys_impl.hpp's BOYS_ORD_* blocks, src/boys_orders_simd.cpp's
// BOYS_ORDERS_*_INSTANTIATIONS).

// Moved: one reciprocal per divisor and one product per step, in place of the refined
// form's quotient recovered from the product's error.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kPlainReciprocal

// Shipped: the committed file's own values, so that the division form above is the
// single value this build resolves differently from the shipped configuration.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kHorner
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kNarrow
