// The test's own override of the build defaults: the header the seam test's
// second configure points BOYS_BUILD_DEFAULTS at, so that a build whose choices
// are not the shipped ones is exercised by the suite rather than by reading
// (include/boys/boys_build_defaults.hpp states the contract a replacement
// satisfies).
//
// It is a fixture and not a measurement: every value it moves is moved to one
// this library carries and certifies, and none of it is a claim about any
// machine. A build that tunes itself for real carries the machine, the date and
// the option probe's own figures beside its choices.
//
//   cmake -S . -B build-defaults-tuned -DBOYS_BUILD_DEFAULTS=tests/build_defaults_tuned.hpp
//
// WHAT IT MOVES, AND WHY A MOVED VALUE IS THE ONLY READING
//
// A replacement that names the shipped value and a macro that no header expands
// are the same build: both resolve an entry that names no policy to the value the
// library had already compiled. Three of these five sat exactly that way - named
// in the seam file, documented, and expanded by nothing - and this fixture could
// not tell, because it moved one choice and copied the library's own values for
// the other four. So a value is moved on every axis this fixture can move one on,
// and tests/boys_build_defaults_test.cpp is what reads the result: it holds each
// of the five values below to the constant that owns it and to the policy an
// entry that names no policy resolves to, so a macro dropped from its reader
// moves the resolved value and fails that check in this configure. An axis left
// at the shipped value is an axis that check has no teeth on, which is why the two
// it leaves for the library's own refusals, and the one it leaves for a pin, each
// say so below.
//
// WHAT IT LEAVES AT THE SHIPPED VALUE, AND WHY
//
// One axis is unmovable: the packing axis, refused by the library in its own
// words at the place a build naming it would fail. The uniform member of the fit
// granularity was the second until this revision, when the batched bodies that
// refused it began handing it to the path that reads the grid (the accuracy
// gate's entry book measures their six cells); no fixture in this tree names it,
// so what a build naming it compiles to is not measured. The third is movable and
// is held here by a pin in the seam test, which is a debt this file names rather
// than a property of the library. Each is quoted where its value is defined
// below.

#define BOYS_BUILD_DEFAULTS_TEST_FIXTURE 1

// Shipped, and the one axis a build can move that this fixture may not: the fit
// route's value is pinned where the seam test reads this file -
// tests/boys_backend_test.cpp, `static_assert(boys::kDefaultFitRoute ==
// boys::FitRoute::kChebyshev, "the fixture's fit route is not in force")` - so a
// fixture naming kRationalMinimax here fails that pin rather than compiling a
// moved route. That pin is what makes a configure that delivered some other
// header fail here rather than pass; moving the route is therefore a change to
// this file AND to that pin, in one commit, and not this one alone.
#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::kChebyshev

// Moved: the same fit, summed by the split Clenshaw recurrence instead of
// Horner's rule. Different rounding, so the values move in their last places.
#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::kSplitClenshaw

// Shipped, and unmovable: the orders axis is not an axis on a shape that
// produces one order, and the library refuses it in its own words where the
// refusal is a static_assert - include/boys/boys_impl.hpp, in BoysSingleImpl:
//
//   "this entry evaluates one order, so it has one order to put in a vector lane
//   and the orders axis is not an axis here: the axis this library carries on
//   this shape is the arguments axis"
//
// The batch entries refuse it the same way and for the same reason: "a packed
// lane keeps four orders of one argument in a register, and this call produces
// exactly one order at every argument of the array, so there are not four orders
// here to fill a lane with". A build that named kOrders here would fail to
// compile in the library's own sources - src/boys.cpp's tier selectors
// instantiate the single-order entry at EvalPolicy<> - which is the answer and
// not a default.
#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::kArguments

// Moved: one reciprocal per divisor and one product per step, in place of the
// refined form's recovered quotient. The axis exists to be moved - which of the
// three forms is cheapest is a property of the host, so all three are carried
// and ranked by the option probe - and the two the fixture does not name are
// instantiated beside this one on every lane the packed entries serve
// (boys/boys_impl.hpp's BOYS_ORD_* blocks, src/boys_orders_simd.cpp's
// BOYS_ORDERS_*_INSTANTIATIONS), so no lane loses a form by this file naming it.
#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::kPlainReciprocal

// Moved: the partition the certified lanes are defined by, in place of the
// narrow partition the shipped header names. It is not the member the build
// cannot move: that is the packing axis, and CONTRIBUTING.md says so. The
// granularity's uniform member was refused by the batched bodies until this
// revision and those bodies now hand it to the path that reads the grid, so that
// refusal is gone; what remains of the member's own refusal is the rational
// route's rung selector, which refuses a rung of it at compile time (the
// static_assert in RationalRouteFitAtRung, include/boys/boys_impl.hpp). No
// fixture in this tree names the member, so a build naming it is unmeasured
// here. kShipped is the reference partition: the accuracy gate reads it
// (tests/boys_accuracy_gate.cpp), the packed lane carries it at every rung beside
// the narrow one, and the entries' own fallbacks name it.
#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::kShipped

// The moves above are the compile-time half of the seam check, and this file's
// own configure is the only reading there is for them: the `Build defaults`
// step in `.github/workflows/ci.yml` points BOYS_BUILD_DEFAULTS at this file on
// one leg, so "the library reads BOYS_BUILD_DEFAULT_DIVISION_FORM and
// BOYS_BUILD_DEFAULT_FIT_GRANULARITY" is established by that configure building
// and passing its suite. Until that step existed no leg set BOYS_BUILD_DEFAULTS,
// and this file was written without running the configure at all. A failure in
// it is a finding about the axis that failed: the scheme is the move that was
// already in force, and the division form and the fit granularity are the ones
// added here.
