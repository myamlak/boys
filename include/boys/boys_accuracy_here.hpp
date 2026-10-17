#pragma once

/// \file boys_accuracy_here.hpp
/// What this build delivers, asked of the build itself: the eight lane figures
/// and each default-policy class's own, answered by the translation unit that
/// asks, with no archive on the link line and no tool to run.
///
/// The library's accuracy accessors are the archive's - `BoysLaneContracts`,
/// `BoysAccuracyGuaranteed`, `BoysAccuracyDelivered` and `QueryCombination` are
/// declared in boys/boys_decl.hpp and defined in src/boys.cpp - and a consumer
/// that links no archive cannot call them. What this header answers is the two
/// measured figures those entries carry, the absolute figure each lane was
/// published with and the figure the arithmetic was measured to deliver, read
/// from a table that lives in the headers and from the preprocessor state this
/// translation unit was compiled at. Nothing here evaluates the function, and
/// nothing here needs a binary.
///
/// **The bound here is the lane's measured absolute figure, not the contract at
/// a cell.** `BoysAccuracyGuaranteed` evaluates the lane's published form at the
/// `(n, x)` it is asked about - on the double lane `1e-13 * F_0(x)`, which is
/// `1e-13` at x = 0 - so the two differ, and neither is wrong: the archive's is
/// the promise a calculation rests on at the cell it evaluates, and this
/// header's is the reading the lane's runs produced over its whole domain. A
/// caller who needs the contract at a cell has the function value in hand and
/// takes the archive's entry; a caller asking what his build measured reads this
/// one.
///
/// **This header is not in boys/boys.hpp's chain.** A unit that includes the
/// front door pulls boys_decl.hpp, and through it accuracy.hpp, but never this
/// file; the answer to "what does this build deliver" is paid for by the units
/// that ask it and by no others. It costs those units boys_decl.hpp's
/// declarations, the route name backend.hpp carries, and one table of eight
/// rows: no kernel, no template instantiation and no run-time work.
///
/// ## What it detects, and how
///
/// The figures below are keyed by the arithmetic a build runs, and a build's
/// arithmetic is the preprocessor's to state:
///
/// - **the instruction-set tier**, `BOYS_ISA_AVX2_FMA` (boys/isa_tiers.hpp),
///   derived from the compiler's own predefines or stated by the consumer. It
///   is part of the identity a report prints and it is not a key to any figure:
///   the two tiers deliver the same bits, which is the whole of what the flag
///   set buys, so every row below holds at both.
/// - **the fp16 seam**, `BOYS_FP16_AXIS`, which decides whether the half lanes'
///   entries exist and not what they deliver.
/// - **the multiply-add route**, `BOYS_MULADD_SEPARATE`, read as a *value* the
///   way backend.hpp reads it: defined nonzero selects the separate route and
///   anything else - `=0` included - selects the fused one.
/// - **whether a bare product-plus-add is one rounding here**, the measured
///   `BOYS_SCALAR_CONTRACTS` the configure pins onto every consumer through the
///   boys-tier target. The macro is a tri-state: 1 is a measured one rounding, 0
///   a measured two, and undefined a build that has not measured at all - which
///   claims the least, because the answer it does not have is the one that costs
///   accuracy when it is assumed (CMakeLists.txt, where the configure measures
///   it by compiling and running the bare step).
///
/// Those state the **route in force**, which is what the figures are keyed by,
/// and it is not the same question as the selection: a build that contracts a
/// bare product-plus-add compiles the separate selection onto the fused step, so
/// the route in force is the fused route there whatever the selection says
/// (boys/backend.hpp, `RouteInForce`). This header answers it the same way, at
/// compile time: the selection, unless the separate route was selected and the
/// build measured that its bare form is one rounding.
///
/// An unmeasured contraction is answered at the reading that claims the least,
/// which is the not-contracted column: the tree's own convention, and the reason
/// an unmeasured build is handed the looser of two measured cells rather than no
/// number at all. Where the *separate* route is selected and contraction is
/// unmeasured the route in force is not a fact this unit carries at all - a
/// contracting build would run the fused arithmetic - and no figure is invented
/// for it: `BoysAccuracyHere` refuses, by name, at the point of asking.
///
/// ## The records
///
/// Every figure here is read from a committed statement, and each carries the
/// name of the statement it came from. The bound rows are the ones
/// `BoysLaneContracts()` states, in the absolute form that list carries; the
/// docs state the same rows as the relative contract the entries hold -
/// `docs/specification.md`, the accuracy contract, gives the double lane
/// `|error| <= the larger of 1e-13 * F_n(x) and 1e-13 * F_0(x)`, and
/// the float lane the larger of `2^-22 * F_n(x)` and 1.5e-7, plus 1e-7 under the
/// plain reciprocal - and the lane list is documented as re-made
/// with the option probe's report rather than with that page, so the four double
/// and float rows say so in their record instead of this header inventing the
/// relative numbers beside them. The delivered rows are the route tables the
/// docs publish (`docs/specification.md`, the route table, and
/// `docs/lane-contract.md`'s two route tables), whose columns are the route and
/// the contraction reading. Each is the class's own default cell - the sweep
/// behind it was taken at the class's default division form and region-B member -
/// so a call that names either of those two axes at another member is answered a
/// bound and no delivered figure: what that entry delivers is its own
/// measurement, and the option probe is the tree's instrument for it.
///
/// A record here is therefore a document and its section, or a record name under
/// `tests/data/` in the sense boys/boys_build_defaults.hpp states; the class
/// answer carries the seam row's own record name beside them, which is the run
/// its cells were measured at where one stands behind the row.
///
/// ## The refusal
///
/// A build whose detected arithmetic is outside the records - the separate route
/// selected with contraction unmeasured - hears it loudly and by name: asking
/// `BoysAccuracyHere` for a class figure is a compile-time error whose message
/// states what was detected and that no figure applies. Nothing is wrong with
/// the bound there: it holds at every arithmetic, and `BoysLaneFiguresHere`
/// states it. The answer to "which build am I" stays askable too -
/// `ArithmeticInForceHere()` compiles at every arithmetic and its name states
/// the detected profile, which is what a program that reports its own build
/// prints.
///
/// \sa boys/boys_decl.hpp for the archive's own answers to the same questions
/// \sa docs/specification.md and docs/lane-contract.md for the records

#include "boys/accuracy.hpp"
#include "boys/backend.hpp"
#include "boys/boys_decl.hpp"
#include "boys/isa_tiers.hpp"

#include <array>
#include <cstddef>
#include <span>

namespace boys {
namespace detail {

/// Whether this build has measured whether a bare product-plus-add is one
/// rounding here: \c BOYS_SCALAR_CONTRACTS is defined at all, whatever value it
/// carries. Left undefined where no answer was measured, so an absent macro is
/// the third state and not the answer "two roundings".
inline constexpr bool kContractionMeasuredHere =
#if defined(BOYS_SCALAR_CONTRACTS)
    true;
#else
    false;
#endif

/// What the measurement answered, read as a value the way backend.hpp reads the
/// route selection: \c BOYS_SCALAR_CONTRACTS=0 is the two-rounding answer and
/// not a present macro.
inline constexpr bool kContractionContractsHere =
#if defined(BOYS_SCALAR_CONTRACTS) && BOYS_SCALAR_CONTRACTS
    true;
#else
    false;
#endif

/// Whether the route in force is a fact this build carries: the selection,
/// unless the separate route was selected and the contraction question is
/// unanswered - which is the one state boys/backend.hpp's `RouteInForce` has to
/// measure at run time to answer, and which no record here holds a figure for.
inline constexpr bool kRouteInForceKnownHere =
    backend::detail::kSelectedRoute == backend::MulAddRoute::kFused || kContractionMeasuredHere;

/// The route in force, as a compile-time value. Read it only where
/// \c kRouteInForceKnownHere is true: an unmeasured contraction is answered here
/// at the reading that claims the least, which is the not-contracted one.
inline constexpr backend::MulAddRoute kRouteInForceHere =
    (backend::detail::kSelectedRoute == backend::MulAddRoute::kFused || kContractionContractsHere)
        ? backend::MulAddRoute::kFused
        : backend::MulAddRoute::kSeparate;

} // namespace detail

/// The arithmetic this translation unit was compiled at, as its preprocessor
/// state answers it: which route is in force, what the build measured, and what
/// it has not.
///
/// This is the profile every figure in this header is keyed by, and it is a
/// report rather than a decision: nothing here selects a lane or a route, and
/// reading this struct compiles at every arithmetic, including the one no figure
/// is stated for.
///
/// \ingroup boys
struct ArithmeticInForce {
    /// \c BOYS_ISA_AVX2_FMA: 1 for the AVX2 + FMA tier, 0 for the scalar one.
    int isaTier = 0;

    /// \c BOYS_FP16_AXIS: 1 where this unit was compiled with the fp16 seam
    /// open, 0 where it was closed.
    int fp16Seam = 0;

    /// Whether this build has measured whether a bare product-plus-add is one
    /// rounding here: \c BOYS_SCALAR_CONTRACTS defined at all.
    bool contractionMeasured = false;

    /// What that measurement answered, false where none was taken.
    bool contracts = false;

    /// The route \c BOYS_MULADD_SEPARATE selected.
    backend::MulAddRoute selectedRoute = backend::MulAddRoute::kFused;

    /// Whether the route in force is a fact this build carries: false exactly
    /// where the separate route was selected and contraction is unmeasured.
    bool routeInForceKnown = false;

    /// The route in force. Read it only where \c routeInForceKnown is true: an
    /// unmeasured contraction leaves the question open - a contracting build
    /// compiles the separate selection onto the fused step - and this field then
    /// holds the fused route as a placeholder rather than as an answer.
    backend::MulAddRoute routeInForce = backend::MulAddRoute::kFused;
};

/// The arithmetic this translation unit was compiled at.
///
/// \returns the profile, read at compile time from the macros above and from no
///          other state; the same value in every call
///
/// \ingroup boys
constexpr ArithmeticInForce ArithmeticInForceHere() noexcept {
    ArithmeticInForce arithmetic;

    arithmetic.isaTier = BOYS_ISA_AVX2_FMA;
    arithmetic.fp16Seam = BOYS_FP16_AXIS;
    arithmetic.contractionMeasured = detail::kContractionMeasuredHere;
    arithmetic.contracts = detail::kContractionContractsHere;
    arithmetic.selectedRoute = backend::detail::kSelectedRoute;
    arithmetic.routeInForceKnown = detail::kRouteInForceKnownHere;
    arithmetic.routeInForce =
        detail::kRouteInForceKnownHere ? detail::kRouteInForceHere : backend::MulAddRoute::kFused;

    return arithmetic;
}

/// The name of an arithmetic, as a report prints it, and never null.
///
/// The name states the profile the figures are keyed by: the route in force
/// where it is one this build carries, and the two macros that leave it
/// undetermined where it is not.
///
/// \param arithmetic the profile to name
///
/// \returns a string literal naming it, valid for the life of the program
///
/// \ingroup boys
inline const char* ArithmeticInForceName(const ArithmeticInForce& arithmetic) noexcept {
    if (!arithmetic.routeInForceKnown)
    {
        return "no route in force: BOYS_MULADD_SEPARATE selects the separate route and "
               "BOYS_SCALAR_CONTRACTS is not defined, so whether a bare product-plus-add is one "
               "rounding here is unmeasured - and a contracting build compiles the separate "
               "selection onto the fused step, so this build has not said which arithmetic it "
               "runs";
    }

    if (arithmetic.selectedRoute == backend::MulAddRoute::kSeparate)
    {
        return "the fused route by contraction: the separate route is selected and a bare "
               "product-plus-add is one rounding here (measured), so the selection compiles onto "
               "the fused step";
    }

    if (!arithmetic.contractionMeasured)
    {
        return "the fused route, with this build's contraction unmeasured: whether a bare "
               "product-plus-add is one rounding here has not been asked, and the figures answer "
               "at the reading that claims the least";
    }

    if (arithmetic.contracts)
    {
        return "the contracted arithmetic: the fused route is selected and a bare "
               "product-plus-add is one rounding here (measured), so every step rounds once";
    }

    return "the fused route: the selected step rounds once and a bare product-plus-add is two "
           "roundings here (measured)";
}

/// One lane's figures as this header states them, with the record they came from.
///
/// The figures are the ones the lane's row of `BoysLaneContracts()` carries, in
/// that row's form and with that row's meaning: \c bound is the figure the lane
/// documents for one value, \c additive a term one of the region-B exponentials
/// adds beside it, \c plainAdditive the term the plain reciprocal adds beside it,
/// and \c additiveMember the member \c additive is under.
///
/// \ingroup boys
struct LaneFigureHere {
    Precision precision = Precision::kFp64; ///< the lane whose figures these are
    const char* name = ""; ///< the name a report prints it under
    double bound = 0.0; ///< the documented base figure per value
    double additive = 0.0; ///< the term the lane's \c additiveMember adds beside the base
    double plainAdditive = 0.0; ///< the term the plain reciprocal adds beside the base
    RegionBExp additiveMember = RegionBExp::kFast; ///< the member \c additive is under
    const char* record = ""; ///< the statement the row's figures are read from
    /// The record for the figure the plain reciprocal gives, where that figure is
    /// not the base. Empty where the forms share one figure, which is what the
    /// lane's \c record then already states.
    const char* plainRecord = "";
};

namespace detail {

/// The lane rows this header states, one per member of \c Precision and in that
/// enumeration's order - the same eight rows and the same order
/// `BoysLaneContracts()` answers with, so the two are read index for index.
///
/// The figures are the absolute ones that list carries: they sit at or below
/// the contract the docs state (`docs/specification.md`) near x = 0, and
/// above it wherever the lane's relative arm falls under the constant. The
/// list's figure is therefore the stricter claim below that crossing and the
/// looser one above it, where the contract is the statement a caller is held
/// to; a query this list answers yes to is one the contract covers only
/// below that crossing.
inline constexpr std::array<LaneFigureHere, 8> kLaneFiguresHere = {{
    {Precision::kFp64, "fp64", 5.5e-14, 0.0, 0.0, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: the double lane's bound is the larger of "
     "1e-13 * F_n(x) and 1e-13 * F_0(x), and 5.5e-14 is the absolute figure the lane measured "
     "(docs/lane-contract.md), which also states where that figure crosses the contract",
     ""},
    {Precision::kFp32, "fp32", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: the float lane's bound is the larger of "
     "2^-22 * F_n(x) and 1.5e-7, plus 1e-7 in the plain-reciprocal form (docs/lane-contract.md), "
     "which also states where that figure crosses the contract",
     "docs/specification.md, the accuracy contract: the plain reciprocal's own 1e-7 term beside "
     "the base, whose measured worst is 1.7514e-07 absolute at n = 0, x = 9.74055, where the two "
     "forms that divide exactly deliver 1.08354e-07 at worst"},
    {Precision::kFp16, "fp16", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: 1.5e-7 plus one half of the returned value's "
     "last representable digit",
     "the fp32 engine's plain-reciprocal term, 1e-7, beside the base, before this format's own "
     "half digit"},
    {Precision::kFp32Device, "fp32-device", 1.5e-7, 8e-8, 0.0, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: the float lane's figure, plus the fast "
     "region-B exponential's own 8e-8 term",
     ""},
    {Precision::kFp64Device, "fp64-device", 5.5e-14, 0.0, 0.0, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract (CUDA fp64): the same contract as the CPU "
     "double lanes, the larger of 1e-13 * F_n(x) and 1e-13 * F_0(x), at this lane's own entries "
     "(docs/lane-contract.md)",
     ""},
    {Precision::kFp16Device, "fp16-device", 1e-7, 0.0, 1e-7, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: 1e-7 plus one half of the returned value's "
     "last representable digit",
     "the device half lane's own plain-reciprocal term, 1e-7, beside the base: at n = 32, "
     "x = 11.9453125 the lane delivers zero there where the forms that divide exactly are inside "
     "the base"},
    {Precision::kBf16, "bf16", 1.5e-7, 0.0, 1e-7, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract: the fp16 lane's figure, at this format's own "
     "half digit (2^-8)",
     "the fp32 engine's plain-reciprocal term, 1e-7, beside the base, before this format's own "
     "half digit"},
    {Precision::kBf16Device, "bf16-device", 1e-7, 0.0, 1e-7, RegionBExp::kFast,
     "docs/specification.md, the accuracy contract (CUDA fp16): 1e-7 plus one half of the "
     "returned value's last representable digit, which is this format's own 2^-8; the class is "
     "the device lane's bfloat16 entries (boys/boys_cuda.hpp, SingleBf16 and AllOrdersBf16)",
     "the device half lane's own plain-reciprocal term, 1e-7, beside the base, before this "
     "format's own half digit"},
}};

/// The delivered figures the records state, one row per arithmetic, with the
/// record they are read from.
///
/// A row's three figures are the docs' route tables at one column: the double
/// lane's single entry over its band, the float lane's single entry and the
/// float lane's batch entry, both over all arguments. A figure is a swept
/// maximum - the worst cell of the sweep the gate's recorded run holds - and not
/// a bound.
struct DeliveredRow {
    double doubleBand = 0.0; ///< the double lane's single entry over the band
    double floatSingle = 0.0; ///< the float lane's single entry over all arguments
    double floatBatch = 0.0; ///< the float lane's batch entry over all arguments
    const char* record = ""; ///< the statement the three are read from
};

/// The fused route on a build whose bare product-plus-add is two roundings.
inline constexpr DeliveredRow kDeliveredFusedTwoRoundings = {
    3.22e-15,
    6.36e-08,
    1.08e-07,
    "docs/specification.md, the route table (the fused route, MSVC x64 Release at this revision) "
    "and docs/lane-contract.md's route tables, the not-contracted column, measured on MSVC arm64, "
    "MSVC x86-64, clang x86-64 and AppleClang x86-64, which agree to the digit"};

/// The separate route on a build whose bare product-plus-add is two roundings.
inline constexpr DeliveredRow kDeliveredSeparate = {
    4.63e-15,
    7.49e-08,
    1.08e-07,
    "docs/specification.md, the route table (the separate route, MSVC x64 Release at this "
    "revision) and docs/lane-contract.md's route table of the alternative"};

/// The contracted arithmetic, whatever the selection says: every bare
/// product-plus-add is the fused step there.
inline constexpr DeliveredRow kDeliveredContracted = {
    2.74e-15,
    6.36e-08,
    9.11e-08,
    "docs/lane-contract.md, the route table of the contracting host (the contracted column: "
    "x86-64 built with -mfma, the licence gcc and clang take on arm64) and docs/specification.md's "
    "contracting sentence"};

/// The record for the not-contracted column where this build has not measured
/// contraction at all. The figure is the same cell; what changes is that the
/// build is answered at the reading that claims the least rather than at a
/// measurement of its own.
inline constexpr const char* kDeliveredUnmeasuredRecord =
    "docs/lane-contract.md's route tables, the not-contracted column, read as the least this "
    "build can claim: BOYS_SCALAR_CONTRACTS is not defined, so the configure has not measured "
    "whether a bare product-plus-add is one rounding here (CMakeLists.txt; "
    "docs/specification.md's route table carries the same reading for MSVC x64 Release)";

/// Why no delivered figure is stated for a half-format lane.
inline constexpr const char* kHalfLaneReason =
    "no delivered figure is stated for a half-format lane: its error is dominated by the format's "
    "own half digit at the returned value, no row of this library measured a half-typed return, "
    "and the guaranteed figure is the one to use (boys/boys_decl.hpp, BoysAccuracyDelivered)";

/// Why no delivered figure is stated for a device lane.
inline constexpr const char* kDeviceLaneReason =
    "a delivered figure for a device lane is a card's, and a gate built with CUDA is where it is "
    "taken; this header quotes no figure for a lane it cannot answer for";

/// Why no delivered figure is stated for the double lane's batch classes.
inline constexpr const char* kDoubleBatchReason =
    "the route tables state the double lane's single entry; the batch shapes' readings are the "
    "gate's own report (tests/data/boys_accuracy_gate_run.txt), in the lane's figure units, and "
    "this header quotes no figure it cannot name";

/// Why no delivered figure is stated for an entry the routes' sweeps were not
/// taken at: the two policy axes that move a figure are the division form and the
/// region-B member, and a call naming either at another member names an entry
/// whose delivered figure is its own measurement.
inline constexpr const char* kNonDefaultCellReason =
    "the delivered rows are the class's own default cell's, measured at the default policy's "
    "division form and region-B member; this call names another member of one of those two axes, "
    "so its entry's delivered figure is its own measurement - the one the option probe takes of a "
    "build - and the bound beside this reason holds at every member";

/// Why no delivered figure is stated for an arithmetic whose route in force is
/// not determined. The same refusal `BoysAccuracyHere` makes as a diagnostic,
/// answered as a value here so that no reader of this table is handed a number.
inline constexpr const char* kUndeterminedArithmeticReason =
    "no delivered figure is stated for an arithmetic whose route in force is not determined: "
    "BOYS_MULADD_SEPARATE selects the separate route and BOYS_SCALAR_CONTRACTS is not defined, so "
    "whether a bare product-plus-add is one rounding here is unmeasured - and a contracting build "
    "compiles the separate selection onto the fused step (boys/backend.hpp, RouteInForce), so the "
    "records' three cells name three arithmetics this build has not said it is one of";

/// Whether the classes this header answers for name a lane, a shape and a device
/// this library carries. A value past an enumeration names no class on any lane,
/// and the table below is indexed by the lane, so a cast-in value would read a
/// row that is not there.
///
/// \returns whether every key is an enumerator of its own enumeration
template <Precision kPrecision, Shape kShape, Device kDevice>
constexpr bool KeysAreEnumerators() noexcept {
    return static_cast<std::size_t>(kPrecision) <=
               static_cast<std::size_t>(Precision::kBf16Device) &&
           static_cast<std::size_t>(kShape) <= static_cast<std::size_t>(Shape::kAllNAtOrders) &&
           static_cast<std::size_t>(kDevice) <= static_cast<std::size_t>(Device::kDevice);
}

/// The class's bound is never refused; a class figure is.
///
/// The route in force is the one fact every figure here is keyed by, and where
/// the build has not stated it there is no figure to answer with: this is the
/// refusal, made where the class is named so that a caller who never asks for a
/// figure - a program reporting its own build, say - compiles and answers.
template <Precision kLane> constexpr void RequireRecordedArithmetic() noexcept {
    static_assert(kRouteInForceKnownHere,
                  "boys: this build's arithmetic is outside the records, so no figure applies. "
                  "Detected: BOYS_MULADD_SEPARATE selects the separate multiply-add route, and "
                  "BOYS_SCALAR_CONTRACTS is not defined, so whether a bare product-plus-add is "
                  "one rounding here has not been measured - and a contracting build compiles "
                  "the separate selection onto the fused step, so the route in force is not a "
                  "fact this translation unit carries. The records name the fused route at both "
                  "contraction readings and the separate route where the bare form is two "
                  "roundings, and this build has said none of the three. Define "
                  "BOYS_SCALAR_CONTRACTS to 0 or 1 as this build's own measurement answers, or "
                  "link boys::boys or boys::boys-tier, which pin the answer the configure "
                  "measured. The bound is not affected - it holds at every arithmetic, and "
                  "BoysLaneFiguresHere states it - and ArithmeticInForceHere names the "
                  "arithmetic that was detected.");
}

/// The figure this build's arithmetic was measured to deliver for one class.
///
/// The rows are keyed by the route in force and by the measured contraction
/// answer, and the class picks one of the three figures its row carries: the
/// double lane's single entry - \c Shape::kAllOrders is one argument and reads
/// it as well - the float lane's single entry, or the float lane's batch entry.
///
/// \c kDefaultCell is whether the caller named the class's own default policy.
/// A call that moved the division form or the region-B member names an entry the
/// sweeps were not taken at, and its delivered figure is its own measurement
/// rather than the cell's; the bound is stated there and no delivered figure is.
///
/// \returns the class's figure, or one whose \c reason says why none is delivered
template <Precision kPrecision, Shape kShape, bool kDefaultCell>
constexpr AccuracyFigure DeliveredFigureHere() noexcept {
    AccuracyFigure figure;
    figure.reading = AccuracyReading::kDelivered;

    if constexpr (!kRouteInForceKnownHere)
    {
        figure.reason = kUndeterminedArithmeticReason;
    } else if constexpr (!kDefaultCell)
    {
        figure.reason = kNonDefaultCellReason;
    } else if constexpr (kPrecision == Precision::kFp16 || kPrecision == Precision::kBf16 ||
                         kPrecision == Precision::kFp16Device ||
                         kPrecision == Precision::kBf16Device)
    {
        figure.reason = kHalfLaneReason;
    } else if constexpr (kPrecision == Precision::kFp32Device ||
                         kPrecision == Precision::kFp64Device)
    {
        figure.reason = kDeviceLaneReason;
    } else if constexpr (kPrecision == Precision::kFp64 && kShape != Shape::kSingle &&
                         kShape != Shape::kAllOrders)
    {
        figure.reason = kDoubleBatchReason;
    } else
    {
        // One route and one contraction reading picked the column. An unmeasured contraction
        // was answered at the not-contracted column, the reading that claims the least, and
        // its record says so rather than leaving the answer silently different.
        const DeliveredRow& row =
            (kRouteInForceHere == backend::MulAddRoute::kSeparate)
                ? kDeliveredSeparate
                : (kContractionContractsHere ? kDeliveredContracted : kDeliveredFusedTwoRoundings);

        figure.available = true;
        figure.source = kContractionMeasuredHere ? row.record : kDeliveredUnmeasuredRecord;

        if constexpr (kPrecision == Precision::kFp64)
        {
            figure.value = row.doubleBand;
        } else if constexpr (kShape == Shape::kSingle || kShape == Shape::kAllOrders)
        {
            figure.value = row.floatSingle;
        } else
        {
            figure.value = row.floatBatch;
        }
    }

    return figure;
}

/// What the delivered figure covers: the region of the argument range the
/// record's row names, empty where no figure is stated.
///
/// \returns the region as text, and \c "" where no figure is stated
template <Precision kPrecision, Shape kShape, bool kDefaultCell>
constexpr const char* DeliveredScopeHere() noexcept {
    if constexpr (!kRouteInForceKnownHere || !kDefaultCell || kPrecision == Precision::kFp16 ||
                  kPrecision == Precision::kBf16 || kPrecision == Precision::kFp16Device ||
                  kPrecision == Precision::kBf16Device || kPrecision == Precision::kFp32Device ||
                  kPrecision == Precision::kFp64Device ||
                  (kPrecision == Precision::kFp64 && kShape != Shape::kSingle &&
                   kShape != Shape::kAllOrders))
    {
        return "";
    } else if constexpr (kPrecision == Precision::kFp64)
    {
        return "the band 1.0855252345349333 <= x < 11.899848152108484, the one region of the "
               "double lane's single entry the route moves";
    } else
    {
        return "all arguments";
    }
}

} // namespace detail

/// The eight lanes' figures, one row per lane, in enumerator order.
///
/// The rows are the ones `BoysLaneContracts()` states, restated here so that a
/// build with no archive reads them from the headers alone; a unit that links
/// the archive and asks both reads one number twice, and the two are meant to
/// agree - a figure this header states that the archive's list does not is a
/// defect in one of them. Where the two documents disagree, the archive's list
/// is the one a program linked against this library is answered from, and the
/// records say which page each row is owed to.
///
/// The table is a pure statement and asks nothing of the arithmetic: every row
/// holds at every route and at both contraction readings, which is what a bound
/// is for, so this accessor compiles and answers on a build whose arithmetic is
/// outside the records and cannot ask for a delivered figure.
///
/// \returns the rows, valid for the life of the program
///
/// \ingroup boys
constexpr std::span<const LaneFigureHere> BoysLaneFiguresHere() noexcept {
    return detail::kLaneFiguresHere;
}

/// What this build answers for one default-policy class.
///
/// \ingroup boys
struct ClassAccuracyHere {
    /// The figure this build states for the class: the lane's row with the terms
    /// the division form and region-B exponential this call names add - the
    /// class's own by default. It is the lane's measured absolute figure and not
    /// `BoysAccuracyGuaranteed`'s answer, which is the contract evaluated at the
    /// cell it is asked about.
    AccuracyFigure bound;

    /// The figure this build's arithmetic was measured to deliver for the class:
    /// the figure `BoysAccuracyDelivered` answers with, from the records this
    /// header quotes. Unavailable, with the reason, where the records state no
    /// figure for the class or for its arithmetic.
    AccuracyFigure delivered;

    /// What the delivered figure covers: the region of the argument range the
    /// record's row names, empty where no figure is stated. A delivered figure is
    /// a swept maximum over that region and not a bound over the whole domain,
    /// and the region is part of what it says.
    const char* deliveredScope = "";

    /// Whether the class's seam row is a run's own winner or a stated choice.
    RowBasis basis = RowBasis::kChosen;

    /// The record the class's seam row was taken from - the name of the run
    /// whose file carries the revision it was measured at - and empty where no
    /// run stands behind the row and its cells are this build's own choice.
    const char* rowRecord = "";
};

/// What this build delivers for one default-policy class: the bound the class's
/// policy carries and the figure its arithmetic was measured to deliver, each
/// with the record it is read from.
///
/// The class is a (precision, shape, device) triple, the same key the
/// default-policy table is read by, and the figures are the class's own: the
/// eight lanes this library carries have eight rows, and the class's row of
/// `DefaultPolicyFor` supplies the terms the figure is read at. The bound is the
/// measured absolute figure this build states for the class, and not the
/// archive's `DefaultGuarantee(n, x)`, which evaluates the contract at a cell:
/// this header's table has no function value to scale by.
///
/// Two of the seven policy axes move the bound, and both are parameters here:
/// the division form, because the plain reciprocal rounds once more per step and
/// the lanes where that costs accuracy state a term beside the base, and the
/// region-B exponential, because a lane whose row states a term under one member
/// is not owed it under the other. Both default to the class's own row, so a
/// call that names none reads the default cell. **The other five axes do not move
/// a figure**: the rows state one bound for every fit route, evaluation scheme,
/// budget, packing axis and partition the option space carries, which is what a
/// bound is for, and the delivered rows are keyed by the multiply-add route in
/// force and not by the fit route a build selects.
///
/// \c kDevice defaults to \c Device::kHost, which is what a name that does not
/// spell a device resolves on; a device class is answered at its lane's own row
/// and its own default policy.
///
/// **A build whose route in force is not determined does not compile a call of
/// this function.** That is the one arithmetic no record holds a delivered
/// figure for, and the refusal is a diagnostic that names what was detected
/// rather than a number nobody measured. The bound is not affected: it holds at
/// every route, and `BoysLaneFiguresHere` states it.
///
/// \tparam kPrecision the precision lane
/// \tparam kShape     the question shape
/// \tparam kDevice    which side of the interface the class is on; \c Device::kHost
///                    where a caller does not spell one
/// \tparam kForm      the division form to answer at; the class's own default
/// \tparam kExp       the region-B exponential to answer at; the class's own default
/// \returns the figures, and whether the records carry them
///
/// \ingroup boys
template <Precision kPrecision,
          Shape kShape,
          Device kDevice = Device::kHost,
          DivisionForm kForm = DefaultPolicy<kPrecision, kShape, kDevice>::kDivision,
          RegionBExp kExp = DefaultPolicy<kPrecision, kShape, kDevice>::kRegionBExp>
constexpr ClassAccuracyHere BoysAccuracyHere() noexcept {
    static_assert(detail::KeysAreEnumerators<kPrecision, kShape, kDevice>(),
                  "the class named is not a (precision, shape, device) triple this library "
                  "carries: one key is a value outside its enumeration, and a value outside an "
                  "enumeration names no class on any lane. Name each key from its own "
                  "enumerators");

    detail::RequireRecordedArithmetic<kPrecision>();

    const LaneFigureHere& lane = detail::kLaneFiguresHere[static_cast<std::size_t>(kPrecision)];
    // Whether the caller named the class's own policy. The delivered rows are the sweeps of
    // this cell and not of another, so the two axes that move a bound and not a sweep are
    // also what tells the answer whether a delivered figure is the cell's at all.
    constexpr bool kDefaultCell = kForm == DefaultPolicy<kPrecision, kShape, kDevice>::kDivision &&
                                  kExp == DefaultPolicy<kPrecision, kShape, kDevice>::kRegionBExp;

    ClassAccuracyHere answer;
    answer.bound.reading = AccuracyReading::kGuaranteed;

    // The plain reciprocal rounds once more per step than the other two forms, so on a lane
    // where that costs accuracy the row publishes a term beside its base and this form's
    // figure is the base plus it. The record moves with the figure for the same reason.
    const bool plain = kForm == DivisionForm::kPlainReciprocal;
    const double formTerm = plain ? lane.plainAdditive : 0.0;

    // Which exponential seeds a region-B ladder is an axis and not a spelling of the
    // combination: a row states the member its term beside the base is under, so a class
    // naming the other member is not owed that term.
    const double memberTerm = kExp == lane.additiveMember ? lane.additive : 0.0;

    answer.bound.value = lane.bound + formTerm + memberTerm;
    answer.bound.available = true;
    answer.bound.source = (plain && lane.plainRecord[0] != '\0') ? lane.plainRecord : lane.record;

    answer.delivered = detail::DeliveredFigureHere<kPrecision, kShape, kDefaultCell>();
    answer.deliveredScope = detail::DeliveredScopeHere<kPrecision, kShape, kDefaultCell>();
    answer.basis = DefaultPolicyFor<kPrecision, kShape, kDevice>::kBasis;
    answer.rowRecord = DefaultPolicyFor<kPrecision, kShape, kDevice>::kRecord;

    return answer;
}

} // namespace boys
