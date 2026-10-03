// The build-defaults seam's own check, the compile half: every choice the seam
// header offers is one the library reads, read as the value an entry that names
// no policy resolves to.
//
// The defect this file exists for is on the record. include/boys/
// boys_build_defaults.hpp names five choices, a build may replace it
// (BOYS_BUILD_DEFAULTS), and until the wiring three of the five - the packing
// axis, the division form and the fit granularity - were expanded by nothing at
// all: the library compiled hard-coded literals, a consumer who replaced the
// header got two choices honoured and three silently ignored, and no test
// noticed, because the fixture moved one choice and copied the library's own
// values for the other four.
//
// TWO READINGS, AND THIS IS THE SECOND ONE
//
// A macro that stops being expanded is a text fact, and tools/check_seam_macros.py
// is what reads it. What a text check cannot see is the half this file is:
// whether the value a header names is the value the entries resolve to. Both are
// needed because they fail differently - the text one fires when a reader is
// deleted, this one when a reader stops being reached.
//
// HOW THIS FAILS
//
// The five macros are expanded here as values, so each comparison below is
// between the value the seam in force names and the value this build's unnamed
// policy carries:
//
//   - the macro is dropped from its reader and the constant that owned it holds
//     a literal: the constant stops equalling the seam's value, and the first
//     block below fails;
//   - the constant still reads the macro and a policy default stops using the
//     constant: EvalPolicy<> or one of the four DefaultPolicy* names stops
//     carrying it, and the second or the third block fails.
//
// Every one of them is a static_assert, so the failure is the configure not
// compiling rather than a program that has to be run to say so - and a macro
// that stops being read is exactly the change no suite was looking at.
//
// WHAT IT CANNOT SEE, so that "this file passes" is read for what it is:
//
//   - a replacement naming the SHIPPED value is, by value, indistinguishable
//     from a macro nothing expands. The comparisons are equalities and they hold
//     in the committed configuration by construction, so this file has teeth
//     only in a build whose seam moves a choice - which is what
//     tests/build_defaults_tuned.hpp is for: it moves a choice on every axis a
//     build can move, except the fit route, which tests/boys_backend_test.cpp
//     pins at the shipped value for a header defining its fixture guard. The
//     configure that points BOYS_BUILD_DEFAULTS at that file is where this check
//     fails when a reader is broken, and it is not only a local one: the
//     `Build defaults (tuned fixture built and tested)` step of
//     `.github/workflows/ci.yml` points the option at that file on the
//     linux-x86 gcc Release leg and runs the suite there, so the loud failure is
//     one CI reports as well as one a person can run.
//   - one of the members the five axes offer cannot be named by a build at all,
//     so no check of this shape can have teeth on it: PackAxis::kOrders, which
//     the single-order entry refuses because a call that produces one order has
//     no second order to put in a vector lane (include/boys/boys_impl.hpp,
//     BoysSingleImpl). FitGranularity::kUniform was the second until this
//     revision: the batched bodies that refused it now hand a policy naming the
//     grid to the path that reads it, and the accuracy gate's entry book measures
//     their six cells (tests/boys_accuracy_gate.cpp). The tuned fixture leaves the
//     packing axis at the shipped value and quotes that refusal; no fixture in
//     this tree names the uniform member, so what a build naming it compiles to is
//     not measured here.
//   - whether the bodies use the policy they are handed is not a text or a type
//     fact: a body reading another partition's table under this policy's name is
//     the accuracy gate's business (tests/boys_accuracy_gate.cpp), not this
//     file's.
//   - the enumeration of what the seam offers is the text check's and not this
//     file's: this file names the five choices an unnamed call resolves to, and a
//     sixth macro added to the seam header is read by tools/check_seam_macros.py
//     (which reads the names off that file) rather than by a list here that would
//     go stale.
//
// WHERE IT IS BUILT
//
// It belongs in the boys-tests source list beside tests/boys_backend_test.cpp:
//
//   add_executable(boys-tests ... tests/boys_build_defaults_test.cpp)

#include "boys/backend.hpp"

#include "boys/boys.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>
#include <type_traits>

namespace {

// The five seam macros expand to enumerator names written as the library's own
// headers write them - FitRoute::kChebyshev and its four siblings - and the
// library's headers write them inside namespace boys. A translation unit that
// reads the macros therefore has to be in that namespace or import it, which is
// what this using-directive is for and the only reason it is here.
using namespace boys;

// The five values the seam in force names. Each macro expands to the enumerator a
// replacement writes, so these are the build's choices as values and not as text:
// the include path delivers the replacement in a replaced build, and the
// committed header in every other one.
constexpr boys::FitRoute kSeamFitRoute = BOYS_BUILD_DEFAULT_FIT_ROUTE;
constexpr boys::EvalScheme kSeamEvalScheme = BOYS_BUILD_DEFAULT_EVAL_SCHEME;
constexpr boys::PackAxis kSeamPackAxis = BOYS_BUILD_DEFAULT_PACK_AXIS;
constexpr boys::DivisionForm kSeamDivisionForm = BOYS_BUILD_DEFAULT_DIVISION_FORM;
constexpr boys::FitGranularity kSeamFitGranularity = BOYS_BUILD_DEFAULT_FIT_GRANULARITY;

// The policy the seam's own five values compose: the point a class the table carries
// no row for resolves to, and the type EvalPolicy<> names. It is NOT what every entry
// that names no policy resolves to - an entry resolves to its class's row, which is
// this combination only where that row spells it (include/boys/boys.hpp expands the
// row list or the five, never both), and the unnamed-call test at the end of this file
// is where that is read. Naming the values here is what makes the assertions below
// claims about the build's header and not about this line.
using SeamPolicy = boys::EvalPolicy<BOYS_BUILD_DEFAULT_FIT_ROUTE,
                                    BOYS_BUILD_DEFAULT_EVAL_SCHEME,
                                    boys::BoysBudget::kFloat,
                                    BOYS_BUILD_DEFAULT_PACK_AXIS,
                                    BOYS_BUILD_DEFAULT_FIT_GRANULARITY,
                                    BOYS_BUILD_DEFAULT_DIVISION_FORM>;

// --- Link one: the constant that owns a macro reads it ------------------------
// Each of the five constants is documented beside the enumeration it belongs to
// and is the name the entries' template defaults read. A constant that stops
// expanding its macro is the shape that sat here before the wiring: the macro
// stays in the seam file, documented, and the library compiles a literal that
// the committed header happens to agree with.
static_assert(kSeamFitRoute == boys::kDefaultFitRoute,
              "the seam names a fit route the library does not read: boys::kDefaultFitRoute is not "
              "BOYS_BUILD_DEFAULT_FIT_ROUTE, so a build replacing the seam would be told the "
              "choice was taken and would compile another route");
static_assert(kSeamEvalScheme == boys::kDefaultEvalScheme,
              "the seam names an evaluation scheme the library does not read: "
              "boys::kDefaultEvalScheme is not BOYS_BUILD_DEFAULT_EVAL_SCHEME");
static_assert(kSeamPackAxis == boys::kDefaultPackAxis,
              "the seam names a packing axis the library does not read: boys::kDefaultPackAxis is "
              "not BOYS_BUILD_DEFAULT_PACK_AXIS");
static_assert(kSeamDivisionForm == boys::kDefaultDivisionForm,
              "the seam names a division form the library does not read: "
              "boys::kDefaultDivisionForm is not BOYS_BUILD_DEFAULT_DIVISION_FORM");
static_assert(kSeamFitGranularity == boys::kDefaultFitGranularity,
              "the seam names a fit granularity the library does not read: "
              "boys::kDefaultFitGranularity is not BOYS_BUILD_DEFAULT_FIT_GRANULARITY");

// --- Link two: the policy the five constants compose --------------------------
// EvalPolicy's template defaults are the five constants, so EvalPolicy<> is the
// combination the seam names and the point a class the table carries no row for
// resolves to. A default that stopped reading the constants would keep every
// constant correct and answer a caller from another policy - the same silent
// substitution one step further down. What an entry that names no policy resolves
// to is its class's row rather than this name, which is the last test below's
// claim rather than this block's.
static_assert(kSeamFitRoute == boys::EvalPolicy<>::kRoute,
              "an entry that names no fit route does not resolve to the one the seam names: "
              "EvalPolicy<>'s default route reads another value");
static_assert(kSeamEvalScheme == boys::EvalPolicy<>::kScheme,
              "an entry that names no evaluation scheme does not resolve to the one the seam "
              "names: EvalPolicy<>'s default scheme reads another value");
static_assert(kSeamPackAxis == boys::EvalPolicy<>::kPack,
              "an entry that names no packing axis does not resolve to the one the seam names: "
              "EvalPolicy<>'s default axis reads another value");
static_assert(kSeamDivisionForm == boys::EvalPolicy<>::kDivision,
              "an entry that names no division form does not resolve to the one the seam names: "
              "EvalPolicy<>'s default form reads another value");
static_assert(kSeamFitGranularity == boys::EvalPolicy<>::kGranularity,
              "an entry that names no fit granularity does not resolve to the one the seam names: "
              "EvalPolicy<>'s default partition reads another value");
static_assert(std::is_same_v<boys::EvalPolicy<>, SeamPolicy>,
              "EvalPolicy<> does not compose the seam's own five values");

// --- Link three: the four names a precision is selected by --------------------
// Each name is what the entries of that precision run when the call site names no
// policy, so each carries the seam's five. The budget is not one of the five and
// is not compared: it is the library's, and it is the one field the fp16 and bf16
// names move.
static_assert(boys::DefaultPolicyFp64::kRoute == kSeamFitRoute &&
                  boys::DefaultPolicyFp64::kScheme == kSeamEvalScheme &&
                  boys::DefaultPolicyFp64::kPack == kSeamPackAxis &&
                  boys::DefaultPolicyFp64::kGranularity == kSeamFitGranularity &&
                  boys::DefaultPolicyFp64::kDivision == kSeamDivisionForm,
              "DefaultPolicyFp64 does not carry the seam's five values");
static_assert(boys::DefaultPolicyFp32::kRoute == kSeamFitRoute &&
                  boys::DefaultPolicyFp32::kScheme == kSeamEvalScheme &&
                  boys::DefaultPolicyFp32::kPack == kSeamPackAxis &&
                  boys::DefaultPolicyFp32::kGranularity == kSeamFitGranularity &&
                  boys::DefaultPolicyFp32::kDivision == kSeamDivisionForm,
              "DefaultPolicyFp32 does not carry the seam's five values");
static_assert(boys::DefaultPolicyFp16::kRoute == kSeamFitRoute &&
                  boys::DefaultPolicyFp16::kScheme == kSeamEvalScheme &&
                  boys::DefaultPolicyFp16::kPack == kSeamPackAxis &&
                  boys::DefaultPolicyFp16::kGranularity == kSeamFitGranularity &&
                  boys::DefaultPolicyFp16::kDivision == kSeamDivisionForm,
              "DefaultPolicyFp16 does not carry the seam's five values");
static_assert(boys::DefaultPolicyBf16::kRoute == kSeamFitRoute &&
                  boys::DefaultPolicyBf16::kScheme == kSeamEvalScheme &&
                  boys::DefaultPolicyBf16::kPack == kSeamPackAxis &&
                  boys::DefaultPolicyBf16::kGranularity == kSeamFitGranularity &&
                  boys::DefaultPolicyBf16::kDivision == kSeamDivisionForm,
              "DefaultPolicyBf16 does not carry the seam's five values");

// Which of the two the file in force is: the committed header defines
// BOYS_BUILD_DEFAULTS_SHIPPED, a replacement does not, and the CMake option puts
// BOYS_BUILD_DEFAULTS_REPLACED on the command line of every unit of a build that
// used it (include/boys/boys_build_defaults.hpp states the contract).
constexpr const char* SeamInForce() {
#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
    return "the committed header (the shipped choices)";
#elif defined(BOYS_BUILD_DEFAULTS_REPLACED)
    return "a replacement header (BOYS_BUILD_DEFAULTS)";
#else
    return "a header that defines neither guard: a replacement not delivered through the option";
#endif
}

// The name a report prints the route under, taken from the library's own table
// rather than from a word written here: the rows are the report surface, and a
// name this file spelled itself would be a second answer to one question.
const char* RouteName(boys::FitRoute route) {
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes()) {
        if (row.route == route) {
            return row.name;
        }
    }

    return "unknown";
}

std::string Cells(bool same, const char* seam, const char* constant, const char* resolved) {
    return std::string(seam) + " / " + constant + " / " + resolved + (same ? "  agree" : "  DISAGREE");
}

} // namespace

// The same equality the static asserts above hold at compile time, printed with
// the names a report uses rather than as a checkmark: a pass here says which five
// values this build resolves, and a failure says which axis and which of the
// three readings disagreed.
TEST(BuildDefaultsTest, TheSeamNamesAreTheValuesThisBuildResolves) {
    std::printf("boys: build defaults in force = %s\n", SeamInForce());
    std::printf("boys: %-16s %-18s %-18s %-18s\n",
                "axis",
                "the seam names",
                "the constant holds",
                "an unnamed call resolves");

    const bool routesAgree = kSeamFitRoute == boys::kDefaultFitRoute &&
                             kSeamFitRoute == boys::EvalPolicy<>::kRoute;
    const bool schemesAgree = kSeamEvalScheme == boys::kDefaultEvalScheme &&
                              kSeamEvalScheme == boys::EvalPolicy<>::kScheme;
    const bool packsAgree =
        kSeamPackAxis == boys::kDefaultPackAxis && kSeamPackAxis == boys::EvalPolicy<>::kPack;
    const bool formsAgree = kSeamDivisionForm == boys::kDefaultDivisionForm &&
                            kSeamDivisionForm == boys::EvalPolicy<>::kDivision;
    const bool granularitiesAgree = kSeamFitGranularity == boys::kDefaultFitGranularity &&
                                    kSeamFitGranularity == boys::EvalPolicy<>::kGranularity;

    std::printf("  %-16s %s\n",
                "fit route",
                Cells(routesAgree,
                      RouteName(kSeamFitRoute),
                      RouteName(boys::kDefaultFitRoute),
                      RouteName(boys::EvalPolicy<>::kRoute))
                    .c_str());
    std::printf("  %-16s %s\n",
                "eval scheme",
                Cells(schemesAgree,
                      boys::EvalSchemeName(kSeamEvalScheme),
                      boys::EvalSchemeName(boys::kDefaultEvalScheme),
                      boys::EvalSchemeName(boys::EvalPolicy<>::kScheme))
                    .c_str());
    std::printf("  %-16s %s\n",
                "packing axis",
                Cells(packsAgree,
                      boys::PackAxisName(kSeamPackAxis),
                      boys::PackAxisName(boys::kDefaultPackAxis),
                      boys::PackAxisName(boys::EvalPolicy<>::kPack))
                    .c_str());
    std::printf("  %-16s %s\n",
                "division form",
                Cells(formsAgree,
                      boys::DivisionFormName(kSeamDivisionForm),
                      boys::DivisionFormName(boys::kDefaultDivisionForm),
                      boys::DivisionFormName(boys::EvalPolicy<>::kDivision))
                    .c_str());
    std::printf("  %-16s %s\n",
                "fit granularity",
                Cells(granularitiesAgree,
                      boys::GranularityName(kSeamFitGranularity),
                      boys::GranularityName(boys::kDefaultFitGranularity),
                      boys::GranularityName(boys::EvalPolicy<>::kGranularity))
                    .c_str());

    EXPECT_TRUE(routesAgree);
    EXPECT_TRUE(schemesAgree);
    EXPECT_TRUE(packsAgree);
    EXPECT_TRUE(formsAgree);
    EXPECT_TRUE(granularitiesAgree);
}

// The entry, not only the policy type: the static asserts above hold EvalPolicy<> to the
// seam, and this holds an entry to the policy its own CLASS resolves to. That is a
// different claim - an entry whose default argument stopped being that alias would keep
// every assert above true and answer a caller from another policy - and it is made at the
// call a caller writes, so the unnamed call and the class default's call have to return the
// same bits at every argument and order this sweep reaches.
//
// THE CLASS DEFAULT IS THE TABLE'S ROW, WHICH IS THE SEAM'S FIVE ONLY WHERE THE ROW SPELLS
// THEM. A replacement that defines BOYS_BUILD_DEFAULT_ROWS is read INSTEAD of the
// five-composed table (include/boys/boys.hpp expands one branch or the other, never both),
// so where its list carries a row for a class that row is the answer for that class and the
// seam's five are not. The sweep below reads which of the two this build's table holds for
// each entry's class and asserts the entry follows it: the class default everywhere, and
// the seam's five exactly where the row that carries the class spells them. A build whose
// row moves the class is therefore the configuration in which the second half has teeth,
// and the configuration a check written against the five alone answers wrongly.
TEST(BuildDefaultsTest, AnUnnamedCallIsItsClasssDefault) {
    using boys::BoysAllOrders;
    using boys::BoysSingle;

    // The class each entry belongs to, as the policy it resolves to. Named through the
    // library's own alias rather than spelled here: this is the name an entry that names no
    // policy resolves to (boys/boys.hpp, DefaultPolicy), so naming it is what makes the
    // comparison a claim about the build's table and not about this line.
    using OrdersClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;
    using SingleClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>;

    // Which of the two the table answers each class with. Read from the class's own policy,
    // so it is the build's answer and not this line's. The budget is not one of the five and
    // is compared as the fp64 lane's: the seam's five name no budget.
    constexpr bool kOrdersClassIsTheSeamFive =
        OrdersClass::kRoute == kSeamFitRoute && OrdersClass::kScheme == kSeamEvalScheme &&
        OrdersClass::kPack == kSeamPackAxis &&
        OrdersClass::kGranularity == kSeamFitGranularity &&
        OrdersClass::kDivision == kSeamDivisionForm &&
        OrdersClass::kBudget == boys::BoysBudget::kFloat;
    constexpr bool kSingleClassIsTheSeamFive =
        SingleClass::kRoute == kSeamFitRoute && SingleClass::kScheme == kSeamEvalScheme &&
        SingleClass::kPack == kSeamPackAxis &&
        SingleClass::kGranularity == kSeamFitGranularity &&
        SingleClass::kDivision == kSeamDivisionForm &&
        SingleClass::kBudget == boys::BoysBudget::kFloat;

    constexpr int kOrders[] = {0, 1, 4, 12, boys::kMaxBoysOrder};
    constexpr double kArguments[] = {0.0, 1e-12, 1e-3, 0.5, 1.0, 3.0, 11.9, 12.0, 60.0, 120.0};

    std::size_t batchClassMoved = 0;
    std::size_t batchSeamMoved = 0;
    std::size_t batchCells = 0;
    std::size_t singleClassMoved = 0;
    std::size_t singleSeamMoved = 0;
    std::size_t singleCells = 0;

    for (const int nmax : kOrders) {
        for (const double x : kArguments) {
            std::array<double, boys::kMaxBoysOrder + 1> unnamed{};
            std::array<double, boys::kMaxBoysOrder + 1> classDefault{};
            std::array<double, boys::kMaxBoysOrder + 1> seam{};

            boys::BoysAllOrders(nmax, x, unnamed.data());
            boys::BoysAllOrders<OrdersClass>(nmax, x, classDefault.data());
            boys::BoysAllOrders<SeamPolicy>(nmax, x, seam.data());

            for (int order = 0; order <= nmax; ++order) {
                const std::size_t sorder = static_cast<std::size_t>(order);

                ++batchCells;

                if (std::bit_cast<std::uint64_t>(unnamed[sorder]) !=
                    std::bit_cast<std::uint64_t>(classDefault[sorder])) {
                    ++batchClassMoved;
                }

                if (std::bit_cast<std::uint64_t>(unnamed[sorder]) !=
                    std::bit_cast<std::uint64_t>(seam[sorder])) {
                    ++batchSeamMoved;
                }
            }

            ++singleCells;

            if (boys::BoysSingle(4, x) != boys::BoysSingle<SingleClass>(4, x)) {
                ++singleClassMoved;
            }

            if (boys::BoysSingle(4, x) != boys::BoysSingle<SeamPolicy>(4, x)) {
                ++singleSeamMoved;
            }
        }
    }

    std::printf("boys: the fp64 all-orders class's row is %s the seam's five, and the fp64 "
                "single class's row is %s them\n",
                kOrdersClassIsTheSeamFive ? "the same as" : "not",
                kSingleClassIsTheSeamFive ? "the same as" : "not");
    std::printf("boys: the unnamed call differs from its class default's call in %zu of %zu batch "
                "values and %zu of %zu single values, and from the seam policy's call in %zu and "
                "%zu of them\n",
                batchClassMoved,
                batchCells,
                singleClassMoved,
                singleCells,
                batchSeamMoved,
                singleSeamMoved);

    EXPECT_EQ(batchClassMoved, 0u) << "an entry that names no policy does not resolve to its own "
                                      "class's default policy, so a caller that names none is "
                                      "answered by a policy the build's table does not name for "
                                      "that class";
    EXPECT_EQ(singleClassMoved, 0u) << "the single-order entry that names no policy does not "
                                       "resolve to its own class's default policy";

    // The table's own precedence, at the call a caller writes: the unnamed call equals the
    // seam's five's call exactly where the row that carries the class spells the five. An
    // entry resolving through the five where the table's row moves the class agrees here
    // when it must not, and one resolving through the row where the row is the five agrees
    // when it must.
    EXPECT_EQ(batchSeamMoved == 0u, kOrdersClassIsTheSeamFive)
        << "the unnamed call and the seam policy's call differ where the table's row for the "
           "class spells the seam's five, or agree where the row moves the class: the entry is "
           "not resolving through the row the build's table carries for it";
    EXPECT_EQ(singleSeamMoved == 0u, kSingleClassIsTheSeamFive)
        << "the single-order entry resolves through the five rather than through the row the "
           "build's table carries for its class, or the other way round";
}
