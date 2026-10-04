// The build-defaults seam's own check, the compile half: every choice the seam
// header offers is one the library reads, read as the value an entry that names
// no policy resolves to.
//
// The defect this file exists for is on the record. include/boys/
// boys_build_defaults.hpp named five choices then - seven now, the host's five
// and the device lane's two - a build may replace it (BOYS_BUILD_DEFAULTS), and
// until the wiring three of the five - the packing axis, the division form and
// the fit granularity - were expanded by nothing at
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
// AND A THIRD READING, OF WHAT THE BUILD SAYS IT IS. The choices above are read
// as values, and a build that replaced the seam could say none of them: nothing a
// consumer could reach reported that the seam had been replaced, or by which file,
// so a consumer whose regression baseline moved had no way to see that the seam is
// why. The library reports the seam's identity now
// (include/boys/version.hpp, BuildDefaultsSeamIdentity()), and the test at the end
// of this file holds that report to the file the configure named - the one claim
// here that a consumed library, and not a source tree, is the thing making.
//
// HOW THIS FAILS
//
// The seven macros are expanded here as values, so each comparison below is
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
//     file's: this file names the seven choices an unnamed call resolves to - the
//     host lane's five and the device lane's two, which are held by the same
//     comparisons because the device lane reads its own names and not the host's -
//     and an eighth macro added to the seam header is read by
//     tools/check_seam_macros.py (which reads the names off that file) rather than
//     by a list here that would go stale.
//
// WHERE IT IS BUILT
//
// It belongs in the boys-tests source list beside tests/boys_backend_test.cpp:
//
//   add_executable(boys-tests ... tests/boys_build_defaults_test.cpp)

#include "boys/backend.hpp"

#include "boys/boys.hpp"

// The device lane's region-B exponential is declared beside the device tables it
// belongs to, and boys.hpp does not carry that header: the device seam name is read
// where the tables are.
#include "boys/boys_device_tables.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <iterator>
#include <string>
#include <type_traits>
#include <vector>

// Supplied at configure time: the seam file this build was configured with, which
// the identity test at the end of this file hashes. A translation unit compiled by
// hand, outside this build system, has no such file named and skips that check
// rather than hashing a path that names nothing.
#ifndef BOYS_DEFAULTS_SEAM_FILE
#define BOYS_DEFAULTS_SEAM_FILE ""
#endif

namespace {

// The five seam macros expand to enumerator names written as the library's own
// headers write them - FitRoute::kChebyshev and its four siblings - and the
// library's headers write them inside namespace boys. A translation unit that
// reads the macros therefore has to be in that namespace or import it, which is
// what this using-directive is for and the only reason it is here.
using namespace boys;

// The seven values the seam in force names: the host lane's five and the device
// lane's two. Each macro expands to the enumerator a replacement writes, so these
// are the build's choices as values and not as text: the include path delivers the
// replacement in a replaced build, and the committed header in every other one.
constexpr boys::FitRoute kSeamFitRoute = BOYS_BUILD_DEFAULT_FIT_ROUTE;
constexpr boys::EvalScheme kSeamEvalScheme = BOYS_BUILD_DEFAULT_EVAL_SCHEME;
constexpr boys::PackAxis kSeamPackAxis = BOYS_BUILD_DEFAULT_PACK_AXIS;
constexpr boys::DivisionForm kSeamDivisionForm = BOYS_BUILD_DEFAULT_DIVISION_FORM;
constexpr boys::FitGranularity kSeamFitGranularity = BOYS_BUILD_DEFAULT_FIT_GRANULARITY;
constexpr boys::DivisionForm kSeamDeviceDivisionForm = BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM;
constexpr boys::RegionBExp kSeamDeviceRegionBExp = BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP;

// The policy the seam's own five values compose: the point a class the table carries
// no row for resolves to, and the type EvalPolicy<> names. It is NOT what every entry
// that names no policy resolves to - an entry resolves to its class's row, which is
// this combination only where that row spells it (include/boys/boys.hpp expands the
// row list or the five, never both), and the unnamed-call test at the end of this file
// is where that is read. Naming the values here is what makes the assertions below
// claims about the build's header and not about this line.
//
// The region-B exponential is named as the library's own default rather than left to
// EvalPolicy's template default, and the seam carries no macro for it: the five macros
// above are the seam's, and this axis is one a ROW moves rather than one the five
// compose. Spelling it keeps the type the same as EvalPolicy<>'s - the default is this
// constant - and makes the axis a reader of this file can see, which is what the
// comparisons below need to tell a class whose row moved the exponential from one whose
// row spells the seam's five.
using SeamPolicy = boys::EvalPolicy<BOYS_BUILD_DEFAULT_FIT_ROUTE,
                                    BOYS_BUILD_DEFAULT_EVAL_SCHEME,
                                    boys::BoysBudget::kFloat,
                                    BOYS_BUILD_DEFAULT_PACK_AXIS,
                                    BOYS_BUILD_DEFAULT_FIT_GRANULARITY,
                                    BOYS_BUILD_DEFAULT_DIVISION_FORM,
                                    boys::kDefaultHostRegionBExp>;

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

// The device lane's two names, held the same way and for the same reason. They are
// separate from the five above because they are the device half of the seam: a
// device class resolves its form and its region-B exponential through these and
// never through the host's, so a build whose device half stopped reading its own
// names would take the host's values in silence - the shape that sat here before
// the device half existed, where an unnamed device call compiled the host macro's
// form and a literal region-B.
static_assert(kSeamDeviceDivisionForm == boys::kDefaultDeviceDivisionForm,
              "the seam names a device division form the library does not read: "
              "boys::kDefaultDeviceDivisionForm is not BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM");
static_assert(kSeamDeviceRegionBExp == boys::kDefaultRegionBExp,
              "the seam names a region-B exponential the library does not read: "
              "boys::kDefaultRegionBExp is not BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP");
static_assert(boys::kDefaultRegionBExp == boys::RegionBExp::kAccurate ||
                  boys::kDefaultRegionBExp == boys::RegionBExp::kFast,
              "boys::kDefaultRegionBExp is no enumerator of RegionBExp: the device lane's "
              "region-B exponential is read from BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP, and the "
              "two values it may name are the library routine and the reduced-argument "
              "polynomial");

// The value the committed header names, and only there: the device rows beside it are figures
// taken at the library routine, so the shipped configuration's own device exponential is that
// routine. A replacement states its own value and this pin is about what this repository
// publishes, not about what a consumer's card measured.
#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
static_assert(boys::kDefaultRegionBExp == boys::RegionBExp::kAccurate,
              "the committed header's device region-B exponential is not the one the device rows "
              "beside it were measured at: boys/boys_build_defaults.hpp states RegionBExp::kAccurate");
#endif

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

// --- Link four: the nine device classes, asked for by name --------------------
//
// A class is a (device, precision, shape) triple, and the device half of the table
// is nine of them: the three device lanes - the precisions a device entry is
// built at, which are lanes of Precision and not formats of one - by the three
// questions a device entry answers. Each of the nine is asked for here the way a
// caller asks: by the class's own default policy, which is the name an unnamed
// call resolves through (include/boys/boys.hpp, DefaultPolicyFor).
//
// THE ASK IS THE CHECK. The seam's rule for a class with no row is a static_assert
// in DefaultPolicyFor, and an assertion fires only where something asks for the
// class: a device row dropped from the list in force leaves one of the nine names
// below unanswered and this file stops compiling, with the seam's own message -
// "this build's default-policy table carries no row for this class". A list that
// stopped carrying one of the nine is what this link exists for, because a table
// whose device half went missing reads complete from the file it is written in.
//
// The two shapes the host lanes carry these beside - kFixedN and kAllNAtOrders -
// are not device classes and are asked for by no name here: the device option
// probe asks three questions, so no device class is keyed by the other two.
//
// In the committed configuration each of these resolves to the row the seam
// writes for that class, which is the row this file's Link four pins below; in a
// replacement the row is the replacement's, and the ask is the same ask.
using DeviceFp64Single = boys::DefaultPolicy<boys::Precision::kFp64Device, boys::Shape::kSingle,
                                              boys::Device::kDevice>;
using DeviceFp64Orders = boys::DefaultPolicy<boys::Precision::kFp64Device, boys::Shape::kAllOrders,
                                             boys::Device::kDevice>;
using DeviceFp64AllN = boys::DefaultPolicy<boys::Precision::kFp64Device, boys::Shape::kAllN,
                                           boys::Device::kDevice>;
using DeviceFp32Single = boys::DefaultPolicy<boys::Precision::kFp32Device, boys::Shape::kSingle,
                                             boys::Device::kDevice>;
using DeviceFp32Orders = boys::DefaultPolicy<boys::Precision::kFp32Device, boys::Shape::kAllOrders,
                                             boys::Device::kDevice>;
using DeviceFp32AllN = boys::DefaultPolicy<boys::Precision::kFp32Device, boys::Shape::kAllN,
                                           boys::Device::kDevice>;
using DeviceFp16Single = boys::DefaultPolicy<boys::Precision::kFp16Device, boys::Shape::kSingle,
                                             boys::Device::kDevice>;
using DeviceFp16Orders = boys::DefaultPolicy<boys::Precision::kFp16Device, boys::Shape::kAllOrders,
                                             boys::Device::kDevice>;
using DeviceFp16AllN = boys::DefaultPolicy<boys::Precision::kFp16Device, boys::Shape::kAllN,
                                           boys::Device::kDevice>;

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

// The budget cell, as a report names it. The library publishes no name function for
// BoysBudget - the enumeration has two members and the cell a row carries is the lane's
// own (detail::LaneFallbackBudget) - so the two names here are this file's reading of a
// cell the report prints rather than a second answer to what a lane resolves to.
const char* BudgetName(boys::BoysBudget budget) {
    switch (budget) {
    case boys::BoysBudget::kFloat:
        return "float";
    case boys::BoysBudget::kFp16:
        return "half";
    }

    return "unknown";
}

// One device class of the nine, as a report prints it: the class, and the row the name it was
// asked for by resolved to. The policy is a template parameter, so the row printed is the one
// the compiler selected for the class - a class this build carries no row for has no policy to
// pass here, and the ask is what refuses it rather than this printer.
template <typename Policy>
void PrintDeviceClass(const char* lane, const char* shape) {
    std::printf("  %-14s %-11s %-8s %-14s %-12s %-14s %-12s %-18s\n",
                lane,
                shape,
                BudgetName(Policy::kBudget),
                RouteName(Policy::kRoute),
                boys::EvalSchemeName(Policy::kScheme),
                boys::PackAxisName(Policy::kPack),
                boys::GranularityName(Policy::kGranularity),
                boys::DivisionFormName(Policy::kDivision));
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
    //
    // The region-B exponential is compared too, and it is the axis a row can move while
    // spelling the seam's five: a class whose row differs from `SeamPolicy` in that cell
    // alone is answered by another arithmetic, so a flag reading the five and stopping would
    // call it the seam's own combination and the assertions below would then require the two
    // calls to agree where the build's own table says they must not. Every axis of the policy
    // is read here, which is what this flag's name claims.
    constexpr bool kOrdersClassIsTheSeamFive =
        OrdersClass::kRoute == kSeamFitRoute && OrdersClass::kScheme == kSeamEvalScheme &&
        OrdersClass::kPack == kSeamPackAxis &&
        OrdersClass::kGranularity == kSeamFitGranularity &&
        OrdersClass::kDivision == kSeamDivisionForm &&
        OrdersClass::kRegionBExp == boys::kDefaultHostRegionBExp &&
        OrdersClass::kBudget == boys::BoysBudget::kFloat;
    constexpr bool kSingleClassIsTheSeamFive =
        SingleClass::kRoute == kSeamFitRoute && SingleClass::kScheme == kSeamEvalScheme &&
        SingleClass::kPack == kSeamPackAxis &&
        SingleClass::kGranularity == kSeamFitGranularity &&
        SingleClass::kDivision == kSeamDivisionForm &&
        SingleClass::kRegionBExp == boys::kDefaultHostRegionBExp &&
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

// THE SAME CLAIM, PER ENTRY, OVER EVERY ENTRY A MISSING ROW WOULD SILENTLY MOVE.
//
// The sweep above holds two entries to their class's row. This one holds the rest of
// the surface to it, because the claim is per entry and not per library: each of these
// is a separate declaration whose default template argument is that entry's own class
// alias, and any one of them stopped reading it answers a caller from a policy its
// class's row does not name.
//
// WHY IT IS HERE RATHER THAN IN THE ENTRY'S OWN SUITE. The suites that sweep these
// entries compare them AGAINST EACH OTHER - boys_all_n_test.cpp holds the all-n entry
// to the all-orders entry, boys_c_test.cpp holds the C entries to the C++ ones - and
// that comparison is a claim about two classes, which holds only where the two classes'
// rows agree. Naming a policy on both sides is what those sweeps have to do to state
// an implementation relationship that survives a replacement, and once they do, nothing
// in them reads the class table at all. This test is what still reads it.
//
// The entries below are the ones a shape-carried row can move apart, each measured
// against its own class: the rows are the question and an entry's default argument is
// the answer this file checks.
TEST(BuildDefaultsTest, EveryEntryResolvesThroughItsOwnClasssRow) {
    using AllOrdersF32Class = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>;
    using AllNF32Class = boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllN>;
    using AllNClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllN>;
    using AtOrdersClass =
        boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllNAtOrders>;

    constexpr int kOrders[] = {0, 3, boys::kMaxBoysOrder};
    constexpr int kCount = 4;
    constexpr double kXs[] = {0.0, 0.25, 5.0, 30.0};

    std::size_t allNMoved = 0;
    std::size_t atOrdersMoved = 0;
    std::size_t ordersF32Moved = 0;
    std::size_t allNF32Moved = 0;
    std::size_t cells = 0;

    for (const int nmax : kOrders) {
        const std::size_t span = static_cast<std::size_t>(nmax) + 1;
        std::vector<double> unnamed(kCount * span);
        std::vector<double> classDefault(kCount * span);

        boys::BoysAllN(nmax, kXs, unnamed.data(), kCount);
        boys::BoysAllN<AllNClass>(nmax, kXs, classDefault.data(), kCount);

        for (std::size_t i = 0; i < unnamed.size(); ++i) {
            ++cells;

            if (std::bit_cast<std::uint64_t>(unnamed[i]) !=
                std::bit_cast<std::uint64_t>(classDefault[i])) {
                ++allNMoved;
            }
        }

        // The per-argument tops entry: each argument's own top order, so the sweep's
        // tops stop at the argument's index and the two calls are asked the same shape.
        std::vector<int> tops(kCount);
        std::fill(tops.begin(), tops.end(), nmax);
        std::vector<double> viaUnnamed(kCount * span);
        std::vector<double> viaClass(kCount * span);

        boys::BoysAllNAtOrders(tops.data(), kXs, viaUnnamed.data(), kCount);
        boys::BoysAllNAtOrders<AtOrdersClass>(tops.data(), kXs, viaClass.data(), kCount);

        for (std::size_t i = 0; i < viaUnnamed.size(); ++i) {
            ++cells;

            if (std::bit_cast<std::uint64_t>(viaUnnamed[i]) !=
                std::bit_cast<std::uint64_t>(viaClass[i])) {
                ++atOrdersMoved;
            }
        }

        // The float lane's two shapes, bit-compared as floats: the fp32 classes are the
        // ones a per-class table separates from the fp64 ones, and their entries' class
        // aliases are not the ones the sweeps above name.
        std::vector<float> xsF32(kCount);
        std::vector<float> f32Unnamed(kCount * span);
        std::vector<float> f32Class(kCount * span);

        for (int i = 0; i < kCount; ++i) {
            xsF32[static_cast<std::size_t>(i)] = static_cast<float>(kXs[i]);
        }

        boys::BoysAllNF32(nmax, xsF32.data(), f32Unnamed.data(), kCount);
        boys::BoysAllNF32<AllNF32Class>(nmax, xsF32.data(), f32Class.data(), kCount);

        for (std::size_t i = 0; i < f32Unnamed.size(); ++i) {
            ++cells;

            if (std::bit_cast<std::uint32_t>(f32Unnamed[i]) !=
                std::bit_cast<std::uint32_t>(f32Class[i])) {
                ++allNF32Moved;
            }
        }

        // The single-argument float entry, one call per argument.
        for (int i = 0; i < kCount; ++i) {
            std::array<float, boys::kMaxBoysOrder + 1> unnamedRow{};
            std::array<float, boys::kMaxBoysOrder + 1> classRow{};

            boys::BoysAllOrdersF32(nmax, xsF32[static_cast<std::size_t>(i)], unnamedRow.data());
            boys::BoysAllOrdersF32<AllOrdersF32Class>(
                nmax, xsF32[static_cast<std::size_t>(i)], classRow.data());

            for (std::size_t k = 0; k < span; ++k) {
                ++cells;

                if (std::bit_cast<std::uint32_t>(unnamedRow[k]) !=
                    std::bit_cast<std::uint32_t>(classRow[k])) {
                    ++ordersF32Moved;
                }
            }
        }
    }

    std::printf("boys: an unnamed call against its own class's default, over %zu values: "
                "all-n %zu, all-n-at-orders %zu, float all-orders %zu, float all-n %zu moved\n",
                cells,
                allNMoved,
                atOrdersMoved,
                ordersF32Moved,
                allNF32Moved);

    EXPECT_EQ(allNMoved, 0u) << "the all-n entry that names no policy does not resolve to the "
                                "default policy the build's table carries for the fp64 all-n class";
    EXPECT_EQ(atOrdersMoved, 0u) << "the per-argument-tops entry that names no policy does not "
                                    "resolve to its own class's default policy";
    EXPECT_EQ(ordersF32Moved, 0u) << "the float all-orders entry that names no policy does not "
                                     "resolve to its own class's default policy";
    EXPECT_EQ(allNF32Moved, 0u) << "the float all-n entry that names no policy does not resolve to "
                                   "its own class's default policy";
}

// The device half printed: one row per class, holding the row that class resolves to. It is the
// run-time reading of Link four - the nine names are instantiated above whether or not this test
// is run, so what the print adds is which row each name resolved to rather than whether it
// resolved at all: a class the table carries no row for stops the build at Link four, and a class
// whose row moved prints the row it moved to.
TEST(BuildDefaultsTest, TheDeviceClassesPrintTheRowEachResolvesTo) {
    std::printf("boys: the device half of the table, one row per class, as this build resolves it\n");
    std::printf("boys: %-14s %-11s %-8s %-14s %-12s %-14s %-12s %-18s\n",
                "lane",
                "shape",
                "budget",
                "fit route",
                "scheme",
                "packing axis",
                "granularity",
                "division form");

    PrintDeviceClass<DeviceFp64Single>("fp64 device", "single");
    PrintDeviceClass<DeviceFp64Orders>("fp64 device", "all-orders");
    PrintDeviceClass<DeviceFp64AllN>("fp64 device", "all-n");
    PrintDeviceClass<DeviceFp32Single>("fp32 device", "single");
    PrintDeviceClass<DeviceFp32Orders>("fp32 device", "all-orders");
    PrintDeviceClass<DeviceFp32AllN>("fp32 device", "all-n");
    PrintDeviceClass<DeviceFp16Single>("fp16 device", "single");
    PrintDeviceClass<DeviceFp16Orders>("fp16 device", "all-orders");
    PrintDeviceClass<DeviceFp16AllN>("fp16 device", "all-n");
}

namespace {

// --- sha256, for the identity check below -------------------------------------
//
// The claim the test at the end of this file makes is that the identity the
// library reports is the sha256 of the seam file this build compiled, and the
// only way to check a digest is to compute one: comparing that answer with
// BOYS_BUILD_DEFAULTS_SEAM_SHA256 would be the check agreeing with the definition
// it just read, which is green whatever file the digest was taken of. Nothing
// else in this tree hashes anything, so the implementation is here, and the test
// that uses it runs FIPS 180-4's two published vectors through it first: a broken
// hash is then a failing assertion rather than a wrong expectation.

inline std::uint32_t RotateRight(std::uint32_t value, int bits) {
    return (value >> bits) | (value << (32 - bits));
}

// The round constants: the first 32 bits of the fractional parts of the cube
// roots of the first 64 primes.
constexpr std::uint32_t kSha256Round[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
    0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
    0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
    0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
    0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
    0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
    0xc67178f2u};

/// The sha256 of a byte string, as 64 lowercase hexadecimal digits.
///
/// \param message the bytes to hash, taken as they are and not as text: the
///                callers hash a file, and a line-ending translation would put a
///                Windows build and a unix one at different digests for the same
///                file
///
/// \returns FIPS 180-4's digest of \p message
std::string Sha256Hex(const std::string& message) {
    // The initial hash value: the first 32 bits of the fractional parts of the
    // square roots of the first eight primes.
    std::uint32_t hash[8] = {0x6a09e667u,
                             0xbb67ae85u,
                             0x3c6ef372u,
                             0xa54ff53au,
                             0x510e527fu,
                             0x9b05688cu,
                             0x1f83d9abu,
                             0x5be0cd19u};

    // The padding: one 0x80 byte, zeroes to 56 modulo 64, then the length in bits
    // as a big-endian 64-bit integer.
    std::string padded = message;
    const std::uint64_t bits = static_cast<std::uint64_t>(message.size()) * 8u;
    padded.push_back(static_cast<char>(0x80u));

    while (padded.size() % 64u != 56u) {
        padded.push_back('\0');
    }

    for (int shift = 56; shift >= 0; shift -= 8) {
        padded.push_back(static_cast<char>((bits >> shift) & 0xffu));
    }

    for (std::size_t offset = 0; offset < padded.size(); offset += 64u) {
        std::uint32_t word[64] = {};

        for (std::size_t i = 0; i < 16u; ++i) {
            const std::size_t at = offset + i * 4u;
            word[i] = (static_cast<std::uint32_t>(static_cast<std::uint8_t>(padded[at])) << 24) |
                      (static_cast<std::uint32_t>(static_cast<std::uint8_t>(padded[at + 1u])) << 16) |
                      (static_cast<std::uint32_t>(static_cast<std::uint8_t>(padded[at + 2u])) << 8) |
                      static_cast<std::uint32_t>(static_cast<std::uint8_t>(padded[at + 3u]));
        }

        for (std::size_t i = 16u; i < 64u; ++i) {
            const std::uint32_t s0 =
                RotateRight(word[i - 15u], 7) ^ RotateRight(word[i - 15u], 18) ^ (word[i - 15u] >> 3);
            const std::uint32_t s1 =
                RotateRight(word[i - 2u], 17) ^ RotateRight(word[i - 2u], 19) ^ (word[i - 2u] >> 10);
            word[i] = word[i - 16u] + s0 + word[i - 7u] + s1;
        }

        std::uint32_t a = hash[0];
        std::uint32_t b = hash[1];
        std::uint32_t c = hash[2];
        std::uint32_t d = hash[3];
        std::uint32_t e = hash[4];
        std::uint32_t f = hash[5];
        std::uint32_t g = hash[6];
        std::uint32_t h = hash[7];

        for (std::size_t i = 0; i < 64u; ++i) {
            const std::uint32_t s1 = RotateRight(e, 6) ^ RotateRight(e, 11) ^ RotateRight(e, 25);
            const std::uint32_t choose = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + choose + kSha256Round[i] + word[i];
            const std::uint32_t s0 = RotateRight(a, 2) ^ RotateRight(a, 13) ^ RotateRight(a, 22);
            const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + majority;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        hash[0] += a;
        hash[1] += b;
        hash[2] += c;
        hash[3] += d;
        hash[4] += e;
        hash[5] += f;
        hash[6] += g;
        hash[7] += h;
    }

    const char* const digits = "0123456789abcdef";
    std::string hex;
    hex.reserve(64u);

    for (const std::uint32_t part : hash) {
        for (int shift = 28; shift >= 0; shift -= 4) {
            hex.push_back(digits[(part >> shift) & 0xfu]);
        }
    }

    return hex;
}

} // namespace

// --- The seam's identity, held to the file it is the identity of --------------
//
// WHAT THIS IS FOR. A build input that changes delivered bits has to be able to
// say which file it was: the seam is such an input - an unnamed call resolves
// through the table the seam file carries, and the tests above this one measure
// that a replaced seam delivers other bits - and until this revision nothing a
// consumer could reach said whether the seam had been replaced, let alone by
// what. A consumer whose regression baseline moved had no way to see that the
// seam is why. The library reports the seam's identity now
// (include/boys/version.hpp, BuildDefaultsSeamIdentity()), and this test is the
// half that makes that report a claim rather than a decoration.
//
// WHAT IT COMPARES, AND WHY THE HASH ABOVE IS WRITTEN HERE. The identity is the
// sha256 of the seam file, and the only way to check a digest is to compute one:
// comparing the accessor against BOYS_BUILD_DEFAULTS_SEAM_SHA256 would be the
// check agreeing with the definition it just read, which stays green whatever
// file that digest was taken of. So this unit hashes the file the configure named
// (BOYS_DEFAULTS_SEAM_FILE, CMakeLists.txt) itself and requires the library's
// answer to be that. The two FIPS 180-4 vectors run first, so a broken hash fails
// here rather than silently agreeing with a wrong expectation.
//
// WHERE EACH HALF RUNS. A default configure runs the shipped half: the seam is
// the committed header, and the library has to say so, in the words the seam's own
// guard lets this unit read beside it (BOYS_BUILD_DEFAULTS_SHIPPED is what the
// committed file defines and a replacement does not). A configure with
// BOYS_BUILD_DEFAULTS runs the other half - CI's `Build defaults (tuned fixture
// built and tested)` step is one - and there the library must report the digest of
// the file that configure named, which on that leg is the tuned fixture.
//
// WHAT IT CANNOT SEE, so that a pass here is read for what it is. It hashes the
// source file the option named and not the copy the compiler reads: the copy is
// configure_file(COPYONLY) of that file, so the two are the same bytes by
// construction, and which of the two a translation unit reached is the seam
// guard's business (boys_build_defaults.hpp). A build directory carried somewhere
// without the sources it was configured from has no file to hash, and this test
// skips there with a printed line rather than passing: the identity is then
// unverified, and this file says so instead of implying otherwise.
TEST(BuildDefaultsTest, TheReportedIdentityIsTheSeamThisBuildCompiled) {
    // Before the implementation is used for anything: FIPS 180-4's published
    // digests of the empty string and of "abc".
    ASSERT_EQ(Sha256Hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    ASSERT_EQ(Sha256Hex("abc"),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");

    const char* identity = boys::BuildDefaultsSeamIdentity();
    std::printf("boys: defaults seam identity = %s\n", identity);
    std::printf("boys: the configure named    = %s\n", BOYS_DEFAULTS_SEAM_FILE);

    // The file, as bytes: opened binary and read whole, so the digest is of what
    // the file holds and not of what a text-mode reader would translate it into.
    std::string seam;
    {
        std::ifstream in(BOYS_DEFAULTS_SEAM_FILE, std::ios::binary);

        if (!in) {
            GTEST_SKIP() << "the seam file the configure named (" << BOYS_DEFAULTS_SEAM_FILE
                         << ") is not readable here, so the identity the library reports could not "
                            "be held to it - a skipped check, not a passing one";
        }

        seam.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    }

    ASSERT_FALSE(seam.empty()) << "the seam file the configure named is empty, which no seam is";

#if defined(BOYS_BUILD_DEFAULTS_REPLACED)
    const std::string digest = Sha256Hex(seam);
    std::printf("boys: the seam file hashes to  = %s\n", digest.c_str());

    EXPECT_EQ(std::string(identity), digest)
        << "the identity this build reports is not the sha256 of the seam file the configure "
           "named: a consumer comparing two builds by this string is comparing something other "
           "than the seam";
    EXPECT_EQ(std::string(identity).size(), 64u)
        << "the reported identity is not 64 characters long, which is the form the accessor "
           "documents for a digest";
    EXPECT_EQ(std::string(identity).find_first_not_of("0123456789abcdef"), std::string::npos)
        << "the reported identity is not lowercase hexadecimal, which is the form the accessor "
           "documents for a digest";
#else
    EXPECT_STREQ(identity, "the committed header (the shipped choices)")
        << "a build that replaced no seam reports an identity that does not name the shipped one, "
           "so a consumer reading it cannot tell a shipped build from a build whose seam carried "
           "a digest";

    // The other half of the claim is the file's own: the committed seam defines
    // BOYS_BUILD_DEFAULTS_SHIPPED and a replacement does not, so this unit can say
    // which of the two seams it read. A unit in neither state read a seam that did
    // not come through the option, and the shipped words are then not about the
    // seam this unit resolved.
#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
    constexpr bool kSeamSaysShipped = true;
#else
    constexpr bool kSeamSaysShipped = false;
#endif
    EXPECT_TRUE(kSeamSaysShipped)
        << "the seam header this unit read defines no BOYS_BUILD_DEFAULTS_SHIPPED, so it is not "
           "the committed file the reported identity names";

    EXPECT_NE(seam.find("BOYS_BUILD_DEFAULTS_SHIPPED"), std::string::npos)
        << "the file the configure named as the seam in force does not define "
           "BOYS_BUILD_DEFAULTS_SHIPPED, so the shipped words the library reports do not describe "
           "the file this build compiled";
#endif
}
