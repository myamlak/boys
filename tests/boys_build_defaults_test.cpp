// The build-defaults seam's compile half: every choice include/boys/boys_build_defaults.hpp offers is
// read as the value an entry naming no policy resolves to. A replacement naming the committed value
// is indistinguishable from a macro nothing expands, and PackAxis::kOrders is no build's to name
// (include/boys/boys_impl.hpp): the teeth are tests/build_defaults_tuned.hpp's configure.

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

// The macros expand to enumerator names the library writes as FitRoute::kChebyshev and its six
// siblings, inside namespace boys: this unit reads them and so has to import that namespace, which is
// the only reason this directive is here.
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

// The policy the seam's five compose: the point a class with no row resolves to, and the type
// EvalPolicy<> names. Not what every unnamed entry resolves to - an entry resolves to its class's row,
// which is this combination only where the row spells it (include/boys/boys.hpp expands the row list
// or the five, never both). Its RegionBExp is boys::kDefaultHostRegionBExp, which no host macro names.
using SeamPolicy = boys::EvalPolicy<BOYS_BUILD_DEFAULT_FIT_ROUTE,
                                    BOYS_BUILD_DEFAULT_EVAL_SCHEME,
                                    boys::BoysBudget::kFloat,
                                    BOYS_BUILD_DEFAULT_PACK_AXIS,
                                    BOYS_BUILD_DEFAULT_FIT_GRANULARITY,
                                    BOYS_BUILD_DEFAULT_DIVISION_FORM,
                                    boys::kDefaultHostRegionBExp>;

// --- Link one: the constant that owns a macro reads it ------------------------
// Each of the five constants is the name the entries' template defaults read. One that stopped
// expanding its macro is the shape that sat here before the wiring: the macro stays in the seam file,
// documented, and the library compiles a literal the committed header happens to agree with.
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

// The device lane's two names, held the same way: a device class resolves its form and its region-B
// exponential through these and never through the host's, so a device half that stopped reading its
// own names would take the host's values in silence.
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
// taken at the library routine, so the committed configuration's own device exponential is that
// routine. A replacement states its own value and this pin is about what this repository
// publishes, not about what a consumer's card measured.
#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
static_assert(boys::kDefaultRegionBExp == boys::RegionBExp::kAccurate,
              "the committed header's device region-B exponential is not the one the device rows "
              "beside it were measured at: boys/boys_build_defaults.hpp states RegionBExp::kAccurate");
#endif

// --- Link two: the policy the five constants compose --------------------------
// EvalPolicy's template defaults are the five constants, so EvalPolicy<> is the host combination the
// seam names; a device class composes the device lane's own two beside four of these rather than this
// type (include/boys/boys_build_defaults.hpp).
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
// Each name is what the entries of that precision run when the call site names no policy, so each
// carries the seam's five. The budget is not compared: it is not one of the five, and it is the one
// field the fp16 and bf16 names move.
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

// --- Link four: the twelve device classes, asked for by name ------------------
// The device half of the table is four device lanes by three questions; each class is asked for as a
// caller asks, by its own default policy (include/boys/boys.hpp, DefaultPolicyFor). That ask is the
// check: a device row dropped leaves one of the twelve unanswered and the static_assert there fails.
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
using DeviceBf16Single = boys::DefaultPolicy<boys::Precision::kBf16Device, boys::Shape::kSingle,
                                             boys::Device::kDevice>;
using DeviceBf16Orders = boys::DefaultPolicy<boys::Precision::kBf16Device,
                                             boys::Shape::kAllOrders, boys::Device::kDevice>;
using DeviceBf16AllN = boys::DefaultPolicy<boys::Precision::kBf16Device, boys::Shape::kAllN,
                                           boys::Device::kDevice>;

// Which of the two the file in force is: the committed header defines
// BOYS_BUILD_DEFAULTS_COMMITTED, a replacement does not, and the CMake option puts
// BOYS_BUILD_DEFAULTS_REPLACED on the command line of every unit of a build that
// used it (include/boys/boys_build_defaults.hpp states the contract).
constexpr const char* SeamInForce() {
#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
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

// One device class of the twelve, as a report prints it: the class, and the row the name it was
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

// The entry, not only the policy type: an entry whose default argument stopped being its class's alias
// would keep every assert above true and answer a caller from another policy. A replacement defining
// BOYS_BUILD_DEFAULT_ROWS is read instead of the five-composed table (include/boys/boys.hpp expands one
// branch or the other), so the sweep asserts the entry follows whichever the build's table holds.
TEST(BuildDefaultsTest, AnUnnamedCallIsItsClasssDefault) {
    using boys::BoysAllOrders;
    using boys::BoysSingle;

    // The class each entry belongs to, as the policy it resolves to. Named through the
    // library's own alias rather than spelled here: this is the name an entry that names no
    // policy resolves to (boys/boys.hpp, DefaultPolicy), so naming it is what makes the
    // comparison a claim about the build's table and not about this line.
    using OrdersClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;
    using SingleClass = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kSingle>;

    // Which of the two the table answers each class with, read from the class's own policy. The budget
    // is not one of the five and is compared as the fp64 lane's. The region-B exponential is compared
    // too, the axis a row can move while spelling the seam's five: a flag that stopped there would call
    // such a class the seam's own combination and require the two calls to agree.
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

    // The table's own precedence, at the call a caller writes: the unnamed call equals the seam
    // policy's call exactly where the row that carries the class spells the five.
    EXPECT_EQ(batchSeamMoved == 0u, kOrdersClassIsTheSeamFive)
        << "the unnamed call and the seam policy's call differ where the table's row for the "
           "class spells the seam's five, or agree where the row moves the class: the entry is "
           "not resolving through the row the build's table carries for it";
    EXPECT_EQ(singleSeamMoved == 0u, kSingleClassIsTheSeamFive)
        << "the single-order entry resolves through the five rather than through the row the "
           "build's table carries for its class, or the other way round";
}

// The same claim, per entry, over every entry a missing row would silently move: each entry below is
// its own declaration, whose default template argument is that entry's class alias, and one that
// stopped reading it answers from a policy its class's row does not name. boys_all_n_test.cpp and
// boys_c_test.cpp compare entries against each other and read the class table nowhere; this reads it.
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
// run-time reading of Link four, which is instantiated whether or not this test runs, so the print
// adds which row each name resolved to rather than whether it resolved at all.
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
    PrintDeviceClass<DeviceBf16Single>("bf16 device", "single");
    PrintDeviceClass<DeviceBf16Orders>("bf16 device", "all-orders");
    PrintDeviceClass<DeviceBf16AllN>("bf16 device", "all-n");
}

namespace {

// --- sha256, for the identity check below -------------------------------------
// The identity the library reports is the sha256 of the seam file this build compiled, and a digest is
// checked only by computing one - comparing it with BOYS_BUILD_DEFAULTS_SEAM_SHA256 would be the check
// agreeing with the definition it read. FIPS 180-4's two published vectors run first.

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
// The library reports the seam's identity (include/boys/version.hpp, BuildDefaultsSeamIdentity()), and
// this test hashes the file the configure named (BOYS_DEFAULTS_SEAM_FILE) rather than comparing it
// with BOYS_BUILD_DEFAULTS_SEAM_SHA256, which would be the check agreeing with the definition it read.
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
        << "a build that replaced no seam reports an identity that does not name the committed one, "
           "so a consumer reading it cannot tell a committed build from a build whose seam carried "
           "a digest";

    // The other half of the claim is the file's own: the committed seam defines
    // BOYS_BUILD_DEFAULTS_COMMITTED and a replacement does not, so this unit can say which of the two
    // seams it read; a unit in neither state read a seam that did not come through the option.
#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
    constexpr bool kSeamIsCommitted = true;
#else
    constexpr bool kSeamIsCommitted = false;
#endif
    EXPECT_TRUE(kSeamIsCommitted)
        << "the seam header this unit read defines no BOYS_BUILD_DEFAULTS_COMMITTED, so it is not "
           "the committed file the reported identity names";

    EXPECT_NE(seam.find("BOYS_BUILD_DEFAULTS_COMMITTED"), std::string::npos)
        << "the file the configure named as the seam in force does not define "
           "BOYS_BUILD_DEFAULTS_COMMITTED, so the committed words the library reports do not describe "
           "the file this build compiled";
#endif
}
