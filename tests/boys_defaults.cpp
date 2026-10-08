// The defaults command (ctest case boys-defaults, BOYS_BUILD_TESTS): what this build's unnamed
// calls resolve to, read from the table it carries (boys::detail::DefaultPolicyRow,
// include/boys/boys.hpp). Holding every class an entry reaches to a row is not this report's:
// tools/check_default_rows_have_entries.py and tests/boys_defaults_link_test.cpp do that.

#include "boys/boys.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <type_traits>

namespace {

using namespace boys;

// The seam file in force, and - for a build that replaced it - the copy the compiler read on the
// include path before include/. Both come from the configure rather than __FILE__, whose own path is
// whatever the include directive spelled and would not say which table produced the report.
#ifndef BOYS_DEFAULTS_HEADER_PATH
#define BOYS_DEFAULTS_HEADER_PATH "(not recorded: this translation unit was not compiled by the boys CMake build)"
#endif
#ifndef BOYS_DEFAULTS_HEADER_COPY
#define BOYS_DEFAULTS_HEADER_COPY ""
#endif

// The names the library does not publish: the fit route, the scheme, the packing axis, the
// partition and the division form have names from the library (RouteName below), while the device,
// the shape and the compute budget have none and are named here. Each switch names
// every enumerator of its axis with no default arm, so a member added later fails a -Werror build.

/// The name this report prints a device under.
constexpr const char* DeviceName(Device device) noexcept {
    switch (device)
    {
    case Device::kHost:
        return "host";
    case Device::kDevice:
        return "device";
    }

    return "unknown";
}

/// The name this report prints a question shape under.
constexpr const char* ShapeName(Shape shape) noexcept {
    switch (shape)
    {
    case Shape::kSingle:
        return "single";
    case Shape::kAllOrders:
        return "all-orders";
    case Shape::kFixedN:
        return "fixed-n";
    case Shape::kAllN:
        return "all-n";
    case Shape::kAllNAtOrders:
        return "all-n-at-orders";
    }

    return "unknown";
}

/// The name this report prints a compute budget under.
constexpr const char* BudgetName(BoysBudget budget) noexcept {
    switch (budget)
    {
    case BoysBudget::kFloat:
        return "float";
    case BoysBudget::kFp16:
        return "fp16";
    }

    return "unknown";
}

// The three enumerations a class key is made of, one walk per axis, spelled out because C++ cannot
// walk one. The lane list is not the walk: the walk's lanes are these unioned with the lanes the
// seam's rows name (MakeLaneWalk), so the 2026-10-04 repair (Precision::kBf16 with five bf16 rows,
// list ending at kFp16Device) cannot repeat; main() holds the list to the library's lane table.
#define BOYS_DEFAULTS_DEVICES(X) X(kHost) X(kDevice)
#define BOYS_DEFAULTS_PRECISIONS(X)                                                               \
    X(kFp64) X(kFp32) X(kFp16) X(kBf16) X(kFp32Device) X(kFp64Device) X(kFp16Device) X(kBf16Device)
#define BOYS_DEFAULTS_SHAPES(X) X(kSingle) X(kAllOrders) X(kFixedN) X(kAllN) X(kAllNAtOrders)

/// One enumerator of \c Device that \c DeviceName spells no name for.
#define BOYS_DEFAULTS_DEVICE_NAMED(Enumerator)                                                     \
    static_assert(DeviceName(Device::Enumerator) != nullptr,                                       \
                  "an enumerator of Device names no member: name it in DeviceName "               \
                  "(tests/boys_defaults.cpp)");
BOYS_DEFAULTS_DEVICES(BOYS_DEFAULTS_DEVICE_NAMED)
#undef BOYS_DEFAULTS_DEVICE_NAMED

/// One enumerator of \c Shape that \c ShapeName spells no name for.
#define BOYS_DEFAULTS_SHAPE_NAMED(Enumerator)                                                      \
    static_assert(ShapeName(Shape::Enumerator) != nullptr,                                          \
                  "an enumerator of Shape names no member: name it in ShapeName "                  \
                  "(tests/boys_defaults.cpp)");
BOYS_DEFAULTS_SHAPES(BOYS_DEFAULTS_SHAPE_NAMED)
#undef BOYS_DEFAULTS_SHAPE_NAMED

#define BOYS_DEFAULTS_DEVICE_ENTRY(Enumerator) Device::Enumerator,
constexpr std::array<Device, 2> kDevices{BOYS_DEFAULTS_DEVICES(BOYS_DEFAULTS_DEVICE_ENTRY)};
#undef BOYS_DEFAULTS_DEVICE_ENTRY

#define BOYS_DEFAULTS_PRECISION_ENTRY(Enumerator) Precision::Enumerator,
constexpr std::array<Precision, 8> kEnumerationLanes{
    BOYS_DEFAULTS_PRECISIONS(BOYS_DEFAULTS_PRECISION_ENTRY)};
#undef BOYS_DEFAULTS_PRECISION_ENTRY

// --- the lanes the seam's own rows name -------------------------------------

/// How many rows this build's seam carries, and one lane per row.
///
/// The count is the row list's own arity, so a row added to the seam moves it
/// with the list rather than leaving a number written here to drift from it.
#if defined(BOYS_BUILD_DEFAULT_ROWS)
#define BOYS_DEFAULTS_ROW_COUNTS(kDevice, kPrecision, kShape, ...) +1
constexpr std::size_t kSeamRowCount = 0 BOYS_BUILD_DEFAULT_ROWS(BOYS_DEFAULTS_ROW_COUNTS);
#undef BOYS_DEFAULTS_ROW_COUNTS

#define BOYS_DEFAULTS_ROW_LANE(kDevice, kPrecision, kShape, ...) Precision::kPrecision,
constexpr std::array<Precision, kSeamRowCount> kSeamLanes{
    BOYS_BUILD_DEFAULT_ROWS(BOYS_DEFAULTS_ROW_LANE)};
#undef BOYS_DEFAULTS_ROW_LANE
#else
/// A build whose seam carries no row list has no cells to read lanes from.
constexpr std::size_t kSeamRowCount = 0;
#endif

/// The lanes this report crosses: one slot per lane of the enumeration plus one
/// per row of the seam, which is the most the union below can hold.
constexpr std::size_t kLaneSlots = kEnumerationLanes.size() + kSeamRowCount;

/// One lane the walk carries, and how many it carries.
struct LaneWalk {
    std::array<Precision, kLaneSlots> members{};
    std::size_t count = 0;
};

/// Appends a lane to a walk unless the walk already carries it.
///
/// \param walk the walk so far
/// \param lane the lane this row names
constexpr void CarryLane(LaneWalk& walk, Precision lane) noexcept {
    for (std::size_t slot = 0; slot < walk.count; ++slot)
    {
        if (walk.members[slot] == lane)
        {
            return;
        }
    }

    walk.members[walk.count] = lane;
    ++walk.count;
}

/// The lanes the walk crosses: the enumeration's own members first, then every
/// lane the seam's rows name, deduplicated and in the library's own enumerator
/// order - the order \c BoysLaneContracts() reports its rows in, so the class
/// table and the lane table below it read lane by lane.
///
/// A seam class is walked by this union and not by a list a reader maintains
/// beside it: the row names the lane, and that is what puts the lane in the
/// walk. The enumeration's own record is the fallback for a build whose seam
/// names no rows at all.
constexpr LaneWalk MakeLaneWalk() noexcept {
    LaneWalk walk{};

    for (const Precision lane : kEnumerationLanes)
    {
        CarryLane(walk, lane);
    }

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    for (const Precision lane : kSeamLanes)
    {
        CarryLane(walk, lane);
    }
#endif

    // The library's order, by insertion: one lane per row of BoysLaneContracts,
    // so a report line and the lane table it is composed from read the same way.
    for (std::size_t index = 1; index < walk.count; ++index)
    {
        const Precision lane = walk.members[index];
        std::size_t slot = index;

        while (slot > 0 &&
               static_cast<unsigned>(walk.members[slot - 1]) > static_cast<unsigned>(lane))
        {
            walk.members[slot] = walk.members[slot - 1];
            --slot;
        }

        walk.members[slot] = lane;
    }

    return walk;
}

constexpr LaneWalk kLaneWalk = MakeLaneWalk();

/// The walk's lanes, as the three loops below read them: the union,
/// deduplicated, in the library's own lane order.
constexpr std::span<const Precision> kPrecisions{kLaneWalk.members.data(), kLaneWalk.count};

#define BOYS_DEFAULTS_SHAPE_ENTRY(Enumerator) Shape::Enumerator,
constexpr std::array<Shape, 5> kShapes{BOYS_DEFAULTS_SHAPES(BOYS_DEFAULTS_SHAPE_ENTRY)};
#undef BOYS_DEFAULTS_SHAPE_ENTRY

// A build whose seam file carries a row list names its classes there, and the walk above must reach
// every one: the lane cell is carried by construction (MakeLaneWalk above), so the assertion below
// is a tripwire for the row's device and shape, the two cells a row list cannot put into the walk by
// naming them. A class it names fails the build here, with a line saying which list to extend.
#if defined(BOYS_BUILD_DEFAULT_ROWS)
/// Whether a class is one of the classes the walks above cross.
constexpr bool ScannedClass(Device device, Precision precision, Shape shape) noexcept {
    bool deviceFound = false;
    bool precisionFound = false;
    bool shapeFound = false;

    for (const Device member : kDevices)
    {
        deviceFound = deviceFound || member == device;
    }

    for (const Precision member : kPrecisions)
    {
        precisionFound = precisionFound || member == precision;
    }

    for (const Shape member : kShapes)
    {
        shapeFound = shapeFound || member == shape;
    }

    return deviceFound && precisionFound && shapeFound;
}

#define BOYS_DEFAULTS_SCAN_COVERS_ROW(kDevice, kPrecision, kShape, ...)                            \
    static_assert(ScannedClass(Device::kDevice, Precision::kPrecision, Shape::kShape),             \
                  "this build's BOYS_BUILD_DEFAULT_ROWS names a class the classes command's walk " \
                  "does not reach: add the enumerator to the walk in tests/boys_defaults.cpp");
BOYS_BUILD_DEFAULT_ROWS(BOYS_DEFAULTS_SCAN_COVERS_ROW)
#undef BOYS_DEFAULTS_SCAN_COVERS_ROW
#endif

// --- the library's own name tables -----------------------------------------

/// The name the library prints a fit route under, read from the route table
/// rather than written here, so a report line cannot name a route the library
/// does not.
const char* RouteName(FitRoute route) noexcept {
    for (const FitRouteInfo& row : BoysFitRoutes())
    {
        if (row.route == route)
        {
            return row.name;
        }
    }

    return "unknown";
}

/// The lane's own contract row.
///
/// \param lane a lane of the walk
/// \returns the row \c BoysLaneContracts() carries for it, which is one row per
///          lane in enumerator order
const LaneContractInfo& LaneRow(Precision lane) noexcept {
    return BoysLaneContracts()[static_cast<std::size_t>(lane)];
}

// --- what the run reports and what it checks -------------------------------

/// What the walk found, and every claim the report makes about itself.
struct Coverage {
    /// Classes the walk asked the table about.
    std::size_t scanned = 0;
    /// Classes the table carries a row for, per (device slot, lane slot).
    std::array<std::array<std::size_t, kPrecisions.size()>, kDevices.size()> carried{};
    /// Classes it does not name, per (device slot, lane slot).
    std::array<std::array<std::size_t, kPrecisions.size()>, kDevices.size()> missing{};
    /// Classes whose reference figure the accessor refused.
    std::size_t figureRefused = 0;
    /// Figures the composed arithmetic did not reproduce, to the bit.
    std::size_t compositionFailed = 0;
    /// Host classes where \c DefaultGuarantee and the axis-taking accessor
    /// disagree.
    std::size_t accessorsDisagree = 0;
};

/// The figure a class's default carries, through the name the library
/// publishes for it.
///
/// Both sides of the interface read the one accessor:
/// \c DefaultGuarantee<Precision, Shape, Device>() is the name the seam documents
/// for "the bound a class's default policy carries", it reads the row the table
/// carries for the class the caller names, and \c kDevice defaults to
/// \c Device::kHost where a caller does not spell one. A device class reads it
/// too, rather than assembling the figure here from the axes its own row resolves
/// to: an assembled reading is this file's answer rather than the library's, and
/// what the report is about is the library's.
///
/// \tparam kDevice the class's device
/// \tparam kPrecision the class's lane
/// \tparam kShape the class's shape
/// \returns the figure and whether this build carries the combination
template <Device kDevice, Precision kPrecision, Shape kShape>
AccuracyFigure ClassGuarantee() noexcept {
    return DefaultGuarantee<kPrecision, kShape, kDevice>();
}

/// The term the class's division form adds beside the lane's base.
///
/// \param lane the lane's contract row
/// \param form the class's division form
/// \returns the term beside the base, 0.0 where the lane's forms share one figure
double FormTerm(const LaneContractInfo& lane, DivisionForm form) noexcept {
    return form == DivisionForm::kPlainReciprocal ? lane.plainAdditive : 0.0;
}

/// The term the class's region-B exponential adds beside the lane's base.
///
/// The row states the member its additive term is under - the fp32-device row's
/// 8e-8 is the fast member's own contribution, certified there and nowhere else
/// (include/boys/boys.hpp, \c LaneContractInfo::additiveMember) - so a class
/// naming the other member is not owed it and is answered without it. That is the
/// accessor's own rule (src/boys.cpp, \c BoysAccuracyGuaranteed: \c memberTerm),
/// written here so the report's terms are the accessor's terms.
///
/// \param lane the lane's contract row
/// \param exp the class's region-B exponential
/// \returns the term beside the base, 0.0 where the class names the other member
double MemberTerm(const LaneContractInfo& lane, RegionBExp exp) noexcept {
    return exp == lane.additiveMember ? lane.additive : 0.0;
}

/// The arithmetic the library states a figure by: the lane's multiplicand plus
/// the term the named form adds beside it, times the multiplier of the one
/// accuracy this library serves, plus the term the named region-B exponential
/// adds beside the base.
///
/// It is the accessor's own expression (src/boys.cpp, \c BoysAccuracyGuaranteed),
/// written out so the report can print the terms a reader recomputes the figure
/// with and can hold the two to each other.
///
/// \param lane the lane's contract row
/// \param form the class's division form
/// \param exp the class's region-B exponential
/// \returns the figure the terms compose to
double ComposedFigure(const LaneContractInfo& lane, DivisionForm form, RegionBExp exp) noexcept {
    return kBoysFullAccuracyMultiplier * (lane.bound + FormTerm(lane, form)) +
           MemberTerm(lane, exp);
}

/// One class's row: the class, the six axes the table resolves it to, and the
/// bound that default carries, stated per the multiplicand and the multiplier it
/// is computed from.
///
/// \tparam kDevice the class's device
/// \tparam kPrecision the class's lane
/// \tparam kShape the class's shape
/// \param coverage the run's counters
template <Device kDevice, Precision kPrecision, Shape kShape>
void PrintClass(Coverage& coverage) {
    using Policy = DefaultPolicy<kPrecision, kShape, kDevice>;

    // The public name and the table's own row are one type, so the report prints
    // the axes a caller who names the policy would get, not a second resolution
    // of them.
    static_assert(
        std::is_same_v<Policy, typename detail::DefaultPolicyRow<kDevice, kPrecision, kShape>::Type>,
        "a class's default policy and the table's row for it are not one type");

    const LaneContractInfo& lane = LaneRow(kPrecision);
    const AccuracyFigure figure = ClassGuarantee<kDevice, kPrecision, kShape>();

    std::printf("  %-6s %-11s %-15s %-16s %-14s %-6s %-9s %-11s %-18s %-8s ",
                DeviceName(kDevice),
                lane.name,
                ShapeName(kShape),
                RouteName(Policy::kRoute),
                EvalSchemeName(Policy::kScheme),
                BudgetName(Policy::kBudget),
                PackAxisName(Policy::kPack),
                GranularityName(Policy::kGranularity),
                DivisionFormName(Policy::kDivision),
                RegionBExpName(Policy::kRegionBExp));

    if (!figure.available)
    {
        std::printf("no bound at m = 1 (%s)\n", figure.reason);
        ++coverage.figureRefused;

        return;
    }

    std::printf("bound %.6g = (%.6g + %.6g) x m %.6g + %.6g\n",
                figure.value,
                lane.bound,
                FormTerm(lane, Policy::kDivision),
                kBoysFullAccuracyMultiplier,
                MemberTerm(lane, Policy::kRegionBExp));

    if (ComposedFigure(lane, Policy::kDivision, Policy::kRegionBExp) != figure.value)
    {
        std::printf("    the terms above do not compose to the figure the accessor returned: "
                    "composed %.17g, returned %.17g\n",
                    ComposedFigure(lane, Policy::kDivision, Policy::kRegionBExp),
                    figure.value);
        ++coverage.compositionFailed;
    }

    // The two named readings of one figure, held to each other: DefaultGuarantee is documented
    // (include/boys/boys.hpp, \c DefaultGuarantee) as the same figure from the same table as the
    // axis-taking accessor asked with the policy's own axes, the region-B exponential included.
    const AccuracyFigure direct = BoysAccuracyGuaranteed(kPrecision,
                                                         Policy::kRoute,
                                                         Policy::kScheme,
                                                         Policy::kPack,
                                                         Policy::kGranularity,
                                                         Policy::kDivision,
                                                         Policy::kRegionBExp);

    if (!direct.available || direct.value != figure.value)
    {
        std::printf("    DefaultGuarantee and the axes the class resolves to disagree: "
                    "%.17g against %.17g\n",
                    figure.value,
                    direct.value);
        ++coverage.accessorsDisagree;
    }
}

/// One class of the walk: the row the table carries for it, or a count of it
/// among the classes the table does not name.
///
/// \tparam kDevice the device's slot in the walk
/// \tparam kLane the lane's slot in the walk
/// \tparam kShape the shape's slot in the walk
/// \param coverage the run's counters
template <std::size_t kDevice, std::size_t kLane, std::size_t kShape>
void AskClass(Coverage& coverage) {
    constexpr Device kDeviceValue = kDevices[kDevice];
    constexpr Precision kPrecisionValue = kPrecisions[kLane];
    constexpr Shape kShapeValue = kShapes[kShape];

    ++coverage.scanned;

    // The gate is the table's own answer, asked of the primary template - which carries no Type
    // for a class it does not name - rather than of DefaultPolicyFor, whose assertion makes the
    // same question the compile error a call site gets.
    if constexpr (detail::DefaultPolicyRow<kDeviceValue, kPrecisionValue, kShapeValue>::kCarried)
    {
        ++coverage.carried[kDevice][kLane];
        PrintClass<kDeviceValue, kPrecisionValue, kShapeValue>(coverage);
    }
    else
    {
        ++coverage.missing[kDevice][kLane];
    }
}

/// Every class the three walks cross, asked one compiled instantiation at a
/// time: the recursion is the three loops a run-time walk would have, and it is
/// unrolled because a class key is a template argument rather than a value. This
/// is what "nothing is looked up" means from the inside.
///
/// \tparam kDevice the device's slot, or the walk's end
/// \tparam kLane the lane's slot
/// \tparam kShape the shape's slot
/// \param coverage the run's counters
template <std::size_t kDevice = 0, std::size_t kLane = 0, std::size_t kShape = 0>
void WalkClasses(Coverage& coverage) {
    if constexpr (kDevice < kDevices.size())
    {
        AskClass<kDevice, kLane, kShape>(coverage);

        if constexpr (kShape + 1 < kShapes.size())
        {
            WalkClasses<kDevice, kLane, kShape + 1>(coverage);
        }
        else if constexpr (kLane + 1 < kPrecisions.size())
        {
            WalkClasses<kDevice, kLane + 1, 0>(coverage);
        }
        else
        {
            WalkClasses<kDevice + 1, 0, 0>(coverage);
        }
    }
}

/// The path line: the seam file in force, and how this translation unit knows
/// which of the two it is.
void PrintHeader() {
    std::printf("defaults header  %s\n", BOYS_DEFAULTS_HEADER_PATH);

#if defined(BOYS_BUILD_DEFAULTS_COMMITTED)
    std::printf("                 the committed file (BOYS_BUILD_DEFAULTS_COMMITTED): the committed "
                "choices every bound in this repository\n"
                "                 was measured with\n");
#elif defined(BOYS_BUILD_DEFAULTS_REPLACED)
    std::printf("                 a replacement (BOYS_BUILD_DEFAULTS_REPLACED): the header the "
                "BOYS_BUILD_DEFAULTS configure\n"
                "                 option names, copied before include/ on this target's include "
                "path, so this unit and the\n"
                "                 library's own instantiations resolve the same choices\n");

    if (BOYS_DEFAULTS_HEADER_COPY[0] != '\0')
    {
        std::printf("                 read as %s\n", BOYS_DEFAULTS_HEADER_COPY);
    }
#else
    std::printf("                 neither state: a seam file defines BOYS_BUILD_DEFAULTS_COMMITTED or "
                "the build defines\n"
                "                 BOYS_BUILD_DEFAULTS_REPLACED, and this unit saw neither, so this "
                "report cannot say\n"
                "                 which table it read\n");
#endif

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    std::printf("table            the header's own row list (BOYS_BUILD_DEFAULT_ROWS), one row per "
                "class the header names\n");
#else
    std::printf("table            no row list (BOYS_BUILD_DEFAULT_ROWS is undefined): the table is "
                "composed from the five\n"
                "                 names below, one row per class, every cell this build's own "
                "choice\n");
#endif

    std::printf("seam's seven     the host lane's five:\n"
                "                 fit route %s | eval scheme %s | packing axis %s\n"
                "                 division form %s | fit granularity %s\n",
                RouteName(BOYS_BUILD_DEFAULT_FIT_ROUTE),
                EvalSchemeName(BOYS_BUILD_DEFAULT_EVAL_SCHEME),
                PackAxisName(BOYS_BUILD_DEFAULT_PACK_AXIS),
                DivisionFormName(BOYS_BUILD_DEFAULT_DIVISION_FORM),
                GranularityName(BOYS_BUILD_DEFAULT_FIT_GRANULARITY));

    std::printf("                 the device lane's own two:\n"
                "                 division form %s | region-B exponential %s\n",
                DivisionFormName(BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM),
                RegionBExpName(BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP));
}

/// The lane rows the figures above are stated per: the multiplicand, and the
/// terms the accessor adds beside it. Printed after the class table so the
/// figures a reader has just read are the ones these terms recompose.
///
/// \returns how many lane rows the library's table carries, which is the count
///          \c main holds the walk to
std::size_t PrintLanes() {
    std::size_t lanesRead = 0;

    std::printf("\nthe multiplicand each figure above is stated per (BoysLaneContracts(), one row "
                "per lane, read by\nthe accessor rather than restated here):\n");

    for (const LaneContractInfo& lane : BoysLaneContracts())
    {
        std::printf("  %-12s multiplicand %.6g  additive %.6g  plain-form term %.6g\n",
                    lane.name,
                    lane.bound,
                    lane.additive,
                    lane.plainAdditive);
        std::printf("               %s\n", lane.source);

        if (lane.plainSource[0] != '\0')
        {
            std::printf("               under the plain reciprocal: %s\n", lane.plainSource);
        }

        ++lanesRead;
    }

    return lanesRead;
}

} // namespace

int main() {
    std::printf("boys-defaults: what this build's unnamed calls resolve to\n\n");

    PrintHeader();

    // The lane table is the library's own statement of how many lanes this build
    // has, one row per lane; the walk below names them, and the two are held to
    // each other before anything is read through an index.
    const std::size_t laneRows = BoysLaneContracts().size();

    if (laneRows != kPrecisions.size())
    {
        std::printf("the lane walk and the library's lane table disagree: this report scans %zu "
                    "lanes and\nBoysLaneContracts() carries %zu rows, so the classes printed below "
                    "are not every class\nthis build has\n",
                    kPrecisions.size(),
                    laneRows);

        return 1;
    }

    std::printf("\nclasses this build's table carries, with the bound each default carries at "
                "\nm = %.6g (one column per axis of the class's policy, then the bound):\n",
                kBoysFullAccuracyMultiplier);
    std::printf("  %-6s %-11s %-15s %-16s %-14s %-6s %-9s %-11s %-18s %-8s %s\n",
                "device",
                "lane",
                "shape",
                "route",
                "scheme",
                "budget",
                "pack",
                "granularity",
                "division",
                "region-B",
                "bound");

    Coverage coverage;
    WalkClasses(coverage);

    std::printf("\nclasses the table does not name, per device and lane, of the %zu shapes: a call "
                "resolving to one is\na compile error and not a default - the assertion in "
                "DefaultPolicyFor (include/boys/boys.hpp)\nfires where the class is asked for, which "
                "is the seam's contract and not a gap in this report:\n",
                kShapes.size());

    for (std::size_t device = 0; device < kDevices.size(); ++device)
    {
        std::printf("  %-6s", DeviceName(kDevices[device]));

        for (std::size_t lane = 0; lane < kPrecisions.size(); ++lane)
        {
            std::printf(" %s %zu", LaneRow(kPrecisions[lane]).name, coverage.missing[device][lane]);
        }

        std::printf("\n");
    }

    const std::size_t lanesPrinted = PrintLanes();

    // --- the checks, stated as lines ---------------------------------------
    std::size_t carried = 0;
    std::size_t missing = 0;

    for (std::size_t device = 0; device < kDevices.size(); ++device)
    {
        for (std::size_t lane = 0; lane < kPrecisions.size(); ++lane)
        {
            carried += coverage.carried[device][lane];
            missing += coverage.missing[device][lane];
        }
    }

    std::printf("\nverdict: %zu of the %zu (device, precision, shape) classes this build's three "
                "enumerations name\ncarry a row, and %zu carry none; %zu lane rows read\n",
                carried,
                coverage.scanned,
                missing,
                lanesPrinted);

    std::size_t failures = 0;

    if (carried == 0)
    {
        std::printf("  FAILED: this table carries no class at all, so an unnamed call has no "
                    "default to resolve\n");
        ++failures;
    }

    if (coverage.figureRefused != 0)
    {
        std::printf("  FAILED: %zu carried classes have no bound at m = 1\n",
                    coverage.figureRefused);
        ++failures;
    }

    if (coverage.compositionFailed != 0)
    {
        std::printf("  FAILED: %zu figures are not the ones the printed multiplicity composes to\n",
                    coverage.compositionFailed);
        ++failures;
    }

    if (coverage.accessorsDisagree != 0)
    {
        std::printf("  FAILED: %zu classes where DefaultGuarantee and the class's own axes "
                    "disagree\n",
                    coverage.accessorsDisagree);
        ++failures;
    }

    if (failures == 0)
    {
        std::printf("  all %zu carried classes resolved: every reference figure is available, the "
                    "printed\n  multiplicity composes to it to the bit, and a class's "
                    "DefaultGuarantee and its own\n  axes are one figure. The axes are the table's "
                    "own specializations; nothing was evaluated\n  and nothing was timed\n",
                    carried);
    }

    return failures == 0 ? 0 : 1;
}
