// The defaults command: what this build's unnamed calls resolve to, class by
// class, read from the table the build carries.
//
// A consumer writes a call and names no policy; the build answers it with the
// combination the seam file in force carries for that class
// (include/boys/boys_build_defaults.hpp). The answer is a compile-time type,
// which is what makes it free - and which is why nothing printed it. A reader
// asking what they get had to find the header, expand its macro list by hand and
// look up each cell; a build configured with BOYS_BUILD_DEFAULTS had no way at
// all to see the choices it was compiled with. This program is that answer, one
// command, and the design that names it is
// .claude/lane-status/api-and-defaults-design.md, "One command prints what this
// build will do".
//
// WHAT IT READS, AND WHY THAT IS THE WHOLE POINT
//
// The table, and not a copy of it. A class is a (device, precision, shape)
// triple; the row is the specialization the compiler selects for that triple
// (boys::detail::DefaultPolicyRow, include/boys/boys.hpp); the axes printed are
// the ones that specialization resolves to, and the bound is the figure the
// library's own accessor answers for those axes. A build that replaced the seam
// file prints the replacement's rows for the same reason the library compiles
// them: the include path delivers the replacement to this translation unit
// exactly as it delivers it to the library's own instantiations, and
// BOYS_BUILD_DEFAULTS_REPLACED is what says which of the two happened.
//
// THE CLASSES ARE WALKED, NOT LISTED. This program asks the table about every
// class the three enumerations name - Device, Precision and Shape - and prints
// the ones the table carries. A class it does not carry is reported as that, and
// is not printed as a policy, because there is no policy to print: asking for
// the default of such a class is a compile error (the assertion in
// DefaultPolicyFor, include/boys/boys.hpp), which is the seam's contract rather
// than a gap in this report.
//
// WHAT IT IS NOT
//
// Not a measurement and not a timing: nothing is evaluated, nothing is called
// but the reporting accessors, so the same build prints the same report on every
// machine - which is why it is registered as a ctest case rather than left as a
// local run. It is not the option probe either: the probe ranks combinations on
// the machine it runs on (benchmarks/boys_option_probe.cpp), and this prints the
// choice a build made, which is data the probe produced rather than data it can
// produce.
//
// THE CHECKS, WHICH ARE WHY THE EXIT STATUS IS NOT DECORATION
//
//  - the lane table (BoysLaneContracts(), one row per lane) and the walk below
//    agree on how many lanes this build has;
//  - a class's figure composed from the lane row the report prints -
//    (multiplicand + the named form's term) x m + additive - is the figure the
//    accessor returned, to the bit. A report that prints a multiplicand and a
//    figure that do not compose is a report a reader cannot use;
//  - DefaultGuarantee<Precision, Shape>() and the axis-taking accessor, asked
//    with the axes the class resolves to, are one figure for a host class. That
//    equality is the seam's own promise (include/boys/boys.hpp), and it is read
//    here rather than assumed.
//
// Failures are printed as lines beside the report, never instead of it, and the
// exit status is 1 when one fires.
//
// WHAT IT DOES NOT CHECK, so that a green run is read for what it is: whether
// every class an ENTRY reaches has a row. A class with no row cannot be asked
// for by this program at all - reading its default is the compile error the
// report describes - and the entries are declarations this program does not see.
// That reading is the source-text checker's (tools/check_default_rows_have_entries.py,
// which reads the table against the entries) and the link test's
// (tests/boys_defaults_link_test.cpp, which calls every default), not this
// report's.
//
// WHERE IT IS BUILT
//
// CMakeLists.txt, the tool block in BOYS_BUILD_TESTS: target boys-defaults, and
// the ctest case of the same name, which runs the report rather than only
// building it.
//
// Run:  cmake --build <build> --target boys-defaults
//       <build>/boys-defaults
//       ctest --test-dir <build> -R boys-defaults

#include "boys/boys.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <span>
#include <type_traits>

namespace {

using namespace boys;

// The path of the seam file in force, and - for a build that replaced it - the
// copy the compiler read on the include path before include/. Both come from the
// configure rather than from __FILE__: a header's own path is whatever the
// include directive spelled, and a report that cannot say which table produced
// it is the defect this whole seam exists to remove.
#ifndef BOYS_DEFAULTS_HEADER_PATH
#define BOYS_DEFAULTS_HEADER_PATH "(not recorded: this translation unit was not compiled by the boys CMake build)"
#endif
#ifndef BOYS_DEFAULTS_HEADER_COPY
#define BOYS_DEFAULTS_HEADER_COPY ""
#endif

// --- the names the library does not publish --------------------------------
//
// The fit route, the scheme, the packing axis, the partition and the division
// form each have a name from the library itself, and this report prints those
// (RouteName below, EvalSchemeName, PackAxisName, GranularityName,
// DivisionFormName). The device, the shape and the compute budget have none at
// this revision, so they are named here - except the precision lane, whose name
// is read from the library's own lane row. Each switch below names every
// enumerator of its axis and has no default arm, so a member a later change adds
// is a missing arm here - a failed build on the toolchains that build this tree
// with -Werror - rather than a class this report quietly omits.

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

// The three enumerations a class key is made of, one walk per axis. C++ has no
// way to walk an enumeration, so the members are written out here and the
// `static_assert` walks below hold each list to the names above; a member added
// to the enumeration and left out of the list is the one case neither catches,
// which is why the report prints how many classes it scanned and what the table
// answered rather than only the classes it found.
#define BOYS_DEFAULTS_DEVICES(X) X(kHost) X(kDevice)
#define BOYS_DEFAULTS_PRECISIONS(X)                        \
    X(kFp64) X(kFp32) X(kFp16) X(kFp32Device) X(kFp64Device) X(kFp16Device)
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
constexpr std::array<Precision, 6> kPrecisions{BOYS_DEFAULTS_PRECISIONS(BOYS_DEFAULTS_PRECISION_ENTRY)};
#undef BOYS_DEFAULTS_PRECISION_ENTRY

#define BOYS_DEFAULTS_SHAPE_ENTRY(Enumerator) Shape::Enumerator,
constexpr std::array<Shape, 5> kShapes{BOYS_DEFAULTS_SHAPES(BOYS_DEFAULTS_SHAPE_ENTRY)};
#undef BOYS_DEFAULTS_SHAPE_ENTRY

// A build whose seam file carries a row list names its classes there, and every
// one of them is a class the walk above must reach: a row keyed by a member the
// walk does not carry would be a row this report omits while a reader saw it in
// the header - the failure this report exists to make impossible. A replacement
// that names such a class fails the build here instead, and the line says which
// list to extend. The block is compiled only where there is a list to check,
// which is also why the walk's members are not asserted against themselves.
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
/// A host class reads \c DefaultGuarantee<Precision, Shape>(), which is the
/// accessor the seam documents for "the bound a class's default policy carries".
/// A device class has no such name yet - the same accessor reads a host class and
/// says so (include/boys/boys.hpp, the device lane's rows are owed work) - so it
/// reads the axis-taking accessor with the axes its own row resolves to, which
/// is the figure the same table answers.
///
/// \tparam kDevice the class's device
/// \tparam kPrecision the class's lane
/// \tparam kShape the class's shape
/// \returns the figure and whether this build carries the combination
template <Device kDevice, Precision kPrecision, Shape kShape>
AccuracyFigure ClassGuarantee() noexcept {
    if constexpr (kDevice == Device::kHost)
    {
        return DefaultGuarantee<kPrecision, kShape>();
    }
    else
    {
        using Policy = DefaultPolicy<kPrecision, kShape, kDevice>;

        return BoysAccuracyGuaranteed(kPrecision, Policy::kRoute, Policy::kScheme, Policy::kPack,
                                      Policy::kGranularity, Policy::kDivision);
    }
}

/// The arithmetic the library states a figure by: the lane's multiplicand plus
/// the term the named form adds beside it, times the multiplier of the one
/// accuracy this library serves, plus the lane's additive term.
///
/// It is the accessor's own expression (src/boys.cpp, \c BoysAccuracyGuaranteed),
/// written out so the report can print the terms a reader recomputes the figure
/// with and can hold the two to each other.
///
/// \param lane the lane's contract row
/// \param form the class's division form
/// \returns the figure the terms compose to
double ComposedFigure(const LaneContractInfo& lane, DivisionForm form) noexcept {
    const double formTerm = form == DivisionForm::kPlainReciprocal ? lane.plainAdditive : 0.0;

    return kBoysFullAccuracyMultiplier * (lane.bound + formTerm) + lane.additive;
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

    const double formTerm =
        Policy::kDivision == DivisionForm::kPlainReciprocal ? lane.plainAdditive : 0.0;

    std::printf("bound %.6g = (%.6g + %.6g) x m %.6g + %.6g\n",
                figure.value,
                lane.bound,
                formTerm,
                kBoysFullAccuracyMultiplier,
                lane.additive);

    if (ComposedFigure(lane, Policy::kDivision) != figure.value)
    {
        std::printf("    the terms above do not compose to the figure the accessor returned: "
                    "composed %.17g, returned %.17g\n",
                    ComposedFigure(lane, Policy::kDivision),
                    figure.value);
        ++coverage.compositionFailed;
    }

    // The two named readings of one figure, held to each other where both exist:
    // DefaultGuarantee is documented as the same figure, from the same table, as
    // the axis-taking accessor asked with the policy's own axes.
    if constexpr (kDevice == Device::kHost)
    {
        const AccuracyFigure direct = BoysAccuracyGuaranteed(kPrecision,
                                                             Policy::kRoute,
                                                             Policy::kScheme,
                                                             Policy::kPack,
                                                             Policy::kGranularity,
                                                             Policy::kDivision);

        if (!direct.available || direct.value != figure.value)
        {
            std::printf("    DefaultGuarantee and the axes the class resolves to disagree: "
                        "%.17g against %.17g\n",
                        figure.value,
                        direct.value);
            ++coverage.accessorsDisagree;
        }
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

    // The gate is the table's own answer, asked of the primary template - which
    // carries no Type for a class it does not name - rather than of
    // DefaultPolicyFor, whose assertion makes the same question a compile error
    // (which is what a call site gets, and is why this program can report the
    // class instead).
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

#if defined(BOYS_BUILD_DEFAULTS_SHIPPED)
    std::printf("                 the committed file (BOYS_BUILD_DEFAULTS_SHIPPED): the shipped "
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
    std::printf("                 neither state: a seam file defines BOYS_BUILD_DEFAULTS_SHIPPED or "
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
                "                 names below, one row per host class, every cell this build's own "
                "choice\n");
#endif

    std::printf("seam's five      fit route %s | eval scheme %s | packing axis %s\n"
                "                 division form %s | fit granularity %s\n",
                RouteName(BOYS_BUILD_DEFAULT_FIT_ROUTE),
                EvalSchemeName(BOYS_BUILD_DEFAULT_EVAL_SCHEME),
                PackAxisName(BOYS_BUILD_DEFAULT_PACK_AXIS),
                DivisionFormName(BOYS_BUILD_DEFAULT_DIVISION_FORM),
                GranularityName(BOYS_BUILD_DEFAULT_FIT_GRANULARITY));
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
        std::printf("  FAILED: %zu host classes where DefaultGuarantee and the class's own axes "
                    "disagree\n",
                    coverage.accessorsDisagree);
        ++failures;
    }

    if (failures == 0)
    {
        std::printf("  all %zu carried classes resolved: every reference figure is available, the "
                    "printed\n  multiplicity composes to it to the bit, and a host class's "
                    "DefaultGuarantee and its own\n  axes are one figure. The axes are the table's "
                    "own specializations; nothing was evaluated\n  and nothing was timed\n",
                    carried);
    }

    return failures == 0 ? 0 : 1;
}
