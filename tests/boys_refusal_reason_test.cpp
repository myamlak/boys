// The refusal's reason, held to: a combination this revision does not carry, and a
// device option this build does not serve, must say why - and this file is what makes a
// refusal that says nothing fail rather than pass.
//
// THE DEFECT THIS FILE EXISTS FOR
//
// An unsupported combination carries a serious, checkable reason, and a refusal without
// one must not read like a refusal with one. Two places left the reason optional and
// nothing checked it:
//
//   * `CombinationCoverage::reason` is `""` by default beside
//     `verdict = ToleranceVerdict::kNotCarried` (include/boys/boys.hpp), so a
//     default-constructed answer is a refusal with an empty reason, and a reader cannot
//     tell it from an answer that was refused deliberately with a reason.
//     `QueryCombination`'s documentation promises "a reason where this revision does not
//     carry the combination": the documentation promised it, the type did not require it
//     and no check read it.
//
//   * `DeviceOptionInfo::refusedBecause` is "why not, when built is false; nullptr
//     otherwise" (include/boys/boys_cuda_options.hpp), and the device probe's own report
//     replaced a null one with "not served by this build" (src/boys_cuda_probe.cpp,
//     EnumerateEntries). An option refusing with no stated reason was reported under a
//     reason-shaped string that is not the library's reason - and a substitute is worse
//     than a blank, because it reads as the library's own sentence. This check removes
//     the supply for that substitution rather than performing it: a row that is not
//     built and states no reason fails here, naming the row, before any report can dress
//     it up. Nothing in this file fills a reason in; filling one in is the substitution
//     the ruling removes.
//
// WHAT IT WALKS, AND WHY THAT IS THE SPACE
//
// Both halves are walked rather than sampled, and both walks are derived from the
// library's own reporting API rather than written out here:
//
//   * the host half crosses `BoysLaneContracts()`, `BoysFitRoutes()`, `BoysEvalSchemes()`,
//     `BoysPackAxes()` and `BoysFitGranularities()` - every combination of the members
//     this revision reports - and, on each axis, the value one past the count of those
//     members. That value is the first one the enumeration does not name, and every
//     carrier's guard refuses it (the tests are `>=` the reported count), so the walk
//     reaches the refusal path in every configuration rather than only in a build whose
//     tables happen to refuse a combination of members.
//
//   * the device half walks every row of `BoysDeviceOptions()`, which is the device
//     lane's option space as the lane itself reports it.
//
// THE REQUIREMENT, READ IN BOTH DIRECTIONS
//
// The two fields state which way their reason runs, and this check reads each the way
// its own comment does:
//
//   * where the verdict is `kNotCarried` the reason is present and not empty, and where
//     there is a verdict the reason is empty ("why there is no verdict, empty where
//     there is one", CombinationCoverage::reason);
//   * where a row is not built `refusedBecause` is present and not empty, and where it
//     is built it is null ("why not, when built is false; nullptr otherwise",
//     DeviceOptionInfo::refusedBecause).
//
// An empty string is not a reason, so the requirement is non-empty rather than
// non-null: the pointer being set is what the field documents, and having something to
// read is what the ruling is about.
//
// HOW IT FAILS
//
// A gtest failure naming the combination or the row it broke on, with the walk's counts
// printed in the same run so a passing transcript says how much was walked and how much
// of it refused.
//
// WHAT IT CANNOT SEE, so that "this check passes" is read for what it is
//
//   * the device half runs only where the CUDA lane is built, because
//     `BoysDeviceOptions()` is defined by that lane (src/boys_cuda.cpp). The target that
//     states `BOYS_REFUSAL_REASON_DEVICE` (CMakeLists.txt) is where it is compiled in;
//     a build without the lane runs the host half alone and says so in its transcript.
//   * no row of the device space is unbuilt in a build whose fp16 seam is open, which
//     every build of the CUDA lane states (`BoysFp16`), so the unbuilt direction of the
//     device requirement is a statement about the space's shape there rather than a
//     refusal that configuration reaches. The count of unbuilt rows is printed, so a run
//     whose device half was vacuous says that too.

#include "boys/boys.hpp"

#if defined(BOYS_REFUSAL_REASON_DEVICE)
#include "boys/boys_cuda_options.hpp"
#endif

#include <cstddef>
#include <cstdio>
#include <string>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>

namespace {

/// The error the host walk asks at: positive and finite, and below every figure this
/// library holds (the smallest are near 1e-14), so a combination this build carries
/// answers with a verdict and no reason and one it does not carry answers with no
/// verdict and a reason. The request decides no part of the requirement under test; it
/// is chosen to keep the two states as far apart as the library keeps them.
constexpr double kWalkRequest = 1e-300;

/// The name a report prints a lane under, read from the library's own row, and the value
/// beside it where no row of that table names it.
std::string LaneName(boys::Precision precision) {
    for (const boys::LaneContractInfo& row : boys::BoysLaneContracts()) {
        if (row.precision == precision) {
            return row.name;
        }
    }

    char text[80];
    std::snprintf(text, sizeof(text), "no lane of this library names the value %d",
                  static_cast<int>(precision));
    return text;
}

/// The name a report prints a fit route under, read from the library's own table, and the
/// value beside it where no row names it.
std::string RouteName(boys::FitRoute route) {
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes()) {
        if (row.route == route) {
            return row.name;
        }
    }

    char text[80];
    std::snprintf(text, sizeof(text), "no fit route of this library names the value %d",
                  static_cast<int>(route));
    return text;
}

/// A combination as one line: each member under the name the library's own table or
/// accessor prints it with. The scheme, axis and granularity accessors answer "unknown"
/// for a value outside their enumerations, which is itself the statement that the value
/// is not a member.
std::string CombinationName(boys::Precision precision,
                            boys::FitRoute route,
                            boys::EvalScheme scheme,
                            boys::PackAxis axis,
                            boys::FitGranularity granularity) {
    char text[256];
    std::snprintf(text,
                  sizeof(text),
                  "%s, %s, %s, %s, %s",
                  LaneName(precision).c_str(),
                  RouteName(route).c_str(),
                  boys::EvalSchemeName(scheme),
                  boys::PackAxisName(axis),
                  boys::GranularityName(granularity));
    return std::string(text);
}

/// The values one axis of the walk takes: every member of it this revision reports, and
/// then the count of those members.
///
/// The count is read from the reporting API rather than written here, and it is the first
/// value the axis's enumeration does not name: a table reports one row per member it
/// carries, so a value equal to the count is past the last of them, and the carriers
/// refuse it (`>=` the count, in the guards of src/boys.cpp). A walk that then visits
/// that value per axis reaches the refusal path without being handed a hand-written list
/// of combinations and without depending on which cells this revision's tables refuse.
template <typename Row, typename Read>
auto AxisValues(std::span<const Row> rows, Read read) {
    using Member = std::invoke_result_t<Read, const Row&>;
    std::vector<Member> values;
    values.reserve(rows.size() + 1);

    for (const Row& row : rows) {
        values.push_back(read(row));
    }

    values.push_back(static_cast<Member>(values.size()));
    return values;
}

/// The host half's walk, and what it found.
struct CombinationWalk {
    std::size_t combinations = 0; ///< combinations asked of QueryCombination
    std::size_t refused = 0; ///< answers with no verdict
    std::vector<std::string> violations; ///< the combinations that broke the requirement
};

/// Every combination `QueryCombination` can be asked about, from the axes the library
/// itself reports, read at the one request above.
CombinationWalk WalkTheCombinations() {
    CombinationWalk walk;

    const std::vector<boys::Precision> lanes =
        AxisValues(boys::BoysLaneContracts(),
                   [](const boys::LaneContractInfo& row) { return row.precision; });
    const std::vector<boys::FitRoute> routes =
        AxisValues(boys::BoysFitRoutes(),
                   [](const boys::FitRouteInfo& row) { return row.route; });
    const std::vector<boys::EvalScheme> schemes =
        AxisValues(boys::BoysEvalSchemes(),
                   [](const boys::EvalSchemeInfo& row) { return row.scheme; });
    const std::vector<boys::PackAxis> axes =
        AxisValues(boys::BoysPackAxes(),
                   [](const boys::PackAxisInfo& row) { return row.axis; });
    const std::vector<boys::FitGranularity> granularities =
        AxisValues(boys::BoysFitGranularities(),
                   [](const boys::FitGranularityInfo& row) { return row.granularity; });

    for (const boys::Precision precision : lanes) {
        for (const boys::FitRoute route : routes) {
            for (const boys::EvalScheme scheme : schemes) {
                for (const boys::PackAxis axis : axes) {
                    for (const boys::FitGranularity granularity : granularities) {
                        ++walk.combinations;

                        const boys::CombinationCoverage answer = boys::QueryCombination(
                            precision, route, scheme, axis, granularity, kWalkRequest);
                        const bool refuse =
                            answer.verdict == boys::ToleranceVerdict::kNotCarried;
                        const bool states = answer.reason != nullptr && answer.reason[0] != '\0';

                        if (refuse) {
                            ++walk.refused;
                        }

                        if (refuse != states) {
                            walk.violations.push_back(
                                CombinationName(precision, route, scheme, axis, granularity) +
                                (refuse ? ": refused with no reason"
                                        : ": a verdict beside a refusal's reason"));
                        }
                    }
                }
            }
        }
    }

    return walk;
}

/// The first few violations as one message, with the total, so a failure names what broke
/// without printing the whole walk when a shared path broke every combination behind it.
std::string Violations(const std::vector<std::string>& violations) {
    constexpr std::size_t kShown = 8;
    std::string text;
    const std::size_t shown = violations.size() < kShown ? violations.size() : kShown;

    for (std::size_t i = 0; i < shown; ++i) {
        text += "\n  ";
        text += violations[i];
    }

    if (violations.size() > shown) {
        text += "\n  ... and ";
        text += std::to_string(violations.size() - shown);
        text += " more combination(s)";
    }

    return text;
}

#if defined(BOYS_REFUSAL_REASON_DEVICE)

/// The device half's walk, and what it found.
struct DeviceWalk {
    std::size_t rows = 0; ///< rows of BoysDeviceOptions()
    std::size_t unbuilt = 0; ///< rows this build does not serve
    std::vector<std::string> violations; ///< the rows that broke the requirement
};

/// Every member of the device option space, read in both directions of the field's own
/// sentence: a row that is not built states why, and a row that is built states nothing.
DeviceWalk WalkTheDeviceOptions() {
    DeviceWalk walk;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions()) {
        ++walk.rows;

        const std::string name =
            row.name != nullptr ? std::string(row.name) : std::string("a row with no name");
        const bool states = row.refusedBecause != nullptr && row.refusedBecause[0] != '\0';

        if (!row.built) {
            ++walk.unbuilt;
        }

        if (!row.built != states) {
            walk.violations.push_back(
                name + (!row.built ? ": not built, and no reason is stated"
                                   : ": built, and states a refusal reason"));
        }
    }

    return walk;
}

#endif

} // namespace

TEST(RefusalReasonTest, EveryRefusalStatesItsReason) {
    const CombinationWalk combinations = WalkTheCombinations();

    std::printf("  the refusal's reason, host half: %zu combination(s) asked of "
                "QueryCombination over the members this\n    build reports and the value one "
                "past each axis's count; %zu refused, %zu broke the requirement\n",
                combinations.combinations,
                combinations.refused,
                combinations.violations.size());

    // Non-vacuity: the axes are reported and the walk reached the refusal path. The
    // second is not an accident of this revision's tables - every axis carries the value
    // one past its count, so a configuration that refused nothing else would still be
    // walked past a refusal here.
    EXPECT_GT(combinations.combinations, 1u);
    EXPECT_GT(combinations.refused, 0u)
        << "the walk reached no refusal at all, so nothing above tested a refusal's reason";
    EXPECT_TRUE(combinations.violations.empty())
        << "a combination's reason does not run with its verdict:" << Violations(
               combinations.violations);

#if defined(BOYS_REFUSAL_REASON_DEVICE)
    const DeviceWalk device = WalkTheDeviceOptions();

    std::printf("  the refusal's reason, device half: %zu row(s) of BoysDeviceOptions(), %zu "
                "not built, %zu broke the\n    requirement\n",
                device.rows,
                device.unbuilt,
                device.violations.size());

    EXPECT_GT(device.rows, 0u);
    EXPECT_TRUE(device.violations.empty())
        << "a device option's reason does not run with whether it is built:"
        << Violations(device.violations);
#else
    std::printf("  the refusal's reason, device half: not built in this configuration (the "
                "CUDA lane is off,\n    so BoysDeviceOptions() has no definition to link); the "
                "arm runs where the lane is built\n");
#endif
}
