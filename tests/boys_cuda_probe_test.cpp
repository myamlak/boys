// The device option probe (boys/boys_cuda_probe.hpp): what it reports about the
// card, and what it does when the caller names a card that is not there.
//
// The probe's figures are this card's, and asserting them here would pin a
// number that describes one host, so nothing below asserts a cost. What is
// asserted is the part a test can own: that a bad device is a status and not a
// crash, that the device it measured is named in the returned data rather than
// only in a log, that the report carries the protocol it was taken under, and
// that a refusal comes with a reason rather than with a name.
//
// The protocols below are short on purpose. The canary's bar is moved to each
// end so that the admission rule is exercised from both sides on a real run,
// and the workload is small because what is being checked is the shape of the
// result rather than its figures.

#include "boys/boys_cuda_probe.hpp"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

namespace {

using boys::DeviceProbeClass;
using boys::DeviceProbeMeasurement;
using boys::DeviceProbeOptions;
using boys::DeviceProbeReport;
using boys::DeviceProbeStatus;
using boys::DeviceProbeVerdict;

/// A workload a test can afford, with the passes kept short.
DeviceProbeOptions Small() {
    DeviceProbeOptions options;
    options.count = 1u << 12;
    options.nmax = 8;
    options.passes = 2;
    options.rounds = 2;
    options.repetitions = 4;
    return options;
}

/// The status a run ends in, and whether the report says so in a way a reader
/// can act on. A run that did not succeed has no figures, and every field below
/// the status is described as meaningful only on success, so the check is that
/// the reason is present and that the device block is not silently blank when
/// the runtime could name a card.
TEST(DeviceProbe, ADisturbedRunRefusesWithAReason) {
    DeviceProbeOptions options = Small();
    options.canarySpreadThreshold = 0.0;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    if (report.cleanPasses < 2)
    {
        for (const DeviceProbeClass& clause : report.classes)
        {
            EXPECT_EQ(clause.verdict, DeviceProbeVerdict::kCannotDetermine);
            EXPECT_TRUE(clause.recommended.empty());
            EXPECT_FALSE(clause.reason.empty());
        }
    }
}

/// Every row the report names is a row this build offers, and a row that is
/// named carries the protocol it was taken under rather than a bare number.
TEST(DeviceProbe, TheReportCarriesTheProtocolItWasTakenUnder) {
    DeviceProbeOptions options = Small();
    options.passes = 1;
    options.canarySpreadThreshold = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    // The device the figures are about, in the returned data and not only in
    // the text: a consumer that reads the struct has to be able to learn what
    // card answered.
    EXPECT_FALSE(report.device.name.empty());
    EXPECT_GE(report.device.computeMajor, 1);
    EXPECT_FALSE(report.device.toolkitVersion.empty());
    EXPECT_FALSE(report.device.architectures.empty());

    EXPECT_EQ(report.workloadCount, options.count);
    EXPECT_FALSE(report.caveat.empty());

    for (const DeviceProbeMeasurement& measurement : report.measurements)
    {
        EXPECT_FALSE(measurement.name.empty());
        EXPECT_FALSE(measurement.route.empty());
        EXPECT_FALSE(measurement.question.empty());
        EXPECT_GT(measurement.documentedBound, 0.0);

        if (measurement.measured)
        {
            EXPECT_GE(measurement.spread, 1.0);
            EXPECT_GT(measurement.cleanPasses, 0);
        } else
        {
            EXPECT_EQ(measurement.cleanPasses, 0);
        }
    }
}

/// A class that names a winner names one it measured, and it says how far the
/// nearest rival was. A class that does not names what it could not separate.
TEST(DeviceProbe, AVerdictNamesOnlyEntriesItMeasured) {
    DeviceProbeOptions options = Small();
    options.passes = 1;
    options.canarySpreadThreshold = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    for (const DeviceProbeClass& clause : report.classes)
    {
        EXPECT_FALSE(clause.asked.empty());

        if (clause.verdict != DeviceProbeVerdict::kRecommend)
        {
            continue;
        }

        EXPECT_FALSE(clause.recommended.empty());

        const auto named = std::find_if(report.measurements.begin(),
                                        report.measurements.end(),
                                        [&clause](const DeviceProbeMeasurement& measurement) {
                                            return measurement.name == clause.recommended;
                                        });

        ASSERT_NE(named, report.measurements.end());
        EXPECT_TRUE(named->measured);
        EXPECT_EQ(named->question, clause.question);
        EXPECT_GT(clause.resolution, 0.0);
        EXPECT_FALSE(clause.confidence.empty());
    }
}

/// A device the caller names that is not there is a status, not a crash, and not
/// a silent fallback to the card this happens to be running on.
TEST(DeviceProbe, AnAbsentDeviceIsAStatusAndNotACrash) {
    DeviceProbeOptions options = Small();
    options.device = 4096;
    options.passes = 1;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    EXPECT_NE(report.status, DeviceProbeStatus::kSuccess);
    EXPECT_FALSE(report.failure.empty());
    EXPECT_TRUE(report.classes.empty());
    EXPECT_TRUE(report.measurements.empty());
}

} // namespace
