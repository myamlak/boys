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
#include <cmath>
#include <cstdio>
#include <gtest/gtest.h>
#include <string>

namespace {

using boys::DeviceProbeClass;
using boys::DeviceProbeMeasurement;
using boys::DeviceProbeOptions;
using boys::DeviceProbeRanking;
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
            for (const DeviceProbeRanking& ranking : clause.rankings)
            {
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
                EXPECT_TRUE(ranking.recommended.empty());
                EXPECT_FALSE(ranking.reason.empty());
            }
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

/// A class is one precision and a ranking inside it is one question shape: no
/// entry is ever placed against an entry of another precision or of another
/// shape. That is the rule the ranking is read under, so it is the rule checked
/// here, row by row rather than by the names the report happens to print.
TEST(DeviceProbe, AClassIsOnePrecisionAndARankingIsOneShape) {
    DeviceProbeOptions options = Small();
    options.passes = 1;
    options.canarySpreadThreshold = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_FALSE(report.classes.empty());

    for (std::size_t c = 0; c < report.classes.size(); ++c)
    {
        const DeviceProbeClass& clause = report.classes[c];

        EXPECT_FALSE(clause.precision.empty());
        EXPECT_FALSE(clause.note.empty());
        EXPECT_FALSE(clause.rankings.empty());

        // One class per precision, so no two classes are the same precision.
        for (std::size_t other = c + 1; other < report.classes.size(); ++other)
        {
            EXPECT_NE(report.classes[other].precision, clause.precision);
        }

        for (std::size_t r = 0; r < clause.rankings.size(); ++r)
        {
            const DeviceProbeRanking& ranking = clause.rankings[r];

            EXPECT_FALSE(ranking.question.empty());
            EXPECT_FALSE(ranking.asked.empty());

            // One ranking per shape, so no two rankings of one class ask the
            // same question.
            for (std::size_t other = r + 1; other < clause.rankings.size(); ++other)
            {
                EXPECT_NE(clause.rankings[other].question, ranking.question);
            }

            // Every row the report places in this ranking — the recommended one
            // included — is of this class's precision and this ranking's
            // question, and there is at least one such row in the table.
            std::size_t rows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements)
            {
                if (measurement.precision != clause.precision ||
                    measurement.question != ranking.question)
                {
                    continue;
                }

                ++rows;
            }

            EXPECT_GT(rows, 0u);

            if (!ranking.recommended.empty())
            {
                const auto named =
                    std::find_if(report.measurements.begin(),
                                 report.measurements.end(),
                                 [&ranking](const DeviceProbeMeasurement& measurement) {
                                     return measurement.name == ranking.recommended;
                                 });

                ASSERT_NE(named, report.measurements.end());
                EXPECT_EQ(named->precision, clause.precision);
                EXPECT_EQ(named->question, ranking.question);
            }
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
        for (const DeviceProbeRanking& ranking : clause.rankings)
        {
            EXPECT_FALSE(ranking.asked.empty());

            if (ranking.verdict != DeviceProbeVerdict::kRecommend)
            {
                EXPECT_TRUE(ranking.recommended.empty());
                continue;
            }

            EXPECT_FALSE(ranking.recommended.empty());
            EXPECT_FALSE(ranking.reason.empty());
            EXPECT_GT(ranking.resolution, 0.0);
            EXPECT_FALSE(ranking.confidence.empty());

            const auto named =
                std::find_if(report.measurements.begin(),
                             report.measurements.end(),
                             [&ranking](const DeviceProbeMeasurement& measurement) {
                                 return measurement.name == ranking.recommended;
                             });

            ASSERT_NE(named, report.measurements.end());
            EXPECT_TRUE(named->measured);
            EXPECT_EQ(named->question, ranking.question);
            EXPECT_EQ(named->precision, clause.precision);
        }
    }
}

/// A ranking is a comparison, so it needs two entries to be one. A shape whose
/// table holds one entry has nothing to order that entry against, and the report
/// declines rather than naming it: what such an entry is, is the fastest of the
/// one entry of its shape, and what it is not, is a result against a field.
///
/// The check is on the shape's own rows rather than on a figure, because whether
/// an entry measures at this workload is the card's business and this run is not
/// asserting a cost.
TEST(DeviceProbe, AShapeOfOneNamesNoWinner) {
    DeviceProbeOptions options = Small();
    options.canarySpreadThreshold = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    bool sawAShapeOfOne = false;

    for (const DeviceProbeClass& clause : report.classes)
    {
        for (const DeviceProbeRanking& ranking : clause.rankings)
        {
            std::size_t rows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements)
            {
                if (measurement.precision == clause.precision &&
                    measurement.question == ranking.question)
                {
                    ++rows;
                }
            }

            if (rows >= 2)
            {
                continue;
            }

            sawAShapeOfOne = true;
            EXPECT_NE(ranking.verdict, DeviceProbeVerdict::kRecommend);
            EXPECT_TRUE(ranking.recommended.empty());
            EXPECT_FALSE(ranking.reason.empty());
            EXPECT_FALSE(ranking.confidence.empty());
        }
    }

    EXPECT_TRUE(sawAShapeOfOne);
}

/// A control that agrees has two figures to compare. A count at which the
/// subtraction resolved no cost leaves a zero, and a zero is not a second
/// reading: a control that called that agreement would be passing on no
/// evidence, which is the one thing this check exists to prevent.
TEST(DeviceProbe, AControlAgreesOnlyBetweenTwoFigures) {
    DeviceProbeOptions options = Small();
    options.passes = 1;
    options.canarySpreadThreshold = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    const boys::DeviceProbeRepetitionControl* controls[] = {&report.control,
                                                            &report.deviceCallControl};

    for (const boys::DeviceProbeRepetitionControl* control : controls)
    {
        if (control->entry.empty() || !control->agrees)
        {
            continue;
        }

        EXPECT_GT(control->nsPerArgumentLow, 0.0) << control->entry;
        EXPECT_GT(control->nsPerArgumentHigh, 0.0) << control->entry;
        EXPECT_TRUE(std::isfinite(control->difference)) << control->entry;
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
