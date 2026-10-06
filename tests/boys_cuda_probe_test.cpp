// The device option probe (boys/boys_cuda_probe.hpp): what it reports about the
// card, and what it does when the caller names a card that is not there.
//
// No cost is asserted here, a figure being the property of the card it was taken
// on. What is asserted is the shape of the report: ratios formed inside a round
// and pooled over every round, a pass the canary flagged used rather than
// dropped, a canary no pass read told apart from a silent one, every name taken
// from a clock rather than off the library's tables, and a class being one
// precision and one question shape.
//
// ATieNamesEveryRivalAndTheBandItFellIn is the exception: it hunts a tie through
// the refinement stage and takes tens of minutes on a real card.

#include "boys/boys_cuda_probe.hpp"

#include "boys/boys_cuda_options.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <span>
#include <string>

namespace {

using boys::DeviceProbeClass;
using boys::DeviceProbeDefaultHow;
using boys::DeviceProbeMeasurement;
using boys::DeviceProbeOptions;
using boys::DeviceProbePass;
using boys::DeviceProbeRanking;
using boys::DeviceProbeReport;
using boys::DeviceProbeStatus;
using boys::DeviceProbeVerdict;

/// A workload a test can afford: two passes of two rounds is four pooled rounds,
/// the least a lower-quartile band can be formed from.
DeviceProbeOptions Small() {
    DeviceProbeOptions options;
    options.count = 1u << 12;
    options.nmax = 8;
    options.passes = 2;
    options.rounds = 2;
    options.repetitions = 4;
    return options;
}

/// The canary's alarm flags a pass and neither drops nor admits it: the canary is
/// fixed work read by a device clock, so it measures the clock as much as the load.
TEST(DeviceProbe, APassAboveTheCanaryAlarmIsFlaggedAndStillUsed) {
    DeviceProbeOptions flagged = Small();
    flagged.canarySpreadAlarm = 0.0;

    const DeviceProbeReport wide = boys::RunDeviceOptionProbe(flagged);

    ASSERT_EQ(wide.status, DeviceProbeStatus::kSuccess);

    EXPECT_EQ(wide.passes.size(), static_cast<std::size_t>(flagged.passes));
    EXPECT_EQ(wide.passesWithinAlarm, 0);
    EXPECT_EQ(wide.pairedRounds, flagged.passes * flagged.rounds);

    // The three counters partition the passes, and with the alarm at zero every pass is above it.
    EXPECT_EQ(wide.passesWithinAlarm + wide.passesAboveAlarm + wide.passesWithoutCanary,
              static_cast<int>(wide.passes.size()));
    EXPECT_EQ(wide.passesAboveAlarm + wide.passesWithoutCanary, flagged.passes);

    for (const boys::DeviceProbePass& pass : wide.passes) {
        EXPECT_EQ(pass.canaryWide,
                  pass.canaryMeasured && pass.canarySpread > flagged.canarySpreadAlarm);
        EXPECT_TRUE(pass.canaryMeasured);
        EXPECT_TRUE(pass.canaryWide);
    }

    DeviceProbeOptions quiet = Small();
    quiet.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport within = boys::RunDeviceOptionProbe(quiet);

    ASSERT_EQ(within.status, DeviceProbeStatus::kSuccess);

    EXPECT_EQ(within.passes.size(), static_cast<std::size_t>(quiet.passes));
    EXPECT_EQ(within.passesAboveAlarm, 0);
    EXPECT_EQ(within.passesWithinAlarm + within.passesWithoutCanary, quiet.passes);

    for (const boys::DeviceProbePass& pass : within.passes) {
        EXPECT_FALSE(pass.canaryWide);
    }

    // The flagged run is the same run: the same passes reported and the same rounds
    // pooled, where a discarded flagged pass would show up as fewer pooled rounds.
    EXPECT_EQ(within.pairedRounds, wide.pairedRounds);
    EXPECT_EQ(within.passes.size(), wide.passes.size());
}

/// A canary that never ran is not a canary that stayed quiet: a spread of zero is a
/// clock that did not move, no reading at all is a clock nobody looked at. The report
/// is built rather than run, a device having no way to fail its canary on command.
TEST(DeviceProbe, ACanaryThatDidNotRunIsNotAQuietCanary) {
    // The zero a pass carries out of a run in which no canary read succeeded.
    DeviceProbePass silent;
    silent.seconds = 1.0;
    silent.canaryMeasured = false;
    silent.canaryMedianMs = 0.0;
    silent.canarySpread = 0.0;
    silent.canaryWide = false;
    silent.pairedSpread = 12.5;

    // A pass that did read, beside it: the two have to be told apart in one table.
    DeviceProbePass read = silent;
    read.canaryMeasured = true;
    read.canaryMedianMs = 3.25;
    read.canarySpread = 9.0;
    read.canaryWide = true;

    DeviceProbeReport unread;
    unread.status = DeviceProbeStatus::kSuccess;
    unread.device.name = "a device";
    unread.passes = {silent};
    unread.passesWithoutCanary = 1;
    unread.pairedRounds = 2;
    unread.options.passes = 1;

    const std::string unreadText = boys::FormatDeviceOptionProbe(unread);

    EXPECT_NE(unreadText.find("no pass took a canary reading"), std::string::npos) << unreadText;
    EXPECT_NE(unreadText.find("is not a canary that stayed quiet"), std::string::npos) << unreadText;

    const std::size_t header = unreadText.find("pass  wall_s");
    ASSERT_NE(header, std::string::npos);

    const std::size_t body = unreadText.find('\n', header) + 1;
    const std::size_t tableEnd = unreadText.find("\n\n", body);
    ASSERT_NE(tableEnd, std::string::npos);

    const std::string row = unreadText.substr(body, tableEnd - body);

    // The words, and not a zero: a reading would print 0.0000 here, the failure this test catches.
    EXPECT_NE(row.find("canary not timed"), std::string::npos) << row;
    EXPECT_EQ(row.find("0.0000"), std::string::npos) << row;

    // A measured pass beside it prints figures and its flag, so the words are a missing reading.
    DeviceProbeReport mixed = unread;
    mixed.passes = {silent, read};
    mixed.passesWithoutCanary = 1;
    mixed.passesAboveAlarm = 1;

    // The counters are the table: the pass that took no reading is in the third of them.
    EXPECT_EQ(mixed.passesWithinAlarm + mixed.passesAboveAlarm + mixed.passesWithoutCanary,
              static_cast<int>(mixed.passes.size()));

    const std::string mixedText = boys::FormatDeviceOptionProbe(mixed);

    EXPECT_NE(mixedText.find("3.2500"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("9.00"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("canary wide - reported, used"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("took no canary reading at all"), std::string::npos) << mixedText;
    EXPECT_EQ(mixedText.find("no pass took a canary reading"), std::string::npos) << mixedText;

    // The resolution line carries the canary only where one was read, not a zero otherwise.
    DeviceProbeRanking refused;
    refused.question = "all-orders";
    refused.asked = "every order";
    // At four paired rounds or more the report prints a band; below that it says so instead.
    refused.rounds = 10;
    refused.resolution = 0.25;
    refused.verdict = DeviceProbeVerdict::kCannotDetermine;
    refused.fastestOverall = "device-all-orders-fp64";
    refused.reason = "two entry(s) of this shape could not be placed";
    refused.confidence = "CANNOT DETERMINE: the widest band this shape showed was 25.00%";

    DeviceProbeClass clause;
    clause.precision = "fp64";
    clause.question = "all-orders";
    clause.asked = "every order";
    clause.note = "note";
    clause.ranking = refused;

    DeviceProbeReport withShape = unread;
    withShape.classes = {clause};

    const std::string shapeText = boys::FormatDeviceOptionProbe(withShape);

    EXPECT_NE(shapeText.find("No pass took a canary reading on this run"), std::string::npos)
        << shapeText;
    EXPECT_EQ(shapeText.find("canary spread, 0.00%"), std::string::npos) << shapeText;
}

/// Every figure rests on every pooled round, and the reference entry's own ratio to
/// itself is exactly one with no drift - the one value in the table known
/// independently of the card, which pins the ratios as formed per round and not across
/// rounds. Every row's cost is the report's one anchor scaled by that row's ratio to
/// it, which is why the anchor is carried on the row and not once for the table.
TEST(DeviceProbe, AFigureIsAWithinRoundRatioOverEveryPooledRound) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_GE(report.pairedRounds, 4);
    ASSERT_FALSE(report.referenceEntry.empty());

    std::size_t measured = 0;
    bool sawReference = false;

    for (const DeviceProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured) {
            // An entry with no figure rests on no round rather than claiming the run's rounds.
            EXPECT_EQ(measurement.rounds, 0);
            continue;
        }

        ++measured;

        EXPECT_EQ(measurement.rounds, report.pairedRounds) << measurement.name;
        // Both ends are ratios of two entries timed in one round, so hi cannot be below lo.
        EXPECT_GE(measurement.ratioHi, measurement.ratioLo) << measurement.name;
        // An in-kernel row resolves when the difference it was reduced to stood above
        // its own baseline; a launched row always does, its cell being its own reading
        // rather than a difference that could be floored. The band's lower end may still
        // be the zero it was floored at, so only the figure decides whether it is ordered.
        EXPECT_EQ(measurement.subtractionResolved,
                  measurement.launchedByLibrary || measurement.nsPerArgument > 0.0)
            << measurement.name;
        EXPECT_GE(measurement.spread, 1.0) << measurement.name;

        // The two columns cannot come from different statistics: the reported cost is this
        // row's own anchor scaled by its ratio to it.
        EXPECT_NEAR(measurement.nsPerArgument,
                    measurement.referenceNsPerArgument * measurement.ratioToReference,
                    1e-9 * std::max(1.0, measurement.referenceNsPerArgument)) << measurement.name;
        EXPECT_NEAR(measurement.nsPerArgumentMax,
                    measurement.referenceNsPerArgument * measurement.ratioHi,
                    1e-9 * std::max(1.0, measurement.referenceNsPerArgument)) << measurement.name;

        // The peak column is the entry's own fastest single round, a raw figure under no
        // anchor, beside a reported cost that is a quartile of ratios anchored to the
        // reference: statistics of different quantities, so neither bounds the other in
        // general. The bound is asserted where it is real - on the reference entry.
        EXPECT_TRUE(std::isfinite(measurement.nsPerArgumentPeak)) << measurement.name;
        EXPECT_GE(measurement.nsPerArgumentPeak, 0.0) << measurement.name;

        if (measurement.name != report.referenceEntry) {
            continue;
        }

        sawReference = true;
        // The reference against itself: one in every round by construction, so every
        // quantile is one and the drift is zero.
        EXPECT_DOUBLE_EQ(measurement.ratioToReference, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioLo, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioHi, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioDrift, 0.0);
        EXPECT_DOUBLE_EQ(measurement.nsPerArgument, measurement.referenceNsPerArgument);
        EXPECT_LE(measurement.nsPerArgumentPeak, measurement.nsPerArgument + 1e-12);

        // The report's one anchor is this entry's own figure, and every row carries it:
        // the anchor is carried on the row and not once for the table.
        EXPECT_DOUBLE_EQ(measurement.referenceNsPerArgument, report.referenceNsPerArgument);
    }

    EXPECT_TRUE(sawReference);
    EXPECT_GT(measured, 0u);
}

/// A run too short for a quartile band still ends with one entry per shape that
/// produced a figure, and says both the count it took and the count a band needs.
/// Below four readings a band's two ends are the same reading twice, so such a run
/// can place nothing; what it may not do is leave a shape that produced a figure
/// without a name - its one entry, or the entry the refinement stage's longer protocol led.
TEST(DeviceProbe, AShortRunNamesAnEntryAndSaysTheBandWasNeverFormed) {
    DeviceProbeOptions options = Small();
    // Short in rounds, not in workload: a fixture small in both may not resolve.
    options.count = 1u << 18;
    options.passes = 1;
    options.repetitions = 8;
    options.refinementRuns = 2;
    options.refinementFactor = 2;
    options.canarySpreadAlarm = 1.0e9;
    // One shape of two rows - one launched and one in-kernel - and two shapes of one,
    // so the run holds a name reached by the stage and one reached with no rival at all.
    options.only = {"single-fp64", "device-single-fp64", "single-fp32", "all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_EQ(report.pairedRounds, options.passes * options.rounds);
    ASSERT_LT(report.pairedRounds, 4);
    ASSERT_FALSE(report.classes.empty());

    std::size_t named = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            EXPECT_EQ(ranking.rounds, report.pairedRounds);

            // Nothing can be placed on two readings: the reason names the count it took and needs.
            EXPECT_NE(ranking.reason.find("2 paired round(s)"), std::string::npos)
                << ranking.reason;
            EXPECT_NE(ranking.reason.find("need 4"), std::string::npos) << ranking.reason;

            // No shape is reported as an ordering: its band was never formed.
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kOrdered) << ranking.question;

            // This shape's own rows: a row of another precision or question is no member
            // of this class.
            std::size_t read = 0;
            bool namedIsRead = false;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                    measurement.precision != clause.precision ||
                    measurement.question != ranking.question) {
                    continue;
                }

                ++read;
                namedIsRead = namedIsRead || measurement.name == ranking.recommended;
            }

            if (read == 0) {
                // The one answer a run cannot name from: no entry of this shape produced a figure.
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine)
                    << ranking.question;
                EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;
                EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
                continue;
            }

            // Entries produced figures, so it ends with exactly one name, one of those entries.
            ++named;
            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend) << ranking.question;
            EXPECT_FALSE(ranking.recommended.empty()) << ranking.question;
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
            EXPECT_TRUE(namedIsRead)
                << ranking.recommended << " is not an entry this shape read a figure for";
        }
    }

    // The premise: at these counts the entries resolve, so a run that read nothing fails here.
    EXPECT_GT(named, 0u) << "no shape of this run read a figure at " << options.count
                         << " arguments, so the naming rule was never reached";

    // A band of two readings is not a resolution, so a run this short prints none: the line
    // names the count it is missing rather than implying bands above it could be ordered.
    const std::string text = boys::FormatDeviceOptionProbe(report);

    EXPECT_EQ(text.find("cannot be ordered on this run"), std::string::npos) << text;
    EXPECT_NE(text.find("resolution: not measurable on this run: it took 2 paired round(s)"),
              std::string::npos)
        << text;
}

/// A shape its own rounds could not order still ends with exactly one entry, and says
/// how it reached the name: the entries it could not separate are re-run alone at a
/// longer protocol and the one that led those runs is named.
///
/// The check is the rule itself, ranking by ranking: an ordering leaves no rival
/// unplaced, any other route carries the route with the name, and only a figureless
/// shape names nothing.
TEST(DeviceProbe, AShapeItsRoundsCouldNotOrderIsNamedByTheRefinementStage) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // The named set holds shapes of one entry, which cannot be ordered and are not refusals.
    options.only = {"all-n-fp64", "single-fp64", "device-single-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    std::size_t namedWithoutAnOrdering = 0;
    std::size_t refused = 0;
    bool anyMeasured = false;

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            EXPECT_FALSE(ranking.asked.empty());

            // This class is one precision and one question shape: a row outside that key
            // is no member of it.
            //
            // Three counts over its rows. A row has *measured* when its rounds produced a
            // reading; it carries a *figure* when the run has a cost to print beside it; it
            // is *placeable* when the run's own checks would order it - its figure resolved
            // and its repetition control agreed. A row of the other kinds is printed among the
            // rows the class set aside, with the check that set it aside.
            std::size_t measuredRows = 0;
            std::size_t figureRows = 0;
            std::size_t placeableRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || measurement.precision != clause.precision ||
                    measurement.question != ranking.question) {
                    continue;
                }

                ++measuredRows;
                figureRows += measurement.nsPerArgument > 0.0 ? 1u : 0u;
                placeableRows +=
                    measurement.subtractionResolved && measurement.repetitionAgrees ? 1u : 0u;
            }

            anyMeasured = anyMeasured || measuredRows > 0;

            if (figureRows == 0) {
                // No row of this class produced a figure, so there is no name to reach: a row
                // that measured and carries no figure is this answer, not an exception to it.
                ++refused;
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
                EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;
                EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kNone);
                EXPECT_FALSE(ranking.reason.empty()) << ranking.question;
                EXPECT_FALSE(ranking.refinement.ran) << "no run was timed to refine";
                continue;
            }

            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend);
            EXPECT_FALSE(ranking.recommended.empty());
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kNone);

            if (ranking.defaultHow == DeviceProbeDefaultHow::kOrdered) {
                // An ordering is only reached past every rival, so none is left unplaced.
                EXPECT_TRUE(ranking.inseparable.empty()) << ranking.recommended;
                continue;
            }

            ++namedWithoutAnOrdering;

            if (ranking.defaultHow == DeviceProbeDefaultHow::kOnlyEntry) {
                // The name rests on there being no alternative to order it against, which is
                // a statement about the rows the run could place and not the rows it timed: a
                // row whose subtraction resolved nothing, or whose repetition control
                // disagreed, is printed beside the name, not counted as a rival.
                EXPECT_LE(placeableRows, 1u)
                    << "a shape ordered against nothing holds at most one row it could order";
                EXPECT_GT(figureRows, 0u)
                    << "the name of a shape ordered against nothing is still a row that produced a "
                       "figure";
                EXPECT_TRUE(ranking.tiedEntries.empty());
                continue;
            }

            // A tie: the stage ran and voted over the entries the shape's own rounds could not
            // separate, and the entry the vote named is the one the shape names. The route is
            // the vote's own result - every run leading with that entry, a majority of them, or
            // a field the runs divided evenly.
            EXPECT_TRUE(ranking.refinement.ran) << ranking.question;
            EXPECT_FALSE(ranking.refinement.winner.empty()) << ranking.question;

            EXPECT_EQ(ranking.recommended, ranking.refinement.winner)
                << ranking.question << ": the default is not the entry the vote named";
            EXPECT_EQ(ranking.defaultHow,
                      ranking.refinement.unanimous
                          ? DeviceProbeDefaultHow::kRefined
                          : (ranking.refinement.plurality ? DeviceProbeDefaultHow::kVote
                                                          : DeviceProbeDefaultHow::kChosenAmongEquals))
                << ranking.question << ": the route is the vote's own result for the entry the "
                << "vote named";

            // The invariant the report rests on: no name is handed to a reader beside a table
            // that contradicts it without the report saying so. Where the vote named the entry
            // the shape's own figures put first, no row the run could place may be faster than
            // that entry; where it named another, the entry those figures put first is printed
            // beside the name.
            const DeviceProbeMeasurement* namedRow = nullptr;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.name == ranking.recommended) {
                    namedRow = &measurement;
                }
            }

            ASSERT_NE(namedRow, nullptr) << ranking.recommended;

            if (ranking.recommended == ranking.fastestOverall) {
                for (const DeviceProbeMeasurement& measurement : report.measurements) {
                    if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                        measurement.precision != clause.precision ||
                        measurement.question != ranking.question ||
                        !measurement.subtractionResolved ||
                        (measurement.repetitionChecked && !measurement.repetitionAgrees)) {
                        continue;
                    }

                    EXPECT_GE(measurement.nsPerArgument, namedRow->nsPerArgument)
                        << ranking.question << " names " << ranking.recommended
                        << " with " << measurement.name << " faster in the same class";
                }
            } else {
                EXPECT_FALSE(ranking.fastestOverall.empty())
                    << ranking.question << ": a vote that named another entry is printed with "
                    << "the entry the shape's own figures put first";
                EXPECT_NE(ranking.reason.find(ranking.fastestOverall), std::string::npos)
                    << ranking.question << ": " << ranking.fastestOverall
                    << " is the entry the shape's own figures put first and is not printed "
                    << "beside a name the vote chose";
            }

            EXPECT_EQ(ranking.refinement.runLeaders.size(),
                      static_cast<std::size_t>(ranking.refinement.runs));
            EXPECT_FALSE(ranking.refinement.tally.empty());
            EXPECT_FALSE(ranking.tiedEntries.empty());
            EXPECT_NE(std::find(ranking.tiedEntries.begin(), ranking.tiedEntries.end(),
                                ranking.recommended),
                      ranking.tiedEntries.end())
                << ranking.recommended << " was named by the vote but was not in the field";
        }
    }

    if (anyMeasured) {
        EXPECT_GT(namedWithoutAnOrdering, 0u)
            << "no shape of this run was named without an ordering, so the run reached no name "
               "by the route this test is about";
    } else {
        EXPECT_GT(refused, 0u) << "a run that measured nothing must refuse every shape";
    }
}

/// A tie is a result, and its evidence is what a reader has to judge: each unplaced rival
/// named with the band its ratio to the leader fell in and how often it was the slower.
///
/// Whether the card produces a tie is the card's answer, so a run that places every shape
/// outright is silent. This is the file's long pole: the refinement stage it reaches takes
/// tens of minutes.
TEST(DeviceProbe, ATieNamesEveryRivalAndTheBandItFellIn) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.count = 65536;
    options.nmax = 16;
    options.repetitions = 16;
    options.passes = 5;
    options.rounds = 2;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            for (const std::string& line : ranking.inseparable) {
                EXPECT_NE(line.find("fell in"), std::string::npos) << line;
                EXPECT_NE(line.find("slower of the two in"), std::string::npos) << line;
            }

            if (!ranking.inseparable.empty()) {
                EXPECT_NE(ranking.reason.find("could not be placed"), std::string::npos)
                    << ranking.reason;
            }

            // A refined tie still names the rivals its own rounds could not separate.
            if (ranking.refinement.ran) {
                EXPECT_FALSE(ranking.tiedEntries.empty()) << ranking.question;
            }

            if (ranking.recommended.empty()) {
                continue;
            }

            const DeviceProbeMeasurement* named = nullptr;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.name == ranking.recommended) {
                    named = &measurement;
                }
            }

            ASSERT_NE(named, nullptr) << ranking.recommended;

            // Which entry the shape ends with, by the route it was reached by. A tie the
            // shape's own rounds could not break is settled by which entry was fastest in
            // most of the refinement runs, so that entry is the one the shape names - the
            // vote decides which entry, and defaultHow states how it was reached. The entry
            // the shape's own figures put first is the record of the shorter protocol: it is
            // printed beside the name, and where the two differ the difference is what says
            // the shape's top entries cannot be separated. Where no vote named an entry, the
            // name is the one those figures put first.
            const bool voted = ranking.refinement.ran && !ranking.refinement.winner.empty();

            if (voted) {
                EXPECT_EQ(ranking.recommended, ranking.refinement.winner)
                    << ranking.question << " names " << ranking.recommended
                    << " where the vote named " << ranking.refinement.winner;
                EXPECT_EQ(ranking.defaultHow,
                          ranking.refinement.unanimous
                              ? DeviceProbeDefaultHow::kRefined
                              : (ranking.refinement.plurality
                                     ? DeviceProbeDefaultHow::kVote
                                     : DeviceProbeDefaultHow::kChosenAmongEquals))
                    << ranking.question << ": the route is the vote's own result for the entry "
                    << "the vote named";
                EXPECT_NE(std::find(ranking.tiedEntries.begin(), ranking.tiedEntries.end(),
                                    ranking.recommended),
                          ranking.tiedEntries.end())
                    << ranking.question << ": the vote ran over the entries this shape could not "
                    << "separate, so the entry it named is one of them";

                if (ranking.recommended != ranking.fastestOverall) {
                    // The vote named another entry, so the name is not the one the table beside
                    // it lists first: the report says so, or a reader comparing the two reads a
                    // contradiction where the report sees a tie.
                    EXPECT_FALSE(ranking.fastestOverall.empty())
                        << ranking.question << ": a vote that named another entry is printed with "
                        << "the entry the shape's own figures put first";
                    EXPECT_NE(ranking.reason.find(ranking.fastestOverall), std::string::npos)
                        << ranking.question << ": " << ranking.fastestOverall
                        << " is the entry the shape's own figures put first and is not printed "
                        << "beside a name the vote chose";
                }
            }

            if (voted && ranking.recommended != ranking.fastestOverall) {
                // The name is the vote's, and the class's own figures put another row faster:
                // that difference is the tie the stage was taken to settle, asserted above.
                continue;
            }

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                    measurement.precision != clause.precision ||
                    measurement.question != ranking.question ||
                    !measurement.subtractionResolved ||
                    (measurement.repetitionChecked && !measurement.repetitionAgrees)) {
                    continue;
                }

                EXPECT_GE(measurement.nsPerArgument, named->nsPerArgument)
                    << ranking.question << " names " << ranking.recommended
                    << " with " << measurement.name << " faster in the same class";
            }
        }
    }
}

/// A refusal carries the clock check as a recommendation does: whether any pair's ratio to
/// the leader moved between the run's halves by more than the run can order - the difference
/// between entries too close to separate and a machine that moved under them.
///
/// Whether a run produces a refusal is the card's answer, so one with no unplaced rival skips
/// rather than failing; the assertions above run either way.
TEST(DeviceProbe, ARefusalCarriesTheClockCheckToo) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // Enough rounds for a quartile band, at a workload the in-kernel rows resolve at.
    options.count = 65536;
    options.nmax = 16;
    options.repetitions = 16;
    options.passes = 5;
    options.rounds = 2;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_GE(report.pairedRounds, 4);

    std::size_t unplacedRankings = 0;
    std::size_t followedPairs = 0;
    std::size_t recommended = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            // One of the two sentences and no other: the warning where a pair moved further
            // than the run can order, and the statement that none did.
            const bool warned = ranking.confidence.find("WARNING: the pair") != std::string::npos;
            const bool held =
                ranking.confidence.find("No pair's ratio moved") != std::string::npos;

            // The check is about a pair this run followed: a recommendation followed every
            // rival of its shape, a refusal that names an unplaced rival followed that one. A
            // refusal that stopped earlier followed no pair, and a sentence about the clock
            // having held would be a claim about a comparison this run never made.
            const bool followedAPair =
                ranking.verdict == DeviceProbeVerdict::kRecommend || !ranking.inseparable.empty();

            if (!followedAPair) {
                EXPECT_FALSE(warned) << ranking.confidence;
                EXPECT_FALSE(held) << ranking.confidence;
                continue;
            }

            ++followedPairs;

            EXPECT_TRUE(warned || held) << ranking.confidence;

            // "Moved" is named against the run's own resolution, so a reader can judge the word.
            EXPECT_NE(ranking.confidence.find("this run can order"), std::string::npos)
                << ranking.confidence;

            if (!ranking.inseparable.empty()) {
                ++unplacedRankings;

                // The refusal names the unplaced rival beside the sentence about the clock:
                // together they are what a reader judges the refusal by.
                EXPECT_FALSE(ranking.reason.empty()) << ranking.question;
            }

            if (ranking.verdict == DeviceProbeVerdict::kRecommend) {
                ++recommended;
            }
        }
    }

    EXPECT_GT(followedPairs, 0u);
    EXPECT_GT(recommended, 0u);

    if (unplacedRankings == 0) {
        GTEST_SKIP() << "no shape of this run left a rival unplaced, so the refusal path is not "
                        "exercised by it; the clock check above was still checked on every pair "
                        "this run followed, and on the refusals that followed none";
    }
}

/// Every name the report carries came out of a clock: a name is always a row this run
/// timed, and a shape whose rows produced no figure names nothing rather than falling
/// back to the library's tables. Both halves are checked - the fallback's own words are
/// absent from the text, and each name belongs to a measured row of its shape.
TEST(DeviceProbe, EveryNameTheReportCarriesWasTimed) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    const std::string text = boys::FormatDeviceOptionProbe(report);

    EXPECT_EQ(text.find("static fallback"), std::string::npos) << text;
    EXPECT_EQ(text.find("heuristic"), std::string::npos) << text;

    std::size_t named = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            if (ranking.recommended.empty()) {
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
                EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kNone);
                EXPECT_FALSE(ranking.reason.empty()) << ranking.question;
                continue;
            }

            ++named;

            bool measured = false;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.name == ranking.recommended && measurement.measured &&
                    measurement.precision == clause.precision &&
                    measurement.question == ranking.question) {
                    measured = true;
                }
            }

            EXPECT_TRUE(measured) << ranking.recommended
                                  << " was named but is not a measured row of its shape";
            EXPECT_NE(text.find(ranking.recommended), std::string::npos) << ranking.question;
        }
    }

    if (!report.hasDefault) {
        EXPECT_EQ(named, 0u);
    }
}

/// The status a run ends in, and whether the report says so in a way a reader can act
/// on: a failed run has no figures, so the reason must be present and the device block
/// must not be silently blank when the runtime could name a card.
TEST(DeviceProbe, ADisturbedRunRefusesWithAReason) {
    DeviceProbeOptions options = Small();
    // The alarm at zero flags every pass, and the run is not filtered by it.
    options.canarySpreadAlarm = 0.0;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    EXPECT_EQ(report.passesAboveAlarm, options.passes);

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            if (ranking.verdict == DeviceProbeVerdict::kRecommend) {
                EXPECT_FALSE(ranking.recommended.empty());
                continue;
            }

            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
            EXPECT_TRUE(ranking.recommended.empty());
            EXPECT_FALSE(ranking.reason.empty());
        }
    }
}

/// Every row the report names is a row this build offers, and a row that is
/// named carries the protocol it was taken under rather than a bare number.
TEST(DeviceProbe, TheReportCarriesTheProtocolItWasTakenUnder) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    // The device the figures are about, in the returned data and not only in the text: a
    // consumer reading the struct has to be able to learn what card answered.
    EXPECT_FALSE(report.device.name.empty());
    EXPECT_GE(report.device.computeMajor, 1);
    EXPECT_FALSE(report.device.toolkitVersion.empty());
    EXPECT_FALSE(report.device.architectures.empty());

    EXPECT_EQ(report.workloadCount, options.count);
    EXPECT_FALSE(report.caveat.empty());

    // The protocol is named, and the paired rounds are the run's own passes times its rounds.
    EXPECT_EQ(report.options.passes, options.passes);
    EXPECT_EQ(report.options.rounds, options.rounds);
    EXPECT_EQ(report.pairedRounds, options.passes * options.rounds);
    // The protocol's bookkeeping, in the data and not only in the text: the passes the
    // canary placed and those it was not read in account for every pass, none dropped.
    EXPECT_EQ(report.passesWithinAlarm + report.passesAboveAlarm + report.passesWithoutCanary,
              options.passes);

    for (const DeviceProbeMeasurement& measurement : report.measurements) {
        EXPECT_FALSE(measurement.name.empty());
        EXPECT_FALSE(measurement.route.empty());
        EXPECT_FALSE(measurement.question.empty());
        EXPECT_GT(measurement.documentedBound, 0.0);

        if (measurement.measured) {
            // A measured row's spread is a ratio of ratios, so one is its floor.
            EXPECT_GE(measurement.spread, 1.0);
            EXPECT_EQ(measurement.rounds, report.pairedRounds);
        } else {
            EXPECT_EQ(measurement.rounds, 0);
        }
    }
}

/// A class is one precision and one question shape: no entry is ever placed against one
/// of another key, which is the rule the ranking is read under.
TEST(DeviceProbe, AClassIsOnePrecisionAndOneShape) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_FALSE(report.classes.empty());

    for (std::size_t c = 0; c < report.classes.size(); ++c) {
        const DeviceProbeClass& clause = report.classes[c];

        EXPECT_FALSE(clause.precision.empty());
        EXPECT_FALSE(clause.question.empty());
        EXPECT_FALSE(clause.asked.empty());
        EXPECT_FALSE(clause.note.empty());

        // One class per key, so no two classes share both members.
        for (std::size_t other = c + 1; other < report.classes.size(); ++other) {
            const bool sameKey = report.classes[other].precision == clause.precision &&
                                 report.classes[other].question == clause.question;
            EXPECT_FALSE(sameKey) << clause.precision << " " << clause.question;
        }

        const DeviceProbeRanking& ranking = clause.ranking;

        EXPECT_EQ(ranking.question, clause.question);
        EXPECT_EQ(ranking.asked, clause.asked);

        // Every row this ranking places, the recommended one included, is of this class's key.
        std::size_t rows = 0;

        for (const DeviceProbeMeasurement& measurement : report.measurements) {
            if (measurement.precision != clause.precision ||
                measurement.question != ranking.question) {
                continue;
            }

            ++rows;
        }

        EXPECT_GT(rows, 0u);

        if (!ranking.recommended.empty()) {
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

/// A name is one row: the table carries an entry once, and a class that names a winner
/// names one row of its own class and no other.
TEST(DeviceProbe, ANameIsOneRowAndAClassNamesOneOfItsOwn) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    // A name is one row of the table: two rows of one name would leave a class's winner
    // ambiguous between two readings of one entry.
    ASSERT_FALSE(report.measurements.empty());

    for (const DeviceProbeMeasurement& measurement : report.measurements) {
        std::size_t occurrences = 0;

        for (const DeviceProbeMeasurement& other : report.measurements) {
            occurrences += other.name == measurement.name ? 1u : 0u;
        }

        EXPECT_EQ(occurrences, 1u) << measurement.name;
    }

    std::size_t namedRows = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        if (clause.ranking.recommended.empty()) {
            continue;
        }

        std::size_t ofThisClass = 0;

        for (const DeviceProbeMeasurement& measurement : report.measurements) {
            if (measurement.name == clause.ranking.recommended &&
                measurement.precision == clause.precision &&
                measurement.question == clause.ranking.question) {
                ++ofThisClass;
            }
        }

        EXPECT_EQ(ofThisClass, 1u) << clause.ranking.recommended;
        ++namedRows;
    }

    // A class that named a winner named a row of its own class and no other; a run whose
    // every class refused still carries the rows the sweep measured.
    (void)namedRows;
}

/// A class that names a winner names one it measured, and a class that compared a rival
/// reports the band it compared it in; a class that compared nothing reports a resolution
/// of zero — printed in words beside the one row it had left to name.
TEST(DeviceProbe, AVerdictNamesOnlyEntriesItMeasured) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            EXPECT_FALSE(ranking.asked.empty());

            if (ranking.verdict != DeviceProbeVerdict::kRecommend) {
                EXPECT_TRUE(ranking.recommended.empty());
                continue;
            }

            EXPECT_FALSE(ranking.recommended.empty());
            EXPECT_FALSE(ranking.reason.empty());

            // A band is wider than nothing, so a class that compared a rival reports a
            // resolution above zero: this is the number the refusal prints and the number the
            // repetition control is judged against.
            if (!ranking.inseparable.empty() ||
                ranking.defaultHow == DeviceProbeDefaultHow::kOrdered) {
                EXPECT_GT(ranking.resolution, 0.0) << ranking.question;
            }

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

/// A ranking is a comparison, so it needs two entries to be one. A shape whose table holds
/// one entry names it as the shape's only entry, not as the winner of a field it never met
/// - one entry is not a ranking, and not a refusal either. The check is on the shape's own
/// rows rather than on a figure, since whether an entry measures at this workload is the
/// card's business.
TEST(DeviceProbe, AShapeOfOneNamesItsOnlyEntry) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    bool sawAShapeOfOne = false;

    for (const DeviceProbeClass& clause : report.classes) {
        {
            const DeviceProbeRanking& ranking = clause.ranking;
            // One precision and one question shape: a row outside that key is no member of
            // this class.
            std::size_t rows = 0;
            std::size_t figureRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.precision != clause.precision ||
                    measurement.question != ranking.question) {
                    continue;
                }

                ++rows;
                figureRows += measurement.measured && measurement.nsPerArgument > 0.0 ? 1u : 0u;
            }

            if (rows >= 2) {
                continue;
            }

            sawAShapeOfOne = true;
            EXPECT_FALSE(ranking.reason.empty());
            EXPECT_FALSE(ranking.confidence.empty());

            if (figureRows == 0) {
                // A shape of one whose entry produced no figure has nothing to name: the count is
                // the figures' and not the rows', so a figureless measured row is this answer.
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
                EXPECT_TRUE(ranking.recommended.empty());
                continue;
            }

            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend);
            EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kOnlyEntry);
            EXPECT_FALSE(ranking.recommended.empty());
            EXPECT_TRUE(ranking.tiedEntries.empty()) << "a shape of one has no rival to tie with";
            EXPECT_TRUE(ranking.inseparable.empty());

                // The line says what the name rests on, so no reader takes it for a winner.
            EXPECT_NE(ranking.reason.find("no alternative"), std::string::npos) << ranking.reason;
        }
    }

    EXPECT_TRUE(sawAShapeOfOne);
}

/// A control that agrees has two figures to compare: a zero left where the subtraction
/// resolved no cost is not a second reading, and calling it one passes on no evidence.
TEST(DeviceProbe, AControlAgreesOnlyBetweenTwoFigures) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    const boys::DeviceProbeRepetitionControl* controls[] = {&report.control,
                                                            &report.deviceCallControl};

    for (const boys::DeviceProbeRepetitionControl* control : controls) {
        if (control->entry.empty() || !control->agrees) {
            continue;
        }

        EXPECT_GT(control->nsPerArgumentLow, 0.0) << control->entry;
        EXPECT_GT(control->nsPerArgumentHigh, 0.0) << control->entry;
        EXPECT_TRUE(std::isfinite(control->difference)) << control->entry;

        // The two counts are judged against the resolution the row's own shape showed and the
        // two figures are quartiles of the control's own rounds: the yardstick is measured.
        EXPECT_GT(control->judgedAgainst, 0.0) << control->entry;
        EXPECT_LE(control->difference, control->judgedAgainst) << control->entry;
    }
}

/// The control's own statistic is the report's: the low and high counts are quartiles of the
/// control's rounds, so the two figures check the estimator rather than a pass.
TEST(DeviceProbe, AControlComparesTwoQuartilesOfTheSameRounds) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    const boys::DeviceProbeRepetitionControl* controls[] = {&report.control,
                                                            &report.deviceCallControl};

    for (const boys::DeviceProbeRepetitionControl* control : controls) {
        if (control->entry.empty()) {
            continue;
        }

        EXPECT_GT(control->repetitionsLow, 0);
        EXPECT_GT(control->repetitionsHigh, control->repetitionsLow);
        EXPECT_FALSE(control->route.empty());

        // The disagreement comes from the same two figures and precision, so it cannot drift.
        if (control->nsPerArgumentHigh > 0.0 && std::isfinite(control->difference)) {
            const double smaller =
                std::min(control->nsPerArgumentLow, control->nsPerArgumentHigh);

            EXPECT_GT(smaller, 0.0) << control->entry;
            EXPECT_NEAR(control->difference,
                        std::abs(control->nsPerArgumentHigh - control->nsPerArgumentLow) / smaller,
                        1e-9)
                << control->entry;
        }
    }
}

/// A device the caller names that is not there is a status, not a crash, not a silent fallback.
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

/// The name one row of the crossed space is printed under, in the report's own grammar:
/// the library's name for the entry where the row runs the build's default division form,
/// and that name with the form's own segment where it does not.
///
/// Written here from the library's two tables rather than read back from the probe: a
/// test that read the probe's own naming back would be reading the thing it is holding.
/// The default carries no segment because that is the row a caller who names no form
/// reaches (\c DeviceProbeOptions::only, boys_cuda_probe.hpp), and the two other members
/// of the axis carry theirs so that no two rows of one entry print one name.
std::string RowSpelling(const boys::DeviceOptionInfo& row, boys::DivisionForm form) {
    if (form == boys::kDefaultDivisionForm) {
        return row.name;
    }

    for (const boys::DivisionFormInfo& member : boys::BoysDivisionForms()) {
        if (member.form == form) {
            return std::string(row.name) + "-" + member.name;
        }
    }

    return std::string(row.name) + "-unnamed-form";
}

/// The option space's rows, as a report carries them.
///
/// The rows are the library's own and the text is rendered by the same function the
/// driver prints, from a report whose measurement grid is the one the probe builds - a
/// place per row this build serves, none measured - so the figures stated about the
/// space are those of a real run of this revision, without a card or a clock.
///
/// A row this build does not serve is refused with the library's own reason and owed:
/// the space states what it has and why, rather than counting a row it cannot run among
/// the rows it did not measure.
TEST(DeviceProbe, ARowIsServedOrRefusedAndTheSpaceIsCountedFromTheRows) {
    const std::span<const boys::DeviceOptionInfo> space = boys::BoysDeviceOptions();

    ASSERT_FALSE(space.empty());

    DeviceProbeReport report;
    report.status = DeviceProbeStatus::kSuccess;
    report.device.name = "a device";

    const std::size_t forms = boys::BoysDivisionForms().size();

    std::size_t refusedRows = 0;
    std::size_t servedRows = 0;
    std::size_t launchedRows = 0;
    std::size_t deviceRows = 0;

    for (const boys::DeviceOptionInfo& row : space)
    {
        (row.group == boys::DeviceOptionGroup::kLaunched ? launchedRows : deviceRows) += 1;

        // A row this build does not serve carries the library's own reason and no other:
        // the closure prints that reason rather than a sentence of its own invention. Its
        // absence is one fact and not one per form — no form of it is an arithmetic this
        // build can run — so the grid below carries no place for it at any form.
        if (!row.built)
        {
            EXPECT_NE(row.refusedBecause, nullptr) << row.name;
            ++refusedRows;
            continue;
        }

        EXPECT_EQ(row.refusedBecause, nullptr) << row.name;
        ++servedRows;

        // One place per form the library reports, named as the report names a row: the
        // default form's row carries no segment, the two other members of the axis do.
        for (const boys::DivisionFormInfo& member : boys::BoysDivisionForms())
        {
            DeviceProbeMeasurement place;
            place.name = RowSpelling(row, member.form);
            place.entry = row.entry;
            place.form = member.form;
            report.measurements.push_back(place);
        }
    }

    const std::size_t servedMembers = servedRows * forms;
    const std::size_t refusedMembers = refusedRows * forms;
    const std::size_t members = space.size() * forms;

    const std::string text = boys::FormatDeviceOptionProbe(report);

    // Stated rather than left to be counted off the report: these are the figures the
    // appendix carries, and a reader checking a revision needs them printed.
    std::printf("    space: %zu option(s) = %zu launched + %zu device-callable, crossed with %zu "
                "division form(s): %zu member(s); %zu served, %zu refused by this build\n",
                space.size(),
                launchedRows,
                deviceRows,
                forms,
                members,
                servedMembers,
                refusedMembers);

    // The space's own count, as the appendix states it, for coverage: the rows of the
    // library's table, the forms the library reports, and the product of the two.
    EXPECT_NE(text.find("the space: " + std::to_string(space.size()) +
                        " row(s) of this library's own option table"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find("crossed with the " + std::to_string(forms) + " division form(s)"),
              std::string::npos)
        << text;
    EXPECT_NE(text.find("one member per (row, form): " + std::to_string(members)),
              std::string::npos)
        << text;

    EXPECT_NE(text.find(std::to_string(refusedMembers) +
                        " refused with the library's own reason and owed"),
              std::string::npos)
        << text;

    // Every member of every row this build serves is a place of this run's grid, and none
    // produced a figure.
    EXPECT_NE(text.find(std::to_string(servedMembers) +
                        " offered and this run carried a place for, producing no figure"),
              std::string::npos)
        << text;

    EXPECT_NE(text.find("the run's own grid: " + std::to_string(servedMembers) +
                        " place(s), against the " + std::to_string(servedMembers)),
              std::string::npos)
        << text;

    // The per-row column counts the row's own members rather than a flag about a suffix
    // somewhere: the denominator is the forms the row stands at, so every row this build
    // serves reads "of 3" beside its name. The numerator is the places of those that carry
    // a figure, and this grid carries none - it is the grid without the card or the clock -
    // so the count is the zero the report is owed rather than a claim of three figures.
    EXPECT_NE(text.find("0 of " + std::to_string(forms) + " measured here"), std::string::npos)
        << text;
}

/// The report's own spelling of a precision and of a question, as a class of it carries
/// them. Written here because the closures below count classes and never name one: a test
/// that had to read the probe's own spelling would be reading the thing it is holding.
const char* PrecisionSpelling(boys::DeviceOptionPrecision precision) {
    switch (precision) {
        case boys::DeviceOptionPrecision::kFp64:
            return "fp64";
        case boys::DeviceOptionPrecision::kFp32:
            return "fp32";
        case boys::DeviceOptionPrecision::kFp16:
            return "fp16";
        case boys::DeviceOptionPrecision::kBf16:
            return "bf16";
        case boys::DeviceOptionPrecision::kCount:
            break;
    }

    return "unnamed";
}

const char* QuestionSpelling(boys::DeviceOptionQuestion question) {
    switch (question) {
        case boys::DeviceOptionQuestion::kSingle:
            return "single";
        case boys::DeviceOptionQuestion::kAllOrders:
            return "all-orders";
        case boys::DeviceOptionQuestion::kAllN:
            return "all-n";
        case boys::DeviceOptionQuestion::kCount:
            break;
    }

    return "unnamed";
}

/// A report whose measurement grid is the one the probe builds — a place per row this
/// build serves **crossed with each division form the library reports**, none of them
/// measured, and one class per precision and question those places fall into — so the
/// closure can be counted, and the text rendered, without a card or a clock.
/// \p servedCells comes back as the places this build serves, which is the cross and not
/// its first column.
DeviceProbeReport GriddedReport(std::size_t& servedCells) {
    DeviceProbeReport report;
    report.status = DeviceProbeStatus::kSuccess;
    report.device.name = "a device";
    servedCells = 0;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions()) {
        if (!row.built) {
            continue;
        }

        for (const boys::DivisionFormInfo& member : boys::BoysDivisionForms()) {
            DeviceProbeMeasurement place;
            place.name = RowSpelling(row, member.form);
            place.entry = row.entry;
            place.form = member.form;
            place.precision = PrecisionSpelling(row.precision);
            place.question = QuestionSpelling(row.question);
            report.measurements.push_back(place);
            ++servedCells;
        }
    }

    // One class per precision and question the places above fall into, which is what the
    // probe's own conclusion loop builds over its grid.
    for (const DeviceProbeMeasurement& place : report.measurements) {
        bool held = false;

        for (const DeviceProbeClass& clause : report.classes) {
            held = held || (clause.precision == place.precision &&
                            clause.question == place.question);
        }

        if (held) {
            continue;
        }

        DeviceProbeClass fresh;
        fresh.precision = place.precision;
        fresh.question = place.question;
        report.classes.push_back(fresh);
    }

    return report;
}

/// The closure is the space counted, and every member of it is in exactly one state: the
/// total is the library's own option table **crossed with the division forms the library
/// reports**, the states add up to it, and the verdict the report's last line prints is
/// that arithmetic's.
TEST(DeviceProbe, TheClosurePutsEveryMemberOfTheSpaceInOneState) {
    std::size_t servedCells = 0;
    DeviceProbeReport report = GriddedReport(servedCells);

    ASSERT_GT(servedCells, 0u);

    const std::size_t rows = boys::BoysDeviceOptions().size();
    const std::size_t forms = boys::BoysDivisionForms().size();
    const std::size_t total = rows * forms;
    std::size_t builtRows = 0;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions()) {
        builtRows += row.built ? 1u : 0u;
    }

    // The grid is the run's own, and the total is the library's: the two are read from
    // different sources and the closure demands they agree. Every row this build serves
    // is one member per form and no fewer - a grid that carried one form of a row would
    // leave the other two unaccounted rather than counted.
    EXPECT_EQ(servedCells, builtRows * forms);
    EXPECT_GT(forms, 1u);

    const boys::DeviceOptionClosure offered = boys::DeviceOptionSpaceClosure(report);
    EXPECT_EQ(offered.rows, rows);
    EXPECT_EQ(offered.forms, forms);
    EXPECT_EQ(offered.total, total);
    EXPECT_EQ(offered.states, total);
    EXPECT_EQ(offered.measured, 0u);
    EXPECT_EQ(offered.offeredNoFigure, servedCells);
    EXPECT_EQ(offered.refusedAndOwed, total - servedCells);
    EXPECT_EQ(offered.notRunnable, 0u);
    EXPECT_EQ(offered.notAsked, 0u);
    EXPECT_EQ(offered.unaccounted, 0u);
    EXPECT_EQ(offered.gridPlaces, servedCells);
    EXPECT_EQ(offered.gridPlacesOwed, servedCells);

    // The third reading: every class the space admits over these rows is one the report
    // carries, and the grid's places are what makes a class, not the figures in them.
    EXPECT_EQ(offered.classesAdmitted, offered.classesPrinted);
    EXPECT_GT(offered.classesAdmitted, 0u);
    EXPECT_TRUE(offered.closed);

    // A place the run offered and no round of which produced a figure is not a failure of
    // the closure: it is a state, and the arithmetic says which.
    const std::string offeredText = boys::FormatDeviceOptionProbe(report);
    EXPECT_NE(offeredText.find("the arithmetic: 0 + "), std::string::npos) << offeredText;
    EXPECT_NE(offeredText.find("the verdict: PASS"), std::string::npos) << offeredText;

    // And the cross is stated in the report's own words, not only in the counts: the total
    // a reader checks is the product of two tables and both are named where it is printed.
    EXPECT_NE(offeredText.find("the space: " + std::to_string(rows) +
                               " row(s) of this library's own option table"),
              std::string::npos)
        << offeredText;
    EXPECT_NE(offeredText.find("crossed with the " + std::to_string(forms) + " division form(s)"),
              std::string::npos)
        << offeredText;
    EXPECT_NE(offeredText.find("one member per (row, form): " + std::to_string(total)),
              std::string::npos)
        << offeredText;

    for (DeviceProbeMeasurement& place : report.measurements) {
        place.measured = true;
    }

    const boys::DeviceOptionClosure measured = boys::DeviceOptionSpaceClosure(report);
    EXPECT_EQ(measured.measured, servedCells);
    EXPECT_EQ(measured.offeredNoFigure, 0u);
    EXPECT_EQ(measured.unaccounted, 0u);
    EXPECT_TRUE(measured.closed);
    EXPECT_NE(boys::FormatDeviceOptionProbe(report).find(
                  "MEMBERS: " + std::to_string(servedCells) + " of " + std::to_string(total) +
                      " member(s) of the option space are measured"),
              std::string::npos);

    // A card that would not hold the degree tables takes every row this build serves out
    // of the measured count and into its own, and the arithmetic still closes: the rows
    // were presented to the device and it would not hold the tables they read.
    report.tablesResident = false;
    report.refusedTables = "the device would not hold the degree tables";

    const boys::DeviceOptionClosure held = boys::DeviceOptionSpaceClosure(report);
    EXPECT_EQ(held.notRunnable, servedCells);
    EXPECT_EQ(held.measured, 0u);
    EXPECT_EQ(held.unaccounted, 0u);
    EXPECT_TRUE(held.closed);

    const std::string heldText = boys::FormatDeviceOptionProbe(report);
    EXPECT_NE(heldText.find(std::to_string(servedCells) + " not runnable on this card"),
              std::string::npos)
        << heldText;
    EXPECT_NE(heldText.find("the verdict: PASS"), std::string::npos) << heldText;
}

/// A run that never reached the space closes on nothing. Its members are in no state of
/// it, the arithmetic carries the count, and the verdict fails — a closure that could not
/// fail would be a decoration.
TEST(DeviceProbe, AClosureOverARunThatMeasuredNothingFails) {
    DeviceProbeReport report;
    report.status = DeviceProbeStatus::kNoDevice;
    report.failure = "this machine has no CUDA device, so there is no device option to measure";

    std::size_t servedCells = 0;
    const DeviceProbeReport grid = GriddedReport(servedCells);

    const boys::DeviceOptionClosure closure = boys::DeviceOptionSpaceClosure(report);

    // The members this build serves and the run never presented: in no state at all, and
    // counted as that rather than into the nearest state above.
    EXPECT_EQ(closure.unaccounted, servedCells);
    EXPECT_EQ(closure.measured, 0u);
    EXPECT_EQ(closure.offeredNoFigure, 0u);
    EXPECT_EQ(closure.notAsked, 0u);
    EXPECT_EQ(closure.states + closure.unaccounted, closure.total);
    EXPECT_GT(closure.classesAdmitted, closure.classesPrinted);
    EXPECT_FALSE(closure.closed);

    const std::string text = boys::FormatDeviceOptionProbe(report);
    EXPECT_NE(text.find("the verdict: FAIL"), std::string::npos) << text;
    EXPECT_NE(text.find("class(es) where the space admits"), std::string::npos) << text;
    EXPECT_NE(text.find(std::to_string(servedCells) + " member(s) are in no state above"),
              std::string::npos)
        << text;
    EXPECT_EQ(text.find("the verdict: PASS"), std::string::npos) << text;

    // The same report with the run succeeded and the grid it builds: the closure holds,
    // so the failure above is the run's state and not a verdict the counting cannot give.
    const boys::DeviceOptionClosure closed = boys::DeviceOptionSpaceClosure(grid);
    EXPECT_TRUE(closed.closed);
}

/// A request that named a set is closed with the rest of the space stated: the members no
/// name was given for are counted as not asked for rather than as members nothing
/// accounts for, and the grid the run owes is the places of the rows it named.
///
/// **The row is the unit of a request and a row is an entry crossed with a form**, so the
/// name a request carries is a member's own: the unmarked spelling asks for the default
/// form's row and the marked spellings for the other two, and a request that wants an
/// entry whole names all three of its rows.
TEST(DeviceProbe, ARequestForOneEntryIsClosedWithTheRestOfTheSpaceStated) {
    std::string named;
    std::string namedPrecision;
    std::string namedQuestion;
    const boys::DeviceOptionInfo* namedRow = nullptr;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions()) {
        if (row.built) {
            namedRow = &row;
            named = row.name;
            namedPrecision = PrecisionSpelling(row.precision);
            namedQuestion = QuestionSpelling(row.question);
            break;
        }
    }

    ASSERT_FALSE(named.empty());
    ASSERT_NE(namedRow, nullptr);

    // The whole space's grid, for the members the space serves, and the grid a request for
    // one row builds: the probe builds its grid from the rows a request names, so every
    // other row's places — and their classes — are absent from it.
    std::size_t servedCells = 0;
    const DeviceProbeReport wholeSpace = GriddedReport(servedCells);

    DeviceProbeReport report;
    report.status = DeviceProbeStatus::kSuccess;
    report.device.name = "a device";
    report.options.only = {named};

    for (const DeviceProbeMeasurement& place : wholeSpace.measurements) {
        if (place.name == named) {
            DeviceProbeMeasurement measured = place;
            measured.measured = true;
            report.measurements.push_back(measured);
        }
    }

    for (const DeviceProbeClass& clause : wholeSpace.classes) {
        if (clause.precision == namedPrecision && clause.question == namedQuestion) {
            report.classes.push_back(clause);
        }
    }

    const std::size_t placesOfTheNamedRow = report.measurements.size();

    // The unmarked name is the default form's row and not all three members of the entry:
    // the other two carry the form's segment, so a request naming one row asks for one
    // place and the grid it owes is that row's.
    EXPECT_EQ(placesOfTheNamedRow, 1u);

    ASSERT_GT(placesOfTheNamedRow, 0u);
    ASSERT_LT(placesOfTheNamedRow, servedCells);

    const boys::DeviceOptionClosure closure = boys::DeviceOptionSpaceClosure(report);

    EXPECT_EQ(closure.notAsked, servedCells - placesOfTheNamedRow);
    EXPECT_EQ(closure.measured, placesOfTheNamedRow);
    EXPECT_EQ(closure.gridPlaces, placesOfTheNamedRow);
    EXPECT_EQ(closure.gridPlacesOwed, placesOfTheNamedRow);
    EXPECT_EQ(closure.classesAdmitted, 1u);
    EXPECT_EQ(closure.classesPrinted, 1u);
    EXPECT_EQ(closure.unaccounted, 0u);
    EXPECT_TRUE(closure.closed);

    const std::string text = boys::FormatDeviceOptionProbe(report);
    EXPECT_NE(text.find("not asked for by this run's request"), std::string::npos) << text;
    EXPECT_NE(text.find("the verdict: PASS"), std::string::npos) << text;

    // The same entry asked for whole: every form the library reports is a row of its own,
    // so naming the entry's three rows is what asks for the entry at every form, and the
    // grid the run owes is three places against the one the unmarked name asked for.
    DeviceProbeReport everyForm;
    everyForm.status = DeviceProbeStatus::kSuccess;
    everyForm.device.name = "a device";
    everyForm.classes = report.classes;

    for (const boys::DivisionFormInfo& member : boys::BoysDivisionForms()) {
        everyForm.options.only.push_back(RowSpelling(*namedRow, member.form));
    }

    for (const DeviceProbeMeasurement& place : wholeSpace.measurements) {
        if (place.entry == namedRow->entry) {
            DeviceProbeMeasurement measured = place;
            measured.measured = true;
            everyForm.measurements.push_back(measured);
        }
    }

    ASSERT_EQ(everyForm.measurements.size(), boys::BoysDivisionForms().size());

    const boys::DeviceOptionClosure whole = boys::DeviceOptionSpaceClosure(everyForm);

    EXPECT_EQ(whole.measured, everyForm.measurements.size());
    EXPECT_EQ(whole.notAsked, servedCells - everyForm.measurements.size());
    EXPECT_EQ(whole.gridPlacesOwed, everyForm.measurements.size());
    EXPECT_EQ(whole.unaccounted, 0u);
    EXPECT_TRUE(whole.closed);

    // The two counts are the two requests told apart by the cross and not by this test:
    // the unmarked name asked for one member of the entry, the three names for three, and
    // the difference is the form axis being enumerated rather than pinned to a row.
    EXPECT_GT(whole.measured, closure.measured);
    EXPECT_EQ(whole.measured - closure.measured, boys::BoysDivisionForms().size() - 1u);
}

/// The run's own grid is the cross and not its first column: a request that names one
/// entry's rows is measured at one place per form, each carrying the form and the name the
/// request named — so an enumeration that pinned the form to a row would fail here rather
/// than measure one form three times under three names, and the closure over that run owes
/// exactly those places.
///
/// This is the device's own table and not a report built here: the places below are the
/// ones \c RunDeviceOptionProbe built, read off the run.
TEST(DeviceProbe, ARequestForAnEntryAtEveryFormIsMeasuredAtEveryForm) {
    const std::span<const boys::DivisionFormInfo> forms = boys::BoysDivisionForms();

    ASSERT_GT(forms.size(), 1u);

    const boys::DeviceOptionInfo* target = nullptr;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions()) {
        if (row.built && row.group == boys::DeviceOptionGroup::kLaunched) {
            target = &row;
            break;
        }
    }

    ASSERT_NE(target, nullptr);

    DeviceProbeOptions options = Small();

    for (const boys::DivisionFormInfo& member : forms) {
        options.only.push_back(RowSpelling(*target, member.form));
    }

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_EQ(report.measurements.size(), forms.size());

    // One place per form, the entry's own, and the form each place carries is the one the
    // request named rather than one value the run fixed for all of the entry's rows.
    for (const boys::DivisionFormInfo& member : forms) {
        const DeviceProbeMeasurement* place = nullptr;

        for (const DeviceProbeMeasurement& candidate : report.measurements) {
            if (candidate.entry == target->entry && candidate.form == member.form) {
                place = &candidate;
            }
        }

        EXPECT_NE(place, nullptr) << member.name;

        if (place != nullptr) {
            EXPECT_EQ(place->name, RowSpelling(*target, member.form)) << member.name;
        }
    }

    // And the closure over that run: the places it carries are the places the request owed,
    // against the whole space the library's two tables state.
    const boys::DeviceOptionClosure closure = boys::DeviceOptionSpaceClosure(report);

    EXPECT_EQ(closure.gridPlaces, forms.size());
    EXPECT_EQ(closure.gridPlacesOwed, forms.size());
    EXPECT_EQ(closure.unaccounted, 0u);
    EXPECT_EQ(closure.states, closure.total);
    EXPECT_EQ(closure.forms, forms.size());
}

} // namespace
