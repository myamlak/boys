// The device option probe (boys/boys_cuda_probe.hpp): what it reports about the
// card, and what it does when the caller names a card that is not there.
//
// No cost is asserted here, a figure being the property of the card it was taken
// on. What is asserted is the shape of the report: ratios formed inside a round and
// pooled over every round, a pass the canary flagged used rather than dropped, a
// canary no pass read told apart from a silent one, every name taken from a clock
// rather than off the library's tables, and a class being one precision, one
// accuracy rung and one question shape.
//
// The protocol is short on purpose, the shape of the result being what is under
// test. ATieNamesEveryRivalAndTheBandItFellIn is the exception: it hunts a tie
// through the refinement stage and takes tens of minutes on a real card.

#include "boys/boys_cuda_probe.hpp"

// The lane's own rung table (kDeviceRungs), which a report's classes are keyed on,
// and the entries whose rung axis is part of it rather than the whole of it.
#include "boys/boys_cuda.hpp"
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
/// the least a lower-quartile band can be formed from, so a test that wants a run
/// able to order anything keeps at least this much.
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
/// Checked at both ends of the alarm, with the same rounds pooled either way.
TEST(DeviceProbe, APassAboveTheCanaryAlarmIsFlaggedAndStillUsed) {
    DeviceProbeOptions flagged = Small();
    flagged.canarySpreadAlarm = 0.0;

    const DeviceProbeReport wide = boys::RunDeviceOptionProbe(flagged);

    ASSERT_EQ(wide.status, DeviceProbeStatus::kSuccess);

    EXPECT_EQ(wide.passes.size(), static_cast<std::size_t>(flagged.passes));
    EXPECT_EQ(wide.passesWithinAlarm, 0);
    EXPECT_EQ(wide.pairedRounds, flagged.passes * flagged.rounds);

    // The three counters partition the passes. With the alarm at zero, every pass a
    // reading was taken in is above it.
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

    // Every pass of the run above was flagged, and the run is the same run: the
    // same number of passes reported and the same rounds pooled. A flagged pass
    // that had been discarded would show up here as fewer pooled rounds.
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

    // The words, and not a zero: a reading would print as 0.0000 here, which is the
    // failure this test exists to catch.
    EXPECT_NE(row.find("canary not timed"), std::string::npos) << row;
    EXPECT_EQ(row.find("0.0000"), std::string::npos) << row;

    // A measured pass beside it prints its figures and its flag, so the words above
    // are the absence of a reading and not a column that never carries one.
    DeviceProbeReport mixed = unread;
    mixed.passes = {silent, read};
    mixed.passesWithoutCanary = 1;
    mixed.passesAboveAlarm = 1;

    // The counters are the table: the pass that took no reading is in the third
    // and in neither of the other two.
    EXPECT_EQ(mixed.passesWithinAlarm + mixed.passesAboveAlarm + mixed.passesWithoutCanary,
              static_cast<int>(mixed.passes.size()));

    const std::string mixedText = boys::FormatDeviceOptionProbe(mixed);

    EXPECT_NE(mixedText.find("3.2500"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("9.00"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("canary wide - reported, used"), std::string::npos) << mixedText;
    EXPECT_NE(mixedText.find("took no canary reading at all"), std::string::npos) << mixedText;
    EXPECT_EQ(mixedText.find("no pass took a canary reading"), std::string::npos) << mixedText;

    // The resolution line carries the canary only where one was read: a shape with no
    // canary spread must not be handed a zero to read as one.
    DeviceProbeRanking refused;
    refused.question = "all-orders";
    refused.asked = "every order";
    // At four paired rounds or more the report prints a band; below that it says so
    // instead.
    refused.rounds = 10;
    refused.resolution = 0.25;
    refused.verdict = DeviceProbeVerdict::kCannotDetermine;
    refused.fastestOverall = "device-all-orders-fp64";
    refused.reason = "two entry(s) of this shape could not be placed";
    refused.confidence = "CANNOT DETERMINE: the widest band this shape showed was 25.00%";

    DeviceProbeClass clause;
    clause.precision = "fp64";
    clause.rung = boys::kBoysFullAccuracyMultiplier;
    clause.rungName = "1";
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
/// itself is exactly one with exactly no drift — the one value in the table known
/// independently of the card, which is what pins the ratios as formed per round and
/// not across rounds.
///
/// The anchor is per rung: a row's cost is its own rung's block scaled by a
/// within-rung ratio, while the report's \c referenceNsPerArgument carries the
/// full-accuracy block alone.
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
            // An entry that produced no figure rests on no round, and it says so
            // rather than claiming the run's rounds.
            EXPECT_EQ(measurement.rounds, 0);
            continue;
        }

        ++measured;

        EXPECT_EQ(measurement.rounds, report.pairedRounds) << measurement.name;
        // Both ends of the band are ratios of two entries timed in one round, so
        // the upper quartile is never below the lower one.
        EXPECT_GE(measurement.ratioHi, measurement.ratioLo) << measurement.name;
        // An in-kernel row resolves when the difference it was reduced to stood above
        // its own baseline, which is what that statistic reads; a launched row always
        // does, its cell being its own reading rather than a difference that could be
        // floored. The band's lower end may still be the zero that difference was
        // floored at, so only the figure decides whether the row is ordered.
        EXPECT_EQ(measurement.subtractionResolved,
                  measurement.launchedByLibrary || measurement.nsPerArgument > 0.0)
            << measurement.name;
        EXPECT_GE(measurement.spread, 1.0) << measurement.name;

        // The reported cost is this row's own anchor scaled by its ratio to it, so the
        // two columns cannot come from different statistics. The anchor is the row's and
        // not the report's: referenceNsPerArgument is the full-accuracy rung's block, and
        // a row of another rung was scaled by that rung's own block.
        EXPECT_NEAR(measurement.nsPerArgument,
                    measurement.referenceNsPerArgument * measurement.ratioToReference,
                    1e-9 * std::max(1.0, measurement.referenceNsPerArgument)) << measurement.name;
        EXPECT_NEAR(measurement.nsPerArgumentMax,
                    measurement.referenceNsPerArgument * measurement.ratioHi,
                    1e-9 * std::max(1.0, measurement.referenceNsPerArgument)) << measurement.name;

        // The peak column is the entry's own fastest single round, a raw figure under no
        // anchor, beside a reported cost that is a quartile of ratios anchored to the
        // reference: statistics of different quantities, so neither bounds the other in
        // general. The bound is asserted where it is real — on the reference entry, whose
        // two columns come from one entry's own rounds.
        EXPECT_TRUE(std::isfinite(measurement.nsPerArgumentPeak)) << measurement.name;
        EXPECT_GE(measurement.nsPerArgumentPeak, 0.0) << measurement.name;

        if (measurement.name != report.referenceEntry) {
            continue;
        }

        sawReference = true;
        // The reference against itself: one in every round by construction, so every
        // quantile is one and the drift is zero — at every rung, the entry being its
        // block's own anchor.
        EXPECT_DOUBLE_EQ(measurement.ratioToReference, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioLo, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioHi, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioDrift, 0.0);
        EXPECT_DOUBLE_EQ(measurement.nsPerArgument, measurement.referenceNsPerArgument);
        EXPECT_LE(measurement.nsPerArgumentPeak, measurement.nsPerArgument + 1e-12);

        // The one anchor the report carries is this entry's figure at the full-accuracy
        // rung, the block it was taken from. Every other row is scaled by its own rung's
        // anchor, which is why the anchor is carried on the row and not once for the
        // table.
        if (measurement.rung == boys::kDeviceRungs.front()) {
            EXPECT_DOUBLE_EQ(measurement.referenceNsPerArgument, report.referenceNsPerArgument);
        }
    }

    EXPECT_TRUE(sawReference);
    EXPECT_GT(measured, 0u);
}

/// A run too short for a quartile band still ends with one entry per shape that
/// produced a figure, and says both the count it took and the count a band needs.
///
/// Below four readings a band's two ends are the same reading twice, so such a run
/// can place nothing. What it may not do is leave a shape that produced a figure
/// without a name: the one entry such a shape has, or the entry the refinement
/// stage's longer protocol led.
TEST(DeviceProbe, AShortRunNamesAnEntryAndSaysTheBandWasNeverFormed) {
    DeviceProbeOptions options = Small();
    // Short in rounds rather than in workload: a fixture small in both is one whose
    // entries may not resolve, and this rule is about rounds.
    options.count = 1u << 18;
    options.passes = 1;
    options.repetitions = 8;
    options.refinementRuns = 2;
    options.refinementFactor = 2;
    options.canarySpreadAlarm = 1.0e9;
    // One shape of two rows — one launched and one in-kernel — and two shapes of one,
    // so the run holds a name reached by the stage and a name reached by there being
    // no rival at all.
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

            // Nothing can be placed on two readings, so the reason says the count it took
            // and the count it needs rather than leaving the shortfall to be inferred.
            EXPECT_NE(ranking.reason.find("2 paired round(s)"), std::string::npos)
                << ranking.reason;
            EXPECT_NE(ranking.reason.find("need 4"), std::string::npos) << ranking.reason;

            // No shape of this run is reported as an ordering: the band one would have
            // been read off was never formed, whichever way the name was reached.
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kOrdered) << ranking.question;

            // The rows of this shape at this class's own rung. Counting a precision's rows
            // without the rung would read a class of one row as a class of twelve.
            std::size_t read = 0;
            bool namedIsRead = false;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                    measurement.precision != clause.precision ||
                    measurement.rung != clause.rung ||
                    measurement.question != ranking.question) {
                    continue;
                }

                ++read;
                namedIsRead = namedIsRead || (measurement.name == ranking.recommended &&
                                              measurement.rung == clause.rung);
            }

            if (read == 0) {
                // The one answer a run cannot name an entry from: no entry of this
                // shape produced a figure at all.
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine)
                    << ranking.question;
                EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;
                EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
                continue;
            }

            // Entries of this shape produced figures, so it ends with exactly one name, and
            // that name is one of those entries and never a row read off a table.
            ++named;
            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend) << ranking.question;
            EXPECT_FALSE(ranking.recommended.empty()) << ranking.question;
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
            EXPECT_TRUE(namedIsRead)
                << ranking.recommended << " is not an entry this shape read a figure for";
        }
    }

    // The premise: at these counts the entries resolve, so a run that read nothing fails
    // here rather than passing vacuously.
    EXPECT_GT(named, 0u) << "no shape of this run read a figure at " << options.count
                         << " arguments, so the naming rule was never reached";

    // A band of two readings is not a resolution, so a run this short prints none: the
    // sentence that would carry one implies bands above it could be ordered, which this
    // run cannot say, so the line names the count it is missing instead.
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
/// The check is the rule itself, ranking by ranking: a shape ordered by its own rounds
/// leaves no rival unplaced, a shape reached any other way carries the route with the
/// name, and a shape whose entries produced no figure is the only one naming nothing.
TEST(DeviceProbe, AShapeItsRoundsCouldNotOrderIsNamedByTheRefinementStage) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // The named set holds shapes of one entry, which cannot be ordered and are not
    // refusals either, so the run holds a name reached without an ordering.
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

            // This class is one precision at one rung for one question shape: the rung is
            // part of the key, and a count taken without it would make a class of one row
            // look like a class of twelve.
            //
            // Three counts over its rows. A row has *measured* when its rounds produced a
            // reading; it carries a *figure* when the run has a cost to print beside it; it
            // is *placeable* when the run's own checks would order it — its figure resolved
            // and its repetition control agreed. A row of the other kinds is not dropped:
            // the class prints it among the rows it set aside, with the check that set it
            // aside.
            std::size_t measuredRows = 0;
            std::size_t figureRows = 0;
            std::size_t placeableRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || measurement.precision != clause.precision ||
                    measurement.rung != clause.rung ||
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
                // No row of this class produced a figure, so there is no name to reach. A
                // row that measured and carries no figure is this answer and not an
                // exception to it: there is no cost to name a row by and nothing for the
                // stage to re-run.
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
                // An ordering is only reached past every rival, so no rival of a
                // shape ordered on its own rounds is left unplaced.
                EXPECT_TRUE(ranking.inseparable.empty()) << ranking.recommended;
                continue;
            }

            ++namedWithoutAnOrdering;

            if (ranking.defaultHow == DeviceProbeDefaultHow::kOnlyEntry) {
                // The name rests on there being no alternative to order it against, which is
                // a statement about the rows the run could place and not about the rows it
                // timed: a row whose subtraction resolved nothing, or whose repetition control
                // disagreed, is printed beside the name with its reason rather than counted as
                // a rival the run failed to name.
                EXPECT_LE(placeableRows, 1u)
                    << "a shape ordered against nothing holds at most one row it could order";
                EXPECT_GT(figureRows, 0u)
                    << "the name of a shape ordered against nothing is still a row that produced a "
                       "figure";
                EXPECT_TRUE(ranking.tiedEntries.empty());
                continue;
            }

            // A tie: the stage ran and voted over the entries the shape's own rounds could
            // not separate. The vote decides how the named entry was reached, never which
            // entry it is, so the route is read back from the vote.
            EXPECT_TRUE(ranking.refinement.ran) << ranking.question;
            EXPECT_FALSE(ranking.refinement.winner.empty()) << ranking.question;

            const bool voteNamedIt = ranking.refinement.winner == ranking.recommended;

            EXPECT_EQ(ranking.defaultHow,
                      !voteNamedIt
                          ? DeviceProbeDefaultHow::kChosenAmongEquals
                          : (ranking.refinement.unanimous
                                 ? DeviceProbeDefaultHow::kRefined
                                 : (ranking.refinement.plurality
                                        ? DeviceProbeDefaultHow::kVote
                                        : DeviceProbeDefaultHow::kChosenAmongEquals)))
                << ranking.question << ": the route is the vote's own result for the entry "
                << "the shape's figures put first, and a vote for another entry is a tie";

            // The invariant the report rests on: the name printed as the default is the
            // cheapest row its own class could be ordered by. A vote may name another entry —
            // and the report says so — but no placeable row of the class may be faster than
            // the name.
            const DeviceProbeMeasurement* namedRow = nullptr;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.name == ranking.recommended &&
                    measurement.rung == clause.rung) {
                    namedRow = &measurement;
                }
            }

            ASSERT_NE(namedRow, nullptr) << ranking.recommended;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                    measurement.precision != clause.precision ||
                    measurement.rung != clause.rung ||
                    measurement.question != ranking.question ||
                    !measurement.subtractionResolved ||
                    (measurement.repetitionChecked && !measurement.repetitionAgrees)) {
                    continue;
                }

                EXPECT_GE(measurement.nsPerArgument, namedRow->nsPerArgument)
                    << ranking.question << " names " << ranking.recommended
                    << " with " << measurement.name << " faster in the same class";
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

/// A tie is a result, and its evidence is what a reader has to be able to judge: each
/// unplaced rival named with the band its ratio to the leader fell in and with how
/// often it was the slower of the two.
///
/// Whether the card produces a tie is the card's answer rather than this test's, so a
/// run that places every shape outright is silent. This is the file's long pole by a
/// wide margin: the refinement stage the tie reaches takes tens of minutes.
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

            // A refined tie still names the rivals its own rounds could not separate: the
            // refinement answers the shape and does not erase what the shape found.
            if (ranking.refinement.ran) {
                EXPECT_FALSE(ranking.tiedEntries.empty()) << ranking.question;
            }

            // The name the shape ends with is the one its own figures put first, whatever
            // the vote said — a tie is where it is easiest to print a name the table beside
            // it contradicts. "Its own figures" means the placeable rows: a row the run
            // declines to rank anything on may carry a figure the name is behind, and the
            // report says so where it prints that route.
            if (ranking.recommended.empty()) {
                continue;
            }

            const DeviceProbeMeasurement* named = nullptr;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.name == ranking.recommended &&
                    measurement.rung == clause.rung) {
                    named = &measurement;
                }
            }

            ASSERT_NE(named, nullptr) << ranking.recommended;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (!measurement.measured || !(measurement.nsPerArgument > 0.0) ||
                    measurement.precision != clause.precision ||
                    measurement.rung != clause.rung ||
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

/// A refusal carries the clock check as a recommendation does: whether any pair's
/// ratio to the leader moved between the run's halves by more than the run can order.
/// A refusal is exactly where a reader wants it — the difference between entries too
/// close to separate and a machine that moved under them.
///
/// Whether a run produces a refusal is the card's answer, so a run with no unplaced
/// rival skips rather than failing; the assertions above it run either way.
TEST(DeviceProbe, ARefusalCarriesTheClockCheckToo) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // Enough rounds for a quartile band, at a workload the in-kernel rows resolve at: the
    // pair this test is about is two rows of one kernel a few percent apart.
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

            // "Moved" is named against the run's own resolution, so a reader can judge the
            // word rather than take it.
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
/// back to the library's tables. Both halves are checked — the words the fallback was
/// printed under are absent from the text, and each name belongs to a measured row of
/// its shape.
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
                if (measurement.name == ranking.recommended && measurement.rung == clause.rung &&
                    measurement.measured &&
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
    // The alarm at zero flags every pass, the most disturbed the canary can report, and
    // the run is not filtered by it.
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

    // The protocol is named, and the number of paired rounds the figures rest on
    // is the run's own passes times its own rounds rather than a claim about it.
    EXPECT_EQ(report.options.passes, options.passes);
    EXPECT_EQ(report.options.rounds, options.rounds);
    EXPECT_EQ(report.pairedRounds, options.passes * options.rounds);
    // The protocol's bookkeeping, in the data and not only in the text: the passes the
    // canary placed and those it was not read in account for every pass, so none was
    // dropped.
    EXPECT_EQ(report.passesWithinAlarm + report.passesAboveAlarm + report.passesWithoutCanary,
              options.passes);

    for (const DeviceProbeMeasurement& measurement : report.measurements) {
        EXPECT_FALSE(measurement.name.empty());
        EXPECT_FALSE(measurement.route.empty());
        EXPECT_FALSE(measurement.question.empty());
        EXPECT_GT(measurement.documentedBound, 0.0);

        if (measurement.measured) {
            // A measured row's spread is a ratio of ratios, so one is its floor,
            // and it rests on the run's pooled rounds rather than on a pass.
            EXPECT_GE(measurement.spread, 1.0);
            EXPECT_EQ(measurement.rounds, report.pairedRounds);
        } else {
            EXPECT_EQ(measurement.rounds, 0);
        }
    }
}

/// A class is one precision, one accuracy rung and one question shape: no entry is ever
/// placed against an entry of another precision, of another rung or of another question.
/// That is the rule the ranking is read under, so it is checked key by key.
TEST(DeviceProbe, AClassIsOnePrecisionOneRungAndOneShape) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_FALSE(report.classes.empty());

    for (std::size_t c = 0; c < report.classes.size(); ++c) {
        const DeviceProbeClass& clause = report.classes[c];

        EXPECT_FALSE(clause.precision.empty());
        EXPECT_FALSE(clause.rungName.empty());
        EXPECT_FALSE(clause.question.empty());
        EXPECT_FALSE(clause.asked.empty());
        EXPECT_FALSE(clause.note.empty());

        // Every rung a class is keyed on is one the library serves, and the rung a default
        // is read from has to be measured: a class keyed on a rung no lane holds holds
        // nothing.
        const auto rung =
            std::find(boys::kDeviceRungs.begin(), boys::kDeviceRungs.end(), clause.rung);
        EXPECT_NE(rung, boys::kDeviceRungs.end()) << clause.rungName;

        // One class per key, so no two classes share all three members.
        for (std::size_t other = c + 1; other < report.classes.size(); ++other) {
            const bool sameKey = report.classes[other].precision == clause.precision &&
                                 report.classes[other].rung == clause.rung &&
                                 report.classes[other].question == clause.question;
            EXPECT_FALSE(sameKey) << clause.precision << " " << clause.rungName << " "
                                  << clause.question;
        }

        const DeviceProbeRanking& ranking = clause.ranking;

        EXPECT_EQ(ranking.question, clause.question);
        EXPECT_EQ(ranking.asked, clause.asked);

        // Every row the report places in this ranking — the recommended one included — is of
        // this class's precision, rung and question, and there is at least one such row.
        std::size_t rows = 0;

        for (const DeviceProbeMeasurement& measurement : report.measurements) {
            if (measurement.precision != clause.precision ||
                measurement.rung != clause.rung ||
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
                             [&ranking, &clause](const DeviceProbeMeasurement& measurement) {
                                 return measurement.name == ranking.recommended &&
                                        measurement.rung == clause.rung;
                             });

            ASSERT_NE(named, report.measurements.end());
            EXPECT_EQ(named->precision, clause.precision);
            EXPECT_EQ(named->rung, clause.rung);
            EXPECT_EQ(named->question, ranking.question);
        }
    }
}

/// A name is not a row: one entry is measured once at every rung the lane serves, so the
/// table carries the same name many times and a class names the one at its own rung.
TEST(DeviceProbe, ANameIsResolvedAtTheClassesOwnRung) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    // The sweep happened: a name appears as many times as the run took rungs, which is
    // precisely what a lookup by name alone would get wrong.
    bool sawRepeatedName = false;

    for (const DeviceProbeMeasurement& measurement : report.measurements) {
        std::size_t occurrences = 0;

        for (const DeviceProbeMeasurement& other : report.measurements) {
            occurrences += other.name == measurement.name ? 1u : 0u;
        }

        sawRepeatedName = sawRepeatedName || occurrences > 1;
    }

    EXPECT_TRUE(sawRepeatedName);

    std::size_t namedRows = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        if (clause.ranking.recommended.empty()) {
            continue;
        }

        std::size_t atRung = 0;

        for (const DeviceProbeMeasurement& measurement : report.measurements) {
            if (measurement.name == clause.ranking.recommended &&
                measurement.rung == clause.rung) {
                ++atRung;
            }
        }

        EXPECT_EQ(atRung, 1u) << clause.ranking.recommended << " at m = " << clause.rungName;
        ++namedRows;
    }

    // A class that named a winner named a row of its own rung and no other; a run whose
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
                             [&ranking, &clause](const DeviceProbeMeasurement& measurement) {
                                 return measurement.name == ranking.recommended &&
                                        measurement.rung == clause.rung;
                             });

            ASSERT_NE(named, report.measurements.end());
            EXPECT_TRUE(named->measured);
            EXPECT_EQ(named->question, ranking.question);
            EXPECT_EQ(named->precision, clause.precision);
        }
    }
}

/// A ranking is a comparison, so it needs two entries to be one. A shape whose table
/// holds one entry names it as the shape's only entry and not as the winner of a field
/// it never met — one entry is not a ranking, and not a refusal either.
///
/// The check is on the shape's own rows rather than on a figure, since whether an entry
/// measures at this workload is the card's business.
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
            // One precision at one rung for one question shape: counting without the rung
            // would call a class of one row a class of twelve as soon as the run measured
            // more than one rung.
            std::size_t rows = 0;
            std::size_t figureRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.precision != clause.precision ||
                    measurement.rung != clause.rung ||
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
                // A shape of one whose entry produced no figure is the case with nothing to
                // name. The count is the figures' and not the rows': a row that measured and
                // came out with no figure is this answer and not an exception to it.
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine);
                EXPECT_TRUE(ranking.recommended.empty());
                continue;
            }

            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend);
            EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kOnlyEntry);
            EXPECT_FALSE(ranking.recommended.empty());
            EXPECT_TRUE(ranking.tiedEntries.empty()) << "a shape of one has no rival to tie with";
            EXPECT_TRUE(ranking.inseparable.empty());

            // The line says what the name rests on, so that a reader does not take
            // it for the winner of a comparison.
            EXPECT_NE(ranking.reason.find("no alternative"), std::string::npos) << ranking.reason;
        }
    }

    EXPECT_TRUE(sawAShapeOfOne);
}

/// A control that agrees has two figures to compare: a zero left where the subtraction
/// resolved no cost is not a second reading, and a control calling that agreement would
/// pass on no evidence.
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

        // The two counts are judged against the resolution the row's own shape showed, and
        // the two figures are quartiles of the control's own rounds, so the yardstick is
        // measured rather than chosen here.
        EXPECT_GT(control->judgedAgainst, 0.0) << control->entry;
        EXPECT_LE(control->difference, control->judgedAgainst) << control->entry;
    }
}

/// The control's own statistic is the report's: the low and high counts are quartiles of
/// the control's rounds, which makes the two figures a check of the estimator rather than
/// of a pass.
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

        // The disagreement is computed from the same two figures and to the same precision,
        // so it cannot drift into a conclusion the numbers do not support.
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

/// The option space's rung axis, as a report carries it.
///
/// The rows are the library's own and the text is rendered by the same function
/// the driver prints, from a report whose measurement grid is the one the probe
/// builds — a cell per (row, rung) this build serves, none of them measured — so
/// the figures the appendix states about the cells are the figures a real run of
/// this revision states, without a card and without a clock.
///
/// A row that refuses eleven of the twelve rungs is what this is about. Such a
/// row used to be a row the space reported as served, with a reason field that
/// named the build seam instead of the rung, so a reader of the option space
/// concluded a caller could ask for arithmetic the entry does not answer with.
TEST(DeviceProbe, ARowStatesTheRungsItIsServedAtAndTheCellsAreCountedFromTheRows) {
    const std::span<const boys::DeviceOptionInfo> space = boys::BoysDeviceOptions();

    ASSERT_FALSE(space.empty());

    DeviceProbeReport report;
    report.status = DeviceProbeStatus::kSuccess;
    report.device.name = "a device";

    std::size_t refusedCells = 0;
    std::size_t partialRows = 0;
    std::size_t launchedRows = 0;
    std::size_t deviceRows = 0;
    std::vector<std::string> partialNames;

    for (const boys::DeviceOptionInfo& row : space)
    {
        (row.group == boys::DeviceOptionGroup::kLaunched ? launchedRows : deviceRows) += 1;

        // Every bit a row sets is a rung of the lane: a mask wider than the axis
        // would be a row claiming a rung the lane does not serve.
        EXPECT_EQ(row.servedRungs & ~boys::kEveryDeviceRung, 0u) << row.name;

        for (std::size_t i = 0; i < boys::kDeviceRungCount; ++i)
        {
            const bool stated = (row.servedRungs & (boys::DeviceRungMask{1} << i)) != 0;

            // The row's mask is the library's own answer per rung, and it is that
            // answer read through the one derivation rather than a second list.
            EXPECT_EQ(stated, boys::DeviceEntryServedAtRung(row.entry, boys::kDeviceRungs[i]))
                << row.name << " at m = " << boys::kDeviceRungs[i];
            EXPECT_EQ(stated,
                      (boys::DeviceServedRungMask(row.entry, row.built) &
                       (boys::DeviceRungMask{1} << i)) != 0)
                << row.name << " at m = " << boys::kDeviceRungs[i];

            if (!stated)
            {
                ++refusedCells;
                continue;
            }

            DeviceProbeMeasurement measurement;
            measurement.name = row.name;
            measurement.entry = row.entry;
            measurement.rung = boys::kDeviceRungs[i];
            report.measurements.push_back(measurement);
        }

        if (row.servedRungs != boys::kEveryDeviceRung)
        {
            ++partialRows;
            partialNames.push_back(row.name);
        }
    }

    // A build that does not serve a row serves it at no rung, whatever its
    // entry's axis states: the mask folds the build seam in rather than leaving a
    // reader to combine two columns.
    EXPECT_EQ(boys::DeviceServedRungMask(boys::DeviceEntry::kSingleF16, false), 0u);
    EXPECT_EQ(boys::DeviceServedRungMask(boys::DeviceEntry::kSingleF16, true),
              boys::kEveryDeviceRung);

    const std::string text = boys::FormatDeviceOptionProbe(report);

    const std::size_t cells = boys::kDeviceRungCount * space.size();
    const std::size_t servedCells = cells - refusedCells;
    std::size_t unbuiltCells = 0;

    // Stated rather than left to be counted off the report: these are the figures
    // the appendix carries, and a reader checking a revision needs them printed
    // rather than only asserted.
    std::printf("    space: %zu option(s) = %zu launched + %zu device-callable; %zu cell(s) at %zu "
                "rung(s): %zu served, %zu refused, %zu row(s) holding part of the axis\n",
                space.size(),
                launchedRows,
                deviceRows,
                cells,
                boys::kDeviceRungCount,
                servedCells,
                refusedCells,
                partialRows);

    for (const boys::DeviceOptionInfo& row : space)
    {
        if (!row.built)
        {
            unbuiltCells += boys::kDeviceRungCount;
        }
    }

    // The space's cells, served and refused, as the appendix states them: the
    // count a reader of this report takes for coverage.
    const std::string counted = std::to_string(boys::kDeviceRungCount) + " rung(s) the lane serves: " +
                                std::to_string(cells) + " cell(s)";
    EXPECT_NE(text.find(counted), std::string::npos) << text;

    EXPECT_NE(text.find(", " + std::to_string(servedCells) + " this build\n  serves and " +
                        std::to_string(refusedCells) + " it refuses"),
              std::string::npos)
        << text;

    EXPECT_NE(text.find("and " + std::to_string(unbuiltCells) +
                        " because the build does not serve the row at all"),
              std::string::npos)
        << text;

    // Every cell the space serves is a cell of this run's grid, and none of them
    // produced a figure: the two figures are the run's own and not the space's.
    EXPECT_NE(text.find("cells it serves, " + std::to_string(servedCells) +
                        " produced no figure here"),
              std::string::npos)
        << text;

    // The rows that hold part of the axis are named under the table, with the
    // rungs each holds: a figure below the whole says how many and not which, and
    // which is what a caller placing the row needs.
    ASSERT_GT(partialRows, 0u);

    EXPECT_NE(text.find(std::to_string(partialRows) +
                        " row(s) hold less than the whole of that axis"),
              std::string::npos)
        << text;

    for (const std::string& name : partialNames)
    {
        // The block's lines are indented four spaces and the table's two, so this
        // finds the line that names the rungs rather than the row in the table.
        const std::size_t at = text.find("\n    " + name);

        ASSERT_NE(at, std::string::npos) << name << " is not named among the rows that hold part "
                                                     "of the rung axis";

        const std::string line = text.substr(at, text.find('\n', at + 1) - at);

        std::size_t rungs = 0;
        std::size_t found = line.find("m = ");

        while (found != std::string::npos)
        {
            ++rungs;
            found = line.find("m = ", found + 1);
        }

        std::size_t held = 0;

        for (const boys::DeviceOptionInfo& row : space)
        {
            if (row.name == name)
            {
                held = static_cast<std::size_t>(std::popcount(row.servedRungs));
            }
        }

        EXPECT_EQ(rungs, held) << line;
    }
}

/// The two lists, tied where a reader meets them: a cell the report refuses is a
/// cell the entry refuses.
///
/// A row of the space states the rungs it is served at. Where the row is a
/// launched one, the entry it names is asked here at every rung the row does not
/// hold and must answer with the library's own refusal — \c kInvalidArgument,
/// before anything is made resident — rather than with arithmetic, which is the
/// claim a reader takes from the row.
///
/// A device-callable row has no such form to ask: its entry is a __device__
/// function, and what refuses a rung is the entry's own test inside the caller's
/// kernel. Those rows are not dropped — a partial one is checked for the shape
/// this build's refusals have, and the entry's own refusal is asked by the device
/// accuracy gate's rung sweep, which runs the entry in a kernel at a rung that is
/// not the resident one. The loop below says which rows are which.
///
/// The rows are the space's own and not a list here: a launched row the space
/// calls partial and this table cannot ask is a failure, and so is a row of this
/// table the space serves at every rung. A new partial row therefore cannot
/// arrive with only one of the two lists moved.
///
/// Nothing of the card is used: every call asked here is refused before the entry
/// reaches a table, a buffer or a launch, so this runs on a host with no device.
TEST(DeviceProbe, EveryCellTheReportRefusesIsRefusedByTheEntryThatOwnsTheRow) {
    std::vector<double> outF64(boys::kMaxBoysOrder + 1);
    std::vector<float> outF32(boys::kMaxBoysOrder + 1);

    /// One row of the table below: the entry, the output buffer its own precision
    /// takes, and the rung-argument form of its entry.
    struct RefusingRow {
        boys::DeviceEntry entry;
        void* out;
        boys::BoysStatus (*call)(double, const int*, const double*, void*, std::size_t);
    };

    // The uniform route's six rows are not in this table: their rung axis is the
    // lane's whole set, so they refuse no cell of it and no entry of theirs is
    // asked a refusal here. What a rung of that route delivers is the route's own
    // arithmetic — the table has no shorter image to read — and the figures a
    // rung of it is worth are the accuracy gate's and the consumer sweep's, both
    // of which follow the space and are not guarded on the multiplier.
    const std::vector<RefusingRow> refuses = {
        {boys::DeviceEntry::kAllOrdersF32Rat, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32RatAtRung(m, n, x, static_cast<float*>(out), count,
                                                          nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32RatHorner, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32RatHornerAtRung(m, n, x, static_cast<float*>(out),
                                                                count, nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32NarrowRat, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32NarrowRatAtRung(m, n, x, static_cast<float*>(out),
                                                                count, nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32NarrowRatHorner, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32NarrowRatHornerAtRung(
                 m, n, x, static_cast<float*>(out), count, nullptr);
         }},
        // The same tables on the packing axis's other side: the cell a rung of
        // them refuses is the cell the row above refuses, because the missing
        // cut is the table's and not the body's.
        {boys::DeviceEntry::kAllOrdersF32OrdersRat, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32OrdersRatAtRung(
                 m, n, x, static_cast<float*>(out), count, nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32OrdersRatHorner, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32OrdersRatHornerAtRung(
                 m, n, x, static_cast<float*>(out), count, nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32NarrowOrdersRat, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32NarrowOrdersRatAtRung(
                 m, n, x, static_cast<float*>(out), count, nullptr);
         }},
        {boys::DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner, outF32.data(),
         [](double m, const int* n, const double* x, void* out, std::size_t count) {
             return boys::BoysCuda::AllOrdersF32NarrowOrdersRatHornerAtRung(
                 m, n, x, static_cast<float*>(out), count, nullptr);
         }},
    };

    const std::vector<int> n(4, 4);
    const std::vector<double> x(4, 0.5);

    std::size_t asked = 0;
    std::size_t devicePartial = 0;

    for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
    {
        if (row.servedRungs == boys::kEveryDeviceRung)
        {
            continue;
        }

        // A device-callable row's refusal is produced inside the caller's own
        // kernel and cannot be asked from here: the entry it names is a
        // __device__ function with no launcher of this library's behind it, and
        // what refuses the call is the entry's own test against the rung the
        // handle holds. The evidence for those rows is the device accuracy
        // gate's rung sweep (CheckRungCalls, tests/boys_cuda_accuracy_gate.cpp),
        // which asks every device row at a rung that is not the resident one and
        // requires the entry's own refusal with nothing written.
        //
        // What is checked here is the shape of this build's refusals: a partial
        // device-callable row is one whose tables have no rung cut — the float
        // lane's rational route alone, now that the narrow partition's two bases
        // are cut here — so a partial row of the double lane, whose every rung
        // cut this build derives, uploads and reads, is a finding rather than a
        // property of the row.
        if (row.group == boys::DeviceOptionGroup::kDeviceCallable)
        {
            ++devicePartial;

            EXPECT_EQ(row.precision, boys::DeviceOptionPrecision::kFp32)
                << row.name
                << " holds part of the rung axis on a lane whose rung cuts this build holds";

            continue;
        }

        const RefusingRow* refused = nullptr;

        for (const RefusingRow& candidate : refuses)
        {
            if (candidate.entry == row.entry)
            {
                refused = &candidate;
            }
        }

        ASSERT_NE(refused, nullptr)
            << row.name << " is served at part of the rung axis and this test cannot ask its "
                           "entry at the rungs the row refuses";

        for (std::size_t i = 0; i < boys::kDeviceRungCount; ++i)
        {
            if ((row.servedRungs & (boys::DeviceRungMask{1} << i)) != 0)
            {
                continue;
            }

            ++asked;

            EXPECT_EQ(refused->call(boys::kDeviceRungs[i], n.data(), x.data(), refused->out, 1),
                      boys::BoysStatus::kInvalidArgument)
                << row.name << " at m = " << boys::kDeviceRungs[i];
        }
    }

    EXPECT_GT(asked, 0u);

    EXPECT_GT(devicePartial, 0u);

    // Stated rather than left to be counted off the space: this is the number of
    // cells the report calls refused and the entries answered for, and it is the
    // one figure of this test a reader cannot get from a passing assertion.
    std::printf("    %zu cell(s) the report refuses, every one refused by its own entry\n", asked);
    std::printf("    %zu device-callable row(s) hold part of the axis, refused inside the "
                "caller's kernel\n",
                devicePartial);

    // The other direction: a row of the table above that the space serves at
    // every rung would be an entry this test asks about a refusal the library
    // does not make.
    for (const RefusingRow& refused : refuses)
    {
        bool partial = false;

        for (const boys::DeviceOptionInfo& row : boys::BoysDeviceOptions())
        {
            partial = partial || (row.entry == refused.entry &&
                                  row.servedRungs != boys::kEveryDeviceRung);
        }

        EXPECT_TRUE(partial) << "a row this test holds a refusal for is one the space serves at "
                                "every rung";
    }
}

} // namespace
