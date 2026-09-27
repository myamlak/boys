// The device option probe (boys/boys_cuda_probe.hpp): the protocol it takes its
// figures under, what it reports about the card, and what it does when the
// caller names a card that is not there.
//
// The probe's figures are this card's, and asserting them here would pin a
// number that describes one host, so nothing below asserts a cost. What is
// asserted is the part a test can own: the shape of the comparison — that the
// ratio between two entries is formed inside a round and that every pooled round
// is used, that a wide canary flags a pass without removing it, that a canary no
// pass ever read is reported as no reading rather than as a still one, that a run
// too short for a quartile band says how short it is and still names one entry per
// shape that produced a figure — as does a shape its own rounds cannot order, by
// the refinement stage's vote or by there being no alternative to name — that such
// a shape carries the same clock check a shape ordered outright does, and that
// every name the report carries came out of a clock rather than off the
// library's tables. Plus the parts that were here before and still hold: that a
// bad device is a status and not a crash, that the device it measured is named
// in the returned data rather than only in a log, and that a ranking is one
// question shape of one precision.
//
// The protocols below are short on purpose. The canary's alarm is moved to each
// end so that the flag is exercised from both sides on a real run, and the
// workload is small because what is being checked is the shape of the result
// rather than its figures. One test is not short: the tie test hunts a tie, which
// means running the refinement stage over everything the shape could not separate,
// and on the card this was written on it took about 52 minutes where every other
// test in the file finished in seconds. A run that looks stuck is usually inside
// it, and nothing is hung while it is.

#include "boys/boys_cuda_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
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

/// A workload a test can afford, with the passes kept short but enough of them.
///
/// Two passes of two rounds is four pooled rounds, which is exactly the least a
/// lower-quartile band can be formed from. A run below that is a refusal by
/// construction, so every test that wants a run to be able to order anything has
/// to keep at least this much.
DeviceProbeOptions Small() {
    DeviceProbeOptions options;
    options.count = 1u << 12;
    options.nmax = 8;
    options.passes = 2;
    options.rounds = 2;
    options.repetitions = 4;
    return options;
}

/// A pass whose canary disagreed with itself is flagged and is used anyway.
///
/// The alarm is an alarm and not an admission rule: the canary is fixed work
/// read by a device clock, so it measures the clock as much as the load, and a
/// pass discarded on its own canary would be the measurement discarded rather
/// than the machine. The check is that the flag goes up on every pass when the
/// alarm is zero, that it stays down when the alarm is out of reach, and that
/// the run pooled the same rounds either way — which is what "used" means here.
TEST(DeviceProbe, APassAboveTheCanaryAlarmIsFlaggedAndStillUsed) {
    DeviceProbeOptions flagged = Small();
    flagged.canarySpreadAlarm = 0.0;

    const DeviceProbeReport wide = boys::RunDeviceOptionProbe(flagged);

    ASSERT_EQ(wide.status, DeviceProbeStatus::kSuccess);

    EXPECT_EQ(wide.passes.size(), static_cast<std::size_t>(flagged.passes));
    EXPECT_EQ(wide.passesWithinAlarm, 0);
    EXPECT_EQ(wide.pairedRounds, flagged.passes * flagged.rounds);

    // The three counters are the pass table: the two the canary placed, and the
    // passes it was not read in. With the alarm at zero every pass a reading was
    // taken in is above it, so the count above is the run's passes less the ones
    // with no reading — and the sum is the table either way.
    EXPECT_EQ(wide.passesWithinAlarm + wide.passesAboveAlarm + wide.passesWithoutCanary,
              static_cast<int>(wide.passes.size()));
    EXPECT_EQ(wide.passesAboveAlarm + wide.passesWithoutCanary, flagged.passes);

    for (const boys::DeviceProbePass& pass : wide.passes) {
        // The flag is the run's own rule applied to the pass's own readings, so
        // it is checked against them rather than against the alarm alone, and a
        // pass whose canary was not read is flagged by nothing.
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

/// A canary that never ran is not a canary that stayed quiet.
///
/// The two are different facts about a machine and the second must never be
/// printed for the first: a spread of zero is a clock that did not move, and no
/// reading at all is a clock nobody looked at. The report is built here rather
/// than run, because a device cannot be asked to fail its canary on command —
/// what is checked is the one place the zero would be printed, so a pass whose
/// canary was not timed must say so in words and put no figure where a reading
/// would go.
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

    // The words, and not a zero: the median would print as 0.0000 and the spread
    // as 0.00 where a reading belongs, and that is the reading this test exists
    // to fail on.
    EXPECT_NE(row.find("canary not timed"), std::string::npos) << row;
    EXPECT_EQ(row.find("0.0000"), std::string::npos) << row;

    // The one reading, one pass later: a measured pass prints its figures and the
    // flag it earned, so the words above are the absence of a reading and not a
    // column that never carries one.
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

    // The resolution line carries the canary beside it only where the canary was
    // read: a shape whose run has no canary spread must not be handed a zero to
    // read as one.
    DeviceProbeRanking refused;
    refused.question = "all-orders";
    refused.asked = "every order";
    // A run with a resolution to print: below four paired rounds the report says
    // so instead of printing a band, and the line this part of the test is about
    // is the one that carries a resolution.
    refused.rounds = 10;
    refused.resolution = 0.25;
    refused.verdict = DeviceProbeVerdict::kCannotDetermine;
    refused.fastestOverall = "device-all-orders-fp64";
    refused.reason = "two entry(s) of this shape could not be placed";
    refused.confidence = "CANNOT DETERMINE: the widest band this shape showed was 25.00%";

    DeviceProbeClass clause;
    clause.precision = "fp64";
    clause.note = "note";
    clause.rankings = {refused};

    DeviceProbeReport withShape = unread;
    withShape.classes = {clause};

    const std::string shapeText = boys::FormatDeviceOptionProbe(withShape);

    EXPECT_NE(shapeText.find("No pass took a canary reading on this run"), std::string::npos)
        << shapeText;
    EXPECT_EQ(shapeText.find("canary spread, 0.00%"), std::string::npos) << shapeText;
}

/// Every figure rests on every pooled round, and the reference entry's own ratio
/// to itself is exactly one with exactly no drift.
///
/// This is what makes the comparison paired. A figure formed by pooling rounds
/// would let a row rest on a subset of them, and a cost taken under a different
/// clock than the one it is compared against; the report says how many rounds a
/// row rests on, and every measured row has to name all of them. The reference
/// entry's ratio is the one number in the table that is known independently of
/// the card — it is its own ratio — so it is the one place a test can pin an
/// exact value, and it pins that the ratios are formed per round and not across
/// rounds.
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
        // A ratio's two ends are formed from two per-round figures, so both are
        // positive whenever the row has a figure at all. An in-kernel row whose
        // subtraction came out at or below its own baseline has none: its cell is
        // the zero it was floored at, and that is exactly the case the report
        // labels unresolved and sets aside from ordering. The two statements have
        // to agree, and this fixture's workload — the smallest one the probe takes
        // — is where the in-kernel rows sit inside their own kernel's traffic, so
        // the test asserts the agreement rather than assuming every row resolved.
        if (measurement.subtractionResolved) {
            EXPECT_GT(measurement.ratioLo, 0.0) << measurement.name;
        } else {
            EXPECT_LE(measurement.ratioLo, 0.0) << measurement.name;
        }
        EXPECT_GE(measurement.spread, 1.0) << measurement.name;

        // The reported cost is the reference's own figure scaled by this row's
        // ratio to it, so the two columns cannot come from different statistics.
        EXPECT_NEAR(measurement.nsPerArgument,
                    report.referenceNsPerArgument * measurement.ratioToReference,
                    1e-9 * std::max(1.0, report.referenceNsPerArgument)) << measurement.name;
        EXPECT_NEAR(measurement.nsPerArgumentMax,
                    report.referenceNsPerArgument * measurement.ratioHi,
                    1e-9 * std::max(1.0, report.referenceNsPerArgument)) << measurement.name;

        // The peak column is the entry's own fastest single round: a raw figure
        // under no anchor, beside a reported cost that is a quartile of ratios
        // anchored to the reference. The two are statistics of different
        // quantities, so neither bounds the other in general — a reference whose
        // own rounds are heavy-tailed drags the anchored cost below an entry's own
        // fastest round, which is precisely the failure the old statistic, formed
        // from the minimum, used to hide. What is asserted here is what the column
        // promises in its own terms, and the bound is asserted where it is real:
        // on the reference entry, whose two columns come from one entry's own
        // rounds, where the minimum of a set can never exceed a quartile of it.
        EXPECT_TRUE(std::isfinite(measurement.nsPerArgumentPeak)) << measurement.name;
        EXPECT_GE(measurement.nsPerArgumentPeak, 0.0) << measurement.name;

        if (measurement.name != report.referenceEntry) {
            continue;
        }

        sawReference = true;
        // The reference entry against itself: the ratio is one in every round by
        // construction, so its lower quartile is exactly one and the movement of
        // that ratio between the run's halves is exactly zero.
        EXPECT_DOUBLE_EQ(measurement.ratioToReference, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioLo, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioHi, 1.0);
        EXPECT_DOUBLE_EQ(measurement.ratioDrift, 0.0);
        EXPECT_DOUBLE_EQ(measurement.nsPerArgument, report.referenceNsPerArgument);
        EXPECT_LE(measurement.nsPerArgumentPeak, measurement.nsPerArgument + 1e-12);
    }

    EXPECT_TRUE(sawReference);
    EXPECT_GT(measured, 0u);
}

/// A run too short for a quartile band still ends with one entry per shape that
/// produced a figure, and says the count it took and the count a band needs.
///
/// A band's two ends are two order statistics; below four readings they are the
/// same reading twice, so such a run places nothing and can place nothing. What it
/// may not do is leave a shape that produced a figure without an answer: the name
/// is the one entry such a shape has, or the entry the refinement stage's own
/// longer protocol led, and the reason carries the round count so that the name is
/// never read as a comparison this run did not make.
TEST(DeviceProbe, AShortRunNamesAnEntryAndSaysTheBandWasNeverFormed) {
    DeviceProbeOptions options = Small();
    // The counts the two readings are spaced by are the ones the protocol is read
    // at, and the run is short in rounds rather than in workload: a fixture small
    // in both is one whose entries may not resolve, and this rule is about rounds.
    options.count = 1u << 18;
    options.passes = 1;
    options.repetitions = 8;
    options.refinementRuns = 2;
    options.refinementFactor = 2;
    options.canarySpreadAlarm = 1.0e9;
    // One shape of two rows - one launched and one in-kernel - and two shapes of
    // one, so the run holds a name reached by the stage and a name reached by there
    // being no rival at all.
    options.only = {"single-fp64", "device-single-fp64", "single-fp32", "all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_EQ(report.pairedRounds, options.passes * options.rounds);
    ASSERT_LT(report.pairedRounds, 4);
    ASSERT_FALSE(report.classes.empty());

    std::size_t named = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            EXPECT_EQ(ranking.rounds, report.pairedRounds);

            // Nothing can be placed on two readings, and the reason says the count
            // it took and the count it needs rather than leaving the shortfall to
            // be inferred from the name below.
            EXPECT_NE(ranking.reason.find("2 paired round(s)"), std::string::npos)
                << ranking.reason;
            EXPECT_NE(ranking.reason.find("need 4"), std::string::npos) << ranking.reason;

            // No shape of this run is reported as an ordering: the band one would
            // have been read off was never formed, whichever way the name was
            // reached.
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kOrdered) << ranking.question;

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
                // The one answer a run cannot name an entry from: no entry of this
                // shape produced a figure at all.
                EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kCannotDetermine)
                    << ranking.question;
                EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;
                EXPECT_EQ(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
                continue;
            }

            // Entries of this shape produced figures, so it ends with exactly one
            // name - and the name is one of those entries, never a row read off a
            // table.
            ++named;
            EXPECT_EQ(ranking.verdict, DeviceProbeVerdict::kRecommend) << ranking.question;
            EXPECT_FALSE(ranking.recommended.empty()) << ranking.question;
            EXPECT_NE(ranking.defaultHow, DeviceProbeDefaultHow::kNone) << ranking.question;
            EXPECT_TRUE(namedIsRead)
                << ranking.recommended << " is not an entry this shape read a figure for";
        }
    }

    // The premise the rule above is tested under: at these counts the entries
    // resolve. A run that read nothing is a fixture to look at, not a rule to
    // relax, and it fails here rather than passing vacuously.
    EXPECT_GT(named, 0u) << "no shape of this run read a figure at " << options.count
                         << " arguments, so the naming rule was never reached";

    // And the text: a band of two readings is not a resolution, so a run this
    // short prints none. The sentence that would carry one — "entries whose
    // within-round ratio band is narrower than X cannot be ordered on this run" —
    // implies that bands above X could be ordered, which is exactly what this run
    // cannot say, so the line names the count it is missing instead.
    const std::string text = boys::FormatDeviceOptionProbe(report);

    EXPECT_EQ(text.find("cannot be ordered on this run"), std::string::npos) << text;
    EXPECT_NE(text.find("resolution: not measurable on this run: it took 2 paired round(s)"),
              std::string::npos)
        << text;
}

/// A shape its own rounds could not order still ends with exactly one entry, and
/// says how it reached the name.
///
/// A rival whose within-round band against the leader straddles one is a pair this
/// run did not separate, in either direction, and a probe that named the leader
/// anyway would be ordering noise. What it does instead is re-run the entries it
/// could not separate, alone, at a longer protocol, and name the one that led
/// those runs. The check is the rule itself, ranking by ranking: a shape reached
/// by its own rounds leaves no rival unplaced, a shape reached any other way
/// carries the route with the name, and a shape whose entries produced no figure
/// is the only one that names nothing.
TEST(DeviceProbe, AShapeItsRoundsCouldNotOrderIsNamedByTheRefinementStage) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // The named set holds shapes of one entry, which are shapes that cannot be
    // ordered and are not refusals either, so the run below holds a name reached
    // without an ordering for this rule to check.
    options.only = {"all-n-fp64", "single-fp64", "device-single-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    std::size_t namedWithoutAnOrdering = 0;
    std::size_t refused = 0;
    bool anyMeasured = false;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            EXPECT_FALSE(ranking.asked.empty());

            std::size_t measuredRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.measured && measurement.precision == clause.precision &&
                    measurement.question == ranking.question) {
                    ++measuredRows;
                }
            }

            anyMeasured = anyMeasured || measuredRows > 0;

            if (measuredRows == 0) {
                // Nothing was timed, so there is no name to reach: this is the one
                // answer a run cannot name an entry from.
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
                EXPECT_EQ(measuredRows, 1u)
                    << "a shape that was ordered against nothing holds one measured row";
                EXPECT_TRUE(ranking.tiedEntries.empty());
                continue;
            }

            // A tie: the stage ran, it voted over the entries the shape's own
            // rounds could not separate, and the name is the entry it chose. The
            // vote is reported run by run, so the name can be read back.
            EXPECT_TRUE(ranking.refinement.ran) << ranking.question;
            EXPECT_EQ(ranking.refinement.winner, ranking.recommended) << ranking.question;
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

/// A shape that could not place a rival behind its leader still names every rival
/// and the band that pair fell in.
///
/// A tie is a result, and its evidence is the part a reader has to be able to
/// judge: each unplaced rival is named with the band its ratio to the leader fell
/// in and with how often it was the slower of the two — not summarised, not
/// counted. The check is by name where the card produces a tie and is silent where
/// it orders every shape outright, because whether two entries come that close is
/// the card's answer rather than this test's.
///
/// This is the file's long pole by a wide margin — about 52 minutes on the card
/// this was written on, where every other test here takes seconds — because the
/// tie is hunted rather than waited for: the shape has to reach the refinement
/// stage and be voted on, and the stage is what costs the time.
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
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            for (const std::string& line : ranking.inseparable) {
                EXPECT_NE(line.find("fell in"), std::string::npos) << line;
                EXPECT_NE(line.find("slower of the two in"), std::string::npos) << line;
            }

            if (!ranking.inseparable.empty()) {
                EXPECT_NE(ranking.reason.find("could not be placed"), std::string::npos)
                    << ranking.reason;
            }

            // A shape whose tie was refined still names the rivals it could not
            // separate: the refinement answers the shape, and it does not erase
            // what the shape's own rounds found.
            if (ranking.refinement.ran) {
                EXPECT_FALSE(ranking.tiedEntries.empty()) << ranking.question;
            }
        }
    }
}

/// A refusal carries the clock check as well as a recommendation does.
///
/// The check is whether any pair's ratio to the leader moved between the run's
/// first and second half of rounds by more than the run can order, and a refusal
/// is exactly where a reader wants it: it is the difference between a shape whose
/// entries are too close to separate and a shape the machine moved under. Every
/// ranking that followed a pair carries one of the two sentences, recommendation
/// or refusal alike, and that part cannot be absent — the refusal is the case the
/// sentence was missing from, so it is checked by name where the run produces one.
///
/// Whether a run produces one is the card's answer rather than the test's: a shape
/// whose entries separate by more than the resolution this run measured orders
/// itself and has nothing left unplaced, so a run with no unplaced rival skips
/// rather than failing — the case is exercised where the machine makes it, and the
/// assertions above it run either way, so the check is never left untested.
TEST(DeviceProbe, ARefusalCarriesTheClockCheckToo) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // Enough rounds for a quartile band and a workload the in-kernel rows resolve
    // at: the pair this test is about is two rows of one kernel whose costs differ
    // by a few percent, which is what leaves one of them unplaced.
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
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            // One of the two sentences, and both of them are the check: the
            // warning where a pair moved further than the run can order, and the
            // statement that none did. Nothing else may be the last word.
            const bool warned = ranking.confidence.find("WARNING: the pair") != std::string::npos;
            const bool held =
                ranking.confidence.find("No pair's ratio moved") != std::string::npos;

            // What the check is about is a pair this run followed: a recommendation
            // followed every rival of its shape, and a refusal that names an
            // unplaced rival followed that one. A refusal that stopped earlier —
            // one entry alone, nothing measured, nothing resolved, a run too short
            // for a band — followed no pair, and it must not carry a sentence about
            // the clock having held, which would be a claim about a comparison this
            // run never made.
            const bool followedAPair =
                ranking.verdict == DeviceProbeVerdict::kRecommend || !ranking.inseparable.empty();

            if (!followedAPair) {
                EXPECT_FALSE(warned) << ranking.confidence;
                EXPECT_FALSE(held) << ranking.confidence;
                continue;
            }

            ++followedPairs;

            EXPECT_TRUE(warned || held) << ranking.confidence;

            // The sentence names what "moved" was measured against — the run's own
            // resolution — so a reader can judge the word rather than take it.
            EXPECT_NE(ranking.confidence.find("this run can order"), std::string::npos)
                << ranking.confidence;

            if (!ranking.inseparable.empty()) {
                ++unplacedRankings;

                // The refusal names the rival it could not place beside the
                // sentence about the clock: the two together are what a reader has
                // to judge the refusal by.
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

/// Every name the report carries came out of a clock.
///
/// What a shape the probe could not order used to leave behind was a name read off
/// the library's own tables, printed in a section of its own and labelled a
/// heuristic. That route is gone: the entries a shape could not separate are
/// re-run and voted on, so a name the report carries is always a row this run
/// timed, and a shape whose rows produced no figure names nothing at all rather
/// than falling back to the tables. The check is both halves — the words the old
/// section lived under are absent, and each name the data carries belongs to a
/// measured row of the shape it is named in.
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
        for (const DeviceProbeRanking& ranking : clause.rankings) {
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

/// The status a run ends in, and whether the report says so in a way a reader
/// can act on. A run that did not succeed has no figures, and every field below
/// the status is described as meaningful only on success, so the check is that
/// the reason is present and that the device block is not silently blank when
/// the runtime could name a card.
TEST(DeviceProbe, ADisturbedRunRefusesWithAReason) {
    DeviceProbeOptions options = Small();
    // The alarm at zero flags every pass, which is the most disturbed a run can
    // be by the canary's own reading. The run is not filtered by it, so what is
    // checked is that the flag came up and that a refusal still carries a reason
    // rather than a bare absence.
    options.canarySpreadAlarm = 0.0;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    EXPECT_EQ(report.passesAboveAlarm, options.passes);

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
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

    // The device the figures are about, in the returned data and not only in
    // the text: a consumer that reads the struct has to be able to learn what
    // card answered.
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
    // The protocol's own bookkeeping, in the run and not only in the text: the
    // passes the canary placed and the passes it was not read in are the passes
    // the run took, so no pass is unaccounted for and none was dropped.
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

/// A class is one precision and a ranking inside it is one question shape: no
/// entry is ever placed against an entry of another precision or of another
/// shape. That is the rule the ranking is read under, so it is the rule checked
/// here, row by row rather than by the names the report happens to print.
TEST(DeviceProbe, AClassIsOnePrecisionAndARankingIsOneShape) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_FALSE(report.classes.empty());

    for (std::size_t c = 0; c < report.classes.size(); ++c) {
        const DeviceProbeClass& clause = report.classes[c];

        EXPECT_FALSE(clause.precision.empty());
        EXPECT_FALSE(clause.note.empty());
        EXPECT_FALSE(clause.rankings.empty());

        // One class per precision, so no two classes are the same precision.
        for (std::size_t other = c + 1; other < report.classes.size(); ++other) {
            EXPECT_NE(report.classes[other].precision, clause.precision);
        }

        for (std::size_t r = 0; r < clause.rankings.size(); ++r) {
            const DeviceProbeRanking& ranking = clause.rankings[r];

            EXPECT_FALSE(ranking.question.empty());
            EXPECT_FALSE(ranking.asked.empty());

            // One ranking per shape, so no two rankings of one class ask the
            // same question.
            for (std::size_t other = r + 1; other < clause.rankings.size(); ++other) {
                EXPECT_NE(clause.rankings[other].question, ranking.question);
            }

            // Every row the report places in this ranking — the recommended one
            // included — is of this class's precision and this ranking's
            // question, and there is at least one such row in the table.
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
}

/// A class that names a winner names one it measured, and it says how far the
/// nearest rival was. A class that does not names what it could not separate.
TEST(DeviceProbe, AVerdictNamesOnlyEntriesItMeasured) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            EXPECT_FALSE(ranking.asked.empty());

            if (ranking.verdict != DeviceProbeVerdict::kRecommend) {
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
/// says so: the entry is named, and named as the shape's only entry rather than as
/// the winner of a field it never met. One entry is not a ranking, and it is not a
/// refusal either — there is no alternative the run declined to name.
///
/// The check is on the shape's own rows rather than on a figure, because whether
/// an entry measures at this workload is the card's business and this run is not
/// asserting a cost.
TEST(DeviceProbe, AShapeOfOneNamesItsOnlyEntry) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    bool sawAShapeOfOne = false;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            std::size_t rows = 0;
            std::size_t measuredRows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.precision == clause.precision &&
                    measurement.question == ranking.question) {
                    ++rows;

                    if (measurement.measured) {
                        ++measuredRows;
                    }
                }
            }

            if (rows >= 2) {
                continue;
            }

            sawAShapeOfOne = true;
            EXPECT_FALSE(ranking.reason.empty());
            EXPECT_FALSE(ranking.confidence.empty());

            if (measuredRows == 0) {
                // A shape of one whose entry produced no figure on this run is the
                // one case with nothing to name, and it says which count is short
                // rather than naming an entry it did not time.
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

/// A control that agrees has two figures to compare. A count at which the
/// subtraction resolved no cost leaves a zero, and a zero is not a second
/// reading: a control that called that agreement would be passing on no
/// evidence, which is the one thing this check exists to prevent.
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

        // The two counts are judged against the resolution the row's own shape
        // showed, and the two figures are quartiles of the control's own rounds,
        // so the yardstick is a measured number and not a bar chosen here.
        EXPECT_GT(control->judgedAgainst, 0.0) << control->entry;
        EXPECT_LE(control->difference, control->judgedAgainst) << control->entry;
    }
}

/// The control's own statistic is the one the report uses: the low and high
/// counts are lower quartiles of the control's rounds, which is what makes the
/// two figures a check of the estimator rather than of a pass.
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

        // The disagreement the note reports is the two figures it reports, in
        // the same breath and to the same precision, so the two cannot drift
        // apart into a conclusion the numbers do not support.
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

} // namespace
