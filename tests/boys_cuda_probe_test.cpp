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
// too short for a quartile band refuses and says how short it is, that a refusal
// names what it could not order, recommends nothing, and carries the same clock
// check a recommendation does, and that the fallback a refusal leaves behind is
// labelled a heuristic and never presented as a measurement. Plus the parts that
// were here before and still hold: that a bad device is a status and not a crash,
// that the device it measured is named in the returned data rather than only in a
// log, and that a ranking is one question shape of one precision.
//
// The protocols below are short on purpose. The canary's alarm is moved to each
// end so that the flag is exercised from both sides on a real run, and the
// workload is small because what is being checked is the shape of the result
// rather than its figures.

#include "boys/boys_cuda_probe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <gtest/gtest.h>
#include <string>

namespace {

using boys::DeviceProbeClass;
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

/// A run too short for a quartile band refuses, names the count it took, and
/// names the count it needs.
///
/// A band's two ends are two order statistics; below four readings they are the
/// same reading twice, and a probe that ordered on them would be printing a
/// resolution it never measured. The refusal is a refusal and not a quiet
/// downgrade: no shape of such a run may recommend anything.
TEST(DeviceProbe, AShortRunRefusesAndNamesTheRoundCount) {
    DeviceProbeOptions options = Small();
    options.passes = 1;
    options.canarySpreadAlarm = 1.0e9;

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);
    ASSERT_EQ(report.pairedRounds, options.passes * options.rounds);
    ASSERT_LT(report.pairedRounds, 4);

    std::size_t refusals = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            EXPECT_EQ(ranking.rounds, report.pairedRounds);
            EXPECT_NE(ranking.verdict, DeviceProbeVerdict::kRecommend);
            EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;

            // The refusal names the count it took where the count is what
            // stopped it. A shape that stopped earlier — nothing measured, or
            // nothing that resolved above its own baseline — says that instead,
            // and both are refusals, so the reason is non-empty either way.
            EXPECT_FALSE(ranking.reason.empty()) << ranking.question;
            EXPECT_NE(ranking.confidence.find("CANNOT DETERMINE"), std::string::npos)
                << ranking.confidence;

            if (ranking.fastestOverall.empty()) {
                continue;
            }

            ++refusals;

            // Four rounds is what the protocol needs and what the refusal has to
            // say it needs, in the same breath as the number it took.
            EXPECT_NE(ranking.reason.find("2 paired round(s)"), std::string::npos) << ranking.reason;
            EXPECT_NE(ranking.reason.find("need 4"), std::string::npos) << ranking.reason;
        }
    }

    EXPECT_GT(refusals, 0u);

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

/// A refusal names what it could not order and recommends nothing.
///
/// A rival whose within-round band against the leader straddles one is a pair
/// this run did not separate, in either direction, and a probe that named the
/// leader anyway would be ordering noise. The check is the rule itself, ranking
/// by ranking: a recommendation is non-empty only where the verdict says so, and
/// every rival left unplaced is named with its band and with how often it was the
/// slower of the two — not summarised, not counted.
TEST(DeviceProbe, ARefusalNamesWhatItCouldNotOrderAndRecommendsNothing) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    // One shape of one entry is a shape that cannot be ordered, so the run below
    // always holds a refusal to check the rule against.
    options.only = {"all-n-fp64", "single-fp64", "device-single-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    std::size_t refusals = 0;
    std::size_t named = 0;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            EXPECT_FALSE(ranking.asked.empty());

            if (ranking.verdict == DeviceProbeVerdict::kRecommend) {
                EXPECT_FALSE(ranking.recommended.empty());
                // A recommendation is only reached past every rival, so no rival
                // of a recommended shape is left unplaced.
                EXPECT_TRUE(ranking.inseparable.empty()) << ranking.recommended;
                continue;
            }

            ++refusals;

            EXPECT_TRUE(ranking.recommended.empty()) << ranking.question;
            EXPECT_FALSE(ranking.reason.empty()) << ranking.question;
            EXPECT_NE(ranking.confidence.find("CANNOT DETERMINE"), std::string::npos)
                << ranking.confidence;

            for (const std::string& line : ranking.inseparable) {
                ++named;

                // Each unplaced rival carries its own band and its own
                // slower-round count: a refusal that did not say how close the
                // pair came, and how often, would leave a reader with nothing to
                // judge the refusal by.
                EXPECT_NE(line.find("fell in"), std::string::npos) << line;
                EXPECT_NE(line.find("slower of the two in"), std::string::npos) << line;
            }

            if (!ranking.inseparable.empty()) {
                EXPECT_NE(ranking.reason.find("could not be placed"), std::string::npos)
                    << ranking.reason;
            }
        }
    }

    EXPECT_GT(refusals, 0u);
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

/// A refusal leaves a consumer a default, and the default is labelled as what it
/// is: a reading of the library's own device entry book, counted and not timed.
///
/// The section is the report's own words, so it is the text that is checked: the
/// heading says a heuristic and not a measurement, the basis line begins with
/// the words that say where the name came from, and no cost is stated in the
/// section at all — a figure there would be read as one of the run's.
TEST(DeviceProbe, TheFallbackIsLabelledAndIsNotAMeasurement) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    const std::string text = boys::FormatDeviceOptionProbe(report);

    const std::string heading = "static fallback — a heuristic, not a measurement";
    const std::size_t at = text.find(heading);

    ASSERT_NE(at, std::string::npos);

    // A refusal that carries a name carries the basis it was chosen on, and the
    // basis says in its own first words that nothing was timed for it.
    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            if (ranking.verdict == DeviceProbeVerdict::kRecommend) {
                continue;
            }

            ASSERT_TRUE(ranking.heuristicBasis.empty() || !ranking.heuristicEntry.empty());

            if (ranking.heuristicEntry.empty()) {
                continue;
            }

            // The basis must open with the words that say it was counted rather
            // than timed — the whole length of them, so a basis that only started
            // with "counted" would not pass.
            const char* const opening = "counted, not timed";
            EXPECT_EQ(ranking.heuristicBasis.compare(0, std::strlen(opening), opening), 0)
                << ranking.heuristicBasis;
            EXPECT_NE(text.find(ranking.heuristicEntry), std::string::npos);
            EXPECT_NE(text.find(ranking.heuristicBasis), std::string::npos);
        }
    }

    // The section runs to the next heading of the report, and holds no cost: the
    // name in it is a property of the tables, and a figure beside it would make
    // the two read as one answer.
    std::size_t end = text.find("\n  static fallback", at + heading.size());
    const std::size_t nextClass = text.find("\nclass ", at + heading.size());

    if (end == std::string::npos || (nextClass != std::string::npos && nextClass < end)) {
        end = nextClass;
    }

    const std::string section = text.substr(at, end == std::string::npos ? std::string::npos
                                                                         : end - at);

    EXPECT_NE(section.find("not a measurement"), std::string::npos);
    EXPECT_NE(section.find("counted, not timed"), std::string::npos);
    EXPECT_EQ(section.find("ns/argument"), std::string::npos) << section;

    // The library's own book names a row for this shape, so the section is a
    // default and not an empty heading.
    EXPECT_FALSE(report.hasDefault && section.find("default: ") == std::string::npos);
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
/// declines rather than naming it: what such an entry is, is the fastest of the
/// one entry of its shape, and what it is not, is a result against a field.
///
/// The check is on the shape's own rows rather than on a figure, because whether
/// an entry measures at this workload is the card's business and this run is not
/// asserting a cost.
TEST(DeviceProbe, AShapeOfOneNamesNoWinner) {
    DeviceProbeOptions options = Small();
    options.canarySpreadAlarm = 1.0e9;
    options.only = {"all-n-fp64"};

    const DeviceProbeReport report = boys::RunDeviceOptionProbe(options);

    ASSERT_EQ(report.status, DeviceProbeStatus::kSuccess);

    bool sawAShapeOfOne = false;

    for (const DeviceProbeClass& clause : report.classes) {
        for (const DeviceProbeRanking& ranking : clause.rankings) {
            std::size_t rows = 0;

            for (const DeviceProbeMeasurement& measurement : report.measurements) {
                if (measurement.precision == clause.precision &&
                    measurement.question == ranking.question) {
                    ++rows;
                }
            }

            if (rows >= 2) {
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
