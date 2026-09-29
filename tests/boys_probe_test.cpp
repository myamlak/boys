// The option probe (boys/boys_probe.hpp): what it measures, what it refuses to
// claim, and that the second does not leave a consumer without an answer.
//
// The probe's whole value is in the claims a test can check without owning the
// machine: that every option it reports is one this build's backend table
// carries, that the comparison is formed inside a round rather than across
// rounds, and that a run which measured its class ends in exactly one default
// with the way it was reached stated — a measured ordering, a vote over
// re-runs, or a pick among equals the report labels as such — never in a set of
// candidates and never in a name a table was counted for instead of timed. A
// class that holds one option is an answer of its own: one entry is not a
// ranking, and there is no alternative to it. The costs themselves are this
// machine's and asserting them here would pin a number that describes one host.
//
// The protocols below are short on purpose: a probe test that took a minute
// would be a probe nobody runs. Two of them are the shortest protocols there
// are — one pass of one round, with an empty calibration window, so the run
// costs nothing and concludes nothing about cost — and the third is the shortest
// protocol that can order anything, since a quartile band needs four paired
// rounds to be formed at all.

#include "boys/boys_probe.hpp"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

using boys::AccuracyRegion;
using boys::AccuracyTier;
using boys::OptionPrecision;
using boys::OptionProbeDefaultHow;
using boys::OptionProbeMeasurement;
using boys::OptionProbeReport;
using boys::OptionProbeVerdict;
using boys::ProbeOptions;
using boys::QueryTier;

// The shortest run there is: one pass of one round, on a workload small enough
// to be free, with an empty calibration window so the load instrument never finds
// a floor. Nothing about cost is concluded from it — a ratio needs two rounds and
// a band needs four — and that is exactly what the tests about the option book,
// the accuracy column and the refusal paths read.
ProbeOptions OneRound() {
    ProbeOptions options;
    options.count = 256;
    options.nmax = 8;
    options.calibrationSeconds = 0.0;
    options.passes = 1;
    options.rounds = 1;
    return options;
}

// The shortest protocol that can order anything: four paired rounds, which is
// what a lower and upper quartile need, over the same small workload. Every
// option is called once in every round, so the rounds put every option under the
// same clock and the ratios between them are formed inside a round.
ProbeOptions Timed() {
    ProbeOptions options = OneRound();
    options.calibrationSeconds = 0.5;
    options.backgroundWindowSeconds = 0.2;
    options.passes = 4;
    options.rounds = 2;
    return options;
}

// The same protocol with the canary alarm moved. The alarm decides one thing —
// the flag printed beside a pass — and a test about that flag has to be able to
// put it anywhere, including below anything repeated fixed work can reach. The
// probe's own default protocol is untouched.
ProbeOptions TimedWithAlarm(double percent) {
    ProbeOptions options = Timed();
    options.canarySpreadAlarm = percent;
    return options;
}

const OptionProbeMeasurement* Find(const OptionProbeReport& report, const std::string& name) {
    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.name == name) {
            return &measurement;
        }
    }
    return nullptr;
}

// The printed line an option's row occupies, so a test about what a row says
// reads the row and not the prose around it. Empty when the row is not printed.
std::string RowLine(const std::string& text, const std::string& name) {
    const std::string lead = "  " + name + " ";

    for (std::size_t at = text.find('\n'); at != std::string::npos;) {
        const std::size_t begin = at + 1;
        const std::size_t end = text.find('\n', begin);

        if (text.compare(begin, lead.size(), lead) == 0) {
            return text.substr(begin, end - begin);
        }

        at = end;
    }

    return {};
}

// The members of the class the probe takes its default from: the certified
// double lane's precision at the library's own full-accuracy multiplier,
// answering the all-orders question — the shape the workload is asked in and the
// shape the library's own default entry answers. The key is that triple, so a
// faster row of another rung, of another precision or of another shape is a
// different class and not this one.
std::vector<const OptionProbeMeasurement*> ReferenceMembers(const OptionProbeReport& report) {
    std::vector<const OptionProbeMeasurement*> members;

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.measured && measurement.precision == boys::OptionPrecision::kFp64 &&
            measurement.tier == AccuracyTier::kReference &&
            measurement.shape == boys::OptionProbeShape::kAllOrders) {
            members.push_back(&measurement);
        }
    }

    return members;
}

// The fastest of that class: the option every ordering this probe makes for the
// default is measured from, and what the run names when its class ordered
// itself.
const OptionProbeMeasurement* ReferenceLeader(const OptionProbeReport& report) {
    const OptionProbeMeasurement* leader = nullptr;

    for (const OptionProbeMeasurement* measurement : ReferenceMembers(report)) {
        if (leader == nullptr || measurement->nsPerArgument < leader->nsPerArgument) {
            leader = measurement;
        }
    }

    return leader;
}

// The class one precision was ranked in at one rung for one question shape,
// empty when the report carries none. Every class of the report is keyed by that
// triple.
const boys::OptionProbeClass* ClassOf(const OptionProbeReport& report,
                                      boys::OptionPrecision precision, AccuracyTier tier,
                                      boys::OptionProbeShape shape) {
    for (const boys::OptionProbeClass& entry : report.classes) {
        if (entry.precision == precision && entry.tier == tier && entry.shape == shape) {
            return &entry;
        }
    }
    return nullptr;
}

// Whether a name was reported as not offered, rather than measured.
bool Unoffered(const OptionProbeReport& report, const std::string& name) {
    for (const std::string& entry : report.unoffered) {
        if (entry == name) {
            return true;
        }
    }
    return false;
}

// Whether a name was reported as one this build does not carry, rather than
// measured: the seam that declares the entry is closed in this build.
bool NotCarried(const OptionProbeReport& report, const std::string& name) {
    for (const std::string& entry : report.notCarried) {
        if (entry == name) {
            return true;
        }
    }
    return false;
}

TEST(ProbeTest, EveryReportedOptionRunsInArithmeticThisBuildCarries) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    ASSERT_FALSE(report.measurements.empty());

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        bool found = false;
        for (const boys::backend::BackendInfo& info : report.backends) {
            if (measurement.arithmetic == info.name) {
                found = true;
                EXPECT_EQ(measurement.contracts, info.contracts)
                    << measurement.name << " reports a contraction flag the table does not";
            }
        }
        EXPECT_TRUE(found) << measurement.name << " names arithmetic '"
                           << measurement.arithmetic << "', which this build's table lacks";
    }
}

// The option whose arithmetic is the machine's own choice: the packed lane when
// the vector tier is live, the scalar lane otherwise, which is the same
// condition the library dispatches on.
TEST(ProbeTest, TheBatchFp64OptionRunsInTheArithmeticTheLibraryWouldUse) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const OptionProbeMeasurement* batch = Find(report, "batch-fp64");

    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->arithmetic,
              boys::BoysAvx2Available() ? "avx2-fp64" : "scalar-fp64");
}

// The fp32 lane is offered whatever the vector tier does, and it is the fp32
// arithmetic, never the fp64 one under a narrower name.
TEST(ProbeTest, TheFp32OptionRunsInFp32Arithmetic) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const OptionProbeMeasurement* batch = Find(report, "batch-fp32");

    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->arithmetic, boys::BoysAvx2Available() ? "avx2-fp32" : "scalar-fp32");
    EXPECT_NE(batch->arithmetic, boys::backend::ScalarFp64::kName);
}

// The relaxed rungs are the tiers the library reports as served, and each is
// named after the multiplier the library reports for it: a tier that fell back
// to the reference multiplier would be the reference option under another name,
// and the probe does not report it.
TEST(ProbeTest, TheRelaxedRungsAreTheTiersTheLibraryServes) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    for (int raw = 1; raw <= 32; ++raw) {
        const auto tier = static_cast<AccuracyTier>(raw);
        const bool served = QueryTier(tier, AccuracyRegion::kA, 0.0).reachable !=
                            QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable;
        char name[32] = {};
        std::snprintf(name, sizeof(name), "tier-%g-fp64", boys::AccuracyMultiplier(tier));
        const bool reported = (Find(report, name) != nullptr) || Unoffered(report, name);
        EXPECT_EQ(reported, served) << "tier at enum value " << raw
                                    << " (multiplier " << boys::AccuracyMultiplier(tier)
                                    << ") is served=" << served << " but reported=" << reported;
    }
}

// Every option the probe reports is either measured under an arithmetic the
// table carries or named as one this build does not offer — never silently
// dropped, and never reported as something this build is not.
TEST(ProbeTest, NothingTheLibraryOffersGoesUnreported) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    EXPECT_TRUE(Unoffered(report, "batch-fp64") == false);
    EXPECT_TRUE(Find(report, "batch-fp64") != nullptr);
    EXPECT_TRUE(Unoffered(report, "batch-fp32") == false);
    EXPECT_TRUE(Find(report, "batch-fp32") != nullptr);
    EXPECT_TRUE(Find(report, "grouped-fp64") != nullptr || Unoffered(report, "grouped-fp64"));
    EXPECT_TRUE(Find(report, "tagged-fp64") != nullptr || Unoffered(report, "tagged-fp64"));

#if BoysFp16
    EXPECT_TRUE(Find(report, "f16-io") != nullptr || Unoffered(report, "f16-io"));
    EXPECT_TRUE(Find(report, "bf16-io") != nullptr || Unoffered(report, "bf16-io"));
    // This build carries them, so neither is named as one it does not carry.
    EXPECT_FALSE(NotCarried(report, "f16-io"));
    EXPECT_FALSE(NotCarried(report, "bf16-io"));
#else
    // A closed seam must not read as a build that never had the lanes: the two
    // are reported as entries this build does not carry, which is a fact a
    // caller can act on, where their absence would be a fact about nothing.
    EXPECT_TRUE(NotCarried(report, "f16-io"));
    EXPECT_TRUE(NotCarried(report, "bf16-io"));
    EXPECT_TRUE(Find(report, "f16-io") == nullptr);
    EXPECT_TRUE(Find(report, "bf16-io") == nullptr);
#endif
}

// The reference bound the report prints is the library's, not a number written
// in the probe: it is what QueryTier reports for the reference tier.
TEST(ProbeTest, TheReferenceBoundIsReadFromTheLibrary) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    for (const AccuracyRegion region :
         {AccuracyRegion::kA, AccuracyRegion::kB, AccuracyRegion::kC}) {
        EXPECT_LE(QueryTier(AccuracyTier::kReference, region, 0.0).reachable,
                  report.referenceBound);
    }
    EXPECT_DOUBLE_EQ(report.referenceBound,
                     QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable);
}

// The accuracy column is measured whether or not a cost was: the values a lane
// returns for an argument are a property of the build, and a run too short to
// form a ratio must not blank the column. The timed rounds ran all the same —
// the load instrument is what an uncalibrated run does without.
TEST(ProbeTest, TheAccuracyColumnDoesNotDependOnACleanPass) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    ASSERT_FALSE(report.calibrated);
    ASSERT_FALSE(report.passes.empty()) << "the timed rounds are not the load instrument's";
    EXPECT_EQ(report.pairedRounds, 1);

    const OptionProbeMeasurement* batch = Find(report, "batch-fp64");
    ASSERT_NE(batch, nullptr);
    EXPECT_FALSE(batch->measured) << "one round cannot form a ratio, so it cannot be a cost";
    EXPECT_EQ(batch->rounds, 0);
    EXPECT_TRUE(batch->meetsBound);
    EXPECT_LE(batch->maxError, batch->bound);
}

// The column is a measurement, not a constant: the fp64 options sit at the
// certified lane's own resolution, and the fp32 lane's errors are orders above
// them. A column that read zero for every option would pass every bound in this
// file and tell a reader nothing.
TEST(ProbeTest, TheAccuracyColumnSeparatesTheLanes) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const OptionProbeMeasurement* fp64 = Find(report, "batch-fp64");
    const OptionProbeMeasurement* fp32 = Find(report, "batch-fp32");

    ASSERT_NE(fp64, nullptr);
    ASSERT_NE(fp32, nullptr);
    EXPECT_GT(fp32->maxError, 0.0);
    EXPECT_LT(fp64->maxError, fp32->bound);
}

// Every option the probe reports is held to its own documented bound, and the
// fp64 options are held to the certified lane's — which is the contract each of
// them states, whether or not it returns the certified bits. The batch entries
// seed region A at the batch's highest order on a vector host, so their F_k
// below it is a recurrence's value: the difference is inside the bound and is
// reported rather than hidden by comparing a batch against a batch.
TEST(ProbeTest, TheFp64OptionsAreHeldToTheCertifiedBound) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    for (const char* name : {"batch-fp64", "grouped-fp64", "tagged-fp64"}) {
        const OptionProbeMeasurement* measurement = Find(report, name);
        if (measurement == nullptr) {
            continue;
        }
        EXPECT_DOUBLE_EQ(measurement->bound, report.referenceBound) << name;
        EXPECT_TRUE(measurement->meetsBound)
            << name << " delivered " << measurement->maxError << " against a bound of "
            << measurement->bound;
    }
}

// A narrower lane is allowed to differ, and is held to its own published bound.
TEST(ProbeTest, TheNarrowerLanesAreHeldToTheirOwnBound) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    for (const char* name : {"batch-fp32", "f16-io", "bf16-io"}) {
        const OptionProbeMeasurement* measurement = Find(report, name);
        if (measurement == nullptr) {
            continue;
        }
        EXPECT_GT(measurement->bound, report.referenceBound) << name;
        EXPECT_TRUE(measurement->meetsBound)
            << name << " delivered " << measurement->maxError << " against a bound of "
            << measurement->bound;
    }
}

// A relaxed rung is documented looser than the certified lane, and it is a
// class of its own at that: the class key is one precision, one rung AND one
// question shape, so a relaxed row is never a member of the class the default is
// taken from, however the two figures compare. The figures are still reported — that is what lets a
// reader see a faster option was faster at a lower accuracy rather than at the
// same one.
TEST(ProbeTest, ARelaxedRungIsNeverInTheCertifiedAccuracyClass) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const boys::OptionProbeClass* reference =
        ClassOf(report, boys::OptionPrecision::kFp64, AccuracyTier::kReference,
                                     boys::OptionProbeShape::kAllOrders);

    ASSERT_NE(reference, nullptr) << "the certified lane's reference class is not in the report";

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.name.rfind("tier-", 0) != 0) {
            continue;
        }

        EXPECT_GT(measurement.bound, report.referenceBound) << measurement.name;
        EXPECT_NE(measurement.tier, AccuracyTier::kReference) << measurement.name;
        EXPECT_EQ(std::find(reference->ranked.begin(), reference->ranked.end(), measurement.name),
                  reference->ranked.end())
            << measurement.name << " is ranked in the reference class and was built at another rung";
    }
}

// An instrument with no floor does not stop the measurement. The load readings
// are context and the comparison is not made in them, so the run still times its
// rounds and still reports them; what it loses is the right to print a load
// percentage, and the report says so rather than printing a figure relative to
// nothing. With one round there is no ratio to form, so this run also ends in the
// refusal — the one that says how many rounds it would need.
TEST(ProbeTest, AnUncalibratedInstrumentStillMeasuresAndSaysWhatItLost) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    EXPECT_FALSE(report.calibrated);
    EXPECT_FALSE(report.passes.empty()) << "the timed rounds do not depend on the load instrument";
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_EQ(report.defaultHow, OptionProbeDefaultHow::kNone)
        << "a run that named no default labelled the way it reached one";
    EXPECT_FALSE(report.hasDefault);
    EXPECT_TRUE(report.fastestAtReferenceAccuracy.empty());
    EXPECT_TRUE(report.inseparable.empty());
    EXPECT_TRUE(report.refinements.empty())
        << "a refinement stage ran over a set nothing was measured in";
    EXPECT_TRUE(report.reason.empty() == false);

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        EXPECT_FALSE(measurement.measured)
            << measurement.name << " was given a cost from a single round";
    }

    // The refusal that names the round count, rather than one that blames a
    // machine it never read.
    EXPECT_NE(report.reason.find("paired round"), std::string::npos) << report.reason;
    EXPECT_NE(report.reason.find("nothing to be relative to"), std::string::npos) << report.reason;

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("CANNOT DETERMINE"), std::string::npos);
    EXPECT_NE(text.find("never established a floor"), std::string::npos);
    EXPECT_NE(text.find("nothing to be relative to"), std::string::npos);
}

// The canary alarm is a flag and not a gate: with the alarm below anything
// repeated fixed work can reach, every pass is flagged — and every pass is still
// used, so the run still produces the same kind of figures it produces with the
// alarm out of the way. Before this change the same run produced no figure at all
// and ended in CANNOT DETERMINE, which is what a fixed work measured by wall
// clock does to its own admission rule on a machine whose clock moves.
TEST(ProbeTest, APassAboveTheCanaryAlarmIsFlaggedAndStillUsed) {
    const OptionProbeReport flagged = boys::RunOptionProbe(TimedWithAlarm(1e-9));

    if (!flagged.calibrated) {
        GTEST_SKIP() << "the load instrument found no floor on this machine, so there is no alarm "
                        "to flag anything against";
    }

    ASSERT_FALSE(flagged.passes.empty());
    EXPECT_EQ(flagged.passesWithinAlarm, 0);
    EXPECT_EQ(flagged.passesAboveAlarm, static_cast<int>(flagged.passes.size()));

    for (const boys::OptionProbePass& pass : flagged.passes) {
        EXPECT_TRUE(pass.canaryWide) << "an alarm of 1e-9 left a pass unflagged";
    }

    // Used, not discarded: the rounds ran, the figures were formed, and the
    // flagged passes are in them.
    EXPECT_EQ(flagged.pairedRounds, flagged.options.passes * flagged.options.rounds);
    EXPECT_FALSE(flagged.reason.empty());

    std::size_t measured = 0;

    for (const OptionProbeMeasurement& measurement : flagged.measurements) {
        if (measurement.measured) {
            ++measured;
            EXPECT_GT(measurement.nsPerArgument, 0.0) << measurement.name;
            EXPECT_EQ(measurement.rounds, flagged.pairedRounds) << measurement.name;
        }
    }

    EXPECT_GT(measured, 0u) << "every flagged pass was discarded after all";

    const std::string text = boys::FormatOptionProbe(flagged);
    EXPECT_NE(text.find("reported, used"), std::string::npos) << "the pass table does not say so";
    EXPECT_NE(text.find("gates nothing"), std::string::npos);

    // The other end of the same knob changes only the flags: with the alarm out
    // of reach nothing is flagged, and the run is the same run. A pass is placed
    // on one side of the alarm or the other against the floor the calibration
    // found, and the second run finds its own floor: on a machine that gave the
    // first run one and this one none, the passes carry no reading and none is
    // flagged, which is what the counts say rather than a quiet pass reported as
    // a measurement.
    const OptionProbeReport quiet = boys::RunOptionProbe(TimedWithAlarm(1e9));

    // The alarm changes the flags and nothing else, so the run behind them is the
    // same run however the alarm was read.
    EXPECT_EQ(quiet.pairedRounds, flagged.pairedRounds);

    if (!quiet.calibrated) {
        EXPECT_EQ(quiet.passesWithinAlarm, 0)
            << "an instrument with no floor put a pass on the quiet side of an alarm";
        EXPECT_EQ(quiet.passesAboveAlarm, 0)
            << "an instrument with no floor put a pass on the wide side of an alarm";
        GTEST_SKIP() << "the load instrument found no floor for this run, so its passes carry no "
                        "alarm reading to be flagged against";
    }

    EXPECT_EQ(quiet.passesWithinAlarm, static_cast<int>(quiet.passes.size()));
    EXPECT_EQ(quiet.passesAboveAlarm, 0);
}

// The resolution is measured in the quantities the ordering is made of, and it
// is not a bar chosen in advance: it is the widest within-round band the
// certified double lane's classes showed — a rival's band against its own
// class's leader — so it is the coarseness of the very comparisons the report
// placed or refused to place. A run that formed no band reports zero rather
// than a width it never measured. The figures move with the machine; this
// relation does not.
TEST(ProbeTest, TheResolutionIsTheWidestBandTheClassShowed) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.pairedRounds < 4 || ReferenceMembers(report).size() < 2) {
        EXPECT_DOUBLE_EQ(report.resolution, 0.0)
            << "a width was reported from a run in which no within-round band was formed";
        return;
    }

    EXPECT_GT(report.resolution, 0.0)
        << "the class holds two measured options over four rounds, and yet no band was formed";
}

// A class the run ordered names its own leader and leaves no rival unplaced;
// a class the run could not order still ends in one default, and the options it
// could not place are named beside it, so a reader is never handed a name
// without the evidence that is missing for it. The class is one precision, one
// rung and one question shape, so a faster row of another precision, another
// rung or another shape is never a rival of it.
TEST(ProbeTest, ARecommendationLeavesNoRivalUnplaced) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const OptionProbeMeasurement* leader = ReferenceLeader(report);
    const boys::OptionProbeClass* doubles =
        ClassOf(report, OptionPrecision::kFp64, AccuracyTier::kReference,
                                 boys::OptionProbeShape::kAllOrders);

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_TRUE(report.hasDefault);

        ASSERT_NE(doubles, nullptr);
        ASSERT_NE(leader, nullptr);
        EXPECT_EQ(doubles->leader, leader->name);

        // The invariant the whole report rests on: the default is the row the
        // report's own figures put first in the class the rule names. Whether the
        // class ordered, the refinement runs agreed or they named another row, the
        // name printed as the default is the name the class block prints first -
        // a report whose two halves disagree about which option is cheapest is a
        // report the reader has to correct, and no vote may produce one.
        EXPECT_EQ(report.recommended, leader->name)
            << "the default is not the cheapest row of its own class by the figures printed "
               "beside it";

        if (report.defaultHow == OptionProbeDefaultHow::kOrdered) {
            EXPECT_TRUE(report.inseparable.empty()) << "an ordering left a rival unplaced";
            EXPECT_EQ(report.recommended, leader->name);
            EXPECT_TRUE(doubles->ordered);

            for (const OptionProbeMeasurement* member : ReferenceMembers(report)) {
                if (member->name != leader->name) {
                    EXPECT_LT(leader->nsPerArgument, member->nsPerArgument) << member->name;
                }
            }

            return;
        }

        // Reached by the refinement's vote, or by a tie this run could not break:
        // the name is one of the class's own options, the pair the class could not
        // place is reported, and the runs taken over the tied set are in the report
        // rather than only their outcome.
        bool member = false;

        for (const OptionProbeMeasurement* candidate : ReferenceMembers(report)) {
            member = member || candidate->name == report.recommended;
        }

        EXPECT_TRUE(member) << report.recommended << " is the default and is not of the class";
        EXPECT_FALSE(report.inseparable.empty())
            << "a tie was decided without naming the pair the class could not place";
        EXPECT_FALSE(report.refinements.empty())
            << "a tie was decided with no refinement run behind the choice";

        // From here the class is tied, and the way the report says the tie was
        // reached has to match the vote's own record: the vote naming this row is
        // what kRefined and kVote mean, and a vote that named another row is a
        // kChosenAmongEquals whose reason prints both figures.
        const boys::OptionProbeRefinement& stage = report.refinements.front();
        const bool voteNamesThisRow = stage.winner == report.recommended;

        if (report.defaultHow == OptionProbeDefaultHow::kChosenAmongEquals && stage.ran &&
            !stage.winner.empty() && !voteNamesThisRow) {
            EXPECT_NE(report.reason.find(stage.winner), std::string::npos)
                << "the vote named another row and the reason does not say which";
            EXPECT_NE(report.reason.find("cannot be separated"), std::string::npos)
                << "a tie the vote did not confirm is not reported as one";

            const std::string text = boys::FormatOptionProbe(report);
            EXPECT_NE(text.find("could not be separated"), std::string::npos)
                << "the printed report does not say the top entries could not be separated";
        } else if (voteNamesThisRow && stage.ran) {
            EXPECT_EQ(report.defaultHow,
                      stage.unanimous
                          ? OptionProbeDefaultHow::kRefined
                          : (stage.plurality ? OptionProbeDefaultHow::kVote
                                             : OptionProbeDefaultHow::kChosenAmongEquals))
                << "the vote named this row and the report does not say so with the right answer";
        }

        return;
    }

    EXPECT_TRUE(report.recommended.empty());
    EXPECT_EQ(report.defaultHow, OptionProbeDefaultHow::kNone);
    EXPECT_FALSE(report.reason.empty());
}

// The output states the resolution in the reader's own terms and in the units
// the comparison is made in, so a refusal is a measurement with a number
// attached rather than a shrug. When the run was too short for a band, the text
// says that instead of printing a width it never measured.
TEST(ProbeTest, TheTextStatesTheResolution) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("resolution:"), std::string::npos);
    EXPECT_NE(text.find("gates nothing"), std::string::npos);
    EXPECT_NE(text.find("this machine"), std::string::npos);
    EXPECT_NE(text.find("paired ratios"), std::string::npos);

    const bool banded = report.pairedRounds >= 4 && ReferenceLeader(report) != nullptr;

    if (banded) {
        EXPECT_NE(text.find("widest within-round band"), std::string::npos);
    } else {
        EXPECT_NE(text.find("not measurable on this run"), std::string::npos);
    }
}

// The clock check, made rather than assumed, and made from the run's own rows.
// Wider vector registers draw a lower clock, so two options that do not run one
// arithmetic can be exposed to the machine differently; the report says which
// case it measured instead of implying that an ordering holds at any clock. It
// states how far the widest-moving pair's ratio travelled between the run's
// halves beside the resolution that figure is read against, and whether every
// option the comparison put against another ran the same arithmetic route.
TEST(ProbeTest, TheClockCheckIsReadFromTheRunsOwnRows) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    std::size_t inClass = 0;
    std::string route;
    bool oneRoute = true;

    for (const OptionProbeMeasurement* measurement : ReferenceMembers(report)) {
        if (inClass == 0) {
            route = measurement->arithmetic;
        } else if (measurement->arithmetic != route) {
            oneRoute = false;
        }

        ++inClass;
    }

    if (report.pairedRounds < 4 || inClass < 2) {
        EXPECT_EQ(report.confidence.find("No pair of the class moved"), std::string::npos)
            << report.confidence;
        return;
    }

    const bool heldStill = report.confidence.find("No pair of the class moved") != std::string::npos;
    const bool warned = report.confidence.find("WARNING: the pair") != std::string::npos;

    // Exactly one of the two, and both name the pair and the resolution the
    // figure was read against: a pair that came in under it, or one that went
    // past it with the ordering's own exposure named.
    EXPECT_NE(heldStill, warned) << report.confidence;
    EXPECT_NE(report.confidence.find("this run can order"), std::string::npos)
        << report.confidence;

    if (heldStill) {
        EXPECT_NE(report.confidence.find("widest was"), std::string::npos) << report.confidence;
    } else {
        EXPECT_NE(report.confidence.find("are not equally exposed"), std::string::npos)
            << report.confidence;
    }

    const std::string expected =
        oneRoute ? "runs the same arithmetic" : "does not run one arithmetic";
    EXPECT_NE(report.confidence.find(expected), std::string::npos)
        << (oneRoute ? route : std::string("the class mixes routes")) << ": "
        << report.confidence;
    EXPECT_NE(text.find(expected), std::string::npos);

    if (oneRoute) {
        EXPECT_NE(report.confidence.find(route), std::string::npos) << report.confidence;
    }
}

// A band is a lower and an upper quartile, and two rounds have neither: no pair
// of the class is placed, and the report says so rather than printing a width it
// never measured. The run still ends in one default — the class's own fastest,
// reached by the refinement runs, which are taken at a longer protocol than the
// run that could not order — and it says which of the two this is.
TEST(ProbeTest, AnOrderingNeedsFourPairedRounds) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 2);
    EXPECT_FALSE(report.measurements.empty());
    ASSERT_TRUE(report.measurements.front().measured)
        << "two rounds can form a ratio, just not a quartile band";

    EXPECT_DOUBLE_EQ(report.resolution, 0.0);
    EXPECT_NE(report.reason.find("quartile"), std::string::npos) << report.reason;

    const boys::OptionProbeClass* doubles =
        ClassOf(report, OptionPrecision::kFp64, AccuracyTier::kReference,
                                 boys::OptionProbeShape::kAllOrders);
    ASSERT_NE(doubles, nullptr);
    ASSERT_FALSE(doubles->leader.empty());
    EXPECT_FALSE(doubles->ordered) << "a class was ordered on too few rounds for a band";
    EXPECT_NE(doubles->note.find("quartile"), std::string::npos) << doubles->note;

    // One default, and it is a member of that class: the options the class could
    // not place were re-run alone at a longer protocol and voted on, and the
    // report carries the runs, their leaders and the vote.
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kRecommend);
    EXPECT_FALSE(report.recommended.empty()) << "a measured class was left without a default";
    EXPECT_TRUE(report.hasDefault);
    EXPECT_NE(report.defaultHow, OptionProbeDefaultHow::kOrdered)
        << "a two-round run reported a measured ordering";
    EXPECT_EQ(doubles->how, report.defaultHow);

    ASSERT_FALSE(report.refinements.empty()) << "the tie was decided with no refinement behind it";
    const boys::OptionProbeRefinement& stage = report.refinements.front();

    EXPECT_EQ(stage.precision, OptionPrecision::kFp64);

    // The default is the class's own fastest row by the printed figure. The vote
    // is the same question asked again at a longer protocol: it confirms that row
    // or names another, and it may not name a row the figures beside it put behind.
    const OptionProbeMeasurement* classLeader = ReferenceLeader(report);
    ASSERT_NE(classLeader, nullptr);
    EXPECT_EQ(report.recommended, classLeader->name)
        << "the default is not the class's own fastest row by the printed figure";
    EXPECT_EQ(report.defaultHow,
              stage.winner == report.recommended
                  ? (stage.unanimous
                         ? OptionProbeDefaultHow::kRefined
                         : (stage.plurality ? OptionProbeDefaultHow::kVote
                                            : OptionProbeDefaultHow::kChosenAmongEquals))
                  : OptionProbeDefaultHow::kChosenAmongEquals);

    EXPECT_EQ(stage.runs, report.options.refinementRuns);
    EXPECT_EQ(stage.passes, report.options.passes * report.options.refinementFactor)
        << "the re-run was not at a larger protocol than the run that could not order";
    EXPECT_EQ(stage.rounds, report.options.rounds * report.options.refinementFactor);
    EXPECT_EQ(stage.runLeaders.size(), static_cast<std::size_t>(stage.runs));
    EXPECT_GT(stage.rounds, report.pairedRounds);
    EXPECT_FALSE(stage.pool.empty());
    EXPECT_NE(stage.pool.end(), std::find(stage.pool.begin(), stage.pool.end(), stage.winner))
        << "the refinement named a winner it did not re-measure";

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("not measurable on this run"), std::string::npos);
    EXPECT_NE(text.find("2 paired round(s)"), std::string::npos);
    EXPECT_NE(text.find("the refinement stage"), std::string::npos);
    EXPECT_NE(text.find("leader of each run"), std::string::npos);
    EXPECT_NE(text.find("default: " + report.recommended), std::string::npos);
}

// The comparison is paired, and this is what that means in the report: every
// option was called in every round the run took — they all rest on the same
// round count, and that count is every pass's every round — and the reference
// lane, being the option every ratio is formed against, is exactly one against
// itself in every round and so has no drift at all. A design that timed one
// option in one set of rounds and another in another would be reporting a ratio
// of two figures taken under two clocks, which is the defect this pins.
TEST(ProbeTest, ThePairsAreFormedInsideTheRounds) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    ASSERT_FALSE(report.referenceOption.empty());
    ASSERT_FALSE(report.passes.empty());
    EXPECT_EQ(report.referenceOption, "batch-fp64")
        << "the anchor is the option book's choice, not the winner of the comparison it anchors";

    const OptionProbeMeasurement* reference = Find(report, report.referenceOption);
    ASSERT_NE(reference, nullptr);
    ASSERT_TRUE(reference->measured);

    EXPECT_DOUBLE_EQ(reference->ratioToReference, 1.0);
    EXPECT_DOUBLE_EQ(reference->ratioLo, 1.0);
    EXPECT_DOUBLE_EQ(reference->ratioHi, 1.0);
    EXPECT_DOUBLE_EQ(reference->ratioDrift, 0.0);
    EXPECT_DOUBLE_EQ(reference->spread, 1.0);
    EXPECT_DOUBLE_EQ(reference->nsPerArgument, report.referenceNsPerArgument);

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured) {
            continue;
        }

        EXPECT_EQ(measurement.rounds, report.pairedRounds) << measurement.name;
        EXPECT_EQ(measurement.rounds, report.options.passes * report.options.rounds)
            << measurement.name << " was timed in a subset of the run's rounds";
    }
}

// A run that formed no figure at all ends in the refusal and in nothing else:
// no name is printed, no candidate set is left for the caller to choose from,
// and no option is offered as a default on the strength of a table rather than a
// measurement. The report gives the reason instead — the rounds a band needs —
// so a consumer who cannot afford a longer run is told what the longer run buys
// rather than handed a name the run never measured. One pass of one round forms
// no ratio at all.
TEST(ProbeTest, ARunThatMeasuredNothingLeavesNoDefault) {
    ProbeOptions options = OneRound();
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 1);
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_EQ(report.defaultHow, OptionProbeDefaultHow::kNone);
    EXPECT_FALSE(report.hasDefault);
    EXPECT_TRUE(report.inseparable.empty());
    EXPECT_TRUE(report.refinements.empty()) << "a vote was taken over a set nothing was measured in";
    EXPECT_NE(report.reason.find("no option produced a figure"), std::string::npos) << report.reason;

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("verdict: CANNOT DETERMINE"), std::string::npos);
    EXPECT_EQ(text.find("  default: "), std::string::npos)
        << "a run that measured nothing printed a default";

    // And every row says so: an option that carries no cost is printed as one,
    // so nothing in the table can be taken for an answer.
    for (const OptionProbeMeasurement& measurement : report.measurements) {
        EXPECT_FALSE(measurement.measured) << measurement.name;
        const std::string row = RowLine(text, measurement.name);
        ASSERT_FALSE(row.empty()) << measurement.name << " is not in the printed table";
        EXPECT_NE(row.find("not measured"), std::string::npos) << row;
    }
}

// Where a class cannot be ordered, the owner's rule is: re-run the options it
// left tied, alone, at a longer protocol; take the one that led most of those
// runs; and where no candidate leads most of them, take one of the leaders and
// say plainly that this is what happened. The protocol here is two rounds, which
// can form no band at all, so the tie is certain and the vote is what decides —
// and the run leaders, the counts and the winner are checked against each other.
TEST(ProbeTest, ATiedClassIsReRunAloneAndVotedOn) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    options.refinementRuns = 6;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_FALSE(report.refinements.empty()) << "a tied class was decided with no runs behind it";

    const boys::OptionProbeRefinement& stage = report.refinements.front();

    EXPECT_EQ(stage.precision, OptionPrecision::kFp64);
    EXPECT_TRUE(stage.ran);
    EXPECT_EQ(stage.runs, 6);
    EXPECT_EQ(stage.runLeaders.size(), 6u) << "a run of the vote placed no leader";
    EXPECT_FALSE(stage.winner.empty()) << "the vote ended with no entry to name";

    // The default is the class's own fastest row by the figure the report prints,
    // and the vote is the longer protocol's reading of the same question: it either
    // names that row — and then the report says the vote is what carried it — or it
    // names another, and then the report prints both figures and says the two
    // cannot be separated. What it never does is hand back a row the figures
    // beside it put behind.
    const OptionProbeMeasurement* classLeader = ReferenceLeader(report);
    ASSERT_NE(classLeader, nullptr);
    EXPECT_EQ(report.recommended, classLeader->name);
    EXPECT_EQ(report.defaultHow,
              stage.winner == report.recommended
                  ? (stage.unanimous
                         ? OptionProbeDefaultHow::kRefined
                         : (stage.plurality ? OptionProbeDefaultHow::kVote
                                            : OptionProbeDefaultHow::kChosenAmongEquals))
                  : OptionProbeDefaultHow::kChosenAmongEquals)
        << "the way-it-was-reached does not match what the vote named";

    EXPECT_GT(stage.rounds, report.pairedRounds)
        << "the re-run was not at a longer protocol than the run that could not order";
    EXPECT_GT(stage.passes, report.options.passes);

    // Every leader is one of the options that were re-measured, and the winner
    // is one of the leaders: the vote counts the runs it ran.
    for (const std::string& leader : stage.runLeaders) {
        EXPECT_NE(std::find(stage.pool.begin(), stage.pool.end(), leader), stage.pool.end())
            << leader << " led a refinement run without having been re-measured";
    }

    EXPECT_NE(std::find(stage.runLeaders.begin(), stage.runLeaders.end(), stage.winner),
              stage.runLeaders.end())
        << stage.winner << " is the vote's winner and led no run at all";

    if (stage.unanimous) {
        for (const std::string& leader : stage.runLeaders) {
            EXPECT_EQ(leader, stage.winner) << "a unanimous vote has more than one leader";
        }
    }

    EXPECT_FALSE(stage.tally.empty()) << "the vote is reported with no counts";
    EXPECT_FALSE(stage.note.empty()) << "the vote does not say how it came out";

    // And whatever the vote did, the report says what carried it: the run is
    // never labelled an ordering it did not measure.
    EXPECT_NE(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.hasDefault);
    EXPECT_NE(report.defaultHow, OptionProbeDefaultHow::kNone);
    EXPECT_NE(report.defaultHow, OptionProbeDefaultHow::kOrdered)
        << "a two-round run reported a measured ordering";

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("the refinement stage"), std::string::npos);
    EXPECT_NE(text.find("vote: " + stage.winner), std::string::npos);
    EXPECT_NE(text.find("default: " + report.recommended), std::string::npos);
    EXPECT_NE(text.find("reached by:"), std::string::npos);

    // Whichever of the two rows the vote preferred, the printed report says which
    // it was: a tie the vote did not confirm is printed as one, with both figures,
    // rather than as an ordering the run did not measure.
    if (stage.winner != report.recommended) {
        EXPECT_NE(report.reason.find(stage.winner), std::string::npos) << report.reason;
        EXPECT_NE(report.reason.find("cannot be separated"), std::string::npos) << report.reason;
        EXPECT_NE(report.confidence.find(stage.winner), std::string::npos) << report.confidence;
        EXPECT_NE(text.find("could not be separated"), std::string::npos);
    }
}

// The owner's acceptance rule, stated where it can be checked: a run that
// measured its class ends with exactly one combination, and the class entry, the
// name and the reason agree on it. Both protocols that measure are read, since
// the rule is about the report's shape and not about how well the machine
// separated the options.
TEST(ProbeTest, EveryRunThatMeasuredEndsInExactlyOneDefault) {
    for (const bool shortRun : {false, true}) {
        ProbeOptions options = Timed();

        if (shortRun) {
            options.passes = 1;
            options.rounds = 2;
        }

        const OptionProbeReport report = boys::RunOptionProbe(options);

        EXPECT_EQ(report.verdict, OptionProbeVerdict::kRecommend) << report.reason;
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_TRUE(report.hasDefault);
        EXPECT_NE(report.defaultHow, OptionProbeDefaultHow::kNone);
        EXPECT_NE(report.reason.find(report.recommended), std::string::npos) << report.reason;

        const boys::OptionProbeClass* doubles =
            ClassOf(report, OptionPrecision::kFp64, AccuracyTier::kReference,
                                 boys::OptionProbeShape::kAllOrders);
        ASSERT_NE(doubles, nullptr);
        EXPECT_EQ(doubles->how, report.defaultHow)
            << "the class the default is taken from reports a different way of reaching one";
        EXPECT_FALSE(doubles->leader.empty());

        // The name is printed with the report and, where it is not the class's
        // own fastest row, with the class — which says that its entry is not its
        // leader rather than leaving the two to be told apart by reading.
        const std::string text = boys::FormatOptionProbe(report);
        EXPECT_NE(text.find("default: " + report.recommended), std::string::npos);

        if (report.defaultHow == OptionProbeDefaultHow::kOrdered) {
            EXPECT_EQ(report.recommended, doubles->leader);
            EXPECT_TRUE(report.inseparable.empty());
        } else if (!report.refinements.empty() &&
                   report.refinements.front().winner != report.recommended) {
            // The vote named another row: the class the default is taken from says
            // so, names the row, and says the two cannot be separated — the class
            // block is where a reader looks for which row it names and why.
            EXPECT_NE(doubles->note.find(report.refinements.front().winner), std::string::npos)
                << "the class the default is taken from does not name the row the vote "
                   "preferred: "
                << doubles->note;
            EXPECT_NE(doubles->note.find("cannot be separated"), std::string::npos)
                << doubles->note;
        }
    }
}

// Whatever the machine did, the protocol's bookkeeping adds up: every pass was
// either within the canary's alarm or above it and all of them were used, every
// figure rests on every paired round, and a figure that exists is a positive
// cost with a band around it. A run whose load instrument never found a floor
// took no canary and places no pass on either side of the alarm, and its
// confidence line says so rather than naming a load of zero.
TEST(ProbeTest, ThePassBookkeepingAddsUp) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.calibrated) {
        EXPECT_EQ(report.passesWithinAlarm + report.passesAboveAlarm,
                  static_cast<int>(report.passes.size()));
    } else {
        EXPECT_EQ(report.passesWithinAlarm, 0);
        EXPECT_EQ(report.passesAboveAlarm, 0);

        // Where the fact is printed depends on the verdict, not on whether it
        // happened: a refusal carries it in its reason, a recommendation in its
        // confidence line. Either way a reader is told no canary ran.
        EXPECT_NE((report.confidence + report.reason).find("never established a floor"),
                  std::string::npos)
            << report.confidence;
        EXPECT_EQ(report.confidence.find("median load"), std::string::npos) << report.confidence;
        EXPECT_EQ(report.confidence.find("canary's own runs wider"), std::string::npos)
            << report.confidence;
    }

    EXPECT_EQ(report.pairedRounds, report.options.passes * report.options.rounds);

    double widestCanary = 0.0;

    for (const boys::OptionProbePass& pass : report.passes) {
        EXPECT_GT(pass.seconds, 0.0);
        EXPECT_EQ(pass.canaryWide, pass.canarySpread > report.options.canarySpreadAlarm);
        widestCanary = std::max(widestCanary, pass.canarySpread);
    }

    EXPECT_DOUBLE_EQ(report.canarySpread, widestCanary)
        << "the run's canary reading is the widest pass it took, and no pass is left out of it";

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.measured) {
            EXPECT_GT(measurement.nsPerArgument, 0.0) << measurement.name;
            EXPECT_GE(measurement.nsPerArgumentMax, measurement.nsPerArgument) << measurement.name;
            EXPECT_GE(measurement.spread, 1.0) << measurement.name;
            EXPECT_DOUBLE_EQ(measurement.spread, measurement.ratioHi / measurement.ratioLo)
                << measurement.name;
            EXPECT_NE(measurement.checkedSum, 0.0) << measurement.name;
        } else {
            EXPECT_EQ(measurement.rounds, 0) << measurement.name;
            EXPECT_DOUBLE_EQ(measurement.nsPerArgument, 0.0) << measurement.name;
        }
    }

    if (report.passes.empty()) {
        EXPECT_FALSE(report.calibrated);
        EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    } else {
        EXPECT_FALSE(report.reason.empty());
        EXPECT_FALSE(report.confidence.empty());
    }
}

// The verdict and the names it is built from always agree with each other: the
// default is an option of the certified lane's precision at the library's own
// multiplier, answering the workload's own question — never a faster row of
// another precision, another rung or another shape — and a
// class the run ordered hands over its own leader while a class it could not
// order hands over one of the options it left tied, with the pair it could not
// place named beside the answer. The narrowest reading names the fastest option
// of that class, and a refusal names nothing.
TEST(ProbeTest, TheVerdictAndTheNamesAgree) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_TRUE(report.hasDefault);

        const OptionProbeMeasurement* named = Find(report, report.recommended);
        ASSERT_NE(named, nullptr);
        EXPECT_TRUE(named->measured);
        EXPECT_GT(named->nsPerArgument, 0.0);
        EXPECT_EQ(named->precision, OptionPrecision::kFp64)
            << "the recommendation is of the certified lane's precision, never of another";
        EXPECT_EQ(named->tier, AccuracyTier::kReference)
            << "the recommendation was built at another multiplier than the library's own";

        if (report.defaultHow == OptionProbeDefaultHow::kOrdered) {
            EXPECT_TRUE(report.inseparable.empty());
            EXPECT_EQ(named, ReferenceLeader(report));
        } else {
            EXPECT_FALSE(report.inseparable.empty())
                << "the class was not ordered, and nothing is named as unplaced beside the default";
        }

        // The class's own fastest, and the narrowest reading of it: the fastest
        // row of the class, which is never held to a looser figure than the lane
        // whose options it names.
        if (!report.fastestAtReferenceAccuracy.empty()) {
            const OptionProbeMeasurement* fastestAtRung =
                Find(report, report.fastestAtReferenceAccuracy);
            ASSERT_NE(fastestAtRung, nullptr);
            EXPECT_TRUE(fastestAtRung->measured);
            EXPECT_EQ(fastestAtRung->tier, AccuracyTier::kReference);
            EXPECT_LE(fastestAtRung->bound, report.referenceBound);
            EXPECT_GE(fastestAtRung->nsPerArgument, ReferenceLeader(report)->nsPerArgument);
        }
    } else {
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_FALSE(report.reason.empty());
        EXPECT_EQ(report.confidence.rfind("CANNOT DETERMINE", 0), 0u) << report.confidence;
    }
}

// A class is one precision, one rung and one question shape, and that triple is
// the whole key: nothing inside a class was built at another precision or
// another multiplier and nothing inside it answers another question, the leader
// of a class is its fastest member, a class that says it is ordered holds no
// unplaced member and no member of one entry — one entry is not a ranking — and
// every figure the run produced is ranked in the class of its own triple and in
// no other. A class is built from the rounds the run took, so an uncalibrated
// instrument still produces them; what a short run cannot produce is a band, and
// then nothing in a class of more than one is ordered.
TEST(ProbeTest, EveryClassIsOnePrecisionOneRungAndOneShapeAndRanksOnlyItsOwn) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    ASSERT_FALSE(report.classes.empty());

    std::set<std::tuple<int, int, int>> keys;

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_FALSE(entry.name.empty()) << "a class with no name is one a reader cannot place";
        EXPECT_TRUE(keys
                        .insert({static_cast<int>(entry.precision), static_cast<int>(entry.tier),
                                 static_cast<int>(entry.shape)})
                        .second)
            << entry.name << " shares its precision, its rung and its shape with another class";
        EXPECT_NE(entry.name.find(boys::OptionProbeShapeName(entry.shape)), std::string::npos)
            << entry.name << " does not carry the question shape it is keyed on";

        std::size_t measured = 0;

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.tier == entry.tier && measurement.shape == entry.shape) {
                ++measured;
            }
        }

        EXPECT_EQ(entry.ranked.size(), measured) << entry.name << " does not rank every member";

        if (entry.leader.empty()) {
            EXPECT_TRUE(entry.ranked.empty()) << entry.name;
            EXPECT_FALSE(entry.ordered) << entry.name << " has no leader to be ordered around";
            EXPECT_FALSE(entry.note.empty()) << entry.name;
            EXPECT_EQ(measured, 0u) << entry.name << " reports no figure yet options were "
                                                         "measured in its precision, rung and shape";
            continue;
        }

        ASSERT_FALSE(entry.ranked.empty()) << entry.name;

        const OptionProbeMeasurement* leader = Find(report, entry.leader);
        ASSERT_NE(leader, nullptr) << entry.name;
        EXPECT_TRUE(leader->measured) << entry.name << " names a leader that was never measured";
        EXPECT_EQ(leader->precision, entry.precision) << entry.name;
        EXPECT_EQ(leader->tier, entry.tier) << entry.name << " names a leader of another rung";
        EXPECT_EQ(leader->shape, entry.shape)
            << entry.name << " names a leader that answers another question";
        EXPECT_DOUBLE_EQ(entry.leaderNsPerArgument, leader->nsPerArgument) << entry.name;
        EXPECT_EQ(entry.ranked.front(), entry.leader)
            << entry.name << " ranks its leader somewhere behind its own first row";

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.tier == entry.tier && measurement.shape == entry.shape) {
                EXPECT_GE(measurement.nsPerArgument, leader->nsPerArgument)
                    << entry.name << " names " << entry.leader << " as its leader with "
                    << measurement.name << " measured faster in it";
            }
        }

        // One entry is not a ranking: a class of one is not ordered and names
        // its entry by there being no alternative.
        if (entry.ranked.size() == 1) {
            EXPECT_FALSE(entry.ordered) << entry.name << " calls a class of one an ordering";
            EXPECT_EQ(entry.how, OptionProbeDefaultHow::kOnlyEntry) << entry.name;
            EXPECT_NE(entry.note.find("not a ranking"), std::string::npos) << entry.note;
        } else {
            EXPECT_NE(entry.how, OptionProbeDefaultHow::kOnlyEntry)
                << entry.name << " names an entry of a class it measured against others";
        }

        // A band over the lower and upper quartiles of the paired ratios needs
        // four rounds; below that a class is measured and not ordered, and the
        // note says which of the two it is.
        if (report.pairedRounds < 4) {
            EXPECT_FALSE(entry.ordered) << entry.name
                                        << " says it is ordered on too few rounds for a band";
        }

        for (const std::string& name : entry.ranked) {
            const OptionProbeMeasurement* member = Find(report, name);
            ASSERT_NE(member, nullptr) << entry.name;
            EXPECT_EQ(member->precision, entry.precision)
                << name << " is ranked in " << entry.name << " but is another precision";
            EXPECT_EQ(member->tier, entry.tier)
                << name << " is ranked in " << entry.name << " but was built at another rung";
            EXPECT_EQ(member->shape, entry.shape)
                << name << " is ranked in " << entry.name << " but answers another question";

            if (entry.ordered && name != entry.leader) {
                EXPECT_GT(member->nsPerArgument, leader->nsPerArgument)
                    << entry.name << " says it is ordered, yet " << name
                    << " is not behind its leader in the run's own statistic";
            }
        }
    }

    // Every figure the run produced is ranked in the class of its own precision
    // and rung, and in no other class: an option measured under a key the report
    // does not class would be a figure a reader could not place.
    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured) {
            continue;
        }

        const boys::OptionProbeClass* entry =
            ClassOf(report, measurement.precision, measurement.tier, measurement.shape);
        ASSERT_NE(entry, nullptr) << measurement.name
                                  << " was measured in a precision, rung and shape the report "
                                     "classes nowhere";
        EXPECT_NE(std::find(entry->ranked.begin(), entry->ranked.end(), measurement.name),
                  entry->ranked.end())
            << measurement.name << " is measured in " << entry->name << " and is not ranked in it";

        for (const boys::OptionProbeClass& other : report.classes) {
            if (other.precision == measurement.precision && other.tier == measurement.tier &&
                other.shape == measurement.shape) {
                continue;
            }

            EXPECT_EQ(std::find(other.ranked.begin(), other.ranked.end(), measurement.name),
                      other.ranked.end())
                << measurement.name << " is ranked in " << other.name << ", another class";
        }
    }
}

// The report says in its own words what a class is and what its key is — one
// precision, one rung and one question shape, decided by the multiplier an
// option was built at and by what it hands back, and never by comparing one
// lane's documented figure against another's — and it says what the winner of a
// class is a claim about, so a reader who takes the name it prints knows which
// question the answer belongs to and how far the claim reaches. Every class the
// run made carries its key in its name, so two of them can never be read as one.
TEST(ProbeTest, TheReportSaysAClassIsOnePrecisionOneRungAndOneShape) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("the accuracy classes"), std::string::npos);
    EXPECT_NE(text.find("one precision, one rung and one question shape"), std::string::npos);
    EXPECT_NE(text.find("A class is the set of options that are alternatives for one need"),
              std::string::npos);
    EXPECT_NE(text.find("never by comparing"), std::string::npos);
    EXPECT_NE(text.find("Every row of a class was built"), std::string::npos);
    EXPECT_NE(text.find("Nothing here is ordered"), std::string::npos);
    EXPECT_NE(text.find("not a slower answer to this one"), std::string::npos);

    // Both shapes the probe measures are named with what they hand back, so a
    // reader of a class key knows which question it stands for.
    EXPECT_NE(text.find("all-orders  one argument per call"), std::string::npos);
    EXPECT_NE(text.find("all-n       one call per order run"), std::string::npos);

    // What a class's entry is a claim about, in the report's own words: one
    // rung, one question, this machine.
    EXPECT_NE(text.find("is its fastest option at"), std::string::npos);
    EXPECT_NE(text.find("for that question, on this machine and no more"), std::string::npos);

    // The axes that only change how one answer is computed are named as columns
    // of a class rather than as part of its key.
    EXPECT_NE(text.find("a column inside the class, so those options compete in one ranking"),
              std::string::npos);

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_NE(text.find(entry.name), std::string::npos) << entry.name;
        EXPECT_NE(entry.name.find("m="), std::string::npos)
            << entry.name << " does not carry the rung it was built at";
        EXPECT_NE(entry.name.find(boys::OptionProbeShapeName(entry.shape)), std::string::npos)
            << entry.name << " does not carry the question shape it is keyed on";
    }
}

// The same seed and the same options are the same workload: the values, and so
// the accuracy column, come back identical. Without this a figure could not be
// reproduced from the report alone.
TEST(ProbeTest, TheSameSeedIsTheSameWorkload) {
    ProbeOptions first = OneRound();
    ProbeOptions second = OneRound();
    second.seed = first.seed + 1;

    const OptionProbeReport a = boys::RunOptionProbe(first);
    const OptionProbeReport b = boys::RunOptionProbe(first);
    const OptionProbeReport c = boys::RunOptionProbe(second);

    ASSERT_EQ(a.measurements.size(), b.measurements.size());
    ASSERT_EQ(a.measurements.size(), c.measurements.size());
    bool anyDiffered = false;

    for (std::size_t i = 0; i < a.measurements.size(); ++i) {
        EXPECT_EQ(a.measurements[i].name, b.measurements[i].name);
        EXPECT_DOUBLE_EQ(a.measurements[i].maxError, b.measurements[i].maxError);
        EXPECT_EQ(a.measurements[i].bitIdenticalToReference,
                  b.measurements[i].bitIdenticalToReference);

        if (a.measurements[i].maxError != c.measurements[i].maxError) {
            anyDiffered = true;
        }
    }

    EXPECT_TRUE(anyDiffered) << "a different seed produced the same errors on every option";
}

// The workload options are clamped to what the kernel serves, and the report
// prints what was actually run rather than what was asked for.
TEST(ProbeTest, TheWorkloadIsClampedAndReportedAsRun) {
    ProbeOptions options = OneRound();
    options.nmax = 1000;
    options.count = 0;
    options.xLo = -1.0;
    options.xHi = -2.0;
    options.passes = 0;
    options.rounds = 0;

    const OptionProbeReport report = boys::RunOptionProbe(options);

    EXPECT_EQ(report.options.nmax, boys::kMaxBoysOrder);
    EXPECT_GE(report.options.count, 1u);
    EXPECT_GT(report.options.xLo, 0.0);
    EXPECT_GT(report.options.xHi, report.options.xLo);
    EXPECT_GE(report.options.passes, 1);
    EXPECT_GE(report.options.rounds, 1);
}

// The output is a report a reader can act on with nothing else at hand: it
// names the machine's arithmetic, the protocol, the load instrument, the
// accuracy comparison's floor, and the fact that the result describes this
// machine.
TEST(ProbeTest, TheTextStatesWhatTheResultIsAbout) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("about this machine"), std::string::npos);
    EXPECT_NE(text.find("a flag, not a gate"), std::string::npos);
    EXPECT_NE(text.find("arithmetic backends this build carries"), std::string::npos);
    EXPECT_NE(text.find("load reading"), std::string::npos);
    EXPECT_NE(text.find("not measured by this probe"), std::string::npos);

    // What the calibration found, so a reader can see whether the tool trusts
    // its own instrument here rather than having to take it on trust.
    EXPECT_NE(text.find("calibration window"), std::string::npos);
    EXPECT_NE(text.find("quiet floor"), std::string::npos);

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        EXPECT_NE(text.find(measurement.name), std::string::npos)
            << measurement.name << " is missing from the text";
    }

    for (const std::string& name : report.unoffered) {
        EXPECT_NE(text.find(name), std::string::npos) << name << " is missing from the text";
    }
}

// A caller who names a set is answered about that set: the options measured are
// the ones named, and the text scopes its fastest option to those rather than
// to the library.
TEST(ProbeTest, NamingASetMeasuresOnlyThatSet) {
    const OptionProbeReport all = boys::RunOptionProbe(OneRound());
    ASSERT_GT(all.measurements.size(), 1u)
        << "this build must offer more than one option for a subset to be narrower";

    ProbeOptions options = OneRound();
    options.only = {all.measurements.front().name, all.measurements.back().name};

    const OptionProbeReport subset = boys::RunOptionProbe(options);

    ASSERT_EQ(subset.measurements.size(), 2u);
    EXPECT_EQ(subset.measurements.front().name, all.measurements.front().name);
    EXPECT_EQ(subset.measurements.back().name, all.measurements.back().name);

    const std::string text = boys::FormatOptionProbe(subset);
    EXPECT_NE(text.find("not of the library"), std::string::npos);
}

// Naming every option measures the same set as naming none, so the default a
// caller who has not chosen yet gets is the library's whole option space.
TEST(ProbeTest, NamingEveryOptionIsTheSameSetAsNamingNone) {
    const OptionProbeReport all = boys::RunOptionProbe(OneRound());

    std::vector<std::string> names;
    for (const OptionProbeMeasurement& measurement : all.measurements)
    {
        names.push_back(measurement.name);
    }

    ProbeOptions named = OneRound();
    named.only = names;

    const OptionProbeReport namedRun = boys::RunOptionProbe(named);

    EXPECT_EQ(namedRun.measurements.size(), all.measurements.size());
    EXPECT_EQ(namedRun.unoffered.size(), all.unoffered.size());
    EXPECT_TRUE(namedRun.notAnOption.empty());
}

// A name that is no option of this library is reported rather than quietly
// measuring nothing, because an empty report otherwise reads as a machine on
// which nothing is fast.
TEST(ProbeTest, ANameThatIsNoOptionIsReported) {
    ProbeOptions options = OneRound();
    options.only = {"batch-fp64-that-never-was"};

    const OptionProbeReport report = boys::RunOptionProbe(options);

    EXPECT_TRUE(report.measurements.empty());
    ASSERT_EQ(report.notAnOption.size(), 1u);
    EXPECT_EQ(report.notAnOption.front(), "batch-fp64-that-never-was");

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("batch-fp64-that-never-was"), std::string::npos);
    EXPECT_NE(text.find("no option of this library"), std::string::npos);
}

// A name this build cannot serve is a build fact, and is kept apart from a name
// that is no option at all.
TEST(ProbeTest, ASetThisBuildCannotServeIsNotAMisspelling) {
    const OptionProbeReport all = boys::RunOptionProbe(OneRound());

    if (all.unoffered.empty())
    {
        GTEST_SKIP() << "this build offers every option, so no unoffered name exists to ask for";
    }

    ProbeOptions options = OneRound();
    options.only = {all.unoffered.front()};

    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.unoffered.size(), 1u);
    EXPECT_EQ(report.unoffered.front(), all.unoffered.front());
    EXPECT_TRUE(report.notAnOption.empty()) << "an unoffered option is a build fact, not a typo";
    EXPECT_TRUE(report.measurements.empty());
}

// The space the probe accounts for is the library's own product, not a list
// written in the probe: one cell per combination of the axes the library
// reports, every combination present, no two cells sharing a name, and a reason
// on every cell this build does not serve. A combination that is missing from
// this book is the defect the book exists to prevent.
TEST(ProbeTest, TheOptionSpaceIsTheLibrarysOwnProduct) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    ASSERT_FALSE(report.cells.empty());
    EXPECT_FALSE(report.granularities.empty());

    // The routes a lane's cells are enumerated over: the double lane's own fit
    // table, or the single-precision engine's, which is the table the float and
    // half lanes' fits are reported in. Read from the library's own report of
    // each lane rather than from one table for all of them.
    const auto routes_of = [](boys::Precision lane) {
        std::vector<boys::FitRoute> routes;
        const std::span<const boys::FitRouteInfo> table =
            (lane == boys::Precision::kFp32 || lane == boys::Precision::kFp16)
                ? boys::BoysFitRoutesF32()
                : boys::BoysFitRoutes();

        for (const boys::FitRouteInfo& row : table) {
            if (std::find(routes.begin(), routes.end(), row.route) == routes.end()) {
                routes.push_back(row.route);
            }
        }

        return routes;
    };

    // The rungs a lane serves, found the way the probe finds them: the double
    // lane's are the tiers whose reported reach differs from the reference
    // multiplier's, and a narrower lane's are the tiers the library answers a
    // figure for at that lane's own default combination.
    const auto rungs_of = [](boys::Precision lane) {
        if (lane != boys::Precision::kFp64) {
            std::size_t served = 0;

            for (int raw = 0; raw <= 32; ++raw) {
                if (boys::BoysAccuracyGuaranteed(lane, boys::kDefaultFitRoute,
                                                 boys::kDefaultEvalScheme,
                                                 boys::kDefaultPackAxis,
                                                 boys::kDefaultFitGranularity,
                                                 static_cast<boys::AccuracyTier>(raw))
                        .available) {
                    ++served;
                }
            }

            return served;
        }

        std::size_t rungs = 1;
        const double referenceReach =
            QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable;
        for (int raw = 1; raw <= 32; ++raw) {
            const auto tier = static_cast<AccuracyTier>(raw);
            if (QueryTier(tier, AccuracyRegion::kA, 0.0).reachable != referenceReach) {
                ++rungs;
            }
        }

        return rungs;
    };

    // One book per precision class, each the product of its own lane's axes, so
    // the space the report accounts for is the library's and not one lane's
    // product read four times: a class measured at fewer cells than its lane
    // serves is the defect this book exists to prevent.
    std::vector<boys::OptionPrecision> classes = {boys::OptionPrecision::kFp64,
                                                  boys::OptionPrecision::kFp32};
#if BoysFp16
    classes.push_back(boys::OptionPrecision::kFp16);
    classes.push_back(boys::OptionPrecision::kBf16);
#endif

    std::size_t expected = 0;

    for (const boys::OptionPrecision precision : classes) {
        const boys::OptionProbeCell* sample = nullptr;

        for (const boys::OptionProbeCell& cell : report.cells) {
            if (cell.precision == precision) {
                sample = &cell;
                break;
            }
        }

        ASSERT_NE(sample, nullptr) << "no cell of the space is enumerated for this precision";

        // Every cell of one class was enumerated from one lane, which is the
        // library's answer and not the test's: the two half formats are one lane
        // and two classes.
        for (const boys::OptionProbeCell& cell : report.cells) {
            if (cell.precision == precision) {
                EXPECT_EQ(cell.lane, sample->lane)
                    << cell.name << " was enumerated from another lane than its own class's";
            }
        }

        expected += routes_of(sample->lane).size() * boys::BoysEvalSchemes().size() *
                    report.granularities.size() * boys::BoysPackAxes().size() *
                    boys::BoysDivisionForms().size() * rungs_of(sample->lane);
    }

    EXPECT_EQ(report.cells.size(), expected)
        << "the coverage is not the product of the axes the library reports, lane by lane";

    std::set<std::string> names;

    for (const boys::OptionProbeCell& cell : report.cells) {
        EXPECT_FALSE(cell.name.empty());
        EXPECT_TRUE(names.insert(cell.name).second) << cell.name << " names two different cells";

        const OptionProbeMeasurement* measurement = Find(report, cell.name);

        if (cell.served) {
            EXPECT_TRUE(cell.reason.empty()) << cell.name << " is served yet gives a reason";
            ASSERT_NE(measurement, nullptr)
                << cell.name << " is served by this build yet nothing was measured for it";
            EXPECT_EQ(measurement->precision, cell.precision) << cell.name;
            EXPECT_EQ(measurement->route, cell.route) << cell.name;
            EXPECT_EQ(measurement->scheme, cell.scheme) << cell.name;
            EXPECT_EQ(measurement->granularity, cell.granularity) << cell.name;
            EXPECT_EQ(measurement->pack, cell.pack) << cell.name;
            EXPECT_EQ(measurement->division, cell.division) << cell.name;
            continue;
        }

        EXPECT_FALSE(cell.reason.empty())
            << cell.name << " is not served and the report gives no reason, which is the "
                           "unstated omission the coverage exists to prevent";
        EXPECT_EQ(measurement, nullptr) << cell.name << " is refused yet was measured";
    }
}

// The division form is an axis of the option space like the others: the members
// are read from the library rather than written out in the probe, every
// combination of the other axes is enumerated at each of them, a cell of a
// non-default form carries the library's own name for it, and the members are
// ranked against each other inside one class rather than in a class apiece. The
// defect this pins is the quiet one an unranked axis leaves: a member the
// instrument never varies reads in a report exactly like a member the library
// does not have.
//
// The protocol is the measuring one, because a class is made where a run
// produced figures and the one-round run the option book is read from produces
// none: a report from it carries no classes at all, and the ranking below would
// be asserted against nothing. The rows read above are the same rows either way,
// measured or not.
TEST(ProbeTest, TheDivisionFormAxisIsEnumeratedAndRanked) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::span<const boys::DivisionFormInfo> forms = boys::BoysDivisionForms();

    ASSERT_EQ(forms.size(), 3u) << "this build reports a division-form axis of another size";

    for (std::size_t i = 0; i < forms.size(); ++i)
    {
        EXPECT_EQ(static_cast<std::size_t>(forms[i].form), i)
            << "the rows are not in enumerator order";
        EXPECT_STREQ(forms[i].name, boys::DivisionFormName(forms[i].form));
    }

    // The no-axis cell of the double lane's reference rung, at each form. The
    // default form's carries the name the shape row has always had, because that
    // row runs the default policy and therefore divides in it; the other two
    // carry the library's own spelling of the member beside it, and the three
    // names differ — one name for two arithmetic bodies is what the segment is
    // there to prevent.
    std::set<std::string> names;

    for (const boys::DivisionFormInfo& form : forms)
    {
        const std::string name = form.form == boys::kDefaultDivisionForm
                                     ? std::string("batch-fp64")
                                     : std::string("batch-") + form.name + "-fp64";

        EXPECT_TRUE(names.insert(name).second) << name << " names two different cells";

        const OptionProbeMeasurement* row = Find(report, name);

        ASSERT_NE(row, nullptr) << name << " is not an option this run measured";
        EXPECT_EQ(row->division, form.form) << name << " is measured at a form it does not name";
        EXPECT_EQ(row->granularity, boys::FitGranularity::kShipped) << name;
    }

    // Ranked against each other, not in three classes of their own: this is the
    // class the default is chosen from, and a row of each form is in it.
    const boys::OptionProbeClass* certified =
        ClassOf(report, boys::OptionPrecision::kFp64, AccuracyTier::kReference,
                boys::OptionProbeShape::kAllOrders);

    ASSERT_NE(certified, nullptr) << "the certified double lane's class is not in the report";

    std::set<boys::DivisionForm> ranked;

    for (const std::string& name : certified->ranked)
    {
        const OptionProbeMeasurement* row = Find(report, name);

        ASSERT_NE(row, nullptr) << name << " is ranked by a class yet not measured";
        ranked.insert(row->division);
    }

    EXPECT_EQ(ranked.size(), forms.size())
        << "the class the default is chosen from ranks fewer division forms than the library "
           "reports, so the axis is carried and not compared";
}

// A name that is a cell of the space this build refuses is answered with the
// library's own reason: it is unbuilt work, counted, and kept apart from both a
// misspelling and an option this build measured.
TEST(ProbeTest, ARefusedCellIsNotAMisspelling) {
    const OptionProbeReport all = boys::RunOptionProbe(OneRound());
    const boys::OptionProbeCell* refusedCell = nullptr;

    for (const boys::OptionProbeCell& cell : all.cells) {
        if (!cell.served) {
            refusedCell = &cell;
            break;
        }
    }

    if (refusedCell == nullptr) {
        GTEST_SKIP() << "this build serves every cell, so no refused cell exists to ask for";
    }

    ProbeOptions options = OneRound();
    options.only = {refusedCell->name};

    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.refused.size(), 1u);
    EXPECT_NE(report.refused.front().find(refusedCell->name), std::string::npos);
    EXPECT_NE(report.refused.front().find(refusedCell->reason), std::string::npos);
    EXPECT_TRUE(report.notAnOption.empty()) << "a refused cell is unbuilt work, not a typo";
    EXPECT_TRUE(report.measurements.empty());

    // The coverage is the library's own book, so a narrowed run still accounts
    // for every cell of it: the caller's selection narrows the measurement and
    // never the book.
    EXPECT_EQ(report.cells.size(), all.cells.size());

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find(refusedCell->name), std::string::npos);
    EXPECT_NE(text.find("unbuilt work"), std::string::npos);
}

// A narrower lane's row is held to the bound its own lane documents, read from
// the library, and its error is reported against the certified lane's floor
// when the two partitions differ by more than the row's own bound allows — the
// honest third state, rather than a pass bought by comparing two partitions.
// A partition's figures are its stored fits' figures, certified over the
// interval those fits cover, and the row is judged by the entry's whole-domain
// figure instead: naming a partition changes the tables an entry reads and not
// the entry's own documented accuracy. The defect this pins is the other way
// round — the partition's figure used as the row's judgement, over a workload
// running past the fitted interval, which reported the narrow partition as
// missing a promise it never made.
TEST(ProbeTest, APartitionRowIsJudgedByTheEntrysWholeDomainFigure) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const boys::FitGranularityInfo* narrow = nullptr;
    for (const boys::FitGranularityInfo& partition : report.granularities) {
        if (partition.granularity == boys::FitGranularity::kNarrow) {
            narrow = &partition;
        }
    }

    ASSERT_NE(narrow, nullptr) << "this build's partition table carries no narrow partition";

    const OptionProbeMeasurement* measured = Find(report, "narrow-fp64");
    const OptionProbeMeasurement* shipped = Find(report, "batch-fp64");
    ASSERT_NE(measured, nullptr);
    ASSERT_NE(shipped, nullptr);

    EXPECT_DOUBLE_EQ(measured->bound, shipped->bound)
        << "naming a partition moved the figure the row is judged against";

    EXPECT_DOUBLE_EQ(measured->ownBound, narrow->bound);
    EXPECT_DOUBLE_EQ(measured->ownLo, narrow->lo);
    EXPECT_DOUBLE_EQ(measured->ownHi, narrow->hi);
    EXPECT_LT(measured->ownBound, measured->bound)
        << "the partition's own figure was carried as one covering the whole line";
    EXPECT_LT(measured->ownHi, std::numeric_limits<double>::infinity())
        << "the partition's own figure was carried with no interval at all";

    EXPECT_TRUE(measured->meetsBound)
        << "narrow-fp64 delivered " << measured->maxError << " against the entry's figure of "
        << measured->bound;
}

// The figure a row is judged by must hold over the arguments the row was
// measured on, and the workload range is the caller's to choose, so the verdict
// and the bound behind it must read the same over a range inside the fitted
// interval and over the default one that runs past it. Before this the narrow
// rows read "within bound" confined and "within the floor" at the default range,
// which was the judgement moving and not the option.
TEST(ProbeTest, TheVerdictDoesNotMoveWithTheWorkloadRange) {
    ProbeOptions whole = OneRound();

    ProbeOptions fitted = OneRound();
    fitted.xLo = 1e-3;
    fitted.xHi = 11.8; // inside the fitted tables' interval and below region B

    const OptionProbeReport wholeReport = boys::RunOptionProbe(whole);
    const OptionProbeReport fittedReport = boys::RunOptionProbe(fitted);

    for (const std::string name : {"narrow-fp64", "narrow-horner-fp64"}) {
        const OptionProbeMeasurement* overWhole = Find(wholeReport, name);
        const OptionProbeMeasurement* overFitted = Find(fittedReport, name);
        ASSERT_NE(overWhole, nullptr) << name;
        ASSERT_NE(overFitted, nullptr) << name;

        EXPECT_DOUBLE_EQ(overWhole->bound, overFitted->bound)
            << name << " is judged at a figure that moved with the workload range";
        EXPECT_TRUE(overWhole->meetsBound && overFitted->meetsBound)
            << name << " delivered " << overWhole->maxError << " against " << overWhole->bound
            << " over the whole range and " << overFitted->maxError << " against "
            << overFitted->bound << " over the fitted one";
    }
}

// A partition's own figure covers a narrower domain than the cells a row is
// measured on, so the row says which figure it was judged at and which figure is
// the partition's, with the interval the second holds on. A reader comparing two
// rows must not have to reach the prose below the table to learn that.
TEST(ProbeTest, APartitionRowPrintsItsOwnFiguresInterval) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const std::string text = boys::FormatOptionProbe(report);

    const OptionProbeMeasurement* measured = Find(report, "narrow-fp64");
    ASSERT_NE(measured, nullptr);

    const std::string row = RowLine(text, measured->name);
    ASSERT_FALSE(row.empty()) << "the narrow row is not in the printed table";

    char interval[64];
    std::snprintf(interval, sizeof(interval), "on x in [%.4g, %.4g) alone", measured->ownLo,
                  measured->ownHi);
    EXPECT_NE(row.find(interval), std::string::npos)
        << "the narrow row does not print the interval its own figure holds on: " << row;

    char figure[64];
    std::snprintf(figure, sizeof(figure), "%.3g", measured->ownBound);
    EXPECT_NE(row.find(figure), std::string::npos)
        << "the narrow row does not print the partition's own figure: " << row;

    EXPECT_NE(row.find("whole-domain figure"), std::string::npos)
        << "the narrow row does not say which of the two figures is the column's: " << row;
}

// What the probe does not measure is stated in its own output, with the counts:
// the cells the library refuses, the call shapes the axes are not crossed with,
// and the lanes the design leaves out. An omission that is not stated is the
// defect this list exists to prevent.
TEST(ProbeTest, TheTextStatesWhatIsNotMeasured) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("not measured by this probe, by design"), std::string::npos);
    EXPECT_NE(text.find("the native half lane"), std::string::npos);
    EXPECT_NE(text.find("region-A matrix-product transform"), std::string::npos);
    EXPECT_NE(text.find("the CUDA lanes"), std::string::npos);
    EXPECT_NE(text.find("outstanding"), std::string::npos);
    EXPECT_NE(text.find("unbuilt work"), std::string::npos);
    EXPECT_NE(text.find("refused by the library where it is named"), std::string::npos);

    // Every cell of the book is in the text under its own name, served and
    // refused alike, so no combination can be missing from the report.
    for (const boys::OptionProbeCell& cell : report.cells) {
        EXPECT_NE(text.find(cell.name), std::string::npos)
            << cell.name << " is a cell of the space and is not in the text";
    }
}

} // namespace
