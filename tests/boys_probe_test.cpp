// The option probe (boys/boys_probe.hpp): what it measures, what it refuses to
// claim, and that the second does not leave a consumer without an answer.
//
// The probe's whole value is in the claims a test can check without owning the
// machine: that every option it reports is one this build's backend table
// carries, that the comparison is formed inside a round rather than across
// rounds, and that a measurement it cannot stand behind ends in a refusal with a
// reason and a labelled fallback rather than in a name. The costs themselves are
// this machine's and asserting them here would pin a number that describes one
// host.
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

// The leader of the double lane's precision class: the fastest option of the
// precision the certified reference lane is in, and the option every ordering
// this probe makes is measured from. A faster row of another precision is not
// this, which is the whole point of the classes.
const OptionProbeMeasurement* FastestFp64(const OptionProbeReport& report) {
    const OptionProbeMeasurement* leader = nullptr;

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.measured && measurement.precision == boys::OptionPrecision::kFp64 &&
            (leader == nullptr || measurement.nsPerArgument < leader->nsPerArgument)) {
            leader = &measurement;
        }
    }

    return leader;
}

// The class a precision was ranked in, empty when the report carries none.
const boys::OptionProbeClass* ClassOf(const OptionProbeReport& report,
                                      boys::OptionPrecision precision) {
    for (const boys::OptionProbeClass& entry : report.classes) {
        if (entry.precision == precision) {
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

// A relaxed rung is documented looser than the certified lane, so it is never
// in the accuracy class the probe recommends from. The figures are still
// reported — that is what lets a reader see a faster option was faster at a
// lower accuracy rather than at the same one.
TEST(ProbeTest, ARelaxedRungIsNeverInTheCertifiedAccuracyClass) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.name.rfind("tier-", 0) == 0) {
            EXPECT_GT(measurement.bound, report.referenceBound) << measurement.name;
        }
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
    EXPECT_TRUE(report.fastestAtReferenceAccuracy.empty());
    EXPECT_TRUE(report.inseparable.empty());
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
    // of reach nothing is flagged, and the run is the same run.
    const OptionProbeReport quiet = boys::RunOptionProbe(TimedWithAlarm(1e9));

    EXPECT_EQ(quiet.passesWithinAlarm, static_cast<int>(quiet.passes.size()));
    EXPECT_EQ(quiet.passesAboveAlarm, 0);
    EXPECT_EQ(quiet.pairedRounds, flagged.pairedRounds);
}

// The resolution is measured in the quantities the ordering is made of, and it
// is not a bar chosen in advance: it is the widest within-round band the
// certified double lane's precision showed, so no option's own band is wider
// than it. The figures move with the machine; this relation does not.
TEST(ProbeTest, TheResolutionIsTheWidestBandTheClassShowed) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.pairedRounds < 4) {
        EXPECT_DOUBLE_EQ(report.resolution, 0.0);
        EXPECT_TRUE(report.recommended.empty());
        return;
    }

    double widest = 0.0;
    std::size_t doubled = 0;

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.measured && measurement.precision == boys::OptionPrecision::kFp64) {
            ++doubled;
            widest = std::max(widest, measurement.spread - 1.0);
        }
    }

    if (doubled == 0) {
        EXPECT_DOUBLE_EQ(report.resolution, 0.0);
        return;
    }

    EXPECT_GE(report.resolution, widest)
        << "a band of the class's own measurement is wider than the class's resolution";
    EXPECT_GT(report.resolution, 0.0);
}

// A recommendation and a refusal are complements of the same rule: the class's
// leader was placed ahead of every other option of that class, or something in
// it could not be placed and is named. The class is the precision, so a faster
// row of another precision is never a rival of it.
TEST(ProbeTest, ARecommendationLeavesNoRivalUnplaced) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const OptionProbeMeasurement* leader = FastestFp64(report);
    const boys::OptionProbeClass* doubles = ClassOf(report, boys::OptionPrecision::kFp64);

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        ASSERT_NE(leader, nullptr);

        EXPECT_TRUE(report.inseparable.empty());
        EXPECT_EQ(report.recommended, leader->name);
        EXPECT_TRUE(report.hasDefault);
        EXPECT_TRUE(report.heuristicOption.empty())
            << "a measured winner was printed with a heuristic fallback beside it";

        ASSERT_NE(doubles, nullptr);
        EXPECT_EQ(doubles->leader, leader->name);
        EXPECT_TRUE(doubles->ordered);

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == boys::OptionPrecision::kFp64 &&
                measurement.name != leader->name) {
                EXPECT_LT(leader->nsPerArgument, measurement.nsPerArgument) << measurement.name;
            }
        }

        return;
    }

    EXPECT_TRUE(report.recommended.empty());
    EXPECT_FALSE(report.reason.empty());

    if (leader != nullptr && report.pairedRounds >= 4) {
        EXPECT_FALSE(report.inseparable.empty())
            << "the class was measured and the verdict refused, yet nothing is named as unplaced";
        ASSERT_NE(doubles, nullptr);
        EXPECT_FALSE(doubles->ordered);
    }
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

    const bool banded = report.pairedRounds >= 4 && FastestFp64(report) != nullptr;

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

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured || measurement.precision != boys::OptionPrecision::kFp64) {
            continue;
        }

        if (inClass == 0) {
            route = measurement.arithmetic;
        } else if (measurement.arithmetic != route) {
            oneRoute = false;
        }

        ++inClass;
    }

    if (report.pairedRounds < 4 || inClass < 2) {
        EXPECT_EQ(report.confidence.find("No pair's ratio moved"), std::string::npos)
            << report.confidence;
        return;
    }

    const bool heldStill = report.confidence.find("No pair's ratio moved") != std::string::npos;
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

// A band is a lower and an upper quartile, and two rounds have neither: the
// probe refuses rather than reporting a width it did not measure. The refusal
// names the round count it needs, and it leaves the caller the static fallback —
// labelled as a heuristic — so a consumer who cannot afford a longer run is not
// left with nothing.
TEST(ProbeTest, AnOrderingNeedsFourPairedRounds) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 2);
    EXPECT_FALSE(report.measurements.empty());
    ASSERT_TRUE(report.measurements.front().measured)
        << "two rounds can form a ratio, just not a quartile band";

    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_TRUE(report.inseparable.empty());
    EXPECT_DOUBLE_EQ(report.resolution, 0.0);
    EXPECT_NE(report.reason.find("quartile"), std::string::npos) << report.reason;
    EXPECT_EQ(report.confidence.rfind("CANNOT DETERMINE", 0), 0u) << report.confidence;

    ASSERT_FALSE(report.heuristicOption.empty())
        << "a refusal left the caller without any default at all";
    EXPECT_TRUE(report.hasDefault);

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("not measurable on this run"), std::string::npos);
    EXPECT_NE(text.find("2 paired round(s)"), std::string::npos);
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

// A refusal leaves a default, and the default is never presented as a
// measurement: it is named in its own labelled section, its basis says in its
// first words that it was counted rather than timed, and the row it belongs to
// still reads 'not measured' in the cost table. This run is one pass of one
// round, so no ratio can be formed at all and nothing about cost is measured.
TEST(ProbeTest, TheFallbackIsLabelledAndIsNotAMeasurement) {
    ProbeOptions options = OneRound();
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 1);
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_NE(report.reason.find("no option produced a figure"), std::string::npos) << report.reason;

    ASSERT_FALSE(report.heuristicOption.empty())
        << "this run's option set holds nothing the static rule can rank";
    EXPECT_TRUE(report.hasDefault);
    EXPECT_EQ(report.heuristicBasis.rfind("counted, not timed", 0), 0u) << report.heuristicBasis;
    EXPECT_EQ(report.heuristicBasis.find("ns/argument"), std::string::npos)
        << "the heuristic's own basis quotes a cost, which is not what it was chosen from";

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("static fallback — a heuristic, not a measurement"), std::string::npos);
    EXPECT_NE(text.find(report.heuristicOption), std::string::npos);
    EXPECT_NE(text.find(report.heuristicBasis), std::string::npos);

    const std::string row = RowLine(text, report.heuristicOption);
    ASSERT_FALSE(row.empty()) << "the fallback's row is not in the printed table";
    EXPECT_NE(row.find("not measured"), std::string::npos)
        << "the heuristic's row carries a cost it was never measured at: " << row;
}

// The other kind of refusal — a class that measured, could not be ordered — is
// the one where the fallback and a cost column can meet, and the report still
// keeps the two kinds of answer apart: the row carries the figure the option was
// measured at and the heuristic says in its own section that it was chosen by
// counting. A reader who takes the fallback's name is told which of the two they
// are taking.
TEST(ProbeTest, AMeasuredButUnorderedRunLabelsItsFallbackToo) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.verdict != OptionProbeVerdict::kCannotDetermine) {
        GTEST_SKIP() << "this run's class ordered itself, so there is no refusal and no fallback";
    }

    ASSERT_FALSE(report.heuristicOption.empty());
    ASSERT_TRUE(report.hasDefault);
    EXPECT_EQ(report.reason.find(report.heuristicBasis), std::string::npos)
        << "the fallback's basis is printed as part of the measured reason";

    const std::string text = boys::FormatOptionProbe(report);
    ASSERT_NE(text.find("static fallback — a heuristic, not a measurement"), std::string::npos);
    EXPECT_EQ(text.find(report.heuristicBasis),
              text.rfind(report.heuristicBasis))
        << "the heuristic's basis appears more than once, so it is printed beside figures too";

    const OptionProbeMeasurement* fallback = Find(report, report.heuristicOption);
    ASSERT_NE(fallback, nullptr);
    EXPECT_TRUE(fallback->measured);

    const std::string row = RowLine(text, report.heuristicOption);
    ASSERT_FALSE(row.empty());
    EXPECT_EQ(row.find("not measured"), std::string::npos) << row;

    // A refusal is where the clock check earns its keep: it is the line that
    // tells a reader whether the class's bands were wide because the clock
    // wandered or because the options are close, so a refusal carries it too.
    const bool clocked = report.confidence.find("No pair's ratio moved") != std::string::npos ||
                         report.confidence.find("WARNING: the pair") != std::string::npos;
    EXPECT_TRUE(clocked) << report.confidence;
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

// The verdict and the names it is built from always agree with each other: a
// recommendation names the leader of the double lane's precision class, the
// narrowest reading names the fastest option of that class at the certified
// bound, and a refusal names nothing.
TEST(ProbeTest, TheVerdictAndTheNamesAgree) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_TRUE(report.inseparable.empty());

        const OptionProbeMeasurement* leader = Find(report, report.recommended);
        ASSERT_NE(leader, nullptr);
        EXPECT_TRUE(leader->measured);
        EXPECT_GT(leader->nsPerArgument, 0.0);
        EXPECT_EQ(leader->precision, boys::OptionPrecision::kFp64)
            << "the recommendation is the leader of the double lane's precision, never of another";
        EXPECT_EQ(leader, FastestFp64(report));

        // The class's own leader, and the narrowest reading of it: the fastest
        // row whose own bound is the certified lane's, which is never faster
        // than the class leader and never held to a looser bound.
        if (!report.fastestAtReferenceAccuracy.empty()) {
            const OptionProbeMeasurement* certified =
                Find(report, report.fastestAtReferenceAccuracy);
            ASSERT_NE(certified, nullptr);
            EXPECT_TRUE(certified->measured);
            EXPECT_LE(certified->bound, report.referenceBound);
            EXPECT_GE(certified->nsPerArgument, leader->nsPerArgument);
        }
    } else {
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_FALSE(report.reason.empty());
        EXPECT_EQ(report.confidence.rfind("CANNOT DETERMINE", 0), 0u) << report.confidence;
    }
}

// A class is one precision. Nothing inside a class was measured in another
// precision, the leader of a class is its fastest member, and a class that says
// it is ordered holds no unplaced member — the rule the verdict is made from,
// stated once here so a report cannot rank across precisions. A class is built
// from the rounds the run took, so an uncalibrated instrument still produces
// them; what a short run cannot produce is a band, and then nothing is ordered.
TEST(ProbeTest, EveryClassIsOnePrecisionAndRanksOnlyItsOwn) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    EXPECT_EQ(report.classes.size(), 4u) << "one class per precision this library declares";

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_FALSE(entry.name.empty()) << "a class with no name is one a reader cannot place";

        if (entry.leader.empty()) {
            EXPECT_TRUE(entry.ranked.empty()) << entry.name;
            EXPECT_FALSE(entry.ordered) << entry.name << " has no leader to be ordered around";
            EXPECT_FALSE(entry.note.empty()) << entry.name;

            for (const OptionProbeMeasurement& measurement : report.measurements) {
                EXPECT_NE(measurement.precision, entry.precision)
                    << entry.name << " reports no figure yet " << measurement.name
                    << " was measured in it";
            }
            continue;
        }

        ASSERT_FALSE(entry.ranked.empty()) << entry.name;

        const OptionProbeMeasurement* leader = Find(report, entry.leader);
        ASSERT_NE(leader, nullptr) << entry.name;
        EXPECT_TRUE(leader->measured) << entry.name << " names a leader that was never measured";
        EXPECT_EQ(leader->precision, entry.precision) << entry.name;
        EXPECT_DOUBLE_EQ(entry.leaderNsPerArgument, leader->nsPerArgument) << entry.name;
        EXPECT_EQ(entry.ranked.front(), entry.leader)
            << entry.name << " ranks its leader somewhere behind its own first row";

        std::size_t measured = 0;

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == entry.precision) {
                ++measured;
                EXPECT_GE(measurement.nsPerArgument, leader->nsPerArgument)
                    << entry.name << " names " << entry.leader << " as its leader with "
                    << measurement.name << " measured faster in it";
            }
        }

        EXPECT_EQ(entry.ranked.size(), measured) << entry.name << " does not rank every member";

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

            if (entry.ordered && name != entry.leader) {
                EXPECT_GT(member->nsPerArgument, leader->nsPerArgument)
                    << entry.name << " says it is ordered, yet " << name
                    << " is not behind its leader in the run's own statistic";
            }
        }
    }

    for (std::size_t i = 0; i < report.classes.size(); ++i) {
        for (std::size_t j = i + 1; j < report.classes.size(); ++j) {
            EXPECT_NE(report.classes[i].precision, report.classes[j].precision)
                << "two classes claim one precision, so an option could be ranked in both";
        }
    }

    // Every figure the run produced is ranked in the class of its own precision
    // and nowhere else: an option measured in a precision whose class does not
    // name it would be a figure a reader could not place.
    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured) {
            continue;
        }

        const boys::OptionProbeClass* entry = ClassOf(report, measurement.precision);
        ASSERT_NE(entry, nullptr)
            << measurement.name << " was measured in a precision the report classes nowhere";
        EXPECT_NE(std::find(entry->ranked.begin(), entry->ranked.end(), measurement.name),
                  entry->ranked.end())
            << measurement.name << " is measured in " << entry->name
            << " and is not ranked in it";

        for (const boys::OptionProbeClass& other : report.classes) {
            if (other.precision == measurement.precision) {
                continue;
            }

            EXPECT_EQ(std::find(other.ranked.begin(), other.ranked.end(), measurement.name),
                      other.ranked.end())
                << measurement.name << " is ranked in another precision's class";
        }
    }
}

// A class states in the report that it is one precision and that the bounds
// inside it differ by row, so nobody reads its leader as the fastest option at
// their own accuracy when it is only the fastest at some accuracy in that
// precision.
TEST(ProbeTest, TheReportSaysAClassIsOnePrecision) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("the precision classes"), std::string::npos);
    EXPECT_NE(text.find("A class is one precision"), std::string::npos);
    EXPECT_NE(text.find("the bounds inside it differ by row"), std::string::npos);
    EXPECT_NE(text.find("and not the fastest at yours"), std::string::npos);
    EXPECT_NE(text.find("Nothing here is ordered across classes"), std::string::npos);

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_NE(text.find(entry.name), std::string::npos) << entry.name;
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

    // The axes' own sizes, read from the tables the library reports.
    std::vector<boys::FitRoute> routes;
    for (const boys::FitRouteInfo& row : boys::BoysFitRoutes()) {
        if (std::find(routes.begin(), routes.end(), row.route) == routes.end()) {
            routes.push_back(row.route);
        }
    }

    // The rungs this build serves, found the way the probe finds them: a tier
    // whose reported reach differs from the reference multiplier's.
    std::size_t rungs = 1;
    const double referenceReach =
        QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable;
    for (int raw = 1; raw <= 32; ++raw) {
        const auto tier = static_cast<AccuracyTier>(raw);
        if (QueryTier(tier, AccuracyRegion::kA, 0.0).reachable != referenceReach) {
            ++rungs;
        }
    }

    const std::size_t expected = routes.size() * boys::BoysEvalSchemes().size() *
                                 report.granularities.size() * boys::BoysPackAxes().size() * rungs;
    EXPECT_EQ(report.cells.size(), expected)
        << "the coverage is not the product of the axes the library reports";

    std::set<std::string> names;

    for (const boys::OptionProbeCell& cell : report.cells) {
        EXPECT_FALSE(cell.name.empty());
        EXPECT_TRUE(names.insert(cell.name).second) << cell.name << " names two different cells";

        const OptionProbeMeasurement* measurement = Find(report, cell.name);

        if (cell.served) {
            EXPECT_TRUE(cell.reason.empty()) << cell.name << " is served yet gives a reason";
            ASSERT_NE(measurement, nullptr)
                << cell.name << " is served by this build yet nothing was measured for it";
            EXPECT_EQ(measurement->precision, boys::OptionPrecision::kFp64) << cell.name;
            EXPECT_EQ(measurement->route, cell.route) << cell.name;
            EXPECT_EQ(measurement->scheme, cell.scheme) << cell.name;
            EXPECT_EQ(measurement->granularity, cell.granularity) << cell.name;
            EXPECT_EQ(measurement->pack, cell.pack) << cell.name;
            continue;
        }

        EXPECT_FALSE(cell.reason.empty())
            << cell.name << " is not served and the report gives no reason, which is the "
                           "unstated omission the coverage exists to prevent";
        EXPECT_EQ(measurement, nullptr) << cell.name << " is refused yet was measured";
    }
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
