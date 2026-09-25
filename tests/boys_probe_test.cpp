// The option probe (boys/boys_probe.hpp): what it measures, what it refuses to
// claim, and that the second does not depend on the first.
//
// The probe's whole value is in the two claims a test can check without owning
// the machine: that every option it reports is one this build's backend table
// carries, and that a measurement it cannot stand behind ends in a refusal with
// a reason rather than in a name. The costs themselves are this machine's and
// asserting them here would pin a number that describes one host.
//
// The protocols below are short on purpose: a probe test that took a minute
// would be a probe nobody runs. One keeps the timed loop out of the way
// entirely — an empty calibration window, so no pass can be judged and none is
// run — and two others move the canary's own bar to each end, so the admission
// rule is exercised on a real run from both sides.

#include "boys/boys_probe.hpp"

#include <algorithm>
#include <cstdio>
#include <gtest/gtest.h>
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

// A workload small enough to run in a test, with a protocol that never reaches
// the timed loop: an empty calibration window leaves the canary without a floor
// to be read against, and a pass that cannot be judged is not run at all.
ProbeOptions Untimed() {
    ProbeOptions options;
    options.count = 256;
    options.nmax = 8;
    options.calibrationSeconds = 0.0;
    return options;
}

// The same workload with a calibration window and two passes of one round: the
// shortest protocol that can both produce a figure and support an ordering, since
// a resolution needs two admitted passes to exist.
ProbeOptions Timed() {
    ProbeOptions options = Untimed();
    options.calibrationSeconds = 0.5;
    options.backgroundWindowSeconds = 0.2;
    options.passes = 2;
    options.rounds = 1;
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

// Whether a name was reported as not offered, rather than measured.
bool Unoffered(const OptionProbeReport& report, const std::string& name) {
    for (const std::string& entry : report.unoffered) {
        if (entry == name) {
            return true;
        }
    }
    return false;
}

TEST(ProbeTest, EveryReportedOptionRunsInArithmeticThisBuildCarries) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());
    const OptionProbeMeasurement* batch = Find(report, "batch-fp64");

    ASSERT_NE(batch, nullptr);
    EXPECT_EQ(batch->arithmetic,
              boys::BoysAvx2Available() ? "avx2-fp64" : "scalar-fp64");
}

// The fp32 lane is offered whatever the vector tier does, and it is the fp32
// arithmetic, never the fp64 one under a narrower name.
TEST(ProbeTest, TheFp32OptionRunsInFp32Arithmetic) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());
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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

    EXPECT_TRUE(Unoffered(report, "batch-fp64") == false);
    EXPECT_TRUE(Find(report, "batch-fp64") != nullptr);
    EXPECT_TRUE(Unoffered(report, "batch-fp32") == false);
    EXPECT_TRUE(Find(report, "batch-fp32") != nullptr);
    EXPECT_TRUE(Find(report, "grouped-fp64") != nullptr || Unoffered(report, "grouped-fp64"));
    EXPECT_TRUE(Find(report, "tagged-fp64") != nullptr || Unoffered(report, "tagged-fp64"));

#if BoysFp16
    EXPECT_TRUE(Find(report, "f16-io") != nullptr || Unoffered(report, "f16-io"));
    EXPECT_TRUE(Find(report, "bf16-io") != nullptr || Unoffered(report, "bf16-io"));
#endif
}

// The reference bound the report prints is the library's, not a number written
// in the probe: it is what QueryTier reports for the reference tier.
TEST(ProbeTest, TheReferenceBoundIsReadFromTheLibrary) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

    for (const AccuracyRegion region :
         {AccuracyRegion::kA, AccuracyRegion::kB, AccuracyRegion::kC}) {
        EXPECT_LE(QueryTier(AccuracyTier::kReference, region, 0.0).reachable,
                  report.referenceBound);
    }
    EXPECT_DOUBLE_EQ(report.referenceBound,
                     QueryTier(AccuracyTier::kReference, AccuracyRegion::kA, 0.0).reachable);
}

// The accuracy column is measured whether or not a single pass was run: the
// values a lane returns for an argument are a property of the build, and a busy
// machine must not be able to blank the column.
TEST(ProbeTest, TheAccuracyColumnDoesNotDependOnACleanPass) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

    ASSERT_FALSE(report.calibrated);
    EXPECT_TRUE(report.passes.empty());

    const OptionProbeMeasurement* batch = Find(report, "batch-fp64");
    ASSERT_NE(batch, nullptr);
    EXPECT_FALSE(batch->measured);
    EXPECT_TRUE(batch->meetsBound);
    EXPECT_LE(batch->maxError, batch->bound);
}

// The column is a measurement, not a constant: the fp64 options sit at the
// certified lane's own resolution, and the fp32 lane's errors are orders above
// them. A column that read zero for every option would pass every bound in this
// file and tell a reader nothing.
TEST(ProbeTest, TheAccuracyColumnSeparatesTheLanes) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.name.rfind("tier-", 0) == 0) {
            EXPECT_GT(measurement.bound, report.referenceBound) << measurement.name;
        }
    }
}

// The refusal when nothing can be measured: an instrument that never found a
// floor cannot say which passes were clean, so no pass is run and no option is
// named. The output must say so in its own words.
TEST(ProbeTest, AnUncalibratedInstrumentRefusesRatherThanGuesses) {
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());

    EXPECT_FALSE(report.calibrated);
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_TRUE(report.fastestAtReferenceAccuracy.empty());
    EXPECT_FALSE(report.reason.empty());
    EXPECT_EQ(report.cleanPasses, 0);

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("CANNOT DETERMINE"), std::string::npos);
    EXPECT_NE(text.find("could not be calibrated"), std::string::npos);
}

// The admission rule from the permissive side: a bar no spread can reach admits
// every pass, however the load readings look, because the load readings are not
// what admits one. The canary's own readings are recorded either way.
TEST(ProbeTest, ACanaryThatRepeatsItselfAdmitsThePass) {
    ProbeOptions options = Timed();
    options.canarySpreadThreshold = 1e9;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    EXPECT_EQ(report.cleanPasses + report.disturbedPasses,
              static_cast<int>(report.passes.size()));
    EXPECT_EQ(report.disturbedPasses, 0);

    double widestCanary = 0.0;

    for (const boys::OptionProbePass& pass : report.passes) {
        EXPECT_FALSE(pass.disturbed) << "a bar of 1e9 discarded a pass";
        EXPECT_GE(pass.canarySpread, 0.0);
        widestCanary = std::max(widestCanary, pass.canarySpread);
    }

    EXPECT_DOUBLE_EQ(report.canarySpread, widestCanary);
}

// The same rule from the other side: a bar below anything repeated fixed work
// can manage discards every pass, and the probe then refuses rather than
// reporting a figure taken on a pass it could not vouch for. This is the one
// refusal that exists because the machine, not the field of options, was the
// problem.
TEST(ProbeTest, APassTheCanaryCannotVouchForIsDiscarded) {
    ProbeOptions options = Timed();
    options.canarySpreadThreshold = 1e-9;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    for (const boys::OptionProbePass& pass : report.passes) {
        EXPECT_TRUE(pass.disturbed) << "a bar of 1e-9 admitted a pass";
    }

    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_TRUE(report.inseparable.empty());
    EXPECT_FALSE(report.reason.empty());

    if (!report.passes.empty()) {
        EXPECT_EQ(report.cleanPasses, 0);
        EXPECT_EQ(report.disturbedPasses, static_cast<int>(report.passes.size()));
        EXPECT_NE(report.reason.find("discarded"), std::string::npos) << report.reason;
    }
}

// The resolution is a measurement and not a bar chosen in advance: it is the
// larger of the two spreads this run actually saw — the canary's widest admitted
// pass and the leading option's own clean passes. The figures move with the
// machine; this relation does not.
TEST(ProbeTest, TheResolutionIsWhatTheRunMeasured) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    double widestCanary = 0.0;

    for (const boys::OptionProbePass& pass : report.passes) {
        if (!pass.disturbed) {
            widestCanary = std::max(widestCanary, pass.canarySpread);
        }
    }

    EXPECT_DOUBLE_EQ(report.canarySpread, widestCanary);

    const OptionProbeMeasurement* leader = Find(report, report.fastestAtReferenceAccuracy);
    const bool orderable = leader != nullptr && report.cleanPasses >= 2;

    if (!orderable) {
        EXPECT_DOUBLE_EQ(report.resolution, 0.0);
        EXPECT_TRUE(report.recommended.empty());
    } else {
        EXPECT_DOUBLE_EQ(report.resolution,
                         std::max(report.canarySpread / 100.0, leader->spread - 1.0));
        EXPECT_GE(report.resolution, 0.0);
    }
}

// Nothing inside the resolution is ever named: a recommendation means every
// rival in the accuracy class was measured further behind than this run could
// order, and the refusal means at least one was not, with that option listed.
// Without this the probe could print a resolution and recommend against it.
TEST(ProbeTest, NothingInsideTheResolutionIsOrdered) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const OptionProbeMeasurement* leader = Find(report, report.fastestAtReferenceAccuracy);

    if (leader == nullptr || report.cleanPasses < 2) {
        // No ordering was possible, so there is nothing for one to disagree with.
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_TRUE(report.inseparable.empty());
        EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
        EXPECT_DOUBLE_EQ(report.resolution, 0.0);
        return;
    }

    bool anyInside = false;

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured || measurement.name == leader->name) {
            continue;
        }

        if (measurement.bound <= report.referenceBound &&
            measurement.nsPerArgument <= leader->nsPerArgument * (1.0 + report.resolution)) {
            anyInside = true;
        }
    }

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(anyInside) << report.reason;
        EXPECT_TRUE(report.inseparable.empty());
        EXPECT_EQ(report.recommended, leader->name);
    } else {
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_EQ(anyInside, !report.inseparable.empty()) << report.reason;
    }
}

// The output states the resolution in the reader's own terms, so a refusal is a
// measurement with a number attached rather than a shrug. When the run had no
// leader to measure a resolution from, the text says that instead.
TEST(ProbeTest, TheTextStatesTheResolution) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("resolution:"), std::string::npos);
    EXPECT_NE(text.find("not the gate"), std::string::npos);
    EXPECT_NE(text.find("this machine"), std::string::npos);

    const bool orderable =
        !report.fastestAtReferenceAccuracy.empty() && report.cleanPasses >= 2;

    if (orderable) {
        EXPECT_NE(text.find("cannot be ordered on this run"), std::string::npos);
    } else {
        EXPECT_NE(text.find("not measurable on this run"), std::string::npos);
    }
}

// One admitted pass is not enough to order anything: the resolution is how far
// the admitted passes disagreed with each other, and a single pass has nothing
// to disagree with. The probe refuses rather than reporting a precision the run
// never measured, and says which pass count it needed.
TEST(ProbeTest, AnOrderingNeedsTwoAdmittedPasses) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.canarySpreadThreshold = 1e9; // so the single pass is admitted
    const OptionProbeReport report = boys::RunOptionProbe(options);

    if (!report.calibrated || report.passes.empty()) {
        return; // this machine was too busy for the canary to settle; nothing to assert
    }

    ASSERT_EQ(report.cleanPasses, 1);
    ASSERT_TRUE(report.measurements.front().measured);
    EXPECT_FALSE(report.fastestAtReferenceAccuracy.empty());
    EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    EXPECT_TRUE(report.recommended.empty());
    EXPECT_TRUE(report.inseparable.empty());
    EXPECT_DOUBLE_EQ(report.resolution, 0.0);
    EXPECT_NE(report.reason.find("second pass"), std::string::npos) << report.reason;

    const std::string text = boys::FormatOptionProbe(report);
    EXPECT_NE(text.find("not measurable on this run"), std::string::npos);
    EXPECT_NE(text.find("1 of 1 was admitted"), std::string::npos);
}

// Whatever the machine did, the protocol's bookkeeping adds up: every pass is
// either clean or disturbed for every option, and a figure that exists was
// taken on a clean pass at a positive cost.
TEST(ProbeTest, ThePassBookkeepingAddsUp) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    EXPECT_EQ(report.cleanPasses + report.disturbedPasses,
              static_cast<int>(report.passes.size()));

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        EXPECT_EQ(measurement.cleanPasses + measurement.disturbedPasses,
                  static_cast<int>(report.passes.size()));

        if (measurement.measured) {
            EXPECT_GT(measurement.cleanPasses, 0);
            EXPECT_GT(measurement.nsPerArgument, 0.0) << measurement.name;
            EXPECT_GE(measurement.nsPerArgumentMax, measurement.nsPerArgument) << measurement.name;
            EXPECT_GE(measurement.spread, 1.0) << measurement.name;
            EXPECT_NE(measurement.checkedSum, 0.0) << measurement.name;
        } else {
            EXPECT_EQ(measurement.cleanPasses, 0) << measurement.name;
        }
    }

    if (!report.calibrated) {
        EXPECT_TRUE(report.passes.empty());
        EXPECT_EQ(report.verdict, OptionProbeVerdict::kCannotDetermine);
    } else {
        EXPECT_FALSE(report.passes.empty());
        EXPECT_FALSE(report.reason.empty());
    }
}

// The verdict and the two option names it is built from always agree with each
// other: a recommendation names the fastest option in the certified class, and
// a refusal names nothing.
TEST(ProbeTest, TheVerdictAndTheNamesAgree) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_EQ(report.recommended, report.fastestAtReferenceAccuracy);
        EXPECT_TRUE(report.inseparable.empty());

        const OptionProbeMeasurement* leader = Find(report, report.recommended);
        ASSERT_NE(leader, nullptr);
        EXPECT_TRUE(leader->measured);
        EXPECT_LE(leader->bound, report.referenceBound);
        EXPECT_GT(leader->nsPerArgument, 0.0);
    } else {
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_FALSE(report.reason.empty());
        EXPECT_EQ(report.confidence.rfind("CANNOT DETERMINE", 0), 0u) << report.confidence;
    }
}

// The same seed and the same options are the same workload: the values, and so
// the accuracy column, come back identical. Without this a figure could not be
// reproduced from the report alone.
TEST(ProbeTest, TheSameSeedIsTheSameWorkload) {
    ProbeOptions first = Untimed();
    ProbeOptions second = Untimed();
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
    ProbeOptions options = Untimed();
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
    const OptionProbeReport report = boys::RunOptionProbe(Untimed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("about this machine"), std::string::npos);
    EXPECT_NE(text.find("disturbed"), std::string::npos);
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
    const OptionProbeReport all = boys::RunOptionProbe(Untimed());
    ASSERT_GT(all.measurements.size(), 1u)
        << "this build must offer more than one option for a subset to be narrower";

    ProbeOptions options = Untimed();
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
    const OptionProbeReport all = boys::RunOptionProbe(Untimed());

    std::vector<std::string> names;
    for (const OptionProbeMeasurement& measurement : all.measurements)
    {
        names.push_back(measurement.name);
    }

    ProbeOptions named = Untimed();
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
    ProbeOptions options = Untimed();
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
    const OptionProbeReport all = boys::RunOptionProbe(Untimed());

    if (all.unoffered.empty())
    {
        GTEST_SKIP() << "this build offers every option, so no unoffered name exists to ask for";
    }

    ProbeOptions options = Untimed();
    options.only = {all.unoffered.front()};

    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.unoffered.size(), 1u);
    EXPECT_EQ(report.unoffered.front(), all.unoffered.front());
    EXPECT_TRUE(report.notAnOption.empty()) << "an unoffered option is a build fact, not a typo";
    EXPECT_TRUE(report.measurements.empty());
}

} // namespace
