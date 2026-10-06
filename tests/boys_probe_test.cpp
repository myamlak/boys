// The option probe (boys/boys_probe.hpp): what it measures, what it refuses to claim, and that the second does not
// leave a consumer without an answer. Every option reported must be one this build's backend table carries; the
// comparison must be formed inside a round and not across rounds; and a measured class must end in exactly one
// default, its way of being reached stated and never a set of candidates or a name a table was counted for.

// The costs are this machine's and neither asserted nor pinned to a number that describes one host. The protocols
// below are short on purpose: two are the shortest there are (one pass of one round, empty calibration window - they
// cost nothing and conclude nothing about cost) and the third is the shortest that can order anything, a quartile
// band needing four paired rounds to be formed at all.

#include "boys/boys_probe.hpp"

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

using boys::OptionPrecision;
using boys::OptionProbeDefaultHow;
using boys::OptionProbeMeasurement;
using boys::OptionProbeReport;
using boys::OptionProbeVerdict;
using boys::ProbeOptions;

// The shortest run there is: one pass of one round on a workload small enough to be free, with an empty
// calibration window so the load instrument never finds a floor. Nothing about cost is concluded from it
// (a ratio needs two rounds, a band four), and the tests about the option book, the accuracy column and
// the refusal paths read it.
ProbeOptions OneRound() {
    ProbeOptions options;
    options.count = 256;
    options.nmax = 8;
    options.calibrationSeconds = 0.0;
    options.passes = 1;
    options.rounds = 1;
    return options;
}

// The shortest protocol that can order anything: four paired rounds, which is what
// a lower and upper quartile need, over the same small workload. Every option is
// called once in every round, so the ratios are formed inside a round.
ProbeOptions Timed() {
    ProbeOptions options = OneRound();
    options.calibrationSeconds = 0.5;
    options.backgroundWindowSeconds = 0.2;
    options.passes = 4;
    options.rounds = 2;
    return options;
}

// The same protocol with the canary alarm moved. The alarm decides one thing - the
// flag printed beside a pass - and a test about that flag has to be able to put it
// anywhere, including below anything repeated fixed work can reach.
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

// The certified double lane's all-orders class, read from the library the way the
// probe reads it: every name below is formed against this class's own row, so a
// test that moves an axis names the cell that row moved the axis on.
using DoubleAllOrders = boys::DefaultPolicy<boys::Precision::kFp64, boys::Shape::kAllOrders>;

// The name the probe prints one cell of the certified double lane's all-orders class under: that class's
// own row, read from the library, with a segment for each axis on which the named cell departs from it.
// Every axis not named is the row's, so the cells these tests reach move the partition and the scheme,
// and the row's own cell - departing on no axis - is `batch-fp64`, the cell the class's call compiles.
std::string DoubleCellName(boys::FitGranularity granularity, boys::EvalScheme scheme) {
    std::string name = "batch";

    if (granularity != DoubleAllOrders::kGranularity) {
        name += "-";
        name += boys::GranularityName(granularity);
    }

    if (scheme != DoubleAllOrders::kScheme) {
        name += "-";
        name += boys::EvalSchemeName(scheme);
    }

    return name + "-fp64";
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

// The members of the class the probe takes its default from: the certified double
// lane's precision at the library's own full-accuracy multiplier, answering the
// all-orders question - the shape the workload is asked in. The key is that pair,
// so a faster row of another precision or shape is a different class.
std::vector<const OptionProbeMeasurement*> ReferenceMembers(const OptionProbeReport& report) {
    std::vector<const OptionProbeMeasurement*> members;

    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (measurement.measured && measurement.precision == boys::OptionPrecision::kFp64 &&
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

// The class one precision was ranked in for one question shape, empty when the
// report carries none. Every class of the report is keyed by that pair.
const boys::OptionProbeClass* ClassOf(const OptionProbeReport& report,
                                      boys::OptionPrecision precision,
                                      boys::OptionProbeShape shape) {
    for (const boys::OptionProbeClass& entry : report.classes) {
        if (entry.precision == precision && entry.shape == shape) {
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

// The distinct routes a lane's own fit table reports, read from the library the way
// the probe reads them: a route has one row per region it supplies, so the route is
// taken once, and that taken-once count is one factor of every class of that lane.
std::vector<boys::FitRoute> DistinctRoutes(boys::Precision lane) {
    const std::span<const boys::FitRouteInfo> table =
        (lane == boys::Precision::kFp32 || lane == boys::Precision::kFp16)
            ? boys::BoysFitRoutesF32()
            : boys::BoysFitRoutes();
    std::vector<boys::FitRoute> routes;

    for (const boys::FitRouteInfo& row : table) {
        if (std::find(routes.begin(), routes.end(), row.route) == routes.end()) {
            routes.push_back(row.route);
        }
    }

    return routes;
}

// Whether each lane's all-N entry declares, beside its own call, the overload taking the arguments as
// already sorted - the one factor of a class's space no axis table reports, asked of the library's own
// declarations as the probe asks it. The fp64 lane's entry carries it and its answer is read from the
// call; the other lanes' answers are templates over the lane's argument type, deferring the requirement.
constexpr bool kFp64AllNDeclaresSorted = requires(int nmax, const double* x, double* out,
                                                  std::size_t count,
                                                  boys::BoysSortedArgs sorted) {
    boys::BoysAllN(nmax, x, out, count, sorted);
};
template <typename Lane = float>
constexpr bool kFp32AllNDeclaresSorted = requires(int nmax, const Lane* x, Lane* out,
                                                  std::size_t count,
                                                  boys::BoysSortedArgs sorted) {
    boys::BoysAllNF32(nmax, x, out, count, sorted);
};
#if BoysFp16
template <typename Lane = boys::F16>
constexpr bool kHalfAllNDeclaresSorted = requires(int nmax, const Lane* x, Lane* out,
                                                  std::size_t count,
                                                  boys::BoysSortedArgs sorted) {
    boys::BoysAllNF16(nmax, x, out, count, sorted);
};
template <typename Lane = boys::Bf16>
constexpr bool kBf16AllNDeclaresSorted = requires(int nmax, const Lane* x, Lane* out,
                                                  std::size_t count,
                                                  boys::BoysSortedArgs sorted) {
    boys::BoysAllNBf16(nmax, x, out, count, sorted);
};
#else
// The half lane's entries leave the surface with the seam, so there is no declaration
// to resolve and the answer is false without one; the parameter is here so that both
// builds spell the trait the same way.
template <typename Lane = void>
constexpr bool kHalfAllNDeclaresSorted = false;
template <typename Lane = void>
constexpr bool kBf16AllNDeclaresSorted = false;
#endif

// The answer for one class's lane, taken from the declarations above.
constexpr bool LaneAllNDeclaresSorted(OptionPrecision precision) noexcept {
    switch (precision) {
    case OptionPrecision::kFp64:
        return kFp64AllNDeclaresSorted;
    case OptionPrecision::kFp32:
        return kFp32AllNDeclaresSorted<>;
    case OptionPrecision::kFp16:
        return kHalfAllNDeclaresSorted<>;
    case OptionPrecision::kBf16:
        return kBf16AllNDeclaresSorted<>;
    case OptionPrecision::kFp32Device:
        break;
    }

    return false;
}

// One class of the option space: the precision its cells were enumerated for and the
// question they answer. That pair is the class key the report prints a class under,
// and it is what makes one combination of the axes two cells of two classes.
struct SpaceClass {
    OptionPrecision precision = OptionPrecision::kFp64;
    boys::OptionProbeShape shape = boys::OptionProbeShape::kAllOrders;

    bool operator==(const SpaceClass&) const = default;
};

// The classes one book is spread over, in the order the book carries them: a book is
// enumerated class by class, so a class's cells stand together in it.
std::vector<SpaceClass> ClassesOf(std::span<const boys::OptionProbeCell> book) {
    std::vector<SpaceClass> classes;

    for (const boys::OptionProbeCell& cell : book) {
        const SpaceClass klass{cell.precision, cell.shape};

        if (std::find(classes.begin(), classes.end(), klass) == classes.end()) {
            classes.push_back(klass);
        }
    }

    return classes;
}

// One class's share of one book, read in a single pass: the lane its cells came from,
// how many of them there are, the packing-axis members they are instantiated at, how
// many of them are the call that takes the arguments as already sorted, and how many
// this build serves and refuses.
struct ClassCells {
    boys::Precision lane = boys::Precision::kFp64;
    std::size_t count = 0;
    std::size_t sorted = 0;
    std::size_t served = 0;
    std::size_t refused = 0;
    std::vector<boys::PackAxis> packs;
};

ClassCells ReadClass(std::span<const boys::OptionProbeCell> book, const SpaceClass& klass) {
    ClassCells read;

    for (const boys::OptionProbeCell& cell : book) {
        if (cell.precision != klass.precision || cell.shape != klass.shape) {
            continue;
        }

        if (read.count == 0) {
            read.lane = cell.lane;
        }

        ++read.count;

        if (std::find(read.packs.begin(), read.packs.end(), cell.pack) == read.packs.end()) {
            read.packs.push_back(cell.pack);
        }

        if (cell.sorted) {
            ++read.sorted;
        }

        if (cell.served) {
            ++read.served;
        } else {
            ++read.refused;
        }
    }

    return read;
}

// The class as the report spells it, for a failure message: "fp64 all-orders".
std::string ClassLabel(const SpaceClass& klass) {
    return std::string(boys::PrecisionName(klass.precision)) + " " +
           boys::OptionProbeShapeName(klass.shape);
}

// The cells one class carries, read from the library's own axis tables for that class's lane and from the
// class's own entry: the routes the class's lane reports its fits in, each taken once, times the library's
// schemes, times its partitions of the fitted regions, times the packing-axis members the class's entry
// carries, times its division forms and region-B exponentials, times the sorted-arguments member where declared.

// Every factor but the last two is a table asked of the library, so a class enumerated short of a route, a
// scheme, a partition, a form or an exponential is a number here rather than a smaller class that still adds
// up. The packing axis is the one factor a shape's entry can refuse, read from the class's own cells.
std::size_t ClassProduct(const SpaceClass& klass, const ClassCells& read) {
    const bool sortedOverload =
        klass.shape == boys::OptionProbeShape::kAllN && LaneAllNDeclaresSorted(klass.precision);

    return DistinctRoutes(read.lane).size() * boys::BoysEvalSchemes().size() *
           boys::BoysFitGranularities().size() * read.packs.size() *
           boys::BoysDivisionForms().size() * boys::BoysRegionBExps().size() *
           (sortedOverload ? 2 : 1);
}

// One class held to the library's own axes, and the cells those axes admit for it answered back: cells = the axes
// the library reports for the class's lane times the members its own entry admits (ClassProduct); packing members
// are of the library's set, short only of the orders axis a one-order entry refuses; the sorted member is all-N's.
// A class failing any of the three is enumerated against a space that is not the library's - the defect here.
std::size_t CheckClassAgainstTheLibrary(const SpaceClass& klass, const ClassCells& read) {
    const std::span<const boys::PackAxisInfo> axes = boys::BoysPackAxes();
    const std::size_t product = ClassProduct(klass, read);

    EXPECT_EQ(read.count, product)
        << ClassLabel(klass) << " carries " << read.count << " cell(s) where this library's "
        << "axes give the class " << product;

    for (const boys::PackAxis axis : read.packs) {
        const bool reported = std::any_of(
            axes.begin(), axes.end(),
            [axis](const boys::PackAxisInfo& row) { return row.axis == axis; });

        EXPECT_TRUE(reported)
            << ClassLabel(klass) << " carries a packing axis the library does not report";
    }

    EXPECT_LE(read.packs.size(), axes.size())
        << ClassLabel(klass) << " carries more packing axes than the library reports";

    if (read.packs.size() < axes.size()) {
        EXPECT_EQ(read.packs.size() + 1, axes.size())
            << ClassLabel(klass)
            << " is short of more than the one packing axis member an entry refuses";

        EXPECT_TRUE(std::find(read.packs.begin(), read.packs.end(), boys::PackAxis::kArguments) !=
                    read.packs.end())
            << ClassLabel(klass) << " is short of a packing axis, and the member an entry "
            << "refuses is the orders axis";
    }

    const bool sortedOverload =
        klass.shape == boys::OptionProbeShape::kAllN && LaneAllNDeclaresSorted(klass.precision);

    if (sortedOverload) {
        EXPECT_EQ(read.sorted * 2, read.count)
            << ClassLabel(klass) << " declares the sorted-arguments overload and carried "
            << read.sorted << " sorted cell(s) of " << read.count
            << ": the member is both calls of every combination";
    } else {
        EXPECT_EQ(read.sorted, 0u)
            << ClassLabel(klass) << " carries a sorted-argument cell and this class's lane "
            << "declares no overload that takes the tag";
    }

    return product;
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
    // A closed seam must not read as a build that never had the lanes: the two are
    // reported as entries this build does not carry, a fact a caller can act on.
    EXPECT_TRUE(NotCarried(report, "f16-io"));
    EXPECT_TRUE(NotCarried(report, "bf16-io"));
    EXPECT_TRUE(Find(report, "f16-io") == nullptr);
    EXPECT_TRUE(Find(report, "bf16-io") == nullptr);
#endif
}

// The reference bound the report prints is the library's, not a number written
// in the probe: it is what the library's own accessor answers for the anchor cell at
// the anchor's own axes - the cell every ratio is in units of, and not another
// combination's figure standing beside it.
TEST(ProbeTest, TheReferenceBoundIsReadFromTheLibrary) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const OptionProbeMeasurement* anchor = Find(report, report.referenceOption);

    ASSERT_NE(anchor, nullptr) << "the report anchors on a row that is not one of its own";

    const boys::AccuracyFigure figure =
        boys::BoysAccuracyGuaranteed(boys::Precision::kFp64, anchor->route, anchor->scheme,
                                     anchor->pack, anchor->granularity, anchor->division);

    ASSERT_TRUE(figure.available) << figure.reason;
    EXPECT_DOUBLE_EQ(report.referenceBound, figure.value);
    EXPECT_DOUBLE_EQ(report.referenceBound, anchor->bound)
        << "the anchor's own bar is not the figure the report prints as the comparison's floor";
}

// The accuracy column is measured whether or not a cost was: the values a lane
// returns for an argument are a property of the build, and a run too short to form
// a ratio must not blank the column. The timed rounds ran all the same.
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
// them. A column that read zero for every option would pass every bound here.
TEST(ProbeTest, TheAccuracyColumnSeparatesTheLanes) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const OptionProbeMeasurement* fp64 = Find(report, "batch-fp64");
    const OptionProbeMeasurement* fp32 = Find(report, "batch-fp32");

    ASSERT_NE(fp64, nullptr);
    ASSERT_NE(fp32, nullptr);
    EXPECT_GT(fp32->maxError, 0.0);
    EXPECT_LT(fp64->maxError, fp32->bound);
}

// Every option the probe reports is held to its own documented bound, and the fp64 options to the
// certified lane's - each of them states that contract, whether or not it returns the certified bits.
// The batch entries seed region A at the batch's highest order on a vector host, so their F_k below it
// is a recurrence's value: the difference is inside the bound and is reported rather than hidden.
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

// A row is judged against the figure the policy it runs publishes, asserted against the library's own answer and
// never a number written here: the class's own cell is what its call compiles when it names no policy, so bar
// and figure come from one combination, while a bar composed from another names an arithmetic no entry under
// that row runs - at a plain-reciprocal seam a bar at (shipped partition, refined reciprocal) is 1.5e-7 where 2.5e-7 is published.
TEST(ProbeTest, ARowsBarIsTheFigureItsOwnPolicyPublishes) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    // The batch rows: the entry takes no policy, so the bar is the class's own guarantee,
    // each format asking it of its own class. The two half formats add their own term
    // beside the base their lane publishes.
    const double fp32Own =
        boys::DefaultGuarantee<boys::Precision::kFp32, boys::Shape::kAllOrders>().value;

    const std::pair<const char*, double> batchRows[] = {
        {"batch-fp32", fp32Own},
        {"f16-io",
         boys::DefaultGuarantee<boys::Precision::kFp16, boys::Shape::kAllOrders>().value + 0x1p-11},
        {"bf16-io",
         boys::DefaultGuarantee<boys::Precision::kBf16, boys::Shape::kAllOrders>().value + 0x1p-8},
    };

    for (const auto& [name, expected] : batchRows) {
        const OptionProbeMeasurement* row = Find(report, name);

        if (row == nullptr) {
            continue;
        }

        EXPECT_DOUBLE_EQ(row->bound, expected)
            << name << "'s bar is the figure of the policy its call resolves to";
    }

    // The cells: the form is a template argument of the engine's entries, so a cell's bar is
    // the library's figure for the cell's own axes at the cell's own form. The half lanes add
    // their format's term, and the double lane's figure carries no form dimension.
    for (const OptionProbeMeasurement& row : report.measurements) {
        if (row.name == "batch-fp32" || row.name == "f16-io" || row.name == "bf16-io") {
            continue;
        }

        boys::Precision lane = boys::Precision::kFp64;
        double formatTerm = 0.0;

        if (row.precision == boys::OptionPrecision::kFp32) {
            lane = boys::Precision::kFp32;
        } else if (row.precision == boys::OptionPrecision::kFp16) {
            lane = boys::Precision::kFp16;
            formatTerm = 0x1p-11;
        } else if (row.precision == boys::OptionPrecision::kBf16) {
            lane = boys::Precision::kFp16;
            formatTerm = 0x1p-8;
        } else {
            continue;
        }

        const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
            lane, row.route, row.scheme, row.pack, row.granularity, row.division);

        ASSERT_TRUE(figure.available) << row.name << " is measured at a combination the library "
                                                  "answers no figure for";

        EXPECT_DOUBLE_EQ(row.bound, figure.value + formatTerm)
            << row.name << " is judged at another form's figure";
    }
}

// An instrument with no floor does not stop the measurement: the load readings are context and the comparison
// is not made in them, so the run still times its rounds and reports them, losing only the right to print a
// load percentage - and the report says so rather than printing a figure relative to nothing. With one round
// there is no ratio to form, so this run also ends in the refusal that says how many rounds it would need.
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

// The canary alarm is a flag and not a gate: with the alarm below anything repeated fixed work can reach,
// every pass is flagged and every pass is still used, so the run produces the same kind of figures it does
// with the alarm out of the way. A fixed work measured by wall clock would otherwise fail its own admission
// rule on a machine whose clock moves.
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

// The other end of the same knob changes only the flags: with the alarm out of reach nothing is flagged and
// the run is the same run. A pass is placed on one side of the alarm or the other against the floor the
// calibration found, and the second run finds its own floor: on a machine that gave the first run one and
// this one none, the passes carry no reading and none is flagged.
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

// The resolution is measured in the quantities the ordering is made of and is not a bar chosen in advance:
// it is the widest within-round band the certified double lane's classes showed (a rival's band against its
// own class's leader), so it is the coarseness of the very comparisons the report placed or refused to place.
// A run that formed no band reports zero rather than a width it never measured.
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

// A class the run ordered names its own leader and leaves no rival unplaced; a class
// the run could not order still ends in one default, and the options it could not
// place are named beside it, so a reader is never handed a name without the evidence
// that is missing for it. The class key is one precision and one shape.
TEST(ProbeTest, ARecommendationLeavesNoRivalUnplaced) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const OptionProbeMeasurement* leader = ReferenceLeader(report);
    const boys::OptionProbeClass* doubles =
        ClassOf(report, OptionPrecision::kFp64, boys::OptionProbeShape::kAllOrders);

    if (report.verdict == OptionProbeVerdict::kRecommend) {
        EXPECT_FALSE(report.recommended.empty());
        EXPECT_TRUE(report.hasDefault);

        ASSERT_NE(doubles, nullptr);
        ASSERT_NE(leader, nullptr);
        EXPECT_EQ(doubles->leader, leader->name);

        // The invariant the whole report rests on: the default is a row of the class the rule names, and which
        // row is what the way-it-was-reached says. Where the class could not be ordered and the refinement ran,
        // the vote names it; where there was no vote, the name is the one the class's own printed figure puts
        // first. Both rows are printed either way, and where they differ that says the top entries cannot part.
        const boys::OptionProbeRefinement* vote = nullptr;

        for (const boys::OptionProbeRefinement& refinement : report.refinements) {
            if (refinement.precision == OptionPrecision::kFp64 &&
                refinement.shape == boys::OptionProbeShape::kAllOrders) {
                vote = &refinement;
            }
        }

        const bool voted = vote != nullptr && vote->ran && !vote->winner.empty();
        EXPECT_EQ(report.recommended, voted ? vote->winner : leader->name)
            << "the default is not the row the report's own rule names for its class";

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

        // Reached by the refinement's vote, or by a tie this run could not break: the
        // name is one of the class's own options, and the runs taken over the tied set
        // are in the report rather than only their outcome.
        bool member = false;

        for (const OptionProbeMeasurement* candidate : ReferenceMembers(report)) {
            member = member || candidate->name == report.recommended;
        }

        EXPECT_TRUE(member) << report.recommended << " is the default and is not of the class";
        EXPECT_FALSE(report.inseparable.empty())
            << "a tie was decided without naming the pair the class could not place";
        EXPECT_FALSE(report.refinements.empty())
            << "a tie was decided with no refinement run behind the choice";

        // From here the class is tied, and the way the report says the tie was reached
        // has to match the vote's own record: the vote naming this row - unanimously or by
        // a majority - is what kRefined and kVote mean, and a vote that ran and could not
        // settle on one row is a kChosenAmongEquals.
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

// The output states the resolution in the reader's own terms and in the units the
// comparison is made in, so a refusal is a measurement with a number attached rather
// than a shrug. When the run was too short for a band, the text says that instead.
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

// The clock check, read from the run's own rows: wider vector registers draw a lower clock, so two options not
// running one arithmetic can be exposed to the machine differently, and the report says which case it measured.
// It states how far the widest-moving pair's ratio travelled between the run's halves beside the resolution it
// is read against, and whether every compared option ran the same arithmetic route.
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

    // Exactly one of the two, and both name the pair and the resolution the figure was
    // read against: a pair that came in under it, or one that went past it.
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

// A band is a lower and an upper quartile, and two rounds have neither: no pair of the
// class is placed, and the report says so rather than printing a width it never
// measured. The run still ends in one default - the class's own fastest, reached by
// the refinement runs, taken at a longer protocol - and it says which of the two this is.
TEST(ProbeTest, AnOrderingNeedsFourPairedRounds) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    options.refinementFactor = 2;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 2);
    EXPECT_FALSE(report.measurements.empty());
    ASSERT_TRUE(report.measurements.front().measured)
        << "two rounds can form a ratio, just not a quartile band";

    EXPECT_DOUBLE_EQ(report.resolution, 0.0);
    EXPECT_NE(report.reason.find("quartile"), std::string::npos) << report.reason;

    const boys::OptionProbeClass* doubles =
        ClassOf(report, OptionPrecision::kFp64, boys::OptionProbeShape::kAllOrders);
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

    // The default is the row the refinement's vote named: the same question asked again at a longer protocol,
    // where options this run cannot separate are settled by which was fastest in most runs. Both rows are
    // re-measured by the vote, and the class's own fastest figure is the record of what the shorter protocol
    // put first; where the two differ, that says the class's top entries cannot be separated.
    const OptionProbeMeasurement* classLeader = ReferenceLeader(report);
    ASSERT_NE(classLeader, nullptr);
    EXPECT_EQ(report.recommended, stage.winner)
        << "the default is not the row the refinement vote named";
    EXPECT_NE(std::find(stage.pool.begin(), stage.pool.end(), classLeader->name), stage.pool.end())
        << "the class's own fastest row was not one of the rows the vote re-measured";
    EXPECT_EQ(report.defaultHow,
              stage.winner == report.recommended
                  ? (stage.unanimous
                         ? OptionProbeDefaultHow::kRefined
                         : (stage.plurality ? OptionProbeDefaultHow::kVote
                                            : OptionProbeDefaultHow::kChosenAmongEquals))
                  : OptionProbeDefaultHow::kChosenAmongEquals);

    EXPECT_EQ(stage.runs, report.options.refinementRuns);
    // The stage is the same protocol run longer, and the factor is what lengthens it: it
    // scales the ROUNDS and leaves the passes alone, so a run is this many times the
    // protocol and not this many times this many. Both counts are read against the factor
    // by equality, and the protocol's being longer is the rounds' own comparison below.
    EXPECT_EQ(stage.passes, report.options.passes)
        << "the refinement lengthened the passes, which the factor does not do";
    EXPECT_EQ(stage.rounds, report.options.rounds * report.options.refinementFactor);
    EXPECT_EQ(stage.runLeaders.size(), static_cast<std::size_t>(stage.runs));
    EXPECT_GT(stage.rounds, report.pairedRounds)
        << "the re-run was not at a larger protocol than the run that could not order";
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

// The comparison is paired, and this is what that means in the report: every option is called in every round
// the run took, so all rest on the same round count, and the reference lane - the option every ratio is formed
// against - is exactly one against itself in every round and has no drift. A design timing one option in one
// set of rounds and another in another would report a ratio of two figures taken under two clocks.
TEST(ProbeTest, ThePairsAreFormedInsideTheRounds) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    ASSERT_FALSE(report.referenceOption.empty());
    ASSERT_FALSE(report.passes.empty());

    const OptionProbeMeasurement* reference = Find(report, report.referenceOption);
    ASSERT_NE(reference, nullptr);
    ASSERT_TRUE(reference->measured);

    // The anchor is a cell of the double all-orders class chosen for the reading and not the winner of the
    // comparison it anchors, stated as this combination: the shipped partition, on the arguments axis, at the
    // shipped route and scheme, in the build's own division form and exponential - the cell the probe's own
    // anchor has always been, whatever name the class grammar gives it.
    EXPECT_EQ(reference->precision, boys::OptionPrecision::kFp64)
        << "the anchor is no cell of the certified double lane";
    EXPECT_EQ(reference->shape, boys::OptionProbeShape::kAllOrders);
    EXPECT_EQ(reference->route, boys::FitRoute::kChebyshev);
    EXPECT_EQ(reference->scheme, boys::EvalScheme::kSplitClenshaw);
    EXPECT_EQ(reference->pack, boys::PackAxis::kArguments);
    EXPECT_EQ(reference->granularity, boys::FitGranularity::kCoarsest);
    EXPECT_EQ(reference->division, boys::kDefaultDivisionForm);
    EXPECT_EQ(reference->regionBExp, boys::kDefaultHostRegionBExp)
        << "the anchor moved with the seam, and a unit that moves with the answer is no unit";

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

// A run that formed no figure at all ends in the refusal and in nothing else: no name printed, no candidate
// set left for the caller to choose from, no option offered as a default on the strength of a table rather
// than a measurement. The report gives the reason instead - the rounds a band needs - so a consumer who
// cannot afford a longer run is told what the longer run buys.
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

// Where a class cannot be ordered, the rule is: re-run the options it left tied, alone, at a longer protocol;
// take the one that led most of those runs; and where no candidate leads most of them, take one of the leaders
// and say plainly that this happened. The protocol here is two rounds, which can form no band, so the tie is
// certain and the vote decides.
TEST(ProbeTest, ATiedClassIsReRunAloneAndVotedOn) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    options.refinementRuns = 6;
    options.refinementFactor = 2;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_FALSE(report.refinements.empty()) << "a tied class was decided with no runs behind it";

    const boys::OptionProbeRefinement& stage = report.refinements.front();

    EXPECT_EQ(stage.precision, OptionPrecision::kFp64);
    EXPECT_TRUE(stage.ran);
    EXPECT_EQ(stage.runs, 6);
    EXPECT_EQ(stage.runLeaders.size(), 6u) << "a run of the vote placed no leader";
    EXPECT_FALSE(stage.winner.empty()) << "the vote ended with no entry to name";

    // The vote names the default: options a class cannot separate are settled by which was fastest in most
    // runs, so the row the refinement named is what the report recommends, and the class's own fastest figure
    // is the record of what the shorter protocol put first. Both are printed, and where they differ that says
    // the two cannot be separated.
    const OptionProbeMeasurement* classLeader = ReferenceLeader(report);
    ASSERT_NE(classLeader, nullptr);
    EXPECT_EQ(report.recommended, stage.winner)
        << "the default is not the row the vote named";
    EXPECT_EQ(report.defaultHow,
              stage.unanimous
                  ? OptionProbeDefaultHow::kRefined
                  : (stage.plurality ? OptionProbeDefaultHow::kVote
                                     : OptionProbeDefaultHow::kChosenAmongEquals))
        << "the way-it-was-reached does not match what the vote named";

    EXPECT_EQ(stage.passes, report.options.passes)
        << "the refinement lengthened the passes, which the factor does not do";
    EXPECT_EQ(stage.rounds, report.options.rounds * report.options.refinementFactor);
    EXPECT_GT(stage.rounds, report.pairedRounds)
        << "the re-run was not at a longer protocol than the run that could not order";

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

// The refinement factor is a claim about one thing: how much longer a refinement run is than one pass protocol.
// The stage refines a tie by asking whether the leader holds up over more ROUNDS of the same comparison, so the
// factor lengthens the ROUNDS and leaves the passes alone - a run is the factor times the protocol, never the
// factor times the factor. Applying it to both squares it: at the shipped factor the stage became 25 passes' worth of rounds where 5 were asked for.
TEST(ProbeTest, TheRefinementFactorLengthensTheRoundsAndNotThePasses) {
    ProbeOptions options = Timed();
    options.passes = 1;
    options.rounds = 2;
    options.refinementRuns = 2;
    options.refinementFactor = 3;
    const OptionProbeReport report = boys::RunOptionProbe(options);

    ASSERT_EQ(report.pairedRounds, 2) << "the main run is the protocol the stage refines";
    ASSERT_EQ(report.options.refinementFactor, 3);

    // The certified double lane's class: two paired rounds form no band, so this
    // class cannot be ordered and is the one a refinement stage is run over.
    const boys::OptionProbeRefinement* stage = nullptr;

    for (const boys::OptionProbeRefinement& refinement : report.refinements) {
        if (refinement.precision == OptionPrecision::kFp64 &&
            refinement.shape == boys::OptionProbeShape::kAllOrders) {
            stage = &refinement;
        }
    }

    ASSERT_NE(stage, nullptr) << "a class of more than one option was left unrefined";
    EXPECT_TRUE(stage->ran);

    EXPECT_EQ(stage->passes, report.options.passes)
        << "the factor lengthened the passes: it lengthens the rounds of the same comparison "
           "and leaves the passes alone, so a run is the protocol lengthened and not squared";
    EXPECT_EQ(stage->rounds, report.options.rounds * report.options.refinementFactor)
        << "the factor did not lengthen the rounds exactly once";

    EXPECT_EQ(stage->passes * stage->rounds,
              report.pairedRounds * report.options.refinementFactor)
        << "the refinement run is " << stage->passes << " pass(es) by " << stage->rounds
        << " round(s) against a protocol of " << report.pairedRounds
        << " paired round(s): the stage is the factor squared, not the factor";
    EXPECT_GT(stage->rounds, report.options.rounds)
        << "the refinement run is no longer a protocol than the run it refines";
}

// The acceptance rule, stated where it can be checked: a run that measured its class ends with exactly one
// combination, and the class entry, the name and the reason agree on it. Both protocols that measure are read,
// since the rule is about the report's shape and not about how well the machine separated the options.
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
            ClassOf(report, OptionPrecision::kFp64, boys::OptionProbeShape::kAllOrders);
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

// Whatever the machine did, the protocol's bookkeeping adds up: every pass was within the canary's alarm or
// above it and all were used, every figure rests on every paired round, and a figure that exists is a positive
// cost with a band around it. A run whose load instrument never found a floor took no canary, and its confidence
// line says so rather than naming a load of zero.
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

// The verdict and the names it is built from agree: the default is an option of the certified lane's precision
// at the library's own multiplier, answering the workload's own question and never a faster row of another
// precision or shape. A class the run ordered hands over its own leader; a class it could not order hands over
// one of the options it left tied, with the pair it could not place named beside it. A refusal names nothing.
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
        EXPECT_EQ(named->shape, boys::OptionProbeShape::kAllOrders)
            << "the recommendation answers another question than the workload asks";

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
            const OptionProbeMeasurement* fastest = Find(report, report.fastestAtReferenceAccuracy);
            ASSERT_NE(fastest, nullptr);
            EXPECT_TRUE(fastest->measured);
            EXPECT_EQ(fastest->precision, OptionPrecision::kFp64);
            EXPECT_EQ(fastest->shape, boys::OptionProbeShape::kAllOrders);
            EXPECT_LE(fastest->bound, report.referenceBound);
            EXPECT_GE(fastest->nsPerArgument, ReferenceLeader(report)->nsPerArgument);
        }
    } else {
        EXPECT_TRUE(report.recommended.empty());
        EXPECT_FALSE(report.reason.empty());
        EXPECT_EQ(report.confidence.rfind("CANNOT DETERMINE", 0), 0u) << report.confidence;
    }
}

// A class is one precision and one question shape, and that pair is the whole key: nothing inside a class was
// built at another precision or multiplier and nothing in it answers another question; the leader is its fastest
// member; a class that says it is ordered holds no unplaced member and none of one entry; every figure is ranked
// in the class of its own pair and no other. A short run can produce no band, and then nothing in a class of more than one is ordered.
TEST(ProbeTest, EveryClassIsOnePrecisionOneShapeAndRanksOnlyItsOwn) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());

    ASSERT_FALSE(report.classes.empty());

    std::set<std::tuple<int, int>> keys;

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_FALSE(entry.name.empty()) << "a class with no name is one a reader cannot place";
        EXPECT_TRUE(keys
                        .insert({static_cast<int>(entry.precision),
                                 static_cast<int>(entry.shape)})
                        .second)
            << entry.name << " shares its precision and its shape with another class";
        EXPECT_NE(entry.name.find(boys::OptionProbeShapeName(entry.shape)), std::string::npos)
            << entry.name << " does not carry the question shape it is keyed on";

        std::size_t measured = 0;

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.shape == entry.shape) {
                ++measured;
            }
        }

        EXPECT_EQ(entry.ranked.size(), measured) << entry.name << " does not rank every member";

        if (entry.leader.empty()) {
            EXPECT_TRUE(entry.ranked.empty()) << entry.name;
            EXPECT_FALSE(entry.ordered) << entry.name << " has no leader to be ordered around";
            EXPECT_FALSE(entry.note.empty()) << entry.name;
            EXPECT_EQ(measured, 0u) << entry.name << " reports no figure yet options were "
                                                         "measured in its precision and shape";
            continue;
        }

        ASSERT_FALSE(entry.ranked.empty()) << entry.name;

        const OptionProbeMeasurement* leader = Find(report, entry.leader);
        ASSERT_NE(leader, nullptr) << entry.name;
        EXPECT_TRUE(leader->measured) << entry.name << " names a leader that was never measured";
        EXPECT_EQ(leader->precision, entry.precision) << entry.name;
        EXPECT_EQ(leader->shape, entry.shape)
            << entry.name << " names a leader that answers another question";
        EXPECT_DOUBLE_EQ(entry.leaderNsPerArgument, leader->nsPerArgument) << entry.name;
        EXPECT_EQ(entry.ranked.front(), entry.leader)
            << entry.name << " ranks its leader somewhere behind its own first row";

        for (const OptionProbeMeasurement& measurement : report.measurements) {
            if (measurement.measured && measurement.precision == entry.precision &&
                measurement.shape == entry.shape) {
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

        // A band over the lower and upper quartiles of the paired ratios needs four
        // rounds; below that a class is measured and not ordered.
        if (report.pairedRounds < 4) {
            EXPECT_FALSE(entry.ordered) << entry.name
                                        << " says it is ordered on too few rounds for a band";
        }

        for (const std::string& name : entry.ranked) {
            const OptionProbeMeasurement* member = Find(report, name);
            ASSERT_NE(member, nullptr) << entry.name;
            EXPECT_EQ(member->precision, entry.precision)
                << name << " is ranked in " << entry.name << " but is another precision";
            EXPECT_EQ(member->shape, entry.shape)
                << name << " is ranked in " << entry.name << " but answers another question";

            if (entry.ordered && name != entry.leader) {
                EXPECT_GT(member->nsPerArgument, leader->nsPerArgument)
                    << entry.name << " says it is ordered, yet " << name
                    << " is not behind its leader in the run's own statistic";
            }
        }
    }

    // Every figure the run produced is ranked in the class of its own precision and
    // shape, and in no other class.
    for (const OptionProbeMeasurement& measurement : report.measurements) {
        if (!measurement.measured) {
            continue;
        }

        const boys::OptionProbeClass* entry =
            ClassOf(report, measurement.precision, measurement.shape);
        ASSERT_NE(entry, nullptr) << measurement.name
                                  << " was measured in a precision and shape the report "
                                     "classes nowhere";
        EXPECT_NE(std::find(entry->ranked.begin(), entry->ranked.end(), measurement.name),
                  entry->ranked.end())
            << measurement.name << " is measured in " << entry->name << " and is not ranked in it";

        for (const boys::OptionProbeClass& other : report.classes) {
            if (other.precision == measurement.precision &&
                other.shape == measurement.shape) {
                continue;
            }

            EXPECT_EQ(std::find(other.ranked.begin(), other.ranked.end(), measurement.name),
                      other.ranked.end())
                << measurement.name << " is ranked in " << other.name << ", another class";
        }
    }
}

// The report says in its own words what a class is and what its key is - one precision and one question shape,
// decided at the library's own full-accuracy multiplier and by what an option hands back, never by comparing
// one lane's documented figure against another's - and it says what the winner of a class is a claim about.
// Every class the run made carries its key in its name.
TEST(ProbeTest, TheReportSaysAClassIsOnePrecisionAndOneShape) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("the accuracy classes"), std::string::npos);
    EXPECT_NE(text.find("one precision and one question shape"), std::string::npos);
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
    // question, this machine.
    EXPECT_NE(text.find("is its fastest option at"), std::string::npos);
    EXPECT_NE(text.find("for that question, on this machine and no more"), std::string::npos);

    // The axes that only change how one answer is computed are named as columns
    // of a class rather than as part of its key.
    EXPECT_NE(text.find("a column inside the class, so those options compete in one ranking"),
              std::string::npos);

    for (const boys::OptionProbeClass& entry : report.classes) {
        EXPECT_NE(text.find(entry.name), std::string::npos) << entry.name;
        EXPECT_EQ(entry.name.find("m="), std::string::npos)
            << entry.name << " carries a multiplier the class is not keyed on";
        EXPECT_NE(entry.name.find(boys::OptionProbeShapeName(entry.shape)), std::string::npos)
            << entry.name << " does not carry the question shape it is keyed on";
    }
}

// The same seed and the same options are the same workload: the values, and so the
// accuracy column, come back identical.
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

// The output is a report a reader can act on with nothing else at hand: it names the
// machine's arithmetic, the protocol, the load instrument, the accuracy comparison's
// floor, and the fact that the result describes this machine.
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

// A name that is no option of this library is reported rather than quietly measuring
// nothing, because an empty report otherwise reads as a machine where nothing is fast.
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

// The space the probe accounts for is the library's own product, not a list written in
// the probe: one cell per combination of the axes the library reports, every
// combination present, no two cells sharing a name, and a reason on every cell this
// build does not serve.
TEST(ProbeTest, TheOptionSpaceIsTheLibrarysOwnProduct) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    ASSERT_FALSE(report.cells.empty());
    EXPECT_FALSE(report.granularities.empty());

    // The multiplier is the library's full-accuracy setting and not an axis of
    // the space: a class is the product of the axes the library reports and no
    // factor stands for a choice the library no longer offers.

    // The space this book accounts for, class by class. A class is a precision and a question shape and not a
    // lane: the shapes do not admit the same axes and the two half formats are two classes. Every class is held
    // to the product of the axes the library reports for its own lane, read from the library's tables and the
    // class's own cells, so a class enumerated against a space that is not the library's fails at that class.
    const std::vector<SpaceClass> classes = ClassesOf(report.cells);

    std::size_t expected = 0;

    for (const SpaceClass& klass : classes) {
        const ClassCells read = ReadClass(report.cells, klass);

        // Every cell of one class was enumerated from one lane, which is the
        // library's answer and not the test's: the two half formats are one lane
        // and two classes.
        for (const boys::OptionProbeCell& cell : report.cells) {
            if (cell.precision == klass.precision && cell.shape == klass.shape) {
                EXPECT_EQ(cell.lane, read.lane)
                    << cell.name << " was enumerated from another lane than its own class's";
            }
        }

        expected += CheckClassAgainstTheLibrary(klass, read);
    }

    EXPECT_EQ(report.cells.size(), expected)
        << "the book is not the space the library's axes admit: the total is the sum of the "
        << "cells every class of it can be instantiated at";

    // The book carries the classes the report names and no others: a class the report
    // offers a figure for and enumerates no cells for is a class the space is short of,
    // and a class of cells the report does not name is one nothing accounts for.
    EXPECT_EQ(classes.size(), report.classes.size())
        << "the book's classes are not the classes the report names";

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
            EXPECT_EQ(measurement->regionBExp, cell.regionBExp) << cell.name;
            continue;
        }

        EXPECT_FALSE(cell.reason.empty())
            << cell.name << " is not served and the report gives no reason, which is the "
                           "unstated omission the coverage exists to prevent";
        EXPECT_EQ(measurement, nullptr) << cell.name << " is refused yet was measured";
    }
}

// The division form is an axis of the option space like the others: the members are read from the library
// rather than written out in the probe, every combination of the other axes is enumerated at each of them, and
// a cell of a non-default form carries the library's own name for it. The members are ranked against each other
// inside one class rather than in a class apiece: a member the instrument never varies reads exactly like one the library does not have.

// The protocol is the measuring one: a class is made where a run produced figures, and one round carries none.
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

    // The double lane's own cell, at each form: the cell that departs from that class's row on this axis and
    // no other. The row's own form carries the class's own name - the name with no segment, because the row is
    // the combination the class's call compiles - and the other two carry the library's own spelling of the
    // member beside it, so the three names differ.
    std::set<std::string> names;

    for (const boys::DivisionFormInfo& form : forms)
    {
        const std::string name = form.form == DoubleAllOrders::kDivision
                                     ? std::string("batch-fp64")
                                     : std::string("batch-") + form.name + "-fp64";

        EXPECT_TRUE(names.insert(name).second) << name << " names two different cells";

        const OptionProbeMeasurement* row = Find(report, name);

        ASSERT_NE(row, nullptr) << name << " is not an option this run measured";
        EXPECT_EQ(row->division, form.form) << name << " is measured at a form it does not name";
        EXPECT_EQ(row->granularity, DoubleAllOrders::kGranularity)
            << name << " is not the row's cell: its partition is not the one the row reads";
    }

    // Ranked against each other, not in three classes of their own: this is the
    // class the default is chosen from, and a row of each form is in it.
    const boys::OptionProbeClass* certified =
        ClassOf(report, boys::OptionPrecision::kFp64, boys::OptionProbeShape::kAllOrders);

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

// The region-B exponential is an axis of the option space like the others: the members are read from the
// library rather than written out in the probe, every combination of the other axes is enumerated at each of
// them, and a cell of the member that is not the host default carries the library's own name for it. The
// members are ranked against each other inside one class rather than in a class apiece.

// The defect this pins is the one this axis was found by: a member that exists on one target and not the other
// reads exactly like a member the library does not have, and a member the instrument carries but never varies
// reads like one that is not there. The last check cannot pass by construction: the two members are two
// arithmetics, so at an argument where they must part they are compared as values and not only as names.
TEST(ProbeTest, TheRegionBExpAxisIsEnumeratedAndRanked) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::span<const boys::RegionBExpInfo> exps = boys::BoysRegionBExps();

    ASSERT_EQ(exps.size(), 2u) << "this build reports a region-B exponential axis of another size";

    for (std::size_t i = 0; i < exps.size(); ++i) {
        EXPECT_EQ(static_cast<std::size_t>(exps[i].exp), i) << "the rows are not in enumerator order";
        EXPECT_STREQ(exps[i].name, boys::RegionBExpName(exps[i].exp));
    }

    // The double lane's own cell, at each member: the cell that departs from that class's row on this axis and
    // no other. The row's own member carries the class's own name - the name with no segment, because the row
    // is the combination the class's call compiles - and the other carries the library's own spelling of the
    // member beside it, so the two names differ.
    std::set<std::string> names;

    for (const boys::RegionBExpInfo& exp : exps) {
        const std::string name = exp.exp == DoubleAllOrders::kRegionBExp
                                     ? std::string("batch-fp64")
                                     : std::string("batch-") + exp.name + "-fp64";

        EXPECT_TRUE(names.insert(name).second) << name << " names two different cells";

        const OptionProbeMeasurement* row = Find(report, name);

        ASSERT_NE(row, nullptr) << name << " is not an option this run measured";
        EXPECT_EQ(row->regionBExp, exp.exp) << name << " is measured at a member it does not name";
        EXPECT_EQ(row->granularity, DoubleAllOrders::kGranularity)
            << name << " is not the row's cell: its partition is not the one the row reads";
    }

    // Ranked against each other, not in two classes of their own: this is the class the
    // default is chosen from, and a row of each member is in it.
    const boys::OptionProbeClass* certified =
        ClassOf(report, boys::OptionPrecision::kFp64, boys::OptionProbeShape::kAllOrders);

    ASSERT_NE(certified, nullptr) << "the certified double lane's class is not in the report";

    std::set<boys::RegionBExp> ranked;

    for (const std::string& name : certified->ranked) {
        const OptionProbeMeasurement* row = Find(report, name);

        ASSERT_NE(row, nullptr) << name << " is ranked by a class yet not measured";
        ranked.insert(row->regionBExp);
    }

    EXPECT_EQ(ranked.size(), exps.size())
        << "the class the default is chosen from ranks fewer region-B exponentials than the "
           "library reports, so the axis is carried and not compared";

    // The two members are two arithmetics and not two spellings of one. The argument is
    // inside region B and above the cut the fast member's reduced-argument polynomial takes
    // over at, which is the one stretch of the domain where the two must part: below the cut
    // the fast member calls the same library routine the accurate member does.
    const double x = 0.5 * (boys::detail::kRegionBExpCheapFrom + boys::detail::kX1);
    const std::uint64_t fast =
        std::bit_cast<std::uint64_t>(boys::detail::RegionBHalfExp<boys::RegionBExp::kFast>(x));
    const std::uint64_t accurate =
        std::bit_cast<std::uint64_t>(boys::detail::RegionBHalfExp<boys::RegionBExp::kAccurate>(x));

    EXPECT_NE(fast, accurate)
        << "the two members of the axis return one value at x = " << x
        << ": one instantiation would satisfy both rows, and the axis would be carried and "
           "never varied";
}

// A partition the library serves is a cell of the space this probe enumerates, measured and reported under its
// own name; the cells the library refuses are refused with the library's reason rather than by the probe. The
// uniform partition is the axis's third member and the one this test is about: a probe reading its space from
// the library and answering this value with the shipped partition's tables would report a uniform row under a name it does not have.
TEST(ProbeTest, TheUniformPartitionIsEnumeratedAndMeasured) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const boys::FitGranularityInfo* uniform = nullptr;

    for (const boys::FitGranularityInfo& partition : report.granularities) {
        if (partition.granularity == boys::FitGranularity::kUniform) {
            uniform = &partition;
        }
    }

    ASSERT_NE(uniform, nullptr) << "this build's partition table carries no uniform partition";

    // The double lane's cell at the uniform partition, at each scheme this build carries:
    // the cell that departs from that class's row on the partition axis alone and, one
    // scheme over, on the scheme axis too.
    for (const boys::EvalSchemeInfo& scheme : boys::BoysEvalSchemes()) {
        const std::string name = DoubleCellName(boys::FitGranularity::kUniform, scheme.scheme);
        const OptionProbeMeasurement* measured = Find(report, name);

        ASSERT_NE(measured, nullptr) << name << " is a cell of the option space the library "
                                                 "serves and the probe measured nothing for it";
        EXPECT_EQ(measured->granularity, boys::FitGranularity::kUniform) << name;

        // The row's own figures, which are the grid's and not an entry's: the
        // option's whole-domain figure is the lane's and is a different promise.
        EXPECT_DOUBLE_EQ(measured->ownBound, uniform->bound) << name;
        EXPECT_DOUBLE_EQ(measured->ownLo, uniform->lo) << name;
        EXPECT_DOUBLE_EQ(measured->ownHi, uniform->hi) << name;
        EXPECT_GT(measured->ownBound, 0.0)
            << name << " carries no figure for the partition's own tables";
        EXPECT_LT(measured->ownBound, measured->bound)
            << name << " carries the partition's own figure as one covering the whole line";
    }

    // And the cells of that partition this build refuses are refused with the
    // library's own sentence: a cell answered "outside the enumeration" would be
    // the probe saying the partition is not one of the options, which is the
    // report this test exists to keep from coming back.
    std::size_t refusedUniform = 0;

    for (const boys::OptionProbeCell& cell : report.cells) {
        if (cell.granularity != boys::FitGranularity::kUniform || cell.served) {
            continue;
        }

        ++refusedUniform;
        EXPECT_EQ(cell.reason.find("outside the enumeration"), std::string::npos)
            << cell.name << " is refused as a value outside the enumeration, which is what this "
                            "partition was before it was a row of it; the reason is: "
            << cell.reason;
    }

    if (refusedUniform == 0u) {
        GTEST_SKIP() << "every uniform cell of this build's space is served, so the partition's "
                        "refusals are not exercised: the member the grid did not carry is derived "
                        "and read, and nothing of this partition is refused";
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

// A narrower lane's row is held to the bound its own lane documents, read from the library, and its error is
// reported against the certified lane's floor when the two partitions differ by more than the row's own bound
// allows - the honest third state, not a pass bought by comparing two partitions. Naming a partition changes the
// tables an entry reads and not the entry's own documented accuracy, so the row is judged by the entry's whole-domain figure.
TEST(ProbeTest, APartitionRowIsJudgedByTheEntrysWholeDomainFigure) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());

    const boys::FitGranularityInfo* narrow = nullptr;
    for (const boys::FitGranularityInfo& partition : report.granularities) {
        if (partition.granularity == boys::FitGranularity::kNarrow) {
            narrow = &partition;
        }
    }

    ASSERT_NE(narrow, nullptr) << "this build's partition table carries no narrow partition";

    // The two cells this test compares are the double lane's all-orders class with the
    // partition axis moved and every other axis left at that class's own row, so the
    // comparison is between partitions of one arithmetic and of one entry's accuracy,
    // and the cells are named the way the class's row names them.
    const OptionProbeMeasurement* measured =
        Find(report, DoubleCellName(boys::FitGranularity::kNarrow, DoubleAllOrders::kScheme));
    const OptionProbeMeasurement* shipped =
        Find(report, DoubleCellName(boys::FitGranularity::kCoarsest, DoubleAllOrders::kScheme));
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
        << measured->name << " delivered " << measured->maxError << " against the entry's figure of "
        << measured->bound;
}

// The figure a row is judged by must hold over the arguments the row was measured on,
// and the workload range is the caller's to choose, so the verdict and the bound behind
// it must read the same over a range inside the fitted interval and over the default one
// that runs past it.
TEST(ProbeTest, TheVerdictDoesNotMoveWithTheWorkloadRange) {
    ProbeOptions whole = OneRound();

    ProbeOptions fitted = OneRound();
    fitted.xLo = 1e-3;
    fitted.xHi = 11.8; // inside the fitted tables' interval and below region B

    const OptionProbeReport wholeReport = boys::RunOptionProbe(whole);
    const OptionProbeReport fittedReport = boys::RunOptionProbe(fitted);

    // Both schemes this build carries, at the narrow partition, every other axis at the
    // double lane's own row: the cells are named the way that row names them.
    for (const boys::EvalSchemeInfo& scheme : boys::BoysEvalSchemes()) {
        const std::string name = DoubleCellName(boys::FitGranularity::kNarrow, scheme.scheme);
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

// A partition's own figure covers a narrower domain than the cells a row is measured on,
// so the row says which figure it was judged at and which figure is the partition's,
// with the interval the second holds on.
TEST(ProbeTest, APartitionRowPrintsItsOwnFiguresInterval) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const std::string text = boys::FormatOptionProbe(report);

    const OptionProbeMeasurement* measured =
        Find(report, DoubleCellName(boys::FitGranularity::kNarrow, DoubleAllOrders::kScheme));
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

// What the probe does not measure is stated in its own output, with the counts: the cells
// the library refuses and the lanes the design leaves out.
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

    // The paragraph's debt is counted rather than described: it cites the closure
    // below, which carries the rows this report holds that are no cell of the space.
    EXPECT_NE(text.find("counted in the closure below"), std::string::npos);
}

// The option space is one space and not two: the cells of every class this build carries - the question shapes
// of the four precision lanes this machine measures and the device lane's book, which it cannot run - are counted
// against the product of the axes the library reports, every cell is in exactly one state, and the report's
// last line prints that arithmetic's verdict. The defect pinned is the two halves never having been reconciled.
TEST(ProbeTest, TheClosurePutsEveryCellOfTheSpaceInOneState) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const boys::OptionProbeClosure closure = boys::OptionProbeSpaceClosure(report);

    // The precision lanes the space is spread over, read off the two books rather than
    // written down here: the lanes this machine measures, and the device lane's beside
    // them. A lane is not a class: the classes are keyed on the lane and the question
    // shape together, and they are counted below.
    std::vector<OptionPrecision> lanes;

    for (const boys::OptionProbeCell& cell : report.cells) {
        if (std::find(lanes.begin(), lanes.end(), cell.precision) == lanes.end()) {
            lanes.push_back(cell.precision);
        }
    }

    for (const boys::OptionProbeCell& cell : report.deviceCells) {
        if (std::find(lanes.begin(), lanes.end(), cell.precision) == lanes.end()) {
            lanes.push_back(cell.precision);
        }
    }

    ASSERT_EQ(lanes.size(), 5u)
        << "the space is the four precision lanes this machine measures and the device lane's "
           "book";

    // The class list the space is spread over is keyed on the two parts of a class and not on
    // the precision alone: one combination of the six run-time axes is a cell of every class
    // whose entries carry it, and the entries of one shape are not the entries of another. The
    // report's own list is that reading; the device lane's book is the one class beside it.
    EXPECT_EQ(closure.classes, report.classes.size() + 1)
        << "the space is spread over the classes the report names, and the device lane's book "
           "beside them";

    // The space class by class, each held to the product of the axes the library reports for its own lane: the
    // classes are the ones both books carry, so the device lane's book is held to the library's axes exactly as
    // the measured classes are. The packing axis is the one factor a shape can be short of, and its own cells
    // state it: a one-order shape evaluates one order, so the orders axis is not an axis on it, leaving that class at half its lane's product.
    std::size_t expected = 0;
    std::size_t refusedCells = 0;
    std::size_t deviceServed = 0;
    std::size_t deviceRefused = 0;

    const std::vector<boys::OptionProbeCell>* books[] = {&report.cells, &report.deviceCells};

    for (const std::vector<boys::OptionProbeCell>* book : books) {
        const bool device = book == &report.deviceCells;

        for (const SpaceClass& klass : ClassesOf(*book)) {
            const ClassCells read = ReadClass(*book, klass);

            expected += CheckClassAgainstTheLibrary(klass, read);

            if (device) {
                deviceServed += read.served;
                deviceRefused += read.refused;
            } else {
                refusedCells += read.refused;
            }
        }
    }

    EXPECT_EQ(closure.admitted, expected)
        << "the space's total is not the product of the axes the library reports";
    EXPECT_EQ(closure.walked, expected);
    EXPECT_EQ(closure.enumerated, report.cells.size() + report.deviceCells.size());
    EXPECT_EQ(closure.total, expected);
    EXPECT_EQ(closure.states, closure.total);
    EXPECT_EQ(closure.unaccounted, 0u);
    EXPECT_EQ(closure.states + closure.unaccounted, closure.total);
    EXPECT_TRUE(closure.closed);

    // Which state each class's cells are in, counted from the books themselves: a cell the
    // library refuses is refused wherever it stands, a served cell of the device lane is
    // counted apart and not against this build, and the rest are the run's own.
    EXPECT_EQ(closure.refused, refusedCells + deviceRefused);
    EXPECT_EQ(closure.deviceNotRun, deviceServed);
    EXPECT_EQ(closure.measured + closure.offeredNoFigure + closure.notAsked + closure.unoffered +
                  closure.notCarried,
              report.cells.size() - refusedCells);

    // The one-round run forms no ratio at all, so every served cell of a class this machine
    // measures was offered a row and produced no figure - and that is a state, not a hole.
    EXPECT_EQ(closure.measured, 0u);
    EXPECT_EQ(closure.offeredNoFigure, closure.rowsOwed);
    EXPECT_EQ(closure.notAsked, 0u);

    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("the closure — the space above counted"), std::string::npos) << text;
    EXPECT_NE(text.find("the arithmetic: 0 + "), std::string::npos) << text;
    EXPECT_NE(
        text.find("the space's own total: " + std::to_string(expected) + " cell(s)"),
        std::string::npos)
        << text;
    EXPECT_NE(text.find("the verdict: PASS"), std::string::npos) << text;
    EXPECT_EQ(text.find("the verdict: FAIL"), std::string::npos) << text;

    // The whole space's total, printed as one number beside the axis product that generates it: the sum the
    // two halves of the report never carried. The line is printed over the classes the closure counted - the
    // report's own class list and the device lane's book beside it - so the count it names is that one and not
    // the lanes the space is spread over.
    EXPECT_NE(text.find("the axes' own product over the " + std::to_string(closure.classes) +
                        " class(es): "),
              std::string::npos)
        << text;
    EXPECT_NE(text.find("= " + std::to_string(expected) + " cell(s)"), std::string::npos) << text;
}

// A closure that cannot fail is a decoration. A run that reached the space and carried no
// row for it leaves every cell this build serves in no state, the arithmetic says which,
// the verdict fails, and the report's own text carries the count - so a caller reading the
// output sees a defect and not an absence.
TEST(ProbeTest, AClosureOverASpaceTheRunNeverReachedFails) {
    const OptionProbeReport measured = boys::RunOptionProbe(OneRound());
    const boys::OptionProbeClosure closed = boys::OptionProbeSpaceClosure(measured);

    ASSERT_TRUE(closed.closed);
    ASSERT_GT(closed.rowsOwed, 0u);

    // The same report with the space it was taken over and no rows at all: the run built
    // its books and never reached the grid, which is the shape a broken run has.
    OptionProbeReport report = measured;
    report.measurements.clear();

    const boys::OptionProbeClosure closure = boys::OptionProbeSpaceClosure(report);

    EXPECT_GT(closure.unaccounted, 0u);
    EXPECT_EQ(closure.unaccounted, closed.rowsOwed)
        << "the cells in no state are the places the space owed this run and did not get";
    EXPECT_EQ(closure.states + closure.unaccounted, closure.total);
    EXPECT_FALSE(closure.closed);

    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("the verdict: FAIL"), std::string::npos) << text;
    EXPECT_EQ(text.find("the verdict: PASS"), std::string::npos) << text;
    EXPECT_NE(text.find(std::to_string(closure.unaccounted) + " cell(s) are in no state above"),
              std::string::npos)
        << text;
}

// The run's own table is closed with the space it was taken over: every row it carries is a place the space
// owes this request. The probe once measured the call shapes the axes were not crossed with at their own
// default policy alone - one row per shape and per lane, the all-N grouping and its sorted-argument overload
// among them - and reported the cells their crossing would add as outstanding work. That count is now zero, read off the report's own table rather than taken from the closure.
TEST(ProbeTest, EveryRowTheRunCarriesIsACellOfTheSpace) {
    const OptionProbeReport report = boys::RunOptionProbe(OneRound());
    const boys::OptionProbeClosure closure = boys::OptionProbeSpaceClosure(report);

    ASSERT_GT(report.measurements.size(), 0u);

    EXPECT_EQ(closure.rows, report.measurements.size());
    EXPECT_EQ(closure.rows, closure.rowsOwed + closure.shapesNotCrossed);
    EXPECT_TRUE(closure.closed);

    for (const OptionProbeMeasurement& row : report.measurements) {
        bool cell = false;

        for (const boys::OptionProbeCell& candidate : report.cells) {
            cell = cell || (candidate.name == row.name);
        }

        for (const boys::OptionProbeCell& candidate : report.deviceCells) {
            cell = cell || (candidate.name == row.name);
        }

        EXPECT_TRUE(cell) << row.name
                          << " is a row of this report and no cell of the space: every call "
                             "shape is a class of it and every row of one a cell";
    }

    EXPECT_EQ(closure.shapesNotCrossed, 0u)
        << "a row that is no cell of the space is a shape the axes were not crossed with";
    EXPECT_EQ(closure.crossedOwed, 0u)
        << "there is no crossing left owed: every shape is enumerated at its own axes";

    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_EQ(text.find("the call shapes the axes are not crossed with"), std::string::npos)
        << "the report named a debt the space no longer has";
}

// A request that named a set is closed with the rest of the space stated: the cells no name
// was given for are counted as not asked for rather than as cells nothing accounts for, and
// the run's own table is held to the places the request owes.
TEST(ProbeTest, ARequestForOneCellIsClosedWithTheRestOfTheSpaceStated) {
    const OptionProbeReport whole = boys::RunOptionProbe(OneRound());
    const boys::OptionProbeClosure closed = boys::OptionProbeSpaceClosure(whole);

    ASSERT_TRUE(closed.closed);

    std::string named;

    for (const boys::OptionProbeCell& cell : whole.cells) {
        if (cell.served) {
            named = cell.name;
            break;
        }
    }

    ASSERT_FALSE(named.empty());

    // The set is named to the run and not set on its report afterwards: the request is what the run builds its
    // table from, so a report whose `only` were written after the fact would carry the whole space's rows under
    // a request for one cell - a report the closure refuses to close, because its own table and its own request
    // would be two different runs.
    ProbeOptions narrowed = OneRound();
    narrowed.only = {named};

    const OptionProbeReport report = boys::RunOptionProbe(narrowed);
    const boys::OptionProbeClosure closure = boys::OptionProbeSpaceClosure(report);

    EXPECT_GT(closure.notAsked, 0u);
    EXPECT_EQ(closure.unaccounted, 0u);
    EXPECT_EQ(closure.states, closure.total);
    EXPECT_TRUE(closure.closed);

    const std::string text = boys::FormatOptionProbe(report);

    EXPECT_NE(text.find("not asked for by this run's request"), std::string::npos) << text;
    EXPECT_NE(text.find("the verdict: PASS"), std::string::npos) << text;
}

// The seam a run writes is a replacement for the seam it read, not a report about one: every class the seam in
// force carries has a row, the five names are present, and the marker the committed file defines and a
// replacement must not is absent. The seam's own list is read here the way the probe reads it and guarded the
// way the probe guards it, because a replacement may carry no list at all (include/boys/boys_build_defaults.hpp).
TEST(ProbeTest, TheEmittedSeamIsAReplacementForTheSeamItRead) {
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatBuildDefaults(report, "a test run");

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    // This build's seam carries a class list, so a run of it has a class to write a row for.
    ASSERT_FALSE(text.empty()) << "a run that measured a class writes a seam";

    // The seven names a replacement must carry - the host's five and the device lane's two -
    // the list macro, and the marker it must not: the committed file defines
    // BOYS_BUILD_DEFAULTS_COMMITTED and a replacement does not, so a build pointed at this file
    // says which of the two it read.
    EXPECT_NE(text.find("#pragma once"), std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_FIT_ROUTE FitRoute::"), std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_EVAL_SCHEME EvalScheme::"), std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_PACK_AXIS PackAxis::"), std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_DIVISION_FORM DivisionForm::"),
              std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_FIT_GRANULARITY FitGranularity::"),
              std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_ROWS(X)\\\n"), std::string::npos);

    // The device lane's own two names travel with the five: a replacement is the whole seam and not the host
    // half of one, so a file that left them out would leave a build resolving an unnamed device call through
    // whatever the file said about the host - and `boys/accuracy.hpp` reads both names, so it would not compile
    // at all. The five above are the file's own choices; these two are the device lane's, read from the seam replaced.
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM DivisionForm::"),
              std::string::npos);
    EXPECT_NE(text.find("#define BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP RegionBExp::"),
              std::string::npos);

    // The marker the committed file defines and a replacement must not: the file may name it
    // in the sentence that says which of the two it is, and must not define it.
    EXPECT_EQ(text.find("#define BOYS_BUILD_DEFAULTS_COMMITTED"), std::string::npos);

    // The classes the seam carries, read from the seam's own list - the device cell included, because a class is
    // a (device, precision, shape) triple and the row the writer emits has to be the row for the class it read. A
    // host cell written for a device class is a row for a different class, and a table whose only row for
    // kFp64Device all-orders is keyed kHost is exactly the shape this has to be checked against.
    struct SeamClassKey {
        const char* device;
        const char* precision;
        const char* shape;
    };
#define BOYS_PROBE_TEST_SEAM_CLASS(device, precision, shape, ...) {#device, #precision, #shape},
    const SeamClassKey classes[] = {BOYS_BUILD_DEFAULT_ROWS(BOYS_PROBE_TEST_SEAM_CLASS)};
#undef BOYS_PROBE_TEST_SEAM_CLASS

    ASSERT_GT(std::size(classes), 0u)
        << "the seam in force carries a class list with no class in it";

    for (const SeamClassKey& klass : classes) {
        const std::string row =
            std::string("X(") + klass.device + ", " + klass.precision + ", " + klass.shape + ", ";

        EXPECT_NE(text.find(row), std::string::npos)
            << row << " is a class the seam carries and the emitted file does not";
    }

    // A measured row is marked as one, and the rows this run ranked no cell of are marked as
    // the choices they are: the two are different claims and the seam's header asks for them
    // to be written differently.
    EXPECT_NE(text.find("/* measured:"), std::string::npos);
    EXPECT_NE(text.find("/* a choice, not a measurement:"), std::string::npos);
#else
    // This build's seam carries no class list: the seven names are the whole of it and include/boys/boys.hpp
    // writes the table they make; no class exists for the writer to key a row to, the rows being one per class
    // the seam names (src/boys_probe.cpp, SeamRows over SeamClasses), so a run of this build writes no seam at
    // all. What it may not write is a file defining the list macro with nothing under it, since boys.hpp expands that list into the whole default-policy table.
    EXPECT_TRUE(text.empty()) << "a build whose seam carries no class list wrote a seam";
#endif

    // A run that ranked no class writes no file: there is no measurement in it, and a table
    // of fallback rows written from a run that measured nothing is the transcription this
    // path exists to replace.
    ProbeOptions nothing;
    nothing.count = 256;
    nothing.nmax = 8;
    nothing.passes = 0;
    nothing.rounds = 0;

    const OptionProbeReport rankedNothing = boys::RunOptionProbe(nothing);

    EXPECT_TRUE(boys::FormatBuildDefaults(rankedNothing, "a test run").empty())
        << "a run that measured no class wrote a seam";
}

// The seam's own key is the whole of what the probe accounts for: every host class the seam can name has a
// line in the block that writes the rows, whether this build's library carries an entry of that shape on that
// lane or not. The defect pinned is a class dropped from the account: a class the probe carries nothing for is
// one whose absence a reader of the block cannot see, and the block is read as an account of the seam it replaces.

// The key is the three host lanes by the five shapes the seam states, and it reaches those classes for every
// shape of seam: one carrying a `BOYS_BUILD_DEFAULT_ROWS` list, whose rows the block accounts for by name, and
// one carrying no list at all, where the file is the five names and no rows and every class of the key has no
// row. That second shape is a replacement seam's own (boys/boys_build_defaults.hpp states it), and reporting it as a run that measured nothing would name the wrong cause.
TEST(ProbeTest, TheDefaultsBlockNamesEveryClassTheSeamKeys) {
    // The protocol that ranks: a run with too few paired rounds to order anything writes no
    // rows at all - its block says so in its own words - and this test is about the block a
    // ranking run writes.
    const OptionProbeReport report = boys::RunOptionProbe(Timed());
    const std::string text = boys::FormatOptionProbe(report);

    // The block that writes the rows, and not the class list above it: the two spell a class
    // differently, and it is this one that is an account of the seam.
    const std::size_t at = text.find("the build-defaults seam this run implies");

    ASSERT_NE(at, std::string::npos) << text;

    // A block that accounted for no class carries no class line to check against. Both a block
    // that ranked none of the seam's classes and one whose seam carries no class list say so in
    // prose instead, and either leaves the check below with nothing to be complete against.
    ASSERT_NE(text.find("\n    ", at), std::string::npos)
        << "the block carries no per-class line, so there is no list here to be complete";

#if defined(BOYS_BUILD_DEFAULT_ROWS)
    // This seam carries a class list, so its block states how many of the list's classes this
    // protocol ranked, one row per class. A protocol that ranked none of them writes the
    // refusal instead, and this is what says so.
    ASSERT_NE(text.find("carry a measured row", at), std::string::npos)
        << "this seam carries a class list and this protocol ranked none of its classes";
#else
    // This seam names no class of its own, so the file it implies carries the five names and no row list, and
    // the block has to say that in those terms. What it may not do is give the RUN as the reason: this protocol
    // ranked cells of every class checked below, and a block reading "this run measured no class" is the seam's
    // shape reported as the run's result, an empty measurement instead of an empty class list.
    EXPECT_EQ(text.find("this run measured no class", at), std::string::npos)
        << "the block blamed the run for a row list this build's seam does not carry";
#endif

    const std::string block = text.substr(at);

    // The host classes the seam's own key reaches: the lanes it keys by, crossed with the
    // shapes it can name. The two half formats are one lane in the library and two classes
    // here, so a bf16 class the lane also carries is read at the lane's own spelling.
    std::vector<std::string> classes;

    for (const boys::OptionProbeClass& klass : report.classes) {
        if (klass.precision == OptionPrecision::kFp32Device) {
            continue;
        }

        bool folded = false;

        if (klass.precision == OptionPrecision::kBf16) {
            for (const boys::OptionProbeClass& lane : report.classes) {
                folded = folded || (lane.precision == OptionPrecision::kFp16 &&
                                    lane.shape == klass.shape);
            }
        }

        if (folded) {
            continue;
        }

        classes.push_back(klass.name);
    }

    EXPECT_EQ(classes.size(), 15u)
        << "the seam keys three host lanes by five shapes, folded over the two half formats";

    for (const std::string& klass : classes) {
        EXPECT_NE(block.find("    " + klass + "\n"), std::string::npos)
            << klass << " is a class the seam's own key reaches and the block does not name";
    }
}

} // namespace
