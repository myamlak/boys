// The archive-free accuracy header, held to the documents it quotes.

// This target links nothing, and CMakeLists.txt asserts its link line is empty: the two
// figures below are answered with no archive on the link line, so a figure that needed
// src/boys.cpp would fail to link here.

// The unit includes boys/boys_accuracy_here.hpp and no other header of the library's, so
// what compiles is exactly what a consumer adds for this question. The figures are written
// here as the documents state them, not read back from the header, so a header that moved a
// number cannot agree with itself.

// The lane's published absolute figure - docs/specification.md's accuracy contract and
// BoysLaneContracts(): 5.5e-14 on the double lane, 1.5e-7 on the float lane, plus that
// lane's documented 1e-7 plain-reciprocal term where the class's own form is the plain form.

// This is what BoysAccuracyHere answers as `bound`, and the header says plainly that it is
// the measured absolute figure and not the cell contract: BoysAccuracyGuaranteed's answer
// at (n, x) is the contract, and it lives in the archive.

// The delivered figure - docs/specification.md's route table and docs/lane-contract.md's
// two route tables: 3.22e-15 on the double single band and 6.36e-08 on the float single
// lane for the fused route, 2.74e-15 and 6.36e-08 on a contracting build, 4.63e-15 and
// 7.49e-08 on the separate route.

// The column is selected by the arithmetic this build reports in ArithmeticInForceHere,
// the header's own key and the route tables' own key.

#include <boys/boys_accuracy_here.hpp>

#include <cstdio>
#include <span>
#include <string_view>

namespace {

int gFailed = 0;

void Check(const char* what, bool held) {
    if (held)
    {
        std::printf("  ok   %s\n", what);
        return;
    }

    std::printf("  FAIL %s\n", what);
    ++gFailed;
}

void Check(const char* what, double measured, double documented) {
    if (measured == documented)
    {
        std::printf("  ok   %-46s %.17g\n", what, measured);
        return;
    }

    std::printf("  FAIL %-46s measured %.17g, documented %.17g\n", what, measured, documented);
    ++gFailed;
}

bool Names(const char* text, std::string_view what) {
    return std::string_view(text).find(what) != std::string_view::npos;
}

} // namespace

int main() {
    using namespace boys;

    std::printf("boys accuracy-here check: %s\n", ArithmeticInForceName(ArithmeticInForceHere()));

    // The eight lanes' published absolute figures, restated from BoysLaneContracts().
    const std::span<const LaneFigureHere> lanes = BoysLaneFiguresHere();
    Check("eight lanes stated", lanes.size() == 8);
    Check("fp64 lane's published figure", lanes[0].bound, 5.5e-14);
    Check("fp32 lane's published figure", lanes[1].bound, 1.5e-7);
    Check("fp32 lane's plain-reciprocal term", lanes[1].plainAdditive, 1e-7);

    // `bound`: the lane's published absolute figure, at the class's own policy.
    const ClassAccuracyHere fp64Single = BoysAccuracyHere<Precision::kFp64, Shape::kSingle>();
    Check("fp64 single bound", fp64Single.bound.value, 5.5e-14);
    Check("fp64 single bound is kMeasured",
          fp64Single.bound.reading == AccuracyReading::kMeasured);

    const ClassAccuracyHere fp32Single = BoysAccuracyHere<Precision::kFp32, Shape::kSingle>();
    // The class's own default form is the build's and not a literal: the committed
    // float single row divides in the plain reciprocal and carries that form's
    // documented 1e-7 term; a replacement that moves the row answers at its member,
    // read here from the policy, with both figures the documents' own.
    constexpr DivisionForm fp32SingleForm =
        DefaultPolicy<Precision::kFp32, Shape::kSingle>::kDivision;
    Check("fp32 single bound at its default form", fp32Single.bound.value,
          fp32SingleForm == DivisionForm::kPlainReciprocal ? 2.5e-7 : 1.5e-7);
    Check("fp32 single bound at the refined form",
          BoysAccuracyHere<Precision::kFp32, Shape::kSingle, Device::kHost,
                           DivisionForm::kRefinedReciprocal>().bound.value,
          1.5e-7);
    Check("fp32 single bound, plain reciprocal",
          BoysAccuracyHere<Precision::kFp32, Shape::kSingle, Device::kHost,
                           DivisionForm::kPlainReciprocal>().bound.value,
          2.5e-7);

    // `delivered`: the route table's column for the arithmetic this build reports.
    const ArithmeticInForce arithmetic = ArithmeticInForceHere();
    double doubleBand = 0.0;
    double floatSingle = 0.0;
    double floatBatch = 0.0;
    const char* column = "";

    if (arithmetic.routeInForce == backend::MulAddRoute::kSeparate)
    {
        doubleBand = 4.63e-15;
        floatSingle = 7.49e-08;
        floatBatch = 1.08e-07;
        column = "the separate route";
    }
    else if (arithmetic.contracts)
    {
        doubleBand = 2.74e-15;
        floatSingle = 6.36e-08;
        floatBatch = 9.11e-08;
        column = "the contracting host";
    }
    else
    {
        doubleBand = 3.22e-15;
        floatSingle = 6.36e-08;
        floatBatch = 1.08e-07;
        column = "the fused route";
    }

    std::printf("  the delivered figures are %s's column of the route tables\n", column);

    Check("fp64 single delivered", fp64Single.delivered.value, doubleBand);
    Check("fp32 single delivered", fp32Single.delivered.value, floatSingle);
    Check("fp32 batch delivered",
          BoysAccuracyHere<Precision::kFp32, Shape::kFixedN>().delivered.value, floatBatch);
    Check("fp64 single delivered is kDelivered",
          fp64Single.delivered.reading == AccuracyReading::kDelivered);
    Check("fp64 single delivered available", fp64Single.delivered.available);

    // The delivered figure is a swept maximum over the region its record names, and the scope is
    // part of what it says: the double lane's band, and all arguments on the float lane.
    Check("fp64 single delivered scope is the band",
          Names(fp64Single.deliveredScope, "1.0855252345349333 <= x < 11.899848152108484"));
    Check("fp32 single delivered scope is all arguments",
          std::string_view(
              BoysAccuracyHere<Precision::kFp32, Shape::kSingle>().deliveredScope) ==
              "all arguments");

    // The header states no delivered figure for a class it has none for, rather than answering a
    // number: the double lane's batch shapes are the route tables' unmeasured cells.
    Check("fp64 batch states no delivered figure",
          !BoysAccuracyHere<Precision::kFp64, Shape::kFixedN>().delivered.available);

    // ArithmeticInForceName on the four measured states a build can report. Each is
    // constructed rather than read, so the name is held to the state it is given and
    // not to this translation unit's own profile. The states are the two selections
    // times the two contraction measurements; on the separate selection the route in
    // force is the fused route only where contraction is measured.
    {
        using boys::backend::MulAddRoute;

        const auto state = [](MulAddRoute selected, bool contracts, MulAddRoute inForce) {
            ArithmeticInForce arithmetic;
            arithmetic.selectedRoute = selected;
            arithmetic.contractionMeasured = true;
            arithmetic.contracts = contracts;
            arithmetic.routeInForceKnown = true;
            arithmetic.routeInForce = inForce;
            return arithmetic;
        };

        const ArithmeticInForce separateContracting =
            state(MulAddRoute::kSeparate, true, MulAddRoute::kFused);
        const ArithmeticInForce separateTwo =
            state(MulAddRoute::kSeparate, false, MulAddRoute::kSeparate);
        const ArithmeticInForce fusedContracting =
            state(MulAddRoute::kFused, true, MulAddRoute::kFused);
        const ArithmeticInForce fusedTwo =
            state(MulAddRoute::kFused, false, MulAddRoute::kFused);

        Check("separate selection that contracts is named the fused route by contraction",
              Names(ArithmeticInForceName(separateContracting), "the fused route by contraction"));
        Check("separate selection that does not contract is named the separate route",
              Names(ArithmeticInForceName(separateTwo), "the separate route"));
        Check("fused selection that contracts is named the contracted arithmetic",
              Names(ArithmeticInForceName(fusedContracting), "the contracted arithmetic"));
        Check("fused selection that does not contract is named the fused route",
              Names(ArithmeticInForceName(fusedTwo), "the fused route"));

        // Each name states the route its own routeInForce carries: the fused route's
        // sentence names the fused route, and the separate route's names only the
        // separate route.
        const auto agrees = [](const ArithmeticInForce& arithmetic) {
            const std::string_view name = ArithmeticInForceName(arithmetic);
            const bool fused = name.find("fused route") != std::string_view::npos;
            const bool separate = name.find("separate route") != std::string_view::npos;
            if (arithmetic.routeInForce == MulAddRoute::kFused)
            {
                return fused;
            }

            return separate && !fused;
        };

        Check("the named route agrees with routeInForce (separate selection, contracting)",
              agrees(separateContracting));
        Check("the named route agrees with routeInForce (separate selection, two roundings)",
              agrees(separateTwo));
        Check("the named route agrees with routeInForce (fused selection, contracting)",
              agrees(fusedContracting));
        Check("the named route agrees with routeInForce (fused selection, two roundings)",
              agrees(fusedTwo));
    }

    if (gFailed != 0)
    {
        std::printf("FAIL: %d figure(s) disagree with the documents they are stated from\n",
                    gFailed);
        return 1;
    }

    std::printf("PASS: every figure this header answers agrees with the document it names, "
                "with no archive on the link line\n");
    return 0;
}
