// The dispatch build's gate: both tiers are in the image, the processor selects one, and the same
// default entries return the same bits through either tier once the forcing hook pins it. The
// comparison is entry_digest::Bits(), the sweep the flag-independence test also prints. Where the
// processor lacks AVX2 the across-tier comparison is skipped; the image still selects baseline.

#include <gtest/gtest.h>

#include <boys/boys.hpp>

#if BOYS_ISA_DISPATCH && BOYS_ISA_TIER_FORCING

#include "entry_digest.hpp"

#include <cstdint>
#include <cstdio>

namespace {

/// Returns the dispatcher to its own selection when a test leaves, including on an assertion's
/// early return: a forced tier left in force would move every later test's arithmetic.
struct ClearForcedTier {
    ~ClearForcedTier() { boys::BoysClearForcedTier(); }
};

/// The tier's name for a diagnostic line.
const char* TierName(boys::BoysIsaTier tier) noexcept {
    return tier == boys::BoysIsaTier::kAvx2 ? "avx2" : "baseline";
}

} // namespace

TEST(BoysDispatch, BothTiersAreCarriedAndTheProcessorSelects) {
    const boys::BoysTierReport report = boys::BoysActiveTier();

    EXPECT_TRUE(report.carriesBaseline) << "the scalar tier is not in this image";
    EXPECT_TRUE(report.carriesAvx2) << "the vector tier is not in this image";
    EXPECT_EQ(report.machineAvx2, boys::BoysAvx2Available())
        << "the report's machine answer and the library's availability entry disagree";
    EXPECT_EQ(report.selected,
              report.machineAvx2 ? boys::BoysIsaTier::kAvx2 : boys::BoysIsaTier::kBaseline)
        << "the selection is the machine's: AVX2+FMA where the processor exposes it, the baseline "
           "tier otherwise";
}

TEST(BoysDispatch, TheForcedTiersAgreeBitForBit) {
    if (!boys::BoysAvx2Available())
    {
        GTEST_SKIP() << "the processor does not expose AVX2 and FMA, so the carried vector tier "
                        "cannot be executed here; the baseline-emulation leg's subject is the "
                        "dispatcher's selection and its freedom from an illegal instruction";
    }

    ClearForcedTier clear;

    ASSERT_TRUE(boys::BoysForceTier(boys::BoysIsaTier::kBaseline))
        << "the baseline tier is carried, so it must be forceable";
    ASSERT_EQ(boys::BoysActiveTier().selected, boys::BoysIsaTier::kBaseline);
    const std::uint64_t baseline = entry_digest::Bits();

    ASSERT_TRUE(boys::BoysForceTier(boys::BoysIsaTier::kAvx2))
        << "the vector tier is carried, so it must be forceable";
    ASSERT_EQ(boys::BoysActiveTier().selected, boys::BoysIsaTier::kAvx2);
    const std::uint64_t avx2 = entry_digest::Bits();

    boys::BoysClearForcedTier();

    std::printf("\n  forced-tier digest: baseline %016llx, avx2 %016llx\n",
                static_cast<unsigned long long>(baseline),
                static_cast<unsigned long long>(avx2));

    EXPECT_EQ(baseline, avx2)
        << "the carried tiers are documented bit-equal on every default entry; a differing "
           "digest is an arithmetic difference between the tiers";
}

TEST(BoysDispatch, TheSelectionDeliversTheBitsOfItsOwnTier) {
    ClearForcedTier clear;

    const boys::BoysTierReport report = boys::BoysActiveTier();
    const std::uint64_t selected = entry_digest::Bits();

    ASSERT_TRUE(boys::BoysForceTier(report.selected))
        << "the selected tier is carried by definition";
    const std::uint64_t forced = entry_digest::Bits();

    boys::BoysClearForcedTier();

    std::printf("  selected tier %s: digest %016llx, the same tier forced %016llx\n",
                TierName(report.selected),
                static_cast<unsigned long long>(selected),
                static_cast<unsigned long long>(forced));

    EXPECT_EQ(selected, forced)
        << "the dispatcher's own selection and the same tier forced must deliver one digest";
}

#else

TEST(BoysDispatch, TheForcingGateIsNotBuiltHere) {
    GTEST_SKIP() << "this build does not carry the tier-forcing hook "
                    "(BOYS_ISA_DISPATCH && BOYS_ISA_TIER_FORCING), so there is no second tier to "
                    "force and no digest to compare";
}

#endif
