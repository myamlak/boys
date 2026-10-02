/// \file
/// The defaults a caller reaches by naming nothing — instantiated, and linked.
///
/// WHY THIS FILE EXISTS. On 2026-10-02 two pushes broke the build in the same way,
/// and no test in this tree would have caught either before it reached CI:
///
///   * a row named `PackAxis::kOrders`, whose body the library holds OUT OF LINE at a
///     fixed list of multipliers. A caller naming no policy therefore linked at the
///     six multipliers the library instantiates and failed at every other — and the
///     failure is a link error at the *call site*, not at the table, so the row that
///     caused it reads as correct;
///   * three single-precision entries were declared `extern template` at a policy the
///     default now resolves to. An extern declaration PROMISES a definition held
///     elsewhere; the library holds a fixed list, and the instantiation a default
///     call site selects is not knowably in it. Four consumer targets became
///     unresolved externals.
///
/// Both are one defect: **a default that names a combination the library cannot serve
/// for every caller.** It is invisible in the table, invisible in the header, and it
/// breaks every call site that names nothing. Nothing in the tree checked it.
///
/// WHAT THIS FILE DOES. It is a consumer: it includes the public header and nothing
/// else, and for every entry that takes a class default it calls that entry NAMING NO
/// POLICY — twice, at the default accuracy multiplier and at one the library cannot
/// have listed. Naming nothing is the whole point; naming a policy explicitly reaches
/// a different instantiation and would not test the default at all.
///
/// The multiplier matters as much as the policy. A default call at the library's own
/// multiplier can be satisfied by an instantiation the library happens to hold; a
/// default call at an arbitrary multiplier can only be satisfied if the body is
/// reachable from the header. Those are two different promises and the second is the
/// one that broke.
///
/// A class the table carries that has no entry at all does not appear here — it
/// cannot, since there is nothing to call — and that is the other half of the same
/// condition: a row for a class no entry reaches is a default nobody can ask for.
/// `tools/check_default_rows_have_entries.py` holds the table and the entries to each
/// other and is where that half is caught.

#include <boys/boys.hpp>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>

namespace {

/// The argument every call is read at, and the order ladder's top.
constexpr int kOrder = 3;
constexpr double kArgument = 0.75;

/// A multiplier no list of instantiations holds. Chosen to be one no rung names, so a
/// default call at it cannot be answered by an instantiation written for a rung.
constexpr double kUnlistedMultiplier = 3.7;

TEST(BoysDefaultsLink, TheDoubleLaneAnswersEveryShapeWithNoPolicyNamed) {
    double ladder[kOrder + 1] = {};
    const std::array<double, 3> xs = {0.25, 0.75, 1.5};
    std::array<double, 3> out = {};
    std::array<std::size_t, 3> workspace = {};
    const std::array<int, 3> ns = {0, 1, 2};

    // Default multiplier: the instantiation the library is expected to hold.
    EXPECT_TRUE(std::isfinite(boys::BoysSingle(kOrder, kArgument)));
    boys::BoysAllOrders(kOrder, kArgument, ladder);
    EXPECT_TRUE(std::isfinite(ladder[kOrder]));
    boys::BoysFixedN(kOrder, xs.data(), out.data(), out.size());
    boys::BoysAllN(kOrder, xs.data(), out.data(), out.size(), workspace.data());
    boys::BoysAllNAtOrders(ns.data(), xs.data(), out.data(), out.size());

    // An unlisted multiplier: only reachable if the body is reachable from the header.
    EXPECT_TRUE(std::isfinite(boys::BoysSingle<kUnlistedMultiplier>(kOrder, kArgument)));
    boys::BoysAllOrders<kUnlistedMultiplier>(kOrder, kArgument, ladder);
    boys::BoysFixedN<kUnlistedMultiplier>(kOrder, xs.data(), out.data(), out.size());
    boys::BoysAllN<kUnlistedMultiplier>(kOrder, xs.data(), out.data(), out.size(),
                                        workspace.data());
    boys::BoysAllNAtOrders<kUnlistedMultiplier>(ns.data(), xs.data(), out.data(), out.size());
}

TEST(BoysDefaultsLink, TheSinglePrecisionLaneAnswersEveryShapeWithNoPolicyNamed) {
    float ladder[kOrder + 1] = {};
    const std::array<float, 3> xs = {0.25f, 0.75f, 1.5f};
    std::array<float, 3> out = {};

    EXPECT_TRUE(std::isfinite(boys::BoysSingleF32(kOrder, static_cast<float>(kArgument))));
    boys::BoysAllOrdersF32(kOrder, static_cast<float>(kArgument), ladder);
    EXPECT_TRUE(std::isfinite(ladder[kOrder]));
    boys::BoysAllNF32(kOrder, xs.data(), out.data(), out.size());

    EXPECT_TRUE(std::isfinite(
        boys::BoysSingleF32<kUnlistedMultiplier>(kOrder, static_cast<float>(kArgument))));
    boys::BoysAllOrdersF32<kUnlistedMultiplier>(kOrder, static_cast<float>(kArgument), ladder);
    boys::BoysAllNF32<kUnlistedMultiplier>(kOrder, xs.data(), out.data(), out.size());
}

#if BoysFp16
TEST(BoysDefaultsLink, TheHalfLaneAnswersBothFormatsWithNoPolicyNamed) {
    F16 ladder16[kOrder + 1] = {};
    Bf16 ladderBf[kOrder + 1] = {};
    const F16 x16 = static_cast<F16>(kArgument);
    const Bf16 xBf = static_cast<Bf16>(kArgument);

    EXPECT_TRUE(std::isfinite(static_cast<double>(boys::BoysSingleF16(kOrder, x16))));
    boys::BoysAllOrdersF16(kOrder, x16, ladder16);
    EXPECT_TRUE(std::isfinite(static_cast<double>(boys::BoysSingleBf16(kOrder, xBf))));
    boys::BoysAllOrdersBf16(kOrder, xBf, ladderBf);

    EXPECT_TRUE(std::isfinite(
        static_cast<double>(boys::BoysSingleF16<kUnlistedMultiplier>(kOrder, x16))));
    boys::BoysAllOrdersF16<kUnlistedMultiplier>(kOrder, x16, ladder16);
    EXPECT_TRUE(std::isfinite(
        static_cast<double>(boys::BoysSingleBf16<kUnlistedMultiplier>(kOrder, xBf))));
    boys::BoysAllOrdersBf16<kUnlistedMultiplier>(kOrder, xBf, ladderBf);
}
#endif // BoysFp16

} // namespace
