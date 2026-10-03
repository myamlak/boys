/// \file
/// The defaults a caller reaches by naming nothing — instantiated, and linked.
///
/// WHY THIS FILE EXISTS. On 2026-10-02 two pushes broke the build in the same way,
/// and no test in this tree would have caught either before it reached CI:
///
///   * a row named `PackAxis::kOrders`, whose body the library holds OUT OF LINE at a
///     fixed list of instantiations. A caller naming no policy therefore linked only
///     where that list held the policy the default resolves to, and failed at every
///     other — and the failure is a link error at the *call site*, not at the table,
///     so the row that caused it reads as correct;
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
/// POLICY. Naming nothing is the whole point; naming a policy explicitly reaches a
/// different instantiation and would not test the default at all.
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

TEST(BoysDefaultsLink, TheDoubleLaneAnswersEveryShapeWithNoPolicyNamed) {
    double ladder[kOrder + 1] = {};
    const std::array<double, 3> xs = {0.25, 0.75, 1.5};
    // The many-argument entries write (nmax + 1) * count slots, not count: the grid is
    // order-major, so the array is sized by the grid while the call's own count is the
    // number of arguments. The workspace is sized by that count too.
    std::array<double, (kOrder + 1) * 3> out = {};
    std::array<std::size_t, 3> workspace = {};
    const std::array<int, 3> ns = {0, 1, 2};

    // The default policy's instantiation: the entry must reach a definition and a value.
    EXPECT_TRUE(std::isfinite(boys::BoysSingle(kOrder, kArgument)));
    boys::BoysAllOrders(kOrder, kArgument, ladder);
    EXPECT_TRUE(std::isfinite(ladder[kOrder]));
    boys::BoysFixedN(kOrder, xs.data(), out.data(), xs.size());
    boys::BoysAllN(kOrder, xs.data(), out.data(), xs.size(), workspace.data());
    boys::BoysAllNAtOrders(ns.data(), xs.data(), out.data(), xs.size());
}

TEST(BoysDefaultsLink, TheSinglePrecisionLaneAnswersEveryShapeWithNoPolicyNamed) {
    float ladder[kOrder + 1] = {};
    const std::array<float, 3> xs = {0.25f, 0.75f, 1.5f};
    std::array<float, (kOrder + 1) * 3> out = {};

    EXPECT_TRUE(std::isfinite(boys::BoysSingleF32(kOrder, static_cast<float>(kArgument))));
    boys::BoysAllOrdersF32(kOrder, static_cast<float>(kArgument), ladder);
    EXPECT_TRUE(std::isfinite(ladder[kOrder]));
    boys::BoysAllNF32(kOrder, xs.data(), out.data(), xs.size());
}

#if BoysFp16
TEST(BoysDefaultsLink, TheHalfLaneAnswersBothFormatsWithNoPolicyNamed) {
    boys::F16 ladder16[kOrder + 1] = {};
    boys::Bf16 ladderBf[kOrder + 1] = {};
    const boys::F16 x16 = static_cast<boys::F16>(kArgument);
    const boys::Bf16 xBf = static_cast<boys::Bf16>(kArgument);

    EXPECT_TRUE(std::isfinite(static_cast<double>(boys::BoysSingleF16(kOrder, x16))));
    boys::BoysAllOrdersF16(kOrder, x16, ladder16);
    EXPECT_TRUE(std::isfinite(static_cast<double>(boys::BoysSingleBf16(kOrder, xBf))));
    boys::BoysAllOrdersBf16(kOrder, xBf, ladderBf);
}
#endif // BoysFp16

} // namespace
