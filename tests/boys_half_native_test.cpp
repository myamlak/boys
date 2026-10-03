// The native half lane: boys/half2.hpp and the two entries it backs,
// boys::BoysAllOrdersHalf2 and boys::BoysAllNF16Native.
//
// The lane is a claim about what packed half arithmetic can carry, graded the way the
// accuracy gate grades a published one: three claims, each with its own domain and
// exactly one verdict (the book at the end of the file prints them).
//
//   * native-half-packed. Every operation in half2.hpp is on a packed pair and correctly
//     rounded to binary16, so the lane's error is the arithmetic's, not an implementation
//     tolerance. Correct rounding is decided exactly, in binary64, no epsilon: a half times
//     a half, or a half times the midpoint to its neighbour (a 25-bit significand), is exact
//     in binary64, so "which side of the rounding boundary" is an exact comparison. Square
//     root: against the squares of the midpoints, exhaustive over every non-negative half.
//     The four operations: over every finite half against a curated operand set plus a random
//     sample. The rounding distribution is what separates this lane from the fp16 I/O lane -
//     the error exceeds half an ULP at 47.3% of values and one ULP at 20.0%, which a lane
//     that widens, evaluates and rounds once at the end cannot produce.
//   * native-half-bound. |out[k] - 2^15 F_k(x)| <= 8 ULP(out[k]), the quantum of the returned
//     value, over region C at its precondition. Measured against the 45-digit committed grid
//     where the argument is exactly a half, and against the certified double lane (5.5e-14
//     relative, nine orders of magnitude inside the half quantum) over a sweep of region C.
//     The domain is the arguments whose returned value is a normal half; the worst measured
//     ratio there is 4.243 ULP, a swept maximum and not a proof bound.
//   * native-half-ceiling. The argument past which no accuracy is claimed, per order: below
//     F_k(x) = 2^-29 the entry returns a subnormal and then a zero by design, and that domain
//     is counted rather than passed. The ceiling is where the value crosses the format's
//     smallest normal, and the span wall (the ladder's own F_0/F_k against the format's
//     range) stops a larger scale from moving it.

#include "boys/boys.hpp"
#include "boys/half2.hpp"
#include "boys/boys_impl.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

using boys::F16;
using boys::Half2;
using boys::kMaxBoysOrder;
using boys::detail::F16Bits;
using boys::detail::F16FromBits;
using boys::detail::F16FromDouble;
using boys::detail::kX1;

// The reading of a half as a double, which both sections below measure against: it rests on
// nothing the seam declares, so it sits outside the guard.
double Value(F16 value) noexcept {
    return static_cast<double>(value);
}

double Value(std::uint16_t bits) noexcept {
    return Value(F16FromBits(bits));
}

// The native half lane's two entries, and the scale constant that goes with them
// (boys/boys.hpp), are declared behind the BoysFp16 seam, and everything this file measures
// about the *lane* is compiled with them: the closed configuration is reported as the
// skipped test at the end of this file rather than as greens over a lane that is not there.
// The packed type's arithmetic is not seam-gated - boys/half2.hpp is a header of this tree a
// consumer may include directly - so the packed section below is outside the guard and runs
// in both configurations. The split is by what a declaration needs: the guard holds the
// lane's claim constants, its reference machinery and the evidence helpers only the book
// names; the format helpers both sections share stand between them.
constexpr double kInfinity = std::numeric_limits<double>::infinity();

// --- The claims, in the accuracy gate's shape ---------------------------------
//
// Each claim carries its own domain and gets exactly one verdict, so one can fail without
// touching the others. The vocabulary is the gate's, name for name.
enum class Verdict { Verified, Exceeded, Vacuous, EvidenceAbsent };

// Only the lane's claim book names a verdict or formats an evidence line, so a build that
// carries no lane has no caller for either.
#if BoysFp16

const char* VerdictName(Verdict verdict) {
    switch (verdict)
    {
    case Verdict::Verified:
        return "verified at this revision";
    case Verdict::Exceeded:
        return "EXCEEDED";
    case Verdict::Vacuous:
        return "vacuous only";
    case Verdict::EvidenceAbsent:
        return "evidence absent from this run";
    }

    return "?";
}

// An evidence line long enough for the longest claim: a string cut mid-claim would read weaker.
std::string Fmt(const char* format, ...) {
    char buffer[512];
    va_list args;
    va_start(args, format);
    std::vsnprintf(buffer, sizeof(buffer), format, args);
    va_end(args);
    return std::string(buffer);
}

#endif // BoysFp16

// native-half-packed: every operation correctly rounded on a packed pair, and
// the rounding distribution that only per-operation arithmetic produces.
struct PackedClaim {
    long sqrtCases = 0;
    long opCases = 0;
    long randomCases = 0;
    long beyondHalfUlp = 0;
    long values = 0;
    double maxUlps = 0.0;
};

PackedClaim gPacked;

// The part of these declarations the seam reaches: the scale below is the public constant
// boys/boys.hpp declares under it, and the claim types name the domain that scale defines.
// The format helpers under them and the packed tests below them use nothing the seam gates.
#if BoysFp16

// The lane's documented bound, in quanta of the returned value. The worst measured ratio is
// 4.3 (order 6, in the region-C sweep below), and the bound is the next power of two above
// it: an assertion of a bound, not a pin of the measurement.
constexpr double kBoundUlps = 8.0;

// The lane's scale, as the public constant spells it (an exact power of two).
constexpr int kScale = boys::kHalfNativeScaleExponent;

// The smallest normal binary16 value, 2^-14: where the scale's output stops being normal.
constexpr double kSmallestNormal = 6.103515625e-05;

// native-half-bound: the ULP bound over the domain where the returned value is
// a normal half, with the no-claim domain counted beside it.
struct BoundClaim {
    long claimed = 0;
    long noClaim = 0;
    double worstUlps = 0.0;
    int worstOrder = 0;
    double worstX = 0.0;
    // The largest true value among the no-claim points, as a multiple of the smallest normal
    // value: where that domain starts, against where the ceiling says it starts.
    double largestNoClaimTruth = 0.0;
};

// native-half-ceiling: the argument past which no accuracy is claimed, per order (the largest
// one whose returned value is still normal), and the span that stops a larger scale moving it.
struct CeilingClaim {
    std::array<double, kMaxBoysOrder + 1> crossing{};
    double spanWall = 0.0;
    int spanOrder = 0;
};

BoundClaim gBound;
CeilingClaim gCeiling;

#endif // BoysFp16

// The format's own helpers, outside the guard because the packed operations below are the
// format's arithmetic and not behind the seam, and the lane's sweeps share them.

// The ordered index of a half: sign-magnitude bits into a monotone integer, -Inf at 0x0400,
// +Inf at 0xFC00. -0 and +0 share an index; neither is the other's value-neighbour.
int HalfIndex(std::uint16_t bits) noexcept {
    const int magnitude = static_cast<int>(bits & 0x7FFFu);
    return (bits & 0x8000u) != 0u ? 0x8000 - magnitude : 0x8000 + magnitude;
}

constexpr int kNegInfIndex = 0x0400;
constexpr int kPosInfIndex = 0xFC00;

std::uint16_t HalfBitsAt(int index) noexcept {
    const int magnitude = index >= 0x8000 ? index - 0x8000 : 0x8000 - index;
    return static_cast<std::uint16_t>((index >= 0x8000 ? 0x0000u : 0x8000u) |
                                      static_cast<std::uint32_t>(magnitude));
}

// The binary16 quantum at a magnitude: 2^(e - 10) for a normal value, the format's floor
// 2^-24 for a subnormal or a zero. Only the lane's error metric calls it.
#if BoysFp16

double QuantumAt(double magnitude) noexcept {
    if (magnitude < std::ldexp(1.0, -14))
    {
        return std::ldexp(1.0, -24);
    }

    return std::ldexp(1.0, std::ilogb(magnitude) - 10);
}

double Quantum(F16 value) noexcept {
    return QuantumAt(std::fabs(Value(value)));
}

#endif // BoysFp16

// The exact value of a op b compared with the exactly representable c, as -1 / 0 / +1.
// '+' '-' '*' of half operands are exact in binary64; '/' is an exact product comparison (a
// half by a half or by a midpoint is at most 37 significand bits), so no rounding enters.
int CompareExact(F16 a, F16 b, double c, char op) {
    const double av = Value(a);
    const double bv = Value(b);

    switch (op)
    {
    case '+':
        return av + bv < c ? -1 : (av + bv > c ? 1 : 0);
    case '-':
        return av - bv < c ? -1 : (av - bv > c ? 1 : 0);
    case '*':
        return av * bv < c ? -1 : (av * bv > c ? 1 : 0);
    default: {
        const double rhs = c * bv;
        const int cmp = av < rhs ? -1 : (av > rhs ? 1 : 0);
        return bv > 0.0 ? cmp : -cmp;
    }
    }
}

// Is `result` the correctly rounded binary16 of the exact value of a op b?
//
// Exact and implementation-independent: nearest exactly when the exact value lies between the
// midpoints to its two neighbours, an exact tie going to the even significand.
bool IsCorrectlyRounded(F16 result, F16 a, F16 b, char op) {
    const std::uint16_t bits = F16Bits(result);
    const std::uint32_t magnitude = bits & 0x7FFFu;

    if (magnitude >= 0x7C00u)
    {
        // An infinity is the correctly rounded result exactly when the exact value is at or
        // past the round-to-infinity threshold: 65520 is the midpoint between 65504 and 65536,
        // and 65504's significand is odd, so the tie goes to the infinity. A NaN is never
        // right here: every NaN comes from a non-finite operand, and the sweeps below run over
        // finite ones.
        if (magnitude != 0x7C00u)
        {
            return false;
        }

        const int comparison = CompareExact(a, b, (bits & 0x8000u) != 0u ? -65520.0 : 65520.0, op);
        return (bits & 0x8000u) != 0u ? comparison <= 0 : comparison >= 0;
    }

    const int index = HalfIndex(bits);
    const double r = Value(result);
    const double below = index > kNegInfIndex ? Value(HalfBitsAt(index - 1)) : -kInfinity;
    const double above = index < kPosInfIndex ? Value(HalfBitsAt(index + 1)) : kInfinity;

    const int lower = CompareExact(a, b, 0.5 * (r + below), op);
    const int upper = CompareExact(a, b, 0.5 * (r + above), op);

    if (lower < 0 || upper > 0)
    {
        return false;
    }

    // A tie goes to the even significand: 0 is even, so a zero wins its tie against the
    // smallest subnormal, and the even-infinity significand wins the overflow tie.
    return (lower != 0 && upper != 0) || (bits & 0x1u) == 0u;
}

// The same test for the square root, against the squares of the midpoints
// (25-bit significands again: their squares are exact in binary64).
bool IsCorrectlyRoundedSqrt(F16 result, F16 a) {
    const std::uint16_t bits = F16Bits(result);

    // The square root of an infinity is that infinity; of a finite non-negative half, finite.
    if ((F16Bits(a) & 0x7FFFu) == 0x7C00u)
    {
        return (bits & 0x7FFFu) == 0x7C00u && (bits & 0x8000u) == (F16Bits(a) & 0x8000u);
    }

    if ((bits & 0x7FFFu) >= 0x7C00u)
    {
        return false;
    }

    const int index = HalfIndex(bits);
    const double r = Value(result);
    const double x = Value(a);
    const double below = index > 0x8000 ? 0.5 * (r + Value(HalfBitsAt(index - 1))) : 0.0;
    const double above = 0.5 * (r + Value(HalfBitsAt(index + 1)));

    if (x < below * below || x > above * above)
    {
        return false;
    }

    return (x != below * below && x != above * above) || (bits & 0x1u) == 0u;
}

// The first operands of the packing sweep: the magnitudes the format
// distinguishes, both signs, each crossed with every finite half.
std::vector<F16> CuratedOperands() {
    const std::uint16_t bits[] = {
        0x0000u, 0x8000u, // the zeroes
        0x0001u, 0x8001u, // the smallest subnormals
        0x0002u, 0x0003u, // subnormal odd significands
        0x000Fu, 0x0401u, // subnormal to normal
        0x03FFu, 0x83FFu, // the largest subnormals
        0x0400u, 0x8400u, // the smallest normals
        0x0410u, 0x3800u, // 2^-13 and one half
        0x3C00u, 0xBC00u, // 1
        0x3C01u, 0x3FFFu, // odd significands around 1 and 2
        0x3E00u, 0xBE00u, // 1.5
        0x4000u, 0xC000u, // 2
        0x4200u, 0xC200u, // 3
        0x5640u, 0xD640u, // 100
        0x6400u, 0xE400u, // 1024
        0x7BFEu, 0x7BFFu, // the maximum
        0xFBFFu, // the maximum, negative
    };

    std::vector<F16> operands;
    for (std::uint16_t b : bits)
    {
        operands.push_back(F16FromBits(b));
    }

    return operands;
}

// Every finite half, both signs.
std::vector<F16> FiniteHalves() {
    std::vector<F16> values;
    values.reserve(63488);

    for (std::uint32_t bits = 0; bits < 0x7C00u; ++bits)
    {
        values.push_back(F16FromBits(static_cast<std::uint16_t>(bits)));
        values.push_back(F16FromBits(static_cast<std::uint16_t>(bits | 0x8000u)));
    }

    return values;
}

#if BoysFp16

// The lane's reference machinery: declared only where this build carries the lane, since in a
// closed build these would be functions nothing refers to.

// The region-C boundary as the format holds it: the fp16 value of kX1.
F16 BoundaryArgument() {
    return F16FromDouble(kX1);
}

// The sweep arguments: every representable value just above the boundary (where the ladder's
// values are largest), then about 2% steps up to the format's maximum, so each order's reach
// and its far tail are sampled where the value changes fastest.
std::vector<F16> SweepArguments() {
    std::vector<F16> xs;
    const int start = HalfIndex(F16Bits(BoundaryArgument()));

    for (int i = start; i < kPosInfIndex;)
    {
        const F16 x = F16FromBits(HalfBitsAt(i));
        xs.push_back(x);

        if (i < start + 256)
        {
            ++i;
            continue;
        }

        const double target = Value(x) * 1.02;
        int j = i;
        while (j < kPosInfIndex && Value(HalfBitsAt(j)) < target)
        {
            ++j;
        }

        if (j >= kPosInfIndex)
        {
            break;
        }

        i = j;
    }

    // The format's maximum, so the sweep ends where the format does.
    if (HalfIndex(F16Bits(xs.back())) != kPosInfIndex - 1)
    {
        xs.push_back(F16FromBits(0x7BFFu));
    }

    return xs;
}

// The certified double lane at a half argument: the reference for a half
// comparison (its budget is nine orders of magnitude inside the quantum).
std::vector<double> Reference(int nmax, F16 x) {
    std::vector<double> out(static_cast<std::size_t>(nmax) + 1);
    boys::BoysAllOrders(nmax, Value(x), out.data());
    return out;
}

// The lane returns 2^kScale F_k(x), so the reference is lifted by the same exact power of two.
double Scaled(double value) noexcept {
    return std::ldexp(value, kScale);
}

struct GridRow {
    int n;
    double x;
    double value;
};

// The committed 45-digit grid (the loader boys_test.cpp and boys_accuracy_test.cpp use).
std::vector<GridRow> LoadReferenceGrid() {
    const std::string path = std::string(BoysDataDir) + "/boys_reference.csv";
    std::ifstream file(path);

    if (!file)
    {
        ADD_FAILURE() << "missing reference data: " << path;
        return {};
    }

    std::vector<GridRow> rows;
    std::string line;
    std::getline(file, line); // header

    while (std::getline(file, line))
    {
        std::stringstream ss(line);
        std::string cell;
        GridRow row{};
        std::getline(ss, cell, ',');
        row.n = std::stoi(cell);
        std::getline(ss, cell, ',');
        row.x = std::strtod(cell.c_str(), nullptr);
        std::getline(ss, cell, ',');
        row.value = std::strtod(cell.c_str(), nullptr);
        rows.push_back(row);
    }

    return rows;
}

#endif // BoysFp16

// --- The packed operations --------------------------------------------------
// boys/half2.hpp is included directly at the top of this file, defines every operation this
// section tests, and is not gated by the seam: these run whether the lane's entries are in the
// build or not.

TEST(NativeHalfLaneTest, PackedContainerRoundTrip) {
    const Half2 pair(F16FromBits(0x3C00u), F16FromBits(0xBC00u));

    EXPECT_EQ(pair.Bits(), 0xBC003C00u);
    EXPECT_EQ(F16Bits(pair.Low()), 0x3C00u);
    EXPECT_EQ(F16Bits(pair.High()), 0xBC00u);
    EXPECT_EQ(Half2::FromBits(0xBC003C00u), pair);
    EXPECT_EQ(Half2::Broadcast(F16FromBits(0x3E00u)).Bits(), 0x3E003E00u);
    EXPECT_EQ(Half2::Broadcast(F16FromBits(0x3E00u)).Low(), F16FromBits(0x3E00u));
    EXPECT_EQ(Half2().Bits(), 0u);
}

TEST(NativeHalfLaneTest, PackedSquareRootIsCorrectlyRounded) {
    // Exhaustive over every non-negative half. A negative argument and a
    // NaN result are outside the ladder's domain and are pinned separately.
    for (std::uint32_t bits = 0; bits <= 0x7C00u; ++bits)
    {
        const F16 x = F16FromBits(static_cast<std::uint16_t>(bits));
        const F16 root = Half2Sqrt(Half2(x, x)).Low();

        ASSERT_TRUE(IsCorrectlyRoundedSqrt(root, x)) << "sqrt of bits " << bits;
        ++gPacked.sqrtCases;
    }

    EXPECT_EQ(gPacked.sqrtCases, 0x7C01L); // 0 .. +inf, every non-negative half
}

TEST(NativeHalfLaneTest, PackedSquareRootIsCorrectlyRoundedForNegativeAndNotANumber) {
    const F16 minusOne = F16FromBits(0xBC00u);
    const F16 minusZero = F16FromBits(0x8000u);
    const F16 notANumber = F16FromBits(0x7E00u);

    // sqrt(-0) is -0; a negative argument and a NaN propagate as a NaN
    // (std::sqrt raises invalid and returns the quieted payload).
    EXPECT_EQ(F16Bits(Half2Sqrt(Half2(minusZero, minusZero)).Low()), 0x8000u);
    EXPECT_TRUE(std::isnan(Value(Half2Sqrt(Half2(minusOne, minusOne)).Low())));
    EXPECT_TRUE(std::isnan(Value(Half2Sqrt(Half2(notANumber, notANumber)).Low())));
}

TEST(NativeHalfLaneTest, PackedArithmeticIsCorrectlyRounded) {
    const char ops[] = {'+', '-', '*', '/'};
    const std::vector<F16> first = CuratedOperands();
    const std::vector<F16> second = FiniteHalves();

    for (char op : ops)
    {
        for (F16 a : first)
        {
            for (F16 b : second)
            {
                if (op == '/' && Value(b) == 0.0)
                {
                    continue; // a zero divisor is a pole, not a rounding
                }

                const Half2 r = op == '+'   ? Half2Add(Half2(a, a), Half2(b, b))
                                : op == '-' ? Half2Sub(Half2(a, a), Half2(b, b))
                                : op == '*' ? Half2Mul(Half2(a, a), Half2(b, b))
                                            : Half2Div(Half2(a, a), Half2(b, b));

                ASSERT_TRUE(IsCorrectlyRounded(r.Low(), a, b, op))
                    << "a op b = " << Value(a) << " " << op << " " << Value(b);
                ASSERT_EQ(r.Low(), r.High());
                ++gPacked.opCases;
            }
        }
    }
}

TEST(NativeHalfLaneTest, PackedArithmeticIsCorrectlyRoundedOnRandomPairs) {
    std::mt19937_64 rng(0x5EED1234u);
    const char ops[] = {'+', '-', '*', '/'};

    for (char op : ops)
    {
        for (int i = 0; i < 200000; ++i)
        {
            const std::uint16_t aBits = static_cast<std::uint16_t>(rng() % 0x7C00u);
            const std::uint16_t bBits = static_cast<std::uint16_t>(rng() % 0x7C00u);
            const F16 a = F16FromBits(static_cast<std::uint16_t>(aBits | (rng() & 0x8000u)));
            const F16 b = F16FromBits(static_cast<std::uint16_t>(bBits | (rng() & 0x8000u)));

            if (op == '/' && Value(b) == 0.0)
            {
                continue;
            }

            const Half2 r = op == '+'   ? Half2Add(Half2(a, a), Half2(b, b))
                            : op == '-' ? Half2Sub(Half2(a, a), Half2(b, b))
                            : op == '*' ? Half2Mul(Half2(a, a), Half2(b, b))
                                        : Half2Div(Half2(a, a), Half2(b, b));

            ASSERT_TRUE(IsCorrectlyRounded(r.Low(), a, b, op))
                << "a op b = " << Value(a) << " " << op << " " << Value(b);
            ++gPacked.randomCases;
        }
    }
}

TEST(NativeHalfLaneTest, PackedArithmeticOnNonFiniteOperands) {
    const F16 plusInf = F16FromBits(0x7C00u);
    const F16 minusInf = F16FromBits(0xFC00u);
    const F16 zero = F16FromBits(0x0000u);
    const F16 notANumber = F16FromBits(0x7E00u);

    EXPECT_TRUE(
        std::isnan(Value(Half2Add(Half2(plusInf, plusInf), Half2(minusInf, minusInf)).Low())));
    EXPECT_TRUE(
        std::isnan(Value(Half2Sub(Half2(plusInf, plusInf), Half2(plusInf, plusInf)).Low())));
    EXPECT_TRUE(std::isnan(Value(Half2Mul(Half2(plusInf, plusInf), Half2(zero, zero)).Low())));
    EXPECT_TRUE(std::isnan(Value(Half2Div(Half2(zero, zero), Half2(zero, zero)).Low())));
    EXPECT_TRUE(
        std::isnan(Value(Half2Add(Half2(notANumber, notANumber), Half2(zero, zero)).Low())));
    EXPECT_EQ(F16Bits(Half2Div(Half2(plusInf, plusInf), Half2(zero, zero)).Low()), 0x7C00u);
    EXPECT_EQ(F16Bits(Half2Div(Half2(minusInf, minusInf), Half2(zero, zero)).Low()), 0xFC00u);
    EXPECT_EQ(F16Bits(Half2Div(Half2(zero, zero), Half2(minusInf, minusInf)).Low()), 0x8000u);
    EXPECT_EQ(F16Bits(Half2Add(Half2(plusInf, plusInf), Half2(plusInf, plusInf)).Low()), 0x7C00u);
}

TEST(NativeHalfLaneTest, PackedHalvesDoNotInterfere) {
    // One operation, two values: the low half's result must not depend on the
    // high half's, which is what makes the packed shape worth having.
    std::mt19937_64 rng(0xC0FFEEu);
    const char ops[] = {'+', '-', '*', '/'};

    for (char op : ops)
    {
        for (int i = 0; i < 20000; ++i)
        {
            const F16 a = F16FromBits(static_cast<std::uint16_t>(rng() % 0x7C00u));
            const F16 b = F16FromBits(static_cast<std::uint16_t>(1u + rng() % 0x7BFFu));
            const F16 other = F16FromBits(static_cast<std::uint16_t>(rng() % 0x7C00u));

            const Half2 pair = op == '+'   ? Half2Add(Half2(a, other), Half2(b, other))
                               : op == '-' ? Half2Sub(Half2(a, other), Half2(b, other))
                               : op == '*' ? Half2Mul(Half2(a, other), Half2(b, other))
                                           : Half2Div(Half2(a, other), Half2(b, other));

            const Half2 scalar = op == '+'   ? Half2Add(Half2(a, other), Half2(b, b))
                                 : op == '-' ? Half2Sub(Half2(a, other), Half2(b, b))
                                 : op == '*' ? Half2Mul(Half2(a, other), Half2(b, b))
                                             : Half2Div(Half2(a, other), Half2(b, b));

            ASSERT_EQ(F16Bits(pair.Low()), F16Bits(scalar.Low()));
        }
    }
}

#if BoysFp16

// --- The lane ---------------------------------------------------------------
// Everything below calls BoysAllOrdersHalf2 or BoysAllNF16Native, or reads the scale constant
// they document, and is compiled with the seam that declares them.

TEST(NativeHalfLaneTest, LaneMatchesTheCommittedReferenceGrid) {
    // The grid's region-C arguments exactly representable in half: there the 45-digit value is
    // the reference and no other lane is involved.
    const std::vector<GridRow> grid = LoadReferenceGrid();
    ASSERT_FALSE(grid.empty());

    std::array<double, kMaxBoysOrder + 1> worst{};
    std::array<double, kMaxBoysOrder + 1> worstX{};
    int checked = 0;

    for (const GridRow& row : grid)
    {
        if (row.x < Value(BoundaryArgument()) || row.x != Value(F16FromDouble(row.x)))
        {
            continue;
        }

        ++checked;
        std::vector<Half2> out(static_cast<std::size_t>(row.n) + 1);
        boys::BoysAllOrdersHalf2(
            row.n, Half2(F16FromDouble(row.x), F16FromDouble(row.x)), out.data());

        const F16 value = out[static_cast<std::size_t>(row.n)].Low();
        const double scaled = Value(value);
        const double error = std::fabs(scaled - Scaled(row.value));
        const double ratio = error / Quantum(value);

        if (ratio > worst[static_cast<std::size_t>(row.n)])
        {
            worst[static_cast<std::size_t>(row.n)] = ratio;
            worstX[static_cast<std::size_t>(row.n)] = row.x;
        }
    }

    std::printf("[grid] %d region-C rows of the 45-digit grid\n", checked);
    std::printf("[grid] order  max err / ULP   at x\n");

    double worstRatio = 0.0;
    int worstOrder = 0;
    for (int n = 0; n <= kMaxBoysOrder; ++n)
    {
        if (worst[static_cast<std::size_t>(n)] == 0.0)
        {
            continue;
        }

        std::printf("[grid] %5d  %12.3f   %.6g\n",
                    n,
                    worst[static_cast<std::size_t>(n)],
                    worstX[static_cast<std::size_t>(n)]);

        if (worst[static_cast<std::size_t>(n)] > worstRatio)
        {
            worstRatio = worst[static_cast<std::size_t>(n)];
            worstOrder = n;
        }
    }

    std::printf("[grid] worst %.3f ULP at order %d\n", worstRatio, worstOrder);
    EXPECT_LE(worstRatio, kBoundUlps);
}

TEST(NativeHalfLaneTest, LaneStaysInsideItsBoundAcrossRegionC) {
    // native-half-bound, over the whole of region C against the certified double lane, per
    // order: the maximum error in quanta of the returned value. The domain is the arguments
    // whose returned value is a normal half; the rest are the no-claim domain, counted and
    // reported, carrying no claim - past the ceiling the entry returns a subnormal and then a
    // zero by design, so a claim there would be met by the format's floor, not by the lane.
    const std::vector<F16> xs = SweepArguments();
    const int nmax = kMaxBoysOrder;

    std::array<double, kMaxBoysOrder + 1> worst{};
    std::array<double, kMaxBoysOrder + 1> worstNoClaim{};
    std::array<double, kMaxBoysOrder + 1> worstX{};
    std::array<double, kMaxBoysOrder + 1> worstTrueUlps{};
    std::array<long, kMaxBoysOrder + 1> noClaim{};
    std::array<long, kMaxBoysOrder + 1> overHalf{};

    std::vector<Half2> out(static_cast<std::size_t>(nmax) + 1);

    for (F16 x : xs)
    {
        boys::BoysAllOrdersHalf2(nmax, Half2(x, x), out.data());
        const std::vector<double> reference = Reference(nmax, x);

        for (int k = 0; k <= nmax; ++k)
        {
            const std::size_t index = static_cast<std::size_t>(k);
            const F16 value = out[index].Low();
            const double scaled = Value(value);
            const double truth = Scaled(reference[index]);
            const double error = std::fabs(scaled - truth);
            const double ratio = error / Quantum(value);

            if (std::fabs(scaled) < kSmallestNormal)
            {
                ++noClaim[index];
                ++gBound.noClaim;
                gBound.largestNoClaimTruth =
                    std::max(gBound.largestNoClaimTruth, truth / kSmallestNormal);

                if (ratio > worstNoClaim[index])
                {
                    worstNoClaim[index] = ratio;
                }

                continue;
            }

            ++gBound.claimed;

            if (ratio > 0.5)
            {
                ++overHalf[index];
            }

            if (ratio > worst[index])
            {
                worst[index] = ratio;
                worstX[index] = Value(x);
                worstTrueUlps[index] = error / std::ldexp(1.0, std::ilogb(truth) - 10);
            }
        }
    }

    std::printf("[sweep] %zu arguments from the region-C boundary to the format maximum\n",
                xs.size());
    std::printf("[sweep] order  max err/ULP   at x        err/ULP(true)  >0.5ULP   no-claim\n");

    double worstRatio = 0.0;
    int worstOrder = 0;
    double worstNoClaimRatio = 0.0;

    for (int k = 0; k <= nmax; ++k)
    {
        const std::size_t index = static_cast<std::size_t>(k);
        worstNoClaimRatio = std::max(worstNoClaimRatio, worstNoClaim[index]);

        if (noClaim[index] == static_cast<long>(xs.size()))
        {
            std::printf("[sweep] %5d  (every output subnormal or zero: %ld, no claim)\n",
                        k,
                        noClaim[index]);
            continue;
        }

        std::printf("[sweep] %5d  %12.3f   %-9.6g  %12.3f   %6.3f    %ld\n",
                    k,
                    worst[index],
                    worstX[index],
                    worstTrueUlps[index],
                    static_cast<double>(overHalf[index]) / static_cast<double>(xs.size()),
                    noClaim[index]);

        if (worst[index] > worstRatio)
        {
            worstRatio = worst[index];
            worstOrder = k;
        }
    }

    gBound.worstUlps = worstRatio;
    gBound.worstOrder = worstOrder;
    gBound.worstX = worstX[static_cast<std::size_t>(worstOrder)];

    std::printf("[sweep] claim domain %ld points, no-claim domain %ld points\n",
                gBound.claimed,
                gBound.noClaim);
    std::printf("[sweep] worst %.3f ULP at order %d, x = %.6g; observed in the no-claim band "
                "%.3f ULP (reported, not claimed)\n",
                gBound.worstUlps,
                gBound.worstOrder,
                gBound.worstX,
                worstNoClaimRatio);

    EXPECT_GT(gBound.claimed, 0);
    EXPECT_LE(gBound.worstUlps, kBoundUlps);

    // The domain restriction, checked rather than assumed: the no-claim points are the ones at
    // or below the ceiling (the format's smallest normal, crossed, within the lane's own
    // error), so the claim stops where the arithmetic says it does.
    EXPECT_LE(gBound.largestNoClaimTruth, 1.0 + kBoundUlps * std::ldexp(1.0, -10));
}

TEST(NativeHalfLaneTest, LaneRoundsPerOperationNotOncePerValue) {
    // The bound alone does not say where the error comes from; this does. A lane that widens,
    // evaluates and rounds once at the end is inside half an ULP of the correctly rounded value
    // at every argument, and this one is not, at a large fraction of them.
    const int nmax = 8;
    const std::vector<F16> xs = SweepArguments();
    std::vector<Half2> out(static_cast<std::size_t>(nmax) + 1);

    long notCorrectlyRounded = 0;
    long beyondOneUlp = 0;
    long total = 0;
    double maxError = 0.0;

    for (F16 x : xs)
    {
        boys::BoysAllOrdersHalf2(nmax, Half2(x, x), out.data());
        const std::vector<double> reference = Reference(nmax, x);

        for (int k = 0; k <= nmax; ++k)
        {
            const F16 value = out[static_cast<std::size_t>(k)].Low();
            const double scaled = Value(value);
            const double truth = Scaled(reference[static_cast<std::size_t>(k)]);
            const double error = std::fabs(scaled - truth);

            if (std::fabs(Value(value)) < std::ldexp(1.0, -14))
            {
                continue;
            }

            ++total;
            maxError = std::max(maxError, error / Quantum(value));

            // Half a quantum of the true value is what one rounding to nearest can leave: more
            // than that cannot come from a single rounding.
            if (error > 0.5 * QuantumAt(std::fabs(truth)))
            {
                ++notCorrectlyRounded;
            }

            if (error > QuantumAt(std::fabs(truth)))
            {
                ++beyondOneUlp;
            }
        }
    }

    const double fraction = static_cast<double>(notCorrectlyRounded) / static_cast<double>(total);
    gPacked.beyondHalfUlp = notCorrectlyRounded;
    gPacked.values = total;
    gPacked.maxUlps = maxError;
    std::printf("[rounding] %ld of %ld values (%.1f%%) are past half an ULP, %ld (%.1f%%) are past "
                "one ULP; max error %.3f ULP\n",
                notCorrectlyRounded,
                total,
                100.0 * fraction,
                beyondOneUlp,
                total == 0 ? 0.0
                           : 100.0 * static_cast<double>(beyondOneUlp) / static_cast<double>(total),
                maxError);

    // A lane that rounds once per value cannot pass half an ULP anywhere; this one passes it at
    // nearly half the arguments and one ULP at hundreds: the roundings are inside the ladder.
    EXPECT_GT(maxError, 1.0);
    EXPECT_GT(fraction, 0.25);
    EXPECT_GT(beyondOneUlp, 0);
    EXPECT_LE(maxError, kBoundUlps);
}

TEST(NativeHalfLaneTest, TheScaleReachesWhereTheUnscaledLadderCannot) {
    // The scale 2^15 is exact, so it buys no accuracy but range: the largest argument whose
    // output is still a normal half, i.e. where 2^15 F_k(x) crosses the smallest normal 2^-14,
    // i.e. where F_k(x) crosses 2^-29. As one equation, three readings: 2^15 F_k(x) = 2^-14,
    // that is F_k(x) = 2^-29.
    //
    // The same test against the unscaled ladder (scale 2^0, the shipped region-C body) crosses
    // at F_k = 2^-14, which at orders 3 and up is left of the region-C boundary: unscaled,
    // those orders never carry a normal value in region C at all.
    struct Reach {
        int order;
        double expected;
    };

    const Reach reaches[] = {{2, 2640.0}, {3, 360.8}, {4, 128.8}, {8, 30.17}};

    const int nmax = 8;
    std::vector<Half2> out(static_cast<std::size_t>(nmax) + 1);
    const std::vector<F16> xs = SweepArguments();

    for (const Reach& reach : reaches)
    {
        int largest = -1;
        for (F16 x : xs)
        {
            boys::BoysAllOrdersHalf2(nmax, Half2(x, x), out.data());
            const double scaled =
                std::fabs(Value(out[static_cast<std::size_t>(reach.order)].Low()));

            if (scaled >= std::ldexp(1.0, -14))
            {
                largest = HalfIndex(F16Bits(x));
            }
        }

        ASSERT_GE(largest, 0) << "order " << reach.order;
        const double crossing = Value(HalfBitsAt(largest));
        gCeiling.crossing[static_cast<std::size_t>(reach.order)] = crossing;
        std::printf("[reach] order %d: normal output up to x = %.6g (expected %.6g)\n",
                    reach.order,
                    crossing,
                    reach.expected);

        EXPECT_GT(crossing, reach.expected * 0.97);
        EXPECT_LT(crossing, reach.expected * 1.03);

        // The reference has to agree that the crossing argument carries a value at the ceiling,
        // within the lane's own error, so the claim stops where the arithmetic says rather than
        // at an argument the sweep happened to reach. The crossing return can be normal while
        // the true value has just crossed: the lane returns a rounded value.
        const std::vector<double> atCrossing =
            Reference(reach.order, F16FromBits(HalfBitsAt(largest)));
        EXPECT_GE(Scaled(atCrossing[static_cast<std::size_t>(reach.order)]),
                  kSmallestNormal * (1.0 - kBoundUlps * std::ldexp(1.0, -10)));

        // The unscaled ladder (scale 2^0) at the same order: from order 3 up its crossing is
        // left of the region-C boundary, so it has no normal output anywhere in region C and the
        // scale is the whole of the range there. At order 2 it still has one.
        if (reach.order >= 3)
        {
            const std::vector<double> atBoundary = Reference(reach.order, BoundaryArgument());
            EXPECT_LT(atBoundary[static_cast<std::size_t>(reach.order)], std::ldexp(1.0, -14));
        }
    }

    // Orders 0 and 1 never reach: their crossing sits above the format's maximum, 65504.
    for (int k = 0; k <= 1; ++k)
    {
        int largest = -1;
        for (F16 x : xs)
        {
            boys::BoysAllOrdersHalf2(nmax, Half2(x, x), out.data());
            if (std::fabs(Value(out[static_cast<std::size_t>(k)].Low())) >= std::ldexp(1.0, -14))
            {
                largest = HalfIndex(F16Bits(x));
            }
        }

        EXPECT_EQ(largest, HalfIndex(F16Bits(xs.back()))) << "order " << k;
    }
}

TEST(NativeHalfLaneTest, TheSpanSetsTheWallNoScalePasses) {
    // The other end of the range statement. One scale carries the whole ladder, F_0 at the top
    // down to F_k at the bottom, so the two ends must fit the format's range together: F_0/F_k
    // passes the format's normal span, 2^29, at about x = 38 at order 8, where the scale 2^15's
    // own reach is x = 30.17. The two limits are within 25% of each other, so a larger scale
    // has nothing left to win.
    const int orders[] = {3, 4, 5, 6, 8, 12};
    const double spanLimit = std::ldexp(1.0, 29);
    std::vector<Half2> out(13);

    for (int order : orders)
    {
        // The span along the sweep, and the largest argument still inside the limit.
        int largest = -1;
        double crossing = 0.0;

        for (F16 x : SweepArguments())
        {
            const std::vector<double> reference = Reference(order, x);
            const double span = reference[0] / reference[static_cast<std::size_t>(order)];

            if (span <= spanLimit)
            {
                largest = HalfIndex(F16Bits(x));
                crossing = Value(x);
            }
        }

        if (largest < 0)
        {
            std::printf("[span]  order %2d: the span is past 2^29 before the boundary\n", order);
            continue;
        }

        std::printf("[span]  order %2d: F_0/F_%d <= 2^29 up to x = %.6g\n", order, order, crossing);
    }

    // The wall at order 8, which is the case the range statement is about.
    int largest = -1;
    double crossing = 0.0;
    for (F16 x : SweepArguments())
    {
        const std::vector<double> reference = Reference(8, x);
        if (reference[0] / reference[8] <= spanLimit)
        {
            largest = HalfIndex(F16Bits(x));
            crossing = Value(x);
        }
    }

    ASSERT_GE(largest, 0);
    gCeiling.spanWall = crossing;
    gCeiling.spanOrder = 8;
    EXPECT_GT(crossing, 34.0);
    EXPECT_LT(crossing, 42.0);

    // The case the entry's documentation names: order 8 at x = 400 is past the ceiling at that
    // order, so the return is a subnormal and then a zero by design, with no accuracy claimed.
    // What is checked is that the argument is in that domain, and the difference is reported.
    {
        const F16 argument = F16FromDouble(400.0);
        std::vector<Half2> out8(9);
        boys::BoysAllOrdersHalf2(8, Half2(argument, argument), out8.data());

        const F16 value = out8[8].Low();
        const double truth = Scaled(Reference(8, argument)[8]);
        std::printf("[reach] order 8, x = 400: bits 0x%04X, true scaled value %.6g (%.4g of the "
                    "smallest normal; no claim past it)\n",
                    F16Bits(value),
                    truth,
                    truth / kSmallestNormal);

        EXPECT_LT(std::fabs(Value(value)), kSmallestNormal);
        EXPECT_LT(truth, kSmallestNormal);
    }
}

TEST(NativeHalfLaneTest, BatchEntryIsThePackedEntryPerArgument) {
    const int nmax = 8;
    const std::vector<F16> xs = SweepArguments();

    std::vector<F16> arguments;
    for (std::size_t i = 0; i < xs.size() && i < 97; i += 2)
    {
        arguments.push_back(xs[i]); // an odd count, so the lone tail runs too
    }

    const std::size_t count = arguments.size();
    std::vector<F16> batch(count * static_cast<std::size_t>(nmax + 1));
    boys::BoysAllNF16Native(nmax, arguments.data(), batch.data(), count);

    std::vector<Half2> out(static_cast<std::size_t>(nmax) + 1);

    for (std::size_t i = 0; i < count; ++i)
    {
        boys::BoysAllOrdersHalf2(nmax, Half2(arguments[i], arguments[i]), out.data());

        for (int k = 0; k <= nmax; ++k)
        {
            ASSERT_EQ(F16Bits(batch[static_cast<std::size_t>(k) * count + i]),
                      F16Bits(out[static_cast<std::size_t>(k)].Low()))
                << "argument " << i << " order " << k;
        }
    }

    // The pairwise entry packs two arguments into one register: a pair fed
    // through it must agree with the two scalar runs half by half.
    std::vector<F16> pairArguments = {arguments[0], arguments[1]};
    std::vector<F16> pairOut(2 * static_cast<std::size_t>(nmax + 1));
    boys::BoysAllNF16Native(nmax, pairArguments.data(), pairOut.data(), pairArguments.size());

    for (int k = 0; k <= nmax; ++k)
    {
        for (std::size_t i = 0; i < pairArguments.size(); ++i)
        {
            boys::BoysAllOrdersHalf2(nmax, Half2(pairArguments[i], pairArguments[i]), out.data());
            ASSERT_EQ(F16Bits(pairOut[static_cast<std::size_t>(k) * 2 + i]),
                      F16Bits(out[static_cast<std::size_t>(k)].Low()));
        }
    }
}

// --- The claims, graded -------------------------------------------------------

// The book this lane publishes: three claims, each with its own domain and exactly one verdict,
// in the accuracy gate's own shape. A claim whose measurement did not run reports evidence
// absent rather than a pass - the gate's --strict rule - so no verdict rests on a skipped run.
TEST(NativeHalfLaneTest, ClaimBookGivesEachClaimOneVerdict) {
    // native-half-bound.
    Verdict boundVerdict = Verdict::EvidenceAbsent;
    std::string boundEvidence = "the region-C sweep did not run in this invocation";
    if (gBound.claimed > 0)
    {
        boundVerdict = gBound.worstUlps > kBoundUlps ? Verdict::Exceeded : Verdict::Verified;
        boundEvidence = Fmt("worst %.3f ULP of %.0f at order %d, x = %.6g, over %ld points; %ld "
                            "points past the ceiling, counted and unclaimed",
                            gBound.worstUlps,
                            kBoundUlps,
                            gBound.worstOrder,
                            gBound.worstX,
                            gBound.claimed,
                            gBound.noClaim);
    }

    // native-half-ceiling.
    Verdict ceilingVerdict = Verdict::EvidenceAbsent;
    std::string ceilingEvidence = "the ceiling scan did not run in this invocation";
    if (gCeiling.spanWall > 0.0)
    {
        ceilingVerdict = Verdict::Verified;
        ceilingEvidence =
            Fmt("orders 0 and 1 none inside the format; order 2 x = %.6g, 3 x = %.6g, 4 x = %.6g, "
                "8 x = %.6g; span wall at order %d, x = %.6g",
                gCeiling.crossing[2],
                gCeiling.crossing[3],
                gCeiling.crossing[4],
                gCeiling.crossing[8],
                gCeiling.spanOrder,
                gCeiling.spanWall);
    }

    // native-half-packed.
    Verdict packedVerdict = Verdict::EvidenceAbsent;
    std::string packedEvidence = "the operation checks did not run in this invocation";
    if (gPacked.sqrtCases > 0 && gPacked.opCases > 0 && gPacked.randomCases > 0 &&
        gPacked.values > 0)
    {
        packedVerdict = gPacked.maxUlps > kBoundUlps ? Verdict::Exceeded : Verdict::Verified;
        packedEvidence =
            Fmt("%ld square roots exhaustive, %ld curated operation decisions, %ld "
                "random pairs, no counterexample; %ld of %ld values (%.1f%%) past half "
                "an ULP, max %.3f ULP",
                gPacked.sqrtCases,
                gPacked.opCases,
                gPacked.randomCases,
                gPacked.beyondHalfUlp,
                gPacked.values,
                100.0 * static_cast<double>(gPacked.beyondHalfUlp) /
                    static_cast<double>(gPacked.values),
                gPacked.maxUlps);
    }

    struct Row {
        const char* id;
        const char* domain;
        Verdict verdict;
        std::string evidence;
    };

    const Row rows[] = {
        {"native-half-bound", "region C, normal return", boundVerdict, boundEvidence},
        {"native-half-ceiling", "argument ceiling per order", ceilingVerdict, ceilingEvidence},
        {"native-half-packed", "every operation, packed", packedVerdict, packedEvidence},
    };

    std::printf("[book] claim               domain                    verdict\n");
    for (const Row& row : rows)
    {
        std::printf("[book] %-19s %-25s %s\n", row.id, row.domain, VerdictName(row.verdict));
        std::printf("[book]   %s\n", row.evidence.c_str());
        EXPECT_NE(row.verdict, Verdict::Exceeded) << row.id;
        EXPECT_NE(row.verdict, Verdict::Vacuous) << row.id;
    }

    // In a full run every claim's evidence is the measurement above it, so every verdict is
    // verified - and a claim that stopped being met fails here as the one it is.
    if (gBound.claimed > 0)
    {
        EXPECT_EQ(boundVerdict, Verdict::Verified);
        EXPECT_EQ(ceilingVerdict, Verdict::Verified);
        EXPECT_EQ(packedVerdict, Verdict::Verified);
    } else
    {
        std::printf(
            "[book] the measurements did not run in this invocation: the verdicts above are "
            "evidence absent, which is not a pass\n");
    }
}

#else // BoysFp16

// The other side of the seam: this build carries no native half lane, so there is no
// measurement to make and no claim to judge. A skipped test says exactly that - neither the
// lane's verdict nor a green.
TEST(NativeHalfLaneTest, LaneIsNotCarriedByThisBuild) {
    GTEST_SKIP() << "the native half lane is declared behind the BoysFp16 seam, which this build "
                    "has closed (BoysFp16 = 0): no entry to measure, no claim to judge";
}

#endif // BoysFp16

} // namespace
