// Structural invariants of the Boys family: positivity, order monotonicity, the adjacent-order
// recurrence (2n+1) F_n - 2x F_(n+1) - e^(-x) = 0 and log-convexity F_(n+1)^2 <= F_n F_(n+2),
// tested on the shipped entries. All four are exact for
//   F_n(x) = integral of t^(2n) e^(-x t^2) over [0, 1],  n >= 0,  x >= 0.

// Each is asserted on the interval the lane's published bound gives the returned value, and the
// strict form only where the exact values and those intervals certify it. The reference grid fixes
// the exact values; the designed sweep crosses every piece, route, lane and packing axis, with the
// suite's twofold two-rounding allowance. Floor-limited and bound-limited cells are counted apart.

// The all-groups umbrella opens every coefficient group before the front door: the policy sweep
// below names fits from all three partitions and both routes, and a group header has to precede
// the front door in the translation unit that names its policy.
#include "boys/boys_all.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"
#include "boys/half2.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>
#include <limits>
#include <map>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kNmax = boys::kMaxBoysOrder;

// The format floors, the domain below which the specification claims nothing: 2^-1022, 2^-126,
// 2^-14 and 2^-126. A value at or above its lane's floor is a claimed cell; below it the return
// may be subnormal or zero.
constexpr double kDoubleFloor = std::numeric_limits<double>::min();
constexpr double kFloatFloor = std::numeric_limits<float>::min();
constexpr double kHalfFloor = 6.103515625e-05; // 2^-14
constexpr double kBf16Floor = 1.1754943508222875e-38; // 2^-126

// The published figures are the fused route's; the two-rounding build is another arithmetic and
// the suite's own comparisons there use a twofold budget. These structural checks carry the same
// route allowance, not a second accuracy claim: the gate remains the measured figure.
#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
constexpr double kRouteAllowance = 2.0;
#else
constexpr double kRouteAllowance = 1.0;
#endif

constexpr double kEpsilon = std::numeric_limits<double>::epsilon();

// F_0 from its closed form, in double: the scale of the published bound, not a reference value.
// The test's own erf is accurate far inside the 1e-13 figure it scales.
double F0Closed(double x) noexcept {
    if (x == 0.0)
    {
        return 1.0;
    }

    // The closed form is 1 - x/3 + O(x^2) near zero, and the direct quotient pi/x overflows
    // before the erf factor can cancel it. The bound is a scale, so the first terms are ample
    // there: the omitted term is x^2/10, below 1e-17 at this cut.
    if (x < 1e-8)
    {
        return 1.0 - x / 3.0;
    }

    return 0.5 * std::sqrt(std::numbers::pi / x) * std::erf(std::sqrt(x));
}

// The published double lane's figure: 1e-13 * F_0(x), the wider arm of the pair. F falls with the
// order, so the F_n arm never binds.
double DoubleBound(double x) noexcept {
    return kRouteAllowance * 1e-13 * F0Closed(x);
}

// The binary16 quantum at a magnitude: 2^(e-10) normal, 2^-24 at or below the smallest normal.
double HalfQuantum(double magnitude) noexcept {
    if (!(magnitude >= kHalfFloor))
    {
        return std::ldexp(1.0, -24);
    }

    return std::ldexp(1.0, std::ilogb(magnitude) - 10);
}

// The bfloat16 quantum: 2^(e-7) normal, 2^-133 at or below 2^-126.
double Bf16Quantum(double magnitude) noexcept {
    if (!(magnitude >= kBf16Floor))
    {
        return std::ldexp(1.0, -133);
    }

    return std::ldexp(1.0, std::ilogb(magnitude) - 7);
}

// The float lane's figure at a cell: max(2^-22 F_n, 1.5e-7), plus the plain reciprocal's 1e-7 where
// that form is the one named. The value passed is the lane's argument and the double lane's reading
// of F_n there, which is inside the float lane's figure by nine orders of magnitude.
double FloatBound(int n, double x, bool plain) noexcept {
    const double fn = std::fabs(boys::BoysSingle(n, x));

    return kRouteAllowance * (std::max(std::ldexp(1.0, -22) * fn, 1.5e-7) +
                              (plain ? 1e-7 : 0.0));
}

// The half lanes' figure: the fp32 engine stated absolutely, plus the format's own half digit.
// The plain reciprocal's term is the one the lane's row publishes beside its base.
double HalfBound(double observed, bool bf16, bool plain) noexcept {
    const double quantum = bf16 ? Bf16Quantum(observed) : HalfQuantum(observed);

    return kRouteAllowance * (1.5e-7 + (plain ? 1e-7 : 0.0) + 0.5 * quantum);
}

// --- The committed reference grid -------------------------------------------

// Columns 0..12 are n, x, value, ..., xf, valuef, ..., x16, value16, ..., xb, valueb, ...; each
// value column is the exact F_n at the argument before it, so a lane is compared at the argument
// its own type holds rather than at the double argument.

struct LaneTruth {
    double argument = 0.0;
    std::array<double, kNmax + 1> value{};
};

struct RefGrid {
    std::size_t rows = 0;
    std::map<double, std::array<double, kNmax + 1>> doubleLadders;
    std::map<std::uint32_t, LaneTruth> floatLadders;
    std::map<std::uint16_t, LaneTruth> f16Ladders;
    std::map<std::uint16_t, LaneTruth> bf16Ladders;
};

std::vector<std::string> SplitCsv(const std::string& line) {
    std::vector<std::string> cells;
    std::string cell;
    std::istringstream stream(line);

    while (std::getline(stream, cell, ','))
    {
        cells.push_back(cell);
    }

    return cells;
}

const RefGrid& Reference() {
    static const RefGrid grid = [] {
        RefGrid loaded;
        const std::string path = std::string(BoysDataDir) + "/boys_accuracy_gate_reference.csv";
        std::ifstream file(path);

        if (!file)
        {
            return loaded;
        }

        std::string line;
        std::getline(file, line); // header

        while (std::getline(file, line))
        {
            const std::vector<std::string> cells = SplitCsv(line);

            if (cells.size() < 12)
            {
                continue;
            }

            const int n = std::stoi(cells[0]);
            const double x = std::strtod(cells[1].c_str(), nullptr);
            const double value = std::strtod(cells[2].c_str(), nullptr);
            const float xf = std::strtof(cells[4].c_str(), nullptr);
            const double valuef = std::strtod(cells[5].c_str(), nullptr);
            const double x16 = std::strtod(cells[7].c_str(), nullptr);
            const double value16 = std::strtod(cells[8].c_str(), nullptr);
            const double xb = std::strtod(cells[10].c_str(), nullptr);
            const double valueb = std::strtod(cells[11].c_str(), nullptr);

            if (n < 0 || n > kNmax)
            {
                continue;
            }

            ++loaded.rows;
            loaded.doubleLadders[x][static_cast<std::size_t>(n)] = value;

            if (std::isfinite(xf))
            {
                LaneTruth& lane = loaded.floatLadders[std::bit_cast<std::uint32_t>(xf)];
                lane.argument = static_cast<double>(xf);
                lane.value[static_cast<std::size_t>(n)] = valuef;
            }

            if (std::isfinite(x16))
            {
                const boys::F16 half = boys::F16(static_cast<float>(x16));
                LaneTruth& lane = loaded.f16Ladders[boys::detail::F16Bits(half)];
                lane.argument = static_cast<double>(half);
                lane.value[static_cast<std::size_t>(n)] = value16;
            }

            if (std::isfinite(xb))
            {
                const boys::Bf16 bfloat = boys::Bf16(static_cast<float>(xb));
                LaneTruth& lane = loaded.bf16Ladders[boys::detail::Bf16Bits(bfloat)];
                lane.argument = static_cast<double>(bfloat);
                lane.value[static_cast<std::size_t>(n)] = valueb;
            }
        }

        return loaded;
    }();

    return grid;
}

// --- The tally ---------------------------------------------------------------

struct Invariants {
    std::size_t ladders = 0;
    std::size_t values = 0;
    std::size_t claimed = 0;
    std::size_t floorLimited = 0;
    std::size_t boundLimited = 0;
    std::size_t nonFinite = 0;
    std::size_t negative = 0;
    std::size_t nonPositive = 0;

    std::size_t strictMono = 0;
    std::size_t monoProtected = 0;
    std::size_t monoViolations = 0;
    double worstMonoDeficit = 0.0;

    std::size_t strictLogCvx = 0;
    std::size_t logCvxUnresolved = 0;
    std::size_t logCvxViolations = 0;
    double worstLogCvxRatio = 0.0;
    int worstLogCvxN = 0;
    double worstLogCvxX = 0.0;

    std::size_t recurrenceChecked = 0;
    std::size_t recurrenceViolations = 0;
    double worstRecurrenceFraction = 0.0;
    int worstRecurrenceN = 0;
    double worstRecurrenceX = 0.0;
    const char* worstRecurrenceLane = "";
    double worstViolationFraction = 0.0;
    int violationN = 0;
    double violationX = 0.0;
    const char* violationLane = "";
};

void Merge(Invariants& into, const Invariants& from) {
    into.ladders += from.ladders;
    into.values += from.values;
    into.claimed += from.claimed;
    into.floorLimited += from.floorLimited;
    into.boundLimited += from.boundLimited;
    into.nonFinite += from.nonFinite;
    into.negative += from.negative;
    into.nonPositive += from.nonPositive;
    into.strictMono += from.strictMono;
    into.monoProtected += from.monoProtected;
    into.monoViolations += from.monoViolations;
    into.worstMonoDeficit = std::max(into.worstMonoDeficit, from.worstMonoDeficit);
    into.strictLogCvx += from.strictLogCvx;
    into.logCvxUnresolved += from.logCvxUnresolved;
    into.logCvxViolations += from.logCvxViolations;

    if (from.worstLogCvxRatio > into.worstLogCvxRatio)
    {
        into.worstLogCvxRatio = from.worstLogCvxRatio;
        into.worstLogCvxN = from.worstLogCvxN;
        into.worstLogCvxX = from.worstLogCvxX;
    }

    into.recurrenceChecked += from.recurrenceChecked;
    into.recurrenceViolations += from.recurrenceViolations;

    if (from.worstRecurrenceFraction > into.worstRecurrenceFraction)
    {
        into.worstRecurrenceFraction = from.worstRecurrenceFraction;
        into.worstRecurrenceN = from.worstRecurrenceN;
        into.worstRecurrenceX = from.worstRecurrenceX;
    }

    if (from.worstViolationFraction > into.worstViolationFraction)
    {
        into.worstViolationFraction = from.worstViolationFraction;
        into.violationN = from.violationN;
        into.violationX = from.violationX;
        into.violationLane = from.violationLane;
    }
}

// One ladder and the intervals its values sit in. `truth` is null on the designed sweep, where the
// exact values are not carried: the bound-aware form is still a statement about the exact family,
// but the strict form needs the exact gap and is skipped.
struct LadderView {
    const char* lane = "";
    double x = 0.0;
    double floor = 0.0;
    double expScale = 1.0;
    const double* truth = nullptr;
    const double* value = nullptr;
    const double* bound = nullptr;
    int nmax = kNmax;
};

Invariants CheckLadder(const LadderView& view) {
    Invariants tally;
    ++tally.ladders;

    for (int n = 0; n <= view.nmax; ++n)
    {
        const double v = view.value[n];
        ++tally.values;

        if (!std::isfinite(v))
        {
            ++tally.nonFinite;
            continue;
        }

        if (v < 0.0)
        {
            ++tally.negative;
        }

        // The figure is absolute, so a zero is within it wherever the exact value is at or below.
        // Positivity is asserted only where the exact value is above both the format floor and the
        // figure; the two restrictions are counted apart.
        const bool aboveFloor = view.truth != nullptr ? (view.truth[n] >= view.floor)
                                                      : (v >= view.floor);
        const bool aboveBound = view.truth != nullptr ? (view.truth[n] > view.bound[n])
                                                      : (v > view.bound[n]);

        if (aboveFloor && aboveBound)
        {
            ++tally.claimed;

            if (!(v > 0.0))
            {
                ++tally.nonPositive;
            }
        } else if (!aboveFloor)
        {
            ++tally.floorLimited;
        } else
        {
            ++tally.boundLimited;
        }
    }

    // 2. Order monotonicity. The bound-aware form holds at every cell on which both values are
    // finite; the strict form is asserted where the exact gap and the two intervals certify it.
    for (int n = 0; n < view.nmax; ++n)
    {
        const double a = view.value[n];
        const double b = view.value[n + 1];
        const double ba = view.bound[n];
        const double bb = view.bound[n + 1];

        if (!std::isfinite(a) || !std::isfinite(b))
        {
            continue;
        }

        const double deficit = (b - a) - (ba + bb);
        tally.worstMonoDeficit = std::max(tally.worstMonoDeficit, deficit);

        if (deficit > 0.0)
        {
            ++tally.monoViolations;
        }

        if (view.truth != nullptr)
        {
            const double A = view.truth[n];
            const double B = view.truth[n + 1];
            // The reference grid is read as doubles, so a gap that sits on the bound to within
            // a double rounding is not certified. The margin is a scale of the values, far
            // below the gaps this grid holds where the bound leaves room.
            const double margin = 1e-12 * std::max({std::fabs(A), std::fabs(B), 1.0});

            if (A - ba > B + bb + margin)
            {
                ++tally.strictMono;

                if (!(a > b))
                {
                    ++tally.monoViolations;
                }
            } else
            {
                ++tally.monoProtected;
            }
        }
    }

    // 4. Log-convexity in order. The necessary condition on the intervals is
    // max(b - B_b, 0)^2 <= (a + B_a)(c + B_c); it is tested first. The strict form is asserted
    // where the exact values and the three intervals certify it.
    for (int n = 0; n + 2 <= view.nmax; ++n)
    {
        const double a = view.value[n];
        const double b = view.value[n + 1];
        const double c = view.value[n + 2];
        const double ba = view.bound[n];
        const double bb = view.bound[n + 1];
        const double bc = view.bound[n + 2];

        if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c) || !(a > 0.0) ||
            !(b > 0.0) || !(c > 0.0))
        {
            continue;
        }

        const long double lower = static_cast<long double>(std::max(b - bb, 0.0));
        const long double upperA = static_cast<long double>(a) + ba;
        const long double upperC = static_cast<long double>(c) + bc;
        const long double lhs = lower * lower;
        const long double rhs = upperA * upperC;
        const long double ratio = rhs > 0.0L ? lhs / rhs : 0.0L;

        if (static_cast<double>(ratio) > tally.worstLogCvxRatio)
        {
            tally.worstLogCvxRatio = static_cast<double>(ratio);
            tally.worstLogCvxN = n;
            tally.worstLogCvxX = view.x;
        }

        if (lhs > rhs * (1.0L + 64.0L * kEpsilon))
        {
            ++tally.logCvxViolations;
        }

        if (view.truth != nullptr)
        {
            const double A = view.truth[n];
            const double B = view.truth[n + 1];
            const double C = view.truth[n + 2];
            const long double exactLowerA = static_cast<long double>(A) - ba;
            const long double exactLowerC = static_cast<long double>(C) - bc;
            const long double exactUpperB = static_cast<long double>(B) + bb;

            if (exactLowerA > 0.0L && exactLowerC > 0.0L &&
                exactUpperB * exactUpperB <
                    exactLowerA * exactLowerC * (1.0L - 1.0e-12L))
            {
                ++tally.strictLogCvx;

                if (!(static_cast<long double>(b) * b <
                      static_cast<long double>(a) * c))
                {
                    ++tally.logCvxViolations;
                }
            } else
            {
                ++tally.logCvxUnresolved;
            }
        }
    }

    // 3. The adjacent-order recurrence. All three terms are positive, so the residual and the
    // denominator are formed without cancellation, and the bound's contribution normalises
    // cleanly. The last term is the test's rounding of the expression itself.
    for (int n = 0; n < view.nmax; ++n)
    {
        const double a = view.value[n];
        const double b = view.value[n + 1];
        const double ba = view.bound[n];
        const double bb = view.bound[n + 1];

        if (!std::isfinite(a) || !std::isfinite(b))
        {
            continue;
        }

        const double order = 2.0 * static_cast<double>(n) + 1.0;
        const double t1 = order * a;
        const double t2 = 2.0 * view.x * b;
        const double t3 = view.expScale * std::exp(-view.x);

        if (!std::isfinite(t1) || !std::isfinite(t2) || !std::isfinite(t3))
        {
            continue;
        }

        const double denominator = t1 + t2 + t3;

        if (!(denominator > 0.0))
        {
            continue;
        }

        const double residual = std::fabs(t1 - t2 - t3);
        const double tolerance =
            (order * ba + 2.0 * view.x * bb) / denominator + 64.0 * kEpsilon;
        const double fraction = residual / (denominator * tolerance);

        ++tally.recurrenceChecked;

        if (fraction > tally.worstRecurrenceFraction)
        {
            tally.worstRecurrenceFraction = fraction;
            tally.worstRecurrenceN = n;
            tally.worstRecurrenceX = view.x;
        }

        if (residual > tolerance * denominator)
        {
            ++tally.recurrenceViolations;

            if (fraction > tally.worstViolationFraction)
            {
                tally.worstViolationFraction = fraction;
                tally.violationN = n;
                tally.violationX = view.x;
                tally.violationLane = view.lane;
            }
        }
    }

    return tally;
}

void Report(const char* label, const Invariants& tally) {
    std::printf("  %-46s %7zu values | claimed %6zu floor-limited %6zu bound-limited %6zu | "
                "mono: strict %6zu, bound-only %6zu, violated %zu (worst %.2e) | log-convex: "
                "strict %6zu, unresolved %6zu, violated %zu (worst %.6f at n = %d, x = %.6g) | "
                "recurrence: %6zu, violated %zu (worst %.6f of tolerance at n = %d, x = %.6g)\n",
                label,
                tally.values,
                tally.claimed,
                tally.floorLimited,
                tally.boundLimited,
                tally.strictMono,
                tally.monoProtected,
                tally.monoViolations,
                tally.worstMonoDeficit,
                tally.strictLogCvx,
                tally.logCvxUnresolved,
                tally.logCvxViolations,
                tally.worstLogCvxRatio,
                tally.worstLogCvxN,
                tally.worstLogCvxX,
                tally.recurrenceChecked,
                tally.recurrenceViolations,
                tally.worstRecurrenceFraction,
                tally.worstRecurrenceN,
                tally.worstRecurrenceX);

    if (tally.recurrenceViolations > 0)
    {
        std::printf("    recurrence violation: %s, n = %d, x = %.17g, %.6f of tolerance\n",
                    tally.violationLane,
                    tally.violationN,
                    tally.violationX,
                    tally.worstViolationFraction);
    }
}

void ExpectClean(const Invariants& tally) {
    EXPECT_EQ(tally.nonFinite, 0u) << "a returned value is not finite";
    EXPECT_EQ(tally.negative, 0u) << "a returned value is negative";
    EXPECT_EQ(tally.nonPositive, 0u)
        << "a cell whose exact value is above the lane's floor returned zero or a negative";
    EXPECT_EQ(tally.monoViolations, 0u)
        << "order monotonicity is outside the two values' intervals";
    EXPECT_EQ(tally.logCvxViolations, 0u)
        << "log-convexity is outside the three values' intervals";
    EXPECT_EQ(tally.recurrenceViolations, 0u)
        << "the adjacent-order recurrence is outside the values' intervals";
}

// --- The designed boundary list ---------------------------------------------

// Every boundary the implementation dispatches on: region boundaries, partition piece edges, the
// route selector's thresholds, the uniform grid's edges, the region-B exponential's cut, a
// logarithmic interior sweep, and arguments past each format floor. Both sides are sampled: a
// property that holds in each body can still fail at the join.
const std::vector<double>& Boundaries() {
    static const std::vector<double> xs = [] {
        std::vector<double> list;

        const auto add = [&list](double value) {
            if (!(value >= 0.0) || !std::isfinite(value))
            {
                return;
            }

            list.push_back(value);
            list.push_back(std::nextafter(value, 0.0));
            list.push_back(std::nextafter(value, std::numeric_limits<double>::infinity()));
        };

        add(0.0);

        for (const boys::detail::OrderPiece& piece : boys::detail::kPieces)
        {
            add(piece.a);
            add(piece.b);
        }

        for (const boys::detail::f32::OrderPiece& piece : boys::detail::f32::kPieces)
        {
            add(static_cast<double>(piece.a));
            add(static_cast<double>(piece.b));
        }

        for (const boys::detail::OrderPiece& piece : boys::detail::kNarrowAPieces)
        {
            add(piece.a);
            add(piece.b);
        }

        for (const boys::detail::f32::OrderPiece& piece : boys::detail::f32::kNarrowAPiecesF32)
        {
            add(static_cast<double>(piece.a));
            add(static_cast<double>(piece.b));
        }

        for (const double edge : boys::detail::kNarrowBEdges)
        {
            add(edge);
        }

        for (const float edge : boys::detail::f32::kNarrowBEdgesF32)
        {
            add(static_cast<double>(edge));
        }

        add(boys::detail::kExtendedBX0);
        add(boys::detail::kX0);
        add(boys::detail::kX1);
        add(boys::detail::kFlatHi);
        add(static_cast<double>(boys::detail::f32::kFlatHiF32));
        add(boys::detail::kSeedCorrectionX1);
        add(boys::detail::kRegionBExpCheapFrom);

        for (const double threshold : boys::detail::kTierThresholds)
        {
            add(threshold);
        }

        for (int k = -2; k <= 2; ++k)
        {
            add(boys::detail::kFlatHi + static_cast<double>(k) * boys::detail::kFlatWidth);
        }

        constexpr int kInteriorSteps = 160;

        for (int i = 0; i <= kInteriorSteps; ++i)
        {
            const double u = static_cast<double>(i) / static_cast<double>(kInteriorSteps);
            list.push_back(std::exp(std::log(1e-8) + u * (std::log(200.0) - std::log(1e-8))));
        }

        for (const double x : {100.0, 1e3, 1e6, 1e12, 1e20, 1e30, 1e38})
        {
            list.push_back(x);
        }

        std::sort(list.begin(), list.end());
        list.erase(std::unique(list.begin(), list.end()), list.end());
        list.erase(std::remove_if(list.begin(),
                                  list.end(),
                                  [](double x) { return !std::isfinite(x) || x < 0.0; }),
                   list.end());
        return list;
    }();

    return xs;
}

// --- The reference-grid sweeps ----------------------------------------------

Invariants CheckDefaultDouble() {
    Invariants total;

    for (const auto& [x, truth] : Reference().doubleLadders)
    {
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrders(kNmax, x, value.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            bound[n] = kRouteAllowance * 1e-13 * truth[0];
        }

        Merge(total, CheckLadder({"double, reference grid",
                                  x,
                                  kDoubleFloor,
                                  1.0,
                                  truth.data(),
                                  value.data(),
                                  bound.data(),
                                  kNmax}));
    }

    return total;
}

Invariants CheckDefaultFloat() {
    Invariants total;
    constexpr bool kPlain =
        boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;

    for (const auto& [key, truth] : Reference().floatLadders)
    {
        const float xf = std::bit_cast<float>(key);
        std::array<float, kNmax + 1> valuef{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrdersF32(kNmax, xf, valuef.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            value[n] = static_cast<double>(valuef[n]);
            bound[n] = kRouteAllowance *
                       (std::max(std::ldexp(1.0, -22) * truth.value[n], 1.5e-7) +
                        (kPlain ? 1e-7 : 0.0));
        }

        Merge(total, CheckLadder({"float, reference grid",
                                  truth.argument,
                                  kFloatFloor,
                                  1.0,
                                  truth.value.data(),
                                  value.data(),
                                  bound.data(),
                                  kNmax}));
    }

    return total;
}

#if BoysFp16

Invariants CheckDefaultF16(bool bf16) {
    Invariants total;
    constexpr bool kF16Plain =
        boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;
    constexpr bool kBf16Plain =
        boys::DefaultPolicy<boys::Precision::kBf16, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;
    const bool plain = bf16 ? kBf16Plain : kF16Plain;
    const double floor = bf16 ? kBf16Floor : kHalfFloor;
    const char* label = bf16 ? "bf16, reference grid" : "fp16, reference grid";

    if (bf16)
    {
        for (const auto& [key, truth] : Reference().bf16Ladders)
        {
            std::array<boys::Bf16, kNmax + 1> raw{};
            std::array<double, kNmax + 1> value{};
            std::array<double, kNmax + 1> bound{};
            boys::BoysAllOrdersBf16(kNmax, boys::detail::Bf16FromBits(key), raw.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                value[n] = static_cast<double>(raw[n]);
                bound[n] = HalfBound(value[n], true, plain);
            }

            Merge(total, CheckLadder({label,
                                      truth.argument,
                                      floor,
                                      1.0,
                                      truth.value.data(),
                                      value.data(),
                                      bound.data(),
                                      kNmax}));
        }
    } else
    {
        for (const auto& [key, truth] : Reference().f16Ladders)
        {
            std::array<boys::F16, kNmax + 1> raw{};
            std::array<double, kNmax + 1> value{};
            std::array<double, kNmax + 1> bound{};
            boys::BoysAllOrdersF16(kNmax, boys::detail::F16FromBits(key), raw.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                value[n] = static_cast<double>(raw[n]);
                bound[n] = HalfBound(value[n], false, plain);
            }

            Merge(total, CheckLadder({label,
                                      truth.argument,
                                      floor,
                                      1.0,
                                      truth.value.data(),
                                      value.data(),
                                      bound.data(),
                                      kNmax}));
        }
    }

    return total;
}

Invariants CheckNativeHalf() {
    Invariants total;
    constexpr int kNativeMax = 8;
    // Not constexpr: MSVC's STL does not make std::ldexp constexpr.
    const double kScale = std::ldexp(1.0, boys::kHalfNativeScaleExponent);

    for (const auto& [key, truth] : Reference().f16Ladders)
    {
        if (truth.argument < 28.984375)
        {
            continue;
        }

        std::array<boys::Half2, kNativeMax + 1> raw{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> scaledTruth{};
        std::array<double, kNmax + 1> bound{};
        const boys::F16 x = boys::detail::F16FromBits(key);
        boys::BoysAllOrdersHalf2(kNativeMax, boys::Half2(x, x), raw.data());

        for (int n = 0; n <= kNativeMax; ++n)
        {
            value[n] = static_cast<double>(raw[n].Low());
            scaledTruth[n] = kScale * truth.value[n];
            bound[n] = kRouteAllowance * 8.0 * HalfQuantum(value[n]);
        }

        Merge(total, CheckLadder({"native half, reference grid",
                                  truth.argument,
                                  kHalfFloor,
                                  kScale,
                                  scaledTruth.data(),
                                  value.data(),
                                  bound.data(),
                                  kNativeMax}));
    }

    return total;
}

#endif // BoysFp16

// --- The policy sweeps -------------------------------------------------------

// Whether the library carries a policy on its lane. A combination the library refuses is reported
// and not swept: the invariant is a statement about values a call returns, and a refused call has
// none. Every combination this file names is one the rows carry on the committed defaults, so a
// refusal here is a finding printed with the refusal's own sentence.
template <boys::Precision kPrecision, class Policy>
bool Carried(const char* label) {
    const boys::AccuracyFigure figure = boys::BoysAccuracyGuaranteed(
        kPrecision, 0, 0.0, Policy::kRoute, Policy::kScheme, Policy::kPack, Policy::kGranularity,
        Policy::kDivision, Policy::kRegionBExp);

    if (!figure.available)
    {
        std::printf("  %-46s refused: %s\n", label, figure.reason);
    }

    return figure.available;
}

template <class Policy>
Invariants SweepDoublePolicy(const char* label) {
    Invariants total;

    if (!Carried<boys::Precision::kFp64, Policy>(label))
    {
        return total;
    }

    for (const double x : Boundaries())
    {
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrders<Policy>(kNmax, x, value.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            bound[n] = DoubleBound(x);
        }

        Merge(total, CheckLadder({label, x, kDoubleFloor, 1.0, nullptr, value.data(), bound.data(),
                                  kNmax}));
    }

    return total;
}

template <class Policy>
Invariants SweepFloatPolicy(const char* label) {
    Invariants total;

    if (!Carried<boys::Precision::kFp32, Policy>(label))
    {
        return total;
    }

    constexpr bool kPlain = Policy::kDivision == boys::DivisionForm::kPlainReciprocal;

    for (const double argument : Boundaries())
    {
        const float x = static_cast<float>(argument);

        if (!std::isfinite(x) || x < 0.0f)
        {
            continue;
        }

        std::array<float, kNmax + 1> raw{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrdersF32<Policy>(kNmax, x, raw.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            value[n] = static_cast<double>(raw[n]);
            bound[n] = FloatBound(n, static_cast<double>(x), kPlain);
        }

        Merge(total, CheckLadder({label,
                                  static_cast<double>(x),
                                  kFloatFloor,
                                  1.0,
                                  nullptr,
                                  value.data(),
                                  bound.data(),
                                  kNmax}));
    }

    return total;
}

#if BoysFp16

template <class Policy>
Invariants SweepHalfPolicy(const char* label, bool bf16) {
    Invariants total;
    constexpr bool kPlain = Policy::kDivision == boys::DivisionForm::kPlainReciprocal;
    const bool carried = bf16 ? Carried<boys::Precision::kBf16, Policy>(label)
                              : Carried<boys::Precision::kFp16, Policy>(label);

    if (!carried)
    {
        return total;
    }

    for (const double argument : Boundaries())
    {
        const float x = static_cast<float>(argument);

        if (!std::isfinite(x) || x < 0.0f)
        {
            continue;
        }

        if (!bf16 && x > 65504.0f)
        {
            continue;
        }

        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        double laneArgument = static_cast<double>(x);

        if (bf16)
        {
            const boys::Bf16 argumentBf16 = boys::Bf16(x);
            laneArgument = static_cast<double>(argumentBf16);
            std::array<boys::Bf16, kNmax + 1> raw{};
            boys::BoysAllOrdersBf16<Policy>(kNmax, argumentBf16, raw.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                value[n] = static_cast<double>(raw[n]);
                bound[n] = HalfBound(value[n], true, kPlain);
            }
        } else
        {
            const boys::F16 argumentF16 = boys::F16(x);
            laneArgument = static_cast<double>(argumentF16);
            std::array<boys::F16, kNmax + 1> raw{};
            boys::BoysAllOrdersF16<Policy>(kNmax, argumentF16, raw.data());

            for (int n = 0; n <= kNmax; ++n)
            {
                value[n] = static_cast<double>(raw[n]);
                bound[n] = HalfBound(value[n], false, kPlain);
            }
        }

        Merge(total, CheckLadder({label,
                                  laneArgument,
                                  bf16 ? kBf16Floor : kHalfFloor,
                                  1.0,
                                  nullptr,
                                  value.data(),
                                  bound.data(),
                                  kNmax}));
    }

    return total;
}

#endif // BoysFp16

// --- The policy names -------------------------------------------------------

// The sweep names both routes, both schemes, all three partitions, both packing axes and all
// three division forms - the axes whose bodies the entries dispatch between. The precision lanes
// are swept beside them.

using DoubleCoarseSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using DoubleCoarseHornerArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using DoubleRationalSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using DoubleRationalHornerArgs =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using DoubleNarrowSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kNarrow>;
using DoubleUniformSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kUniform>;
using DoubleCoarseSplitOrders =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kCoarsest>;
using DoubleRationalSplitOrders =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kCoarsest>;
using DoubleUniformHornerOrders =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kUniform>;
using DoublePlainReciprocal =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::DivisionForm::kPlainReciprocal>;
using DoubleExactDivision =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::DivisionForm::kExactDivision>;
using DoubleAccurateExp =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::kDefaultDivisionForm,
                     boys::RegionBExp::kAccurate>;

using FloatCoarseSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using FloatCoarseHornerArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using FloatRationalSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using FloatRationalHornerArgs =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using FloatNarrowSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kNarrow>;
using FloatUniformSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kUniform>;
using FloatCoarseSplitOrders =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kCoarsest>;
using FloatRationalSplitOrders =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kCoarsest>;
using FloatUniformHornerOrders =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kHorner,
                     boys::BoysBudget::kFloat, boys::PackAxis::kOrders,
                     boys::FitGranularity::kUniform>;
using FloatPlainReciprocal =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::DivisionForm::kPlainReciprocal>;
using FloatExactDivision =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFloat, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::DivisionForm::kExactDivision>;

#if BoysFp16

using HalfCoarseSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using HalfRationalSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kRationalMinimax, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest>;
using HalfNarrowSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kArguments,
                     boys::FitGranularity::kNarrow>;
using HalfUniformSplitArgs =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kArguments,
                     boys::FitGranularity::kUniform>;
using HalfCoarseSplitOrders =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kOrders,
                     boys::FitGranularity::kCoarsest>;
using HalfPlainReciprocal =
    boys::SeamPolicy<boys::FitRoute::kChebyshev, boys::EvalScheme::kSplitClenshaw,
                     boys::BoysBudget::kFp16, boys::PackAxis::kArguments,
                     boys::FitGranularity::kCoarsest, boys::DivisionForm::kPlainReciprocal>;

#endif // BoysFp16

// --- The tests ---------------------------------------------------------------

TEST(BoysStructuralInvariants, ReferenceGridPositivity) {
    if (Reference().rows == 0)
    {
        GTEST_SKIP() << "the committed reference grid is not present in this tree";
    }

    const Invariants doubleLane = CheckDefaultDouble();
    const Invariants floatLane = CheckDefaultFloat();

    Report("double", doubleLane);
    Report("float", floatLane);

    ExpectClean(doubleLane);
    ExpectClean(floatLane);
    EXPECT_GT(doubleLane.claimed, 0u);
    EXPECT_GT(floatLane.claimed, 0u);

#if BoysFp16
    const Invariants f16Lane = CheckDefaultF16(false);
    const Invariants bf16Lane = CheckDefaultF16(true);
    const Invariants nativeLane = CheckNativeHalf();

    Report("fp16", f16Lane);
    Report("bf16", bf16Lane);
    Report("native half (scaled by 2^15)", nativeLane);

    ExpectClean(f16Lane);
    ExpectClean(bf16Lane);
    ExpectClean(nativeLane);
    EXPECT_GT(f16Lane.claimed, 0u);
    EXPECT_GT(bf16Lane.claimed, 0u);
    EXPECT_GT(nativeLane.claimed, 0u);
#else
    GTEST_SKIP() << "the half entries are behind the closed BoysFp16 seam";
#endif
}

TEST(BoysStructuralInvariants, ReferenceGridOrderMonotonicity) {
    if (Reference().rows == 0)
    {
        GTEST_SKIP() << "the committed reference grid is not present in this tree";
    }

    const Invariants doubleLane = CheckDefaultDouble();
    const Invariants floatLane = CheckDefaultFloat();

    Report("double", doubleLane);
    Report("float", floatLane);

    ExpectClean(doubleLane);
    ExpectClean(floatLane);
    EXPECT_GT(doubleLane.strictMono, 0u)
        << "no adjacent pair is certified strict by the exact gap and the bound";
    EXPECT_GT(floatLane.strictMono, 0u)
        << "no adjacent pair is certified strict by the exact gap and the bound";

#if BoysFp16
    const Invariants f16Lane = CheckDefaultF16(false);
    const Invariants bf16Lane = CheckDefaultF16(true);
    const Invariants nativeLane = CheckNativeHalf();

    Report("fp16", f16Lane);
    Report("bf16", bf16Lane);
    Report("native half (scaled by 2^15)", nativeLane);

    ExpectClean(f16Lane);
    ExpectClean(bf16Lane);
    ExpectClean(nativeLane);
    EXPECT_GT(f16Lane.strictMono, 0u);
    EXPECT_GT(bf16Lane.strictMono, 0u);
    EXPECT_GT(nativeLane.strictMono, 0u);
#endif
}

TEST(BoysStructuralInvariants, ReferenceGridLogConvexity) {
    if (Reference().rows == 0)
    {
        GTEST_SKIP() << "the committed reference grid is not present in this tree";
    }

    const Invariants doubleLane = CheckDefaultDouble();
    const Invariants floatLane = CheckDefaultFloat();

    Report("double", doubleLane);
    Report("float", floatLane);

    ExpectClean(doubleLane);
    ExpectClean(floatLane);
    EXPECT_GT(doubleLane.strictLogCvx, 0u)
        << "no triple is certified strict by the exact values and the bounds";
    EXPECT_GT(floatLane.strictLogCvx, 0u)
        << "no triple is certified strict by the exact values and the bounds";

#if BoysFp16
    const Invariants f16Lane = CheckDefaultF16(false);
    const Invariants bf16Lane = CheckDefaultF16(true);
    const Invariants nativeLane = CheckNativeHalf();

    Report("fp16", f16Lane);
    Report("bf16", bf16Lane);
    Report("native half (scaled by 2^15)", nativeLane);

    ExpectClean(f16Lane);
    ExpectClean(bf16Lane);
    ExpectClean(nativeLane);
    EXPECT_GT(f16Lane.strictLogCvx, 0u);
    EXPECT_GT(bf16Lane.strictLogCvx, 0u);
    EXPECT_GT(nativeLane.strictLogCvx, 0u);
#endif
}

TEST(BoysStructuralInvariants, ReferenceGridAdjacentOrderRecurrence) {
    if (Reference().rows == 0)
    {
        GTEST_SKIP() << "the committed reference grid is not present in this tree";
    }

    const Invariants doubleLane = CheckDefaultDouble();
    const Invariants floatLane = CheckDefaultFloat();

    Report("double", doubleLane);
    Report("float", floatLane);

    ExpectClean(doubleLane);
    ExpectClean(floatLane);
    EXPECT_GT(doubleLane.recurrenceChecked, 0u);
    EXPECT_GT(floatLane.recurrenceChecked, 0u);

#if BoysFp16
    const Invariants f16Lane = CheckDefaultF16(false);
    const Invariants bf16Lane = CheckDefaultF16(true);
    const Invariants nativeLane = CheckNativeHalf();

    Report("fp16", f16Lane);
    Report("bf16", bf16Lane);
    Report("native half (scaled by 2^15)", nativeLane);

    ExpectClean(f16Lane);
    ExpectClean(bf16Lane);
    ExpectClean(nativeLane);
    EXPECT_GT(f16Lane.recurrenceChecked, 0u);
    EXPECT_GT(bf16Lane.recurrenceChecked, 0u);
    EXPECT_GT(nativeLane.recurrenceChecked, 0u);
#endif
}

TEST(BoysStructuralInvariants, ClosedFormAtZero) {
    // F_n(0) = 1/(2n+1) exactly, and the recurrence at x = 0 is the same statement. The lane's
    // own figure is the tolerance: a returned value that is not the closed form within it is a
    // defect the reference grid at x = 0 would also see, but this test names the reason.
    std::array<double, kNmax + 1> value{};
    boys::BoysAllOrders(kNmax, 0.0, value.data());

    for (int n = 0; n <= kNmax; ++n)
    {
        const double exact = 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        EXPECT_NEAR(value[n], exact, 1e-13) << "n = " << n;
    }

    std::array<float, kNmax + 1> valuef{};
    boys::BoysAllOrdersF32(kNmax, 0.0f, valuef.data());

    for (int n = 0; n <= kNmax; ++n)
    {
        const double exact = 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        EXPECT_NEAR(static_cast<double>(valuef[n]), exact, 1.5e-7) << "n = " << n;
    }

#if BoysFp16
    std::array<boys::F16, kNmax + 1> value16{};
    boys::BoysAllOrdersF16(kNmax, boys::F16(0.0f), value16.data());

    for (int n = 0; n <= kNmax; ++n)
    {
        const double exact = 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        const double observed = static_cast<double>(value16[n]);
        EXPECT_NEAR(observed, exact, HalfBound(observed, false, false)) << "n = " << n;
    }

    std::array<boys::Bf16, kNmax + 1> valueb{};
    boys::BoysAllOrdersBf16(kNmax, boys::Bf16(0.0f), valueb.data());

    for (int n = 0; n <= kNmax; ++n)
    {
        const double exact = 1.0 / (2.0 * static_cast<double>(n) + 1.0);
        const double observed = static_cast<double>(valueb[n]);
        EXPECT_NEAR(observed, exact, HalfBound(observed, true, false)) << "n = " << n;
    }
#endif
}

TEST(BoysStructuralInvariants, PolicySweepAcrossBoundaries) {
    Invariants total;

    Merge(total, SweepDoublePolicy<DoubleCoarseSplitArgs>("double, cheb/split/shipped/args"));
    Merge(total, SweepDoublePolicy<DoubleCoarseHornerArgs>("double, cheb/horner/shipped/args"));
    Merge(total, SweepDoublePolicy<DoubleRationalSplitArgs>("double, rational/split/shipped/args"));
    Merge(total,
          SweepDoublePolicy<DoubleRationalHornerArgs>("double, rational/horner/shipped/args"));
    Merge(total, SweepDoublePolicy<DoubleNarrowSplitArgs>("double, cheb/split/narrow/args"));
    Merge(total, SweepDoublePolicy<DoubleUniformSplitArgs>("double, cheb/split/uniform/args"));
    Merge(total, SweepDoublePolicy<DoubleCoarseSplitOrders>("double, cheb/split/shipped/orders"));
    Merge(total,
          SweepDoublePolicy<DoubleRationalSplitOrders>("double, rational/split/shipped/orders"));
    Merge(total,
          SweepDoublePolicy<DoubleUniformHornerOrders>("double, cheb/horner/uniform/orders"));
    Merge(total, SweepDoublePolicy<DoublePlainReciprocal>("double, plain reciprocal"));
    Merge(total, SweepDoublePolicy<DoubleExactDivision>("double, exact division"));
    Merge(total, SweepDoublePolicy<DoubleAccurateExp>("double, accurate region-B exponential"));

    Merge(total, SweepFloatPolicy<FloatCoarseSplitArgs>("float, cheb/split/shipped/args"));
    Merge(total, SweepFloatPolicy<FloatCoarseHornerArgs>("float, cheb/horner/shipped/args"));
    Merge(total, SweepFloatPolicy<FloatRationalSplitArgs>("float, rational/split/shipped/args"));
    Merge(total, SweepFloatPolicy<FloatRationalHornerArgs>("float, rational/horner/shipped/args"));
    Merge(total, SweepFloatPolicy<FloatNarrowSplitArgs>("float, cheb/split/narrow/args"));
    Merge(total, SweepFloatPolicy<FloatUniformSplitArgs>("float, cheb/split/uniform/args"));
    Merge(total, SweepFloatPolicy<FloatCoarseSplitOrders>("float, cheb/split/shipped/orders"));
    Merge(total,
          SweepFloatPolicy<FloatRationalSplitOrders>("float, rational/split/shipped/orders"));
    Merge(total, SweepFloatPolicy<FloatUniformHornerOrders>("float, cheb/horner/uniform/orders"));
    Merge(total, SweepFloatPolicy<FloatPlainReciprocal>("float, plain reciprocal"));
    Merge(total, SweepFloatPolicy<FloatExactDivision>("float, exact division"));

#if BoysFp16
    Merge(total, SweepHalfPolicy<HalfCoarseSplitArgs>("fp16, cheb/split/shipped/args", false));
    Merge(total,
          SweepHalfPolicy<HalfRationalSplitArgs>("fp16, rational/split/shipped/args", false));
    Merge(total, SweepHalfPolicy<HalfNarrowSplitArgs>("fp16, cheb/split/narrow/args", false));
    Merge(total, SweepHalfPolicy<HalfUniformSplitArgs>("fp16, cheb/split/uniform/args", false));
    Merge(total, SweepHalfPolicy<HalfCoarseSplitOrders>("fp16, cheb/split/shipped/orders", false));
    Merge(total, SweepHalfPolicy<HalfPlainReciprocal>("fp16, plain reciprocal", false));

    Merge(total, SweepHalfPolicy<HalfCoarseSplitArgs>("bf16, cheb/split/shipped/args", true));
    Merge(total, SweepHalfPolicy<HalfRationalSplitArgs>("bf16, rational/split/shipped/args", true));
    Merge(total, SweepHalfPolicy<HalfNarrowSplitArgs>("bf16, cheb/split/narrow/args", true));
    Merge(total, SweepHalfPolicy<HalfUniformSplitArgs>("bf16, cheb/split/uniform/args", true));
    Merge(total, SweepHalfPolicy<HalfCoarseSplitOrders>("bf16, cheb/split/shipped/orders", true));
    Merge(total, SweepHalfPolicy<HalfPlainReciprocal>("bf16, plain reciprocal", true));
#endif

    Report("all policies", total);
    ExpectClean(total);
    EXPECT_GT(total.claimed, 0u);
}

TEST(BoysStructuralInvariants, ScalarAndPackedAxesAgreeWithinTheirBounds) {
    // The packing axis is the scalar/SIMD boundary: the arguments axis evaluates one order at a
    // time, the orders axis packs four (double) or eight (float) orders of one argument into a
    // register. The two approximate the same F_n, so their difference is bounded by the sum of
    // their two figures; a lane that reads another table or transposes an index leaves it.
    const bool vector = boys::BoysAvx2Available();
    std::printf("  AVX2 tier available: %s\n", vector ? "yes" : "no");

    const std::vector<double>& xs = Boundaries();
    std::size_t compared = 0;
    std::size_t differing = 0;
    double worst = 0.0;
    double worstFraction = 0.0;

    for (const double x : xs)
    {
        std::array<double, kNmax + 1> args{};
        std::array<double, kNmax + 1> orders{};
        boys::BoysAllOrders<DoubleCoarseSplitArgs>(kNmax, x, args.data());
        boys::BoysAllOrders<DoubleCoarseSplitOrders>(kNmax, x, orders.data());
        const double bound = DoubleBound(x);

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            const double delta = std::fabs(args[at] - orders[at]);
            ++compared;
            worst = std::max(worst, delta);
            worstFraction = std::max(worstFraction, delta / (2.0 * bound));

            if (std::memcmp(&args[at], &orders[at], sizeof(double)) != 0)
            {
                ++differing;
            }

            EXPECT_LE(delta, 2.0 * bound * (1.0 + 1e-12))
                << "double axes part at n = " << n << ", x = " << x;
        }
    }

    std::printf("  double axes: %zu values, %zu differ in bits, worst %.3e (%.2f of the two "
                "bounds)\n",
                compared,
                differing,
                worst,
                worstFraction);

    compared = 0;
    differing = 0;
    worst = 0.0;
    worstFraction = 0.0;

    for (const double argument : xs)
    {
        const float x = static_cast<float>(argument);

        if (!std::isfinite(x) || x < 0.0f)
        {
            continue;
        }

        std::array<float, kNmax + 1> args{};
        std::array<float, kNmax + 1> orders{};
        boys::BoysAllOrdersF32<FloatCoarseSplitArgs>(kNmax, x, args.data());
        boys::BoysAllOrdersF32<FloatCoarseSplitOrders>(kNmax, x, orders.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            const double delta = std::fabs(static_cast<double>(args[at]) -
                                           static_cast<double>(orders[at]));
            const double bound = FloatBound(n, static_cast<double>(x), false);
            ++compared;
            worst = std::max(worst, delta);
            worstFraction = std::max(worstFraction, delta / (2.0 * bound));

            if (std::memcmp(&args[at], &orders[at], sizeof(float)) != 0)
            {
                ++differing;
            }

            EXPECT_LE(delta, 2.0 * bound * (1.0 + 1e-12))
                << "float axes part at n = " << n << ", x = " << x;
        }
    }

    std::printf("  float axes:  %zu values, %zu differ in bits, worst %.3e (%.2f of the two "
                "bounds)\n",
                compared,
                differing,
                worst,
                worstFraction);

    EXPECT_GT(compared, 0u);
}

TEST(BoysStructuralInvariants, ReducedPrecisionFloorsAreClassified) {
    if (Reference().rows == 0)
    {
        GTEST_SKIP() << "the committed reference grid is not present in this tree";
    }

    // The floor is the format's range boundary; the bound is the contract's absolute figure.
    // Positivity is asserted only where the exact value is above both: above the floor it is
    // representable, and above the bound a zero would violate the figure. Between them a zero is
    // permitted and the cell is counted as bound-limited; the zero counts show the split.
    struct ZeroExample {
        double x = 0.0;
        int n = 0;
        double exact = 0.0;
        double bound = 0.0;
    };

    struct FloorTally {
        std::size_t claimed = 0;
        std::size_t floorLimited = 0;
        std::size_t boundLimited = 0;
        std::size_t zeros = 0;
        std::size_t zerosFloor = 0;
        std::size_t zerosBound = 0;
        std::size_t subnormal = 0;
        std::size_t exampleCount = 0;
        std::array<ZeroExample, 4> examples{};
        double smallestClaimed = std::numeric_limits<double>::infinity();
        double largestFloorLimited = 0.0;
        double largestBoundLimited = 0.0;
    };

    const auto accumulate = [](FloorTally& tally, double x,
                               const std::array<double, kNmax + 1>& truth,
                               const std::array<double, kNmax + 1>& bound,
                               const std::array<double, kNmax + 1>& value, double floor) {
        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            const double v = value[at];
            const double exact = truth[at];

            EXPECT_TRUE(std::isfinite(v)) << "n = " << n;
            EXPECT_GE(v, 0.0) << "n = " << n;

            if (exact < floor)
            {
                ++tally.floorLimited;
                tally.largestFloorLimited = std::max(tally.largestFloorLimited, exact);
            } else if (!(exact > bound[at]))
            {
                ++tally.boundLimited;
                tally.largestBoundLimited = std::max(tally.largestBoundLimited, exact);
            } else
            {
                ++tally.claimed;
                tally.smallestClaimed = std::min(tally.smallestClaimed, exact);
                EXPECT_GT(v, 0.0) << "claimed cell returned zero, n = " << n;
            }

            if (v == 0.0)
            {
                ++tally.zeros;

                if (exact < floor)
                {
                    ++tally.zerosFloor;
                } else if (!(exact > bound[at]))
                {
                    ++tally.zerosBound;

                    // The counterexample to the literal positivity statement: the exact value is
                    // above the format's floor but inside the lane's absolute figure, and the
                    // return is zero. Record the first few so the restriction can be quoted.
                    if (tally.exampleCount < tally.examples.size())
                    {
                        tally.examples[tally.exampleCount] =
                            ZeroExample{x, n, exact, bound[at]};
                        ++tally.exampleCount;
                    }
                }
            } else if (v < floor)
            {
                ++tally.subnormal;
            }
        }
    };

    const auto report = [](const char* label, double floor, const FloorTally& tally,
                           bool expectBoundLimited) {
        std::printf("  %-10s floor %.3e | claimed %6zu (smallest %.3e) | floor-limited %6zu "
                    "(largest %.3e) | bound-limited %6zu (largest %.3e) | returned zero %zu "
                    "(floor %zu, bound %zu), subnormal %zu\n",
                    label,
                    floor,
                    tally.claimed,
                    tally.smallestClaimed,
                    tally.floorLimited,
                    tally.largestFloorLimited,
                    tally.boundLimited,
                    tally.largestBoundLimited,
                    tally.zeros,
                    tally.zerosFloor,
                    tally.zerosBound,
                    tally.subnormal);

        for (std::size_t i = 0; i < tally.exampleCount; ++i)
        {
            const ZeroExample& example = tally.examples[i];
            std::printf("    zero above the floor: n = %d, x = %.17g, exact %.6e, bound %.3e\n",
                        example.n,
                        example.x,
                        example.exact,
                        example.bound);
        }

        EXPECT_GT(tally.claimed, 0u) << label << " has no claimed cell on the grid";
        EXPECT_GT(tally.floorLimited, 0u) << label << " has no floor-limited cell on the grid";

        if (expectBoundLimited)
        {
            EXPECT_GT(tally.boundLimited, 0u)
                << label << " has no bound-limited cell on the grid";
        } else
        {
            EXPECT_EQ(tally.boundLimited, 0u)
                << label << ": the lane's figure is below its floor, so no cell can be inside "
                          "the figure and above the floor";
        }
    };

    FloorTally doubleTally;

    for (const auto& [x, truth] : Reference().doubleLadders)
    {
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrders(kNmax, x, value.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            bound[n] = kRouteAllowance * 1e-13 * truth[0];
        }

        accumulate(doubleTally, x, truth, bound, value, kDoubleFloor);
    }

    report("double", kDoubleFloor, doubleTally, true);

    constexpr bool kFloatPlain =
        boys::DefaultPolicy<boys::Precision::kFp32, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;

    FloorTally floatTally;

    for (const auto& [key, truth] : Reference().floatLadders)
    {
        std::array<float, kNmax + 1> raw{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrdersF32(kNmax, std::bit_cast<float>(key), raw.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            value[at] = static_cast<double>(raw[at]);
            bound[at] = kRouteAllowance *
                        (std::max(std::ldexp(1.0, -22) * truth.value[at], 1.5e-7) +
                         (kFloatPlain ? 1e-7 : 0.0));
        }

        accumulate(floatTally, truth.argument, truth.value, bound, value, kFloatFloor);
    }

    report("float", kFloatFloor, floatTally, true);

#if BoysFp16
    constexpr bool kF16Plain =
        boys::DefaultPolicy<boys::Precision::kFp16, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;
    constexpr bool kBf16Plain =
        boys::DefaultPolicy<boys::Precision::kBf16, boys::Shape::kAllOrders>::kDivision ==
        boys::DivisionForm::kPlainReciprocal;

    FloorTally f16Tally;

    for (const auto& [key, truth] : Reference().f16Ladders)
    {
        std::array<boys::F16, kNmax + 1> raw{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrdersF16(kNmax, boys::detail::F16FromBits(key), raw.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            value[at] = static_cast<double>(raw[at]);
            bound[at] = HalfBound(value[at], false, kF16Plain);
        }

        accumulate(f16Tally, truth.argument, truth.value, bound, value, kHalfFloor);
    }

    // The half lanes' base figure is 1.5e-7 and the format's floor is 2^-14: the floor is above
    // the figure, so every representable cell is outside the error interval.
    report("fp16", kHalfFloor, f16Tally, false);

    FloorTally bf16Tally;

    for (const auto& [key, truth] : Reference().bf16Ladders)
    {
        std::array<boys::Bf16, kNmax + 1> raw{};
        std::array<double, kNmax + 1> value{};
        std::array<double, kNmax + 1> bound{};
        boys::BoysAllOrdersBf16(kNmax, boys::detail::Bf16FromBits(key), raw.data());

        for (int n = 0; n <= kNmax; ++n)
        {
            const std::size_t at = static_cast<std::size_t>(n);
            value[at] = static_cast<double>(raw[at]);
            bound[at] = HalfBound(value[at], true, kBf16Plain);
        }

        accumulate(bf16Tally, truth.argument, truth.value, bound, value, kBf16Floor);
    }

    report("bf16", kBf16Floor, bf16Tally, true);
#endif
}

} // namespace
