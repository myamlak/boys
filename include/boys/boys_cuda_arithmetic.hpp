#pragma once

/// \cond
// Not API: the seed fits and the region recurrences the CUDA lane runs. The header
// ships because the lane's device-callable entries are header-defined and have to
// run the arithmetic the batch kernels run rather than a second copy of it; the API
// reference documents the entries.
///
/// One body per (precision, shape), each taking the tables as a lane object, so the
/// same body serves a batch kernel reading the __constant__ tables and a device
/// entry reading the caller's own handle. A lane object supplies, all inlineable and
/// all cheap:
///
///   int          Count(int order)                  pieces the fit is cut into
///   T            A(int order, int piece)           piece lower edge
///   T            B(int order, int piece)           piece upper edge
///   const T*     Coeffs(int order, int piece)      the piece's coefficients
///   int          Deg(int order, int piece)         the piece's degree
///   T            BSeed(T x, int order)             region-B seed at an argument
///
/// A lane that stores the monomial form of its fits carries a `kMonomial` member as
/// well, and the piece summation reads it: the same pieces at the same degrees,
/// summed by Horner rather than by the split Clenshaw. The member is optional
/// because a lane's stored form is a property of the tables it was handed and not of
/// the body reading them; a lane without it reads the Chebyshev pool.
///
/// A lane whose pieces are rational pairs carries a `kRational` member and answers
/// the piece read with four members in place of one: `NumDeg` and `DenDeg` for the
/// pair's two degrees, `Coeffs` for the numerator's block and `DenCoeffs` for the
/// denominator's. The route is a family and not a basis: a pair has one stored form
/// and either scheme sums it.
///
/// T is double for the double lane's seeds and float for the float lane's. BSeed is
/// the whole of region B's seed, taken at the argument rather than as one polynomial
/// over the region: the shipped partition's seed is one fit over [kX0, kX1] and
/// another partition's is piecewise. It takes the order because the relaxed batches
/// read their region-B degree from the order-0 entry — the F_0 seed's error reaches
/// every output with gain at most 1 + 1.846e-17 — while the single lanes read it per
/// order; the lane object decides which, and the arithmetic is the same either way.
///
/// The region structure is the library's: region A (x below kX0) is the piecewise
/// Chebyshev fit of F_n itself, region B (up to kX1) is the upward recursion seeded
/// by the region-B fit of F_0, and region C is the F_0 asymptotic form with the same
/// recursion. The all-orders bodies carry a second, downward recursion inside region
/// A: the fit there is of F_n, and the lower orders come back down from it.

#include "boys/accuracy.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_cuda_muladd.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <type_traits>

namespace boys::detail {

/// The F_0 asymptotic prefactor: the integral that gives F_0 at large x is
/// sqrt(pi/x)/2, so F_0 = sqrt(pi)/2 * rsqrt(x) is the value the region-C
/// recursion starts from.
inline constexpr double kHalfSqrtPi = 0.886226925452758014;

// ---------------------------------------------------------------------------
// the multiply-add route
// ---------------------------------------------------------------------------

// The two multiply-add routes as the device spells them: the fused step is the round-to-nearest
// intrinsic, the value every device figure here was measured at. The separate step is a fused
// step with a zero addend, never a bare a * b + c: a bare product-plus-add is a contraction
// licence the device compiler takes up by default, and that spelling would be the fused route.

// That a toolchain keeps the zero fold is measured rather than assumed:
// tests/boys_cuda_route_test.cu evaluates both routes against exact host references.
template <typename T>
__device__ __forceinline__ T DeviceFusedMulAdd(T a, T b, T c) {
    if constexpr (std::is_same_v<T, float>)
    {
        return __fmaf_rn(a, b, c);
    } else
    {
        return __fma_rn(a, b, c);
    }
}

// `a * b + c` with two roundings: the product rounds, then the sum rounds.
template <typename T>
__device__ __forceinline__ T DeviceSeparateMulAdd(T a, T b, T c) {
    if constexpr (std::is_same_v<T, float>)
    {
        return __fmaf_rn(a, b, 0.0f) + c;
    } else
    {
        return __fma_rn(a, b, 0.0) + c;
    }
}

// The step at one route. The route is a template argument with the build's own
// selection as its default, so a body that names no route runs the route the
// build selected and a check can name either one and compare them.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename T>
__device__ __forceinline__ T DeviceMulAdd(T a, T b, T c) {
    if constexpr (kRoute == backend::MulAddRoute::kFused)
    {
        return DeviceFusedMulAdd(a, b, c);
    } else
    {
        return DeviceSeparateMulAdd(a, b, c);
    }
}

// ---------------------------------------------------------------------------
// piecewise Chebyshev evaluation
// ---------------------------------------------------------------------------

// Split Clenshaw: the even and odd parts are summed apart and combined once,
// which keeps every step a fused multiply-add on a value of one sign. The
// deg <= 2 cases are unrolled because the split below assumes the odd part
// has at least two terms.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ double DeviceClenshawSplit(const double* c, int deg, double t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return DeviceMulAdd<kRoute>(t, c[1], c[0]);
    }

    const double v = DeviceMulAdd<kRoute>(2.0, t * t, -1.0);
    const double twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4
        // (the odd part's m = 1 finalization would double the t*c[1] term).
        return DeviceMulAdd<kRoute>(t, c[1], DeviceMulAdd<kRoute>(v, c[2], c[0]));
    }

    const int m = deg / 2;
    double b1 = c[2 * m];
    double b2 = 0.0;

    for (int k = m - 1; k >= 1; --k)
    {
        const double b0 = DeviceMulAdd<kRoute>(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const double even = DeviceMulAdd<kRoute>(v, b1, c[0] - b2);

    double o1 = c[2 * m - 1];
    double o2 = 0.0;

    for (int k = m - 2; k >= 1; --k)
    {
        const double o0 = DeviceMulAdd<kRoute>(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const double odd = DeviceMulAdd<kRoute>(twoV - 1.0, o1, c[1] - o2);
    return DeviceMulAdd<kRoute>(t, odd, even);
}

template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ float DeviceClenshawSplit32(const float* c, int deg, float t) {
    if (deg == 0)
    {
        return c[0];
    }

    if (deg == 1)
    {
        return DeviceMulAdd<kRoute>(t, c[1], c[0]);
    }

    const float v = DeviceMulAdd<kRoute>(2.0f, t * t, -1.0f);
    const float twoV = v + v;

    if (deg == 2)
    {
        // T_2(t) = 2t^2 - 1 = v; the even/odd split below assumes deg >= 4
        // (the odd part's m = 1 finalization would double the t*c[1] term).
        return DeviceMulAdd<kRoute>(t, c[1], DeviceMulAdd<kRoute>(v, c[2], c[0]));
    }

    const int m = deg / 2;
    float b1 = c[2 * m];
    float b2 = 0.0f;

    for (int k = m - 1; k >= 1; --k)
    {
        const float b0 = DeviceMulAdd<kRoute>(twoV, b1, c[2 * k] - b2);
        b2 = b1;
        b1 = b0;
    }

    const float even = DeviceMulAdd<kRoute>(v, b1, c[0] - b2);

    float o1 = c[2 * m - 1];
    float o2 = 0.0f;

    for (int k = m - 2; k >= 1; --k)
    {
        const float o0 = DeviceMulAdd<kRoute>(twoV, o1, c[2 * k + 1] - o2);
        o2 = o1;
        o1 = o0;
    }

    const float odd = DeviceMulAdd<kRoute>(twoV - 1.0f, o1, c[1] - o2);
    return DeviceMulAdd<kRoute>(t, odd, even);
}

// The monomial scheme's summation: the same fit in the other basis, at the same offsets and degrees
// as the Chebyshev one, Horner in ascending order. One multiply-add per coefficient against the
// split Clenshaw's two, so the scheme is a cost choice at equal degree, and each form's delivered
// accuracy is its own certified row.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ double DeviceHornerMono(const double* c, int deg, double t) {
    double acc = c[deg];

    for (int j = deg - 1; j >= 0; --j)
    {
        acc = DeviceMulAdd<kRoute>(acc, t, c[j]);
    }

    return acc;
}

// The float lanes' form of the same summation, in the lane's own arithmetic, so a float piece's two
// forms are summed the way the certified rows of their table were measured (kNarrowARowsF32,
// kFlatRowsF32, boys_coefficients.hpp); the double helper above would sum a double polynomial read
// through a float pointer, a different arithmetic and rounding.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ float DeviceHornerMono32(const float* c, int deg, float t) {
    float acc = c[deg];

    for (int j = deg - 1; j >= 0; --j)
    {
        acc = DeviceMulAdd<kRoute>(acc, t, c[j]);
    }

    return acc;
}

// The rational route's summation: both blocks stored ascending, both Horner, divided once, with the
// denominator's held-at-one constant written as the final multiply-add; a pair whose denominator
// degree is zero has no denominator. The reading is coefficient for coefficient the host lane's own
// (boys_impl.hpp, RationalPieceAtCut, RationalSeedAtCut), so the two lanes' figures are comparable.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ double DeviceRatSum(const double* num,
                                               int numDeg,
                                               const double* den,
                                               int denDeg,
                                               double t) {
    double numerator = num[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        numerator = DeviceMulAdd<kRoute>(numerator, t, num[j]);
    }

    if (denDeg == 0)
    {
        return numerator;
    }

    double denominator = den[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        denominator = DeviceMulAdd<kRoute>(denominator, t, den[j]);
    }

    return numerator / DeviceMulAdd<kRoute>(denominator, t, 1.0);
}

// The float lanes' form of that summation, every step fused, reading the same stored form as the
// host lane's own float rational evaluation, so the two lanes' figures are comparable. The two
// blocks are the caller's to place: the shipped region-B seed in two arrays, the narrow partition
// in one pool, its denominator above the numerator.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute>
__device__ __forceinline__ float DeviceRatSum32(const float* num,
                                                int numDeg,
                                                const float* den,
                                                int denDeg,
                                                float t) {
    float numerator = num[numDeg];

    for (int j = numDeg - 1; j >= 0; --j)
    {
        numerator = DeviceMulAdd<kRoute>(numerator, t, num[j]);
    }

    if (denDeg == 0)
    {
        return numerator;
    }

    float denominator = den[denDeg - 1];

    for (int j = denDeg - 2; j >= 0; --j)
    {
        denominator = DeviceMulAdd<kRoute>(denominator, t, den[j]);
    }

    return numerator / DeviceMulAdd<kRoute>(denominator, t, 1.0f);
}

// Which basis a lane's coefficients are stored in: a lane reading the monomial pool states so with
// a kMonomial member, and the trait reads its value rather than its presence - a form sharing its
// geometry with the other form of the same table states false here and the bodies below take the
// Chebyshev arm, which is the pool that form reads.
template <typename Lane, typename = void>
struct LaneMonomial : std::false_type {};

template <typename Lane>
struct LaneMonomial<Lane, std::void_t<decltype(Lane::kMonomial)>>
    : std::bool_constant<Lane::kMonomial> {};

// A lane whose pieces are rational pairs rather than polynomials: it carries a kRational member
// and names its two coefficient blocks and their two degrees in place of Deg. A rational piece has
// neither Chebyshev nor monomial form and neither scheme changes its numbers, so the route is a
// family of its own and the trait sits beside the scheme's rather than under it.
template <typename Lane, typename = void>
struct LaneRational : std::false_type {};

template <typename Lane>
struct LaneRational<Lane, std::void_t<decltype(Lane::kRational)>>
    : std::bool_constant<Lane::kRational> {};

template <backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane>
__device__ __forceinline__ double DevicePieceSum(const Lane& lane,
                                                 const double* c,
                                                 int deg,
                                                 double t) {
    if constexpr (LaneMonomial<Lane>::value)
    {
        return DeviceHornerMono<kRoute>(c, deg, t);
    } else
    {
        return DeviceClenshawSplit<kRoute>(c, deg, t);
    }
}

// ---------------------------------------------------------------------------
// seeds
// ---------------------------------------------------------------------------

// The piece a value falls in: the pieces of an order ascend and their upper edges partition the
// order's fitted range, so the first edge above x names the piece and the last piece holds
// everything above the topmost edge. The scan is linear - a binary search would not pay for its
// branches until the rows are much longer than the twelve pieces a lane may have.
template <backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane>
__device__ __forceinline__ double DeviceSeed(const Lane& lane, int order, double x) {
    const int count = lane.Count(order);
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < lane.B(order, i))
        {
            p = i;
            break;
        }
    }

    const double t = 2.0 * (x - lane.A(order, p)) / (lane.B(order, p) - lane.A(order, p)) - 1.0;

    if constexpr (LaneRational<Lane>::value)
    {
        return DeviceRatSum<kRoute>(lane.Coeffs(order, p),
                                    lane.NumDeg(order, p),
                                    lane.DenCoeffs(order, p),
                                    lane.DenDeg(order, p),
                                    t);
    } else
    {
        return DevicePieceSum<kRoute>(lane, lane.Coeffs(order, p), lane.Deg(order, p), t);
    }
}

template <backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane>
__device__ __forceinline__ float DeviceSeed32(const Lane& lane, int order, float x) {
    const int count = lane.Count(order);
    int p = count - 1;

    for (int i = 0; i < count; ++i)
    {
        if (x < lane.B(order, i))
        {
            p = i;
            break;
        }
    }

    const float* c = lane.Coeffs(order, p);
    const float t =
        2.0f * (x - lane.A(order, p)) / (lane.B(order, p) - lane.A(order, p)) - 1.0f;
    return DeviceClenshawSplit32<kRoute>(c, lane.Deg(order, p), t);
}

// ---------------------------------------------------------------------------
// the region-B exponential
// ---------------------------------------------------------------------------

// The arithmetic the region-B seed's t = e^{-x}/2 takes, at the member the option names
// (RegionBExp, boys/accuracy.hpp): the one factor of the region-B path a caller can trade
// accuracy for speed on. kFastExp corrects the hardware exponential's argument-scaling residual
// back to its own few ulp, flat in the argument; the other member is the library routine.

// The double lane's two members are the host's own (boys_impl.hpp, RegionBHalfExp), ported
// rather than re-derived so that a name has one bound and not two.

// The host's constants at the device's own names: kDeviceRegionBExpReduced is e^{-r} on
// |r| <= ln 2/2, degree 7, Chebyshev fit, 8.336e-11 relative, and kDeviceRegionBExpCheapFrom
// the smallest x whose ladder requirement reaches ten times that error. A device object because
// the sum indexes it at a loop variable, an odr-use the device compiler needs a copy for.
inline constexpr double kDeviceRegionBExpCheapFrom = 16.173039304440838;
__device__ constexpr double kDeviceRegionBExpReduced[8] = {
    0.9999999999190966,
    -0.9999999999910163,
    0.5000000168381031,
    -0.16666666853653805,
    0.04166608365203089,
    -0.008333268585902836,
    0.0013956053200368968,
    -0.00019915868993476617,
};
inline constexpr double kDeviceRegionBExpLog2e = 1.4426950408889634073599246810018921;
inline constexpr double kDeviceRegionBExpLn2 = 0.6931471805599453094172321214581766;
inline constexpr double kDeviceRegionBExpRoundMagic = 6755399441055744.0; // 1.5 * 2^52

// The reduction is x = k ln 2 + r, |r| <= ln 2/2, so e^{-x} = 2^{-k} e^{-r}; over region B
// k in [25, 42] keeps the exponent-field scale exact. Below kDeviceRegionBExpCheapFrom the
// ladder's requirement (kRegionBExpBar / T(N, x)) is tighter than the polynomial's error, so
// this arm reads the library routine there - the polynomial alone fails a bound interior to it.

// The steps run at the build's multiply-add route; at the separate route the extra rounding is
// four orders below the 8.336e-11 the polynomial carries.
template <bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename T>
__device__ __forceinline__ T DeviceRegionBExp(T xx) {
    if constexpr (std::is_same_v<T, float>)
    {
        if constexpr (kFastExp)
        {
            const float y = -xx;
            const float t = y * 1.4426950408889634f;
            const float d = DeviceMulAdd<kRoute>(y, 1.4426950408889634f, -t);
            return 0.5f * __expf(y) * DeviceMulAdd<kRoute>(d, 0.6931471805599453f, 1.0f);
        } else
        {
            return 0.5f * expf(-xx);
        }
    }
    else
    {
        if constexpr (kFastExp)
        {
            if (xx < kDeviceRegionBExpCheapFrom)
            {
                return 0.5 * exp(-xx);
            }

            const double biased =
                DeviceMulAdd<kRoute>(xx, kDeviceRegionBExpLog2e, kDeviceRegionBExpRoundMagic);
            const double kd = biased - kDeviceRegionBExpRoundMagic;
            const double r = DeviceMulAdd<kRoute>(-kd, kDeviceRegionBExpLn2, xx);

            double p = kDeviceRegionBExpReduced[7];

            for (int k = 6; k >= 0; --k)
            {
                p = DeviceMulAdd<kRoute>(p, r, kDeviceRegionBExpReduced[k]);
            }

            const int k = static_cast<int>(kd);
            return 0.5 * __longlong_as_double(static_cast<long long>(1023 - k) << 52) * p;
        } else
        {
            return 0.5 * exp(-xx);
        }
    }
}

// ---------------------------------------------------------------------------
// the division form
// ---------------------------------------------------------------------------

// The three forms of a ladder step's division, as the host's bodies spell them (boys_impl.hpp,
// DivideStep): correctly rounded, a product by a rounded reciprocal, and the refinement that
// recovers the correctly rounded quotient with one fused multiply-add per step. The form is a
// template argument, never a run-time branch, so a probe times one form at a time.

// The two fused multiply-adds, named per value type: the refinement is only the
// correctly rounded quotient when the step is one instruction, so both are the
// round-to-nearest fused intrinsics and never a bare a * b + c.
template <typename T>
__device__ __forceinline__ T DeviceFma(T a, T b, T c) {
    if constexpr (std::is_same_v<T, float>)
    {
        return __fmaf_rn(a, b, c);
    } else
    {
        return __fma_rn(a, b, c);
    }
}

// The exact form: the correctly rounded quotient, and what a published bound over a
// region is stated for.
template <typename T>
__device__ __forceinline__ T DeviceDivideExact(T a, T x) {
    return a / x;
}

// The plain form: the quotient through the divisor's reciprocal. It rounds twice
// where the exact form rounds once, so a step may differ by an ulp and a ladder of
// them accumulates the difference.
template <typename T>
__device__ __forceinline__ T DeviceDividePlain(T a, T invx) {
    return a * invx;
}

// The refined form: the plain product, then the classical refinement. An infinite divisor is the
// one argument where the recovery cannot run - the residual a - quotient * x is zero there, so
// the first fused multiply-add hands back a NaN the second spreads - and the branch costs
// nothing, a finite numerator over an infinite divisor being the signed zero the product is.
template <typename T>
__device__ __forceinline__ T DeviceDivideByReciprocal(T a, T x, T invx) {
    const T quotient = a * invx;

    if (isinf(x))
    {
        return quotient;
    }

    return DeviceFma(DeviceFma(-quotient, x, a), invx, quotient);
}

// One divide in the form named, taking whichever of the divisor and its reciprocal
// that form reads.
template <DivisionForm kForm, typename T>
__device__ __forceinline__ T DeviceDivideStep(T a, T x, T invx) {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(invx);
        return DeviceDivideExact(a, x);
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        static_cast<void>(x);
        return DeviceDividePlain(a, invx);
    } else
    {
        return DeviceDivideByReciprocal(a, x, invx);
    }
}

// The reciprocal a form needs, formed once per call and multiplied through the
// ladder. The exact form's is not formed at all: the three forms are ranked on the
// work each actually does, so a form that divides must not also pay for a
// reciprocal it never reads.
template <DivisionForm kForm, typename T>
__device__ __forceinline__ T DeviceStepReciprocal(T x) {
    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        static_cast<void>(x);
        return T{0};
    } else
    {
        return T{1} / x;
    }
}

// The downward step's divisor l + 1/2 as a table of its reciprocals, exact in both value types
// at every order the recurrences run to (boys_impl.hpp, kDownwardReciprocals).

// A raw member array of a device variable, not a std::array of a host one: std::array's element
// access is a constexpr HOST function a __device__ body may not call, a host constexpr variable
// is undefined in device code, and a __device__ variable template may not be const-qualified on
// Windows.
template <typename T>
struct DeviceDownwardReciprocalTable {
    T values[static_cast<std::size_t>(kMaxBoysOrder) + 1];
};

template <typename T>
constexpr DeviceDownwardReciprocalTable<T> MakeDeviceDownwardReciprocals() noexcept {
    DeviceDownwardReciprocalTable<T> table{};

    for (int l = 0; l <= kMaxBoysOrder; ++l)
    {
        table.values[static_cast<std::size_t>(l)] = T{1} / (static_cast<T>(l) + T{0.5});
    }

    return table;
}

__device__ constexpr DeviceDownwardReciprocalTable<double> kDeviceDownwardReciprocalsF64 =
    MakeDeviceDownwardReciprocals<double>();
__device__ constexpr DeviceDownwardReciprocalTable<float> kDeviceDownwardReciprocalsF32 =
    MakeDeviceDownwardReciprocals<float>();

/// The reciprocal of \c l + 1/2, for the value type the ladder runs in.
template <typename T>
__device__ __forceinline__ T DeviceDownwardReciprocal(std::size_t index) noexcept {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>,
                  "the downward ladder runs in the double and float value types; a table of "
                  "another type's reciprocals is not stated (boys_cuda_arithmetic.hpp)");

    if constexpr (std::is_same_v<T, float>)
    {
        return kDeviceDownwardReciprocalsF32.values[index];
    } else
    {
        return kDeviceDownwardReciprocalsF64.values[index];
    }
}

// The downward step's divide, in the form named.
template <DivisionForm kForm, typename T>
__device__ __forceinline__ T DeviceDivideDownwardStep(int l, T a) {
    const std::size_t index = static_cast<std::size_t>(l);

    if constexpr (kForm == DivisionForm::kExactDivision)
    {
        return DeviceDivideExact(a, static_cast<T>(l) + T{0.5});
    } else if constexpr (kForm == DivisionForm::kPlainReciprocal)
    {
        return DeviceDividePlain(a, DeviceDownwardReciprocal<T>(index));
    } else
    {
        return DeviceDivideByReciprocal(a, static_cast<T>(l) + T{0.5},
                                        DeviceDownwardReciprocal<T>(index));
    }
}

// ---------------------------------------------------------------------------
// one order at one argument
// ---------------------------------------------------------------------------

// Region A asks the fit for the order directly; the higher regions seed F_0 and recur upward once
// per order - a fit of every order over the whole range would cost more table than the recursion
// costs work. kFastExp selects the region-B exponential (RegionBExp, boys/accuracy.hpp) and nothing
// else: region A is not region B, so its own 0.5 * exp(-xx) is the accurate member.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane>
__device__ __forceinline__ double DeviceSingleF64(const Lane& lane, int order, double xx) {
    if (xx < kX0)
    {
        return DeviceSeed<kRoute>(lane, order, xx);
    }

    if (xx < kX1)
    {
        double f = lane.BSeed(xx, order);
        const double expx = DeviceRegionBExp<kFastExp, kRoute>(xx);
        const double invx = DeviceStepReciprocal<kForm>(xx);

        for (int l = 0; l < order; ++l)
        {
            f = DeviceDivideStep<kForm>((l + 0.5) * f - expx, xx, invx);
        }

        return f;
    }

    double f = kHalfSqrtPi * rsqrt(xx);
    const double invx = DeviceStepReciprocal<kForm>(xx);

    for (int l = 0; l < order; ++l)
    {
        f = DeviceDivideStep<kForm>((l + 0.5) * f, xx, invx);
    }

    return f;
}

// The float lane's single body, and the fp16 lane's: the fp16 entries round
// at the boundary and run this engine in between, so there is one body and
// not two.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane>
__device__ __forceinline__ float DeviceSingleF32(const Lane& lane, int order, float xx) {
    if (xx < static_cast<float>(kX0))
    {
        return DeviceSeed32<kRoute>(lane, order, xx);
    }

    if (xx < static_cast<float>(kX1))
    {
        float f = lane.BSeed(xx, order);
        const float expx = DeviceRegionBExp<kFastExp, kRoute>(xx);
        const float invx = DeviceStepReciprocal<kForm>(xx);

        for (int l = 0; l < order; ++l)
        {
            f = DeviceDivideStep<kForm>((l + 0.5f) * f - expx, xx, invx);
        }

        return f;
    }

    float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    const float invx = DeviceStepReciprocal<kForm>(xx);

    for (int l = 0; l < order; ++l)
    {
        f = DeviceDivideStep<kForm>((l + 0.5f) * f, xx, invx);
    }

    return f;
}

// ---------------------------------------------------------------------------
// every order at one argument
// ---------------------------------------------------------------------------

// `store(int order, double value)` is called once per order, in the order the recursion produces
// it, so a batch kernel writes each value straight to its plane and a caller keeps the ladder in
// registers. kFastExp selects the region-B exponential and nothing else: region A's own
// 0.5 * exp(-xx) is the accurate member at every value of kFastExp.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF64(
    const Lane& lane, int order, double xx, Store store) {
    if (xx < kX0)
    {
        double f = DeviceSeed<kRoute>(lane, order, xx);
        store(order, f);
        const double expx = 0.5 * exp(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = DeviceDivideDownwardStep<kForm>(l, xx * f + expx);
            store(l, f);
        }

        return;
    }

    if (xx < kX1)
    {
        double f = lane.BSeed(xx, order);
        store(0, f);
        const double expx = DeviceRegionBExp<kFastExp, kRoute>(xx);
        const double invx = DeviceStepReciprocal<kForm>(xx);

        for (int l = 1; l <= order; ++l)
        {
            f = DeviceDivideStep<kForm>((l - 0.5) * f - expx, xx, invx);
            store(l, f);
        }

        return;
    }

    double f = kHalfSqrtPi * rsqrt(xx);
    store(0, f);
    const double invx = DeviceStepReciprocal<kForm>(xx);

    for (int l = 1; l <= order; ++l)
    {
        f = DeviceDivideStep<kForm>((l - 0.5) * f, xx, invx);
        store(l, f);
    }
}

// The region-A seed is the DOUBLE piece table even on the float lanes - the downward recursion
// amplifies a float seed error past the float budget - computed in double and rounded once on
// entry, which is what the second lane argument is. kFastExp selects the region-B exponential
// and nothing else.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename SeedLane, typename Lane, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, Store store) {
    if (xx < static_cast<float>(kX0))
    {
        const double seed = DeviceSeed<kRoute>(seedLane, order, static_cast<double>(xx));
        float f = static_cast<float>(seed);
        store(order, f);
        const float expx = 0.5f * expf(-xx);

        for (int l = order - 1; l >= 0; --l)
        {
            f = DeviceDivideDownwardStep<kForm>(l, xx * f + expx);
            store(l, f);
        }

        return;
    }

    if (xx < static_cast<float>(kX1))
    {
        float f = lane.BSeed(xx, order);
        store(0, f);
        const float expx = DeviceRegionBExp<kFastExp, kRoute>(xx);
        const float invx = DeviceStepReciprocal<kForm>(xx);

        for (int l = 1; l <= order; ++l)
        {
            f = DeviceDivideStep<kForm>((l - 0.5f) * f - expx, xx, invx);
            store(l, f);
        }

        return;
    }

    float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
    store(0, f);
    const float invx = DeviceStepReciprocal<kForm>(xx);

    for (int l = 1; l <= order; ++l)
    {
        f = DeviceDivideStep<kForm>((l - 0.5f) * f, xx, invx);
        store(l, f);
    }
}

// ---------------------------------------------------------------------------
// every order at one argument, from one uniform grid
// ---------------------------------------------------------------------------

// The route whose fit is one table of equal intervals over [0, kFlatHi): an interval is one
// multiply and a truncation, and the degree is a property of the stored table rather than of the
// argument. The geometry is compile-time, so these bodies take the two pools and nothing else.

// kMonomial selects the reader and nothing else - the index arithmetic, the join at kFlatHi and
// the arm above it are one spelling for both forms - and the join needs no interpolation:
// kFlatHi is above kX1, so an argument the table does not serve is one the asymptotic served.

// The uniform route's ladder, in double.
template <DivisionForm kForm, bool kMonomial, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF64Flat(const double* cheb,
                                                       const double* mono,
                                                       const int* degs,
                                                       const int* offsets,
                                                       int order,
                                                       double xx,
                                                       Store store) {
    if (xx >= kFlatHi)
    {
        double f = kHalfSqrtPi * rsqrt(xx);
        const double invx = DeviceStepReciprocal<kForm>(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = DeviceDivideStep<kForm>((l + 0.5) * f, xx, invx);
        }

        return;
    }

    // The interval index is the argument times the reciprocal of the stored width, truncated;
    // the reciprocal is exact here, so the product is the interval's own boundary and not a
    // rounding of it, and the two assertions below hold the factor and the grid to the stored
    // width. The clamp is unreachable below the join, kept as the guard the host map keeps.
    constexpr double kPerUnit = 1.0 / kFlatWidth;
    static_assert(kFlatWidth * kPerUnit == 1.0,
                  "the uniform grid's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * kFlatHi == static_cast<double>(kFlatIntervals),
                  "the uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const double u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > kFlatIntervals - 1)
    {
        iv = kFlatIntervals - 1;
    }

    const double t = 2.0 * (u - static_cast<double>(iv)) - 1.0;

    // The interval's own block at the interval's own degree, read per interval because the cells
    // do not all carry the same count: each holds the smallest admissible even degree its own
    // truncation bound allows, so the table has no stride a reader could assume. Assuming one
    // would sum a neighbouring cell's polynomial, which no check of the coefficients reports.
    const int deg = degs[iv];
    const double* interval = (kMonomial ? mono : cheb) + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const double* c =
            interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        if constexpr (kMonomial)
        {
            store(l, DeviceHornerMono<kRoute>(c, deg, t));
        }
        else
        {
            store(l, DeviceClenshawSplit<kRoute>(c, deg, t));
        }
    }
}

// The double lane's uniform grid on its RATIONAL route: one numerator/denominator pair per
// interval, read at the mapped argument DeviceAllOrdersF64Flat builds and at the interval's own
// pair, in the stored form DeviceRatSum reads. A body of its own because the two routes store
// different things: a block is addressed at its own pair and its own stored count, not one degree.

// The four per-interval tables are the emitter's (tools/gen_boys_coefficients.py,
// flat_rat_block_lines): the two degrees, the stored count of one row, and where the block
// starts. Assume one stride and a reader takes a neighbouring interval's pair, which no check of
// the coefficients alone would report.

// The summation is DeviceRatSum, so the figures the host gate certifies are this entry's too.
template <DivisionForm kForm, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF64FlatRat(const double* rat,
                                                          const int* numDegs,
                                                          const int* denDegs,
                                                          const int* storeds,
                                                          const int* offsets,
                                                          int order,
                                                          double xx,
                                                          Store store) {
    if (xx >= kFlatHi)
    {
        double f = kHalfSqrtPi * rsqrt(xx);
        const double invx = DeviceStepReciprocal<kForm>(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = DeviceDivideStep<kForm>((l + 0.5) * f, xx, invx);
        }

        return;
    }

    constexpr double kPerUnit = 1.0 / kFlatWidth;
    static_assert(kFlatWidth * kPerUnit == 1.0,
                  "the uniform grid's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * kFlatHi == static_cast<double>(kFlatIntervals),
                  "the uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const double u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > kFlatIntervals - 1)
    {
        iv = kFlatIntervals - 1;
    }

    const double t = 2.0 * (u - static_cast<double>(iv)) - 1.0;

    const int m = numDegs[iv];
    const int k = denDegs[iv];
    const int stored = storeds[iv];
    const double* interval = rat + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const double* c = interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(stored);

        store(l, DeviceRatSum<kRoute>(c, m, c + m + 1, k, t));
    }
}

// The float lane's own map, not the double body's: 2 (x - a)/(b - a) - 1 with a, b the interval's
// own edges. One map for the lane (tools/gen_boys_coefficients.py, f32_map, and the narrow
// pieces' reader ChebyshevValueF32), on the grid's own edges as the generator fitted them, so the
// interval an argument is mapped inside is the one the fit was measured in.
template <DivisionForm kForm, bool kMonomial, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF32Flat(const float* cheb,
                                                       const float* mono,
                                                       const int* degs,
                                                       const int* offsets,
                                                       int order,
                                                       float xx,
                                                       Store store) {
    if (xx >= f32::kFlatHiF32)
    {
        float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
        const float invx = DeviceStepReciprocal<kForm>(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = DeviceDivideStep<kForm>((l + 0.5f) * f, xx, invx);
        }

        return;
    }

    // The lane's own factor, and the host lane's spelling of it: the width's
    // reciprocal, which the locate multiplies by, and which the grid's own
    // identity below is written against. The two are one requirement, and
    // boys_impl.hpp asserts the same product from the host side.
    constexpr float kPerUnit = 1.0f / f32::kFlatWidthF32;
    static_assert(f32::kFlatWidthF32 * kPerUnit == 1.0f,
                  "the float lane's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * f32::kFlatHiF32 == static_cast<float>(f32::kFlatIntervalsF32),
                  "the float lane's uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const float u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > f32::kFlatIntervalsF32 - 1)
    {
        iv = f32::kFlatIntervalsF32 - 1;
    }

    const float a = static_cast<float>(iv) / kPerUnit;
    const float b = static_cast<float>(iv + 1) / kPerUnit;
    const float t = 2.0f * (xx - a) / (b - a) - 1.0f;

    // The interval's own block at the interval's own degree, as the double
    // body reads it and for the same reason: this lane's cells carry their own
    // degrees too, so there is no stride to read them with.
    const int deg = degs[iv];
    const float* interval = (kMonomial ? mono : cheb) + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const float* c =
            interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(deg + 1);

        if constexpr (kMonomial)
        {
            store(l, DeviceHornerMono32<kRoute>(c, deg, t));
        }
        else
        {
            store(l, DeviceClenshawSplit32<kRoute>(c, deg, t));
        }
    }
}

// The float lane's uniform grid on its RATIONAL route: the double body above at this lane's
// own width, grid and tables, with DeviceRatSum32. Every figure the host gate certifies here
// was measured on the mapping this locate builds (tools/gen_boys_coefficients.py, f32_map) at
// BOTH multiply-add routes, which reach this summation too.
template <DivisionForm kForm, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Store>
__device__ __forceinline__ void DeviceAllOrdersF32FlatRat(const float* rat,
                                                          const int* numDegs,
                                                          const int* denDegs,
                                                          const int* storeds,
                                                          const int* offsets,
                                                          int order,
                                                          float xx,
                                                          Store store) {
    if (xx >= f32::kFlatHiF32)
    {
        float f = static_cast<float>(kHalfSqrtPi) * rsqrtf(xx);
        const float invx = DeviceStepReciprocal<kForm>(xx);

#pragma unroll 4
        for (int l = 0; l <= order; ++l)
        {
            store(l, f);
            f = DeviceDivideStep<kForm>((l + 0.5f) * f, xx, invx);
        }

        return;
    }

    constexpr float kPerUnit = 1.0f / f32::kFlatWidthF32;
    static_assert(f32::kFlatWidthF32 * kPerUnit == 1.0f,
                  "the float lane's index factor must be the reciprocal of its stored width: "
                  "the locate is one multiply by this factor and a truncation, so a width whose "
                  "reciprocal does not round back reads the table one interval off its cell");
    static_assert(kPerUnit * f32::kFlatHiF32 == static_cast<float>(f32::kFlatIntervalsF32),
                  "the float lane's uniform grid must reach kFlatHi in kFlatIntervals intervals");

    const float u = xx * kPerUnit;
    int iv = static_cast<int>(u);

    if (iv > f32::kFlatIntervalsF32 - 1)
    {
        iv = f32::kFlatIntervalsF32 - 1;
    }

    const float a = static_cast<float>(iv) / kPerUnit;
    const float b = static_cast<float>(iv + 1) / kPerUnit;
    const float t = 2.0f * (xx - a) / (b - a) - 1.0f;

    const int m = numDegs[iv];
    const int k = denDegs[iv];
    const int stored = storeds[iv];
    const float* interval = rat + static_cast<std::size_t>(offsets[iv]);

#pragma unroll 4
    for (int l = 0; l <= order; ++l)
    {
        const float* c = interval + static_cast<std::size_t>(l) * static_cast<std::size_t>(stored);

        store(l, DeviceRatSum32<kRoute>(c, m, c + m + 1, k, t));
    }
}

// ---------------------------------------------------------------------------
// every order at one argument, each from its own fit
// ---------------------------------------------------------------------------

// The same ladder with region A read the other way: every order's own piece is located and its
// own fit summed, so no value here came down a recurrence from a higher order's fit. The two
// agree to the fit's own accuracy and differ in where the rounding happens.

// The axis covers region A and nothing else: past kX0 this body is the certified one above, so a
// row carrying it names its interval as region A and not the rest.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename Lane, typename Store>
__device__ __forceinline__ void DeviceOrdersF64(
    const Lane& lane, int order, double xx, Store store) {
    if (xx < kX0)
    {
        for (int l = 0; l <= order; ++l)
        {
            store(l, DeviceSeed<kRoute>(lane, l, xx));
        }

        return;
    }

    DeviceAllOrdersF64<kForm, kFastExp, kRoute>(lane, order, xx, store);
}

// The float lane's and the fp16 lane's, over the double region-A seed lane the
// body above takes for the same reason.
template <DivisionForm kForm, bool kFastExp, backend::MulAddRoute kRoute = kDeviceMulAddRoute, typename SeedLane, typename Lane, typename Store>
__device__ __forceinline__ void DeviceOrdersF32(
    const SeedLane& seedLane, const Lane& lane, int order, float xx, Store store) {
    if (xx < static_cast<float>(kX0))
    {
        for (int l = 0; l <= order; ++l)
        {
            store(l, static_cast<float>(DeviceSeed<kRoute>(seedLane, l, static_cast<double>(xx))));
        }

        return;
    }

    DeviceAllOrdersF32<kForm, kFastExp, kRoute>(seedLane, lane, order, xx, store);
}

} // namespace boys::detail

/// \endcond
