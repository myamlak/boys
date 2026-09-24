#include "boys_orders_simd.hpp"

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

// --- Architecture guard -----------------------------------------------------
//
// The same one boys_simd.cpp carries, for the same reason and with the same
// three answers: the build states BOYS_SIMD_X86 from its configure-time probe,
// an unstated x86_64 target is detected from the compiler's own predefines,
// and anything else is an error rather than a silently scalar lane. CMake
// compiles this TU with the same /arch:AVX2 (MSVC) or -mavx2 -mfma (GCC/Clang)
// set that TU gets. See boys_simd.cpp for the full argument.
#ifdef BOYS_SIMD_X86

// The build answered; nothing to detect.

#elif defined(__x86_64__) || defined(_M_X64)

#define BOYS_SIMD_X86 1

#else

#error "BOYS_SIMD_X86 is unset and this is not x86_64: define it (1 on x86_64, 0 elsewhere)"

#endif

#if BOYS_SIMD_X86

#include <immintrin.h>

namespace boys::detail {
namespace {

// --- The premise the lane rests on ------------------------------------------
//
// Every order's region-A fit is cut at the same boundaries to the same degree.
// A fixed x therefore selects the same piece index, the same mapped argument
// and the same degree for every order, and the orders differ only in where
// their coefficients begin. That is what lets the vector carry four orders
// with no masking, no divergence and no per-lane degree: one t, one degree,
// one recurrence, and a coefficient fetch that is a single stride.
//
// It is a property of the generated table rather than a given, so it is
// checked against the table: a regenerated table with per-order degrees or
// per-order splits sends the entry to the certified scalar fits instead of to
// a wrong answer.
bool PiecesShareShape() noexcept {
    const int perOrder = kPieceStart[1] - kPieceStart[0];

    if (perOrder < 1)
    {
        return false;
    }

    for (int n = 0; n <= kMaxBoysOrder; ++n)
    {
        if (kPieceStart[n + 1] - kPieceStart[n] != perOrder)
        {
            return false;
        }
    }

    for (int p = 0; p < perOrder; ++p)
    {
        const OrderPiece& shape = kPieces[kPieceStart[0] + p];

        for (int n = 1; n <= kMaxBoysOrder; ++n)
        {
            const OrderPiece& other = kPieces[kPieceStart[n] + p];

            if (other.a != shape.a || other.b != shape.b || other.deg != shape.deg)
            {
                return false;
            }
        }
    }

    return true;
}

bool UniformPieces() noexcept {
    static const bool uniform = PiecesShareShape();
    return uniform;
}

// The coefficients one step of a summation reads, for the four orders the
// vector carries, at the stride the table already has between one order's
// coefficients and the next order's.
//
// Two ways to fetch them, and they are not equivalent. The gather is one
// instruction and, on the machines measured here, a microcode assist worth
// many retirement slots; the composed form is four loads and the shuffles to
// join them, which is more instructions and no assist. Which of the two costs
// less is a property of the machine rather than of the algorithm, so both are
// here and the benchmark measures them.
template <bool kComposed>
__m256d StepCoefficients(const double* base, int orderStride, __m128i step, int k) noexcept {
    if constexpr (kComposed)
    {
        return _mm256_set_pd(
            base[3 * orderStride + k], base[2 * orderStride + k], base[orderStride + k], base[k]);
    } else
    {
        return _mm256_i32gather_pd(base + k, step, 8);
    }
}

// The same fetch for the scalar tail, where the four lanes are one order.
struct BroadcastCoefficients {
    const double* base;

    __m256d operator()(int k) const noexcept {
        return _mm256_set1_pd(base[k]);
    }
};

// Split Clenshaw, transcribed from boys_impl.hpp's ClenshawSplit onto gathered
// coefficients: same steps, same order, same fused operations, so a lane's
// value is the across-arguments lane's value for that order, bit for bit. The
// across-orders test asserts exactly that, which is what holds the two bodies
// in lockstep.
template <class C> __m256d ClenshawSplitGathered(C coeff, int deg, __m256d t) noexcept {
    if (deg == 0)
    {
        return coeff(0);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_pd(t, coeff(1), coeff(0));
    }

    const __m256d v =
        _mm256_fmadd_pd(_mm256_set1_pd(2.0), _mm256_mul_pd(t, t), _mm256_set1_pd(-1.0));
    const __m256d twoV = _mm256_add_pd(v, v);

    if (deg == 2)
    {
        return _mm256_fmadd_pd(t, coeff(1), _mm256_fmadd_pd(v, coeff(2), coeff(0)));
    }

    assert(deg >= 4 && deg % 2 == 0);

    const int m = deg / 2;
    __m256d b1 = coeff(2 * m);
    __m256d b2 = _mm256_setzero_pd();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256d b0 = _mm256_fmadd_pd(twoV, b1, _mm256_sub_pd(coeff(2 * k), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256d even = _mm256_fmadd_pd(v, b1, _mm256_sub_pd(coeff(0), b2));

    __m256d o1 = coeff(2 * m - 1);
    __m256d o2 = _mm256_setzero_pd();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256d o0 = _mm256_fmadd_pd(twoV, o1, _mm256_sub_pd(coeff(2 * k + 1), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256d odd =
        _mm256_fmadd_pd(_mm256_sub_pd(twoV, _mm256_set1_pd(1.0)), o1, _mm256_sub_pd(coeff(1), o2));
    return _mm256_fmadd_pd(t, odd, even);
}

// The direct sum: the Chebyshev series term by term, T_k by the forward
// recurrence T_k = 2t T_{k-1} - T_{k-2}. Two multiply-adds per coefficient
// against the split Clenshaw's one, and no dependence between the terms - the
// shape an across-orders vector has to fill, since it already carries the
// orders the terms belong to.
template <class C> __m256d ChebyshevDirectSum(C coeff, int deg, __m256d t) noexcept {
    if (deg == 0)
    {
        return coeff(0);
    }

    __m256d sum = _mm256_fmadd_pd(t, coeff(1), coeff(0));

    if (deg == 1)
    {
        return sum;
    }

    const __m256d twoT = _mm256_add_pd(t, t);
    __m256d prev = _mm256_set1_pd(1.0); // T_0
    __m256d cur = t; // T_1

    for (int k = 2; k <= deg; ++k)
    {
        const __m256d next = _mm256_fmsub_pd(twoT, cur, prev);
        sum = _mm256_fmadd_pd(coeff(k), next, sum);
        prev = cur;
        cur = next;
    }

    return sum;
}

// Horner over the monomial form of the fit, transcribed from HornerMono onto
// gathered coefficients.
template <class C> __m256d HornerGathered(C coeff, int deg, __m256d t) noexcept {
    __m256d acc = coeff(deg);

    for (int k = deg - 1; k >= 0; --k)
    {
        acc = _mm256_fmadd_pd(acc, t, coeff(k));
    }

    return acc;
}

// One order's fit at one argument, in the lane's own arithmetic and mapping:
// the body the scalar tail runs so that its values are the vector's, not the
// scalar lane's.
template <OrdersScheme kScheme> double ScalarFit(const double* base, int deg, double t) noexcept {
    const BroadcastCoefficients coeff{base};
    const __m256d tv = _mm256_set1_pd(t);
    double lanes[4];

    if constexpr (kScheme == OrdersScheme::kHorner)
    {
        _mm256_storeu_pd(lanes, HornerGathered(coeff, deg, tv));
    } else if constexpr (kScheme == OrdersScheme::kDirectSum)
    {
        _mm256_storeu_pd(lanes, ChebyshevDirectSum(coeff, deg, tv));
    } else
    {
        _mm256_storeu_pd(lanes, ClenshawSplitGathered(coeff, deg, tv));
    }

    return lanes[0];
}

// The region-A body: the shared geometry once, then vector groups of four
// orders and a scalar tail for the remainder.
template <OrdersScheme kScheme, bool kComposed>
void OrdersBody(int nmax, double x, double* out, std::size_t stride) noexcept {
    // Both tables are cut identically, so the geometry is the same whichever
    // one the scheme reads: a scheme picks a table and a summation.
    const double* const table =
        (kScheme == OrdersScheme::kHorner) ? kMonoCoeffs.data() : kCoeffs.data();

    const OrderPiece& piece = FindPiece(0, x);
    const int index = static_cast<int>(&piece - kPieces.data()) - kPieceStart[0];
    const int orderStride =
        kPieces[kPieceStart[1] + index].offset - piece.offset; // equal to nmax for the rest

    const double* const base = table + piece.offset;
    const double t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);

    const __m128i step = _mm_set_epi32(3 * orderStride, 2 * orderStride, orderStride, 0);
    const __m256d tv = _mm256_set1_pd(t);

    int l = 0;

    if (nmax >= 3)
    {
        for (; l + 3 <= nmax; l += 4)
        {
            // The group's four orders start at this order's coefficients; the
            // stride between them and the next order's is the table's.
            const double* const groupBase = base + static_cast<std::ptrdiff_t>(l) * orderStride;
            const auto coeff = [&](int k) {
                return StepCoefficients<kComposed>(groupBase, orderStride, step, k);
            };
            __m256d value;

            if constexpr (kScheme == OrdersScheme::kHorner)
            {
                value = HornerGathered(coeff, piece.deg, tv);
            } else if constexpr (kScheme == OrdersScheme::kDirectSum)
            {
                value = ChebyshevDirectSum(coeff, piece.deg, tv);
            } else
            {
                value = ClenshawSplitGathered(coeff, piece.deg, tv);
            }

            if (stride == 1)
            {
                _mm256_storeu_pd(out + l, value);
            } else
            {
                // No scatter in AVX2: the plane's order stride costs one
                // store per lane, which is the across-orders axis's own price
                // on that shape.
                alignas(32) double lanes[4];
                _mm256_store_pd(lanes, value);

                for (int j = 0; j < 4; ++j)
                {
                    out[static_cast<std::size_t>(l + j) * stride] = lanes[j];
                }
            }
        }
    }

    for (; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] =
            ScalarFit<kScheme>(base + static_cast<std::ptrdiff_t>(l) * orderStride, piece.deg, t);
    }
}

template <bool kComposed>
void OrdersByScheme(OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) {
    switch (scheme)
    {
    case OrdersScheme::kDirectSum:
        OrdersBody<OrdersScheme::kDirectSum, kComposed>(nmax, x, out, stride);
        break;

    case OrdersScheme::kHorner:
        OrdersBody<OrdersScheme::kHorner, kComposed>(nmax, x, out, stride);
        break;

    case OrdersScheme::kSplitClenshaw:
    default:
        OrdersBody<OrdersScheme::kSplitClenshaw, kComposed>(nmax, x, out, stride);
        break;
    }
}

// Whether the lane applies at this argument: inside the stored fits'
// interval, on a machine whose tier is present, against a table that carries
// the lane's premise.
bool OrdersLaneApplies(double x) noexcept {
    return !(x >= kX0) && BoysAvx2Available() && UniformPieces();
}

void ScalarOrders(int nmax, double x, double* out, std::size_t stride) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<kBoysFullAccuracyMultiplier>(l, x);
    }
}

} // namespace

// The zero the library states in closed form, and the fallback outside the
// lane's domain: outside the stored fits' interval, on a machine without the
// tier, or against a table that does not carry the lane's premise, the entry
// is the certified scalar lanes, one order at a time. Both are shared by the
// two fetch entries.
bool OrdersShortcut(int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);

    if (x == 0.0)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            out[static_cast<std::size_t>(l) * stride] = 1.0 / (2.0 * l + 1.0);
        }

        return true;
    }

    if (!OrdersLaneApplies(x))
    {
        ScalarOrders(nmax, x, out, stride);
        return true;
    }

    return false;
}

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(out != nullptr);
    assert(stride >= 1);

    if (!OrdersShortcut(nmax, x, out, stride))
    {
        OrdersByScheme<false>(scheme, nmax, x, out, stride);
    }
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(out != nullptr);
    assert(stride >= 1);

    if (!OrdersShortcut(nmax, x, out, stride))
    {
        OrdersByScheme<true>(scheme, nmax, x, out, stride);
    }
}

// The entry the public surface's orders axis dispatches to (boys_impl.hpp). The
// stored fit is summed composed rather than gathered: the two fetches are the
// same lane value for value, and the composed one is the cheaper of the two in
// the counter that decides - retired slots - on the machine this lane was
// measured on, where the gather is a microcode assist worth several slots per
// step. The gathered entry stays beside it so the pair remains measurable.
template <EvalScheme kScheme>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    constexpr OrdersScheme kOrdersScheme =
        (kScheme == EvalScheme::kHorner) ? OrdersScheme::kHorner : OrdersScheme::kSplitClenshaw;

    BoysAllOrdersSimdComposed(kOrdersScheme, nmax, x, out, 1);
}

template void BoysAllOrdersPacked<kDefaultEvalScheme>(int, double, double*) noexcept;
template void BoysAllOrdersPacked<EvalScheme::kHorner>(int, double, double*) noexcept;

} // namespace boys::detail

#else // BOYS_SIMD_X86

// Non-x86 targets: the entry exists and is defined, against the certified
// scalar lanes, exactly as the region entries of boys_simd.cpp are. The
// vector tier is absent here, so the lane is the fits it would have
// vectorised.
namespace boys::detail {

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);
    assert(stride >= 1);

    static_cast<void>(scheme);

    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<kBoysFullAccuracyMultiplier>(l, x);
    }
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimd(scheme, nmax, x, out, stride);
}

// The orders axis's entry on a target without the vector tier: the same
// certified scalar single lane the packed bodies fall back to, so the option
// exists and is defined everywhere the library is.
template <EvalScheme kScheme>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    static_cast<void>(kScheme);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingle<kBoysFullAccuracyMultiplier>(l, x);
    }
}

template void BoysAllOrdersPacked<kDefaultEvalScheme>(int, double, double*) noexcept;
template void BoysAllOrdersPacked<EvalScheme::kHorner>(int, double, double*) noexcept;

} // namespace boys::detail

#endif // BOYS_SIMD_X86
