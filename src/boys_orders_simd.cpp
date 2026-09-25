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

// --- The degree a stored fit is read at -------------------------------------
//
// The lane's reading is an argument, not a property of the body: the two
// readers below differ only in the numbers they hand a group of four orders,
// and everything else about the group - the geometry, the fetch, the
// summation, the store - is the same under either. That is what makes a rung a
// table the lane reads rather than a second lane.

// The reference reading: every stored fit at its own stored degree.
struct StoredDegree {
    static int At(std::size_t, int stored) noexcept { return stored; }
};

// A rung's reading: every stored fit at the degree the truncation criterion
// certifies for that fit's piece at the rung's multiplier.
//
// The table is flat over the piece table and the pieces are cut the same way
// for every order (the lane's premise above), so one index names the piece and
// kPieceStart[order] + index names that order's own copy of it.
template <typename Table>
struct RungDegree {
    const Table& table;

    int At(std::size_t flat, int stored) const noexcept {
        static_cast<void>(stored);
        return table[flat];
    }
};

// Which summation of the polynomial table a policy's scheme names. The
// across-orders axis offers the direct sum beside the two the public
// enumeration carries; a public scheme picks its table and its summation here
// as it does on every other entry.
constexpr OrdersScheme OrdersSchemeOf(EvalScheme scheme) noexcept {
    return scheme == EvalScheme::kHorner ? OrdersScheme::kHorner : OrdersScheme::kSplitClenshaw;
}

// The degree a vector of four orders is read at: the largest of the four
// lanes' own. One split Clenshaw serves the whole vector from one degree, and
// reading an order at a degree above its own cut costs nothing that was
// claimed - the dropped-coefficient tail is non-increasing in the degree, so a
// longer series is a smaller tail - which is why the group carries no more
// than the shortest lane's own cut does.
template <class Degrees>
int GroupDegree(Degrees degrees, std::size_t flat, int pieceStride, int stored) noexcept {
    int deg = degrees.At(flat, stored);

    for (int j = 1; j < 4; ++j)
    {
        const int own = degrees.At(flat + static_cast<std::size_t>(j) * pieceStride, stored);
        deg = own > deg ? own : deg;
    }

    return deg;
}

// One vector of four orders' values from a polynomial table, the group's first
// order being l.
template <OrdersScheme kScheme, bool kComposed>
__m256d ShippedGroup(
    const double* base, int l, int orderStride, int deg, __m128i step, __m256d tv) noexcept {
    const double* const groupBase = base + static_cast<std::ptrdiff_t>(l) * orderStride;
    const auto coeff = [&](int k) {
        return StepCoefficients<kComposed>(groupBase, orderStride, step, k);
    };

    if constexpr (kScheme == OrdersScheme::kHorner)
    {
        return HornerGathered(coeff, deg, tv);
    } else if constexpr (kScheme == OrdersScheme::kDirectSum)
    {
        return ChebyshevDirectSum(coeff, deg, tv);
    } else
    {
        return ClenshawSplitGathered(coeff, deg, tv);
    }
}

// A group of four orders into the caller's layout.
void StoreGroup(double* out, int l, std::size_t stride, __m256d value) noexcept {
    if (stride == 1)
    {
        _mm256_storeu_pd(out + l, value);
        return;
    }

    // No scatter in AVX2: the plane's order stride costs one store per lane,
    // which is the across-orders axis's own price on that shape.
    alignas(32) double lanes[4];
    _mm256_store_pd(lanes, value);

    for (int j = 0; j < 4; ++j)
    {
        out[static_cast<std::size_t>(l + j) * stride] = lanes[j];
    }
}

// The region-A body: the shared geometry once, then vector groups of four
// orders and a scalar tail for the remainder.
template <OrdersScheme kScheme, bool kComposed, class Degrees>
void OrdersBody(int nmax, double x, double* out, std::size_t stride, Degrees degrees) noexcept {
    // Both tables are cut identically, so the geometry is the same whichever
    // one the scheme reads: a scheme picks a table and a summation.
    const double* const table =
        (kScheme == OrdersScheme::kHorner) ? kMonoCoeffs.data() : kCoeffs.data();

    const OrderPiece& piece = FindPiece(0, x);
    const int index = static_cast<int>(&piece - kPieces.data()) - kPieceStart[0];
    const int orderStride =
        kPieces[kPieceStart[1] + index].offset - piece.offset; // equal to nmax for the rest
    const int pieceStride = kPieceStart[1] - kPieceStart[0];
    const std::size_t firstFlat = static_cast<std::size_t>(kPieceStart[0] + index);

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
            const std::size_t flat = firstFlat + static_cast<std::size_t>(l) * pieceStride;
            const int deg = GroupDegree(degrees, flat, pieceStride, piece.deg);

            StoreGroup(out,
                       l,
                       stride,
                       ShippedGroup<kScheme, kComposed>(base, l, orderStride, deg, step, tv));
        }
    }

    for (; l <= nmax; ++l)
    {
        const std::size_t flat = firstFlat + static_cast<std::size_t>(l) * pieceStride;

        out[static_cast<std::size_t>(l) * stride] = ScalarFit<kScheme>(
            base + static_cast<std::ptrdiff_t>(l) * orderStride,
            degrees.At(flat, piece.deg),
            t);
    }
}

template <bool kComposed>
void OrdersByScheme(OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) {
    const StoredDegree degrees{};

    switch (scheme)
    {
    case OrdersScheme::kDirectSum:
        OrdersBody<OrdersScheme::kDirectSum, kComposed>(nmax, x, out, stride, degrees);
        break;

    case OrdersScheme::kHorner:
        OrdersBody<OrdersScheme::kHorner, kComposed>(nmax, x, out, stride, degrees);
        break;

    case OrdersScheme::kSplitClenshaw:
    default:
        OrdersBody<OrdersScheme::kSplitClenshaw, kComposed>(nmax, x, out, stride, degrees);
        break;
    }
}

// --- The rational route on the orders axis ----------------------------------
//
// The other route's region-A fits are a numerator and a denominator Horner
// over the shipped pieces, in the same mapped argument, so the axis carries
// them with the shipped geometry and a different reading: two recurrences and
// one division. Their costs are not the shipped fits' costs - a pair is stored
// to its own numerator and denominator degrees and the route certifies a cut
// of both - so the cut is the order's own, and the four lanes of one vector
// need not share a degree. The fetch is masked per lane to match.
//
// The mapped argument is the route's own form, 2(x - a)/(b - a) - 1, rather
// than the fused form the shipped body maps with. Every value below is then
// the route's scalar value for the same piece and the same cut, bit for bit,
// which is what the axis's contract test holds it to.

// The steps a lane does not take: all-ones where the lane's cut is below k.
// The cuts are small integers, so the comparison is exact in double lanes and
// the mask is a lane-sized one rather than a packed integer that would have to
// be widened.
__m256d AboveCut(int k, __m256d cut) noexcept {
    return _mm256_cmp_pd(_mm256_set1_pd(static_cast<double>(k)), cut, _CMP_GT_OQ);
}

// The largest of four lanes' cut orders: a group's recurrences run from the
// longest lane's cut down, and every shorter lane is masked off above its own.
int LaneMax(const int* lanes) noexcept {
    int best = lanes[0];

    for (int j = 1; j < 4; ++j)
    {
        best = lanes[j] > best ? lanes[j] : best;
    }

    return best;
}

// Four orders' values from the stored pairs, each lane at its own cut.
//
// The mask is what makes one recurrence serve four different degrees: a lane's
// coefficients are live from its own cut down and zero above it, and a lane
// whose every step so far has been masked off still holds exactly zero, so its
// sequence is the scalar reading's own - same first coefficient, same order,
// same fused operations. The gathers are clamped to each piece's stored
// degrees so that a masked-off lane cannot read outside the coefficient table;
// the clamp is invisible to a live lane, whose cut never exceeds its piece's
// stored degree.
template <class Pairs>
__m256d RationalGroup(const double* coeffs,
                      std::size_t firstFlat,
                      int pieceStride,
                      int l,
                      const Pairs& pairs,
                      __m256d tv) noexcept {
    const __m128i flat = _mm_add_epi32(
        _mm_set1_epi32(static_cast<int>(firstFlat) + l * pieceStride),
        _mm_mullo_epi32(_mm_set_epi32(3, 2, 1, 0), _mm_set1_epi32(pieceStride)));
    const __m128i offset = _mm_i32gather_epi32(kRatAOffset.data(), flat, 4);
    const __m128i storedNum = _mm_i32gather_epi32(kRatANumDeg.data(), flat, 4);
    const __m128i storedDen = _mm_i32gather_epi32(kRatADenDeg.data(), flat, 4);
    const __m128i numCut = _mm_i32gather_epi32(pairs.num.data(), flat, 4);
    const __m128i denCut = _mm_i32gather_epi32(pairs.den.data(), flat, 4);
    const __m256d numCutD = _mm256_cvtepi32_pd(numCut);
    const __m256d denCutD = _mm256_cvtepi32_pd(denCut);
    int cutNum[4];
    int cutDen[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cutNum), numCut);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(cutDen), denCut);

    __m256d num = _mm256_setzero_pd();

    for (int k = LaneMax(cutNum); k >= 0; --k)
    {
        const __m128i idx = _mm_add_epi32(offset, _mm_min_epi32(_mm_set1_epi32(k), storedNum));
        const __m256d c = _mm256_i32gather_pd(coeffs, idx, 8);
        num = _mm256_fmadd_pd(num, tv, _mm256_andnot_pd(AboveCut(k, numCutD), c));
    }

    // The denominator is stored above the piece's FULL numerator, so the
    // denominator's coefficient j sits at the stored numerator's degree plus
    // j, whatever the two parts were cut to. A lane cut to no denominator at
    // all keeps the zero accumulator, and the closing multiply-add turns it
    // into exactly one, which is the route's own early return.
    __m256d den = _mm256_setzero_pd();

    for (int k = LaneMax(cutDen); k >= 1; --k)
    {
        const __m128i idx = _mm_add_epi32(_mm_add_epi32(offset, storedNum),
                                         _mm_min_epi32(_mm_set1_epi32(k), storedDen));
        const __m256d c = _mm256_i32gather_pd(coeffs, idx, 8);
        den = _mm256_fmadd_pd(den, tv, _mm256_andnot_pd(AboveCut(k, denCutD), c));
    }

    den = _mm256_fmadd_pd(den, tv, _mm256_set1_pd(1.0));
    return _mm256_div_pd(num, den);
}

// The rational route's region-A body: the shipped geometry, the per-order rule
// the route's per-argument body applies, and the route's own reading of the
// pairs where that rule hands an order to the route.
//
// The shipped part is read with the shipped body's own group rule, so an
// order's value below its end of region A does not depend on which route the
// policy names - only which orders the route answers does.
template <EvalScheme kScheme, class Pairs, class Degrees>
void RationalOrdersBody(int nmax,
                        double x,
                        double* out,
                        std::size_t stride,
                        const Pairs& pairs,
                        Degrees degrees) noexcept {
    constexpr OrdersScheme kOrdersScheme = OrdersSchemeOf(kScheme);

    const double* const shippedTable =
        (kScheme == EvalScheme::kHorner) ? kMonoCoeffs.data() : kCoeffs.data();
    const OrderPiece& piece = FindPiece(0, x);
    const int index = static_cast<int>(&piece - kPieces.data()) - kPieceStart[0];
    const int orderStride = kPieces[kPieceStart[1] + index].offset - piece.offset;
    const int pieceStride = kPieceStart[1] - kPieceStart[0];
    const std::size_t firstFlat = static_cast<std::size_t>(kPieceStart[0] + index);

    const double* const base = shippedTable + piece.offset;
    const double* const ratCoeffs = kRatACoeffs.data();
    const double t = 2.0 * (x - piece.a) / (piece.b - piece.a) - 1.0;

    const __m128i step = _mm_set_epi32(3 * orderStride, 2 * orderStride, orderStride, 0);
    const __m256d tv = _mm256_set1_pd(t);

    // The order from which the route's own fits are what the lane reads: the
    // route's per-argument body hands an order over at that order's own end of
    // region A, and the ends are non-decreasing in the order, so the orders the
    // route answers are a prefix.
    int served = 0;

    while (served <= nmax && x >= kTierThresholds[static_cast<std::size_t>(served)])
    {
        ++served;
    }

    const auto routeOrder = [&](int l) {
        const std::size_t flat = firstFlat + static_cast<std::size_t>(l) * pieceStride;
        return RationalPieceAtCut(flat, pairs.num[flat], pairs.den[flat], t);
    };

    const auto shippedOrder = [&](int l) {
        const std::size_t flat = firstFlat + static_cast<std::size_t>(l) * pieceStride;
        return ScalarFit<kOrdersScheme>(base + static_cast<std::ptrdiff_t>(l) * orderStride,
                                        degrees.At(flat, piece.deg),
                                        t);
    };

    int l = 0;

    if (nmax >= 3)
    {
        for (; l + 3 <= nmax; l += 4)
        {
            const std::size_t flat = firstFlat + static_cast<std::size_t>(l) * pieceStride;

            if (l >= served)
            {
                // None of the four takes the route's fits: the whole group is
                // the shipped reading.
                StoreGroup(out,
                           l,
                           stride,
                           ShippedGroup<kOrdersScheme, true>(base,
                                                            l,
                                                            orderStride,
                                                            GroupDegree(degrees,
                                                                        flat,
                                                                        pieceStride,
                                                                        piece.deg),
                                                            step,
                                                            tv));
            } else if (l + 4 <= served)
            {
                StoreGroup(out, l, stride, RationalGroup(ratCoeffs, firstFlat, pieceStride, l, pairs, tv));
            } else
            {
                // The handover falls inside this group: one order at a time, in
                // the same two readings the whole-group rules above use.
                for (int j = 0; j < 4; ++j)
                {
                    out[static_cast<std::size_t>(l + j) * stride] =
                        (l + j < served) ? routeOrder(l + j) : shippedOrder(l + j);
                }
            }
        }
    }

    for (; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] =
            (l < served) ? routeOrder(l) : shippedOrder(l);
    }
}

// Whether the lane applies at this argument: inside the stored fits'
// interval, on a machine whose tier is present, against a table that carries
// the lane's premise.
bool OrdersLaneApplies(double x) noexcept {
    return !(x >= kX0) && BoysAvx2Available() && UniformPieces();
}

// The certified scalar single lane at the policy the axis names, one order at
// a time: what the entry is outside the packed interval, on a host without the
// vector tier, and against a table that does not carry the lane's premise.
//
// The multiplier is the entry's own, so a rung falls back to the rung of the
// per-order lane rather than to the full-accuracy one. The budget is the
// policy's engine choice and this path is the double engine at every budget,
// so the fallback names the float budget the double entries are built with.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute>
void ScalarOrders(int nmax, double x, double* out, std::size_t stride) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<kAccuracyMultiplier, Policy>(l, x);
    }
}

} // namespace

// The zero the library states in closed form, and the fallback outside the
// lane's domain: outside the stored fits' interval, on a machine without the
// tier, or against a table that does not carry the lane's premise, the entry
// is the certified scalar lanes, one order at a time. Both are shared by the
// two fetch entries.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute>
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
        ScalarOrders<kScheme, kAccuracyMultiplier, kRoute>(nmax, x, out, stride);
        return true;
    }

    return false;
}

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(out != nullptr);
    assert(stride >= 1);

    if (!OrdersShortcut<kDefaultEvalScheme, kBoysFullAccuracyMultiplier, kDefaultFitRoute>(
            nmax, x, out, stride))
    {
        OrdersByScheme<false>(scheme, nmax, x, out, stride);
    }
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(out != nullptr);
    assert(stride >= 1);

    if (!OrdersShortcut<kDefaultEvalScheme, kBoysFullAccuracyMultiplier, kDefaultFitRoute>(
            nmax, x, out, stride))
    {
        OrdersByScheme<true>(scheme, nmax, x, out, stride);
    }
}

// The entry the public surface's orders axis dispatches to (boys_impl.hpp).
//
// Three choices reach this one entry and each is a template argument:
//
//  - the scheme, which picks the polynomial table and the summation the
//    shipped route's fits are read with;
//  - the route, which picks which region-A fits the lane carries: the shipped
//    route's per-order pieces, or the rational route's per-piece pairs, which
//    are cut at the same pieces and read in the same mapped argument;
//  - the accuracy multiplier, which picks the degree a fit is read at. At the
//    reference rung every fit is read whole, which is the lane's shipped
//    reading; at a relaxed rung each fit is read at the degree the truncation
//    criterion certifies for that fit's piece and that multiplier.
//
// The stored fit is summed composed rather than gathered: the two fetches are
// the same lane value for value, and the composed one is the cheaper of the two
// in the counter that decides - retired slots - on the machine this lane was
// measured on, where the gather is a microcode assist worth several slots per
// step. The gathered entry stays beside it so the pair remains measurable.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a policy naming a route outside the FitRoute enumeration is not one this "
                  "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (kRoute == kDefaultFitRoute && kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        // The reference rung of the shipped route is the lane exactly as it
        // shipped: the same body, so the same tables, the same arithmetic and
        // the same fallback the suite pins bit for bit.
        BoysAllOrdersSimdComposed(OrdersSchemeOf(kScheme), nmax, x, out, 1);
        return;
    } else
    {
        if (x == 0.0)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0 / (2.0 * l + 1.0);
            }

            return;
        }

        if (!OrdersLaneApplies(x))
        {
            ScalarOrders<kScheme, kAccuracyMultiplier, kRoute>(nmax, x, out, 1);
            return;
        }

        // The degrees the shipped table's fits are read at. The criterion's
        // budget is zero at the reference rung, which leaves every fit at its
        // stored degree - the same reading the entry above takes - so the one
        // table covers both rungs.
        static constexpr auto kDegrees = RegionADegrees<kAccuracyMultiplier,
                                                        BoysRole::kDoubleSingle,
                                                        SchemeTailBasis<kScheme>()>();

        if constexpr (kRoute == FitRoute::kRationalMinimax)
        {
            static constexpr auto kPairs = RationalRegionADegrees<kAccuracyMultiplier>();

            RationalOrdersBody<kScheme>(
                nmax, x, out, 1, kPairs, RungDegree<decltype(kDegrees)>{kDegrees});
        } else
        {
            OrdersBody<OrdersSchemeOf(kScheme), true>(
                nmax, x, out, 1, RungDegree<decltype(kDegrees)>{kDegrees});
        }
    }
}

// The shapes a policy can name: two schemes, two routes, the reference
// multiplier and the six relaxed rungs. Each is instantiated here so that the
// dispatch in boys_impl.hpp is a branch over code the library already holds
// rather than a further instantiation per call site.
#define BOYS_ORDERS_PACKED_INSTANTIATIONS(kScheme)                                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev>(int, double,             \
                                                                          double*) noexcept;       \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev>(int, double,            \
                                                                           double*) noexcept;      \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev>(int, double,           \
                                                                            double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev>(int, double,          \
                                                                             double*) noexcept;    \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev>(int, double,          \
                                                                             double*) noexcept;    \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev>(int, double,         \
                                                                              double*) noexcept;   \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev>(int, double,         \
                                                                              double*) noexcept;   \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax>(int, double,       \
                                                                                double*) noexcept; \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax>(int, double,      \
                                                                                 double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax>(int, double,     \
                                                                                  double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax>(int, double,    \
                                                                                   double*)        \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax>(int, double,    \
                                                                                   double*)        \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax>(int, double,   \
                                                                                    double*)       \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax>(int, double,   \
                                                                                    double*)       \
        noexcept;

BOYS_ORDERS_PACKED_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS

} // namespace boys::detail

#else // BOYS_SIMD_X86

// Non-x86 targets: the entries exist and are defined, against the certified
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
// certified scalar single lane the packed bodies fall back to, at the policy
// the axis names, so the option exists, is defined and is the policy's own
// answer everywhere the library is.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingle<kAccuracyMultiplier, Policy>(l, x);
    }
}

#define BOYS_ORDERS_PACKED_INSTANTIATIONS(kScheme)                                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev>(int, double,             \
                                                                          double*) noexcept;       \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev>(int, double,            \
                                                                           double*) noexcept;      \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev>(int, double,           \
                                                                            double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev>(int, double,          \
                                                                             double*) noexcept;    \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev>(int, double,          \
                                                                             double*) noexcept;    \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev>(int, double,         \
                                                                              double*) noexcept;   \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev>(int, double,         \
                                                                              double*) noexcept;   \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax>(int, double,       \
                                                                                double*) noexcept; \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax>(int, double,      \
                                                                                 double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax>(int, double,     \
                                                                                  double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax>(int, double,    \
                                                                                   double*)        \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax>(int, double,    \
                                                                                   double*)        \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax>(int, double,   \
                                                                                    double*)       \
        noexcept;                                                                                  \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax>(int, double,   \
                                                                                    double*)       \
        noexcept;

BOYS_ORDERS_PACKED_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS

} // namespace boys::detail

#endif // BOYS_SIMD_X86
