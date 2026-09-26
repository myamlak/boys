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

// --- The premise the narrow partition's lane rests on ------------------------
//
// The narrow partition does not carry the premise above: its pieces are cut per
// order, so a fixed argument selects a different piece for each of the four
// orders a vector holds. The lane reads them anyway, and the count of pieces an
// order has is the table's own - 8 to 11 of them, against the shipped
// partition's single shape - which is the whole difference between the two
// fetching rules below.
//
// What it does need is one thing the premise above also implied: every piece is
// stored to the SAME degree. A group is read at the largest of its four lanes'
// certified degrees - one recurrence serves all four - so a lane whose own cut
// is lower is still read up to the group's degree, and that read has to stay
// inside the lane's own stored block rather than run off the end of the table.
// A uniform stored degree is what makes it so, and it is a property of the
// generated table rather than a given, so it is checked against the table.
bool NarrowPiecesShareDegree() noexcept {
    for (const OrderPiece& piece : kNarrowAPieces)
    {
        if (piece.deg != kNarrowADeg)
        {
            return false;
        }
    }

    return true;
}

bool NarrowPiecesUniform() noexcept {
    static const bool uniform = NarrowPiecesShareDegree();
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

// --- The narrow partition on the orders axis ---------------------------------
//
// The same axis over the other partition. Four orders of one argument still
// share a vector register; what changes is where each lane's coefficients come
// from. The shipped lane's fetch is a stride through one shared piece - the
// four orders' copies of it lie one order apart - and the narrow partition has
// no such stride, because each order's region-A fit is cut at its own edges:
// at one argument the four lanes are routinely in four different pieces, of
// four different degrees, with four different mapped arguments.
//
// So the fetch is per lane rather than by stride. Each lane's own piece is
// looked up, each lane's own interval maps the argument, and the four
// coefficients a step reads are gathered from the four lanes' own offsets -
// one gather whose index vector is the lanes' piece offsets, which is a
// gathered fetch of four different pieces rather than of four copies of one.
// The recurrences, the store and the scalar tail are the shipped body's,
// unchanged, so the values this lane returns are the stored narrow fits summed
// exactly as the shipped lane sums its own.
//
// What the lane costs is the lookup: the shipped body maps the argument once
// and derives the whole group's geometry from the single piece it lands in,
// where this one runs a scan and a mapping per lane. That is the price of a
// partition cut per order, and it is measured rather than assumed away.

// One group's four lanes at one argument: their pieces' coefficient offsets,
// the argument mapped into each lane's own interval, and the largest of their
// certified degrees.
template <class Degrees>
void NarrowGeometry(int l,
                    double x,
                    Degrees degrees,
                    __m128i& offsets,
                    int& deg,
                    __m256d& tv) noexcept {
    alignas(32) double mapped[4];
    int off[4];
    int degs[4];

    for (int j = 0; j < 4; ++j)
    {
        const OrderPiece& piece = FindNarrowAPiece(l + j, x);
        const std::size_t flat = static_cast<std::size_t>(&piece - kNarrowAPieces.data());
        off[j] = piece.offset;
        degs[j] = degrees.At(flat, piece.deg);
        // The shipped lane's own mapping, so a lane's value here is the value
        // its fit would return under the same summation. The narrow partition
        // is cut at its own edges, so the interval is read from the lane's own
        // piece rather than shared across the group.
        mapped[j] = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);
    }

    offsets = _mm_loadu_si128(reinterpret_cast<const __m128i*>(off));
    tv = _mm256_load_pd(mapped);

    deg = degs[0];

    for (int j = 1; j < 4; ++j)
    {
        deg = degs[j] > deg ? degs[j] : deg;
    }
}

// One vector of four orders from the narrow pieces, each lane fetched from its
// own piece's block: `offsets` holds the four lanes' coefficient offsets, so
// the gather at step k reads the k-th coefficient of each lane's own piece.
template <OrdersScheme kScheme>
__m256d NarrowGroup(const double* table, __m128i offsets, int deg, __m256d tv) noexcept {
    const auto coeff = [&](int k) { return _mm256_i32gather_pd(table + k, offsets, 8); };

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

// The narrow partition's region-A body: per-lane geometry, then vector groups
// of four orders and a scalar tail for the remainder, exactly as the shipped
// body is shaped.
template <OrdersScheme kScheme, class Degrees>
void NarrowOrdersBody(int nmax, double x, double* out, std::size_t stride, Degrees degrees) noexcept {
    const double* const table =
        (kScheme == OrdersScheme::kHorner) ? kNarrowAMonoCoeffs.data() : kNarrowACoeffs.data();

    int l = 0;

    if (nmax >= 3)
    {
        for (; l + 3 <= nmax; l += 4)
        {
            __m128i offsets{};
            int deg = 0;
            __m256d tv{};
            NarrowGeometry(l, x, degrees, offsets, deg, tv);
            StoreGroup(out, l, stride, NarrowGroup<kScheme>(table, offsets, deg, tv));
        }
    }

    for (; l <= nmax; ++l)
    {
        const OrderPiece& piece = FindNarrowAPiece(l, x);
        const std::size_t flat = static_cast<std::size_t>(&piece - kNarrowAPieces.data());
        const double t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);

        out[static_cast<std::size_t>(l) * stride] =
            ScalarFit<kScheme>(table + piece.offset, degrees.At(flat, piece.deg), t);
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

// The same question for the narrow partition's lane. Its premise is the one
// above's weaker form - the pieces need not share their edges, only their
// stored degree - and the interval is the same, because the narrow partition
// tiles the same region A and ends at the same kX0.
bool NarrowOrdersLaneApplies(double x) noexcept {
    return !(x >= kX0) && BoysAvx2Available() && NarrowPiecesUniform();
}

// The certified scalar single lane at the policy the axis names, one order at
// a time: what the entry is outside the packed interval, on a host without the
// vector tier, and against a table that does not carry the lane's premise.
//
// The multiplier is the entry's own, so a rung falls back to the rung of the
// per-order lane rather than to the full-accuracy one. The budget is the
// policy's engine choice and this path is the double engine at every budget,
// so the fallback names the float budget the double entries are built with.
// The partition is the policy's too, for the same reason: the fallback is the
// partition the caller named, at the rung the caller named, and never another
// one's values under this one's name.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity>
void ScalarOrders(int nmax, double x, double* out, std::size_t stride) noexcept {
    using Policy =
        EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments, kGranularity>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<kAccuracyMultiplier, Policy>(l, x);
    }
}

// --- The single-precision lane ----------------------------------------------
//
// Eight orders to a register, and a premise the double lane above does not
// need. Every order's float region-A cover is its own: order 0 is cut into two
// pieces where order 14 is cut into three, and a break one order carries is not
// a break the next one carries at the same argument. A fixed x therefore
// selects a different piece in every lane, and the offset from one lane's
// coefficients to the next is not a stride. So the group carries the geometry:
// one coefficient base, one mapped argument and one degree per lane, fetched
// per lane rather than stepped.
//
// What the eight lanes share is the degree the group is summed AT, because the
// split Clenshaw's even/odd structure belongs to the degree and not to a
// coefficient. The group runs at its lanes' largest degree and a lane whose own
// cut is below it reads zeros above that cut. Reading zeros above a cut is that
// lane's own polynomial and, down the recurrence, its own arithmetic: the extra
// top step has an exact zero for both of its terms, so it hands the lane
// exactly the state its own-degree summation would have started from. That is
// what keeps a lane's packed value the per-order value bit for bit rather than
// near it, and the lane's own test holds it to that.

// The steps a lane does not take: all-ones where k is above the lane's own
// degree. The degrees are small integers, so the comparison is exact in float
// lanes and the mask is a lane-sized one rather than a packed integer that
// would have to be widened first.
__m256 AboveDegreeF32(int k, __m256 deg) noexcept {
    return _mm256_cmp_ps(_mm256_set1_ps(static_cast<float>(k)), deg, _CMP_GT_OQ);
}

// One group of eight orders' geometry at one argument: where each lane's
// coefficients begin, what degree each lane is read at, and what argument each
// lane's fit is summed at.
//
// A lane's base and its stored degree are what bound the fetch's index; its
// degree at this reading is what the mask is taken against. When every lane is
// read at the group's own degree - which is every lane at the reference rung,
// where each fit is read whole - there is no cut to mask and the fetch is one
// index per lane.
struct F32Group {
    const float* table;
    std::int32_t base[8];
    std::int32_t stored[8];
    std::int32_t degree[8];
    float t[8];
    int degMax;
    bool uniform;
};

// The coefficient bases and mapped arguments a group of eight orders is read
// at. Eight scalar piece lookups per group is this lane's own price: the double
// lane's shared cover makes its index a stride it can step, and the float
// lane's per-order cover does not.
template <class Degrees>
F32Group BuildF32Group(const float* table, int l, float x, Degrees degrees) noexcept {
    F32Group group{table, {}, {}, {}, {}, 0, true};

    for (int j = 0; j < 8; ++j)
    {
        const f32::OrderPiece& piece = FindPieceF32(l + j, x);
        const auto flat = static_cast<std::size_t>(&piece - f32::kPieces.data());

        group.base[j] = piece.offset;
        group.stored[j] = piece.deg;
        group.degree[j] = degrees.At(flat, piece.deg);
        group.t[j] = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        group.degMax = group.degree[j] > group.degMax ? group.degree[j] : group.degMax;
    }

    for (int j = 0; j < 8; ++j)
    {
        group.uniform = group.uniform && group.degree[j] == group.degMax;
    }

    return group;
}

// The eight orders' k-th coefficients, each lane read at the lower of k and its
// own stored degree - so a lane already cut off cannot read past its piece -
// and zeroed wherever k is above the degree it is being read at.
//
// Two ways to fetch them, as on the double lane: the gather is one instruction
// and, on the machines measured here, a microcode assist; the composed form is
// eight loads and the shuffles that join them, more instructions and no assist.
// Which costs less is a property of the machine, so both are here.
template <bool kComposed>
__m256 F32Coefficients(const F32Group& group, int k) noexcept {
    if (group.uniform)
    {
        if constexpr (kComposed)
        {
            return _mm256_set_ps(group.table[group.base[7] + k],
                                 group.table[group.base[6] + k],
                                 group.table[group.base[5] + k],
                                 group.table[group.base[4] + k],
                                 group.table[group.base[3] + k],
                                 group.table[group.base[2] + k],
                                 group.table[group.base[1] + k],
                                 group.table[group.base[0] + k]);
        } else
        {
            return _mm256_i32gather_ps(
                group.table,
                _mm256_set_epi32(group.base[7] + k,
                                 group.base[6] + k,
                                 group.base[5] + k,
                                 group.base[4] + k,
                                 group.base[3] + k,
                                 group.base[2] + k,
                                 group.base[1] + k,
                                 group.base[0] + k),
                4);
        }
    }

    alignas(32) std::int32_t index[8];

    for (int j = 0; j < 8; ++j)
    {
        index[j] = group.base[j] + (k < group.stored[j] ? k : group.stored[j]);
    }

    const __m256 c =
        kComposed ? _mm256_set_ps(group.table[index[7]],
                                  group.table[index[6]],
                                  group.table[index[5]],
                                  group.table[index[4]],
                                  group.table[index[3]],
                                  group.table[index[2]],
                                  group.table[index[1]],
                                  group.table[index[0]])
                  : _mm256_i32gather_ps(group.table,
                                        _mm256_load_si256(reinterpret_cast<const __m256i*>(index)),
                                        4);
    const __m256 deg = _mm256_cvtepi32_ps(
        _mm256_load_si256(reinterpret_cast<const __m256i*>(group.degree)));

    return _mm256_andnot_ps(AboveDegreeF32(k, deg), c);
}

// The same fetch for the scalar tail, where the eight lanes are one order: the
// tail runs the vector body rather than the library's scalar one so that its
// values are this lane's values, at whatever degree it was handed.
struct BroadcastCoefficientsF32 {
    const float* table;
    int base;

    __m256 operator()(int k) const noexcept { return _mm256_set1_ps(table[base + k]); }
};

// Split Clenshaw, transcribed from boys_impl.hpp's ClenshawSplit onto per-lane
// coefficients: same steps, same order, same fused operations, so a lane's
// value is that order's single-precision value at the same degree, bit for bit.
template <class C> __m256 ClenshawSplitF32(C coeff, int deg, __m256 t) noexcept {
    if (deg == 0)
    {
        return coeff(0);
    }

    if (deg == 1)
    {
        return _mm256_fmadd_ps(t, coeff(1), coeff(0));
    }

    const __m256 v =
        _mm256_fmadd_ps(_mm256_set1_ps(2.0f), _mm256_mul_ps(t, t), _mm256_set1_ps(-1.0f));
    const __m256 twoV = _mm256_add_ps(v, v);

    if (deg == 2)
    {
        return _mm256_fmadd_ps(t, coeff(1), _mm256_fmadd_ps(v, coeff(2), coeff(0)));
    }

    assert(deg >= 4 && deg % 2 == 0);

    const int m = deg / 2;
    __m256 b1 = coeff(2 * m);
    __m256 b2 = _mm256_setzero_ps();

    for (int k = m - 1; k >= 1; --k)
    {
        const __m256 b0 = _mm256_fmadd_ps(twoV, b1, _mm256_sub_ps(coeff(2 * k), b2));
        b2 = b1;
        b1 = b0;
    }

    const __m256 even = _mm256_fmadd_ps(v, b1, _mm256_sub_ps(coeff(0), b2));

    __m256 o1 = coeff(2 * m - 1);
    __m256 o2 = _mm256_setzero_ps();

    for (int k = m - 2; k >= 1; --k)
    {
        const __m256 o0 = _mm256_fmadd_ps(twoV, o1, _mm256_sub_ps(coeff(2 * k + 1), o2));
        o2 = o1;
        o1 = o0;
    }

    const __m256 odd = _mm256_fmadd_ps(
        _mm256_sub_ps(twoV, _mm256_set1_ps(1.0f)), o1, _mm256_sub_ps(coeff(1), o2));
    return _mm256_fmadd_ps(t, odd, even);
}

// The direct sum: the Chebyshev series term by term, T_k by the forward
// recurrence. Two multiply-adds per coefficient against the split Clenshaw's
// one, and no dependence between the terms.
template <class C> __m256 ChebyshevDirectSumF32(C coeff, int deg, __m256 t) noexcept {
    if (deg == 0)
    {
        return coeff(0);
    }

    __m256 sum = _mm256_fmadd_ps(t, coeff(1), coeff(0));

    if (deg == 1)
    {
        return sum;
    }

    const __m256 twoT = _mm256_add_ps(t, t);
    __m256 prev = _mm256_set1_ps(1.0f); // T_0
    __m256 cur = t; // T_1

    for (int k = 2; k <= deg; ++k)
    {
        const __m256 next = _mm256_fmsub_ps(twoT, cur, prev);
        sum = _mm256_fmadd_ps(coeff(k), next, sum);
        prev = cur;
        cur = next;
    }

    return sum;
}

// Horner over the monomial form of the fit, transcribed from HornerMono onto
// per-lane coefficients.
template <class C> __m256 HornerGatheredF32(C coeff, int deg, __m256 t) noexcept {
    __m256 acc = coeff(deg);

    for (int k = deg - 1; k >= 0; --k)
    {
        acc = _mm256_fmadd_ps(acc, t, coeff(k));
    }

    return acc;
}

// One order's fit at one argument, in the group's own arithmetic: the body the
// scalar tail runs so that its values are the vector's.
template <OrdersScheme kScheme>
float F32ScalarFit(const float* table, int base, int deg, float t) noexcept {
    const BroadcastCoefficientsF32 coeff{table, base};
    const __m256 tv = _mm256_set1_ps(t);
    alignas(32) float lanes[8];

    if constexpr (kScheme == OrdersScheme::kHorner)
    {
        _mm256_store_ps(lanes, HornerGatheredF32(coeff, deg, tv));
    } else if constexpr (kScheme == OrdersScheme::kDirectSum)
    {
        _mm256_store_ps(lanes, ChebyshevDirectSumF32(coeff, deg, tv));
    } else
    {
        _mm256_store_ps(lanes, ClenshawSplitF32(coeff, deg, tv));
    }

    return lanes[0];
}

// One vector of eight orders' values from a polynomial table, at the group's
// largest degree, every lane masked to its own.
template <OrdersScheme kScheme, bool kComposed>
__m256 F32ShippedGroup(const F32Group& group) noexcept {
    const auto coeff = [&](int k) { return F32Coefficients<kComposed>(group, k); };
    const __m256 tv = _mm256_loadu_ps(group.t);

    if constexpr (kScheme == OrdersScheme::kHorner)
    {
        return HornerGatheredF32(coeff, group.degMax, tv);
    } else if constexpr (kScheme == OrdersScheme::kDirectSum)
    {
        return ChebyshevDirectSumF32(coeff, group.degMax, tv);
    } else
    {
        return ClenshawSplitF32(coeff, group.degMax, tv);
    }
}

// The reference reading: every stored fit at its own stored degree.
struct F32StoredDegree {
    static int At(std::size_t, int stored) noexcept { return stored; }
};

// A rung's reading: every stored fit at the degree the truncation criterion
// certifies for that fit's piece at the rung's multiplier. The table is flat
// over the float lane's piece table, one degree per piece, which is how the
// lookups above index it.
template <typename Table>
struct F32RungDegree {
    const Table& table;

    int At(std::size_t flat, int stored) const noexcept {
        static_cast<void>(stored);
        return table[flat];
    }
};

// The region-A body: vector groups of eight orders, eight coefficient bases to
// a group, and a scalar tail for the remainder.
template <OrdersScheme kScheme, bool kComposed, class Degrees>
void F32OrdersBody(int nmax, float x, float* out, Degrees degrees) noexcept {
    const float* const table =
        (kScheme == OrdersScheme::kHorner) ? f32::kMonoCoeffs.data() : f32::kCoeffs.data();

    int l = 0;

    for (; l + 8 <= nmax + 1; l += 8)
    {
        _mm256_storeu_ps(out + l, F32ShippedGroup<kScheme, kComposed>(BuildF32Group(table, l, x, degrees)));
    }

    for (; l <= nmax; ++l)
    {
        const f32::OrderPiece& piece = FindPieceF32(l, x);
        const auto flat = static_cast<std::size_t>(&piece - f32::kPieces.data());

        out[l] = F32ScalarFit<kScheme>(table,
                                       piece.offset,
                                       degrees.At(flat, piece.deg),
                                       2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f);
    }
}

// --- The rational route on the float orders axis ----------------------------
//
// The other family's region-A fits are a numerator and a denominator over the
// route's own pieces, read by Horner in the same mapped argument. The float
// route needs no per-order handover: every order over the whole of the route's
// region A is the route's own value, where the double route hands an order back
// to the shipped family below its own end of the interval. So the axis carries
// the route's pairs alone, each lane at its own stored numerator and
// denominator degrees.
//
// The group runs from its lanes' longest numerator and longest denominator
// down, with every lane masked to its own stored degree - and the denominator's
// coefficients sit above the piece's FULL numerator, so a lane's index is its
// own numerator degree plus j whatever the group's is.

// One group of eight orders' rational geometry at one argument.
struct F32RatGroup {
    std::int32_t base[8];
    std::int32_t numdeg[8];
    std::int32_t dendeg[8];
    float t[8];
    int numMax;
    int denMax;
};
F32RatGroup BuildF32RatGroup(int l, float x) noexcept {
    F32RatGroup group{{}, {}, {}, {}, 0, 0};

    for (int j = 0; j < 8; ++j)
    {
        const f32::RatPiece& piece = FindRatPieceF32(l + j, x);

        group.base[j] = piece.offset;
        group.numdeg[j] = piece.numdeg;
        group.dendeg[j] = piece.dendeg;
        group.t[j] = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        group.numMax = piece.numdeg > group.numMax ? piece.numdeg : group.numMax;
        group.denMax = piece.dendeg > group.denMax ? piece.dendeg : group.denMax;
    }

    return group;
}

// The eight orders' k-th coefficient of one part of the pair, read as a Horner
// array that begins at each lane's own `shift` into its piece: the index is
// clamped to the lane's own stored degree so a masked lane stays inside its
// piece, and the value is zeroed above the degree the lane is read at.
__m256 F32RatCoefficients(const float* coeffs,
                          const std::int32_t* base,
                          const std::int32_t* shift,
                          const std::int32_t* stored,
                          int k) noexcept {
    alignas(32) float c[8];

    for (int j = 0; j < 8; ++j)
    {
        const int m = k < stored[j] ? k : stored[j];
        c[j] = (k > stored[j]) ? 0.0f : coeffs[base[j] + shift[j] + m];
    }

    return _mm256_load_ps(c);
}

// Eight orders' values from the stored pairs, each lane at its own degrees.
//
// The mask is what makes one recurrence serve eight different pairs: a lane's
// coefficients are live down to its own degree and zero above it, and a lane
// whose every step so far has been masked off still holds exactly zero, so its
// sequence is the route's own scalar reading - same first coefficient, same
// order, same fused operations. A lane with no denominator at all keeps the
// zero accumulator and the closing multiply-add turns it into exactly one,
// which is the route's own early return.
__m256 F32RationalGroup(const F32RatGroup& group) noexcept {
    const float* const coeffs = f32::kRatACoeffs.data();
    const __m256 tv = _mm256_loadu_ps(group.t);
    const std::int32_t kNumShift[8] = {};

    __m256 num = _mm256_setzero_ps();

    for (int k = group.numMax; k >= 0; --k)
    {
        num = _mm256_fmadd_ps(
            num, tv, F32RatCoefficients(coeffs, group.base, kNumShift, group.numdeg, k));
    }

    // The denominator is stored above the piece's full numerator, so its
    // Horner array begins at the lane's own numerator degree plus one and runs
    // to its own denominator degree less one.
    alignas(32) std::int32_t denShift[8];
    alignas(32) std::int32_t denStored[8];

    for (int j = 0; j < 8; ++j)
    {
        denShift[j] = group.numdeg[j] + 1;
        denStored[j] = group.dendeg[j] - 1;
    }

    __m256 den = _mm256_setzero_ps();

    for (int k = group.denMax - 1; k >= 0; --k)
    {
        den = _mm256_fmadd_ps(
            den, tv, F32RatCoefficients(coeffs, group.base, denShift, denStored, k));
    }

    den = _mm256_fmadd_ps(den, tv, _mm256_set1_ps(1.0f));
    return _mm256_div_ps(num, den);
}

// The route's region-A body on this axis: eight orders to a vector and a
// scalar tail, the tail again in the group's own arithmetic.
void F32RationalBody(int nmax, float x, float* out) noexcept {
    int l = 0;

    for (; l + 8 <= nmax + 1; l += 8)
    {
        _mm256_storeu_ps(out + l, F32RationalGroup(BuildF32RatGroup(l, x)));
    }

    for (; l <= nmax; ++l)
    {
        const f32::RatPiece& piece = FindRatPieceF32(l, x);
        const float* const c = f32::kRatACoeffs.data() + piece.offset;
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        const BroadcastCoefficientsF32 numCoeff{c, 0};
        const BroadcastCoefficientsF32 denCoeff{c, piece.numdeg + 1};
        const __m256 tv = _mm256_set1_ps(t);
        alignas(32) float lanes[8];

        _mm256_store_ps(lanes, HornerGatheredF32(numCoeff, piece.numdeg, tv));
        float num = lanes[0];

        if (piece.dendeg == 0)
        {
            out[l] = num;
            continue;
        }

        _mm256_store_ps(lanes, HornerGatheredF32(denCoeff, piece.dendeg - 1, tv));
        out[l] = num / _mm256_cvtss_f32(
                           _mm256_fmadd_ps(_mm256_set1_ps(lanes[0]), tv, _mm256_set1_ps(1.0f)));
    }
}

// Whether the lane applies at this argument: inside the stored fits' interval
// and on a machine whose vector tier is present. The double lane's third
// condition - a table whose pieces share their shape - is the premise this lane
// was built not to need.
bool F32OrdersLaneApplies(float x) noexcept {
    return !(x >= static_cast<float>(kX0)) && BoysAvx2Available();
}

// The certified scalar single lane at the reference multiplier and the default
// policy: what the measurement entries answer outside the packed interval and
// on a host without the vector tier, exactly as the double lane's measurement
// entries fall back to theirs.
void F32ScalarOrdersDefault(int nmax, float x, float* out) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kBoysFullAccuracyMultiplier>(l, x);
    }
}

// The certified scalar single lane at the policy the axis names, one order at a
// time: what the public entry answers outside its own interval and on a host
// without the vector tier.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget>
void F32ScalarOrders(int nmax, float x, float* out) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, kBudget, PackAxis::kArguments>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kAccuracyMultiplier, Policy>(l, x);
    }
}

// The zero the library states in closed form, and the fallback outside the
// lane's domain, for the measurement entries: both are shared by the two fetch
// entries above.
bool F32OrdersShortcut(int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);

    if (x == 0.0f)
    {
        for (int l = 0; l <= nmax; ++l)
        {
            out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
        }

        return true;
    }

    if (!F32OrdersLaneApplies(x))
    {
        F32ScalarOrdersDefault(nmax, x, out);
        return true;
    }

    return false;
}

template <bool kComposed>
void F32OrdersByRoute(OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    if (route == FitRoute::kRationalMinimax)
    {
        F32RationalBody(nmax, x, out);
        return;
    }

    switch (scheme)
    {
    case OrdersScheme::kDirectSum:
        F32OrdersBody<OrdersScheme::kDirectSum, kComposed>(nmax, x, out, F32StoredDegree{});
        break;

    case OrdersScheme::kHorner:
        F32OrdersBody<OrdersScheme::kHorner, kComposed>(nmax, x, out, F32StoredDegree{});
        break;

    case OrdersScheme::kSplitClenshaw:
    default:
        F32OrdersBody<OrdersScheme::kSplitClenshaw, kComposed>(nmax, x, out, F32StoredDegree{});
        break;
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
        ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kDefaultFitGranularity>(
            nmax, x, out, stride);
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

void BoysAllOrdersF32Simd(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if (!F32OrdersShortcut(nmax, x, out))
    {
        F32OrdersByRoute<false>(scheme, route, nmax, x, out);
    }
}

void BoysAllOrdersF32SimdComposed(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if (!F32OrdersShortcut(nmax, x, out))
    {
        F32OrdersByRoute<true>(scheme, route, nmax, x, out);
    }
}

// The float engines' entry on the orders axis (boys_impl.hpp).
//
// Four choices reach this one entry and each is a template argument: the
// scheme, the route, the accuracy multiplier and the computation budget. The
// last two are what a rung costs here that it does not cost the double lane:
// the degree table a rung reads is certified against one stored table of one
// fit family on the one hand, and against a region budget on the other, so a
// relaxed rung reaches the float lane through the shipped route and scheme at
// the budget the policy named.
//
// The stored fit is summed gathered rather than composed. The two fetches are
// the same lane value for value - the lane's own test asserts the pair is
// bit-identical over every scheme and the whole region-A sweep - so this is a
// choice of instruction and nothing else, and the counter decides it. On this
// lane the gather is the cheaper of the two on both retired counters, which is
// not the double lane's result: eight floats fill one gather where four doubles
// filled a microcoded one. The composed entry stays beside it so the pair
// remains measurable.
constexpr bool kF32Composed = false;

template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a policy naming a route outside the FitRoute enumeration is not one this "
                  "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if constexpr (kRoute == kDefaultFitRoute && kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        // The reference rung of the shipped route is the lane exactly as it
        // stands: the same body the measurement entries above run, so the same
        // tables and the same values. The budget is inert here - at the
        // reference multiplier the region-A seed is the double lane's, which
        // takes no budget - so one entry serves both.
        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        // Past the interval, and on a host without the vector tier, the entry
        // is the certified scalar single lane at the scheme, the route and the
        // budget the caller named, one order at a time.
        if (!F32OrdersLaneApplies(x))
        {
            F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget>(nmax, x, out);
            return;
        }

        F32OrdersBody<OrdersSchemeOf(kScheme), kF32Composed>(nmax, x, out, F32StoredDegree{});
    } else
    {
        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        if (!F32OrdersLaneApplies(x))
        {
            F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget>(nmax, x, out);
            return;
        }

        // The degrees the shipped table's fits are read at. The lane evaluates
        // each order independently, so the amplification it pays is the
        // single-order one and the table is the single-order role's; the budget
        // picks which bar that role is certified against.
        constexpr BoysRole kRole = (kBudget == BoysBudget::kFloat) ? BoysRole::kF32Single
                                                                   : BoysRole::kF32Fp16Single;
        static constexpr auto kDegreesA = RegionADegrees<kAccuracyMultiplier, kRole>();

        if constexpr (kRoute == FitRoute::kRationalMinimax)
        {
            F32RationalBody(nmax, x, out);
        } else
        {
            F32OrdersBody<OrdersSchemeOf(kScheme), kF32Composed>(
                nmax, x, out, F32RungDegree<decltype(kDegreesA)>{kDegreesA});
        }
    }
}

// The shapes a float policy can name on this axis: two schemes and two
// computation budgets at the reference multiplier, with either route; and the
// six relaxed rungs of the shipped route and scheme alone, which is the
// combination the engine above admits at a rung. Each is instantiated here so
// that the dispatch in boys_impl.hpp is a branch over code the library already
// holds rather than a further instantiation per call site.
#define BOYS_ORDERS_F32_PACKED_REFERENCE(kScheme, kBudget)                                         \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget>(             \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget>(       \
        int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_RUNGS(kBudget)                                                      \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 64.0, FitRoute::kChebyshev, kBudget>(               \
        int, float, float*) noexcept;                                                              \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 256.0, FitRoute::kChebyshev, kBudget>(              \
        int, float, float*) noexcept;                                                              \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 1024.0, FitRoute::kChebyshev, kBudget>(             \
        int, float, float*) noexcept;                                                              \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 4096.0, FitRoute::kChebyshev, kBudget>(             \
        int, float, float*) noexcept;                                                              \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 16384.0, FitRoute::kChebyshev, kBudget>(            \
        int, float, float*) noexcept;                                                              \
    template void                                                                                  \
    BoysAllOrdersF32Packed<kDefaultEvalScheme, 65536.0, FitRoute::kChebyshev, kBudget>(            \
        int, float, float*) noexcept;

BOYS_ORDERS_F32_PACKED_REFERENCE(kDefaultEvalScheme, BoysBudget::kFloat)
BOYS_ORDERS_F32_PACKED_REFERENCE(kDefaultEvalScheme, BoysBudget::kFp16)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFloat)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFp16)

#undef BOYS_ORDERS_F32_PACKED_REFERENCE
#undef BOYS_ORDERS_F32_PACKED_RUNGS

// The entry the public surface's orders axis dispatches to (boys_impl.hpp).
//
// Four choices reach this one entry and each is a template argument:
//
//  - the scheme, which picks the polynomial table and the summation the
//    shipped route's fits are read with;
//  - the route, which picks which region-A fits the lane carries: the shipped
//    route's per-order pieces, or the rational route's per-piece pairs, which
//    are cut at the same pieces and read in the same mapped argument;
//  - the accuracy multiplier, which picks the degree a fit is read at. At the
//    reference rung every fit is read whole, which is the lane's shipped
//    reading; at a relaxed rung each fit is read at the degree the truncation
//    criterion certifies for that fit's piece and that multiplier;
//  - the partition, which picks the table the lane reads. The shipped one is
//    the per-order pieces the lane has always read, whose shared shape is what
//    lets one stride fetch four orders' coefficients; the narrow one is cut
//    per order, so the lane reads each order's own piece instead.
//
// The stored fit is summed composed rather than gathered: the two fetches are
// the same lane value for value, and the composed one is the cheaper of the two
// in the counter that decides - retired slots - on the machine this lane was
// measured on, where the gather is a microcode assist worth several slots per
// step. The gathered entry stays beside it so the pair remains measurable.
// The narrow partition's fetch is a gather by construction - the four lanes'
// pieces have no stride between them - so it has no composed twin.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a policy naming a route outside the FitRoute enumeration is not one this "
                  "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (kRoute == kDefaultFitRoute &&
                  kAccuracyMultiplier == kBoysFullAccuracyMultiplier &&
                  kGranularity == kDefaultFitGranularity)
    {
        // The reference rung of the shipped route, on the shipped partition, is
        // the lane exactly as it shipped: the same body, so the same tables, the
        // same arithmetic and the same fallback the suite pins bit for bit. The
        // partition is part of the condition and not only the rung, because
        // that body reads the shipped table: a policy naming the narrow one
        // reaches the narrow body below instead.
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

        if constexpr (kGranularity == kDefaultFitGranularity)
        {
            if (!OrdersLaneApplies(x))
            {
                ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity>(nmax, x, out, 1);
                return;
            }
        } else
        {
            if (!NarrowOrdersLaneApplies(x))
            {
                ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity>(nmax, x, out, 1);
                return;
            }
        }

        // The degrees this partition's fits are read at. The criterion's budget
        // is zero at the reference rung, which leaves every fit at its stored
        // degree - the same reading the shipped path takes - so the one table
        // covers every rung.
        static constexpr auto kDegrees =
            RegionADegreeTableOf<kAccuracyMultiplier,
                                 EvalPolicy<kRoute, kScheme, BoysBudget::kFloat,
                                            PackAxis::kOrders, kGranularity>,
                                 BoysRole::kDoubleSingle>();

        if constexpr (kGranularity != kDefaultFitGranularity)
        {
            // The narrow partition is a partition of the shipped route's own
            // region-A fits; the rational route stores its pairs over the
            // shipped pieces and has no narrow partition of them, which
            // RouteFit refuses where such a policy is named rather than here.
            static_assert(kRoute == FitRoute::kChebyshev,
                          "the rational route's region-A pairs cover the shipped partition's "
                          "per-order pieces, and the narrow partition is cut per order, so "
                          "there is no narrow partition of the route's pairs to read: the "
                          "route carries the shipped partition on this axis");
            NarrowOrdersBody<OrdersSchemeOf(kScheme)>(
                nmax, x, out, 1, RungDegree<decltype(kDegrees)>{kDegrees});
        } else if constexpr (kRoute == FitRoute::kRationalMinimax)
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

// The narrow partition, at both schemes and every rung, on the shipped route
// alone: the partition is a partition of that route's region-A fits.
#define BOYS_ORDERS_NARROW_INSTANTIATIONS(kScheme)                                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev, FitGranularity::kNarrow>( \
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev, FitGranularity::kNarrow>( \
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;

BOYS_ORDERS_PACKED_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner)
BOYS_ORDERS_NARROW_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_INSTANTIATIONS

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

// The float lane's measurement entries exist here for the reason the double
// lane's do: the vector tier is absent, so the lane is the fits it would have
// vectorised, one order at a time.
void BoysAllOrdersF32Simd(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    static_cast<void>(scheme);
    static_cast<void>(route);

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kBoysFullAccuracyMultiplier>(l, x);
    }
}

void BoysAllOrdersF32SimdComposed(
    OrdersScheme scheme, FitRoute route, int nmax, float x, float* out) noexcept {
    BoysAllOrdersF32Simd(scheme, route, nmax, x, out);
}

// The orders axis's float entry on a target without the vector tier: the same
// certified scalar single lane the packed bodies fall back to, at the policy
// the axis names, so the option exists, is defined and is the policy's own
// answer everywhere the library is.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, kBudget, PackAxis::kArguments>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kAccuracyMultiplier, Policy>(l, x);
    }
}

#define BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(kScheme, kBudget)                                    \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget>(             \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget>(       \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 64.0, FitRoute::kChebyshev, kBudget>(            \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 256.0, FitRoute::kChebyshev, kBudget>(           \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 1024.0, FitRoute::kChebyshev, kBudget>(          \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 4096.0, FitRoute::kChebyshev, kBudget>(          \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 16384.0, FitRoute::kChebyshev, kBudget>(         \
        int, float, float*) noexcept;                                                              \
    template void BoysAllOrdersF32Packed<kScheme, 65536.0, FitRoute::kChebyshev, kBudget>(         \
        int, float, float*) noexcept;

BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(kDefaultEvalScheme, BoysBudget::kFloat)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(kDefaultEvalScheme, BoysBudget::kFp16)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFloat)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFp16)

#undef BOYS_ORDERS_F32_PACKED_INSTANTIATIONS

// The orders axis's entry on a target without the vector tier: the same
// certified scalar single lane the packed bodies fall back to, at the policy
// the axis names, so the option exists, is defined and is the policy's own
// answer everywhere the library is.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    using Policy =
        EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments, kGranularity>;

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

// The narrow partition, at both schemes and every rung, on the shipped route
// alone: the partition is a partition of that route's region-A fits.
#define BOYS_ORDERS_NARROW_INSTANTIATIONS(kScheme)                                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev, FitGranularity::kNarrow>( \
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev, FitGranularity::kNarrow>( \
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;     \
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow>(int, double, double*) noexcept;

BOYS_ORDERS_PACKED_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner)
BOYS_ORDERS_NARROW_INSTANTIATIONS(kDefaultEvalScheme)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_INSTANTIATIONS

} // namespace boys::detail

#endif // BOYS_SIMD_X86
