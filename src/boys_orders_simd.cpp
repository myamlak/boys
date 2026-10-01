#include "boys_orders_simd.hpp"

#include "boys/boys.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/boys_impl.hpp"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>

// --- Architecture guard -----------------------------------------------------
//
// As boys_simd.cpp, which carries the argument: /arch:AVX2 (MSVC) or -mavx2
// -mfma (GCC/Clang).
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
// Four orders in one vector with no masking and no per-lane degree rests on every
// order's region-A fit being cut at the same boundaries to the same degree. That
// is a property of the generated table, not a given, so it is checked against it:
// a regenerated table with per-order degrees or splits sends the entry to the
// certified scalar fits instead of to a wrong answer.
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
// These pieces are cut per order - 8 to 11 an order, against the shipped
// partition's single shared shape - so a fixed argument lands in a different piece
// in each of the four lanes; that is the difference between the two fetching rules
// below.
//
// What they do share, and what the lane needs, is that every piece is stored to the
// SAME degree: a group is read at the largest of its four lanes' certified degrees,
// one recurrence serving all four, so a lane cut lower is still read up to the
// group's degree and that read has to stay inside the lane's own stored block.
// That is a property of the generated table, so it is checked against the table.
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

// The coefficients one step of a summation reads, for the four orders the vector
// carries, at the table's stride between one order and the next.
//
// The gather is one instruction and a microcode assist worth many retirement slots
// on the machines measured here; the composed form is four loads and the shuffles
// that join them, more instructions and no assist. Both are here because which
// costs less is a property of the machine, and the benchmark measures it.
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
// coefficients: same steps, order and fused operations, so a lane's value is the
// across-arguments lane's value for that order, bit for bit - the identity the
// across-orders test asserts.
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

// The direct sum: the Chebyshev series term by term, T_k by the forward recurrence
// T_k = 2t T_{k-1} - T_{k-2}. Two multiply-adds per coefficient against the split
// Clenshaw's one, and no dependence between the terms - the shape an across-orders
// vector fills.
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
// the scalar tail runs this so that its value is the vector's.
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
// The two readers below differ only in the numbers they hand a group of four
// orders; geometry, fetch, summation and store are the same under either - what
// makes a rung a table the lane reads rather than a second lane.

// The reference reading: every stored fit at its own stored degree.
struct StoredDegree {
    static int At(std::size_t, int stored) noexcept { return stored; }
};

// A rung's reading: every stored fit at the degree the truncation criterion
// certifies for that fit's piece at the rung's multiplier. The table is flat over
// the piece table, so one index names the piece.
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
// enumeration carries.
constexpr OrdersScheme OrdersSchemeOf(EvalScheme scheme) noexcept {
    return scheme == EvalScheme::kHorner ? OrdersScheme::kHorner : OrdersScheme::kSplitClenshaw;
}

// The degree a vector of four orders is read at: the largest of the four lanes'
// own. Reading above a lane's own cut costs nothing claimed - the
// dropped-coefficient tail is non-increasing in the degree.
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

// --- The uniform grid on the orders axis -------------------------------------
//
// The grid is interval-major - [interval][order][coefficient] - so one argument's
// whole ladder is contiguous and the step from one order's coefficients to the
// next is the same stride for every order, every interval and both schemes. The
// shipped body derives piece, mapped argument, degree and coefficient base from the
// piece table a criterion cut; this one reads the same four numbers off the index
// one multiply and a truncation produce, and nothing recurs: each order the vector
// holds is its own polynomial at the grid's stored degree.

// The grid's stored shape is the lane's premise - the interval's degree fixes how
// many coefficients an order holds there, the offsets where its block starts - so
// both are checked against the table: a regenerated grid of another shape is then a
// build that does not compile rather than a lane reading the wrong coefficients.
//
// The degree is the interval's own and not one stride for the table: a gathered
// group's four orders are one argument's, so they share the interval, its degree and
// the stride the group steps by - a variable degree costs this lane only the array
// read.
static_assert(std::size(kFlatOffsets) == static_cast<std::size_t>(kFlatIntervals) + 1 &&
                  kFlatOffsets[kFlatIntervals] == static_cast<int>(std::size(kFlatCoeffs)),
              "the uniform grid is addressed as [interval][order][coefficient] through the "
              "offsets, so they must run one per interval and end at the stored count");
static_assert(std::size(kFlatMonoCoeffs) == std::size(kFlatCoeffs),
              "the grid's two stored forms are parallel - same intervals, same orders, same "
              "degrees - so a scheme picks a table and a summation and changes no geometry");
static_assert(FlatDegreesCarried(),
              "every interval of the uniform grid must be fitted at an even degree between 4 "
              "and the read cap: the split Clenshaw recurrence this lane sums the far orders "
              "with is written for an even degree of at least four, and a degree above the cap "
              "is beyond the coefficients the interval stores");

template <OrdersScheme kScheme, bool kComposed>
void UniformOrdersBody(int nmax, double x, double* out, std::size_t stride) noexcept {
    const double* const table =
        (kScheme == OrdersScheme::kHorner) ? kFlatMonoCoeffs.data() : kFlatCoeffs.data();

    const FlatPoint at = FlatLocate(x);
    const int deg = kFlatDegs[at.iv];
    const int orderStride = deg + 1;
    const double* const base = table + at.block;

    const __m128i step = _mm_set_epi32(3 * orderStride, 2 * orderStride, orderStride, 0);
    const __m256d tv = _mm256_set1_pd(at.t);

    int l = 0;

    if (nmax >= 3)
    {
        for (; l + 3 <= nmax; l += 4)
        {
            // The four orders' copies of the cell lie one order apart, which is
            // the stride the fetch steps by; every order is read at this
            // interval's own stored degree, so the group has no degree to take
            // the largest of.
            StoreGroup(out,
                       l,
                       stride,
                       ShippedGroup<kScheme, kComposed>(base, l, orderStride, deg, step, tv));
        }
    }

    for (; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] =
            ScalarFit<kScheme>(base + static_cast<std::ptrdiff_t>(l) * orderStride, deg, at.t);
    }
}

// --- The narrow partition on the orders axis ---------------------------------
//
// Four orders of one argument still share a vector register; what changes is where
// each lane's coefficients come from. The shipped fetch is a stride through one
// shared piece, and these pieces are cut per order, so at one argument the four
// lanes are routinely in four different pieces, of four different degrees, with
// four different mapped arguments.
//
// So the fetch is per lane: each lane's own piece is looked up, each lane's own
// interval maps the argument, and the four coefficients a step reads are gathered
// from the four lanes' own offsets. The recurrences, the store and the scalar tail
// are the shipped body's, so the values returned are the stored narrow fits summed
// as the shipped lane sums its own. What the per-lane lookup costs is measured, not
// assumed away.

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
        // its fit returns under the same summation; the interval is the lane's
        // own piece's, the narrow partition being cut per order.
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

/// Whether the dispatches below name a summation for an enumerator of
/// OrdersScheme, in one place for both of them.
///
/// It exists beside those dispatches for the reason DevicePartitionName exists
/// beside the device rows: a switch whose last arm is merged with `default`
/// answers an enumerator it does not name with the body that arm holds, and the
/// body here - the certified split-Clenshaw sum - is a plausible one that
/// nothing distinguishes from the scheme the caller named. A fourth scheme
/// added to the enumeration and not named below would therefore be measured as
/// the certified scheme, silently, under its own name. The check at the end of
/// this block holds SchemeIsNamed, which is the map; what holds these dispatches
/// is gcc's and clang's -Wswitch, which fires on them because neither carries a
/// `default` arm, together with tools/check_enum_arms_covered.py, which is the
/// instrument that covers them where no such warning is emitted.
constexpr bool SchemeIsNamed(OrdersScheme scheme) noexcept {
    switch (scheme)
    {
    case OrdersScheme::kSplitClenshaw:
    case OrdersScheme::kDirectSum:
    case OrdersScheme::kHorner:
        return true;

    // The sentinel one past the last scheme, and not a scheme a caller can
    // name. It is named rather than left to a default arm: a default would
    // swallow the next scheme as quietly as it swallows this one.
    case OrdersScheme::kCount:
        break;
    }

    return false;
}

/// Whether SchemeIsNamed names every enumerator of OrdersScheme, read over the
/// enumeration's own count. A scheme added to the enumeration and not named
/// there falls through that switch's last return, this sees it, and the
/// assertion below stops the build - the omission is the error, rather than a
/// `default` arm quietly summing the certified scheme.
constexpr bool SchemesAllNamed() noexcept {
    for (int i = 0; i < static_cast<int>(OrdersScheme::kCount); ++i)
    {
        if (!SchemeIsNamed(static_cast<OrdersScheme>(i)))
        {
            return false;
        }
    }

    return true;
}

static_assert(SchemesAllNamed(),
              "an enumerator of OrdersScheme names no summation: name it in SchemeIsNamed "
              "(boys_orders_simd.cpp), in both dispatches of this file");

template <bool kComposed>
void OrdersByScheme(OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) {
    const StoredDegree degrees{};

    switch (scheme)
    {
    case OrdersScheme::kDirectSum:
        OrdersBody<OrdersScheme::kDirectSum, kComposed>(nmax, x, out, stride, degrees);
        return;

    case OrdersScheme::kHorner:
        OrdersBody<OrdersScheme::kHorner, kComposed>(nmax, x, out, stride, degrees);
        return;

    // Named, so that a scheme added to the enumeration is this switch's
    // business rather than the fall-through's: the certified sum is what runs
    // below, and it is named here as the arm that takes it.
    case OrdersScheme::kSplitClenshaw:
        break;

    // The sentinel one past the last scheme, and not a scheme a caller can name.
    case OrdersScheme::kCount:
        break;
    }

    // What a scheme this build does not serve takes, and what the sentinel
    // reaches: the certified split-Clenshaw sum, the arithmetic boys_impl.hpp's
    // own ClenshawSplit runs. Every enumerator of OrdersScheme is an arm above -
    // SchemesAllNamed is what keeps that true - so a scheme a caller can name
    // never arrives here.
    OrdersBody<OrdersScheme::kSplitClenshaw, kComposed>(nmax, x, out, stride, degrees);
}

// --- The rational route on the orders axis ----------------------------------
//
// The other route's region-A fits are a numerator and a denominator Horner over
// the shipped pieces, in the same mapped argument, so the axis carries them
// with the shipped geometry and a different reading: two recurrences and one
// division. A pair is stored to its own numerator and denominator degrees and
// the route certifies a cut of both, so the cut is the order's own and the four
// lanes of one vector need not share a degree; the fetch is masked per lane to
// match.
//
// The mapped argument is the route's own form, 2(x - a)/(b - a) - 1, rather
// than the fused form the shipped body maps with, so every value below is the
// route's scalar value for the same piece and the same cut, bit for bit - the
// identity the axis's contract test holds it to.

// The steps a lane does not take: all-ones where the lane's cut is below k.
// The cuts are small integers, so the comparison is exact in double lanes and
// the mask is a lane-sized one rather than a packed integer to be widened.
__m256d AboveCut(int k, __m256d cut) noexcept {
    return _mm256_cmp_pd(_mm256_set1_pd(static_cast<double>(k)), cut, _CMP_GT_OQ);
}

// The largest of four lanes' cut orders.
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
// sequence is the scalar reading's own. The gathers are clamped to each piece's
// stored degrees so that a masked-off lane cannot read outside the coefficient
// table; the clamp is invisible to a live lane, whose cut never exceeds its
// piece's stored degree.
//
// The four lanes' geometry is handed in, because the two partitions reach it
// differently: the shipped cover steps one flat index per lane at a table
// stride, the narrow cover looks each lane's own piece up.
template <class Pairs>
__m256d RationalGroupAt(const double* coeffs,
                        __m128i flat,
                        __m128i offset,
                        __m128i storedNum,
                        __m128i storedDen,
                        const Pairs& pairs,
                        __m256d tv) noexcept {
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

// One shipped-cover group: the four lanes' flat indices are derived from the
// group's first flat index and the table's stride.
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

    return RationalGroupAt(coeffs,
                           flat,
                           _mm_i32gather_epi32(kRatAOffset.data(), flat, 4),
                           _mm_i32gather_epi32(kRatANumDeg.data(), flat, 4),
                           _mm_i32gather_epi32(kRatADenDeg.data(), flat, 4),
                           pairs,
                           tv);
}

// The rational route's region-A body: the shipped geometry and the route's own
// reading of the pairs, where the route's per-order rule hands an order over.
//
// The orders below their end of region A are read with the shipped body's own
// group rule, so an order's value there does not depend on which route the
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
    // route hands an order over at that order's own end of region A, and the
    // ends are non-decreasing in the order, so the orders the route answers are
    // a prefix.
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

// The same body over the narrow partition's own cover: a lane's piece, its
// interval, its coefficient offset and its stored pair are its own piece's
// rather than a stride away from the group's first, so the four lanes' geometry
// is looked up lane by lane and handed to the same recurrence. The per-order
// rule is the body above's, unchanged, so what a call on this axis returns is
// the per-order narrow rational lane's value for the same order, four at a
// time.
template <EvalScheme kScheme, class Pairs, class Degrees>
void NarrowRationalOrdersBody(int nmax,
                              double x,
                              double* out,
                              std::size_t stride,
                              const Pairs& pairs,
                              Degrees degrees) noexcept {
    constexpr OrdersScheme kOrdersScheme = OrdersSchemeOf(kScheme);

    const double* const table =
        (kScheme == EvalScheme::kHorner) ? kNarrowAMonoCoeffs.data() : kNarrowACoeffs.data();
    const double* const ratCoeffs = kNarrowRatACoeffs.data();

    int served = 0;

    while (served <= nmax && x >= kTierThresholds[static_cast<std::size_t>(served)])
    {
        ++served;
    }

    const auto narrowPiece = [&](int l) -> const OrderPiece& { return FindNarrowAPiece(l, x); };

    const auto routeOrder = [&](int l) {
        const OrderPiece& piece = narrowPiece(l);
        const std::size_t flat = static_cast<std::size_t>(&piece - kNarrowAPieces.data());
        const double t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);
        return RationalPieceNarrowAtCut(flat, pairs.num[flat], pairs.den[flat], t);
    };

    const auto partitionOrder = [&](int l) {
        const OrderPiece& piece = narrowPiece(l);
        const std::size_t flat = static_cast<std::size_t>(&piece - kNarrowAPieces.data());
        const double t = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);
        return ScalarFit<kOrdersScheme>(table + piece.offset, degrees.At(flat, piece.deg), t);
    };

    int l = 0;

    if (nmax >= 3)
    {
        for (; l + 3 <= nmax; l += 4)
        {
            if (l >= served)
            {
                // None of the four takes the route's pair: the whole group is
                // the narrow partition's own fit at this rung's degrees.
                int deg = 0;
                __m128i offsets{};
                __m256d tv{};
                NarrowGeometry(l, x, degrees, offsets, deg, tv);
                StoreGroup(out, l, stride, NarrowGroup<kOrdersScheme>(table, offsets, deg, tv));
            } else if (l + 4 <= served)
            {
                alignas(32) int flatIdx[4];
                alignas(32) int off[4];
                alignas(32) int storedN[4];
                alignas(32) int storedD[4];
                alignas(32) double mapped[4];

                for (int j = 0; j < 4; ++j)
                {
                    const OrderPiece& piece = narrowPiece(l + j);
                    const std::size_t flat =
                        static_cast<std::size_t>(&piece - kNarrowAPieces.data());

                    flatIdx[j] = static_cast<int>(flat);
                    off[j] = kNarrowRatAOffset[flat];
                    storedN[j] = kNarrowRatANumDeg[flat];
                    storedD[j] = kNarrowRatADenDeg[flat];
                    mapped[j] = std::fma(x - piece.a, 2.0 / (piece.b - piece.a), -1.0);
                }

                StoreGroup(out,
                           l,
                           stride,
                           RationalGroupAt(
                               ratCoeffs,
                               _mm_loadu_si128(reinterpret_cast<const __m128i*>(flatIdx)),
                               _mm_loadu_si128(reinterpret_cast<const __m128i*>(off)),
                               _mm_loadu_si128(reinterpret_cast<const __m128i*>(storedN)),
                               _mm_loadu_si128(reinterpret_cast<const __m128i*>(storedD)),
                               pairs,
                               _mm256_load_pd(mapped)));
            } else
            {
                // The handover falls inside this group: one order at a time, in
                // the same two readings the whole-group rules above use.
                for (int j = 0; j < 4; ++j)
                {
                    out[static_cast<std::size_t>(l + j) * stride] =
                        (l + j < served) ? routeOrder(l + j) : partitionOrder(l + j);
                }
            }
        }
    }

    for (; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] =
            (l < served) ? routeOrder(l) : partitionOrder(l);
    }
}

// Whether the lane applies at this argument: inside the stored fits'
// interval, on a machine whose tier is present, against a table that carries
// the lane's premise.
bool OrdersLaneApplies(double x) noexcept {
    return !(x >= kX0) && BoysAvx2Available() && UniformPieces();
}

// The same question for the narrow partition's lane: the premise above's weaker
// form - the pieces need not share their edges, only their stored degree - and
// the same interval, the narrow partition tiling the same region A to kX0.
bool NarrowOrdersLaneApplies(double x) noexcept {
    return !(x >= kX0) && BoysAvx2Available() && NarrowPiecesUniform();
}

// The certified scalar single lane at the policy the axis names, one order at a time:
// what the entry is outside the packed interval, on a host without the vector tier, and
// against a table that does not carry the lane's premise.
//
// The multiplier, the route, the partition and the division form are the entry's own,
// so a rung falls back to that rung of the per-order lane rather than to the
// full-accuracy one, and never to another partition's values or another form's
// arithmetic under this one's name. The budget is the policy's engine choice and this
// path is the double engine at every budget, so the fallback names the float budget the
// double entries are built with.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity,
          DivisionForm kForm>
void ScalarOrders(int nmax, double x, double* out, std::size_t stride) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments,
                              kGranularity, kForm>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[static_cast<std::size_t>(l) * stride] = BoysSingle<kAccuracyMultiplier, Policy>(l, x);
    }
}

// --- The single-precision lane ----------------------------------------------
//
// Eight orders to a register, on a premise the double lane above does not need:
// every order's float region-A cover is its own - order 0 is cut into two pieces
// where order 14 is cut into three - so a fixed x selects a different piece in every
// lane and the offset from one lane's coefficients to the next is not a stride.
//
// What the eight lanes share is the degree the group is summed AT, because the split
// Clenshaw's even/odd structure belongs to the degree and not to a coefficient. A
// lane cut below the group's degree reads zeros above that cut: the extra top step
// has an exact zero for both of its terms and hands the lane exactly the state its
// own-degree summation would have started from. That keeps a lane's packed value the
// per-order value bit for bit, and the lane's own test holds it to that.

// The steps a lane does not take: all-ones where k is above the lane's own
// degree.
__m256 AboveDegreeF32(int k, __m256 deg) noexcept {
    return _mm256_cmp_ps(_mm256_set1_ps(static_cast<float>(k)), deg, _CMP_GT_OQ);
}

// One group of eight orders' geometry at one argument: where each lane's coefficients
// begin, the degree it is read at, and its mapped argument.
//
// A lane's base and its stored degree bound the fetch's index; its degree at this
// reading is what the mask is taken against, and at the reference rung, where every
// lane is read whole, there is no cut to mask.
struct F32Group {
    const float* table;
    std::int32_t base[8];
    std::int32_t stored[8];
    // A group is a stack local, so this array lands wherever the frame puts it
    // and not necessarily on a 32-byte boundary: the load that reads it is the
    // unaligned one, as the argument member's is.
    std::int32_t degree[8];
    float t[8];
    int degMax;
    bool uniform;
};

// The coefficient bases and mapped arguments a group of eight orders is read at.
// Eight scalar piece lookups per group is the price: the double lane's shared cover
// makes its index a stride it can step, this cover does not.
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

// --- The narrow partition on this lane ---------------------------------------
//
// The float lane's narrow pieces are cut per order too, so each of the eight lanes
// looks its own piece up and the fetch has no stride to share. It needs the premise
// the double lane's narrow body states: every piece stored to the same degree.
bool NarrowPiecesShareDegreeF32() noexcept {
    for (const f32::OrderPiece& piece : f32::kNarrowAPiecesF32)
    {
        if (piece.deg != f32::kNarrowADegF32)
        {
            return false;
        }
    }

    return true;
}

bool NarrowPiecesUniformF32() noexcept {
    static const bool uniform = NarrowPiecesShareDegreeF32();
    return uniform;
}

// The same for the narrow pieces: the shipped group's own shape and its own
// eight scalar lookups, over the other piece table.
template <class Degrees>
F32Group BuildF32NarrowGroup(const float* table, int l, float x, Degrees degrees) noexcept {
    F32Group group{table, {}, {}, {}, {}, 0, true};

    for (int j = 0; j < 8; ++j)
    {
        const f32::OrderPiece& piece = FindNarrowPieceF32(l + j, x);
        const auto flat = static_cast<std::size_t>(&piece - f32::kNarrowAPiecesF32.data());

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

// The eight orders' k-th coefficients, each lane read at the lower of k and its own
// stored degree - so a lane already cut off cannot read past its piece - and zeroed
// wherever k is above the degree it is being read at.
//
// Two ways to fetch them, for the reason the double lane's give.
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
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(group.degree)));

    return _mm256_andnot_ps(AboveDegreeF32(k, deg), c);
}

// The same fetch for the scalar tail, where the eight lanes are one order: the
// tail runs the vector body rather than the library's scalar one, so its values
// are this lane's at whatever degree it was handed.
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

// The direct sum; see ChebyshevDirectSum for what it is and why the shape suits
// this axis.
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
// certifies for that fit's piece at the rung's multiplier, flat over the float
// lane's piece table.
template <typename Table>
struct F32RungDegree {
    const Table& table;

    int At(std::size_t flat, int stored) const noexcept {
        static_cast<void>(stored);
        return table[flat];
    }
};

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

// The shipped body's shape over the narrow pieces, whose fit is the narrow table's
// rather than the shipped one's. The degrees are this partition's own table for the
// same reason: a cut degree is certified against the coefficients it is cut from.
template <OrdersScheme kScheme, bool kComposed, class Degrees>
void F32NarrowOrdersBody(int nmax, float x, float* out, Degrees degrees) noexcept {
    const float* const table = (kScheme == OrdersScheme::kHorner)
                                   ? f32::kNarrowAMonoCoeffsF32.data()
                                   : f32::kNarrowACoeffsF32.data();

    int l = 0;

    for (; l + 8 <= nmax + 1; l += 8)
    {
        _mm256_storeu_ps(
            out + l, F32ShippedGroup<kScheme, kComposed>(BuildF32NarrowGroup(table, l, x, degrees)));
    }

    for (; l <= nmax; ++l)
    {
        const f32::OrderPiece& piece = FindNarrowPieceF32(l, x);
        const auto flat = static_cast<std::size_t>(&piece - f32::kNarrowAPiecesF32.data());

        out[l] = F32ScalarFit<kScheme>(table,
                                       piece.offset,
                                       degrees.At(flat, piece.deg),
                                       2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f);
    }
}

// --- The rational route on the float orders axis ----------------------------
//
// The route's float region-A fits are a numerator and a denominator over the route's
// own pieces, read by Horner in the route's mapped argument. Over the whole of this
// lane's region A every order is the route's own value - no per-order handover, where
// the double route hands an order back to the shipped family below its own end of the
// interval - so the axis carries the route's pairs alone, each lane at its own stored
// numerator and denominator degrees and the group running from its lanes' longest
// numerator and longest denominator down with every lane masked to its own stored
// degree. The denominator's coefficients sit above the piece's FULL numerator, so a
// lane's index is its own numerator degree plus j whatever the group's is.

// One group of eight orders' rational geometry at one argument.
struct F32RatGroup {
    std::int32_t base[8];
    std::int32_t storednum[8];
    std::int32_t numdeg[8];
    std::int32_t dendeg[8];
    float t[8];
    int numMax;
    int denMax;
};

// The pair a lane reads: the stored degrees, which is the reference rung's
// reading - every stored fit read whole.
struct F32StoredPairs {
    static int Num(std::size_t, int stored) noexcept { return stored; }
    static int Den(std::size_t, int stored) noexcept { return stored; }
};

// A rung's reading: the pair the pair criterion certifies for that piece at the
// rung's multiplier, flat over the rational cover's piece table.
template <typename Pairs>
struct F32RungPairs {
    const Pairs& pairs;

    int Num(std::size_t flat, int /*stored*/) const noexcept { return pairs.num[flat]; }
    int Den(std::size_t flat, int /*stored*/) const noexcept { return pairs.den[flat]; }
};

template <class CutPairs>
F32RatGroup BuildF32RatGroup(int l, float x, CutPairs cuts) noexcept {
    F32RatGroup group{{}, {}, {}, {}, {}, 0, 0};

    for (int j = 0; j < 8; ++j)
    {
        const f32::RatPiece& piece = FindRatPieceF32(l + j, x);
        const std::size_t flat = static_cast<std::size_t>(&piece - f32::kRatAPieces.data());

        group.base[j] = piece.offset;
        group.storednum[j] = piece.numdeg;

        // The cut keeps the low-order terms of both parts. The numerator's array
        // begins at zero, so a lane's top coefficient is its cut's own; the
        // denominator's begins at the piece's FULL numerator, so its top coefficient is
        // the smaller of the cut and the stored denominator degree, which is what
        // `dendeg` holds here.
        const int numDeg = cuts.Num(flat, piece.numdeg);
        const int denDeg = cuts.Den(flat, piece.dendeg);

        group.numdeg[j] = numDeg;
        group.dendeg[j] = denDeg < piece.dendeg ? denDeg : piece.dendeg;
        group.t[j] = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        group.numMax = numDeg > group.numMax ? numDeg : group.numMax;
        group.denMax = group.dendeg[j] > group.denMax ? group.dendeg[j] : group.denMax;
    }

    return group;
}

// The same group over the narrow partition's own cover of region A: the lookup and
// the offsets are the narrow table's, the degrees, the cuts and the mapped arguments
// the shipped group's, because a pair is read the same way over either cover.
//
// The cut is this partition's own - the narrow pairs' coefficients are what the
// rung's criterion is run over, and the table the scalar lane's rung body reads is the
// same one (`NarrowRationalRegionAF32Degrees`) - so a rung here is the rung the
// per-order lane answers at rather than the stored fits under a rung's name. Reading
// the pieces whole instead parted the packed lane from the certified scalar lane on
// 901 of 2304 values at m = 64 and 2296 of 2304 at m = 65536.
template <class CutPairs>
F32RatGroup BuildF32NarrowRatGroup(int l, float x, CutPairs cuts) noexcept {
    F32RatGroup group{{}, {}, {}, {}, {}, 0, 0};

    for (int j = 0; j < 8; ++j)
    {
        const f32::RatPiece& piece = FindNarrowRatPieceF32(l + j, x);
        const std::size_t flat =
            static_cast<std::size_t>(&piece - f32::kNarrowRatAPiecesF32.data());

        group.base[j] = piece.offset;
        group.storednum[j] = piece.numdeg;

        // The cut keeps the low-order terms of both parts, as over the shipped cover:
        // the numerator's array begins at zero, so a lane's top coefficient is its
        // cut's own, while the denominator's begins at the piece's FULL numerator, so
        // its top coefficient is the smaller of the cut and the stored denominator
        // degree.
        const int numDeg = cuts.Num(flat, piece.numdeg);
        const int denDeg = cuts.Den(flat, piece.dendeg);

        group.numdeg[j] = numDeg;
        group.dendeg[j] = denDeg < piece.dendeg ? denDeg : piece.dendeg;
        group.t[j] = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;
        group.numMax = numDeg > group.numMax ? numDeg : group.numMax;
        group.denMax = group.dendeg[j] > group.denMax ? group.dendeg[j] : group.denMax;
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
// coefficients are live down to its own degree and zero above it, and a lane whose
// every step so far has been masked off still holds exactly zero, so its sequence is
// the route's own scalar reading. A lane with no denominator at all keeps the zero
// accumulator and the closing multiply-add turns it into exactly one, the route's own
// early return.
__m256 F32RationalGroup(const F32RatGroup& group, const float* coeffs) noexcept {
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
        denShift[j] = group.storednum[j] + 1;
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

// The route's region-A body on this axis: eight orders to a vector and a scalar
// tail in the group's own arithmetic - the route's own cut reading, so the
// tail's value is the vector's for the lane it stands for.
template <class CutPairs>
void F32RationalBody(int nmax, float x, float* out, CutPairs cuts) noexcept {
    int l = 0;

    for (; l + 8 <= nmax + 1; l += 8)
    {
        _mm256_storeu_ps(
            out + l,
            F32RationalGroup(BuildF32RatGroup(l, x, cuts), f32::kRatACoeffs.data()));
    }

    for (; l <= nmax; ++l)
    {
        const f32::RatPiece& piece = FindRatPieceF32(l, x);
        const std::size_t index = static_cast<std::size_t>(&piece - f32::kRatAPieces.data());
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        // The partition is the one this body's table is: the tail reads the
        // piece the shipped lookup above returned, out of the array the group
        // loop hands the vector reader.
        out[l] = RationalPieceF32AtCut<FitGranularity::kShipped>(
            index, cuts.Num(index, piece.numdeg), cuts.Den(index, piece.dendeg), t);
    }
}

// The shipped body above, reading the family's own cover: what changes is which piece
// table the lanes look their piece up in, which coefficient array the offsets index and
// which pairs table the rung's cut is read from. The tail is the shipped body's too, at
// this partition's own table, so its value is the group reader's for the same lane.
template <class CutPairs>
void F32NarrowRationalBody(int nmax, float x, float* out, CutPairs cuts) noexcept {
    int l = 0;

    for (; l + 8 <= nmax + 1; l += 8)
    {
        _mm256_storeu_ps(out + l,
                         F32RationalGroup(BuildF32NarrowRatGroup(l, x, cuts),
                                          f32::kNarrowRatACoeffsF32.data()));
    }

    for (; l <= nmax; ++l)
    {
        const f32::RatPiece& piece = FindNarrowRatPieceF32(l, x);
        const std::size_t index =
            static_cast<std::size_t>(&piece - f32::kNarrowRatAPiecesF32.data());
        const float t = 2.0f * (x - piece.a) / (piece.b - piece.a) - 1.0f;

        out[l] = RationalPieceF32AtCut<FitGranularity::kNarrow>(
            index, cuts.Num(index, piece.numdeg), cuts.Den(index, piece.dendeg), t);
    }
}

// Whether the lane applies at this argument: inside the stored fits' interval
// and on a machine whose vector tier is present. The double lane's third
// condition - a table whose pieces share their shape - is the premise this lane
// does not need.
bool F32OrdersLaneApplies(float x) noexcept {
    return !(x >= static_cast<float>(kX0)) && BoysAvx2Available();
}

// The certified scalar single lane at the reference multiplier and the default
// policy: what the measurement entries answer outside the packed interval and
// on a host without the vector tier.
void F32ScalarOrdersDefault(int nmax, float x, float* out) noexcept {
    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kBoysFullAccuracyMultiplier>(l, x);
    }
}

// The certified scalar single lane at the policy the axis names, one order at a time:
// what the public entry answers outside its own interval and on a host without the
// vector tier.
//
// The division form is the entry's own for the reason the double lane's fallback above
// states: the across-orders body divides nowhere, so the form reaches this lane and the
// recurrence steps it runs.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget,
          FitGranularity kGranularity = kDefaultFitGranularity,
          DivisionForm kForm = kDefaultDivisionForm>
void F32ScalarOrders(int nmax, float x, float* out) noexcept {
    using Policy =
        EvalPolicy<kRoute, kScheme, kBudget, PackAxis::kArguments, kGranularity, kForm>;

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
        F32RationalBody(nmax, x, out, F32StoredPairs{});
        return;
    }

    switch (scheme)
    {
    case OrdersScheme::kDirectSum:
        F32OrdersBody<OrdersScheme::kDirectSum, kComposed>(nmax, x, out, F32StoredDegree{});
        return;

    case OrdersScheme::kHorner:
        F32OrdersBody<OrdersScheme::kHorner, kComposed>(nmax, x, out, F32StoredDegree{});
        return;

    // Named, so that a scheme added to the enumeration is this switch's
    // business rather than the fall-through's: the certified sum is what runs
    // below, and it is named here as the arm that takes it.
    case OrdersScheme::kSplitClenshaw:
        break;

    // The sentinel one past the last scheme, and not a scheme a caller can name.
    case OrdersScheme::kCount:
        break;
    }

    // What a scheme this build does not serve takes, and what the sentinel
    // reaches: the certified split-Clenshaw sum of this lane's own width, the
    // same fallback the double-lane dispatch above takes and for the same
    // reason. Every enumerator of OrdersScheme is an arm above - SchemesAllNamed
    // is what keeps that true - so a scheme a caller can name never arrives
    // here.
    F32OrdersBody<OrdersScheme::kSplitClenshaw, kComposed>(nmax, x, out, F32StoredDegree{});
}

} // namespace

// The zero the library states in closed form, and the fallback outside the lane's
// domain - past the stored fits' interval, off the vector tier, or against a table that
// does not carry the lane's premise: the entry is then the certified scalar lanes, one
// order at a time. Both are shared by the two fetch entries. The division form is a
// parameter because the lane above the fallback divides nowhere and the scalar lane it
// falls to divides at every step, so a caller naming a form reaches it through here.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          DivisionForm kForm>
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
        ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, FitGranularity::kShipped, kForm>(
            nmax, x, out, stride);
        return true;
    }

    return false;
}

// The shortcut above at a scheme the caller selected at run time: the scheme this
// entry's signature carries is a run-time choice while the shortcut is a compile-time
// dispatch, so the two are joined by the same two-way branch the entries below take.
template <DivisionForm kForm>
bool BoysAllOrdersSimdShortcut(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    return scheme == OrdersScheme::kHorner
               ? OrdersShortcut<EvalScheme::kHorner, kBoysFullAccuracyMultiplier,
                                FitRoute::kChebyshev, kForm>(nmax, x, out, stride)
               : OrdersShortcut<EvalScheme::kSplitClenshaw, kBoysFullAccuracyMultiplier,
                                FitRoute::kChebyshev, kForm>(nmax, x, out, stride);
}

// The lane at a division form, with the two coefficient fetches as its template
// argument. The region-A body divides nowhere and is the same at every form; the
// fallback past the interval divides at each of its steps, so the form reaches the
// lane's caller through here rather than being fixed at the default.
template <bool kComposed, DivisionForm kForm>
void BoysAllOrdersSimdAtForm(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    assert(out != nullptr);
    assert(stride >= 1);

    // The shortcut is the shipped partition's and the scheme is the caller's: answering
    // the reference scheme's fallback to a caller naming the other would mix two
    // summations in one answer.
    if (BoysAllOrdersSimdShortcut<kForm>(scheme, nmax, x, out, stride))
    {
        return;
    }

    OrdersByScheme<kComposed>(scheme, nmax, x, out, stride);
}

void BoysAllOrdersSimd(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimdAtForm<false, kDefaultDivisionForm>(scheme, nmax, x, out, stride);
}

void BoysAllOrdersSimdComposed(
    OrdersScheme scheme, int nmax, double x, double* out, std::size_t stride) noexcept {
    BoysAllOrdersSimdAtForm<true, kDefaultDivisionForm>(scheme, nmax, x, out, stride);
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

// The float engines' entry on the orders axis (boys_impl.hpp). Five choices reach it,
// each a template argument: scheme, route, accuracy multiplier, budget, division form.
//
// The division form selects nothing this body evaluates - the across-orders lane reads
// stored fits and its region-A sums divide nowhere - so it is threaded to the scalar
// single lane the fallbacks hand their orders to, which divides at every step: a caller
// naming kPlainReciprocal is served plain steps and not the default form's.
//
// kF32Composed picks the fetch. The two are the same lane value for value - the lane's
// test asserts the pair is bit-identical over every scheme and the whole region-A sweep
// - so the choice is of instruction and nothing else: the gather wins here, where it
// loses on the double lane (eight floats fill one gather, four doubles a microcoded one).
// The composed entry stays beside it so the pair remains measurable.
constexpr bool kF32Composed = false;

template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget,
          FitGranularity kGranularity, DivisionForm kForm>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a policy naming a route outside the FitRoute enumeration is not one this "
                  "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0f);
    assert(out != nullptr);

    if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The fixed grid: its cells are fitted per order with no recurrence to enter and
        // no stride between one order's cell and the next for a gather to step, so the
        // ladder this lane carries is the scalar body the arguments axis reads and the
        // two axes are filled with the same numbers.
        //
        // The rung is not read - the cells are stored at the degrees the derivation
        // fitted them at, so a relaxed multiplier reads the same cells uncut and inside
        // the bound it named - but the route is, and it picks the fit rather than the
        // lane: the Chebyshev member's one degree per interval is what this lane packs,
        // and the rational member's pairs have no stride a gather could step either.
        //
        // The two are branches of one `if constexpr` rather than an early return above
        // the Chebyshev path: a taken early return leaves the code after it unreachable
        // in that instantiation, and this build refuses that warning.
        if constexpr (kRoute == FitRoute::kRationalMinimax)
        {
            F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget, kGranularity, kForm>(
                nmax, x, out);

            return;
        }
        else
        {
            if (x == 0.0f)
            {
                for (int l = 0; l <= nmax; ++l)
                {
                    out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
                }

                return;
            }

            if (x >= detail::f32::kFlatHiF32)
            {
                // Past the grid. This lane's join is inside region B rather than above
                // it, so what answers here is the entry's own region path at the
                // partition the caller named - the region-B seed and, past kX1, the
                // asymptotic - not a second fit under the uniform name.
                F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget, kGranularity,
                                kForm>(nmax, x, out);

                return;
            }

            UniformAllOrdersF32<boys::EvalPolicy<kRoute, kScheme, kBudget, PackAxis::kOrders,
                                                 kGranularity, kForm>>(nmax, x, out);

            return;
        }
    } else if constexpr (kRoute == FitRoute::kChebyshev &&
                         kAccuracyMultiplier == kBoysFullAccuracyMultiplier &&
                         kGranularity == FitGranularity::kShipped)
    {
        // The reference rung of the shipped route on the shipped partition is the lane
        // exactly as it stands: the same body the measurement entries above run, so the
        // same tables and values, and the budget is inert because the region-A seed is
        // the double lane's. The partition is part of the condition because this body
        // reads the shipped table.
        if (x == 0.0f)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0f / (2.0f * static_cast<float>(l) + 1.0f);
            }

            return;
        }

        // Past the interval, and on a host without the vector tier, the entry is the
        // certified scalar single lane at the scheme, route, budget and partition the
        // caller named, one order at a time. The partition is spelled and not left to
        // the default: this branch is the shipped partition's.
        if (!F32OrdersLaneApplies(x))
        {
            F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget, kGranularity, kForm>(nmax, x,
                                                                                         out);
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

        // The lane's premise is the partition's: the shipped cover lets a fixed argument
        // step one order's coefficients to the next at a stride, the narrow one does not.
        // Past the interval, and where the condition fails, the entry is the certified
        // scalar single lane at the policy the caller named - same partition, same rung.
        const bool laneApplies = (kGranularity == FitGranularity::kShipped)
                                     ? F32OrdersLaneApplies(x)
                                     : (!(x >= static_cast<float>(kX0)) && BoysAvx2Available() &&
                                        NarrowPiecesUniformF32());

        if (!laneApplies)
        {
            F32ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kBudget, kGranularity, kForm>(nmax, x,
                                                                                        out);
            return;
        }

        // The degrees the fits are read at: each family's own table, cut from the
        // coefficients that family stores and read in the basis the scheme sums, so the
        // scheme is a choice of table here and not only of summation. The lane evaluates
        // each order independently, so the role is the single-order one and the budget
        // picks which bar it is certified against.
        constexpr BoysRole kRole = (kBudget == BoysBudget::kFloat) ? BoysRole::kF32Single
                                                                   : BoysRole::kF32Fp16Single;

        if constexpr (kGranularity == FitGranularity::kShipped)
        {
            static constexpr auto kDegreesA =
                RegionADegrees<kAccuracyMultiplier, kRole, SchemeTailBasis<kScheme>()>();

            if constexpr (kRoute == FitRoute::kRationalMinimax)
            {
                static constexpr auto kPairsA =
                    RationalRegionAF32Degrees<kAccuracyMultiplier, kRole>();

                F32RationalBody(nmax, x, out, F32RungPairs<decltype(kPairsA)>{kPairsA});
            } else
            {
                F32OrdersBody<OrdersSchemeOf(kScheme), kF32Composed>(
                    nmax, x, out, F32RungDegree<decltype(kDegreesA)>{kDegreesA});
            }
        } else
        {
            // The basis is the table this scheme's summation reads, as above: the two
            // stored forms of one fit hold different numbers.
            static constexpr auto kDegreesA =
                detail::NarrowRegionADegrees<kAccuracyMultiplier, kRole,
                                             SchemeTailBasis<kScheme>()>();

            if constexpr (kRoute == FitRoute::kRationalMinimax)
            {
                // The rung's own pairs, cut from the narrow partition's coefficients at
                // this rung's multiplier and role - the table the scalar lane's rung body
                // reads, so the two answer a rung with the same values.
                static constexpr auto kPairsA =
                    NarrowRationalRegionAF32Degrees<kAccuracyMultiplier, kRole>();

                F32NarrowRationalBody(nmax, x, out, F32RungPairs<decltype(kPairsA)>{kPairsA});
            } else
            {
                F32NarrowOrdersBody<OrdersSchemeOf(kScheme), kF32Composed>(
                    nmax, x, out, F32RungDegree<decltype(kDegreesA)>{kDegreesA});
            }
        }
    }
}

// The shapes a float policy can name on this axis: two schemes and two budgets at the
// reference multiplier with either route, and the six relaxed rungs of either route at
// either scheme, each instantiated here so the dispatch in boys_impl.hpp is a branch over
// code the library already holds rather than a further instantiation per call site.
//
// The narrow partition's shapes sit beside them, the same count: the four (scheme, route)
// pairs at the reference rung and those four at each of the six relaxed rungs, cut
// against that partition's own table, expanded by BOYS_ORDERS_F32_PACKED_NARROW and the
// two macros under it. tools/check_orders_packed_cells.py holds this list and the
// header's declarations to each other.
//
// A rung is a table of effective degrees cut from the coefficients the named family
// stores - the shipped route's pieces for the polynomial table, the route's own pairs
// for the rational one - so a rung of either route is a reading of the family named.
#define BOYS_ORDERS_F32_PACKED_REFERENCE(kScheme, kBudget, kForm)                                  \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget,              \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;\
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget,        \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, kMultiplier, kBudget, kForm)                  \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, kRoute, kBudget,                    \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;

// The reference rung on the narrow partition: both routes are served there and
// the scheme is the lane's own, so one entry per scheme.
#define BOYS_ORDERS_F32_PACKED_NARROW(kScheme, kBudget, kForm)                                     \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget,              \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;\
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget,        \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;

// The relaxed rungs on the narrow partition: the same four (scheme, route) pairs the
// shipped lane serves at a rung, cut against the narrow table.
#define BOYS_ORDERS_F32_PACKED_NARROW_RUNG(kScheme, kRoute, kMultiplier, kBudget, kForm)           \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, kRoute, kBudget,                    \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(kBudget, kForm)                                        \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 64.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 256.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 1024.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 4096.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 16384.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 65536.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 64.0,      \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 256.0,     \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 1024.0,    \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 4096.0,    \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 16384.0,   \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 65536.0,   \
                                       kBudget, kForm)

#define BOYS_ORDERS_F32_PACKED_RUNGS(kBudget, kForm)                                               \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,                  \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 64.0, kBudget, kForm)   \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 256.0, kBudget, kForm)  \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 1024.0, kBudget, kForm) \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 4096.0, kBudget, kForm) \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 16384.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 65536.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,            \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 64.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 256.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 1024.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 4096.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 16384.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 65536.0, kBudget, kForm)

// The uniform partition, which this lane carries at every multiplier: the grid's cells
// are stored at the degrees the derivation fitted them at, so the rung and the reference
// multiplier read the same table - which is why the block below is the shipped and narrow
// blocks' shape at one route and all seven multipliers rather than the double lane's
// single line. A relaxed call is served the stored cells uncut, a saving left unclaimed
// rather than a value missing, and the probe measures a cell of the partition at each
// tier on both axes. Both routes are carried: the rational member is a fit of this lane's
// grid, and the arm delegates its call to the scalar lane.
#define BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, kMultiplier, kBudget, kForm)                  \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, FitRoute::kChebyshev, kBudget,      \
                                         FitGranularity::kUniform, kForm>(int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, kMultiplier, kBudget, kForm)         \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, FitRoute::kRationalMinimax,         \
                                         kBudget, FitGranularity::kUniform, kForm>(                \
        int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_UNIFORM(kScheme, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 1.0, kBudget, kForm)                              \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 64.0, kBudget, kForm)                             \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 256.0, kBudget, kForm)                            \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 1024.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 4096.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 16384.0, kBudget, kForm)                          \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 65536.0, kBudget, kForm)                        \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 1.0, kBudget, kForm)                     \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 64.0, kBudget, kForm)                    \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 256.0, kBudget, kForm)                   \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 1024.0, kBudget, kForm)                  \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 4096.0, kBudget, kForm)                  \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 16384.0, kBudget, kForm)                 \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 65536.0, kBudget, kForm)

BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                    \
                                 DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                     \
                                 DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFloat,                           \
                                 DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_REFERENCE(EvalScheme::kHorner, BoysBudget::kFp16,                            \
                                 DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFloat, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFloat, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFp16, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_RUNGS(BoysBudget::kFp16, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kRefinedReciprocal)

BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kRefinedReciprocal)

#undef BOYS_ORDERS_F32_PACKED_REFERENCE
#undef BOYS_ORDERS_F32_PACKED_NARROW
#undef BOYS_ORDERS_F32_PACKED_NARROW_RUNG
#undef BOYS_ORDERS_F32_PACKED_NARROW_RUNGS
#undef BOYS_ORDERS_F32_PACKED_RUNGS
#undef BOYS_ORDERS_F32_PACKED_RUNG
#undef BOYS_ORDERS_F32_PACKED_UNIFORM
#undef BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG
#undef BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG

// The entry the public surface's orders axis dispatches to (boys_impl.hpp): the scheme,
// the route, the accuracy multiplier, the partition and the division form, each a
// template argument.
//
// The accuracy multiplier picks the degree a fit is read at - the stored degree at the
// reference rung, the truncation criterion's at a relaxed one - and the partition picks
// the table it is read from: the shipped pieces, whose shared shape lets one stride fetch
// four orders' coefficients, or the narrow ones, cut per order. The division form selects
// nothing here; it is carried for the certified scalar single lane each fallback hands its
// orders to, which divides at every step.
//
// The stored fit is summed composed rather than gathered: the two fetches are the same
// lane value for value, and composed is the cheaper in retired slots on the machine this
// lane was measured on, where the gather is a microcode assist. The gathered entry stays
// beside it so the pair remains measurable. The narrow partition's fetch is a gather by
// construction - its pieces have no stride between them - so it has no composed twin.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity,
          DivisionForm kForm>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    static_assert(kAccuracyMultiplier >= 1.0,
                  "kAccuracyMultiplier must be >= 1.0 (1.0 = full static accuracy)");
    static_assert(kRoute == FitRoute::kChebyshev || kRoute == FitRoute::kRationalMinimax,
                  "a policy naming a route outside the FitRoute enumeration is not one this "
                  "library serves: name FitRoute::kChebyshev or FitRoute::kRationalMinimax");
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x >= 0.0);
    assert(out != nullptr);

    if constexpr (kGranularity == FitGranularity::kUniform)
    {
        // The fixed grid, which the lane reads as the one contiguous ladder it is: the
        // coefficients of every order of one interval sit one after the other, so the
        // fetch that steps an order to the next is a stride here as it is on the shipped
        // cover, and the argument's interval is one multiply and a truncation rather than
        // a scan of piece edges.
        //
        // The partition names how the fitted intervals are cut, not which family is
        // fitted, so the Chebyshev family's path is the one the packed body below reads
        // and the scheme picks which of its two stored forms is summed.
        //
        // The rung is read as the partition's own rows read it: the whole ladder is the
        // grid's stored cells below the join and the certified scalar lane above it, and
        // neither reads a degree the multiplier cuts, so every rung is served by this body
        // and by the same coefficients. The delegation below carries the multiplier past
        // the join, to the entry that owns this partition's rung answer.

        if (x == 0.0)
        {
            for (int l = 0; l <= nmax; ++l)
            {
                out[l] = 1.0 / (2.0 * l + 1.0);
            }

            return;
        }

        // The rational member's rows are per interval and not one stride, so there is no
        // packed fetch to make of them: the axis is served by the certified scalar orders
        // lane, which reads the same member one order at a time through the policy's own
        // single-order entry. The narrow rational route takes the same path, its pieces
        // having no stride between them either.
        //
        // The two are branches of one `if constexpr` rather than an early return above
        // the Chebyshev path: a taken early return leaves everything after it unreachable
        // in that instantiation, and this build refuses that warning. The branch taken
        // also decides which of the two the compiler keeps, so no argument of this route
        // reaches UniformOrdersBody, which sums the Chebyshev member.
        if constexpr (kRoute == FitRoute::kRationalMinimax)
        {
            ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity, kForm>(nmax, x, out,
                                                                                   1);
        }
        else
        {
            // Past the grid - which reaches above kX1, so this is the asymptotic's own
            // domain - and on a host without the vector tier, the entry is the certified
            // scalar single lane at the policy the caller named, one order at a time.
            // That lane reads this partition's own table and domain.
            if (x >= detail::kFlatHi || !BoysAvx2Available())
            {
                ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity, kForm>(nmax, x,
                                                                                       out, 1);
                return;
            }

            UniformOrdersBody<OrdersSchemeOf(kScheme), true>(nmax, x, out, 1);
        }

        return;
    } else if constexpr (kRoute == FitRoute::kChebyshev &&
                         kAccuracyMultiplier == kBoysFullAccuracyMultiplier &&
                         kGranularity == FitGranularity::kShipped)
    {
        // The shipped route's reference rung on the shipped partition, at the form the
        // caller named: the same body and tables, and the same fallback the suite pins bit
        // for bit when no form is named. The form is passed through rather than defaulted
        // because the fallback past the interval is where the two forms part, and the
        // partition is part of the condition because this body reads the shipped table.
        BoysAllOrdersSimdAtForm<true, kForm>(OrdersSchemeOf(kScheme), nmax, x, out, 1);
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

        if constexpr (kGranularity == FitGranularity::kShipped)
        {
            if (!OrdersLaneApplies(x))
            {
                ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity, kForm>(
                    nmax, x, out, 1);
                return;
            }
        } else
        {
            if (!NarrowOrdersLaneApplies(x))
            {
                ScalarOrders<kScheme, kAccuracyMultiplier, kRoute, kGranularity, kForm>(
                    nmax, x, out, 1);
                return;
            }
        }

        // The degrees this partition's fits are read at. The criterion's budget is zero at
        // the reference rung, which leaves every fit at its stored degree, so the one table
        // covers every rung. The form is named in the policy because this body reads one
        // policy and not one policy's table: the criterion consults the partition and the
        // scheme, so a policy written with another form's field would be this body
        // claiming a policy the caller did not name.
        static constexpr auto kDegrees =
            RegionADegreeTableOf<kAccuracyMultiplier,
                                 EvalPolicy<kRoute, kScheme, BoysBudget::kFloat,
                                            PackAxis::kOrders, kGranularity, kForm>,
                                 BoysRole::kDoubleSingle>();

        if constexpr (kGranularity != FitGranularity::kShipped)
        {
            // The narrow partition is a partition of the shipped route's own region-A
            // fits, read with the per-order rule the route's per-argument body applies:
            // below an order's own end of region A the partition's fit answers at this
            // rung's degree, at and above it the route's pair at this rung's cut.
            if constexpr (kRoute == FitRoute::kRationalMinimax)
            {
                static constexpr auto kPairs = RationalRegionANarrowDegrees<kAccuracyMultiplier>();

                NarrowRationalOrdersBody<kScheme>(
                    nmax, x, out, 1, kPairs, RungDegree<decltype(kDegrees)>{kDegrees});
            }
            else
            {
                NarrowOrdersBody<OrdersSchemeOf(kScheme)>(
                    nmax, x, out, 1, RungDegree<decltype(kDegrees)>{kDegrees});
            }
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
// multiplier and the six relaxed rungs, instantiated here for the reason above.
#define BOYS_ORDERS_PACKED_INSTANTIATIONS(kScheme, kForm)                                          \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev,                          \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev,                         \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax,                    \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax,                   \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax,                  \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;

// The narrow partition, at both schemes and every rung, on the shipped route
// alone: the partition is a partition of that route's region-A fits.
#define BOYS_ORDERS_NARROW_INSTANTIATIONS(kScheme, kForm)                                          \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev, FitGranularity::kNarrow, kForm>(\
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev, FitGranularity::kNarrow, kForm>(\
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;

// The same partition on the other route: the two routes' regions do not
// coincide - the route answers an order from that order's own end of region A -
// so a rung of the pair is a combination of its own.
#define BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(kScheme, kForm)                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax,                    \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax,                   \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax,                  \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;

// The uniform partition, whose cells are the scheme's and the route's. The Chebyshev
// member's cell is the packed body below, instantiated at every rung: the grid's cells
// are stored at the degrees the derivation fitted them at and the criterion that would
// cut them reaches the full degree at every multiplier, so the rung is the reference
// reading. The rational member's rows are per interval and have no stride to step, so
// its cell is the body's delegation to the scalar orders lane, at the same seven
// multipliers.
#define BOYS_ORDERS_UNIFORM_INSTANTIATIONS(kScheme, kForm)                                         \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev,                          \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev,                         \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;

#define BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, kMultiplier, kForm)                             \
    template void BoysAllOrdersPacked<kScheme, kMultiplier, FitRoute::kRationalMinimax,            \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;

#define BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(kScheme, kForm)                                \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 1.0, kForm)                                         \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 64.0, kForm)                                        \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 256.0, kForm)                                       \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 1024.0, kForm)                                      \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 4096.0, kForm)                                      \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 16384.0, kForm)                                     \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 65536.0, kForm)

BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                           DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                           DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_RATIONAL_RUNG

} // namespace boys::detail

#else // BOYS_SIMD_X86

// Non-x86 targets: the entries exist and are defined against the certified scalar
// lanes, exactly as the region entries of boys_simd.cpp are.
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

// The float lane's measurement entries, for the reason above: the vector tier
// is absent, so the lane is the fits it would have vectorised, one order at a
// time.
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

// The orders axis's float entry on a target without the vector tier: the same certified
// scalar single lane the packed bodies fall back to, at the policy the axis names. The
// partition and the division form are part of the policy and not only of the body that
// reads it, so both are parameters here as on the vector tier's entry.
template <EvalScheme kScheme, double kAccuracyMultiplier, FitRoute kRoute, BoysBudget kBudget,
          FitGranularity kGranularity, DivisionForm kForm>
void BoysAllOrdersF32Packed(int nmax, float x, float* out) noexcept {
    using Policy =
        EvalPolicy<kRoute, kScheme, kBudget, PackAxis::kArguments, kGranularity, kForm>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingleF32<kAccuracyMultiplier, Policy>(l, x);
    }
}

#define BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(kScheme, kBudget, kForm)                             \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget,              \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;\
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget,        \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;\
    BOYS_ORDERS_F32_PACKED_RUNGS_ROUTE(kScheme, FitRoute::kChebyshev, kBudget, kForm)              \
    BOYS_ORDERS_F32_PACKED_RUNGS_ROUTE(kScheme, FitRoute::kRationalMinimax, kBudget, kForm)

// The six relaxed rungs on a target without the vector tier, at either route: the entry
// there is the certified scalar single lane at the policy the axis names, so a rung is
// served by that lane's own rung body.
#define BOYS_ORDERS_F32_PACKED_RUNGS_ROUTE(kScheme, kRoute, kBudget, kForm)                        \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 64.0, kBudget, kForm)                             \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 256.0, kBudget, kForm)                            \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 1024.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 4096.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 16384.0, kBudget, kForm)                          \
    BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, 65536.0, kBudget, kForm)

#define BOYS_ORDERS_F32_PACKED_RUNG(kScheme, kRoute, kMultiplier, kBudget, kForm)                  \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, kRoute, kBudget,                    \
                                         FitGranularity::kShipped, kForm>(int, float, float*) noexcept;

// The narrow partition's shapes, which the axis serves on this target as it does on the
// vector tier, and it is the same count: both routes and both schemes at the reference
// rung, and the four (scheme, route) pairs at each of the six relaxed rungs below.
#define BOYS_ORDERS_F32_PACKED_NARROW(kScheme, kBudget, kForm)                                     \
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kChebyshev, kBudget,              \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;\
    template void BoysAllOrdersF32Packed<kScheme, 1.0, FitRoute::kRationalMinimax, kBudget,        \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;

// The relaxed rungs on the narrow partition, for the reason the x86 branch above gives.
#define BOYS_ORDERS_F32_PACKED_NARROW_RUNG(kScheme, kRoute, kMultiplier, kBudget, kForm)           \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, kRoute, kBudget,                    \
                                         FitGranularity::kNarrow, kForm>(int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(kBudget, kForm)                                        \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kChebyshev,           \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 64.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 256.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 1024.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 4096.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 16384.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kChebyshev, 65536.0, kBudget, kForm)\
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       64.0, kBudget, kForm)                                       \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       256.0, kBudget, kForm)                                      \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       1024.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       4096.0, kBudget, kForm)                                     \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       16384.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax,     \
                                       65536.0, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 64.0,      \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 256.0,     \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 1024.0,    \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 4096.0,    \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 16384.0,   \
                                       kBudget, kForm)                                             \
    BOYS_ORDERS_F32_PACKED_NARROW_RUNG(EvalScheme::kHorner, FitRoute::kRationalMinimax, 65536.0,   \
                                       kBudget, kForm)

// The uniform partition, for the reason the x86 branch above gives: the target changes
// which lane serves the cells rather than which of them exist.
#define BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, kMultiplier, kBudget, kForm)                  \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, FitRoute::kChebyshev, kBudget,      \
                                         FitGranularity::kUniform, kForm>(int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, kMultiplier, kBudget, kForm)         \
    template void BoysAllOrdersF32Packed<kScheme, kMultiplier, FitRoute::kRationalMinimax,         \
                                         kBudget, FitGranularity::kUniform, kForm>(                \
        int, float, float*) noexcept;

#define BOYS_ORDERS_F32_PACKED_UNIFORM(kScheme, kBudget, kForm)                                    \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 1.0, kBudget, kForm)                              \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 64.0, kBudget, kForm)                             \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 256.0, kBudget, kForm)                            \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 1024.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 4096.0, kBudget, kForm)                           \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 16384.0, kBudget, kForm)                          \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG(kScheme, 65536.0, kBudget, kForm)                        \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 1.0, kBudget, kForm)                     \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 64.0, kBudget, kForm)                    \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 256.0, kBudget, kForm)                   \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 1024.0, kBudget, kForm)                  \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 4096.0, kBudget, kForm)                  \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 16384.0, kBudget, kForm)                 \
    BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG(kScheme, 65536.0, kBudget, kForm)

BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,               \
                                      DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,               \
                                      DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,               \
                                      DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                \
                                      DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                \
                                      DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                \
                                      DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFloat,                      \
                                      DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFloat,                      \
                                      DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFloat,                      \
                                      DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFp16,                       \
                                      DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFp16,                       \
                                      DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_INSTANTIATIONS(EvalScheme::kHorner, BoysBudget::kFp16,                       \
                                      DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                       \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                        \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFloat,                              \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW(EvalScheme::kHorner, BoysBudget::kFp16,                               \
                              DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFloat, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_NARROW_RUNGS(BoysBudget::kFp16, DivisionForm::kRefinedReciprocal)

BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFloat,                      \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kSplitClenshaw, BoysBudget::kFp16,                       \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFloat,                             \
                               DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16, DivisionForm::kExactDivision)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kPlainReciprocal)
BOYS_ORDERS_F32_PACKED_UNIFORM(EvalScheme::kHorner, BoysBudget::kFp16,                              \
                               DivisionForm::kRefinedReciprocal)

#undef BOYS_ORDERS_F32_PACKED_INSTANTIATIONS
#undef BOYS_ORDERS_F32_PACKED_NARROW
#undef BOYS_ORDERS_F32_PACKED_NARROW_RUNG
#undef BOYS_ORDERS_F32_PACKED_NARROW_RUNGS
#undef BOYS_ORDERS_F32_PACKED_RUNGS_ROUTE
#undef BOYS_ORDERS_F32_PACKED_RUNG
#undef BOYS_ORDERS_F32_PACKED_UNIFORM
#undef BOYS_ORDERS_F32_PACKED_UNIFORM_RUNG
#undef BOYS_ORDERS_F32_PACKED_UNIFORM_RATIONAL_RUNG

// The orders axis's double entry on a target without the vector tier: the same
// certified scalar single lane the packed bodies fall back to, at the policy
// the axis names.
template <EvalScheme kScheme,
          double kAccuracyMultiplier,
          FitRoute kRoute,
          FitGranularity kGranularity,
          DivisionForm kForm>
void BoysAllOrdersPacked(int nmax, double x, double* out) noexcept {
    using Policy = EvalPolicy<kRoute, kScheme, BoysBudget::kFloat, PackAxis::kArguments,
                              kGranularity, kForm>;

    for (int l = 0; l <= nmax; ++l)
    {
        out[l] = BoysSingle<kAccuracyMultiplier, Policy>(l, x);
    }
}

#define BOYS_ORDERS_PACKED_INSTANTIATIONS(kScheme, kForm)                                          \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev,                          \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev,                         \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax,                    \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax,                   \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax,                  \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kShipped, kForm>(int, double, double*) noexcept;

// The narrow partition, at both schemes and every rung, on the shipped route
// alone: the partition is a partition of that route's region-A fits.
#define BOYS_ORDERS_NARROW_INSTANTIATIONS(kScheme, kForm)                                          \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev, FitGranularity::kNarrow, kForm>(\
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev, FitGranularity::kNarrow, kForm>(\
        int, double, double*) noexcept;                                                            \
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;

// The same partition on the other route, for the reason the vector tier
// instantiates it.
#define BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(kScheme, kForm)                                 \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kRationalMinimax,                    \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kRationalMinimax,                   \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kRationalMinimax,                  \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kRationalMinimax,                 \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kRationalMinimax,                \
                                      FitGranularity::kNarrow, kForm>(int, double, double*) noexcept;

// The uniform partition's cells, for the reason the vector branch states: the entry here
// is the certified scalar single lane at the policy the axis names, and that lane reads
// this partition's own table, so a cell the vector tier serves is a cell this target
// serves the same way. The rational member's cell is the same loop at the other route, one
// instantiation beside the Chebyshev member's rather than a body of its own.
#define BOYS_ORDERS_UNIFORM_INSTANTIATIONS(kScheme, kForm)                                         \
    template void BoysAllOrdersPacked<kScheme, 1.0, FitRoute::kChebyshev,                          \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 64.0, FitRoute::kChebyshev,                         \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 256.0, FitRoute::kChebyshev,                        \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 1024.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 4096.0, FitRoute::kChebyshev,                       \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 16384.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;\
    template void BoysAllOrdersPacked<kScheme, 65536.0, FitRoute::kChebyshev,                      \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;

#define BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, kMultiplier, kForm)                             \
    template void BoysAllOrdersPacked<kScheme, kMultiplier, FitRoute::kRationalMinimax,            \
                                      FitGranularity::kUniform, kForm>(int, double, double*) noexcept;

#define BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(kScheme, kForm)                                \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 1.0, kForm)                                         \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 64.0, kForm)                                        \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 256.0, kForm)                                       \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 1024.0, kForm)                                      \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 4096.0, kForm)                                      \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 16384.0, kForm)                                     \
    BOYS_ORDERS_UNIFORM_RATIONAL_RUNG(kScheme, 65536.0, kForm)

BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_PACKED_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                           DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                           DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kSplitClenshaw, DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kSplitClenshaw,                              \
                                            DivisionForm::kRefinedReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kExactDivision)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kPlainReciprocal)
BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS(EvalScheme::kHorner, DivisionForm::kRefinedReciprocal)

#undef BOYS_ORDERS_PACKED_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_INSTANTIATIONS
#undef BOYS_ORDERS_NARROW_RATIONAL_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_RATIONAL_INSTANTIATIONS
#undef BOYS_ORDERS_UNIFORM_RATIONAL_RUNG

} // namespace boys::detail

#endif // BOYS_SIMD_X86
