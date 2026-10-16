#pragma once

/// \file boys_arguments_lanes.hpp
/// The arguments axis' packed lanes: four ARGUMENTS of one order to a register.
///
/// A lane here is either the tier's own packed body or, at the scalar tier, the certified scalar
/// lane walked one argument at a time - the two deliver the same bits, which is what the tier's
/// flag set buys in place of the second body a run-time probe would need. The tier is
/// \c BOYS_ISA_AVX2_FMA (boys/isa_tiers.hpp); the vector bodies below are the only place on a
/// consumer-reachable path where this library names an AVX2 intrinsic.
///
/// What lives here is the part of the arguments axis a consumer's own translation unit reaches:
/// region A's within-the-fits lane, and the far region's ladder in the arguments lane with the
/// run kernel the sorted and partitioned entries hand their far tail to. The region B and C
/// lanes, the half-format lanes and the backend registry append are measurement surfaces of the
/// library's own binaries and stay out of line, in src/boys_simd.cpp.

#include "boys/backend.hpp"
#include "boys/boys_backend_simd.hpp"
#include "boys/boys_coefficients.hpp"
#include "boys/isa_tiers.hpp"

#include <cassert>
#include <cstddef>

#if BOYS_ISA_AVX2_FMA

#include <immintrin.h>

namespace boys::detail {

// The lanes below pack four arguments of one order. Nothing here is a second arithmetic: every
// body is the scalar lane's own operations in the scalar lane's own order, so a value a lane
// returns is that lane's value at every bit.

using detail::kX0;
using detail::kX1;

constexpr double kHalfSqrtPi = 0.886226925452758014;

// Split Clenshaw (even/odd), packed, half-depth FMA chains: T_{2j+1}(t) = t * D_j(v) with the D
// recurrence. One body for every packed width and precision - which multiply-add a step uses is
// the backend's. What stays here is the mapped argument, in the packed lanes' own form.
template <backend::ArithmeticBackend B, typename Piece>
typename B::Packed RegionAClenshaw(const typename B::Value* c,
                                   const Piece& piece,
                                   int deg,
                                   typename B::Packed xv) noexcept {
    using V = typename B::Value;
    const typename B::Packed t = B::MulAdd(B::Sub(xv, B::Broadcast(piece.a)),
                                           B::Broadcast(V{2} / (piece.b - piece.a)),
                                           B::Broadcast(-V{1}));
    return ClenshawSplit<B>(c + piece.offset, deg, t);
}

// 4-wide split Clenshaw for one piece; even deg >= 4 only.
inline __m256d Clenshaw4SplitDeg(const detail::OrderPiece& piece, int deg, __m256d xv) noexcept {
    return RegionAClenshaw<backend::Avx2Fp64<>>(detail::kCoeffs.data(), piece, deg, xv);
}

// The m = 1 entry: the piece's full degree.
inline __m256d Clenshaw4Split(const detail::OrderPiece& piece, __m256d xv) noexcept {
    return Clenshaw4SplitDeg(piece, piece.deg, xv);
}


inline void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    const int first = detail::kPieceStart[n];
    const int last = detail::kPieceStart[n + 1];

    {
        for (std::size_t i = 0; i + 3 < count; i += 4)
        {
            const __m256d xv = _mm256_loadu_pd(x + i);
            __m256d acc = _mm256_setzero_pd();

            for (int p = first; p < last; ++p)
            {
                const detail::OrderPiece& piece = detail::kPieces[p];
                const __m256d mask =
                    _mm256_and_pd(_mm256_cmp_pd(xv, _mm256_set1_pd(piece.a), _CMP_GE_OQ),
                                  _mm256_cmp_pd(xv, _mm256_set1_pd(piece.b), _CMP_LT_OQ));
                acc = _mm256_blendv_pd(acc, Clenshaw4Split(piece, xv), mask);
            }

            _mm256_storeu_pd(out + i, acc);
        }
    }

    for (std::size_t i = count - (count % 4); i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}


// The far region's ladder in the arguments lane: four arguments of ONE ladder's terms at once,
// written through the scalar body's own operations in its own order, so the lane's values are the
// per-argument body's at every bit. Neither the division form (the ladder's one division is the
// seed's closed form, not a step of a chain) nor the multiply-add route (no a * b + c below) reaches it.

// One order's four values into the four columns the writer named. Four consecutive doubles are
// one register store - the sorted entry's groups, which is the shape the tile exists for - and a
// permutation's four addresses share no store, so each lane's value lands on its own.
template <bool kContiguous>
void StoreLanes(int order, std::size_t stride, double* const* cols, __m256d f) noexcept {
    const std::size_t at = static_cast<std::size_t>(order) * stride;

    if constexpr (kContiguous)
    {
        _mm256_storeu_pd(cols[0] + at, f);
    } else
    {
        alignas(32) double lane[4];
        _mm256_store_pd(lane, f);

        for (std::size_t j = 0; j < 4; ++j)
        {
            cols[j][at] = lane[j];
        }
    }
}

// One run of the ladder, four arguments at once: FarLadderRun's `w`, its running h^(4j) and a term
// as two multiplies against the table entry, lane for lane. The order steps in fours and the last
// group of three or fewer terms takes the same registers at the running power - the scalar loop's
// own cut - so the four arguments are the scalar body's four values bit for bit.
template <DivisionForm kForm, bool kContiguous>
void FarLadderLanesAt(int nmax,
                      const double* x,
                      std::size_t stride,
                      double* const* cols) noexcept {
    static_cast<void>(kForm); // the ladder's one division is the seed's closed form, as in the body

    const __m256d xv = _mm256_loadu_pd(x);
    const __m256d r = _mm256_div_pd(_mm256_set1_pd(1.0), _mm256_sqrt_pd(xv));
    const __m256d h = _mm256_mul_pd(_mm256_set1_pd(0.5), _mm256_mul_pd(r, r));
    const __m256d h2 = _mm256_mul_pd(h, h);
    const __m256d h3 = _mm256_mul_pd(h2, h);
    const __m256d h4 = _mm256_mul_pd(h2, h2);
    const __m256d w0 = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), r);
    const __m256d w1 = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_mul_pd(r, h));
    const __m256d w2 = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_mul_pd(r, h2));
    const __m256d w3 = _mm256_mul_pd(_mm256_set1_pd(kHalfSqrtPi), _mm256_mul_pd(r, h3));
    const __m256d w[4] = {w0, w1, w2, w3};

    __m256d p = _mm256_set1_pd(1.0);
    int l = 0;

    for (; l + 4 <= nmax + 1; l += 4)
    {
        StoreLanes<kContiguous>(
            l + 0, stride, cols,
            _mm256_mul_pd(_mm256_mul_pd(w0, _mm256_set1_pd(kDoubleFactorial[l + 0])), p));
        StoreLanes<kContiguous>(
            l + 1, stride, cols,
            _mm256_mul_pd(_mm256_mul_pd(w1, _mm256_set1_pd(kDoubleFactorial[l + 1])), p));
        StoreLanes<kContiguous>(
            l + 2, stride, cols,
            _mm256_mul_pd(_mm256_mul_pd(w2, _mm256_set1_pd(kDoubleFactorial[l + 2])), p));
        StoreLanes<kContiguous>(
            l + 3, stride, cols,
            _mm256_mul_pd(_mm256_mul_pd(w3, _mm256_set1_pd(kDoubleFactorial[l + 3])), p));
        p = _mm256_mul_pd(p, h4);
    }

    for (; l <= nmax; ++l)
    {
        StoreLanes<kContiguous>(
            l, stride, cols,
            _mm256_mul_pd(_mm256_mul_pd(w[l & 3], _mm256_set1_pd(kDoubleFactorial[l])), p));
    }
}

/// The run's writers, the entry's two shapes: see the declaration for which one a run takes and
/// why. Both are the per-argument body's own arithmetic - the lane's groups and that body's rows.
template <DivisionForm kForm>
void FarRunDirect(int nmax,
                  const double* x,
                  const std::size_t* index,
                  double* out,
                  std::size_t count,
                  std::size_t begin,
                  std::size_t end) noexcept {
    double xs[4];
    double* cols[4];
    std::size_t j = begin;

    for (; j + 4 <= end; j += 4)
    {
        for (std::size_t t = 0; t < 4; ++t)
        {
            const std::size_t i = (index != nullptr) ? index[j + t] : j + t;
            xs[t] = x[i];
            cols[t] = out + i;
        }

        const bool contiguous = cols[1] == cols[0] + 1 && cols[2] == cols[0] + 2 &&
                                cols[3] == cols[0] + 3;

        if (contiguous)
        {
            FarLadderLanesAt<kForm, true>(nmax, xs, count, cols);
        }
        else
        {
            FarLadderLanesAt<kForm, false>(nmax, xs, count, cols);
        }
    }

    for (; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;
        BoysAllNBodyRegionC<kForm>(nmax, x[i], count, out + i);
    }
}

template <DivisionForm kForm>
void FarRunTiled(int nmax,
                 const double* x,
                 double* out,
                 std::size_t count,
                 std::size_t begin,
                 std::size_t end) noexcept {
    BoysAllNTile(nmax, count, nullptr, out, begin, end,
                 [&](std::size_t base, std::size_t block, double* stage) {
                     double xs[4];
                     double* cols[4];
                     std::size_t t = 0;

                     for (; t + 4 <= block; t += 4)
                     {
                         for (std::size_t q = 0; q < 4; ++q)
                         {
                             xs[q] = x[base + t + q];
                             cols[q] = stage + t + q;
                         }

                         FarLadderLanesAt<kForm, true>(nmax, xs, block, cols);
                     }

                     for (; t < block; ++t)
                     {
                         BoysAllNBodyRegionC<kForm>(nmax, x[base + t], block, stage + t);
                     }
                 });
}

template <DivisionForm kForm>
void FarRunAt(int nmax,
              const double* x,
              const std::size_t* index,
              double* out,
              std::size_t count,
              std::size_t begin,
              std::size_t end) noexcept {
    if (index == nullptr)
    {
        FarRunTiled<kForm>(nmax, x, out, count, begin, end);
        return;
    }

    FarRunDirect<kForm>(nmax, x, index, out, count, begin, end);
}

inline void BoysFarLadderLanes(int nmax,
                        const double* x,
                        std::size_t stride,
                        double* const* cols,
                        DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(x[0] >= kX1 && x[1] >= kX1 && x[2] >= kX1 && x[3] >= kX1);

    // Which of the two stores a group takes is a property of the columns and not of the shape:
    // the sorted entry's groups are four consecutive doubles of a plane - or of the tile, which
    // is the same four addresses - and the unsorted entry's are wherever its permutation put
    // them. Decided once per group, and the values either way are the same.
    const bool contiguous = cols[1] == cols[0] + 1 && cols[2] == cols[0] + 2 &&
                            cols[3] == cols[0] + 3;

    if (contiguous)
    {
        switch (form)
        {
        case DivisionForm::kExactDivision:
            FarLadderLanesAt<DivisionForm::kExactDivision, true>(nmax, x, stride, cols);
            break;

        case DivisionForm::kPlainReciprocal:
            FarLadderLanesAt<DivisionForm::kPlainReciprocal, true>(nmax, x, stride, cols);
            break;

        case DivisionForm::kRefinedReciprocal:
            FarLadderLanesAt<DivisionForm::kRefinedReciprocal, true>(nmax, x, stride, cols);
            break;
        }

        return;
    }

    switch (form)
    {
    case DivisionForm::kExactDivision:
        FarLadderLanesAt<DivisionForm::kExactDivision, false>(nmax, x, stride, cols);
        break;

    case DivisionForm::kPlainReciprocal:
        FarLadderLanesAt<DivisionForm::kPlainReciprocal, false>(nmax, x, stride, cols);
        break;

    case DivisionForm::kRefinedReciprocal:
        FarLadderLanesAt<DivisionForm::kRefinedReciprocal, false>(nmax, x, stride, cols);
        break;
    }
}

inline void BoysFarLadderRun(int nmax,
                      const double* x,
                      const std::size_t* index,
                      double* out,
                      std::size_t count,
                      std::size_t begin,
                      std::size_t end,
                      DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(begin < end && end - begin >= 4);
    assert(index == nullptr || (index[begin] < count && index[end - 1] < count));

    switch (form)
    {
    case DivisionForm::kExactDivision:
        FarRunAt<DivisionForm::kExactDivision>(nmax, x, index, out, count, begin, end);
        break;

    case DivisionForm::kPlainReciprocal:
        FarRunAt<DivisionForm::kPlainReciprocal>(nmax, x, index, out, count, begin, end);
        break;

    case DivisionForm::kRefinedReciprocal:
        FarRunAt<DivisionForm::kRefinedReciprocal>(nmax, x, index, out, count, begin, end);
        break;
    }
}

} // namespace boys::detail

#else // BOYS_ISA_AVX2_FMA

// The scalar tier: the same entry points, defined against the certified scalar lanes. There is no
// vector engine in this unit to assert - BoysAvx2Available() answers for the MACHINE and not for
// this build, so it is not what a lane here rests on. The bodies mirror the scalar tails the
// vector entries above run for their last count % 4 elements, and are all a non-x86 target has.

namespace boys::detail {

inline void BoysRegionASimd(int n, const double* x, double* out, std::size_t count) noexcept {
    assert(n >= 0 && n <= kMaxBoysOrder);

    for (std::size_t i = 0; i < count; ++i)
    {
        out[i] = BoysSingle<>(n, x[i]);
    }
}

// The far ladder's lane at the scalar tier: the region's certified scalar body, one column at a
// time. It is the body the entry's own scalar path runs, at the same arguments and the same
// stride, so the lane returns the values the entry returns - the host's AVX2 being present or
// absent says nothing about which tier this translation unit was compiled at.
inline void BoysFarLadderLanes(int nmax,
                        const double* x,
                        std::size_t stride,
                        double* const* cols,
                        DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);

    for (std::size_t j = 0; j < kBoysFarLaneWidth; ++j)
    {
        switch (form)
        {
        case DivisionForm::kExactDivision:
            BoysAllNBodyRegionC<DivisionForm::kExactDivision>(nmax, x[j], stride, cols[j]);
            break;

        case DivisionForm::kPlainReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kPlainReciprocal>(nmax, x[j], stride, cols[j]);
            break;

        case DivisionForm::kRefinedReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kRefinedReciprocal>(nmax, x[j], stride, cols[j]);
            break;
        }
    }
}

// The far run's kernel at the scalar tier: the region's certified scalar body, one argument of
// the run at a time, in the caller's own planes. It is the body the entry's scalar path runs, so
// the kernel returns what the entry returns.
inline void BoysFarLadderRun(int nmax,
                      const double* x,
                      const std::size_t* index,
                      double* out,
                      std::size_t count,
                      std::size_t begin,
                      std::size_t end,
                      DivisionForm form) noexcept {
    assert(nmax >= 0 && nmax <= kMaxBoysOrder);
    assert(begin < end);

    for (std::size_t j = begin; j < end; ++j)
    {
        const std::size_t i = (index != nullptr) ? index[j] : j;

        switch (form)
        {
        case DivisionForm::kExactDivision:
            BoysAllNBodyRegionC<DivisionForm::kExactDivision>(nmax, x[i], count, out + i);
            break;

        case DivisionForm::kPlainReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kPlainReciprocal>(nmax, x[i], count, out + i);
            break;

        case DivisionForm::kRefinedReciprocal:
            BoysAllNBodyRegionC<DivisionForm::kRefinedReciprocal>(nmax, x[i], count, out + i);
            break;
        }
    }
}

} // namespace boys::detail

#endif // BOYS_ISA_AVX2_FMA
