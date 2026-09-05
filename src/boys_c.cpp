// Implementation of the C linkage surface (boys_c.h).
//
// The m = 1 entries route to the library's exported certified
// instantiations; the sampled-m entries (m > 1) instantiate the engine from
// the internal headers at the dispatch set below — the library exports only
// the m = 1 instantiations (see boys_impl.hpp), so the relaxed bodies are
// compiled here, exactly as the contract tests compile them.

#include "boysymmetriad/boys_c.h"

#include "boys_effective_degrees.hpp"
#include "boys_impl.hpp"
#include "boysymmetriad/boys.hpp"

namespace {

template <double kAccuracyMultiplier> double BoysSingleRelaxed(int n, double x) noexcept {
    return boysymmetriad::BoysSingle<kAccuracyMultiplier>(n, x);
}

template <double kAccuracyMultiplier> float BoysSingleF32Relaxed(int n, float x) noexcept {
    return boysymmetriad::BoysSingleF32<kAccuracyMultiplier>(n, x);
}

// The sampled multiplier set of the accuracy contract. Dispatch is by exact
// double equality: the set members are exactly representable, so callers
// pass the same literals.
constexpr double kSampledMultipliers[] = {1.0, 2.0, 10.0, 100.0, 1e4, 1e8};

int FindMultiplier(double m) {
    for (int i = 0; i < 6; ++i)
    {
        if (m == kSampledMultipliers[i])
        {
            return i;
        }
    }

    return -1;
}

bool ValidOrder(int n) {
    return n >= 0 && n <= boysymmetriad::kMaxBoysOrder;
}

bool ValidX(double x) {
    return x >= 0.0; // rejects negatives and NaN
}

int RunBatchDouble(int nmax, int count, const double* x, double* out) {
    if (!ValidOrder(nmax) || count < 0)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    if (count == 0)
    {
        return BOYS_SUCCESS;
    }

    if (x == nullptr || out == nullptr)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    double row[boysymmetriad::kMaxBoysOrder + 1];

    for (int i = 0; i < count; ++i)
    {
        if (!ValidX(x[i]))
        {
            return BOYS_ERROR_INVALID_ARGUMENT;
        }

        boysymmetriad::BoysBatch(nmax, x[i], row);

        for (int k = 0; k <= nmax; ++k)
        {
            out[k * count + i] = row[k];
        }
    }

    return BOYS_SUCCESS;
}

} // namespace

extern "C" {

int BoysDouble(int n, double x, double* out) {
    if (out == nullptr || !ValidOrder(n) || !ValidX(x))
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    *out = boysymmetriad::BoysSingle(n, x);
    return BOYS_SUCCESS;
}

int BoysFloat(int n, float x, float* out) {
    if (out == nullptr || !ValidOrder(n) || !ValidX(x))
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    *out = boysymmetriad::BoysSingleF32(n, x);
    return BOYS_SUCCESS;
}

int BoysDoubleWithMultiplier(double m, int n, double x, double* out) {
    if (out == nullptr || !ValidOrder(n) || !ValidX(x))
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    switch (FindMultiplier(m))
    {
    case 0:
        *out = boysymmetriad::BoysSingle(n, x);
        return BOYS_SUCCESS;
    case 1:
        *out = BoysSingleRelaxed<2.0>(n, x);
        return BOYS_SUCCESS;
    case 2:
        *out = BoysSingleRelaxed<10.0>(n, x);
        return BOYS_SUCCESS;
    case 3:
        *out = BoysSingleRelaxed<100.0>(n, x);
        return BOYS_SUCCESS;
    case 4:
        *out = BoysSingleRelaxed<1e4>(n, x);
        return BOYS_SUCCESS;
    case 5:
        *out = BoysSingleRelaxed<1e8>(n, x);
        return BOYS_SUCCESS;
    default:
        return BOYS_ERROR_UNSUPPORTED_MULTIPLIER;
    }
}

int BoysFloatWithMultiplier(double m, int n, float x, float* out) {
    if (out == nullptr || !ValidOrder(n) || !ValidX(x))
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    switch (FindMultiplier(m))
    {
    case 0:
        *out = boysymmetriad::BoysSingleF32(n, x);
        return BOYS_SUCCESS;
    case 1:
        *out = BoysSingleF32Relaxed<2.0>(n, x);
        return BOYS_SUCCESS;
    case 2:
        *out = BoysSingleF32Relaxed<10.0>(n, x);
        return BOYS_SUCCESS;
    case 3:
        *out = BoysSingleF32Relaxed<100.0>(n, x);
        return BOYS_SUCCESS;
    case 4:
        *out = BoysSingleF32Relaxed<1e4>(n, x);
        return BOYS_SUCCESS;
    case 5:
        *out = BoysSingleF32Relaxed<1e8>(n, x);
        return BOYS_SUCCESS;
    default:
        return BOYS_ERROR_UNSUPPORTED_MULTIPLIER;
    }
}

int BoysDoubleBatch(int nmax, int count, const double* x, double* out) {
    return RunBatchDouble(nmax, count, x, out);
}

int BoysFloatBatch(int nmax, int count, const float* x, float* out) {
    if (!ValidOrder(nmax) || count < 0)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    if (count == 0)
    {
        return BOYS_SUCCESS;
    }

    if (x == nullptr || out == nullptr)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    float row[boysymmetriad::kMaxBoysOrder + 1];

    for (int i = 0; i < count; ++i)
    {
        if (!ValidX(x[i]))
        {
            return BOYS_ERROR_INVALID_ARGUMENT;
        }

        boysymmetriad::BoysBatchF32(nmax, x[i], row);

        for (int k = 0; k <= nmax; ++k)
        {
            out[k * count + i] = row[k];
        }
    }

    return BOYS_SUCCESS;
}

} // extern "C"
