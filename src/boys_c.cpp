// Implementation of the C linkage surface (boys_c.h).
//
// The m = 1 entries route to the library's exported certified
// instantiations; the relaxed-rung entries (m > 1) instantiate the engine from
// the shipped headers at the rung vocabulary below — those call sites name
// their own multiplier, so their bodies are compiled here, exactly as the
// contract tests compile them.

#include "boys/boys_c.h"

#include "boys/boys.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

#include <iterator>

namespace {

template <double kAccuracyMultiplier> double BoysSingleRelaxed(int n, double x) noexcept {
    return boys::BoysSingle<kAccuracyMultiplier>(n, x);
}

template <double kAccuracyMultiplier> float BoysSingleF32Relaxed(int n, float x) noexcept {
    return boys::BoysSingleF32<kAccuracyMultiplier>(n, x);
}

// The library's accuracy-rung vocabulary: the multipliers this library names,
// ascending, and the rungs every run-time surface of it is named at. The
// accuracy contract's rungs are the seven the CPU tier lane's AccuracyTier
// names — m = 1, 64, 256, 1024, 4096, 16384 and 65536 — which the contract is
// published over and every CPU lane serves at run time; the device lane
// carries a finer-at-the-low-end sample set beside them, m = 2, 10, 100, 1e4
// and 1e8. The two overlap at m = 1 alone, which is every lane's default and
// full-accuracy rung, and their union is the twelve below — the same twelve the
// device lane publishes as kDeviceRungs (boys_cuda_options.hpp).
//
// Dispatch is by exact double equality: the set members are exactly
// representable, so callers pass the same literals.
constexpr double kRungMultipliers[] = {
    1.0, 2.0, 10.0, 64.0, 100.0, 256.0, 1024.0, 4096.0, 1e4, 16384.0, 65536.0, 1e8};

int FindMultiplier(double m) {
    for (int i = 0; i < static_cast<int>(std::size(kRungMultipliers)); ++i)
    {
        if (m == kRungMultipliers[i])
        {
            return i;
        }
    }

    return -1;
}

bool ValidOrder(int n) {
    return n >= 0 && n <= boys::kMaxBoysOrder;
}

bool ValidX(double x) {
    return x >= 0.0; // rejects negatives and NaN
}

// The per-argument top order is validated for the whole batch before any output
// is written, so a batch this entry rejects leaves the caller's buffer as it
// found it. Validation is this surface's own job: the C++ entry is a total
// function whose precondition it cannot check.
int RunBatchDoubleAtOrders(const int* n, int count, const double* x, double* out) {
    if (count < 0)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    if (count == 0)
    {
        return BOYS_SUCCESS;
    }

    if (n == nullptr || x == nullptr || out == nullptr)
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    for (int i = 0; i < count; ++i)
    {
        if (!ValidOrder(n[i]) || !ValidX(x[i]))
        {
            return BOYS_ERROR_INVALID_ARGUMENT;
        }
    }

    boys::BoysAllNAtOrders(n, x, out, static_cast<std::size_t>(count));
    return BOYS_SUCCESS;
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

    double row[boys::kMaxBoysOrder + 1];

    for (int i = 0; i < count; ++i)
    {
        if (!ValidX(x[i]))
        {
            return BOYS_ERROR_INVALID_ARGUMENT;
        }

        boys::BoysAllOrders(nmax, x[i], row);

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

    *out = boys::BoysSingle(n, x);
    return BOYS_SUCCESS;
}

int BoysFloat(int n, float x, float* out) {
    if (out == nullptr || !ValidOrder(n) || !ValidX(x))
    {
        return BOYS_ERROR_INVALID_ARGUMENT;
    }

    *out = boys::BoysSingleF32(n, x);
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
        *out = boys::BoysSingle(n, x);
        return BOYS_SUCCESS;
    case 1:
        *out = BoysSingleRelaxed<2.0>(n, x);
        return BOYS_SUCCESS;
    case 2:
        *out = BoysSingleRelaxed<10.0>(n, x);
        return BOYS_SUCCESS;
    case 3:
        *out = BoysSingleRelaxed<64.0>(n, x);
        return BOYS_SUCCESS;
    case 4:
        *out = BoysSingleRelaxed<100.0>(n, x);
        return BOYS_SUCCESS;
    case 5:
        *out = BoysSingleRelaxed<256.0>(n, x);
        return BOYS_SUCCESS;
    case 6:
        *out = BoysSingleRelaxed<1024.0>(n, x);
        return BOYS_SUCCESS;
    case 7:
        *out = BoysSingleRelaxed<4096.0>(n, x);
        return BOYS_SUCCESS;
    case 8:
        *out = BoysSingleRelaxed<1e4>(n, x);
        return BOYS_SUCCESS;
    case 9:
        *out = BoysSingleRelaxed<16384.0>(n, x);
        return BOYS_SUCCESS;
    case 10:
        *out = BoysSingleRelaxed<65536.0>(n, x);
        return BOYS_SUCCESS;
    case 11:
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
        *out = boys::BoysSingleF32(n, x);
        return BOYS_SUCCESS;
    case 1:
        *out = BoysSingleF32Relaxed<2.0>(n, x);
        return BOYS_SUCCESS;
    case 2:
        *out = BoysSingleF32Relaxed<10.0>(n, x);
        return BOYS_SUCCESS;
    case 3:
        *out = BoysSingleF32Relaxed<64.0>(n, x);
        return BOYS_SUCCESS;
    case 4:
        *out = BoysSingleF32Relaxed<100.0>(n, x);
        return BOYS_SUCCESS;
    case 5:
        *out = BoysSingleF32Relaxed<256.0>(n, x);
        return BOYS_SUCCESS;
    case 6:
        *out = BoysSingleF32Relaxed<1024.0>(n, x);
        return BOYS_SUCCESS;
    case 7:
        *out = BoysSingleF32Relaxed<4096.0>(n, x);
        return BOYS_SUCCESS;
    case 8:
        *out = BoysSingleF32Relaxed<1e4>(n, x);
        return BOYS_SUCCESS;
    case 9:
        *out = BoysSingleF32Relaxed<16384.0>(n, x);
        return BOYS_SUCCESS;
    case 10:
        *out = BoysSingleF32Relaxed<65536.0>(n, x);
        return BOYS_SUCCESS;
    case 11:
        *out = BoysSingleF32Relaxed<1e8>(n, x);
        return BOYS_SUCCESS;
    default:
        return BOYS_ERROR_UNSUPPORTED_MULTIPLIER;
    }
}

int BoysDoubleBatch(int nmax, int count, const double* x, double* out) {
    return RunBatchDouble(nmax, count, x, out);
}

int BoysDoubleBatchAtOrders(const int* n, int count, const double* x, double* out) {
    return RunBatchDoubleAtOrders(n, count, x, out);
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

    for (int i = 0; i < count; ++i)
    {
        if (!ValidX(x[i]))
        {
            return BOYS_ERROR_INVALID_ARGUMENT;
        }
    }

    boys::BoysAllNF32(nmax, x, out, static_cast<std::size_t>(count));
    return BOYS_SUCCESS;
}

} // extern "C"
