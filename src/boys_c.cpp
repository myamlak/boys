// Implementation of the C linkage surface (boys_c.h).
//
// Every entry routes to the library's exported certified instantiations.

#include "boys/boys_c.h"

#include "boys/boys.hpp"
#include "boys/boys_effective_degrees.hpp"
#include "boys/boys_impl.hpp"

#include <iterator>

namespace {

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
