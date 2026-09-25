// The consumer side of the device-callable Boys entries.
//
// Every kernel here is shaped like a fused integral kernel rather than like a
// batch evaluator: it forms x in a register from a density factor and a squared
// distance, and it asks the entries for the orders it needs at that x, in the
// same kernel, writing each value straight into that thread's own output block.
// Nothing in this file touches the library's implementation: it includes the
// public device header and the CUDA runtime, so it is what a consumer's
// translation unit would be. That is the property the gate uses it for - a
// host-side wrapper around a batch launcher would prove the launcher works, and
// would prove nothing about a consumer's own kernel reaching the arithmetic.
//
// The argument is formed as rho * d2 with rho a power of two and d2 the
// argument divided by it, which is exact. The gate picks the pair that way so
// that the x a thread forms is the argument the committed reference grid holds,
// and every value returned from inside these kernels is comparable with that
// grid cell for cell. A general density factor would only move which argument
// is measured; the arithmetic under test is the same.
//
// There is one kernel per public entry: three precisions, and per precision the
// runtime-top-order ladder, the compile-time-top-order ladder, the
// one-value-at-a-time sink and the single order. The ladder kernels keep the
// whole family in a per-thread array, which is the shape that costs registers;
// the sink kernel writes each order as it arrives and keeps none, which is the
// shape for a caller that contracts as the orders come.
//
// A family entry writes kMaxBoysOrder + 1 values per element - the whole
// family, in the element's own block - and a single entry writes one value per
// element. A family kernel writes slot l of the block for every l up to the
// element's own top order, so what a block holds is the value the entry
// returned for its order, whichever of the four shapes produced it.

#include "boys/boys_cuda_device.hpp"
#include "boys_cuda_device_demo.hpp"

#include <cuda_runtime.h>
#include <stddef.h>

// The fp16 kernels receive the library's F16 through a void* the gate fills:
// the two are the same binary16 format, and this is what says so at compile
// time rather than at the first surprising value.
#if BoysFp16
#include "boys/f16.hpp"

static_assert(sizeof(boys::F16) == sizeof(__half), "F16 must be one binary16 value");
#endif

namespace {

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + 255) / 256);
}

__device__ __forceinline__ std::size_t BlockStart(std::size_t i) {
    return i * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);
}

__device__ __forceinline__ std::size_t Thread() {
    return static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
}

} // namespace

// ===========================================================================
// double
// ===========================================================================

// The runtime top order: the entry is handed the order the caller's quartet
// needs and the capacity of the caller's own array.
__global__ void BoysDeviceDemoLadder64Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    double* out,
    std::size_t count,
    int capacity,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double x = rho[i] * d2[i];
    double ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllOrdersF64(tables, order, x, ladder, capacity);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= order; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoLadder64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status) {
    BoysDeviceDemoLadder64Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, capacity,
                                                        status);
    return static_cast<int>(cudaGetLastError());
}

// The top order fixed at the call site, where it is one kernel argument fewer
// and the recursion loops have a bound the compiler can unroll.
__global__ void BoysDeviceDemoAllN64Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const double* rho,
    const double* d2,
    double* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const double x = rho[i] * d2[i];
    double ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllNF64<boys::kMaxBoysOrder>(tables, x, ladder);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= boys::kMaxBoysOrder; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoAllN64(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoAllN64Kernel<<<Blocks(count), 256>>>(*tables, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

// The sink shape: the values are consumed as they arrive, so no ladder is live
// at any point and the register cost does not grow with the top order.
__global__ void BoysDeviceDemoEach64Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    double* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const double x = rho[i] * d2[i];

    const boys::BoysDeviceStatus got = boys::BoysDeviceEachOrderF64(
        tables, order, x, [&](int l, double v) {
            out[BlockStart(i) + static_cast<std::size_t>(l)] = v;
        });
    status[i] = static_cast<int>(got);
}

extern "C" int BoysDeviceDemoEach64(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    double* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoEach64Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

// One order at one argument: the shape for a kernel whose order is known
// per call and which needs a single F_n.
__global__ void BoysDeviceDemoSingle64Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    double* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const double x = rho[i] * d2[i];
    double value = 0.0;

    const boys::BoysDeviceStatus got = boys::BoysDeviceSingleF64(tables, n[i], x, &value);
    status[i] = static_cast<int>(got);

    if (got == boys::BoysDeviceStatus::kSuccess)
    {
        out[i] = value;
    }
}

extern "C" int BoysDeviceDemoSingle64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      double* out,
                                      std::size_t count,
                                      int* status) {
    BoysDeviceDemoSingle64Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

// ===========================================================================
// float
// ===========================================================================

__global__ void BoysDeviceDemoLadder32Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    float* out,
    std::size_t count,
    int capacity,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float x = static_cast<float>(rho[i] * d2[i]);
    float ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllOrdersF32(tables, order, x, ladder, capacity);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= order; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoLadder32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status) {
    BoysDeviceDemoLadder32Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, capacity,
                                                        status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoAllN32Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const double* rho,
    const double* d2,
    float* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const float x = static_cast<float>(rho[i] * d2[i]);
    float ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllNF32<boys::kMaxBoysOrder>(tables, x, ladder);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= boys::kMaxBoysOrder; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoAllN32(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoAllN32Kernel<<<Blocks(count), 256>>>(*tables, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoEach32Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    float* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const float x = static_cast<float>(rho[i] * d2[i]);

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceEachOrderF32(tables, order, x, [&](int l, float v) {
            out[BlockStart(i) + static_cast<std::size_t>(l)] = v;
        });
    status[i] = static_cast<int>(got);
}

extern "C" int BoysDeviceDemoEach32(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    float* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoEach32Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoSingle32Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    float* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const float x = static_cast<float>(rho[i] * d2[i]);
    float value = 0.0f;

    const boys::BoysDeviceStatus got = boys::BoysDeviceSingleF32(tables, n[i], x, &value);
    status[i] = static_cast<int>(got);

    if (got == boys::BoysDeviceStatus::kSuccess)
    {
        out[i] = value;
    }
}

extern "C" int BoysDeviceDemoSingle32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      float* out,
                                      std::size_t count,
                                      int* status) {
    BoysDeviceDemoSingle32Kernel<<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

// The same entry with the lane's other region-B exponential. The two kernels are
// one kernel with one template argument different, which is what the option is:
// a caller that wants it names it at the call site, and the exponential it did
// not pick is not in the binary.
template <boys::RegionBExp kExp>
__global__ void BoysDeviceDemoSingle32ExpKernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    float* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const float x = static_cast<float>(rho[i] * d2[i]);
    float value = 0.0f;

    const boys::BoysDeviceStatus got = boys::BoysDeviceSingleF32<kExp>(tables, n[i], x, &value);
    status[i] = static_cast<int>(got);

    if (got == boys::BoysDeviceStatus::kSuccess)
    {
        out[i] = value;
    }
}

extern "C" int BoysDeviceDemoSingle32Fast(const boys::BoysDeviceTables* tables,
                                          const int* n,
                                          const double* rho,
                                          const double* d2,
                                          float* out,
                                          std::size_t count,
                                          int* status) {
    BoysDeviceDemoSingle32ExpKernel<boys::RegionBExp::kFast>
        <<<Blocks(count), 256>>>(*tables, n, rho, d2, out, count, status);
    return static_cast<int>(cudaGetLastError());
}

// ===========================================================================
// fp16, rounded at the boundary of the f32 engine
// ===========================================================================
#if BoysFp16
__global__ void BoysDeviceDemoLadder16Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    __half* out,
    std::size_t count,
    int capacity,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const __half x = __float2half(static_cast<float>(rho[i] * d2[i]));
    __half ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllOrdersF16(tables, order, x, ladder, capacity);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= order; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoLadder16(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      void* out,
                                      std::size_t count,
                                      int capacity,
                                      int* status) {
    BoysDeviceDemoLadder16Kernel<<<Blocks(count), 256>>>(
        *tables, n, rho, d2, static_cast<__half*>(out), count, capacity, status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoAllN16Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const double* rho,
    const double* d2,
    __half* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const __half x = __float2half(static_cast<float>(rho[i] * d2[i]));
    __half ladder[boys::kMaxBoysOrder + 1];

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceAllNF16<boys::kMaxBoysOrder>(tables, x, ladder);
    status[i] = static_cast<int>(got);

    if (got != boys::BoysDeviceStatus::kSuccess)
    {
        return;
    }

    for (int l = 0; l <= boys::kMaxBoysOrder; ++l)
    {
        out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];
    }
}

extern "C" int BoysDeviceDemoAllN16(const boys::BoysDeviceTables* tables,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoAllN16Kernel<<<Blocks(count), 256>>>(
        *tables, rho, d2, static_cast<__half*>(out), count, status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoEach16Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    __half* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const __half x = __float2half(static_cast<float>(rho[i] * d2[i]));

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceEachOrderF16(tables, order, x, [&](int l, __half v) {
            out[BlockStart(i) + static_cast<std::size_t>(l)] = v;
        });
    status[i] = static_cast<int>(got);
}

extern "C" int BoysDeviceDemoEach16(const boys::BoysDeviceTables* tables,
                                    const int* n,
                                    const double* rho,
                                    const double* d2,
                                    void* out,
                                    std::size_t count,
                                    int* status) {
    BoysDeviceDemoEach16Kernel<<<Blocks(count), 256>>>(
        *tables, n, rho, d2, static_cast<__half*>(out), count, status);
    return static_cast<int>(cudaGetLastError());
}

__global__ void BoysDeviceDemoSingle16Kernel(
    __grid_constant__ const boys::BoysDeviceTables tables,
    const int* n,
    const double* rho,
    const double* d2,
    __half* out,
    std::size_t count,
    int* status) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const __half x = __float2half(static_cast<float>(rho[i] * d2[i]));
    __half value = __float2half(0.0f);

    const boys::BoysDeviceStatus got = boys::BoysDeviceSingleF16(tables, n[i], x, &value);
    status[i] = static_cast<int>(got);

    if (got == boys::BoysDeviceStatus::kSuccess)
    {
        out[i] = value;
    }
}

extern "C" int BoysDeviceDemoSingle16(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* rho,
                                      const double* d2,
                                      void* out,
                                      std::size_t count,
                                      int* status) {
    BoysDeviceDemoSingle16Kernel<<<Blocks(count), 256>>>(
        *tables, n, rho, d2, static_cast<__half*>(out), count, status);
    return static_cast<int>(cudaGetLastError());
}
#endif // BoysFp16

// ===========================================================================
// the status values, so the gate compares against the entries' own enum
// ===========================================================================
extern "C" int BoysDeviceDemoStatusSuccess() {
    return static_cast<int>(boys::BoysDeviceStatus::kSuccess);
}

extern "C" int BoysDeviceDemoStatusOrderOutOfRange() {
    return static_cast<int>(boys::BoysDeviceStatus::kOrderOutOfRange);
}

extern "C" int BoysDeviceDemoStatusCapacityTooSmall() {
    return static_cast<int>(boys::BoysDeviceStatus::kCapacityTooSmall);
}
