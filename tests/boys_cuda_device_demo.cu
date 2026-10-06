// The consumer side of the device-callable Boys entries: one kernel per public entry, each shaped
// like a fused integral kernel rather than like a batch evaluator. The file includes the public
// device header and the CUDA runtime and nothing of the library's implementation, so it is what a
// consumer's own translation unit is.

// A thread forms x as rho * d2, with rho a power of two so that the product is exact: the x it
// forms is therefore the argument the committed reference grid holds, and every value written back
// is comparable with that grid cell for cell.

// A family entry writes kMaxBoysOrder + 1 values per element — the element's own block — and a
// single entry writes one value per element. An entry that refuses a call records that refusal per
// element and writes nothing for it.

#include "boys/boys_cuda_device.hpp"
#include "boys_cuda_device_demo.hpp"

#include <cuda_runtime.h>
#include <stddef.h>

// The fp16 kernels receive the library's F16 through a void*: the two are the
// same binary16 format, and this is what says so at compile time.
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

// Runtime top order: the entry is handed the order the caller's quartet needs,
// and the capacity of the caller's own array.
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

// Top order fixed at the call site: one kernel argument fewer, and a bound the
// compiler can unroll.
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
        boys::BoysDeviceAllNF64<boys::kDefaultDivisionForm, boys::kMaxBoysOrder>(tables, x, ladder);
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

// The sink shape: each value is consumed as it arrives, so no ladder is live and
// the register cost does not grow with the top order.
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

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceEachOrderF64(tables, order, x, [&](int l, double v) {
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

// One order at one argument, for a kernel that knows its order per call and
// needs a single F_n.
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
        boys::BoysDeviceAllNF32<boys::kDefaultDivisionForm, boys::kMaxBoysOrder>(tables, x, ladder);
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

// The same entry with the lane's other region-B exponential, named at the call
// site: the exponential it did not pick is not in the binary.
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

    const boys::BoysDeviceStatus got =
        boys::BoysDeviceSingleF32<boys::kDefaultDivisionForm, kExp>(tables, n[i], x, &value);
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
        boys::BoysDeviceAllNF16<boys::kDefaultDivisionForm, boys::kMaxBoysOrder>(tables, x, ladder);
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
// the partition and route axes of the ladder shape
// ===========================================================================

// Sixteen entries of one shape, each the ladder BoysDeviceDemoLadder64 and BoysDeviceDemoLadder32
// already reach — a thread forms x from the handle's quartet, calls the entry and writes the
// element's own block back — over another partition, another route, or the other stored form of the
// one it is already on.

// The body is the body above and the entry is the only thing that moves, so it is written once and
// instantiated per entry rather than copied sixteen times: a copy that drifted would be a consumer
// kernel that measures something other than the entry it is named for. The two names a pair carries
// — the route's two scheme names, the grid's two stored forms — are two entries with two names.
#define BOYS_DEVICE_DEMO_LADDER(SUFFIX, VALUE, ENTRY)                                             \
    __global__ void BoysDeviceDemoLadder##SUFFIX##Kernel(                                         \
        __grid_constant__ const boys::BoysDeviceTables tables,                                    \
        const int* n,                                                                             \
        const double* rho,                                                                        \
        const double* d2,                                                                         \
        VALUE* out,                                                                               \
        std::size_t count,                                                                        \
        int capacity,                                                                             \
        int* status) {                                                                            \
        const std::size_t i = Thread();                                                           \
                                                                                                  \
        if (i >= count)                                                                           \
        {                                                                                         \
            return;                                                                               \
        }                                                                                         \
                                                                                                  \
        const int order = n[i];                                                                   \
        const double x = rho[i] * d2[i];                                                          \
        VALUE ladder[boys::kMaxBoysOrder + 1];                                                    \
                                                                                                  \
        const boys::BoysDeviceStatus got =                                                        \
            boys::ENTRY(tables, order, x, ladder, capacity);                                      \
        status[i] = static_cast<int>(got);                                                        \
                                                                                                  \
        if (got != boys::BoysDeviceStatus::kSuccess)                                              \
        {                                                                                         \
            return;                                                                               \
        }                                                                                         \
                                                                                                  \
        for (int l = 0; l <= order; ++l)                                                          \
        {                                                                                         \
            out[BlockStart(i) + static_cast<std::size_t>(l)] = ladder[l];                          \
        }                                                                                         \
    }                                                                                             \
                                                                                                  \
    extern "C" int BoysDeviceDemoLadder##SUFFIX(const boys::BoysDeviceTables* tables,             \
                                                const int* n,                                     \
                                                const double* rho,                                \
                                                const double* d2,                                 \
                                                VALUE* out,                                       \
                                                std::size_t count,                                \
                                                int capacity,                                     \
                                                int* status) {                                    \
        BoysDeviceDemoLadder##SUFFIX##Kernel<<<Blocks(count), 256>>>(                             \
            *tables, n, rho, d2, out, count, capacity, status);                                   \
        return static_cast<int>(cudaGetLastError());                                              \
    }

BOYS_DEVICE_DEMO_LADDER(64Narrow, double, BoysDeviceAllOrdersF64Narrow)
BOYS_DEVICE_DEMO_LADDER(64NarrowMono, double, BoysDeviceAllOrdersF64NarrowMono)
BOYS_DEVICE_DEMO_LADDER(64Rat, double, BoysDeviceAllOrdersF64Rat)
BOYS_DEVICE_DEMO_LADDER(64RatHorner, double, BoysDeviceAllOrdersF64RatHorner)
BOYS_DEVICE_DEMO_LADDER(64NarrowRat, double, BoysDeviceAllOrdersF64NarrowRat)
BOYS_DEVICE_DEMO_LADDER(64NarrowRatHorner, double, BoysDeviceAllOrdersF64NarrowRatHorner)
BOYS_DEVICE_DEMO_LADDER(64Uniform, double, BoysDeviceAllOrdersF64Uniform)
BOYS_DEVICE_DEMO_LADDER(64UniformHorner, double, BoysDeviceAllOrdersF64UniformHorner)
BOYS_DEVICE_DEMO_LADDER(64UniformRat, double, BoysDeviceAllOrdersF64UniformRat)
BOYS_DEVICE_DEMO_LADDER(64UniformRatHorner, double, BoysDeviceAllOrdersF64UniformRatHorner)

BOYS_DEVICE_DEMO_LADDER(32Narrow, float, BoysDeviceAllOrdersF32Narrow)
BOYS_DEVICE_DEMO_LADDER(32NarrowMono, float, BoysDeviceAllOrdersF32NarrowMono)
BOYS_DEVICE_DEMO_LADDER(32Rat, float, BoysDeviceAllOrdersF32Rat)
BOYS_DEVICE_DEMO_LADDER(32RatHorner, float, BoysDeviceAllOrdersF32RatHorner)
BOYS_DEVICE_DEMO_LADDER(32NarrowRat, float, BoysDeviceAllOrdersF32NarrowRat)
BOYS_DEVICE_DEMO_LADDER(32NarrowRatHorner, float, BoysDeviceAllOrdersF32NarrowRatHorner)
BOYS_DEVICE_DEMO_LADDER(32Uniform, float, BoysDeviceAllOrdersF32Uniform)
BOYS_DEVICE_DEMO_LADDER(32UniformHorner, float, BoysDeviceAllOrdersF32UniformHorner)
BOYS_DEVICE_DEMO_LADDER(32UniformRat, float, BoysDeviceAllOrdersF32UniformRat)
BOYS_DEVICE_DEMO_LADDER(32UniformRatHorner, float, BoysDeviceAllOrdersF32UniformRatHorner)

#undef BOYS_DEVICE_DEMO_LADDER

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
