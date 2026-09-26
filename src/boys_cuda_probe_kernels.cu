// Device side of the option probe: the timed regions, the kernels of this
// library's entries are launched into, and the subtraction kernels the
// device-callable entries are measured through.
//
// The host side (boys_cuda_probe.cpp) owns the protocol - which entry, how many
// passes and rounds, how a spread folds into a figure. This file owns the clock:
// a timed region here opens with a CUDA event, queues the launches back to back
// into buffers that are already resident, closes with a second event and
// synchronises on it, so what the elapsed time reports is the device timeline
// rather than the host's submission. C++20 with a CUDA-safe include list only,
// as the lane's own .cu is: the library's C++23 headers would poison this
// translation unit.
//
// The device-callable entries are measured as a difference, not as a launch. A
// caller-shaped kernel forms x per thread and calls the entry; the same kernel
// with the call removed writes the same slots with the same traffic. The host
// subtracts one bracket from the other, which leaves the arithmetic with the
// launch, the indexing and the memory passes already cancelled.

#include "boys_cuda_probe_entries.hpp"

#include "boys/boys_cuda_device.hpp"

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// The toolkit and the target architectures, stated by the report. CMake passes
// both; the fallbacks are what a translation unit compiled outside this tree
// would say rather than a guess at a version.
#if defined(BOYS_CUDA_TOOLKIT_TEXT) && defined(BOYS_CUDA_ARCHITECTURES_TEXT)
#define BOYS_PROBE_TOOLKIT BOYS_CUDA_TOOLKIT_TEXT
#define BOYS_PROBE_ARCHITECTURES BOYS_CUDA_ARCHITECTURES_TEXT
#else
#define BOYS_PROBE_TOOLKIT "unknown"
#define BOYS_PROBE_ARCHITECTURES "unknown"
#endif

namespace boys {
namespace probe_detail {
namespace {

// ---------------------------------------------------------------------------
// Launch geometry. One thread per argument, as a consumer's kernel would be,
// and the same geometry for every entry so that what is compared is the entry.
// ---------------------------------------------------------------------------
constexpr int kThreadsPerBlock = 256;

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

__device__ __forceinline__ std::size_t Thread() {
    return static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
}

// One argument's own block of the output, wide enough for the tallest ladder
// any entry writes.
__device__ __forceinline__ std::size_t BlockStart(std::size_t i) {
    return i * (static_cast<std::size_t>(kMaxBoysOrder) + 1);
}

// ---------------------------------------------------------------------------
// The instrument, and the floor.
// ---------------------------------------------------------------------------

// The canary: fixed work, no floating point, no memory. It holds no
// floating-point state on purpose, so the arithmetic this probe ranks cannot
// change the cost of the instrument that judges the run.
__global__ void CanaryKernel(unsigned long long seed, unsigned long long* sink) {
    unsigned long long s = seed + static_cast<unsigned long long>(Thread());

#pragma unroll 1
    for (int i = 0; i < kProbeCanaryRounds; ++i)
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
    }

    // This is never taken for the seeds this probe uses, and it is not provably
    // so, which is the point: the chain cannot be dropped as dead.
    if (s == 0x9E3779B97F4A7C15ull)
    {
        *sink = s;
    }
}

// The floor: a kernel launched exactly as the entries are - same grid, same
// repetition count, same event pair - that does no work at all. Its device time
// per launch is the part of every launched figure that is getting a kernel
// started rather than evaluating anything: the host's submission of the launch,
// the runtime's own launch, and the grid's ramp. It is measured rather than
// assumed small.
//
// It deliberately keeps no memory traffic, because the traffic is what the
// subtraction kernel's own baseline half is for: mixing the two here would
// measure the memory of a kernel that is not the caller's.
__global__ void FloorKernel(int* touched) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && touched != nullptr)
    {
        *touched = 1;
    }
}

// ---------------------------------------------------------------------------
// The subtraction kernel.
//
// One body per precision, with the four shapes written out at their own call
// sites. A run-time branch between the precisions would compile every lane's
// arithmetic into every kernel and measure the wrong one; a body per precision
// keeps each entry's arithmetic where it belongs, in the caller's kernel, which
// is the shape the device-callable entries exist for.
// ---------------------------------------------------------------------------

/// The double lane's four entries, and the cheap stand-in the removed-call half
/// writes in their place.
struct Dev64 {
    using Value = double;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              double x,
                                                              double* out) {
        return BoysDeviceSingleF64(tables, order, x, out);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 double* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF64(tables, order, x, out, capacity);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            double x,
                                                            double* out) {
        return BoysDeviceAllNF64<kProbeInKernelTopOrder>(tables, x, out);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF64(tables, order, x, sink);
    }

    static __device__ __forceinline__ double Cheap(double x, int l) {
        return x * static_cast<double>(l + 1);
    }
};

/// The float lane; the region-A seed is double, as the entry documents.
///
/// The region-B exponential is the entry's one axis, and it is a template
/// argument here because that is how the library offers it: two members, two
/// bounds, the same tables. The parameter carries the member; every other
/// method of this body is the same arithmetic for both.
template <bool kFastExp>
struct Dev32T {
    using Value = float;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              float x,
                                                              float* out) {
        return BoysDeviceSingleF32<kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(tables,
                                                                                        order,
                                                                                        x,
                                                                                        out);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 float* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF32(tables, order, x, out, capacity);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            float x,
                                                            float* out) {
        return BoysDeviceAllNF32<kProbeInKernelTopOrder>(tables, x, out);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF32(tables, order, x, sink);
    }

    static __device__ __forceinline__ float Cheap(float x, int l) {
        return x * static_cast<float>(l + 1);
    }
};

using Dev32 = Dev32T<false>;
using Dev32Fast = Dev32T<true>;

/// The fp16 lane, which rounds at the boundary and runs the fp32 engine between.
struct Dev16 {
    using Value = __half;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __half x,
                                                              __half* out) {
        return BoysDeviceSingleF16(tables, order, x, out);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 __half* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF16(tables, order, x, out, capacity);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            __half x,
                                                            __half* out) {
        return BoysDeviceAllNF16<kProbeInKernelTopOrder>(tables, x, out);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF16(tables, order, x, sink);
    }

    static __device__ __forceinline__ __half Cheap(__half x, int l) {
        return __float2half(__half2float(x) * static_cast<float>(l + 1));
    }
};

/// The four shapes, as a template parameter. Every shape writes its whole
/// output: the single order one value, a ladder the values of the ladder. The
/// removed-call half writes the same slots with the same traffic, so the
/// subtraction cancels the stores and leaves the arithmetic - and, with it, the
/// register cost the entry imposes on the caller's kernel, which is a real part
/// of what a fused entry costs.
enum class Shape : int {
    kSingle = 0,
    kAllOrders,
    kAllN,
    kEachOrder,
};

template <typename Dev, Shape kShape, bool kWithBoys>
__global__ void InKernelKernel(__grid_constant__ const BoysDeviceTables tables,
                               const int* n,
                               const typename Dev::Value* x,
                               typename Dev::Value* out,
                               std::size_t count) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    const int order = n[i];
    const typename Dev::Value arg = x[i];

    if constexpr (kShape == Shape::kSingle)
    {
        typename Dev::Value value = Dev::Cheap(arg, 0);

        if constexpr (kWithBoys)
        {
            (void)Dev::Single(tables, order, arg, &value);
        }

        out[i] = value;
    } else if constexpr (kShape == Shape::kAllN)
    {
        typename Dev::Value ladder[kMaxBoysOrder + 1];
        const std::size_t base = BlockStart(i);

        if constexpr (kWithBoys)
        {
            (void)Dev::AllN(tables, arg, ladder);
        } else
        {
            for (int l = 0; l <= kProbeInKernelTopOrder; ++l)
            {
                ladder[l] = Dev::Cheap(arg, l);
            }
        }

        for (int l = 0; l <= kProbeInKernelTopOrder; ++l)
        {
            out[base + static_cast<std::size_t>(l)] = ladder[l];
        }
    } else
    {
        // The ladder to this argument's own order, produced either in one call
        // or one order at a time into a sink.
        typename Dev::Value ladder[kMaxBoysOrder + 1];
        const std::size_t base = BlockStart(i);

        if constexpr (kWithBoys)
        {
            if constexpr (kShape == Shape::kAllOrders)
            {
                (void)Dev::AllOrders(tables, order, arg, ladder, kMaxBoysOrder + 1);
            } else
            {
                (void)Dev::EachOrder(tables, order, arg, [&](int l, typename Dev::Value v) {
                    ladder[l] = v;
                });
            }
        } else
        {
            for (int l = 0; l <= order; ++l)
            {
                ladder[l] = Dev::Cheap(arg, l);
            }
        }

        for (int l = 0; l <= order; ++l)
        {
            out[base + static_cast<std::size_t>(l)] = ladder[l];
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch.
// ---------------------------------------------------------------------------

/// Queues one launch of the caller-shaped kernel for one entry, with or without
/// the Boys call in it.
int LaunchInKernel(ProbeEntry entry,
                   const BoysDeviceTables* handle,
                   const int* n,
                   const double* xd,
                   const float* xf,
                   const __half* xh,
                   void* out,
                   std::size_t count,
                   bool withBoys,
                   cudaStream_t stream) {
    const unsigned int blocks = Blocks(count);

#define BOYS_PROBE_LAUNCH(dev, value_type, shape, arg)                                       \
    InKernelKernel<dev, Shape::shape, true><<<blocks, kThreadsPerBlock, 0, stream>>>(         \
        *handle, n, static_cast<const value_type*>(arg), static_cast<value_type*>(out), count)

#define BOYS_PROBE_LAUNCH_PLAIN(dev, value_type, shape, arg)                                 \
    InKernelKernel<dev, Shape::shape, false><<<blocks, kThreadsPerBlock, 0, stream>>>(        \
        *handle, n, static_cast<const value_type*>(arg), static_cast<value_type*>(out), count)

    if (withBoys)
    {
        switch (entry)
        {
            case ProbeEntry::kDeviceSingleF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kSingle, xd);
                break;
            case ProbeEntry::kDeviceSingleF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kSingle, xf);
                break;
            case ProbeEntry::kDeviceSingleF32Fast:
                BOYS_PROBE_LAUNCH(Dev32Fast, float, kSingle, xf);
                break;
            case ProbeEntry::kDeviceSingleF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kSingle, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceAllNF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceAllNF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kEachOrder, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kEachOrder, xh);
                break;
            default:
                return 1;
        }
    } else
    {
        switch (entry)
        {
            case ProbeEntry::kDeviceSingleF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kSingle, xd);
                break;
            case ProbeEntry::kDeviceSingleF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kSingle, xf);
                break;
            case ProbeEntry::kDeviceSingleF32Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Fast, float, kSingle, xf);
                break;
            case ProbeEntry::kDeviceSingleF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kSingle, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceAllNF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceAllNF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kEachOrder, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kEachOrder, xh);
                break;
            default:
                return 1;
        }
    }

#undef BOYS_PROBE_LAUNCH
#undef BOYS_PROBE_LAUNCH_PLAIN

    return 0;
}

// The launched entries are this library's own kernels, reached through the
// exported launchers boys_cuda.cu defines and boys_cuda.cpp wraps. The wrapper
// is host work and cannot appear on the device timeline, so the launcher is what
// is timed here.
extern "C" {
int BoysCudaLaunchSingleF32(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF32Fast(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllNF32(int, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF64(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF64(int, const double*, double*, std::size_t, void*);
int BoysCudaLaunchSingleF16(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16(int, const void*, void*, std::size_t, void*);
}

int LaunchLaunched(ProbeEntry entry,
                   const int* n,
                   const double* x,
                   const void* xh,
                   void* out,
                   std::size_t count,
                   int nmax,
                   cudaStream_t stream) {
    switch (entry)
    {
        case ProbeEntry::kSingleF64:
            BoysCudaLaunchSingleF64(n, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kSingleF32:
            BoysCudaLaunchSingleF32(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kSingleF32Fast:
            BoysCudaLaunchSingleF32Fast(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kSingleF16:
            BoysCudaLaunchSingleF16(n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF64:
            BoysCudaLaunchAllOrdersF64(n, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32:
            BoysCudaLaunchAllOrdersF32(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF16:
            BoysCudaLaunchAllOrdersF16(n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllNF64:
            BoysCudaLaunchAllNF64(nmax, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllNF32:
            BoysCudaLaunchAllNF32(nmax, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllNF16:
            BoysCudaLaunchAllNF16(nmax, xh, out, count, stream);
            break;
        default:
            return 1;
    }

    return 0;
}

// ---------------------------------------------------------------------------
// The timed region itself.
// ---------------------------------------------------------------------------

/// Queues one launch and returns a status code. The request is the context.
using RunOne = int (*)(const ProbeTimeRequest&, cudaStream_t);

int RunCanary(const ProbeTimeRequest& request, cudaStream_t stream) {
    CanaryKernel<<<kProbeCanaryBlocks, kProbeCanaryThreads, 0, stream>>>(
        0x243F6A8885A308D3ull, request.canarySink);
    return 0;
}

int RunFloor(const ProbeTimeRequest& request, cudaStream_t stream) {
    // The same grid the entries are launched on, so the ramp is the same shape.
    FloorKernel<<<Blocks(static_cast<std::size_t>(request.count)), kThreadsPerBlock, 0, stream>>>(
        request.floorSink);
    return 0;
}

int RunLaunched(const ProbeTimeRequest& request, cudaStream_t stream) {
    return LaunchLaunched(static_cast<ProbeEntry>(request.entry),
                          request.n,
                          request.x,
                          request.xh,
                          request.out,
                          static_cast<std::size_t>(request.count),
                          request.nmax,
                          stream);
}

int RunInKernel(const ProbeTimeRequest& request, cudaStream_t stream) {
    if (request.handle == nullptr)
    {
        // The tables are passed by value into the kernel, so a null handle is a
        // host-side dereference rather than a launch that fails: refuse it here
        // instead of reading through it.
        return 1;
    }

    return LaunchInKernel(static_cast<ProbeEntry>(request.entry),
                          static_cast<const BoysDeviceTables*>(request.handle),
                          request.n,
                          request.x,
                          request.xf,
                          static_cast<const __half*>(request.xh),
                          request.out,
                          static_cast<std::size_t>(request.count),
                          request.withBoys != 0,
                          stream);
}

/// Runs \p reps launches between two events and reports the device
/// milliseconds between them. The launches are queued back to back into buffers
/// the caller has already filled, so the host's submission cost is hidden by
/// the queue and what the elapsed time contains is the device's own work.
int TimeRegion(RunOne run, const ProbeTimeRequest& request, double* outMs) {
    cudaStream_t stream = nullptr;

    if (outMs == nullptr || request.reps <= 0)
    {
        // A region that would launch nothing measures nothing, and reporting it
        // as zero device time would read as an infinitely fast entry.
        return 1;
    }

    if (cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess)
    {
        return 2;
    }

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess)
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaStreamDestroy(stream);
        return 2;
    }

    int code = 0;

    if (cudaEventRecord(start, stream) != cudaSuccess)
    {
        code = 2;
    }

    for (int r = 0; r < request.reps && code == 0; ++r)
    {
        code = run(request, stream);
    }

    if (code == 0 && cudaEventRecord(stop, stream) != cudaSuccess)
    {
        code = 2;
    }

    if (code == 0 && cudaEventSynchronize(stop) != cudaSuccess)
    {
        code = 2;
    }

    if (code == 0)
    {
        // A launch that failed asynchronously surfaces here rather than being
        // read as a very fast entry.
        code = cudaGetLastError() == cudaSuccess ? 0 : 2;
    }

    if (code == 0)
    {
        // The runtime reports event elapsed time in single precision
        // milliseconds; the caller works in double, so the widening happens
        // here and nowhere else.
        float elapsed = 0.0f;

        if (cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess)
        {
            *outMs = static_cast<double>(elapsed);
        } else
        {
            code = 2;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    return code;
}

} // namespace
} // namespace probe_detail
} // namespace boys

// ---------------------------------------------------------------------------
// The boundary boys_cuda_probe.cpp calls. Return codes as the lane's own
// exported functions: 0 success, 1 an internal error, 2 a CUDA failure.
// ---------------------------------------------------------------------------
using boys::probe_detail::ProbeBuildFacts;
using boys::probe_detail::ProbeDeviceFacts;
using boys::probe_detail::ProbeTimeRequest;
using boys::probe_detail::ProbeWhat;
using boys::probe_detail::RunCanary;
using boys::probe_detail::RunFloor;
using boys::probe_detail::RunInKernel;
using boys::probe_detail::RunLaunched;
using boys::probe_detail::TimeRegion;

extern "C" {

int BoysCudaProbeDeviceCount(int* out) {
    if (out == nullptr)
    {
        return 1;
    }

    return cudaGetDeviceCount(out) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeSetDevice(int ordinal) {
    return cudaSetDevice(ordinal) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeCurrentDevice(int* out) {
    if (out == nullptr)
    {
        return 1;
    }

    return cudaGetDevice(out) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeDeviceQuery(int ordinal, ProbeDeviceFacts* out) {
    if (out == nullptr)
    {
        return 1;
    }

    cudaDeviceProp prop{};

    if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess)
    {
        return 2;
    }

    for (int i = 0; i < 256; ++i)
    {
        out->name[i] = prop.name[i];
    }

    out->computeMajor = prop.major;
    out->computeMinor = prop.minor;
    out->totalMemoryBytes = static_cast<unsigned long long>(prop.totalGlobalMem);
    out->multiProcessorCount = prop.multiProcessorCount;
    out->clockKHz = 0;
    out->memoryClockKHz = 0;
    out->singleToDoublePrecisionPerfRatio = 0;

    // Read as attributes rather than off the property struct: the property
    // fields for these three are gone from the runtime, and the attributes
    // survive. Zero means the runtime declined to say, which the report prints
    // rather than filling in from what this card is expected to be.
    int value = 0;

    if (cudaDeviceGetAttribute(&value, cudaDevAttrClockRate, ordinal) == cudaSuccess)
    {
        out->clockKHz = value;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrMemoryClockRate, ordinal) == cudaSuccess)
    {
        out->memoryClockKHz = value;
    }

    if (cudaDeviceGetAttribute(&value, cudaDevAttrSingleToDoublePrecisionPerfRatio, ordinal) ==
        cudaSuccess)
    {
        out->singleToDoublePrecisionPerfRatio = value;
    }

    out->driverVersion = 0;
    out->runtimeVersion = 0;

    int driver = 0;
    int runtime = 0;

    if (cudaDriverGetVersion(&driver) == cudaSuccess)
    {
        out->driverVersion = driver;
    }

    if (cudaRuntimeGetVersion(&runtime) == cudaSuccess)
    {
        out->runtimeVersion = runtime;
    }

    return 0;
}

int BoysCudaProbeBuildFacts(ProbeBuildFacts* out) {
    if (out == nullptr)
    {
        return 1;
    }

    const char* const toolkit = BOYS_PROBE_TOOLKIT;
    const char* const architectures = BOYS_PROBE_ARCHITECTURES;

    int i = 0;

    for (; toolkit[i] != '\0' && i < 63; ++i)
    {
        out->toolkit[i] = toolkit[i];
    }

    out->toolkit[i] = '\0';

    for (i = 0; architectures[i] != '\0' && i < 127; ++i)
    {
        out->architectures[i] = architectures[i];
    }

    out->architectures[i] = '\0';
    return 0;
}

int BoysCudaProbeAlloc(void** out, std::size_t bytes) {
    if (out == nullptr || bytes == 0)
    {
        return 1;
    }

    return cudaMalloc(out, bytes) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeFree(void* p) {
    if (p == nullptr)
    {
        return 0;
    }

    return cudaFree(p) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeUpload(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeDownload(void* dst, const void* src, std::size_t bytes) {
    return cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeSynchronize() {
    return cudaDeviceSynchronize() == cudaSuccess ? 0 : 2;
}

int BoysCudaProbeTime(ProbeTimeRequest* request) {
    if (request == nullptr || request->outMs == nullptr || request->reps <= 0)
    {
        return 1;
    }

    switch (static_cast<ProbeWhat>(request->what))
    {
        case ProbeWhat::kCanary:
            return request->canarySink == nullptr ? 1
                                                  : TimeRegion(RunCanary, *request, request->outMs);
        case ProbeWhat::kFloor:
            return TimeRegion(RunFloor, *request, request->outMs);
        case ProbeWhat::kLaunchedEntry:
            return TimeRegion(RunLaunched, *request, request->outMs);
        default:
            return request->handle == nullptr
                       ? 1
                       : TimeRegion(RunInKernel, *request, request->outMs);
    }
}

} // extern "C"
