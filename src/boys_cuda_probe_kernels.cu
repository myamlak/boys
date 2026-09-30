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
// the runtime's own launch, and the grid's ramp.
//
// It keeps no memory traffic, because the traffic is what the subtraction
// kernel's own baseline half is for: mixing the two here would measure the
// memory of a kernel that is not the caller's.
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
// sites: a run-time branch between the precisions would compile every lane's
// arithmetic into every kernel and measure the wrong one.
// ---------------------------------------------------------------------------

/// The double lane's four entries, and the cheap stand-in the removed-call half
/// writes in their place.
struct Dev64 {
    using Value = double;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              double x,
                                                              double* out,
                                                              double multiplier) {
        return BoysDeviceSingleF64(tables, order, x, out, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 double* out,
                                                                 int capacity,
                                                                 double multiplier) {
        return BoysDeviceAllOrdersF64(tables, order, x, out, capacity, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            double x,
                                                            double* out,
                                                            double multiplier) {
        return BoysDeviceAllNF64<kProbeInKernelTopOrder>(tables, x, out, multiplier);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 Sink sink,
                                                                 double multiplier) {
        return BoysDeviceEachOrderF64(tables, order, x, sink, multiplier);
    }

    static __device__ __forceinline__ double Cheap(double x, int l) {
        return x * static_cast<double>(l + 1);
    }
};

/// The float lane; the region-A seed is double, as the entry documents.
///
/// The region-B exponential is the entry's one axis, and it is a template
/// argument here because that is how the library offers it: two members, two
/// bounds, the same tables.
template <bool kFastExp>
struct Dev32T {
    using Value = float;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              float x,
                                                              float* out,
                                                              double multiplier) {
        return BoysDeviceSingleF32<kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 float* out,
                                                                 int capacity,
                                                                 double multiplier) {
        return BoysDeviceAllOrdersF32(tables, order, x, out, capacity, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            float x,
                                                            float* out,
                                                            double multiplier) {
        return BoysDeviceAllNF32<kProbeInKernelTopOrder>(tables, x, out, multiplier);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 Sink sink,
                                                                 double multiplier) {
        return BoysDeviceEachOrderF32(tables, order, x, sink, multiplier);
    }

    static __device__ __forceinline__ float Cheap(float x, int l) {
        return x * static_cast<float>(l + 1);
    }
};

using Dev32 = Dev32T<false>;
using Dev32Fast = Dev32T<true>;

#if BoysFp16
/// The fp16 lane, which rounds at the boundary and runs the fp32 engine between.
///
/// It is behind the BoysFp16 seam that declares the entries it calls: a build
/// without them has no fp16 rows to measure.
struct Dev16 {
    using Value = __half;

    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __half x,
                                                              __half* out,
                                                              double multiplier) {
        return BoysDeviceSingleF16(tables, order, x, out, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 __half* out,
                                                                 int capacity,
                                                                 double multiplier) {
        return BoysDeviceAllOrdersF16(tables, order, x, out, capacity, multiplier);
    }

    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            __half x,
                                                            __half* out,
                                                            double multiplier) {
        return BoysDeviceAllNF16<kProbeInKernelTopOrder>(tables, x, out, multiplier);
    }

    template <typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 Sink sink,
                                                                 double multiplier) {
        return BoysDeviceEachOrderF16(tables, order, x, sink, multiplier);
    }

    static __device__ __forceinline__ __half Cheap(__half x, int l) {
        return __float2half(__half2float(x) * static_cast<float>(l + 1));
    }
};
#endif // BoysFp16

// The partition and route axes of the ladder shape, one policy per entry.
//
// Each is the all-orders ladder the policies above reach, over another
// partition, another route or the other stored form of the one it is already on;
// only that member is written, because that is the shape these entries have — a
// kernel instantiated at another shape with one of them does not compile, which
// is what says so.
//
// One policy per entry rather than one per axis, because the two names a pair
// carries are two entries and two rows of the space: measuring one of them would
// leave the other reported as a row this run carried no figure for.
#define BOYS_PROBE_LADDER_POLICY(NAME, VALUE, ENTRY) \
    struct NAME { \
        using Value = VALUE; \
 \
        static __device__ __forceinline__ BoysDeviceStatus AllOrders( \
            const BoysDeviceTables& tables, \
            int order, \
            VALUE arg, \
            VALUE* out, \
            int capacity, \
            double multiplier) { \
            return ENTRY(tables, order, arg, out, capacity, multiplier); \
        } \
 \
        static __device__ __forceinline__ VALUE Cheap(VALUE x, int l) { \
            return x * static_cast<VALUE>(l + 1); \
        } \
    };

BOYS_PROBE_LADDER_POLICY(Dev64Narrow, double, BoysDeviceAllOrdersF64Narrow)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowMono, double, BoysDeviceAllOrdersF64NarrowMono)
BOYS_PROBE_LADDER_POLICY(Dev64Rat, double, BoysDeviceAllOrdersF64Rat)
BOYS_PROBE_LADDER_POLICY(Dev64RatHorner, double, BoysDeviceAllOrdersF64RatHorner)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowRat, double, BoysDeviceAllOrdersF64NarrowRat)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowRatHorner, double, BoysDeviceAllOrdersF64NarrowRatHorner)
BOYS_PROBE_LADDER_POLICY(Dev64Uniform, double, BoysDeviceAllOrdersF64Uniform)
BOYS_PROBE_LADDER_POLICY(Dev64UniformHorner, double, BoysDeviceAllOrdersF64UniformHorner)
BOYS_PROBE_LADDER_POLICY(Dev64UniformRat, double, BoysDeviceAllOrdersF64UniformRat)
BOYS_PROBE_LADDER_POLICY(Dev64UniformRatHorner, double, BoysDeviceAllOrdersF64UniformRatHorner)

BOYS_PROBE_LADDER_POLICY(Dev32Narrow, float, BoysDeviceAllOrdersF32Narrow)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowMono, float, BoysDeviceAllOrdersF32NarrowMono)
BOYS_PROBE_LADDER_POLICY(Dev32Rat, float, BoysDeviceAllOrdersF32Rat)
BOYS_PROBE_LADDER_POLICY(Dev32RatHorner, float, BoysDeviceAllOrdersF32RatHorner)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowRat, float, BoysDeviceAllOrdersF32NarrowRat)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowRatHorner, float, BoysDeviceAllOrdersF32NarrowRatHorner)
BOYS_PROBE_LADDER_POLICY(Dev32Uniform, float, BoysDeviceAllOrdersF32Uniform)
BOYS_PROBE_LADDER_POLICY(Dev32UniformHorner, float, BoysDeviceAllOrdersF32UniformHorner)
BOYS_PROBE_LADDER_POLICY(Dev32UniformRat, float, BoysDeviceAllOrdersF32UniformRat)
BOYS_PROBE_LADDER_POLICY(Dev32UniformRatHorner, float, BoysDeviceAllOrdersF32UniformRatHorner)

#undef BOYS_PROBE_LADDER_POLICY

/// The four shapes, as a template parameter. Every shape writes its whole
/// output, and the removed-call half writes the same slots with the same
/// traffic, so the subtraction cancels the stores and leaves the arithmetic -
/// and, with it, the register cost the entry imposes on the caller's kernel,
/// which is a real part of what a fused entry costs.
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
                               std::size_t count,
                               double multiplier) {
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
            (void)Dev::Single(tables, order, arg, &value, multiplier);
        }

        out[i] = value;
    } else if constexpr (kShape == Shape::kAllN)
    {
        typename Dev::Value ladder[kMaxBoysOrder + 1];
        const std::size_t base = BlockStart(i);

        if constexpr (kWithBoys)
        {
            (void)Dev::AllN(tables, arg, ladder, multiplier);
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
        // The ladder to this argument's own order: one call, or one order at a
        // time into a sink.
        typename Dev::Value ladder[kMaxBoysOrder + 1];
        const std::size_t base = BlockStart(i);

        if constexpr (kWithBoys)
        {
            if constexpr (kShape == Shape::kAllOrders)
            {
                (void)Dev::AllOrders(tables, order, arg, ladder, kMaxBoysOrder + 1, multiplier);
            } else
            {
                (void)Dev::EachOrder(
                    tables,
                    order,
                    arg,
                    [&](int l, typename Dev::Value v) { ladder[l] = v; },
                    multiplier);
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
                   double multiplier,
                   cudaStream_t stream) {
    const unsigned int blocks = Blocks(count);

#define BOYS_PROBE_LAUNCH(dev, value_type, shape, arg)                                       \
    InKernelKernel<dev, Shape::shape, true><<<blocks, kThreadsPerBlock, 0, stream>>>(         \
        *handle, n, static_cast<const value_type*>(arg), static_cast<value_type*>(out), count, \
        multiplier)

#define BOYS_PROBE_LAUNCH_PLAIN(dev, value_type, shape, arg)                                 \
    InKernelKernel<dev, Shape::shape, false><<<blocks, kThreadsPerBlock, 0, stream>>>(        \
        *handle, n, static_cast<const value_type*>(arg), static_cast<value_type*>(out), count, \
        multiplier)

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
            case ProbeEntry::kDeviceAllOrdersF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllNF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceAllNF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF64:
                BOYS_PROBE_LAUNCH(Dev64, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF32:
                BOYS_PROBE_LAUNCH(Dev32, float, kEachOrder, xf);
                break;
#if BoysFp16
            // The fp16 arms, with the lane: the entries they name are behind
            // the same seam, and a build without them has no such enumerator
            // to be asked for.
            case ProbeEntry::kDeviceSingleF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kSingle, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF16:
                BOYS_PROBE_LAUNCH(Dev16, __half, kEachOrder, xh);
                break;
#else
            // A build with the seam closed has no fp16 enumerator to be asked
            // for and no half array to read, so this is the only arm that would
            // have touched the parameter.
            (void)xh;
#endif // BoysFp16
            // The partition and route axes of the ladder shape, one arm per
            // entry. Their argument array is the one their lane reads, as it is
            // above: the double lane's is xd and the float lane's is xf.
            case ProbeEntry::kDeviceAllOrdersF64Narrow:
                BOYS_PROBE_LAUNCH(Dev64Narrow, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowMono:
                BOYS_PROBE_LAUNCH(Dev64NarrowMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Rat:
                BOYS_PROBE_LAUNCH(Dev64Rat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatHorner:
                BOYS_PROBE_LAUNCH(Dev64RatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRat:
                BOYS_PROBE_LAUNCH(Dev64NarrowRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatHorner:
                BOYS_PROBE_LAUNCH(Dev64NarrowRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Uniform:
                BOYS_PROBE_LAUNCH(Dev64Uniform, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformHorner:
                BOYS_PROBE_LAUNCH(Dev64UniformHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformRat:
                BOYS_PROBE_LAUNCH(Dev64UniformRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformRatHorner:
                BOYS_PROBE_LAUNCH(Dev64UniformRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Narrow:
                BOYS_PROBE_LAUNCH(Dev32Narrow, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowMono:
                BOYS_PROBE_LAUNCH(Dev32NarrowMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Rat:
                BOYS_PROBE_LAUNCH(Dev32Rat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatHorner:
                BOYS_PROBE_LAUNCH(Dev32RatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRat:
                BOYS_PROBE_LAUNCH(Dev32NarrowRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatHorner:
                BOYS_PROBE_LAUNCH(Dev32NarrowRatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Uniform:
                BOYS_PROBE_LAUNCH(Dev32Uniform, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformHorner:
                BOYS_PROBE_LAUNCH(Dev32UniformHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformRat:
                BOYS_PROBE_LAUNCH(Dev32UniformRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformRatHorner:
                BOYS_PROBE_LAUNCH(Dev32UniformRatHorner, float, kAllOrders, xf);
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
            case ProbeEntry::kDeviceAllOrdersF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllNF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceAllNF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF64:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF32:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32, float, kEachOrder, xf);
                break;
#if BoysFp16
            // The removed-call half of the same four arms; see above.
            case ProbeEntry::kDeviceSingleF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kSingle, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF16:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16, __half, kEachOrder, xh);
                break;
#endif // BoysFp16
            // The removed-call half of the sixteen arms above: the same kernel
            // with the entry left out, which is the other side of the
            // subtraction.
            case ProbeEntry::kDeviceAllOrdersF64Narrow:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Narrow, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Rat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Rat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64RatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Uniform:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Uniform, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64UniformHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64UniformRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64UniformRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64UniformRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Narrow:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Narrow, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Rat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Rat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32RatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowRatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Uniform:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Uniform, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32UniformHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32UniformRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32UniformRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32UniformRatHorner, float, kAllOrders, xf);
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
int BoysCudaLaunchAllOrdersF32Narrow(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowMono(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Uniform(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32UniformRat(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32UniformHorner(
    const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Rat(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRat(const int*, const double*, float*, std::size_t, void*);
// The float lane's other packing axis, whose rows this file arms below. The
// grid's two orders rows need no launcher of their own: they launch the two
// declared above.
int BoysCudaLaunchAllOrdersF32Orders(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrders(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRat(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32RatEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRatEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRatEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRatEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersEff(
    const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowMonoEff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMonoEff(
    const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllNF32(int, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF64(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Narrow(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Orders(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrders(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Mono(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersMono(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowMono(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Rat(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersRat(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowRat(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Uniform(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64UniformRat(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64UniformHorner(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF64(int, const double*, double*, std::size_t, void*);
#if BoysFp16
int BoysCudaLaunchSingleF16(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16(int, const void*, void*, std::size_t, void*);
#endif // BoysFp16

// The same entries at a relaxed rung: each kernel reads the resident rung's
// effective degrees from the device instead of the degrees compiled into the
// full-accuracy kernel above, so a region launched at any multiplier but m = 1
// goes through this set. The pair is the one the host entries select between at
// the same multiplier (BoysCuda::SingleF64 and its siblings, which upload the
// rung's tables and then launch the Eff launcher).
int BoysCudaLaunchSingleF64Eff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Eff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowEff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersEff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64MonoEff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersMonoEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowMonoEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64RatEff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersRatEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowRatEff(const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff(
    const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF64Eff(int, const double*, double*, std::size_t, void*);
int BoysCudaLaunchSingleF32Eff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF32EffFast(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Eff(const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllNF32Eff(int, const double*, float*, std::size_t, void*);
#if BoysFp16
int BoysCudaLaunchSingleF16Eff(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Eff(const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16Eff(int, const void*, void*, std::size_t, void*);
#endif // BoysFp16
}

int LaunchLaunched(ProbeEntry entry,
                   const int* n,
                   const double* x,
                   const void* xh,
                   void* out,
                   std::size_t count,
                   int nmax,
                   double multiplier,
                   cudaStream_t stream) {
    // m = 1 is the entry's full-accuracy kernel and every other rung is the same
    // entry's effective-degree kernel, which reads the rung the host side made
    // resident. The host entry at the same multiplier launches exactly this
    // member of the same pair.
    const bool relaxed = multiplier != kBoysFullAccuracyMultiplier;

    switch (entry)
    {
        case ProbeEntry::kSingleF64:
            (relaxed ? BoysCudaLaunchSingleF64Eff
                     : BoysCudaLaunchSingleF64)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kSingleF32:
            (relaxed ? BoysCudaLaunchSingleF32Eff
                     : BoysCudaLaunchSingleF32)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kSingleF32Fast:
            (relaxed ? BoysCudaLaunchSingleF32EffFast
                     : BoysCudaLaunchSingleF32Fast)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64:
            (relaxed ? BoysCudaLaunchAllOrdersF64Eff
                     : BoysCudaLaunchAllOrdersF64)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64Narrow:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowEff
                     : BoysCudaLaunchAllOrdersF64Narrow)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64Orders:
            (relaxed ? BoysCudaLaunchAllOrdersF64OrdersEff
                     : BoysCudaLaunchAllOrdersF64Orders)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrders:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowOrdersEff
                     : BoysCudaLaunchAllOrdersF64NarrowOrders)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64Mono:
            (relaxed ? BoysCudaLaunchAllOrdersF64MonoEff
                     : BoysCudaLaunchAllOrdersF64Mono)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersMono:
            (relaxed ? BoysCudaLaunchAllOrdersF64OrdersMonoEff
                     : BoysCudaLaunchAllOrdersF64OrdersMono)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowMono:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowMonoEff
                     : BoysCudaLaunchAllOrdersF64NarrowMono)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersMono:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff
                     : BoysCudaLaunchAllOrdersF64NarrowOrdersMono)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        // The fit route's rows: the two scheme names of a shape select one
        // arithmetic, so both names launch the same kernel.
        case ProbeEntry::kAllOrdersF64Rat:
        case ProbeEntry::kAllOrdersF64RatHorner:
            (relaxed ? BoysCudaLaunchAllOrdersF64RatEff
                     : BoysCudaLaunchAllOrdersF64Rat)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersRat:
        case ProbeEntry::kAllOrdersF64OrdersRatHorner:
            (relaxed ? BoysCudaLaunchAllOrdersF64OrdersRatEff
                     : BoysCudaLaunchAllOrdersF64OrdersRat)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowRat:
        case ProbeEntry::kAllOrdersF64NarrowRatHorner:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowRatEff
                     : BoysCudaLaunchAllOrdersF64NarrowRat)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersRat:
        case ProbeEntry::kAllOrdersF64NarrowOrdersRatHorner:
            (relaxed ? BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff
                     : BoysCudaLaunchAllOrdersF64NarrowOrdersRat)(n, x,
                             static_cast<double*>(out), count, stream);
            break;
        // The uniform route's four rows, and the one route of this lane whose
        // rung axis is empty rather than occupied: its table is stored at one
        // degree for every order and every interval, so no rung's criterion has
        // a cut to make of it and there is no relaxed launcher to name. The
        // probe therefore measures each of them at m = 1 and takes no figure at
        // the rungs above, which is what their rows report — a reading this run
        // did not take rather than full-accuracy arithmetic filed under a
        // relaxed name.
        //
        // The four names are two launchers: the scheme axis is a real choice on
        // this route (the Chebyshev image against the monomial one) and the
        // packing axis is the route's own single reading, so the pair of rows
        // naming it launches the kernel of the scheme it carries — the same
        // arms the fit route's pairs have, for the same reason.
        case ProbeEntry::kAllOrdersF64Uniform:
        case ProbeEntry::kAllOrdersF64OrdersUniform:
            // Every rung of this route runs this launcher: the table is stored at
            // one degree for every order and every interval, so there is no
            // shorter image of it to read at a relaxed one and the route's
            // arithmetic is the same at all twelve. The row says so
            // (DeviceEntryServedAtRung) and the entry answers so, and the figure
            // this files under a rung is the route's own at that rung.
            BoysCudaLaunchAllOrdersF64Uniform(n,
                                              x,
                                              static_cast<double*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF64UniformHorner:
        case ProbeEntry::kAllOrdersF64OrdersUniformHorner:
            // The other form of the same table, at the same contract: the degree
            // is a property of the stored fit and not of the basis it is summed
            // in, so a rung of it is this launcher too.
            BoysCudaLaunchAllOrdersF64UniformHorner(n,
                                                    x,
                                                    static_cast<double*>(out),
                                                    count,
                                                    stream);
            break;

        // The grid's rational member, one launcher for both scheme names and for
        // the orders rows: the pair is stored in one form, so the route's packing
        // axis has one member and no second arithmetic is timed under a second
        // name.
        case ProbeEntry::kAllOrdersF64UniformRat:
            BoysCudaLaunchAllOrdersF64UniformRat(n, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64UniformRatHorner:
            BoysCudaLaunchAllOrdersF64UniformRat(n, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersUniformRat:
            BoysCudaLaunchAllOrdersF64UniformRat(n, x, static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersF64UniformRat(n, x, static_cast<double*>(out), count, stream);
            break;
        // The float lane's own partition, both of whose bases this lane holds a
        // rung cut for: the arms below are the two forms of each row and not a
        // refusal, because the degree tables a rung reads are resident under the
        // row's own name.
        case ProbeEntry::kAllOrdersF32Narrow:
            // This partition's cut is held — region B's float degrees beside the
            // double lane's region-A cut — so the row answers at every rung and
            // the two arms are the two forms of it.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowEff
                     : BoysCudaLaunchAllOrdersF32Narrow)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowMono:
            // The same partition in the other basis, whose cut is a table of
            // that basis's own and is held as well — so this row answers at
            // every rung too and the two arms are the two forms of it.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowMonoEff
                     : BoysCudaLaunchAllOrdersF32NarrowMono)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        // The same partition on the packing axis's other side, at the same
        // rungs: the cut is the table's and the row that reads it on either axis
        // is served where its twin is and nowhere else.
        case ProbeEntry::kAllOrdersF32NarrowOrders:
            // The same two tables on the packing axis's other side.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowOrdersEff
                     : BoysCudaLaunchAllOrdersF32NarrowOrders)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersMono:
            // The same two tables on the packing axis's other side.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowOrdersMonoEff
                     : BoysCudaLaunchAllOrdersF32NarrowOrdersMono)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        // The float lane's grid, which is the uniform route at that lane's width
        // and serves every rung for the reason the double lane's four do: one
        // degree for every order and every interval, so no rung has a cut to make
        // of it and every rung of it runs this launcher.
        case ProbeEntry::kAllOrdersF32Uniform:
            BoysCudaLaunchAllOrdersF32Uniform(n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32UniformHorner:
            // The other form of that grid, at the same contract.
            BoysCudaLaunchAllOrdersF32UniformHorner(n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;

        case ProbeEntry::kAllOrdersF32UniformRat:
            BoysCudaLaunchAllOrdersF32UniformRat(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32UniformRatHorner:
            BoysCudaLaunchAllOrdersF32UniformRat(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformRat:
            BoysCudaLaunchAllOrdersF32UniformRat(n, x, static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersF32UniformRat(n, x, static_cast<float*>(out), count, stream);
            break;
        // The grid's two orders rows, which are those rows' kernels: the route's
        // packing axis has one member, so a rung of either is this launcher's and
        // no second arithmetic is timed under a second name.
        case ProbeEntry::kAllOrdersF32OrdersUniform:
            BoysCudaLaunchAllOrdersF32Uniform(n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformHorner:
            BoysCudaLaunchAllOrdersF32UniformHorner(n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        // The float lane's rational route, whose two scheme names reach one
        // launcher each: a pair of rows per partition, one kernel per partition.
        // A relaxed rung is refused here for the reason the entry refuses it —
        // no float rational cut is held by this lane — so the probe is told the
        // same answer the call site is.
        case ProbeEntry::kAllOrdersF32Rat:
        case ProbeEntry::kAllOrdersF32RatHorner:
            // This route's cut is held on both lanes, so the row answers at every rung.
            (relaxed ? BoysCudaLaunchAllOrdersF32RatEff
                     : BoysCudaLaunchAllOrdersF32Rat)(n,
                                          x,
                                          static_cast<float*>(out),
                                          count,
                                          stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowRat:
        case ProbeEntry::kAllOrdersF32NarrowRatHorner:
            // The same pair tables on the narrow partition, cut by the same criterion.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowRatEff
                     : BoysCudaLaunchAllOrdersF32NarrowRat)(n,
                                                x,
                                                static_cast<float*>(out),
                                                count,
                                                stream);
            break;
        // The route's two pairs on the other packing axis, refused at a relaxed
        // rung for the reason their per-argument twins are: the float lane's
        // region-B pairs have no rung cut here.
        case ProbeEntry::kAllOrdersF32OrdersRat:
        case ProbeEntry::kAllOrdersF32OrdersRatHorner:
            // The same pair on the packing axis's other side.
            (relaxed ? BoysCudaLaunchAllOrdersF32OrdersRatEff
                     : BoysCudaLaunchAllOrdersF32OrdersRat)(n,
                                                x,
                                                static_cast<float*>(out),
                                                count,
                                                stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersRat:
        case ProbeEntry::kAllOrdersF32NarrowOrdersRatHorner:
            // And on that side's narrow partition.
            (relaxed ? BoysCudaLaunchAllOrdersF32NarrowOrdersRatEff
                     : BoysCudaLaunchAllOrdersF32NarrowOrdersRat)(n,
                                                      x,
                                                      static_cast<float*>(out),
                                                      count,
                                                      stream);
            break;
        case ProbeEntry::kAllOrdersF32:
            (relaxed ? BoysCudaLaunchAllOrdersF32Eff
                     : BoysCudaLaunchAllOrdersF32)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        // The float lane's other packing axis, one arm per row of it. These are
        // separate arms and not a fall-through: the entry the row names is which
        // kernel runs, and an entry this switch does not name is refused below
        // rather than measured as whichever arm happens to sit nearest.
        case ProbeEntry::kAllOrdersF32Orders:
            (relaxed ? BoysCudaLaunchAllOrdersF32OrdersEff
                     : BoysCudaLaunchAllOrdersF32Orders)(n, x,
                             static_cast<float*>(out), count, stream);
            break;
        case ProbeEntry::kAllNF64:
            (relaxed ? BoysCudaLaunchAllNF64Eff
                     : BoysCudaLaunchAllNF64)(nmax, x,
                             static_cast<double*>(out), count, stream);
            break;
        case ProbeEntry::kAllNF32:
            (relaxed ? BoysCudaLaunchAllNF32Eff
                     : BoysCudaLaunchAllNF32)(nmax, x,
                             static_cast<float*>(out), count, stream);
            break;
#if BoysFp16
        // The fp16 lane's launched entries, behind the seam that declares both
        // the launchers above and the batch entries that reach them.
        case ProbeEntry::kSingleF16:
            (relaxed ? BoysCudaLaunchSingleF16Eff
                     : BoysCudaLaunchSingleF16)(n, xh,
                             out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16:
            (relaxed ? BoysCudaLaunchAllOrdersF16Eff
                     : BoysCudaLaunchAllOrdersF16)(n, xh,
                             out, count, stream);
            break;
        case ProbeEntry::kAllNF16:
            (relaxed ? BoysCudaLaunchAllNF16Eff
                     : BoysCudaLaunchAllNF16)(nmax, xh,
                             out, count, stream);
            break;
#else
        // As in LaunchInKernel: with the seam closed the fp16 arms are the only
        // ones that would have read a half array.
        (void)xh;
#endif // BoysFp16
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
                          request.multiplier,
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
                          request.multiplier,
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
