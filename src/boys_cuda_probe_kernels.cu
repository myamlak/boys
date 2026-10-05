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
//
// The division form is a template argument of every call below for the same
// reason: the forms a kernel is not run at must be absent from it rather than
// merely untaken.
// ---------------------------------------------------------------------------

/// The double lane's four entries, and the cheap stand-in the removed-call half
/// writes in their place.
struct Dev64 {
    using Value = double;

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              double x,
                                                              double* out) {
        return BoysDeviceSingleF64<kForm>(tables, order, x, out);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 double* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF64<kForm>(tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            double x,
                                                            double* out) {
        return BoysDeviceAllNF64<kForm, kProbeInKernelTopOrder>(tables, x, out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF64<kForm>(tables, order, x, sink);
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

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              float x,
                                                              float* out) {
        return BoysDeviceSingleF32<kForm, kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 float* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF32<kForm>(tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            float x,
                                                            float* out) {
        return BoysDeviceAllNF32<kForm, kProbeInKernelTopOrder>(tables, x, out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF32<kForm>(tables, order, x, sink);
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

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __half x,
                                                              __half* out) {
        return BoysDeviceSingleF16<kForm>(tables, order, x, out);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 __half* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF16<kForm>(tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            __half x,
                                                            __half* out) {
        return BoysDeviceAllNF16<kForm, kProbeInKernelTopOrder>(tables, x, out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF16<kForm>(tables, order, x, sink);
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
        template <boys::DivisionForm kForm> \
        static __device__ __forceinline__ BoysDeviceStatus AllOrders( \
            const BoysDeviceTables& tables, \
            int order, \
            VALUE arg, \
            VALUE* out, \
            int capacity) { \
            return ENTRY<kForm>(tables, order, arg, out, capacity); \
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

// The orders-axis rows of the two wider lanes: the packing axis's other side, on
// the partition each row names, one policy per row as the rows above are.
BOYS_PROBE_LADDER_POLICY(Dev64Orders, double, BoysDeviceAllOrdersF64Orders)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowOrders, double, BoysDeviceAllOrdersF64NarrowOrders)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowOrdersMono, double, BoysDeviceAllOrdersF64NarrowOrdersMono)
BOYS_PROBE_LADDER_POLICY(Dev64OrdersRat, double, BoysDeviceAllOrdersF64OrdersRat)
BOYS_PROBE_LADDER_POLICY(Dev64NarrowOrdersRat, double, BoysDeviceAllOrdersF64NarrowOrdersRat)
BOYS_PROBE_LADDER_POLICY(Dev64OrdersRatHorner, double, BoysDeviceAllOrdersF64OrdersRatHorner)
BOYS_PROBE_LADDER_POLICY(
    Dev64NarrowOrdersRatHorner, double, BoysDeviceAllOrdersF64NarrowOrdersRatHorner)

BOYS_PROBE_LADDER_POLICY(Dev32Orders, float, BoysDeviceAllOrdersF32Orders)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowOrders, float, BoysDeviceAllOrdersF32NarrowOrders)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowOrdersMono, float, BoysDeviceAllOrdersF32NarrowOrdersMono)
BOYS_PROBE_LADDER_POLICY(Dev32OrdersRat, float, BoysDeviceAllOrdersF32OrdersRat)
BOYS_PROBE_LADDER_POLICY(Dev32NarrowOrdersRat, float, BoysDeviceAllOrdersF32NarrowOrdersRat)
BOYS_PROBE_LADDER_POLICY(Dev32OrdersRatHorner, float, BoysDeviceAllOrdersF32OrdersRatHorner)
BOYS_PROBE_LADDER_POLICY(
    Dev32NarrowOrdersRatHorner, float, BoysDeviceAllOrdersF32NarrowOrdersRatHorner)

#undef BOYS_PROBE_LADDER_POLICY

#if BoysFp16
// The half lane's rows, one policy per row, over the body the row names: the same
// ladder the policy above writes, at the lane's own value type. The cheap
// stand-in is the lane's, spelled as Dev16 spells it.
#define BOYS_PROBE_HALF_POLICY(NAME, ENTRY) \
    struct NAME { \
        using Value = __half; \
 \
        template <boys::DivisionForm kForm> \
        static __device__ __forceinline__ BoysDeviceStatus AllOrders( \
            const BoysDeviceTables& tables, \
            int order, \
            __half arg, \
            __half* out, \
            int capacity) { \
            return ENTRY<kForm>(tables, order, arg, out, capacity); \
        } \
 \
        static __device__ __forceinline__ __half Cheap(__half x, int l) { \
            return __float2half(__half2float(x) * static_cast<float>(l + 1)); \
        } \
    };

BOYS_PROBE_HALF_POLICY(Dev16Narrow, BoysDeviceAllOrdersF16Narrow)
BOYS_PROBE_HALF_POLICY(Dev16NarrowMono, BoysDeviceAllOrdersF16NarrowMono)
BOYS_PROBE_HALF_POLICY(Dev16NarrowOrders, BoysDeviceAllOrdersF16NarrowOrders)
BOYS_PROBE_HALF_POLICY(Dev16NarrowOrdersMono, BoysDeviceAllOrdersF16NarrowOrdersMono)
BOYS_PROBE_HALF_POLICY(Dev16NarrowRat, BoysDeviceAllOrdersF16NarrowRat)
BOYS_PROBE_HALF_POLICY(Dev16NarrowRatHorner, BoysDeviceAllOrdersF16NarrowRatHorner)
BOYS_PROBE_HALF_POLICY(Dev16NarrowOrdersRat, BoysDeviceAllOrdersF16NarrowOrdersRat)
BOYS_PROBE_HALF_POLICY(Dev16NarrowOrdersRatHorner, BoysDeviceAllOrdersF16NarrowOrdersRatHorner)
BOYS_PROBE_HALF_POLICY(Dev16Rat, BoysDeviceAllOrdersF16Rat)
BOYS_PROBE_HALF_POLICY(Dev16RatHorner, BoysDeviceAllOrdersF16RatHorner)
BOYS_PROBE_HALF_POLICY(Dev16Orders, BoysDeviceAllOrdersF16Orders)
BOYS_PROBE_HALF_POLICY(Dev16OrdersRat, BoysDeviceAllOrdersF16OrdersRat)
BOYS_PROBE_HALF_POLICY(Dev16OrdersRatHorner, BoysDeviceAllOrdersF16OrdersRatHorner)
BOYS_PROBE_HALF_POLICY(Dev16Uniform, BoysDeviceAllOrdersF16Uniform)
BOYS_PROBE_HALF_POLICY(Dev16UniformHorner, BoysDeviceAllOrdersF16UniformHorner)
BOYS_PROBE_HALF_POLICY(Dev16UniformRat, BoysDeviceAllOrdersF16UniformRat)
BOYS_PROBE_HALF_POLICY(Dev16UniformRatHorner, BoysDeviceAllOrdersF16UniformRatHorner)

#undef BOYS_PROBE_HALF_POLICY

/// The half lane's fast region-B reading, the one row of that lane that is not a
/// ladder: the shape is one value per argument, so the policy is the single-order
/// entry the row names and not an all-orders body.
struct Dev16Fast {
    using Value = __half;

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __half x,
                                                              __half* out) {
        return BoysDeviceSingleF16Fast<kForm>(tables, order, x, out);
    }

    static __device__ __forceinline__ __half Cheap(__half x, int l) {
        return __float2half(__half2float(x) * static_cast<float>(l + 1));
    }
};
#endif // BoysFp16

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

template <typename Dev, Shape kShape, boys::DivisionForm kForm, bool kWithBoys>
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
            // The cast reaches the whole call: written `(void)Dev::Single<kForm>(...)` the
            // cast takes `Dev::Single` as its operand and the form's angle brackets are then
            // read as a comparison, which nvcc reports as "no operator "<" matches these
            // operands ... void < boys::DivisionForm".
            (void)(Dev::Single<kForm>(tables, order, arg, &value));
        }

        out[i] = value;
    } else if constexpr (kShape == Shape::kAllN)
    {
        typename Dev::Value ladder[kMaxBoysOrder + 1];
        const std::size_t base = BlockStart(i);

        if constexpr (kWithBoys)
        {
            (void)(Dev::AllN<kForm>(tables, arg, ladder));
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
                (void)(Dev::AllOrders<kForm>(tables, order, arg, ladder, kMaxBoysOrder + 1));
            } else
            {
                // As above: the cast encloses the call, so the form is a template argument
                // rather than the left side of a comparison.
                (void)(Dev::EachOrder<kForm>(
                    tables,
                    order,
                    arg,
                    [&](int l, typename Dev::Value v) { ladder[l] = v; }));
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

/// Queues one launch of the caller-shaped kernel for one entry, at one division
/// form, with or without the Boys call in it.
template <boys::DivisionForm kForm>
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
    InKernelKernel<dev, Shape::shape, kForm, true><<<blocks, kThreadsPerBlock, 0, stream>>>(  \
        *handle, n, static_cast<const value_type*>(arg), static_cast<value_type*>(out), count)

#define BOYS_PROBE_LAUNCH_PLAIN(dev, value_type, shape, arg)                                 \
    InKernelKernel<dev, Shape::shape, kForm, false><<<blocks, kThreadsPerBlock, 0, stream>>>( \
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
            // The orders-axis rows of the two wider lanes: the same ladder one
            // packing axis over, on the partition each row names. Their argument
            // array is the lane's, as it is for every row above.
            case ProbeEntry::kDeviceAllOrdersF64Orders:
                BOYS_PROBE_LAUNCH(Dev64Orders, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrders:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrders, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersMono:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRat:
                BOYS_PROBE_LAUNCH(Dev64OrdersRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRat:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev64OrdersRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Orders:
                BOYS_PROBE_LAUNCH(Dev32Orders, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrders:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrders, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersMono:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRat:
                BOYS_PROBE_LAUNCH(Dev32OrdersRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRat:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev32OrdersRatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersRatHorner, float, kAllOrders, xf);
                break;
            // The half lane's rows, one arm per row of the option table that
            // names one. The label is written outside the seam's guard and the
            // call inside it, so the row is named in every configuration: with
            // the seam open the arm measures the entry the row names, and with
            // it closed it refuses a row this build does not carry rather than
            // leaving it to the default, which does not name it either.
            case ProbeEntry::kDeviceAllOrdersF16Narrow:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16Narrow, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowMono:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowMono, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrders:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowOrders, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMono:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersMono, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Rat:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16Rat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16RatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16RatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Orders:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16Orders, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16OrdersRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16OrdersRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16OrdersRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Uniform:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16Uniform, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16UniformHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16UniformRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH(Dev16UniformRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceSingleF16Fast:
#if BoysFp16
                // The one row of the lane that is not a ladder: its shape is one
                // value per argument, so the call is the single-order entry.
                BOYS_PROBE_LAUNCH(Dev16Fast, __half, kSingle, xh);
                break;
#else
                return 1;
#endif
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
            // The removed-call half of the thirty-two arms above, in the same
            // order and over the same policies: the subtraction's other side.
            case ProbeEntry::kDeviceAllOrdersF64Orders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Orders, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrders, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersRat, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersRatHorner, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Orders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Orders, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrders, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersRat, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersRatHorner, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersRatHorner, float, kAllOrders, xf);
                break;
            // The half lane's rows, at the same shape as the arms above: the
            // label outside the seam's guard, the call inside it.
            case ProbeEntry::kDeviceAllOrdersF16Narrow:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Narrow, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowMono:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowMono, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrders:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrders, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMono:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersMono, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Rat:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Rat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16RatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16RatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Orders:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Orders, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16OrdersRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16Uniform:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Uniform, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformRat:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformRat, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceAllOrdersF16UniformRatHorner:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformRatHorner, __half, kAllOrders, xh);
                break;
#else
                return 1;
#endif
            case ProbeEntry::kDeviceSingleF16Fast:
#if BoysFp16
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Fast, __half, kSingle, xh);
                break;
#else
                return 1;
#endif
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
int BoysCudaLaunchSingleF32(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF32Fast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Narrow(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowMono(
    int, const int*, const double*, float*, std::size_t, void*);
// The whole-range partition's monomial basis, which is a kernel of its own here
// as it is one launcher up.
int BoysCudaLaunchAllOrdersF32Mono(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Uniform(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32UniformRat(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32UniformHorner(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Rat(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRat(
    int, const int*, const double*, float*, std::size_t, void*);
// The float lane's other packing axis, whose rows this file arms below. The
// grid's two orders rows need no launcher of their own: they launch the two
// declared above.
int BoysCudaLaunchAllOrdersF32Orders(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrders(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    int, const int*, const double*, float*, std::size_t, void*);
// The whole-range partition on that side, in the monomial basis: the grid's two
// orders rows share a launcher and this partition's two do not, because the
// monomial image is a kernel of its own.
int BoysCudaLaunchAllOrdersF32OrdersMono(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRat(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllNF32(int, int, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleF64(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Narrow(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Orders(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrders(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Mono(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersMono(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowMono(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Rat(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersRat(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowRat(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Uniform(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64UniformRat(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64UniformHorner(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF64(int, int, const double*, double*, std::size_t, void*);
#if BoysFp16
int BoysCudaLaunchSingleF16(int, const int*, const void*, void*, std::size_t, void*);
// The lane's fast region-B reading: the symbol names the choice, as it does one
// launcher up in the float lane.
int BoysCudaLaunchSingleF16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16(int, int, const void*, void*, std::size_t, void*);
// The lane's partition and route bodies, the same one-per-body list the float
// lane's launchers above are declared in: a half entry reaches one of these and
// no other, and the probe names the one each entry's row carries.
int BoysCudaLaunchAllOrdersF16Narrow(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowMono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Mono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Rat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Uniform(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16UniformHorner(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16UniformRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Orders(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrders(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersMono(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersMono(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersRat(
    int, const int*, const void*, void*, std::size_t, void*);
#endif // BoysFp16
}

int LaunchLaunched(boys::DivisionForm form,
                   ProbeEntry entry,
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
            BoysCudaLaunchSingleF64(static_cast<int>(form),
                                    n,
                                    x,
                                    static_cast<double*>(out),
                                    count,
                                    stream);
            break;
        case ProbeEntry::kSingleF32:
            BoysCudaLaunchSingleF32(static_cast<int>(form),
                                    n,
                                    x,
                                    static_cast<float*>(out),
                                    count,
                                    stream);
            break;
        case ProbeEntry::kSingleF32Fast:
            BoysCudaLaunchSingleF32Fast(static_cast<int>(form),
                                        n,
                                        x,
                                        static_cast<float*>(out),
                                        count,
                                        stream);
            break;
        case ProbeEntry::kAllOrdersF64:
            BoysCudaLaunchAllOrdersF64(static_cast<int>(form),
                                       n,
                                       x,
                                       static_cast<double*>(out),
                                       count,
                                       stream);
            break;
        case ProbeEntry::kAllOrdersF64Narrow:
            BoysCudaLaunchAllOrdersF64Narrow(static_cast<int>(form),
                                             n,
                                             x,
                                             static_cast<double*>(out),
                                             count,
                                             stream);
            break;
        case ProbeEntry::kAllOrdersF64Orders:
            BoysCudaLaunchAllOrdersF64Orders(static_cast<int>(form),
                                             n,
                                             x,
                                             static_cast<double*>(out),
                                             count,
                                             stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrders:
            BoysCudaLaunchAllOrdersF64NarrowOrders(static_cast<int>(form),
                                                   n,
                                                   x,
                                                   static_cast<double*>(out),
                                                   count,
                                                   stream);
            break;
        case ProbeEntry::kAllOrdersF64Mono:
            BoysCudaLaunchAllOrdersF64Mono(static_cast<int>(form),
                                           n,
                                           x,
                                           static_cast<double*>(out),
                                           count,
                                           stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersMono:
            BoysCudaLaunchAllOrdersF64OrdersMono(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowMono:
            BoysCudaLaunchAllOrdersF64NarrowMono(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersMono:
            BoysCudaLaunchAllOrdersF64NarrowOrdersMono(static_cast<int>(form),
                                                       n,
                                                       x,
                                                       static_cast<double*>(out),
                                                       count,
                                                       stream);
            break;
        // The fit route's rows: the two scheme names of a shape select one
        // arithmetic, so both names launch the same kernel.
        case ProbeEntry::kAllOrdersF64Rat:
        case ProbeEntry::kAllOrdersF64RatHorner:
            BoysCudaLaunchAllOrdersF64Rat(static_cast<int>(form),
                                          n,
                                          x,
                                          static_cast<double*>(out),
                                          count,
                                          stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersRat:
        case ProbeEntry::kAllOrdersF64OrdersRatHorner:
            BoysCudaLaunchAllOrdersF64OrdersRat(static_cast<int>(form),
                                                n,
                                                x,
                                                static_cast<double*>(out),
                                                count,
                                                stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowRat:
        case ProbeEntry::kAllOrdersF64NarrowRatHorner:
            BoysCudaLaunchAllOrdersF64NarrowRat(static_cast<int>(form),
                                                n,
                                                x,
                                                static_cast<double*>(out),
                                                count,
                                                stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersRat:
        case ProbeEntry::kAllOrdersF64NarrowOrdersRatHorner:
            BoysCudaLaunchAllOrdersF64NarrowOrdersRat(static_cast<int>(form),
                                                      n,
                                                      x,
                                                      static_cast<double*>(out),
                                                      count,
                                                      stream);
            break;
        // The uniform route's four rows, whose table is stored at one degree for
        // every order and every interval: the route has one arithmetic and no
        // shorter image of it to read.
        //
        // The four names are two launchers: the scheme axis is a real choice on
        // this route (the Chebyshev image against the monomial one) and the
        // packing axis is the route's own single reading, so the pair of rows
        // naming it launches the kernel of the scheme it carries; the same arms
        // the fit route's pairs have, for the same reason.
        case ProbeEntry::kAllOrdersF64Uniform:
        case ProbeEntry::kAllOrdersF64OrdersUniform:
            // One degree for every order and every interval, so the stored table
            // is the whole of this route's arithmetic.
            BoysCudaLaunchAllOrdersF64Uniform(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<double*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF64UniformHorner:
        case ProbeEntry::kAllOrdersF64OrdersUniformHorner:
            // The other form of the same table, at the same contract.
            BoysCudaLaunchAllOrdersF64UniformHorner(static_cast<int>(form),
                                                    n,
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
            BoysCudaLaunchAllOrdersF64UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64UniformRatHorner:
            BoysCudaLaunchAllOrdersF64UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersUniformRat:
            BoysCudaLaunchAllOrdersF64UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersF64UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        // The float lane's own partition, whose two bases carry their own degrees:
        // the two arms below are that partition's two stored forms.
        case ProbeEntry::kAllOrdersF32Narrow:
            // The Chebyshev form of that partition's tables.
            BoysCudaLaunchAllOrdersF32Narrow(static_cast<int>(form),
                                             n,
                                             x,
                                             static_cast<float*>(out),
                                             count,
                                             stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowMono:
            // The same partition in the monomial basis.
            BoysCudaLaunchAllOrdersF32NarrowMono(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32Mono:
            // The whole-range partition in the monomial basis, which is a kernel
            // of its own: this row is that basis and not the Chebyshev one the
            // wide row above launches.
            BoysCudaLaunchAllOrdersF32Mono(static_cast<int>(form),
                                           n,
                                           x,
                                           static_cast<float*>(out),
                                           count,
                                           stream);
            break;
        // The same partition on the packing axis's other side.
        case ProbeEntry::kAllOrdersF32NarrowOrders:
            BoysCudaLaunchAllOrdersF32NarrowOrders(static_cast<int>(form),
                                                   n,
                                                   x,
                                                   static_cast<float*>(out),
                                                   count,
                                                   stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersMono:
            BoysCudaLaunchAllOrdersF32NarrowOrdersMono(static_cast<int>(form),
                                                       n,
                                                       x,
                                                       static_cast<float*>(out),
                                                       count,
                                                       stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersMono:
            // The same side and the same basis on the whole-range partition.
            BoysCudaLaunchAllOrdersF32OrdersMono(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        // The float lane's grid, which is the uniform route at that lane's width:
        // one degree for every order and every interval, so the route has one
        // arithmetic here too.
        case ProbeEntry::kAllOrdersF32Uniform:
            BoysCudaLaunchAllOrdersF32Uniform(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32UniformHorner:
            // The other form of that grid, at the same contract.
            BoysCudaLaunchAllOrdersF32UniformHorner(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;

        case ProbeEntry::kAllOrdersF32UniformRat:
            BoysCudaLaunchAllOrdersF32UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32UniformRatHorner:
            BoysCudaLaunchAllOrdersF32UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformRat:
            BoysCudaLaunchAllOrdersF32UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersF32UniformRat(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        // The grid's two orders rows, which are those rows' kernels: the route's
        // packing axis has one member, so no second arithmetic is timed under a
        // second name.
        case ProbeEntry::kAllOrdersF32OrdersUniform:
            BoysCudaLaunchAllOrdersF32Uniform(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersUniformHorner:
            BoysCudaLaunchAllOrdersF32UniformHorner(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        // The float lane's rational route, whose two scheme names reach one
        // launcher each: a pair of rows per partition, one kernel per partition,
        // each reading the pairs its own table stores.
        case ProbeEntry::kAllOrdersF32Rat:
        case ProbeEntry::kAllOrdersF32RatHorner:
            BoysCudaLaunchAllOrdersF32Rat(static_cast<int>(form),
                                          n,
                                          x,
                                          static_cast<float*>(out),
                                          count,
                                          stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowRat:
        case ProbeEntry::kAllOrdersF32NarrowRatHorner:
            // The same pair tables on the narrow partition.
            BoysCudaLaunchAllOrdersF32NarrowRat(static_cast<int>(form),
                                                n,
                                                x,
                                                static_cast<float*>(out),
                                                count,
                                                stream);
            break;
        // The route's two pairs on the other packing axis.
        case ProbeEntry::kAllOrdersF32OrdersRat:
        case ProbeEntry::kAllOrdersF32OrdersRatHorner:
            // The same pair on the packing axis's other side.
            BoysCudaLaunchAllOrdersF32OrdersRat(static_cast<int>(form),
                                                n,
                                                x,
                                                static_cast<float*>(out),
                                                count,
                                                stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersRat:
        case ProbeEntry::kAllOrdersF32NarrowOrdersRatHorner:
            // And on that side's narrow partition.
            BoysCudaLaunchAllOrdersF32NarrowOrdersRat(static_cast<int>(form),
                                                      n,
                                                      x,
                                                      static_cast<float*>(out),
                                                      count,
                                                      stream);
            break;
        case ProbeEntry::kAllOrdersF32:
            BoysCudaLaunchAllOrdersF32(static_cast<int>(form),
                                       n,
                                       x,
                                       static_cast<float*>(out),
                                       count,
                                       stream);
            break;
        // The float lane's other packing axis, one arm per row of it. These are
        // separate arms and not a fall-through: the entry the row names is which
        // kernel runs, and an entry this switch does not name is refused below
        // rather than measured as whichever arm happens to sit nearest.
        case ProbeEntry::kAllOrdersF32Orders:
            BoysCudaLaunchAllOrdersF32Orders(static_cast<int>(form),
                                             n,
                                             x,
                                             static_cast<float*>(out),
                                             count,
                                             stream);
            break;
        case ProbeEntry::kAllNF64:
            BoysCudaLaunchAllNF64(static_cast<int>(form),
                                  nmax,
                                  x,
                                  static_cast<double*>(out),
                                  count,
                                  stream);
            break;
        case ProbeEntry::kAllNF32:
            BoysCudaLaunchAllNF32(static_cast<int>(form),
                                  nmax,
                                  x,
                                  static_cast<float*>(out),
                                  count,
                                  stream);
            break;
#if BoysFp16
        // The fp16 lane's launched entries, behind the seam that declares both
        // the launchers above and the batch entries that reach them.
        case ProbeEntry::kSingleF16:
            BoysCudaLaunchSingleF16(static_cast<int>(form),
                                    n,
                                    xh,
                                    out,
                                    count,
                                    stream);
            break;
        case ProbeEntry::kSingleF16Fast:
            // The lane's other region-B bound, which is its own kernel and its
            // own launcher: the wide arm above is not this row.
            BoysCudaLaunchSingleF16Fast(static_cast<int>(form),
                                        n,
                                        xh,
                                        out,
                                        count,
                                        stream);
            break;
        case ProbeEntry::kAllOrdersF16:
            BoysCudaLaunchAllOrdersF16(static_cast<int>(form),
                                       n,
                                       xh,
                                       out,
                                       count,
                                       stream);
            break;
        case ProbeEntry::kAllNF16:
            BoysCudaLaunchAllNF16(static_cast<int>(form),
                                  nmax,
                                  xh,
                                  out,
                                  count,
                                  stream);
            break;
        // The lane's partition and route bodies, one arm per entry of the
        // option table that names one, in the table's own order. Two entries a
        // scheme pair reaches share a launcher as they share a kernel: the
        // forwarding entry's arm names the launcher its sibling names, so what
        // is timed is the arithmetic the entry runs rather than a second name
        // for it. The uniform grid's packing axis has one member, so its
        // orders-axis entries run the per-argument kernel the table's own
        // comment records.
        case ProbeEntry::kAllOrdersF16Narrow:
            BoysCudaLaunchAllOrdersF16Narrow(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowMono:
            BoysCudaLaunchAllOrdersF16NarrowMono(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16Mono:
            // The whole-range partition in the monomial basis, a kernel of its
            // own here as it is one lane up.
            BoysCudaLaunchAllOrdersF16Mono(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16Uniform:
        case ProbeEntry::kAllOrdersF16OrdersUniform:
            BoysCudaLaunchAllOrdersF16Uniform(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16UniformHorner:
        case ProbeEntry::kAllOrdersF16OrdersUniformHorner:
            BoysCudaLaunchAllOrdersF16UniformHorner(
                static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16Rat:
        case ProbeEntry::kAllOrdersF16RatHorner:
            BoysCudaLaunchAllOrdersF16Rat(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowRat:
        case ProbeEntry::kAllOrdersF16NarrowRatHorner:
            BoysCudaLaunchAllOrdersF16NarrowRat(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16UniformRat:
        case ProbeEntry::kAllOrdersF16UniformRatHorner:
        case ProbeEntry::kAllOrdersF16OrdersUniformRat:
        case ProbeEntry::kAllOrdersF16OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersF16UniformRat(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16Orders:
            BoysCudaLaunchAllOrdersF16Orders(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrders:
            BoysCudaLaunchAllOrdersF16NarrowOrders(
                static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersMono:
            BoysCudaLaunchAllOrdersF16NarrowOrdersMono(
                static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersMono:
            // The same side and the same basis on the whole-range partition.
            BoysCudaLaunchAllOrdersF16OrdersMono(
                static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersRat:
        case ProbeEntry::kAllOrdersF16OrdersRatHorner:
            BoysCudaLaunchAllOrdersF16OrdersRat(static_cast<int>(form), n, xh, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersRat:
        case ProbeEntry::kAllOrdersF16NarrowOrdersRatHorner:
            BoysCudaLaunchAllOrdersF16NarrowOrdersRat(
                static_cast<int>(form), n, xh, out, count, stream);
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
    return LaunchLaunched(static_cast<boys::DivisionForm>(request.form),
                          static_cast<ProbeEntry>(request.entry),
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

    // The form is the caller's and it arrives as an int across this boundary, so
    // the enumeration is exhausted here: a value outside it is a caller's int and
    // is refused rather than measured as whichever form sits nearest.
    switch (static_cast<boys::DivisionForm>(request.form))
    {
        case boys::DivisionForm::kExactDivision:
            return LaunchInKernel<boys::DivisionForm::kExactDivision>(
                static_cast<ProbeEntry>(request.entry),
                static_cast<const BoysDeviceTables*>(request.handle),
                request.n,
                request.x,
                request.xf,
                static_cast<const __half*>(request.xh),
                request.out,
                static_cast<std::size_t>(request.count),
                request.withBoys != 0,
                stream);
        case boys::DivisionForm::kPlainReciprocal:
            return LaunchInKernel<boys::DivisionForm::kPlainReciprocal>(
                static_cast<ProbeEntry>(request.entry),
                static_cast<const BoysDeviceTables*>(request.handle),
                request.n,
                request.x,
                request.xf,
                static_cast<const __half*>(request.xh),
                request.out,
                static_cast<std::size_t>(request.count),
                request.withBoys != 0,
                stream);
        case boys::DivisionForm::kRefinedReciprocal:
            return LaunchInKernel<boys::DivisionForm::kRefinedReciprocal>(
                static_cast<ProbeEntry>(request.entry),
                static_cast<const BoysDeviceTables*>(request.handle),
                request.n,
                request.x,
                request.xf,
                static_cast<const __half*>(request.xh),
                request.out,
                static_cast<std::size_t>(request.count),
                request.withBoys != 0,
                stream);
        default:
            return 1;
    }
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
        // The other half of the subtraction kernel, named as its own region rather than left
        // to the fall-through: the arm a switch does not name answers with the body it holds,
        // and the body here is a region a report prints as a measured figure.
        case ProbeWhat::kInKernelEntry:
            return request->handle == nullptr
                       ? 1
                       : TimeRegion(RunInKernel, *request, request->outMs);
        // A region this enumeration does not name. \c what arrives as an int across this
        // entry's own ABI, so a value outside the enumerators is a caller's int and not an
        // arm nothing can reach. Refused, as the entry dispatches above refuse the entries
        // they do not name, rather than measured as whichever region sits nearest.
        default:
            return 1;
    }
}

} // extern "C"
