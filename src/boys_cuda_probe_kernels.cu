// Device side of the option probe: the timed regions, the kernels the library's entries are
// launched into, and the subtraction kernels the device-callable entries are measured through.
// The host side owns the protocol; this file owns the clock - a CUDA event pair around launches
// into resident buffers. C++20 and a CUDA-safe include list only: C++23 headers poison this TU.

#include "boys_cuda_probe_entries.hpp"

#include "boys/boys_cuda_device.hpp"

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

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

// The floor: a kernel launched exactly as the entries are - same grid, same repetition count,
// same event pair - that does no work. Its device time per launch is the part of every launched
// figure that is getting a kernel started rather than evaluating anything. It keeps no memory
// traffic: that is the subtraction kernel's own baseline half, which measures the caller's.
__global__ void FloorKernel(int* touched) {
    if (blockIdx.x == 0 && threadIdx.x == 0 && touched != nullptr)
    {
        *touched = 1;
    }
}

// The subtraction kernel. One body per precision, with the four shapes written out at their own
// call sites: a run-time branch between the precisions would compile every lane's arithmetic into
// every kernel and measure the wrong one. The division form is a template argument of every call
// below for the same reason: the forms a kernel is not run at must be absent from it.

/// The double lane's four entries, and the cheap stand-in the half without the Boys
/// call writes in their place.
///
/// The region-B exponential is the entry's one axis on this lane too, and it is a
/// template argument here for the same reason it is one on the float lane below:
/// that is how the library offers it, as two members with two figures over one
/// set of tables.
template <bool kFastExp>
struct Dev64T {
    using Value = double;

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              double x,
                                                              double* out) {
        return BoysDeviceSingleF64<kForm, kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 double* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF64<kForm, kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            double x,
                                                            double* out) {
        return BoysDeviceAllNF64<kForm, kProbeInKernelTopOrder,
                                 kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(tables, x,
                                                                                       out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 double x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF64<kForm, Sink,
                                      kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, sink);
    }

    static __device__ __forceinline__ double Cheap(double x, int l) {
        return x * static_cast<double>(l + 1);
    }
};

using Dev64 = Dev64T<false>;
using Dev64Fast = Dev64T<true>;

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
        return BoysDeviceAllOrdersF32<kForm, kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            float x,
                                                            float* out) {
        return BoysDeviceAllNF32<kForm, kProbeInKernelTopOrder,
                                 kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, x, out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 float x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF32<kForm, Sink,
                                      kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, sink);
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

// The partition and route axes of the ladder shape, one policy per entry: each is the all-orders
// ladder the policies above reach, over another partition, another route or the other stored form
// of the one it is already on. One policy per entry rather than per axis, because the two names a
// pair carries are two entries and two rows of the space.
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
// The coarsest partition in the monomial basis. It is its own row on every lane that carries
// it rather than a second reading of the Chebyshev one: the two stored forms are one fit over
// one cut, read by Horner instead of the split Clenshaw recurrence.
BOYS_PROBE_LADDER_POLICY(Dev64Mono, double, BoysDeviceAllOrdersF64Mono)
BOYS_PROBE_LADDER_POLICY(Dev64OrdersMono, double, BoysDeviceAllOrdersF64OrdersMono)
BOYS_PROBE_LADDER_POLICY(Dev32Mono, float, BoysDeviceAllOrdersF32Mono)
BOYS_PROBE_LADDER_POLICY(Dev32OrdersMono, float, BoysDeviceAllOrdersF32OrdersMono)

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

// The fast region-B reading's rows over the same axes: one policy per row, over the same body the
// row's accurate sibling names. It is a second macro rather than a fourth field of the one above
// because the accurate reading is what an invocation saying nothing means.
#define BOYS_PROBE_LADDER_POLICY_FAST(NAME, VALUE, ENTRY) \
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
            return ENTRY<kForm, RegionBExp::kFast>(tables, order, arg, out, capacity); \
        } \
 \
        static __device__ __forceinline__ VALUE Cheap(VALUE x, int l) { \
            return x * static_cast<VALUE>(l + 1); \
        } \
    };

// The float lane's fast rows, one policy per row of the option table that names
// one, in the enumeration's own order. The row that is not an axis of the ladder
// is not here: its shape is the whole policy, and Dev32Fast above is it.
BOYS_PROBE_LADDER_POLICY_FAST(Dev32OrdersFast, float, BoysDeviceAllOrdersF32Orders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32RatFast, float, BoysDeviceAllOrdersF32Rat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32OrdersRatFast, float, BoysDeviceAllOrdersF32OrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32RatHornerFast, float, BoysDeviceAllOrdersF32RatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev32OrdersRatHornerFast, float, BoysDeviceAllOrdersF32OrdersRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32NarrowFast, float, BoysDeviceAllOrdersF32Narrow)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32NarrowOrdersFast, float, BoysDeviceAllOrdersF32NarrowOrders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32NarrowMonoFast, float, BoysDeviceAllOrdersF32NarrowMono)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev32NarrowOrdersMonoFast, float, BoysDeviceAllOrdersF32NarrowOrdersMono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32NarrowRatFast, float, BoysDeviceAllOrdersF32NarrowRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev32NarrowOrdersRatFast, float, BoysDeviceAllOrdersF32NarrowOrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev32NarrowRatHornerFast, float, BoysDeviceAllOrdersF32NarrowRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev32NarrowOrdersRatHornerFast, float, BoysDeviceAllOrdersF32NarrowOrdersRatHorner)

// The double lane's fast rows, the same one-policy-per-row list the float lane's
// fast rows above are, over the same bodies the row's accurate sibling names: the
// lane carries no grid row at this reading, because the grid reads no exponential
// at any argument, so the list is the thirteen ladders and not the whole block.
BOYS_PROBE_LADDER_POLICY_FAST(Dev64OrdersFast, double, BoysDeviceAllOrdersF64Orders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64RatFast, double, BoysDeviceAllOrdersF64Rat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64OrdersRatFast, double, BoysDeviceAllOrdersF64OrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64RatHornerFast, double, BoysDeviceAllOrdersF64RatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev64OrdersRatHornerFast, double, BoysDeviceAllOrdersF64OrdersRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64NarrowFast, double, BoysDeviceAllOrdersF64Narrow)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64NarrowOrdersFast, double, BoysDeviceAllOrdersF64NarrowOrders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64NarrowMonoFast, double, BoysDeviceAllOrdersF64NarrowMono)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev64NarrowOrdersMonoFast, double, BoysDeviceAllOrdersF64NarrowOrdersMono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64NarrowRatFast, double, BoysDeviceAllOrdersF64NarrowRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev64NarrowOrdersRatFast, double, BoysDeviceAllOrdersF64NarrowOrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev64NarrowRatHornerFast, double, BoysDeviceAllOrdersF64NarrowRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev64NarrowOrdersRatHornerFast, double, BoysDeviceAllOrdersF64NarrowOrdersRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64MonoFast, double, BoysDeviceAllOrdersF64Mono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev64OrdersMonoFast, double, BoysDeviceAllOrdersF64OrdersMono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32MonoFast, float, BoysDeviceAllOrdersF32Mono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev32OrdersMonoFast, float, BoysDeviceAllOrdersF32OrdersMono)

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
BOYS_PROBE_HALF_POLICY(Dev16Mono, BoysDeviceAllOrdersF16Mono)
BOYS_PROBE_HALF_POLICY(Dev16OrdersMono, BoysDeviceAllOrdersF16OrdersMono)

#undef BOYS_PROBE_HALF_POLICY

// The half lane's fast rows, the same one-policy-per-row list the float lane's
// fast rows above are, over the lane's own bodies: a build with the seam closed
// has no such row to measure, so they carry the lane's guard as the rows do.
BOYS_PROBE_LADDER_POLICY_FAST(Dev16OrdersFast, __half, BoysDeviceAllOrdersF16Orders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16RatFast, __half, BoysDeviceAllOrdersF16Rat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16OrdersRatFast, __half, BoysDeviceAllOrdersF16OrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16RatHornerFast, __half, BoysDeviceAllOrdersF16RatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev16OrdersRatHornerFast, __half, BoysDeviceAllOrdersF16OrdersRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16NarrowFast, __half, BoysDeviceAllOrdersF16Narrow)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16NarrowOrdersFast, __half, BoysDeviceAllOrdersF16NarrowOrders)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16NarrowMonoFast, __half, BoysDeviceAllOrdersF16NarrowMono)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev16NarrowOrdersMonoFast, __half, BoysDeviceAllOrdersF16NarrowOrdersMono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16NarrowRatFast, __half, BoysDeviceAllOrdersF16NarrowRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev16NarrowOrdersRatFast, __half, BoysDeviceAllOrdersF16NarrowOrdersRat)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev16NarrowRatHornerFast, __half, BoysDeviceAllOrdersF16NarrowRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(
    Dev16NarrowOrdersRatHornerFast, __half, BoysDeviceAllOrdersF16NarrowOrdersRatHorner)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16MonoFast, __half, BoysDeviceAllOrdersF16Mono)
BOYS_PROBE_LADDER_POLICY_FAST(Dev16OrdersMonoFast, __half, BoysDeviceAllOrdersF16OrdersMono)

/// The half lane's fast region-B reading, the four shapes `Dev16` carries at the
/// lane's own reading: the single-order row is a symbol of its own, and the other
/// three are the bodies above at the fast reading, spelled the way `Dev32T` above
/// spells it.
struct Dev16Fast {
    using Value = __half;

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __half x,
                                                              __half* out) {
        return BoysDeviceSingleF16Fast<kForm>(tables, order, x, out);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 __half* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersF16<kForm, RegionBExp::kFast>(tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            __half x,
                                                            __half* out) {
        return BoysDeviceAllNF16<kForm, kProbeInKernelTopOrder, RegionBExp::kFast>(tables, x, out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __half x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderF16<kForm, Sink, RegionBExp::kFast>(tables, order, x, sink);
    }

    static __device__ __forceinline__ __half Cheap(__half x, int l) {
        return __float2half(__half2float(x) * static_cast<float>(l + 1));
    }
};

/// The bfloat16 lane, which is the half lane's own engine with the other format
/// around it: the same float arithmetic, the same tables, and a store into
/// bfloat16 in place of the store into fp16.
///
/// The region-B exponential is a template argument for the reason it is one on
/// every lane above, and this lane offers it the two ways the library does: at
/// the single shape as a symbol of its own, because that entry's fast member is
/// a certified choice of arithmetic rather than a template argument of the
/// accurate one, and at the other three as the parameter the body already takes.
///
/// The bfloat16 bodies this lane names are behind the seam that declares them,
/// so the lane and the rows that reach it carry the lane's guard.
template <bool kFastExp>
struct DevBf16T {
    using Value = __nv_bfloat16;

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus Single(const BoysDeviceTables& tables,
                                                              int order,
                                                              __nv_bfloat16 x,
                                                              __nv_bfloat16* out) {
        if constexpr (kFastExp)
        {
            return BoysDeviceSingleBf16Fast<kForm>(tables, order, x, out);
        }
        else
        {
            return BoysDeviceSingleBf16<kForm>(tables, order, x, out);
        }
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllOrders(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __nv_bfloat16 x,
                                                                 __nv_bfloat16* out,
                                                                 int capacity) {
        return BoysDeviceAllOrdersBf16<kForm, kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, out, capacity);
    }

    template <boys::DivisionForm kForm>
    static __device__ __forceinline__ BoysDeviceStatus AllN(const BoysDeviceTables& tables,
                                                            __nv_bfloat16 x,
                                                            __nv_bfloat16* out) {
        return BoysDeviceAllNBf16<kForm, kProbeInKernelTopOrder,
                                  kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(tables, x,
                                                                                        out);
    }

    template <boys::DivisionForm kForm, typename Sink>
    static __device__ __forceinline__ BoysDeviceStatus EachOrder(const BoysDeviceTables& tables,
                                                                 int order,
                                                                 __nv_bfloat16 x,
                                                                 Sink sink) {
        return BoysDeviceEachOrderBf16<kForm, Sink,
                                       kFastExp ? RegionBExp::kFast : RegionBExp::kAccurate>(
            tables, order, x, sink);
    }

    static __device__ __forceinline__ __nv_bfloat16 Cheap(__nv_bfloat16 x, int l) {
        return __float2bfloat16(__bfloat162float(x) * static_cast<float>(l + 1));
    }
};

using DevBf16 = DevBf16T<false>;
using DevBf16Fast = DevBf16T<true>;

// The bfloat16 lane's ladder and orders rows, one policy per row the option
// table carries, over the body the row names: the same one-policy-per-row list
// the half lane's rows above are, at this lane's value type. The stand-in is
// this lane's, spelled as DevBf16 spells it.
#define BOYS_PROBE_BF16_POLICY(NAME, ENTRY) \
    struct NAME { \
        using Value = __nv_bfloat16; \
 \
        template <boys::DivisionForm kForm> \
        static __device__ __forceinline__ BoysDeviceStatus AllOrders( \
            const BoysDeviceTables& tables, \
            int order, \
            __nv_bfloat16 arg, \
            __nv_bfloat16* out, \
            int capacity) { \
            return ENTRY<kForm>(tables, order, arg, out, capacity); \
        } \
 \
        static __device__ __forceinline__ __nv_bfloat16 Cheap(__nv_bfloat16 x, int l) { \
            return __float2bfloat16(__bfloat162float(x) * static_cast<float>(l + 1)); \
        } \
    };

BOYS_PROBE_BF16_POLICY(DevBf16Narrow, BoysDeviceAllOrdersBf16Narrow)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowMono, BoysDeviceAllOrdersBf16NarrowMono)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowOrders, BoysDeviceAllOrdersBf16NarrowOrders)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowOrdersMono, BoysDeviceAllOrdersBf16NarrowOrdersMono)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowRat, BoysDeviceAllOrdersBf16NarrowRat)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowRatHorner, BoysDeviceAllOrdersBf16NarrowRatHorner)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowOrdersRat, BoysDeviceAllOrdersBf16NarrowOrdersRat)
BOYS_PROBE_BF16_POLICY(DevBf16NarrowOrdersRatHorner, BoysDeviceAllOrdersBf16NarrowOrdersRatHorner)
BOYS_PROBE_BF16_POLICY(DevBf16Rat, BoysDeviceAllOrdersBf16Rat)
BOYS_PROBE_BF16_POLICY(DevBf16RatHorner, BoysDeviceAllOrdersBf16RatHorner)
BOYS_PROBE_BF16_POLICY(DevBf16Orders, BoysDeviceAllOrdersBf16Orders)
BOYS_PROBE_BF16_POLICY(DevBf16OrdersRat, BoysDeviceAllOrdersBf16OrdersRat)
BOYS_PROBE_BF16_POLICY(DevBf16OrdersRatHorner, BoysDeviceAllOrdersBf16OrdersRatHorner)
BOYS_PROBE_BF16_POLICY(DevBf16Uniform, BoysDeviceAllOrdersBf16Uniform)
BOYS_PROBE_BF16_POLICY(DevBf16UniformHorner, BoysDeviceAllOrdersBf16UniformHorner)
BOYS_PROBE_BF16_POLICY(DevBf16UniformRat, BoysDeviceAllOrdersBf16UniformRat)
BOYS_PROBE_BF16_POLICY(DevBf16UniformRatHorner, BoysDeviceAllOrdersBf16UniformRatHorner)
BOYS_PROBE_BF16_POLICY(DevBf16Mono, BoysDeviceAllOrdersBf16Mono)
BOYS_PROBE_BF16_POLICY(DevBf16OrdersMono, BoysDeviceAllOrdersBf16OrdersMono)

#undef BOYS_PROBE_BF16_POLICY

// The bfloat16 lane's rows at the other region-B exponential, the same
// one-policy-per-row list the float lane's fast rows are, over the bodies the
// rows name. The grid rows are absent for the reason the double lane's fast list
// states: the grid reads no exponential at any argument.
#define BOYS_PROBE_BF16_POLICY_FAST(NAME, ENTRY) \
    struct NAME { \
        using Value = __nv_bfloat16; \
 \
        template <boys::DivisionForm kForm> \
        static __device__ __forceinline__ BoysDeviceStatus AllOrders( \
            const BoysDeviceTables& tables, \
            int order, \
            __nv_bfloat16 arg, \
            __nv_bfloat16* out, \
            int capacity) { \
            return ENTRY<kForm, RegionBExp::kFast>(tables, order, arg, out, capacity); \
        } \
 \
        static __device__ __forceinline__ __nv_bfloat16 Cheap(__nv_bfloat16 x, int l) { \
            return __float2bfloat16(__bfloat162float(x) * static_cast<float>(l + 1)); \
        } \
    };

BOYS_PROBE_BF16_POLICY_FAST(DevBf16OrdersFast, BoysDeviceAllOrdersBf16Orders)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16RatFast, BoysDeviceAllOrdersBf16Rat)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16OrdersRatFast, BoysDeviceAllOrdersBf16OrdersRat)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16RatHornerFast, BoysDeviceAllOrdersBf16RatHorner)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16OrdersRatHornerFast, BoysDeviceAllOrdersBf16OrdersRatHorner)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowFast, BoysDeviceAllOrdersBf16Narrow)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowOrdersFast, BoysDeviceAllOrdersBf16NarrowOrders)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowMonoFast, BoysDeviceAllOrdersBf16NarrowMono)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowOrdersMonoFast, BoysDeviceAllOrdersBf16NarrowOrdersMono)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowRatFast, BoysDeviceAllOrdersBf16NarrowRat)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowOrdersRatFast, BoysDeviceAllOrdersBf16NarrowOrdersRat)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16NarrowRatHornerFast, BoysDeviceAllOrdersBf16NarrowRatHorner)
BOYS_PROBE_BF16_POLICY_FAST(
    DevBf16NarrowOrdersRatHornerFast, BoysDeviceAllOrdersBf16NarrowOrdersRatHorner)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16MonoFast, BoysDeviceAllOrdersBf16Mono)
BOYS_PROBE_BF16_POLICY_FAST(DevBf16OrdersMonoFast, BoysDeviceAllOrdersBf16OrdersMono)

#undef BOYS_PROBE_BF16_POLICY_FAST
#endif // BoysFp16

#undef BOYS_PROBE_LADDER_POLICY_FAST

/// The four shapes, as a template parameter. Every shape writes its whole
/// output, and the half without the Boys call writes the same slots with the same
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
                   const __nv_bfloat16* xb,
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
            // The coarsest partition in the monomial basis, one arm per row the table
            // carries: the same bodies as the arms above, over the row's own stored form.
            case ProbeEntry::kDeviceAllOrdersF16Mono:
                BOYS_PROBE_LAUNCH(Dev16Mono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16MonoFast:
                BOYS_PROBE_LAUNCH(Dev16MonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersMono:
                BOYS_PROBE_LAUNCH(Dev16OrdersMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev16OrdersMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Mono:
                BOYS_PROBE_LAUNCH(DevBf16Mono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16MonoFast:
                BOYS_PROBE_LAUNCH(DevBf16MonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersMono:
                BOYS_PROBE_LAUNCH(DevBf16OrdersMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersMonoFast:
                BOYS_PROBE_LAUNCH(DevBf16OrdersMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
#else
            // A build with the seam closed has no fp16 enumerator to be asked
            // for and no half array to read, so this is the only arm that would
            // have touched the parameter.
            (void)xh;
            (void)xb;
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
            // The double lane's rows at the other region-B exponential, one arm
            // per row the table carries: the arm of the row's accurate sibling
            // with the row's own member selected, over the same body and the
            // same tables.
            case ProbeEntry::kDeviceSingleF64Fast:
                BOYS_PROBE_LAUNCH(Dev64Fast, double, kSingle, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Fast:
                BOYS_PROBE_LAUNCH(Dev64Fast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllNF64Fast:
                BOYS_PROBE_LAUNCH(Dev64Fast, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF64Fast:
                BOYS_PROBE_LAUNCH(Dev64Fast, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersFast:
                BOYS_PROBE_LAUNCH(Dev64OrdersFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatFast:
                BOYS_PROBE_LAUNCH(Dev64RatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev64OrdersRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatHornerFast:
                BOYS_PROBE_LAUNCH(Dev64RatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev64OrdersRatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowMonoFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowRatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev64NarrowOrdersRatHornerFast, double, kAllOrders, xd);
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
            // The coarsest partition in the monomial basis on the two wider lanes, one
            // arm per row the table carries: the same bodies as the arms above, over the
            // row's own stored form. Their rows are the lanes' own and the seam does not
            // close them, so the arms stand outside it.
            case ProbeEntry::kDeviceAllOrdersF64Mono:
                BOYS_PROBE_LAUNCH(Dev64Mono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64MonoFast:
                BOYS_PROBE_LAUNCH(Dev64MonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersMono:
                BOYS_PROBE_LAUNCH(Dev64OrdersMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev64OrdersMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Mono:
                BOYS_PROBE_LAUNCH(Dev32Mono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32MonoFast:
                BOYS_PROBE_LAUNCH(Dev32MonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersMono:
                BOYS_PROBE_LAUNCH(Dev32OrdersMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev32OrdersMonoFast, float, kAllOrders, xf);
                break;
#if BoysFp16
            // The half lane's rows of those same axes, one arm per row, with the lane behind the same seam: a
            // build without those entries has no enumerator to be asked for and no half array to read, and a
            // row such a build is without is refused before this switch is reached.
            case ProbeEntry::kDeviceAllOrdersF16Narrow:
                BOYS_PROBE_LAUNCH(Dev16Narrow, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowMono:
                BOYS_PROBE_LAUNCH(Dev16NarrowMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrders:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrders, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMono:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRat:
                BOYS_PROBE_LAUNCH(Dev16NarrowRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHorner:
                BOYS_PROBE_LAUNCH(Dev16NarrowRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRat:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Rat:
                BOYS_PROBE_LAUNCH(Dev16Rat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatHorner:
                BOYS_PROBE_LAUNCH(Dev16RatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Orders:
                BOYS_PROBE_LAUNCH(Dev16Orders, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRat:
                BOYS_PROBE_LAUNCH(Dev16OrdersRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHorner:
                BOYS_PROBE_LAUNCH(Dev16OrdersRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Uniform:
                BOYS_PROBE_LAUNCH(Dev16Uniform, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformHorner:
                BOYS_PROBE_LAUNCH(Dev16UniformHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformRat:
                BOYS_PROBE_LAUNCH(Dev16UniformRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformRatHorner:
                BOYS_PROBE_LAUNCH(Dev16UniformRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceSingleF16Fast:
                // The one row of the lane that is not a ladder: its shape is one
                // value per argument, so the call is the single-order entry.
                BOYS_PROBE_LAUNCH(Dev16Fast, __half, kSingle, xh);
                break;
#endif // BoysFp16
            // The fast region-B reading's rows, in the enumeration's own order: the same bodies as the arms
            // above, at the one template argument the row's own name carries, and armed here because the
            // enumeration holds them after the shapes and axes they share their bodies with.
            case ProbeEntry::kDeviceAllOrdersF32Fast:
                BOYS_PROBE_LAUNCH(Dev32Fast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersFast:
                BOYS_PROBE_LAUNCH(Dev32OrdersFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatFast:
                BOYS_PROBE_LAUNCH(Dev32RatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev32OrdersRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatHornerFast:
                BOYS_PROBE_LAUNCH(Dev32RatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev32OrdersRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowMonoFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowMonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersMonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev32NarrowOrdersRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllNF32Fast:
                BOYS_PROBE_LAUNCH(Dev32Fast, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF32Fast:
                BOYS_PROBE_LAUNCH(Dev32Fast, float, kEachOrder, xf);
                break;
#if BoysFp16
            // The half lane's fast rows, under the lane's guard as the arms above
            // are: a build with the seam closed has no such enumerator to be asked
            // for and no half array to read.
            case ProbeEntry::kDeviceAllOrdersF16Fast:
                BOYS_PROBE_LAUNCH(Dev16Fast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersFast:
                BOYS_PROBE_LAUNCH(Dev16OrdersFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatFast:
                BOYS_PROBE_LAUNCH(Dev16RatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev16OrdersRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatHornerFast:
                BOYS_PROBE_LAUNCH(Dev16RatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev16OrdersRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowMonoFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(Dev16NarrowOrdersRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF16Fast:
                BOYS_PROBE_LAUNCH(Dev16Fast, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF16Fast:
                BOYS_PROBE_LAUNCH(Dev16Fast, __half, kEachOrder, xh);
                break;
            // The bfloat16 lane's rows, in the enumeration's own order: the same bodies as the fp16 arms
            // above, reading this lane's argument array. A second array rather than a reinterpretation of xh -
            // the two formats are 5/10 and 8/7 over the same sixteen bits.
            case ProbeEntry::kDeviceSingleBf16:
                BOYS_PROBE_LAUNCH(DevBf16, __nv_bfloat16, kSingle, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16:
                BOYS_PROBE_LAUNCH(DevBf16, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllNBf16:
                BOYS_PROBE_LAUNCH(DevBf16, __nv_bfloat16, kAllN, xb);
                break;
            case ProbeEntry::kDeviceEachOrderBf16:
                BOYS_PROBE_LAUNCH(DevBf16, __nv_bfloat16, kEachOrder, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Orders:
                BOYS_PROBE_LAUNCH(DevBf16Orders, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Narrow:
                BOYS_PROBE_LAUNCH(DevBf16Narrow, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrders:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrders, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Uniform:
                BOYS_PROBE_LAUNCH(DevBf16Uniform, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowMono:
                BOYS_PROBE_LAUNCH(DevBf16NarrowMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersMono:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformHorner:
                BOYS_PROBE_LAUNCH(DevBf16UniformHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Rat:
                BOYS_PROBE_LAUNCH(DevBf16Rat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRat:
                BOYS_PROBE_LAUNCH(DevBf16OrdersRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRat:
                BOYS_PROBE_LAUNCH(DevBf16NarrowRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRat:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformRat:
                BOYS_PROBE_LAUNCH(DevBf16UniformRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatHorner:
                BOYS_PROBE_LAUNCH(DevBf16RatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatHorner:
                BOYS_PROBE_LAUNCH(DevBf16OrdersRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatHorner:
                BOYS_PROBE_LAUNCH(DevBf16NarrowRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformRatHorner:
                BOYS_PROBE_LAUNCH(DevBf16UniformRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceSingleBf16Fast:
                BOYS_PROBE_LAUNCH(DevBf16Fast, __nv_bfloat16, kSingle, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Fast:
                BOYS_PROBE_LAUNCH(DevBf16Fast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersFast:
                BOYS_PROBE_LAUNCH(DevBf16OrdersFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatFast:
                BOYS_PROBE_LAUNCH(DevBf16RatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatFast:
                BOYS_PROBE_LAUNCH(DevBf16OrdersRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatHornerFast:
                BOYS_PROBE_LAUNCH(DevBf16RatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(DevBf16OrdersRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowMonoFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH(DevBf16NarrowOrdersRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllNBf16Fast:
                BOYS_PROBE_LAUNCH(DevBf16Fast, __nv_bfloat16, kAllN, xb);
                break;
            case ProbeEntry::kDeviceEachOrderBf16Fast:
                BOYS_PROBE_LAUNCH(DevBf16Fast, __nv_bfloat16, kEachOrder, xb);
                break;
#endif // BoysFp16
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
            // The removed-call half of the double lane's fast rows above, in the
            // same order and over the same policies: the subtraction's other
            // side.
            case ProbeEntry::kDeviceSingleF64Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Fast, double, kSingle, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Fast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllNF64Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Fast, double, kAllN, xd);
                break;
            case ProbeEntry::kDeviceEachOrderF64Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Fast, double, kEachOrder, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64RatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64RatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64RatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersRatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersRatFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowRatHornerFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64NarrowOrdersRatHornerFast, double, kAllOrders, xd);
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
            // The coarsest partition in the monomial basis on the two wider lanes, one
            // arm per row the table carries: the same bodies as the arms above, over the
            // row's own stored form. Their rows are the lanes' own and the seam does not
            // close them, so the arms stand outside it.
            case ProbeEntry::kDeviceAllOrdersF64Mono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64Mono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64MonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64MonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersMono, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF64OrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev64OrdersMonoFast, double, kAllOrders, xd);
                break;
            case ProbeEntry::kDeviceAllOrdersF32Mono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Mono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32MonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32MonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersMono, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersMonoFast, float, kAllOrders, xf);
                break;
#if BoysFp16
            // The removed-call half of the same eighteen arms; see above.
            case ProbeEntry::kDeviceAllOrdersF16Narrow:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Narrow, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrders, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Rat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Rat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16RatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Orders:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Orders, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16Uniform:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Uniform, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformRat:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformRat, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16UniformRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16UniformRatHorner, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceSingleF16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Fast, __half, kSingle, xh);
                break;
#endif // BoysFp16
            // The removed-call half of the fast rows armed above, in the same
            // order and over the same policies: the subtraction's other side.
            case ProbeEntry::kDeviceAllOrdersF32Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Fast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32RatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32RatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32RatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32OrdersRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowMonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersMonoFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersRatFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllOrdersF32NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32NarrowOrdersRatHornerFast, float, kAllOrders, xf);
                break;
            case ProbeEntry::kDeviceAllNF32Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Fast, float, kAllN, xf);
                break;
            case ProbeEntry::kDeviceEachOrderF32Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev32Fast, float, kEachOrder, xf);
                break;
#if BoysFp16
            // The removed-call half of the lane's fast rows; see above.
            case ProbeEntry::kDeviceAllOrdersF16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Fast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16RatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16RatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16RatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRatFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16NarrowOrdersRatHornerFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllNF16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Fast, __half, kAllN, xh);
                break;
            case ProbeEntry::kDeviceEachOrderF16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Fast, __half, kEachOrder, xh);
                break;
            // The removed-call half of the coarsest partition's monomial rows above, in
            // the same order and over the same policies: the subtraction's other side.
            case ProbeEntry::kDeviceAllOrdersF16Mono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16Mono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16MonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16MonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersMono, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersF16OrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(Dev16OrdersMonoFast, __half, kAllOrders, xh);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Mono:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Mono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16MonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16MonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            // The removed-call half of the bfloat16 rows above, in the same order
            // and over the same policies: the subtraction's other side.
            case ProbeEntry::kDeviceSingleBf16:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16, __nv_bfloat16, kSingle, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllNBf16:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16, __nv_bfloat16, kAllN, xb);
                break;
            case ProbeEntry::kDeviceEachOrderBf16:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16, __nv_bfloat16, kEachOrder, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Orders:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Orders, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Narrow:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Narrow, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrders:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrders, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Uniform:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Uniform, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowMono:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersMono:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersMono, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16UniformHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Rat:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Rat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRat:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRat:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformRat:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16UniformRat, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16RatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16UniformRatHorner:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16UniformRatHorner, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceSingleBf16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Fast, __nv_bfloat16, kSingle, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Fast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16RatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16RatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16RatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16OrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16OrdersRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersMonoFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersMonoFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersRatFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllOrdersBf16NarrowOrdersRatHornerFast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16NarrowOrdersRatHornerFast, __nv_bfloat16, kAllOrders, xb);
                break;
            case ProbeEntry::kDeviceAllNBf16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Fast, __nv_bfloat16, kAllN, xb);
                break;
            case ProbeEntry::kDeviceEachOrderBf16Fast:
                BOYS_PROBE_LAUNCH_PLAIN(DevBf16Fast, __nv_bfloat16, kEachOrder, xb);
                break;
#endif // BoysFp16
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
int BoysCudaLaunchAllOrdersF64Fast(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64MonoFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersMonoFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowMonoFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64RatFast(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64OrdersRatFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowRatFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRatFast(
    int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchEachOrderF64(int, const int*, const double*, const int*, double*, std::size_t,
                               void*);
int BoysCudaLaunchEachOrderF64Fast(int, const int*, const double*, const int*, double*, std::size_t,
                                   void*);
int BoysCudaLaunchEachOrderF32(int, const int*, const double*, const int*, float*, std::size_t,
                               void*);
int BoysCudaLaunchEachOrderF32Fast(int, const int*, const double*, const int*, float*, std::size_t,
                                   void*);
// The float lane's ladders at the other region-B exponential, one per kernel this lane has: the
// same bodies as the launchers above with the seed's exponential moved, which is why the member is
// a second symbol and not a parameter.
int BoysCudaLaunchAllOrdersF32Fast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32MonoFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersMonoFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowMonoFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMonoFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32RatFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRatFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRatFast(
    int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRatFast(
    int, const int*, const double*, float*, std::size_t, void*);
// The single and all-N shapes at that exponential, the double lane's beside it.
int BoysCudaLaunchSingleF64Fast(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF64Fast(int, int, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllNF32Fast(int, int, const double*, float*, std::size_t, void*);
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
int BoysCudaLaunchEachOrderF16(int, const int*, const void*, const int*, void*, std::size_t, void*);
int BoysCudaLaunchEachOrderF16Fast(
    int, const int*, const void*, const int*, void*, std::size_t, void*);
// The half lane's ladders at the other region-B exponential, one per kernel this lane has, as the
// float lane's block above is declared.
int BoysCudaLaunchAllOrdersF16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16MonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersMonoFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowMonoFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersMonoFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16RatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersRatFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowRatFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersRatFast(
    int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16Fast(int, int, const void*, void*, std::size_t, void*);
// The bfloat16 lane's launched launchers, one per symbol the rows above reach.
int BoysCudaLaunchAllNBf16(int, int, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNBf16Fast(int, int, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Mono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16MonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Narrow(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowMono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrders(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersMono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowRatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Orders(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersMono(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersRatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersUniform(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersUniformHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersUniformRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersUniformRatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Rat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16RatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16RatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16RatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Uniform(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16UniformHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16UniformRat(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16UniformRatHorner(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchEachOrderBf16(
    int, const int*, const void*, const int*, void*, std::size_t, void*);
int BoysCudaLaunchEachOrderBf16Fast(
    int, const int*, const void*, const int*, void*, std::size_t, void*);
int BoysCudaLaunchSingleBf16(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchSingleBf16Fast(int, const int*, const void*, void*, std::size_t, void*);
#endif // BoysFp16
}

// The each-order arms' placement array: the index each argument's ladder starts at, which is the
// probe's own choice rather than the library's. It builds one without a prefix sum -
// offset[i] = i * (kMaxBoysOrder + 1) - which is the placement the padded all-orders rows write, so
// the ladders do not overlap and each lands inside the out block the arms are handed.
int* EachOrderOffsets(std::size_t count) {
    static int* offsets = nullptr;
    static std::size_t capacity = 0;

    if (count > capacity)
    {
        if (offsets != nullptr)
        {
            cudaFree(offsets);
            offsets = nullptr;
            capacity = 0;
        }

        if (count == 0)
        {
            return nullptr;
        }

        if (cudaMalloc(reinterpret_cast<void**>(&offsets), count * sizeof(int)) != cudaSuccess)
        {
            offsets = nullptr;
            return nullptr;
        }

        capacity = count;

        std::vector<int> host(count);

        for (std::size_t i = 0; i < count; ++i)
        {
            host[i] = static_cast<int>(i) * (kMaxBoysOrder + 1);
        }

        const cudaError_t copied = cudaMemcpy(offsets, host.data(), count * sizeof(int),
                                              cudaMemcpyHostToDevice);

        if (copied != cudaSuccess)
        {
            cudaFree(offsets);
            offsets = nullptr;
            capacity = 0;
            return nullptr;
        }
    }

    return offsets;
}

int LaunchLaunched(boys::DivisionForm form,
                   ProbeEntry entry,
                   const int* n,
                   const double* x,
                   const void* xh,
                   const void* xb,
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
        // The single shape at the lane's other region-B exponential, the arm
        // above at the row's own member: one launcher, one kernel instantiation,
        // and the same tables.
        case ProbeEntry::kSingleF64Fast:
            BoysCudaLaunchSingleF64Fast(static_cast<int>(form),
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
        // The double lane's ladders at the other region-B exponential, one arm per row the table
        // carries, in the table's own order. Each is the arm above it at the second member of the
        // axis: one launcher per kernel, over the same tables.
        case ProbeEntry::kAllOrdersF64Fast:
            BoysCudaLaunchAllOrdersF64Fast(static_cast<int>(form),
                                           n,
                                           x,
                                           static_cast<double*>(out),
                                           count,
                                           stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersFast:
            BoysCudaLaunchAllOrdersF64OrdersFast(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowFast:
            BoysCudaLaunchAllOrdersF64NarrowFast(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<double*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersFast:
            BoysCudaLaunchAllOrdersF64NarrowOrdersFast(static_cast<int>(form),
                                                       n,
                                                       x,
                                                       static_cast<double*>(out),
                                                       count,
                                                       stream);
            break;
        case ProbeEntry::kAllOrdersF64MonoFast:
            BoysCudaLaunchAllOrdersF64MonoFast(static_cast<int>(form),
                                               n,
                                               x,
                                               static_cast<double*>(out),
                                               count,
                                               stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersMonoFast:
            BoysCudaLaunchAllOrdersF64OrdersMonoFast(static_cast<int>(form),
                                                     n,
                                                     x,
                                                     static_cast<double*>(out),
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowMonoFast:
            BoysCudaLaunchAllOrdersF64NarrowMonoFast(static_cast<int>(form),
                                                     n,
                                                     x,
                                                     static_cast<double*>(out),
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersMonoFast:
            BoysCudaLaunchAllOrdersF64NarrowOrdersMonoFast(static_cast<int>(form),
                                                           n,
                                                           x,
                                                           static_cast<double*>(out),
                                                           count,
                                                           stream);
            break;
        case ProbeEntry::kAllOrdersF64RatFast:
            BoysCudaLaunchAllOrdersF64RatFast(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<double*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF64RatHornerFast:
            BoysCudaLaunchAllOrdersF64RatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<double*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersRatFast:
            BoysCudaLaunchAllOrdersF64OrdersRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<double*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF64OrdersRatHornerFast:
            BoysCudaLaunchAllOrdersF64OrdersRatFast(static_cast<int>(form),
                                                          n,
                                                          x,
                                                          static_cast<double*>(out),
                                                          count,
                                                          stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowRatFast:
            BoysCudaLaunchAllOrdersF64NarrowRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<double*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowRatHornerFast:
            BoysCudaLaunchAllOrdersF64NarrowRatFast(static_cast<int>(form),
                                                          n,
                                                          x,
                                                          static_cast<double*>(out),
                                                          count,
                                                          stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersRatFast:
            BoysCudaLaunchAllOrdersF64NarrowOrdersRatFast(static_cast<int>(form),
                                                          n,
                                                          x,
                                                          static_cast<double*>(out),
                                                          count,
                                                          stream);
            break;
        case ProbeEntry::kAllOrdersF64NarrowOrdersRatHornerFast:
            BoysCudaLaunchAllOrdersF64NarrowOrdersRatFast(static_cast<int>(form),
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
        // The uniform route's four rows, whose table is stored at one degree for every order and every
        // interval: the route has one arithmetic and no shorter image of it. The four names are two
        // launchers - the scheme axis is a real choice on this route and the packing axis is not.
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
        // The float lane's ladders at the other region-B exponential, one arm per row the table
        // carries, in the double lane's order above. Each is the arm of its own name at the second
        // member of the axis: one launcher per kernel, over the same lane, the same pieces and the
        // same tables.
        case ProbeEntry::kAllOrdersF32Fast:
            BoysCudaLaunchAllOrdersF32Fast(static_cast<int>(form),
                                           n,
                                           x,
                                           static_cast<float*>(out),
                                           count,
                                           stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersFast:
            BoysCudaLaunchAllOrdersF32OrdersFast(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowFast:
            BoysCudaLaunchAllOrdersF32NarrowFast(static_cast<int>(form),
                                                 n,
                                                 x,
                                                 static_cast<float*>(out),
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersFast:
            BoysCudaLaunchAllOrdersF32NarrowOrdersFast(static_cast<int>(form),
                                                       n,
                                                       x,
                                                       static_cast<float*>(out),
                                                       count,
                                                       stream);
            break;
        case ProbeEntry::kAllOrdersF32MonoFast:
            BoysCudaLaunchAllOrdersF32MonoFast(static_cast<int>(form),
                                               n,
                                               x,
                                               static_cast<float*>(out),
                                               count,
                                               stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersMonoFast:
            BoysCudaLaunchAllOrdersF32OrdersMonoFast(static_cast<int>(form),
                                                     n,
                                                     x,
                                                     static_cast<float*>(out),
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowMonoFast:
            BoysCudaLaunchAllOrdersF32NarrowMonoFast(static_cast<int>(form),
                                                     n,
                                                     x,
                                                     static_cast<float*>(out),
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersMonoFast:
            BoysCudaLaunchAllOrdersF32NarrowOrdersMonoFast(static_cast<int>(form),
                                                           n,
                                                           x,
                                                           static_cast<float*>(out),
                                                           count,
                                                           stream);
            break;
        case ProbeEntry::kAllOrdersF32RatFast:
            BoysCudaLaunchAllOrdersF32RatFast(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32RatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF32RatFast(static_cast<int>(form),
                                              n,
                                              x,
                                              static_cast<float*>(out),
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersRatFast:
            BoysCudaLaunchAllOrdersF32OrdersRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF32OrdersRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF32OrdersRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowRatFast:
            BoysCudaLaunchAllOrdersF32NarrowRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF32NarrowRatFast(static_cast<int>(form),
                                                    n,
                                                    x,
                                                    static_cast<float*>(out),
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersRatFast:
            BoysCudaLaunchAllOrdersF32NarrowOrdersRatFast(static_cast<int>(form),
                                                          n,
                                                          x,
                                                          static_cast<float*>(out),
                                                          count,
                                                          stream);
            break;
        case ProbeEntry::kAllOrdersF32NarrowOrdersRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF32NarrowOrdersRatFast(static_cast<int>(form),
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
        // The launched each-order rows. The offsets are this probe's own placement and not the
        // library's figure (EachOrderOffsets); a failed allocation leaves the arm silent, which is
        // the same treatment every other arm of this switch gives a launch it cannot make.
        case ProbeEntry::kEachOrderF64: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF64(static_cast<int>(form),
                                           n,
                                           x,
                                           offset,
                                           static_cast<double*>(out),
                                           count,
                                           stream);
            }

            break;
        }
        case ProbeEntry::kEachOrderF64Fast: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF64Fast(static_cast<int>(form),
                                               n,
                                               x,
                                               offset,
                                               static_cast<double*>(out),
                                               count,
                                               stream);
            }

            break;
        }
        case ProbeEntry::kEachOrderF32: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF32(static_cast<int>(form),
                                           n,
                                           x,
                                           offset,
                                           static_cast<float*>(out),
                                           count,
                                           stream);
            }

            break;
        }
        case ProbeEntry::kEachOrderF32Fast: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF32Fast(static_cast<int>(form),
                                               n,
                                               x,
                                               offset,
                                               static_cast<float*>(out),
                                               count,
                                               stream);
            }

            break;
        }
        case ProbeEntry::kAllNF64:
            BoysCudaLaunchAllNF64(static_cast<int>(form),
                                  nmax,
                                  x,
                                  static_cast<double*>(out),
                                  count,
                                  stream);
            break;
        // The all-N shape at the lane's other region-B exponential, the arm above
        // at the row's own member.
        case ProbeEntry::kAllNF64Fast:
            BoysCudaLaunchAllNF64Fast(static_cast<int>(form),
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
        case ProbeEntry::kAllNF32Fast:
            BoysCudaLaunchAllNF32Fast(static_cast<int>(form),
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
        // The half lane's ladders at the other region-B exponential, one arm per row the table
        // carries, in the float lane's order above. Each is the arm of its own name at the second
        // member of the axis, and the two scheme names of a rational pair reach the one member as
        // their accurate rows do.
        case ProbeEntry::kAllOrdersF16Fast:
            BoysCudaLaunchAllOrdersF16Fast(static_cast<int>(form),
                                           n,
                                           xh,
                                           out,
                                           count,
                                           stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersFast:
            BoysCudaLaunchAllOrdersF16OrdersFast(static_cast<int>(form),
                                                 n,
                                                 xh,
                                                 out,
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowFast:
            BoysCudaLaunchAllOrdersF16NarrowFast(static_cast<int>(form),
                                                 n,
                                                 xh,
                                                 out,
                                                 count,
                                                 stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersFast:
            BoysCudaLaunchAllOrdersF16NarrowOrdersFast(static_cast<int>(form),
                                                       n,
                                                       xh,
                                                       out,
                                                       count,
                                                       stream);
            break;
        case ProbeEntry::kAllOrdersF16MonoFast:
            BoysCudaLaunchAllOrdersF16MonoFast(static_cast<int>(form),
                                               n,
                                               xh,
                                               out,
                                               count,
                                               stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersMonoFast:
            BoysCudaLaunchAllOrdersF16OrdersMonoFast(static_cast<int>(form),
                                                     n,
                                                     xh,
                                                     out,
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowMonoFast:
            BoysCudaLaunchAllOrdersF16NarrowMonoFast(static_cast<int>(form),
                                                     n,
                                                     xh,
                                                     out,
                                                     count,
                                                     stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersMonoFast:
            BoysCudaLaunchAllOrdersF16NarrowOrdersMonoFast(static_cast<int>(form),
                                                           n,
                                                           xh,
                                                           out,
                                                           count,
                                                           stream);
            break;
        case ProbeEntry::kAllOrdersF16RatFast:
            BoysCudaLaunchAllOrdersF16RatFast(static_cast<int>(form),
                                              n,
                                              xh,
                                              out,
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF16RatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF16RatFast(static_cast<int>(form),
                                              n,
                                              xh,
                                              out,
                                              count,
                                              stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersRatFast:
            BoysCudaLaunchAllOrdersF16OrdersRatFast(static_cast<int>(form),
                                                    n,
                                                    xh,
                                                    out,
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF16OrdersRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF16OrdersRatFast(static_cast<int>(form),
                                                    n,
                                                    xh,
                                                    out,
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowRatFast:
            BoysCudaLaunchAllOrdersF16NarrowRatFast(static_cast<int>(form),
                                                    n,
                                                    xh,
                                                    out,
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF16NarrowRatFast(static_cast<int>(form),
                                                    n,
                                                    xh,
                                                    out,
                                                    count,
                                                    stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersRatFast:
            BoysCudaLaunchAllOrdersF16NarrowOrdersRatFast(static_cast<int>(form),
                                                          n,
                                                          xh,
                                                          out,
                                                          count,
                                                          stream);
            break;
        case ProbeEntry::kAllOrdersF16NarrowOrdersRatHornerFast:
            // The other scheme name of the pair above, over the same kernel.
            BoysCudaLaunchAllOrdersF16NarrowOrdersRatFast(static_cast<int>(form),
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
        case ProbeEntry::kAllNF16Fast:
            // The lane's other region-B exponential, and behind the same seam
            // this arm is: a build with the seam closed has no such enumerator
            // to be asked for.
            BoysCudaLaunchAllNF16Fast(static_cast<int>(form),
                                      nmax,
                                      xh,
                                      out,
                                      count,
                                      stream);
            break;
        // The half lane's launched each-order rows, at the probe's own placement (see
        // EachOrderOffsets).
        case ProbeEntry::kEachOrderF16: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF16(
                    static_cast<int>(form), n, xh, offset, out, count, stream);
            }

            break;
        }
        case ProbeEntry::kEachOrderF16Fast: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderF16Fast(
                    static_cast<int>(form), n, xh, offset, out, count, stream);
            }

            break;
        }
        // The lane's partition and route bodies, one arm per entry of the option table that names one, in
        // the table's own order. Two entries a scheme pair reaches share a launcher as they share a
        // kernel, so what is timed is the arithmetic the entry runs rather than a second name for it.
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
        // The bfloat16 lane's launched rows, in the table's own order: the same bodies
        // as the fp16 arms above over this lane's argument array, which is a second array
        // rather than a reinterpretation of the half one.
        case ProbeEntry::kSingleBf16:
            BoysCudaLaunchSingleBf16(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16:
            BoysCudaLaunchAllOrdersBf16(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16Narrow:
            BoysCudaLaunchAllOrdersBf16Narrow(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowMono:
            BoysCudaLaunchAllOrdersBf16NarrowMono(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16Uniform:
            BoysCudaLaunchAllOrdersBf16Uniform(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16UniformHorner:
            BoysCudaLaunchAllOrdersBf16UniformHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16Rat:
            BoysCudaLaunchAllOrdersBf16Rat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16RatHorner:
            BoysCudaLaunchAllOrdersBf16RatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowRat:
            BoysCudaLaunchAllOrdersBf16NarrowRat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowRatHorner:
            BoysCudaLaunchAllOrdersBf16NarrowRatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16UniformRat:
            BoysCudaLaunchAllOrdersBf16UniformRat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16UniformRatHorner:
            BoysCudaLaunchAllOrdersBf16UniformRatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16Orders:
            BoysCudaLaunchAllOrdersBf16Orders(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrders:
            BoysCudaLaunchAllOrdersBf16NarrowOrders(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersMono:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersMono(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersRat:
            BoysCudaLaunchAllOrdersBf16OrdersRat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersRatHorner:
            BoysCudaLaunchAllOrdersBf16OrdersRatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersRat:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersRat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersRatHorner:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersUniform:
            BoysCudaLaunchAllOrdersBf16OrdersUniform(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersUniformHorner:
            BoysCudaLaunchAllOrdersBf16OrdersUniformHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersUniformRat:
            BoysCudaLaunchAllOrdersBf16OrdersUniformRat(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersUniformRatHorner:
            BoysCudaLaunchAllOrdersBf16OrdersUniformRatHorner(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllNBf16:
            BoysCudaLaunchAllNBf16(static_cast<int>(form),
                                   nmax,
                                   xb,
                                   out,
                                   count,
                                   stream);
            break;
        case ProbeEntry::kAllOrdersBf16Mono:
            BoysCudaLaunchAllOrdersBf16Mono(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersMono:
            BoysCudaLaunchAllOrdersBf16OrdersMono(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kSingleBf16Fast:
            BoysCudaLaunchSingleBf16Fast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kEachOrderBf16: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderBf16(
                    static_cast<int>(form), n, xb, offset, out, count, stream);
            }

            break;
        }
        case ProbeEntry::kEachOrderBf16Fast: {
            int* const offset = EachOrderOffsets(count);

            if (offset != nullptr)
            {
                BoysCudaLaunchEachOrderBf16Fast(
                    static_cast<int>(form), n, xb, offset, out, count, stream);
            }

            break;
        }
        case ProbeEntry::kAllOrdersBf16Fast:
            BoysCudaLaunchAllOrdersBf16Fast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersFast:
            BoysCudaLaunchAllOrdersBf16OrdersFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowFast:
            BoysCudaLaunchAllOrdersBf16NarrowFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersFast:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16MonoFast:
            BoysCudaLaunchAllOrdersBf16MonoFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersMonoFast:
            BoysCudaLaunchAllOrdersBf16OrdersMonoFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowMonoFast:
            BoysCudaLaunchAllOrdersBf16NarrowMonoFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersMonoFast:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersMonoFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16RatFast:
            BoysCudaLaunchAllOrdersBf16RatFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16RatHornerFast:
            BoysCudaLaunchAllOrdersBf16RatHornerFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersRatFast:
            BoysCudaLaunchAllOrdersBf16OrdersRatFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16OrdersRatHornerFast:
            BoysCudaLaunchAllOrdersBf16OrdersRatHornerFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowRatFast:
            BoysCudaLaunchAllOrdersBf16NarrowRatFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowRatHornerFast:
            BoysCudaLaunchAllOrdersBf16NarrowRatHornerFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersRatFast:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersRatFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllOrdersBf16NarrowOrdersRatHornerFast:
            BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHornerFast(static_cast<int>(form), n, xb, out, count, stream);
            break;
        case ProbeEntry::kAllNBf16Fast:
            BoysCudaLaunchAllNBf16Fast(static_cast<int>(form),
                                       nmax,
                                       xb,
                                       out,
                                       count,
                                       stream);
            break;
#else
        // As in LaunchInKernel: with the seam closed the fp16 arms are the only
        // ones that would have read a half array.
        (void)xh;
        (void)xb;
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
                          request.xb,
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
                static_cast<const __nv_bfloat16*>(request.xb),
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
                static_cast<const __nv_bfloat16*>(request.xb),
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
                static_cast<const __nv_bfloat16*>(request.xb),
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
