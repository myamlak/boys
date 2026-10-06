// The device half of the axis-sensitivity check: the caller's own kernel with one entry of
// boys_cuda_device.hpp in it, launched over the same arguments the host half runs the launched
// entries over.
//
// Why this translation unit exists at all. The check compares two things that a row of the
// option table claims differ in one axis, and 168 of that table's 352 rows are entries of the
// in-kernel surface, which no host translation unit can call: they are __device__ functions.
// The alternative — one .cpp compiled as CUDA — cannot reach the launched half either, because
// the public header it would call through (boys/boys_cuda.hpp) is C++23 and nvcc does not take
// it (measured: nvcc 13.3 with this MSVC rejects boys.hpp's tables). And the one way a CUDA
// translation unit could reach the launched entries — restating the library's positional table
// address order and its launch ABI here — would make this file a third statement of an order
// the library's own two files already call out for being stated twice. So the split is the same
// one tests/boys_cuda_accuracy_gate.cpp and tests/boys_cuda_device_demo.cu already make: the
// host half owns the option table, the arguments and the comparison, and this file owns the
// kernels and nothing else. The tables are handed in as the value BoysCuda::DeviceTables
// filled, which is the public way a consumer obtains them.
//
// C++20 with a CUDA-safe include list only, as the lane's own .cu is.
//
// Not a source the registered target compiles alone: it is one test, registered beside the
// other boys-cuda-* targets in CMakeLists.txt, and the host half declares its one entry point.

#include "boys/boys_cuda_device.hpp"
#include "boys/boys_cuda_options.hpp" // the shape a row states, and the entry it names

#include <cstddef>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace {

using boys::BoysDeviceTables;
using boys::DeviceEntry;
using boys::DivisionForm;
using boys::RegionBExp;

// The width of one argument's block in the caller's array, the tallest ladder any shape here
// writes: the host half calls an entry over arguments whose top order is kMaxBoysOrder.
constexpr int kLadder = boys::kMaxBoysOrder + 1;

// The common top order the all-n shape is asked for, which is a template argument of that entry:
// the same top order the host half's argument set carries.
constexpr int kTopOrder = boys::kMaxBoysOrder;

constexpr int kThreadsPerBlock = 128;

// The half lane's two formats on the device side are the toolkit's own: the in-kernel entries take
// __half and __nv_bfloat16, where the launched entries take this library's F16 and Bf16. The host
// half of this test fills one array per format and the device reads it as the type its arm names.
static_assert(sizeof(__half) == 2 && sizeof(__nv_bfloat16) == 2,
              "a half value is one 16-bit pattern on both sides of the boundary");

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + kThreadsPerBlock - 1) / kThreadsPerBlock);
}

__device__ __forceinline__ std::size_t Thread() {
    return static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

// ---------------------------------------------------------------------------
// One kernel per shape, the entry chosen inside it by the value the host half passes
// ---------------------------------------------------------------------------
//
// The switch is over the enumerator the option table states, so the arm a row reaches is the
// arm its own entry names, and a row with no arm is one this file does not run - which the
// host half prints and fails on rather than counting as covered. Every arm calls the device
// entry the row names; the compiler may inline it or not, and what is compared is the values
// written either way.
//
// The arguments are read from the caller's arrays: x is the double array for the two wider
// lanes and the F16 array for the half lane, which is what those entries take.

template <DivisionForm F>
__global__ void SingleKernel(const BoysDeviceTables tables,
                             int entry,
                             const int* order,
                             const double* x,
                             const __half* x16,
                             const __nv_bfloat16* xbf,
                             void* out,
                             std::size_t count) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    switch (static_cast<DeviceEntry>(entry))
    {
    case boys::DeviceEntry::kDeviceSingleF64:
        BoysDeviceSingleF64<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleF32:
        BoysDeviceSingleF32<F, RegionBExp::kAccurate>(
            tables, order[i], static_cast<float>(x[i]), static_cast<float*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleF32Fast:
        BoysDeviceSingleF32<F, RegionBExp::kFast>(
            tables, order[i], static_cast<float>(x[i]), static_cast<float*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleF16:
        BoysDeviceSingleF16<F>(tables, order[i], x16[i], static_cast<__half*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleF16Fast:
        BoysDeviceSingleF16Fast<F>(tables, order[i], x16[i], static_cast<__half*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleF64Fast:
        BoysDeviceSingleF64<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleBf16:
        BoysDeviceSingleBf16<F>(tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i);
        break;
    case boys::DeviceEntry::kDeviceSingleBf16Fast:
        BoysDeviceSingleBf16Fast<F>(tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i);
        break;
    default:
        break;
    }
}

template <DivisionForm F>
__global__ void LadderKernel(const BoysDeviceTables tables,
                             int entry,
                             const int* order,
                             const double* x,
                             const __half* x16,
                             const __nv_bfloat16* xbf,
                             void* out,
                             std::size_t count) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    switch (static_cast<DeviceEntry>(entry))
    {
    case boys::DeviceEntry::kDeviceAllOrdersF64:
        BoysDeviceAllOrdersF64<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32:
        BoysDeviceAllOrdersF32<F, RegionBExp::kAccurate>(tables,
                                                         order[i],
                                                         static_cast<float>(x[i]),
                                                         static_cast<float*>(out) + i * kLadder,
                                                         kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16:
        BoysDeviceAllOrdersF16<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Narrow:
        BoysDeviceAllOrdersF64Narrow<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowMono:
        BoysDeviceAllOrdersF64NarrowMono<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Rat:
        BoysDeviceAllOrdersF64Rat<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64RatHorner:
        BoysDeviceAllOrdersF64RatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowRat:
        BoysDeviceAllOrdersF64NarrowRat<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner:
        BoysDeviceAllOrdersF64NarrowRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Uniform:
        BoysDeviceAllOrdersF64Uniform<F>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64UniformHorner:
        BoysDeviceAllOrdersF64UniformHorner<F>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64UniformRat:
        BoysDeviceAllOrdersF64UniformRat<F>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64UniformRatHorner:
        BoysDeviceAllOrdersF64UniformRatHorner<F>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Narrow:
        BoysDeviceAllOrdersF32Narrow<F, RegionBExp::kAccurate>(tables,
                                                               order[i],
                                                               static_cast<float>(x[i]),
                                                               static_cast<float*>(out) +
                                                                   i * kLadder,
                                                               kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowMono:
        BoysDeviceAllOrdersF32NarrowMono<F, RegionBExp::kAccurate>(tables,
                                                                   order[i],
                                                                   static_cast<float>(x[i]),
                                                                   static_cast<float*>(out) +
                                                                       i * kLadder,
                                                                   kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Rat:
        BoysDeviceAllOrdersF32Rat<F, RegionBExp::kAccurate>(tables,
                                                            order[i],
                                                            static_cast<float>(x[i]),
                                                            static_cast<float*>(out) + i * kLadder,
                                                            kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32RatHorner:
        BoysDeviceAllOrdersF32RatHorner<F, RegionBExp::kAccurate>(tables,
                                                                  order[i],
                                                                  static_cast<float>(x[i]),
                                                                  static_cast<float*>(out) +
                                                                      i * kLadder,
                                                                  kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowRat:
        BoysDeviceAllOrdersF32NarrowRat<F, RegionBExp::kAccurate>(tables,
                                                                  order[i],
                                                                  static_cast<float>(x[i]),
                                                                  static_cast<float*>(out) +
                                                                      i * kLadder,
                                                                  kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner:
        BoysDeviceAllOrdersF32NarrowRatHorner<F, RegionBExp::kAccurate>(tables,
                                                                        order[i],
                                                                        static_cast<float>(x[i]),
                                                                        static_cast<float*>(out) +
                                                                            i * kLadder,
                                                                        kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Uniform:
        BoysDeviceAllOrdersF32Uniform<F>(tables,
                                         order[i],
                                         static_cast<float>(x[i]),
                                         static_cast<float*>(out) + i * kLadder,
                                         kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32UniformHorner:
        BoysDeviceAllOrdersF32UniformHorner<F>(tables,
                                               order[i],
                                               static_cast<float>(x[i]),
                                               static_cast<float*>(out) + i * kLadder,
                                               kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32UniformRat:
        BoysDeviceAllOrdersF32UniformRat<F>(tables,
                                            order[i],
                                            static_cast<float>(x[i]),
                                            static_cast<float*>(out) + i * kLadder,
                                            kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32UniformRatHorner:
        BoysDeviceAllOrdersF32UniformRatHorner<F>(tables,
                                                  order[i],
                                                  static_cast<float>(x[i]),
                                                  static_cast<float*>(out) + i * kLadder,
                                                  kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Orders:
        BoysDeviceAllOrdersF64Orders<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrders:
        BoysDeviceAllOrdersF64NarrowOrders<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMono:
        BoysDeviceAllOrdersF64NarrowOrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersRat:
        BoysDeviceAllOrdersF64OrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRat:
        BoysDeviceAllOrdersF64NarrowOrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersRatHorner:
        BoysDeviceAllOrdersF64OrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHorner:
        BoysDeviceAllOrdersF64NarrowOrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Orders:
        BoysDeviceAllOrdersF32Orders<F, RegionBExp::kAccurate>(tables,
                                                               order[i],
                                                               static_cast<float>(x[i]),
                                                               static_cast<float*>(out) +
                                                                   i * kLadder,
                                                               kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrders:
        BoysDeviceAllOrdersF32NarrowOrders<F, RegionBExp::kAccurate>(tables,
                                                                     order[i],
                                                                     static_cast<float>(x[i]),
                                                                     static_cast<float*>(out) +
                                                                         i * kLadder,
                                                                     kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMono:
        BoysDeviceAllOrdersF32NarrowOrdersMono<F, RegionBExp::kAccurate>(tables,
                                                                         order[i],
                                                                         static_cast<float>(x[i]),
                                                                         static_cast<float*>(out) +
                                                                             i * kLadder,
                                                                         kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersRat:
        BoysDeviceAllOrdersF32OrdersRat<F, RegionBExp::kAccurate>(tables,
                                                                  order[i],
                                                                  static_cast<float>(x[i]),
                                                                  static_cast<float*>(out) +
                                                                      i * kLadder,
                                                                  kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRat:
        BoysDeviceAllOrdersF32NarrowOrdersRat<F, RegionBExp::kAccurate>(tables,
                                                                        order[i],
                                                                        static_cast<float>(x[i]),
                                                                        static_cast<float*>(out) +
                                                                            i * kLadder,
                                                                        kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersRatHorner:
        BoysDeviceAllOrdersF32OrdersRatHorner<F, RegionBExp::kAccurate>(tables,
                                                                        order[i],
                                                                        static_cast<float>(x[i]),
                                                                        static_cast<float*>(out) +
                                                                            i * kLadder,
                                                                        kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHorner:
        BoysDeviceAllOrdersF32NarrowOrdersRatHorner<F, RegionBExp::kAccurate>(
            tables,
            order[i],
            static_cast<float>(x[i]),
            static_cast<float*>(out) + i * kLadder,
            kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Orders:
        BoysDeviceAllOrdersF16Orders<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Narrow:
        BoysDeviceAllOrdersF16Narrow<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrders:
        BoysDeviceAllOrdersF16NarrowOrders<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Uniform:
        BoysDeviceAllOrdersF16Uniform<F>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowMono:
        BoysDeviceAllOrdersF16NarrowMono<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMono:
        BoysDeviceAllOrdersF16NarrowOrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16UniformHorner:
        BoysDeviceAllOrdersF16UniformHorner<F>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Rat:
        BoysDeviceAllOrdersF16Rat<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersRat:
        BoysDeviceAllOrdersF16OrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowRat:
        BoysDeviceAllOrdersF16NarrowRat<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRat:
        BoysDeviceAllOrdersF16NarrowOrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16UniformRat:
        BoysDeviceAllOrdersF16UniformRat<F>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16RatHorner:
        BoysDeviceAllOrdersF16RatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersRatHorner:
        BoysDeviceAllOrdersF16OrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowRatHorner:
        BoysDeviceAllOrdersF16NarrowRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHorner:
        BoysDeviceAllOrdersF16NarrowOrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16UniformRatHorner:
        BoysDeviceAllOrdersF16UniformRatHorner<F>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Fast:
        BoysDeviceAllOrdersF32<F, RegionBExp::kFast>(tables,
                                                     order[i],
                                                     static_cast<float>(x[i]),
                                                     static_cast<float*>(out) + i * kLadder,
                                                     kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersFast:
        BoysDeviceAllOrdersF32Orders<F, RegionBExp::kFast>(tables,
                                                           order[i],
                                                           static_cast<float>(x[i]),
                                                           static_cast<float*>(out) + i * kLadder,
                                                           kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32RatFast:
        BoysDeviceAllOrdersF32Rat<F, RegionBExp::kFast>(tables,
                                                        order[i],
                                                        static_cast<float>(x[i]),
                                                        static_cast<float*>(out) + i * kLadder,
                                                        kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersRatFast:
        BoysDeviceAllOrdersF32OrdersRat<F, RegionBExp::kFast>(tables,
                                                              order[i],
                                                              static_cast<float>(x[i]),
                                                              static_cast<float*>(out) +
                                                                  i * kLadder,
                                                              kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32RatHornerFast:
        BoysDeviceAllOrdersF32RatHorner<F, RegionBExp::kFast>(tables,
                                                              order[i],
                                                              static_cast<float>(x[i]),
                                                              static_cast<float*>(out) +
                                                                  i * kLadder,
                                                              kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersRatHornerFast:
        BoysDeviceAllOrdersF32OrdersRatHorner<F, RegionBExp::kFast>(tables,
                                                                    order[i],
                                                                    static_cast<float>(x[i]),
                                                                    static_cast<float*>(out) +
                                                                        i * kLadder,
                                                                    kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowFast:
        BoysDeviceAllOrdersF32Narrow<F, RegionBExp::kFast>(tables,
                                                           order[i],
                                                           static_cast<float>(x[i]),
                                                           static_cast<float*>(out) + i * kLadder,
                                                           kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersFast:
        BoysDeviceAllOrdersF32NarrowOrders<F, RegionBExp::kFast>(tables,
                                                                 order[i],
                                                                 static_cast<float>(x[i]),
                                                                 static_cast<float*>(out) +
                                                                     i * kLadder,
                                                                 kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowMonoFast:
        BoysDeviceAllOrdersF32NarrowMono<F, RegionBExp::kFast>(tables,
                                                               order[i],
                                                               static_cast<float>(x[i]),
                                                               static_cast<float*>(out) +
                                                                   i * kLadder,
                                                               kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersMonoFast:
        BoysDeviceAllOrdersF32NarrowOrdersMono<F, RegionBExp::kFast>(tables,
                                                                     order[i],
                                                                     static_cast<float>(x[i]),
                                                                     static_cast<float*>(out) +
                                                                         i * kLadder,
                                                                     kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatFast:
        BoysDeviceAllOrdersF32NarrowRat<F, RegionBExp::kFast>(tables,
                                                              order[i],
                                                              static_cast<float>(x[i]),
                                                              static_cast<float*>(out) +
                                                                  i * kLadder,
                                                              kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatFast:
        BoysDeviceAllOrdersF32NarrowOrdersRat<F, RegionBExp::kFast>(tables,
                                                                    order[i],
                                                                    static_cast<float>(x[i]),
                                                                    static_cast<float*>(out) +
                                                                        i * kLadder,
                                                                    kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowRatHornerFast:
        BoysDeviceAllOrdersF32NarrowRatHorner<F, RegionBExp::kFast>(tables,
                                                                    order[i],
                                                                    static_cast<float>(x[i]),
                                                                    static_cast<float*>(out) +
                                                                        i * kLadder,
                                                                    kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32NarrowOrdersRatHornerFast:
        BoysDeviceAllOrdersF32NarrowOrdersRatHorner<F, RegionBExp::kFast>(tables,
                                                                          order[i],
                                                                          static_cast<float>(x[i]),
                                                                          static_cast<float*>(out) +
                                                                              i * kLadder,
                                                                          kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Fast:
        BoysDeviceAllOrdersF16<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersFast:
        BoysDeviceAllOrdersF16Orders<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16RatFast:
        BoysDeviceAllOrdersF16Rat<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersRatFast:
        BoysDeviceAllOrdersF16OrdersRat<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16RatHornerFast:
        BoysDeviceAllOrdersF16RatHorner<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersRatHornerFast:
        BoysDeviceAllOrdersF16OrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowFast:
        BoysDeviceAllOrdersF16Narrow<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersFast:
        BoysDeviceAllOrdersF16NarrowOrders<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowMonoFast:
        BoysDeviceAllOrdersF16NarrowMono<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersMonoFast:
        BoysDeviceAllOrdersF16NarrowOrdersMono<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowRatFast:
        BoysDeviceAllOrdersF16NarrowRat<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatFast:
        BoysDeviceAllOrdersF16NarrowOrdersRat<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowRatHornerFast:
        BoysDeviceAllOrdersF16NarrowRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16NarrowOrdersRatHornerFast:
        BoysDeviceAllOrdersF16NarrowOrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Fast:
        BoysDeviceAllOrdersF64<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersFast:
        BoysDeviceAllOrdersF64Orders<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowFast:
        BoysDeviceAllOrdersF64Narrow<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersFast:
        BoysDeviceAllOrdersF64NarrowOrders<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowMonoFast:
        BoysDeviceAllOrdersF64NarrowMono<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersMonoFast:
        BoysDeviceAllOrdersF64NarrowOrdersMono<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64RatFast:
        BoysDeviceAllOrdersF64Rat<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64RatHornerFast:
        BoysDeviceAllOrdersF64RatHorner<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersRatFast:
        BoysDeviceAllOrdersF64OrdersRat<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersRatHornerFast:
        BoysDeviceAllOrdersF64OrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatFast:
        BoysDeviceAllOrdersF64NarrowRat<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowRatHornerFast:
        BoysDeviceAllOrdersF64NarrowRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatFast:
        BoysDeviceAllOrdersF64NarrowOrdersRat<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64NarrowOrdersRatHornerFast:
        BoysDeviceAllOrdersF64NarrowOrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16:
        BoysDeviceAllOrdersBf16<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Orders:
        BoysDeviceAllOrdersBf16Orders<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Narrow:
        BoysDeviceAllOrdersBf16Narrow<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrders:
        BoysDeviceAllOrdersBf16NarrowOrders<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Uniform:
        BoysDeviceAllOrdersBf16Uniform<F>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowMono:
        BoysDeviceAllOrdersBf16NarrowMono<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersMono:
        BoysDeviceAllOrdersBf16NarrowOrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16UniformHorner:
        BoysDeviceAllOrdersBf16UniformHorner<F>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Rat:
        BoysDeviceAllOrdersBf16Rat<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersRat:
        BoysDeviceAllOrdersBf16OrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowRat:
        BoysDeviceAllOrdersBf16NarrowRat<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersRat:
        BoysDeviceAllOrdersBf16NarrowOrdersRat<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16UniformRat:
        BoysDeviceAllOrdersBf16UniformRat<F>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16RatHorner:
        BoysDeviceAllOrdersBf16RatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersRatHorner:
        BoysDeviceAllOrdersBf16OrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowRatHorner:
        BoysDeviceAllOrdersBf16NarrowRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersRatHorner:
        BoysDeviceAllOrdersBf16NarrowOrdersRatHorner<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16UniformRatHorner:
        BoysDeviceAllOrdersBf16UniformRatHorner<F>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Fast:
        BoysDeviceAllOrdersBf16<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersFast:
        BoysDeviceAllOrdersBf16Orders<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16RatFast:
        BoysDeviceAllOrdersBf16Rat<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersRatFast:
        BoysDeviceAllOrdersBf16OrdersRat<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16RatHornerFast:
        BoysDeviceAllOrdersBf16RatHorner<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersRatHornerFast:
        BoysDeviceAllOrdersBf16OrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowFast:
        BoysDeviceAllOrdersBf16Narrow<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersFast:
        BoysDeviceAllOrdersBf16NarrowOrders<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowMonoFast:
        BoysDeviceAllOrdersBf16NarrowMono<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersMonoFast:
        BoysDeviceAllOrdersBf16NarrowOrdersMono<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowRatFast:
        BoysDeviceAllOrdersBf16NarrowRat<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersRatFast:
        BoysDeviceAllOrdersBf16NarrowOrdersRat<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowRatHornerFast:
        BoysDeviceAllOrdersBf16NarrowRatHorner<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16NarrowOrdersRatHornerFast:
        BoysDeviceAllOrdersBf16NarrowOrdersRatHorner<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64Mono:
        BoysDeviceAllOrdersF64Mono<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64MonoFast:
        BoysDeviceAllOrdersF64Mono<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersMono:
        BoysDeviceAllOrdersF64OrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF64OrdersMonoFast:
        BoysDeviceAllOrdersF64OrdersMono<F, RegionBExp::kFast>(
            tables, order[i], x[i], static_cast<double*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32Mono:
        BoysDeviceAllOrdersF32Mono<F, RegionBExp::kAccurate>(tables,
                                                             order[i],
                                                             static_cast<float>(x[i]),
                                                             static_cast<float*>(out) + i * kLadder,
                                                             kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32MonoFast:
        BoysDeviceAllOrdersF32Mono<F, RegionBExp::kFast>(tables,
                                                         order[i],
                                                         static_cast<float>(x[i]),
                                                         static_cast<float*>(out) + i * kLadder,
                                                         kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersMono:
        BoysDeviceAllOrdersF32OrdersMono<F, RegionBExp::kAccurate>(tables,
                                                                   order[i],
                                                                   static_cast<float>(x[i]),
                                                                   static_cast<float*>(out) +
                                                                       i * kLadder,
                                                                   kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF32OrdersMonoFast:
        BoysDeviceAllOrdersF32OrdersMono<F, RegionBExp::kFast>(tables,
                                                               order[i],
                                                               static_cast<float>(x[i]),
                                                               static_cast<float*>(out) +
                                                                   i * kLadder,
                                                               kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16Mono:
        BoysDeviceAllOrdersF16Mono<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16MonoFast:
        BoysDeviceAllOrdersF16Mono<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersMono:
        BoysDeviceAllOrdersF16OrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersF16OrdersMonoFast:
        BoysDeviceAllOrdersF16OrdersMono<F, RegionBExp::kFast>(
            tables, order[i], x16[i], static_cast<__half*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16Mono:
        BoysDeviceAllOrdersBf16Mono<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16MonoFast:
        BoysDeviceAllOrdersBf16Mono<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersMono:
        BoysDeviceAllOrdersBf16OrdersMono<F, RegionBExp::kAccurate>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllOrdersBf16OrdersMonoFast:
        BoysDeviceAllOrdersBf16OrdersMono<F, RegionBExp::kFast>(
            tables, order[i], xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder, kLadder);
        break;
    default:
        break;
    }
}

template <DivisionForm F>
__global__ void AllNKernel(const BoysDeviceTables tables,
                           int entry,
                           const double* x,
                           const __half* x16,
                           const __nv_bfloat16* xbf,
                           void* out,
                           std::size_t count) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    switch (static_cast<DeviceEntry>(entry))
    {
    case boys::DeviceEntry::kDeviceAllNF64:
        BoysDeviceAllNF64<F, kTopOrder, RegionBExp::kAccurate>(
            tables, x[i], static_cast<double*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNF32:
        BoysDeviceAllNF32<F, kTopOrder, RegionBExp::kAccurate>(
            tables, static_cast<float>(x[i]), static_cast<float*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNF16:
        BoysDeviceAllNF16<F, kTopOrder, RegionBExp::kAccurate>(
            tables, x16[i], static_cast<__half*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNF32Fast:
        BoysDeviceAllNF32<F, kTopOrder, RegionBExp::kFast>(
            tables, static_cast<float>(x[i]), static_cast<float*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNF16Fast:
        BoysDeviceAllNF16<F, kTopOrder, RegionBExp::kFast>(
            tables, x16[i], static_cast<__half*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNF64Fast:
        BoysDeviceAllNF64<F, kTopOrder, RegionBExp::kFast>(
            tables, x[i], static_cast<double*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNBf16:
        BoysDeviceAllNBf16<F, kTopOrder, RegionBExp::kAccurate>(
            tables, xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder);
        break;
    case boys::DeviceEntry::kDeviceAllNBf16Fast:
        BoysDeviceAllNBf16<F, kTopOrder, RegionBExp::kFast>(
            tables, xbf[i], static_cast<__nv_bfloat16*>(out) + i * kLadder);
        break;
    default:
        break;
    }
}

template <DivisionForm F>
__global__ void EachOrderKernel(const BoysDeviceTables tables,
                                int entry,
                                const int* order,
                                const double* x,
                                const __half* x16,
                                const __nv_bfloat16* xbf,
                                void* out,
                                std::size_t count) {
    const std::size_t i = Thread();

    if (i >= count)
    {
        return;
    }

    switch (static_cast<DeviceEntry>(entry))
    {
    case boys::DeviceEntry::kDeviceEachOrderF64: {
        auto sink = [&](int l, double v) { static_cast<double*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF64<F, decltype(sink), RegionBExp::kAccurate>(
            tables, order[i], x[i], sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderF32: {
        auto sink = [&](int l, float v) { static_cast<float*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF32<F, decltype(sink), RegionBExp::kAccurate>(
            tables, order[i], static_cast<float>(x[i]), sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderF16: {
        auto sink = [&](int l, __half v) { static_cast<__half*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF16<F, decltype(sink), RegionBExp::kAccurate>(
            tables, order[i], x16[i], sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderF32Fast: {
        auto sink = [&](int l, float v) { static_cast<float*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF32<F, decltype(sink), RegionBExp::kFast>(
            tables, order[i], static_cast<float>(x[i]), sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderF16Fast: {
        auto sink = [&](int l, __half v) { static_cast<__half*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF16<F, decltype(sink), RegionBExp::kFast>(
            tables, order[i], x16[i], sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderF64Fast: {
        auto sink = [&](int l, double v) { static_cast<double*>(out)[i * kLadder + l] = v; };
        BoysDeviceEachOrderF64<F, decltype(sink), RegionBExp::kFast>(tables, order[i], x[i], sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderBf16: {
        auto sink = [&](int l, __nv_bfloat16 v) {
            static_cast<__nv_bfloat16*>(out)[i * kLadder + l] = v;
        };
        BoysDeviceEachOrderBf16<F, decltype(sink), RegionBExp::kAccurate>(
            tables, order[i], xbf[i], sink);
        break;
    }
    case boys::DeviceEntry::kDeviceEachOrderBf16Fast: {
        auto sink = [&](int l, __nv_bfloat16 v) {
            static_cast<__nv_bfloat16*>(out)[i * kLadder + l] = v;
        };
        BoysDeviceEachOrderBf16<F, decltype(sink), RegionBExp::kFast>(
            tables, order[i], xbf[i], sink);
        break;
    }
    default:
        break;
    }
}

// ---------------------------------------------------------------------------
// The launch, one template instantiation per division form
// ---------------------------------------------------------------------------

template <DivisionForm F>
int Launch(int entry,
           int shape,
           const BoysDeviceTables* tables,
           int nmax,
           const int* order,
           const double* x,
           const __half* x16,
           const __nv_bfloat16* xbf,
           void* out,
           std::size_t count) {
    (void)nmax;

    switch (static_cast<boys::DeviceOptionShape>(shape))
    {
    case boys::DeviceOptionShape::kSingle:
        SingleKernel<F>
            <<<Blocks(count), kThreadsPerBlock>>>(*tables, entry, order, x, x16, xbf, out, count);
        break;
    case boys::DeviceOptionShape::kAllOrders:
        LadderKernel<F>
            <<<Blocks(count), kThreadsPerBlock>>>(*tables, entry, order, x, x16, xbf, out, count);
        break;
    case boys::DeviceOptionShape::kAllN:
        AllNKernel<F><<<Blocks(count), kThreadsPerBlock>>>(*tables, entry, x, x16, xbf, out, count);
        break;
    case boys::DeviceOptionShape::kEachOrder:
        EachOrderKernel<F>
            <<<Blocks(count), kThreadsPerBlock>>>(*tables, entry, order, x, x16, xbf, out, count);
        break;
    default:
        return 1;
    }

    return static_cast<int>(cudaGetLastError());
}

} // namespace

/// Runs one in-kernel entry of the option table over the host half's arguments, on the card.
///
/// \param entry  the row's enumerator, as \c DeviceOptionInfo::entry states it
/// \param form   the division form to run, as \c DivisionForm states it
/// \param shape  the row's shape, as \c DeviceOptionInfo::shape states it
/// \param tables the handle \c BoysCuda::DeviceTables filled
/// \param nmax   the common top order the all-n shape is asked for
/// \param order  device array of per-argument orders
/// \param x      device array of arguments, double
/// \param x16    device array of the same arguments in the half lane's format
/// \param offset unused; the each-order shape's offsets are the argument's own block here
/// \param out    device array the entry writes its values into
/// \param count  number of arguments
///
/// \returns 0 when the launch was accepted and the entry ran, a nonzero code otherwise
extern "C" int BoysAxisRunDevice(int entry,
                                 int form,
                                 int shape,
                                 const BoysDeviceTables* tables,
                                 int nmax,
                                 const int* order,
                                 const double* x,
                                 const void* x16,
                                 const void* xbf,
                                 const int* offset,
                                 void* out,
                                 std::size_t count) {
    (void)offset;

    if (tables == nullptr || order == nullptr || x == nullptr || x16 == nullptr || xbf == nullptr ||
        out == nullptr)
    {
        return 1;
    }

    const __half* half = static_cast<const __half*>(x16);
    const __nv_bfloat16* wide = static_cast<const __nv_bfloat16*>(xbf);

    switch (static_cast<DivisionForm>(form))
    {
    case DivisionForm::kExactDivision:
        return Launch<DivisionForm::kExactDivision>(
            entry, shape, tables, nmax, order, x, half, wide, out, count);
    case DivisionForm::kPlainReciprocal:
        return Launch<DivisionForm::kPlainReciprocal>(
            entry, shape, tables, nmax, order, x, half, wide, out, count);
    case DivisionForm::kRefinedReciprocal:
        return Launch<DivisionForm::kRefinedReciprocal>(
            entry, shape, tables, nmax, order, x, half, wide, out, count);
    }

    return 1;
}
