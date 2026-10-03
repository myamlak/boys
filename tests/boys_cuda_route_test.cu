// The device half of the multiply-add route check: the lane's own arithmetic,
// evaluated at both routes and at the build's own selection, so that the host
// half can hold each of the three to what it means.
//
// Naming a route explicitly is what a check can do and a call site cannot: the
// route is a build fact (BOYS_MULADD_SEPARATE), selected once and read by both
// sides of the device boundary. This file exists to hold the build's arithmetic
// to the route it was built at, so that "the device ran the fused step" is a
// result rather than a sentence in a header.
//
// The bodies are the lane's own (boys/boys_cuda_arithmetic.hpp): the
// multiply-add step the host lane's backend names, and the uniform grid's flat
// ladder, which is the arithmetic BoysCuda::AllOrdersF64Uniform,
// AllOrdersF64UniformHorner, AllOrdersF32Uniform and AllOrdersF32UniformHorner
// launch. The three arrays each kernel writes are the same arithmetic at the
// fused route, at the separate route, and at the route the build selected —
// the last of them through the bodies' own default, which is what every entry
// of the public surface runs.

#include "boys/boys_cuda_arithmetic.hpp"
#include "boys/boys_device_tables.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace {

using boys::backend::MulAddRoute;

unsigned int RouteBlocks(std::size_t count) {
    return static_cast<unsigned int>((count + 255) / 256);
}

__device__ __forceinline__ std::size_t RouteThread() {
    return static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) +
           static_cast<std::size_t>(threadIdx.x);
}

// ---------------------------------------------------------------------------
// the multiply-add step
// ---------------------------------------------------------------------------

__global__ void RouteStepF64Kernel(const double* a,
                                   const double* b,
                                   const double* c,
                                   std::size_t count,
                                   double* fused,
                                   double* separate,
                                   double* delivered) {
    const std::size_t i = RouteThread();

    if (i >= count)
    {
        return;
    }

    fused[i] = boys::detail::DeviceMulAdd<MulAddRoute::kFused>(a[i], b[i], c[i]);
    separate[i] = boys::detail::DeviceMulAdd<MulAddRoute::kSeparate>(a[i], b[i], c[i]);
    delivered[i] = boys::detail::DeviceMulAdd(a[i], b[i], c[i]);
}

__global__ void RouteStepF32Kernel(const float* a,
                                   const float* b,
                                   const float* c,
                                   std::size_t count,
                                   float* fused,
                                   float* separate,
                                   float* delivered) {
    const std::size_t i = RouteThread();

    if (i >= count)
    {
        return;
    }

    fused[i] = boys::detail::DeviceMulAdd<MulAddRoute::kFused>(a[i], b[i], c[i]);
    separate[i] = boys::detail::DeviceMulAdd<MulAddRoute::kSeparate>(a[i], b[i], c[i]);
    delivered[i] = boys::detail::DeviceMulAdd(a[i], b[i], c[i]);
}

// ---------------------------------------------------------------------------
// the uniform ladder
// ---------------------------------------------------------------------------

// The ladder to each argument's own order, order-major, which is the layout
// BoysCuda::AllOrdersF64Uniform writes.
__global__ void RouteLadderF64Kernel(__grid_constant__ const boys::BoysDeviceTables tables,
                                     const int* n,
                                     const double* x,
                                     std::size_t count,
                                     bool monomial,
                                     double* fused,
                                     double* separate,
                                     double* delivered) {
    const std::size_t i = RouteThread();

    if (i >= count)
    {
        return;
    }

    if (monomial)
    {
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,true, MulAddRoute::kFused>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { fused[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,true, MulAddRoute::kSeparate>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { separate[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,true>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { delivered[static_cast<std::size_t>(l) * count + i] = v; });
    }
    else
    {
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,false, MulAddRoute::kFused>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { fused[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,false, MulAddRoute::kSeparate>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { separate[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF64Flat<boys::kDefaultDivisionForm,false>(
            tables.flatCoeffs, tables.flatMonoCoeffs, tables.flatDegs, tables.flatOffsets, n[i],
            x[i], [&](int l, double v) { delivered[static_cast<std::size_t>(l) * count + i] = v; });
    }
}

__global__ void RouteLadderF32Kernel(__grid_constant__ const boys::BoysDeviceTables tables,
                                     const int* n,
                                     const float* x,
                                     std::size_t count,
                                     bool monomial,
                                     float* fused,
                                     float* separate,
                                     float* delivered) {
    const std::size_t i = RouteThread();

    if (i >= count)
    {
        return;
    }

    if (monomial)
    {
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,true, MulAddRoute::kFused>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i], [&](int l, float v) { fused[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,true, MulAddRoute::kSeparate>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i],
            [&](int l, float v) { separate[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,true>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i],
            [&](int l, float v) { delivered[static_cast<std::size_t>(l) * count + i] = v; });
    }
    else
    {
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,false, MulAddRoute::kFused>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i], [&](int l, float v) { fused[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,false, MulAddRoute::kSeparate>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i],
            [&](int l, float v) { separate[static_cast<std::size_t>(l) * count + i] = v; });
        boys::detail::DeviceAllOrdersF32Flat<boys::kDefaultDivisionForm,false>(
            tables.flatCoeffs32, tables.flatMonoCoeffs32, tables.flatDegs32, tables.flatOffsets32,
            n[i], x[i],
            [&](int l, float v) { delivered[static_cast<std::size_t>(l) * count + i] = v; });
    }
}

} // namespace

// C linkage, so a drift between the two halves is a link error rather than a
// silent second reading. Each returns 0 on success and the CUDA runtime's error
// code otherwise.

extern "C" int BoysCudaRouteStepF64(const double* a,
                                    const double* b,
                                    const double* c,
                                    std::size_t count,
                                    double* fused,
                                    double* separate,
                                    double* delivered) {
    RouteStepF64Kernel<<<RouteBlocks(count), 256>>>(a, b, c, count, fused, separate, delivered);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaRouteStepF32(const float* a,
                                    const float* b,
                                    const float* c,
                                    std::size_t count,
                                    float* fused,
                                    float* separate,
                                    float* delivered) {
    RouteStepF32Kernel<<<RouteBlocks(count), 256>>>(a, b, c, count, fused, separate, delivered);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaRouteLadderF64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* x,
                                      std::size_t count,
                                      int monomial,
                                      double* fused,
                                      double* separate,
                                      double* delivered) {
    RouteLadderF64Kernel<<<RouteBlocks(count), 256>>>(*tables, n, x, count, monomial != 0, fused,
                                                      separate, delivered);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int BoysCudaRouteLadderF32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const float* x,
                                      std::size_t count,
                                      int monomial,
                                      float* fused,
                                      float* separate,
                                      float* delivered) {
    RouteLadderF32Kernel<<<RouteBlocks(count), 256>>>(*tables, n, x, count, monomial != 0, fused,
                                                      separate, delivered);
    return static_cast<int>(cudaGetLastError());
}
