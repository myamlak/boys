// The device half of the consumer check on the CUDA lane's default. The lane's
// f32 single entries take two selection axes and no others — the accuracy
// multiplier and the region-B exponential — both template parameters with
// defaults, so a caller that has chosen the precision and the format's f32
// writes one call and gets the lane's shipped arithmetic.
//
// The check is two translation units because the lane is: the batch entry
// BoysCuda::SingleF32 is declared in <boys/boys_cuda.hpp>, a host header, and
// the device-callable BoysDeviceSingleF32 in <boys/boys_cuda_device.hpp>, which
// is what a .cu may include. This file is the device half.

#include "boys/boys_cuda_device.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

static_assert(boys::kDefaultRegionBExp == boys::RegionBExp::kAccurate);

template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

// Two spellings of one entry compare equal as function addresses exactly when
// they are one instantiation, so the assertion below says the two calls are one
// call and not two that happen to agree. The negated one is the control, without
// which an equality that held for every pair would prove nothing.
static_assert(
    SameCall<&boys::BoysDeviceSingleF32<>, &boys::BoysDeviceSingleF32<boys::kDefaultRegionBExp>>);
static_assert(!SameCall<&boys::BoysDeviceSingleF32<boys::kDefaultRegionBExp>,
                        &boys::BoysDeviceSingleF32<boys::RegionBExp::kFast>>);

namespace {

constexpr int kThreads = 256;

/// One row of the report: the cells compared, how many differed, and the
/// furthest apart the two values were.
struct Row {
    const char* name = "";
    std::size_t cells = 0;
    std::size_t differing = 0;
    double worst = 0.0;
};

std::uint32_t Bits(float v) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

void Accumulate(Row& row, float left, float right) {
    ++row.cells;

    if (Bits(left) == Bits(right))
    {
        return;
    }

    ++row.differing;
    row.worst =
        std::max(row.worst, std::fabs(static_cast<double>(left) - static_cast<double>(right)));
}

void Print(const Row& row) {
    std::printf("  %-46s %6zu cells, %zu differing, worst |left - right| = %.3g\n",
                row.name,
                row.cells,
                row.differing,
                row.worst);
}

// The entry called all three ways from one kernel, reading no internal header:
// this is what a consumer's own kernel is.
__global__ void DeviceEntryKernel(__grid_constant__ const boys::BoysDeviceTables tables,
                                  const int* n,
                                  const double* xs,
                                  float* plain,
                                  float* named,
                                  float* other,
                                  int* status,
                                  std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const float x = static_cast<float>(xs[i]);
    float left = 0.0f;
    float right = 0.0f;
    float spared = 0.0f;

    const boys::BoysDeviceStatus a =
        boys::BoysDeviceSingleF32(tables, n[i], x, &left, boys::kBoysFullAccuracyMultiplier);
    const boys::BoysDeviceStatus b = boys::BoysDeviceSingleF32<boys::kDefaultRegionBExp>(
        tables, n[i], x, &right, boys::kBoysFullAccuracyMultiplier);
    const boys::BoysDeviceStatus c = boys::BoysDeviceSingleF32<boys::RegionBExp::kFast>(
        tables, n[i], x, &spared, boys::kBoysFullAccuracyMultiplier);

    status[i] = (a == b && a == c) ? static_cast<int>(a) : -1;
    plain[i] = left;
    named[i] = right;
    other[i] = spared;
}

} // namespace

/// The device-callable entry, run on the card in both spellings.
///
/// \returns the number of cells the two identity rows differ on, or -1 when the
/// launch or an allocation failed, having printed why. \c statusDiffering
/// receives the number of elements whose three calls did not agree on the
/// status.
extern "C" int BoysConsumerCudaDefaultsDeviceRows(const boys::BoysDeviceTables* tables,
                                                  const int* n,
                                                  const double* xs,
                                                  std::size_t count,
                                                  std::size_t* statusDiffering) {
    *statusDiffering = 0;

    float* plain = nullptr;
    float* named = nullptr;
    float* other = nullptr;
    float* hostPlain = nullptr;
    float* hostNamed = nullptr;
    float* hostOther = nullptr;
    int* status = nullptr;
    int* hostStatus = nullptr;

    auto release = [&]() {
        cudaFree(plain);
        cudaFree(named);
        cudaFree(other);
        cudaFree(status);
        cudaFreeHost(hostPlain);
        cudaFreeHost(hostNamed);
        cudaFreeHost(hostOther);
        cudaFreeHost(hostStatus);
    };

    cudaError_t error = cudaMalloc(&plain, count * sizeof(float));
    error = error == cudaSuccess ? cudaMalloc(&named, count * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMalloc(&other, count * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMalloc(&status, count * sizeof(int)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostPlain, count * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostNamed, count * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostOther, count * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostStatus, count * sizeof(int)) : error;

    if (error != cudaSuccess)
    {
        std::printf("  BoysDeviceSingleF32 allocation failed: %s\n", cudaGetErrorString(error));
        release();
        return -1;
    }

    const unsigned int blocks = static_cast<unsigned int>((count + kThreads - 1) / kThreads);
    DeviceEntryKernel<<<blocks, kThreads>>>(*tables, n, xs, plain, named, other, status, count);
    error = cudaGetLastError();

    if (error == cudaSuccess)
    {
        error = cudaDeviceSynchronize();
    }

    if (error != cudaSuccess)
    {
        std::printf("  BoysDeviceSingleF32 launch failed: %s\n", cudaGetErrorString(error));
        release();
        return -1;
    }

    cudaMemcpy(hostPlain, plain, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNamed, named, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostOther, other, count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostStatus, status, count * sizeof(int), cudaMemcpyDeviceToHost);

    Row identity;
    identity.name = "device BoysDeviceSingleF32, named vs unnamed";
    Row option;
    option.name = "device BoysDeviceSingleF32, default vs kFast";

    for (std::size_t i = 0; i < count; ++i)
    {
        Accumulate(identity, hostPlain[i], hostNamed[i]);
        Accumulate(option, hostOther[i], hostNamed[i]);

        if (hostStatus[i] < 0)
        {
            ++*statusDiffering;
        }
    }

    Print(identity);
    Print(option);

    std::printf("  %zu of %zu elements returned statuses the three calls disagree on\n",
                *statusDiffering,
                count);

    release();

    return static_cast<int>(identity.differing + *statusDiffering);
}
