// The host half of the consumer check on the CUDA lane's default. Naming the
// lane's defaults and naming nothing are one call rather than two that agree:
// the static assertions below hold the entry's template default and the value it
// names to be one instantiation, with the lane's other region-B exponential as a
// negative control, so the equality is not a name against itself.
//
// This is the half that can include <boys/boys_cuda.hpp>, a host header. The
// kernel and the device entry are in tests/consumer_cuda_defaults.cu.
//
// Run:  cmake --build <build> --target boys-consumer-cuda-defaults   (BUILD_CUDA=ON)
//       <build>/boys-consumer-cuda-defaults
//       ctest --test-dir <build> -R boys-consumer-cuda-defaults

#include <algorithm>
#include <boys/boys_cuda.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

static_assert(boys::kDefaultRegionBExp == boys::RegionBExp::kAccurate);

template <auto Left, auto Right> constexpr bool SameCall = (Left == Right);

static_assert(SameCall<&boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier>,
                       &boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier,
                                                  boys::kDefaultRegionBExp>>);
static_assert(SameCall<&boys::BoysCuda::DeviceTables<>,
                       &boys::BoysCuda::DeviceTables<boys::kBoysFullAccuracyMultiplier>>);
static_assert(
    !SameCall<
        &boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier, boys::kDefaultRegionBExp>,
        &boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier, boys::RegionBExp::kFast>>);

// C linkage, so a drift between the two sides is a link error rather than a
// silent second reading. Defined in tests/consumer_cuda_defaults.cu.
extern "C" int BoysConsumerCudaDefaultsDeviceRows(const boys::BoysDeviceTables* tables,
                                                  const int* n,
                                                  const double* xs,
                                                  std::size_t count,
                                                  std::size_t* statusDiffering);

namespace {

constexpr std::size_t kCount = 33u * 128u;

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

Row Compare(const std::vector<float>& left, const std::vector<float>& right, const char* name) {
    Row row;
    row.name = name;

    for (std::size_t i = 0; i < left.size(); ++i)
    {
        ++row.cells;

        if (Bits(left[i]) == Bits(right[i]))
        {
            continue;
        }

        ++row.differing;
        row.worst = std::max(
            row.worst, std::fabs(static_cast<double>(left[i]) - static_cast<double>(right[i])));
    }

    return row;
}

void Print(const Row& row) {
    std::printf("  %-46s %6zu cells, %zu differing, worst |left - right| = %.3g\n",
                row.name,
                row.cells,
                row.differing,
                row.worst);
}

} // namespace

int main() {
    // The handle first: filling it uploads the tables the device entry reads,
    // and the batch entry would upload them on its first call anyway.
    boys::BoysDeviceTables tables{};
    const boys::BoysStatus tablesStatus = boys::BoysCuda::DeviceTables(&tables);

    if (tablesStatus != boys::BoysStatus::kSuccess)
    {
        std::printf(
            "device lane default check: no usable device — DeviceTables returned status %d\n",
            static_cast<int>(tablesStatus));
        return 2;
    }

    // Orders 0..32 against a sweep of arguments, with four arguments pinned below.
    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        hostN[i] = static_cast<int>(i % 33u);
        hostX[i] = 28.98933773882074 * static_cast<double>(i) / static_cast<double>(kCount - 1);
    }

    hostX[0] = 0.0;
    hostX[1] = 1.0855252345349333;
    hostX[2] = 11.899848152108484;
    hostX[3] = 28.98933773882074;
    hostX[4] = 1.0e6;

    int* n = nullptr;
    double* xs = nullptr;
    float* outPlain = nullptr;
    float* outNamed = nullptr;
    float* outFast = nullptr;
    float* hostPlain = nullptr;
    float* hostNamed = nullptr;
    float* hostFast = nullptr;

    cudaError_t error = cudaMalloc(&n, kCount * sizeof(int));
    error = error == cudaSuccess ? cudaMalloc(&xs, kCount * sizeof(double)) : error;
    error = error == cudaSuccess ? cudaMalloc(&outPlain, kCount * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMalloc(&outNamed, kCount * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMalloc(&outFast, kCount * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostPlain, kCount * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostNamed, kCount * sizeof(float)) : error;
    error = error == cudaSuccess ? cudaMallocHost(&hostFast, kCount * sizeof(float)) : error;

    if (error != cudaSuccess)
    {
        std::printf("device lane default check: allocation failed: %s\n",
                    cudaGetErrorString(error));
        return 2;
    }

    cudaMemcpy(n, hostN.data(), kCount * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(xs, hostX.data(), kCount * sizeof(double), cudaMemcpyHostToDevice);

    std::printf("consumer check of the device lane's default — <boys/boys_cuda.hpp> here,"
                " <boys/boys_cuda_device.hpp> in the kernel half: %zu elements, orders 0..%d\n",
                kCount,
                boys::kMaxBoysOrder);

    // --- the device-callable entry, in the kernel that defines it -------------
    std::size_t deviceStatusDiffering = 0;
    const int deviceDiffering =
        BoysConsumerCudaDefaultsDeviceRows(&tables, n, xs, kCount, &deviceStatusDiffering);

    if (deviceDiffering < 0)
    {
        return 2;
    }

    // --- the batch entry, both spellings --------------------------------------
    const boys::BoysStatus plain = boys::BoysCuda::SingleF32<>(n, xs, outPlain, kCount, nullptr);
    const boys::BoysStatus named =
        boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier, boys::kDefaultRegionBExp>(
            n, xs, outNamed, kCount, nullptr);

    if (plain != boys::BoysStatus::kSuccess || named != boys::BoysStatus::kSuccess)
    {
        std::printf("  BoysCuda::SingleF32 launch failed: %d / %d\n",
                    static_cast<int>(plain),
                    static_cast<int>(named));
        return 2;
    }

    cudaMemcpy(hostPlain, outPlain, kCount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostNamed, outNamed, kCount * sizeof(float), cudaMemcpyDeviceToHost);

    const Row batch = Compare(std::vector<float>(hostPlain, hostPlain + kCount),
                              std::vector<float>(hostNamed, hostNamed + kCount),
                              "batch BoysCuda::SingleF32, named vs unnamed");
    Print(batch);

    // --- the default beside the lane's other exponential ----------------------
    const boys::BoysStatus fast =
        boys::BoysCuda::SingleF32<boys::kBoysFullAccuracyMultiplier, boys::RegionBExp::kFast>(
            n, xs, outFast, kCount, nullptr);

    if (fast != boys::BoysStatus::kSuccess)
    {
        std::printf("  BoysCuda::SingleF32<kFast> launch failed: %d\n", static_cast<int>(fast));
        return 2;
    }

    cudaMemcpy(hostFast, outFast, kCount * sizeof(float), cudaMemcpyDeviceToHost);

    const Row option = Compare(std::vector<float>(hostNamed, hostNamed + kCount),
                               std::vector<float>(hostFast, hostFast + kCount),
                               "batch BoysCuda::SingleF32, default vs kFast");
    Print(option);

    cudaFree(n);
    cudaFree(xs);
    cudaFree(outPlain);
    cudaFree(outNamed);
    cudaFree(outFast);
    cudaFreeHost(hostPlain);
    cudaFreeHost(hostNamed);
    cudaFreeHost(hostFast);

    // The device half returns its rows' difference and the elements whose calls
    // disagreed on the status, so the three identity comparisons are the batch
    // row, the device row and the statuses.
    const std::size_t identityDiffering =
        batch.differing + static_cast<std::size_t>(deviceDiffering);

    std::printf("  %zu of the %d identity comparisons differ anywhere; "
                "the name and the entry's default are one call\n",
                identityDiffering,
                3);

    // The option row does not fail the run: kFast carries its own bound, and how
    // far it sits from the default is printed rather than judged.
    return identityDiffering == 0 ? 0 : 1;
}
