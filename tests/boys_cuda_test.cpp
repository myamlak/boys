#include "boysymmetriad/boys.hpp"
#include "boysymmetriad/boys_cuda.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <limits>
#include <random>
#include <vector>

#if BoysFp16
#include "boysymmetriad/f16.hpp"
#endif

namespace {

constexpr std::size_t kCount = 1u << 16;
constexpr double kDoubleTolerance = 5.5e-14;
// Cross-lane budget: the CPU lane is validated <= 1.5e-7 against the
// reference grid; the GPU lane carries its own <= ~1.5e-7 plus the __expf
// slack of the single kernels, so GPU-vs-CPU agreement holds within ~3.5e-7.
constexpr float kFloatTolerance = 3.5e-7f;

struct DeviceSetup {
    DeviceSetup() {
        cudaError_t error = cudaMalloc(&n, kCount * sizeof(int));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&x, kCount * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF32, kCount * (boysymmetriad::kMaxBoysOrder + 1) * sizeof(float));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&outF64, kCount * (boysymmetriad::kMaxBoysOrder + 1) * sizeof(double));
        EXPECT_EQ(error, cudaSuccess);

        std::mt19937_64 rng(20260817);
        std::uniform_real_distribution<double> xd(1e-4, 60.0);
        std::vector<int> hostN(kCount);
        std::vector<double> hostX(kCount);

        for (std::size_t i = 0; i < kCount; ++i)
        {
            hostN[i] = static_cast<int>(rng() % (boysymmetriad::kMaxBoysOrder + 1));
            hostX[i] = xd(rng);
        }
        // The x == 0 device path is a dedicated branch in every kernel.
        hostN[0] = 3;
        hostX[0] = 0.0;
        hostN[1] = 17;
        hostX[1] = 0.0;
        cudaMemcpy(n, hostN.data(), kCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(x, hostX.data(), kCount * sizeof(double), cudaMemcpyHostToDevice);
    }

    ~DeviceSetup() {
        cudaFree(n);
        cudaFree(x);
        cudaFree(outF32);
        cudaFree(outF64);
    }

    int* n = nullptr;
    double* x = nullptr;
    float* outF32 = nullptr;
    double* outF64 = nullptr;
};

#if BoysFp16
// The fp16 lane compares against the CPU fp16 lane (both compute in float
// and round to fp16): the cross-lane float budget is the F32 one above, and
// one ULP of quantization covers the fp16 rounding (HalfUlp here is the
// full grid step, NextUp(x) - x, not half of it).
double HalfUlp(boysymmetriad::F16 x) {
    return static_cast<double>(boysymmetriad::NextUp(x)) - static_cast<double>(x);
}

double F16Tolerance(boysymmetriad::F16 cpuValue) {
    return 3.5e-7 + HalfUlp(cpuValue);
}
#endif // BoysFp16

} // namespace

TEST(BoysCudaTest, SingleF32MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    ASSERT_EQ(boysymmetriad::BoysCuda::SingleF32(setup.n, setup.x, setup.outF32, kCount, nullptr),
              boysymmetriad::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<float> hostOut(kCount);
    cudaMemcpy(hostOut.data(), setup.outF32, kCount * sizeof(float), cudaMemcpyDeviceToHost);
    float worst = 0.0f;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const float cpu = boysymmetriad::BoysSingleF32(hostN[i], static_cast<float>(hostX[i]));
        worst = std::max(worst, std::abs(hostOut[i] - cpu));
    }

    EXPECT_LE(worst, kFloatTolerance);
    std::printf("SingleF32 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, SingleF64MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    ASSERT_EQ(boysymmetriad::BoysCuda::SingleF64(setup.n, setup.x, setup.outF64, kCount, nullptr),
              boysymmetriad::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> hostOut(kCount);
    cudaMemcpy(hostOut.data(), setup.outF64, kCount * sizeof(double), cudaMemcpyDeviceToHost);
    double worst = 0.0;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const double cpu = boysymmetriad::BoysSingle(hostN[i], hostX[i]);
        worst = std::max(worst, std::abs(hostOut[i] - cpu));
    }

    EXPECT_LE(worst, kDoubleTolerance);
    std::printf("SingleF64 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, BatchF32MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    ASSERT_EQ(boysymmetriad::BoysCuda::BatchF32(setup.n, setup.x, setup.outF32, kCount, nullptr),
              boysymmetriad::BoysStatus::kSuccess);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<float> hostOut(kCount * (boysymmetriad::kMaxBoysOrder + 1));
    cudaMemcpy(hostOut.data(),
               setup.outF32,
               kCount * (boysymmetriad::kMaxBoysOrder + 1) * sizeof(float),
               cudaMemcpyDeviceToHost);
    float worst = 0.0f;
    std::vector<float> cpuBatch(boysymmetriad::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boysymmetriad::BoysBatchF32(hostN[i], static_cast<float>(hostX[i]), cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            worst = std::max(worst, std::abs(hostOut[k * kCount + i] - cpuBatch[k]));
        }
    }

    EXPECT_LE(worst, kFloatTolerance);
    std::printf("BatchF32 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, BatchF64MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    DeviceSetup setup;
    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(boysymmetriad::BoysCuda::BatchF64(setup.n, setup.x, setup.outF64, kCount, stream),
              boysymmetriad::BoysStatus::kSuccess);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();

    std::vector<int> hostN(kCount);
    std::vector<double> hostX(kCount);
    cudaMemcpy(hostN.data(), setup.n, kCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostX.data(), setup.x, kCount * sizeof(double), cudaMemcpyDeviceToHost);

    std::vector<double> hostOut(kCount * (boysymmetriad::kMaxBoysOrder + 1));
    cudaMemcpy(hostOut.data(),
               setup.outF64,
               kCount * (boysymmetriad::kMaxBoysOrder + 1) * sizeof(double),
               cudaMemcpyDeviceToHost);
    double worst = 0.0;
    std::vector<double> cpuBatch(boysymmetriad::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boysymmetriad::BoysBatch(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            worst = std::max(worst, std::abs(hostOut[k * kCount + i] - cpuBatch[k]));
        }
    }

    EXPECT_LE(worst, kDoubleTolerance);
    std::printf("BatchF64 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

#if BoysFp16
// Device buffers shared by the fp16 lane tests: every entry takes device
// pointers (cudaMalloc/cudaMemcpy here — the caller owns the memory) and is
// asynchronous, so each test synchronizes the stream before reading back.
struct F16DeviceSetup {
    F16DeviceSetup(std::size_t count) {
        cudaError_t error = cudaMalloc(&n, count * sizeof(int));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&x, count * sizeof(boysymmetriad::F16));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&out, count * sizeof(boysymmetriad::F16));
        EXPECT_EQ(error, cudaSuccess);
        error = cudaMalloc(&batchOut,
                           count * (boysymmetriad::kMaxBoysOrder + 1) * sizeof(boysymmetriad::F16));
        EXPECT_EQ(error, cudaSuccess);
    }

    ~F16DeviceSetup() {
        cudaFree(n);
        cudaFree(x);
        cudaFree(out);
        cudaFree(batchOut);
    }

    int* n = nullptr;
    boysymmetriad::F16* x = nullptr;
    boysymmetriad::F16* out = nullptr;
    boysymmetriad::F16* batchOut = nullptr;
};

void FillF16Inputs(std::vector<int>& hostN,
                   std::vector<boysymmetriad::F16>& hostX,
                   std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> xd(1e-4f, 60.0f);

    for (std::size_t i = 0; i < hostN.size(); ++i)
    {
        hostN[i] = static_cast<int>(rng() % (boysymmetriad::kMaxBoysOrder + 1));
        hostX[i] = static_cast<boysymmetriad::F16>(xd(rng));
    }
    // The x == 0 device path is a dedicated branch in every kernel.
    hostN[0] = 3;
    hostX[0] = boysymmetriad::F16{0.0f};
    hostN[1] = 17;
    hostX[1] = boysymmetriad::F16{0.0f};
}

void UploadF16Inputs(F16DeviceSetup& setup,
                     const std::vector<int>& hostN,
                     const std::vector<boysymmetriad::F16>& hostX) {
    cudaMemcpy(setup.n, hostN.data(), hostN.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(
        setup.x, hostX.data(), hostX.size() * sizeof(boysymmetriad::F16), cudaMemcpyHostToDevice);
}

TEST(BoysCudaTest, SingleF16MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boysymmetriad::F16> hostX(kCount);
    std::vector<boysymmetriad::F16> hostOut(kCount);
    FillF16Inputs(hostN, hostX, 20260823);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    ASSERT_EQ(boysymmetriad::BoysCuda::SingleF16(setup.n, setup.x, setup.out, kCount, nullptr),
              boysymmetriad::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(
        hostOut.data(), setup.out, kCount * sizeof(boysymmetriad::F16), cudaMemcpyDeviceToHost);

    double worst = 0.0;

    for (std::size_t i = 0; i < kCount; ++i)
    {
        const boysymmetriad::F16 cpu = boysymmetriad::BoysSingleF16(hostN[i], hostX[i]);
        const double error = std::abs(static_cast<double>(hostOut[i]) - static_cast<double>(cpu));
        EXPECT_LE(error, F16Tolerance(cpu))
            << "i=" << i << " n=" << hostN[i] << " x=" << static_cast<float>(hostX[i]);
        worst = std::max(worst, error);
    }

    std::printf("SingleF16 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, BatchF16MatchesCpu) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boysymmetriad::F16> hostX(kCount);
    std::vector<boysymmetriad::F16> hostOut(kCount * (boysymmetriad::kMaxBoysOrder + 1));
    FillF16Inputs(hostN, hostX, 20260824);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    ASSERT_EQ(boysymmetriad::BoysCuda::BatchF16(setup.n, setup.x, setup.batchOut, kCount, nullptr),
              boysymmetriad::BoysStatus::kSuccess);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOut.data(),
               setup.batchOut,
               hostOut.size() * sizeof(boysymmetriad::F16),
               cudaMemcpyDeviceToHost);

    double worst = 0.0;
    std::vector<boysymmetriad::F16> cpuBatch(boysymmetriad::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boysymmetriad::BoysBatchF16(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            const double error = std::abs(static_cast<double>(hostOut[k * kCount + i]) -
                                          static_cast<double>(cpuBatch[k]));
            EXPECT_LE(error, F16Tolerance(cpuBatch[k]))
                << "i=" << i << " k=" << k << " x=" << static_cast<float>(hostX[i]);
            worst = std::max(worst, error);
        }
    }

    std::printf("BatchF16 GPU vs CPU: worst |diff| = %.3e\n", worst);
}

TEST(BoysCudaTest, BatchF16MatchesCpuOnAStream) {
    // The async contract: a non-default stream must produce the same values
    // as the default-stream path once the stream is synchronized.
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    ASSERT_EQ(boysymmetriad::BoysCuda::InitializeTables(), boysymmetriad::BoysStatus::kSuccess);

    std::vector<int> hostN(kCount);
    std::vector<boysymmetriad::F16> hostX(kCount);
    std::vector<boysymmetriad::F16> hostOut(kCount * (boysymmetriad::kMaxBoysOrder + 1));
    FillF16Inputs(hostN, hostX, 20260824);
    F16DeviceSetup setup(kCount);
    UploadF16Inputs(setup, hostN, hostX);

    cudaStream_t stream = nullptr;
    ASSERT_EQ(cudaStreamCreate(&stream), cudaSuccess);
    ASSERT_EQ(boysymmetriad::BoysCuda::BatchF16(setup.n, setup.x, setup.batchOut, kCount, stream),
              boysymmetriad::BoysStatus::kSuccess);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();
    cudaMemcpy(hostOut.data(),
               setup.batchOut,
               hostOut.size() * sizeof(boysymmetriad::F16),
               cudaMemcpyDeviceToHost);

    std::vector<boysymmetriad::F16> cpuBatch(boysymmetriad::kMaxBoysOrder + 1);

    for (std::size_t i = 0; i < kCount; ++i)
    {
        boysymmetriad::BoysBatchF16(hostN[i], hostX[i], cpuBatch.data());

        for (int k = 0; k <= hostN[i]; ++k)
        {
            EXPECT_LE(std::abs(static_cast<double>(hostOut[k * kCount + i]) -
                               static_cast<double>(cpuBatch[k])),
                      F16Tolerance(cpuBatch[k]))
                << "i=" << i << " k=" << k << " x=" << static_cast<float>(hostX[i]);
        }
    }
}

TEST(BoysCudaTest, F16RejectsZeroCount) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
    {
        GTEST_SKIP() << "no CUDA device";
    }

    boysymmetriad::F16 x = boysymmetriad::F16{1.0f};
    int n = 1;
    ASSERT_EQ(boysymmetriad::BoysCuda::SingleF16(&n, &x, &x, 0, nullptr),
              boysymmetriad::BoysStatus::kInvalidArgument);
    ASSERT_EQ(boysymmetriad::BoysCuda::BatchF16(&n, &x, &x, 0, nullptr),
              boysymmetriad::BoysStatus::kInvalidArgument);
}
#endif // BoysFp16
