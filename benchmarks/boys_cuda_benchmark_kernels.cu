// GPU competitor kernels of the benchmark suite: the erf-F0 and the
// Tsuji-style gridded-LUT lanes. nvcc TUs compile at CMAKE_CUDA_STANDARD 20, so
// C++23 headers must not reach them: this file includes only CUDA and libm, and
// reaches the host through the extern "C" seam of the companion header.

#include <cuda_runtime.h>
#include <math.h>
#include <stddef.h>

namespace {

constexpr double kHalfSqrtPi = 0.886226925452758014;

// Tsuji-style LUT geometry ([Tsuji2025]): 1025 grid points over x in [0, 32] at
// step 2^-5, degree-5 Taylor corrections, and the asymptotic series above the
// boundary x < a*n + b.
constexpr double kLutInterval = 0.03125;
constexpr int kLutXiCount = 1025;
constexpr int kLutKmax = 5;
constexpr int kLutMaxOrder = 32;
constexpr int kLutRows = kLutMaxOrder + kLutKmax + 1; // 38: rows 0..32 + 33..37
constexpr double kLutThresholdA = 0.064048916778075;
constexpr double kLutThresholdB = 28.487431543672;

// 38 x 1025 x 8 B = 311.6 KB — too large for __constant__, so it lives in
// global memory.
__device__ double gLut[kLutRows][kLutXiCount];

__global__ void BoysErfF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int nn = n[i];
    const double xx = x[i];

    if (xx == 0.0)
    {
        out[i] = 1.0 / (2.0 * nn + 1.0);
        return;
    }
    // The erf-F0 seed in rsqrt form.
    double f = kHalfSqrtPi * rsqrt(xx) * erf(sqrt(xx));
    const double expx = exp(-xx);
    const double inv2x = 0.5 / xx;

    for (int l = 0; l < nn; ++l)
    {
        f = __fma_rn(2.0 * l + 1.0, f, -expx) * inv2x;
    }

    out[i] = f;
}

__device__ double GriddedTaylor(int nn, double x) {
    const int idx = static_cast<int>(__double2uint_rd(x / kLutInterval + 0.5));
    const double dx = x - kLutInterval * idx;
    double boys = __ldg(&gLut[nn][idx]);
    double num = 1.0;
    int fact = 1;

    for (int k = 1; k <= kLutKmax; ++k)
    {
        num *= -dx;
        fact *= k;
        boys += __ldg(&gLut[nn + k][idx]) * num / fact;
    }

    return boys;
}

__global__ void BoysLutF64Kernel(const int* n, const double* x, double* out, size_t count) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const int nn = n[i];
    const double xx = x[i];

    if (xx == 0.0)
    {
        out[i] = 1.0 / (2.0 * nn + 1.0);
        return;
    }

    if (xx < kLutThresholdA * nn + kLutThresholdB)
    {
        out[i] = GriddedTaylor(nn, xx);
        return;
    }

    double f = kHalfSqrtPi * rsqrt(xx);

    for (int l = 0; l < nn; ++l)
    {
        f = (l + 0.5) * f / xx;
    }

    out[i] = f;
}

} // namespace

// Return codes are cudaError_t values, 0 = success, so the host TU reports a
// failure with cudaGetErrorString. Launch errors surface on the host's sync, so
// the launch functions return 0 unconditionally.
extern "C" {

int BoysBenchSetDevice(int device) {
    return static_cast<int>(cudaSetDevice(device));
}

int BoysBenchUploadLut(const double* rows) {
    return static_cast<int>(cudaMemcpyToSymbol(gLut, rows, sizeof(gLut)));
}

int BoysBenchLaunchErfF64(const int* n,
                          const double* x,
                          double* out,
                          size_t count,
                          int blocks,
                          int threads,
                          void* stream) {
    BoysErfF64Kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return 0;
}

int BoysBenchLaunchLutF64(const int* n,
                          const double* x,
                          double* out,
                          size_t count,
                          int blocks,
                          int threads,
                          void* stream) {
    BoysLutF64Kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(n, x, out, count);
    return 0;
}

} // extern "C"
