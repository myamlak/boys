// The host half of the multiply-add route check on the CUDA lane. It prints the
// route a caller reads off the lane, holds the device's arithmetic at both
// routes to the exact references a host can state — std::fma for the fused
// route, round(round(a * b) + c) for the separate one — and holds the route the
// build selected to the arithmetic it delivers. The rows come in two sets: the
// lane's bodies as this file's kernels instantiate them, and the shipped batch
// entry, whose arithmetic the lane's own translation unit compiled. The second
// set is the one that can see whether the build's selection reached the lane.
//
// This is the half that can include <boys/boys_cuda.hpp>, a host header. The
// kernels and the device-side entries are in tests/boys_cuda_route_test.cu.
//
// Run:  cmake --build <build> --target boys-cuda-route-tests       (needs -DBUILD_CUDA=ON)
//       <build>/Release/boys-cuda-route-tests
//       ctest --test-dir <build> -R boys-cuda-route-tests
//
// The exit code is the verdict: 0 when every requirement below holds, 1 when a
// requirement failed, 2 when the card or the lane could not be used. A build
// with -DBOYS_MULADD_SEPARATE=ON is a second run of the same check, and the one
// where the two routes part: the same rows are required of it, so a lane that
// ran the fused step under the separate selection fails here rather than
// answering with the other route's bits.

#include <boys/boys_cuda.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

namespace {

using boys::backend::MulAddRoute;

/// The multiply-add step's own row count: enough triples that a route's
/// difference is a count and not an anecdote.
constexpr std::size_t kStepCount = 1u << 20;

/// The ladder's argument count. Orders cycle 0..kMaxBoysOrder over it, so every
/// order's own fit is stepped at both routes.
constexpr std::size_t kLadderCount = 8192;

/// The build's own selection, read the way a consumer reads it: the definition
/// the library's target publishes.
#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
constexpr MulAddRoute kBuildSelection = MulAddRoute::kSeparate;
constexpr const char* kBuildSelectionName = "BOYS_MULADD_SEPARATE";
#else
constexpr MulAddRoute kBuildSelection = MulAddRoute::kFused;
constexpr const char* kBuildSelectionName = "the shipped default";
#endif

std::uint32_t Bits(float v) noexcept {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

std::uint64_t Bits(double v) noexcept {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof bits);
    return bits;
}

/// One comparison of two runs of the same entries: the cells compared, how many
/// of them differ in their bits, and the worst absolute difference between them.
struct Row {
    const char* name = "";
    std::size_t cells = 0;
    std::size_t differing = 0;
    double worst = 0.0;
};

template <typename T>
Row Compare(const std::vector<T>& left, const std::vector<T>& right, const char* name) {
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
    std::printf("  %-54s %8zu cells %8zu differing, worst %.3g\n", row.name, row.cells,
                row.differing, row.worst);
}

int g_failures = 0;

void Require(const char* what, bool holds, const char* detail = "") {
    std::printf("  %-6s %-48s %s\n", holds ? "ok" : "FAIL", what, detail);

    if (!holds)
    {
        ++g_failures;
    }
}

// ---------------------------------------------------------------------------
// the value sets
// ---------------------------------------------------------------------------

std::uint64_t g_state = 0x243F6A8885A308D3ull;

std::uint64_t Next64() {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

/// A full-mantissa value in [1, 2) scaled by a power of two, so the product is
/// inexact and the cancellation below is real. A generator whose mantissa
/// carries twelve bits makes every product exact, and then the two routes agree
/// on every value — a value set that cannot see the difference it is looking
/// for.
double RandUnitDouble(std::uint64_t u) {
    const std::uint64_t bits = (u & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;
    double m = 0.0;
    std::memcpy(&m, &bits, sizeof m);

    const int sig = static_cast<int>((u >> 52) % 20) - 10;
    return m * std::ldexp(1.0, sig);
}

float RandUnitFloat(std::uint64_t u) {
    const std::uint32_t bits = (static_cast<std::uint32_t>(u) & 0x007FFFFFu) | 0x3F800000u;
    float m = 0.0f;
    std::memcpy(&m, &bits, sizeof m);

    const int sig = static_cast<int>((u >> 23) % 20) - 10;
    return m * std::ldexp(1.0f, sig);
}

/// A triple whose product is inexact and whose addend cancels it, which is
/// where the two routes must part; the spread of magnitudes keeps them from
/// parting only at one exponent.
void MakeTriple(double& a, double& b, double& c, std::uint64_t u) {
    a = RandUnitDouble(u);
    b = RandUnitDouble(u * 0x9E3779B97F4A7C15ull + 1);

    const double p = std::fma(a, b, 0.0);
    const double rel = 1.0 + static_cast<double>(u % 8) * (1.0 / 256.0);
    c = -p * rel;
}

void MakeTriple(float& a, float& b, float& c, std::uint64_t u) {
    a = RandUnitFloat(u);
    b = RandUnitFloat(u * 0x9E3779B97F4A7C15ull + 1);

    const float p = std::fmaf(a, b, 0.0f);
    const float rel = 1.0f + static_cast<float>(u % 8) * (1.0f / 256.0f);
    c = -p * rel;
}

/// The exact references the two routes mean on a conforming host:
/// std::fma rounds once, and fma(fma(a, b, 0), 1, c) is round(round(a * b) + c)
/// — the product rounded once, then the sum rounded once.
double RefFused(double a, double b, double c) {
    return std::fma(a, b, c);
}

double RefSeparate(double a, double b, double c) {
    return std::fma(std::fma(a, b, 0.0), 1.0, c);
}

float RefFused(float a, float b, float c) {
    return std::fmaf(a, b, c);
}

float RefSeparate(float a, float b, float c) {
    return std::fmaf(std::fmaf(a, b, 0.0f), 1.0f, c);
}

// ---------------------------------------------------------------------------
// the checks
// ---------------------------------------------------------------------------

/// One precision's step rows: both routes against their exact references, the
/// two against each other, and the delivered arithmetic against the reported
/// route.
template <typename T>
void StepRows(const char* tag,
              const std::vector<T>& a,
              const std::vector<T>& b,
              const std::vector<T>& c,
              const std::vector<T>& fused,
              const std::vector<T>& separate,
              const std::vector<T>& delivered,
              MulAddRoute reported,
              int deviceStatus) {
    std::printf("\n%s multiply-add step, %zu triples:\n", tag, a.size());

    if (deviceStatus != 0)
    {
        Require("the step kernel ran", false, cudaGetErrorString(static_cast<cudaError_t>(deviceStatus)));
        return;
    }

    std::vector<T> refFused(a.size());
    std::vector<T> refSeparate(a.size());

    for (std::size_t i = 0; i < a.size(); ++i)
    {
        refFused[i] = RefFused(a[i], b[i], c[i]);
        refSeparate[i] = RefSeparate(a[i], b[i], c[i]);
    }

    const Row atFused = Compare(fused, refFused, "device fused route vs one rounding");
    Print(atFused);
    Require("the fused route is the one-rounding arithmetic", atFused.differing == 0);

    const Row atSeparate = Compare(separate, refSeparate, "device separate route vs two roundings");
    Print(atSeparate);
    Require("the separate route is the two-rounding arithmetic", atSeparate.differing == 0);

    const Row routes = Compare(fused, separate, "device fused vs device separate");
    Print(routes);
    Require("the two routes are different arithmetic", routes.differing > 0,
            routes.differing == 0 ? "no value parted the routes" : "");

    const bool reportedSeparate = reported == MulAddRoute::kSeparate;
    const std::vector<T>& named = reportedSeparate ? separate : fused;
    const std::vector<T>& other = reportedSeparate ? fused : separate;

    const Row deliveredNamed =
        Compare(delivered, named, reportedSeparate ? "delivered vs the reported separate route"
                                                   : "delivered vs the reported fused route");
    Print(deliveredNamed);
    Require("the delivered arithmetic is the reported route", deliveredNamed.differing == 0);

    const Row deliveredOther = Compare(delivered, other, "delivered vs the other route");
    Print(deliveredOther);
    Require("the delivered arithmetic is not the other route", deliveredOther.differing > 0);
}

/// The uniform ladder's rows. No host can state this ladder's exact reference
/// (the fits are the library's), so these rows are relative: the two routes
/// must part, and the delivered arithmetic must be the reported one.
template <typename T>
void LadderRows(const char* tag,
                const std::vector<T>& fused,
                const std::vector<T>& separate,
                const std::vector<T>& delivered,
                MulAddRoute reported,
                int deviceStatus) {
    std::printf("\n%s uniform ladder, %zu arguments:\n", tag, fused.size());

    if (deviceStatus != 0)
    {
        Require("the ladder kernel ran", false, cudaGetErrorString(static_cast<cudaError_t>(deviceStatus)));
        return;
    }

    const Row routes = Compare(fused, separate, "device fused vs device separate");
    Print(routes);
    Require("the route reaches the ladder's arithmetic", routes.differing > 0,
            routes.differing == 0 ? "no value parted the routes" : "");

    const bool reportedSeparate = reported == MulAddRoute::kSeparate;
    const std::vector<T>& named = reportedSeparate ? separate : fused;
    const std::vector<T>& other = reportedSeparate ? fused : separate;

    const Row deliveredNamed = Compare(delivered, named, "delivered vs the reported route");
    Print(deliveredNamed);
    Require("the delivered ladder is the reported route", deliveredNamed.differing == 0);

    const Row deliveredOther = Compare(delivered, other, "delivered vs the other route");
    Print(deliveredOther);
    Require("the delivered ladder is not the other route", deliveredOther.differing > 0);
}

/// The shipped entry's rows. The rows above run the lane's bodies as this file
/// compiles them; these run the arithmetic the library's own translation unit
/// compiled, which is the one a caller of BoysCuda gets. A build that selects the
/// separate route and reaches only this file — the selection missing from the
/// lane's own target — passes every row above and parts here.
template <typename T>
void ShippedRows(const char* tag,
                 const std::vector<T>& shipped,
                 const std::vector<T>& delivered,
                 const std::vector<T>& other,
                 bool launched) {
    std::printf("\n%s shipped batch entry, %zu cells:\n", tag, shipped.size());

    if (!launched)
    {
        Require("the shipped entry ran", false);
        return;
    }

    const Row same = Compare(shipped, delivered, "shipped entry vs the delivered route");
    Print(same);
    Require("the shipped entry delivers the reported route", same.differing == 0);

    const Row apart = Compare(shipped, other, "shipped entry vs the other route");
    Print(apart);
    Require("the shipped entry is not the other route", apart.differing > 0);
}

} // namespace

// Defined in tests/boys_cuda_route_test.cu. Each returns 0 on success and the
// CUDA runtime's error code otherwise.
extern "C" int BoysCudaRouteStepF64(const double* a,
                                    const double* b,
                                    const double* c,
                                    std::size_t count,
                                    double* fused,
                                    double* separate,
                                    double* delivered);
extern "C" int BoysCudaRouteStepF32(const float* a,
                                    const float* b,
                                    const float* c,
                                    std::size_t count,
                                    float* fused,
                                    float* separate,
                                    float* delivered);
extern "C" int BoysCudaRouteLadderF64(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const double* x,
                                      std::size_t count,
                                      int monomial,
                                      double* fused,
                                      double* separate,
                                      double* delivered);
extern "C" int BoysCudaRouteLadderF32(const boys::BoysDeviceTables* tables,
                                      const int* n,
                                      const float* x,
                                      std::size_t count,
                                      int monomial,
                                      float* fused,
                                      float* separate,
                                      float* delivered);

int main() {
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        std::printf("device multiply-add route check: no usable device\n");
        return 2;
    }

    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, device);

    boys::BoysDeviceTables tables{};
    const boys::BoysStatus tablesStatus = boys::BoysCuda::DeviceTables(&tables);

    if (tablesStatus != boys::BoysStatus::kSuccess)
    {
        std::printf("device multiply-add route check: DeviceTables returned status %d\n",
                    static_cast<int>(tablesStatus));
        return 2;
    }

    // --- what a caller reads -------------------------------------------------
    const MulAddRoute reported = boys::BoysCuda::MulAddRouteInForce();

    std::printf("device multiply-add route check — %s, compute capability %d.%d\n",
                properties.name, properties.major, properties.minor);
    std::printf("  BoysCuda::MulAddRouteInForce() reports %s; this build selected %s\n",
                boys::backend::MulAddRouteName(reported), kBuildSelectionName);
    Require("the lane reports the route this build selected", reported == kBuildSelection,
            reported == kBuildSelection ? "" : "the lane's report and the build disagree");

    const std::size_t ladderCells = kLadderCount * (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1);

    // --- the step, both precisions -------------------------------------------
    {
        std::vector<double> a(kStepCount);
        std::vector<double> b(kStepCount);
        std::vector<double> c(kStepCount);

        for (std::size_t i = 0; i < kStepCount; ++i)
        {
            MakeTriple(a[i], b[i], c[i], Next64());
        }

        double* deviceA = nullptr;
        double* deviceB = nullptr;
        double* deviceC = nullptr;
        double* fused = nullptr;
        double* separate = nullptr;
        double* delivered = nullptr;
        cudaMalloc(&deviceA, kStepCount * sizeof(double));
        cudaMalloc(&deviceB, kStepCount * sizeof(double));
        cudaMalloc(&deviceC, kStepCount * sizeof(double));
        cudaMalloc(&fused, kStepCount * sizeof(double));
        cudaMalloc(&separate, kStepCount * sizeof(double));
        cudaMalloc(&delivered, kStepCount * sizeof(double));
        cudaMemcpy(deviceA, a.data(), kStepCount * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, b.data(), kStepCount * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceC, c.data(), kStepCount * sizeof(double), cudaMemcpyHostToDevice);

        const int status =
            BoysCudaRouteStepF64(deviceA, deviceB, deviceC, kStepCount, fused, separate, delivered);

        std::vector<double> hostFused(kStepCount);
        std::vector<double> hostSeparate(kStepCount);
        std::vector<double> hostDelivered(kStepCount);
        cudaMemcpy(hostFused.data(), fused, kStepCount * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostSeparate.data(), separate, kStepCount * sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDelivered.data(), delivered, kStepCount * sizeof(double), cudaMemcpyDeviceToHost);

        StepRows<double>("f64", a, b, c, hostFused, hostSeparate, hostDelivered, reported, status);

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        cudaFree(fused);
        cudaFree(separate);
        cudaFree(delivered);
    }

    {
        std::vector<float> a(kStepCount);
        std::vector<float> b(kStepCount);
        std::vector<float> c(kStepCount);

        for (std::size_t i = 0; i < kStepCount; ++i)
        {
            MakeTriple(a[i], b[i], c[i], Next64());
        }

        float* deviceA = nullptr;
        float* deviceB = nullptr;
        float* deviceC = nullptr;
        float* fused = nullptr;
        float* separate = nullptr;
        float* delivered = nullptr;
        cudaMalloc(&deviceA, kStepCount * sizeof(float));
        cudaMalloc(&deviceB, kStepCount * sizeof(float));
        cudaMalloc(&deviceC, kStepCount * sizeof(float));
        cudaMalloc(&fused, kStepCount * sizeof(float));
        cudaMalloc(&separate, kStepCount * sizeof(float));
        cudaMalloc(&delivered, kStepCount * sizeof(float));
        cudaMemcpy(deviceA, a.data(), kStepCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceB, b.data(), kStepCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceC, c.data(), kStepCount * sizeof(float), cudaMemcpyHostToDevice);

        const int status =
            BoysCudaRouteStepF32(deviceA, deviceB, deviceC, kStepCount, fused, separate, delivered);

        std::vector<float> hostFused(kStepCount);
        std::vector<float> hostSeparate(kStepCount);
        std::vector<float> hostDelivered(kStepCount);
        cudaMemcpy(hostFused.data(), fused, kStepCount * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostSeparate.data(), separate, kStepCount * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(hostDelivered.data(), delivered, kStepCount * sizeof(float), cudaMemcpyDeviceToHost);

        StepRows<float>("f32", a, b, c, hostFused, hostSeparate, hostDelivered, reported, status);

        cudaFree(deviceA);
        cudaFree(deviceB);
        cudaFree(deviceC);
        cudaFree(fused);
        cudaFree(separate);
        cudaFree(delivered);
    }

    // --- the uniform ladder, which is what the batch entries launch ----------
    {
        std::vector<int> n(kLadderCount);
        std::vector<double> x(kLadderCount);

        for (std::size_t i = 0; i < kLadderCount; ++i)
        {
            n[i] = static_cast<int>(i % (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1));
            x[i] = 19.0 * static_cast<double>(i) / static_cast<double>(kLadderCount - 1);
        }

        // The region boundaries and the asymptotic arm, so the ladder is
        // stepped where the argument is not one smooth sweep.
        x[0] = 0.0;
        x[1] = 1.0;
        x[2] = 11.899848152108484;
        x[3] = 18.0;
        x[4] = 30.0;
        x[5] = 1.0e6;

        int* deviceN = nullptr;
        double* deviceX = nullptr;
        double* fused = nullptr;
        double* separate = nullptr;
        double* delivered = nullptr;
        cudaMalloc(&deviceN, kLadderCount * sizeof(int));
        cudaMalloc(&deviceX, kLadderCount * sizeof(double));
        cudaMalloc(&fused, ladderCells * sizeof(double));
        cudaMalloc(&separate, ladderCells * sizeof(double));
        cudaMalloc(&delivered, ladderCells * sizeof(double));
        cudaMemcpy(deviceN, n.data(), kLadderCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceX, x.data(), kLadderCount * sizeof(double), cudaMemcpyHostToDevice);

        std::vector<double> hostFused(ladderCells);
        std::vector<double> hostSeparate(ladderCells);
        std::vector<double> hostDelivered(ladderCells);

        for (int monomial = 0; monomial <= 1; ++monomial)
        {
            const int status = BoysCudaRouteLadderF64(&tables, deviceN, deviceX, kLadderCount, monomial,
                                                      fused, separate, delivered);
            cudaMemcpy(hostFused.data(), fused, ladderCells * sizeof(double), cudaMemcpyDeviceToHost);
            cudaMemcpy(hostSeparate.data(), separate, ladderCells * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(hostDelivered.data(), delivered, ladderCells * sizeof(double),
                       cudaMemcpyDeviceToHost);

            LadderRows<double>(monomial != 0 ? "f64 monomial" : "f64 chebyshev", hostFused,
                               hostSeparate, hostDelivered, reported, status);

            std::vector<double> shipped(ladderCells);
            double* deviceShipped = nullptr;
            cudaMalloc(&deviceShipped, ladderCells * sizeof(double));

            const boys::BoysStatus shippedStatus =
                monomial != 0 ? boys::BoysCuda::AllOrdersF64UniformHorner(
                                    deviceN, deviceX, deviceShipped, kLadderCount, nullptr)
                              : boys::BoysCuda::AllOrdersF64Uniform(deviceN, deviceX, deviceShipped,
                                                                    kLadderCount, nullptr);

            cudaMemcpy(shipped.data(), deviceShipped, ladderCells * sizeof(double),
                       cudaMemcpyDeviceToHost);
            cudaFree(deviceShipped);

            ShippedRows<double>(monomial != 0 ? "f64 monomial" : "f64 chebyshev", shipped, hostDelivered,
                                reported == MulAddRoute::kSeparate ? hostFused : hostSeparate,
                                shippedStatus == boys::BoysStatus::kSuccess);
        }

        cudaFree(deviceN);
        cudaFree(deviceX);
        cudaFree(fused);
        cudaFree(separate);
        cudaFree(delivered);
    }

    {
        std::vector<int> n(kLadderCount);
        std::vector<float> x(kLadderCount);

        for (std::size_t i = 0; i < kLadderCount; ++i)
        {
            n[i] = static_cast<int>(i % (static_cast<std::size_t>(boys::kMaxBoysOrder) + 1));
            x[i] = 11.0f * static_cast<float>(i) / static_cast<float>(kLadderCount - 1);
        }

        x[0] = 0.0f;
        x[1] = 1.0f;
        x[2] = 11.899848152108484f;
        x[3] = 18.0f;
        x[4] = 30.0f;
        x[5] = 1.0e6f;

        // The float entries take their arguments as doubles and narrow them
        // themselves, as every float entry of the lane does; this array is the
        // float array widened, so the narrowing inside the entry is exact and the
        // shipped kernel evaluates at the same float this file's kernel does.
        std::vector<double> xWide(kLadderCount);

        for (std::size_t i = 0; i < kLadderCount; ++i)
        {
            xWide[i] = static_cast<double>(x[i]);
        }

        int* deviceN = nullptr;
        float* deviceX = nullptr;
        double* deviceXWide = nullptr;
        float* fused = nullptr;
        float* separate = nullptr;
        float* delivered = nullptr;
        cudaMalloc(&deviceN, kLadderCount * sizeof(int));
        cudaMalloc(&deviceX, kLadderCount * sizeof(float));
        cudaMalloc(&deviceXWide, kLadderCount * sizeof(double));
        cudaMalloc(&fused, ladderCells * sizeof(float));
        cudaMalloc(&separate, ladderCells * sizeof(float));
        cudaMalloc(&delivered, ladderCells * sizeof(float));
        cudaMemcpy(deviceN, n.data(), kLadderCount * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceX, x.data(), kLadderCount * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceXWide, xWide.data(), kLadderCount * sizeof(double), cudaMemcpyHostToDevice);

        std::vector<float> hostFused(ladderCells);
        std::vector<float> hostSeparate(ladderCells);
        std::vector<float> hostDelivered(ladderCells);

        for (int monomial = 0; monomial <= 1; ++monomial)
        {
            const int status = BoysCudaRouteLadderF32(&tables, deviceN, deviceX, kLadderCount, monomial,
                                                      fused, separate, delivered);

            cudaMemcpy(hostFused.data(), fused, ladderCells * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(hostSeparate.data(), separate, ladderCells * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(hostDelivered.data(), delivered, ladderCells * sizeof(float),
                       cudaMemcpyDeviceToHost);

            LadderRows<float>(monomial != 0 ? "f32 monomial" : "f32 chebyshev", hostFused,
                              hostSeparate, hostDelivered, reported, status);

            std::vector<float> shipped(ladderCells);
            float* deviceShipped = nullptr;
            cudaMalloc(&deviceShipped, ladderCells * sizeof(float));

            const boys::BoysStatus shippedStatus =
                monomial != 0 ? boys::BoysCuda::AllOrdersF32UniformHorner(
                                    deviceN, deviceXWide, deviceShipped, kLadderCount, nullptr)
                              : boys::BoysCuda::AllOrdersF32Uniform(deviceN, deviceXWide, deviceShipped,
                                                                    kLadderCount, nullptr);

            cudaMemcpy(shipped.data(), deviceShipped, ladderCells * sizeof(float),
                       cudaMemcpyDeviceToHost);
            cudaFree(deviceShipped);

            ShippedRows<float>(monomial != 0 ? "f32 monomial" : "f32 chebyshev", shipped,
                               hostDelivered,
                               reported == MulAddRoute::kSeparate ? hostFused : hostSeparate,
                               shippedStatus == boys::BoysStatus::kSuccess);
        }

        cudaFree(deviceN);
        cudaFree(deviceXWide);
        cudaFree(deviceX);
        cudaFree(fused);
        cudaFree(separate);
        cudaFree(delivered);
    }

    std::printf("\n%s: %d requirement(s) failed\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
