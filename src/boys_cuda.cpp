#include "boysymmetriad/boys_cuda.hpp"

#include "boys_effective_degrees.hpp"
#include "boysymmetriad/boys.hpp"
#include "boysymmetriad/f16.hpp"

#include <array>
#include <cstddef>

// Status layer of the CUDA lane. The kernels and the table uploads live in
// boys_cuda.cu (C++20, CUDA-safe include list only — the C++23 headers of
// this library would poison the nvcc translation unit); this file wraps the
// exported plain functions in BoysStatus.
//
// Upload/launch return codes (boys_cuda.cu): 0 = success, 1 = internal
// table-layout error, otherwise a cudaError_t from the launch or the sync.

extern "C" {
int BoysCudaUploadTables();
int BoysCudaUploadEffTables(double m, const int* degA, const int* degB);
int BoysCudaLaunchSingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
#if BoysFp16
int BoysCudaLaunchSingleF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF16(const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchBatchF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
#endif
}

namespace boysymmetriad {
namespace {

// code == 1 is the internal table-layout error; every other nonzero code is
// a cudaError_t from the launch or the sync. The status enum carries no
// payload — callers diagnose device failures through the CUDA runtime's own
// error reporting.
BoysStatus FromLaunchCode(int code) {
    if (code == 0)
    {
        return BoysStatus::kSuccess;
    }

    return BoysStatus::kDeviceError;
}

} // namespace

namespace {
// ---------------------------------------------------------------------------
// The accuracy-multiplier effective-degree tables (design record D-F6 of
// the accompanying paper). The CUDA side cannot see the constexpr degree
// machinery (nvcc translation units get a CUDA-safe include list only), so
// the host layer computes the six lanes' degree tables here and uploads
// them through BoysCudaUploadEffTables, which caches per (device, m). The
// lane order matches the cDegEff lane axis in boys_cuda.cu:
//   0 = double single, 1 = double batch, 2 = float single, 3 = float batch,
//   4 = fp16 single, 5 = fp16 batch.
// degA layout: [lane][order][pieceInOrder]; degB layout: [lane][order].
// ---------------------------------------------------------------------------
// kMaxPieces in boys_cuda.cu (anonymous namespace there — spelled out here
// to keep the two layout formulas in lockstep).
constexpr int kEffLaneCount = 6;
constexpr int kEffMaxOrder = detail::kMaxOrder;
constexpr int kEffMaxPieces = 12;

std::array<int, kEffLaneCount*(kEffMaxOrder + 1) * kEffMaxPieces> gEffDegA{};
std::array<int, kEffLaneCount*(kEffMaxOrder + 1)> gEffDegB{};
double gEffCachedM = -1.0;

template <double kAccuracyMultiplier, detail::BoysRole kRole, bool kDoublePieces>
void FillEffLane(int lane) {
    static constexpr auto kDegreesA = detail::RegionADegrees<kAccuracyMultiplier, kRole>();
    static constexpr auto kDegreesB = detail::RegionBDegrees<kAccuracyMultiplier, kRole>();
    const auto& pieceStart = kDoublePieces ? detail::kPieceStart : detail::f32::kPieceStart;

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        for (int p = pieceStart[static_cast<std::size_t>(order)];
             p < pieceStart[static_cast<std::size_t>(order) + 1];
             ++p)
        {
            const int pieceInOrder = p - pieceStart[static_cast<std::size_t>(order)];
            gEffDegA[static_cast<std::size_t>(lane * (kEffMaxOrder + 1) * kEffMaxPieces +
                                              order * kEffMaxPieces + pieceInOrder)] =
                kDegreesA[static_cast<std::size_t>(p)];
        }
    }

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        gEffDegB[static_cast<std::size_t>(lane * (kEffMaxOrder + 1) + order)] =
            kDegreesB[static_cast<std::size_t>(order)];
    }
}

// The six CUDA lanes' roles: the double batch seed evaluates the DOUBLE
// piece table even for the float/fp16 batch lanes (RoleUsesDoubleTables —
// the downward recursion amplifies float seed errors beyond their budgets).
template <double kAccuracyMultiplier> BoysStatus EnsureEffTables() {
    if (gEffCachedM == kAccuracyMultiplier)
    {
        return BoysStatus::kSuccess;
    }

    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleSingle, true>(0);
    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch, true>(1);
    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Single, false>(2);
    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Batch, true>(3);
    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Single, false>(4);
    FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Batch, true>(5);

    const auto status = FromLaunchCode(
        BoysCudaUploadEffTables(kAccuracyMultiplier, gEffDegA.data(), gEffDegB.data()));

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    gEffCachedM = kAccuracyMultiplier;
    return BoysStatus::kSuccess;
}

} // namespace

BoysStatus BoysCuda::InitializeTables() {
    return FromLaunchCode(BoysCudaUploadTables());
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::SingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        // Byte-identical to the full-accuracy path: the m = 1 instantiation
        // calls the existing kernel and cDeg tables.
        return FromLaunchCode(BoysCudaLaunchSingleF32(n, x, out, count, stream));
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return FromLaunchCode(BoysCudaLaunchSingleF32Eff(n, x, out, count, stream));
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::BatchF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return FromLaunchCode(BoysCudaLaunchBatchF32(n, x, out, count, stream));
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return FromLaunchCode(BoysCudaLaunchBatchF32Eff(n, x, out, count, stream));
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::SingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return FromLaunchCode(BoysCudaLaunchSingleF64(n, x, out, count, stream));
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return FromLaunchCode(BoysCudaLaunchSingleF64Eff(n, x, out, count, stream));
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::BatchF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return FromLaunchCode(BoysCudaLaunchBatchF64(n, x, out, count, stream));
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return FromLaunchCode(BoysCudaLaunchBatchF64Eff(n, x, out, count, stream));
}

#if BoysFp16
namespace {
BoysStatus CheckCount(std::size_t count) {
    // The fp16 kernels require at least one element (the count == 0 pin).
    if (count == 0)
    {
        return BoysStatus::kInvalidArgument;
    }

    return BoysStatus::kSuccess;
}

// Shared fp16-launch path: device pointers throughout (the caller owns the
// device memory), uniform async contract — the kernel is queued on the
// caller's stream and the call returns once the launch is accepted.
BoysStatus RunF16Call(int (*launcher)(const int*, const void*, void*, std::size_t, void*),
                      const int* n,
                      const void* x,
                      void* out,
                      std::size_t count,
                      void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    const auto valid = CheckCount(count);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    return FromLaunchCode(launcher(n, x, out, count, stream));
}
} // namespace

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::SingleF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunF16Call(BoysCudaLaunchSingleF16, n, x, out, count, stream);
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return RunF16Call(BoysCudaLaunchSingleF16Eff, n, x, out, count, stream);
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::BatchF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunF16Call(BoysCudaLaunchBatchF16, n, x, out, count, stream);
    }

    const auto status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return RunF16Call(BoysCudaLaunchBatchF16Eff, n, x, out, count, stream);
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// Explicit instantiations at the sampled multipliers. The entry definitions
// live in this TU (the header stays CUDA-runtime-free), so the call sites in
// other TUs link only the instantiations spelled out here — m = 1.0 first
// (the full-accuracy pin), then the relaxation sample set.
// ---------------------------------------------------------------------------
template BoysStatus BoysCuda::SingleF32<1.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<1.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<1.0>(const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<1.0>(const int*, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<2.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<2.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<2.0>(const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<2.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<2.0>(const int*, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<10.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<10.0>(
    const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<10.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<10.0>(const int*, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<100.0>(
    const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<100.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<100.0>(const int*, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<1e4>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<1e4>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<1e4>(const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1e4>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<1e4>(const int*, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<1e8>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF32<1e8>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF64<1e8>(const int*, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::BatchF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
#endif

} // namespace boysymmetriad
