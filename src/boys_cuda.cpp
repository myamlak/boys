#include "boys/boys_cuda.hpp"

#include "boys/boys.hpp"
#include "boys/f16.hpp"
#include "boys/boys_effective_degrees.hpp"

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
int BoysCudaDeviceTableAddresses(void** out);
int BoysCudaUploadEffTables(double m, const int* degA, const int* degB);
int BoysCudaLaunchSingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32Fast(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF32(int nmax, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF64(int nmax, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32EffFast(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Eff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF32Eff(
    int nmax, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Eff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF64Eff(
    int nmax, const double* x, double* out, std::size_t count, void* stream);
#if BoysFp16
int BoysCudaLaunchSingleF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF16(int nmax, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Eff(
    const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF16Eff(int nmax, const void* x, void* out, std::size_t count, void* stream);
#endif
}

namespace boys {
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

// The uniform-order entries' one host-visible scalar: the order array of the
// per-element entries cannot be checked without a device copy, this can.
BoysStatus CheckOrder(int nmax) {
    if (nmax < 0 || nmax > kMaxBoysOrder)
    {
        return BoysStatus::kInvalidArgument;
    }

    return BoysStatus::kSuccess;
}

// Shared launch path for every entry: the count == 0 no-op (a zero-block launch
// is a CUDA error, an empty batch is a success that writes nothing) and the one
// status mapping. Order is either entry's first argument: the per-element order
// array or the uniform batch's nmax. The pointers carry the entry's own element
// type, the fp16 lane's crossing as the void* the .cu exports take.
template <typename Launcher, typename Order, typename X, typename Value>
BoysStatus RunLaunch(
    Launcher launcher, Order order, X x, Value* out, std::size_t count, void* stream) {
    if (count == 0)
    {
        return BoysStatus::kSuccess;
    }

    return FromLaunchCode(launcher(order, x, out, count, stream));
}

} // namespace

namespace {
// ---------------------------------------------------------------------------
// The accuracy-multiplier effective-degree tables. The CUDA side cannot see
// the constexpr degree machinery (nvcc translation units get a CUDA-safe
// include list only), so
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
    // The host-side degree tables depend on the multiplier alone, so they are
    // computed once per multiplier. Whether the device already holds them is
    // the upload's question, and its guard names the device as well as the
    // multiplier: answering it here would skip the upload a second device
    // still needs, and that device's kernels would read a zero table.
    if (gEffCachedM != kAccuracyMultiplier)
    {
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleSingle, true>(0);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch, true>(1);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Single, false>(2);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Batch, true>(3);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Single, false>(4);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Batch, true>(5);
        gEffCachedM = kAccuracyMultiplier;
    }

    return FromLaunchCode(
        BoysCudaUploadEffTables(kAccuracyMultiplier, gEffDegA.data(), gEffDegB.data()));
}

} // namespace

BoysStatus BoysCuda::InitializeTables() {
    return FromLaunchCode(BoysCudaUploadTables());
}

BoysStatus BoysCuda::DeviceTables(BoysDeviceTables* out) {
    if (out == nullptr)
    {
        return BoysStatus::kInvalidArgument;
    }

    // The address order BoysCudaDeviceTableAddresses fills, one slot per
    // symbol: the double lane's pieceStart, offset, a, b, deg, coeffs and
    // region-B seed, then the float lane's seven. Both sides state the order;
    // the .cu cannot name this type and this file cannot name a symbol.
    void* addresses[14] = {};
    const BoysStatus status = FromLaunchCode(BoysCudaDeviceTableAddresses(addresses));

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    BoysDeviceTables tables;
    tables.pieceStart = static_cast<const int*>(addresses[0]);
    tables.pieceOffset = static_cast<const int*>(addresses[1]);
    tables.pieceA = static_cast<const double*>(addresses[2]);
    tables.pieceB = static_cast<const double*>(addresses[3]);
    tables.pieceDeg = static_cast<const int*>(addresses[4]);
    tables.coeffs = static_cast<const double*>(addresses[5]);
    tables.bSeedCoeffs = static_cast<const double*>(addresses[6]);
    tables.bSeedDeg = detail::kBDeg;
    tables.pieceStart32 = static_cast<const int*>(addresses[7]);
    tables.pieceOffset32 = static_cast<const int*>(addresses[8]);
    tables.pieceA32 = static_cast<const float*>(addresses[9]);
    tables.pieceB32 = static_cast<const float*>(addresses[10]);
    tables.pieceDeg32 = static_cast<const int*>(addresses[11]);
    tables.coeffs32 = static_cast<const float*>(addresses[12]);
    tables.bSeedCoeffs32 = static_cast<const float*>(addresses[13]);
    tables.bSeedDeg32 = detail::f32::kBDeg;

    *out = tables;
    return BoysStatus::kSuccess;
}

template <double kAccuracyMultiplier, RegionBExp kExp>
BoysStatus BoysCuda::SingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The exponential is part of the call's identity, not a run-time switch:
    // each option is its own kernel with its own bound, and no path here
    // substitutes one for the other.
    if constexpr (kAccuracyMultiplier == 1.0)
    {
        // Byte-identical to the full-accuracy path: the m = 1 instantiation
        // calls the existing kernel and cDeg tables.
        if constexpr (kExp == RegionBExp::kFast)
        {
            return RunLaunch(BoysCudaLaunchSingleF32Fast, n, x, out, count, stream);
        } else
        {
            return RunLaunch(BoysCudaLaunchSingleF32, n, x, out, count, stream);
        }
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        if constexpr (kExp == RegionBExp::kFast)
        {
            return RunLaunch(BoysCudaLaunchSingleF32EffFast, n, x, out, count, stream);
        } else
        {
            return RunLaunch(BoysCudaLaunchSingleF32Eff, n, x, out, count, stream);
        }
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF32Eff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllNF32(
    int nmax, const double* x, float* out, std::size_t count, void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllNF32, nmax, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllNF32Eff, nmax, x, out, count, stream);
    }
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
        return RunLaunch(BoysCudaLaunchSingleF64, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchSingleF64Eff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64Eff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllNF64(
    int nmax, const double* x, double* out, std::size_t count, void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllNF64, nmax, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllNF64Eff, nmax, x, out, count, stream);
    }
}

#if BoysFp16
template <double kAccuracyMultiplier>
BoysStatus BoysCuda::SingleF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchSingleF16, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchSingleF16Eff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF16, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF16Eff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllNF16(int nmax, const F16* x, F16* out, std::size_t count, void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllNF16, nmax, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllNF16Eff, nmax, x, out, count, stream);
    }
}
#endif // BoysFp16

// ---------------------------------------------------------------------------
// Explicit instantiations at the sampled multipliers. The entry definitions
// live in this TU (the header stays CUDA-runtime-free), so the call sites in
// other TUs link only the instantiations spelled out here — m = 1.0 first
// (the full-accuracy pin), then the relaxation sample set. The f32 single
// entry is instantiated once per calibrated (multiplier, exponential) pair it
// offers; every other entry has one arithmetic and one instantiation per
// multiplier.
// ---------------------------------------------------------------------------
template BoysStatus BoysCuda::SingleF32<1.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<1.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<1.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<1.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<1.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<1.0>(int, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<2.0>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<2.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<2.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<2.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<2.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<2.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<2.0>(int, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<10.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<10.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<10.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<10.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<10.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<10.0>(int, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<100.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<100.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<100.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<100.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<100.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<100.0>(int, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<1e4>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<1e4, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<1e4>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<1e4>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1e4>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<1e4>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<1e4>(int, const F16*, F16*, std::size_t, void*);
#endif
template BoysStatus BoysCuda::SingleF32<1e8>(const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<1e8, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<1e8>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<1e8>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<1e8>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<1e8>(int, const F16*, F16*, std::size_t, void*);
#endif

} // namespace boys
