#include "boys/boys_cuda.hpp"

#include "boys/boys.hpp"
#include "boys/f16.hpp"
#include "boys/boys_effective_degrees.hpp"

#include <array>
#include <cstddef>
#include <iterator>

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
int BoysCudaDeviceTableAddressesTail(void** out);
int BoysCudaEffTablesResident(double m);
int BoysCudaUploadEffTables(double m,
                            const int* degA,
                            const int* degB,
                            const int* narrowA,
                            const int* narrowB,
                            const int* narrowB32,
                            const int* monoA,
                            const int* monoB,
                            const int* narrowMonoA,
                            const int* narrowMonoB,
                            const int* ratSeedA,
                            const int* ratOrdA,
                            const int* ratB,
                            const int* narrowRatSeedA,
                            const int* narrowRatOrdA,
                            const int* narrowRatB);
int BoysCudaLaunchSingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32Fast(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF32(int nmax, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Uniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32UniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Narrow(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Rat(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Orders(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32OrdersEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrders(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrdersEff(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32OrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Uniform(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64UniformHorner(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Orders(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Narrow(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrders(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Mono(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64MonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Rat(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64RatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersRatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowRatEff(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff(
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
// array or the uniform batch's nmax.
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
// The accuracy-multiplier effective-degree tables. A CUDA-safe include list
// cannot see the constexpr degree machinery, so the host layer computes the six
// lanes' degree tables here and uploads them through BoysCudaUploadEffTables,
// which caches per (device, m). The lane order matches the cDegEff lane axis in
// boys_cuda.cu:
//   0 = double single, 1 = double batch, 2 = float single, 3 = float batch,
//   4 = fp16 single, 5 = fp16 batch.
// degA layout: [lane][order][pieceInOrder]; degB layout: [lane][order].
// ---------------------------------------------------------------------------
// kMaxPieces in boys_cuda.cu (anonymous namespace there — spelled out here
// to keep the two layout formulas in lockstep).
constexpr int kEffLaneCount = 6;
constexpr int kEffMaxOrder = detail::kMaxOrder;
constexpr int kEffMaxPieces = 12;
// The flat relaxed image's lane stride: the wider of the two piece tables,
// which is the same count boys_cuda.cu cuts its own copy of it with.
constexpr int kRelaxedStride = detail::kPieceStart[detail::kMaxOrder + 1] >
                                       detail::f32::kPieceStart[detail::kMaxOrder + 1]
                                   ? detail::kPieceStart[detail::kMaxOrder + 1]
                                   : detail::f32::kPieceStart[detail::kMaxOrder + 1];

std::array<int, kEffLaneCount*(kEffMaxOrder + 1) * kEffMaxPieces> gEffDegA{};
std::array<int, kEffLaneCount*(kEffMaxOrder + 1)> gEffDegB{};

// The narrow partition's cut, flat over the partition's rows as the derivation
// returns it and as the device lane indexes it. One table and not a lane axis:
// the entries carrying the partition are the double batch, so this is that
// role's table.
constexpr int kNarrowFlatPieces = detail::kNarrowAPieceStart[detail::kMaxOrder + 1];
constexpr int kNarrowFlatB = detail::kNarrowBPieces * (kEffMaxOrder + 1);

std::array<int, kNarrowFlatPieces> gNarrowDegA{};
std::array<int, kNarrowFlatB> gNarrowDegB{};

// The float lane's own cut of that rung, region B alone. The float lane's
// region-A seed is the double lane's piece table — that is what the seed lane
// argument of the shared body is — so the rung cuts that lane's region A and
// this lane's is its region B: a different fit, over the float lane's 218
// pieces at degree 6 rather than the double lane's 311 at degree 10.
constexpr int kNarrowFlatB32 = detail::f32::kNarrowBPiecesF32 * (kEffMaxOrder + 1);
std::array<int, kNarrowFlatB32> gNarrowDegB32{};

// The monomial scheme's cut of the same rung, over the same pieces. Region A is
// order-major as the CUDA lane's cDegEff axis is — the derivation is flat over
// the pieces, and the lane reads it by (order, pieceInOrder) — and region B is
// one row per order, read at order 0 by the batch shape. One table and not a
// lane axis, for the same reason the narrow partition's is one.
std::array<int, (kEffMaxOrder + 1) * kEffMaxPieces> gMonoDegA{};
std::array<int, kEffMaxOrder + 1> gMonoDegB{};
std::array<int, kNarrowFlatPieces> gNarrowMonoDegA{};
std::array<int, kNarrowFlatB> gNarrowMonoDegB{};

// Which multiplier the host-side tables above were last computed for. It is a
// record of the HOST computation and not of what the device holds, so answering
// residency from it would skip the upload a second device still needs and that
// device's kernels would read a zero table.
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

// The narrow partition's cut for one multiplier, for the role the entries that
// carry it have. Region B is kept in the whole derived form — flat over (piece,
// order) — rather than at the order-0 column the batch shape reads.
template <double kAccuracyMultiplier> void FillNarrowLane() {
    static constexpr auto kDegreesA =
        detail::NarrowRegionADegrees<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch>();
    static constexpr auto kDegreesB =
        detail::NarrowRegionBDegrees<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch>();

    for (int p = 0; p < kNarrowFlatPieces; ++p)
    {
        gNarrowDegA[static_cast<std::size_t>(p)] = kDegreesA[static_cast<std::size_t>(p)];
    }

    for (int k = 0; k < kNarrowFlatB; ++k)
    {
        gNarrowDegB[static_cast<std::size_t>(k)] = kDegreesB[static_cast<std::size_t>(k)];
    }
}

// The same partition's cut for the float lane, region B alone and for the
// reason the table above states: this lane's region-A seed is the double
// lane's, so the degrees a rung cuts for this lane are its region-B pieces'.
// The role is the float batch's, which is the shape the entries carrying the
// partition have.
template <double kAccuracyMultiplier> void FillNarrowF32Lane() {
    static constexpr auto kDegreesB =
        detail::NarrowRegionBDegrees<kAccuracyMultiplier, detail::BoysRole::kF32Batch>();

    for (int k = 0; k < kNarrowFlatB32; ++k)
    {
        gNarrowDegB32[static_cast<std::size_t>(k)] = kDegreesB[static_cast<std::size_t>(k)];
    }
}

// The double batch's cut of the shipped and the narrow fits in the monomial
// basis: the same derivations as the Chebyshev tables above at the other form
// of the same stored table (TailBasis::kMonomial), which is what a Horner rung
// drops its coefficients from.
template <double kAccuracyMultiplier> void FillMonoLane() {
    static constexpr auto kDegreesA = detail::RegionADegrees<kAccuracyMultiplier,
                                                             detail::BoysRole::kDoubleBatch,
                                                             detail::TailBasis::kMonomial>();
    static constexpr auto kDegreesB = detail::RegionBDegrees<kAccuracyMultiplier,
                                                             detail::BoysRole::kDoubleBatch,
                                                             detail::TailBasis::kMonomial>();

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        for (int p = detail::kPieceStart[static_cast<std::size_t>(order)];
             p < detail::kPieceStart[static_cast<std::size_t>(order) + 1];
             ++p)
        {
            const int pieceInOrder = p - detail::kPieceStart[static_cast<std::size_t>(order)];
            gMonoDegA[static_cast<std::size_t>(order * kEffMaxPieces + pieceInOrder)] =
                kDegreesA[static_cast<std::size_t>(p)];
        }

        gMonoDegB[static_cast<std::size_t>(order)] =
            kDegreesB[static_cast<std::size_t>(order)];
    }
}

template <double kAccuracyMultiplier> void FillNarrowMonoLane() {
    static constexpr auto kDegreesA = detail::NarrowRegionADegrees<kAccuracyMultiplier,
                                                                  detail::BoysRole::kDoubleBatch,
                                                                  detail::TailBasis::kMonomial>();
    static constexpr auto kDegreesB = detail::NarrowRegionBDegrees<kAccuracyMultiplier,
                                                                  detail::BoysRole::kDoubleBatch,
                                                                  detail::TailBasis::kMonomial>();

    for (int p = 0; p < kNarrowFlatPieces; ++p)
    {
        gNarrowMonoDegA[static_cast<std::size_t>(p)] = kDegreesA[static_cast<std::size_t>(p)];
    }

    for (int k = 0; k < kNarrowFlatB; ++k)
    {
        gNarrowMonoDegB[static_cast<std::size_t>(k)] = kDegreesB[static_cast<std::size_t>(k)];
    }
}

// The fit route's cut of the same rung, two ints per cell: the numerator's cut
// degree then the denominator's, which is the pair the criterion certifies
// together. Region A lands twice because the route's cut is per reading — the
// arguments shape seeds at its top order's piece and pays that piece's w(b), the
// orders shape reads each order's own piece at A = 1 — and both are taken from
// the derivation over the partition's rows and reshaped to the lane's
// (order, pieceInOrder) indexing. Region B is one pair either way.
constexpr int kRatCutCells = (kEffMaxOrder + 1) * kEffMaxPieces * 2;
constexpr int kNarrowRatCutPieces = detail::kNarrowBPieces * 2;

std::array<int, kRatCutCells> gRatSeedA{};
std::array<int, kRatCutCells> gRatOrdA{};
std::array<int, 2> gRatB{};
std::array<int, kNarrowFlatPieces * 2> gNarrowRatSeedA{};
std::array<int, kNarrowFlatPieces * 2> gNarrowRatOrdA{};
std::array<int, kNarrowRatCutPieces> gNarrowRatB{};

template <double kAccuracyMultiplier> void FillRatLane() {
    static constexpr auto kSeed = detail::RationalRegionASeedDegrees<kAccuracyMultiplier,
                                                                    detail::BoysRole::kDoubleBatch>();
    static constexpr auto kOrd = detail::RationalRegionADegrees<kAccuracyMultiplier>();
    static constexpr auto kB = detail::RationalRegionBDegrees<kAccuracyMultiplier>();

    for (int order = 0; order <= detail::kMaxOrder; ++order)
    {
        for (int p = detail::kPieceStart[static_cast<std::size_t>(order)];
             p < detail::kPieceStart[static_cast<std::size_t>(order) + 1];
             ++p)
        {
            const int pieceInOrder = p - detail::kPieceStart[static_cast<std::size_t>(order)];
            const std::size_t cell = static_cast<std::size_t>(order * kEffMaxPieces + pieceInOrder);
            const std::size_t index = static_cast<std::size_t>(p);
            gRatSeedA[cell * 2] = kSeed.num[index];
            gRatSeedA[cell * 2 + 1] = kSeed.den[index];
            gRatOrdA[cell * 2] = kOrd.num[index];
            gRatOrdA[cell * 2 + 1] = kOrd.den[index];
        }
    }

    gRatB[0] = kB.num[0];
    gRatB[1] = kB.den[0];
}

template <double kAccuracyMultiplier> void FillNarrowRatLane() {
    static constexpr auto kSeed =
        detail::RationalNarrowRegionASeedDegrees<kAccuracyMultiplier,
                                                detail::BoysRole::kDoubleBatch>();
    static constexpr auto kOrd = detail::RationalNarrowRegionADegrees<kAccuracyMultiplier>();
    static constexpr auto kB = detail::RationalNarrowRegionBDegrees<kAccuracyMultiplier>();

    for (int p = 0; p < kNarrowFlatPieces; ++p)
    {
        const std::size_t index = static_cast<std::size_t>(p);
        gNarrowRatSeedA[index * 2] = kSeed.num[index];
        gNarrowRatSeedA[index * 2 + 1] = kSeed.den[index];
        gNarrowRatOrdA[index * 2] = kOrd.num[index];
        gNarrowRatOrdA[index * 2 + 1] = kOrd.den[index];
    }

    for (int k = 0; k < detail::kNarrowBPieces; ++k)
    {
        const std::size_t index = static_cast<std::size_t>(k);
        gNarrowRatB[index * 2] = kB.num[index];
        gNarrowRatB[index * 2 + 1] = kB.den[index];
    }
}

// The six CUDA lanes' roles: the double batch seed evaluates the DOUBLE
// piece table even for the float/fp16 batch lanes (RoleUsesDoubleTables —
// the downward recursion amplifies float seed errors beyond their budgets).
template <double kAccuracyMultiplier> BoysStatus EnsureEffTables() {
    // Whether the device in hand already holds these tables is the .cu's
    // question: its record names the device as well as the multiplier, so
    // asking it is what keeps a device switch from being answered with the
    // previous device's tables.
    if (BoysCudaEffTablesResident(kAccuracyMultiplier) == 1)
    {
        return BoysStatus::kSuccess;
    }

    // The host-side tables depend on the multiplier alone, so they are computed
    // once per multiplier. The upload decides for itself as well: it answers for
    // whichever device it is about to write to.
    if (gEffCachedM != kAccuracyMultiplier)
    {
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleSingle, true>(0);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kDoubleBatch, true>(1);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Single, false>(2);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Batch, true>(3);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Single, false>(4);
        FillEffLane<kAccuracyMultiplier, detail::BoysRole::kF32Fp16Batch, true>(5);
        FillNarrowLane<kAccuracyMultiplier>();
        FillNarrowF32Lane<kAccuracyMultiplier>();
        FillMonoLane<kAccuracyMultiplier>();
        FillNarrowMonoLane<kAccuracyMultiplier>();
        FillRatLane<kAccuracyMultiplier>();
        FillNarrowRatLane<kAccuracyMultiplier>();
        gEffCachedM = kAccuracyMultiplier;
    }

    return FromLaunchCode(BoysCudaUploadEffTables(kAccuracyMultiplier,
                                                 gEffDegA.data(),
                                                 gEffDegB.data(),
                                                 gNarrowDegA.data(),
                                                 gNarrowDegB.data(),
                                                 gNarrowDegB32.data(),
                                                 gMonoDegA.data(),
                                                 gMonoDegB.data(),
                                                 gNarrowMonoDegA.data(),
                                                 gNarrowMonoDegB.data(),
                                                 gRatSeedA.data(),
                                                 gRatOrdA.data(),
                                                 gRatB.data(),
                                                 gNarrowRatSeedA.data(),
                                                 gNarrowRatOrdA.data(),
                                                 gNarrowRatB.data()));
}

// ---------------------------------------------------------------------------
// The run-time rung of the launched entries.
//
// Every entry of the surface above is instantiated once per rung, and which rung
// a call answers at is a property of the instantiation the call site names. The
// AtRung siblings take the rung as a run-time argument instead, and this is
// where that argument becomes one of those instantiations: each arm below is one
// rung of kDeviceRungs in the table's own order, and its body is the same two
// steps the template spelling takes — make that rung resident, then run that
// rung's launcher.
//
// The rung named is therefore the rung that becomes resident and the rung whose
// arithmetic runs, and a call answers at the rung it was handed and never at
// another. One rung is resident at a time, on the per-(device, m) upload this
// entry shares with the device-callable ones, so an AtRung call at a relaxed
// rung retires whichever other relaxed rung was resident and a device-callable
// entry asked for that rung then reports it. m = 1 has no relaxed table, is
// resident from the first upload, and retires nothing.
//
// A multiplier the table does not hold has no arm and is refused: no launcher is
// reached and the caller's output is untouched. Answering instead at whichever
// rung happens to be resident is the one outcome the rung argument exists to
// rule out, and a value outside the twelve is resident at none of them.
// ---------------------------------------------------------------------------

// One rung's own step: the tables, then the launcher that reads them. The
// launchers of a rung are not templates — one compiled launcher serves every
// relaxed rung, because what selects the rung is which degree tables are
// resident rather than which kernel runs — so this is the whole of what an arm
// adds over the reference rung's.
template <double kAccuracyMultiplier, typename Eff, typename... Args>
BoysStatus LaunchEffRung(Eff eff, Args... args) {
    const BoysStatus status = EnsureEffTables<kAccuracyMultiplier>();

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    return RunLaunch(eff, args...);
}

// The dispatch every AtRung entry is written with: `full` is the entry's
// full-accuracy launcher, `eff` the one that reads the resident rung, and
// `args` the entry's own launch arguments.
template <typename Full, typename Eff, typename... Args>
BoysStatus LaunchAtRung(double multiplier, Full full, Eff eff, Args... args) {
    switch (DeviceRungIndex(multiplier))
    {
    case 0:
        return RunLaunch(full, args...);
    case 1:
        return LaunchEffRung<2.0>(eff, args...);
    case 2:
        return LaunchEffRung<10.0>(eff, args...);
    case 3:
        return LaunchEffRung<64.0>(eff, args...);
    case 4:
        return LaunchEffRung<100.0>(eff, args...);
    case 5:
        return LaunchEffRung<256.0>(eff, args...);
    case 6:
        return LaunchEffRung<1024.0>(eff, args...);
    case 7:
        return LaunchEffRung<4096.0>(eff, args...);
    case 8:
        return LaunchEffRung<1e4>(eff, args...);
    case 9:
        return LaunchEffRung<16384.0>(eff, args...);
    case 10:
        return LaunchEffRung<65536.0>(eff, args...);
    case 11:
        return LaunchEffRung<1e8>(eff, args...);
    default:
        return BoysStatus::kInvalidArgument;
    }
}

} // namespace

BoysStatus BoysCuda::InitializeTables() {
    return FromLaunchCode(BoysCudaUploadTables());
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::DeviceTables(BoysDeviceTables* out) {
    if (out == nullptr)
    {
        return BoysStatus::kInvalidArgument;
    }

    // The address order BoysCudaDeviceTableAddresses fills, one slot per
    // symbol: the double lane's pieceStart, offset, a, b, deg, coeffs and
    // region-B seed, then the float lane's seven, then the relaxed image's
    // three — the resident rung's scalar, its region-A degrees and its
    // region-B degrees — then the uniform grid's four pools. Both sides state
    // the order; the .cu cannot name this type and this file cannot name a
    // symbol.
    void* addresses[21] = {};
    const BoysStatus status = FromLaunchCode(BoysCudaDeviceTableAddresses(addresses));

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    // The rest of the order, in the array the second export fills. Its width is
    // the handle's own count of the fields those slots belong to, so the two
    // halves of the order cannot come to disagree about where the second one
    // starts, and the device image asserts that count against the order's own
    // length.
    void* tail[kBoysDeviceTablesTailCount] = {};
    const BoysStatus tailStatus =
        FromLaunchCode(BoysCudaDeviceTableAddressesTail(tail));

    if (tailStatus != BoysStatus::kSuccess)
    {
        return tailStatus;
    }

    // The rung is made resident before the handle that reads it is handed over.
    // The full-accuracy tables are not in that image: a call at m = 1 has
    // nothing to upload and leaves whatever relaxed rung is resident in place.
    if constexpr (kAccuracyMultiplier != kBoysFullAccuracyMultiplier)
    {
        const BoysStatus rung = EnsureEffTables<kAccuracyMultiplier>();

        if (rung != BoysStatus::kSuccess)
        {
            return rung;
        }
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
    tables.relaxedRung = static_cast<const double*>(addresses[14]);
    const int* const relaxedA = static_cast<const int*>(addresses[15]);
    const int* const relaxedB = static_cast<const int*>(addresses[16]);

    for (int lane = 0; lane < kEffLaneCount; ++lane)
    {
        // The flat image's lane stride is the wider of the two piece tables
        // (boys_cuda.cu counts the same one from the same constants), so one
        // axis serves both.
        tables.relaxedDegA[lane] = relaxedA + lane * kRelaxedStride;
        tables.relaxedDegB[lane] = relaxedB + lane * (kEffMaxOrder + 1);
    }

    // The uniform grid's pools. They are the same image at every multiplier and
    // are never retired: each cell carries the degree the grid's own cell law
    // placed it at, a rung's degree table says nothing about it, and there is no
    // relaxed image of it to make resident.
    tables.flatCoeffs = static_cast<const double*>(addresses[17]);
    tables.flatMonoCoeffs = static_cast<const double*>(addresses[18]);
    tables.flatCoeffs32 = static_cast<const float*>(addresses[19]);
    tables.flatMonoCoeffs32 = static_cast<const float*>(addresses[20]);

    // The narrow partition's and the fit route's tables, in the order the tail
    // export fills them: the partition's double lane (slots 0 to 13), its float
    // lane (14 to 23), the route on the shipped partition and then on the narrow
    // one (24 to 34 and 35 to 45), and the route's region-B pair in the float
    // lane (46 to 49).
    //
    // Every one of these is read by an entry of boys_cuda_device.hpp, and a
    // handle that carried none of them refused every such call with
    // kTablesNotReady: the entries existed and the tables they read were
    // resident, and nothing handed the two to each other. The names are the
    // handle's own fields and the slots are the order's positions, which is the
    // one thing these two files have to state twice — the .cu cannot name this
    // type and this file cannot name a symbol.
    tables.narrowPieceStart = static_cast<const int*>(tail[0]);
    tables.narrowPieceOffset = static_cast<const int*>(tail[1]);
    tables.narrowPieceA = static_cast<const double*>(tail[2]);
    tables.narrowPieceB = static_cast<const double*>(tail[3]);
    tables.narrowStoredDeg = static_cast<const int*>(tail[4]);
    tables.narrowCoeffs = static_cast<const double*>(tail[5]);
    tables.narrowMonoCoeffs = static_cast<const double*>(tail[6]);
    tables.narrowBEdges = static_cast<const double*>(tail[7]);
    tables.narrowBCoeffs = static_cast<const double*>(tail[8]);
    tables.narrowBMonoCoeffs = static_cast<const double*>(tail[9]);
    tables.narrowRelaxedDegA = static_cast<const int*>(tail[10]);
    tables.narrowRelaxedDegB = static_cast<const int*>(tail[11]);
    tables.narrowMonoRelaxedDegA = static_cast<const int*>(tail[12]);
    tables.narrowMonoRelaxedDegB = static_cast<const int*>(tail[13]);
    tables.narrowPieceStart32 = static_cast<const int*>(tail[14]);
    tables.narrowPieceOffset32 = static_cast<const int*>(tail[15]);
    tables.narrowPieceA32 = static_cast<const float*>(tail[16]);
    tables.narrowPieceB32 = static_cast<const float*>(tail[17]);
    tables.narrowStoredDeg32 = static_cast<const int*>(tail[18]);
    tables.narrowCoeffs32 = static_cast<const float*>(tail[19]);
    tables.narrowMonoCoeffs32 = static_cast<const float*>(tail[20]);
    tables.narrowBEdges32 = static_cast<const float*>(tail[21]);
    tables.narrowBCoeffs32 = static_cast<const float*>(tail[22]);
    tables.narrowBMonoCoeffs32 = static_cast<const float*>(tail[23]);
    tables.ratCoeffs = static_cast<const double*>(tail[24]);
    tables.ratOffset = static_cast<const int*>(tail[25]);
    tables.ratDenOffset = static_cast<const int*>(tail[26]);
    tables.ratNumDeg = static_cast<const int*>(tail[27]);
    tables.ratDenDeg = static_cast<const int*>(tail[28]);
    tables.ratBNum = static_cast<const double*>(tail[29]);
    tables.ratBDen = static_cast<const double*>(tail[30]);
    tables.ratRelaxedDegB = static_cast<const int*>(tail[31]);
    tables.ratSeedDeg = static_cast<const int*>(tail[32]);
    tables.ratBNum32 = static_cast<const float*>(tail[33]);
    tables.ratBDen32 = static_cast<const float*>(tail[34]);
    tables.narrowRatCoeffs = static_cast<const double*>(tail[35]);
    tables.narrowRatOffset = static_cast<const int*>(tail[36]);
    tables.narrowRatDenOffset = static_cast<const int*>(tail[37]);
    tables.narrowRatNumDeg = static_cast<const int*>(tail[38]);
    tables.narrowRatDenDeg = static_cast<const int*>(tail[39]);
    tables.narrowRatSeedDeg = static_cast<const int*>(tail[40]);
    tables.narrowRatBCoeffs = static_cast<const double*>(tail[41]);
    tables.narrowRatBOffset = static_cast<const int*>(tail[42]);
    tables.narrowRatBStoredNumDeg = static_cast<const int*>(tail[43]);
    tables.narrowRatBDenDeg = static_cast<const int*>(tail[44]);
    tables.narrowRatRelaxedDegB = static_cast<const int*>(tail[45]);
    tables.narrowRatBCoeffs32 = static_cast<const float*>(tail[46]);
    tables.narrowRatBOffset32 = static_cast<const int*>(tail[47]);
    tables.narrowRatBStoredNumDeg32 = static_cast<const int*>(tail[48]);
    tables.narrowRatBDenDeg32 = static_cast<const int*>(tail[49]);

    // The same grid's two per-interval tables per lane, which its bodies
    // address a cell with: one degree and one block start per interval. They are
    // read out of the tail export's last four slots, the group the handle's own
    // constant names, and they are handed over with the coefficients they
    // describe — a handle carrying the pools without these would be read at a
    // block length the table does not have, which is what the route's readiness
    // test refuses.
    tables.flatDegs = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid]);
    tables.flatOffsets = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 1]);
    tables.flatDegs32 = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 2]);
    tables.flatOffsets32 = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 3]);

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

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        // Byte-identical to the full-accuracy path.
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
BoysStatus BoysCuda::AllOrdersF64Orders(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64Orders, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64Narrow(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        // The full-accuracy kernels read the degrees the partition was stored
        // at, which BoysCudaUploadTables has already placed; nothing about the
        // rung is uploaded here, so this path leaves a resident rung alone.
        return RunLaunch(BoysCudaLaunchAllOrdersF64Narrow, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64NarrowOrders(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrders, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersEff, n, x, out, count, stream);
    }
}

// The monomial scheme's four shapes: the entries above' path, with the launcher
// naming the pool and the summation the scheme reads.
template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64Mono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64Mono, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64MonoEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64OrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersMono, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersMonoEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64NarrowMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowMono, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowMonoEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersMono, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff, n, x, out, count, stream);
    }
}

// The fit route's four shapes, on the same path: the launcher names the pair a
// piece is read from, and the route's cut is per reading, so each shape's
// launcher names which of the two cuts it reads.
template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64Rat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64Rat, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64RatEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64OrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersRat, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersRatEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64NarrowRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowRat, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowRatEff, n, x, out, count, stream);
    }
}

template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == 1.0)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersRat, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff, n, x, out, count, stream);
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

// The AtRung siblings of the entries above: one definition each, one rung
// dispatch each. They are not templates — the rung is the call's argument — and
// every one of them is the entry's own prologue followed by LaunchAtRung, which
// is where the argument becomes a rung of kDeviceRungs. CheckOrder comes before
// the launch where the entry checks it, so an nmax outside the range is reported
// whether or not the rung is served and whatever the batch's count is.


// The f32 single sibling is a template for the same reason the entry is: the
// exponential selects which arithmetic runs and not how much accuracy is bought,
// so it stays where the call site writes it while the rung moves into the call.
// Both of the entry's calibrated pairs are defined, which is what its own two
// instantiations are.
template <RegionBExp kExp>
BoysStatus BoysCuda::SingleF32AtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count, void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kExp == RegionBExp::kFast)
    {
        return LaunchAtRung(multiplier, BoysCudaLaunchSingleF32Fast, BoysCudaLaunchSingleF32EffFast,
                            n, x, out, count, stream);
    }
    else
    {
        return LaunchAtRung(multiplier, BoysCudaLaunchSingleF32, BoysCudaLaunchSingleF32Eff, n, x,
                            out, count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32AtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF32,
                        BoysCudaLaunchAllOrdersF32Eff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllNF32AtRung(
    double multiplier, int nmax, const double* x, float* out, std::size_t count,
    void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllNF32,
                        BoysCudaLaunchAllNF32Eff, nmax, x, out, count, stream);
}


BoysStatus BoysCuda::SingleF64AtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchSingleF64,
                        BoysCudaLaunchSingleF64Eff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64AtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64,
                        BoysCudaLaunchAllOrdersF64Eff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64Narrow,
                        BoysCudaLaunchAllOrdersF64NarrowEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64OrdersAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64Orders,
                        BoysCudaLaunchAllOrdersF64OrdersEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowOrdersAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64NarrowOrders,
                        BoysCudaLaunchAllOrdersF64NarrowOrdersEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64MonoAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64Mono,
                        BoysCudaLaunchAllOrdersF64MonoEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64OrdersMonoAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64OrdersMono,
                        BoysCudaLaunchAllOrdersF64OrdersMonoEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowMonoAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64NarrowMono,
                        BoysCudaLaunchAllOrdersF64NarrowMonoEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMonoAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64NarrowOrdersMono,
                        BoysCudaLaunchAllOrdersF64NarrowOrdersMonoEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64RatAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64Rat,
                        BoysCudaLaunchAllOrdersF64RatEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64OrdersRatAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64OrdersRat,
                        BoysCudaLaunchAllOrdersF64OrdersRatEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowRatAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64NarrowRat,
                        BoysCudaLaunchAllOrdersF64NarrowRatEff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRatAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF64NarrowOrdersRat,
                        BoysCudaLaunchAllOrdersF64NarrowOrdersRatEff, n, x, out, count, stream);
}


namespace {

// The rung-argument sibling of the twelve rows whose rung axis is not the
// lane's whole set: the uniform route's six rows, and the float lane's narrow
// partition and rational route.
//
// One body and not twelve, which is what the caller gets, and the entry is its
// first argument rather than a comment beside it: which rungs a call answers at
// is the entry's own axis, read from DeviceEntryServedAtRung
// (boys_cuda_options.hpp) — the one statement of it, which the option space's
// rows, the probe and this dispatch all read rather than a list each. A rung the
// entry does not serve is refused here, before anything is launched and before
// anything is written, rather than answered at another rung's arithmetic.
//
// What a served rung delivers is the row's own: the dispatch is LaunchAtRung's,
// so the rung named is the rung made resident and the launcher run is the row's
// own. The two kinds of entry that reach this helper differ in which rung that
// is, and each difference is stated where the row is:
//
//  - the uniform route's six rows, whose table is stored at one degree for
//    every order and every interval: a rung of it is answered by that one
//    degree, which the criterion admits at every multiplier (its Delta(deg) is
//    zero, so the full degree is always admissible — boys_effective_degrees.hpp,
//    EffectiveDegree). Every rung is served by the route's own coefficients, the
//    row's bound at a rung is its figure times m, and what a rung buys is no
//    less work: the shorter degree a relaxation would read is not derived here,
//    and deriving one is owed work rather than a property of the route; and
//  - the float lane's narrow partition and its rational route, whose cut this
//    lane does not hold: their relaxed image is a cut of tables this lane has
//    never derived, uploaded or read, so the entry serves the reference
//    multiplier alone and the rest is refused here.
template <typename Launcher, typename Value>
BoysStatus RungServedByEntry(
    DeviceEntry entry,
    double multiplier,
    Launcher launcher,
    const int* n,
    const double* x,
    Value* out,
    std::size_t count,
    void* stream) {
    if (!DeviceEntryServedAtRung(entry, multiplier))
    {
        return BoysStatus::kInvalidArgument;
    }

    return LaunchAtRung(multiplier, launcher, launcher, n, x, out, count, stream);
}

} // namespace


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64Uniform(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    // The route's table is stored at one degree for every order and every
    // interval, so no rung's criterion cuts it and there is no shorter image of
    // it to make resident: every rung's arithmetic is this one degree, which is
    // why the multiplier selects no table here and the entry serves every rung
    // of the lane. A call at a rung makes that rung resident, so a
    // device-callable entry asked at it afterwards finds it, and the arithmetic
    // it delivers is the route's at every rung.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64Uniform, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF64Uniform, n, x, out,
                                                 count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF64UniformAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF64Uniform, multiplier,
                             BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64UniformHorner(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    // The other form of the same table, at the same contract: the route's rung
    // axis is empty for the monomial image exactly as it is for the Chebyshev
    // one, because the degree is a property of the stored table and not of the
    // basis it is summed in, so every rung is served and reads this one degree.
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64UniformHorner, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out,
                                                  count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF64UniformHornerAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF64UniformHorner, multiplier,
                             BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64OrdersUniform(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64OrdersUniform,
                                         kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The route's one reading of the grid, and therefore the kernel
    // AllOrdersF64Uniform launches: the packing axis this entry names has one
    // member here, for the reason the header's declaration states.
    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF64Uniform, n, x, out,
                                                 count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF64OrdersUniformAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF64OrdersUniform, multiplier,
                             BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner(
    const int* n, const double* x, double* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF64OrdersUniformHorner,
                                         kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out,
                                                  count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF64OrdersUniformHornerAtRung(
    double multiplier, const int* n, const double* x, double* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF64OrdersUniformHorner, multiplier,
                             BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32Uniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The float lane's uniform table is stored at one degree for every order and
    // interval exactly as the double lane's is, so the same contract holds: no
    // rung's criterion cuts it, every rung is served, and every one of them
    // reads this one degree. What the route's own placement of that degree buys
    // is the lane's budget and nothing per rung.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Uniform, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF32Uniform, n, x, out,
                                                 count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32UniformAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32Uniform, multiplier,
                             BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32UniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32UniformHorner, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out,
                                                  count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32UniformHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32UniformHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32Narrow(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The lane's narrow partition has a cut to make at a rung, and this lane
    // holds it: region B's degrees are the float lane's own
    // (NarrowRegionBDegrees over the float pieces, FillNarrowF32Lane) and are
    // resident from the rung's own upload. Region A needs no table beside them
    // because this lane's region-A seed is the double lane's, whose cut the same
    // upload carries.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Narrow, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32Narrow, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowEff, n, x, out, count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32NarrowAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF32Narrow,
                        BoysCudaLaunchAllOrdersF32NarrowEff, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The other form of the partition above, at the same contract and for the
    // same reason: the cut is the monomial image's and this lane does not hold
    // it either.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowMono, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowMono, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowMonoAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowMono, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowMono, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32Rat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The float lane's rational route serves the reference multiplier only, and
    // this is unbuilt work rather than a property of the route: the device lane
    // derives and uploads the double role's cut alone, so neither the float
    // lane's narrow degrees nor its rational pairs' cuts are held here, and a
    // relaxed call would have to be answered by degrees no kernel of this lane
    // has. The refusal is here, where a rung would be named.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Rat, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Rat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32RatAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32Rat, multiplier,
                             BoysCudaLaunchAllOrdersF32Rat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32RatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The other scheme name of the route above, over the same pair: both names a
    // caller may use reach one kernel and one arithmetic, and one rung axis.
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32RatHorner, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Rat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32RatHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32RatHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32Rat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The route above on the narrow partition, refused for the same missing
    // table: the float lane's region-B pairs on that partition.
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowRat, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowRatAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowRat, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowRat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The other scheme name of the pair above, over the same kernel.
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowRatHorner, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowRatHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowRatHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowRat, n, x, out, count, stream);
}


// The float lane's orders axis, one pair per shape: the entry whose multiplier is
// its template argument, and the rung-argument sibling. The halves differ in
// which rungs their rows serve and not in how a rung is reached — the shipped
// partition's cut is this lane's own and its row has the two forms a rung has,
// while the narrow partition's and the rational route's are stored at one rung
// here — and each states that through the immediate assertion below rather than
// by the shape of the body, so a row whose rungs move cannot leave a body behind.
template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32Orders(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32Orders, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32Orders, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF32OrdersEff, n, x, out, count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32OrdersAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    // The two launchers and not one, because this row's table has a cut to make:
    // a relaxed rung reads the rung's own degrees and a full-accuracy call the
    // stored ones, so the pair is what the compile-time spelling picks between
    // and what this sibling has to pick between as well. A row of the uniform
    // grid takes one launcher through RungServedByEntry for the opposite reason —
    // its table is stored at one degree per interval and there is no second
    // arithmetic to choose.
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF32Orders,
                        BoysCudaLaunchAllOrdersF32OrdersEff, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowOrders(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowOrders, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The same two tables its per-argument twin reads, on the other axis: the
    // rung is a property of the stored fit and not of the reading.
    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrders, n, x, out, count, stream);
    } else
    {
        const auto status = EnsureEffTables<kAccuracyMultiplier>();

        if (status != BoysStatus::kSuccess)
        {
            return status;
        }

        return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersEff, n, x, out, count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF32NarrowOrders,
                        BoysCudaLaunchAllOrdersF32NarrowOrdersEff, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowOrdersMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowOrdersMono, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersMono, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersMonoAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowOrdersMono, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowOrdersMono, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32OrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32OrdersRat, kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32OrdersRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32OrdersRatAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32OrdersRat, multiplier,
                             BoysCudaLaunchAllOrdersF32OrdersRat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32OrdersRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32OrdersRatHorner, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32OrdersRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32OrdersRatHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32OrdersRatHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32OrdersRat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowOrdersRat, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRatAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowOrdersRat, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowOrdersRat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner,
                                         kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersRat, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRatHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32NarrowOrdersRat, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32OrdersUniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    // The route's one reading of the grid, and therefore the kernel
    // AllOrdersF32Uniform launches: the packing axis this entry names has one
    // member here, for the reason the header's declaration states.
    static_assert(
        DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32OrdersUniform, kAccuracyMultiplier),
        "this entry does not serve the rung this instantiation names: which rungs it serves is "
        "DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF32Uniform, n, x, out,
                                                 count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32OrdersUniformAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32OrdersUniform, multiplier,
                             BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream);
}


template <double kAccuracyMultiplier>
BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream) {
    static_assert(DeviceEntryServedAtRung(DeviceEntry::kAllOrdersF32OrdersUniformHorner,
                                         kAccuracyMultiplier),
                  "this entry does not serve the rung this instantiation names: which rungs it "
                  "serves is DeviceEntryServedAtRung (boys_cuda_options.hpp)");

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    if constexpr (kAccuracyMultiplier == kBoysFullAccuracyMultiplier)
    {
        return RunLaunch(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream);
    } else
    {
        return LaunchEffRung<kAccuracyMultiplier>(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out,
                                                  count, stream);
    }
}


BoysStatus BoysCuda::AllOrdersF32OrdersUniformHornerAtRung(
    double multiplier, const int* n, const double* x, float* out, std::size_t count,
    void* stream) {
    return RungServedByEntry(DeviceEntry::kAllOrdersF32OrdersUniformHorner, multiplier,
                             BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllNF64AtRung(
    double multiplier, int nmax, const double* x, double* out, std::size_t count,
    void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllNF64,
                        BoysCudaLaunchAllNF64Eff, nmax, x, out, count, stream);
}


#if BoysFp16

BoysStatus BoysCuda::SingleF16AtRung(
    double multiplier, const int* n, const F16* x, F16* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchSingleF16,
                        BoysCudaLaunchSingleF16Eff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllOrdersF16AtRung(
    double multiplier, const int* n, const F16* x, F16* out, std::size_t count,
    void* stream) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllOrdersF16,
                        BoysCudaLaunchAllOrdersF16Eff, n, x, out, count, stream);
}


BoysStatus BoysCuda::AllNF16AtRung(
    double multiplier, int nmax, const F16* x, F16* out, std::size_t count,
    void* stream) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return LaunchAtRung(multiplier, BoysCudaLaunchAllNF16,
                        BoysCudaLaunchAllNF16Eff, nmax, x, out, count, stream);
}

#endif // BoysFp16

// ---------------------------------------------------------------------------
// Explicit instantiations at the rungs this lane serves, kDeviceRungs
// (boys_cuda_options.hpp). The entry definitions live in this TU (the header
// stays CUDA-runtime-free), so the call sites in other TUs link only the
// instantiations spelled out here — m = 1.0 first (the full-accuracy pin), then
// this lane's own relaxation sample set, then the option space's rungs below.
// The f32 single entry is instantiated once per calibrated (multiplier,
// exponential) pair it offers; the uniform route's six entries are instantiated
// at every rung of the lane, the eleven relaxed ones at the end of this list,
// because that is the axis their rows state; every other entry has one
// arithmetic and one instantiation per multiplier.
// ---------------------------------------------------------------------------
// The one rung-argument sibling that is itself a template: the f32 single
// entry's exponential axis is a compile-time choice of arithmetic and stays one,
// so this entry has two instantiations of its sibling and not twelve. Which rung
// it answers at is the run-time argument, dispatched in the definition above.
template BoysStatus BoysCuda::SingleF32AtRung<RegionBExp::kAccurate>(
    double, const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32AtRung<RegionBExp::kFast>(
    double, const int*, const double*, float*, std::size_t, void*);

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
template BoysStatus BoysCuda::AllOrdersF64Narrow<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<1.0>(
    const int*, const double*, double*, std::size_t, void*);
// The uniform route's reference instantiation, which is the first of the twelve
// each of its six entries carries: the route's rung axis is the lane's whole
// set, and the other eleven are spelled at the end of this list. A rung of this
// route is not a second arithmetic — there is no shorter image of the table to
// cut — so what a rung adds over this instantiation is the residency of the rung
// the call names and nothing else.
template BoysStatus BoysCuda::AllOrdersF64Uniform<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<1.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<1.0>(
    const int*, const double*, double*, std::size_t, void*);
// The float lane's partition and grid carry the same one each, for their own
// reasons (boys_cuda_options.hpp, DeviceEntryServedAtRung).
template BoysStatus BoysCuda::AllOrdersF32Uniform<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowMono<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Rat<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32RatHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowRat<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowRatHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
// The float lane's orders axis at its full-accuracy form: the shipped
// partition's row and the two uniform-grid rows, which carry every rung, beside
// the narrow partition's and the rational route's, which carry this one alone.
template BoysStatus BoysCuda::AllOrdersF32Orders<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrdersMono<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersRat<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersRatHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRat<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRatHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<1.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<1.0>(
    const int*, const double*, float*, std::size_t, void*);
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
template BoysStatus BoysCuda::AllOrdersF64Narrow<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<2.0>(
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
template BoysStatus BoysCuda::AllOrdersF64Narrow<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<10.0>(
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
template BoysStatus BoysCuda::AllOrdersF64Narrow<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<100.0>(
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
template BoysStatus BoysCuda::AllOrdersF64Narrow<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<1e4>(
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
template BoysStatus BoysCuda::AllOrdersF64Narrow<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<1e8>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<1e8>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<1e8>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<1e8>(int, const F16*, F16*, std::size_t, void*);
#endif

// ---------------------------------------------------------------------------
// The option space's rungs on this lane: the CPU tier lane's seven multipliers
// (AccuracyTier), instantiated exactly as the lane's own rungs are — the same
// entries, the same degree tables cut at that multiplier — so the set the API
// answers for and the set the kernels are compiled at are one set,
// kDeviceRungs (boys_cuda_options.hpp), twelve multipliers wide.
// ---------------------------------------------------------------------------

template BoysStatus BoysCuda::SingleF32<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<64.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<64.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<64.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<64.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<64.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<64.0>(int, const F16*, F16*, std::size_t, void*);
#endif

template BoysStatus BoysCuda::SingleF32<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<256.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<256.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<256.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<256.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<256.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<256.0>(int, const F16*, F16*, std::size_t, void*);
#endif

template BoysStatus BoysCuda::SingleF32<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<1024.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<1024.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<1024.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<1024.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<1024.0>(
    const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<1024.0>(int, const F16*, F16*, std::size_t, void*);
#endif

template BoysStatus BoysCuda::SingleF32<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<4096.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<4096.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<4096.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<4096.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<4096.0>(
    const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<4096.0>(int, const F16*, F16*, std::size_t, void*);
#endif

template BoysStatus BoysCuda::SingleF32<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<16384.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<16384.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<16384.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<16384.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<16384.0>(
    const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<16384.0>(int, const F16*, F16*, std::size_t, void*);
#endif

template BoysStatus BoysCuda::SingleF32<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF32<65536.0, RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF32<65536.0>(int, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::SingleF64<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Narrow<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Orders<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrders<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Mono<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersMono<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowMono<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Rat<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersRat<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowRat<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF64<65536.0>(int, const double*, double*, std::size_t, void*);
#if BoysFp16
template BoysStatus BoysCuda::SingleF16<65536.0>(const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF16<65536.0>(
    const int*, const F16*, F16*, std::size_t, void*);
template BoysStatus BoysCuda::AllNF16<65536.0>(int, const F16*, F16*, std::size_t, void*);
#endif

// The uniform route's rows at the rungs above them: the same eleven, for the
// same six entries, because the route's rung axis is the lane's whole set. The
// table is stored at one degree for every order and every interval, so each of
// these instantiations reads that one degree and the rung it names is the rung
// its call makes resident — the arithmetic is the route's at every one of them.
// Spelled per (entry, rung) and not shared, because the surface a caller writes
// against is a name and a rung: a rung missing here is a call site that does not
// link, which is the one thing the header's declaration cannot express.
template BoysStatus BoysCuda::AllOrdersF64Uniform<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64Uniform<1e8>(
    const int*, const double*, double*, std::size_t, void*);

template BoysStatus BoysCuda::AllOrdersF64UniformHorner<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64UniformHorner<1e8>(
    const int*, const double*, double*, std::size_t, void*);

template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniform<1e8>(
    const int*, const double*, double*, std::size_t, void*);

template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<2.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<10.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<64.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<100.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<256.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<1024.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<4096.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<1e4>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<16384.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<65536.0>(
    const int*, const double*, double*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner<1e8>(
    const int*, const double*, double*, std::size_t, void*);

template BoysStatus BoysCuda::AllOrdersF32Uniform<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Uniform<1e8>(
    const int*, const double*, float*, std::size_t, void*);

template BoysStatus BoysCuda::AllOrdersF32UniformHorner<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32UniformHorner<1e8>(
    const int*, const double*, float*, std::size_t, void*);

// The float lane's orders rows whose table has a cut to make — the shipped
// partition's, over the float batch lane's own cut of the float Chebyshev table
// — and the two uniform rows, which are the route's one reading of the grid at
// every rung. Their reference instantiations are above; these are the other
// eleven rungs each of them serves, spelled here for the same reason the uniform
// route's six are spelled in this block: the row's axis is the lane's whole set
// and the rung that makes it resident is the rung the call named.
template BoysStatus BoysCuda::AllOrdersF32Orders<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Orders<1e8>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniform<1e8>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner<1e8>(
    const int*, const double*, float*, std::size_t, void*);

// The narrow partition's two rows, whose rung cut this lane now derives,
// uploads and reads: the eleven relaxed instantiations each, beside the
// reference one in the block above.
template BoysStatus BoysCuda::AllOrdersF32Narrow<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<2.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<10.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<64.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<100.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<256.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<1024.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<4096.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<1e4>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<16384.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<65536.0>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32Narrow<1e8>(
    const int*, const double*, float*, std::size_t, void*);
template BoysStatus BoysCuda::AllOrdersF32NarrowOrders<1e8>(
    const int*, const double*, float*, std::size_t, void*);

// The handle for each rung the lane serves, kDeviceRungs (boys_cuda_options.hpp)
// — one instantiation per rung, the same list the entries above are compiled at,
// because the rung a handle is filled at is the rung its entries then read. m = 1
// is the default argument's own instantiation; each other makes its own rung
// resident.
template BoysStatus BoysCuda::DeviceTables<kBoysFullAccuracyMultiplier>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<2.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<10.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<64.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<100.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<256.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<1024.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<4096.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<1e4>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<16384.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<65536.0>(BoysDeviceTables*);
template BoysStatus BoysCuda::DeviceTables<1e8>(BoysDeviceTables*);

// ---------------------------------------------------------------------------
// The device option space.
//
// One row per option, and the rows are read from the entries above rather than
// from a list kept beside them: a name here is the name an entry is documented
// and reported under, a bound is the bound that entry states, and the degree
// lane is the lane its own documentation names.
//
// The fp16 rows are the build-time case of an unserved option: they are here
// whatever the seam is set to, with the reason when it is closed, so the space
// this revision defines is one number in every configuration.
// ---------------------------------------------------------------------------

#if BoysFp16
constexpr bool kFp16Served = true;
constexpr const char* kFp16Refusal = nullptr;
#else
constexpr bool kFp16Served = false;
constexpr const char* kFp16Refusal = "the fp16 seam is closed in this build (BoysFp16 = 0)";
#endif

// The documented forms, as the entries of this header state them. The
// multiplier m enters every one of them, and the two constant parts that are
// not the lane's own bound are the reason the form is carried beside the
// number: the fast f32 option's seed contribution and the fp16 lane's half
// ULP are terms a report must state and cannot fold into one figure.
constexpr const char* kFormF64 = "m * 5.5e-14";
constexpr const char* kFormF32 = "m * 1.5e-7";
constexpr const char* kFormF32Fast = "m * 1.5e-7 + 8e-8";
constexpr const char* kFormF16 = "m * 1e-7 + half an ULP of the returned value";

// The figures at m = 1, with any term a returned value decides dropped, which
// is the fp16 half ULP and nothing else: every other form is a number here.
constexpr double kBoundF64 = 5.5e-14;
constexpr double kBoundF32 = 1.5e-7;
constexpr double kBoundF32Fast = 1.5e-7 + 8e-8;
constexpr double kBoundF16 = 1e-7;

// The one bound term in this table that the accuracy multiplier does not scale:
// the fast region-B exponential's corrected seed, whose form adds 8e-8 to the
// truncated bound. It is the difference of the two figures above rather than a
// third spelling of 8e-8, so the two rows cannot drift from it.
constexpr double kFastSeedFixed = kBoundF32Fast - kBoundF32;

constexpr DeviceOptionInfo kDeviceOptions[] = {
    {DeviceEntry::kSingleF64, "single-fp64", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF64Single, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kSingleF32, "single-fp32", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kRegionBExp, RegionBExp::kAccurate, BoysDeviceLane::kF32Single, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kSingleF32Fast, "single-fp32-fast", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kRegionBExp, RegionBExp::kFast, BoysDeviceLane::kF32Single, kBoundF32Fast,
     kFormF32Fast, true, nullptr, kDefaultEvalScheme, kDefaultFitRoute, kFastSeedFixed},
    {DeviceEntry::kSingleF16, "single-fp16", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF16Single, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},

    {DeviceEntry::kAllOrdersF64, "all-orders-fp64", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF32, "all-orders-fp32", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF16, "all-orders-fp16", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},

    {DeviceEntry::kAllOrdersF64Narrow, "all-orders-fp64-narrow", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kPartition, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64Orders, "all-orders-fp64-orders", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kPacking, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowOrders, "all-orders-fp64-narrow-orders",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPacking, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64Mono, "all-orders-fp64-mono", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kScheme, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF64OrdersMono, "all-orders-fp64-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF64NarrowMono, "all-orders-fp64-narrow-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF64NarrowOrdersMono, "all-orders-fp64-narrow-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner},

    // The fit route's rows, one per scheme name the surface offers on it. Both
    // names select one arithmetic — the pair is stored once and summed by
    // Horner — so each pair of rows below runs one kernel and reports one
    // delivered figure, and the row's own \c scheme says which name reached it.
    {DeviceEntry::kAllOrdersF64Rat, "all-orders-fp64-rat", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kRoute, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr, EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64RatHorner, "all-orders-fp64-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64OrdersRat, "all-orders-fp64-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64OrdersRatHorner, "all-orders-fp64-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64NarrowRat, "all-orders-fp64-narrow-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64NarrowRatHorner, "all-orders-fp64-narrow-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64NarrowOrdersRat, "all-orders-fp64-narrow-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner, "all-orders-fp64-narrow-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},

    // The uniform route's four rows. One route and not four: the table stores
    // both forms of every fit and both are certified (the two rows of
    // kFlatRows), and it stores one fit per order per interval, which is the
    // packing axis's per-order member.
    //
    // So the four rows are two readings and four names. The scheme axis is a
    // real choice here — the Chebyshev image summed by a split Clenshaw against
    // the monomial image summed by Horner — and the packing axis is not: a grid
    // with one fit per order and per interval has no seeded ladder to step and
    // the route's fit refuses a band source at compile time, so the two rows
    // that name that axis run the route's own kernel. A pair of rows over one
    // arithmetic is what the fit route's own pairs are (see the block above),
    // and it is stated here for the same reason: a report that named two
    // arithmetics where the lane has one would be a report a chooser cannot act
    // on.
    //
    // Each row's \c bound is the lane's, not the table's own: a bound is what
    // the entry guarantees over the whole argument range it serves, and above
    // kFlatHi this route runs the same asymptotic every other double entry
    // runs, so the figure a caller places it by is the double batch lane's.
    {DeviceEntry::kAllOrdersF64Uniform, "all-orders-fp64-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kSplitClenshaw},
    {DeviceEntry::kAllOrdersF64UniformHorner, "all-orders-fp64-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF64OrdersUniform, "all-orders-fp64-orders-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kSplitClenshaw},
    {DeviceEntry::kAllOrdersF64OrdersUniformHorner, "all-orders-fp64-orders-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr, EvalScheme::kHorner},

    // The float lane's own partition and its own grid, which the device lane had
    // no row for. Both are the lane's tables and not a re-cut of the double
    // lane's: the narrow pieces are the float lane's 218 at degree 6 against the
    // double lane's 311 at degree 10, and the grid is 245 intervals at degree 4.
    //
    // The grid's two rows serve every rung of this lane, because no rung's
    // criterion cuts a table stored at one degree per order and interval: the
    // rung's arithmetic is that one degree, and what a rung of it does not buy is
    // less work. The narrow partition's Chebyshev row carries every rung: its cut
    // is the float lane's own region-B degrees, which this lane derives
    // (NarrowRegionBDegrees over the float pieces, FillNarrowF32Lane), uploads
    // and reads (Lane32NarrowRelaxed), beside the double lane's region-A cut the
    // same upload carries. Its monomial row is served at the full-accuracy
    // multiplier alone, and that is a different kind of statement: the monomial
    // basis's cut of that rung is one this lane does not derive, upload or read,
    // and building it in is owed work rather than a property of the partition.
    // Each row's own entry states which of the two it is, and the rung mask the
    // report carries is read from that statement and not listed here.
    {DeviceEntry::kAllOrdersF32Narrow, "all-orders-fp32-narrow",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw},
    {DeviceEntry::kAllOrdersF32NarrowMono, "all-orders-fp32-narrow-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF32Uniform, "all-orders-fp32-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw},
    {DeviceEntry::kAllOrdersF32UniformHorner, "all-orders-fp32-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner},

    // The float lane's rational route, one pair of rows per partition. The route
    // is a family and not a basis, so its pair is stored in one form and both
    // scheme names reach the one arithmetic — the pair is a numerator and a
    // denominator read by Horner — and each pair of rows below runs one kernel
    // and reports one delivered figure. The double lane's route carries the same
    // two pairs, and this lane's rows carry the float lane's region-B seed: the
    // region-A seed is the double lane's pair at the same partition, which is
    // what the entry's own contract states. The bound is the float lane's,
    // because the region-B seed and the ladder above it are the float lane's.
    {DeviceEntry::kAllOrdersF32Rat, "all-orders-fp32-rat", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kRoute, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr, EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32RatHorner, "all-orders-fp32-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32NarrowRat, "all-orders-fp32-narrow-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32NarrowRatHorner, "all-orders-fp32-narrow-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},

    // The float lane's other packing axis, the counterpart of the shipped,
    // narrow and rational rows above. The bound each row states is the row it
    // answers with: the axis changes which fit an order's value is read from in
    // region A and not the lane's arithmetic or the accuracy of the stored
    // table, so a row of it carries its per-argument twin's figure and form. The
    // rung mask is read from the entry, as every row's is — the shipped and the
    // two grid rows hold every rung, the narrow partition's and the route's the
    // full-accuracy one — and nothing here states it a second time.
    {DeviceEntry::kAllOrdersF32Orders, "all-orders-fp32-orders", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kPacking, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowOrders, "all-orders-fp32-narrow-orders",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPacking, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowOrdersMono, "all-orders-fp32-narrow-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner},
    {DeviceEntry::kAllOrdersF32OrdersRat, "all-orders-fp32-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32OrdersRatHorner, "all-orders-fp32-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32NarrowOrdersRat, "all-orders-fp32-narrow-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kSplitClenshaw,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner, "all-orders-fp32-narrow-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner,
     FitRoute::kRationalMinimax},
    {DeviceEntry::kAllOrdersF32OrdersUniform, "all-orders-fp32-orders-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kSplitClenshaw},
    {DeviceEntry::kAllOrdersF32OrdersUniformHorner, "all-orders-fp32-orders-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr, EvalScheme::kHorner},

    {DeviceEntry::kAllNF64, "all-n-fp64", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kAllNF32, "all-n-fp32", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kAllNF16, "all-n-fp16", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},

    {DeviceEntry::kDeviceSingleF64, "device-single-fp64", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF64Single, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kDeviceSingleF32, "device-single-fp32", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kRegionBExp, RegionBExp::kAccurate, BoysDeviceLane::kF32Single, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kDeviceSingleF32Fast, "device-single-fp32-fast",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32, DeviceOptionShape::kSingle,
     DeviceOptionQuestion::kSingle, DeviceOptionAxis::kRegionBExp, RegionBExp::kFast,
     BoysDeviceLane::kF32Single, kBoundF32Fast, kFormF32Fast, true, nullptr, kDefaultEvalScheme,
     kDefaultFitRoute, kFastSeedFixed},
    {DeviceEntry::kDeviceSingleF16, "device-single-fp16", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kSingle, DeviceOptionQuestion::kSingle,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF16Single, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},

    {DeviceEntry::kDeviceAllOrdersF64, "device-all-orders-fp64",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32, "device-all-orders-fp32",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF16, "device-all-orders-fp16",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp16,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served,
     kFp16Refusal},

    {DeviceEntry::kDeviceAllNF64, "device-all-n-fp64", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllNF32, "device-all-n-fp32", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllNF16, "device-all-n-fp16", DeviceOptionGroup::kDeviceCallable,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllN, DeviceOptionQuestion::kAllN,
     DeviceOptionAxis::kNone, RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},

    {DeviceEntry::kDeviceEachOrderF64, "device-each-order-fp64",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kEachOrder, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceEachOrderF32, "device-each-order-fp32",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kEachOrder, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceEachOrderF16, "device-each-order-fp16",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp16,
     DeviceOptionShape::kEachOrder, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kNone,
     RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served,
     kFp16Refusal},

    // The partition and route axes reached from the caller's own kernel. Each of
    // these is the option its launched row above names, in the group a caller
    // reaches through the handle instead of through a launch: the precision, the
    // shape, the question, the axis, the scheme, the route, the lane and the
    // bound are that row's, because the arithmetic is that row's, and the two
    // rows differ in the group and the name. The rung axis is the launched row's
    // too, and it is read from the launched row rather than stated again here
    // (DeviceEntryServedAtRung), so the two rows of an option cannot come to
    // disagree about which rungs this build holds a cut for.
    {DeviceEntry::kDeviceAllOrdersF64Narrow, "device-all-orders-fp64-narrow",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64NarrowMono, "device-all-orders-fp64-narrow-mono",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kHorner},
    {DeviceEntry::kDeviceAllOrdersF64Rat, "device-all-orders-fp64-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF64RatHorner, "device-all-orders-fp64-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kHorner, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF64NarrowRat, "device-all-orders-fp64-narrow-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner, "device-all-orders-fp64-narrow-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kHorner, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF64Uniform, "device-all-orders-fp64-uniform",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kSplitClenshaw},
    {DeviceEntry::kDeviceAllOrdersF64UniformHorner, "device-all-orders-fp64-uniform-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr,
     EvalScheme::kHorner},

    {DeviceEntry::kDeviceAllOrdersF32Narrow, "device-all-orders-fp32-narrow",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw},
    {DeviceEntry::kDeviceAllOrdersF32NarrowMono, "device-all-orders-fp32-narrow-mono",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kHorner},
    {DeviceEntry::kDeviceAllOrdersF32Rat, "device-all-orders-fp32-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF32RatHorner, "device-all-orders-fp32-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kHorner, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF32NarrowRat, "device-all-orders-fp32-narrow-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner, "device-all-orders-fp32-narrow-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kHorner, FitRoute::kRationalMinimax},
    {DeviceEntry::kDeviceAllOrdersF32Uniform, "device-all-orders-fp32-uniform",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kSplitClenshaw},
    {DeviceEntry::kDeviceAllOrdersF32UniformHorner, "device-all-orders-fp32-uniform-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr,
     EvalScheme::kHorner},
};

// The report's contract, checked at compile time: one row per DeviceEntry and
// row i is entry i.
constexpr bool DeviceOptionsAreInEnumeratorOrder() {
    if (std::size(kDeviceOptions) != static_cast<std::size_t>(DeviceEntry::kCount))
    {
        return false;
    }

    for (std::size_t i = 0; i < std::size(kDeviceOptions); ++i)
    {
        if (static_cast<std::size_t>(kDeviceOptions[i].entry) != i)
        {
            return false;
        }
    }

    return true;
}

static_assert(DeviceOptionsAreInEnumeratorOrder(),
              "the device option report has one row per DeviceEntry, in its enumerator order");
std::span<const DeviceOptionInfo> BoysDeviceOptions() noexcept {
    return kDeviceOptions;
}

} // namespace boys
