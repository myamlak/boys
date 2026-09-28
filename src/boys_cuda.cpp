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
int BoysCudaEffTablesResident(double m);
int BoysCudaUploadEffTables(double m,
                            const int* degA,
                            const int* degB,
                            const int* narrowA,
                            const int* narrowB,
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
int BoysCudaLaunchSingleF64(
    const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64(
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
    // region-B degrees. Both sides state the order; the .cu cannot name this
    // type and this file cannot name a symbol.
    void* addresses[17] = {};
    const BoysStatus status = FromLaunchCode(BoysCudaDeviceTableAddresses(addresses));

    if (status != BoysStatus::kSuccess)
    {
        return status;
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
// exponential) pair it offers; every other entry has one arithmetic and one
// instantiation per multiplier.
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
