#include "boys/boys_cuda.hpp"

#include "boys/boys.hpp"
#include "boys/f16.hpp"
#include "boys/boys_effective_degrees.hpp"

#include <array>
#include <cstddef>
#include <iterator>

// Status layer of the CUDA lane. The kernels and the table uploads live in
// boys_cuda.cu (C++20, CUDA-safe include list only - the C++23 headers would poison
// the nvcc translation unit); this file wraps the exported plain functions in
// BoysStatus.
//
// Upload/launch return codes (boys_cuda.cu): 0 = success, 1 = internal table-layout
// error, otherwise a cudaError_t from the launch or the sync.

extern "C" {
int BoysCudaUploadTables();
int BoysCudaDeviceTableAddresses(void** out);
int BoysCudaDeviceTableAddressesTail(void** out);
int BoysCudaLaunchSingleF32(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF32Fast(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF32(
    int form, int nmax, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Uniform(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32UniformHorner(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32UniformRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64UniformRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Narrow(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Rat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32Orders(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrders(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32OrdersRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRat(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleF64(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Uniform(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64UniformHorner(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Orders(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Narrow(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrders(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Mono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMono(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64Rat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRat(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF64(
    int form, int nmax, const double* x, double* out, std::size_t count, void* stream);
#if BoysFp16
int BoysCudaLaunchSingleF16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllNF16(
    int form, int nmax, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Narrow(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16NarrowMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Rat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16NarrowRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Uniform(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16UniformHorner(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16UniformRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Orders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16NarrowOrders(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16NarrowOrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16OrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16NarrowOrdersRat(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
#endif
}

namespace boys {
namespace {

// code == 1 is the internal table-layout error; every other nonzero code is a
// cudaError_t from the launch or the sync. The status enum carries no payload -
// callers diagnose device failures through the CUDA runtime's own reporting.
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

// The division form a caller may name. One kernel holds one division (the launch picks
// the instantiation; a kernel never branches on it), so a value outside the three has no
// kernel to select and is refused here, before anything is launched.
BoysStatus CheckDivisionForm(DivisionForm form) {
    switch (form)
    {
    case DivisionForm::kExactDivision:
    case DivisionForm::kPlainReciprocal:
    case DivisionForm::kRefinedReciprocal:
        return BoysStatus::kSuccess;
    }

    return BoysStatus::kInvalidArgument;
}

// Shared launch path for every entry: the form check, the count == 0 no-op (a zero-block
// launch is a CUDA error, an empty batch is a success that writes nothing) and the one
// status mapping. Order is either entry's first argument; the form is the division the
// launched kernel is built with, and the launcher takes it as its own first argument.
template <typename Launcher, typename Order, typename X, typename Value>
BoysStatus RunLaunch(Launcher launcher,
                     Order order,
                     X x,
                     Value* out,
                     std::size_t count,
                     void* stream,
                     DivisionForm form) {
    const BoysStatus formStatus = CheckDivisionForm(form);

    if (formStatus != BoysStatus::kSuccess)
    {
        return formStatus;
    }

    if (count == 0)
    {
        return BoysStatus::kSuccess;
    }

    return FromLaunchCode(launcher(static_cast<int>(form), order, x, out, count, stream));
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

    // The address order BoysCudaDeviceTableAddresses fills, one slot per symbol: the double
    // lane's pieceStart, offset, a, b, deg, coeffs and region-B seed, then the float lane's
    // seven, then three reserved slots the export fills with null, then the uniform grid's four
    // pools. Both sides state the order; the .cu cannot name this type and this file cannot
    // name a symbol.
    void* addresses[21] = {};
    const BoysStatus status = FromLaunchCode(BoysCudaDeviceTableAddresses(addresses));

    if (status != BoysStatus::kSuccess)
    {
        return status;
    }

    // The rest of the order, in the array the second export fills. Its width is the handle's own
    // count of the fields those slots belong to, so the two halves cannot disagree about where
    // the second one starts, and the device image asserts that count against the order's length.
    void* tail[kBoysDeviceTablesTailCount] = {};
    const BoysStatus tailStatus =
        FromLaunchCode(BoysCudaDeviceTableAddressesTail(tail));

    if (tailStatus != BoysStatus::kSuccess)
    {
        return tailStatus;
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
    // Slots 14 to 16 are reserved and exported null: the handle's relaxed-residency fields stay
    // at the defaults their type gives them, and no reading of this handle consults them.

    // The uniform grid's pools: one image, never retired. Each cell carries the degree the
    // grid's own cell law placed it at, and no degree table beside it states anything about it.
    tables.flatCoeffs = static_cast<const double*>(addresses[17]);
    tables.flatMonoCoeffs = static_cast<const double*>(addresses[18]);
    tables.flatCoeffs32 = static_cast<const float*>(addresses[19]);
    tables.flatMonoCoeffs32 = static_cast<const float*>(addresses[20]);

    // The narrow partition's and the fit route's tables, in the order the tail export fills
    // them: the partition's double lane (slots 0 to 13), its float lane (14 to 23), the route on
    // the coarsest partition and then on the narrow one (24 to 34 and 35 to 45), and the route's
    // region-B pair in the float lane (46 to 49).
    //
    // The export fills the handles' *Relaxed* slots with null - those fields are read by no
    // entry of this revision - and they are assigned here all the same, because the slot order
    // is the one thing these two files have to state twice.
    //
    // A handle that carried none of these refused every call that reads one with
    // kTablesNotReady, the entries and the tables they read both existing. The names are the
    // handle's own fields and the slots are the order's positions - the .cu cannot name this
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
    tables.narrowRelaxedDegB32 = static_cast<const int*>(tail[50]);
    tables.narrowMonoRelaxedDegB32 = static_cast<const int*>(tail[51]);
    tables.ratRelaxedDegB32 = static_cast<const int*>(tail[52]);
    tables.narrowRatRelaxedDegB32 = static_cast<const int*>(tail[53]);

    // The same grid's two per-interval tables per lane, which its bodies address a cell with:
    // one degree and one block start per interval. They come out of the tail export's last four
    // slots, the group the handle's own constant names, and they are handed over with the
    // coefficients they describe - a handle carrying the pools without these would be read at a
    // block length the table does not have, which is what the route's readiness test refuses.
    tables.flatDegs = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid]);
    tables.flatOffsets = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 1]);
    tables.flatDegs32 = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 2]);
    tables.flatOffsets32 = static_cast<const int*>(tail[kBoysDeviceTablesTailFlatGrid + 3]);

    // The same two grids on their rational route, out of the two groups appended after that
    // one: the pool and the four columns one row is addressed with. Handed over together for
    // the reason the Chebyshev grid's are - the pool alone is a table no reader can step,
    // because this route's stride is the interval's own stored count and not a constant of the
    // grid - and the readiness test of that route is what a handle missing one of them fails.
    tables.flatRatCoeffs = static_cast<const double*>(tail[kBoysDeviceTablesTailRatGrid]);
    tables.flatRatNumDeg = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid + 1]);
    tables.flatRatDenDeg = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid + 2]);
    tables.flatRatStored = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid + 3]);
    tables.flatRatOffsets = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid + 4]);

    tables.flatRatCoeffs32 = static_cast<const float*>(tail[kBoysDeviceTablesTailRatGrid32]);
    tables.flatRatNumDeg32 = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid32 + 1]);
    tables.flatRatDenDeg32 = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid32 + 2]);
    tables.flatRatStored32 = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid32 + 3]);
    tables.flatRatOffsets32 = static_cast<const int*>(tail[kBoysDeviceTablesTailRatGrid32 + 4]);

    *out = tables;
    return BoysStatus::kSuccess;
}

template <RegionBExp kExp>
BoysStatus BoysCuda::SingleF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The two readings differ only in the region-B exponential.
    if constexpr (kExp == RegionBExp::kFast)
    {
        return RunLaunch(BoysCudaLaunchSingleF32Fast, n, x, out, count, stream, form);
    } else
    {
        return RunLaunch(BoysCudaLaunchSingleF32, n, x, out, count, stream, form);
    }
}

BoysStatus BoysCuda::AllOrdersF32(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllNF32(
    int nmax, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllNF32, nmax, x, out, count, stream, form);
}

BoysStatus BoysCuda::SingleF64(const int* n,
                               const double* x,
                               double* out,
                               std::size_t count,
                               void* stream,
                               DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchSingleF64, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64(const int* n,
                                  const double* x,
                                  double* out,
                                  std::size_t count,
                                  void* stream,
                                  DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64Orders(const int* n,
                                        const double* x,
                                        double* out,
                                        std::size_t count,
                                        void* stream,
                                        DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64Orders, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64Narrow(const int* n,
                                        const double* x,
                                        double* out,
                                        std::size_t count,
                                        void* stream,
                                        DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The kernels read the degrees the partition was stored at, which BoysCudaUploadTables
    // has already placed.
    return RunLaunch(BoysCudaLaunchAllOrdersF64Narrow, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64NarrowOrders(const int* n,
                                              const double* x,
                                              double* out,
                                              std::size_t count,
                                              void* stream,
                                              DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrders, n, x, out, count, stream, form);
}

// The monomial scheme's four shapes: the entries above's path, with the launcher naming the
// pool and the summation the scheme reads.
BoysStatus BoysCuda::AllOrdersF64Mono(const int* n,
                                      const double* x,
                                      double* out,
                                      std::size_t count,
                                      void* stream,
                                      DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64Mono, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64OrdersMono(const int* n,
                                            const double* x,
                                            double* out,
                                            std::size_t count,
                                            void* stream,
                                            DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersMono, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64NarrowMono(const int* n,
                                            const double* x,
                                            double* out,
                                            std::size_t count,
                                            void* stream,
                                            DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowMono, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64NarrowOrdersMono(const int* n,
                                                  const double* x,
                                                  double* out,
                                                  std::size_t count,
                                                  void* stream,
                                                  DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersMono, n, x, out, count, stream, form);
}

// The fit route's four shapes, on the same path: the launcher names the pair a piece is read
// from, and the route's cut is per reading, so each shape's launcher names which cut it reads.
BoysStatus BoysCuda::AllOrdersF64Rat(const int* n,
                                     const double* x,
                                     double* out,
                                     std::size_t count,
                                     void* stream,
                                     DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64Rat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64OrdersRat(const int* n,
                                           const double* x,
                                           double* out,
                                           std::size_t count,
                                           void* stream,
                                           DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64OrdersRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64NarrowRat(const int* n,
                                           const double* x,
                                           double* out,
                                           std::size_t count,
                                           void* stream,
                                           DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64NarrowOrdersRat(const int* n,
                                                 const double* x,
                                                 double* out,
                                                 std::size_t count,
                                                 void* stream,
                                                 DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64NarrowOrdersRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllNF64(
    int nmax, const double* x, double* out, std::size_t count, void* stream, DivisionForm form) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllNF64, nmax, x, out, count, stream, form);
}

#if BoysFp16
BoysStatus BoysCuda::SingleF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchSingleF16, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllNF16(
    int nmax, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    const auto valid = CheckOrder(nmax);

    if (valid != BoysStatus::kSuccess)
    {
        return valid;
    }

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllNF16, nmax, x, out, count, stream, form);
}

// The half lane's other bodies. Each of these is its float counterpart above at the same kernel
// body with the half store around it, so the line that decides what an entry runs is the one the
// float entry of that name already states; what these add is the fp16 store and its own figure.
BoysStatus BoysCuda::AllOrdersF16Narrow(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16Narrow, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowMono(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16NarrowMono, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16Uniform(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16Uniform, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16UniformHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16UniformHorner, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16Rat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The route's pair is stored in one form, so the two scheme names a caller may use reach this
    // one kernel and the twin below forwards here.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16Rat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16RatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair: both names reach one kernel
    // and one arithmetic.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16Rat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowRat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16NarrowRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowRatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16NarrowRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16UniformRat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16UniformRatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16UniformRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16Orders(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16Orders, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowOrders(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16NarrowOrders, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowOrdersMono(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16NarrowOrdersMono, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersRat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16OrdersRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersRatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16OrdersRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowOrdersRat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF16NarrowOrdersRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16NarrowOrdersRatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16NarrowOrdersRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersUniform(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The grid's packing axis has one member, so this entry and its float counterpart run the
    // per-argument kernel of the grid rather than a gather of their own.
    return RunLaunch(BoysCudaLaunchAllOrdersF16Uniform, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersUniformHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The grid's monomial blocks, the same kernel AllOrdersF16UniformHorner launches.
    return RunLaunch(BoysCudaLaunchAllOrdersF16UniformHorner, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersUniformRat(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The grid's rational route, one pair per interval and no second packing: the same kernel
    // AllOrdersF16UniformRat launches.
    return RunLaunch(BoysCudaLaunchAllOrdersF16UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF16OrdersUniformRatHorner(
    const int* n, const F16* x, F16* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF16OrdersUniformRat(n, x, out, count, stream, form);
}
#endif // BoysFp16

BoysStatus BoysCuda::AllOrdersF64Uniform(const int* n,
                                         const double* x,
                                         double* out,
                                         std::size_t count,
                                         void* stream,
                                         DivisionForm form) {
    // The route's table is stored at one degree for every order and interval, so there is no
    // shorter image of it to cut: that one degree is the arithmetic this entry documents.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF64UniformHorner(const int* n,
                                               const double* x,
                                               double* out,
                                               std::size_t count,
                                               void* stream,
                                               DivisionForm form) {
    // The other form of the same table, at the same contract: the degree is a property of the
    // stored table and not of the basis it is summed in.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF64OrdersUniform(const int* n,
                                               const double* x,
                                               double* out,
                                               std::size_t count,
                                               void* stream,
                                               DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The route's one reading of the grid, and therefore the kernel
    // AllOrdersF64Uniform launches: the packing axis this entry names has one
    // member here, for the reason the header's declaration states.
    return RunLaunch(BoysCudaLaunchAllOrdersF64Uniform, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF64OrdersUniformHorner(const int* n,
                                                     const double* x,
                                                     double* out,
                                                     std::size_t count,
                                                     void* stream,
                                                     DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64UniformHorner, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32Uniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The float lane's uniform table is stored at one degree for every order and interval
    // exactly as the double lane's is, so the same contract holds. What the route's own
    // placement of that degree buys is the lane's table budget.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32UniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream, form);
}


// The grid's rational member, whose contract is the entry above's with one thing different:
// the table it reads is one numerator/denominator pair per interval rather than one polynomial
// per interval, so the launch below is a kernel of its own and not the Chebyshev one.
BoysStatus BoysCuda::AllOrdersF32UniformRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF32UniformRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    return AllOrdersF32UniformRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64UniformRat(const int* n,
                                            const double* x,
                                            double* out,
                                            std::size_t count,
                                            void* stream,
                                            DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64UniformRatHorner(const int* n,
                                                  const double* x,
                                                  double* out,
                                                  std::size_t count,
                                                  void* stream,
                                                  DivisionForm form) {

    return AllOrdersF64UniformRat(n, x, out, count, stream, form);
}

// The grid's rational member over the route's other packing axis, which has one member here:
// the pair is stored per interval at the interval's own pair and stored count, so no gather
// has a stride to step and the launch below is the per-argument entry's kernel.
BoysStatus BoysCuda::AllOrdersF32OrdersUniformRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF32OrdersUniformRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    return AllOrdersF32OrdersUniformRat(n, x, out, count, stream, form);
}

// The same four on the double lane's grid: that lane's own pair per interval over its own
// intervals, read by that lane's launcher.
BoysStatus BoysCuda::AllOrdersF64OrdersUniformRat(const int* n,
                                                  const double* x,
                                                  double* out,
                                                  std::size_t count,
                                                  void* stream,
                                                  DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF64UniformRat, n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF64OrdersUniformRatHorner(const int* n,
                                                        const double* x,
                                                        double* out,
                                                        std::size_t count,
                                                        void* stream,
                                                        DivisionForm form) {

    return AllOrdersF64OrdersUniformRat(n, x, out, count, stream, form);
}

BoysStatus BoysCuda::AllOrdersF32Narrow(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The lane's narrow partition reads region B's degrees as the float lane's own
    // (NarrowRegionBDegrees over the float pieces). Region A needs no table beside them
    // because this lane's region-A seed is the double lane's.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Narrow, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The other form of the partition above: the degrees are the monomial coefficients' own
    // (NarrowRegionBDegrees at TailBasis::kMonomial). Region A is no second table here either:
    // this lane's region-A seed is the double lane's narrow monomial table.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowMono, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32Rat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The lane's rational route reads this lane's own fit: the region-B pair's degrees are its
    // own (RationalRegionBF32Degrees), and region A's seed is the double lane's pair.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Rat, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32RatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the route above, over the same pair: both names reach one
    // kernel and one arithmetic.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF32Rat(n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The route above on the narrow partition, whose region-B pair is this lane's
    // own fit there, read at its own degrees.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowRat, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The other scheme name of the pair above, over the same kernel.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF32NarrowRat(n, x, out, count, stream, form);
}


// The float lane's orders axis, one pair per shape: the same table its per-argument twin
// reads, gathered at each order's own piece rather than shared across the batch, so the
// coarsest partition under both its names, the narrow partition in both bases and the
// rational route out of its own float pairs each have both packings.
BoysStatus BoysCuda::AllOrdersF32Orders(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Orders, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrders(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The same two tables its per-argument twin reads, on the other axis: the
    // degrees are a property of the stored fit and not of the reading.
    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrders, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersMono(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    // The same two tables its per-argument twin reads, on the other axis: the
    // degrees are a property of the stored fit and not of the reading.
    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersMono, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32OrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32OrdersRat, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32OrdersRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF32OrdersRat(n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRat(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32NarrowOrdersRat, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32NarrowOrdersRatHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return AllOrdersF32NarrowOrdersRat(n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32OrdersUniform(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {
    // The route's one reading of the grid, and therefore the kernel
    // AllOrdersF32Uniform launches: the packing axis this entry names has one
    // member here, for the reason the header's declaration states.

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32Uniform, n, x, out, count, stream, form);
}


BoysStatus BoysCuda::AllOrdersF32OrdersUniformHorner(
    const int* n, const double* x, float* out, std::size_t count, void* stream, DivisionForm form) {

    if (BoysCuda::InitializeTables() != BoysStatus::kSuccess)
    {
        return BoysStatus::kDeviceError;
    }

    return RunLaunch(BoysCudaLaunchAllOrdersF32UniformHorner, n, x, out, count, stream, form);
}


// ---------------------------------------------------------------------------
// The one templated entry of this lane: the f32 single entry's compile-time choice
// of region-B exponential. The entry definitions live in this TU (the header stays
// CUDA-runtime-free), so call sites in other TUs link the instantiations spelled out
// here; every other entry of this lane is a plain function defined above.
// ---------------------------------------------------------------------------
template BoysStatus BoysCuda::SingleF32<RegionBExp::kAccurate>(
    const int*, const double*, float*, std::size_t, void*, DivisionForm);
template BoysStatus BoysCuda::SingleF32<RegionBExp::kFast>(
    const int*, const double*, float*, std::size_t, void*, DivisionForm);

// ---------------------------------------------------------------------------
// The device option space.
//
// One row per option, read from the entries above rather than from a list kept beside them: a
// name here is the name an entry is documented and reported under, a bound is the bound that
// entry states, and the degree lane is the lane its own documentation names.
//
// The fp16 rows are the build-time case of an unserved option: they are here whatever the seam
// is set to, with the reason when it is closed, so the space this revision defines is one
// number in every configuration.
// ---------------------------------------------------------------------------

#if BoysFp16
constexpr bool kFp16Served = true;
constexpr const char* kFp16Refusal = nullptr;
#else
constexpr bool kFp16Served = false;
constexpr const char* kFp16Refusal = "the fp16 seam is closed in this build (BoysFp16 = 0)";
#endif

// The documented forms, as the entries of this header state them. Two constant parts that are
// not the lane's own bound are why the form is carried beside the number: the fast f32 option's
// seed contribution and the fp16 lane's half ULP are terms a report must state and cannot fold
// into one figure.
constexpr const char* kFormF64 = "5.5e-14";
constexpr const char* kFormF32 = "1.5e-7";
constexpr const char* kFormF32Fast = "1.5e-7 + 8e-8";
constexpr const char* kFormF16 = "1e-7 + half an ULP of the returned value";

// The figures, with any term a returned value decides dropped, which is the
// fp16 half ULP and nothing else: every other form is a number here.
constexpr double kBoundF64 = 5.5e-14;
constexpr double kBoundF32 = 1.5e-7;
constexpr double kBoundF32Fast = 1.5e-7 + 8e-8;
constexpr double kBoundF16 = 1e-7;

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
     kFormF32Fast, true, nullptr},
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
     kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersMono, "all-orders-fp64-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowMono, "all-orders-fp64-narrow-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowOrdersMono, "all-orders-fp64-narrow-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    // The fit route's rows, one per scheme name the surface offers on it. Both names select one
    // arithmetic - the pair is stored once and summed by Horner - so each pair of rows below
    // runs one kernel and reports one delivered figure, and the row's own \c scheme says which
    // name reached it.
    {DeviceEntry::kAllOrdersF64Rat, "all-orders-fp64-rat", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kRoute, RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64,
     kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64RatHorner, "all-orders-fp64-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersRat, "all-orders-fp64-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersRatHorner, "all-orders-fp64-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowRat, "all-orders-fp64-narrow-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowRatHorner, "all-orders-fp64-narrow-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowOrdersRat, "all-orders-fp64-narrow-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64NarrowOrdersRatHorner, "all-orders-fp64-narrow-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    // The uniform route's four rows. One route and not four: the table stores both forms of
    // every fit and both are certified (the two rows of kFlatRows), and it stores one fit per
    // order per interval, which is the packing axis's per-order member.
    //
    // So the four rows are two readings and four names. The scheme axis is a real choice - the
    // Chebyshev image summed by a split Clenshaw against the monomial image summed by Horner -
    // and the packing axis is not: a grid with one fit per order and interval has no seeded
    // ladder to step and the route's fit refuses a band source at compile time, so the two rows
    // naming that axis run the route's own kernel. A report that named two arithmetics where the
    // lane has one would be a report a chooser cannot act on.
    //
    // Each row's \c bound is the lane's, not the table's own: a bound is what the entry
    // guarantees over the whole argument range it serves, and above kFlatHi this route runs the
    // same asymptotic every other double entry runs, so the figure a caller places it by is the
    // double batch lane's.
    {DeviceEntry::kAllOrdersF64Uniform, "all-orders-fp64-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64UniformHorner, "all-orders-fp64-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersUniform, "all-orders-fp64-orders-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersUniformHorner, "all-orders-fp64-orders-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    // The same grid on its RATIONAL route, the four rows the enumerators above name. The pair is
    // stored in the monomial form the family is stored in everywhere, so both scheme names reach
    // one arithmetic and each pair below runs one kernel. The bound is the double batch lane's
    // for the reason the Chebyshev grid's rows state; the partition is already named by the rows
    // above, so what these add is the family of fit over the grid.
    {DeviceEntry::kAllOrdersF64UniformRat, "all-orders-fp64-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64UniformRatHorner, "all-orders-fp64-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersUniformRat, "all-orders-fp64-orders-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kAllOrdersF64OrdersUniformRatHorner, "all-orders-fp64-orders-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp64, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    // The float lane's own partition and its own grid, which the device lane had no row for. Both
    // are the lane's tables and not a re-cut of the double lane's: the narrow pieces are the float
    // lane's 218 at degree 6 against the double lane's 311 at degree 10, and the grid is 245
    // intervals at degree 4.
    //
    // The grid's rows read the table's one degree per order and interval. The narrow
    // partition's two rows each carry their own basis: the Chebyshev table for the row above,
    // the monomial one for the row below.
    {DeviceEntry::kAllOrdersF32Narrow, "all-orders-fp32-narrow",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowMono, "all-orders-fp32-narrow-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32Uniform, "all-orders-fp32-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32UniformHorner, "all-orders-fp32-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},


    // The float lane's rational route, one pair of rows per partition. The route is a family
    // and not a basis, so its pair is stored in one form and both scheme names reach the one
    // arithmetic - a numerator and a denominator read by Horner - and each pair of rows below
    // runs one kernel and reports one delivered figure. The double lane's route carries the same
    // two pairs, and this lane's rows carry the float lane's region-B seed: the region-A seed is
    // the double lane's pair at the same partition, which is what the entry's own contract
    // states. The bound is the float lane's, because the region-B seed and the ladder above it
    // are the float lane's.
    {DeviceEntry::kAllOrdersF32Rat, "all-orders-fp32-rat", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kRoute, RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32,
     kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32RatHorner, "all-orders-fp32-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowRat, "all-orders-fp32-narrow-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowRatHorner, "all-orders-fp32-narrow-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    // The grid on the rational route, the float lane's own member over its own intervals. The
    // bound is this lane's, as the piecewise rational rows' is: the member is certified in this
    // lane's arithmetic against this lane's bar.
    {DeviceEntry::kAllOrdersF32UniformRat, "all-orders-fp32-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32UniformRatHorner, "all-orders-fp32-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},

    // The float lane's other packing axis, the counterpart of the coarsest, narrow and rational
    // rows above. The bound each row states is the row it answers with: the axis changes which
    // fit an order's value is read from in region A and not the lane's arithmetic or the accuracy
    // of the stored table, so a row of it carries its per-argument twin's figure and form.
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
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32OrdersRat, "all-orders-fp32-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32OrdersRatHorner, "all-orders-fp32-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowOrdersRat, "all-orders-fp32-narrow-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32NarrowOrdersRatHorner, "all-orders-fp32-narrow-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32OrdersUniform, "all-orders-fp32-orders-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32OrdersUniformHorner, "all-orders-fp32-orders-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},

    // The same two rows on the lane's other packing axis, which for this route has one member:
    // the grid's rows are stored per interval at the interval's own pair, so no gather has a
    // stride to step and the two rows run the per-argument kernel.
    {DeviceEntry::kAllOrdersF32OrdersUniformRat, "all-orders-fp32-orders-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kAllOrdersF32OrdersUniformRatHorner, "all-orders-fp32-orders-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp32, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},

    // The half lane's ladder family, the float lane's rows above entry for entry. The bound each
    // row states is this lane's own: the arithmetic is the float lane's and the store is this
    // lane's, so the figure is the float lane's with the half format's term, which is the fp16
    // row's form and not a second bound.
    {DeviceEntry::kAllOrdersF16Narrow, "all-orders-fp16-narrow",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowMono, "all-orders-fp16-narrow-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16Uniform, "all-orders-fp16-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16UniformHorner, "all-orders-fp16-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16Rat, "all-orders-fp16-rat", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kRoute, RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16RatHorner, "all-orders-fp16-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowRat, "all-orders-fp16-narrow-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowRatHorner, "all-orders-fp16-narrow-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16UniformRat, "all-orders-fp16-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16UniformRatHorner, "all-orders-fp16-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},

    // The half lane's other packing axis, the float lane's orders rows at this lane's store.
    {DeviceEntry::kAllOrdersF16Orders, "all-orders-fp16-orders", DeviceOptionGroup::kLaunched,
     DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders,
     DeviceOptionAxis::kPacking, RegionBExp::kAccurate, BoysDeviceLane::kF16Batch, kBoundF16,
     kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowOrders, "all-orders-fp16-narrow-orders",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPacking, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowOrdersMono, "all-orders-fp16-narrow-orders-mono",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersRat, "all-orders-fp16-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersRatHorner, "all-orders-fp16-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowOrdersRat, "all-orders-fp16-narrow-orders-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16NarrowOrdersRatHorner, "all-orders-fp16-narrow-orders-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersUniform, "all-orders-fp16-orders-uniform",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersUniformHorner, "all-orders-fp16-orders-uniform-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersUniformRat, "all-orders-fp16-orders-uniform-rat",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},
    {DeviceEntry::kAllOrdersF16OrdersUniformRatHorner, "all-orders-fp16-orders-uniform-rat-horner",
     DeviceOptionGroup::kLaunched, DeviceOptionPrecision::kFp16, DeviceOptionShape::kAllOrders,
     DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute, RegionBExp::kAccurate,
     BoysDeviceLane::kF16Batch, kBoundF16, kFormF16, kFp16Served, kFp16Refusal},

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
     BoysDeviceLane::kF32Single, kBoundF32Fast, kFormF32Fast, true, nullptr},
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

    // The partition and route axes reached from the caller's own kernel. Each of these is the
    // option its launched row above names, in the group a caller reaches through the handle
    // instead of through a launch: precision, shape, question, axis, scheme, route, lane and
    // bound are that row's, because the arithmetic is that row's, and the two rows differ in the
    // group and the name.
    {DeviceEntry::kDeviceAllOrdersF64Narrow, "device-all-orders-fp64-narrow",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64NarrowMono, "device-all-orders-fp64-narrow-mono",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64Rat, "device-all-orders-fp64-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64RatHorner, "device-all-orders-fp64-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64NarrowRat, "device-all-orders-fp64-narrow-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64NarrowRatHorner, "device-all-orders-fp64-narrow-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64Uniform, "device-all-orders-fp64-uniform",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64UniformHorner, "device-all-orders-fp64-uniform-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    // The grid's rational member, in the caller's own kernel. Every launched row of this lane has
    // a device-callable twin, and this one's is the same body: the entry reads the handle's
    // pointers and the reader behind it is the one the launched kernel runs.
    {DeviceEntry::kDeviceAllOrdersF64UniformRat, "device-all-orders-fp64-uniform-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF64UniformRatHorner, "device-all-orders-fp64-uniform-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp64,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF64Batch, kBoundF64, kFormF64, true, nullptr},

    {DeviceEntry::kDeviceAllOrdersF32Narrow, "device-all-orders-fp32-narrow",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32NarrowMono, "device-all-orders-fp32-narrow-mono",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32Rat, "device-all-orders-fp32-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32RatHorner, "device-all-orders-fp32-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32NarrowRat, "device-all-orders-fp32-narrow-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32NarrowRatHorner, "device-all-orders-fp32-narrow-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32Uniform, "device-all-orders-fp32-uniform",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kPartition,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32UniformHorner, "device-all-orders-fp32-uniform-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kScheme,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},

    {DeviceEntry::kDeviceAllOrdersF32UniformRat, "device-all-orders-fp32-uniform-rat",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
    {DeviceEntry::kDeviceAllOrdersF32UniformRatHorner, "device-all-orders-fp32-uniform-rat-horner",
     DeviceOptionGroup::kDeviceCallable, DeviceOptionPrecision::kFp32,
     DeviceOptionShape::kAllOrders, DeviceOptionQuestion::kAllOrders, DeviceOptionAxis::kRoute,
     RegionBExp::kAccurate, BoysDeviceLane::kF32Batch, kBoundF32, kFormF32, true, nullptr},
};

// The report's contract, checked at compile time: one row per DeviceEntry, row i is entry i.
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
