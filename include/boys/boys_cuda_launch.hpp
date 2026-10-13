#pragma once

/// \file boys_cuda_launch.hpp
/// The CUDA lane's launch seam: the launchers boys_cuda.cu exports and the host code every
/// entry of boys_cuda.hpp runs around them.
///
/// The prototypes are the C symbols the .cu defines; the helpers are the checks and the one
/// status mapping the entries share, called from the entries' bodies. Host-side only, so no
/// CUDA runtime include: a stream is an opaque `void*` here and a cudaStream_t inside the .cu.

#include "boys/accuracy.hpp"
#include "boys/status.hpp"

#include <cstddef>

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
int BoysCudaLaunchAllNBf16Fast(int, int, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF16Fast(int, int, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllNF32Fast(int, int, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllNF64Fast(int, int, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16MonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16NarrowRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16OrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersBf16RatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16MonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersMonoFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersRatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16RatFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32Fast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32MonoFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowMonoFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersMonoFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRatFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRatFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersMonoFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRatFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32RatFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchSingleBf16Fast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32RatHornerFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32OrdersRatHornerFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowRatHornerFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF32NarrowOrdersRatHornerFast(int, const int*, const double*, float*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16RatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16OrdersRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF16NarrowOrdersRatHornerFast(int, const int*, const void*, void*, std::size_t, void*);
// The launched bfloat16 half's launchers, one per symbol the class's wrappers reach: the
// same read as the declarations above, from the definitions in boys_cuda.cu.
int BoysCudaLaunchAllNBf16(int form, int nmax, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16Mono(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16Narrow(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowMono(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowOrders(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersMono(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRat(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowOrdersRatHornerFast(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowRat(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16NarrowRatHornerFast(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16Orders(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16OrdersMono(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16OrdersRat(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16OrdersRatHornerFast(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16Rat(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16RatHornerFast(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16Uniform(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16UniformHorner(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersBf16UniformRat(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchEachOrderBf16(int form, const int* n, const void* x, const int* offset, void* out, std::size_t count, void* stream);
int BoysCudaLaunchEachOrderBf16Fast(int form, const int* n, const void* x, const int* offset, void* out, std::size_t count, void* stream);
int BoysCudaLaunchSingleBf16(int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
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
int BoysCudaLaunchSingleF64Fast(int, const int*, const double*, double*, std::size_t, void*);
int BoysCudaLaunchAllOrdersF64Fast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64MonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersMonoFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64RatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64OrdersRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF64NarrowOrdersRatFast(
    int form, const int* n, const double* x, double* out, std::size_t count, void* stream);
int BoysCudaLaunchEachOrderF64(
    int form, const int* n, const double* x, const int* offset, double* out, std::size_t count,
    void* stream);
int BoysCudaLaunchEachOrderF64Fast(
    int form, const int* n, const double* x, const int* offset, double* out, std::size_t count,
    void* stream);
int BoysCudaLaunchEachOrderF32(
    int form, const int* n, const double* x, const int* offset, float* out, std::size_t count,
    void* stream);
int BoysCudaLaunchEachOrderF32Fast(
    int form, const int* n, const double* x, const int* offset, float* out, std::size_t count,
    void* stream);
int BoysCudaLaunchAllOrdersF32Mono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF32OrdersMono(
    int form, const int* n, const double* x, float* out, std::size_t count, void* stream);
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
int BoysCudaLaunchSingleF16Fast(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16Mono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchAllOrdersF16OrdersMono(
    int form, const int* n, const void* x, void* out, std::size_t count, void* stream);
int BoysCudaLaunchEachOrderF16(
    int form, const int* n, const void* x, const int* offset, void* out, std::size_t count,
    void* stream);
int BoysCudaLaunchEachOrderF16Fast(
    int form, const int* n, const void* x, const int* offset, void* out, std::size_t count,
    void* stream);
#endif
}

namespace boys {
namespace detail {

// code == 1 is the internal table-layout error; every other nonzero code is a
// cudaError_t from the launch or the sync. The status enum carries no payload -
// callers diagnose device failures through the CUDA runtime's own reporting.
inline BoysStatus FromLaunchCode(int code) {
    if (code == 0)
    {
        return BoysStatus::kSuccess;
    }

    return BoysStatus::kDeviceError;
}

// The uniform-order entries' one host-visible scalar: the order array of the
// per-element entries cannot be checked without a device copy, this can.
inline BoysStatus CheckOrder(int nmax) {
    if (nmax < 0 || nmax > kMaxBoysOrder)
    {
        return BoysStatus::kInvalidArgument;
    }

    return BoysStatus::kSuccess;
}

// The division form a caller may name. One kernel holds one division (the launch picks
// the instantiation; a kernel never branches on it), so a value outside the three has no
// kernel to select and is refused here, before anything is launched.
inline BoysStatus CheckDivisionForm(DivisionForm form) {
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
inline BoysStatus RunLaunch(Launcher launcher,
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

// The each-order shape's form of the same seam: one more argument than RunLaunch, because the
// ladders land at the caller's offsets rather than at planes the launcher derives. The checks,
// the empty-batch case and the launch-code reading are RunLaunch's, so a caller sees the same
// answers from either shape.
template <typename Launcher, typename Order, typename X, typename Value>
inline BoysStatus RunLaunchOffset(Launcher launcher,
                                  Order order,
                                  X x,
                                  const int* offset,
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

    return FromLaunchCode(launcher(static_cast<int>(form), order, x, offset, out, count, stream));
}

} // namespace detail
} // namespace boys
