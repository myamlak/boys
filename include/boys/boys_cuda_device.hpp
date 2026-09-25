#pragma once

/// \file
/// Boys evaluation inside the caller's own kernel: F_0..F_n for an argument
/// the caller computed in a register, with no round trip through global
/// memory.
///
/// **What it is for.** A production GPU integral kernel forms x per thread —
/// one x per shell-quartet per primitive pair — and needs F_0..F_n for that
/// thread's own x, usually inside the same kernel that formed it. The batch
/// entries of this lane cannot serve that directly: they evaluate an array of
/// arguments, so a caller has to materialise every x to global memory, launch
/// a second kernel, synchronise and read the results back — two extra memory
/// passes per integral batch. The entries here take one (order, x) per call and
/// return the values to the calling thread, so the fused kernel needs neither
/// the pass out nor the pass back.
///
/// **How a caller obtains it, and what it costs the build.** The tables are
/// handed over at run time as a value, BoysDeviceTables, filled host-side by
/// BoysCuda::DeviceTables. The entries are therefore header-defined device
/// functions with no device-side symbol to link: including this header is the
/// whole of what the caller's build pays. No relocatable device code
/// (\c -rdc=true), no device link step, no library on the link line, no
/// additional compilation flag — a kernel that calls an entry compiles with the
/// same nvcc command line as one that does not, and the arithmetic is inlined
/// into the calling kernel, so there is no call overhead per order. What it
/// does cost is the compile of the coefficient tables this header pulls in
/// (boys_coefficients.hpp) and the instructions the compiler emits for the
/// arithmetic at each call site, which is the same arithmetic the batch kernels
/// run.
///
/// The alternative designs were weighed and rejected for stated reasons rather
/// than by omission. A device-linked library (relocatable device code) would
/// put the arithmetic in a separate object, so a fused kernel could not inline
/// it — the recursion loops would become device function calls — and it would
/// require \c -rdc=true plus a device link in every consumer's build, which is
/// a larger demand on the caller than the compile time this header costs. A
/// copy of the coefficient tables inside the header would duplicate roughly
/// 21 KB of tables per translation unit that includes it, and would make the
/// tables a compile-time property of the consumer rather than the device the
/// library uploaded them to.
///
/// **Precision.** The double entries are the primary surface: a fused integral
/// kernel is a double-precision computation, and the fp64 bound is what decides
/// whether it can use the lane at all. The float entries follow — on consumer
/// hardware fp32 runs many times faster, and the fused kernel is where that
/// matters most, since the arithmetic is done once per primitive pair — and the
/// fp16 entries mirror the fp16 batch lane for a caller packing many orders
/// into registers. All three read the same handle: there is one handle type and
/// one upload, not one per precision. The float single entry carries one option
/// the others do not: its region-B exponential is a certified choice of
/// arithmetic (boys::RegionBExp), which the caller names as a template
/// argument, exactly as the f32 batch single entry names it as an argument.
///
/// **Order range, and what the entries do when they cannot serve a request.**
/// Orders run 0..kMaxBoysOrder (32). Two shapes are offered per precision. The
/// runtime-top-order entry takes the order per call, which is what a fused
/// kernel needs (the order follows the shell quartets) and takes a capacity —
/// the number of values the caller's array holds — so that an order the array
/// cannot receive is reported rather than written past the end. The
/// compile-time-top-order entry takes the order as a template argument, where
/// it bounds the recursion loops and the compiler can unroll them.
///
/// Every failure is a BoysDeviceStatus the caller branches on, and a call that
/// fails writes nothing: no truncated ladder, no partial fill. An order outside
/// the range is \c kOrderOutOfRange; an order whose values do not fit the
/// caller's array is \c kCapacityTooSmall; a handle that carries no tables is
/// \c kTablesNotReady. None of the three is silent, and none of them returns a
/// value in place of the ones it could not compute.
///
/// **Cost in registers.** A caller keeping the whole ladder live holds order + 1
/// doubles — 33 of them at the top order, which is the real cost of this
/// interface and the reason the entries write into the caller's array rather
/// than into an internal one: the caller's own liveness decides what stays in
/// registers, and a caller that consumes each order as it arrives can hold
/// none. BoysDeviceEachOrderF64 and its siblings are that shape: they hand one
/// value at a time to a caller-supplied sink and keep no ladder at all.
///
/// **The handle as a kernel argument.** Pass it by value into the kernel — it
/// is plain data, and kernel parameters live in the constant bank — and declare
/// that parameter \c __grid_constant__ so that taking its address inside the
/// kernel does not copy it to local memory per thread:
/// \code
/// __global__ void Fused(__grid_constant__ const boys::BoysDeviceTables tables,
///                       const double* rho, const double* d2, ...)
/// \endcode
/// A caller on a toolchain without \c __grid_constant__ passes the same value
/// without the qualifier and reads the handle through a generic pointer, which
/// is correct and slower; the arithmetic is identical either way.
///
/// **The accuracy multiplier is not read here.** These entries are the
/// full-accuracy arithmetic, m = 1 (the bit-identical path). Relaxation is a
/// property of the batch entries' instantiations, uploaded per (device, m) into
/// the lane's constant tables; the handle carries the full-accuracy tables and
/// always did. Lifting that is a work item with a known shape — the handle
/// would carry the relaxed degree tables as well, and the entries would take
/// the multiplier's lane as a parameter — and it is called out here rather than
/// left for a caller to discover, because a kernel that fused the batch entry's
/// bound with these entries' values would be reading a bound the device
/// arithmetic is not holding.
///
/// **The bound.** Every entry holds the lane's documented bound for its
/// precision at m = 1 — the same bound the corresponding batch entry documents,
/// because it is the same arithmetic: the device gate measures these entries
/// against the committed 45-digit reference grid and reports the worst ratio in
/// the same vocabulary as every other lane.
///
/// \ingroup boys

#include "boys/boys_cuda_arithmetic.hpp"
#include "boys/boys_device_tables.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace boys {

/// Result status of the device-callable entries.
///
/// Device code cannot throw and cannot report an error out of band, so every
/// refusal is one of these values, returned to the calling thread. A call that
/// does not return \c kSuccess wrote nothing: the caller's array is untouched
/// and no value is in flight. \c kSuccess is 0.
enum class BoysDeviceStatus : int {
    kSuccess = 0, ///< the values were written
    /// The handle carries no tables: BoysCuda::DeviceTables was never called
    /// for this device, or it failed and the caller ignored the status.
    kTablesNotReady,
    /// The order is outside 0..kMaxBoysOrder. Nothing was written.
    kOrderOutOfRange,
    /// The caller's array holds fewer than order + 1 values. Nothing was
    /// written.
    kCapacityTooSmall,
};

/// \cond
namespace detail {

// The handle as a lane object — the same seven members the batch kernels' lane
// objects carry (boys_cuda_arithmetic.hpp), reading the caller's handle
// instead of a __constant__ symbol. The piece index of order n, piece p is
// pieceStart[n] + p, so no stride constant has to agree between this header
// and the tables the library uploaded.
struct TableLane64 {
    const BoysDeviceTables* tables;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart[order + 1] - tables->pieceStart[order];
    }

    __device__ __forceinline__ double A(int order, int piece) const {
        return tables->pieceA[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ double B(int order, int piece) const {
        return tables->pieceB[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* Coeffs(int order, int piece) const {
        return tables->coeffs + tables->pieceOffset[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return tables->pieceDeg[tables->pieceStart[order] + piece];
    }

    __device__ __forceinline__ const double* BSeedCoeffs() const {
        return tables->bSeedCoeffs;
    }

    __device__ __forceinline__ int BSeedDeg(int) const {
        return tables->bSeedDeg;
    }
};

struct TableLane32 {
    const BoysDeviceTables* tables;

    __device__ __forceinline__ int Count(int order) const {
        return tables->pieceStart32[order + 1] - tables->pieceStart32[order];
    }

    __device__ __forceinline__ float A(int order, int piece) const {
        return tables->pieceA32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ float B(int order, int piece) const {
        return tables->pieceB32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* Coeffs(int order, int piece) const {
        return tables->coeffs32 + tables->pieceOffset32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ int Deg(int order, int piece) const {
        return tables->pieceDeg32[tables->pieceStart32[order] + piece];
    }

    __device__ __forceinline__ const float* BSeedCoeffs() const {
        return tables->bSeedCoeffs32;
    }

    __device__ __forceinline__ int BSeedDeg(int) const {
        return tables->bSeedDeg32;
    }
};

// The two refusals every entry shares. The table test is one pointer, and it
// is the difference between a reported status and a null dereference; the order
// test is what makes an out-of-range request a value rather than a read outside
// the piece tables.
__device__ __forceinline__ BoysDeviceStatus DeviceReady(const BoysDeviceTables& tables) {
    return tables.pieceStart == nullptr ? BoysDeviceStatus::kTablesNotReady
                                        : BoysDeviceStatus::kSuccess;
}

__device__ __forceinline__ bool DeviceOrderValid(int order) {
    return order >= 0 && order <= kMaxBoysOrder;
}

} // namespace detail
/// \endcond

/// F_n(x) in double precision, inside the caller's kernel.
///
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the order n, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_n(x); untouched unless kSuccess is returned
///
/// \pre \c x >= 0, as for the batch entries. A negative argument names a
///      different function than the one the tables fit, and is not checked: the
///      arithmetic's domain is the caller's to keep, exactly as the batch
///      entries' is.
/// \pre \c out is a device-writable location for one double.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady when \c tables
/// carries no tables; kOrderOutOfRange when \c order is outside
/// 0..kMaxBoysOrder. A refused call writes nothing.
__device__ BoysDeviceStatus BoysDeviceSingleF64(
    const BoysDeviceTables& tables, int order, double x, double* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    *out = detail::DeviceSingleF64(detail::TableLane64{&tables}, order, x);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, inside the caller's kernel.
///
/// \param tables   the handle BoysCuda::DeviceTables filled
/// \param order    the top order, 0..kMaxBoysOrder
/// \param x        the argument, >= 0, formed by the calling thread
/// \param out      receives F_0(x)..F_order(x), order + 1 consecutive doubles
/// \param capacity the number of doubles \c out holds
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c order + 1 <= \c capacity; when it is not, the call is refused with
///      kCapacityTooSmall and writes nothing.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF64(
    const BoysDeviceTables& tables, int order, double x, double* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables}, order, x, [&](int l, double v) {
        out[l] = v;
    });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in double precision, with the top order fixed where
/// the call site names it.
///
/// The order is a template argument, so the recursion bounds are compile-time
/// constants: the loops can be unrolled and no order has to be validated. For a
/// caller whose fused kernel works at one order — or at a handful it
/// instantiates separately — this is the entry to use, and it is also the
/// shape a caller can wrap in its own dispatch where the order is a run-time
/// value.
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///               doubles
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady when
/// \c tables carries no tables. There is no order or capacity check: the top
/// order is compiled in and \c out is the caller's declaration.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF64(
    const BoysDeviceTables& tables, double x, double* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables}, kTopOrder, x, [&](int l, double v) {
        out[l] = v;
    });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in double precision, one value at a time into a caller's
/// sink.
///
/// The shape to use when the caller consumes each order as it arrives — an
/// integral kernel contracting F_l with a coefficient does exactly that — and
/// therefore has no reason to hold the ladder: nothing is stored inside this
/// call and the register cost is the caller's sink's own. The sink is called
/// once per order, in the order the recursion produces it, which is the top
/// order downwards inside region A and 0 upwards in regions B and C: a sink
/// that cares about the sequence of calls can see region A's descending
/// arrival, and a sink that contracts into an accumulator cannot.
///
/// \tparam Sink a callable taking (int order, double value), callable from
///         device code. Passing it by value is deliberate: a lambda capturing
///         the caller's accumulators stays in registers.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the top order, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param sink   receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x >= 0, as BoysDeviceSingleF64 states.
/// \pre \c sink is device-callable and its own work does not fault.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, in which case \c sink is not
/// called at all.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF64(
    const BoysDeviceTables& tables, int order, double x, Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::DeviceAllOrdersF64(detail::TableLane64{&tables}, order, x, sink);
    return BoysDeviceStatus::kSuccess;
}

/// F_n(x) in single precision, inside the caller's kernel.
///
/// The float lane's bound is the one the f32 batch entries document; this
/// entry is the same arithmetic at m = 1, and it takes the lane's own choice of
/// region-B exponential.
///
/// \tparam kExp  the region-B exponential to evaluate. RegionBExp::kAccurate is
///         the default and the tighter bound of the two; a caller that selects
///         RegionBExp::kFast takes that option's bound, which is the lane's
///         plus the corrected seed's own contribution. The choice is a template
///         argument and not a run-time one because it selects an arithmetic
///         inside the caller's own kernel rather than branching within one: the
///         exponential is evaluated once per call, outside the order loop, so
///         the option not selected is absent from the caller's kernel instead
///         of merely untaken. Both options of a given template argument are one
///         binary; a caller that needs both instantiates both.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the order n, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_n(x); untouched unless kSuccess is returned
///
/// \pre \c x >= 0; \c out is a device-writable location for one float.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady or
/// kOrderOutOfRange otherwise, without writing.
template <RegionBExp kExp = RegionBExp::kAccurate>
__device__ BoysDeviceStatus BoysDeviceSingleF32(
    const BoysDeviceTables& tables, int order, float x, float* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    *out = detail::DeviceSingleF32<kExp == RegionBExp::kFast>(detail::TableLane32{&tables},
                                                              order,
                                                              x);
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in single precision, inside the caller's kernel.
///
/// The region-A seed is computed in double precision, as the f32 batch entry
/// documents: the downward recursion amplifies a float seed error beyond the
/// lane's float budget.
///
/// \param tables   the handle BoysCuda::DeviceTables filled
/// \param order    the top order, 0..kMaxBoysOrder
/// \param x        the argument, >= 0, formed by the calling thread
/// \param out      receives F_0(x)..F_order(x), order + 1 consecutive floats
/// \param capacity the number of floats \c out holds
///
/// \pre \c x >= 0; \c order + 1 <= \c capacity, or the call is refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF32(
    const BoysDeviceTables& tables, int order, float x, float* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               order,
                               x,
                               [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in single precision, with the top order fixed where
/// the call site names it (see BoysDeviceAllNF64).
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///               floats
///
/// \pre \c x >= 0.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady
/// otherwise.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF32(
    const BoysDeviceTables& tables, float x, float* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               kTopOrder,
                               x,
                               [&](int l, float v) { out[l] = v; });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in single precision, one value at a time into a caller's
/// sink (see BoysDeviceEachOrderF64 for the arrival order and the reason the
/// shape exists).
///
/// \tparam Sink a callable taking (int order, float value), callable from
///         device code.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the top order, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param sink   receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x >= 0; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, with \c sink not called.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF32(
    const BoysDeviceTables& tables, int order, float x, Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               order,
                               x,
                               sink);
    return BoysDeviceStatus::kSuccess;
}

#if BoysFp16
/// F_n(x) in fp16, inside the caller's kernel.
///
/// The fp16 lane rounds at the boundary and runs the fp32 engine in between, so
/// this entry takes and returns the raw IEEE-754 binary16 bit patterns (the
/// __half type), as the fp16 batch entries take and return this library's F16.
/// The bound is the fp16 lane's.
///
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the order n, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_n(x) in fp16; untouched unless kSuccess is returned
///
/// \pre \c x is a binary16 value, hence >= 0 and finite; \c out is a
///      device-writable location for one __half.
///
/// \returns kSuccess after writing F_n(x); kTablesNotReady or
/// kOrderOutOfRange otherwise, without writing.
__device__ BoysDeviceStatus BoysDeviceSingleF16(
    const BoysDeviceTables& tables, int order, __half x, __half* out) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    *out = __float2half(
        detail::DeviceSingleF32<false>(detail::TableLane32{&tables}, order, __half2float(x)));
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, inside the caller's kernel.
///
/// \param tables   the handle BoysCuda::DeviceTables filled
/// \param order    the top order, 0..kMaxBoysOrder
/// \param x        the argument, >= 0, formed by the calling thread
/// \param out      receives F_0(x)..F_order(x), order + 1 consecutive __half
/// \param capacity the number of __half values \c out holds
///
/// \pre \c x is a binary16 value; \c order + 1 <= \c capacity, or the call is
///      refused.
///
/// \returns kSuccess after writing order + 1 values; kTablesNotReady,
/// kOrderOutOfRange or kCapacityTooSmall otherwise, in every case without
/// writing.
__device__ BoysDeviceStatus BoysDeviceAllOrdersF16(
    const BoysDeviceTables& tables, int order, __half x, __half* out, int capacity) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    if (capacity < order + 1)
    {
        return BoysDeviceStatus::kCapacityTooSmall;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               order,
                               __half2float(x),
                               [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_kTopOrder(x) in fp16, with the top order fixed where the call site
/// names it (see BoysDeviceAllNF64).
///
/// \tparam kTopOrder the top order, 0..kMaxBoysOrder.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param x      the argument, >= 0, formed by the calling thread
/// \param out    receives F_0(x)..F_kTopOrder(x), kTopOrder + 1 consecutive
///               __half values
///
/// \pre \c x is a binary16 value.
///
/// \returns kSuccess after writing kTopOrder + 1 values; kTablesNotReady
/// otherwise.
template <int kTopOrder>
__device__ BoysDeviceStatus BoysDeviceAllNF16(
    const BoysDeviceTables& tables, __half x, __half* out) {
    static_assert(kTopOrder >= 0 && kTopOrder <= kMaxBoysOrder,
                  "the top order must be inside 0..kMaxBoysOrder");

    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               kTopOrder,
                               __half2float(x),
                               [&](int l, float v) { out[l] = __float2half(v); });
    return BoysDeviceStatus::kSuccess;
}

/// F_0(x)..F_order(x) in fp16, one value at a time into a caller's sink (see
/// BoysDeviceEachOrderF64 for the arrival order and the reason the shape
/// exists).
///
/// \tparam Sink a callable taking (int order, __half value), callable from
///         device code.
/// \param tables the handle BoysCuda::DeviceTables filled
/// \param order  the top order, 0..kMaxBoysOrder
/// \param x      the argument, >= 0, formed by the calling thread
/// \param sink   receives (l, F_l(x)) for every l in 0..order
///
/// \pre \c x is a binary16 value; \c sink is device-callable.
///
/// \returns kSuccess once every order has been handed to \c sink;
/// kTablesNotReady or kOrderOutOfRange otherwise, with \c sink not called.
template <typename Sink>
__device__ BoysDeviceStatus BoysDeviceEachOrderF16(
    const BoysDeviceTables& tables, int order, __half x, Sink sink) {
    const BoysDeviceStatus ready = detail::DeviceReady(tables);

    if (ready != BoysDeviceStatus::kSuccess)
    {
        return ready;
    }

    if (!detail::DeviceOrderValid(order))
    {
        return BoysDeviceStatus::kOrderOutOfRange;
    }

    detail::DeviceAllOrdersF32(detail::TableLane64{&tables},
                               detail::TableLane32{&tables},
                               order,
                               __half2float(x),
                               [&](int l, float v) { sink(l, __float2half(v)); });
    return BoysDeviceStatus::kSuccess;
}
#endif // BoysFp16

} // namespace boys
