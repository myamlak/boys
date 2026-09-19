#pragma once

#include "boys/boys.hpp"

#include <cstddef>

#if BoysFp16
#include "boys/f16.hpp"
#endif

/// \file
/// CUDA lane of the Boys kernel.
///
/// Kept free of CUDA runtime headers so plain C++ translation units can
/// include it; the implementation lives in boys_cuda.cu. Streams are
/// opaque `void*` here and `cudaStream_t` inside the .cu file. All entry
/// points take raw device pointers — the caller owns the device memory
/// (cudaMalloc/cudaMemcpy/cudaFree) and the stream.

namespace boys {

/// Result status of the CUDA lane entry points.
///
/// The CUDA lane is the one fallible surface of this library: table
/// uploads, parameter validation, and launches report through this status
/// (never exceptions). \c kSuccess is 0. The enum carries no payload —
/// when \c kDeviceError is returned the caller can use the CUDA runtime's
/// own error reporting (cudaGetLastError, stream capture) for the detail.
enum class BoysStatus {
    kSuccess = 0, ///< the call succeeded
    kInvalidArgument, ///< a parameter was invalid (see the entry's contract)
    kDeviceError, ///< a CUDA operation failed
};

/// Device-side Boys evaluation over arrays of (n, x) inputs.
///
/// The coefficient tables are uploaded on first use by any entry point
/// (idempotent per device; switching devices re-uploads automatically).
/// The relaxed-multiplier degree tables are uploaded once per (device, m)
/// on the first call at that m — InitializeTables and the entries are
/// safe to call from multiple host threads.
///
/// All entries are asynchronous: the kernel is queued on the caller's
/// stream and the call returns once the launch is accepted (errors are
/// reported by the return status). Synchronize the stream (or use
/// cudaStreamSynchronize on a per-call stream) before reading the outputs.
///
/// Count-0 contract asymmetry (documented, not changed — changing it
/// would be a MAJOR-version error-behavior break): the fp16/bf16 family
/// validates count == 0 and returns kInvalidArgument, while the fp32/fp64
/// families accept count == 0 as a no-op zero-thread launch (no kernel
/// queued, the output untouched, kSuccess).
///
/// \ingroup boys
class BoysCuda {
public:
    /// Uploads the Chebyshev coefficient tables (double and float lanes) to
    /// the current CUDA device. Idempotent; synchronous with respect to the
    /// host.
    ///
    /// \returns kDeviceError when a device operation fails.
    static BoysStatus InitializeTables();

    /// F_n(x[i]) in single precision — the recommended GPU lane on
    /// consumer hardware (double precision runs there at a fraction of
    /// single-precision throughput).
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 1.5e-7 (region B/C) via compile-time Chebyshev degree
    ///   truncation. Monotone in m.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus SingleF32(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in single precision per input (i).
    ///
    /// Output layout: out[order * count + i] = F_order(x[i]), order 0..nmax[i].
    /// The region-A seed is computed in double precision on the device — the
    /// downward recursion amplifies float seed errors beyond the certified
    /// 1.5e-7 float budget (same reasoning as the CPU float batch).
    ///
    /// \tparam kAccuracyMultiplier as SingleF32; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry (the F0
    ///   seed's error reaches every output with amplification <= 1).
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) floats
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus BatchF32(
        const int* n, const double* x, float* out, std::size_t count, void* stream);

    /// F_n(x[i]) in double precision, |error| <= 5e-14.
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 5.5e-14 via compile-time Chebyshev degree truncation.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array receiving F_n(x[i])
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus SingleF64(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in double precision per input (i), layout as
    /// BatchF32.
    ///
    /// \tparam kAccuracyMultiplier as SingleF64; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) doubles
    /// \param count  number of elements
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kDeviceError when the launch fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus BatchF64(
        const int* n, const double* x, double* out, std::size_t count, void* stream);

#if BoysFp16
    /// F_n(x[i]) in fp16 — the fp16 lane of the certified mixed-precision
    /// boundary (behind the BoysFp16 seam). Device pointers and stream
    /// contract as the F32/F64 entries; the fp16 values are the raw
    /// IEEE-754 binary16 bit patterns of this library's F16 type (see
    /// f16.hpp), so host copies are plain byte copies of count * sizeof(F16).
    ///
    /// \tparam kAccuracyMultiplier the accuracy multiplier: m = 1 is the
    ///   bit-identical full-accuracy path; m > 1 relaxes the asserted bound
    ///   to m * 1e-7 + 1/2 ULP via compile-time Chebyshev degree truncation.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array receiving F_n(x[i]) in fp16
    /// \param count  number of elements (must be > 0)
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kInvalidArgument when count == 0, kDeviceError when a device
    /// operation fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus SingleF16(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);

    /// F_0(x[i])..F_nmax(x[i]) in fp16 per input (i), layout as BatchF32
    /// (out[order * count + i] = F_order(x[i])), device pointers and
    /// stream contract as SingleF16.
    ///
    /// \tparam kAccuracyMultiplier as SingleF16; the batch relaxation covers
    ///   the whole output family via the order-0 region-B entry.
    /// \param n      device array of orders, 0..kMaxBoysOrder
    /// \param x      device array of fp16 arguments, >= 0
    /// \param out    device array, at least count * (kMaxBoysOrder + 1) fp16 values
    /// \param count  number of elements (must be > 0)
    /// \param stream device stream (cudaStream_t) or nullptr for the default
    ///
    /// \returns kInvalidArgument when count == 0, kDeviceError when a device
    /// operation fails.
    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>
    static BoysStatus BatchF16(
        const int* n, const F16* x, F16* out, std::size_t count, void* stream);
#endif // BoysFp16
};

} // namespace boys
