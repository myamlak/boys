#pragma once

// The option table the device-cost probe measures, shared by its host
// translation unit (boys_cuda_probe.cpp) and its device one
// (boys_cuda_probe_kernels.cu). Both sides name the same enumerators, so the
// integer crossing the boundary is never a literal on one side and a switch
// arm on the other.
//
// Header only, no CUDA runtime and no library header: the nvcc translation
// unit gets a CUDA-safe include list (see the .cu preamble) and this file has
// to be on it.

#include <cstddef>

namespace boys {
namespace probe_detail {

/// One measurable entry of the CUDA lane.
///
/// The first group is launched by this library — a consumer call reaches them
/// through boys_cuda.hpp. The second group is the device-callable entries of
/// boys_cuda_device.hpp, which are designed to run inside the caller's own
/// kernel: those are timed by subtraction against the same kernel with the
/// call removed, which is why the table has to say which group an entry is in.
///
/// The enumerator order is the report's row order within a question class: the
/// launched entry of a shape, then the device-callable one.
enum class ProbeEntry : int {
    kSingleF64 = 0,
    kSingleF32,
    kSingleF32Fast,
    kSingleF16,

    kAllOrdersF64,
    kAllOrdersF32,
    kAllOrdersF16,

    kAllNF64,
    kAllNF32,
    kAllNF16,

    kDeviceSingleF64,
    kDeviceSingleF32,
    kDeviceSingleF16,

    kDeviceAllOrdersF64,
    kDeviceAllOrdersF32,
    kDeviceAllOrdersF16,

    kDeviceAllNF64,
    kDeviceAllNF32,
    kDeviceAllNF16,

    kDeviceEachOrderF64,
    kDeviceEachOrderF32,
    kDeviceEachOrderF16,

    kCount,
};

/// How an entry's figure was obtained. The two are not comparable as methods
/// and the report names which one produced each row.
enum class ProbeRoute : int {
    /// The library's own kernel, launched at the entry and bracketed by events.
    kLaunched = 0,
    /// The caller's kernel with the entry in it, minus the same kernel without
    /// it. No launch of this library's is inside either bracket.
    kInKernel,
};

/// The report's question classes: an entry produces a definite amount of output
/// for a given workload, and two entries that produce different amounts are not
/// being asked the same question. The probe orders entries only within a class.
enum class ProbeQuestion : int {
    /// F_n(x) for each argument, at that argument's own order.
    kSingle = 0,
    /// F_0(x)..F_n(x) for each argument, at that argument's own order.
    kAllOrders,
    /// F_0(x)..F_nmax(x) for every argument of the batch, one common top order.
    kAllN,
    kCount,
};

/// The device-callable entries are instantiated at one compile-time top order
/// (BoysDeviceAllNF64's order is a template argument). This is that order, and
/// it is what the in-kernel all-n rows are measured at.
inline constexpr int kProbeInKernelTopOrder = 32;

/// The canary's work: rounds of a 64-bit xorshift per thread of a fixed grid.
/// An integer chain deliberately, as the host probe's canary is — no
/// floating-point state, so the arithmetic this probe ranks cannot change the
/// cost of the instrument that judges the run.
inline constexpr int kProbeCanaryBlocks = 64;
inline constexpr int kProbeCanaryThreads = 256;
inline constexpr int kProbeCanaryRounds = 1 << 14;

/// What the device-side query fills. Plain data, so the boundary between the
/// nvcc translation unit and the host one stays a struct and not a CUDA type.
/// The name is `cudaDeviceProp::name`'s own width.
struct ProbeDeviceFacts {
    char name[256];
    int computeMajor;
    int computeMinor;
    unsigned long long totalMemoryBytes;
    int multiProcessorCount;
    int clockKHz;
    int memoryClockKHz;
    int singleToDoublePrecisionPerfRatio;
    int driverVersion;
    int runtimeVersion;
};

/// The toolkit and architecture strings the report states. A device figure
/// whose toolkit and target architecture are unstated cannot be reproduced.
struct ProbeBuildFacts {
    char toolkit[64];
    char architectures[128];
};

/// What one timed region should run. One struct rather than a dozen positional
/// arguments across the boundary, so that the host side names what it is timing
/// at the call site instead of counting parameters.
struct ProbeTimeRequest {
    /// Which region: see the ProbeWhat enumerators.
    int what;

    /// The entry, read when \c what is kLaunchedEntry or kInKernelEntry.
    int entry;

    /// Launches inside the region. The elapsed device time is divided by this
    /// by the caller.
    int reps;

    /// For kInKernelEntry: 1 times the caller's kernel with the entry in it,
    /// 0 times the same kernel with the call removed.
    int withBoys;

    /// The all-n entries' common top order, for the launched ones.
    int nmax;

    /// The handle the device-callable entries read, for kInKernelEntry.
    const void* handle;

    /// The order array, for every region but the canary.
    const int* n;

    /// The arguments the double lane and every launched entry take.
    const double* x;

    /// The arguments the in-kernel fp32 entries take.
    const float* xf;

    /// The arguments the fp16 lane takes, launched and in-kernel alike.
    const void* xh;

    /// The output block, wide enough for the tallest ladder of any entry.
    void* out;

    /// The canary's sink, for kCanary.
    unsigned long long* canarySink;

    /// One int the floor kernel writes, so that a kernel doing no work is not
    /// removed as having no effect. A slot of its own, because the floor runs
    /// after the workload has been read back and must not be able to disturb a
    /// figure that has already been taken.
    int* floorSink;

    /// Arguments in the batch.
    unsigned long long count;

    /// Receives the region's device milliseconds.
    double* outMs;
};

/// The regions ProbeTimeRequest::what selects.
enum class ProbeWhat : int {
    kCanary = 0,      ///< the fixed-work instrument
    kFloor,           ///< a kernel launched the same way that does no arithmetic
    kLaunchedEntry,   ///< a kernel of this library, launched at the entry
    kInKernelEntry,   ///< one half of the subtraction kernel
};

} // namespace probe_detail
} // namespace boys
