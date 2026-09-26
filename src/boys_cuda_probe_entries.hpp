#pragma once

// The option space the device-cost probe measures, shared by its host
// translation unit (boys_cuda_probe.cpp) and its device one
// (boys_cuda_probe_kernels.cu). The rows are the library's own
// (BoysDeviceOptions, boys_cuda_options.hpp) and this file only names them, so
// the integer crossing the host/device boundary is never a literal on one side
// and a switch arm on the other, and the two translation units cannot disagree
// about which options exist.
//
// It includes one library header and it is CUDA-header-free, so the nvcc
// translation unit's include list (see the .cu preamble) can hold it.

#include "boys/boys_cuda_options.hpp"

namespace boys {
namespace probe_detail {

/// The device option space the probe measures, as the library reports it. The
/// rows are `boys::DeviceEntry`, the enumerators of BoysDeviceOptions(), so the
/// integer crossing the host/device boundary is never a literal on one side and
/// a switch arm on the other, and a row the library adds is a row this file
/// has to handle rather than a row it silently lacks.
using ProbeEntry = ::boys::DeviceEntry;

/// The question classes the report ranks inside, one per member of the
/// library's own DeviceOptionQuestion: an entry produces a definite amount of
/// output for a given workload, and two entries that produce different amounts
/// are not being asked the same question.
using ProbeQuestion = ::boys::DeviceOptionQuestion;

/// The two routes an option is reached by, the library's own grouping: the
/// library launched it, or the caller's kernel calls it and this probe times
/// the difference.
using ProbeGroup = ::boys::DeviceOptionGroup;

/// How an entry's figure was obtained. The two are not comparable as methods
/// and the report names which one produced each row.
enum class ProbeRoute : int {
    /// The library's own kernel, launched at the entry and bracketed by events.
    kLaunched = 0,
    /// The caller's kernel with the entry in it, minus the same kernel without
    /// it. No launch of this library's is inside either bracket.
    kInKernel,
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
