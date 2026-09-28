// The device half of the consumer check on the CUDA lane's run-time rung.
//
// The host half (tests/consumer_cuda_rungs_host.cpp) drives the launched
// entries whose rung is a call argument. This half is the other surface of the
// same lane, and it is here for one property: the device-callable entry is the
// entry that READS the resident rung, so it is the entry that can refuse one.
// A kernel of this file calls BoysDeviceSingleF64 with a multiplier its caller
// names, at the top order the host asks for, and hands back what the entry
// answered — the value, or the status it refused with. The host half runs it
// against a resident rung and against a rung that is not resident, and prints
// what came back for each: that is the rule the launched entries must not
// soften, checked on the card rather than argued about in a comment.
//
// Nothing here touches the library's implementation. It includes the public
// device header and the CUDA runtime, so it is what a consumer's translation
// unit would be.

#include "boys/boys_cuda_device.hpp"

#include <cuda_runtime.h>
#include <stddef.h>

namespace {

unsigned int Blocks(std::size_t count) {
    return static_cast<unsigned int>((count + 255) / 256);
}

// One thread per element, one value per element: F_{n[i]}(x[i]) at the
// multiplier the launch names. The status is recorded per element so that a
// refused call is a value the host can read rather than a kernel that stopped.
// A refused call writes nothing, so the output slot is the sentinel the host
// filled it with, and the host reads both.
__global__ void SingleRungKernel(__grid_constant__ const boys::BoysDeviceTables tables,
                                 const int* n,
                                 const double* x,
                                 double* out,
                                 std::size_t count,
                                 double multiplier,
                                 int* statuses) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (i >= count)
    {
        return;
    }

    const boys::BoysDeviceStatus status =
        boys::BoysDeviceSingleF64(tables, n[i], x[i], out + i, multiplier);
    statuses[i] = static_cast<int>(status);
}

} // namespace

// F_n(x[i]) in double precision inside the caller's own kernel, at the
// multiplier \c multiplier, for count elements.
//
// \param tables     the handle the host filled with BoysCuda::DeviceTables
// \param n          device array of orders
// \param x          device array of arguments
// \param out        device array receiving one double per element; an element
//                   whose call was refused keeps what it held
// \param count      number of elements; 0 launches nothing
// \param multiplier the accuracy multiplier every element's call names
// \param statuses   device array receiving the entry's status per element
// \returns 0 when the launch was accepted, the CUDA error otherwise
extern "C" int BoysConsumerCudaRungSingle(const boys::BoysDeviceTables* tables,
                                          const int* n,
                                          const double* x,
                                          double* out,
                                          std::size_t count,
                                          double multiplier,
                                          int* statuses) {
    if (count == 0)
    {
        return 0;
    }

    SingleRungKernel<<<Blocks(count), 256>>>(*tables, n, x, out, count, multiplier, statuses);
    return static_cast<int>(cudaGetLastError());
}

// The two statuses the host half reads, named from this side because
// boys_cuda_device.hpp is a CUDA header: the host half cannot include it, and a
// second copy of the enumerator's value written there would be a second answer
// to what the entry returns.
extern "C" int BoysConsumerCudaRungSuccess(void) {
    return static_cast<int>(boys::BoysDeviceStatus::kSuccess);
}

// The refusal a call naming a rung that is not resident gets.
extern "C" int BoysConsumerCudaRungNotResident(void) {
    return static_cast<int>(boys::BoysDeviceStatus::kMultiplierNotResident);
}
