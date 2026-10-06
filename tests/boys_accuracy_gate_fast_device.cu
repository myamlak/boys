// The device-callable arm of the accuracy gate's fast-member reading.
//
// The gate measures a device lane through the CUDA surface's launched entries, which
// are host calls that carry their own launch. The surface's other group is not a call
// at all: an in-kernel entry is a __device__ function the caller's own kernel calls,
// and it can only be reached from a kernel of the caller's - which is why this arm is
// a translation unit of its own and not a few lines of tests/boys_accuracy_gate.cpp,
// whose extension the CUDA compiler does not take. The shape is
// tests/boys_cuda_device_demo.cu's: a kernel that forms x per thread, calls the entry
// from inside itself, and a host launcher the caller hands its own handle and arrays
// to - a kernel translation unit may include the device header and not the host one,
// so the handle is filled by the caller, which is a host translation unit.
//
// The member this arm reads is RegionBExp::kFast, the member the gate's launched arms
// do not read: they launch the all-orders family, whose rows the option table carries
// at RegionBExp::kAccurate. The entry is the lane's own single order at that member,
// BoysDeviceSingleF32 at RegionBExp::kFast, and the figure it is judged at is the one
// BoysAccuracyGuaranteed composes for the member (src/boys.cpp): the lane's base plus
// the term its row publishes under that member. The gate does the judging; what this
// file returns is the card's values.
//
// The division form is a template argument of a device entry, because it selects an
// arithmetic inside the caller's kernel rather than a step of the launch, so the
// runtime form the gate names is dispatched here onto the three instantiations the
// axis carries. A form outside them is refused rather than run at another.

#include <cuda_runtime.h>

#include <boys/boys_cuda_device.hpp>

#include <cstddef>

namespace {

/// One value per element. The argument is read as a double and narrowed per thread,
/// which is the argument list and the narrowing the launched float entries are read
/// through (src/boys_cuda.cu, the float launchers), so this arm's cells are the same
/// grid the lane's other arms measure rather than a second rounding of it.
template <boys::DivisionForm kForm>
__global__ void GateFastSingleF32Kernel(__grid_constant__ const boys::BoysDeviceTables tables,
                                        const int* n,
                                        const double* x,
                                        float* out,
                                        std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) *
                              static_cast<std::size_t>(blockDim.x) +
                          static_cast<std::size_t>(threadIdx.x);

    if (i >= count) {
        return;
    }

    float value = 0.0f;

    if (boys::BoysDeviceSingleF32<kForm, boys::RegionBExp::kFast>(
            tables, n[i], static_cast<float>(x[i]), &value) ==
        boys::BoysDeviceStatus::kSuccess) {
        out[i] = value;
    }
}

inline unsigned GateBlocks(std::size_t count) {
    return static_cast<unsigned>((count + 255u) / 256u);
}

} // namespace

/// F_n(x[i]) in float at RegionBExp::kFast, one value per element, through the entry
/// the caller's own kernel calls. The argument list is the launched entries' own,
/// beside the handle, so that the gate's sweep reads this arm through one call shape.
///
/// \param tables the handle the caller filled with BoysCuda::DeviceTables
/// \param n      device array of orders, 0..kMaxBoysOrder
/// \param x      device array of arguments, >= 0, narrowed to float per thread
/// \param out    device array receiving F_n(x[i]), count floats
/// \param count  number of elements
/// \param stream the stream the caller named; this arm takes none and refuses others
/// \param form   which division form the call runs: one of the three the axis carries
///
/// \returns 0 for a launch that ran, non-zero for one this arm's contract refuses - a
///          stream it does not take, or a form outside the axis - or for a launch the
///          runtime reported as failed
extern "C" int BoysGateFastDeviceSingleF32(const boys::BoysDeviceTables* tables,
                                           const int* n,
                                           const double* x,
                                           float* out,
                                           std::size_t count,
                                           void* stream,
                                           boys::DivisionForm form) {
    if (tables == nullptr || stream != nullptr) {
        return 1;
    }

    switch (form) {
    case boys::DivisionForm::kExactDivision:
        GateFastSingleF32Kernel<boys::DivisionForm::kExactDivision>
            <<<GateBlocks(count), 256>>>(*tables, n, x, out, count);
        break;
    case boys::DivisionForm::kPlainReciprocal:
        GateFastSingleF32Kernel<boys::DivisionForm::kPlainReciprocal>
            <<<GateBlocks(count), 256>>>(*tables, n, x, out, count);
        break;
    case boys::DivisionForm::kRefinedReciprocal:
        GateFastSingleF32Kernel<boys::DivisionForm::kRefinedReciprocal>
            <<<GateBlocks(count), 256>>>(*tables, n, x, out, count);
        break;
    default:
        return 2;
    }

    return static_cast<int>(cudaGetLastError());
}
