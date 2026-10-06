// The device-callable arm of the accuracy gate's fast-member reading: an in-kernel entry is a
// __device__ function only the caller's own kernel reaches, so this arm is a translation unit of
// its own and not lines of tests/boys_accuracy_gate.cpp, whose extension the CUDA compiler does
// not take.

// The shape is tests/boys_cuda_device_demo.cu's: a kernel that calls the entry, the caller filling
// the handle, a kernel translation unit being unable to include the host header.

// The member read is RegionBExp::kFast, the one the gate's launched arms do not read: they launch
// the all-orders family, whose rows the option table carries at RegionBExp::kAccurate. The entry
// is BoysDeviceSingleF32 at that member, judged at the figure BoysAccuracyGuaranteed composes for
// it (src/boys.cpp), the lane's base plus the term its row publishes; the gate judges.

// The division form is a template argument of a device entry, since it selects an arithmetic
// inside the caller's kernel rather than a step of the launch, so the gate's runtime form is
// dispatched here onto the three instantiations the axis carries; one outside them is refused
// rather than run at another.

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
