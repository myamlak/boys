#pragma once

/// \file
/// Which multiply-add the CUDA lane's arithmetic is compiled with: the one
/// build fact both sides of the device boundary read.

/// \cond

#include "boys/backend.hpp"

namespace boys::detail {

/// The multiply-add route this build's device arithmetic runs.
///
/// **One selection, two lanes.** `BOYS_MULADD_SEPARATE` reaches the host
/// translation units and the nvcc ones alike: it is a PUBLIC compile definition
/// of the library target, so the kernels' translation unit and a consumer's own
/// translation unit inherit the same value the host lane reads, and the device
/// arithmetic cannot be built at a route the host build did not select.
///
/// **The device's contract is the stronger one.** The host's separate route is
/// written bare (backend.hpp, `Separate`), which a contracting build compiles to
/// the fused step, so the host *measures* what its arithmetic delivered and
/// reports the measurement rather than the selection (backend.hpp,
/// `RouteInForce`). The device cannot lean on that reading, because nvcc
/// contracts a bare product-plus-add by default: a bare spelling there is the
/// fused route under a different name. The device therefore spells both routes
/// out (boys_cuda_arithmetic.hpp, `DeviceMulAdd`) — the fused step as the fused
/// intrinsic, the separate step as a product rounded once and then summed, a
/// form no contraction setting has anything to fuse — so what the device
/// delivers is what was selected, and `BoysCuda::MulAddRouteInForce()` reports
/// it.
///
/// That the spelling delivers the route it names is measured and not asserted:
/// tests/boys_cuda_route_test.cu evaluates both routes on the device over a
/// fixed value set and holds each to the arithmetic it names, bit for bit.
/// The reading is the delivered one and not a property of the flags.
///
/// **What the route reaches.** It covers the multiply-add sites the host's own
/// route covers — the piece summations and the region-B fast exponential, which
/// the host writes through its backend's `MulAdd`. The recurrence steps that
/// carry a product and a difference are the host's route-independent `MulSub`
/// sites: the host contracts them on no build, the device writes them bare, and
/// they are left alone here, because respelling them would move the fused
/// route's certified values.
inline constexpr backend::MulAddRoute kDeviceMulAddRoute =
#if defined(BOYS_MULADD_SEPARATE)
    backend::MulAddRoute::kSeparate;
#else
    backend::MulAddRoute::kFused;
#endif

} // namespace boys::detail

/// \endcond
