#include "boys/kernels.hpp"

// The front door's entries: each is the entry template at the policy this build's table carries
// for that entry's class, so a call site that names no policy links one of these instead of
// compiling the kernel itself. The namespace below is named from the build's seam, so two
// libraries built under different seams do not link each other's entries.

namespace boys {

inline namespace BOYS_DETAIL_SEAM_NAMESPACE {

double BoysSingle(int n, double x) noexcept {
    return boys::BoysSingle<DefaultPolicy<Precision::kFp64, Shape::kSingle>>(n, x);
}

BoysStatus BoysSingleChecked(int n, double x, double* out) noexcept {
    return boys::BoysSingleChecked<DefaultPolicy<Precision::kFp64, Shape::kSingle>>(n, x, out);
}

void BoysAllOrders(int nmax, double x, double* out) noexcept {
    return boys::BoysAllOrders<DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>(nmax, x, out);
}

void BoysFixedN(int n, const double* x, double* out, std::size_t count, std::size_t stride) noexcept {
    return boys::BoysFixedN<DefaultPolicy<Precision::kFp64, Shape::kFixedN>>(n, x, out, count, stride);
}

void BoysAllN(int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace) noexcept {
    return boys::BoysAllN<DefaultPolicy<Precision::kFp64, Shape::kAllN>>(nmax, x, out, count, workspace);
}

void BoysAllN(int nmax, const double* x, double* out, std::size_t count, BoysSortedArgs) noexcept {
    return boys::BoysAllN<DefaultPolicy<Precision::kFp64, Shape::kAllN>>(nmax, x, out, count, BoysSortedArgs{});
}

void BoysAllNAtOrders(const int* n, const double* x, double* out, std::size_t count) noexcept {
    return boys::BoysAllNAtOrders<DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>(n, x, out, count);
}

BoysStatus BoysAllOrdersChecked(int nmax, double x, double* out) noexcept {
    return boys::BoysAllOrdersChecked<DefaultPolicy<Precision::kFp64, Shape::kAllOrders>>(nmax, x, out);
}

BoysStatus BoysFixedNChecked(int n, const double* x, double* out, std::size_t count, std::size_t stride, std::size_t* badIndex) noexcept {
    return boys::BoysFixedNChecked<DefaultPolicy<Precision::kFp64, Shape::kFixedN>>(n, x, out, count, stride, badIndex);
}

BoysStatus BoysAllNChecked(int nmax, const double* x, double* out, std::size_t count, std::size_t* workspace, std::size_t* badIndex) noexcept {
    return boys::BoysAllNChecked<DefaultPolicy<Precision::kFp64, Shape::kAllN>>(nmax, x, out, count, workspace, badIndex);
}

BoysStatus BoysAllNAtOrdersChecked(const int* n, const double* x, double* out, std::size_t count, std::size_t* badIndex) noexcept {
    return boys::BoysAllNAtOrdersChecked<DefaultPolicy<Precision::kFp64, Shape::kAllNAtOrders>>(n, x, out, count, badIndex);
}

float BoysSingleF32(int n, float x) noexcept {
    return boys::BoysSingleF32<DefaultPolicy<Precision::kFp32, Shape::kSingle>>(n, x);
}

void BoysAllOrdersF32(int nmax, float x, float* out) noexcept {
    return boys::BoysAllOrdersF32<DefaultPolicy<Precision::kFp32, Shape::kAllOrders>>(nmax, x, out);
}

void BoysAllNF32(int nmax, const float* x, float* out, std::size_t count) noexcept {
    return boys::BoysAllNF32<DefaultPolicy<Precision::kFp32, Shape::kAllN>>(nmax, x, out, count);
}

void BoysFixedNF32(int n, const float* x, float* out, std::size_t count, std::size_t stride) noexcept {
    return boys::BoysFixedNF32<DefaultPolicy<Precision::kFp32, Shape::kFixedN>>(n, x, out, count, stride);
}

void BoysAllNAtOrdersF32(const int* n, const float* x, float* out, std::size_t count) noexcept {
    return boys::BoysAllNAtOrdersF32<DefaultPolicy<Precision::kFp32, Shape::kAllNAtOrders>>(n, x, out, count);
}

F16 BoysSingleF16(int n, F16 x) noexcept {
    return boys::BoysSingleF16<DefaultPolicy<Precision::kFp16, Shape::kSingle>>(n, x);
}

void BoysAllOrdersF16(int nmax, F16 x, F16* out) noexcept {
    return boys::BoysAllOrdersF16<DefaultPolicy<Precision::kFp16, Shape::kAllOrders>>(nmax, x, out);
}

void BoysFixedNF16(int n, const F16* x, F16* out, std::size_t count, std::size_t stride) noexcept {
    return boys::BoysFixedNF16<DefaultPolicy<Precision::kFp16, Shape::kFixedN>>(n, x, out, count, stride);
}

void BoysAllNF16(int nmax, const F16* x, F16* out, std::size_t count) noexcept {
    return boys::BoysAllNF16<DefaultPolicy<Precision::kFp16, Shape::kAllN>>(nmax, x, out, count);
}

void BoysAllNAtOrdersF16(const int* n, const F16* x, F16* out, std::size_t count) noexcept {
    return boys::BoysAllNAtOrdersF16<DefaultPolicy<Precision::kFp16, Shape::kAllNAtOrders>>(n, x, out, count);
}

Bf16 BoysSingleBf16(int n, Bf16 x) noexcept {
    return boys::BoysSingleBf16<DefaultPolicy<Precision::kBf16, Shape::kSingle>>(n, x);
}

void BoysAllOrdersBf16(int nmax, Bf16 x, Bf16* out) noexcept {
    return boys::BoysAllOrdersBf16<DefaultPolicy<Precision::kBf16, Shape::kAllOrders>>(nmax, x, out);
}

void BoysFixedNBf16(int n, const Bf16* x, Bf16* out, std::size_t count, std::size_t stride) noexcept {
    return boys::BoysFixedNBf16<DefaultPolicy<Precision::kBf16, Shape::kFixedN>>(n, x, out, count, stride);
}

void BoysAllNBf16(int nmax, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    return boys::BoysAllNBf16<DefaultPolicy<Precision::kBf16, Shape::kAllN>>(nmax, x, out, count);
}

void BoysAllNAtOrdersBf16(const int* n, const Bf16* x, Bf16* out, std::size_t count) noexcept {
    return boys::BoysAllNAtOrdersBf16<DefaultPolicy<Precision::kBf16, Shape::kAllNAtOrders>>(n, x, out, count);
}

} // namespace BOYS_DETAIL_SEAM_NAMESPACE

} // namespace boys
