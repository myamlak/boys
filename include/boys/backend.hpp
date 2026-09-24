#pragma once

/// \file backend.hpp
/// Named arithmetic backends: the value type, the packed type, the load /
/// store / broadcast transport, and the multiply-add with its contraction
/// behaviour, in one place per precision and width.
///
/// A lane is an arithmetic over one value type at one width, and the kernels
/// that differ only in that arithmetic are written once against this concept
/// instead of once per width. Which arithmetic a lane runs in is a name a
/// report can print (see BoysBackends), so a number a caller holds can be
/// attributed to the arithmetic that produced it rather than to a type only
/// the compiler knows.
///
/// Two multiply-adds are named, and the difference between them is the whole
/// point of naming them:
///
///  - MulAdd is fused: the product is exact and the sum rounds once. It is the
///    operation a Chebyshev recurrence wants, and the one the kernels are
///    written to use.
///  - MulSub rounds twice: the product rounds, then the difference rounds. The
///    two are not interchangeable, and neither is a compiler option: a source
///    that leaves `a * b - c` bare is a different arithmetic on a target that
///    contracts than on one that does not, so a kernel that needs the two
///    roundings says MulSub and gets them on every build.
///
/// Contracts() reports whether a bare `a * b + c` in a backend's arithmetic is
/// a single rounding in the build at hand. It is measured rather than
/// declared, because contraction is a property of the target and the flags
/// rather than of the source.
///
/// \ingroup boys

#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>

namespace boys {
namespace backend {

/// The arithmetic one lane runs in: a value type, a storage type, the packed
/// width the kernel operates at, the transport between them, and the
/// multiply-adds the kernels are built from.
///
/// The storage type is the one the lane's arguments and results are held in
/// and equals the value type except in the lanes that keep half-precision
/// arguments and compute in single (f16.hpp), where the same arithmetic is
/// reached through a narrower transport.
template <typename B>
concept ArithmeticBackend = requires(const typename B::Storage* storage,
                                     typename B::Storage* destination,
                                     typename B::Packed packed,
                                     typename B::Value scalar) {
    typename B::Value;
    typename B::Storage;
    typename B::Packed;
    { B::kWidth } -> std::convertible_to<std::size_t>;
    { B::kName } -> std::convertible_to<const char*>;
    { B::Load(storage) } -> std::same_as<typename B::Packed>;
    { B::Store(destination, packed) } -> std::same_as<void>;
    { B::Broadcast(scalar) } -> std::same_as<typename B::Packed>;
    { B::Mul(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Add(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Sub(packed, packed) } -> std::same_as<typename B::Packed>;
    { B::MulAdd(packed, packed, packed) } -> std::same_as<typename B::Packed>;
    { B::MulSub(packed, packed, packed) } -> std::same_as<typename B::Packed>;
    { B::Contracts() } -> std::same_as<bool>;
};

namespace detail {

/// The fused multiply-add of one scalar type, spelled for that type: the
/// single-precision form is the single-precision operation, not the
/// double-precision one narrowed afterwards.
template <typename T>
T Fused(T a, T b, T c) noexcept {
    if constexpr (std::is_same_v<T, float>)
    {
        return std::fmaf(a, b, c);
    } else
    {
        return std::fma(a, b, c);
    }
}

} // namespace detail

/// The scalar arithmetic of one precision: one value per operation, one
/// rounding per operation, and the fused multiply-add where one rounding is
/// what the kernel wants.
///
/// The scalar backend is also the arithmetic the split-precision lanes run
/// their recurrences in, and the reference the packed backends are held to.
template <typename T>
struct Scalar {
    static_assert(std::is_floating_point_v<T>, "a scalar backend is a floating-point type");

    /// The arithmetic's value type.
    using Value = T;

    /// The type the lane's arguments and results are stored in.
    using Storage = T;

    /// One value per operation: the scalar backend is its own packed form.
    using Packed = T;

    /// Values per packed operand.
    static constexpr std::size_t kWidth = 1;

    static constexpr const char* kName =
        std::is_same_v<T, float> ? "scalar-fp32" : "scalar-fp64";

    static Packed Load(const Storage* p) noexcept { return *p; }

    static void Store(Storage* p, Packed v) noexcept { *p = v; }

    static Packed Broadcast(Value v) noexcept { return v; }

    static Packed Mul(Packed a, Packed b) noexcept { return a * b; }

    static Packed Add(Packed a, Packed b) noexcept { return a + b; }

    static Packed Sub(Packed a, Packed b) noexcept { return a - b; }

    /// `a * b + c`, fused: the product is exact and the sum rounds once.
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        return detail::Fused(a, b, c);
    }

    /// `a * b - c` with two roundings: the product rounds, then the difference
    /// rounds. Written through the fused form with a zero addend rather than
    /// as the bare product and difference, because the bare form is a single
    /// fused step wherever the target has FMA and the contraction setting
    /// allows it — the default on aarch64, and on any x86 build that passes
    /// -mfma — so the same source would be two arithmetics. The zero addend is
    /// the product rounded once and nothing more, leaving a contraction pass
    /// nothing to fuse.
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return detail::Fused(a, b, Packed{0}) - c;
    }

    /// Whether a bare `a * b + c` written in this arithmetic is a single
    /// rounding in this build; see the file comment.
    static bool Contracts() noexcept;
};

/// The double-precision scalar arithmetic.
using ScalarFp64 = Scalar<double>;

/// The single-precision scalar arithmetic.
using ScalarFp32 = Scalar<float>;

namespace detail {

/// Whether the build contracts a bare product-plus-add in the scalar
/// arithmetic of `T`.
///
/// Measured, not declared: contraction follows from the target and the flags,
/// so the only answer worth reporting is the one the build gives. The operands
/// make the product inexact and then cancel its leading term, so a bare step
/// that contracts reaches the fused value exactly while one that does not
/// lands 2^-54 away in double and 2^-26 in single. The operands are read
/// through volatile so the compiler evaluates the expression rather than the
/// constant.
template <typename T>
bool MeasureContraction() noexcept {
    constexpr int kShift = std::numeric_limits<T>::digits / 2 + 1;
    const T unit{1};
    const T small = static_cast<T>(std::ldexp(1.0, -kShift));
    const T half = static_cast<T>(std::ldexp(1.0, -(kShift - 1)));
    volatile const T a = unit + small;
    volatile const T b = unit + small;
    volatile const T c = -(unit + half);
    const T productThenSum = a * b + c;
    const T fused = Fused<T>(a, b, c);
    return productThenSum == fused;
}

} // namespace detail

template <typename T>
bool Scalar<T>::Contracts() noexcept {
    return detail::MeasureContraction<T>();
}

/// One arithmetic backend as a report states it.
struct BackendInfo {
    /// The name a report prints for this arithmetic.
    const char* name;

    /// Whether a bare product-plus-add in this arithmetic is a single rounding
    /// in this build.
    bool contracts;
};

/// The arithmetic backends this build carries.
///
/// The scalar pair is always present. The packed pair is present exactly when
/// the build has the AVX2 + FMA tier, which is decided at run time on an
/// x86_64 build and absent entirely elsewhere.
///
/// \returns the backends, in a fixed order.
///
/// \ingroup boys
std::span<const BackendInfo> BoysBackends() noexcept;

} // namespace backend
} // namespace boys
