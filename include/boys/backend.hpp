#pragma once

/// \file backend.hpp
/// Named arithmetic backends: the value type, the packed type, the load / store /
/// broadcast transport, and the multiply-add with its contraction behaviour, in
/// one place per precision and width.
///
/// A lane is an arithmetic over one value type at one width, and the kernels that
/// differ only in that arithmetic are written once against this concept. Which
/// arithmetic a lane runs in is a name a report can print (see BoysBackends), so a
/// number a caller holds can be attributed to the arithmetic that produced it.
///
/// Two multiply-adds are named, because they are two arithmetics:
///
///  - MulAdd is fused: the product is exact and the sum rounds once. It is what a
///    Chebyshev recurrence wants, and where the target has no fused instruction it
///    is a call into the C runtime.
///  - MulSub rounds twice: the product rounds, then the difference rounds. A kernel
///    that needs the two roundings says so, because a bare `a * b - c` is one fused
///    step on a target that contracts and two roundings on one that does not.
///
/// The two routes coincide on a build that contracts a bare product-plus-add,
/// because the bare form then IS the fused step. That is measured rather than
/// assumed (Contracts()), and it is why RouteInForce() can report the fused route
/// for a build that was asked for the separate one.
///
/// Contracts() reports whether a bare `a * b + c` is a single rounding in a
/// backend's arithmetic. It is measured because contraction follows from the target
/// and the flags rather than from the source, so the answer belongs to one compiled
/// translation unit. BackendInfo carries the answer measured where that backend's
/// kernels are compiled, so the table describes the arithmetic the lanes actually
/// run.
///
/// What a call site selects about *how* an evaluation runs - the fit route
/// (FitRoute), the scheme the fit's coefficients are summed in (EvalScheme), a
/// single-precision engine's budget (BoysBudget), the axis a packed evaluation
/// vectorises over (PackAxis) and how narrowly the fitted domain is cut into pieces
/// (FitGranularity) - is carried as ONE parameter (SeamPolicy) rather than one per
/// axis, so a further axis is a field on the policy rather than a parameter on
/// every entry, engine and kernel between the call site and the fit.
///
/// \ingroup boys

#include "boys/accuracy.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <type_traits>

namespace boys {

namespace backend {

/// The arithmetic one lane runs in: a value type, a storage type, the packed width
/// the kernel operates at, the transport between them, the multiply-add route, and
/// the multiply-adds the kernels are built from.
///
/// The storage type is the one the lane's arguments and results are held in and
/// equals the value type except in the lanes that keep half-precision arguments and
/// compute in single (f16.hpp).
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
    { B::kRoute } -> std::convertible_to<MulAddRoute>;
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

inline const char* MulAddRouteName(MulAddRoute route) noexcept {
    switch (route)
    {
    case MulAddRoute::kFused:
        return "fused";
    case MulAddRoute::kSeparate:
        return "separate";
    }

    return "unknown";
}

namespace detail {

/// The fused multiply-add of one scalar type, spelled for that type: the
/// single-precision form is the single-precision operation, not the double-precision
/// one narrowed afterwards.
///
/// One rounding on every build. What it costs is the build's: a translation unit
/// compiled without the fused instruction set calls the runtime for it, and one
/// compiled with the set does not - MSVC stops referencing the runtime's `fma` under
/// /arch:AVX2, which the build facts record per leg as `fma.route`. Naming the
/// intrinsic here was measured to change nothing: the units that pay the call are
/// exactly those compiled without the flag.
///
/// \param a the multiplicand
/// \param b the multiplier
/// \param c the addend
///
/// \returns `a * b + c` with the product exact and the sum rounded once
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

/// The separate multiply-add: the product rounds, then the sum rounds.
///
/// Written bare, which costs no call on any target. On a target that contracts a bare
/// product-plus-add it is the fused step instead of two roundings, so a route that
/// means two roundings has to know its build; Contracts() is that question's answer
/// and RouteInForce() is where the two are put together.
///
/// \param a the multiplicand
/// \param b the multiplier
/// \param c the addend
///
/// \returns `a * b + c` with two roundings, where the build does not contract
template <typename T>
T Separate(T a, T b, T c) noexcept {
    return a * b + c;
}

/// The route this translation unit's arithmetic was compiled with.
///
/// A build fact, set once for the library and inherited by a consumer through the
/// same public compile definition, so a call site that instantiates a kernel in its
/// own unit gets the arithmetic the library reports. The default is the fused route:
/// the certified bounds are the fused arithmetic's, and a build that changes route
/// changes them.
///
/// **The value is read and not only the spelling**, so `-DBOYS_MULADD_SEPARATE=0` is
/// the fused route and not the separate one. CMake defines this as 1 or not at all,
/// but a consumer may pass the option's name with a value of their own, and a header
/// that answered the separate route for `=0` would select the arithmetic the caller
/// asked against - silently.
#if defined(BOYS_MULADD_SEPARATE) && BOYS_MULADD_SEPARATE
inline constexpr MulAddRoute kSelectedRoute = MulAddRoute::kSeparate;
#else
inline constexpr MulAddRoute kSelectedRoute = MulAddRoute::kFused;
#endif

/// Whether the build contracts a bare product-plus-add in the scalar
/// arithmetic of `T`. Declared here, defined below, because the route in force
/// is asked of it.
///
/// \returns whether the bare step and the fused step agree here
template <typename T>
bool MeasureContraction() noexcept;

/// The route in force for one precision.
///
/// The selected route, unless the build contracts the bare form: a contracting
/// build compiles both routes to the fused arithmetic, so the route in force is
/// the fused one whatever the selection says, and reporting the selection would
/// describe an arithmetic the lane does not run.
///
/// \returns the route the arithmetic of `T` is actually delivered by
template <typename T>
MulAddRoute RouteInForce() noexcept {
    // The selection is a compile-time constant and is tested as one: MSVC reports a
    // constant conditional expression under /W4, and this tree makes that an error.
    // Only a build that selected the separate route asks the contraction question.
    if constexpr (kSelectedRoute == MulAddRoute::kSeparate)
    {
        if (MeasureContraction<T>())
        {
            return MulAddRoute::kFused;
        }
    }

    return kSelectedRoute;
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

    /// The name a report prints this arithmetic under.
    static constexpr const char* kName =
        std::is_same_v<T, float> ? "scalar-fp32" : "scalar-fp64";

    /// The multiply-add route this backend was compiled with. The selection,
    /// not the route in force: whether the separate route is really two
    /// roundings is a property of the build's contraction, which is measured
    /// per translation unit, and BoysBackends reports that answer.
    static constexpr MulAddRoute kRoute = detail::kSelectedRoute;

    /// \param p one stored value
    ///
    /// \returns it as a packed operand
    static Packed Load(const Storage* p) noexcept { return *p; }

    /// \param p where the value is written
    /// \param v the value to write
    static void Store(Storage* p, Packed v) noexcept { *p = v; }

    /// \param v the value to replicate
    ///
    /// \returns `v` in the one lane there is
    static Packed Broadcast(Value v) noexcept { return v; }

    /// \param a the multiplicand
    /// \param b the multiplier
    ///
    /// \returns `a * b`, one rounding
    static Packed Mul(Packed a, Packed b) noexcept { return a * b; }

    /// \param a the augend
    /// \param b the addend
    ///
    /// \returns `a + b`, one rounding
    static Packed Add(Packed a, Packed b) noexcept { return a + b; }

    /// \param a the minuend
    /// \param b the subtrahend
    ///
    /// \returns `a - b`, one rounding
    static Packed Sub(Packed a, Packed b) noexcept { return a - b; }

    /// `a * b + c` at this backend's route.
    ///
    /// At the fused route the product is exact and the sum rounds once, which on a
    /// target without the instruction is a call into the C runtime. At the separate
    /// route the product rounds and then the sum rounds: two roundings, no call, and
    /// a different value.
    ///
    /// \param a the multiplicand
    /// \param b the multiplier
    /// \param c the addend
    ///
    /// \returns `a * b + c` at the selected route
    static Packed MulAdd(Packed a, Packed b, Packed c) noexcept {
        if constexpr (kRoute == MulAddRoute::kFused)
        {
            return detail::Fused(a, b, c);
        } else
        {
            return detail::Separate(a, b, c);
        }
    }

    /// `a * b - c` with two roundings: the product rounds, then the difference
    /// rounds.
    ///
    /// Written through the fused form with a zero addend rather than as the bare
    /// product and difference, because a bare `a * b - c` is a single fused step
    /// wherever the target has FMA and the contraction setting allows it - the default
    /// on aarch64, and on any x86 build that passes -mfma - so the same source would
    /// be two arithmetics. The zero addend is the product rounded once and nothing
    /// more, leaving a contraction pass nothing to fuse.
    ///
    /// \param a the multiplicand
    /// \param b the multiplier
    /// \param c the subtrahend
    ///
    /// The route does not reach this operation: its contract is two roundings on
    /// EVERY build, and the spelling above is the portable way to keep it. The zero
    /// addend costs one call on a target without the instruction, and that is the
    /// residue the separate route leaves.
    ///
    /// \returns `a * b - c` with two roundings
    static Packed MulSub(Packed a, Packed b, Packed c) noexcept {
        return detail::Fused(a, b, Packed{0}) - c;
    }

    /// Whether a bare `a * b + c` written in this arithmetic is a single rounding, as
    /// the translation unit that calls this compiles it. The answer is the asking
    /// unit's, because contraction follows from that unit's target and flags.
    ///
    /// \returns whether the bare form and the fused step agree here
    static bool Contracts() noexcept;
};

/// The double-precision scalar arithmetic.
using ScalarFp64 = Scalar<double>;

/// The single-precision scalar arithmetic.
using ScalarFp32 = Scalar<float>;

namespace detail {

/// Whether the build contracts a bare product-plus-add in the scalar arithmetic of
/// `T`.
///
/// Measured, not declared: contraction follows from the target and the flags, so the
/// only answer worth reporting is the one the build gives. The operands make the
/// product inexact and then cancel its leading term, so a bare step that contracts
/// reaches the fused value exactly while one that does not lands 2^-54 away in
/// double and 2^-26 in single. The operands are read through volatile so the compiler
/// evaluates the expression rather than the constant.
///
/// \returns whether the bare step and the fused step agree here
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

    /// The multiply-add route in force for this arithmetic: the route its values were
    /// computed at, and not merely the one the build selected. The two differ where a
    /// contracting build was asked for the separate route, because there the bare form
    /// IS the fused step.
    MulAddRoute route;
};

/// The arithmetic backends this build carries.
///
/// The scalar pair is always present. The packed pair is present exactly when the
/// build has the AVX2 + FMA tier, which is decided at run time on an x86_64 build and
/// absent entirely elsewhere.
///
/// \returns the backends, in a fixed order.
///
/// \ingroup boys
std::span<const BackendInfo> BoysBackends() noexcept;

} // namespace backend
} // namespace boys
