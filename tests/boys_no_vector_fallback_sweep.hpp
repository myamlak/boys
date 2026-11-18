#pragma once

// Claim P10's measurement: the no-vector fallback's entry points, each called over its
// documented region and every supported order, compared bit for bit with the certified scalar
// lane. The dedicated test and the accuracy gate's code.no_vector_fallback row share it, so the
// two report one measurement.

#include <bit>
#include <boys/boys.hpp>
#include <boys/f16.hpp>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace boys_no_vector_fallback {

/// What one run of the sweep measured.
struct Result {
    std::size_t cells = 0; ///< (entry, order, argument, output) cells compared
    std::size_t mismatches = 0; ///< cells where the return is not the scalar lane's bits
    std::string firstFailure; ///< the first mismatch, named; empty when none
};

namespace detail {

/// The batch sizes the sweep calls each entry at: the width of the vector body, one more, and
/// shapes below and beside it, so the region partitioning is exercised at every tail.
inline const std::size_t* BatchSizes() noexcept {
    static const std::size_t kSizes[] = {1, 3, 8, 9, 17};
    return kSizes;
}

inline constexpr std::size_t kBatchSizeCount = 5;

inline std::uint64_t BitsOf(double value) noexcept {
    return std::bit_cast<std::uint64_t>(value);
}

/// Records one compared cell. The first mismatch is kept with its location; the rest are
/// counted, because one failure is enough to refuse the claim and the count says whether it is
/// isolated.
inline void Compare(Result& result,
                    const char* entry,
                    int order,
                    double x,
                    std::size_t cell,
                    std::uint64_t got,
                    std::uint64_t want) {
    ++result.cells;

    if (got == want)
    {
        return;
    }

    ++result.mismatches;

    if (!result.firstFailure.empty())
    {
        return;
    }

    char text[256];
    std::snprintf(text,
                  sizeof(text),
                  "%s: n=%d x=%.17g cell %zu returned 0x%llx, the certified scalar lane "
                  "0x%llx",
                  entry,
                  order,
                  x,
                  cell,
                  static_cast<unsigned long long>(got),
                  static_cast<unsigned long long>(want));
    result.firstFailure = text;
}

inline std::vector<double> RegionAArguments() {
    return {0.0, 1e-12, 1e-8, 0.125, 0.5, 1.0, 2.5, 5.0, 8.0, 11.0, 11.8998};
}

inline std::vector<double> RegionBArguments() {
    return {boys::detail::kX0,
            11.9,
            12.5,
            15.0,
            20.0,
            25.0,
            28.0,
            std::nextafter(boys::detail::kX1, 0.0)};
}

inline std::vector<double> RegionCArguments() {
    return {boys::detail::kX1, 29.0, 29.5, 31.0, 40.0, 100.0, 1e3, 1e6};
}

inline std::vector<float> HalfRegionAArguments() {
    return {0.0f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 11.0f, 11.5f};
}

inline std::vector<float> HalfRegionBArguments() {
    return {12.0f, 16.0f, 20.0f, 24.0f, 28.0f, 28.5f};
}

inline std::vector<float> HalfRegionCArguments() {
    return {29.0f, 32.0f, 64.0f, 100.0f, 1000.0f, 65504.0f};
}

inline std::vector<float> Bf16RegionCArguments() {
    return {29.0f, 32.0f, 64.0f, 100.0f, 1000.0f, 1e6f, 1e30f};
}

/// One double entry, region A or C: out[i] is F_n(x[i]).
inline void SweepDoubleRegion(Result& result,
                              const char* entry,
                              bool regionA,
                              const std::vector<double>& arguments) {
    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t c = 0; c < kBatchSizeCount; ++c)
        {
            const std::size_t count = BatchSizes()[c];
            std::vector<double> x(count);
            std::vector<double> out(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = arguments[i % arguments.size()];
            }

            if (regionA)
            {
                boys::detail::BoysRegionASimd(n, x.data(), out.data(), count);
            } else
            {
                boys::detail::BoysRegionCSimd(n, x.data(), out.data(), count);
            }

            for (std::size_t i = 0; i < count; ++i)
            {
                Compare(
                    result, entry, n, x[i], i, BitsOf(out[i]), BitsOf(boys::BoysSingle(n, x[i])));
            }
        }
    }
}

/// Region B's double entry: out[order * count + i] is F_order(x[i]) for every order to n.
inline void SweepDoubleB(Result& result, const std::vector<double>& arguments) {
    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t c = 0; c < kBatchSizeCount; ++c)
        {
            const std::size_t count = BatchSizes()[c];
            std::vector<double> x(count);
            std::vector<double> out(count * static_cast<std::size_t>(n + 1));

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = arguments[i % arguments.size()];
            }

            boys::detail::BoysRegionBSimd(n, x.data(), out.data(), count);

            std::vector<double> reference(static_cast<std::size_t>(n) + 1);

            for (std::size_t i = 0; i < count; ++i)
            {
                boys::BoysAllOrders(n, x[i], reference.data());

                for (int l = 0; l <= n; ++l)
                {
                    const std::size_t cell = static_cast<std::size_t>(l) * count + i;
                    Compare(result,
                            "RegionBSimd",
                            n,
                            x[i],
                            cell,
                            BitsOf(out[cell]),
                            BitsOf(reference[static_cast<std::size_t>(l)]));
                }
            }
        }
    }
}

/// The half lanes' I/O type, its entry points and its bit access.
template <typename HalfT> struct HalfLane;

template <> struct HalfLane<boys::F16> {
    static constexpr const char* kRegionA = "RegionASimdF16";
    static constexpr const char* kRegionB = "RegionBSimdF16";
    static constexpr const char* kRegionC = "RegionCSimdF16";

    static void RegionA(int n, const boys::F16* x, boys::F16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionASimdF16(n, x, out, count);
    }

    static void RegionB(int n, const boys::F16* x, boys::F16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionBSimdF16(n, x, out, count);
    }

    static void RegionC(int n, const boys::F16* x, boys::F16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionCSimdF16(n, x, out, count);
    }

    static boys::F16 Single(int n, boys::F16 x) noexcept {
        return boys::BoysSingleF16(n, x);
    }

    static void Batch(int n, boys::F16 x, boys::F16* out) noexcept {
        boys::BoysAllOrdersF16(n, x, out);
    }

    static std::uint16_t Bits(boys::F16 value) noexcept {
        return boys::detail::F16Bits(value);
    }
};

template <> struct HalfLane<boys::Bf16> {
    static constexpr const char* kRegionA = "RegionASimdBf16";
    static constexpr const char* kRegionB = "RegionBSimdBf16";
    static constexpr const char* kRegionC = "RegionCSimdBf16";

    static void RegionA(int n, const boys::Bf16* x, boys::Bf16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionASimdBf16(n, x, out, count);
    }

    static void RegionB(int n, const boys::Bf16* x, boys::Bf16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionBSimdBf16(n, x, out, count);
    }

    static void RegionC(int n, const boys::Bf16* x, boys::Bf16* out, std::size_t count) noexcept {
        boys::detail::BoysRegionCSimdBf16(n, x, out, count);
    }

    static boys::Bf16 Single(int n, boys::Bf16 x) noexcept {
        return boys::BoysSingleBf16(n, x);
    }

    static void Batch(int n, boys::Bf16 x, boys::Bf16* out) noexcept {
        boys::BoysAllOrdersBf16(n, x, out);
    }

    static std::uint16_t Bits(boys::Bf16 value) noexcept {
        return boys::detail::Bf16Bits(value);
    }
};

/// The register region a widened half argument falls in, with the same float boundaries the
/// accuracy gate's half sweep partitions by.
inline int HalfRegion(float x) noexcept {
    const float x0 = static_cast<float>(boys::detail::kX0);
    const float x1 = static_cast<float>(boys::detail::kX1);

    if (x < x0)
    {
        return 0;
    }

    return x < x1 ? 1 : 2;
}

template <typename HalfT> std::vector<HalfT> Widen(const std::vector<float>& arguments) {
    std::vector<HalfT> out;
    out.reserve(arguments.size());

    for (const float argument : arguments)
    {
        out.emplace_back(argument);
    }

    return out;
}

/// Refuses an argument the sweep itself placed outside the entry's documented region, before it
/// is measured: a precondition the sweep breaks is a defect in the sweep and not in the entry.
template <typename HalfT>
bool ArgumentsAreInRegion(Result& result,
                          const char* entry,
                          const std::vector<HalfT>& arguments,
                          int region) {
    for (const HalfT argument : arguments)
    {
        const float x = static_cast<float>(argument);

        if (HalfRegion(x) != region)
        {
            ++result.mismatches;
            result.firstFailure = std::string(entry) + ": the sweep's own argument " +
                                  std::to_string(static_cast<double>(x)) + " is outside region " +
                                  std::to_string(region);
            return false;
        }
    }

    return true;
}

/// One half entry, region A or C: out[i] is F_n(x[i]).
template <typename HalfT>
void SweepHalfRegion(Result& result, int region, const std::vector<float>& arguments) {
    using Lane = HalfLane<HalfT>;
    const char* entry = region == 0 ? Lane::kRegionA : Lane::kRegionC;
    const std::vector<HalfT> values = Widen<HalfT>(arguments);

    if (!ArgumentsAreInRegion(result, entry, values, region))
    {
        return;
    }

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t c = 0; c < kBatchSizeCount; ++c)
        {
            const std::size_t count = BatchSizes()[c];
            std::vector<HalfT> x(count);
            std::vector<HalfT> out(count);

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = values[i % values.size()];
            }

            if (region == 0)
            {
                Lane::RegionA(n, x.data(), out.data(), count);
            } else
            {
                Lane::RegionC(n, x.data(), out.data(), count);
            }

            for (std::size_t i = 0; i < count; ++i)
            {
                Compare(result,
                        entry,
                        n,
                        static_cast<double>(static_cast<float>(x[i])),
                        i,
                        Lane::Bits(out[i]),
                        Lane::Bits(Lane::Single(n, x[i])));
            }
        }
    }
}

/// Region B's half entry: out[order * count + i] is F_order(x[i]) for every order to n.
template <typename HalfT> void SweepHalfB(Result& result, const std::vector<float>& arguments) {
    using Lane = HalfLane<HalfT>;
    const std::vector<HalfT> values = Widen<HalfT>(arguments);

    if (!ArgumentsAreInRegion(result, Lane::kRegionB, values, 1))
    {
        return;
    }

    for (int n = 0; n <= boys::kMaxBoysOrder; ++n)
    {
        for (std::size_t c = 0; c < kBatchSizeCount; ++c)
        {
            const std::size_t count = BatchSizes()[c];
            std::vector<HalfT> x(count);
            std::vector<HalfT> out(count * static_cast<std::size_t>(n + 1));

            for (std::size_t i = 0; i < count; ++i)
            {
                x[i] = values[i % values.size()];
            }

            Lane::RegionB(n, x.data(), out.data(), count);

            std::vector<HalfT> reference(static_cast<std::size_t>(n) + 1);

            for (std::size_t i = 0; i < count; ++i)
            {
                Lane::Batch(n, x[i], reference.data());

                for (int l = 0; l <= n; ++l)
                {
                    const std::size_t cell = static_cast<std::size_t>(l) * count + i;
                    Compare(result,
                            Lane::kRegionB,
                            n,
                            static_cast<double>(static_cast<float>(x[i])),
                            cell,
                            Lane::Bits(out[cell]),
                            Lane::Bits(reference[static_cast<std::size_t>(l)]));
                }
            }
        }
    }
}

} // namespace detail

/// Runs every entry the no-vector fallback of src/boys_simd.cpp defines, for both lane families,
/// over each entry's documented region and every supported order.
inline Result Sweep() {
    Result result;

    detail::SweepDoubleRegion(result, "RegionASimd", true, detail::RegionAArguments());
    detail::SweepDoubleB(result, detail::RegionBArguments());
    detail::SweepDoubleRegion(result, "RegionCSimd", false, detail::RegionCArguments());

#if BoysFp16
    detail::SweepHalfRegion<boys::F16>(result, 0, detail::HalfRegionAArguments());
    detail::SweepHalfB<boys::F16>(result, detail::HalfRegionBArguments());
    detail::SweepHalfRegion<boys::F16>(result, 2, detail::HalfRegionCArguments());
    detail::SweepHalfRegion<boys::Bf16>(result, 0, detail::HalfRegionAArguments());
    detail::SweepHalfB<boys::Bf16>(result, detail::HalfRegionBArguments());
    detail::SweepHalfRegion<boys::Bf16>(result, 2, detail::Bf16RegionCArguments());
#endif

    return result;
}

} // namespace boys_no_vector_fallback
