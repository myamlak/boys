// One digest of every default entry's bits, for the build to print from four sets of flags: the
// entries are the caller's to compile, and the four agree because the arithmetic is one source and
// the ISA tiers are documented bit-equal to the scalar twins. A digest that differs is a finding.

#include <boys/boys.hpp>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace {

std::uint64_t gDigest = 14695981039346656037ull;

void MixByte(std::uint8_t byte) {
    gDigest ^= byte;
    gDigest *= 1099511628211ull;
}

template <typename UInt>
void MixUInt(UInt value) {
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    {
        MixByte(static_cast<std::uint8_t>(value >> (8 * byte)));
    }
}

/// The bits of one value, mixed into the digest. Values are mixed as bits and not as numbers: a
/// digest over rounded decimals would report agreement between two arithmetics that differ in
/// the last bit, which is the difference this program exists to find.
void Mix(double value) { MixUInt(std::bit_cast<std::uint64_t>(value)); }
void Mix(float value) { MixUInt(std::bit_cast<std::uint32_t>(value)); }
#if BoysFp16
void Mix(boys::F16 value) { MixUInt(std::bit_cast<std::uint16_t>(value)); }
void Mix(boys::Bf16 value) { MixUInt(std::bit_cast<std::uint16_t>(value)); }
#endif

constexpr std::size_t kGrid = 32000; ///< x = i / 64 for i below this, so 0 .. 499.984375
constexpr int kOrders = boys::kMaxBoysOrder;
constexpr std::size_t kBatch = 64;

/// The i-th argument: `i / 64`.
///
/// Every argument is a multiple of 1/64 below 500, so it is exact in binary and every variant
/// computes the same value for it. An argument whose own rounding moved with the flags would put
/// a difference into the digest that the entries are not responsible for.
///
/// \param index which argument, below the grid
/// \returns the argument at that index
template <typename T>
T Argument(std::size_t index) {
    return static_cast<T>(static_cast<double>(index) / 64.0);
}

/// One batch argument: spread over the same range so the array entries meet all three regions.
template <typename T>
T BatchArgument(std::size_t index) {
    return Argument<T>(((index * 499) % kGrid));
}

void SweepDouble() {
    double all[kOrders + 1] = {};
    for (std::size_t i = 0; i < kGrid; i += 7)
    {
        const double x = Argument<double>(i);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(boys::BoysSingle(n, x));
        }
        boys::BoysAllOrders(kOrders, x, all);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(all[n]);
        }
        double checked = 0.0;
        if (boys::BoysSingleChecked(kOrders, x, &checked) == boys::BoysStatus::kSuccess)
        {
            Mix(checked);
        }
    }

    double xs[kBatch] = {};
    double ys[kBatch] = {};
    double planes[kBatch * (kOrders + 1)] = {};
    int tops[kBatch] = {};
    for (std::size_t i = 0; i < kBatch; ++i)
    {
        xs[i] = BatchArgument<double>(i);
        tops[i] = static_cast<int>(i % (kOrders + 1));
    }
    boys::BoysFixedN(3, xs, ys, kBatch, 1);
    for (double value : ys)
    {
        Mix(value);
    }
    std::size_t workspace[kBatch] = {};
    boys::BoysAllN(kOrders, xs, planes, kBatch, workspace);
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planes[cell]);
    }
    boys::BoysAllN(kOrders, xs, planes, kBatch, boys::BoysSortedArgs{});
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planes[cell]);
    }
    boys::BoysAllNAtOrders(tops, xs, planes, kBatch);
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planes[cell]);
    }
}

void SweepFloat() {
    float all[kOrders + 1] = {};
    for (std::size_t i = 0; i < kGrid; i += 7)
    {
        const float x = Argument<float>(i);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(boys::BoysSingleF32(n, x));
        }
        boys::BoysAllOrdersF32(kOrders, x, all);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(all[n]);
        }
    }

    float xs[kBatch] = {};
    float ys[kBatch] = {};
    float planes[kBatch * (kOrders + 1)] = {};
    int tops[kBatch] = {};
    for (std::size_t i = 0; i < kBatch; ++i)
    {
        xs[i] = BatchArgument<float>(i);
        tops[i] = static_cast<int>(i % (kOrders + 1));
    }
    boys::BoysFixedNF32(3, xs, ys, kBatch, 1);
    for (float value : ys)
    {
        Mix(value);
    }
    boys::BoysAllNF32(kOrders, xs, planes, kBatch);
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planes[cell]);
    }
    boys::BoysAllNAtOrdersF32(tops, xs, planes, kBatch);
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planes[cell]);
    }
}

#if BoysFp16
void SweepHalf() {
    for (std::size_t i = 0; i < kGrid; i += 7)
    {
        const boys::F16 x = Argument<boys::F16>(i);
        const boys::Bf16 y = Argument<boys::Bf16>(i);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(boys::BoysSingleF16(n, x));
            Mix(boys::BoysSingleBf16(n, y));
        }
        boys::F16 allF16[kOrders + 1] = {};
        boys::Bf16 allBf16[kOrders + 1] = {};
        boys::BoysAllOrdersF16(kOrders, x, allF16);
        boys::BoysAllOrdersBf16(kOrders, y, allBf16);
        for (int n = 0; n <= kOrders; ++n)
        {
            Mix(allF16[n]);
            Mix(allBf16[n]);
        }
    }

    boys::F16 xs[kBatch] = {};
    boys::Bf16 ys[kBatch] = {};
    boys::F16 ysF16[kBatch] = {};
    boys::Bf16 ysBf16[kBatch] = {};
    boys::F16 planesF16[kBatch * (kOrders + 1)] = {};
    boys::Bf16 planesBf16[kBatch * (kOrders + 1)] = {};
    int tops[kBatch] = {};
    for (std::size_t i = 0; i < kBatch; ++i)
    {
        xs[i] = BatchArgument<boys::F16>(i);
        ys[i] = BatchArgument<boys::Bf16>(i);
        tops[i] = static_cast<int>(i % (kOrders + 1));
    }
    boys::BoysFixedNF16(3, xs, ysF16, kBatch, 1);
    boys::BoysFixedNBf16(3, ys, ysBf16, kBatch, 1);
    boys::BoysAllNF16(kOrders, xs, planesF16, kBatch);
    boys::BoysAllNBf16(kOrders, ys, planesBf16, kBatch);
    boys::BoysAllNAtOrdersF16(tops, xs, planesF16, kBatch);
    boys::BoysAllNAtOrdersBf16(tops, ys, planesBf16, kBatch);
    for (std::size_t i = 0; i < kBatch; ++i)
    {
        Mix(ysF16[i]);
        Mix(ysBf16[i]);
    }
    for (std::size_t cell = 0; cell < kBatch * (kOrders + 1); ++cell)
    {
        Mix(planesF16[cell]);
        Mix(planesBf16[cell]);
    }
}
#endif // BoysFp16

} // namespace

int main() {
    SweepDouble();
    SweepFloat();
#if BoysFp16
    SweepHalf();
#endif
    std::printf("%016llx\n", static_cast<unsigned long long>(gDigest));
    return 0;
}
