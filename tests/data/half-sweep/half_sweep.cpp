// Exhaustive sweep of the shipped half lanes over every positive finite 16-bit argument.
//
// The lane measured is the one the gate's half rows call, in tests/boys_accuracy_gate.cpp,
// "fp16 and bf16, the store-half lane": BoysSingleF16(n, F16) and BoysSingleBf16(n, Bf16), the
// scalar store-half entries, judged by the claim slots "fp16 store-half"/"single" (kF16Single)
// and "bf16 store-half"/"single" (kBf16Single). Their bound is HalfBound(got, mantissaBits,
// minNormalExp) from tests/boys_gate_reference.hpp, copied verbatim below: an absolute figure,
// kBoundHalfBase = 1e-7, plus half an ulp of the returned value.
//
// For every positive finite bit pattern of each format, and for orders 0..32, one row:
//   format, n, argument bits, returned bits, returned value as double, the lane's bound there.
// The arguments are the bit patterns themselves (FromBits), so no conversion can move one.
//
// Build (MSVC x64, Release), from this directory:
//   cl /std:c++20 /O2 /EHsc /I<boys>/include half_sweep.cpp <boys>.lib /Fe:half_sweep.exe
// or with CMake against the v5.1.1 tree, as the deposited build.txt records.

#include <boys/boys.hpp>
#include <boys/f16.hpp>

#ifndef BoysFp16
#error "build with BoysFp16=1: the half entries sit behind the library's half seam"
#endif

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr double kBoundHalfBase = 1e-7;   // tests/boys_gate_reference.hpp
constexpr int kF16MantissaBits = 10;      // tests/boys_gate_reference.hpp
constexpr int kF16MinNormalExp = -14;     // tests/boys_gate_reference.hpp
constexpr int kBf16MantissaBits = 7;      // tests/boys_accuracy_gate.cpp
constexpr int kBf16MinNormalExp = -126;   // tests/boys_accuracy_gate.cpp

// Copied from tests/boys_gate_reference.hpp: the bound and the ulp it is built from.
inline double UlpOf(double v, int mantissaBits, int minNormalExp) {
    if (v == 0.0 || !std::isfinite(v))
    {
        return std::ldexp(1.0, minNormalExp - mantissaBits);
    }

    int e = std::ilogb(std::fabs(v));

    if (e < minNormalExp)
    {
        e = minNormalExp;
    }

    return std::ldexp(1.0, e - mantissaBits);
}

inline double HalfBound(double got, int mantissaBits, int minNormalExp) {
    return kBoundHalfBase + 0.5 * UlpOf(got, mantissaBits, minNormalExp);
}

constexpr int kMaxOrder = 32;
constexpr std::uint32_t kMagic = 0x50535748u; // "HSWP", little-endian

struct Sink {
    std::FILE* f = nullptr;
    std::vector<unsigned char> buf;
    static constexpr std::size_t kFlushAt = std::size_t(1) << 22;

    explicit Sink(const char* path) {
        f = std::fopen(path, "wb");
        if (f == nullptr)
        {
            std::perror("half_sweep: cannot open output");
            std::exit(2);
        }
        buf.reserve(kFlushAt + 64);
    }

    void Byte(unsigned char b) { buf.push_back(b); }
    void U16(std::uint16_t v) { Byte(v & 0xFF); Byte((v >> 8) & 0xFF); }
    void U32(std::uint32_t v) { U16(v & 0xFFFF); U16((v >> 16) & 0xFFFF); }
    void U64(std::uint64_t v) { U32(v & 0xFFFFFFFFu); U32(v >> 32); }
    void F64(double v) { std::uint64_t u; std::memcpy(&u, &v, 8); U64(u); }

    void MaybeFlush() {
        if (buf.size() >= kFlushAt)
        {
            Flush();
        }
    }

    void Flush() {
        if (!buf.empty())
        {
            std::fwrite(buf.data(), 1, buf.size(), f);
            buf.clear();
        }
        std::fflush(f);
    }
};

void Cell(Sink& s,
          unsigned char format,
          std::uint8_t n,
          std::uint16_t argBits,
          std::uint16_t gotBits,
          double got,
          double bound) {
    s.Byte(format);
    s.Byte(n);
    s.U16(argBits);
    s.U16(gotBits);
    s.F64(got);
    s.F64(bound);
    s.MaybeFlush();
}

} // namespace

int main(int argc, char** argv) {
    const char* outPath = argc > 1 ? argv[1] : "half_sweep.bin";
    Sink s(outPath);

    s.U32(kMagic);
    s.U32(1);                       // format version
    s.U32(kMaxOrder);               // orders 0..kMaxOrder
    s.U64(0);                       // row count, filled after the sweep, below

    std::uint64_t f16Args = 0, f16Rows = 0, bf16Args = 0, bf16Rows = 0;

    // binary16: every positive finite pattern, 0x0001 .. 0x7BFF.
    for (std::uint32_t bits = 0x0001; bits <= 0x7BFFu; ++bits)
    {
        const boys::F16 x = boys::F16::FromBits(static_cast<std::uint16_t>(bits));
        ++f16Args;

        for (int n = 0; n <= kMaxOrder; ++n)
        {
            const boys::F16 got = boys::BoysSingleF16(n, x);
            const double gotd = static_cast<double>(static_cast<float>(got));
            Cell(s,
                 0,
                 static_cast<std::uint8_t>(n),
                 static_cast<std::uint16_t>(bits),
                 got.Bits(),
                 gotd,
                 HalfBound(gotd, kF16MantissaBits, kF16MinNormalExp));
            ++f16Rows;
        }
    }

    // bfloat16: every positive finite pattern, 0x0001 .. 0x7F7F.
    for (std::uint32_t bits = 0x0001; bits <= 0x7F7Fu; ++bits)
    {
        const boys::Bf16 x = boys::Bf16::FromBits(static_cast<std::uint16_t>(bits));
        ++bf16Args;

        for (int n = 0; n <= kMaxOrder; ++n)
        {
            const boys::Bf16 got = boys::BoysSingleBf16(n, x);
            const double gotd = static_cast<double>(static_cast<float>(got));
            Cell(s,
                 1,
                 static_cast<std::uint8_t>(n),
                 static_cast<std::uint16_t>(bits),
                 got.Bits(),
                 gotd,
                 HalfBound(gotd, kBf16MantissaBits, kBf16MinNormalExp));
            ++bf16Rows;
        }
    }

    s.Flush();
    std::fclose(s.f);

    // The row count the header reserved: rewritten in place, so the file is self-describing.
    if (std::FILE* g = std::fopen(outPath, "r+b"))
    {
        const std::uint64_t rows = f16Rows + bf16Rows;
        std::fseek(g, 12, SEEK_SET);
        std::uint32_t lo = static_cast<std::uint32_t>(rows & 0xFFFFFFFFu);
        std::uint32_t hi = static_cast<std::uint32_t>(rows >> 32);
        std::fwrite(&lo, 4, 1, g);
        std::fwrite(&hi, 4, 1, g);
        std::fclose(g);
    }

    std::printf("binary16: %llu arguments, %llu cells\n",
                static_cast<unsigned long long>(f16Args),
                static_cast<unsigned long long>(f16Rows));
    std::printf("bfloat16: %llu arguments, %llu cells\n",
                static_cast<unsigned long long>(bf16Args),
                static_cast<unsigned long long>(bf16Rows));
    std::printf("total: %llu cells\n",
                static_cast<unsigned long long>(f16Rows + bf16Rows));
    return 0;
}
