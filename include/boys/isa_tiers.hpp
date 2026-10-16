#pragma once

/// \file isa_tiers.hpp
/// The instruction-set tier a translation unit is compiled at, and the guard that refuses a
/// program assembled from more than one.
///
/// The tier is chosen at compile time, from the flags the translation unit itself carries, and
/// never from the host's processor: a body selected at run time has to be present in the binary
/// twice, which is a library behind an entry and not a header-only one. Every consumer-reachable
/// lane therefore has one body per tier, chosen by the preprocessor, and MSVC's `/arch:AVX2` and
/// GCC's and Clang's `-mavx2 -mfma -mf16c` are the two spellings of the same choice.
///
/// ## The macros
///
/// \c BOYS_ISA_AVX2_FMA is 0 or 1 and is derived from the compiler's own predefines. A consumer
/// may define it before including any header of this library to answer it directly, which is what
/// a test that wants the other tier of a build asks for.
///
/// ## What no flags means
///
/// No flags is the scalar tier, and it is a supported, portable default rather than a
/// degradation: every entry the library declares has a body that compiles without a single
/// intrinsic, and those bodies are the certified lane the rest of the library is measured
/// against. A build that wants the vector tier says so once, for the whole program.
///
/// ## One flag set per program
///
/// **Every translation unit of a program must be compiled with the same flag set**, the
/// library's own included. It is not a suggestion: the two tiers' bodies are two definitions of
/// the same inline function, and a program that carries both has one of them and no way to say
/// which - the linker would pick, silently. What this header does about that is refuse: a small
/// registry records the tier each translation unit announces, and the first program state where
/// two of them disagree prints the two flag profiles and aborts, before any entry runs. The
/// registry is one object across the whole program (an inline function's function-local static)
/// and the announcement is one internal-linkage object per translation unit, so the check costs
/// a comparison at start-up and no call at all afterwards.
///
/// Being one flag set is what lets a consumer's translation unit define the same inline entry
/// the library does and get the same arithmetic from it.

#include <cstdio>
#include <cstdlib>

#if defined(__x86_64__) || defined(_M_X64)

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

#endif

// The AVX2 + FMA tier's flag set, as one question a body can ask. AVX2 alone does not answer it:
// the packed lanes enter the fused instruction unguarded, so a target with one and not the other
// is the scalar tier. One flag set, in each driver's spelling: GCC and Clang define __AVX2__ with
// __FMA__; cl 19.x defines __AVX2__ alone for /arch:AVX2 (measured), so there that macro answers.
#if !defined(BOYS_ISA_AVX2_FMA)
#if (defined(__AVX2__) && defined(__FMA__)) || (defined(_MSC_VER) && defined(__AVX2__)) || \
    defined(_M_AVX2)
#define BOYS_ISA_AVX2_FMA 1
#else
#define BOYS_ISA_AVX2_FMA 0
#endif
#endif

#if (BOYS_ISA_AVX2_FMA != 0 && BOYS_ISA_AVX2_FMA != 1)
#error "BOYS_ISA_AVX2_FMA is 0 or 1; a consumer overriding it states which"
#endif

// The fp16 seam is the same kind of per-translation-unit choice as an ISA tier - the entries it
// gates are added to the surface, not switched at run time - so it is registered beside them
// and a program that carries both readings of it is refused for the same reason.
#if BoysFp16
#define BOYS_FP16_AXIS 1
#else
#define BOYS_FP16_AXIS 0
#endif

namespace boys::detail {

/// Reads the AVX2 + FMA scope off the processor: CPUID and OSXSAVE on x86_64, and false on every
/// other target.
///
/// This answers what the MACHINE carries and not what this translation unit was compiled for -
/// \c BOYS_ISA_AVX2_FMA is the other question, and the two are allowed to disagree. It is a
/// report the library hands a caller (`boys::BoysAvx2Available()`); no arm of a lane is selected
/// by it. The body is identical in every translation unit, which is what makes one inline
/// definition of it correct.
///
/// \returns \c true when the processor carries AVX2, FMA and the OS state that enables them
///
/// \ingroup boys
inline bool DetectAvx2() noexcept {
#if defined(__x86_64__) || defined(_M_X64)
#ifdef _MSC_VER
    int cpuInfo[4] = {};
    int leaf1[4] = {};
    __cpuidex(cpuInfo, 7, 0);
    __cpuid(leaf1, 1);
    const bool osXsave = (leaf1[2] & (1u << 27)) != 0; // OSXSAVE
    const bool fma = (leaf1[2] & (1u << 12)) != 0; // FMA
    const bool avx2 = (cpuInfo[1] & (1u << 5)) != 0;
    return osXsave && fma && avx2;
#else
    // GCC/Clang builds: the kernel enables the AVX XCR0 state whenever the CPU
    // supports it, so the leaf-1 feature bits are authoritative.
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    const bool osXsave = (ecx & (1u << 27)) != 0; // OSXSAVE
    const bool fma = (ecx & (1u << 12)) != 0; // FMA

    if (!osXsave || !fma)
    {
        return false;
    }

    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx) == 0)
    {
        return false;
    }

    const bool avx2 = (ebx & (1u << 5)) != 0;
    return avx2;
#endif
#else
    // Not a target with a vector engine at all: there is no scope to report and no
    // lane behind this answer.
    return false;
#endif
}

/// The cached answer to DetectAvx2(), for the library's own callers: inline, so an entry that
/// answers one argument at a time pays a load rather than a call. One function-local static, so
/// it is one object across every translation unit. boys/boys.hpp declares the entry of this name
/// that a consumer calls; src/boys.cpp defines it.
inline bool Avx2Available() noexcept {
    static const bool available = DetectAvx2();
    return available;
}

/// The registry of the tiers a program's translation units announced, and the refusal when they
/// disagree. Not API: a consumer reads the contract at the top of this header and never names
/// anything here.
namespace tier_guard {

/// One program's announced tiers, one field per axis. A field left at kUnannounced has heard
/// from no translation unit yet.
struct Registry {
    static constexpr int kUnannounced = -1;

    int isa = kUnannounced; ///< BOYS_ISA_AVX2_FMA, as a translation unit announced it
    int fp16 = kUnannounced; ///< the fp16 seam, as a translation unit announced it
};

/// The one registry in the program: the function is inline, so its function-local static is the
/// same object in every translation unit.
inline Registry& Announced() noexcept {
    static Registry registry;
    return registry;
}

/// The AVX2 + FMA tier's profile as its own flag set spells it.
inline const char* IsaProfile(int tier) noexcept {
    return tier != 0 ? "AVX2+FMA (BOYS_ISA_AVX2_FMA=1)" : "scalar (BOYS_ISA_AVX2_FMA=0)";
}

/// The fp16 seam's profile as its own flag set spells it.
inline const char* Fp16Profile(int seam) noexcept {
    return seam != 0 ? "the fp16 seam open (BoysFp16=1)" : "the fp16 seam closed (BoysFp16=0)";
}

/// Refuses a program whose translation units were not compiled alike: one line naming the two
/// profiles and the axis they disagree on, then an abort.
[[noreturn]] inline void MixedTiers(const char* axis,
                                    const char* mine,
                                    const char* theirs) noexcept {
    std::fprintf(stderr,
                 "boys: %s is not one flag set across this program - this translation unit is "
                 "%s, an earlier one was %s; compile every translation unit of a program, this "
                 "library's included, with one flag set\n",
                 axis, mine, theirs);
    std::fflush(stderr);
    std::abort();
}

/// Announces this translation unit's two axes, refusing a program that announced both readings
/// of either. Announcing a value that is already registered is the common case - a program's
/// translation units are compiled alike - and costs a comparison.
inline void RegisterTier(int isa, int fp16) noexcept {
    Registry& registry = Announced();

    if (registry.isa == Registry::kUnannounced)
    {
        registry.isa = isa;
    } else if (registry.isa != isa)
    {
        MixedTiers("the instruction-set tier", IsaProfile(isa), IsaProfile(registry.isa));
    }

    if (registry.fp16 == Registry::kUnannounced)
    {
        registry.fp16 = fp16;
    } else if (registry.fp16 != fp16)
    {
        MixedTiers("the fp16 seam", Fp16Profile(fp16), Fp16Profile(registry.fp16));
    }
}

/// One translation unit's announcement, made before main: the tier this unit's flags name, in a
/// variable of its own - internal linkage, so every translation unit has one and every one of
/// them runs.
template <int kIsaTier, int kFp16Seam>
struct TierProbe {
    TierProbe() noexcept { RegisterTier(kIsaTier, kFp16Seam); }
};

/// The probe itself, one per translation unit that includes this header.
[[maybe_unused]] const TierProbe<BOYS_ISA_AVX2_FMA, BOYS_FP16_AXIS> kTierProbe;

} // namespace tier_guard

} // namespace boys::detail
