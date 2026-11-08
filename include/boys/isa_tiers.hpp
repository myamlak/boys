#pragma once

/// \file isa_tiers.hpp
/// The instruction-set tier a translation unit is compiled at, the token that gives each tier's
/// bodies their own symbol names, and the guard that keeps a program's tiers consistent.
///
/// ## The two ways a tier is chosen
///
/// A program compiled with \c BOYS_ISA_DISPATCH=0 (the default) chooses its tier once, from the
/// flags each translation unit carries, and never from the host's processor. That is today's
/// shape: a body selected at run time would have to be present in the binary twice, which is a
/// library behind an entry and not a header-only one.
///
/// A program compiled with \c BOYS_ISA_DISPATCH=1 carries every tier the target can build, in one
/// image, and a dispatcher selects among them on the first call. The default entries become
/// declarations in the header and are defined once in \c boys_dispatch.cpp; each carried tier's
/// entries are defined in that tier's own unit, compiled at that tier's flags. The selection is
/// one cached branch in the dispatcher, not a function pointer and not a per-order choice.
///
/// ## The token
///
/// The two tiers instantiate the same header templates with different bodies, so the linker must
/// not be allowed to pick one of them: each tier's bodies live in a namespace whose name is part
/// of every symbol it defines. \c BOYS_DETAIL_ISA_NAMESPACE is that token, derived here from the
/// tier macro in force:
///
/// - \c isa_avx2 when \c BOYS_ISA_AVX2_FMA is 1,
/// - \c isa_baseline otherwise,
/// - or \c BOYS_DETAIL_ISA_TOKEN when a unit states the token directly.
///
/// The token namespace is inline (the standard library's \c std::__cxx11 pattern): the public
/// spelling \c boys::BoysSingle is unchanged, while the mangled name carries the token and two
/// tiers' bodies are two symbols. The tier bodies sit one level further in,
/// \c boys::isa_avx2::kernel, so that the tier-independent coefficient tables keep their
/// \c boys::detail spelling and a body's own \c detail::kX0 keeps naming those tables.
///
/// The token is internal: a client names \c boys::BoysSingle, never a token namespace. A dispatch
/// build's client units are compiled at the baseline tier (the guard refuses any other), so the
/// only token a client sees is \c isa_baseline.
///
/// The guard's functions carry the token too: they are inline in this header, because a program
/// that includes only the headers links the announcement it makes and has no library unit to
/// call, and each tier's copy is its own symbol. The registry they announce to is one object
/// outside the token, shared by every translation unit and every tier of one image.
///
/// ## What no flags means
///
/// No flags is the scalar tier, and it is a supported, portable default rather than a
/// degradation: every entry the library declares has a body that compiles without a single
/// intrinsic, and those bodies are the certified lane the rest of the library is measured
/// against.
///
/// ## One flag set per program, or a dispatcher
///
/// With dispatch off, **every translation unit of a program must be compiled with the same flag
/// set**, the library's own included: the two tiers' bodies would be two definitions of one
/// inline function, and a program that carries both has one of them and no way to say which. The
/// registry below records the tier each translation unit announces and aborts with both profiles
/// on the first disagreement. With dispatch on the rule is different, not absent: a client
/// registers as a dispatch client and must be at the baseline tier; a tier unit registers its
/// tier as carried; the dispatcher's first call asserts the selected tier is in the carried set.

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

/// This translation unit's AVX2 + FMA tier: \c 1 on the vector tier, \c 0 on the scalar one.
#if !defined(BOYS_ISA_AVX2_FMA)
#if (defined(__AVX2__) && defined(__FMA__)) || (defined(_MSC_VER) && defined(__AVX2__))
#define BOYS_ISA_AVX2_FMA 1
#else
#define BOYS_ISA_AVX2_FMA 0
#endif
#endif

#if (BOYS_ISA_AVX2_FMA != 0 && BOYS_ISA_AVX2_FMA != 1)
#error "BOYS_ISA_AVX2_FMA is 0 or 1; a consumer overriding it states which"
#endif

// The build mode. A dispatch build states it once for the whole target; every translation unit of
// the program, the library's and the consumer's, reads the same answer. A unit that defines
// BOYS_ISA_TIER_UNIT is one of the carried tiers; the one unit that defines
// BOYS_ISA_DISPATCH_DEFINE is the dispatcher.

/// This translation unit's build mode: \c 1 when the library carries every tier, \c 0 for the
/// compile-time choice.
#if !defined(BOYS_ISA_DISPATCH)
#define BOYS_ISA_DISPATCH 0
#endif

#if (BOYS_ISA_DISPATCH != 0 && BOYS_ISA_DISPATCH != 1)
#error "BOYS_ISA_DISPATCH is 0 or 1; a build states which shape it has"
#endif

/// This unit is one carried tier of a dispatch build.
#if defined(BOYS_ISA_TIER_UNIT) && BOYS_ISA_DISPATCH == 0
#error "BOYS_ISA_TIER_UNIT names a carried tier and exists only in a dispatch build"
#endif

/// This unit is the dispatcher of a dispatch build.
#if defined(BOYS_ISA_DISPATCH_DEFINE) && BOYS_ISA_DISPATCH == 0
#error "BOYS_ISA_DISPATCH_DEFINE defines the dispatcher and exists only in a dispatch build"
#endif

// The token: the namespace whose name is part of every tier-varying symbol. Derived here rather
// than written by configure because one dispatch configure compiles both tiers and a configure
// step has one answer, not two.

/// The token namespace of this translation unit's tier: \c isa_avx2 or \c isa_baseline.
#if defined(BOYS_DETAIL_ISA_TOKEN)
#define BOYS_DETAIL_ISA_NAMESPACE BOYS_DETAIL_ISA_TOKEN
#elif BOYS_ISA_AVX2_FMA
#define BOYS_DETAIL_ISA_NAMESPACE isa_avx2
#else
#define BOYS_DETAIL_ISA_NAMESPACE isa_baseline
#endif

/// A function the compiler is asked not to inline: the dispatcher's tier entries, so that each
/// stays in the unit compiled at its own tier.
#if defined(_MSC_VER)
#define BOYS_NOINLINE __declspec(noinline)
#else
#define BOYS_NOINLINE __attribute__((noinline))
#endif

// The fp16 seam is the same kind of per-translation-unit choice as an ISA tier - the entries it
// gates are added to the surface, not switched at run time - so it is registered beside them
// and a program that carries both readings of it is refused for the same reason.

/// This translation unit's fp16 seam: \c 1 where the seam is open, \c 0 where it is closed.
#if BoysFp16
#define BOYS_FP16_AXIS 1
#else
#define BOYS_FP16_AXIS 0
#endif

namespace boys {

/// The tier-independent internals: the coefficient tables and the machinery every tier shares.
/// Declared before the token because the token's bodies name it; the headers that carry it add
/// their content to this namespace.
namespace detail {}

/// The token namespace, marked inline here and at every later reopening (one per header that
/// carries tier-varying bodies), so a consumer's Clang build sees no reopened-noninline warning.
inline namespace BOYS_DETAIL_ISA_NAMESPACE {

/// The tier-varying bodies: the kernel implementations, the lanes and the packed entries. The
/// public templates sit in the token itself; this inner namespace keeps the body names out of the
/// public lookup path while keeping their symbols token-qualified.
namespace kernel {

/// These two namespaces were one before the tier split: a body moved into the token still writes
/// the tables' names unqualified, exactly as it did when both lived in `boys::detail`. One
/// directive restores that spelling; it does not merge the symbols, which is the whole point of
/// the split.
using namespace detail;

} // namespace kernel

} // namespace BOYS_DETAIL_ISA_NAMESPACE

namespace detail {

/// The internal spelling the bodies and the tests already use: `boys::detail::X` names this
/// translation unit's own tier body, and, when no tier body of that name exists, the
/// tier-independent tables that live here. One directive, so neither spelling has to change at
/// the call site.
using namespace BOYS_DETAIL_ISA_NAMESPACE::kernel;

} // namespace detail

namespace dispatch {

/// The tiers a dispatcher selects among. The values are stable: the registry's carried set is a
/// bit per value.
enum IsaTier {
    kIsaBaseline = 0, ///< the portable scalar tier
    kIsaAvx2 = 1      ///< the AVX2 + FMA tier
};

} // namespace dispatch

} // namespace boys

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
///
/// \returns \c true when the processor carries AVX2, FMA and the OS state that enables them
inline bool Avx2Available() noexcept {
    static const bool available = DetectAvx2();
    return available;
}

/// The registry of the tiers a program's translation units announced, and the refusal when they
/// disagree. Not API: a consumer reads the contract at the top of this header and never names
/// anything here.
namespace tier_guard {

/// What a translation unit is, so that the registry can apply the right rule to it.
enum TierRole {
    kRoleClient = 0,     ///< an ordinary unit, including the library's own non-tier units
    kRoleCarried = 1,    ///< one carried tier of a dispatch build
    kRoleDispatcher = 2  ///< the dispatcher itself
};

/// One program's announced tiers, one field per axis. A field left at kUnannounced has heard
/// from no translation unit yet.
struct Registry {
    static constexpr int kUnannounced = -1; ///< the value before any translation unit has announced

    int isa = kUnannounced; ///< BOYS_ISA_AVX2_FMA, as a translation unit announced it
    int fp16 = kUnannounced; ///< the fp16 seam, as a translation unit announced it
#if BOYS_ISA_DISPATCH
    int carried = 0; ///< the carried tiers, one bit per dispatch::IsaTier
    bool dispatcher = false; ///< the dispatcher unit has announced itself
#endif
};

/// The program's one registry: inline, so every translation unit and every tier of one image
/// reads and writes the same object. Constant-initialised, so no announcement can run ahead of
/// its construction.
///
/// This is the one first-party object the two tier object sets both define, and it is meant to
/// be: a program has one registry, and each tier announces itself to it.
inline Registry kTierRegistry;

} // namespace tier_guard

} // namespace boys::detail

namespace boys {

inline namespace BOYS_DETAIL_ISA_NAMESPACE {

namespace kernel {

/// The guard, compiled by each translation unit at its own flags. The functions are not a tier
/// choice - they read and write the one registry above - but they are inline in this header: a
/// program that includes only the headers links the announcement it makes and has no library
/// unit to call. The token keeps each tier's copies distinct symbols, so the linker never
/// chooses between two tier objects; the one object they all name is `kTierRegistry`, outside
/// the token.
namespace tier_guard {

/// The registry and its type, named here so the bodies below read them unqualified.
using ::boys::detail::tier_guard::kTierRegistry;
/// The announced-tier registry type, imported so the signatures below name it unqualified.
using ::boys::detail::tier_guard::Registry;

/// The program's one registry.
///
/// \returns the registry every translation unit and every tier of this image shares
inline Registry& Announced() noexcept {
    return kTierRegistry;
}

/// The AVX2 + FMA tier's profile as its own flag set spells it.
///
/// \param tier the announced \c BOYS_ISA_AVX2_FMA value
///
/// \returns a string literal naming that flag set, for the refusal's own line
inline const char* IsaProfile(int tier) noexcept {
    return tier != 0 ? "AVX2+FMA (BOYS_ISA_AVX2_FMA=1)" : "scalar (BOYS_ISA_AVX2_FMA=0)";
}

/// The fp16 seam's profile as its own flag set spells it.
///
/// \param seam the announced fp16 seam value
///
/// \returns a string literal naming that flag set, for the refusal's own line
inline const char* Fp16Profile(int seam) noexcept {
    return seam != 0 ? "the fp16 seam open (BoysFp16=1)" : "the fp16 seam closed (BoysFp16=0)";
}

/// Refuses a program whose translation units were not compiled alike: one line naming the two
/// profiles and the axis they disagree on, then an abort.
///
/// \param axis   the axis the two translation units disagree on
/// \param mine   the profile this translation unit announced
/// \param theirs the profile an earlier one announced
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

#if BOYS_ISA_DISPATCH
/// Refuses a client unit compiled at a non-baseline tier in a dispatch build: the vector tier is
/// the library's, and a client that carried it would put vector code into its own policy
/// instantiations.
///
/// \param tier the tier the client announced
[[noreturn]] inline void RefusedClientTier(int tier) noexcept {
    std::fprintf(stderr,
                 "boys: this translation unit was compiled at %s in a dispatch build, where every "
                 "unit that is not one of the library's tier units must be compiled at %s; the "
                 "vector tier is selected at run time by the library's dispatcher\n",
                 IsaProfile(tier), IsaProfile(0));
    std::fflush(stderr);
    std::abort();
}

/// Refuses a dispatcher that ran before its own unit announced itself: the selection must not
/// run in an image that does not contain the dispatcher.
[[noreturn]] inline void NoDispatcher() noexcept {
    std::fprintf(stderr,
                 "boys: a default entry ran before the dispatcher was registered in this image; "
                 "the dispatcher unit (boys_dispatch.cpp) is missing or its initialisation did "
                 "not run\n");
    std::fflush(stderr);
    std::abort();
}

/// Refuses a selected tier that this image does not carry: the selection and the carried set
/// disagree, which is the failure a missing or mis-flagged tier unit produces.
///
/// \param selected the tier the dispatcher selected
/// \param carried  the carried set, one bit per dispatch::IsaTier
[[noreturn]] inline void SelectedNotCarried(int selected, int carried) noexcept {
    const bool baseline = (carried & (1 << dispatch::kIsaBaseline)) != 0;
    const bool avx2 = (carried & (1 << dispatch::kIsaAvx2)) != 0;
    std::fprintf(stderr,
                 "boys: the dispatcher selected %s, which this image does not carry (carried: "
                 "%s%s%s); every carried tier must be linked from its own tier unit\n",
                 IsaProfile(selected), baseline ? "scalar" : "",
                 (baseline && avx2) ? " and " : "", avx2 ? "AVX2+FMA" : "");
    std::fflush(stderr);
    std::abort();
}
#endif

/// Announces this translation unit's two axes, refusing a program that announced both readings
/// of either. Announcing a value that is already registered is the common case - a program's
/// translation units are compiled alike - and costs a comparison.
///
/// With dispatch on the isa rule is the role's: a carried tier records itself, the dispatcher
/// records itself, and a client must be at the baseline tier. The fp16 axis is one seam across
/// the program either way.
///
/// \param isa  this unit's \c BOYS_ISA_AVX2_FMA value
/// \param fp16 this unit's fp16 seam value
/// \param role what this unit is
inline void RegisterTier(int isa, int fp16, int role) noexcept {
    Registry& registry = Announced();

#if BOYS_ISA_DISPATCH
    if (role == ::boys::detail::tier_guard::kRoleCarried)
    {
        registry.carried |= 1 << isa;
    } else if (role == ::boys::detail::tier_guard::kRoleDispatcher)
    {
        if (isa != 0)
        {
            RefusedClientTier(isa);
        }
        registry.dispatcher = true;
    } else if (isa != 0)
    {
        RefusedClientTier(isa);
    }
    static_cast<void>(registry.isa);
#else
    static_cast<void>(role);

    if (registry.isa == Registry::kUnannounced)
    {
        registry.isa = isa;
    } else if (registry.isa != isa)
    {
        MixedTiers("the instruction-set tier", IsaProfile(isa), IsaProfile(registry.isa));
    }
#endif

    if (registry.fp16 == Registry::kUnannounced)
    {
        registry.fp16 = fp16;
    } else if (registry.fp16 != fp16)
    {
        MixedTiers("the fp16 seam", Fp16Profile(fp16), Fp16Profile(registry.fp16));
    }
}

} // namespace tier_guard

} // namespace kernel

} // namespace BOYS_DETAIL_ISA_NAMESPACE

} // namespace boys

namespace boys::detail {

namespace tier_guard {

/// \cond
/// The spelling the rest of the library already uses: `detail::tier_guard::RegisterTier` and the
/// refusals name this translation unit's copy, the one its own flags compiled. These
/// declarations add no definition; they make that spelling reach the token-qualified one.
using ::boys::BOYS_DETAIL_ISA_NAMESPACE::kernel::tier_guard::Announced;
using ::boys::BOYS_DETAIL_ISA_NAMESPACE::kernel::tier_guard::RegisterTier;
#if BOYS_ISA_DISPATCH
using ::boys::BOYS_DETAIL_ISA_NAMESPACE::kernel::tier_guard::NoDispatcher;
using ::boys::BOYS_DETAIL_ISA_NAMESPACE::kernel::tier_guard::SelectedNotCarried;
#endif
/// \endcond

/// One translation unit's announcement, made before main: the tier this unit's flags name, in a
/// variable of its own - internal linkage, so every translation unit has one and every one of
/// them runs.
///
/// \tparam kIsaTier  \c BOYS_ISA_AVX2_FMA as this unit reads it
/// \tparam kFp16Seam \c BOYS_FP16_AXIS as this unit reads it
/// \tparam kTierRole what this unit is, in a dispatch build
template <int kIsaTier, int kFp16Seam, int kTierRole = kRoleClient>
struct TierProbe {
    /// Announces this translation unit's two axes once, before main.
    TierProbe() noexcept { RegisterTier(kIsaTier, kFp16Seam, kTierRole); }
};

/// What this translation unit is: a carried tier only where the build says so, the dispatcher
/// only in the unit that defines it, and a client everywhere else - including every unit of a
/// non-dispatch build, where the role is not read.
#if BOYS_ISA_DISPATCH && defined(BOYS_ISA_TIER_UNIT)
constexpr int kTierRole = kRoleCarried;
#elif BOYS_ISA_DISPATCH && defined(BOYS_ISA_DISPATCH_DEFINE)
constexpr int kTierRole = kRoleDispatcher;
#else
constexpr int kTierRole = kRoleClient;
#endif

/// The probe itself, one per translation unit that includes this header.
[[maybe_unused]] const TierProbe<BOYS_ISA_AVX2_FMA, BOYS_FP16_AXIS, kTierRole> kTierProbe;

} // namespace tier_guard

} // namespace boys::detail

namespace boys::dispatch {

/// The machine's cached AVX2 + FMA answer, for the dispatcher's selection.
///
/// \returns \c true when the processor carries AVX2, FMA and the OS state that enables them
inline bool CpuHasAvx2Fma() noexcept { return detail::Avx2Available(); }

#if BOYS_ISA_DISPATCH
/// The selected tier, fixed on the first call by the dispatcher. Defined in
/// \c boys_dispatch.cpp: one function-local static across the program, so the branch is a load
/// and a compare and the CPUID cache is populated once.
///
/// \returns \c kIsaAvx2 or \c kIsaBaseline
int ActiveIsa() noexcept;
#else
/// The compiled tier, the only one this program carries.
///
/// \returns \c kIsaAvx2 or \c kIsaBaseline
constexpr int ActiveIsa() noexcept { return BOYS_ISA_AVX2_FMA; }
#endif

} // namespace boys::dispatch
