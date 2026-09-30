// The dispatch assertion, built by every CI leg (CMake option BOYS_EXPECT_AVX2)
// and run as the boys-avx2-probe ctest case.
//
// It asserts what BoysAvx2Available() MUST report for this leg's architecture —
// true on x86_64, false on arm64 — and fails the leg when it disagrees. The SIMD
// correctness tests gate themselves on the same predicate with a soft
// GTEST_SKIP, so a wrong CPUID bit would make every SIMD test skip while CI
// reported green with the whole vector tier dead; this assertion turns that
// silent skip into a red leg.
//
// It observes the predicate, not the dispatch: an entry point that ignored the
// predicate would still pass here unless the predicate itself were wrong.

#include <boys/boys.hpp>
#include <cstdio>

#ifndef BOYS_EXPECT_AVX2
#error "BOYS_EXPECT_AVX2 must be defined by the build (1 on x86_64 legs, 0 on arm64 legs)"
#endif

int main() {
    const bool available = boys::BoysAvx2Available();
    const bool expected = (BOYS_EXPECT_AVX2 != 0);

    std::printf("BoysAvx2Available() = %s, expected %s on this leg\n",
                available ? "true" : "false",
                expected ? "true" : "false");

    if (available != expected)
    {
        std::printf("FAIL: BoysAvx2Available() disagrees with this runner.\n"
                    "On x86_64 that means the AVX2/FMA vector tier is DEAD (a wrong\n"
                    "CPUID bit) while the soft-skipping SIMD tests above stayed green.\n"
                    "On arm64 it must report NOT available; a true there means the gate\n"
                    "is reading an x86-only feature bit.\n");

        return 1;
    }

    return 0;
}
