// The dispatch assertion, built by every CI leg (CMake option BOYS_EXPECT_AVX2) and run as the
// boys-avx2-probe ctest case: it fails a leg whose BoysAvx2Available() disagrees with the
// architecture (true on x86_64, false on arm64), which the SIMD tests' GTEST_SKIP would otherwise
// leave green with the vector tier dead. It observes the predicate, not the dispatch.

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
