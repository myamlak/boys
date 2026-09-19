// The dispatch assertion, built by every CI leg (CMake option BOYS_EXPECT_AVX2)
// and run as the boys-avx2-probe ctest case.
//
// It asserts the capability PREDICATE, not which path a given call took: each
// leg states what BoysAvx2Available() MUST report for its architecture — true
// on the x86_64 legs, false on the arm64 legs — and this program fails the leg
// when it disagrees. That is worth more than the rest of the matrix combined,
// because the SIMD correctness tests in boys_test.cpp gate themselves with a
// soft GTEST_SKIP on the same predicate: a wrong CPUID bit therefore makes
// every SIMD test SKIP while CI reports green with the whole vector tier dead,
// which is precisely the failure this repository already suffered once. A hard
// assertion turns that silent skip into a red leg.
//
// On the arm64 legs the false is not a formality either: it carries the
// no-architecture-guard finding that added those legs (an x86-only TU and flag
// set in a tree that claimed none), and it is the fact that separates "the
// vector tier is absent on this target" from "the vector tier is silently
// broken on this target".
//
// The residual gap, named rather than papered over: this observes the
// predicate, not the dispatch. An entry point that ignored the predicate and
// ran a vector body on a machine reporting false would still be reported
// available by this probe only if the predicate itself were wrong.

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
