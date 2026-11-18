// The no-vector scalar fallback, compiled as a no-vector target compiles it: src/boys_simd.cpp
// with BOYS_SIMD_X86=0 and BOYS_ISA_AVX2_FMA=0, called over every region's precondition and every
// order, and compared bit for bit with the certified scalar lane. No library is linked: its copy
// is at the vector tier, and a baseline unit beside it would be a mixed-tier program.

#include "boys_no_vector_fallback_sweep.hpp"

#if !defined(BOYS_SIMD_X86) || BOYS_SIMD_X86 != 0
#error "this target must be compiled with BOYS_SIMD_X86=0: it is the no-vector fallback's test"
#endif

#if BOYS_ISA_AVX2_FMA != 0
#error "this target must be compiled at the scalar tier (BOYS_ISA_AVX2_FMA=0)"
#endif

#include <cstdio>

int main() {
    const boys_no_vector_fallback::Result result = boys_no_vector_fallback::Sweep();

    std::printf("no-vector scalar fallback: BOYS_SIMD_X86=0, BOYS_ISA_AVX2_FMA=0\n");
    std::printf("  entries       : RegionA/RegionB/RegionC, fp16 and bf16, every order 0..%d\n",
                boys::kMaxBoysOrder);
    std::printf("  cells compared: %zu\n", result.cells);
    std::printf("  mismatches    : %zu\n", result.mismatches);

    if (result.mismatches != 0)
    {
        std::printf("  first failure : %s\n", result.firstFailure.c_str());
        std::printf("NO-VECTOR FALLBACK FAIL\n");
        return 1;
    }

    std::printf("NO-VECTOR FALLBACK PASS\n");
    return 0;
}
