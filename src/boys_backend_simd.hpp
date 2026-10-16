#pragma once

// The packed arithmetic backends of the AVX2 + FMA tier now live on the installed include path,
// with the lanes that name them: a consumer's own translation unit compiles the same backend the
// library's does. This name is kept for the units and tests that already spell it.

#include "boys/boys_backend_simd.hpp"
