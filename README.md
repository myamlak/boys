# The Boys function kernel

Self-contained C++23 evaluation of the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32 — scalar fp64/fp32
lanes, an AVX2 region-sorted SIMD lane, fp16/bf16 I/O wrappers, and
optional CUDA kernels. The library depends on nothing outside the C++
standard library, the optional CUDA toolkit, and the committed generated
tables. Every lane is validated against a committed 30-digit reference
grid, reproduced from the cited formulas.

**Full API documentation: <https://myamlak.github.io/boys/>**

## Quick start

    git clone https://github.com/myamlak/boys.git --recurse-submodules
    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

```cpp
#include <boys/boys.hpp>

int main()
{
    double f0 = boys::BoysSingle(0, 0.5);  // F_0(0.5)

    std::array<double, 8> out;
    boys::BoysBatch(7, 1.25, out.data());  // F_0..F_7(1.25)
}
```

## Accuracy contract

The bound is |F̂_n(x) − F_n(x)| ≤ m·B_region for every supported n, x,
lane, and region, with m the compile-time accuracy multiplier of the
call (default 1; see the API documentation for the multiplier surface
and the region definitions):

| Lane | region A | extended band | region B | region C |
|---|---|---|---|---|
| double single | ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·3e-14 | ≤ m·5.5e-14 |
| double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP |
| CUDA fp64 | same m·budgets as the CPU double lanes, verified directly against the reference grid | |
| CUDA fp32 | same m·budgets as the CPU float lanes; GPU-vs-CPU cross-lane budget 3.5e-7 | |

Public function signatures and supported domains are stable within a
major version. Bitwise outputs, internal region thresholds, seed
selection, recursion order, and dispatch logic are not stable between
minor releases as long as the bound above remains satisfied; exact
bitwise reproducibility requires pinning the release tag, compiler, and
build flags.

**Reporting a suspected violation.** If you believe a lane exceeded its
documented bound, open an issue with the exact `(n, x)`, the lane, and
the region. Reproduce first with
`ctest --test-dir build --output-on-failure`; if the generated tables
are in question, add `python3 tools/gen_boys_coefficients.py --check`.

## Building and consuming

Requires CMake (>= 3.25), git, and a C++23 compiler. Tested toolchains:
MSVC (Visual Studio 2022 17.x and 2026), GCC 13+, Clang 17+; macOS
untested. Optional builds: `-DBUILD_BENCHMARKS=ON` (CPU benchmark
drivers, local-only by design) and `-DBUILD_CUDA=ON` (needs the CUDA
toolkit).

As a submodule:

    git submodule add https://github.com/myamlak/boys external/boys

```cmake
add_subdirectory(external/boys boys)
target_link_libraries(app PRIVATE boys::boys)
```

The tree sets no global standard or flags; tests and benchmarks are OFF
unless this tree is the top-level project. See the API documentation and
CONTRIBUTING.md for the details.

## Reproducibility

- `tests/data/boys_reference.csv` — the committed 30-digit reference grid.
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev tables
  and the grid from the cited formulas (public `mpmath`, pinned in
  `requirements-boys.txt`); `--check` is the byte-identity proof, and
  every CI leg runs the full regeneration protocol.
- The public history is compressed to two commits for the first public
  release (the squashed release-cut practice), tagged v1.0.0 and v1.1.0.

## Citation

Please cite the accompanying paper — see `CITATION.bib` (submitted; the
identifier will be added after acceptance).

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin
Makowski`. Dependency details in THIRD_PARTY_NOTICES.md.
