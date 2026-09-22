# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys)](https://github.com/myamlak/boys/tags)

Self-contained C++23 evaluation of the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32 — scalar fp64/fp32
lanes, an AVX2 vector tier reached through the same entries, fp16/bf16 I/O wrappers, and
optional CUDA kernels. The library depends on nothing outside the C++
standard library, the optional CUDA toolkit, and the committed generated
tables. Every lane is validated against a committed 45-digit reference
grid, reproduced from the cited formulas.

**Full API documentation: <https://myamlak.github.io/boys/>**

## Quick start

    git clone https://github.com/myamlak/boys.git
    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

```cpp
#include <boys/boys.hpp>

int main()
{
    double f0 = boys::BoysSingle(0, 0.5);  // F_0(0.5)

    std::array<double, 8> out;
    boys::BoysAllOrders(7, 1.25, out.data());  // F_0..F_7(1.25)

    // Many arguments, all orders: the batch shape an integral engine needs.
    // Arguments may arrive in any order; the entry classifies, groups and
    // dispatches internally. out[k * count + i] = F_k(x[i]).
    const double x[3] = {0.5, 12.5, 30.0};
    std::array<double, 3 * 33> planes;
    boys::BoysAllN(32, x, planes.data(), 3);

    // Already non-decreasing? Say so and skip the sort.
    boys::BoysAllN(32, x, planes.data(), 3, boys::BoysSortedArgs{});
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
| native half | — | — | ≤ 8 ULP of the returned value |
| CUDA fp64 | same m·budgets as the CPU double lanes, verified directly against the reference grid | |
| CUDA fp32 | same m·budgets as the CPU float lanes; GPU-vs-CPU cross-lane budget 3.5e-7 | |

The last of those is the only lane that does not round once per value. The
fp16/bf16 row is fp16 *I/O* around the fp32 engine, so its error is the
engine's; the native half lane (`BoysAllOrdersHalf2`, `BoysAllNF16Native`)
evaluates region C's ladder in packed binary16, one correctly rounded
operation per step, which is what buys two arguments to a register and costs
the accuracy its own bound names: ≤ 8 ULP of the returned value, measured at
a worst of 4.3 ULP. Its output is 2^15 F_k(x) — the exact scale that keeps
the ladder in the format's normal range down to F_k(x) = 2^-29, which is x up
to 359 / 128 / 30 at orders 3 / 4 / 8. **That is the ceiling of the claim:**
past those arguments the entry returns a subnormal half and then a zero by
design, and no accuracy is claimed there, so a caller that has to be right at
those orders and arguments wants the double or float lane. It is a region-C
entry: x must be at or above the region-C boundary, and there is no fallback
to the table regions.

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

Requires CMake (>= 3.25), git, and a C++23 compiler. Built and tested on
MSVC (Visual Studio 2022 17.x and 2026), GCC 13+, Clang 17+ and AppleClang,
on x86_64 and on arm64; the leg-by-leg record is the "Supported platforms"
table below. Optional builds: `-DBUILD_BENCHMARKS=ON` (CPU benchmark
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
[CONTRIBUTING.md](CONTRIBUTING.md) for the details.

## Supported platforms

Every row below is a CI leg that runs on every push to `main` and every pull
request, with the architecture its own dispatch assertion holds it to (see
`tests/boys_avx2_probe.cpp`: the AVX2 tier is present on x86_64 and absent
elsewhere, where the region entry points dispatch to the scalar lanes). The
table is generated from `.github/workflows/ci.yml` and
`.github/required-checks.txt` by `tools/gen_platform_table.py`, which the CI
legs re-check, so it cannot drift from what actually runs.

"Required" means branch protection on `main` will not merge a pull request
until that check is green. **Every leg that runs is required** — there is no
recorded-but-unenforced leg any more. `linux-arm64 gcc Release` was excluded
while its boundary pin was red: the cells were a recording-machine lattice draw
of an oscillating error envelope, and glibc-aarch64 drew them past the recorded
±20% band (kmax = 4 landed 21.9% off). The pin was replaced by a stable
measurement — the failure top of a descending sweep at a stated resolution —
which spreads 5.1% across gcc, clang and MSVC where the retired draw spread
27.0% from the lattice alone on one machine. The leg went green with the rest.

<!-- platform-table:begin (generated by tools/gen_platform_table.py; do not edit) -->
| CI leg (the check name) | Runner | AVX2 tier | Required |
|---|---|---|---|
| `windows-msvc Release` | `windows-latest` | present | yes |
| `windows-msvc Debug` | `windows-latest` | present | yes |
| `linux-x86 gcc Release` | `ubuntu-latest` | n/a | yes |
| `linux-x86 gcc Debug` | `ubuntu-latest` | n/a | yes |
| `linux-x86 clang Release` | `ubuntu-latest` | n/a | yes |
| `linux-x86 clang Debug` | `ubuntu-latest` | n/a | yes |
| `linux-sanitizers asan+ubsan` | `ubuntu-latest` | n/a | yes |
| `linux-arm64 gcc Release` | `ubuntu-24.04-arm` | n/a | yes |
| `windows-arm64 msvc Release` | `windows-11-arm` | n/a | yes |
| `macos arm64` | `macos-latest` | absent | yes |
| `macos x64` | `macos-15-intel` | present | yes |
| `linux-x86 clang-tidy` | `ubuntu-latest` | n/a | yes |
<!-- platform-table:end -->

## Reproducibility

- `tests/data/boys_reference.csv` — the committed 45-digit reference grid.
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev tables
  and the grid from the cited formulas (public `mpmath`, pinned in
  `requirements-boys.txt`); `--check` is the byte-identity proof, and
  every CI leg runs the full regeneration protocol. The emitted header is
  clang-format output, so `requirements-boys.txt` pins the formatter too and
  the script reads `$CLANG_FORMAT` before searching `PATH` — without a
  formatter it fails rather than writing bytes the gate would report as drift.
- The public history is a single squashed commit carrying this release's
  tree, tagged `v1.1.4`; earlier release tags are no longer published.

## Citation

The published works the numerical claims rest on are listed in
[CITATION.bib](CITATION.bib).

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin
Makowski`. Dependency details in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
