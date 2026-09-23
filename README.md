# The Boys function kernel

[![CI](https://github.com/myamlak/boys/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/myamlak/boys/actions/workflows/ci.yml)
[![docs](https://github.com/myamlak/boys/actions/workflows/docs.yml/badge.svg)](https://github.com/myamlak/boys/actions/workflows/docs.yml)
[![license](https://img.shields.io/badge/license-BSD--3--Clause-blue)](LICENSE)
[![version](https://img.shields.io/github/v/tag/myamlak/boys)](https://github.com/myamlak/boys/tags)

Self-contained C++23 evaluation of the Boys function family
F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt for n = 0..32.

It ships scalar fp64 and fp32 lanes, an AVX2 vector tier reached through the same entries, fp16 and
bf16 I/O wrappers, a native packed-half lane, a matrix-product entry for region A that can run on
tensor cores, and optional CUDA kernels. It depends on nothing outside the C++ standard library, the
optional CUDA toolkit, and the committed generated tables. Every lane is validated against a
committed 45-digit reference grid.

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

The bound is |F̂_n(x) − F_n(x)| ≤ m·B, for every supported n, x and lane. Here m is the compile-time
accuracy multiplier of the call, and defaults to 1.

The library evaluates the function differently at different argument sizes, and one lane is tighter
over part of the range than over the rest. Every figure below holds for **all** x ≥ 0:

| Lane | Error bound |
|---|---|
| double single | ≤ m·5.5e-14 everywhere; ≤ m·3e-14 below x = 11.899848152108484; ≤ m·1e-15 below about x = 1.0855 |
| double batch | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP |
| native half, x ≥ 28.984375 | ≤ 8 ULP of the returned value |
| CUDA fp64 | same m·budgets as the CPU double lanes |
| CUDA fp32 | same m·budgets as the CPU float lanes; GPU-vs-CPU cross-lane budget 3.5e-7 |

"ULP" is the last representable digit of the result in the format concerned. On the C++ surface the
multiplier is any value at or above 1, with no upper end, and raising it loosens the bound and
reduces the work. The C surface takes a sampled set instead, listed in its own header.

The fp16 and bf16 rows are fp16 *I/O* around the fp32 engine, so their error is the engine's.

The native half lane (`BoysAllOrdersHalf2`, `BoysAllNF16Native`) is a different thing. It evaluates
region C's ladder in packed binary16, one correctly rounded operation per step. That buys two
arguments to a register. It costs the accuracy its own bound names: ≤ 8 ULP of the returned value,
measured at a worst of 4.3 ULP.

Its output is 2^15 F_k(x). That exact scale keeps the ladder in the format's normal range down to
F_k(x) = 2^-29, which is x up to 359, 128 and 30 at orders 3, 4 and 8. **Past those arguments the
entry returns a subnormal half, then zero, by design, and no accuracy is claimed there.** A caller
that must be right at those orders and arguments wants the double or float lane. The entry covers
region C only, so x must be at or above the region-C boundary, and there is no fallback to the table
regions.

Its range shrinks steeply with the order, and above order 8 it has none at all. **Orders 9 to 32 have
no argument at which this lane returns a value inside its bound**, because the function is already
below 2^-29 at the smallest argument the entry accepts. Orders 0 and 1 cover the whole 16-bit range,
and orders 2 to 8 reach x of about 2636, 360, 129, 70, 47, 36 and 30.

The **region-A transform** (`BoysRegionAProduct`) is a separate entry rather than a lane. It computes
region A's fits as a matrix product instead of by the fitted recurrence — one product per band, all
orders at once, for a batch of arguments — in an arithmetic mode the caller names: `ProductMode::kFp64`,
`kTf32x3` or `kBf16x6`. It changes none of the lanes above. At fp64 it adds nothing measurable to the
coefficients' own truncation, 1.11e-16 over the band. The two split modes deliver 1.92e-07 and
1.95e-07, which is their 32-bit accumulator's floor rather than the operand split's: no amount of
operand precision reaches 64-bit-grade accuracy on a 32-bit accumulator. **Those two bounds are
arithmetic on a model of a 32-bit tensor-core accumulator, not a measurement of a card** — a tensor
core's fused sum is less accurate than the model, never more, and **no speed is claimed for any mode**.
A caller who seeds the downward recursion rather than reading values is held to a tighter
requirement: at fp64 only orders 0 to 2 and 25 to 32 may seed, and the two split modes may not seed
at all.

Public function signatures and supported domains are stable within a major version. Bitwise outputs
are not. Internal region thresholds, seed selection, recursion order and dispatch logic may change
between minor releases, as long as the bounds above hold. Byte-for-byte reproducibility requires
pinning the release tag, the compiler and the build flags.

**Reporting a suspected violation.** Open an issue with the exact `(n, x)`, the lane and the region.
Reproduce it first with `ctest --test-dir build --output-on-failure`. If you suspect the generated
tables, add `python3 tools/gen_boys_coefficients.py --check`.

## Building and consuming

Requires CMake (>= 3.25), git, and a C++23 compiler. The leg-by-leg record of what is built and
tested is the "Supported platforms" table below.

Optional builds: `-DBUILD_BENCHMARKS=ON` for the CPU benchmark drivers, which are local-only by
design, and `-DBUILD_CUDA=ON`, which needs the CUDA toolkit.

As a submodule:

    git submodule add https://github.com/myamlak/boys external/boys

```cmake
add_subdirectory(external/boys boys)
target_link_libraries(app PRIVATE boys::boys)
```

The tree sets no global standard or flags. Tests and benchmarks are OFF unless this tree is the
top-level project. See the API documentation and [CONTRIBUTING.md](CONTRIBUTING.md) for the details.

For what each lane guarantees, where it stops and why, see
[docs/lane-contract.md](docs/lane-contract.md).

## Supported platforms

Every row below is a CI leg that runs on every push to `main` and every pull request. The
architecture column is what that leg's dispatch assertion holds it to: the AVX2 tier is present on
x86_64 and absent everywhere else, where the entry points dispatch to the scalar lanes.

The table is generated from `.github/workflows/ci.yml` and `.github/required-checks.txt` by
`tools/gen_platform_table.py`. The CI legs re-check it, so it cannot drift from what actually runs.

"Required" means branch protection on `main` will not merge a pull request until that check is green.
Every leg that runs is required.

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
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev tables and the grid from the cited
  formulas, using a pinned public `mpmath`. Its `--check` flag is the byte-identity proof, and every
  CI leg runs the full regeneration protocol. The emitted header is clang-format output, so
  `requirements-boys.txt` pins the formatter as well. The script reads `$CLANG_FORMAT` before
  searching `PATH`. Without a formatter it fails, rather than writing bytes the gate would report as
  drift.

## Citation

The published works the numerical claims rest on are listed in [CITATION.bib](CITATION.bib).

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin Makowski`. Dependency details in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
