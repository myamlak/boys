# boysymmetriad

A self-contained C++23 implementation of the Boys function family
F_n(x) for n = 0..32, released with its paper
**[boysymmetriad evaluation paper — arXiv ID / DOI pending]**.

The library ships scalar fp64/fp32 lanes, an AVX2 region-sorted SIMD
lane, fp16/bf16 wrappers, a C linkage surface, and optional CUDA
kernels. Every lane is validated against a committed 30-digit reference
grid, and everything in the library depends on nothing outside the C++
standard library, the CUDA toolkit (optional), and the committed
generated tables.

## What this is

The Boys function

    F_n(x) = ∫₀¹ t^(2n) exp(−x t²) dt

is the fundamental building block of Gaussian-basis quantum chemistry:
every Gaussian integral recursion — McMurchie–Davidson, Obara–Saika, Rys —
bottoms out in it. This library evaluates it in the standard three-region
scheme, with the regions chosen so that each evaluation form is stable
and division-free:

- **Region A [0, x0):** per-order piecewise Chebyshev fits in the split
  Clenshaw form, weighted against downward-recursion seed-error
  amplification.
- **Region B [x0, x1):** one F0 fit plus upward recursion.
- **Region C [x1, ∞):** asymptotic series.

Region boundaries for the shipped `kMaxBoysOrder = 32` configuration:
x0 = 11.899848152108484, x1 = 28.989337738820740.

**Portability.** The region boundaries above are calibration constants of
this release, not mathematical definitions: they are the fixed values of
the three-region scheme (published by Vikhamar-Sandberg and Repisky,
arXiv:2512.10059), grid-validated on the reference build behind this
library's contract. The per-region accuracy bounds in the contract table
below are asserted over the whole domain and are not affected by
toolchain or platform. What is machine-sensitive is the last-ulp detail
at the boundaries themselves: the genuinely measured stability
thresholds of the evaluation forms are lattice draws of an oscillating
error envelope whose recorded positions move across libms within roughly
10–20%. The paper reports the reference-machine values; the boundary
harness (`tests/boys_boundary_standalone.cpp`) ships with the tests and
re-measures them on any machine.

**Coverage margin.** kMaxBoysOrder = 32 covers every order an l = 6 shell
quartet can demand (n = 24 for a (6,6,6,6) ERI) plus the gradient (n + 1)
and Hessian (n + 2) lifts, with margin — this width is the paper's
validated configuration, and the generator reproduces all of it from the
cited formulas.

## API overview

All evaluation entry points live in `<boysymmetriad/boys.hpp>` in namespace
`boysymmetriad`; the fp16/bf16 I/O types live in `<boysymmetriad/f16.hpp>`,
the C linkage surface in `<boysymmetriad/boys_c.h>` (C-clean: callable from
C, Fortran via ISO_C_BINDING, and other FFI consumers), and the CUDA lane
in `<boysymmetriad/boys_cuda.hpp>`.

| Lane | Entry points | Notes |
|---|---|---|
| scalar fp64 | `BoysSingle`, `BoysBatch` | portable C++23, no architecture flags |
| scalar fp32 | `BoysSingleF32`, `BoysBatchF32` | the batch form seeds in double precision |
| fp16 / bf16 scalar | `BoysSingleF16`, `BoysBatchF16` (+ `Bf16` twins) | I/O-only around the certified fp32 engine; behind the `BoysFp16` seam |
| AVX2 region-sorted fp64 | `BoysRegionASimd`, `BoysRegionBSimd`, `BoysRegionCSimd` | arrays must be pre-partitioned by region; AVX2 per-TU compile flags; gate on `BoysAvx2Available()` |
| fp16 / bf16 AVX2 region | `BoysRegionASimdF16`, `BoysRegionBSimdF16`, `BoysRegionCSimdF16` (+ `Bf16` twins) | 8-lane F16C load/store |
| CUDA (optional build) | `BoysCuda::SingleF32`/`BatchF32`/`SingleF64`/`BatchF64`/`SingleF16`/`BatchF16` (+ `InitializeTables`) | raw device pointers + a stream; see below |
| C surface | `BoysDouble`, `BoysFloat`, `BoysDoubleBatch`, `BoysFloatBatch` (+ multiplier variants) | C-clean, integer status returns; see below |

**CPU entries** take `(int n, double x)` for the scalar forms and
`(int nmax, double x, double* out)` for the batch forms (F_0(x)…F_nmax(x),
lowest order first; an output span of `nmax + 1` values). All CPU entries
are `noexcept` and total: preconditions are n in [0, 32], x >= 0, and
output spans of the documented size; violations are UB by contract
(documented per entry in the headers). The CUDA lane is the one fallible
C++ surface — every entry returns a `BoysStatus` (`kSuccess`,
`kInvalidArgument`, `kDeviceError`), never exceptions.

**CUDA lane.** `BoysCuda` entry points take raw device pointers and a
stream (`cudaStream_t`, passed as `void*`; `nullptr` selects the default
stream) — the caller owns device memory and scheduling. The coefficient
tables are uploaded on first use (idempotent per device), and all entries
are asynchronous: the kernel is queued on the caller's stream and the
call returns once the launch is accepted, so the caller synchronizes the
stream before reading the outputs. Error detail behind `kDeviceError`
comes from the CUDA runtime's own reporting (`cudaGetLastError`). The
fp16 entries take raw device pointers to this library's `F16` type, whose
host representation is byte-compatible with IEEE-754 binary16.

**The accuracy-multiplier surface.** Every evaluation entry of the CPU
and CUDA lanes is templated on a single compile-time accuracy multiplier:

    template <double kAccuracyMultiplier = kBoysFullAccuracyMultiplier>

with `kBoysFullAccuracyMultiplier = 1.0` the default of every entry. The
contract is |F̂_n(x) − F_n(x)| ≤ m·B_region per lane and region, m the
multiplier of the call (see the table in the next section):

- **m = 1 is bit-identical to the certified lanes** (the contract of the
  accompanying paper): the
  default instantiation selects the full-accuracy bodies verbatim — no
  branch, indirection, or runtime dispatch on the m = 1 path, and existing
  call sites compile unchanged.
- **m > 1 relaxes the bound to m·B_region via compile-time Chebyshev degree
  truncation**: the seed fits shrink to the certified effective degree
  d'(m), and the delivered error follows from the m = 1 asserted bound plus
  a dropped-coefficient tail bound — a-priori, never tuned. The work is
  monotone in m; region C has no relaxable resource and its contract holds
  at every m.
- **m < 1 is a compile-time error.**

Instantiate a relaxed entry as e.g. `BoysSingle<2.0>(n, x)` — the same
entry points, one template argument. The library exports only the m = 1
instantiations; relaxed instantiations compile from the internal headers,
which the test suite exercises at the sampled set {1.0, 2.0, 10.0, 100.0,
1e4, 1e8}.

**C surface.** `<boysymmetriad/boys_c.h>` is C-clean (no exceptions, no
templates, no C++ headers). Every entry returns an `int` status —
`BOYS_SUCCESS`, `BOYS_ERROR_INVALID_ARGUMENT`, or
`BOYS_ERROR_UNSUPPORTED_MULTIPLIER` — and writes results through pointer
arguments; `BoysDouble(n, x, &out)` is `F_n(x)` in double precision, and
`BoysDoubleBatch(nmax, count, x, out)` fills `out[k * count + i] =
F_k(x[i])` (order-major, matching the C++ batch lanes). The multiplier
entries (`BoysDoubleWithMultiplier`, `BoysFloatWithMultiplier`) dispatch on
m by exact equality over the sampled set above. This is the surface for
Fortran consumers through ISO_C_BINDING and for ctypes/Rust/Julia FFI.

**fp16/bf16 I/O contract.** The half lanes are I/O-only: F16/Bf16 values
enter and leave the kernel while the computation runs around the certified
fp32 engine, so the lane delivers the fp32 engine's certified bound up to
one half-ULP of fp16 representation. `F16`/`Bf16` alias
`std::float16_t`/`std::bfloat16_t` where the toolchain ships them (GCC 13+,
Clang 17+); the MSVC STL does not, so the headers supply self-contained
wrapper classes there — code written against the `boysymmetriad` aliases is
portable across both.

**A minimal example** (scalar form, then the batch form):

    #include <boysymmetriad/boys.hpp>

    #include <array>

    int main()
    {
        double f0 = boysymmetriad::BoysSingle(0, 0.5);  // F_0(0.5)

        std::array<double, 8> out;
        boysymmetriad::BoysBatch(7, 1.25, out.data());  // F_0..F_7(1.25)

        return f0 > 0.0 && out[7] > 0.0 ? 0 : 1;
    }

and the same through the C surface:

    #include <boysymmetriad/boys_c.h>

    int main()
    {
        double f0 = 0.0;
        if (BoysDouble(0, 0.5, &f0) != BOYS_SUCCESS) return 1;
        return f0 > 0.0 ? 0 : 1;
    }

## Accuracy contract

Every claim below is validated per-order against the committed 30-digit
mpmath reference grid (`tests/data/boys_reference.csv`), reproduced from
the cited formulas. The grid covers x = 0, 1e-12, log-spaced 1e-8..100,
and both region boundaries.

| Lane | region A (x < x0) | region B (x0 ≤ x < x1) | region C (x ≥ x1) |
|---|---|---|---|
| double single | ≤ m·1e-15 | ≤ m·3e-14 | ≤ m·5.5e-14 |
| double batch | ≤ m·5.5e-14 | ≤ m·5.5e-14 | ≤ m·5.5e-14 |
| float single / batch | ≤ m·1.5e-7 | ≤ m·1.5e-7 | ≤ m·1.5e-7 |
| fp16 / bf16 | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP | ≤ m·1e-7 + ½ ULP |
| CUDA fp64 / fp32 | the same m·budgets as the CPU lanes (GPU-vs-CPU cross-lane agreement within ~3.5e-7) |

with m the accuracy multiplier of the call (m = 1 by default; see the
section above). The fp32 budget is an a-priori bound, not a tuning
artifact: the fit tolerances that meet it are derived by the generator
from the recursion's error-amplification analysis.

## Building and testing

Requires only CMake (>= 3.25), a C++23 compiler (MSVC, GCC, or Clang),
git, and — for the byte-identity gate — Python 3 with the pinned `mpmath`
(`pip install -r requirements-boys.txt`). No vcpkg, no network at
configure time, no FetchContent — submodules are pinned and cloned
explicitly.

    git clone https://github.com/[owner]/boysymmetriad.git --recurse-submodules
    cmake -S . -B build
    cmake --build build
    ctest --test-dir build --output-on-failure

The test suite covers the accuracy contract per order, the region
boundaries, the multiplier surface at the sampled set, the C surface, the
fp16/bf16 I/O contract, and the BOUNDARY-CHECK standalone
(`boys-boundary-standalone`, reproducing the paper's measured
recursion-boundary cells).

Optional: `-DBUILD_BENCHMARKS=ON` builds the CPU benchmark drivers (the
paper's Table 2 source; default ON locally — the timed runs themselves are
local-only by design and never part of CI). `-DBUILD_CUDA=ON` additionally
builds the CUDA lane and its tests (needs the CUDA toolkit); CUDA is a
documented local gate — CI has no GPU leg and never builds or runs it.

## Reproducibility

- `tests/data/boys_reference.csv` — the 30-digit reference grid,
  committed to the repo.
- `tools/gen_boys_coefficients.py` — regenerates the Chebyshev
  coefficient tables and the reference grid from the cited formulas
  (requires only the public `mpmath` package, pinned in
  `requirements-boys.txt`).
- Every CI leg runs the full regeneration protocol: the in-place
  regeneration run, `--check`, and `git diff --exit-code` on the two
  generated files — a drift fails the leg. Locally, `--check` alone is the
  identity proof:
  `python3 tools/gen_boys_coefficients.py --check` (or
  `cmake --build build --target boys-generator-check`).
- The benchmark drivers record the machine, clocks, and commit in their
  runs logs, per the paper's methodology section.

## Contributing

See `CONTRIBUTING.md` at the repo root — build instructions, style rules,
test conventions, and the comment policy.

## Citation

If you use this library, please cite:

    [paper citation — arXiv ID / DOI pending]

The implementation, tests, benchmarks, and generator accompany that
paper's "Code availability" declaration.

## License

BSD-3-Clause — see [LICENSE](LICENSE), `Copyright (c) 2026 Marcin
Makowski`.

## Dependencies

| Component | Version | Purpose | License |
|---|---|---|---|
| GoogleTest | 1.17.0 (pinned submodule) | test harness | BSD-3-Clause |
| Google Benchmark | 1.9.5 (pinned submodule, optional) | CPU benchmark drivers | Apache-2.0 |
| mpmath (Python) | 1.4.1 (pinned in requirements-boys.txt) | build-time-only table/reference regeneration | BSD-3-Clause |
| CUDA toolkit | optional (`-DBUILD_CUDA=ON`, local gate) | CUDA lane | NVIDIA toolkit license (the library's CUDA code is ours) |

No other dependencies. Third-party details in THIRD_PARTY_NOTICES.md.

## Not included (by design)

This library is the boysymmetriad kernel only: the scalar and batch lanes,
the SIMD and fp16/bf16 lanes, the C surface, the CUDA kernels, the
benchmark drivers, and the generator behind the committed tables. It is
the citable, reproducible artifact behind the paper, cut from the exact
revision the paper's numbers were measured against; the surrounding
quantum-chemistry machinery (integral engines, screening, Fock
construction) is out of scope here.
