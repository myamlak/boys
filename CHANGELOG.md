# Changelog

Notable changes to the boys library. Versioning rules live in
`CONTRIBUTING.md` (the "Versioning" section); the public numerical
contract (the `m*B_region` bound) is documented in the README.

## [1.1.4]

### Repository hygiene (no API, ABI or numerical change)

- **Affected inputs:** none. No lane, domain, budget, table or returned
  bit changes; the kernel sources differ from v1.1.3 in comments only.
- **Removed:** the `supplementary/` tree — it documented a separate
  measurement workflow, not the library's own contract.
- **Documentation:** references to unpublished work removed from the
  README, `CONTRIBUTING.md`, the public API header, the Doxygen main page,
  the benchmark drivers and the test harnesses. The measured values
  themselves are unchanged and remain the regression pins.
- **Notices:** `THIRD_PARTY_NOTICES.md` now describes GoogleTest and
  Google Benchmark accurately — they are vendored in full under
  `third_party/`, not submodules (the tree has no `.gitmodules` and no
  gitlink). The same correction is applied to the CMake, README and
  contributing prose that repeated it.
- **Tooling:** `CMakeLists.txt` declares the release version again; it had
  been left at 1.1.1 across the two preceding cuts.

## [Unreleased — the v1.1.1 cut]

### Non-x86_64 targets now build and run (the architecture guard)

- **Affected inputs:** none. This is a build/portability change; nothing
  returned changes on x86_64.
- **Old behavior:** `src/boys_simd.cpp` included `<immintrin.h>`
  unconditionally and `CMakeLists.txt` applied `-mavx2 -mfma -mf16c`
  (GCC/Clang) or `/arch:AVX2` (MSVC) with no architecture guard anywhere in
  the tree, so the library could not be configured on arm64 at all: GCC and
  Clang rejected the flags outright, and the MSVC ARM64 leg was building an
  x64 binary instead.
- **New behavior:** CMake probes the compiler at configure time
  (`BOYS_ARCH_X86_64`, `check_cxx_source_compiles`) and passes the answer into
  the SIMD translation unit as `BOYS_SIMD_X86`, so the flag set and the guard
  in that file are one decision rather than two that can drift apart. On
  x86_64 nothing changes. On any other target the flags and the intrinsics are
  compiled out, and the twelve region entry points (`BoysRegionASimd`,
  `BoysRegionBSimd`, `BoysRegionCSimd` and their F16/Bf16 forms) are defined
  against the certified scalar lanes: same signatures, same contracts, and the
  same numerics as the scalar tails the x86 entries already run for their last
  `count % 4` elements. `BoysAvx2Available()` reports `false`.
- **Contract impact:** none on x86_64. Elsewhere the region entry points'
  partitioning precondition stays *sufficient* but is no longer *required* —
  the scalar lanes accept any `x >= 0`.
- **Bitwise impact:** none. The fallback bodies are the x86 entries' own
  scalar tails applied to every element instead of the remainder.
- **Docs:** the `BoysAvx2Available` contract (`include/boys/boys.hpp`) and
  `docs/mainpage.md` (a new "Architecture" section) state the x86_64-only
  construction and the fallback.
- **CI:** the arm64 legs were added red on the first run and now build and run
  the full suite. They assert `BoysAvx2Available() == false`, which is what
  keeps "the vector tier is absent on this target" distinguishable from "the
  vector tier is silently dead on this target".

### Corrected dispatch gate (AVX2 + FMA)

- **Affected inputs:** none on a processor whose AVX2 comes with FMA
  (every shipping x86 CPU). The change is visible only on a processor that
  reports AVX2 without the FMA feature bit: `BoysAvx2Available()` now
  returns `false` there, so callers stay on the scalar lanes.
- **Old behavior:** the predicate asked OSXSAVE + AVX2 only, while the
  kernels it gates are FMA chains — the split-Clenshaw recurrences, the
  region-B upward recursion (`f = fma(l + 0.5, f, ...)`), and the
  `e^{-x}` Horner, in a translation unit compiled with `/arch:AVX2`
  (MSVC) or `-mavx2 -mfma` (GCC/Clang). A processor with AVX2 but
  without FMA was therefore dispatched into instructions it does not
  implement.
- **New behavior:** the predicate also requires the FMA feature bit
  (CPUID.01H:ECX[12]), in both the MSVC and the GCC/Clang branch.
- **Contract impact:** no documented domain, lane, or budget changes; the
  scalar lanes cover the affected processor. Where the SIMD lane is
  selected the code path is unchanged.
- **Bitwise impact:** none; no returned bit changes on hardware where the
  SIMD lane is selected.
- **Docs:** the `BoysAvx2Available` contract (`include/boys/boys.hpp`) and
  the lane table (`docs/mainpage.md`) now state the FMA requirement.

## [1.1.0]

### Numerical behavior change (the per-range seed design)

- **Affected inputs:** `x` in `[1.0855252345349333, 11.899848152108484)`
  (the extended band below the region-B floor), the double lane, `m = 1`,
  every order `n = 0..32`.
- **Old evaluation:** the region-A per-order Chebyshev fits.
- **New evaluation:** the extended-band seed (a second `F_0` fit on the
  band, degree 24, the split-Clenshaw form) plus the upward recursion,
  dispatched per order through the certified per-order threshold table
  `kTierThresholds` (the pure per-`(n, x)` rule: an order `n` takes the
  extended seed exactly when `x >= kTierThresholds[n]`, so every entry
  point returns the same value for the same order and argument). The
  `m > 1` branch and the float lanes keep their previous evaluation in
  the band.
- **Contract impact:** no documented domain or budget changes; validated
  against the committed reference grid; max observed error over the
  band's grid rows `2.1e-15`
  (measured on the corrected dispatch over the regenerated grid, which
  pins the corrected boundaries and the vacated slices' edges), not
  exceeding the `3e-14` budget the band asserts. (Honesty note: the
  band's certificate is the range-uniform a-priori seed bound
  `delta_0' = 1.0527e-15` --- the truncation tail `4.112e-17` plus the
  split-Clenshaw forward rounding bound `1.0116e-15`, no x-sampling ---
  not the measured width, which enters only as a consistency check; see
  the Boundary-table entry.)
- **Bitwise impact:** not bitwise-identical to v1.0.0 inside the band
  (up to the asserted budget; well below it in practice);
  hash-pipeline consumers must regenerate baselines.
- **Boundary table:** the certified per-`k_max` recursion boundaries
  changed from the honest-extension table
  `11.8372 / 11.8372 / 11.8372 / 11.8464` (the shipped seed alone, the
  `k_max = 4/8/16/32` rows) to the per-range design's certified rows
  `1.0855 / 2.0136 / 4.8960 / 10.7818` (1-ulp-exp form), wired into the
  kernel as the dispatch thresholds. (Honesty note: the initial v1.1
  increment carried the empirical-bound crossings
  `1.8575 / 4.7031 / 10.6551` --- the envelope crossings under the
  per-argument measured interval width of the extended seed; an
  independent verification pass derived the band's range-uniform
  a-priori seed bound
  `delta_0' = 1.0527e-15` --- the truncation tail `4.112e-17` plus the
  split-Clenshaw forward rounding bound `1.0116e-15`, no x-sampling ---
  and the crossings under it, `2.0136 / 4.8960 / 10.7818`, replaced the
  empirical rows before the v1.1.0 tag cut.)
- **Unaffected:** the `m > 1` branch, the float lanes, and every input
  outside the band.
- **Old behavior:** pin tag `v1.0.0`.

The public history is intentionally compressed to two commits: the Initial
commit (tagged v1.0.0) and this release's cumulative commit; the items
above are the content of that cumulative commit.
