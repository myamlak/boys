# Contributing to boys

Thanks for considering a contribution. This library's accuracy contract is a tested claim, and a
change to the kernel can invalidate it. Read this whole file before opening a pull request.

## What this project is

A standalone C++23 library evaluating the Boys function family F_n(x) for n = 0..32, in a
three-region scheme.

The committed generated tables and the 45-digit reference grid are part of the artifact. They are the
reproducibility evidence behind the documented accuracy contract.

## Ground rules

1. **The accuracy contract is the contract.** Every merged change must keep the committed reference
   grid green (`ctest`) at the documented budgets: 5.5e-14 fp64, its loosest per-region budget;
   1.5e-7 fp32; 1e-7 plus one half-ULP for fp16 and bf16; and 8 ULP of the returned value for the
   native half lane, whose bound is stated in ULP rather than as a region budget. A change that needs
   a budget relaxed is a change to the contract. Open an issue for the maintainer first, and do not
   put it in a pull request.
2. **No internal references, ever.** Comments, commit messages and docs must not reference internal
   decision numbers, internal document paths, stage or track names, or any private repository. The
   only sanctioned citations are published ones: the entries of `CITATION.bib`.
3. **Regeneration is local-only.** The committed tables and reference grid are the source of truth
   for the build and for CI. The generator verifies them byte-for-byte through `--check`. Never
   commit regenerated tables without running `--check`, and never wire regeneration into CI.
4. **Small, reviewable changes.** One logical change per pull request.

## Build and test

Requirements: CMake >= 3.25, a C++23 compiler (MSVC, GCC or Clang), and git. Dependencies are
vendored in-tree, so there is no vcpkg and no FetchContent.

```bash
git clone https://github.com/myamlak/boys.git
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional: `-DBUILD_BENCHMARKS=ON` for the benchmark drivers, which are default ON locally, and
`-DBUILD_CUDA=ON` for the CUDA lane, which needs the CUDA toolkit and is local-only.

Regeneration check, which is local and never CI:

```bash
python3 tools/gen_boys_coefficients.py --check
```

CI runs the full platform matrix listed in the README on every push to `main` and every pull request.
A pull request that breaks any leg fails. A maintainer run of the native toolchain gate is expected
before merge.

## Style rules

- **Naming:** PascalCase for types and functions, camelCase for variables, `_camelCase` for private
  members, kPascalCase for constants and enums, short lowercase namespaces. No snake_case anywhere.
- **Formatting:** the repository `.clang-format` is enforced (4-space indent, 100 columns, attached
  braces, left pointers). Run it on your changes before pushing. Control-statement braces go on their
  own line, always, with no one-liners. Multi-line control statements get a blank line before and
  after.
- **Headers:** `#pragma once`. Every public API entry carries Doxygen `///` comments with `\param`,
  `\return` and `\pre` for preconditions. The Doxygen gate runs with `EXTRACT_ALL = NO` and zero
  warnings.
- **Raw arrays:** 3-vector coordinates use `std::array<double, 3>`. Pointer-plus-count APIs use
  `std::span`. Raw arrays only where an ABI mandates them.
- **Errors:** no exceptions. The CPU lanes are total functions with documented preconditions. The
  CUDA lane reports through the `BoysError` enum. New fallible surfaces follow the same pattern.
- **Comments:** brief, self-contained and why-focused. Ground rule 2 limits what may be cited; the
  reasoning itself always stays.

## Tests

- Kernel changes land with tests. Expect per-order accuracy against the committed reference grid,
  region-boundary coverage with x0 and x1 exact, and the fp16/bf16 I/O contract. Do not extend the
  grid or the tables in the same pull request as a kernel change without the `--check` evidence.
- Tests must pass on the CI matrix, and locally in Debug and Release.
- Benchmark changes belong with the benchmark suite under `BUILD_BENCHMARKS`, and must record the
  machine state per the runs-log convention.

## What never goes in

- Anything referencing the private source repository: internal paths, decision numbers, stage or
  track names, or person-specific internal notes.
- Vendored code beyond the in-tree GoogleTest and Google Benchmark trees and the generated tables.
- Generated-table or reference-grid edits without `--check` evidence.
- Licence-unattributed third-party code or data.

## License and legal

The library is BSD-3-Clause; see LICENSE, `Copyright (c) 2026 Marcin Makowski`. By contributing you
agree that your contribution is licensed under the same terms. Third-party components and their
licences are listed in THIRD_PARTY_NOTICES.md.

## Versioning

- **MAJOR** — an incompatible public API or ABI change: a removal, rename, signature or layout
  change, a namespace change, or the narrowing or removal of supported domains, lanes, types or
  m-values. Also here: weakening a documented accuracy budget, or an error-behaviour change that can
  break callers.
- **MINOR** — additive public API, or any internal change to the algorithm, dispatch, region
  boundaries or certified tables that may alter returned bits while preserving every documented
  domain and budget.
- **PATCH** — no intended public-API or numerical-result change. If a change can alter any returned
  bit for any supported input, it is at least minor.

The public numerical contract is that for every supported n, x, lane and region the returned value
satisfies |F̂_n(x) − F_n(x)| ≤ m·B_region at the documented m. Public function signatures and
supported domains are stable within a major version. Bitwise outputs are not: internal region
thresholds, seed selection, recursion order and dispatch logic may change between minor releases, as
long as the bound holds. Byte-for-byte reproducibility requires pinning the release tag, the compiler
and the build flags.

## Getting help

Open an issue for questions, bugs, or proposals to relax the contract, before writing code. The
maintainer is the sole reviewer; expect standards.
