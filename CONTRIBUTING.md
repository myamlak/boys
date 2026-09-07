# Contributing to boys

Thanks for considering a contribution. This library is a paper-backed,
measurement-cited artifact: the accuracy contract in the README is a
published claim, and changes to the kernel can invalidate it. Read this
whole file before opening a PR.

## What this project is

A standalone C++23 library evaluating the Boys function family F_n(x),
n = 0..32, in the three-region scheme, cut from the qcx quantum-chemistry
framework and released with its paper (citation in the README). The
committed generated tables and the 30-digit reference grid are part of
the artifact: they are the reproducibility evidence behind the paper's
numbers. The public history is intentionally compressed to two commits
(the squashed release-cut practice) for the first public release; the
v1.0.0 and v1.1.0 tags mark the two release cuts.

## Ground rules

1. **The accuracy contract is the contract.** Every merged change must
   keep the committed reference grid green (`ctest`) at the documented
   budgets: 5e-14 fp64, 1e-7 fp32, fp32 + one half-ULP fp16/bf16. A
   change that needs a budget relaxation is a paper-level claim change —
   it belongs in an issue for the maintainer first, not in a PR.
2. **No internal references, ever.** This is a public repository cut from
   a private monorepo. Comments, commit messages, and docs must not
   reference internal decision numbers, internal doc paths, stage or
   track names, or the private repository. The only sanctioned citations
   are published ones: the entries of `CITATION.bib`, which mirror the
   accompanying paper's bibliography.
3. **Regeneration is local-only.** The committed tables and reference
   grid are the source of truth for the build and CI; the generator
   (`tools/gen_boys_coefficients.py`) verifies them byte-for-byte via
   `--check`. Never commit regenerated tables without running `--check`
   and never wire regeneration into CI.
4. **Small, reviewable changes.** One logical change per PR.

## Build and test

Requirements: CMake >= 3.25, a C++23 compiler (MSVC, GCC, or Clang), git.
No vcpkg, no FetchContent — dependencies are pinned submodules.

```bash
git clone https://github.com/myamlak/boys.git --recurse-submodules
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional: `-DBUILD_BENCHMARKS=ON` (benchmarks; default ON locally) and
`-DBUILD_CUDA=ON` (CUDA lane; needs the CUDA toolkit, local-only).
Regeneration check (local, never CI):

```bash
python3 tools/gen_boys_coefficients.py --check
```

CI runs two legs (windows-msvc and linux-gcc, Release) per main push;
CUDA is
maintained locally — a PR that breaks the Linux matrix fails, and a
maintainer run of the native toolchain gate is expected before merge.

## Style rules

The library's style rules:

- **Naming:** PascalCase for types and functions, camelCase variables,
  `_camelCase` private members, kPascalCase constants/enums, short
  lowercase namespaces. No snake_case anywhere.
- **Formatting:** the repository `.clang-format` (4-space, 100 columns,
  attached braces, left pointers) is enforced; run it on your changes
  before pushing. Control-statement braces go on their own line, always
  — no one-liners — and multi-line control statements get a blank line
  before and after.
- **Headers:** `#pragma once`; every public API entry carries Doxygen
  `///` comments with `\param`, `\return`, and `\pre` for preconditions
  (the Doxygen gate is `EXTRACT_ALL = NO`, zero warnings).
- **Raw arrays:** 3-vector coordinates use `std::array<double, 3>`;
  pointer + count APIs use `std::span`; raw arrays only where ABIs
  mandate them.
- **Errors:** no exceptions. CPU lanes are total functions with
  documented preconditions; the CUDA lane reports via the `BoysError`
  enum. New fallible surfaces follow the same pattern.
- **Comments:** brief, self-contained, and why-focused (ground rule 2
  limits what may be cited; the reasoning itself always stays).

## Tests

- Kernel changes land with tests: per-order accuracy against the
  committed reference grid, region-boundary coverage (x0 and x1 exact),
  and the fp16/bf16 I/O contract. Do not extend the grid or the tables
  in the same PR as a kernel change without the `--check` evidence.
- Tests must pass on the CI matrix and locally in Debug and Release.
- Benchmark changes belong with the benchmark suite (`BUILD_BENCHMARKS`)
  and must record machine state per the runs-log convention.

## What never goes in

- Anything referencing the private source repository: internal paths, decision
  numbers, stage/track names, person-specific internal notes.
- Vendored code beyond the pinned submodules (GoogleTest, optional
  Google Benchmark) and the generated tables.
- Generated-table or reference-grid edits without `--check` evidence.
- License-unattributed third-party code or data.

## License and legal

The library is BSD-3-Clause (see LICENSE, `Copyright (c) 2026 Marcin
Makowski`). By contributing, you agree your contribution is licensed
under the same terms. Third-party components and their licenses are
listed in THIRD_PARTY_NOTICES.md.

## Versioning

- **MAJOR** = an incompatible public API/ABI change (removal, rename,
  signature, layout, namespace) or the narrowing/removal of supported
  domains, lanes, types, or m-values, the weakening of a documented
  accuracy budget, or an error-behavior change that can break callers.
- **MINOR** = additive public API, or any internal
  algorithm/dispatch/region-boundary/certified-table change that may
  alter returned bits but preserves every documented domain and budget
  (the per-range seed design of v1.1.0 is this class).
- **PATCH** = no intended public-API or numerical-result change; if a
  change can alter any returned bit for any supported input, it is at
  least minor.

The public numerical contract is that for every supported n, x, lane,
and region the returned value satisfies |F_hat_n(x)-F_n(x)| <= m*B_region
at the documented m; public function signatures and supported domains are
stable within a major version. Bitwise outputs, internal region
thresholds, seed selection, recursion order, and dispatch logic are not
stable between minor releases and may change as long as that bound
remains satisfied; exact bitwise reproducibility requires pinning the
release tag, compiler, and build flags.

## Getting help

Open an issue for questions, bugs, or contract-relaxation proposals
before writing code. The maintainer is the sole reviewer; expect
standards.
