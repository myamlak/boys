# Contributing to boysymmetriad

Thanks for considering a contribution. This library is a paper-backed,
measurement-cited artifact: the accuracy contract in the README is a
published claim, and changes to the kernel can invalidate it. Read this
whole file before opening a PR.

## What this project is

A standalone C++23 library evaluating the Boys function family F_n(x),
n = 0..32, in the three-region scheme, released with its accompanying
paper (citation in the README). The committed generated tables and the
30-digit reference grid are part of the artifact: they are the
reproducibility evidence behind the paper's numbers.

## Ground rules

1. **The accuracy contract is the contract.** Every merged change must
   keep the committed reference grid green (`ctest`) at the documented
   budgets: 1e-15/5.5e-14 fp64 by lane, 1.5e-7 fp32, fp32 + one half-ULP
   fp16/bf16. A change that needs a budget relaxation is a paper-level
   claim change — it belongs in an issue for the maintainer first, not in
   a PR.
2. **Published citations only.** Comments, commit messages, and docs must
   cite only published work — the papers behind the evaluation scheme and
   the library's own paper. Internal process labels, private-repository
   references, and unpublished design material must never appear.
3. **Generated files are verified, never hand-edited.** The committed
   tables and reference grid are the source of truth for the build and
   CI; the generator (`tools/gen_boys_coefficients.py`) reproduces them
   byte-for-byte, and `--check` is the identity proof. Never commit
   regenerated tables without running `--check`; every CI leg runs the
   full regeneration protocol, so a drift fails the leg.
4. **Small, reviewable changes.** One logical change per PR.

## Build and test

Requirements: CMake >= 3.25, a C++23 compiler (MSVC, GCC, or Clang), git.
No vcpkg, no FetchContent — dependencies are pinned submodules.

```bash
git clone https://github.com/[owner]/boysymmetriad.git --recurse-submodules
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional: `-DBUILD_BENCHMARKS=ON` (benchmarks; default ON locally) and
`-DBUILD_CUDA=ON` (CUDA lane; needs the CUDA toolkit — a local-only gate,
CI has no GPU leg). The regeneration identity check:

```bash
python3 tools/gen_boys_coefficients.py --check
```

CI runs two legs per push — Windows MSVC (Release) and Linux GCC
(Release) — each with the full gate: build, ctest, the generator
regeneration protocol, the BOUNDARY-CHECK standalone, and the
zero-warning Doxygen build. CUDA and the timed benchmark runs are
maintained locally; a maintainer run of those native gates is expected
before merge.

## Style rules

The repository style rules apply to all changes:

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
  documented preconditions; the CUDA lane reports via the `BoysStatus`
  enum and the C surface via integer status codes. New fallible surfaces
  follow the same patterns.
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

- Unpublished references of any kind: internal decision numbers, process
  or track labels, private-repository paths, person-specific internal
  notes.
- Vendored code beyond the pinned submodules (GoogleTest, optional
  Google Benchmark) and the generated tables.
- Generated-table or reference-grid edits without `--check` evidence.
- License-unattributed third-party code or data.

## License and legal

The library is BSD-3-Clause (see LICENSE, `Copyright (c) 2026 Marcin
Makowski`). By contributing, you agree your contribution is licensed
under the same terms. Third-party components and their licenses are
listed in THIRD_PARTY_NOTICES.md.

## Getting help

Open an issue for questions, bugs, or contract-relaxation proposals
before writing code. The maintainer is the sole reviewer; expect
standards.
