# Contributing to boys

Thanks for considering a contribution. This library's accuracy contract is a tested claim. A change
to the kernel can invalidate it. Read this whole file before opening a pull request.

## What this project is

A standalone C++20 library that evaluates the Boys function family F_n(x) for n = 0..32, in a
three-region scheme.

The committed generated tables and the 45-digit reference grid are part of the artifact. They are
the reproducibility evidence behind the documented accuracy contract.

## Ground rules

1. **The accuracy contract is the contract.** Every merged change must keep the committed reference
   grid green (`ctest`) at these budgets:
   - 5.5e-14 for fp64, its loosest per-region budget;
   - 1.5e-7 for fp32;
   - 1e-7 plus one half-ULP for fp16 and bf16. This is the bar the suite asserts, and it is
     tighter than the 1.5e-7 plus one half-ULP those lanes publish;
   - 8 ULP of the returned value for the native half lane, whose bound is stated in ULP rather than
     as a region budget.

   A change that needs a budget relaxed is a change to the contract. Open an issue for the
   maintainer first. Do not put it in a pull request.
2. **Write for a reader who has only this repository.** A citation a reader cannot follow is worse
   than no citation. It implies a reference that is missing. Where you would have pointed at
   something they cannot open, name the measurement, the reasoning or the result instead. The
   citations that are always safe are the published ones: the entries of `CITATION.bib`.
3. **Regeneration is local-only.** The committed tables and reference grid are the source of truth
   for the build and for CI. The generator verifies them byte-for-byte through `--check`, and that
   check is what CI runs. What is local-only is the mode that **writes** the tables: never wire that
   into CI, and never commit regenerated tables without running `--check` first.
4. **Small, reviewable changes.** One logical change per pull request.
5. **Every figure has one home.** The README states each lane's bound and the command that measures
   it. The per-lane detail lives in `docs/lane-contract.md`: the fit routes and their stored counts,
   the packing axis and its measured counts, the multiply-add route's own bounds. The README
   summarises it and links to it. A table copied into both is a table that will disagree with
   itself. A figure belongs where a reader who wants to check it would look. Everywhere else gets
   the summary and the link.

## Build and test

Requirements: CMake >= 3.25, git, **and a build tool — Ninja or Make**. The last one is easy to miss:
`cmake -S . -B build` names no generator, so it picks whatever the machine has. A machine with
neither fails at configure with a message about generators, not about the tool that is missing.
`-G Ninja` is what this repository's own configurations use.

The compiler the block below wants is a **C++23** one (MSVC, GCC or Clang), because that block is
the contributor's build: it compiles the test suite, the gate and the benchmark drivers, and those
are the targets that ask for C++23. The library itself is standalone **C++20** — README.md's recipe,
with `-DBOYS_BUILD_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_BENCHMARKS=OFF`, builds it and nothing
else.
Dependencies are vendored in-tree, so there is no vcpkg and no FetchContent.

```bash
git clone https://github.com/myamlak/boys.git
cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

Optional: `-DBUILD_BENCHMARKS=ON` for the benchmark drivers, which are default ON locally, and
`-DBUILD_CUDA=ON` for the CUDA lane, which needs the CUDA toolkit; the `linux-cuda` CI leg
compiles and links that lane and runs no kernel, so every device figure stays a local gate.

`-DBOYS_BUILD_DEFAULTS=<header>` builds with a header of your own carrying the five choices an entry
that names no policy resolves to, in place of the committed ones: the fit route, the evaluation
scheme, the packing axis, the division form and the fit granularity. They are compile-time values,
so the build compiles them into every call that names none, and pays nothing for them at run time.
The choice was a template argument before and is one still. **One of the five a build cannot move**,
and it refuses at compile time with the library's own reason rather than compiling something else:
the packing axis, because an entry that produces one order has no second order to pack into a vector
lane. The uniform member of the fit granularity was the second until this revision: the four batched
bodies that refused it now hand a policy naming the grid to the path that reads it, and the accuracy
gate's entry book measures the six cells they cover over the committed grid. Whether a build naming
that member now compiles is not stated here, because it has not been measured - no fixture in this
tree names it, and a configure that did would be the first reading of that configuration.
Setting the packing axis to what the library already runs is fine; setting it to another member is a
build that does not compile, and that is the answer rather than a defect. The file is
`include/boys/boys_build_defaults.hpp`. Its own comment is the contract a replacement satisfies: the
five names, and beside them the machine, the date and the option probe's own figures, because a
choice made there is a measurement taken on one host and a reader has to be able to tell a tuned
build from a committed one. The default is OFF. An untuned build is the committed configuration, the one
every bound in this repository was measured at. **Nothing in this tree writes such a header**:
running `boys-option-probe` at configure time and emitting your machine's ranking is not part of the
build. So this option takes a header you already have.

The accuracy gate re-measures every documented bound against the committed reference grid and prints
the comparison lane by lane and region by region:

```bash
cmake --build build --target boys-accuracy-gate
./build/Release/boys-accuracy-gate --strict      # the config directory is your generator's
```

It exits non-zero if any figure in the documents is not verified at the revision you are on, naming
each claim it could not confirm. Run it before changing a bound, a threshold, or a fitted table.
`ctest` runs it as one of its tests, but a passing `ctest` prints only how long the test took. The
figures are in this binary's own output.

The regeneration check, which is what CI runs - the mode that writes the tables is the local one:

```bash
python3 tools/gen_boys_coefficients.py --check
```

The packed orders lane is a template instantiated once per cell. Its cells are written out twice: as
`extern template` declarations in `include/boys/boys_impl.hpp`, and as definitions in
`src/boys_orders_simd.cpp`. A cell in one list and not the other fails silently, so the two are
compared cell by cell:

```bash
python3 tools/check_orders_packed_cells.py --check
```

Run it after touching either list. It compares the two lists to each other, never to the option
space. So it cannot see a class of cell that neither list names; that one reaches you as an
unresolved external at link time, named.

Ratios, shares, margins and percentages that the documents state over their own numbers are checked
against the numbers the same sentence prints:

```bash
python3 tools/check_doc_arithmetic.py --strict
```

It reads `README.md`, `CONTRIBUTING.md` and the pages under `docs/`, and recomputes every derived
figure it finds. It exits non-zero, quoting the sentence, when the arithmetic does not come out.
Where a sentence does not print both sides of the relation it states, it reports that and names the
missing side. It does not guess from whatever figure sits nearby. A "not derivable" line in its
output is therefore a reading of the prose, not a failure. Run it before changing a figure that
appears in prose. The same command runs as a CI step.

The lane rows' `source` strings are the library's own words for the terms a base figure does not
carry. The example programs print them, and the documents quote them. A row whose string grows while
a document keeps the text it used to carry leaves the document describing output no program
produces:

```bash
python3 tools/check_doc_transcripts.py --check
```

It reads the four rows out of `src/boys.cpp`, and every run of the three shapes a document quotes
in: a `[...]`, a `(from ...)`, and the `bound: ...` cell of a lane-contract row. It exits non-zero
on a run that is the beginning of a source string and stops there, naming the row, how much of that
string's length the run carries, and what the field says next. Every other run it read is printed as
one it makes no claim about, because the same brackets carry the documents' cross-reference labels
and the same cells their own prose. It faces one way on purpose: a run *longer* than every source
string cannot be told from a paraphrase. It reports that rather than guessing. Run it after editing
a lane's `source` or a document that quotes one. The same command runs as a CI step. `--source`
reads the strings from another file, which is how a revision before a change is checked as a
negative control.

The accuracy gate judges every lane and region against a bound. Those bounds are constants it
transcribes; the block they sit in says so: "transcribed from README.md and include/boys/boys.hpp".
A figure edited in one of those documents and not in the gate leaves the gate green, certifying its
own transcription, with the document's claim printed beside its own number:

```bash
python3 tools/check_bound_transcripts.py --check
```

It reads the gate's constants, the contract table in `include/boys/boys.hpp`, both the contract
table and the printed-run table in `README.md`, and the lane rows `BoysLaneContracts()` carries. It
exits non-zero, naming the constant, both files and both lines, when a figure the two state as one
fact disagrees. The correspondence between the tables is not one to one. The header has a cell per
lane and region, the README's contract table resolves by region in its prose, and the library
carries one figure per lane. So the tool declares each tie it makes, compares the README's per-lane
rows as sets rather than reading their prose as a partition, and prints every figure it did not tie,
with the reason it did not. Two things fail the run until they are declared in the tool: a constant
of the gate holding a documented figure under a name of its own, and a figure-bearing table row no
tie holds. The drift it is written for is a new transcription nobody tied. Run it after changing a
bound, a document's bound table, or a lane row. The same command runs as a CI step. `--gate`,
`--header`, `--readme` and `--library` read a side from another file, which is how a shifted figure
is shown to fail as a negative control.

The build-defaults seam offers a consumer five choices, and a choice read by nothing is what this
check is for: three of the five once sat in the seam file, documented, while the library compiled
hard-coded literals, so a consumer who replaced it had two choices honoured and three silently
ignored and nothing reported it.

```
python3 tools/check_seam_macros.py --check
```

It reads the macro names off `include/boys/boys_build_defaults.hpp` and requires each to be expanded
somewhere in `include/` or `src/`, not counting the seam file itself or the fixtures that replace it.
It does not check that a reader is *correct*, or that a build can move the axis at all - those are
compile-time facts, and a build that moves an axis and fails to compile is the reading for them. Run
it after touching the seam or a header that reads it. The same command runs as a CI step. `--seam`
and `--root` read the two sides from elsewhere, which is how a macro nothing reads is shown to fail
as a negative control.

The device surface is described by several hand-written lists, and the two a caller meets - the
option enumeration and the entries the class declares - are held to each other in both directions:

```
python3 tools/check_device_entry_bijection.py --check
```

A row naming a member the class does not declare is an option the library advertises and cannot
deliver; a member no row names is an entry no report describes and no book counts. They already
disagreed once, in the first direction, and four mechanisms consumed the description without one of
them asking whether the entry existed. Run it after touching the enumeration or the class. The same
command runs as a CI step. `--options` and `--header` read the two sides from elsewhere, which is how
a deleted member is shown to fail as a negative control.

The host surface has the same exposure, and the same check:

```
python3 tools/check_host_entry_bijection.py --check
```

It reads the names the headers declare and the shapes `tools/host-shapes.json` places them in, and
requires each side to carry exactly the other's. A declared name the table does not place is a call
whose shape nothing states; a name the table places that no header declares is a shape described for
a call that does not exist, which a later reader takes for a decision. The unit is the shape rather
than the name - a run-time tier, a route-named sibling, a sorted-argument overload and a span overload
are spellings of one shape - and what a shape offers is derived from its tuple rather than written in
the table, so a one-order shape has a packing axis of one value by construction rather than by a
refusal in the file. Names are read from the headers rather than from a second list, so the check
cannot agree with itself while disagreeing with the code. Run it after touching a declaration in
`include/boys/` or the shape table. The same command runs as a CI step. `--header` and `--table` read
the two sides from elsewhere, which is how a deleted declaration is shown to fail as a negative
control.

That check's unit is the shape, and a shape counts as placed when any one precision has an entry for
it. The class is the unit a caller meets, and it is the cross of the two enumerations the library
declares, so a second check walks that cross:

```
python3 tools/check_class_surface.py --check
```

It reads `Precision` and `Shape` from `include/boys/boys.hpp` and requires every cell of the cross to
be served by an entry whose default policy names that class. A cell served at one precision and not
at another is exactly what the shape-unit check cannot see: the shape is placed, and the class is
absent. A class may instead be listed in the table's `absent` array with a reason, and the reason must
be non-empty - an impossibility, stated - because the array being empty is this project's statement
that the cross is served.

CI runs the full platform matrix listed in the README on every push to `main` and every pull request.
A pull request that breaks any leg fails. A maintainer run of the native toolchain gate is expected
before merge.

**Two more checks beyond those above are worth knowing about, because they are easier to trip than
to find.** The option matrix (`.github/option-matrix.json`, generated by `tools/gen_option_matrix.py`)
is checked in both directions against the library's own declarations, so adding or renaming an axis
means regenerating it. `tests/boys_avx2_probe.cpp` asserts that the CPU feature dispatch reports
what it claims. Without it, a wrong CPUID bit lets every SIMD test skip quietly while CI stays
green. That is the failure it was written for. `ci.yml`'s own comments explain the coverage
strategy: why the matrix is architecture- and sanitizer-focused rather than a fast subset. They are
worth reading before changing which legs run.

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
  CUDA lane reports through the `BoysStatus` enum. New fallible surfaces follow the same pattern.
- **Comments:** brief, self-contained and why-focused. The reasoning always stays; only the pointer
  to somewhere the reader cannot go is dropped.

## Tests

- Kernel changes land with tests. Expect per-order accuracy against the committed reference grid,
  region-boundary coverage with x0 and x1 exact, and the fp16/bf16 I/O contract. Do not extend the
  grid or the tables in the same pull request as a kernel change without the `--check` evidence.
- Tests must pass on the CI matrix, and locally in Debug and Release.
- Benchmark changes belong with the benchmark suite under `BUILD_BENCHMARKS`, and must record the
  machine, the clock and the commit beside each number.

## What never goes in

- Anything a reader of this repository cannot open, per ground rule 2.
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
  m-values. Weakening a documented accuracy budget also belongs here, as does an error-behaviour
  change that can break callers.
- **MINOR** — additive public API, or any internal change to the algorithm, dispatch, region
  boundaries or certified tables that may alter returned bits while preserving every documented
  domain and budget.
- **PATCH** — no intended public-API or numerical-result change. If a change can alter any returned
  bit for any supported input, it is at least minor.

The version is written down once, in the top-level `project(boys VERSION ...)`, and it reaches a
caller through `boys/version.hpp`. The release tag and that `project()` call name the same version:
the consumer check compiles against the project value and asserts the header agrees. So a release
bumps the one place, and a tag that disagrees with it is a mistake, not a second answer.

The public numerical contract is that for every supported n, x, lane and region the returned value
satisfies |F̂_n(x) − F_n(x)| ≤ m·B_region at the documented m. Public function signatures and
supported domains are stable within a major version. Bitwise outputs are not: internal region
thresholds, seed selection, recursion order and dispatch logic may change between minor releases, as
long as the bound holds. Byte-for-byte reproducibility requires pinning the release tag, the compiler
and the build flags.

## Getting help

Open an issue for questions, bugs, or proposals to relax the contract, before writing code. The
maintainer is the sole reviewer; expect standards.
