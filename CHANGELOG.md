# Changelog

What changed between released versions of this library, newest first. The version this tree carries
lives in one place, `CMakeLists.txt`'s `project(boys VERSION ...)`; `docs/specification.md` says it
in a sentence, and `boys::VersionString()` answers it at run time.

## 5.0.1

**A documentation release: the numerical contract, the stored tables and the entries are those of
5.0.0.** The bound, the fits and the generated headers are unchanged.

**The libint2 comparison note is removed from the 5.0.0 notes.** The comparator it described was
never carried in this tree, so the note pointed at something a consumer could not build.

**The region-B seed figures are corrected.** The docs described an earlier revision's seed: 19
Chebyshev coefficients and 12 rational ones, delivering `9.92e-15` and `4.46e-14`. This tree stores
25 and 13 — the Chebyshev seed at degree 24, the rational one a degree-4/8 numerator/denominator
pair — delivering `5.09e-17` and `1.69e-14`. Its bar is `2.57e-14`, the band maximum of the release's
`1e-13 · F_0(x)` contract, not the pre-release absolute `5e-14` the earlier text printed.

**The granularity figures are re-read from the committed gate run.** 1345 stored coefficients on the
shipped partition and 2650 on the rational route's narrow cut; an evaluation reads 19 to 25, against
the narrow cut's 11. The certification table, the scheme table's region-B row (degree 24, 25 stored,
bound `1.110e-16`), the run's book counts and its carrying fractions now state what
`tests/data/boys_accuracy_gate_run.txt` records.

**The single-precision lane's region-A stored count is corrected from 1067 to 1408.** Its measured
and bar columns are unchanged.

## 5.0.0

**The numerical contract is unchanged from 4.0.0.** The bound, the delivered figures, and the
committed reference grid are the same; the changes below are the API, the headers, and the records.

**The seam policy is renamed.** `boys::EvalPolicy<...>` is `boys::SeamPolicy<...>`. The name says
what the alias is: the seven names the build's seam header (`boys/boys_build_defaults.hpp`) states,
and not a class's own row, which is what a call that names no policy resolves to. The lane aliases
(`DefaultPolicyFp64`, `DefaultPolicyFp32`, `DefaultPolicyFp16`, `DefaultPolicyBf16`) keep their
names and are `SeamPolicy<>` at their lane's budget. A caller who wrote `EvalPolicy<>` writes
`SeamPolicy<>`; no other spelling moved.

**The accuracy accessors answer the contract at the cell asked.** `BoysAccuracyGuaranteed`,
`BoysAccuracyGuaranteedStated` and `DefaultGuarantee` take `(n, x)`, and their figure is the lane's
published form evaluated there: on the double lane the larger of `1e-13 · F_n(x)` and
`1e-13 · F_0(x)`, on the float lane the larger of `2^-22 · F_n(x)` and `1.5e-7`, plus the term the
lane's row publishes for the division form and the region-B member named. The measured absolute
figures – `5.5e-14` on the double lane, `1.5e-7` on the float lane – are the lane's documentation
(`BoysLaneContracts()`) and not the machine-readable promise, which is the cell's contract; a cell
outside the documented domain is refused with a reason. A figure that is a measurement now says so:
`AccuracyReading` carries `kMeasured` beside `kGuaranteed` and `kDelivered`, and the archive-free
header's bound is labelled with it. `BoysAccuracyDelivered` is unchanged: a delivered figure is a
swept maximum and not a function of the cell.

**`BoysFixedN`'s container overload spells its stride through `BoysStride{}`.** The `std::span`
overload in `boys/boys_span.hpp` took a bare `std::size_t` stride in the position where the pointer
entry takes the count, so the same position meant two things under one name; it now takes
`BoysStride{stride}` (default `BoysStride{}`, which is one). The pointer entry is unchanged, and a
count written in the container overload's stride position is a compile error rather than a stride
no size-check can see. The three container overloads also refuse a span whose extent arithmetic
would overflow, rather than admitting it.

**The CUDA lane is published as the CPU+GPU target.** `boys::boys-cuda` is one complete path with
its own figures: the device option probe measured its whole option space and the device accuracy
gate answers for it against the same committed reference grid. A build either carries the lane or
does not; there is no partial offload. The CPU-only path and the cluster path are the host targets,
as before.

**The archive-free accuracy header.** `boys/boys_accuracy_here.hpp` answers the eight lane figures
and each default-policy class's bound and delivered figure from the headers alone, with no archive
on the link line and no tool to run. It is not in `boys/boys.hpp`'s chain: a unit that includes the
front door does not pay for it.

**The coefficient tables are split by what a consumer uses.** The generated
`include/boys/boys_coefficients.hpp` used to define every stored array in one 16 MB header, parsed
by every translation unit that included the front door. It is now the master — metadata,
declarations and the seam the group headers share — and twenty-one group headers define the arrays,
one group per stored table. A call that names no policy parses the groups its class rows and the
seam select — nine groups, about 6.3 MB of tables, 6.5 MB with the master — in the scalar default;
a consumer that names another route, scheme,
partition or packing axis includes that group's header, before the front door, because it uses it;
a named policy without its header is a compile-time diagnostic naming the header.
`include/boys/boys_all.hpp`
opens every group. The values are bit-for-bit the
ones the monolith carried, and the entries run the same code. The generator and its manifest
(`tests/data/boys_coefficients_provenance.txt`) are committed beside them; `--check` verifies every
artifact and the manifest.

**A record's identity is its version and its content digests.** The accuracy gate's recorded run
and the option probe's report no longer carry a revision line. Each states the version it was taken
at and one digest per group of the files the program read, so a squash merge keeps the record's
identity and a checker compares content rather than history (`tools/record_stamp.py`).

## 4.0.0

**The double lane's accuracy contract is read against the ladder's first rung.** For every
`n <= 32` and `x >= 0` the error of a returned value is at or below the larger of `1e-13 · F_n(x)`
and `1e-13 · F_0(x)`. Releases before 4.0.0 carried absolute figures instead, the widest
`5.5e-14`. The change breaks a caller who pinned on that figure, and the new form is not
uniform read against the returned value: the value is held to `1e-13` of `F_n(x)` below the
region-C boundary and again above `x ≈ 93` at `n = 32`, and in between the delivered error reaches
140 % of the value at the worst cell, `n = 32, x = 31`. The float lane states the larger
of `2^-22 · F_n(x)` and the `1.5e-7` it already published, plus `1e-7` in the plain-reciprocal form;
the half lanes keep their own figures. `docs/specification.md#accuracy-contract` is the contract,
lane by lane, and it names the two combinations that hold the wider arm rather than the figure read
against `F_n(x)`.

**The coefficient file gains the uniform-grid tables** (`include/boys/boys_coefficients.hpp`): one
derived grid over `[0, 31.0)` of width `1/16` — 496 intervals — each order fitted at its interval's
own degree, cap 5, serving `FitGranularity::kUniform`. **The shipped partitions' tables are carried
beside the grid's rows, not replaced by them**: the file's growth is the uniform grid, and the rows
the coarsest and narrow partitions are read from are carried unchanged. What an unnamed call reaches
is its class's row rather than the seam's seven: every host double row in this build names the
uniform grid, read by Horner on the monomial form, while the seam's fallback (`SeamPolicy<>`) names
the narrow partition. The generator and its provenance record
(`tests/data/boys_coefficients_provenance.txt`) are committed beside them.

**The accuracy rung is not offered.** The axis traded accuracy for time, and the library does not
carry it: the slot the device handle reserves for it is read by no
entry.

**The shipped defaults are measured rather than assumed.** `boys-option-probe --emit-defaults`
writes a build's own winners, and `BOYS_BUILD_DEFAULTS` points a build at that header. The committed
defaults are one host's measured choices; each row names the run it came from, and a later run of
the same protocol on that host is the transcript `tests/data/boys_option_probe_report.txt`. The far
region's default representation is the bare asymptotic recurrence above the fitted grid's edge.

**Every option the policy space carries is reachable and stated.** Each combination either carries
its own measured bound or refuses to compile with a sentence saying why, and `README.md`'s "Which
entry do I call?" is the consumer's view of the surface.

### Platforms and tiers

x86-64 at the AVX2+FMA tier (the default) and at the portable scalar tier (`BOYS_ISA_AVX2=OFF`), on
Windows/MSVC and Linux (gcc, clang); arm64 at the scalar tier on Linux (gcc), Windows (MSVC) and
macOS. Optional: the fp16/bfloat16 seam (`BOYS_FP16_SEAM`, on by default), the C linkage surface,
the option probe, and the CUDA device lane (`BUILD_CUDA=ON`).

### What certifies a build of it

- `boys-accuracy-gate --strict` ends in its `PASS: every documented claim met at this revision`
  line, and the run it records is `tests/data/boys_accuracy_gate_run.txt`: 33 orders over 1797
  arguments, whose double lane's worst cell sits at `0.288` of the bound, at `n = 32, x = 31`.
- `boys-cuda-accuracy-gate` answers for the device lane against the same committed reference grid,
  and the option probe and the device probe measure their whole spaces
  (`tests/data/boys_option_probe_report.txt`, `tests/data/boys_device_probe_report.txt`).

## Before 4.0.0

The tag `v1.1.4` predates this file.
