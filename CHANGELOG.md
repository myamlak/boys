# Changelog

What changed between released versions of this library, newest first. The version this tree carries
lives in one place, `CMakeLists.txt`'s `project(boys VERSION ...)`; `docs/specification.md` says it
in a sentence, and `boys::VersionString()` answers it at run time.

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

**The coefficient file gains the uniform-grid tables**
(`include/boys/boys_coefficients_d_uniform_*.hpp` and the float lane's
`boys_coefficients_f32_uniform_*.hpp`): one
derived grid over `[0, 31.0)` of width `1/16` — 496 intervals — each order fitted at its interval's
own degree, cap 5, serving `FitGranularity::kUniform`. **The shipped partitions' tables are carried
beside the grid's rows, not replaced by them**: the file's growth is the uniform grid, and the rows
the coarsest and narrow partitions are read from are carried unchanged. What an unnamed call reaches
is its class's row rather than the seam's seven: every host double row in this build names the
uniform grid, read by Horner on the monomial form, while the seam's fallback (`SeamPolicy<>`) names
the narrow partition. The generator and its provenance record
(`tests/data/boys_coefficients_provenance.txt`) are committed beside them.

**The generated coefficients are one master and one header per group.** The vector-of-arrays text
that used to be a single `boys_coefficients.hpp` is now a master carrying the metadata, the extern
declarations with their extents, and the group tags, plus twenty-one group headers that define the
arrays. `<boys/boys.hpp>` opens the nine groups the committed class rows and the half-format
batch seeds read; a caller whose policy names another combination includes the group header the
fit's own assertion names, and a missing one is a compile-time diagnostic naming that header.
`include/boys/boys_all.hpp` opens
every group. The split moves no coefficient value and no arithmetic order; the provenance record's
manifest covers the whole set.

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
- The libint2 v2.13.1 comparison, `BoysAllOrders` against `FmEval_Chebyshev7<double>` at matched
  accuracy, was measured while preparing this release; its comparator is not carried in this tree.
  A figure from it is one host's, and `docs/consumer-perspective.md` states what may be read off a
  timing.

## Before 4.0.0

The tag `v1.1.4` predates this file.
