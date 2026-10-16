# Changelog

What changed between released versions of this library, newest first. The version this tree is
lives in one place, `CMakeLists.txt`'s `project(boys VERSION ...)`; `docs/specification.md` says it
in a sentence, and `boys::VersionString()` answers it at run time.

## 4.0.0

**The double lane's accuracy contract is read against the ladder's first rung.** For every
`n <= 32` and `x >= 0` the error of a returned value is at or below the larger of `1e-13 · F_n(x)`
and `1e-13 · F_0(x)`. Releases before 4.0.0 carried one absolute figure for the whole range,
`5.5e-14`, instead. The change breaks a caller who pinned on that figure, and the new form is not
uniform read against the returned value: the value is held to `1e-13` of `F_n(x)` below the
region-C boundary and again above `x ≈ 93` at `n = 32`, and in between the bound is a fraction of
the value that reaches 140 % at the worst cell, `n = 32, x = 31`. The float lane states the larger
of `2^-22 · F_n(x)` and the `1.5e-7` it already published, plus `1e-7` in the plain-reciprocal form;
the half lanes keep their own figures. `docs/specification.md#accuracy-contract` is the contract,
lane by lane, and it names the two combinations that hold the wider arm rather than the figure read
against `F_n(x)`.

**The coefficient tables are regenerated** (`include/boys/boys_coefficients.hpp`), on one derived
grid over `[0, 31.0)` of width `1/16` — 496 intervals — each order fitted at its interval's own
degree, cap 5. The generator and its provenance record
(`tests/data/boys_coefficients_provenance.txt`) are committed beside them.

**The accuracy rung is not offered.** The axis permitted 4096 times the error for a few per cent of
time, and the library does not carry it: the slot the device handle reserves for it is read by no
entry.

**The shipped defaults are measured rather than assumed.** `boys-option-probe --emit-defaults`
writes a build's own winners, and `BOYS_BUILD_DEFAULTS` points a build at that header. The committed
defaults are one host's measured choices, and the run behind them is
`tests/data/boys_option_probe_report.txt`. The far region's default representation is the bare
asymptotic recurrence above the fitted grid's edge.

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
- `benchmarks/boys_libint2_benchmark.cpp` is the committed comparator: `BoysAllOrders` against
  libint2 v2.13.1's `FmEval_Chebyshev7<double>` at matched accuracy. A figure from it is one host's,
  and `docs/consumer-perspective.md` states what may be read off a timing.

## Before 4.0.0

The tags `v1.1.4` and `v2.0.0` predate this file.
