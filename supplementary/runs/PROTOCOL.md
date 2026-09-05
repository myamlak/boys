# Clean-run benchmark protocol — Boys-kernel manuscript (tab:throughput)

Every cell of tab:throughput in the accompanying manuscript and every ratio
in its caption traces to a row of the recorded runs logs shipped beside this
file. The manuscript's Declarations promise ("every numerical claim below is
reproduced by the implementation and benchmark suite that accompany the
submission") is satisfied by traceability, not by determinism: wall-clock
throughput is inherently machine- and environment-dependent, so the protocol
records the environment and the statistics rather than pretending they are
absent.

**Conventions.** Paper cells are **medians** of 3 recorded passes after 1
warmup pass. Ratios (e.g. the 4.4x / 13.8x / 3.2x of the manuscript) are
computed from the recorded rows, never asserted independently.

## Protocol (each recorded run)

(a) **Environment closure** — no other benchmark process and no heavy
    build in flight (a process-list check for benchmark executables,
    compiler processes, and interpreters before every timed pass — timing
    lanes must not overlap on the same machine), no interactive load,
    Windows power plan "High performance" (`powercfg /getactivescheme`
    must report it; the runs log records the GUID).
(b) **Device state (GPU lanes)** — `nvidia-smi --query-gpu=clocks.sm,
    clocks.mem,power.draw,temperature.gpu` sampled before, during, and
    after each kernel. A pass whose SM clock drops from its steady-state
    value (laptop thermal throttling) is discarded and re-run after the
    device cools.
(c) **Statistics** — one warmup pass, then 3 recorded passes; the runs
    log reports min/median/max per kernel; the paper cell is the median.
(d) **Identity** — machine (CPU model, RAM, GPU), driver version, CUDA
    toolkit, MSVC version, commit hash, build preset, power plan GUID,
    date, and the protocol fields in every runs log.
(e) **Ratios** — computed from the recorded rows, never asserted
    independently.

## Benchmark executables

Accuracy cells come from the `--self-check` runs (asserted against the
certified CPU references; the paper's tab:accuracy rows cite the contract
test suite for the shipped lanes and these self-check cells for the
competitor/unsorted lanes). Throughput cells come from the timing mode.
Three benchmark programs accompany the submission:

- the sorted CPU benchmark (scalar double/float, batch, SIMD lanes),
  run with `--benchmark_repetitions=3` — the 3 recorded passes per lane
  (the first repetition of the first lane doubles as the process warmup),
  the paper cell is the median of the 3 rows;
- the unsorted-SIMD benchmark (custom main, runs-log rows), run as
  `--self-check` then the timing mode;
- the GPU benchmark (custom main, runs-log rows; built with CUDA for the
  demonstration device), run as `--self-check` then the timing mode.

## CPU-side application (the CPU rows of tab:throughput)

The sorted SIMD lanes, the scalar lanes, and the unsorted-SIMD lane follow
the same protocol: environment closure, Windows power plan, one warmup
pass, 3 recorded passes, median; MSVC `/O2`, `/arch:AVX2` as stated in the
paper's tab:throughput caption. They record into the runs logs accompanying
the submission. The process-list closure check applies; device-state
sampling does not. The unsorted lane measures the mixed per-vector kernel
on the unsorted uniform-x40 workload (order 8, rng(46) — the design-study
shape); the flat-Taylor competitor is not included (scope decision,
2026-08-29).

## GPU-side application (the GPU rows of tab:throughput)

The GPU benchmark measures five lanes on the uniform (n in [0,32], x in
[0,40], mt19937_64(42)) workload — identical to the CPU benchmark's
uniform workload: cheb-f64, cheb-f32 (the certified GPU lanes), erf-f64
and lut-f64 (the competitor kernels), and fp16-single (the I/O-only lane;
the host-pointer synchronous API owns its device buffers per call, so its
row includes the per-call allocation cost — the honest number for the
shipped API). `--self-check` verifies every lane against the CPU
references before any timing run.

## Runs-log template

Every runs log carries:

- identity block: date, machine (CPU model, RAM, GPU), driver version,
  CUDA toolkit, MSVC version, commit hash, build preset, power plan GUID
- per kernel, per pass: the recorded ms (3 passes after the warmup),
  plus min/median/max computed over the 3 passes and the median items/s
- the `--self-check` accuracy rows (one line per lane: max abs error vs
  the CPU reference and the asserted budget)
- a summary row per kernel: lane, workload (uniform/molecular), items/s
  (median) — the paper cell
- the statement "Recorded under `PROTOCOL.md`"

Example row layout (placeholder values):

```
self-check: cheb-f32 | max_abs_err: 1.77e-07 | budget: 3.50e-07 | PASS
kernel: cheb-f32 | workload: uniform-n32-x40 | count: 4194304 | passes: 3 |
min_ms: 4.021 | median_ms: 4.055 | max_ms: 4.113 | median_Mvals_per_s: 1034.1
```
