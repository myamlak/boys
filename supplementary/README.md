# Supplementary material — "Certified mixed precision for quantum-chemistry special functions: a-priori error bounds and the Boys function as a worked case study"

This directory accompanies the manuscript `main.tex` (the compiled
`main.pdf` is included). Everything here is self-contained: no path in any
file refers to the authors' private repositories, and every table of the
manuscript is regenerable with the commands below. The only requirements
are Python 3 with `mpmath` and (for the boundaries table) any C++17
compiler.

Install (once):

```
pip install mpmath
```

## Package contents

| File | What it is |
|---|---|
| `main.tex` / `main.pdf` | the manuscript and its compiled form |
| `boys_reference.csv` | the 30-digit reference grid (n = 0..32, 70 x-values in [0, 100]) |
| `gen_boys_coefficients.py` | the coefficient-table and reference-grid generator |
| `fp16_bf16_io_worst.py` | regenerates the fp16/bf16 worst cells of tab:accuracy |
| `extract_throughput.py` | checks every tab:throughput cell against the recorded runs |
| `boys_boundary_standalone.cpp` | regenerates tab:boundaries (the three measurement records) |
| `vs_missing_trailing_term.py` | reproducibility finding (i): the V&S region-B denominator |
| `tsuji_table_step_sweep.py` | reproducibility finding (ii): the Tsuji table step d = 2^-3 vs 2^-5 |
| `boys_benchmark.cpp` | the CPU benchmark sources (tab:throughput CPU rows) |
| `boys_unsorted_simd_benchmark.cpp` | the unsorted-SIMD benchmark (the 3.2x penalty row) |
| `boys_cuda_benchmark.cpp` | the GPU benchmark sources (tab:throughput GPU rows) |
| `runs/` | the recorded runs logs, the accuracy record, and the run protocol |

The benchmark sources accompany the submission as the measurement
instruments; they compile against the implementation library, which the
manuscript's Declarations promise to release as open source upon
publication (available from the authors upon request in the meantime).

## One command per manuscript table

**tab:accuracy** (fp16/bf16 worst cells 2.4e-4 / 1.9e-3):

```
python fp16_bf16_io_worst.py
```

Runs the I/O-only rounding emulation over the shipped reference grid and
prints the per-region and overall worsts; the overall worsts must match
the manuscript cells 2.4e-4 (fp16) and 1.9e-3 (bf16). The remaining rows
of tab:accuracy are budget-style cells (the <= bounds of the certified
lanes); their measured worsts are recorded in
`runs/boys-accuracy-i7-9850h-2026-08-28.txt` (asserted by the
accompanying test suite, with the GPU comparison rows in the GPU runs
log).

**tab:throughput** (all seven rows and the caption ratios):

```
python extract_throughput.py
```

Extracts the recorded medians from `runs/boys-cpu-i7-9850h-2026-08-29.txt`
and `runs/boys-gpu-t1000-2026-08-29.txt` and checks each cell and each
caption ratio (4.4x, 13.8x, 3.2x) at the cells' stated precision.

**tab:boundaries** (the four measured rows; a C++17 compiler is needed):

```
cl /O2 /std:c++17 /EHsc boys_boundary_standalone.cpp /Feboys_boundary.exe
boys_boundary.exe
```

(or `g++ -O2 -std=c++17 boys_boundary_standalone.cpp -o boys_boundary &&
./boys_boundary`). Reproduces the three measurement records: [3] the
0.0001-resolution descending sweep (the `--postcheck` mode below),
whose first failing samples the manuscript cells 0.4625 / 1.6373 /
4.2367 / 10.0492 record; [2] the fine two-phase sweep (0.01 then
0.001 steps), whose cells (0.393 / 1.484 / 4.150 / 9.866) the
manuscript previously recorded and now supersedes; and [1] the coarse
geometric grid (x = 0.05, x *= 1.02) of the original design study,
whose cells (0.244 / 0.754 / 3.82 / 9.70) are likewise superseded.
The [2] and [1] cells are reproduced here as the
configuration-sensitivity record: the recursion error oscillates
through the 5e-14 line near the transition (402-3351 pass/fail
alternations per band at 0.0001 resolution), so each grid records its
own lattice draw, ulp-sensitive within ~10-20% — which is why the
`BOUNDARY-CHECK` gate accepts the fine sweep within a 20% band of the
manuscript cells and the accompanying test suite pins them with the
same tolerance — see the file header.

The 0.0001-resolution descending sweep the manuscript's tab:boundaries
note cites (same executable, `--postcheck` mode):

```
boys_boundary.exe --postcheck
```

Re-runs the fine two-phase sweep for context, then walks the whole
[0.8 x0, 1.2 x0] band of each of the four recorded cells (kmax = 4:
0.4625, band [0.37, 0.555]; kmax = 8: 1.6373, band [1.30984,
1.96476]; kmax = 16: 4.2367, band [3.38936, 5.08404]; kmax = 32:
10.0492, band [8.03936, 12.05904]) at step 0.0001 with the same
seed, step, and 5e-14 criterion, counting the pass/fail alternations
and reporting the first failing sample. A row is confirmed when the
band top passes and the walk's first failing sample reproduces its
recorded cell within the 20% band. On the recording machine (MSVC,
2026-09-05) all four rows are confirmed: the first failing samples
sit at 0.4625 (kmax = 4; last pass 0.4626), 1.6373 (kmax = 8; last
pass 1.6374), 4.2367 (kmax = 16; last pass 4.2368) and 10.0492
(kmax = 32; last pass 10.0493), with 402 / 1330 / 2629 / 3351
pass/fail alternations per band — this is the sweep whose first
failing samples the manuscript cells record. The earlier 0.001
two-phase cells (0.393 / 1.484 / 4.150 / 9.866) failed this sweep
(+17.7% / +10.3% / +2.1% / +1.9% at 0.0001 resolution), and 1e-5
spot scans of the envelope-top neighborhoods extend the largest
failing x to ~0.4625 / ~1.65 / ~4.298 / ~10.059 (kmax = 4 / 8 / 16 /
32), so the cells are resolution-limited first-failure records, not
stability thresholds. The post-check exits 0 only when all four rows
are confirmed. Like the sweep it re-samples, the post-check is
resolution-limited and not a formal proof.

## The reference grid itself

```
python gen_boys_coefficients.py --check
```

Regenerates the reference CSV from scratch (30-digit mpmath) and verifies
it is byte-identical to the shipped `boys_reference.csv`. Running the
script without arguments additionally regenerates the Chebyshev
coefficient header (`boys_coefficients.hpp`) for both lanes.

## The reproducibility findings (manuscript Section "Reproducibility findings")

```
python vs_missing_trailing_term.py      # finding (i): ~1e-3 on [x0, x1)
python tsuji_table_step_sweep.py        # finding (ii): 5.6e-13 (2^-3) / 4.1e-16 (2^-5)
```

Both are mpmath-only and re-analyze the published artifacts (arXiv:
2512.10059 and the companion repository of the Tsuji et al. paper) from
hardcoded published coefficients — no access to the authors' private code
is assumed.
