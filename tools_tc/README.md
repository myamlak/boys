# tools_tc — the tensor-core mode measurements

The instrument behind the three modes of `boys::ProductMode` and the bounds
`include/boys/boys_transform.hpp` publishes for them: `kFp64` at `m·1e-15`,
`kTf32x3` and `kBf16x6` at `m·1e-15 + 2.5e-7`. The delivered worst each mode
claims over region A — 1.110e-16, 1.916e-07, 1.946e-07 — is measured here,
and `tests/boys_accuracy_gate.cpp` holds a sweep to those figures.

**It is a CPU simulation and needs no tensor-core hardware.** A mode is fixed
by the format its operands are rounded to, the precision its products are
accumulated in, and the order the degree loop is reduced in; `modes.py`
simulates all three in numpy. What is verified is the arithmetic of the three
modes, not any card's implementation of them — in particular an fp32
tensor-core accumulator truncates its addends where the model rounds, which
`include/boys/boys_transform.hpp` states as the direction its two split-mode
bounds can be wrong in.

## Running it

Requires Python 3 with `numpy` and `mpmath` (`mpmath` is pinned by
`requirements-boys.txt`). Nothing else, and no GPU.

```
python tools_tc/compare.py                 # the three bounds, beside the published figures
python tools_tc/compare.py --dense 1000000 # the sweep density they were taken at
```

`compare.py` is the one to start with: it reads the published figures out of
`tests/boys_accuracy_gate.cpp` rather than restating them, so the comparison
cannot drift from the claim it is checking.

## What it measured

| Mode | Published | Reproduced here |
| --- | --- | --- |
| `kFp64` | 1.110e-16 | 1.1102e-16 |
| `kTf32x3` | 1.916e-07 | 1.9155e-07 |
| `kBf16x6` | 1.946e-07 | 1.9519e-07 |

`kFp64` reproduces on the 129-point grid; the two split modes are dense-sweep
figures — the header names a 200000-point sweep of the lower band's left end —
and reach the published values at a million points a piece. `kBf16x6` lands
0.3% above its published figure there, which is the same measurement at a
slightly finer sweep than the one that number was taken from.

Two properties of the reduction are worth knowing before reading any row here.
The degrees may be summed into one running total from the constant term up, or
laid out high to low and reduced pairwise; the shipped kernel does the second,
and the two differ by about three times at `kFp64` and at the split modes'
formats. `modes.gemm` defaults to the running total, so `measure.py`'s own
logs record that family — `measure2.log`'s 3.331e-16, 1.871e-07 and 1.972e-07
are the running total, and only the first two `compare.py` columns with
`pairwise` are the published figures. `compare.py` prints both layouts side by
side for that reason.

## The files

| File | What it does |
| --- | --- |
| `ref.py` | The Boys function at 50 digits, by two independent routes (the integral by quadrature, and the all-positive convergent series), cross-validated against each other. |
| `parse_tables.py` | Reads `include/boys/boys_coefficients.hpp` as data: per order, the region-A pieces `(a, b, degree, coefficients)`. Never imports the library. |
| `modes.py` | The modes. Operand rounding by round-to-nearest-even on the significand with fp16's exponent range enforced, exact products, one rounding per accumulation step, and both degree-reduction orders. |
| `measure.py` | The delivered error of the region-A transform per mode, over both bands and every order, against the 50-digit reference. Writes `res_double.json`, `res_float.json`; its printed table is `measure2.log`. |
| `bound.py` | The bound each mode could claim per order count, folding in the recursion's own rounding taken from the shipped library's values on the same grid. |
| `contract.py` | What a mode delivers *after* the recursion it feeds, reproducing each shipped dispatch path with the mode's transform in place of the shipped one. |
| `scheme.py` | The shipped split Clenshaw against the GEMM-shaped dot product at the same precision, so the mode's precision and the evaluation scheme can be told apart. |
| `share.py` | The Chebyshev transform's share of a ladder's fused-op cost, counted from the shipped dispatch. |
| `workload.py` | That share weighted by a measured Boys-call workload. The call counts and the ladder are copied in from qcx's `boys_argument_probe` and `boys_argument_scaling_probe`; they are not regenerated here. |
| `compare.py` | The published bounds against the measurement — the entry point. |
| `dump_values.cpp` | Cross-check driver: the shipped library's region-A values, both lanes, on an argument list from a file. |

## The data

| File | What it is |
| --- | --- |
| `shipped_grid.txt` | The shipped library's region-A values on `xgrid.txt`: 7 order counts × 61 arguments × both lanes. Produced by `dump_values`. |
| `shipped_values.txt` | The same dump at a coarser argument set and orders 8, 16 and 32. |
| `xgrid.txt` | The argument list `shipped_grid.txt` was taken on. |
| `res_double.json`, `res_float.json` | `measure.py`'s raw output, keyed `mode\|band\|order` → the worst absolute and relative error and the argument each was found at. |
| `measure.log`, `measure2.log` | `measure.py`'s printed table, before and after the mode set was extended. |
| `ref_cache.json` | The 50-digit references, cached across runs. Not committed. |

## Building the cross-check driver

```
cmake -S . -B build
cmake --build build --target boys
cmake -S tools_tc/driver -B build/tools_tc-driver
cmake --build build/tools_tc-driver
build/tools_tc-driver/dump_values tools_tc/xgrid.txt > tools_tc/shipped_grid.txt
```

The dump is byte-identical to the committed `shipped_grid.txt` apart from the
line endings, so the pinned grid is what the current library produces.
