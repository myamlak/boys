# Half-lane exhaustive sweep

Re-derivation of the two published half-lane margins. Run on 27 October 2026 against the
library at tag `v5.1.1`, commit `834bdbb18a30ac8538b519436429f78d41f187ce`, working tree
clean, default build options.

## What is measured

- **Lane entries.** `boys::BoysSingleF16(n, boys::F16)` and `boys::BoysSingleBf16(n, boys::Bf16)`,
  the scalar store-half entries. The gate measures them in `tests/boys_accuracy_gate.cpp`, block
  "fp16 and bf16, the store-half lane": the rows `kF16Single` and `kBf16Single` call exactly these
  two functions, once per `(n, x)`. The batch rows `kF16Orders`/`kBf16Orders` reach the same
  arithmetic through `BoysAllOrdersF16`/`BoysAllOrdersBf16`. This sweep calls the single entry, and
  it reproduces the record's margins (below).
- **Bound.** `HalfBound(got, mantissaBits, minNormalExp) = kBoundHalfBase + 0.5 * UlpOf(got, ...)`
  from `tests/boys_gate_reference.hpp:341`, with `kBoundHalfBase = 1e-7`. The format constants are
  the gate's: `(10, -14)` for binary16 (`boys_gate_reference.hpp:29`) and `(7, -126)` for bfloat16
  (`boys_accuracy_gate.cpp:220`). The claim slots are created with the default absolute
  `BoundForm` — no additive term, no floor — so a cell's budget is exactly `HalfBound(got)` at the
  returned value.
- **Arguments and orders.** Every positive finite bit pattern: `0x0001..0x7BFF` (binary16) and
  `0x0001..0x7F7F` (bfloat16), i.e. zero, infinity and the NaNs excluded, 64,382 arguments; orders
  0..32.

## Counts

| format | arguments | cells |
| --- | --- | --- |
| binary16 | 31,743 | 1,047,519 |
| bfloat16 | 32,639 | 1,077,087 |
| total | 64,382 | 2,124,606 |

## Reference

45 decimal digits, from the boys generator
(`external/boys/tools/gen_boys_coefficients.py`): `boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS)`,
the all-positive series, with the argument passed as its exact double.

That series needs roughly `x` terms and refuses past its cap. With `RAT_TERMS = 900` it converges
at x = 250 and refuses at x = 300, while the sweep runs to 65504 (binary16) and 3.3895e38
(bfloat16). Above x = 250 the same file's identity is used at the same precision,

    F_n(x) = [Gamma(n+1/2) - Gamma(n+1/2, x)] / (2 x^(n+1/2)),

which is `closed_form_error`'s own expression. The substitution is measured, not assumed: the two
routes agree to at most 1.9e-45 relative at x = 100, 150, 200, 250 for n = 0, 16, 32 (recorded in
`results.json`), and the series with a raised term cap extends that comparison to x = 2000 at the
same level. Above 2000 the correction term is negligible and the identity was checked against the
asymptotic expansion of `Gamma(s, x)` at 45 digits. 778,932 cells used the identity (271,887
binary16, 507,045 bfloat16); the rest used the series.

## Result

Worst delivered error as a fraction of the lane's budget:

| format | worst ratio | margin | worst cell |
| --- | --- | --- | --- |
| binary16 | 0.999299207483 | 0.0700792517 per cent | n=0, x=0x1B82 (0.003665924072265625), delivered 0.9990234375, true 0.998779368037002, bound 0.000244240625 |
| bfloat16 | 0.999915078150 | 0.0084921850 per cent | n=0, x=0x3FA4 (1.28125), delivered 0.69921875, true 0.697265690871480, bound 0.001953225 |

Both worst cells are correctly rounded returns landing just above a rounding midpoint: the
binary16 true value sits 7.1e-8 above the midpoint between 0.99853515625 and 0.9990234375, the
bfloat16 one 6.6e-8 above the midpoint between 0.6953125 and 0.69921875. Each maximum is unique
to 1e-9; no cell in either format exceeds its bound, and no return is non-finite.

## Agreement with the audit record

| figure | record | measured | difference |
| --- | --- | --- | --- |
| binary16 worst ratio | 0.999299207 | 0.999299207483 | +4.8e-10 |
| binary16 margin | 0.070079 per cent | 0.0700792517 per cent | +2.5e-7 |
| bfloat16 worst ratio | 0.999915078 | 0.999915078150 | +1.5e-10 |
| bfloat16 margin | 0.008492 per cent | 0.0084921850 per cent | +1.9e-7 |

The record's figures are these values rounded to the digits it states; the counts agree exactly.
The record names no worst cell, so the cells above are this sweep's own.

## Runs

Toolchain: MSVC 19.51.36256.0 (Visual Studio 18, toolset 14.51.36231), CMake 4.4.2;
Python 3.14.7, mpmath 1.4.1, numpy 2.5.2. Exact commands in `build.txt`.

| step | command | wall time | exit |
| --- | --- | --- | --- |
| configure | `cmake -S ... -B build ...` | 5.7 s | 0 |
| library | `cmake --build build --config Release --target boys` | (see note) | 0 |
| harness | `cl ... half_sweep.cpp ...\boys.lib /Fe:half_sweep.exe` | — | 0 |
| sweep | `half_sweep.exe half_sweep.bin` | 0.96 s | 0 |
| reference | `python half_sweep_oracle.py half_sweep.bin results.json --jobs 8` | 309.1 s (reference 308.1 s) | 0 |

The two entries are header-defined, so the harness also links without `boys.lib`; both links
produce byte-identical sweep files (`cmp`, exit 0).

Note: the library build was run alongside an unrelated `boys-accuracy-gate` build already running
on the machine, so its duration is not a clean figure and is not quoted.

## Files

- `half_sweep.cpp` — the harness: enumerates the arguments, calls the two entries, writes one row
  per cell.
- `half_sweep_oracle.py` — reads the sweep, computes the 45-digit reference, the per-cell ratio,
  the worst cell and the margin per format; writes `results.json`.
- `results.json` — counts, worst cells, per-order worst, margins, the route each cell took, and
  the series-versus-identity agreement.
- `build.txt` — exact build and run commands.
- `README.md` — this file.

The sweep file itself (46,741,352 bytes) is not deposited; `build.txt` regenerates it.
