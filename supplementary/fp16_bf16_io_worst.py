#!/usr/bin/env python3
"""fp16/bf16 I/O-only lane worst errors — the tab:accuracy worst cells.

Regenerates the fp16/bf16 worst cells of the accompanying manuscript's
tab:accuracy (2.4e-4 for the fp16 rows, 1.9e-3 for the bf16 rows, measured
on the n = 0..32, x in [0, 100] reference grid) from the shipped 30-digit
reference grid (boys_reference.csv).

The fp16/bf16 lanes are I/O-only: the argument is rounded to the half
type, evaluated in the certified float engine, and the result rounded back
to the half type; half never takes part in arithmetic. The accuracy
contract is the certified float bound (1.5e-7; the in-repo suite asserts
the strictly stronger 1e-7 base of the D-F2 fp16 formula — whose 1e-7
is the fp16 lanes' asserted base, not the float budget) plus one half-ULP
of representation of the rounded reference value. Because the measured
worst errors are representation-dominated (the half-ULP term, ~1e-3..1e-4,
dwarfs the ~1e-7 asserted engine base), an exact-arithmetic emulation of
the I/O rounding reproduces the measured worst cells to the printed
precision without running the float engine itself.

For each grid row (n, x): xh = round-to-half(x), v = F_n(xh) at 30 digits
(the certified double lane's reference, whose ~5e-14 error is immaterial
at this scale), r = round-to-half(v); the error is |r - v|. Regions are
the manuscript's, bucketed by the grid x: A = [0, x0), B = [x0, x1),
C = [x1, inf) with the fixed boundaries x0 = 11.899848152108484,
x1 = 28.989337738820740. The comparison against the manuscript cells is
at the cells' stated precision (2 significant digits).

Usage:  python fp16_bf16_io_worst.py
Requires: Python 3 + mpmath only (pip install mpmath).
"""

import math

from mpmath import mp, mpf, exp

mp.dps = 30

X0 = mpf("11.899848152108484")
X1 = mpf("28.989337738820740")


def round_fp16(v):
    """Round to nearest IEEE-754 binary16 (11 significant bits)."""
    with mp.workprec(11):
        return mpf(v)


def round_bf16(v):
    """Round to nearest bfloat16 (8 significant bits)."""
    with mp.workprec(8):
        return mpf(v)


def boys_ref(n, x):
    """F_n(x) via the provably stable series (V&S eq. 26), 30 digits —
    the same series the shipped reference grid is generated from."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    s = mpf(0)
    term = mpf(1) / (n + mpf("0.5"))
    e = exp(-x) / 2
    for l in range(300):
        s += term
        if l > 20 and term < mpf("1e-26"):
            break
        term *= x / (n + l + mpf("1.5"))
    return e * s


def load_grid(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        assert header.strip() == "n,x,value", header
        for line in f:
            n, x, v = line.strip().split(",")
            rows.append((int(n), mpf(x), mpf(v)))
    return rows


def region(x):
    if x < X0:
        return "A"
    if x < X1:
        return "B"
    return "C"


def worst_per_region(rows, rounding):
    worst = {"A": (mpf(0), None), "B": (mpf(0), None), "C": (mpf(0), None)}
    overall = (mpf(0), None)
    for n, x, _ in rows:
        xh = rounding(x)
        v = boys_ref(n, xh)
        r = rounding(v)
        err = abs(r - v)
        if err > overall[0]:
            overall = (err, (n, x, xh, v, r))
        key = region(x)
        if err > worst[key][0]:
            worst[key] = (err, (n, x, xh, v, r))
    return worst, overall


def fmt4(v):
    """4 significant digits, fixed notation."""
    return mp.nstr(v, 4, min_fixed=0, max_fixed=0)


def round2(value):
    """Round to 2 significant digits (the cells' stated precision)."""
    if value == 0:
        return 0.0
    exp10 = math.floor(math.log10(abs(value)))
    scaled = value / 10 ** exp10
    return round(scaled, 1) * 10 ** exp10


def main():
    rows = load_grid("boys_reference.csv")
    print("fp16/bf16 I/O-only lane worst errors (tab:accuracy worst cells)")
    print("  grid: boys_reference.csv, n = 0..32 x %d x-values in [0, 100];"
          % len(set(x for _, x, _ in rows)))
    print("  reference: the 30-digit stable series at the half-rounded argument")
    print()
    failures = 0

    for name, rounding, paper_cell in (
        ("fp16", round_fp16, 2.4e-4),
        ("bf16", round_bf16, 1.9e-3),
    ):
        worst, overall = worst_per_region(rows, rounding)
        print("  %s rows (paper worst cell: %.1e):" % (name, paper_cell))
        for key in ("A", "B", "C"):
            err, where = worst[key]
            print("    region %s: worst |err| = %s at (n=%d, x=%s)"
                  % (key, fmt4(err), where[0], mp.nstr(where[1], 10)))
        err, where = overall
        print("    overall worst |err| = %s at (n=%d, x=%s)"
              % (fmt4(err), where[0], mp.nstr(where[1], 10)))
        match = round2(float(err)) == round2(paper_cell)
        failures += 0 if match else 1
        print("    vs the manuscript cell %.1e: %s"
              % (paper_cell, "MATCH" if match else "MISMATCH"))
        print()

    print("RESULT: %s" % ("CELLS MATCH" if failures == 0 else
                          "%d cell(s) MISMATCH" % failures))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
