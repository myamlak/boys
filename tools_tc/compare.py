"""The three shipped transform bounds against this harness's own measurement.

The library publishes one delivered figure per mode - kFp64, kTf32x3,
kBf16x6 - as the worst |C.T - F_n(x)| over region A: both bands, every order
0..32, every argument of the band. This reproduces that quantity from the
simulated arithmetic and prints it beside the published one. The published
figures are parsed out of tests/boys_accuracy_gate.cpp, the gate that enforces
them, rather than restated here, so the two cannot drift apart.

The reduction order is measured both ways, because it is worth three times.
The shipped kernel lays a pass's degrees out from the highest down and reduces
them pairwise before folding them into the total; gemm's default sums the
degrees into one running total from the constant term up. The published
figures are the pairwise family - reading a "running" row against them is
reading two different layouts.

Density is what the two split modes' published figures were taken at: the
header names a 200000-point sweep of the lower band's left end, and the
default 129 points a piece reads low by about 1.6 times at those modes. So
the plain run reports pairwise ratios under one for them, and it is the
density and not the arithmetic. --dense N sweeps the lower band at N points a
piece and is the run that reproduces them; at 1000000 both agree with the
published figures to their own four digits.

The dense sweep takes its reference from a vectorised double-precision series
instead of mpmath, because the cached 25-digit grid does not extend to a
million arguments. That reference is exact to about 1e-14 relative, eight
orders tighter than the 1.9e-07 it is measuring, and nowhere near tight enough
for kFp64's 1.1e-16 - which is why kFp64 is reported from the cached grid and
not from the dense sweep.

Usage:
    python tools_tc/compare.py              # the grid, all three modes, both layouts
    python tools_tc/compare.py --dense N    # the lower band at N points a piece
"""
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as ME   # noqa: E402
import modes as M   # noqa: E402
import parse_tables as P   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GATE = os.path.join(HERE, os.pardir, "tests", "boys_accuracy_gate.cpp")
CLAIM = {"fp64": "Fp64", "tf32x3": "Tf32x3", "bf16x6": "Bf16x6"}
SPLIT = ("tf32x3", "bf16x6")
# The single-pass narrow modes. They are not part of CLAIM: nothing in the gate
# publishes a figure for them, so there is no name to read a published number
# out of and no ratio to print. This script is where their bounds are measured,
# and the dense path below is the one that measures them - the grid reads low
# at these modes for the reason it reads low at the split ones.
SINGLE_PASS = ("tf32", "bf16", "fp16")


def published():
    """The delivered worst each mode claims, read out of the gate."""
    with open(GATE, "r", encoding="utf-8") as f:
        src = f.read()
    out = {}
    for name, value in re.findall(r"kTransform(\w+)Delivered = ([0-9.eE+-]+);", src):
        out[name] = float(value)
    if len(out) != len(CLAIM):
        raise SystemExit(f"expected three delivered figures in {GATE}, found {sorted(out)}")
    return out


def worst_over_region_a(mode, reduction, tables, npts=ME.NPTS):
    """Worst |C.T - F_n(x)| over both bands, every order, every argument."""
    L = tables["double"]
    pieces = [(n, a, b, cs, np.linspace(a, b, npts))
              for n, ps in enumerate(L["orders"]) for (a, b, _, cs) in ps]
    cache = ME.build_refs([(n, float(x)) for (n, _, _, _, xs) in pieces for x in xs])
    worst = 0.0
    for (n, a, b, cs, xs) in pieces:
        T = ME.basis(2.0 * (xs - a) / (b - a) - 1.0, len(cs))
        got = M.gemm(np.asarray(cs, dtype=np.float64)[None, :], T, mode, reduction)[0]
        r = np.array([float(ME.ref_value(cache, n, float(x))) for x in xs])
        worst = max(worst, float(np.abs(got - r).max()))
    return worst


def series(n, x, tail=1e-30, max_terms=4000):
    """The convergent series in double: all terms positive, rel err ~1e-14."""
    x = np.asarray(x, dtype=np.float64)
    total = np.zeros_like(x)
    term = np.full_like(x, 1.0 / (n + 0.5))
    for l in range(max_terms):
        total += term
        if l > 20 and np.all(term < tail):
            break
        term = term * x / (n + l + 1.5)
    return np.exp(-x) / 2 * total


def worst_lower_band(mode, tables, npts):
    """Worst over the lower band's pieces, pairwise, on a dense argument set."""
    L = tables["double"]
    worst = 0.0
    for n, ps in enumerate(L["orders"]):
        (a, b, _, cs) = ps[0]                      # the lower band's fit
        xs = np.linspace(a, b, npts, endpoint=False)
        T = ME.basis(2.0 * (xs - a) / (b - a) - 1.0, len(cs))
        got = M.gemm(np.asarray(cs, dtype=np.float64)[None, :], T, mode, "pairwise")[0]
        worst = max(worst, float(np.abs(got - series(n, xs)).max()))
    return worst


def main():
    argv = sys.argv[1:]
    dense = 0
    if argv and argv[0] == "--dense":
        dense = int(argv[1])
    claim = published()
    tables = P.load()
    if dense:
        print(f"worst |C.T - F_n(x)| over the lower band, {dense} points a piece, pairwise")
        print("reduction, against a double-precision series (exact to ~1e-14).")
        print(f"{'mode':10s} {'measured':>12s} {'published':>12s} {'ratio':>8s}")
        for mode in SPLIT:
            got = worst_lower_band(mode, tables, dense)
            print(f"{mode:10s} {got:12.4e} {claim[CLAIM[mode]]:12.4e} {got / claim[CLAIM[mode]]:8.3f}")
        # The single-pass modes carry no published figure: their bound is what
        # this measurement is for, so the row prints the measurement alone.
        print("the single-pass narrow modes (no published figure; this is the measurement):")
        for mode in SINGLE_PASS:
            got = worst_lower_band(mode, tables, dense)
            print(f"{mode:10s} {got:12.4e} {'-':>12s} {'-':>8s}")
        return
    print("worst |C.T - F_n(x)| over region A, from the simulated arithmetic, beside the")
    print(f"figure tests/boys_accuracy_gate.cpp holds a sweep to ({ME.NPTS} points a piece;")
    print("the published split-mode figures are dense sweeps, so they read high - try --dense).")
    print()
    print(f"{'mode':10s} {'reduction':10s} {'measured':>12s} {'published':>12s} {'ratio':>8s}")
    for mode in CLAIM:
        for reduction in ("pairwise", "running"):
            got = worst_over_region_a(mode, reduction, tables)
            pub = claim[CLAIM[mode]]
            print(f"{mode:10s} {reduction:10s} {got:12.4e} {pub:12.4e} {got / pub:8.3f}")
    for mode in SINGLE_PASS:
        got = worst_over_region_a(mode, "pairwise", tables)
        print(f"{mode:10s} {'pairwise':10s} {got:12.4e} {'-':>12s} {'-':>8s}")


if __name__ == "__main__":
    main()
