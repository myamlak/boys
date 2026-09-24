"""The transform's share of the library's work, weighted by a MEASURED workload.

Counted, not estimated. Two measured inputs:

  * the library's dispatch, counted from the shipped source (share.py): the
    fused-op cost of a ladder and how much of it is the Chebyshev transform,
    as a function of (nmax, x);
  * the workload: qcx's benchmarks/boys_argument_probe.cpp at C24H50/def2-SVP
    (586 basis functions, 965,035,861 Boys calls), which reports the calls per
    order L and per DISPATCH PATH - zero, A-table, A-recur, A-mixed, B, C -
    using this library's own region boundaries, plus an x histogram. The
    per-size ladder comes from benchmarks/boys_argument_scaling_probe.cpp.

The dispatch path is what makes this exact: an A-table call is x below
kTierThresholds[0] (one transform carries the ladder, M = 1 by construction),
and an A-recur call is x at or above it, where the extended F0 seed covers the
orders a tier reaches and only the orders above it are transform values. The
only quantity the histogram has to supply is how much of A-recur falls below
2.0153, which decides whether those calls have any transform rows at all; both
extremes are reported.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import share as S   # noqa: E402
import parse_tables as P   # noqa: E402

X0 = 11.899848152108484
X1 = 28.989337738820740
# C24H50 / def2-SVP: Boys calls by order L and dispatch path (the probe's own
# classification, which is the library's own).
PATH = {  # L: (zero, A-table, A-recur, A-mixed, B, C)
    0: (35737, 2807485, 14408932, 0, 12273800, 172316704),
    1: (10961, 4501659, 25102322, 0, 22788936, 281308986),
    2: (30944, 3509301, 21011746, 0, 19053055, 216685844),
    3: (7512, 1638672, 10767146, 0, 9260506, 100729700),
    4: (11741, 512894, 3600425, 0, 2893107, 30678230),
    5: (1222, 114234, 705722, 98612, 604158, 6353586),
    6: (1797, 18170, 102946, 14922, 84724, 890791),
    7: (46, 2108, 8486, 1414, 7280, 75434),
    8: (87, 3, 342, 174, 305, 2953),
}
LADDER = [(1, 34, 74.3758), (4, 106, 39.9000), (8, 202, 25.1300),
          (24, 586, 9.2150), (56, 1354, 3.9949)]
TIER0 = 1.0855252345349333
TIER1 = 2.015297705335114


def ops(r):
    return r["transform"] + r["recursion"] + r["other"]


def main():
    th = P.load()["double"]["thresholds"]
    total_calls = sum(sum(v) for v in PATH.values())
    num = den = 0.0
    num_hi = 0.0
    rows = []
    for L, (z, at, ar, am, b, c) in PATH.items():
        # A-table: x just below TIER0, so the downward shape, M = 1
        rd = S.ladder(TIER0 * 0.999, L, th, X0, X1, TIER0)
        # A-recur: the transform carries orders above the tier. Both extremes:
        # every A-recur call above TIER1 has no transform rows (served = nmax
        # for nmax <= 8); those below it have M = nmax - served rows.
        ru = S.ladder(np.sqrt(TIER0 * TIER1), L, th, X0, X1, TIER0)
        rv = S.ladder(TIER1 * 1.001, L, th, X0, X1, TIER0)
        rb = S.ladder(X0 * 1.5, L, th, X0, X1, TIER0)
        rc = S.ladder(200.0, L, th, X0, X1, TIER0)
        num += at * rd["transform"] + ar * ru["transform"] + am * ru["transform"]
        num_hi += at * rd["transform"] + ar * ru["transform"] + am * ru["transform"] \
            + (ar + am) * (max(rv["transform"], ru["transform"]) - ru["transform"])
        den += at * ops(rd) + (ar + am) * rv["transform"] + (ar + am) * (ops(ru) - ru["transform"]) \
            + b * ops(rb) + c * ops(rc) + z * 1.0
        rows.append((L, at, ar, rd["transform"] / ops(rd), ru["mmax"], ru["transform"] / ops(ru)))
    print(f"C24H50/def2-SVP, {total_calls} Boys calls "
          f"(qcx's boys_argument_probe, engine-cross-checked)")
    print(f"{'L':>2s} {'A-table calls':>14s} {'share/tf':>9s} {'A-recur calls':>14s} "
          f"{'M below 2.0153':>15s} {'share/tf':>9s}")
    for (L, at, ar, s_at, m, s_ar) in rows:
        print(f"{L:2d} {at:14d} {s_at:9.2f} {ar:14d} {m:15d} {s_ar:9.2f}")
    print(f"\n  transform share of ALL Boys work at C24H50: {100 * num / den:.3f}%"
          f"  (upper extreme, every A-recur call below 2.0153: {100 * num_hi / den:.3f}%)")
    print("  rows of the product: M = 0 or 1 for every call at L <= 4 (95.15% of calls); "
          "M <= 4 at L = 8")
    print("\n  upper bound per system size: the region-A share of calls x the best")
    print("  per-ladder share a transform can have (0.96, at nmax = 0):")
    for (c, nbf, a) in LADDER:
        print(f"    C{c}H{2 * c + 2}  {nbf:5d} BF: A = {a:6.2f}%  ->  <= {0.96 * a:.2f}% of work")
    print("\n  what the share is, by nmax, over region A (transform ops / ladder ops):")
    for L in (0, 1, 2, 4, 8, 16, 32):
        worst = max(S.ladder(float(x), L, th, X0, X1, TIER0)["transform"] /
                    ops(S.ladder(float(x), L, th, X0, X1, TIER0))
                    for x in np.linspace(1e-6, X0 - 1e-9, 301))
        print(f"    nmax={L:2d}: best case {worst:.2f}")


if __name__ == "__main__":
    main()
