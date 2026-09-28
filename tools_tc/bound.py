"""The delivered bound each mode could honestly claim, per order count.

Three measured pieces, none of them a theory:

  E(mode, n, x)      the transform's own error at order n and argument x, from
                     simulating the mode's arithmetic - the operands are
                     rounded, the products are exact, the accumulation rounds
                     as that mode's does;
  R(lane, nmax, x)   the recursion's own rounding, taken from the SHIPPED
                     library's values on the same grid (a double seed through
                     the float recursion, so what is left is the recursion);
  the amplification, the shipped design's own seed weight: prod x/(j + 1/2)
                     downward, prod (j - 1/2)/x upward.

Two shapes, as the shipped dispatch has them. Below kTierThresholds[0] one
transform carries the whole ladder, so its error arrives at order 0 multiplied
by the full weight. At or above it the extended F0 seed carries the orders a
recursion reaches and the rest are transform values delivered as they stand.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as ME   # noqa: E402
import modes as M   # noqa: E402
import parse_tables as P   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
X0 = 11.899848152108484
BUDGET = {"D": 1e-15, "F": 1.5e-7}
NMAXES = (0, 1, 2, 4, 8, 16, 32)


def weight_down(N, x):
    p = 1.0
    for j in range(N):
        p *= x / (j + 0.5)
    return max(1.0, p)


def weight_up(n, x):
    p = 1.0
    for j in range(1, n + 1):
        p *= (j - 0.5) / x
    return max(1.0, p)


def load_grid(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            lane, nmax, x, n, v = line.split()
            out.setdefault((lane, int(nmax), float(x)), {})[int(n)] = float(v)
    return out


def piece(L, n, x, extended):
    if extended and x >= L["ext"][2]:
        return (L["ext"][2], X0, L["ext"][0], L["ext"][1])
    for (a, b, d, cs) in L["orders"][n]:
        if x < b:
            return (a, b, d, cs)
    return L["orders"][n][-1]


def transform(L, n, x, mode, extended=False):
    a, b, d, cs = piece(L, n, x, extended)
    T = ME.basis(np.array([2.0 * (x - a) / (b - a) - 1.0]), len(cs))
    return float(M.gemm(np.asarray(cs, dtype=np.float64)[None, :], T, mode)[0][0])


def main():
    tabs = P.load()
    Ld, th = tabs["double"], tabs["double"]["thresholds"]
    grid = load_grid(os.path.join(HERE, "shipped_grid.txt"))
    xs = sorted({x for (_, _, x) in grid})
    cache = ME.build_refs([(n, float(x)) for x in xs for n in range(33)])
    ref = lambda n, x: float(ME.ref_value(cache, n, float(x)))
    modes = ["fp64", "fp32", "tf32", "tf32x3", "bf16", "bf16x6", "fp16", "fp16acc16"]
    rows = {}
    for lane in ("F", "D"):
        for (lg, nmax, x) in [k for k in grid if k[0] == lane]:
            R = max(abs(grid[(lane, nmax, x)][n] - ref(n, x)) for n in range(nmax + 1))
            extended = (lane == "D") and x >= th[0]
            served = 0
            if extended:
                while served < nmax and x >= th[served + 1]:
                    served += 1
            for mode in modes:
                if extended:
                    e0 = abs(transform(Ld, 0, x, mode, True) - ref(0, x))
                    seed_l = max((abs(transform(Ld, l, x, mode) - ref(l, x))
                                  for l in range(served + 1, nmax + 1)), default=0.0)
                    worst = max(e0 * weight_up(served, x), seed_l)
                else:
                    e = abs(transform(Ld, nmax, x, mode) - ref(nmax, x))
                    worst = e * weight_down(nmax, x)
                cur = rows.setdefault((lane, mode, nmax), 0.0)
                rows[(lane, mode, nmax)] = max(cur, worst + R)
    for lane in ("D", "F"):
        print(f"=== lane {lane}, budget {BUDGET[lane]:g}: bound / budget, worst over "
              f"the argument grid")
        print(f"{'mode':12s} " + " ".join(f"nmax={n:<3d}" for n in NMAXES))
        for mode in modes:
            cells = [rows.get((lane, mode, n), 0.0) / BUDGET[lane] for n in NMAXES]
            print(f"{mode:12s} " + " ".join(f"{c:8.2f}" for c in cells))


if __name__ == "__main__":
    main()
