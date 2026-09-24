"""How much of the shipped region-A work is the Chebyshev transform?

Counted from the shipped dispatch, not estimated. For a ladder of orders
0..nmax at argument x the double lane (m = 1) does one of three things in
region A, and this counts the arithmetic of each:

  x < kTierThresholds[0]          one transform (order nmax) + nmax downward
                                  recursion steps
  x >= kTierThresholds[0]         one extended-band seed + `served` upward
                                  recursion steps + (nmax - served) per-order
                                  transforms
  x >= kX0                        no region-A transform at all

Units are fused ops: one fma, multiply, add or divide each counts 1, and a
ffma counts 2. That flatters the recursion, which is division-heavy (a
division costs several times a multiply on every mainstream core), so the
transform's share below is a floor rather than a ceiling.

The GEMM framing's own cost is counted too, because it is not free: a tensor
core path must build the basis matrix T for the band (one fma per basis value
per argument, shared by the orders the band serves) before the product.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parse_tables as P   # noqa: E402

X0 = 11.899848152108484
REPEAT = 1  # per ladder step below


def clenshaw_cost(deg):
    """Fused ops of the shipped ClenshawSplit: the split recurrence plus t, v."""
    if deg == 0:
        return 0
    if deg == 1:
        return 3          # fma + the t computation
    if deg == 2:
        return 6
    m = deg // 2
    loops = (m - 1) + (m - 2)
    tail = 6              # even, odd, final fma, v, two_v, t
    head = 3              # t: sub, div, sub
    return loops + tail + head


def recursion_step(x, up):
    """(l -+ 1/2) * f -+ expx) / x  ->  one mul, one add, one divide."""
    return 3


def transform_cost(deg_k, orders_touched, build_basis=True):
    """The GEMM shape: build T once per band, then one dot per order."""
    basis = (deg_k + 1) if build_basis else 0
    return basis + orders_touched * (deg_k + 1), basis


def ladder(x, nmax, thresholds, x0, x1, ext_x0):
    """Fused-op counts of the shipped double m = 1 path for one ladder."""
    if x == 0.0:
        return dict(transform=0.0, recursion=0.0, other=0.0, n_tf=0, mmax=0,
                    band=None, served=0)
    if x >= x1:
        return dict(transform=0.0, recursion=nmax * 2, other=1.0, n_tf=0, mmax=0,
                    band="C", served=0)
    if x >= x0:
        deg_k = 19                      # region-B F0 fit, deg 18
        tf = clenshaw_cost(18 - 1 + 1) if False else clenshaw_cost(18)
        return dict(transform=tf, recursion=nmax * 2, other=1.0, n_tf=1, mmax=1,
                    band="B", served=nmax)
    served = 0
    while served < nmax and x >= thresholds[served + 1]:
        served += 1
    if x >= thresholds[0]:
        # extended seed (deg 24 on the extended band) + upward recursion +
        # per-order transforms for the orders the tier does not reach
        seed = clenshaw_cost(24)
        rec = served * 2
        n_tf = nmax - served
        deg_k = 20 if x < x0 / 2 else 18
        tf = n_tf * clenshaw_cost(deg_k)
        return dict(transform=tf, recursion=rec, other=seed + 1.0, n_tf=n_tf,
                    mmax=n_tf, band="A-ext", served=served)
    # the downward shape: one transform carries the whole ladder
    deg_k = 20 if x < x0 / 2 else 18
    return dict(transform=clenshaw_cost(deg_k), recursion=nmax * 2, other=1.0,
                n_tf=1, mmax=1, band="A-down", served=0)


def main():
    T = P.load()
    th = T["double"]["thresholds"]
    for nmax in (0, 1, 2, 4, 8, 16, 32):
        xs = np.concatenate([np.linspace(1e-6, X0, 4001, endpoint=False),
                             np.linspace(X0, 30, 1001)])
        rows = [ladder(float(x), nmax, th, X0, 28.989337738820740,
                       T["double"]["ext"][2]) for x in xs]
        for i, r in enumerate(rows):
            tot = r["transform"] + r["recursion"] + r["other"]
            r["share"] = r["transform"] / tot if tot else 0.0
        print(f"nmax={nmax:2d}: transforms per ladder "
              f"{min(r['n_tf'] for r in rows)}..{max(r['n_tf'] for r in rows)}; "
              f"M (orders per band) max {max(r['mmax'] for r in rows)}")
        # share over a uniform argument grid in region A only (stated
        # distribution: uniform x, not a workload)
        ra = [r for r in rows if r["band"] in ("A-down", "A-ext")]
        if nmax >= 8:
            for lo, hi, lbl in ((0.0, 1.0855, "x<1.0855 (down)"),
                                (1.0855, X0 / 2, "1.0855..5.95 (ext)"),
                                (X0 / 2, 10.7836, "5.95..10.78 (ext)")):
                sel = [r for r in rows if lo <= xs[rows.index(r)] < hi]
                if sel:
                    print(f"    {lbl:24s} share {min(r['share'] for r in sel):.2f}"
                          f"..{max(r['share'] for r in sel):.2f}")
        if nmax == 32:
            for xv in (0.5, 1.5, 3.0, 6.0, 10.0, 11.0, 11.8, 15.0, 40.0):
                i = int(np.argmin(np.abs(xs - xv)))
                r = rows[i]
                print(f"    x={xv:6.2f} band={r['band']:7s} served={r['served']:2d} "
                      f"n_tf={r['n_tf']:2d} transform={r['transform']:6.1f} "
                      f"recursion={r['recursion']:5.1f} other={r['other']:5.1f} "
                      f"share={r['share']:.3f}")
    # The two shapes' GEMM tiles
    print("\nthe product's shape, per band, nmax = 32 (M x K, K = deg + 1):")
    for xv in (0.5, 1.5, 3.0, 6.0, 10.0, 11.0):
        r = ladder(xv, 32, th, X0, 28.989337738820740, T["double"]["ext"][2])
        k = 21 if xv < X0 / 2 else 19
        print(f"  x={xv:5.2f}: M={r['mmax']:2d} K={k:2d} band={r['band']}")


if __name__ == "__main__":
    main()
