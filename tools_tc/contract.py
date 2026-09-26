"""What a mode delivers AFTER the recursion it feeds, not just at the seed.

The transform in region A is a seed for a recursion, and the recursion
amplifies the seed's error. So the delivered error is not the transform's own
error: for the downward shape (x < kTierThresholds[0]) one transform carries
the whole ladder and the amplification is

    d F_l / d F_N = prod_{j=l}^{N-1} x/(j + 1/2),   maximised at l = 0,

which is the generator's own seed_weight(N, x). For the extended shape the
low orders come from the F0 seed through the upward recursion and the high
orders are transform values delivered as they stand.

This reproduces each shipped path with the mode's transform in place of the
shipped one and reports the worst delivered error over the orders and the
argument grid, beside the lane's budget.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as ME   # noqa: E402
import modes as M   # noqa: E402
import parse_tables as P   # noqa: E402

X0 = 11.899848152108484
X1 = 28.989337738820740
F32 = np.float32


def weight(N, x):
    """Downward seed amplification to order 0 (the generator's seed_weight)."""
    p = 1.0
    for j in range(N):
        p *= x / (j + 0.5)
    return max(1.0, p)


def up_weight(n, x):
    """Upward-recursion amplification of an F0 seed error to order n."""
    p = 1.0
    for j in range(1, n + 1):
        p *= (j - 0.5) / x
    return max(1.0, p)


class Lane:
    """The precision the recursion itself runs in, and the transform it seeds."""

    def __init__(self, name, lane_key, cast):
        self.name = name
        self.key = lane_key
        self.cast = cast


def _piece_for(L, n, x, with_extended):
    if with_extended and x >= L["ext"][2]:
        return (L["ext"][2], X0, L["ext"][0], L["ext"][1])
    for (a, b, d, cs) in L["orders"][n]:
        if x < b:
            return (a, b, d, cs)
    a, b, d, cs = L["orders"][n][-1]
    return (a, b, d, cs)


def transform(L, n, x, mode, with_extended=False):
    a, b, d, cs = _piece_for(L, n, x, with_extended)
    t = np.array([2.0 * (x - a) / (b - a) - 1.0])
    T = ME.basis(t, len(cs))
    C = np.asarray(cs, dtype=np.float64)[None, :]
    return float(M.gemm(C, T, mode)[0][0])


def ladder(L, nmax, x, mode, cast_fn, thresholds, extended, with_extended, Lseed=None):
    """Reproduce the shipped region-A path with the mode's transform.

    Lseed is the table the transform reads. It is the double lane's table for
    both lanes' shipped code: the float lane's all-orders entry seeds from
    ChebyshevValue(nmax, (double)x) and casts the result, so a tensor-core path
    replacing that seed works on double coefficients and only then rounds."""
    Lseed = Lseed if Lseed is not None else L
    out = np.zeros(nmax + 1)
    if extended and x >= thresholds[0]:
        served = 0
        while served < nmax and x >= thresholds[served + 1]:
            served += 1
        f = cast_fn(transform(Lseed, 0, x, mode, True))
        out[0] = float(f)
        expx = cast_fn(0.5 * np.exp(-x))
        for l in range(1, served + 1):
            f = cast_fn((cast_fn(l - 0.5) * f - expx) / cast_fn(x))
            out[l] = float(f)
        for l in range(served + 1, nmax + 1):
            out[l] = float(cast_fn(transform(Lseed, l, x, mode, False)))
        return out
    # the downward shape
    f = cast_fn(transform(Lseed, nmax, x, mode, False))
    out[nmax] = float(f)
    expx = cast_fn(0.5 * np.exp(-x))
    for l in range(nmax - 1, -1, -1):
        f = cast_fn((cast_fn(x) * f + expx) / cast_fn(l + 0.5))
        out[l] = float(f)
    return out


def run(lane_key, cast_fn, modes, nmaxes=(0, 1, 2, 4, 8, 16, 32), npts=61,
        extended=None, seed_lane="double"):
    tabs = P.load()
    L, Lseed = tabs[lane_key], tabs[seed_lane]
    thresholds = L["thresholds"] or [1.0855252345349333] * 33
    with_extended = extended if extended is not None else bool(L["thresholds"])
    xs = np.linspace(1e-6, X0 - 1e-9, npts, endpoint=False)
    cache = ME.build_refs([(n, float(x)) for x in xs for n in range(33)])
    res = {}
    for nmax in nmaxes:
        for x in xs:
            rmax = {n: float(ME.ref_value(cache, n, float(x))) for n in range(nmax + 1)}
            for mode in modes:
                got = ladder(L, nmax, float(x), mode, cast_fn, thresholds,
                             with_extended, with_extended, Lseed)
                err = max(abs(got[n] - rmax[n]) for n in range(nmax + 1))
                cur = res.setdefault((mode, nmax), [0.0, 0.0])
                if err > cur[0]:
                    cur[0], cur[1] = err, float(x)
    return res


if __name__ == "__main__":
    modes = ["fp64", "fp32", "tf32", "tf32x3", "bf16", "bf16x6", "fp16"]
    print("=== double lane, recursion in fp64, budget 1e-15  (delivered, worst over "
          "orders and x in region A)")
    res = run("double", lambda v: float(v), modes)
    for (mode, nmax), v in sorted(res.items()):
        if nmax in (0, 1, 2, 4, 8, 16, 32):
            print(f"  {mode:8s} nmax={nmax:2d}  delivered {v[0]:.3e} "
                  f"(x={v[1]:.4f})  budget_ratio {v[0] / 1e-15:.3e}")
    print("\n=== float lane, recursion in fp32, budget 1.5e-7")
    res = run("float", F32, modes, extended=False)
    for (mode, nmax), v in sorted(res.items()):
        if nmax in (0, 1, 2, 4, 8, 16, 32):
            print(f"  {mode:8s} nmax={nmax:2d}  delivered {v[0]:.3e} "
                  f"(x={v[1]:.4f})  budget_ratio {v[0] / 1.5e-7:.3e}")
