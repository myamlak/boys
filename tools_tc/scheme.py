"""The shipped evaluation scheme against the GEMM shape, at the SAME precision.

The library evaluates a band by a split Clenshaw recurrence (even/odd split of
T_k(t) = T_k(v) for even k, t*D_j(v) for odd k). A tensor-core path replaces
that with a dot product C . T_k(t) - the GEMM shape. Those are two different
rounding behaviours at the same precision, and this measures the gap, so that
"the mode's precision" and "the evaluation scheme" can be told apart.

Both schemes are reproduced exactly as the shipped code has them: float32
arithmetic with fmaf (computed as fma in double, then rounded - which is a
correctly rounded f32 fma, since binary64 has more than 2p+2 bits over
binary32), fp64 arithmetic with fma.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as ME   # noqa: E402
import parse_tables as P   # noqa: E402

F32 = np.float32


def _fmaf(a, b, c):
    """Correctly rounded float32 fma."""
    return F32(math.fma(float(a), float(b), float(c)))


def clenshaw_split(c, deg, t, rnd=None):
    """The shipped ClenshawSplit, in fp64 (rnd None) or fp32 (rnd F32)."""
    def R(x):
        return x if rnd is None else rnd(x)
    if deg == 0:
        return c[0]
    if deg == 1:
        return R(math.fma(t, c[1], c[0]))
    v = R(math.fma(2.0, t * t, -1.0))
    tv = R(v + v)
    if deg == 2:
        return R(math.fma(t, c[1], math.fma(v, c[2], c[0])))
    m = deg // 2
    b1, b2 = c[2 * m], R(0.0)
    for k in range(m - 1, 0, -1):
        b0 = R(math.fma(tv, b1, R(c[2 * k] - b2)))
        b2, b1 = b1, b0
    even = R(math.fma(v, b1, R(c[0] - b2)))
    o1, o2 = c[2 * m - 1], R(0.0)
    for k in range(m - 2, 0, -1):
        o0 = R(math.fma(tv, o1, R(c[2 * k + 1] - o2)))
        o2, o1 = o1, o0
    odd = R(math.fma(R(tv - 1.0), o1, R(c[1] - o2)))
    return R(math.fma(t, odd, even))


def run(lane, npts=ME.NPTS):
    tables = P.load()
    L = tables[lane]
    rnd = F32 if lane == "float" else None
    pieces = [(n, a, b, d, cs, np.linspace(a, b, npts))
              for n, ps in enumerate(L["orders"]) for (a, b, d, cs) in ps]
    cache = ME.build_refs([(n, float(x)) for (n, _, _, _, _, xs) in pieces for x in xs])
    res = {}
    for (n, a, b, d, cs, xs) in pieces:
        worst_c = worst_d = 0.0
        for x in xs:
            r = float(ME.ref_value(cache, n, float(x)))
            # the shipped t for each lane, in that lane's precision
            if rnd is None:
                t = 2.0 * (float(x) - a) / (b - a) - 1.0
                got = clenshaw_split([float(c) for c in cs], d, t, None)
            else:
                xf = F32(float(x))
                t = F32(F32(F32(2.0) * (xf - F32(a))) / (F32(b) - F32(a)) - F32(1.0))
                got = clenshaw_split([F32(float(c)) for c in cs], d, t, F32)
            worst_c = max(worst_c, abs(float(got) - r))
        K = len(cs)
        T = ME.basis(2.0 * (xs - a) / (b - a) - 1.0, K)
        mode = "fp64" if rnd is None else "fp32"
        got = __import__("modes").gemm(np.asarray(cs, dtype=np.float64)[None, :], T, mode)[0]
        rr = np.array([float(ME.ref_value(cache, n, float(x))) for x in xs])
        worst_d = float(np.abs(got - rr).max())
        band = ME.band_of(a, b)
        for k, val in (("clenshaw", worst_c), ("dot", worst_d)):
            cur = res.setdefault((k, band, n), [0.0, 0.0])
            if val > cur[0]:
                cur[0], cur[1] = val, float(xs[int(np.argmax(np.abs(got - rr)))] if k == "dot" else 0.0)
    return res


if __name__ == "__main__":
    for lane in ("double", "float"):
        res = run(lane)
        print(f"=== {lane}: shipped split Clenshaw vs the GEMM-shaped dot product, "
              f"same precision")
        for band in ("A1", "A2"):
            c = max((v[0] for (k, b, n), v in res.items() if k == "clenshaw" and b == band), default=0)
            d = max((v[0] for (k, b, n), v in res.items() if k == "dot" and b == band), default=0)
            print(f"  band {band}: clenshaw {c:.3e}   dot {d:.3e}   dot/clenshaw {d / c:.2f}x"
                  f"   (budget {ME.BUDGET[lane]:g})")
