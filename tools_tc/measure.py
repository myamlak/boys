"""Delivered error of the region-A Chebyshev transform, per tensor-core mode.

The transform is F_n(x) = sum_k c_{n,k} T_k(t(x)) on a band - one row of an
(orders x degrees) by (degrees x batch) product. This measures, for each
simulated mode, |C.T - F_n(x)| against the mpmath reference (ref.py), worst
over the band and over the orders that band serves.

Two calibrations bracket the table:

  * the shipped coefficients evaluated EXACTLY (mpmath, 50 digits, the stored
    doubles taken as exact reals) give the fit residual: what the polynomial
    itself delivers when the storage is free;
  * the same coefficients through an fp64 product give the delivered double
    lane, whose gap to the first number is the coefficient storage's cost.

Results are keyed (mode, band, order) -> [worst_abs, x_at_abs, worst_rel,
x_at_rel], and the bands partition region A.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import modes as M   # noqa: E402
import parse_tables as P   # noqa: E402
import ref   # noqa: E402

NPTS = 129
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "ref_cache.json")
X0 = 11.899848152108484

# The lane budgets the table is calibrated against (absolute, at m = 1):
# docs/lane-contract.md, region A.
BUDGET = {"double": 1e-15, "float": 1.5e-7}


def band_of(a, b, x0=X0):
    """Which half of region A a piece sits in."""
    return "A1" if b <= (x0 / 2.0) * (1 + 1e-6) else "A2"


def build_refs(pairs):
    """{(n, x): F_n(x)} at 50 digits, cached across runs."""
    cache = {}
    if os.path.exists(CACHE):
        with open(CACHE, "r", encoding="utf-8") as f:
            cache = json.load(f)
    todo = [p for p in pairs if f"{p[0]}|{p[1]!r}" not in cache]
    if todo:
        import mpmath as mp
        with mp.workdps(30):
            for (n, x) in todo:
                cache[f"{n}|{x!r}"] = mp.nstr(ref.F_series(n, x), 25)
        with open(CACHE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    return cache


def ref_value(cache, n, x):
    import mpmath as mp
    return mp.mpf(cache[f"{n}|{x!r}"])


def basis(t, K):
    """Chebyshev basis matrix (K x S), built in fp64 by the recurrence.

    Shared by every order on the band - that is what makes the product a GEMM
    rather than a set of independent dot products."""
    T = np.empty((K, t.size), dtype=np.float64)
    T[0] = 1.0
    if K > 1:
        T[1] = t
    for k in range(2, K):
        T[k] = 2.0 * t * T[k - 1] - T[k - 2]
    return T


def exact_polynomial(cs, a, b, xs):
    """The shipped polynomial evaluated exactly: the fit residual."""
    import mpmath as mp
    out = []
    with mp.workdps(50):
        for x in xs:
            t = mp.mpf(2.0) * (mp.mpf(x) - mp.mpf(a)) / (mp.mpf(b) - mp.mpf(a)) - 1
            t0, t1 = mp.mpf(1), t
            s = mp.mpf(cs[0])
            if len(cs) > 1:
                s += mp.mpf(cs[1]) * t1
            for k in range(2, len(cs)):
                t0, t1 = t1, 2 * t * t1 - t0
                s += mp.mpf(cs[k]) * t1
            out.append(float(s))
    return out


def _worst(cur, absd, reld, xs):
    i, j = int(absd.argmax()), int(np.nanargmax(reld))
    if absd[i] > cur[0]:
        cur[0], cur[1] = float(absd[i]), float(xs[i])
    if reld[j] > cur[2]:
        cur[2], cur[3] = float(reld[j]), float(xs[j])


def measure_lane(tables, lane, mode_names, npts=NPTS):
    L = tables[lane]
    pieces = []
    for n, ps in enumerate(L["orders"]):
        for (a, b, d, cs) in ps:
            pieces.append(dict(n=n, a=a, b=b, d=d, cs=cs, xs=np.linspace(a, b, npts)))
    cache = build_refs([(p["n"], float(x)) for p in pieces for x in p["xs"]])
    res = {}
    for p in pieces:
        n, a, b, cs, xs = p["n"], p["a"], p["b"], p["cs"], p["xs"]
        K = len(cs)
        t = 2.0 * (xs - a) / (b - a) - 1.0
        T = basis(t, K)
        r = np.array([float(ref_value(cache, n, float(x))) for x in xs])
        with np.errstate(divide="ignore", invalid="ignore"):
            den = np.abs(r)
        for mode in mode_names:
            got = M.gemm(np.asarray(cs, dtype=np.float64)[None, :], T, mode)[0]
            absd = np.abs(got - r)
            reld = np.where(den > 0, absd / den, np.inf)
            _worst(res.setdefault((mode, band_of(a, b), n), [0.0, 0.0, 0.0, 0.0]),
                   absd, reld, xs)
        nsub = min(33, npts)
        ex = np.array(exact_polynomial(cs, a, b, xs[:nsub]))
        d17 = np.abs(ex - r[:nsub])
        with np.errstate(divide="ignore", invalid="ignore"):
            rel17 = np.where(den[:nsub] > 0, d17 / den[:nsub], np.inf)
        _worst(res.setdefault(("fit-residual", band_of(a, b), n), [0.0, 0.0, 0.0, 0.0]),
               d17, rel17, xs[:nsub])
    return res


def summarise(res, band):
    """Worst over orders, per mode, on one band -> (abs, x, rel, x)."""
    out = {}
    for (mode, bd, n), v in res.items():
        if bd != band:
            continue
        cur = out.setdefault(mode, [-1.0, 0.0, -1.0, 0.0])
        if v[0] > cur[0]:
            cur[0], cur[1] = v[0], v[1]
        if v[2] > cur[2]:
            cur[2], cur[3] = v[2], v[3]
    return out


def run(lane, mode_names, npts=NPTS):
    tables = P.load()
    res = measure_lane(tables, lane, mode_names, npts=npts)
    with open(os.path.join(HERE, f"res_{lane}.json"), "w", encoding="utf-8") as f:
        json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in res.items()}, f)
    return res


if __name__ == "__main__":
    order = ["fit-residual", "fp64", "tf32", "tf32x3", "bf16", "bf16x3", "fp16", "fp16acc16"]
    for lane in ("double", "float"):
        res = run(lane, list(M.MODES))
        print(f"\n########## {lane} lane (region-A budget {BUDGET[lane]:g}) ##########")
        for band in ("A1", "A2"):
            s = summarise(res, band)
            print(f"--- band {band} (worst over orders) ---")
            for mode in list(s):
                if mode in order or True:
                    v = s[mode]
                    print(f"  {mode:24s} abs {v[0]:.3e} (x={v[1]:.6g})  "
                          f"rel {v[2]:.3e}  budget_ratio {v[0] / BUDGET[lane]:.3e}")
