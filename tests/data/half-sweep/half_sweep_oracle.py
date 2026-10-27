"""Read the half-lane sweep and re-derive the two published margins.

The sweep file is written by half_sweep.cpp: a 20-byte header (magic 'HSWP', version, nmax,
row count) then one 22-byte row per cell -
    format u8 (0 = binary16, 1 = bfloat16), n u8, argument bits u16, returned bits u16,
    returned value f64 (the exact widening of the returned half), bound f64 (the lane's budget
    at that returned value, HalfBound from tests/boys_gate_reference.hpp).

For every cell the true F_n(x) is computed at 45 decimal digits, the error is divided by the
lane's bound, and the worst ratio per format is reported with the margin (1 - ratio) in per cent.

The reference is the boys generator's own: boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS) at 45 dps,
which is the all-positive series the paper's 60-digit reference is built from. That series needs
about x terms and refuses past its cap (RAT_TERMS = 900 reaches x = 250, not 65504 and certainly
not bfloat16's 3.39e38). Above the series' reach the same file's identity is used at the same
precision:

    F_n(x) = [Gamma(n+1/2) - Gamma(n+1/2, x)] / (2 x^(n+1/2)),

which is closed_form_error's own expression (gen_boys_coefficients.py), written as one
subtraction that stays exact where it is used (x well above n+1/2). The two routes are compared
in the band where both run and that comparison is written into the result, so the substitution is
measured rather than asserted.

Usage:  python half_sweep_oracle.py <sweep.bin> <results.json> [--jobs N]
"""

import argparse
import json
import multiprocessing as mp_
import os
import struct
import sys
import time

import numpy as np

GENERATOR = r"C:\Users\myaml\dev\bladawiec\external\boys\tools"
if GENERATOR not in sys.path:
    sys.path.insert(0, GENERATOR)  # workers inherit sys.path from the spawning process

ROW_DTYPE = np.dtype(
    [
        ("format", "u1"),
        ("n", "u1"),
        ("argbits", "<u2"),
        ("gotbits", "<u2"),
        ("got", "<f8"),
        ("bound", "<f8"),
    ]
)
HEADER_SIZE = 20
MAGIC = 0x50535748  # 'HSWP'

DPS = 45
SERIES_MAX_X = 250.0  # boys_ref with RAT_TERMS=900 converges below this; see the fallback count
F16_MANTISSA_BITS, F16_MIN_NORMAL_EXP = 10, -14
BF16_MANTISSA_BITS, BF16_MIN_NORMAL_EXP = 7, -126

# Per-worker globals, set once from the initializer.
_G = {}


def _init_worker(dps):
    import gen_boys_coefficients as g
    from mpmath import mp

    mp.dps = dps
    _G["g"] = g
    _G["mp"] = mp
    _G["mpf"] = mp.mpf


def f_n(n, x, x_float):
    """F_n(x) at the working precision: the generator's series, or its gamma identity."""
    g = _G["g"]
    mp = _G["mp"]
    mpf = _G["mpf"]

    if x_float <= SERIES_MAX_X:
        try:
            return g.boys_ref(n, mpf(x), g.RAT_TAIL_FLOOR, g.RAT_TERMS), 0
        except RuntimeError:
            pass  # the cap could not carry this argument; the identity below does

    s = mpf(n) + mpf("0.5")
    return (mp.gamma(s) - mp.gammainc(s, mpf(x), mp.inf)) / (2 * mpf(x) ** s), 1


def _chunk(args):
    """Evaluate orders 0..nmax for each argument in the chunk."""
    xs, nmax = args
    out = []
    used_gamma = 0
    for x_float in xs:
        row = []
        for n in range(nmax + 1):
            v, route = f_n(n, x_float, x_float)
            used_gamma += route
            row.append(float(v))
        out.append(row)
    return out, used_gamma


def read_sweep(path):
    with open(path, "rb") as f:
        head = f.read(HEADER_SIZE)
        magic, version, nmax, rows = struct.unpack("<IIIQ", head)
        if magic != MAGIC:
            raise SystemExit(f"{path}: not a half sweep file (magic 0x{magic:08x})")
        data = np.fromfile(f, dtype=ROW_DTYPE)
    if data.size != rows:
        raise SystemExit(f"{path}: header says {rows} rows, file holds {data.size}")
    return version, nmax, data


def half_value(fmt, bits):
    """The exact double value of a binary16 or bfloat16 bit pattern."""
    b = int(bits)
    if fmt == 0:
        sign = -1.0 if (b >> 15) else 1.0
        exp = (b >> 10) & 0x1F
        man = b & 0x3FF
        if exp == 0:
            return sign * man * 2.0 ** -24
        return sign * (1.0 + man / 1024.0) * 2.0 ** (exp - 15)
    sign = -1.0 if (b >> 15) else 1.0
    exp = (b >> 7) & 0xFF
    man = b & 0x7F
    if exp == 0:
        return sign * man * 2.0 ** -133
    return sign * (1.0 + man / 128.0) * 2.0 ** (exp - 127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sweep")
    ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    t_start = time.perf_counter()
    version, nmax, data = read_sweep(args.sweep)
    print(f"read {data.size} cells, nmax {nmax}, version {version}", flush=True)

    # The file's own structure: per format, amax+1 arguments, nmax+1 orders each.
    formats = {}
    for fmt, name in ((0, "binary16"), (1, "bfloat16")):
        sel = data[data["format"] == fmt]
        if sel.size % (nmax + 1) != 0:
            raise SystemExit(f"{name}: {sel.size} cells is not a whole number of orders")
        n_args = sel.size // (nmax + 1)
        # The sweep is emitted argument-major, orders 0..nmax inside each argument.
        if not np.array_equal(sel["n"].reshape(n_args, nmax + 1),
                              np.broadcast_to(np.arange(nmax + 1, dtype=np.uint8), (n_args, nmax + 1))):
            raise SystemExit(f"{name}: rows are not argument-major with orders 0..{nmax}")
        formats[name] = (fmt, sel, n_args)

    # Enumerated arguments must be exactly the positive finite patterns, ascending.
    for name, (fmt, sel, n_args) in formats.items():
        expected_last = 0x7BFF if fmt == 0 else 0x7F7F
        firsts = sel["argbits"].reshape(n_args, nmax + 1)[:, 0]
        if int(firsts[0]) != 1 or int(firsts[-1]) != expected_last:
            raise SystemExit(f"{name}: argument range {int(firsts[0])}..{int(firsts[-1])} "
                             f"is not 1..0x{expected_last:04X}")
        if np.any(np.diff(firsts.astype(np.int32)) != 1):
            raise SystemExit(f"{name}: arguments are not every pattern in ascending order")

    # The exact double value of each argument, and the reference, computed in parallel.
    refs = {}
    route_counts = {}
    t_ref = time.perf_counter()
    pool = mp_.Pool(args.jobs, initializer=_init_worker, initargs=(DPS,))
    try:
        for name, (fmt, sel, n_args) in formats.items():
            firsts = sel["argbits"].reshape(n_args, nmax + 1)[:, 0]
            xs = [half_value(fmt, b) for b in firsts]
            chunks = [(xs[i:i + 256], nmax) for i in range(0, len(xs), 256)]
            out = []
            used = 0
            for rows, g_used in pool.imap(_chunk, chunks):
                out.extend(rows)
                used += g_used
            refs[name] = np.array(out, dtype=np.float64)
            route_counts[name] = {"cells_via_gamma_identity": used}
            print(f"{name}: {n_args} arguments, reference done "
                  f"({used} cells via the gamma identity)", flush=True)
    finally:
        pool.close()
        pool.join()
    t_ref = time.perf_counter() - t_ref

    # The overlap check: where the series runs, the identity must agree with it.
    overlap = []
    for x in (100.0, 150.0, 200.0, 250.0):
        for n in (0, 16, 32):
            a, _ = _series_only(n, x)
            b, _ = _gamma_only(n, x)
            overlap.append({"x": x, "n": n, "rel_agreement": float(abs(a - b) / abs(a))})

    results = {
        "revision": {
            "library_tag": "v5.1.1",
            "library_commit": "834bdbb18a30ac8538b519436429f78d41f187ce",
            "lane": "BoysSingleF16 / BoysSingleBf16 (store-half scalar entries)",
            "bound": "HalfBound = 1e-7 + 0.5*ulp(got)",
        },
        "orders": [0, nmax],
        "reference": {
            "dps": DPS,
            "series": "boys_ref(n, x, RAT_TAIL_FLOOR, RAT_TERMS)",
            "series_max_x": SERIES_MAX_X,
            "identity": "F_n = (Gamma(n+1/2) - Gamma(n+1/2, x)) / (2 x^(n+1/2))",
            "overlap_series_vs_identity": overlap,
        },
        "formats": {},
    }

    for name, (fmt, sel, n_args) in formats.items():
        ref = refs[name].reshape(-1)
        got = sel["got"].astype(np.float64)
        bound = sel["bound"].astype(np.float64)
        err = np.abs(got - ref)
        ratio = err / bound

        nonfinite = ~np.isfinite(got)
        zeros = got == 0.0
        worst_i = int(np.nanargmax(np.where(np.isfinite(ratio), ratio, np.nan)))
        n_mat = nmax + 1
        row = {
            "arguments": int(n_args),
            "cells": int(sel.size),
            "non_finite_returns": int(nonfinite.sum()),
            "zero_returns": int(zeros.sum()),
            "worst": {
                "n": int(sel["n"][worst_i]),
                "arg_bits_hex": f"0x{int(sel['argbits'][worst_i]):04X}",
                "x": float(half_value(fmt, sel["argbits"][worst_i])),
                "got_bits_hex": f"0x{int(sel['gotbits'][worst_i]):04X}",
                "delivered": float(got[worst_i]),
                "reference": float(ref[worst_i]),
                "bound": float(bound[worst_i]),
                "error": float(err[worst_i]),
                "ratio": float(ratio[worst_i]),
            },
            "margin_per_cent": float((1.0 - ratio[worst_i]) * 100.0),
            "route": route_counts[name],
        }

        # Ties: a worst ratio several arguments reach is one number and many cells, and a report
        # that names one of them without saying so reads as if the cell were unique.
        top = np.argsort(np.where(np.isfinite(ratio), -ratio, np.inf))[:10]
        argbits = sel["argbits"].reshape(n_args, nmax + 1)[:, 0]
        row["top_cells"] = [
            {
                "n": int(sel["n"][i]),
                "arg_bits_hex": f"0x{int(sel['argbits'][i]):04X}",
                "x": float(half_value(fmt, sel["argbits"][i])),
                "got_bits_hex": f"0x{int(sel['gotbits'][i]):04X}",
                "delivered": float(got[i]),
                "bound": float(bound[i]),
                "ratio": float(ratio[i]),
            }
            for i in top
        ]
        row["cells_within_1e-9_of_worst"] = int(
            (np.abs(ratio - ratio[worst_i]) <= 1e-9 * abs(ratio[worst_i])).sum())
        # Per-order worst, so a reader can see where the worst sits.
        per_order = []
        ratios2 = ratio.reshape(n_args, n_mat)
        for n in range(n_mat):
            j = int(np.nanargmax(ratios2[:, n]))
            per_order.append({"n": n, "ratio": float(ratios2[j, n]),
                              "arg_bits_hex": f"0x{int(sel['argbits'].reshape(n_args, n_mat)[j, 0]):04X}"})
        row["per_order_worst"] = per_order
        results["formats"][name] = row

    t_all = time.perf_counter() - t_start
    results["wall_seconds"] = {"reference": t_ref, "total": t_all}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    for name in formats:
        w = results["formats"][name]["worst"]
        print(f"{name}: {results['formats'][name]['cells']} cells, worst ratio {w['ratio']:.9f} "
              f"at n={w['n']} x={w['arg_bits_hex']} ({w['x']:.6g}), "
              f"margin {results['formats'][name]['margin_per_cent']:.6f} per cent")
    print(f"reference {t_ref:.1f} s, total {t_all:.1f} s")


def _series_only(n, x):
    from mpmath import mp, mpf
    import gen_boys_coefficients as g
    mp.dps = DPS
    return g.boys_ref(n, mpf(x), g.RAT_TAIL_FLOOR, g.RAT_TERMS), 0


def _gamma_only(n, x):
    from mpmath import mp, mpf
    mp.dps = DPS
    s = mpf(n) + mpf("0.5")
    return (mp.gamma(s) - mp.gammainc(s, mpf(x), mp.inf)) / (2 * mpf(x) ** s), 1


if __name__ == "__main__":
    sys.path.insert(0, GENERATOR)
    main()
