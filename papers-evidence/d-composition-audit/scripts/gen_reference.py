#!/usr/bin/env python3
"""High-precision Boys reference for the composed-bounds interaction run.

The argument grid mixes values exactly representable in every measured format
with generic decimals that no measured format represents. A lane never sees the
decimal: it narrows it to its own format first. This generator therefore reads
the per-lane arguments emitted by the driver (`--emit-arguments`), which carry
each lane's exact narrowed input, and computes F_n at that input. A measured
difference is thus the arithmetic's and the store's, not a comparison of two
different arguments.

Each value is split into a double-double pair (hi, lo) with hi + lo equal to the
60-digit value. A single double reference would carry a 1e-16 relative error,
which is the order of the double lane's own delivered error and would corrupt
the very measurement it is meant to judge.

Two independent routes are computed and compared before anything is written:
the incomplete-gamma closed form, and direct quadrature on a subset. The
three-term identity (2n+1) F_n(x) - 2x F_{n+1}(x) = exp(-x) is checked on the
whole grid, lane by lane.

Usage:
    python gen_reference.py --arguments results/arguments.csv --out results/reference.csv

Exit: 0 written and checked, 1 a check failed, 2 mpmath missing or bad input.

Do not run this with `python -I` on this host: isolated mode hides the
user-site mpmath installation and the import fails.
"""

import argparse
import csv
import sys

try:
    from mpmath import mp, mpf, gammainc, exp, quad
except ImportError:  # reported, not raised
    print("SKIP: mpmath is not installed (pip install mpmath)", file=sys.stderr)
    sys.exit(2)

NMAX = 32


def fn_gamma(n, x):
    """F_n(x) by the incomplete-gamma closed form."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    return mpf(1) / 2 * mpf(x) ** (-(mpf(n) + mpf(1) / 2)) * gammainc(
        mpf(n) + mpf(1) / 2, 0, mpf(x)
    )


def fn_quad(n, x):
    """F_n(x) by direct quadrature of the defining integral."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    return quad(lambda t: t ** (2 * n) * exp(-x * t * t), [0, 1])


def split(v):
    """Split an mpf into (hi, lo) doubles with hi + lo = v to ~1e-32."""
    hi = float(v)
    lo = float(v - mpf(hi))
    return hi, lo


def read_arguments(path):
    """Return [(lane, nominal, x, kind)], rejecting duplicate lane-arguments."""
    rows = []
    seen = set()

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            lane = raw["lane"]
            nominal = float(raw["x_nominal"])
            x = float(raw["x"])
            kind = raw["kind"]
            if kind not in ("exact", "generic"):
                raise ValueError(f"unknown argument kind {kind!r}")
            key = (lane, x)
            if key in seen:
                raise ValueError(f"duplicate lane argument {lane} {x!r}")
            seen.add(key)
            rows.append((lane, nominal, x, kind))

    if not rows:
        raise ValueError("the arguments file is empty")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arguments", required=True)
    parser.add_argument("--out", default="results/reference.csv")
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument(
        "--quad-subset",
        default="0.125,1.125,11.7,12.7,30.3",
        help="comma-separated nominal arguments cross-checked by quadrature",
    )
    args = parser.parse_args()

    try:
        rows = read_arguments(args.arguments)
    except (OSError, ValueError, KeyError) as error:
        print(f"FAIL: cannot read arguments: {error}", file=sys.stderr)
        return 2

    mp.dps = args.dps

    values = {}
    for lane, _nominal, x, _kind in rows:
        for n in range(NMAX + 1):
            values[(lane, x, n)] = fn_gamma(n, x)

    # Whole-grid identity check, per lane argument.
    worst_identity = mpf(0)
    worst_at = None
    for lane, _nominal, x, _kind in rows:
        for n in range(NMAX):
            residual = (
                (2 * n + 1) * values[(lane, x, n)]
                - 2 * mpf(x) * values[(lane, x, n + 1)]
                - exp(-mpf(x))
            )
            scale = abs(values[(lane, x, n)]) + abs(values[(lane, x, n + 1)]) + 1
            rel = abs(residual) / scale
            if rel > worst_identity:
                worst_identity = rel
                worst_at = (lane, n, x)

    # Subset cross-check by quadrature, at each lane's own narrowed input.
    subset = {float(s) for s in args.quad_subset.split(",") if s.strip()}
    worst_quad = mpf(0)
    worst_quad_at = None
    quad_points = 0
    for lane, nominal, x, _kind in rows:
        if nominal not in subset:
            continue
        quad_points += 1
        for n in range(NMAX + 1):
            q = fn_quad(n, x)
            d = abs(q - values[(lane, x, n)])
            if d > worst_quad:
                worst_quad = d
                worst_quad_at = (lane, n, x)

    print(
        f"arguments: {len(rows)} lane-arguments x {NMAX + 1} orders = "
        f"{len(rows) * (NMAX + 1)} values"
    )
    print(f"identity residual (relative, whole grid) = {mp.nstr(worst_identity, 6)} at {worst_at}")
    print(
        f"quadrature vs gamma (absolute, {quad_points} lane-arguments) = "
        f"{mp.nstr(worst_quad, 6)} at {worst_quad_at}"
    )

    if worst_identity > mpf(10) ** (-args.dps + 8):
        print("FAIL: the three-term identity does not hold at this precision", file=sys.stderr)
        return 1
    if worst_quad > mpf(10) ** (-args.dps + 12):
        print("FAIL: the two routes disagree beyond the reference precision", file=sys.stderr)
        return 1

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("lane,n,x,hi,lo\n")
        for lane, _nominal, x, _kind in rows:
            for n in range(NMAX + 1):
                hi, lo = split(values[(lane, x, n)])
                f.write(f"{lane},{n},{x!r},{hi!r},{lo!r}\n")

    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
