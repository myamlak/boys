#!/usr/bin/env python3
"""Independent high-precision reference grid for the accuracy gate.

This script is deliberately separate from tools/gen_boys_coefficients.py: it
shares no code with the library under test and none with the committed grid,
so that a bug in either cannot be confirmed by the other. It computes

    F_n(x) = integral_0^1 t^(2n) exp(-x t^2) dt ,  n = 0..32, x >= 0

by two structurally different routes and cross-checks them:

  route A  the lower incomplete gamma closed form
           F_n(x) = (1/2) x^(-(n+1/2)) gamma(n+1/2, 0, x), and, for n = 0,
           the elementary F_0(x) = (1/2) sqrt(pi/x) erf(sqrt(x));
  route B  direct numerical quadrature of the defining integral, with the
           substitution u = t sqrt(x) above x = 50 to keep the quadrature
           interval smooth;
  route C  the three-term identity (2n+1) F_n(x) - 2x F_(n+1)(x) = exp(-x),
           which any correct table satisfies and a corrupted one does not.

The disagreement between the routes is printed, not assumed: it is the
evidence that the reference is right.

Output: a CSV of n, x, value with x written as a round-tripping %.17g double
and value at 25 significant digits (absolute error << 1e-25 for |F| <= 1,
which is four orders below the tightest documented bound of 1e-15). The
argument list reaches the top of every format's finite range, so that a lane's
tested range is never limited by which grid points have a representation in the
format it evaluates in; `grid()` states that rule in full.
"""

import argparse
import math
import os
import struct
import sys

try:
    from mpmath import mp, mpf, cos, erf, exp, gammainc, pi, quad, sqrt
except ImportError:  # pragma: no cover - reported, not raised
    print("SKIP: mpmath is not installed (`pip install mpmath`)")
    sys.exit(2)

# The library's own region boundaries, quoted here so the grid spans them
# explicitly; written as the decimal literals of the committed constants.
K_EXTENDED_B_X0 = 1.0855252345349333
K_X0 = 11.899848152108484
K_X1 = 28.98933773882074
TIER_THRESHOLDS = {}
for _n, _t in (
    (range(0, 5), 1.0855252345349333),
    (range(5, 9), 2.015297705335114),
    (range(9, 17), 4.897870299825657),
    (range(17, 33), 10.78358785891676),
):
    for _k in _n:
        TIER_THRESHOLDS[_k] = _t

X_MAX = 1e18
VALUE_DIGITS = 25
DPS = 80


def boys_gamma(n, x):
    """Route A: the lower incomplete gamma closed form."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    x = mpf(x)
    return mpf("0.5") * x ** -(n + mpf("0.5")) * gammainc(n + mpf("0.5"), 0, x)


def boys_erf0(x):
    """Route A': the elementary closed form of F_0, valid for every x > 0."""
    if x == 0:
        return mpf(1)
    x = mpf(x)
    return mpf("0.5") * sqrt(pi / x) * erf(sqrt(x))


def boys_quad(n, x):
    """Route B: quadrature of the defining expression."""
    if x == 0:
        return mpf(1) / (2 * n + 1)
    x = mpf(x)
    if x < 50:
        return quad(lambda t: t ** (2 * n) * exp(-x * t * t), [0, 1])
    s = sqrt(x)
    return x ** -(n + mpf("0.5")) * quad(lambda u: u ** (2 * n) * exp(-u * u), [0, s])


def boys_series(n, x):
    """Route B': the Taylor series sum_k (-x)^k / (k! (2n+2k+1)), for modest x."""
    x = mpf(x)
    total = mpf(0)
    term = mpf(1)
    k = 0
    while True:
        add = term / (2 * n + 2 * k + 1)
        total += add
        if abs(add) < abs(total) * mpf(10) ** (-mp.dps + 10) or k > 40 * mp.dps:
            break
        k += 1
        term = -term * x / k
    return total


def next_double(v):
    """The next double above v (the grid walks the exact dispatch boundary)."""
    return math.nextafter(v, math.inf)


def prev_double(v):
    return math.nextafter(v, -math.inf)


def rng_stream(seed):
    """A small deterministic LCG: off-grid points, reproducible across runs."""
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def grid(committed_xs):
    """The swept arguments: one list shared by every order.

    A rectangular table (every order at every argument) is what lets the gate
    check an all-orders batch call element by element against one reference,
    so the list carries every dispatch boundary of every order, not just the
    boundaries of one.
    """
    xs = []
    add = xs.append

    for v in (0.0, 5e-324, 1e-300, 1e-12, 1e-8, 1e-5, 1e-3, 0.01, 0.1):
        add(v)

    def uniform(a, b, count):
        for i in range(count):
            add(a + (b - a) * i / (count - 1))

    band_starts = sorted(set(TIER_THRESHOLDS.values()))
    uniform(0.0, K_EXTENDED_B_X0, 150)
    edges = [K_EXTENDED_B_X0] + band_starts + [K_X0]
    for a, b in zip(edges, edges[1:]):
        uniform(a, b, 80)
    uniform(K_X0, K_X1, 200)

    lo, hi, steps = K_X1, X_MAX, 250
    for i in range(steps):
        add(lo * (hi / lo) ** (i / (steps - 1)))

    # A denser geometric sweep over the two decades above region C's start:
    # that is where the half lanes' values leave the format's normal range and
    # where the published per-order usable range is stated, so the boundary is
    # resolved to a per-cent rather than a per-16-per-cent step.
    lo, hi, steps = K_X1, 1e4, 200
    for i in range(steps):
        add(lo * (hi / lo) ** (i / (steps - 1)))

    # The published arguments themselves, so the gate measures the exact cells
    # the documentation names rather than the nearest grid node to them: the
    # half lane's 0.990 worst cell at x = 721, the bounded-domain arguments the
    # region-C paragraph names, and the asymptotic branch's stated domain edge.
    for v in (12.0, 15.9, 16.0, 30.0, 35.0, 38.0, 40.0, 129.0, 361.0, 721.0):
        add(v)

    # Every dispatch boundary and its two nearest doubles: the region is
    # chosen by `<` comparisons, so the pair straddling a boundary is exactly
    # where a claim's region, and therefore its bound, flips.
    for b in edges[1:] + [K_X1]:
        add(prev_double(b))
        add(b)
        add(next_double(b))

    # Off-grid samples, so a narrow error spike cannot hide between the
    # uniform nodes.
    nxt = rng_stream(0x5EED)
    for a, b, count in (
        (0.0, K_EXTENDED_B_X0, 40),
        (K_EXTENDED_B_X0, K_X0, 60),
        (K_X0, K_X1, 40),
        (K_X1, 1e3, 40),
        (1e3, 1e6, 40),
    ):
        for _ in range(count):
            add(a + (b - a) * nxt())

    # The top of every format's finite range, so that no lane's tested argument
    # range is limited by which grid points happen to have a representation in
    # that format. The rule, stated once so the next format does not reopen it:
    # for each format F narrower than binary64 the grid carries (a) the largest
    # finite value of F, (b) a geometric ladder of F-representable values from
    # the grid's previous maximum up to it, and (c) where F's spacing leaves a
    # band wider than a grid step, every F-representable value in that band.
    # Without (c) a binary16 lane is unchecked from 60672, the largest grid
    # point casting finite, to 65504, the format's maximum - seven per cent of
    # its range. The ladder stops at binary32's maximum: no narrower lane can
    # receive an argument above it (the cast is out of range there), and
    # binary64 is never representation-limited, because the grid is a list of
    # doubles to begin with.
    for v in range(60672, 65505, 32):
        add(float(v))
    for top, rounds_to in (
        (3.3895313892515355e38, float_to_bf16),                    # bfloat16
        (3.4028234663852886e38,
         lambda v: struct.unpack("<f", struct.pack("<f", v))[0]),  # binary32
    ):
        for i in range(64):
            add(rounds_to(X_MAX * (top / X_MAX) ** (i / 63.0)))
        add(top)

    # The committed grid's arguments, so the gate's numbers are comparable
    # with the suite's on exactly the points the suite asserts on.
    xs.extend(committed_xs)

    seen = set()
    out = []
    for x in xs:
        if x >= 0.0 and not math.isinf(x) and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def float_to_fp16(v):
    """The double that FloatToHalf(float(v)) holds (inf past the fp16 range)."""
    try:
        return struct.unpack("<e", struct.pack("<e", v))[0]
    except OverflowError:
        return math.inf if v > 0 else -math.inf


def float_to_bf16(v):
    """Round-to-nearest-even to bfloat16, returned as the double it holds."""

    bits = struct.unpack("<I", struct.pack("<f", v))[0]
    if (bits & 0x7F800000) == 0x7F800000:
        return v
    # Round the 23-bit significand to 8 bits, ties to even.
    lsb = (bits >> 16) & 1
    rounding = 0x7FFF + lsb
    bits = (bits + rounding) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def load_committed_xs(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        rows = f.read().splitlines()[1:]
    out = set()
    for row in rows:
        parts = row.split(",")
        if len(parts) == 3:
            out.add(float(parts[1]))
    return sorted(out)


def reference_agreement(samples):
    """The number that says the reference is right, measured not asserted."""
    worst_abs = mpf(0)
    worst_rel = mpf(0)
    worst_at = None
    for n, x in samples:
        a = boys_gamma(n, x)
        b = boys_quad(n, x)
        d = abs(a - b)
        if d > worst_abs:
            worst_abs = d
            worst_at = (n, x)
        if abs(a) > mpf("1e-8") and d / abs(a) > worst_rel:
            worst_rel = d / abs(a)
    return worst_abs, worst_rel, worst_at


def n0_agreement(samples):
    """F_0 against the elementary closed form, and against the series."""
    worst = mpf(0)
    at = None
    for _, x in samples:
        d = abs(boys_erf0(x) - boys_gamma(0, x))
        if d > worst:
            worst = d
            at = x
    return worst, at


def series_agreement(samples):
    worst = mpf(0)
    at = None
    for n, x in samples:
        if x > 40:
            continue
        d = abs(boys_series(n, x) - boys_gamma(n, x))
        if d > worst:
            worst = d
            at = (n, x)
    return worst, at


def identity_residual(rows, orders):
    """max |(2n+1)F_n - 2x F_(n+1) - exp(-x)| over the grid, relative.

    Restricted to x <= 40. The identity's right-hand side is far smaller than
    either term it is the difference of (at x = 40, exp(-x) is 4e-18 against
    terms of order 1e-2), so it is a cancellation test that needs the working
    precision to cover the gap; past x = 40 the gap exceeds dps and the
    residual measures the working precision, not the table.
    """
    table = {}
    for n, x, v in rows:
        table[(n, x)] = v
    worst = mpf(0)
    at = None
    for n in range(33):
        if n + 1 > 32:
            break
        for x in orders:
            if x > 40.0:
                continue
            if (n, x) not in table or (n + 1, x) not in table:
                continue
            lhs = (2 * n + 1) * table[(n, x)] - 2 * mpf(x) * table[(n + 1, x)]
            rhs = exp(-mpf(x))
            d = abs(lhs - rhs) / (abs(rhs) if rhs != 0 else mpf(1))
            if d > worst:
                worst = d
                at = (n, x)
    return worst, at


def fmt_value(v):
    return mp.nstr(v, VALUE_DIGITS, strip_zeros=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/data/boys_accuracy_gate_reference.csv")
    parser.add_argument("--committed", default="tests/data/boys_reference.csv")
    parser.add_argument("--validate-only", action="store_true",
                        help="print the route agreements and write nothing")
    args = parser.parse_args()

    mp.dps = DPS
    committed = load_committed_xs(args.committed)

    validation = []
    for n in range(33):
        for x in (0.0, 1e-12, 1e-4, 0.1, 0.5, 1.0, 1.0855252345349333, 2.0,
                  4.897870299825657, 8.0, 11.899848152108484, 20.0,
                  28.98933773882074, 40.0, 100.0, 1e3, 1e6):
            validation.append((n, x))

    worst_abs, worst_rel, at = reference_agreement(validation)
    n0_abs, n0_at = n0_agreement(validation)
    ser_abs, ser_at = series_agreement(validation)

    print(f"reference route agreement (dps={DPS}, {len(validation)} points)")
    print(f"  gamma vs quad   : max |A-B| = {mp.nstr(worst_abs, 6)}"
          f"  max rel = {mp.nstr(worst_rel, 6)}  at (n,x)={at}")
    print(f"  F_0 erf vs gamma: max |A-B| = {mp.nstr(n0_abs, 6)}  at x={n0_at}")
    print(f"  series vs gamma : max |A-B| = {mp.nstr(ser_abs, 6)}  at (n,x)={ser_at}")

    if args.validate_only:
        return 0

    rows = []
    xs = grid(committed)
    half_xs = [float_to_fp16(x) for x in xs]
    bf_xs = [float_to_bf16(x) for x in xs]
    f32_xs = [struct.unpack("<f", struct.pack("<f", x))[0] for x in xs]
    for n in range(33):
        for i, x in enumerate(xs):
            rows.append((n, x, f32_xs[i], half_xs[i], bf_xs[i],
                         boys_gamma(n, x), boys_gamma(n, f32_xs[i]),
                         boys_gamma(n, half_xs[i]) if math.isfinite(half_xs[i]) else mpf(0),
                         boys_gamma(n, bf_xs[i])))

    resid, resid_at = identity_residual([(r[0], r[1], r[5]) for r in rows], xs)
    print(f"  recurrence identity residual (relative) = {mp.nstr(resid, 6)}"
          f"  at (n,x)={resid_at}")

    # The magnitude columns let the gate tell "the returned value is right"
    # from "the returned value is below what the format can hold": a value
    # whose decade underflows the parsing double is reported, not silently
    # read as zero. The xf/x16/xb columns carry the argument each narrower
    # lane actually evaluates at (its own rounding of x), so the gate never
    # has to re-derive a rounding rule in a second implementation.
    def decade(v):
        return int(mp.floor(mp.log10(abs(v)))) if v != 0 else 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("n,x,value,log10abs,"
                "xf,valuef,log10_f,"
                "x16,value16,log10_16,"
                "xb,valueb,log10_b\n")
        for n, x, xf, x16, xb, v, vf, v16, vb in rows:
            half_field = f"{x16:.17g}" if math.isfinite(x16) else "inf"
            f.write(f"{n},{x:.17g},{fmt_value(v)},{decade(v)},"
                    f"{xf:.17g},{fmt_value(vf)},{decade(vf)},"
                    f"{half_field},{fmt_value(v16)},{decade(v16)},"
                    f"{xb:.17g},{fmt_value(vb)},{decade(vb)}\n")
    outside = sum(1 for x in xs if not math.isfinite(float_to_fp16(x)))
    print(f"wrote {args.out}: {len(rows)} rows, {len(xs)} x per order, "
          f"{outside} arguments past the fp16 range")
    return 0


if __name__ == "__main__":
    sys.exit(main())
