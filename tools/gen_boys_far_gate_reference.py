#!/usr/bin/env python3
"""The far region's own reference grid, for the far-representation book.

The accuracy gate's grid reaches the far region broadly and coarsely where the
figures are set. Its smallest arguments above the uniform partition's landed top edge
are the boundary's own neighbour and the far ladder's first node above it,
and the worst far cell of every far-representation arm sits at an argument
inside that band - kX1 itself and the first double above kFlatHi. Read
against the gate's grid alone, an arm passes with a delivered
error several times its own figure, which is a test of nothing.

So this grid is the far region's: the same function, the same closed form and
the same working precision as the gate's grid, over an argument list that
resolves every boundary the far arms dispatch on and the two bands the gate's
grid does not reach into.

    F_n(x) = gamma(n + 1/2, 0, x) / (2 x^(n+1/2)),  n = 0..32, x >= kX1

with gamma the unregularized lower incomplete gamma, from mpmath at a pinned
dps. Nothing here shares a route with the library's tables or with the gate's
own grid, and the two routes this file carries are cross-checked against each
other in the run rather than assumed to agree.

The row shape is the gate's, byte for byte: the far book reads this file with
the gate's own loader, so a column added to one and not the other is a load
error rather than a silently unmeasured cell.

Usage:
    python tools/gen_boys_far_gate_reference.py            # rewrite the CSV
    python tools/gen_boys_far_gate_reference.py --check    # fail if it drifted

Exit: 0 agree, 1 drift, 2 mpmath missing.
"""

import argparse
import math
import os
import struct
import sys

try:
    from mpmath import mp, mpf, erf, exp, gammainc, pi, sqrt
except ImportError:  # pragma: no cover - reported, not raised
    print("mpmath is not installed: a check that cannot run is not one that passed",
          file=sys.stderr)
    sys.exit(2)

# The tree's own constants, transcribed so this script shares no import with
# it: the argument list has to move with the library's boundaries, and a
# boundary read from the header would move this grid without saying so.
K_X1 = 28.98933773882074           # region C's left edge; the far table's, too
K_FLAT_HI = 31.0                   # the uniform partition's exclusive right edge
K_SEED_CORRECTION_X1 = 34.381626105834208  # where the corrected seed's correction ends
K_SCALED_FAR_X1 = 117.0            # the scaled table's exclusive right edge

X_MAX = 3.4028234663852886e38      # binary32's maximum, the gate grid's own top
DPS = 80
VALUE_DIGITS = 25

FIRST_ORDER = 0
LAST_ORDER = 32

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data",
                   "boys_far_gate_reference.csv")
GATE_GRID = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "data",
                         "boys_accuracy_gate_reference.csv")

HEADER = ("n,x,value,log10abs,"
          "xf,valuef,log10_f,"
          "x16,value16,log10_16,"
          "xb,valueb,log10_b")


# --- the reference ---------------------------------------------------------

def boys_gamma(n, x):
    """Route A: the lower incomplete gamma closed form, the gate grid's own route."""
    x = mpf(x)
    return mpf("0.5") * x ** -(n + mpf("0.5")) * gammainc(n + mpf("0.5"), 0, x)


def boys_erf0(x):
    """Route A': F_0's elementary closed form. Structurally unrelated to route A."""
    x = mpf(x)
    return mpf("0.5") * sqrt(pi / x) * erf(sqrt(x))


def fmt_value(v):
    return mp.nstr(v, VALUE_DIGITS, strip_zeros=False)


def decade(v):
    return int(mp.floor(mp.log10(abs(v)))) if v != 0 else 0


def float_to_fp16(v):
    """The double FloatToHalf(float(v)) holds (inf past the fp16 range)."""
    try:
        return struct.unpack("<e", struct.pack("<e", v))[0]
    except (OverflowError, struct.error):
        return math.inf if v > 0 else -math.inf


def float_to_bf16(v):
    """Round-to-nearest-even to bfloat16, returned as the double it holds."""
    bits = struct.unpack("<I", struct.pack("<f", v))[0]

    if (bits & 0x7F800000) == 0x7F800000:
        return v

    lsb = (bits >> 16) & 1
    bits = (bits + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


# --- the arguments ---------------------------------------------------------

def walk(x, steps):
    """`steps` doubles from x upward, x itself first."""
    out = [x]

    for _ in range(steps):
        x = math.nextafter(x, math.inf)
        out.append(x)

    return out


def back(x, steps):
    """`steps` doubles from x downward, x itself last."""
    out = []

    for _ in range(steps):
        x = math.nextafter(x, 0.0)

    for _ in range(steps):
        out.append(x)
        x = math.nextafter(x, math.inf)

    return out


def committed_far_arguments():
    """The far arguments the gate's own grid already carries.

    They are here so the far book's reading of this grid is a superset of the
    reading the gate's grid gives: a figure that moves between the two grids is
    a finding about the arguments and not about which file was read.
    """
    if not os.path.exists(GATE_GRID):
        return []

    out = set()

    with open(GATE_GRID, encoding="utf-8") as f:
        next(f, None)

        for row in f:
            parts = row.split(",")

            if len(parts) != 13:
                continue

            if parts[0] != str(FIRST_ORDER):
                continue

            x = float(parts[1])

            if x >= K_X1:
                out.add(x)

    return sorted(out)


def arguments():
    """The swept arguments: one list shared by every order.

    Every boundary the far arms dispatch on carries the doubles on both sides
    of it, because the two sides are two different arithmetics and the claim
    changes at the crossing rather than near it.
    """
    xs = []

    def add(v):
        if v > 0.0 and math.isfinite(v):
            xs.append(v)

    # The tables' and the region's own edges, each with the doubles around it.
    for b in (K_X1, K_FLAT_HI, K_SEED_CORRECTION_X1, K_SCALED_FAR_X1):
        for v in back(b, 16):
            add(v)

        for v in walk(b, 16):
            add(v)

    # The band the far figures are set in: the arms open at kX1 and at kFlatHi,
    # and the band between the two edges is carried by committed_far_arguments,
    # so what this grid adds above the gate's step is the top's own band.
    x = K_FLAT_HI + 0.005

    while x <= 32.0:
        add(x)
        x += 0.005

    # kFlatHi to 32 resolves the band the recurrence's dropped term is largest
    # in, in steps narrower than the band itself. Outward from there the error
    # falls monotonically, so those arguments state the domain and sample the
    # table's own intervals rather than hunt for a maximum.
    x = 32.0

    while x <= 36.0:
        add(x)
        x += 0.02

    x = 36.0

    while x <= K_SCALED_FAR_X1:
        add(x)
        x += 0.125

    # The table's span's upper end, where the two right-hand neighbours and the
    # last interval's own edge sit: the gate's grid carried no argument at 117.
    x = K_SCALED_FAR_X1

    while x <= 130.0:
        add(x)
        x += 0.25

    # Outward geometrically to the top of the formats the gate's grid reaches,
    # so an arm that answers the default entry's values far out is measured
    # there too.
    lo, hi, steps = 130.0, X_MAX, 180

    for i in range(steps):
        add(lo * (hi / lo) ** (i / (steps - 1)))

    for x in committed_far_arguments():
        add(x)

    seen = set()
    out = []

    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out


# --- rendering -------------------------------------------------------------

def render():
    xs = arguments()
    f32_xs = [struct.unpack("<f", struct.pack("<f", x))[0] for x in xs]
    half_xs = [float_to_fp16(x) for x in xs]
    bf_xs = [float_to_bf16(x) for x in xs]

    cache = {}

    def value(n, x):
        key = (n, x)

        if key not in cache:
            cache[key] = boys_gamma(n, x)

        return cache[key]

    lines = [HEADER]
    cells = 0

    for n in range(FIRST_ORDER, LAST_ORDER + 1):
        for i, x in enumerate(xs):
            xf = f32_xs[i]
            x16 = half_xs[i]
            xb = bf_xs[i]
            v = value(n, x)
            vf = value(n, xf) if xf == x else boys_gamma(n, xf)
            v16 = (value(n, x16) if x16 == x else boys_gamma(n, x16)) \
                if math.isfinite(x16) else mpf(0)
            vb = value(n, xb) if xb == x else boys_gamma(n, xb)
            half_field = f"{x16:.17g}" if math.isfinite(x16) else "inf"
            lines.append(f"{n},{x:.17g},{fmt_value(v)},{decade(v)},"
                         f"{xf:.17g},{fmt_value(vf)},{decade(vf)},"
                         f"{half_field},{fmt_value(v16)},{decade(v16)},"
                         f"{xb:.17g},{fmt_value(vb)},{decade(vb)}")
            cells += 1

    return "\n".join(lines) + "\n", xs, cells


def evidence(xs):
    """The number that says the reference is right, measured and not asserted."""
    samples = [(0, x) for x in xs[::17]]
    worst = mpf(0)
    at = None

    for _, x in samples:
        d = abs(boys_erf0(x) - boys_gamma(0, x))

        if d > worst:
            worst = d
            at = x

    print(f"  F_0, the gamma closed form against the elementary erf form: worst "
          f"absolute disagreement {mp.nstr(worst, 6)} at x={at:.17g}")

    # The three-term identity (2n+1) F_n - 2x F_(n+1) = exp(-x), which any
    # correct table satisfies. Restricted to x <= 38: past it the identity's
    # right-hand side is far smaller than the terms it is the difference of and
    # the residual measures the working precision rather than the reference.
    near = [x for x in xs if x <= 38.0]
    worst = mpf(0)
    at = None

    for n in range(0, LAST_ORDER):
        for x in near[::11]:
            lhs = (2 * n + 1) * boys_gamma(n, x) - 2 * mpf(x) * boys_gamma(n + 1, x)
            rhs = exp(-mpf(x))
            d = abs(lhs - rhs) / abs(rhs)

            if d > worst:
                worst = d
                at = (n, x)

    print(f"  the recurrence identity (2n+1)F_n - 2xF_(n+1) = exp(-x): worst relative "
          f"residual {mp.nstr(worst, 6)} at (n,x)={at}")


def first_difference(committed, rendered):
    stored = committed.splitlines()
    produced = rendered.splitlines()

    for i in range(max(len(stored), len(produced))):
        a = stored[i] if i < len(stored) else "<the committed file ends here>"
        b = produced[i] if i < len(produced) else "<the rendered file ends here>"

        if a != b:
            return i + 1, a, b

    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT)
    parser.add_argument("--check", action="store_true",
                        help="exit nonzero if the committed file is not what this "
                             "generator produces")
    args = parser.parse_args()

    mp.dps = DPS
    print(f"boys_far_gate_reference: mpmath {mp.__version__ if hasattr(mp, '__version__') else ''}"
          f" at dps={DPS}, values to {VALUE_DIGITS} significant digits")

    rendered, xs, cells = render()
    print(f"  {len(xs)} argument(s) x {LAST_ORDER - FIRST_ORDER + 1} orders = {cells} cell(s)")
    print(f"  far arguments of the gate's own grid carried here: "
          f"{len(committed_far_arguments())}")
    print(f"  the two bands this grid resolves past the gate's own step: "
          f"{sum(1 for x in xs if K_FLAT_HI < x < 32.0)} argument(s) in "
          f"({K_FLAT_HI}, 32.0), "
          f"{sum(1 for x in xs if 116.0 < x < 117.0)} in (116, {K_SCALED_FAR_X1})")
    evidence(xs)

    if args.check:
        if not os.path.exists(args.out):
            print(f"DRIFT: {args.out} is not there", file=sys.stderr)
            return 1

        with open(args.out, encoding="utf-8") as f:
            committed = f.read()

        if committed != rendered:
            line, was, now = first_difference(committed, rendered)
            print(f"DRIFT: {args.out} differs at line {line}", file=sys.stderr)
            print(f"  committed: {was}", file=sys.stderr)
            print(f"  rendered : {now}", file=sys.stderr)
            return 1

        print(f"  {args.out} is byte for byte what this generator renders")
        return 0

    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.write(rendered)

    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
