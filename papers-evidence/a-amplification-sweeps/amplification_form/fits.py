#!/usr/bin/env python3
"""The second sweep's seed fits.

For each band, each seed order used as a recurrence start (the union of the
upward and downward starts: 0, 2, 4, 8, 16, 24, 28, 32) and each degree on
the ladder (0, 1, 2, 4, 6, ..., 24), this script writes the Chebyshev
coefficients of F_s over the band, computed at 70 working digits and rounded
to the double the harness will use.

The coefficients are the Chebyshev interpolant at the degree+1 Gauss-
Chebyshev nodes, c_j = (2/N) sum_k f(t_k) cos(pi j (k+1/2)/N), with c_0
halved.  The rounded coefficients are then validated the way the harness will
use them: the double-precision Clenshaw of `../common/replica.hpp`, evaluated
on a dense test set, against the tree's own 60-digit gamma reference.  The
printed validation is the fit's truncation error, not the reference's.

Outputs:
    data/fits.csv             band,seed_order,degree,index,value
    data/fits_validation.txt  worst absolute and relative fit error per fit

The degree ladder and start orders are stated in the manuscript.  The
interpolation and rounding are a reconstruction: the lost tree's fit
construction is not recoverable from the surviving records.

mpmath lives in the user site.  Do not run with `python -I`.

Usage:
    python fits.py --out data/fits.csv --validation data/fits_validation.txt
"""

import argparse
import csv
import importlib.util
import os
import sys
from pathlib import Path

if sys.flags.isolated:
    print("SKIP: isolated mode hides user-site mpmath; run without python -I")
    sys.exit(77)

try:
    from mpmath import mp, mpf
except ImportError:  # pragma: no cover
    print("SKIP: mpmath is not installed (`pip install mpmath`)")
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import boysprovenance  # noqa: E402

DEFAULT_TREE = boysprovenance.DEFAULT_TREE
SEED_ORDERS = [0, 2, 4, 8, 16, 24, 28, 32]
DEGREES = [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24]
VALIDATION_POINTS = 259


def load_generator(tree):
    path = tree / "tools/gen_boys_accuracy_gate_reference.py"
    spec = importlib.util.spec_from_file_location("paper_a_boys_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def reference_value(generator, n, x):
    if n == 0:
        return generator.boys_erf0(x)
    return generator.boys_gamma(n, x)


def chebyshev_coefficients(generator, order, left, right, degree):
    count = degree + 1
    center = (mpf(left) + mpf(right)) / 2
    half = (mpf(right) - mpf(left)) / 2
    values = []
    for k in range(count):
        t = mp.cos(mp.pi * (k + mpf("0.5")) / count)
        x = center + half * t
        values.append(reference_value(generator, order, x))

    coefficients = []
    for j in range(count):
        total = mpf(0)
        for k in range(count):
            total += values[k] * mp.cos(mp.pi * j * (k + mpf("0.5")) / count)
        factor = mpf(2) / count
        if j == 0:
            factor /= 2
        coefficients.append(total * factor)
    return coefficients


def clenshaw_double(coefficients, degree, x, left, right):
    t = 2.0 * (x - left) / (right - left) - 1.0
    b1 = 0.0
    b2 = 0.0
    for j in range(degree, 0, -1):
        b0 = 2.0 * t * b1 - b2 + coefficients[j]
        b2 = b1
        b1 = b0
    return t * b1 - b2 + coefficients[0]


def validation_points(left, right):
    center = (mpf(left) + mpf(right)) / 2
    half = (mpf(right) - mpf(left)) / 2
    points = [float(left), float(right)]
    for j in range(VALIDATION_POINTS - 2):
        t = mp.cos(mp.pi * (j + mpf("0.5")) / (VALIDATION_POINTS - 2))
        points.append(float(center + half * t))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--dps", type=int, default=70)
    parser.add_argument("--max-fits", type=int, default=0, help="smoke test only")
    args = parser.parse_args()

    tree = Path(args.boys_tree).resolve()
    generator = load_generator(tree)
    constants = boysprovenance.parse_constants(tree)
    edge = constants["kRegionA1Edge"]
    end = constants["kRegionAEnd"]
    bands = [("A1", 0.0, edge), ("A2", edge, end)]

    mp.dps = args.dps
    tests = {name: validation_points(left, right) for name, left, right in bands}
    references = {}
    for name, left, right in bands:
        for order in SEED_ORDERS:
            references[(name, order)] = [
                reference_value(generator, order, mpf(x)) for x in tests[name]
            ]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    worst = []
    written = 0

    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "seed_order", "degree", "index", "value"])

        for name, left, right in bands:
            for order in SEED_ORDERS:
                for degree in DEGREES:
                    if args.max_fits and written >= args.max_fits:
                        break
                    exact = chebyshev_coefficients(generator, order, left, right, degree)
                    rounded = [float(value) for value in exact]
                    for index, value in enumerate(rounded):
                        writer.writerow([name, order, degree, index, repr(value)])
                    written += 1

                    worst_abs = mpf(0)
                    worst_rel = mpf(0)
                    worst_at = None
                    for point, expected in zip(tests[name], references[(name, order)]):
                        value = clenshaw_double(rounded, degree, point, left, right)
                        error = abs(mpf(value) - expected)
                        relative = error / expected if expected != 0 else mpf("inf")
                        if error > worst_abs:
                            worst_abs = error
                            worst_at = point
                        if relative > worst_rel:
                            worst_rel = relative
                    worst.append(
                        (
                            name,
                            order,
                            degree,
                            worst_abs,
                            worst_rel,
                            worst_at,
                        )
                    )
                if args.max_fits and written >= args.max_fits:
                    break
            if args.max_fits and written >= args.max_fits:
                break

    lines = [
        "seed-fit validation",
        "degrees: %s" % DEGREES,
        "seed orders: %s" % SEED_ORDERS,
        "dps: %d, validation points per fit: %d" % (args.dps, VALIDATION_POINTS),
        "value: double Clenshaw of the rounded coefficients, against the gamma reference",
        "",
        "%-3s %5s %6s %14s %14s   at x" % ("band", "order", "degree", "worst_abs", "worst_rel"),
    ]
    for name, order, degree, worst_abs, worst_rel, worst_at in worst:
        lines.append(
            "%-3s %5d %6d %14.4e %14.4e   %s"
            % (
                name,
                order,
                degree,
                float(worst_abs),
                float(worst_rel) if worst_rel != mpf("inf") else float("inf"),
                mp.nstr(worst_at, 17),
            )
        )

    validation_path = (
        Path(args.validation) if args.validation else out.with_suffix(".validation.txt")
    )
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print("wrote %s: %d fits" % (out, written))
    print("wrote %s" % validation_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
