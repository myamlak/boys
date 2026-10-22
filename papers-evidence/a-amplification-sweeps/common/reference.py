#!/usr/bin/env python3
"""High-precision reference values for the Paper A sweeps.

The reference is the boys v5.0.0 tree's own generator, imported rather than
copied: `tools/gen_boys_accuracy_gate_reference.py` computes F_n(x) from the
lower incomplete gamma function (the error function at n = 0).  This script
adds a shared-factor Taylor series for whole-order columns, and validates the
gamma reference against every independent route the tree already carries:

    R2  Taylor series, the (-x)^k / k! factor shared across all orders
    R3  direct quadrature of the defining integral (a subset)
    R4  the kernel's committed 45-digit grid (cells whose argument is present)
    R5  the three-term identity (2n+1)F_n - 2xF_{n+1} = exp(-x)
    R6  F_0 from the gamma route against the elementary error-function form
    R7  the kernel's committed accuracy-gate reference grid

The output is one CSV: band, x, and F_0 .. F_32 at --digits significant
decimal digits.  The arguments are written back exactly as the grid spelled
them; the reference is evaluated at the exact double they denote
(mp.mpf(float(x))), not at the decimal text.

mpmath lives in the user site on this machine.  Do not run this script with
`python -I`: isolated mode skips the user site and mpmath disappears.

Usage:
    python reference.py --grid data/grid.csv --out data/reference.csv \
        --validation data/reference_validation.txt --dps 60 --digits 60
"""

import argparse
import csv
import importlib.util
import math
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

DEFAULT_TREE = r"C:\Users\myaml\dev\bladawiec\external\boys"

NMAX = 32

ROUTE_THRESHOLDS = {
    "R2 series": 1e-45,
    "R2b imported series": 1e-45,
    "R3 quadrature": 1e-50,
    "R4 committed grid (absolute)": 1e-40,
    "R5 identity": 1e-45,
    "R6 F0 gamma against erf": 1e-50,
    "R7 accuracy-gate grid": 1e-18,
}


class Worst:
    def __init__(self):
        self.value = None
        self.where = None
        self.cells = 0

    def add(self, value, where):
        self.cells += 1
        if self.value is None or value > self.value:
            self.value = value
            self.where = where

    def format(self):
        if self.value is None:
            return "not evaluated"
        return "%.6e at %s over %d cells" % (float(self.value), self.where, self.cells)


def load_generator(tree):
    path = tree / "tools/gen_boys_accuracy_gate_reference.py"
    spec = importlib.util.spec_from_file_location("paper_a_boys_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_grid(path):
    rows = []
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append((row["band"], row["x"], float(row["x"])))
    return rows


def reference_values(generator, x, nmax):
    values = [generator.boys_erf0(x)]
    for n in range(1, nmax + 1):
        values.append(generator.boys_gamma(n, x))
    return values


def series_all(x, nmax, dps):
    """Taylor series with the (-x)^k / k! factor shared across orders."""
    totals = [mpf(0) for _ in range(nmax + 1)]
    factor = mpf(1)
    cutoff = mpf(10) ** (-(dps + 5))
    k = 0
    while True:
        for n in range(nmax + 1):
            totals[n] += factor / (2 * n + 2 * k + 1)
        k += 1
        factor = -x * factor / k
        if abs(factor) < cutoff:
            break
        if k > 100000:
            raise RuntimeError("series did not converge at x = %s" % x)
    return totals


def relative_difference(a, b):
    if b == 0:
        return mpf("inf")
    return abs(a - b) / abs(b)


def parse_committed(path):
    cells = {}
    if not path.is_file():
        return cells
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                cells[(int(row["n"]), float(row["x"]))] = mpf(row["value"])
            except (KeyError, ValueError):
                continue
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--grid", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--dps", type=int, default=60)
    parser.add_argument("--digits", type=int, default=60)
    parser.add_argument("--quad-every", type=int, default=128, help="0 disables R3")
    parser.add_argument("--series-check-every", type=int, default=97, help="0 disables the imported-series cross-check")
    parser.add_argument("--max-arguments", type=int, default=0, help="smoke test only")
    parser.add_argument("--no-strict", action="store_true")
    args = parser.parse_args()

    tree = Path(args.boys_tree).resolve()
    generator = load_generator(tree)
    mp.dps = args.dps

    grid = parse_grid(args.grid)
    if args.max_arguments:
        grid = grid[: args.max_arguments]

    committed_dir = tree / "tests/data"
    committed_45 = parse_committed(committed_dir / "boys_reference.csv")
    gate_grid = parse_committed(committed_dir / "boys_accuracy_gate_reference.csv")

    worst = {name: Worst() for name in ROUTE_THRESHOLDS}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arguments = 0
    with open(out_path, "w", newline="") as out_handle:
        writer = csv.writer(out_handle)
        writer.writerow(["band", "x"] + ["f%d" % n for n in range(NMAX + 1)])

        for index, (band, x_text, x_float) in enumerate(grid):
            x = mpf(x_float)
            values = reference_values(generator, x, NMAX)
            writer.writerow(
                [band, x_text]
                + [mp.nstr(value, args.digits, strip_zeros=False) for value in values]
            )
            arguments += 1

            if x_float != 0.0:
                series = series_all(x, NMAX, args.dps)
                for n in range(NMAX + 1):
                    worst["R2 series"].add(
                        relative_difference(series[n], values[n]), "%s n=%d x=%s" % (band, n, x_text)
                    )

                if args.series_check_every and index % args.series_check_every == 0:
                    for n in range(NMAX + 1):
                        imported = generator.boys_series(n, x)
                        worst["R2b imported series"].add(
                            relative_difference(imported, values[n]),
                            "imported series %s n=%d x=%s" % (band, n, x_text),
                        )

                for n in range(NMAX):
                    residual = abs(
                        (2 * n + 1) * values[n] - 2 * x * values[n + 1] - mp.exp(-x)
                    ) / mp.exp(-x)
                    worst["R5 identity"].add(
                        residual, "%s n=%d x=%s" % (band, n, x_text)
                    )

                worst["R6 F0 gamma against erf"].add(
                    relative_difference(generator.boys_gamma(0, x), values[0]),
                    "%s x=%s" % (band, x_text),
                )

                if args.quad_every and index % args.quad_every == 0:
                    for n in range(NMAX + 1):
                        quadrature = generator.boys_quad(n, x)
                        worst["R3 quadrature"].add(
                            relative_difference(quadrature, values[n]),
                            "%s n=%d x=%s" % (band, n, x_text),
                        )

            for n in range(NMAX + 1):
                key = (n, x_float)
                if key in committed_45:
                    worst["R4 committed grid (absolute)"].add(
                        abs(values[n] - committed_45[key]),
                        "%s n=%d x=%s" % (band, n, x_text),
                    )
                if key in gate_grid:
                    worst["R7 accuracy-gate grid"].add(
                        relative_difference(gate_grid[key], values[n]),
                        "%s n=%d x=%s" % (band, n, x_text),
                    )

    lines = [
        "reference validation",
        "grid: %s" % Path(args.grid).resolve(),
        "arguments: %d, orders: %d, dps: %d, digits: %d"
        % (arguments, NMAX + 1, args.dps, args.digits),
        "",
    ]
    failures = []
    for name in ROUTE_THRESHOLDS:
        lines.append("%-32s %s" % (name, worst[name].format()))
        value = worst[name].value
        if value is not None and value > ROUTE_THRESHOLDS[name]:
            failures.append(name)

    lines += ["", "strict: %s" % ("off" if args.no_strict else "on")]
    if failures:
        lines.append("FAILED: " + ", ".join(failures))
    else:
        lines.append("all evaluated routes inside their stated bounds")

    validation_path = Path(args.validation) if args.validation else out_path.with_suffix(".validation.txt")
    validation_path.parent.mkdir(parents=True, exist_ok=True)
    with open(validation_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    if failures and not args.no_strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
