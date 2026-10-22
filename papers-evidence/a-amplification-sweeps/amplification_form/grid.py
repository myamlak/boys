#!/usr/bin/env python3
"""The second sweep's argument list.

The manuscript states 836 distinct arguments in Section 5.1.  The design that
accounts for that number is the committed accuracy-gate grid plus a
Chebyshev-variable set in each band:

    G  every argument of the committed gate grid inside either band
    V  95 Chebyshev-Gauss points in the band's own mapped variable,
       t = cos(pi (j + 1/2) / 95), j = 0 .. 94, x = centre + half t

The variable points are a reconstruction: the lost tree's selection is not
recoverable from the surviving records.  They are placed in the Chebyshev
variable because the experiment varies the seed-fit degree, so its argument
set has to resolve the fit on the same variable the fit is written in.

Overlaps with G are de-duplicated; the actual total is printed and recorded
in the grid file.  If it is not 836, the difference is reported rather than
padded.

Usage:
    python grid.py --out data/grid.csv
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import boysprovenance  # noqa: E402

DEFAULT_TREE = boysprovenance.DEFAULT_TREE
VARIABLE_POINTS = 95


def gate_arguments(tree):
    path = Path(tree) / "tests/data/boys_accuracy_gate_reference.csv"
    values = set()
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            values.add(float(row["x"]))
    return values


def add(store, value, tag):
    if not math.isfinite(value):
        return
    store.setdefault(value, set()).add(tag)


def band_points(left, right, gates):
    points = {}

    for value in gates:
        if left <= value < right:
            add(points, value, "G")

    center = 0.5 * (left + right)
    half = 0.5 * (right - left)
    for j in range(VARIABLE_POINTS):
        add(points, center + half * math.cos(math.pi * (j + 0.5) / VARIABLE_POINTS), "V")

    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = Path(args.boys_tree).resolve()
    constants = boysprovenance.parse_constants(tree)
    gates = gate_arguments(tree)
    edge = constants["kRegionA1Edge"]
    end = constants["kRegionAEnd"]

    rows = []
    for band, left, right in (("A1", 0.0, edge), ("A2", edge, end)):
        points = band_points(left, right, gates)
        gate_count = sum(1 for tags in points.values() if "G" in tags)
        variable_count = sum(1 for tags in points.values() if "V" in tags)
        print(
            "%s: %d arguments (%d gate, %d variable points, %d overlap)"
            % (band, len(points), gate_count, variable_count, gate_count + variable_count - len(points))
        )
        for value in sorted(points):
            rows.append((band, value, "+".join(sorted(points[value]))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "x", "sets"])
        for band, value, tags in rows:
            writer.writerow([band, repr(value), tags])

    expected = 836
    print("wrote %s: %d arguments (manuscript: %d)" % (out, len(rows), expected))
    if len(rows) != expected:
        print("deviation: %+d arguments; recorded, not padded" % (len(rows) - expected))


if __name__ == "__main__":
    main()
