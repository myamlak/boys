#!/usr/bin/env python3
"""The first sweep's argument grid.

The designed point sets are the paper's Table 1, applied to the two
committed bands at the v5.0.0 constants:

    U  (b - a) k / 2048, k = 0 .. 2047
    C  the midpoint plus half-width times cos(pi (k + 1/2) / 2048)
    L  a + (b - a) 10^p, p log-spaced over [-12, -1], 128 values
    R  b - (b - a) 10^p, the same p
    G  every argument of the committed accuracy-gate grid inside the band
    T  each dispatch threshold inside the band, one ULP either side
    E  the exact band endpoints and, in A1, x = 0

The union is de-duplicated per band.  The two L/R conventions (128 points,
both ends included) are reconstructed from the surviving at_x values of the
lost results files; the paper states the rule but not the count.  G is read
from `tests/data/boys_accuracy_gate_reference.csv` of the pinned boys tree,
so it is the committed grid of this revision rather than of the lost one.

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


def band_points(left, right, thresholds, gates):
    points = {}

    for k in range(2048):
        add(points, left + (right - left) * k / 2048.0, "U")

    center = 0.5 * (left + right)
    half = 0.5 * (right - left)
    for k in range(2048):
        add(points, center + half * math.cos(math.pi * (k + 0.5) / 2048.0), "C")

    for j in range(128):
        exponent = -12.0 + 11.0 * j / 127.0
        offset = (right - left) * 10.0 ** exponent
        add(points, left + offset, "L")
        add(points, right - offset, "R")

    for value in gates:
        if left <= value < right:
            add(points, value, "G")

    for threshold in thresholds:
        if left <= threshold < right:
            add(points, math.nextafter(threshold, -math.inf), "T")
            add(points, threshold, "T")
            add(points, math.nextafter(threshold, math.inf), "T")

    add(points, left, "E")
    previous = math.nextafter(right, -math.inf)
    add(points, previous, "E")
    add(points, math.nextafter(previous, -math.inf), "E")
    if left == 0.0:
        add(points, 0.0, "E")

    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = Path(args.boys_tree).resolve()
    constants = boysprovenance.parse_constants(tree)
    thresholds = sorted(set(constants["kTierThresholds"]))
    gates = gate_arguments(tree)
    edge = constants["kRegionA1Edge"]
    end = constants["kRegionAEnd"]

    bands = [("A1", 0.0, edge), ("A2", edge, end)]
    rows = []
    for band, left, right in bands:
        points = band_points(left, right, thresholds, gates)
        print(
            "%s: %d unique arguments, band [%r, %r)"
            % (band, len(points), left, right)
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
    print("wrote %s: %d arguments" % (out, len(rows)))


if __name__ == "__main__":
    main()
