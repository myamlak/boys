#!/usr/bin/env python3
"""Reduce the first sweep's raw cells to the manuscript's tables.

Inputs (under --tree):
    data/grid.csv, data/reference.csv
    results/cells.csv, results/admission.csv, results/replica_gain.csv
    results/provenance.json (or --provenance)

Outputs (under --tree):
    results/per_order.csv, band_summary.csv, gain_up.csv, growth.csv
    results/decomposition.csv, results/summary.txt
    data/documented.csv

make_tables.py reads the first four and documented.csv; the other two are
the re-derivation's own evidence.  All error arithmetic is done in mpmath at
70 digits from the 60-digit reference, and the computed doubles are parsed
as exact doubles (`mpf(float(text))`), so a sub-ULP error is not lost.

Usage:
    python analyze.py --tree .
"""

import argparse
import csv
import json
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import boysprovenance  # noqa: E402

DEFAULT_TREE = boysprovenance.DEFAULT_TREE
NMAX = 32


def read_rows(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def read_reference(path):
    reference = {}
    with open(path, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)

        if header[:2] != ["band", "x"]:
            raise SystemExit("unexpected reference header: %r" % (header[:2],))

        for row in reader:
            key = (row[0], float(row[1]))
            reference[key] = [mpf(text) for text in row[2:]]
    return reference


def text(value):
    return mp.nstr(value, 17)


def amplification(n, x):
    if n == 0:
        return mpf(1)
    odd = math.factorial(2 * n) // (2**n * math.factorial(n))
    return mpf(odd) / (2 * x) ** n


def update(store, key, value, x_text):
    current = store.get(key)

    if current is None or value > current[0]:
        store[key] = (value, x_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--tree", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--provenance", default=None)
    parser.add_argument("--validation", default=None)
    args = parser.parse_args()

    tree = Path(args.tree).resolve()
    reference_path = tree / "data/reference.csv"
    provenance_path = Path(args.provenance) if args.provenance else tree / "results/provenance.json"

    if provenance_path.is_file():
        with open(provenance_path, encoding="utf-8") as handle:
            provenance = json.load(handle)
        constants = provenance["constants"]
    else:
        constants = boysprovenance.parse_constants(Path(args.boys_tree).resolve())

    edge = float(constants["kRegionA1Edge"])
    end = float(constants["kRegionAEnd"])

    mp.dps = 70
    reference = read_reference(reference_path)

    ship_abs = {}
    ship_rel = {}
    prod_abs = {}
    prod_rel = {}
    growth = {}
    ship_error0 = {}
    band_ship = {"A1": {}, "A2": {}}
    band_prod = {"A1": {}, "A2": {}}
    max_ratio = {"A1": None, "A2": None}
    rel_max_ratio = {"A1": None, "A2": None}
    paper_cells = {
        ("A1", 16, float("4.9911960755181584")): "paper A1 decomposition",
        ("A2", 32, float("11.417327754350195")): "paper A2 decomposition",
    }
    paper_values = {}
    previous_key = None
    error0 = None
    cells = 0

    for row in read_rows(tree / "results/cells.csv"):
        name = row["band"]
        x_text = row["x"]
        x = float(x_text)
        n = int(row["n"])
        key = (name, x)

        if key != previous_key:
            previous_key = key
            error0 = None

        if key not in reference:
            raise SystemExit("cell argument missing from reference: %s %s" % (name, x_text))

        values = reference[key]
        recursed = abs(mpf(float(row["recursed"])) - values[n])
        direct = abs(mpf(float(row["direct"])) - values[n])
        value = values[n]
        recursed_rel = recursed / value if value != 0 else mpf("inf")
        direct_rel = direct / value if value != 0 else mpf("inf")

        update(ship_abs, (name, n), recursed, x_text)
        update(prod_abs, (name, n), direct, x_text)
        update(ship_rel, (name, n), recursed_rel, x_text)
        update(prod_rel, (name, n), direct_rel, x_text)

        for metric, value_now in (("absolute", recursed), ("relative", recursed_rel)):
            entry = band_ship[name].get(metric)

            if entry is None or value_now > entry[0]:
                band_ship[name][metric] = (value_now, n, x_text)

        for metric, value_now in (("absolute", direct), ("relative", direct_rel)):
            entry = band_prod[name].get(metric)

            if entry is None or value_now > entry[0]:
                band_prod[name][metric] = (value_now, n, x_text)

        if n == 0:
            error0 = recursed
            ship_error0[key] = recursed

        if n > 0 and error0 is not None and error0 > 0:
            update(growth, (name, n), recursed / error0, x_text)

        if direct > 0:
            ratio = recursed / direct
            entry = max_ratio[name]

            if entry is None or ratio > entry[0]:
                max_ratio[name] = (ratio, n, x_text, recursed, direct, x)

        if direct_rel > 0:
            ratio = recursed_rel / direct_rel
            entry = rel_max_ratio[name]

            if entry is None or ratio > entry[0]:
                rel_max_ratio[name] = (ratio, n, x_text, recursed_rel, direct_rel, x)

        for cell, label in paper_cells.items():
            if cell[0] == name and cell[1] == n and cell[2] == x:
                paper_values[label] = (recursed, direct, x_text)

        cells += 1

    arguments = {}
    for name in ("A1", "A2"):
        arguments[name] = len({key for key in reference if key[0] == name})

    results = tree / "results"
    results.mkdir(parents=True, exist_ok=True)

    with open(results / "per_order.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "n", "method", "worst_abs", "at_x_abs", "worst_rel", "at_x_rel"])

        for name in ("A1", "A2"):
            for n in range(NMAX + 1):
                for method, store_abs, store_rel in (
                    ("ship", ship_abs, ship_rel),
                    ("product", prod_abs, prod_rel),
                ):
                    worst_abs, at_abs = store_abs[(name, n)]
                    worst_rel, at_rel = store_rel[(name, n)]
                    writer.writerow(
                        [name, n, method, text(worst_abs), at_abs, text(worst_rel), at_rel]
                    )

    with open(results / "band_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "band",
                "metric",
                "ship_worst",
                "ship_at_n",
                "ship_at_x",
                "prod_worst",
                "prod_at_n",
                "prod_at_x",
                "ratio_of_band_worsts",
                "max_cell_ratio",
                "max_cell_n",
                "max_cell_x",
            ]
        )

        for name in ("A1", "A2"):
            for metric in ("absolute", "relative"):
                ship_worst, ship_n, ship_x = band_ship[name][metric]
                prod_worst, prod_n, prod_x = band_prod[name][metric]
                ratio = ship_worst / prod_worst if prod_worst > 0 else mpf("inf")
                best = rel_max_ratio[name] if metric == "relative" else max_ratio[name]
                writer.writerow(
                    [
                        name,
                        metric,
                        text(ship_worst),
                        ship_n,
                        ship_x,
                        text(prod_worst),
                        prod_n,
                        prod_x,
                        text(ratio),
                        text(best[0]),
                        best[1],
                        best[2],
                    ]
                )

    gain = {}
    down_gain = {"A1": None, "A2": None}
    ungated = {"A1": None, "A2": None}
    gain_formula_error = None
    gain_formula_where = None

    for row in read_rows(tree / "results/replica_gain.csv"):
        case = row["case"]
        name = row["band"]
        n = int(row["n"])
        x = float(row["x"])
        value = mpf(row["gain"])

        if case == "up":
            if n == 0:
                continue
            update(gain, (name, n), value, row["x"])
        elif case == "down":
            entry = down_gain[name]
            if entry is None or value > entry[0]:
                down_gain[name] = (value, n, row["x"])
        elif case == "up_ungated":
            entry = ungated[name]
            if entry is None or value > entry[0]:
                ungated[name] = (value, n, row["x"])

    for (name, n), (value, x_text) in gain.items():
        predicted = amplification(n, mpf(float(x_text)))

        if predicted > 0:
            error = abs(value - predicted) / predicted
            if gain_formula_error is None or error > gain_formula_error:
                gain_formula_error = error
                gain_formula_where = "%s n=%d x=%s" % (name, n, x_text)

    with open(results / "gain_up.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "n", "worst_gain_over_x", "at_x"])

        for name in ("A1", "A2"):
            for n in range(NMAX + 1):
                if n == 0 or (name, n) not in gain:
                    writer.writerow([name, n, "-1", "0"])
                else:
                    value, x_text = gain[(name, n)]
                    writer.writerow([name, n, text(value), x_text])

    with open(results / "growth.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["band", "n", "worst_err_ratio_ship_n_over_ship_0", "at_x"])

        for name in ("A1", "A2"):
            for n in range(NMAX + 1):
                if n == 0 or (name, n) not in growth:
                    writer.writerow([name, n, "-1", "0"])
                else:
                    value, x_text = growth[(name, n)]
                    writer.writerow([name, n, text(value), x_text])

    documented_path = tree / "data/documented.csv"
    with open(documented_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "w_at_band_right_end"])

        for n in range(NMAX + 1):
            writer.writerow([n, text(1 / amplification(n, mpf(end)))])

    with open(results / "decomposition.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "label",
                "band",
                "metric",
                "n",
                "x",
                "recursed_error",
                "direct_error",
                "ratio",
                "recursed_order0_error",
                "recursion_gain",
                "seed_factor",
                "growth_ship",
            ]
        )

        for name in ("A1", "A2"):
            best = max_ratio[name]
            ratio, n, x_text, recursed, direct, x = best
            error0 = ship_error0.get((name, x))
            gain_n = amplification(n, mpf(x))
            seed_factor = error0 / direct if error0 is not None and direct > 0 else mpf("inf")
            growth_value = recursed / error0 if error0 else mpf("inf")
            writer.writerow(
                [
                    "largest per-cell ratio",
                    name,
                    "absolute",
                    n,
                    x_text,
                    text(recursed),
                    text(direct),
                    text(ratio),
                    text(error0) if error0 is not None else "nan",
                    text(gain_n),
                    text(seed_factor),
                    text(growth_value),
                ]
            )

        for label, payload in paper_values.items():
            recursed, direct, x_text = payload
            name = label.split()[1]
            n = None
            for cell in paper_cells:
                if paper_cells[cell] == label:
                    n = cell[1]
            error0 = ship_error0.get((name, float(x_text)))
            writer.writerow(
                [
                    label,
                    name,
                    "absolute",
                    n,
                    x_text,
                    text(recursed),
                    text(direct),
                    text(recursed / direct) if direct > 0 else "inf",
                    text(error0) if error0 is not None else "nan",
                    text(amplification(n, mpf(float(x_text)))),
                    text(error0 / direct) if error0 is not None and direct > 0 else "inf",
                    text(recursed / error0) if error0 else "nan",
                ]
            )

    admission_rows = read_rows(tree / "results/admission.csv")
    admission_summary = {}
    mismatches = []

    for row in admission_rows:
        direction = row["direction"]
        checked = int(row["checked"])
        matched = int(row["matched"])
        entry = admission_summary.setdefault(direction, [0, 0, 0])
        entry[0] += 1
        entry[1] += checked
        entry[2] += matched

        if checked != matched:
            mismatches.append((row["band"], row["x"], direction, checked, matched))

    validation_lines = []
    validation_path = Path(args.validation) if args.validation else tree / "data/reference_validation.txt"
    if validation_path.is_file():
        validation_lines = validation_path.read_text(encoding="utf-8").splitlines()

    summary = [
        "first sweep summary",
        "reference: %s" % reference_path.resolve(),
        "arguments: A1 %d, A2 %d, total %d; cells %d" % (arguments["A1"], arguments["A2"], arguments["A1"] + arguments["A2"], cells),
        "",
        "replica admission:",
    ]
    for direction in sorted(admission_summary):
        entries, checked, matched = admission_summary[direction]
        summary.append(
            "  %-9s arguments %5d, cells %7d, bit-identical %7d%s"
            % (direction, entries, checked, matched, "" if checked == matched else "  MISMATCH")
        )
    for mismatch in mismatches:
        summary.append("  mismatch: %s" % (mismatch,))

    summary += ["", "gain replica against the closed form:"]
    if gain_formula_error is not None:
        summary.append(
            "  worst relative difference %.3e at %s" % (float(gain_formula_error), gain_formula_where)
        )
    else:
        summary.append("  no gain rows")

    for name in ("A1", "A2"):
        if down_gain[name] is not None:
            value, n, x_text = down_gain[name]
            predicted = amplification(n, mpf(float(x_text))) / amplification(NMAX, mpf(float(x_text)))
            summary.append(
                "  downward window %s: worst accumulated gain %s at n=%d x=%s (closed form %s)"
                % (name, text(value), n, x_text, text(predicted))
            )
        if ungated[name] is not None:
            value, n, x_text = ungated[name]
            predicted = amplification(n, mpf(float(x_text)))
            summary.append(
                "  ungated upward %s: A_%d at x=%s measured %s, closed form %s"
                % (name, n, x_text, text(value), text(predicted))
            )

    summary += ["", "band worsts:"]
    for name in ("A1", "A2"):
        for metric in ("absolute", "relative"):
            ship_worst, ship_n, ship_x = band_ship[name][metric]
            prod_worst, prod_n, prod_x = band_prod[name][metric]
            ratio = ship_worst / prod_worst if prod_worst > 0 else mpf("inf")
            summary.append(
                "  %s %-8s recursed %s at (n=%d, x=%s), direct %s at (n=%d, x=%s), ratio %s"
                % (
                    name,
                    metric,
                    text(ship_worst),
                    ship_n,
                    ship_x,
                    text(prod_worst),
                    prod_n,
                    prod_x,
                    text(ratio),
                )
            )

    best = max_ratio["A1"]
    summary.append(
        "  largest A1 per-cell ratio %s at (n=%d, x=%s), recursed %s, direct %s"
        % (text(best[0]), best[1], best[2], text(best[3]), text(best[4]))
    )
    best = max_ratio["A2"]
    summary.append(
        "  largest A2 per-cell ratio %s at (n=%d, x=%s), recursed %s, direct %s"
        % (text(best[0]), best[1], best[2], text(best[3]), text(best[4]))
    )

    summary += [
        "",
        "point-count deviation: the paper reports 9354 arguments (4812 A1, 4542 A2).",
        "This grid yields %d A1 and %d A2 arguments, %d in total."
        % (arguments["A1"], arguments["A2"], arguments["A1"] + arguments["A2"]),
        "The two missing A2 arguments are not recoverable from the surviving records and",
        "are not synthesised. The argument 5.949924076060192 in the surviving gain table",
        "is not one of them: it is the p = -12 L point of A2 and this grid contains it.",
        "",
        "reference validation:",
    ]
    summary += ["  " + line for line in validation_lines]
    summary += [
        "",
        "reconstructed protocol: the L/R sets use 128 log-spaced points, both ends included,",
        "recovered from surviving at_x values; the paper states the rule but not the count.",
    ]

    with open(results / "summary.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")

    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
