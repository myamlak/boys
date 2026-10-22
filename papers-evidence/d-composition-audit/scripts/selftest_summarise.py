#!/usr/bin/env python3
"""Synthetic pre-flight check for summarise_interaction.py.

This is not a library measurement and reads no Boys output. It builds a cells CSV
in a temporary directory with the driver's exact schema and plants a known
form x exp interaction: 25% of the largest response pairwise, but 8/144 = 5.56%
of the point's centred sum of squares in the multi-way decomposition. The two
lenses therefore cross the margins differently on purpose: the pairwise term is
material at 10% and 20%, while the order-2 variance share is material only at
5%. The check exercises both definitions, the argument profile and the
lane-aware cell count. It then plants a single figure ratio above 1 and checks
that the verdict becomes `bound-violated`.

A third fixture is the bf16 row the smoke run produced: its absolute error
exceeds the accessor figure by a factor of about 4869, and is inside the row's
stated limit once the value's half digit is added. That row must be reported as
an accessor-only excess, not a violation, and the half-lane cell must be kept
out of the interaction verdict while still being counted and safety-checked.

The driver is not built or run. The script exists so the one probe run is not
spent on a summary that cannot parse or that silently drops a finding.

Usage:
    python selftest_summarise.py

Exit 0 all checks passed, 1 a check failed or the summariser failed.
"""

import csv
import itertools
import json
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import summarise_interaction as summary  # noqa: E402

LANES = ["fp64", "fp32"]
POINTS = [(1.0, "exact"), (1.3, "generic")]
ORDERS = range(33)

HEADER = [
    "precision",
    "route",
    "scheme",
    "partition",
    "pack",
    "form",
    "exp",
    "n",
    "x_nominal",
    "x",
    "value",
    "ref_hi",
    "ref_lo",
    "abs_err",
    "f0",
    "fn",
    "figure",
    "half_digit",
    "limit",
    "accessor_ratio",
    "figure_ratio",
    "pack_live",
    "partition_in_domain",
    "argument_kind",
    "status",
]


def base_ratio(form, exp):
    """The planted response: base + form + exp + a form x exp interaction."""
    ratio = 1e-12
    if form == "plain-reciprocal":
        ratio += 1e-12
    if exp == "fast":
        ratio += 1e-12
    if form == "plain-reciprocal" and exp == "fast":
        ratio += 1e-12
    return ratio


def write_cells(path, violation):
    """Write the synthetic cells CSV; optionally plant one bound violation."""
    steps = [summary.LEVELS[axis] for axis in summary.AXES]
    rows = 0

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)

        for precision in LANES:
            for combo in itertools.product(*steps):
                cell = dict(zip(summary.AXES, combo))
                if cell["precision"] != precision:
                    continue
                for nominal, kind in POINTS:
                    for n in ORDERS:
                        ratio = base_ratio(cell["form"], cell["exp"])
                        f0 = 1.0
                        fn = 1.0 / (2 * n + 1)
                        abs_err = ratio * f0
                        figure_ratio = 0.5
                        if (
                            violation
                            and precision == "fp64"
                            and cell["route"] == "chebyshev"
                            and cell["scheme"] == "split-clenshaw"
                            and cell["partition"] == "shipped"
                            and cell["pack"] == "arguments"
                            and cell["form"] == "exact-division"
                            and cell["exp"] == "accurate"
                            and n == 0
                            and nominal == 1.0
                        ):
                            figure_ratio = 1.2
                        figure = abs_err / figure_ratio

                        writer.writerow(
                            [
                                precision,
                                cell["route"],
                                cell["scheme"],
                                cell["partition"],
                                cell["pack"],
                                cell["form"],
                                cell["exp"],
                                n,
                                nominal,
                                nominal,
                                fn + abs_err,
                                fn,
                                0.0,
                                abs_err,
                                f0,
                                fn,
                                figure,
                                0.0,
                                figure,
                                figure_ratio,
                                figure_ratio,
                                1,
                                1,
                                kind,
                                "ok",
                            ]
                        )
                        rows += 1

    return rows


def write_half_lane_cells(path):
    """One bf16 row whose accessor figure is exceeded but whose limit holds.

    The numbers are the smoke run's bf16 n=0, x=1 row: the returned value is
    0.74609375, the accessor figure is 1.5e-7, and the row adds half of the
    format's last representable digit (2^-9 at this value). The accessor ratio
    is about 4869; the limit ratio is about 0.37. This is the shape the driver
    must classify as inside the contract.
    """
    value = 0.74609375
    ref_hi = 0.74682413281242699
    abs_err = 0.00073038281242702541
    figure = 1.5e-07
    half_digit = 2.0 ** -9
    limit = figure + half_digit
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerow(
            [
                "bf16",
                "chebyshev",
                "split-clenshaw",
                "shipped",
                "arguments",
                "exact-division",
                "accurate",
                0,
                1.0,
                1.0,
                value,
                ref_hi,
                0.0,
                abs_err,
                ref_hi,
                ref_hi,
                figure,
                half_digit,
                limit,
                abs_err / figure,
                abs_err / limit,
                1,
                1,
                "exact",
                "ok",
            ]
        )


def run_summary(cells, out):
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "summarise_interaction.py")
    proc = subprocess.run(
        [sys.executable, script, "--cells", cells, "--out", out],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise RuntimeError(f"summariser exited {proc.returncode}")
    with open(out, encoding="utf-8") as f:
        return json.load(f)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def check_interaction(data, rows):
    counts = data["counts"]
    require(data["verdict"] == "interaction-material",
            f"verdict is {data['verdict']!r}, expected interaction-material")
    require(counts["data_rows"] == rows, "the summariser lost rows")
    require(counts["cells"] == 288, f"cells {counts['cells']}, expected 288")
    require(counts["expected_cells"] == 288,
            f"expected_cells {counts['expected_cells']}, expected 288")
    require(counts["precisions_present"] == ["fp32", "fp64"],
            f"precisions_present {counts['precisions_present']}")

    pair = next(p for p in data["pairs"] if p["pair"] == ["form", "exp"])
    require(pair["max_relative"] >= 0.24,
            f"form x exp max_relative {pair['max_relative']}, expected about 0.25")
    require(pair["material_at_primary"] > 0,
            "the planted form x exp interaction is not material at 10%")
    require(pair["material"]["0.2"] > 0,
            "the planted pairwise interaction should also be material at 20%")
    require(pair["profiled_contrasts"] > 0 and pair["profiled_sign_changes"] == 0,
            "the argument profile is missing or changes sign unexpectedly")

    within = data["multiway"]["within_lane"]["fp64"]
    require(within["points"] == 66,
            f"within-lane points {within['points']}, expected 66")
    require(within["point_kinds"] == {"exact": 33, "generic": 33},
            f"point kinds {within['point_kinds']}")
    order_two = within["orders"]["2"]
    require(order_two["material_points"]["0.05"]["all"] == 66,
            "the 5.56% order-2 variance share is not material at 5% everywhere")
    require(order_two["material_points"]["0.1"]["all"] == 0,
            "the 5.56% order-2 variance share must not be material at 10%")
    require(order_two["material_points"]["0.2"]["all"] == 0,
            "the 5.56% order-2 variance share must not be material at 20%")
    require(abs(order_two["ss_share"] - 8.0 / 144.0) < 1e-9,
            f"order-2 pooled share {order_two['ss_share']}, expected 8/144")
    require(abs(order_two["max_order_share"] - 8.0 / 144.0) < 1e-9,
            f"order-2 point share {order_two['max_order_share']}, expected 8/144")
    require(abs(order_two["max_component_share"] - 8.0 / 144.0) < 1e-9,
            f"order-2 component share {order_two['max_component_share']}, "
            "expected 8/144")
    require(within["top_components"],
            "no multi-way interaction component was reported")
    component = next((t for t in within["top_components"]
                      if t["factors"] == ["form", "exp"]), None)
    require(component is not None,
            "the form x exp component was not selected as a top component")
    require(len(component["by_x"]) == 2,
            "the form x exp component has no two-argument profile")
    require(abs(component["max_share"] - 8.0 / 144.0) < 1e-9,
            f"component share {component['max_share']}, expected 8/144")
    require(component["argument_dependence"]["max_share"] is not None,
            "the component profile does not report a variance share")

    cross = data["multiway"]["cross_lane_exact"]
    require(cross["points"] == 33,
            f"the resolving cross-lane block has {cross['points']} points, expected 33")
    require(cross["levels"]["precision"] == ["fp64", "fp32"],
            f"resolving cross-lane levels {cross['levels']['precision']}")
    cross_all = data["multiway"]["cross_lane_exact_all_lanes"]
    require(cross_all["points"] == 0 and cross_all["missing_count"] == 288,
            "the two-lane fixture should report the missing half-lane factorial "
            "in the all-lane block")

    require(data["verdict_by_alpha"]["0.05"] == "interaction-material"
            and data["verdict_by_alpha"]["0.1"] == "interaction-material"
            and data["verdict_by_alpha"]["0.2"] == "interaction-material",
            f"per-margin verdicts {data['verdict_by_alpha']}")

    require(len(within["main_effects"]) == 8,
            f"{len(within['main_effects'])} main-effect components, expected 8")
    require({e["factor"] for e in within["main_effects"]}
            == {"route", "scheme", "partition", "pack", "form", "exp"},
            "the main-effect report does not cover every factor")
    require(within["orders"]["1"]["material_points"]["0.1"]["all"] > 0,
            "the planted form main effect is not material at 10%")
    require(any(s["factors"] == ["form"] for s in within["sets"]),
            "the form main effect has no factor-set entry")

    form_live = data["liveness"]["axis_value_changes"]["form"]["fp64"]
    require(form_live["live_points"] > 0 and form_live["live_points"] <= form_live["points"],
            f"form liveness {form_live}")
    require(data["liveness"]["pack_by_lane"]["fp64"]["rows"] == 9504,
            "the pack liveness row count is wrong")

    # The safety block is limit-based. This fixture has no half lane, so the
    # accessor and the limit coincide and nothing is an accessor-only excess.
    require(data["safety"]["accessor_only_exceeded"] == 0,
            "the fp64/fp32 fixture reports an accessor-only excess")
    require(data["safety"]["by_lane"]["fp64"]["violations"] == 0,
            "the clean fp64 fixture reports a violation")


def check_violation(data):
    require(data["verdict"] == "bound-violated",
            f"verdict is {data['verdict']!r}, expected bound-violated")
    require(data["safety"]["violation_count"] == 1,
            f"violation count {data['safety']['violation_count']}, expected 1")
    require(data["safety"]["violations"][0]["figure_ratio"] > 1.2 - 1e-12,
            "the planted violation is not the reported one")


def check_half_lane(data):
    require(data["safety"]["violation_count"] == 0,
            "the half-lane row is inside its limit and must not be a violation")
    require(data["safety"]["accessor_only_exceeded"] == 1,
            f"accessor-only excesses {data['safety']['accessor_only_exceeded']}, "
            "expected 1")
    require(data["safety"]["max_accessor_ratio"] > 1000.0,
            "the accessor ratio is not the expected oversized one")
    require(data["safety"]["max_figure_ratio"] < 1.0,
            "the limit ratio should be below 1")
    require(data["counts"]["store_dominated_cells"] == 1
            and data["counts"]["resolving_cells"] == 0,
            f"lane split {data['counts']['resolving_cells']} / "
            f"{data['counts']['store_dominated_cells']}")
    require(data["verdict"] == "inconclusive",
            f"verdict {data['verdict']!r}, expected inconclusive: one store lane "
            "has no interaction contrast to report")


def main():
    with tempfile.TemporaryDirectory(prefix="composed-bounds-selftest-") as tmp:
        cells = os.path.join(tmp, "cells.csv")
        out = os.path.join(tmp, "summary.json")

        rows = write_cells(cells, violation=False)
        data = run_summary(cells, out)
        check_interaction(data, rows)
        print(f"interaction fixture: {rows} rows, verdict {data['verdict']}")

        rows = write_cells(cells, violation=True)
        data = run_summary(cells, out)
        check_violation(data)
        print(f"violation fixture: {rows} rows, verdict {data['verdict']}")

        write_half_lane_cells(cells)
        data = run_summary(cells, out)
        check_half_lane(data)
        print(f"half-lane fixture: 1 row, verdict {data['verdict']}")

    print("selftest: ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"selftest FAILED: {error}", file=sys.stderr)
        sys.exit(1)
    except (OSError, RuntimeError, KeyError, ValueError) as error:
        print(f"selftest FAILED: {error}", file=sys.stderr)
        sys.exit(1)
