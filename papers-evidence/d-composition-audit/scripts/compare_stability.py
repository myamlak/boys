#!/usr/bin/env python3
"""Compare the four stability configurations of the composed-bounds probe.

The window run is repeated in four configurations: two targets (MSVC on Windows,
GCC on Linux) crossed with the two multiply-add routes the library supports
(BOYS_MULADD_SEPARATE off and on). Each configuration writes its own results
directory with the driver's `summary.json`, `environment.txt` and
`arguments.csv`.

This script reads those directories and asks one question: does the verdict the
summariser reached depend on the target or the arithmetic route? The agreement
rule is pre-registered in PROBE.md Section 5 and is deliberately coarse: the
verdict at every margin, the set of axis pairs with a material contrast at the
primary margin, and the set of material interaction factor sets per resolving
lane must be identical. Magnitudes are reported with their spread across
configurations but do not gate the verdict; the paper quotes the range.

Exit 0 for a completed comparison whatever the stability outcome, because an
unstable result is a finding and not a failure of this script. Exit 2 for a
usage or input error: a missing run directory, a missing file, or fewer than two
runs. The stability verdict is in the JSON and on the console.

Usage:
    python compare_stability.py --run results --run results-msvc-separate \
        --run results-gcc-fused --run results-gcc-separate --out stability.json
"""

import argparse
import hashlib
import json
import os
import sys

PRIMARY_ALPHA = "0.1"
RESOLVING_LANES = ["fp64", "fp32"]


def fail(message):
    print(message, file=sys.stderr)
    raise SystemExit(2)


def read_environment(path):
    facts = {}
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                facts[key] = value
    except OSError as error:
        fail(f"compare_stability: cannot read '{path}': {error}")
    return facts


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as error:
        fail(f"compare_stability: cannot read '{path}': {error}")
    except json.JSONDecodeError as error:
        fail(f"compare_stability: '{path}' is not valid JSON: {error}")


def file_digest(path):
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        fail(f"compare_stability: cannot hash '{path}': {error}")
    return digest.hexdigest()


def label_of(directory, environment):
    if "stability_configuration" in environment:
        return environment["stability_configuration"]
    target = environment.get("target")
    route = environment.get("muladd_route") or environment.get("muladd_fp64", "unknown")
    if target:
        return f"{target}-{route}"
    return os.path.basename(directory) or directory


def material_sets(block):
    """The interaction factor sets with at least one point material at 10%."""
    selected = []
    for entry in block.get("sets", []):
        factors = entry.get("factors", [])
        if len(factors) < 2:
            continue
        points = entry.get("material_points", {})
        if points.get(PRIMARY_ALPHA, 0) > 0:
            selected.append("+".join(factors))
    return sorted(selected)


def material_pairs(summary):
    """The axis pairs with at least one contrast material at the primary margin."""
    selected = []
    for pair in summary.get("pairs", []):
        if pair.get("material", {}).get(PRIMARY_ALPHA, 0) > 0:
            selected.append("+".join(pair.get("pair", [])))
    return sorted(selected)


def max_interaction_share(block):
    shares = [
        entry.get("max_set_share", 0.0)
        for entry in block.get("sets", [])
        if len(entry.get("factors", [])) >= 2
    ]
    return max(shares, default=0.0)


def fingerprint(summary):
    within = summary.get("multiway", {}).get("within_lane", {})
    sets = {}
    shares = {}
    for lane in RESOLVING_LANES:
        block = within.get(lane)
        if block and block.get("used_in_verdict"):
            sets[lane] = material_sets(block)
            shares[lane] = max_interaction_share(block)
    cross = summary.get("multiway", {}).get("cross_lane_exact")
    if cross and cross.get("points", 0) > 0:
        sets["cross_lane_exact"] = material_sets(cross)
        shares["cross_lane_exact"] = max_interaction_share(cross)
    return {
        "verdict": summary.get("verdict"),
        "verdict_by_alpha": summary.get("verdict_by_alpha", {}),
        "violation_count": summary.get("safety", {}).get("violation_count"),
        "max_figure_ratio": summary.get("safety", {}).get("max_figure_ratio"),
        "max_figure_ratio_at": summary.get("safety", {}).get("max_figure_ratio_at"),
        "cells": summary.get("counts", {}).get("cells"),
        "expected_cells": summary.get("counts", {}).get("expected_cells"),
        "precisions_present": summary.get("counts", {}).get("precisions_present"),
        "material_pairs": material_pairs(summary),
        "material_sets": sets,
        "max_interaction_share": shares,
        "pair_max_relative": {
            "+".join(pair.get("pair", [])): pair.get("max_relative")
            for pair in summary.get("pairs", [])
        },
    }


def load_run(directory):
    summary_path = os.path.join(directory, "summary.json")
    environment_path = os.path.join(directory, "environment.txt")
    arguments_path = os.path.join(directory, "arguments.csv")
    for path in (summary_path, environment_path, arguments_path):
        if not os.path.isfile(path):
            fail(
                f"compare_stability: '{directory}' is not a run directory: "
                f"'{os.path.basename(path)}' is missing"
            )
    environment = read_environment(environment_path)
    summary = read_json(summary_path)
    return {
        "dir": directory,
        "label": label_of(directory, environment),
        "target": environment.get("target", environment.get("compiler", "unknown")),
        "route": environment.get("muladd_route", environment.get("muladd_fp64")),
        "compiler": environment.get("compiler"),
        "compiler_version": environment.get("compiler_version"),
        "boys_revision": environment.get("boys_revision"),
        "canary_verdict": environment.get("canary_verdict"),
        "arguments_sha256": file_digest(arguments_path),
        "fingerprint": fingerprint(summary),
    }


def same(value, runs):
    return all(run["fingerprint"][value] == runs[0]["fingerprint"][value]
               for run in runs[1:])


def compare(runs):
    fingerprints = [run["fingerprint"] for run in runs]

    grid_identical = len({run["arguments_sha256"] for run in runs}) == 1
    cells_identical = same("cells", runs) and same("expected_cells", runs)
    lanes_identical = same("precisions_present", runs)
    safety_clean = all(fp["violation_count"] == 0 for fp in fingerprints)
    verdict_identical = same("verdict_by_alpha", runs)
    pairs_identical = same("material_pairs", runs)
    sets_identical = same("material_sets", runs)

    if not safety_clean:
        stability = "bound-violated"
    elif not (grid_identical and cells_identical and lanes_identical):
        stability = "incomplete"
    elif not verdict_identical:
        stability = "verdict-unstable"
    elif not (pairs_identical and sets_identical):
        stability = "material-set-unstable"
    else:
        stability = "stable"

    # Descriptive, not a gate: how far the largest relative interaction in each
    # pair moves across the configurations. The paper quotes the range beside
    # the verdict, so a reader sees the size of the configuration dependence
    # even where the materiality class agrees.
    pair_names = sorted(set().union(*[fp["pair_max_relative"].keys()
                                      for fp in fingerprints]))
    magnitudes = {}
    for name in pair_names:
        values = [fp["pair_max_relative"].get(name) for fp in fingerprints]
        values = [v for v in values if v is not None]
        if not values:
            continue
        high = max(values)
        low = min(values)
        magnitudes[name] = {
            "min": low,
            "max": high,
            "spread_fraction": (high - low) / high if high > 0.0 else 0.0,
            "by_run": {
                run["label"]: run["fingerprint"]["pair_max_relative"].get(name)
                for run in runs
            },
        }

    share_spread = {}
    for key in sorted(set().union(*[fp["max_interaction_share"].keys()
                                    for fp in fingerprints])):
        values = [fp["max_interaction_share"].get(key) for fp in fingerprints]
        values = [v for v in values if v is not None]
        if values:
            share_spread[key] = {"min": min(values), "max": max(values)}

    detail = {}
    if not verdict_identical:
        detail["verdict_by_run"] = {
            run["label"]: run["fingerprint"]["verdict_by_alpha"] for run in runs
        }
    if not pairs_identical:
        detail["material_pairs_by_run"] = {
            run["label"]: run["fingerprint"]["material_pairs"] for run in runs
        }
    if not sets_identical:
        detail["material_sets_by_run"] = {
            run["label"]: run["fingerprint"]["material_sets"] for run in runs
        }
    if not safety_clean:
        detail["violations_by_run"] = {
            run["label"]: {
                "violation_count": run["fingerprint"]["violation_count"],
                "max_figure_ratio": run["fingerprint"]["max_figure_ratio"],
                "max_figure_ratio_at": run["fingerprint"]["max_figure_ratio_at"],
            }
            for run in runs
        }
    if not (grid_identical and cells_identical and lanes_identical):
        detail["shape_by_run"] = {
            run["label"]: {
                "arguments_sha256": run["arguments_sha256"],
                "cells": run["fingerprint"]["cells"],
                "expected_cells": run["fingerprint"]["expected_cells"],
                "precisions_present": run["fingerprint"]["precisions_present"],
            }
            for run in runs
        }

    notes = []
    if len({run["target"] for run in runs}) < 2:
        notes.append("only one target among the configurations; the two-target "
                     "requirement is not met")
    if len({run["route"] for run in runs}) < 2:
        notes.append("only one multiply-add route among the configurations; the "
                     "two-route requirement is not met")
    if any(run["canary_verdict"] not in (None, "quiet") for run in runs):
        notes.append("at least one configuration reports a canary verdict other "
                     "than quiet; its rows are provisional")
    if not grid_identical:
        notes.append("the configurations did not measure the same narrowed "
                     "argument grid; the comparison is not like for like")

    return {
        "schema": 1,
        "runs": [
            {
                "label": run["label"],
                "dir": run["dir"],
                "target": run["target"],
                "route": run["route"],
                "compiler": run["compiler"],
                "compiler_version": run["compiler_version"],
                "boys_revision": run["boys_revision"],
                "canary_verdict": run["canary_verdict"],
                "arguments_sha256": run["arguments_sha256"],
                "verdict": run["fingerprint"]["verdict"],
                "verdict_by_alpha": run["fingerprint"]["verdict_by_alpha"],
                "violation_count": run["fingerprint"]["violation_count"],
                "max_figure_ratio": run["fingerprint"]["max_figure_ratio"],
                "cells": run["fingerprint"]["cells"],
                "precisions_present": run["fingerprint"]["precisions_present"],
                "material_pairs": run["fingerprint"]["material_pairs"],
                "material_sets": run["fingerprint"]["material_sets"],
            }
            for run in runs
        ],
        "agreement": {
            "grid_identical": grid_identical,
            "cells_identical": cells_identical,
            "lanes_identical": lanes_identical,
            "safety_clean": safety_clean,
            "verdict_identical": verdict_identical,
            "material_pairs_identical": pairs_identical,
            "material_sets_identical": sets_identical,
        },
        "magnitudes": magnitudes,
        "max_interaction_share_by_configuration": share_spread,
        "detail": detail,
        "notes": notes,
        "stability": stability,
    }


def print_report(result):
    for run in result["runs"]:
        print(f"{run['label']}: {run['verdict']} "
              f"(violations {run['violation_count']}, cells {run['cells']}, "
              f"canary {run['canary_verdict']})")
    agreement = result["agreement"]
    print("agreement: "
          + ", ".join(f"{key}={'yes' if value else 'no'}"
                      for key, value in agreement.items()))
    for note in result["notes"]:
        print(f"note: {note}")
    print(f"stability: {result['stability']}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", action="append", required=True, metavar="DIR",
                        help="a results directory with summary.json, "
                             "environment.txt and arguments.csv; repeat per "
                             "configuration")
    parser.add_argument("--out", required=True, help="the stability JSON to write")
    args = parser.parse_args(argv)

    if len(args.run) < 2:
        fail("compare_stability: at least two --run directories are needed for "
             "a comparison")

    runs = [load_run(directory) for directory in args.run]
    for run in runs:
        missing = [key for key in ("verdict", "verdict_by_alpha")
                   if run["fingerprint"][key] is None]
        if missing:
            fail(f"compare_stability: '{run['dir']}' summary lacks "
                 f"{', '.join(missing)}; is it a current summariser output?")

    result = compare(runs)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print_report(result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
