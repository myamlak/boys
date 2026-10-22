#!/usr/bin/env python3
"""Synthetic pre-flight check for compare_stability.py.

This is not a library measurement. It builds run directories in a temporary
directory with the shape the stability comparator reads: `summary.json`,
`environment.txt` and `arguments.csv`, in four synthetic configurations. It
plants the four outcomes the comparator must distinguish — a stable pair whose
largest interaction moves in magnitude but not in kind, a verdict disagreement,
a disagreement in the material factor sets, a bound violation, and a run that
measured a different grid or a different number of cells. It also checks that a
one-run invocation is refused.

The comparator is not run against live results here. The script exists so the
stability pass after the window does not spend a four-run measurement on a
comparison that cannot parse or that calls an unstable result stable.

Usage:
    python selftest_stability.py

Exit 0 all checks passed, 1 a check failed, 2 the comparator was misused.
"""

import json
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "compare_stability.py")

ARGUMENTS = "lane,x_nominal,x,kind\nfp64,1,1,exact\nfp64,1.3,1.3,generic\n"


def set_entry(factors, points, share):
    return {
        "factors": factors,
        "max_set_share": share,
        "material_points": {"0.05": points, "0.1": points, "0.2": points},
    }


def lane_block(sets):
    return {
        "points": 66,
        "used_in_verdict": True,
        "sets": sets,
        "orders": {"1": {"ss_share": 0.1}, "2": {"ss_share": 0.3}},
    }


def make_summary(verdict, material_points, factors, relative, violation=0,
                 cells=576):
    material = {
        "0.05": material_points,
        "0.1": material_points,
        "0.2": material_points,
    }
    pairs = [
        {
            "pair": ["form", "exp"],
            "material": material,
            "max_relative": relative,
        },
        {
            "pair": ["route", "scheme"],
            "material": {"0.05": 0, "0.1": 0, "0.2": 0},
            "max_relative": 0.01,
        },
    ]
    sets = [set_entry(factors, material_points, 0.3)] if factors else []
    main = [set_entry(["route"], 2, 0.1)]
    return {
        "verdict": verdict,
        "verdict_by_alpha": {
            "0.05": verdict,
            "0.1": verdict,
            "0.2": verdict,
        },
        "counts": {
            "cells": cells,
            "expected_cells": 576,
            "precisions_present": ["fp32", "fp64"],
        },
        "safety": {
            "violation_count": violation,
            "max_figure_ratio": 1.2 if violation else 0.5,
            "max_figure_ratio_at": ["cell", 0, 1.0] if violation else ["cell", 0, 1.0],
        },
        "pairs": pairs,
        "multiway": {
            "within_lane": {
                "fp64": lane_block(sets + main),
                "fp32": lane_block(sets + main),
            },
            "cross_lane_exact": {
                "points": 33,
                "sets": sets,
            },
        },
    }


def write_run(root, name, target, route, summary, arguments=ARGUMENTS,
              label=None):
    directory = os.path.join(root, name)
    os.makedirs(directory)
    environment = "\n".join(
        [
            f"stability_configuration={label or (target + '-' + route)}",
            f"target={target}",
            f"muladd_route={route}",
            "muladd_fp64=" + route,
            "compiler=" + ("msvc" if target == "msvc" else "gnu-or-clang"),
            "compiler_version=synthetic",
            "boys_revision=482fd3388c276091834a6c12a642d9de2481c9f2",
            "canary_verdict=quiet",
        ]
    ) + "\n"
    with open(os.path.join(directory, "environment.txt"), "w", encoding="utf-8") as f:
        f.write(environment)
    with open(os.path.join(directory, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(os.path.join(directory, "arguments.csv"), "w", encoding="utf-8") as f:
        f.write(arguments)
    return directory


def run_comparator(runs, out):
    command = [sys.executable, SCRIPT]
    for directory in runs:
        command += ["--run", directory]
    command += ["--out", out]
    return subprocess.run(command, capture_output=True, text=True)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def compare_ok(root, name, runs, expected):
    out = os.path.join(root, f"{name}.json")
    proc = run_comparator(runs, out)
    if proc.returncode != 0:
        print(proc.stdout, end="")
        print(proc.stderr, end="", file=sys.stderr)
        raise AssertionError(f"{name}: comparator exited {proc.returncode}")
    with open(out, encoding="utf-8") as f:
        result = json.load(f)
    require(result["stability"] == expected,
            f"{name}: stability {result['stability']!r}, expected {expected!r}")
    return result


def main():
    material = "interaction-material"
    additive = "additive-within-10pct"

    with tempfile.TemporaryDirectory(prefix="composed-bounds-stability-") as tmp:
        a = write_run(tmp, "a", "msvc", "fused",
                      make_summary(material, 4, ["form", "exp"], 0.25))
        b = write_run(tmp, "b", "msvc", "separate",
                      make_summary(material, 4, ["form", "exp"], 0.30))
        c = write_run(tmp, "c", "gcc", "fused",
                      make_summary(additive, 0, [], 0.0))
        d = write_run(tmp, "d", "gcc", "separate",
                      make_summary(material, 4, ["scheme", "partition"], 0.25))
        e = write_run(tmp, "e", "msvc", "fused",
                      make_summary(material, 4, ["form", "exp"], 0.25,
                                   violation=1),
                      label="msvc-fused-violation")
        f = write_run(tmp, "f", "gcc", "fused",
                      make_summary(material, 4, ["form", "exp"], 0.25, cells=288))
        g = write_run(tmp, "g", "msvc", "fused",
                      make_summary(material, 4, ["form", "exp"], 0.25),
                      arguments="lane,x_nominal,x,kind\nfp64,2,2,exact\n")

        result = compare_ok(tmp, "stable", [a, b], "stable")
        agreement = result["agreement"]
        require(all(agreement.values()), f"stable pair agreement {agreement}")
        spread = result["magnitudes"]["form+exp"]["spread_fraction"]
        require(abs(spread - 0.05 / 0.30) < 1e-12,
                f"magnitude spread {spread}, expected 0.05/0.30")
        require(result["notes"], "one-target pair should carry a note")
        print(f"stable pair: verdict {result['stability']}, "
              f"spread {spread:.3f}")

        result = compare_ok(tmp, "verdict", [a, c], "verdict-unstable")
        require("verdict_by_run" in result["detail"],
                "verdict-unstable result lacks the per-run verdicts")
        print(f"verdict fixture: {result['stability']}")

        result = compare_ok(tmp, "sets", [a, d], "material-set-unstable")
        require("material_sets_by_run" in result["detail"],
                "set-unstable result lacks the per-run sets")
        print(f"material-set fixture: {result['stability']}")

        result = compare_ok(tmp, "violation", [a, e], "bound-violated")
        require(result["detail"]["violations_by_run"]
                ["msvc-fused-violation"]["violation_count"] == 1,
                "the planted violation is not the reported one")
        print(f"violation fixture: {result['stability']}")

        result = compare_ok(tmp, "cells", [a, f], "incomplete")
        require(not result["agreement"]["cells_identical"],
                "the 288-cell run was not detected as a different shape")
        print(f"cell-count fixture: {result['stability']}")

        result = compare_ok(tmp, "grid", [a, g], "incomplete")
        require(not result["agreement"]["grid_identical"],
                "the different argument grid was not detected")
        print(f"grid fixture: {result['stability']}")

        proc = run_comparator([a], os.path.join(tmp, "one.json"))
        require(proc.returncode == 2,
                f"one-run invocation exited {proc.returncode}, expected 2")
        print("one-run refusal: exit 2")

        missing = os.path.join(tmp, "missing")
        os.makedirs(missing)
        proc = run_comparator([a, missing], os.path.join(tmp, "missing.json"))
        require(proc.returncode == 2,
                f"missing-file invocation exited {proc.returncode}, expected 2")
        print("missing-file refusal: exit 2")

    print("selftest: ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"selftest FAILED: {error}", file=sys.stderr)
        sys.exit(1)
