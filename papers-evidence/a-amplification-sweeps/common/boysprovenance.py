#!/usr/bin/env python3
"""Revision, constants, toolchain and digests for one Paper A sweep run.

Everything a sweep pins is read from the boys tree it was compiled against,
never written down a second time here: the region-A edges, the dispatch
thresholds, the band degrees and the build's default policy all come out of
the headers.  The output is one JSON file, and every results file a sweep
writes sits beside it.

Usage:
    python boysprovenance.py --harness sweep1 --harness-dir . --out provenance.json
"""

import argparse
import datetime
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_TREE = r"C:\Users\myaml\dev\bladawiec\external\boys"

HEADER_FILES = [
    "include/boys/accuracy.hpp",
    "include/boys/boys_build_defaults.hpp",
    "include/boys/boys_coefficients.hpp",
    "include/boys/boys_decl.hpp",
    "include/boys/boys.hpp",
    "include/boys/boys_fit_chebyshev.hpp",
    "include/boys/boys_fit_core.hpp",
    "include/boys/boys_impl.hpp",
    "include/boys/boys_transform.hpp",
    "include/boys/kernels.hpp",
    "include/boys/policy.hpp",
    "include/boys/version.hpp",
]

REFERENCE_FILES = [
    "tools/gen_boys_accuracy_gate_reference.py",
    "tests/data/boys_accuracy_gate_reference.csv",
    "tests/data/boys_reference.csv",
]


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        return handle.read()


def scalar(text, name, where):
    match = re.search(
        r"inline\s+constexpr\s+(?:double|int)\s+" + re.escape(name) + r"\s*=\s*"
        r"([-+]?[0-9][0-9eE+.\-]*)\s*;",
        text,
    )
    if not match:
        raise SystemExit("cannot read %s from %s" % (name, where))
    return float(match.group(1))


def parse_constants(tree):
    coefficients_path = tree / "include/boys/boys_coefficients.hpp"
    transform_path = tree / "include/boys/boys_transform.hpp"
    accuracy_path = tree / "include/boys/accuracy.hpp"
    coefficients = read(coefficients_path)
    transform = read(transform_path)
    accuracy = read(accuracy_path)

    thresholds_match = re.search(
        r"kTierThresholds\s*=\s*std::to_array<double>\(\{(.*?)\}\);",
        coefficients,
        re.S,
    )
    if not thresholds_match:
        raise SystemExit("cannot read kTierThresholds from %s" % coefficients_path)
    thresholds = [
        float(value)
        for value in re.findall(
            r"[-+]?[0-9]+\.[0-9]+(?:[eE][-+]?[0-9]+)?", thresholds_match.group(1)
        )
    ]

    pieces_match = re.search(
        r"inline\s+constexpr\s+auto\s+kPieces\s*=\s*std::to_array<OrderPiece>\(\{(.*?)\}\);",
        coefficients,
        re.S,
    )
    if not pieces_match:
        raise SystemExit("cannot read kPieces from %s" % coefficients_path)
    degrees = sorted(
        {
            int(degree)
            for degree in re.findall(
                r"\{\s*[-+0-9.eE]+\s*,\s*[-+0-9.eE]+\s*,\s*([0-9]+)\s*,\s*[0-9]+\s*\}",
                pieces_match.group(1),
            )
        }
    )
    pieces = [
        {"from": float(left), "to": float(right), "degree": int(degree)}
        for left, right, degree in re.findall(
            r"\{\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([0-9]+)\s*,\s*[0-9]+\s*\}",
            pieces_match.group(1),
        )
    ]
    # The first block is F_0's: one piece per region-A band, in order.
    region_a_pieces = pieces[:2]

    extended_degree_match = re.search(
        r"inline\s+constexpr\s+int\s+kExtendedBDeg\s*=\s*([0-9]+)\s*;", coefficients
    )
    max_order_match = re.search(
        r"inline\s+constexpr\s+int\s+kMaxBoysOrder\s*=\s*([0-9]+)\s*;", accuracy
    )

    return {
        "kExtendedBX0": scalar(coefficients, "kExtendedBX0", coefficients_path),
        "kExtendedBDeg": int(extended_degree_match.group(1))
        if extended_degree_match
        else None,
        "kTierThresholds": thresholds,
        "kRegionA1Edge": scalar(transform, "kRegionA1Edge", transform_path),
        "kRegionAEnd": scalar(transform, "kRegionAEnd", transform_path),
        "kMaxBoysOrder": int(max_order_match.group(1)) if max_order_match else None,
        "regionADegrees": degrees,
        "regionAPieces": region_a_pieces,
    }


def parse_build_defaults(tree):
    """The committed defaults: the seven seam names and every class row.

    The seam names are the fallback a class with no row resolves to; they are
    not what an unnamed entry compiles.  An entry's template default is its
    class's row in BOYS_BUILD_DEFAULT_ROWS, and the host fp64 all-orders row
    need not hold the seam's cells.  Both are returned, and the row the
    harness's recursed entry would compile without its explicit policy is
    named separately.
    """
    path = tree / "include/boys/boys_build_defaults.hpp"
    text = read(path)
    wanted = [
        "BOYS_BUILD_DEFAULT_FIT_ROUTE",
        "BOYS_BUILD_DEFAULT_EVAL_SCHEME",
        "BOYS_BUILD_DEFAULT_PACK_AXIS",
        "BOYS_BUILD_DEFAULT_DIVISION_FORM",
        "BOYS_BUILD_DEFAULT_FIT_GRANULARITY",
        "BOYS_BUILD_DEFAULT_DEVICE_DIVISION_FORM",
        "BOYS_BUILD_DEFAULT_DEVICE_REGION_B_EXP",
    ]
    seam = {}
    for name in wanted:
        match = re.search(r"^#define\s+" + name + r"\s+(.+?)\s*$", text, re.M)
        seam[name] = match.group(1) if match else None

    # Strip comments first: a row is separated from the next by prose, and the
    # doc-comment example is not a row.
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    stripped = re.sub(r"^\s*//.*$", "", stripped, flags=re.M)
    rows = []
    for match in re.finditer(r"\bX\(([^()]*)\)", stripped):
        cells = [
            cell.replace("\\", "").strip() for cell in match.group(1).split(",")
        ]
        if len(cells) != 12:
            continue
        row = {
            "device": cells[0],
            "precision": cells[1],
            "shape": cells[2],
            "fit_route": cells[3],
            "eval_scheme": cells[4],
            "budget": cells[5],
            "pack_axis": cells[6],
            "granularity": cells[7],
            "division_form": cells[8],
            "region_b_exp": cells[9],
            "basis": cells[10],
            "record": cells[11].strip('"'),
        }
        rows.append(row)

    default_row = next(
        (
            row
            for row in rows
            if row["device"] == "kHost"
            and row["precision"] == "kFp64"
            and row["shape"] == "kAllOrders"
        ),
        None,
    )
    return {"seam": seam, "host_fp64_all_orders": default_row, "rows": rows}


def parse_pinned_policy():
    """The policy the harness names, read from common/replica.hpp.

    One source for the paper's policy: the header the sweeps compile.  The
    names are returned in StatedEvalPolicy's order.
    """
    path = Path(__file__).resolve().parents[1] / "common/replica.hpp"
    text = read(path)
    match = re.search(r"StatedEvalPolicy\s*<([^;]*?)>", text, re.S)
    if not match:
        raise SystemExit("cannot read the pinned policy from %s" % path)
    cells = [cell.strip() for cell in match.group(1).split(",") if cell.strip()]
    names = [
        "fit_route",
        "eval_scheme",
        "budget",
        "pack_axis",
        "granularity",
        "division_form",
        "region_b_exp",
    ]
    if len(cells) != len(names):
        raise SystemExit("the pinned policy has %d cells, expected %d" % (len(cells), len(names)))
    return dict(zip(names, cells))


def run_git(tree, *args):
    try:
        completed = subprocess.run(
            ["git", "-C", str(tree)] + list(args),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return {"error": str(error)}
    return {
        "argv": ["git", "-C", str(tree)] + list(args),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def revision(tree):
    exact = run_git(tree, "describe", "--tags", "--exact-match", "HEAD")
    describe = run_git(tree, "describe", "--tags", "--always", "--dirty")
    head = run_git(tree, "rev-parse", "HEAD")
    status = run_git(tree, "status", "--porcelain")
    return {
        "tree": str(tree),
        "tag_at_head": exact["stdout"] if exact.get("returncode") == 0 else None,
        "describe": describe.get("stdout", ""),
        "commit": head.get("stdout", ""),
        "dirty": bool(status.get("stdout", "")),
        "status_porcelain": status.get("stdout", "").splitlines(),
    }


def compiler_banner():
    path = shutil.which("cl") or shutil.which("cl.exe")
    banner = None
    if path:
        completed = subprocess.run([path], capture_output=True, text=True, check=False)
        text = (completed.stdout + "\n" + completed.stderr).strip()
        banner = text.splitlines()[0] if text else None
    return {
        "path": path,
        "banner": banner,
        "environment": {
            "VSCMD_VER": os.environ.get("VSCMD_VER"),
            "VisualStudioVersion": os.environ.get("VisualStudioVersion"),
        },
    }


def machine():
    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "platform": platform.platform(),
        "logical_processors": os.cpu_count(),
        "python": sys.version.split()[0],
    }


def digest(path):
    hasher = hashlib.sha256()
    size = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            size += len(block)
            hasher.update(block)
    return {"sha256": hasher.hexdigest(), "bytes": size}


def digests(tree, harness_dir):
    result = {}
    for relative in HEADER_FILES + REFERENCE_FILES:
        path = tree / relative
        if path.is_file():
            result["boys/" + relative] = digest(path)
    if harness_dir:
        root = Path(harness_dir).resolve()
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".cpp", ".hpp", ".ps1", ".md"}:
                result["harness/" + path.relative_to(root).as_posix()] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boys-tree", default=os.environ.get("BOYS_TREE", DEFAULT_TREE))
    parser.add_argument("--harness", required=True)
    parser.add_argument("--harness-dir", default=".")
    parser.add_argument("--flags", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    tree = Path(args.boys_tree).resolve()
    if not (tree / "include/boys/version.hpp").is_file():
        raise SystemExit("not a boys tree: %s" % tree)

    payload = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "harness": args.harness,
        "flags": args.flags,
        "boys": revision(tree),
        "constants": parse_constants(tree),
        "build_defaults": parse_build_defaults(tree),
        "pinned_policy": parse_pinned_policy(),
        "compiler": compiler_banner(),
        "machine": machine(),
        "digests": digests(tree, args.harness_dir),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
