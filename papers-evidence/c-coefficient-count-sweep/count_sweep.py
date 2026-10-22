#!/usr/bin/env python3
"""Read the v5.0.0 coefficient counts, then measure them against width.

The modes are deliberately separate:

  --static     parse the committed header and report what it stores. It reads
               the generated constants and array lengths, checks the emitted
               counts against the array definitions, and writes a JSON record
               with a line number for every figure. No fit is run.

  --sweep      import the library's own generator and measure the smallest
               stored count each family needs on [X0, X0 + W) at one target,
               over a width sweep. This mode is not run until the build window
               is free; --probe measures one width first.

  --paper-cell run the same measurement at the nearest-neighbour's exact
               interval and target, as a cross-check of its published count.

  --plan       print the width grid, targets, counting conventions, the exact
               commands and the only runtime datum that exists, without
               running a fit. Run this before a window run.

A sweep invocation is resumable. It rewrites sweep_all.json and the per-target
files after every completed row; --max-seconds stops it before starting a new
row once the budget is spent. Re-running the same command reuses the rows on
disk and continues. A row already in progress is never interrupted, so the
budget bounds when the next row may start, not the wall time of an invocation.

The fitting and evaluation code is the library's:
  cheb_coeffs, clenshaw_split_double, rat_remez, rat_lawson, rat_eval_double,
  boys_ref, from tools/gen_boys_coefficients.py.

The wrapper here supplies the width, the target, the grid and the scan. It does
not reimplement any fit or evaluation that places a count. The single exception
is the paper cell's cross-check, added for Q-258: a rational interpolant at
Chebyshev-Gauss nodes, written here and sharing no code with either of the
library's solvers. It asks only whether a stored count can reach the target;
its answers appear in paper_cell.json and in no sweep row.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BOYS = Path(r"C:\Users\myaml\dev\bladawiec\external\boys")
HEADER_REL = Path("include/boys/boys_coefficients.hpp")
GENERATOR_REL = Path("tools/gen_boys_coefficients.py")
LANE_CONTRACT_REL = Path("docs/lane-contract.md")

# The generator's worker pool cannot pickle an imported module, so a job names
# the generator by path and each worker loads it once. The parent sets this to
# the root it actually imported, not to the default.
BOYS_ROOT_STR = str(DEFAULT_BOYS)
_WORKER_GEN = None

N_NODES = 2000          # certifying grid: N_NODES + 1 nodes, uniform in t
COARSE_STRIDE = 5       # the generator's own scan stride (RAT_STRIDE)
CHEB_MIN_DEG = 4        # lowest degree the split-Clenshaw body asserts
CHEB_MAX_DEG = 64
RAT_MIN_STORED = 3      # [1/1]
RAT_MAX_STORED = 40
RAT_SCAN_OUTER = 100    # survey setting for Lawson; refinement uses 300
RAT_REFINE_OUTER = 300
RAT_REMEZ_ITERS = 60    # the generator's Remez exchange setting

# The paper cell's independent witness. It is the one solver here that is not
# the library's, it feeds no sweep count, and it is never used to place a
# table: it answers only whether a given stored count can reach the target.
RAT_INTERP_DPS = 60
RAT_INTERP_RESID = 1e-40     # a solve below this relative residual is refused
RAT_INTERP_DEN_FLOOR = 1e-3  # the generator's pole guard, applied to the grid

# The result files carry this schema number; a resume that finds another one
# starts fresh rather than mixing rows from two definitions of a row.
ROW_SCHEMA = 2

# The settings a count depends on. A row written under other values is not the
# row this invocation would produce, so a resume re-runs it. Worker count is
# provenance, not identity: the parallel and serial paths are measured equal.
SETTING_KEYS = ("nodes", "coarse_stride", "scan_outer", "refine_outer",
                "remez_iters")

# The only runtime datum that exists. The 2026-10-19 probe ran one width
# (ship, 19.100151847891516) at rel:1e-13 on a loaded machine, printed its
# progress, and was stopped at 600 s without completing. That is a lower bound
# for that one row and nothing else: no per-width cost and no total wall time
# are claimed until the quiet probe measures one width. The quiet windows own
# that measurement; this record is what the harness prints in --plan.
PROBE_LOADED_2026_10_19 = {
    "date": "2026-10-19",
    "machine": "loaded; not a run-alone machine",
    "width_name": "ship",
    "width": 19.100151847891516,
    "target": "rel:1e-13",
    "completed": False,
    "stopped_at_seconds": 600,
    "reading": (
        "the widest width at one target exceeded 600 s on a loaded machine; "
        "this is a lower bound for that row, not an estimate of the sweep"
    ),
    "implication": (
        "treat the 15-width x 2-target grid as a multi-hour job until the "
        "quiet probe replaces this bound; every invocation is capped by "
        "--max-seconds and resumes from its written rows"
    ),
}

NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?[fF]?"
)

F32_NAMESPACE_MARKER = "namespace boys::detail::f32 {"

# The paper's counted object. Stated once here and written into every result
# file, so no reader has to infer it from a column heading.
PRECISION_SCOPE = {
    "reported_precision": "fp64",
    "stored_type": "binary64",
    "evaluation": "binary64, worse of the fused and separate multiply-add routes",
    "count_unit": "coefficients of one representation of the fit",
    "why_fp64": (
        "the count trade is resolved at the double lane's 1e-13 F_0 contract; "
        "the nearest-neighbour comparison is 5e-14 absolute in binary64; and a "
        "binary64 coefficient is 8 bytes while a binary32 one is 4, so counts "
        "from the two lanes are not like-for-like storage"
    ),
    "fp32_in_this_paper": (
        "static counts only, a secondary column; not part of the width law"
    ),
    "sweep_scope": (
        "fp64 only: binary64 stored coefficients, binary64 evaluation, targets "
        "rel:1e-13 and abs:5e-14"
    ),
    "fp32_comparison_would_need": (
        "the float lane's own path: fits stored and evaluated in binary32, both "
        "multiply-add routes, certified at max(2^-22 F_n(x), 1.5e-7) from the "
        "generator's f32 fit and acceptance functions, on the float lane's own "
        "edges and partitions"
    ),
}


# ---------------------------------------------------------------------------
# small utilities


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_of(text: str, pos: int) -> int:
    return text[:pos].count("\n") + 1


def git_rev(root: Path) -> dict:
    out = {}
    for key, args in (("commit", ["rev-parse", "HEAD"]),
                      ("tag", ["describe", "--tags", "--exact-match"])):
        try:
            out[key] = subprocess.check_output(
                ["git", "-C", str(root), *args], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            out[key] = None
    return out


def parse_scalar(text: str, name: str):
    m = re.search(
        rf"inline constexpr (?:double|int|float)\s+{re.escape(name)}\s*=\s*([^;]+);",
        text,
    )
    if not m:
        raise KeyError(name)
    raw = m.group(1).strip()
    line = line_of(text, m.start())
    if re.fullmatch(r"[-+]?\d+", raw):
        value = int(raw)
    else:
        value = float(raw.rstrip("fF"))
    return value, line, raw


def parse_array(text: str, name: str):
    m = re.search(
        rf"inline constexpr auto\s+{re.escape(name)}\s*=\s*std::to_array<[^>]*>\(\s*\{{",
        text,
    )
    if not m:
        raise KeyError(name)
    end = text.find("});", m.end())
    if end < 0:
        raise KeyError(name)
    body = re.sub(r"//[^\n]*", "", text[m.end():end])
    tokens = NUM_RE.findall(body)
    values = []
    for tok in tokens:
        tok = tok.rstrip("fF")
        values.append(int(tok) if re.fullmatch(r"[-+]?\d+", tok) else float(tok))
    return values, line_of(text, m.start()), body


def parse_rows(text: str, name: str):
    values, line, body = parse_array(text, name)
    rows = []
    for row in re.findall(r"\{([^{}]+)\}", body):
        cells = [c.strip() for c in row.split(",") if c.strip()]
        parsed = []
        for cell in cells:
            cell = cell.rstrip("fF")
            parsed.append(int(cell) if re.fullmatch(r"[-+]?\d+", cell) else float(cell))
        rows.append(parsed)
    return rows, line, values


def arr(values) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


# ---------------------------------------------------------------------------
# static read


def static_mode(args) -> dict:
    boys = Path(args.boys_root)
    header_path = boys / HEADER_REL
    gen_path = boys / GENERATOR_REL
    text = read_text(header_path)

    scalar_names = [
        "kMaxOrder", "kX0", "kX1", "kFlatHi", "kFlatWidth", "kFlatIntervals",
        "kBDeg", "kRatBnumDeg", "kRatBdenDeg",
        "kRegionBFitBar", "kRegionBFitChebStored", "kRegionBFitChebDelivered",
        "kRegionBFitRatStored", "kRegionBFitRatDelivered",
        "kRegionAFitBar", "kRegionAFitChebStored", "kRegionAFitChebDelivered",
        "kRegionAFitRatStored", "kRegionAFitRatDelivered",
        "kNarrowBDeg", "kNarrowBPieces", "kNarrowRatBStored",
        "kNarrowRatBBound", "kNarrowRatBDeliveredFused", "kNarrowRatBDeliveredSeparate",
        "kNarrowRatAStored", "kNarrowRatABound",
        "kFlatRatIntervals", "kFlatRatStoredTotal", "kFlatRatBar",
    ]
    constants = {}
    for name in scalar_names:
        value, line, raw = parse_scalar(text, name)
        constants[name] = {"value": value, "line": line, "raw": raw}

    arrays = {}
    for name in [
        "kBcoeffs", "kMonoBcoeffs", "kRatBnum", "kRatBden",
        "kNarrowBcoeffs", "kNarrowBMonoCoeffs", "kNarrowBEdges",
        "kNarrowRatBOffset", "kNarrowRatBNumDeg", "kNarrowRatBDenDeg",
        "kFlatDegs", "kFlatOffsets", "kFlatRatNumDeg", "kFlatRatDenDeg",
        "kFlatRatStored", "kFlatRatOffsets",
    ]:
        values, line, body = parse_array(text, name)
        arrays[name] = {"values": values, "line": line, "length": len(values)}

    narrow_rows, narrow_rows_line, _ = parse_rows(text, "kNarrowRows")
    flat_rows, flat_rows_line, _ = parse_rows(text, "kFlatRows")
    flat_rat_rows, flat_rat_rows_line, _ = parse_rows(text, "kFlatRatRows")
    narrow_a_rows, narrow_a_rows_line, _ = parse_rows(text, "kNarrowARows")

    kx0 = constants["kX0"]["value"]
    kx1 = constants["kX1"]["value"]
    flat_hi = constants["kFlatHi"]["value"]
    max_order = constants["kMaxOrder"]["value"]
    orders = max_order + 1

    narrow_edges = arrays["kNarrowBEdges"]["values"]
    narrow_widths = [narrow_edges[i + 1] - narrow_edges[i]
                     for i in range(len(narrow_edges) - 1)]

    # Stored counts derived from the arrays themselves, not from the summary
    # constants. Each check is (name, value, expected, ok, detail).
    checks = []

    def check(name, value, expected, detail=""):
        checks.append({"name": name, "value": value, "expected": expected,
                       "ok": value == expected, "detail": detail})

    check("coarse region B chebyshev: len(kBcoeffs)",
          arrays["kBcoeffs"]["length"], constants["kBDeg"]["value"] + 1)
    check("coarse region B chebyshev: emitted stored",
          constants["kRegionBFitChebStored"]["value"], arrays["kBcoeffs"]["length"])
    check("coarse region B chebyshev: monomial mirror length",
          arrays["kMonoBcoeffs"]["length"], arrays["kBcoeffs"]["length"],
          "the Horner scheme reads a monomial copy of the same coefficients")
    check("coarse region B rational: len(kRatBnum)",
          arrays["kRatBnum"]["length"], constants["kRatBnumDeg"]["value"] + 1)
    check("coarse region B rational: len(kRatBden)",
          arrays["kRatBden"]["length"], constants["kRatBdenDeg"]["value"])
    rat_stored = arrays["kRatBnum"]["length"] + arrays["kRatBden"]["length"]
    check("coarse region B rational: stored sum",
          rat_stored, constants["kRegionBFitRatStored"]["value"],
          "numerator + q_1..q_k, q_0 held at 1")

    check("narrow region B chebyshev: len(kNarrowBcoeffs)",
          arrays["kNarrowBcoeffs"]["length"],
          constants["kNarrowBPieces"]["value"] * (constants["kNarrowBDeg"]["value"] + 1))
    check("narrow region B chebyshev: row stored",
          narrow_rows[0][2], arrays["kNarrowBcoeffs"]["length"])
    check("narrow region B chebyshev: monomial mirror length",
          arrays["kNarrowBMonoCoeffs"]["length"],
          arrays["kNarrowBcoeffs"]["length"])
    narrow_rat_total = sum(
        (m + 1) + k
        for m, k in zip(arrays["kNarrowRatBNumDeg"]["values"],
                        arrays["kNarrowRatBDenDeg"]["values"])
    )
    check("narrow region B rational: sum(m+1+k)",
          narrow_rat_total, constants["kNarrowRatBStored"]["value"])

    flat_cheb_total = orders * sum(d + 1 for d in arrays["kFlatDegs"]["values"])
    check("uniform chebyshev: orders * sum(deg+1)",
          flat_cheb_total, flat_rows[0][3])
    check("uniform chebyshev: len(kFlatDegs)",
          arrays["kFlatDegs"]["length"], constants["kFlatIntervals"]["value"])
    flat_rat_total = orders * sum(arrays["kFlatRatStored"]["values"])
    check("uniform rational: orders * sum(per-cell stored)",
          flat_rat_total, constants["kFlatRatStoredTotal"]["value"])
    per_cell_ok = all(
        stored == (m + 1) + k
        for stored, m, k in zip(arrays["kFlatRatStored"]["values"],
                                arrays["kFlatRatNumDeg"]["values"],
                                arrays["kFlatRatDenDeg"]["values"])
    )
    check("uniform rational: every cell stored == m+1+k",
          per_cell_ok, True)
    check("uniform rational: offsets end at total",
          arrays["kFlatRatOffsets"]["values"][-1],
          constants["kFlatRatStoredTotal"]["value"])
    check("uniform chebyshev: offsets end at total",
          arrays["kFlatOffsets"]["values"][-1], flat_rows[0][3])
    check("narrow region A chebyshev row",
          narrow_a_rows[0][3], 3421)
    check("narrow region A rational stored",
          constants["kNarrowRatAStored"]["value"], 2611)

    widths = {
        "coarse_region_b": flat_hi - kx0,
        "paper_region_b": kx1 - kx0,
        "narrow_region_b_pieces": narrow_widths,
        "uniform_cell": constants["kFlatWidth"]["value"],
        "region_b_left": kx0,
        "region_b_top": flat_hi,
    }

    # The bar's provenance, from the generator itself, is a light check.
    bar_check = {"ran": False}
    try:
        g = load_generator(boys)
        with g.mp.workdps(g.RAT_DPS):
            f0_x0 = g.boys_ref(0, g.X0, g.RAT_TAIL_FLOOR, g.RAT_TERMS)
            bar = g.RAT_B_REL * f0_x0
        bar_check = {
            "ran": True,
            "f0_kx0": g.mp.nstr(f0_x0, 20),
            "bar_1e13_f0": g.mp.nstr(bar, 20),
            "header_bar": constants["kRegionBFitBar"]["raw"],
            "rel_diff": float(abs(bar - g.mpf(constants["kRegionBFitBar"]["raw"])) / bar),
            "ok": float(abs(bar - g.mpf(constants["kRegionBFitBar"]["raw"])) / bar) < 1e-12,
        }
    except Exception as exc:  # the static read does not depend on the generator
        bar_check = {"ran": False, "error": repr(exc)}

    # The lane-contract table is known to lag the generated header. Read it and
    # record the discrepancy rather than silently using either figure.
    stale = {"checked": False}
    lane_path = boys / LANE_CONTRACT_REL
    if lane_path.exists():
        lane = read_text(lane_path)
        rows = {}
        for label, key in (("chebyshev (default)", "chebyshev"),
                           ("rational minimax", "rational")):
            m = re.search(
                rf"^\|\s*{re.escape(label)}\s*\|\s*B\s*\|([^\n]*)$", lane, re.M
            )
            if m:
                cells = [c.strip() for c in m.group(1).split("|")]
                rows[key] = {"line": line_of(lane, m.start()), "cells": cells}
        stale = {
            "checked": True,
            "doc": str(lane_path),
            "rows": rows,
            "header_counts": {
                "chebyshev": constants["kRegionBFitChebStored"]["value"],
                "rational": constants["kRegionBFitRatStored"]["value"],
            },
            "header_bar": constants["kRegionBFitBar"]["value"],
            "stale_if_table_says": {"chebyshev": "19", "rational": "12", "bar": "5e-14"},
        }

    doc_counts = {
        "region_b_coarse": {
            "chebyshev": {
                "stored": constants["kRegionBFitChebStored"]["value"],
                "degree": constants["kBDeg"]["value"],
                "stored_rule": "degree + 1",
                "line": constants["kRegionBFitChebStored"]["line"],
                "array_line": arrays["kBcoeffs"]["line"],
                "monomial_mirror": arrays["kMonoBcoeffs"]["length"],
                "delivered": constants["kRegionBFitChebDelivered"]["value"],
            },
            "rational": {
                "stored": constants["kRegionBFitRatStored"]["value"],
                "numerator_degree": constants["kRatBnumDeg"]["value"],
                "denominator_degree": constants["kRatBdenDeg"]["value"],
                "stored_rule": "(m+1) + k, q_0 held at 1",
                "line": constants["kRegionBFitRatStored"]["line"],
                "array_lines": {
                    "num": arrays["kRatBnum"]["line"],
                    "den": arrays["kRatBden"]["line"],
                },
                "delivered": constants["kRegionBFitRatDelivered"]["value"],
            },
        },
        "region_b_narrow": {
            "chebyshev": {
                "stored": narrow_rows[0][2],
                "pieces": constants["kNarrowBPieces"]["value"],
                "degree": constants["kNarrowBDeg"]["value"],
                "line": narrow_rows_line,
                "array_line": arrays["kNarrowBcoeffs"]["line"],
                "monomial_mirror": arrays["kNarrowBMonoCoeffs"]["length"],
            },
            "rational": {
                "stored": constants["kNarrowRatBStored"]["value"],
                "pairs": list(zip(arrays["kNarrowRatBNumDeg"]["values"],
                                  arrays["kNarrowRatBDenDeg"]["values"])),
                "line": constants["kNarrowRatBStored"]["line"],
                "offsets_line": arrays["kNarrowRatBOffset"]["line"],
            },
        },
        "uniform": {
            "chebyshev": {
                "stored": flat_rows[0][3],
                "intervals": constants["kFlatIntervals"]["value"],
                "orders": orders,
                "per_cell": arrays["kFlatDegs"]["values"][0] + 1,
                "line": flat_rows_line,
                "monomial_mirror_extent": 98208,
                "monomial_note": (
                    "boys_coefficients.hpp:817 asserts kFlatMonoCoeffs is the "
                    "same 98208; the monomial copy is an evaluation-scheme "
                    "mirror, not a second fit"
                ),
            },
            "rational": {
                "stored": constants["kFlatRatStoredTotal"]["value"],
                "intervals": constants["kFlatRatIntervals"]["value"],
                "orders": orders,
                "per_cell_range": [min(arrays["kFlatRatStored"]["values"]),
                                   max(arrays["kFlatRatStored"]["values"])],
                "line": constants["kFlatRatStoredTotal"]["line"],
            },
        },
    }

    # ---- the float lane: the same header below `namespace boys::detail::f32`.
    # Its names shadow the double lane's (kBDeg, kBcoeffs, kRegionBFitBar, ...),
    # so every lookup here is scoped to the f32 slice; the line number is the
    # slice's own plus the offset of the namespace in the full header.
    f32_pos = text.index(F32_NAMESPACE_MARKER)
    f32_text = text[f32_pos:]
    f32_line0 = line_of(text, f32_pos)

    def f32_line(line):
        return line + f32_line0 - 1

    fp32_scalar_names = [
        "kFlatIntervalsF32", "kFlatWidthF32", "kFlatHiF32", "kFlatReadCapF32",
        "kBDeg", "kRatBnumDeg", "kRatBdenDeg",
        "kRegionBFitBar", "kRegionBFitChebStored", "kRegionBFitChebDelivered",
        "kRegionBFitChebDeliveredHorner", "kRegionBFitRatStored",
        "kRegionBFitRatDelivered",
        "kRegionAFitBar", "kRegionAFitChebStored", "kRegionAFitChebDelivered",
        "kRegionAFitChebDeliveredHorner", "kRegionAFitRatStored",
        "kRegionAFitRatDelivered",
        "kNarrowADegF32", "kNarrowBDegF32", "kNarrowBPiecesF32",
        "kNarrowRatAStoredF32", "kNarrowRatBPiecesCountF32",
        "kNarrowRatBStoredF32", "kNarrowRatBDeliveredFusedF32",
        "kNarrowRatBDeliveredSeparateF32",
        "kFlatRatIntervalsF32", "kFlatRatStoredTotalF32", "kFlatRatBarF32",
    ]
    fp32_constants = {}
    for name in fp32_scalar_names:
        value, line, raw = parse_scalar(f32_text, name)
        fp32_constants[name] = {"value": value, "line": f32_line(line), "raw": raw}

    fp32_arrays = {}
    for name in [
        "kBcoeffs", "kMonoBcoeffs", "kRatBnum", "kRatBden",
        "kNarrowBcoeffsF32", "kNarrowBMonoCoeffsF32", "kNarrowBEdgesF32",
        "kFlatDegsF32", "kFlatOffsetsF32",
        "kFlatRatNumDegF32", "kFlatRatDenDegF32",
        "kFlatRatStoredF32", "kFlatRatOffsetsF32",
    ]:
        values, line, body = parse_array(f32_text, name)
        fp32_arrays[name] = {"values": values, "line": f32_line(line),
                             "length": len(values)}

    fp32_rat_pieces, fp32_rat_pieces_line, _ = parse_rows(
        f32_text, "kNarrowRatBPiecesF32")
    fp32_rat_pieces_line = f32_line(fp32_rat_pieces_line)
    fp32_narrow_rows, fp32_narrow_rows_line, _ = parse_rows(
        f32_text, "kNarrowBRowsF32")
    fp32_narrow_rows_line = f32_line(fp32_narrow_rows_line)
    fp32_narrow_a_rows, fp32_narrow_a_rows_line, _ = parse_rows(
        f32_text, "kNarrowARowsF32")
    fp32_narrow_a_rows_line = f32_line(fp32_narrow_a_rows_line)
    fp32_flat_rows, fp32_flat_rows_line, _ = parse_rows(f32_text, "kFlatRowsF32")
    fp32_flat_rows_line = f32_line(fp32_flat_rows_line)
    fp32_flat_rat_rows, fp32_flat_rat_rows_line, _ = parse_rows(
        f32_text, "kFlatRatRowsF32")
    fp32_flat_rat_rows_line = f32_line(fp32_flat_rat_rows_line)

    # The extern declarations carry the stored extent of every array defined in
    # one of the float lane's own headers. They are a second witness to the
    # counts the emitted constants and the row tables give.
    fp32_extents = {}
    for m in re.finditer(
        r"extern const std::array<float,\s*(\d+)>\s*([A-Za-z_][A-Za-z0-9_]*)\s*;",
        f32_text,
    ):
        fp32_extents[m.group(2)] = {
            "extent": int(m.group(1)),
            "line": line_of(text, f32_pos + m.start()),
        }

    fp32_narrow_edges = fp32_arrays["kNarrowBEdgesF32"]["values"]
    fp32_narrow_widths = [fp32_narrow_edges[i + 1] - fp32_narrow_edges[i]
                          for i in range(len(fp32_narrow_edges) - 1)]
    fp32_rat_total = sum((m + 1) + k for _a, _b, m, k, _off in fp32_rat_pieces)
    fp32_flat_cheb_total = orders * sum(
        d + 1 for d in fp32_arrays["kFlatDegsF32"]["values"])
    fp32_flat_rat_total = orders * sum(
        fp32_arrays["kFlatRatStoredF32"]["values"])

    def check32(name, value, expected, detail=""):
        checks.append({"name": "fp32 " + name, "value": value,
                       "expected": expected, "ok": value == expected,
                       "detail": detail})

    check32("coarse region B chebyshev: len(kBcoeffs)",
            fp32_arrays["kBcoeffs"]["length"],
            fp32_constants["kBDeg"]["value"] + 1)
    check32("coarse region B chebyshev: emitted stored",
            fp32_constants["kRegionBFitChebStored"]["value"],
            fp32_arrays["kBcoeffs"]["length"])
    check32("coarse region B chebyshev: monomial mirror length",
            fp32_arrays["kMonoBcoeffs"]["length"],
            fp32_arrays["kBcoeffs"]["length"])
    check32("coarse region B rational: len(kRatBnum)",
            fp32_arrays["kRatBnum"]["length"],
            fp32_constants["kRatBnumDeg"]["value"] + 1)
    check32("coarse region B rational: len(kRatBden)",
            fp32_arrays["kRatBden"]["length"],
            fp32_constants["kRatBdenDeg"]["value"])
    check32("coarse region B rational: stored sum",
            fp32_arrays["kRatBnum"]["length"] + fp32_arrays["kRatBden"]["length"],
            fp32_constants["kRegionBFitRatStored"]["value"],
            "numerator + q_1..q_k, q_0 held at 1")
    check32("narrow region B chebyshev: len(kNarrowBcoeffsF32)",
            fp32_arrays["kNarrowBcoeffsF32"]["length"],
            fp32_constants["kNarrowBPiecesF32"]["value"]
            * (fp32_constants["kNarrowBDegF32"]["value"] + 1))
    check32("narrow region B chebyshev: row stored",
            fp32_narrow_rows[0][3],
            fp32_arrays["kNarrowBcoeffsF32"]["length"])
    check32("narrow region B chebyshev: monomial mirror length",
            fp32_arrays["kNarrowBMonoCoeffsF32"]["length"],
            fp32_arrays["kNarrowBcoeffsF32"]["length"])
    check32("narrow region B rational: pieces",
            len(fp32_rat_pieces),
            fp32_constants["kNarrowRatBPiecesCountF32"]["value"])
    check32("narrow region B rational: sum(m+1+k)",
            fp32_rat_total, fp32_constants["kNarrowRatBStoredF32"]["value"])
    check32("narrow region B rational: extern extent",
            fp32_extents.get("kNarrowRatBCoeffsF32", {}).get("extent"),
            fp32_constants["kNarrowRatBStoredF32"]["value"])
    check32("uniform chebyshev: orders * sum(deg+1)",
            fp32_flat_cheb_total, fp32_flat_rows[0][3])
    check32("uniform chebyshev: len(kFlatDegsF32)",
            fp32_arrays["kFlatDegsF32"]["length"],
            fp32_constants["kFlatIntervalsF32"]["value"])
    check32("uniform chebyshev: offsets end at total",
            fp32_arrays["kFlatOffsetsF32"]["values"][-1],
            fp32_flat_rows[0][3])
    check32("uniform rational: orders * sum(per-cell stored)",
            fp32_flat_rat_total,
            fp32_constants["kFlatRatStoredTotalF32"]["value"])
    check32("uniform rational: every cell stored == m+1+k",
            all(stored == (m + 1) + k
                for stored, m, k in zip(
                    fp32_arrays["kFlatRatStoredF32"]["values"],
                    fp32_arrays["kFlatRatNumDegF32"]["values"],
                    fp32_arrays["kFlatRatDenDegF32"]["values"])),
            True)
    check32("uniform rational: offsets end at total",
            fp32_arrays["kFlatRatOffsetsF32"]["values"][-1],
            fp32_constants["kFlatRatStoredTotalF32"]["value"])
    check32("coarse region A chebyshev: extern extent",
            fp32_extents.get("kCoeffs", {}).get("extent"),
            fp32_constants["kRegionAFitChebStored"]["value"])
    check32("coarse region A rational: extern extent",
            fp32_extents.get("kRatACoeffs", {}).get("extent"),
            fp32_constants["kRegionAFitRatStored"]["value"])
    check32("narrow region A chebyshev: row stored",
            fp32_narrow_a_rows[0][3], 1526)
    check32("narrow region A rational stored",
            fp32_constants["kNarrowRatAStoredF32"]["value"], 1241)
    check32("uniform chebyshev: extern extent",
            fp32_extents.get("kFlatCoeffsF32", {}).get("extent"), 12705)
    check32("uniform rational: extern extent",
            fp32_extents.get("kFlatRatCoeffsF32", {}).get("extent"), 15972)

    fp32_bar_check = {"ran": False}
    try:
        g = load_generator(boys)
        fp32_bar_check = {
            "ran": True,
            "generator_TOL_FLOAT": g.mp.nstr(g.TOL_FLOAT, 20),
            "generator_F32_RAT_BOUND": g.mp.nstr(g.F32_RAT_BOUND, 20),
            "generator_F32_RAT_NARROW_ACCEPT": g.mp.nstr(
                g.F32_RAT_NARROW_ACCEPT, 20),
            "header_kRegionBFitBar": fp32_constants["kRegionBFitBar"]["raw"],
            "header_kFlatRatBarF32": fp32_constants["kFlatRatBarF32"]["raw"],
            # The generator's constants were rounded to mpf at import time,
            # before this module raised the working precision, so compare them
            # relatively rather than by bit equality.
            "ok": (abs(g.F32_RAT_BOUND - g.mpf("1.5e-7"))
                   / g.mpf("1.5e-7") < g.mpf("1e-15")
                   and abs(g.TOL_FLOAT - g.mpf("1e-7"))
                   / g.mpf("1e-7") < g.mpf("1e-15")),
        }
    except Exception as exc:  # the static read does not depend on the generator
        fp32_bar_check = {"ran": False, "error": repr(exc)}

    fp32_counts = {
        "region_b_coarse": {
            "chebyshev": {
                "stored": fp32_constants["kRegionBFitChebStored"]["value"],
                "degree": fp32_constants["kBDeg"]["value"],
                "stored_rule": "degree + 1",
                "line": fp32_constants["kRegionBFitChebStored"]["line"],
                "array_line": fp32_arrays["kBcoeffs"]["line"],
                "monomial_mirror": fp32_arrays["kMonoBcoeffs"]["length"],
                "delivered": fp32_constants["kRegionBFitChebDelivered"]["value"],
                "delivered_horner":
                fp32_constants["kRegionBFitChebDeliveredHorner"]["value"],
                "bar": fp32_constants["kRegionBFitBar"]["value"],
            },
            "rational": {
                "stored": fp32_constants["kRegionBFitRatStored"]["value"],
                "numerator_degree": fp32_constants["kRatBnumDeg"]["value"],
                "denominator_degree": fp32_constants["kRatBdenDeg"]["value"],
                "stored_rule": "(m+1) + k, q_0 held at 1",
                "line": fp32_constants["kRegionBFitRatStored"]["line"],
                "array_lines": {
                    "num": fp32_arrays["kRatBnum"]["line"],
                    "den": fp32_arrays["kRatBden"]["line"],
                },
                "delivered": fp32_constants["kRegionBFitRatDelivered"]["value"],
                "bar": fp32_constants["kRegionBFitBar"]["value"],
            },
        },
        "region_b_narrow": {
            "chebyshev": {
                "stored": fp32_narrow_rows[0][3],
                "pieces": fp32_constants["kNarrowBPiecesF32"]["value"],
                "degree": fp32_constants["kNarrowBDegF32"]["value"],
                "line": fp32_narrow_rows_line,
                "array_line": fp32_arrays["kNarrowBcoeffsF32"]["line"],
                "monomial_mirror": fp32_arrays["kNarrowBMonoCoeffsF32"]["length"],
                "delivered_fused": fp32_narrow_rows[0][4],
                "delivered_separate": fp32_narrow_rows[0][5],
            },
            "rational": {
                "stored": fp32_constants["kNarrowRatBStoredF32"]["value"],
                "pieces": fp32_constants["kNarrowRatBPiecesCountF32"]["value"],
                "pairs": [[m, k] for _a, _b, m, k, _off in fp32_rat_pieces],
                "line": fp32_constants["kNarrowRatBStoredF32"]["line"],
                "pieces_line": fp32_rat_pieces_line,
                "delivered_fused":
                fp32_constants["kNarrowRatBDeliveredFusedF32"]["value"],
                "delivered_separate":
                fp32_constants["kNarrowRatBDeliveredSeparateF32"]["value"],
            },
        },
        "uniform": {
            "chebyshev": {
                "stored": fp32_flat_rows[0][3],
                "intervals": fp32_constants["kFlatIntervalsF32"]["value"],
                "orders": orders,
                "per_cell": fp32_arrays["kFlatDegsF32"]["values"][0] + 1,
                "line": fp32_flat_rows_line,
                "monomial_mirror_extent":
                fp32_extents.get("kFlatMonoCoeffsF32", {}).get("extent"),
                "monomial_note": (
                    "boys_coefficients.hpp:2067 asserts kFlatMonoCoeffsF32 is "
                    "the same 12705; the monomial copy is an evaluation-scheme "
                    "mirror, not a second fit"
                ),
            },
            "rational": {
                "stored": fp32_constants["kFlatRatStoredTotalF32"]["value"],
                "intervals": fp32_constants["kFlatRatIntervalsF32"]["value"],
                "orders": orders,
                "per_cell_range": [
                    min(fp32_arrays["kFlatRatStoredF32"]["values"]),
                    max(fp32_arrays["kFlatRatStoredF32"]["values"]),
                ],
                "line": fp32_constants["kFlatRatStoredTotalF32"]["line"],
                "delivered": fp32_flat_rat_rows[0][1],
                "bar": fp32_constants["kFlatRatBarF32"]["value"],
            },
        },
        "region_a": {
            "coarse_chebyshev_stored":
            fp32_constants["kRegionAFitChebStored"]["value"],
            "coarse_rational_stored":
            fp32_constants["kRegionAFitRatStored"]["value"],
            "narrow_chebyshev_stored": fp32_narrow_a_rows[0][3],
            "narrow_rational_stored":
            fp32_constants["kNarrowRatAStoredF32"]["value"],
            "note": "context only; region A's routes are not a like-for-like pair",
        },
        "widths": {
            "coarse_region_b": kx1 - kx0,
            "narrow_region_b_pieces": fp32_narrow_widths,
            "uniform_cell": fp32_constants["kFlatWidthF32"]["value"],
            "float_edges": fp32_narrow_edges,
            "region_b_left": kx0,
            "region_b_top": kx1,
            "note": ("the float lane has no placed top, so its band ends at "
                     "kX1; the narrow edges are stored as binary32"),
        },
    }

    result = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "boys_root": str(boys),
            "header": str(header_path),
            "header_sha256": sha256(header_path),
            "generator": str(gen_path),
            "generator_sha256": sha256(gen_path) if gen_path.exists() else None,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_rev(boys),
        },
        "precision_scope": PRECISION_SCOPE,
        "constants": constants,
        "arrays": arrays,
        "derived_counts": doc_counts,
        "widths": widths,
        "constants_fp32": fp32_constants,
        "arrays_fp32": fp32_arrays,
        "derived_counts_fp32": fp32_counts,
        "extents_fp32": fp32_extents,
        "checks": checks,
        "bar_check": bar_check,
        "bar_check_fp32": fp32_bar_check,
        "lane_contract": stale,
    }
    return result


# ---------------------------------------------------------------------------
# generator import and sweep


def load_generator(boys: Path):
    global BOYS_ROOT_STR
    BOYS_ROOT_STR = str(boys)
    path = boys / GENERATOR_REL
    spec = importlib.util.spec_from_file_location("boys_gen_coefficients", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["boys_gen_coefficients"] = mod
    spec.loader.exec_module(mod)
    # The fit path runs at RAT_DPS; set it before the first reference call so
    # the lru_cache never stores a lower-precision value under the same key.
    mod.mp.dps = mod.RAT_DPS
    return mod


def _worker_generator(boys_root: str):
    global _WORKER_GEN
    if _WORKER_GEN is None:
        _WORKER_GEN = load_generator(Path(boys_root))
    return _WORKER_GEN


def rat_candidate_job(spec):
    """One (m,k) solve in a worker process; returns the solver candidates.

    The grid and reference arrive as decimal strings, not mpf objects. A spawn
    unpickles the arguments before this function runs, at the child's default
    precision, and an mpf unpickled there is rounded to that default: measured
    60-digit value -> 15 digits, with the fit changing in its third digit. The
    strings are rebuilt after the generator has set RAT_DPS, so the worker
    solves the parent's problem, not a truncated one.
    """
    boys_root, m, k, indices, ts_text, ref_text, outer, remez_iters = spec
    g = _worker_generator(boys_root)
    ts = [g.mpf(s) for s in ts_text]
    ref = [g.mpf(s) for s in ref_text]
    return rational_candidates(g, m, k, indices, ts, ref, outer, remez_iters)


def ship_text(g, values):
    """Decimal strings that round-trip a 60-dps value through a spawn."""
    return [g.mp.nstr(v, 62) for v in values]


def make_nodes(g, a, b, n):
    ts = [g.mpf(2 * i) / n - 1 for i in range(n + 1)]
    xs = [a + (b - a) * (t + 1) / 2 for t in ts]
    return ts, xs


def reference(g, xs):
    with g.mp.workdps(g.RAT_DPS):
        return [g.boys_ref(0, x, g.RAT_TAIL_FLOOR, g.RAT_TERMS) for x in xs]


def kernel_t(g, a, b, x) -> float:
    """The kernel's mapping to t, in binary64."""
    return 2.0 * (float(x) - float(a)) / (float(b) - float(a)) - 1.0


def errors_from_values(g, values, xs, ref):
    worst_rel = g.mpf(0)
    worst_abs = g.mpf(0)
    at_rel = None
    at_abs = None
    for value, x, r in zip(values, xs, ref):
        e = abs(g.mpf(value) - r)
        if e > worst_abs:
            worst_abs, at_abs = e, x
        rel = e / r
        if rel > worst_rel:
            worst_rel, at_rel = rel, x
    return worst_rel, worst_abs, at_rel, at_abs


def meets(target, worst_rel, worst_abs) -> bool:
    return worst_rel <= target["value"] if target["kind"] == "rel" else worst_abs <= target["value"]


def target_tag(target) -> str:
    return f"{target['kind']}:{target['value']}"


# --- Chebyshev


def cheb_fit(g, a, b, deg):
    with g.mp.workdps(g.RAT_DPS):
        cs = g.cheb_coeffs(0, a, b, deg)
    return [float(c) for c in cs]


def cheb_errors(g, cs, a, b, xs, ref):
    values = []
    for x in xs:
        try:
            values.append(g.clenshaw_split_double(cs, kernel_t(g, a, b, x)))
        except Exception:
            values.append(float("nan"))
    if any(not math.isfinite(v) for v in values):
        return None
    return errors_from_values(g, values, xs, ref)


def cheb_count(g, a, b, xs, ref, target, max_deg=CHEB_MAX_DEG, verbose=False):
    started = time.perf_counter()
    for deg in range(CHEB_MIN_DEG, max_deg + 1):
        if verbose:
            print(f"  cheb degree {deg} ...", flush=True)
        cs = cheb_fit(g, a, b, deg)
        err = cheb_errors(g, cs, a, b, xs, ref)
        if err is None:
            continue
        wr, wa, at_rel, at_abs = err
        if meets(target, wr, wa):
            return {
                "family": "chebyshev",
                "stored": deg + 1,
                "m": deg,
                "k": None,
                "delivered_rel": g.mp.nstr(wr, 17),
                "delivered_abs": g.mp.nstr(wa, 17),
                "at_x_rel": g.mp.nstr(at_rel, 17),
                "at_x_abs": g.mp.nstr(at_abs, 17),
                "refined": True,
                "seconds": round(time.perf_counter() - started, 3),
            }
    return {
        "family": "chebyshev",
        "stored": None,
        "m": None,
        "k": None,
        "status": f"not reached by degree {max_deg}",
        "seconds": round(time.perf_counter() - started, 3),
    }


# --- rational


def rational_candidates(g, m, k, indices, ts, ref, outer,
                        remez_iters=RAT_REMEZ_ITERS):
    out = []
    for name, solver, kwargs in (
        ("remez", g.rat_remez, {"iters": remez_iters}),
        ("lawson", g.rat_lawson, {"outer": outer}),
    ):
        try:
            r = solver(m, k, indices, ts, ref, None, **kwargs)
        except Exception:
            r = None
        if r is not None:
            _, p, q = r
            out.append((name, p, q))
    return out


def rat_errors(g, p, q, a, b, xs, ref):
    pd = [float(c) for c in p]
    qd = [float(c) for c in q]
    best = None
    for fused in (True, False):
        values = []
        for x in xs:
            try:
                values.append(g.rat_eval_double(pd, qd, kernel_t(g, a, b, x), fused))
            except Exception:
                values.append(float("nan"))
        if any(not math.isfinite(v) for v in values):
            continue
        err = errors_from_values(g, values, xs, ref)
        if best is None:
            best = err
        else:
            # The two routes are alternative evaluations; the bound is the
            # worse of the two for each criterion, as the library measures it.
            wr = max(best[0], err[0])
            wa = max(best[1], err[1])
            at_rel = best[2] if best[0] >= err[0] else err[2]
            at_abs = best[3] if best[1] >= err[1] else err[3]
            best = (wr, wa, at_rel, at_abs)
    return best


# --- the paper cell's independent cross-check


def rat_interp(g, m, k, a, b, xs, ref):
    """Interpolate f rationally at Chebyshev-Gauss nodes; an independent witness.

    The interpolation conditions p(t_i) - f(t_i) q(t_i) = 0 with q_0 = 1 form a
    square system of m + k + 1 equations in as many unknowns, solved at 60
    digits. The method shares no code with rat_remez or rat_lawson and is not a
    minimax solver: an interpolant that meets the target is an existence
    witness at its stored count, while one that misses says nothing about the
    minimax count. Both readings are reported as they fall.
    """
    n = (m + 1) + k
    with g.mp.workdps(RAT_INTERP_DPS):
        nodes_t = [g.mp.cos(g.mp.pi * (2 * i + 1) / (2 * n)) for i in range(n)]
        nodes_f = [g.boys_ref(0, a + (b - a) * (t + 1) / 2,
                              g.RAT_TAIL_FLOOR, g.RAT_TERMS) for t in nodes_t]
        mat = g.mp.matrix(n, n)
        rhs = g.mp.matrix(n, 1)
        for i, (t, f) in enumerate(zip(nodes_t, nodes_f)):
            for j in range(m + 1):
                mat[i, j] = t ** j
            for j in range(1, k + 1):
                mat[i, m + j] = -f * t ** j
            rhs[i] = f
        try:
            sol = g.mp.lu_solve(mat, rhs)
        except (ZeroDivisionError, ValueError):
            return None
        # lu_solve can answer a rank-starved system with something that is not
        # its solution. A residual this large means the equations were not the
        # ones solved, whatever the function does on the grid.
        scale = max(abs(f) for f in nodes_f)
        resid = max(abs(sum(mat[i, j] * sol[j] for j in range(n)) - rhs[i])
                    for i in range(n))
        if not resid <= scale * RAT_INTERP_RESID:
            return None
    p = [sol[j] for j in range(m + 1)]
    q = [sol[m + j] for j in range(1, k + 1)]
    # The kernel divides by 1 + q_1 t + ... + q_k t^k. A denominator that comes
    # near zero inside the interval is not a fit, however tidy its residual on
    # the interpolation nodes. The same guard the generator's Remez applies.
    qd = [float(c) for c in q]
    dens = []
    for x in xs:
        t = kernel_t(g, a, b, x)
        acc = 0.0
        for c in reversed(qd):
            acc = acc * t + c
        dens.append(abs(1.0 + acc * t))
    if (any(not math.isfinite(d) for d in dens)
            or min(dens, default=0.0)
            < max(dens, default=0.0) * RAT_INTERP_DEN_FLOOR):
        return None
    err = rat_errors(g, p, q, a, b, xs, ref)
    if err is None:
        return None
    wr, wa, at_rel, at_abs = err
    return {
        "m": m, "k": k, "stored": (m + 1) + k,
        "rel": g.mp.nstr(wr, 17), "abs": g.mp.nstr(wa, 17),
        "at_x_rel": g.mp.nstr(at_rel, 17), "at_x_abs": g.mp.nstr(at_abs, 17),
    }


def independent_cross_check(g, a, b, xs, ref, target, stored_counts=(11, 12)):
    """Ask a non-library method whether these stored counts can reach the target.

    Stored 12 is the neighbour's count: a meeting interpolant is an independent
    witness that the count is attainable. Stored 11 is the count below it: a
    meeting interpolant would contradict the sweep's below-count result and is
    reported as a contradiction, never folded quietly into the record.
    """
    out = {
        "method": ("rational interpolation at Chebyshev-Gauss nodes, "
                   "monomial basis in t, solved in mpmath"),
        "dps": RAT_INTERP_DPS,
        "shares_code_with_the_sweep_solvers": False,
        "one_sided": (
            "Interpolation is not minimax. A meeting interpolant witnesses that "
            "the stored count can reach the target; a miss is inconclusive and "
            "is not evidence against the minimax count."
        ),
        "target": target["spec"],
        "at": {},
    }
    for stored in stored_counts:
        pairs = [(stored - 1 - k, k)
                 for k in range(1, min(g.RAT_K_MAX, stored - 2) + 1)]
        rows = []
        for m, k in pairs:
            fit = rat_interp(g, m, k, a, b, xs, ref)
            if fit is not None:
                rows.append(fit)
        rows.sort(key=lambda r: float(r["abs"]))
        best = rows[0] if rows else None
        met = False
        if best is not None:
            met = (float(best["rel"]) <= target["value"]
                   if target["kind"] == "rel"
                   else float(best["abs"]) <= target["value"])
        out["at"][str(stored)] = {
            "stored": stored,
            "pairs_tried": [f"[{m}/{k}]" for m, k in pairs],
            "fits_that_solved": len(rows),
            "best": best,
            "meets": met,
        }
    out["stored_12_witnessed"] = out["at"]["12"]["meets"]
    out["stored_11_contradicts_the_below_check"] = out["at"]["11"]["meets"]
    return out


def scan_level(g, a, b, ts, xs, ref, indices, target, stored, outer,
               remez_iters=RAT_REMEZ_ITERS):
    """Every (m,k) at one stored count, on the given grid. Returns hits."""
    hits = []
    k_max = min(g.RAT_K_MAX, stored - 2)
    if k_max < 1:
        return hits
    sub_xs = [xs[i] for i in indices]
    sub_ref = [ref[i] for i in indices]
    pairs = [(stored - 1 - k, k) for k in range(1, k_max + 1)]
    if g.worker_count() > 1 and len(pairs) > 1:
        # The generator's own pool and job runner. Each worker loads the
        # generator once (it is the same file the parent imported) and solves
        # its share; results come back in submission order. The inputs travel
        # as text because a spawn unpickles them too early for an mpf.
        ts_text = ship_text(g, ts)
        ref_text = ship_text(g, ref)
        specs = [(BOYS_ROOT_STR, m, k, indices, ts_text, ref_text, outer,
                  remez_iters) for m, k in pairs]
        results = g.run_jobs(rat_candidate_job, specs)
        candidates = [(m, k, cand)
                      for (m, k), out in zip(pairs, results)
                      for cand in out]
    else:
        candidates = [(m, k, cand) for m, k in pairs
                      for cand in rational_candidates(
                          g, m, k, indices, ts, ref, outer, remez_iters)]
    for m, k, (name, p, q) in candidates:
        err = rat_errors(g, p, q, a, b, sub_xs, sub_ref)
        if err is None:
            continue
        wr, wa, at_rel, at_abs = err
        hit = {
            "name": name, "m": m, "k": k, "p": p, "q": q,
            "rel": wr, "abs": wa,
            "at_rel": at_rel, "at_abs": at_abs,
        }
        if meets(target, wr, wa):
            hits.append(hit)
    hits.sort(key=lambda h: h["rel"] if target["kind"] == "rel" else h["abs"])
    return hits


def refine_pairs(g, pairs, a, b, ts, xs, ref, target,
                 refine_outer=RAT_REFINE_OUTER,
                 remez_iters=RAT_REMEZ_ITERS):
    """Solve each (m,k) on the full certifying grid; return the meeting fits.

    This is the generator's own refinement shape: the whole batch of pairs is
    one list of jobs, and its worker pool runs them. One pair costs minutes at
    the full grid and the pool is the difference between a row and a night.
    """
    full_idx = list(range(len(ts)))
    if g.worker_count() > 1 and len(pairs) > 1:
        ts_text = ship_text(g, ts)
        ref_text = ship_text(g, ref)
        specs = [(BOYS_ROOT_STR, m, k, full_idx, ts_text, ref_text,
                  refine_outer, remez_iters) for m, k in pairs]
        outs = g.run_jobs(rat_candidate_job, specs)
        candidates = [(m, k, cand)
                      for (m, k), out in zip(pairs, outs)
                      for cand in out]
    else:
        candidates = [(m, k, cand) for m, k in pairs
                      for cand in rational_candidates(
                          g, m, k, full_idx, ts, ref, refine_outer,
                          remez_iters)]
    refined = []
    for m, k, (name, p, q) in candidates:
        err = rat_errors(g, p, q, a, b, xs, ref)
        if err is None:
            continue
        wr, wa, at_rel, at_abs = err
        if meets(target, wr, wa):
            refined.append({
                "name": name, "m": m, "k": k,
                "rel": wr, "abs": wa,
                "at_rel": at_rel, "at_abs": at_abs,
            })
    return refined


def rational_count(g, a, b, ts, xs, ref, target, coarse_idx,
                   max_stored=RAT_MAX_STORED,
                   scan_outer=RAT_SCAN_OUTER,
                   refine_outer=RAT_REFINE_OUTER,
                   remez_iters=RAT_REMEZ_ITERS,
                   verbose=False):
    started = time.perf_counter()
    for stored in range(RAT_MIN_STORED, max_stored + 1):
        if verbose:
            print(f"  rat stored {stored} ...", flush=True)
        hits = scan_level(g, a, b, ts, xs, ref, coarse_idx, target,
                          stored, scan_outer, remez_iters)
        if not hits:
            continue
        # Re-solve every pair that reached on the coarse grid on the full
        # certifying grid. The count is the smallest stored that reaches there.
        refined = refine_pairs(
            g, [(hit["m"], hit["k"]) for hit in hits], a, b, ts, xs, ref,
            target, refine_outer, remez_iters)
        if not refined:
            continue
        refined.sort(key=lambda h: h["rel"] if target["kind"] == "rel" else h["abs"])
        best = refined[0]
        # The knee is the smallest reaching count only if the count below does
        # not reach. Every degree pair at stored - 1 is solved on the full
        # grid, not only the pairs the coarse scan liked: a pair that misses
        # on the coarse grid can still reach on the full one.
        below_meets = False
        if stored > RAT_MIN_STORED:
            below_total = stored - 1
            below_pairs = [(below_total - 1 - k, k)
                           for k in range(1, min(g.RAT_K_MAX,
                                                 below_total - 2) + 1)]
            below_meets = bool(refine_pairs(
                g, below_pairs, a, b, ts, xs, ref, target, refine_outer,
                remez_iters))
        return {
            "family": "rational",
            "stored": (best["m"] + 1) + best["k"],
            "m": best["m"],
            "k": best["k"],
            "solver": best["name"],
            "delivered_rel": g.mp.nstr(best["rel"], 17),
            "delivered_abs": g.mp.nstr(best["abs"], 17),
            "at_x_rel": g.mp.nstr(best["at_rel"], 17),
            "at_x_abs": g.mp.nstr(best["at_abs"], 17),
            "refined": True,
            "below_meets": below_meets,
            "seconds": round(time.perf_counter() - started, 3),
        }
    return {
        "family": "rational",
        "stored": None,
        "m": None,
        "k": None,
        "status": f"not reached by {max_stored} stored with k <= {g.RAT_K_MAX}",
        "seconds": round(time.perf_counter() - started, 3),
    }


# ---------------------------------------------------------------------------
# sweep driver


def parse_targets(spec: str):
    targets = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        kind, value = part.split(":", 1)
        if kind not in ("rel", "abs"):
            raise ValueError(f"target kind must be rel or abs: {part}")
        targets.append({"kind": kind, "value": float(value), "spec": part})
    if not targets:
        raise ValueError("no targets")
    return targets


def width_set(g, which: str):
    """The width grid. Named sets, or a comma list of widths.

    Tokens: a dyadic power ``2^-7`` .. ``2^5``, the names ``ship`` and
    ``paper``, or any decimal width. ``ship`` is kFlatHi - kX0, the shipped
    coarse region-B interval; ``paper`` is kX1 - kX0, the nearest-neighbour's
    region-B interval. The grid is measured left edge kX0 for every width.
    """
    powers = list(range(-7, 6))
    ship = g.mpf("31") - g.X0
    paper = g.X1 - g.X0
    named = {"ship": ship, "paper": paper}
    if which == "all":
        ordered = [("2^%d" % j, g.mpf(2) ** j) for j in powers]
        ordered += [("ship", ship), ("paper", paper)]
    elif which == "quick":
        keep = {-5, -3, -1, 1, 3, 5}
        ordered = [("2^%d" % j, g.mpf(2) ** j) for j in powers if j in keep]
        ordered += [("ship", ship), ("paper", paper)]
    else:
        ordered = []
        seen = set()
        for token in which.split(","):
            token = token.strip()
            if not token:
                continue
            if token in seen:
                raise ValueError(f"width named twice: {token}")
            seen.add(token)
            if token in named:
                ordered.append((token, named[token]))
            elif re.fullmatch(r"2\^-?\d+", token):
                width = g.mpf(2) ** int(token[2:])
                ordered.append((token, width))
            else:
                width = g.mpf(token)
                ordered.append((token, width))
            if ordered[-1][1] <= 0:
                raise ValueError(f"width must be positive: {token}")
        if not ordered:
            raise ValueError("no widths")
    return ordered


def sweep_one(g, a, width_name, width, target, n_nodes, stride,
              scan_outer=RAT_SCAN_OUTER, refine_outer=RAT_REFINE_OUTER,
              remez_iters=RAT_REMEZ_ITERS, verbose=False):
    b = a + width
    ts, xs = make_nodes(g, a, b, n_nodes)
    ref = reference(g, xs)
    coarse_idx = list(range(0, n_nodes + 1, stride))
    cheb = cheb_count(g, a, b, xs, ref, target, verbose=verbose)
    rat = rational_count(g, a, b, ts, xs, ref, target, coarse_idx,
                         scan_outer=scan_outer,
                         refine_outer=refine_outer,
                         remez_iters=remez_iters,
                         verbose=verbose)
    row = {
        "precision": "fp64",
        "target": target_tag(target),
        "target_kind": target["kind"],
        "target_value": target["value"],
        "width_name": width_name,
        "width": g.mp.nstr(width, 17),
        "width_f": float(width),
        "a": g.mp.nstr(a, 17),
        "b": g.mp.nstr(b, 17),
        "cheb": cheb,
        "rat": rat,
        "saving": (cheb["stored"] - rat["stored"])
        if cheb.get("stored") is not None and rat.get("stored") is not None else None,
        "settings": {
            "nodes": n_nodes + 1,
            "coarse_stride": stride,
            "scan_outer": scan_outer,
            "refine_outer": refine_outer,
            "remez_iters": remez_iters,
            "workers": g.worker_count(),
        },
    }
    return row


def json_ready(obj):
    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_ready(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(obj), indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "precision", "target", "target_kind", "target_value", "width_name",
        "width", "width_f",
        "family", "stored", "m", "k", "solver", "delivered_rel", "delivered_abs",
        "at_x_rel", "at_x_abs", "refined", "below_meets", "seconds", "status",
        "nodes", "coarse_stride", "scan_outer", "refine_outer", "remez_iters",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            settings = row.get("settings", {})
            for family in ("cheb", "rat"):
                entry = row[family]
                writer.writerow({
                    "precision": row.get("precision"),
                    "target": row["target"],
                    "target_kind": row["target_kind"],
                    "target_value": row["target_value"],
                    "width_name": row["width_name"],
                    "width": row["width"],
                    "width_f": row["width_f"],
                    "family": entry["family"],
                    "stored": entry.get("stored"),
                    "m": entry.get("m"),
                    "k": entry.get("k"),
                    "solver": entry.get("solver"),
                    "delivered_rel": entry.get("delivered_rel"),
                    "delivered_abs": entry.get("delivered_abs"),
                    "at_x_rel": entry.get("at_x_rel"),
                    "at_x_abs": entry.get("at_x_abs"),
                    "refined": entry.get("refined"),
                    "below_meets": entry.get("below_meets"),
                    "seconds": entry.get("seconds"),
                    "status": entry.get("status"),
                    "nodes": settings.get("nodes"),
                    "coarse_stride": settings.get("coarse_stride"),
                    "scan_outer": settings.get("scan_outer"),
                    "refine_outer": settings.get("refine_outer"),
                    "remez_iters": settings.get("remez_iters"),
                })


def print_table(rows):
    print()
    print(f"{'target':<12} {'width':<10} {'W':>10} {'cheb':>5} {'rat':>5} "
          f"{'saving':>7} {'cheb_res':>11} {'rat_res':>11}")
    for row in rows:
        c = row["cheb"].get("stored")
        r = row["rat"].get("stored")
        cres = row["cheb"].get("delivered_rel") if row["target_kind"] == "rel" \
            else row["cheb"].get("delivered_abs")
        rres = row["rat"].get("delivered_rel") if row["target_kind"] == "rel" \
            else row["rat"].get("delivered_abs")
        print(f"{row['target']:<12} {row['width_name']:<10} {row['width_f']:>10.6g} "
              f"{str(c):>5} {str(r):>5} {str(row['saving']):>7} "
              f"{str(cres):>11} {str(rres):>11}")


def load_done_rows(path: Path):
    """Rows already written by an earlier invocation, keyed by target/width."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"warning: cannot read {path} ({exc}); starting fresh", flush=True)
        return {}
    if data.get("meta", {}).get("schema") != ROW_SCHEMA:
        print(f"warning: {path} has schema "
              f"{data.get('meta', {}).get('schema')!r}, expected {ROW_SCHEMA}; "
              f"starting fresh", flush=True)
        return {}
    return {(row["target"], row["width_name"]): row
            for row in data.get("rows", []) if row.get("cheb") and row.get("rat")}


def ordered_rows(done, targets, widths):
    return [done[(target_tag(t), name)]
            for t in targets for name, _ in widths
            if (target_tag(t), name) in done]


def sweep_meta(boys: Path, g, args) -> dict:
    """What produced these rows: source identity, grid, solver settings."""
    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": ROW_SCHEMA,
        "boys_root": str(boys),
        "git": git_rev(boys),
        "header": {"path": str(boys / HEADER_REL),
                   "sha256": sha256(boys / HEADER_REL)},
        "generator": {"path": str(boys / GENERATOR_REL),
                      "sha256": sha256(boys / GENERATOR_REL)},
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "precision_scope": PRECISION_SCOPE,
        "grid": {"nodes": args.nodes + 1, "coarse_stride": args.stride},
        "solver": {
            "scan_outer": args.scan_outer,
            "refine_outer": args.refine_outer,
            "remez_iters": args.remez_iters,
            "cheb_deg": [CHEB_MIN_DEG, CHEB_MAX_DEG],
            "rat_stored": [RAT_MIN_STORED, RAT_MAX_STORED],
            "rat_k_max": g.RAT_K_MAX,
        },
    }


def write_sweep_outputs(out_dir: Path, rows, targets, meta):
    for target in targets:
        tag = target_tag(target).replace(":", "_")
        trows = [r for r in rows if r["target"] == target_tag(target)]
        if trows:
            write_json(out_dir / f"sweep_{tag}.json",
                       {"meta": meta, "precision_scope": PRECISION_SCOPE,
                        "rows": trows})
            write_csv(out_dir / f"sweep_{tag}.csv", trows)
    write_json(out_dir / "sweep_all.json",
               {"meta": meta, "precision_scope": PRECISION_SCOPE,
                "rows": rows})


def sweep_mode(args) -> dict:
    boys = Path(args.boys_root)
    g = load_generator(boys)
    g.JOBS = args.jobs if args.jobs and args.jobs > 0 else None
    a = g.X0
    targets = parse_targets(args.targets)
    widths = width_set(g, args.widths)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_path = out_dir / "sweep_all.json"
    done = load_done_rows(all_path)
    current = {"nodes": args.nodes + 1, "coarse_stride": args.stride,
               "scan_outer": args.scan_outer,
               "refine_outer": args.refine_outer,
               "remez_iters": args.remez_iters}
    stale = [key for key in done
             if {k: done[key].get("settings", {}).get(k)
                 for k in SETTING_KEYS} != current]
    for key in stale:
        print(f"warning: {key[0]}/{key[1]} was written with other solver "
              f"settings; re-running it", flush=True)
        del done[key]
    meta = sweep_meta(boys, g, args)
    print(f"grid: {len(widths)} width(s) x {len(targets)} target(s) = "
          f"{len(widths) * len(targets)} row(s); {len(done)} already on disk", flush=True)
    print(f"workers: {g.worker_count()} (jobs={args.jobs or 'auto'})", flush=True)
    started = time.perf_counter()
    remaining = []
    for target in targets:
        for name, width in widths:
            key = (target_tag(target), name)
            if key in done:
                continue
            if args.max_seconds and time.perf_counter() - started >= args.max_seconds:
                remaining.append(key)
                continue
            row = sweep_one(g, a, name, width, target, args.nodes, args.stride,
                            scan_outer=args.scan_outer,
                            refine_outer=args.refine_outer,
                            remez_iters=args.remez_iters,
                            verbose=args.progress)
            done[key] = row
            rows = ordered_rows(done, targets, widths)
            write_sweep_outputs(out_dir, rows, targets, meta)
            c = row["cheb"].get("stored")
            r = row["rat"].get("stored")
            print(f"[{key[0]}] {name:<10} W={row['width_f']:<10.6g} "
                  f"cheb={c} rat={r} saving={row['saving']} "
                  f"({row['cheb'].get('seconds')}s + {row['rat'].get('seconds')}s)",
                  flush=True)
    rows = ordered_rows(done, targets, widths)
    write_sweep_outputs(out_dir, rows, targets, meta)
    print_table(rows)
    if remaining:
        print(f"\n{len(remaining)} row(s) remain in this grid: "
              f"{', '.join(f'{t}/{w}' for t, w in remaining)}")
        print("re-run the same command; written rows are reused")
    print(f"invocation wall time: {time.perf_counter() - started:.1f} s")
    return {"rows": rows, "remaining": remaining}


def probe_mode(args) -> dict:
    """One width at one target, to calibrate the sweep's runtime."""
    boys = Path(args.boys_root)
    g = load_generator(boys)
    g.JOBS = args.jobs if args.jobs and args.jobs > 0 else None
    a = g.X0
    targets = parse_targets(args.targets)
    widths = width_set(g, args.width_name)
    name, width = widths[0]
    started = time.perf_counter()
    row = sweep_one(g, a, name, width, targets[0], args.nodes, args.stride,
                    scan_outer=args.scan_outer,
                    refine_outer=args.refine_outer,
                    remez_iters=args.remez_iters,
                    verbose=True)
    result = {"schema": ROW_SCHEMA, "meta": sweep_meta(boys, g, args),
              "precision_scope": PRECISION_SCOPE, "row": row}
    print(json.dumps(json_ready(result), indent=2))
    print(f"probe wall time: {time.perf_counter() - started:.1f} s")
    if args.out:
        write_json(Path(args.out), result)
    return result


def paper_cell_mode(args) -> dict:
    boys = Path(args.boys_root)
    g = load_generator(boys)
    g.JOBS = args.jobs if args.jobs and args.jobs > 0 else None
    a = g.X0
    b = g.X1
    width = b - a
    target = {"kind": "abs", "value": 5e-14, "spec": "abs:5e-14"}
    ts, xs = make_nodes(g, a, b, args.nodes)
    ref = reference(g, xs)
    coarse_idx = list(range(0, args.nodes + 1, args.stride))
    cheb = cheb_count(g, a, b, xs, ref, target)
    rat = rational_count(g, a, b, ts, xs, ref, target, coarse_idx,
                         scan_outer=args.scan_outer,
                         refine_outer=args.refine_outer,
                         remez_iters=args.remez_iters)
    cross = independent_cross_check(g, a, b, xs, ref, target)
    result = {
        "meta": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "boys_root": str(boys),
            "git": git_rev(boys),
            "header": {"path": str(boys / HEADER_REL),
                       "sha256": sha256(boys / HEADER_REL)},
            "generator": {"path": str(boys / GENERATOR_REL),
                          "sha256": sha256(boys / GENERATOR_REL)},
            "python": sys.version.split()[0],
            "note": "the nearest-neighbour's exact region-B interval and target",
            "precision_scope": PRECISION_SCOPE,
        },
        "settings": {
            "nodes": args.nodes + 1,
            "coarse_stride": args.stride,
            "scan_outer": args.scan_outer,
            "refine_outer": args.refine_outer,
            "remez_iters": args.remez_iters,
            "workers": g.worker_count(),
        },
        "precision": "fp64",
        "a": g.mp.nstr(a, 17),
        "b": g.mp.nstr(b, 17),
        "width": g.mp.nstr(width, 17),
        "width_f": float(width),
        "target": target["spec"],
        "expected_rational_stored_from_the_paper": 12,
        "cheb": cheb,
        "rat": rat,
        "independent_cross_check": cross,
    }
    if args.out:
        write_json(Path(args.out), result)
    print(json.dumps(json_ready(result), indent=2))
    return result


def plan_mode(args) -> dict:
    """Print the grid, conventions, commands and the measured runtime bound."""
    boys = Path(args.boys_root)
    g = load_generator(boys)
    a = g.X0
    targets = parse_targets(args.targets)
    widths = width_set(g, args.widths)
    g.JOBS = args.jobs if args.jobs and args.jobs > 0 else None
    header_path = boys / HEADER_REL
    script = Path(__file__).resolve()
    py = sys.executable
    boys_arg = f'--boys-root "{boys}"'
    stability = ("--scan-outer 200 --refine-outer 600 --remez-iters 120 "
                 "--nodes 4000")
    plan = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "boys_root": str(boys),
        "git": git_rev(boys),
        "header": {"path": str(header_path), "sha256": sha256(header_path)},
        "generator": {"path": str(boys / GENERATOR_REL),
                      "sha256": sha256(boys / GENERATOR_REL)},
        "python": sys.version.split()[0],
        "counting_rules": {
            "chebyshev": "degree d stores d + 1; Clenshaw reads one array; "
                         "the monomial mirror is the same coefficients and is "
                         "not counted",
            "rational": "(m + 1) numerator + k denominator; q_0 is held at 1 "
                        "and not stored; one representation only",
            "matched": "both families meet the same target on the same "
                       "interval; the count is the smallest stored that "
                       "reaches on the full certifying grid",
        },
        "precision_scope": PRECISION_SCOPE,
        "interval_left_edge": g.mp.nstr(a, 17),
        "targets": [t["spec"] for t in targets],
        "widths": [{"name": name, "width": g.mp.nstr(width, 17),
                    "width_f": float(width)} for name, width in widths],
        "grid": {"nodes": args.nodes + 1, "coarse_stride": args.stride,
                 "rows": len(widths) * len(targets)},
        "solver": {
            "cheb_deg": [CHEB_MIN_DEG, CHEB_MAX_DEG],
            "rat_stored": [RAT_MIN_STORED, RAT_MAX_STORED],
            "rat_k_max": g.RAT_K_MAX,
            "scan_outer": args.scan_outer,
            "refine_outer": args.refine_outer,
            "remez_iters": args.remez_iters,
            "workers": g.worker_count(),
        },
        "runtime": PROBE_LOADED_2026_10_19,
        "commands": {
            "probe": f'"{py}" "{script}" --probe {boys_arg} '
                     f'--width-name ship --targets rel:1e-13 '
                     f'--out results/probe_ship.json',
            "sweep_chunk": f'"{py}" "{script}" --sweep {boys_arg} '
                           f'--out-dir results --targets rel:1e-13,abs:5e-14 '
                           f'--widths all --max-seconds 3600',
            "stability": f'"{py}" "{script}" --sweep {boys_arg} '
                         f'--out-dir results-stability '
                         f'--targets rel:1e-13,abs:5e-14 --widths all '
                         f'--max-seconds 3600 {stability}',
            "paper_cell": f'"{py}" "{script}" --paper-cell {boys_arg} '
                          f'--out results/paper_cell.json',
        },
        "paper_cell_cross_check": (
            "the paper cell also runs a rational interpolant at Chebyshev-Gauss "
            "nodes, written in this file and independent of the library's two "
            "solvers. A meeting interpolant at 12 witnesses the neighbour's "
            "count; a meeting at 11 would contradict the below-count check. A "
            "miss at either is inconclusive, because interpolation is not "
            "minimax. It feeds no sweep count."
        ),
        "resume": (
            "sweep_all.json is rewritten after every completed row; --max-seconds "
            "stops before starting a new row once the budget is spent, and the "
            "next invocation reuses every row already on disk"
        ),
        "no_library_change": (
            "this harness imports the generator; it does not edit the Boys "
            "repository and does not need a rebuild"
        ),
    }
    print(json.dumps(json_ready(plan), indent=2))
    if args.out:
        write_json(Path(args.out), plan)
    return plan


# ---------------------------------------------------------------------------
# entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--boys-root", default=str(DEFAULT_BOYS))
    parser.add_argument("--static", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--paper-cell", action="store_true")
    parser.add_argument("--plan", action="store_true",
                        help="print the grid, conventions, commands and the "
                             "measured runtime bound; run no fit")
    parser.add_argument("--out", default=None,
                        help="JSON output for static/probe/paper-cell/plan")
    parser.add_argument("--out-dir", default=None, help="directory for sweep outputs")
    parser.add_argument("--targets", default="rel:1e-13,abs:5e-14",
                        help="comma list of rel:V or abs:V")
    parser.add_argument("--widths", default="all",
                        help="all, quick, or a comma list of widths: 2^-7..2^5, "
                             "ship, paper, or a decimal width")
    parser.add_argument("--width-name", default="ship",
                        help="width for --probe (a --widths token)")
    parser.add_argument("--nodes", type=int, default=N_NODES)
    parser.add_argument("--stride", type=int, default=COARSE_STRIDE)
    parser.add_argument("--scan-outer", type=int, default=RAT_SCAN_OUTER,
                        help="Lawson outer iterations in the coarse scan")
    parser.add_argument("--refine-outer", type=int, default=RAT_REFINE_OUTER,
                        help="Lawson outer iterations in the full-grid refinement")
    parser.add_argument("--remez-iters", type=int, default=RAT_REMEZ_ITERS,
                        help="Remez exchange iterations")
    parser.add_argument("--jobs", type=int, default=0,
                        help="rational-scan workers; 0 uses the generator's "
                             "default (one per CPU)")
    parser.add_argument("--max-seconds", type=float, default=0,
                        help="stop before starting a row once this much of the "
                             "invocation has elapsed (0 = no limit); a row "
                             "already running is not interrupted")
    parser.add_argument("--progress", action="store_true",
                        help="print each degree / stored count as it is tried")
    args = parser.parse_args(argv)

    modes = [args.static, args.sweep, args.probe, args.paper_cell, args.plan]
    if sum(bool(m) for m in modes) != 1:
        parser.error("name exactly one of --static, --sweep, --probe, "
                     "--paper-cell, --plan")

    if args.static:
        result = static_mode(args)
        if args.out:
            write_json(Path(args.out), result)
        for check in result["checks"]:
            state = "ok" if check["ok"] else "FAIL"
            print(f"{state:>4}  {check['name']}: {check['value']} (expected {check['expected']})")
        print()
        counts = result["derived_counts"]["region_b_coarse"]
        print(f"coarse region B: chebyshev {counts['chebyshev']['stored']} stored "
              f"(line {counts['chebyshev']['line']}), rational "
              f"{counts['rational']['stored']} stored (line {counts['rational']['line']})")
        narrow = result["derived_counts"]["region_b_narrow"]
        print(f"narrow region B: chebyshev {narrow['chebyshev']['stored']} stored, "
              f"rational {narrow['rational']['stored']} stored")
        uniform = result["derived_counts"]["uniform"]
        print(f"uniform: chebyshev {uniform['chebyshev']['stored']} stored, "
              f"rational {uniform['rational']['stored']} stored")
        print(f"widths: coarse {result['widths']['coarse_region_b']:.15g}, "
              f"paper {result['widths']['paper_region_b']:.15g}")
        fp32c = result["derived_counts_fp32"]["region_b_coarse"]
        print(f"fp32 coarse region B: chebyshev {fp32c['chebyshev']['stored']} "
              f"stored (line {fp32c['chebyshev']['line']}), rational "
              f"{fp32c['rational']['stored']} stored "
              f"(line {fp32c['rational']['line']})")
        fp32n = result["derived_counts_fp32"]["region_b_narrow"]
        print(f"fp32 narrow region B: chebyshev {fp32n['chebyshev']['stored']} "
              f"stored, rational {fp32n['rational']['stored']} stored")
        fp32u = result["derived_counts_fp32"]["uniform"]
        print(f"fp32 uniform: chebyshev {fp32u['chebyshev']['stored']} stored, "
              f"rational {fp32u['rational']['stored']} stored")
        if result["bar_check"].get("ran"):
            print(f"bar check: 1e-13 F0(kX0) = {result['bar_check']['bar_1e13_f0']}, "
                  f"header {result['bar_check']['header_bar']}, "
                  f"{'ok' if result['bar_check']['ok'] else 'FAIL'}")
        if result["bar_check_fp32"].get("ran"):
            print(f"fp32 bar check: TOL_FLOAT "
                  f"{result['bar_check_fp32']['generator_TOL_FLOAT']}, "
                  f"F32_RAT_BOUND "
                  f"{result['bar_check_fp32']['generator_F32_RAT_BOUND']}, "
                  f"{'ok' if result['bar_check_fp32']['ok'] else 'FAIL'}")
        if result["lane_contract"].get("checked"):
            rows = result["lane_contract"]["rows"]
            print(f"lane-contract table: {rows}")
        return 0 if all(c["ok"] for c in result["checks"]) else 1

    if args.sweep:
        if not args.out_dir:
            parser.error("--sweep needs --out-dir")
        sweep_mode(args)
        return 0

    if args.plan:
        plan_mode(args)
        return 0

    if args.probe:
        probe_mode(args)
        return 0

    if args.paper_cell:
        paper_cell_mode(args)
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
