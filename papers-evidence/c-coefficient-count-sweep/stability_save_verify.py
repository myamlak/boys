#!/usr/bin/env python3
"""The stability pass as one batched boundary check.

Protocol (SWEEP-PLAN.md section 2.5, amended 2026-10-20 before any stability
row ran):

  batch    One flat job list over the measured core rows: for each row, its
           accepted rational pair and every candidate pair at stored - 1.
           Every pair is solved once at its row's own settings (2,001 nodes,
           Remez 60, Lawson 300) in one worker pool via the generator's
           run_jobs. This is a single below-check per row, not a scan.

  grid     Every saved fit is evaluated on the row's 4,001-node grid. The even
           nodes are the row's own 2,001-node grid, so their reference values
           are reused; only the odd midpoints are computed. Evaluation is in
           binary64 through both multiply-add routes.

  iterate  A doubled-settings solve (4,001 nodes, Remez 120, Lawson 600) is run
           only for a pair whose delivered residual is within a factor of 2 of
           the target in either family: the verdict could move there. These
           are the only 4x solves in the pass.

  verdict  A row is stable when every accepted fit still meets on the 4,001
           node grid and every fit at stored - 1 still fails. If a verdict
           moves, the row's full ladder is re-solved cold at the doubled
           settings; the old and new counts are both recorded. A row that
           still moves is a documented correction, never a silent edit.

Chebyshev is cheap: its full doubled-grid degree ladder runs for every row.

The script is implemented in two modes: --plan prints the job list, the
recorded margin flags and the projection without solving; --run executes the
batch and writes results-stability/save_verify_batch.json. The worker job is a
module-level function so the spawn pool can import it, and every row's grid is
rebuilt inside the worker from decimal strings at 60 digits, the same guard
the sweep uses against truncated mpf pickles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import count_sweep as cs  # noqa: E402

DEFAULT_SOURCES = (
    "results/sweep_rel_1e-13.json",
    "results/sweep_abs_5e-14.json",
)
DOUBLED_NODES = 4000
DOUBLED_REMEZ = 120
DOUBLED_OUTER = 600

# A pair solve is the expensive unit; a failed later phase must not force the
# whole batch to be solved again. Every worker writes its result to a file
# keyed by the full spec, and a rerun reuses it.
CACHE_DIR = HERE / "results-stability" / "cache"


def cache_path(spec):
    key = "|".join(str(spec[k]) for k in
                   ("row", "role", "m", "k", "nodes", "outer", "remez_iters",
                    "doubled"))
    return CACHE_DIR / (hashlib.sha1(key.encode()).hexdigest() + ".json")

# Per-worker row cache: the reference at 60 digits is the expensive part of a
# row, and a row's pairs usually land on more than one worker.
_ROW_CACHE = {}


def row_key(row):
    return f"{row['target']}|{row['width_name']}"


def load_core_rows(sources):
    rows = []
    for path in sources:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row.get("cheb", {}).get("stored") is None:
                continue
            if row.get("rat", {}).get("stored") is None:
                continue
            rows.append(row)
    return rows


def boundary_pairs(row):
    """The accepted pair and every pair at stored - 1."""
    rat = row["rat"]
    stored = rat["stored"]
    pairs = [(rat["m"], rat["k"], "accepted")]
    below_total = stored - 1
    k_max = min(8, below_total - 2)
    for k in range(1, k_max + 1):
        pairs.append((below_total - 1 - k, k, "below"))
    return pairs


def pair_jobs(rows, doubled=False):
    """One flat list of pair solves over every row."""
    jobs = []
    for row in rows:
        nodes = DOUBLED_NODES if doubled else row["settings"]["nodes"] - 1
        outer = DOUBLED_OUTER if doubled else row["settings"]["refine_outer"]
        remez = DOUBLED_REMEZ if doubled else row["settings"]["remez_iters"]
        for m, k, role in boundary_pairs(row):
            if doubled and role == "accepted":
                continue  # a thin accepted pair is added by its own flag
            jobs.append({
                "row": row_key(row), "target": row["target"],
                "width_name": row["width_name"], "a": row["a"], "b": row["b"],
                "nodes": nodes, "m": m, "k": k, "role": role,
                "outer": outer, "remez_iters": remez, "doubled": doubled,
            })
    return jobs


def pair_job(spec):
    """Worker: solve one pair with both solvers and return decimal strings."""
    path = cache_path(spec)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    g = cs._worker_generator(spec["boys_root"])
    key = (spec["a"], spec["b"], spec["nodes"])
    if key not in _ROW_CACHE:
        a = g.mpf(spec["a"])
        b = g.mpf(spec["b"])
        ts, xs = cs.make_nodes(g, a, b, spec["nodes"])
        ref = cs.reference(g, xs)
        _ROW_CACHE[key] = (ts, ref)
    ts, ref = _ROW_CACHE[key]
    indices = list(range(len(ts)))
    cands = cs.rational_candidates(g, spec["m"], spec["k"], indices, ts, ref,
                                   spec["outer"], spec["remez_iters"])
    out = []
    for name, p, q in cands:
        out.append({"solver": name,
                    "p": [g.mp.nstr(c, 62) for c in p],
                    "q": [g.mp.nstr(c, 62) for c in q]})
    result = {"row": spec["row"], "role": spec["role"], "m": spec["m"],
              "k": spec["k"], "doubled": spec["doubled"], "fits": out}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, path)
    return result


def attach_boys(jobs, boys_root):
    return [dict(job, boys_root=boys_root) for job in jobs]


def row_grid(g, row, nodes):
    a = g.mpf(row["a"])
    b = g.mpf(row["b"])
    ts, xs = cs.make_nodes(g, a, b, nodes)
    ref = cs.reference(g, xs)
    return ts, xs, ref


def doubled_grid(g, row, ts_default, xs_default, ref_default):
    """4,001 nodes: the row's 2,001 even nodes plus 2,000 odd midpoints."""
    a = g.mpf(row["a"])
    b = g.mpf(row["b"])
    n = len(xs_default) - 1              # 2,000 intervals
    odd_xs = []
    for i in range(n):
        t = 2 * (i + g.mpf("0.5")) / n - 1
        odd_xs.append(a + (b - a) * (t + 1) / 2)
    odd_ref = cs.reference(g, odd_xs)
    xs4 = []
    ref4 = []
    for i in range(n):
        xs4.append(xs_default[i])
        ref4.append(ref_default[i])
        xs4.append(odd_xs[i])
        ref4.append(odd_ref[i])
    xs4.append(xs_default[n])
    ref4.append(ref_default[n])
    ts4, _ = cs.make_nodes(g, a, b, 2 * n)
    return ts4, xs4, ref4


def evaluate_fits(g, row, fits, xs, ref):
    """Delivered residual of every solver candidate on a grid."""
    out = []
    for fit in fits:
        p = [g.mpf(s) for s in fit["p"]]
        q = [g.mpf(s) for s in fit["q"]]
        err = cs.rat_errors(g, p, q, g.mpf(row["a"]), g.mpf(row["b"]), xs, ref)
        if err is None:
            out.append({"solver": fit["solver"], "meets": None,
                        "residual": None})
            continue
        wr, wa = err[0], err[1]
        out.append({"solver": fit["solver"], "meets": meets(row, wr, wa),
                    "residual": {"rel": g.mp.nstr(wr, 17),
                                 "abs": g.mp.nstr(wa, 17)}})
    return out


def target_of(row):
    return {"kind": row["target_kind"], "value": row["target_value"]}


def meets(row, wr, wa):
    return (wr <= row["target_value"] if row["target_kind"] == "rel"
            else wa <= row["target_value"])


def margin(row, wr, wa):
    r = float(wa if row["target_kind"] == "abs" else wr)
    return row["target_value"] / r if r <= row["target_value"] else r / row["target_value"]


def thin_from_recorded(rows):
    """Rows whose accepted fit sits within a factor of 2 of the target."""
    flags = []
    for row in rows:
        for fam in ("cheb", "rat"):
            entry = row[fam]
            if entry.get("stored") is None:
                continue
            rel = float(entry["delivered_rel"])
            abs_ = float(entry["delivered_abs"])
            m = margin(row, rel, abs_)
            if m < 2:
                flags.append({"row": row_key(row), "family": fam,
                              "stored": entry["stored"], "margin": m})
    return flags


def plan(rows, sources):
    jobs = pair_jobs(rows, doubled=False)
    flags = thin_from_recorded(rows)
    pairs = len(jobs)
    return {
        "sources": [str(s) for s in sources],
        "core_rows": len(rows),
        "rows_by_target": {
            t: sum(1 for r in rows if r["target"] == t)
            for t in sorted({r["target"] for r in rows})
        },
        "default_pair_jobs": pairs,
        "solver_runs": 2 * pairs,
        "recorded_margin_flags": flags,
        "flagged_rows": len({f["row"] for f in flags}),
        "projection": {
            "measured_pair_seconds_2^0": 76,
            "measured_by": ("coordinator, rel:1e-13 width 2^0, four boundary "
                            "solves serial, 303 s"),
            "uniform_12_worker_minutes": round(pairs * 76 / 12 / 60, 1),
            "stated_allowance_minutes": "20-25 (wide rows' pairs cost more)",
        },
    }


def run(rows, sources, out):
    started = time.perf_counter()
    g = cs.load_generator(cs.DEFAULT_BOYS)
    boys_root = str(cs.DEFAULT_BOYS)

    # One row grid and reference per row, built once in the parent.
    grids = {}
    for row in rows:
        grids[row_key(row)] = row_grid(g, row, row["settings"]["nodes"] - 1)

    jobs = attach_boys(pair_jobs(rows, doubled=False), boys_root)
    t0 = time.perf_counter()
    results = g.run_jobs(pair_job, jobs)
    batch_seconds = time.perf_counter() - t0

    saved = {}
    for job, res in zip(jobs, results):
        ts, xs, ref = grids[job["row"]]
        saved.setdefault(job["row"], []).append({
            "role": job["role"], "m": job["m"], "k": job["k"],
            "stored": (job["m"] + 1) + job["k"],
            "fits": res["fits"],
            "default": evaluate_fits(g, next(r for r in rows
                                             if row_key(r) == job["row"]),
                                     res["fits"], xs, ref),
        })

    # The doubled grid: reuse the row's even reference values, compute the odds.
    doubled = {}
    grid_seconds = 0.0
    for row in rows:
        key = row_key(row)
        ts, xs, ref = grids[key]
        t_row = time.perf_counter()
        ts4, xs4, ref4 = doubled_grid(g, row, ts, xs, ref)
        grid_seconds += time.perf_counter() - t_row
        doubled[key] = {"xs": xs4, "ref": ref4}

    # Evaluate every saved fit directly on the doubled grid.
    t0 = time.perf_counter()
    for row in rows:
        key = row_key(row)
        xs4, ref4 = doubled[key]["xs"], doubled[key]["ref"]
        for pair in saved.get(key, []):
            pair["doubled"] = evaluate_fits(g, row, pair["fits"], xs4, ref4)
    eval_seconds = time.perf_counter() - t0

    # Chebyshev full doubled ladder for every row.
    cheb_checks = {}
    for row in rows:
        key = row_key(row)
        xs4, ref4 = doubled[key]["xs"], doubled[key]["ref"]
        cheb_checks[key] = {
            "reported": row["cheb"]["stored"],
            "doubled": cs.cheb_count(g, g.mpf(row["a"]), g.mpf(row["b"]),
                                     xs4, ref4, target_of(row),
                                     max_deg=min(row["cheb"]["stored"] + 2,
                                                 cs.CHEB_MAX_DEG)),
        }

    # Thin pairs: the verdict could move within a factor of 2 of the target.
    thin = []
    for row in rows:
        key = row_key(row)
        for pair in saved.get(key, []):
            thin_here = any(
                e["residual"] is not None
                and margin(row, g.mpf(e["residual"]["rel"]),
                           g.mpf(e["residual"]["abs"])) < 2
                for e in pair.get("doubled", []) + pair.get("default", []))
            if thin_here:
                thin.append(pair | {"row": key, "target": row["target"]})

    # Doubled-settings solves for the thin pairs, in the same pool shape.
    thin_jobs = []
    for pair in thin:
        row = next(r for r in rows if row_key(r) == pair["row"])
        thin_jobs.append({
            "row": pair["row"], "target": row["target"],
            "width_name": row["width_name"], "a": row["a"], "b": row["b"],
            "nodes": DOUBLED_NODES, "m": pair["m"], "k": pair["k"],
            "role": pair["role"], "outer": DOUBLED_OUTER,
            "remez_iters": DOUBLED_REMEZ, "doubled": True,
            "boys_root": boys_root,
        })
    thin_results = g.run_jobs(pair_job, thin_jobs) if thin_jobs else []
    thin_eval = []
    for job, res in zip(thin_jobs, thin_results):
        xs4, ref4 = doubled[job["row"]]["xs"], doubled[job["row"]]["ref"]
        row = next(r for r in rows if row_key(r) == job["row"])
        thin_eval.append({
            "row": job["row"], "role": job["role"], "m": job["m"],
            "k": job["k"],
            "fits": evaluate_fits(g, row, res["fits"], xs4, ref4),
        })

    # Verdict per row: accepted meets, every below fails, Chebyshev unmoved.
    verdicts = {}
    for row in rows:
        key = row_key(row)
        accepted = [p for p in saved.get(key, []) if p["role"] == "accepted"]
        below = [p for p in saved.get(key, []) if p["role"] == "below"]
        acc_meets = [e["meets"] for p in accepted for e in p["doubled"]]
        bel_meets = [e["meets"] for p in below for e in p["doubled"]]
        for tv in thin_eval:
            if tv["row"] != key:
                continue
            target_list = acc_meets if tv["role"] == "accepted" else bel_meets
            target_list.extend(e["meets"] for e in tv["fits"])
        cheb = cheb_checks[key]
        cheb_stable = cheb["doubled"].get("stored") == cheb["reported"]
        stable = (any(m is True for m in acc_meets)
                  and not any(m is True for m in bel_meets)
                  and cheb_stable)
        verdicts[key] = {
            "accepted_meets": acc_meets,
            "below_meets": bel_meets,
            "cheb_reported": cheb["reported"],
            "cheb_doubled": cheb["doubled"].get("stored"),
            "cheb_stable": cheb_stable,
            "stable": bool(stable),
            "cold_ladder_needed": not bool(stable),
        }

    record = {
        "schema": "stability-save-verify-1",
        "amended": "2026-10-20, before any stability row ran",
        "sources": [str(s) for s in sources],
        "rows": [row_key(r) for r in rows],
        "batch_seconds": round(batch_seconds, 1),
        "grid_seconds": round(grid_seconds, 1),
        "evaluate_seconds": round(eval_seconds, 1),
        "default_pair_jobs": len(jobs),
        "thin_pairs": len(thin),
        "thin_evaluated": thin_eval,
        "saved": saved,
        "cheb_checks": cheb_checks,
        "verdicts": verdicts,
        "total_seconds": round(time.perf_counter() - started, 1),
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(cs.json_ready(record), indent=2) + "\n",
                         encoding="utf-8")
    print(json.dumps({k: record[k] for k in
                      ("rows", "batch_seconds", "grid_seconds",
                       "evaluate_seconds", "default_pair_jobs", "thin_pairs",
                       "total_seconds")}, indent=2))
    print(json.dumps(cs.json_ready(
        {k: v for k, v in verdicts.items() if not v["stable"]}), indent=2))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", nargs="*", default=list(DEFAULT_SOURCES))
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--only", default=None,
                    help="one row key, for a smoke test; not part of the pass")
    ap.add_argument("--out", default="results-stability/save_verify_batch.json")
    args = ap.parse_args(argv)
    rows = load_core_rows(args.source)
    if args.only:
        rows = [r for r in rows if row_key(r) == args.only]
        if not rows:
            raise SystemExit(f"no such row: {args.only}")
    if args.plan:
        print(json.dumps(plan(rows, args.source), indent=2))
        return 0
    return run(rows, args.source, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
