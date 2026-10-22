#!/usr/bin/env python3
"""The cold ladder for the one row the stability pass moved.

The pass is a single below-check per row.  When a row's verdict moves, its
full rational ladder is re-solved cold at the doubled settings, as a separate
follow-up with its own record.  This script is that follow-up for
``rel:1e-13|2^5``, the only row the pass moved.

For each stored count from the pass's count upward, every rational pair with
k <= 8 is solved on the row's 4,001-node grid at Remez 120 / Lawson 600, in
one batched worker pool.  The solves go through the pass's own per-pair cache,
so a pair already solved at these settings is reused, not recomputed.  The
stable count is the smallest stored whose pair meets the target; every pair at
the count below it must fail.  The record names the old count, the new count,
and both residuals of every pair.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import count_sweep as cs
import stability_save_verify as ss

PASS_RECORD = HERE / "results-stability" / "pass_2026-10-20.json"
CORE_RECORD = HERE / "results" / "sweep_rel_1e-13.json"
OUT = HERE / "results-stability" / "followup_cold_ladder_2026-10-20.json"
ROW_KEY = "rel:1e-13|2^5"
WIDTH_NAME = "2^5"
START_STORED = 16
MAX_STORED = 20


def core_row():
    data = json.loads(CORE_RECORD.read_text(encoding="utf-8"))
    for row in data["rows"]:
        if ss.row_key(row) == ROW_KEY:
            return row
    raise SystemExit(f"no core row {ROW_KEY}")


def pass_facts():
    data = json.loads(PASS_RECORD.read_text(encoding="utf-8"))
    verdict = data["verdicts"][ROW_KEY]
    accepted = [p for p in data["saved"][ROW_KEY] if p["role"] == "accepted"]
    if len(accepted) != 1:
        raise SystemExit("expected one accepted pair in the pass record")
    pair = accepted[0]
    residual = pair["doubled"][0]["residual"]["rel"]
    return {
        "verdict": verdict,
        "old_stored": pair["stored"],
        "old_m": pair["m"],
        "old_k": pair["k"],
        "old_residual_rel": residual,
    }


def ladder_jobs(row, stored, boys_root, already):
    """Every rational pair at one stored count, minus cached solves."""
    k_max = min(8, stored - 2)
    jobs = []
    for k in range(1, k_max + 1):
        m = stored - 1 - k
        role = ("accepted" if (stored == START_STORED
                              and (m, k) == (already["old_m"], already["old_k"]))
                else ("below" if stored == START_STORED else "ladder"))
        spec = {
            "row": ROW_KEY, "target": row["target"],
            "width_name": WIDTH_NAME, "a": row["a"], "b": row["b"],
            "nodes": ss.DOUBLED_NODES, "m": m, "k": k, "role": role,
            "outer": ss.DOUBLED_OUTER, "remez_iters": ss.DOUBLED_REMEZ,
            "doubled": True,
        }
        cached = ss.cache_path(spec).exists()
        spec["boys_root"] = boys_root
        jobs.append((spec, cached))
    return jobs


def batch(g, row, stored, boys_root, xs4, ref4, already):
    jobs = ladder_jobs(row, stored, boys_root, already)
    n_cached = sum(1 for _, cached in jobs if cached)
    started = time.perf_counter()
    results = g.run_jobs(ss.pair_job, [spec for spec, _ in jobs])
    seconds = time.perf_counter() - started
    pairs = []
    for (spec, cached), res in zip(jobs, results):
        evaluated = ss.evaluate_fits(g, row, res["fits"], xs4, ref4)
        pairs.append({
            "stored": stored, "m": spec["m"], "k": spec["k"],
            "role": spec["role"], "cached": cached,
            "solvers": evaluated,
            "meets": any(e["meets"] is True for e in evaluated),
        })
    return {"stored": stored, "seconds": round(seconds, 1),
            "jobs": len(jobs), "cached_jobs": n_cached, "pairs": pairs}


def main():
    row = core_row()
    facts = pass_facts()
    g = cs.load_generator(cs.DEFAULT_BOYS)
    boys_root = str(cs.DEFAULT_BOYS)
    ts, xs = cs.make_nodes(g, g.mpf(row["a"]), g.mpf(row["b"]),
                           ss.DOUBLED_NODES)
    ref = cs.reference(g, xs)

    record = {
        "schema": "cold-ladder-followup-1",
        "follow_up_to": str(PASS_RECORD.relative_to(HERE)),
        "reason": ("the pass moved this row: the accepted pair fails on the "
                   "4,001-node grid"),
        "row": ROW_KEY,
        "target": {"kind": row["target_kind"], "value": row["target_value"]},
        "grid_nodes": len(xs),
        "settings": {"nodes": ss.DOUBLED_NODES,
                     "remez_iters": ss.DOUBLED_REMEZ,
                     "outer": ss.DOUBLED_OUTER,
                     "dps": g.RAT_DPS, "k_max": 8},
        "pass": facts,
        "batches": [],
        "new_stored": None,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    started = time.perf_counter()
    lowest_meeting = None
    for stored in range(START_STORED, MAX_STORED + 1):
        out = batch(g, row, stored, boys_root, xs, ref, facts)
        record["batches"].append(out)
        meets = any(p["meets"] for p in out["pairs"])
        print(f"stored {stored}: {out['jobs']} pairs "
              f"({out['cached_jobs']} cached), {out['seconds']} s, "
              f"meets={meets}", flush=True)
        OUT.write_text(json.dumps(cs.json_ready(record), indent=2) + "\n",
                        encoding="utf-8")
        if meets:
            lowest_meeting = stored
            break

    record["new_stored"] = lowest_meeting
    record["total_seconds"] = round(time.perf_counter() - started, 1)
    OUT.write_text(json.dumps(cs.json_ready(record), indent=2) + "\n",
                    encoding="utf-8")
    print(json.dumps({"row": ROW_KEY, "new_stored": lowest_meeting,
                      "total_seconds": record["total_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
