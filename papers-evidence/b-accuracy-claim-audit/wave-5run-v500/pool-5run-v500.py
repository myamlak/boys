#!/usr/bin/env python3
"""Pool the v5.0.0 repeated five-run wave: per-repetition aggregates, then
their mean and spread across repetitions, with the window's own canary
quoted beside every ratio.

Usage: python3 pool-5run-v500.py [reps] [repeat-dir]

The pooling logic is pool-5run.py unchanged; only the tree it describes and
the canary reading are new. Each repetition is one full pass of the same
instruments, so the honest average is over per-repetition aggregates (each
item-weighted within itself), reported with min/max. The whole-stream ratio
uses each stream's median Mvalues/s: ratio = libint2_rate / qcx_rate (below
1.0 = qcx faster in time).

The canary (far-kernel mode a vs a2) is read once before the wave, once
before every repetition, and once after the last. Its deltas are quoted in
the header, in full, and beside every pooled ratio as the pre/post triples
(n = 0,1,2) and the window's largest absolute delta. A window whose largest
absolute delta reaches WIDE_PERCENT is printed PROVISIONAL, and no figure
from it may be used.
"""

import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "repeat"
FIELD = re.compile(r"([A-Za-z_][A-Za-z0-9_]*):\s*([^|]+)")
REPS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
WIDE_PERCENT = 10.0


def cells(path):
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("cell:") or "ratio_median" not in line:
            continue
        f = {k: v.strip() for k, v in FIELD.findall(line)}
        out.append((f.get("x", "?"), int(f["n"]), f.get("band", "?"), int(f["count"]),
                    float(f["qcx_median_ns"]), float(f["libint2_median_ns"]), float(f["ratio_median"])))
    return out


def weighted(rows):
    return sum(r[4] * r[3] for r in rows) / sum(r[5] * r[3] for r in rows)


def rates(path):
    q = l = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("kernel:"):
            continue
        f = {k: v.strip() for k, v in FIELD.findall(line)}
        if f.get("workload") is None:
            continue
        if line.split("|")[0].strip().endswith("qcx-batch"):
            q = float(f["median_Mvals_per_s"])
        elif "libint2" in line.split("|")[0]:
            l = float(f["median_Mvals_per_s"])
    return q, l


def read_canary(path):
    """[(tag, [delta, delta, delta])] from canary.log, refusing a missing or
    unparsable reading: the window must be quotable or the ratio is not read."""
    if not path.is_file():
        sys.exit(f"canary log missing: {path} - refusing to pool an unquoted window")
    out = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        tag, deltas = parts[1], parts[2:]
        try:
            vals = [float(d) for d in deltas]
        except ValueError:
            continue
        out.append((tag, vals))
    if not out:
        sys.exit(f"canary log has no readable readings: {path}")
    if not any(tag == "wave-before" for tag, _ in out) or not any(tag == "wave-after" for tag, _ in out):
        sys.exit(f"canary log lacks the wave-before/wave-after brackets: {path}")
    return out


CANARY = read_canary(REPDIR / "canary.log")
CANARY_PRE = next(v for t, v in CANARY if t == "wave-before")
CANARY_POST = next(v for t, v in CANARY if t == "wave-after")
CANARY_MAX = max(abs(x) for _, vals in CANARY for x in vals)
PROVISIONAL = CANARY_MAX >= WIDE_PERCENT
TRIPLE = lambda vals: "/".join(f"{x:+.2f}" for x in vals)
CANARY_BESIDE = (f"canary pre {TRIPLE(CANARY_PRE)} %  post {TRIPLE(CANARY_POST)} %  "
                 f"max|delta| {CANARY_MAX:.2f} %" + ("  PROVISIONAL" if PROVISIONAL else ""))


def stat(name, values):
    m = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    print(f"  {name:<34} mean {m:.3f}   min {min(values):.3f}   max {max(values):.3f}   "
          f"sd {sd:.4f}   ({len(values)} reps)")
    print(f"      {CANARY_BESIDE}")


print(f"pooled over {REPS} repetitions (ratio qcx/libint2; below 1.0 = qcx faster):")
print(f"canary readings ({len(CANARY)}; tag and deltas in n = 0,1,2 order):")
for tag, vals in CANARY:
    print(f"  {tag:<16} {TRIPLE(vals)} %")
print(f"window canary: pre {TRIPLE(CANARY_PRE)} %  post {TRIPLE(CANARY_POST)} %  "
      f"max|delta| {CANARY_MAX:.2f} %" + ("  PROVISIONAL: no figure ships" if PROVISIONAL else ""))

per_rep_region = []
per_rep_A = []
per_rep_B = []
per_rep_C = []
per_rep_order_u = []
per_rep_order_m = []
per_rep_strat = []
per_rep_uniform = []
per_rep_molecular = []

for r in range(1, REPS + 1):
    region = cells(REPDIR / f"rep{r}-region.txt")
    per_rep_region.append(weighted(region))
    prim = [c for c in region if c[2] in ("x0", "A0", "A1", "B", "C1", "C2")]
    per_rep_A.append(weighted([c for c in prim if c[2] in ("A0", "A1")]))
    per_rep_B.append(weighted([c for c in prim if c[2] == "B"]))
    per_rep_C.append(weighted([c for c in prim if c[2] in ("C1", "C2")]))

    order = cells(REPDIR / f"rep{r}-order.txt")
    per_rep_order_u.append(weighted([c for c in order if c[0] == "uniform-x-marginal"]))
    per_rep_order_m.append(weighted([c for c in order if c[0] == "molecular-x-marginal"]))

    strat = cells(REPDIR / f"rep{r}-stratify.txt")
    per_rep_strat.append(weighted([c for c in strat if c[0] == "molecular" and c[1] >= 0]))

    for stream, bucket in (("uniform", per_rep_uniform), ("molecular", per_rep_molecular)):
        q, l = rates(REPDIR / f"rep{r}-whole-{stream}.txt")
        bucket.append(l / q)

stat("region-sweep, all rows", per_rep_region)
stat("region A", per_rep_A)
stat("region B", per_rep_B)
stat("region C", per_rep_C)
stat("order-sweep uniform marginal", per_rep_order_u)
stat("order-sweep molecular marginal", per_rep_order_m)
stat("stratify molecular", per_rep_strat)
stat("whole uniform stream", per_rep_uniform)
stat("whole molecular stream", per_rep_molecular)
