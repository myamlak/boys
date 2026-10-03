#!/usr/bin/env python
"""The completion status: one closed arithmetic per option space, or the reason it does not close.

WHY THIS EXISTS. The owner has been given "nearly done" for days and each time a question found a
dimension that was never in the answer. The mechanism was not carelessness: a status composed from
memory is a summary of the spaces I happen to be thinking about, and a space nobody has touched is
absent from that summary rather than visible in it as a hole. On 2026-10-02 the host space closed
exactly - 504 + 0 + 168 + 0 + 0 = 672 - while the device's classes were in no arithmetic at all,
and every GPU question asked that day was a gap in the enumeration, not in my prose.

So the status is not written. It is printed, by this, over EVERY space, and a message that claims
anything about completion quotes it.

THE CONTRACT, per space:
  * every space the library has is a row, whether or not a number exists for it;
  * a space whose parts do not sum to its total is printed NOT CLOSED and exits nonzero;
  * a space with no arithmetic at all is printed ABSENT - never omitted, because a space missing
    from a status is indistinguishable from a space that is complete;
  * the revision each figure was taken at is printed beside it, so a stale figure is visible as
    one rather than inherited as current.

Failures print and exit nonzero; the tool never reports success it did not read.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The four states and no fifth. A member is served, refused with the library's own reason and owed,
# not runnable on this host, or unaccounted - and unaccounted is the one that must be zero.
STATES = ("certified/served", "refused and owed", "not runnable on this host", "unaccounted")

RUN = os.path.join(REPO, "tests", "data", "boys_accuracy_gate_run.txt")

# The gate's coverage block, as the gate prints it.
COMBINATIONS = re.compile(
    r"COMBINATIONS:\s*(\d+)\s+of\s+(\d+)\s+member\(s\) of the option space")
REFUSED = re.compile(r"^\s*(\d+)\s+refused with the library's own reason and owed", re.M)
NOT_RUNNABLE = re.compile(
    r"^\s*(\d+)\s+not runnable on this host, counted apart and not against the library", re.M)
UNCOVERED = re.compile(r"^\s*(\d+)\s+offered and covered by no cell of this block", re.M)
ARITHMETIC = re.compile(r"the arithmetic:\s*([\d\s+]+)=\s*(\d+)")
REVISION = re.compile(r"accuracy gate,\s*revision\s+([0-9a-f]{7,40})")


def head_revision() -> str:
    try:
        done = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=20)
    except Exception:
        return ""
    return done.stdout.strip() if done.returncode == 0 else ""


def read(path: str) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()
    except OSError:
        return ""


def host_space() -> tuple[str, list[str], bool]:
    """The host option space, from the recorded gate run. Returns (summary, lines, closed)."""
    text = read(RUN)
    if not text:
        return ("ABSENT: no recorded gate run at tests/data/boys_accuracy_gate_run.txt",
                [], False)

    match = COMBINATIONS.search(text)
    if not match:
        return ("ABSENT: the recorded run carries no COMBINATIONS block", [], False)

    served, total = int(match.group(1)), int(match.group(2))
    refused = int((REFUSED.search(text) or [0, 0])[1])
    not_runnable = int((NOT_RUNNABLE.search(text) or [0, 0])[1])
    unaccounted = int((UNCOVERED.search(text) or [0, 0])[1])

    revision = (REVISION.search(text) or [None, "unknown"])[1]
    head = head_revision()
    stale = "" if not head or head.startswith(revision) or revision.startswith(head) else \
            f"   STALE: this run is {revision[:9]}, HEAD is {head[:9]}"

    lines = [
        f"  served                 {served:>6}",
        f"  refused and owed       {refused:>6}",
        f"  not runnable here      {not_runnable:>6}",
        f"  unaccounted            {unaccounted:>6}",
        f"  ----------------------------------",
        f"  total                  {total:>6}",
    ]

    # WHY EVERY NON-SERVED ROW IS NON-SERVED, taken from the run itself. A count with no reason
    # beside it is a number this tool would be laundering: on 2026-10-02 the run booked 168 rows
    # "not runnable on this host" on a machine that has the device and a CUDA build, because the
    # classifier decides by lane identity. Printing the reason is what makes a false one visible
    # here rather than inherited from the run.
    # Only the combination rows: a line whose first field is a lane the cross enumerates. Every
    # other table in the run prints its own trailing column, and counting those as "reasons" buries
    # the one that matters under noise.
    LANE = r"(?:fp64|fp32|fp16|fp32-device)"
    reasons: dict[str, int] = {}
    for row in re.findall(rf"^\s{{2,}}({LANE},.*?)\s{{2,}}([a-z][a-z0-9 -]+?)\s*$", text, re.M):
        state = row[1].strip()
        if state.startswith("certified"):
            continue
        reasons[state] = reasons.get(state, 0) + 1
    if reasons:
        lines.append("  the run states these reasons for the rows it did not serve:")
        for state, count in sorted(reasons.items(), key=lambda kv: -kv[1])[:6]:
            lines.append(f"    {count:>6}  {state[:96]}")
    else:
        lines.append("  the run records NO reason for any non-served row: whatever put those rows "
                     "in their states cannot be checked from this artifact")

    found = ARITHMETIC.search(text)
    if found:
        parts = [int(n) for n in re.findall(r"\d+", found.group(1))]
        summed = sum(parts)
        agrees = summed == total and parts[0] == served
        lines.append(
            f"  the run's own arithmetic {found.group(1).strip()} = {found.group(2)}"
            f"   {'agrees' if agrees else 'DISAGREES with the counts above'}")
        closed = agrees and unaccounted == 0
    else:
        lines.append("  the run prints no arithmetic line")
        closed = False

    if stale:
        lines.append(stale)
        closed = False

    summary = f"{served} of {total} served, {unaccounted} unaccounted"
    return (summary, lines, closed)


def device_space() -> tuple[str, list[str], bool]:
    """The device option space, from the probe report that carries its closure arithmetic.

    The probe prints the closure as its last block: the space's total read off the library's own
    tables (`BoysDeviceOptions().size()` x `kDeviceRungs.size()`), the states each member can be in,
    the arithmetic summing them, and a verdict. A log written before that block existed carries no
    closure at all - which is a fact about the log, not about the library - so the search is for the
    block, and the absence of it is reported as the absence of a RUN rather than of a space.
    """
    directory = os.path.join(REPO, ".claude", "tmp")
    try:
        names = [n for n in os.listdir(directory)
                 if n.endswith(".log") and ("probe" in n.lower())]
    except OSError:
        names = []
    if not names:
        return ("ABSENT: no device probe log under .claude/tmp/", [], False)

    carrying = [os.path.join(directory, n) for n in names]
    carrying = [p for p in carrying if "the arithmetic:" in read(p)]
    if not carrying:
        return (f"ABSENT: none of the {len(names)} probe log(s) on disk carries the closure block",
                [f"  the probe prints a closure (MEMBERS / the arithmetic / the verdict);",
                 f"  no log under .claude/tmp/ was written by a build that had it, so the device",
                 f"  space cannot be closed from what is on disk. Re-run the probe.",
                 f"  logs present: {len(names)}"], False)

    newest = max(carrying, key=os.path.getmtime)
    text = read(newest)

    members = re.search(r"MEMBERS:\s*(\d+)\s+of\s+(\d+)\s+member\(s\)", text)
    arith = re.search(r"the arithmetic:\s*([\d\s+]+)=\s*(\d+)", text)
    total = re.search(r"the space's own total:\s*(\d+)\s+member\(s\)\s*=\s*the\s*(\d+)\s+above\s*\+"
                      r"\s*(\d+)\s+in no state", text)
    verdict = re.search(r"the verdict:\s*(PASS|FAIL)", text)

    if not (members and arith and total and verdict):
        return ("ABSENT: the closure block is present but not in the shape this reads",
                [f"  source {os.path.basename(newest)}",
                 "  found: MEMBERS" if members else "  MISSING: MEMBERS",
                 "  found: the arithmetic" if arith else "  MISSING: the arithmetic",
                 "  found: the space's own total" if total else "  MISSING: the space's own total",
                 "  found: the verdict" if verdict else "  MISSING: the verdict"], False)

    measured, space = int(members.group(1)), int(members.group(2))
    parts = [int(n) for n in re.findall(r"\d+", arith.group(1))]
    unaccounted = int(total.group(3))
    closed = (sum(parts) == int(arith.group(2)) == int(total.group(1))
              and unaccounted == 0 and verdict.group(1) == "PASS")

    # The labels are READ FROM THE RUN, not written here. The probe's arithmetic has as many terms
    # as the states that exist, and that number is not fixed: removing the accuracy rung took the
    # rung state out and the arithmetic went from six terms to five. A reader that hard-codes the
    # count breaks on the next state the library gains or loses - which is what this one did.
    labels = [state.strip() for _, state in re.findall(
        r"^ {10,}(\d+)\s+([A-Za-z].{6,70})$", text, re.M)
        if not state.strip().startswith(("member(s", "cell(s"))]
    if len(labels) != len(parts):
        labels = [f"state {i + 1}" for i in range(len(parts))]

    lines = [f"  source                 {os.path.basename(newest)}"]
    for label, value in zip(labels, parts):
        lines.append(f"  {label[:22]:<22} {value:>6}")
    lines += [
        f"  unaccounted            {unaccounted:>6}",
        f"  ----------------------------------",
        f"  total                  {space:>6}",
        f"  the probe's own arithmetic {arith.group(1).strip()} = {arith.group(2)}",
        f"  the verdict as printed: {'PASS' if verdict.group(1) == 'PASS' else 'FAIL'}",
    ]
    return (f"{measured} of {space} measured, {unaccounted} in no state", lines, closed)


def main() -> int:
    print("COMPLETION STATUS - one arithmetic per option space")
    print(f"HEAD {head_revision()[:12] or 'unknown'}")
    print()

    overall = True
    for name, fn in (("HOST   (four lanes x route x scheme x partition x packing)", host_space),
                     ("DEVICE (classes over precision x shape)", device_space)):
        summary, lines, closed = fn()
        print(f"{name}")
        print(f"  {summary}")
        for line in lines:
            print(line)
        print(f"  => {'CLOSED' if closed else 'NOT CLOSED'}")
        print()
        overall = overall and closed

    print("SPACES THIS REPORT DOES NOT COVER")
    for space in ("the 32-cell entry book (accounted separately by the gate)",
                  "any space a build configuration other than the recorded one would have"):
        print(f"  - {space}")
    print()
    print("VERDICT:", "every space above closes" if overall else
          "at least one space does not close - the counts above are the status, not a summary of it")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
