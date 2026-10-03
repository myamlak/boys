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
LANES = re.compile(r"over\s+(\d+)\s+lane\(s\)")

# An arithmetic with NUMBERS in it. The words alone are not evidence: `src/boys_probe.cpp` contains
# the literal format strings ("MEMBERS:", "the arithmetic: %s = %zu") because it is the code that
# prints them, so a phrase test reads the source as a run. It did exactly that.
ARITHMETIC_WITH_NUMBERS = re.compile(r"the arithmetic:\s*\d+(?:\s*\+\s*\d+)+\s*=\s*\d+")


def has_closure(text: str) -> bool:
    """A whole closure block: four markers and an arithmetic that actually carries numbers."""
    return (bool(ARITHMETIC_WITH_NUMBERS.search(text))
            and "MEMBERS:" in text and "the space's own total:" in text
            and "the verdict:" in text)


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
    carrying = []
    for directory in (os.path.join(REPO, ".claude", "tmp"),
                      os.path.join(REPO, ".claude", "lane-status")):
        # os.walk, not listdir: a lane files its transcripts under
        # .claude/lane-status/<lane>/, so the closures are one level deeper than the directory the
        # first version of this searched. It reported the space ABSENT while the closure was on disk.
        for root, _dirs, names in os.walk(directory):
            for name in names:
                path = os.path.join(root, name)
                text = read(path)
                # The WHOLE closure block, not the phrase. A lane's write-up quotes "the arithmetic:"
                # while discussing a run, and reading that as a closure is exactly the wrong read this
                # tool exists to make impossible: it printed a bookkeeping file as a measurement and
                # reported the block "present but not in the shape this reads".
                if has_closure(text) and "BoysDeviceOptions" in text:
                    carrying.append(path)

    if not carrying:
        return ("ABSENT: no file under .claude/tmp or .claude/lane-status carries a whole closure",
                ["  the probe prints a closure (MEMBERS / the arithmetic / the space's own total /",
                 "  the verdict) and no file on disk carries all four, so the device space cannot",
                 "  be closed from what is here. Re-run the probe."], False)

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
    # The closure carries NO revision, so "newest file on disk" is not the same as "current space".
    # This tool reported CLOSED on a closure reading 87 members while the library's space was 261,
    # because that file was the most recently written. The space sentence is printed verbatim so the
    # reader can see which space the arithmetic closes over; the count alone would hide it.
    space_line = re.search(r"^\s+the space:\s*(.+?)$", text, re.M)
    if space_line:
        lines.append(f"  the space it closes    {' '.join(space_line.group(1).split())[:150]}")
    else:
        lines.append("  the space it closes    (the closure states no space sentence)")
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


def host_label() -> str:
    """The host space's label, with its lane count READ FROM THE RUN rather than written here.

    This said "four lanes" until the tree grew to six, at which point the label became a false
    statement about the very artifact it summarises. A count written into a status is a count that
    goes stale in silence, which is the failure this whole tool exists to make impossible.
    """
    found = LANES.search(read(RUN))
    lanes = found.group(1) if found else "an unread number of"
    return f"HOST   ({lanes} lanes x route x scheme x partition x packing)"


def probe_space() -> tuple[str, list[str], bool]:
    """The host probe's own cell space - the THIRD arithmetic, reported as a hole, not omitted.

    The probe prints a closure of its own (measured + offered-no-figure + not-asked + unoffered +
    not-carried + device-not-run + refused). It is a different space from the gate's: the gate's
    arithmetic is the option space's served/refused count, while the probe's is the cell cross it
    walks. On 2026-10-03 those read `96 of 96 served` and `288 + 0 + 0 + 0 + 0 + 72 + 0 = 360` - two
    arithmetics over the same library, neither derivable from the other.

    It is reported ABSENT rather than read, and the reason is measured rather than assumed: the
    closures on disk under the directory this tool searches were listed on 2026-10-03 and they carry
    the gate's and the device probe's arithmetic only (`72 + 0 + 24 + 0 + 0 = 96`,
    `87 + 0 + 0 + 0 + 0 = 87`, and rung-era ones at 672 and 1044). The probe that prints the 360 ran
    in a pinned worktree and its closure reached no log here, so there is nothing to read.

    Reading it is owed work, and it is not done here because **a closure read the wrong way is worse
    than one not read** - the point of this tool is that a wrong arithmetic is never printed as a
    right one. The tool's own contract is that a space with no arithmetic is printed ABSENT and never
    omitted, because a space missing from a status is indistinguishable from a space that is complete.
    """
    # NOT READ, and the reason is measured rather than assumed. This tool was extended to read this
    # space and it read the WRONG one: among the closures on disk that are not the device probe's
    # there are at least three different spaces - the host probe's own cell cross (360), the option
    # book's axes cross (720), and the region-B axis book (720) - and nothing in a closure says which
    # space its arithmetic is over. It picked the newest, which was the option book's, and printed
    # 720 members as this space's answer.
    #
    # "A closure read the wrong way is worse than one not read" is this tool's own contract, so the
    # space goes back to being a hole until a reader can tell the three apart. What that needs is a
    # stamp: the probe's closure should name the space it closes over, the way its arithmetic names
    # its own terms.
    return ("ABSENT: the closures on disk carry three different spaces and none says which it is",
            ["  the probe prints an arithmetic of its own, over a space that is not the gate's;",
             "  files on disk carry closures for the host probe's cell cross, the option book and",
             "  the region-B book, and a closure does not state which space its arithmetic closes",
             "  over. Picking the newest read the option book's 720 as this space's answer, so this",
             "  space is printed as a hole rather than left out of the status entirely.",
             "  owed: a probe run for keeps, and a stamp by which its closure can be identified."],
            False)


def main() -> int:
    print("COMPLETION STATUS - one arithmetic per option space")
    print(f"HEAD {head_revision()[:12] or 'unknown'}")
    print()

    overall = True
    for name, fn in ((host_label(), host_space),
                     ("DEVICE (classes over precision x shape)", device_space),
                     ("PROBE  (the host probe's own cell space)", probe_space)):
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
